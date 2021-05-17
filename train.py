import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import sys

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import logging
logging.basicConfig(level=logging.ERROR)

def preprocess(filename, label):
  df = pd.read_csv(filename)
  df = df[['sentence',label]]
  #df = df.dropna()
  df['ENCODE_CAT'] = df[label].astype('category').cat.codes
  return df

def monitor_metrics(outputs, targets):
    if targets is None:
        return {}
    outputs = torch.argmax(outputs, dim=0).cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    accuracy = metrics.accuracy_score(targets, outputs)
    return accuracy


def process_dataset(df, batch_size, num_workers):
  df = df.reset_index(drop=True)
  this_dataset = dataset.BERTDataset(
        review=df.sentence.values, target=df.ENCODE_CAT.values
    )
  data_loader = torch.utils.data.DataLoader(
      this_dataset, batch_size=batch_size, num_workers=num_workers)
  return data_loader

def get_num_train_steps(train_filename, label):
  df = preprocess(train_filename, label)
  return int(len(df) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

def run():

    train_filename, label = sys.argv[1:3]

    model_path = "models/" + label + "_best.pt"

    assert 'train' in train_filename
    filenames = {'train': train_filename,
        'dev': train_filename.replace('train', 'dev'),
        'test':train_filename.replace('train', 'test')}

    dataframes = {}
    num_classes = 0
    for subset, filename in filenames.items():
      dataframes[subset] = preprocess(filename, label)
      num_classes = max(num_classes, max(dataframes[subset].ENCODE_CAT) + 1)
      print(dataframes[subset])
      print(len(dataframes[subset]))

    dataloaders = {}
    for subset, filename in filenames.items():
      if subset == 'train':
        batch_size = config.TRAIN_BATCH_SIZE
        num_workers = 4
      else:
        batch_size = config.VALID_BATCH_SIZE
        num_worker = 1
      dataloaders[subset] = process_dataset(
          dataframes[subset], batch_size, num_workers)

    device = torch.device(config.DEVICE)
    model = BERTBaseUncased(num_classes)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=get_num_train_steps(filenames["train"], label)
    )


    best_val_accuracy = float('-inf')
    best_val_epoch = None

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(
            dataloaders["train"], model, optimizer, device, scheduler, epoch)
        outputs, targets = engine.eval_fn(
            dataloaders['dev'], model, device, epoch)
        accuracy =  metrics.accuracy_score(outputs, targets)
        print(f"Validation Accuracy  = {accuracy}")
        if accuracy > best_val_accuracy:
            torch.save(model.state_dict(), model_path)
            best_val_accuracy = accuracy
            best_val_epoch = epoch
            print("Best val accuracy till now {}".format(best_val_accuracy))

        if best_val_epoch < (epoch - config.PATIENCE):
          break

    model.load_state_dict(torch.load(model_path))
    for subset in ['train', 'dev', 'test']:
      outputs, targets = engine.eval_fn(
            dataloaders[subset], model, device, epoch)

      result_df_dicts = []
      for o, t in zip(outputs, targets):
        result_df_dicts.append({"output":o, "target":t})

      result_df = pd.DataFrame.from_dict(result_df_dicts)

      final_df = pd.concat([dataframes[subset], result_df], axis=1)
      for i in final_df.itertuples():
        assert i.ENCODE_CAT == i.target

      result_file = "results/" + subset + "_" + label + ".csv"
      final_df.to_csv(result_file)




if __name__ == "__main__":
    run()
