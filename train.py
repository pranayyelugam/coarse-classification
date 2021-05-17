import config
import dataset
import engine
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from model import BERTBaseUncased
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import logging
logging.basicConfig(level=logging.ERROR)

def preprocess(filename):
  df = pd.read_csv(filename)
  df = df[['sentence','coarse']]
  df = df.dropna()
  df['ENCODE_CAT'] = df['coarse'].astype('category').cat.codes
  return df

def monitor_metrics(outputs, targets):
    if targets is None:
        return {}
    print(outputs.size())
    outputs = torch.argmax(outputs, dim=0).cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    accuracy = metrics.accuracy_score(targets, outputs)
    return accuracy

def run():

    df_train=preprocess('./review-sentence_train_clean.csv')
    df_valid=preprocess('./review-sentence_dev_clean.csv')


    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        review=df_train.sentence.values, target=df_train.ENCODE_CAT.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.BERTDataset(
        review=df_valid.sentence.values, target=df_valid.ENCODE_CAT.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
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

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )


    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler, epoch)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device, epoch)
        accuracy =  metrics.accuracy_score(outputs, targets)
        print(f"Validation Accuracy  = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy
            print("Best val accuracy till now {}".format(best_accuracy))


if __name__ == "__main__":
    run()
