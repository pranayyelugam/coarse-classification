import config
import json
import contextlib
import torch
import torch.nn as nn
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.ERROR)

with open("labels.json", 'r') as f:
  LABEL_MAP = json.load(f)



class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.review)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review,
            None,
            add_special_tokens=True,
            max_length=128,
            truncate=True,
            padding='max_length',
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }

def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

class ModelMode(object):
  TRAIN = "train"
  EVAL  = "eval"
  BOTH = [TRAIN, EVAL]

def train_eval_fn(train_or_eval, data_loader, model, device, epoch,
             optimizer=None, scheduler=None):

    assert train_or_eval in ModelMode.BOTH
    is_train = train_or_eval == ModelMode.TRAIN

    if is_train:
      model.train()
      context = contextlib.nullcontext()
      assert optimizer is not None
      assert scheduler is not None
    else:
      model.eval()
      context = torch.no_grad()
    
    tr_loss = 0
    nb_tr_steps = 0

    target_accumulator = []
    output_accumulator = []

    with context:
      for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
          ids = d["ids"].to(device, dtype=torch.long)
          token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
          mask = d["mask"].to(device, dtype=torch.long)
          targets = d["targets"].to(device, dtype=torch.long)

          if is_train:
            optimizer.zero_grad()
          
          outputs = model(
              ids=ids, mask=mask, token_type_ids=token_type_ids)

          loss = loss_fn(outputs, targets)
          tr_loss += loss.item()
          nb_tr_steps += 1

          if is_train:
            loss.backward()
            optimizer.step()
            scheduler.step()

          target_accumulator.extend(
              targets.cpu().detach().numpy().tolist())
          output_accumulator.extend(
              torch.argmax(
                outputs, dim=1).cpu().detach().numpy().tolist())

      epoch_loss = tr_loss/nb_tr_steps
      print(f"Training Loss for Epoch: {epoch} {epoch_loss}")
      return target_accumulator, output_accumulator

