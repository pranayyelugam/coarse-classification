import config
import transformers
import torch.nn as nn

import logging
logging.basicConfig(level=logging.ERROR)

class BERTBaseUncased(nn.Module):
    def __init__(self, num_classes):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout =nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
