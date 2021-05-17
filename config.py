import transformers

DEVICE = "cpu"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
EPOCHS = 100
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "./classification.bin"
TRAINING_FILE = "./review-sentence_train_head.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
PATIENCE = 5
