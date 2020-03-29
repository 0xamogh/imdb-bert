import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = "D:/Coding/EmEL/NLP/Models/bert-based-uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "D:/Coding/EmEL/NLP/Datasets/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case = True,
    
    )
