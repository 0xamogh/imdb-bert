import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = "/kaggle/input/bert-base-uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE = "/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case = True,
    
    )
