# Import all dependencies for the BERT model.

import pandas as pd
import preprocessing
import helpers
import torch
import numpy as np


from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss

from tools import *
from multiprocessing import Pool, cpu_count

from tqdm.notebook import tqdm
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification

from BERT import tools
from BERT import convert_examples_to_features

import logging


# DATA LOADING

DATA_FOLDER = './Datasets/twitter-datasets/test_data.txt'
DEV_FOLDER = './Datasets/twitter-datasets/dev.tsv'

# Load tweet dataset
init_tweets = helpers.load_test_tweets(DATA_FOLDER)
# Remove useless tags
test_tweets = preprocessing.remove_tags(init_tweets)

# Create BERT friendly dataset
dev_df_bert = pd.DataFrame({
    'id':test_tweets['id'],
    'label':[0]*test_tweets.shape[0],
    'alpha':['a']*test_tweets.shape[0],
    'text': test_tweets['tweet'].replace(r'\n', ' ', regex=True)
})
dev_df_bert.to_csv(DEV_FOLDER, sep='\t', index=False, header=False)

# MODEL PREDICTION

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Choose to run on CPU or GPU

# The input data dir. Should contain the .tsv files (or other data files) for the task.
DATA_DIR = "./Datasets/twitter-datasets/"

# Bert pre-trained model selected in the list: bert-base-uncased,
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'tweets10percent.tgz'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'tweet_sentiment_analysis'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = f'BERT/outputs/{TASK_NAME}/'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = './BERT/cache/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 128

TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(OUTPUT_DIR + 'vocab.txt', do_lower_case=False)

processor = tools.BinaryClassificationProcessor()
eval_examples = processor.get_dev_examples(DATA_DIR)
label_list = processor.get_labels() # [0, 1] for binary classification
num_labels = len(label_list)
eval_examples_len = len(eval_examples)

label_map = {label: i for i, label in enumerate(label_list)}
eval_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in eval_examples]

# Apply the preprocessing to features

process_count = cpu_count() - 1
if __name__ ==  '__main__':
    print(f'Preparing to convert {eval_examples_len} examples..')
    print(f'Spawning {process_count} processes..')
    with Pool(process_count) as p:
        eval_features = list(tqdm(p.imap(convert_examples_to_features.convert_example_to_feature, eval_examples_for_processing), total=eval_examples_len))

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

# Load pre-trained model (weights)
model = BertForSequenceClassification.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))

model.to(device)

# Run the model and make the predictions

model.eval()
eval_loss = 0
nb_eval_steps = 0
preds = []

for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)

    # create eval loss and other metric required by the task
    if OUTPUT_MODE == "classification":
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    elif OUTPUT_MODE == "regression":
        loss_fct = MSELoss()
        tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

    eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
    else:
        preds[0] = np.append(
            preds[0], logits.detach().cpu().numpy(), axis=0)

eval_loss = eval_loss / nb_eval_steps
preds = preds[0]
if OUTPUT_MODE == "classification":
    preds = np.argmax(preds, axis=1)
elif OUTPUT_MODE == "regression":
    preds = np.squeeze(preds)


# Create the submission file

def replace_pred(x):
    if(x > 0):
        return 1
    else:
        return -1


submission = pd.DataFrame({
    'Id': test_tweets['id'],
    'Prediction': preds,

})

submission['Prediction'] = submission['Prediction'].apply(replace_pred)

SUBMISSION_PATH = './Datasets/twitter-datasets/submission.csv'
submission.to_csv(SUBMISSION_PATH, index =False)

