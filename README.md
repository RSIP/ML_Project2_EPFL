# Machine Learning EPFL 2019 - Project 2

## EPFL ML Text Classification 2019

### Abstract:

Text  classification  is  a  challenging  problem  thathas  many  applications  such  as  spam  filtering,  language  iden-tification...   Researchers   started   to   address   the   problem   inthe  1980s  but  with  the  recent  explosion  of  data  availableand  advances  in  Machine  Learning,  the  field  gained  a  lot  oftraction. This report details our work on the ”EPFL ML TextClassification  Challenge”  where  the  goal  is  to  use  machinelearning  to  classify  tweets.  We  have  to  predict  if  the  tweetcontained  a  ”:)”  or  a  ”:(”  smiley.  Thus  we  need  to  classify  atweet as positive or negative. 



### Introduction:

Hosted on the competitive platform AIcrowd, this projectaims to introduce us to text classification tasks. Competitors are evaluated using the accuracy and the F1- score of their predictions.The training dataset contains 2,458,297 tweets. 1,218,655of them are labelled as positive (they contained a ”:)” smiley)and 1,239,642 are labelled as negative (they contained a ”:(”smiley). In the report, we will first explore the dataset. We will thentry  several  approaches:  computing  tweet  embedding  fromwords  embedding  and  training  a  classifier  model  using  theobtained vectors, using a neural network called fasttext, andusing transformer model. Finally we will present our resultsand discuss our work.

### Setup:

Here are the different packages needed to reproduce our experiments:

- `pandas`
- `numpy`
- `pickle`
- `nltk`: for pre-processing text.
- `tqdm`: for displaying a progress bar during training.
- `gensim`: for building Word2Vec model.
- `sklearn`: for training ML models.
- `fasttext`: for text classification model.
- `torch`: for BERT model.
- `multiprocessing`: for displaying number of processes used.
- `pytorch_pretrained_bert`: for BERT model.

You can download our trained BERT model, and the vocab here:

- BERT model: https://drive.google.com/file/d/1xhYsQ_wfs9YLE7neKH9bHXjm7AOjIRfj/view?usp=sharing and put it in `./BERT/cache/`.

- vocab: https://drive.google.com/file/d/1G9lCWutsnQMmxszruJk6F_-hDS9FpShc/view?usp=sharing and put it in `./BERT/outputs/tweet_sentiment_analysis/`.

If you want to obtain it by yourself:
1. Run `./BERT/data_prep_tweets.ipynb`.
2. Run `./BERT/BERT_train_tweets.ipynb`.
3. Archive the obtained files `pytorch_model.bin` and `config.json` from `./BERT/outputs/tweet_sentiment_analysis/` into a `tweets10percent.tgz` file and place it into `./BERT/cache/`.
4. Run `run.py` to obtain the submission file in `./Datasets/twitter-datasets/`.

### Directory structure:

The following directory contrains different text documents, code and data files. The structure is detailed below:

#### Documents:

- `project2_description.pdf`: Describe the task to perform and the tools availables.
- `project2_report.pdf`: Describe our approach, work, and conclusion while solving the problem.

#### Code:

##### Python files:

- `helpers.py`: contains helper methods used for loading data or creating file.
- `preprocessing.py`: contains methods used for preprocessing text data.
- `word2vec_training.py`: running this file will create the two Word2Vec models that we used for our experiences.
- `run.py`: running this file will create our best submission as a csv file.
- `./BERT/convert_examples_to_features.py`: convert dataset to BERT features.
- `./BERT/converter.py`: load dataset and convert it to BERT features.
- `./BERT/tools.py`: define BERT features Objects.


##### Jupyter Notebooks:
- `models_testing-trained_Word2Vec.ipynb`: experiments conducted using our trained Word2Vec models and classification models. Run `word2vec_training.py` to obtain the models.
- `models_testing-pretrained_GloVe.ipynb`: experiments conducted using a pretrained GloVe model and classification models.
- `fasttext_model_tuning.ipynb`: experiments conducted using the `fasttext` model for text classification. 
- `./BERT/data_prep_tweets.ipynb`: convert tweet text dataset into BERT-friendly csv.
- `./BERT/BERT_train_tweets.ipynb`: load obtained csv, pre-processed them into BERT features and train the model.
- `./BERT/BERT_eval_tweets.ipynb`: load test set, pre-processed it into BERT features and evaluate the model.
