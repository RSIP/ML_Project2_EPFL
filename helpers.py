import pandas as pd

def load_tweets(full = False, bert = False):
    """
    This method is used to load the tweets dataset. You can either load the full dataset or 10% of it.

    Attributes:

    full: boolean
        whether or not you want the full dataset

    bert: boolean
        wether or not it is used for the BERT approach
    """

    DATA_FOLDER = './data/twitter-datasets/'
    
    if bert:
        DATA_FOLDER = '.' + DATA_FOLDER  
    
    positive_tweets_path = DATA_FOLDER + 'train_pos'
    negative_tweets_path = DATA_FOLDER + 'train_neg'
    
    if full:
        positive_tweets_path += '_full'
        negative_tweets_path += '_full'
        
    positive_tweets_path += '.txt'
    negative_tweets_path += '.txt'
    
    
    positive_tweets = pd.read_csv(positive_tweets_path, delimiter="\t", header=None, names = ['tweet'])
    positive_tweets['polarity'] = 1
    
    negative_tweets = pd.read_csv(negative_tweets_path, delimiter="\t", header=None, names = ['tweet'])
    negative_tweets['polarity'] = 0
    
    tweets = pd.concat([positive_tweets, negative_tweets]).reset_index(drop=True)
    
    return tweets

def load_test_tweets(csv_path):
    """This method takes the csv_path for the test tweets and returns pandas DataFrame
    with a column 'id' that and a column 'tweet'.
    
    Attribute:
    csv_path: str
        path of the test dataset
        
    """

    tweets = []
    ids = []

    with open(csv_path) as f:
        for line in f:
            id, tweet = line.split(',', 1)

            tweets.append(tweet)
            ids.append(id)

    return pd.DataFrame(list(zip(ids, tweets)), columns=['id', 'tweet'])

def construct_fasttext_input(tweets, csv_path):
    """
    This method takes as pandas DataFrame tweets as input and construct a csv file compatible with 
    the fasttext library.

    Attributes:
    
    tweets: pandas DataFrame
        tweets has a column 'tweet' that corresponds to a tweet and a column 'polarity' such that
        if it is 0, then the tweet is labelled as negative and if it is 1, it is labelled as positive.

    csv_path: str
        path where to save the constructed csv file
    """
    input_tweets = tweets.copy()
    input_tweets['label'] = input_tweets['polarity'].copy().apply(lambda polarity: '__label__POSITIVE' if polarity == 1 else '__label__NEGATIVE')
    formatted = input_tweets[['label', 'tweet']]
    
    formatted.to_csv(csv_path, sep = '\t', header = False) 