import pandas as pd

def load_tweets(full = False, bert = False):
    
    DATA_FOLDER = './data/twitter-datasets/'
    
    if bert:
        DATA_FOLDER = '.'+ DATA_FOLDER
    
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

