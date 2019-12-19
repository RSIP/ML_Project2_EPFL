import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def remove_tags(tweets):
    """"
    This method takes a pandas DataFrame as input that has a column 'tweet'.
    It returns a copy of tha DataFrame such that all tweets in column 'tweet' will have
    their tag <user> and <url> removed.

    Attribute:

    tweets: pandas DataFrame
        DataFrame containing tweets in a column 'tweet'
    """
    
    clean_tweets = tweets.copy()
    clean_tweets['tweet'] = clean_tweets['tweet'].copy().apply(lambda x: x.replace('<user>', '').replace('<url>', '').strip())
    
    return clean_tweets

def tokenize_tweets(tweets, stop_words = False, stemming = False):
    """"
    This method takes a pandas DataFrame as input that has a column 'tweet'.
    It returns a copy of tha DataFrame such with an additional column 'tokens' that correspond
    to the list of words present in column 'tweet'.
    This method can tokenize tweets and there is the option to remove english stopwords as well as
    perform word stemming.

    Attributes:

    tweets: pandas DataFrame
        DataFrame containing tweets in a column 'tweet'

    stop_words: boolean
        whether or not tokens have stopwords

    stemming: boolean
        whether or not perform word stemming
    """
    
    tweets_output = tweets.copy()
    table = str.maketrans('', '', string.punctuation)
    
    tweets_output['tokens'] = tweets_output['tweet'].copy()\
                                                    .apply(lambda sentence: 
                                                            list(filter(
                                                                None,
                                                                [w.translate(table) for w in word_tokenize(sentence)]
                                                            )))
    
    if stop_words:
        stop_words = stopwords.words('english')
        tweets_output['tokens'] = tweets_output['tokens'].copy()\
                                                       .apply(lambda tokens: [token for token in tokens if token not in stop_words])
    
    if stemming:
        porter = PorterStemmer()
        tweets_output['tokens'] = tweets_output['tokens'].copy()\
                                                        .apply(lambda tokens: [porter.stem(token) for token in tokens])
    
    return tweets_output