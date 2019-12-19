import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def remove_tags(tweets):
    
    clean_tweets = tweets.copy()
    clean_tweets['tweet'] = clean_tweets['tweet'].copy().apply(lambda x: x.replace('<user>', '').replace('<url>', '').strip())
    
    return clean_tweets

def tokenize_tweets(tweets, stop_words = False, stemming = False):
    
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