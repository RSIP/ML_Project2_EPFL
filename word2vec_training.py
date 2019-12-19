from helpers import *
from preprocessing import *
from gensim.models import Word2Vec 

def build_word2vec_model(sentences, min_count, size, nb_iter, model_name):
    """
    This method builds a word2vec model and saves it.

    Attributes:

    sentences: pandas DataFrame
        must contain a column 'tokens' that is use as the corpus input for the word2vec model

    min_count: int
        words present less than this number are not considered 

    size: int
        size of the word embedding
    
    nb_iter: int
        number of training iteration

    model_name: str
        path and name of the model
    """

    model = Word2Vec(sentences, 
                     min_count = min_count,   
                     size = size,     
                     workers = 2,     
                     window= 5,     
                     iter = nb_iter)
    
    model.save(model_name)

tweets = load_tweets(full = True)
clean_tweets = remove_tags(tweets)

tokens = tokenize_tweets(clean_tweets, stop_words = False, stemming = False).tokens
sentences = list(tokens)

tokens_stemming = tokenize_tweets(clean_tweets, stop_words = False, stemming = True).tokens
sentences_stemming = list(tokens_stemming)

build_word2vec_model(sentences, 5, 250, 30, 'word2vec.model')
build_word2vec_model(sentences_stemming, 5, 250, 30, 'word2vec_stemming.model')