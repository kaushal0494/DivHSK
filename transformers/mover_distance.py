import gensim.downloader as api

# Import and download stopwords from NLTK.
from nltk.corpus import stopwords
import numpy as np 

from .gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
stop_words = stopwords.words('english')

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

model = api.load('word2vec-google-news-300')

def wordmoverdistance(model,sentence1,sentence2):
    n = len(sentence1)
    distance = 0 
    for s1,s2 in zip(sentence1,sentence2):
        preprocess_s1 = preprocess(s1)
        preprocess_s2 = preprocess(s2)
        
        distance+= model.WordEmbeddingsKeyedVectors.wmdistance(preprocess_s1, preprocess_s2)
    print("mover distance ",distance/n)
    return np.round(distance/n,4)

s1 = ['i am checking it']
s2 = ['i used to check it']

wordmoverdistance(model,s1,s2)
