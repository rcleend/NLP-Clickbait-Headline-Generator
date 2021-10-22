import pandas as ps
import numpy as np
from gensim.models.word2vec import LineSentence, Word2Vec
import spacy
import nltk
import csv

data = ps.read_csv("clickbait_data.csv")
data = data[data['clickbait'] == 1].headline.tolist()
sentences = [nltk.word_tokenize(sent) for sent in data]
model = Word2Vec(sentences, sg=1, hs=0, max_vocab_size=None,
                 max_final_vocab=None, min_count=1, vector_size=300)
model.save("word2vec.model")
