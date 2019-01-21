# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:32:52 2019

@author: mickey
"""

import json
import urllib.request
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import pandas as pd
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = ''
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result+=lemmatize_stemming(token)+' '
    return result


with open('videos_upv_cleaned.json') as f:
	data = json.load(f)

stemmer = SnowballStemmer("spanish")

documents = []
preprocessedDocuments = []

for i in range(1,45000):
    if data[i]["transcription"] is not "":
        documents.append(data[i]["transcription"])
        preprocessedDocuments.append(preprocess(data[i]["transcription"]))
        
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessedDocuments)


'''
def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


X=X.toarray()
np.savetxt("tfid2.csv", X, delimiter=",")

scipy.sparse.save_npz("matrix.npz", X)

save_sparse_csr('D:\Machine Learning\VideoRecommender\matrix',X);

scipy.sparse.save_npz('matrix.npz',X,False)
'''

true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

dictionary = []

for i in range(true_k):
    dictionary.append(gensim.corpora.Dictionary())
    
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    words = []
    for ind in order_centroids[i, :5000]:
        words.append([terms[ind]])
    dictionary[i].add_documents(words)
    
count = 0
for k, v in dictionary[2].iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
lda_models = [None] * true_k
bow_corpus = [None] * true_k  
  
for i in range(true_k):
    bow_corpus[i] = [dictionary[i].doc2bow(doc) for doc in words]
    lda_models[i] = gensim.models.LdaMulticore(bow_corpus[i], num_topics=true_k, id2word=dictionary[i], passes=2, workers=2)
  

for idx, topic in lda_models[0].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

for i in range(1,45):
    if data[i]["transcription"] is not "":
        Y = vectorizer.transform([preprocess(data[i]["transcription"])])
        prediction = model.predict(Y)
        print(prediction)
        documents.append(data[i]["transcription"])
        preprocessedDocuments.append(preprocess(data[i]["transcription"]))
        bow_corpus_predict = [dictionary[i].doc2bow(doc) for doc in words]
        for index, score in sorted(lda_models[prediction[0]][bow_corpus_predict], key=lambda tup: -1*tup[1])[0]:
            print("Score: {}\t Topic: {}".format(score, lda_models[prediction[0]].print_topic(index, 5)))




print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)