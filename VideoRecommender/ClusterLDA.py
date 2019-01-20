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

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

'''
X=X.toarray()
np.savetxt("tfid2.csv", X, delimiter=",")

scipy.sparse.save_npz("matrix.npz", X)

save_sparse_csr('D:\Machine Learning\VideoRecommender\matrix',X);

scipy.sparse.save_npz('matrix.npz',X,False)
'''

true_k = 6
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

words = []

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :5]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)