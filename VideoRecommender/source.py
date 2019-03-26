# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 15:32:52 2019
@author: mickey
"""

# python -m nltk.downloader -> download all

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
import re
from elmoformanylangs import Embedder
from overrides import overrides
import tensorflow_hub as hub
import tensorflow as tf


with open('videos_upv_cleaned.json') as f:
	data = json.load(f)

#stemmer = SnowballStemmer("spanish")

documents = []
preprocessedDocuments = []
preprocessedDocumentsList = []

embed = hub.Module("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/1")

for i in range(1,45000):
    if data[i]["transcription"] is not "":
        documents.append(data[i]["transcription"])
        preprocessedDocuments.append(data[i]["transcription"])
        '''
        wordList = re.sub("[^\w]", " ",  data[i]["transcription"]).split()
        words = []
        for word in wordList:
            words.append(word)
        preprocessedDocumentsList.append(words)
        '''
        

X = []

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    X = session.run(embed(preprocessedDocuments))        

data_folder = "C:\\Users\\Visiting\\Desktop\\ELMoForManyLangs-master\\145"
e = Embedder(data_folder)
sents = preprocessedDocumentsList
# the list of lists which store the sentences 
# after segment if necessary.

e.sents2elmo(sents)
# will return a list of numpy arrays 
# each with the shape=(seq_len, embedding_size)
        
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
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=3,verbose=True)
model.fit(X)

clusterWords = [None] * true_k
dictionary = []
preprocessedListForDictionary = []

for i in range(true_k):
    preprocessedListForDictionary.append([])

nr=0
for i in range(1,45000):
    if data[i]["transcription"] is not "":
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            Y = session.run(embed(data[i]["transcription"]))
        prediction = model.predict(Y)
        preprocessedListForDictionary[prediction[0]].append(preprocessedDocumentsList[nr])
        nr+=1
        '''
        wordList = re.sub("[^\w]", " ", preprocess(data[i]["transcription"])).split()
        words = []
        for word in wordList:
            words.append([word])
        dictionary[prediction[0]].add_documents(words)
        words = []
        for word in wordList:
            words.append(word)
        clusterWords[prediction[0]].append(words)
        '''

for i in range(true_k):
    dictionary.append(gensim.corpora.Dictionary(preprocessedListForDictionary[i]))
    clusterWords[i]=[]

'''    
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
clusterWords = [None] * true_k
for i in range(true_k):
    words = []
    for ind in order_centroids[i, :5000]:
        words.append([terms[ind]])
    dictionary[i].add_documents(words)
    clusterWords[i]=words
'''

count = 0
for k, v in dictionary[3].iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
    
lda_models = [None] * true_k
bow_corpus = [None] * true_k  
corpus_tfidf = [None] * true_k
tfidf_models = [None] * true_k
  
for i in range(true_k):
    bow_corpus[i] = [dictionary[i].doc2bow(doc) for doc in preprocessedListForDictionary[i]]
    tfidf_models[i] = gensim.models.TfidfModel(bow_corpus[i])
    corpus_tfidf[i]= tfidf_models[i][bow_corpus[i]]
    lda_models[i] = gensim.models.LdaMulticore(corpus_tfidf[i], num_topics=10, id2word=dictionary[i], passes=2, workers=4)
  

for idx, topic in lda_models[0].print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

for i in range(1,10):
    if data[i]["transcription"] is not "":
        Y = vectorizer.transform([preprocess(data[i]["transcription"])])
        prediction = model.predict(Y)
        print('Cluster :{}'.format(prediction))
        #documents.append(data[i]["transcription"])
        #preprocessedDocuments.append(preprocess(data[i]["transcription"]))
        wordList = re.sub("[^\w]", " ",  data[i]["transcription"]).split()
        words = []
        for word in wordList:
            words.append(word)
        bow_corpus_predict = dictionary[prediction[0]].doc2bow(words)
        for index, score in sorted(lda_models[prediction[0]][bow_corpus_predict], key=lambda tup: -1*tup[1]):
            print("Score: {}\t Topic: {}".format(score, lda_models[prediction[0]].print_topic(index, 5)))
        print()

'''
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break
'''

querryDataFrame = pd.DataFrame(columns=['VideoID','Assigned Cluster','CorrLDA Scores'])
#a1 = pd.DataFrame([[0,1,[1,2,3]]],columns=['VideoID','Assigned Cluster','CorrLDA Scores'])
#querryDataFrame = querryDataFrame.append(a1,ignore_index = True)

for i in range(1,45000):
    if data[i]["transcription"] is not "":
        Y = vectorizer.transform([preprocess(data[i]["transcription"])])
        prediction = model.predict(Y)
        wordList = re.sub("[^\w]", " ",  data[i]["transcription"]).split()
        words = []
        for word in wordList:
            words.append(word)
        bow_corpus_predict = dictionary[prediction[0]].doc2bow(words)
        newLine = pd.DataFrame([[i,prediction[0],lda_models[prediction[0]][bow_corpus_predict]]],columns=['VideoID','Assigned Cluster','CorrLDA Scores'])
        querryDataFrame = querryDataFrame.append(newLine,ignore_index = True)

querry = 'Ciencias de la Computación'
Y = vectorizer.transform([querry])
querryCluster = model.predict(Y)[0]
wordList = re.sub("[^\w]", " ", querry).split()
words = []
for word in wordList:
    words.append(word)
bow_corpus_querry = dictionary[querryCluster].doc2bow(words)
bow_corpus_querry_tfidf= tfidf_models[querryCluster][bow_corpus_querry]
querryScoresList = lda_models[querryCluster][bow_corpus_querry_tfidf]
querryScores = {}
for score in querryScoresList:
    querryScores[score[0]]=score[1]
    
    
scoreDifferences = {}
for index,entry in querryDataFrame.iterrows():
    if entry['Assigned Cluster'] == querryCluster:
        totalScore=0
        for score in querryDataFrame.at[index,'CorrLDA Scores']:
            if score[0] in querryScores:
                totalScore += abs(score[1]-querryScores[score[0]])
            else:
                totalScore +=0.1
        scoreDifferences[entry['VideoID']]=totalScore

import operator    
sorted_x = sorted(scoreDifferences.items(), key=operator.itemgetter(1))

'''
print("\n")
print("Prediction")
Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)
'''