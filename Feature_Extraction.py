#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[12]:


def tfidf_vectorizer(X):
    
    vectorizer = TfidfVectorizer(min_df=3, norm='l2',use_idf=True, smooth_idf=True, ngram_range=([1, 1]))
    
    X = vectorizer.fit_transform(X).toarray()  
    
    with open("Results.txt",'a') as writer:
        h = "Number of initial features " + str(len(X[0]))
        writer.write(h)
        writer.write('\n')
        
    return X


# In[16]:


def bow_vectorizer(X):
    
    vectorizer = CountVectorizer(min_df=4, ngram_range=([1, 1]))
    
    X = vectorizer.fit_transform(X).toarray()  
    
    with open("Results.txt",'a') as writer:
        h = "Number of initial features " + str(len(X[0]))
        writer.write(h)
        writer.write('\n')
        
    return X

