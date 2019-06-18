#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn import linear_model


# In[3]:


def chi_square(X,y):
    
    chi2_features = chi2(X,y) 
    f = chi2_features[1] > 0.05
    count = 0
    for item in f:
        if item:
            count = count + 1  
            
    chi2_selector = SelectKBest(chi2, k=count)
    
    return chi2_selector


# In[1]:


def info_gain(X,y):

    ig_features = mutual_info_classif(X, y) 
    f = ig_features > 0
    count = 0
    for item in f:
        if item:
            count = count + 1
    ig_selector = SelectKBest(mutual_info_classif, k=count)
    
    return ig_selector


# In[4]:


def Random_Algorithm(X):
    
    X_random = np.random.randint(low=0, high=2, size=len(X[0]))
    
    X_new = np.random.uniform(low=0, high=2, size=(len(X),sum(X_random)))
    c = 0
    
    for i in range(0,len(X)-1):
        if(X_random[i] == 1 ):
            X_new[:,c] = X[:,i]
            c = c + 1
    return X_new 

