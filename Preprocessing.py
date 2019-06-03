#!/usr/bin/env python
# coding: utf-8

# In[149]:


import numpy as np  
import pandas as pd
import re  
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files  
import pickle  
import os.path
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer


# In[181]:


def remove_stopwords(text,stop_words_list):
    

    stop = list(stopwords.words('english'))

    text = text.lower()

    spamwords = open('stop_words_spam.txt', "r")

    for line in spamwords.readlines():
        stop.append(line.replace('\n',''))

    spamwords.close()

    for word in stop: 
        text = text.replace(word+' ',' ').replace('  ',' ')
        
    return text


# In[182]:


def stem_lem(text):
    
    porter = PorterStemmer()
    text = porter.stem(text)
    
    return text


# In[194]:


def general_removing(text):
    text = text.lower()  
    text = re.sub('&','and',text)
    #removing any single character
    text = re.sub(' \W ', ' ',text)
    text = re.sub('\W',' ',text)
    text = re.sub(' \w ',' ',text)
    cleanhtml = re.compile('<.*?>')
    text = re.sub(cleanhtml,'', text)
     
    return text


# In[195]:


def spam_removing(text):
    
    text = text.replace('@','a').replace('4','A').replace('3','E').replace('¡','i').replace('!','i').replace('0','o')
    text = text.replace('8','g').replace('6','G').replace('9','q')
    text = text.replace('1','l').replace('|','l').replace('5','s').replace('2','s').replace('+','t').replace('7','T')
    
    
    return text


# In[196]:


def save_csv(file,cat,text, csv_file):
    
    with open(csv_file,'a') as writer:
            h = str(file)+','+str(cat)+','+str(text)
            writer.write(h)
            writer.write('\n') 


# In[199]:


def main_preprocessing(path, stop_words_list,csv_file):
    
    data_frame = pd.read_csv(path)
    text = ' '
    with open(csv_file,'a') as writer:
        h = "Name,Category,Text"
        writer.write(h)
        writer.write('\n')  

    address = data_frame['Name_File']
    for file in address:
        cat = data_frame[data_frame['Name_File']==file]['Category'].iloc[0]
        if cat == 1:
            path_spam = "./spam/"+file
            if os.path.isfile(path_spam):     
                with open(path_spam, 'r',encoding='utf-8',errors='ignore') as txt_spam:
                    text = str(txt_spam.read()).replace('\n','').replace(',','')  
        else:
            path_ham = "./ham/"+file
            if os.path.isfile(path_ham): 
                with open(path_ham, 'r',errors='ignore') as txt:
                    text = str(txt.read()).replace('\n', '').replace(',','')
        
        text = spam_removing(text)
        
        text = general_removing(text)
        text = stem_lem(text)
        text = remove_stopwords(text,stop_words_list)              
        save_csv(file,cat,text,csv_file)

