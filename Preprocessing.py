#!/usr/bin/env python
# coding: utf-8

# In[439]:


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


# In[440]:


def remove_stopwords(text):
    
    stop = list(stopwords.words('english'))

    text = text.lower()

    for word in stop: 
        text = text.replace(word+' ',' ').replace('  ',' ')
        
    return text


# In[441]:


def stem_lem(text):
    
    porter = PorterStemmer()
    text = porter.stem(text)
    
    return text


# In[442]:


def general_removing(text):
    text = text.lower()  
    text = re.sub('&','and',text)
    text = re.sub(r'[<>]', ' ',text) 
    text = re.sub(r'[(/\({][}+*^]','',text)
    text = re.sub(r'[)(]','',text)
    text = re.sub(r'''[.]|,|:|/|=|(\s-\s)|(\s(-)+)|((-)+\s)|'|;|[?]|"|#''' ,' ',text)
    text = re.sub('\s\W([0-9])+\s','', text)
    text = re.sub('([a-zA-Z])@([a-zA-Z])',r'\1 \2',text)
    text = re.sub(r'(\w{3,})([0-9])+', r'\1',text)
    text = re.sub(r'([0-9])+(\w{3,})', r'\2',text)
    text = re.sub(r'[0-9]',' ',text)
    text = re.sub(r"[-+]+?[.\d]*[\d]+[:,.\d]", ' ',text)
    text = re.sub(r'-', ' ',text)
    text = re.sub('\s\w\s', ' ',text)
    text = re.sub('\s+',' ',text)
    
    return text


# In[443]:


def spam_removing(text):
    
    

    
    text = re.sub('([a-zA-Z])4([a-zA-Z]|\s)',r'\1a\2',text)
    text = re.sub('([a-zA-Z])2([a-zA-Z]|\s)',r'\1s\2',text)
    text = re.sub('([a-zA-Z])4([a-zA-Z]|\s)',r'\1a\2',text)
    text = re.sub('([a-zA-Z])3([a-zA-Z]|\s)',r'\1e\2',text)
    text = re.sub('([a-zA-Z])0([a-zA-Z]|\s)',r'\1o\2',text)
    text = re.sub('([a-zA-Z])9([a-zA-Z]|\s)',r'\1q\2',text)
    text = re.sub('([a-zA-Z])2([a-zA-Z]|\s)',r'\1s\2',text)
    text = re.sub('([a-zA-Z])5([a-zA-Z]|\s)',r'\1s\2',text)
    text = re.sub('([a-zA-Z])6([a-zA-Z]|\s)',r'\1g\2',text)
    text = re.sub('([a-zA-Z])8([a-zA-Z]|\s)',r'\1g\2',text)
    text = re.sub('([a-zA-Z])1([a-zA-Z]|\s)',r'\1l\2',text)
    text = re.sub('([a-zA-Z])[+]([a-zA-Z]|\s)',r'\1t\2',text)
    text = re.sub('([a-zA-Z])7([a-zA-Z]|\s)',r'\1t\2',text)
    text = re.sub('([a-zA-Z])[|]([a-zA-Z]|\s)',r'\1l\2',text)
    text = re.sub('(|[a-zA-Z])!([a-zA-Z]|\s)',r'\1i\2',text)
    text = re.sub('(|[a-zA-Z])¡([a-zA-Z]|\s)',r'\1i\2',text)
    text = re.sub('(\s|[a-zA-Z])4([a-zA-Z])',r'\1a\2',text)
    text = re.sub('(\s|[a-zA-Z])2([a-zA-Z])',r'\1s\2',text)
    text = re.sub('(\s|[a-zA-Z])4([a-zA-Z])',r'\1a\2',text)
    text = re.sub('(\s|[a-zA-Z])3([a-zA-Z])',r'\1e\2',text)
    text = re.sub('(\s|[a-zA-Z])0([a-zA-Z])',r'\1o\2',text)
    text = re.sub('(\s|[a-zA-Z])6([a-zA-Z])',r'\1g\2',text)
    text = re.sub('(\s|[a-zA-Z])8([a-zA-Z])',r'\1g\2',text)
    text = re.sub('(\s|[a-zA-Z])9([a-zA-Z])',r'\1q\2',text)
    text = re.sub('(\s|[a-zA-Z])2([a-zA-Z])',r'\1s\2',text)
    text = re.sub('(\s|[a-zA-Z])5([a-zA-Z])',r'\1s\2',text)
    text = re.sub('(\s|[a-zA-Z])1([a-zA-Z])',r'\1l\2',text)
    text = re.sub('(\s|[a-zA-Z])[+]([a-zA-Z])',r'\1t\2',text)
    text = re.sub('(\s|[a-zA-Z])7([a-zA-Z])',r'\1t\2',text)
    text = re.sub('(\s|[a-zA-Z])[|]([a-zA-Z])',r'\1l\2',text)
    text = re.sub('(\s|[a-zA-Z])!([a-zA-Z])',r'\1i\2',text)
    text = re.sub('(\s|[a-zA-Z])¡([a-zA-Z])',r'\1i\2',text)
    
    return text


# In[444]:


def save_csv(file,cat,text, csv_file):
    
    with open(csv_file,'a') as writer:
            h = str(file)+','+str(cat)+','+str(text)
            writer.write(h)
            writer.write('\n') 


# In[445]:


def main_preprocessing(path,csv_file):
    
    data_frame = pd.read_csv(path)
    text = ' '
    with open(csv_file,'w') as writer:
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
        text = remove_stopwords(text)              
        save_csv(file,cat,text,csv_file)


# In[ ]:




