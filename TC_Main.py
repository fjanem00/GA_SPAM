#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd
import time
import os.path
import pickle
import sys
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

#own modules
import Feature_Selector as FS
import GA_FS as GA
import Preprocessing as prepro


# # Saves

# In[77]:


def save_inform(info):
    #print(info+'\n')
    info = str(info)
    with open("inform.txt", "a") as filetxt:
        filetxt.write(info)


# In[78]:


def save_model(model, model_name):
    pickle.dump(model, open(model_name, 'wb'))


# # Evaluation methods

# In[79]:


def cross_val(clf,X,y):
    scores = cross_val_score(estimator=clf,X=X,y=y,cv=5,n_jobs=1,scoring='recall')
    save_inform('CV accuracy scores: %s \n' % scores)
    save_inform('CV accuracy: %.3f +/- %.3f\n'% (np.average(scores), np.std(scores)))


# In[80]:


def ROC_curve(clf,X,y,nClf,fs):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                    random_state=0)
    if nClf == '1' or nClf == '4':
        y_score = clf.fit(X_train,y_train).decision_function(X_test)
    else:
        y_score = clf.fit(X_train,y_train).predict(X_test)
        
    #y_score = cross_val_predict(pipeline, X_test, y_test, cv=5)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('./fig/ROC'+fs+nClf+'.png')


# In[81]:


def learning_curve_build(method,X,y,nClf,fs):
    train_sizes = np.linspace(0.1, 1.0, 5)
    X_shuf, Y_shuf = shuffle(X,y)
    train_sizes, train_scores, test_scores = learning_curve(estimator=method,X=X_shuf,y=Y_shuf,train_sizes=train_sizes,cv=5,n_jobs=-1)

    plt.figure()
    title = "Learning Curves"
    plt.title(title)

    plt.xlabel("Number of Samplex")
    plt.ylabel("Accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label='Training Accuracy')

    plt.fill_between(train_sizes,
                     train_scores_mean + train_scores_std,
                     train_scores_mean - train_scores_std,
                     alpha=0.15, color='red')

    plt.plot(train_sizes, test_scores_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='Validation Accuracy')

    plt.fill_between(train_sizes,
                     test_scores_mean + test_scores_std,
                     test_scores_mean - test_scores_std,
                     alpha=0.15, color='blue')
    plt.grid()
    plt.tight_layout()
    ylim = (0.1, 1.01)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.savefig('./fig/LC'+fs+nClf+'.png')
    


# In[82]:


def perf_measure(y_test, y_pred):
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(y_test)): 
        if y_test[i]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_test[i]!=y_pred[i]:
            FP += 1        
        if y_test[i]==y_pred[i]==0:
            TN += 1            
        if y_pred[i]==0 and y_test[i]!=y_pred[i]:
            FN += 1
    
    return(TP, FP, TN, FN)


# In[83]:


def predict_model(clf, X, y,cl,fs):

    #y_frame_binary = label_binarize(y, classes=['0','1'])
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y_frame_binary, test_size=.3, random_state=9)
    
    #pipeline = clf.fit(X,y)
    
    print("Training...")
    y_pred = cross_val_predict(clf, X, y, cv=5, n_jobs=-1)
    print("Cross Validation metrics...")
    accuracy = accuracy_score(y, y_pred)
    save_inform('CV accuracy scores: %s \n' % accuracy)

    precision, recall, f1 ,support = precision_recall_fscore_support(y_pred,y)
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_pred,y, average = 'micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_pred,y, average = 'macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_pred,y, average = 'weighted')
    
    save_inform('Micro: \n'+ "Precision: " + str(precision_micro) +" Recall: " + str(recall_micro) + " F1 Score: " + str(f1_micro) + '\n')
    save_inform('Macro: \n'+ "Precision: " + str(precision_macro) +" Recall: " + str(recall_macro) + " F1 Score: " + str(f1_macro) + '\n')
    save_inform('Macro: \n'+ "Precision: " + str(precision_weighted) +" Recall: " + str(recall_weighted) + " F1 Score: " + str(f1_weighted) + '\n')
    save_inform("Support: " + str(support) + '\n')
    
    confusion_matrix = perf_measure(y,y_pred)
   
    save_inform("Confusion Matrix: \n" + '('+str(confusion_matrix[0]) + '\t' +  str(confusion_matrix[3]) + ')\n' +'('+str(confusion_matrix[1]) + '\t' +  str(confusion_matrix[2]) + ')\n' )
    per_FP = (confusion_matrix[1]/sum(confusion_matrix))*100
    per_FN = (confusion_matrix[3]/sum(confusion_matrix))*100
    save_inform("Percent of FP: " + str(per_FP) + '\n' + "Percent of FN: " + str(per_FN) + '\n')
    
    learning_curve_build(clf,X,y,cl,fs)
    ROC_curve(clf,X,y,cl,fs)


# # Preprocessing

# In[98]:

print("SELECT THE NUMBER")
print("Do you want to use preprocessing?\n1)Yes\n2)No")
pre = input()   

# # Feature Extraction

# ### TF-IDF

# In[99]:


def tfidf_pipeline():
    return TfidfVectorizer(min_df=3, norm='l2',use_idf=True, smooth_idf=True, ngram_range=([1, 1]))


# ### BOW

# In[100]:


def bow_pipeline():
    return CountVectorizer(min_df=3, ngram_range=([1, 1]))


# # Text Classification

# In[101]:


seconds = time.time()
save_inform("######### Inform of ")     

#3. TO CHANGE THE DATASET
path = "./prueba_200.csv"

data_frame = pd.read_csv(path)
my_tags = ['Ham','Spam']


print("What does feature extraction want to use?\n1)TF-IDF\n2)BOW")
fe = str(input())
print("What does feature selection want to use?\n1)None\n2)Random\n3)Genetic Algorithm\n4)Chi-Square test")
fs = str(input())
print("What does classifier want to use?\n1)SVM\n2)NB\n3)RF\n4)LR")
cl = str(input())


if pre is '1':
    seconds_prepro = time.time()
    #1. TO CHANGE THE DATASET, same name as in point 3. 
    csv_prepro = "./SpamAssasin_Dataset_prepro.csv"
    #2. TO CHANGE THE DATASET, name of the csv file with contain the name of emails and categories.
    prepro.main_preprocessing("./SpamAssasin_Dataset.csv",csv_prepro)
    total_time_prepro = time.time() - seconds_prepro
    save_inform("Total time required to preprocess: " + str(total_time_prepro)+ " seconds\n") 



if fe is '1':
    vectorizer = tfidf_pipeline()
    save_inform("TF-IDF ")
elif fe is '2':
    vectorizer = bow_pipeline()
else:
    print("ERROR! User did not introduce correct option")
    sys.exit()

X = vectorizer.fit_transform(data_frame['Text'].astype(str)).toarray() 
y = data_frame['Category'].astype(str)
y = label_binarize(y, classes=['0','1'])
X_initial_len = len(X[0])


# 
# # Feature Selection

# In[102]:


#Determine the feature selection to use the best number of features. Chi_square, Genetic Algorithms and random choice
if fs is '1':
    save_inform("without FS and")
elif fs is '2':
    save_inform("Random as FS and")
    X = FS.Random_Algorithm(X)
elif fs is '3':
    save_inform("GA as FS and")
    X = GA.main_GA(X,y,cl)
elif fs is '4':
    save_inform("IG as FS and")
    X = FS.chi_square(X,y).fit_transform(X,y)
else:
    print("ERROR! User did not introduce correct option")
    sys.exit()


# # Classifiers

# In[103]:


if cl is '1':
    save_inform(" SVM #########\n") 
    clf = svm.SVC(kernel='linear', C=1000, class_weight ='balanced')
elif cl is '2':
    save_inform(" NB #########\n") 
    clf = MultinomialNB()
elif cl is '3':
    save_inform(" RF #########\n") 
    clf = RandomForestClassifier(n_jobs=-1, class_weight = 'balanced', n_estimators = 100)
elif cl is '4':
    save_inform(" LR #########\n") 
    clf = linear_model.LogisticRegression(penalty='l2', C=1000,class_weight ='balanced')

else:
    print("ERROR! User did not introduce correct option")
    sys.exit()

save_inform("Number of initial features: " + str(X_initial_len)+'\n')
save_inform("Number of selected features: " + str(len(X[0]))+'\n') 

y = np.ravel(y)
predict_model(clf, X, y,cl,fs)
total_time = time.time() - seconds 
save_inform("Total time required: " + str(total_time)+ " seconds\n") 

saving = '1'
print("Do you want to save the model?\n1)Yes2)No")
saving = input()

if saving is '1':
    clf = clf.fit(X,y)
    model_name = "model.pkl"
    save_model(clf,model_name)


# In[ ]:




