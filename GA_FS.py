#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np, random, operator, pandas as pd
import os.path
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


# In[27]:


def initialPopulation(popSize, num_weights):
    
    pop_weight = (popSize, num_weights)
    new_population = np.random.randint(low=0, high=2, size=pop_weight)
    
    return new_population


# In[37]:


def acc_fitness(pop,X,y,n_clf):
    
    X_fitness = np.random.uniform(low=0, high=2, size=(len(X),sum(pop)))
    c = 0
    for i in range(0,len(pop)-1):
        if(pop[i] == 1 ):
            X_fitness[:,c] = X[:,i]
            c = c + 1
    
    if(n_clf == 1):
        clf = svm.SVC(kernel='linear', C=1000)
    elif(n_clf == 2):
        clf = MultinomialNB()
    elif(n_clf == 3):
        clf = RandomForestClassifier(n_jobs=-1, random_state=0)
    else:
        clf = linear_model.LogisticRegression(penalty='l2', C=1000,random_state=0)
    
   # scores = cross_val_score(estimator=clf,X=X_fitness,y=y,cv=5,n_jobs=1,scoring='recall')
    y_pred = cross_val_predict(clf, X, y, cv=5, n_jobs = -1 )
    print("Cross Validation metrics...")
    accuracy = accuracy_score(y, y_pred)
    n_features = sum(pop)
    
    fitness = accuracy + (1/n_features)
    
    return fitness


# In[38]:


def fitness_function(pop,X,y,n_clf):
    return acc_fitness(pop,X,y,n_clf)


# In[39]:


#Se utiliza un método de selección basado en el torneo binario que produce un padre por cada torneo
def select_parents(pop_fitness,num_parent):
    
    parents = np.random.uniform(low=0, high=2, size=num_parent)

    for i in range(0,num_parent):
        
        ind1 = random.randint(0, len(pop_fitness)-1)
        ind2 = random.randint(0, len(pop_fitness)-1)
        
        if(pop_fitness[ind1] >= pop_fitness[ind2]):
            parents[i] = ind1
        else:
            parents[i] = ind2
            
    return parents


# In[40]:


#Se utiliza un cruce UX que produce un hijo cada dos padres
def crossover(pop_total,pop_fitness,parents,prob_cross):
    
    count = 0
    offsprings = np.random.uniform(low=0, high=2, size=(int((len(parents)/2)), len(pop_total[0])))
    
    
    while(count != len(parents)/2):
        
        par1 = random.randint(0, len(parents)-1)
        par2 = random.randint(0, len(parents)-1)

        while(parents[par1] == -1 or parents[par2] == -1):
            
            par1 = random.randint(0, len(parents)-1)
            par2 = random.randint(0, len(parents)-1)
        
        parents[par1] = -1 
        parents[par2] = -1
        
        for i in range(0,len(pop_total[0])):
            
            if(pop_fitness[par1] > pop_fitness[par2]):
                best_par = par1
                worst_par = par2
            else:
                best_par = par2
                worst_par = par1
            
            r = random.uniform(0,1)
            
            if(r <= prob_cross):
                offsprings[count][i] = pop_total[best_par][i]
            else:
                offsprings[count][i] = pop_total[worst_par][i]
                
                
        count = count + 1
        
    return offsprings


# In[41]:


def mutation(offsprings,prob_mut):
    
    for i in range(0,len(offsprings)):
        for j in range(0, len(offsprings[0])):
            if(random.uniform(0,1) < prob_mut):
                if(offsprings[i][j] == 0):
                    offsprings[i][j] = 1
                else:
                    offsprings[i][j] = 0
                    
    return offsprings
    


# In[42]:


def replace(pop_total,offsprings,worst_fitness):
    
    for i in range(0,len(worst_fitness[0])):
        pop_total[worst_fitness[1][i]] = offsprings[i]
    
    return pop_total


# In[43]:


def cross_val(clf,X,y):
    scores = cross_val_score(estimator=clf,
                             X=X,
                             y=y,
                             cv=5,
                             n_jobs=1,
                             scoring='recall_macro')
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.average(scores), np.std(scores)))


# In[47]:


def main_GA(X,y,n_clf):
    #pop = 40 n_gen = 100
    pop_initial = 40
    num_generations = 100
    num_parents = 20
    prob_cross = 0.9

    pop_total = initialPopulation(pop_initial,len(X[0]))
    pop_fitness = np.random.uniform(low=0, high=2, size=pop_initial)

    prob_mut = 1.0/len(pop_total[0])
    
    progress_list = []
    num_generations_list = []
    f = 0
    count = 0
    for generation in range(num_generations):

        print("Number of generation: "+str(generation))
        num_generations_list.append(generation)
        best_ind = 0
        best_fitness = 0
        worst_fitness = np.random.randint(low=10000, high=10001, size=(2,int((num_parents/2))))
        
        for i in range(0,len(pop_total)):  
            
            if generation == 0:
                f = fitness_function(pop_total[i], X,y,n_clf)
                pop_fitness[i] = f
                count += 1
            else:
                if pop_fitness[i] == -1:
                    f = fitness_function(pop_total[i], X,y,n_clf)
                    pop_fitness[i] = f
                    count +=1
            if (f >= best_fitness):
                best_fitness = f
                best_ind = i
                n_features = sum(pop_total[i])

            for j in range(0,len(worst_fitness[0])):
                if(f <= worst_fitness[0][j]):
                    worst_fitness[1][j] = i
                    worst_fitness[0][j] = f
                    pop_fitness[i] = -1
                    break

        parents = select_parents(pop_fitness,num_parents)
        offsprings = crossover(pop_total,pop_fitness,parents,prob_cross) 
        offsprings = mutation(offsprings,prob_mut)
        pop_total = replace(pop_total,offsprings,worst_fitness)
        progress_list.append(best_fitness-(1/n_features))

    X_fitness = np.random.uniform(low=0, high=2, size=(len(X),sum(pop_total[best_ind])))
    c = 0

    for x in range(0,len(pop_total[0])-1):
        if(pop_total[best_ind][x] == 1 ):
            X_fitness[:,c] = X[:,x]
            c = c + 1
    plt.axis([0,100,0.7,1.10])
    plt.plot(progress_list)
    plt.ylabel('Accuracy')
    plt.xlabel('Generation')
    plt.savefig('./fig/generation_acc.png')
    
    print("El mejor resultado es: "+ str(best_fitness+(1/len(X_fitness[0]))))
    
    return X_fitness

