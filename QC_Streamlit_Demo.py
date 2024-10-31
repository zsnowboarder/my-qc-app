#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install numpy==1.26.4


# In[2]:

import nltk
# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import numpy as np
import pandas as pd
import string
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import _stop_words

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
from nltk.corpus import wordnet

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.semi_supervised import SelfTrainingClassifier


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn import svm

#import tensorflow as tf
#from tensorflow.keras import layers, models, preprocessing # for text classification
#from tensorflow.keras.preprocessing.text import Tokenizer
#from gensim.models import Word2Vec, KeyedVectors


# In[3]:


# digits and stopwords removal function. stemming and lemmatization can be done here as well

stopwords = _stop_words.ENGLISH_STOP_WORDS
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
spell_check = SpellChecker(distance=1)

def clean(doc): # doc is a string of text
    #doc = " ".join([lemmatizer.lemmatize(token) for token in doc.split()])
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = " ".join([token for token in doc.split() if token not in stopwords])
    return doc

def spell(doc):
    doc = " ".join([spell_check.correction(token) for token in doc.split() if spell_check.correction(token) is not None])
    return doc
    
# stemming
def stem(doc): # doc is a string of text
    doc = " ".join([stemmer.stem(token) for token in doc.split()])
    return doc

def lemmatize(doc): # doc is a string of text
    doc = " ".join([lemmatizer.lemmatize(token) for token in doc.split()])
    return doc

# combined all preprocessors 
def preprocessors(doc):
    doc = spell(doc) # very slow
    doc = clean(doc)
    doc = lemmatize(doc)
    doc = stem(doc)
    return doc


# **Load data**

# In[5]:


# load training set
mydata = pd.read_csv("incidents.csv") #loading from the same directory as the notebook file.

#mydata = pd.read_csv("incidents.csv") #loading from the same directory as the notebook file.
mydata["X"] = mydata["X"].str.lower()
mydata = mydata.dropna()


# **Random sample if neccessary**

# In[7]:


sample_size = 1000
rows = mydata.shape[0] # get how many rows does mydata have

# if rows is smaller than the sample size, then the sample size will be the number of rows.
if rows < sample_size:
    sample_size = rows
    mydata = mydata.sample(n=sample_size, random_state=42)
else:
    mydata = mydata.sample(n=sample_size, random_state=42)
    
print(mydata.shape)


# **Show what it the data looks like after cleaning**

# In[9]:


X_col = mydata.X
mydata["clean"] = [clean(d) for d in X_col]
clean_col = mydata.clean
mydata["lemmatize"] = [lemmatize(d) for d in clean_col] # here lammatization is applied first. but sometimes it depends. stemming can be applied first in specific cases.
lemmatize_col = mydata.lemmatize
mydata["stem"] = [stem(d) for d in lemmatize_col] # we need to stem the clean data not the orginal one


# In[10]:


# alternatively we can use a function to do all the preprocessing at once

mydata["all_preprocessors"] = [preprocessors(d) for d in X_col]


# In[11]:


mydata = mydata.drop_duplicates() # removal all duplicates
mydata


# **Convert the labels to numerica data if neccessary**

# In[13]:


#mydata['y'] = pd.factorize(mydata['y'])[0] # convert y col to numerical data if neccessary

print(mydata.y.value_counts())


# **Data exploration, pre-processing and cleansing**

# In[15]:


# take a look
print(mydata.head())
display(mydata.shape)
mydata['y'].value_counts() / mydata.shape[0]


# **train test split**

# In[17]:


# Assign col X to X and col y to y
X = mydata.X
y = mydata.y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# **Feature generation**

# In[19]:


# apply tokenization and clean the text using tfidf

#vect_tfidf = TfidfVectorizer(preprocessor=clean)
vect_tfidf = TfidfVectorizer(preprocessor=preprocessors)
X_train_dtm = vect_tfidf.fit_transform(X_train)
X_test_dtm = vect_tfidf.transform(X_test)


# **Model building**

# In[21]:


# Naive Bayes Model

nb = MultinomialNB()
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')
nb_pred = nb.predict(X_test_dtm)


# In[22]:


# Logistic Regression

logreg = LogisticRegression(class_weight="balanced")
get_ipython().run_line_magic('time', 'logreg.fit(X_train_dtm, y_train)')
logreg_pred = logreg.predict(X_test_dtm)
logreg_pred_prob = logreg.predict_proba(X_test_dtm)[:,1]


# In[23]:


# knn

#knn = KNeighborsClassifier(n_neighbors=30)
knn = KNeighborsClassifier()
knn.fit(X_train_dtm, y_train)
knn_pred = knn.predict(X_test_dtm)


# In[24]:


# Linear SVC

svm = LinearSVC(dual=False)
svm.fit(X_train_dtm, y_train)
svm_pred = svm.predict(X_test_dtm)


# In[25]:


# Decision tree

dt = DecisionTreeClassifier()
dt.fit(X_train_dtm, y_train)
dt_pred = dt.predict(X_test_dtm)


# In[26]:


# Use Grid Search to find the best estimator for Random Forest

rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [None, 10, 20, 30],
              'min_samples_split': [2, 5, 10]
             }
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_dtm, y_train)
best_rf = grid_search.best_estimator_
rf_pred = best_rf.predict(X_test_dtm)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)


# In[27]:


# Random forest ensemble learning

rf = RandomForestClassifier(**best_params, random_state=42) # using the best params found in Grid SearchCV
rf.fit(X_train_dtm, y_train)
rf_pred = rf.predict(X_test_dtm)


# In[28]:


# AdaBoost

base_estimator = DecisionTreeClassifier(max_depth=1)
ada = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
ada.fit(X_train_dtm, y_train)
ada_pred = ada.predict(X_test_dtm)


# **Semi-supervised learning**

# In[30]:


# create the self learning classifier using the logreg model

base_classifier = LogisticRegression()
self_training_model = SelfTrainingClassifier(base_classifier)

# train the model
self_training_model.fit(X_train_dtm, y_train)
self_training_model_pred = self_training_model.predict(X_test_dtm)


# **Neural Net**

# **Evaluate models**

# In[33]:


print("NB\n", classification_report(y_test, nb_pred))
print("Logistic Regression\n", classification_report(y_test, logreg_pred))
print("KNN\n", classification_report(y_test, knn_pred))
print("SVM\n", classification_report(y_test, svm_pred))
print("Decision Tree\n", classification_report(y_test, dt_pred))
print("Random Forest\n", classification_report(y_test, rf_pred))
print("AdaBoost\n", classification_report(y_test, ada_pred))
print("semi-supervised\n", classification_report(y_test, self_training_model_pred))



# In[34]:


# example on how to store the values of a classification report to a dict
nb_report = classification_report(y_test, nb_pred, output_dict=True)
nb_f1 = nb_report['weighted avg']['f1-score']
logreg_report =classification_report(y_test, logreg_pred, output_dict=True)
logreg_f1 = logreg_report['weighted avg']['f1-score']
knn_report = classification_report(y_test, knn_pred, output_dict=True)
knn_f1 = knn_report['weighted avg']['f1-score']
dt_report = classification_report(y_test, dt_pred, output_dict=True)
dt_f1 = dt_report['weighted avg']['f1-score']
rf_report = classification_report(y_test, rf_pred, output_dict=True)
rf_f1 = rf_report['weighted avg']['f1-score']
ada_report = classification_report(y_test, ada_pred, output_dict=True)
ada_f1 = ada_report['weighted avg']['f1-score']
self_training_model_report = classification_report(y_test, self_training_model_pred, output_dict=True)
self_training_model_f1 = self_training_model_report['weighted avg']['f1-score']

scores = {"Model": ["NB", "Logistic Regression", "KNN", "Decision Tree", "Random Forrest", "ADA", "Self Training"],\
          "Name": [nb, logreg, knn, dt, rf, ada, self_training_model], "F1-Score": [nb_f1,logreg_f1,\
                                                                                  knn_f1,dt_f1,rf_f1,\
                                                                                  ada_f1,self_training_model_f1]}
df_scores = pd.DataFrame(scores)
df_scores
# it can be added to a df too


# In[35]:


# Select the model with the highest F1 score 
best_model_name = df_scores.loc[df_scores['F1-Score'].idxmax()]['Name'] 
best_model_name


# **Get the scores for the tokens**

# **Test example**

# In[38]:


test_data = {"X": ["I arrived home and found that my door lock was damage. ", "I was walking and someone punched me. I called the police.","I came back to my car after shopping and found that the door was damaged. My wallet left in the car was stolen.","a man came into my store and point a knife at me. he asked me to give him all the money."]}
pd_test_data = pd.DataFrame(test_data)
pd_test_data = pd_test_data.X
pd_test_data


# In[39]:


# transform the data
test_dtm = vect_tfidf.transform(pd_test_data)
pred = best_model_name.predict(test_dtm)
pred


# In[40]:


user_input = st.text_input("Enter your text here:")
st.write(user_input)

