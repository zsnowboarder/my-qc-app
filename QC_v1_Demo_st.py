#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Import libaries for the project
import streamlit as st
import numpy as np
import pandas as pd
import string
import pickle

from sklearn.feature_extraction import _stop_words
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('punkt')

# In[17]:
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
    #doc = spell(doc) # very slow
    doc = clean(doc)
    doc = lemmatize(doc)
    doc = stem(doc)
    return doc


with open('/mount/src/my-qc-app/vect_tfidf.pkl', 'rb') as file:
    vect_tfidf = pickle.load(file)
    
with open("/mount/src/my-qc-app/model_logreg.pkl", "rb") as file:
    my_model = pickle.load(file)

class_labels = my_model.classes_ # get the class name from the model.
vocabulary_size = len(vect_tfidf.vocabulary_)
# In[25]:


# create interface


intro = """This demo showcases a simple yet powerful tool for classifying police reports using machine learning and natural language processing.\n
Despite being a very basic model trained on just 40 artificial data points and 423 vocabularies without any advanced text pre-processing techniques and model fine-tuning, 
it still achieves impressive classification results. Imagine the potential this tool when fully optimized 
and trained on more extensive data. Machine learning can streamline the process of a diverse classification tasks, saving 
valuable time and resources. It’s designed to be user friendly and efficient, making it an invaluable asset for police departments and other industries."""

st.title("Incident Classifier")
st.write('')
st.write(intro)
st.write('')
st.write('This model can classify assaults, thefts, TFA, BNE, and robberies. Offences outside of these categories will be classified with a low probability indicator.')

new_data = st.text_area("Enter a synopsis. The more text entered, the better the classification.", height=200, value="I was walking and someone punched me for no reason. I had minor injuries. I reported the incident to police.")

if st.button("Classify"):
    
    new_data = {"X":[new_data]}
    df_new_data = pd.DataFrame(new_data)
    df_new_data = df_new_data.X.str.lower()
    
    new_data_vect = vect_tfidf.transform(df_new_data)
    new_pred = my_model.predict(new_data_vect)
    pred_prob = my_model.predict_proba(new_data_vect)[0]
    
    # sort the prob in descending order and then get the first and second highest
    sorted_index = np.argsort(pred_prob)[::-1]
    highest_prob = round(pred_prob[sorted_index[0]]*100)
    second_prob = round(pred_prob[sorted_index[1]]*100)
    new_pred_highest = my_model.classes_[sorted_index[0]]
    new_pred_second = my_model.classes_[sorted_index[1]]

    # clear the pred text label
    pred_msg = ""
    # set the result to a label
    if highest_prob > 30:
        pred_msg = "I am " + str(highest_prob) + "% confident that this can be classified as " + new_pred_highest + "."
    
    elif highest_prob > 20:
        pred_msg = "Since I was trained on only an **extremely small dataset with only 423 vocabularies**, I will have to make an inference on your text input. It would be " + new_pred_highest + " or " + new_pred_second + " or both."
    
    else: 
        pred_msg = "I am not familiar with most words entered here because I did not see them during training. Please retype and click **Classify** again."
        
    st.write(pred_msg)
    st.write('----------------------------------')
    st.write('**My accuracy will improve with more training data from real police reports. Probabilities are shown below.**')
    
    
    for label, prob in zip(class_labels, pred_prob):
        st.write(f"**Category:** {label}  --->  **Probability:** {prob * 100:.2f}%")
    


# In[ ]:




