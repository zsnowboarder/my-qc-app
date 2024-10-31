#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Import libaries for the project
import streamlit as st
import numpy as np
import pandas as pd


# In[17]:


import pickle

with open("/mount/src/my-qc-app/model_svm.pkl", "rb") as file:
    svm_model = pickle.load(file)


# In[25]:


# create interface


intro = """This demo showcases a simple yet powerful tool for classifying police reports using machine learning and natural language processing.\n
Despite being a very basic model trained on just 40 data points without any advanced text pre-processing techniques and model fine-tuning, 
it still achieves impressive classification results. Imagine the potential this tool when fully optimized 
and trained on more extensive data. Machine learning can streamline the process of a diverse classification tasks, saving 
valuable time and resources. It’s designed to be user-friendly and efficient, making it an invaluable asset for police departments and other industries."""

st.title("Incident Classifier")
st.write('')
st.write(intro)
st.write('')
st.write('This model can classify assaults, thefts, TFA, BNE, and robberies. Offences outside of these categories will be classified with a low probability indicator.')

new_data = st.text_area("Enter a synopsis. The more text entered, the better the classification.", height=200, value="I was walking and someone punched me for no reason. I had minor injuries. I reported the incident to police.")

if st.button("Classify"):
    new_data = [new_data]
    new_pred = svm_model.predict(new_data)
    pred_prob = svm_model.predict_proba(new_data)[0]

    # clear the pred text label
    pred_msg = ""
    # set the result to a label
    pred_msg = "I am " + str(round(max(pred_prob)*100)) + "% confident that this can be classified as " + new_pred[0] + "."

        
    st.write(pred_msg)


# In[ ]:




