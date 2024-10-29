#!/usr/bin/env python
# coding: utf-8

# In[15]:


# Import libaries for the project
import streamlit as st
import numpy as np
import pandas as pd


# In[17]:


import pickle

with open("model_svm.pkl", "rb") as file:
    svm_model = pickle.load(file)


# In[25]:


# create interface

st.title("QC Classifier")

new_data = st.text_input("Text", "I was walking and someone hit me.")

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




