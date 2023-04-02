#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Data Set

# Data Pre-Processing

# In[2]:


# Loading packages.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#

import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

#

import math
import random
import os
import time

from numpy import interp

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Disabling warnings:

import warnings
warnings.filterwarnings('ignore') 

data = pd.read_csv('/Users/JumpMan/Downloads/UH Spring 2023/heart_cleveland_upload.csv')
data.head()


# In[6]:


data.info()


# In[7]:


df = data


# # SVM

# In[8]:


# Split the data into training and testing sets
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM models with different kernel functions
models = {'linear': SVC(kernel='linear', random_state=42),
          'rbf': SVC(kernel='rbf', random_state=42, gamma=0.1),
          'sigmoid': SVC(kernel='sigmoid', random_state=42, gamma=0.1)}

# Train and evaluate the models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append([name, accuracy])

# Print the results in a table
print(pd.DataFrame(results, columns=['Kernel', 'Accuracy']))


# # MLP

# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MLP models with different hyperparameters
models = {'relu_SGD_0.01': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', learning_rate_init=0.01, random_state=42),
          'relu_SGD_0.001': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='sgd', learning_rate_init=0.001, random_state=42),
          'tanh_SGD_0.01': MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='sgd', learning_rate_init=0.01, random_state=42),
          'tanh_SGD_0.001': MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='sgd', learning_rate_init=0.001, random_state=42),
          'relu_Adam_0.01': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate_init=0.01, random_state=42),
          'relu_Adam_0.001': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', learning_rate_init=0.001, random_state=42),
          'tanh_Adam_0.01': MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='adam', learning_rate_init=0.01, random_state=42),
          'tanh_Adam_0.001': MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', solver='adam', learning_rate_init=0.001, random_state=42)}

# Train and evaluate the models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append([name, accuracy])

# Print the results in a table
print(pd.DataFrame(results, columns=['Hyperparameters', 'Accuracy']))


# In[ ]:




