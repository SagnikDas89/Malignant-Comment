#!/usr/bin/env python
# coding: utf-8

# In[8]:


get_ipython().system('pip install wordcloud')


# In[9]:


# importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import wordcloud
import torch

# for data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[10]:


# NLP tools
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# train split and fit models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


# In[11]:


# model selection
from sklearn.metrics import confusion_matrix, accuracy_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[13]:


#uploading csv file
data = pd.read_csv(r"C:\Users\SAGNIK DAS\OneDrive\Desktop\New folder (3)\train.csv")
dataset = data
dataset.dropna(inplace = True )
dataset


# In[14]:


dataset.info()

dataset.describe()
# In[15]:


dt_transformed = dataset[['malignant', 'highly_malignant', 'rude', 'threat', 'abuse', 'loathe', 'comment_text']]
y = dt_transformed.iloc[:, :-1].values


# In[16]:


# encodinng the dependent variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
y = np.array(ct.fit_transform(y))


# In[17]:


print(y)


# In[18]:


# this data has been split into two variables that will be used to fit hate speech and ofensive speech models.
y_df = pd.DataFrame(y)
y_hate = np.array(y_df[0])
y_offensive = np.array(y_df[1])


# In[19]:


print(y_hate)
print(y_offensive)


# In[23]:


# cleaning the text
corpus = []
for i in range (0, 159570):
    review = re.sub('[^a-zA-Z]', '', dt_transformed['comment_text'][i])
    review = review.lower()
    review = review.split()
    
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ''.join(review)
    corpus.append(review)


# In[20]:


cv = CountVectorizer(max_features = 5000)
x = cv.fit_transform(corpus).torray()


# In[30]:


# splitting the dataset into test set and train set. 
x_train, x_test, y_train, y_test = train_test_split(x, y_hate, test_size = .40, random_state = 0)


# In[31]:


# Naive Bayes
classifier_np = GaussianNB()
classifier_np.fit(x_train, y_train)


# In[32]:


# Logistic Regression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(x_train, y_train)


# In[33]:


# SVM Classifier
classifier_svm = svm.SVC()
classifier_svm.fit(x_train, y_train)


# In[34]:


# MAking the confussion matrix for each model
# Naive Bayes
y_pred_np = classifier_np.predict(x_test)
cm = confusion_matrix(x_test, y_pred_np)
print(cm)


# In[35]:


# SVM Classifier
y_pred_svm = classifier_svm.predict(x_test)
cm = confusion_matrix(x_test, y_pred_svm)
print(cm)


# In[36]:


# Logistic Regression
y_pred_lr = classifier_lr.predict(x_test)
cm = confusion_matrix(x_test, y_pred_lr)
print(cm)


# In[37]:


svm_score = accuracy_score(x_test, y_pred_svm)
lr_score = accuracy_score(x_test, y_pred_lr) 
np_score = accuracy_score(x_test, y_pred_np)


# In[38]:


# So based on this dataset, Support Vector Machine appears to be a superior predictor of hate speech. Logistic regression also produced excellent results. This dataset appears to be an artificial intelligence product used to classify hate and abusive speech.


# In[ ]:





# In[ ]:





# In[ ]:




