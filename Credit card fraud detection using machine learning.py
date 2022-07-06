#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[4]:


data=pd.read_csv(r"C:\Users\Acer Nitro\Downloads\archive\creditcard.csv")


# In[5]:


data


# In[9]:


data.head()


# In[10]:


data.tail()


# In[13]:


fraud = data.loc[data['Class'] == 1]
normal = data.loc[data['Class'] == 0]


# In[14]:


fraud.sum()


# In[15]:


len(normal)


# In[16]:


len(fraud)


# In[19]:


sns.relplot(x = "Amount", y ="Time", hue ="Class",data=data)


# In[20]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[22]:


X = data.iloc[:,:-1]
y = data['Class']


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.35)


# In[25]:


clf = linear_model.LogisticRegression(C=1e5)


# In[27]:


clf.fit(X_train, y_train)


# In[29]:


y_pred = np.array(clf.predict(X_test))
y = np.array(y_test)


# In[30]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[31]:


print(confusion_matrix(y_test, y_pred))


# In[32]:


print(accuracy_score(y_test, y_pred))


# In[33]:


print(classification_report(y_test, y_pred))


# In[ ]:




