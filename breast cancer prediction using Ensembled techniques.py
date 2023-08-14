#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#import cancefr Dataset
df=pd.read_csv('C:/Users/ravin/Downloads/cancer.csv')


# In[16]:


df.head()


# In[12]:


df.shape


# In[117]:


df['diagnosis'].value_counts()


# In[ ]:





# In[6]:


df.columns


# In[7]:


df.dtypes


# In[9]:


df.isnull().sum()


# In[17]:


df.drop('Unnamed: 32',axis=1,inplace=True)


# In[19]:


df


# In[20]:


df['diagnosis']=df['diagnosis'].map({'M':1,'B':0})


# In[21]:


df.head()


# In[22]:


df.drop('id',axis=1,inplace=True)


# In[31]:


df.head()


# In[39]:


x=df.iloc[:,1:]
y=df['diagnosis']


# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(
     x, y, test_size=0.33, random_state=42)


# In[ ]:


#Ensemble techniques


# In[ ]:


#1.Decession tree


# In[48]:


from sklearn.tree import DecisionTreeClassifier


# In[49]:


classiffier=DecisionTreeClassifier()
classiffier.fit(x_train,y_train)


# In[54]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(classiffier, filled=True)


# In[93]:


decessiontreeAccuracy=classiffier.score(x_test,y_test)


# In[94]:


decessiontreeAccuracy


# In[58]:


y_pred=classiffier.predict(x_test)


# In[59]:


y_pred


# In[80]:


from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report


# In[99]:


decessiontreematrix=metrics.confusion_matrix(y_test,y_pred)


# In[100]:


decessiontreematrix


# In[116]:


metrics.classification_report(y_test,y_pred)


# In[69]:


#2.Random forest
from sklearn.ensemble import RandomForestClassifier
ranfclassiffier= RandomForestClassifier()


# In[70]:


ranfclassiffier.fit(x_train,y_train)


# In[79]:


y_pred=ranfclassiffier.predict(x_test)
y_pred


# In[95]:


randomforestaccuracy=ranfclassiffier.score(x_test,y_test)


# In[96]:


randomforestaccuracy


# In[77]:


ranfmatrix=metrics.confusion_matrix(y_test,y_pred)


# In[78]:


ranfmatrix


# In[82]:


metrics.classification_report(y_test,y_pred)


# In[83]:


#Adaboost classiffier
from sklearn.ensemble import AdaBoostClassifier
adaclassiffier=AdaBoostClassifier()


# In[84]:


adaclassiffier.fit(x_train,y_train)


# In[98]:


adaboostAccuracy=adaclassiffier.score(x_test,y_test)
adaboostAccuracy


# In[88]:


y_pred=adaclassiffier.predict(x_test)


# In[101]:


adamatrix=metrics.confusion_matrix(y_test,y_pred)


# In[102]:


adamatrix


# In[91]:


metrics.classification_report(y_test,y_pred)


# In[103]:


#xgboost classiffier
from sklearn.ensemble import GradientBoostingClassifier
xgclassiffier=GradientBoostingClassifier()


# In[104]:


xgclassiffier.fit(x_train,y_train)


# In[106]:


xgaccuracy=xgclassiffier.score(x_test,y_test)


# In[107]:


xgaccuracy


# In[108]:


y_pred=xgclassiffier.predict(x_test)


# In[109]:


y_pred


# In[110]:


xgmatrix=metrics.confusion_matrix(y_test,y_pred)


# In[111]:


xgmatrix


# In[112]:


metrics.classification_report(y_test,y_pred)


# In[ ]:




