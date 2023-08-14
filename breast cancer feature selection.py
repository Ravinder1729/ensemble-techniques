#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#feature selection with variance


# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('C:/Users/ravin/Downloads/cancer.csv')


# In[4]:


df.head()


# In[5]:


from sklearn.feature_selection import VarianceThreshold


# In[6]:


selector= VarianceThreshold()


# In[7]:


df.shape


# In[8]:


x=df.drop(labels=['diagnosis'],axis=1)
y=df['diagnosis']


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(
     x, y, test_size=0.33, random_state=0)


# In[11]:


selector= VarianceThreshold(threshold=0.0)


# In[12]:


selector.fit(x_train)


# In[13]:


x_train.shape


# In[14]:


selector.get_support()


# In[15]:


sum(selector.get_support())


# In[16]:


constant_columns=[columns for columns in x_train.columns if columns not in x_train.columns[selector.get_support()]]


# In[17]:


print(constant_columns)


# In[18]:


x_train.drop('Unnamed: 32',axis=1,inplace=True)


# In[ ]:


#removing features baased on pearson correlation


# In[19]:


x_train.corr()


# In[20]:


import seaborn as sns
plt.figure(figsize=(15,10))
cor=x_train.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.Accent
           )


# In[21]:


import numpy as np
import pandas as pd

def correlation(dataset, threshold):
    corr_matrix = dataset.corr()
    high_corr_features = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                high_corr_features.add(colname)

    return high_corr_features



# In[22]:


cor_features=correlation(x_train,0.9)


# In[23]:


len(set(cor_features))


# In[24]:


cor_features


# In[25]:


x_train.head()


# In[26]:


x_train=x_train.drop(columns=['area_mean',
 'area_se',
 'area_worst',
 'compactness_worst',
 'concave points_mean',
 'concave points_worst',
 'concavity_mean',
 'concavity_worst',
 'fractal_dimension_worst',
 'perimeter_mean',
 'perimeter_se',
 'perimeter_worst',
 'radius_worst',
 'smoothness_worst',
 'texture_worst'], axis=1)


# In[27]:


x_train.drop('id',axis=1,inplace=True)


# In[29]:


x_train


# In[ ]:


#feature selection based on mutual information


# In[30]:


from sklearn.feature_selection import mutual_info_classif


# In[31]:


mutual_info=mutual_info_classif(x_train,y_train)
mutual_info


# In[32]:


mutual_info=pd.Series(mutual_info)
mutual_info.index=x_train.columns
mutual_info.sort_values(ascending=False)


# In[33]:


mutual_info.sort_values(ascending=False).plot.bar(figsize=(20,8))


# In[34]:


from sklearn.feature_selection import SelectKBest


# In[37]:


sel_five_col=SelectKBest(mutual_info_classif,k=5)
sel_five_col.fit(x_train,y_train)
x_train.columns[sel_five_col.get_support()]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




