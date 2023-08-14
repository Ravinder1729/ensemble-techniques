#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[1]:


df=pd.read_excel('C:/Users/ravin/Downloads/cust_data.xlsx')


# In[2]:


df.head()


# In[3]:


df.shape


# In[8]:


df.columns


# In[7]:


df.info()


# In[9]:


df.describe()


# In[15]:


[columns for columns in df.columns if df[columns].isnull().sum()>0]


# In[17]:


df['Gender'].value_counts()


# In[18]:


df['Gender'].isnull().sum()


# In[19]:


df['Gender'].unique()


# In[29]:


df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)


# In[30]:


df.head()


# In[31]:


[columns for columns in df.columns if df[columns].isnull().sum()>0]


# In[32]:


df.duplicated()


# In[42]:


df['Gender']=df['Gender'].map({'M':1,'F':0})


# In[43]:


df['Gender'] = df['Gender'].astype('int')


# In[45]:


df.dtypes


# In[49]:


x=df.drop(labels='Orders',axis=1)
y=df['Orders']


# In[54]:


x.drop('Cust_ID',axis=1,inplace=True)


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


x_train, x_test, y_train, y_test = train_test_split(
     x, y, test_size=0.33, random_state=0)


# In[70]:


x_train


# In[ ]:


#removing features of low variance


# In[74]:


from sklearn.feature_selection import VarianceThreshold
selector= VarianceThreshold()


# In[75]:


selector.fit(x_train)


# In[76]:


x_train.shape


# In[77]:


selector.get_support()


# In[62]:


sum(selector.get_support())


# In[ ]:


#removing features baased on pearson correlation


# In[78]:


x_train.corr()


# In[79]:


import seaborn as sns
plt.figure(figsize=(15,10))
cor=x_train.corr()
sns.heatmap(cor,annot=True,cmap=plt.cm.Accent
           )


# In[65]:


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



# In[80]:


cor_features=correlation(x_train,0.7)


# In[81]:


cor_features


# In[84]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score


# In[101]:


wcss=[]
for i in range(1,11):
    kmeans= KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x_train)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.show()
    


# In[104]:


clusterer=KMeans(n_clusters=4,random_state=10)
cluster_labels=clusterer.fit_predict(x_train)
print(cluster_labels)


# In[112]:


X=x_train
X


# In[116]:


import numpy as np
import matplotlib.cm as cm


# In[129]:


range_n_clusters = [ 2,3,4,5,6]

for n_clusters in range_n_clusters:
   
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    sample_silhouette_values = silhouette_samples(X, cluster_labels)


# In[ ]:




