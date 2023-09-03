#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Library#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[4]:


#Import Data#
df_customer = pd.read_csv('Case Study - Customer.csv', sep = ';')
df_product = pd.read_csv('Case Study - Product.csv', sep = ';')
df_store = pd.read_csv('Case Study - Store.csv', sep = ';')
df_transaction = pd.read_csv('Case Study - Transaction.csv', sep = ';')


# In[5]:


#Data Preparation and Data Cleansing#
#Mengecek nilai yang hilang
missing_values1 = df_customer.isnull().sum()
missing_values2 = df_product.isnull().sum()
missing_values3 = df_store.isnull().sum()
missing_values4 = df_transaction.isnull().sum()


# In[6]:


#Mengecek duplikat
duplikat_rows1 = df_customer.duplicated()
duplikat_rows2 = df_product.duplicated()
duplikat_rows3 = df_store.duplicated()
duplikat_rows4 = df_transaction.duplicated()


# In[7]:


df_customer.info()


# In[8]:


df_product.info()


# In[9]:


df_store.info()


# In[10]:


df_transaction.info()


# In[11]:


#Menggabungkan data#
merged_data1 = pd.merge(df_customer, df_transaction, on = 'CustomerID')
merged_data2 = pd.merge(merged_data1, df_store, on = 'StoreID')
merged = pd.merge(merged_data2, df_product, on = 'ProductID')
merged.head()


# In[12]:


#Time Series#
#Mengubah tipe data tanggal ke datetime
merged['Date'] = pd.to_datetime(merged['Date'])
merged['Longitude'] = merged['Longitude'].apply(lambda x: x.replace(',','.')).astype(float)
merged['Latitude'] = merged['Latitude'].apply(lambda x: x.replace(',','.')).astype(float)


# In[13]:


#Membuat Data Time Series
daily_data = merged.groupby('Date')['Qty'].sum().reset_index()
data = daily_data.set_index('Date')
data2 = data.resample('D').sum()
data2.head()


# In[14]:


#Pisahkan data
train_size = int(len(data2)*0.8)
train_data, test_data = data2[:train_size], data2[train_size:]
print(train_data.shape, test_data.shape)


# In[15]:


#Import Library#
import seaborn as sns
plt.figure(figsize=(15,6))
sns.lineplot(data=train_data, x=train_data.index, y=train_data['Qty'])
sns.lineplot(data=test_data, x=test_data.index, y=test_data['Qty'])
plt.show()


# In[16]:


#Membuat Model ARIMA#
from statsmodels.tsa.arima.model import ARIMA
#1. Tentukan nilai p, d dan q
p = 2
d = 2
q = 2
#2. Buat model ARIMA
model = ARIMA(train_data, order=(p, d, q))
#3. Latih model dengan data
model_fit = model.fit()


# In[17]:


start_idx = len(train_data)
end_idx = len(train_data) + len(test_data) - 1
predictions = model_fit.predict(start=start_idx, end=end_idx, dynamic=False)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_data, predictions)
print(f"Mean Squared Error: {mse}")


# In[18]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,6))
plt.plot(test_data, label='Qty')
plt.plot(predictions, color="green", label='Predicted')
plt.legend()
plt.show()


# In[19]:


#Clustering#
#Menggabungkan data CustomerID dan menghitung metrik
aggregated = merged.groupby('CustomerID').agg({
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}).reset_index()
aggregated


# In[20]:


Y = aggregated[['TransactionID','Qty','TotalAmount']]
#Clustering KMeans#
from sklearn.cluster import KMeans
Kmeans = KMeans(n_clusters=3, random_state=42)
aggregated['cluster']= Kmeans.fit_predict(Y)


# In[21]:


#Membuat Plot
import matplotlib.pyplot as plt
plt.scatter(aggregated['Qty'], aggregated['TotalAmount'], c=aggregated['cluster'], cmap='rainbow')
plt.title('Clustering Result')
plt.xlabel('Qty')
plt.ylabel('Total Amount')
plt.show()


# In[22]:


#Within-Cluster Sum of Squared (WCSS)#
wcss = []
for n in range (1,11):
    model1 = KMeans(n_clusters=n, init = 'k-means++', n_init = 10, max_iter=100, tol = 0.0001, random_state = 100)
    model1.fit(Y)
    wcss.append(model1.inertia_)
print(wcss)


# In[23]:


plt.figure(figsize=(12,9))
plt.plot(list(range(1,11)), wcss, color = 'pink', marker = 'o', linewidth = 2, markersize = 15, markerfacecolor = 'm',
        markeredgecolor= 'm')
plt.title('WCSS VS Number of Cluster')
plt.xlabel('Number of Cluster')
plt.ylabel('WCSS')
plt.xticks(list(range(1,11)))
plt.grid()
plt.show()


# In[48]:


#Model Cluster dengan K optimal
model1 = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=100)
model1.fit(Y)
labels1 = model1.labels_


# In[51]:


#Input Cluster ke Dataset
df_cluster['cluster'] = model1.labels_
df_cluster.head()


# In[43]:


plt.figure(figsize=(6,6))
sns.pairplot(data=df_cluster, hue='cluster', palette='Set1')
plt.show()


# In[ ]:




