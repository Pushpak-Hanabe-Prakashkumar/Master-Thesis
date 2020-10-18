#!/usr/bin/env python
# coding: utf-8
# start
# In[1]:


# Import the required packages 

import sys # Used to acess the module within the function
import numpy as np # Used for linear algebra
from scipy.stats import randint # Used for statistical functions
import pandas as pd # Used for 2d labeled data structure with colums of different types
import matplotlib.pyplot as plt # used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # Used to split the data into two parts
from sklearn.model_selection import KFold # Used for cross validation
from sklearn.preprocessing import StandardScaler # Used for normalization
from sklearn.preprocessing import MinMaxScaler # Used for scaling the feature
from sklearn.pipeline import Pipeline # Used for pipeline making
from sklearn.model_selection import cross_val_score # Used to split data and score from the separate folder
from sklearn.feature_selection import SelectFromModel # Used for selection of model from the used model
from sklearn import metrics # Used for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score # Used to measure the avg difference between true and estimated value

## For LSTM and ML

import keras # Used to run neural network library 
from keras.layers import Dense # Used for the operation on input to return output
from keras.models import Sequential # Used to create models layer by layer
from keras.utils import to_categorical # Used to convert array to vector
from keras.optimizers import SGD # Used for optimization 
from keras.callbacks import EarlyStopping # Used to train model and stops when its performace ends
from keras.utils import np_utils # Used to convert array to vector
import itertools # Used to iterate the sequential data sets 
from keras.layers import LSTM # Used for time-series forecasting
from keras.layers.convolutional import Conv1D # Used to predict the continious data 
from keras.layers.convolutional import MaxPooling1D # Used to calculate max value
from keras.layers import Dropout # Used to prevent model from overfitting


# In[2]:


## Download the file from this link ( https://www.kaggle.com/uciml/electric-power-consumption-data-set ) and extract it
## file 'household_power_consumption.txt' put it in the directory and run the code

df = pd.read_csv('household_power_consumption.txt', sep=';', 
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'], index_col='dt')


# In[3]:


df.head() 


# In[4]:


df.info()


# In[5]:


df.dtypes


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


for j in range(1,7):
       print(df.iloc[:, j].unique())


# In[10]:


## finding all columns that have nan:

droping_list_all=[]
for j in range(0,7):
    if not df.iloc[:, j].notnull().all():
        droping_list_all.append(j)        
        #print(df.iloc[:,j].unique())
droping_list_all


# In[11]:


# filling nan with mean in any columns

for j in range(0,7):        
        df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())


# In[12]:


# another sanity check to make sure that there are not more any nan
df.isnull().sum()


# In[13]:


df.describe()


# In[14]:


df['Global_active_power'].resample('M').sum()


# In[15]:


df.Global_active_power.resample('D').sum().plot(title='Global_active_power resampled over day for sum') 
 
plt.tight_layout()
plt.show()   

df.Global_active_power.resample('D').mean().plot(title='Global_active_power resampled over day for mean', color='red') 
plt.tight_layout()
plt.show()


# In[16]:


r = df.Global_intensity.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='Global_intensity resampled over day')
plt.show()


# In[17]:


r = df.Global_intensity.resample('D').agg(['mean', 'std'])
r.plot(subplots = True, title='Global_intensity resampled over day')
plt.show()


# In[ ]:




