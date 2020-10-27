#!/usr/bin/env python
# coding: utf-8

# In[26]:


# Import the required packages 

import pandas as pd
import numpy as np
from pandas import datetime 
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


consumption = pd.read_csv('household_power_consumption.txt', sep = ';', parse_dates= ['Date'], infer_datetime_format=True, low_memory=False,  na_values=['nan','?'])


# In[5]:


consumption.head()


# In[6]:


consumption.describe()


# In[8]:


consumption.info()


# In[9]:


consumption.isna().sum()


# # Drop null values

# In[10]:


consumption = consumption.dropna()
consumption.isna().sum()


# # Average consumption of each day in 4 years

# In[13]:


from pandas import datetime
mean_consumption_gby_date = consumption.groupby(['Date']).mean()


# In[12]:


fig, axs = plt.subplots(3, 2, figsize = (30, 25))
columns = mean_consumption_gby_date.columns
axs[0, 0].plot(mean_consumption_gby_date[columns[0]])
axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)

axs[0, 1].plot(mean_consumption_gby_date[columns[1]])
axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)

axs[1, 0].plot(mean_consumption_gby_date[columns[2]])
axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)

axs[1, 1].plot(mean_consumption_gby_date[columns[3]])
axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)

axs[2, 0].plot(mean_consumption_gby_date[columns[4]])
axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)

axs[2, 1].plot(mean_consumption_gby_date[columns[5]])
axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)

fig, axs = plt.subplots( figsize = (20, 4))
axs.plot(mean_consumption_gby_date[columns[6]])
axs.set_title(columns[6], fontweight = 'bold', size = 15)


# In[14]:


mean_consumption_gby_month = consumption.groupby(consumption['Date'].dt.strftime('%B')).mean()
reorderlist = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December' ]
mean_consumption_gby_month = mean_consumption_gby_month.reindex(reorderlist)


# In[15]:


fig, axs = plt.subplots(3, 2, figsize = (30, 25))
columns = mean_consumption_gby_month.columns

axs[0, 0].plot(mean_consumption_gby_month[columns[0]])
axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)

axs[0, 1].plot(mean_consumption_gby_month[columns[1]])
axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)

axs[1, 0].plot(mean_consumption_gby_month[columns[2]])
axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)

axs[1, 1].plot(mean_consumption_gby_month[columns[3]])
axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)


axs[2, 0].plot(mean_consumption_gby_month[columns[4]])
axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)

axs[2, 1].plot(mean_consumption_gby_month[columns[5]])
axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)

fig, axs = plt.subplots( figsize = (20, 4))
axs.plot(mean_consumption_gby_month[columns[6]])
axs.set_title(columns[6], fontweight = 'bold', size = 15)


# In[16]:


import pandas as pd
import numpy as np
from pandas import datetime as dt
import matplotlib.pyplot as plt
pd.plotting.register_matplotlib_converters()


# In[17]:


consumption_2 = pd.read_csv('household_power_consumption.txt', sep=';', 
                 parse_dates={'dt' : ['Date', 'Time']}, infer_datetime_format=True, 
                 low_memory=False, na_values=['nan','?'], index_col='dt')


# # Average consumption of each day in a month

# In[18]:


mean_consumption_gby_day_month = consumption_2.groupby(consumption_2.index.day).mean()


# In[19]:


fig, axs = plt.subplots(3, 2, figsize = (30, 25))
columns = mean_consumption_gby_day_month.columns

axs[0, 0].plot(mean_consumption_gby_day_month[columns[0]])
axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)

axs[0, 1].plot(mean_consumption_gby_day_month[columns[1]])
axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)

axs[1, 0].plot(mean_consumption_gby_day_month[columns[2]])
axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)

axs[1, 1].plot(mean_consumption_gby_day_month[columns[3]])
axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)

axs[2, 0].plot(mean_consumption_gby_day_month[columns[4]])
axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)

axs[2, 1].plot(mean_consumption_gby_day_month[columns[5]])
axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)

fig, axs = plt.subplots( figsize = (20, 4))
axs.plot(mean_consumption_gby_day_month[columns[6]])
axs.set_title(columns[6], fontweight = 'bold', size = 15)


# # Average consumption of each day in a week

# In[20]:


days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']
mean_consumption_gby_day_week = consumption_2.groupby(consumption_2.index.day_name()).mean().reindex(days)


# In[21]:


fig, axs = plt.subplots(3, 2, figsize = (30, 25))
columns = mean_consumption_gby_day_week.columns

axs[0, 0].plot(mean_consumption_gby_day_week[columns[0]])
axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)

axs[0, 1].plot(mean_consumption_gby_day_week[columns[1]])
axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)

axs[1, 0].plot(mean_consumption_gby_day_week[columns[2]])
axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)

axs[1, 1].plot(mean_consumption_gby_day_week[columns[3]])
axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)

axs[2, 0].plot(mean_consumption_gby_day_week[columns[4]])
axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)

axs[2, 1].plot(mean_consumption_gby_day_week[columns[5]])
axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)

fig, axs = plt.subplots( figsize = (20, 4))
axs.plot(mean_consumption_gby_day_week[columns[6]])
axs.set_title(columns[6], fontweight = 'bold', size = 15)


# # Average consumption of each hour in a day

# In[22]:


consumption_resampled_in_a_day = consumption_2.resample('H').sum()
consumption_resampled_in_a_day.index = consumption_resampled_in_a_day.index.time
mean_consumption_gby_time = consumption_resampled_in_a_day.groupby(consumption_resampled_in_a_day.index).mean()


# In[23]:


fig, axs = plt.subplots(3, 2, figsize = (30, 25))
columns = mean_consumption_gby_time.columns

axs[0, 0].plot(mean_consumption_gby_time[columns[0]])
axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)

axs[0, 1].plot(mean_consumption_gby_time[columns[1]])
axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)

axs[1, 0].plot(mean_consumption_gby_time[columns[2]])
axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)

axs[1, 1].plot(mean_consumption_gby_time[columns[3]])
axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)

axs[2, 0].plot(mean_consumption_gby_time[columns[4]])
axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)

axs[2, 1].plot(mean_consumption_gby_time[columns[5]])
axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)

fig, axs = plt.subplots( figsize = (20, 4))
axs.plot(mean_consumption_gby_time[columns[6]])
axs.set_title(columns[6], fontweight = 'bold', size = 15)


# # Average consumption of each month in 4 years

# In[24]:


mean_consumption_resampled_mnthly = consumption_2.resample('M').mean()


# In[25]:


fig, axs = plt.subplots(3, 2, figsize = (30, 25))
columns = mean_consumption_resampled_mnthly.columns

axs[0, 0].plot(mean_consumption_resampled_mnthly[columns[0]])
axs[0, 0].set_title(columns[0], fontweight = 'bold', size = 20)

axs[0, 1].plot(mean_consumption_resampled_mnthly[columns[1]])
axs[0, 1].set_title(columns[1], fontweight = 'bold', size = 20)

axs[1, 0].plot(mean_consumption_resampled_mnthly[columns[2]])
axs[1, 0].set_title(columns[2], fontweight = 'bold', size = 20)

axs[1, 1].plot(mean_consumption_resampled_mnthly[columns[3]])
axs[1, 1].set_title(columns[3], fontweight = 'bold', size = 20)

axs[2, 0].plot(mean_consumption_resampled_mnthly[columns[4]])
axs[2, 0].set_title(columns[4], fontweight = 'bold', size = 20)

axs[2, 1].plot(mean_consumption_resampled_mnthly[columns[5]])
axs[2, 1].set_title(columns[5], fontweight = 'bold', size = 20)

fig, axs = plt.subplots( figsize = (20, 4))
axs.plot(mean_consumption_resampled_mnthly[columns[6]])
axs.set_title(columns[6], fontweight = 'bold', size = 15)


# In[ ]:




