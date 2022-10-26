# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:06:25 2022

@author: Admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import math
import seaborn as sns

#%%

#importing co2 data

year,month,decimal_date,average,deseasonalized,ndays,sdev,unc = np.loadtxt('Data/co2_month.csv', skiprows = 53, delimiter = ',', unpack = True)

monthly_co2_df = pd.read_csv('Data/co2_month.csv', skiprows = 52)

monthly_temp_df = pd.read_csv('Data/temp_monthly.csv')


df_gistemp = monthly_temp_df[monthly_temp_df.Source != "GCAG"]


df_gistemp['Date'] = pd.to_datetime(df_gistemp['Date'])


print(df_gistemp.head(10))


#%%

#plotting temperature

df_gistemp.plot('Date','Mean')

#plotting co2

monthly_co2_df.plot('decimal_date','average')

#%%


#Determining decadal mean temperature

dec_mean_temp = df_gistemp.resample('10A', on = 'Date').mean()


dec_mean_temp.plot()


#%%

print(monthly_co2_df.head())

#%%

monthly_co2_df['TimeStamp'] = pd.to_datetime(monthly_co2_df['decimal_date'], unit='y')






