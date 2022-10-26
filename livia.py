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

co2_datetime = pd.date_range(start='1/03/1958', periods = 775,freq = 'M')

monthly_co2_df['Date'] = co2_datetime


#%%

#plotting decadal mean co2
#off by about 5 months - doesn't really matter


dec_mean_co2 = monthly_co2_df.resample('10A', on = 'Date').mean()

dec_mean_co2.plot('decimal_date','average', marker = 'x')

#%%


#annual growth rates since the 60s of co2

#1958 - 1968
co2_gr1 = (dec_mean_co2['average'][1] - dec_mean_co2['average'][0])/10
co2_gr2 = (dec_mean_co2['average'][2] - dec_mean_co2['average'][1])/10
co2_gr3 = (dec_mean_co2['average'][3] - dec_mean_co2['average'][2])/10
co2_gr4 = (dec_mean_co2['average'][4] - dec_mean_co2['average'][3])/10
co2_gr5 = (dec_mean_co2['average'][5] - dec_mean_co2['average'][4])/10
co2_gr6 = (dec_mean_co2['average'][6] - dec_mean_co2['average'][5])/10
co2_gr7 = (dec_mean_co2['average'][7] - dec_mean_co2['average'][6])/10

#co2 growth rates

growth_rates_co2 = np.array([co2_gr1, co2_gr2, co2_gr3, co2_gr4, co2_gr5, co2_gr6, co2_gr7])

decades_co2 = np.array([1968, 1978 , 1988, 1998, 2008, 2018,2028])


plt.title('growth rate against end of decade for co2')
plt.xlabel('Decades')
plt.ylabel('growth rate of co2 ppm/yrs')
plt.plot(decades_co2, growth_rates_co2, marker = 'x')

#%%

#temperature growth rates

temp_gr1 = (dec_mean_temp['Mean'][1] - dec_mean_temp['Mean'][0])/10
temp_gr2 = (dec_mean_temp['Mean'][2] - dec_mean_temp['Mean'][1])/10
temp_gr3 = (dec_mean_temp['Mean'][3] - dec_mean_temp['Mean'][2])/10
temp_gr4 = (dec_mean_temp['Mean'][4] - dec_mean_temp['Mean'][3])/10
temp_gr5 = (dec_mean_temp['Mean'][5] - dec_mean_temp['Mean'][4])/10
temp_gr6 = (dec_mean_temp['Mean'][6] - dec_mean_temp['Mean'][5])/10
temp_gr7 = (dec_mean_temp['Mean'][7] - dec_mean_temp['Mean'][6])/10
temp_gr8 = (dec_mean_temp['Mean'][8] - dec_mean_temp['Mean'][7])/10
temp_gr9 = (dec_mean_temp['Mean'][9] - dec_mean_temp['Mean'][8])/10
temp_gr10 = (dec_mean_temp['Mean'][10] - dec_mean_temp['Mean'][9])/10
temp_gr11 = (dec_mean_temp['Mean'][11] - dec_mean_temp['Mean'][10])/10
temp_gr12 = (dec_mean_temp['Mean'][12] - dec_mean_temp['Mean'][11])/10
temp_gr13 = (dec_mean_temp['Mean'][13] - dec_mean_temp['Mean'][12])/10
temp_gr14 = (dec_mean_temp['Mean'][14] - dec_mean_temp['Mean'][13])/10

growth_rates_temp = np.array([temp_gr1, temp_gr2, temp_gr3, temp_gr4, temp_gr5, temp_gr6, temp_gr7, temp_gr8, temp_gr9, temp_gr10, temp_gr11, temp_gr12, temp_gr13, temp_gr14])


decades_temp = np.array([1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])

plt.title('decadal growth rate in temperature anomaly in degrees/year')
plt.ylabel('temperature anomaly annual growth rate')
plt.xlabel('end of decade')
plt.plot(decades_temp, growth_rates_temp, marker = 'x')














