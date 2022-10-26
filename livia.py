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


print(dec_mean_co2['average'])
print(dec_mean_temp)

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

#%%

growth_rates = np.array([co2_gr1, co2_gr2, co2_gr3, co2_gr4, co2_gr5, co2_gr6, co2_gr7])

decades = np.array([1960, 1970, 1980 , 1990, 2000, 2010, 2022])


plt.title('growth rate against decade for co2')
plt.xlabel('Decades')
plt.ylabel('growth rate ppm/yrs')
plt.plot(decades, growth_rates, marker = 'x')




