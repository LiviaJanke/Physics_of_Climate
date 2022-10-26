#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd 


# In[4]:


hadcrut = pd.read_csv(
    "monthlyclimatedata.txt",
    delim_whitespace=True,
    usecols=[0, 1],
    header=None)

hadcrut['year'] = hadcrut.iloc[:, 0].apply(lambda x: x.split("/")[0]).astype(int)
hadcrut['month'] = hadcrut.iloc[:, 0].apply(lambda x: x.split("/")[1]).astype(int)
hadcrut = hadcrut.rename(columns={1: "value"})
hadcrut = hadcrut.iloc[:, 1:]


# In[5]:


hadcrut = hadcrut.set_index(['year', 'month'])
hadcrut -= hadcrut.loc[1850:1900].mean()
hadcrut = hadcrut.reset_index()

hc_1850 = hadcrut[hadcrut['year'] == 1850]
fig = plt.figure(figsize=(8,8))
ax1 = plt.subplot(111, projection='polar')
r = hc_1850['value'] + 1
theta = np.linspace(0, 2*np.pi, 12)


# In[6]:


fig = plt.figure(figsize=(14,14))
ax1 = plt.subplot(111, projection='polar')

ax1.axes.get_yaxis().set_ticklabels([])
ax1.axes.get_xaxis().set_ticklabels([])
fig.set_facecolor("#323331")
ax1.set_ylim(0, 3.25)

theta = np.linspace(0, 2*np.pi, 12)

ax1.set_title("Global Temperature Change (1850-2021)", color='white', fontdict={'fontsize': 20})
ax1.set_facecolor('#000100')

years = hadcrut['year'].unique()

for year in years:
    r = hadcrut[hadcrut['year'] == year]['value'] + 1
     # ax1.text(0,0, str(year), color='white', size=30, ha='center')
    ax1.plot(theta, r)

    
#Temperature rings

full_circle_thetas = np.linspace(0, 2*np.pi, 1000)
blue_line_one_radii = [1.0]*1000
red_line_one_radii = [2.5]*1000
red_line_two_radii = [3.0]*1000

ax1.plot(full_circle_thetas, blue_line_one_radii, c='blue')
ax1.plot(full_circle_thetas, red_line_one_radii, c='red')
ax1.plot(full_circle_thetas, red_line_two_radii, c='red')

ax1.text(np.pi/2, 1.0, "0.0 C", color="blue", ha='center', fontdict={'fontsize': 20})
ax1.text(np.pi/2, 2.5, "1.5 C", color="red", ha='center', fontdict={'fontsize': 20})
ax1.text(np.pi/2, 3.0, "2.0 C", color="red", ha='center', fontdict={'fontsize': 20})


# In[ ]:




