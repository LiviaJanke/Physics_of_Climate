
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def linear(x, a, b):
    return a*x + b

# Data analysis

#Load temperature data
t_mean, t_low_conf, t_up_conf = np.loadtxt("HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.csv",
                                            unpack=True, delimiter=",", skiprows=1, usecols=(1,2,3))
t_date = np.loadtxt("HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.csv",
                    unpack=True, delimiter=",", skiprows=1, usecols=0, dtype=str)

#Unpack datestring
t_year  = np.array([])
t_month = np.array([])
for timestr in t_date:
    t_year = np.append(t_year, float(timestr.split("-")[0]))
    t_month = np.append(t_month, float(timestr.split("-")[1]))

#Decadal mean and median
t_dec_mean   = np.array([])
t_dec_median = np.array([])
for i in range(0, len(t_mean[120:])):
    t_dec_mean   = np.append(t_dec_mean, np.mean(t_mean[i:120+i]))
    t_dec_median = np.append(t_dec_median, np.median(t_mean[i:120+i]))

#Fitting (linear fits of decades)
t_popt_array = np.array([])
t_pcov_array = np.array([])
t_fit = np.array([])
for d in range(185,202): #from 1850s to 2010s
    decade = d*10
    mask   = np.logical_and(t_year>=decade, t_year<(decade+10))
    popt, pcov = curve_fit(linear, np.linspace(decade, decade+10, 120, endpoint=False),
                          t_mean[mask]) #Parameter a is then in °C/year
    t_popt_array = np.append(t_popt_array, popt)
    t_pcov_array = np.append(t_pcov_array, pcov)
    t_fit = np.append(t_fit, linear(np.linspace(decade, decade+10, 120, endpoint=False), *popt))

#Load CO2 data
c_mean = np.loadtxt("co2_mm_mlo.txt", usecols=3)
c_year, c_month = np.loadtxt("co2_mm_mlo.txt", usecols=(0,1), unpack=True, dtype=int)

#Create datestring
c_date = np.array([])
for i in range(0, len(c_mean)):
    c_date = np.append(c_date, "{0:4d}-{1:02d}".format(c_year[i], c_month[i]))

#Decadal mean and median
c_dec_mean   = np.array([])
c_dec_median = np.array([])
for i in range(0, len(c_mean[120:])):
    c_dec_mean   = np.append(c_dec_mean, np.mean(c_mean[i:120+i]))
    c_dec_median = np.append(c_dec_median, np.median(c_mean[i:120+i]))

#Fitting (linear fits of decades)
c_popt_array = np.array([])
c_pcov_array = np.array([])
c_fit = np.array([])
for d in range(196,202): #from 1960s-2010s
    decade = d*10
    mask   = np.logical_and(c_year>=decade, c_year<decade+10)
    popt, pcov = curve_fit(linear, np.linspace(decade, decade+10, 120, endpoint=False),
                           c_mean[mask]) #Parameter a is then in °C/year
    c_popt_array = np.append(c_popt_array, popt)
    c_pcov_array = np.append(c_pcov_array, pcov)
    c_fit = np.append(c_fit, linear(np.linspace(decade, decade+10, 120, endpoint=False), *popt))


#Plots

fig = plt.figure(figsize=(12,10))

#temperature
ax = fig.add_subplot(211)
x = np.linspace(0,len(t_mean), len(t_mean))
ax.set_xticks(x[::60])
ax.set_xticklabels(t_year[::60], rotation=45)
ax.plot(x, t_mean, label="data")
ax.fill_between(x, t_up_conf, t_low_conf, alpha=0.4, label="95% confidence level")
ax.plot(x[60:-60], t_dec_mean, linewidth=2, color="orange", label="decadal mean")
ax.plot(x[60:-60], t_dec_median, linewidth=2, color="red", label="decadal median")
#decadal mean/median plotted at time t is mean/median of period (t-5yr, t+5yr)
ax.plot(x[np.logical_and(t_year>=1850, t_year<2020)], t_fit, ls="--", color="black", label="decadal linear fits") #Fits

ax.set_title("Time series of temperature anomaly\nreference period:1961-1990", y=0.85)
ax.set_ylabel("Temperature anomaly in °C")
ax.set_xlabel("Time in years")
ax.legend(loc="lower right")
plt.show()

#CO2
ax2 = fig.add_subplot(212)
x = np.linspace(0,len(c_mean), len(c_mean))
ax2.set_xticks(x[8::60])
ax2.set_xticklabels(c_year[8::60], rotation=45)
ax2.plot(x, c_mean, label="data")
ax2.plot(x[60:-60], c_dec_mean, linewidth=2, color="orange", label="decadal mean")
ax2.plot(x[60:-60], c_dec_median, linewidth=2, color="red", label="decadal median")
#decadal mean/median plotted at time t is mean/median of period (t-5yr, t+5yr)
ax2.plot(x[np.logical_and(c_year>=1960, c_year<2020)], c_fit, ls="--", color="black", label="decadal linear fits") #Fits
ax2.set_title("Time series of CO2-concentration", y=0.9)
ax2.set_ylabel("CO2 concentration in ppm")
ax2.set_xlabel("Time in years")
ax2.legend(loc="lower right")
plt.show()



# Andere Darstellung (alle in einem)

fig = plt.figure(figsize=(12,5))
x = np.linspace(0,len(t_mean), len(t_mean)) #use len(t_mean), because len(t_mean)>len(c_mean)

#Temperature axis
ax = fig.add_subplot(111)
line, = ax.plot(x, t_mean, label="temperature")
ax.fill_between(x, t_up_conf, t_low_conf, alpha=0.4, label="95% confidence level\n of temperature data")

ax.set_title("Time series of temperature anomaly\nand CO2-concentration", y=0.85)
ax.set_ylabel("Temperature anomaly in °C", color=line.get_color())
ax.text(60,0.7,"Reference period: 1961-1990")
ax.set_xlabel("Time in years")
ax.set_xticks(x[::60])
ax.set_xticklabels(t_year[::60], rotation=45)
ax.tick_params(axis="y", color=line.get_color(), labelcolor=line.get_color())

#Co2-axis
axt = ax.twinx()
linet, = axt.plot(x[len(t_mean)-len(c_mean):], c_mean, color="green", label="CO2")

axt.set_ylabel("CO2 concentration in ppm", color=linet.get_color())
axt.spines["left"].set_color(line.get_color())
axt.spines["right"].set_color(linet.get_color())

#Legend
handles, labels= ax.get_legend_handles_labels()
handlest, labelst = axt.get_legend_handles_labels()
handles = np.append(handlest, handles)
labels  = np.append(labelst, labels)
axt.tick_params(axis="y", color=linet.get_color(), labelcolor=linet.get_color())
axt.legend(handles, labels, loc="lower right", frameon=False)
