import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from global_land_mask import globe
import scipy.stats as stats
import os
import geopandas as gpd

def plot_hist(all_time_periods, earthquake_only, ax1, ax2, title1, title2):
    
    # Cumulative histogram

    bins = calculate_bin_sizes(earthquake_only)
    ax1.hist(earthquake_only, bins, density = True, cumulative=True, histtype='step',
            label='Time periods with an earthquake',linewidth=1.5)
    ax1.hist(all_time_periods, bins, density = True, cumulative=True,histtype='step',
            label='All time periods',linewidth=1.5)
    ax1.set_ylim((-0.1,1.3))
    ax1.legend()
    ax1.set_xlabel('Surface load (cm-we)', fontsize = 17)
    ax1.set_ylabel("Cumulative probability", fontsize = 17)
    ax1.set_title(title1)
    
    # Non-cumulative histogram

    ax2.hist(earthquake_only, bins, density = True, cumulative=False, histtype='step',
            label='Time periods with an earthquake',linewidth=1.5)
    ax2.hist(all_time_periods, bins, density = True, cumulative=False,histtype='step',
            label='All time periods',linewidth=1.5)
    yl = ax2.get_ylim()
    ax2.set_ylim((-0.01,1.4*yl[1]))
    ax2.set_xlim((-40,60))
    ax2.legend()
    ax2.set_xlabel('Surface load (cm-we)', fontsize = 17)
    ax2.set_ylabel("Probability", fontsize = 17)
    ax2.set_title(title2)
    
def plot_bayes(all_time_periods, earthquake_only, ax, title):
    
    plt.style.use('fivethirtyeight')

    cp,bins = calculate_bayes(earthquake_only,all_time_periods)

    wid = np.mean(np.diff(bins))
    print(len(bins))
    print(len(cp))
          
    ax.bar(bins[:-1],cp,width=wid,align='edge')

    ax.plot([-80,80],[1, 1],'--r')
#     ax.text(48,1.2,'P(M|SL)=P(SL)',color='r',fontsize=15)
    ax.set_xlabel('Surface load (cm-we.)',fontsize = 17)
    ax.set_ylabel('Relative conditional probability',fontsize = 17)
    ax.set_title(title, fontsize = 17)
    
def plot_bayes_fd(all_time_periods, earthquake_only, ax, title, fd):
    
    plt.style.use('fivethirtyeight')

    cp,bins = calculate_bayes(earthquake_only,all_time_periods)

#     wid = np.mean(np.diff(bins))
#     print(len(bins))
#     print(len(cp))
          
    ax.bar(fd[:-1],cp,width=0.9483457,align='edge')

    ax.plot([-80,80],[1, 1],'--r')
#     ax.text(48,1.2,'P(M|SL)=P(SL)',color='r',fontsize=15)
    ax.set_xlabel('Surface load (cm-we.)',fontsize = 17)
    ax.set_ylabel('Relative conditional probability',fontsize = 17)
    ax.set_title(title, fontsize = 17)

def calc_stats(a,b):
    '''
    Calculate stats for the distributions a and b
    a: distribution during earthquakes
    b: distribution over all time periods
    '''
    
    result = {} # this creates a dictionary
    
    result['cvm'] = stats.cramervonmises_2samp(a, b, method='auto')
    result['ks'] = stats.ks_2samp(a, b)
    result['median_all'] = np.median(b)
    result['median_eq'] = np.median(a)
    result['mean_all'] = np.mean(b)
    result['mean_eq'] = np.mean(a)
    result['mean_all_minus_mean_eq'] = np.mean(b)-np.mean(a)
    result['median_all_minus_median_eq'] = np.median(b)-np.median(a)
    
    return result

def plot_hist_rate(rate_at_all_times, rate_during_eq, ax1, ax2,title1, title2):
    
#     fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    plt.style.use('fivethirtyeight')
    
    # Cumulative histogram
    bins = calculate_bin_sizes(rate_during_eq)
    
    ax1.hist(rate_during_eq, bins, density = True, cumulative=True, histtype='step',
            label='Time periods with an earthquake',linewidth=1.5)
    ax1.hist(rate_at_all_times, bins, density = True, cumulative=True,histtype='step',
            label='All time periods',linewidth=1.5)
    yl = ax1.get_ylim()
    ax1.set_ylim((-0.1,1.4*yl[1]))
    ax1.legend()
    ax1.set_xlabel('Rate of surface loading (cm-we/month)', fontsize = 17)
    ax1.set_ylabel("Cumulative probability", fontsize = 17)
    ax1.set_title('A. Cumulative Distribution')
                 
    # Non-cumulative histogram

#     bins = np.linspace(-80,80,41)
    ax2.hist(rate_during_eq, bins, density = True, cumulative=False, histtype='step',
            label='Time periods with an earthquake',linewidth=1.5)
    ax2.hist(rate_at_all_times, bins, density = True, cumulative=False,histtype='step',
            label='All time periods',linewidth=1.5)
    yl = ax2.get_ylim()
    ax2.set_ylim(-0.01,1.4*yl[1])
    ax2.legend()
    ax2.set_xlabel('Rate of surface loading (cm-we/month)', fontsize = 17)
    ax2.set_ylabel("Probability", fontsize = 17)
    ax2.set_title('B. Probability Density')

def plot_rel_hist_rate(all_time_periods, earthquake_only, ax, title):

#     fig,ax = plt.subplots(figsize=(7,7))
    plt.style.use('fivethirtyeight')

    xmin=np.min(earthquake_only)
    xmax=np.max(earthquake_only)
    bins = calculate_bin_sizes(earthquake_only)
    
    LgE = np.histogram(earthquake_only, bins=bins, density = True)[0]
    L   = np.histogram(all_time_periods,bins=bins, density = True)[0]

    wid = np.mean(np.diff(bins))
    ax.bar(bins[:-1]+wid/2,LgE/L,width=wid)

    ax.plot([xmin,xmax],[1, 1],'--r')
    ax.text(-10, 1.5,'P(E|L)=P(E)',color='r',fontsize=20)
    ax.set_xlabel('Rate of surface loading (cm-we/month)',fontsize = 17)
    ax.set_ylabel('Relative conditional probability',fontsize = 17)
    ax.set_title(title, fontsize = 17)


def plot_same_map(eq_load1, eq_load2, bounds1, bounds2, label1, label2):

    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(color='white', edgecolor='black', figsize=(15,10))

    # first PC
    df_bigmass = bounds1
    gdf = gpd.GeoDataFrame(df_bigmass,
                       geometry=gpd.points_from_xy(df_bigmass.longitude, df_bigmass.latitude))
    gdf.plot(ax=ax, label=label1)

    # second pc
    df_bigmass = bounds2
    gdf = gpd.GeoDataFrame(df_bigmass,
                       geometry=gpd.points_from_xy(df_bigmass.longitude, df_bigmass.latitude))
    gdf.plot(ax=ax, label=label2)


    leg = ax.legend()
    ax.set_xlabel('Longitude', fontsize = 15)
    ax.set_ylabel("Latitude", fontsize = 15)
    plt.show()
    
def get_cond_probability(all_time_periods, earthquake_only, loads):
    
    cp,bins = calculate_bayes(earthquake_only,all_time_periods)
#     print(cp)
#     print(bins)


    cp_for_each_event = []
    
    for load in loads:
        
        this_bin = bins[0]
        i = 0
    # Remember that the values in 'bins' are the left edges of the histogram bars
        while this_bin < load:
#             print('%f <= %f'%(this_bin,load))
            if i == len(cp):
                break
            else:
                i = i + 1
                this_bin = bins[i]
#         print('Load %f belongs in the bin bounded on the left by the value %f'%(load,bins[i-1]))
        cp_for_each_event.append(cp[i-1])
        
    return np.array(cp_for_each_event)

def calculate_bin_sizes(some_data,method="Sturge"):
    xmin=np.min(some_data)
    xmax=np.max(some_data)
    rng = xmax-xmin
    xmin = xmin - rng/1e3
    xmax = xmax + rng/1e3
    if method=="Sturge":
        bins = np.linspace(xmin, xmax,
                       int(1 + 3.322*np.log(some_data.size)))
    return bins

def calculate_bayes(earthquake_only,all_time_periods):

    bins = calculate_bin_sizes(earthquake_only)

    LgE = np.histogram(earthquake_only, bins=bins, density = True)[0]
    L   = np.histogram(all_time_periods,bins=bins, density = True)[0]
    cp = LgE/L

    return cp, bins
