import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from global_land_mask import globe
import scipy.stats as stats
import os
import geopandas as gpd

def plot_hist(all_time_periods, earthquake_only, ax1, ax2, title1, title2, method):
    
    # Cumulative histogram

    bins = calculate_bin_sizes(earthquake_only,method)
    ax1.hist(earthquake_only, bins, density = True, cumulative=True, histtype='step',
            label='Time periods with an earthquake',linewidth=1.5)
    ax1.hist(all_time_periods, bins, density = True, cumulative=True,histtype='step',
            label='All time periods',linewidth=1.5)
    yl = ax1.get_ylim()
    ax1.set_ylim((-0.01,1.4*yl[1]))
    xl = ax1.get_xlim()
    ax1.set_xlim(xl[0],xl[1]-4.4)
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
    xl = ax2.get_xlim()
    ax2.set_xlim(xl[0],xl[1]-4.4)
    ax2.legend()
    ax2.set_xlabel('Surface load (cm-we)', fontsize = 17)
    ax2.set_ylabel("Probability", fontsize = 17)
    ax2.set_title(title2)
    
def plot_bayes(all_time_periods, earthquake_only, ax, title,method):
    
    plt.style.use('fivethirtyeight')

    cp,bins = calculate_bayes(earthquake_only,all_time_periods,method)

    wid = np.mean(np.diff(bins))
    print(len(bins))
    print(len(cp))
          
    ax.bar(bins[:-1],cp,width=wid,align='edge')
    xl = ax.get_xlim()
    ax.set_xlim(xl[0],xl[1]-4.4)
    ax.plot([-80,80],[1, 1],'--r')
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
    
def get_cond_probability(all_time_periods, earthquake_only, loads, method):
    
    cp,bins = calculate_bayes(earthquake_only,all_time_periods,method)
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

def freedman_diaconis(data, returnas):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width. 
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively. 


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin. 
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale="raw", nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if returnas=="width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    return(result)

def calculate_bin_sizes(some_data,method):
    xmin=np.min(some_data)
    xmax=np.max(some_data)
    rng = xmax-xmin
    xmin = xmin - rng/1e3
    xmax = xmax + rng/1e3
    if method=="Sturge": # Uses Sturge's Rule
        bins = np.linspace(xmin, xmax,
                       int(1 + 3.322*np.log(some_data.size)))
    else: # Uses Freedman-Diaconis Rule
        bins = np.linspace(xmin, xmax,freedman_diaconis(data=some_data, returnas="bins"))
    return bins

def calculate_bayes(earthquake_only,all_time_periods,method):

    bins = calculate_bin_sizes(earthquake_only,method)

    LgE = np.histogram(earthquake_only, bins=bins, density = True)[0]
    L   = np.histogram(all_time_periods,bins=bins, density = True)[0]
    cp = LgE/L

    return cp, bins
