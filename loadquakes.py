import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from global_land_mask import globe
import scipy.stats as stats
from scipy.stats import cramervonmises_2samp
import os
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm

def plot_hist(all_time_periods, earthquake_only, ax1, ax2, title1, title2, method,
             label1='During earthquakes',
             label2='All times',
             label3='During earthquakes, $f(L|E)$',
             label4='All times, $f(L)$'):
    
    # Cumulative histogram
    
    fs = 16

    bins = calculate_bin_sizes(earthquake_only,method)
#     print(len(bins))
    ax1.hist(earthquake_only, bins, density = True, cumulative=True, histtype='step',
            label=label1,linewidth=1.5)
    ax1.hist(all_time_periods, bins, density = True, cumulative=True,histtype='step',
            label=label2,linewidth=1.5)
    yl = ax1.get_ylim()
    ax1.set_ylim((-0.01,1.4*yl[1]))
    xl = ax1.get_xlim()
    ax1.set_xlim(xl[0],xl[1]-10)
    ax1.legend(fontsize = fs)
    ax1.set_xlabel('Surface load (cm-we)', fontsize = fs)
    ax1.set_ylabel("Probability", fontsize = fs)
    ax1.set_title(title1, fontsize = fs, loc='left')
    
    # Non-cumulative histogram

    ax2.hist(earthquake_only, bins, density = True, cumulative=False, histtype='step',
            label=label3,linewidth=1.5)
    ax2.hist(all_time_periods, bins, density = True, cumulative=False,histtype='step',
            label=label4,linewidth=1.5)
    yl = ax2.get_ylim()
    ax2.set_ylim((-0.01,1.4*yl[1]))
    xl = ax2.get_xlim()
    ax2.set_xlim(xl[0],xl[1]-4.5)
    ax2.legend(fontsize = fs)
    ax2.set_xlabel('Surface load (cm-we)', fontsize = fs)
    ax2.set_ylabel("Probability", fontsize = fs)
    ax2.set_title(title2, fontsize = fs, loc='left')
    
def plot_bayes(all_time_periods, earthquake_only, ax, title, method):
    fs = 16

    cp,bins = calculate_bayes(earthquake_only,all_time_periods,method)

    wid = np.mean(np.diff(bins))
    print(len(bins))
    print(len(cp))
          
    ax.bar(bins[:-1],cp,width=wid,align='edge')
#     xl = ax.get_xlim()
#     ax.set_xlim(xl[0],xl[1])
#     ax.plot([-80,80],[1.74, 1.74],'--r')
    ax.set_xlabel('Surface load (cm-we)',fontsize = fs)
    ax.set_ylabel('Probability Normalized by $P(E)$',fontsize = fs)
    ax.set_title(title, fontsize = fs, loc='left')
    
def calc_stats(a,b):
    '''
    Calculate stats for the distributions a and b
    a: distribution during earthquakes
    b: distribution over all time periods
    '''
    
    result = {} # this creates a dictionary
    
    result['cvm'] = cramervonmises_2samp(a, b, method='auto')
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
    ax1.set_ylabel("Probability", fontsize = 17)
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

#     ax.plot([xmin,xmax],[1, 1],'--r')
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
        datmin, datmax = np.nanmin(data), np.nanmax(data)
        datrng = datmax - datmin
        
        print(f"datrng:{datrng}, bw:{bw}, result:{int((datrng / bw) + 1)}") 
        
        result = int((datrng / bw) + 1)
    return(result)

from scipy import stats
import numpy as np


def calculate_bin_sizes(some_data,method):
    xmin=np.nanmin(some_data)
    xmax=np.nanmax(some_data)
    rng = xmax-xmin
    xmin=xmin-rng/1e3
    xmax=xmax+rng/1e3
    
    if method=="Sturge": # Uses Sturge's Rule
        bins = np.linspace(xmin, xmax,
                       int(1 + 3.322*np.log(len(some_data))))
    else: # Uses Freedman-Diaconis Rule
        bins = np.linspace(xmin, xmax, 
                           freedman_diaconis(data=some_data, returnas="bins"))
    return bins

def calculate_bayes(earthquake_only,all_time_periods,method):

    bins = calculate_bin_sizes(earthquake_only,method)

    LgE = np.histogram(earthquake_only, bins=bins, density = True)[0]
    L   = np.histogram(all_time_periods,bins=bins, density = True)[0]
    cp = LgE/L

    return cp, bins

def set_of_figures_load(all_time, earthquake_only,bayes_title,method):

    fig,(ax1,ax2,ax3) = plt.subplots(3,1, figsize=(7,14))

    plt.style.use('fivethirtyeight')
    plot_hist(all_time, earthquake_only, ax1, ax2, 
              'a. Cumulative Distribution', 'b. Probability Density', method)

    plot_bayes(all_time, earthquake_only, ax3, bayes_title,
                         method)

    fig.tight_layout()
    plt.show()
    

def probability_map_cb(full_catalog,events,color,label,vmin,vmax,markersize_scale,circle_scale=1e-5):

    gdf=gpd.GeoDataFrame(events,
                           geometry=gpd.points_from_xy(events.sort_values('magnitude').longitude, 
                                                       events.sort_values('magnitude').latitude))
    world=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax=world.plot(color='white', edgecolor='black', figsize=(15,10))
    divider=make_axes_locatable(ax)
    cax=divider.append_axes("bottom", size="5%", pad=0.6)
    
    ax.legend(scatterpoints=1,
           loc='upper left',
           ncol=1,
           fontsize=12,
           labelspacing=5)
    
    # plotting all events in tiny grey dots
    ax.scatter(full_catalog.longitude,full_catalog.latitude,c="darkgrey",marker=".")
    
    for i in [5,6,7,8]:
        ax.scatter(0,
                   1000,
                   c="silver", 
                   s=circle_scale*i**(markersize_scale),
                   label=f'        M {i}.0',
                   edgecolor='k')
        
    cmap = cm.get_cmap('viridis', 12) # 12 discrete colors
    gdf.plot(ax=ax,cax=cax,alpha=0.5,column=color,cmap=cmap,legend=True,
             edgecolor='k',
             markersize=circle_scale*(events.magnitude)**markersize_scale,
             legend_kwds={'label': "Relative conditional probability of event",
                            'orientation': "horizontal"},
            vmax=vmax,
            vmin=vmin)
    
    gdf.plot(ax=ax,facecolor="None",
             edgecolor='k',
             markersize=circle_scale*(events.magnitude)**markersize_scale)

    ax.set_xlabel('Longitude', fontsize=15)
    ax.set_ylabel("Latitude", fontsize=15)
    ax.set_title(label)
    
    ax.legend(
       fontsize=12,
       bbox_to_anchor=(1.01, 0.99, 0.1, 0.1),
       labelspacing=6,
       frameon=False,
       borderpad=3)
   
    ax.set_ylim([-90,90])
    return ax
    
def load_map_cb(full_catalog,events,color,title,vmin,vmax,circle_scale=0.07,markersize_scale=1.5):

    gdf=gpd.GeoDataFrame(events,
                           geometry=gpd.points_from_xy(events.longitude, 
                                                       events.latitude))
    world=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax=world.plot(color='white', edgecolor='black', figsize=(15,10))
    divider=make_axes_locatable(ax)
    cax=divider.append_axes("bottom", size="5%", pad=0.6)
    
    # plotting all events in tiny grey dots
    ax.scatter(full_catalog.longitude,full_catalog.latitude,c="darkgrey",marker=".")
    
    for i in [5,6,7,8]:
        ax.scatter(0,
                   1000,
                   c="silver",
                   s=np.exp(i*markersize_scale)*(circle_scale),
                   label=f'  M {i}',
                   edgecolor='k', alpha=0.5)
        
    cmap = cm.get_cmap('Reds',100) 
    gdf.plot(ax=ax,cax=cax,alpha=0.5,column=color,cmap=cmap,legend=True,
             edgecolor='k',
             markersize=np.exp(events.magnitude*markersize_scale)*(circle_scale),
             legend_kwds={'label': "Surface mass load during event (cm-we)",
                          'orientation': "horizontal"},
            vmax=vmax,
            vmin=vmin)
    
    gdf.plot(ax=ax,facecolor="None",
         edgecolor='k',
         markersize=np.exp(events.magnitude*markersize_scale)*(circle_scale) )
    
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.set_title(title, fontsize=19)
    
    ax.legend(
       fontsize=12,
       bbox_to_anchor=(1.01, 0.9, 0.1, 0.1),
       labelspacing=4,
       frameon=False,
       borderpad=0.5)
    
    ax.set_ylim([-90,90])
    return ax
    
def depth_fig(ax,catalog,cumulative,label,title,sliq):

    bins_depth = calculate_bin_sizes(catalog.depth,'fd')

    ax.hist(sliq.depth,bins_depth,density = True,cumulative=cumulative, histtype='step',
            label='SLIQs',linewidth=1.5)
    ax.hist(catalog.depth, bins_depth,density = True, cumulative=cumulative,histtype='step',
            label=label,linewidth=1.5)
    yl = ax.get_ylim()
    ax.set_ylim((-0.01,1.4*yl[1]))
    xl = ax.get_xlim()
    ax.set_xlim(xl[0],xl[1])
    ax.legend()
    ax.set_xlabel('Depth (km)', fontsize = 17)
    ax.set_ylabel("Cumulative probability", fontsize = 17)
    ax.set_title(title, fontsize = 17)
   