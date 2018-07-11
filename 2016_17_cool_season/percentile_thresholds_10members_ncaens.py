#%%
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

import pygrib, os, sys, glob
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import time
from datetime import date, timedelta
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal
from matplotlib.mlab import bivariate_normal
from matplotlib import colors, ticker, cm
from datetime import date, timedelta
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colors, ticker, cm
import matplotlib.lines as mlines


wmont = [-117.0, 43.0, -108.5, 49.0]
utah = [-114.7, 36.7, -108.9, 42.5]
colorado = [-110.0, 36.0, -104.0, 41.9]
wasatch = [-113.4, 39.5, -110.7, 41.9]
cascades = [-125.3, 42.0, -116.5, 49.1]
west = [-125.3, 31.0, -102.5, 49.2]


region = 'west'



if region == 'wmont':
    latlon = wmont
    
if region == 'utah':
    latlon = utah
    
if region == 'colorado':
    latlon = colorado
    
if region == 'wasatch':
    latlon = wasatch
    
if region == 'cascades':
    latlon = cascades

if region == 'west':
    latlon = west
    

###############################################################################
############ Read in  12Z to 12Z data   #######################################
###############################################################################

            
x = 0
q = 0
v = 0
i = 0   

         

links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/snotel_precip_2016_2017_qc.csv"]
for mem in range(1,11):
    links.append("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/ncarens%d" % mem + "_precip_12Zto12Z_interp.txt")

#data = ['inhouse_data', 'ncar_data', 'nam4k_data', 'hrrr_data', 'nam12k_data']        
data = zeros((len(links),798,185))

         
for c in range(len(links)):
    x = 0
    q = 0
    v = 0
    i = 0     
    with open(links[c], "rt") as f:
        for line in f:
            commas = line.count(',') + 1
            year = line.split(',')[0]
            month = line.split(',')[1]
            day = line.split(',')[2]
            date = year + month + day
            y = 0
      

            if i == 0:     
                for t in range(3,commas):
                    station_id = line.split(',')[t]
                    data[c,x,0] = station_id
                    x = x + 1
                    
            if i == 1:     
                for t in range(3,commas):
                    lat = line.split(',')[t]
                    data[c,q,1] = lat
                    q = q + 1
            
            if i == 2:     
                for t in range(3,commas):
                    lon = line.split(',')[t]
                    data[c,v,2] = lon
                    v = v + 1

            if i != 0 and i != 1 and i != 2:
                for t in range(3,commas):   
                    precip = line.split(',')[t]
                    precip = float(precip)
                    data[c,y,i] = precip

                    y = y + 1
            
            i = i + 1

data[np.isnan(data)] = 9999




regions = np.array([[37,40, -122,-118,0,0,0,0],###Sierra Nevada
                    [40,50, -125,-120, 42.97, -121.69,0,0], ##Far NW minus bottom right(>42.97, <-121.69)
                    [35.5,44, -108.7,-104,0,0,0,0], ## CO Rockies
                    [37,44.5, -114,-109.07, 39.32, -109, 43.6, -111.38], ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    [44,50, -117.2,-109, 45.28,-115.22,44.49, -110.84], ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    [43,45.5, -116.5,-113.5,44.46,-114.5,0,0]]) ### SW ID minus top right (< 44.46, <-114.5)



#%%
############ Determine percentiles for entire interior and pacific
ncar_data = data[1,:,:]
inhouse_data = data[0,:,:]


inhouse_pac = zeros((182,124))
ncar_pac = zeros((10,182,124))

inhouse_int = zeros((182,427))
ncar_int = zeros((10,182,427))


for w in range(2):
    for mem in range(10):
        i = 0
        j = 0
        for x in range(len(ncar_data[:,0])):
            for y in range(len(inhouse_data[:,0])):
                ################### PACIFIC ###################            
                if w == 0:
                        if ((regions[0,0] <= ncar_data[x,1] <= regions[0,1] and regions[0,2] <= ncar_data[x,2] <= regions[0,3]) or ###Sierra Nevada
                                
                            (regions[1,0] <= ncar_data[x,1] <= regions[1,1] and regions[1,2] <= ncar_data[x,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
                            (ncar_data[x,1] >= regions[1,4] or ncar_data[x,2] <= regions[1,5])):
                            if ncar_data[x,0] == inhouse_data[y,0]:
                            
                                inhouse_pac[:,i] = data[0,y,3:185]
                                ncar_pac[mem,:,i] = data[mem+1,x,3:185]
                                
                                i = i + 1

  
                ################  INTERMOUNTAIN #################                                        
                if w == 1:
                        if ((regions[2,0] <= ncar_data[x,1] <= regions[2,1] and regions[2,2] <= ncar_data[x,2] <= regions[2,3]) or ## CO Rockies
        
                            (regions[3,0] <= ncar_data[x,1] <= regions[3,1] and regions[3,2] <= ncar_data[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                            (ncar_data[x,1] >= regions[3,4] or ncar_data[x,2] <= regions[3,5]) and 
                            (ncar_data[x,1] <= regions[3,6] or ncar_data[x,2] >= regions[3,7]) or
                            
                            (regions[4,0] <= ncar_data[x,1] <= regions[4,1] and regions[4,2] <= ncar_data[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                            (ncar_data[x,1] >= regions[4,4] or ncar_data[x,2] >= regions[4,5]) and 
                            (ncar_data[x,1] >= regions[4,6] or ncar_data[x,2] <= regions[4,7]) or
                            
                            (regions[5,0] <= ncar_data[x,1] <= regions[5,1] and regions[5,2] <= ncar_data[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
                            (ncar_data[x,1] <= regions[5,4] or ncar_data[x,2] <= regions[5,5])):  
                            if ncar_data[x,0] == inhouse_data[y,0]:
                                
                                inhouse_int[:,j] = data[0,y,3:185]
                                ncar_int[mem,:,j] = data[mem+1,x,3:185]
                                
                                j = j + 1

                            
#%%                            
#### Remove bad data

loc = np.where(inhouse_pac > 1000)

row = loc[0]
col = loc[1]
#%%


for mem in range(10):
    for i in range(len(row)):
        inhouse_pac[row[i],col[i]] = np.nan
        ncar_pac[mem,row[i],col[i]] = np.nan
        
        


loc = np.where(inhouse_int > 1000)

row = loc[0]
col = loc[1]
#%%


for mem in range(10):
    for i in range(len(row)):
        inhouse_int[row[i],col[i]] = np.nan
        ncar_int[mem,row[i],col[i]] = np.nan




#%%
#Calculate percentiles
percent_pac = zeros((19,11))
percent_int = zeros((19,11))

for model in range(11):
    
    if model == 0:
        for p in range(5,96,5):
            i = p/5-1
            percent_pac[i,model] = np.nanpercentile(inhouse_pac[:,:],p)
            percent_int[i,model] = np.nanpercentile(inhouse_int[:,:],p)
            
    else:
        for p in range(5,96,5):
            i = p/5-1
            percent_pac[i,model] = np.nanpercentile(ncar_pac[model-1,:,:],p)
            percent_int[i,model] = np.nanpercentile(ncar_int[model-1,:,:],p)
            
#%%
#### Save percentile info
np.save('ncarens_percentile_pac', percent_pac[:,1:])   
np.save('ncarens_percentile_int', percent_int[:,1:])
#%%    
#Plot for Pac and Int



#x = np.arange(5.08,50.800001,2.54)
x = np.arange(50,95.1,5)
linecolor = ['blue', 'blue', 'blue', 'blue','blue', 'blue', 'blue', 'blue','blue', 'blue','k']
fig1=plt.figure(num=None, figsize=(14,8), dpi=500, facecolor='w', edgecolor='k')
fig1.subplots_adjust(hspace=.4, bottom = 0.2)


ax1 = fig1.add_subplot(121)
plt.gca().set_color_cycle(linecolor)
ax1.plot(x,percent_pac[9:,1:],linewidth = 2,marker = "o", markeredgecolor = 'none')
ax1.plot(x,percent_pac[9:,0],linewidth = 2,marker = "o", markeredgecolor = 'none')
plt.grid(True)
black = mlines.Line2D([], [], color='k',
                           label='SNOTEL', linewidth = 2,marker = "o", markeredgecolor = 'none')
blue = mlines.Line2D([], [], color='blue',
                           label='NCAR ENS', linewidth = 2,marker = "o", markeredgecolor = 'none')

plt.ylabel('Absolute Threshold (mm)', fontsize = 17)
plt.xlabel('Percentile Threshold', fontsize = 17)
plt.title('Pacific Ranges', fontsize = 17)
plt.xlim([50,95.1])
plt.ylim([0,55])
plt.xticks(np.arange(5,96,5))
ax1.set_xticklabels(np.arange(5,96,5), fontsize = 12)
ax1.tick_params(axis='y', labelsize=12)




ax1 = fig1.add_subplot(122, sharex = ax1)
plt.gca().set_color_cycle(linecolor)
ax1.plot(x,percent_int[9:,1:],linewidth = 2,marker = "o", markeredgecolor = 'none')
ax1.plot(x,percent_int[9:,0],linewidth = 2,marker = "o", markeredgecolor = 'none')
plt.grid(True)
plt.title('Interior Ranges', fontsize = 17)
plt.xlabel('Percentile Threshold', fontsize = 17)
plt.ylim([0,25])
plt.xticks(np.arange(5,96,5))
ax1.set_xticklabels(np.arange(5,96,5), fontsize = 12)
ax1.tick_params(axis='y', labelsize=12)
plt.xlim([50,95])

plt.legend(handles=[black, blue], loc='upper left', bbox_to_anchor=(-1.2, 1), 
           ncol=1, fontsize = 13)
plt.savefig("../../../public_html/ncarens_forecast_thresholds.pdf")
















    
