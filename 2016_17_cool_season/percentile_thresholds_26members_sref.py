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



###############################################################################
############ Read in  12Z to 12Z data   #######################################
###############################################################################

            
x = 0
q = 0
v = 0
i = 0   

#%%         

links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/snotel_precip_2016_2017_qc.csv"]

for model in ['arw', 'nmb']:
    links.append('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_%s' % model + '_ctl_precip_12Z_to_12Z_interp.nc')
    for num in range(1,7):
        for typ in ['n', 'p']:
            links.append('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_%s' % model + '_%s'% typ + '%d' % num + '_precip_12Z_to_12Z_interp.nc')

#%%
       
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
sref_data = data[1,:,:]
inhouse_data = data[0,:,:]


inhouse_pac = zeros((182,124))
sref_pac = zeros((26,182,124))

inhouse_int = zeros((182,427))
sref_int = zeros((26,182,427))


for w in range(2):
    for mem in range(26):
        i = 0
        j = 0
        for x in range(len(sref_data[:,0])):
            for y in range(len(inhouse_data[:,0])):
                ################### PACIFIC ###################            
                if w == 0:
                        if ((regions[0,0] <= sref_data[x,1] <= regions[0,1] and regions[0,2] <= sref_data[x,2] <= regions[0,3]) or ###Sierra Nevada
                                
                            (regions[1,0] <= sref_data[x,1] <= regions[1,1] and regions[1,2] <= sref_data[x,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
                            (sref_data[x,1] >= regions[1,4] or sref_data[x,2] <= regions[1,5])):
                            if sref_data[x,0] == inhouse_data[y,0]:
                            
                                inhouse_pac[:,i] = data[0,y,3:185]
                                sref_pac[mem,:,i] = data[mem+1,x,3:185]
                                
                                i = i + 1

  
                ################  INTERMOUNTAIN #################                                        
                if w == 1:
                        if ((regions[2,0] <= sref_data[x,1] <= regions[2,1] and regions[2,2] <= sref_data[x,2] <= regions[2,3]) or ## CO Rockies
        
                            (regions[3,0] <= sref_data[x,1] <= regions[3,1] and regions[3,2] <= sref_data[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                            (sref_data[x,1] >= regions[3,4] or sref_data[x,2] <= regions[3,5]) and 
                            (sref_data[x,1] <= regions[3,6] or sref_data[x,2] >= regions[3,7]) or
                            
                            (regions[4,0] <= sref_data[x,1] <= regions[4,1] and regions[4,2] <= sref_data[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                            (sref_data[x,1] >= regions[4,4] or sref_data[x,2] >= regions[4,5]) and 
                            (sref_data[x,1] >= regions[4,6] or sref_data[x,2] <= regions[4,7]) or
                            
                            (regions[5,0] <= sref_data[x,1] <= regions[5,1] and regions[5,2] <= sref_data[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
                            (sref_data[x,1] <= regions[5,4] or sref_data[x,2] <= regions[5,5])):  
                            if sref_data[x,0] == inhouse_data[y,0]:
                                
                                inhouse_int[:,j] = data[0,y,3:185]
                                sref_int[mem,:,j] = data[mem+1,x,3:185]
                                
                                j = j + 1

                            
                           
#### Remove bad data
### Pacific #####
loc = np.where(inhouse_pac > 1000)

row = loc[0]
col = loc[1]

for mem in range(26):
    for i in range(len(row)):
        inhouse_pac[row[i],col[i]] = np.nan
        sref_pac[mem,row[i],col[i]] = np.nan
      
        

### Interior ####
loc = np.where(inhouse_int > 1000)

row = loc[0]
col = loc[1]

for mem in range(26):
    for i in range(len(row)):
        inhouse_int[row[i],col[i]] = np.nan
        sref_int[mem,row[i],col[i]] = np.nan




#Calculate percentiles
percent_pac = zeros((19,27))
percent_int = zeros((19,27))

for model in range(27):
    if model == 0:
        for p in range(5,96,5):
            i = p/5-1
            percent_pac[i,model] = np.nanpercentile(inhouse_pac[:,:],p)
            percent_int[i,model] = np.nanpercentile(inhouse_int[:,:],p)
            
    else:
        for p in range(5,96,5):
            i = p/5-1
            percent_pac[i,model] = np.nanpercentile(sref_pac[model-1,:,:],p)
            percent_int[i,model] = np.nanpercentile(sref_int[model-1,:,:],p)
#%%     
## Save SREF Percentile Data
#### Save percentile info
np.save('sref_percentile_pac', percent_pac[:,1:])   
np.save('sref_percentile_int', percent_int[:,1:])
np.save('snotel_percentile_int', percent_int[:,0])
np.save('snotel_percentile_pac', percent_pac[:,0])


test1 = np.load('sref_percentile_pac.npy')   
test2 = np.load('sref_percentile_int.npy')
test3 = np.load('snotel_percentile_int.npy')
       
### Load in NCAR ENS percentile data
percent_pac_ncar = np.load('ncarens_percentile_pac.npy')   
percent_int_ncar = np.load('ncarens_percentile_int.npy')
#%%    
#Plot for Pac and Int



#x = np.arange(5.08,50.800001,2.54)
x = np.arange(50,95.1,5)
linecolor = ['green', 'green', 'green', 'green','green', 'green', 'green', 'green','green', 'green', 'green', 'green','green',
             'gold', 'gold', 'gold', 'gold','gold', 'gold', 'gold', 'gold','gold', 'gold', 'gold', 'gold', 'gold',
             'blue', 'blue', 'blue', 'blue','blue', 'blue', 'blue', 'blue','blue', 'blue','k']
fig1=plt.figure(num=None, figsize=(12,8), dpi=500, facecolor='w', edgecolor='k')
fig1.subplots_adjust(hspace=.4, bottom = 0.2)


ax1 = fig1.add_subplot(121)
plt.gca().set_color_cycle(linecolor)
ax1.plot(x,percent_pac[9:,1:],linewidth = 2,marker = "o", markeredgecolor = 'none')
ax1.plot(x,percent_pac_ncar[9:,:],linewidth = 2,marker = "o", markeredgecolor = 'none')
ax1.plot(x,percent_pac[9:,0],linewidth = 2,marker = "o", markeredgecolor = 'none')
plt.grid(True)
black = mlines.Line2D([], [], color='k',
                           label='SNOTEL', linewidth = 2,marker = "o", markeredgecolor = 'none')
blue = mlines.Line2D([], [], color='blue',
                           label='NCAR ENS (10 members)', linewidth = 2,marker = "o", markeredgecolor = 'none')
green = mlines.Line2D([], [], color='green',
                           label='SREF ARW (13 members)', linewidth = 2,marker = "o", markeredgecolor = 'none')
gold = mlines.Line2D([], [], color='gold',
                           label='SREF NMMB (13 members)', linewidth = 2,marker = "o", markeredgecolor = 'none')

plt.ylabel('Absolute Event Threshold (mm)', fontsize = 17)
plt.xlabel('Percentile Event Threshold', fontsize = 17)
plt.xlim([50,95.1])
plt.ylim([0,50])
plt.xticks(np.arange(5,96,5))
ax1.set_xticklabels(np.arange(5,96,5), fontsize = 12)
ax1.tick_params(axis='y', labelsize=12)
props = dict(boxstyle='square', facecolor='white', alpha=1)
fig1.text(0.076, 0.92, '(a) Pacific Ranges', fontsize = 19, bbox = props)




ax1 = fig1.add_subplot(122, sharex = ax1)
plt.gca().set_color_cycle(linecolor)
ax1.plot(x,percent_int[9:,1:],linewidth = 2,marker = "o", markeredgecolor = 'none')
ax1.plot(x,percent_int_ncar[9:,:],linewidth = 2,marker = "o", markeredgecolor = 'none')
ax1.plot(x,percent_int[9:,0],linewidth = 2,marker = "o", markeredgecolor = 'none')
plt.grid(True)
plt.xlabel('Percentile Event Threshold', fontsize = 17)
plt.ylim([0,50])
plt.xticks(np.arange(5,96,5))
ax1.set_xticklabels(np.arange(5,96,5), fontsize = 12)
ax1.tick_params(axis='y', labelsize=12)
plt.xlim([50,95])
fig1.text(0.557,0.92, '(b) Interior Ranges', fontsize = 19, bbox = props)

plt.legend(handles=[black, blue, green, gold], loc='upper left', bbox_to_anchor=(-1.107, 0.923), 
           ncol=1, fontsize = 13)
plt.tight_layout()
plt.savefig("../../../public_html/sref_ncarens_forecast_thresholds.pdf")
















    
