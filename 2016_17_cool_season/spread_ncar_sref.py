import matplotlib as mpl
#mpl.use('Agg')
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
############ Read in NCARENS data_ncar   ###########################################
###############################################################################

            
x = 0
q = 0
v = 0
i = 0   

         

links = []
for mem in range(1,11):
    links.append("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/ncarens%d" % mem + "_precip_12Zto12Z_interp.txt")

       
data_ncar = zeros((len(links),798,185))

         
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
                    data_ncar[c,x,0] = station_id
                    x = x + 1
                    
            if i == 1:     
                for t in range(3,commas):
                    lat = line.split(',')[t]
                    data_ncar[c,q,1] = lat
                    q = q + 1
            
            if i == 2:     
                for t in range(3,commas):
                    lon = line.split(',')[t]
                    data_ncar[c,v,2] = lon
                    v = v + 1

            if i != 0 and i != 1 and i != 2:
                for t in range(3,commas):   
                    precip = line.split(',')[t]
                    precip = float(precip)
                    data_ncar[c,y,i] = precip

                    y = y + 1
            
            i = i + 1

data_ncar[np.isnan(data_ncar)] = 9999
     
     
     
     

     
###############################################################################
############ Read in  SREF data   #############################################
###############################################################################

            
x = 0
q = 0
v = 0
i = 0   

       

links = []

for model in ['arw', 'nmb']:
    links.append('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_%s' % model + '_ctl_precip_12Z_to_12Z_interp.nc')
    for num in range(1,7):
        for typ in ['n', 'p']:
            links.append('/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_%s' % model + '_%s'% typ + '%d' % num + '_precip_12Z_to_12Z_interp.nc')


       
data_sref = zeros((len(links),798,185))

         
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
                    data_sref[c,x,0] = station_id
                    x = x + 1
                    
            if i == 1:     
                for t in range(3,commas):
                    lat = line.split(',')[t]
                    data_sref[c,q,1] = lat
                    q = q + 1
            
            if i == 2:     
                for t in range(3,commas):
                    lon = line.split(',')[t]
                    data_sref[c,v,2] = lon
                    v = v + 1

            if i != 0 and i != 1 and i != 2:
                for t in range(3,commas):   
                    precip = line.split(',')[t]
                    precip = float(precip)
                    data_sref[c,y,i] = precip

                    y = y + 1
            
            i = i + 1

data_sref[np.isnan(data_sref)] = 9999
          

#%%

###############################################################################
############ Determine mean SD as func of event size   ########################
###############################################################################

data_arw = np.copy(data_sref[:13,:,:])
data_nmmb = np.copy(data_sref[13:,:,:])

ens_mean = np.zeros((4,len(data_ncar[0,:,0]), len(data_ncar[0,0,:])))
ens_std = np.zeros((4,len(data_ncar[0,:,0]), len(data_ncar[0,0,:])))

#Copy station data over
ens_std[:,:,:3] = data_ncar[:4,:,:3]

### Calculate mean
ens_mean[0,:,:] = np.mean(data_ncar, 0)
ens_mean[1,:,:] = np.mean(data_sref, 0)
ens_mean[2,:,:] = np.mean(data_arw, 0)
ens_mean[3,:,:] = np.mean(data_nmmb, 0)

##Calculate SD
ens_std[0,:,3:] = np.std(data_ncar[:,:,3:], 0)
ens_std[1,:,3:] = np.std(data_sref[:,:,3:], 0)
ens_std[2,:,3:] = np.std(data_arw[:,:,3:], 0)
ens_std[3,:,3:] = np.std(data_nmmb[:,:,3:], 0)



### Steenburgh/Lewis regions (only Pacific (Far NW[1] and Sierrza Nevada[2]) and Intermountain (CO Rockies[3], Intermountain[4], Intermountain NW[5], Soutwest ID[6]))                   
regions = np.array([[37,40, -122,-118,0,0,0,0],###Sierra Nevada
                    [40,50, -125,-120, 42.97, -121.69,0,0], ##Far NW minus bottom right(>42.97, <-121.69)
                    [35.5,44, -108.7,-104,0,0,0,0], ## CO Rockies
                    [37,44.5, -114,-109.07, 39.32, -109, 43.6, -111.38], ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    [44,50, -117.2,-109, 45.28,-115.22,44.49, -110.84], ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    [43,45.5, -116.5,-113.5,44.46,-114.5,0,0]]) ### SW ID minus top right (< 44.46, <-114.5)


counter_pac = np.zeros((50,20))
sum_std_pac = np.zeros((50,20))
counter_pac[:,0] = np.linspace(2.54,2.54+(2.54*50),50)
sum_std_pac[:,0] = np.linspace(2.54,2.54+(2.54*50),50)

counter_int = np.zeros((50,20))
sum_std_int = np.zeros((50,20))
counter_int[:,0] = np.linspace(2.54,2.54+(2.54*50),50)
sum_std_int[:,0] = np.linspace(2.54,2.54+(2.54*50),50)



### Start loop to divide up by region and sum std
for ens in range(4): #Loop over all ensemble
    for stn in range(len(ens_mean[0,:,0])): #Loop over all stations
        ################### PACIFIC ###################            
        if ((regions[0,0] <= ens_mean[ens,stn,1] <= regions[0,1] and regions[0,2] <= ens_mean[ens,stn,2] <= regions[0,3]) or ###Sierra Nevada
    
            (regions[1,0] <= ens_mean[ens,stn,1] <= regions[1,1] and regions[1,2] <= ens_mean[ens,stn,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
            (ens_mean[ens,stn,1] >= regions[1,4] or ens_mean[ens,stn,2] <= regions[1,5])):
            
            for day in range(len(ens_mean[0,0,3:])):
                for thresh in range(len(counter_pac[:,0])):
                    if counter_pac[thresh,0]-1.27 < ens_mean[ens, stn, day] <= counter_pac[thresh,0]+1.27:
                        counter_pac[thresh,ens+1] = counter_pac[thresh,ens+1] + 1
                        sum_std_pac[thresh,ens+1] = sum_std_pac[thresh,ens+1] + ens_std[ens,stn,day]
            
    
            
                      
        ################  INTERMOUNTAIN #################                                        
        if ((regions[2,0] <= ens_mean[ens,stn,1] <= regions[2,1] and regions[2,2] <= ens_mean[ens,stn,2] <= regions[2,3]) or ## ensO Roenskies
    
            (regions[3,0] <= ens_mean[ens,stn,1] <= regions[3,1] and regions[3,2] <= ens_mean[ens,stn,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
            (ens_mean[ens,stn,1] >= regions[3,4] or ens_mean[ens,stn,2] <= regions[3,5]) and 
            (ens_mean[ens,stn,1] <= regions[3,6] or ens_mean[ens,stn,2] >= regions[3,7]) or
            
            (regions[4,0] <= ens_mean[ens,stn,1] <= regions[4,1] and regions[4,2] <= ens_mean[ens,stn,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
            (ens_mean[ens,stn,1] >= regions[4,4] or ens_mean[ens,stn,2] >= regions[4,5]) and 
            (ens_mean[ens,stn,1] >= regions[4,6] or ens_mean[ens,stn,2] <= regions[4,7]) or
                
            (regions[5,0] <= ens_mean[ens,stn,1] <= regions[5,1] and regions[5,2] <= ens_mean[ens,stn,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
            (ens_mean[ens,stn,1] <= regions[5,4] or ens_mean[ens,stn,2] <= regions[5,5])): 
            
            for day in range(len(ens_mean[0,0,3:])):
                for thresh in range(len(counter_int[:,0])):
                    if counter_int[thresh,0]-1.27 < ens_mean[ens, stn, day] <= counter_int[thresh,0]+1.27:
                        counter_int[thresh,ens+1] = counter_int[thresh,ens+1] + 1
                        sum_std_int[thresh,ens+1] = sum_std_int[thresh,ens+1] + ens_std[ens,stn,day]


###Divide std sum by counters
std_pac = np.divide(sum_std_pac[:,1:5], counter_pac[:,1:5])
std_int = np.divide(sum_std_int[:,1:5], counter_int[:,1:5])








#%%




##############################   Plot   #######################################
### Pacific ##########
props = dict(boxstyle='square', facecolor='white', alpha=1)
fig1=plt.figure(num=None, figsize=(11, 11), dpi=500, facecolor='w', edgecolor='k')
fig1.subplots_adjust(hspace=.15, bottom = 0.2)
ax1 = fig1.add_subplot(211)
plt.xlim([0,50])
plt.xticks(np.arange(1.27,80,2.54))

xx = np.copy(counter_pac[:,0])

w = 0
plt.grid(True)
line1 = ax1.plot(xx,std_pac[:,0])
line2 = ax1.plot(xx,std_pac[:,1])
line3 = ax1.plot(xx,std_pac[:,2])
line4 = ax1.plot(xx,std_pac[:,3])


x = np.linspace(0, 100, 100)
y = np.linspace(1,1,100)


plt.yticks(np.arange(0,31,5))
ax1.set_yticklabels(np.arange(0,31,5), fontsize = 16)
plt.ylim([0,30])
ax1.set_xticklabels(['1.3', ' ','6.4',' ','11.4',' ','16.5',' ','21.6',' ','26.7',
                     ' ','31.8',' ','36.8',' ','41.9',' ','47.0'], fontsize = 16)
plt.xlim([0,50])
#ax1.set_yticks([50,20,10,5,2,1,0.5,0.2,0.1,0.05])
#ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.2f'))

plt.setp(line1, color='blue', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line2, color='red', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line3, color='green', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line4, color='gold', linewidth=2.0,marker = "o", markeredgecolor = 'none')

blue = mlines.Line2D([], [], color='blue',
                           label='NCAR ENS', linewidth = 2,marker = "o", markeredgecolor = 'none')
red = mlines.Line2D([], [], color='red',
                           label='SREF', linewidth = 2,marker = "o", markeredgecolor = 'none')
green = mlines.Line2D([], [], color='green',
                           label='SREF ARW', linewidth = 2,marker = "o", markeredgecolor = 'none')
gold = mlines.Line2D([], [], color='gold',
                           label='SREF NMMB', linewidth = 2,marker = "o", markeredgecolor = 'none')
ax1.text(1, 26.2, '(a) Pacific Ranges', fontsize = 25, bbox = props)
plt.title('         ', fontsize = 22, y = 1.04)

plt.ylabel('Mean SD of Spread (mm)', fontsize = 18, labelpad = 10)












### Intermpountain ##########


ax2 = fig1.add_subplot(212)
plt.xlim([0,50])
plt.xticks(np.arange(1.27,80,2.54))


w = 1
plt.grid(True)
line1 = ax2.plot(xx,std_int[:,0])
line2 = ax2.plot(xx,std_int[:,1])
line3 = ax2.plot(xx,std_int[:,2])
line4 = ax2.plot(xx,std_int[:,3])

x = np.linspace(0, 100, 100)
y = np.linspace(1,1,100)


plt.yticks(np.arange(0,31,5))
ax2.set_yticklabels(np.arange(0,31,5), fontsize = 16)
plt.ylim([0,30])
ax2.set_xticklabels(['1.3', ' ','6.4',' ','11.4',' ','16.5',' ','21.6',' ','26.7',
                     ' ','31.8',' ','36.8',' ','41.9',' ','47.0'], fontsize = 16)
plt.xlim([0,50])
#ax1.set_yticks([50,20,10,5,2,1,0.5,0.2,0.1,0.05])
#ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
#ax1.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%1.2f'))

plt.setp(line1, color='blue', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line2, color='red', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line3, color='green', linewidth=2.0,marker = "o", markeredgecolor = 'none')
plt.setp(line4, color='gold', linewidth=2.0,marker = "o", markeredgecolor = 'none')

blue = mlines.Line2D([], [], color='blue',
                           label='NCAR ENS', linewidth = 2,marker = "o", markeredgecolor = 'none')
red = mlines.Line2D([], [], color='red',
                           label='SREF', linewidth = 2,marker = "o", markeredgecolor = 'none')
green = mlines.Line2D([], [], color='green',
                           label='SREF ARW', linewidth = 2,marker = "o", markeredgecolor = 'none')
gold = mlines.Line2D([], [], color='gold',
                           label='SREF NMMB', linewidth = 2,marker = "o", markeredgecolor = 'none')
#plt.legend(handles=[blue, green, red, cyan], loc = 'upper left',bbox_to_anchor=(0.03, 1), fontsize = 16)



        
        
plt.xlabel('Ensemble Mean Event Size Bin (mm)', fontsize = 18)
plt.ylabel('Mean SD of Spread (mm)', fontsize = 18, labelpad = 10)
ax2.text(1, 26.2, '(b) Interior Ranges', fontsize = 25, bbox = props)
plt.legend(handles=[ blue, red, green, gold], loc='upper center', bbox_to_anchor=(0.5, -0.2), 
             ncol=4,fontsize = 15)

##Sample size
#a = plt.axes([.21, .757, .2, .07], axisbg='white')
#plt.bar(x,sample[0,:,0],width = 1.6, color = 'k', edgecolor ='none',align='center')
#plt.xlim([0,50])
#
##plt.title('SREF NMMB', y = 1.05, fontsize = 13)
##plt.text(0.56, 2000, 'Interior\nRanges', fontsize = 12)
#plt.ylabel('# Obs.', fontsize = 11)
#plt.xlabel('Precip. Bin (mm)',fontsize = 11)
#a.set_yscale('log')
#plt.ylim([1,100000])
##plt.xticks()
#plt.xticks(np.arange(0,51,10), fontsize = 10)
##plt.yticks(np.arange(0,5001,600), fontsize = 10)
##a.set_yticklabels(['0', '600', '1200', '1800', '2400', '>3000'])
#plt.grid(True)
#
#
#a = plt.axes([.21, .382, .2, .07], axisbg='white')
#plt.bar(sample[0,:,7],sample[1,:,0],width = 1.6, color = 'k', edgecolor ='none',align='center')
#plt.xlim([0,50])
##plt.ylim([0,3000])
##plt.title('SREF NMMB', y = 1.05, fontsize = 13)
##plt.text(0.56, 2000, 'Interior\nRanges', fontsize = 12)
#plt.ylabel('# Obs.', fontsize = 11)
#plt.xlabel('Precip. Bin (mm)',fontsize = 11)
#a.set_yscale('log')
##plt.xticks()
#plt.xticks(np.arange(0,51,10), fontsize = 10)
##plt.yticks(np.arange(0,5001,600), fontsize = 10)
##a.set_yticklabels(['0', '600', '1200', '1800', '2400', '>3000'])
#plt.grid(True)

plt.savefig("../../../public_html/ms_thesis_plots/spread_ncar_sref.pdf")
plt.close(fig1)


























          


