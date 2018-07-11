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
import matplotlib as mpl
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
import matplotlib.ticker as mtick


wmont = [-117.0, 43.0, -108.5, 49.0]
utah = [-114.7, 36.7, -108.9, 42.5]
colorado = [-110.0, 36.0, -104.0, 41.9]
wasatch = [-113.4, 39.5, -110.7, 41.9]
cascades = [-125.3, 42.0, -116.5, 49.1]
west = [-125.3, 31.0, -102.5, 49.2]


region = sys.argv[1]



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
    

    





inhouse_data = zeros((798,186))
ncar_data = zeros((798,186))
nam4k_data = zeros((798,186))
nam12k_data = zeros((798,186))
hrrr_data = zeros((798,186))


###############################################################################
############ Read in  12Z to 12Z data   #######################################
###############################################################################
             
x = 0
q = 0
v = 0
i = 0   


links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/snotel_precip_2015_2016_qc.csv", 
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/pre_2016_17_cool_season/ncarens_precip_12Zto12Z_upperquart_prob_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/pre_2016_17_cool_season/ncarens_precip_12Zto12Z_upperdec_prob_interp.txt"]
#data = ['inhouse_data', 'ncar_data', 'nam4k_data', 'hrrr_data', 'nam12k_data']        
data = zeros((3,798,186))

         
for c in range(3):
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


inhouse_data = data[0,:,:] 
ncar_data75 = data[1,:,:] 
ncar_data90 = data[2,:,:] 


###############################################################################
############ Determine percentiles      #######################################
###############################################################################

percent  = np.array([75,90])
percentiles  = zeros((len(data[0,:,0]),3))
p_array = []
             


for y in range(len(data[0,:,0])):
            if data[0,y,0] != 0:
                #p_array is all precip days for one station.  Created to determine percentiles for each station
                p_array = data[0,y,3:185]

                p_array = np.delete(p_array, np.where(p_array < 2.54))
                p_array = np.delete(p_array, np.where(p_array > 1000))
                
                percentile75 = np.percentile(p_array,percent[0])
                percentile90 = np.percentile(p_array,percent[1])
            
                percentiles[y,0] = data[0,y,0]
                percentiles[y,1] = percentile75
                percentiles[y,2] = percentile90


###############################################################################
############ Determine on which days an upper quartile and decile event occured    
###############################################################################

inhouse_data75 = zeros((798,186))
inhouse_data90 = zeros((798,186))
inhouse_data75[:,0:3] = inhouse_data[:,0:3]
inhouse_data90[:,0:3] = inhouse_data[:,0:3]

for w in range(len(inhouse_data[:,0])):
    if inhouse_data[w,0] != 0:
        for i in range(3,186):
            if percentiles[w,1] <= inhouse_data[w,i] < 1000 :
                inhouse_data75[w,i] = 1
            elif percentiles[w,1] > inhouse_data[w,i]:
                inhouse_data75[w,i] = 0
            elif 1000 < inhouse_data[w,i]:
                inhouse_data75[w,i] = 9999
          
          
for w in range(len(inhouse_data[:,0])):
    if inhouse_data[w,0] != 0:
        for i in range(3,186):
            if percentiles[w,2] <= inhouse_data[w,i] < 1000 :
                inhouse_data90[w,i] = 1
            elif percentiles[w,2] > inhouse_data[w,i]:
                inhouse_data90[w,i] = 0
            elif 1000 < inhouse_data[w,i]:
                inhouse_data90[w,i] = 9999   
         
###############################################################################
############ Calc forecast freq as function of observed freq ##################   
###############################################################################
         
         
### First divide into regions
region = ['Pacific Northwest', 'Sierra Nevada','Blue Mountains, OR','Idaho/Western MT','NW Wyoming','Utah','Colorado' ,'Arizona/New Mexico']                                       
###Serrezze regions
'''
regions = np.array([[41.5,49.2, -123.0,-120.5],
                    [37.0,41.0, -121.0,-118.0], 
                    [43.7,46.2, -120.0,-116.8], 
                    [43.0,49.3, -116.8,-112.2], 
                    [41.8,47.0, -112.5,-105.5],
                    [37.2,41.8, -113.9,-109.2],
                    [35.6,41.5, -108.7,-104.5],
                    [32.5,35.5, -113.0,-107.0]])
'''                    


### Steenburgh/Lewis regions (only Pacific (Far NW[1] and Sierrza Nevada[2]) and Intermountain (CO Rockies[3], Intermountain[4], Intermountain NW[5], Soutwest ID[6]))                   
regions = np.array([[37,40, -122,-118,0,0,0,0],###Sierra Nevada
                    [40,50, -125,-120, 42.97, -121.69,0,0], ##Far NW minus bottom right(>42.97, <-121.69)
                    [35.5,44, -108.7,-104,0,0,0,0], ## CO Rockies
                    [37,44.5, -114,-109.07, 39.32, -109, 43.6, -111.38], ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    [44,50, -117.2,-109, 45.28,-115.22,44.49, -110.84], ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    [43,45.5, -116.5,-113.5,44.46,-114.5,0,0]]) ### SW ID minus top right (< 44.46, <-114.5)
'''
regions = np.array([[37,40, -122,-118,0,0,0,0],###Sierra Nevada
                    [40,50, -125,-120, 42.97, -121.69,0,0], ##Far NW minus bottom right(>42.97, <-121.69)
                    [35.5,42, -108.7,-104,0,0,0,0], ## CO Rockies
                    [37,44.5, -114,-109.07, 39.32, -109.54, 43.6, -111.38], ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    [44,50, -117.2,-109, 45.28,-115.22,44.49, -110.84], ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    [43,45.5, -116.5,-113.5,44.46,-114.5,0,0]]) ### SW ID minus top right (< 44.46, <-114.5)                 
'''                    
### Caluclate observed frequencies for each forcasted probability                    

xi75pac = np.array([0])
xi90pac = np.array([0])
xi75int = np.array([0])
xi90int = np.array([0])
        
freq = zeros((2,11,7))
ss = 0
tt = 0

pacificloc = zeros((700,2))
interloc = zeros((700,2))

for r in range(0,2):    
    freq[r,:,0] = np.arange(0,1.001,.1)
    for x in range(len(ncar_data75[:,0])):
        for w in range(len(inhouse_data75[:,0])):


################### PACIFIC ###################            
            if r == 0:
                if ((regions[0,0] <= inhouse_data75[w,1] <= regions[0,1] and regions[0,2] <= inhouse_data75[w,2] <= regions[0,3]) or ###Sierra Nevada
                    
                    (regions[1,0] <= inhouse_data75[w,1] <= regions[1,1] and regions[1,2] <= inhouse_data75[w,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
                    (inhouse_data75[w,1] >= regions[1,4] or inhouse_data75[w,2] <= regions[1,5])):
                    
                        
               
                    if inhouse_data75[w,0] == ncar_data75[x,0]:
                        pacificloc[ss,:] = inhouse_data75[w,1:3]
                        ss = ss + 1   
                        if inhouse_data75[w,0] != 0:
                            for i in range(3,186):
                                if 0 < inhouse_data[w,i] < 1000:
                                    for t in range(11):
                                        if ncar_data75[x,i] == t/10.:
                                            freq[r,t,1] = freq[r,t,1] + inhouse_data75[w,i]
                                            freq[r,t,2] = freq[r,t,2] + 1
                                            xi75pac = np.append(xi75pac, ncar_data75[x,i])
                                            
                                            
################  INTERMOUNTAIN #################                                        
            if r == 1:
                if ((regions[2,0] <= inhouse_data75[w,1] <= regions[2,1] and regions[2,2] <= inhouse_data75[w,2] <= regions[2,3]) or ## CO Rockies
                    
                    (regions[3,0] <= inhouse_data75[w,1] <= regions[3,1] and regions[3,2] <= inhouse_data75[w,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    (inhouse_data75[w,1] >= regions[3,4] or inhouse_data75[w,2] <= regions[3,5]) and 
                    (inhouse_data75[w,1] <= regions[3,6] or inhouse_data75[w,2] >= regions[3,7]) or
                    
                    (regions[4,0] <= inhouse_data75[w,1] <= regions[4,1] and regions[4,2] <= inhouse_data75[w,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    (inhouse_data75[w,1] >= regions[4,4] or inhouse_data75[w,2] >= regions[4,5]) and 
                    (inhouse_data75[w,1] >= regions[4,6] or inhouse_data75[w,2] <= regions[4,7]) or
                        
                    (regions[5,0] <= inhouse_data75[w,1] <= regions[5,1] and regions[5,2] <= inhouse_data75[w,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
                    (inhouse_data75[w,1] <= regions[5,4] or inhouse_data75[w,2] <= regions[5,5])): 
                     
                        
                        
                    if inhouse_data75[w,0] == ncar_data75[x,0]:
                        interloc[tt,:] = inhouse_data75[w,1:3]
                        tt = tt + 1
                        if inhouse_data75[w,0] != 0:
                            for i in range(3,186):
                                if 0 < inhouse_data[w,i] < 1000:
                                    for t in range(11):
                                        if ncar_data75[x,i] == t/10.:
                                            freq[r,t,1] = freq[r,t,1] + inhouse_data75[w,i]
                                            freq[r,t,2] = freq[r,t,2] + 1
                                            xi75int = np.append(xi75int, ncar_data75[x,i])

print ss
print tt



######################### Test plot to determine which SNOTEL sites fall into regions  ################                                           
fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(inhouse_data75[:,2], inhouse_data75[:,1])
xi, yi = map(pacificloc[:,1], pacificloc[:,0])
xii, yii = map(interloc[:,1], interloc[:,0])
levels = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 2.2]
cmap=plt.cm.BrBG
csAVG = map.scatter(x,y, marker='o',  c = 'green',s = 125, vmin = 0.4, vmax = 2.2)
csAVG = map.scatter(xi,yi, marker='o',  c = 'blue',s = 125, vmin = 0.4, vmax = 2.2)
csAVG2 = map.scatter(xii,yii,  marker='o', c = 'red', s = 125, vmin = 0.4, vmax = 2.2)
map.drawcoastlines() 
map.drawstates()
map.drawcountries()
plt.savefig("../plots/snotel_regions.pdf")    
plt.show()
                        
                        



                    
                     

###################################   Upper decile events ############################                                        
for r in range(0,2):    
    freq[r,:,0] = np.arange(0,1.001,.1)
    for x in range(len(ncar_data90[:,0])):
        for w in range(len(inhouse_data90[:,0])):


################### PACIFIC ###################            
            if r == 0:

                if ((regions[0,0] <= inhouse_data75[w,1] <= regions[0,1] and regions[0,2] <= inhouse_data75[w,2] <= regions[0,3]) or ###Sierra Nevada
                    
                    (regions[1,0] <= inhouse_data75[w,1] <= regions[1,1] and regions[1,2] <= inhouse_data75[w,2] <= regions[1,3]) and  ##Far NW minus bottom right(>42.97, <-121.69)
                    (inhouse_data75[w,1] >= regions[1,4] or inhouse_data75[w,2] <= regions[1,5])):             
                        
               
                    if inhouse_data90[w,0] == ncar_data90[x,0]:
                        if inhouse_data90[w,0] != 0:
                            for i in range(3,186):
                                if 0 < inhouse_data[w,i] < 1000:
                                    for t in range(11):
                                        if ncar_data90[x,i] == t/10.:
                                            freq[r,t,4] = freq[r,t,4] + inhouse_data90[w,i]
                                            freq[r,t,5] = freq[r,t,5] + 1
                                            xi90pac = np.append(xi90pac, ncar_data90[x,i])
                                            
                                            
################  INTERIOR #################                                        
            if r == 1:
                 if ((regions[2,0] <= inhouse_data75[w,1] <= regions[2,1] and regions[2,2] <= inhouse_data75[w,2] <= regions[2,3]) or ## CO Rockies
                    
                    (regions[3,0] <= inhouse_data75[w,1] <= regions[3,1] and regions[3,2] <= inhouse_data75[w,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    (inhouse_data75[w,1] >= regions[3,4] or inhouse_data75[w,2] <= regions[3,5]) and 
                    (inhouse_data75[w,1] <= regions[3,6] or inhouse_data75[w,2] >= regions[3,7]) or
                    
                    (regions[4,0] <= inhouse_data75[w,1] <= regions[4,1] and regions[4,2] <= inhouse_data75[w,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    (inhouse_data75[w,1] >= regions[4,4] or inhouse_data75[w,2] >= regions[4,5]) and 
                    (inhouse_data75[w,1] >= regions[4,6] or inhouse_data75[w,2] <= regions[4,7]) or
                        
                    (regions[5,0] <= inhouse_data75[w,1] <= regions[5,1] and regions[5,2] <= inhouse_data75[w,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
                    (inhouse_data75[w,1] <= regions[5,4] or inhouse_data75[w,2] <= regions[5,5])): 
                     
                        
                        
                    if inhouse_data90[w,0] == ncar_data90[x,0]:
                        if inhouse_data90[w,0] != 0:
                            for i in range(3,186):
                                if 0 < inhouse_data[w,i] < 1000:
                                    for t in range(11):
                                        if ncar_data90[x,i] == t/10.:
                                            freq[r,t,4] = freq[r,t,4] + inhouse_data90[w,i]
                                            freq[r,t,5] = freq[r,t,5] + 1
                                            xi90int = np.append(xi90int, ncar_data90[x,i])
                                
freq[:,:,3] = freq[:,:,1]/freq[:,:,2]          
freq[:,:,6] = freq[:,:,4]/freq[:,:,5]         





###############################################################################
############### Resampling to create consistancy bars #########################  
###############################################################################
numtime75 = zeros((2))
numtime90 = zeros((2))
numtime75[0] = len(xi75pac)
numtime90[0] = len(xi90pac)
numtime75[1] = len(xi75int)
numtime90[1] = len(xi90int)



#Initialize array to hold results rom all 1000 resample cycles
resamp_con75 = zeros((2,11,1001))
resamp_con75[:,:,0] = linspace(0,1,11)

for r in range(0,2):
    for i in range(1000):
        print i
        #Intilaize array to sroe data for resample cycle
        counter = zeros((11,3))
        counter[:,0] = linspace(0,1,11)

        #Determine forecast (resampled) and observed (uniform distribution)
        if r == 0:
            for75 = np.random.choice(xi75pac,len(xi75pac))
            ob75 = np.random.uniform(0, 1, len(xi75pac))
        if r == 1:
            for75 = np.random.choice(xi75int,len(xi75int))
            ob75 = np.random.uniform(0, 1, len(xi75int))
    
        #Loop over all forecast/observe pairs in one cycle
        for t in range(int(numtime75[r])):
            num = int(for75[t]*10)
            counter[num,1] = counter[num,1] + 1
        
            if ob75[t] < for75[t]:
                counter[num,2] = counter[num,2] + 1
    
        #Store observed relative frequencies from each resample cycle
        for j in range(11):
            resamp_con75[r,:,i+1] = counter[:,2]/counter[:,1]



###############################################################################
############### Resampling to determine uncertainty ###########################  
###############################################################################

#Initialize array to hold results rom all 1000 resample cycles
resamp_unc75= zeros((2,11,1001))
resamp_unc75[:,:,0] = linspace(0,1,11)

for r in range(0,2):
    for i in range(1000):
        print i
        #Intilaize array to sroe data for resample cycle
        counter = zeros((11,3))
        counter[:,0] = linspace(0,1,11)

        #Determine forecast (resampled) and observed (uniform distribution)
        if r ==0:
            for75 = np.random.choice(xi75pac,len(xi75pac))
        if r ==1:
            for75 = np.random.choice(xi75int,len(xi75int))
    
        #Loop over all forecast/observe pairs in one cycle
        for t in range(int(numtime75[r])):
            num = int(for75[t]*10)
            counter[num,1] = counter[num,1] + 1
        
        counter[:,2] = freq[r,:,1]
    
        #Store observed relative frequencies from each resample cycle
        for j in range(11):
            resamp_unc75[r,:,i+1] = counter[:,2]/counter[:,1]
        


###############################################################################
######################## Determine Percentiles ################################  
###############################################################################
pr_con75 = zeros((2,11,3))
pr_unc75 = zeros((2,11,3))

pr_con75[:,:,0] = linspace(0,1,11)
pr_unc75[:,:,0] = linspace(0,1,11)

for r in range(0,2):
    pr_con75[r,:,1] = np.percentile(resamp_con75[r,:,1:], 5, axis = 1)
    pr_con75[r,:,2] = np.percentile(resamp_con75[r,:,1:], 95, axis = 1)
    
    pr_unc75[r,:,1] = np.percentile(resamp_unc75[r,:,1:], 5, axis = 1)
    pr_unc75[r,:,2] = np.percentile(resamp_unc75[r,:,1:], 95, axis = 1)

#%%


###############################################################################
######################## For upper decile      ################################  
###############################################################################


###############################################################################
############### Resampling to create consistancy bars #########################  
###############################################################################


#Initialize array to hold results rom all 1000 resample cycles
resamp_con90 = zeros((2,11,1001))
resamp_con90[:,:,0] = linspace(0,1,11)

for r in range(0,2):
    for i in range(1000):
        print i
        #Intilaize array to sroe data for resample cycle
        counter = zeros((11,3))
        counter[:,0] = linspace(0,1,11)

        #Determine forecast (resampled) and observed (uniform distribution)
        if r == 0:
            for90 = np.random.choice(xi90pac,len(xi90pac))
            ob90 = np.random.uniform(0, 1, len(xi90pac))
        if r == 1:
            for90 = np.random.choice(xi90int,len(xi90int))
            ob90 = np.random.uniform(0, 1, len(xi90int))
    
        #Loop over all forecast/observe pairs in one cycle
        for t in range(int(numtime90[r])):
            num = int(for90[t]*10)
            counter[num,1] = counter[num,1] + 1
        
            if ob90[t] < for90[t]:
                counter[num,2] = counter[num,2] + 1
    
        #Store observed relative frequencies from each resample cycle
        for j in range(11):
            resamp_con90[r,:,i+1] = counter[:,2]/counter[:,1]



###############################################################################
############### Resampling to determine uncertainty ###########################  
###############################################################################

#Initialize array to hold results rom all 1000 resample cycles
resamp_unc90= zeros((2,11,1001))
resamp_unc90[:,:,0] = linspace(0,1,11)

for r in range(0,2):
    for i in range(1000):
        print i
        #Intilaize array to sroe data for resample cycle
        counter = zeros((11,3))
        counter[:,0] = linspace(0,1,11)

        #Determine forecast (resampled) and observed (uniform distribution)
        if r ==0:
            for90 = np.random.choice(xi90pac,len(xi90pac))
        if r ==1:
            for90 = np.random.choice(xi90int,len(xi90int))
    
        #Loop over all forecast/observe pairs in one cycle
        for t in range(int(numtime90[r])):
            num = int(for90[t]*10)
            counter[num,1] = counter[num,1] + 1
        
        counter[:,2] = freq[r,:,4]
    
        #Store observed relative frequencies from each resample cycle
        for j in range(11):
            resamp_unc90[r,:,i+1] = counter[:,2]/counter[:,1]
        


###############################################################################
######################## Determine Percentiles ################################  
###############################################################################
pr_con90 = zeros((2,11,3))
pr_unc90 = zeros((2,11,3))

pr_con90[:,:,0] = linspace(0,1,11)
pr_unc90[:,:,0] = linspace(0,1,11)

for r in range(0,2):
    pr_con90[r,:,1] = np.percentile(resamp_con90[r,:,1:], 5, axis = 1)
    pr_con90[r,:,2] = np.percentile(resamp_con90[r,:,1:], 95, axis = 1)
    
    pr_unc90[r,:,1] = np.percentile(resamp_unc90[r,:,1:], 5, axis = 1)
    pr_unc90[r,:,2] = np.percentile(resamp_unc90[r,:,1:], 95, axis = 1)








###############################################################################
############ BSS and BS   #####################################################   
###############################################################################   


####### rows are BSSquart//BSquart//BSSdec//BSdec
bss= zeros((2,11,10))


### bssf rows are BS, reliabilty, resolution, uncertatinty, BSS 
bssf = zeros((2,10,5))


for r in range(0,2):
    obquart = (sum(freq[r,:,1])/sum(freq[r,:,2]))
    print(sum(freq[r,:,1]))
    print(sum(freq[r,:,2]))
    print(sum(freq[r,:,4]))
    print(sum(freq[r,:,5]))
    obdec = (sum(freq[r,:,4])/sum(freq[r,:,5]))

    Bref75 = (obquart)*(1.-obquart)
    Bref90 = (obdec)*(1.-obdec)

    for i in range(len(freq[r,:,0])):
        ### BS
        bss[r,i,0] =((((freq[r,i,0]-0)**2)*(freq[r,i,2]-freq[r,i,1]))+(((freq[r,i,0]-1)**2)*(freq[r,i,1])))
        bss[r,i,1] =((((freq[r,i,0]-0)**2)*(freq[r,i,5]-freq[r,i,4]))+(((freq[r,i,0]-1)**2)*(freq[r,i,4])))
    
        ### Reliability
        bss[r,i,2] = (freq[r,i,2]*(freq[r,i,0]-freq[r,i,3])**2)
        bss[r,i,3] = (freq[r,i,5]*(freq[r,i,0]-freq[r,i,6])**2)
    
    
        ### Resolution
        bss[r,i,4] = (freq[r,i,2]*(freq[r,i,3]-obquart)**2)
        bss[r,i,5] = (freq[r,i,5]*(freq[r,i,6]-obdec)**2)
    
        ### Uncertainty is same as BSref (climatology)
    


    ##### Upper quartile
    bssf[r,0,0] = sum(bss[r,:,0])/sum(freq[r,:,2])
    bssf[r,1,0] = sum(bss[r,:,2])/sum(freq[r,:,2])
    bssf[r,2,0] = sum(bss[r,:,4])/sum(freq[r,:,2])
    bssf[r,3,0] = Bref75
    bssf[r,4,0] = 1-(bssf[r,0,0]/Bref75) 

    #### Upper decile 
    bssf[r,0,1] = sum(bss[r,:,1])/sum(freq[r,:,5])
    bssf[r,1,1] = sum(bss[r,:,3])/sum(freq[r,:,5])
    bssf[r,2,1] = sum(bss[r,:,5])/sum(freq[r,:,2])
    bssf[r,3,1] = Bref90
    bssf[r,4,1] = 1-(bssf[r,0,1]/Bref90)




#%%

        
                        
                                
###############################################################################
################################# Plots #######################################   
###############################################################################                                
region = ['Pacific', 'Interior']                                  
linecolor = ['blue', 'red', 'red']                            
fig=plt.figure(num=None, figsize=(18,12), dpi=500, facecolor='w', edgecolor='k')
no_res = np.full((21),.25)
no_skill = np.arange(.125,.625001,.025)
freq_fill = np.arange(0,1.0001,0.05)







################  Upper Quartile #############################################



ax1 = fig.add_subplot(121)
fig.subplots_adjust(bottom=0.4)
plt.gca().set_color_cycle(linecolor)
for r in range(0,2):
    a = ax1.plot(freq[r,:,0],freq[r,:,3], linewidth = 2, c = linecolor[r], marker = "o", markeredgecolor = 'none')
    ax1.errorbar(freq[r,:,0],freq[r,:,3], yerr= [abs(pr_unc75[r,:,1]-freq[r,:,3]), abs(pr_unc75[r,:,2]-freq[r,:,3])], c = linecolor[r])
    
ax1.errorbar(freq[1,:,0],freq[1,:,0], yerr= [abs(pr_con75[0,:,1]-freq[1,:,0]), abs(pr_con75[0,:,2]-freq[1,:,0])], c = 'b')   
ax1.errorbar(freq[1,:,0],freq[1,:,0], yerr= [abs(pr_con75[1,:,1]-freq[1,:,0]), abs(pr_con75[1,:,2]-freq[1,:,0])], c = 'r')     
c = ax1.plot(freq[0,:,0],freq[0,:,0], linewidth = 2, c = 'k', markeredgecolor = 'none')
d = ax1.plot(freq_fill,no_res, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
e = ax1.plot(freq_fill,no_skill, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')

plt.text(.6,.78,'Perfect Reliability',rotation = 47)
plt.text(.6,.27,'No Resolution (Climatology)',rotation = 0)
plt.text(.6,.48,'No Skill',rotation = 27)

ax1.fill_between(freq_fill,no_skill, 1, where=no_skill >= .24999,facecolor = 'grey',alpha=0.5)
ax1.fill_between(freq_fill,no_skill, 0, where=no_skill <= .25, facecolor = 'grey',alpha=0.5)

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(np.arange(0,1.0001,0.1))
plt.yticks(np.arange(0,1.0001,0.1))
plt.grid(True)




                
                
###### Table to show all stats for both models
                
#tab.table(cellText=stats,colWidths=cwid,rowLabels=rows,colLabels=columns,loc='center',fontsize=5)      

the_table = plt.table(cellText=[('%.3f' % bssf[0,0,0],'%.3f' % bssf[0,1,0],'%.3f' % bssf[0,2,0],'%.3f' % bssf[0,3,0],'%.3f' % bssf[0,4,0]),
                                ('%.3f' % bssf[1,0,0],'%.3f' % bssf[1,1,0],'%.3f' % bssf[1,2,0],'%.3f' % bssf[1,3,0],'%.3f' % bssf[1,4,0]),
                                ('%.3f' % bssf[0,0,1],'%.3f' % bssf[0,1,1],'%.3f' % bssf[0,2,1],'%.3f' % bssf[0,3,1],'%.3f' % bssf[0,4,1]),
                                ('%.3f' % bssf[1,0,1],'%.3f' % bssf[1,1,1],'%.3f' % bssf[1,2,1],'%.3f' % bssf[1,3,1],'%.3f' % bssf[1,4,1])],
          rowLabels=["Pacific\n(Upper Quart.)","Interior\n(Upper Quart.)","Pacific\n(Upper Dec.)","Interior\n(Upper Dec.)"],
          colLabels=["Brier Score","Reliability", "Resolution", "Uncertainty", "Brier Skill Score"],
          loc="center",
          cellLoc = "center",
          rowColours=['lightgrey','lightgrey','lightgrey','lightgrey'],
          colColours=['lightgrey','lightgrey','lightgrey','lightgrey','lightgrey'],
          bbox=[.64,-0.65,1.135,.47])
the_table.set_fontsize(12)
the_table.scale(1.05,1.3)
the_table.auto_set_font_size(False)
the_table.set_fontsize(12)
       
          

blue_line = mlines.Line2D([],[] , color='blue',
                           label='Pacific',  linewidth = 2,marker = "o", markeredgecolor = 'none')
red_line = mlines.Line2D([],[] , color='red',
                           label='Interior',  linewidth = 2,marker = "o", markeredgecolor = 'none')
plt.legend(handles=[ blue_line, red_line], loc = "lower right",prop={'size':10.5})
plt.title('Upper Quartile 24-Hour Precipitation Events', fontsize = 20)
plt.ylabel('Observed Relative Frequency', fontsize = 15)
plt.xlabel('Forecast Probability',fontsize = 15)

















################  UpperDecile #############################################


no_res = np.full((21),.1)
no_skill = np.arange(.05,.55001,.025)
freq_fill = np.arange(0,1.0001,0.05)


ax2 = fig.add_subplot(122)
fig.subplots_adjust(bottom=0.4)
plt.gca().set_color_cycle(linecolor)
for r in range(0,2):
    b = ax2.plot(freq[r,:,0],freq[r,:,6], linewidth = 2, c = linecolor[r],marker = "o", markeredgecolor = 'none')
    ax2.errorbar(freq[r,:,0],freq[r,:,6], yerr= [abs(pr_unc90[r,:,1]-freq[r,:,6]), abs(pr_unc90[r,:,2]-freq[r,:,6])], c = linecolor[r])
    
ax2.errorbar(freq[1,:,0],freq[1,:,0], yerr= [abs(pr_con90[0,:,1]-freq[1,:,0]), abs(pr_con90[0,:,2]-freq[1,:,0])], c = 'b')   
ax2.errorbar(freq[1,:,0],freq[1,:,0], yerr= [abs(pr_con90[1,:,1]-freq[1,:,0]), abs(pr_con90[1,:,2]-freq[1,:,0])], c = 'r')   
c = ax2.plot(freq[1,:,0],freq[1,:,0], linewidth = 2, c = 'k', markeredgecolor = 'none')
d = ax2.plot(freq_fill,no_res, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
e = ax2.plot(freq_fill,no_skill, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
plt.text(.6,.78,'Perfect Reliability',rotation = 47)
plt.text(.6,.11,'No Resolution (Climatology)',rotation = 0)
plt.text(.8,.51,'No Skill',rotation = 27)


                


ax2.fill_between(freq_fill,no_skill, 1, where=no_skill >= .0999999,facecolor = 'grey',alpha=0.5)
ax2.fill_between(freq_fill,no_skill, 0, where=no_skill <= .10001, facecolor = 'grey',alpha=0.5)

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(np.arange(0,1.0001,0.1))
plt.yticks(np.arange(0,1.0001,0.1))
plt.grid(True)

#blue_line = mlines.Line2D([],[] , color='blue',
#                           label='NCAR (Upper Quartile)',  linewidth = 2,marker = "o", markeredgecolor = 'none')
blue_line = mlines.Line2D([], [], color='blue',
                           label='Pacific',   linewidth = 2,marker = "o", markeredgecolor = 'none')
red_line = mlines.Line2D([], [], color='red',
                           label='Interior',   linewidth = 2,marker = "o", markeredgecolor = 'none')
#red_line = mlines.Line2D([], [], color='red',
#                           label='HRRR',  linewidth = 2,marker = "o", markeredgecolor = 'none')
#cyan_line = mlines.Line2D([], [], color='c',
#                           label='NAM-12km', linewidth = 2,marker = "o", markeredgecolor = 'none')
plt.legend(handles=[blue_line, red_line], loc = "lower right",prop={'size':10.5})
plt.title('Upper Decile 24-Hour Precipitation Events', fontsize = 20)
plt.ylabel('Observed Relative Frequency', fontsize = 15)
plt.xlabel('Forecast Probability',fontsize = 15)











################## Sample frequency bar graphs ################################

a = plt.axes([.15, .24, .1, .1], axisbg='white')
plt.bar(freq[0,:,0],freq[0,:,2],width = .08, color = 'blue', edgecolor ='none', align='center')
plt.xlim([-0.04,1.04])
plt.title('Pacific (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
plt.xticks(np.arange(0,1.0001,.2))
a.set_yscale('log')
plt.tick_params(axis='x',which='both', bottom='off')
#plt.yticks(np.arange(0e4,4.0001e4,.5e4))
#a.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

a = plt.axes([.15, .07, .1, .1], axisbg='white')
plt.bar(freq[1,:,0],freq[1,:,2],width = .08, color = 'red', edgecolor ='none', align='center')
plt.xlim([-0.04,1.04])
plt.title('Interior (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
plt.xticks(np.arange(0,1.0001,.2))
a.set_yscale('log')
plt.tick_params(axis='x',which='both', bottom='off')
#plt.yticks(np.arange(0e4,4.0001e4,.5e4))
#a.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))


a = plt.axes([.8, .24, .1, .1], axisbg='white')
plt.bar(freq[0,:,0],freq[0,:,5],width = .08, color = 'blue', edgecolor ='none',align='center')
plt.xlim([-0.04,1.04])
plt.title('Pacific (Upper Dec.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
plt.xticks()
a.set_yscale('log')
plt.xticks(np.arange(0,1.0001,.2))
plt.tick_params(axis='x',which='both', bottom='off')







a = plt.axes([.8, .07, .1, .1], axisbg='white')
plt.bar(freq[1,:,0],freq[1,:,5],width = .08, color = 'red', edgecolor ='none',align='center')
plt.xlim([-0.04,1.04])
plt.title('Interior (Upper Dec.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
plt.xticks()
a.set_yscale('log')
plt.xticks(np.arange(0,1.0001,.2))
plt.tick_params(axis='x',which='both', bottom='off')








plt.savefig("../../public_html/reliability_diagram_regional_interp_bootstrap.pdf")                               
       
#%%      
          
          
          
          
          
          
          



