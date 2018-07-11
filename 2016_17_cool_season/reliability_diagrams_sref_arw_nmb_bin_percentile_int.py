
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
links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/snotel_precip_2016_2017_qc.csv", 
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_arw_precip_12Zto12Z_upper85_prob_interp_percentiles_int.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_arw_precip_12Zto12Z_upper95_prob_interp_percentiles_int.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_nmb_precip_12Zto12Z_upper85_prob_interp_percentiles_int.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_nmb_precip_12Zto12Z_upper95_prob_interp_percentiles_int.txt"]

#data = ['inhouse_data', 'ncar_data', 'nam4k_data', 'hrrr_data', 'nam12k_data']        
data = zeros((len(links),798,186))

         
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


inhouse_data = data[0,:,:] 
ncar_data75 = data[1,:,:] 
ncar_data90 = data[2,:,:]
sref_data75 = data[3,:,:] 
sref_data90 = data[4,:,:]



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

inhouse_data75 = zeros((798,185))
inhouse_data90 = zeros((798,185))
inhouse_data75[:,0:3] = inhouse_data[:,0:3]
inhouse_data90[:,0:3] = inhouse_data[:,0:3]


##Calculated percentiles

###############################################################################     
###################  Read in percentile information  ##########################
###############################################################################

ncar_percent_pac = np.load('ncarens_percentile_pac.npy')   
ncar_percent_int = np.load('ncarens_percentile_int.npy')

sref_percent_pac = np.load('sref_percentile_pac.npy') 
sref_percent_int = np.load('sref_percentile_int.npy')  

snotel_percent_pac = np.load('snotel_percentile_pac.npy') 
snotel_percent_int = np.load('snotel_percentile_int.npy') 
    
#%%
for w in range(len(inhouse_data[:,0])):
    if inhouse_data[w,0] != 0:
        for i in range(3,185):
            if snotel_percent_int[16] <= inhouse_data[w,i] < 1000 :
                inhouse_data75[w,i] = 1
            elif snotel_percent_int[16] > inhouse_data[w,i]:
                inhouse_data75[w,i] = 0
            elif 1000 < inhouse_data[w,i]:
                inhouse_data75[w,i] = 9999
          
          
for w in range(len(inhouse_data[:,0])):
    if inhouse_data[w,0] != 0:
        for i in range(3,185):
            if snotel_percent_int[18] <= inhouse_data[w,i] < 1000 :
                inhouse_data90[w,i] = 1
            elif snotel_percent_int[18] > inhouse_data[w,i]:
                inhouse_data90[w,i] = 0
            elif 1000 < inhouse_data[w,i]:
                inhouse_data90[w,i] = 9999   
         
###############################################################################
############ Calc forecast freq as function of observed freq ##################   
###############################################################################   

xi75ncar = np.array([0])
xi90ncar = np.array([0])


     
freq = zeros((14,20))
freq[:,0] = np.arange(0,1.001,0.07692307692)
freq_sref = zeros((14,20))
freq_sref[:,0] = np.arange(0,1.001,0.07692307692)




regions = np.array([[37,40, -122,-118,0,0,0,0],###Sierra Nevada
                    [40,50, -125,-120, 42.97, -121.69,0,0], ##Far NW minus bottom right(>42.97, <-121.69)
                    [35.5,44, -108.7,-104,0,0,0,0], ## CO Rockies
                    [37,44.5, -114,-109.07, 39.32, -109, 43.6, -111.38], ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
                    [44,50, -117.2,-109, 45.28,-115.22,44.49, -110.84], ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
                    [43,45.5, -116.5,-113.5,44.46,-114.5,0,0]]) ### SW ID minus top right (< 44.46, <-114.5)

#%%
      
for w in range(len(inhouse_data75[:,0])):  # NCAR 75
    for x in range(len(ncar_data75[:,0])):
                ################  INTERMOUNTAIN #################                                        
        if ((regions[2,0] <= ncar_data75[x,1] <= regions[2,1] and regions[2,2] <= ncar_data75[x,2] <= regions[2,3]) or ## CO Rockies
            (regions[3,0] <= ncar_data75[x,1] <= regions[3,1] and regions[3,2] <= ncar_data75[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
            (ncar_data75[x,1] >= regions[3,4] or ncar_data75[x,2] <= regions[3,5]) and 
            (ncar_data75[x,1] <= regions[3,6] or ncar_data75[x,2] >= regions[3,7]) or
            (regions[4,0] <= ncar_data75[x,1] <= regions[4,1] and regions[4,2] <= ncar_data75[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
            (ncar_data75[x,1] >= regions[4,4] or ncar_data75[x,2] >= regions[4,5]) and 
            (ncar_data75[x,1] >= regions[4,6] or ncar_data75[x,2] <= regions[4,7]) or
            (regions[5,0] <= ncar_data75[x,1] <= regions[5,1] and regions[5,2] <= ncar_data75[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
            (ncar_data75[x,1] <= regions[5,4] or ncar_data75[x,2] <= regions[5,5])):  
            if ncar_data75[x,0] == inhouse_data75[w,0]:     
                if inhouse_data75[w,0] != 0:
                    for i in range(3,185):
                        if inhouse_data[w,i] < 1000:
                            for t in range(14):
                                g = round(t/13.,3)
                                if ncar_data75[x,i] == g:
                                    freq[t,1] = freq[t,1] + inhouse_data75[w,i]
                                    freq[t,2] = freq[t,2] + 1
#%%                                

for w in range(len(inhouse_data90[:,0])):  # NCAR 90
    for x in range(len(ncar_data90[:,0])):
                ################  INTERMOUNTAIN #################                                        
        if ((regions[2,0] <= ncar_data90[x,1] <= regions[2,1] and regions[2,2] <= ncar_data90[x,2] <= regions[2,3]) or ## CO Rockies
            (regions[3,0] <= ncar_data90[x,1] <= regions[3,1] and regions[3,2] <= ncar_data90[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
            (ncar_data90[x,1] >= regions[3,4] or ncar_data90[x,2] <= regions[3,5]) and 
            (ncar_data90[x,1] <= regions[3,6] or ncar_data90[x,2] >= regions[3,7]) or
            (regions[4,0] <= ncar_data90[x,1] <= regions[4,1] and regions[4,2] <= ncar_data90[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
            (ncar_data90[x,1] >= regions[4,4] or ncar_data90[x,2] >= regions[4,5]) and 
            (ncar_data90[x,1] >= regions[4,6] or ncar_data90[x,2] <= regions[4,7]) or
            (regions[5,0] <= ncar_data90[x,1] <= regions[5,1] and regions[5,2] <= ncar_data90[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
            (ncar_data90[x,1] <= regions[5,4] or ncar_data90[x,2] <= regions[5,5])): 
            if inhouse_data90[w,0] == ncar_data90[x,0]:
                if inhouse_data90[w,0] != 0:
                    for i in range(3,185):
                        if inhouse_data[w,i] < 1000:
                            for t in range(14):
                                g = round(t/13.,3)
                                if ncar_data90[x,i] == g:
                                    freq[t,4] = freq[t,4] + inhouse_data90[w,i]
                                    freq[t,5] = freq[t,5] + 1
                                

for w in range(len(inhouse_data75[:,0])):  # sref 75
    for x in range(len(sref_data75[:,0])):
                ################  INTERMOUNTAIN #################                                        
        if ((regions[2,0] <= sref_data75[x,1] <= regions[2,1] and regions[2,2] <= sref_data75[x,2] <= regions[2,3]) or ## CO Rockies
            (regions[3,0] <= sref_data75[x,1] <= regions[3,1] and regions[3,2] <= sref_data75[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
            (sref_data75[x,1] >= regions[3,4] or sref_data75[x,2] <= regions[3,5]) and 
            (sref_data75[x,1] <= regions[3,6] or sref_data75[x,2] >= regions[3,7]) or
            (regions[4,0] <= sref_data75[x,1] <= regions[4,1] and regions[4,2] <= sref_data75[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
            (sref_data75[x,1] >= regions[4,4] or sref_data75[x,2] >= regions[4,5]) and 
            (sref_data75[x,1] >= regions[4,6] or sref_data75[x,2] <= regions[4,7]) or
            (regions[5,0] <= sref_data75[x,1] <= regions[5,1] and regions[5,2] <= sref_data75[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
            (sref_data75[x,1] <= regions[5,4] or sref_data75[x,2] <= regions[5,5])): 
            if inhouse_data75[w,0] == sref_data75[x,0]:
                if inhouse_data75[w,0] != 0:
                    for i in range(3,185):
                        if inhouse_data[w,i] < 1000:
                            for t in range(14):
                                g = round(t/13.,3)
                                if sref_data75[x,i] == g:
                                    freq_sref[t,1] = freq_sref[t,1] + inhouse_data75[w,i]
                                    freq_sref[t,2] = freq_sref[t,2] + 1
                              
                                

for w in range(len(inhouse_data90[:,0])):  # sref 90
    for x in range(len(sref_data90[:,0])):
                ################  INTERMOUNTAIN #################                                        
        if ((regions[2,0] <= sref_data90[x,1] <= regions[2,1] and regions[2,2] <= sref_data90[x,2] <= regions[2,3]) or ## CO Rockies
            (regions[3,0] <= sref_data90[x,1] <= regions[3,1] and regions[3,2] <= sref_data90[x,2] <= regions[3,3]) and  ### Intermounaint mimus bottom right and top left (> 39.32, < -109.54, <43.6, > -111.38)
            (sref_data90[x,1] >= regions[3,4] or sref_data90[x,2] <= regions[3,5]) and 
            (sref_data90[x,1] <= regions[3,6] or sref_data90[x,2] >= regions[3,7]) or
            (regions[4,0] <= sref_data90[x,1] <= regions[4,1] and regions[4,2] <= sref_data90[x,2] <= regions[4,3]) and  ### Intermountain NW minus bottom left and bottom right ( > 45.28, > -115.22, > 44.49, < -110.84)
            (sref_data90[x,1] >= regions[4,4] or sref_data90[x,2] >= regions[4,5]) and 
            (sref_data90[x,1] >= regions[4,6] or sref_data90[x,2] <= regions[4,7]) or
            (regions[5,0] <= sref_data90[x,1] <= regions[5,1] and regions[5,2] <= sref_data90[x,2] <= regions[5,3]) and  ### SW ID minus top right (< 44.46, <-114.5)
            (sref_data90[x,1] <= regions[5,4] or sref_data90[x,2] <= regions[5,5])): 
            if inhouse_data90[w,0] == sref_data90[x,0]:
                if inhouse_data90[w,0] != 0:
                    for i in range(3,185):
                        if inhouse_data[w,i] < 1000:
                            for t in range(14):
                                g = round(t/13.,3)
                                if sref_data90[x,i] == g:
                                    freq_sref[t,4] = freq_sref[t,4] + inhouse_data90[w,i]
                                    freq_sref[t,5] = freq_sref[t,5] + 1
                            
                                
                                
                                
                                

                                
                                
                                
                                
freq[:,3] = freq[:,1]/freq[:,2]      # NCAR 75     
freq[:,6] = freq[:,4]/freq[:,5]      # NCAR 90    
freq_sref[:,3] = freq_sref[:,1]/freq_sref[:,2]      # sref 75     
freq_sref[:,6] = freq_sref[:,4]/freq_sref[:,5]      # sref 90

##### Bin Freq SREF into 0-0.5, 0.5-1.5, 1.5-2.5.......
freq_sref_bin = zeros((11,20))
intv = 0
for j in range(11):
    intvl = intv -0.05
    intvh = intv + 0.05
    ## Upper Quartile
    f = np.array([])
    for i in range(len(freq_sref[:,0])):
        if intvl <= freq_sref[i,0] < intvh:
            for t in range(int(freq_sref[i,2])):
                f = np.append(f,freq_sref[i,0])
            freq_sref_bin[j,1] = freq_sref_bin[j,1] + freq_sref[i,1]
            freq_sref_bin[j,2] = freq_sref_bin[j,2] + freq_sref[i,2]
        freq_sref_bin[j,0] = np.mean(f)
        
    ### Upper Decile
    f = np.array([])
    for i in range(len(freq_sref[:,0])):
        if intvl <= freq_sref[i,0] < intvh:
            for t in range(int(freq_sref[i,5])):
                f = np.append(f,freq_sref[i,0])
            freq_sref_bin[j,5] = freq_sref_bin[j,5] + freq_sref[i,4]
            freq_sref_bin[j,6] = freq_sref_bin[j,6] + freq_sref[i,5]
        freq_sref_bin[j,4] = np.mean(f)
    
    intv = intv + 0.1

freq_sref_bin[:,3] = freq_sref_bin[:,1]/freq_sref_bin[:,2]      # sref 75     
freq_sref_bin[:,7] = freq_sref_bin[:,5]/freq_sref_bin[:,6]      # sref 90            
                
        
##### Bin Freq  into 0-0.5, 0.5-1.5, 1.5-2.5.......
freq_bin = zeros((11,20))
intv = 0
for j in range(11):
    intvl = intv -0.05
    intvh = intv + 0.05
    ## Upper Quartile
    f = np.array([])
    for i in range(len(freq[:,0])):
        if intvl <= freq[i,0] < intvh:
            for t in range(int(freq[i,2])):
                f = np.append(f,freq[i,0])
            freq_bin[j,1] = freq_bin[j,1] + freq[i,1]
            freq_bin[j,2] = freq_bin[j,2] + freq[i,2]
        freq_bin[j,0] = np.mean(f)
        
    ### Upper Decile
    f = np.array([])
    for i in range(len(freq[:,0])):
        if intvl <= freq[i,0] < intvh:
            for t in range(int(freq[i,5])):
                f = np.append(f,freq[i,0])
            freq_bin[j,5] = freq_bin[j,5] + freq[i,4]
            freq_bin[j,6] = freq_bin[j,6] + freq[i,5]
        freq_bin[j,4] = np.mean(f)
    
    intv = intv + 0.1

freq_bin[:,3] = freq_bin[:,1]/freq_bin[:,2]      # sref 75     
freq_bin[:,7] = freq_bin[:,5]/freq_bin[:,6]      # sref 90            
                
        






### Create forecast array
xi75sref = np.array([0])
xi90sref = np.array([0])

for j in range(11):
    for i in range(int(freq_sref_bin[j,2])):
        xi75sref = np.append(xi75sref, freq_sref_bin[j,0])
        
for j in range(11):
    for i in range(int(freq_sref_bin[j,6])):
        xi90sref = np.append(xi90sref, freq_sref_bin[j,4])


### Create forecast array
xi75ncar = np.array([0])
xi90ncar = np.array([0])

for j in range(11):
    for i in range(int(freq_bin[j,2])):
        xi75ncar = np.append(xi75ncar, freq_bin[j,0])
        
for j in range(11):
    for i in range(int(freq_bin[j,6])):
        xi90ncar = np.append(xi90ncar, freq_bin[j,4])



###############################################################################
############### Resampling to create consistancy bars #########################  
###############################################################################
numtime75 = zeros((2))
numtime90 = zeros((2))
numtime75[0] = len(xi75ncar)
numtime90[0] = len(xi90ncar)
numtime75[1] = len(xi75sref)
numtime90[1] = len(xi90sref)



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
            for75 = np.random.choice(xi75ncar,len(xi75ncar))
            ob75 = np.random.uniform(0, 1, len(xi75ncar))
        if r == 1:
            for75 = np.random.choice(xi75sref,len(xi75sref))
            ob75 = np.random.uniform(0, 1, len(xi75sref))
    
        #Loop over all forecast/observe pairs in one cycle
        for t in range(int(numtime75[r])):
            num = int(round(for75[t],1)*10)
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
            for75 = np.random.choice(xi75ncar,len(xi75ncar))
        if r ==1:
            for75 = np.random.choice(xi75sref,len(xi75sref))
    
        #Loop over all forecast/observe pairs in one cycle
        for t in range(int(numtime75[r])):
            num = int(round(for75[t],1)*10)
            counter[num,1] = counter[num,1] + 1
        
        if r == 0:
            counter[:,2] = freq_bin[:,1]
        if r == 1:
            counter[:,2] = freq_sref_bin[:,1]

    
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
            for90 = np.random.choice(xi90ncar,len(xi90ncar))
            ob90 = np.random.uniform(0, 1, len(xi90ncar))
        if r == 1:
            for90 = np.random.choice(xi90sref,len(xi90sref))
            ob90 = np.random.uniform(0, 1, len(xi90sref))
    
        #Loop over all forecast/observe pairs in one cycle
        for t in range(int(numtime90[r])):
            num = int(round(for90[t],1)*10)
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
            for90 = np.random.choice(xi90ncar,len(xi90ncar))
        if r ==1:
            for90 = np.random.choice(xi90sref,len(xi90sref))
    
        #Loop over all forecast/observe pairs in one cycle
        for t in range(int(numtime90[r])):
            num = int(round(for90[t],1)*10)
            counter[num,1] = counter[num,1] + 1
            
        if r == 0:
            counter[:,2] = freq_bin[:,5]
        if r == 1:
            counter[:,2] = freq_sref_bin[:,5]
    
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






#%%






#%%






###############################################################################
############ BSS and BS   #####################################################   
###############################################################################   


#####  For srefMWF #######
 
####### rows are BSSquart//BSquart//BSSdsref//BSdsref
bss= zeros((11,10))


### bssf rows are BS, reliabilty, resolution, uncertatinty, BSS 
bssf = zeros((10,5))
obquart = (sum(freq_bin[:,1])/sum(freq_bin[:,2]))
obdsref = (sum(freq_bin[:,5])/sum(freq_bin[:,6]))

Bref75 = (obquart)*(1-obquart)

Bref90 = (obdsref)*(1-obdsref)

for i in range(len(freq_bin[:,0])):
    ### BS
    bss[i,0] =((((freq_bin[i,0]-0)**2)*(freq_bin[i,2]-freq_bin[i,1]))+(((freq_bin[i,0]-1)**2)*(freq_bin[i,1])))
    bss[i,1] =((((freq_bin[i,4]-0)**2)*(freq_bin[i,6]-freq_bin[i,5]))+(((freq_bin[i,4]-1)**2)*(freq_bin[i,5])))
    
    ### Reliability
    bss[i,2] = (freq_bin[i,2]*(freq_bin[i,0]-freq_bin[i,3])**2)
    bss[i,3] = (freq_bin[i,6]*(freq_bin[i,4]-freq_bin[i,7])**2)
    
    
    ### Resolution
    bss[i,4] = (freq_bin[i,2]*(freq_bin[i,3]-obquart)**2)
    bss[i,5] = (freq_bin[i,6]*(freq_bin[i,7]-obdsref)**2)
    
    ### Uncertainty is same as BSref (climatology)



##### Upper quartile
bssf[0,0] = sum(bss[:,0])/sum(freq_bin[:,2])
bssf[1,0] = sum(bss[:,2])/sum(freq_bin[:,2])
bssf[2,0] = sum(bss[:,4])/sum(freq_bin[:,2])
bssf[3,0] = Bref75
bssf[4,0] = 1-(bssf[0,0]/Bref75) 

#### Upper dsrefile 
bssf[0,1] = sum(bss[:,1])/sum(freq_bin[:,6])
bssf[1,1] = sum(bss[:,3])/sum(freq_bin[:,6])
bssf[2,1] = sum(bss[:,5])/sum(freq_bin[:,2])
bssf[3,1] = Bref90
bssf[4,1] = 1-(bssf[0,1]/Bref90)







#####  For srefMWF #######
 
####### rows are BSSquart//BSquart//BSSdsref//BSdsref
bss_sref= zeros((11,10))


### bssf rows are BS, reliabilty, resolution, uncertatinty, BSS 
bssf_sref = zeros((10,5))
obquart = (sum(freq_sref_bin[:,1])/sum(freq_sref_bin[:,2]))
obdsref = (sum(freq_sref_bin[:,5])/sum(freq_sref_bin[:,6]))

Bref75 = (obquart)*(1-obquart)

Bref90 = (obdsref)*(1-obdsref)

for i in range(len(freq_sref_bin[:,0])):
    ### BS
    bss_sref[i,0] =((((freq_sref_bin[i,0]-0)**2)*(freq_sref_bin[i,2]-freq_sref_bin[i,1]))+(((freq_sref_bin[i,0]-1)**2)*(freq_sref_bin[i,1])))
    bss_sref[i,1] =((((freq_sref_bin[i,4]-0)**2)*(freq_sref_bin[i,6]-freq_sref_bin[i,5]))+(((freq_sref_bin[i,4]-1)**2)*(freq_sref_bin[i,5])))
    
    ### Reliability
    bss_sref[i,2] = (freq_sref_bin[i,2]*(freq_sref_bin[i,0]-freq_sref_bin[i,3])**2)
    bss_sref[i,3] = (freq_sref_bin[i,6]*(freq_sref_bin[i,4]-freq_sref_bin[i,7])**2)
    
    
    ### Resolution
    bss_sref[i,4] = (freq_sref_bin[i,2]*(freq_sref_bin[i,3]-obquart)**2)
    bss_sref[i,5] = (freq_sref_bin[i,6]*(freq_sref_bin[i,7]-obdsref)**2)
    
    ### Uncertainty is same as BSref (climatology)



##### Upper quartile
bssf_sref[0,0] = sum(bss_sref[:,0])/sum(freq_sref_bin[:,2])
bssf_sref[1,0] = sum(bss_sref[:,2])/sum(freq_sref_bin[:,2])
bssf_sref[2,0] = sum(bss_sref[:,4])/sum(freq_sref_bin[:,2])
bssf_sref[3,0] = Bref75
bssf_sref[4,0] = 1-(bssf_sref[0,0]/Bref75) 

#### Upper dsrefile 
bssf_sref[0,1] = sum(bss_sref[:,1])/sum(freq_sref_bin[:,6])
bssf_sref[1,1] = sum(bss_sref[:,3])/sum(freq_sref_bin[:,6])
bssf_sref[2,1] = sum(bss_sref[:,5])/sum(freq_sref_bin[:,2])
bssf_sref[3,1] = Bref90
bssf_sref[4,1] = 1-(bssf_sref[0,1]/Bref90)


###### Save Variables #######
#Frequency data
np.save('sref_arw_percentile_prob_int', freq_bin)
np.save('sref_nmb_percentile_prob_int', freq_sref_bin)
#Skill Score data
np.save('sref_arw_percentile_prob_bss_int', bssf)
np.save('sref_nmb_percentile_prob_bss_int', bssf_sref)
#Uncertainty data
np.save('sref_arw_sref_nmb_percentile_prob_uncertainty_con75_int', pr_con75)
np.save('sref_arw_sref_nmb_percentile_prob_uncertainty_unc75_int', pr_unc75)
np.save('sref_arw_sref_nmb_percentile_prob_uncertainty_con90_int', pr_con90)
np.save('sref_arw_sref_nmb_percentile_prob_uncertainty_unc90_int', pr_unc90)


#%%
                        
                                
###############################################################################
################################# Plots #######################################   
###############################################################################  
                              
                                
linecolor = ['blue', 'green', 'red', 'c']                               
fig=plt.figure(num=None, figsize=(18,12), dpi=500, facecolor='w', edgecolor='k')
no_res = np.full((21),.25)
no_skill = np.arange(.125,.625001,.025)
freq_fill = np.arange(0,1.0001,0.05)







################  Upper Quartile #############################################



ax1 = fig.add_subplot(121)
fig.subplots_adjust(bottom=0.4)
plt.gca().set_color_cycle(linecolor)
a = ax1.plot(freq_bin[:,0],freq_bin[:,3], linewidth = 2, c = 'blue', marker = "o", markeredgecolor = 'none')

ax1.errorbar(freq_bin[:,0],freq_bin[:,3], yerr= [abs(pr_unc75[0,:,1]-freq_bin[:,3]), abs(pr_unc75[0,:,2]-freq_bin[:,3])], c = 'b')
ax1.errorbar(freq_sref_bin[:,0],freq_sref_bin[:,3], yerr= [abs(pr_unc75[1,:,1]-freq_sref_bin[:,3]), abs(pr_unc75[1,:,2]-freq_sref_bin[:,3])], c = 'r')

ax1.errorbar(freq_bin[:,0],freq_bin[:,0], yerr= [abs(pr_con75[0,:,1]-freq_bin[:,0]), abs(pr_con75[0,:,2]-freq_bin[:,0])], c = 'b')
ax1.errorbar(freq_sref_bin[:,0],freq_sref_bin[:,0], yerr= [abs(pr_con75[1,:,1]-freq_sref_bin[:,0]), abs(pr_con75[1,:,2]-freq_sref_bin[:,0])], c = 'r')




ax1.plot(freq_sref_bin[:,0],freq_sref_bin[:,3], linewidth = 2, c = 'red',marker = "o", markeredgecolor = 'none')
c = ax1.plot(freq_bin[:,0],freq_bin[:,0], linewidth = 2, c = 'k', markeredgecolor = 'none')
d = ax1.plot(freq_fill,no_res, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
e = ax1.plot(freq_fill,no_skill, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
plt.text(.53,.72,'Perfect Reliability',rotation = 45)
plt.text(.6,.27,'No Resolution (Climatology)',rotation = 0)
plt.text(.6,.48,'No Skill',rotation = 27)

ax1.fill_between(freq_fill,no_skill, 1, where=no_skill >= .24999,facecolor = 'grey',alpha=0.5)
ax1.fill_between(freq_fill,no_skill, 0, where=no_skill <= .25, facecolor = 'grey',alpha=0.5)

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(np.arange(0,1.0001,0.1))
plt.yticks(np.arange(0,1.0001,0.1))
plt.grid(True)



########
                
                


###### Table to show all stats for both models
                
#tab.table(cellText=stats,colWidths=cwid,rowLabels=rows,colLabels=columns,loc='center',fontsize=5)      

the_table = plt.table(cellText=[('%.3f' % bssf[0,0],'%.3f' % bssf[1,0],'%.3f' % bssf[2,0],'%.3f' % bssf[3,0],'%.3f' % bssf[4,0]),
                                ('%.3f' % bssf_sref[0,0],'%.3f' % bssf_sref[1,0],'%.3f' % bssf_sref[2,0],'%.3f' % bssf_sref[3,0],'%.3f' % bssf_sref[4,0]),
                                ('%.3f' % bssf[0,1],'%.3f' % bssf[1,1],'%.3f' % bssf[2,1],'%.3f' % bssf[3,1],'%.3f' % bssf[4,1]),
                                ('%.3f' % bssf_sref[0,1],'%.3f' % bssf_sref[1,1],'%.3f' % bssf_sref[2,1],'%.3f' % bssf_sref[3,1],'%.3f' % bssf_sref[4,1])],

                                
          rowLabels=["NCAR \n(Upper Quart.)","SREF \n(Upper Quart.)","NCAR \n(Upper Dec.)","SREF\n(Upper Dec.)"],
          colLabels=["Brier Score","Reliability", "Resolution", "Uncertainty", "Brier Skill Score"],
          loc="center",
          cellLoc = "center",
          rowColours=['lightgrey','lightgrey','lightgrey','lightgrey'],
          colColours=['lightgrey','lightgrey','lightgrey','lightgrey','lightgrey'],
          bbox=[.64,-0.65,1.135,.47],
          edges = 'BRLT')
the_table.auto_set_font_size(False)
the_table.scale(1.05,1.3)
the_table.set_fontsize(12)

                    

blue_line = mlines.Line2D([],[] , color='blue',
                           label='NCAR (10 Mem.)',  linewidth = 2,marker = "o", markeredgecolor = 'none')
red_line = mlines.Line2D([],[] , color='red',
                           label='SREF (26 Mem.)',  linewidth = 2,marker = "o", markeredgecolor = 'none')

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
b = ax2.plot(freq_bin[:,4],freq_bin[:,7], linewidth = 2, c = 'blue',marker = "o", markeredgecolor = 'none')

ax2.errorbar(freq_bin[:,4],freq_bin[:,7], yerr= [abs(pr_unc90[0,:,1]-freq_bin[:,7]), abs(pr_unc90[0,:,2]-freq_bin[:,7])], c = 'b')
ax2.errorbar(freq_sref_bin[:,4],freq_sref_bin[:,7], yerr= [abs(pr_unc90[1,:,1]-freq_sref_bin[:,7]), abs(pr_unc90[1,:,2]-freq_sref_bin[:,7])], c = 'r')

ax2.errorbar(freq_bin[:,4],freq_bin[:,4], yerr= [abs(pr_con90[0,:,1]-freq_bin[:,4]), abs(pr_con90[0,:,2]-freq_bin[:,4])], c = 'b')
ax2.errorbar(freq_sref_bin[:,4],freq_sref_bin[:,4], yerr= [abs(pr_con90[1,:,1]-freq_sref_bin[:,4]), abs(pr_con90[1,:,2]-freq_sref_bin[:,4])], c = 'r')



ax2.plot(freq_sref_bin[:,4],freq_sref_bin[:,7], linewidth = 2, c = 'red',marker = "o", markeredgecolor = 'none')
c = ax2.plot(freq_bin[:,0],freq_bin[:,0], linewidth = 2, c = 'k', markeredgecolor = 'none')
d = ax2.plot(freq_fill,no_res, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
e = ax2.plot(freq_fill,no_skill, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
plt.text(.75,.94,'Perfect Reliability',rotation = 45)
plt.text(.3,.11,'No Resolution (Climatology)',rotation = 0)
plt.text(.7,.46,'No Skill',rotation = 27)


                


ax2.fill_between(freq_fill,no_skill, 1, where=no_skill >= .0999999,facecolor = 'grey',alpha=0.5)
ax2.fill_between(freq_fill,no_skill, 0, where=no_skill <= .10001, facecolor = 'grey',alpha=0.5)

plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(np.arange(0,1.0001,0.1))
plt.yticks(np.arange(0,1.0001,0.1))
plt.grid(True)


plt.legend(handles=[blue_line, red_line], loc = "lower right",prop={'size':10.5})
plt.title('Upper Decile 24-Hour Precipitation Events', fontsize = 20)
plt.ylabel('Observed Relative Frequency', fontsize = 15)
plt.xlabel('Forecast Probability',fontsize = 15)





################## Sample frequency bar graphs ################################

a = plt.axes([.15, .24, .1, .1], axisbg='white')
plt.bar(freq_bin[:,0],freq_bin[:,2],width = .06, color = 'blue', edgecolor ='none', align='center')
plt.xlim([-0.04,1.04])
plt.ylim([10,10000])
plt.title('NCAR (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Probab.',fontsize = 10)
plt.xticks(np.arange(0,1.0001,.2))
#a.set_yscale('log')
plt.tick_params(axis='x',which='both', bottom='off')





a = plt.axes([.15, .07, .1, .1], axisbg='white')
plt.bar(freq_sref_bin[:,0],freq_sref_bin[:,2],width = .06, color = 'red', edgecolor ='none', align='center')
plt.xlim([-0.01,1.01])
plt.ylim([10,10000])
plt.title('SREF (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
#a.set_yscale('log')
plt.xticks(np.arange(0,1.0001,.2))
plt.tick_params(axis='x',which='both', bottom='off')






### Upper Decile plots
a = plt.axes([.81, .24, .1, .1], axisbg='white')
plt.bar(freq_bin[:,4],freq_bin[:,6],width = .06, color = 'blue', edgecolor ='none',align='center')
plt.xlim([-0.04,1.04])
plt.ylim([10,10000])
plt.title('NCAR (Upper Dec.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
#a.set_yscale('log')

plt.xticks()
plt.xticks(np.arange(0,1.0001,.2))
plt.tick_params(axis='x',which='both', bottom='off')







a = plt.axes([.81, .07, .1, .1], axisbg='white')
plt.bar(freq_sref_bin[:,4],freq_sref_bin[:,6],width = 0.06, color = 'red', edgecolor ='none',align='center')
plt.xlim([-0.01,1.01])
plt.ylim([10,10000])
plt.title('SREF (Upper Dec.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
#a.set_yscale('log')
plt.xticks()
plt.xticks(np.arange(0,1.0001,.2))
plt.tick_params(axis='x',which='both', bottom='off')



plt.savefig("../../../public_html/reliability_diagram_sref_arw_sref_nmb_interp_bin_percentile_85_95_int.pdf")                               
        
     
          
          
          
          
          



