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
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_precip_12Zto12Z_upperquart_prob_interp_10random_mem_arw_nmb_day.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_precip_12Zto12Z_upperdec_prob_interp_10random_mem_arw_nmb_day.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_precip_12Zto12Z_upperquart_prob_interp_10random_mem_arw_day.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_precip_12Zto12Z_upperdec_prob_interp_10random_mem_arw_day.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_precip_12Zto12Z_upperquart_prob_interp_10random_mem_nmb_day.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/2016_17_cool_season/sref_precip_12Zto12Z_upperdec_prob_interp_10random_mem_nmb_day.txt"]

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
sref_data75 = data[1,:,:] 
sref_data90 = data[2,:,:]
arw_data75 = data[3,:,:] 
arw_data90 = data[4,:,:]
nmb_data75 = data[5,:,:] 
nmb_data90 = data[6,:,:]



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

for w in range(len(inhouse_data[:,0])):
    if inhouse_data[w,0] != 0:
        for i in range(3,185):
            if percentiles[w,1] <= inhouse_data[w,i] < 1000 :
                inhouse_data75[w,i] = 1
            elif percentiles[w,1] > inhouse_data[w,i]:
                inhouse_data75[w,i] = 0
            elif 1000 < inhouse_data[w,i]:
                inhouse_data75[w,i] = 9999
          
          
for w in range(len(inhouse_data[:,0])):
    if inhouse_data[w,0] != 0:
        for i in range(3,185):
            if percentiles[w,2] <= inhouse_data[w,i] < 1000 :
                inhouse_data90[w,i] = 1
            elif percentiles[w,2] > inhouse_data[w,i]:
                inhouse_data90[w,i] = 0
            elif 1000 < inhouse_data[w,i]:
                inhouse_data90[w,i] = 9999   
         
###############################################################################
############ Calc forecast freq as function of observed freq ##################   
###############################################################################   

xi75sref = np.array([0])
xi90sref = np.array([0])

xi75arw = np.array([0])
xi90arw = np.array([0])

xi75nmb = np.array([0])
xi90nmb = np.array([0])


     
freq_sref = zeros((11,20))
freq_sref[:,0] = np.arange(0,1.001,.1)
freq_arw = zeros((11,20))
freq_arw[:,0] = np.arange(0,1.001,.1)

freq_nmb = zeros((11,20))
freq_nmb[:,0] = np.arange(0,1.001,.1)

#%%
      
for w in range(len(inhouse_data75[:,0])):  # sref 75
    for x in range(len(sref_data75[:,0])):
        if inhouse_data75[w,0] == sref_data75[x,0]:
            if inhouse_data75[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(11):
                            if sref_data75[x,i] == t/10.:
                                freq_sref[t,1] = freq_sref[t,1] + inhouse_data75[w,i]
                                freq_sref[t,2] = freq_sref[t,2] + 1
                                xi75sref = np.append(xi75sref, sref_data75[x,i])
                                

for w in range(len(inhouse_data90[:,0])):  # sref 90
    for x in range(len(sref_data90[:,0])):
        if inhouse_data90[w,0] == sref_data90[x,0]:
            if inhouse_data90[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(11):
                            if sref_data90[x,i] == t/10.:
                                freq_sref[t,4] = freq_sref[t,4] + inhouse_data90[w,i]
                                freq_sref[t,5] = freq_sref[t,5] + 1
                                xi90sref = np.append(xi90sref, sref_data90[x,i])
                                
#%%
for w in range(len(inhouse_data75[:,0])):  # arw 75
    for x in range(len(arw_data75[:,0])):
        if inhouse_data75[w,0] == arw_data75[x,0]:
            if inhouse_data75[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(11):
                            if arw_data75[x,i] == t/10.:
                                freq_arw[t,1] = freq_arw[t,1] + inhouse_data75[w,i]
                                freq_arw[t,2] = freq_arw[t,2] + 1
                                xi75arw = np.append(xi75arw, arw_data75[x,i])
                                

for w in range(len(inhouse_data90[:,0])):  # arw 90
    for x in range(len(arw_data90[:,0])):
        if inhouse_data90[w,0] == arw_data90[x,0]:
            if inhouse_data90[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(11):
                            if arw_data90[x,i] == t/10.:
                                freq_arw[t,4] = freq_arw[t,4] + inhouse_data90[w,i]
                                freq_arw[t,5] = freq_arw[t,5] + 1
                                xi90arw = np.append(xi90arw, arw_data90[x,i])
                        
 
#%%
                               
for w in range(len(inhouse_data75[:,0])):  # nmb 75
    for x in range(len(nmb_data75[:,0])):
        if inhouse_data75[w,0] == nmb_data75[x,0]:
            if inhouse_data75[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(11):
                            if nmb_data75[x,i] == t/10.:
                                freq_nmb[t,1] = freq_nmb[t,1] + inhouse_data75[w,i]
                                freq_nmb[t,2] = freq_nmb[t,2] + 1
                                xi75nmb = np.append(xi75nmb, nmb_data75[x,i])
                              
                                

for w in range(len(inhouse_data90[:,0])):  # nmb 90
    for x in range(len(nmb_data90[:,0])):
        if inhouse_data90[w,0] == nmb_data90[x,0]:
            if inhouse_data90[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(11):
                            if nmb_data90[x,i] == t/10.:
                                freq_nmb[t,4] = freq_nmb[t,4] + inhouse_data90[w,i]
                                freq_nmb[t,5] = freq_nmb[t,5] + 1                               
                                xi90nmb = np.append(xi90nmb, nmb_data90[x,i])
                                

                                
                                
                                
                                
freq_sref[:,3] = freq_sref[:,1]/freq_sref[:,2]      # sref 75     
freq_sref[:,6] = freq_sref[:,4]/freq_sref[:,5]      # sref 90    
freq_arw[:,3] = freq_arw[:,1]/freq_arw[:,2]      # arw 75     
freq_arw[:,6] = freq_arw[:,4]/freq_arw[:,5]      # arw 90
freq_nmb[:,3] = freq_nmb[:,1]/freq_nmb[:,2]      # nmb 75     
freq_nmb[:,6] = freq_nmb[:,4]/freq_nmb[:,5]      # nmb 90






#%%


###############################################################################
############### Resampling to create consistancy bars #########################  
###############################################################################
numtime75 = zeros((3))
numtime90 = zeros((3))
numtime75[0] = len(xi75sref)
numtime90[0] = len(xi90sref)
numtime75[1] = len(xi75arw)
numtime90[1] = len(xi90arw)
numtime75[2] = len(xi75nmb)
numtime90[2] = len(xi90nmb)

cycles = 1000

#Initialize array to hold results rom all 1000 resample cycles
resamp_con75 = zeros((3,11,1001))
resamp_con75[:,:,0] = linspace(0,1,11)

for r in range(0,3):
    for i in range(cycles):
        print i
        #Intilaize array to sroe data for resample cycle
        counter = zeros((11,3))
        counter[:,0] = linspace(0,1,11)

        #Determine forecast (resampled) and observed (uniform distribution)
        if r == 0:
            for75 = np.random.choice(xi75sref,len(xi75sref))
            ob75 = np.random.uniform(0, 1, len(xi75sref))
        if r == 1:
            for75 = np.random.choice(xi75arw,len(xi75arw))
            ob75 = np.random.uniform(0, 1, len(xi75arw))
        if r == 2:
            for75 = np.random.choice(xi75nmb,len(xi75nmb))
            ob75 = np.random.uniform(0, 1, len(xi75nmb))
    
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
resamp_unc75= zeros((3,11,1001))
resamp_unc75[:,:,0] = linspace(0,1,11)

for r in range(0,3):
    for i in range(cycles):
        print i
        #Intilaize array to sroe data for resample cycle
        counter = zeros((11,3))
        counter[:,0] = linspace(0,1,11)

        #Determine forecast (resampled) and observed (uniform distribution)
        if r ==0:
            for75 = np.random.choice(xi75sref,len(xi75sref))
        if r ==1:
            for75 = np.random.choice(xi75arw,len(xi75arw))
        if r ==2:
            for75 = np.random.choice(xi75nmb,len(xi75nmb))
    
        #Loop over all forecast/observe pairs in one cycle
        for t in range(int(numtime75[r])):
            num = int(round(for75[t],1)*10)
            counter[num,1] = counter[num,1] + 1
        
        if r == 0:
            counter[:,2] = freq_sref[:,1]
        if r == 1:
            counter[:,2] = freq_arw[:,1]
        if r == 2:
            counter[:,2] = freq_nmb[:,1]

    
        #Store observed relative frequencies from each resample cycle
        for j in range(11):
            resamp_unc75[r,:,i+1] = counter[:,2]/counter[:,1]
        


###############################################################################
######################## Determine Percentiles ################################  
###############################################################################
pr_con75 = zeros((3,11,3))
pr_unc75 = zeros((3,11,3))

pr_con75[:,:,0] = linspace(0,1,11)
pr_unc75[:,:,0] = linspace(0,1,11)

for r in range(0,3):
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
resamp_con90 = zeros((3,11,1001))
resamp_con90[:,:,0] = linspace(0,1,11)

for r in range(0,3):
    for i in range(cycles):
        print i
        #Intilaize array to sroe data for resample cycle
        counter = zeros((11,3))
        counter[:,0] = linspace(0,1,11)

        #Determine forecast (resampled) and observed (uniform distribution)
        if r == 0:
            for90 = np.random.choice(xi90sref,len(xi90sref))
            ob90 = np.random.uniform(0, 1, len(xi90sref))
        if r == 1:
            for90 = np.random.choice(xi90arw,len(xi90arw))
            ob90 = np.random.uniform(0, 1, len(xi90arw))
        if r == 2:
            for90 = np.random.choice(xi90nmb,len(xi90nmb))
            ob90 = np.random.uniform(0, 1, len(xi90nmb))
    
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
resamp_unc90= zeros((3,11,1001))
resamp_unc90[:,:,0] = linspace(0,1,11)

for r in range(0,3):
    for i in range(cycles):
        print i
        #Intilaize array to sroe data for resample cycle
        counter = zeros((11,3))
        counter[:,0] = linspace(0,1,11)

        #Determine forecast (resampled) and observed (uniform distribution)
        if r ==0:
            for90 = np.random.choice(xi90sref,len(xi90sref))
        if r ==1:
            for90 = np.random.choice(xi90arw,len(xi90arw))
        if r ==2:
            for90 = np.random.choice(xi90nmb,len(xi90nmb))
    
        #Loop over all forecast/observe pairs in one cycle
        for t in range(int(numtime90[r])):
            num = int(round(for90[t],1)*10)
            counter[num,1] = counter[num,1] + 1
            
        if r == 0:
            counter[:,2] = freq_sref[:,4]
        if r == 1:
            counter[:,2] = freq_arw[:,4]
        if r == 2:
            counter[:,2] = freq_nmb[:,4]
    
        #Store observed relative frequencies from each resample cycle
        for j in range(11):
            resamp_unc90[r,:,i+1] = counter[:,2]/counter[:,1]
        


###############################################################################
######################## Determine Percentiles ################################  
###############################################################################
pr_con90 = zeros((3,11,3))
pr_unc90 = zeros((3,11,3))

pr_con90[:,:,0] = linspace(0,1,11)
pr_unc90[:,:,0] = linspace(0,1,11)

for r in range(0,3):
    pr_con90[r,:,1] = np.percentile(resamp_con90[r,:,1:], 5, axis = 1)
    pr_con90[r,:,2] = np.percentile(resamp_con90[r,:,1:], 95, axis = 1)
    
    pr_unc90[r,:,1] = np.percentile(resamp_unc90[r,:,1:], 5, axis = 1)
    pr_unc90[r,:,2] = np.percentile(resamp_unc90[r,:,1:], 95, axis = 1)






#%%









###############################################################################
############ BSS and BS   #####################################################   
###############################################################################   


#####  For SREF #######
####### rows are BSSquart//BSquart//BSSdec//BSdec
bss_sref= zeros((11,10))


### BSS rows are BS, reliabilty, resolution, uncertatinty, BSS 
bssf_sref = zeros((10,5))
obquart = (sum(freq_sref[:,1])/sum(freq_sref[:,2]))
obdec = (sum(freq_sref[:,4])/sum(freq_sref[:,5]))

Bref75 = (obquart)*(1-obquart)

Bref90 = (obdec)*(1-obdec)

for i in range(len(freq_sref[:,0])):
    ### BS
    bss_sref[i,0] =((((freq_sref[i,0]-0)**2)*(freq_sref[i,2]-freq_sref[i,1]))+(((freq_sref[i,0]-1)**2)*(freq_sref[i,1])))
    bss_sref[i,1] =((((freq_sref[i,0]-0)**2)*(freq_sref[i,5]-freq_sref[i,4]))+(((freq_sref[i,0]-1)**2)*(freq_sref[i,4])))
    
    ### Reliability
    bss_sref[i,2] = (freq_sref[i,2]*(freq_sref[i,0]-freq_sref[i,3])**2)
    bss_sref[i,3] = (freq_sref[i,5]*(freq_sref[i,0]-freq_sref[i,6])**2)
    
    
    ### Resolution
    bss_sref[i,4] = (freq_sref[i,2]*(freq_sref[i,3]-obquart)**2)
    bss_sref[i,5] = (freq_sref[i,5]*(freq_sref[i,6]-obdec)**2)
    
    ### Uncertainty is same as BSref (climatology)



##### Upper quartile
bssf_sref[0,0] = sum(bss_sref[:,0])/sum(freq_sref[:,2])
bssf_sref[1,0] = sum(bss_sref[:,2])/sum(freq_sref[:,2])
bssf_sref[2,0] = sum(bss_sref[:,4])/sum(freq_sref[:,2])
bssf_sref[3,0] = Bref75
bssf_sref[4,0] = 1-(bssf_sref[0,0]/Bref75) 

#### Upper decile 
bssf_sref[0,1] = sum(bss_sref[:,1])/sum(freq_sref[:,5])
bssf_sref[1,1] = sum(bss_sref[:,3])/sum(freq_sref[:,5])
bssf_sref[2,1] = sum(bss_sref[:,5])/sum(freq_sref[:,2])
bssf_sref[3,1] = Bref90
bssf_sref[4,1] = 1-(bssf_sref[0,1]/Bref90)






#####  For arw #######
####### rows are BSSquart//BSquart//BSSdec//BSdec
bss_arw= zeros((11,10))


### BSS rows are BS, reliabilty, resolution, uncertatinty, BSS 
bssf_arw = zeros((10,5))
obquart = (sum(freq_arw[:,1])/sum(freq_arw[:,2]))
obdec = (sum(freq_arw[:,4])/sum(freq_arw[:,5]))

Bref75 = (obquart)*(1-obquart)

Bref90 = (obdec)*(1-obdec)

for i in range(len(freq_arw[:,0])):
    ### BS
    bss_arw[i,0] =((((freq_arw[i,0]-0)**2)*(freq_arw[i,2]-freq_arw[i,1]))+(((freq_arw[i,0]-1)**2)*(freq_arw[i,1])))
    bss_arw[i,1] =((((freq_arw[i,0]-0)**2)*(freq_arw[i,5]-freq_arw[i,4]))+(((freq_arw[i,0]-1)**2)*(freq_arw[i,4])))
    
    ### Reliability
    bss_arw[i,2] = (freq_arw[i,2]*(freq_arw[i,0]-freq_arw[i,3])**2)
    bss_arw[i,3] = (freq_arw[i,5]*(freq_arw[i,0]-freq_arw[i,6])**2)
    
    
    ### Resolution
    bss_arw[i,4] = (freq_arw[i,2]*(freq_arw[i,3]-obquart)**2)
    bss_arw[i,5] = (freq_arw[i,5]*(freq_arw[i,6]-obdec)**2)
    
    ### Uncertainty is same as Barw (climatology)



##### Upper quartile
bssf_arw[0,0] = sum(bss_arw[:,0])/sum(freq_arw[:,2])
bssf_arw[1,0] = sum(bss_arw[:,2])/sum(freq_arw[:,2])
bssf_arw[2,0] = sum(bss_arw[:,4])/sum(freq_arw[:,2])
bssf_arw[3,0] = Bref75
bssf_arw[4,0] = 1-(bssf_arw[0,0]/Bref75) 

#### Upper decile 
bssf_arw[0,1] = sum(bss_arw[:,1])/sum(freq_arw[:,5])
bssf_arw[1,1] = sum(bss_arw[:,3])/sum(freq_arw[:,5])
bssf_arw[2,1] = sum(bss_arw[:,5])/sum(freq_arw[:,2])
bssf_arw[3,1] = Bref90
bssf_arw[4,1] = 1-(bssf_arw[0,1]/Bref90)








#####  For nmb #######
####### rows are BSSquart//BSquart//BSSdec//BSdec
bss_nmb= zeros((11,10))


### BSS rows are BS, reliabilty, resolution, uncertatinty, BSS 
bssf_nmb = zeros((10,5))
obquart = (sum(freq_nmb[:,1])/sum(freq_nmb[:,2]))
obdec = (sum(freq_nmb[:,4])/sum(freq_nmb[:,5]))

Bref75 = (obquart)*(1-obquart)

Bref90 = (obdec)*(1-obdec)

for i in range(len(freq_nmb[:,0])):
    ### BS
    bss_nmb[i,0] =((((freq_nmb[i,0]-0)**2)*(freq_nmb[i,2]-freq_nmb[i,1]))+(((freq_nmb[i,0]-1)**2)*(freq_nmb[i,1])))
    bss_nmb[i,1] =((((freq_nmb[i,0]-0)**2)*(freq_nmb[i,5]-freq_nmb[i,4]))+(((freq_nmb[i,0]-1)**2)*(freq_nmb[i,4])))
    
    ### Reliability
    bss_nmb[i,2] = (freq_nmb[i,2]*(freq_nmb[i,0]-freq_nmb[i,3])**2)
    bss_nmb[i,3] = (freq_nmb[i,5]*(freq_nmb[i,0]-freq_nmb[i,6])**2)
    
    
    ### Resolution
    bss_nmb[i,4] = (freq_nmb[i,2]*(freq_nmb[i,3]-obquart)**2)
    bss_nmb[i,5] = (freq_nmb[i,5]*(freq_nmb[i,6]-obdec)**2)
    
    ### Uncertainty is same as Bnmb (climatology)



##### Upper quartile
bssf_nmb[0,0] = sum(bss_nmb[:,0])/sum(freq_nmb[:,2])
bssf_nmb[1,0] = sum(bss_nmb[:,2])/sum(freq_nmb[:,2])
bssf_nmb[2,0] = sum(bss_nmb[:,4])/sum(freq_nmb[:,2])
bssf_nmb[3,0] = Bref75
bssf_nmb[4,0] = 1-(bssf_nmb[0,0]/Bref75) 

#### Upper decile 
bssf_nmb[0,1] = sum(bss_nmb[:,1])/sum(freq_nmb[:,5])
bssf_nmb[1,1] = sum(bss_nmb[:,3])/sum(freq_nmb[:,5])
bssf_nmb[2,1] = sum(bss_nmb[:,5])/sum(freq_nmb[:,2])
bssf_nmb[3,1] = Bref90
bssf_nmb[4,1] = 1-(bssf_nmb[0,1]/Bref90)














#%%
                        
                                
###############################################################################
################################# Plots #######################################   
###############################################################################  
                              
                                
linecolor = ['red', 'green', 'gold']                               
fig=plt.figure(num=None, figsize=(18,12), dpi=500, facecolor='w', edgecolor='k')
no_res = np.full((21),.25)
no_skill = np.arange(.125,.625001,.025)
freq_fill = np.arange(0,1.0001,0.05)







################  Upper Quartile #############################################



ax1 = fig.add_subplot(121)
fig.subplots_adjust(bottom=0.4)
plt.gca().set_color_cycle(linecolor)
a = ax1.plot(freq_sref[:,0],freq_sref[:,3], linewidth = 2, c = 'red', marker = "o", markeredgecolor = 'none')

ax1.errorbar(freq_sref[:,0],freq_sref[:,3], yerr= [abs(pr_unc75[0,:,1]-freq_sref[:,3]), abs(pr_unc75[0,:,2]-freq_sref[:,3])], c = 'r')
ax1.errorbar(freq_arw[:,0],freq_arw[:,3], yerr= [abs(pr_unc75[1,:,1]-freq_arw[:,3]), abs(pr_unc75[1,:,2]-freq_arw[:,3])], c = 'green')
ax1.errorbar(freq_nmb[:,0],freq_nmb[:,3], yerr= [abs(pr_unc75[2,:,1]-freq_nmb[:,3]), abs(pr_unc75[2,:,2]-freq_nmb[:,3])], c = 'gold')

ax1.errorbar(freq_sref[:,0]-0.005,freq_sref[:,0]-0.005, yerr= [abs(pr_con75[0,:,1]-freq_sref[:,0]), abs(pr_con75[0,:,2]-freq_sref[:,0])], c = 'r')
ax1.errorbar(freq_arw[:,0],freq_arw[:,0], yerr= [abs(pr_con75[1,:,1]-freq_arw[:,0]), abs(pr_con75[1,:,2]-freq_arw[:,0])], c = 'green')
ax1.errorbar(freq_nmb[:,0]+0.005,freq_nmb[:,0]+0.005, yerr= [abs(pr_con75[2,:,1]-freq_nmb[:,0]), abs(pr_con75[2,:,2]-freq_nmb[:,0])], c = 'gold')



ax1.plot(freq_arw[:,0],freq_arw[:,3], linewidth = 2, c = 'green',marker = "o", markeredgecolor = 'none')
ax1.plot(freq_nmb[:,0],freq_nmb[:,3], linewidth = 2, c = 'gold',marker = "o", markeredgecolor = 'none')
c = ax1.plot(freq_sref[:,0],freq_sref[:,0], linewidth = 2, c = 'k', markeredgecolor = 'none')
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


the_table = plt.table(cellText=[('%.3f' % bssf_sref[0,0],'%.3f' % bssf_sref[1,0],'%.3f' % bssf_sref[2,0],'%.3f' % bssf_sref[3,0],'%.3f' % bssf_sref[4,0]),
                                ('%.3f' % bssf_arw[0,0],'%.3f' % bssf_arw[1,0],'%.3f' % bssf_arw[2,0],'%.3f' % bssf_arw[3,0],'%.3f' % bssf_arw[4,0]),
                                ('%.3f' % bssf_nmb[0,0],'%.3f' % bssf_nmb[1,0],'%.3f' % bssf_nmb[2,0],'%.3f' % bssf_nmb[3,0],'%.3f' % bssf_nmb[4,0]),
                                ('%.3f' % bssf_sref[0,1],'%.3f' % bssf_sref[1,1],'%.3f' % bssf_sref[2,1],'%.3f' % bssf_sref[3,1],'%.3f' % bssf_sref[4,1]),
                                ('%.3f' % bssf_arw[0,1],'%.3f' % bssf_arw[1,1],'%.3f' % bssf_arw[2,1],'%.3f' % bssf_arw[3,1],'%.3f' % bssf_arw[4,1]),
                                ('%.3f' % bssf_nmb[0,1],'%.3f' % bssf_nmb[1,1],'%.3f' % bssf_nmb[2,1],'%.3f' % bssf_nmb[3,1],'%.3f' % bssf_nmb[4,1])],

                                
    
          rowLabels=["SREF \n(Upper Quart.)","ARW \n(Upper Quart.)","NMMB \n(Upper Quart.)","SREF \n(Upper Dec.)","ARW \n(Upper Dec.)","NMMB \n(Upper Dec.)"],
          colLabels=["Brier Score","Reliability", "Resolution", "Uncertainty", "Brier Skill Score"],
          loc="center",
          cellLoc = "center",
          rowColours=['lightgrey','lightgrey','lightgrey','lightgrey','lightgrey','lightgrey'],
          colColours=['lightgrey','lightgrey','lightgrey','lightgrey','lightgrey'],
          bbox=[.64,-0.71,1.135,.56],
          edges = 'BRLT')
the_table.auto_set_font_size(False)
the_table.scale(1.05,1.3)
the_table.set_fontsize(12)

                    

red_line = mlines.Line2D([],[] , color='red',
                           label='SREF (10 Mem.)',  linewidth = 2,marker = "o", markeredgecolor = 'none')
green_line = mlines.Line2D([],[] , color='green',
                           label='ARW (10 Mem.)',  linewidth = 2,marker = "o", markeredgecolor = 'none')
gold_line = mlines.Line2D([],[] , color='gold',
                           label='NMMB (10 Mem.)',  linewidth = 2,marker = "o", markeredgecolor = 'none')

plt.legend(handles=[ red_line, green_line, gold_line], loc = "lower right",prop={'size':10.5})
plt.title('Upper Quartile Events', fontsize = 20)
plt.ylabel('Observed Relative Frequency', fontsize = 15)
plt.xlabel('Forecast Probability',fontsize = 15)

















################  UpperDecile #############################################


no_res = np.full((21),.1)
no_skill = np.arange(.05,.55001,.025)
freq_fill = np.arange(0,1.0001,0.05)


ax2 = fig.add_subplot(122)
fig.subplots_adjust(bottom=0.4)
plt.gca().set_color_cycle(linecolor)
b = ax2.plot(freq_sref[:,0],freq_sref[:,6], linewidth = 2, c = 'red',marker = "o", markeredgecolor = 'none')

ax2.errorbar(freq_sref[:,0],freq_sref[:,6], yerr= [abs(pr_unc90[0,:,1]-freq_sref[:,6]), abs(pr_unc90[0,:,2]-freq_sref[:,6])], c = 'r')
ax2.errorbar(freq_arw[:,0],freq_arw[:,6], yerr= [abs(pr_unc90[1,:,1]-freq_arw[:,6]), abs(pr_unc90[1,:,2]-freq_arw[:,6])], c = 'green')
ax2.errorbar(freq_nmb[:,0],freq_nmb[:,6], yerr= [abs(pr_unc90[2,:,1]-freq_nmb[:,6]), abs(pr_unc90[2,:,2]-freq_nmb[:,6])], c = 'gold')

ax2.errorbar(freq_sref[:,0]-0.005,freq_sref[:,0]-0.005, yerr= [abs(pr_con90[0,:,1]-freq_sref[:,0]), abs(pr_con90[0,:,2]-freq_sref[:,0])], c = 'r')
ax2.errorbar(freq_arw[:,0],freq_arw[:,0], yerr= [abs(pr_con90[1,:,1]-freq_arw[:,0]), abs(pr_con90[1,:,2]-freq_arw[:,0])], c = 'green')
ax2.errorbar(freq_nmb[:,0]+0.005,freq_nmb[:,0]+0.005, yerr= [abs(pr_con90[2,:,1]-freq_nmb[:,0]), abs(pr_con90[2,:,2]-freq_nmb[:,0])], c = 'gold')



ax2.plot(freq_arw[:,0],freq_arw[:,6], linewidth = 2, c = 'green',marker = "o", markeredgecolor = 'none')
ax2.plot(freq_nmb[:,0],freq_nmb[:,6], linewidth = 2, c = 'gold',marker = "o", markeredgecolor = 'none')
c = ax2.plot(freq_sref[:,0],freq_sref[:,0], linewidth = 2, c = 'k', markeredgecolor = 'none')
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


plt.legend(handles=[red_line, green_line, gold_line], loc = "lower right",prop={'size':10.5})
plt.title('Upper Decile Events', fontsize = 20)
plt.ylabel('Observed Relative Frequency', fontsize = 15)
plt.xlabel('Forecast Probability',fontsize = 15)





################## Sample frequency bar graphs ################################

a = plt.axes([.15, .28, .1, .055], axisbg='white')
plt.bar(freq_sref[:,0],freq_sref[:,2],width = .06, color = 'red', edgecolor ='none', align='center')
plt.xlim([-0.04,1.04])
plt.ylim([10,10000])
#plt.title('NCAR (Upper Quart.)', y = 1.05, fontsize = 10)
plt.title('SREF (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Probab.',fontsize = 10)
plt.xticks(np.arange(0,1.0001,.2))
plt.yticks(np.arange(0,10000,3000))
#a.set_yscale('log')
plt.tick_params(axis='x',which='both', bottom='off')





a = plt.axes([.15, .16, .1, .055], axisbg='white')
plt.bar(freq_arw[:,0],freq_arw[:,2],width = .06, color = 'green', edgecolor ='none', align='center')
plt.xlim([-0.04,1.04])
plt.ylim([10,10000])
#plt.title('SREF (Upper Quart.)', y = 1.05, fontsize = 10)
plt.title('ARW (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
#a.set_yscale('log')
plt.xticks(np.arange(0,1.0001,.2))
plt.yticks(np.arange(0,10000,3000))
plt.tick_params(axis='x',which='both', bottom='off')


a = plt.axes([.15, .04, .1, .055], axisbg='white')
plt.bar(freq_nmb[:,0],freq_nmb[:,2],width = .06, color = 'gold', edgecolor ='none', align='center')
plt.xlim([-0.04,1.04])
plt.ylim([10,10000])
#plt.title('SREF (Upper Quart.)', y = 1.05, fontsize = 10)
plt.title('NMMB (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
#a.set_yscale('log')
plt.xticks(np.arange(0,1.0001,.2))
plt.yticks(np.arange(0,10000,3000))
plt.tick_params(axis='x',which='both', bottom='off')









### Upper Decile plots
a = plt.axes([.81, .28, .1, .055], axisbg='white')
plt.bar(freq_sref[:,0],freq_sref[:,5],width = .06, color = 'red', edgecolor ='none',align='center')
plt.xlim([-0.04,1.04])
plt.ylim([10,10000])
#plt.title('NCAR (Upper Dec.)', y = 1.05, fontsize = 10)
plt.title('SREF (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
#a.set_yscale('log')

plt.xticks()
plt.xticks(np.arange(0,1.0001,.2))
plt.yticks(np.arange(0,10000,3000))
plt.tick_params(axis='x',which='both', bottom='off')







a = plt.axes([.81, .16, .1, .055], axisbg='white')
plt.bar(freq_arw[:,0],freq_arw[:,5],width = 0.06, color = 'green', edgecolor ='none',align='center')
plt.xlim([-0.04,1.04])
plt.ylim([10,10000])
#plt.title('SREF (Upper Dec.)', y = 1.05, fontsize = 10)
plt.title('ARW (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
#a.set_yscale('log')
plt.xticks()
plt.xticks(np.arange(0,1.0001,.2))
plt.yticks(np.arange(0,10000,3000))
plt.tick_params(axis='x',which='both', bottom='off')



a = plt.axes([.81,.04, .1, .055], axisbg='white')
plt.bar(freq_nmb[:,0],freq_nmb[:,5],width = 0.06, color = 'gold', edgecolor ='none',align='center')
plt.xlim([-0.04,1.04])
plt.ylim([10,10000])
#plt.title('SREF (Upper Dec.)', y = 1.05, fontsize = 10)
plt.title('NMMB (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
#a.set_yscale('log')
plt.xticks()
plt.xticks(np.arange(0,1.0001,.2))
plt.yticks(np.arange(0,10000,3000))
plt.tick_params(axis='x',which='both', bottom='off')




plt.savefig("../../../public_html/reliability_diagram_sref_arw_nmb_interp_bin_10random.pdf")                               
        
#%%        
          
          
          
          
          
          
          
          



