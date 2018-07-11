# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 14:24:06 2016

@author: u1013082
"""



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
links = ["/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/snotel_precip_2015_2016_qc.csv", 
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/ncarens_precip_12Zto12Z_upperquart_prob_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/ncarens_precip_12Zto12Z_upperdec_prob_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/ecmwf_precip_12Zto12Z_upperquart_prob.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/ecmwf_precip_12Zto12Z_upperdec_prob.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/ecmwf_10mem_precip_12Zto12Z_upperquart_prob.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/ecmwf_10mem_precip_12Zto12Z_upperdec_prob.txt"]

#data = ['inhouse_data', 'ncar_data', 'nam4k_data', 'hrrr_data', 'nam12k_data']        
data = zeros((7,798,186))

         
for c in range(7):
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
ec_data75 = data[3,:,:] 
ec_data90 = data[4,:,:]
ec10_data75 = data[5,:,:] 
ec10_data90 = data[6,:,:]  


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
freq = zeros((11,20))
freq[:,0] = np.arange(0,1.001,.1)
freq_ec = zeros((51,20))
freq_ec[:,0] = np.arange(0,1.001,.02)
freq10_ec = zeros((11,20))
freq10_ec[:,0] = np.arange(0,1.001,.1)

      
for w in range(len(inhouse_data75[:,0])):  # NCAR 75
    for x in range(len(ncar_data75[:,0])):
        if inhouse_data75[w,0] == ncar_data75[x,0]:
            if inhouse_data75[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(11):
                            if ncar_data75[x,i] == t/10.:
                                freq[t,1] = freq[t,1] + inhouse_data75[w,i]
                                freq[t,2] = freq[t,2] + 1
                                

for w in range(len(inhouse_data90[:,0])):  # NCAR 90
    for x in range(len(ncar_data90[:,0])):
        if inhouse_data90[w,0] == ncar_data90[x,0]:
            if inhouse_data90[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(11):
                            if ncar_data90[x,i] == t/10.:
                                freq[t,4] = freq[t,4] + inhouse_data90[w,i]
                                freq[t,5] = freq[t,5] + 1
                                

for w in range(len(inhouse_data75[:,0])):  # EC 75
    for x in range(len(ec_data75[:,0])):
        if inhouse_data75[w,0] == ec_data75[x,0]:
            if inhouse_data75[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(51):
                            if ec_data75[x,i] == t/50.:
                                freq_ec[t,1] = freq_ec[t,1] + inhouse_data75[w,i]
                                freq_ec[t,2] = freq_ec[t,2] + 1
                                

for w in range(len(inhouse_data90[:,0])):  # EC 90
    for x in range(len(ec_data90[:,0])):
        if inhouse_data90[w,0] == ec_data90[x,0]:
            if inhouse_data90[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(51):
                            if ec_data90[x,i] == t/50.:
                                freq_ec[t,4] = freq_ec[t,4] + inhouse_data90[w,i]
                                freq_ec[t,5] = freq_ec[t,5] + 1
                                
                                
                                
                                
for w in range(len(inhouse_data75[:,0])):  # EC10mem 75
    for x in range(len(ec10_data75[:,0])):
        if inhouse_data75[w,0] == ec_data75[x,0]:
            if inhouse_data75[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(11):
                            if ec10_data75[x,i] == t/10.:
                                freq10_ec[t,1] = freq10_ec[t,1] + inhouse_data75[w,i]
                                freq10_ec[t,2] = freq10_ec[t,2] + 1
                                

for w in range(len(inhouse_data90[:,0])):  # EC10mem 90
    for x in range(len(ec10_data90[:,0])):
        if inhouse_data90[w,0] == ec_data90[x,0]:
            if inhouse_data90[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                        for t in range(11):
                            if ec10_data90[x,i] == t/10.:
                                freq10_ec[t,4] = freq10_ec[t,4] + inhouse_data90[w,i]
                                freq10_ec[t,5] = freq10_ec[t,5] + 1
                                
                                
                                
                                
                                
freq[:,3] = freq[:,1]/freq[:,2]      # NCAR 75     
freq[:,6] = freq[:,4]/freq[:,5]      # NCAR 90    
freq_ec[:,3] = freq_ec[:,1]/freq_ec[:,2]      # EC 75     
freq_ec[:,6] = freq_ec[:,4]/freq_ec[:,5]      # EC 90
freq10_ec[:,3] = freq10_ec[:,1]/freq10_ec[:,2]      # EC10 75     
freq10_ec[:,6] = freq10_ec[:,4]/freq10_ec[:,5]      # EC10 90




###############################################################################
############ BSS and BS   #####################################################   
###############################################################################   


####### rows are BSSquart//BSquart//BSSdec//BSdec
bss= zeros((11,10))


### bssf rows are BS, reliabilty, resolution, uncertatinty, BSS 
bssf = zeros((10,5))
obquart = (sum(freq[:,1])/sum(freq[:,2]))
obdec = (sum(freq[:,4])/sum(freq[:,5]))

Bref75 = (obquart)*(1-obquart)

Bref90 = (obdec)*(1-obdec)

for i in range(len(freq[:,0])):
    ### BS
    bss[i,0] =((((freq[i,0]-0)**2)*(freq[i,2]-freq[i,1]))+(((freq[i,0]-1)**2)*(freq[i,1])))
    bss[i,1] =((((freq[i,0]-0)**2)*(freq[i,5]-freq[i,4]))+(((freq[i,0]-1)**2)*(freq[i,4])))
    
    ### Reliability
    bss[i,2] = (freq[i,2]*(freq[i,0]-freq[i,3])**2)
    bss[i,3] = (freq[i,5]*(freq[i,0]-freq[i,6])**2)
    
    
    ### Resolution
    bss[i,4] = (freq[i,2]*(freq[i,3]-obquart)**2)
    bss[i,5] = (freq[i,5]*(freq[i,6]-obdec)**2)
    
    ### Uncertainty is same as BSref (climatology)



##### Upper quartile
bssf[0,0] = sum(bss[:,0])/sum(freq[:,2])
bssf[1,0] = sum(bss[:,2])/sum(freq[:,2])
bssf[2,0] = sum(bss[:,4])/sum(freq[:,2])
bssf[3,0] = Bref75
bssf[4,0] = 1-(bssf[0,0]/Bref75) 

#### Upper decile 
bssf[0,1] = sum(bss[:,1])/sum(freq[:,5])
bssf[1,1] = sum(bss[:,3])/sum(freq[:,5])
bssf[2,1] = sum(bss[:,5])/sum(freq[:,2])
bssf[3,1] = Bref90
bssf[4,1] = 1-(bssf[0,1]/Bref90)







#####  For ECMWF #######
 
####### rows are BSSquart//BSquart//BSSdec//BSdec
bss_ec= zeros((51,10))


### bssf rows are BS, reliabilty, resolution, uncertatinty, BSS 
bssf_ec = zeros((10,5))
obquart = (sum(freq_ec[:,1])/sum(freq_ec[:,2]))
obdec = (sum(freq_ec[:,4])/sum(freq_ec[:,5]))

Bref75 = (obquart)*(1-obquart)

Bref90 = (obdec)*(1-obdec)

for i in range(len(freq_ec[:,0])):
    ### BS
    bss_ec[i,0] =((((freq_ec[i,0]-0)**2)*(freq_ec[i,2]-freq_ec[i,1]))+(((freq_ec[i,0]-1)**2)*(freq_ec[i,1])))
    bss_ec[i,1] =((((freq_ec[i,0]-0)**2)*(freq_ec[i,5]-freq_ec[i,4]))+(((freq_ec[i,0]-1)**2)*(freq_ec[i,4])))
    
    ### Reliability
    bss_ec[i,2] = (freq_ec[i,2]*(freq_ec[i,0]-freq_ec[i,3])**2)
    bss_ec[i,3] = (freq_ec[i,5]*(freq_ec[i,0]-freq_ec[i,6])**2)
    
    
    ### Resolution
    bss_ec[i,4] = (freq_ec[i,2]*(freq_ec[i,3]-obquart)**2)
    bss_ec[i,5] = (freq_ec[i,5]*(freq_ec[i,6]-obdec)**2)
    
    ### Uncertainty is same as BSref (climatology)



##### Upper quartile
bssf_ec[0,0] = sum(bss_ec[:,0])/sum(freq_ec[:,2])
bssf_ec[1,0] = sum(bss_ec[:,2])/sum(freq_ec[:,2])
bssf_ec[2,0] = sum(bss_ec[:,4])/sum(freq_ec[:,2])
bssf_ec[3,0] = Bref75
bssf_ec[4,0] = 1-(bssf_ec[0,0]/Bref75) 

#### Upper decile 
bssf_ec[0,1] = sum(bss_ec[:,1])/sum(freq_ec[:,5])
bssf_ec[1,1] = sum(bss_ec[:,3])/sum(freq_ec[:,5])
bssf_ec[2,1] = sum(bss_ec[:,5])/sum(freq_ec[:,2])
bssf_ec[3,1] = Bref90
bssf_ec[4,1] = 1-(bssf_ec[0,1]/Bref90)



### for 10 member EC


####### rows are BSSquart//BSquart//BSSdec//BSdec
bss10_ec= zeros((11,10))


### bssf rows are BS, reliabilty, resolution, uncertatinty, BSS 
bssf10_ec = zeros((10,5))
obquart = (sum(freq10_ec[:,1])/sum(freq10_ec[:,2]))
obdec = (sum(freq10_ec[:,4])/sum(freq10_ec[:,5]))

Bref75 = (obquart)*(1-obquart)

Bref90 = (obdec)*(1-obdec)

for i in range(len(freq10_ec[:,0])):
    ### BS
    bss10_ec[i,0] =((((freq10_ec[i,0]-0)**2)*(freq10_ec[i,2]-freq10_ec[i,1]))+(((freq10_ec[i,0]-1)**2)*(freq10_ec[i,1])))
    bss10_ec[i,1] =((((freq10_ec[i,0]-0)**2)*(freq10_ec[i,5]-freq10_ec[i,4]))+(((freq10_ec[i,0]-1)**2)*(freq10_ec[i,4])))
    
    ### Reliability
    bss10_ec[i,2] = (freq10_ec[i,2]*(freq10_ec[i,0]-freq10_ec[i,3])**2)
    bss10_ec[i,3] = (freq10_ec[i,5]*(freq10_ec[i,0]-freq10_ec[i,6])**2)
    
    
    ### Resolution
    bss10_ec[i,4] = (freq10_ec[i,2]*(freq10_ec[i,3]-obquart)**2)
    bss10_ec[i,5] = (freq10_ec[i,5]*(freq10_ec[i,6]-obdec)**2)
    
    ### Uncertainty is same as BSref (climatology)



##### Upper quartile
bssf10_ec[0,0] = sum(bss10_ec[:,0])/sum(freq10_ec[:,2])
bssf10_ec[1,0] = sum(bss10_ec[:,2])/sum(freq10_ec[:,2])
bssf10_ec[2,0] = sum(bss10_ec[:,4])/sum(freq10_ec[:,2])
bssf10_ec[3,0] = Bref75
bssf10_ec[4,0] = 1-(bssf10_ec[0,0]/Bref75) 

#### Upper decile 
bssf10_ec[0,1] = sum(bss10_ec[:,1])/sum(freq10_ec[:,5])
bssf10_ec[1,1] = sum(bss10_ec[:,3])/sum(freq10_ec[:,5])
bssf10_ec[2,1] = sum(bss10_ec[:,5])/sum(freq10_ec[:,2])
bssf10_ec[3,1] = Bref90
bssf10_ec[4,1] = 1-(bssf10_ec[0,1]/Bref90)



            
                        
                                
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
a = ax1.plot(freq[:,0],freq[:,3], linewidth = 2, c = 'blue', marker = "o", markeredgecolor = 'none')
ax1.plot(freq_ec[:,0],freq_ec[:,3], linewidth = 2, c = 'g',marker = "o", markeredgecolor = 'none')
ax1.plot(freq10_ec[:,0],freq10_ec[:,3], linewidth = 2, c = 'gold',marker = "o", markeredgecolor = 'none')
c = ax1.plot(freq[:,0],freq[:,0], linewidth = 2, c = 'k', markeredgecolor = 'none')
d = ax1.plot(freq_fill,no_res, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
e = ax1.plot(freq_fill,no_skill, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
plt.text(.53,.7,'Perfect Reliability',rotation = 45)
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
                                ('%.3f' % bssf10_ec[0,0],'%.3f' % bssf10_ec[1,0],'%.3f' % bssf10_ec[2,0],'%.3f' % bssf10_ec[3,0],'%.3f' % bssf10_ec[4,0]),
                                ('%.3f' % bssf_ec[0,0],'%.3f' % bssf_ec[1,0],'%.3f' % bssf_ec[2,0],'%.3f' % bssf_ec[3,0],'%.3f' % bssf_ec[4,0]),
                                ('%.3f' % bssf[0,1],'%.3f' % bssf[1,1],'%.3f' % bssf[2,1],'%.3f' % bssf[3,1],'%.3f' % bssf[4,1]),
                                ('%.3f' % bssf10_ec[0,1],'%.3f' % bssf10_ec[1,1],'%.3f' % bssf10_ec[2,1],'%.3f' % bssf10_ec[3,1],'%.3f' % bssf10_ec[4,1]),
                                ('%.3f' % bssf_ec[0,1],'%.3f' % bssf_ec[1,1],'%.3f' % bssf_ec[2,1],'%.3f' % bssf_ec[3,1],'%.3f' % bssf_ec[4,1])],

                                
          rowLabels=["NCAR \n(Upper Quart.)","ECMWF 10 \n(Upper Quart.)","ECMWF 50 \n(Upper Quart.)","NCAR \n(Upper Dec.)","ECMWF 10 \n(Upper Dec.)","ECMWF 50 \n(Upper Dec.)"],
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

                    

blue_line = mlines.Line2D([],[] , color='blue',
                           label='NCAR (10 Mem.)',  linewidth = 2,marker = "o", markeredgecolor = 'none')
green_line = mlines.Line2D([],[] , color='g',
                           label='ECMWF (50 Mem.)',  linewidth = 2,marker = "o", markeredgecolor = 'none')
gold_line = mlines.Line2D([],[] , color='gold',
                           label='ECMWF (10 Mem.)',  linewidth = 2,marker = "o", markeredgecolor = 'none')
plt.legend(handles=[ blue_line, gold_line,green_line], loc = "lower right",prop={'size':10.5})
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
b = ax2.plot(freq[:,0],freq[:,6], linewidth = 2, c = 'blue',marker = "o", markeredgecolor = 'none')
ax2.plot(freq_ec[:,0],freq_ec[:,6], linewidth = 2, c = 'g',marker = "o", markeredgecolor = 'none')
ax2.plot(freq10_ec[:,0],freq10_ec[:,6], linewidth = 2, c = 'gold',marker = "o", markeredgecolor = 'none')
c = ax2.plot(freq[:,0],freq[:,0], linewidth = 2, c = 'k', markeredgecolor = 'none')
d = ax2.plot(freq_fill,no_res, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
e = ax2.plot(freq_fill,no_skill, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
plt.text(.75,.925,'Perfect Reliability',rotation = 45)
plt.text(.3,.11,'No Resolution (Climatology)',rotation = 0)
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

#red_line = mlines.Line2D([], [], color='red',
#                           label='HRRR',  linewidth = 2,marker = "o", markeredgecolor = 'none')
#cyan_line = mlines.Line2D([], [], color='c',
#                           label='NAM-12km', linewidth = 2,marker = "o", markeredgecolor = 'none')
plt.legend(handles=[blue_line, gold_line, green_line], loc = "lower right",prop={'size':10.5})
plt.title('Upper Decile 24-Hour Precipitation Events', fontsize = 20)
plt.ylabel('Observed Relative Frequency', fontsize = 15)
plt.xlabel('Forecast Probability',fontsize = 15)





################## Sample frequency bar graphs ################################

a = plt.axes([.15, .28, .1, .055], axisbg='white')
plt.bar(freq[:,0],freq[:,2],width = .08, color = 'blue', edgecolor ='none', align='center')
plt.xlim([-0.04,1.04])
plt.title('NCAR (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Probab.',fontsize = 10)
plt.xticks(np.arange(0,1.0001,.2))
a.set_yscale('log')
plt.tick_params(axis='x',which='both', bottom='off')


a = plt.axes([.15, .16, .1, .055], axisbg='white')
plt.bar(freq10_ec[:,0],freq10_ec[:,2],width = .08, color = 'gold', edgecolor ='none', align='center')
plt.xlim([-0.04,1.04])
plt.title('ECMWF 10 (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
a.set_yscale('log')
plt.xticks(np.arange(0,1.0001,.2))
plt.tick_params(axis='x',which='both', bottom='off')


a = plt.axes([.15, .04, .1, .055], axisbg='white')
plt.bar(freq_ec[:,0],freq_ec[:,2],width = .015, color = 'g', edgecolor ='none', align='center')
plt.xlim([-0.01,1.01])
plt.title('ECMWF 50 (Upper Quart.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
a.set_yscale('log')
plt.xticks(np.arange(0,1.0001,.2))
plt.tick_params(axis='x',which='both', bottom='off')












### Upper Decile plots
a = plt.axes([.8, .28, .1, .055], axisbg='white')  # NCAR
plt.bar(freq[:,0],freq[:,5],width = .08, color = 'blue', edgecolor ='none',align='center')
plt.xlim([-0.04,1.04])
plt.title('NCAR (Upper Dec.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
a.set_yscale('log')

plt.xticks()
plt.xticks(np.arange(0,1.0001,.2))
plt.tick_params(axis='x',which='both', bottom='off')





a = plt.axes([.8, .16, .1, .055], axisbg='white')   ## EC 10
plt.bar(freq10_ec[:,0],freq10_ec[:,5],width = .08, color = 'gold', edgecolor ='none',align='center')
plt.xlim([-0.04,1.04])
plt.title('ECMWF 10 (Upper Dec.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
a.set_yscale('log')
plt.xticks()
plt.xticks(np.arange(0,1.0001,.2))
plt.tick_params(axis='x',which='both', bottom='off')




a = plt.axes([.8, .04, .1, .055], axisbg='white')   ### EC 50
plt.bar(freq_ec[:,0],freq_ec[:,5],width = .015, color = 'g', edgecolor ='none',align='center')
plt.xlim([-0.01,1.01])
plt.title('ECMWF 50 (Upper Dec.)', y = 1.05, fontsize = 10)
plt.ylabel('Num. Samples', fontsize = 10)
plt.xlabel('Forecast Prob.',fontsize = 10)
a.set_yscale('log')
plt.xticks()
plt.xticks(np.arange(0,1.0001,.2))
plt.tick_params(axis='x',which='both', bottom='off')











plt.savefig("../plots/reliability_diagram_ec_ncar_interp.pdf")                               
        
          
          
          
          
          
          
          
          
          



