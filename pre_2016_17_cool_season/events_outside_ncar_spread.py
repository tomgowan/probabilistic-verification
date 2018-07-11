
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
    
    
   
Date_event = zeros((183))
Date2= '20151001'
Date = []
x = 4
for i in range(0,183):
    t=time.strptime(Date2,'%Y%m%d')
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(i)
    Date3 = newdate.strftime('%Y%m%d')
    Date_event[i] = int(Date3)
    x = x + 1
    if x == 5:
        Date.append(Date3)
        x = 0
    
  
    






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
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/gefs_precip_12Zto12Z_upperquart_prob_interp.txt",
         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/gefs_precip_12Zto12Z_upperdec_prob_interp.txt"]
         
         
#         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/ncarens_precip_12Zto12Z_upperquart_prob_interp.txt",
#         "/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/snotel_data/ncarens_precip_12Zto12Z_upperdec_prob_interp.txt"]

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

extreme_events = zeros((798,10))
event_date = zeros((183,10)) # To dtermine when the extreme events happened // Dates from oct 2015 to march 2016 going down // same colums as extreme events 
event_date[:,0] = Date_event[:]
extreme_events[:,0:3] = inhouse_data[:,0:3] # col 3: upper quart above spread  
                                            # col 4: upper quart below spread 
                                            # col 5: upper dec above spread
                                            # col 6: upper dec below spread
      
for w in range(len(inhouse_data75[:,0])):
    for x in range(len(ncar_data75[:,0])):
        if inhouse_data75[w,0] == ncar_data75[x,0]:
            num_events = 0
            if inhouse_data75[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                            if inhouse_data75[w,i] == 1:
                                num_events = num_events + 1 ### Determine number of upper quartile events for each station
                            if ncar_data75[x,i] == 0 and inhouse_data75[w,i] == 1:  ## If ncar says 0% chance of upper quart and upper quart occurs (row 3)
                                extreme_events[w,3] = extreme_events[w,3] + 1
                                event_date[i-3,1] = event_date[i-3,1] + 1

                                
                            if ncar_data75[x,i] == 1 and inhouse_data75[w,i] == 0:  ## If ncar says 100% chance of upper quart and upper quart does occur (row 4)
                                extreme_events[w,4] = extreme_events[w,4] + 1
                                event_date[i-3,2] = event_date[i-3,2] + 1
                                
    extreme_events[w,3] = extreme_events[w,3]/num_events*100 #Nomalize so that values are "per 100 events"
    extreme_events[w,4] = extreme_events[w,4]/num_events*100
    extreme_events[w,7] = num_events          


                                

for w in range(len(inhouse_data90[:,0])):
    for x in range(len(ncar_data90[:,0])):
        if inhouse_data90[w,0] == ncar_data90[x,0]:
            num_events = 0
            if inhouse_data90[w,0] != 0:
                for i in range(3,186):
                    if 0 < inhouse_data[w,i] < 1000:
                            if inhouse_data90[w,i] == 1:
                                num_events = num_events + 1 ## Determine number of upper decile events for each station
                            if ncar_data90[x,i] == 0 and inhouse_data90[w,i] == 1:  ## If ncar says 0% chance of upper dec and upper dec occurs (row 5)
                                extreme_events[w,5] = extreme_events[w,5] + 1
                                event_date[i-3,3] = event_date[i-3,3] + 1
                                
                            if ncar_data90[x,i] == 1 and inhouse_data90[w,i] == 0:  ## If ncar says 100% chance of upper dec and upper dec does occur (row 6)
                                extreme_events[w,6] = extreme_events[w,6] + 1
                                event_date[i-3,4] = event_date[i-3,4] + 1
                                
    extreme_events[w,5] = extreme_events[w,5]/num_events*100#Nomalize so that values are "per 100 events"
    extreme_events[w,6] = extreme_events[w,6]/num_events*100
    extreme_events[w,8] = num_events
''' 
### Get location for 20160205 event
xx = 0
lat_lon_20160205 = zeros((100,2))
for w in range(len(inhouse_data75[:,0])):
    for x in range(len(ncar_data75[:,0])):
        if inhouse_data75[w,0] == ncar_data75[x,0]:
            num_events = 0
            if inhouse_data75[w,0] != 0:
                for i in range(127+3,128+3):  #20160205 (+3 becuase of way array is set up)
                    if 0 < inhouse_data[w,i] < 1000:
                            if ncar_data75[x,i] == 0 and inhouse_data75[w,i] == 1:  ## If ncar says 0% chance of upper quart and upper quart occurs (row 3)
                                lat_lon_20160205[xx,:] = ncar_data75[x,1:3]
                                xx = xx + 1
np.savetxt('snotel_locs_20160205_above_quart.txt', lat_lon_20160205)
'''



###############################################################################
########################  Exclude stations without enough data ################   
###############################################################################  


# If upper quart event is <= 2.54, exclude location 
for i in range(798):
    if percentiles[i,1] <= 5.08:
        extreme_events[i,:] = 0
# If less than 10 upper decile exclude region
    if extreme_events[i,8] < 10:
        extreme_events[i,:] = 0
        
        
###############################################################################
################################# Divide into regions #########################   
###############################################################################        
        

regions = np.array([[41.5,49.2, -123.0,-120.5],
                    [37.0,41.0, -121.0,-118.0], 
                    [43.7,46.2, -120.0,-116.8], 
                    [43.0,49.3, -116.8,-112.2], 
                    [41.8,47.0, -112.5,-105.5],
                    [37.2,41.8, -113.9,-109.2],
                    [35.6,41.5, -108.7,-104.5],
                    [32.5,35.5, -113.0,-107.0]])
                    
                    
region_events = zeros((8,4))



for w in range(8):
    count = 0
    for i in range(len(extreme_events[:,0])):
        if regions[w,0] <= extreme_events[i,1] <= regions[w,1] and regions[w,2] <= extreme_events[i,2] <= regions[w,3]:
            count = count + 1
            region_events[w,0:4] = region_events[w,0:4] + extreme_events[i,3:7]
    
    region_events[w,0:4] = region_events[w,0:4]/count
    print count




                 
                                
###############################################################################
################################# Plots #######################################   
###############################################################################                                
                            
        
###### Get elevation data
        
NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/random/wrfinput_d02'
fh = Dataset(NCARens_file, mode='r')

elevation = fh.variables['HGT'][:]
lat_netcdf = fh.variables['XLAT'][:]
long_netcdf = fh.variables['XLONG'][:]      
        
 

###############################################################################
################################## PLOTS ######################################
###############################################################################

cmap = matplotlib.cm.get_cmap('YlOrRd')
fig = plt.figure(figsize=(17,19))



#levels = np.arange(0,21,1)
levels_el = np.arange(0,5000,100)
levels = np.arange(0,50,4)
top = 25
left = 25
tick = 18
info = 14
dots = 100

lat = extreme_events[:,1]
lon = extreme_events[:,2]
above_quart = extreme_events[:,3]
below_quart = extreme_events[:,4]
above_dec = extreme_events[:,5]
below_dec = extreme_events[:,6]

ax = fig.add_subplot(221)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon, lat)
x2, y2 = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
csAVG2 = map.contourf(x2,y2,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)
csAVG = map.scatter(x,y, c = above_quart,  cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N),marker='o', s = dots)  
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")#, ticks = np.arange(0.000001, .80001, 0.10))
cbar.ax.tick_params(labelsize=tick)
plt.title('Above Ensemble Spread', fontsize = top)
#cbar.ax.set_xticklabels(np.arange(0,0.80001,0.10), fontsize = tick)
plt.annotate('Upper Quartile Events', xy=(-0.06, .73),
             xycoords='axes fraction', fontsize = left, rotation = 90)
#plt.annotate('Mean ETS = %1.3f' % np.average(n[:,6], weights=(n[:,6] >-5)), xy=(0.013, .013),
#             xycoords='axes fraction', fontsize = info)




ax = fig.add_subplot(222)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon, lat)
x2, y2 = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
csAVG2 = map.contourf(x2,y2,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)
csAVG = map.scatter(x,y, c = below_quart, cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), marker='o',  s = dots)  
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")#, ticks = np.arange(0.000001, .80001, 0.1))
cbar.ax.tick_params(labelsize=tick)
plt.title('Below Ensemble Spread', fontsize = top)
#cbar.ax.set_xticklabels(np.arange(0,0.80001,0.1), fontsize = tick)
#plt.annotate('Mean ETS = %1.3f' % np.average(h[:,6], weights=(h[:,6] >-5)), xy=(0.013, .013),
#             xycoords='axes fraction', fontsize = info)


#levels = np.arange(0,24,4)

ax = fig.add_subplot(223)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon, lat)
x2, y2 = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
csAVG2 = map.contourf(x2,y2,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)
csAVG = map.scatter(x,y, c = above_dec,  cmap=cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), marker='o', s = dots)  
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")#, ticks = np.arange(0.000001, .80001, 0.1))
cbar.ax.tick_params(labelsize=tick)
#cbar.ax.set_xticklabels(np.arange(0,0.80001,0.1), fontsize = tick)
plt.annotate('Upper Decile Events', xy=(-0.06, .73),
             xycoords='axes fraction', fontsize = left, rotation = 90)
cbar.ax.set_xlabel('Occurances per 100 Events', fontsize = top)
#plt.annotate('Mean ETS = %1.3f' % np.average(n4[:,6], weights=(n4[:,6] >-5)), xy=(0.013, .013),
#             xycoords='axes fraction', fontsize = info)


             
             

ax = fig.add_subplot(224)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lon, lat)
x2, y2 = map(long_netcdf[0,:,:], lat_netcdf[0,:,:])
csAVG2 = map.contourf(x2,y2,elevation[0,:,:], levels_el, cmap = 'Greys', zorder = 0)
csAVG = map.scatter(x,y, c = below_dec, cmap=cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), marker='o', s = dots)  
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")#, ticks = np.arange(0.000001, .80001, 0.1))
cbar.ax.tick_params(labelsize=tick)
cbar.ax.set_xlabel('Occurances per 100 Events', fontsize = top)
#cbar.ax.set_xticklabels(np.arange(0,0.80001,0.1), fontsize = tick)
#plt.annotate('Mean ETS = %1.3f' % np.average(n12[:,6], weights=(n12[:,6] >-5)), xy=(0.013, .013),
#             xycoords='axes fraction', fontsize = info)




plt.tight_layout()
plt.savefig("../../public_html/events_outside_gefs_spread_interp_gefs.png")                               
      
          


#######   Plot date that each extreme event occured #######
x = np.arange(0,183)


    

fig1=plt.figure(num=None, figsize=(15,20), dpi=500)
fig1.subplots_adjust(hspace=.4)

ax1 = fig1.add_subplot(411)
ax1.bar(x,event_date[:,1], edgecolor = 'none', width = 1)
plt.xlim([0,183])
plt.xticks(np.arange(0,183,5))
ax1.set_xticklabels(Date, rotation = 45)
plt.grid(True)
plt.title('Upper Quartile Above Spread', fontsize = 22)
plt.ylabel('Number of Locations', fontsize = 18)



ax1 = fig1.add_subplot(412)
ax1.bar(x,event_date[:,2], edgecolor = 'none', width = 1, color = 'green')
plt.xlim([0,183])
plt.xticks(np.arange(0,183,5))
ax1.set_xticklabels(Date, rotation = 45)
plt.grid(True)
plt.title('Upper Quartile Below Spread', fontsize = 22)
plt.ylabel('Number of Locations', fontsize = 18)



ax1 = fig1.add_subplot(413)
ax1.bar(x,event_date[:,3], edgecolor = 'none', width = 1, color = 'red')
plt.xlim([0,183])
plt.xticks(np.arange(0,183,5))
ax1.set_xticklabels(Date, rotation = 45)
plt.grid(True)
plt.title('Upper Decile Above Spread', fontsize = 22)
plt.ylabel('Number of Locations', fontsize = 18)




ax1 = fig1.add_subplot(414)
ax1.bar(x,event_date[:,4], edgecolor = 'none', width = 1, color = 'gold')
plt.xlim([0,183])
plt.xticks(np.arange(0,183,5))
ax1.set_xticklabels(Date, rotation = 45)
plt.grid(True)
plt.title('Upper Decile Below Spread', fontsize = 22)
plt.ylabel('Number of Locations', fontsize = 18)



plt.savefig("../../public_html/events_outsdie_spread_date_interp_gefs.png")
          
          
     
          
          
#######   Bar graph for regional plot
          
          
          
          
N = len(region_events[:,0])
x = range(N)
region = ['Pacific\nNorthwest', 'Sierra\nNevada','Blue Mtns,\nOR','Idaho/\nWestern MT','NW\nWyoming','Utah','Colorado']   
#title = ['Above Spread (Upper Quart.)', 'Below Spread (Upper Quart.)','Above Spread (Upper Dec.)', 'Below Spread (Upper Dec.)']
sub = 221

fig = plt.figure(figsize=(16,12))


for i in range(4):
    ax1 = fig.add_subplot(sub)
    #ax1.set_title(title[i], fontsize = 20)
    ax1.bar(x,region_events[:,i],width = 1, color = ['blue', 'green', 'red', 'c', 'y', 'darkred', 'purple', 'salmon'], edgecolor ='none', alpha = 0.8)
    if sub == 221:
        ax1.set_ylabel('Upper Quartile Events', fontsize = 25)
    if sub == 222:
        ax1.set_title('Below Ensemble Spread', fontsize = 25)
    if sub == 223:
        ax1.set_ylabel('Upper Decile Events', fontsize = 25)
    if sub == 221:
        ax1.set_title('Above Ensemble Spread', fontsize = 25)

    ax1.set_yticks(np.arange(0,60.001,10))
    ax1.set_yticklabels(np.arange(0,61,10), fontsize = 16)
    plt.tight_layout()
    ax1.set_xticks([.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    if sub == 223 or sub == 224:
        ax1.set_xticklabels((region), fontsize  = 16, rotation = 50)
    else:
        ax1.set_xticklabels(('', '', '', ''))
    plt.ylim([0.5999,1.5])
    ax1.yaxis.grid()
    plt.ylim([0,60])
    sub = sub + 1 
    
plt.savefig("../../public_html/extreme_events_by_region_interp_gefs.pdf")
plt.show()           



