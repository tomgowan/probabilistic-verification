
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, maskoceans
import pygrib, os, sys
from netCDF4 import Dataset
from numpy import *
import numpy as np
from pylab import *
import time
from datetime import date, timedelta
import pyart
from matplotlib import animation
import matplotlib.animation as animation
import types



WM = [-117.0, 43.0, -108.5, 49.0]
UT = [-114.7, 36.7, -108.9, 42.5]
CO = [-110.0, 36.0, -104.0, 41.9]
NU = [-113.4, 39.5, -110.7, 41.9]
NW = [-125.3, 42.0, -116.5, 49.1]
WE = [-125.3, 31.0, -102.5, 49.2]
US = [-125, 24.0, -66.5, 49.5]
SN = [-123.5, 33.5, -116.0, 41.0]

region = sys.argv[1]

if region == 'WM':
    latlon = WM
    
if region == 'US':
    latlon = US
    
if region == 'UT':
    latlon = UT
    
if region == 'CO':
    latlon = CO
    
if region == 'NU':
    latlon = NU
    
if region == 'NW':
    latlon = NW

if region == 'WE':
    latlon = WE

if region == 'SN':
    latlon = SN

'''
Date2= '20150930'
Date = zeros((184))
num_days = 184

for i in range(0,num_days):
    t=time.strptime(Date2,'%Y%m%d')
    newdate=date(t.tm_year,t.tm_mon,t.tm_mday)+timedelta(i)
    Date3 = newdate.strftime('%Y%m%d')
    Date[i] = int(Date3)  






###############################################################################
##############   Read in ncar  and prism precip   #############################
###############################################################################



for i in range(0,num_days-1):
    totalprecip = zeros((10,985,1580))
    x = 0 


    #### Make sure all ncar and prism files are present
    for mem in range(1,11):
        for j in range(13,37):
            NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/ncarens/%08d00' % Date[i] + '/ncar_3km_%08d00' % Date[i] + '_mem%d' % mem + '_f0%02d' % j + '.grb2'
            if os.path.exists(NCARens_file):
                x = x + 1



    print x
    if x == 240:
            for mem in range(1,11):
                for j in range(13,37): #12Z to 12Z
                    NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/ncarens/%08d00' % Date[i] + '/ncar_3km_%08d00' % Date[i] + '_mem%d' % mem + '_f0%02d' % j + '.grb2'
                    print NCARens_file
                    
                    grbs = pygrib.open(NCARens_file)
                    grb = grbs.select(name='Total Precipitation')[0]
                    
                    lat_ncar,lon_ncar = grb.latlons()
                    
                    
                    tmpmsgs = grbs.select(name='Total Precipitation')
                    
                    msg = grbs[16]
                    precip_vals = msg.values
                    precip_vals = precip_vals*0.0393689*25.4
                    totalprecip[mem-1,:,:] = totalprecip[mem-1,:,:] + precip_vals 


    sdev = zeros((len(totalprecip[0,:,0]),len(totalprecip[0,0,:]))) 
    days_count = zeros((len(totalprecip[0,:,0]),len(totalprecip[0,0,:])))
   
    for i in range(len(totalprecip[0,:,0])):
        for j in range(len(totalprecip[0,0,:])):
            print i
            if np.mean(totalprecip[:,i,j]) >= 2.54:
                days_count[i,j] = days_count[i,j] + 1
                sdev[i,j] = sdev[i,j] + (np.std(totalprecip[:,i,j])/np.mean(totalprecip[:,i,j]))  #Coeffcient of variation
    
sdev = sdev/days_count
np.savetxt('ens_spread_sdev.txt', sdev)

'''
sdev = np.loadtxt('ens_spread_sdev.txt')
###############################################################################
########################   Plot   #############################################
###############################################################################

'''
fig = plt.figure(figsize=(35,17))
cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
levels = [0.1, 0.5, 1,1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42,46,50,55,60, 65]
locs = np.loadtxt('snotel_locs_20160205_above_quart.txt')

########################   NCAR Ensemble All Members   ########################
for mem in range(1,11):
    ax = fig.add_subplot(2,5,mem)
    map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
    x, y = map(lon_ncar, lat_ncar)
    xloc, yloc = map(locs[:,1], locs[:,0])
    precip_ncar = maskoceans(lon_ncar, lat_ncar, totalprecip[mem-1,:,:])
    csAVG = map.contourf(x,y,precip_ncar, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
    map.scatter(xloc, yloc, s = 20, c = 'k')
    map.drawcoastlines(linewidth = .5)
    map.drawstates()
    map.drawcountries()
    cbar = map.colorbar(csAVG, location='bottom', pad="5%")
    cbar.ax.tick_params(labelsize=12)
    plt.title('NCAR Ens. Member %d' % mem, fontsize = 18)
plt.tight_layout()
plt.savefig("../plots/ncar_allmems_%s" % Date + ".png")


'''
cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
levels = np.arange(0,2,.1)
########################  Prism   #####################################
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
x, y = map(lon_ncar, lat_ncar)
precip_tot = maskoceans(lon_ncar, lat_ncar, sdev)
csAVG = map.contourf(x,y,sdev,levels, cmap = cmap)
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('PRISM', fontsize = 18)
plt.tight_layout()
plt.savefig("../../public_html/ens_spread_sdev.png")

