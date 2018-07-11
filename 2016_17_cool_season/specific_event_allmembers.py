
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

'''

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





###############################################################################
##############   Create lat lon grid for psirm    #############################
###############################################################################




lats_prism = zeros((621,1405))
lons_prism = zeros((621,1405))

for i in range(621):
    lats_prism[620-i,:] = 24.062500000000 + i*.0416666666666666666666666667

for i in range(1405):
    lons_prism[:,i] = -125.02083333333333333333 + i*.0416666666666666666666667




###############################################################################
##############   Read in ncar  and prism precip   #############################
###############################################################################
Date = '20160205'
Date_ncar = '20160204'
totalprecip = zeros((10,985,1580))
z = 0
x = 0 


    #### Make sure all ncar and prism files are present
for mem in range(1,11):
    for j in range(13,37):
        NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/ncarens/%s00' % Date_ncar + '/ncar_3km_%s00' % Date_ncar + '_mem%d' % mem + '_f0%02d' % j + '.grb2'
        if os.path.exists(NCARens_file):
            x = x + 1
try:
    prism_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_stable_4kmD2_%s' % Date + '_asc.asc'
    if os.path.exists(prism_file):
        z = 1
except:
    pass
    
try:
    prism_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_provisional_4kmD2_%s' % Date + '_asc.asc'
    if os.path.exists(prism_file):
        z = 1
except:
    pass


print x
if x == 240 and z == 1:
        for mem in range(1,11):
            for j in range(13,37):
                NCARens_file = '/uufs/chpc.utah.edu/common/home/steenburgh-group5/cstar/ncarens/%s00' % Date_ncar + '/ncar_3km_%s00' % Date_ncar + '_mem%d' % mem + '_f0%02d' % j + '.grb2'
                print NCARens_file

                grbs = pygrib.open(NCARens_file)
                grb = grbs.select(name='Total Precipitation')[0]

                lat_ncar,lon_ncar = grb.latlons()


                tmpmsgs = grbs.select(name='Total Precipitation')

                msg = grbs[16]
                precip_vals = msg.values
                precip_vals = precip_vals*0.0393689*25.4
                totalprecip[mem-1,:,:] = totalprecip[mem-1,:,:] + precip_vals 

        precip_tot = zeros((621,1405))
        ############### Prism #####################################
        try:
            precip = np.loadtxt("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_stable_4kmD2_%s" % Date + "_asc.asc", skiprows = 6)
        except:
            print(prism_file)
        try:
            precip = np.loadtxt("/uufs/chpc.utah.edu/common/home/steenburgh-group5/tom/climatology/prism/PRISM_ppt_provisional_4kmD2_%s" % Date + "_asc.asc", skiprows = 6)
        except:
            print(prism_file)
        
        precip_tot = precip_tot + precip
    
 
'''


###############################################################################
########################   Plot   #############################################
###############################################################################


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




########################  Prism   #####################################
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
x, y = map(lons_prism, lats_prism)
xloc, yloc = map(locs[:,1], locs[:,0])
map.scatter(xloc, yloc, s = 75, c = 'k')
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.scatter(xloc, yloc, s = 20, c = 'k')
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)

plt.title('PRISM', fontsize = 18)
plt.tight_layout()
plt.savefig("../plots/prism_%s" % Date + ".png")

'''
########################   bias (NCAR)   ######################################
ax = fig.add_subplot(333)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
ax.set_title('NCAR/PRISM', fontsize = 18)
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_ncar/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
plt.annotate('Mean bias = %1.3f\n' % bias_mean_ncar  +
             'Mean dry bias (bias < 1) = %1.3f\n' % bias_mean_ncar_low + 
             'Mean wet bias (bias > 1) = %1.3f' % bias_mean_ncar_high, xy=(0.02, .02),
             xycoords='axes fraction', fontsize = 10, backgroundcolor = 'w')







########################   hrrr   #############################################
cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
levels = np.arange(.0001,37,.5)
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]

ax = fig.add_subplot(334)
x, y = map(lons_prism, lats_prism)
precip_hrrr = maskoceans(lons_prism, lats_prism, precip_hrrr)
csAVG = map.contourf(x,y,precip_hrrr, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('HRRR', fontsize = 18)





########################   prism (hrrr)   #####################################
ax = fig.add_subplot(335)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('PRISM', fontsize = 18)




########################   bias (hrrr)   ######################################
ax = fig.add_subplot(336)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_hrrr/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
plt.title('HRRR/PRISM', fontsize = 18)
plt.annotate('Mean bias = %1.3f\n' % bias_mean_hrrr  +
             'Mean dry bias (bias < 1) = %1.3f\n' % bias_mean_hrrr_low + 
             'Mean wet bias (bias > 1) = %1.3f' % bias_mean_hrrr_high, xy=(0.02, .02),
             xycoords='axes fraction', fontsize = 10, backgroundcolor = 'w')








########################   nam4km   #############################################


cmap = matplotlib.cm.get_cmap('pyart_NWSRef')
levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5, 8 ,8.5, 9,9.5, 10,11, 12, 13, 14, 15, 16, 18, 20, 22,26,30,34,38,42]


ax = fig.add_subplot(337)
x, y = map(lons_prism, lats_prism)
precip_nam4k = maskoceans(lons_prism, lats_prism, precip_nam4k)
csAVG = map.contourf(x,y,precip_nam4k, levels, cmap = cmap,norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('NAM4km', fontsize = 18)
cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)






########################   prism (nam4k)   ####################################
ax = fig.add_subplot(338)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
x, y = map(lons_prism, lats_prism)
precip_tot = maskoceans(lons_prism, lats_prism, precip_tot)
csAVG = map.contourf(x,y,precip_tot, levels, cmap = cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N))
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar = map.colorbar(csAVG, location='bottom', pad="5%")
cbar.ax.tick_params(labelsize=12)
plt.title('PRISM', fontsize = 18)
cbar.ax.set_xlabel('Mean Daily Precipitation from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)









########################   bias (nam4k)   #####################################
ax = fig.add_subplot(339)
map = Basemap(projection='merc',llcrnrlon=latlon[0],llcrnrlat=latlon[1],urcrnrlon=latlon[2],urcrnrlat=latlon[3],resolution='i')
cmap=plt.cm.BrBG
levels = [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2, 5]
x, y = map(lons_prism, lats_prism)
csAVG = map.contourf(x,y,precip_nam4k/precip_tot, levels,cmap=cmap, norm=matplotlib.colors.BoundaryNorm(levels,cmap.N), vmin = 0.1, vmax = 5)
map.drawcoastlines(linewidth = .5)
map.drawstates()
map.drawcountries()
cbar.ax.tick_params(labelsize=12)
cbar = map.colorbar(csAVG, location='bottom', pad="5%", ticks= [0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8,2,5])
cbar.ax.set_xticklabels(['<0.5','0.5','0.6', '0.7', '0.8', '0.9', '1', '1.2', '1.4', '1.6', '1.8','2','>2'])
plt.title('NAM4km/PRISM', fontsize = 18)
cbar.ax.set_xlabel('Mean Daily Precipitation Bias from Oct. 2015 to Mar. 2016 (mm)', fontsize = 10)
plt.annotate('Mean bias = %1.3f' % bias_mean_nam4k, xy=(0.01, .01), xycoords='axes fraction', fontsize = 11)
plt.annotate('Mean bias = %1.3f\n' % bias_mean_nam4k +
             'Mean dry bias (bias < 1) = %1.3f\n' % bias_mean_nam4k_low + 
             'Mean wet bias (bias > 1) = %1.3f' % bias_mean_nam4k_high, xy=(0.02, .02),
             xycoords='axes fraction', fontsize = 10, backgroundcolor = 'w')
plt.tight_layout()
plt.savefig("./plots/prism_climo_allmodels%s" % region + ".pdf")
plt.show()










'''

