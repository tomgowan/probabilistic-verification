
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




###### Load Variables #######
#Frequency data
freq_int = np.load('sref_arw_percentile_prob_int_non_bias_corrected.npy')#, freq)
freq_sref_bin_int = np.load('sref_nmb_percentile_prob_int_non_bias_corrected.npy')#, freq_sref_bin)
#Skill Score data
bssf_int = np.load('sref_arw_percentile_prob_bss_int_non_bias_corrected.npy')#, bssf)
bssf_sref_int = np.load('sref_nmb_percentile_prob_bss_int_non_bias_corrected.npy')#, bssf_sref)
##Uncertainty data
#pr_con75_int = np.load('sref_arw_sref_nmb_percentile_prob_uncertainty_con75_int.npy')#, pr_con75)
#pr_unc75_int = np.load('sref_arw_sref_nmb_percentile_prob_uncertainty_unc75_int.npy')#, pr_unc75)
#pr_con90_int = np.load('sref_arw_sref_nmb_percentile_prob_uncertainty_con90_int.npy')#, pr_con90)
#pr_unc90_int = np.load('sref_arw_sref_nmb_percentile_prob_uncertainty_unc90_int.npy')#, pr_unc90)


#Frequency data
freq_pac = np.load('sref_arw_percentile_prob_pac_non_bias_corrected.npy')#, freq)
freq_sref_bin_pac = np.load('sref_nmb_percentile_prob_pac_non_bias_corrected.npy')#, freq_sref_bin)
#Skill Score data
bssf_pac = np.load('sref_arw_percentile_prob_bss_pac_non_bias_corrected.npy')#, bssf)
bssf_sref_pac = np.load('sref_nmb_percentile_prob_bss_pac_non_bias_corrected.npy')#, bssf_sref)
##Uncertainty data
#pr_con75_pac = np.load('sref_arw_sref_nmb_percentile_prob_uncertainty_con75_pac.npy')#, pr_con75)
#pr_unc75_pac = np.load('sref_arw_sref_nmb_percentile_prob_uncertainty_unc75_pac.npy')#, pr_unc75)
#pr_con90_pac = np.load('sref_arw_sref_nmb_percentile_prob_uncertainty_con90_pac.npy')#, pr_con90)
#pr_unc90_pac = np.load('sref_arw_sref_nmb_percentile_prob_uncertainty_unc90_pac.npy')#, pr_unc90)
#
#





                                
###############################################################################
################################# Upper 85 ####################################
###############################################################################  
                              
                                
linecolor = ['green', 'gold']                               
fig=plt.figure(num=None, figsize=(18,12), dpi=500, facecolor='w', edgecolor='k')
no_res = np.full((21),.15)
no_skill = np.arange(.075,.575001,.025)
freq_fill = np.arange(0,1.0001,0.05)


################  Pac 85#############################################



ax1 = fig.add_subplot(121)
ax1.text(0.7, 1.03, '$\mathregular{85^{th}}$ Percentile Events (Non Bias Corrected)', fontsize = 33)

fig.subplots_adjust(bottom=0.4)
plt.gca().set_color_cycle(linecolor)
a = ax1.plot(freq_pac[:,0],freq_pac[:,3], linewidth = 2, c = 'green', marker = "o", markeredgecolor = 'none')

#ax1.errorbar(freq_pac[:,0],freq_pac[:,3], yerr= [abs(pr_unc75_pac[0,:,1]-freq_pac[:,3]), abs(pr_unc75_pac[0,:,2]-freq_pac[:,3])], c = 'green')
#ax1.errorbar(freq_sref_bin_pac[:,0],freq_sref_bin_pac[:,3], yerr= [abs(pr_unc75_pac[1,:,1]-freq_sref_bin_pac[:,3]), abs(pr_unc75_pac[1,:,2]-freq_sref_bin_pac[:,3])], c = 'gold')
#
#ax1.errorbar(freq_pac[:,0]-0.003,freq_pac[:,0]-0.003, yerr= [abs(pr_con75_pac[0,:,1]-freq_pac[:,0]), abs(pr_con75_pac[0,:,2]-freq_pac[:,0])], c = 'green')
#ax1.errorbar(freq_sref_bin_pac[:,0]+0.003,freq_sref_bin_pac[:,0]+0.003, yerr= [abs(pr_con75_pac[1,:,1]-freq_sref_bin_pac[:,0]), abs(pr_con75_pac[1,:,2]-freq_sref_bin_pac[:,0])], c = 'gold')
ax1.plot(freq_sref_bin_pac[:,0],freq_sref_bin_pac[:,3], linewidth = 2, c = 'gold',marker = "o", markeredgecolor = 'none')

c = ax1.plot(freq_pac[:,0],freq_pac[:,0], linewidth = 2, c = 'k', markeredgecolor = 'none')
d = ax1.plot(freq_fill,no_res, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
e = ax1.plot(freq_fill,no_skill, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
plt.text(.6,.79,'Perfect Reliability',rotation = 45, fontsize = 12)
plt.text(.6,.17,'No Resolution (Climatology)',rotation = 0, fontsize = 12)
plt.text(.6,.435,'No Skill',rotation = 27, fontsize = 12)

ax1.fill_between(freq_fill,no_skill, 1, where=no_skill >= .15,facecolor = 'grey',alpha=0.5)
ax1.fill_between(freq_fill,no_skill, 0, where=no_skill <= .1501, facecolor = 'grey',alpha=0.5)
props = dict(boxstyle='square', facecolor='white', alpha=1)
ax1.text(0.03, 0.92, '(a) Pacific Ranges', fontsize = 25, bbox = props)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(np.arange(0,1.0001,0.1), fontsize = 12)
plt.yticks(np.arange(0,1.0001,0.1), fontsize = 12)
plt.grid(True)


###### Table to show all stats for both models
                

the_table = plt.table(cellText=[('%.5f' % bssf_pac[0,0],'%.5f' % bssf_pac[1,0],'%.5f' % bssf_pac[2,0],'%.5f' % bssf_pac[3,0],'%.5f' % bssf_pac[4,0]),
                                ('%.5f' % bssf_sref_pac[0,0],'%.5f' % bssf_sref_pac[1,0],'%.5f' % bssf_sref_pac[2,0],'%.5f' % bssf_sref_pac[3,0],'%.5f' % bssf_sref_pac[4,0]),
                                ('%.5f' % bssf_int[0,0],'%.5f' % bssf_int[1,0],'%.5f' % bssf_int[2,0],'%.5f' % bssf_int[3,0],'%.5f' % bssf_int[4,0]),
                                ('%.5f' % bssf_sref_int[0,0],'%.5f' % bssf_sref_int[1,0],'%.5f' % bssf_sref_int[2,0],'%.5f' % bssf_sref_int[3,0],'%.5f' % bssf_sref_int[4,0])],

                                
          rowLabels=["SREF ARW\n(Pacific Ranges)","SREF NMMB\n(Pacific Ranges)","SREF ARW\n(Interior Ranges)","SREF NMMB\n(Interior Ranges)"],
          colLabels=["Brier Score","Reliability", "Resolution", "Uncertainty", "Brier Skill Score"],
          loc="center",
          cellLoc = "center",
          rowColours=['lightgrey','lightgrey','lightgrey','lightgrey'],
          colColours=['lightgrey','lightgrey','lightgrey','lightgrey','lightgrey'],
          bbox=[.65,-0.625,1.135,.49],
          edges = 'BRLT')
the_table.auto_set_font_size(False)
the_table.scale(1.05,1.3)
the_table.set_fontsize(12.5)

                    

blue_line = mlines.Line2D([],[] , color='green',
                           label='SREF ARW (13 members)',  linewidth = 2,marker = "o", markeredgecolor = 'none')
red_line = mlines.Line2D([],[] , color='gold',
                           label='SREF NMMB (13 members)',  linewidth = 2,marker = "o", markeredgecolor = 'none')

plt.legend(handles=[ blue_line, red_line], loc = "lower right",prop={'size':12})
plt.ylabel('Observed Relative Frequency', fontsize = 15)
plt.xlabel('Forecast Probability',fontsize = 15)




################ Int 85 #############################################


no_res = np.full((21),.15)
no_skill = np.arange(.075,.575001,.025)
freq_fill = np.arange(0,1.0001,0.05)


ax1 = fig.add_subplot(122)
fig.subplots_adjust(bottom=0.4)
plt.gca().set_color_cycle(linecolor)
a = ax1.plot(freq_int[:,0],freq_int[:,3], linewidth = 2, c = 'green', marker = "o", markeredgecolor = 'none')

#ax1.errorbar(freq_int[:,0],freq_int[:,3], yerr= [abs(pr_unc75_int[0,:,1]-freq_int[:,3]), abs(pr_unc75_int[0,:,2]-freq_int[:,3])], c = 'green')
#ax1.errorbar(freq_sref_bin_int[:,0],freq_sref_bin_int[:,3], yerr= [abs(pr_unc75_int[1,:,1]-freq_sref_bin_int[:,3]), abs(pr_unc75_int[1,:,2]-freq_sref_bin_int[:,3])], c = 'gold')
#
#ax1.errorbar(freq_int[:,0]-0.003,freq_int[:,0]-0.003, yerr= [abs(pr_con75_int[0,:,1]-freq_int[:,0]), abs(pr_con75_int[0,:,2]-freq_int[:,0])], c = 'green')
#ax1.errorbar(freq_sref_bin_int[:,0]+0.003,freq_sref_bin_int[:,0]+0.003, yerr= [abs(pr_con75_int[1,:,1]-freq_sref_bin_int[:,0]), abs(pr_con75_int[1,:,2]-freq_sref_bin_int[:,0])], c = 'gold')
#



ax1.plot(freq_sref_bin_int[:,0],freq_sref_bin_int[:,3], linewidth = 2, c = 'gold',marker = "o", markeredgecolor = 'none')
c = ax1.plot(freq_int[:,0],freq_int[:,0], linewidth = 2, c = 'k', markeredgecolor = 'none')
d = ax1.plot(freq_fill,no_res, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
e = ax1.plot(freq_fill,no_skill, linewidth = 2, c = 'k',  markeredgecolor = 'none', linestyle = 'dashed')
plt.text(.4,.59,'Perfect Reliability',rotation = 45, fontsize = 12)
plt.text(.6,.17,'No Resolution (Climatology)',rotation = 0, fontsize = 12)
plt.text(.6,.435,'No Skill',rotation = 27, fontsize = 12)

ax1.fill_between(freq_fill,no_skill, 1, where=no_skill >= .15,facecolor = 'grey',alpha=0.5)
ax1.fill_between(freq_fill,no_skill, 0, where=no_skill <= .1501, facecolor = 'grey',alpha=0.5)
props = dict(boxstyle='square', facecolor='white', alpha=1)
ax1.text(0.03, 0.92, '(b) Interior Ranges', fontsize = 25, bbox = props)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xticks(np.arange(0,1.0001,0.1), fontsize = 12)
plt.yticks(np.arange(0,1.0001,0.1), fontsize = 12)
plt.grid(True)
         

blue_line = mlines.Line2D([],[] , color='green',
                           label='SREF ARW (13 members)',  linewidth = 2,marker = "o", markeredgecolor = 'none')
red_line = mlines.Line2D([],[] , color='gold',
                           label='SREF NMMB (13 members)',  linewidth = 2,marker = "o", markeredgecolor = 'none')

plt.legend(handles=[ blue_line, red_line], loc = "lower right",prop={'size':12})
plt.ylabel('Observed Relative Frequency', fontsize = 15)
plt.xlabel('Forecast Probability',fontsize = 15)




################## Sample frequency bar graphs ################################

a = plt.axes([.145, .23, .1, .1], axisbg='white')
plt.bar(freq_pac[:,0],freq_pac[:,2],width = .06, color = 'green', edgecolor ='none', align='center')
plt.xlim([-0.01,1.01])
plt.ylim([10,6000])
plt.title('SREF ARW', y = 1.05, fontsize = 13)
plt.text(0.56, 3500, 'Pacific\nRanges', fontsize = 12)
plt.ylabel('# Forecasts', fontsize = 12)
plt.xlabel('Forecast Prob.',fontsize = 12)
plt.xticks(np.arange(0,1.0001,.2), fontsize = 11)
plt.yticks(np.arange(0,6001,1500), fontsize = 11)
#a.set_yscale('log')
plt.tick_params(axis='x',which='both', bottom='off')





a = plt.axes([.145, .06, .1, .1], axisbg='white')
plt.bar(freq_sref_bin_pac[:,0],freq_sref_bin_pac[:,2],width = .06, color = 'gold', edgecolor ='none', align='center')
plt.xlim([-0.01,1.01])
plt.ylim([10,6000])
plt.title('SREF NMMB', y = 1.05, fontsize = 13)
plt.text(0.56, 3500, 'Pacific\nRanges', fontsize = 12)
plt.ylabel('# Forecasts', fontsize = 12)
plt.xlabel('Forecast Prob.',fontsize = 12)
#a.set_yscale('log')
plt.xticks(np.arange(0,1.0001,.2), fontsize = 11)
plt.yticks(np.arange(0,6001,1500), fontsize = 11)
plt.tick_params(axis='x',which='both', bottom='off')



### Upper Decile plots
a = plt.axes([.81, .23, .1, .1], axisbg='white')
plt.bar(freq_int[:,0],freq_int[:,2],width = .06, color = 'green', edgecolor ='none',align='center')
plt.xlim([-0.01,1.01])
plt.ylim([10,6000])
plt.title('SREF ARW', y = 1.05, fontsize = 13)
plt.text(0.56, 3500, 'Interior\nRanges', fontsize = 12)
plt.ylabel('# Forecasts', fontsize = 12)
plt.xlabel('Forecast Prob.',fontsize = 12)
plt.xticks()
plt.xticks(np.arange(0,1.0001,.2), fontsize = 11)
plt.yticks(np.arange(0,6001,1500), fontsize = 11)
plt.tick_params(axis='x',which='both', bottom='off')



a = plt.axes([.81, .06, .1, .1], axisbg='white')
plt.bar(freq_sref_bin_int[:,0],freq_sref_bin_int[:,2],width = 0.06, color = 'gold', edgecolor ='none',align='center')
plt.xlim([-0.01,1.01])
plt.ylim([10,6000])
plt.title('SREF NMMB', y = 1.05, fontsize = 13)
plt.text(0.56, 3500, 'Interior\nRanges', fontsize = 12)
plt.ylabel('# Forecasts', fontsize = 12)
plt.xlabel('Forecast Prob.',fontsize = 12)
#a.set_yscale('log')
plt.xticks()
plt.xticks(np.arange(0,1.0001,.2), fontsize = 11)
plt.yticks(np.arange(0,6001,1500), fontsize = 11)
plt.tick_params(axis='x',which='both', bottom='off')



plt.savefig("../../../public_html/reliability_diagram_sref_arw_sref_nmb_interp_bin_percentile_85_pac_int_non_bias_corrected.pdf")                               
        
     
          














