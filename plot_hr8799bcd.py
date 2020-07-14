#HR8799b Photometry plots
import numpy as np
import os
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

inputdir = "/data/osiris_data/low_res/"
osiris_data_dir = "/data/osiris_data/"
out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
fontsize = 12
f1 = plt.figure(1,figsize=(18,12))
legend_list = []
 #b
plT,pllogg,plCtoO =1160.0, 3.3,  0.5724985412658228#1160.0, 4.266666666666667,  0.5724985412658228 ##
# plT,pllogg,plCtoO =1000.0, 3.0,  0.55 ##
# plT,pllogg,plCtoO =1000.0, 3.5,  0.65 ##
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
#  #c
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
# # plT,pllogg,plCtoO = 1000.0, 3.8, 0.5615070792405064
#  #d
# plT,pllogg,plCtoO = 1200.0, 3.0, 0.5450198862025316
# # plT,pllogg,plCtoO = 800.0, 3.8, 0.5615070792405064

R = 4000
IFSfilter = "Kbb"
tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
hdulist = pyfits.open(tmpfilename)
planet_model_grid =  hdulist[0].data
wvs_Kbb =  hdulist[1].data
Tlistunique =  hdulist[2].data
logglistunique =  hdulist[3].data
CtoOlistunique =  hdulist[4].data
hdulist.close()
myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)
model_Kbb = myinterpgrid([plT,pllogg,plCtoO])[0]

IFSfilter = "Hbb"
tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
hdulist = pyfits.open(tmpfilename)
planet_model_grid =  hdulist[0].data
wvs_Hbb =  hdulist[1].data
Tlistunique =  hdulist[2].data
logglistunique =  hdulist[3].data
CtoOlistunique =  hdulist[4].data
hdulist.close()
myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)
model_Hbb = myinterpgrid([plT,pllogg,plCtoO])[0]

where_notoverlap = np.where(wvs_Hbb<wvs_Kbb[0])
wvs = np.concatenate([wvs_Hbb[where_notoverlap],wvs_Kbb])
model = np.concatenate([model_Hbb[where_notoverlap],model_Kbb])


#import photometry first
t1 = np.genfromtxt(inputdir+'currie2011_hr8799b.txt')
C11fluxes = t1[:,0]
C11err = t1[:,1]
C11microns = t1[:,2]

t2 = np.genfromtxt(inputdir+'currie2014_hr8799b.txt')
C14fluxes = t2[:,0]
C14err = t2[:,1]
C14microns = t2[:,2]

# t3 = np.genfromtxt(inputdir+'galicher2011_hr8799d_phot.txt')
# G11fluxes = t3[:,0]
# G11err = t3[:,1]
# G11microns = t3[:,2]

#galicher 2011
G11fluxes = 1.31327E-16
G11err = 3.62869E-17
G11microns = 4.67
#1.31327E-16	3.62869E-17	4.67	M	galicher 2011

t4 = np.genfromtxt(inputdir+'zurlo2016_hr8799b.txt')
Z16fluxes = t4[:,0]
Z16err = t4[:,1]
Z16microns = t4[:,2]

t5 = np.genfromtxt(inputdir+'marois2008_hr8799b.txt')
M08fluxes = t5[:,0]
M08err = t5[:,1]
M08microns = t5[:,2]

# t6 = np.genfromtxt(inputdir+'skemer2012_hr8799d.txt')
# S12fluxes = t6[:,0]
# S12err = t6[:,1]
# S12microns = t6[:,2]

#Skemer 2012
S12fluxes = 1.0776E-15
S12err = 1.29026E-16
S12microns = 1.633
#1.0776E-15	1.29026E-16	1.633	H	skemer 2012

#t7 = np.genfromtxt(inputdir+'skemer2014_hr8799d.txt')
#S14fluxes = t7[:,0]
#S14err = t7[:,1]
#S14microns = t7[:,2]


#import spectra next

#Bowler:
t8 = np.genfromtxt(inputdir+'hr8799b_bowler.txt')
Bowfluxes = t8[:,1] *1.1e-15#need to find units
Bowerr = t8[:,2]*1.1e-15
Bowmicrons = t8[:,0]


#Hbb low res
t9 = np.genfromtxt(inputdir+'HR8799b_Hbb_lowres.txt')
Hbbfluxes = t9[:,1] #*(41.3/10)**2
Hbberr = t9[:,2] #*(41.3/10)**2
Hbbmicrons = t9[:,0]
Hbbfluxes = 10**Hbbfluxes
Hbberr = 10**Hbberr

#Kbb medres
t10 = np.genfromtxt(inputdir+'HR8799b_Kbb_medres.txt')
Kbbfluxes = t10[:,1] #*(41.3/10)**2
Kbberr = t10[:,2] #*(41.3/10)**2
Kbbmicrons = t10[:,0]
Kbbfluxes = 10**Kbbfluxes
Kbberr = 10**Kbberr

#P1640 spec, something funky with wavelength
t11 = np.genfromtxt(inputdir+'hr8799b_p1640_spec.txt')
p1640fluxes = t11[:,1] *1.1e-15 #*(41.3/10)**2
p1640err = t11[:,2] *1.1e-15 #*(41.3/10)**2
p1640microns = t11[:,0]/1000


#plt.errorbar(S14microns, S14fluxes, yerr = S14err, fmt='o', color = 'steelblue', capsize=5,
#    elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2014')
#plt.errorbar(Kbbmicrons, Kbbfluxes, yerr = Kbberr, color = 'k', capsize = 5,
#    elinewidth=1, markeredgewidth=1, label = 'Kbb med res')
# plt.plot(C11microns, C11fluxes,'o',label = 'Currie et al. 2011', color = 'm')
# plt.plot(C14microns, C14fluxes,'o',label = 'Currie et al. 2014', color = 'o')
# plt.plot(G11microns, G11fluxes,'o',label = 'Galicher et al. 2011', color = 'g')
# plt.plot(Z16microns, Z16fluxes,'o',label = 'Zurlo et al. 2016')

#set up plot
plt.subplot2grid((6,1),(0,0),rowspan=1)
plt.errorbar(Bowmicrons, Bowfluxes, yerr = Bowerr, color = '#00ffff', capsize = 5,
    elinewidth=1, markeredgewidth=1, label = 'Bowler et. al 2010')
plt.fill_between(Bowmicrons, Bowfluxes-Bowerr,Bowfluxes+Bowerr, color = '#00ffff', alpha=0.5)
plt.errorbar(p1640microns, p1640fluxes, yerr = p1640err, color = '#0033cc', capsize = 5,
    elinewidth=1, markeredgewidth=1, label = 'Oppenheimer et. al 2013')
plt.fill_between(p1640microns, p1640fluxes-p1640err,p1640fluxes+p1640err, color = '#0033cc', alpha=0.5)
plt.errorbar(Hbbmicrons, Hbbfluxes, yerr = Hbberr, color = '#00cc99', capsize=5,
    elinewidth=1, markeredgewidth=1,label = 'Barman et al. 2011')
plt.fill_between(Hbbmicrons, Hbbfluxes-Hbberr,Hbbfluxes+Hbberr, color = '#00cc99', alpha=0.5)

scaling_factor = np.nanmean(Kbbfluxes)/np.nanmean(model[np.where((Kbbmicrons[0]<wvs)*(wvs<Kbbmicrons[-1]))])
plt.plot(wvs,model*scaling_factor,linestyle="--",color="black", alpha=1,linewidth=0.5,label="Model")
plt.plot(Kbbmicrons, Kbbfluxes,label = 'Barman et al. 2015', color = '#0099cc', alpha=1,linewidth=0.5)

plt.ylim([0,2e-15])
plt.xlim([1.5,2.5])
lgd = plt.legend(loc="lower left",bbox_to_anchor=(1,0),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15])

plt.subplot2grid((6,1),(1,0),rowspan=1)
plt.plot(wvs,model*scaling_factor,linestyle="--",color="black", alpha=1,linewidth=0.5,label="Model")

# plt.errorbar(C11microns, C11fluxes, yerr = C11err, fmt='o', color = '#0099cc', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2011')
# plt.errorbar(C14microns, C14fluxes, yerr = C14err, fmt='v', color = '#0099cc', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2014')
# plt.errorbar(G11microns, G11fluxes, yerr = G11err, fmt='^', color = '#0099cc', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Galicher et al. 2011')
plt.errorbar(Z16microns, Z16fluxes, yerr = Z16err, fmt='x', color = '#0099cc', capsize=5,
    elinewidth=0, markeredgewidth=1, label = 'Zurlo et al. 2016')
plt.errorbar(M08microns, M08fluxes, yerr = M08err, fmt='s', color = '#0099cc', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Marois et al. 2008')
plt.errorbar(S12microns, S12fluxes, yerr = S12err, fmt='+', color = '#0099cc', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2012')

plt.ylim([0,2e-15])
plt.xlim([1.5,2.5])
lgd = plt.legend(loc="upper left",bbox_to_anchor=(1,1),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15])

#b
# plT,pllogg,plCtoO =1160.0, 4.266666666666667,  0.5724985412658228 ##
# plT,pllogg,plCtoO =1000.0, 3.0,  0.55 ##
# plT,pllogg,plCtoO =1000.0, 3.5,  0.65 ##
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
#  #c
plT,pllogg,plCtoO = 1200.0, 3.7, 0.5615070792405064#1200.0, 3.8, 0.5615070792405064
# # plT,pllogg,plCtoO = 1000.0, 3.8, 0.5615070792405064
#  #d
# plT,pllogg,plCtoO = 1200.0, 3.0, 0.5450198862025316
# # plT,pllogg,plCtoO = 800.0, 3.8, 0.5615070792405064

R = 4000
IFSfilter = "Kbb"
tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
hdulist = pyfits.open(tmpfilename)
planet_model_grid =  hdulist[0].data
wvs_Kbb =  hdulist[1].data
Tlistunique =  hdulist[2].data
logglistunique =  hdulist[3].data
CtoOlistunique =  hdulist[4].data
hdulist.close()
myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)
model_Kbb = myinterpgrid([plT,pllogg,plCtoO])[0]

IFSfilter = "Hbb"
tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
hdulist = pyfits.open(tmpfilename)
planet_model_grid =  hdulist[0].data
wvs_Hbb =  hdulist[1].data
Tlistunique =  hdulist[2].data
logglistunique =  hdulist[3].data
CtoOlistunique =  hdulist[4].data
hdulist.close()
myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)
model_Hbb = myinterpgrid([plT,pllogg,plCtoO])[0]

where_notoverlap = np.where(wvs_Hbb<wvs_Kbb[0])
wvs = np.concatenate([wvs_Hbb[where_notoverlap],wvs_Kbb])
model = np.concatenate([model_Hbb[where_notoverlap],model_Kbb])

#import photometry first
t1 = np.genfromtxt(inputdir+'currie2011_hr8799c.txt')
C11fluxes = t1[:,0]
C11err = t1[:,1]
C11microns = t1[:,2]

t2 = np.genfromtxt(inputdir+'currie2014_hr8799c.txt')
C14fluxes = t2[:,0]
C14err = t2[:,1]
C14microns = t2[:,2]

# t3 = np.genfromtxt(inputdir+'galicher2011_hr8799d_phot.txt')
# G11fluxes = t3[:,0]
# G11err = t3[:,1]
# G11microns = t3[:,2]

#galicher 2011
G11fluxes = 3.36E-16
G11err = 4.33E-17
G11microns = 4.67
#3.36E-16	4.33E-17	4.67	galicher 2011

t4 = np.genfromtxt(inputdir+'zurlo2016_phot_hr8799c.txt')
Z16fluxes = t4[:,0]
Z16err = t4[:,1]
Z16microns = t4[:,2]

t5 = np.genfromtxt(inputdir+'marois2008_hr8799c.txt')
M08fluxes = t5[:,0]
M08err = t5[:,1]
M08microns = t5[:,2]

# t6 = np.genfromtxt(inputdir+'skemer2012_hr8799d.txt')
# S12fluxes = t6[:,0]
# S12err = t6[:,1]
# S12microns = t6[:,2]

#Skemer 2012
S12fluxes = 2.47E-15
S12err = 3.18E-16
S12microns = 1.633
#2.47E-15	3.18E-16	1.633	skemer 2012

t7 = np.genfromtxt(inputdir+'skemer2014_hr8799c.txt')
S14fluxes = t7[:,0]
S14err = t7[:,1]
S14microns = t7[:,2]


#import spectra next
t8 = np.genfromtxt(inputdir+'greenbaum2018_hr8799c_H.txt')
G18Hfluxes = t8[:,1]
G18Herr = t8[:,2]
G18Hmicrons = t8[:,0]
t8 = np.genfromtxt(inputdir+'greenbaum2018_hr8799c_K.txt')
G18Kfluxes = t8[:,1]
G18Kerr = t8[:,2]
G18Kmicrons = t8[:,0]
t8 = np.genfromtxt(inputdir+'greenbaum2018_hr8799c_K2.txt')
G18K2fluxes = t8[:,1]
G18K2err = t8[:,2]
G18K2microns = t8[:,0]



#project 1640
t9 = np.genfromtxt(inputdir+'hr8799c_p1640_spec.txt')
Pfluxes = t9[:,1] *1.1e-15#(41.3/10)**2
Perr = t9[:,2] *1.1e-15#(41.3/10)**2
Pmicrons = t9[:,0]/1000

#Konopacky 2013
t10 = np.genfromtxt(inputdir+'hr8799c_kbb_medres_Konopacky2013.txt')
K13fluxes = t10[:,1]
K13err = t10[:,2]
K13microns = t10[:,0]
K13fluxes = 10**K13fluxes
K13err = 10**K13err

plt.subplot2grid((6,1),(2,0),rowspan=1)

# plt.errorbar(S14microns, S14fluxes, yerr = S14err, fmt='o', color = '#ff9900', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2014')
# plt.fill_between(S14microns, S14fluxes-S14err,S14fluxes+S14err, color = '#ff9900', alpha=0.5)
plt.errorbar(G18Hmicrons, G18Hfluxes, yerr = G18Herr, color = '#ff3300', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Greenbaum et al. 2018', alpha=0.5)
plt.fill_between(G18Hmicrons, G18Hfluxes-G18Herr,G18Hfluxes+G18Herr, color = '#ff3300', alpha=0.25)
plt.errorbar(G18Kmicrons, G18Kfluxes, yerr = G18Kerr, color = '#ff3300', capsize=5,
    elinewidth=1, markeredgewidth=1, alpha=0.5)
plt.fill_between(G18Kmicrons, G18Kfluxes-G18Kerr,G18Kfluxes+G18Kerr, color = '#ff3300', alpha=0.25)
plt.errorbar(G18K2microns, G18K2fluxes, yerr = G18K2err, color = '#ff3300', capsize=5,
    elinewidth=1, markeredgewidth=1, alpha=0.5)
plt.fill_between(G18K2microns, G18K2fluxes-G18K2err,G18K2fluxes+G18K2err, color = '#ff3300', alpha=0.25)
plt.errorbar(Pmicrons, Pfluxes, yerr = Perr, color = '#cccc00', capsize = 5,
    elinewidth=1, markeredgewidth=1, label = 'Oppenheimer et. al 2013', alpha=0.5)
plt.fill_between(Pmicrons, Pfluxes-Perr,Pfluxes+Perr, color = '#cccc00', alpha=0.25)

scaling_factor = np.nanmean(K13fluxes)/np.nanmean(model[np.where((K13microns[0]<wvs)*(wvs<K13microns[-1]))])
plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1,linewidth=0.5,label="Model")

plt.plot(K13microns, K13fluxes,label = 'Konopacky et al. 2013', color = '#ff9900', alpha=1,linewidth=0.5)

plt.ylim([0,5e-15])
plt.xlim([1.5,2.5])
lgd = plt.legend(loc="lower left",bbox_to_anchor=(1,0),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])

plt.subplot2grid((6,1),(3,0),rowspan=1)

# plt.errorbar(C11microns, C11fluxes, yerr = C11err, fmt='o', color = '#ff9900', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2011')
# plt.errorbar(C14microns, C14fluxes, yerr = C14err, fmt='v', color = '#ff9900', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2014')
# plt.errorbar(G11microns, G11fluxes, yerr = G11err, fmt='^', color = '#ff9900', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Galicher et al. 2011')
plt.errorbar(Z16microns, Z16fluxes, yerr = Z16err, fmt='x', color = '#ff9900', capsize=5,
    elinewidth=0, markeredgewidth=1, label = 'Zurlo et al. 2016')
plt.errorbar(M08microns, M08fluxes, yerr = M08err, fmt='s', color = '#ff9900', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Marois et al. 2008')
plt.errorbar(S12microns, S12fluxes, yerr = S12err, fmt='+', color = '#ff9900', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2012')

plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1,linewidth=0.5,label="Model")
plt.ylim([0,5e-15])
plt.xlim([1.5,2.5])
lgd = plt.legend(loc="upper left",bbox_to_anchor=(1,1),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])


plt.subplot2grid((6,1),(4,0),rowspan=1)


 #b
# plT,pllogg,plCtoO =1160.0, 4.266666666666667,  0.5724985412658228 ##
# plT,pllogg,plCtoO =1000.0, 3.0,  0.55 ##
# plT,pllogg,plCtoO =1000.0, 3.5,  0.65 ##
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
#  #c
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
# # plT,pllogg,plCtoO = 1000.0, 3.8, 0.5615070792405064
#  #d
plT,pllogg,plCtoO = 1200.0, 4.5, 0.5450198862025316#1200.0, 3.0, 0.5450198862025316
# # plT,pllogg,plCtoO = 800.0, 3.8, 0.5615070792405064

R = 4000
IFSfilter = "Kbb"
tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
hdulist = pyfits.open(tmpfilename)
planet_model_grid =  hdulist[0].data
wvs_Kbb =  hdulist[1].data
Tlistunique =  hdulist[2].data
logglistunique =  hdulist[3].data
CtoOlistunique =  hdulist[4].data
hdulist.close()
myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)
model_Kbb = myinterpgrid([plT,pllogg,plCtoO])[0]

IFSfilter = "Hbb"
tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
hdulist = pyfits.open(tmpfilename)
planet_model_grid =  hdulist[0].data
wvs_Hbb =  hdulist[1].data
Tlistunique =  hdulist[2].data
logglistunique =  hdulist[3].data
CtoOlistunique =  hdulist[4].data
hdulist.close()
myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)
model_Hbb = myinterpgrid([plT,pllogg,plCtoO])[0]

where_notoverlap = np.where(wvs_Hbb<wvs_Kbb[0])
wvs = np.concatenate([wvs_Hbb[where_notoverlap],wvs_Kbb])
model = np.concatenate([model_Hbb[where_notoverlap],model_Kbb])

#import photometry first
t1 = np.genfromtxt(inputdir+'currie2011_hr8799d.txt')
C11fluxes = t1[:,0]
C11err = t1[:,1]
C11microns = t1[:,2]

t2 = np.genfromtxt(inputdir+'currie2014_hr8799d.txt')
C14fluxes = t2[:,0]
C14err = t2[:,1]
C14microns = t2[:,2]

# t3 = np.genfromtxt(inputdir+'galicher2011_hr8799d_phot.txt')
# G11fluxes = t3[:,0]
# G11err = t3[:,1]
# G11microns = t3[:,2]

#galicher 2011
G11fluxes = 4.76818E-16
G11err = 1.53708E-16
G11microns = 4.67


t4 = np.genfromtxt(inputdir+'zurlo2016_hr8799d_phot.txt')
Z16fluxes = t4[:,0]
Z16err = t4[:,1]
Z16microns = t4[:,2]

t5 = np.genfromtxt(inputdir+'marois2008_phot_hr8799d.txt')
M08fluxes = t5[:,0]
M08err = t5[:,1]
M08microns = t5[:,2]

# t6 = np.genfromtxt(inputdir+'skemer2012_hr8799d.txt')
# S12fluxes = t6[:,0]
# S12err = t6[:,1]
# S12microns = t6[:,2]

#Skemer 2012
S12fluxes = 2.35753E-15
S12err = 4.34274E-16
S12microns = 1.633


t7 = np.genfromtxt(inputdir+'skemer2014_hr8799d.txt')
S14fluxes = t7[:,0]
S14err = t7[:,1]
S14microns = t7[:,2]


#import spectra next
t8 = np.genfromtxt(inputdir+'greenbaum2018_hr8799d_H.txt')
G18Hfluxes = t8[:,1]
G18Herr = t8[:,2]
G18Hmicrons = t8[:,0]
t8 = np.genfromtxt(inputdir+'greenbaum2018_hr8799d_K.txt')
G18Kfluxes = t8[:,1]
G18Kerr = t8[:,2]
G18Kmicrons = t8[:,0]
t8 = np.genfromtxt(inputdir+'greenbaum2018_hr8799d_K2.txt')
G18K2fluxes = t8[:,1]
G18K2err = t8[:,2]
G18K2microns = t8[:,0]


t9 = np.genfromtxt(inputdir+'zurlo2016_hr8799d.txt')
Z16Sfluxes = t9[:,1]*(41.3/10)**2
Z16Serr = t9[:,2]*(41.3/10)**2
Z16Smicrons = t9[:,0]


# plt.errorbar(S14microns, S14fluxes, yerr = S14err, fmt='o', color = '#6600ff', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2014')
# plt.fill_between(S14microns, S14fluxes-S14err,S14fluxes+S14err, color = '#6600ff', alpha=0.5)
plt.errorbar(G18Hmicrons, G18Hfluxes, yerr = G18Herr, color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Greenbaum et al. 2018')
plt.fill_between(G18Hmicrons, G18Hfluxes-G18Herr,G18Hfluxes+G18Herr, color = '#6600ff', alpha=0.5)
plt.errorbar(G18Kmicrons, G18Kfluxes, yerr = G18Kerr, color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1)
plt.fill_between(G18Kmicrons, G18Kfluxes-G18Kerr,G18Kfluxes+G18Kerr, color = '#6600ff', alpha=0.5)
plt.errorbar(G18K2microns, G18K2fluxes, yerr = G18K2err, color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1)
plt.fill_between(G18K2microns, G18K2fluxes-G18K2err,G18K2fluxes+G18K2err, color = '#6600ff', alpha=0.5)

scaling_factor = np.nanmean(G18K2fluxes)/np.nanmean(model[np.where((G18K2microns[0]<wvs)*(wvs<G18K2microns[-1]))])
plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1,linewidth=0.5,label="Model")

plt.ylim([0,5e-15])
plt.xlim([1.5,2.5])
lgd = plt.legend(loc="lower left",bbox_to_anchor=(1,0),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])

plt.subplot2grid((6,1),(5,0),rowspan=1)

plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1,linewidth=0.5,label="Model")

# plt.errorbar(C11microns, C11fluxes, yerr = C11err, fmt='o', color = '#6600ff', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2011')
# plt.errorbar(C14microns, C14fluxes, yerr = C14err, fmt='v', color = '#6600ff', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2014')
# plt.errorbar(G11microns, G11fluxes, yerr = G11err, fmt='^', color = '#6600ff', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Galicher et al. 2011')
plt.errorbar(Z16microns, Z16fluxes, yerr = Z16err, fmt='x', color = '#6600ff', capsize=5,
    elinewidth=0, markeredgewidth=1, label = 'Zurlo et al. 2016')
plt.errorbar(M08microns, M08fluxes, yerr = M08err, fmt='s', color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Marois et al. 2008')
plt.errorbar(S12microns, S12fluxes, yerr = S12err, fmt='+', color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2012')

plt.ylim([0,5e-15])
plt.xlim([1.5,2.5])
lgd = plt.legend(loc="upper left",bbox_to_anchor=(1,1),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])


plt.ylabel("Flux (W/$m^2$/$\mu$m)", fontsize=fontsize)
plt.xlabel("$\lambda$ ($\mu$m)", fontsize=fontsize)

f1.subplots_adjust(wspace=0,hspace=0)#
# print("Saving "+os.path.join(out_pngs,"HR8799bcd_lowresspec.png"))
# plt.savefig(os.path.join(out_pngs,"HR8799bcd_lowresspec.png"),bbox_inches='tight',bbox_extra_artists=legend_list)
# plt.savefig(os.path.join(out_pngs,"HR8799bcd_lowresspec.pdf"),bbox_inches='tight',bbox_extra_artists=legend_list)

plt.show()

