#HR8799b Photometry plots
import numpy as np
import os
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from glob import glob

from scipy.interpolate import interp1d
def get_bands_xerr(filters, microns):
    xerr = []
    for photfilter,wv in zip(filters,microns):
        filter_arr = np.loadtxt(photfilter)
        wvs = filter_arr[:,0]/1e4
        trans = filter_arr[:,1]

        cutid =np.argmax(trans)

        wvs_firsthalf = interp1d(trans[0:cutid],wvs[0:cutid])
        wvs_secondhalf = interp1d(trans[cutid::],wvs[cutid::])

        xerr.append([wv-wvs_firsthalf(0.5),wvs_secondhalf(0.5)-wv])

    return np.array(xerr).T

osiris_data_dir = "/data/osiris_data/"
out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
fontsize = 12
f1 = plt.figure(1,figsize=(18,12))
legend_list = []
 #b
inputdir = "/data/osiris_data/low_res/HR_8799_b/"
plT,pllogg,plCtoO =1180.0, 3.1666666666666665, 0.577994272278481##
# plT,pllogg,plCtoO =1160.0, 4.266666666666667,  0.5724985412658228 ##
# plT,pllogg,plCtoO =1000.0, 3.0,  0.55 ##
# plT,pllogg,plCtoO =1000.0, 3.5,  0.65 ##
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
#  #c
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
# # plT,pllogg,plCtoO = 1000.0, 3.8, 0.5615070792405064
#  #d
# plT,pllogg,plCtoO = 1200.0, 3.0, 0.5450198862025316
# # plT,pllogg,plCtoO = 800.0, 3.8, 0.5615070792405064
myxlim = [1,5]

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
t1 = np.loadtxt(inputdir+'currie2011_hr8799b.txt',dtype=np.str)
t1 = t1[None,:]
C11fluxes = t1[:,0].astype(np.float)
C11err = t1[:,1].astype(np.float)
C11microns = t1[:,2].astype(np.float)
C11filters = t1[:,3]
C11xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in C11filters], C11microns)

t7 = np.loadtxt(inputdir+'currie2014_hr8799b.txt',dtype=np.str)
C14fluxes = t7[:,0].astype(np.float)
C14err = t7[:,1].astype(np.float)
C14microns = t7[:,2].astype(np.float)
C14filters = t7[:,3]
C14xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in C14filters], C14microns)


# t3 = np.genfromtxt(inputdir+'galicher2011_hr8799d_phot.txt')
# G11fluxes = t3[:,0]
# G11err = t3[:,1]
# G11microns = t3[:,2]

#galicher 2011
t1 = np.loadtxt(inputdir+'galicher2011_hr8799b.txt',dtype=np.str)
t1 = t1[None,:]
G11fluxes = t1[:,0].astype(np.float)
G11err = t1[:,1].astype(np.float)
G11microns = t1[:,2].astype(np.float)
G11filters = t1[:,3]
G11xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in G11filters], G11microns)

t4 = np.loadtxt(inputdir+'zurlo2016_phot_hr8799b.txt',dtype=np.str)
Z16fluxes = t4[:,0].astype(np.float)
Z16err = t4[:,1].astype(np.float)
Z16microns = t4[:,2].astype(np.float)
Z16filters = t4[:,3]
Z16xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in Z16filters], Z16microns)

t5 = np.loadtxt(inputdir+'marois2008_hr8799b.txt',dtype=np.str)
M08fluxes = t5[:,0].astype(np.float)
M08err = t5[:,1].astype(np.float)
M08microns = t5[:,2].astype(np.float)
M08filters = t5[:,3]
M08xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in M08filters], M08microns)


# t6 = np.genfromtxt(inputdir+'skemer2012_hr8799d.txt')
# S12fluxes = t6[:,0]
# S12err = t6[:,1]
# S12microns = t6[:,2]

#Skemer 2012
t1 = np.loadtxt(inputdir+'skemer2012_hr8799b.txt',dtype=np.str)
t1 = t1[None,:]
S12fluxes = t1[:,0].astype(np.float)
S12err = t1[:,1].astype(np.float)
S12microns = t1[:,2].astype(np.float)
S12filters = t1[:,3]
S12xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in S12filters], S12microns)



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
# t9 = np.genfromtxt(inputdir+'HR8799b_Hbb_lowres_R=60.txt')
# Hbbfluxes = t9[:,1] #*(41.3/10)**2
# Hbberr = t9[:,2] #*(41.3/10)**2
# Hbbmicrons = t9[:,0]
# Hbbfluxes = 10**Hbbfluxes
# Hbberr = 10**Hbberr
# np.savetxt(inputdir+'JB_HR8799b_Hbb_lowres_R=60.txt',np.concatenate([Hbbmicrons[:,None],Hbbfluxes[:,None],Hbberr[:,None]],axis=1),delimiter=" ")
t9 = np.genfromtxt(inputdir+'JB_HR8799b_Hbb_lowres_R=60.txt')
Hbbfluxes = t9[:,1] #*(41.3/10)**2
Hbberr = t9[:,2] #*(41.3/10)**2
Hbbmicrons = t9[:,0]

#Kbb medres
t10 = np.genfromtxt(inputdir+'HR8799b_Kbb_medres.txt')
Kbbfluxes_HR = t10[:,1] #*(41.3/10)**2
Kbberr_HR = t10[:,2] #*(41.3/10)**2
Kbbmicrons_HR = t10[:,0]
Kbbfluxes_HR = 10**Kbbfluxes_HR
Kbberr_HR = 10**Kbberr_HR
# if 0:
#     # plt.plot(Kbbmicrons,Kbbfluxes,color="blue")
#     # plt.plot(Kbbmicrons,Kbberr)
#     R = 200
#     midwv = np.median(Kbbmicrons)
#     dwvs = Kbbmicrons[1::]-Kbbmicrons[0:np.size(Kbbmicrons)-1]
#     dwvs = np.insert(dwvs,0,dwvs[0])
#     new_dwv = midwv/R
#     new_Kbbmicrons = np.arange(Kbbmicrons[0]+new_dwv,Kbbmicrons[-1]-new_dwv,2*new_dwv)
#     new_Kbberr = np.zeros(new_Kbbmicrons.shape) + np.nanmedian(Kbberr)
#     new_Kbbfluxes = np.zeros(new_Kbbmicrons.shape)
#     for k,spec_wv in enumerate(new_Kbbmicrons):
#         trans_f = lambda x: np.exp(-0.5*(x-spec_wv)**2/(spec_wv/R/2.634)**2)
#         new_Kbbfluxes[k] = np.sum(Kbbfluxes*trans_f(Kbbmicrons)*dwvs)/np.sum(trans_f(Kbbmicrons)*dwvs)
#     # print(new_Kbbfluxes)
#     # plt.plot(new_Kbbmicrons,new_Kbbfluxes,color="red")
#     # plt.show()
#     np.savetxt(inputdir+'JB_HR8799b_Kbb_lowres_R=200.txt',np.concatenate([new_Kbbmicrons[:,None],new_Kbbfluxes[:,None],new_Kbberr[:,None]],axis=1),delimiter=" ")
t10 = np.genfromtxt(inputdir+'JB_HR8799b_Kbb_lowres_R=200.txt')
Kbbfluxes = t10[:,1]
Kbberr = t10[:,2]
Kbbmicrons = t10[:,0]

#P1640 spec, something funky with wavelength
t11 = np.genfromtxt(inputdir+'hr8799b_p1640_spec.txt')
p1640fluxes = t11[:,1] *1.1e-15 #*(41.3/10)**2
p1640err = t11[:,2] *1.1e-15 #*(41.3/10)**2
p1640microns = t11[:,0]/1000



#set up plot
plt.subplot2grid((6,1),(0,0),rowspan=1)
scaling_factor = np.nanmean(Kbbfluxes)/np.nanmean(model[np.where((Kbbmicrons[0]<wvs)*(wvs<Kbbmicrons[-1]))])
plt.plot(wvs,model*scaling_factor,linestyle="--",color="black", alpha=1,linewidth=0.5,label="Model")

plt.errorbar(Bowmicrons, Bowfluxes, yerr = Bowerr, color = '#00ffff', capsize = 5,
    elinewidth=1, markeredgewidth=1, label = 'Bowler et. al 2010')
plt.fill_between(Bowmicrons, Bowfluxes-Bowerr,Bowfluxes+Bowerr, color = '#00ffff', alpha=0.5)
plt.errorbar(p1640microns, p1640fluxes, yerr = p1640err, color = '#0033cc', capsize = 5,
    elinewidth=1, markeredgewidth=1, label = 'Oppenheimer et. al 2013')
plt.fill_between(p1640microns, p1640fluxes-p1640err,p1640fluxes+p1640err, color = '#0033cc', alpha=0.5)

plt.errorbar(Hbbmicrons, Hbbfluxes, yerr = Hbberr, color = '#00cc99', capsize=5,
    elinewidth=1, markeredgewidth=1,label = 'Barman et al. 2011')
plt.fill_between(Hbbmicrons, Hbbfluxes-Hbberr,Hbbfluxes+Hbberr, color = '#00cc99', alpha=0.5)
plt.errorbar(Kbbmicrons, Kbbfluxes, yerr = Kbberr, color = '#0099cc', capsize=5,
    elinewidth=1, markeredgewidth=1,label = 'Barman et al. 2015')
plt.fill_between(Kbbmicrons, Kbbfluxes-Kbberr,Kbbfluxes+Kbberr, color = '#0099cc', alpha=0.5)
plt.plot(Kbbmicrons_HR, Kbbfluxes_HR, color = '#0099cc', alpha=1,linewidth=0.5)


plt.ylim([0,2e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="lower left",bbox_to_anchor=(1,0),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15])

plt.subplot2grid((6,1),(1,0),rowspan=1)
plt.plot(wvs,model*scaling_factor,linestyle="--",color="black", alpha=1,linewidth=0.5,label="Model")

plt.errorbar(M08microns, M08fluxes, yerr = M08err, xerr=M08xerr, fmt='s', color = '#0099cc', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Marois et al. 2008')
if 1:
    plt.errorbar(C11microns, C11fluxes, yerr = C11err, xerr=C11xerr, fmt='o', color = '#0099cc', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2011')
    plt.errorbar(C14microns, C14fluxes, yerr = C14err, xerr=C14xerr,  fmt='v', color = '#0099cc', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2014')
    plt.errorbar(G11microns, G11fluxes, yerr = G11err, xerr=G11xerr, fmt='^', color = '#0099cc', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Galicher et al. 2011')
plt.errorbar(S12microns, S12fluxes, yerr = S12err, xerr=S12xerr, fmt='+', color = '#0099cc', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2012')
plt.errorbar(Z16microns, Z16fluxes, yerr = Z16err, xerr=Z16xerr, fmt='x', color = '#0099cc', capsize=5,
    elinewidth=0, markeredgewidth=1, label = 'Zurlo et al. 2016')

plt.ylim([0,2e-15])
plt.xlim(myxlim)
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
inputdir = "/data/osiris_data/low_res/HR_8799_c/"
plT,pllogg,plCtoO = 1200.0, 3.6666666666666665 ,0.5670028102531646
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

t5 = np.loadtxt(inputdir+'marois2008_hr8799c.txt',dtype=np.str)
M08fluxes = t5[:,0].astype(np.float)
M08err = t5[:,1].astype(np.float)
M08microns = t5[:,2].astype(np.float)
M08filters = t5[:,3]
M08xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in M08filters], M08microns)

#import photometry first
t1 = np.loadtxt(inputdir+'currie2011_hr8799c.txt',dtype=np.str)
t1 = t1[None,:]
C11fluxes = t1[:,0].astype(np.float)
C11err = t1[:,1].astype(np.float)
C11microns = t1[:,2].astype(np.float)
C11filters = t1[:,3]
C11xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in C11filters], C11microns)

#galicher 2011
t1 = np.loadtxt(inputdir+'galicher2011_hr8799c.txt',dtype=np.str)
t1 = t1[None,:]
G11fluxes = t1[:,0].astype(np.float)
G11err = t1[:,1].astype(np.float)
G11microns = t1[:,2].astype(np.float)
G11filters = t1[:,3]
G11xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in G11filters], G11microns)

#Skemer 2012 # missing filter?
t1 = np.loadtxt(inputdir+'skemer2012_hr8799c.txt',dtype=np.str)
t1 = t1[None,:]
S12fluxes = t1[:,0].astype(np.float)
S12err = t1[:,1].astype(np.float)
S12microns = t1[:,2].astype(np.float)
S12filters = t1[:,3]
S12xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in S12filters], S12microns)

t7 = np.loadtxt(inputdir+'currie2014_hr8799c.txt',dtype=np.str)
C14fluxes = t7[:,0].astype(np.float)
C14err = t7[:,1].astype(np.float)
C14microns = t7[:,2].astype(np.float)
C14filters = t7[:,3]
C14xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in C14filters], C14microns)


t7 = np.loadtxt(inputdir+'skemer2014_hr8799c.txt',dtype=np.str)
S14fluxes = t7[:,0].astype(np.float)
S14err = t7[:,1].astype(np.float)
S14microns = t7[:,2].astype(np.float)
S14filters = t7[:,3]
S14xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in S14filters], S14microns)



t4 = np.loadtxt(inputdir+'zurlo2016_phot_hr8799c.txt',dtype=np.str)
Z16fluxes = t4[:,0].astype(np.float)
Z16err = t4[:,1].astype(np.float)
Z16microns = t4[:,2].astype(np.float)
Z16filters = t4[:,3]
Z16xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in Z16filters], Z16microns)


#import spectra next
t8 = np.genfromtxt(glob(inputdir+'greenbaum2018_hr8799c_H_R=*.txt')[0])
G18Hfluxes = t8[:,1]
G18Herr = t8[:,2]
G18Hmicrons = t8[:,0]
t8 = np.genfromtxt(glob(inputdir+'greenbaum2018_hr8799c_K_R=*.txt')[0])
G18Kfluxes = t8[:,1]
G18Kerr = t8[:,2]
G18Kmicrons = t8[:,0]
t8 = np.genfromtxt(glob(inputdir+'greenbaum2018_hr8799c_K2_R=*.txt')[0])
G18K2fluxes = t8[:,1]
G18K2err = t8[:,2]
G18K2microns = t8[:,0]

#project 1640
t9 = np.genfromtxt(glob(os.path.join(inputdir,'hr8799c_p1640_spec_R=*.txt'))[0])
Pfluxes = t9[:,1] *1.1e-15#(41.3/10)**2
Perr = t9[:,2] *1.1e-15#(41.3/10)**2
Pmicrons = t9[:,0]/1000

#Konopacky 2013
t10 = np.genfromtxt(os.path.join(inputdir,'hr8799c_kbb_medres_Konopacky2013.txt'))
K13fluxes = t10[:,1]
K13err = t10[:,2]
K13microns = t10[:,0]
K13fluxes = 10**K13fluxes
K13err = 10**K13err

plt.subplot2grid((6,1),(2,0),rowspan=1)

scaling_factor = np.nanmean(K13fluxes)/np.nanmean(model[np.where((K13microns[0]<wvs)*(wvs<K13microns[-1]))])
plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1,linewidth=0.5,label="Model")

plt.errorbar(G18Hmicrons, G18Hfluxes, yerr = G18Herr, color = '#ff9900', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Greenbaum et al. 2018')
plt.fill_between(G18Hmicrons, G18Hfluxes-G18Herr,G18Hfluxes+G18Herr, color = '#ff9900', alpha=0.5)
plt.errorbar(G18Kmicrons, G18Kfluxes, yerr = G18Kerr, color = '#ff9900', capsize=5,
    elinewidth=1, markeredgewidth=1)
plt.fill_between(G18Kmicrons, G18Kfluxes-G18Kerr,G18Kfluxes+G18Kerr, color = '#ff9900', alpha=0.5)
plt.errorbar(G18K2microns, G18K2fluxes, yerr = G18K2err, color = '#ff9900', capsize=5,
    elinewidth=1, markeredgewidth=1)
plt.fill_between(G18K2microns, G18K2fluxes-G18K2err,G18K2fluxes+G18K2err, color = '#ff9900', alpha=0.5)

plt.errorbar(Pmicrons, Pfluxes, yerr = Perr, color = '#cccc00', capsize = 5,
    elinewidth=1, markeredgewidth=1, label = 'Oppenheimer et. al 2013')
plt.fill_between(Pmicrons, Pfluxes-Perr,Pfluxes+Perr, color = '#cccc00', alpha=0.25)


c_kms = 299792.458
plt.plot(K13microns*(1-75/c_kms), K13fluxes,label = 'Konopacky et al. 2013', color = '#ff9900', alpha=1,linewidth=0.5)

plt.ylim([0,5e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="lower left",bbox_to_anchor=(1,0),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])

plt.subplot2grid((6,1),(3,0),rowspan=1)

plt.errorbar(M08microns, M08fluxes, yerr = M08err, xerr=M08xerr, fmt='s', color = '#ff3300', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Marois et al. 2008')
if 1:
    plt.errorbar(C11microns, C11fluxes, yerr = C11err, xerr=C11xerr, fmt='o', color = '#ff3300', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2011')
    plt.errorbar(C14microns, C14fluxes, yerr = C14err, xerr=C14xerr,  fmt='v', color = '#ff3300', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2014')
    plt.errorbar(G11microns, G11fluxes, yerr = G11err, xerr=G11xerr, fmt='^', color = '#ff3300', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Galicher et al. 2011')
    plt.errorbar(S14microns, S14fluxes, yerr = S14err, xerr=S14xerr, fmt='o', color = '#ff3300', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2014')
plt.errorbar(S12microns, S12fluxes, yerr = S12err, xerr=S12xerr, fmt='+', color = '#ff3300', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2012')
plt.errorbar(Z16microns, Z16fluxes, yerr = Z16err, xerr=Z16xerr, fmt='x', color = '#ff3300', capsize=5,
    elinewidth=0, markeredgewidth=1, label = 'Zurlo et al. 2016')

plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1,linewidth=0.5,label="Model")
plt.ylim([0,5e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="upper left",bbox_to_anchor=(1,1),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])




 #b
# plT,pllogg,plCtoO =1160.0, 4.266666666666667,  0.5724985412658228 ##
# plT,pllogg,plCtoO =1000.0, 3.0,  0.55 ##
# plT,pllogg,plCtoO =1000.0, 3.5,  0.65 ##
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
#  #c
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
# # plT,pllogg,plCtoO = 1000.0, 3.8, 0.5615070792405064
#  #d
inputdir = "/data/osiris_data/low_res/HR_8799_d/"
plT,pllogg,plCtoO = 1200.0, 4.5, 0.5505156172151899
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


t5 = np.loadtxt(inputdir+'marois2008_hr8799d.txt',dtype=np.str)
M08fluxes = t5[:,0].astype(np.float)
M08err = t5[:,1].astype(np.float)
M08microns = t5[:,2].astype(np.float)
M08filters = t5[:,3]
M08xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in M08filters], M08microns)

#import photometry first
t1 = np.loadtxt(inputdir+'currie2011_hr8799d.txt',dtype=np.str)
t1 = t1[None,:]
C11fluxes = t1[:,0].astype(np.float)
C11err = t1[:,1].astype(np.float)
C11microns = t1[:,2].astype(np.float)
C11filters = t1[:,3]
C11xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in C11filters], C11microns)

#galicher 2011
t1 = np.loadtxt(inputdir+'galicher2011_hr8799d.txt',dtype=np.str)
t1 = t1[None,:]
G11fluxes = t1[:,0].astype(np.float)
G11err = t1[:,1].astype(np.float)
G11microns = t1[:,2].astype(np.float)
G11filters = t1[:,3]
G11xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in G11filters], G11microns)

#Skemer 2012 # missing filter?
t1 = np.loadtxt(inputdir+'skemer2012_hr8799d.txt',dtype=np.str)
t1 = t1[None,:]
S12fluxes = t1[:,0].astype(np.float)
S12err = t1[:,1].astype(np.float)
S12microns = t1[:,2].astype(np.float)
S12filters = t1[:,3]
S12xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in S12filters], S12microns)

t7 = np.loadtxt(inputdir+'currie2014_hr8799d.txt',dtype=np.str)
C14fluxes = t7[:,0].astype(np.float)
C14err = t7[:,1].astype(np.float)
C14microns = t7[:,2].astype(np.float)
C14filters = t7[:,3]
C14xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in C14filters], C14microns)


t7 = np.loadtxt(inputdir+'skemer2014_hr8799d.txt',dtype=np.str)
S14fluxes = t7[:,0].astype(np.float)
S14err = t7[:,1].astype(np.float)
S14microns = t7[:,2].astype(np.float)
S14filters = t7[:,3]
S14xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in S14filters], S14microns)



t4 = np.loadtxt(inputdir+'zurlo2016_phot_hr8799d.txt',dtype=np.str)
Z16fluxes = t4[:,0].astype(np.float)
Z16err = t4[:,1].astype(np.float)
Z16microns = t4[:,2].astype(np.float)
Z16filters = t4[:,3]
Z16xerr = get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in Z16filters], Z16microns)


#import spectra next
t8 = np.genfromtxt(glob(inputdir+'greenbaum2018_hr8799d_H_R=*.txt')[0])
G18Hfluxes = t8[:,1]
G18Herr = t8[:,2]
G18Hmicrons = t8[:,0]
t8 = np.genfromtxt(glob(inputdir+'greenbaum2018_hr8799d_K_R=*.txt')[0])
G18Kfluxes = t8[:,1]
G18Kerr = t8[:,2]
G18Kmicrons = t8[:,0]
t8 = np.genfromtxt(glob(inputdir+'greenbaum2018_hr8799d_K2_R=*.txt')[0])
G18K2fluxes = t8[:,1]
G18K2err = t8[:,2]
G18K2microns = t8[:,0]


# t9 = np.genfromtxt(inputdir+'zurlo2016_hr8799d_R=30.txt')
# Z16Sfluxes = t9[:,1]*(41.3/10)**2
# Z16Serr = t9[:,2]*(41.3/10)**2
# Z16Smicrons = t9[:,0]
# np.savetxt(inputdir+'JB_zurlo2016_hr8799d_R=30.txt',np.concatenate([Z16Smicrons[:,None],Z16Sfluxes[:,None],Z16Serr[:,None]],axis=1),delimiter=" ")
t9 = np.genfromtxt(inputdir+'JB_zurlo2016_hr8799d_R=30.txt')
Z16Sfluxes = t9[:,1]
Z16Serr = t9[:,2]
Z16Smicrons = t9[:,0]

plt.subplot2grid((6,1),(4,0),rowspan=1)
scaling_factor = np.nanmean(G18K2fluxes)/np.nanmean(model[np.where((G18K2microns[0]<wvs)*(wvs<G18K2microns[-1]))])
plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1,linewidth=0.5,label="Model")

plt.errorbar(G18Hmicrons, G18Hfluxes, yerr = G18Herr, color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Greenbaum et al. 2018')
plt.fill_between(G18Hmicrons, G18Hfluxes-G18Herr,G18Hfluxes+G18Herr, color = '#6600ff', alpha=0.5)
plt.errorbar(G18Kmicrons, G18Kfluxes, yerr = G18Kerr, color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1)
plt.fill_between(G18Kmicrons, G18Kfluxes-G18Kerr,G18Kfluxes+G18Kerr, color = '#6600ff', alpha=0.5)
plt.errorbar(G18K2microns, G18K2fluxes, yerr = G18K2err, color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1)
plt.fill_between(G18K2microns, G18K2fluxes-G18K2err,G18K2fluxes+G18K2err, color = '#6600ff', alpha=0.5)

plt.errorbar(Z16Smicrons, Z16Sfluxes, yerr = Z16Serr, color = '#990066', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Zurlo et al. 2016')
plt.fill_between(Z16Smicrons, Z16Sfluxes-Z16Serr,Z16Sfluxes+Z16Serr, color = '#990066', alpha=0.5)


plt.ylim([0,5e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="lower left",bbox_to_anchor=(1,0),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])

plt.subplot2grid((6,1),(5,0),rowspan=1)

plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1,linewidth=0.5,label="Model")
plt.errorbar(M08microns, M08fluxes, yerr = M08err, xerr=M08xerr, fmt='s', color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Marois et al. 2008')
if 1:
    plt.errorbar(C11microns, C11fluxes, yerr = C11err, xerr=C11xerr, fmt='o', color = '#6600ff', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2011')
    plt.errorbar(C14microns, C14fluxes, yerr = C14err, xerr=C14xerr,  fmt='v', color = '#6600ff', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2014')
    plt.errorbar(G11microns, G11fluxes, yerr = G11err, xerr=G11xerr, fmt='^', color = '#6600ff', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Galicher et al. 2011')
    plt.errorbar(S14microns, S14fluxes, yerr = S14err, xerr=S14xerr, fmt='o', color = '#6600ff', capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2014')
plt.errorbar(S12microns, S12fluxes, yerr = S12err, xerr=S12xerr, fmt='+', color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2012')
plt.errorbar(Z16microns, Z16fluxes, yerr = Z16err, xerr=Z16xerr, fmt='x', color = '#6600ff', capsize=5,
    elinewidth=0, markeredgewidth=1, label = 'Zurlo et al. 2016')

plt.ylim([0,5e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="upper left",bbox_to_anchor=(1,1),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])


plt.ylabel("Flux (W/$m^2$/$\mu$m)", fontsize=fontsize)
plt.xlabel("$\lambda$ ($\mu$m)", fontsize=fontsize)

f1.subplots_adjust(wspace=0,hspace=0)#
print("Saving "+os.path.join(out_pngs,"HR8799bcd_lowresspec.png"))
plt.savefig(os.path.join(out_pngs,"HR8799bcd_lowresspec.png"),bbox_inches='tight',bbox_extra_artists=legend_list)
plt.savefig(os.path.join(out_pngs,"HR8799bcd_lowresspec.pdf"),bbox_inches='tight',bbox_extra_artists=legend_list)

plt.show()

