#HR8799 Photometry plots
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

inputdir = "/data/osiris_data/low_res/HR_8799_d/"
osiris_data_dir = "/data/osiris_data/"
fontsize = 15
 #b
# plT,pllogg,plCtoO =1160.0, 4.266666666666667,  0.5724985412658228 ##
# plT,pllogg,plCtoO =1000.0, 3.0,  0.55 ##
# plT,pllogg,plCtoO =1000.0, 3.5,  0.65 ##
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
#  #c
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
# # plT,pllogg,plCtoO = 1000.0, 3.8, 0.5615070792405064
#  #d
plT,pllogg,plCtoO = 1200.0, 4.5, 0.5505156172151899
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


# plt.plot(C14microns, C14fluxes,'o',label = 'Currie et al. 2014', color = 'o')
# plt.plot(G11microns, G11fluxes,'o',label = 'Galicher et al. 2011', color = 'g')
# plt.plot(Z16microns, Z16fluxes,'o',label = 'Zurlo et al. 2016')

plt.figure(1,figsize=(18,4))
plt.errorbar(M08microns, M08fluxes, yerr = M08err, xerr=M08xerr, fmt='s', color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Marois et al. 2008')
plt.errorbar(C11microns, C11fluxes, yerr = C11err, xerr=C11xerr, fmt='o', color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2011')
plt.errorbar(C14microns, C14fluxes, yerr = C14err, xerr=C14xerr,  fmt='v', color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2014')
plt.errorbar(G11microns, G11fluxes, yerr = G11err, xerr=G11xerr, fmt='^', color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Galicher et al. 2011')
plt.errorbar(S12microns, S12fluxes, yerr = S12err, xerr=S12xerr, fmt='+', color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2012')
plt.errorbar(S14microns, S14fluxes, yerr = S14err, xerr=S14xerr, fmt='o', color = '#6600ff', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2014')
plt.errorbar(Z16microns, Z16fluxes, yerr = Z16err, xerr=Z16xerr, fmt='x', color = '#6600ff', capsize=5,
    elinewidth=0, markeredgewidth=1, label = 'Zurlo et al. 2016')
#set up plot



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

scaling_factor = np.nanmean(G18K2fluxes)/np.nanmean(model[np.where((G18K2microns[0]<wvs)*(wvs<G18K2microns[-1]))])
plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1,linewidth=0.5)

plt.ylabel("Flux (W/$m^2$/$\mu$m)", fontsize=fontsize)
plt.xlabel("$\lambda$ ($\mu$m)", fontsize=fontsize)
# plt.xlim([1.5,2.5])
lgd = plt.legend(loc="upper right",frameon=False,fontsize=fontsize*0.9,ncol=1)#,bbox_to_anchor=(1,1)
plt.tight_layout()
plt.show()

