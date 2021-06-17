#HR8799b Photometry plots
import numpy as np
import os
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

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

inputdir = "/data/osiris_data/low_res/HR_8799_b/"
osiris_data_dir = "/data/osiris_data/"

 #b
plT,pllogg,plCtoO =1180.0, 3.1666666666666665, 0.577994272278481##
# plT,pllogg,plCtoO =1100.0, 4.266666666666667,  0.5724985412658228 ##
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
t1 = np.loadtxt(inputdir+'currie2011_hr8799b.txt',dtype=np.str)
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
# t10 = np.genfromtxt(inputdir+'HR8799b_Kbb_medres.txt')
# Kbbfluxes = t10[:,1] #*(41.3/10)**2
# Kbberr = t10[:,2] #*(41.3/10)**2
# Kbbmicrons = t10[:,0]
# Kbbfluxes = 10**Kbbfluxes
# Kbberr = 10**Kbberr
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


#plt.errorbar(S14microns, S14fluxes, yerr = S14err, fmt='o', color = 'steelblue', capsize=5,
#    elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2014')
#plt.errorbar(Kbbmicrons, Kbbfluxes, yerr = Kbberr, color = 'k', capsize = 5,
#    elinewidth=1, markeredgewidth=1, label = 'Kbb med res')
# plt.plot(C11microns, C11fluxes,'o',label = 'Currie et al. 2011', color = 'm')
# plt.plot(C14microns, C14fluxes,'o',label = 'Currie et al. 2014', color = 'o')
# plt.plot(G11microns, G11fluxes,'o',label = 'Galicher et al. 2011', color = 'g')
# plt.plot(Z16microns, Z16fluxes,'o',label = 'Zurlo et al. 2016')

#set up plot

#good?
plt.errorbar(M08microns, M08fluxes, yerr = M08err, xerr=M08xerr, fmt='s', color = '#0099cc', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Marois et al. 2008')
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

# missing something
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


# scaling_factor = np.nanmean(Kbbfluxes)/np.nanmean(model[np.where((Kbbmicrons[0]<wvs)*(wvs<Kbbmicrons[-1]))])
# plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1)


plt.ylabel("Flux (W/$m^2$/$\mu$m)", fontsize=15)
plt.xlabel("$\lambda$ ($\mu$m)", fontsize=15)
# plt.xlim([1.0,2.5])
plt.legend(frameon=False, loc=1, fontsize=13) #no frame, and upper right loc #
plt.show()
