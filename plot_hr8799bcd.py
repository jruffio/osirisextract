#HR8799b Photometry plots
import numpy as np
import os
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from glob import glob

from scipy.interpolate import interp1d
import multiprocessing as mp
from reduce_HPFonly_diagcov_resmodel_v2 import convolve_spectrum

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

def get_vega_fluxes(filters, microns,vega_wvs,vega_spec):
    # vega_dwvs = vega_wvs[1::]-vega_wvs[0:np.size(vega_wvs)-1]
    # vega_dwvs = np.insert(vega_dwvs,0,vega_dwvs[0])
    flux_vega_list = []
    for photfilter,wv in zip(filters,microns):
        filter_arr = np.loadtxt(photfilter)
        wvs = filter_arr[:,0]
        trans = filter_arr[:,1]

        trans_f = interp1d(wvs,trans,bounds_error=False,fill_value=0)

        flux_vega_list.append(np.trapz(vega_spec*trans_f(vega_wvs),x=vega_wvs)/np.trapz(trans_f(vega_wvs),x=vega_wvs))

    return np.array(flux_vega_list)/0.1 # convert erg/s/cm^2/A to W/m2/mum


# /home/anaconda3/bin/python3 /home/sda/jruffio/pyOSIRIS/osirisextract/plot_all_post_CtoO_or_clouds.py
# b Photometry 1200.0 4.4 0.6604302374683544 37.25445673158307 1.2537683644661261e-15
# b Low resolution spectra 940.0 3.2666666666666666 0.5670028102531646 1.2304612717649738e-15
# b Forward model OSIRIS 1180.0 3.1666666666666665 0.577994272278481
# b Forward model OSIRIS (best 10 exposures) 1160.0 3.2 0.5889857343037974
# c Photometry 1200.0 4.366666666666667 0.4900625760759494 36.195809419872184 3.324116469339739e-15
# c Low resolution spectra 1200.0 4.5 0.5615070792405064 2.8852987399530044e-15
# c Forward model OSIRIS 1200.0 3.6666666666666665 0.5615070792405064
# c Forward model OSIRIS (best 10 exposures) 1180.0 3.4333333333333336 0.5670028102531646
# d Photometry 1140.0 4.5 0.45708819 36.304700320254064 3.0069061400378204e-15
# d Low resolution spectra 1020.0 4.5 0.6879088925316456 2.8889774797549904e-15
# d Forward model OSIRIS 1200.0 4.433333333333334 0.5230369621518988
# d Forward model OSIRIS (best 10 exposures) 1200.0 4.1 0.5944814653164557
# Saving /home/sda/jruffio/pyOSIRIS/figures/HR8799bcd_hr8799b_modelgrid_all_post.png



# myxlim = [1,5]
myxlim = [1.52,2.4]
numthreads=30
specpool = mp.Pool(processes=numthreads)
osiris_data_dir = "/data/osiris_data/"
out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
fontsize = 12
f1 = plt.figure(1,figsize=(18,12))
# f1 = plt.figure(1,figsize=(9,6))
legend_list = []
 #b
model_label_list = []
model_label_list.append("Model - Photometry")
model_label_list.append("model - Low resolution spectra")
model_label_list.append("model - FM OSIRIS")
# model_label_list.append("model - FM OSIRIS (10 best)")
model_paras_list = []
model_paras_list.append([1200.0, 4.4, 0.6604302374683544,1.2537683644661261e-15])
model_paras_list.append([940.0, 3.3, 0.5670028102531646,1.2304612717649738e-15])
model_paras_list.append([1180.0, 3.10, 0.577994272278481,1.6424043715285344e-15])
# model_paras_list.append([1160.0, 3.2, 0.5889857343037974,5.902517102241156e-16])

R = 4000
IFSfilter = "Kbb"
tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
hdulist = pyfits.open(tmpfilename)
planet_model_grid =  hdulist[0].data
wvs_Kbb =  hdulist[1].data
# print(np.median(wvs_Kbb[1::]-wvs_Kbb[0:np.size(wvs_Kbb)-1]))
# print(np.median(wvs_Kbb))
Tlistunique =  hdulist[2].data
logglistunique =  hdulist[3].data
CtoOlistunique =  hdulist[4].data
hdulist.close()
myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)
model_Kbb_list = [plfl*myinterpgrid([plT,pllogg,plCtoO])[0] for plT,pllogg,plCtoO,plfl in model_paras_list]

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
model_Hbblist = [plfl*myinterpgrid([plT,pllogg,plCtoO])[0] for plT,pllogg,plCtoO,plfl in model_paras_list]



vega_filename =  "/data/osiris_data/low_res/alpha_lyr_stis_010.fits"
hdulist = pyfits.open(vega_filename)
from astropy.table import Table
vega_table = Table(pyfits.getdata(vega_filename,1))
vega_wvs =  np.array(vega_table["WAVELENGTH"])  # angstroms -> mum
vega_spec = np.array(vega_table["FLUX"])  # erg s-1 cm-2 A-1
# vega_f = interp1d(vega_wvs,vega_spec)


osiris_data_dir = "/data/osiris_data/"
inputdir = "/data/osiris_data/low_res/HR_8799_b/"
import csv
planet = "b"
with open("/data/osiris_data/low_res/"+'HR_8799_fluxes.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    list_table = list(csv_reader)
    colnames = list_table[0]
    N_col = len(colnames)
    list_data = np.array(list_table[1::])
    absmags = np.array(list_data[:,colnames.index(planet)],dtype=np.float)
    list_data = list_data[np.where(np.isfinite(absmags))[0],:]
    absmags = np.array(list_data[:,colnames.index(planet)],dtype=np.float)
    filters = list_data[:,colnames.index("filter")]
    filters = [os.path.join(osiris_data_dir,"filters",filter) for filter in filters]
    refs = list_data[:,colnames.index("ref")]
    absmags_err = np.array(list_data[:,colnames.index(planet+" unc")],dtype=np.float)
    absmags_wvs = np.array(list_data[:,colnames.index("wv")],dtype=np.float)
    xerrs = get_bands_xerr(filters, absmags_wvs)
    vega_fluxes = get_vega_fluxes(filters, absmags_wvs,vega_wvs,vega_spec)

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
t11 = np.genfromtxt(inputdir+'hr8799b_p1640_spec_R=35.txt')
p1640fluxes = t11[:,1] *1.1e-15 #*(41.3/10)**2
p1640err = t11[:,2] *1.1e-15 #*(41.3/10)**2
p1640microns = t11[:,0]/1000



where_notoverlap = np.where(wvs_Hbb<wvs_Kbb[0])
wvs = np.concatenate([wvs_Hbb[where_notoverlap],wvs_Kbb])
model_list = []
for model_Kbb,model_Hbb,model_label in zip(model_Kbb_list,model_Hbblist,model_label_list):
    tmpspec = np.concatenate([model_Hbb[where_notoverlap],model_Kbb])
    if "Photometry" in model_label or "Low resolution spectra" in model_label:
        tmpspec = convolve_spectrum(wvs,tmpspec,100,mypool=specpool)
        model_list.append(tmpspec)
    elif "FM" in model_label:
        scaling_factor = np.nanmean(Kbbfluxes)/np.nanmean(tmpspec[np.where((Kbbmicrons[0]<wvs)*(wvs<Kbbmicrons[-1]))])
        model_list.append(scaling_factor*tmpspec)


#set up plot
plt.subplot2grid((6,1),(0,0),rowspan=1)
# scaling_factor = np.nanmean(Kbbfluxes)/np.nanmean(model[np.where((Kbbmicrons[0]<wvs)*(wvs<Kbbmicrons[-1]))])
# plt.plot(wvs,model*scaling_factor,linestyle="--",color="black", alpha=1,linewidth=0.5,label="Model")
for model,model_label,color_mod,ls,lw in zip(model_list,model_label_list,["grey","black","grey"],["-","--",":"],[1,2,0.5]):
    plt.plot(wvs,model,linestyle=ls,color=color_mod, alpha=1,linewidth=lw,label=model_label)

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


plt.gca().text(myxlim[0],0.98*2e-15,"HR 8799 b - Low resolution spectra",ha="left",va="top",rotation=0,size=fontsize,color='#0099cc',alpha=1)
plt.ylim([0,2e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="lower left",bbox_to_anchor=(1,0),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15])

plt.subplot2grid((6,1),(1,0),rowspan=1)
for model,model_label,color_mod,ls,lw in zip(model_list,model_label_list,["black","grey","grey"],["-","--",":"],[2,1,0.5]):
    plt.plot(wvs,model,linestyle=ls,color=color_mod, alpha=1,linewidth=lw,label=model_label)

# label_list  = ["Marois 2008","Currie 2011","Currie 2014","Galicher 2014","Skemer 2012","Zurlo 2016"]
label_list  = ["Marois 2008","Currie 2014","Skemer 2012","Zurlo 2016"]
fmt_list = ['s','o','v','^','+','o','x']
for label,fmt in zip(label_list,fmt_list):
    where_refs = np.where(label==refs)
    fluxes = vega_fluxes[where_refs]*10**(absmags[where_refs]/-2.5)
    yerr = [fluxes - vega_fluxes[where_refs]*10**((absmags[where_refs]+absmags_err[where_refs])/-2.5),
            vega_fluxes[where_refs]*10**((absmags[where_refs]-absmags_err[where_refs])/-2.5)-fluxes]
    plt.errorbar(absmags_wvs[where_refs], fluxes,yerr = yerr, xerr=None, fmt=fmt, color = '#0099cc', capsize=5,
        elinewidth=2, markeredgewidth=1, label = label)
    plt.errorbar(absmags_wvs[where_refs], fluxes,yerr = None, xerr=xerrs[:,where_refs[0]], fmt=fmt, color = '#0099cc', capsize=0,
        elinewidth=1, markeredgewidth=1)

plt.gca().text(myxlim[0],0.98*2e-15,"HR 8799 b - Photometry",ha="left",va="top",rotation=0,size=fontsize,color='#0099cc',alpha=1)
plt.ylim([0,2e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="upper left",bbox_to_anchor=(1,1),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15])

# plt.show()

#c
model_label_list = []
model_label_list.append("Model - Photometry")
model_label_list.append("model - Low resolution spectra")
model_label_list.append("model - FM OSIRIS")
# model_label_list.append("model - FM OSIRIS (10 best)")
model_paras_list = []
model_paras_list.append([1200.0, 4.366666666666667, 0.4900625760759494, 3.324116469339739e-15])
model_paras_list.append([1200.0, 4.5, 0.5615070792405064, 2.8852987399530044e-15])
model_paras_list.append([1200.0, 3.63, 0.562, 3.847076882400877e-15])
# model_paras_list.append([1180.0, 3.4333333333333336, 0.5670028102531646, 2.8852987399530044e-15])

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
model_Kbb_list = [plfl*myinterpgrid([plT,pllogg,plCtoO])[0] for plT,pllogg,plCtoO,plfl in model_paras_list]

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
model_Hbblist = [plfl*myinterpgrid([plT,pllogg,plCtoO])[0] for plT,pllogg,plCtoO,plfl in model_paras_list]



vega_filename =  "/data/osiris_data/low_res/alpha_lyr_stis_010.fits"
hdulist = pyfits.open(vega_filename)
from astropy.table import Table
vega_table = Table(pyfits.getdata(vega_filename,1))
vega_wvs =  np.array(vega_table["WAVELENGTH"])  # angstroms -> mum
vega_spec = np.array(vega_table["FLUX"])  # erg s-1 cm-2 A-1
# vega_f = interp1d(vega_wvs,vega_spec)


osiris_data_dir = "/data/osiris_data/"
inputdir = "/data/osiris_data/low_res/HR_8799_c/"
import csv
planet = "c"
with open("/data/osiris_data/low_res/"+'HR_8799_fluxes.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    list_table = list(csv_reader)
    colnames = list_table[0]
    N_col = len(colnames)
    list_data = np.array(list_table[1::])
    absmags = np.array(list_data[:,colnames.index(planet)],dtype=np.float)
    list_data = list_data[np.where(np.isfinite(absmags))[0],:]
    absmags = np.array(list_data[:,colnames.index(planet)],dtype=np.float)
    filters = list_data[:,colnames.index("filter")]
    filters = [os.path.join(osiris_data_dir,"filters",filter) for filter in filters]
    refs = list_data[:,colnames.index("ref")]
    absmags_err = np.array(list_data[:,colnames.index(planet+" unc")],dtype=np.float)
    absmags_wvs = np.array(list_data[:,colnames.index("wv")],dtype=np.float)
    xerrs = get_bands_xerr(filters, absmags_wvs)
    vega_fluxes = get_vega_fluxes(filters, absmags_wvs,vega_wvs,vega_spec)


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


where_notoverlap = np.where(wvs_Hbb<wvs_Kbb[0])
wvs = np.concatenate([wvs_Hbb[where_notoverlap],wvs_Kbb])
model_list = []
for model_Kbb,model_Hbb,model_label in zip(model_Kbb_list,model_Hbblist,model_label_list):
    tmpspec = np.concatenate([model_Hbb[where_notoverlap],model_Kbb])
    if "Photometry" in model_label or "Low resolution spectra" in model_label:
        tmpspec = convolve_spectrum(wvs,tmpspec,100,mypool=specpool)
        model_list.append(tmpspec)
    elif "FM" in model_label:
        scaling_factor = np.nanmean(K13fluxes)/np.nanmean(tmpspec[np.where((K13microns[0]<wvs)*(wvs<K13microns[-1]))])
        model_list.append(scaling_factor*tmpspec)

plt.subplot2grid((6,1),(2,0),rowspan=1)
for model,model_label,color_mod,ls,lw in zip(model_list,model_label_list,["grey","black","grey"],["-","--",":"],[1,2,0.5]):
    plt.plot(wvs,model,linestyle=ls,color=color_mod, alpha=1,linewidth=lw,label=model_label)


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
plt.plot(K13microns*(1-75/c_kms), K13fluxes,label = 'Konopacky et al. 2013', color = '#ff3300', alpha=1,linewidth=0.5)

plt.gca().text(myxlim[0],0.98*5e-15,"HR 8799 c - Low resolution spectra",ha="left",va="top",rotation=0,size=fontsize,color='#ff9900',alpha=1)
plt.ylim([0,5e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="lower left",bbox_to_anchor=(1,0),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])

plt.subplot2grid((6,1),(3,0),rowspan=1)
for model,model_label,color_mod,ls,lw in zip(model_list,model_label_list,["black","grey","grey"],["-","--",":"],[2,1,0.5]):
    plt.plot(wvs,model,linestyle=ls,color=color_mod, alpha=1,linewidth=lw,label=model_label)

# label_list  = ["Marois 2008","Currie 2011","Currie 2014","Galicher 2014","Skemer 2012","Skemer 2014","Zurlo 2016"]
label_list  = ["Marois 2008","Currie 2014","Skemer 2012","Zurlo 2016"]
fmt_list = ['s','o','v','^','+','o','x']
for label,fmt in zip(label_list,fmt_list):
    where_refs = np.where(label==refs)
    fluxes = vega_fluxes[where_refs]*10**(absmags[where_refs]/-2.5)
    yerr = [fluxes - vega_fluxes[where_refs]*10**((absmags[where_refs]+absmags_err[where_refs])/-2.5),
            vega_fluxes[where_refs]*10**((absmags[where_refs]-absmags_err[where_refs])/-2.5)-fluxes]
    plt.errorbar(absmags_wvs[where_refs], fluxes,yerr = yerr, xerr=None, fmt=fmt, color = '#ff3300', capsize=5,
        elinewidth=2, markeredgewidth=1, label = label)
    plt.errorbar(absmags_wvs[where_refs], fluxes,yerr = None, xerr=xerrs[:,where_refs[0]], fmt=fmt, color = '#ff3300', capsize=0,
        elinewidth=1, markeredgewidth=1)

plt.gca().text(myxlim[0],0.98*5e-15,"HR 8799 c - Photometry",ha="left",va="top",rotation=0,size=fontsize,color='#ff9900',alpha=1)
plt.ylim([0,5e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="upper left",bbox_to_anchor=(1,1),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])
# plt.show()



 #d
model_label_list = []
model_label_list.append("Model - Photometry")
model_label_list.append("model - Low resolution spectra")
model_label_list.append("model - FM OSIRIS")
# model_label_list.append("model - FM OSIRIS (10 best)")
model_paras_list = []
model_paras_list.append([1140.0, 4.5, 0.45708819, 3.0069061400378204e-15])
model_paras_list.append([1020.0, 4.5, 0.6879088925316456, 2.8889774797549904e-15])
model_paras_list.append([1200.0, 3.7, 0.551, 4.147534522659406e-15])
# model_paras_list.append([1200.0, 4.1, 0.5944814653164557, 2.8889774797549904e-15])

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
model_Kbb_list = [plfl*myinterpgrid([plT,pllogg,plCtoO])[0] for plT,pllogg,plCtoO,plfl in model_paras_list]

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
model_Hbblist = [plfl*myinterpgrid([plT,pllogg,plCtoO])[0] for plT,pllogg,plCtoO,plfl in model_paras_list]



vega_filename =  "/data/osiris_data/low_res/alpha_lyr_stis_010.fits"
hdulist = pyfits.open(vega_filename)
from astropy.table import Table
vega_table = Table(pyfits.getdata(vega_filename,1))
vega_wvs =  np.array(vega_table["WAVELENGTH"])  # angstroms -> mum
vega_spec = np.array(vega_table["FLUX"])  # erg s-1 cm-2 A-1
# vega_f = interp1d(vega_wvs,vega_spec)


osiris_data_dir = "/data/osiris_data/"
inputdir = "/data/osiris_data/low_res/HR_8799_d/"
import csv
planet = "d"
with open("/data/osiris_data/low_res/"+'HR_8799_fluxes.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    list_table = list(csv_reader)
    colnames = list_table[0]
    N_col = len(colnames)
    list_data = np.array(list_table[1::])
    absmags = np.array(list_data[:,colnames.index(planet)],dtype=np.float)
    list_data = list_data[np.where(np.isfinite(absmags))[0],:]
    absmags = np.array(list_data[:,colnames.index(planet)],dtype=np.float)
    filters = list_data[:,colnames.index("filter")]
    filters = [os.path.join(osiris_data_dir,"filters",filter) for filter in filters]
    refs = list_data[:,colnames.index("ref")]
    absmags_err = np.array(list_data[:,colnames.index(planet+" unc")],dtype=np.float)
    absmags_wvs = np.array(list_data[:,colnames.index("wv")],dtype=np.float)
    xerrs = get_bands_xerr(filters, absmags_wvs)
    vega_fluxes = get_vega_fluxes(filters, absmags_wvs,vega_wvs,vega_spec)



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

where_notoverlap = np.where(wvs_Hbb<wvs_Kbb[0])
wvs = np.concatenate([wvs_Hbb[where_notoverlap],wvs_Kbb])
model_list = []
for model_Kbb,model_Hbb,model_label in zip(model_Kbb_list,model_Hbblist,model_label_list):
    tmpspec = np.concatenate([model_Hbb[where_notoverlap],model_Kbb])
    if "Photometry" in model_label or "Low resolution spectra" in model_label:
        tmpspec = convolve_spectrum(wvs,tmpspec,100,mypool=specpool)
        model_list.append(tmpspec)
    elif "FM" in model_label:
        scaling_factor = np.nanmean(G18K2fluxes)/np.nanmean(tmpspec[np.where((G18K2microns[0]<wvs)*(wvs<G18K2microns[-1]))])
        model_list.append(scaling_factor*tmpspec)


plt.subplot2grid((6,1),(4,0),rowspan=1)
for model,model_label,color_mod,ls,lw in zip(model_list,model_label_list,["grey","black","grey"],["-","--",":"],[1,2,0.5]):
    plt.plot(wvs,model,linestyle=ls,color=color_mod, alpha=1,linewidth=lw,label=model_label)

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


plt.gca().text(myxlim[0],0.98*5e-15,"HR 8799 d - Low resolution spectra",ha="left",va="top",rotation=0,size=fontsize,color='#6600ff',alpha=1)
plt.ylim([0,5e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="lower left",bbox_to_anchor=(1,0),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.xticks([])
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])

plt.subplot2grid((6,1),(5,0),rowspan=1)
for model,model_label,color_mod,ls,lw in zip(model_list,model_label_list,["black","grey","grey"],["-","--",":"],[2,1,0.5]):
    plt.plot(wvs,model,linestyle=ls,color=color_mod, alpha=1,linewidth=lw,label=model_label)

# label_list  = ["Marois 2008","Currie 2011","Currie 2014","Galicher 2014","Skemer 2012","Skemer 2014","Zurlo 2016"]
label_list  = ["Marois 2008","Currie 2014","Skemer 2012","Zurlo 2016"]
fmt_list = ['s','o','v','^','+','o','x']
for label,fmt in zip(label_list,fmt_list):
    where_refs = np.where(label==refs)
    fluxes = vega_fluxes[where_refs]*10**(absmags[where_refs]/-2.5)
    yerr = [fluxes - vega_fluxes[where_refs]*10**((absmags[where_refs]+absmags_err[where_refs])/-2.5),
            vega_fluxes[where_refs]*10**((absmags[where_refs]-absmags_err[where_refs])/-2.5)-fluxes]
    plt.errorbar(absmags_wvs[where_refs], fluxes,yerr = yerr, xerr=None, fmt=fmt, color = '#6600ff', capsize=5,
        elinewidth=2, markeredgewidth=1, label = label)
    plt.errorbar(absmags_wvs[where_refs], fluxes,yerr = None, xerr=xerrs[:,where_refs[0]], fmt=fmt, color = '#6600ff', capsize=0,
        elinewidth=1, markeredgewidth=1)

plt.gca().text(myxlim[0],0.98*5e-15,"HR 8799 d - Photometry",ha="left",va="top",rotation=0,size=fontsize,color='#6600ff',alpha=1)
plt.ylim([0,5e-15])
plt.xlim(myxlim)
lgd = plt.legend(loc="upper left",bbox_to_anchor=(1,1),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
legend_list.append(lgd)
plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
plt.yticks([0e-15,1e-15,2e-15,3e-15,4e-15])


plt.ylabel("Flux density (W/$m^2$/$\mu$m)", fontsize=fontsize)
plt.xlabel("$\lambda$ ($\mu$m)", fontsize=fontsize)

f1.subplots_adjust(wspace=0,hspace=0)#
print("Saving "+os.path.join(out_pngs,"HR8799bcd_lowresspec.png"))
plt.savefig(os.path.join(out_pngs,"HR8799bcd_lowresspec.png"),bbox_inches='tight',bbox_extra_artists=legend_list)
plt.savefig(os.path.join(out_pngs,"HR8799bcd_lowresspec.pdf"),bbox_inches='tight',bbox_extra_artists=legend_list)

plt.show()

