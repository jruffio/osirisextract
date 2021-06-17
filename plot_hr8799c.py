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
    # print(vega_fluxes*10**(absmags/-2.5))
# print(absmags)
# print(xerrs)
# exit()
fontsize = 15
 #b
# plT,pllogg,plCtoO =1160.0, 4.266666666666667,  0.5724985412658228 ##
# plT,pllogg,plCtoO =1000.0, 3.0,  0.55 ##
# plT,pllogg,plCtoO =1000.0, 3.5,  0.65 ##
# plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
#  #c
plT,pllogg,plCtoO = 1200.0, 3.6666666666666665, 0.5615070792405064
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

# plt.plot(C14microns, C14fluxes,'o',label = 'Currie et al. 2014', color = 'o')
# plt.plot(G11microns, G11fluxes,'o',label = 'Galicher et al. 2011', color = 'g')
# plt.plot(Z16microns, Z16fluxes,'o',label = 'Zurlo et al. 2016')

plt.figure(1,figsize=(18,4))
label_list  = ["Marois 2008","Currie 2011","Currie 2014","Galicher 2014","Skemer 2012","Skemer 2014","Zurlo 2016"]
fmt_list = ['s','o','v','^','+','o','x']
for label,fmt in zip(label_list,fmt_list):
    print(label)
    where_refs = np.where(label==refs)
    print(where_refs)
    fluxes = vega_fluxes[where_refs]*10**(absmags[where_refs]/-2.5)
    yerr = [fluxes - vega_fluxes[where_refs]*10**((absmags[where_refs]+absmags_err[where_refs])/-2.5),
            vega_fluxes[where_refs]*10**((absmags[where_refs]-absmags_err[where_refs])/-2.5)-fluxes]
    print(yerr)
    plt.errorbar(absmags_wvs[where_refs], fluxes,yerr = yerr, xerr=xerrs[:,where_refs[0]], fmt=fmt, color = '#ff3300', capsize=5,
        elinewidth=1, markeredgewidth=1, label = label)
# plt.errorbar(M08microns, M08fluxes, yerr = M08err, xerr=M08xerr, fmt='s', color = '#ff3300', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Marois et al. 2008')
# plt.errorbar(C11microns, C11fluxes, yerr = C11err, xerr=C11xerr, fmt='o', color = '#ff3300', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2011')
# plt.errorbar(C14microns, C14fluxes, yerr = C14err, xerr=C14xerr,  fmt='v', color = '#ff3300', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Currie et al. 2014')
# plt.errorbar(G11microns, G11fluxes, yerr = G11err, xerr=G11xerr, fmt='^', color = '#ff3300', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Galicher et al. 2011')
# plt.errorbar(S12microns, S12fluxes, yerr = S12err, xerr=S12xerr, fmt='+', color = '#ff3300', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2012')
# plt.errorbar(S14microns, S14fluxes, yerr = S14err, xerr=S14xerr, fmt='o', color = '#ff3300', capsize=5,
#     elinewidth=1, markeredgewidth=1, label = 'Skemer et al. 2014')
# plt.errorbar(Z16microns, Z16fluxes, yerr = Z16err, xerr=Z16xerr, fmt='x', color = '#ff3300', capsize=5,
#     elinewidth=0, markeredgewidth=1, label = 'Zurlo et al. 2016')
#set up plot


plt.errorbar(G18Hmicrons, G18Hfluxes, yerr = G18Herr, color = '#ff9900', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'Greenbaum et al. 2018')
plt.fill_between(G18Hmicrons, G18Hfluxes-G18Herr,G18Hfluxes+G18Herr, color = '#ff9900', alpha=0.5)
plt.errorbar(G18Kmicrons, G18Kfluxes, yerr = G18Kerr, color = '#ff9900', capsize=5,
    elinewidth=1, markeredgewidth=1)
plt.fill_between(G18Kmicrons, G18Kfluxes-G18Kerr,G18Kfluxes+G18Kerr, color = '#ff9900', alpha=0.5)
plt.errorbar(G18K2microns, G18K2fluxes, yerr = G18K2err, color = '#ff9900', capsize=5,
    elinewidth=1, markeredgewidth=1)
plt.fill_between(G18K2microns, G18K2fluxes-G18K2err,G18K2fluxes+G18K2err, color = '#ff9900', alpha=0.5)

scaling_factor = np.nanmean(K13fluxes)/np.nanmean(model[np.where((K13microns[0]<wvs)*(wvs<K13microns[-1]))])
plt.plot(wvs,model*scaling_factor,linestyle="--",color="black",alpha=1,linewidth=0.5)

plt.errorbar(Pmicrons, Pfluxes, yerr = Perr, color = '#cccc00', capsize = 5,
    elinewidth=1, markeredgewidth=1, label = 'Oppenheimer et. al 2013')
plt.fill_between(Pmicrons, Pfluxes-Perr,Pfluxes+Perr, color = '#cccc00', alpha=0.25)

c_kms = 299792.458
plt.plot(K13microns*(1-75/c_kms), K13fluxes,label = 'Konopacky et al. 2013', color = '#ff9900', alpha=1,linewidth=0.5)

plt.ylabel("Flux (W/$m^2$/$\mu$m)", fontsize=fontsize)
plt.xlabel("$\lambda$ ($\mu$m)", fontsize=fontsize)
# plt.xlim([1.5,2.5])
lgd = plt.legend(loc="upper right",frameon=False,fontsize=fontsize*0.9,ncol=1)#,bbox_to_anchor=(1,1)
plt.tight_layout()
plt.show()


