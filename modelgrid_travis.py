__author__ = 'jruffio'


import csv
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits
import glob
import time

def LPFvsHPF(myvec,cutoff,nansmooth=10):
    myvec_cp = copy(myvec)
    #handling nans:
    wherenans = np.where(np.isnan(myvec_cp))
    for k in wherenans[0]:
        myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-nansmooth]):np.min([np.size(myvec_cp),k+nansmooth])])

    fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec

    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan
    return LPF_myvec,HPF_myvec


out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"

suffix = "all"
planet = "b"
grid = "Travis"

# gridfolder = "20200217_grid_travis"
gridfolder = "20200301_grid_travis"







# data_filename1 = "/data/osiris_data/HR_8799_c/20100715/reduced_jb/sherlock/20200217_grid_travis/s100715_a010001_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl10.fits"
# data_filename2 = "/data/osiris_data/s100715_a010001_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl10.fits"
# hdulist = pyfits.open(data_filename1)
# _,Nmodels,_,Nrvs,ny,nx = hdulist[0].data.shape
# logposterior1 = hdulist[0].data[-1,:,9,:,:,:]
#
# hdulist = pyfits.open(data_filename2)
# _,Nmodels,_,Nrvs,ny,nx = hdulist[0].data.shape
# logposterior2 = hdulist[0].data[-1,:,9,:,:,:]
#
# print(np.nanmax(logposterior1-logposterior2))
# exit()
# plot grid
if 0:
    gridname = os.path.join("/data/osiris_data/","hr8799b_modelgrid")


    IFSfilter = "Kbb"
    cutoff=40
    if IFSfilter=="Kbb": #Kbb 1965.0 0.25
        CRVAL1 = 1965.
        CDELT1 = 0.25
        nl=1665
        R=4000
    elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1473.
        CDELT1 = 0.2
        nl=1651
        R=4000#5000
    elif IFSfilter=="Jbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1180.
        CDELT1 = 0.15
        nl=1574
        R=4000
    dwv = CDELT1/1000.
    init_wv = CRVAL1/1000. # wv for first slice in mum
    dwv = CDELT1/1000. # wv interval between 2 slices in mum
    wvs=np.linspace(init_wv,init_wv+dwv*nl,nl,endpoint=False)

    planet_template_filename=os.path.join("/data/osiris_data/","planets_templates",
                                      "HR8799c_"+IFSfilter[0:1]+"_3Oct2018_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))
    with open(planet_template_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ')
        list_starspec = list(csv_reader)
        oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
        col_names = oriplanet_spec_str_arr[0]
        oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
        oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
        h8799c_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)

    import scipy.io as scio
    travis_spectrum = scio.readsav(os.path.join("/data/osiris_data/","planets_templates","HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save"))
    ori_planet_spec0 = np.array(travis_spectrum["fmod"])
    wmod0 = np.array(travis_spectrum["wmod"])/1.e4

    grid_filelist = glob.glob(os.path.join(gridname,"lte*-*-0.0.aces_hr8799b_pgs=4d6_Kzz=1d8_C=*_O=*_gs=5um.exoCH4_hiresHK.7.D2e.sorted"))
    gridconv_filelist = [grid_filename.replace("hiresHK.7.D2e.sorted","hiresHK.7.D2e.sorted_gaussconv_R{0}_{1}.csv".format(R,IFSfilter)) for grid_filename in grid_filelist]

    Tlist = np.array([int(float(os.path.basename(grid_filename).split("lte")[-1].split("-")[0])*100) for grid_filename in grid_filelist])
    logglist = np.array([-float(os.path.basename(grid_filename).split("-")[1]) for grid_filename in grid_filelist])
    Clist = np.array([float(os.path.basename(grid_filename).split("C=")[-1].split("_O")[0]) for grid_filename in grid_filelist])
    Olist = np.array([float(os.path.basename(grid_filename).split("O=")[-1].split("_gs")[0]) for grid_filename in grid_filelist])

    print(np.unique(Tlist))
    print(np.unique(logglist))
    print(np.unique(Clist))
    print(np.unique(Olist))
    print(np.unique(Clist)-np.unique(Olist))
    # exit()
    for grid_filename,gridconv_filename,T,logg,C,O in zip(grid_filelist,gridconv_filelist,Tlist,logglist,Clist,Olist):
        if not ((T == 800 ) or (T == 1200)) :
            continue
        # if not ((logg == -4.5 ) or (logg == -3) ) :
        #     continue
        # if not ((C == 8.48 and O == 8.82) or (C == 8.25 and O == 8.3)):
        #     continue
        # if not ((C == -8.48 and O == -8.82 and (logg == -3))):# or (C == -8.25 and O == -8.3 and (logg == -4.5 ))):
        #     continue
        # if T != 1000:
        #     continue
        if logg != -3.5:
            continue
        if C != 8.33:
            continue
        if O != 8.51:
            continue
        # if not ((C == 8.48 and O == 8.82 and logg == -3 and T == 1000)):# or (C == 8.25 and O == 8.3 and logg == -4.5 and T == 1000)):
        #     continue
        # if not (C == -8.33 and O == -8.51 and logg == -3.5 and T == 1100):
        #     continue
        print(gridconv_filename)
        with open(gridconv_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            list_starspec = list(csv_reader)
            oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
            col_names = oriplanet_spec_str_arr[0]
            oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
            oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
            planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)


        plt.figure(1)
        spec = planet_spec_func(wvs)
        plt.plot(wvs,spec/np.max(spec),label="T{0} logg{1} C{2} O{3}".format(T,logg,C,O),alpha=0.5)

        plt.figure(2)
        lpf,hpf = LPFvsHPF(planet_spec_func(wvs),cutoff)
        plt.plot(wvs,hpf/np.std(hpf),label="T{0} logg{1} C{2} O{3}".format(T,logg,C,O),alpha=0.5)


        out = np.loadtxt(grid_filename,skiprows=0)
        wmod = out[:,0]/1e4
        ori_planet_spec = 10**(out[:,1]-np.max(out[:,1]))
        plt.figure(3)
        plt.plot(wmod,ori_planet_spec/np.max(ori_planet_spec),label="T{0} logg{1} C{2} O{3}".format(T,logg,C,O),alpha=0.5)

    plt.figure(1)
    spec = h8799c_spec_func(wvs)
    plt.plot(wvs,spec/np.max(spec),label="Konopacky 2013",alpha=1, color="black")
    plt.legend()
    plt.figure(2)
    lpf,hpf = LPFvsHPF(h8799c_spec_func(wvs),cutoff)
    plt.plot(wvs,hpf/np.std(hpf),label="Konopacky 2013",alpha=1, color="black")
    plt.legend()
    plt.figure(3)
    plt.plot(wmod0,ori_planet_spec0/np.max(ori_planet_spec0),label="Konopacky 2013",alpha=1)
    plt.legend()
    plt.show()
    exit()
    print(Tlist)

    exit()

# numbasis = 3
for nbid,numbasis in enumerate([10]):
    if numbasis == 0:
        mylabel = "No PCA"
        mymark = "x"
        mycolor = "#ff9900"
    elif numbasis ==1:
        mylabel = "1 PCA mode"
        mymark = "o"
        mycolor = "#0099cc"
    elif numbasis ==2:
        mylabel = "{0} PCA modes".format(numbasis)
        mymark = "*"
        mycolor = "#6600ff"
    elif numbasis >2:
        mylabel = "{0} PCA modes".format(numbasis)
        mymark = "*"
        mycolor = "grey"

    if numbasis ==0:
        c_fileinfos_filename = "/data/osiris_data/HR_8799_"+planet+"/fileinfos_Kbb_jb.csv"
    else:
        c_fileinfos_filename = "/data/osiris_data/HR_8799_"+planet+"/fileinfos_Kbb_jb_kl{0}.csv".format(numbasis)

    print(c_fileinfos_filename)
    # exit()
    #read file
    with open(c_fileinfos_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        c_list_table = list(csv_reader)
        c_colnames = c_list_table[0]
        c_N_col = len(c_colnames)
        c_list_data = c_list_table[1::]

        try:
            c_cen_filename_id = c_colnames.index("cen filename")
            c_kcen_id = c_colnames.index("kcen")
            c_lcen_id = c_colnames.index("lcen")
            c_rvcen_id = c_colnames.index("RVcen")
            c_rvcensig_id = c_colnames.index("RVcensig")
        except:
            pass
        try:
            c_rvfakes_id = c_colnames.index("RVfakes")
            c_rvfakessig_id = c_colnames.index("RVfakessig")
        except:
            pass
        c_filename_id = c_colnames.index("filename")
        c_mjdobs_id = c_colnames.index("MJD-OBS")
        c_bary_rv_id = c_colnames.index("barycenter rv")
        c_ifs_filter_id = c_colnames.index("IFS filter")
        c_xoffset_id = c_colnames.index("header offset x")
        c_yoffset_id = c_colnames.index("header offset y")
        c_sequence_id = c_colnames.index("sequence")
        c_status_id = c_colnames.index("status")
        c_wvsolerr_id = c_colnames.index("wv sol err")
        c_snr_id = c_colnames.index("snr")
        c_DTMP6_id = c_colnames.index("DTMP6")
    c_filelist = np.array([item[c_filename_id] for item in c_list_data])
    c_out_filelist = np.array([item[c_cen_filename_id] for item in c_list_data])
    c_kcen_list = np.array([int(item[c_kcen_id]) if item[c_kcen_id] != "nan" else 0 for item in c_list_data])
    c_lcen_list = np.array([int(item[c_lcen_id]) if item[c_lcen_id] != "nan" else 0  for item in c_list_data])
    c_date_list = np.array([item[c_filename_id].split(os.path.sep)[4] for item in c_list_data])
    c_ifs_fitler_list = np.array([item[c_ifs_filter_id] for item in c_list_data])
    c_snr_list = np.array([float(item[c_snr_id]) if item[c_snr_id] != "nan" else 0  for item in c_list_data])
    c_DTMP6_list = np.array([float(item[c_DTMP6_id]) if item[c_DTMP6_id] != "nan" else 0  for item in c_list_data])
    c_cen_filelist = np.array([item[c_cen_filename_id] for item in c_list_data])
    c_status_list = np.array([int(item[c_status_id]) for item in c_list_data])

    posterior_list = []
    N = 0
    for c_cen_filename,status,ifsfilter in zip(c_cen_filelist,c_status_list,c_ifs_fitler_list):
        if status !=1:
            continue
        if "Hbb" in ifsfilter:
            continue

        data_filename = c_cen_filename.replace("20191205_RV",gridfolder).replace("search","centroid")
        print(data_filename)
        # exit()
        modelgrid_filename = data_filename.replace(".fits","_modelgrid.txt")
        # modelgrid_filename = "/data/osiris_data/"+"HR_8799_c"+"/20"+"100715"+"/reduced_jb/20191209_gridtest/s100715_a010001_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl0_modelgrid.txt"
        print(modelgrid_filename)
        try:
            txtfile = open(modelgrid_filename, 'r')
        except:
            continue
        grid_filelist = [s.strip() for s in txtfile.readlines()]
        print(grid_filelist[0])
        Tlist = np.array([int(float(os.path.basename(grid_filename).split("lte")[-1].split("-")[0])*100) for grid_filename in grid_filelist])
        logglist = np.array([-float(os.path.basename(grid_filename).split("-")[1]) for grid_filename in grid_filelist])
        Clist = np.array([float(os.path.basename(grid_filename).split("C=")[-1].split("_O")[0]) for grid_filename in grid_filelist])
        Olist = np.array([float(os.path.basename(grid_filename).split("O=")[-1].split("_gs")[0]) for grid_filename in grid_filelist])

        selec_models = np.where(Tlist==900)
        Tlist = Tlist[selec_models]
        logglist = logglist[selec_models]
        Clist = Clist[selec_models]
        Olist = Olist[selec_models]
        print(selec_models)
        # exit()

        print(np.unique(Tlist))
        # print(np.unique(logglist))
        hdulist = pyfits.open(data_filename)
        _,Nmodels,_,Nrvs,ny,nx = hdulist[0].data.shape
        logposterior = hdulist[0].data[-1,selec_models[0],9,:,:,:]
        posterior = np.exp(logposterior-np.nanmax(logposterior))
        posterior_posmargi = np.nansum(posterior,axis=(2,3))
        posterior_posmargi /= np.nanmax(posterior_posmargi)
        posterior_rvposmargi = np.nansum(posterior,axis=(1,2,3))
        posterior_rvposmargi /= np.nanmax(posterior_posmargi)
        print(np.nansum(posterior_rvposmargi))
        posterior_list.append(posterior_rvposmargi)


        # plt.figure(nbid+20)
        # plt.scatter(Clist,Olist,c=posterior_rvposmargi,s=100*posterior_rvposmargi)
        # plt.plot(Clist,Olist,"x",color="black")
        # plt.xlabel("C")
        # plt.ylabel("O")
        # plt.show()

        # plt.figure(1)
        # plt.plot(Tlist,posterior_rvposmargi,"x")
        # plt.xlabel("T (K)")
        # plt.ylabel("Posterior")
        # plt.show()

        # print(posterior_rvposmargi.shape)

    log10_combined_posterior = np.nansum(np.log10(posterior_list),axis=0)
    # log10_combined_posterior = np.log10(np.nanmean(posterior_list,axis=0))*len(posterior_list)
    log10_combined_posterior -= np.nanmax(log10_combined_posterior)
    combined_posterior = 10**(log10_combined_posterior)
    print(combined_posterior)

    # plt.figure(nbid)
    # plt.plot(combined_posterior)
    # print(grid_filelist[np.argmax(combined_posterior)])
    # print("N cubes",len(posterior_list))

    plt.figure(nbid+10,figsize=(12,3))
    plt.subplot(1,3,1)
    plt.plot(Tlist,combined_posterior,"x")
    plt.xlabel("T (K)")
    plt.ylabel("Posterior")
    plt.title("HR 8799 "+planet)

    plt.subplot(1,3,2)
    plt.plot(logglist,combined_posterior,"x")
    plt.xlabel("log(g)")

    plt.subplot(1,3,3)
    plt.plot(10**(Clist-Olist),combined_posterior,"x")
    plt.xlabel("C/O")

    plt.tight_layout()

plt.show()
