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


# BE REALLY CAREFUL
# if 0: # delete old folders in osiris_data!
#     for mydir in glob.glob(os.path.join("/data/osiris_data/HR_8799_*/","*","reduced_jb")):
#         # for myinsidedir in os.listdir(mydir):
#         #     if os.path.isdir(os.path.join(mydir,myinsidedir)) and "sherlock" not in myinsidedir:
#         #         os.system("du -sh {0}".format(os.path.join(mydir,myinsidedir)))
#         #         os.system("rm -R {0}".format(os.path.join(mydir,myinsidedir)))
#         #         time.sleep(1)
#         try:
#             for mysherlockdir in os.listdir(os.path.join(mydir,"sherlock")):
#                 if os.path.isdir(os.path.join(mydir,"sherlock",mysherlockdir)):# and "20190416_no_persis_corr" != myinsidedir :
#             #         pass
#                     if "20190416_no_persis_corr" != mysherlockdir \
#                             and "20191204_grid" != mysherlockdir \
#                             and "20191120_newres_RV" != mysherlockdir \
#                             and "20191202_newresmodel" != mysherlockdir \
#                             and "20191120_newresmodel" != mysherlockdir \
#                             and "20190510_spec_esti" != mysherlockdir \
#                             and "20190508_models2" != mysherlockdir \
#                             and "20191018_RVsearch" != mysherlockdir \
#                             and "logs" != mysherlockdir:
#                         os.system("du -sh {0}".format(os.path.join(mydir,"sherlock",mysherlockdir)))
#                         # os.system("rm -R {0}".format(os.path.join(mydir,"sherlock",mysherlockdir)))
#                         time.sleep(1)
#         except:
#             print("failed "+ mydir)


# interpolate and save grid spectra
if 0:
    ifsfilter = "Kbb"
    R=4000
    whichgrid = "BTsettl"
    if "sonora" in whichgrid:
        grid_folder = os.path.join("/data/osiris_data","sonora","spectra")
        Tlist =  np.array([ 200,  225  ,250  ,275  ,300  ,325  ,350  ,375  ,400  ,425  ,450  ,475  ,500  ,525  ,550  ,575  ,600  ,650  ,700  ,750  ,800  ,850  ,900  ,950 ,1000 ,1100 ,1200 ,1300 ,1400 ,1500 ,1600 ,1700 ,1800 ,1900 ,2000 ,2100 ,2200 ,2300 ,2400])
        para2list = np.array([  10   ,17   ,31   ,56  ,100  ,178  ,316  ,562 ,1000 ,1780 ,3160])
        Tlist = Tlist[np.where((1500<Tlist)*(Tlist<2000))]
        para2list = para2list[np.where((500<para2list)*(para2list<2000))]
        new_Tlist= np.arange(1700,1900,5)
        new_para2list = np.arange(750,1300,50)
    if "BTsettl" in whichgrid:
        grid_folder = os.path.join("/data/osiris_data","BTsettl")
        Tlist = np.array([1200, 1300 ,1400 ,1500,1550 ,1600 ,1650 ,1700 ,1750 ,1800 ,1850 ,1900 ,1950 ,
                  2000,2050 ,2100 ,2150 ,2200 ,2250 ,2300 ,2350 ,2400 ,2500 ,2600 ,2700 ,2800 ,
                  2900 ,3000,3100 ,3200 ,3300 ,3400 ,3500 ,3600 ,3700 ,3800 ,3900 ,4000 ,4100 ,
                  4200 ,4300 ,4400,4500, 4600 ,4700 ,4800 ,4900 ,5000 ,5100 ,5200 ,5300 ,5400 ,
                  5500 ,5600 ,5700 ,5800,5900 ,6000 ,6100 ,6200 ,6300 ,6400 ,6500 ,6600 ,6700 ,6800 ,6900 ,7000])
        para2list = np.array([-5.5, -5.  ,-4.5 ,-4.  ,-3.5 ,-3. , -2.5])
        Tlist = Tlist[np.where((1100<Tlist)*(Tlist<1510))]
        para2list = para2list[np.where((-4.2<para2list)*(para2list<-2.8))]
        new_Tlist= np.arange(1300,1500,5)
        new_para2list = np.arange(-3.8,-3.2,0.05)


    if "sonora" in whichgrid:
        filename = os.path.join(grid_folder,"sp_t{0}g{1}nc_m0.0_gaussconv_R4000_{2}.csv".format(Tlist[0],para2list[0],ifsfilter))
    if "BTsettl" in whichgrid:
        filename = os.path.join(grid_folder,"lte{0:0>3d}.{1:1d}-{2:.1f}-0.0a+0.0.BT-Settl.spec_gaussconv_R4000_{3}.csv".format(int(Tlist[0]/100),int((Tlist[0]/100-int(Tlist[0]/100))*10),-para2list[0],ifsfilter))
    data_arr=np.loadtxt(filename,skiprows=1)
    mywvs0 = data_arr[:,0]
    # exit()

    data = np.zeros((np.size(Tlist),np.size(para2list),np.size(mywvs0)))
    for wid, temp in enumerate(Tlist):
        for aid, para2 in enumerate(para2list):
            print(temp,para2)
            if "sonora" in whichgrid:
                filename = os.path.join(grid_folder,"sp_t{0}g{1}nc_m0.0_gaussconv_R4000_{2}.csv".format(temp,para2,ifsfilter))
            if "BTsettl" in whichgrid:
                filename = os.path.join(grid_folder,"lte{0:0>3d}.{1:1d}-{2:.1f}-0.0a+0.0.BT-Settl.spec_gaussconv_R4000_{3}.csv".format(int(temp/100),int((temp/100-int(temp/100))*10),-para2,ifsfilter))
            data_arr=np.loadtxt(filename,skiprows=1)
            mywvs = data_arr[:,0]
            # if np.sum(mywvs0-mywvs) != 0:
            #     print("wavelengths not compatible")
            #     exit()
            myspec = data_arr[:,1]
            data[wid,aid,:] = myspec

    from scipy.interpolate import RegularGridInterpolator
    interp_object = RegularGridInterpolator((Tlist,para2list),data,method="linear",bounds_error=False,fill_value=np.nan)

    N = 0
    if "sonora" in whichgrid:
        outputdir =  os.path.join(grid_folder,"interpolated_{0}-{1}_{2}-{3}".format(new_Tlist[0],new_Tlist[-1],new_para2list[0],new_para2list[-1]))
    if "BTsettl" in whichgrid:
        outputdir =  os.path.join(grid_folder,"interpolated_{0}-{1}_{2:.2f}-{3:.2f}".format(new_Tlist[0],new_Tlist[-1],new_para2list[0],new_para2list[-1]))
    if not os.path.exists(os.path.join(outputdir)):
        os.makedirs(os.path.join(outputdir))
    for wid, temp in enumerate(new_Tlist):
        for aid, para2 in enumerate(new_para2list):
            N+=1
            if "sonora" in whichgrid:
                out_filename = os.path.join(outputdir,"sp_t{0}g{1}nc_m0.0_gaussconv_R4000_{2}.csv".format(temp,para2,ifsfilter))
            if "BTsettl" in whichgrid:
                out_filename = os.path.join(outputdir,"lte{0:0>3d}.{1:1d}-{2:.1f}-0.0a+0.0.BT-Settl.spec_gaussconv_R4000_{3}.csv".format(int(temp/100),int((temp/100-int(temp/100))*10),-para2,ifsfilter))
            with open(out_filename, 'w+') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ')
                csvwriter.writerows([["wvs","spectrum"]])
                csvwriter.writerows([[a,b] for a,b in zip(mywvs0,interp_object((temp,para2)))])
            # exit()
    print(N)
    #
    # filename = os.path.join(sky_transmission_folder,"mktrans_zm_{0}_{1}.dat".format(10,10))
    # skybg_arr=np.loadtxt(filename)
    # skytrans_spec = skybg_arr[:,1]
    # skytrans_spec = skytrans_spec[selec_skytrans]
    # skytrans_spec = convolve_spectrum(skytrans_wvs,skytrans_spec,R,specpool)
    # plt.plot(skytrans_spec)
    # filename = os.path.join(sky_transmission_folder,"mktrans_zm_{0}_{1}.dat".format(10,15))
    # skybg_arr=np.loadtxt(filename)
    # skytrans_spec = skybg_arr[:,1]
    # skytrans_spec = skytrans_spec[selec_skytrans]
    # skytrans_spec = convolve_spectrum(skytrans_wvs,skytrans_spec,R,specpool)
    # plt.plot(skytrans_spec)
    # plt.plot(interp_object((10,12.5)),"--")
    # plt.show()
    exit()

# exit()

out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"

suffix = "all"
# grid = "BTsettl"
grid = "sonora"

if "BTsettl" == grid:
    gridfolder = "20191204_grid"
    # gridfolder = "20191209_grid_sonora"

if "sonora" == grid:
    gridfolder = "20191204_grid_sonora"

# numbasis = 3
for nbid,numbasis in enumerate([1]):
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
        c_fileinfos_filename = "/data/osiris_data/HR_8799_d/fileinfos_Kbb_jb.csv"
    else:
        c_fileinfos_filename = "/data/osiris_data/HR_8799_d/fileinfos_Kbb_jb_kl{0}.csv".format(numbasis)


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
    for c_cen_filename,status,ifsfilter in zip(c_cen_filelist[::2],c_status_list[::2],c_ifs_fitler_list[::2]):
        if status !=1:
            continue
        if "Hbb" in ifsfilter:
            continue

        data_filename = c_cen_filename.replace("20191205_RV",gridfolder).replace("search","centroid")
        print(data_filename)
        try:
            hdulist = pyfits.open(data_filename)
        except:
            continue
        _,Nmodels,_,Nrvs,ny,nx = hdulist[0].data.shape
        logposterior = hdulist[0].data[-1,:,9,:,:,:]
        posterior = np.exp(logposterior-np.nanmax(logposterior))
        posterior_posmargi = np.nansum(posterior,axis=(2,3))
        posterior_posmargi /= np.nanmax(posterior_posmargi)
        posterior_rvposmargi = np.nansum(posterior,axis=(1,2,3))
        posterior_rvposmargi /= np.nanmax(posterior_posmargi)
        print(np.nansum(posterior_rvposmargi))
        posterior_list.append(posterior_rvposmargi)


        modelgrid_filename = data_filename.replace(".fits","_modelgrid.txt")
        # modelgrid_filename = "/data/osiris_data/"+"HR_8799_c"+"/20"+"100715"+"/reduced_jb/20191209_gridtest/s100715_a010001_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl0_modelgrid.txt"
        print(modelgrid_filename)
        with open(modelgrid_filename, 'r') as txtfile:
            grid_filelist = [s.strip() for s in txtfile.readlines()]
            if grid == "sonora":
                Tlist = np.array([int(os.path.basename(grid_filename).split("_t")[-1].split("g")[0]) for grid_filename in grid_filelist])
                para2list = np.array([int(os.path.basename(grid_filename).split("g")[1].split("nc")[0]) for grid_filename in grid_filelist])
                logglist = np.log10(para2list*100)
            if grid == "BTsettl":
                Tlist = np.array([int(float(os.path.basename(grid_filename).split("lte")[-1].split("-")[0])*100) for grid_filename in grid_filelist])
                para2list = np.array([-float(os.path.basename(grid_filename).split("-")[1].split("-0.0a")[0]) for grid_filename in grid_filelist])
            print(np.unique(Tlist))
            # print(np.unique(logglist))

        # plt.figure(nbid)
        # plt.scatter(Tlist,para2list,c=posterior_rvposmargi,s=100)
        # plt.xlabel("T (K)")
        # if grid == "sonora":
        #     plt.ylabel("g (cm/s)")
        # if grid == "BTsettl":
        #     plt.ylabel("log[M/H]")
        # plt.show()

        # plt.figure(1)
        # plt.subplot(1,2,1)
        # plt.plot(posterior_rvposmargi)
        # print(grid_filelist[np.argmax(posterior_rvposmargi)])
        #
        # plt.subplot(1,2,2)
        # plt.scatter(Tlist,para2list,c=posterior_rvposmargi,s=10)
        # plt.xlabel("T (K)")
        # if grid == "sonora":
        #     plt.ylabel("g (cm/s)")
        # if grid == "BTsettl":
        #     plt.ylabel("log[M/H]")
        # plt.show()

        # print(posterior_rvposmargi.shape)

    log10_combined_posterior = np.nansum(np.log10(posterior_list),axis=0)
    # log10_combined_posterior = np.log10(np.nanmean(posterior_list,axis=0))*len(posterior_list)
    log10_combined_posterior -= np.nanmax(log10_combined_posterior)
    combined_posterior = 10**(log10_combined_posterior)
    print(combined_posterior)

    plt.figure(nbid+10)
    plt.plot(combined_posterior)
    print(grid_filelist[np.argmax(combined_posterior)])
    print("N cubes",len(posterior_list))

    plt.figure(nbid)
    plt.scatter(Tlist,para2list,c=combined_posterior,s=100)
    plt.xlabel("T (K)")
    if grid == "sonora":
        plt.ylabel("g (cm/s)")
    if grid == "BTsettl":
        plt.ylabel("log[M/H]")

plt.show()


# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl1.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl1_autocorrres.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl1_estispec.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl1_klgrids.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl1_modelgrid.txt
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl1_out1dfit.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl1_planetRV.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_centroid_resinmodel_kl1_res.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_rescalc.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_rescalc_autocorrres.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_rescalc_estispec.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_rescalc_out1dfit.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_rescalc_planetRV.fits
# 20171103/reduced_jb/sherlock/20191204_grid/s171103_a026002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_rescalc_res.fits
