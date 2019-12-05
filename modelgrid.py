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

for mydir in glob.glob(os.path.join("/data/osiris_data/HR_8799_*/","*","reduced_jb")):
    # for myinsidedir in os.listdir(mydir):
    #     if os.path.isdir(os.path.join(mydir,myinsidedir)) and "sherlock" not in myinsidedir:
    #         os.system("du -sh {0}".format(os.path.join(mydir,myinsidedir)))
    #         os.system("rm -R {0}".format(os.path.join(mydir,myinsidedir)))
    #         time.sleep(1)
    try:
        for mysherlockdir in os.listdir(os.path.join(mydir,"sherlock")):
            if os.path.isdir(os.path.join(mydir,"sherlock",mysherlockdir)):# and "20190416_no_persis_corr" != myinsidedir :
        #         pass
                if "20190416_no_persis_corr" != mysherlockdir \
                        and "20191204_grid" != mysherlockdir \
                        and "20191120_newres_RV" != mysherlockdir \
                        and "20191202_newresmodel" != mysherlockdir \
                        and "20191120_newresmodel" != mysherlockdir \
                        and "20190510_spec_esti" != mysherlockdir \
                        and "20190508_models2" != mysherlockdir \
                        and "20191018_RVsearch" != mysherlockdir \
                        and "logs" != mysherlockdir:
                    os.system("du -sh {0}".format(os.path.join(mydir,"sherlock",mysherlockdir)))
                    os.system("rm -R {0}".format(os.path.join(mydir,"sherlock",mysherlockdir)))
                    time.sleep(1)
    except:
        print("failed "+ mydir)

exit()

out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"

suffix = "all"

# numbasis = 3
for nbid,numbasis in enumerate([0,1]):
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
        c_fileinfos_filename = "/data/osiris_data/HR_8799_c/fileinfos_Kbb_jb.csv"
    else:
        c_fileinfos_filename = "/data/osiris_data/HR_8799_c/fileinfos_Kbb_jb_kl{0}.csv".format(numbasis)


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
    for c_cen_filename,status,ifsfilter in zip(c_cen_filelist,c_status_list,c_ifs_fitler_list):
        if status !=1:
            continue
        if "Hbb" in ifsfilter:
            continue
        print(c_cen_filelist[0])
        print(c_cen_filelist[0].replace("20191104_RVsearch","20191204_grid").replace("search","centroid").replace(".fits","_planetRV.fits"))
        print(c_cen_filelist[0].replace("20191104_RVsearch","20191204_grid").replace("search","centroid"))

        data_filename = c_cen_filename.replace("20191104_RVsearch","20191204_grid").replace("search","centroid")
        hdulist = pyfits.open(data_filename)
        _,Nmodels,_,Nrvs,ny,nx = hdulist[0].data.shape
        logposterior = hdulist[0].data[-1,:,9,:,:,:]
        posterior = np.exp(logposterior-np.nanmax(logposterior))
        posterior_posmargi = np.nansum(posterior,axis=(2,3))
        posterior_posmargi /= np.nanmax(posterior_posmargi)
        posterior_rvposmargi = np.nansum(posterior,axis=(1,2,3))
        posterior_rvposmargi /= np.nanmax(posterior_posmargi)
        posterior_list.append(posterior_rvposmargi)
        print(posterior_rvposmargi.shape)

    modelgrid_filename = data_filename.replace(".fits","_modelgrid.txt")
    print(modelgrid_filename)
    with open(modelgrid_filename, 'r') as txtfile:
        grid_filelist = [s.strip() for s in txtfile.readlines()]
        Tlist = np.array([int(os.path.basename(grid_filename).split("_t")[-1].split("g")[0]) for grid_filename in grid_filelist])
        glist = np.array([int(os.path.basename(grid_filename).split("g")[-1].split("nc")[0]) for grid_filename in grid_filelist])
        logglist = np.log10(glist*100)
        print(np.unique(Tlist))
        print(np.unique(logglist))

    combined_posterior = np.prod(posterior_list,axis=0)

    plt.figure(nbid+10)
    plt.plot(combined_posterior)
    print(grid_filelist[np.argmax(combined_posterior)])

    plt.figure(nbid)
    plt.scatter(Tlist,glist,c=combined_posterior,s=100)
    plt.xlabel("T (K)")
    plt.ylabel("g (cm/s)")
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
