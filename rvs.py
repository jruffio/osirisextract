__author__ = 'jruffio'


import csv
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"

suffix = "all"

c_fileinfos_filename = "/data/osiris_data/HR_8799_c/fileinfos_Kbb_jb.csv"

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
    c_filename_id = c_colnames.index("filename")
    c_mjdobs_id = c_colnames.index("MJD-OBS")
    c_bary_rv_id = c_colnames.index("barycenter rv")
    c_ifs_filter_id = c_colnames.index("IFS filter")
    c_xoffset_id = c_colnames.index("header offset x")
    c_yoffset_id = c_colnames.index("header offset y")
    c_sequence_id = c_colnames.index("sequence")
    c_status_id = c_colnames.index("status")
    c_wvsolerr_id = c_colnames.index("wv sol err")
c_filelist = [item[c_filename_id] for item in c_list_data]

b_fileinfos_filename = "/data/osiris_data/HR_8799_b/fileinfos_Kbb_jb.csv"
#read file
with open(b_fileinfos_filename, 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=';')
    b_list_table = list(csv_reader)
    b_colnames = b_list_table[0]
    b_N_col = len(b_colnames)
    b_list_data = b_list_table[1::]

    try:
        b_cen_filename_id = b_colnames.index("cen filename")
        b_kcen_id = b_colnames.index("kcen")
        b_lcen_id = b_colnames.index("lcen")
        b_rvcen_id = b_colnames.index("RVcen")
        b_rvcensig_id = b_colnames.index("RVcensig")
    except:
        pass
    b_filename_id = b_colnames.index("filename")
    b_mjdobs_id = b_colnames.index("MJD-OBS")
    b_bary_rv_id = b_colnames.index("barycenter rv")
    b_ifs_filter_id = b_colnames.index("IFS filter")
    b_xoffset_id = b_colnames.index("header offset x")
    b_yoffset_id = b_colnames.index("header offset y")
    b_sequence_id = b_colnames.index("sequence")
    b_status_id = b_colnames.index("status")
    b_wvsolerr_id = b_colnames.index("wv sol err")
b_filelist = [item[b_filename_id] for item in b_list_data]

# plot RVs
if 1:
    rv_star = -12.6#-12.6+-1.4km/s HR 8799 Rob and Simbad
    b_bary_star_list = np.array([-float(item[b_bary_rv_id])/1000+rv_star for item in b_list_data])
    b_mjdobs_list = np.array([float(item[b_mjdobs_id]) for item in b_list_data])
    # b_mjdobs_list = np.arange(np.size(b_mjdobs_list))
    b_baryrv_list = np.array([-float(item[b_bary_rv_id])/1000 for item in b_list_data])
    b_rv_list = np.array([float(item[b_rvcen_id]) for item in b_list_data])
    b_status_list = np.array([float(item[b_status_id]) for item in b_list_data])
    b_rvsig_list = np.array([float(item[b_rvcensig_id]) for item in b_list_data])
    b_wvsolerr_list = np.array([float(item[b_wvsolerr_id]) for item in b_list_data])
    b_good_rv_list = copy(b_rv_list)
    b_good_rv_list[np.where(b_status_list<1)] = np.nan
    b_bad_rv_list= copy(b_rv_list)
    b_bad_rv_list[np.where(b_status_list>0)] = np.nan

    c_bary_star_list = np.array([-float(item[c_bary_rv_id])/1000+rv_star for item in c_list_data])
    c_mjdobs_list = np.array([float(item[c_mjdobs_id]) for item in c_list_data])
    # c_mjdobs_list = np.arange(np.size(c_mjdobs_list))
    c_baryrv_list = np.array([-float(item[c_bary_rv_id])/1000 for item in c_list_data])
    c_rv_list = np.array([float(item[c_rvcen_id]) for item in c_list_data])
    c_status_list = np.array([float(item[c_status_id]) for item in c_list_data])
    c_rvsig_list = np.array([float(item[c_rvcensig_id]) for item in c_list_data])
    c_wvsolerr_list = np.array([float(item[c_wvsolerr_id]) for item in c_list_data])
    c_good_rv_list = copy(c_rv_list)
    c_good_rv_list[np.where(c_status_list<1)] = np.nan
    c_bad_rv_list= copy(c_rv_list)
    c_bad_rv_list[np.where(c_status_list>0)] = np.nan


    plt.figure(1,figsize=(9,0.75*9))
    plt.subplot(2,1,1)

    plt.errorbar(c_mjdobs_list,c_good_rv_list,yerr=np.sqrt(c_rvsig_list**2+c_wvsolerr_list**2),fmt="x",color="red",label="c Measured raw RV ($1\sigma$)")
    plt.errorbar(c_mjdobs_list,c_good_rv_list,yerr=c_rvsig_list,fmt="x",color="pink",label="c Measured raw RV ($1\sigma$)")
    # plt.plot(c_mjdobs_list,c_bad_rv_list,linestyle="",marker="o",color="blue",label="Bad Data")
    plt.plot(c_mjdobs_list,c_baryrv_list,color="#006699",label="c Barycentric RV")
    plt.plot(c_mjdobs_list,c_bary_star_list,color="#ff9900",label="c Barycentric + HR8799 RV")

    plt.errorbar(b_mjdobs_list,b_good_rv_list,yerr=np.sqrt(b_rvsig_list**2+b_wvsolerr_list**2),fmt="x",color="green",label="b Measured raw RV ($1\sigma$)")
    plt.errorbar(b_mjdobs_list,b_good_rv_list,yerr=b_rvsig_list,fmt="x",color="cyan",label="b Measured raw RV ($1\sigma$)")
    # plt.plot(b_mjdobs_list,b_bad_rv_list,linestyle="",marker="o",color="blue",label="Bad Data")
    plt.plot(b_mjdobs_list,b_baryrv_list,color="#006699",label="b Barycentric RV")
    plt.plot(b_mjdobs_list,b_bary_star_list,color="#ff9900",label="b Barycentric + HR8799 RV")

    plt.xlabel("Exposure Index",fontsize=15)
    plt.ylabel("RV (km/s)",fontsize=15)
    plt.legend(fontsize=10)
    plt.legend(fontsize=10,loc="upper left")
    plt.subplot(2,1,2)
    # plt.plot(rv_list-bary_star_list,"x",color="red",label="Estimated Planet RV")

    # plt.plot(c_mjdobs_list,np.zeros(c_rv_list.shape)+np.nanmean(c_rv_list-c_bary_star_list),linestyle=":",color="pink",label="Mean Planet RV")
    plt.errorbar(c_mjdobs_list,c_good_rv_list-c_bary_star_list,yerr=np.sqrt(c_rvsig_list**2+c_wvsolerr_list**2),fmt="x",color="red",label="c Estimated Planet RV ($1\sigma$)")
    plt.errorbar(c_mjdobs_list,c_good_rv_list-c_bary_star_list,yerr=c_rvsig_list,fmt="x",color="pink",label="c Estimated Planet RV ($1\sigma$)")
    # plt.plot(c_mjdobs_list,c_bad_rv_list-c_bary_star_list,linestyle="",marker="o",color="blue",label="Bad data")

    # plt.plot(b_mjdobs_list,np.zeros(b_rv_list.shape)+np.nanmean(b_rv_list-b_bary_star_list),linestyle=":",color="pink",label="Mean Planet RV")
    plt.errorbar(b_mjdobs_list,b_good_rv_list-b_bary_star_list,yerr=np.sqrt(b_rvsig_list**2+b_wvsolerr_list**2),fmt="x",color="green",label="b Estimated Planet RV ($1\sigma$)")
    plt.errorbar(b_mjdobs_list,b_good_rv_list-b_bary_star_list,yerr=b_rvsig_list,fmt="x",color="cyan",label="b Estimated Planet RV ($1\sigma$)")
    # plt.plot(b_mjdobs_list,b_bad_rv_list-b_bary_star_list,linestyle="",marker="o",color="blue",label="Bad data")

    plt.xlabel("Exposure Index",fontsize=15)
    plt.ylabel("RV (km/s)",fontsize=15)
    # print(np.nansum((c_good_rv_list-c_bary_star_list)/c_rvsig_list)/(np.sum(1./c_rvsig_list[np.where(np.isfinite(c_good_rv_list))])),
    #     np.sqrt(np.size(c_rvsig_list[np.where(np.isfinite(c_good_rv_list))]))/(np.sum(1./c_rvsig_list[np.where(np.isfinite(c_good_rv_list))])))
    # print(np.nanmean(c_good_rv_list[10:50]-c_bary_star_list[10:50]))
    # plt.ylim([-20,20])
    plt.legend(fontsize=10,loc="upper left")
    # plt.show()
    # print("Saving "+os.path.join(out_pngs,"HR8799"+planet+"_"+IFSfilter+"_RV.pdf"))
    # plt.savefig(os.path.join(out_pngs,"HR8799"+planet+"_"+IFSfilter+"_RV.pdf"),bbox_inches='tight')
    # plt.savefig(os.path.join(out_pngs,"HR8799"+planet+"_"+IFSfilter+"_RV.png"),bbox_inches='tight')

    plt.figure(2)
    where_b = np.where(np.isfinite(b_good_rv_list)*(b_mjdobs_list > 55388)*(b_mjdobs_list < 56000))
    print(np.size(where_b[0]))
    where_c = np.where(np.isfinite(c_good_rv_list)*(c_mjdobs_list > 55388)*(c_mjdobs_list < 56000))
    print(np.size(where_c[0]))
    plt.subplot(3,1,1)
    plt.errorbar(c_good_rv_list[where_c]-c_bary_star_list[where_c],np.arange(np.size(where_c[0])),xerr=np.sqrt(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2),fmt="x",color="red",label="c Measured raw RV ($1\sigma$)")
    plt.errorbar(c_good_rv_list[where_c]-c_bary_star_list[where_c],np.arange(np.size(where_c[0])),xerr=c_rvsig_list[where_c],fmt="x",color="pink",label="c Measured raw RV ($1\sigma$)")
    plt.xlim([-15,15])
    plt.subplot(3,1,2)
    plt.errorbar(b_good_rv_list[where_b]-b_bary_star_list[where_b],np.arange(np.size(where_b[0])),xerr=np.sqrt(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2),fmt="x",color="green",label="b Measured raw RV ($1\sigma$)")
    plt.errorbar(b_good_rv_list[where_b]-b_bary_star_list[where_b],np.arange(np.size(where_b[0])),xerr=b_rvsig_list[where_b],fmt="x",color="cyan",label="b Measured raw RV ($1\sigma$)")
    plt.xlim([-15,15])
    plt.subplot(3,1,3)
    plt.hist(b_good_rv_list[where_b]-b_bary_star_list[where_b], label="b", range=[-20,20],bins=20,histtype="step")
    print(np.sqrt((b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2)))
    b_combined_avg,b_combined_sig = np.mean(b_good_rv_list[where_b]-b_bary_star_list[where_b]),np.sqrt(np.sum(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2)/np.size(where_b[0])**2)
    print(b_combined_avg,b_combined_sig)
    plt.hist(c_good_rv_list[where_c]-c_bary_star_list[where_c], label="c", range=[-20,20],bins=20,histtype="step")
    print(np.sqrt(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))
    c_combined_avg,c_combined_sig = np.mean(c_good_rv_list[where_c]-c_bary_star_list[where_c]),np.sqrt(np.sum(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2)/np.size(where_c[0])**2)
    print(c_combined_avg,c_combined_sig )
    print((b_combined_avg-c_combined_avg),np.sqrt((c_combined_sig**2+b_combined_sig**2)))
    plt.xlim([-15,15])
    plt.legend()

    plt.figure(4)
    where_b = np.where(np.isfinite(b_good_rv_list))
    print(np.size(where_b[0]))
    where_c = np.where(np.isfinite(c_good_rv_list))
    print(np.size(where_c[0]))
    plt.subplot(3,1,1)
    plt.errorbar(c_good_rv_list[where_c]-c_bary_star_list[where_c],np.arange(np.size(where_c[0])),xerr=np.sqrt(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2),fmt="x",color="red",label="c Measured raw RV ($1\sigma$)")
    plt.errorbar(c_good_rv_list[where_c]-c_bary_star_list[where_c],np.arange(np.size(where_c[0])),xerr=c_rvsig_list[where_c],fmt="x",color="pink",label="c Measured raw RV ($1\sigma$)")
    plt.xlim([-15,15])
    plt.subplot(3,1,2)
    plt.errorbar(b_good_rv_list[where_b]-b_bary_star_list[where_b],np.arange(np.size(where_b[0])),xerr=np.sqrt(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2),fmt="x",color="green",label="b Measured raw RV ($1\sigma$)")
    plt.errorbar(b_good_rv_list[where_b]-b_bary_star_list[where_b],np.arange(np.size(where_b[0])),xerr=b_rvsig_list[where_b],fmt="x",color="cyan",label="b Measured raw RV ($1\sigma$)")
    plt.xlim([-15,15])
    plt.subplot(3,1,3)
    plt.hist(b_good_rv_list[where_b]-b_bary_star_list[where_b], label="b", range=[-20,20],bins=20,histtype="step")
    print(np.sqrt((b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2)))
    b_combined_avg,b_combined_sig = np.mean(b_good_rv_list[where_b]-b_bary_star_list[where_b]),np.sqrt(np.sum(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2)/np.size(where_b[0])**2)
    print(b_combined_avg,b_combined_sig)
    plt.hist(c_good_rv_list[where_c]-c_bary_star_list[where_c], label="c", range=[-20,20],bins=20,histtype="step")
    print(np.sqrt(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))
    c_combined_avg,c_combined_sig = np.mean(c_good_rv_list[where_c]-c_bary_star_list[where_c]),np.sqrt(np.sum(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2)/np.size(where_c[0])**2)
    print(c_combined_avg,c_combined_sig )
    print((b_combined_avg-c_combined_avg),np.sqrt((c_combined_sig**2+b_combined_sig**2)))
    plt.xlim([-15,15])
    plt.legend()

plt.show()
exit()