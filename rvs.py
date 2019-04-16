__author__ = 'jruffio'


import csv
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits

def get_err_from_posterior(x,posterior):
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    if len(x[0:argmax_post]) < 2:
        lx = np.nan
    else:
        lf = interp1d(cum_posterior[0:argmax_post],x[0:argmax_post],bounds_error=False,fill_value=np.nan)
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx = np.nan
    else:
        rf = interp1d(cum_posterior[argmax_post::],x[argmax_post::],bounds_error=False,fill_value=np.nan)
        rx = rf(1-0.6827)
    return x[argmax_post],(rx-lx)/2.,argmax_post

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
c_filelist = np.array([item[c_filename_id] for item in c_list_data])
c_out_filelist = np.array([item[c_cen_filename_id] for item in c_list_data])
c_kcen_list = np.array([int(item[c_kcen_id]) if item[c_kcen_id] != "nan" else 0 for item in c_list_data])
c_lcen_list = np.array([int(item[c_lcen_id]) if item[c_lcen_id] != "nan" else 0  for item in c_list_data])
c_date_list = np.array([item[c_filename_id].split(os.path.sep)[4] for item in c_list_data])
c_ifs_fitler_list = np.array([item[c_ifs_filter_id] for item in c_list_data])

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
b_filelist = np.array([item[b_filename_id] for item in b_list_data])
b_out_filelist = np.array([item[b_cen_filename_id] for item in b_list_data])
b_kcen_list = np.array([int(item[b_kcen_id])  if item[c_kcen_id] != "nan" else 0 for item in b_list_data])
b_lcen_list = np.array([int(item[b_lcen_id])  if item[c_lcen_id] != "nan" else 0 for item in b_list_data])
b_date_list = np.array([item[b_filename_id].split(os.path.sep)[4] for item in b_list_data])
b_ifs_fitler_list = np.array([item[b_ifs_filter_id] for item in b_list_data])

# plot RVs
if 1:
    rv_star = -12.6#-12.6+-1.4km/s HR 8799 Rob and Simbad
    # b_bary_star_list = np.array([-float(item[b_bary_rv_id])/1000+rv_star for item in b_list_data])
    b_bary_star_list = np.array([-float(item[b_bary_rv_id])/1000 for item in b_list_data])
    b_mjdobs_list = np.array([float(item[b_mjdobs_id]) for item in b_list_data])
    # b_mjdobs_list = np.arange(np.size(b_mjdobs_list))
    b_baryrv_list = np.array([-float(item[b_bary_rv_id])/1000 for item in b_list_data])
    b_rv_list = np.array([float(item[b_rvcen_id]) for item in b_list_data])
    b_status_list = np.array([float(item[b_status_id]) for item in b_list_data])
    b_rvsig_list = np.array([float(item[b_rvcensig_id]) for item in b_list_data])
    b_wvsolerr_list = np.array([float(item[b_wvsolerr_id]) for item in b_list_data])
    b_good_rv_list = copy(b_rv_list)
    b_where_bad = np.where(b_status_list!=1)
    b_good_rv_list[b_where_bad] = np.nan
    b_bad_rv_list= copy(b_rv_list)
    b_bad_rv_list[np.where(b_status_list>0)] = np.nan

    # c_bary_star_list = np.array([-float(item[c_bary_rv_id])/1000+rv_star for item in c_list_data])
    c_bary_star_list = np.array([-float(item[c_bary_rv_id])/1000 for item in c_list_data])
    c_mjdobs_list = np.array([float(item[c_mjdobs_id]) for item in c_list_data])
    # c_mjdobs_list = np.arange(np.size(c_mjdobs_list))
    c_baryrv_list = np.array([-float(item[c_bary_rv_id])/1000 for item in c_list_data])
    c_rv_list = np.array([float(item[c_rvcen_id]) for item in c_list_data])
    c_status_list = np.array([float(item[c_status_id]) for item in c_list_data])
    c_rvsig_list = np.array([float(item[c_rvcensig_id]) for item in c_list_data])
    c_wvsolerr_list = np.array([float(item[c_wvsolerr_id]) for item in c_list_data])
    c_good_rv_list = copy(c_rv_list)
    c_where_bad = np.where(c_status_list!=1)
    c_good_rv_list[c_where_bad] = np.nan
    c_bad_rv_list= copy(c_rv_list)
    c_bad_rv_list[np.where(c_status_list>0)] = np.nan


    # plt.figure(1,figsize=(9,0.75*9))
    # plt.subplot(2,1,1)
    #
    # plt.errorbar(c_mjdobs_list,c_good_rv_list,yerr=np.sqrt(c_rvsig_list**2+c_wvsolerr_list**2),fmt="x",color="red",label="c Measured raw RV ($1\sigma$)")
    # plt.errorbar(c_mjdobs_list,c_good_rv_list,yerr=c_rvsig_list,fmt="x",color="pink",label="c Measured raw RV ($1\sigma$)")
    # # plt.plot(c_mjdobs_list,c_bad_rv_list,linestyle="",marker="o",color="blue",label="Bad Data")
    # plt.plot(c_mjdobs_list,c_baryrv_list,color="#006699",label="c Barycentric RV")
    # plt.plot(c_mjdobs_list,c_bary_star_list,color="#ff9900",label="c Barycentric + HR8799 RV")
    #
    # plt.errorbar(b_mjdobs_list,b_good_rv_list,yerr=np.sqrt(b_rvsig_list**2+b_wvsolerr_list**2),fmt="x",color="green",label="b Measured raw RV ($1\sigma$)")
    # plt.errorbar(b_mjdobs_list,b_good_rv_list,yerr=b_rvsig_list,fmt="x",color="cyan",label="b Measured raw RV ($1\sigma$)")
    # # plt.plot(b_mjdobs_list,b_bad_rv_list,linestyle="",marker="o",color="blue",label="Bad Data")
    # plt.plot(b_mjdobs_list,b_baryrv_list,color="#006699",label="b Barycentric RV")
    # plt.plot(b_mjdobs_list,b_bary_star_list,color="#ff9900",label="b Barycentric + HR8799 RV")
    #
    # plt.xlabel("Exposure Index",fontsize=15)
    # plt.ylabel("RV (km/s)",fontsize=15)
    # plt.legend(fontsize=10)
    # plt.legend(fontsize=10,loc="upper left")
    # plt.subplot(2,1,2)
    # # plt.plot(rv_list-bary_star_list,"x",color="red",label="Estimated Planet RV")
    #
    # # plt.plot(c_mjdobs_list,np.zeros(c_rv_list.shape)+np.nanmean(c_rv_list-c_bary_star_list),linestyle=":",color="pink",label="Mean Planet RV")
    # plt.errorbar(c_mjdobs_list,c_good_rv_list-c_bary_star_list,yerr=np.sqrt(c_rvsig_list**2+c_wvsolerr_list**2),fmt="x",color="red",label="c Estimated Planet RV ($1\sigma$)")
    # plt.errorbar(c_mjdobs_list,c_good_rv_list-c_bary_star_list,yerr=c_rvsig_list,fmt="x",color="pink",label="c Estimated Planet RV ($1\sigma$)")
    # # plt.plot(c_mjdobs_list,c_bad_rv_list-c_bary_star_list,linestyle="",marker="o",color="blue",label="Bad data")
    #
    # # plt.plot(b_mjdobs_list,np.zeros(b_rv_list.shape)+np.nanmean(b_rv_list-b_bary_star_list),linestyle=":",color="pink",label="Mean Planet RV")
    # plt.errorbar(b_mjdobs_list,b_good_rv_list-b_bary_star_list,yerr=np.sqrt(b_rvsig_list**2+b_wvsolerr_list**2),fmt="x",color="green",label="b Estimated Planet RV ($1\sigma$)")
    # plt.errorbar(b_mjdobs_list,b_good_rv_list-b_bary_star_list,yerr=b_rvsig_list,fmt="x",color="cyan",label="b Estimated Planet RV ($1\sigma$)")
    # # plt.plot(b_mjdobs_list,b_bad_rv_list-b_bary_star_list,linestyle="",marker="o",color="blue",label="Bad data")
    #
    # plt.xlabel("Exposure Index",fontsize=15)
    # plt.ylabel("RV (km/s)",fontsize=15)
    # # print(np.nansum((c_good_rv_list-c_bary_star_list)/c_rvsig_list)/(np.sum(1./c_rvsig_list[np.where(np.isfinite(c_good_rv_list))])),
    # #     np.sqrt(np.size(c_rvsig_list[np.where(np.isfinite(c_good_rv_list))]))/(np.sum(1./c_rvsig_list[np.where(np.isfinite(c_good_rv_list))])))
    # # print(np.nanmean(c_good_rv_list[10:50]-c_bary_star_list[10:50]))
    # # plt.ylim([-20,20])
    # plt.legend(fontsize=10,loc="upper left")
    # # plt.show()
    # # print("Saving "+os.path.join(out_pngs,"HR8799"+planet+"_"+IFSfilter+"_RV.pdf"))
    # # plt.savefig(os.path.join(out_pngs,"HR8799"+planet+"_"+IFSfilter+"_RV.pdf"),bbox_inches='tight')
    # # plt.savefig(os.path.join(out_pngs,"HR8799"+planet+"_"+IFSfilter+"_RV.png"),bbox_inches='tight')

    fontsize=12
    where_b = np.where(np.isfinite(b_good_rv_list)*(b_mjdobs_list > 55388)*(b_mjdobs_list < 56000))
    # where_b = np.where(np.isfinite(b_good_rv_list)*(b_mjdobs_list > 55388))
    where_b_Kbb = np.where((b_ifs_fitler_list[where_b]=="Kbb"))
    where_b_Hbb = np.where((b_ifs_fitler_list[where_b]=="Hbb"))
    print(np.size(where_b[0]))
    where_c = np.where(np.isfinite(c_good_rv_list)*(c_mjdobs_list > 55388)*(c_mjdobs_list < 56000))
    # where_c = np.where(np.isfinite(c_good_rv_list)*(c_mjdobs_list > 55388))
    where_c_Kbb = np.where((c_ifs_fitler_list[where_c]=="Kbb"))
    where_c_Hbb = np.where((c_ifs_fitler_list[where_c]=="Hbb"))
    print(np.size(where_c[0]))

    # b_date_list
    # get posteriors
    if 1:
        plt.show()
        c_logposterior = []
        final_planetRV_hd = np.linspace(-30,30,6000)
        for filename,kcen,lcen,bary_star,wvsolerr in zip(c_out_filelist[where_c],c_kcen_list[where_c],c_lcen_list[where_c],c_bary_star_list[where_c],c_wvsolerr_list[where_c]):
            hdulist = pyfits.open(filename.replace(".fits","_planetRV.fits"))
            planetRV = hdulist[0].data
            hdulist = pyfits.open(filename)
            NplanetRV_hd = np.where((planetRV[1::]-planetRV[0:(np.size(planetRV)-1)]) < 0)[0][0]+1
            planetRV_hd = planetRV[0:NplanetRV_hd]
            logposterior = hdulist[0].data[0,0,9,0:NplanetRV_hd,kcen,lcen]
            logpost_func = interp1d(planetRV_hd-bary_star,logposterior,bounds_error=False,fill_value=np.min(logposterior))
            logposterior = logpost_func(final_planetRV_hd)
            posterior = np.exp(logposterior-np.nanmax(logposterior))
            conv_psoterior = np.convolve(posterior,1/(np.sqrt(2*np.pi)*wvsolerr)*np.exp(-0.5*final_planetRV_hd**2/wvsolerr**2),mode="same")
            c_logposterior.append(np.log(conv_psoterior/np.nanmax(conv_psoterior)))
        b_logposterior = []
        for filename,kcen,lcen,bary_star,wvsolerr in zip(b_out_filelist[where_b],b_kcen_list[where_b],b_lcen_list[where_b],b_bary_star_list[where_b],b_wvsolerr_list[where_b]):
            hdulist = pyfits.open(filename.replace(".fits","_planetRV.fits"))
            planetRV = hdulist[0].data
            hdulist = pyfits.open(filename)
            NplanetRV_hd = np.where((planetRV[1::]-planetRV[0:(np.size(planetRV)-1)]) < 0)[0][0]+1
            planetRV_hd = planetRV[0:NplanetRV_hd]
            logposterior = hdulist[0].data[0,0,9,0:NplanetRV_hd,kcen,lcen]
            logpost_func = interp1d(planetRV_hd-bary_star,logposterior,bounds_error=False,fill_value=np.min(logposterior))
            logposterior = logpost_func(final_planetRV_hd)
            posterior = np.exp(logposterior-np.nanmax(logposterior))
            conv_psoterior = np.convolve(posterior,1/(np.sqrt(2*np.pi)*wvsolerr)*np.exp(-0.5*final_planetRV_hd**2/wvsolerr**2),mode="same")
            b_logposterior.append(np.log(conv_psoterior/np.nanmax(conv_psoterior)))

        c_sumlogposterior = np.nansum(c_logposterior,axis=0)
        b_sumlogposterior = np.nansum(b_logposterior,axis=0)
        c_posterior = np.exp(c_sumlogposterior-np.max(c_sumlogposterior))
        b_posterior = np.exp(b_sumlogposterior-np.max(b_sumlogposterior))

        b_combined_avg,b_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,b_posterior)
        c_combined_avg,c_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,c_posterior)

    plt.figure(1,figsize=(12,8))
    plt.subplot(2,1,1)
    # c_combined_avg = np.sum((c_good_rv_list[where_c]-c_bary_star_list[where_c])/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))/np.sum(1/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))
    # c_combined_sig = 1/np.sqrt(np.sum(1/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2)))
    # b_combined_avg = np.sum((b_good_rv_list[where_b]-b_bary_star_list[where_b])/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2))/np.sum(1/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2))
    # b_combined_sig = 1/np.sqrt(np.sum(1/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2)))
    b_combined_avg,b_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,b_posterior)
    c_combined_avg,c_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,c_posterior)
    print(c_combined_avg,c_combined_sig )
    print(b_combined_avg,b_combined_sig)
    print(b_combined_avg-c_combined_avg,np.sqrt(c_combined_sig**2+b_combined_sig**2))
    # plt.fill_betweenx([0,10],c_combined_avg-c_combined_sig,c_combined_avg+c_combined_sig,alpha=1,color="#cc3300")
    # plt.fill_betweenx([0,10],b_combined_avg-b_combined_sig,b_combined_avg+b_combined_sig,alpha=1,color="#003366")
    # plt.hist(c_good_rv_list[where_c]-c_bary_star_list[where_c], range=[-20,20],bins=20,histtype="bar",alpha=0.5,color="#ff9900",label="c: RV histogram")
    plt.plot([c_combined_avg-c_combined_sig,c_combined_avg-c_combined_sig],[0,10],linestyle="--",linewidth=2,color="#cc6600",label="c: RV $1\sigma$ error")
    plt.plot([c_combined_avg,c_combined_avg],[0,10],linestyle="-",linewidth=2,color="#cc3300",label="c: Expected")
    plt.gca().text(c_combined_avg,0.1,"${0:.2f}\pm {1:.2f}$ km/s".format(c_combined_avg,c_combined_sig),ha="right",va="bottom",rotation=90,size=fontsize,color="#cc3300")
    plt.plot([c_combined_avg+c_combined_sig,c_combined_avg+c_combined_sig],[0,10],linestyle="--",linewidth=2,color="#cc6600")
    plt.plot(final_planetRV_hd,c_posterior,linestyle="-",linewidth=3,color="#ff9900",label="c: posterior")
    # plt.plot(final_planetRV_hd,np.exp(-0.5*(final_planetRV_hd-c_combined_avg)**2/c_combined_sig**2),linestyle="--",linewidth=3,color="black",label="c: posterior")

    # plt.hist(b_good_rv_list[where_b]-b_bary_star_list[where_b], range=[-20,20],bins=20,histtype="bar",alpha=0.5,color="#0099cc",label="b: RV histogram")
    plt.plot([b_combined_avg-b_combined_sig,b_combined_avg-b_combined_sig],[0,10],linestyle=":",linewidth=2,color="#006699",label="b: RV $1\sigma$ error")
    plt.plot([b_combined_avg,b_combined_avg],[0,10],linestyle="-.",linewidth=2,color="#003366",label="b: Expected")
    plt.gca().text(b_combined_avg,0.1,"${0:.2f}\pm {1:.2f}$ km/s".format(b_combined_avg,b_combined_sig),ha="right",va="bottom",rotation=90,size=fontsize,color="#003366")
    plt.plot([b_combined_avg+b_combined_sig,b_combined_avg+b_combined_sig],[0,10],linestyle=":",linewidth=2,color="#006699")
    plt.plot(final_planetRV_hd,b_posterior,linestyle="-",linewidth=3,color="#0099cc",label="b: posterior")
    # plt.plot(final_planetRV_hd,np.exp(-0.5*(final_planetRV_hd-b_combined_avg)**2/b_combined_sig**2),linestyle="--",linewidth=3,color="black",label="b: posterior")

    plt.xlim([-1,7])
    plt.ylim([0,1.1])
    plt.xlabel("RV (km/s)",fontsize=fontsize)
    plt.ylabel("$\propto \mathcal{P}(RV|d)$",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",0))
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#

    plt.subplot(2,1,2)
    delta_posterior = np.correlate(b_posterior,c_posterior,mode="same")
    deltaRV,deltaRV_sig,_ = get_err_from_posterior(final_planetRV_hd,delta_posterior)
    confidence_interval = (1-np.cumsum(delta_posterior)[np.argmin(np.abs(final_planetRV_hd))]/np.sum(delta_posterior))
    # plt.plot(final_planetRV_hd,np.cumsum(delta_posterior)/np.max(np.cumsum(delta_posterior)),linestyle="-",linewidth=3,color="#6600ff",label="b: posterior") #9966ff
    plt.plot([deltaRV-deltaRV_sig,deltaRV-deltaRV_sig],[0,10],linestyle=":",linewidth=2,color="#9966ff",label="$1\sigma$ error")
    plt.plot([deltaRV,deltaRV],[0,10],linestyle="-.",linewidth=2,color="#660066",label="Expected")
    plt.gca().text(deltaRV,0.1,"${0:.2f}\pm {1:.2f}$ km/s".format(deltaRV,deltaRV_sig),ha="right",va="bottom",rotation=90,size=fontsize,color="#660066")
    plt.plot([deltaRV+deltaRV_sig,deltaRV+deltaRV_sig],[0,10],linestyle=":",linewidth=2,color="#9966ff")
    plt.plot(final_planetRV_hd,delta_posterior/np.max(delta_posterior),linestyle="-",linewidth=3,color="#6600ff",label="posterior") #9966ff
    # plt.fill_between(final_planetRV_hd[np.where(final_planetRV_hd>0)],
    #                  np.zeros(np.size(final_planetRV_hd[np.where(final_planetRV_hd>0)])),
    #                  (delta_posterior/np.max(delta_posterior))[np.where(final_planetRV_hd>0)],alpha=0.2,color="#9966ff")
    plt.xlim([-1,7])
    plt.ylim([0,1.1])
    plt.xlabel(r"$RV_b-RV_c$ (km/s)",fontsize=fontsize)
    plt.ylabel("$\propto \mathcal{P}(RV_b-RV_c|d)$",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",0))
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#


    if 0:
        print("Saving "+os.path.join(out_pngs,"RV_HR_8799_bc_posterior.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bc_posterior.pdf"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bc_posterior.png"),bbox_inches='tight')

    plt.figure(2,figsize=(12,8))
    plt.subplot(2,1,1)
    plt.plot([0,np.size(where_c[0])],[c_combined_avg,c_combined_avg],color="#cc3300",label="Expected")
    plt.errorbar(np.arange(np.size(where_c[0]))[where_c_Kbb],c_good_rv_list[where_c][where_c_Kbb]-c_bary_star_list[where_c][where_c_Kbb],
                 yerr=np.sqrt(c_rvsig_list[where_c][where_c_Kbb]**2+c_wvsolerr_list[where_c][where_c_Kbb]**2),fmt="none",color="#cc6600",label="$1\sigma$ error")
    plt.errorbar(np.arange(np.size(where_c[0]))[where_c_Kbb],c_good_rv_list[where_c][where_c_Kbb]-c_bary_star_list[where_c][where_c_Kbb],
                 yerr=c_rvsig_list[where_c][where_c_Kbb],fmt="none",color="#ff9900")
    plt.plot(np.arange(np.size(where_c[0]))[where_c_Kbb],c_good_rv_list[where_c][where_c_Kbb]-c_bary_star_list[where_c][where_c_Kbb],"x",color="#ff9900",label="Kbb")
    plt.errorbar(np.arange(np.size(where_c[0]))[where_c_Hbb],c_good_rv_list[where_c][where_c_Hbb]-c_bary_star_list[where_c][where_c_Hbb],
                 yerr=np.sqrt(c_rvsig_list[where_c][where_c_Hbb]**2+c_wvsolerr_list[where_c][where_c_Hbb]**2),fmt="none",color="#cc6600")
    plt.errorbar(np.arange(np.size(where_c[0]))[where_c_Hbb],c_good_rv_list[where_c][where_c_Hbb]-c_bary_star_list[where_c][where_c_Hbb],
                 yerr=c_rvsig_list[where_c][where_c_Hbb],fmt="none",color="#ff9900")
    plt.plot(np.arange(np.size(where_c[0]))[where_c_Hbb],c_good_rv_list[where_c][where_c_Hbb]-c_bary_star_list[where_c][where_c_Hbb],"o",color="#ff9900",label="Hbb")
    plt.fill_between([0,np.size(where_c[0])],c_combined_avg-c_combined_sig,c_combined_avg+c_combined_sig,alpha=0.2,color="#cc3300")
    plt.ylim([-30+rv_star,30+rv_star])
    for date in np.unique(c_date_list[where_c]):
        where_data = np.where(c_date_list[where_c]==date)
        first = where_data[0][0]
        last = where_data[0][-1]
        plt.annotate("",xy=(first,-15+rv_star),xytext=(last+0.001,-15+rv_star),xycoords="data",arrowprops={'arrowstyle':"|-|",'shrinkA':0.1,'shrinkB':0.1})
        plt.gca().text((first+last)/2,-20+rv_star,date,ha="left",va="top",rotation=-45,size=fontsize)
    plt.gca().text(np.size(where_c[0]),30+rv_star,"HR 8799 c",ha="right",va="top",rotation=0,size=fontsize*1.5)
    plt.ylabel("RV (km/s)",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_position(("data",0))
    plt.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="upper left",bbox_to_anchor=[0,1.1],frameon=True,fontsize=fontsize)#

    plt.subplot(2,1,2)
    plt.plot([0,np.size(where_b[0])],[b_combined_avg,b_combined_avg],color="#003366",label="Expected")
    plt.errorbar(np.arange(np.size(where_b[0]))[where_b_Kbb],b_good_rv_list[where_b][where_b_Kbb]-b_bary_star_list[where_b][where_b_Kbb],
                 yerr=np.sqrt(b_rvsig_list[where_b][where_b_Kbb]**2+b_wvsolerr_list[where_b][where_b_Kbb]**2),fmt="none",color="#006699",label="$1\sigma$ error")
    plt.errorbar(np.arange(np.size(where_b[0]))[where_b_Kbb],b_good_rv_list[where_b][where_b_Kbb]-b_bary_star_list[where_b][where_b_Kbb],
                 yerr=b_rvsig_list[where_b][where_b_Kbb],fmt="none",color="#0099cc")
    plt.plot(np.arange(np.size(where_b[0]))[where_b_Kbb],b_good_rv_list[where_b][where_b_Kbb]-b_bary_star_list[where_b][where_b_Kbb],"x",color="#0099cc",label="Kbb")
    plt.errorbar(np.arange(np.size(where_b[0]))[where_b_Hbb],b_good_rv_list[where_b][where_b_Hbb]-b_bary_star_list[where_b][where_b_Hbb],
                 yerr=np.sqrt(b_rvsig_list[where_b][where_b_Hbb]**2+b_wvsolerr_list[where_b][where_b_Hbb]**2),fmt="none",color="#006699")
    plt.errorbar(np.arange(np.size(where_b[0]))[where_b_Hbb],b_good_rv_list[where_b][where_b_Hbb]-b_bary_star_list[where_b][where_b_Hbb],
                 yerr=b_rvsig_list[where_b][where_b_Hbb],fmt="none",color="#0099cc")
    plt.plot(np.arange(np.size(where_b[0]))[where_b_Hbb],b_good_rv_list[where_b][where_b_Hbb]-b_bary_star_list[where_b][where_b_Hbb],"o",color="#0099cc",label="Hbb")
    plt.fill_between([0,np.size(where_b[0])],b_combined_avg-b_combined_sig,b_combined_avg+b_combined_sig,alpha=0.2,color="#003366")
    plt.ylim([-30+rv_star,30+rv_star])
    for date in np.unique(b_date_list[where_b]):
        where_data = np.where(b_date_list[where_b]==date)
        first = where_data[0][0]
        last = where_data[0][-1]
        plt.annotate("",xy=(first,-15+rv_star),xytext=(last+0.001,-15+rv_star),xycoords="data",arrowprops={'arrowstyle':"|-|",'shrinkA':0.1,'shrinkB':0.1})
        plt.gca().text((first+last)/2,-20+rv_star,date,ha="left",va="top",rotation=-45,size=fontsize)
    plt.gca().text(np.size(where_b[0]),30+rv_star,"HR 8799 b",ha="right",va="top",rotation=0,size=fontsize*1.5)
    plt.ylabel("RV (km/s)",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_position(("data",0))
    plt.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="upper left",bbox_to_anchor=[0,1.1],frameon=True,fontsize=fontsize)#
    # plt.errorbar(b_good_rv_list[where_b][where_b_Kbb]-b_bary_star_list[where_b][where_b_Kbb],np.arange(np.size(where_b[0]))[where_b_Kbb],xerr=np.sqrt(b_rvsig_list[where_b][where_b_Kbb]**2+b_wvsolerr_list[where_b][where_b_Kbb]**2),fmt="x",color="green",label="b Measured raw RV ($1\sigma$)")
    # plt.errorbar(b_good_rv_list[where_b][where_b_Kbb]-b_bary_star_list[where_b][where_b_Kbb],np.arange(np.size(where_b[0]))[where_b_Kbb],xerr=b_rvsig_list[where_b][where_b_Kbb],fmt="x",color="cyan",label="b Measured raw RV ($1\sigma$)")
    # plt.errorbar(b_good_rv_list[where_b][where_b_Hbb]-b_bary_star_list[where_b][where_b_Hbb],np.arange(np.size(where_b[0]))[where_b_Hbb],xerr=np.sqrt(b_rvsig_list[where_b][where_b_Hbb]**2+b_wvsolerr_list[where_b][where_b_Hbb]**2),fmt="x",color="green",label="b Measured raw RV ($1\sigma$)")
    # plt.errorbar(b_good_rv_list[where_b][where_b_Hbb]-b_bary_star_list[where_b][where_b_Hbb],np.arange(np.size(where_b[0]))[where_b_Hbb],xerr=b_rvsig_list[where_b][where_b_Hbb],fmt="x",color="cyan",label="b Measured raw RV ($1\sigma$)")
    # plt.xlim([-20,20])

    if 0:
        print("Saving "+os.path.join(out_pngs,"RV_HR_8799_bc_measurements.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bc_measurements.pdf"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bc_measurements.png"),bbox_inches='tight')

    plt.figure(3,figsize=(6,5))
    plt.hist(b_good_rv_list[where_b]-b_bary_star_list[where_b], label="b", range=[-20,20],bins=20,histtype="bar",alpha=0.5)
    plt.hist(c_good_rv_list[where_c]-c_bary_star_list[where_c], label="c", range=[-20,20],bins=20,histtype="bar",alpha=0.5)
    plt.xlabel("RV (km/s)",fontsize=fontsize)
    plt.ylabel("# measurements",fontsize=fontsize)
    plt.legend()



    # plt.figure(4)
    # where_b = np.where(np.isfinite(b_good_rv_list)*(b_mjdobs_list > 55388))
    # where_b_Kbb = np.where((b_ifs_fitler_list[where_b]=="Kbb"))
    # where_b_Hbb = np.where((b_ifs_fitler_list[where_b]=="Hbb"))
    # print(np.size(where_b[0]))
    # where_c = np.where(np.isfinite(c_good_rv_list)*(c_mjdobs_list > 55388))
    # where_c_Kbb = np.where((c_ifs_fitler_list[where_c]=="Kbb"))
    # where_c_Hbb = np.where((c_ifs_fitler_list[where_c]=="Hbb"))
    # print(np.size(where_c[0]))
    #
    # # b_date_list
    #
    # plt.subplot(3,1,1)
    # plt.errorbar(c_good_rv_list[where_c][where_c_Kbb]-c_bary_star_list[where_c][where_c_Kbb],np.arange(np.size(where_c[0]))[where_c_Kbb],xerr=np.sqrt(c_rvsig_list[where_c][where_c_Kbb]**2+c_wvsolerr_list[where_c][where_c_Kbb]**2),fmt="x",color="red",label="c Measured raw RV ($1\sigma$)")
    # plt.errorbar(c_good_rv_list[where_c][where_c_Kbb]-c_bary_star_list[where_c][where_c_Kbb],np.arange(np.size(where_c[0]))[where_c_Kbb],xerr=c_rvsig_list[where_c][where_c_Kbb],fmt="x",color="pink",label="c Measured raw RV ($1\sigma$)")
    # plt.errorbar(c_good_rv_list[where_c][where_c_Hbb]-c_bary_star_list[where_c][where_c_Hbb],np.arange(np.size(where_c[0]))[where_c_Hbb],xerr=np.sqrt(c_rvsig_list[where_c][where_c_Hbb]**2+c_wvsolerr_list[where_c][where_c_Hbb]**2),fmt="o",color="red",label="c Measured raw RV ($1\sigma$)")
    # plt.errorbar(c_good_rv_list[where_c][where_c_Hbb]-c_bary_star_list[where_c][where_c_Hbb],np.arange(np.size(where_c[0]))[where_c_Hbb],xerr=c_rvsig_list[where_c][where_c_Hbb],fmt="o",color="pink",label="c Measured raw RV ($1\sigma$)")
    # plt.xlim([-15,15])
    #
    # plt.subplot(3,1,2)
    # plt.errorbar(b_good_rv_list[where_b][where_b_Kbb]-b_bary_star_list[where_b][where_b_Kbb],np.arange(np.size(where_b[0]))[where_b_Kbb],xerr=np.sqrt(b_rvsig_list[where_b][where_b_Kbb]**2+b_wvsolerr_list[where_b][where_b_Kbb]**2),fmt="x",color="green",label="b Measured raw RV ($1\sigma$)")
    # plt.errorbar(b_good_rv_list[where_b][where_b_Kbb]-b_bary_star_list[where_b][where_b_Kbb],np.arange(np.size(where_b[0]))[where_b_Kbb],xerr=b_rvsig_list[where_b][where_b_Kbb],fmt="x",color="cyan",label="b Measured raw RV ($1\sigma$)")
    # plt.errorbar(b_good_rv_list[where_b][where_b_Hbb]-b_bary_star_list[where_b][where_b_Hbb],np.arange(np.size(where_b[0]))[where_b_Hbb],xerr=np.sqrt(b_rvsig_list[where_b][where_b_Hbb]**2+b_wvsolerr_list[where_b][where_b_Hbb]**2),fmt="x",color="green",label="b Measured raw RV ($1\sigma$)")
    # plt.errorbar(b_good_rv_list[where_b][where_b_Hbb]-b_bary_star_list[where_b][where_b_Hbb],np.arange(np.size(where_b[0]))[where_b_Hbb],xerr=b_rvsig_list[where_b][where_b_Hbb],fmt="x",color="cyan",label="b Measured raw RV ($1\sigma$)")
    # plt.xlim([-15,15])
    #
    # plt.subplot(3,1,3)
    # plt.hist(b_good_rv_list[where_b]-b_bary_star_list[where_b], label="b", range=[-20,20],bins=20,histtype="step")
    # plt.hist(c_good_rv_list[where_c]-c_bary_star_list[where_c], label="c", range=[-20,20],bins=20,histtype="step")
    # c_combined_avg = np.sum((c_good_rv_list[where_c]-c_bary_star_list[where_c])/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))/np.sum(1/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))
    # c_combined_sig = 1/np.sqrt(np.sum(1/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2)))
    # print(c_combined_avg,c_combined_sig )
    # b_combined_avg = np.sum((b_good_rv_list[where_b]-b_bary_star_list[where_b])/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2))/np.sum(1/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2))
    # b_combined_sig = 1/np.sqrt(np.sum(1/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2)))
    # print(b_combined_avg,b_combined_sig)
    # print(b_combined_avg-c_combined_avg,np.sqrt(c_combined_sig**2+b_combined_sig**2))
    # plt.xlim([-15,15])
    # plt.legend()

    plt.show()
    exit()