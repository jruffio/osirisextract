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

# numbasis = 3
# numbasis_list = [0,1,5,10,15]
numbasis_list = [10]
for fakecorr in [False]:#[False,True]:
    combined_RV_list= []
    combined_RVerr_list = []
    for nbid,numbasis in enumerate(numbasis_list):
        myoutfilename = "RV_HR_8799_d_measurements_kl{0}.pdf".format(numbasis)
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
            c_time_id = c_colnames.index("itime")
        c_filelist = np.array([item[c_filename_id] for item in c_list_data])
        c_out_filelist = np.array([item[c_cen_filename_id] for item in c_list_data])
        c_kcen_list = np.array([int(item[c_kcen_id]) if item[c_kcen_id] != "nan" else 0 for item in c_list_data])
        c_lcen_list = np.array([int(item[c_lcen_id]) if item[c_lcen_id] != "nan" else 0  for item in c_list_data])
        c_date_list = np.array([item[c_filename_id].split(os.path.sep)[4] for item in c_list_data])
        c_ifs_fitler_list = np.array([item[c_ifs_filter_id] for item in c_list_data])
        c_snr_list = np.array([float(item[c_snr_id]) if item[c_snr_id] != "nan" else 0  for item in c_list_data])
        c_DTMP6_list = np.array([float(item[c_DTMP6_id]) if item[c_DTMP6_id] != "nan" else 0  for item in c_list_data])
        c_itime_list = np.array([float(item[c_time_id]) if item[c_time_id] != "nan" else 0  for item in c_list_data])


        # plot RVs
        if 1:
            fontsize=12
            rv_star = -12.7#

            # c_bary_star_list = np.array([-float(item[c_bary_rv_id])/1000+rv_star for item in c_list_data])
            c_bary_star_list = np.array([-float(item[c_bary_rv_id])/1000 for item in c_list_data])
            c_mjdobs_list = np.array([float(item[c_mjdobs_id]) for item in c_list_data])
            # c_mjdobs_list = np.arange(np.size(c_mjdobs_list))
            c_baryrv_list = np.array([-float(item[c_bary_rv_id])/1000 for item in c_list_data])
            c_rv_list = np.array([float(item[c_rvcen_id]) for item in c_list_data])
            c_status_list = np.array([float(item[c_status_id]) for item in c_list_data])
            c_rvsig_list = np.array([float(item[c_rvcensig_id]) for item in c_list_data])
            c_wvsolerr_list = np.array([float(item[c_wvsolerr_id]) for item in c_list_data])
            # c_status_list[np.where(np.isnan(c_rvsig_list))] = 0


            # try:
            if fakecorr:
                if nbid==0:
                    myoutfilename = myoutfilename.replace(".pdf","_fakes.pdf")
                c_rvfakes_list = np.array([float(item[c_rvfakes_id]) for item in c_list_data])
                c_rvfakessig_list = np.array([float(item[c_rvfakessig_id]) for item in c_list_data])

                # c_rvoffsets_list = c_rv_list-c_rvfakes_list
                c_rvoffsets_list = -14-c_rvfakes_list
                c_rv_list = c_rv_list+c_rvoffsets_list

                # plt.figure(2)
                # plt.scatter(c_rvsig_list,c_rvfakessig_list)
                # plt.plot([0,2],[0,2])
                # plt.xlabel("$\sigma$ analytical formula")
                # plt.ylabel("$\sigma$ from fakes injection")
                # print("Saving "+os.path.join(out_pngs,"kap_And","RV_kap_And_error_calc_check.pdf"))
                # plt.savefig(os.path.join(out_pngs,"kap_And","RV_kap_And_error_calc_check.pdf"),bbox_inches='tight')
                # plt.savefig(os.path.join(out_pngs,"kap_And","RV_kap_And_error_calc_check.png"),bbox_inches='tight')
                # # plt.show()
            # except:
            #     pass
            else:
                c_rvoffsets_list = np.zeros(c_rv_list.shape)

            if 0:
                if nbid==0:
                    myoutfilename = myoutfilename.replace(".pdf","_binned.pdf")
                c_rv_list_new = []
                c_rvsig_list_new = []
                c_bary_star_list_new = []
                c_wvsolerr_list_new = []
                c_date_list_new = []
                c_status_list_new = []
                for date in np.unique(c_date_list):
                    where_data = np.where(c_date_list==date)
                    first = where_data[0][0]
                    last = where_data[0][-1]
                    print(first,last)
                    d = 2
                    for k in np.arange(first,last+1,d):
                        c_rv_list_new.append(np.mean(c_rv_list[k:k+d]))
                        c_bary_star_list_new.append(np.mean(c_bary_star_list[k:k+d]))
                        c_rvsig_list_new.append(np.max(c_rvsig_list[k:k+d]))
                        c_wvsolerr_list_new.append(c_wvsolerr_list[k])
                        c_date_list_new.append(c_date_list[k])
                        c_status_list_new.append(np.min(c_status_list[k:k+d]))
                    # exit()
                c_rv_list = np.array(c_rv_list_new)
                c_rvsig_list = np.array(c_rvsig_list_new)
                c_bary_star_list = np.array(c_bary_star_list_new)
                c_wvsolerr_list = np.array(c_wvsolerr_list_new)
                c_date_list = np.array(c_date_list_new)
                c_status_list = np.array(c_status_list_new)


            c_good_rv_list = copy(c_rv_list)
            c_where_bad = np.where(c_status_list!=1)
            c_good_rv_list[c_where_bad] = np.nan
            c_bad_rv_list= copy(c_rv_list)
            c_bad_rv_list[np.where(c_status_list>0)] = np.nan


            where_all_c = np.where((c_status_list>0)*(c_mjdobs_list > 55388))
            where_esti_c = np.where((c_status_list>0)*np.isfinite(c_good_rv_list)*np.isfinite(c_rvsig_list)*(c_mjdobs_list > 55388))
            # where_all_c = np.where(np.isfinite(c_good_rv_list))
            # where_esti_c = np.where(np.isfinite(c_good_rv_list))
            # print(np.size(where_all_c[0]))
            # print((c_status_list>0))
            # continue

            # print(c_good_rv_list[where_esti_c]-c_bary_star_list[where_esti_c])
            # print(c_rvsig_list[where_esti_c])
            # print(c_wvsolerr_list[where_esti_c])
            # print(c_filelist[where_esti_c])
            # exit()

            if 1:
                c_combined_avg = np.sum((c_good_rv_list[where_esti_c]-c_bary_star_list[where_esti_c])/(c_rvsig_list[where_esti_c]**2+c_wvsolerr_list[where_esti_c]**2))/np.sum(1/(c_rvsig_list[where_esti_c]**2+c_wvsolerr_list[where_esti_c]**2))
                c_combined_sig = 1/np.sqrt(np.sum(1/(c_rvsig_list[where_esti_c]**2+c_wvsolerr_list[where_esti_c]**2)))
                # filter outliers
                dist2avg = (c_good_rv_list-c_bary_star_list-c_combined_avg)/np.sqrt(c_rvsig_list**2+c_wvsolerr_list**2)
                where_esti_c = np.where((c_status_list>0)*np.isfinite(c_good_rv_list)*np.isfinite(c_rvsig_list)*(c_mjdobs_list > 55388)*(np.abs(dist2avg)<3))

            # get posteriors
            if 1:
                final_planetRV_hd = np.linspace(-30,30,6000)
                c_logposterior = []
                for filename,kcen,lcen,bary_star,wvsolerr,c_rvoffset in zip(c_out_filelist[where_esti_c],c_kcen_list[where_esti_c],c_lcen_list[where_esti_c],c_bary_star_list[where_esti_c],c_wvsolerr_list[where_esti_c],c_rvoffsets_list[where_esti_c]):
                    hdulist = pyfits.open(filename.replace(".fits","_planetRV.fits"))
                    planetRV = hdulist[0].data
                    hdulist = pyfits.open(filename)
                    NplanetRV_hd = np.where((planetRV[1::]-planetRV[0:(np.size(planetRV)-1)]) < 0)[0][0]+1
                    planetRV_hd = planetRV[0:NplanetRV_hd]
                    logposterior = hdulist[0].data[0,0,9,0:NplanetRV_hd,kcen,lcen]
                    logpost_func = interp1d(planetRV_hd-bary_star+c_rvoffset,logposterior,bounds_error=False,fill_value=np.min(logposterior))
                    logposterior = logpost_func(final_planetRV_hd)
                    posterior = np.exp(logposterior-np.nanmax(logposterior))
                    conv_psoterior = np.convolve(posterior,1/(np.sqrt(2*np.pi)*wvsolerr)*np.exp(-0.5*final_planetRV_hd**2/wvsolerr**2),mode="same")
                    c_logposterior.append(np.log(conv_psoterior/np.nanmax(conv_psoterior)))

                c_sumlogposterior = np.nansum(c_logposterior,axis=0)
                c_optimistic_posterior = np.exp(c_sumlogposterior-np.max(c_sumlogposterior))
                c_combined_avg_post,c_combined_sig_post,_ = get_err_from_posterior(final_planetRV_hd,c_optimistic_posterior)
                c_chi2 = np.sum(((c_good_rv_list[where_esti_c]-c_bary_star_list[where_esti_c]-c_combined_avg_post)/np.sqrt(c_rvsig_list[where_esti_c]**2+c_wvsolerr_list[where_esti_c]**2))**2)
                c_N_data = np.size(c_good_rv_list[where_esti_c])

                print("coucou",c_chi2,c_chi2/c_N_data)
                if c_chi2 <= c_N_data:
                    c_chi2 = c_N_data

                c_optimistic_posterior_func = interp1d(final_planetRV_hd,c_optimistic_posterior,bounds_error=False,fill_value=0)
                c_posterior = c_optimistic_posterior_func((final_planetRV_hd-final_planetRV_hd[np.argmax(c_optimistic_posterior)])/np.sqrt(c_chi2/c_N_data)+final_planetRV_hd[np.argmax(c_optimistic_posterior)])
                c_posterior = c_posterior/np.max(c_posterior)
                c_combined_avg,c_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,c_posterior)

                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=np.concatenate([final_planetRV_hd[None,:],c_posterior[None,:]],axis=0)))
                try:
                    hdulist.writeto(os.path.join(out_pngs,"HR_8799_d",myoutfilename.replace(".pdf","_posterior.fits")), overwrite=True)
                except TypeError:
                    hdulist.writeto(os.path.join(out_pngs,"HR_8799_d",myoutfilename.replace(".pdf","_posterior.fits")), clobber=True)
                hdulist.close()
                print(c_combined_avg_post,c_combined_sig_post)
                # plt.plot(final_planetRV_hd,c_posterior)
                # plt.show()
                #
                # exit()
            else:
                c_combined_avg = np.sum((c_good_rv_list[where_esti_c]-c_bary_star_list[where_esti_c])/(c_rvsig_list[where_esti_c]**2+c_wvsolerr_list[where_esti_c]**2))/np.sum(1/(c_rvsig_list[where_esti_c]**2+c_wvsolerr_list[where_esti_c]**2))
                c_combined_sig = 1/np.sqrt(np.sum(1/(c_rvsig_list[where_esti_c]**2+c_wvsolerr_list[where_esti_c]**2)))


            c_rel_res = (c_good_rv_list-c_bary_star_list-c_combined_avg)/np.sqrt(c_rvsig_list**2+c_wvsolerr_list**2)



            if 1: #print estimates per night
                c_combined_avg_list = []
                c_combined_sig_list = []
                c_chi2_list = []
                c_N_data_list = []
                c_uniquedate_list = []
                c_mjd_list = []
                c_yr_list_new = [day for day in c_date_list]
                # c_yr_list_new = [day[0:4] for day in c_date_list]
                unique_date = np.unique(c_yr_list_new)
                for k,day in enumerate(unique_date):
                    # print(day,unique_date)
                    # print(c_date_list_new)
                    # print((day == np.array(c_date_list_new)))
                    # print(c_good_rv_list)
                    # print(np.isfinite(c_good_rv_list))
                    # exit()
                    where_c = np.where(np.isfinite(c_good_rv_list)*np.isfinite(c_rvsig_list)*(day == np.array(c_yr_list_new))*(np.abs(c_rel_res)<3))
                    if np.size(where_c[0]) == 0:
                        c_combined_avg_list.append(np.nan)
                        c_combined_sig_list.append(np.nan)
                        c_chi2_list.append(np.nan)
                        c_N_data_list.append(np.nan)
                        c_uniquedate_list.append("")
                        c_mjd_list.append(np.nan)
                        print("skip")
                        continue
                    c_combined_avg = np.sum((c_good_rv_list[where_c]-c_bary_star_list[where_c])/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))/np.sum(1/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))
                    c_combined_sig = 1/np.sqrt(np.sum(1/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2)))

                    c_chi2 = np.sum(((c_good_rv_list[where_c]-c_bary_star_list[where_c]-c_combined_avg)/np.sqrt(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))**2)
                    # c_N_data = np.size(c_good_rv_list[where_c])
                    c_N_data = np.sum(c_itime_list[where_c])

                    c_combined_avg_list.append(c_combined_avg)
                    c_combined_sig_list.append(c_combined_sig)
                    c_chi2_list.append(c_chi2)
                    c_N_data_list.append(c_N_data)
                    c_uniquedate_list.append(c_date_list[where_c][0])
                    c_mjd_list.append(np.mean(c_mjdobs_list[where_c]))
                c_combined_avg_list = np.array(c_combined_avg_list)
                c_combined_sig_list = np.array(c_combined_sig_list)
                c_pessimistic_weighted_mean = np.nansum(c_combined_avg_list/c_combined_sig_list**2)/np.nansum(1/c_combined_sig_list**2)
                c_pessimisticweighted_mean_sig = 1/np.sqrt(np.nansum(1/c_combined_sig_list**2))
                print(numbasis)
                print(np.array(unique_date)[np.where(np.isfinite(c_combined_avg_list))])
                print(np.array(c_combined_avg_list)[np.where(np.isfinite(c_combined_avg_list))])
                print(np.array(c_combined_sig_list)[np.where(np.isfinite(c_combined_avg_list))])
                print(np.array(c_N_data_list)[np.where(np.isfinite(c_combined_avg_list))])
                print(np.array(c_mjd_list)[np.where(np.isfinite(c_combined_avg_list))])
    exit()

    if 0:
        if 0:

            # print(np.size(c_good_rv_list))
            # print(c_rel_res)
            plt.figure(1,figsize=(12,4))
            plt.subplot(1,1,1)
            c_indices = np.arange(np.size(where_all_c[0]))
            where_c_Kbb = np.where(np.isfinite(c_good_rv_list[where_all_c])*(np.abs(c_rel_res[where_all_c])<3))
            # where_c_Hbb = np.where(np.isfinite(c_good_rv_list[where_all_c])*(c_mjdobs_list[where_all_c] < 56498)*(np.abs(c_rel_res[where_all_c])<10)*("Hbb"==c_ifs_fitler_list[where_all_c]))
            where_c_Kbbbad = np.where((np.abs(c_rel_res[where_all_c])>=3))
            # where_c_Hbbbad = np.where(np.isfinite(c_good_rv_list[where_all_c])*((c_mjdobs_list[where_all_c] >= 56498)+(np.abs(c_rel_res[where_all_c])>=10))*("Hbb"==c_ifs_fitler_list[where_all_c]))
            plt.plot([0,np.size(where_all_c[0])],[c_combined_avg,c_combined_avg],color=mycolor)

            plt.errorbar(c_indices[where_c_Kbb]+nbid/5,c_good_rv_list[where_all_c][where_c_Kbb]-c_bary_star_list[where_all_c][where_c_Kbb],
                         yerr=np.sqrt(c_rvsig_list[where_all_c][where_c_Kbb]**2+c_wvsolerr_list[where_all_c][where_c_Kbb]**2),fmt="none",color=mycolor)#,label="$1\sigma$ error"
            # plt.errorbar(c_indices[where_c_Kbb],c_good_rv_list[where_all_c][where_c_Kbb]-c_bary_star_list[where_all_c][where_c_Kbb],
            #              yerr=c_rvsig_list[where_all_c][where_c_Kbb],fmt="none",color=mycolor)
            # eb1 = plt.errorbar(c_indices[where_c_Kbbbad],c_good_rv_list[where_all_c][where_c_Kbbbad]-c_bary_star_list[where_all_c][where_c_Kbbbad],
            #              yerr=np.sqrt(c_rvsig_list[where_all_c][where_c_Kbbbad]**2+c_wvsolerr_list[where_all_c][where_c_Kbbbad]**2),fmt="x",color="grey")
            # eb1[-1][0].set_linestyle(":")
            plt.plot(c_indices[where_c_Kbb]+nbid/5,c_good_rv_list[where_all_c][where_c_Kbb]-c_bary_star_list[where_all_c][where_c_Kbb],mymark,color=mycolor,label=mylabel)

            # plt.errorbar(c_indices[where_c_Hbb],c_good_rv_list[where_all_c][where_c_Hbb]-c_bary_star_list[where_all_c][where_c_Hbb],
            #              yerr=np.sqrt(c_rvsig_list[where_all_c][where_c_Hbb]**2+c_wvsolerr_list[where_all_c][where_c_Hbb]**2),fmt="none",color="#cc6600")
            # plt.errorbar(c_indices[where_c_Hbb],c_good_rv_list[where_all_c][where_c_Hbb]-c_bary_star_list[where_all_c][where_c_Hbb],
            #              yerr=c_rvsig_list[where_all_c][where_c_Hbb],fmt="none",color="#ff9900")
            # eb1 = plt.errorbar(c_indices[where_c_Hbbbad],c_good_rv_list[where_all_c][where_c_Hbbbad]-c_bary_star_list[where_all_c][where_c_Hbbbad],
            #              yerr=np.sqrt(c_rvsig_list[where_all_c][where_c_Hbbbad]**2+c_wvsolerr_list[where_all_c][where_c_Hbbbad]**2),fmt="o",color="grey")
            # eb1[-1][0].set_linestyle(":")
            # plt.plot(c_indices[where_c_Hbb],c_good_rv_list[where_all_c][where_c_Hbb]-c_bary_star_list[where_all_c][where_c_Hbb],"o",color="#ff9900",label="H-band")

            # plt.fill_between([0,np.size(where_all_c[0])],c_combined_avg-c_combined_sig,c_combined_avg+c_combined_sig,alpha=0.2,color="#cc3300")

            c_chi2 = np.sum(((c_good_rv_list[where_esti_c]-c_bary_star_list[where_esti_c]-c_combined_avg)/np.sqrt(c_rvsig_list[where_esti_c]**2+c_wvsolerr_list[where_esti_c]**2))**2)
            c_N_data = np.size(c_good_rv_list[where_esti_c])
            print(numbasis,"chi2",c_chi2,c_N_data,c_chi2/c_N_data)
            print(numbasis,"RV",c_combined_avg,c_combined_sig,c_combined_sig*c_chi2/c_N_data)

            combined_RV_list.append(c_combined_avg)
            combined_RVerr_list.append(c_combined_sig)

    # plt.plot([0,np.size(where_all_c[0])],[1000,1000],color="black",label="Weighted Mean")
    plt.fill_between([0,np.size(where_all_c[0])],rv_star-0.8,rv_star+0.8,alpha=0.3,color="grey",label="HR 8799 RV")
    # plt.ylim([-35,10])
    for date in np.unique(c_date_list[where_all_c]):
        where_data = np.where(c_date_list[where_all_c]==date)
        first = where_data[0][0]
        last = where_data[0][-1]
        plt.annotate("",xy=(first,-15+rv_star),xytext=(last+0.001,-15+rv_star),xycoords="data",arrowprops={'arrowstyle':"|-|",'shrinkA':0.1,'shrinkB':0.1})
        # plt.gca().text((first+last)/2-1,-20+rv_star,date,ha="left",va="top",rotation=-60,size=fontsize)
        plt.gca().text((first+last)/2-1,-17+rv_star,date,ha="left",va="top",rotation=0,size=fontsize)
    plt.gca().text(np.size(where_all_c[0]),30+rv_star,"HR 8799 d",ha="right",va="top",rotation=0,size=fontsize*1.5)
    plt.ylabel("RV (km/s)",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_position(("data",0))
    plt.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="upper left",bbox_to_anchor=[0,1.0],frameon=True,fontsize=fontsize,ncol=4)
    if 1:
        print("Saving "+os.path.join(out_pngs,"HR_8799_d",myoutfilename))
        plt.savefig(os.path.join(out_pngs,"HR_8799_d",myoutfilename),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"HR_8799_d",myoutfilename.replace(".pdf",".png")),bbox_inches='tight')
        # print("Saving "+os.path.join(out_pngs,"kap_And","RV_kap_And_measurements_kl{0}.pdf".format(numbasis)))
        # plt.savefig(os.path.join(out_pngs,"kap_And","RV_kap_And_measurements_kl{0}.pdf".format(numbasis)),bbox_inches='tight')
        # plt.savefig(os.path.join(out_pngs,"kap_And","RV_kap_And_measurements_kl{0}.png".format(numbasis)),bbox_inches='tight')
    plt.show()
    plt.close(1)

    print(combined_RV_list)
    print(combined_RVerr_list)
    plt.figure(2)
    if fakecorr:
        plt.plot(numbasis_list,combined_RV_list)
        plt.errorbar(numbasis_list,combined_RV_list,yerr=combined_RVerr_list,label="Correted from simulated planets")
    else:
        plt.plot(numbasis_list,combined_RV_list)
        plt.errorbar(numbasis_list,combined_RV_list,yerr=combined_RVerr_list,label="No additional correction")
plt.legend()
plt.show()
exit()

if 0:
    exit()
    final_planetRV_hd = np.linspace(-30,30,6000)


    b_int_mjdobs_list = [int(day) for day in b_mjdobs_list]
    c_int_mjdobs_list = [int(day) for day in c_mjdobs_list]
    b_int_mjdobs_list = [int(day) for day in b_mjdobs_list]
    c_int_mjdobs_list = [int(day) for day in c_mjdobs_list]

    where_esti_b = np.where(np.isfinite(b_good_rv_list)*(b_mjdobs_list > 55388)*(np.abs(b_rel_res)<3))#*(b_snr_list>6))
    where_esti_c = np.where(np.isfinite(c_good_rv_list)*(c_mjdobs_list > 55388)*(c_mjdobs_list < 56498)*(np.abs(c_rel_res)<3))

    unique_date = np.unique(np.concatenate([np.array(b_int_mjdobs_list)[where_esti_b],np.array(c_int_mjdobs_list)[where_esti_c]]))

    b_combined_avg_list = []
    b_combined_sig_list = []
    b_chi2_list = []
    b_N_data_list = []
    b_uniquedate_list = []
    for k,day in enumerate(unique_date):
        where_b = np.where(np.isfinite(b_good_rv_list)*(day == b_int_mjdobs_list)*(b_mjdobs_list > 55388)*(np.abs(b_rel_res)<3))
        if np.size(where_b[0]) == 0:
            b_combined_avg_list.append(np.nan)
            b_combined_sig_list.append(np.nan)
            b_chi2_list.append(np.nan)
            b_N_data_list.append(np.nan)
            b_uniquedate_list.append("")
            continue
        b_combined_avg = np.sum((b_good_rv_list[where_b]-b_bary_star_list[where_b])/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2))/np.sum(1/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2))
        b_combined_sig = 1/np.sqrt(np.sum(1/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2)))

        b_chi2 = np.sum(((b_good_rv_list[where_b]-b_bary_star_list[where_b]-b_combined_avg)/np.sqrt(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2))**2)
        b_N_data = np.size(b_good_rv_list[where_b])

        b_combined_avg_list.append(b_combined_avg)
        b_combined_sig_list.append(b_combined_sig)
        b_chi2_list.append(b_chi2)
        b_N_data_list.append(b_N_data)
        b_uniquedate_list.append(b_date_list[where_b][0])
    b_combined_avg_list = np.array(b_combined_avg_list)
    b_combined_sig_list = np.array(b_combined_sig_list)
    b_pessimistic_weighted_mean = np.nansum(b_combined_avg_list/b_combined_sig_list**2)/np.nansum(1/b_combined_sig_list**2)
    weighted_mean_sig = 1/np.sqrt(np.nansum(1/b_combined_sig_list**2))
    chi2 = np.nansum(((b_combined_avg_list-b_pessimistic_weighted_mean)**2/b_combined_sig_list**2))
    print(np.sqrt(chi2/np.size(b_combined_avg_list)))
    b_pessimistic_weighted_mean_sig = weighted_mean_sig*np.sqrt(chi2/np.size(b_combined_avg_list))

    print(b_pessimistic_weighted_mean,b_pessimistic_weighted_mean_sig)

    # b_combined_avg_list,b_combined_sig_list
    b_combined_avg_nonans_list = b_combined_avg_list[np.where(np.isfinite(b_combined_avg_list))]
    b_combined_sig_nonans_list = b_combined_sig_list[np.where(np.isfinite(b_combined_avg_list))]
    # b_combined_sig_nonans_list = np.zeros(b_combined_sig_nonans_list.shape)
    print(b_combined_avg_nonans_list)
    print(b_combined_sig_nonans_list)
    sig2_list = np.linspace(0.2,5,2000)
    dsig2 = sig2_list[1]-sig2_list[0]
    alpha_list = np.linspace(b_pessimistic_weighted_mean-10,b_pessimistic_weighted_mean+10,10000)
    dalpha = alpha_list[1]-alpha_list[0]
    sig2_grid,alpha_grid = np.meshgrid(sig2_list,alpha_list)
    print(sig2_grid.shape)
    print("coucou")
    sig2_post = 1/np.sqrt((2*np.pi)**(np.size(b_combined_sig_nonans_list))*np.prod(b_combined_sig_nonans_list[None,None,:]**2+sig2_grid[:,:,None]**2,axis=2))*\
                np.exp(-0.5*np.sum((b_combined_avg_nonans_list[None,None,:]-alpha_grid[:,:,None])**2/(b_combined_sig_nonans_list[None,None,:]**2+sig2_grid[:,:,None]**2),axis=2))
                # *1/sig2_list**2
    alpha_post = 1/np.sqrt((2*np.pi)**(np.size(b_combined_sig_nonans_list))*np.prod(b_combined_sig_nonans_list[None,:]**2,axis=1))*\
                np.exp(-0.5*np.sum((b_combined_avg_nonans_list[None,:]-alpha_list[:,None])**2/(b_combined_sig_nonans_list[None,:]**2),axis=1))
    print(sig2_post.shape)
    # plt.figure(12)
    # plt.imshow(sig2_post)
    # plt.show()
    new_b_pessimistic_posterior  = np.sum(sig2_post,axis=1)
    new_b_pessimistic_posterior = new_b_pessimistic_posterior/np.max(new_b_pessimistic_posterior)
    new_b_pessimistic_posterior_f = interp1d(alpha_list,new_b_pessimistic_posterior,bounds_error=False,fill_value=0)
    print("P(d|H1)",np.sum(sig2_post)*dsig2*dalpha)
    print("P(d|H2)",np.sum(alpha_post)*dalpha)
    print("P(d|H1)/P(d|H2)",np.sum(sig2_post)*dsig2*dalpha/(np.sum(alpha_post)*dalpha))
    print("P(d|H0)",1/np.sqrt((2*np.pi)**(np.size(b_combined_sig_nonans_list))*np.prod(b_combined_sig_nonans_list[None,:]**2))*\
                np.exp(-0.5*np.sum((b_combined_avg_nonans_list[None,:]-b_pessimistic_weighted_mean)**2/(b_combined_sig_nonans_list[None,:]**2),axis=1)))
    # plt.figure(10)
    # plt.imshow(sig2_post,extent=[sig2_list[0],sig2_list[-1],alpha_list[0],alpha_list[-1]])
    # plt.figure(11)
    # plt.errorbar(np.arange(np.size(b_combined_avg_nonans_list)),b_combined_avg_nonans_list-b_pessimistic_weighted_mean,yerr=b_combined_sig_nonans_list,ecolor="red")
    # plt.show()
    # print()
    # exit()
    out = np.zeros(1000)
    for k in range(np.size(out)):
        out[k] = np.mean(np.random.choice(b_combined_avg_nonans_list,size=np.size(b_combined_avg_nonans_list),replace=True))
    bootstrap_std = np.std(out)
    print("bootstrap_std",bootstrap_std)
    # exit()
    # print(np.std(b_combined_avg_nonans_list))
    # exit()
    # b_pessimistic_posterior,bin_edges = np.histogram(out,bins=200,range=[-20,0])
    # b_pessimistic_posterior = b_pessimistic_posterior/np.max(b_pessimistic_posterior)
    # bin_center = (bin_edges[1::]+bin_edges[0:np.size(bin_edges)-1])/2
    # b_pessimistic_posterior = interp1d(bin_center,b_pessimistic_posterior,bounds_error=False,fill_value=0)(final_planetRV_hd)
    # plt.figure(10)
    # plt.plot(bin_center,hist)
    # plt.show()
    # exit()
    b_posterior = np.exp(-0.5*(final_planetRV_hd-b_pessimistic_weighted_mean)**2/b_pessimistic_weighted_mean_sig**2)
    b_posterior = b_posterior/np.max(b_posterior)

    c_combined_avg_list = []
    c_combined_sig_list = []
    c_chi2_list = []
    c_N_data_list = []
    c_uniquedate_list = []
    for k,day in enumerate(unique_date):
        where_c = np.where(np.isfinite(c_good_rv_list)*(c_mjdobs_list > 55388)*(c_mjdobs_list < 56498)*(day == c_int_mjdobs_list)*(np.abs(c_rel_res)<3))
        if np.size(where_c[0]) == 0:
            c_combined_avg_list.append(np.nan)
            c_combined_sig_list.append(np.nan)
            c_chi2_list.append(np.nan)
            c_N_data_list.append(np.nan)
            c_uniquedate_list.append("")
            continue
        c_combined_avg = np.sum((c_good_rv_list[where_c]-c_bary_star_list[where_c])/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))/np.sum(1/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))
        c_combined_sig = 1/np.sqrt(np.sum(1/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2)))

        c_chi2 = np.sum(((c_good_rv_list[where_c]-c_bary_star_list[where_c]-c_combined_avg)/np.sqrt(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))**2)
        c_N_data = np.size(c_good_rv_list[where_c])

        c_combined_avg_list.append(c_combined_avg)
        c_combined_sig_list.append(c_combined_sig)
        c_chi2_list.append(c_chi2)
        c_N_data_list.append(c_N_data)
        c_uniquedate_list.append(c_date_list[where_c][0])
    c_combined_avg_list = np.array(c_combined_avg_list)
    c_combined_sig_list = np.array(c_combined_sig_list)
    c_pessimistic_weighted_mean = np.nansum(c_combined_avg_list/c_combined_sig_list**2)/np.nansum(1/c_combined_sig_list**2)
    c_pessimisticweighted_mean_sig = 1/np.sqrt(np.nansum(1/c_combined_sig_list**2))
    # chi2 = np.sum(((c_combined_avg_list-weighted_mean)**2/c_combined_sig_list**2))
    # weighted_mean_sig = weighted_mean_sig*np.sqrt(chi2/np.size(c_combined_avg_list))


    # get posteriors
    if 1:
        c_logposterior = []
        for filename,kcen,lcen,bary_star,wvsolerr in zip(c_out_filelist[where_esti_c],c_kcen_list[where_esti_c],c_lcen_list[where_esti_c],c_bary_star_list[where_esti_c],c_wvsolerr_list[where_esti_c]):
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
        for filename,kcen,lcen,bary_star,wvsolerr in zip(b_out_filelist[where_esti_b],b_kcen_list[where_esti_b],b_lcen_list[where_esti_b],b_bary_star_list[where_esti_b],b_wvsolerr_list[where_esti_b]):
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
        c_optimistic_posterior = np.exp(c_sumlogposterior-np.max(c_sumlogposterior))
        b_optimistic_posterior = np.exp(b_sumlogposterior-np.max(b_sumlogposterior))

        b_combined_avg,b_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,b_optimistic_posterior)
        c_combined_avg,c_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,c_optimistic_posterior)
        c_chi2 = np.sum(((c_good_rv_list[where_esti_c]-c_bary_star_list[where_esti_c]-c_combined_avg)/np.sqrt(c_rvsig_list[where_esti_c]**2+c_wvsolerr_list[where_esti_c]**2))**2)
        c_N_data = np.size(c_good_rv_list[where_esti_c])
        b_chi2 = np.sum(((b_good_rv_list[where_esti_b]-b_bary_star_list[where_esti_b]-b_combined_avg)/np.sqrt(b_rvsig_list[where_esti_b]**2+b_wvsolerr_list[where_esti_b]**2))**2)
        b_N_data = np.size(b_good_rv_list[where_esti_b])

        print("coucou",c_chi2/c_N_data,b_chi2/b_N_data)


        c_optimistic_posterior_func = interp1d(final_planetRV_hd,c_optimistic_posterior,bounds_error=False,fill_value=0)
        b_optimistic_posterior_func = interp1d(final_planetRV_hd,b_optimistic_posterior,bounds_error=False,fill_value=0)
        c_posterior = c_optimistic_posterior_func((final_planetRV_hd-final_planetRV_hd[np.argmax(c_optimistic_posterior)])/np.sqrt(c_chi2/c_N_data)+final_planetRV_hd[np.argmax(c_optimistic_posterior)])
        b_posterior = b_optimistic_posterior_func((final_planetRV_hd-final_planetRV_hd[np.argmax(b_optimistic_posterior)])/np.sqrt(b_chi2/b_N_data)+final_planetRV_hd[np.argmax(b_optimistic_posterior)])
        c_posterior = c_posterior/np.max(c_posterior)
        b_posterior = b_posterior/np.max(b_posterior)
        c_pessimistic_posterior = np.convolve(c_posterior,np.exp(-0.5*final_planetRV_hd**2/bootstrap_std**2),mode="same")
        b_pessimistic_posterior = np.convolve(b_posterior,np.exp(-0.5*final_planetRV_hd**2/bootstrap_std**2),mode="same")
        c_pessimistic_posterior = c_pessimistic_posterior/np.max(c_pessimistic_posterior)
        b_pessimistic_posterior = b_pessimistic_posterior/np.max(b_pessimistic_posterior)


        where_esti_b = np.where(np.isfinite(b_good_rv_list)\
                                #>2009
                                *(b_mjdobs_list > 55388) \
                                # <2018
                                #*(b_mjdobs_list < 58321) \
                                # !=20100713
                                #*(b_mjdobs_list.astype(np.int) != 55390) \
                                # !=20161106
                                #*(b_mjdobs_list.astype(np.int) != 57698) \
                                *(np.abs(b_rel_res)<3) )#*(b_snr_list>6))
        b_logposterior = []
        for filename,kcen,lcen,bary_star,wvsolerr in zip(b_out_filelist[where_esti_b],b_kcen_list[where_esti_b],b_lcen_list[where_esti_b],b_bary_star_list[where_esti_b],b_wvsolerr_list[where_esti_b]):
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

        b_sumlogposterior = np.nansum(b_logposterior,axis=0)
        b_optimistic_posterior = np.exp(b_sumlogposterior-np.max(b_sumlogposterior))

        b_combined_avg,b_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,b_optimistic_posterior)
        b_chi2 = np.sum(((b_good_rv_list[where_esti_b]-b_bary_star_list[where_esti_b]-b_combined_avg)/np.sqrt(b_rvsig_list[where_esti_b]**2+b_wvsolerr_list[where_esti_b]**2))**2)
        b_N_data = np.size(b_good_rv_list[where_esti_b])
        b_optimistic_posterior_func = interp1d(final_planetRV_hd,b_optimistic_posterior,bounds_error=False,fill_value=0)
        b_posterior = b_optimistic_posterior_func((final_planetRV_hd-final_planetRV_hd[np.argmax(b_optimistic_posterior)])/np.sqrt(b_chi2/b_N_data)+final_planetRV_hd[np.argmax(b_optimistic_posterior)])
        b_posterior = b_posterior/np.max(b_posterior)

        b_combined_avg,b_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,b_posterior)
        c_combined_avg,c_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,c_posterior)

        b_combined_avg_pess,b_combined_sig_pess,_ = get_err_from_posterior(final_planetRV_hd,b_pessimistic_posterior)
        c_combined_avg_pess,c_combined_sig_pess,_ = get_err_from_posterior(final_planetRV_hd,c_pessimistic_posterior)

        print("coucou",c_chi2,b_chi2)

    unique_strdate = np.unique(np.concatenate([c_uniquedate_list,b_uniquedate_list]))
    unique_strdate = unique_strdate[np.where(unique_strdate!="")]

    plt.figure(2,figsize=(12,12))
    plt.subplot(3,1,1)
    plt.fill_betweenx([0,np.size(unique_date)],rv_star-1.4,rv_star+1.4,alpha=0.2,color="grey",label="HR 8799 RV")

    print("bonjour",c_combined_avg_list,c_combined_sig_list,c_uniquedate_list,c_N_data_list)
    print("Planet & Date & RV & N cubes \\\\")
    for a,b,c,d in zip(c_combined_avg_list,c_combined_sig_list,c_uniquedate_list,c_N_data_list):
        if np.isnan(a):
            continue
        formated_date =  c[0:4]+"-"+c[4:6]+"-"+c[6:8]
        print("& {0} & ${1:.1f} \\pm {2:.1f}$ & {3} \\\\".format(formated_date,a,b,int(d)))
    plt.errorbar(c_combined_avg_list,np.arange(np.size(unique_date)),xerr=c_combined_sig_list,fmt="none",color="#ff9900")
    plt.plot(c_combined_avg_list,np.arange(np.size(unique_date)),"x",color="#ff9900",label="c (# exposures)")
    wherenotnans = np.where(np.isfinite(c_combined_avg_list))
    for y,(x,date,num) in enumerate(zip(c_combined_avg_list,c_uniquedate_list,c_N_data_list)):
        if np.isnan(x) or (x< (-16)):
            continue
        plt.gca().text(x,y,"{0}".format(int(num)),ha="center",va="bottom",rotation=0,size=fontsize,color="#cc3300",alpha=1)
    # print(unique_strdate)
    plt.plot([c_combined_avg-c_combined_sig,c_combined_avg-c_combined_sig],[0,np.size(unique_date)],linestyle="--",linewidth=2,color="#cc6600",alpha=0.4)
    plt.plot([c_combined_avg,c_combined_avg],[0,np.size(unique_date)],linestyle="-",linewidth=2,color="#cc3300",alpha=0.4)
    plt.plot([c_combined_avg+c_combined_sig,c_combined_avg+c_combined_sig],[0,np.size(unique_date)],linestyle="--",linewidth=2,color="#cc6600",alpha=0.4)
    # plt.gca().text(c_combined_avg-0.25,np.size(unique_date)+0.5,"${0:.1f}\pm {1:.1f}$ km/s".format(c_combined_avg,c_combined_sig),ha="center",va="bottom",rotation=0,size=fontsize,color="#cc3300")
    # plt.gca().text(c_combined_avg_pess+0.05,0.7,"$\pm {1:.1f}$ km/s".format(c_combined_avg_pess,c_combined_sig_pess),ha="left",va="top",rotation=90,size=fontsize,color="#cc3300",alpha=0.5)

    b_N_data_list = np.array(b_N_data_list)
    # where_solid = np.where((b_N_data_list>2)*(unique_strdate != "20180722"))
    # where_dash  = np.where(((b_N_data_list<=2)+(unique_strdate == "20180722")))
    where_solid = np.where((b_N_data_list>=0))
    where_dash  = np.where(((b_N_data_list<0)))
    print("bonjour",b_combined_avg_list,b_combined_sig_list)
    print("Planet & Date & RV  & N cubes \\\\")
    for a,b,c,d in zip(b_combined_avg_list,b_combined_sig_list,b_uniquedate_list,b_N_data_list):
        if np.isnan(a):
            continue
        formated_date =  c[0:4]+"-"+c[4:6]+"-"+c[6:8]
        print("& {0} & ${1:.1f} \\pm {2:.1f}$ & {3} \\\\".format(formated_date,a,b,int(d)))
    plt.errorbar(b_combined_avg_list[where_solid],np.arange(np.size(unique_date))[where_solid],xerr=b_combined_sig_list[where_solid],fmt="none",color="#0099cc")
    plt.plot(b_combined_avg_list[where_solid],np.arange(np.size(unique_date))[where_solid],"x",color="#0099cc",label="b (# exposures)")
    eb1 = plt.errorbar(b_combined_avg_list[where_dash],np.arange(np.size(unique_date))[where_dash],xerr=b_combined_sig_list[where_dash],fmt="none",color="#0099cc")
    eb1[-1][0].set_linestyle("--")
    plt.plot(b_combined_avg_list[where_dash],np.arange(np.size(unique_date))[where_dash],"x",color="#0099cc")
    wherenotnans = np.where(np.isfinite(b_combined_avg_list))
    for y,(x,date,num) in enumerate(zip(b_combined_avg_list,unique_strdate,b_N_data_list)):
        if np.isnan(x):
            continue
        if date == "20180722":
            plt.gca().text(x,y,"{0} (35 mas scale)".format(int(num)),ha="center",va="bottom",rotation=0,size=fontsize,color="#003366",alpha=1)
        else:
            plt.gca().text(x,y,"{0}".format(int(num)),ha="center",va="bottom",rotation=0,size=fontsize,color="#003366",alpha=1)
    plt.plot([b_combined_avg-b_combined_sig,b_combined_avg-b_combined_sig],[0,np.size(unique_date)],linestyle=":",linewidth=2,color="#006699",alpha=0.4)
    plt.plot([b_combined_avg,b_combined_avg],[0,np.size(unique_date)],linestyle="-.",linewidth=2,color="#003366",alpha=0.4)
    plt.plot([b_combined_avg+b_combined_sig,b_combined_avg+b_combined_sig],[0,np.size(unique_date)],linestyle=":",linewidth=2,color="#006699",alpha=0.4)
    # plt.gca().text(b_combined_avg+0.25,np.size(unique_date)+0.5,"${0:.1f}\pm {1:.1f}$ km/s".format(b_combined_avg,b_combined_sig),ha="center",va="bottom",rotation=0,size=fontsize,color="#003366")
    # plt.gca().text(b_combined_avg_pess+0.05,0.7,"$\pm {1:.1f}$ km/s".format(b_combined_avg_pess,b_combined_sig_pess),ha="left",va="top",rotation=90,size=fontsize,color="#003366",alpha=0.5)

    # er1 = plt.errorbar(-8.9,-1,xerr=2.5,color="#ff9900")
    # er1[-1][0].set_linestyle("--")
    # er2 = plt.errorbar(-10.9,-1.25,xerr=0.5,color="grey")
    # er2[-1][0].set_linestyle("--")
    # unique_strdate = np.insert(unique_strdate,0,"2016-17")


    plt.xlim([rv_star-4,rv_star+12])
    # plt.ylim([0,1.1])
    plt.xlabel("RV (km/s)",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.yticks(np.arange(0,np.size(unique_date)),unique_strdate)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",-14))
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#

    # print(b_filelist[where_esti_b])
    # print(c_filelist[where_esti_c])
    # exit()
    plt.subplot(3,1,2)
    # c_combined_avg = np.sum((c_good_rv_list[where_b]-c_bary_star_list[where_c])/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))/np.sum(1/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2))
    # c_combined_sig = 1/np.sqrt(np.sum(1/(c_rvsig_list[where_c]**2+c_wvsolerr_list[where_c]**2)))
    # b_combined_avg = np.sum((b_good_rv_list[where_b]-b_bary_star_list[where_b])/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2))/np.sum(1/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2))
    # b_combined_sig = 1/np.sqrt(np.sum(1/(b_rvsig_list[where_b]**2+b_wvsolerr_list[where_b]**2)))
    # b_combined_avg,b_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,b_posterior)
    # c_combined_avg,c_combined_sig,_ = get_err_from_posterior(final_planetRV_hd,c_posterior)
    # print(c_combined_avg,c_combined_sig )
    # print(b_combined_avg,b_combined_sig)
    # print(b_combined_avg-c_combined_avg,np.sqrt(c_combined_sig**2+b_combined_sig**2))
    # plt.fill_betweenx([0,10],c_combined_avg-c_combined_sig,c_combined_avg+c_combined_sig,alpha=1,color="#cc3300")
    # plt.fill_betweenx([0,10],b_combined_avg-b_combined_sig,b_combined_avg+b_combined_sig,alpha=1,color="#003366")
    # plt.hist(c_good_rv_list[where_c]-c_bary_star_list[where_c], range=[-20,20],bins=20,histtype="bar",alpha=0.5,color="#ff9900",label="c: RV histogram")
    # plt.plot([c_combined_avg,c_combined_avg],[0,10],linestyle="-",linewidth=2,color="#cc3300",label="c: Best estimate")
    # plt.plot([c_combined_avg-c_combined_sig,c_combined_avg-c_combined_sig],[0,10],linestyle="--",linewidth=2,color="#cc6600",label="c: $1\sigma$ error")
    # plt.plot([c_combined_avg+c_combined_sig,c_combined_avg+c_combined_sig],[0,10],linestyle="--",linewidth=2,color="#cc6600")
    # plt.gca().text(c_combined_avg,0.7,"${0:.1f}\pm {1:.1f}$ km/s".format(c_combined_avg,c_combined_sig),ha="right",va="top",rotation=90,size=fontsize,color="#cc3300")
    # plt.gca().text(c_combined_avg_pess+0.05,0.7,"$\pm {1:.1f}$ km/s".format(c_combined_avg_pess,c_combined_sig_pess),ha="left",va="top",rotation=90,size=fontsize,color="#cc3300",alpha=0.5)
    plt.gca().text(c_combined_avg+0.25,1,"${0:.1f}\pm {1:.1f}$ km/s".format(c_combined_avg,c_combined_sig),ha="center",va="bottom",rotation=0,size=fontsize,color="#cc3300")
    plt.plot(final_planetRV_hd,c_posterior,linestyle="-",linewidth=3,color="#ff9900",label="c: Posterior")
    # plt.fill_between(final_planetRV_hd,c_posterior,c_pessimistic_posterior,linestyle="-",linewidth=3,color="#ff9900",alpha=0.3,label="c: With bootstrap")
    # plt.plot(final_planetRV_hd,np.exp(-0.5*(final_planetRV_hd-c_combined_avg)**2/c_combined_sig**2),linestyle="--",linewidth=3,color="black",label="c: posterior")

    # plt.hist(b_good_rv_list[where_b]-b_bary_star_list[where_b], range=[-20,20],bins=20,histtype="bar",alpha=0.5,color="#0099cc",label="b: RV histogram")
    # plt.plot([b_combined_avg,b_combined_avg],[0,10],linestyle="-.",linewidth=2,color="#003366",label="b: Best estimate")
    # plt.plot([b_combined_avg-b_combined_sig,b_combined_avg-b_combined_sig],[0,10],linestyle=":",linewidth=2,color="#006699",label="b: $1\sigma$ error")
    # plt.plot([b_combined_avg+b_combined_sig,b_combined_avg+b_combined_sig],[0,10],linestyle=":",linewidth=2,color="#006699")
    # plt.gca().text(b_combined_avg,0.7,"${0:.1f}\pm {1:.1f}$ km/s".format(b_combined_avg,b_combined_sig),ha="right",va="top",rotation=90,size=fontsize,color="#003366")
    # plt.gca().text(b_combined_avg_pess+0.05,0.7,"$\pm {1:.1f}$ km/s".format(b_combined_avg_pess,b_combined_sig_pess),ha="left",va="top",rotation=90,size=fontsize,color="#003366",alpha=0.5)
    plt.gca().text(b_combined_avg-0.7,1,"${0:.1f}\pm {1:.1f}$ km/s".format(b_combined_avg,b_combined_sig),ha="left",va="bottom",rotation=0,size=fontsize,color="#003366")
    plt.plot(final_planetRV_hd,b_posterior,linestyle="-",linewidth=3,color="#0099cc",label="b: Posterior")
    # plt.fill_between(final_planetRV_hd,b_posterior,b_pessimistic_posterior,linestyle="-",linewidth=3,color="#0099cc",alpha=0.3,label="b: With bootstrap")
    # plt.fill_between(final_planetRV_hd,b_posterior,new_b_pessimistic_posterior_f(final_planetRV_hd),linestyle="-",linewidth=3,color="#0099cc",alpha=0.3,label="b: joint sig2 fit")
    # plt.plot(final_planetRV_hd,np.exp(-0.5*(final_planetRV_hd-b_combined_avg)**2/b_combined_sig**2),linestyle="--",linewidth=3,color="black",label="b: posterior")
    
    plt.xlim([rv_star-4,rv_star+12])
    plt.ylim([0,1.1])
    plt.xlabel("RV (km/s)",fontsize=fontsize)
    plt.ylabel("$\propto \mathcal{P}(RV|d)$",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",rv_star))
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = handles#handles[0:3]+[handles[6]]+handles[3:6]+[handles[7]]
    new_labels = labels#labels[0:3]+[labels[6]]+labels[3:6]+[labels[7]]
    # print(handles)
    # print(labels)
    # exit()
    plt.legend(new_handles,new_labels,loc="upper right",frameon=True,fontsize=fontsize)#

    plt.subplot(3,1,3)
    with open("/data/osiris_data/jason_rv_bc_2011.csv", 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        dRV_list_table = list(csv_reader)
        dRV_list_data = np.array(dRV_list_table[1::]).astype(np.float)
        print("Jason",np.mean(np.abs(dRV_list_data)),np.std(np.abs(dRV_list_data)))
        dRV_list_data = np.concatenate([dRV_list_data,-dRV_list_data])
    dRV_hist,bin_edges = np.histogram(dRV_list_data,bins=400,range=[-20,20])
    dRV_hist = dRV_hist/np.max(dRV_hist)
    bin_center = (bin_edges[1::]+bin_edges[0:np.size(bin_edges)-1])/2
    dRV_posterior = interp1d(bin_center,dRV_hist,bounds_error=False,fill_value=0)(final_planetRV_hd)
    # plt.plot(final_planetRV_hd,dRV_posterior,linestyle="-",linewidth=1,color="black") #9966ff
    plt.fill_between(final_planetRV_hd,
                     dRV_posterior*0,
                     dRV_posterior,alpha=0.2,color="grey",label="Wang 2018 (Astrometry; bcde coplanar & stable)")


    astrometry_DATADIR = os.path.join("/data/osiris_data","astrometry")
    userv = "includingrvdata"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_bc",'rvs_diffbc_55392_{0}.fits'.format(userv))) as hdulist:
        diffrvs = hdulist[0].data
    diffrvs_post,xedges = np.histogram(diffrvs,bins=10*5,range=[-5,5])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    plt.plot(x_centers,diffrvs_post/np.max(diffrvs_post),linestyle="--",linewidth=1,color="#6600ff",label="Orbit fit (Astrometry & RV; bc coplanar)") #9966ff
    userv = "norvdata"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_bc",'rvs_diffbc_55392_{0}.fits'.format(userv))) as hdulist:
        diffrvs = hdulist[0].data
        diffrvs = np.concatenate([diffrvs,-diffrvs])
    diffrvs_post,xedges = np.histogram(diffrvs,bins=10*5,range=[-5,5])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    plt.plot(x_centers,diffrvs_post/np.max(diffrvs_post),linestyle=":",linewidth=1,color="black",label="Orbit fit (Astrometry; bc coplanar)") #9966ff

    delta_posterior = np.correlate(b_posterior,c_posterior,mode="same")
    delta_posterior = delta_posterior/np.max(delta_posterior)
    delta_pessimistic_posterior = np.correlate(b_pessimistic_posterior,c_pessimistic_posterior,mode="same")
    delta_pessimistic_posterior = delta_pessimistic_posterior/np.max(delta_pessimistic_posterior)
    deltaRV,deltaRV_sig,_ = get_err_from_posterior(final_planetRV_hd,delta_posterior)
    deltaRV_pess,deltaRV_sig_pess,_ = get_err_from_posterior(final_planetRV_hd,delta_pessimistic_posterior)
    confidence_interval = (1-np.cumsum(delta_posterior)[np.argmin(np.abs(final_planetRV_hd))]/np.sum(delta_posterior))
    # plt.plot(final_planetRV_hd,np.cumsum(delta_posterior)/np.max(np.cumsum(delta_posterior)),linestyle="-",linewidth=3,color="#6600ff",label="b: posterior") #9966ff
    # plt.plot([deltaRV,deltaRV],[0,10],linestyle="-",linewidth=2,color="#660066",label="Best estimate")
    # plt.plot([deltaRV-deltaRV_sig,deltaRV-deltaRV_sig],[0,10],linestyle="--",linewidth=2,color="#9966ff",label="$1\sigma$ error")
    # plt.plot([deltaRV+deltaRV_sig,deltaRV+deltaRV_sig],[0,10],linestyle="--",linewidth=2,color="#9966ff")
    # plt.gca().text(deltaRV,0.7,"${0:.1f}\pm {1:.1f}$ km/s".format(deltaRV,deltaRV_sig),ha="right",va="top",rotation=90,size=fontsize,color="#660066")
    # plt.gca().text(deltaRV_pess+0.05,0.7,"$\pm {1:.1f}$ km/s".format(deltaRV_pess,deltaRV_sig_pess),ha="left",va="top",rotation=90,size=fontsize,color="#660066",alpha=0.5)
    plt.gca().text(deltaRV-1,1.0,"${0:.1f}\pm {1:.1f}$ km/s".format(deltaRV,deltaRV_sig),ha="left",va="bottom",rotation=0,size=fontsize,color="#660066")
    plt.plot(final_planetRV_hd,delta_posterior,linestyle="-",linewidth=3,color="#6600ff",label="Data only (RV)") #9966ff
    # plt.fill_between(final_planetRV_hd,delta_posterior,delta_pessimistic_posterior,linestyle="-",linewidth=3,color="#6600ff",alpha=0.3,label="With bootstrap") #9966ff
    # plt.fill_between(final_planetRV_hd[np.where(final_planetRV_hd>0)],
    #                  np.zeros(np.size(final_planetRV_hd[np.where(final_planetRV_hd>0)])),
    #                  (delta_posterior/np.max(delta_posterior))[np.where(final_planetRV_hd>0)],alpha=0.2,color="#9966ff")


    plt.xlim([-4,12])
    plt.ylim([0,1.1])
    plt.xlabel(r"$RV_b-RV_c$ (km/s)",fontsize=fontsize)
    plt.ylabel("$\propto \mathcal{P}(RV_b-RV_c|d)$",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",0))
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#

    if 1:
        print("Saving "+os.path.join(out_pngs,"RV_HR_8799_bc_posterior.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bc_posterior.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bc_posterior.png"))
    plt.show()


    # print(np.mean(np.sqrt(c_rvsig_list[where_c][where_c_Kbb]**2+c_wvsolerr_list[where_c][where_c_Kbb]**2)))
    # print(np.std(b_good_rv_list[where_b][where_b_Kbb]-b_bary_star_list[where_b][where_b_Kbb]-b_combined_avg))
    # print(np.mean(np.sqrt(b_rvsig_list[where_b][where_b_Kbb]**2+b_wvsolerr_list[where_b][where_b_Kbb]**2)))
    # exit()


    plt.figure(3,figsize=(6,5))
    plt.hist(b_good_rv_list[where_b]-b_bary_star_list[where_b], label="b", range=[-20,20],bins=40,histtype="bar",alpha=0.5)
    plt.hist(c_good_rv_list[where_c]-c_bary_star_list[where_c], label="c", range=[-20,20],bins=40,histtype="bar",alpha=0.5)
    plt.xlabel("RV (km/s)",fontsize=fontsize)
    plt.ylabel("# measurements",fontsize=fontsize)
    plt.legend()





    plt.show()
    exit()