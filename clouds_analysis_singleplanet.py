__author__ = 'jruffio'

import sys
import multiprocessing as mp
import numpy as np
from copy import copy
from scipy.ndimage.filters import median_filter
import astropy.io.fits as pyfits
import itertools
from scipy import interpolate
import glob
import os
import csv
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


def get_err_from_posterior(x,posterior):
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    if len(x[0:np.min([argmax_post+1,len(x)])]) < 2:
        lx = x[0]
    else:
        tmp_cumpost = cum_posterior[0:np.min([argmax_post+1,len(x)])]
        tmp_x= x[0:np.min([argmax_post+1,len(x)])]
        deriv_tmp_cumpost = np.insert(tmp_cumpost[1::]-tmp_cumpost[0:np.size(tmp_cumpost)-1],np.size(tmp_cumpost)-1,0)
        try:
            whereinflection = np.where(deriv_tmp_cumpost<0)[0][0]
            where2keep = np.where((tmp_x<=tmp_x[whereinflection])+(tmp_cumpost>=tmp_cumpost[whereinflection]))
            tmp_cumpost = tmp_cumpost[where2keep]
            tmp_x = tmp_x[where2keep]
        except:
            pass
        lf = interp1d(tmp_cumpost,tmp_x,bounds_error=False,fill_value=x[0])
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx=x[-1]
    else:
        tmp_cumpost = cum_posterior[argmax_post::]
        tmp_x= x[argmax_post::]
        deriv_tmp_cumpost = np.insert(tmp_cumpost[1::]-tmp_cumpost[0:np.size(tmp_cumpost)-1],np.size(tmp_cumpost)-1,0)
        try:
            whereinflection = np.where(deriv_tmp_cumpost>0)[0][0]
            where2keep = np.where((tmp_x>=tmp_x[whereinflection])+(tmp_cumpost>=tmp_cumpost[whereinflection]))
            tmp_cumpost = tmp_cumpost[where2keep]
            tmp_x = tmp_x[where2keep]
        except:
            pass
        rf = interp1d(tmp_cumpost,tmp_x,bounds_error=False,fill_value=x[-1])
        rx = rf(1-0.6827)

    # plt.plot(x,posterior/np.max(posterior))
    # plt.fill_betweenx([0,1],[lx,lx],[rx,rx],alpha=0.5)
    # plt.show()
    return lx,x[argmax_post],rx,argmax_post


#------------------------------------------------
if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    print("CPU COUNT: {0}".format(mp.cpu_count()))



    kl = 10
    useprior = False
    if not useprior:
        priorTeff,priorTeff_sig = 1000,1e+4#1e-9
        priorlogg,priorlogg_sig = 4,1e+3#1e-9
    planet,color = "HR_8799_b","#0099cc"
    if useprior:
        priorTeff,priorTeff_sig = 900,1e-9#1e-9
        priorlogg,priorlogg_sig = 3.9,1e-9#1e-9
    # planet,color = "HR_8799_c","#ff9900"
    # if useprior:
    #     priorTeff,priorTeff_sig = 1060,1e-9#1e-9
    #     priorlogg,priorlogg_sig = 4.1,1e-9#1e-9
    # planet,color = "HR_8799_d","#6600ff"
    # if useprior:
    #     priorTeff,priorTeff_sig = 1060,1e-9#1e-9
    #     priorlogg,priorlogg_sig = 4.1,1e-9#1e-9
    IFSfilter = "*"
    scale = "*"
    date = "*"
    if "*" not in IFSfilter:
        suffix = IFSfilter
    else:
        suffix = "HK"
    if useprior:
        suffix = suffix + "_prior"
    addfilename = False
    if addfilename:
        suffix = suffix + "_wfilenames"
    inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/"
    # filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+"*"+"_"+scale+".fits"))
    filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
    filelist.sort()
    # outputfolder = "20200309_model"
    # outputfolder = "sherlock/20200312_travisgridpost"
    # outputfolder = "sherlock/20200714_clouds_gridpost"
    outputfolder = "sherlock/20200823_clouds_gridpost"
    fake_str = ""
    # outputfolder = "sherlock/20200427_travisgridpost"
    # fake_str = "_fk"
    gridname = os.path.join("/data/osiris_data/","hr8799b_modelgrid")
    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
    myoutfilename = "clouds_"+planet+"_measurements_kl{0}_{1}{2}.pdf".format(kl,suffix,fake_str)
    #myoutfilename = "clouds_"+planet+"_measurements_kl{0}_{1}{2}_best10SNR.pdf".format(kl,suffix,fake_str)

    c_kms = 299792.458
    R= 4000
    fontsize=12


    if kl ==0:
        c_fileinfos_filename = "/data/osiris_data/"+planet+"/fileinfos_Kbb_jb.csv"
    else:
        c_fileinfos_filename = "/data/osiris_data/"+planet+"/fileinfos_Kbb_jb_kl{0}.csv".format(kl)


    #read file
    with open(c_fileinfos_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        c_list_table = list(csv_reader)
        c_colnames = c_list_table[0]
        c_N_col = len(c_colnames)
        c_list_data = c_list_table[1::]
        c_filename_id = c_colnames.index("filename")
        c_snr_id = c_colnames.index("snr")
        c_status_id = c_colnames.index("status")
    db_filelist = np.array([item[c_filename_id] for item in c_list_data])
    db_snr_list = np.array([float(item[c_snr_id]) if item[c_snr_id] != "nan" else 0  for item in c_list_data])
    db_status_list = np.array([float(item[c_status_id]) for item in c_list_data])
    db_snr_selec = [db_snr_list[k] for k,db_status in enumerate(db_status_list) if db_status == 1]
    print(np.sort(db_snr_selec)[::-1])
    snr_threshold = np.sort(db_snr_selec)[::-1][9]
    # print(np.sort(db_snr_list)[::-1][0:np.size(db_snr_selec)])
    # exit()


    tmpfilename = os.path.join("/data/osiris_data/clouds_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,"Kbb"))
    hdulist = pyfits.open(tmpfilename)
    planet_model_grid =  hdulist[0].data
    oriplanet_spec_wvs =  hdulist[1].data
    Tlistunique =  hdulist[2].data
    logglistunique =  hdulist[3].data
    pgslistunique =  hdulist[4].data
    # Tlistunique =  hdulist[1].data
    # logglistunique =  hdulist[2].data
    # pgslistunique =  hdulist[3].data
    hdulist.close()

    print(planet_model_grid.shape,np.size(Tlistunique),np.size(logglistunique),np.size(pgslistunique),np.size(oriplanet_spec_wvs))
    myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,pgslistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

    bad_data = []#["s130725_a040001_Kbb_020.fits","s130726_a036001_Kbb_020.fits"]#np.array(["s161108_a043002_Kbb_020.fits"])#,"s161106_a040002_Kbb_020.fits","s161106_a040002_Kbb_020.fits","s100711_a033001_Kbb_020.fits"
    date_list = []
    filter_list = []
    new_file_list = []
    rawlogpost_list = []
    for file_id, filename in enumerate(filelist):
        tmpfilename = os.path.join(os.path.dirname(filename),outputfolder,os.path.basename(filename).replace(".fits","_kl{0}_logpost".format(kl)+fake_str+".fits"))
        # if "20100715" not in tmpfilename:
        #     continue
        # if "2009" in tmpfilename:
        #     continue
        if len(glob.glob(tmpfilename))!=1:
            print("No data on "+filename)
            continue
        else:
            print(filename)
        if os.path.basename(filename) in bad_data:
            # print("coucou")
            continue
        #if db_snr_list[np.where(db_filelist==filename)[0]] < snr_threshold:
        #    continue
        hdulist = pyfits.open(tmpfilename)
        logpost =  hdulist[0].data
        fitT_list =  hdulist[1].data
        fitlogg_list =  hdulist[2].data
        fitpgs_list =  hdulist[3].data
        planetRV_array0 =  hdulist[4].data
        if logpost.shape[0] != 25:
            continue
        if np.max(fitpgs_list) <= 1e6:
            continue
        rawlogpost_list.append(logpost)
        date = os.path.basename(filename).split("_a")[0].split("s")[1]
        date_list.append(date)
        new_file_list.append(filename)
        hdulist.close()
        filter_list.append(os.path.basename(filename).split("_")[2])
        # print(fitT_list)
        # print(fitlogg_list)
        # print(fitpgs_list)
        # exit()
    filter_list = np.array(filter_list)
    # print(filter_list)
    # exit()
    logpriorTeff_vec = -0.5/priorTeff_sig**2*(fitT_list-priorTeff)**2
    logpriorlogg_vec = -0.5/priorlogg_sig**2*(fitlogg_list-priorlogg)**2
    if 0: # remove outliers:
        print("coucou")
        print(np.array(rawlogpost_list).shape)
        tmp_combinedlogpost = np.nansum(np.array(rawlogpost_list),axis=0)
        # tmp_combinedlogpost =  tmp_combinedlogpost+ tmp_combinedlogpost[:,None,None,None] + tmp_combinedlogpost[None,:,None,None]
        tmp_combinedpost = np.exp(tmp_combinedlogpost-np.nanmax(tmp_combinedlogpost))
        tmp_combinedpost_rvmargi = np.nansum(tmp_combinedpost,axis=3)
        myargmax = np.unravel_index(np.nanargmax(tmp_combinedpost_rvmargi),tmp_combinedpost_rvmargi.shape)
        refopti = [fitpgs_list[myargmax[2]],fitT_list[myargmax[0]],fitlogg_list[myargmax[1]]]
        print(myargmax,refopti)
        for file_id, filename in enumerate(new_file_list):
            tmp_logpostlist = np.array(rawlogpost_list[0:file_id]+rawlogpost_list[file_id+1::])
            # print(tmp_logpostlist.shape)
            tmp_combinedlogpost = np.nansum(tmp_logpostlist,axis=0)
            # tmp_combinedlogpost =  tmp_combinedlogpost+ tmp_combinedlogpost[:,None,None,None] + tmp_combinedlogpost[None,:,None,None]
            tmp_combinedpost = np.exp(tmp_combinedlogpost-np.nanmax(tmp_combinedlogpost))
            tmp_combinedpost_rvmargi = np.nansum(tmp_combinedpost,axis=3)
            myargmax = np.unravel_index(np.nanargmax(tmp_combinedpost_rvmargi),tmp_combinedpost_rvmargi.shape)
            # print(myargmax)
            print(fitpgs_list[myargmax[2]]-refopti[2],fitT_list[myargmax[0]]-refopti[0],fitlogg_list[myargmax[1]]-refopti[1],filename)
        exit()
# (19, 40, 23) [1180.0, 4.333333333333333, 0.5834900032911392]
# -0.0054957310126582115 0.0 0.0 /data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a028001_Kbb_020.fits
# 0.0 0.0 0.0333333333333341 /data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a029001_Kbb_020.fits
# -0.0054957310126582115 0.0 0.0 /data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a035001_Kbb_020.fits
# 0.0 0.0 0.0333333333333341 /data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a040001_Kbb_020.fits
# -0.0054957310126582115 -20.0 -0.033333333333333215 /data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a041001_Kbb_020.fits
# -0.0054957310126582115 -20.0 -0.033333333333333215 /data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a059001_Kbb_020.fits
# 0.0 -20.0 0.0 /data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a039001_Kbb_020.fits
# 0.0 0.0 0.0333333333333341 /data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a036001_Kbb_020.fits
# 0.0 0.0 -0.033333333333333215 /data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a043002_Kbb_020.fits


    hr_fitT_list = np.linspace(fitT_list[0],fitT_list[-1],(len(fitT_list)-1)*1+1,endpoint=True)
    hr_fitlogg_list = np.linspace(fitlogg_list[0],fitlogg_list[-1],(len(fitlogg_list)-1)*1+1,endpoint=True)
    hr_fitpgs_list = np.linspace(fitpgs_list[0],fitpgs_list[-1],(len(fitpgs_list)-1)*2+1,endpoint=True)
    hr_planetRV_array0 = np.linspace(planetRV_array0[0],planetRV_array0[-1],(len(planetRV_array0)-1)+1,endpoint=True)
    Npts = np.size(hr_fitT_list)*np.size(hr_fitlogg_list)*np.size(hr_fitpgs_list)*np.size(hr_planetRV_array0)
    pts = np.rollaxis( np.rollaxis(np.array(np.meshgrid(hr_fitT_list,hr_fitlogg_list,hr_fitpgs_list,hr_planetRV_array0)),0,5),0,2).reshape(Npts,4)
    hr_logpriorTeff_vec = -0.5/priorTeff_sig**2*(hr_fitT_list-priorTeff)**2
    hr_logpriorlogg_vec = -0.5/priorlogg_sig**2*(hr_fitlogg_list-priorlogg)**2

    CtoO_CI_list = []
    CtoO_post_list = []
    post_list = []
    date_list = np.array(date_list)
    unique_dates = np.unique(date_list)
    combined_nightly_logpost = {}
    for file_id, (filename,date) in enumerate(zip(new_file_list,date_list)):
        tmpfilename = os.path.join(os.path.dirname(filename),outputfolder,os.path.basename(filename).replace(".fits","_kl{0}_logpost".format(kl)+fake_str+".fits"))
        print(tmpfilename)
        hdulist = pyfits.open(tmpfilename)
        logpost =  hdulist[0].data
        fitT_list =  hdulist[1].data
        fitlogg_list =  hdulist[2].data
        fitpgs_list =  hdulist[3].data
        planetRV_array0 =  hdulist[4].data
        hdulist.close()
        # print(fitpgs_list)
        myinterpgrid = RegularGridInterpolator((fitT_list,fitlogg_list,fitpgs_list,planetRV_array0),logpost,method="linear",bounds_error=False,fill_value=np.nanmin(logpost))
        hr_logpost = myinterpgrid(pts).reshape((np.size(hr_fitT_list),np.size(hr_fitlogg_list),np.size(hr_fitpgs_list),np.size(hr_planetRV_array0)))
        # hr_logpost = logpost
        try:
            combined_logpost += hr_logpost
            # combined_logpost += logpost
        except:
            combined_logpost = copy(hr_logpost)
            # combined_logpost = copy(logpost)
        try:
            combined_nightly_logpost[date] += hr_logpost
            # combined_nightly_logpost[date] += logpost
        except:
            combined_nightly_logpost[date] = copy(hr_logpost)
            # combined_nightly_logpost[date] = copy(logpost)
        logpost = logpost + logpriorTeff_vec[:,None,None,None] + logpriorlogg_vec[None,:,None,None]
        post = np.exp(logpost-np.nanmax(logpost))
        # logpriorTeff_vec = 1/(np.sqrt(2*np.pi)*priorTeff_sig)*np.exp(-0.5/priorTeff_sig**2*(fitT_list-priorTeff)**2)
        # logpriorlogg_vec = 1/(np.sqrt(2*np.pi)*priorlogg_sig)*np.exp(-0.5/priorlogg_sig**2*(fitlogg_list-priorlogg)**2)

        post_list.append(np.nansum(post,axis=3))
        # post_list.append(post[:,:,:,20])

        # import matplotlib.pyplot as plt
        # plt.figure(1)
        # temp_post = np.nansum(post,axis=(1,2,3))
        # temp_post /= np.nanmax(temp_post)
        # plt.plot(fitT_list,temp_post)
        #
        # plt.figure(2)
        # logg_post = np.nansum(post,axis=(0,2,3))
        # logg_post /= np.nanmax(logg_post)
        # plt.plot(fitlogg_list,logg_post)
        #
        # plt.figure(3)
        # # CtoO_post = np.nansum(post,axis=(0,1,3))
        # CtoO_post /= np.nanmax(CtoO_post)
        # plt.plot(fitpgs_list,CtoO_post)
        # plt.show()

        # CtoO_post = np.nansum(post[-1,-2,:,:],axis=1)
        CtoO_post = np.nansum(post,axis=(0,1,3))
        CtoO_post /= np.nanmax(CtoO_post)
        # CtoO_post /= np.nanmean(CtoO_post)
        leftCI,argmaxpost,rightCI,_ = get_err_from_posterior(fitpgs_list,CtoO_post)
        # print(leftCI,argmaxpost,rightCI)
        CtoO_CI_list.append([leftCI,argmaxpost,rightCI])
        CtoO_post_list.append(CtoO_post)
        # print(fitpgs_list)
        # print(leftCI,argmaxpost,rightCI)

        # plt.figure(4)
        # rv_post = np.nansum(post,axis=(0,1,2))
        # rv_post /= np.nanmax(rv_post)
        # plt.plot(planetRV_array0,rv_post)

        # plt.figure(2,figsize=(8,8))
        # combined_post_rvmargi = np.nansum(post,axis=3)
        # para_vec_list = [fitT_list,fitlogg_list,fitpgs_list]
        # xlabel_list = ["T (K)", "log(g) (cgs?)","C/O"]
        # Nparas = len(para_vec_list)
        # for k in range(Nparas):
        #     dims = np.arange(Nparas).tolist()
        #     dims.pop(k)
        #
        #     plt.subplot(Nparas,Nparas,k+1+k*Nparas)
        #     tmppost = np.sum(combined_post_rvmargi,axis=(*dims,))
        #     tmppost /= np.max(tmppost)
        #     plt.plot(para_vec_list[k],tmppost)
        #     plt.xlabel(xlabel_list[k])
        #
        #     for l in np.arange(k+1,Nparas):
        #         plt.subplot(Nparas,Nparas,k+1+(l)*Nparas)
        #         dims = np.arange(Nparas).tolist()
        #         dims.pop(k)
        #         dims.pop(l-1)
        #         tmppost = np.sum(combined_post_rvmargi,axis=(*dims,))
        #         tmppost /= np.max(tmppost)
        #         tmppost = tmppost.T
        #
        #         raveltmppost = np.ravel(tmppost)
        #         ind = np.argsort(raveltmppost)
        #         cum_posterior = np.zeros(np.shape(raveltmppost))
        #         cum_posterior[ind] = np.cumsum(raveltmppost[ind])
        #         cum_posterior = cum_posterior/np.max(cum_posterior)
        #         cum_posterior = np.reshape(cum_posterior,tmppost.shape)
        #         extent = [para_vec_list[k][0],para_vec_list[k][-1],para_vec_list[l][0],para_vec_list[l][-1]]
        #         # tmppost[np.where(cum_posterior<1-0.6827)] = np.nan
        #         plt.imshow(np.log10(tmppost),origin="lower",extent=extent,aspect=(para_vec_list[k][-1]-para_vec_list[k][0])/(para_vec_list[l][-1]-para_vec_list[l][0]))
        #         plt.contour(para_vec_list[k],para_vec_list[l],cum_posterior,levels=[1-0.9973,1-0.9545,1-0.6827],colors="black")
        #         plt.xlabel(xlabel_list[k])
        #         plt.ylabel(xlabel_list[l])
        #     plt.show()

    yval = np.array([argmaxpost for leftCI,argmaxpost,rightCI in CtoO_CI_list])
    m_yerr = np.array([argmaxpost-leftCI for leftCI,argmaxpost,rightCI in CtoO_CI_list])
    p_yerr = np.array([rightCI - argmaxpost for leftCI,argmaxpost,rightCI in CtoO_CI_list])
    # plt.errorbar(np.arange(len(CtoO_CI_list)),yval,yerr=[m_yerr,p_yerr ])

    plt.figure(1,figsize=(12,4))
    print(len(filter_list),len(CtoO_CI_list))
    wherefilter = np.where("Kbb" == filter_list)
    # plt.errorbar(np.arange(len(CtoO_CI_list))[wherefilter],yval[wherefilter],yerr=[m_yerr[wherefilter],p_yerr[wherefilter] ],fmt="none",color=color)
    # plt.plot(np.arange(len(CtoO_CI_list))[wherefilter],yval[wherefilter],"x",color=color) # #0099cc  #ff9900
    maxCtoOpost = 1#np.nanmax(np.array(CtoO_post_list))
    for k in wherefilter[0]:
        CtoO_post = CtoO_post_list[k]
        filename = new_file_list[k]
        if k == wherefilter[0][0]:
            plt.fill_betweenx(fitpgs_list,k-CtoO_post/maxCtoOpost/2.,k+CtoO_post/maxCtoOpost/2.,alpha=0.5,color=color,label="K-band")
        else:
            plt.fill_betweenx(fitpgs_list,k-CtoO_post/maxCtoOpost/2.,k+CtoO_post/maxCtoOpost/2.,alpha=0.5,color=color)
        if addfilename:
            plt.gca().text(k,fitpgs_list[-1],os.path.basename(filename),ha="left",va="top",rotation=-90,size=fontsize/2)
    wherefilter = np.where("Hbb" == filter_list)
    # eb = plt.errorbar(np.arange(len(CtoO_CI_list))[wherefilter],yval[wherefilter],yerr=[m_yerr[wherefilter],p_yerr[wherefilter] ],fmt="none",color=color)
    # eb[-1][0].set_linestyle("--")
    # plt.plot(np.arange(len(CtoO_CI_list))[wherefilter],yval[wherefilter],"o",color=color) # #0099cc  #ff9900
    for k in wherefilter[0]:
        CtoO_post = CtoO_post_list[k]
        filename = new_file_list[k]
        if k == wherefilter[0][0]:
            plt.fill_betweenx(fitpgs_list,k-CtoO_post/maxCtoOpost/2.,k+CtoO_post/maxCtoOpost/2.,alpha=0.5,hatch="/",facecolor="none",edgecolor=color,label="H-band")
        else:
            plt.fill_betweenx(fitpgs_list,k-CtoO_post/maxCtoOpost/2.,k+CtoO_post/maxCtoOpost/2.,alpha=0.5,hatch="/",facecolor="none",edgecolor=color)
        if addfilename:
            plt.gca().text(k,fitpgs_list[-1],os.path.basename(filename),ha="left",va="top",rotation=-90,size=fontsize/2)



    print(fitpgs_list)
    print(date_list)
    plt.ylim([fitpgs_list[0],fitpgs_list[-1]])
    # plt.xlim([0,len(CtoO_CI_list)+1])
    totlength = 75
    plt.xlim([0,totlength])
    for date in unique_dates:
        where_data = np.where(date_list==date)
        first = where_data[0][0]
        last = where_data[0][-1]
        # print(date,first,last)
        plt.annotate("",xy=(first,fitpgs_list[0]),xytext=(last+0.001,fitpgs_list[0]),xycoords="data",arrowprops={'arrowstyle':"|-|",'shrinkA':0.1,'shrinkB':0.1})
        plt.gca().text((first+last)/2-0.5,fitpgs_list[0]-0.01,date[2:4]+"-"+date[4:6]+"-20"+date[0:2],ha="left",va="top",rotation=-60,size=fontsize)
    plt.gca().text(len(CtoO_CI_list),fitpgs_list[-1],planet.replace("_"," "),ha="left",va="top",rotation=0,size=fontsize*1.5)
    plt.ylabel("C/O",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)#set_position(("data",0))
    plt.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="upper left",bbox_to_anchor=[len(CtoO_CI_list)/totlength,0.9],frameon=True,fontsize=fontsize)#
    # plt.tight_layout()
    # exit()

    if 1:
        print("Saving "+os.path.join(out_pngs,planet,myoutfilename))
        plt.savefig(os.path.join(out_pngs,planet,myoutfilename),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf",".png")),bbox_inches='tight')

    nightly_CtoO_CI_list = []
    nightly_CtoO_post_list = []
    nightly_Nexp = []
    for date in unique_dates:
        where_data = np.where(date_list==date)
        nightly_Nexp.append(len(where_data[0]))

        tmp_logpost = combined_nightly_logpost[date] + hr_logpriorTeff_vec[:,None,None,None] + hr_logpriorlogg_vec[None,:,None,None]
        tmp_post = np.exp(tmp_logpost-np.nanmax(tmp_logpost))
        tonight_CtoOpost = np.nansum(tmp_post,axis=(0,1,3))
        nightly_CtoO_post_list.append(tonight_CtoOpost)
        leftCI,argmaxpost,rightCI,_ = get_err_from_posterior(hr_fitpgs_list,tonight_CtoOpost)
        nightly_CtoO_CI_list.append([leftCI,argmaxpost,rightCI])


    nightly_yval = [argmaxpost for leftCI,argmaxpost,rightCI in nightly_CtoO_CI_list]
    nightly_m_yerr = [argmaxpost-leftCI for leftCI,argmaxpost,rightCI in nightly_CtoO_CI_list]
    nightly_p_yerr = [rightCI - argmaxpost for leftCI,argmaxpost,rightCI in nightly_CtoO_CI_list]
    print(["20"+date for date in  unique_dates])
    print("nightly_yval nightly_m_yerr nightly_p_yerr nightly_Nexp")
    print(nightly_yval)
    print(nightly_m_yerr)
    print(nightly_p_yerr)
    print(nightly_Nexp)





    combined_logpost = combined_logpost + hr_logpriorTeff_vec[:,None,None,None] + hr_logpriorlogg_vec[None,:,None,None]
    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=combined_logpost))
    hdulist.append(pyfits.ImageHDU(data=hr_fitT_list))
    hdulist.append(pyfits.ImageHDU(data=hr_fitlogg_list))
    hdulist.append(pyfits.ImageHDU(data=hr_fitpgs_list))
    hdulist.append(pyfits.ImageHDU(data=hr_planetRV_array0))
    try:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_logposterior.fits")), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_logposterior.fits")), clobber=True)
    hdulist.close()
    combined_post = np.exp(combined_logpost-np.nanmax(combined_logpost))
    combined_post_rvmargi = np.nansum(combined_post,axis=3)
    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=combined_post_rvmargi))
    hdulist.append(pyfits.ImageHDU(data=hr_fitT_list))
    hdulist.append(pyfits.ImageHDU(data=hr_fitlogg_list))
    hdulist.append(pyfits.ImageHDU(data=hr_fitpgs_list))
    try:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_posterior.fits")), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_posterior.fits")), clobber=True)
    hdulist.close()


    final_leftCI,final_yval,final_rightCI,_ = get_err_from_posterior(hr_fitpgs_list,np.nansum(combined_post,axis=(0,1,3)))
    final_m_yerr = final_yval - final_leftCI
    final_p_yerr = final_rightCI - final_yval
    print("final_yval,final_m_yerr,final_p_yerr")
    print(final_yval,final_m_yerr,final_p_yerr)
    # exit()
    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=np.array(["20"+date for date in  unique_dates]).astype(np.int)))
    hdulist.append(pyfits.ImageHDU(data=nightly_yval))
    hdulist.append(pyfits.ImageHDU(data=nightly_m_yerr))
    hdulist.append(pyfits.ImageHDU(data=nightly_p_yerr))
    hdulist.append(pyfits.ImageHDU(data=nightly_Nexp))
    hdulist.append(pyfits.ImageHDU(data=np.array([final_yval,final_m_yerr,final_p_yerr])))
    try:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_data4plotting.fits")), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_data4plotting.fits")), clobber=True)
    hdulist.close()

    f = plt.figure(2,figsize=(6,6))
    para_vec_list = [hr_fitT_list,hr_fitlogg_list,hr_fitpgs_list]
    xlabel_list = ["T [K]", "log(g/[1 cm/$\mathrm{s}^2$])","pgs"]
    xticks_list = [[800,1000,1200], [3.5,4.0,4.5,5.0],[5e5,1e6,4e6]]
    Nparas = len(para_vec_list)

    myargmax = np.unravel_index(np.nanargmax(combined_post_rvmargi),combined_post_rvmargi.shape)
    print(myargmax)
    print(hr_fitT_list[myargmax[0]],hr_fitlogg_list[myargmax[1]],hr_fitpgs_list[myargmax[2]])
    for k in range(Nparas):
        dims = np.arange(Nparas).tolist()
        dims.pop(k)

        plt.subplot(Nparas,Nparas,k+1+k*Nparas)
        ax = plt.gca()
        tmppost = np.sum(combined_post_rvmargi,axis=(*dims,))
        tmppost /= np.max(tmppost)
        # plt.plot(para_vec_list[k],tmppost,color=color)
        plt.fill_between(para_vec_list[k],tmppost*0,tmppost,color=color)
        plt.xlabel(xlabel_list[k], fontsize=fontsize)
        if k != 0:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        plt.ylim([0.0,1.1])
        # print(xticks_list[k])
        plt.xlim([para_vec_list[k][0],para_vec_list[k][-1]])
        if k != Nparas-1:
            # ax.xaxis.tick_top()
            # ax.xaxis.set_label_position("top")
            ax.yaxis.set_ticks([0.5,1.0])
            ax.xaxis.set_ticks(xticks_list[k])
        else:
            ax.yaxis.set_ticks([0.0,0.5,1.0])
            plt.xticks(xticks_list[k],["5e5","1e6","4e6"])
            # ax.spines['right'].set_visible(False)
            # ax.spines['top'].set_visible(False)
            # ax.xaxis.set_ticks_position('bottom')
            # ax.yaxis.set_ticks_position('left')
        for l in np.arange(k+1,Nparas):
            plt.subplot(Nparas,Nparas,k+1+(l)*Nparas)
            ax = plt.gca()
            dims = np.arange(Nparas).tolist()
            dims.pop(k)
            dims.pop(l-1)
            tmppost = np.sum(combined_post_rvmargi,axis=(*dims,))
            tmppost /= np.max(tmppost)
            tmppost = tmppost.T

            raveltmppost = np.ravel(tmppost)
            ind = np.argsort(raveltmppost)
            cum_posterior = np.zeros(np.shape(raveltmppost))
            cum_posterior[ind] = np.cumsum(raveltmppost[ind])
            cum_posterior = cum_posterior/np.max(cum_posterior)
            cum_posterior = np.reshape(cum_posterior,tmppost.shape)
            # dk = para_vec_list[k][1]-para_vec_list[k][0]
            # dl = para_vec_list[l][1]-para_vec_list[l][0]
            extent = [para_vec_list[k][0],para_vec_list[k][-1],para_vec_list[l][0],para_vec_list[l][-1]]
            # tmppost[np.where(cum_posterior<1-0.6827)] = np.nan
            plt.imshow(np.log10(tmppost),origin="lower",cmap="gray",extent=extent,aspect=float(para_vec_list[k][-1]-para_vec_list[k][0])/float(para_vec_list[l][-1]-para_vec_list[l][0]))
            # plt.figure(4)
            plt.contour(para_vec_list[k],para_vec_list[l],cum_posterior,levels=[1-0.9973,1-0.9545,1-0.6827],linestyles=[":","--","-"],colors=color)
            # plt.show()
            # plt.xlim([para_vec_list[k][0],para_vec_list[k][-1]])
            # plt.ylim([para_vec_list[l][0],para_vec_list[l][-1]])
            if k!=0:
                ax.yaxis.set_ticks([])
            else:
                plt.yticks(xticks_list[l])
                plt.ylabel(xlabel_list[l], fontsize=fontsize)
            if k==0 and l==Nparas-1:
                plt.yticks(xticks_list[l],["5e5","1e6","4e6"])
            if l!=Nparas-1:
                ax.xaxis.set_ticks([])
            else:
                ax.xaxis.set_ticks(xticks_list[k])
                plt.xlabel(xlabel_list[k], fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)
            ax.tick_params(axis='x', labelsize=fontsize)

            f.subplots_adjust(wspace=0,hspace=0)
    if 1:
        print("Saving "+os.path.join(out_pngs,planet,myoutfilename.replace("measurements","corner").replace(".pdf",".png")))
        plt.savefig(os.path.join(out_pngs,planet,myoutfilename.replace("measurements","corner")),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,planet,myoutfilename.replace("measurements","corner").replace(".pdf",".png")),bbox_inches='tight')

    plt.show()


