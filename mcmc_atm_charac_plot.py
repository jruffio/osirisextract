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
    # plt.figure(10)
    # plt.plot(x,posterior)
    # plt.plot(x,cum_posterior)
    # plt.show()
    if len(x[0:np.min([argmax_post+1,len(x)])]) < 2:
        lx = x[0]
    else:
        lf = interp1d(cum_posterior[0:np.min([argmax_post+1,len(x)])],
                      x[0:np.min([argmax_post+1,len(x)])],bounds_error=False,fill_value=x[0])
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx=x[-1]
    else:
        rf = interp1d(cum_posterior[argmax_post::],x[argmax_post::],bounds_error=False,fill_value=x[-1])
        rx = rf(1-0.6827)
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



    # planet = "HR_8799_b"
    # date = "090722"
    # date = "090730"
    # date = "090903"
    # date = "100711"
    # date = "100712"
    # date = "100713"
    # date = "130725"
    # date = "130726"
    # date = "130727"
    # date = "161106"
    # date = "180722"
    planet = "HR_8799_c"
    # date = "100715"
    # date = "101028"
    # date = "101104"
    # date = "110723"
    # date = "110724"
    # date = "110725"
    # date = "130726"
    # date = "171103"
    # planet = "HR_8799_d"
    # date = "150720"
    # date = "150722"
    # date = "150723"
    # date = "150828"
    # planet = "51_Eri_b"
    # date = "171103"
    # date = "171104"
    # planet = "kap_And"
    # date = "161106"
    IFSfilter = "Kbb"
    scale = "*"
    date = "*"
    inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/"
    filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
    filelist.sort()
    outputfolder = "20200309_model"
    gridname = os.path.join("/data/osiris_data/","hr8799b_modelgrid")
    myoutfilename = "CtoO_"+planet+"_measurements.pdf"

    c_kms = 299792.458
    cutoff = 40
    numthreads = 30
    N_kl = 10
    R= 4000
    fontsize=12

    tmpfilename = os.path.join("/data/osiris_data/hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
    hdulist = pyfits.open(tmpfilename)
    planet_model_grid =  hdulist[0].data
    oriplanet_spec_wvs =  hdulist[1].data
    Tlistunique =  hdulist[2].data
    logglistunique =  hdulist[3].data
    CtoOlistunique =  hdulist[4].data
    # Tlistunique =  hdulist[1].data
    # logglistunique =  hdulist[2].data
    # CtoOlistunique =  hdulist[3].data
    hdulist.close()

    print(planet_model_grid.shape,np.size(Tlistunique),np.size(logglistunique),np.size(CtoOlistunique),np.size(oriplanet_spec_wvs))
    from scipy.interpolate import RegularGridInterpolator
    myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

    CtoO_CI_list = []
    CtoO_post_list = []
    date_list = []
    for file_id, filename in enumerate(filelist):
        tmpfilename = os.path.join(os.path.dirname(filename),outputfolder,os.path.basename(filename).replace(".fits","_logpost.fits"))
        if len(glob.glob(tmpfilename))!=1:
            print("No data on "+filename)
            continue
        date = os.path.basename(filename).split("_a")[0].split("s")[1]
        date_list.append(date)
        tmpfilename = os.path.join(tmpfilename)
        hdulist = pyfits.open(tmpfilename)
        logpost =  hdulist[0].data
        fitT_list =  hdulist[1].data
        fitlogg_list =  hdulist[2].data
        fitCtoO_list =  hdulist[3].data
        planetRV_array0 =  hdulist[4].data
        hdulist.close()

        post = np.exp(logpost-np.nanmax(logpost))

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
        # plt.plot(fitCtoO_list,CtoO_post)
        # plt.show()

        CtoO_post = np.nansum(post[-1,-2,:,:],axis=1)
        CtoO_post /= np.nanmax(CtoO_post)
        leftCI,argmaxpost,rightCI,_ = get_err_from_posterior(fitCtoO_list,CtoO_post)
        CtoO_CI_list.append([leftCI,argmaxpost,rightCI])
        CtoO_post_list.append(CtoO_post)
        # print(fitCtoO_list)
        print(leftCI,argmaxpost,rightCI)

        # plt.figure(4)
        # rv_post = np.nansum(post,axis=(0,1,2))
        # rv_post /= np.nanmax(rv_post)
        # plt.plot(planetRV_array0,rv_post)
    date_list = np.array(date_list)

    yval = [argmaxpost for leftCI,argmaxpost,rightCI in CtoO_CI_list]
    m_yerr = [argmaxpost-leftCI for leftCI,argmaxpost,rightCI in CtoO_CI_list]
    p_yerr = [rightCI - argmaxpost for leftCI,argmaxpost,rightCI in CtoO_CI_list]
    # plt.errorbar(np.arange(len(CtoO_CI_list)),yval,yerr=[m_yerr,p_yerr ])

    plt.figure(1,figsize=(12,4))
    plt.errorbar(np.arange(len(CtoO_CI_list)),yval,yerr=[m_yerr,p_yerr ],fmt="none",color="#cc6600")
    plt.plot(np.arange(len(CtoO_CI_list)),yval,"x",color="#ff9900",label="K-band")

    print(fitCtoO_list)
    plt.ylim([fitCtoO_list[0],fitCtoO_list[-1]])
    plt.xlim([0,len(CtoO_CI_list)+1])
    unique_dates = np.unique(date_list)
    for date in unique_dates:
        where_data = np.where(date_list==date)
        first = where_data[0][0]
        last = where_data[0][-1]
        print(date,first,last)
        plt.annotate("",xy=(first,fitCtoO_list[0]),xytext=(last+0.001,fitCtoO_list[0]),xycoords="data",arrowprops={'arrowstyle':"|-|",'shrinkA':0.1,'shrinkB':0.1})
        plt.gca().text((first+last)/2-0.5,fitCtoO_list[0]+0.01,date[2:4]+"-"+date[4:6]+"-20"+date[0:2],ha="left",va="bottom",rotation=+60,size=fontsize)
    plt.gca().text(len(CtoO_CI_list),1,planet,ha="right",va="top",rotation=0,size=fontsize*1.5)
    plt.ylabel("C/O",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_position(("data",0))
    plt.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis="y",labelsize=fontsize)
    # plt.legend(loc="upper left",bbox_to_anchor=[0,1.0],frameon=True,fontsize=fontsize)#
    # plt.tight_layout()




    nightly_CtoO_CI_list = []
    nightly_CtoO_post_list = []
    nightly_Nexp = []
    for date in unique_dates:
        where_data = np.where(date_list==date)
        nightly_Nexp.append(len(where_data[0]))
        print(np.array(CtoO_post_list).shape)
        tonight_CtoOpost = np.prod(np.array(CtoO_post_list)[where_data[0],:],axis=0)
        nightly_CtoO_post_list.append(tonight_CtoOpost)
        leftCI,argmaxpost,rightCI,_ = get_err_from_posterior(fitCtoO_list,tonight_CtoOpost)
        nightly_CtoO_CI_list.append([leftCI,argmaxpost,rightCI])


    nightly_yval = [argmaxpost for leftCI,argmaxpost,rightCI in nightly_CtoO_CI_list]
    nightly_m_yerr = [argmaxpost-leftCI for leftCI,argmaxpost,rightCI in nightly_CtoO_CI_list]
    nightly_p_yerr = [rightCI - argmaxpost for leftCI,argmaxpost,rightCI in nightly_CtoO_CI_list]
    print(["20"+date for date in  unique_dates])
    print(nightly_yval)
    print(nightly_m_yerr)
    print(nightly_p_yerr)
    print(nightly_Nexp)

    final_CtoO_post = np.prod(np.array(CtoO_post_list),axis=0)
    final_leftCI,final_yval,final_rightCI,_ = get_err_from_posterior(fitCtoO_list,final_CtoO_post)
    final_m_yerr = final_yval - final_leftCI
    final_p_yerr = final_rightCI - final_yval
    print(final_yval,final_m_yerr,final_p_yerr)
    # exit()

    plt.figure(2,figsize=(12,12))
    plt.subplot2grid((4,1),(0,0),rowspan=2)
    plt.fill_betweenx([0,np.size(unique_dates)],final_leftCI,final_rightCI,alpha=0.2,color="#cc6600")

    # print("bonjour",nightly_yval,epochs_rverr_c,epochs_c,epochs_Nexp_c)
    # print("Planet & Date & RV & N cubes \\\\")
    # for a,b,c,d in zip(epochs_rv_c,epochs_rverr_c,epochs_c,epochs_Nexp_c):
    #     if np.isnan(a):
    #         continue
    #     formated_date =  c[0:4]+"-"+c[4:6]+"-"+c[6:8]
    #     print("& {0} & ${1:.1f} \\pm {2:.1f}$ & {3} \\\\".format(formated_date,a,b,int(d)))
    plt.errorbar(nightly_yval,np.arange(np.size(unique_dates)),xerr=(nightly_m_yerr,nightly_p_yerr),fmt="none",color="#ff9900")
    plt.plot(nightly_yval,np.arange(np.size(unique_dates)),"x",color="#ff9900",label="c (# exposures)")
    wherenotnans = np.where(np.isfinite(nightly_yval))
    for y,(x,date,num) in enumerate(zip(nightly_yval,unique_dates,nightly_Nexp)):
        if np.isfinite(x):
            plt.gca().text(x,y,"{0}".format(int(num)),ha="center",va="bottom",rotation=0,size=fontsize,color="#cc3300",alpha=1)
    # plt.plot([rv_c[2]-rverr_c[2],rv_c[2]-rverr_c[2]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#cc6600",alpha=0.4)
    plt.plot([final_yval,final_yval],[0,np.size(unique_dates)],linestyle="-",linewidth=2,color="#cc3300",alpha=0.4)
    # plt.plot([rv_c[2]+rverr_c[2],rv_c[2]+rverr_c[2]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#cc6600",alpha=0.4)

    print([fitCtoO_list[0],fitCtoO_list[-1]])
    plt.xlim([fitCtoO_list[0],fitCtoO_list[-1]])
    plt.xlabel("C/O",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.yticks(np.arange(0,np.size(unique_dates)),[date[2:4]+"-"+date[4:6]+"-20"+date[0:2] for date in unique_dates ])
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",-20))
    plt.legend(loc="center right",frameon=True,fontsize=fontsize)#
    plt.show()

        # c_combined_avg_post,c_combined_sig_post,_ = get_err_from_posterior(final_planetRV_hd,c_optimistic_posterior)

    # print(logpost.shape)
    # import matplotlib.pyplot as plt
    # for temp_id, temp in enumerate(fitT_list):
    #     for fitlogg_id,fitlogg in enumerate(fitlogg_list):
    #         for CtoO_id,CtoO in enumerate(fitCtoO_list):
    #             plt.plot(planetRV_array0,np.exp(logpost[temp_id,fitlogg_id,CtoO_id,:]-np.max(logpost)))
    # plt.show()
