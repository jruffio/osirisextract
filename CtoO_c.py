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
    priorTeff,priorTeff_sig = 1000,1#1e-9
    priorlogg,priorlogg_sig = -3.5,10#1e-9
    # planet = "HR_8799_c"
    # priorTeff,priorTeff_sig = 1100,1e-3
    # priorlogg,priorlogg_sig = -3.5,1e-3
    planet = "HR_8799_b"
    IFSfilter = "Kbb"
    scale = "*"
    date = "*"
    inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/"
    filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
    filelist.sort()
    outputfolder = "20200309_model"
    gridname = os.path.join("/data/osiris_data/","hr8799b_modelgrid")
    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
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
    post_list = []
    for file_id, filename in enumerate(filelist):
        tmpfilename = os.path.join(os.path.dirname(filename),outputfolder,os.path.basename(filename).replace(".fits","_logpost.fits"))
        if len(glob.glob(tmpfilename))!=1:
            print("No data on "+filename)
            continue
        else:
            print(filename)
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
        # print(np.nanmax(logpost))
        # exit()
        logpriorTeff_vec = -0.5/priorTeff_sig**2*(fitT_list-priorTeff)**2
        logpriorlogg_vec = -0.5/priorlogg_sig**2*(fitlogg_list-priorlogg)**2
        try:
            combined_logpost += logpost
        except:
            combined_logpost = logpost
        logpost = logpost + logpriorTeff_vec[:,None,None,None] + logpriorlogg_vec[:,None,None,None]
        post = np.exp(logpost-np.nanmax(logpost))
        # logpriorTeff_vec = 1/(np.sqrt(2*np.pi)*priorTeff_sig)*np.exp(-0.5/priorTeff_sig**2*(fitT_list-priorTeff)**2)
        # logpriorlogg_vec = 1/(np.sqrt(2*np.pi)*priorlogg_sig)*np.exp(-0.5/priorlogg_sig**2*(fitlogg_list-priorlogg)**2)

        post_list.append(np.nansum(post,axis=3))
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

        # CtoO_post = np.nansum(post[-1,-2,:,:],axis=1)
        CtoO_post = np.nansum(post,axis=(0,1,3))
        CtoO_post /= np.nanmax(CtoO_post)
        leftCI,argmaxpost,rightCI,_ = get_err_from_posterior(fitCtoO_list,CtoO_post)
        CtoO_CI_list.append([leftCI,argmaxpost,rightCI])
        CtoO_post_list.append(CtoO_post)
        # print(fitCtoO_list)
        # print(leftCI,argmaxpost,rightCI)

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
    plt.errorbar(np.arange(len(CtoO_CI_list)),yval,yerr=[m_yerr,p_yerr ],fmt="none",color="#ff9900")
    plt.plot(np.arange(len(CtoO_CI_list)),yval,"x",color="#ff9900",label="K-band") # #0099cc  #ff9900

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
        plt.gca().text((first+last)/2-0.5,fitCtoO_list[0]-0.01,date[2:4]+"-"+date[4:6]+"-20"+date[0:2],ha="right",va="top",rotation=-60,size=fontsize)
    plt.gca().text(len(CtoO_CI_list),fitCtoO_list[-1],planet.replace("_"," "),ha="left",va="top",rotation=0,size=fontsize*1.5)
    plt.ylabel("C/O",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(False)#set_position(("data",0))
    plt.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)
    plt.tick_params(axis="y",labelsize=fontsize)
    # plt.legend(loc="upper left",bbox_to_anchor=[0,1.0],frameon=True,fontsize=fontsize)#
    # plt.tight_layout()

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
        print(np.array(CtoO_post_list).shape)
        tonight_CtoOpost = np.prod(np.array(CtoO_post_list)[where_data[0],:],axis=0)
        nightly_CtoO_post_list.append(tonight_CtoOpost)
        leftCI,argmaxpost,rightCI,_ = get_err_from_posterior(fitCtoO_list,tonight_CtoOpost)
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

    final_CtoO_post = np.prod(np.array(CtoO_post_list),axis=0)
    final_CtoO_post /= np.max(final_CtoO_post)
    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=np.concatenate([fitCtoO_list[None,:],final_CtoO_post[None,:]],axis=0)))
    try:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_CtoOposterior.fits")), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_CtoOposterior.fits")), clobber=True)
    hdulist.close()
    final_post = np.prod(np.array(post_list),axis=0)
    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=final_post))
    hdulist.append(pyfits.PrimaryHDU(data=fitT_list))
    hdulist.append(pyfits.PrimaryHDU(data=fitlogg_list))
    hdulist.append(pyfits.PrimaryHDU(data=fitCtoO_list))
    try:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_posterior.fits")), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_posterior.fits")), clobber=True)
    hdulist.close()




    final_leftCI,final_yval,final_rightCI,_ = get_err_from_posterior(fitCtoO_list,final_CtoO_post)
    final_m_yerr = final_yval - final_leftCI
    final_p_yerr = final_rightCI - final_yval
    print("final_yval,final_m_yerr,final_p_yerr")
    print(final_yval,final_m_yerr,final_p_yerr)
    # exit()

    combined_logpost = combined_logpost + logpriorTeff_vec[:,None,None,None] + logpriorlogg_vec[:,None,None,None]
    combined_post = np.exp(combined_logpost-np.nanmax(combined_logpost))
    combined_post_rvmargi = np.nansum(combined_post,axis=3)
    plt.figure(2,figsize=(12,12))
    para_vec_list = [fitT_list,fitlogg_list,fitCtoO_list]
    xlabel_list = ["T (K)", "log(g) (cgs?)","C/O"]
    Nparas = len(para_vec_list)
    for k in range(Nparas):
        dims = np.arange(Nparas).tolist()
        dims.pop(k)

        plt.subplot(Nparas,Nparas,k+1+k*Nparas)
        tmppost = np.sum(combined_post_rvmargi,axis=(*dims,))
        tmppost /= np.max(tmppost)
        plt.plot(para_vec_list[k],tmppost)
        plt.xlabel(xlabel_list[k])

        for l in np.arange(k+1,Nparas):
            plt.subplot(Nparas,Nparas,k+1+(l)*Nparas)
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
            extent = [para_vec_list[k][0],para_vec_list[k][-1],para_vec_list[l][0],para_vec_list[l][-1]]
            tmppost[np.where(cum_posterior<1-0.6827)] = np.nan
            plt.imshow(tmppost,origin="lower",extent=extent,aspect=(para_vec_list[k][-1]-para_vec_list[k][0])/(para_vec_list[l][-1]-para_vec_list[l][0]))
            plt.contour(para_vec_list[k],para_vec_list[l],cum_posterior,levels=[1-0.9973,1-0.9545,1-0.6827],colors="black")
            plt.xlabel(xlabel_list[k])
            plt.ylabel(xlabel_list[l])



    if 1:
        print("Saving "+os.path.join(out_pngs,planet,"corner_"+planet+".pdf"))
        plt.savefig(os.path.join(out_pngs,planet,"corner_"+planet+".pdf"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,planet,"corner_"+planet+".png"),bbox_inches='tight')

    plt.show()


