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
import ctypes
import pandas as pd
import matplotlib.pyplot as plt
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


def LPFvsHPF(myvec,cutoff,nansmooth=10):
    myvec_cp = np.zeros(myvec.shape)
    myvec_cp[:] = copy(myvec[:])
    wherenans = np.where(np.isnan(myvec_cp))
    myvec_cp = np.array(pd.DataFrame(myvec_cp).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:,0]
    # for k in wherenans[0]:
    #     myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-nansmooth]):np.min([np.size(myvec_cp),k+nansmooth])])

    fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec

    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan
    return LPF_myvec,HPF_myvec

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

    print(len(sys.argv))
    if len(sys.argv) == 1:
        out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
        osiris_data_dir = "/data/osiris_data/"
        IFSfilter = "Kbb"
        planet = "HR_8799_d"
        fit_folder = os.path.join(osiris_data_dir,"low_res",planet,"fit_test")
        gridname = os.path.join(osiris_data_dir,"hr8799b_modelgrid")
        # gridname = os.path.join(osiris_data_dir,"clouds_modelgrid")
        numthreads=32
        R=4000
        resnumbasis = 10
        cutoff = 40
    else:
        pass

    c_kms = 299792.458
    filename = os.path.join(out_pngs,planet+"_spec_"+"wvs"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
    with pyfits.open(filename) as hdulist:
        spectra_wvs = hdulist[0].data
    filename = os.path.join(out_pngs,planet+"_spec"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
    with pyfits.open(filename) as hdulist:
        spectra = hdulist[0].data
    # filename = os.path.join(out_pngs,planet+"_spec_"+"model"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
    # with pyfits.open(filename) as hdulist:
    #     final_model_arr = hdulist[0].data
    filename = os.path.join(out_pngs,planet+"_specstd"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
    with pyfits.open(filename) as hdulist:
        spectra_err = hdulist[0].data
    filename = os.path.join(out_pngs,planet+"_spec_"+"trans"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
    with pyfits.open(filename) as hdulist:
        trans = hdulist[0].data
    spectra_R = np.zeros(spectra.shape)+R

    if "hr8799b_modelgrid" in gridname:
        tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
        hdulist = pyfits.open(tmpfilename)
        planet_model_grid =  hdulist[0].data
        oriplanet_spec_wvs =  hdulist[1].data
        Tlistunique =  hdulist[2].data
        logglistunique =  hdulist[3].data
        paralistunique =  hdulist[4].data
        hdulist.close()
        from scipy.interpolate import RegularGridInterpolator
        myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,paralistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)
    if "clouds_modelgrid" in gridname:
        tmpfilename = os.path.join(osiris_data_dir,"clouds_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
        hdulist = pyfits.open(tmpfilename)
        planet_model_grid =  hdulist[0].data
        oriplanet_spec_wvs =  hdulist[1].data
        Tlistunique =  hdulist[2].data
        logglistunique =  hdulist[3].data
        paralistunique =  hdulist[4].data
        hdulist.close()
        from scipy.interpolate import RegularGridInterpolator
        myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,paralistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

    if planet == "HR_8799_b":
        plrv = -9.
    if planet == "HR_8799_c":
        plrv = -11.1
    if planet == "HR_8799_d":
        plrv = -15.7

    if "clouds_modelgrid" in gridname:
        fitT_list = np.linspace(800,1300,25,endpoint=True)
        fitlogg_list = np.linspace(3.,5.,15,endpoint=True)
        fitpara_list = np.linspace(5e5,4e6,10,endpoint=True)

    else:
        fitT_list = np.linspace(800,1200,21,endpoint=True)
        fitlogg_list = np.linspace(3,4.5,46,endpoint=True)
        fitpara_list = np.linspace(0.45708819,0.89125094,80,endpoint=True)
        # fitT_list = np.linspace(800,1200,8,endpoint=True)
        # fitlogg_list = np.linspace(3,4.5,12,endpoint=True)
        # fitpara_list = np.linspace(0.45708819,0.89125094,20,endpoint=True)

    variances = spectra_err**2

    logpost = np.zeros((len(fitT_list),len(fitlogg_list),len(fitpara_list)))
    ampl = np.zeros((len(fitT_list),len(fitlogg_list),len(fitpara_list)))
    ampl_err = np.zeros((len(fitT_list),len(fitlogg_list),len(fitpara_list)))
    for Tid,T in enumerate(fitT_list):
        for loggid,logg in enumerate(fitlogg_list):
            for paraid,para in enumerate(fitpara_list):

                planet_spec_func = interp1d(oriplanet_spec_wvs,myinterpgrid([T,logg,para])[0],bounds_error=False,fill_value=np.nan)
                model = LPFvsHPF(planet_spec_func(spectra_wvs*(1-(plrv)/c_kms))*trans,cutoff=cutoff)[1]

                wherefinite = np.where(np.isfinite(model)*np.isfinite(spectra))

                _model = model[wherefinite]
                _data = spectra[wherefinite]
                _variances = variances[wherefinite]
                logdet_Sigma = np.sum(2*np.log(spectra_err[wherefinite]))
                slogdet_icovphi0 = np.log(1/np.sum(_model**2/_variances))
                Npixs_HPFdata = np.size(_data)


                ampl[Tid,loggid,paraid] = np.sum(_data*_model/_variances)/np.sum(_model**2/_variances)
                ampl_err[Tid,loggid,paraid]= 1/np.sqrt(np.sum(_model**2/_variances))

                chi2 = np.sum((_data-ampl[Tid,loggid,paraid]*_model)**2/_variances)
                # print(chi2)

                logpost[Tid,loggid,paraid] = -0.5*logdet_Sigma-0.5*slogdet_icovphi0- (Npixs_HPFdata-1+2-1)/(2)*np.log(chi2)

                # plt.plot(spectra_wvs,spectra)
                # plt.scatter(spectra_wvs[wherefinite],_model*ampl[Tid,loggid,paraid])
                # # print(len(spectra_wvs),len(spectra),xerrors.shape,spectra_err.shape)
                # # plt.errorbar(spectra_wvs,spectra,xerr=xerrors,yerr=spectra_err,color="red")
                # # plt.errorbar(microns,fluxes,yerr=errors,color="blue")
                #
                # plt.legend()
                # plt.show()
                # exit()

    post = np.exp(logpost -np.nanmax(logpost))
    post[np.where(np.isnan(post))] = 0

    fontsize = 15
    para_vec_list = [fitT_list,fitlogg_list,fitpara_list]
    if "hr8799b_modelgrid" in gridname:
        xlabel_list = ["T [K]", "log(g/[1 cm/$\mathrm{s}^2$])","C/O"]
        xticks_list = [[800,1000,1200], [3.5,4.0,4.5],[0.5,0.7,0.9]]
    if "clouds_modelgrid" in gridname:
        xlabel_list = ["T [K]", "log(g/[1 cm/$\mathrm{s}^2$])","pgs"]
        xticks_list = [[800,1000,1200], [3.5,4.0,4.5],[5e5,1e6,4e6]]
    if "HR_8799_b" in fit_folder:
        planet,color = "HR_8799_b","#0099cc"
    if "HR_8799_c" in fit_folder:
        planet,color = "HR_8799_c","#ff9900"
    if "HR_8799_d" in fit_folder:
        planet,color = "HR_8799_d","#6600ff"
    Nparas = len(para_vec_list)

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=post))
    hdulist.append(pyfits.ImageHDU(data=fitT_list))
    hdulist.append(pyfits.ImageHDU(data=fitlogg_list))
    hdulist.append(pyfits.ImageHDU(data=fitpara_list))
    myoutfilename = planet+"_"+os.path.basename(gridname)+"_fit_OSIRISspec"+"_"+IFSfilter+".pdf"
    try:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_posterior.fits")), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(out_pngs,planet,myoutfilename.replace(".pdf","_posterior.fits")), clobber=True)
    hdulist.close()

    myargmax = np.unravel_index(np.nanargmax(post),post.shape)
    print(myargmax)
    print(fitT_list[myargmax[0]],fitlogg_list[myargmax[1]],fitpara_list[myargmax[2]])


    plt.figure(1,figsize=(35,5))
    # plt.errorbar(spectra_wvs, spectra, yerr = spectra_err, color = color, capsize=5,
    #     elinewidth=1, markeredgewidth=1, label = 'spectra')
    plt.fill_between(spectra_wvs, spectra-spectra_err,spectra+spectra_err, color = color, label = 'spectra',alpha=0.5)
    plt.plot(spectra_wvs, spectra, color = color)

    planet_spec_func = interp1d(oriplanet_spec_wvs, myinterpgrid([fitT_list[myargmax[0]],fitlogg_list[myargmax[1]],fitpara_list[myargmax[2]]])[0],bounds_error=False,fill_value=np.nan)
    model = LPFvsHPF(planet_spec_func(spectra_wvs*(1-(plrv)/c_kms))*trans,cutoff=cutoff)[1]
    plt.plot(spectra_wvs,model*ampl[myargmax[0],myargmax[1],myargmax[2]],label="model",color="black")
    plt.legend()
    # plt.show()


    if 1:
        print("Saving "+os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_{0}_osiris_extspec.pdf".format(IFSfilter)))
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_{0}_osiris_extspec.pdf".format(IFSfilter)),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_{0}_osiris_extspec.png".format(IFSfilter)),bbox_inches='tight')



    f = plt.figure(2,figsize=(6,6))
    for k in range(Nparas):
        dims = np.arange(Nparas).tolist()
        dims.pop(k)

        plt.subplot(Nparas,Nparas,k+1+k*Nparas)
        ax = plt.gca()
        tmppost = np.sum(post,axis=(*dims,))
        tmppost /= np.max(tmppost)
        # plt.plot(para_vec_list[k],tmppost,color=color)
        plt.fill_between(para_vec_list[k],tmppost*0,tmppost,color=color)
        plt.xlabel(xlabel_list[k], fontsize=fontsize)
        if k != 0:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        if k != Nparas-1:
            # ax.xaxis.tick_top()
            # ax.xaxis.set_label_position("top")
            ax.yaxis.set_ticks([0.5,1.0])
        else:
            ax.yaxis.set_ticks([0.0,0.5,1.0])
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        plt.ylim([0.0,1.1])
        plt.xlim([para_vec_list[k][0],para_vec_list[k][-1]+1e-2])
        if k != Nparas-1 or "clouds" not in gridname:
            ax.yaxis.set_ticks([0.5,1.0])
            ax.xaxis.set_ticks(xticks_list[k])
        else:
            ax.yaxis.set_ticks([0.0,0.5,1.0])
            plt.xticks(xticks_list[k],["5e5","1e6","4e6"])
            # ax.spines['right'].set_visible(False)
            # ax.spines['top'].set_visible(False)
            # ax.xaxis.set_ticks_position('bottom')
            # ax.yaxis.set_ticks_position('left')
        # plt.show()


        for l in np.arange(k+1,Nparas):
            plt.subplot(Nparas,Nparas,k+1+(l)*Nparas)
            ax = plt.gca()
            dims = np.arange(Nparas).tolist()
            dims.pop(k)
            dims.pop(l-1)
            tmppost = np.sum(post,axis=(*dims,))
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
            if k == 0 and l == Nparas-1:
                a = -8
            else:
                a=0
            plt.imshow(np.log10(tmppost),origin="lower",cmap="gray",extent=extent,aspect=float(a+para_vec_list[k][-1]-para_vec_list[k][0])/float(para_vec_list[l][-1]-para_vec_list[l][0]))
            plt.contour(para_vec_list[k],para_vec_list[l],cum_posterior,levels=[1-0.9973,1-0.9545,1-0.6827],linestyles=[":","--","-"],colors=color)
            # plt.xlim([para_vec_list[k][0],para_vec_list[k][-1]])
            # plt.ylim([para_vec_list[l][0],para_vec_list[l][-1]])
            if k!=0:
                ax.yaxis.set_ticks([])
            else:
                ax.yaxis.set_ticks(xticks_list[l])
                plt.ylabel(xlabel_list[l], fontsize=fontsize)
            if "clouds" in gridname and (k==0 and l==Nparas-1):
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
        print("Saving "+os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_{0}_corner_osiris_extspec.pdf".format(IFSfilter)))
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_{0}_corner_osiris_extspec.pdf").format(IFSfilter),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_{0}_corner_osiris_extspec.png").format(IFSfilter),bbox_inches='tight')
    plt.show()

