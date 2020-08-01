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
        fit_folder = os.path.join(osiris_data_dir,"low_res","HR_8799_d","fit_test")
        gridname = os.path.join(osiris_data_dir,"hr8799b_modelgrid")
        # gridname = os.path.join("/data/osiris_data/","clouds_modelgrid")
        numthreads=32
    else:
        pass

    fluxes = []
    errors = []
    microns = []
    filters = []
    xerrors = []
    spectra = []
    spectra_err = []
    spectra_wvs = []
    spectra_R = []
    for data_filename in glob.glob(os.path.join(fit_folder,"*.txt")):
        print(data_filename)
        if "_R" in os.path.basename(data_filename):
            print("coucou")
            R = float(os.path.basename(data_filename).split("_R=")[-1].replace(".txt",""))
            t = np.genfromtxt(data_filename)
            spectra.extend(t[:,1])
            spectra_err.extend(t[:,2])
            spectra_wvs.extend(t[:,0])
            spectra_R.extend([R,]*np.size(t[:,0]))
            # out = os.path.join(os.path.dirname(data_filename),os.path.basename(data_filename).split("_R=")[0]+"_skip"+"_R={0}.txt".format(R))
            # print(out)
            # np.savetxt(out,np.concatenate([t[2::4,0][:,None],t[2::4,1][:,None],t[2::4,2][:,None]],axis=1),delimiter=" ")
        else:
            t = np.loadtxt(data_filename,dtype=np.str)
            if np.size(t.shape) == 1:
                t = t[None,:]
            fluxes.extend(t[:,0].astype(np.float))
            errors.extend(t[:,1].astype(np.float))
            microns.extend(t[:,2].astype(np.float))
            filters.extend(t[:,3])
            xerrors.extend(get_bands_xerr([os.path.join(osiris_data_dir,"filters",filter) for filter in t[:,3]], t[:,2].astype(np.float)).T)
    # exit()
    # print(xerrors)
    # exit()
    N_phot = len(spectra)
    xerrors = np.array(xerrors).T
    spectra = np.array(spectra)
    spectra_err = np.array(spectra_err)
    spectra_wvs = np.array(spectra_wvs)
    spectra_R = np.array(spectra_R)

    print(spectra.shape,spectra_err.shape,spectra_wvs.shape,spectra_R.shape)
    print(len(microns),microns)
    print(len(fluxes),fluxes)
    print(len(errors),errors)
    print(xerrors.shape)
    # exit()

    # # plt.fill_between(spectra_wvs, spectra-spectra_err,spectra+spectra_err, color = '#ff9900', alpha=0.5)
    # plt.show()

    # tmpfilename = os.path.join(gridname,"hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
    # hdulist = pyfits.open(tmpfilename)
    # planet_model_grid =  hdulist[0].data
    # oriplanet_spec_wvs =  hdulist[1].data
    # Tlistunique =  hdulist[2].data
    # logglistunique =  hdulist[3].data
    # paralistunique =  hdulist[4].data
    # # Tlistunique =  hdulist[1].data
    # # logglistunique =  hdulist[2].data
    # # paralistunique =  hdulist[3].data
    # hdulist.close()
    #
    # print(planet_model_grid.shape,np.size(Tlistunique),np.size(logglistunique),np.size(paralistunique),np.size(oriplanet_spec_wvs))
    # from scipy.interpolate import RegularGridInterpolator
    # myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,paralistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

    if "hr8799b_modelgrid" in gridname:
        planet_model_list = []
        grid_filelist = glob.glob(os.path.join(gridname,"lte*-*-0.0.aces_hr8799b_pgs=4d6_Kzz=1d8_C=*_O=*_gs=5um.exoCH4_hiresHK.7.D2e.sorted"))
        # grid_filelist = glob.glob(os.path.join(gridname,"lte12-4.5-0.0.aces_hr8799b_pgs=4d6_Kzz=1d8_C=8.38_O=*_gs=5um.exoCH4_hiresHK.7.D2e.sorted"))
        grid_filelist.sort()

        Tlist = np.array([int(float(os.path.basename(grid_filename).split("lte")[-1].split("-")[0])*100) for grid_filename in grid_filelist])
        logglist = np.array([float(os.path.basename(grid_filename).split("-")[1]) for grid_filename in grid_filelist])
        Clist = np.array([float(os.path.basename(grid_filename).split("C=")[-1].split("_O")[0]) for grid_filename in grid_filelist])
        Olist = np.array([float(os.path.basename(grid_filename).split("O=")[-1].split("_gs")[0]) for grid_filename in grid_filelist])
        paralist = 10**(Clist-Olist)
        Tlistunique = np.unique(Tlist)
        logglistunique = np.unique(logglist)
        paralistunique = np.unique(paralist)
    if "clouds_modelgrid" in gridname:
        planet_model_list = []
        #lte0800-3.0-0.0.aces_pgs=4d6_Kzz=1d8_gs=1um_4osiris.7
        grid_filelist = glob.glob(os.path.join(gridname,"lte*-*-0.0.aces_pgs=*_Kzz=1d8_gs=1um_4osiris.7.D2e.sorted"))
        # grid_filelist = glob.glob(os.path.join(gridname,"lte1200-4.5-0.0.aces_pgs=*_Kzz=1d8_gs=1um_4osiris.7"))
        grid_filelist.sort()

        Tlist = np.array([int(float(os.path.basename(grid_filename).split("lte")[-1].split("-")[0])) for grid_filename in grid_filelist])
        logglist = np.array([float(os.path.basename(grid_filename).split("-")[1]) for grid_filename in grid_filelist])
        paralist = np.array([float(os.path.basename(grid_filename).split("pgs=")[-1].split("_Kzz")[0].replace("d","e")) for grid_filename in grid_filelist])
        Tlistunique = np.unique(Tlist)
        logglistunique = np.unique(logglist)
        paralistunique = np.unique(paralist)
    print(Tlistunique,len(Tlistunique))
    print(logglistunique,len(logglistunique))
    print(paralistunique,len(paralistunique))
    print(len(Tlist),len(Tlistunique)*len(logglistunique)*len(paralistunique))
    print(os.path.basename(grid_filelist[0]))
    # exit()
    outgridname = os.path.join(fit_folder,os.path.basename(gridname)+".fits")
    print(outgridname)
    # exit()

    if len(glob.glob(outgridname)) == 0:
        #print(gridname)
        specpool = mp.Pool(processes=numthreads)
        for file_id,grid_filename in enumerate(grid_filelist):
            # if os.path.basename(grid_filename) == "lte0800-3.0-0.0.aces_pgs=1d6_Kzz=1d8_gs=1um_4osiris.7":
            #     continue
            print(os.path.basename(grid_filename))

            with open(grid_filename, 'r') as csvfile:
                out = np.loadtxt(grid_filename,skiprows=0)
                # print(np.size(oriplanet_spec_wvs))
                oriplanet_spec_wvs = out[:,0]/1e4
                dwvs = oriplanet_spec_wvs[1::]-oriplanet_spec_wvs[0:np.size(oriplanet_spec_wvs)-1]
                dwvs = np.insert(dwvs,0,dwvs[0])
                oriplanet_spec = 10**(out[:,1]-np.max(out[:,1]))
                # oriplanet_spec = out[:,1]
                oriplanet_spec /= np.nanmean(oriplanet_spec)

            model = []
            for flux, err, wv,filt,xerr in zip(fluxes,errors,microns,filters,xerrors.T):
                # print(wv,wv-xerr[0],oriplanet_spec_wvs[0] , wv+xerr[1] ,oriplanet_spec_wvs[-1])
                if wv-xerr[0] < oriplanet_spec_wvs[0] or wv+xerr[1] > oriplanet_spec_wvs[-1]:
                    model.append(np.nan)
                    continue
                filter_arr = np.loadtxt(os.path.join(osiris_data_dir,"filters",filt))
                wvs = filter_arr[:,0]/1e4
                trans = filter_arr[:,1]
                trans_f = interp1d(wvs,trans,bounds_error=False,fill_value=0)
                model.append(np.sum(oriplanet_spec*trans_f(oriplanet_spec_wvs)*dwvs)/np.sum(trans_f(oriplanet_spec_wvs)*dwvs))
            for spec,spec_err, spec_wv,spec_R in zip(spectra,spectra_err,spectra_wvs,spectra_R):
                if spec_wv < oriplanet_spec_wvs[0] or spec_wv > oriplanet_spec_wvs[-1]:
                    model.append(np.nan)
                    continue
                trans_f = lambda x: np.exp(-0.5*(x-spec_wv)**2/(spec_wv/spec_R/2.634)**2)
                model.append(np.sum(oriplanet_spec*trans_f(oriplanet_spec_wvs)*dwvs)/np.sum(trans_f(oriplanet_spec_wvs)*dwvs))
            model = np.array(model)
            model_wvs = np.concatenate([microns,spectra_wvs])
            # print(model)
            # # plt.plot(oriplanet_spec_wvs,oriplanet_spec)
            # plt.scatter(model_wvs,model)
            # plt.show()
            # exit()
            # print(len(phot_list))
            # plt.scatter(np.concatenate([microns,spectra_wvs]),phot_list*1e-15,color="red")
            # plt.show()
            planet_model_list.append(model)

        planet_model_grid = np.zeros((np.size(Tlistunique),np.size(logglistunique),np.size(paralistunique),np.size(model_wvs)))
        for T_id,T in enumerate(Tlistunique):
            for logg_id,logg in enumerate(logglistunique):
                for pgs_id,pgs in enumerate(paralistunique):
                    planet_model_grid[T_id,logg_id,pgs_id,:] = planet_model_list[np.where((Tlist==T)*(logglist==logg)*(paralist==pgs))[0][0]]
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=planet_model_grid))
        hdulist.append(pyfits.ImageHDU(data=model_wvs))
        hdulist.append(pyfits.ImageHDU(data=Tlistunique))
        hdulist.append(pyfits.ImageHDU(data=logglistunique))
        hdulist.append(pyfits.ImageHDU(data=paralistunique))
        try:
            hdulist.writeto(outgridname, overwrite=True)
        except TypeError:
            hdulist.writeto(outgridname, clobber=True)
        # try:
        #     hdulist.writeto(os.path.join(gridname,"hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter)), overwrite=True)
        # except TypeError:
        #     hdulist.writeto(os.path.join(gridname,"hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter)), clobber=True)
        hdulist.close()
        exit()
    else:
        hdulist = pyfits.open(outgridname)
        planet_model_grid =  hdulist[0].data
        print(planet_model_grid.shape)
        model_wvs =  hdulist[1].data
        Tlistunique =  hdulist[2].data
        logglistunique =  hdulist[3].data
        paralistunique =  hdulist[4].data
        hdulist.close()
        myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,paralistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

    if "clouds_modelgrid" in gridname:
        fitT_list = np.linspace(800,1300,25,endpoint=True)
        fitlogg_list = np.linspace(3.,5.,15,endpoint=True)
        fitpara_list = np.linspace(5e5,4e6,10,endpoint=True)

    else:
        # fitT_list = np.linspace(800,1200,21,endpoint=True)
        # fitlogg_list = np.linspace(3,4.5,46,endpoint=True)
        # fitpara_list = np.linspace(0.45708819,0.89125094,80,endpoint=True)
        fitT_list = np.linspace(800,1200,8,endpoint=True)
        fitlogg_list = np.linspace(3,4.5,12,endpoint=True)
        fitpara_list = np.linspace(0.45708819,0.89125094,20,endpoint=True)

    print(np.size(fluxes),np.size(spectra))
    print(np.size(errors),np.size(spectra_err))

    data = np.concatenate([fluxes,spectra])
    sigmas = np.concatenate([errors,spectra_err])
    variances = sigmas**2

    logpost = np.zeros((len(fitT_list),len(fitlogg_list),len(fitpara_list)))
    ampl = np.zeros((len(fitT_list),len(fitlogg_list),len(fitpara_list)))
    ampl_err = np.zeros((len(fitT_list),len(fitlogg_list),len(fitpara_list)))
    for Tid,T in enumerate(fitT_list):
        for loggid,logg in enumerate(fitlogg_list):
            for paraid,para in enumerate(fitpara_list):

                model = myinterpgrid([T,logg,para])[0]

                wherefinite = np.where(np.isfinite(model))

                _model = model[wherefinite]
                _data = data[wherefinite]
                _variances = variances[wherefinite]


                ampl[Tid,loggid,paraid] = np.sum(_data*_model/_variances)/np.sum(_model**2/_variances)
                ampl_err[Tid,loggid,paraid]= 1/np.sqrt(np.sum(_model**2/_variances))

                chi2 = np.sum((_data-ampl[Tid,loggid,paraid]*_model)**2/_variances)
                # print(chi2)

                logpost[Tid,loggid,paraid] = -0.5*chi2

                # plt.scatter(model_wvs[wherefinite],_model*ampl[Tid,loggid,paraid])
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

    myargmax = np.unravel_index(np.nanargmax(post),post.shape)
    print(myargmax)
    print(fitT_list[myargmax[0]],fitlogg_list[myargmax[1]],fitpara_list[myargmax[2]])


    plt.figure(1)
    plt.errorbar(microns, fluxes, yerr = errors, xerr=xerrors, fmt='s', color = 'gray', capsize=5,
    elinewidth=1, markeredgewidth=1, label = 'photometry')
    plt.errorbar(spectra_wvs, spectra, yerr = spectra_err, color = color, capsize=5,
        elinewidth=1, markeredgewidth=1, label = 'spectra')
    model = myinterpgrid([fitT_list[myargmax[0]],fitlogg_list[myargmax[1]],fitpara_list[myargmax[2]]])[0]
    plt.scatter(model_wvs[wherefinite],model[wherefinite]*ampl[myargmax[0],myargmax[1],myargmax[2]],label="model",color="black")
    # plt.show()


    if 1:
        print("Saving "+os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_fit_sed.pdf"))
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_fit_sed.pdf"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_fit_sed.png"),bbox_inches='tight')



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
        print("Saving "+os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_corner_sed.pdf"))
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_corner_sed.pdf"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+os.path.basename(gridname)+"_corner_sed.png"),bbox_inches='tight')
    plt.show()

