__author__ = 'jruffio'

# import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
import pyklip.klip as klip
import glob
import os
import pyklip.instruments.osiris as osi
import scipy.linalg as linalg
from scipy.interpolate import UnivariateSpline
import multiprocessing as mp
import itertools
from scipy import interpolate
import random
import scipy.ndimage as ndimage
from scipy.signal import correlate2d
import ctypes
from scipy.ndimage.filters import median_filter
# from scipy.signal import medfilt2d
from copy import copy
from astropy.stats import mad_std
import scipy.io as scio
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import sys
import xml.etree.ElementTree as ET
import csv
import time
from PyAstronomy import pyasl
import scipy.linalg as la
from scipy.sparse.linalg import lsqr
from scipy.sparse import csc_matrix
from scipy.sparse import bsr_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dia_matrix
from scipy.optimize import lsq_linear
#Logic to test mkl exists
from reduce_HPFonly_diagcov_resmodel_v2 import return_64x19,_spline_psf_model,LPFvsHPF,_arraytonumpy
import matplotlib.pyplot as plt
try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

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

    planet = "HR_8799_d"
    # date = "200729"
    # date = "200730"
    date = "200731"
    # date = "200803"
    # IFSfilter = "Kbb"
    IFSfilter = "Kbb"
    # IFSfilter = "Jbb" # "Kbb" or "Hbb"
    scale = "020"
    # scale = "035"
    imnum,ply,plx = 16,40,11
    cutoff=40
    fontsize = 12

    inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/"
    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"

    if 0:
        fig = plt.figure(6,figsize=(6,4))
        psf_filelist = glob.glob(os.path.join(inputDir,"..","reduced_telluric_jb","HR_8799","s"+date+"*"+IFSfilter+"_"+scale+".fits"))
        print(psf_filelist)
        myim_list = []
        for filename in psf_filelist:
            print(filename)
            with pyfits.open(filename) as hdulist:
                prihdr = hdulist[0].header
                curr_mjdobs = prihdr["MJD-OBS"]
                imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                imgs = return_64x19(imgs)
                imgs = np.moveaxis(imgs,0,2)
                imgs_hdrbadpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
                imgs_hdrbadpix = return_64x19(imgs_hdrbadpix)
                imgs_hdrbadpix = np.moveaxis(imgs_hdrbadpix,0,2)
                imgs_hdrbadpix = imgs_hdrbadpix.astype(dtype=ctypes.c_double)
                imgs_hdrbadpix[np.where(imgs_hdrbadpix==0)] = np.nan
                imgs[np.where(imgs_hdrbadpix==0)] = 0
            ny,nx,nz = imgs.shape
            init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
            dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
            wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

            wvid = np.argmin(np.abs(wvs-2.3))

            im = np.nanmedian(imgs,axis=2)
            im/=np.nanmax(im)
            ny,nx = im.shape
            ymax,xmax = np.unravel_index(np.argmax(im),im.shape)
            x_psf_vec, y_psf_vec = np.arange(nx)-xmax, np.arange(ny)-ymax
            x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
            r_grid = np.sqrt(x_psf_grid**2+y_psf_grid**2)*0.02
            im[np.where(im<=0)] = np.nan
            im[np.where(r_grid>0.8)] = np.nan
            im_ravel = np.ravel(im)
            where_fin = np.where(np.isfinite(im_ravel))
            im_ravel = im_ravel[where_fin]
            r_ravel = np.ravel(r_grid)[where_fin]
            myim_list.append(im)
            plt.scatter(r_ravel,im_ravel,s=5,alpha=0.1,c="black")
        plt.yscale("log")
        plt.ylim([1e-4,1])
        plt.xlim([0,0.6])
        plt.ylabel("Normalized PSF")
        plt.xlabel(r"Separation (arcsec)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        # plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"OSIRIS_PSF.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"OSIRIS_PSF.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        # plt.show()
        exit()

    if 1:
        sky_nodarksub_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_nosub","s"+date+"*{0}*".format(9)+IFSfilter+"_"+scale+".fits"))
        sky2_nodarksub_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_nosub","s"+date+"*{0}*".format(22)+IFSfilter+"_"+scale+".fits"))
        sky_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_simpledarksub","s"+date+"*{0}*".format(9)+IFSfilter+"_"+scale+".fits"))
        sky2_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_simpledarksub","s"+date+"*{0}*".format(22)+IFSfilter+"_"+scale+".fits"))
        sc_noskysub_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_nosub","s"+date+"*{0}*".format(imnum)+IFSfilter+"_"+scale+".fits"))
        sc_filelist = glob.glob(os.path.join(inputDir,"..","reduced_jb","s"+date+"*{0}*".format(imnum)+IFSfilter+"_"+scale+".fits"))
        filelist = [sky_nodarksub_filelist[0],
                    sky2_nodarksub_filelist[0],
                    sky_filelist[0],
                    sky2_filelist[0],
                    sc_noskysub_filelist[0],sc_filelist[0]]
        # label_list = ["sky w/o dark sub","sky w/ dark sub","science w/o sky sub","science w/ sky sub"]
        myvec_list = []
        for filename in filelist:
            print(filename)
            with pyfits.open(filename) as hdulist:
                prihdr = hdulist[0].header
                curr_mjdobs = prihdr["MJD-OBS"]
                imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                imgs = return_64x19(imgs)
                imgs = np.moveaxis(imgs,0,2)
                imgs_hdrbadpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
                imgs_hdrbadpix = return_64x19(imgs_hdrbadpix)
                imgs_hdrbadpix = np.moveaxis(imgs_hdrbadpix,0,2)
                imgs_hdrbadpix = imgs_hdrbadpix.astype(dtype=ctypes.c_double)
                imgs_hdrbadpix[np.where(imgs_hdrbadpix==0)] = np.nan
                imgs[np.where(imgs_hdrbadpix==0)] = 0
            ny,nx,nz = imgs.shape
            init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
            dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
            wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

            myvec_list.append(imgs[ply,plx,:])
            myvec = copy(imgs[ply,plx,:])
            window_size=100
            threshold=7
            smooth_vec = median_filter(myvec,footprint=np.ones(window_size),mode="reflect")
            _myvec = myvec - smooth_vec
            wherefinite = np.where(np.isfinite(_myvec))
            mad = mad_std(_myvec[wherefinite])
            whereoutliers = np.where(np.abs(_myvec)>threshold*mad)[0]
            myvec[whereoutliers] = np.nan

            # plt.plot(wvs,myvec,label=label)


        dark = myvec_list[0]-myvec_list[2]
        dark2 = myvec_list[1]-myvec_list[3]
        sky = myvec_list[0]
        delta_sky_nodarksub = myvec_list[0]-myvec_list[1]
        dark_pix2pix = dark-dark2
        window_size=100
        threshold=5
        smooth_vec = median_filter(dark,footprint=np.ones(window_size),mode="reflect")
        _myvec = dark - smooth_vec
        wherefinite = np.where(np.isfinite(_myvec))
        mad = mad_std(_myvec[wherefinite])
        whereoutliers = np.where(np.abs(_myvec)>threshold*mad)[0]
        dark[whereoutliers] = np.nan
        dark_HPF = LPFvsHPF(dark,cutoff=cutoff)[1]
        smooth_vec = median_filter(dark_pix2pix,footprint=np.ones(window_size),mode="reflect")
        _myvec = dark_pix2pix - smooth_vec
        wherefinite = np.where(np.isfinite(_myvec))
        mad = mad_std(_myvec[wherefinite])
        whereoutliers = np.where(np.abs(_myvec)>threshold*mad)[0]
        dark_pix2pix[whereoutliers] = np.nan
        dark_HPF = LPFvsHPF(dark,cutoff=cutoff)[1]
        smooth_vec = median_filter(delta_sky_nodarksub,footprint=np.ones(window_size),mode="reflect")
        _myvec = delta_sky_nodarksub - smooth_vec
        wherefinite = np.where(np.isfinite(_myvec))
        mad = mad_std(_myvec[wherefinite])
        whereoutliers = np.where(np.abs(_myvec)>threshold*mad)[0]
        delta_sky_nodarksub[whereoutliers] = np.nan
        delta_sky_nodarksub = LPFvsHPF(delta_sky_nodarksub,cutoff=cutoff)[1]
        sky[whereoutliers] = np.nan

        # print(np.nanmean(dark),np.nanstd(dark))
        # plt.plot(wvs,dark,label='"dark"')
        # # plt.plot(dark_pix2pix)
        # plt.plot(wvs,sky,label="sky")
        # plt.ylabel("Data number")
        # plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        # plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)
        # plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        # plt.tight_layout()
        # # fig.savefig(os.path.join(out_pngs,"noise_analysis_4.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        # # fig.savefig(os.path.join(out_pngs,"noise_analysis_4.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        # plt.show()
        # exit()

        # res_filelist = glob.glob(os.path.join(inputDir,"sherlock","20191205_RV",os.path.basename(sc_filelist[0]).replace(".fits","")+"*_outputHPF_cutoff40_sherlock_v1_search_rescalc_res.fits"))
        # print(res_filelist[0])
        # with pyfits.open(res_filelist[0]) as hdulist:
        #     hpf = hdulist[0].data[0,0,0,:,:,:]
        #     lpf = hdulist[0].data[0,0,5,:,:,:]
        #     hpfres = hdulist[0].data[0,0,6,:,:,:]
        #     print(lpf.shape)
        #     # exit()
        #     # myvec = (hpf+lpf)[:,ply+5,plx+5]()
        #     myvec = hpfres[:,ply+5,plx+5]
        #     plt.plot(wvs,myvec,linestyle="--",label="Mean 5x5 Residuals H0")
        #



        # plt.legend()
        # plt.show()

    if 1:
        modelfolder = "20200309_model"
        gridname = os.path.join("/data/osiris_data/","hr8799b_modelgrid")
        N_kl = 0#10
        numthreads = 32#16
        small = True
        inj_fake_str=""

        Tfk,loggfk,ctoOfk = 1200,3.7,0.55

        c_kms = 299792.458
        cutoff = 40
        R= 4000

        tmpfilename = os.path.join(gridname,"hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
        hdulist = pyfits.open(tmpfilename)
        planet_model_grid =  hdulist[0].data
        oriplanet_spec_wvs =  hdulist[1].data
        Tlistunique =  hdulist[2].data
        logglistunique =  hdulist[3].data
        CtoOlistunique =  hdulist[4].data
        hdulist.close()

        from scipy.interpolate import RegularGridInterpolator
        myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

        filename = glob.glob(os.path.join(inputDir,"..","reduced_jb","s"+date+"*{0}*".format(imnum)+IFSfilter+"_"+scale+".fits"))[0]
        print(filename)

        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_corrwvs"+inj_fake_str+".fits"))
        if len(glob.glob(tmpfilename))!=1:
            print("No data on "+filename)
            exit()
        hdulist = pyfits.open(tmpfilename)
        # wvs =  hdulist[0].data
        if small:
            wvs =  hdulist[0].data[2:7,2:7,:]
        else:
            wvs =  hdulist[0].data
        hdulist.close()

        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_LPFdata"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            LPFdata =  hdulist[0].data[2:7,2:7,:]
        else:
            LPFdata =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_HPFdata"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        print(hdulist[0].data.shape)
        if small:
            HPFdata =  hdulist[0].data[2:7,2:7,:]
        else:
            HPFdata =  hdulist[0].data
        hdulist.close()

        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_badpix"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            data_badpix =  hdulist[0].data[2:7,2:7,:]
        else:
            data_badpix =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_sigmas"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            data_sigmas =  hdulist[0].data[2:7,2:7,:]
        else:
            data_sigmas =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_trans"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        transmission_vec =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_starspec"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            star_obsspec =  hdulist[0].data[2:7,2:7,:]
        else:
            star_obsspec =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_reskl"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if N_kl == 0:
            res4model_kl = None
        else:
            res4model_kl =  hdulist[0].data[:,0:N_kl]
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_plrv0"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        plrv0 =  hdulist[0].data
        data_ny,data_nx,data_nz = HPFdata.shape
        w = int((data_nx-1)//2)
        star_flux = np.nansum(star_obsspec[w,w,:])
        hdulist.close()

        ##############################
        ## Create PSF model
        ##############################
        ref_star_folder = os.path.join(os.path.dirname(filename),"..","reduced_telluric_jb")
        with pyfits.open(glob.glob(os.path.join(ref_star_folder,"*"+IFSfilter+"_hdpsfs_v2.fits"))[0]) as hdulist:
            psfs_refstar_arr = hdulist[0].data[None,:,:,:]
        with pyfits.open(glob.glob(os.path.join(ref_star_folder,"*"+IFSfilter+"_hdpsfs_xy_v2.fits"))[0]) as hdulist:
            hdpsfs_xy = hdulist[0].data
            hdpsfs_x,hdpsfs_y = hdpsfs_xy

        nx_psf,ny_psf = 15,15
        nz_psf = psfs_refstar_arr.shape[1]
        x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
        x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)

        print("starting psf")
        specpool = mp.Pool(processes=numthreads)
        chunk_size=20
        N_chunks = nz_psf//chunk_size
        psfs_chunks = []
        for k in range(N_chunks-1):
            psfs_chunks.append(psfs_refstar_arr[:,k*chunk_size:(k+1)*chunk_size,:,:])
        psfs_chunks.append(psfs_refstar_arr[:,(N_chunks-1)*chunk_size:nz_psf,:,:])
        outputs_list = specpool.map(_spline_psf_model, zip(psfs_chunks,
                                                           itertools.repeat(hdpsfs_x[None,:,:]),
                                                           itertools.repeat(hdpsfs_y[None,:,:]),
                                                           itertools.repeat(x_psf_grid[0,0:nx_psf-1]+0.5),
                                                           itertools.repeat(y_psf_grid[0:ny_psf-1,0]+0.5),
                                                           np.arange(len(psfs_chunks))))

        normalized_psfs_func_list = []
        chunks_ids = []
        for out in outputs_list:
            normalized_psfs_func_list.extend(out[1])
            chunks_ids.append(out[0])
        print("finish psf")
        specpool.close()
        specpool.join()
        print("closed psf")

        dx,dy = 0,0
        nospec_planet_model = np.zeros(HPFdata.shape)
        pl_x_vec = np.arange(-w,w+1) + dx
        pl_y_vec = np.arange(-w,w+1) + dy
        for z in range(data_nz):
            nospec_planet_model[:,:,z] = normalized_psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()


        ravelHPFdata = np.ravel(HPFdata)
        ravelLPFdata = np.ravel(LPFdata)
        ravelsigmas = np.ravel(data_sigmas)
        where_finite_data = np.where(np.isfinite(np.ravel(data_badpix)))
        where_bad_data = np.where(~(np.isfinite(np.ravel(data_badpix))))
        ravelLPFdata = ravelLPFdata[where_finite_data]
        sigmas_vec = ravelsigmas[where_finite_data]#np.ones(ravelLPFdata.shape)#np.sqrt(np.abs(ravelLPFdata))
        ravelHPFdata = ravelHPFdata[where_finite_data]
        ravelHPFdata = ravelHPFdata/sigmas_vec
        logdet_Sigma = np.sum(2*np.log(sigmas_vec))

        planetRV_array = np.array([plrv0])

        HPFmodelH0_list = []
        if 1:
            bkg_model = np.zeros((2*w+1,2*w+1,2*w+1,2*w+1,data_nz))
            for bkg_k in range(2*w+1):
                for bkg_l in range(2*w+1):
                    if 1:
                        star_obsspec_tmp = star_obsspec[bkg_k,bkg_l]
                        smooth_model = median_filter(star_obsspec_tmp,footprint=np.ones(50),mode="reflect")
                        where_bad_pix_4model = np.where(np.isnan(data_badpix[bkg_k,bkg_l,:]))
                        star_obsspec_tmp[where_bad_pix_4model] = smooth_model[where_bad_pix_4model]
                        star_obsspec_tmp[np.where(np.isnan(HPFdata[bkg_k,bkg_l,:]))] = np.nan
                    LPF_star_obsspec_tmp,_ = LPFvsHPF(star_obsspec_tmp,cutoff)

                    myspec = LPFdata[bkg_k,bkg_l,:]*star_obsspec_tmp/LPF_star_obsspec_tmp
                    if 1:
                        smooth_model = median_filter(myspec,footprint=np.ones(50),mode="reflect")
                        where_bad_pix_4model = np.where(np.isnan(data_badpix[bkg_k,bkg_l,:]))
                        myspec[where_bad_pix_4model] = smooth_model[where_bad_pix_4model]
                        myspec[np.where(np.isnan(HPFdata[bkg_k,bkg_l,:]))] = np.nan
                    _,myspec = LPFvsHPF(myspec,cutoff)

                    bkg_model[bkg_k,bkg_l,bkg_k,bkg_l,:] = myspec
            HPFmodelH0_list.append(np.reshape(bkg_model,((2*w+1)**2,(2*w+1)**2*data_nz)).transpose())
        if res4model_kl is not None:
            for kid in range(res4model_kl.shape[1]):
                res4model = res4model_kl[:,kid]
                LPF4resmodel = np.nansum(LPFdata*nospec_planet_model,axis=(0,1))/np.nansum(nospec_planet_model**2,axis=(0,1))
                resmodel = nospec_planet_model*LPF4resmodel[None,None,:]*res4model[None,None,:]
                HPFmodelH0_list.append(np.ravel(resmodel)[:,None])


        HPFmodel_H0 = np.concatenate(HPFmodelH0_list,axis=1)

        HPFmodel_H0 = HPFmodel_H0[where_finite_data[0],:]/sigmas_vec[:,None]


        cp_HPFmodel_H0 = copy(HPFmodel_H0)
        cp_HPFmodel_H0[np.where(np.isnan(HPFmodel_H0))] = 0

        w = int((nospec_planet_model.shape[0]-1)/2)
        c_kms = 299792.458
        # print(temp,fitlogg,CtoO)
        planet_template_func = interp1d(oriplanet_spec_wvs,myinterpgrid([Tfk,loggfk,ctoOfk])[0],bounds_error=False,fill_value=np.nan)

        planet_model = copy(nospec_planet_model)
        for bkg_k in range(2*w+1):
            for bkg_l in range(2*w+1):
                # print(wvs.shape,plrv,c_kms)
                wvs4planet_model = wvs[bkg_k,bkg_l,:]*(1-(plrv0)/c_kms)
                planet_model[bkg_k,bkg_l,:] *= planet_template_func(wvs4planet_model) * transmission_vec

        star_model = copy(nospec_planet_model)
        for bkg_k in range(2*w+1):
            for bkg_l in range(2*w+1):
                # print(wvs.shape,plrv,c_kms)
                star_model[bkg_k,bkg_l,:] *= star_obsspec[3,3,:]
        star_model = star_model/np.nansum(star_model)*star_flux


        planet_model = planet_model/np.nansum(planet_model)*star_flux*1e-5
        HPF_planet_model = np.zeros(planet_model.shape)
        for bkg_k in range(2*w+1):
            for bkg_l in range(2*w+1):
                HPF_planet_model[bkg_k,bkg_l,:]  = LPFvsHPF(planet_model[bkg_k,bkg_l,:] ,cutoff)[1]


        HPFmodel_H1only = (HPF_planet_model.ravel())[:,None]
        HPFmodel_H1only = HPFmodel_H1only[where_finite_data[0],:]/sigmas_vec[:,None] # where_finite_data[0]
        HPFmodel_H1only[np.where(np.isnan(HPFmodel_H1only))] = 0

        HPFmodel = np.concatenate([HPFmodel_H1only,cp_HPFmodel_H0],axis=1)
        HPFmodel_H1_cp_4res = copy(HPFmodel)

        where_valid_parameters = np.where(np.nansum(np.abs(HPFmodel)>0,axis=0)>=50)
        HPFmodel = HPFmodel[:,where_valid_parameters[0]]

        HPFparas,HPFchi2,rank,s = np.linalg.lstsq(HPFmodel,ravelHPFdata,rcond=None)
        data_model = np.dot(HPFmodel,HPFparas)
        ravelresiduals = ravelHPFdata-data_model
        # HPFchi2 = np.nansum((ravelresiduals)**2)
        # Npixs_HPFdata = HPFmodel.shape[0]
        # covphi =  HPFchi2/Npixs_HPFdata*np.linalg.inv(np.dot(HPFmodel.T,HPFmodel))
        # slogdet_icovphi0 = np.linalg.slogdet(np.dot(HPFmodel.T,HPFmodel))
        # logpost_rv[temp_id,plrv_id] = -0.5*logdet_Sigma-0.5*slogdet_icovphi0[1]- (Npixs_HPFdata-HPFmodel.shape[-1]+2-1)/(2)*np.log(HPFchi2)


        canvas_res= np.zeros(HPFdata.shape) + np.nan
        canvas_res = np.reshape(canvas_res,((2*w+1)**2*data_nz))
        canvas_res[where_finite_data[0]] = ravelresiduals
        canvas_res = np.reshape(canvas_res,((2*w+1),(2*w+1),data_nz))
        canvas_res *= data_sigmas
        canvas_model= np.zeros(HPFdata.shape) + np.nan
        canvas_model = np.reshape(canvas_model,((2*w+1)**2*data_nz))
        canvas_model[where_finite_data[0]] = data_model
        canvas_model = np.reshape(canvas_model,((2*w+1),(2*w+1),data_nz))
        canvas_model *= data_sigmas

        _res = canvas_res[2,2,:]
        _res[np.where(np.isnan(_res))] = 0
        # _res = np.ones(_res.shape)
        res_ccf = np.correlate(_res,_res,mode="same")/np.size(_res)
        res_ccf_argmax = np.argmax(res_ccf)


        fig = plt.figure(1,figsize=(12,4))
        # colors=["#006699","#ff9900","#6600ff","grey"]
        plt.plot(wvs[2,2,:],star_model[2,2,:],linestyle="--",linewidth=3,color="#006699",label="On-axis star")
        plt.plot(wvs[2,2,:],LPFdata[2,2,:]+HPFdata[2,2,:],linestyle="-",linewidth=2,color="#ff9900",label="Starlight + planet")
        plt.plot(wvs[2,2,:],planet_model[2,2,:]*HPFparas[0],linestyle="-",linewidth=1,color="#6600ff",label="Scaled planet (fit)")
        plt.plot(wvs[2,2,:],sky,linestyle="-.",linewidth=0.5,color="blue",label="Sky")
        # plt.plot(wvs[2,2,:],dark,linestyle="--",linewidth=0.5,color="grey",label="Irreducible noise")
        plt.yscale("log")
        plt.ylim([1e-2,1e5])
        plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        plt.ylabel(r"Data Number",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_1.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_1.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

        fig = plt.figure(2,figsize=(12,3))
        # colors=["#006699","#ff9900","#6600ff","grey"]
        print(np.nansum(planet_model[2,2,:]*HPFparas[0])/np.nansum((LPFdata[2,2,:]+HPFdata[2,2,:])))
        # exit()
        plt.plot(wvs[2,2,:],planet_model[2,2,:]*HPFparas[0]/(LPFdata[2,2,:]+HPFdata[2,2,:]),linestyle="--",linewidth=1,color="#6600ff",label="Scaled planet / Data")
        plt.plot(wvs[2,2,:],(LPFdata[2,2,:]+HPFdata[2,2,:])/star_model[2,2,:],linestyle="-",linewidth=2,color="#ff9900",label="(Starlight + planet) / On-axis star")
        plt.plot(wvs[2,2,:],planet_model[2,2,:]*HPFparas[0]/star_model[2,2,:],linestyle="-",linewidth=1,color="#6600ff",label="Scaled planet / On-axis star")
        plt.yscale("log")
        plt.ylim([1e-6,1e-1])
        plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        plt.ylabel(r"Flux ratio",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_2.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_2.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

        fig = plt.figure(3,figsize=(12,3))
        plt.plot(wvs[2,2,:],HPFdata[2,2,:],linestyle="-",linewidth=2,color="#ff9900",label="Data (HPF; Starlight + planet)")
        plt.plot(wvs[2,2,:],canvas_model[2,2,:],linestyle="--",linewidth=0.5,color="black",label="Forward Model")
        plt.plot(wvs[2,2,:],HPF_planet_model[2,2,:]*HPFparas[0],linestyle="-",linewidth=1,color="#6600ff",label="Scaled planet (HPF)")
        plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        plt.ylabel(r"Data Number",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_3.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_3.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

        fig = plt.figure(4,figsize=(12,2))
        # plt.plot(wvs,dark_pix2pix/np.sqrt(2),linestyle="--",label="dark_pix2pix: std={0}".format()
        # plt.plot(wvs,delta_sky_nodarksub/np.sqrt(2),linestyle="-.",label="sky + dark pix2pix: std={0}".format(np.sqrt((np.nanvar(delta_sky_nodarksub)-np.nanvar(dark_pix2pix))/2)))
        # plt.plot(wvs[2,2,:],canvas_res[2,2,:],label="res: std_pix2pix={0} ; std_corr={1}".format(np.sqrt(res_ccf[res_ccf_argmax]-res_ccf[res_ccf_argmax-1]),
        #                                                                                          np.sqrt(res_ccf[res_ccf_argmax-1])))
        # plt.subplot(1,2,1)
        # plt.plot(wvs[2,2,:],dark_pix2pix/np.sqrt(2),linestyle="--",linewidth=0.5,color="grey",label="$\Delta$(Irr. noise)/$\sqrt{2}$ (HPF)")
        plt.plot(wvs[2,2,:],delta_sky_nodarksub/np.sqrt(2),linestyle="-.",linewidth=0.5,color="blue",label="$\Delta$Sky (HPF)")
        plt.plot(wvs[2,2,:],canvas_res[2,2,:],linestyle="-",linewidth=1,color="#6600ff",label="Residuals",alpha=0.5)
        plt.ylim([-0.2,0.2])
        plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        plt.ylabel(r"Data Number",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        # plt.subplot(1,2,2)
        # plt.plot(wvs[2,2,:],dark_pix2pix/np.sqrt(2),linestyle="--",linewidth=0.5,color="grey",label="$\Delta$(Irr. noise)/$\sqrt{2}$ (HPF)")
        # plt.plot(wvs[2,2,:],delta_sky_nodarksub/np.sqrt(2),linestyle="-.",linewidth=0.5,color="blue",label="$\Delta$Sky/$\sqrt{2}$ (HPF)")
        # plt.plot(wvs[2,2,:],canvas_res[2,2,:],linestyle="-",linewidth=1,color="#6600ff",label="Residuals",alpha=0.5)
        # plt.ylim([-0.2,0.2])
        # plt.xlim([2.28,2.38])
        # plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        # plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_4.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_4.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

        fig = plt.figure(5,figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(np.arange(-100,101),res_ccf[(res_ccf_argmax-100):(res_ccf_argmax+101)])
        plt.xlabel(r"Spectral pixel",fontsize=fontsize)
        plt.ylabel(r"Auto-correlation (Variance)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.subplot(1,2,2)
        plt.plot(np.arange(-5,6),res_ccf[(res_ccf_argmax-5):(res_ccf_argmax+6)])
        plt.xlabel(r"Spectral pixel",fontsize=fontsize)
        plt.ylabel(r"Auto-correlation (Variance)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_5.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_5.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

        print(res_ccf[res_ccf_argmax-1],res_ccf[res_ccf_argmax],res_ccf[res_ccf_argmax+1])
        # print("var res pix2pix - var res corr",res_ccf[res_ccf_argmax]-res_ccf[res_ccf_argmax-1])
        # print("var res corr",res_ccf[res_ccf_argmax-1])
        # print("var dark pix2pix",np.nanvar(dark_pix2pix)/2)
        # # print("var dark HPF",np.nanvar(dark_HPF))
        # print("var sky pix2pix",(np.nanvar(delta_sky_nodarksub)-np.nanvar(dark_pix2pix))/2)
        #
        # print("dark_pix2pix: var={0}".format((np.nanvar(dark_pix2pix)/2)))
        # print("sky - dark pix2pix: var={0}".format(((np.nanvar(delta_sky_nodarksub)-np.nanvar(dark_pix2pix))/2)))
        # print("res: var_pix2pix={0} ; var_corr={1} ; var_tot={2}".format((res_ccf[res_ccf_argmax]-res_ccf[res_ccf_argmax-1]),
        #                                                                                          (res_ccf[res_ccf_argmax-1]),
        #                                                                                          (res_ccf[res_ccf_argmax])))
        # print("dark_pix2pix: std={0}".format(np.sqrt(np.nanvar(dark_pix2pix)/2)))
        # print("sky - dark pix2pix: std={0}".format(np.sqrt((np.nanvar(delta_sky_nodarksub)-np.nanvar(dark_pix2pix))/2)))
        # print("res: std_pix2pix={0} ; std_corr={1} ; var_tot={2}".format(np.sqrt(res_ccf[res_ccf_argmax]-res_ccf[res_ccf_argmax-1]),
        #                                                                                          np.sqrt(res_ccf[res_ccf_argmax-1]),
        #                                                                                          np.sqrt(res_ccf[res_ccf_argmax])))


        print("sky std", np.nanstd(delta_sky_nodarksub))
        print("res std", np.nanstd(canvas_res[2,2,:]))
        print("photon std", np.sqrt(np.nanmedian((LPFdata[2,2,:]+HPFdata[2,2,:]))*600*2.15)/600)
        print(np.nanmedian(LPFdata[2,2,:]+HPFdata[2,2,:]))

        print("bin planet SNR",np.median(planet_model[2,2,:]*HPFparas[0])/np.nanstd(canvas_res[2,2,:]))
        print("bin star SNR",np.nanmedian(HPFdata[2,2,:]+LPFdata[2,2,:])/np.nanstd(canvas_res[2,2,:]))
        print("starlight",np.nanmedian((LPFdata[2,2,:]+HPFdata[2,2,:])/star_model[2,2,:]))
        print(np.median(np.nanmax(nospec_planet_model,axis=(0,1))/np.nansum(nospec_planet_model,axis=(0,1))))


    plt.show()
    exit()

    exit()


    # inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb_pairsub/"
    # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb_pairsub/20190228_HPF_only/"

    print(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
    filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
    filelist.sort()
    filelist = [filelist[7],]
    print(filelist)