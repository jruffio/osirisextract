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
try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

def _arraytonumpy(shared_array, shape=None, dtype=None):
    """
    Covert a shared array to a numpy array
    Args:
        shared_array: a multiprocessing.Array array
        shape: a shape for the numpy array. otherwise, will assume a 1d array
        dtype: data type of the arrays. Should be either ctypes.c_float(default) or ctypes.c_double

    Returns:
        numpy_array: numpy array for vectorized operation. still points to the same memory!
                     returns None is shared_array is None
    """
    if dtype is None:
        dtype = ctypes.c_float

    # if you passed in nothing you get nothing
    if shared_array is None:
        return None

    numpy_array = np.frombuffer(shared_array.get_obj(), dtype=dtype)
    if shape is not None:
        numpy_array.shape = shape

    return numpy_array

def _tpool_init(original_imgs,sigmas_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps, output_maps_shape,
                wvs_imgs,psfs_stamps,psfs_stamps_shape,_outres,_outres_shape,_outautocorrres,_outautocorrres_shape,persistence_imgs,_out1dfit,_out1dfit_shape,_estispec,_estispec_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Args:
    """
    global original,sigmas,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, \
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape,estispec,estispec_shape
    # original images from files to read and align&scale. Shape of (N,y,x)
    original = original_imgs
    sigmas = sigmas_imgs
    badpix = badpix_imgs
    originalLPF = originalLPF_imgs
    originalHPF = originalHPF_imgs
    original_shape = original_imgs_shape
    # output images after KLIP processing (amplitude and ...) (5, y, x)
    output = output_maps
    output_shape = output_maps_shape
    outres = _outres
    outres_shape = _outres_shape
    outautocorrres = _outautocorrres
    outautocorrres_shape = _outautocorrres_shape
    out1dfit_shape = _out1dfit_shape
    estispec_shape = _estispec_shape

    out1dfit = _out1dfit
    estispec = _estispec
    persistence = persistence_imgs

    # parameters for each image (PA, wavelegnth, image center, image number)
    lambdas = wvs_imgs
    psfs = psfs_stamps
    psfs_shape = psfs_stamps_shape
    Npixproc= 0
    Npixtot=0


def _remove_bad_pixels_z(col_index,nan_mask_boxsize,dtype,window_size=50,threshold=7.):
    global original,sigmas,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, \
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape,estispec,estispec_shape
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    ny,nx,nz = original_shape
    tmpcube = copy(original_np[:,col_index,:])
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    for m in np.arange(0,original_shape[0]):
        try:
            myvec = tmpcube[m,:]
            # wherefinite = np.where(np.isfinite(myvec))
            # if np.size(wherefinite[0])==0:
            #     continue
            # smooth_vec = median_filter(myvec,footprint=np.ones(window_size),mode="constant",
            #                            cval=np.nanmedian(myvec[np.where(np.isfinite(badpix_np[m,col_index,:]))]))
            smooth_vec = median_filter(myvec,footprint=np.ones(window_size),mode="reflect")
            myvec = myvec - smooth_vec
            wherefinite = np.where(np.isfinite(myvec))
            mad = mad_std(myvec[wherefinite])
            whereoutliers = np.where(np.abs(myvec)>threshold*mad)[0]
            badpix_np[m,col_index,whereoutliers] = np.nan
            widen_badpix_vec = np.correlate(badpix_np[m,col_index,:],np.ones(nan_mask_boxsize),mode="same")
            widen_nans = np.where(np.isnan(widen_badpix_vec))[0]
            badpix_np[m,col_index,widen_nans] = np.nan
            original_np[m,col_index,widen_nans] = smooth_vec[widen_nans]
            # badpix_np[m,col_index,:] = smooth_vec
        except:
            print("Error",m,col_index)

# def LPFvsHPF(myvec,cutoff):
#     LPF_myvec = np.zeros(myvec.shape)
#
#     for k in np.arange(np.size(myvec)):
#         LPF_myvec[k] = np.nanmean(myvec[np.nanmax([k-30,0]):np.nanmin([k+30,np.size(myvec)])])
#     # for k in np.arange(np.size(myvec)):
#     #     print((myvec-LPF_myvec)[k],np.nanmean((myvec-LPF_myvec)[np.nanmax([k-30,0]):np.nanmin([k+30,np.size(myvec)])]))
#         # print()
#     LPF_myvec[np.where(np.isnan(myvec))] = np.nan
#     return LPF_myvec,myvec-LPF_myvec

def LPFvsHPF(myvec,cutoff,nansmooth=10):
    myvec_cp = copy(myvec)
    #handling nans:
    wherenans = np.where(np.isnan(myvec_cp))
    for k in wherenans[0]:
        myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-nansmooth]):np.min([np.size(myvec_cp),k+nansmooth])])

    fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec

    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan
    return LPF_myvec,HPF_myvec

def _HPF_z(col_index,cutoff,dtype):
    global original,sigmas,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, \
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape,estispec,estispec_shape
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    originalLPF_np = _arraytonumpy(originalLPF, original_shape,dtype=dtype)
    originalHPF_np = _arraytonumpy(originalHPF, original_shape,dtype=dtype)
    # badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    for m in np.arange(0,original_shape[0]):
    # if 1:
    #     m = 30
        myvec = copy(original_np[m,col_index,:])
        originalLPF_np[m,col_index,:],originalHPF_np[m,col_index,:] = LPFvsHPF(myvec,cutoff)
        # wherebad = np.where(np.isnan(badpix_np[m,col_index,:]))
        # myvec[wherebad] = originalLPF_np[m,col_index,wherebad[0]]
        # originalLPF_np[m,col_index,:],originalHPF_np[m,col_index,:] = LPFvsHPF(myvec,cutoff)
        # fftmyvec = np.fft.fft(np.concatenate([myvec,myvec[::-1]],axis=0))
        # LPF_fftmyvec = copy(fftmyvec)
        # LPF_fftmyvec[cutoff:(2*np.size(myvec)-cutoff+1)] = 0
        # LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec)]
        # HPF_myvec = myvec - LPF_myvec
        # originalLPF_np[m,col_index,:] = LPF_myvec
        # originalHPF_np[m,col_index,:] = HPF_myvec
        # original_np[m,col_index,:] = LPF_myvec

        # print(col_index,np.size(np.where(np.isfinite(originalLPF_np))[0]))

        # plt.plot(myvec,label="original")
        # plt.plot(LPF_myvec,label="LPF_myvec")
        # plt.plot(HPF_myvec,label="HPF_myvec")
        # plt.show()
        # # plt.plot(np.abs(fftmyvec),label="original")
        # # plt.plot(np.abs(LPF_fftmyvec),label="original")
        # # smooth_vec = median_filter(myvec,footprint=np.ones(100),mode="constant",cval=0.0)
        # # original_np[m,col_index,:] = original_np[m,col_index,:] - smooth_vec

def _remove_edges(wvs_indices,nan_mask_boxsize,dtype):
    global original,sigmas,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, \
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape,estispec,estispec_shape
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    for k in wvs_indices:
        tmp = np.ones(badpix_np.shape[0:2])#badpix_np[:,:,k]
        tmp[np.where(original[:,:,k]==0)] = np.nan
        tmp[np.where(np.isnan(correlate2d(tmp,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
    badpix_np[0:nan_mask_boxsize//2,:,:] = np.nan
    badpix_np[-nan_mask_boxsize//2+1::,:,:] = np.nan
    badpix_np[:,0:nan_mask_boxsize//2,:] = np.nan
    badpix_np[:,-nan_mask_boxsize//2+1::,:] = np.nan


def _remove_bad_pixels_xy(wvs_indices,dtype):
    global original,sigmas,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, \
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape,estispec,estispec_shape
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    for k in wvs_indices:
        # tmp = original_np[:,:,k]
        tmpcopy= copy(original_np[:,:,k])
        tmpbadpix = badpix_np[:,:,k]
        # tmpbadpixcopy = copy(badpix_np[:,:,k])
        wherefinite = np.where(np.isfinite(tmpbadpix))
        wherenans = np.where(np.isnan(tmpbadpix))
        if np.size(wherefinite[0])==0:
            continue
        tmpcopy[wherenans] = 0
        smooth_map = median_filter(tmpcopy,footprint=np.ones((5,5)),mode="constant",cval=0.0)#medfilt2d(tmpcopy,5)
        tmpcopy[wherenans] = np.nan
        tmpcopy = tmpcopy - smooth_map
        mad = mad_std(tmpcopy[wherefinite])
        whereoutliers = np.where(np.abs(tmpcopy)>7*mad)
        tmpbadpix[whereoutliers] = np.nan
        # tmp[whereoutliers] = smooth_map[whereoutliers]
        # tmp[np.where(np.isnan(correlate2d(tmp,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
        # print(mad)
        # plt.imshow((original_np[:,:,k]),interpolation="nearest")
        # plt.show()


def combine_spectra(_spec_list):
    _spec_LPF_list = []
    _spec_HPF_list = []
    for fid,spec_it in enumerate(_spec_list):
        a,b = LPFvsHPF(spec_it/np.nanmean(spec_it),cutoff=10,nansmooth=50)
        _spec_LPF_list.append(a)
        _spec_HPF_list.append(b)
    _spec = np.nanmean(_spec_LPF_list, axis=0) + np.nanmean(_spec_HPF_list, axis=0)
    return _spec

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
    return x[argmax_post],lx,rx,lx-x[argmax_post],rx-x[argmax_post],argmax_post

# plcen_k_valid_pix[::-1],plcen_l_valid_pix[::-1],row_valid_pix[::-1],col_valid_pix[::-1],
#                                     normalized_psfs_func_list,
#                                     hr8799modelspec_Rlist,
#                                     transmission_Rlist,
#                                     planet_template_func_Rlist,
#                                     func_skytrans_Rlist,
#                                     wvs_imgs,planetRV_array,
#                                     dtype,cutoff,planet_search,(plcen_k,plcen_l),
#                                     R_list,
#                                     numbasis_list,wvsol_offsets)
def _process_pixels_onlyHPF(curr_k_indices,curr_l_indices,row_indices,col_indices,
                            normalized_psfs_func_list,
                            transmission_table,
                            planet_model_func_table,
                            HR8799pho_spec_func_list,
                            transmission4planet_list,
                            hr8799_flux,
                            wvs,planetRV_array,dtype,cutoff,planet_search,centroid_guess,
                            R_list,numbasis_list,wvsol_offsets,R_calib_arr=None,model_persistence=False,res4model_kl=None,lpf_res_calib=None,fake_paras=None):
    global original,sigmas,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, \
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape,estispec,estispec_shape
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    sigmas_np = _arraytonumpy(sigmas, original_shape,dtype=dtype)
    originalLPF_np = _arraytonumpy(originalLPF, original_shape,dtype=dtype)
    originalHPF_np = _arraytonumpy(originalHPF, original_shape,dtype=dtype)
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    output_maps_np = _arraytonumpy(output_maps, output_maps_shape,dtype=dtype)
    outres_np = _arraytonumpy(outres, outres_shape,dtype=dtype)
    outautocorrres_np = _arraytonumpy(outautocorrres, outautocorrres_shape,dtype=dtype)
    psfs_tlc = _arraytonumpy(psfs, psfs_shape,dtype=dtype)
    padny,padnx,padnz = original_shape
    persistence_np = _arraytonumpy(persistence, original_shape,dtype=dtype)
    out1dfit_np = _arraytonumpy(out1dfit, out1dfit_shape,dtype=dtype)
    estispec_np = _arraytonumpy(estispec, estispec_shape,dtype=dtype)
    # print("coucou2")

    tmpwhere = np.where(np.isfinite(badpix_np))
    chi2ref = 0#np.nansum((originalHPF_np[tmpwhere]/sigmas_imgs_np[tmpwhere])**2)


    # # import matplotlib.pyplot as plt
    # # cube_cp = copy(original_np)
    # # cube_cp[np.where(cube_cp < np.nanmedian(cube_cp,axis=(0,1))[None,None,:])] = np.nan
    # LPF_self_ref,HPF_self_ref = LPFvsHPF(np.nanmean(original_np,axis=(0,1)),cutoff)
    # self_line_spec = HPF_self_ref/LPF_self_ref
    # # plt.plot(np.nanmean(cube_cp,axis=(0,1)))
    # # plt.plot(LPF_self_ref)
    # # plt.plot(HPF_self_ref)
    # # plt.show()
    # # exit()


    for curr_k,curr_l,row,col in zip(curr_k_indices,curr_l_indices,row_indices,col_indices):
        # print("coucou3")
        if planet_search:
            k,l = int(np.round(curr_k)),int(np.round(curr_l))
            # print(curr_k,curr_l,row,col)
            w = 2
        else:
            k,l = int(np.round(centroid_guess[0])),int(np.round(centroid_guess[1]))
            # w = 4
            w = 2

        ###################################
        ## Extrac the data
        HPFdata = copy(originalHPF_np[k-w:k+w+1,l-w:l+w+1,:])
        LPFdata = copy(originalLPF_np[k-w:k+w+1,l-w:l+w+1,:])
        data_sigmas = sigmas_np[k-w:k+w+1,l-w:l+w+1,:]
        data_badpix = badpix_np[k-w:k+w+1,l-w:l+w+1,:]
        data_wvsol_offsets = wvsol_offsets[k-w:k+w+1,l-w:l+w+1]

        data_ny,data_nx,data_nz = HPFdata.shape
        x_vec, y_vec = np.arange(padnx * 1.)-curr_l,np.arange(padny* 1.)-curr_k
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        x_data_grid, y_data_grid = x_grid[k-w:k+w+1,l-w:l+w+1], y_grid[k-w:k+w+1,l-w:l+w+1]

        ###################################
        ## Define planet model
        nospec_planet_model = np.zeros(HPFdata.shape)
        pl_x_vec = x_data_grid[0,:]
        pl_y_vec = y_data_grid[:,0]
        for z in range(data_nz):
            nospec_planet_model[:,:,z] = normalized_psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()

        if fake_paras is not None:
            c_kms = 299792.458
            planet_model = copy(nospec_planet_model)
            for bkg_k in range(2*w+1):
                for bkg_l in range(2*w+1):

                    wvs4planet_model = wvs*(1-fake_paras["RV"]/c_kms) \
                                       -data_wvsol_offsets[bkg_k,bkg_l]
                    planet_model[bkg_k,bkg_l,:] *= planet_model_func_table[0][0](wvs4planet_model) * \
                        transmission4planet_list[0](wvs)
            planet_model = planet_model/np.nansum(planet_model)*hr8799_flux

            HPF_fake = np.zeros(planet_model.shape)
            LPF_fake = np.zeros(planet_model.shape)
            for bkg_k in range(2*w+1):
                for bkg_l in range(2*w+1):
                    LPF_fake[bkg_k,bkg_l,:],HPF_fake[bkg_k,bkg_l,:]  = LPFvsHPF(planet_model[bkg_k,bkg_l,:] ,cutoff)


            HPFdata = HPFdata + HPF_fake*fake_paras["contrast"]
            LPFdata = LPFdata + LPF_fake*fake_paras["contrast"]

            # import matplotlib.pyplot as plt
            # plt.figure(1)
            # plt.plot(np.ravel(HPFdata),label="before HPFdata")
            # plt.plot(np.ravel(HPFdata2),label="after HPFdata")
            # plt.plot(np.ravel(HPFdata2-HPFdata),label="diff HPFdata")
            # plt.legend()
            # plt.figure(1)
            # plt.plot(np.ravel(LPFdata),label="before LPFdata")
            # plt.plot(np.ravel(LPFdata2),label="after LPFdata")
            # plt.plot(np.ravel(LPFdata2-LPFdata),label="diff LPFdata")
            # plt.legend()
            #
            # plt.show()

        data_R_calib = R_calib_arr[k-w:k+w+1,l-w:l+w+1]
        if model_persistence:
            data_persistence = persistence_np[k-w:k+w+1,l-w:l+w+1]


        # import matplotlib.pyplot as plt
        # for k in range(5):
        #     for l in range(5):
        #         plt.subplot(5,5,5*k+l+1)
        #         plt.plot(HPFdata[k,l,:],label="HPF")
        #         wherefinite = np.where(np.isfinite(data_badpix[k,l,:]))
        #         plt.plot(wherefinite[0],HPFdata[k,l,:][wherefinite],label="BP after")
        # plt.legend()
        # plt.show()


        #(4,padimgs.shape[-1],padny,padnx)
        #data
        if 1:
            where_bad_data_3d = np.where(np.isnan(data_badpix))
            copy_model = copy(nospec_planet_model)
            copy_model[where_bad_data_3d] = np.nan
            # print(out1dfit_np.shape)
            # canvas_HPFdata = copy(HPFdata)
            # canvas_LPFdata = copy(LPFdata)
            # canvas_LPFdata[where_bad_data_3d] = np.nan
            # canvas_sigmas = copy(data_sigmas)
            # canvas_sigmas[where_bad_data_3d] = np.nan
            data_var = data_sigmas**2
            out1dfit_np[2,row,col,:] = 1/np.nansum(copy_model**2/data_var,axis=(0,1))
            out1dfit_np[0,row,col,:] = np.nansum(HPFdata*copy_model/data_var,axis=(0,1))*out1dfit_np[2,row,col,:]
            out1dfit_np[1,row,col,:] = np.nansum(LPFdata*copy_model/data_var,axis=(0,1))*out1dfit_np[2,row,col,:]
            if model_persistence:
                out1dfit_np[3,row,col,:] = np.nansum(data_persistence*nospec_planet_model,axis=(0,1))

        # import matplotlib.pyplot as plt
        # for bkg_k in range(2*w+1):
        #     for bkg_l in range(2*w+1):
        #         plt.subplot(2*w+1,2*w+1,bkg_k*w+bkg_l+1)
        #         plt.plot(nospec_planet_model[bkg_k,bkg_l,:])
        # plt.show()

        ravelHPFdata = np.ravel(copy(HPFdata))
        ravelLPFdata = np.ravel(copy(LPFdata))
        ravelsigmas = np.ravel(copy(data_sigmas))
        # todo maybe to change back
        # where_finite_data = np.where(np.isfinite(np.ravel(data_badpix))*(ravelLPFdata>0))
        where_finite_data = np.where(np.isfinite(np.ravel(data_badpix)))
        where_bad_data = np.where(~(np.isfinite(np.ravel(data_badpix))))
        ravelLPFdata = ravelLPFdata[where_finite_data]
        sigmas_vec = ravelsigmas[where_finite_data]#np.ones(ravelLPFdata.shape)#np.sqrt(np.abs(ravelLPFdata))
        # print(ravelHPFdata.shape)
        ravelHPFdata = ravelHPFdata[where_finite_data]
        # import matplotlib.pyplot as plt
        # plt.plot(ravelHPFdata,label="HPF")
        # plt.plot(sigmas_vec,label="sig")
        # print(ravelHPFdata.shape)
        # exit()
        ravelHPFdata = ravelHPFdata/sigmas_vec
        # plt.plot(ravelHPFdata,label="HPF/sig")
        # plt.show()
        logdet_Sigma = np.sum(2*np.log(sigmas_vec))

        # import matplotlib.pyplot as plt
        # plt.plot(np.ravel(copy(LPFdata)),label="LPF")
        # plt.plot(np.ravel(copy(data_sigmas)),label="sigmas")
        # plt.legend()
        # plt.show()

        # print(transmission_table)
        # print(planet_model_func_table)
        # print(HR8799pho_spec_func_list)
        # print(transmission4planet_list)
        for model_id, (tr_list,planet_partial_template_func_list,HR8799pho_spec_func,tr4planet) in \
                enumerate(zip(transmission_table,planet_model_func_table,HR8799pho_spec_func_list,transmission4planet_list)):
            # print("coucou4")

            # line_list = []
            # for transmission in tr_list:
            #     HR8799_obsspec = transmission(wvs) * \
            #                     HR8799pho_spec_func(wvs)
            #     LPF_HR8799_obsspec,HPF_HR8799_obsspec = LPFvsHPF(HR8799_obsspec,cutoff)
            #
            #     line_spec = HPF_HR8799_obsspec/LPF_HR8799_obsspec
            #     if lpf_res_calib is not None:
            #         line_spec = line_spec*lpf_res_calib
            #     # line_spec = HR8799_obsspec/LPF_HR8799_obsspec
            #     line_list.append(line_spec)

            for numbasis_id,numbasis in enumerate(numbasis_list):
                # print("coucou5")
                HPFmodelH0_list = []
                # if 1:
                #     meanline = np.nanmean(line_list,axis=0)
                #     mean_wherenans = np.where(np.isnan(meanline))
                #     for myline in line_list:
                #         wherenans = np.where(np.isnan(myline))
                #         myline[wherenans] = meanline[wherenans]
                #         myline[mean_wherenans] = 0
                #
                #     covar_trans = np.cov(np.array(line_list))
                #     evals, evecs = la.eigh(covar_trans, eigvals=(len(line_list)-numbasis, len(line_list)-1))
                #
                #     evals = np.copy(evals[::-1])
                #     evecs = np.copy(evecs[:,::-1], order='F') #fortran order to improve memory caching in matrix multiplication
                #
                #     kl_basis = np.dot(np.array(line_list).T, evecs)
                #     kl_basis = kl_basis * (1. / np.sqrt(evals * (np.size(line_list[0]) - 1)))[None, :]  #multiply a value for each row
                #     kl_basis = kl_basis.T
                #
                #     kl_basis[:,mean_wherenans[0]] = np.nan
                #
                #     # import  matplotlib.pyplot as plt
                #     # plt.plot(meanline,label="mean")
                #     # for n in range(numbasis):
                #     #     plt.plot(kl_basis[n,:],label="{0}".format(n))
                #     # plt.legend()
                #     # plt.show()
                #
                #     new_line_list = kl_basis
                #
                # # new_line_list =np.concatenate([new_line_list,self_line_spec[None,:]])
                #
                # # import matplotlib.pyplot as plt
                # # HPFdata[np.where(np.isnan(data_badpix))] = np.nan
                # # LPFdata[np.where(np.isnan(data_badpix))] = np.nan
                # # plt.subplot(3,1,1)
                # # plt.plot(np.nansum(HPFdata,axis=(0,1)))
                # # plt.subplot(3,1,2)
                # # plt.plot(np.nansum(LPFdata,axis=(0,1))*new_line_list[0])
                # # plt.subplot(3,1,3)
                # # plt.plot(np.nansum(data_sigmas,axis=(0,1)))
                # # plt.show()

                    # plt.plot(line_spec,label="line_spec")
                # for line_spec in new_line_list:
                transmission_vec = tr4planet(wvs)#np.nanmean(np.array([tr(wvs) for tr in tr_list]),axis=0)
                if 1:
                    bkg_model = np.zeros((2*w+1,2*w+1,2*w+1,2*w+1,data_nz))
                    for bkg_k in range(2*w+1):
                        for bkg_l in range(2*w+1):

                            HR8799_obsspec =transmission_vec * \
                                            HR8799pho_spec_func(wvs-data_wvsol_offsets[bkg_k,bkg_l])
                            if 1:
                                smooth_model = median_filter(HR8799_obsspec,footprint=np.ones(50),mode="reflect")
                                where_bad_pix_4model = np.where(np.isnan(data_badpix[bkg_k,bkg_l,:]))
                                HR8799_obsspec[where_bad_pix_4model] = smooth_model[where_bad_pix_4model]
                                HR8799_obsspec[np.where(np.isnan(HPFdata[bkg_k,bkg_l,:]))] = np.nan
                            LPF_HR8799_obsspec,HPF_HR8799_obsspec = LPFvsHPF(HR8799_obsspec,cutoff)

                            myspec = LPFdata[bkg_k,bkg_l,:]*HR8799_obsspec/LPF_HR8799_obsspec
                            if 1:
                                smooth_model = median_filter(myspec,footprint=np.ones(50),mode="reflect")
                                where_bad_pix_4model = np.where(np.isnan(data_badpix[bkg_k,bkg_l,:]))
                                myspec[where_bad_pix_4model] = smooth_model[where_bad_pix_4model]
                                myspec[np.where(np.isnan(HPFdata[bkg_k,bkg_l,:]))] = np.nan
                            _,myspec = LPFvsHPF(myspec,cutoff)

                            # plt.plot(myspec,label="model")
                            # plt.plot(HPFdata[bkg_k,bkg_l,:],label="HPFdata")
                            # plt.plot(LPFdata[bkg_k,bkg_l,:],label="LPFdata")
                            # plt.plot(HPFdata[bkg_k,bkg_l,:]+LPFdata[bkg_k,bkg_l,:],label="data")
                            # plt.legend()
                            # plt.show()

                            bkg_model[bkg_k,bkg_l,bkg_k,bkg_l,:] = myspec
                    HPFmodelH0_list.append(np.reshape(bkg_model,((2*w+1)**2,(2*w+1)**2*data_nz)).transpose())
                if model_persistence:
                    persistence_model = np.zeros((2*w+1,2*w+1,2*w+1,2*w+1,data_nz))
                    for bkg_k in range(2*w+1):
                        for bkg_l in range(2*w+1):
                            if np.nanmean(data_persistence[bkg_k,bkg_l,:]) != 0.0:
                                persistence_model[bkg_k,bkg_l,bkg_k,bkg_l,:] = data_persistence[bkg_k,bkg_l,:]/np.nanstd(data_persistence[bkg_k,bkg_l,:])*np.nanstd(HPFdata[bkg_k,bkg_l,:])/10
                                # import matplotlib.pyplot as plt
                                # print(bkg_k,bkg_l)
                                # print(np.nansum(data_persistence[bkg_k,bkg_l,:]))
                                # plt.plot(data_persistence[bkg_k,bkg_l,:]/np.nanstd(data_persistence[bkg_k,bkg_l,:]))
                                # plt.show()
                    HPFmodelH0_list.append(np.reshape(persistence_model,((2*w+1)**2,(2*w+1)**2*data_nz)).transpose())
                    # print(np.nansum(np.abs(np.reshape(persistence_model,((2*w+1)**2,(2*w+1)**2*data_nz)).transpose()),axis=0))
                if res4model_kl is not None:
                    for kid in range(res4model_kl.shape[1]):
                        res4model = res4model_kl[:,kid]
                        LPF4resmodel = np.nansum(LPFdata*nospec_planet_model,axis=(0,1))/np.nansum(nospec_planet_model**2,axis=(0,1))
                        resmodel = nospec_planet_model*LPF4resmodel[None,None,:]*res4model[None,None,:]
                        # print(nospec_planet_model.shape,LPF4resmodel.shape)
                        # print(np.ravel(resmodel).shape)
                        # print(HPFmodelH0_list[-1].shape)
                        HPFmodelH0_list.append(np.ravel(resmodel)[:,None])

                    #     res4model = res4model_kl[:,kid]
                    #     resmodel = np.zeros((2*w+1,2*w+1,2*w+1,2*w+1,data_nz))
                    #     for bkg_k in range(2*w+1):
                    #         for bkg_l in range(2*w+1):
                    #             myspec = LPFdata[bkg_k,bkg_l,:]*res4model#*np.nanstd(new_line_list[-1])
                    #             resmodel[bkg_k,bkg_l,bkg_k,bkg_l,:] = myspec
                    #             # import matplotlib.pyplot as plt
                    #             # print(bkg_k,bkg_l)
                    #             # plt.plot(myspec)
                    #             # plt.show()
                    #     HPFmodelH0_list.append(np.reshape(resmodel,((2*w+1)**2,(2*w+1)**2*data_nz)).transpose())

                HPFmodel_H0 = np.concatenate(HPFmodelH0_list,axis=1)

                HPFmodel_H0_cp_4res = copy(HPFmodel_H0)

                # #planet spec extraction
                # if 1:
                #     where_bad_data_cube = np.where(np.isnan(data_badpix))
                #
                #     HPFmodel_H0_4spec_fd = HPFmodel_H0_cp_4res[where_finite_data[0],:]
                #     where_valid_parameters = np.where(np.sum(np.abs(HPFmodel_H0_4spec_fd),axis=0)!=0)
                #     HPFmodel_H0_4spec_fd = HPFmodel_H0_4spec_fd[:,where_valid_parameters[0]]
                #     ravelHPFdata_4spec  = np.ravel(HPFdata)[where_finite_data]
                #     HPFmodel_H0_4spec_fd[np.where(np.isnan(HPFmodel_H0_4spec_fd))]=0
                #
                #     HPFparas_H0,HPFchi2_H0,rank,s = np.linalg.lstsq(HPFmodel_H0_4spec_fd/sigmas_vec[:,None],ravelHPFdata_4spec/sigmas_vec,rcond=None)
                #     # print(HPFparas_H0)
                #
                #     data_model_H0 = np.dot(HPFmodel_H0_cp_4res[:,where_valid_parameters[0]],HPFparas_H0)
                #     where_nan_data_model_H0 = np.where(np.isnan(data_model_H0))
                #     data_model_H0[where_nan_data_model_H0] = 0
                #     canvas_data_model_H0 = np.reshape(data_model_H0,HPFdata.shape)
                #     canvas_residuals = HPFdata-canvas_data_model_H0
                #     canvas_residuals_with_nans = copy(canvas_residuals)
                #     canvas_residuals_with_nans = np.ravel(canvas_residuals_with_nans)
                #     canvas_residuals_with_nans[where_nan_data_model_H0] = np.nan
                #     canvas_residuals_with_nans = np.reshape(canvas_residuals_with_nans,HPFdata.shape)
                #     canvas_residuals_with_nans[where_bad_data_cube] = np.nan
                #
                #     PSF = copy(nospec_planet_model)
                #     PSF = PSF/np.nansum(PSF,axis=(0,1))[None,None,:]
                #     PSF[where_bad_data_cube] = np.nan
                #     estispec_np[0,row,col,:] = np.nanmean(wvs[None,None,:]-data_wvsol_offsets[:,:,None],axis=(0,1))
                #     estispec_np[1,row,col,:] = np.nansum(canvas_residuals_with_nans*PSF,axis=(0,1))/np.nansum(PSF**2,axis=(0,1))
                #     estispec_np[2,row,col,:] = np.nansum(canvas_residuals_with_nans*PSF,axis=(0,1))/np.nansum(PSF**2,axis=(0,1))/tr4planet(wvs)
                #
                #     outres_np[numbasis_id,model_id,0,:,row,col] = np.nanmean(HPFdata,axis=(0,1))
                #     outres_np[numbasis_id,model_id,1,:,row,col] = np.nanmean(canvas_data_model_H0,axis=(0,1))
                #     outres_np[numbasis_id,model_id,2,:,row,col] = 0#final_res
                #     outres_np[numbasis_id,model_id,3,:,row,col] = 0#final_planet
                #     outres_np[numbasis_id,model_id,4,:,row,col] = 0#final_sigmas
                #     outres_np[numbasis_id,model_id,5,:,row,col] = np.nanmean(LPFdata,axis=(0,1))
                #     outres_np[numbasis_id,model_id,6,:,row,col] = np.nanmean(canvas_residuals_with_nans,axis=(0,1))

                HPFmodel_H0 = HPFmodel_H0[where_finite_data[0],:]/sigmas_vec[:,None]

                noplrv_id = np.argmin(np.abs(planetRV_array))
                # print(planetRV_array)
                # print(np.size(planetRV_array))
                # exit()
                for plrv_id in range(np.size(planetRV_array)):
                    # print("coucou6")
                    # print(plrv_id)
                    # try:
                    if 1:
                        c_kms = 299792.458
    
                        HPFmodelH1_list = []
                        for pl_fun_id,planet_partial_template_func in enumerate(planet_partial_template_func_list):
                            planet_model = copy(nospec_planet_model)
                            for bkg_k in range(2*w+1):
                                for bkg_l in range(2*w+1):
                                    # import matplotlib.pyplot as plt
                                    # plt.plot(planet_model[bkg_k,bkg_l,:],label="psf")

                                    wvs4planet_model = wvs*(1-(planetRV_array[plrv_id])/c_kms) \
                                                       -data_wvsol_offsets[bkg_k,bkg_l]

                                    # import matplotlib.pyplot as plt
                                    # plt.subplot(1,2,1)
                                    # plt.plot(wvs,label="wvs")
                                    # plt.plot(wvs4planet_model,label="wvs4planet_model")
                                    # plt.legend()
                                    # plt.subplot(1,2,2)
                                    # plt.plot((wvs-wvs4planet_model)/dwv)
                                    # plt.show()
                                    if pl_fun_id == 0:
                                        planet_model[bkg_k,bkg_l,:] *= planet_partial_template_func(wvs4planet_model) * \
                                            tr4planet(wvs)
                                    else:
                                        planet_model[bkg_k,bkg_l,:] *= planet_partial_template_func(wvs4planet_model)

                                    # import matplotlib.pyplot as plt
                                    # plt.plot(planet_partial_template_func_list[0](wvs4planet_model),label="planet_partial_template_func 1")
                                    # plt.plot(planet_partial_template_func_list[1](wvs4planet_model),label="planet_partial_template_func 2")
                                    # plt.plot(planet_partial_template_func_list[2](wvs4planet_model),label="planet_partial_template_func 3")
                                    # plt.plot(planet_partial_template_func_list[3](wvs4planet_model),label="planet_partial_template_func 4")
                                    # plt.plot(tr4planet(wvs-data_wvsol_offsets[bkg_k,bkg_l]-custom_wvoffset),label="tr4planet")
                                    # plt.legend()
                                    # plt.show()

                            planet_model = planet_model/np.nansum(planet_model)*hr8799_flux*1e-5
                            HPF_planet_model = np.zeros(planet_model.shape)
                            for bkg_k in range(2*w+1):
                                for bkg_l in range(2*w+1):
                                    HPF_planet_model[bkg_k,bkg_l,:]  = LPFvsHPF(planet_model[bkg_k,bkg_l,:] ,cutoff)[1]

                            HPFmodelH1_list.append((HPF_planet_model.ravel())[:,None])

                        HPFmodel_H1only = np.concatenate(HPFmodelH1_list,axis=1)
    
                        HPFmodel_H1only = HPFmodel_H1only[where_finite_data[0],:]/sigmas_vec[:,None]
                        HPFmodel_H1only[np.where(np.isnan(HPFmodel_H1only))] = 0
                        HPFmodel_H0[np.where(np.isnan(HPFmodel_H0))] = 0

                        HPFmodel = np.concatenate([HPFmodel_H1only,HPFmodel_H0],axis=1)

                        # print(np.sum(np.abs(HPFmodel),axis=0))
                        # print(np.sum(np.abs(HPFmodel),axis=0)!=0)
                        # where_valid_parameters = np.where(np.nansum(np.abs(HPFmodel),axis=0)!=0)
                        # HPFmodel = HPFmodel[:,where_valid_parameters[0]]
                        # where_valid_parameters = np.where(np.nansum(np.abs(HPFmodel_H0),axis=0)!=0)
                        # HPFmodel_H0 = HPFmodel_H0[:,where_valid_parameters[0]]
                        where_valid_parameters = np.where(np.nansum(np.abs(HPFmodel)>0,axis=0)>=50)
                        HPFmodel = HPFmodel[:,where_valid_parameters[0]]
                        where_valid_parameters = np.where(np.nansum(np.abs(HPFmodel_H0)>0,axis=0)>50)
                        HPFmodel_H0 = HPFmodel_H0[:,where_valid_parameters[0]]

                        # import matplotlib.pyplot as plt
                        # plt.plot(np.abs(np.linalg.eigvalsh((np.dot(HPFmodel.T,HPFmodel)))),label="H1")
                        # plt.plot(np.abs(np.linalg.eigvalsh((np.dot(HPFmodel_H0.T,HPFmodel_H0)))),label="H0")
                        # plt.yscale("log")
                        # plt.show()

                        HPFparas,HPFchi2,rank,s = np.linalg.lstsq(HPFmodel,ravelHPFdata,rcond=None)
                        HPFparas_H0,HPFchi2_H0,rank,s = np.linalg.lstsq(HPFmodel_H0,ravelHPFdata,rcond=None)
                        # print("H1",HPFparas)
                        # print("H0",HPFparas_H0)
                        # exit()
    
                        data_model = np.dot(HPFmodel,HPFparas)
                        data_model_H0 = np.dot(HPFmodel_H0,HPFparas_H0)
                        deltachi2 = 0#chi2ref-np.sum(ravelHPFdata**2)
                        ravelresiduals = ravelHPFdata-data_model
                        ravelresiduals_H0 = ravelHPFdata-data_model_H0
                        HPFchi2 = np.nansum((ravelresiduals)**2)
                        HPFchi2_H0 = np.nansum((ravelresiduals_H0)**2)


                        # import matplotlib.pyplot as plt
                        # print(HPFchi2,HPFchi2_H0)
                        # plt.figure(1)
                        # print(HPFmodel_H0.shape)
                        # # plt.plot(HPFmodel[:,0],label="data")
                        # # plt.show()
                        # plt.plot(ravelHPFdata,label="data")
                        # for k in np.arange(0,HPFmodel.shape[1]):
                        # # for k in [0,5,5+25]:#np.arange(0,26):
                        # # for k in np.arange(0,1):
                        #     plt.plot(HPFparas[k]*HPFmodel[:,k],label="{0}".format(k),alpha=0.5)
                        # plt.legend()
                        # plt.figure(2)
                        # plt.plot(-ravelresiduals_H0,label="res H0")
                        # plt.plot(-ravelresiduals,label="res")
                        # plt.legend()
                        # plt.show()

                        if plrv_id == noplrv_id:
                            res_ccf = np.correlate(ravelresiduals,ravelresiduals,mode="same")
                            res_ccf_argmax = np.argmax(res_ccf)
                            outautocorrres_np[:,row,col] = res_ccf[(res_ccf_argmax-500):(res_ccf_argmax+500)]

                            where_bad_data_cube = np.where(np.isnan(data_badpix))

                            data_model_H0_allpix = np.dot(HPFmodel_H0_cp_4res[:,where_valid_parameters[0]],HPFparas_H0)
                            where_nan_data_model_H0 = np.where(np.isnan(data_model_H0_allpix))
                            data_model_H0_allpix[where_nan_data_model_H0] = 0
                            canvas_data_model_H0 = np.reshape(data_model_H0_allpix,HPFdata.shape)
                            canvas_residuals = HPFdata-canvas_data_model_H0
                            canvas_residuals_with_nans = copy(canvas_residuals)
                            canvas_residuals_with_nans = np.ravel(canvas_residuals_with_nans)
                            canvas_residuals_with_nans[where_nan_data_model_H0] = np.nan
                            canvas_residuals_with_nans = np.reshape(canvas_residuals_with_nans,HPFdata.shape)
                            canvas_residuals_with_nans[where_bad_data_cube] = np.nan

                            PSF = copy(nospec_planet_model)
                            PSF = PSF/np.nansum(PSF,axis=(0,1))[None,None,:]
                            PSF[where_bad_data_cube] = np.nan
                            estispec_np[0,row,col,:] = np.nanmean(wvs[None,None,:]-data_wvsol_offsets[:,:,None],axis=(0,1))
                            estispec_np[1,row,col,:] = np.nansum(canvas_residuals_with_nans*PSF,axis=(0,1))/np.nansum(PSF**2,axis=(0,1))
                            estispec_np[2,row,col,:] = tr4planet(wvs)
                            if fake_paras is not None:
                                estispec_np[3,row,col,:] = np.nansum(HPF_fake*fake_paras["contrast"],axis=(0,1))
                                estispec_np[4,row,col,:] = np.nan#np.nansum(HPF_fake*fake_paras["contrast"],axis=(0,1))/tr4planet(wvs)
                            else:
                                estispec_np[3,row,col,:] = np.nan
                                estispec_np[4,row,col,:] = np.nan

                            outres_np[numbasis_id,model_id,0,:,row,col] = np.nanmean(HPFdata,axis=(0,1))
                            outres_np[numbasis_id,model_id,1,:,row,col] = np.nanmean(canvas_data_model_H0,axis=(0,1))
                            outres_np[numbasis_id,model_id,2,:,row,col] = 0#final_res
                            outres_np[numbasis_id,model_id,3,:,row,col] = 0#final_planet
                            outres_np[numbasis_id,model_id,4,:,row,col] = 0#final_sigmas
                            outres_np[numbasis_id,model_id,5,:,row,col] = np.nanmean(LPFdata,axis=(0,1))
                            outres_np[numbasis_id,model_id,6,:,row,col] = np.nanmean(canvas_residuals_with_nans,axis=(0,1))

    
                        Npixs_HPFdata = HPFmodel.shape[0]
                        minus2logL_HPF = Npixs_HPFdata*(1+np.log(HPFchi2/Npixs_HPFdata)+logdet_Sigma+np.log(2*np.pi))
                        minus2logL_HPF_H0 = Npixs_HPFdata*(1+np.log(HPFchi2_H0/Npixs_HPFdata)+logdet_Sigma+np.log(2*np.pi))
                        AIC_HPF = 2*(HPFmodel.shape[-1])+minus2logL_HPF
                        AIC_HPF_H0 = 2*(HPFmodel_H0.shape[-1])+minus2logL_HPF_H0

                        covphi =  HPFchi2/Npixs_HPFdata*np.linalg.inv(np.dot(HPFmodel.T,HPFmodel))
                        slogdet_icovphi0 = np.linalg.slogdet(np.dot(HPFmodel.T,HPFmodel))

                        # delta AIC ~ likelihood ratio
                        if HPFparas[0]>0:
                            output_maps_np[numbasis_id,model_id,0,row,col,plrv_id] = AIC_HPF_H0-AIC_HPF
                        else:
                            # output_maps_np[numbasis_id,model_id,0,row,col,plrv_id] = AIC_HPF_H0-AIC_HPF
                            output_maps_np[numbasis_id,model_id,0,row,col,plrv_id] = 0
                        # AIC for planet + star model
                        output_maps_np[numbasis_id,model_id,1,row,col,plrv_id] = AIC_HPF
                        # AIC for star model only
                        output_maps_np[numbasis_id,model_id,2,row,col,plrv_id] = AIC_HPF_H0
                        # Chi2 of the stamp
                        output_maps_np[numbasis_id,model_id,3,row,col,plrv_id] = HPFchi2
                        # Size of the data
                        output_maps_np[numbasis_id,model_id,4,row,col,plrv_id] = Npixs_HPFdata
                        # number of parameters of the model
                        output_maps_np[numbasis_id,model_id,5,row,col,plrv_id] = HPFmodel.shape[-1]
                        # estimated scaling factor of the covariance matrix of the data
                        output_maps_np[numbasis_id,model_id,6,row,col,plrv_id] = HPFchi2/Npixs_HPFdata
                        #
                        output_maps_np[numbasis_id,model_id,7,row,col,plrv_id] = logdet_Sigma
                        output_maps_np[numbasis_id,model_id,8,row,col,plrv_id] =  slogdet_icovphi0[0]*slogdet_icovphi0[1]
                        # marginalized posterior
                        output_maps_np[numbasis_id,model_id,9,row,col,plrv_id] = -0.5*logdet_Sigma-0.5*slogdet_icovphi0[1]- (Npixs_HPFdata-HPFmodel.shape[-1]+2-1)/(2)*np.log(HPFchi2+deltachi2)
                        for plmod_id in range(len(planet_partial_template_func_list)):
                            # print(plmod_id)
                            # SNR
                            output_maps_np[numbasis_id,model_id,10+plmod_id*3,row,col,plrv_id] = HPFparas[plmod_id]/np.sqrt(np.abs(covphi[plmod_id,plmod_id]))
                            # estimated planet to star flux ratio
                            output_maps_np[numbasis_id,model_id,10+plmod_id*3+1,row,col,plrv_id] = HPFparas[plmod_id]
                            # error bar on estimated planet to star flux ratio
                            output_maps_np[numbasis_id,model_id,10+plmod_id*3+2,row,col,plrv_id] = np.sign(covphi[plmod_id,plmod_id])*np.sqrt(np.abs(covphi[plmod_id,plmod_id]))
                        # print(output_maps_np[numbasis_id,model_id,:,row,col,plrv_id])
                        # print(HPFchi2,deltachi2)
                        # exit()
                    # except:
                    #     pass
        # #remove
        # plt.legend()
        # plt.show()
        #             print("jb here0")
        #             exit()
    #             print("jb here1")
    #             exit()
    #         print("jb here2")
    #         exit()
    #     print("jb here3")
    #     exit()
    # print("jb here4")
    # exit()
    return

def _task_convolve_spectrum(paras):
    indices,wvs,spectrum,R = paras

    conv_spectrum = np.zeros(np.size(indices))
    dwvs = wvs[1::]-wvs[0:(np.size(wvs)-1)]
    med_dwv = np.median(dwvs)
    for l,k in enumerate(indices):
        pwv = wvs[k]
        FWHM = pwv/R
        sig = FWHM/(2*np.sqrt(2*np.log(2)))
        w = int(np.round(sig/med_dwv*10.))
        stamp_spec = spectrum[np.max([0,k-w]):np.min([np.size(spectrum),k+w])]
        stamp_wvs = wvs[np.max([0,k-w]):np.min([np.size(wvs),k+w])]
        stamp_dwvs = stamp_wvs[1::]-stamp_wvs[0:(np.size(stamp_spec)-1)]
        gausskernel = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(stamp_wvs-pwv)**2/sig**2)
        conv_spectrum[l] = np.sum(gausskernel[1::]*stamp_spec[1::]*stamp_dwvs)
    return conv_spectrum

def convolve_spectrum(wvs,spectrum,R,mypool=None):
    if mypool is None:
        return _task_convolve_spectrum((np.arange(np.size(spectrum)).astype(np.int),wvs,spectrum,R))
    else:
        conv_spectrum = np.zeros(spectrum.shape)

        chunk_size=100
        N_chunks = np.size(spectrum)//chunk_size
        indices_list = []
        for k in range(N_chunks-1):
            indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
        indices_list.append(np.arange((N_chunks-1)*chunk_size,np.size(spectrum)).astype(np.int))
        outputs_list = mypool.map(_task_convolve_spectrum, zip(indices_list,
                                                               itertools.repeat(wvs),
                                                               itertools.repeat(spectrum),
                                                               itertools.repeat(R)))
        for indices,out in zip(indices_list,outputs_list):
            conv_spectrum[indices] = out

        return conv_spectrum

def _spline_psf_model(paras):
    psfs,xs,ys,xvec,yvec,chunk_id = paras
    normalized_psfs_func_list = []
    for wv_index in range(psfs.shape[1]):
        if 0:#np.isnan(psf_func(0,0)[0,0]):
            model_psf = psfs[:,wv_index,:,:]
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(1)
            ax = fig.add_subplot(111,projection="3d")
            for k,color in zip(range(model_psf.shape[0]),["pink","blue","green","purple","orange"]):
                ax.scatter(xs[k].ravel(),ys[k].ravel(),model_psf[k].ravel(),c=color)
        model_psf = psfs[:,wv_index,:,:].ravel()
        where_nans = np.where(np.isfinite(model_psf))
        psf_func = interpolate.LSQBivariateSpline(xs.ravel()[where_nans],ys.ravel()[where_nans],model_psf[where_nans],xvec,yvec,kx=3,ky=3,eps=0.01)
        if 0:
            print(psf_func(0,0))
            x_psf_vec, y_psf_vec = np.arange(2*nx_psf * 1.)/2.-nx_psf//2, np.arange(2*ny_psf* 1.)/2.-ny_psf//2
            x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
            ax.scatter(x_psf_grid.ravel(),y_psf_grid.ravel(),psf_func(x_psf_vec,y_psf_vec).transpose().ravel(),c="red")
            plt.show()
        normalized_psfs_func_list.append(psf_func)
    # print(len(normalized_psfs_func_list))
    return chunk_id,normalized_psfs_func_list

def return_64x19(cube):
    #cube should be nz,ny,nx
    if np.size(cube.shape) == 3:
        _,ny,nx = cube.shape
    else:
        ny,nx = cube.shape
    onesmask = np.ones((64,19))
    if (ny != 64 or nx != 19):
        mask = copy(cube).astype(np.float)
        mask[np.where(mask==0)]=np.nan
        mask[np.where(np.isfinite(mask))]=1
        if np.size(cube.shape) == 3:
            im = np.nansum(mask,axis=0)
        else:
            im = mask
        ccmap =np.zeros((3,3))
        for dk in range(3):
            for dl in range(3):
                ccmap[dk,dl] = np.nansum(im[dk:np.min([dk+64,ny]),dl:np.min([dl+19,nx])]*onesmask[0:(np.min([dk+64,ny])-dk),0:(np.min([dl+19,nx])-dl)])
        dk,dl = np.unravel_index(np.argmax(ccmap),ccmap.shape)
        if np.size(cube.shape) == 3:
            return cube[:,dk:(dk+64),dl:(dl+19)]
        else:
            return cube[dk:(dk+64),dl:(dl+19)]
    else:
        return cube

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


    ##############################
    ## Variable parameters
    ##############################
    print(len(sys.argv))
    if len(sys.argv) == 1:
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
        date = "200729"
        # planet = "HR_8799_d"
        # date = "150720"
        # date = "150722"
        # date = "150723"
        # date = "150828"
        # date = "200729"
        # date = "200730"
        # date = "200731"
        # planet = "51_Eri_b"
        # date = "171103"
        # date = "171104"
        # planet = "kap_And"
        # date = "161106"
        # IFSfilter = "Kbb"
        IFSfilter = "Kbb"
        # IFSfilter = "Jbb" # "Kbb" or "Hbb"
        scale = "020"
        # scale = "035"

        inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/"
        # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/20190520_LPF/"
        # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/20190906_HPF_restest2/"
        # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/20190923_HPF_restest2/"
        # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/20191120_new_resmodel/"
        # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/20200213_molectest/"
        # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/20190305_HPF_only_noperscor/"
        # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/20190228_mol_temp/"
        outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/20200731_livereduc/"

        # inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb_pairsub/"
        # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb_pairsub/20190228_HPF_only/"

        print(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
        filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
        filelist.sort()
        filelist = [filelist[-1],]
        print(filelist)
        # exit()
        # print(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_020.fits"))
        # filelist = filelist[1:]
        # filelist = filelist[len(filelist)-3:len(filelist)-2]

        res_numbasis = 1
        numthreads = 10
        planet_search = True
        debug_paras = True
        plot_transmissions = False
        plt_psfs = False
        plot_persistence = False
        planet_model_string = "model"
        # planet_model_string = "CO2 CO H2O CH4"#"CO"#
        # planet_model_string = "CO2 CO H2O CH4 joint"
        # planet_model_string = "CO joint"
        inject_fakes = False

        osiris_data_dir = "/data/osiris_data"
    else:
        osiris_data_dir = sys.argv[1]
        inputDir = sys.argv[2]
        outputdir = sys.argv[3]
        filename = sys.argv[4]
        numthreads = int(sys.argv[5])
        planet_search = bool(int(sys.argv[6]))
        planet_model_string = sys.argv[7]
        debug_paras = bool(int(sys.argv[8]))
        res_numbasis = int(sys.argv[9])
        inject_fakes = bool(int(sys.argv[10]))

        filelist = [filename]
        IFSfilter = filename.split("_")[-2]
        date = os.path.basename(filename).split("_")[0].replace("s","")

        plot_transmissions = False
        plt_psfs = False
        plot_persistence = False
        #nice -n 15 /home/anaconda3/bin/python ./reduce_HPFonly_diagcov.py /data/osiris_data /data/osiris_data/HR_8799_c/20100715/reduced_jb/ /data/osiris_data/HR_8799_c/20100715/reduced_jb/20190308_HPF_only_sherlock_test/ /data/osiris_data/HR_8799_c/20100715/reduced_jb/s100715_a011001_Kbb_020.fits 20 1 'CO test' 1
        #CO CH4 CO2 H2O
        #nice -n 15 /home/anaconda3/bin/python ./reduce_HPFonly_diagcov.py /data/osiris_data /data/osiris_data/HR_8799_b/20180722/reduced_jb/ /data/osiris_data/HR_8799_b/20180722/reduced_jb/20190326_HPF_only_sherlock_test/ /data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a033002_Kbb_035.fits 20 1 'CO' 1

    osiris_data_dir0 = copy(osiris_data_dir)
    inputDir0 = copy(inputDir)
    outputdir0 = copy(outputdir)
    numthreads0 = copy(numthreads)
    planet_search0 = copy(planet_search)
    planet_model_string0 = copy(planet_model_string)
    debug_paras0 = copy(debug_paras)
    res_numbasis0 = copy(res_numbasis)
    filelist0 = copy(filelist)
    IFSfilter0 = copy(IFSfilter)
    date0 = copy(date)
    plot_transmissions0 = copy(plot_transmissions)
    plt_psfs0 = copy(plt_psfs)
    plot_persistence0 = copy(plot_persistence)
    inject_fakes0 = copy(inject_fakes)
    for res_it in range(2):

        osiris_data_dir = copy(osiris_data_dir0)
        inputDir = copy(inputDir0)
        outputdir = copy(outputdir0)
        numthreads = copy(numthreads0)
        planet_search = copy(planet_search0)
        planet_model_string = copy(planet_model_string0)
        if res_it == 0:
            # continue
            if res_numbasis == 0:
                continue
            debug_paras = True
            res_numbasis = 0
            inject_fakes = False
        else:
            debug_paras = copy(debug_paras0)
            res_numbasis = copy(res_numbasis0)
            inject_fakes = copy(inject_fakes0)
        # print(res_numbasis)
        # exit()
        filelist = copy(filelist0)
        IFSfilter = copy(IFSfilter0)
        date = copy(date0)
        plot_transmissions = copy(plot_transmissions0)
        plt_psfs = copy(plt_psfs0)
        plot_persistence = copy(plot_persistence0)

        phoenix_folder = os.path.join(osiris_data_dir,"phoenix")
        planet_template_folder = os.path.join(osiris_data_dir,"planets_templates")
        molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
        sky_transmission_folder = os.path.join(osiris_data_dir,"sky_transmission")
        ref_star_folder = os.path.join(os.path.dirname(filelist[0]),"..","reduced_telluric_jb")
        if res_numbasis == 0 or (not inject_fakes):
            fileinfos_filename = os.path.join(inputDir,"..","..","fileinfos_Kbb_jb.csv")
        else:
            fileinfos_filename = os.path.join(inputDir,"..","..","fileinfos_Kbb_jb_kl{0}.csv".format(res_numbasis))
        fileinfos_refstars_filename = os.path.join(osiris_data_dir,"fileinfos_refstars_jb.csv")

        if "HR_8799_b" in filelist[0]:
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "HR8799b_"+IFSfilter[0:1]+"_3Oct2018.save")
            RV4fakes = -9.06
        if "HR_8799_c" in filelist[0]:
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save")
            RV4fakes = -11.13
        if "HR_8799_d" in filelist[0]:
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save")
            RV4fakes = -14
        if "HR_8799" in filelist[0]:
            phoenix_model_host_filename = glob.glob(os.path.join(phoenix_folder,"HR_8799"+"*.fits"))[0]
            if IFSfilter == "Jbb":
                host_mag = 5.383
            elif IFSfilter == "Hbb":
                host_mag = 5.280
            elif IFSfilter == "Kbb":
                host_mag = 5.240
            else:
                raise("IFS filter name unknown")
            host_type = "F0"
            host_rv = -12.6 #+-1.4
            host_limbdark = 0.5
            host_vsini = 49 # true = 49
            star_name = "HR_8799"
        if "51_Eri_b" in filelist[0]:
            phoenix_model_host_filename = glob.glob(os.path.join(phoenix_folder,"51_Eri"+"*.fits"))[0]
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "51Eri_b_highres_template.save")
            if IFSfilter == "Jbb":
                host_mag = 4.744
            elif IFSfilter == "Hbb":
                host_mag = 4.770
            elif IFSfilter == "Kbb":
                host_mag = 4.537
            else:
                raise("IFS filter name unknown")
            host_type = "F0"
            host_rv = 12.6 #+-0.3
            host_limbdark = 0.5
            host_vsini = 80
            star_name = "51_Eri"
            RV4fakes = -12.6
        if "kap_And" in filelist[0]:
            phoenix_model_host_filename = glob.glob(os.path.join(phoenix_folder,"kap_And"+"*.fits"))[0]
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "KapAnd_lte19-3.50-0.0.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019.7.save")
            if IFSfilter == "Jbb":
                host_mag = 4.29
            elif IFSfilter == "Hbb":
                host_mag = 4.31
            elif IFSfilter == "Kbb":
                host_mag = 4.34
            else:
                raise("IFS filter name unknown")
            host_type = "A0"
            host_rv = -12.7 #+-0.8
            host_limbdark = 0.5
            host_vsini = 150 #unknown
            star_name = "kap_And"
            RV4fakes = -13.9


        for filename in filelist:
            print("Processing "+filename)

            ##############################
            ## Read OSIRIS spectral cube
            ##############################
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
            ny,nx,nz = imgs.shape
            init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
            dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
            wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)
            # print(wvs[0],wvs[-1])
            # exit()

            ##############################
            ## Fixed parameters
            ##############################
            if IFSfilter=="Kbb": #Kbb 1965.0 0.25
                CRVAL1 = 1965.
                CDELT1 = 0.25
                nl=1665
                R0=4000
            elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
                CRVAL1 = 1473.
                CDELT1 = 0.2
                nl=1651
                R0=4000#5000
            elif IFSfilter=="Jbb": #Hbb 1651 1473.0 0.2
                CRVAL1 = 1180.
                CDELT1 = 0.15
                nl=1574
                R0=4000
            dwv = CDELT1/1000.

            debug = False
            if debug:
                planet_search = False
            model_based_sky_trans = False
            # if "HR_8799_d" in filename and ("20130727" in filename or "20150720" in filename or "20150722" in filename or "20150723" in filename):
            if len(glob.glob(os.path.join(inputDir,"..","master_wvshifts_"+IFSfilter+".fits"))) == 1:
                use_wvsol_offsets = True
            else:
                use_wvsol_offsets = False
            use_R_calib = False
            mask_starline = False
            model_persistence = False
            if "s101104_a014" in filename or "s101104_a016" in filename:
                mask_20101104_artifact = True
            else:
                mask_20101104_artifact = False
            if not model_based_sky_trans and use_R_calib:
                print("incompatible modes")
                exit()
            if "model" in planet_model_string:
                use_molecular_template = False
            else:
                use_molecular_template = True
                molecules_list_ref = ["CO2","CO","H2O","CH4"]
                molecules_list = []
                for molecule in molecules_list_ref:
                    if molecule in planet_model_string:
                        molecules_list.append(molecule)
                        planet_model_string = planet_model_string.replace(molecule,"")
            if "joint" in planet_model_string:
                joint_fit = True
            else:
                joint_fit = False
                # print(molecules_list)
                # exit(0)
            pairsub = "pairsub" in inputDir

            numbasis_list = [1]
            # if planet == "d" or (IFSfilter == "Hbb" and date == "171103"):
            #     numbasis_list = [1,2]#[1,2,3]

            padding = 5
            nan_mask_boxsize=3
            cutoff = 40
            dtype = ctypes.c_double
            # dtype = ctypes.c_longdouble

            scale = filename.split("_")[-1].split(".fits")[0]
            if mask_starline:
                suffix = "HPF_cutoff{0}_sherlock_v1_starline".format(cutoff)
            else:
                suffix = "HPF_cutoff{0}_sherlock_v1".format(cutoff)

            c_kms = 299792.458
            dprv = 3e5*dwv/(init_wv+dwv*nz//2) # 38.167938931297705

            ##############################
            ## Dependent parameters
            ##############################
            specpool = mp.Pool(processes=numthreads)

            if use_R_calib:
                R_list = [0]
                R_calib_arr = np.zeros((ny+2*padding,nx+2*padding))+R0
            else:
                R_calib_arr = np.zeros((ny+2*padding,nx+2*padding))
                if debug_paras:
                    R_list = [R0]
                else:
                    R_list = [R0]#[2000,2500,3000,3500,4000,4500,5000]

            if use_wvsol_offsets:
                wvsol_offsets_filename = os.path.join(inputDir,"..","master_wvshifts_"+IFSfilter+".fits")
                hdulist = pyfits.open(wvsol_offsets_filename)
                wvsol_offsets = hdulist[0].data
                hdulist.close()
            else:
                wvsol_offsets = np.zeros((ny,nx))
            wvsol_offsets = np.pad(wvsol_offsets,((padding,padding),(padding,padding)),mode="constant",constant_values=0)

            ## file specific info
            with open(fileinfos_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=';')
                list_table = list(csv_reader)
                colnames = list_table[0]
                N_col = len(colnames)
                list_data = list_table[1::]
                N_lines =  len(list_data)

            filename_id = colnames.index("filename")
            filelist = [os.path.basename(item[filename_id]) for item in list_data]
            fileid = filelist.index(os.path.basename(filename))
            fileitem = list_data[fileid]
            for colname,it in zip(colnames,fileitem):
                print(colname+": "+it)

            baryrv_id = colnames.index("barycenter rv")
            host_bary_rv = -float(fileitem[baryrv_id])/1000
            # print(host_bary_rv)
            # exit()

            phoenix_db_folder = os.path.join(osiris_data_dir,"phoenix","PHOENIX-ACES-AGSS-COND-2011")
            if 1:
                splitpostfilename = os.path.basename(filelist[0]).split("_")
                imtype = "science"
                # print(date,star_name)
                # exit()
                # phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011_R{0}.fits".format(R0))
                # with pyfits.open(phoenix_wv_filename) as hdulist:
                #     phoenix_wvs = hdulist[0].data
                if len(glob.glob(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_rv_samples.fits".format(star_name,IFSfilter,date,imtype)))) >0:
                    print(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_rv_samples.fits".format(star_name,IFSfilter,date,imtype)))
                    hdulist = pyfits.open(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_rv_samples.fits".format(star_name,IFSfilter,date,imtype)))
                    rv_samples = hdulist[0].data
                    hdulist = pyfits.open(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_vsini_samples.fits".format(star_name,IFSfilter,date,imtype)))
                    vsini_samples = hdulist[0].data
                    with open(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_models.txt".format(star_name,IFSfilter,date,imtype)), 'r') as txtfile:
                        grid_refstar_filelist = [s.strip().replace(os.path.dirname(s),osiris_data_dir) for s in txtfile.readlines()]
                else:
                    print(glob.glob(os.path.join(osiris_data_dir,"stellar_fits","{0}_*_*_{1}_rv_samples.fits".format(star_name,imtype)))[0])
                    hdulist = pyfits.open(glob.glob(os.path.join(osiris_data_dir,"stellar_fits","{0}_*_*_{1}_rv_samples.fits".format(star_name,imtype)))[0])
                    rv_samples = hdulist[0].data
                    hdulist = pyfits.open(glob.glob(os.path.join(osiris_data_dir,"stellar_fits","{0}_*_*_{1}_vsini_samples.fits".format(star_name,imtype)))[0])
                    vsini_samples = hdulist[0].data
                    print(os.path.join(osiris_data_dir,"stellar_fits","{0}_*_*_{1}_models.fits".format(star_name,imtype)))
                    print(glob.glob(os.path.join(osiris_data_dir,"stellar_fits","{0}_*_*_{1}_models.txt".format(star_name,imtype))))
                    with open(glob.glob(os.path.join(osiris_data_dir,"stellar_fits","{0}_*_*_{1}_models.txt".format(star_name,imtype)))[0], 'r') as txtfile:
                        grid_refstar_filelist = [s.strip().replace(os.path.dirname(s),osiris_data_dir) for s in txtfile.readlines()]
                post_filename = os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_posterior.fits".format(star_name,IFSfilter,date,imtype))
                if len(glob.glob(post_filename))>0:
                    print(post_filename)
                    hdulist = pyfits.open(post_filename)
                else:
                    print(os.path.join(osiris_data_dir,"stellar_fits","{0}_combined_posterior.fits".format(star_name)))
                    hdulist = pyfits.open(os.path.join(osiris_data_dir,"stellar_fits","{0}_combined_posterior.fits".format(star_name)))
                # exit()
                posterior = hdulist[0].data[0]
                logpost_arr = hdulist[0].data[1]
                chi2_arr = hdulist[0].data[2]
                posterior_rv_vsini = np.nansum(posterior,axis=0)
                posterior_model = np.nansum(posterior,axis=(1,2))
                rv_posterior = np.nansum(posterior_rv_vsini,axis=1)
                vsini_posterior = np.nansum(posterior_rv_vsini,axis=0)
                bestrv,_,_,bestrv_merr,bestrv_perr,_ = get_err_from_posterior(rv_samples,rv_posterior)
                bestrv_merr = np.abs(bestrv_merr)
                bestvsini,_,_,bestvsini_merr,bestvsini_perr,_ = get_err_from_posterior(vsini_samples,vsini_posterior)
                bestvsini_merr = np.abs(bestvsini_merr)
                best_model_id = np.argmax(posterior_model)

                phoenix_model_host_filename = grid_refstar_filelist[best_model_id]
                host_rv = bestrv
                host_vsini = bestvsini

            if planet_search:
                suffix = suffix+"_search"
                if debug_paras:
                    planetRV_array = np.array([host_bary_rv+host_rv])
                    # planetRV_array = np.array([0,19,38,47])
                    # print(host_bary_rv-10)
                    # exit()
                    # planetRV_array = np.concatenate([np.arange(-2*dprv,2*dprv,dprv/10)])
                else:
                    planetRV_array = np.concatenate([np.arange(-2*dprv,2*dprv,dprv/100),np.arange(-100*dprv,100*dprv,dprv)])
                plcen_k,plcen_l = np.nan,np.nan
            else:
                suffix = suffix+"_centroid"

                if debug_paras:
                    planetRV_array = np.array([host_bary_rv+host_rv,])
                    dl_grid,dk_grid = np.array([[0]]),np.array([[0]])
                    # kcen_id = colnames.index("kcen")
                    # lcen_id = colnames.index("lcen")
                    # plcen_k,plcen_l = float(fileitem[kcen_id]),float(fileitem[lcen_id])
                    plcen_k,plcen_l = 32-10,-35.79802955665025+46.8
                    # plcen_k,plcen_l = 26,6#44,8
                else:
                    planetRV_array = np.arange(-3*dprv,3*dprv,dprv/100)
                    dl_grid,dk_grid = np.meshgrid(np.linspace(-1.,1.,2*20+1),np.linspace(-1.,1.,2*20+1))

                    cen_filename_id = colnames.index("cen filename")
                    kcen_id = colnames.index("kcen")
                    lcen_id = colnames.index("lcen")
                    rvcen_id = colnames.index("RVcen")
                    plcen_k,plcen_l = float(fileitem[kcen_id]),float(fileitem[lcen_id])

            plcen_k,plcen_l = plcen_k+padding,plcen_l+padding


            ##############################
            ## Planet spectrum model
            ##############################
            planet_model_func_table = []
            for R in R_list:
                planet_partial_template_func_list = []
                if use_molecular_template: # iterate over molecular templates
                    if use_R_calib:
                        print("use_R_calib not comaptible with molecular template")
                        exit()
                    for molecule in molecules_list:
                        suffix = suffix+"_"+molecule
                        print(molecule)
                        travis_mol_filename=os.path.join(molecular_template_folder,
                                                      "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7")
                        travis_mol_filename_D2E=os.path.join(molecular_template_folder,
                                                      "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7_D2E")
                        mol_template_filename=travis_mol_filename+"_gaussconv_R{0}_{1}.csv".format(R,IFSfilter)

                        # file1 = open(travis_mol_filename_D2E, 'r')
                        # file2 = open(travis_mol_filename_D2E.replace("7_D2E","7_D2E2"), 'w')
                        #
                        # for k,line in enumerate(file1):
                        #     if "0 9" in line[88:91]:
                        #         file2.write(line.replace("0 9","009"))
                        #     else:
                        #         file2.write(line)
                        # exit()

                        if len(glob.glob(mol_template_filename)) == 0:
                            data = np.loadtxt(travis_mol_filename_D2E)
                            print(data.shape)
                            wmod = data[:,0]/10000.
                            wmod_argsort = np.argsort(wmod)
                            wmod= wmod[wmod_argsort]
                            crop_moltemp = np.where((wmod>wvs[0]-(wvs[-1]-wvs[0])/2)*(wmod<wvs[-1]+(wvs[-1]-wvs[0])/2))
                            wmod = wmod[crop_moltemp]
                            mol_temp = data[wmod_argsort,1][crop_moltemp]
                            mol_temp = 10**(mol_temp-np.max(mol_temp))

                            print("convolving: "+mol_template_filename)
                            planet_convspec = convolve_spectrum(wmod,mol_temp,R,specpool)

                            # import matplotlib.pyplot as plt
                            # plt.plot(wmod,planet_convspec)#,data[::100,1])
                            # print(mol_temp.shape)
                            # plt.show()
                            # exit()

                            with open(mol_template_filename, 'w+') as csvfile:
                                csvwriter = csv.writer(csvfile, delimiter=' ')
                                csvwriter.writerows([["wvs","spectrum"]])
                                csvwriter.writerows([[a,b] for a,b in zip(wmod,planet_convspec)])

                        with open(mol_template_filename, 'r') as csvfile:
                            csv_reader = csv.reader(csvfile, delimiter=' ')
                            list_starspec = list(csv_reader)
                            oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                            col_names = oriplanet_spec_str_arr[0]
                            oriplanet_spec = oriplanet_spec_str_arr[1::3,1].astype(np.float)
                            oriplanet_spec_wvs = oriplanet_spec_str_arr[1::3,0].astype(np.float)
                            where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                            oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                            planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
                            planet_partial_template_func_list.append(planet_spec_func)
                        # import matplotlib.pyplot as plt
                        # plt.plot(wvs,planet_spec_func(wvs))#,data[::100,1])
                        # plt.show()
                    exit()
                    if len(molecules_list) >= 2 or joint_fit:
                        suffix = suffix+"_"+"joint"
                        # print("uh...")
                        # exit()
                        planet_template_filename=travis_spec_filename.replace(".save",
                                                                              "_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))

                        if use_R_calib:
                            travis_spectrum = scio.readsav(travis_spec_filename)
                            ori_planet_spec = np.array(travis_spectrum["fmod"])
                            ori_planet_convspec = np.array(travis_spectrum["fmods"])
                            wmod = np.array(travis_spectrum["wmod"])/1.e4
                            planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)

                            planet_partial_template_func_list.append(planet_spec_func)
                        else:
                            if len(glob.glob(planet_template_filename)) == 0:
                                travis_spectrum = scio.readsav(travis_spec_filename)
                                if "HR8799" in os.path.basename(travis_spec_filename):
                                    ori_planet_spec = np.array(travis_spectrum["fmod"])
                                    ori_planet_convspec = np.array(travis_spectrum["fmods"])
                                    wmod = np.array(travis_spectrum["wmod"])/1.e4
                                elif "51Eri" in os.path.basename(travis_spec_filename):
                                    ori_planet_spec = np.array(travis_spectrum["flux"])
                                    wmod = np.array(travis_spectrum["wave"])
                                elif "KapAnd" in os.path.basename(travis_spec_filename):
                                    ori_planet_spec = np.array(travis_spectrum["f"])
                                    wmod = np.array(travis_spectrum["w"])/1.e4
                                print("convolving: "+planet_template_filename)
                                planet_convspec = convolve_spectrum(wmod,ori_planet_spec,R,specpool)

                                with open(planet_template_filename, 'w+') as csvfile:
                                    csvwriter = csv.writer(csvfile, delimiter=' ')
                                    csvwriter.writerows([["wvs","spectrum"]])
                                    csvwriter.writerows([[a,b] for a,b in zip(wmod,planet_convspec)])

                            with open(planet_template_filename, 'r') as csvfile:
                                csv_reader = csv.reader(csvfile, delimiter=' ')
                                list_starspec = list(csv_reader)
                                oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                                col_names = oriplanet_spec_str_arr[0]
                                oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
                                oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
                                where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                                oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                                continuum = LPFvsHPF(oriplanet_spec,cutoff)[0]
                                continuum_func = interp1d(oriplanet_spec_wvs,continuum,bounds_error=False,fill_value=np.nan)

                        # # continuum = np.nanmax(np.array([temp_func(oriplanet_spec_wvs) for temp_func in planet_partial_template_func_list]),axis=0)
                        # import matplotlib.pyplot as plt
                        # plt.plot(oriplanet_spec_wvs,continuum)#,data[::100,1])
                        # plt.show()
                        planet_partial_template_func_list.insert(0,interp1d(oriplanet_spec_wvs,continuum,bounds_error=False,fill_value=np.nan))
                else:
                    planet_template_filename=travis_spec_filename.replace(".save",
                                                                          "_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))

                    if use_R_calib:
                        travis_spectrum = scio.readsav(travis_spec_filename)
                        ori_planet_spec = np.array(travis_spectrum["fmod"])
                        ori_planet_convspec = np.array(travis_spectrum["fmods"])
                        wmod = np.array(travis_spectrum["wmod"])/1.e4
                        planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)

                        planet_partial_template_func_list.append(planet_spec_func)
                    else:
                        if len(glob.glob(planet_template_filename)) == 0:
                            travis_spectrum = scio.readsav(travis_spec_filename)
                            if "HR8799" in os.path.basename(travis_spec_filename):
                                ori_planet_spec = np.array(travis_spectrum["fmod"])
                                ori_planet_convspec = np.array(travis_spectrum["fmods"])
                                wmod = np.array(travis_spectrum["wmod"])/1.e4
                            elif "51Eri" in os.path.basename(travis_spec_filename):
                                ori_planet_spec = np.array(travis_spectrum["flux"])
                                wmod = np.array(travis_spectrum["wave"])
                            elif "KapAnd" in os.path.basename(travis_spec_filename):
                                ori_planet_spec = np.array(travis_spectrum["f"])
                                wmod = np.array(travis_spectrum["w"])/1.e4
                            print("convolving: "+planet_template_filename)
                            planet_convspec = convolve_spectrum(wmod,ori_planet_spec,R,specpool)

                            with open(planet_template_filename, 'w+') as csvfile:
                                csvwriter = csv.writer(csvfile, delimiter=' ')
                                csvwriter.writerows([["wvs","spectrum"]])
                                csvwriter.writerows([[a,b] for a,b in zip(wmod,planet_convspec)])

                        with open(planet_template_filename, 'r') as csvfile:
                            csv_reader = csv.reader(csvfile, delimiter=' ')
                            list_starspec = list(csv_reader)
                            oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                            col_names = oriplanet_spec_str_arr[0]
                            oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
                            oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
                            where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                            oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                            planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
                            # import matplotlib.pyplot as plt
                            # plt.plot(planet_spec_func(wvs))
                            # plt.show()
                            planet_partial_template_func_list.append(planet_spec_func)
                planet_model_func_table.append(planet_partial_template_func_list)

            # import matplotlib.pyplot as plt
            # plt.plot(wvs,planet_model_func_table[0][0](wvs*(1-(-1*38)/c_kms)))
            # plt.plot(wmod,ori_planet_spec)
            # plt.show()

            # import matplotlib.pyplot as plt
            # tmp = planet_model_func_table[0][0](wvs*(1-(-1*38)/c_kms))
            # for k,planet_partial_template_func in enumerate(planet_model_func_table[0][1:2]):
            #     # print(k)
            #     # tmp /= planet_partial_template_func(wvs*(1-(-1*38)/c_kms))
            #     plt.plot(wvs,planet_partial_template_func(wvs*(1-(-1*38)/c_kms)))
            # plt.plot(wvs,tmp)
            # plt.legend()
            # plt.show()
            #
            # import matplotlib.pyplot as plt
            # for k,planet_partial_template_func in enumerate(planet_model_func_table[0]):
            #     plt.plot(wvs,LPFvsHPF(planet_partial_template_func(wvs*(1-(-1*38)/c_kms)),cutoff)[1],label="{0}".format(k))
            # plt.legend()
            # plt.show()
            #
            # import matplotlib.pyplot as plt
            # for k,planet_partial_template_func in enumerate(planet_model_func_table[0][1::]):
            #     plt.plot(wvs,planet_model_func_table[0][0](wvs*(1-(-1*38)/c_kms))+3*LPFvsHPF(planet_partial_template_func(wvs*(1-(-1*38)/c_kms)),cutoff)[1],label="{0}".format(k))
            # plt.legend()
            # plt.show()


            ##############################
            ## Reference star*transmission spectrum
            ##############################
            with open(fileinfos_refstars_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=';')
                refstarsinfo_list_table = list(csv_reader)
                refstarsinfo_colnames = refstarsinfo_list_table[0]
                refstarsinfo_list_data = refstarsinfo_list_table[1::]
            refstarsinfo_filename_id = refstarsinfo_colnames.index("filename")
            refstarsinfo_filelist = [os.path.basename(item[refstarsinfo_filename_id]) for item in refstarsinfo_list_data]
            type_id = refstarsinfo_colnames.index("type")
            Jmag_id = refstarsinfo_colnames.index("Jmag")
            Hmag_id = refstarsinfo_colnames.index("Hmag")
            Kmag_id = refstarsinfo_colnames.index("Kmag")
            rv_simbad_id = refstarsinfo_colnames.index("RV Simbad")
            starname_id = refstarsinfo_colnames.index("star name")

            # phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
            # with pyfits.open(phoenix_wv_filename) as hdulist:
            #     phoenix_wvs = hdulist[0].data/1.e4
            # crop_phoenix = np.where((phoenix_wvs>wvs[0]-(wvs[-1]-wvs[0])/2)*(phoenix_wvs<wvs[-1]+(wvs[-1]-wvs[0])/2))
            # phoenix_wvs = phoenix_wvs[crop_phoenix]

            transmission_table = []
            transmission4planet_list = []
            HR8799pho_spec_func_list = []

            for Rid,R in enumerate(R_list):
                # refstar_name_filter = "HIP_1123"
                # refstar_name_filter = "HD_210501"
                if float(date) > 200000 and star_name == "HR_8799":
                    refstar_name_filter = "HR_8799"
                else:
                    refstar_name_filter = "*"
                transmission_filelist = []
                transmission_filelist.extend(glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"s*"+IFSfilter+"_"+scale+"_psfs_repaired_spec_v2_transmission_v3.fits")))
                transmission_filelist.extend(glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"ao_off_s*"+IFSfilter+"_"+scale+"_spec_v2_transmission_v3.fits")))
                transmission_filelist.sort()
                transmission_list = []
                for transmission_filename in transmission_filelist:
                    print(transmission_filename)
                    with pyfits.open(transmission_filename) as hdulist:
                        transmission_wvs = hdulist[0].data[0,:]
                        transmission_spec = hdulist[0].data[1,:]
                        if np.size(np.where(np.isnan(transmission_spec))[0]) >= 0.1*np.size(transmission_spec):
                            continue
                        transmission_list.append(transmission_spec/np.nanmean(transmission_spec))
                # mean_transmission = np.nanmean(np.array(transmission_list),axis=0)
                mean_transmission = combine_spectra(transmission_list)
                ## better tranmission combination
                # ????
                # exit()

                if 0:
                    mean_wherenans = np.where(np.isnan(mean_transmission))
                    for mytransmission in transmission_list:
                        wherenans = np.where(np.isnan(mytransmission))
                        print(wherenans)
                        mytransmission[wherenans] = mean_transmission[wherenans]

                        mytransmission[mean_wherenans] = 0

                    # for mytransmission in transmission_list:
                    #     wherenans = np.where(np.isnan(mytransmission))
                    #     print(2, wherenans)
                    #     exit()
                    covar_trans = np.cov(np.array(transmission_list))
                    numbasis = 3
                    evals, evecs = la.eigh(covar_trans, eigvals=(len(transmission_list)-numbasis, len(transmission_list)-1))

                    evals = np.copy(evals[::-1])
                    evecs = np.copy(evecs[:,::-1], order='F') #fortran order to improve memory caching in matrix multiplication

                    kl_basis = np.dot(np.array(transmission_list).T, evecs)
                    kl_basis = kl_basis * (1. / np.sqrt(evals * (np.size(transmission_list[0]) - 1)))[None, :]  #multiply a value for each row
                    kl_basis = kl_basis.T

                    print(kl_basis.shape)
                    # exit()
                    for n in range(numbasis):
                        kl_basis[n,mean_wherenans[0]] = np.nan

                    import  matplotlib.pyplot as plt
                    plt.plot(mean_transmission,label="mean")
                    for n in range(numbasis):
                        plt.plot(kl_basis[n,:],label="{0}".format(n))
                    plt.legend()
                    plt.show()

                mean_transmission_func1 = interp1d(wvs,mean_transmission,bounds_error=False,fill_value=np.nan)
                imgs_hdrbadpix[:,:,np.where(np.isnan(mean_transmission))[0]] = np.nan


                # refstar_name_filter = "*"
                # transmission_filelist = []
                # # transmission_filelist.extend(glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"s*"+IFSfilter+"_"+scale+"_psfs_repaired_spec_v2_cutoff20_transmission.fits")))
                # transmission_filelist.extend(glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"ao_off_s*"+IFSfilter+"_"+scale+"_spec_v2_cutoff20_transmission.fits")))
                # transmission_filelist.sort()
                # print(transmission_filelist)
                # transmission_list = []
                # for transmission_filename in transmission_filelist[0:2]:
                #     with pyfits.open(transmission_filename) as hdulist:
                #         transmission_wvs = hdulist[0].data[0,:]
                #         transmission_spec = hdulist[0].data[1,:]
                #         transmission_list.append(transmission_spec/np.nanmean(transmission_spec))
                # mean_transmission_func2 = interp1d(wvs,np.nanmean(np.array(transmission_list),axis=0),bounds_error=False,fill_value=np.nan)

                if 0 or plot_transmissions:
                    import matplotlib.pyplot as plt
                    print(R)
                    print(transmission_list)
                    # plt.figure(1)
                    for transid,trans in enumerate(transmission_list):
                        plt.plot(wvs,trans,label="{0}".format(transid))
                    plt.legend()
                    plt.show()

                # transmission_func_list = [mean_transmission_func1]
                # transmission_func_list = [mean_transmission_func1,mean_transmission_func2]
                transmission_func_list = [interp1d(wvs,mytrans,bounds_error=False,fill_value=np.nan) for mytrans in transmission_list]
                transmission4planet_list.append(mean_transmission_func1)
                transmission_table.append(transmission_func_list)

                ##############################
                ## host star phoenix model
                ##############################

                # phoenix_host_filename=phoenix_model_host_filename.replace(".fits","_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))
                # if use_R_calib:
                #     with pyfits.open(phoenix_model_host_filename) as hdulist:
                #         phoenix_HR8799 = hdulist[0].data[crop_phoenix]
                #     where_IFSfilter = np.where((phoenix_wvs>wvs[0])*(phoenix_wvs<wvs[-1]))
                #     phoenix_HR8799 = phoenix_HR8799/np.mean(phoenix_HR8799[where_IFSfilter])
                #     HR8799pho_spec_func = interp1d(phoenix_wvs,phoenix_HR8799,bounds_error=False,fill_value=np.nan)
                # else:
                #     if len(glob.glob(phoenix_host_filename)) == 0:
                #         with pyfits.open(phoenix_model_host_filename) as hdulist:
                #             phoenix_HR8799 = hdulist[0].data[crop_phoenix]
                #         print("convolving: "+phoenix_model_host_filename)
                #         phoenix_HR8799_conv = convolve_spectrum(phoenix_wvs,phoenix_HR8799,R,specpool)
                #
                #         with open(phoenix_host_filename, 'w+') as csvfile:
                #             csvwriter = csv.writer(csvfile, delimiter=' ')
                #             csvwriter.writerows([["wvs","spectrum"]])
                #             csvwriter.writerows([[a,b] for a,b in zip(phoenix_wvs,phoenix_HR8799_conv)])
                #
                #     with open(phoenix_host_filename, 'r') as csvfile:
                #         csv_reader = csv.reader(csvfile, delimiter=' ')
                #         list_starspec = list(csv_reader)
                #         HR8799pho_spec_str_arr = np.array(list_starspec, dtype=np.str)
                #         col_names = HR8799pho_spec_str_arr[0]
                #         HR8799pho_spec = HR8799pho_spec_str_arr[1::,1].astype(np.float)
                #         HR8799pho_spec_wvs = HR8799pho_spec_str_arr[1::,0].astype(np.float)
                if 1:
                    phoenix_wv_filename = os.path.join(phoenix_db_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011_R{0}.fits".format(R0))
                    with pyfits.open(phoenix_wv_filename) as hdulist:
                        HR8799pho_spec_wvs = hdulist[0].data

                    with pyfits.open(phoenix_model_host_filename) as hdulist:
                        HR8799pho_spec = hdulist[0].data
                    # phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
                    # with pyfits.open(phoenix_wv_filename) as hdulist:
                    #     phoenix_wvs = hdulist[0].data/1.e4
                    # crop_phoenix = np.where((phoenix_wvs>wvs[0]-(wvs[-1]-wvs[0])/2)*(phoenix_wvs<wvs[-1]+(wvs[-1]-wvs[0])/2))
                    # phoenix_wvs = phoenix_wvs[crop_phoenix]
                    #
                    # phoenix_host_filename=phoenix_model_host_filename.replace(".fits","_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))
                    # if len(glob.glob(phoenix_host_filename)) == 0:
                    #     with pyfits.open(phoenix_model_host_filename) as hdulist:
                    #         phoenix_HR8799 = hdulist[0].data[crop_phoenix]
                    #     print("convolving: "+phoenix_model_host_filename)
                    #     phoenix_HR8799_conv = convolve_spectrum(phoenix_wvs,phoenix_HR8799,R,specpool)
                    #
                    #     with open(phoenix_host_filename, 'w+') as csvfile:
                    #         csvwriter = csv.writer(csvfile, delimiter=' ')
                    #         csvwriter.writerows([["wvs","spectrum"]])
                    #         csvwriter.writerows([[a,b] for a,b in zip(phoenix_wvs,phoenix_HR8799_conv)])
                    #
                    # with open(phoenix_host_filename, 'r') as csvfile:
                    #     csv_reader = csv.reader(csvfile, delimiter=' ')
                    #     list_starspec = list(csv_reader)
                    #     HR8799pho_spec_str_arr = np.array(list_starspec, dtype=np.str)
                    #     col_names = HR8799pho_spec_str_arr[0]
                    #     HR8799pho_spec = HR8799pho_spec_str_arr[1::,1].astype(np.float)
                    #     HR8799pho_spec_wvs = HR8799pho_spec_str_arr[1::,0].astype(np.float)


                    HR8799pho_spec = HR8799pho_spec/np.mean(HR8799pho_spec)

                    HR8799pho_spec_func = interp1d(HR8799pho_spec_wvs,HR8799pho_spec,bounds_error=False,fill_value=np.nan)
                    # import matplotlib.pyplot as plt
                    # plt.plot(HR8799pho_spec_wvs,HR8799pho_spec,label="ori")
                    wvs4broadening = np.arange(HR8799pho_spec_wvs[0],HR8799pho_spec_wvs[-1],1e-4)
                    # plt.plot(wvs4broadening,HR8799pho_spec_func(wvs4broadening),label="sampling")
                    broadened_HR8799pho_spec = pyasl.rotBroad(wvs4broadening, HR8799pho_spec_func(wvs4broadening), host_limbdark, host_vsini)
                    # plt.plot(wvs4broadening,broadened_HR8799pho_spec,label="broad")
                    # plt.legend()
                    # plt.show()

                    HR8799pho_spec_func = interp1d(wvs4broadening/(1-(host_rv+host_bary_rv)/c_kms),broadened_HR8799pho_spec,bounds_error=False,fill_value=np.nan)

                    # # #remove
                    # import matplotlib.pyplot as plt
                    # plt.plot(wvs,HR8799pho_spec_func(wvs)/np.mean(HR8799pho_spec_func(wvs)),color="red")
                    # a = np.nansum(imgs,axis=(0,1))
                    # plt.plot(wvs,a/np.nanmean(a),color="blue")
                    #
                    # plt.plot(wvs,(a/HR8799pho_spec_func(wvs))/np.nanmean(a/HR8799pho_spec_func(wvs)),color="blue")
                    # plt.show()

                HR8799pho_spec_func_list.append(HR8799pho_spec_func)

            ##############################
            ## calibrate flux
            ##############################
            #... extract fluxes from the spectra
            refstar_name_filter = "*"
            psfs_rep4flux_filelist = glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"s*"+IFSfilter+"_"+scale+"_psfs_repaired_v2.fits"))
            psfs_rep4flux_filelist.sort()
            hr8799_flux_list = []
            for psfs_rep4flux_filename in psfs_rep4flux_filelist:
                for refstar_fileid,refstarsinfo_file in enumerate(refstarsinfo_filelist):
                    if os.path.basename(refstarsinfo_file).replace(".fits","") in psfs_rep4flux_filename:
                        fileitem = refstarsinfo_list_data[refstar_fileid]
                        break
                refstar_RV = float(fileitem[rv_simbad_id])
                ref_star_type = fileitem[type_id]
                if IFSfilter == "Jbb":
                    refstar_mag = float(fileitem[Jmag_id])
                elif IFSfilter == "Hbb":
                    refstar_mag = float(fileitem[Hmag_id])
                elif IFSfilter == "Kbb":
                    refstar_mag = float(fileitem[Kmag_id])

                with pyfits.open(psfs_rep4flux_filename) as hdulist:
                    psfs_repaired = hdulist[0].data
                    bbflux = np.nansum(psfs_repaired)
                hr8799_flux_list.append(bbflux* 10**(-1./2.5*(host_mag-refstar_mag)))

            hr8799_flux = np.mean(hr8799_flux_list)

            ##############################
            ## Sky transmission spectrum model
            ##############################
            if model_based_sky_trans:
                transmission_table = []
                filelist_skytrans = glob.glob(os.path.join(sky_transmission_folder,"mktrans_zm_*_*.dat"))
                if debug_paras:
                    filelist_skytrans = filelist_skytrans[3:5]

                for Rid,R in enumerate(R_list):
                    for filename_skytrans in filelist_skytrans:
                        skybg_arr=np.loadtxt(filename_skytrans)
                        skytrans_wvs = skybg_arr[:,0]
                        skytrans_spec = skybg_arr[:,1]
                        selec_skytrans = np.where((skytrans_wvs>wvs[0]-(wvs[-1]-wvs[0])/2)*(skytrans_wvs<wvs[-1]+(wvs[-1]-wvs[0])/2))
                        skytrans_wvs = skytrans_wvs[selec_skytrans]
                        skytrans_spec = skytrans_spec[selec_skytrans]

                        if not use_R_calib:
                            print("convolving: "+filename_skytrans+" with R={0}".format(R))
                            skytrans_spec = convolve_spectrum(skytrans_wvs,skytrans_spec,R,specpool)

                        transmission_table.append([interp1d(skytrans_wvs,skytrans_spec,bounds_error=False,fill_value=np.nan)])

            # #remove
            # for transid,trans in enumerate(transmission_list):
            #     plt.plot(trans,label="{0}".format(transid))
            # plt.legend()
            # plt.show()

            ##############################
            ## Create PSF model
            ##############################
            with pyfits.open(glob.glob(os.path.join(ref_star_folder,"*"+IFSfilter+"_hdpsfs_v2.fits"))[0]) as hdulist:
                psfs_refstar_arr = hdulist[0].data[None,:,:,:]
            with pyfits.open(glob.glob(os.path.join(ref_star_folder,"*"+IFSfilter+"_hdpsfs_xy_v2.fits"))[0]) as hdulist:
                hdpsfs_xy = hdulist[0].data
                hdpsfs_x,hdpsfs_y = hdpsfs_xy

            nx_psf,ny_psf = 15,15
            nz_psf = psfs_refstar_arr.shape[1]
            x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
            x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)

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
            specpool.close()

            if plt_psfs:
                w=2
                pl_x_vec = np.linspace(-w,w+1,100)
                pl_y_vec = np.linspace(-w,w+1,100)
                nospec_planet_model = np.zeros((100,100,nz))
                # nospec_planet_model = np.zeros((2*w+1,2*w+1,nz))
                for z in range(nz):
                    nospec_planet_model[:,:,z] = normalized_psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()

                import matplotlib.pyplot as plt
                plt.figure(1)
                for k in range(25):
                    plt.subplot(5,5,k+1)
                    plt.imshow(nospec_planet_model[:,:,int(1600//25)*k],interpolation="nearest")
                # for bkg_k in range(2*w+1):
                #     for bkg_l in range(2*w+1):
                #         print(bkg_k*w+bkg_l+1)
                #         plt.subplot(2*w+1,2*w+1,bkg_k*(2*w+1)+bkg_l+1)
                #         plt.plot(nospec_planet_model[bkg_k,bkg_l,:])
                plt.show()



            padimgs = np.pad(imgs,((padding,padding),(padding,padding),(0,0)),mode="constant",constant_values=0)
            padimgs_hdrbadpix = np.pad(imgs_hdrbadpix,((padding,padding),(padding,padding),(0,0)),mode="constant",constant_values=0)
            padny,padnx,padnz = padimgs.shape

            persistence_imgs = mp.Array(dtype, np.size(padimgs))
            persistence_imgs_shape = padimgs.shape
            persistence_imgs_np = _arraytonumpy(persistence_imgs, persistence_imgs_shape,dtype=dtype)
            persistence_imgs_np[:] = np.nan
            sigmas_imgs = mp.Array(dtype, np.size(padimgs))
            sigmas_imgs_shape = padimgs.shape
            sigmas_imgs_np = _arraytonumpy(sigmas_imgs, sigmas_imgs_shape,dtype=dtype)
            sigmas_imgs_np[:] = np.nan
            original_imgs = mp.Array(dtype, np.size(padimgs))
            original_imgs_shape = padimgs.shape
            original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
            original_imgs_np[:] = padimgs
            badpix_imgs = mp.Array(dtype, np.size(padimgs))
            badpix_imgs_shape = padimgs.shape
            badpix_imgs_np = _arraytonumpy(badpix_imgs, badpix_imgs_shape,dtype=dtype)
            badpix_imgs_np[:] = padimgs_hdrbadpix
            # badpix_imgs_np[np.where(original_imgs_np==0)] = np.nan
            originalHPF_imgs = mp.Array(dtype, np.size(padimgs))
            originalHPF_imgs_shape = padimgs.shape
            originalHPF_imgs_np = _arraytonumpy(originalHPF_imgs, originalHPF_imgs_shape,dtype=dtype)
            originalHPF_imgs_np[:] = np.nan
            originalLPF_imgs = mp.Array(dtype, np.size(padimgs))
            originalLPF_imgs_shape = padimgs.shape
            originalLPF_imgs_np = _arraytonumpy(originalLPF_imgs, originalLPF_imgs_shape,dtype=dtype)
            originalLPF_imgs_np[:] = np.nan
            nout = 10+3*(len(planet_model_func_table[0]))
            # print(nout)
            nshifts = np.size(planetRV_array)
            if planet_search:
                output_maps = mp.Array(dtype, len(numbasis_list)*len(transmission_table)*nout*padny*padnx*nshifts)
                output_maps_shape = (len(numbasis_list),len(transmission_table),nout,padny,padnx,nshifts)
            else:
                output_maps = mp.Array(dtype, len(numbasis_list)*len(transmission_table)*nout*dl_grid.shape[0]*dl_grid.shape[1]*nshifts)
                output_maps_shape = (len(numbasis_list),len(transmission_table),nout,dl_grid.shape[0],dl_grid.shape[1],nshifts)
            output_maps_np = _arraytonumpy(output_maps,output_maps_shape,dtype=dtype)
            output_maps_np[:] = np.nan
            if planet_search:
                out1dfit = mp.Array(dtype, 4*padny*padnx*padimgs.shape[-1])
                out1dfit_shape = (4,padny,padnx,padimgs.shape[-1])
            else:
                out1dfit = mp.Array(dtype, 4*dl_grid.shape[0]*dl_grid.shape[1]*padimgs.shape[-1])
                out1dfit_shape = (4,dl_grid.shape[0],dl_grid.shape[1],padimgs.shape[-1])
            out1dfit_np = _arraytonumpy(out1dfit,out1dfit_shape,dtype=dtype)
            out1dfit_np[:] = np.nan
            if planet_search:
                estispec = mp.Array(dtype, 5*padny*padnx*padimgs.shape[-1])
                estispec_shape = (5,padny,padnx,padimgs.shape[-1])
            else:
                estispec = mp.Array(dtype, 5*dl_grid.shape[0]*dl_grid.shape[1]*padimgs.shape[-1])
                estispec_shape = (5,dl_grid.shape[0],dl_grid.shape[1],padimgs.shape[-1])
            estispec_np = _arraytonumpy(estispec,estispec_shape,dtype=dtype)
            estispec_np[:] = np.nan
            if planet_search:
                outres = mp.Array(dtype, len(numbasis_list)*len(transmission_table)*7*padny*padnx*padimgs.shape[-1])
                outres_shape = (len(numbasis_list),len(transmission_table),7,padimgs.shape[-1],padny,padnx)
            else:
                outres = mp.Array(dtype, len(numbasis_list)*len(transmission_table)*7*dl_grid.shape[0]*dl_grid.shape[1]*padimgs.shape[-1])
                outres_shape = (len(numbasis_list),len(transmission_table),7,padimgs.shape[-1],dl_grid.shape[0],dl_grid.shape[1])
            outres_np = _arraytonumpy(outres,outres_shape,dtype=dtype)
            outres_np[:] = np.nan
            if planet_search:
                outautocorrres = mp.Array(dtype, padny*padnx*1000)
                outautocorrres_shape = (1000,padny,padnx)
            else:
                outautocorrres = mp.Array(dtype, dl_grid.shape[0]*dl_grid.shape[1]*1000)
                outautocorrres_shape = (1000,dl_grid.shape[0],dl_grid.shape[1])
            outautocorrres_np = _arraytonumpy(outautocorrres,outautocorrres_shape,dtype=dtype)
            outautocorrres_np[:] = np.nan
            wvs_imgs = wvs
            psfs_stamps = mp.Array(dtype, np.size(psfs_refstar_arr))
            psfs_stamps_shape = psfs_refstar_arr.shape
            psfs_stamps_np = _arraytonumpy(psfs_stamps, psfs_stamps_shape,dtype=dtype)
            psfs_stamps_np[:] = psfs_refstar_arr

            ##############################
            ## INIT threads and shared memory
            ##############################
            tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                            initargs=(original_imgs,sigmas_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps,
                                      output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence_imgs,out1dfit,out1dfit_shape,estispec,estispec_shape),
                            maxtasksperchild=50)



            ##############################
            ## CLEAN IMAGE
            ##############################
            chunk_size = padnz//(3*numthreads)
            N_chunks = padnz//chunk_size
            wvs_indices_list = []
            for k in range(N_chunks-1):
                wvs_indices_list.append(np.arange((k*chunk_size),((k+1)*chunk_size)))
            wvs_indices_list.append(np.arange(((N_chunks-1)*chunk_size),padnz))

            # import matplotlib.pyplot as plt
            # tmpk, tmpl = 44,8
            # for k in range(5):
            #     for l in range(5):
            #         plt.subplot(5,5,5*k+l+1)
            #         plt.plot(original_imgs_np[tmpk+5+k,tmpl+5+l,:],label="before")
            #         wherenans = np.where(np.isfinite(badpix_imgs_np[tmpk+5+k,tmpl+5+l,:]))
            #         plt.plot(wherenans[0],original_imgs_np[tmpk+5+k,tmpl+5+l,:][wherenans],label="BP before")

            tasks = [tpool.apply_async(_remove_bad_pixels_z, args=(col_index,nan_mask_boxsize, dtype,100,7))
                     for col_index in range(padnx)]
            #save it to shared memory
            for col_index, bad_pix_task in enumerate(tasks):
                print("Finished rm bad pixel z col {0}".format(col_index))
                bad_pix_task.wait()


            tasks = [tpool.apply_async(_remove_edges, args=(wvs_indices,nan_mask_boxsize,dtype))
                     for wvs_indices in wvs_indices_list]
            #save it to shared memory
            for chunk_index, rmedge_task in enumerate(tasks):
                print("Finished rm edge chunk {0}".format(chunk_index))
                rmedge_task.wait()


            # tasks = [tpool.apply_async(_remove_bad_pixels_xy, args=(wvs_indices,dtype))
            #          for wvs_indices in wvs_indices_list]
            # #save it to shared memory
            # for chunk_index, rmedge_task in enumerate(tasks):
            #     print("Finished rm bad pixel xy chunk {0}".format(chunk_index))
            #     rmedge_task.wait()

            ##############################
            ## derive sigma for pairsub
            ##############################
            if pairsub:
                save_original_imgs_np = copy(original_imgs_np[:])

                for line_num,line in enumerate(prihdr["COMMENT"]):
                    if "Subtract Frame" in line:
                        followingline = 0
                        while ".fits" not in prihdr["COMMENT"][line_num+followingline]:
                            followingline += 1
                        subfilename = "s"+prihdr["COMMENT"][line_num+followingline].split(".fits")[0].split("s")[-1]+".fits"
                        break

                filename1_skysub = filename.replace("_pairsub","")
                filename2_skysub = os.path.join(os.path.dirname(filename1_skysub),
                                                subfilename.replace(".fits","_"+IFSfilter+"_"+scale+".fits"))
                with pyfits.open(filename1_skysub) as hdulist:
                    imgs1_skysub = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                    imgs1_skysub = np.moveaxis(imgs1_skysub,0,2)
                with pyfits.open(filename2_skysub) as hdulist:
                    imgs2_skysub = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                    imgs2_skysub = np.moveaxis(imgs2_skysub,0,2)

                padimgs1_skysub = np.pad(imgs1_skysub,((padding,padding),(padding,padding),(0,0)),mode="constant",constant_values=0)
                original_imgs_np[:] = copy(padimgs1_skysub)
                tasks = [tpool.apply_async(_HPF_z, args=(col_index,cutoff, dtype))
                         for col_index in range(padnx)]
                #save it to shared memory
                for col_index, task in enumerate(tasks):
                    print("Finished col {0}".format(col_index))
                    task.wait()
                imgs1_skysub_LPF = copy(originalLPF_imgs_np)

                padimgs2_skysub = np.pad(imgs2_skysub,((padding,padding),(padding,padding),(0,0)),mode="constant",constant_values=0)
                original_imgs_np[:] = copy(padimgs2_skysub)
                tasks = [tpool.apply_async(_HPF_z, args=(col_index,cutoff, dtype))
                         for col_index in range(padnx)]
                #save it to shared memory
                for col_index, task in enumerate(tasks):
                    print("Finished col {0}".format(col_index))
                    task.wait()
                imgs2_skysub_LPF = copy(originalLPF_imgs_np)

                # sigmas_imgs = np.clip(np.sqrt(np.abs(imgs1_skysub)+np.abs(imgs2_skysub)),
                #                       np.sqrt(np.abs(np.nanmedian(imgs1_skysub))+np.abs(np.nanmedian(imgs2_skysub))),np.inf)
                padsigmas_imgs = np.sqrt(np.abs(imgs1_skysub_LPF)+np.abs(imgs2_skysub_LPF))

                original_imgs_np[:] = save_original_imgs_np

            ##############################
            if model_persistence:
                save_original_imgs_np = copy(original_imgs_np[:])

                pad_persistence_arr = np.pad(persistence_arr,((padding,padding),(padding,padding),(0,0)),mode="constant",constant_values=0)
                original_imgs_np[:] = copy(pad_persistence_arr)
                tasks = [tpool.apply_async(_HPF_z, args=(col_index,cutoff, dtype))
                         for col_index in range(padnx)]
                #save it to shared memory
                for col_index, task in enumerate(tasks):
                    print("Finished col {0}".format(col_index))
                    task.wait()
                persistence_imgs_np[:] = copy(originalHPF_imgs_np)


                original_imgs_np[:] = save_original_imgs_np


            tasks = [tpool.apply_async(_HPF_z, args=(col_index,cutoff, dtype))
                     for col_index in range(padnx)]
            #save it to shared memory
            for col_index, task in enumerate(tasks):
                print("Finished col {0}".format(col_index))
                task.wait()

            badpix_imgs_np[np.where(padimgs_hdrbadpix==0)] = np.nan
            badpix_imgs_np[:,:,(padnz-20)::]=np.nan
            badpix_imgs_np[:,:,0:20]=np.nan
            # badpix_imgs_np[:,:,1621]=np.nan
            if mask_starline:
                badpix_imgs_np[:,:,795:810]=np.nan
            if mask_20101104_artifact:
                badpix_imgs_np[0:5+padding,:,:]=np.nan
                badpix_imgs_np[(padny-5-padding):padny,:,:]=np.nan
                badpix_imgs_np[43:50,:,0:400]=np.nan
                badpix_imgs_np[43:50,:,0:400]=np.nan
            if IFSfilter == "Jbb":
                badpix_imgs_np[:,:,1000::]=np.nan
                badpix_imgs_np[:,:,0:100]=np.nan
            if IFSfilter == "Hbb" and date == "101104":
                badpix_imgs_np[0:10,:,:]=np.nan
            if IFSfilter == "Hbb" and date == "171103":
                badpix_imgs_np[:,:,(padnz-50)::]=np.nan
                badpix_imgs_np[:,:,0:50]=np.nan


            # import matplotlib.pyplot as plt
            # tmpk, tmpl = 49,12
            # for k in range(5):
            #     for l in range(5):
            #         plt.subplot(5,5,5*k+l+1)
            #         plt.plot(original_imgs_np[tmpk+5+k,tmpl+5+l,:],label="original_imgs_np")
            #         plt.plot(originalHPF_imgs_np[tmpk+5+k,tmpl+5+l,:],label="originalHPF_imgs_np")
            #         plt.plot(originalLPF_imgs_np[tmpk+5+k,tmpl+5+l,:],label="originalLPF_imgs_np")
            #         # plt.plot(imgs1_skysub_LPF[tmpk+5+k,tmpl+5+l,:],label="imgs1_skysub_LPF")
            #         # plt.plot(imgs2_skysub_LPF[tmpk+5+k,tmpl+5+l,:],label="imgs2_skysub_LPF")
            #         plt.ylim([-3,3])
            # plt.legend()
            # tpool.close()
            # plt.show()

            # import matplotlib.pyplot as plt
            # tmpk, tmpl = 14,12
            # for k in range(5):
            #     for l in range(5):
            #         plt.subplot(5,5,5*k+l+1)
            #         plt.plot(original_imgs_np[tmpk+5+k,tmpl+5+l,:],label="after")
            #         plt.plot(originalHPF_imgs_np[tmpk+5+k,tmpl+5+l,:],label="HPF after")
            #         wherenans = np.where(np.isfinite(badpix_imgs_np[tmpk+5+k,tmpl+5+l,:]))
            #         plt.plot(wherenans[0],originalHPF_imgs_np[tmpk+5+k,tmpl+5+l,:][wherenans],label="BP after")
            # plt.legend()
            # tpool.close()
            # plt.show()

            if pairsub:
                sigmas_imgs_np[:] = padsigmas_imgs
                for k in range(padny):
                    for l in range(padnx):
                        sigmas_imgs_np[k,l,:] = np.clip(sigmas_imgs_np[k,l,:],np.nanmedian(sigmas_imgs_np[k,l,:])/2,np.inf)
            else:
                sigmas_imgs_np[:] = np.sqrt(np.abs(originalLPF_imgs_np))
                for k in range(padny):
                    for l in range(padnx):
                        sigmas_imgs_np[k,l,:] = np.clip(sigmas_imgs_np[k,l,:],np.nanmedian(sigmas_imgs_np[k,l,:])/2,np.inf)
            # import matplotlib.pyplot as plt
            # tmpk, tmpl = 44,8
            # for k in range(5):
            #     for l in range(5):
            #         plt.subplot(5,5,5*k+l+1)
            #         plt.plot(originalHPF_imgs_np[tmpk+5+k,tmpl+5+l,:],label="originalLPF_imgs_np")
            #         plt.plot(sigmas_imgs_np[tmpk+5+k,tmpl+5+l,:],label="sigmas_imgs_np")
            #         plt.ylim([-3,3])
            # plt.legend()
            # tpool.close()
            # plt.show()

            if 0: #"JBhere"
                out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
                import matplotlib.pyplot as plt
                print(original_imgs_np.shape)
                fontsize = 12
                plt.figure(1,figsize=(12,3))
                ax1 = plt.gca()
                plt.figure(2,figsize=(12,3))
                ax2 = plt.gca()
                for myk in np.arange(25,35):
                    for myl in np.arange(7,9):
                        myvec = copy(original_imgs_np[myk,myl,:])
                        myvec_cp = copy(myvec)#/transmission4planet_list[0](wvs)
                        print(np.nanmax(myvec_cp))
                        plt.sca(ax1)
                        plt.plot(wvs,myvec_cp,color="grey",alpha = 0.2)
                        wherenans = np.where(np.isnan(myvec_cp))
                        for k in wherenans[0]:
                            myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-10]):np.min([np.size(myvec_cp),k+10])])

                        fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
                        LPF_fftmyvec = copy(fftmyvec)
                        LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
                        LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
                        HPF_myvec = myvec_cp - LPF_myvec
                        HPF_myvec[wherenans] = np.nan
                        plt.sca(ax2)
                        plt.plot(wvs,HPF_myvec,color="grey",alpha = 0.2)
                plt.sca(ax1)
                plt.plot(wvs,myvec,color="grey",alpha = 0.2,label="Noise sample")
                plt.sca(ax2)
                plt.plot(wvs,HPF_myvec,color="grey",alpha = 0.2,label="Noise sample")

                myvec_cp = planet_partial_template_func_list[0](wvs)*transmission4planet_list[0](wvs)
                # plt.figure(3)
                # plt.plot(planet_partial_template_func_list[0](wvs))
                # plt.figure(4)
                # plt.plot(transmission4planet_list[0](wvs))
                # plt.show()
                plt.sca(ax1)
                plt.plot(wvs,myvec_cp/np.nanmax(myvec_cp)*0.4,color="#ff9900",alpha=0.75,label="HR 8799 c model")
                wherenans = np.where(np.isnan(myvec_cp))
                for k in wherenans[0]:
                    myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-10]):np.min([np.size(myvec_cp),k+10])])
                fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
                LPF_fftmyvec = copy(fftmyvec)
                LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
                LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
                HPF_myvec = myvec_cp - LPF_myvec
                plt.sca(ax2)
                plt.plot(wvs,HPF_myvec/np.max(myvec_cp)*0.4,color="#ff9900",alpha=0.75,label="HR 8799 c model")

                plt.sca(ax1)
                plt.ylim([0,0.5])
                plt.legend(loc="upper right",frameon=True,fontsize=fontsize,ncol=3)
                plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                # plt.tick_params(axis="y",which="both",labelleft=False,right=False,left=False)
                # plt.gca().spines["right"].set_visible(False)
                # plt.gca().spines["left"].set_visible(False)
                # plt.gca().spines["top"].set_visible(False)
                plt.tight_layout()
                print("Saving "+os.path.join(out_pngs,"speckles_uncorr.png"))
                plt.savefig(os.path.join(out_pngs,"speckles_uncorr.png"),bbox_inches='tight')
                plt.savefig(os.path.join(out_pngs,"speckles_uncorr.pdf"),bbox_inches='tight')

                plt.sca(ax2)
                plt.ylim([-0.2,0.2])
                plt.legend(loc="upper right",frameon=True,fontsize=fontsize,ncol=3)
                plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                plt.tight_layout()
                print("Saving "+os.path.join(out_pngs,"speckles_HPF_uncorr.png"))
                plt.savefig(os.path.join(out_pngs,"speckles_HPF_uncorr.png"),bbox_inches='tight')
                plt.savefig(os.path.join(out_pngs,"speckles_HPF_uncorr.pdf"),bbox_inches='tight')

                plt.figure(3,figsize=(12,3))
                fftx = 2*(wvs[-1]-wvs[0])/np.arange(padnz)
                plt.fill_betweenx([0,1e5],[100,100],[fftx[40],fftx[40]],color="#0099cc",alpha=0.2,label="Filtered out")
                for myk in np.arange(25,35):
                    for myl in np.arange(7,9):
                        myvec = original_imgs_np[myk,myl,:]
                        myvec_cp = copy(myvec)#/transmission4planet_list[0](wvs)
                        #handling nans:
                        wherenans = np.where(np.isnan(myvec_cp))
                        for k in wherenans[0]:
                            myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-10]):np.min([np.size(myvec_cp),k+10])])

                        fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
                        # LPF_fftmyvec = copy(fftmyvec)
                        # LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
                        # LPF_fftmyvec[:] = 0
                        # LPF_fftmyvec[cutoff] = 1
                        # LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
                        # HPF_myvec = myvec_cp - LPF_myvec

                        plt.plot(fftx,np.abs(fftmyvec)[0:np.size(myvec_cp)],color="grey",alpha = 0.2)
                plt.plot(fftx,np.abs(fftmyvec)[0:np.size(myvec_cp)],color="grey",alpha = 0.2,label="Noise sample")

                myplvec = planet_partial_template_func_list[0](wvs)*transmission4planet_list[0](wvs)
                wherenans = np.where(np.isnan(myplvec))
                for k in wherenans[0]:
                    myplvec[k] = np.nanmedian(myplvec[np.max([0,k-10]):np.min([np.size(myplvec),k+10])])
                fftmyvec = np.fft.fft(np.concatenate([myplvec,myplvec[::-1]],axis=0))
                plt.plot(fftx,np.abs(fftmyvec)[0:np.size(myplvec)],color="#ff9900",alpha=0.75,label="HR 8799 c model")
                # plt.xlim([0.100,0.416])
                plt.legend(loc="upper right",frameon=True,fontsize=fontsize,ncol=3)
                plt.xlim([np.min(fftx),1])
                plt.ylim([1e-1,1e3])
                plt.xscale("log")
                plt.yscale("log")
                plt.xlabel(r"Period ($\mu$m)",fontsize=fontsize)
                plt.ylabel(r"Power spectrum",fontsize=fontsize)
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                plt.gca().invert_xaxis()
                plt.tight_layout()
                print("Saving "+os.path.join(out_pngs,"speckles_fft_uncorr.png"))
                plt.savefig(os.path.join(out_pngs,"speckles_fft_uncorr.png"),bbox_inches='tight')
                plt.savefig(os.path.join(out_pngs,"speckles_fft_uncorr.pdf"),bbox_inches='tight')
                plt.show()
                exit()

            if 0:
                if not os.path.exists(os.path.join(outputdir)):
                    os.makedirs(os.path.join(outputdir))
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=np.moveaxis(originalLPF_imgs_np,len(originalLPF_imgs_np.shape)-1,len(originalLPF_imgs_np.shape)-3)[:,padding:(padny-padding),padding:(padnx-padding)],header=prihdr))
                try:
                    hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_LPF.fits")), overwrite=True)
                except TypeError:
                    hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_LPF.fits")), clobber=True)
                hdulist.close()
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=np.moveaxis(badpix_imgs_np,len(badpix_imgs_np.shape)-1,len(badpix_imgs_np.shape)-3)[:,padding:(padny-padding),padding:(padnx-padding)],header=prihdr))
                try:
                    hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_LPF_badpix.fits")), overwrite=True)
                except TypeError:
                    hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_LPF_badpix.fits")), clobber=True)
                hdulist.close()
                continue

            ressuffix = suffix + "_rescalc"
            if res_it == 0:
                suffix = ressuffix
            else:
                suffix = suffix+"_resinmodel_kl{0}".format(res_numbasis)
            if res_numbasis <= -1:
                # s171103_a023002_Hbb_020_outputHPF_cutoff40_sherlock_v1_search_res
                # res_filename = os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+ressuffix+"_res.fits"))
                res_filelist = glob.glob(os.path.join(outputdir,os.path.basename(filename).split("_")[0]+"*"+os.path.basename(filename).split("_")[2]+"*_output"+ressuffix+"_res.fits"))
                print(os.path.join(outputdir,os.path.basename(filename).split("_")[0]+"*"+os.path.basename(filename).split("_")[2]+"*_output"+ressuffix+"_res.fits"))
                print(res_filelist)
                # exit()
                X_list = []
                for res_filename in res_filelist:
                    with pyfits.open(res_filename) as hdulist:
                        # hpfres = hdulist[0].data[0,0,2,:,:,:]
                        hpfres = hdulist[0].data[0,0,6,:,:,:]
                        lpfres = hdulist[0].data[0,0,5,:,:,:]
                        res4model = hpfres/lpfres
                        X = np.reshape(res4model,(res4model.shape[0],res4model.shape[1]*res4model.shape[2])).T
                        X_list.append(X)
                X = np.concatenate(X_list)

            if res_numbasis >= 1:
                res_filelist = glob.glob(os.path.join(outputdir,os.path.basename(filename).replace(".fits","")+"*_output"+ressuffix+"_res.fits"))
                # res_filelist = glob.glob(os.path.join(outputdir,os.path.basename(filename).replace(".fits","")+"*_output"+"*kl1*"+"_res.fits"))
                res_filename = res_filelist[0]
                with pyfits.open(res_filename) as hdulist:
                    # hpfres = hdulist[0].data[0,0,2,:,:,:]
                    hpf = hdulist[0].data[0,0,0,:,:,:]
                    lpf = hdulist[0].data[0,0,5,:,:,:]
                    hpfres = hdulist[0].data[0,0,6,:,:,:]

                res_filelist = glob.glob(os.path.join(outputdir,os.path.basename(filename).replace(".fits","")+"*_output"+ressuffix+"_estispec.fits"))
                # res_filelist = glob.glob(os.path.join(outputdir,os.path.basename(filename).replace(".fits","")+"*_output"+"*kl1*"+"_res.fits"))
                res_filename = res_filelist[0]
                with pyfits.open(res_filename) as hdulist:
                    hpfres = hdulist[0].data[1,:,:,:]
                    hpfres = np.pad(hpfres,((0,0),(padding,padding),(padding,padding)),mode="constant",constant_values=0)
                res4model = hpfres/lpf

                # import matplotlib.pyplot as plt
                # plt.subplot(1,2,1)
                # plt.imshow(np.nansum(original_imgs_np,axis=2))

                try:
                # if 1:
                    status_id = colnames.index("status")
                    if int(list_data[fileid][status_id]) == 1:
                        kcen_id = colnames.index("kcen")
                        lcen_id = colnames.index("lcen")
                        kplloc,lplloc = int(list_data[fileid][kcen_id])+padding,int(list_data[fileid][lcen_id])+padding
                        # print(kplloc,lplloc)
                        res4model[:,kplloc-5:kplloc+6,lplloc-5:lplloc+6] = np.nan
                except:
                    pass

                # plt.subplot(1,2,2)
                # plt.imshow(np.nansum(res4model,axis=0))
                # plt.show()

                X = np.reshape(res4model,(res4model.shape[0],res4model.shape[1]*res4model.shape[2])).T

                print(X.shape)
                # exit()


            if res_numbasis != 0:
                X = X[np.where(np.nansum(X,axis=1)!=0)[0],:]
                X = X/np.nanstd(X,axis=1)[:,None]
                X[np.where(np.isnan(X))] = 0

                # lpf = np.nanmean(lpf,axis=(1,2))
                # lpf_res,_ = LPFvsHPF(np.nanmean(X,axis=0),cutoff,nansmooth=30)
                # lpf_res_calib = 1+ lpf_res/lpf
                # # X = X-lpf_res[None,:]

                print(X.shape)
                C = np.cov(X)
                # print(C.shape)
                # exit()
                tot_basis = C.shape[0]
                tmp_res_numbasis = np.clip(np.abs(res_numbasis) - 1, 0, tot_basis-1)  # clip values, for output consistency we'll keep duplicates
                max_basis = np.max(tmp_res_numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate
                evals, evecs = la.eigh(C, eigvals=(tot_basis-max_basis, tot_basis-1))
                check_nans = np.any(evals <= 0) # alternatively, check_nans = evals[0] <= 0
                evals = np.copy(evals[::-1])
                evecs = np.copy(evecs[:,::-1], order='F') #fortran order to improve memory caching in matrix multiplication
                # calculate the KL basis vectors
                kl_basis = np.dot(X.T, evecs)
                # JB question: Why is there this [None, :]? (It adds an empty first dimension)
                res4model_kl = kl_basis * (1. / np.sqrt(evals * (res4model.shape[0] - 1)))[None, :]  #multiply a value for each row
                print(res4model_kl.shape)
                # exit()

                # res4model = np.nansum(res4model,axis=(1,2))
                # res4model = res4model/np.nanstd(res4model)

                # import matplotlib.pyplot as plt
                # # plt.plot(wvs,-np.nanmean(X,axis=0)/np.nanstd(np.nanmean(X,axis=0)),label="0")
                # plt.plot(wvs,res4model_kl[:,0]/np.nanstd(res4model_kl[:,0]),label="0")
                # # plt.plot(-res4model_kl[:,1]/np.nanstd(res4model_kl[:,1]),label="1")
                # # plt.plot(-res4model_kl[:,2]/np.nanstd(res4model_kl[:,2]),label="2")
                # # plt.plot(res4model/np.nanstd(res4model),label="ref")
                # plt.legend()
                # plt.show()
                # exit()
                lpf_res_calib = None
            else:
                res4model_kl = None
                lpf_res_calib = None
            # print(res4model)
            # exit()

            ##############################
            ## Define fakes
            ##############################
            if inject_fakes:
                try:
                # if 1:
                    contrast_id = colnames.index("contrast")
                    RVcen_id = colnames.index("RVcen")
                    contrast,RVcen = float(list_data[fileid][contrast_id]),float(list_data[fileid][RVcen_id])
                    # fake_paras = {"contrast":contrast,"RV":RVcen}RV4fakes
                    fake_paras = {"contrast":contrast,"RV":RV4fakes}
                    suffix = suffix+"_fakes"
                except:
                    print("Cannot inject fake planets")
                    exit()
            else:
                fake_paras = None


            ##############################
            ## Define tasks
            ##############################
            # 1: planet search, 0: astrometry
            if planet_search:
                wherenotnans = np.where(np.nansum(np.isfinite(badpix_imgs_np),axis=2)!=0)
                plcen_k_valid_pix = wherenotnans[0]
                plcen_l_valid_pix = wherenotnans[1]
                row_valid_pix = wherenotnans[0]
                col_valid_pix = wherenotnans[1]
            else:
                plcen_k_grid,plcen_l_grid = dk_grid+plcen_k, dl_grid+plcen_l
                plcen_k_valid_pix = plcen_k_grid.ravel()
                plcen_l_valid_pix = plcen_l_grid.ravel()
                l_grid,k_grid = np.meshgrid(np.arange(plcen_k_grid.shape[1]),np.arange(plcen_k_grid.shape[0]))
                row_valid_pix = k_grid.ravel()
                col_valid_pix = l_grid.ravel()

            where_not2close2edge = np.where((plcen_k_valid_pix>padding)*(plcen_l_valid_pix>padding)*
                                            (plcen_k_valid_pix<=padny-padding)*(plcen_l_valid_pix<=padnx-padding))
            plcen_k_valid_pix = plcen_k_valid_pix[where_not2close2edge]
            plcen_l_valid_pix = plcen_l_valid_pix[where_not2close2edge]
            row_valid_pix = row_valid_pix[where_not2close2edge]
            col_valid_pix = col_valid_pix[where_not2close2edge]

            N_valid_pix = np.size(row_valid_pix)
            Npixtot = N_valid_pix

            ##############################
            ## Process
            ##############################
            if 0 or debug:
                print("coucou1")
                plcen_k,plcen_l = 30+padding,10+padding
                plcen_k_valid_pix = [plcen_k]
                plcen_l_valid_pix = [plcen_l]
                row_valid_pix = [plcen_k]
                col_valid_pix = [plcen_l]

                # print(planetRV_array)
                # exit()
                tpool.close()
                _tpool_init(original_imgs,sigmas_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps,
                            output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape,outres,outres_shape,outautocorrres,
                            outautocorrres_shape,persistence_imgs,out1dfit,out1dfit_shape,estispec,estispec_shape)
                _process_pixels_onlyHPF(plcen_k_valid_pix[::-1],plcen_l_valid_pix[::-1],row_valid_pix[::-1],col_valid_pix[::-1],
                                        normalized_psfs_func_list,
                                        transmission_table,
                                        planet_model_func_table,
                                        HR8799pho_spec_func_list,
                                        transmission4planet_list,
                                        hr8799_flux,
                                        wvs_imgs,planetRV_array,
                                        dtype,cutoff,planet_search,(plcen_k,plcen_l),
                                        R_list,
                                        numbasis_list,wvsol_offsets,R_calib_arr,model_persistence=model_persistence,res4model_kl=res4model_kl,lpf_res_calib=lpf_res_calib,fake_paras=fake_paras)
                exit()
            else:
                chunk_size = 5#N_valid_pix//(3*numthreads)
                N_chunks = N_valid_pix//chunk_size
                row_indices_list = []
                col_indices_list = []
                plcen_k_indices_list = []
                plcen_l_indices_list = []
                for k in range(N_chunks-1):
                    row_indices_list.append(row_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
                    col_indices_list.append(col_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
                    plcen_k_indices_list.append(plcen_k_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
                    plcen_l_indices_list.append(plcen_l_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
                row_indices_list.append(row_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])
                col_indices_list.append(col_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])
                plcen_k_indices_list.append(plcen_k_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])
                plcen_l_indices_list.append(plcen_l_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])

                tasks = [tpool.apply_async(_process_pixels_onlyHPF, args=(plcen_k_indices,plcen_l_indices,row_indices,col_indices,
                                                                          normalized_psfs_func_list,
                                                                          transmission_table,
                                                                          planet_model_func_table,
                                                                          HR8799pho_spec_func_list,
                                                                          transmission4planet_list,
                                                                          hr8799_flux,
                                                                          wvs_imgs,planetRV_array,
                                                                          dtype,cutoff,planet_search,(plcen_k,plcen_l),
                                                                          R_list,
                                                                          numbasis_list,wvsol_offsets,R_calib_arr,
                                                                          model_persistence,res4model_kl,lpf_res_calib,fake_paras))
                         for plcen_k_indices, plcen_l_indices, row_indices, col_indices in zip(plcen_k_indices_list,plcen_l_indices_list, row_indices_list, col_indices_list)]
                #save it to shared memory
                for row_index, proc_pixs_task in enumerate(tasks):
                    print("Finished image chunk {0}/{1}".format(row_index,len(plcen_k_indices_list)))
                    proc_pixs_task.wait()

            ##############################
            ## Save data to disk
            ##############################
            if not os.path.exists(os.path.join(outputdir)):
                os.makedirs(os.path.join(outputdir))

            hdulist = pyfits.HDUList()
            if planet_search:
                hdulist.append(pyfits.PrimaryHDU(data=np.moveaxis(output_maps_np,len(output_maps_shape)-1,len(output_maps_shape)-3)[:,:,:,:,padding:(padny-padding),padding:(padnx-padding)]))
            else:
                hdulist.append(pyfits.PrimaryHDU(data=np.moveaxis(output_maps_np,len(output_maps_shape)-1,len(output_maps_shape)-3)))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits")), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits")), clobber=True)
            hdulist.close()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=planetRV_array))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_planetRV.fits")), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_planetRV.fits")), clobber=True)
            hdulist.close()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=outres_np))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_res.fits")), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_res.fits")), clobber=True)
            hdulist.close()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=outautocorrres_np))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_autocorrres.fits")), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_autocorrres.fits")), clobber=True)
            hdulist.close()

            hdulist = pyfits.HDUList()
            if planet_search:
                hdulist.append(pyfits.PrimaryHDU(data=np.moveaxis(out1dfit_np,len(out1dfit_shape)-1,len(out1dfit_shape)-3)[:,:,padding:(padny-padding),padding:(padnx-padding)]))
            else:
                hdulist.append(pyfits.PrimaryHDU(data=np.moveaxis(out1dfit_np,len(out1dfit_shape)-1,len(out1dfit_shape)-3)))
            # hdulist.append(pyfits.PrimaryHDU(data=out1dfit_np))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_out1dfit.fits")), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_out1dfit.fits")), clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            if planet_search:
                hdulist.append(pyfits.PrimaryHDU(data=np.moveaxis(estispec_np,len(estispec_shape)-1,len(estispec_shape)-3)[:,:,padding:(padny-padding),padding:(padnx-padding)]))
            else:
                hdulist.append(pyfits.PrimaryHDU(data=np.moveaxis(estispec_np,len(estispec_shape)-1,len(estispec_shape)-3)))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_estispec.fits")), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_estispec.fits")), clobber=True)
            hdulist.close()

            if not planet_search:
                hdulist = pyfits.HDUList()
                if planet_search:
                    hdulist.append(pyfits.PrimaryHDU(data=np.concatenate([-padding+plcen_k_grid[None,padding:(padny-padding),padding:(padnx-padding)],-padding+plcen_l_grid[None,padding:(padny-padding),padding:(padnx-padding)]],axis=0)))
                else:
                    hdulist.append(pyfits.PrimaryHDU(data=np.concatenate([-padding+plcen_k_grid[None,:,:],-padding+plcen_l_grid[None,:,:]],axis=0)))
                try:
                    hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_klgrids.fits")), overwrite=True)
                except TypeError:
                    hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_klgrids.fits")), clobber=True)
                hdulist.close()

            print("Closing threadpool")
            tpool.close()
            tpool.join()
            # plt.figure(2)
            # plt.subplot(1,2,1)
            # plt.imshow(output_maps_np[0,:,:])
            # plt.subplot(1,2,2)
            # plt.imshow(output_maps_np[1,:,:])
            # print("bonjour")
            # plt.show()


            # exit()
