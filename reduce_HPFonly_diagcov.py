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
                wvs_imgs,psfs_stamps,psfs_stamps_shape,_outres,_outres_shape,_outautocorrres,_outautocorrres_shape,persistence_imgs,_out1dfit,_out1dfit_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Args:
    """
    global original,sigmas,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, \
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape
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

    out1dfit = _out1dfit
    persistence = persistence_imgs

    # parameters for each image (PA, wavelegnth, image center, image number)
    lambdas = wvs_imgs
    psfs = psfs_stamps
    psfs_shape = psfs_stamps_shape
    Npixproc= 0
    Npixtot=0


def _remove_bad_pixels_z(col_index,nan_mask_boxsize,dtype,window_size=50,threshold=7.):
    global original,sigmas,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, \
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape
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


def _HPF_z(col_index,cutoff,dtype):
    global original,sigmas,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, \
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    originalLPF_np = _arraytonumpy(originalLPF, original_shape,dtype=dtype)
    originalHPF_np = _arraytonumpy(originalHPF, original_shape,dtype=dtype)
    for m in np.arange(0,original_shape[0]):
    # if 1:
    #     m = 30
        myvec = original_np[m,col_index,:]
        fftmyvec = np.fft.fft(np.concatenate([myvec,myvec[::-1]],axis=0))
        LPF_fftmyvec = copy(fftmyvec)
        LPF_fftmyvec[cutoff:(2*np.size(myvec)-cutoff+1)] = 0
        LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec)]
        HPF_myvec = myvec - LPF_myvec
        originalLPF_np[m,col_index,:] = LPF_myvec
        originalHPF_np[m,col_index,:] = HPF_myvec
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
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape
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
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape
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

def LPFvsHPF(myvec,cutoff):
    myvec_cp = copy(myvec)
    #handling nans:
    wherenans = np.where(np.isnan(myvec_cp))
    for k in wherenans[0]:
        myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-10]):np.min([np.size(myvec_cp),k+10])])

    fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec

    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan
    return LPF_myvec,HPF_myvec

# plcen_k_valid_pix[::-1],plcen_l_valid_pix[::-1],row_valid_pix[::-1],col_valid_pix[::-1],
#                                     normalized_psfs_func_list,
#                                     hr8799modelspec_Rlist,
#                                     transmission_Rlist,
#                                     planet_template_func_Rlist,
#                                     func_skytrans_Rlist,
#                                     wvs_imgs,planetRV_array,
#                                     dtype,cutoff,planet_search,(plcen_k,plcen_l),
#                                     R_list,
#                                     wvsol_shifts,wvsol_offsets)
def _process_pixels_onlyHPF(curr_k_indices,curr_l_indices,row_indices,col_indices,
                            normalized_psfs_func_list,
                            transmission_table,
                            planet_model_func_table,
                            HR8799pho_spec_func_list,
                            transmission4planet_list,
                            hr8799_flux,
                            wvs,planetRV_array,dtype,cutoff,planet_search,centroid_guess,
                            R_list,wvsol_shifts,wvsol_offsets,R_calib_arr=None,model_persistence=False):
    global original,sigmas,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, \
        psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence,out1dfit,out1dfit_shape
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
    # print("coucou2")

    tmpwhere = np.where(np.isfinite(badpix_np))
    chi2ref = 0#np.nansum((originalHPF_np[tmpwhere]/sigmas_imgs_np[tmpwhere])**2)

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
        data_R_calib = R_calib_arr[k-w:k+w+1,l-w:l+w+1]
        if model_persistence:
            data_persistence = persistence_np[k-w:k+w+1,l-w:l+w+1]
        data_ny,data_nx,data_nz = HPFdata.shape


        # import matplotlib.pyplot as plt
        # for k in range(5):
        #     for l in range(5):
        #         plt.subplot(5,5,5*k+l+1)
        #         plt.plot(HPFdata[k,l,:],label="HPF")
        #         wherefinite = np.where(np.isfinite(data_badpix[k,l,:]))
        #         plt.plot(wherefinite[0],HPFdata[k,l,:][wherefinite],label="BP after")
        # plt.legend()
        # plt.show()

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
            for custwvoff_id,custom_wvoffset in enumerate(wvsol_shifts):
                # print("coucou5")
                HPFmodelH0_list = []

                for transmission in tr_list:
                    HR8799_obsspec = transmission(wvs-custom_wvoffset) * \
                                    HR8799pho_spec_func(wvs-custom_wvoffset)
                    LPF_HR8799_obsspec,HPF_HR8799_obsspec = LPFvsHPF(HR8799_obsspec,cutoff)

                    # import matplotlib.pyplot as plt
                    # plt.plot(transmission(wvs-data_wvsol_offsets[bkg_k,bkg_l]-custom_wvoffset),label="trans")
                    # plt.plot(HR8799pho_spec_func(wvs-data_wvsol_offsets[bkg_k,bkg_l]-custom_wvoffset),label="HR8799")
                    # plt.plot(HR8799_obsspec,label="HR8799 model")
                    # plt.plot(LPF_HR8799_obsspec,label="LPF HR8799 model")
                    # plt.plot(HPF_HR8799_obsspec,label="HPF HR8799 model")

                    line_spec = HPF_HR8799_obsspec/LPF_HR8799_obsspec

                    # plt.plot(line_spec,label="line_spec")

                    bkg_model = np.zeros((2*w+1,2*w+1,2*w+1,2*w+1,data_nz))
                    for bkg_k in range(2*w+1):
                        for bkg_l in range(2*w+1):
                            myspec = LPFdata[bkg_k,bkg_l,:]*line_spec

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

                HPFmodel_H0 = np.concatenate(HPFmodelH0_list,axis=1)



    
                HPFmodel_H0 = HPFmodel_H0[where_finite_data[0],:]/sigmas_vec[:,None]
    
                noplrv_id = np.argmin(np.abs(planetRV_array))
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
                                                       -data_wvsol_offsets[bkg_k,bkg_l]-custom_wvoffset

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
                                            tr4planet(wvs-custom_wvoffset)
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

                            # import matplotlib.pyplot as plt
                            # plt.plot(3e-5*HPF_planet_model.ravel(),label="model")
                            # plt.plot(HPFdata.ravel(),label="data")
                            # plt.legend()
                            # plt.show()

                            HPFmodelH1_list.append((HPF_planet_model.ravel())[:,None])

                        # import matplotlib.pyplot as plt
                        # for HPFmodelH1 in HPFmodelH1_list:
                        #     plt.plot(HPFmodelH1)
                        # plt.show()

                        HPFmodel_H1only = np.concatenate(HPFmodelH1_list,axis=1)
    
                        HPFmodel_H1only = HPFmodel_H1only[where_finite_data[0],:]/sigmas_vec[:,None]
                        HPFmodel_H1only[np.where(np.isnan(HPFmodel_H1only))] = 0
                        HPFmodel_H0[np.where(np.isnan(HPFmodel_H0))] = 0

                        HPFmodel = np.concatenate([HPFmodel_H1only,HPFmodel_H0],axis=1)

                        # print(np.sum(np.abs(HPFmodel),axis=0))
                        # print(np.sum(np.abs(HPFmodel),axis=0)!=0)
                        where_valid_parameters = np.where(np.sum(np.abs(HPFmodel),axis=0)!=0)
                        HPFmodel = HPFmodel[:,where_valid_parameters[0]]
                        where_valid_parameters = np.where(np.sum(np.abs(HPFmodel_H0),axis=0)!=0)
                        HPFmodel_H0 = HPFmodel_H0[:,where_valid_parameters[0]]


                        HPFparas,HPFchi2,rank,s = np.linalg.lstsq(HPFmodel,ravelHPFdata,rcond=None)
                        HPFparas_H0,HPFchi2_H0,rank,s = np.linalg.lstsq(HPFmodel_H0,ravelHPFdata,rcond=None)
                        # print("H1",HPFparas)
                        # print("H0",HPFparas_H0)
                        # exit()
    
                        data_model = np.dot(HPFmodel,HPFparas)
                        data_model_H0 = np.dot(HPFmodel_H0,HPFparas_H0)
                        deltachi2 = 0#chi2ref-np.sum(ravelHPFdata**2)
                        ravelresiduals = data_model-ravelHPFdata
                        ravelresiduals_H0 = data_model_H0-ravelHPFdata
                        HPFchi2 = np.nansum((ravelresiduals)**2)
                        HPFchi2_H0 = np.nansum((ravelresiduals_H0)**2)


                        # import matplotlib.pyplot as plt
                        # print(HPFchi2,HPFchi2_H0)
                        # plt.figure(1)
                        # print(HPFmodel_H0.shape)
                        # plt.plot(ravelHPFdata,label="data")
                        # for k in range(HPFmodel.shape[1]):
                        #     plt.plot(HPFmodel[:,k],label="{0}".format(k),alpha=0.5)
                        # plt.legend()
                        # plt.figure(2)
                        # plt.plot(ravelresiduals_H0,label="res H0")
                        # plt.plot(ravelresiduals,label="res")
                        # plt.legend()
                        # plt.show()

                        if plrv_id == noplrv_id:
                            canvas_model = np.zeros((2*w+1,2*w+1,data_nz))
                            canvas_model.shape = ((2*w+1)*(2*w+1)*data_nz,)
                            canvas_model[where_finite_data] = (data_model-HPFparas[0]*HPFmodel[:,0])*sigmas_vec
                            canvas_model.shape = (2*w+1,2*w+1,data_nz)
                            canvas_planet = np.zeros((2*w+1,2*w+1,data_nz))
                            canvas_planet.shape = ((2*w+1)*(2*w+1)*data_nz,)
                            for plmod_id in range(len(planet_partial_template_func_list)):
                                canvas_planet[where_finite_data] += HPFparas[plmod_id]*HPFmodel[:,plmod_id]*sigmas_vec
                            canvas_planet.shape = (2*w+1,2*w+1,data_nz)
                            canvas_HPFdata = copy(HPFdata)
                            canvas_HPFdata.shape = ((2*w+1)*(2*w+1)*data_nz,)
                            canvas_HPFdata[where_bad_data] = 0
                            canvas_HPFdata.shape = (2*w+1,2*w+1,nl)
                            canvas_sigmas = copy(data_sigmas)
                            canvas_sigmas.shape = ((2*w+1)*(2*w+1)*data_nz,)
                            canvas_sigmas[where_bad_data] = 0
                            canvas_sigmas.shape = (2*w+1,2*w+1,nl)
                            canvas_norma_res = np.zeros((2*w+1,2*w+1,data_nz))
                            canvas_norma_res.shape = ((2*w+1)*(2*w+1)*data_nz,)
                            canvas_norma_res[where_finite_data] = ravelHPFdata-data_model
                            canvas_norma_res.shape = (2*w+1,2*w+1,data_nz)
    
                            final_template = np.nansum(canvas_model,axis=(0,1))
                            final_HPFdata = np.nansum(canvas_HPFdata,axis=(0,1))
                            final_sigmas = np.nansum(canvas_sigmas,axis=(0,1))
                            final_planet = np.nansum(canvas_planet,axis=(0,1))
                            final_res = final_HPFdata-final_template
                            final_norma_res = np.nansum(canvas_norma_res,axis=(0,1))
                            outres_np[custwvoff_id,model_id,0,:,row,col] = final_HPFdata
                            outres_np[custwvoff_id,model_id,1,:,row,col] = final_template
                            outres_np[custwvoff_id,model_id,2,:,row,col] = final_res
                            outres_np[custwvoff_id,model_id,3,:,row,col] = final_planet
                            outres_np[custwvoff_id,model_id,4,:,row,col] = final_sigmas
                            outres_np[custwvoff_id,model_id,5,:,row,col] = final_norma_res

                            # #remove
                            # import matplotlib.pyplot as plt
                            # # plt.subplot(1,5,custwvoff_id+1 )
                            # plt.plot(final_norma_res,label="{0}".format(custom_wvoffset))
                            # print(custom_wvoffset,np.nanstd(final_norma_res))
                            # plt.ylim([-2,2])
                            # plt.show()

    
                            res_ccf = np.correlate(ravelresiduals,ravelresiduals,mode="same")
                            res_ccf_argmax = np.argmax(res_ccf)
                            outautocorrres_np[:,row,col] = res_ccf[(res_ccf_argmax-500):(res_ccf_argmax+500)]
    
                        Npixs_HPFdata = HPFmodel.shape[0]
                        minus2logL_HPF = Npixs_HPFdata*(1+np.log(HPFchi2/Npixs_HPFdata)+logdet_Sigma+np.log(2*np.pi))
                        minus2logL_HPF_H0 = Npixs_HPFdata*(1+np.log(HPFchi2_H0/Npixs_HPFdata)+logdet_Sigma+np.log(2*np.pi))
                        AIC_HPF = 2*(HPFmodel.shape[-1])+minus2logL_HPF
                        AIC_HPF_H0 = 2*(HPFmodel_H0.shape[-1])+minus2logL_HPF_H0
    
                        covphi =  HPFchi2/Npixs_HPFdata*np.linalg.inv(np.dot(HPFmodel.T,HPFmodel))
                        slogdet_icovphi0 = np.linalg.slogdet(np.dot(HPFmodel.T,HPFmodel))

                        # delta AIC ~ likelihood ratio
                        if HPFparas[0]>0:
                            output_maps_np[custwvoff_id,model_id,0,row,col,plrv_id] = AIC_HPF_H0-AIC_HPF
                        else:
                            output_maps_np[custwvoff_id,model_id,0,row,col,plrv_id] = 0
                        # AIC for planet + star model
                        output_maps_np[custwvoff_id,model_id,1,row,col,plrv_id] = AIC_HPF
                        # AIC for star model only
                        output_maps_np[custwvoff_id,model_id,2,row,col,plrv_id] = AIC_HPF_H0
                        # Chi2 of the stamp
                        output_maps_np[custwvoff_id,model_id,3,row,col,plrv_id] = HPFchi2
                        # Size of the data
                        output_maps_np[custwvoff_id,model_id,4,row,col,plrv_id] = Npixs_HPFdata
                        # number of parameters of the model
                        output_maps_np[custwvoff_id,model_id,5,row,col,plrv_id] = HPFmodel.shape[-1]
                        # estimated scaling factor of the covariance matrix of the data
                        output_maps_np[custwvoff_id,model_id,6,row,col,plrv_id] = HPFchi2/Npixs_HPFdata
                        #
                        output_maps_np[custwvoff_id,model_id,7,row,col,plrv_id] = logdet_Sigma
                        output_maps_np[custwvoff_id,model_id,8,row,col,plrv_id] =  slogdet_icovphi0[0]*slogdet_icovphi0[1]
                        # marginalized posterior
                        output_maps_np[custwvoff_id,model_id,9,row,col,plrv_id] = -0.5*logdet_Sigma-0.5*slogdet_icovphi0[1]- (Npixs_HPFdata-HPFmodel.shape[-1]+2-1)/(2)*np.log(HPFchi2+deltachi2)
                        for plmod_id in range(len(planet_partial_template_func_list)):
                            # print(plmod_id)
                            # SNR
                            output_maps_np[custwvoff_id,model_id,10+plmod_id*3,row,col,plrv_id] = HPFparas[plmod_id]/np.sqrt(np.abs(covphi[plmod_id,plmod_id]))
                            # estimated planet to star flux ratio
                            output_maps_np[custwvoff_id,model_id,10+plmod_id*3+1,row,col,plrv_id] = HPFparas[plmod_id]
                            # error bar on estimated planet to star flux ratio
                            output_maps_np[custwvoff_id,model_id,10+plmod_id*3+2,row,col,plrv_id] = np.sign(covphi[plmod_id,plmod_id])*np.sqrt(np.abs(covphi[plmod_id,plmod_id]))
                        # print(output_maps_np[custwvoff_id,model_id,:,row,col,plrv_id])
                        # print(HPFchi2,deltachi2)
                        # exit()
                    # except:
                    #     pass
        # #remove
        # plt.legend()
        # plt.show()
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
    psfs,xs,ys,xvec,yvec = paras
    normalized_psfs_func_list = []
    for wv_index in range(psfs.shape[-1]):
        # if 1:#np.isnan(psf_func(0,0)[0,0]):
        #     if wv_index <1660:
        #         continue
        #     model_psf = psfs[:,:, :, wv_index]
        #     import matplotlib.pyplot as plt
        #     from mpl_toolkits.mplot3d import Axes3D
        #     fig = plt.figure(1)
        #     ax = fig.add_subplot(111,projection="3d")
        #     for k,color in zip(range(model_psf.shape[0]),["pink","blue","green","purple","orange"]):
        #         ax.scatter(xs[k].ravel(),ys[k].ravel(),model_psf[k].ravel(),c=color)
        #     # plt.show()
        model_psf = psfs[:,:, :, wv_index].ravel()
        where_finite = np.where(np.isfinite(model_psf))
        psf_func = interpolate.LSQBivariateSpline(xs.ravel()[where_finite],ys.ravel()[where_finite],model_psf[where_finite],xvec,yvec)
        # if 1:
        #     print(psf_func(0,0))
        #     x_psf_vec, y_psf_vec = np.arange(2*nx_psf * 1.)/2.-nx_psf//2, np.arange(2*ny_psf* 1.)/2.-ny_psf//2
        #     x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
        #     ax.scatter(x_psf_grid.ravel(),y_psf_grid.ravel(),psf_func(x_psf_vec,y_psf_vec).transpose().ravel(),c="red")
        #     plt.show()
        normalized_psfs_func_list.append(psf_func)
    # print(len(normalized_psfs_func_list))
    return normalized_psfs_func_list

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
    if 1:# HR 8799 c 20100715
        # planet = "b"
        planet = "c"
        date = "100715"
        # date = "101104"
        # date = "110723"
        # planet = "d"
        # date = "150720"
        # date = "150722"
        # date = "150723"
        # date = "150828"
        IFSfilter = "Kbb"
        # IFSfilter = "Hbb" # "Kbb" or "Hbb"

        inputDir = "/data/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb/"
        outputdir = "/data/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb/20190308_HPF_only/"
        # outputdir = "/data/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb/20190305_HPF_only_noperscor/"
        # outputdir = "/data/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb/20190228_mol_temp/"

        # inputDir = "/data/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb_pairsub/"
        # outputdir = "/data/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb_pairsub/20190228_HPF_only/"

        filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_020.fits"))
        filelist.sort()
        # print(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_020.fits"))
        filelist = filelist[0:1]
        # filelist = filelist[len(filelist)-3:len(filelist)-2]

        numthreads = 28
        planet_search = True
        planet_model_string = "model"
        # planet_model_string = "CO"#"CO2 CO H2O CH4"

        planet = "c" # for selection of the templates
        pairsub = "pairsub" in inputDir
        # print(pairsub)
        # exit()

        osiris_data_dir = "/data/osiris_data"
    else:
        inputDir = sys.argv[1]
        outputdir = sys.argv[2]
        filename = sys.argv[3]
        numthreads = int(sys.argv[4])
        planet_search = bool(int(sys.argv[5]))

        filelist = [filename]
        IFSfilter = filename.split("_")[-2]
        template_spec_filename=os.path.join(os.path.dirname(filename),"..","..",
                                            "HR8799c_"+IFSfilter[0:1]+"_3Oct2018_conv"+IFSfilter+".csv")

        osiris_data_dir = None
        #nice -n 15 /home/anaconda3/bin/python ./reduce_HPFonly_diagcov.py /data/osiris_data/HR_8799_c/20100715/reduced_jb/ /data/osiris_data/HR_8799_c/20100715/reduced_jb/20181205_HPF_only_sherlock_test/ /data/osiris_data/HR_8799_c/20100715/reduced_jb/s100715_a010001_Kbb_020.fits 20 1


    for filename in filelist:
        print("Processing "+filename)
        
        ##############################
        ## Read OSIRIS spectral cube
        ##############################
        with pyfits.open(filename) as hdulist:
            imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
            prihdr = hdulist[0].header
            curr_mjdobs = prihdr["MJD-OBS"]
            imgs = np.moveaxis(imgs,0,2)
            imgs_hdrbadpix = np.moveaxis(np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1),0,2)
        ny,nx,nz = imgs.shape
        init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
        dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
        wvs=np.arange(init_wv,init_wv+dwv*nz-1e-6,dwv)
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
            R0=5000
        dwv = CDELT1/1000.

        debug = False
        if debug:
            planet_search = False
        debug_paras = True
        model_based_sky_trans = False
        use_wvsol_offsets = False
        use_R_calib = False
        mask_starline = False
        model_persistence = True
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
                    planet_model_string.replace(molecule,"")
            # print(molecules_list)
            # exit(0)

        padding = 5
        nan_mask_boxsize=3
        cutoff = 40
        dtype = ctypes.c_double

        if mask_starline:
            suffix = "HPF_cutoff{0}_sherlock_v1_starline".format(cutoff)
        else:
            suffix = "HPF_cutoff{0}_sherlock_v1".format(cutoff)

        if IFSfilter == "Hbb":
            hr8799_mag = 5.240
        elif IFSfilter == "Kbb":
            hr8799_mag = 5.240
        else:
            raise("IFS filter name unknown")
        hr8799_type = "F0"
        hr8799_rv = -12.6 #+-1.4
        c_kms = 299792.458
        dprv = 3e5*dwv/(init_wv+dwv*nz//2) # 38.167938931297705
        
        ##############################
        ## Dependent parameters
        ##############################
        specpool = mp.Pool(processes=numthreads)

        phoenix_folder = os.path.join(osiris_data_dir,"phoenix")
        planet_template_folder = os.path.join(osiris_data_dir,"planets_templates")
        molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
        sky_transmission_folder = os.path.join(osiris_data_dir,"sky_transmission")
        ref_star_folder = os.path.join(os.path.dirname(filename),"..","reduced_telluric_jb")
        fileinfos_filename = os.path.join(inputDir,"..","..","fileinfos_"+IFSfilter+"_jb.csv")
        fileinfos_refstars_filename = os.path.join(osiris_data_dir,"fileinfos_refstars_jb.csv")

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
            wvsol_shifts = np.array([0,])

            wvsol_offsets_inputdir = os.path.join(inputDir,"..","..")
            wvsol_offsets_filename = os.path.join(wvsol_offsets_inputdir,"master_wvshifts.fits")
            hdulist = pyfits.open(wvsol_offsets_filename)
            wvsol_offsets = hdulist[0].data
            hdulist.close()
        else:
            if debug_paras:
                wvsol_shifts = np.array([0,])
                # wvsol_shifts = np.linspace(-2*dwv,2*dwv,5)
            else:
                wvsol_shifts = np.array([0,])
                # wvsol_shifts = np.linspace(-2*dwv,2*dwv,5)

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
        hr8799_bary_rv = -float(fileitem[baryrv_id])/1000
        # print(hr8799_bary_rv)
        # exit()

        if planet_search:
            suffix = suffix+"_search"
            if debug_paras:
                planetRV_array = np.array([hr8799_bary_rv-10])
                # planetRV_array = np.concatenate([np.arange(-2*dprv,2*dprv,dprv/10)])
            else:
                planetRV_array = np.concatenate([np.arange(-2*dprv,2*dprv,dprv/100),np.arange(-100*dprv,100*dprv,dprv)])
            plcen_k,plcen_l = np.nan,np.nan
        else:
            suffix = suffix+"_centroid"

            if debug_paras:
                planetRV_array = np.array([hr8799_bary_rv-10,])
                dl_grid,dk_grid = np.array([[0]]),np.array([[0]])
                # plcen_k,plcen_l = float(fileitem[kcen_id]),float(fileitem[lcen_id])
                # plcen_k,plcen_l = 32-10,-35.79802955665025+46.8
                plcen_k,plcen_l = 28,4#44,8
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
        ## refstars info (barycenter correction)
        ##############################
        with open(fileinfos_refstars_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            refstarsinfo_list_table = list(csv_reader)
            refstarsinfo_colnames = refstarsinfo_list_table[0]
            refstarsinfo_list_data = refstarsinfo_list_table[1::]
        refstarsinfo_filename_id = refstarsinfo_colnames.index("filename")
        refstars_filelist = [os.path.basename(item[refstarsinfo_filename_id]) for item in refstarsinfo_list_data]

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
                                                  "lte11-4.0_hr8799"+planet+"_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7")
                    travis_mol_filename_D2E=os.path.join(molecular_template_folder,
                                                  "lte11-4.0_hr8799"+planet+"_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7_D2E")
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

                        # import matplotlib.pyplot as plt
                        # plt.plot(wmod,mol_temp)#,data[::100,1])
                        # print(mol_temp.shape)
                        # plt.show()
                        # exit()
                        print("convolving: "+mol_template_filename)
                        planet_convspec = convolve_spectrum(wmod,mol_temp,R,specpool)

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
                    # exit()
                if len(molecules_list) >= 2:
                    print("uh...")
                    exit()
                    # travis_spec_filename=os.path.join(planet_template_folder,
                    #                                   "HR8799"+planet+"_"+IFSfilter[0:1]+"_3Oct2018.save")
                    # planet_template_filename=travis_spec_filename.replace(".save",
                    #                                                       "_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))
                    #
                    # if use_R_calib:
                    #     travis_spectrum = scio.readsav(travis_spec_filename)
                    #     ori_planet_spec = np.array(travis_spectrum["fmod"])
                    #     ori_planet_convspec = np.array(travis_spectrum["fmods"])
                    #     wmod = np.array(travis_spectrum["wmod"])/1.e4
                    #     planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
                    #
                    #     planet_partial_template_func_list.append(planet_spec_func)
                    # else:
                    #     if len(glob.glob(planet_template_filename)) == 0:
                    #         travis_spectrum = scio.readsav(travis_spec_filename)
                    #         ori_planet_spec = np.array(travis_spectrum["fmod"])
                    #         ori_planet_convspec = np.array(travis_spectrum["fmods"])
                    #         wmod = np.array(travis_spectrum["wmod"])/1.e4
                    #         print("convolving: "+planet_template_filename)
                    #         planet_convspec = convolve_spectrum(wmod,ori_planet_spec,R,specpool)
                    #
                    #         with open(planet_template_filename, 'w+') as csvfile:
                    #             csvwriter = csv.writer(csvfile, delimiter=' ')
                    #             csvwriter.writerows([["wvs","spectrum"]])
                    #             csvwriter.writerows([[a,b] for a,b in zip(wmod,planet_convspec)])
                    #
                    #     with open(planet_template_filename, 'r') as csvfile:
                    #         csv_reader = csv.reader(csvfile, delimiter=' ')
                    #         list_starspec = list(csv_reader)
                    #         oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                    #         col_names = oriplanet_spec_str_arr[0]
                    #         oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
                    #         oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
                    #         where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                    #         oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                    #         continuum = LPFvsHPF(oriplanet_spec,cutoff)[0]
                    #         continuum_func = interp1d(oriplanet_spec_wvs,continuum,bounds_error=False,fill_value=np.nan)
                    #
                    # # # continuum = np.nanmax(np.array([temp_func(oriplanet_spec_wvs) for temp_func in planet_partial_template_func_list]),axis=0)
                    # # import matplotlib.pyplot as plt
                    # # plt.plot(oriplanet_spec_wvs,continuum)#,data[::100,1])
                    # # plt.show()
                    # planet_partial_template_func_list.insert(0,interp1d(oriplanet_spec_wvs,continuum,bounds_error=False,fill_value=np.nan))
            else:
                travis_spec_filename=os.path.join(planet_template_folder,
                                                  "HR8799"+planet+"_"+IFSfilter[0:1]+"_3Oct2018.save")
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
                        ori_planet_spec = np.array(travis_spectrum["fmod"])
                        ori_planet_convspec = np.array(travis_spectrum["fmods"])
                        wmod = np.array(travis_spectrum["wmod"])/1.e4
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
                        planet_partial_template_func_list.append(planet_spec_func)
            planet_model_func_table.append(planet_partial_template_func_list)

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
        # refstar_name_filter = "HIP_1123"
        refstar_name_filter = "*"
        spdc_refstar_filelist = glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"s*"+IFSfilter+"_020.fits"))
        spdc_refstar_filelist.sort()
        psfs_refstar_filelist = glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"s*"+IFSfilter+"_020_psfs_badpix2.fits"))
        psfs_refstar_filelist.sort()
        spec_refstar_filelist = glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"s*"+IFSfilter+"_020_psfs_repaired.fits"))
        spec_refstar_filelist.sort()
        centers_refstar_filelist = glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"s*"+IFSfilter+"_020_psfs_centers.fits"))
        centers_refstar_filelist.sort()

        if model_persistence:
            persistence_arr = np.zeros((ny,nx,nz))
            persistence_filelist = glob.glob(os.path.join(ref_star_folder,"*","s*"+IFSfilter+"_020.fits"))
            persistence_filelist.extend(glob.glob(os.path.join(ref_star_folder,"*","hacked_persistence_*"+IFSfilter+"_020.fits")))
            # for spdc_refstar_filename in persistence_filelist[0:1]:
            #     print(spdc_refstar_filename)
            #     with pyfits.open(spdc_refstar_filename) as hdulist:
            #         coucou = copy(hdulist[0].header)
            #         print(coucou["MJD-OBS"])
            for spdc_refstar_filename in persistence_filelist:
                # print(spdc_refstar_filename)
                with pyfits.open(spdc_refstar_filename) as hdulist:
                    spdc_refstar_prihdr = hdulist[0].header
                    print(spdc_refstar_prihdr["MJD-OBS"],spdc_refstar_prihdr["MJD-OBS"] < curr_mjdobs,spdc_refstar_filename)
                    if spdc_refstar_prihdr["MJD-OBS"] < curr_mjdobs:
                        spdc_refstar_cube = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                        spdc_refstar_cube = np.moveaxis(spdc_refstar_cube,0,2)
                        spdc_refstar_im = np.nansum(spdc_refstar_cube,axis=2)
                        persis_where2mask = np.where(spdc_refstar_im<np.nanmax(spdc_refstar_im)/10)
                        spdc_refstar_cube[persis_where2mask[0],persis_where2mask[1],:] = 0
                        persistence_arr += spdc_refstar_cube
                        # import matplotlib.pyplot as plt
                        # plt.figure(1)
                        # plt.imshow(spdc_refstar_cube[:,:,100],interpolation="nearest")
                        # plt.show()

                    if 0:
                        print(coucou["MJD-OBS"])
                        # tmpdata = np.zeros(hdulist[0].data.shape)
                        # maxx,maxy = np.unravel_index(np.nanargmax(np.nansum(hdulist[0].data,axis=2)),(nx,ny))
                        # newmaxy,newmaxx = 33,9
                        # tmpdata[newmaxx-5:newmaxx+5,newmaxy-5:newmaxy+5,:] = hdulist[0].data[maxx-5:maxx+5,maxy-5:maxy+5,:]

                        hdulist2 = pyfits.HDUList()
                        hdulist2.append(pyfits.PrimaryHDU(data=hdulist[0].data,header=coucou))
                        try:
                            hdulist2.writeto(os.path.join(os.path.dirname(spdc_refstar_filename),"hacked_persistence_s100715_a005001_"+IFSfilter+"_020.fits"), overwrite=True)
                        except TypeError:
                            hdulist2.writeto(os.path.join(os.path.dirname(spdc_refstar_filename),"hacked_persistence_s100715_a005001_"+IFSfilter+"_020.fits"), clobber=True)
                        hdulist2.close()
                        exit()

            # import matplotlib.pyplot as plt
            # # spdc_refstar_im[persis_where2mask]=np.nan
            # # plt.imshow(spdc_refstar_im,interpolation="nearest")
            # plt.figure(1)
            # plt.imshow(persistence_arr[:,:,100],interpolation="nearest")
            # plt.show()
            # exit()
            # # # plt.figure(2)
            # # # plt.plot(persistence_arr[48,8,:],label="before")

            window_size=100
            threshold=7
            for m in range(ny):
                for n in range(nx):
                    myvec = copy(persistence_arr[m,n,:])
                    smooth_vec = median_filter(myvec,footprint=np.ones(window_size),mode="reflect")
                    myvec = myvec - smooth_vec
                    wherefinite = np.where(np.isfinite(myvec))
                    mad = mad_std(myvec[wherefinite])
                    whereoutliers = np.where(np.abs(myvec)>threshold*mad)[0]
                    persistence_arr[m,n,whereoutliers] = np.nan
                    widen_badpix_vec = np.correlate(persistence_arr[m,n,:],np.ones(nan_mask_boxsize),mode="same")
                    widen_nans = np.where(np.isnan(widen_badpix_vec))[0]
                    persistence_arr[m,n,widen_nans] = np.nan
                    persistence_arr[m,n,widen_nans] = smooth_vec[widen_nans]

            # plt.plot(persistence_arr[48,8,:],label="after")
            # plt.legend()
            # plt.show()

        phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
        with pyfits.open(phoenix_wv_filename) as hdulist:
            phoenix_wvs = hdulist[0].data/1.e4
        crop_phoenix = np.where((phoenix_wvs>wvs[0]-(wvs[-1]-wvs[0])/2)*(phoenix_wvs<wvs[-1]+(wvs[-1]-wvs[0])/2))
        phoenix_wvs = phoenix_wvs[crop_phoenix]


        psfs_refstar_list = []
        transmission_table = []
        transmission4planet_list = []
        HR8799pho_spec_func_list = []
        hr8799_flux_list = []
        for Rid,R in enumerate(R_list):
            transmission_list = []
            for ori_refstar_filename,psfs_refstar_filename,spec_refstar_filename in \
                    zip(spdc_refstar_filelist,psfs_refstar_filelist,spec_refstar_filelist):
                refstar_name = ori_refstar_filename.split(os.path.sep)[-2]

                if refstar_name == "HD_210501":
                    refstar_RV = -20.20 #+-2.5
                    ref_star_type = "A0"
                    if IFSfilter == "Hbb":
                        refstar_mag = 7.606
                    elif IFSfilter == "Kbb":
                        refstar_mag = 7.597
                elif refstar_name == "HIP_1123":
                    refstar_RV = -0.9 #+-2
                    ref_star_type = "A1"
                    if IFSfilter == "Hbb":
                        refstar_mag = 6.219
                    elif IFSfilter == "Kbb":
                        refstar_mag = 6.189
                elif refstar_name == "HIP_116886":
                    refstar_RV = 0 #unknown
                    ref_star_type = "A5"
                    if IFSfilter == "Hbb":
                        refstar_mag = 9.212
                    elif IFSfilter == "Kbb":
                        refstar_mag = 9.189
                else:
                    raise(Exception("Ref star name unknown"))
                refstarsinfo_fileid = refstars_filelist.index(os.path.basename(ori_refstar_filename))
                refstarsinfo_fileitem = refstarsinfo_list_data[refstarsinfo_fileid]
                refstarsinfo_baryrv_id = refstarsinfo_colnames.index("barycenter rv")
                refstarsinfo_bary_rv = -float(refstarsinfo_fileitem[refstarsinfo_baryrv_id])/1000

                ##############################
                ## Reference star phoenix model
                ##############################
                phoenix_model_refstar_filename = glob.glob(os.path.join(phoenix_folder,refstar_name+"*.fits"))[0]
                phoenix_refstar_filename=phoenix_model_refstar_filename.replace(".fits","_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))

                if len(glob.glob(phoenix_refstar_filename)) == 0:
                    with pyfits.open(phoenix_model_refstar_filename) as hdulist:
                        phoenix_refstar = hdulist[0].data[crop_phoenix]
                    print("convolving: "+phoenix_model_refstar_filename)
                    phoenix_refstar_conv = convolve_spectrum(phoenix_wvs,phoenix_refstar,R,specpool)

                    with open(phoenix_refstar_filename, 'w+') as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=' ')
                        csvwriter.writerows([["wvs","spectrum"]])
                        csvwriter.writerows([[a,b] for a,b in zip(phoenix_wvs,phoenix_refstar_conv)])

                with open(phoenix_refstar_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=' ')
                    list_starspec = list(csv_reader)
                    refstarpho_spec_str_arr = np.array(list_starspec, dtype=np.str)
                    col_names = refstarpho_spec_str_arr[0]
                    refstarpho_spec = refstarpho_spec_str_arr[1::,1].astype(np.float)
                    refstarpho_spec_wvs = refstarpho_spec_str_arr[1::,0].astype(np.float)
                    where_IFSfilter = np.where((refstarpho_spec_wvs>wvs[0])*(refstarpho_spec_wvs<wvs[-1]))
                    refstarpho_spec = refstarpho_spec/np.mean(refstarpho_spec[where_IFSfilter])
                    refstarpho_spec_func = interp1d(refstarpho_spec_wvs,refstarpho_spec,bounds_error=False,fill_value=np.nan)

                ##############################
                ## Reference star OSIRIS
                ##############################
                with pyfits.open(spec_refstar_filename) as hdulist:
                    spec_refstar = np.nansum(hdulist[0].data,axis=(1,2))
                with pyfits.open(psfs_refstar_filename) as hdulist:
                    psfs = hdulist[0].data
                    psfs = np.moveaxis(psfs,0,2)
                    psfs_spec = np.nansum(psfs,axis=(0,1))
                # import matplotlib.pyplot as plt
                # plt.subplot(1,2,1)
                # plt.plot(spec_refstar,label="repaired")
                # plt.plot(psfs_spec,label="psfs")
                # plt.subplot(1,2,2)
                # plt.plot(np.abs(psfs_spec-spec_refstar)/spec_refstar)
                # plt.show()
                if Rid == 0:
                    hr8799_flux_list.append(np.nansum(spec_refstar)* 10**(-1./2.5*(hr8799_mag-refstar_mag)))
                if Rid == 0:
                    psfs_refstar_list.append(psfs/spec_refstar[None,None,:])
                where_bad_slices = np.where(np.abs(psfs_spec-spec_refstar)/spec_refstar>0.01)
                if len(where_bad_slices[0])<0.1*nz:
                    spec_refstar[where_bad_slices] = np.nan

                    # import matplotlib.pyplot as plt
                    # print(R)
                    # tmp = spec_refstar
                    # tmp = tmp/np.nanmean(tmp)
                    # plt.plot(tmp,label="spec_refstar")
                    # plt.plot(refstarpho_spec_func(wvs*(1-(refstarsinfo_bary_rv)/c_kms)),label="refstarpho bary")
                    # tmp = refstarpho_spec_func(wvs*(1-(refstar_RV+refstarsinfo_bary_rv)/c_kms))
                    # tmp = tmp/np.nanmean(tmp)
                    # plt.plot(tmp,label="refstarpho bary + RV")
                    # tmp = refstarpho_spec_func(wvs*(1-(refstar_RV+38/2+refstarsinfo_bary_rv)/c_kms))
                    # tmp = tmp/np.nanmean(tmp)
                    # plt.plot(tmp,label="refstarpho bary + RV ++")
                    # tmp = refstarpho_spec_func(wvs*(1-(refstar_RV-38/2+refstarsinfo_bary_rv)/c_kms))
                    # tmp = tmp/np.nanmean(tmp)
                    # plt.plot(tmp,label="refstarpho bary + RV --")
                    # plt.legend()
                    # plt.show()
                    transmission = spec_refstar / refstarpho_spec_func(wvs*(1-(refstar_RV+refstarsinfo_bary_rv)/c_kms))
                    transmission = transmission/np.nanmean(transmission)
                    # print(refstar_name)
                    # if refstar_name == "HD_210501":
                    #     transmission[795:810] = np.nan

                    transmission_list.append(transmission)

            # # #remove
            # import matplotlib.pyplot as plt
            # print(R)
            # plt.figure(1)
            # for transid,trans in enumerate(transmission_list):
            #     plt.plot(trans,label="{0}".format(transid))
            # plt.legend()
            # plt.show()

            mean_transmission_func = interp1d(wvs,np.nanmean(np.array(transmission_list),axis=0),bounds_error=False,fill_value=np.nan)
            transmission_func_list = [mean_transmission_func]
            transmission4planet_list.append(mean_transmission_func)
            transmission_table.append(transmission_func_list)

            # extract phoenix model for HR8799
            phoenix_model_HR8799_filename = glob.glob(os.path.join(phoenix_folder,"HR_8799"+"*.fits"))[0]
            phoenix_HR8799_filename=phoenix_model_HR8799_filename.replace(".fits","_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))

            ##############################
            ## HR 8799 phoenix model
            ##############################
            if use_R_calib:
                with pyfits.open(phoenix_model_HR8799_filename) as hdulist:
                    phoenix_HR8799 = hdulist[0].data[crop_phoenix]
                where_IFSfilter = np.where((phoenix_wvs>wvs[0])*(phoenix_wvs<wvs[-1]))
                phoenix_HR8799 = phoenix_HR8799/np.mean(phoenix_HR8799[where_IFSfilter])
                HR8799pho_spec_func = interp1d(phoenix_wvs,phoenix_HR8799,bounds_error=False,fill_value=np.nan)
            else:
                if len(glob.glob(phoenix_HR8799_filename)) == 0:
                    with pyfits.open(phoenix_model_HR8799_filename) as hdulist:
                        phoenix_HR8799 = hdulist[0].data[crop_phoenix]
                    print("convolving: "+phoenix_model_HR8799_filename)
                    phoenix_HR8799_conv = convolve_spectrum(phoenix_wvs,phoenix_HR8799,R,specpool)

                    with open(phoenix_HR8799_filename, 'w+') as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=' ')
                        csvwriter.writerows([["wvs","spectrum"]])
                        csvwriter.writerows([[a,b] for a,b in zip(phoenix_wvs,phoenix_HR8799_conv)])

                with open(phoenix_HR8799_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=' ')
                    list_starspec = list(csv_reader)
                    HR8799pho_spec_str_arr = np.array(list_starspec, dtype=np.str)
                    col_names = HR8799pho_spec_str_arr[0]
                    HR8799pho_spec = HR8799pho_spec_str_arr[1::,1].astype(np.float)
                    HR8799pho_spec_wvs = HR8799pho_spec_str_arr[1::,0].astype(np.float)

                where_IFSfilter = np.where((HR8799pho_spec_wvs>wvs[0])*(HR8799pho_spec_wvs<wvs[-1]))
                HR8799pho_spec = HR8799pho_spec/np.mean(HR8799pho_spec[where_IFSfilter])
                HR8799pho_spec_func = interp1d(HR8799pho_spec_wvs/(1-(hr8799_rv+hr8799_bary_rv)/c_kms),HR8799pho_spec,bounds_error=False,fill_value=np.nan)
                # #remove
                # import matplotlib.pyplot as plt
                # plt.plot(HR8799pho_spec_func(wvs),color="red")
                # HR8799pho_spec_func = interp1d(HR8799pho_spec_wvs,HR8799pho_spec,bounds_error=False,fill_value=np.nan)
                # plt.plot(HR8799pho_spec_func(wvs),color="blue")
                # plt.show()

            HR8799pho_spec_func_list.append(HR8799pho_spec_func)

        hr8799_flux = np.mean(hr8799_flux_list)

        # import matplotlib.pyplot as plt
        # for R,planet_partial_template_func_list,transmission4planet,HR8799pho_spec_func,transmission_list in zip(R_list,planet_model_func_table,transmission4planet_list,HR8799pho_spec_func_list,transmission_table):
        #     planet_model = planet_partial_template_func_list[0](wvs*(1-(-12.)/c_kms))*transmission4planet(wvs)
        #     planet_model = planet_model/np.sum(planet_model)*hr8799_flux
        #     star_model = HR8799pho_spec_func(wvs)*transmission_list[0](wvs)
        #     star_model = star_model/np.sum(star_model)*hr8799_flux
        #     print(hr8799_flux)
        #     plt.plot(wvs,transmission4planet(wvs),label="transmission {0}".format(R))
        #     plt.plot(wvs,planet_partial_template_func_list[0](wvs*(1-(-12.)/c_kms)),label="plante {0}".format(R))
        #     # plt.plot(wvs,2e-5*planet_model,label="{0}".format(R))
        #     # plt.plot(wvs,transmission4planet(wvs),label="{0}".format(R))
        # plt.legend()
        # plt.show()

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
        psfs_refstar_arr = np.array(psfs_refstar_list)
        Npsfs, ny_psf,nx_psf,nz_psf = psfs_refstar_arr.shape
        x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
        x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
        x_psf_vec_hd, y_psf_vec_hd = np.linspace(0,nx_psf * 1.,100)-nx_psf//2,np.linspace(0,ny_psf* 1.,100)-ny_psf//2
        x_psf_grid_list = np.zeros((Npsfs,)+x_psf_grid.shape)
        y_psf_grid_list = np.zeros((Npsfs,)+y_psf_grid.shape)

        # import matplotlib.pyplot as plt

        for k,centers_filename in enumerate(centers_refstar_filelist):
            with pyfits.open(centers_filename) as hdulist:
                psfs_centers = hdulist[0].data

            # plt.subplot(2,2,1)
            # plt.plot(psfs_centers[:,0],label=os.path.basename(centers_filename)+suffix)
            # plt.subplot(2,2,3)
            # plt.plot(psfs_centers[:,0]-np.median(psfs_centers[:,0]),label=os.path.basename(centers_filename)+suffix)
            # plt.subplot(2,2,2)
            # plt.plot(psfs_centers[:,1],label=os.path.basename(centers_filename)+suffix)
            # plt.subplot(2,2,4)
            # plt.plot(psfs_centers[:,1]-np.median(psfs_centers[:,1]),label=os.path.basename(centers_filename)+suffix)

            avg_center = np.median(psfs_centers,axis=0)
            # print(avg_center)
            x_psf_grid_list[k,:,:] = x_psf_grid+(nx_psf//2-avg_center[0])
            y_psf_grid_list[k,:,:] = y_psf_grid+(ny_psf//2-avg_center[1])
        # plt.show()

        # import matplotlib.pyplot as plt
        # for k in range(Npsfs):
        #     plt.subplot(1,Npsfs,k+1)
        #     plt.imshow(psfs_refstar_arr[k,:,:,1500])#1660
        # plt.show()
        normalized_psfs_func_list = []
        # if debug:
        #     specpool.close()
        #     a = _spline_psf_model((psfs_refstar_arr,x_psf_grid_list,
        #                                                       y_psf_grid_list,
        #                                                       x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5))
        #     import matplotlib.pyplot as plt
        #     tmp = np.zeros(len(a))
        #     for k in range(len(a)):
        #         tmp[k] = a[k](0,0)
        #
        #     plt.plot(tmp)
        #     plt.show()
        #     exit()
        chunk_size=20
        N_chunks = nz_psf//chunk_size
        psfs_list = []
        for k in range(N_chunks-1):
            psfs_list.append(psfs_refstar_arr[:,:,:,k*chunk_size:(k+1)*chunk_size])
        psfs_list.append(psfs_refstar_arr[:,:,:,(N_chunks-1)*chunk_size:nz_psf])
        outputs_list = specpool.map(_spline_psf_model, zip(psfs_list,
                                                           itertools.repeat(x_psf_grid_list),
                                                           itertools.repeat(y_psf_grid_list),
                                                           itertools.repeat(x_psf_grid[0,0:nx_psf-1]+0.5),
                                                           itertools.repeat(y_psf_grid[0:ny_psf-1,0]+0.5)))
        for out in outputs_list:
            normalized_psfs_func_list.extend(out)

        specpool.close()


        w=2
        pl_x_vec = np.arange(-w,w+1)
        pl_y_vec = np.arange(-w,w+1)
        nospec_planet_model = np.zeros((2*w+1,2*w+1,nz))
        for z in range(nz):
            nospec_planet_model[:,:,z] = normalized_psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()

        # import matplotlib.pyplot as plt
        # for bkg_k in range(2*w+1):
        #     for bkg_l in range(2*w+1):
        #         print(bkg_k*w+bkg_l+1)
        #         plt.subplot(2*w+1,2*w+1,bkg_k*(2*w+1)+bkg_l+1)
        #         plt.plot(nospec_planet_model[bkg_k,bkg_l,:])
        # plt.show()



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
        badpix_imgs_np[:] = 0#padimgs_hdrbadpix
        badpix_imgs_np[np.where(original_imgs_np==0)] = np.nan
        originalHPF_imgs = mp.Array(dtype, np.size(padimgs))
        originalHPF_imgs_shape = padimgs.shape
        originalHPF_imgs_np = _arraytonumpy(originalHPF_imgs, originalHPF_imgs_shape,dtype=dtype)
        originalHPF_imgs_np[:] = np.nan
        originalLPF_imgs = mp.Array(dtype, np.size(padimgs))
        originalLPF_imgs_shape = padimgs.shape
        originalLPF_imgs_np = _arraytonumpy(originalLPF_imgs, originalLPF_imgs_shape,dtype=dtype)
        originalLPF_imgs_np[:] = np.nan
        nout = 10+3*4
        nshifts = np.size(planetRV_array)
        if planet_search:
            output_maps = mp.Array(dtype, len(wvsol_shifts)*len(transmission_table)*nout*padny*padnx*nshifts)
            output_maps_shape = (len(wvsol_shifts),len(transmission_table),nout,padny,padnx,nshifts)
        else:
            output_maps = mp.Array(dtype, len(wvsol_shifts)*len(transmission_table)*nout*dl_grid.shape[0]*dl_grid.shape[1]*nshifts)
            output_maps_shape = (len(wvsol_shifts),len(transmission_table),nout,dl_grid.shape[0],dl_grid.shape[1],nshifts)
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
            outres = mp.Array(dtype, len(wvsol_shifts)*len(transmission_table)*6*padny*padnx*padimgs.shape[-1])
            outres_shape = (len(wvsol_shifts),len(transmission_table),6,padimgs.shape[-1],padny,padnx)
        else:
            outres = mp.Array(dtype, len(wvsol_shifts)*len(transmission_table)*6*dl_grid.shape[0]*dl_grid.shape[1]*padimgs.shape[-1])
            outres_shape = (len(wvsol_shifts),len(transmission_table),6,padimgs.shape[-1],dl_grid.shape[0],dl_grid.shape[1])
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
                                  output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape,outres,outres_shape,outautocorrres,outautocorrres_shape,persistence_imgs,out1dfit,out1dfit_shape),
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
                                            subfilename.replace(".fits","_"+IFSfilter+"_020.fits"))
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
            # import matplotlib.pyplot as plt
            # plt.plot(persistence_imgs_np[48+5,8+5,:],label="after")
            # plt.legend()
            # plt.show()

            original_imgs_np[:] = save_original_imgs_np


        tasks = [tpool.apply_async(_HPF_z, args=(col_index,cutoff, dtype))
                 for col_index in range(padnx)]
        #save it to shared memory
        for col_index, task in enumerate(tasks):
            print("Finished col {0}".format(col_index))
            task.wait()

        badpix_imgs_np[np.where(padimgs_hdrbadpix==0)] = np.nan
        if mask_starline:
            badpix_imgs_np[:,:,795:810]=np.nan
        if mask_20101104_artifact:
            badpix_imgs_np[0:5+padding,:,:]=np.nan
            badpix_imgs_np[(padny-5-padding):padny,:,:]=np.nan
            badpix_imgs_np[43:50,:,0:400]=np.nan
            badpix_imgs_np[43:50,:,0:400]=np.nan

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
        if debug:
            # print("coucou1")
            tpool.close()
            _tpool_init(original_imgs,sigmas_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps,
                        output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape,outres,outres_shape,outautocorrres,
                        outautocorrres_shape,persistence_imgs,out1dfit,out1dfit_shape)
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
                                    wvsol_shifts,wvsol_offsets,R_calib_arr,model_persistence=model_persistence)
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
                                                                      wvsol_shifts,wvsol_offsets,R_calib_arr,
                                                                      model_persistence))
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