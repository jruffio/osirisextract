__author__ = 'jruffio'

# import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
from scipy.ndimage.filters import median_filter
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

def _tpool_init(original_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps, output_maps_shape,wvs_imgs,psfs_stamps,psfs_stamps_shape,_outres,_outres_shape,_outautocorrres,_outautocorrres_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Args:
    """
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape
    # original images from files to read and align&scale. Shape of (N,y,x)
    original = original_imgs
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
    # parameters for each image (PA, wavelegnth, image center, image number)
    lambdas = wvs_imgs
    psfs = psfs_stamps
    psfs_shape = psfs_stamps_shape
    Npixproc= 0
    Npixtot=0


def _remove_bad_pixels_z(col_index,nan_mask_boxsize,dtype,window_size=100,threshold=7.):
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    tmpcube = copy(original_np[:,col_index,:])
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    for m in np.arange(0,original_shape[0]):
        try:
            myvec = tmpcube[m,:]
            # wherefinite = np.where(np.isfinite(myvec))
            # if np.size(wherefinite[0])==0:
            #     continue
            smooth_vec = median_filter(myvec,footprint=np.ones(window_size),mode="constant",
                                       cval=np.nanmedian(myvec[np.where(np.isfinite(badpix_np[m,col_index,:]))]))
            myvec = myvec - smooth_vec
            wherefinite = np.where(np.isfinite(myvec))
            mad = mad_std(myvec[wherefinite])
            whereoutliers = np.where(np.abs(myvec)>threshold*mad)[0]
            badpix_np[m,col_index,whereoutliers] = np.nan
            widen_nans = np.where(np.isnan(np.correlate(badpix_np[m,col_index,:],np.ones(nan_mask_boxsize),mode="same")))[0]
            badpix_np[m,col_index,widen_nans] = np.nan
            original_np[m,col_index,widen_nans] = smooth_vec[widen_nans]
            # badpix_np[m,col_index,:] = smooth_vec
        except:
            print("Error",m,col_index)


def _HPF_z(col_index,cutoff,dtype):
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape
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
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    for k in wvs_indices:
        tmp = badpix_np[:,:,k]
        tmp[np.where(np.isnan(correlate2d(tmp,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
    badpix_np[0:nan_mask_boxsize//2,:,:] = np.nan
    badpix_np[-nan_mask_boxsize//2+1::,:,:] = np.nan
    badpix_np[:,0:nan_mask_boxsize//2,:] = np.nan
    badpix_np[:,-nan_mask_boxsize//2+1::,:] = np.nan


def _remove_bad_pixels_xy(wvs_indices,dtype):
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape
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
    fftmyvec = np.fft.fft(np.concatenate([myvec,myvec[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec)]
    HPF_myvec = myvec - LPF_myvec
    return LPF_myvec,HPF_myvec

def _process_pixels_onlyHPF(curr_k_indices,curr_l_indices,row_indices,col_indices,normalized_psfs_func_list,tlc_spec_list,star_spec,planet_spec_func,wvs,wvshifts_array,dtype,cutoff,planet_search,centroid_guess,iU,U):
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot,outres,outres_shape,outautocorrres,outautocorrres_shape
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    originalLPF_np = _arraytonumpy(originalLPF, original_shape,dtype=dtype)
    originalHPF_np = _arraytonumpy(originalHPF, original_shape,dtype=dtype)
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    output_maps_np = _arraytonumpy(output_maps, output_maps_shape,dtype=dtype)
    outres_np = _arraytonumpy(outres, outres_shape,dtype=dtype)
    outautocorrres_np = _arraytonumpy(outautocorrres, outautocorrres_shape,dtype=dtype)
    nshifts = output_maps_shape[-1]
    psfs_tlc = _arraytonumpy(psfs, psfs_shape,dtype=dtype)
    padny,padnx,padnz = original_shape

    tlc_lines_list = []
    for tlc_spectrum in tlc_spec_list:
        mytlclpfspec, mytlchpfspec = LPFvsHPF(tlc_spectrum,cutoff)
        tlc_lines_list.append((mytlchpfspec/mytlclpfspec))

    if 1: # temporary until I am smarter
        tlc_lines_list = [vec[None,:] for vec in tlc_lines_list]
        tlc_lines_list=[np.nanmean(np.concatenate(tlc_lines_list,axis=0),axis=0)]
        tlc_spec_list_tmp = [vec[None,:] for vec in tlc_spec_list]
        meantlc_spec=[np.nanmean(np.concatenate(tlc_spec_list_tmp,axis=0),axis=0)]
        # HPFtlc_spec, LPFtlc_spec = LPFvsHPF(meantlc_spec,cutoff)

    for curr_k,curr_l,row,col in zip(curr_k_indices,curr_l_indices,row_indices,col_indices):

        if planet_search:
            k,l = int(np.round(curr_k)),int(np.round(curr_l))
            # print(curr_k,curr_l,row,col)
            w = 2
        else:
            k,l = int(np.round(centroid_guess[0])),int(np.round(centroid_guess[1]))
            w = 4


        HPFdata = copy(originalHPF_np[k-w:k+w+1,l-w:l+w+1,:])
        chi2ref = np.nansum(HPFdata**2)
        LPFdata = copy(originalLPF_np[k-w:k+w+1,l-w:l+w+1,:])
        data = LPFdata+HPFdata
        HPFdata_badpix = badpix_np[k-w:k+w+1,l-w:l+w+1,:]
        data_ny,data_nx,data_nz = HPFdata.shape

        x_vec, y_vec = np.arange(padnx * 1.)-curr_l,np.arange(padny* 1.)-curr_k
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        x_data_grid, y_data_grid = x_grid[k-w:k+w+1,l-w:l+w+1], y_grid[k-w:k+w+1,l-w:l+w+1]

        planet_model = np.zeros(HPFdata.shape)
        pl_x_vec = x_data_grid[0,:]
        pl_y_vec = y_data_grid[:,0]
        for z in range(data_nz):
            planet_model[:,:,z] = normalized_psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()

        HPFpolydeg = 0
        HPFmodel_list = []
        # plt.figure(1)
        for tlc_lines in tlc_lines_list:
            # plt.plot(HPFtlc_spectrum)
            for kk in range(HPFpolydeg+1):
                bkg_model = np.zeros((2*w+1,2*w+1,2*w+1,2*w+1,data_nz))
                for bkg_k in range(2*w+1):
                    for bkg_l in range(2*w+1):
                        myspec = LPFdata[bkg_k,bkg_l,:]*(tlc_lines*(wvs**kk))/np.nansum(data[bkg_k,bkg_l,:])*np.nansum(meantlc_spec)
                        bkg_model[bkg_k,bkg_l,bkg_k,bkg_l,:] = myspec#/np.nanstd(myspec)
                HPFmodel_list.append(np.reshape(bkg_model,((2*w+1)**2,(2*w+1)**2*data_nz)).transpose())




        HPFmodel_H0 = np.concatenate(HPFmodel_list,axis=1)
        # plt.show()

        # for bkg_k in range(2*w+1):
        #     for bkg_l in range(2*w+1):
        #         plt.subplot(2*w+1,2*w+1,bkg_k*(2*w+1)+bkg_l+1)
        #         plt.plot(HPFdata[bkg_k,bkg_l,:]/np.nanstd(HPFdata[bkg_k,bkg_l,:]),label="data",color="red")
        #         for HPFtlc_spectrum in HPFtlc_spec_list:
        #             myspec = HPFtlc_spectrum*LPFdata[bkg_k,bkg_l,:]
        #             plt.plot(myspec/np.nanstd(myspec),linestyle="--",label="tlc",alpha=0.5)
        #
        # plt.show()


        ravelHPFdata = np.ravel(copy(HPFdata))
        ravelLPFdata = np.ravel(copy(LPFdata))
        # HPFmodel_H0 = np.reshape(HPFmodel_H0,((2*w+1)**2*data_nz,HPFmodel_H0.shape[-1]))
        where_bad_data = np.where(~(np.isfinite(np.ravel(HPFdata_badpix))))
        # ravelLPFdata[where_bad_data] = 1
        ravelHPFdata[where_bad_data] = 0
        # where_finite_data = np.where(np.isfinite(np.ravel(HPFdata_badpix))*(np.ravel(LPFdata)>0))
        # ravelLPFdata = ravelLPFdata[where_finite_data]
        sigmas = np.sqrt(np.abs(ravelLPFdata))
        # ravelHPFdata = ravelHPFdata[where_finite_data]
        # ravelHPFdata = ravelHPFdata/sigmas
        # HPFmodel = HPFmodel[where_finite_data[0],:]
        # where_valid_parameters = np.where(np.sum(np.abs(HPFmodel),axis=0)!=0)
        # HPFmodel = HPFmodel[:,where_valid_parameters[0]]
        # HPFmodel = HPFmodel/sigmas[:,None]
        # logdet_Sigma = np.sum(np.log(sigmas**2))

        logdet_Sigma=0
        for k in range((2*w+1)**2):
            # print(ravelHPFdata[k*np.size(wvs):(k+1)*np.size(wvs)].shape,iU.T.shape,ravelHPFdata[k*np.size(wvs):(k+1)*np.size(wvs)].shape,sigmas[k*np.size(wvs):(k+1)*np.size(wvs)].shape)
            ravelHPFdata[k*np.size(wvs):(k+1)*np.size(wvs)] = np.dot(iU.T,ravelHPFdata[k*np.size(wvs):(k+1)*np.size(wvs)]/sigmas[k*np.size(wvs):(k+1)*np.size(wvs)])
            HPFmodel_H0[k*np.size(wvs):(k+1)*np.size(wvs),:] = np.dot(iU.T,HPFmodel_H0[k*np.size(wvs):(k+1)*np.size(wvs),:]/sigmas[k*np.size(wvs):(k+1)*np.size(wvs),None])
            logdet_Sigma = logdet_Sigma+np.sum(np.log(np.diagonal(U)*sigmas[k*np.size(wvs):(k+1)*np.size(wvs)]**2))

        # dwv = wvs[1]-wvs[0]
        noshift_id = np.argmin(np.abs(wvshifts_array))
        for wvshift_id in range(nshifts):
            # print(wvshift_id)
            try:
            # if 1:
                # planet_spec = planet_spec_func(wvs+(wvshift_id-nshifts//2)*dwv)
                planet_spec = planet_spec_func(wvs-wvshifts_array[wvshift_id])
                planet_spec = planet_spec/np.mean(planet_spec)
                norma_planet_spec = planet_spec/star_spec

                LPFtlcplanet_spec,HPFtlcplanet_spec = LPFvsHPF(tlc_spec_list[0]*norma_planet_spec,cutoff)
                # plt.figure(2)
                # plt.plot(HPFtlcplanet_spec)
                HPFplanet_model = planet_model*(HPFtlcplanet_spec/np.nansum(LPFtlcplanet_spec+HPFtlcplanet_spec)*np.nansum(meantlc_spec))[None,None,:]
                HPFplanet_model = np.ravel(HPFplanet_model)

                for k in range((2*w+1)**2):
                    HPFplanet_model[k*np.size(wvs):(k+1)*np.size(wvs)] = np.dot(iU.T,HPFplanet_model[k*np.size(wvs):(k+1)*np.size(wvs)]/sigmas[k*np.size(wvs):(k+1)*np.size(wvs)])

                # print(HPFplanet_model.shape,HPFmodel_H0.shape)
                HPFmodel = np.concatenate([HPFplanet_model[:,None],HPFmodel_H0],axis=1)
                # HPFmodel_H1 = HPFmodel[:,0:1]

                # tmp_HPFplanet_model = np.ravel(HPFplanet_model)
                # tmp_HPFplanet_model = tmp_HPFplanet_model[where_finite_data]

                # plt.plot(ravelHPFdata)
                # # print(LPFdata[np.where(LPFdata<=0)])
                # plt.show()
                # print(HPFmodel.shape)
                # for k in range(HPFmodel.shape[-1]):
                #     plt.plot(HPFmodel[:,k],label="{0}".format(k))
                # plt.legend()
                # plt.show()

                # t1 = time.time()
                HPFparas,HPFchi2,rank,s = np.linalg.lstsq(HPFmodel,ravelHPFdata,rcond=None)
                # HPFparas_H1,HPFchi2_H1,rank,s = np.linalg.lstsq(HPFmodel_H1,ravelHPFdata,rcond=None)
                HPFparas_H0,HPFchi2_H0,rank,s = np.linalg.lstsq(HPFmodel_H0,ravelHPFdata,rcond=None)

                data_model = np.dot(HPFmodel,HPFparas)
                # data_model_H1 = np.dot(HPFmodel_H1,HPFparas_H1)
                data_model_H0 = np.dot(HPFmodel_H0,HPFparas_H0)
                # deltachi2 = chi2ref-np.sum(ravelHPFdata**2)
                ravelresiduals = data_model-ravelHPFdata
                HPFchi2 = np.nansum((ravelresiduals)**2)
                # HPFchi2_H1 = np.nansum((data_model_H1-ravelHPFdata)**2)
                HPFchi2_H0 = np.nansum((data_model_H0-ravelHPFdata)**2)

                if wvshift_id == noshift_id:
                    # TODO TODO TODO TODOTODO TODOTODO TODOTODO TODOTODO TODOTODO TODOTODO TODOTODO TODOTODO TODO
                    # canvas_model = np.zeros((2*w+1,2*w+1,data_nz))
                    # canvas_model.shape = ((2*w+1)*(2*w+1)*data_nz,)
                    # canvas_model[where_finite_data] = (data_model-HPFparas[0]*HPFmodel[:,0])*sigmas
                    # canvas_model.shape = (2*w+1,2*w+1,data_nz)
                    # canvas_data = copy(HPFdata)
                    # canvas_data.shape = ((2*w+1)*(2*w+1)*data_nz,)
                    # canvas_data[where_bad_data] = 0
                    # canvas_data.shape = (2*w+1,2*w+1,nl)
                    #
                    # final_template = np.nansum(canvas_model,axis=(0,1))
                    # final_data = np.nansum(canvas_data,axis=(0,1))
                    # final_res = final_data-final_template
                    # outres_np[0,:,row,col] = final_data
                    # outres_np[1,:,row,col] = final_template
                    # outres_np[2,:,row,col] = final_res

                    outres_np[0,:,row,col] = 0
                    outres_np[1,:,row,col] = 0
                    outres_np[2,:,row,col] = 0

                    # res_ccf = np.correlate(ravelresiduals,ravelresiduals,mode="same")
                    # res_ccf_argmax = np.argmax(res_ccf)
                    # outautocorrres_np[:,row,col] = res_ccf[(res_ccf_argmax-500):(res_ccf_argmax+500)]

                    outautocorrres_np[:,row,col] = 0

                # plt.figure(1)
                # plt.plot(HPFdata,label="HPFdata")
                # plt.plot(data_model,label="data_model")
                # tmp = np.zeros(HPFparas.shape)
                # tmp[0] = HPFparas[0]
                # plt.plot(np.dot(HPFmodel,tmp),label="planet")
                # plt.plot(data_model_H1,label="data_model_H1")
                # plt.plot(data_model_H0,label="data_model_H0")
                # plt.legend()
                # plt.figure(2)
                # plt.plot(HPFdata,label="HPFdata")
                # plt.plot(HPFdata-data_model,label="data_model")
                # # tmp = np.zeros(HPFparas.shape)
                # # tmp[0] = HPFparas[0]
                # # plt.plot(np.dot(HPFmodel,tmp),label="planet")
                # # plt.plot(HPFdata-data_model_H1,label="data_model_H1")
                # # plt.plot(HPFdata-data_model_H0,label="data_model_H0")
                # plt.legend()
                # plt.show()

                # print(LPFchi2,LPFchi2_H0,HPFchi2,HPFchi2_H0)
                Npixs_HPFdata = HPFmodel.shape[0]
                # norma_sig = np.sqrt((HPFchi2/np.size(HPFchi2)))
                # tmp = np.linalg.lstsq(np.dot(HPFmodel.T,HPFmodel),np.dot(HPFmodel.T,sigmas))[0]#,HPFmodel.T)
                # print(np.sqrt(tmp)*norma_sig)
                # print(tmp.shape,HPFmodel.shape)
                # exit()
                minus2logL_HPF = Npixs_HPFdata*(1+np.log(HPFchi2/Npixs_HPFdata)+logdet_Sigma+np.log(2*np.pi))
                # minus2logL_HPF_H1 = Npixs_HPFdata*np.log(HPFchi2_H1/Npixs_HPFdata)+1./Npixs_HPFdata
                minus2logL_HPF_H0 = Npixs_HPFdata*(1+np.log(HPFchi2_H0/Npixs_HPFdata)+logdet_Sigma+np.log(2*np.pi))
                AIC_HPF = 2*(HPFmodel.shape[-1])+minus2logL_HPF
                # AIC_HPF_H1 = 2*(HPFmodel_H1.shape[-1])+minus2logL_HPF_H1
                AIC_HPF_H0 = 2*(HPFmodel_H0.shape[-1])+minus2logL_HPF_H0

                covphi =  HPFchi2/Npixs_HPFdata*np.linalg.inv(np.dot(HPFmodel.T,HPFmodel))
                slogdet_icovphi0 = np.linalg.slogdet(np.dot(HPFmodel.T,HPFmodel))

                output_maps_np[0,row,col,wvshift_id] = HPFparas[0]
                output_maps_np[1,row,col,wvshift_id] = np.sign(covphi[0,0])*np.sqrt(np.abs(covphi[0,0]))
                output_maps_np[2,row,col,wvshift_id] = AIC_HPF_H0-AIC_HPF
                output_maps_np[3,row,col,wvshift_id] = AIC_HPF
                output_maps_np[4,row,col,wvshift_id] = AIC_HPF_H0
                output_maps_np[5,row,col,wvshift_id] = HPFchi2
                # output_maps_np[6,row,col,wvshift_id] = deltachi2
                # output_maps_np[7,row,col,wvshift_id] = HPFchi2+deltachi2
                output_maps_np[6,row,col,wvshift_id] = 0
                output_maps_np[7,row,col,wvshift_id] = HPFchi2
                output_maps_np[8,row,col,wvshift_id] = HPFchi2/Npixs_HPFdata
                output_maps_np[9,row,col,wvshift_id] = logdet_Sigma
                output_maps_np[10,row,col,wvshift_id] = slogdet_icovphi0[1]
                output_maps_np[11,row,col,wvshift_id] = -0.5*logdet_Sigma-0.5*slogdet_icovphi0[1]- (Npixs_HPFdata-HPFmodel.shape[-1]+2-1)/(2)*np.log(HPFchi2)
                # print(output_maps_np[:,row,col,wvshift_id])
            except:
                pass
    return

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


    if 0:# HR 8799 c 20100715
        # planet = "b"
        planet = "c"
        # date = "100715"
        # date = "101104"
        date = "110723"
        # date = "*"
        # planet = "d"
        # date = "150720"
        # date = "150722"
        # date = "150723"
        # date = "150828"
        IFSfilter = "Kbb"
        # IFSfilter = "Hbb" # "Kbb" or "Hbb"
        inputDir = "/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb/"
        # outputdir = "/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb/20181120_out/"
        outputdir = "/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb/20181205_HPF_only_cov/"
        filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_020.fits"))
        filelist.sort()
        filelist = filelist[0:1]
        # psfs_tlc_filename = "/home/sda/jruffio/osiris_data/HR_8799_c/20"+date+"/reduced_telluric_JB/HD_210501/s"+date+"_a005002_Kbb_020_psfs.fits"
        psfs_tlc_filelist = glob.glob("/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_telluric_jb/*/s*"+IFSfilter+"_020_psfs.fits")
        # psfs_tlc_filelist = [psfs_tlc_filelist[0]]
        # print(psfs_tlc_filelist)
        # template_spec_filename="/home/sda/jruffio/osiris_data/HR_8799_c/hr8799c_osiris_template.save"
        template_spec_filename="/home/sda/jruffio/osiris_data/HR_8799_c/HR8799c_"+IFSfilter[0:1]+"_3Oct2018_conv"+IFSfilter+".csv"
        # template_spec_filename="/home/sda/jruffio/osiris_data/HR_8799_c/HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save"
        # star_spec_filename = "/home/sda/jruffio/osiris_data/HR_8799_c/hr_8799_c_pickles_spectrum_"+IFSfilter+".csv"
        # template_spec_filename="/home/sda/jruffio/osiris_data/HR_8799_b/HR8799b_"+IFSfilter[0:1]+"_3Oct2018.save"
        # star_spec_filename = "/home/sda/jruffio/osiris_data/HR_8799_b/hr_8799_b_pickles_spectrum_"+IFSfilter+".csv"
        # planet_coords = [[11,32],[12,27],[12,33],[12,39],[10,33],[9,28],[8,38],[10,32.5],[9,32],[10,33],[10,35],[10,33], #in order: image 10 to 21
        #                  [7,34],[5,35],[8,35],[7.5,33],[9.5,34.5]]
        # file_centers = [[x-sep_planet/ 0.0203,y] for x,y in planet_coords]
        numthreads = 25
        phoenix_folder = os.path.join("/home/sda/jruffio/osiris_data/phoenix")#"/home/sda/jruffio/osiris_data/phoenix/"
        planet_search = True

        lcorr = 0.5
    else:
        inputDir = sys.argv[1]
        outputdir = sys.argv[2]
        filename = sys.argv[3]
        numthreads = int(sys.argv[4])
        planet_search = bool(int(sys.argv[5]))
        lcorr=float(sys.argv[6])
        # star_spec_filename  = sys.argv[6]
        # centermode = sys.argv[8]

        filelist = [filename]
        IFSfilter = filename.split("_")[-2]
        psfs_tlc_filelist = glob.glob(os.path.join(os.path.dirname(filename),"..","reduced_telluric_jb/*/s*"+IFSfilter+"_020_psfs.fits"))
        # fileinfos_filename = "/home/users/jruffio/OSIRIS/osirisextract/fileinfos_jb.xml"
        template_spec_filename=os.path.join(os.path.dirname(filename),"..","..","HR8799c_"+IFSfilter[0:1]+"_3Oct2018_conv"+IFSfilter+".csv")

        # print(psfs_tlc_filelist)
        # print(template_spec_filename)
        # print(os.path.join(os.path.dirname(filename),"..","/educed_telluric_jb/*/s*"+IFSfilter+"_020_psfs.fits"))
        # print(os.path.join(os.path.dirname(filename),"..","reduced_telluric_jb/*/"))
        # exit()

        phoenix_folder = os.path.join(os.path.dirname(filename),"..","..","..","phoenix")#"/home/sda/jruffio/osiris_data/phoenix/"
        #nice -n 15 /home/anaconda3/bin/python ./reduce_HPFonly_cov.py /home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/ /home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/20181205_HPF_only_sherlock_test/ /home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/s100715_a010001_Kbb_020.fits 20 1 0
        #nice -n 15 /home/anaconda3/bin/python ./reduce_HPFonly_cov.py /home/sda/jruffio/osiris_data/HR_8799_c/20110723/reduced_jb/ /home/sda/jruffio/osiris_data/HR_8799_c/20110723/reduced_jb/20181205_HPF_only_sherlock_test/ /home/sda/jruffio/osiris_data/HR_8799_c/20110723/reduced_jb/s110723_a033001_Kbb_020.fits 20 1 0

    if IFSfilter=="Kbb": #Kbb 1965.0 0.25
        CRVAL1 = 1965.
        CDELT1 = 0.25
        nl=1665
        R=4000
    elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1473.
        CDELT1 = 0.2
        nl=1651
        R=5000
    dwv = CDELT1/1000.

    padding = 5


    dtype = ctypes.c_float
    nan_mask_boxsize=3
    cutoff = 80#80
    debug = False

    # prihdr_list = []
    # for filename in filelist[0:10]:
    #     with pyfits.open(filename) as hdulist:
    #         prihdr_list.append(hdulist[0].header)
    # from pyklip.instruments.osiris import determine_mosaic_offsets_from_header
    # delta_x,delta_y = determine_mosaic_offsets_from_header(prihdr_list)
    # print(delta_x)
    # print(delta_y)
    # prihdr_list = []
    # for filename in filelist[10::]:
    #     with pyfits.open(filename) as hdulist:
    #         prihdr_list.append(hdulist[0].header)
    # from pyklip.instruments.osiris import determine_mosaic_offsets_from_header
    # delta_x,delta_y = determine_mosaic_offsets_from_header(prihdr_list)
    # print(delta_x)
    # print(delta_y)
    # exit()

    for filename in filelist:#[13::]:
    # if 1:
    #     filename = filelist[2]
    #     print(filename)
    #     filename2 = filelist[3]
    #     suffix = "HPF_cutoff{0}_new_sig_phoenix_wvshift_centroid".format(cutoff)
    #     suffix = "HPF_cutoff{0}_new_sig_phoenix_wvshift_normalcruncher".format(cutoff)
    #     suffix = "HPF_cutoff{0}_cov{1}".format(cutoff,lcorr)


        if planet_search:
            suffix = "HPF_cutoff{0}_sherlock_v1_search_empcov".format(cutoff)
            wvshifts_array = np.concatenate([np.arange(-2*dwv,2*dwv,dwv/100),np.arange(-100*dwv,100*dwv,dwv)])
            # wvshifts_array = np.arange(-1*dwv,1*dwv,dwv)
            # wvshifts_array = np.linspace(-1.1*dwv,-.7*dwv,40)#np.arange(-1.1*dwv,-.8*dwv,dwv/100)
            real_k,real_l = np.nan,np.nan
        else:
            suffix = "HPF_cutoff{0}_sherlock_v1_centroid_empcov".format(cutoff)
            fileinfos_filename = os.path.join(inputDir,"..","..","fileinfos_"+IFSfilter+"_jb.csv")

            #read file
            with open(fileinfos_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=';')
                list_table = list(csv_reader)
                colnames = list_table[0]
                N_col = len(colnames)
                list_data = list_table[1::]
                N_lines =  len(list_data)

            filename_id = colnames.index("filename")
            cen_filename_id = colnames.index("cen filename")
            kcen_id = colnames.index("kcen")
            lcen_id = colnames.index("lcen")
            rvcen_id = colnames.index("RVcen")

            filelist = [os.path.basename(item[filename_id]) for item in list_data]

            fileid = filelist.index(os.path.basename(filename))
            fileitem = list_data[fileid]
            print(os.path.basename(filename), fileitem)

            real_k,real_l = float(fileitem[kcen_id]),float(fileitem[lcen_id])
            # real_k,real_l = 32-10,-35.79802955665025+46.8
            # real_k,real_l = 50,15
            # real_k,real_l = 32+padding-10,-35.79802955665025+46.8+padding
            # real_k,real_l = 51,17
            # real_k,real_l = 39,16
            # real_k,real_l = 39+5,12

            wvshifts_array = np.arange(-3*dwv,3*dwv,dwv/100)
            # dl_grid,dk_grid = np.array([[0]]),np.array([[0]])
            dl_grid,dk_grid = np.meshgrid(np.linspace(-1.5,1.5,4*20+1),np.linspace(-1.5,1.5,4*20+1))
            # wvshifts_array = np.arange(-1*dwv,1*dwv,dwv)

            real_k,real_l = real_k+padding,real_l+padding

        if not os.path.exists(os.path.join(outputdir)):
            os.makedirs(os.path.join(outputdir))

        with pyfits.open(filename) as hdulist:
            imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
            prihdr = hdulist[0].header
            imgs = np.moveaxis(imgs,0,2)
            print(filename)
            # print(np.nansum(imgs))
            # continue
            # # plt.imshow(np.nansum(imgs,axis=2))
            # plt.imshow(imgs[:,:,0])
            # plt.colorbar()
            # plt.show()
            # exit()
        # # imgs[np.where(imgs==0)] = np.nan
        # with pyfits.open(filename2) as hdulist:
        #     imgs2 = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        #     prihdr2 = hdulist[0].header
        #     imgs2 = np.moveaxis(imgs2,0,2)
        #     imgs = imgs-imgs2
        ny,nx,nz = imgs.shape
        # print(IFSfilter,nz,prihdr["CRVAL1"],prihdr["CDELT1"])
        # exit()
        init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
        dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
        # print(3e5*dwv/(init_wv+dwv*nz//2)) # 38.167938931297705
        # exit()
        wvs=np.arange(init_wv,init_wv+dwv*nz-1e-6,dwv)
        # print(imgs.shape)
        # print(wvs.shape)
        # print(wvs)
        # print(init_wv,init_wv+dwv*nz,dwv)
        # exit()
        # print(dwv)
        # plt.plot(wvs - wvs/(1+0.0000834254+0.02406147/(130-(1./wvs)**2)))
        # plt.show()
        # vac2air_wv = True

        # travis_spectrum = scio.readsav(template_spec_filename)
        # planet_spec = np.array(travis_spectrum["fk_bin"])
        # planet_spec = planet_spec/np.mean(planet_spec)


        # travis_spectrum = scio.readsav(template_spec_filename)
        # ori_planet_spec = np.array(travis_spectrum["fmods"])
        # wmod = np.array(travis_spectrum["wmod"])/1.e4
        # ori_planet_spec = ori_planet_spec/np.mean(ori_planet_spec)
        # z = interp1d(wmod,ori_planet_spec)
        # planet_spec = z(wvs)
        # planet_spec = planet_spec/np.mean(planet_spec)

        if 0:
            x = np.arange(0,np.size(wvs),1)
            dx2 = (x[:,None]-x[None,:])**2
            cov = np.exp(-dx2/(2*lcorr**2))
        else:
            x = np.arange(0,np.size(wvs),1)
            dx = (x[:,None]-x[None,:])
            # autocorrresfilename = os.path.join(os.path.dirname(filename),"..","..","20100715/reduced_jb/20181205_HPF_only/s100715_a010001_Kbb_020_outputHPF_cutoff80_sherlock_v0_test_autocorrres.fits")
            # print(autocorrresfilename)
            autocorrresfilename = glob.glob(os.path.join(os.path.dirname(filename),"sherlock/20190125_HPFonly",
                               os.path.basename(filename).replace(".fits","_outputHPF_cutoff{0}".format(cutoff)+"*_autocorrres.fits")))[0]
            # print(autocorrresfilename)
            # print(glob.glob(autocorrresfilename))
            # exit()
            hdulist = pyfits.open(autocorrresfilename)
            autocorr_cube = hdulist[0].data
            nz,ny,nx = autocorr_cube.shape
            avg_autocorr = np.nanmean(autocorr_cube,axis=(1,2))
            avg_autocorr = avg_autocorr/np.max(avg_autocorr)
            max_id = np.argmax(avg_autocorr)
            autocorr_func = interp1d(np.arange(40),avg_autocorr[max_id:(max_id+40)],bounds_error=False,fill_value=0)
            cov = autocorr_func(np.abs(dx))
            import matplotlib.pyplot as plt
            mus,vs= np.linalg.eig(cov)
            print(mus.shape,vs.shape)
            where_negmus = np.where(mus<=0.01)
            mus[where_negmus] = 0.01
            mus_argsort = np.argsort(mus)
            mus = mus[mus_argsort[::-1]]
            vs = vs[:,mus_argsort[::-1]]
            # mus[1600::] = 0.01
            recons_cov = np.dot(vs,np.dot(np.diag(mus),vs.T))
            # plt.figure(1)
            # plt.plot(mus)
            # plt.figure(2)
            # plt.subplot(1,3,1)
            # plt.imshow(cov,interpolation="nearest")
            # plt.colorbar()
            # plt.subplot(1,3,2)
            # plt.imshow(recons_cov,interpolation="nearest")
            # plt.colorbar()
            # plt.subplot(1,3,3)
            # plt.imshow((cov-recons_cov),interpolation="nearest")
            # plt.colorbar()
            # plt.show()

            cov = recons_cov

        import scipy.linalg as sclin
        U = sclin.cholesky(cov)
        iU = np.linalg.inv(U)
        # exit()

        with open(template_spec_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            list_starspec = list(csv_reader)
            oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
            col_names = oriplanet_spec_str_arr[0]
            oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
            oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec)
            oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
            # if vac2air_wv:
            #     oriplanet_spec_wvs = oriplanet_spec_wvs/(1+0.0000834254+0.02406147/(130-(1./oriplanet_spec_wvs)**2))
            planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec)
        # plt.plot(oriplanet_spec_wvs,oriplanet_spec)
        # plt.plot(wvs,planet_spec)
        # plt.show()
        # exit()

        # with open(star_spec_filename, 'r') as csvfile:
        #     csv_reader = csv.reader(csvfile, delimiter=' ')
        #     list_starspec = list(csv_reader)
        #     starspec_str_arr = np.array(list_starspec, dtype=np.str)
        #     col_names = starspec_str_arr[0]
        #     star_spec = starspec_str_arr[1::,1].astype(np.float)
        #     star_spec = star_spec/np.mean(star_spec)
        #     star_spec_wvs = starspec_str_arr[1::,0].astype(np.float)
        #     z = interp1d(star_spec_wvs,star_spec)
        #     star_spec = z(wvs)
        #     star_spec = star_spec/np.mean(star_spec)

        psfs_tlc = []
        tlc_spec_list = []
        for psfs_tlc_filename in psfs_tlc_filelist:
            print(psfs_tlc_filename)
            ref_star_name = psfs_tlc_filename.split(os.path.sep)[-2]
            if IFSfilter == "Hbb":
                hr8799_mag = 5.240
            elif IFSfilter == "Kbb":
                hr8799_mag = 5.240
            else:
                raise("IFS filter name unknown")
            hr8799_type = "F0"

            phoenix_HR8799_filename = glob.glob(os.path.join(phoenix_folder,"HR_8799"+"*conv"+IFSfilter+".csv"))[0]
            with open(phoenix_HR8799_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                list_starspec = list(csv_reader)
                starspec_str_arr = np.array(list_starspec, dtype=np.str)
                col_names = starspec_str_arr[0]
                star_spec = starspec_str_arr[1::,1].astype(np.float)
                star_spec = star_spec/np.mean(star_spec)
                star_spec_wvs = starspec_str_arr[1::,0].astype(np.float)
                # if vac2air_wv:
                #     star_spec_wvs = oriplanet_spec_wvs/(1+0.0000834254+0.02406147/(130-(1./oriplanet_spec_wvs)**2))
                hr8799_spec_func = interp1d(star_spec_wvs,star_spec)
                star_spec = hr8799_spec_func(wvs)
                phoenix_HR8799 = star_spec/np.mean(star_spec)
            phoenix_tlc_filename = glob.glob(os.path.join(phoenix_folder,ref_star_name+"*conv"+IFSfilter+".csv"))[0]
            with open(phoenix_tlc_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                list_starspec = list(csv_reader)
                starspec_str_arr = np.array(list_starspec, dtype=np.str)
                col_names = starspec_str_arr[0]
                star_spec = starspec_str_arr[1::,1].astype(np.float)
                star_spec = star_spec/np.mean(star_spec)
                star_spec_wvs = starspec_str_arr[1::,0].astype(np.float)
                star_spec_func = interp1d(star_spec_wvs,star_spec)
                star_spec = star_spec_func(wvs)
                phoenix_tlc = star_spec/np.mean(star_spec)
            # print(phoenix_HR8799.shape)
            # plt.plot(wvs,phoenix_HR8799,label="8799")
            # plt.plot(wvs,phoenix_tlc,label="tlc")
            # plt.xlabel("mum")
            # plt.legend()
            # plt.show()
            # exit()
            if ref_star_name == "HD_210501":
                ref_star_type = "A0"
                if IFSfilter == "Hbb":
                    ref_star_mag = 7.606
                elif IFSfilter == "Kbb":
                    ref_star_mag = 7.597
            elif ref_star_name == "HIP_1123":
                ref_star_type = "A1"
                if IFSfilter == "Hbb":
                    ref_star_mag = 6.219
                elif IFSfilter == "Kbb":
                    ref_star_mag = 6.189
            elif ref_star_name == "HIP_116886":
                ref_star_type = "A5"
                if IFSfilter == "Hbb":
                    ref_star_mag = 9.212
                elif IFSfilter == "Kbb":
                    ref_star_mag = 9.189
            else:
                raise(Exception("Ref star name unknown"))
            with pyfits.open(psfs_tlc_filename.replace("_psfs","")) as hdulist:
                psfs_tlc_prihdr = hdulist[0].header
            with pyfits.open(psfs_tlc_filename) as hdulist:
                mypsfs = hdulist[0].data
                psfs_tlc_itime = psfs_tlc_prihdr["ITIME"]
                mypsfs = np.moveaxis(mypsfs,0,2)
                # plt.imshow(mypsfs[:,:,0],interpolation="nearest")
                # plt.show()
                mypsfs[np.where(np.isnan(mypsfs))] = 0
                myspec = np.nansum(mypsfs,axis=(0,1))
                # myspec = np.nanmax(mypsfs,axis=(0,1))
                psfs_tlc.append(mypsfs/myspec[None,None,:])
                # print(IFSfilter,ref_star_name,np.nanmean(myspec),psfs_tlc_itime)
                # plt.plot(wvs,myspec,label="myspec before")
                # plt.plot(wvs,myspec,label="myspec after")
                # plt.xlabel("mum")
                # plt.legend()
                # plt.show()
                myspec = myspec * 10**(-1./2.5*(hr8799_mag-ref_star_mag))
                myspec_flux = np.sum(myspec)
                myspec = myspec * (phoenix_HR8799/phoenix_tlc)
                myspec = myspec / np.sum(myspec) *myspec_flux
                # print(IFSfilter,ref_star_name,np.nanmean(myspec),psfs_tlc_itime)
                tlc_spec_list.append(myspec)
        # exit()

        psfs_tlc = [cube[None,:,:,:] for cube in psfs_tlc]
        psfs_tlc = np.concatenate(psfs_tlc,axis=0)


        padimgs = np.pad(imgs,((padding,padding),(padding,padding),(0,0)),mode="constant",constant_values=0)
        padny,padnx,padnz = padimgs.shape

        original_imgs = mp.Array(dtype, np.size(padimgs))
        original_imgs_shape = padimgs.shape
        original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
        original_imgs_np[:] = padimgs
        badpix_imgs = mp.Array(dtype, np.size(padimgs))
        badpix_imgs_shape = padimgs.shape
        badpix_imgs_np = _arraytonumpy(badpix_imgs, badpix_imgs_shape,dtype=dtype)
        badpix_imgs_np[:] = 0
        badpix_imgs_np[np.where(original_imgs_np==0)] = np.nan
        originalHPF_imgs = mp.Array(dtype, np.size(padimgs))
        originalHPF_imgs_shape = padimgs.shape
        originalHPF_imgs_np = _arraytonumpy(originalHPF_imgs, originalHPF_imgs_shape,dtype=dtype)
        originalHPF_imgs_np[:] = np.nan
        originalLPF_imgs = mp.Array(dtype, np.size(padimgs))
        originalLPF_imgs_shape = padimgs.shape
        originalLPF_imgs_np = _arraytonumpy(originalLPF_imgs, originalLPF_imgs_shape,dtype=dtype)
        originalLPF_imgs_np[:] = np.nan
        nout = 12
        nshifts = np.size(wvshifts_array)
        if planet_search:
            output_maps = mp.Array(dtype, nout*padny*padnx*nshifts)
            output_maps_shape = (nout,padny,padnx,nshifts)
        else:
            output_maps = mp.Array(dtype, nout*dl_grid.shape[0]*dl_grid.shape[1]*nshifts)
            output_maps_shape = (nout,dl_grid.shape[0],dl_grid.shape[1],nshifts)
        output_maps_np = _arraytonumpy(output_maps,output_maps_shape,dtype=dtype)
        output_maps_np[:] = np.nan
        if planet_search:
            outres = mp.Array(dtype, 3*padny*padnx*padimgs.shape[-1])
            outres_shape = (3,padimgs.shape[-1],padny,padnx)
        else:
            outres = mp.Array(dtype, 3*dl_grid.shape[0]*dl_grid.shape[1]*padimgs.shape[-1])
            outres_shape = (3,padimgs.shape[-1],dl_grid.shape[0],dl_grid.shape[1])
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
        psfs_stamps = mp.Array(dtype, np.size(psfs_tlc))
        psfs_stamps_shape = psfs_tlc.shape
        psfs_stamps_np = _arraytonumpy(psfs_stamps, psfs_stamps_shape,dtype=dtype)
        psfs_stamps_np[:] = psfs_tlc


        ######################
        # INIT threads and shared memory
        tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                        initargs=(original_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps,
                                  output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape,outres,outres_shape,outautocorrres,outautocorrres_shape),
                        maxtasksperchild=50)


        # plt.plot(np.ravel(original_imgs_np[30,:,:]),label="original")

        ######################
        # CLEAN IMAGE


        chunk_size = padnz//(3*numthreads)
        N_chunks = padnz//chunk_size
        wvs_indices_list = []
        for k in range(N_chunks-1):
            wvs_indices_list.append(np.arange((k*chunk_size),((k+1)*chunk_size)))
        wvs_indices_list.append(np.arange(((N_chunks-1)*chunk_size),padnz))


        tasks = [tpool.apply_async(_remove_bad_pixels_xy, args=(wvs_indices,dtype))
                 for wvs_indices in wvs_indices_list]
        #save it to shared memory
        for chunk_index, rmedge_task in enumerate(tasks):
            print("Finished rm bad pixel xy chunk {0}".format(chunk_index))
            rmedge_task.wait()
        ######################

        # plt.plot(np.ravel(original_imgs_np[30,:,:]),linestyle=":",label="bad pix xy")

        tasks = [tpool.apply_async(_remove_bad_pixels_z, args=(col_index,nan_mask_boxsize, dtype))
                 for col_index in range(padnx)]
        #save it to shared memory
        for col_index, bad_pix_task in enumerate(tasks):
            print("Finished rm bad pixel z col {0}".format(col_index))
            bad_pix_task.wait()

        # plt.plot(np.ravel(original_imgs_np[30,:,:]),linestyle=":",label="bad pix z")

        tasks = [tpool.apply_async(_remove_edges, args=(wvs_indices,nan_mask_boxsize,dtype))
                 for wvs_indices in wvs_indices_list]
        #save it to shared memory
        for chunk_index, rmedge_task in enumerate(tasks):
            print("Finished rm edge chunk {0}".format(chunk_index))
            rmedge_task.wait()

        # plt.plot(np.ravel(original_imgs_np[30,:,:]),linestyle=":",label="remove edge")
        # plt.plot(np.ravel(badpix_imgs_np[30,:,:]),linestyle="-",label="mask")


        tasks = [tpool.apply_async(_HPF_z, args=(col_index,cutoff, dtype))
                 for col_index in range(padnx)]
        #save it to shared memory
        for col_index, bad_pix_task in enumerate(tasks):
            print("Finished col {0}".format(col_index))
            bad_pix_task.wait()

        # plt.plot(np.ravel(originalLPF_imgs_np[:,:,:]),linestyle=":",label="LPF")
        # plt.plot(np.ravel(originalHPF_imgs_np[:,:,:]),linestyle="-",label="HPF")
        # plt.legend()
        # plt.show()

        # ny_psf,nx_psf,nz_psf = psfs_tlc[0].shape
        # x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
        # x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
        # psfs_func_list = []
        # normalized_psfs_func_list = []
        # psfs_tlc[0][np.where(np.isnan(psfs_tlc[0]))] = 0
        # import warnings
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     for wv_index in range(nz_psf):
        #         print(wv_index)
        #         model_psf = psfs_tlc[0][:, :, wv_index]
        #         psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
        #         psfs_func_list.append(psf_func)
        #         psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),(model_psf/np.nansum(model_psf)).ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
        #         normalized_psfs_func_list.append(psf_func)


        Npsfs, ny_psf,nx_psf,nz_psf = psfs_tlc.shape
        x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
        x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
        x_psf_vec_hd, y_psf_vec_hd = np.linspace(0,nx_psf * 1.,100)-nx_psf//2,np.linspace(0,ny_psf* 1.,100)-ny_psf//2
        print(x_psf_vec.shape, y_psf_vec.shape, x_psf_grid.shape)
        x_psf_grid_list = np.zeros((len(psfs_tlc_filelist),)+x_psf_grid.shape)
        y_psf_grid_list = np.zeros((len(psfs_tlc_filelist),)+y_psf_grid.shape)
        for k,(psfs_tlc_filename,tmp) in enumerate(zip(psfs_tlc_filelist,psfs_tlc)):
            centers_filename = psfs_tlc_filename.replace("_psfs.fits","_psfs_centers.fits")
            with pyfits.open(centers_filename) as hdulist:
                psfs_centers = hdulist[0].data
                avg_center = np.mean(psfs_centers,axis=0)
                x_psf_grid_list[k,:,:] = x_psf_grid+(nx_psf//2-avg_center[0])
                y_psf_grid_list[k,:,:] = y_psf_grid+(ny_psf//2-avg_center[1])
        normalized_psfs_func_list = []
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for wv_index in range(nz_psf):
                print(wv_index)
                model_psf = psfs_tlc[:,:, :, wv_index]
                # model_psf = model_psf/np.sqrt(np.nansum(model_psf**2,axis=(1,2)))[:,None,None]
                psf_func = interpolate.LSQBivariateSpline(x_psf_grid_list.ravel(),y_psf_grid_list.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
                normalized_psfs_func_list.append(psf_func)

        # 1: planet search, 0: astrometry
        if planet_search:
            wherenotnans = np.where(np.nansum(original_imgs_np,axis=2)!=0)
            real_k_valid_pix = wherenotnans[0]
            real_l_valid_pix = wherenotnans[1]
            row_valid_pix = wherenotnans[0]
            col_valid_pix = wherenotnans[1]
        else:
            real_k_grid,real_l_grid = dk_grid+real_k, dl_grid+real_l
            real_k_valid_pix = real_k_grid.ravel()
            real_l_valid_pix = real_l_grid.ravel()
            l_grid,k_grid = np.meshgrid(np.arange(real_k_grid.shape[1]),np.arange(real_k_grid.shape[0]))
            row_valid_pix = k_grid.ravel()
            col_valid_pix = l_grid.ravel()

        where_not2close2edge = np.where((real_k_valid_pix>padding)*(real_l_valid_pix>padding)*
                                        (real_k_valid_pix<=padny-padding)*(real_l_valid_pix<=padnx-padding))
        real_k_valid_pix = real_k_valid_pix[where_not2close2edge]
        real_l_valid_pix = real_l_valid_pix[where_not2close2edge]
        row_valid_pix = row_valid_pix[where_not2close2edge]
        col_valid_pix = col_valid_pix[where_not2close2edge]

        N_valid_pix = np.size(row_valid_pix)
        Npixtot = N_valid_pix

        # row_indices_list = [[32+5],[32+5],[32+5],[31+5],[31+5],[31+5],[33+5],[33+5],[33+5],[32+15],[32+15],[32+15],[31+15],[31+15],[31+15],[33+15],[33+15],[33+15]]
        # col_indices_list = [[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5]]

        if debug:
            _tpool_init(original_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps,
                                      output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape,outres,outres_shape,outautocorrres,outautocorrres_shape)
            _process_pixels_onlyHPF(real_k_valid_pix[::-1],real_l_valid_pix[::-1],row_valid_pix[::-1],col_valid_pix[::-1],
                                    normalized_psfs_func_list,tlc_spec_list,star_spec,planet_spec_func,wvs_imgs,wvshifts_array,
                                    dtype,cutoff,planet_search,(real_k,real_l),iU,U)
            exit()
        else:
            chunk_size = 5#N_valid_pix//(3*numthreads)
            N_chunks = N_valid_pix//chunk_size
            row_indices_list = []
            col_indices_list = []
            real_k_indices_list = []
            real_l_indices_list = []
            for k in range(N_chunks-1):
                row_indices_list.append(row_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
                col_indices_list.append(col_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
                real_k_indices_list.append(real_k_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
                real_l_indices_list.append(real_l_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
            row_indices_list.append(row_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])
            col_indices_list.append(col_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])
            real_k_indices_list.append(real_k_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])
            real_l_indices_list.append(real_l_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])

            tasks = [tpool.apply_async(_process_pixels_onlyHPF, args=(real_k_indices,real_l_indices,row_indices,col_indices,
                                                                      normalized_psfs_func_list,tlc_spec_list,star_spec,planet_spec_func,wvs_imgs,wvshifts_array,
                                                                      dtype,cutoff,planet_search,(real_k,real_l),iU,U))
                     for real_k_indices, real_l_indices, row_indices, col_indices in zip(real_k_indices_list,real_l_indices_list, row_indices_list, col_indices_list)]
            #save it to shared memory
            for row_index, proc_pixs_task in enumerate(tasks):
                print("Finished image chunk {0}/{1}".format(row_index,len(real_k_indices_list)))
                proc_pixs_task.wait()

        hdulist = pyfits.HDUList()
        if planet_search:
            hdulist.append(pyfits.PrimaryHDU(data=np.moveaxis(output_maps_np,3,1)[:,:,padding:(padny-padding),padding:(padnx-padding)]))
        else:
            hdulist.append(pyfits.PrimaryHDU(data=np.moveaxis(output_maps_np,3,1)))
        try:
            hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits")), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits")), clobber=True)
        hdulist.close()

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=wvshifts_array))
        try:
            hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_wvshifts.fits")), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+"_wvshifts.fits")), clobber=True)
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

        if not planet_search:
            hdulist = pyfits.HDUList()
            if planet_search:
                hdulist.append(pyfits.PrimaryHDU(data=np.concatenate([-padding+real_k_grid[None,padding:(padny-padding),padding:(padnx-padding)],-padding+real_l_grid[None,padding:(padny-padding),padding:(padnx-padding)]],axis=0)))
            else:
                hdulist.append(pyfits.PrimaryHDU(data=np.concatenate([-padding+real_k_grid[None,:,:],-padding+real_l_grid[None,:,:]],axis=0)))
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