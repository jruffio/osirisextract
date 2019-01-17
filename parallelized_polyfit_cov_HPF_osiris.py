__author__ = 'jruffio'

import matplotlib.pyplot as plt
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

def _tpool_init(original_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps, output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Args:
    """
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    # original images from files to read and align&scale. Shape of (N,y,x)
    original = original_imgs
    badpix = badpix_imgs
    originalLPF = originalLPF_imgs
    originalHPF = originalHPF_imgs
    original_shape = original_imgs_shape
    # output images after KLIP processing (amplitude and ...) (5, y, x)
    output = output_maps
    output_shape = output_maps_shape
    # parameters for each image (PA, wavelegnth, image center, image number)
    lambdas = wvs_imgs
    psfs = psfs_stamps
    psfs_shape = psfs_stamps_shape
    Npixproc= 0
    Npixtot=0


def _remove_bad_pixels_z(col_index,nan_mask_boxsize,dtype,window_size=100,threshold=7.):
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
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
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
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
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    for k in wvs_indices:
        tmp = badpix_np[:,:,k]
        tmp[np.where(np.isnan(correlate2d(tmp,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
    badpix_np[0:nan_mask_boxsize//2,:,:] = np.nan
    badpix_np[-nan_mask_boxsize//2+1::,:,:] = np.nan
    badpix_np[:,0:nan_mask_boxsize//2,:] = np.nan
    badpix_np[:,-nan_mask_boxsize//2+1::,:] = np.nan


def _remove_bad_pixels_xy(wvs_indices,dtype):
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
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


# def costfunc_fixed_cov(paras,model,data,cov):
#     diff = (data-np.dot(model,paras_model))
#     tmp = np.linalg.lstsq(cov,diff)
#     cost = np.dot(diff.T,tmp)
#     return cost
#
def solve_known_cov(paras,model_blocks,data_blocks,sqdwvsmatrix_blocks):
    # t1 = time.time()

    q = paras[0]
    corrlen = paras[1]
    # q = 0.1
    # corrlen=10

    data = np.concatenate(data_blocks,axis=0)
    model = np.concatenate(model_blocks,axis=0)
    isigm_blocks = []
    isigd_blocks = []
    sig_blocks_logdets = []
    # mTisigd_blocks = []
    for datablock,modelblock,sqdwvsmatrix in zip(data_blocks,model_blocks,sqdwvsmatrix_blocks):
        block_cov = (1-q)*np.identity(np.shape(sqdwvsmatrix)[0])+q*np.exp(-sqdwvsmatrix/corrlen**2)
        # print(np.linalg.slogdet(block_cov))
        sig_blocks_logdets.append(np.linalg.slogdet(block_cov)[1])
        # sig_blocks_logdets.append(np.log(np.linalg.det(block_cov)))
        # print(modelblock.T.shape,block_cov.shape,modelblock.shape,modelblock.T.shape,block_cov.shape,datablock.shape)
        isig_dm = np.linalg.solve(block_cov,np.concatenate([datablock[:,None],modelblock],axis=1))
        isigd = isig_dm[:,0]
        isigm = isig_dm[:,1::]
        isigm_blocks.append(isigm)
        isigd_blocks.append(isigd)
        # mTisigd_blocks.append(mTisigd)
    isigm = np.concatenate(isigm_blocks,axis=0)
    isigd = np.concatenate(isigd_blocks,axis=0)
    mTisigd = np.dot(isigm.T,data)
    mTisigm = np.dot(model.T,isigm)
    # theta = np.linalg.solve(mTisigm,mTisigd)
    theta,residuals,rank,s = np.linalg.lstsq(mTisigm,mTisigd)

    chi2 = np.dot(data.T,isigd)+np.dot(mTisigd.T,theta)
    minus2logL = np.sum(sig_blocks_logdets)+np.size(data)*np.log(chi2/np.size(data))+1/np.size(data)

    # # t2 = time.time()
    # # print("time2",t1,t2,(t2-t1)/60)
    # print(np.sum(sig_blocks_logdets),np.size(data),np.log(chi2))
    # est_data = np.dot(model,theta)
    # planetsig = model[:,0]*theta[0]
    # plt.plot(data,label="data")
    # plt.plot(est_data,label="model")
    # plt.plot(data-est_data,label="res")
    # plt.plot(planetsig,label="planetsig")
    # plt.legend()
    # plt.show()

    return minus2logL,theta

def _process_pixels(real_k_indices,real_l_indices,row_indices,col_indices,psfs_func_list,star_spec,planet_spec, dtype,sep_planet,usecov,corrlen,q,stardir):
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    output_maps_np = _arraytonumpy(output_maps, output_maps_shape,dtype=dtype)
    psfs_tlc = _arraytonumpy(psfs, psfs_shape,dtype=dtype)
    padny,padnx,padnz = original_shape

    norma_planet_spec = planet_spec/star_spec

    for real_k,real_l,row,col in zip(real_k_indices,real_l_indices,row_indices,col_indices):
        # real_k,real_l = 32+padding,-35.79802955665025+46.8+padding
        k,l = int(np.floor(real_k)),int(np.floor(real_l))
        # print(k,l)

        w = 2
        data = copy(original_np[k-w:k+w+1,:,:])
        data_badpix = badpix_np[k-w:k+w+1,:,:]
        data_ny,data_nx,data_nz = data.shape
        data_wvs = np.tile(np.arange(data_nz)[None,None,:],(data_ny,data_nx,1))
        data_ys = np.tile(np.arange(data_ny)[:,None,None],(1,data_nx,data_nz))
        data_xs = np.tile(np.arange(data_nx)[None,:,None],(data_ny,1,data_nz))

        x_vec, y_vec = np.arange(padnx * 1.)-real_l+sep_planet/0.0203,np.arange(padny* 1.)-real_k
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        x_data_grid, y_data_grid = x_grid[k-w:k+w+1,:], y_grid[k-w:k+w+1,:]
        r_data_grid = np.sqrt(x_data_grid**2+y_data_grid**2)
        th_data_grid = np.arctan2( y_data_grid,x_data_grid) % (2.0 * np.pi)

        planet_model = np.zeros(data.shape)
        pl_x_vec = x_data_grid[0,:]-sep_planet/0.0203
        pl_y_vec = y_data_grid[:,0]
        for z in range(data_nz):
            planet_model[:,:,z] = psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()

        tlc_spec = np.nansum(planet_model,axis=(0,1))
        planet_model = planet_model*norma_planet_spec[None,None,:]

        planet_footprint = np.abs(np.nansum(planet_model/np.nanmax(planet_model,axis=(0,1))[None,None,:],axis=2))
        planet_footprint = planet_footprint/np.nanmax(planet_footprint)

        wv_ref = wvs[0]
        speckle_model = np.zeros((data_ny,data_nx,data_ny,data_nx,data_nz))
        footprint_normalization = np.nanmax(psfs_tlc,axis=(0,1))
        footprint_overlap = np.zeros((data_ny,data_nx))
        for sp_k in range(data_ny):
            for sp_l in range(data_nx):
                sp_r,sp_th = r_data_grid[sp_k,sp_l],th_data_grid[sp_k,sp_l]
                for z in range(data_nz):
                    sp_x = sp_r*wvs[z]/wv_ref*np.cos(sp_th)
                    sp_y = sp_r*wvs[z]/wv_ref*np.sin(sp_th)
                    sp_x_vec = x_data_grid[0,:]-sp_x
                    sp_y_vec = y_data_grid[:,0]-sp_y
                    speckle_model[sp_k,sp_l,:,:,z] = psfs_func_list[z](sp_x_vec,sp_y_vec).transpose()
                speckle_footprint = np.abs(np.nansum(speckle_model[sp_k,sp_l,:,:,:]/footprint_normalization[None,None,:],axis=2))
                speckle_footprint = speckle_footprint/np.nanmax(speckle_footprint)

                footprint_overlap[sp_k,sp_l] = np.sum(planet_footprint*speckle_footprint)

        footprint_overlap_ravel = footprint_overlap.ravel()
        # relevant_speckles = np.where(footprint_overlap_ravel>0.5)[0]
        relevant_speckles = np.where(footprint_overlap_ravel>-1)[0]

        bkg_model = np.zeros((data_ny,data_nx,data_ny,data_nx,data_nz))
        for bkg_k in range(data_ny):
            for bkg_l in range(data_nx):
                bkg_model[bkg_k,bkg_l,bkg_k,bkg_l,:] = 1

        polydeg = 2
        bgkpolydeg = 0#False #3
        # data0 = copy(data)
        # for polydeg in np.arange(1,8):
        #     data = copy(data0)
        if 1:
            model_list = [np.reshape(speckle_model*(wvs**k)[None,None,None,None,:],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
                    for k in range(polydeg+1)]
            if bgkpolydeg:
                bgk_model_list = [np.reshape(bkg_model*(wvs**k)[None,None,None,None,:],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
                        for k in range(bgkpolydeg+1)]
                model = np.concatenate([np.ravel(planet_model)[:,None],]+bgk_model_list+model_list,axis=1)
                modelindices_list = [relevant_speckles+k*data_ny*data_nx+1+(bgkpolydeg+1)*data_ny*data_nx for k in range(polydeg+1)]
                relevant_para = np.concatenate([np.arange((bgkpolydeg+1)*data_ny*data_nx+1),]+modelindices_list)
            else:
                model = np.concatenate([np.ravel(planet_model)[:,None],]+model_list,axis=1)
                modelindices_list = [relevant_speckles+k*data_ny*data_nx+1 for k in range(polydeg+1)]
                relevant_para = np.concatenate([[0],]+modelindices_list)
            # if 0:
            #     pass
            #     modela = np.reshape(speckle_model,(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            #     modelb = np.reshape(speckle_model*wvs[None,None,:,None,None],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            #     modelbkga = np.reshape(bkg_model,(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            #     modelbkgb = np.reshape(bkg_model*wvs[None,None,:,None,None],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            #     model = np.concatenate([np.ravel(planet_model)[:,None],modela,modelb,modelbkga,modelbkgb],axis=1)
            #     relevant_para = np.concatenate([[0],
            #                                     relevant_speckles+1,
            #                                     relevant_speckles+data_ny*data_nx+1,
            #                                     relevant_speckles+2*data_ny*data_nx+1,
            #                                     relevant_speckles+3*data_ny*data_nx+1])

            if not usecov:
                pass
                # data = np.ravel(data)
                # model = np.reshape(model,(data_ny*data_nx*data_nz,np.size(relevant_para)))
                # where_finite_data = np.where(np.isfinite(mybadpixblock))
                # data = data[where_finite_data]
                # model = model[where_finite_data[0],:]
                #
                # paras,residuals,rank,s = np.linalg.lstsq(model,data)

            if usecov:
                model = model[:,relevant_para]
                model = np.reshape(model,(data_ny,data_nx,data_nz,np.size(relevant_para)))

                sqdwvsmatrix_blocks = []
                data_blocks = []
                model_blocks = []
                model_H0_blocks = []
                for bkg_k in range(data_ny):
                    for bkg_l in range(data_nx):
                        mydatablock = data[bkg_k,bkg_l,:]
                        mybadpixblock = data_badpix[bkg_k,bkg_l,:]
                        where_finite_block = np.where(np.isfinite(mybadpixblock))
                        if np.size(where_finite_block[0]) ==0:
                            continue
                        mydatablock = mydatablock[where_finite_block]
                        mywvsblock = wvs[where_finite_block]
                        mysqdwvsblock = (mywvsblock[:,None]-mywvsblock[None,:])**2
                        mymodelblock = model[bkg_k,bkg_l,where_finite_block[0],:]

                        data_blocks.append(mydatablock)
                        sqdwvsmatrix_blocks.append(mysqdwvsblock)
                        model_blocks.append(mymodelblock)
                        model_H0_blocks.append(mymodelblock[:,1::])

                minus2logL,paras = solve_known_cov([q,corrlen],model_blocks,data_blocks,sqdwvsmatrix_blocks)
                AIC = 2*(model.shape[-1])+minus2logL
                minus2logL_H0,paras_H0 = solve_known_cov([q,corrlen],model_H0_blocks,data_blocks,sqdwvsmatrix_blocks)
                AIC_H0 = 2*(model.shape[-1]-1)+minus2logL_H0
                # print(row,col,paras[0],polydeg,AIC,AIC_H0,AIC-AIC_H0)
                output_maps_np[0,row,col] = paras[0]
                output_maps_np[1,row,col] = minus2logL
                output_maps_np[2,row,col] = minus2logL_H0
                output_maps_np[3,row,col] = AIC-AIC_H0
                output_maps_np[4,row,col] = 1
                # exit()
    return

def LPFvsHPF(myvec,cutoff):
    fftmyvec = np.fft.fft(np.concatenate([myvec,myvec[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec)]
    HPF_myvec = myvec - LPF_myvec
    return LPF_myvec,HPF_myvec

def _process_pixels_splitLPFvsHPF(real_k_indices,real_l_indices,row_indices,col_indices,psfs_func_list,star_spec,planet_spec, dtype,sep_planet,usecov,corrlen,q,cutoff,stardir):
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    originalLPF_np = _arraytonumpy(originalLPF, original_shape,dtype=dtype)
    originalHPF_np = _arraytonumpy(originalHPF, original_shape,dtype=dtype)
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    output_maps_np = _arraytonumpy(output_maps, output_maps_shape,dtype=dtype)
    psfs_tlc = _arraytonumpy(psfs, psfs_shape,dtype=dtype)
    padny,padnx,padnz = original_shape

    norma_planet_spec = planet_spec/star_spec

    for real_k,real_l,row,col in zip(real_k_indices,real_l_indices,row_indices,col_indices):
        # real_k,real_l = 32+padding,-35.79802955665025+46.8+padding
        k,l = int(np.floor(real_k)),int(np.floor(real_l))
        # print(k,l)

        w = 2
        data = copy(originalLPF_np[k-w:k+w+1,:,:])
        HPFdata = copy(originalHPF_np[k-w:k+w+1,l-w:l+w+1,:])
        data_badpix = badpix_np[k-w:k+w+1,:,:]
        HPFdata_badpix = badpix_np[k-w:k+w+1,l-w:l+w+1,:]
        data_ny,data_nx,data_nz = data.shape
        # data_wvs = np.tile(np.arange(data_nz)[None,None,:],(data_ny,data_nx,1))
        # data_ys = np.tile(np.arange(data_ny)[:,None,None],(1,data_nx,data_nz))
        # data_xs = np.tile(np.arange(data_nx)[None,:,None],(data_ny,1,data_nz))

                    # if fileelement.attrib["stardir"] == "left":
                    #     fileelement.set("xdefcen",str(19//2-float(fileelement.attrib["sep"])/ 0.0203))
                    #     fileelement.set("ydefcen",str(64//2))
                    # elif fileelement.attrib["stardir"] == "down":
                    #     fileelement.set("xdefcen",str(19//2))
                    #     fileelement.set("ydefcen",str(64//2+float(fileelement.attrib["sep"])/ 0.0203))
        if stardir == "left":
            x_vec, y_vec = np.arange(padnx * 1.)-real_l+sep_planet/0.0203,np.arange(padny* 1.)-real_k
        elif stardir == "down":
            x_vec, y_vec = np.arange(padnx * 1.)-real_l,np.arange(padny* 1.)-real_k-sep_planet/0.0203
        elif stardir == "right":
            pass
        elif stardir == "up":
            pass
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        x_data_grid, y_data_grid = x_grid[k-w:k+w+1,:], y_grid[k-w:k+w+1,:]
        r_data_grid = np.sqrt(x_data_grid**2+y_data_grid**2)
        th_data_grid = np.arctan2( y_data_grid,x_data_grid) % (2.0 * np.pi)

        planet_model = np.zeros(data.shape)
        pl_x_vec = x_data_grid[0,:]-sep_planet/0.0203
        pl_y_vec = y_data_grid[:,0]
        for z in range(data_nz):
            planet_model[:,:,z] = psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()

        tlc_spec = np.nansum(planet_model,axis=(0,1))
        LPFtlc_spec = LPFvsHPF(tlc_spec,cutoff)[0]
        HPFtlc_spec = LPFvsHPF(np.nanmax(planet_model,axis=(0,1)),cutoff)[1]
        # plt.figure(1)
        # plt.plot(LPFtlc_spec/np.nanmean(LPFtlc_spec),label="LPFtlc_spec")
        # plt.plot(data[0,l-w,:]/np.nanmean(data[0,l-w,:]), label="LPFdata")
        # plt.plot(HPFtlc_spec/np.nanstd(HPFtlc_spec),label="HPFtlc_spec")
        # plt.plot(HPFdata[0,0,:]/np.nanstd(HPFdata[0,0,:]), label="HPFdata")
        # plt.legend()
        # plt.show()

        LPFindices = np.arange(0,data_nz,int(data_nz//cutoff))
        LPFtlcplanet_spec,HPFtlcplanet_spec = LPFvsHPF(tlc_spec*norma_planet_spec,cutoff)
        LPFplanet_model = planet_model[:,:,LPFindices]*(LPFtlcplanet_spec/tlc_spec)[None,None,LPFindices]
        HPFplanet_model = planet_model[:,l-w:l+w+1,:]*(HPFtlcplanet_spec/tlc_spec)[None,None,:]

        planet_footprint = np.abs(np.nansum(planet_model/np.nanmax(planet_model,axis=(0,1))[None,None,:],axis=2))
        planet_footprint = planet_footprint/np.nanmax(planet_footprint)


        ####################################
        ## CAUTION: the LPFspeckle_model is not rigorously defined, To be exact I should filter each speckle model the
        # same way I do the original science image, but I don't think it is a big deal which is why I didn't bother.
        ####################################
        Npixs = np.size(LPFindices)*data_ny*data_nx
        wv_ref = wvs[0]
        LPFspeckle_model = np.zeros((data_ny,data_nx,data_ny,data_nx,np.size(LPFindices)))
        footprint_normalization = LPFtlc_spec[LPFindices]
        footprint_overlap = np.zeros((data_ny,data_nx))
        for sp_k in range(data_ny):
            for sp_l in range(data_nx):
                sp_r,sp_th = r_data_grid[sp_k,sp_l],th_data_grid[sp_k,sp_l]
                for z in range(np.size(LPFindices)):
                    sp_x = sp_r*wvs[LPFindices[z]]/wv_ref*np.cos(sp_th)
                    sp_y = sp_r*wvs[LPFindices[z]]/wv_ref*np.sin(sp_th)
                    sp_x_vec = x_data_grid[0,:]-sp_x
                    sp_y_vec = y_data_grid[:,0]-sp_y
                    LPFspeckle_model[sp_k,sp_l,:,:,z] = psfs_func_list[LPFindices[z]](sp_x_vec,sp_y_vec).transpose()/tlc_spec[LPFindices[z]]*LPFtlc_spec[LPFindices[z]]
                speckle_footprint = np.abs(np.nansum(LPFspeckle_model[sp_k,sp_l,:,:,:]/footprint_normalization[None,None,:],axis=2))
                speckle_footprint = speckle_footprint/np.nanmax(speckle_footprint)

                footprint_overlap[sp_k,sp_l] = np.sum(planet_footprint*speckle_footprint)

                # plt.subplot(3,1,1)
                # plt.imshow(speckle_footprint,interpolation="nearest")
                # plt.subplot(3,1,2)
                # plt.imshow(LPFspeckle_model[sp_k,sp_l,:,:,0],interpolation="nearest")
                # plt.subplot(3,1,3)
                # plt.imshow(LPFspeckle_model[sp_k,sp_l,:,:,-1],interpolation="nearest")
                # plt.show()
                # exit()

        footprint_overlap_ravel = footprint_overlap.ravel()
        # relevant_speckles = np.where(footprint_overlap_ravel>0.5)[0]
        relevant_speckles = np.where(footprint_overlap_ravel>-1)[0]

        bkg_model = np.zeros((2*w+1,2*w+1,2*w+1,2*w+1,data_nz))
        for bkg_k in range(2*w+1):
            for bkg_l in range(2*w+1):
                bkg_model[bkg_k,bkg_l,bkg_k,bkg_l,:] = HPFtlc_spec

        LPFpolydeg = 2
        LPFmodel_list = [np.reshape(LPFspeckle_model*(wvs[LPFindices]**k)[None,None,None,None,:],(data_ny*data_nx,Npixs)).transpose()
                for k in range(LPFpolydeg+1)]
        LPFmodel = np.concatenate([np.ravel(LPFplanet_model)[:,None],]+LPFmodel_list,axis=1)
        LPFmodelindices_list = [relevant_speckles+k*data_ny*data_nx+1 for k in range(LPFpolydeg+1)]
        LPFrelevant_para = np.concatenate([[0],]+LPFmodelindices_list)
        LPFmodel = LPFmodel[:,LPFrelevant_para]


        HPFpolydeg = 2
        HPFmodel_list = [np.reshape(bkg_model*(wvs**k)[None,None,None,None,:],((2*w+1)**2,(2*w+1)**2*data_nz)).transpose()
                for k in range(HPFpolydeg+1)]
        HPFmodel = np.concatenate([np.ravel(HPFplanet_model)[:,None],]+HPFmodel_list,axis=1)

        # plt.figure(1)
        # plt.plot(np.ravel(data),label="LPFdata")
        # plt.plot(np.ravel(HPFdata),label="HPFdata")
        # plt.legend()
        # plt.show()
        LPFdata = np.ravel(data[:,:,LPFindices])
        LPFmodel = np.reshape(LPFmodel,(Npixs,LPFmodel.shape[-1]))
        where_finite_data = np.where(np.isfinite(np.ravel(data_badpix[:,:,LPFindices])))
        LPFdata = LPFdata[where_finite_data]
        LPFmodel = LPFmodel[where_finite_data[0],:]
        LPFmodel_H0 = LPFmodel[:,1::]

        t1 = time.time()
        LPFparas,LPFchi2,rank,s = np.linalg.lstsq(LPFmodel,LPFdata)
        LPFparas_H0,LPFchi2_H0,rank,s = np.linalg.lstsq(LPFmodel_H0,LPFdata)
        # bounds = (np.array([0,]+[-np.inf,]*(LPFmodel.shape[-1]-1)),np.array([np.inf,]+[np.inf,]*(LPFmodel.shape[-1]-1)))
        # out = lsq_linear(LPFmodel,LPFdata,bounds=bounds)
        # print(out.status,"coucou")
        # LPFparas = out.x
        # bounds_H0 = (np.array([-np.inf,]*(LPFmodel.shape[-1]-1)),np.array([np.inf,]*(LPFmodel.shape[-1]-1)))
        # LPFparas_H0 = lsq_linear(LPFmodel_H0,LPFdata,bounds=bounds_H0).x
        t2 = time.time()
        if 0:
            print("LPF full",t2-t1)
            # t1 = time.time()
            # sparse_LPFmodel = bsr_matrix(LPFmodel)
            # sparse_LPFmodel_H0 = bsr_matrix(LPFmodel_H0)
            # t2 = time.time()
            # print("LPF bsr_matrix (def)",t2-t1)
            # out = lsqr(sparse_LPFmodel,LPFdata)
            # out = lsqr(sparse_LPFmodel_H0,LPFdata)
            # t2 = time.time()
            # print("LPF bsr_matrix",t2-t1)
            # t1 = time.time()
            # sparse_LPFmodel = csc_matrix(LPFmodel)
            # sparse_LPFmodel_H0 = csc_matrix(LPFmodel_H0)
            # t2 = time.time()
            # print("LPF csc_matrix (def)",t2-t1)
            # out = lsqr(sparse_LPFmodel,LPFdata)
            # out = lsqr(sparse_LPFmodel_H0,LPFdata)
            # t2 = time.time()
            # print("LPF csc_matrix",t2-t1)
            # t1 = time.time()
            # sparse_LPFmodel = csr_matrix(LPFmodel)
            # sparse_LPFmodel_H0 = csr_matrix(LPFmodel_H0)
            # t2 = time.time()
            # print("LPF csr_matrix (def)",t2-t1)
            # out = lsqr(sparse_LPFmodel,LPFdata)
            # out = lsqr(sparse_LPFmodel_H0,LPFdata)
            # t2 = time.time()
            # print("LPF csr_matrix",t2-t1)
            # t1 = time.time()
            # sparse_LPFmodel = dia_matrix(LPFmodel)
            # sparse_LPFmodel_H0 = dia_matrix(LPFmodel_H0)
            # t2 = time.time()
            # print("LPF dia_matrix (def)",t2-t1)
            # out = lsqr(sparse_LPFmodel,LPFdata)
            # out = lsqr(sparse_LPFmodel_H0,LPFdata)
            # t2 = time.time()
            # print("LPF dia_matrix",t2-t1)


        data_model = np.dot(LPFmodel,LPFparas)
        data_model_H0 = np.dot(LPFmodel_H0,LPFparas_H0)
        LPFchi2 = np.nansum((data_model-LPFdata)**2)
        LPFchi2_H0 = np.nansum((data_model_H0-LPFdata)**2)
        # # print(LPFchi2,np.nansum((data_model-LPFdata)**2))
        # plt.figure(1)
        # plt.plot(LPFdata,label="LPFdata")
        # plt.plot(data_model,label="data_model")
        # plt.plot(LPFmodel[:,0]*LPFparas[0],label="signal")
        # plt.plot(data_model-LPFdata,label="data_model-LPFdata")
        # plt.plot(data_model_H0,label="data_model_H0")
        # plt.plot(data_model_H0-LPFdata,label="data_model_H0-LPFdata")
        # plt.legend()
        # plt.show()


        HPFdata = np.ravel(HPFdata)
        HPFmodel = np.reshape(HPFmodel,((2*w+1)**2*data_nz,HPFmodel.shape[-1]))
        where_finite_data = np.where(np.isfinite(np.ravel(HPFdata_badpix)))
        HPFdata = HPFdata[where_finite_data]
        HPFmodel = HPFmodel[where_finite_data[0],:]
        HPFmodel_H0 = HPFmodel[:,1::]

        # t1 = time.time()
        HPFparas,HPFchi2,rank,s = np.linalg.lstsq(HPFmodel,HPFdata)
        HPFparas_H0,HPFchi2_H0,rank,s = np.linalg.lstsq(HPFmodel_H0,HPFdata)
        # t2 = time.time()
        # bounds = (np.array([0,]+[-np.inf,]*(HPFmodel.shape[-1]-1)),np.array([np.inf,]+[np.inf,]*(HPFmodel.shape[-1]-1)))
        # HPFparas = lsq_linear(HPFmodel,HPFdata,bounds=bounds).x
        # bounds_H0 = (np.array([-np.inf,]*(HPFmodel.shape[-1]-1)),np.array([np.inf,]*(HPFmodel.shape[-1]-1)))
        # HPFparas_H0 = lsq_linear(HPFmodel_H0,HPFdata,bounds=bounds_H0).x
        if 0:
            print("HPF full",t2-t1)
            # t1 = time.time()
            # sparse_HPFmodel = bsr_matrix(HPFmodel)
            # sparse_HPFmodel_H0 = bsr_matrix(HPFmodel_H0)
            # t2 = time.time()
            # print("HPF bsr_matrix (def)",t2-t1)
            # out = lsqr(sparse_HPFmodel,HPFdata)
            # out = lsqr(sparse_HPFmodel_H0,HPFdata)
            # t2 = time.time()
            # print("HPF bsr_matrix",t2-t1)
            # t1 = time.time()
            # sparse_HPFmodel = csc_matrix(HPFmodel)
            # sparse_HPFmodel_H0 = csc_matrix(HPFmodel_H0)
            # t2 = time.time()
            # print("HPF csc_matrix (def)",t2-t1)
            # out = lsqr(sparse_HPFmodel,HPFdata)
            # out = lsqr(sparse_HPFmodel_H0,HPFdata)
            # t2 = time.time()
            # print("HPF csc_matrix",t2-t1)
            # t1 = time.time()
            # sparse_HPFmodel = csr_matrix(HPFmodel)
            # sparse_HPFmodel_H0 = csr_matrix(HPFmodel_H0)
            # t2 = time.time()
            # print("HPF csr_matrix (def)",t2-t1)
            # out = lsqr(sparse_HPFmodel,HPFdata)
            # out = lsqr(sparse_HPFmodel_H0,HPFdata)
            # t2 = time.time()
            # print("HPF csr_matrix",t2-t1)
            # t1 = time.time()
            # sparse_HPFmodel = dia_matrix(HPFmodel)
            # sparse_HPFmodel_H0 = dia_matrix(HPFmodel_H0)
            # t2 = time.time()
            # print("HPF dia_matrix (def)",t2-t1)
            # out = lsqr(sparse_HPFmodel,HPFdata)
            # out = lsqr(sparse_HPFmodel_H0,HPFdata)
            # t2 = time.time()
            # print("HPF dia_matrix",t2-t1)
            # exit()

        data_model = np.dot(HPFmodel,HPFparas)
        data_model_H0 = np.dot(HPFmodel_H0,HPFparas_H0)
        HPFchi2 = np.nansum((data_model-HPFdata)**2)
        HPFchi2_H0 = np.nansum((data_model_H0-HPFdata)**2)
        # plt.figure(2)
        # plt.plot(HPFdata,label="HPFdata")
        # plt.plot(data_model,label="data_model")
        # plt.plot(HPFmodel[:,0]*HPFparas[0],label="signal")
        # plt.plot(data_model-HPFdata,label="data_model-HPFdata")
        # plt.plot(data_model_H0,label="data_model_H0")
        # plt.plot(data_model_H0-HPFdata,label="data_model_H0-HPFdata")
        # plt.legend()
        # plt.show()

        # plt.figure(1)
        # CCF0 = np.correlate(HPFmodel[:,0],HPFmodel[:,0],mode="same")
        # CCF = np.correlate(HPFdata,HPFmodel[:,0],mode="same")
        # plt.plot(CCF0/np.max(CCF0),label="CCF0")
        # plt.plot(CCF/np.max(CCF),label="CCF")
        # plt.legend()
        # plt.show()

        # print(LPFchi2,LPFchi2_H0,HPFchi2,HPFchi2_H0)
        Npixs_LPFdata = np.size(LPFdata)
        minus2logL_LPF = Npixs_LPFdata*np.log(LPFchi2/Npixs_LPFdata)+1./Npixs_LPFdata
        minus2logL_LPF_H0 = Npixs_LPFdata*np.log(LPFchi2_H0/Npixs_LPFdata)+1./Npixs_LPFdata
        Npixs_HPFdata = np.size(HPFdata)
        minus2logL_HPF = Npixs_HPFdata*np.log(HPFchi2/Npixs_HPFdata)+1./Npixs_HPFdata
        minus2logL_HPF_H0 = Npixs_HPFdata*np.log(HPFchi2_H0/Npixs_HPFdata)+1./Npixs_HPFdata
        AIC_LPF = 2*(LPFmodel.shape[-1])+minus2logL_LPF
        AIC_LPF_H0 = 2*(LPFmodel_H0.shape[-1])+minus2logL_LPF_H0
        AIC_HPF = 2*(HPFmodel.shape[-1])+minus2logL_HPF
        AIC_HPF_H0 = 2*(HPFmodel_H0.shape[-1])+minus2logL_HPF_H0
        # print(minus2logL_LPF,minus2logL_LPF_H0,minus2logL_HPF,minus2logL_HPF_H0)
        # print(AIC_LPF,AIC_LPF_H0,AIC_HPF,AIC_HPF_H0)
        output_maps_np[0,row,col] = LPFparas[0]
        output_maps_np[1,row,col] = AIC_LPF
        output_maps_np[2,row,col] = AIC_LPF_H0
        if LPFparas[0]>=0:
            output_maps_np[3,row,col] = AIC_LPF_H0-AIC_LPF
        else:
            output_maps_np[3,row,col] = 0
        output_maps_np[4,row,col] = HPFparas[0]
        output_maps_np[5,row,col] = AIC_HPF
        output_maps_np[6,row,col] = AIC_HPF_H0
        if HPFparas[0] >= 0:
            output_maps_np[7,row,col] = AIC_HPF_H0-AIC_HPF
        else:
            output_maps_np[7,row,col] = 0
        output_maps_np[8,row,col] = minus2logL_LPF_H0+minus2logL_HPF_H0-minus2logL_LPF-minus2logL_HPF
        # print(output_maps_np[:,row,col])
        # exit()
    return


def _process_pixels_onlyHPF(real_k_indices,real_l_indices,row_indices,col_indices,normalized_psfs_func_list,tlc_spec_list,star_spec,planet_spec, dtype,sep_planet,usecov,corrlen,q,cutoff,stardir):
    global original,badpix,originalLPF,originalHPF, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    originalLPF_np = _arraytonumpy(originalLPF, original_shape,dtype=dtype)
    originalHPF_np = _arraytonumpy(originalHPF, original_shape,dtype=dtype)
    badpix_np = _arraytonumpy(badpix, original_shape,dtype=dtype)
    output_maps_np = _arraytonumpy(output_maps, output_maps_shape,dtype=dtype)
    psfs_tlc = _arraytonumpy(psfs, psfs_shape,dtype=dtype)
    padny,padnx,padnz = original_shape

    norma_planet_spec = planet_spec/star_spec
    HPFtlc_spec_list = []
    for tlc_spectrum in tlc_spec_list:
        mytlchpfspec = LPFvsHPF(tlc_spectrum,cutoff)[1]
        HPFtlc_spec_list.append(mytlchpfspec/np.nanstd(mytlchpfspec))

    for real_k,real_l,row,col in zip(real_k_indices,real_l_indices,row_indices,col_indices):
        # real_k,real_l = 32+padding,-35.79802955665025+46.8+padding
        k,l = int(np.floor(real_k)),int(np.floor(real_l))
        # print(k,l)

        w = 2
        HPFdata = copy(originalHPF_np[k-w:k+w+1,l-w:l+w+1,:])
        LPFdata = copy(originalLPF_np[k-w:k+w+1,l-w:l+w+1,:])
        HPFdata_badpix = badpix_np[k-w:k+w+1,l-w:l+w+1,:]
        data_ny,data_nx,data_nz = HPFdata.shape

        x_vec, y_vec = np.arange(padnx * 1.)-real_l,np.arange(padny* 1.)-real_k
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        x_data_grid, y_data_grid = x_grid[k-w:k+w+1,l-w:l+w+1], y_grid[k-w:k+w+1,l-w:l+w+1]

        planet_model = np.zeros(HPFdata.shape)
        pl_x_vec = x_data_grid[0,:]
        pl_y_vec = y_data_grid[:,0]
        for z in range(data_nz):
            planet_model[:,:,z] = normalized_psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()

        # HPFtlc_spec = LPFvsHPF(np.nanmax(planet_model,axis=(0,1)),cutoff)[1]



        LPFtlcplanet_spec,HPFtlcplanet_spec = LPFvsHPF(tlc_spec_list[0]*norma_planet_spec,cutoff)
        # plt.figure(2)
        # plt.plot(HPFtlcplanet_spec)
        HPFplanet_model = planet_model*(HPFtlcplanet_spec/np.nanstd(HPFtlcplanet_spec)*np.nanstd(HPFtlc_spec_list[0]))[None,None,:]


        HPFpolydeg = 0
        HPFmodel_list = []
        # plt.figure(1)
        for HPFtlc_spectrum in HPFtlc_spec_list:
            # plt.plot(HPFtlc_spectrum)
            for k in range(HPFpolydeg+1):
                bkg_model = np.zeros((2*w+1,2*w+1,2*w+1,2*w+1,data_nz))
                for bkg_k in range(2*w+1):
                    for bkg_l in range(2*w+1):
                        myspec = LPFdata[bkg_k,bkg_l,:]*(HPFtlc_spectrum*(wvs**k))
                        bkg_model[bkg_k,bkg_l,bkg_k,bkg_l,:] = myspec/np.nanstd(myspec)
                HPFmodel_list.append(np.reshape(bkg_model,((2*w+1)**2,(2*w+1)**2*data_nz)).transpose())
        HPFmodel = np.concatenate([np.ravel(HPFplanet_model)[:,None],]+HPFmodel_list,axis=1)
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

        HPFdata = np.ravel(HPFdata)
        HPFmodel = np.reshape(HPFmodel,((2*w+1)**2*data_nz,HPFmodel.shape[-1]))
        where_finite_data = np.where(np.isfinite(np.ravel(HPFdata_badpix)))
        HPFdata = HPFdata[where_finite_data]
        HPFmodel = HPFmodel[where_finite_data[0],:]
        HPFmodel_H0 = HPFmodel[:,1::]
        HPFmodel_H1 = HPFmodel[:,0:1]

        # t1 = time.time()
        HPFparas,HPFchi2,rank,s = np.linalg.lstsq(HPFmodel,HPFdata)
        HPFparas_H1,HPFchi2_H1,rank,s = np.linalg.lstsq(HPFmodel_H1,HPFdata)
        HPFparas_H0,HPFchi2_H0,rank,s = np.linalg.lstsq(HPFmodel_H0,HPFdata)

        data_model = np.dot(HPFmodel,HPFparas)
        data_model_H1 = np.dot(HPFmodel_H1,HPFparas_H1)
        data_model_H0 = np.dot(HPFmodel_H0,HPFparas_H0)
        HPFchi2 = np.nansum((data_model-HPFdata)**2)
        HPFchi2_H1 = np.nansum((data_model_H1-HPFdata)**2)
        HPFchi2_H0 = np.nansum((data_model_H0-HPFdata)**2)


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
        Npixs_HPFdata = np.size(HPFdata)
        minus2logL_HPF = Npixs_HPFdata*np.log(HPFchi2/Npixs_HPFdata)+1./Npixs_HPFdata
        minus2logL_HPF_H1 = Npixs_HPFdata*np.log(HPFchi2_H1/Npixs_HPFdata)+1./Npixs_HPFdata
        minus2logL_HPF_H0 = Npixs_HPFdata*np.log(HPFchi2_H0/Npixs_HPFdata)+1./Npixs_HPFdata
        AIC_HPF = 2*(HPFmodel.shape[-1])+minus2logL_HPF
        AIC_HPF_H1 = 2*(HPFmodel_H1.shape[-1])+minus2logL_HPF_H1
        AIC_HPF_H0 = 2*(HPFmodel_H0.shape[-1])+minus2logL_HPF_H0
        output_maps_np[0,row,col] = HPFparas[0]
        output_maps_np[1,row,col] = AIC_HPF
        output_maps_np[2,row,col] = AIC_HPF_H0
        output_maps_np[3,row,col] = AIC_HPF_H0-AIC_HPF
        if HPFparas[0] >= 0:
            output_maps_np[4,row,col] = AIC_HPF_H0-AIC_HPF
        else:
            output_maps_np[4,row,col] = 0

        output_maps_np[5,row,col] = HPFparas_H1[0]
        output_maps_np[6,row,col] = AIC_HPF_H1
        output_maps_np[7,row,col] = AIC_HPF_H0
        output_maps_np[8,row,col] = AIC_HPF_H0-AIC_HPF_H1
        if HPFparas_H1[0] >= 0:
            output_maps_np[9,row,col] = AIC_HPF_H0-AIC_HPF_H1
        else:
            output_maps_np[9,row,col] = 0
        # print(output_maps_np[:,row,col])
        # exit()
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


    if 1:# HR 8799 c 20100715
        # date = "100715"
        date = "101104"
        inputDir = "/home/sda/jruffio/osiris_data/HR_8799_c/20"+date+"/reduced_jb/"
        outputdir = "/home/sda/jruffio/osiris_data/HR_8799_c/20"+date+"/reduced_jb/20181120_out/"
        outputdir = "/home/sda/jruffio/osiris_data/HR_8799_c/20"+date+"/reduced_jb/20181205_HPF_only/"
        filelist = glob.glob(os.path.join(inputDir,"s"+date+"*20.fits"))
        filelist.sort()
        # filename = filelist[12]
        # psfs_tlc_filename = "/home/sda/jruffio/osiris_data/HR_8799_c/20"+date+"/reduced_telluric_JB/HD_210501/s"+date+"_a005002_Kbb_020_psfs.fits"
        psfs_tlc_filelist = glob.glob("/home/sda/jruffio/osiris_data/HR_8799_c/20"+date+"/reduced_telluric_JB/*/s*_psfs_badpix.fits")[0:1]
        template_spec_filename="/home/sda/jruffio/osiris_data/HR_8799_c/hr8799c_osiris_template.save"
        star_spec_filename = "/home/sda/jruffio/osiris_data/HR_8799_c/hr_8799_c_pickles_spectrum_Kbb.csv"
        # planet_coords = [[11,32],[12,27],[12,33],[12,39],[10,33],[9,28],[8,38],[10,32.5],[9,32],[10,33],[10,35],[10,33], #in order: image 10 to 21
        #                  [7,34],[5,35],[8,35],[7.5,33],[9.5,34.5]]
        # file_centers = [[x-sep_planet/ 0.0203,y] for x,y in planet_coords]
        numthreads = 32
        centermode = "visu" #ADI #def
        fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos_jb.xml"

        padding = 5
        planet_search = True
        debug = False
        real_k,real_l = 32+padding,-35.79802955665025+46.8+padding
        # real_k,real_l = 50,15
        # real_k,real_l = 32+padding-10,-35.79802955665025+46.8+padding
        # real_k,real_l = 51,17
        # real_k,real_l = 39,16
        real_k,real_l = 39+5,12
        #for astro
        # real_l_grid,real_k_grid = np.meshgrid(np.linspace(-3,3,7),np.linspace(-3,3,7))
        real_l_grid,real_k_grid = np.array([[0]]),np.array([[0]])
    else:
        inputDir = sys.argv[1]
        outputdir = sys.argv[2]
        filename = sys.argv[3]
        psfs_tlc_filename = sys.argv[4]
        template_spec_filename = sys.argv[5]
        star_spec_filename  = sys.argv[6]
        numthreads = int(sys.argv[7])
        centermode = sys.argv[8]
        fileinfos_filename = "/home/users/jruffio/OSIRIS/osirisextract/fileinfos_jb.xml"
        padding = 5

    if 0: # test of sparse matrices
        from scipy.sparse.linalg import lsqr
        from scipy.sparse import csc_matrix
        from scipy.sparse import bsr_matrix
        from scipy.sparse import csr_matrix
        m=200
        r=500
        n=m*r
        mat = np.zeros((n,m))
        for k in range(m):
            mat[k*r:(k+1)*r,k] = 1
        mat[:,0] = np.random.randn(n)
        sparse_csc = csc_matrix(mat)
        sparse_bsr = bsr_matrix(mat)
        sparse_csr = csr_matrix(mat)
        data = np.random.randn(n)
        t1 = time.time()
        for k in range(1):
            out = np.linalg.lstsq(mat,data)
            # print(out[0])
        t2 = time.time()
        print("full",t2-t1)
        t1 = time.time()
        for k in range(1):
            out = lsqr(sparse_csc,data)
            # print(out[0])
        t2 = time.time()
        print("sparse_csc",t2-t1)
        t1 = time.time()
        for k in range(1):
            out = lsqr(sparse_bsr,data)
            # print(out[0])
        t2 = time.time()
        print("sparse_bsr",t2-t1)
        t1 = time.time()
        for k in range(1):
            out = lsqr(sparse_csr,data)
            # print(out[0])
        t2 = time.time()
        print("sparse_csr",t2-t1)
        exit()

    # if splitLPFvsHPF = False
    usecov = True
    corrlen = 10
    q = 0.1
    #
    dtype = ctypes.c_float
    nan_mask_boxsize=3
    #
    smalldata = False
    #
    splitLPFvsHPF = True
    if smalldata:
        cutoff = int(np.round(80./10.))
    else:
        cutoff = 80#80

    for filename in filelist:#[13::]:
    # if 1:
        # suffix = "polyfit_"+centermode+"cen"+"_testmaskbadpix"
        # suffix = "polyfit_"+centermode+"cen"+"_resmask_maskbadpix"
        # suffix = "polyfit_"+centermode+"cen"+"_resmask_norma"
        # suffix = "polyfit_"+centermode+"cen"+"_resmask_norma_bkg"
        # suffix = "polyfit_"+centermode+"cen"+"_cov_all"
        suffix = "HPF_cutoff{0}_new".format(cutoff)
        # suffix = "HPFtestmultipletlc_cutoff{0}".format(cutoff)

        if 0:
            out_file = os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits"))
            with pyfits.open(out_file) as hdulist:
                cube = hdulist[0].data
                plt.plot(cube[7,:,17],label="CCF y-dir")
                plt.show()
            exit()

        tree = ET.parse(fileinfos_filename)
        root = tree.getroot()

        filebasename = os.path.basename(filename)
        planet_c = root.find("c")
        fileelement = planet_c.find(filebasename)
        stardir = fileelement.attrib["stardir"]
        # if stardir != "left":
        #     continue
        # print(stardir)
        # continue
        center = [float(fileelement.attrib["x"+centermode+"cen"]),float(fileelement.attrib["y"+centermode+"cen"])]
        sep_planet = float(fileelement.attrib["sep"])

        if not os.path.exists(os.path.join(outputdir)):
            os.makedirs(os.path.join(outputdir))


        with pyfits.open(filename) as hdulist:
            imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
            prihdr = hdulist[0].header
            imgs = np.moveaxis(imgs,0,2)
            # print(filename)
            # plt.imshow(np.nansum(imgs,axis=2))
            # plt.show()
            # exit()
        # imgs[np.where(imgs==0)] = np.nan
        ny,nx,nz = imgs.shape
        init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
        dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
        wvs=np.arange(init_wv,init_wv+dwv*nz,dwv)

        psfs_tlc = []
        tlc_spec_list = []
        for psfs_tlc_filename in psfs_tlc_filelist:
            with pyfits.open(psfs_tlc_filename) as hdulist:
                mypsfs = hdulist[0].data
                # psfs_tlc_prihdr = hdulist[0].header
                mypsfs = np.moveaxis(mypsfs,0,2)
                psfs_tlc.append(mypsfs)
                tlc_spec_list.append(np.nansum(mypsfs,axis=(0,1)))


        travis_spectrum = scio.readsav(template_spec_filename)
        planet_spec = np.array(travis_spectrum["fk_bin"])
        planet_spec = planet_spec/np.mean(planet_spec)


        with open(star_spec_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            list_starspec = list(csv_reader)
            starspec_str_arr = np.array(list_starspec, dtype=np.str)
            col_names = starspec_str_arr[0]
            star_spec = starspec_str_arr[1::,1].astype(np.float)
            star_spec = star_spec/np.mean(star_spec)
            star_spec_wvs = starspec_str_arr[1::,0].astype(np.float)

        # reduce dimensionality
        if smalldata:
            for k in range(len(psfs_tlc_filelist)):
                psfs_tlc[k] = psfs_tlc[k][:,:,::10]
            wvs = wvs[::10]
            imgs = imgs[:,:,::10]
            planet_spec = planet_spec[::10]
            star_spec = star_spec[::10]
            nz,ny,nx = imgs.shape


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
        if planet_search:
            output_maps = mp.Array(dtype, 10*padny*padnx)
            output_maps_shape = (10,padny,padnx)
        else:
            output_maps_shape = (10,real_l_grid.shape[0],real_l_grid.shape[1])
            output_maps = mp.Array(dtype, 10*real_l_grid.shape[0]*real_l_grid.shape[1])
        output_maps_np = _arraytonumpy(output_maps,output_maps_shape,dtype=dtype)
        output_maps_np[:] = np.nan
        wvs_imgs = wvs
        psfs_stamps = mp.Array(dtype, len(psfs_tlc_filelist)*np.size(psfs_tlc[0]))
        psfs_stamps_shape = [len(psfs_tlc_filelist),psfs_tlc[0].shape[0],psfs_tlc[0].shape[1],psfs_tlc[0].shape[2]]
        psfs_stamps_np = _arraytonumpy(psfs_stamps, psfs_stamps_shape,dtype=dtype)
        for k in range(len(psfs_tlc_filelist)):
            psfs_stamps_np[k,:,:,:] = psfs_tlc[k]


        ######################
        # INIT threads and shared memory
        tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                        initargs=(original_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps,
                                  output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape),
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


        if 0:
            _tpool_init(original_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps,
                                      output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape)
            _HPF_z(11,cutoff, dtype)

            plt.figure(2)
            myvec = planet_spec/star_spec*tlc_spec
            fftmyvec = np.fft.fft(np.concatenate([myvec,myvec[::-1]],axis=0))
            LPF_fftmyvec = copy(fftmyvec)
            LPF_fftmyvec[cutoff:(2*np.size(myvec)-cutoff+1)] = 0
            LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec)]
            HPF_myvec = myvec - LPF_myvec
            plt.plot(myvec,label="original")
            plt.plot(LPF_myvec,label="LPF_myvec")
            plt.plot(HPF_myvec,label="HPF_myvec")
            plt.figure(3)
            myvec = planet_spec/star_spec
            fftmyvec = np.fft.fft(np.concatenate([myvec,myvec[::-1]],axis=0))
            LPF_fftmyvec = copy(fftmyvec)
            LPF_fftmyvec[cutoff:(2*np.size(myvec)-cutoff+1)] = 0
            LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec)]
            HPF_myvec = myvec - LPF_myvec
            plt.plot(myvec,label="original")
            plt.plot(LPF_myvec,label="LPF_myvec")
            plt.plot(HPF_myvec,label="HPF_myvec")

            plt.figure(4)
            myvec = tlc_spec
            fftmyvec = np.fft.fft(np.concatenate([myvec,myvec[::-1]],axis=0))
            LPF_fftmyvec = copy(fftmyvec)
            LPF_fftmyvec[cutoff:(2*np.size(myvec)-cutoff+1)] = 0
            LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec)]
            HPF_myvec = myvec - LPF_myvec
            plt.plot(myvec,label="original")
            plt.plot(LPF_myvec,label="LPF_myvec")
            plt.plot(HPF_myvec,label="HPF_myvec")

            plt.show()
            exit()

        if splitLPFvsHPF:
            tasks = [tpool.apply_async(_HPF_z, args=(col_index,cutoff, dtype))
                     for col_index in range(padnx)]
            #save it to shared memory
            for col_index, bad_pix_task in enumerate(tasks):
                print("Finished col {0}".format(col_index))
                bad_pix_task.wait()

        # plt.plot(np.ravel(originalLPF_imgs_np[:,:,:]),linestyle=":",label="LPF")
        # plt.plot(np.ravel(originalHPF_imgs_np[:,:,:]),linestyle=":",label="HPF")
        # plt.legend()
        # plt.show()

        ny_psf,nx_psf,nz_psf = psfs_tlc[0].shape
        x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
        x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
        psfs_func_list = []
        normalized_psfs_func_list = []
        psfs_tlc[0][np.where(np.isnan(psfs_tlc[0]))] = 0
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for wv_index in range(nz_psf):
                print(wv_index)
                model_psf = psfs_tlc[0][:, :, wv_index]
                psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
                psfs_func_list.append(psf_func)
                psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),(model_psf/np.nansum(model_psf)).ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
                normalized_psfs_func_list.append(psf_func)

        # 1: planet search, 0: astrometry
        if planet_search:
            wherenotnans = np.where(np.nansum(original_imgs_np,axis=2)!=0)
            real_k_valid_pix = wherenotnans[0]
            real_l_valid_pix = wherenotnans[1]
            row_valid_pix = wherenotnans[0]
            col_valid_pix = wherenotnans[1]
        else:
            real_k_grid,real_l_grid = real_k_grid+real_k, real_l_grid+real_l
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
        # print(Npixtot)
        # exit()

        # row_indices_list = [[32+5],[32+5],[32+5],[31+5],[31+5],[31+5],[33+5],[33+5],[33+5],[32+15],[32+15],[32+15],[31+15],[31+15],[31+15],[33+15],[33+15],[33+15]]
        # col_indices_list = [[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5]]

        if debug:
            _tpool_init(original_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps,
                                      output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape)
            if splitLPFvsHPF:
                _process_pixels_onlyHPF(real_k_valid_pix[::-1],real_l_valid_pix[::-1],row_valid_pix[::-1],col_valid_pix[::-1],normalized_psfs_func_list,tlc_spec_list,star_spec,planet_spec, dtype,sep_planet,usecov,corrlen,q,cutoff,stardir)
                # _process_pixels_splitLPFvsHPF(real_k_valid_pix[::-1],real_l_valid_pix[::-1],row_valid_pix[::-1],col_valid_pix[::-1],psfs_func_list,star_spec,planet_spec, dtype,sep_planet,usecov,corrlen,q,cutoff,stardir)
            else:
                _process_pixels(real_k_valid_pix,real_l_valid_pix,row_valid_pix,col_valid_pix,psfs_func_list,star_spec,planet_spec, dtype,sep_planet,usecov,corrlen,q,stardir)
            exit()
        else:
            chunk_size = N_valid_pix//(3*numthreads)
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

            # if 0:
            #     mask = np.zeros((padny,padnx))
            #     _tpool_init(original_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps,
            #                               output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape)
            #     for real_k_indices, real_l_indices, row_indices, col_indices in zip(real_k_indices_list,real_l_indices_list, row_indices_list, col_indices_list):
            #         _process_pixels(real_k_valid_pix,real_l_valid_pix,row_valid_pix,col_valid_pix,psfs_func_list,star_spec,planet_spec, dtype,sep_planet,usecov,corrlen,q)
            #         for row,col in zip(row_indices, col_indices):
            #             print(row,col)
            #             mask[row,col] = 1
            #     plt.imshow(mask,interpolation="nearest")
            #     plt.show()

            if splitLPFvsHPF:
                tasks = [tpool.apply_async(_process_pixels_onlyHPF, args=(real_k_indices,real_l_indices,row_indices,col_indices,normalized_psfs_func_list,tlc_spec_list,star_spec,planet_spec, dtype,sep_planet,usecov,corrlen,q,cutoff,stardir))
                         for real_k_indices, real_l_indices, row_indices, col_indices in zip(real_k_indices_list,real_l_indices_list, row_indices_list, col_indices_list)]
                # tasks = [tpool.apply_async(_process_pixels_splitLPFvsHPF, args=(real_k_indices,real_l_indices,row_indices,col_indices,psfs_func_list,star_spec,planet_spec, dtype,sep_planet,usecov,corrlen,q,cutoff,stardir))
                #          for real_k_indices, real_l_indices, row_indices, col_indices in zip(real_k_indices_list,real_l_indices_list, row_indices_list, col_indices_list)]
            else:
                tasks = [tpool.apply_async(_process_pixels, args=(real_k_indices,real_l_indices,row_indices,col_indices,psfs_func_list,star_spec,planet_spec, dtype,sep_planet,usecov,corrlen,q,stardir))
                         for real_k_indices, real_l_indices, row_indices, col_indices in zip(real_k_indices_list,real_l_indices_list, row_indices_list, col_indices_list)]

        #save it to shared memory
        for row_index, proc_pixs_task in enumerate(tasks):
            print("Finished image chunk {0}".format(row_index))
            proc_pixs_task.wait()

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=output_maps_np))
        try:
            hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits")), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits")), clobber=True)
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