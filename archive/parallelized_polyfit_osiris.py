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
from copy import copy
from astropy.stats import mad_std
import scipy.io as scio
from scipy.optimize import minimize
import sys
import xml.etree.ElementTree as ET
import csv

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

def _tpool_init(original_imgs, original_imgs_shape, output_maps, output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Args:
    """
    global original, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    # original images from files to read and align&scale. Shape of (N,y,x)
    original = original_imgs
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


def _remove_bad_pixels(col_index,dtype):
    global original, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    tmpcube = copy(original_np[:,:,col_index])
    nan_mask_boxsize = 3
    for m in np.arange(0,ny):
        myvec = tmpcube[:,m]
        wherefinite = np.where(np.isfinite(myvec))
        if np.size(wherefinite[0])<10:
            continue
        smooth_vec = median_filter(myvec,footprint=np.ones(100),mode="constant",cval=0.0)
        myvec = myvec - smooth_vec
        wherefinite = np.where(np.isfinite(myvec))
        mad = mad_std(myvec[wherefinite])
        original_np[np.where(np.abs(myvec)>7*mad)[0],m,col_index] = np.nan
        widen_nans = np.where(np.isnan(np.correlate(original_np[:,m,col_index],np.ones(nan_mask_boxsize),mode="same")))[0]
        original_np[widen_nans,m,col_index] = np.nan

def _remove_edges(wvs_indices,nan_mask_boxsize,dtype):
    global original, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)

    for k in wvs_indices:
        original_np[k][np.where(np.isnan(correlate2d(original_np[k],np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
    original_np[:,0:nan_mask_boxsize//2,:] = np.nan
    original_np[:,-nan_mask_boxsize//2+1::,:] = np.nan
    original_np[:,:,0:nan_mask_boxsize//2] = np.nan
    original_np[:,:,-nan_mask_boxsize//2+1::] = np.nan

def _process_pixels(real_k_indices,real_l_indices,row_indices,col_indices,psfs_func_list,star_spec,planet_spec, dtype,sep_planet):
    global original, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    output_maps_np = _arraytonumpy(output_maps, output_maps_shape,dtype=dtype)
    psfs_tlc = _arraytonumpy(psfs, psfs_shape,dtype=dtype)
    padnz,padny,padnx = original_shape

    norma_planet_spec = planet_spec/star_spec

    for real_k,real_l,row,col in zip(real_k_indices,real_l_indices,row_indices,col_indices):
        # real_k,real_l = 32+padding,-35.79802955665025+46.8+padding
        k,l = int(np.floor(real_k)),int(np.floor(real_l))
        # print(k,l)

        w = 2
        data = copy(original_np[:,k-w:k+w+1,:])
        data_nz,data_ny,data_nx = data.shape

        x_vec, y_vec = np.arange(padnx * 1.)-real_l+sep_planet/0.0203,np.arange(padny* 1.)-real_k
        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
        x_data_grid, y_data_grid = x_grid[k-w:k+w+1,:], y_grid[k-w:k+w+1,:]
        r_data_grid = np.sqrt(x_data_grid**2+y_data_grid**2)
        th_data_grid = np.arctan2( y_data_grid,x_data_grid) % (2.0 * np.pi)

        planet_model = np.zeros(data.shape)
        pl_x_vec = x_data_grid[0,:]-sep_planet/0.0203
        pl_y_vec = y_data_grid[:,0]
        for z in range(data_nz):
            planet_model[z,:,:] = psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()

        tlc_spec = np.nansum(planet_model,axis=(1,2))
        planet_model = planet_model*norma_planet_spec[:,None,None]

        planet_footprint = np.abs(np.nansum(planet_model/np.nanmax(planet_model,axis=(1,2))[:,None,None],axis=0))
        planet_footprint = planet_footprint/np.nanmax(planet_footprint)

        wv_ref = wvs[0]
        speckle_model = np.zeros((data_ny,data_nx,data_nz,data_ny,data_nx))
        footprint_normalization = np.nanmax(psfs_tlc,axis=(1,2))
        footprint_overlap = np.zeros((data_ny,data_nx))
        for sp_k in range(data_ny):
            for sp_l in range(data_nx):
                sp_r,sp_th = r_data_grid[sp_k,sp_l],th_data_grid[sp_k,sp_l]
                for z in range(data_nz):
                    sp_x = sp_r*wvs[z]/wv_ref*np.cos(sp_th)
                    sp_y = sp_r*wvs[z]/wv_ref*np.sin(sp_th)
                    sp_x_vec = x_data_grid[0,:]-sp_x
                    sp_y_vec = y_data_grid[:,0]-sp_y
                    speckle_model[sp_k,sp_l,z,:,:] = psfs_func_list[z](sp_x_vec,sp_y_vec).transpose()
                speckle_footprint = np.abs(np.nansum(speckle_model[sp_k,sp_l,:,:,:]/footprint_normalization[:,None,None],axis=0))
                speckle_footprint = speckle_footprint/np.nanmax(speckle_footprint)

                footprint_overlap[sp_k,sp_l] = np.sum(planet_footprint*speckle_footprint)

        footprint_overlap_ravel = footprint_overlap.ravel()
        # relevant_speckles = np.where(footprint_overlap_ravel>0.5)[0]
        relevant_speckles = np.where(footprint_overlap_ravel>-1)[0]

        bkg_model = np.zeros((data_ny,data_nx,data_nz,data_ny,data_nx))
        for bkg_k in range(data_ny):
            for bkg_l in range(data_nx):
                bkg_model[bkg_k,bkg_l,:,bkg_k,bkg_l] = 1

        data = np.ravel(data)
        if 1: # linear
            modela = np.reshape(speckle_model,(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            modelb = np.reshape(speckle_model*wvs[None,None,:,None,None],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            model = np.concatenate([np.ravel(planet_model)[:,None],modela,modelb],axis=1)
            relevant_para = np.concatenate([[0],relevant_speckles+1,relevant_speckles+data_ny*data_nx+1])
        if 0: # 3rd order
            modela = np.reshape(speckle_model,(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            modelb = np.reshape(speckle_model*wvs[None,None,:,None,None],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            modelc = np.reshape(speckle_model*(wvs**2)[None,None,:,None,None],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            modeld = np.reshape(speckle_model*(wvs**3)[None,None,:,None,None],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            model = np.concatenate([np.ravel(planet_model)[:,None],
                                    modela,
                                    modelb,
                                    modelc,
                                    modeld],axis=1)
            relevant_para = np.concatenate([[0],
                                            relevant_speckles+1,
                                            relevant_speckles+data_ny*data_nx+1,
                                            relevant_speckles+2*data_ny*data_nx+1,
                                            relevant_speckles+3*data_ny*data_nx+1])
        if 0:
            pass
            modela = np.reshape(speckle_model,(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            modelb = np.reshape(speckle_model*wvs[None,None,:,None,None],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            modelbkga = np.reshape(bkg_model,(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            modelbkgb = np.reshape(bkg_model*wvs[None,None,:,None,None],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
            model = np.concatenate([np.ravel(planet_model)[:,None],modela,modelb,modelbkga,modelbkgb],axis=1)
            relevant_para = np.concatenate([[0],
                                            relevant_speckles+1,
                                            relevant_speckles+data_ny*data_nx+1,
                                            relevant_speckles+2*data_ny*data_nx+1,
                                            relevant_speckles+3*data_ny*data_nx+1])
        where_finite_data = np.where(np.isfinite(data))

        data = data[where_finite_data]
        model = model[where_finite_data[0],:]
        model = model[:,relevant_para]

        model_H0 = model[:,1::]

        paras,residuals,rank,s = np.linalg.lstsq(model,data)
        paras_H0,residuals_H0,rank_H0,s_H0 = np.linalg.lstsq(model_H0,data)


        canvas_data = np.zeros((data_nz,data_ny,data_nx)) + np.nan
        canvas_data.shape = (data_nz*data_ny*data_nx,)
        canvas_data[where_finite_data] = data
        canvas_data.shape = (data_nz,data_ny,data_nx)

        canvas_model = np.zeros((data_nz,data_ny,data_nx)) + np.nan
        modeled_data = np.dot(model,paras)
        canvas_model.shape = (data_nz*data_ny*data_nx,)
        canvas_model[where_finite_data] = modeled_data
        canvas_model.shape = (data_nz,data_ny,data_nx)


        canvas_model_H0 = np.zeros((data_nz,data_ny,data_nx)) + np.nan
        modeled_data_H0 = np.dot(model_H0,paras_H0)
        canvas_model_H0.shape = (data_nz*data_ny*data_nx,)
        canvas_model_H0[where_finite_data] = modeled_data_H0
        canvas_model_H0.shape = (data_nz,data_ny,data_nx)

        canvas_model_planet = planet_model*paras[0]

        # canvas_data[1220:1240,:,:] = np.nan
        res = canvas_data[:,:,l-w:l+w+1]-canvas_model[:,:,l-w:l+w+1]
        res_H0 = canvas_data[:,:,l-w:l+w+1]-canvas_model_H0[:,:,l-w:l+w+1]
        # res = canvas_data-canvas_model
        # res_H0 = canvas_data-canvas_model_H0
        output_maps_np[0,row,col] = paras[0]
        output_maps_np[1,row,col] = np.nansum(res**2)
        output_maps_np[2,row,col] = np.nansum(res_H0**2)
        output_maps_np[3,row,col] = (output_maps_np[2,row,col]-output_maps_np[1,row,col])/np.nanvar(res)
        output_maps_np[4,row,col] = np.sign(paras[0])*np.sqrt(np.abs(output_maps_np[2,row,col]-output_maps_np[1,row,col])/np.nanvar(res))
        # print("output_maps_np[:,row,col]",output_maps_np[:,row,col])
        Npixproc+=1
        print(Npixproc,"done")

        if 1: # plot
            res = canvas_data-canvas_model
            res_H0 = canvas_data-canvas_model_H0
            norm_model = planet_model/np.nanmax(planet_model,axis=(1,2))[:,None,None]
            # norm_model = norm_model/np.nansum(planet_model**2,axis=(1,2))[:,None,None]
            print(canvas_data.shape)
            # myspeckle = speckle_model[1,l-2,:,:,:]
            myspeckle = speckle_model[3,10,:,:,:]
            mybkg = bkg_model[1,l-2,:,:,:]

            # plt.figure(20)
            # plt.plot(np.nansum(canvas_data**2,axis=(1,2)),label="data")
            # plt.plot(np.nansum(res**2,axis=(1,2)),label="res")
            # plt.plot(np.nansum(res_H0**2,axis=(1,2)),label="res_H0")
            # plt.legend()
            #
            # plt.figure(21)
            # plt.plot(np.nansum(canvas_data*norm_model,axis=(1,2)),label="data")
            # plt.plot(np.nansum(res*norm_model,axis=(1,2)),label="res")
            # plt.plot(np.nansum(res_H0*norm_model,axis=(1,2)),label="res_H0")
            # plt.legend()

            plt.figure(1)
            plt.plot(canvas_data[:,w,l],label="data")
            plt.plot(canvas_model_planet[:,w,l],label="model")
            plt.plot(res[:,w,l],label="res")
            plt.plot(res_H0[:,w,l],label="res_H0")
            plt.plot(star_spec/np.nanmean(star_spec)*np.nanmean(canvas_data[:,w,l]),linestyle="--",label="star")
            plt.plot(planet_spec/np.nanmean(planet_spec)*np.nanmean(canvas_data[:,w,l]),linestyle="--",label="planet")
            plt.plot(tlc_spec/np.nanmean(tlc_spec)*np.nanmean(canvas_data[:,w,l]),linestyle="--",label="tlc")
            plt.legend()

            plt.figure(2)
            norm_myspeckle = myspeckle/np.nanmax(myspeckle,axis=(1,2))[:,None,None]
            dot_data = np.nansum(norm_myspeckle*canvas_data,axis=(1,2))
            dot_model = np.nansum(norm_myspeckle*planet_model,axis=(1,2))
            dot_res = np.nansum(norm_myspeckle*res,axis=(1,2))
            dot_res_H0 = np.nansum(norm_myspeckle*res_H0,axis=(1,2))
            plt.plot(dot_data,label="dot_data")
            plt.plot(dot_model/np.nanmean(dot_model)*np.nanmean(dot_data),label="dot_model")
            plt.plot(dot_res,label="dot_res")
            plt.plot(dot_res_H0,label="dot_res_H0")
            plt.plot(tlc_spec/np.nanmean(tlc_spec)*np.nanmean(dot_data),linestyle="--",label="tlc")
            plt.legend()


            # plt.figure(31,figsize=(4,10))
            # for z in range(20):
            #     plt.subplot(20,1,z+1)
            #     plt.imshow(myspeckle[z*6,:,:],interpolation="nearest")
            # plt.figure(32,figsize=(4,10))
            # for z in range(20):
            #     plt.subplot(20,1,z+1)
            #     plt.imshow(planet_model[z*6,:,:],interpolation="nearest")
            # plt.figure(33,figsize=(4,10))
            # for z in range(20):
            #     plt.subplot(20,1,z+1)
            #     plt.imshow(mybkg[z*6,:,:],interpolation="nearest")

            tmp = "nomask_all"
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=canvas_data))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","canvas_data_{0}.fits".format(tmp))), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","canvas_data__{0}.fits".format(tmp))), clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=canvas_model_planet))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","canvas_model_planet_{0}.fits".format(tmp))), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","canvas_model_planet_{0}.fits".format(tmp))), clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=canvas_model))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","canvas_model_{0}.fits".format(tmp))), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","canvas_model_{0}.fits".format(tmp))), clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=res))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_res_{0}.fits".format(tmp))), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_res_{0}.fits".format(tmp))), clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=res_H0))
            try:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_res_H0_{0}.fits".format(tmp))), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_res_H0_{0}.fits".format(tmp))), clobber=True)
            hdulist.close()

            plt.show()


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
        inputDir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/"
        outputdir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/20181120_out/"
        filelist = glob.glob(os.path.join(inputDir,"s100715*20.fits"))
        filelist.sort()
        filename = filelist[0]
        psfs_tlc_filename = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_telluric_JB/HD_210501/s100715_a005002_Kbb_020_psfs.fits"
        template_spec="/home/sda/jruffio/osiris_data/HR_8799_c/hr8799c_osiris_template.save"
        star_spec = "/home/sda/jruffio/osiris_data/HR_8799_c/hr_8799_c_pickles_spectrum_Kbb.csv"
        # planet_coords = [[11,32],[12,27],[12,33],[12,39],[10,33],[9,28],[8,38],[10,32.5],[9,32],[10,33],[10,35],[10,33], #in order: image 10 to 21
        #                  [7,34],[5,35],[8,35],[7.5,33],[9.5,34.5]]
        # file_centers = [[x-sep_planet/ 0.0203,y] for x,y in planet_coords]
        numthreads = 32
        centermode = "visu" #ADI #def
        fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos_jb.xml"

        padding = 5
        planet_search = False
        debug = True
        real_k,real_l = 32+padding,-35.79802955665025+46.8+padding
        # real_k,real_l = 32+padding-10,-35.79802955665025+46.8+padding
        # real_k,real_l = 51,17
        # real_k,real_l = 39,16
        # real_k,real_l = 50,16
        #for astro
        # real_l_grid,real_k_grid = np.meshgrid(np.linspace(-3,3,7),np.linspace(-3,3,7))
        real_l_grid,real_k_grid = np.array([[0]]),np.array([[0]])
    else:
        inputDir = sys.argv[1]
        outputdir = sys.argv[2]
        filename = sys.argv[3]
        psfs_tlc_filename = sys.argv[4]
        template_spec = sys.argv[5]
        star_spec  = sys.argv[6]
        numthreads = int(sys.argv[7])
        centermode = sys.argv[8]
        fileinfos_filename = "/home/users/jruffio/OSIRIS/osirisextract/fileinfos_jb.xml"

    tree = ET.parse(fileinfos_filename)
    root = tree.getroot()

    filebasename = os.path.basename(filename)
    planet_c = root.find("c")
    fileelement = planet_c.find(filebasename)
    center = [float(fileelement.attrib["x"+centermode+"cen"]),float(fileelement.attrib["y"+centermode+"cen"])]
    sep_planet = float(fileelement.attrib["sep"])
    # suffix = "polyfit_"+centermode+"cen"+"_testmaskbadpix"
    # suffix = "polyfit_"+centermode+"cen"+"_resmask_maskbadpix"
    suffix = "polyfit_"+centermode+"cen"+"_resmask_norma"
    # suffix = "polyfit_"+centermode+"cen"+"_resmask_norma_bkg"

    if not os.path.exists(os.path.join(outputdir)):
        os.makedirs(os.path.join(outputdir))

    dtype = ctypes.c_float
    nan_mask_boxsize=3

    with pyfits.open(filename) as hdulist:
        imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        prihdr = hdulist[0].header
    imgs[np.where(imgs==0)] = np.nan
    nz,ny,nx = imgs.shape
    init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
    dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
    wvs=np.arange(init_wv,init_wv+dwv*nz,dwv)


    with pyfits.open(psfs_tlc_filename) as hdulist:
        psfs_tlc = hdulist[0].data
        psfs_tlc_prihdr = hdulist[0].header
    print(psfs_tlc.shape)


    travis_spectrum = scio.readsav(template_spec)
    planet_spec = np.array(travis_spectrum["fk_bin"])
    planet_spec = planet_spec/np.mean(planet_spec)

    with open(star_spec, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ')
        list_starspec = list(csv_reader)
        starspec_str_arr = np.array(list_starspec, dtype=np.str)
        col_names = starspec_str_arr[0]
        star_spec = starspec_str_arr[1::,1].astype(np.float)
        star_spec = star_spec/np.mean(star_spec)
        star_spec_wvs = starspec_str_arr[1::,0].astype(np.float)

    
    # reduce dimensionality
    if 0:
        psfs_tlc = psfs_tlc[::10,:,:]
        wvs = wvs[::10]
        imgs = imgs[::10,:,:]
        planet_spec = planet_spec[::10]
        star_spec = star_spec[::10]
        nz,ny,nx = imgs.shape


    padimgs = np.pad(imgs,((0,0),(padding,padding),(padding,padding)),mode="constant",constant_values=np.nan)
    padnz,padny,padnx = padimgs.shape
    print(imgs.shape)


    original_imgs = mp.Array(dtype, np.size(padimgs))
    original_imgs_shape = padimgs.shape
    original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
    original_imgs_np[:] = padimgs
    if planet_search:
        output_maps = mp.Array(dtype, 5*padny*padnx)
        output_maps_shape = (5,padny,padnx)
    else:
        output_maps_shape = (5,real_l_grid.shape[0],real_l_grid.shape[1])
        output_maps = mp.Array(dtype, 5*real_l_grid.shape[0]*real_l_grid.shape[1])
    output_maps_np = _arraytonumpy(output_maps,output_maps_shape,dtype=dtype)
    output_maps_np[:] = np.nan
    wvs_imgs = wvs
    psfs_stamps = mp.Array(dtype, np.size(psfs_tlc))
    psfs_stamps_shape = psfs_tlc.shape
    psfs_stamps_np = _arraytonumpy(psfs_stamps, psfs_stamps_shape,dtype=dtype)
    psfs_stamps_np[:] = psfs_tlc


    ######################
    # INIT threads and shared memory
    tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                    initargs=(original_imgs, original_imgs_shape, output_maps,
                              output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape),
                    maxtasksperchild=50)
    ######################
    # CLEAN IMAGE
    tasks = [tpool.apply_async(_remove_bad_pixels, args=(col_index, dtype))
             for col_index in range(nx)]

    #save it to shared memory
    for row_index, bad_pix_task in enumerate(tasks):
        print("Finished row {0}".format(row_index))
        bad_pix_task.wait()

    chunk_size = padnz//numthreads
    N_chunks = padnz//chunk_size
    wvs_indices_list = []
    for k in range(N_chunks-1):
        wvs_indices_list.append(np.arange((k*chunk_size),((k+1)*chunk_size)))
    wvs_indices_list.append(np.arange(((N_chunks-1)*chunk_size),padnz))

    tasks = [tpool.apply_async(_remove_edges, args=(wvs_indices,nan_mask_boxsize,dtype))
             for wvs_indices in wvs_indices_list]

    #save it to shared memory
    for chunk_index, rmedge_task in enumerate(tasks):
        print("Finished rm edge chunk {0}".format(chunk_index))
        rmedge_task.wait()

    ######################


    nz_psf,ny_psf,nx_psf = psfs_tlc.shape
    x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
    x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
    psfs_func_list = []
    psfs_tlc[np.where(np.isnan(psfs_tlc))] = 0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for wv_index in range(nz_psf):
            print(wv_index)
            model_psf = psfs_tlc[wv_index, :, :]
            psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
            psfs_func_list.append(psf_func)

    # 1: planet search, 0: astrometry
    if planet_search:
        wherenotnans = np.where(np.nansum(original_imgs_np,axis=0)!=0)
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

    # row_indices_list = [[32+5],[32+5],[32+5],[31+5],[31+5],[31+5],[33+5],[33+5],[33+5],[32+15],[32+15],[32+15],[31+15],[31+15],[31+15],[33+15],[33+15],[33+15]]
    # col_indices_list = [[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5]]

    if debug:
        _tpool_init(original_imgs, original_imgs_shape, output_maps,
                                  output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape)
        _process_pixels(real_k_valid_pix,real_l_valid_pix,row_valid_pix,col_valid_pix,psfs_func_list,star_spec,planet_spec, dtype,sep_planet)
    else:
        chunk_size = N_valid_pix//numthreads
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

        tasks = [tpool.apply_async(_process_pixels, args=(real_k_indices,real_l_indices,row_indices,col_indices,psfs_func_list,star_spec,planet_spec, dtype,sep_planet))
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