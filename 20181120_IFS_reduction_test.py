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
        outputdir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/20181120_out/"
        inputDir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/"
        # inputDir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/"
        psfs_tlc_filename = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_telluric_JB/HD_210501/s100715_a005002_Kbb_020_psfs.fits"
        template_spec="/home/sda/jruffio/osiris_data/HR_8799_c/hr8799c_osiris_template.save"
        star_spec = "/home/sda/jruffio/osiris_data/HR_8799_c/hr_8799_c_pickles_spectrum_Kbb.csv"
        filelist = glob.glob(os.path.join(inputDir,"s100715*20.fits"))
        filelist.sort()
        filename = filelist[0]
        sep_planet = 0.950
        # planet_coords = [[11,32],[12,27],[12,33],[12,39],[10,33],[9,28],[8,38],[10,32.5],[9,32],[10,33],[10,35],[10,33], #in order: image 10 to 21
        #                  [7,34],[5,35],[8,35],[7.5,33],[9.5,34.5]]
        # file_centers = [[x-sep_planet/ 0.0203,y] for x,y in planet_coords]
        numthreads = 32
        centermode = "visu" #ADI #def
        fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos.xml"
    else:
        pass
        # inputDir = sys.argv[1]
        # outputdir = sys.argv[2]
        # filename = sys.argv[3]
        # telluric_cube = sys.argv[4]
        # template_spec = sys.argv[5]
        # sep_planet = float(sys.argv[6])
        # numthreads = int(sys.argv[7])
        # centermode = sys.argv[8]
        # fileinfos_filename = "/home/users/jruffio/OSIRIS/osirisextract/fileinfos.xml"

    tree = ET.parse(fileinfos_filename)
    root = tree.getroot()

    filebasename = os.path.basename(filename)
    planet_c = root.find("c")
    fileelement = planet_c.find(filebasename)
    # center = [float(fileelement.attrib["x"+centermode+"cen"]),float(fileelement.attrib["y"+centermode+"cen"])]
    # print("Center=",center)
    suffix = "20181120test_"+centermode+"cen"

    if not os.path.exists(os.path.join(outputdir)):
        os.makedirs(os.path.join(outputdir))


    dtype = ctypes.c_float
    nan_mask_boxsize=3

    with pyfits.open(filename) as hdulist:
        imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        prihdr = hdulist[0].header
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

    norma_planet_spec = planet_spec/star_spec
    # print(star_spec_wvs.shape)
    # plt.plot(star_spec_wvs,star_spec/np.mean(star_spec))
    # plt.plot(star_spec_wvs,planet_spec/np.mean(planet_spec))
    # plt.show()


    # reduce dimensionality
    if 0:
        psfs_tlc = psfs_tlc[::100,:,:]
        wvs = wvs[::100]
        imgs = imgs[::100,:,:]
        nz,ny,nx = imgs.shape

    padding = 5
    padimgs = np.pad(imgs,((0,0),(padding,padding),(padding,padding)),mode="constant",constant_values=np.nan)
    padnz,padny,padnx = padimgs.shape
    print(imgs.shape)

    padimgs[np.where(padimgs==0)] = np.nan
    for k in range(padnz):
        padimgs[k][np.where(np.isnan(correlate2d(padimgs[k],np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
    padimgs[:,0:nan_mask_boxsize//2,:] = np.nan
    padimgs[:,-nan_mask_boxsize//2+1::,:] = np.nan
    padimgs[:,:,0:nan_mask_boxsize//2] = np.nan
    padimgs[:,:,-nan_mask_boxsize//2+1::] = np.nan


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

    # current pixel
    # <s100715_a010001_tlc_Kbb_020.fits sep="0.95" sequence="0" stardir="left" xADIcen="-32.40914067" xdefcen="-37.79802955665025" xvisucen="-35.79802955665025" yADIcen="32.94444444" ydefcen="32" yvisucen="32"/>
    real_k,real_l = 32+padding,-35.79802955665025+46.8+padding
    k,l = int(np.floor(real_k)),int(np.floor(real_l))
    print(k,l)

    if k<padding or l<padding or k>=padny-padding or l>=padny-padding:
        raise("Bad index")

    data = padimgs[:,k-1:k+2,:]
    data_nz,data_ny,data_nx = data.shape

    # plt.figure(1)
    # plt.imshow(imgs[0],interpolation="nearest")
    # plt.clim([0,0.5])
    #
    # plt.figure(2)
    # plt.imshow(data[0],interpolation="nearest")
    # plt.clim([0,0.5])
    # plt.show()

    # plt.figure(1,figsize=(2,10))
    # for l in range(data_nz):
    #     plt.subplot(data_nz,1,l+1)
    #     plt.imshow(data[l],interpolation="nearest")
    #     plt.clim([0,0.5])
    # plt.show()

    x_vec, y_vec = np.arange(padnx * 1.)-real_l+sep_planet/0.0203,np.arange(padny* 1.)-real_k
    x_grid, y_grid = np.meshgrid(x_vec, y_vec)
    x_data_grid, y_data_grid = x_grid[k-1:k+2,:], y_grid[k-1:k+2,:]
    r_data_grid = np.sqrt(x_data_grid**2+y_data_grid**2)
    th_data_grid = np.arctan2( y_data_grid,x_data_grid) % (2.0 * np.pi)
    # print(x_data_grid)
    # print(r_data_grid*np.cos(th_data_grid))
    # print(y_data_grid)
    # print(r_data_grid*np.sin(th_data_grid))
    # exit()



    planet_model = np.zeros(data.shape)
    pl_x_vec = x_data_grid[0,:]-sep_planet/0.0203
    pl_y_vec = y_data_grid[:,0]
    for z in range(data_nz):
        planet_model[z,:,:] = psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()*norma_planet_spec[z]

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
    relevant_speckles = np.where(footprint_overlap_ravel>0.5)[0]
    irrelevant_speckles = np.where(footprint_overlap_ravel<=0.5)[0]
    # plt.figure(1)
    # plt.subplot(1,3,1)
    # plt.imshow(planet_footprint,interpolation="nearest")
    # plt.subplot(1,3,2)
    # plt.imshow(speckle_footprint,interpolation="nearest")
    # plt.subplot(1,3,3)
    # plt.imshow(footprint_overlap,interpolation="nearest")
    # plt.show()

    # plt.figure(1,figsize=(4,10))
    # for z in range(data_nz):
    #     plt.subplot(data_nz,1,z+1)
    #     plt.imshow(speckle_model[1,15,z,:,:],interpolation="nearest")
    # plt.figure(2,figsize=(4,10))
    # for z in range(data_nz):
    #     plt.subplot(data_nz,1,z+1)
    #     plt.imshow(planet_model[z,:,:],interpolation="nearest")
    # plt.figure(3,figsize=(4,10))
    # for z in range(data_nz):
    #     plt.subplot(data_nz,1,z+1)
    #     plt.imshow(data[z,:,:],interpolation="nearest")
    #     # plt.clim([0,0.5])
    # plt.figure(4)
    # plt.imshow(padimgs[0,:,:],interpolation="nearest")
    # plt.show()

    data = np.ravel(data)
    modela = np.reshape(speckle_model,(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
    modelb = np.reshape(speckle_model*wvs[None,None,:,None,None],(data_ny*data_nx,data_nz*data_ny*data_nx)).transpose()
    model = np.concatenate([np.ravel(planet_model)[:,None],modela,modelb],axis=1)
    where_finite_data = np.where(np.isfinite(data))

    relevant_para = np.concatenate([[0],relevant_speckles+1,relevant_speckles+data_ny*data_nx+1])

    data = data[where_finite_data]
    model = model[where_finite_data[0],:]
    model = model[:,relevant_para]

    model_H0 = model[:,1::]

    print("before",model.shape,data.shape)
    paras,residuals,rank,s = np.linalg.lstsq(model,data)
    print("after",paras)
    paras_H0,residuals_H0,rank_H0,s_H0 = np.linalg.lstsq(model_H0,data)

    canvas_data = np.zeros((data_nz,data_ny,data_nx))
    canvas_data.shape = (data_nz*data_ny*data_nx,)
    canvas_data[where_finite_data] = data
    canvas_data.shape = (data_nz,data_ny,data_nx)

    canvas_model = np.zeros((data_nz,data_ny,data_nx))
    modeled_data = np.dot(model,paras)
    canvas_model.shape = (data_nz*data_ny*data_nx,)
    canvas_model[where_finite_data] = modeled_data
    canvas_model.shape = (data_nz,data_ny,data_nx)


    canvas_model_H0 = np.zeros((data_nz,data_ny,data_nx))
    modeled_data_H0 = np.dot(model_H0,paras_H0)
    canvas_model_H0.shape = (data_nz*data_ny*data_nx,)
    canvas_model_H0[where_finite_data] = modeled_data_H0
    canvas_model_H0.shape = (data_nz,data_ny,data_nx)

    canvas_model_planet = planet_model*paras[0]

    plt.figure(20)
    res = canvas_data-canvas_model
    res_H0 = canvas_data-canvas_model_H0
    plt.plot(np.nansum(canvas_data**2,axis=(1,2)),label="data")
    plt.plot(np.nansum(res**2,axis=(1,2)),label="res")
    plt.plot(np.nansum(res_H0**2,axis=(1,2)),label="res_H0")
    plt.legend()

    plt.figure(21)
    norm_model = canvas_model_planet/np.nanmax(canvas_model_planet,axis=(1,2))[:,None,None]
    # norm_model = norm_model/np.nansum(canvas_model_planet**2,axis=(1,2))[:,None,None]
    plt.plot(np.nansum(canvas_data*norm_model,axis=(1,2)),label="data")
    plt.plot(np.nansum(res*norm_model,axis=(1,2)),label="res")
    plt.plot(np.nansum(res_H0*norm_model,axis=(1,2)),label="res_H0")
    plt.legend()
    plt.show()

    # plt.figure(1,figsize=(4,10))
    # for z in range(20):
    #     plt.subplot(20,1,z+1)
    #     plt.imshow(canvas_data[z*50,:,:],interpolation="nearest")
    # plt.figure(2,figsize=(4,10))
    # for z in range(20):
    #     plt.subplot(20,1,z+1)
    #     plt.imshow(canvas_model[z*50,:,:],interpolation="nearest")
    # plt.figure(3,figsize=(4,10))
    # for z in range(20):
    #     plt.subplot(20,1,z+1)
    #     plt.imshow(canvas_data[z*50,:,:]-canvas_model[z*50,:,:],interpolation="nearest")
    # plt.figure(4,figsize=(4,10))
    # for z in range(20):
    #     plt.subplot(20,1,z+1)
    #     plt.imshow(canvas_model_planet[z*50,:,:],interpolation="nearest")
    # plt.figure(5,figsize=(4,10))
    # for z in range(20):
    #     plt.subplot(20,1,z+1)
    #     plt.imshow(canvas_data[z*50,:,:]-canvas_model[z*50,:,:],interpolation="nearest")
    #
    #
    # plt.figure(12,figsize=(4,10))
    # for z in range(20):
    #     plt.subplot(20,1,z+1)
    #     plt.imshow(canvas_model_H0[z*50,:,:],interpolation="nearest")
    # plt.figure(13,figsize=(4,10))
    # for z in range(20):
    #     plt.subplot(20,1,z+1)
    #     plt.imshow(canvas_data[z*50,:,:]-canvas_model_H0[z*50,:,:],interpolation="nearest")
    # plt.show()

    # plt.figure(1,figsize=(4,10))
    # for z in range(data_nz):
    #     plt.subplot(data_nz,1,z+1)
    #     plt.imshow(canvas_data[z,:,:],interpolation="nearest")
    # plt.figure(2,figsize=(4,10))
    # for z in range(data_nz):
    #     plt.subplot(data_nz,1,z+1)
    #     plt.imshow(canvas_model[z,:,:],interpolation="nearest")
    # plt.figure(3,figsize=(4,10))
    # for z in range(data_nz):
    #     plt.subplot(data_nz,1,z+1)
    #     plt.imshow(canvas_data[z,:,:]-canvas_model[z,:,:],interpolation="nearest")
    # plt.figure(4,figsize=(4,10))
    # for z in range(data_nz):
    #     plt.subplot(data_nz,1,z+1)
    #     plt.imshow(canvas_model_planet[z,:,:],interpolation="nearest")
    # plt.figure(5,figsize=(4,10))
    # for z in range(data_nz):
    #     plt.subplot(data_nz,1,z+1)
    #     plt.imshow(canvas_model[z,:,:]-canvas_model_planet[z,:,:],interpolation="nearest")
    #
    #
    # plt.figure(12,figsize=(4,10))
    # for z in range(data_nz):
    #     plt.subplot(data_nz,1,z+1)
    #     plt.imshow(canvas_model_H0[z,:,:],interpolation="nearest")
    # plt.figure(13,figsize=(4,10))
    # for z in range(data_nz):
    #     plt.subplot(data_nz,1,z+1)
    #     plt.imshow(canvas_data[z,:,:]-canvas_model_H0[z,:,:],interpolation="nearest")
    # plt.show()