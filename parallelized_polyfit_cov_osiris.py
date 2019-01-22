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


def _remove_bad_pixels_z(col_index,dtype):
    global original, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    tmpcube = copy(original_np[:,col_index,:])
    nan_mask_boxsize = 3
    for m in np.arange(0,original_shape[0]):
        myvec = tmpcube[m,:]
        wherefinite = np.where(np.isfinite(myvec))
        if np.size(wherefinite[0])<10:
            continue
        smooth_vec = median_filter(myvec,footprint=np.ones(100),mode="constant",cval=0.0)
        myvec = myvec - smooth_vec
        wherefinite = np.where(np.isfinite(myvec))
        mad = mad_std(myvec[wherefinite])
        original_np[m,col_index,np.where(np.abs(myvec)>7*mad)[0]] = np.nan
        widen_nans = np.where(np.isnan(np.correlate(original_np[m,col_index,:],np.ones(nan_mask_boxsize),mode="same")))[0]
        original_np[m,col_index,widen_nans] = np.nan


def _remove_edges(wvs_indices,nan_mask_boxsize,dtype):
    global original, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    # print("coucou")
    for k in wvs_indices:
        tmp = original_np[:,:,k]
        tmp[np.where(np.isnan(correlate2d(tmp,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
    original_np[0:nan_mask_boxsize//2,:,:] = np.nan
    original_np[-nan_mask_boxsize//2+1::,:,:] = np.nan
    original_np[:,0:nan_mask_boxsize//2,:] = np.nan
    original_np[:,-nan_mask_boxsize//2+1::,:] = np.nan


def _remove_bad_pixels_xy(wvs_indices,dtype):
    global original, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    for k in wvs_indices:
        tmp = original_np[:,:,k]
        tmpcopy= copy(original_np[:,:,k])
        wherefinite = np.where(np.isfinite(tmpcopy))
        wherenans = np.where(np.isnan(tmpcopy))
        if np.size(wherefinite[0])<10:
            continue
        tmpcopy[wherenans] = 0
        smooth_map = median_filter(tmpcopy,footprint=np.ones((5,5)),mode="constant",cval=0.0)#medfilt2d(tmpcopy,5)
        tmpcopy[wherenans] = np.nan
        tmpcopy = tmpcopy - smooth_map
        mad = mad_std(tmpcopy[wherefinite])
        tmp[np.where(np.abs(tmpcopy)>7*mad)] = np.nan
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
def costfunc_unknown_cov(paras,model_blocks,data_blocks,sqdwvsmatrix_blocks):
    # t1 = time.time()

    # q = paras[0]
    # corrlen = paras[1]
    q = 0.1
    corrlen=10

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
    minus2logL = np.sum(sig_blocks_logdets)+np.size(data)*np.log(chi2)+1

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

def _process_pixels(real_k_indices,real_l_indices,row_indices,col_indices,psfs_func_list,star_spec,planet_spec, dtype,sep_planet):
    global original, original_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
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

            model = model[:,relevant_para]
            model = np.reshape(model,(data_ny,data_nx,data_nz,np.size(relevant_para)))

            sqdwvsmatrix_blocks = []
            data_blocks = []
            model_blocks = []
            model_H0_blocks = []
            for bkg_k in range(data_ny):
                for bkg_l in range(data_nx):
                    mydatablock = data[bkg_k,bkg_l,:]
                    where_finite_block = np.where(np.isfinite(mydatablock))
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

            data = np.ravel(data)
            model = np.reshape(model,(data_ny*data_nx*data_nz,np.size(relevant_para)))
            where_finite_data = np.where(np.isfinite(data))
            data = data[where_finite_data]
            model = model[where_finite_data[0],:]

            # print(model.shape)
            # iniparas,residuals,rank,s = np.linalg.lstsq(model,data)

            # x =q
            minus2logL,paras = costfunc_unknown_cov([0.5,100],model_blocks,data_blocks,sqdwvsmatrix_blocks)
            AIC = 2*(model.shape[1])+minus2logL
            minus2logL_H0,paras_H0 = costfunc_unknown_cov([0.5,10],model_H0_blocks,data_blocks,sqdwvsmatrix_blocks)
            AIC_H0 = 2*(model.shape[1]-1)+minus2logL_H0
            # print(row,col,paras[0],polydeg,AIC,AIC_H0,AIC-AIC_H0)
            output_maps_np[0,row,col] = paras[0]
            output_maps_np[1,row,col] = minus2logL
            output_maps_np[2,row,col] = minus2logL_H0
            output_maps_np[3,row,col] = AIC-AIC_H0
            output_maps_np[4,row,col] = 1
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
        # fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos_jb.xml"

        padding = 5
        planet_search = True
        debug = False
        real_k,real_l = 32+padding,-35.79802955665025+46.8+padding
        # real_k,real_l = 13,13
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
        # fileinfos_filename = "/home/users/jruffio/OSIRIS/osirisextract/fileinfos_jb.xml"

    # tree = ET.parse(fileinfos_filename)
    # root = tree.getroot()

    filebasename = os.path.basename(filename)
    # planet_c = root.find("c")
    fileelement = planet_c.find(filebasename)
    center = [float(fileelement.attrib["x"+centermode+"cen"]),float(fileelement.attrib["y"+centermode+"cen"])]
    sep_planet = float(fileelement.attrib["sep"])
    # suffix = "polyfit_"+centermode+"cen"+"_testmaskbadpix"
    # suffix = "polyfit_"+centermode+"cen"+"_resmask_maskbadpix"
    # suffix = "polyfit_"+centermode+"cen"+"_resmask_norma"
    # suffix = "polyfit_"+centermode+"cen"+"_resmask_norma_bkg"
    suffix = "polyfit_"+centermode+"cen"+"_cov_all"

    if not os.path.exists(os.path.join(outputdir)):
        os.makedirs(os.path.join(outputdir))

    dtype = ctypes.c_float
    nan_mask_boxsize=3

    with pyfits.open(filename) as hdulist:
        imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        prihdr = hdulist[0].header
        imgs = np.moveaxis(imgs,0,2)
    imgs[np.where(imgs==0)] = np.nan
    ny,nx,nz = imgs.shape
    init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
    dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
    wvs=np.arange(init_wv,init_wv+dwv*nz,dwv)


    with pyfits.open(psfs_tlc_filename) as hdulist:
        psfs_tlc = hdulist[0].data
        psfs_tlc_prihdr = hdulist[0].header
        psfs_tlc = np.moveaxis(psfs_tlc,0,2)
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
        psfs_tlc = psfs_tlc[:,:,::10]
        wvs = wvs[::10]
        imgs = imgs[:,:,::10]
        planet_spec = planet_spec[::10]
        star_spec = star_spec[::10]
        nz,ny,nx = imgs.shape


    padimgs = np.pad(imgs,((padding,padding),(padding,padding),(0,0)),mode="constant",constant_values=np.nan)
    padny,padnx,padnz = padimgs.shape
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
    tasks = [tpool.apply_async(_remove_bad_pixels_z, args=(col_index, dtype))
             for col_index in range(padnx)]
    #save it to shared memory
    for row_index, bad_pix_task in enumerate(tasks):
        print("Finished row {0}".format(row_index))
        bad_pix_task.wait()

    chunk_size = padnz//(3*numthreads)
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

    tasks = [tpool.apply_async(_remove_bad_pixels_xy, args=(wvs_indices,dtype))
             for wvs_indices in wvs_indices_list]
    #save it to shared memory
    for chunk_index, rmedge_task in enumerate(tasks):
        print("Finished rm bad pixel xy chunk {0}".format(chunk_index))
        rmedge_task.wait()
    ######################

    ny_psf,nx_psf,nz_psf = psfs_tlc.shape
    x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
    x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
    psfs_func_list = []
    psfs_tlc[np.where(np.isnan(psfs_tlc))] = 0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for wv_index in range(nz_psf):
            print(wv_index)
            model_psf = psfs_tlc[:, :, wv_index]
            psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
            psfs_func_list.append(psf_func)

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

    # row_indices_list = [[32+5],[32+5],[32+5],[31+5],[31+5],[31+5],[33+5],[33+5],[33+5],[32+15],[32+15],[32+15],[31+15],[31+15],[31+15],[33+15],[33+15],[33+15]]
    # col_indices_list = [[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5]]

    if debug:
        _tpool_init(original_imgs, original_imgs_shape, output_maps,
                                  output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape)
        _process_pixels(real_k_valid_pix,real_l_valid_pix,row_valid_pix,col_valid_pix,psfs_func_list,star_spec,planet_spec, dtype,sep_planet)
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

        if 0:
            mask = np.zeros((padny,padnx))
            _tpool_init(original_imgs, original_imgs_shape, output_maps,
                                      output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape)
            for real_k_indices, real_l_indices, row_indices, col_indices in zip(real_k_indices_list,real_l_indices_list, row_indices_list, col_indices_list):
                _process_pixels(real_k_valid_pix,real_l_valid_pix,row_valid_pix,col_valid_pix,psfs_func_list,star_spec,planet_spec, dtype,sep_planet)
                for row,col in zip(row_indices, col_indices):
                    print(row,col)
                    mask[row,col] = 1
            plt.imshow(mask,interpolation="nearest")
            plt.show()

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