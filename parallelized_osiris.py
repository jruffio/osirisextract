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

def _tpool_init(original_imgs, original_imgs_shape, scaled_imgs, scaled_imgs_shape, output_maps, output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Args:
    """
    global original, original_shape, scaled, scaled_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    # original images from files to read and align&scale. Shape of (N,y,x)
    original = original_imgs
    original_shape = original_imgs_shape
    # aligned and scaled images for processing. Shape of (wv, N, y, x)
    scaled = scaled_imgs
    scaled_shape = scaled_imgs_shape
    # output images after KLIP processing (amplitude and khi^2) (2, y, x)
    output = output_maps
    output_shape = output_maps_shape
    # parameters for each image (PA, wavelegnth, image center, image number)
    lambdas = wvs_imgs
    psfs = psfs_stamps
    psfs_shape = psfs_stamps_shape
    Npixproc= 0
    Npixtot=0


def _remove_bad_pixels(col_index,dtype):
    global original, original_shape, scaled, scaled_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    tmpcube = copy(original_np[:,:,col_index])
    nan_mask_boxsize = 3
    x = np.arange(nl)
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
    global original, original_shape, scaled, scaled_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)

    for k in wvs_indices:
        original_np[k][np.where(np.isnan(correlate2d(original_np[k],np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
    original_np[:,0:nan_mask_boxsize//2,:] = np.nan
    original_np[:,-nan_mask_boxsize//2+1::,:] = np.nan
    original_np[:,:,0:nan_mask_boxsize//2] = np.nan
    original_np[:,:,-nan_mask_boxsize//2+1::] = np.nan

def _scale_star(wvs_ref_indices, wvs, padcenter, dtype):
    global original, original_shape, scaled, scaled_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    scaled_imgs_np = _arraytonumpy(scaled_imgs, scaled_shape,dtype=dtype)
    for wvs_ref_ind in wvs_ref_indices:
        ref_wv = wvs[wvs_ref_ind]
        for k in range(np.size(wvs)):
            scaled_imgs_np[wvs_ref_ind,k,:,:] = klip.align_and_scale(original_np[k,:,:],padcenter,old_center=padcenter,scale_factor=ref_wv/wvs[k])
    return output

def costfunc(amplitude,data,model,forjac=False):
    nl,nl,ny,nx = data.shape
    res = data - amplitude*model
    x = np.arange(nl)

    for m in np.arange(0,ny):
        for n in np.arange(0,nx):
            for k in np.arange(0,nl):
                myvec = res[k,:,m,n]
                wherefinite = np.where(np.isfinite(myvec))
                if np.size(wherefinite[0]) <= 10:
                    res[k,wherefinite[0],m,n] = np.nan
                    # print("skip")
                    continue
                poly_coefs = np.polyfit(x[wherefinite],myvec[wherefinite],1)# polynomial for speckles
                res[k,wherefinite[0],m,n] -= np.polyval(poly_coefs, x[wherefinite])

    localres = res[np.arange(nl),np.arange(nl),:,:]
    # cost = np.nanvar(localres)
    # cost = np.nansum(localres**2)
    cost = np.nansum(localres**2)/(np.sum(np.isfinite(localres))-1)
    # if not forjac:
    #     print(amplitude,cost)
    return cost

def jaccostfunc(amplitude,data,model):
    eps = 0.000001
    a1 = costfunc(amplitude-eps,data,model,forjac=True)
    a2 = costfunc(amplitude+eps,data,model,forjac=True)
    # print("jac",(a2-a1)/(2*eps))
    return (a2-a1)/(2*eps)

def _process_pixels(row_indices,col_indices,psfs_func_list, padcenter,hr8799_spec,hr8799c_spec, dtype):#,scaled_imgs_np,output_maps_np):
    global original, original_shape, scaled, scaled_shape, output, output_shape, lambdas, img_center, psfs, psfs_shape, Npixproc, Npixtot
    # original_np = _arraytonumpy(original, original_shape,dtype=dtype)
    scaled_imgs_np = _arraytonumpy(scaled_imgs, scaled_shape,dtype=dtype)
    output_maps_np = _arraytonumpy(output_maps, output_maps_shape,dtype=dtype)
    nl,ny,nx = original_shape
    # nl,ny,nx=1665,74,29

    x_vec,y_vec = np.arange(nx * 1.)-padcenter[0],np.arange(ny* 1.)-padcenter[1]
    x_grid, y_grid = np.meshgrid(x_vec,y_vec)
    for m,n in zip(row_indices,col_indices):
        # print(m,n)
        # print(x_grid[m,n],y_grid[m,n]) #46.79802955665025 0.0
        template = np.zeros((nl,nl,3,3))
        for k,lambref in enumerate(wvs):
            # print(k)
            for l,lamb in enumerate(wvs):
                template[k,l,:,:] = psfs_func_list[k](x_vec[n-1:n+2]*(lamb/lambref)-x_grid[m,n], y_vec[m-1:m+2]*(lamb/lambref)-y_grid[m,n]).transpose()

        planet_norma = np.ones((nl,nl,3,3))*(hr8799c_spec/np.mean(hr8799c_spec))[None,:,None,None]/(hr8799_spec/np.mean(hr8799_spec))[None,:,None,None]
        template = template*planet_norma

        data = copy(scaled_imgs_np[:,:,m-1:m+2,n-1:n+2])
        data = data/(hr8799_spec/np.mean(hr8799_spec))[None,:,None,None]

        # tmp = np.zeros((nl,nl*2,3,3))+np.nan
        # tmp_planet = np.zeros((nl,nl*2,3,3))
        # for m in np.arange(0,3):
        #     for n in np.arange(0,3):
        #         print(n,m)
        #         for k in range(nl):
        #             tmp[k,(nl-k):(2*nl-k),m,n] = data[k,:,m,n]
        #             tmp_planet[k,(nl-k):(2*nl-k),m,n] = template[k,:,m,n]
        #
        # plt.figure(3)
        # for m in np.arange(0,3):
        #     for n in np.arange(0,3):
        #         plt.subplot(3,6,m*6+1+n*2)
        #         plt.imshow(tmp[:,:,m,n],interpolation="nearest")
        #         plt.subplot(3,6,m*6+2+n*2)
        #         plt.imshow(tmp_planet[:,:,1,1])
        # plt.show()

        res = minimize(costfunc,0.0,args=(data,template),method="Newton-CG",tol=1e-3,jac=jaccostfunc,options={"maxiter":1})
        # print(res.x)
        # print(res.fun)
        # print(res.jac)
        # print(res.x/np.sqrt(res.fun))
        # print(res.x/np.sqrt(res.fun/(np.sum(np.isfinite(data))-1)))
        output_maps_np[0,m,n] = res.x
        output_maps_np[1,m,n] = res.fun
        output_maps_np[2,m,n] = res.x[0]/res.fun
        Npixproc+=1
        print("done")



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
        outputdir = "/home/sda/Dropbox (GPI)/TEST_SCRATCH/scratch/JB/OSIRIS_utils/bruce_inspired_outputs/"
        inputDir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_quinn/"
        # inputDir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/"
        telluric_cube = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_telluric/HD_210501/s100715_a005001_Kbb_020.fits"
        filelist = glob.glob(os.path.join(inputDir,"s100715*20.fits"))
        filelist.sort()
        filename = filelist[1]
        sep_planet = 0.950
        # planet_coords = [[11,32],[12,27],[12,33],[12,39],[10,33],[9,28],[8,38],[10,32.5],[9,32],[10,33],[10,35],[10,33], #in order: image 10 to 21
        #                  [7,34],[5,35],[8,35],[7.5,33],[9.5,34.5]]
        # file_centers = [[x-sep_planet/ 0.0203,y] for x,y in planet_coords]
        numthreads = 32
    else:
        inputDir = sys.argv[1]
        outputdir = sys.argv[2]
        filename = sys.argv[3]
        telluric_cube = sys.argv[4]
        sep_planet = float(sys.argv[5])
        numthreads = int(sys.argv[6])

    if not os.path.exists(os.path.join(outputdir)):
        os.makedirs(os.path.join(outputdir))

    # #------------------------------------------------
    # im_index=1
    # with pyfits.open(os.path.join(outputdir,os.path.basename(filelist[im_index]).replace(".fits","_output.fits"))) as hdulist:
    #     output = hdulist[0].data
    #     prihdr = hdulist[0].header
    # finaloutput = output[0,:,:]/output[1,:,:]
    #
    # output2 = np.zeros((3,output.shape[1],output.shape[2]))
    # output2[0,:,:] = output[0,:,:]
    # output2[1,:,:] = output[1,:,:]
    # output2[2,:,:] = output[0,:,:]/output[1,:,:]
    # hdulist = pyfits.HDUList()
    # hdulist.append(pyfits.PrimaryHDU(data=output2))
    # try:
    #     hdulist.writeto(os.path.join(outputdir,os.path.basename(filelist[im_index]).replace(".fits","_output.fits")), overwrite=True)
    # except TypeError:
    #     hdulist.writeto(os.path.join(outputdir,os.path.basename(filelist[im_index]).replace(".fits","_output.fits")), clobber=True)
    # hdulist.close()
    # plt.imshow(finaloutput,interpolation="nearest")
    # plt.show()


    dtype = ctypes.c_float
    nan_mask_boxsize=3

    # im_index = 1

    with pyfits.open(filename) as hdulist:
        imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        prihdr = hdulist[0].header

    if 0:
        pass
        # center = file_centers[im_index]
        #Center= [-35.79802955665025, 32]
    else:
        suffix = "_centerADI"
        # center = [-32.40914067, 32.94444444]
        # print(filelist[im_index:(im_index+1)])
        # exit()
        dataset = osi.Ifs([filename],telluric_cube, #[filelist[0],filelist[7]] filelist[0:12]
                         guess_center=[19//2-sep_planet/ 0.0203,64//2],recalculate_center_cadi=True, centers = None,
                         psf_cube_size=21,
                         coaddslices=None, nan_mask_boxsize=0,median_filter_boxsize = 0,badpix2nan=False,ignore_PAs=True)

        center = dataset.centers[0]
        #Center= [-32.40914067  32.94444444]
    print("Center=",center)
    exit()

    nl,ny,nx = imgs.shape
    init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
    dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
    wvs=np.arange(init_wv,init_wv+dwv*nl,dwv)
    with pyfits.open(os.path.join(outputdir,os.path.basename(telluric_cube).replace(".fits","_medcombinedpsfs.fits"))) as hdulist:
        psfs = hdulist[0].data
        hdulist.close()

    travis_spectrum = scio.readsav("/data/Dropbox (GPI)/TEST_SCRATCH/scratch/JB/hr8799c_osiris_template.save")
    hr8799c_spec = np.array(travis_spectrum["fk_bin"])
    hr8799_spec = np.nanmean(imgs,axis=(1,2))


    padimgs = np.pad(imgs,((0,0),(5,5),(5,5)),mode="constant",constant_values=np.nan)
    padcenter = [center[0]+5, center[1]+5]
    padnl,padny,padnx = padimgs.shape

    # #-------------------------
    # canvas = np.zeros((padny,padnx))
    # wherenotnans = np.where(np.nansum(padimgs,axis=0)!=0)
    # N_valid_pix = np.size(wherenotnans[0])
    # print(N_valid_pix)
    # row_valid_pix = wherenotnans[0]
    # col_valid_pix = wherenotnans[1]
    # canvas = np.zeros((padny,padnx))
    #
    # chunk_size = N_valid_pix//numthreads
    # N_chunks = N_valid_pix//chunk_size
    # row_indices_list = []
    # col_indices_list = []
    # for k in range(N_chunks-1):
    #     row_indices_list.append(row_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
    #     col_indices_list.append(col_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
    #
    #
    # row_indices_list = [[32+5],[32+5],[32+5],[31+5],[31+5],[31+5],[33+5],[33+5],[33+5],[32+15],[32+15],[32+15],[31+15],[31+15],[31+15],[33+15],[33+15],[33+15]]
    # col_indices_list = [[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5]]
    # for ms,ns in zip(row_indices_list,col_indices_list):
    #     canvas[ms,ns] = 1
    # plt.imshow(canvas,interpolation="nearest")
    # plt.show()
    # row_indices_list.append(row_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])
    # col_indices_list.append(col_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])
    # # print(wherenotnans[1])
    # # canvas[wherenotnans] = 1
    # # plt.imshow(canvas,interpolation="nearest")
    # # plt.show()
    # #-------------------------

    # #####
    # with pyfits.open(os.path.join(inputDir,"4drescaled",os.path.basename(filelist[im_index]).replace(".fits","_4drescaled_pad_badpixv2.fits"))) as hdulist: #padv2
    #     tmp_scaled_imgs_np = hdulist[0].data
    #     hdulist.close()
    # padnl,padnl,padny,padnx = tmp_scaled_imgs_np.shape
    # print(padnl,padnl,padny,padnx)
    # output_maps_np = np.zeros((3,padny,padnx))
    # psfs_stamps_np = psfs
    # nl_psf,ny_psf,nx_psf = psfs_stamps_np.shape
    # x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
    # x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
    # psfs_func_list = []
    # psfs_stamps_np[np.where(np.isnan(psfs_stamps_np))] = 0
    # import warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     for wv_index in range(nl_psf):
    #         print(wv_index)
    #         model_psf = psfs_stamps_np[wv_index, :, :]
    #         psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
    #         psfs_func_list.append(psf_func)
    # global Npixproc, Npixtot
    # Npixproc= 0
    # Npixtot=0
    #
    # _process_pixels([32+5],[11+5],psfs_func_list, padcenter,hr8799_spec,hr8799c_spec, dtype,tmp_scaled_imgs_np,output_maps_np)
    # # _process_pixels([40+5],[11+5],psfs_func_list, padcenter,hr8799_spec,hr8799c_spec, dtype,tmp_scaled_imgs_np,output_maps_np)
    # exit()
    # ####


    original_imgs = mp.Array(dtype, np.size(padimgs))
    original_imgs_shape = padimgs.shape
    original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
    original_imgs_np[:] = padimgs
    scaled_imgs = mp.Array(dtype, np.size(padimgs)*padnl)
    scaled_imgs_shape =(padnl,padnl,padny,padnx)
    scaled_imgs_np = _arraytonumpy(scaled_imgs,scaled_imgs_shape,dtype=dtype)
    scaled_imgs_np[:] = np.nan
    output_maps = mp.Array(dtype, 3*padny*padnx)
    output_maps_shape = (3,padny,padnx)
    output_maps_np = _arraytonumpy(output_maps,output_maps_shape,dtype=dtype)
    output_maps_np[:] = np.nan
    wvs_imgs = wvs
    psfs_stamps = mp.Array(dtype, np.size(psfs))
    psfs_stamps_shape = psfs.shape
    psfs_stamps_np = _arraytonumpy(psfs_stamps, psfs_stamps_shape,dtype=dtype)
    psfs_stamps_np[:] = psfs

    # plt.plot(original_imgs_np[:,44,12],color="red")
    # print("coucou")

    # init threads and shared memory
    tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                    initargs=(original_imgs, original_imgs_shape, scaled_imgs, scaled_imgs_shape, output_maps,
                              output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape),
                    maxtasksperchild=50)

    original_imgs_np[np.where(original_imgs_np==0)] = np.nan

    # Clean
    tasks = [tpool.apply_async(_remove_bad_pixels, args=(col_index, dtype))
             for col_index in range(nx)]

    #save it to shared memory
    for row_index, bad_pix_task in enumerate(tasks):
        print("Finished row {0}".format(row_index))
        bad_pix_task.wait()

    chunk_size = nl//numthreads
    N_chunks = nl//chunk_size
    wvs_indices_list = []
    for k in range(N_chunks-1):
        wvs_indices_list.append(np.arange((k*chunk_size),((k+1)*chunk_size)))
    wvs_indices_list.append(np.arange(((N_chunks-1)*chunk_size),nl))

    tasks = [tpool.apply_async(_remove_edges, args=(wvs_indices,nan_mask_boxsize,dtype))
             for wvs_indices in wvs_indices_list]

    #save it to shared memory
    for chunk_index, rmedge_task in enumerate(tasks):
        print("Finished rm edge chunk {0}".format(chunk_index))
        rmedge_task.wait()

    # print(os.path.join(inputDir,"4drescaled",os.path.basename(filelist[im_index]).replace(".fits","_4drescaled_pad_badpixv2.fits")))
    # print(glob.glob(os.path.join(inputDir,"4drescaled",os.path.basename(filelist[im_index]).replace(".fits","_4drescaled_pad_badpixv2.fits"))))
    # print(len(glob.glob(os.path.join(inputDir,"4drescaled",os.path.basename(filelist[im_index]).replace(".fits","_4drescaled_pad_badpixv2.fits")))))
    # print(len(glob.glob(os.path.join(inputDir,"4drescaled",os.path.basename(filelist[im_index]).replace(".fits","_4drescaled_pad_badpixv2.fits")))) == 1)
    # exit()

    if len(glob.glob(os.path.join(inputDir,"4drescaled",os.path.basename(filename).replace(".fits","_4drescaled_pad_badpixv2.fits")))) == 1:
        print("coucou")
        with pyfits.open(os.path.join(inputDir,"4drescaled",os.path.basename(filename).replace(".fits","_4drescaled_pad_badpixv2.fits"))) as hdulist: #padv2
            scaled_imgs_np[:] = hdulist[0].data
            hdulist.close()
        print("Loaded 4drescaled file: "+os.path.join(inputDir,"4drescaled",os.path.basename(filename).replace(".fits","_4drescaled_pad_badpixv2.fits")))
    else:
        # print("No!!!")
        # exit()
        chunk_size = padnl//numthreads
        N_chunks = padnl//chunk_size
        wvs_ref_indices = []
        for k in range(N_chunks-1):
            wvs_ref_indices.append(np.arange((k*chunk_size),((k+1)*chunk_size)))
        wvs_ref_indices.append(np.arange(((N_chunks-1)*chunk_size),nl))

        tasks = [tpool.apply_async(_scale_star, args=(wvs_ref_ind, wvs, padcenter, dtype))
                 for wvs_ref_ind in wvs_ref_indices]

        #save it to shared memory
        for row_index, scale_task in enumerate(tasks):
            print("Finished scale chunk {0}".format(row_index))
            scale_task.wait()

        # hdulist = pyfits.HDUList()
        # hdulist.append(pyfits.PrimaryHDU(data=scaled_imgs_np))
        # try:
        #     hdulist.writeto(os.path.join(inputDir,"4drescaled",os.path.basename(filelist[im_index]).replace(".fits","_4drescaled_pad_badpixv2.fits")), overwrite=True)
        # except TypeError:
        #     hdulist.writeto(os.path.join(inputDir,"4drescaled",os.path.basename(filelist[im_index]).replace(".fits","_4drescaled_pad_badpixv2.fits")), clobber=True)
        # hdulist.close()


    nl_psf,ny_psf,nx_psf = psfs_stamps_np.shape
    x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
    x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
    psfs_func_list = []
    psfs_stamps_np[np.where(np.isnan(psfs_stamps_np))] = 0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for wv_index in range(nl_psf):
            print(wv_index)
            model_psf = psfs_stamps_np[wv_index, :, :]
            psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
            psfs_func_list.append(psf_func)

    canvas = np.zeros((padny,padnx))
    wherenotnans = np.where(np.nansum(original_imgs_np,axis=0)!=0)
    N_valid_pix = np.size(wherenotnans[0])
    Npixtot = N_valid_pix
    row_valid_pix = wherenotnans[0]
    col_valid_pix = wherenotnans[1]
    chunk_size = N_valid_pix//numthreads
    N_chunks = N_valid_pix//chunk_size
    row_indices_list = []
    col_indices_list = []
    for k in range(N_chunks-1):
        row_indices_list.append(row_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
        col_indices_list.append(col_valid_pix[(k*chunk_size):((k+1)*chunk_size)])
    row_indices_list.append(row_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])
    col_indices_list.append(col_valid_pix[((N_chunks-1)*chunk_size):N_valid_pix])

    # row_indices_list = [[32+5],[32+5],[32+5],[31+5],[31+5],[31+5],[33+5],[33+5],[33+5],[32+15],[32+15],[32+15],[31+15],[31+15],[31+15],[33+15],[33+15],[33+15]]
    # col_indices_list = [[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5],[10+5],[11+5],[12+5]]

    tasks = [tpool.apply_async(_process_pixels, args=(row_indices,col_indices,psfs_func_list, padcenter,hr8799_spec,hr8799c_spec, dtype))
             for row_indices,col_indices in zip(row_indices_list, col_indices_list)]

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