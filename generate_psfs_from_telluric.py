__author__ = 'jruffio'


import os
import sys
import glob
import time
import astropy.io.fits as pyfits
import numpy as np
import itertools
import multiprocessing as mp
import pyklip.klip as klip
import matplotlib.pyplot as plt
from pyklip.fakes import gaussfit2d
from scipy import interpolate

def align_and_scale_star(params):
    ref_wvs = params[0]
    cube = params[1]
    nl,ny,nx = cube.shape
    wvs = params[2]
    center = params[3]
    try:
        centers = params[4]
    except:
        centers = [center,]*nl
    output = np.zeros((np.size(ref_wvs),nl,ny,nx))
    for l,ref_wv in enumerate(ref_wvs):
        for k in range(np.size(wvs)):
            output[l,k,:,:] = klip.align_and_scale(cube[k,:,:],center,old_center=centers[k],scale_factor=ref_wv/wvs[k])
    return output


# OSIRISDATA = "/scratch/groups/bmacint/osiris_data/"
OSIRISDATA = "/data/osiris_data/"
if 1:
    IFSfilter = "Kbb"#"Hbb"#"Kbb"
    foldername = "HR_8799_c"
    # date = "*"
    # date_list = ["20100715","20101104","20110723"]
    # date_list = ["20101104"]
    # date = "20100715"
    # date = "20101104"
    # date = "20110723"
    # sep = 0.950
    foldername = "HR_8799_d"
    # date_list = ["20150720","20150722","20150723","20150828"]
    # date = "*"
    # date = "20150720"
    date_list = ["20150722"]
    # date = "20150723"
    # date = "20150828"
    # # telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
    # telluric1 = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric_JB/HD_210501","s100715_a005001_Kbb_020.fits")
    # telluric2 = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric_JB/HD_210501","s100715_a005002_Kbb_020.fits")
    # outputdir = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric_JB/")



# filename_filer = "HD_210501/s100715_a00500*_Kbb_020.fits"
filename_filter = "*/*"+IFSfilter+"*020.fits"
# filename_filter = "*/*042001*Kbb*020.fits"
# generate psfs
if 1:
    for date in date_list:
        badpix = True
        refstar_filelist = glob.glob(os.path.join(OSIRISDATA,foldername,date,"reduced_telluric_jb",filename_filter))
        for refstar_filename in refstar_filelist:
            print(refstar_filename)
            psf_cube_size = 15
            with pyfits.open(refstar_filename) as hdulist:
                oripsfs = hdulist[0].data # Nwvs, Ny, Nx

                if 1 and badpix:
                    # oribadpixs = hdulist[1].data
                    # print(hdulist[0].data.shape)
                    # print(hdulist[1].data.shape)
                    # print(hdulist[2].data.shape)
                    # plt.imshow(np.nansum(oribadpixs,axis=2),interpolation="nearest")
                    # plt.show()
                    oribadpixs = hdulist[2].data.astype(np.float)
                    oribadpixs[np.where(oribadpixs==0)] = np.nan
                    for m in range(oribadpixs.shape[0]):
                        print(m+1,oribadpixs.shape[0])
                        for n in range(oribadpixs.shape[1]):
                            widen_nans = np.where(np.isnan(np.correlate(oribadpixs[m,n,:],np.ones(3),mode="same")))[0]
                            oribadpixs[m,n,widen_nans] = np.nan
                    oripsfs[np.where(np.isnan(oribadpixs))] = np.nan
                    where_borders = np.where(np.nansum(oribadpixs,axis=2)<=200)
                    oripsfs[where_borders[0],where_borders[1],:] = 0
                # plt.subplot(1,3,1)
                # plt.imshow(oripsfs[:,:,464])
                # plt.subplot(1,3,2)
                # plt.imshow(oribadpixs[:,:,464])
                # plt.show()

                # Move dimensions of input array to match pyklip conventions
                oripsfs = np.rollaxis(np.rollaxis(oripsfs,2),2,1)

                # if 0 and badpix:
                #     numthreads = 32
                #     prihdr = hdulist[0].header
                #     oripsfs = np.moveaxis(oripsfs,0,2)
                #     ny,nx,nz = oripsfs.shape
                #     init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
                #     dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
                #     wvs=np.arange(init_wv,init_wv+dwv*nz,dwv)
                #     import ctypes
                #     dtype = ctypes.c_float
                #     nan_mask_boxsize=3
                #     from parallelized_polyfit_cov_HPF_osiris import _tpool_init
                #     from parallelized_polyfit_cov_HPF_osiris import _remove_bad_pixels_xy
                #     from parallelized_polyfit_cov_HPF_osiris import _remove_bad_pixels_z
                #     from parallelized_polyfit_cov_HPF_osiris import _remove_edges
                #     from parallelized_polyfit_cov_HPF_osiris import _arraytonumpy
                #     original_imgs = mp.Array(dtype, np.size(oripsfs))
                #     original_imgs_shape = oripsfs.shape
                #     original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
                #     original_imgs_np[:] = oripsfs
                #     badpix_imgs = mp.Array(dtype, np.size(oripsfs))
                #     badpix_imgs_shape = oripsfs.shape
                #     badpix_imgs_np = _arraytonumpy(badpix_imgs, badpix_imgs_shape,dtype=dtype)
                #     badpix_imgs_np[:] = 0
                #     badpix_imgs_np[np.where(original_imgs_np==0)] = np.nan
                #     originalHPF_imgs = mp.Array(dtype, 1)
                #     originalHPF_imgs_shape = (1,)
                #     originalLPF_imgs = mp.Array(dtype, 1)
                #     originalLPF_imgs_shape = (1,)
                #     output_maps_shape = (1,)
                #     output_maps = mp.Array(dtype, 1)
                #     wvs_imgs = wvs
                #     psfs_stamps = mp.Array(dtype, 1)
                #     psfs_stamps_shape = (1,)
                #
                #
                #     ######################
                #     # INIT threads and shared memory
                #     tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                #                     initargs=(original_imgs,badpix_imgs,originalLPF_imgs,originalHPF_imgs, original_imgs_shape, output_maps,
                #                               output_maps_shape,wvs_imgs,psfs_stamps, psfs_stamps_shape),
                #                     maxtasksperchild=50)
                #
                #
                #     # plt.plot(np.ravel(original_imgs_np[30,:,:]),label="original")
                #
                #     ######################
                #     # CLEAN IMAGE
                #
                #
                #     chunk_size = nz//(3*numthreads)
                #     N_chunks = nz//chunk_size
                #     wvs_indices_list = []
                #     for k in range(N_chunks-1):
                #         wvs_indices_list.append(np.arange((k*chunk_size),((k+1)*chunk_size)))
                #     wvs_indices_list.append(np.arange(((N_chunks-1)*chunk_size),nz))
                #
                #
                #     # tasks = [tpool.apply_async(_remove_bad_pixels_xy, args=(wvs_indices,dtype))
                #     #          for wvs_indices in wvs_indices_list]
                #     # #save it to shared memory
                #     # for chunk_index, rmedge_task in enumerate(tasks):
                #     #     print("Finished rm bad pixel xy chunk {0}".format(chunk_index))
                #     #     rmedge_task.wait()
                #     ######################
                #
                #     # plt.plot(np.ravel(original_imgs_np[30,:,:]),linestyle=":",label="bad pix xy")
                #
                #     tasks = [tpool.apply_async(_remove_bad_pixels_z, args=(col_index,nan_mask_boxsize, dtype,100,15.0))
                #              for col_index in range(nx)]
                #     #save it to shared memory
                #     for col_index, bad_pix_task in enumerate(tasks):
                #         print("Finished rm bad pixel z col {0}".format(col_index))
                #         bad_pix_task.wait()
                #
                #     # plt.plot(np.ravel(original_imgs_np[30,:,:]),linestyle=":",label="bad pix z")
                #
                #     # tasks = [tpool.apply_async(_remove_edges, args=(wvs_indices,nan_mask_boxsize,dtype))
                #     #          for wvs_indices in wvs_indices_list]
                #     # #save it to shared memory
                #     # for chunk_index, rmedge_task in enumerate(tasks):
                #     #     print("Finished rm edge chunk {0}".format(chunk_index))
                #     #     rmedge_task.wait()
                #
                #     oripsfs = np.moveaxis(original_imgs_np,2,0)

                nwvs,ny,nx = oripsfs.shape
                init_wv = hdulist[0].header["CRVAL1"]/1000. # wv for first slice in mum
                dwv = hdulist[0].header["CDELT1"]/1000. # wv interval between 2 slices in mum
                wvs=np.arange(init_wv,init_wv+dwv*nwvs,dwv)

                pixelsbefore = psf_cube_size//2
                pixelsafter = psf_cube_size - pixelsbefore

                oripsfs = np.pad(oripsfs,((0,0),(pixelsbefore,pixelsafter),(pixelsbefore,pixelsafter)),mode="constant",constant_values=0)
                psfs_centers = np.array([np.unravel_index(np.nanargmax(img),img.shape) for img in oripsfs])
                # Change center index order to match y,x convention
                psfs_centers = [(cent[1],cent[0]) for cent in psfs_centers]
                psfs_centers = np.array(psfs_centers)
                center0 = np.median(psfs_centers,axis=0)

                psfs_xcenters = []
                psfs_ycenters = []
                psfs_stamps_centers = []
                psfs_stampcenters = []
                star_peaks = []
                psf_stamps = np.zeros((nwvs,psf_cube_size,psf_cube_size))
                for k,im in enumerate(oripsfs):
                    print("center",k)
                    corrflux, fwhm, spotx, spoty = gaussfit2d(im, center0[0], center0[1], searchrad=5, guessfwhm=3, guesspeak=np.nanmax(im), refinefit=True)
                    #spotx, spoty = center0
                    psfs_xcenters.append(spotx)
                    psfs_ycenters.append(spoty)
                    star_peaks.append(corrflux)

                xarr_spot = int(np.round(np.mean(psfs_xcenters)))
                yarr_spot = int(np.round(np.mean(psfs_ycenters)))
                for k,(im,spotx, spoty) in enumerate(zip(oripsfs,psfs_xcenters,psfs_ycenters)):
                    # Get the closest pixel
                    # xarr_spot = int(np.round(spotx))
                    # yarr_spot = int(np.round(spoty))
                    # Extract a stamp around the sat spot
                    stamp = im[(yarr_spot-pixelsbefore):(yarr_spot+pixelsafter),\
                                    (xarr_spot-pixelsbefore):(xarr_spot+pixelsafter)]
                    # Define coordinates grids for the stamp
                    stamp_x, stamp_y = np.meshgrid(np.arange(psf_cube_size, dtype=np.float32),
                                                   np.arange(psf_cube_size, dtype=np.float32))
                    # Calculate the shift of the sat spot centroid relative to the closest pixel.
                    dx = spotx-xarr_spot
                    dy = spoty-yarr_spot
                    stamp_r = np.sqrt((stamp_x-dx-psf_cube_size//2)**2+(stamp_y-dy-psf_cube_size//2)**2)
                    # stamp[np.where(stamp_r>psf_cube_size//2)] = 0
                    stamp[np.where(stamp_r>5)] = 0

                    # plt.imshow(stamp)
                    # plt.show()

                    psfs_stamps_centers.append((dx+psf_cube_size//2, dy+psf_cube_size//2))
                    # print((dx+psf_cube_size//2, dy+psf_cube_size//2))
                    psf_stamps[k,:,:] = stamp
                    # print((spotx, spoty))
                    # print(psf_cube_size,dx,dy,psf_cube_size//2)
                    # exit()

            if badpix:
                suffix = "_badpix2"
            else:
                suffix = ""
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=psf_stamps))
            try:
                hdulist.writeto(refstar_filename.replace(".fits","_psfs"+suffix+".fits"), overwrite=True)
            except TypeError:
                hdulist.writeto(refstar_filename.replace(".fits","_psfs"+suffix+".fits"), clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=np.array(psfs_stamps_centers)))
            try:
                hdulist.writeto(refstar_filename.replace(".fits","_psfs_centers"+suffix+".fits"), overwrite=True)
            except TypeError:
                hdulist.writeto(refstar_filename.replace(".fits","_psfs_centers"+suffix+".fits"), clobber=True)
            hdulist.close()

def _spline_psf_model(paras):
    psfs,xs,ys,xvec,yvec = paras
    normalized_psfs_func_list = []
    for wv_index in range(psfs.shape[-1]):
        # if 1:#np.isnan(psf_func(0,0)[0,0]):
        #     model_psf = psfs[:,:, :, wv_index]
        #     import matplotlib.pyplot as plt
        #     from mpl_toolkits.mplot3d import Axes3D
        #     fig = plt.figure(1)
        #     ax = fig.add_subplot(111,projection="3d")
        #     for k,color in zip(range(model_psf.shape[0]),["pink","blue","green","purple","orange"]):
        #         ax.scatter(xs[k].ravel(),ys[k].ravel(),model_psf[k].ravel(),c=color)
        model_psf = psfs[:,:, :, wv_index].ravel()
        where_nans = np.where(np.isfinite(model_psf))
        psf_func = interpolate.LSQBivariateSpline(xs.ravel()[where_nans],ys.ravel()[where_nans],model_psf[where_nans],xvec,yvec)
        # if 1:
        #     print(psf_func(0,0))
        #     x_psf_vec, y_psf_vec = np.arange(2*nx_psf * 1.)/2.-nx_psf//2, np.arange(2*ny_psf* 1.)/2.-ny_psf//2
        #     x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
        #     ax.scatter(x_psf_grid.ravel(),y_psf_grid.ravel(),psf_func(x_psf_vec,y_psf_vec).transpose().ravel(),c="red")
        #     plt.show()
        normalized_psfs_func_list.append(psf_func)
    # print(len(normalized_psfs_func_list))
    return normalized_psfs_func_list

# repair PSFs
if 1:
    for date in date_list:#["20150720"]:#date_list:
        psfs_tlc = []
        tlc_spec_list = []
        psfs_tlc_filelist = glob.glob("/data/osiris_data/"+foldername+"/"+date+"/reduced_telluric_jb/*/s*"+IFSfilter+"_020_psfs_badpix2.fits")
        psfs_centers_filelist = glob.glob("/data/osiris_data/"+foldername+"/"+date+"/reduced_telluric_jb/*/s*"+IFSfilter+"_020_psfs_centers_badpix2.fits")
        # print(psfs_tlc_filelist)
        # print(psfs_centers_filelist)
        # exit()
        for k,psfs_tlc_filename in enumerate(psfs_tlc_filelist):
            print(psfs_tlc_filename)

            with pyfits.open(psfs_tlc_filename) as hdulist:
                mypsfs = hdulist[0].data
                mypsfs = np.moveaxis(mypsfs,0,2)
                psfs_tlc.append(mypsfs/np.nansum(mypsfs))

        ##############################
        ## Create PSF model
        ##############################
        numthreads=30
        specpool = mp.Pool(processes=numthreads)
        psfs_refstar_arr = np.array(psfs_tlc)
        Npsfs, ny_psf,nx_psf,nz_psf = psfs_refstar_arr.shape
        x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
        x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
        x_psf_vec_hd, y_psf_vec_hd = np.linspace(0,nx_psf * 1.,100)-nx_psf//2,np.linspace(0,ny_psf* 1.,100)-ny_psf//2
        x_psf_grid_list = np.zeros((Npsfs,)+x_psf_grid.shape)
        y_psf_grid_list = np.zeros((Npsfs,)+y_psf_grid.shape)
        for k,centers_filename in enumerate(psfs_centers_filelist):
            with pyfits.open(centers_filename) as hdulist:
                psfs_centers = hdulist[0].data
            # plt.plot(psfs_centers[:,0],label="x {0}".format(k))
            # plt.plot(psfs_centers[:,1],label="y {0}".format(k))
            avg_center = np.mean(psfs_centers,axis=0)
            x_psf_grid_list[k,:,:] = x_psf_grid+(nx_psf//2-avg_center[0])
            y_psf_grid_list[k,:,:] = y_psf_grid+(ny_psf//2-avg_center[1])
        # plt.legend()
        # plt.show()
        normalized_psfs_func_list = []
        # if 1:
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

        new_psfs = np.zeros((nz_psf,ny_psf,nx_psf))
        for psfs_tlc_filename,psfs_centers_filename in zip(psfs_tlc_filelist,psfs_centers_filelist):
            print(psfs_tlc_filename)
            with pyfits.open(psfs_centers_filename) as hdulist:
                centers = hdulist[0].data
            avg_center = np.mean(centers,axis=0)
            with pyfits.open(psfs_tlc_filename) as hdulist:
                mypsfs = hdulist[0].data
                print(mypsfs.shape)
                for z,(im,xycen) in enumerate(zip(mypsfs,centers)):
                    # if z != 465:
                    #     continue
                    new_psfs[z,:,:] = im
                    if np.sum(np.isnan(im)) > 0:
                        print(z)
                        where_nans = np.where(np.isnan(im))
                        where_finite = np.where(np.isfinite(im))
                        x_vec, y_vec = np.arange(nx_psf * 1.)-avg_center[0],np.arange(ny_psf* 1.)-avg_center[1]
                        x_grid, y_grid = np.meshgrid(x_vec, y_vec)
                        pl_x_vec = x_grid[0,:]
                        pl_y_vec = y_grid[:,0]
                        tmpstamp = normalized_psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()
                        fitpsf = tmpstamp*np.nansum(tmpstamp[where_finite]*im[where_finite])/np.sum(tmpstamp[where_finite]**2)
                        (new_psfs[z,:,:])[where_nans] = fitpsf[where_nans]

                        # plt.subplot(2,2,1)
                        # plt.imshow(im,interpolation="nearest")
                        # plt.subplot(2,2,2)
                        # plt.imshow(tmpstamp,interpolation="nearest")
                        # plt.colorbar()
                        # plt.subplot(2,2,3)
                        # plt.imshow(fitpsf,interpolation="nearest")
                        # plt.colorbar()
                        # plt.subplot(2,2,4)
                        # plt.imshow(new_psfs[z,:,:],interpolation="nearest")
                        # plt.show()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=new_psfs))
            try:
                hdulist.writeto(psfs_tlc_filename.replace("_badpix2.fits","_repaired.fits"), overwrite=True)
            except TypeError:
                hdulist.writeto(psfs_tlc_filename.replace("_badpix2.fits","_repaired.fits"), clobber=True)
            hdulist.close()
    # exit()

# PLOT telluric PSF
if 0:
    from scipy import interpolate
    filename_filter = "*/*Kbb_020_psfs.fits"
    psfs_tlc_filelist = glob.glob(os.path.join(OSIRISDATA,foldername,date,"reduced_telluric_jb",filename_filter))
    psfs_tlc_filelist.sort()
    psfs_tlc = []
    tlc_spec_list = []
    for psfs_tlc_filename in psfs_tlc_filelist:
        with pyfits.open(psfs_tlc_filename) as hdulist:
            mypsfs = hdulist[0].data
            # psfs_tlc_prihdr = hdulist[0].header
            mypsfs = np.moveaxis(mypsfs,0,2)
            mypsfs[np.where(np.isnan(mypsfs))] = 0
            psfs_tlc.append(mypsfs)
            tlc_spec_list.append(np.nansum(mypsfs,axis=(0,1)))

    psfs_tlc = [cube[None,:,:,:] for cube in psfs_tlc]
    psfs_tlc = np.concatenate(psfs_tlc,axis=0)
    print(psfs_tlc.shape)

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
            # xcoords= psfs_centers[:,0]
            # ycoords= psfs_centers[:,0]
            avg_center = np.mean(psfs_centers,axis=0)
            x_psf_grid_list[k,:,:] = x_psf_grid+(nx_psf//2-avg_center[0])
            y_psf_grid_list[k,:,:] = y_psf_grid+(ny_psf//2-avg_center[1])
            # plt.figure(k+1)
            # print(k+1,avg_center)
            # plt.subplot(1,2,1)
            # plt.imshow(np.sqrt((x_psf_grid+(nx_psf//2-avg_center[0]))**2+(y_psf_grid+(ny_psf//2-avg_center[1]))**2))
            # plt.subplot(1,2,2)
            # plt.imshow(np.sum(tmp/np.sum(tmp,axis=(0,1))[None,None,:],axis=2),interpolation="nearest")
    # plt.show()

    psfs_func_list = []
    normalized_psfs_func_list = []
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for wv_index in range(nz_psf):
            print(wv_index)
            model_psf = psfs_tlc[:,:, :, wv_index]
            # print(np.sqrt(np.nansum(model_psf**2,axis=(1,2))))
            # print(np.nanmax(model_psf**2,axis=(1,2)))
            model_psf = model_psf/np.sqrt(np.nansum(model_psf**2,axis=(1,2)))[:,None,None]
            print(model_psf.shape)
            print(x_psf_grid_list.shape)
            print(y_psf_grid_list.shape)
            print(np.sqrt(np.nansum(model_psf**2,axis=(1,2))))
            print(np.nanmax(model_psf**2,axis=(1,2)))
            # exit()
            psf_func = interpolate.LSQBivariateSpline(x_psf_grid_list.ravel(),y_psf_grid_list.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
            normalized_psfs_func_list.append(psf_func)
            if wv_index%100 == 0:
                plt.imshow(normalized_psfs_func_list[wv_index](x_psf_vec_hd, y_psf_vec_hd).transpose(),interpolation="nearest")
                plt.show()
    exit()

# PLOT telluric spectra
if 1:
    date = date_list[0]#"20150720"
    # suffix = "_badpix"
    suffix = ""
    # suffix = "_repaired"
    # filename_filter = "*/*042001*Kbb_020_psfs"+suffix+".fits"
    filename_filter = "*/*Kbb_020_psfs"+suffix+".fits"

    plt.figure(1)
    refstar_filelist = glob.glob(os.path.join(OSIRISDATA,foldername,date,"reduced_telluric_jb",filename_filter))
    for refstar_filename in refstar_filelist:
        print(refstar_filename)
        psf_cube_size = 15
        with pyfits.open(refstar_filename) as hdulist:
            oripsfs = hdulist[0].data # Nwvs, Ny, Nx
        tlc_spec = np.nansum(oripsfs,axis=(1,2))
        plt.plot(tlc_spec/np.nanmedian(tlc_spec),label=os.path.basename(refstar_filename))
        try:
        # if 1:
        #     with pyfits.open(refstar_filename.replace(".fits","_badpix2"+".fits")) as hdulist:
            with pyfits.open(refstar_filename.replace(".fits","_repaired"+".fits")) as hdulist:
                oripsfs_bp = hdulist[0].data # Nwvs, Ny, Nx
            tlc_spec_bp = np.sum(oripsfs_bp,axis=(1,2))
            plt.plot(tlc_spec_bp/np.nanmedian(tlc_spec_bp),linestyle="--",label=os.path.basename(refstar_filename)+" bad pix")
            plt.plot((tlc_spec-tlc_spec_bp)/np.nanmedian(tlc_spec_bp),linestyle=":")
        except:
            pass
    plt.legend()

    center_filename_filter = "*/*Kbb_020_psfs_centers_badpix2.fits"
    plt.figure(2)
    refstar_filelist = glob.glob(os.path.join(OSIRISDATA,foldername,date,"reduced_telluric_jb",center_filename_filter))
    print(os.path.join(OSIRISDATA,foldername,date,"reduced_telluric_jb",center_filename_filter))
    # print(refstar_filelist)
    # exit()
    for refstar_filename in refstar_filelist:
        print(refstar_filename)
        psf_cube_size = 15
        with pyfits.open(refstar_filename) as hdulist:
            oripsfs_center = hdulist[0].data # Nwvs, Ny, Nx
        plt.subplot(2,2,1)
        plt.plot(oripsfs_center[:,0],label=os.path.basename(refstar_filename)+suffix)
        plt.subplot(2,2,3)
        plt.plot(oripsfs_center[:,0]-np.median(oripsfs_center[:,0]),label=os.path.basename(refstar_filename)+suffix)
        plt.subplot(2,2,2)
        plt.plot(oripsfs_center[:,1],label=os.path.basename(refstar_filename)+suffix)
        plt.subplot(2,2,4)
        plt.plot(oripsfs_center[:,1]-np.median(oripsfs_center[:,1]),label=os.path.basename(refstar_filename)+suffix)
    plt.legend()
    plt.show()

    # plt.plot(psfs_centers)
    # plt.show()
    #
    # spec1 = np.nansum(psf_stamps1,axis=(1,2))
    # spec2 = np.nansum(psf_stamps2,axis=(1,2))
    # plt.plot(spec1,color="blue")
    # plt.plot(spec2,color="red")
    # plt.plot(spec1-spec2,color="red")
    # plt.show()

# # generate psfs
# if 0:
#     psf_cube_size = 15
#     with pyfits.open(telluric2) as hdulist:
#         oripsfs = hdulist[0].data # Nwvs, Ny, Nx
#         # Move dimensions of input array to match pyklip conventions
#         oripsfs = np.rollaxis(np.rollaxis(oripsfs,2),2,1)
#         nwvs,ny,nx = oripsfs1.shape
#         init_wv = hdulist[0].header["CRVAL1"]/1000. # wv for first slice in mum
#         dwv = hdulist[0].header["CDELT1"]/1000. # wv interval between 2 slices in mum
#         wvs=np.arange(init_wv,init_wv+dwv*nwvs,dwv)
#
#     if 1:
#         # trim the cube
#         pixelsbefore = psf_cube_size//2
#         pixelsafter = psf_cube_size - pixelsbefore
#
#         oripsfs = np.pad(oripsfs,((0,0),(pixelsbefore,pixelsafter),(pixelsbefore,pixelsafter)),mode="constant",constant_values=0)
#         psfs_centers = np.array([np.unravel_index(np.nanargmax(img),img.shape) for img in oripsfs])
#         # Change center index order to match y,x convention
#         psfs_centers = [(cent[1],cent[0]) for cent in psfs_centers]
#         psfs_centers = np.array(psfs_centers)
#         center0 = np.median(psfs_centers,axis=0)
#
#         from pyklip.fakes import gaussfit2d
#         psfs_centers = []
#         psfs_stampcenters = []
#         star_peaks = []
#         psf_notrescaled = np.zeros((nwvs,psf_cube_size,psf_cube_size))
#         for k,im in enumerate(oripsfs):
#             print("center",k)
#             corrflux, fwhm, spotx, spoty = gaussfit2d(im, center0[0], center0[1], searchrad=5, guessfwhm=3, guesspeak=np.nanmax(im), refinefit=True)
#             #spotx, spoty = center0
#             psfs_centers.append((spotx, spoty))
#             star_peaks.append(corrflux)
#
#             # Get the closest pixel
#             xarr_spot = int(np.round(spotx))
#             yarr_spot = int(np.round(spoty))
#             # Extract a stamp around the sat spot
#             stamp = im[(yarr_spot-pixelsbefore):(yarr_spot+pixelsafter),\
#                             (xarr_spot-pixelsbefore):(xarr_spot+pixelsafter)]
#             # Define coordinates grids for the stamp
#             stamp_x, stamp_y = np.meshgrid(np.arange(psf_cube_size, dtype=np.float32),
#                                            np.arange(psf_cube_size, dtype=np.float32))
#             # Calculate the shift of the sat spot centroid relative to the closest pixel.
#             dx = spotx-xarr_spot
#             dy = spoty-yarr_spot
#             # print(spotx, spoty)
#             # print(xarr_spot,yarr_spot)
#             # print(dx,dy)
#             # print(pixelsbefore,pixelsafter)
#             # print(pixelsbefore+dx,pixelsbefore+dy)
#             # plt.figure(1)
#             # plt.imshow(stamp,interpolation="nearest")
#             # plt.show()
#             psfs_stampcenters.append((pixelsbefore+dx,pixelsbefore+dy))
#
#             # The goal of the following section is to remove the local background (or sky) around the sat spot.
#             # The plane is defined by 3 constants (a,b,c) such that z = a*x+b*y+c
#             # In order to do so we fit a 2D plane to the stamp after having masked the sat spot (centered disk)
#             stamp_r = np.sqrt((stamp_x-dx-psf_cube_size//2)**2+(stamp_y-dy-psf_cube_size//2)**2)
#             from copy import copy
#             stamp_masked = copy(stamp)
#             stamp_x_masked = stamp_x-dx
#             stamp_y_masked = stamp_y-dy
#             stamp_center = np.where(stamp_r<7)
#             stamp_masked[stamp_center] = np.nan
#             stamp_x_masked[stamp_center] = np.nan
#             stamp_y_masked[stamp_center] = np.nan
#             background_med =  np.nanmedian(stamp_masked)
#             stamp_masked = stamp_masked - background_med
#             #Solve 2d linear fit to remove background
#             xx = np.nansum(stamp_x_masked**2)
#             yy = np.nansum(stamp_y_masked**2)
#             xy = np.nansum(stamp_y_masked*stamp_x_masked)
#             xz = np.nansum(stamp_masked*stamp_x_masked)
#             yz = np.nansum(stamp_y_masked*stamp_masked)
#             #Cramer's rule
#             a = (xz*yy-yz*xy)/(xx*yy-xy*xy)
#             b = (xx*yz-xy*xz)/(xx*yy-xy*xy)
#             # plt.figure(1)
#             # plt.imshow(stamp,interpolation="nearest")
#             stamp = stamp - (a*(stamp_x-dx)+b*(stamp_y-dy) + background_med)
#
#             # plt.figure(2)
#             # plt.imshow(stamp,interpolation="nearest")
#             # plt.show()
#
#             stamp[np.where(stamp_r>psf_cube_size//2)] = np.nan
#             psf_notrescaled[k,:,:] = stamp
#
#
#         N_threads=30
#         chunk_size = nwvs//N_threads
#         pool = mp.Pool(processes=N_threads)
#         N_chunks = nwvs//chunk_size
#         wvs_ref_cut = []
#         for k in range(N_chunks-1):
#             wvs_ref_cut.append(wvs[(k*chunk_size):((k+1)*chunk_size)])
#         wvs_ref_cut.append(wvs[((N_chunks-1)*chunk_size):nwvs])
#
#         psf_notrescaled[294:302] = np.nan
#         psf_notrescaled[319:325] = np.nan
#         psf_notrescaled[327:332] = np.nan
#         psf_notrescaled[654:662] = np.nan
#         psf_notrescaled[692:698] = np.nan
#         psf_notrescaled[1050:1057] = np.nan
#
#         outputs_list = pool.map(align_and_scale_star, zip(wvs_ref_cut,
#                                                            itertools.repeat(psf_notrescaled),
#                                                            itertools.repeat(wvs),
#                                                            itertools.repeat([psf_cube_size//2,psf_cube_size//2]),
#                                                            itertools.repeat(psfs_stampcenters)))
#         aligned_psfs = []
#         for out in outputs_list:
#             print(out.shape)
#             aligned_psfs.append(out)
#         print(len(aligned_psfs))
#         aligned_psfs = np.concatenate(aligned_psfs,axis=0)
#         print(aligned_psfs.shape)
#         hdulist = pyfits.HDUList()
#         hdulist.append(pyfits.PrimaryHDU(data=aligned_psfs))
#         try:
#             hdulist.writeto(os.path.join(outputdir,os.path.basename(telluric_cube).replace(".fits","_4dalignedpsfs.fits")), overwrite=True)
#         except TypeError:
#             hdulist.writeto(os.path.join(outputdir,os.path.basename(telluric_cube).replace(".fits","_4dalignedpsfs.fits")), clobber=True)
#         hdulist.close()
#     else:
#         if 0:
#             with pyfits.open(os.path.join(outputdir,os.path.basename(telluric_cube).replace(".fits","_4dalignedpsfs.fits"))) as hdulist:
#                 aligned_psfs = hdulist[0].data
#                 hdulist.close()
#             # plt.figure(1)
#             # plt.imshow(aligned_psfs[:,:,psf_cube_size//2,psf_cube_size//2],interpolation ="nearest")
#             # plt.show()
#             aligned_psfs = aligned_psfs/aligned_psfs[:,:,psf_cube_size//2,psf_cube_size//2][:,:,None,None]
#
#             # plt.figure(1)
#             # for k in range(16):
#             #     for l in range(16):
#             #         plt.subplot(16,16,k*16+l +1)
#             #         plt.imshow(aligned_psfs[k,l,:,:],interpolation="nearest")
#             # plt.show()
#
#             medcombinedpsfs = np.nanmedian(aligned_psfs,axis=1)
#
#             medcombinedpsfs[np.where(np.isnan(medcombinedpsfs))] = 0.0
#             medcombinedpsfs[:,0:1,:]=0
#             medcombinedpsfs[:,:,0:1]=0
#             medcombinedpsfs[:,psf_cube_size-1:psf_cube_size,:]=0
#             medcombinedpsfs[:,:,psf_cube_size-1:psf_cube_size]=0
#
#             hdulist = pyfits.HDUList()
#             hdulist.append(pyfits.PrimaryHDU(data=medcombinedpsfs))
#             try:
#                 hdulist.writeto(os.path.join(outputdir,os.path.basename(telluric_cube).replace(".fits","_medcombinedpsfs.fits")), overwrite=True)
#             except TypeError:
#                 hdulist.writeto(os.path.join(outputdir,os.path.basename(telluric_cube).replace(".fits","_medcombinedpsfs.fits")), clobber=True)
#             hdulist.close()
#         else:
#             with pyfits.open(os.path.join(outputdir,os.path.basename(telluric_cube).replace(".fits","_medcombinedpsfs.fits"))) as hdulist:
#                 medcombinedpsfs = hdulist[0].data
#                 hdulist.close()
#
#         plt.figure(1)
#         plt.plot(np.sum(medcombinedpsfs,axis=(1,2)))
#         plt.show()
#     exit()