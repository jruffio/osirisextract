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
OSIRISDATA = "/home/sda/jruffio/osiris_data/"
if 1:
    foldername = "HR_8799_c"
    sep = 0.950
    # telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
    telluric1 = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric_JB/HD_210501","s100715_a005001_Kbb_020.fits")
    telluric2 = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric_JB/HD_210501","s100715_a005002_Kbb_020.fits")
    outputdir = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric_JB/")

# filename_filer = "HD_210501/s100715_a00500*_Kbb_020.fits"
filename_filter = "*/*.fits"
# generate psfs
if 0:
    telluric_list = glob.glob(os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric_JB/",filename_filter))
    for telluric in telluric_list:
        psf_cube_size = 15
        with pyfits.open(telluric) as hdulist:
            oripsfs = hdulist[0].data # Nwvs, Ny, Nx
            # Move dimensions of input array to match pyklip conventions
            oripsfs = np.rollaxis(np.rollaxis(oripsfs,2),2,1)
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

            psfs_centers = []
            psfs_stampcenters = []
            star_peaks = []
            psf_stamps = np.zeros((nwvs,psf_cube_size,psf_cube_size))
            for k,im in enumerate(oripsfs):
                print("center",k)
                corrflux, fwhm, spotx, spoty = gaussfit2d(im, center0[0], center0[1], searchrad=5, guessfwhm=3, guesspeak=np.nanmax(im), refinefit=True)
                #spotx, spoty = center0
                psfs_centers.append((spotx, spoty))
                star_peaks.append(corrflux)

                # Get the closest pixel
                xarr_spot = int(np.round(spotx))
                yarr_spot = int(np.round(spoty))
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
                stamp[np.where(stamp_r>psf_cube_size//2)] = np.nan

                # plt.imshow(stamp)
                # plt.show()

                psf_stamps[k,:,:] = stamp

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=psf_stamps))
        try:
            hdulist.writeto(telluric.replace(".fits","_psfs.fits"), overwrite=True)
        except TypeError:
            hdulist.writeto(telluric.replace(".fits","_psfs.fits"), clobber=True)
        hdulist.close()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=np.array(psfs_centers)))
        try:
            hdulist.writeto(telluric.replace(".fits","_psfs_centers.fits"), overwrite=True)
        except TypeError:
            hdulist.writeto(telluric.replace(".fits","_psfs_centers.fits"), clobber=True)
        hdulist.close()

# PLOT telluric spectra
if 1:
    filename_filter = "*/*_psfs.fits"

    plt.figure(1)
    telluric_list = glob.glob(os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric_JB/",filename_filter))
    for telluric in telluric_list:
        psf_cube_size = 15
        with pyfits.open(telluric) as hdulist:
            oripsfs = hdulist[0].data # Nwvs, Ny, Nx
        tlc_spec = np.nansum(oripsfs,axis=(1,2))
        plt.plot(tlc_spec/np.median(tlc_spec),label=os.path.basename(telluric))
    plt.legend()

    center_filename_filter = "*/*_psfs_centers.fits"
    plt.figure(2)
    telluric_list = glob.glob(os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric_JB/",center_filename_filter))
    for telluric in telluric_list:
        psf_cube_size = 15
        with pyfits.open(telluric) as hdulist:
            oripsfs_center = hdulist[0].data # Nwvs, Ny, Nx
        plt.plot(oripsfs_center,label=os.path.basename(telluric))
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