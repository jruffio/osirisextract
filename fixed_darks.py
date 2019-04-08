__author__ = 'jruffio'

import csv
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import astropy.io.fits as pyfits

if 0:
    inputdir = "/data/osiris_data/HR_8799_b"

    if 1:
        filelist = glob.glob(os.path.join(inputdir,"2009*/dark_900/s*.fits"))
        filelist.sort()
        print(filelist)
        print(os.path.join(inputdir,"2009*/dark_900/s*.fits"))
        for k,filename in enumerate(filelist):
            hdulist = pyfits.open(filename)
            prihdr = hdulist[0].header
            print(os.path.basename(filename),prihdr["SCURTMP"],prihdr["ITIME"])
        exit()



if 1:
    # 20100715 => np.roll(hdulist[0].data,1328,axis=1)
    inputdir = "/data/osiris_data/HR_8799_b"
    filelist = glob.glob(os.path.join(inputdir,"2009*/dark_900/s*.fits"))

    if 0:
        filelist = glob.glob(os.path.join(inputdir,"2017*/raw/s*.fits"))
        filelist.sort()
        for k,filename in enumerate(filelist):
            hdulist = pyfits.open(filename)
            prihdr = hdulist[0].header
            print(os.path.basename(filename),prihdr["ITIME"])
        exit()

    ref_dark_filename = os.path.join("/data/osiris_data/HR_8799_b","20100711/dark_600/s100711_a001001.fits")
    ref_hdulist = pyfits.open(ref_dark_filename)
    prihdr = ref_hdulist[0].header
    ref_dark = ref_hdulist[0].data
    ref_dark[np.where(ref_dark>0.3)] = np.nan
    plt.figure(1)
    plt.subplot(1,len(filelist)+1,1)
    plt.imshow(ref_dark)
    plt.colorbar()
    plt.clim([0,0.5])
    for k,filename in enumerate(filelist):
        hdulist = pyfits.open(filename)
        prihdr = hdulist[0].header
        dark = hdulist[0].data
        dark[np.where(dark>0.5)] = np.nan
        plt.subplot(1,len(filelist)+1,1+k+1)
        plt.imshow(dark)
        # plt.imshow(np.roll(dark,100,axis=1))
        plt.colorbar()
        plt.clim([0,0.5])

        hdulist2save = pyfits.HDUList()
        hdulist2save.append(pyfits.PrimaryHDU(data=np.roll(hdulist[0].data,1328,axis=1),header=hdulist[0].header))
        hdulist2save.append(pyfits.PrimaryHDU(data=ref_hdulist[1].data,header=hdulist[1].header))
        hdulist2save.append(pyfits.PrimaryHDU(data=hdulist[2].data,header=hdulist[2].header))
        try:
            hdulist2save.writeto(filename.replace(".fits","_fixed.fits"), overwrite=True)
        except TypeError:
            hdulist2save.writeto(filename.replace(".fits","_fixed.fits"), clobber=True)
        hdulist2save.close()
    exit()

    ny,nx = ref_dark.shape
    # print(prihdr["ITIME"])
    plt.show()

    ccf_ref = np.zeros(nx)
    for l in range(nx):
        print(l)
        ccf_ref[l] = np.nansum(ref_dark*np.roll(ref_dark,l,axis=1))
    plt.figure(2)
    plt.plot(ccf_ref,label="ref")

    plt.figure(3)
    ccf = np.zeros((len(ref_dark_filename),nx))
    for k,filename in enumerate(filelist):
        hdulist = pyfits.open(filename)
        prihdr = hdulist[0].header
        dark = hdulist[0].data
        dark[np.where(dark>0.5)] = np.nan
        # print(prihdr["ITIME"])

        for l in range(nx):
            print(k,l)
            ccf[k,l] = np.nansum(ref_dark*np.roll(dark,l,axis=1))
        plt.plot(ccf[k,:],label="{0}".format(k))
        print(np.argmax(ccf[k,:])) #1328
    plt.legend()
    plt.show()
