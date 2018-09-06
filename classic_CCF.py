__author__ = 'jruffio'


import sys
import glob
import os
import multiprocessing as mp
import numpy as np
import astropy.io.fits as pyfits

import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
import xml.etree.ElementTree as ET

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
        outputdir = "/home/sda/Dropbox (GPI)/TEST_SCRATCH/scratch/JB/OSIRIS_utils/bruce_inspired_outputs/"
        inputDir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_quinn/"
        # inputDir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/"
        telluric_cube = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_telluric/HD_210501/s100715_a005001_Kbb_020.fits"
        template_spec="/data/Dropbox (GPI)/TEST_SCRATCH/scratch/JB/hr8799c_osiris_template.save"
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
        template_spec = sys.argv[5]
        sep_planet = float(sys.argv[6])
        numthreads = int(sys.argv[7])


    with pyfits.open(filename) as hdulist:
        input0 = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
    nl,ny,nx = input0.shape

    fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos.xml"
    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
    tree = ET.parse(fileinfos_filename)
    root = tree.getroot()

    filebasename = os.path.basename(filename)
    planet_c = root.find("c")
    fileelement = planet_c.find(filebasename)
    print(fileelement.attrib["xADIcen"],fileelement.attrib["yADIcen"])

    #########
    # compute spectrum of HR 8799 (the star)
    hr8799_spec = np.nanmean(input0,axis=(1,2))
    hr8799_spec = hr8799_spec/np.mean(hr8799_spec)

    #################
    # Load planet template spectrum
    import scipy.io as scio
    travis_spectrum = scio.readsav("/data/Dropbox (GPI)/TEST_SCRATCH/scratch/JB/hr8799c_osiris_template.save")
    hr8799c_spec = np.array(travis_spectrum["fk_bin"])
    lambdas = np.array(travis_spectrum["kwave"])

    mn_input = np.nanmean(input0,axis=(1,2))
    input0 = input0/hr8799_spec[:,None,None]

    from scipy.interpolate import UnivariateSpline
    ccmap = np.zeros((ny,nx))
    model = hr8799c_spec/hr8799_spec/np.nanmean(hr8799c_spec/hr8799_spec)
    fwhm = 75
    sig = fwhm/(2*np.sqrt(2.*np.log(2.)))
    template = np.exp(-np.arange(-4*fwhm,4*fwhm)**2/sig**2)
    template /= np.sum(template**2)
    # smooth_model = np.correlate(model,template,mode="same")[100:(nl-100)]
    # hpf_model = model-smooth_model*(np.sum(smooth_model*model)/np.sum(smooth_model*smooth_model))
    smooth_model = median_filter(model,footprint=np.ones(200),mode="constant",cval=0.0)
    hpf_model = model - smooth_model
    hpf_model = hpf_model[100:(nl-100)]
    for k in range(ny):
        for l in range(nx):
            print(k,l)
            data = input0[:, k, l]/np.mean(input0[:, k, l])
            if np.sum(np.isnan(data)) >0:
                continue
            # smooth_data = np.correlate(data,template,mode="same")[100:(nl-100)]
            # data2 = data[100:(nl-100)]
            # hpf_data = data2-smooth_data*(np.sum(smooth_data*data2)/np.sum(smooth_data*smooth_data))
            hpf_data = data - median_filter(data,footprint=np.ones(200),mode="constant",cval=0.0)
            hpf_data = hpf_data[100:(nl-100)]
            data[(np.where(np.abs(hpf_data)>3*np.nanstd(hpf_data))[0]+100,)] = 0
            # smooth_data = np.correlate(data,template,mode="same")[100:(nl-100)]
            # data = data[100:(nl-100)]
            # hpf_data = data-smooth_data*(np.sum(smooth_data*data)/np.sum(smooth_data*smooth_data))
            hpf_data = data - median_filter(data,footprint=np.ones(200),mode="constant",cval=0.0)
            hpf_data = hpf_data[100:(nl-100)]
            ccmap[k,l] = np.sum(hpf_data*hpf_model)

    suffix = "_medfilt_ccmap"
    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=ccmap))
    try:
        hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits")), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits")), clobber=True)
    hdulist.close()

    PSF = np.ones((3,3))
    from scipy.signal import correlate2d
    ccmap_convo = correlate2d(ccmap,PSF,mode="same")
    ccmap_convo[np.where(np.sum(input0,axis=0)==0)] = np.nan
    suffix = "_medfilt_ccmapconvo"
    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=ccmap_convo))
    try:
        hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits")), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits")), clobber=True)
    hdulist.close()