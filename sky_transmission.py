__author__ = 'jruffio'

import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import os
import glob
import csv
from copy import copy
from scipy.interpolate import interp1d
import multiprocessing as mp

def convolve_spectrum(wvs,spectrum,R):
    conv_spectrum = np.zeros(spectrum.shape)
    dwvs = wvs[1::]-wvs[0:(np.size(wvs)-1)]
    med_dwv = np.median(dwvs)
    for k,pwv in enumerate(wvs):
        FWHM = pwv/R
        sig = FWHM/(2*np.sqrt(2*np.log(2)))
        w = int(np.round(sig/med_dwv*10.))
        stamp_spec = spectrum[np.max([0,k-w]):np.min([np.size(spectrum),k+w])]
        stamp_wvs = wvs[np.max([0,k-w]):np.min([np.size(wvs),k+w])]
        stamp_dwvs = stamp_wvs[1::]-stamp_wvs[0:(np.size(stamp_spec)-1)]
        gausskernel = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(stamp_wvs-pwv)**2/sig**2)
        conv_spectrum[k] = np.sum(gausskernel[1::]*stamp_spec[1::]*stamp_dwvs)
    return conv_spectrum

def LPFvsHPF(myvec,cutoff):
    fftmyvec = np.fft.fft(np.concatenate([myvec,myvec[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec)]
    HPF_myvec = myvec - LPF_myvec
    return LPF_myvec,HPF_myvec

def CCF(dwvs,wvs,spec,hd_wvs,hd_spec):
    f = interp1d(hd_wvs,hd_spec,bounds_error=False,fill_value=0)
    ccf_arr = np.zeros(dwvs.shape)
    for k,dwv in enumerate(dwvs):
        ccf_arr[k]=np.nansum(spec*f(wvs-dwv))
        # print(k,dwv)
        # plt.plot(spec)
        # plt.plot(f(wvs-dwv))
        # plt.show()
    return ccf_arr


if 1:
    # filename = glob.glob("/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/20190131_HPF_only/*search.fits")[0]
    filename = glob.glob("/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/20190131_HPF_only/*search.fits")[0]
    hdulist = pyfits.open(filename)
    output = hdulist[0].data
    print(output.shape)

    N_wvshifts,N_R,N_models,_,_,ny,nx = output.shape

    output_AIC_H1 = output[:,:,:,3,0,:,:]
    output_AIC_H0 = output[:,:,:,4,0,:,:]
    output_dAIC = output[:,:,:,2,0,:,:]

    # plt.subplot(1,4,1)
    # plt.imshow(output[0,0,0,2,0,:,:],interpolation="nearest")
    # plt.subplot(1,4,2)
    # plt.imshow(output[0,0,0,4,0,:,:]-output[0,0,0,3,0,:,:],interpolation="nearest")
    # plt.subplot(1,4,3)
    # plt.imshow(output[0,0,0,4,0,:,:],interpolation="nearest")
    # plt.subplot(1,4,4)
    # plt.imshow(output[0,0,0,3,0,:,:],interpolation="nearest")
    # plt.show()


    final_dAIC = np.zeros((ny,nx))
    for row in range(ny):
        for col in range(nx):
            # AIC_allpara = output_AIC_H0[:,:,:,row,col]
            # unravel_index_argmax = np.unravel_index(np.argmmax(AIC_allpara),(N_wvshifts,N_R,N_models))
            # final_dAIC[row,col] = output_AIC_H0[unravel_index_argmax[0],unravel_index_argmax[1],unravel_index_argmax[2],row,col] -\
            #     output_AIC_H1[unravel_index_argmax[0],unravel_index_argmax[1],unravel_index_argmax[2],row,col]
            AIC_allpara = output_dAIC[:,:,:,row,col]
            unravel_index_argmax = np.unravel_index(np.argmax(AIC_allpara),(N_wvshifts,N_R,N_models))
            final_dAIC[row,col] = output_dAIC[unravel_index_argmax[0],unravel_index_argmax[1],unravel_index_argmax[2],row,col]

    plt.imshow(final_dAIC[::-1,:],interpolation="nearest")
    plt.show()


    exit()

if 0:
    IFSfilter = "Kbb"
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
    init_wv = CRVAL1/1000.
    dwv = CDELT1/1000.
    wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)

    cutoff = 160
    nccf = 201
    dwvs_CCF = np.linspace(-dwv*1,dwv*1,nccf)
    debug = False
    numthreads = 28

    fun_list = []
    # # filename = "/home/sda/jruffio/osiris_data/sky_transmission/mktrans_zm_50_20.dat"
    filelist_skytr = glob.glob("/home/sda/jruffio/osiris_data/sky_transmission/mktrans_zm_50_20.dat")
    for filename_skytr in filelist_skytr[::-1]:
        print(filename_skytr)
        skybg_arr=np.loadtxt(filename_skytr)
        print(skybg_arr.shape)
        skybg_wvs = skybg_arr[:,0]
        skybg_spec = skybg_arr[:,1]
        selec_skybg = np.where((skybg_wvs>wvs[0]-(wvs[-1]-wvs[0])/2)*(skybg_wvs<wvs[-1]+(wvs[-1]-wvs[0])/2))
        skybg_wvs = skybg_wvs[selec_skybg]
        skybg_spec = convolve_spectrum(skybg_wvs,skybg_spec[selec_skybg],R)
        fun_list.append(interp1d(skybg_wvs,skybg_spec/np.nanstd(skybg_spec),bounds_error=False,fill_value=0))
    # plt.figure(1)
    # plt.plot(skybg_wvs,skybg_spec)#/np.mean(skybg_spec),label="ref")
    # plt.show()
    # exit()

    if 1:
        planet = "c"
        date = "100715"
        IFSfilter = "Kbb"
        inputDir = "/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb/"
        filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_020.fits"))
        filelist.sort()
        filelist = filelist[0:1]
        psfs_tlc_filelist = glob.glob("/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_telluric_jb/*/s*"+IFSfilter+"_020_psfs.fits") #_repaired
        phoenix_folder = os.path.join("/home/sda/jruffio/osiris_data/phoenix")

        psfs_tlc = []
        tlc_spec_list = []
        for psfs_tlc_filename in psfs_tlc_filelist[0:1]:
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
            with pyfits.open(psfs_tlc_filename.replace("_psfs","").replace("_repaired","")) as hdulist:
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

                # myspec = myspec * 10**(-1./2.5*(hr8799_mag-ref_star_mag))
                # myspec_flux = np.sum(myspec)
                # myspec = myspec * (phoenix_HR8799/phoenix_tlc)
                # myspec = myspec / np.sum(myspec) *myspec_flux
                # # print(IFSfilter,ref_star_name,np.nanmean(myspec),psfs_tlc_itime)
                # tlc_spec_list.append(myspec)

                myspec = myspec/phoenix_tlc
                plt.figure(1)
                plt.plot(wvs,myspec/np.nanmean(myspec),label=os.path.basename(psfs_tlc_filename))

                plt.figure(2)
                plt.plot(wvs,(myspec*phoenix_tlc)/np.nanmean(myspec*phoenix_tlc),label=os.path.basename(psfs_tlc_filename))
                plt.plot(wvs,(phoenix_tlc)/np.nanmean(phoenix_tlc),label=os.path.basename(psfs_tlc_filename))
                plt.plot(wvs,(fun_list[0](wvs))/np.nanmean(fun_list[0](wvs)),label=os.path.basename(psfs_tlc_filename))
                # plt.figure(2)
                # for k in range(12):
                #     plt.subplot(4,3,k+1)
                #     plt.plot(wvs,(myspec/fun_list[k](wvs))/np.nanmean(myspec/fun_list[k](wvs)),label=os.path.basename(psfs_tlc_filename))
                #     plt.title(os.path.basename(filelist_skytr[k]))



    plt.figure(1)
    plt.legend()
    # plt.subplot(1,2,2)
    # plt.legend()

    plt.show()