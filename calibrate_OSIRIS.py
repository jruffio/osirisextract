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
from scipy.signal import correlate2d
from scipy.ndimage.filters import median_filter
from astropy.stats import mad_std

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


def LPFvsHPF_median(myvec,window_size=100):
    threshold = 7
    myvec_cp = copy(myvec)
    lpf_myvec_cp = median_filter(myvec_cp,footprint=np.ones(window_size),mode="constant",
                               cval=0)
    # hpf_myvec_cp = myvec_cp - lpf_myvec_cp
    # wherefinite = np.where(np.isfinite(hpf_myvec_cp))
    # mad = mad_std(hpf_myvec_cp[wherefinite])
    # whereoutliers = np.where(np.abs(hpf_myvec_cp)>threshold*mad)[0]
    # myvec_cp[whereoutliers] = lpf_myvec_cp[whereoutliers]
    # lpf_myvec_cp = median_filter(myvec_cp,footprint=np.ones(window_size),mode="constant",
    #                            cval=0)
    return lpf_myvec_cp,myvec-lpf_myvec_cp

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

def _tpool_init(_original_imgs,_badpix_imgs,_output_ccf,_original_imgs_shape,_badpix_imgs_shape,_output_ccf_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Args:
    """
    global original_imgs,badpix_imgs,output_ccf, original_imgs_shape, badpix_imgs_shape,output_ccf_shape
    original_imgs = _original_imgs
    badpix_imgs = _badpix_imgs
    output_ccf = _output_ccf
    original_imgs_shape = _original_imgs_shape
    badpix_imgs_shape = _badpix_imgs_shape
    output_ccf_shape = _output_ccf_shape


def _CCF_thread(row_index,R_list,dwvs,wvs,skybg_wvs_list,skybg_spec_list,medwindowsize):
    global original_imgs,badpix_imgs,output_ccf, original_imgs_shape, badpix_imgs_shape,output_ccf_shape
    original_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
    badpix_np = _arraytonumpy(badpix_imgs, badpix_imgs_shape,dtype=dtype)
    output_ccf_np = _arraytonumpy(output_ccf, output_ccf_shape,dtype=dtype)

    badpix_np[0:medwindowsize//2,:,:] = np.nan
    # badpix_np[(badpix_imgs_shape[0]-medwindowsize//2)::,:,:] = np.nan
    badpix_np[np.where(wvs>2.25)[0],:,:] = np.nan

    for k,(skybg_wvs,skybg_spec) in enumerate(zip(skybg_wvs_list,skybg_spec_list)):
        for l,R in enumerate(R_list):
            skybg_spec = convolve_spectrum(skybg_wvs,skybg_spec,R)
            hd_spec_func = interp1d(skybg_wvs,skybg_spec/np.nanstd(skybg_spec),bounds_error=False,fill_value=0)

            for m in range(original_imgs_shape[2]):
            # if 1:
            #     m= 15
                myvec = original_np[:,row_index,m]
                myvec_bad_pix = badpix_np[:,row_index,m]
                if np.sum(np.isfinite(myvec_bad_pix)) == 0:
                    output_ccf_np[k,l,:,row_index,m]=np.nan
                    continue
                sky_LPF,sky_HPF = LPFvsHPF_median(myvec,medwindowsize)
                where_badpix = np.where(np.isnan(myvec_bad_pix))
                myvec[where_badpix] = sky_LPF[where_badpix]
                sky_LPF,sky_HPF = LPFvsHPF_median(myvec,medwindowsize)
                sky_HPF[where_badpix] = np.nan
                for n,dwv in enumerate(dwvs):
                    hd_spec = hd_spec_func(wvs-dwv)
                    hd_spec_LPF,hd_spec_HPF = LPFvsHPF_median(hd_spec,cutoff)
                    output_ccf_np[k,l,n,row_index,m]=np.nansum(sky_HPF*hd_spec_HPF)**2/np.nansum(hd_spec_HPF[where_badpix]**2)

                # hdwvs = np.linspace(wvs[0],wvs[-1],5000)
                # plt.subplot(1,2,1)
                # plt.plot(wvs,sky_HPF/np.nanstd(sky_HPF),label="data")
                # plt.plot(hdwvs,hd_spec_func(hdwvs)/np.nanstd(hd_spec_func(hdwvs)),label="OH")
                # plt.legend()
                # plt.subplot(1,2,2)
                # plt.plot(dwvs,output_ccf_np[k,l,:,row_index,m])
                # plt.show()
                # break


def _remove_bad_pixels_xy(wvs_indices,dtype):
    global original_imgs,badpix_imgs,output_ccf, original_imgs_shape, badpix_imgs_shape,output_ccf_shape
    original_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
    badpix_np = _arraytonumpy(badpix_imgs, badpix_imgs_shape,dtype=dtype)
    for k in wvs_indices:
        # tmp = original_np[:,:,k]
        tmpcopy= copy(original_np[k,:,:])
        tmpbadpix = badpix_np[k,:,:]
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

def _remove_edges(wvs_indices,nan_mask_boxsize,dtype):
    global original_imgs,badpix_imgs,output_ccf, original_imgs_shape, badpix_imgs_shape,output_ccf_shape
    badpix_np = _arraytonumpy(badpix_imgs, badpix_imgs_shape,dtype=dtype)
    for k in wvs_indices:
        tmp = badpix_np[k,:,:]
        tmp[np.where(np.isnan(correlate2d(tmp,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
    badpix_np[0:nan_mask_boxsize//2,:,:] = np.nan
    badpix_np[-nan_mask_boxsize//2+1::,:,:] = np.nan
    badpix_np[:,0:nan_mask_boxsize//2,:] = np.nan
    badpix_np[:,-nan_mask_boxsize//2+1::,:] = np.nan


# nice -n 15 /home/anaconda3/bin/python ./calibrate_OSIRIS.py

if 1:
    IFSfilter = "Kbb"
    if IFSfilter=="Kbb": #Kbb 1965.0 0.25
        CRVAL1 = 1965.
        CDELT1 = 0.25
        nl=1665
        R=3000
    elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1473.
        CDELT1 = 0.2
        nl=1651
        R=5000
    init_wv = CRVAL1/1000.
    dwv = CDELT1/1000.
    wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)

    cutoff = 40
    medwindowsize=200
    nccf = 21 #11
    # nccf = 41
    dwvs_CCF = np.linspace(-dwv*1,dwv*1,nccf)
    # dwvs_CCF = np.linspace(-dwv*0.1,dwv*0.1,nccf)
    # R_list = np.linspace(1000,5000,16+1)
    # R_list = np.linspace(2000,6000,4+1)
    R_list = np.array([R,])
    debug = False
    numthreads = 28
    suffix="_Rfixed"


    # inputdir = "/home/sda/jruffio/osiris_data/HR_8799_d"
    inputdir = "/home/sda/jruffio/osiris_data/HR_8799_c"
    # filename = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_sky_jb/s100715_a001001_Kbb_020.fits"
    # filelist = glob.glob("/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_sky_jb/s*_Kbb_020.fits")
    # filelist = glob.glob("/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_sky_Kbb/s*_Kbb_020.fits")
    filelist = glob.glob(os.path.join(inputdir,"*/reduced_sky_jb_Kbb/s*_Kbb_020.fits"))

if 1:
    # lambdas_air = lambdas_vac/(1+2.735182e-4+131.4182/lambdas_vac**2+2.76249e8/lambdas_vac**4)
    if 1:
        skybg_spec_list = []
        skybg_wvs_list = []
        skyem_filelist = glob.glob("/home/sda/jruffio/osiris_data/sky_emission/mk_skybg_zm_*_ph.dat")[::3]
        for filename in skyem_filelist:#filename = "/home/sda/jruffio/osiris_data/sky_emission/mk_skybg_zm_10_10_ph.dat"
            print(filename)
            skybg_arr=np.loadtxt(filename)
            skybg_wvs = skybg_arr[:,0]/1000.
            skybg_spec = skybg_arr[:,1]
            selec_skybg = np.where((skybg_wvs>wvs[0]-(wvs[-1]-wvs[0])/2)*(skybg_wvs<wvs[-1]+(wvs[-1]-wvs[0])/2))
            skybg_wvs = skybg_wvs[selec_skybg]
            skybg_spec = skybg_spec[selec_skybg]
            skybg_wvs_list.append(skybg_wvs)
            skybg_spec_list.append(skybg_spec)
        #     plt.plot(skybg_wvs,skybg_spec,label=os.path.basename(filename))
        #     # skybg_spec = convolve_spectrum(skybg_wvs,skybg_spec[selec_skybg],R)
        #     # skybg_spec_LPF,skybg_spec_HPF = LPFvsHPF(skybg_spec,cutoff)
        # plt.legend()
        # plt.show()
    else:
        filename = "/home/sda/jruffio/osiris_data/sky_emission/mk_skybg_zm_50_20_ph.dat"
        skybg_arr=np.loadtxt(filename)
        skybg_wvs = skybg_arr[:,0]/1000.
        skybg_spec = skybg_arr[:,1]
        selec_skybg = np.where((skybg_wvs>wvs[0]-(wvs[-1]-wvs[0])/2)*(skybg_wvs<wvs[-1]+(wvs[-1]-wvs[0])/2))
        skybg_wvs = skybg_wvs[selec_skybg]
        skybg_spec = skybg_spec[selec_skybg]
        # skybg_spec = convolve_spectrum(skybg_wvs,skybg_spec,R)


    ccf_arr_list = []
    wvshift_arr_list = []
    for filename in filelist:
        print(filename)
        # continue
        hdulist = pyfits.open(filename)
        prihdr = hdulist[0].header
        skycube = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        skycube_badpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
        nz,ny,nx = skycube.shape
        print(skycube.shape)

        if 1:
            import ctypes
            dtype = ctypes.c_float
            original_imgs = mp.Array(dtype, np.size(skycube))
            original_imgs_shape = skycube.shape
            original_imgs_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
            original_imgs_np[:] = skycube
            badpix_imgs = mp.Array(dtype, np.size(skycube_badpix))
            badpix_imgs_shape = skycube_badpix.shape
            badpix_imgs_np = _arraytonumpy(badpix_imgs, badpix_imgs_shape,dtype=dtype)
            badpix_imgs_np[:] = 0#skycube_badpix
            badpix_imgs_np[np.where(original_imgs_np==0)] = np.nan
            output_ccf = mp.Array(dtype, len(skybg_spec_list)*np.size(R_list)*nccf*ny*nx)
            output_ccf_shape = (len(skybg_spec_list),np.size(R_list),nccf,ny,nx)
            output_ccf_np = _arraytonumpy(output_ccf, output_ccf_shape,dtype=dtype)
            output_ccf_np[:] = np.nan

            tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                            initargs=(original_imgs,badpix_imgs,output_ccf, original_imgs_shape, badpix_imgs_shape,
                                      output_ccf_shape),
                            maxtasksperchild=50)

            chunk_size = nz//(3*numthreads)
            N_chunks = nz//chunk_size
            wvs_indices_list = []
            for k in range(N_chunks-1):
                wvs_indices_list.append(np.arange((k*chunk_size),((k+1)*chunk_size)))
            wvs_indices_list.append(np.arange(((N_chunks-1)*chunk_size),nz))

            tasks = [tpool.apply_async(_remove_bad_pixels_xy, args=(wvs_indices,dtype))
                     for wvs_indices in wvs_indices_list]
            #save it to shared memory
            for chunk_index, rmedge_task in enumerate(tasks):
                print("Finished rm bad pixel xy chunk {0}".format(chunk_index))
                rmedge_task.wait()

            # nan_mask_boxsize = 3
            # tasks = [tpool.apply_async(_remove_edges, args=(wvs_indices,nan_mask_boxsize,dtype))
            #          for wvs_indices in wvs_indices_list]
            # #save it to shared memory
            # for chunk_index, rmedge_task in enumerate(tasks):
            #     print("Finished rm edge chunk {0}".format(chunk_index))
            #     rmedge_task.wait()

        if 0:
            tpool.close()
            # skycube[np.where(np.isnan(badpix_imgs_np))] = np.nan
            myvec = np.nanmean(skycube,axis=(1,2))
            # myvec = skycube[:,16,7]
            myvec_bad_pix = badpix_imgs_np[:,16,7]
            sky_LPF,sky_HPF = LPFvsHPF_median(myvec,medwindowsize)
            where_badpix = np.where(np.isnan(myvec_bad_pix))
            myvec[where_badpix] = sky_LPF[where_badpix]
            sky_LPF,sky_HPF = LPFvsHPF_median(myvec,medwindowsize)
            sky_HPF[where_badpix] = np.nan
            plt.figure(1)
            plt.plot(wvs,myvec,label="sky spec")
            plt.plot(wvs,sky_HPF,label="sky HPF")
            plt.plot(wvs,sky_LPF,label="sky LPF")
            # plt.subplot(1,3,1)
            # plt.plot(dwvs_CCF/dwv,CCF(dwvs_CCF,wvs,sky_HPF/np.nanstd(sky_HPF),skybg_wvs,skybg_spec/np.nanstd(skybg_spec)))
            # plt.subplot(1,3,2)
            plt.plot(skybg_wvs,skybg_spec/np.nanstd(skybg_spec)*np.nanstd(sky_HPF),label="emission spec")
            # for skybg_spec in skybg_spec_list:
            #     plt.plot(skybg_wvs,skybg_spec/np.nanstd(skybg_spec),label="transmission spec")
            plt.legend()
            # plt.subplot(1,3,3)
            # plt.plot(sky_HPF)
            plt.figure(2)
            plt.subplot(1,3,1)
            plt.imshow(skycube[321,:,:],interpolation="nearest")
            plt.subplot(1,3,2)
            original_imgs_np[np.where(np.isnan(badpix_imgs_np))] = np.nan
            plt.imshow(original_imgs_np[321,:,:])
            plt.subplot(1,3,3)
            plt.imshow(original_imgs_np[325,:,:])
            plt.show()

        # ccf_arr_list.append(CCF(dwvs_CCF,wvs,sky_HPF/np.nanstd(sky_HPF),skybg_wvs,skybg_spec/np.nanstd(skybg_spec)))

        if 1:


            if debug:
                _tpool_init(original_imgs,badpix_imgs,output_ccf, original_imgs_shape, badpix_imgs_shape,
                                          output_ccf_shape)
                _CCF_thread(37,R_list,dwvs_CCF,wvs,skybg_wvs_list,skybg_spec_list,medwindowsize)
                exit()
            else:
                # tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                #                 initargs=(original_imgs,badpix_imgs,output_ccf, original_imgs_shape, badpix_imgs_shape,
                #                           output_ccf_shape),
                #                 maxtasksperchild=50)

                tasks = [tpool.apply_async(_CCF_thread, args=(row_index,R_list,dwvs_CCF,wvs,skybg_wvs_list,skybg_spec_list,medwindowsize))
                         for row_index in range(ny)]
                #save it to shared memory
                for col_index, task in enumerate(tasks):
                    print("Finished row {0}".format(col_index))
                    task.wait()

            ccf_arr_list.append(output_ccf_np)

            wvshift_arr = np.zeros((ny,nx))
            R_arr = np.zeros((ny,nx))
            bestmodel_arr = np.zeros((ny,nx))
            for row in range(ny):
                for col in range(nx):
                    chi2_allpara = output_ccf_np[:,:,:,row,col]
                    # print(row,col,chi2_allpara)
                    # print(output_ccf_shape[0:3],np.argmax(chi2_allpara))
                    unravel_index_argmax = np.unravel_index(np.argmax(chi2_allpara),output_ccf_shape[0:3])
                    # print(unravel_index_argmax)
                    bestmodel_arr[row,col] = unravel_index_argmax[0]
                    R_arr[row,col] = R_list[unravel_index_argmax[1]]
                    wvshift_arr[row,col] = dwvs_CCF[unravel_index_argmax[2]]/dwv

            # print(np.argmax(output_ccf_np,axis=(0,1,2)))
            # output_ccf_argmax = np.unravel_index(np.argmax(output_ccf_np,axis=(0,1,2)),output_ccf_shape[0:3])
            # print(output_ccf_argmax)
            # exit()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=output_ccf_np))
            try:
                hdulist.writeto(filename.replace(".fits","_OHccf"+suffix+"_output.fits"), overwrite=True)
            except TypeError:
                hdulist.writeto(filename.replace(".fits","_OHccf"+suffix+"_output.fits"), clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=bestmodel_arr))
            try:
                hdulist.writeto(filename.replace(".fits","_OHccf"+suffix+"_model.fits"), overwrite=True)
            except TypeError:
                hdulist.writeto(filename.replace(".fits","_OHccf"+suffix+"_model.fits"), clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=R_arr))
            try:
                hdulist.writeto(filename.replace(".fits","_OHccf"+suffix+"_R.fits"), overwrite=True)
            except TypeError:
                hdulist.writeto(filename.replace(".fits","_OHccf"+suffix+"_R.fits"), clobber=True)
            hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=wvshift_arr))
            try:
                hdulist.writeto(filename.replace(".fits","_OHccf"+suffix+"_dwv.fits"), overwrite=True)
            except TypeError:
                hdulist.writeto(filename.replace(".fits","_OHccf"+suffix+"_dwv.fits"), clobber=True)
            hdulist.close()

            tpool.close()


if 1:
    # filename_filter = "/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_sky_Kbb/s*_Kbb_020.fits"
    filename_filter = os.path.join(inputdir,"*/reduced_sky_jb_Kbb/s*_Kbb_020.fits")
    filelist_out = glob.glob(filename_filter.replace(".fits","_OHccf"+suffix+"_output.fits"))
    filelist_R = glob.glob(filename_filter.replace(".fits","_OHccf"+suffix+"_R.fits"))
    filelist_dwv = glob.glob(filename_filter.replace(".fits","_OHccf"+suffix+"_dwv.fits"))
    filelist_model = glob.glob(filename_filter.replace(".fits","_OHccf"+suffix+"_model.fits"))

    wvshift_arr_list = []
    for k,(filename_R,filename_dwv,filename_model,filename_out) in enumerate(zip(filelist_R,filelist_dwv,filelist_model,filelist_out)):
        hdulist = pyfits.open(filename_out)
        output = hdulist[0].data


        plt.subplot(3,len(filelist_R),k+1)
        hdulist = pyfits.open(filename_R)
        R_arr = hdulist[0].data
        plt.imshow(R_arr,interpolation="nearest")
        plt.clim([2000,6000])
        plt.colorbar()

        plt.subplot(3,len(filelist_R),k+1+len(filelist_R))
        if 0:
            hdulist = pyfits.open(filename_dwv)
            wvshift_arr = hdulist[0].data
        else:
            wvshift_arr = np.zeros(output.shape[3::])
            for row in range(output.shape[3]):
                for col in range(output.shape[4]):
                    chi2_allpara = output[3,0,:,row,col]
                    wvshift_arr[row,col] = dwvs_CCF[np.argmax(chi2_allpara)]
        wvshift_arr_list.append(wvshift_arr[None,:,:])
        plt.imshow(wvshift_arr/dwv,interpolation="nearest")
        plt.clim([-0.5,0.5])
        plt.colorbar()

        plt.subplot(3,len(filelist_R),k+1+2*len(filelist_R))
        hdulist = pyfits.open(filename_model)
        bestmodel_arr = hdulist[0].data
        plt.imshow(bestmodel_arr,interpolation="nearest")
        plt.clim([0,12])
        plt.colorbar()


    master_wvshift_arr = np.nanmedian(np.concatenate(wvshift_arr_list,axis=0),axis=0)
    master_wvshift_arr[np.where(np.abs(master_wvshift_arr)==1)]=np.nan
    plt.figure(2)
    plt.imshow(master_wvshift_arr/dwv,interpolation="nearest")
    plt.clim([-0.5,0.5])
    plt.colorbar()


    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=master_wvshift_arr))
    try:
        hdulist.writeto(os.path.join(inputdir,"master_wvshifts.fits"), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(inputdir,"master_wvshifts.fits"), clobber=True)
    hdulist.close()

    plt.show()



    # plt.subplot(1,4,1)
    # plt.imshow(wvshift_arr_list[0]/dwv,interpolation="nearest")
    # plt.clim([-0.5,0.5])
    # plt.colorbar()
    # plt.subplot(1,4,2)
    # plt.imshow(wvshift_arr_list[1]/dwv,interpolation="nearest")
    # plt.clim([-0.5,0.5])
    # plt.colorbar()
    # plt.subplot(1,4,3)
    # plt.imshow(wvshift_arr_list[2]/dwv,interpolation="nearest")
    # plt.clim([-0.5,0.5])
    # plt.colorbar()
    # plt.subplot(1,4,4)
    # plt.imshow(wvshift_arr_list[3]/dwv,interpolation="nearest")
    # plt.clim([-0.5,0.5])
    # plt.colorbar()
    # plt.show()
    #
    # exit()
    # plt.plot(wvs,sky_HPF/np.nanstd(sky_HPF),label=os.path.basename(filename))
    #
    # plt.plot(skybg_wvs,skybg_spec/np.nanstd(skybg_spec))
    # plt.legend()
    #
    # plt.figure(2)
    # for ccf_arr,filename in zip(ccf_arr_list,filelist):
    #     plt.plot(dwvs_CCF/dwv,ccf_arr,label=os.path.basename(filename))
    # plt.legend()
    plt.show()