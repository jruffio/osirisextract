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
from reduce_HPFonly_diagcov_resmodel_v2 import return_64x19

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
    # # badpix_np[(badpix_imgs_shape[0]-medwindowsize//2)::,:,:] = np.nan
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
                where_validpix = np.where(np.isfinite(myvec_bad_pix))
                for n,dwv in enumerate(dwvs):
                    hd_spec = hd_spec_func(wvs-dwv)
                    hd_spec_LPF,hd_spec_HPF = LPFvsHPF_median(hd_spec,cutoff)
                    output_ccf_np[k,l,n,row_index,m]=np.nansum(sky_HPF*hd_spec_HPF)**2/np.nansum(hd_spec_HPF[where_validpix]**2)
                    # print(output_ccf_np[k,l,n,row_index,m])

                # hdwvs = np.linspace(wvs[0],wvs[-1],5000)
                # plt.subplot(1,3,1)
                # plt.plot(wvs,sky_HPF/np.nanstd(sky_HPF),label="data")
                # plt.plot(hdwvs,hd_spec_func(hdwvs)/np.nanstd(hd_spec_func(hdwvs)),label="OH")
                # plt.legend()
                # plt.subplot(1,3,2)
                # plt.plot(dwvs,output_ccf_np[k,l,:,row_index,m])
                # plt.subplot(1,3,3)
                # plt.plot(np.linspace(-1,1,np.size(dwvs)),output_ccf_np[k,l,:,row_index,m])
                # plt.show()
                # # break


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

######
## Step 1
# run the cross correlation with the OH template
if 1:
    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
    IFSfilter = "Kbb"
    # IFSfilter = "Hbb"
    # inputdir = "/data/osiris_data/HR_8799_d"
    # inputdir = "/data/osiris_data/HR_8799_c"
    # filelist = glob.glob(os.path.join(inputdir,"20100715/reduced_sky_jb/s*_Kbb_020.fits"))
    # filelist = glob.glob(os.path.join(inputdir,"20101104/reduced_sky_jb/s*_Kbb_020.fits"))
    # inputdir = "/data/osiris_data/HR_8799_b"
    # filelist = glob.glob(os.path.join(inputdir,"201007/reduced_sky_jb/s*_Kbb_020.fits"))
    # inputdir = "/data/osiris_data/HR_8799_*"
    # filelist = glob.glob(os.path.join(inputdir,"2010*/reduced_sky_jb/s*_"+IFSfilter+"_[0-9][0-9][0-9].fits"))
    # inputdir = "/data/osiris_data/kap_And"
    # inputdir = "/data/osiris_data/kap_And"
    # filelist = glob.glob(os.path.join(inputdir,"2017*/reduced_sky_jb/s*_"+IFSfilter+"_[0-9][0-9][0-9].fits"))
    # inputdir = "/data/osiris_data/51_Eri_b"
    # filelist = glob.glob(os.path.join(inputdir,"2017*/reduced_sky_jb/s*_"+IFSfilter+"_[0-9][0-9][0-9].fits"))
    # inputdir = "/data/osiris_data/HR_8799_b"
    # filelist = glob.glob(os.path.join(inputdir,"2009*/reduced_sky_jb/s*_"+IFSfilter+"_[0-9][0-9][0-9].fits"))
    # inputdir = "/data/osiris_data/HR_8799_d"
    # filelist = glob.glob(os.path.join(inputdir,"2015*/reduced_sky_jb/s*_"+IFSfilter+"_[0-9][0-9][0-9].fits"))
    # filelist = glob.glob(os.path.join(inputdir,"20200730*/reduced_sky_jb/s*_"+IFSfilter+"_[0-9][0-9][0-9].fits"))
    inputdir = "/data/osiris_data/GJ_504_b"
    filelist = glob.glob(os.path.join(inputdir,"2019*/reduced_sky_jb/s*_"+IFSfilter+"_[0-9][0-9][0-9].fits"))
    # inputdir = "/data/osiris_data/HR_8799_c"
    # filelist = glob.glob(os.path.join(inputdir,"202010*/reduced_sky_jb/s*_"+IFSfilter+"_[0-9][0-9][0-9].fits"))
    # inputdir = "/data/osiris_data/HD_1160"
    # filelist = glob.glob(os.path.join(inputdir,"2018*/reduced_sky_jb/s*_"+IFSfilter+"_[0-9][0-9][0-9].fits"))
    # filelist = [filelist[1]]
    print(filelist)
    # exit()

    if IFSfilter=="Kbb": #Kbb 1965.0 0.25
        CRVAL1 = 1965.
        CDELT1 = 0.25
        nl=1665
        R=4000
    elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1473.
        CDELT1 = 0.2
        nl=1651
        R=4000
    init_wv = CRVAL1/1000.
    dwv = CDELT1/1000.
    wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)
    dprv = 3e5*dwv/(init_wv+dwv*nl//2)

    cutoff = 40
    medwindowsize=200
    nccf = 401 #11
    # nccf = 41
    dwvs_CCF = np.linspace(-dwv*2,dwv*2,nccf)
    # dwvs_CCF = np.linspace(-dwv*0.1,dwv*0.1,nccf)
    # R_list = np.linspace(1000,5000,16+1)
    # R_list = np.linspace(2000,6000,4+1)
    R_list = np.array([R,])
    debug = False
    numthreads = 28
    suffix="_Rfixed"

    # lambdas_air = lambdas_vac/(1+2.735182e-4+131.4182/lambdas_vac**2+2.76249e8/lambdas_vac**4)
    if 1:
        skybg_spec_list = []
        skybg_wvs_list = []
        skyem_filelist = glob.glob("/data/osiris_data/sky_emission/mk_skybg_zm_10_10_ph.dat")[0:1]
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
        filename = "/data/osiris_data/sky_emission/mk_skybg_zm_50_20_ph.dat"
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

        # Plotting section?????? Not sure I should comment it out
        if 0:
            original_np = _arraytonumpy(original_imgs, original_imgs_shape,dtype=dtype)
            badpix_np = _arraytonumpy(badpix_imgs, badpix_imgs_shape,dtype=dtype)
            output_ccf_np = _arraytonumpy(output_ccf, output_ccf_shape,dtype=dtype)

            badpix_np[0:medwindowsize//2,:,:] = np.nan
            # # badpix_np[(badpix_imgs_shape[0]-medwindowsize//2)::,:,:] = np.nan
            badpix_np[np.where(wvs>2.25)[0],:,:] = np.nan

            my_output_ccf = np.zeros((np.size(skybg_spec_list),np.size(dwvs_CCF),3))
            myhpfvec_list = []
            for k,(skybg_wvs,skybg_spec) in enumerate(zip(skybg_wvs_list,skybg_spec_list)):
                skybg_spec = convolve_spectrum(skybg_wvs,skybg_spec,R)
                hd_spec_func = interp1d(skybg_wvs,skybg_spec/np.nanstd(skybg_spec),bounds_error=False,fill_value=0)

                for myid,(row_index,m) in enumerate(zip([10,20,30],[10,10,10])):
                    print(myid)
                    myvec = original_np[:,row_index,m]
                    myvec_bad_pix = badpix_np[:,row_index,m]
                    sky_LPF,sky_HPF = LPFvsHPF_median(myvec,medwindowsize)
                    where_badpix = np.where(np.isnan(myvec_bad_pix))
                    myvec[where_badpix] = sky_LPF[where_badpix]
                    sky_LPF,sky_HPF = LPFvsHPF_median(myvec,medwindowsize)
                    sky_HPF[where_badpix] = np.nan
                    where_validpix = np.where(np.isfinite(myvec_bad_pix))
                    myhpfvec_list.append(sky_HPF)
                    for n,dwvccf in enumerate(dwvs_CCF):
                        hd_spec = hd_spec_func(wvs-dwvccf)
                        hd_spec_LPF,hd_spec_HPF = LPFvsHPF_median(hd_spec,cutoff)
                        my_output_ccf[k,n,myid]=np.nansum(sky_HPF*hd_spec_HPF)**2/np.nansum(hd_spec_HPF[where_validpix]**2)

            # skycube[np.where(np.isnan(badpix_imgs_np))] = np.nan
            myvec = np.nanmean(skycube,axis=(1,2))
            # myvec = skycube[:,16,7]
            myvec_bad_pix = badpix_imgs_np[:,16,7]
            sky_LPF,sky_HPF = LPFvsHPF_median(myvec,medwindowsize)
            where_badpix = np.where(np.isnan(myvec_bad_pix))
            myvec[where_badpix] = sky_LPF[where_badpix]
            sky_LPF,sky_HPF = LPFvsHPF_median(myvec,medwindowsize)
            sky_HPF[where_badpix] = np.nan

            fontsize = 12
            plt.figure(1,figsize=(7,5))
            plt.subplot(2,1,1)
            plt.plot(wvs,myvec,label="Combined",color="#ff9900")
            plt.plot(wvs,sky_LPF,label="Combined LPF",color="#6600ff",linestyle=":")
            plt.plot(wvs,sky_HPF,label="Combined HPF",color="#0099cc",linestyle="--")
            skybg_spec = convolve_spectrum(skybg_wvs,skybg_spec,R)
            hd_spec_func = interp1d(skybg_wvs,skybg_spec/np.nanstd(skybg_spec),bounds_error=False,fill_value=0)
            hd_spec = hd_spec_func(wvs-0)
            hd_spec_LPF,hd_spec_HPF = LPFvsHPF_median(hd_spec,cutoff)
            where_validpix = np.where(np.isfinite(myvec_bad_pix))
            plt.plot(wvs,hd_spec_HPF*np.nansum(sky_HPF*hd_spec_HPF)/np.nansum(hd_spec_HPF[where_validpix]**2),label="Sky model",color="black")
            plt.legend(loc="upper center",frameon=True,fontsize=fontsize)#,ncol=4)#
            if IFSfilter == "Kbb":
                plt.xlim([1.99,2.25])
            elif IFSfilter == "Hbb":
                plt.xlim([1.47,1.8])
            plt.tick_params(axis="x",labelsize=fontsize)
            # plt.xlabel("$\lambda (\mu\mathrm{m})$",fontsize=fontsize)
            plt.ylabel("$\propto$ ADU",fontsize=fontsize)
            plt.tick_params(axis="y",which="both",labelleft=False,bottom=False,top=False)

            plt.subplot(2,1,2)
            color_list = ["#ff9900","#0099cc","#6600ff"]
            linestyle_list = ["-","--",":"]
            for myid,color,linestyle in zip(np.arange(3),color_list[::-1],linestyle_list[::-1]):
                plt.plot(wvs,myhpfvec_list[myid],label="Sample {0} HPF".format(myid),color=color,linestyle=linestyle)
            plt.legend(loc="upper center",frameon=True,fontsize=fontsize)#,ncol=3)
            plt.tick_params(axis="x",labelsize=fontsize)
            plt.xlabel("$\lambda (\mu\mathrm{m})$",fontsize=fontsize)
            plt.ylabel("$\propto$ ADU",fontsize=fontsize)
            plt.tick_params(axis="y",which="both",labelleft=False,bottom=False,top=False)
            if IFSfilter == "Kbb":
                plt.xlim([1.99,2.25])
            elif IFSfilter == "Hbb":
                plt.xlim([1.47,1.8])

            if 0:
                print("Saving "+os.path.join(out_pngs,"sky_emission_"+IFSfilter+".pdf"))
                plt.savefig(os.path.join(out_pngs,"sky_emission_"+IFSfilter+".pdf"),bbox_inches='tight')
                plt.savefig(os.path.join(out_pngs,"sky_emission_"+IFSfilter+".png"),bbox_inches='tight')

            plt.figure(2,figsize=(4,5))
            for myid,color,linestyle in zip(np.arange(3),color_list[::-1],linestyle_list[::-1]):
                myargmax = np.argmax(my_output_ccf[0,:,myid])
                plt.plot(dwvs_CCF/dwv,my_output_ccf[0,:,myid]/np.max(my_output_ccf[0,:,myid]),label="Sample {0}".format(myid),color=color,linestyle=linestyle,linewidth=2)
                plt.plot([dwvs_CCF[myargmax]/dwv,dwvs_CCF[myargmax]/dwv],[0,1],color=color)
                plt.gca().text(dwvs_CCF[myargmax]/dwv-0.02,0.05+0.25*myid,"{0:.2f} pix \n {1:.2f} $\AA$ \n {2:.2f} km/s".format(dwvs_CCF[myargmax]/dwv,dwvs_CCF[myargmax]*10000.,dwvs_CCF[myargmax]/dwv*dprv),ha="right",va="bottom",rotation=0,size=fontsize,color=color)
            plt.tick_params(axis="y",labelsize=fontsize)
            plt.tick_params(axis="x",labelsize=fontsize)
            plt.ylabel("CCF",fontsize=fontsize)
            plt.xlabel("Pixel",fontsize=fontsize)
            plt.xlim([-1,1])
            plt.ylim([0,1.1])

            if 0:
                print("Saving "+os.path.join(out_pngs,"sky_emission_CCF_"+IFSfilter+".pdf"))
                plt.savefig(os.path.join(out_pngs,"sky_emission_CCF_"+IFSfilter+".pdf"),bbox_inches='tight')
                plt.savefig(os.path.join(out_pngs,"sky_emission_CCF_"+IFSfilter+".png"),bbox_inches='tight')
            #
            # skybg_spec = convolve_spectrum(skybg_wvs,skybg_spec,R)
            # hd_spec_func = interp1d(skybg_wvs,skybg_spec/np.nanstd(skybg_spec),bounds_error=False,fill_value=0)
            # where_validpix = np.where(np.isfinite(myvec_bad_pix))
            # output_ccf_np=np.zeros(dwvs_CCF.shape)
            # for n,dwv in enumerate(dwvs_CCF):
            #     hd_spec = hd_spec_func(wvs-dwv)
            #     hd_spec_LPF,hd_spec_HPF = LPFvsHPF_median(hd_spec,cutoff)
            #     output_ccf_np[n]=np.nansum(sky_HPF*hd_spec_HPF)**2/np.nansum(hd_spec_HPF[where_validpix]**2)
            # plt.figure(3)
            # plt.subplot(1,2,1)
            # plt.plot(dwvs_CCF,output_ccf_np)
            # plt.subplot(1,2,2)
            # plt.plot(dwvs_CCF/dwv,output_ccf_np)

            tpool.close()
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


# if 1:
#     IFSfilter = "*"
#     inputdir = "/data/osiris_data/HR_8799_c"
#     filename_filter = os.path.join(inputdir,"*/reduced_sky_jb/s*_"+IFSfilter+"_020.fits")
#     # filename_filter = os.path.join(inputdir,"*/reduced_jb/s*_"+IFSfilter+"_020.fits")
#     filelist = glob.glob(filename_filter)
#     print(len(filelist))
#     k = 0
#     for filename in filelist:
#         k+=1
#         hdulist = pyfits.open(filename)
#         data = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
#         print(data.shape)
#         data = return_64x19(data)
#         plt.subplot(5,5,k)
#         mask = copy(data)
#         mask[np.where(data==0)]=np.nan
#         mask[np.where(np.isfinite(mask))]=1
#         im = np.nansum(mask,axis=0)
#         plt.imshow(im,interpolation="nearest")
#     plt.show()
#     exit()

#####################
## Step 2
# exit()
if 1:
    std_factor = np.zeros(20)
    for k in np.arange(2,20):
        a = np.random.randn(1000,k)
        std_factor[k] = np.std(a - np.mean(a,axis=1)[:,None])/np.std(a)
    print(std_factor)

    #Kbb
    #20100711 b = 20100712 b = 20100715 c = 20101104 c
    #20110723 c = 20110724 c = 20110725 c
    #20130725 b = 20130726 b = 20130727 b
    #2015 d
    #20161106 b = 20161107 b= 20161108 b + kap and
    #20171103 c + kap and + 51 eri
    #20180722 b
    #Hbb
    #20100713 b  = 20101028 c = 20101104 c
    #20110724 c = 20110725 c
    #20171103 c
    suffix="_Rfixed"
    IFSfilter = "Kbb"
    # IFSfilter = "Hbb"
    year = 2019
    # scale = "020"
    scale = "050"
    # scale = "035"
    # inputdir = "/data/osiris_data/HR_8799_*"
    filename_filter_list = []
    # filename_filter = os.path.join(inputdir,"2010*/reduced_sky_jb/s*_"+IFSfilter+"_020.fits")
    # filename_filter = "/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_sky_Kbb/s*_Kbb_020.fits"
    # filename_filter = os.path.join(inputdir,"20101104/reduced_sky_jb/s*_Kbb_020.fits")
    # filename_filter = os.path.join(inputdir,"2013*/reduced_sky_jb/s*_Kbb_020.fits")
    inputdir = "/data/osiris_data/51_Eri_b"
    filename_filter_list.append(os.path.join(inputdir,"{0}*/reduced_sky_jb/s*_".format(year)+IFSfilter+"_"+scale+".fits"))
    inputdir = "/data/osiris_data/HR_8799_*"
    filename_filter_list.append(os.path.join(inputdir,"{0}*/reduced_sky_jb/s*_".format(year)+IFSfilter+"_"+scale+".fits"))
    inputdir = "/data/osiris_data/kap_And"
    filename_filter_list.append(os.path.join(inputdir,"{0}*/reduced_sky_jb/s*_".format(year)+IFSfilter+"_"+scale+".fits"))
    inputdir = "/data/osiris_data/GJ_504_b"
    filename_filter_list.append(os.path.join(inputdir,"{0}*/reduced_sky_jb/s*_".format(year)+IFSfilter+"_"+scale+".fits"))
    inputdir = "/data/osiris_data/HD_1160"
    filename_filter_list.append(os.path.join(inputdir,"{0}*/reduced_sky_jb/s*_".format(year)+IFSfilter+"_"+scale+".fits"))
    filename_filter_list.sort()
    filelist = []
    filelist_out = []
    filelist_R = []
    filelist_dwv = []
    filelist_model = []
    for filename_filter in filename_filter_list:
        filelist = filelist+glob.glob(filename_filter)
        filelist_out = filelist_out+glob.glob(filename_filter.replace(".fits","_OHccf"+suffix+"_output.fits"))
        filelist_R = filelist_R+glob.glob(filename_filter.replace(".fits","_OHccf"+suffix+"_R.fits"))
        filelist_dwv = filelist_dwv+glob.glob(filename_filter.replace(".fits","_OHccf"+suffix+"_dwv.fits"))
        filelist_model = filelist_model+glob.glob(filename_filter.replace(".fits","_OHccf"+suffix+"_model.fits"))

    filelist.sort()
    for fstr in filelist:
        print(fstr)
    # print(filelist)
    # exit()
    filelist_out.sort()
    filelist_R.sort()
    filelist_dwv.sort()
    filelist_model.sort()
    print(filelist_out)

    if IFSfilter=="Kbb": #Kbb 1965.0 0.25
        CRVAL1 = 1965.
        CDELT1 = 0.25
        nl=1665
        R=4000
    elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1473.
        CDELT1 = 0.2
        nl=1651
        R=4000
    init_wv = CRVAL1/1000.
    dwv = CDELT1/1000.
    wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)
    dprv = 3e5*dwv/(init_wv+dwv*nl//2)

    dwv_map_list = []
    for k,(filename_R,filename_dwv,filename_model,filename_out) in enumerate(zip(filelist_R,filelist_dwv,filelist_model,filelist_out)):
        print(filename_dwv)
        if k in [3,4,5,7,8]:
            continue
        hdulist = pyfits.open(filename_dwv)
        dwv_map = hdulist[0].data
        dwv_map[np.where(np.abs(dwv_map)>0.75)] = np.nan
        dwv_map = return_64x19(dwv_map)
        dwv_map_list.append(dwv_map)
    if len(dwv_map_list) <=2:
        master_wvshift = np.nanmean(dwv_map_list,axis=0)
    else:
        print([dwv_map.shape for dwv_map in dwv_map_list])
        master_wvshift = np.nanmedian(dwv_map_list,axis=0)
    master_wvshift -= np.nanmedian(master_wvshift)
    # # saving
    # hdulist = pyfits.HDUList()
    # hdulist.append(pyfits.PrimaryHDU(data=master_wvshift*dwv))
    # try:
    #     hdulist.writeto(os.path.join("/data/osiris_data/","master_wvshifts_"+IFSfilter+".fits"), overwrite=True)
    # except TypeError:
    #     hdulist.writeto(os.path.join("/data/osiris_data/","master_wvshifts_"+IFSfilter+".fits"), clobber=True)
    # hdulist.close()

    # for planet in ["b","c","d"]:
    #     inputdir = "/data/osiris_data/HR_8799_"+planet
    #     filename_filter = os.path.join(inputdir,"*/reduced_sky_jb/s*_"+IFSfilter+"_[0-9][0-9][0-9].fits")
    #     filelist = glob.glob(filename_filter)
    #     epoch_list = np.array([filename.split(os.path.sep)[4] for filename in filelist])
    #     epoch_unique = np.unique(epoch_list)
    #     for epoch in epoch_unique:
    #         hdulist = pyfits.HDUList()
    #         hdulist.append(pyfits.PrimaryHDU(data=master_wvshift*dwv))
    #         try:
    #             hdulist.writeto(os.path.join(inputdir,epoch,"master_wvshifts_"+IFSfilter+".fits"), overwrite=True)
    #         except TypeError:
    #             hdulist.writeto(os.path.join(inputdir,epoch,"master_wvshifts_"+IFSfilter+".fits"), clobber=True)
    #         hdulist.close()

    # exit()

    thresh = 0.5
    wvshift_arr_list = []
    temp_list = []
    cst_offset_list = []
    for k,(filename,filename_R,filename_dwv,filename_model,filename_out) in enumerate(zip(filelist,filelist_R,filelist_dwv,filelist_model,filelist_out)):
        print(k)

        # plt.subplot(4,len(filelist_R),k+1)
        # hdulist = pyfits.open(filename_R)
        # R_arr = hdulist[0].data
        # plt.imshow(R_arr,interpolation="nearest",origin="lower")
        # plt.clim([2000,6000])
        # plt.colorbar()

        plt.figure(1)
        plt.subplot(3,len(filelist_R),k+1)
        hdulist = pyfits.open(filename_dwv)
        dwv_map = return_64x19(hdulist[0].data)
        dwv_map[np.where(np.abs(dwv_map-np.nanmedian(dwv_map))>0.9)] = np.nan
            # wvshift_arr = np.zeros(output.shape[3::])
            # for row in range(output.shape[3]):
            #     for col in range(output.shape[4]):
            #         chi2_allpara = output[3,0,:,row,col]
            #         wvshift_arr[row,col] = dwvs_CCF[np.argmax(chi2_allpara)]
        plt.imshow(dwv_map,interpolation="nearest",origin="lower")
        mymed = np.nanmedian(dwv_map)
        plt.clim([-0.25+mymed,0.25+mymed])
        plt.colorbar()

        plt.subplot(3,len(filelist_R),k+1+len(filelist_R))
        diff = dwv_map-master_wvshift
        diff[np.where(np.abs(diff-np.nanmedian(diff))>thresh)] = np.nan
        plt.title("{0:.2f} || {1:.2f}".format(np.nanmedian(diff)*dprv,
                                      np.nanstd(diff)/std_factor[len(filelist_R)]*dprv),fontsize=7)
        plt.imshow(diff,interpolation="nearest",origin="lower")
        # mymed = np.nanmedian((dwv_map-master_wvshift)[np.where((dwv_map>-1)*(dwv_map<1))])
        plt.clim([-0.25+mymed,0.25+mymed])
        plt.colorbar()

        plt.subplot(3,len(filelist_R),k+1+2*len(filelist_R))
        hdulist = pyfits.open(filename_model)
        bestmodel_arr = hdulist[0].data
        plt.imshow(bestmodel_arr,interpolation="nearest",origin="lower")
        plt.clim([0,12])
        plt.colorbar()

        plt.figure(2)
        plt.subplot(2,len(filelist_R),k+1)
        plt.hist(dwv_map[np.isfinite(dwv_map)],bins=100,range=[-1,1])
        plt.title("{0:.2f}".format(np.nanstd(dwv_map)))
        # plt.xlim([-1,1])

        plt.subplot(2,len(filelist_R),k+1+len(filelist_R))
        plt.hist(diff[np.where(np.isfinite(diff))],bins=100,range=[-1,1])
        plt.title("{0:.3f} // {1:.3f} // {2:.3f}".format(np.nanstd(diff)/std_factor[len(filelist_R)],
                                                         np.nanstd(diff)/std_factor[len(filelist_R)]*dprv,
                                                         np.nanstd(dwv_map)*std_factor[len(filelist_R)]),fontsize=8)
        plt.xlim([-0.5,1])

        cst_offset_list.append(np.nanmean(diff))
        hdulist = pyfits.open(filename)
        temp_list.append(float(hdulist[0].header["DTMP7"]))
        print(filename,np.nanmean(diff),float(hdulist[0].header["DTMP7"]))
        plt.figure(4)
        plt.scatter(float(hdulist[0].header["DTMP7"]),np.nanmean(diff),label=filename)


    mymed = np.nanmedian(master_wvshift)
    plt.figure(3)
    plt.subplot(1,2,1)
    plt.imshow(master_wvshift,interpolation="nearest",origin="lower")
    plt.clim([-0.25+mymed,0.25+mymed])
    plt.subplot(1,2,2)
    plt.imshow(master_wvshift*dprv,interpolation="nearest",origin="lower")
    plt.clim([-10+mymed*dprv,10+mymed*dprv])
    plt.colorbar()

    plt.figure(4)
    plt.legend()

    print("error",np.std(cst_offset_list)/std_factor[len(cst_offset_list)]*dprv,std_factor[len(cst_offset_list)])

    cst_offset_list = np.array(cst_offset_list)
    filelist = np.array(filelist)
    epoch_list = np.array([filename.split(os.path.sep)[4] for filename in filelist])
    epoch_unique = np.unique(epoch_list)
    # print(len(filelist))
    # print(len(epoch_list))
    # print(len(temp_list))
    # print(len(cst_offset_list))
    # exit()
    master_wvshift[np.isnan(master_wvshift)] = 0
    for filename,epoch,temp in zip(filelist,epoch_list,temp_list):
        # pass
        print(os.path.join(os.path.dirname(filename),"..","master_wvshifts_"+IFSfilter+".fits"))
        print(cst_offset_list[np.where(epoch == epoch_list)]*38.167938931297705)
        print((cst_offset_list[np.where(epoch == epoch_list)]-cst_offset_list[np.where(epoch == epoch_list)])*38.167938931297705)
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=(master_wvshift+np.mean(cst_offset_list[np.where(epoch == epoch_list)]))*dwv))
        try:
            hdulist.writeto(os.path.join(os.path.dirname(filename),"..","master_wvshifts_"+IFSfilter+".fits"), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(os.path.dirname(filename),"..","master_wvshifts_"+IFSfilter+".fits"), clobber=True)
        print("saving " + os.path.join(os.path.dirname(filename),"..","master_wvshifts_"+IFSfilter+".fits"))
        hdulist.close()

    plt.show()

if 0: # plot master sky calibrations
    fontsize=12
    k = 0
    f,axes = plt.subplots(1,15,sharex="col",sharey="row",figsize=(12,4))
    for year in np.arange(2009,2021,1):
        for ifsfilter in ["Kbb","Hbb"]:
            if ifsfilter=="Kbb": #Kbb 1965.0 0.25
                CRVAL1 = 1965.
                CDELT1 = 0.25
                nl=1665
                R=4000
            elif ifsfilter=="Hbb": #Hbb 1651 1473.0 0.2
                CRVAL1 = 1473.
                CDELT1 = 0.2
                nl=1651
                R=4000
            elif ifsfilter=="Jbb": #Hbb 1651 1473.0 0.2
                CRVAL1 = 1180.
                CDELT1 = 0.15
                nl=1574
                R0=4000
            init_wv = CRVAL1/1000.
            dwv = CDELT1/1000.
            wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)
            dprv = 3e5*dwv/(init_wv+dwv*nl//2)

            filename = glob.glob("/data/osiris_data/HR_8799_*/{0}*/master_wvshifts_{1}.fits".format(year,ifsfilter))
            if len(filename) >= 1:
                filename = filename[0]
            else:
                continue
            print(filename)
            hdulist = pyfits.open(filename)
            master_dwv = return_64x19(hdulist[0].data)
            master_dwv[np.where(np.abs(master_dwv)>0.9)] = np.nan
            master_dwv -= np.nanmedian(master_dwv)

            k+=1
            # plt.subplot(1,10,k)
            plt.sca(axes[k-1])
            img = plt.imshow(master_dwv/dwv*dprv,interpolation="nearest",origin="lower")
            plt.clim([-5,5])
            plt.gca().text(0,64,"{0} {1}".format(year,ifsfilter),ha="left",va="bottom",rotation=0,size=fontsize)
            if k == 1:
                plt.xlabel("x (pix)",fontsize=fontsize)
                plt.ylabel("y (pix)",fontsize=fontsize)
            else:
                plt.tick_params(axis="x",labelbottom=False)
            if k ==10:
                cax = plt.axes([0.91,0.11,0.024,0.77])
                cbar = plt.colorbar(cax=cax)
                cbar.ax.set_ylabel("RV offset (km/s)",fontsize=fontsize)
    plt.subplots_adjust(wspace=0,hspace=0)


    if 0:
        print("Saving "+os.path.join(out_pngs,"master_wavcal.pdf"))
        plt.savefig(os.path.join(out_pngs,"master_wavcal.pdf"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"master_wavcal.png"),bbox_inches='tight')
    plt.show()
