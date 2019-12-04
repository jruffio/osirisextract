__author__ = 'jruffio'

import matplotlib
matplotlib.use("Agg")
import os
import csv
import glob
import itertools
import sys
import numpy as np
import astropy.io.fits as pyfits
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from PyAstronomy import pyasl
from copy import copy
from scipy.ndimage.filters import median_filter
#Logic to test mkl exists
try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

def _task_convolve_spectrum(paras):
    indices,wvs,spectrum,R = paras
    conv_spectrum = np.zeros(np.size(indices))
    dwvs = wvs[1::]-wvs[0:(np.size(wvs)-1)]
    med_dwv = np.median(dwvs)
    for l,k in enumerate(indices):
        pwv = wvs[k]
        FWHM = pwv/R
        sig = FWHM/(2*np.sqrt(2*np.log(2)))
        w = int(np.round(sig/med_dwv*10.))
        stamp_spec = spectrum[np.max([0,k-w]):np.min([np.size(spectrum),k+w])]
        stamp_wvs = wvs[np.max([0,k-w]):np.min([np.size(wvs),k+w])]
        stamp_dwvs = stamp_wvs[1::]-stamp_wvs[0:(np.size(stamp_spec)-1)]
        gausskernel = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(stamp_wvs-pwv)**2/sig**2)
        conv_spectrum[l] = np.sum(gausskernel[1::]*stamp_spec[1::]*stamp_dwvs)
    return conv_spectrum

def convolve_spectrum(wvs,spectrum,R,mypool=None):
    if mypool is None:
        return _task_convolve_spectrum((np.arange(np.size(spectrum)).astype(np.int),wvs,spectrum,R))
    else:
        conv_spectrum = np.zeros(spectrum.shape)

        # chunk_size=1000
        # N_chunks = np.size(spectrum)//chunk_size
        N_chunks = mypool._processes*5
        chunk_size = int(np.size(spectrum)//N_chunks)
        indices_list = []
        for k in range(N_chunks-1):
            indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
        indices_list.append(np.arange((N_chunks-1)*chunk_size,np.size(spectrum)).astype(np.int))
        outputs_list = mypool.map(_task_convolve_spectrum, zip(indices_list,
                                                               itertools.repeat(wvs),
                                                               itertools.repeat(spectrum),
                                                               itertools.repeat(R)))
        for indices,out in zip(indices_list,outputs_list):
            conv_spectrum[indices] = out

        return conv_spectrum

def LPFvsHPF(myvec,cutoff,nansmooth=10):
    myvec_cp = copy(myvec)
    #handling nans:
    wherenans = np.where(np.isnan(myvec_cp))
    for k in wherenans[0]:
        myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-nansmooth]):np.min([np.size(myvec_cp),k+nansmooth])])

    fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec

    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan
    return LPF_myvec,HPF_myvec

def get_best_telluric_model(wvs_corr,_spec,phoenix_refstar_func,transmission_list,where_lines,rv0,bary_rv,cutoff):
    c_kms = 299792.458
    chi2_list = []
    # plt.figure(10)
    for k,trans in  enumerate(transmission_list):
        _spec_model = phoenix_refstar_func(wvs_corr*(1-(rv0+bary_rv)/c_kms))
        res_hpf = LPFvsHPF(_spec/trans(wvs_corr)/_spec_model,cutoff,nansmooth=10)[1]
        res_hpf[where_lines] = np.nan
        chi2_list.append(np.nanvar(res_hpf))
        # plt.plot(_spec/trans(wvs_corr)/_spec_model,label="{0}".format(k))
    bestmid = np.argmin(chi2_list)
    # print(bestmid)

    # plt.figure(11)
    # trans = transmission_list[bestmid]
    # for alpha in np.arange(0.95,1.15,0.05):
    #     res_hpf = LPFvsHPF(_spec/(trans(wvs_corr)**alpha)/_spec_model,cutoff,nansmooth=10)[1]
    #     res_hpf[where_lines] = np.nan
    #     print(alpha,np.nanvar(res_hpf))
    #     plt.plot(_spec/((trans(wvs_corr))**alpha)/_spec_model,label="{0}".format(alpha))
    # plt.legend()
    # plt.show()
    return bestmid

def get_err_from_posterior(x,posterior):
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    if len(x[0:argmax_post]) < 2:
        lx = np.nan
    else:
        lf = interp1d(cum_posterior[0:argmax_post],x[0:argmax_post],bounds_error=False,fill_value=np.nan)
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx = np.nan
    else:
        rf = interp1d(cum_posterior[argmax_post::],x[argmax_post::],bounds_error=False,fill_value=np.nan)
        rx = rf(1-0.6827)
    return x[argmax_post],lx,rx,lx-x[argmax_post],rx-x[argmax_post],argmax_post

def get_upperlim_from_posterior(x,posterior):
    cum_posterior = np.cumsum(posterior)
    cum_posterior = cum_posterior/np.max(cum_posterior)
    lf = interp1d(cum_posterior,x,bounds_error=False,fill_value=np.nan)
    upplim = lf(0.6827)
    return upplim

def get_residuals(rv, vsini,wvs_corr,spec,ref_star_func, trans_func,bary_rv,limbdark0,Hlines,dwvfit,degpoly=1):
    c_kms = 299792.458

    where_wvs4broadening = np.where((np.min(Hlines)-10*dwvfit<wvs_corr)*(wvs_corr<np.max(Hlines)+10*dwvfit))[0]
    wvs4broadening = np.arange(wvs_corr[where_wvs4broadening][0],wvs_corr[where_wvs4broadening][-1],1e-5)

    #make stamps
    wvs_stamps = []
    spec_stamps = []
    first_index_list = []
    last_index_list = []
    first_index = 0
    for Hline in Hlines:
        where_line = np.where((Hline-dwvfit<wvs_corr)*(wvs_corr<Hline+dwvfit))
        first_index_list.append(first_index)
        wvs_stamps.append(wvs_corr[where_line])
        spec_stamps.append(spec[where_line])
        first_index = first_index + np.size(where_line[0])
        last_index_list.append(first_index)
    wvs_stamps_conca = np.concatenate(wvs_stamps)
    spec_stamps_conca = np.concatenate(spec_stamps)
    where_data_finite = np.where(np.isfinite(spec_stamps_conca))

    limbdark = limbdark0
    broadened_spec = pyasl.rotBroad(wvs4broadening, ref_star_func(wvs4broadening), limbdark, vsini)
    refstar_broad_func = interp1d(wvs4broadening,broadened_spec,bounds_error=False,fill_value=np.nan)

    # print(vsini_id,rv_id,vsini,rv)
    if 0:
        M = np.zeros((np.size(spec_stamps_conca),1+(degpoly+1)*len(wvs_stamps)))
        M[:,0] = refstar_broad_func(wvs_stamps_conca*(1-(rv+bary_rv)/c_kms))*trans_func(wvs_stamps_conca)
        M0_mn = np.mean(M[:,0])
        M[:,0] /= M0_mn
        for stamp_id,(wvs_stamp,spec_stamp,first_index,last_index) in enumerate(zip(wvs_stamps,spec_stamps,first_index_list,last_index_list)):
            for polypower in range(degpoly+1):
                tmp = wvs_stamp**polypower
                M[first_index:last_index,polypower+(degpoly+1)*stamp_id+1] = tmp/np.mean(tmp)
        d = spec_stamps_conca[where_data_finite]
        d_wvs = wvs_stamps_conca[where_data_finite]
        M = M[where_data_finite[0],:]
        p,chi2,rank,s = np.linalg.lstsq(M,d,rcond=None)
        m = np.dot(M,p)
        m_line = np.dot(M[:,0],p[0])
        m_bckg = np.dot(M[:,1::],p[1::])
        res = d-m
    else:
        M = np.zeros((np.size(spec_stamps_conca),(degpoly+1)*len(wvs_stamps)))
        tmp = refstar_broad_func(wvs_stamps_conca*(1-(rv+bary_rv)/c_kms))*trans_func(wvs_stamps_conca)
        M0_mn = np.mean(tmp)
        tmp /= M0_mn
        for stamp_id,(wvs_stamp,spec_stamp,first_index,last_index) in enumerate(zip(wvs_stamps,spec_stamps,first_index_list,last_index_list)):
            for polypower in range(degpoly+1):
                M[first_index:last_index,polypower+(degpoly+1)*stamp_id] = \
                    tmp[first_index:last_index]*(wvs_stamp**polypower)
        d = spec_stamps_conca[where_data_finite]
        d_wvs = wvs_stamps_conca[where_data_finite]
        M = M[where_data_finite[0],:]
        p,chi2,rank,s = np.linalg.lstsq(M,d,rcond=None)
        m = np.dot(M,p)
        m_line = np.nan
        m_bckg = np.nan
        res = d-m



    return d_wvs,d,m,res,m_line,m_bckg

def _fit_rv_vsin(paras):
    return fit_rv_vsin(*paras)

def fit_rv_vsin(rv_samples, vsini_samples,wvs_corr,spec,ref_star_func, trans_func,bary_rv,limbdark0,Hlines,dwvfit,degpoly=1):
    c_kms = 299792.458


    chi2_arr = np.zeros((len(rv_samples),len(vsini_samples)))
    logpost_arr = np.zeros((len(rv_samples),len(vsini_samples)))

    where_wvs4broadening = np.where((np.min(Hlines)-10*dwvfit<wvs_corr)*(wvs_corr<np.max(Hlines)+10*dwvfit))[0]
    wvs4broadening = np.arange(wvs_corr[where_wvs4broadening][0],wvs_corr[where_wvs4broadening][-1],1e-5)

    #make stamps
    wvs_stamps = []
    spec_stamps = []
    first_index_list = []
    last_index_list = []
    first_index = 0
    for Hline in Hlines:
        where_line = np.where((Hline-dwvfit<wvs_corr)*(wvs_corr<Hline+dwvfit))
        first_index_list.append(first_index)
        wvs_stamps.append(wvs_corr[where_line])
        spec_stamps.append(spec[where_line])
        first_index = first_index + np.size(where_line[0])
        last_index_list.append(first_index)
    wvs_stamps_conca = np.concatenate(wvs_stamps)
    spec_stamps_conca = np.concatenate(spec_stamps)
    where_data_finite = np.where(np.isfinite(spec_stamps_conca))

    limbdark = limbdark0
    for vsini_id,vsini in enumerate(vsini_samples):
        broadened_spec = pyasl.rotBroad(wvs4broadening, ref_star_func(wvs4broadening), limbdark, vsini)
        refstar_broad_func = interp1d(wvs4broadening,broadened_spec,bounds_error=False,fill_value=np.nan)

        for rv_id,rv in enumerate(rv_samples):
            # print(vsini_id,rv_id,vsini,rv)
            if 0:
                M = np.zeros((np.size(spec_stamps_conca),1+(degpoly+1)*len(wvs_stamps)))
                M[:,0] = refstar_broad_func(wvs_stamps_conca*(1-(rv+bary_rv)/c_kms))*trans_func(wvs_stamps_conca)
                M0_mn = np.mean(M[:,0])
                M[:,0] /= M0_mn
                for stamp_id,(wvs_stamp,spec_stamp,first_index,last_index) in enumerate(zip(wvs_stamps,spec_stamps,first_index_list,last_index_list)):
                    for polypower in range(degpoly+1):
                        tmp = wvs_stamp**polypower
                        M[first_index:last_index,polypower+(degpoly+1)*stamp_id+1] = tmp/np.mean(tmp)
            else:
                M = np.zeros((np.size(spec_stamps_conca),(degpoly+1)*len(wvs_stamps)))
                tmp = refstar_broad_func(wvs_stamps_conca*(1-(rv+bary_rv)/c_kms))*trans_func(wvs_stamps_conca)
                M0_mn = np.mean(tmp)
                tmp /= M0_mn
                for stamp_id,(wvs_stamp,spec_stamp,first_index,last_index) in enumerate(zip(wvs_stamps,spec_stamps,first_index_list,last_index_list)):
                    for polypower in range(degpoly+1):
                        M[first_index:last_index,polypower+(degpoly+1)*stamp_id] = \
                            tmp[first_index:last_index]*(wvs_stamp**polypower)

            d = spec_stamps_conca[where_data_finite]
            M = M[where_data_finite[0],:]
            p,chi2,rank,s = np.linalg.lstsq(M,d,rcond=None)
            m = np.dot(M,p)
            res = d-m
            chi2 = np.nansum((res)**2)
            slogdet_icovphi0 = np.linalg.slogdet(np.dot(M.T,M))
            logpost = 0.5*slogdet_icovphi0[1]- (np.size(d)-M.shape[-1]+2-1)/(2)*np.log(chi2)

            chi2_arr[rv_id,vsini_id] = chi2
            logpost_arr[rv_id,vsini_id] = logpost
            # plt.figure(3)
            # # for m in M.T:
            # #     plt.plot(m)
            # plt.plot(spec_stamps_conca)
            # plt.plot(data_model)
            # plt.show()


            # model_list.append(spec_model)
            # for

    return chi2_arr, logpost_arr


def fit_rv_vsin_model(rv_samples, vsini_samples,wvs_corr,spec,ref_star_func_list, trans_func,bary_rv,limbdark0,Hlines,dwvfit,degpoly=1,numthreads=None):
    chi2_arr = np.zeros((len(ref_star_func_list),len(rv_samples),len(vsini_samples)))
    logpost_arr = np.zeros((len(ref_star_func_list),len(rv_samples),len(vsini_samples)))
    if numthreads is None:
        for ref_id,ref_star_func in enumerate(ref_star_func_list):
            chi2_arr[ref_id,:,:],logpost_arr[ref_id,:,:] = \
                fit_rv_vsin(rv_samples, vsini_samples,wvs_corr,spec,ref_star_func, trans_func,bary_rv,limbdark0,Hlines,dwvfit,degpoly)
    else:
        numthreads=32
        fitpool = mp.Pool(processes=numthreads)

        outputs_list = fitpool.map(_fit_rv_vsin, zip(itertools.repeat(rv_samples),
                                                     itertools.repeat(vsini_samples),
                                                     itertools.repeat(wvs_corr),
                                                     itertools.repeat(spec),
                                                     ref_star_func_list,
                                                     itertools.repeat(trans_func),
                                                     itertools.repeat(bary_rv),
                                                     itertools.repeat(limbdark0),
                                                     itertools.repeat(Hlines),
                                                     itertools.repeat(dwvfit),
                                                     itertools.repeat(degpoly)))

        for ref_id,out in enumerate(outputs_list):
            chi2_arr[ref_id,:,:],logpost_arr[ref_id,:,:] = out

        fitpool.close()
        fitpool.join()

    return chi2_arr, logpost_arr

##############################
wvs = dict()
CRVAL1_Kbb = 1965.
CDELT1_Kbb = 0.25
nl_Kbb =1665
# R0_Kbb =4000
init_wv_Kbb = CRVAL1_Kbb/1000. # wv for first slice in mum
dwv_Kbb = CDELT1_Kbb/1000.
wvs["Kbb"]=np.linspace(init_wv_Kbb,init_wv_Kbb+dwv_Kbb*nl_Kbb,nl_Kbb,endpoint=False)

CRVAL1_Hbb = 1473.
CDELT1_Hbb = 0.2
nl_Hbb=1651
# R0_Hbb=4000#5000
init_wv_Hbb = CRVAL1_Hbb/1000. # wv for first slice in mum
dwv_Hbb = CDELT1_Hbb/1000.
wvs["Hbb"]=np.linspace(init_wv_Hbb,init_wv_Hbb+dwv_Hbb*nl_Hbb,nl_Hbb,endpoint=False)

CRVAL1_Jbb = 1180.
CDELT1_Jbb = 0.15
nl_Jbb=1574
# R0_Jbb=4000
init_wv_Jbb = CRVAL1_Jbb/1000. # wv for first slice in mum
dwv_Jbb = CDELT1_Jbb/1000.
wvs["Jbb"]=np.linspace(init_wv_Jbb,init_wv_Jbb+dwv_Jbb*nl_Jbb,nl_Jbb,endpoint=False)

R=4000

# Paschen 1.875; 1.282; 1.094; 1.005; 0.9546; 0.8204 # air
# Brackett 4.051; 2.625; 2.166; 1.944; 1.817; 1.458 # air
# Pfund 7.460; 4.654; 3.741; 3.297; 3.039; 2.279 # vacuum
# all_Hlines = np.array([1.875, 1.282, 1.094, 1.005, 0.9546, 0.8204,
#              4.051, 2.625, 2.166, 1.944, 1.817, 1.458,
#              7.460, 4.654, 3.741, 3.297, 3.039, 2.279])
all_Hlines = np.array([2.1661,1.9451,1.87561,1.8179,1.73668,1.6811,1.64116,1.61137])
dwvfit = dict()
dwvfit["Hbb"] = 0.02
dwvfit["Kbb"] = 0.04
Hlines = dict()
Hlines["Kbb"] = np.sort(all_Hlines[np.where((wvs["Kbb"][0]<all_Hlines)*(all_Hlines<wvs["Kbb"][-1]))])
Hlines["Hbb"] = np.sort(all_Hlines[np.where((wvs["Hbb"][0]<all_Hlines)*(all_Hlines<wvs["Hbb"][-1]))])
Hlines["Jbb"] = np.sort(all_Hlines[np.where((wvs["Jbb"][0]<all_Hlines)*(all_Hlines<wvs["Jbb"][-1]))])
print(Hlines["Kbb"])
print(Hlines["Hbb"])
print(Hlines["Jbb"])
print(wvs["Kbb"][0],wvs["Kbb"][-1])
print(wvs["Hbb"][0],wvs["Hbb"][-1])
print(wvs["Jbb"][0],wvs["Jbb"][-1])
# exit()


# move phoenix database
if 0:
    from shutil import copyfile
    phoenix_ori_dir = "/data/derosa/Raw_Model_Spectra/Goettingen/v2/PHOENIX-ACES-AGSS-COND-2011/"
    N = 0
    for folder in os.listdir(phoenix_ori_dir):
        print(folder)
        phoenix_db_filelist = glob.glob(os.path.join(phoenix_ori_dir,folder,"lte*PHOENIX-ACES-AGSS-COND-2011-HiRes.fits.xz"))
        # print(os.path.basename(phoenix_db_filelist[0])[3:8])
        # exit()
        # print(phoenix_db_filelist)
        # Teff_list = [int(os.path.basename(phoenix_db_filename).split("lte")[-1].split("-")[0]) for phoenix_db_filename in phoenix_db_filelist]
        Teff_list = np.array([int(os.path.basename(phoenix_db_filename)[3:8]) for phoenix_db_filename in phoenix_db_filelist])
        where_temp = np.where((6000<Teff_list)*(Teff_list<7010))
        # where_temp = np.where((7000<Teff_list)*(Teff_list<13000))
        phoenix_db_filelist = np.array(phoenix_db_filelist)[where_temp]
        Teff_list = Teff_list[where_temp]

        ph_outputdir = os.path.join("/data/osiris_data/phoenix","PHOENIX-ACES-AGSS-COND-2011",folder)
        if len(phoenix_db_filelist) !=0 and not os.path.exists(ph_outputdir):
            os.makedirs(ph_outputdir)
        for phoenix_db_filename,Teff in zip(phoenix_db_filelist,Teff_list):
            src = phoenix_db_filename
            dst = os.path.join("/data/osiris_data/phoenix","PHOENIX-ACES-AGSS-COND-2011",folder,os.path.basename(phoenix_db_filename))
            N = N+1
            print(N,10563)
            # copyfile(src,dst)
            # os.popen("unxz "+dst)
    print(N)
    exit()

# convolve phoenix db
if 0:
    R=4000
    numthreads=32
    specpool = mp.Pool(processes=numthreads)

    phoenix_dir = "/data/osiris_data/phoenix/PHOENIX-ACES-AGSS-COND-2011"
    phoenix_wv_filename = os.path.join(phoenix_dir,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
    with pyfits.open(phoenix_wv_filename) as hdulist:
        phoenix_wvs = hdulist[0].data/1.e4
    crop_phoenix = np.where((phoenix_wvs>wvs["Jbb"][0]-(wvs["Jbb"][-1]-wvs["Jbb"][0])/2)*\
                            (phoenix_wvs<wvs["Kbb"][-1]+(wvs["Kbb"][-1]-wvs["Kbb"][0])/2))
    phoenix_wvs = phoenix_wvs[crop_phoenix]

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=phoenix_wvs))
    try:
        hdulist.writeto(phoenix_wv_filename.replace(".fits","_R{0}.fits".format(R)), overwrite=True)
    except TypeError:
        hdulist.writeto(phoenix_wv_filename.replace(".fits","_R{0}.fits".format(R)), clobber=True)
    hdulist.close()

    N = 0
    for folder in os.listdir(phoenix_dir):
        print(folder)
        if "Alpha" in folder:
            continue
        phoenix_db_filelist = glob.glob(os.path.join(phoenix_dir,folder,"lte*PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"))

        for phoenix_db_filename in phoenix_db_filelist:
            with pyfits.open(phoenix_db_filename) as hdulist:
                phoenix_refstar = hdulist[0].data[crop_phoenix]
            if len(glob.glob(phoenix_db_filename.replace(".fits","_R{0}.fits".format(R)))) !=0:
                continue
            N = N+1
            print(N,254,phoenix_db_filename)

            phoenix_refstar = convolve_spectrum(phoenix_wvs,phoenix_refstar,R,specpool)


            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=phoenix_refstar))
            try:
                hdulist.writeto(phoenix_db_filename.replace(".fits","_R{0}.fits".format(R)), overwrite=True)
            except TypeError:
                hdulist.writeto(phoenix_db_filename.replace(".fits","_R{0}.fits".format(R)), clobber=True)
            hdulist.close()
    print(N)
    specpool.close()
    exit()


# # interpolate and save transmission spectra
# if 0:
#     R=4000
#     numthreads=32
#     specpool = mp.Pool(processes=numthreads)
#     sky_transmission_folder = os.path.join("/data/osiris_data","sky_transmission")
#     water_vapor_list = np.array([10,16,30,50])
#     airmass_list = np.array([10,15,20])
#
#     filename = os.path.join(sky_transmission_folder,"mktrans_zm_10_10.dat")
#     skybg_arr=np.loadtxt(filename)
#     skytrans_wvs = skybg_arr[:,0]
#     selec_skytrans = np.where((skytrans_wvs>wvs["Jbb"][0]-(wvs["Jbb"][-1]-wvs["Jbb"][0])/2)*\
#                               (skytrans_wvs<wvs["Kbb"][-1]+(wvs["Kbb"][-1]-wvs["Kbb"][0])/2))
#     skytrans_wvs = skytrans_wvs[selec_skytrans]
#
#     data = np.zeros((np.size(water_vapor_list),np.size(airmass_list),np.size(skytrans_wvs)))
#     for wid, watvap in enumerate(water_vapor_list):
#         for aid, airmass in enumerate(airmass_list):
#             print(watvap,airmass)
#             filename = os.path.join(sky_transmission_folder,"mktrans_zm_{0}_{1}.dat".format(watvap,airmass))
#             skybg_arr=np.loadtxt(filename)
#             skytrans_spec = skybg_arr[:,1]
#             skytrans_spec = skytrans_spec[selec_skytrans]
#             skytrans_spec = convolve_spectrum(skytrans_wvs,skytrans_spec,R,specpool)
#             data[wid,aid,:] = skytrans_spec
#     specpool.close()
#
#     from scipy.interpolate import RegularGridInterpolator
#     interp_object = RegularGridInterpolator((water_vapor_list,airmass_list),data,method="linear",bounds_error=False,fill_value=np.nan)
#
#     for wid, watvap in enumerate(np.arange(water_vapor_list[0],water_vapor_list[-1])):
#         for aid, airmass in enumerate(np.arange(airmass_list[0],airmass_list[-1])):
#             out_filename = os.path.join(sky_transmission_folder,"interpolated","mktrans_zm_{0}_{1}_R{2}.csv".format(watvap,airmass,R))
#             with open(out_filename, 'w+') as csvfile:
#                 csvwriter = csv.writer(csvfile, delimiter=' ')
#                 csvwriter.writerows([[a,b] for a,b in zip(skytrans_wvs,interp_object((watvap,airmass)))])
#             # exit()
#     #
#     # filename = os.path.join(sky_transmission_folder,"mktrans_zm_{0}_{1}.dat".format(10,10))
#     # skybg_arr=np.loadtxt(filename)
#     # skytrans_spec = skybg_arr[:,1]
#     # skytrans_spec = skytrans_spec[selec_skytrans]
#     # skytrans_spec = convolve_spectrum(skytrans_wvs,skytrans_spec,R,specpool)
#     # plt.plot(skytrans_spec)
#     # filename = os.path.join(sky_transmission_folder,"mktrans_zm_{0}_{1}.dat".format(10,15))
#     # skybg_arr=np.loadtxt(filename)
#     # skytrans_spec = skybg_arr[:,1]
#     # skytrans_spec = skytrans_spec[selec_skytrans]
#     # skytrans_spec = convolve_spectrum(skytrans_wvs,skytrans_spec,R,specpool)
#     # plt.plot(skytrans_spec)
#     # plt.plot(interp_object((10,12.5)),"--")
#     # plt.show()
#     exit()

def _interp_sky_trans(paras):
    indices,skytrans_spec_list = paras
    out = []
    for filename_skytrans in skytrans_spec_list:
        skybg_arr=np.loadtxt(filename_skytrans)
        skytrans_wvs = skybg_arr[:,0]
        skytrans_spec = skybg_arr[:,1]
        out.append(interp1d(skytrans_wvs,skytrans_spec,bounds_error=False,fill_value=np.nan))
    return out

def _interp_phoenix(paras):
    indices,skytrans_spec_list,skytrans_wvs = paras
    out = []
    for filename_skytrans in skytrans_spec_list:
        with pyfits.open(filename_skytrans) as hdulist:
            skytrans_spec = hdulist[0].data
        out.append(interp1d(skytrans_wvs,skytrans_spec,bounds_error=False,fill_value=np.nan))
    return out

def combine_spectra(_spec_list):
    _spec_LPF_list = []
    _spec_HPF_list = []
    for fid,spec_it in enumerate(_spec_list):
        a,b = LPFvsHPF(spec_it/np.nanmean(spec_it),cutoff=10,nansmooth=50)
        _spec_LPF_list.append(a)
        _spec_HPF_list.append(b)
    _spec = np.nanmean(_spec_LPF_list, axis=0) + np.nanmean(_spec_HPF_list, axis=0)
    return _spec

#------------------------------------------------
if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    result = []

    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv))
    print("CPU COUNT: {0}".format(mp.cpu_count()))

    numthreads = 32

    osiris_data_dir = sys.argv[1]#"/data/osiris_data"
    # osiris_data_dir = "/data/osiris_data/"
    fileinfos_refstars_filename = os.path.join(osiris_data_dir,"fileinfos_refstars_jb.csv")
    phoenix_folder = os.path.join(osiris_data_dir,"phoenix")




    with open(fileinfos_refstars_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        refstarsinfo_list_table = list(csv_reader)
        refstarsinfo_colnames = refstarsinfo_list_table[0]
        refstarsinfo_list_data = refstarsinfo_list_table[1::]
    IFSfilter_id = refstarsinfo_colnames.index("IFS filter")
    ref_IFSfilter_list = np.array([item[IFSfilter_id] for item in refstarsinfo_list_data])
    refstarsinfo_list_data = np.array(refstarsinfo_list_data)[np.where("Jbb" != ref_IFSfilter_list)]

    refstarsinfo_filename_id = refstarsinfo_colnames.index("filename")
    type_id = refstarsinfo_colnames.index("type")
    Jmag_id = refstarsinfo_colnames.index("Jmag")
    Hmag_id = refstarsinfo_colnames.index("Hmag")
    Kmag_id = refstarsinfo_colnames.index("Kmag")
    starname_id = refstarsinfo_colnames.index("star name")
    baryrv_id = refstarsinfo_colnames.index("barycenter rv")
    vsini_fixed_id = refstarsinfo_colnames.index("vsini fixed")
    rv_simbad_id = refstarsinfo_colnames.index("RV Simbad")
    ref_stars_filelist = np.array([item[refstarsinfo_filename_id].replace("/data/osiris_data",osiris_data_dir) for item in refstarsinfo_list_data])
    # print(ref_stars_filelist[0])
    # exit()
    ref_dates_list = np.array([os.path.basename(filename).split("_a")[0].replace("s","").replace("ao_off_","") for filename in ref_stars_filelist])
    ref_unique_dates = np.unique(ref_dates_list)
    print(ref_unique_dates)
    # exit()
    ref_baryrv_list = np.array([-float(item[baryrv_id])/1000 for item in refstarsinfo_list_data])
    ref_IFSfilter_list = np.array([item[IFSfilter_id] for item in refstarsinfo_list_data])
    ref_rv_simbad_list = np.array([float(item[rv_simbad_id]) for item in refstarsinfo_list_data])
    ref_vsini_fixed_list = np.array([float(item[vsini_fixed_id]) for item in refstarsinfo_list_data])

    ref_starname_list = np.array([item[starname_id] for item in refstarsinfo_list_data])
    uni_starname_list = np.unique(np.concatenate([ref_starname_list,["HR_8799","51_Eri","kap_And"]]))
    vsini_fixed_dict = dict()
    rv_simbad_dict = dict()
    for starname in uni_starname_list:
        if starname == "51_Eri":
            rv_simbad_dict[starname] = 12.6
            vsini_fixed_dict[starname] = 80
        elif starname == "kap_And":
            rv_simbad_dict[starname] = -12.7
            vsini_fixed_dict[starname] = 150
        else:
            rv_simbad_dict[starname] = ref_rv_simbad_list[np.where((starname == ref_starname_list))[0]][0]
            vsini_fixed_dict[starname] = ref_vsini_fixed_list[np.where((starname == ref_starname_list))[0]][0]
    # print(rv_simbad_dict)
    # print(vsini_fixed_dict)
    # exit()

    uni_starname_list = ['kap_And','HIP_111538','51_Eri','HIP_25453','HD_7215','HIP_1123','HIP_116886','HR_8799','HD_210501','BD+14_4774']
    print(len(uni_starname_list)) #10
    print(len(ref_unique_dates)) #26
    starname = uni_starname_list[int(sys.argv[2])]
    date = ref_unique_dates[int(sys.argv[3])]
    if 1:
        print(uni_starname_list)
        # delta_teff = 50

        print(uni_starname_list)
        # exit()
        # for starname in uni_starname_list:
        if 1:
            # HIP_25453 (done)
            # HIP_111538 (done)
            # HD_210501

            specpool = mp.Pool(processes=numthreads)
            ###########################################
            ## Sky transmission
            transmission_list = []
            # # mktrans_zm_[1,16,30,50; 10*mm of water vapor]_[10,15,20;10*airmass].dat
            # # filelist_skytrans = glob.glob(os.path.join(sky_transmission_folder,"mktrans_zm_16_20.dat"))
            # sky_transmission_folder = os.path.join(osiris_data_dir,"sky_transmission")
            # filelist_skytrans = glob.glob(os.path.join(sky_transmission_folder,"mktrans_zm_*_*.dat"))
            #
            # def interp_sky_trans(skytrans_wvs,skytrans_spec_list):
            #     return [interp1d(skytrans_wvs,skytrans_spec,bounds_error=False,fill_value=np.nan) for skytrans_spec in skytrans_spec_list]
            #
            # for filename_skytrans in filelist_skytrans:
            #     print(filename_skytrans)
            #     skybg_arr=np.loadtxt(filename_skytrans)
            #     skytrans_wvs = skybg_arr[:,0]
            #     skytrans_spec = skybg_arr[:,1]
            #     selec_skytrans = np.where((skytrans_wvs>wvs["Jbb"][0]-(wvs["Jbb"][-1]-wvs["Jbb"][0])/2)*\
            #                               (skytrans_wvs<wvs["Kbb"][-1]+(wvs["Kbb"][-1]-wvs["Kbb"][0])/2))
            #     skytrans_wvs = skytrans_wvs[selec_skytrans]
            #     skytrans_spec = skytrans_spec[selec_skytrans]
            #
            #
            #     # skytrans_spec = convolve_spectrum(skytrans_wvs,skytrans_spec,R,specpool)
            #     transmission_list.append(interp1d(skytrans_wvs,skytrans_spec,bounds_error=False,fill_value=np.nan))
            #
            #     # plt.plot(skytrans_wvs,skytrans_spec/np.nanmax(skytrans_spec))

            sky_transmission_folder = os.path.join(osiris_data_dir,"sky_transmission","interpolated")
            filelist_skytrans = glob.glob(os.path.join(sky_transmission_folder,"mktrans_zm_*_*_R{0}.csv".format(R)))#[30:31]

            # N_chunks = specpool._processes*5
            # chunk_size = int(len(filelist_skytrans)//N_chunks)
            transmission_list = []
            chunk_size=10
            N_chunks = int(len(filelist_skytrans)//chunk_size)
            indices_list = []
            skytrans_spec_table = []
            for k in range(N_chunks-1):
                indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
                skytrans_spec_table.append(filelist_skytrans[k*chunk_size:(k+1)*chunk_size])
            indices_list.append(np.arange((N_chunks-1)*chunk_size,len(filelist_skytrans)).astype(np.int))
            skytrans_spec_table.append(filelist_skytrans[(N_chunks-1)*chunk_size::])
            outputs_list = specpool.map(_interp_sky_trans, zip(indices_list,skytrans_spec_table))
            for indices,out in zip(indices_list,outputs_list):
                transmission_list.extend(out)

            # ###########################################
            # ## Sky transmission - ATRAN
            # sky_transmission_folder = os.path.join(osiris_data_dir,"atran")
            # transmission_list = []
            # filelist_atran = glob.glob(os.path.join(sky_transmission_folder,"atran_13599_30_*_2_*_135_245_0.txt"))
            # atran_arr=np.loadtxt(filelist_atran[0])
            # print(atran_arr.shape)
            # atran_wvs = atran_arr[:,1]
            # atran_spec = atran_arr[:,2]
            # selec_atran = np.where((atran_wvs>wvs["Jbb"][0]-(wvs["Jbb"][-1]-wvs["Jbb"][0])/2)*\
            #                           (atran_wvs<wvs["Kbb"][-1]+(wvs["Kbb"][-1]-wvs["Kbb"][0])/2))
            # atran_wvs = atran_wvs[selec_atran]
            # atran_spec = atran_spec[selec_atran]
            #
            # plt.plot(atran_wvs,atran_spec/np.nanmax(atran_spec),"--")


            # ###########################################
            # ## instrument filters
            # instfilter_wvs = dict()
            # instfilter_profile = dict()
            # instfilter_func = dict()
            # for IFSfilter in ["Jbb","Hbb","Kbb"]:
            #     instfilter_folder = os.path.join(osiris_data_dir,"filters")
            #     filename_instfilter = os.path.join(instfilter_folder,"osiris_spec_{0}_data.txt".format(IFSfilter))
            #     instfilter_arr=np.loadtxt(filename_instfilter)
            #     instfilter_wvs[IFSfilter] = instfilter_arr[:,0]/1000
            #     instfilter_profile[IFSfilter] = instfilter_arr[:,1]
            #     instfilter_profile[IFSfilter] = interp1d(instfilter_arr[:,0]/1000,instfilter_arr[:,1],bounds_error=False,fill_value=np.nan)
            #
            #     # plt.plot(instfilter_arr[:,0]/1000,instfilter_arr[:,1]/np.nanmax(instfilter_arr[:,1]),"--")


            ###########################################
            ## Stellar spectrum
            phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
            with pyfits.open(phoenix_wv_filename) as hdulist:
                phoenix_wvs = hdulist[0].data/1.e4
            crop_phoenix = np.where((phoenix_wvs>wvs["Jbb"][0]-(wvs["Jbb"][-1]-wvs["Jbb"][0])/2)*\
                                    (phoenix_wvs<wvs["Kbb"][-1]+(wvs["Kbb"][-1]-wvs["Kbb"][0])/2))
            phoenix_wvs = phoenix_wvs[crop_phoenix]
            print(os.path.join(phoenix_folder,starname+"*.fits"))
            phoenix_model_refstar_filename = glob.glob(os.path.join(phoenix_folder,starname+"*.fits"))[0]
            with pyfits.open(phoenix_model_refstar_filename) as hdulist:
                phoenix_refstar = hdulist[0].data[crop_phoenix]
            teff0 = int(os.path.basename(phoenix_model_refstar_filename).split("_lte")[-1][0:5])
            phoenix_refstar = convolve_spectrum(phoenix_wvs,phoenix_refstar,R,specpool)
            phoenix_refstar_func = interp1d(phoenix_wvs,phoenix_refstar,bounds_error=False,fill_value=np.nan)

            grid_refstar_func_list = []
            with pyfits.open(os.path.join(phoenix_folder,"PHOENIX-ACES-AGSS-COND-2011","WAVE_PHOENIX-ACES-AGSS-COND-2011_R{0}.fits".format(R))) as hdulist:
                phoenix_grid_wvs = hdulist[0].data
            grid_refstar_filelist = glob.glob(os.path.join(phoenix_folder,"PHOENIX-ACES-AGSS-COND-2011","*","lte*_R{0}.fits".format(R)))
            grid_refstar_filelist.sort()
            # grid_refstar_filelist = grid_refstar_filelist[0:20]
            print(len(grid_refstar_filelist))
            # exit()
            # parameters for refstar
            Teff_grid_list = np.array([int(os.path.basename(phoenix_db_filename)[3:8]) for phoenix_db_filename in grid_refstar_filelist])
            logg_grid_list = np.array([float(os.path.basename(phoenix_db_filename)[8:13]) for phoenix_db_filename in grid_refstar_filelist])
            #[-6.5 -6.  -5.5 -5.  -4.5 -4.  -3.5 -3.  -2.5 -2.  -1.5 -1.  -0.5 -0. ]
            Fe_H_grid_list = np.array([float(os.path.basename(phoenix_db_filename)[13:17]) for phoenix_db_filename in grid_refstar_filelist])
            print(Teff_grid_list)
            print(teff0,phoenix_model_refstar_filename)
            # where_teff = np.where((teff0-delta_teff<Teff_grid_list)*(Teff_grid_list<teff0+delta_teff)*\
            #                       (-1.1<Fe_H_grid_list)*(Fe_H_grid_list<1.1))
            where_teff = np.where((teff0-10<Teff_grid_list)*(Teff_grid_list<teff0+10)*\
                                  (0-1.1<Fe_H_grid_list)*(Fe_H_grid_list<0+1.1)*\
                                  (-4.-1.1<logg_grid_list)*(logg_grid_list<-4.+1.1))
            grid_refstar_filelist = np.array(grid_refstar_filelist)[where_teff]
            print(len(grid_refstar_filelist))
            # exit()

            # filter frames

            chunk_size=10
            N_chunks = int(len(grid_refstar_filelist)//chunk_size)
            indices_list = []
            skytrans_spec_table = []
            for k in range(N_chunks-1):
                indices_list.append(np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int))
                skytrans_spec_table.append(grid_refstar_filelist[k*chunk_size:(k+1)*chunk_size])
            indices_list.append(np.arange((N_chunks-1)*chunk_size,len(grid_refstar_filelist)).astype(np.int))
            skytrans_spec_table.append(grid_refstar_filelist[(N_chunks-1)*chunk_size::])
            outputs_list = specpool.map(_interp_phoenix, zip(indices_list,skytrans_spec_table,itertools.repeat(phoenix_grid_wvs)))
            for indices,out in zip(indices_list,outputs_list):
                grid_refstar_func_list.extend(out)

            # plt.figure(5)
            # print(len(grid_refstar_func_list))
            # for f,T,logg,FeH in zip(grid_refstar_func_list,Teff_grid_list,logg_grid_list,Fe_H_grid_list):
            #     plt.plot(phoenix_wvs,f(phoenix_wvs),label="{0} {1} {2}".format(T,logg,FeH))
            # plt.legend(loc="center left",bbox_to_anchor = (1,0.5))
            # plt.show()



            ###########################################
            ## broadening
            c_kms = 299792.458
            cutoff = 40

            limbdark0 = 0.5
            rv0 = rv_simbad_dict[starname] #13.10
            vsini0 = vsini_fixed_dict[starname]
            print(rv0,vsini0)

            wvs4broadening = np.arange(phoenix_wvs[0],phoenix_wvs[-1],
                                       1e-5)
            broadened_spec = pyasl.rotBroad(wvs4broadening, phoenix_refstar_func(wvs4broadening), limbdark0, vsini0)
            phoenix_refstar_broad0_func = interp1d(wvs4broadening,broadened_spec,bounds_error=False,fill_value=np.nan)
            # phoenix_refstar_broad0_func = phoenix_refstar_func

            wvs_list = []
            spec_list = []
            IFSfilter_list = []
            baryrv_list = []
            date_list = []
            type_list = []

            plt.figure(2)
            # for date in ref_unique_dates:
            if 1:
                for IFSfilter in ["Hbb","Kbb"]:
                    where_files = np.where((starname == ref_starname_list) *\
                                           np.array(["ao_off_" not in a for a in ref_stars_filelist])*\
                                           (date == ref_dates_list)*\
                                           (IFSfilter == ref_IFSfilter_list))


                    if len(where_files[0]) != 0:
                        print(ref_stars_filelist[where_files])

                        _spec_list = []
                        for fid,filename in enumerate(np.unique(ref_stars_filelist[where_files])):
                            hdulist = pyfits.open(filename.replace(".fits","_psfs_repaired_spec_v2.fits"))
                            _spec_list.append(hdulist[0].data[1,:])
                            hdulist.close()
                        _spec = combine_spectra(_spec_list)

                        wvsol_offsets_filename = os.path.join(os.path.dirname(ref_stars_filelist[where_files][0]),
                                                              "..","..","master_wvshifts_{0}.fits".format(IFSfilter))
                        hdulist = pyfits.open(wvsol_offsets_filename)
                        wvsol_offsets = hdulist[0].data
                        hdulist.close()
                        wvs_corr = wvs[IFSfilter]-np.nanmean(wvsol_offsets)
                        ##
                        wvs_list.append(wvs_corr)
                        spec_list.append(_spec)
                        IFSfilter_list.append(IFSfilter)
                        baryrv_list.append(np.mean(ref_baryrv_list[where_files]))
                        date_list.append(date)
                        type_list.append("psf")
                        ##
                        plt.plot(wvs_corr,_spec/np.nanmean(_spec),label=date+" "+type_list[-1])

                    where_files = np.where((starname == ref_starname_list) *\
                                           np.array(["ao_off_" in a for a in ref_stars_filelist])*\
                                           (date == ref_dates_list)*\
                                           (IFSfilter == ref_IFSfilter_list))

                    if 1 and len(where_files[0]) != 0:
                        print(ref_stars_filelist[where_files])

                        _spec_list = []
                        for fid,filename in enumerate(np.unique(ref_stars_filelist[where_files])):
                            hdulist = pyfits.open(filename.replace(".fits","_spec_v2.fits"))
                            _spec_list.append(hdulist[0].data[1,:])
                            hdulist.close()
                        _spec = combine_spectra(_spec_list)

                        wvsol_offsets_filename = os.path.join(os.path.dirname(ref_stars_filelist[where_files][0]),
                                                              "..","..","master_wvshifts_{0}.fits".format(IFSfilter))
                        hdulist = pyfits.open(wvsol_offsets_filename)
                        wvsol_offsets = hdulist[0].data
                        hdulist.close()
                        wvs_corr = wvs[IFSfilter]-np.nanmean(wvsol_offsets)
                        ##
                        wvs_list.append(wvs_corr)
                        spec_list.append(_spec)
                        IFSfilter_list.append(IFSfilter)
                        baryrv_list.append(np.mean(ref_baryrv_list[where_files]))
                        date_list.append(date)
                        type_list.append("aooff")
                        ##
                        plt.plot(wvs_corr,_spec/np.nanmean(_spec),label=date+" "+type_list[-1])

                    if 1 and starname in ["HR_8799","kap_And","51_Eri"]:
                        _filelist = glob.glob(os.path.join(osiris_data_dir,starname+"*","20"+date,"reduced_jb","s*_a*_{0}_*.fits".format(IFSfilter)))
                        filelist = []
                        for filename in _filelist:
                            if "HR_8799_b" in filename:
                                continue
                            filelist.append(filename)
                        if len(filelist) != 0:
                            print(filelist)
                            _spec_list = []
                            _bary_list = []
                            for fid,filename in enumerate(filelist):

                                fileinfos_filename = os.path.join(os.path.dirname(filename),"..","..","fileinfos_Kbb_jb.csv")
                                with open(fileinfos_filename, 'r') as csvfile:
                                    csv_reader = csv.reader(csvfile, delimiter=';')
                                    list_table = list(csv_reader)
                                    colnames = list_table[0]
                                    N_col = len(colnames)
                                    list_data = list_table[1::]
                                    N_lines =  len(list_data)
                                filename_id = colnames.index("filename")
                                infofilelist = [os.path.basename(item[filename_id]) for item in list_data]
                                fileid = infofilelist.index(os.path.basename(filename))
                                fileitem = list_data[fileid]
                                baryrv_id = colnames.index("barycenter rv")
                                bary_rv = -float(fileitem[baryrv_id])/1000

                                hdulist = pyfits.open(filename)
                                prihdr = hdulist[0].header
                                curr_mjdobs = prihdr["MJD-OBS"]
                                imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                                imgs_hdrbadpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
                                imgs[np.where(imgs_hdrbadpix==0)] = np.nan
                                hdulist.close()

                                _bary_list.append(bary_rv)
                                _spec_list.append(np.nansum(imgs,axis=(1,2)))
                            _spec = combine_spectra(_spec_list)

                            wvsol_offsets_filename = os.path.join(os.path.dirname(filelist[0]),
                                                                  "..","master_wvshifts_{0}.fits".format(IFSfilter))
                            hdulist = pyfits.open(wvsol_offsets_filename)
                            wvsol_offsets = hdulist[0].data
                            hdulist.close()
                            wvs_corr = wvs[IFSfilter]-np.nanmean(wvsol_offsets)
                            ##
                            wvs_list.append(wvs_corr)
                            spec_list.append(_spec)
                            IFSfilter_list.append(IFSfilter)
                            baryrv_list.append(np.mean(_bary_list))
                            date_list.append(date)
                            type_list.append("science")
                            ##
                            plt.plot(wvs_corr,_spec/np.nanmean(_spec),label=date+" "+type_list[-1])
            plt.legend()
            print("Saving "+os.path.join(osiris_data_dir,"stellar_fits","data_spectra_{0}.pdf".format(starname)))
            plt.savefig(os.path.join(osiris_data_dir,"stellar_fits","data_spectra_{0}.png".format(starname)),bbox_inches='tight')
            plt.savefig(os.path.join(osiris_data_dir,"stellar_fits","data_spectra_{0}.pdf".format(starname)),bbox_inches='tight')
            specpool.close()

            # os.path.join(osiris_data_dir,"stellar_fits","data_spectra_{0}_{1}_{2}_{3}.pdf".format(starname,IFSfilter,date,type))
            # plt.show()
            # exit()

            for spec_id,(wvs_corr,spec,IFSfilter,bary_rv,date,type) in enumerate(zip(wvs_list,spec_list,IFSfilter_list,baryrv_list,date_list,type_list)):
                where_lines_list = []
                for Hline in Hlines[IFSfilter]:
                    where_lines_list.append(np.where((Hline-dwvfit[IFSfilter]<wvs_corr)*(wvs_corr<Hline+dwvfit[IFSfilter]))[0])

                best_tr_id = get_best_telluric_model(wvs_corr,spec,phoenix_refstar_broad0_func,transmission_list,(np.concatenate(where_lines_list),),rv0,bary_rv,cutoff=5)
                print("Done optimizing transmission", best_tr_id,IFSfilter,bary_rv,date,type)

                # rv_samples = np.arange(-30,10,1)
                # vsini_samples = np.arange(10,250,50)
                rv_samples = np.arange(-100,100,0.5)
                vsini_samples = np.arange(10,500,10)

                print(len(grid_refstar_func_list))
                chi2_arr, logpost_arr = fit_rv_vsin_model(rv_samples, vsini_samples,wvs_corr,spec,
                                                grid_refstar_func_list, transmission_list[best_tr_id],
                                                bary_rv,limbdark0,Hlines[IFSfilter],dwvfit[IFSfilter],degpoly=1,numthreads=numthreads)
                # print(chi2_arr.shape)
                # exit()
                # chi2_arr, logpost_arr = fit_rv_vsin(rv_samples, vsini_samples,wvs_corr,spec,
                #                                     phoenix_refstar_func, transmission_list[best_tr_id],
                #                                     bary_rv,limbdark0,Hlines[IFSfilter],dwvfit)
                posterior = np.exp(logpost_arr-np.nanmax(logpost_arr))
                posterior_rv_vsini = np.nansum(posterior,axis=0)
                posterior_model = np.nansum(posterior,axis=(1,2))

                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=np.concatenate((posterior[None,:,:,:],logpost_arr[None,:,:,:],chi2_arr[None,:,:,:]))))
                try:
                    hdulist.writeto(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_posterior.fits".format(starname,IFSfilter,date,type)), overwrite=True)
                except TypeError:
                    hdulist.writeto(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_posterior.fits".format(starname,IFSfilter,date,type)), clobber=True)
                hdulist.close()
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=vsini_samples))
                try:
                    hdulist.writeto(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_vsini_samples.fits".format(starname,IFSfilter,date,type)), overwrite=True)
                except TypeError:
                    hdulist.writeto(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_vsini_samples.fits".format(starname,IFSfilter,date,type)), clobber=True)
                hdulist.close()
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=rv_samples))
                try:
                    hdulist.writeto(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_rv_samples.fits".format(starname,IFSfilter,date,type)), overwrite=True)
                except TypeError:
                    hdulist.writeto(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_rv_samples.fits".format(starname,IFSfilter,date,type)), clobber=True)
                hdulist.close()
                with open(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_models.txt".format(starname,IFSfilter,date,type)), 'w+') as txtfile:
                    txtfile.writelines([s+"\n" for s in grid_refstar_filelist])

                rv_posterior = np.nansum(posterior_rv_vsini,axis=1)
                vsini_posterior = np.nansum(posterior_rv_vsini,axis=0)
                bestrv,_,_,bestrv_merr,bestrv_perr,_ = get_err_from_posterior(rv_samples,rv_posterior)

                bestvsini,_,_,bestvsini_merr,bestvsini_perr,_ = get_err_from_posterior(vsini_samples,vsini_posterior)
                best_model_id = np.argmax(posterior_model)

                d_wvs,d,m,res,m_line,m_bckg = get_residuals(bestrv, bestvsini,wvs_corr,spec,
                                                grid_refstar_func_list[best_model_id], transmission_list[best_tr_id],
                                                bary_rv,limbdark0,Hlines[IFSfilter],dwvfit[IFSfilter],degpoly=1)
                _,_,m_p10,_,_,_ = get_residuals(bestrv+10, bestvsini,wvs_corr,spec,
                                                grid_refstar_func_list[best_model_id], transmission_list[best_tr_id],
                                                bary_rv,limbdark0,Hlines[IFSfilter],dwvfit[IFSfilter],degpoly=1)
                _,_,m_m10,_,_,_  = get_residuals(bestrv-10, bestvsini,wvs_corr,spec,
                                                grid_refstar_func_list[best_model_id], transmission_list[best_tr_id],
                                                bary_rv,limbdark0,Hlines[IFSfilter],dwvfit[IFSfilter],degpoly=1)
                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=np.concatenate((d_wvs[None,:],d[None,:],m[None,:],res[None,:],m_m10[None,:],m_p10[None,:]),axis=0)))
                try:
                    hdulist.writeto(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_best_data_fit.fits".format(starname,IFSfilter,date,type)), overwrite=True)
                except TypeError:
                    hdulist.writeto(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_best_data_fit.fits".format(starname,IFSfilter,date,type)), clobber=True)
                hdulist.close()

                spec_trans = spec/transmission_list[best_tr_id](wvs_corr)
                broadenedspec = pyasl.rotBroad(wvs4broadening, grid_refstar_func_list[best_model_id](wvs4broadening), limbdark0, bestvsini)
                refstar_broad_func = interp1d(wvs4broadening,broadenedspec,bounds_error=False,fill_value=np.nan)
                spec_trans_model = spec/transmission_list[best_tr_id](wvs_corr)/refstar_broad_func(wvs_corr*(1-(bestrv+bary_rv)/c_kms))

                hdulist = pyfits.HDUList()
                hdulist.append(pyfits.PrimaryHDU(data=np.concatenate((wvs_corr[None,:],spec[None,:],spec_trans[None,:],spec_trans_model[None,:]),axis=0)))
                try:
                    hdulist.writeto(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_whole_spec_band.fits".format(starname,IFSfilter,date,type)), overwrite=True)
                except TypeError:
                    hdulist.writeto(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_whole_spec_band.fits".format(starname,IFSfilter,date,type)), clobber=True)
                hdulist.close()

        exit()
    if 1:
        # ['kap_And','HIP_111538','51_Eri','HIP_25453','HD_7215','HIP_1123','HIP_116886','HR_8799','HD_210501','BD+14_4774']
        filter_postfilename = "kap_And_*_posterior.fits"
        # filter_postfilename = "HIP_111538_*_posterior.fits"
        # filter_postfilename = "51_Eri_*_posterior.fits"
        # filter_postfilename = "HIP_25453_*_posterior.fits"
        # filter_postfilename = "HD_210501_*_posterior.fits"
        # filter_postfilename = "HR_8799_*_posterior.fits"

        postfilelist = glob.glob(os.path.join(osiris_data_dir,"stellar_fits",filter_postfilename))

        dates = []
        rvs = np.zeros(len(postfilelist))
        rvs_perr = np.zeros(len(postfilelist))
        rvs_nerr = np.zeros(len(postfilelist))

        for spec_id,postfilename  in enumerate(postfilelist):
            print(postfilename)
            splitpostfilename = os.path.basename(postfilename).split("_")
            IFSfilter,date,type = splitpostfilename[-4],splitpostfilename[-3],splitpostfilename[-2]
            starname = os.path.basename(postfilename).split("_"+IFSfilter)[0]

            dates.append(date)

            nrowplot = 6
            plt.figure(1,figsize=(nrowplot*4,len(postfilelist)*4))

            hdulist = pyfits.open(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_rv_samples.fits".format(starname,IFSfilter,date,type)))
            rv_samples = hdulist[0].data
            hdulist = pyfits.open(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_vsini_samples.fits".format(starname,IFSfilter,date,type)))
            vsini_samples = hdulist[0].data

            hdulist = pyfits.open(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_posterior.fits".format(starname,IFSfilter,date,type)))
            posterior = hdulist[0].data[0]
            logpost_arr = hdulist[0].data[1]
            chi2_arr = hdulist[0].data[2]
            posterior_rv_vsini = np.nansum(posterior,axis=0)
            posterior_model = np.nansum(posterior,axis=(1,2))
            posterior_model /= np.nanmax(posterior_model)

            with open(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_models.txt".format(starname,IFSfilter,date,type)), 'r') as txtfile:
                grid_refstar_filelist = [s.strip() for s in txtfile.readlines()]
                Teff_grid_list = np.array([int(os.path.basename(phoenix_db_filename)[3:8]) for phoenix_db_filename in grid_refstar_filelist])
                logg_grid_list = np.array([float(os.path.basename(phoenix_db_filename)[8:13]) for phoenix_db_filename in grid_refstar_filelist])
                Fe_H_grid_list = np.array([float(os.path.basename(phoenix_db_filename)[13:17]) for phoenix_db_filename in grid_refstar_filelist])

                print(grid_refstar_filelist[0])

            plt.subplot(len(postfilelist),nrowplot,0+nrowplot*spec_id+1)
            plt.imshow(posterior_rv_vsini,interpolation="nearest",origin="lower",extent=[vsini_samples[0],vsini_samples[-1],rv_samples[0],rv_samples[-1]])
            plt.ylabel("RV (km/s)")
            plt.xlabel("vsini (km/s)")

            rv_posterior = np.nansum(posterior_rv_vsini,axis=1)
            rv_posterior/=np.nanmax(rv_posterior)
            vsini_posterior = np.nansum(posterior_rv_vsini,axis=0)
            vsini_posterior/=np.nanmax(vsini_posterior)
            bestrv,_,_,bestrv_merr,bestrv_perr,_ = get_err_from_posterior(rv_samples,rv_posterior)
            bestrv_merr = np.abs(bestrv_merr)
            plt.subplot(len(postfilelist),nrowplot,1+nrowplot*spec_id+1)
            plt.plot(rv_samples,rv_posterior)
            plt.xlabel("RV (km/s)")
            plt.title("RV={0:.2f}-{1:.2f}+{2:.2f}".format(bestrv,bestrv_merr,bestrv_perr))
            rvs[spec_id] = bestrv
            rvs_perr[spec_id] = bestrv_perr
            rvs_nerr[spec_id] = bestrv_merr

            bestvsini,_,_,bestvsini_merr,bestvsini_perr,_ = get_err_from_posterior(vsini_samples,vsini_posterior)
            bestvsini_merr = np.abs(bestvsini_merr)
            plt.subplot(len(postfilelist),nrowplot,2+nrowplot*spec_id+1)
            plt.plot(vsini_samples,vsini_posterior)
            plt.xlabel("vsini (km/s)")
            plt.title("vsin={0:.2f}-{1:.2f}+{2:.2f}".format(bestvsini,bestvsini_merr,bestvsini_perr))

            best_model_id = np.argmax(posterior_model)
            plt.subplot(len(postfilelist),nrowplot,3+nrowplot*spec_id+1)
            plt.title(os.path.basename(grid_refstar_filelist[best_model_id]),fontsize=5)
            plt.scatter(Teff_grid_list,Fe_H_grid_list,s=100*posterior_model,c=logg_grid_list,alpha=0.5)
            plt.xlabel("$T_{eff}$ (K)")
            plt.ylabel("Fe/H")
            plt.colorbar(label="log(g)")

            hdulist = pyfits.open(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_best_data_fit.fits".format(starname,IFSfilter,date,type)))
            d_wvs = hdulist[0].data[0]
            d = hdulist[0].data[1]
            m = hdulist[0].data[2]
            res = hdulist[0].data[3]

            # plt.subplot(len(postfilelist),nrowplot,4+nrowplot*spec_id+1)
            plt.figure(3)
            plt.subplot(1,4,spec_id+1)
            plt.plot(d_wvs,d,label="data")
            plt.plot(d_wvs,m,label="model")
            plt.plot(d_wvs,res,label="res")
            plt.legend()
            plt.figure(1)

            hdulist = pyfits.open(os.path.join(osiris_data_dir,"stellar_fits","{0}_{1}_{2}_{3}_whole_spec_band.fits".format(starname,IFSfilter,date,type)))
            # print(hdulist[0].data.shape)
            # exit()
            wvs_corr = hdulist[0].data[0]
            spec = hdulist[0].data[1]
            spec_trans = hdulist[0].data[2]
            spec_trans_model = hdulist[0].data[3]

            plt.subplot(len(postfilelist),nrowplot,5+nrowplot*spec_id+1)
            plt.plot(wvs_corr,spec/np.nanmean(spec),label="data")
            plt.plot(wvs_corr,spec_trans/np.nanmean(spec_trans)+1,label="data/trans")
            plt.plot(wvs_corr,spec_trans_model/np.nanmean(spec_trans_model)+2,label="data/trans/star")
            plt.title("{0} {1} {2}".format(IFSfilter,date,type))
            plt.legend(loc="center left",bbox_to_anchor = (1,0.5))
        # plt.show()

        plt.tight_layout()
        # print("Saving "+os.path.join(osiris_data_dir,"stellar_fits","fit_quicklook_{0}.pdf".format(starname)))
        # plt.savefig(os.path.join(osiris_data_dir,"stellar_fits","fit_quicklook_{0}.png".format(starname)),bbox_inches='tight')
        # plt.savefig(os.path.join(osiris_data_dir,"stellar_fits","fit_quicklook_{0}.pdf".format(starname)),bbox_inches='tight')

        plt.figure(2,figsize=(10,5))
        plt.subplot(1,2,1)
        from astropy.time import Time
        times = ["20{0}-{1}-{2}T00:00:00".format(d[0:2],d[2:4],d[4:6]) for d in dates]
        print(times)
        print(dates)
        # exit()
        t = Time(times, format="isot",scale='utc')
        print(t.mjd)
        # plt.errorbar(np.arange(0,len(dates)),rvs,
        #              yerr=[rvs_nerr,rvs_perr],fmt="none",color="#ff9900")
        # plt.plot(np.arange(0,len(dates)),rvs,"x",color="#ff9900")
        plt.errorbar(t.mjd,rvs,
                     yerr=[rvs_nerr,rvs_perr],fmt="none",color="#ff9900")
        plt.plot(t.mjd,rvs,"x",color="#ff9900")
        plt.xlabel("Epochs")
        plt.ylabel("RV (km/s)")
        # plt.xticks(np.arange(0,len(dates)),dates)
        plt.subplot(1,2,2)
        plt.errorbar(np.arange(0,len(dates)),rvs,
                     yerr=[rvs_nerr,rvs_perr],fmt="none",color="#ff9900")
        plt.plot(np.arange(0,len(dates)),rvs,"x",color="#ff9900")
        plt.xlabel("Epochs")
        plt.ylabel("RV (km/s)")
        plt.xticks(np.arange(0,len(dates)),dates)
        plt.show()

        plt.tight_layout()
        print("Saving "+os.path.join(osiris_data_dir,"stellar_fits","stellar_RVs_{0}.pdf".format(starname)))
        plt.savefig(os.path.join(osiris_data_dir,"stellar_fits","stellar_RVs_{0}.png".format(starname)),bbox_inches='tight')
        plt.savefig(os.path.join(osiris_data_dir,"stellar_fits","stellar_RVs_{0}.pdf".format(starname)),bbox_inches='tight')

        plt.show()

        exit()


