__author__ = 'jruffio'



import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pylab as plt
from PyAstronomy import funcFit as fuf
from PyAstronomy import pyasl
import scipy.integrate as sci
#
# # Create a spectrum with a single Gaussian
# # line using funcFit's GaussFit1d object.
# # Note that this object is not used for
# # fitting here, but only a calculate a
# # Gaussian.
# g = fuf.GaussFit1d()
# g["mu"] = 5005.
# g["A"] = -0.1
# g["sig"] = 0.1
# g["off"] = 1.0
#
# # Evaluate the spectrum with 0.01 A bin size
# wvl = np.linspace(5003., 5007., 400)
# flux = g.evaluate(wvl)
#
# # Obtain the broadened spectrum using
# # vsini = 13.3 km/s and no limb-darkening
# rflux = pyasl.rotBroad(wvl, flux, 0.0, 13.3)
#
# # Obtain the broadened spectrum using
# # vsini = 13.3 km/s and strong limb-darkening
# lflux = pyasl.rotBroad(wvl, flux, 0.9, 13.3)
#
# # Check that the area of the line did not change
# # in response to the broadening
# print("Initial EW [A]: ", 4. - sci.trapz(flux, wvl))
# print("After broadening without LD: ", 4. - sci.trapz(rflux, wvl))
# print("After broadening with LD: ", 4. - sci.trapz(lflux, wvl))
#
# # Plot the results
# plt.title("Rotational broadening")
# plt.xlabel("Wavelength [A]")
# plt.ylabel("Normalized flux")
# plt.plot(wvl, flux, 'b-')
# plt.plot(wvl, rflux, 'r-')
# plt.plot(wvl, lflux, 'g-')
# plt.show()


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
from astropy.stats import mad_std
from copy import copy
import csv
from scipy.interpolate import interp1d



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

        chunk_size=100
        N_chunks = np.size(spectrum)//chunk_size
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

def LPFvsHPF(myvec,cutoff):
    myvec_cp = copy(myvec)
    #handling nans:
    wherenans = np.where(np.isnan(myvec_cp))
    for k in wherenans[0]:
        myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-10]):np.min([np.size(myvec_cp),k+10])])

    fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec

    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan
    return LPF_myvec,HPF_myvec



def _task_fit_refstar(paras):
    vsini_arr, limbdark_arr,RV_arr,wvs,refstarpho_spec_func,atm_model,data = paras
    vsini_ravel= np.ravel(vsini_arr)
    limbdark_ravel= np.ravel(limbdark_arr)
    RV_ravel= np.ravel(RV_arr)
    posterior = np.zeros(vsini_arr.shape)
    posterior.shape = (np.size(vsini_arr),)
    c_kms = 299792.458

    for k,(vsini,limbdark,rv) in enumerate(zip(vsini_ravel,limbdark_ravel,RV_ravel)):
        refstarpho_spec = refstarpho_spec_func(wvs*(1-rv/c_kms))
        broadened_refstarpho = pyasl.rotBroad(wvs, refstarpho_spec, limbdark, vsini)
        broadened_refstarpho = LPFvsHPF(broadened_refstarpho,cutoff)[1]

        # refstarpho_spec_hpf = LPFvsHPF(refstarpho_spec_func(wvs*(1-rv/c_kms)),cutoff)[1]
        # broadened_refstarpho = pyasl.rotBroad(wvs, refstarpho_spec_hpf, limbdark, vsini)

        model = copy(atm_model)
        model.insert(0,broadened_refstarpho)
        model = np.array(model).transpose()

        sigmas_vec = np.ones(data.shape)
        logdet_Sigma = np.sum(2*np.log(sigmas_vec))
        model = model/sigmas_vec[:,None]
        data_sig = data/sigmas_vec

        where_finite_data = np.where(np.isfinite(data_sig))
        data_sig = data_sig[where_finite_data]
        model = model[where_finite_data[0],:]

        # for k in range(model.shape[1]):
        #     plt.plot(model[:,k],label="model")
        # plt.plot(data,label="data")
        # plt.plot(refstarpho_spec_hpf)
        # plt.legend()
        # plt.show()

        HPFparas,HPFchi2,rank,s = np.linalg.lstsq(model,data_sig,rcond=None)
        data_model = np.dot(model,HPFparas)
        residuals = np.zeros(data.shape) +np.nan
        residuals[where_finite_data] = data_model-data[where_finite_data]
        HPFchi2 = np.nansum((residuals)**2)
        slogdet_icovphi0 = np.linalg.slogdet(np.dot(model.T,model))

        posterior[k] = -0.5*logdet_Sigma-0.5*slogdet_icovphi0[1]- (model.shape[0]-model.shape[-1]+2-1)/(2)*np.log(HPFchi2)

    posterior.shape = vsini_arr.shape
    # vsini_ravel.shape = vsini_arr.shape
    # print(vsini_ravel)
    # print("posterior.shape",posterior.shape)
    return posterior

def fit_refstar(vsini_grid,limbdark_grid,RV_grid,wvs,refstarpho_spec_func,atm_model,data,mypool=None):
    if mypool is None:
        return _task_fit_refstar((vsini_grid,limbdark_grid,RV_grid,wvs,refstarpho_spec_func,atm_model,data))
    else:
        posterior = np.zeros(vsini_grid.shape)

        chunk_size=5
        N_chunks = vsini_grid.shape[1]//chunk_size
        indices_list = []
        vsini_grid_list = []
        limbdark_grid_list = []
        RV_grid_list = []
        for k in range(N_chunks-1):
            indices = np.arange(k*chunk_size,(k+1)*chunk_size).astype(np.int)
            indices_list.append(indices)
            vsini_grid_list.append(vsini_grid[:,indices,:])
            limbdark_grid_list.append(limbdark_grid[:,indices,:])
            RV_grid_list.append(RV_grid[:,indices,:])
        indices = np.arange((N_chunks-1)*chunk_size,vsini_grid.shape[1]).astype(np.int)
        indices_list.append(indices)
        vsini_grid_list.append(vsini_grid[:,indices,:])
        limbdark_grid_list.append(limbdark_grid[:,indices,:])
        RV_grid_list.append(RV_grid[:,indices,:])

        outputs_list = mypool.map(_task_fit_refstar, zip(vsini_grid_list,limbdark_grid_list,RV_grid_list,
                                                       itertools.repeat(wvs),
                                                       itertools.repeat(refstarpho_spec_func),
                                                       itertools.repeat(atm_model),
                                                       itertools.repeat(data)))
        for k,(indices,out) in enumerate(zip(indices_list,outputs_list)):
            posterior[:,indices,:] = out

        return posterior


try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

#------------------------------------------------
if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    if 0:
        OSIRISDATA = "/data/osiris_data/"
        # IFSfilter = "Jbb"
        IFSfilter = "Hbb"
        # IFSfilter = "Kbb"
        # refstar_name = "HD_210501"
        refstar_name = "HR_8799"
        # refstar_name = "BD+14_4774"
        cutoff = 5

        filelist = []
        date = "*"
        date = "20090723"
        date=   "20100713"
        filename_filter = "s*"+IFSfilter+"*020_psfs_repaired_spec_v2.fits"
        filelist.extend(glob.glob(os.path.join(OSIRISDATA,"HR_8799_*",date,"reduced_telluric_jb",refstar_name,filename_filter)))
        # filename_filter = "ao_off_s*"+IFSfilter+"*020_spec_v2.fits"
        # filelist.extend(glob.glob(os.path.join(OSIRISDATA,"HR_8799_*",date,"reduced_telluric_jb",refstar_name,filename_filter)))
        print(filelist)
        spec_filename = filelist[1]
        numthreads = 28
        print(OSIRISDATA)
        print(spec_filename)
        print(refstar_name)
        print(IFSfilter)
        print(numthreads)
        print(cutoff)
        # exit()
    else:
        OSIRISDATA = sys.argv[1]
        spec_filename = sys.argv[2]
        refstar_name = sys.argv[3]
        IFSfilter = sys.argv[4]
        numthreads = int(sys.argv[5])
        cutoff = int(sys.argv[6])
        #nice -n 15 /home/anaconda3/bin/python charac_reference_stars.py /data/osiris_data/ /data/osiris_data/HR_8799_b/20130726/reduced_telluric_jb/HR_8799/s130726_a063001_Jbb_020_psfs_repaired_spec_v2.fits HR_8799 Jbb 28 5

    if IFSfilter=="Kbb": #Kbb 1965.0 0.25
        CRVAL1 = 1965.
        CDELT1 = 0.25
        nl=1665
        R0=4000
    elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1473.
        CDELT1 = 0.2
        nl=1651
        R0=4000
    elif IFSfilter=="Jbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1180.
        CDELT1 = 0.15
        nl=1574
        R0=4000

    fileinfos_refstars_filename = os.path.join(OSIRISDATA,"fileinfos_refstars_jb.csv")
    with open(fileinfos_refstars_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        refstarsinfo_list_table = list(csv_reader)
        refstarsinfo_colnames = refstarsinfo_list_table[0]
        refstarsinfo_list_data = refstarsinfo_list_table[1::]
    refstarsinfo_filename_id = refstarsinfo_colnames.index("filename")
    refstarsinfo_filelist = [os.path.basename(item[refstarsinfo_filename_id]) for item in refstarsinfo_list_data]

    for fileid,refstarsinfo_file in enumerate(refstarsinfo_filelist):
        if os.path.basename(refstarsinfo_file).replace(".fits","") in spec_filename:
            fileitem = refstarsinfo_list_data[fileid]
            break

    type_id = refstarsinfo_colnames.index("type")
    Jmag_id = refstarsinfo_colnames.index("Jmag")
    Hmag_id = refstarsinfo_colnames.index("Hmag")
    Kmag_id = refstarsinfo_colnames.index("Kmag")
    rv_simbad_id = refstarsinfo_colnames.index("RV Simbad")
    starname_id = refstarsinfo_colnames.index("star name")

    refstar_RV = float(fileitem[rv_simbad_id])
    ref_star_type = fileitem[type_id]
    if IFSfilter == "Jbb":
        refstar_mag = float(fileitem[Jmag_id])
    elif IFSfilter == "Hbb":
        refstar_mag = float(fileitem[Hmag_id])
    elif IFSfilter == "Kbb":
        refstar_mag = float(fileitem[Kmag_id])

    if np.isnan(refstar_mag):
        raise(Exception("Ref star name unknown"))

    if 1:
        specpool = mp.Pool(processes=numthreads)

        sky_transmission_folder = os.path.join(OSIRISDATA,"sky_transmission")
        filelist_skytrans = glob.glob(os.path.join(sky_transmission_folder,"mktrans_zm_*_*.dat"))
        # if 1:
        #     filelist_skytrans = filelist_skytrans[3:5]

        atm_trans_list = []
        atm_trans_wvs_list = []
        for filename_skytrans in filelist_skytrans:
            if len(glob.glob(filename_skytrans.replace(".dat","_bb_R{0}.csv".format(R0)))) == 0:
                skybg_arr=np.loadtxt(filename_skytrans)
                skytrans_wvs = skybg_arr[:,0]
                skytrans_spec = skybg_arr[:,1]
                print("convolving: "+filename_skytrans)
                skytrans_spec_conv = convolve_spectrum(skytrans_wvs,skytrans_spec,R0,specpool)

                with open(filename_skytrans.replace(".dat","_bb_R{0}.csv".format(R0)), 'w+') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=' ')
                    csvwriter.writerows([["wvs","spectrum"]])
                    csvwriter.writerows([[a,b] for a,b in zip(skytrans_wvs,skytrans_spec_conv)])

            with open(filename_skytrans.replace(".dat","_bb_R{0}.csv".format(R0)), 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                list_starspec = list(csv_reader)
                spec_str_arr = np.array(list_starspec, dtype=np.str)
                col_names = spec_str_arr[0]
                skytrans_spec = spec_str_arr[1::,1].astype(np.float)
                skytrans_wvs = spec_str_arr[1::,0].astype(np.float)


            atm_trans_wvs_list.append(skytrans_wvs)
            atm_trans_list.append(interp1d(skytrans_wvs,skytrans_spec,bounds_error=False,fill_value=np.nan))


        # for wvs,spec in zip(atm_trans_wvs_list,atm_trans_list):
        #         plt.plot(wvs,spec(wvs))
        # plt.show()
        # exit()

        print(spec_filename)
        for refstarsinfo_id, refstarsinfo_file in enumerate(refstarsinfo_filelist):
            if os.path.basename(refstarsinfo_file).replace(".fits","") in spec_filename:
                refstarsinfo_fileid = refstarsinfo_id
                break
        refstarsinfo_fileitem = refstarsinfo_list_data[refstarsinfo_fileid]
        refstarsinfo_baryrv_id = refstarsinfo_colnames.index("barycenter rv")
        refstarsinfo_bary_rv = -float(refstarsinfo_fileitem[refstarsinfo_baryrv_id])/1000

        with pyfits.open(spec_filename) as hdulist:
            wvs = hdulist[0].data[0,:]
            spec = hdulist[0].data[1,:]


        phoenix_folder = os.path.join(OSIRISDATA,"phoenix")
        phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
        with pyfits.open(phoenix_wv_filename) as hdulist:
            phoenix_wvs = hdulist[0].data/1.e4
        crop_phoenix = np.where((phoenix_wvs>wvs[0]-(wvs[-1]-wvs[0])/2)*(phoenix_wvs<wvs[-1]+(wvs[-1]-wvs[0])/2))
        phoenix_wvs = phoenix_wvs[crop_phoenix]
        try:
            phoenix_model_refstar_filename = glob.glob(os.path.join(phoenix_folder,refstar_name+"*.fits"))[0]
        except:
            phoenix_model_refstar_filename = glob.glob(os.path.join(phoenix_folder,ref_star_type+"*.fits"))[0]
        phoenix_refstar_filename=phoenix_model_refstar_filename.replace(".fits","_gaussconv_R{0}_{1}.csv".format(R0,IFSfilter))

        if len(glob.glob(phoenix_refstar_filename)) == 0:
            with pyfits.open(phoenix_model_refstar_filename) as hdulist:
                phoenix_refstar = hdulist[0].data[crop_phoenix]
            print("convolving: "+phoenix_model_refstar_filename)

            phoenix_refstar_conv = convolve_spectrum(phoenix_wvs,phoenix_refstar,R0,specpool)

            with open(phoenix_refstar_filename, 'w+') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ')
                csvwriter.writerows([["wvs","spectrum"]])
                csvwriter.writerows([[a,b] for a,b in zip(phoenix_wvs,phoenix_refstar_conv)])

        with open(phoenix_refstar_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            list_starspec = list(csv_reader)
            refstarpho_spec_str_arr = np.array(list_starspec, dtype=np.str)
            col_names = refstarpho_spec_str_arr[0]
            refstarpho_spec = refstarpho_spec_str_arr[1::,1].astype(np.float)
            refstarpho_spec_wvs = refstarpho_spec_str_arr[1::,0].astype(np.float)
            where_IFSfilter = np.where((refstarpho_spec_wvs>wvs[0])*(refstarpho_spec_wvs<wvs[-1]))
            refstarpho_spec = refstarpho_spec/np.mean(refstarpho_spec[where_IFSfilter])
            refstarpho_spec_func = interp1d(refstarpho_spec_wvs,refstarpho_spec,bounds_error=False,fill_value=np.nan)



        data_lpf,data_hpf = LPFvsHPF(spec,cutoff)
        data_norma = np.nanstd(data_hpf)
        data = data_hpf/data_norma

        atm_model = []
        for atm_trans_func in atm_trans_list:
            tmp_model = LPFvsHPF(atm_trans_func(wvs)*data_lpf,cutoff)[1]
            tmp_model /= np.nanstd(tmp_model)
            atm_model.append(tmp_model)

        if 1:
            N_vsini = 500
            vsini_vec = np.linspace(1,500,N_vsini)
            N_limbdark = 4#5
            limbdark_vec = np.linspace(0,1,N_limbdark)
            # limbdark_vec = np.array([0.0,1.0])
            N_RV = 80*4+1#201
            RV_vec = np.linspace(-40+refstarsinfo_bary_rv+refstar_RV,40+refstarsinfo_bary_rv+refstar_RV,N_RV)
            # print(RV_vec)
            # print(refstarsinfo_bary_rv,refstar_RV)
            # exit()
            vsini_grid, limbdark_grid,RV_grid = np.meshgrid(vsini_vec, limbdark_vec, RV_vec)
        else:
            N_vsini = 100
            vsini_vec = np.linspace(1,500,N_vsini)
            N_limbdark = 2#5
            limbdark_vec = np.linspace(0,1,N_limbdark)
            # limbdark_vec = np.array([0.0])
            N_RV = 5#201
            RV_vec = np.linspace(-50+refstarsinfo_bary_rv+refstar_RV,50+refstarsinfo_bary_rv+refstar_RV,N_RV)
            # RV_vec = np.array([refstarsinfo_bary_rv+refstar_RV,])
            # print(RV_vec)
            # print(refstarsinfo_bary_rv,refstar_RV)
            # exit()
            vsini_grid, limbdark_grid,RV_grid = np.meshgrid(vsini_vec, limbdark_vec, RV_vec)

        posterior = fit_refstar(vsini_grid,
                                limbdark_grid,
                                RV_grid,
                                wvs,
                                refstarpho_spec_func,
                                atm_model,
                                data,
                                mypool=specpool)#specpool)

        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=posterior))
        try:
            hdulist.writeto(os.path.join(spec_filename.replace(".fits","_cutoff{0}_logpost.fits".format(cutoff))), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(spec_filename.replace(".fits","_cutoff{0}_logpost.fits".format(cutoff))), clobber=True)
        hdulist.close()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=vsini_vec))
        try:
            hdulist.writeto(os.path.join(spec_filename.replace(".fits","_cutoff{0}_logpost_vsini.fits".format(cutoff))), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(spec_filename.replace(".fits","_cutoff{0}_logpost_vsini.fits".format(cutoff))), clobber=True)
        hdulist.close()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=limbdark_vec))
        try:
            hdulist.writeto(os.path.join(spec_filename.replace(".fits","_cutoff{0}_logpost_limbdar.fits".format(cutoff))), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(spec_filename.replace(".fits","_cutoff{0}_logpost_limbdar.fits".format(cutoff))), clobber=True)
        hdulist.close()
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=RV_vec))
        try:
            hdulist.writeto(os.path.join(spec_filename.replace(".fits","_cutoff{0}_logpost_rv.fits".format(cutoff))), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(spec_filename.replace(".fits","_cutoff{0}_logpost_rv.fits".format(cutoff))), clobber=True)
        hdulist.close()

        specpool.close()
    if 1:
        print(spec_filename)
        post_filename = spec_filename.replace(".fits","_cutoff{0}_logpost.fits".format(cutoff))
        print(post_filename)
        with pyfits.open(post_filename) as hdulist:
            logposterior = hdulist[0].data
        with pyfits.open(post_filename.replace(".fits","_vsini.fits")) as hdulist:
            vsini_vec = hdulist[0].data
        with pyfits.open(post_filename.replace(".fits","_limbdar.fits")) as hdulist:
            limbdark_vec = hdulist[0].data
        with pyfits.open(post_filename.replace(".fits","_rv.fits")) as hdulist:
            RV_vec = hdulist[0].data

        posterior = np.exp(logposterior-np.nanmax(logposterior))
        plt.figure(1,figsize=(9,9))
        plt.subplot(3,3,1)
        plt.plot(vsini_vec,np.sum(posterior,axis=(0,2)))
        plt.xlabel("vsini (km/s)")
        plt.subplot(3,3,5)
        plt.plot(limbdark_vec,np.sum(posterior,axis=(1,2)))
        plt.xlabel("Limb Darkening")
        plt.subplot(3,3,9)
        plt.plot(RV_vec,np.sum(posterior,axis=(0,1)))
        plt.xlabel("Stellar RV")

        plt.subplot(3,3,4)
        plt.imshow(np.sum(posterior,axis=2),interpolation="nearest",origin="lower",extent=[vsini_vec[0],vsini_vec[-1],limbdark_vec[0],limbdark_vec[-1]],aspect="auto")
        plt.subplot(3,3,7)
        plt.imshow(np.sum(posterior,axis=0).T,interpolation="nearest",origin="lower",extent=[vsini_vec[0],vsini_vec[-1],RV_vec[0],RV_vec[-1]],aspect="auto")
        plt.subplot(3,3,8)
        plt.imshow(np.sum(posterior,axis=1).T,interpolation="nearest",origin="lower",extent=[limbdark_vec[0],limbdark_vec[-1],RV_vec[0],RV_vec[-1]],aspect="auto")


        if 1:
            c_kms = 299792.458

            max_k,max_l,max_m = np.unravel_index(np.argmax(posterior),posterior.shape)

            vsini,limbdark,rv=vsini_grid[max_k,max_l,max_m],limbdark_grid[max_k,max_l,max_m],RV_grid[max_k,max_l,max_m]

            refstarpho_spec = refstarpho_spec_func(wvs*(1-rv/c_kms))
            broadened_refstarpho = pyasl.rotBroad(wvs, refstarpho_spec, limbdark, vsini)
            broadened_refstarpho_hpf = LPFvsHPF(broadened_refstarpho,cutoff)[1]
            model = copy(atm_model)
            model.insert(0,broadened_refstarpho_hpf)
            model = np.array(model).transpose()

            sigmas_vec = np.ones(data.shape)
            logdet_Sigma = np.sum(2*np.log(sigmas_vec))
            model = model/sigmas_vec[:,None]
            data_sig = data/sigmas_vec

            where_finite_data = np.where(np.isfinite(data_sig))
            data_sig = data_sig[where_finite_data]
            model = model[where_finite_data[0],:]

            HPFparas,HPFchi2,rank,s = np.linalg.lstsq(model,data_sig,rcond=None)
            data_model = np.zeros(data.shape) +np.nan
            data_model[where_finite_data] = np.dot(model,HPFparas)
            residuals = data_model-data
            HPFchi2 = np.nansum((residuals)**2)

            plt.subplot(3,3,3)
            plt.plot(data,label="data")
            plt.plot(data_model,label="data_model")
            plt.plot(residuals,label="residuals")
            plt.legend()

            transmission_model = spec - broadened_refstarpho*HPFparas[0]*data_norma
            plt.subplot(3,3,6)
            plt.plot(spec,label="original")
            plt.plot(broadened_refstarpho*HPFparas[0]*data_norma,label="star model")
            plt.plot(transmission_model,label="transmission")
            plt.legend()

            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=transmission_model))
            try:
                hdulist.writeto(os.path.join(spec_filename.replace(".fits","_cutoff{0}_transmission.fits".format(cutoff))), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(spec_filename.replace(".fits","_cutoff{0}_transmission.fits".format(cutoff))), clobber=True)
            hdulist.close()

        plt.savefig(os.path.join(spec_filename.replace(".fits","_corner.png")),bbox_inches='tight')
        # plt.show()

        # plt.show()