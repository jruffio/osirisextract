__author__ = 'jruffio'

import sys
import multiprocessing as mp
import numpy as np
from copy import copy
from scipy.ndimage.filters import median_filter
import astropy.io.fits as pyfits
import itertools
from scipy import interpolate
import glob
import os
import csv
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
import ctypes
import pandas as pd

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

def _tpool_init(_transmission_vec,_nospec_planet_model,_wvs,_sigmas_vec,_where_finite_data,_ravelHPFdata,_oriplanet_spec_wvs,_HPFmodel_H0,
                _transmission_vec_shape,_nospec_planet_model_shape,_wvs_shape,_sigmas_vec_shape,_where_finite_data_shape,_ravelHPFdata_shape,_oriplanet_spec_wvs_shape,_HPFmodel_H0_shape):
    """
    Initializer function for the thread pool that initializes various shared variables. Main things to note that all
    except the shapes are shared arrays (mp.Array).

    Args:
    """
    global transmission_vec,nospec_planet_model,wvs,sigmas_vec,where_finite_data,ravelHPFdata,oriplanet_spec_wvs,HPFmodel_H0, \
        transmission_vec_shape,nospec_planet_model_shape,wvs_shape,sigmas_vec_shape,where_finite_data_shape,ravelHPFdata_shape,oriplanet_spec_wvs_shape,HPFmodel_H0_shape
    # original images from files to read and align&scale. Shape of (N,y,x)
    transmission_vec=_transmission_vec
    nospec_planet_model=_nospec_planet_model
    wvs=_wvs
    sigmas_vec=_sigmas_vec
    where_finite_data=_where_finite_data
    ravelHPFdata=_ravelHPFdata
    oriplanet_spec_wvs=_oriplanet_spec_wvs
    HPFmodel_H0=_HPFmodel_H0

    transmission_vec_shape=_transmission_vec_shape
    nospec_planet_model_shape=_nospec_planet_model_shape
    wvs_shape=_wvs_shape
    sigmas_vec_shape=_sigmas_vec_shape
    where_finite_data_shape=_where_finite_data_shape
    ravelHPFdata_shape=_ravelHPFdata_shape
    oriplanet_spec_wvs_shape=_oriplanet_spec_wvs_shape
    HPFmodel_H0_shape=_HPFmodel_H0_shape


def LPFvsHPF(myvec,cutoff,nansmooth=10):
    myvec_cp = np.zeros(myvec.shape)
    myvec_cp[:] = copy(myvec[:])
    wherenans = np.where(np.isnan(myvec_cp))
    myvec_cp = np.array(pd.DataFrame(myvec_cp).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:,0]
    # for k in wherenans[0]:
    #     myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-nansmooth]):np.min([np.size(myvec_cp),k+nansmooth])])

    fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec

    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan
    return LPF_myvec,HPF_myvec

def _spline_psf_model(paras):
    psfs,xs,ys,xvec,yvec,chunk_id = paras
    normalized_psfs_func_list = []
    for wv_index in range(psfs.shape[1]):
        model_psf = psfs[:,wv_index,:,:].ravel()
        where_nans = np.where(np.isfinite(model_psf))
        psf_func = interpolate.LSQBivariateSpline(xs.ravel()[where_nans],ys.ravel()[where_nans],model_psf[where_nans],xvec,yvec,kx=3,ky=3,eps=0.01)
        normalized_psfs_func_list.append(psf_func)
    return chunk_id,normalized_psfs_func_list



def get_rv_logpost(paras):
    global transmission_vec,nospec_planet_model,wvs,sigmas_vec,where_finite_data,ravelHPFdata,oriplanet_spec_wvs,HPFmodel_H0, \
        transmission_vec_shape,nospec_planet_model_shape,wvs_shape,sigmas_vec_shape,where_finite_data_shape,ravelHPFdata_shape,oriplanet_spec_wvs_shape,HPFmodel_H0_shape
    paras_list,myinterpgrid,planetRV_array,star_flux,cutoff,logdet_Sigma= paras


    oriplanet_spec_wvs_np =  _arraytonumpy(oriplanet_spec_wvs, oriplanet_spec_wvs_shape,dtype=dtype)
    transmission_vec_np = _arraytonumpy(transmission_vec, transmission_vec_shape,dtype=dtype)
    nospec_planet_model_np = _arraytonumpy(nospec_planet_model, nospec_planet_model_shape,dtype=dtype)
    wvs_np = _arraytonumpy(wvs, wvs_shape,dtype=dtype)
    sigmas_vec_np = _arraytonumpy(sigmas_vec, sigmas_vec_shape,dtype=dtype)
    where_finite_data_np = _arraytonumpy(where_finite_data, where_finite_data_shape,dtype=np.int)
    ravelHPFdata_np = _arraytonumpy(ravelHPFdata, ravelHPFdata_shape,dtype=dtype)
    HPFmodel_H0_np = _arraytonumpy(HPFmodel_H0, HPFmodel_H0_shape,dtype=dtype)

    cp_HPFmodel_H0 = copy(HPFmodel_H0_np)
    cp_HPFmodel_H0[np.where(np.isnan(HPFmodel_H0_np))] = 0


    logpost_rv = np.zeros((len(paras_list),np.size(planetRV_array)))
    for temp_id,(temp,fitlogg,CtoO) in enumerate(paras_list):
        w = int((nospec_planet_model_np.shape[0]-1)/2)
        c_kms = 299792.458
        # print(temp,fitlogg,CtoO)
        planet_template_func = interp1d(oriplanet_spec_wvs,myinterpgrid([temp,fitlogg,CtoO])[0],bounds_error=False,fill_value=np.nan)

        for plrv_id,plrv in enumerate(planetRV_array):
            planet_model = copy(nospec_planet_model_np)
            for bkg_k in range(2*w+1):
                for bkg_l in range(2*w+1):
                    # print(wvs.shape,plrv,c_kms)
                    wvs4planet_model = wvs_np[bkg_k,bkg_l,:]*(1-(plrv)/c_kms)
                    planet_model[bkg_k,bkg_l,:] *= planet_template_func(wvs4planet_model) * transmission_vec_np

            planet_model = planet_model/np.nansum(planet_model)*star_flux*1e-5
            HPF_planet_model = np.zeros(planet_model.shape)
            for bkg_k in range(2*w+1):
                for bkg_l in range(2*w+1):
                    HPF_planet_model[bkg_k,bkg_l,:]  = LPFvsHPF(planet_model[bkg_k,bkg_l,:] ,cutoff)[1]

            HPFmodel_H1only = (HPF_planet_model.ravel())[:,None]

            HPFmodel_H1only = HPFmodel_H1only[where_finite_data_np,:]/sigmas_vec_np[:,None] # where_finite_data[0]
            HPFmodel_H1only[np.where(np.isnan(HPFmodel_H1only))] = 0

            # print(HPFmodel_H1only.shape,HPFmodel_H0.shape)
            # print(HPFmodel_H1only[0:5])
            # print(HPFmodel_H0[0:5,0:5])
            # print(plrv)
            # exit()
            HPFmodel = np.concatenate([HPFmodel_H1only,cp_HPFmodel_H0],axis=1)

            where_valid_parameters = np.where(np.nansum(np.abs(HPFmodel)>0,axis=0)>=50)
            HPFmodel = HPFmodel[:,where_valid_parameters[0]]

            # print(HPFmodel.shape,ravelHPFdata.shape)
            HPFparas,HPFchi2,rank,s = np.linalg.lstsq(HPFmodel,ravelHPFdata_np,rcond=None)
            # print(HPFparas)
            # print(HPFparas.shape)
            # exit()
            data_model = np.dot(HPFmodel,HPFparas)
            ravelresiduals = ravelHPFdata_np-data_model
            HPFchi2 = np.nansum((ravelresiduals)**2)
            Npixs_HPFdata = HPFmodel.shape[0]
            covphi =  HPFchi2/Npixs_HPFdata*np.linalg.inv(np.dot(HPFmodel.T,HPFmodel))
            slogdet_icovphi0 = np.linalg.slogdet(np.dot(HPFmodel.T,HPFmodel))

            logpost_rv[temp_id,plrv_id] = -0.5*logdet_Sigma-0.5*slogdet_icovphi0[1]- (Npixs_HPFdata-HPFmodel.shape[-1]+2-1)/(2)*np.log(HPFchi2)
    return logpost_rv
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

    print(len(sys.argv))
    if len(sys.argv) == 1:
        osiris_data_dir = "/data/osiris_data/"
        # IFSfilter = "Kbb"
        # planet = "HR_8799_d"
        IFSfilter = "Kbb"
        planet = "HR_8799_c"
        scale = "*"
        date = "*"
        inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/"
        filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
        filelist.sort()
        # modelfolder = "20200309_model"
        # outputfolder = "20200309_model"
        modelfolder = "20200427_model_fk"
        outputfolder = "20200427_model_fk"
        gridname = os.path.join("/data/osiris_data/","hr8799b_modelgrid")
        N_kl = 10
        numthreads = 16#16
        small = True
        inj_fake = 2e-5  #2e-5 #None
        # for filename in filelist:
        #     print(filename)
        # print(outputdir)
        # exit()
    else:
        #

        osiris_data_dir = sys.argv[1]
        modelfolder = sys.argv[2]
        outputfolder = sys.argv[3]
        filename = sys.argv[4]
        numthreads = int(sys.argv[5])
        gridname = sys.argv[6]
        N_kl = int(sys.argv[7])
        small = bool(int(sys.argv[8]))
        try:
            inj_fake = float(sys.argv[9])
        except:
            inj_fake = None

        filelist = [filename]
        IFSfilter = filename.split("_")[-2]
        planet = filename.split(os.path.sep)[3]
        # date = os.path.basename(filename).split("_")[0].replace("s","")

    Tfk,loggfk,ctoOfk = 1000,3.75,0.7

    fitT_list = np.linspace(800,1200,21,endpoint=True)
    fitlogg_list = np.linspace(3,4.5,46,endpoint=True)
    fitCtoO_list = np.linspace(0.45708819,0.89125094,80,endpoint=True)
    # fitCtoO_list = np.linspace(10**(8.48 - 8.82),10**(8.33 - 8.51),40,endpoint=True)
    print(fitCtoO_list)
    # exit()
    # fitT_list = np.linspace(1000,1200,2,endpoint=True)
    # fitlogg_list = np.linspace(3.75,4.5,2,endpoint=True)
    # fitCtoO_list = np.linspace(0.7,0.89125094,2,endpoint=True)
    # fitT_list = np.arange(900,1200,50)
    # fitlogg_list = np.arange(-4.5,-3,0.25)
    # fitCtoO_list = np.arange(10**(8.48 - 8.82),10**(8.33 - 8.51),0.005)
    print(fitT_list.shape)
    print(fitlogg_list.shape)
    print(fitCtoO_list.shape)
    # exit()
# /data/osiris_data/hr8799b_modelgrid/lte11-4.0-0.0.aces_hr8799b_pgs=4d6_Kzz=1d8_C=8.31_O=8.46_gs=5um.exoCH4_hiresHK.7.D2e.sorted_gaussconv_R4000_Kbb.csv
#     fitT_list = np.array([1100])
#     fitlogg_list = np.array([-4])
#     fitCtoO_list = np.array([0.7079457843841374])
    planetRV_array0 = np.arange(-20,20,1)
    # planetRV_array0 = np.arange(-1,1,1)

    c_kms = 299792.458
    cutoff = 40
    R= 4000

    if 0:
        planet_model_list = []

        if "interpolated" not in  gridname:
            if "sonora" in gridname:
                grid_filelist = glob.glob(os.path.join(gridname,"sp_t*g*nc_m[0-9].[0-9]"))
                gridconv_filelist = [grid_filename+"_gaussconv_R{0}_{1}.csv".format(R,IFSfilter) for grid_filename in grid_filelist]
            elif "BTsettl" in gridname:
                grid_filelist = glob.glob(os.path.join(gridname,"lte*BT-Settl.spec.fits"))
                gridconv_filelist = [grid_filename.replace(".fits","_gaussconv_R{0}_{1}.csv".format(R,IFSfilter)) for grid_filename in grid_filelist]
            elif "hr8799b_modelgrid" in gridname:
                grid_filelist = glob.glob(os.path.join(gridname,"lte*-*-0.0.aces_hr8799b_pgs=4d6_Kzz=1d8_C=*_O=*_gs=5um.exoCH4_hiresHK.7.D2e.sorted"))
                gridconv_filelist = [grid_filename.replace("hiresHK.7.D2e.sorted","hiresHK.7.D2e.sorted_gaussconv_R{0}_{1}.csv".format(R,IFSfilter)) for grid_filename in grid_filelist]
        else:
            if "sonora" in gridname:
                gridconv_filelist = glob.glob(os.path.join(gridname,"sp_t*g*nc_m[0-9].[0-9]"+"_gaussconv_R{0}_{1}.csv".format(R,IFSfilter)))
            elif "BTsettl" in gridname:
                gridconv_filelist = glob.glob(os.path.join(gridname,"lte*BT-Settl.spec"+"_gaussconv_R{0}_{1}.csv".format(R,IFSfilter)))
            grid_filelist = ["",]*len(gridconv_filelist)

        # grid_filelist = grid_filelist[::10]
        # gridconv_filelist = gridconv_filelist[::10]

        Tlist = np.array([int(float(os.path.basename(grid_filename).split("lte")[-1].split("-")[0])*100) for grid_filename in grid_filelist])
        logglist = np.array([float(os.path.basename(grid_filename).split("-")[1]) for grid_filename in grid_filelist])
        Clist = np.array([float(os.path.basename(grid_filename).split("C=")[-1].split("_O")[0]) for grid_filename in grid_filelist])
        Olist = np.array([float(os.path.basename(grid_filename).split("O=")[-1].split("_gs")[0]) for grid_filename in grid_filelist])
        CtoOlist = 10**(Clist-Olist)
        Tlistunique = np.unique(Tlist)
        logglistunique = np.unique(logglist)
        CtoOlistunique = np.unique(CtoOlist)
        print(Tlistunique)
        print(logglistunique)
        print(CtoOlistunique)
        # exit()

        #print(gridname)
        for file_id,(grid_filename,gridconv_filename) in enumerate(zip(grid_filelist,gridconv_filelist)):
            print(gridconv_filename)
            with open(gridconv_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                list_starspec = list(csv_reader)
                oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                col_names = oriplanet_spec_str_arr[0]
                oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
                oriplanet_spec /= np.nanmean(oriplanet_spec)
                oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
                # where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                # oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                # planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
                # planet_partial_template_func_list.append(planet_spec_func)
                planet_model_list.append(oriplanet_spec)
                if 0:
                    tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
                    hdulist = pyfits.open(tmpfilename)
                    planet_model_grid =  hdulist[0].data
                    oriplanet_spec_wvs =  hdulist[1].data
                    Tlistunique =  hdulist[2].data
                    logglistunique =  hdulist[3].data
                    CtoOlistunique =  hdulist[4].data
                    # Tlistunique =  hdulist[1].data
                    # logglistunique =  hdulist[2].data
                    # CtoOlistunique =  hdulist[3].data
                    hdulist.close()

                    print(planet_model_grid.shape,np.size(Tlistunique),np.size(logglistunique),np.size(CtoOlistunique),np.size(oriplanet_spec_wvs))
                    from scipy.interpolate import RegularGridInterpolator
                    myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

                    import matplotlib.pyplot as plt
                    print(Tlist[file_id],logglist[file_id],CtoOlist[file_id])
                    plt.plot(oriplanet_spec_wvs,myinterpgrid([Tlist[file_id],logglist[file_id],CtoOlist[file_id]])[0],label="interp")
                    plt.plot(oriplanet_spec_wvs,oriplanet_spec,label="ori",linestyle="--")
                    plt.legend()
                    plt.show()

        print(len(planet_model_list),np.size(Tlistunique)*np.size(logglistunique)*np.size(CtoOlistunique))
        print((np.size(Tlistunique),np.size(logglistunique),np.size(CtoOlistunique),np.size(oriplanet_spec_wvs)))
        if len(planet_model_list) != np.size(Tlistunique)*np.size(logglistunique)*np.size(CtoOlistunique):
            raise Exception("Missing model(s) to complete the grid")
        planet_model_grid = np.zeros((np.size(Tlistunique),np.size(logglistunique),np.size(CtoOlistunique),np.size(oriplanet_spec_wvs)))
        for T_id,T in enumerate(Tlistunique):
            for logg_id,logg in enumerate(logglistunique):
                for CtoO_id,CtoO in enumerate(CtoOlistunique):
                    planet_model_grid[T_id,logg_id,CtoO_id,:] = planet_model_list[np.where((Tlist==T)*(logglist==logg)*(CtoOlist==CtoO))[0][0]]
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=planet_model_grid))
        hdulist.append(pyfits.ImageHDU(data=oriplanet_spec_wvs))
        hdulist.append(pyfits.ImageHDU(data=Tlistunique))
        hdulist.append(pyfits.ImageHDU(data=logglistunique))
        hdulist.append(pyfits.ImageHDU(data=CtoOlistunique))
        try:
            hdulist.writeto(os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter)), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter)), clobber=True)
        hdulist.close()
    else:
        tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
        hdulist = pyfits.open(tmpfilename)
        planet_model_grid =  hdulist[0].data
        oriplanet_spec_wvs =  hdulist[1].data
        Tlistunique =  hdulist[2].data
        logglistunique =  hdulist[3].data
        CtoOlistunique =  hdulist[4].data
        # Tlistunique =  hdulist[1].data
        # logglistunique =  hdulist[2].data
        # CtoOlistunique =  hdulist[3].data
        hdulist.close()

        print(planet_model_grid.shape,np.size(Tlistunique),np.size(logglistunique),np.size(CtoOlistunique),np.size(oriplanet_spec_wvs))
        from scipy.interpolate import RegularGridInterpolator
        myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

        # import matplotlib.pyplot as plt
        # print(Tlistunique[0],logglistunique[0],CtoOlistunique[0])
        # plt.plot(oriplanet_spec_wvs,myinterpgrid([Tlistunique[0],logglistunique[0],CtoOlistunique[0]])[0])
        # plt.show()

    # exit()
    logpost = np.zeros((len(fitT_list),len(fitlogg_list),len(fitCtoO_list),len(planetRV_array0)))

    for file_id, filename in enumerate(filelist[0::]):
        print(filename)

        #_corrwvs _LPFdata _HPFdata _badpix _sigmas _trans _starspec _reskl _plrv0

        if inj_fake is not None:
            inj_fake_str = "_fk"
        else:
            inj_fake_str = ""

        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_corrwvs"+inj_fake_str+".fits"))
        if len(glob.glob(tmpfilename))!=1:
            print("No data on "+filename)
            continue
        hdulist = pyfits.open(tmpfilename)
        wvs =  hdulist[0].data
        hdulist.close()

        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_LPFdata"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            LPFdata =  hdulist[0].data[1:6,1:6,:]
        else:
            LPFdata =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_HPFdata"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            HPFdata =  hdulist[0].data[1:6,1:6,:]
        else:
            HPFdata =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_badpix"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            data_badpix =  hdulist[0].data[1:6,1:6,:]
        else:
            data_badpix =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_sigmas"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            data_sigmas =  hdulist[0].data[1:6,1:6,:]
        else:
            data_sigmas =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_trans"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        transmission_vec =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_starspec"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            star_obsspec =  hdulist[0].data[1:6,1:6,:]
        else:
            star_obsspec =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_reskl"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        res4model_kl =  hdulist[0].data[:,0:N_kl]
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_plrv0"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        plrv0 =  hdulist[0].data
        data_ny,data_nx,data_nz = HPFdata.shape
        w = int((data_nx-1)//2)
        star_flux = np.nansum(star_obsspec[w,w,:])
        hdulist.close()

        ##############################
        ## Create PSF model
        ##############################
        ref_star_folder = os.path.join(os.path.dirname(filename),"..","reduced_telluric_jb")
        with pyfits.open(glob.glob(os.path.join(ref_star_folder,"*"+IFSfilter+"_hdpsfs_v2.fits"))[0]) as hdulist:
            psfs_refstar_arr = hdulist[0].data[None,:,:,:]
        with pyfits.open(glob.glob(os.path.join(ref_star_folder,"*"+IFSfilter+"_hdpsfs_xy_v2.fits"))[0]) as hdulist:
            hdpsfs_xy = hdulist[0].data
            hdpsfs_x,hdpsfs_y = hdpsfs_xy

        nx_psf,ny_psf = 15,15
        nz_psf = psfs_refstar_arr.shape[1]
        x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
        x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)

        print("starting psf")
        specpool = mp.Pool(processes=numthreads)
        chunk_size=20
        N_chunks = nz_psf//chunk_size
        psfs_chunks = []
        for k in range(N_chunks-1):
            psfs_chunks.append(psfs_refstar_arr[:,k*chunk_size:(k+1)*chunk_size,:,:])
        psfs_chunks.append(psfs_refstar_arr[:,(N_chunks-1)*chunk_size:nz_psf,:,:])
        outputs_list = specpool.map(_spline_psf_model, zip(psfs_chunks,
                                                           itertools.repeat(hdpsfs_x[None,:,:]),
                                                           itertools.repeat(hdpsfs_y[None,:,:]),
                                                           itertools.repeat(x_psf_grid[0,0:nx_psf-1]+0.5),
                                                           itertools.repeat(y_psf_grid[0:ny_psf-1,0]+0.5),
                                                           np.arange(len(psfs_chunks))))

        normalized_psfs_func_list = []
        chunks_ids = []
        for out in outputs_list:
            normalized_psfs_func_list.extend(out[1])
            chunks_ids.append(out[0])
        print("finish psf")
        specpool.close()
        specpool.join()
        print("closed psf")

        dx,dy = 0,0
        nospec_planet_model = np.zeros(HPFdata.shape)
        pl_x_vec = np.arange(-w,w+1) + dx
        pl_y_vec = np.arange(-w,w+1) + dy
        for z in range(data_nz):
            nospec_planet_model[:,:,z] = normalized_psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()


        ravelHPFdata = np.ravel(HPFdata)
        ravelLPFdata = np.ravel(LPFdata)
        ravelsigmas = np.ravel(data_sigmas)
        where_finite_data = np.where(np.isfinite(np.ravel(data_badpix)))
        where_bad_data = np.where(~(np.isfinite(np.ravel(data_badpix))))
        ravelLPFdata = ravelLPFdata[where_finite_data]
        sigmas_vec = ravelsigmas[where_finite_data]#np.ones(ravelLPFdata.shape)#np.sqrt(np.abs(ravelLPFdata))
        ravelHPFdata = ravelHPFdata[where_finite_data]
        ravelHPFdata = ravelHPFdata/sigmas_vec
        logdet_Sigma = np.sum(2*np.log(sigmas_vec))

        # planetRV_array = np.array([plrv0])
        planetRV_array = planetRV_array0+plrv0

        HPFmodelH0_list = []
        if 1:
            bkg_model = np.zeros((2*w+1,2*w+1,2*w+1,2*w+1,data_nz))
            for bkg_k in range(2*w+1):
                for bkg_l in range(2*w+1):
                    if 1:
                        star_obsspec_tmp = star_obsspec[bkg_k,bkg_l]
                        smooth_model = median_filter(star_obsspec_tmp,footprint=np.ones(50),mode="reflect")
                        where_bad_pix_4model = np.where(np.isnan(data_badpix[bkg_k,bkg_l,:]))
                        star_obsspec_tmp[where_bad_pix_4model] = smooth_model[where_bad_pix_4model]
                        star_obsspec_tmp[np.where(np.isnan(HPFdata[bkg_k,bkg_l,:]))] = np.nan
                    LPF_star_obsspec_tmp,_ = LPFvsHPF(star_obsspec_tmp,cutoff)

                    myspec = LPFdata[bkg_k,bkg_l,:]*star_obsspec_tmp/LPF_star_obsspec_tmp
                    if 1:
                        smooth_model = median_filter(myspec,footprint=np.ones(50),mode="reflect")
                        where_bad_pix_4model = np.where(np.isnan(data_badpix[bkg_k,bkg_l,:]))
                        myspec[where_bad_pix_4model] = smooth_model[where_bad_pix_4model]
                        myspec[np.where(np.isnan(HPFdata[bkg_k,bkg_l,:]))] = np.nan
                    _,myspec = LPFvsHPF(myspec,cutoff)

                    bkg_model[bkg_k,bkg_l,bkg_k,bkg_l,:] = myspec
            HPFmodelH0_list.append(np.reshape(bkg_model,((2*w+1)**2,(2*w+1)**2*data_nz)).transpose())
        if res4model_kl is not None:
            for kid in range(res4model_kl.shape[1]):
                res4model = res4model_kl[:,kid]
                LPF4resmodel = np.nansum(LPFdata*nospec_planet_model,axis=(0,1))/np.nansum(nospec_planet_model**2,axis=(0,1))
                resmodel = nospec_planet_model*LPF4resmodel[None,None,:]*res4model[None,None,:]
                HPFmodelH0_list.append(np.ravel(resmodel)[:,None])

        HPFmodel_H0 = np.concatenate(HPFmodelH0_list,axis=1)

        HPFmodel_H0 = HPFmodel_H0[where_finite_data[0],:]/sigmas_vec[:,None]

        paras_id_list = []
        paras_list = []
        # planet_template_func_list = []
        for temp_id, temp in enumerate(fitT_list):
            for fitlogg_id,fitlogg in enumerate(fitlogg_list):
                for CtoO_id,CtoO in enumerate(fitCtoO_list):
                    paras_id_list.append((temp_id,fitlogg_id,CtoO_id))
                    paras_list.append((temp,fitlogg,CtoO))
                    # planet_template_func = interp1d(oriplanet_spec_wvs,myinterpgrid([temp,fitlogg,CtoO])[0],bounds_error=False,fill_value=np.nan)
                    # planet_template_func_list.append(planet_template_func)

                    # logpost[temp_id,fitlogg_id,CtoO_id,:] = get_rv_logpost((planet_template_func,planetRV_array,star_flux,cutoff,logdet_Sigma,transmission_vec,nospec_planet_model,wvs,sigmas_vec,where_finite_data,ravelHPFdata))

        # print(len(paras_id_list))
        # exit()

        if inj_fake is not None:
            planet_template_func = interp1d(oriplanet_spec_wvs,myinterpgrid([Tfk,loggfk,ctoOfk])[0],bounds_error=False,fill_value=np.nan)

            planet_model = copy(nospec_planet_model)
            for bkg_k in range(2*w+1):
                for bkg_l in range(2*w+1):
                    # print(wvs.shape,plrv,c_kms)
                    wvs4planet_model = wvs[bkg_k,bkg_l,:]*(1-(plrv0)/c_kms)
                    planet_model[bkg_k,bkg_l,:] *= planet_template_func(wvs4planet_model) * transmission_vec

            planet_model = planet_model/np.nansum(planet_model)*star_flux*inj_fake
            HPF_planet_model = np.zeros(planet_model.shape)
            for bkg_k in range(2*w+1):
                for bkg_l in range(2*w+1):
                    HPF_planet_model[bkg_k,bkg_l,:]  = LPFvsHPF(planet_model[bkg_k,bkg_l,:] ,cutoff)[1]

            HPFmodel_H1only = (HPF_planet_model.ravel())[:,None]

            HPFmodel_H1only = HPFmodel_H1only[where_finite_data[0],:]/sigmas_vec[:,None] # where_finite_data[0]
            HPFmodel_H1only[np.where(np.isnan(HPFmodel_H1only))] = 0

            # print(ravelHPFdata.shape,HPFmodel_H1only.shape)
            ravelHPFdata = ravelHPFdata+ HPFmodel_H1only[:,0]
            # print("YOUHOU")
            # print(ravelHPFdata.shape)
            # exit()


        ##############################
        ## INIT threads and shared memory
        #############################
        print(transmission_vec.shape)
        print(nospec_planet_model.shape)
        print(wvs.shape)
        print(sigmas_vec.shape)
        print(where_finite_data[0].shape)
        print(ravelHPFdata.shape)
        print(np.size(transmission_vec))
        print(np.size(nospec_planet_model))
        print(np.size(wvs))
        print(np.size(sigmas_vec))
        print(np.size(where_finite_data[0]))
        print(np.size(ravelHPFdata))
        # exit()
        dtype = ctypes.c_double
        _transmission_vec = mp.Array(dtype, np.size(transmission_vec))
        _transmission_vec_shape = transmission_vec.shape
        transmission_vec_np = _arraytonumpy(_transmission_vec, _transmission_vec_shape,dtype=dtype)
        transmission_vec_np[:] = transmission_vec[:]

        _nospec_planet_model = mp.Array(dtype, np.size(nospec_planet_model))
        _nospec_planet_model_shape = nospec_planet_model.shape
        nospec_planet_model_np = _arraytonumpy(_nospec_planet_model, _nospec_planet_model_shape,dtype=dtype)
        nospec_planet_model_np[:] = nospec_planet_model[:]

        _wvs = mp.Array(dtype, np.size(wvs))
        _wvs_shape = wvs.shape
        wvs_np = _arraytonumpy(_wvs, _wvs_shape,dtype=dtype)
        wvs_np[:] = wvs[:]

        _sigmas_vec = mp.Array(dtype, np.size(sigmas_vec))
        _sigmas_vec_shape = sigmas_vec.shape
        sigmas_vec_np = _arraytonumpy(_sigmas_vec, _sigmas_vec_shape,dtype=dtype)
        sigmas_vec_np[:] = sigmas_vec[:]

        _where_finite_data = mp.Array(dtype, np.size(where_finite_data[0]))
        _where_finite_data_shape = where_finite_data[0].shape
        where_finite_data_np = _arraytonumpy(_where_finite_data, _where_finite_data_shape,dtype=np.int)
        where_finite_data_np[:] = where_finite_data[0][:]

        _ravelHPFdata = mp.Array(dtype, np.size(ravelHPFdata))
        _ravelHPFdata_shape = ravelHPFdata.shape
        ravelHPFdata_np = _arraytonumpy(_ravelHPFdata, _ravelHPFdata_shape,dtype=dtype)
        ravelHPFdata_np[:] = ravelHPFdata[:]

        _oriplanet_spec_wvs = mp.Array(dtype, np.size(oriplanet_spec_wvs))
        _oriplanet_spec_wvs_shape = oriplanet_spec_wvs.shape
        oriplanet_spec_wvs_np = _arraytonumpy(_oriplanet_spec_wvs, _oriplanet_spec_wvs_shape,dtype=dtype)
        oriplanet_spec_wvs_np[:] = oriplanet_spec_wvs[:]

        _HPFmodel_H0 = mp.Array(dtype, np.size(HPFmodel_H0))
        _HPFmodel_H0_shape = HPFmodel_H0.shape
        HPFmodel_H0_np = _arraytonumpy(_HPFmodel_H0, _HPFmodel_H0_shape,dtype=dtype)
        HPFmodel_H0_np[:] = HPFmodel_H0[:]

        if numthreads==1:
            _tpool_init(_transmission_vec,_nospec_planet_model,_wvs,_sigmas_vec,_where_finite_data,_ravelHPFdata,_oriplanet_spec_wvs,_HPFmodel_H0,
                    _transmission_vec_shape,_nospec_planet_model_shape,_wvs_shape,_sigmas_vec_shape,_where_finite_data_shape,_ravelHPFdata_shape,_oriplanet_spec_wvs_shape,_HPFmodel_H0_shape)
            outlist = get_rv_logpost((paras_list,myinterpgrid,planetRV_array,star_flux,cutoff,logdet_Sigma))
            for paras_id,out in zip(paras_id_list,outlist):
                print(paras_id)
                temp_id,fitlogg_id,CtoO_id = paras_id
                logpost[temp_id,fitlogg_id,CtoO_id,:] = out
        else:
            # planetRV_array,star_flux,cutoff,logdet_Sigma
            tpool = mp.Pool(processes=numthreads, initializer=_tpool_init,
                            initargs=(_transmission_vec,_nospec_planet_model,_wvs,_sigmas_vec,_where_finite_data,_ravelHPFdata,_oriplanet_spec_wvs,_HPFmodel_H0,
                    _transmission_vec_shape,_nospec_planet_model_shape,_wvs_shape,_sigmas_vec_shape,_where_finite_data_shape,_ravelHPFdata_shape,_oriplanet_spec_wvs_shape,_HPFmodel_H0_shape),
                            maxtasksperchild=50)

            chunk_size=400
            N_chunks = len(paras_id_list)//chunk_size
            parasidlist_list = []
            paraslist_list = []
            speclist_list = []
            for k in range(N_chunks-1):
                parasidlist_list.append(paras_id_list[k*chunk_size:(k+1)*chunk_size])
                paraslist_list.append(paras_list[k*chunk_size:(k+1)*chunk_size])
                # speclist_list.append(planet_template_func_list[k*chunk_size:(k+1)*chunk_size])
            parasidlist_list.append(paras_id_list[(N_chunks-1)*chunk_size:len(paras_id_list)])
            paraslist_list.append(paras_list[(N_chunks-1)*chunk_size:len(paras_id_list)])
            # speclist_list.append(planet_template_func_list[(N_chunks-1)*chunk_size:len(planet_template_func_list)])

            print("starting paral")
            # print(out)
            # exit()
            outputs_list = tpool.map(get_rv_logpost, zip(paraslist_list,
                                                            itertools.repeat(myinterpgrid),
                                                            itertools.repeat(planetRV_array),
                                                            itertools.repeat(star_flux),
                                                            itertools.repeat(cutoff),
                                                            itertools.repeat(logdet_Sigma)))
            print("done paral. retrieving results")
            for myid,(parasidlist,outlist) in enumerate(zip(parasidlist_list,outputs_list)):
                print("myid",myid)
                for paras_id,out in zip(parasidlist,outlist):
                    print(paras_id)
                    temp_id,fitlogg_id,CtoO_id = paras_id
                    logpost[temp_id,fitlogg_id,CtoO_id,:] = out
        #

        # print(logpost.shape)
        # import matplotlib.pyplot as plt
        # for temp_id, temp in enumerate(fitT_list):
        #     for fitlogg_id,fitlogg in enumerate(fitlogg_list):
        #         for CtoO_id,CtoO in enumerate(fitCtoO_list):
        #             plt.plot(planetRV_array0,np.exp(logpost[temp_id,fitlogg_id,CtoO_id,:]-np.max(logpost)))
        # plt.show()

        print("creating path")
        if not os.path.exists(os.path.join(os.path.dirname(filename),outputfolder)):
            os.makedirs(os.path.join(os.path.dirname(filename),outputfolder))
        print("1")
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=logpost))
        hdulist.append(pyfits.ImageHDU(data=fitT_list))
        hdulist.append(pyfits.ImageHDU(data=fitlogg_list))
        hdulist.append(pyfits.ImageHDU(data=fitCtoO_list))
        hdulist.append(pyfits.ImageHDU(data=planetRV_array0))
        print("2")
        out = os.path.join(os.path.dirname(filename),outputfolder,os.path.basename(filename).replace(".fits","_kl{0}_logpost{1}.fits".format(N_kl,inj_fake_str)))
        print("saving" + out)
        try:
            hdulist.writeto(out, overwrite=True)
        except TypeError:
            hdulist.writeto(out, clobber=True)
        print("3")
        hdulist.close()
        print("4")
        if numthreads!=1:
            tpool.close()
            tpool.join()
        print("5")
