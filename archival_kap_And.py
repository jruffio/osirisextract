__author__ = 'jruffio'

import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
from glob import glob
import os
from astropy import constants as const
import multiprocessing as mp
from PyAstronomy import pyasl
from scipy.interpolate import interp1d
from astropy import units as u
import scipy.io as scio
from copy import copy
from scipy.optimize import minimize
import warnings
from scipy import interpolate
import itertools
try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

from archival_kap_And_utils import read_osiris,findbadpix,aper_spec,convolve_spectrum,detecplanet,make_linear_model,get_spline_model,LPFvsHPF,like_fit_pixgauss2d

# def get_speckle_model(center, z0, z1, Nw,w, N_knots_per_speckle,xvec,yvec,zvec,spline_degree=3):
#     xref,yref,zref = 10,10
#     rref = np.sqrt((xref-center[0])**2+(yref-center[1])**2)
#     speckle_ref_loc_list =
#
#     nz,ny,nx = np.size(zvec),np.size(yvec),np.size(xvec)
#
#     x_vec, y_vec = np.arange(nx * 1.)-center[0],np.arange(ny* 1.)-center[1]
#     x_grid, y_grid = np.meshgrid(x_vec, y_vec)
#     r_grid = np.sqrt(x_grid**2+y_grid**2)
#
#     M = np.zeros((np.size(x_samples),(len(speckle_ref_loc_list)*N_knots_per_speckle)))
#     for knot,(x_knot, y_knot) in enumerate(speckle_ref_loc_list):
#         tmp_y_vec = np.zeros(np.size(x_knots))
#         tmp_y_vec[chunk] = 1
#         spl = InterpolatedUnivariateSpline(x_knots, tmp_y_vec, k=spline_degree, ext=0)
#         M[:,chunk] = spl(x_samples)
#     return M



def _spline_psf_model(paras):
    psfs,xs,ys,xvec,yvec,chunk_id = paras
    normalized_psfs_func_list = []
    for wv_index in range(psfs.shape[1]):
        if 0:#np.isnan(psf_func(0,0)[0,0]):
            model_psf = psfs[:,wv_index,:,:]
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(1)
            ax = fig.add_subplot(111,projection="3d")
            for k,color in zip(range(model_psf.shape[0]),["pink","blue","green","purple","orange"]):
                ax.scatter(xs[k].ravel(),ys[k].ravel(),model_psf[k].ravel(),c=color)
        model_psf = psfs[:,wv_index,:,:].ravel()
        where_nans = np.where(np.isfinite(model_psf))
        psf_func = interpolate.LSQBivariateSpline(xs.ravel()[where_nans],ys.ravel()[where_nans],model_psf[where_nans],xvec,yvec,kx=3,ky=3,eps=0.01)
        if 0:
            print(psf_func(0,0))
            x_psf_vec, y_psf_vec = np.arange(2*nx_psf * 1.)/2.-nx_psf//2, np.arange(2*ny_psf* 1.)/2.-ny_psf//2
            x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
            ax.scatter(x_psf_grid.ravel(),y_psf_grid.ravel(),psf_func(x_psf_vec,y_psf_vec).transpose().ravel(),c="red")
            plt.show()
        normalized_psfs_func_list.append(psf_func)
    # print(len(normalized_psfs_func_list))
    return chunk_id,normalized_psfs_func_list

#------------------------------------------------
if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass
    warnings.filterwarnings('ignore')

    # hdfactor = 2
    # ny,nx = 2,2
    # xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor*nx).astype(np.float)/hdfactor,np.arange(hdfactor*ny).astype(np.float)/hdfactor)
    # # gaussA_hd = A/(2*np.pi*w**2)*np.exp(-0.5*((xA-xhdgrid)**2+(yA-yhdgrid)**2)/w**2)
    # # gaussA = np.nanmean(,axis=(1,3))
    # print(np.reshape(xhdgrid,(ny,hdfactor,nx,hdfactor)))
    # print(np.reshape(yhdgrid,(ny,hdfactor,nx,hdfactor))[:,:,0,0])
    # exit()

    # test = "/data/osiris_data/kap_And/20161106/reduced_jb/sherlock/20191104_RVsearch/s161106_a020002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl0.fits"
    # with pyfits.open(test) as hdulist:
    #     arr = hdulist[0].data[0,0,10,212,:,:]
    # arr_cp = copy(arr)
    # arr_cp[30:40,:] = np.nan
    # snr_scaling = np.nanstd(arr_cp[3:30,5:15])
    # plt.subplot(1,2,1)
    # plt.imshow(arr/snr_scaling,interpolation="nearest",origin="lower")
    # plt.clim([-5,20])
    # plt.subplot(1,2,2)
    # plt.imshow(arr,interpolation="nearest",origin="lower")
    # plt.clim([-10,40])
    # plt.figure(3)
    # SNR_hist,bin_edges = np.histogram(np.ravel(arr_cp)[np.where(np.isfinite(np.ravel(arr_cp)))],bins=200,range=[-50,50])
    # bin_center = (bin_edges[1::]+bin_edges[0:np.size(bin_edges)-1])/2
    # SNR_hist2,bin_edges = np.histogram(np.ravel(arr_cp/snr_scaling)[np.where(np.isfinite(np.ravel(arr_cp)))],bins=200,range=[-50,50])
    # bin_center = (bin_edges[1::]+bin_edges[0:np.size(bin_edges)-1])/2
    # plt.plot(bin_center,SNR_hist)
    # plt.plot(bin_center,SNR_hist2)
    # plt.xlim([-5,5])
    # print(arr.shape)
    # plt.show()
    # exit()

    # out_pngs = "/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov02/out_Hbb/"
    out_pngs = "/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov03/out_Kbb/"


    R=4000
    numthreads=32

    # dir = "/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov03/reduced_jb_Kbb/"
    # tel_dir = "/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov03/reduced_telluric_Kbb/"
    # filelist = glob(os.path.join(dir,"*_Kbb_020.fits"))
    # # filelist = glob(os.path.join(dir,"s131103_a00500[0-9]_Kbb_020.fits"))
    # tel_filelist = glob(os.path.join(tel_dir,"*_Kbb_020.fits"))

    # dir = "/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov02/reduced_jb_Kbb/"
    # filelist = glob(os.path.join(dir,"*_Kbb_020.fits"))
    # tel_dir = "/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov03/reduced_telluric_Kbb/"
    # tel_filelist = glob(os.path.join(tel_dir,"*_Kbb_020.fits"))

    # dir = "/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov02/reduced_jb_Hbb/"
    # filelist = glob(os.path.join(dir,"*a014003*_020.fits"))
    # tel_dir = "/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov02/reduced_telluric_Hbb/"
    # tel_filelist = glob(os.path.join(tel_dir,"*_020.fits")) #/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov02/reduced_jb_Hbb/s131102_a014003_Hbb_020.fits

    # dir = "/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov03/reduced_jb_Jbb/"
    # filelist = glob(os.path.join(dir,"*_020.fits"))
    # tel_dir = "/data/osiris_data/kap_And/archive/KOA_73443/OSIRIS/2013nov03/reduced_telluric_Jbb/"
    # tel_filelist = glob(os.path.join(tel_dir,"*_020.fits"))

    # dir = "/data/osiris_data/kap_And/20161106/reduced_jb/"
    # # filelist = glob(os.path.join(dir,"*_020.fits"))
    # filelist = glob(os.path.join(dir,"*a019002*_020.fits"))
    # tel_dir = "/data/osiris_data/kap_And/20161106/reduced_telluric_jb/HIP_111538/"
    # tel_filelist = glob(os.path.join(tel_dir,"*_020.fits"))

    dir = "/data/osiris_data/kap_And/20161106/reduced_jb/"
    filelist = glob(os.path.join(dir,"*_020.fits"))
    # filelist = glob(os.path.join(dir,"*a019002*_020.fits"))
    tel_dir = "/data/osiris_data/kap_And/20161106/reduced_telluric_jb/HIP_111538/"
    tel_filelist = glob(os.path.join(tel_dir,"*_020.fits"))


    # dir = "/data/osiris_data/HR_8799_c/20200729/reduced_jb/"
    # filelist = glob(os.path.join(dir,"*_020.fits"))
    # tel_dir = "/data/osiris_data/HR_8799_c/20200729/reduced_telluric_jb/HR_8799/"
    # tel_filelist = glob(os.path.join(tel_dir,"*_020.fits"))

    # dir = "/data/osiris_data/HR_8799_c/20201006/reduced_jb/"
    # filelist = glob(os.path.join(dir,"*_035.fits"))
    # tel_dir = "/data/osiris_data/HR_8799_c/20201006/reduced_telluric_jb/HR_8799/"
    # tel_filelist = glob(os.path.join(tel_dir,"*_035.fits"))



    filelist.sort()
    tel_filelist.sort()
    # sc_filename,filename2 = filelist[1],filelist[0]#Kbb
    # sc_filename,filename2 = filelist[9],filelist[8]#Hbb [11,36]
    # sc_filename = filelist[9]#Hbb
    # sc_filename,filename2 = filelist[0],glob(os.path.join(dir,"*a020002*_020.fits"))[0] #20161106
    sc_filename,filename2 = filelist[1],filelist[2]#Kbb
    # print(sc_filename,filename2)
    # exit()
    # for sc_filename in filelist:
    if 1:
        mypool = mp.Pool(processes=numthreads)

        standard_spec_list = []
        for filename in tel_filelist:
            wvs, standard_cube, standard_noisecube, standard_badpixcube, standard_bary_rv = read_osiris(filename,skip_baryrv = True)
            standard_spec_list.append(aper_spec(standard_cube,aper=2,center=None))
        med_standard_spec = np.nanmedian(standard_spec_list,axis=0)

        standard_spec_list = []
        gw_list = []
        for filename in tel_filelist:
            wvs, standard_cube, standard_noisecube, standard_badpixcube, standard_bary_rv = read_osiris(filename)
            # plt.plot(standard_cube[:,6,14],label="standard_cube")
            standard_badpixcube,corr_standard_cube,standard_res = findbadpix(standard_cube, noisecube=standard_noisecube, badpixcube=standard_badpixcube,chunks=20,mypool=mypool,med_spec=med_standard_spec)
            standard_spec = aper_spec(corr_standard_cube,aper=5,center=None)
            standard_spec_list.append(standard_spec)
            # plt.plot(corr_standard_cube[:,6,14],label="corr_standard_cube")
            # print(standard_res[:,6,14])
            # plt.plot(standard_spec,label="corr_standard_cube")

            # plt.show()
            standard_im = np.nanmean(corr_standard_cube,axis=0)
            ycen,xcen = np.unravel_index(np.nanargmax(standard_im),standard_im.shape)
            w_guess = 1
            para0 = [1,xcen,ycen,w_guess,0]
            bounds = [(0,np.inf),(xcen-2,xcen+2),(ycen-2,ycen+2),(0.1*w_guess,w_guess*5),(-np.inf,np.inf)]
            A,xA,yA,gw,bkg = minimize(like_fit_pixgauss2d,para0,bounds=bounds,args=(standard_im,5),options={"maxiter":1e5}).x
            gw_list.append(gw)
            # print( A,xA,yA,gw,bkg )
            # plt.imshow(standard_im,origin="lower",interpolation="nearest")
            # plt.show()
            # plt.plot(standard_spec)
            # plt.show()
        # plt.show()
        gw = np.median(gw_list)
        print(gw_list)
        print(gw)
        # exit()
        mn_standard_spec = np.nanmean(np.array(standard_spec_list),axis=0)
        # plt.plot(mn_standard_spec)
        # plt.show()

        phoenix_folder = "/data/osiris_data/phoenix/"
        phoenix_A0_filename = glob(os.path.join(phoenix_folder, "kap_And_lte11600-4.00-0.5.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"))[0]
        phoenix_wv_filename = os.path.join(phoenix_folder, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
        # host_rv = -12.7 #+-0.8
        # host_limbdark = 0.5
        # host_vsini = 150 #unknown
        standard_rv = -3#+-4.4
        standard_limbdark = 0.5
        standard_vsini = 60

        with pyfits.open(phoenix_wv_filename) as hdulist:
            phoenix_wvs = hdulist[0].data / 1.e4
        crop_phoenix = np.where((phoenix_wvs > wvs[0] - (wvs[-1] - wvs[0]) / 4) * (phoenix_wvs < wvs[-1] + (wvs[-1] - wvs[0]) / 4))
        phoenix_wvs = phoenix_wvs[crop_phoenix]
        with pyfits.open(phoenix_A0_filename) as hdulist:
            phoenix_A0 = hdulist[0].data[crop_phoenix]
        phoenix_A0_func = interp1d(phoenix_wvs,phoenix_A0,bounds_error=False,fill_value=np.nan)
        wvs4broadening = np.arange(phoenix_wvs[0],phoenix_wvs[-1],1e-4)
        broadened_phoenix_A0 = pyasl.rotBroad(wvs4broadening, phoenix_A0_func(wvs4broadening), standard_limbdark, standard_vsini)
        broadened_phoenix_A0 = convolve_spectrum(wvs4broadening,broadened_phoenix_A0,R,mypool=mypool)
        phoenix_A0_func = interp1d(wvs4broadening/(1-(standard_rv+standard_bary_rv)/const.c.to('km/s').value),broadened_phoenix_A0,bounds_error=False,fill_value=np.nan)

        telluric_transmission =  mn_standard_spec/phoenix_A0_func(wvs)
        # plt.plot(wvs,telluric_transmission/np.nanmean(telluric_transmission),label="new")
        # tr_list = []
        # for test in glob("/data/osiris_data/kap_And/20161106/reduced_telluric_jb/HIP_111538/s161106_*_Kbb_020_psfs_repaired_spec_v2_transmission_v3.fits"):
        #     with pyfits.open(test) as hdulist:
        #         tr_list.append(hdulist[0].data[1,:]/np.nanmean(hdulist[0].data[1,:]))
        # wherefinite = np.where(np.isfinite(np.nanmedian(tr_list,axis=0)))
        # telluric_transmission[wherefinite] =  np.nanmedian(tr_list,axis=0)[wherefinite]
        # plt.plot(wvs,telluric_transmission/np.nanmean(telluric_transmission),label="old")
        # plt.plot(wvs,phoenix_A0_func(wvs)/np.nanmean(phoenix_A0_func(wvs)),label="model")
        # plt.legend()
        # plt.show()

        # plt.plot(telluric_transmission,label="1")
        # star_spectrum = telluric_transmission*phoenix_A0_func(wvs)
        # telluric_transmission = (star_spectrum/phoenix_A0_func(wvs))/LPFvsHPF(star_spectrum/phoenix_A0_func(wvs),40)[0]*LPFvsHPF(telluric_transmission,40)[0]
        # plt.plot(telluric_transmission,label="2")
        # plt.legend()
        # plt.show()
        # exit()
        print(sc_filename)
        # exit()
        wvs, cube, noisecube, badpixcube, science_bary_rv = read_osiris(sc_filename) # 36,12
        # plt.subplot(1,2,1)
        # plt.imshow(badpixcube[1000,:,:])
        badpixcube,corr_cube,res_cube = findbadpix(cube, noisecube=noisecube, badpixcube=badpixcube,chunks=20,mypool=mypool,med_spec=mn_standard_spec)
        # plt.subplot(1,2,2)
        # plt.imshow(badpixcube[1000,:,:])
        # plt.show()
        wvs2, cube2, noisecube2, badpixcube2, science_bary_rv2 = read_osiris(filename2) # 36,12
        badpixcube2,corr_cube2,res_cube2 = findbadpix(cube2, noisecube=noisecube2, badpixcube=badpixcube2,chunks=20,mypool=mypool,med_spec=mn_standard_spec)
        # plt.imshow(np.sum(np.isnan(badpixcube),axis=0)/1665)
        # plt.clim([0,0.2])
        # plt.show()
        # exit()
        # canvas = np.zeros(res_cube.shape)
        # canvas[np.where(res_cube==0)] = 1
        # plt.imshow(np.nansum(canvas,axis=0))
        # print(np.size(np.where(res_cube==0)[0]))
        # plt.show()
        # exit()
        # plt.imshow(np.nanmean(corr_cube,axis=0),origin="lower",interpolation="nearest")
        # plt.show()
        # pl_center = [12,36] #filelist[0]
        # pl_center = [11,32] #filelist[1]
        # pl_center = [11,39]
        # pl_center = [12,35]
        # pl_center = [11,36] # H band  filelist[9]
        pl_center = [10,37]
        # plt.plot(corr_cube[:,center[1]+1,center[0]],label="corr")
        # plt.plot(cube[:,center[1]+1,center[0]],label="ori") #34,12
        # plt.legend()
        # plt.show()

        nz,ny,nx=cube.shape
        chunks = 20
        x = np.arange(nz)
        x_knots = x[np.linspace(0,nz-1,chunks+1,endpoint=True).astype(np.int)]
        M_spline = get_spline_model(x_knots,x,spline_degree=3)
        from scipy.optimize import lsq_linear
        fit_cube = np.zeros(corr_cube.shape) + np.nan
        norm_cube_LPFvsHPF = np.zeros(corr_cube.shape) + np.nan
        fit_cube_noise = np.zeros(corr_cube.shape) + np.nan
        for k in range(ny):
            for l in range(nx):
                # if k != 30 or l != 10:
                #     continue

                print(k,l)
                where_data_finite = np.where(np.isfinite(badpixcube[:,k,l])*np.isfinite(corr_cube[:,k,l])*np.isfinite(noisecube[:,k,l])*(noisecube[:,k,l]!=0))


                d = mn_standard_spec[where_data_finite]
                d_err = noisecube[where_data_finite[0],k,l]

                M = M_spline[where_data_finite[0],:]*corr_cube[where_data_finite[0],k,l][:,None]
                bounds_min = [0, ]* M.shape[1]
                bounds_max = [np.inf, ] * M.shape[1]
                p = lsq_linear(M/d_err[:,None],d/d_err,bounds=(bounds_min, bounds_max)).x
                m = np.dot(M,p)

                fit_cube[where_data_finite[0],k,l] = m
                fit_cube_noise[where_data_finite[0],k,l] = d_err*np.sqrt(np.nanmean(((d-m)/d_err)**2))

                tmp = corr_cube[:,k,l]*badpixcube[:,k,l]
                norm_cube_LPFvsHPF[where_data_finite[0],k,l] = (tmp/LPFvsHPF(tmp,40)[0])[where_data_finite]

                # plt.plot(mn_standard_spec,label="ref")
                # plt.plot(fit_cube[:,k,l],label="fitcube")
                # plt.plot(fit_cube[:,k,l]+fit_cube_noise[:,k,l],label="noise")
                # plt.plot(tmp/np.nanmedian(tmp))
                # plt.plot(norm_cube_LPFvsHPF[:,k,l])
                # plt.legend()
                # plt.show()

        star_spectrum = np.nansum(fit_cube/fit_cube_noise**2,axis=(1,2))/np.nansum(1/fit_cube_noise**2,axis=(1,2))
        res4model = fit_cube-star_spectrum[:,None,None]


        star_spectrum_LPFvsHPF = np.nanmean(norm_cube_LPFvsHPF,axis=(1,2))
        # plt.plot(np.nanmedian(norm_cube_LPFvsHPF,axis=(1,2)), label="LPF")
        # plt.plot(star_spectrum/np.nanmedian(star_spectrum), label="spline")
        # plt.legend()
        # plt.show()

        if 1:
            import scipy.linalg as la
            res_numbasis = 10
            X = np.reshape(res4model,(res4model.shape[0],res4model.shape[1]*res4model.shape[2])).T
            X = X[np.where(np.nansum(X,axis=1)!=0)[0],:]
            X = X/np.nanstd(X,axis=1)[:,None]
            X[np.where(np.isnan(X))] = 0
            C = np.cov(X)
            tot_basis = C.shape[0]
            tmp_res_numbasis = np.clip(np.abs(res_numbasis) - 1, 0, tot_basis-1)  # clip values, for output consistency we'll keep duplicates
            max_basis = np.max(tmp_res_numbasis) + 1  # maximum number of eigenvectors/KL basis we actually need to use/calculate
            evals, evecs = la.eigh(C, eigvals=(tot_basis-max_basis, tot_basis-1))
            check_nans = np.any(evals <= 0) # alternatively, check_nans = evals[0] <= 0
            evals = np.copy(evals[::-1])
            evecs = np.copy(evecs[:,::-1], order='F') #fortran order to improve memory caching in matrix multiplication
            # calculate the KL basis vectors
            kl_basis = np.dot(X.T, evecs)
            res4model_kl = kl_basis * (1. / np.sqrt(evals * (res4model.shape[0] - 1)))[None, :]  #multiply a value for each row
            # print(res4model_kl.shape)
            # for k in range(res4model_kl.shape[1]):
            #     plt.plot(res4model_kl[:,k],label="{0}".format(k))
            # plt.show()
            # exit()


        # star_spectrum2 = np.nansum(fit_cube/fit_cube_noise**2,axis=(1,2))/np.nansum(1/fit_cube_noise**2,axis=(1,2))
        star_spectrum2 = mn_standard_spec#np.nanmean(fit_cube,axis=(1,2))
        # plt.plot(np.nanmedian(corr_cube,axis=(1,2))/np.nanmean(np.nanmedian(corr_cube,axis=(1,2))),label="corr_cube")
        # plt.plot(np.nanmedian(cube,axis=(1,2))/np.nanmean(np.nanmedian(cube,axis=(1,2))),label="cube")
        # plt.plot(star_spectrum/np.nanmean(star_spectrum),label="star_spectrum")
        # plt.plot(star_spectrum2/np.nanmean(star_spectrum2),label="test")
        # plt.legend()
        # plt.show()

        # tmp_cube = copy(corr_cube)
        # tmp_cube[:,pl_center[1]-3:pl_center[1]+4,pl_center[0]-3:pl_center[0]+4] = np.nan
        # # plt.subplot(1,2,1)
        # # plt.imshow(np.nanmean(corr_cube,axis=0),origin="lower",interpolation="nearest")
        # # plt.subplot(1,2,2)
        # # plt.imshow(np.nanmean(tmp_cube,axis=0),origin="lower",interpolation="nearest")
        # # plt.show()
        # # plt.plot(corr_cube[:,center[1]-2,center[0]-2])
        # # plt.show()
        # star_spectrum = np.nanmedian(tmp_cube,axis=(1,2))#corr_cube2[:,center[1]-5,center[0]-5]#
        # # corr_cube2 = corr_cube2*(LPFvsHPF(np.nanmedian(corr_cube,axis=(1,2)),40)[0]/LPFvsHPF(np.nanmedian(corr_cube2,axis=(1,2)),40)[0])[:,None,None]
        # # plt.subplot(2,1,1)
        # # plt.plot(np.nanmedian(corr_cube,axis=(1,2)),label="1")
        # # plt.plot(np.nanmedian(corr_cube2,axis=(1,2)),label="2")
        # # plt.legend()
        # # plt.subplot(2,1,2)
        # # plt.plot(np.nanmedian(corr_cube,axis=(1,2))/np.nanmedian(corr_cube2,axis=(1,2)),label="1")
        # # plt.show()

        osiris_data_dir = "/data/osiris_data"
        planet_template_folder = os.path.join(osiris_data_dir,"planets_templates")
        if 1:
            travis_spec_filename=os.path.join(planet_template_folder,
                                              "KapAnd_lte19-3.50-0.0.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019.7.save")
            travis_spectrum = scio.readsav(travis_spec_filename)
            ori_planet_spec = np.array(travis_spectrum["f"])
            wmod = np.array(travis_spectrum["w"])/1.e4
            crop_model = np.where((wmod > wvs[0] - (wvs[-1] - wvs[0]) / 4) * (wmod < wvs[-1] + (wvs[-1] - wvs[0]) / 4))
            wmod = wmod[crop_model]
            ori_planet_spec = ori_planet_spec[crop_model]
            broadened_planet_spec = ori_planet_spec#pyasl.rotBroad(wmod,ori_planet_spec, 0.5, 40)
        if 0:
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "HR8799c_"+"K"+"_3Oct2018.save")
            travis_spectrum = scio.readsav(travis_spec_filename)
            ori_planet_spec = np.array(travis_spectrum["fmod"])
            ori_planet_convspec = np.array(travis_spectrum["fmods"])
            wmod = np.array(travis_spectrum["wmod"])/1.e4

            crop_model = np.where((wmod > wvs[0] - (wvs[-1] - wvs[0]) / 4) * (wmod < wvs[-1] + (wvs[-1] - wvs[0]) / 4))
            wmod = wmod[crop_model]
            ori_planet_spec = ori_planet_spec[crop_model]
            broadened_planet_spec = ori_planet_spec
        # mypool = mp.Pool(processes=numthreads)
        broadened_planet_spec = convolve_spectrum(wmod,broadened_planet_spec,R,mypool=mypool)
        # mypool.close()
        # mypool.join()
        planet_spec_func = interp1d(wmod,broadened_planet_spec,bounds_error=False,fill_value=np.nan)

        # plt.plot(planet_spec_func(wvs))
        # plt.show()


        wvsol_offsets_filename = os.path.join(dir,"..","master_wvshifts_"+"Kbb"+".fits")
        hdulist = pyfits.open(wvsol_offsets_filename)
        wvsol_offsets = hdulist[0].data
        wvs = wvs - wvsol_offsets[37,10]
        hdulist.close()


        ##############################
        ## Create PSF model
        ##############################
        ref_star_folder = os.path.join(os.path.dirname(filelist[0]),"..","reduced_telluric_jb")
        with pyfits.open(glob(os.path.join(ref_star_folder,"*"+"Kbb"+"_hdpsfs_v2.fits"))[0]) as hdulist:
            psfs_refstar_arr = hdulist[0].data[None,:,:,:]
        with pyfits.open(glob(os.path.join(ref_star_folder,"*"+"Kbb"+"_hdpsfs_xy_v2.fits"))[0]) as hdulist:
            hdpsfs_xy = hdulist[0].data
            hdpsfs_x,hdpsfs_y = hdpsfs_xy

        nx_psf,ny_psf = 15,15
        nz_psf = psfs_refstar_arr.shape[1]
        x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
        x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)

        chunk_size=20
        N_chunks = nz_psf//chunk_size
        psfs_chunks = []
        for k in range(N_chunks-1):
            psfs_chunks.append(psfs_refstar_arr[:,k*chunk_size:(k+1)*chunk_size,:,:])
        psfs_chunks.append(psfs_refstar_arr[:,(N_chunks-1)*chunk_size:nz_psf,:,:])
        outputs_list = mypool.map(_spline_psf_model, zip(psfs_chunks,
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
        mypool.close()


        w = 2
        # center = pl_center
        center=None#[0,0]
        plxvec,plyvec = None,None#np.arange(0,19,1),np.arange(0,64,1)
        # plxvec,plyvec = np.array([0]),np.array([0])
        # plxvec,plyvec = np.array([-5]),np.array([-6])
        # plxvec,plyvec = np.arange(-1,2,1),np.arange(-1,2,1)
        # plxvec,plyvec = np.arange(-2,3,1),np.arange(-10,11,1)
        plrvvec = np.array([-16])
        # plrvvec = np.linspace(-1000,1000,21,endpoint=True)
        # M_func = (lambda _cube,_noisecube,_badpixcube,_plx,_ply,_plrv,_center: \
        #               make_linear_model(_cube,_noisecube,_badpixcube,_plx,_ply,_plrv,_center,
        #                                 wvs,telluric_transmission,star_spectrum,planet_spec_func,science_bary_rv,w=w,psfwidth0=0.3,chunks=chunks))
        # todo solve padding problem with corr_cube2
        # M_func = (lambda _cube,_noisecube,_badpixcube,_plx,_ply,_plrv,_center: \
        #               make_linear_model2(_cube,_noisecube,_badpixcube,_plx,_ply,_plrv,_center,
        #                                 wvs,telluric_transmission, np.pad(corr_cube2,((0,0),(w,w),(w,w)),mode="constant",constant_values=0),
        #                                  planet_spec_func,science_bary_rv,w=w,psfwidth0=0.3,chunks=chunks))
        # M = M_func(cube,noisecube,badpixcube,0,0,0)
        # print(M.shape)
        # plt.figure(1)
        # plt.imshow(np.nanmean(M[:,:,:,0],axis=0))
        # plt.figure(2)
        # plt.plot(np.nanmean(M[:,:,:,0],axis=(1,2)))
        # plt.show()
        # exit()
        # out,res = detecplanet(corr_cube,0, center=center,plxvec=plxvec,plyvec=plyvec,plrvvec=plrvvec,
        #                                    noisecube=noisecube, badpixcube=badpixcube,numthreads=32,
        #                   wvs=wvs,telluric_transmission=telluric_transmission,star_spectrum=star_spectrum,cube_model =corr_cube2,
        #                 planet_spec_func=planet_spec_func,science_bary_rv=science_bary_rv,psfwidth0=0.3,chunks=chunks)
        # # res[:,25:31,:] = np.nan
        # res[:,45:55,:] = np.nan
        # med_res = np.nanmedian(res,axis=(1,2))
        # # plt.plot(med_res)
        # # plt.show()
        # # med_res = np.nanmedian(res[:,10:55,10::],axis=(1,2))
        # center,plxvec,plyvec,plrvvec = [0,0],np.array([10]),np.array([37]),np.array([-16])
        # center,plxvec,plyvec,plrvvec = [0,0],np.array([9]),np.array([23]),np.array([-16])
        # center,plxvec,plyvec,plrvvec = [0,0],np.array([10]),np.array([37]),np.linspace(-50,50,101,endpoint=True)
        # center,plxvec,plyvec,plrvvec = [0,0],np.array([9,10,11]),np.array([36,37,38]),np.array([-15])
        # center,plxvec,plyvec,plrvvec = pl_center,np.array([0]),np.array([0]),np.array([-15])#np.linspace(-50,50,101,endpoint=True)
        # badpixcube[1640::,:,:] = np.nan
        # badpixcube[1400::,:,:] = np.nan
        badpixcube[0:50,:,:] = np.nan
        badpixcube[1600::,:,:] = np.nan
        # badpixcube[1::2,:,:] = np.nan
        out,res = detecplanet(corr_cube,w, center=center,plxvec=plxvec,plyvec=plyvec,plrvvec=plrvvec,
                                           noisecube=noisecube, badpixcube=badpixcube,numthreads=32,
                          wvs=wvs,telluric_transmission=telluric_transmission,star_spectrum=[star_spectrum,res4model_kl],cube_model =corr_cube2,#star_spectrum_LPFvsHPF
                        planet_spec_func=planet_spec_func,science_bary_rv=science_bary_rv,psfwidth0=gw,chunks=chunks,res_model = normalized_psfs_func_list)
        print(out.shape)
        print(res.shape)
        nz,ny,nx = res.shape

        # plt.figure(2)
        # # plt.subplot(1,2,1)
        # plt.imshow(np.nanstd(res,axis=0),interpolation="nearest",origin="lower")
        plt.figure(2)
        res[:,30:40,:] = np.nan
        res = np.reshape(res,(nz,nx*ny))
        # choose = np.random.randint(0, high=nx*ny, size=100)
        # for k in choose:
        #     plt.plot(res[:,k],alpha=0.5)
        plt.plot(np.nanmedian(res,axis=1))
        plt.plot(np.nanstd(res,axis=1))
        # for k in range(res4model_kl.shape[1]):
        plt.plot(res4model_kl[:,0]/np.nanstd(res4model_kl[:,0])*np.nanstd(np.nanstd(res,axis=1)),label="{0}".format(k))
        plt.ylim([-0.1,0.1])
        # plt.show()


        N_linpara = (out.shape[0]-2)//2
        # N_linpara = 1+(2*w+1)**2*(chunks+1)
        # print(out)
        if plxvec is not None:
            plxid0 = np.argmin(np.abs(plxvec))
        else:
            plxid0 = 11
        if plyvec is not None:
            plyid0 = np.argmin(np.abs(plyvec))
        else:
            plyid0 = 32
        rvid0 = np.argmin(np.abs(plrvvec))
        plt.figure(1)
        snr_map = out[2,rvid0,:,:]/out[2+N_linpara,rvid0,:,:]
        bayes_factor_ratio_map = out[0,rvid0,:,:]-out[1,rvid0,:,:]#np.exp(out[0,rvid0,:,:]-out[1,rvid0,:,:])
        # from scipy.signal import convolve2d
        # kernel = 0.7*np.ones((3,3))
        # kernel[1,1] = 1
        # snr_map = convolve2d(snr_map,kernel , mode='same', boundary='fill', fillvalue=0)
        plt.subplot(1,5,1)
        snr_scaling = np.nanstd(snr_map[3:30,5:15])
        # snr_scaling = np.nanstd(np.concatenate([snr_map[45:60,5:15],snr_map[0:20,5:15]],axis=0))
        print(snr_scaling)
        plt.imshow(snr_map/snr_scaling,origin="lower",interpolation="nearest")
        plt.clim([-5,20])
        plt.subplot(1,5,2)
        plt.imshow(snr_map,origin="lower",interpolation="nearest")
        plt.clim([-5,20])
        plt.subplot(1,5,3)
        plt.imshow(out[2,rvid0,:,:],origin="lower",interpolation="nearest")
        plt.subplot(1,5,4)
        plt.imshow(out[2+N_linpara,np.argmin(np.abs(plrvvec)),:,:],origin="lower",interpolation="nearest")
        plt.subplot(1,5,5)
        plt.imshow(bayes_factor_ratio_map,origin="lower",interpolation="nearest")
        plt.clim([0,50])
        plt.colorbar()

        # plt.figure(3)
        # plt.plot(plrvvec,out[2,:,plyid0,plxid0]/out[2+N_linpara,:,plyid0,plxid0])
        # # plt.plot(plrvvec,out[2+N_linpara,:,plyid0,plxid0])
        # # plt.fill_between(plrvvec,
        # #                  out[2,:,plyid0,plxid0]-out[2+N_linpara,:,plyid0,plxid0],
        # #                  out[2,:,plyid0,plxid0]+out[2+N_linpara,:,plyid0,plxid0],alpha=0.5)
        # # plt.ylim([0,30])
        # plt.figure(4)
        # plt.plot(plrvvec,np.exp(out[0,:,plyid0,plxid0]-np.nanmax(out[0,:,plyid0,plxid0])))

        # hdulist = pyfits.HDUList()
        # hdulist.append(pyfits.PrimaryHDU(data=snr_map/snr_scaling))
        # hdulist.writeto(os.path.join(out_pngs,os.path.basename(sc_filename).replace(".fits","_snr.fits")), clobber=True)
        # hdulist.close()
        # plt.savefig(os.path.join(out_pngs,os.path.basename(sc_filename).replace(".fits","_snr.png")),bbox_inches='tight')

        plt.show()
        exit()
