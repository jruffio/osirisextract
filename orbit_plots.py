__author__ = 'jruffio'
import warnings
import h5py
import astropy.units as u
import astropy.constants as consts
from astropy.io import fits
from astropy.time import Time
from astropy._erfa.core import ErfaWarning

import matplotlib as mpl
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

import orbitize.kepler as kepler
import orbitize.system


import numpy as np
import os
import sys
import multiprocessing as mp
from astropy.time import Time
import multiprocessing as mp
import astropy.io.fits as pyfits
from copy import copy
if len(sys.argv) == 1:
    import orbitize
    import orbitize.driver
    import orbitize
    from orbitize import driver

    osiris_data_dir = "/data/osiris_data"
    astrometry_DATADIR = os.path.join(osiris_data_dir,"astrometry")
    uservs = False
    # planet = "d"
    planet = "bc"
    # MCMC parameters
    num_temps = 20
    num_walkers = 100
    total_orbits = 10000 # number of steps x number of walkers (at lowest temperature)
    burn_steps = 100 # steps to burn in per walker
    thin = 2 # only save every 2nd step
    num_threads = mp.cpu_count() # or a different number if you prefer
else:
    import matplotlib
    matplotlib.use("Agg")
    import orbitize
    import orbitize.driver
    import orbitize
    from orbitize import driver

    osiris_data_dir = sys.argv[1]
    astrometry_DATADIR = os.path.join(osiris_data_dir,"astrometry")
    filename = sys.argv[2]
    planet = sys.argv[3]
    # MCMC parameters
    num_temps = int(sys.argv[4])
    num_walkers = int(sys.argv[5])
    total_orbits = int(sys.argv[6]) # number of steps x number of walkers (at lowest temperature)
    burn_steps = int(sys.argv[7]) # steps to burn in per walker
    thin = int(sys.argv[8]) # only save every 2nd step
    num_threads = int(sys.argv[9]) # or a different number if you prefer
    suffix = sys.argv[10]
#     sbatch --partition=hns,owners,iric --qos=normal --time=2-00:00:00 --mem=20G --output=/scratch/groups/bmacint/osiris_data/astrometry/logs/20190703_203155_orbit_fit_HR8799b.csv --error=/scratch/groups/bmacint/osiris_data/astrometry/logs/20190703_203155_orbit_fit_HR8799b.csv --nodes=1 --ntasks-per-node=10 --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="
#  nice -n 15 /home/anaconda3/bin/python3 /home/sda/jruffio/pyOSIRIS/osirisextract/orbit_fit.py /data/osiris_data/ /data/osiris_data/astrometry/HR8799b.csv b 20 100 20000 100 2 10 sherlock

sysrv=-12.6
sysrv_err=1.4

# print(rv_str,sysrv,sysrv_err)
# exit()

from orbitize import results
try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
fontsize = 12

# system parameters
num_secondary_bodies = len(planet)
system_mass = 1.47 # [Msol]
plx = None#25.38 # [mas]
mass_err = 0.3#0.3 # [Msol]
plx_err = None#0.7#0.7 # [mas]
# suffix = "test4_coplanar"
# suffix_norv = "test_joint_16_512_1000_2_False_coplanar"
# suffix_withrvs = "test_joint_16_512_1000_2_True_coplanar"
# suffix_norv = "gpicruncher_joint_16_512_10000_2_False_coplanar"
# suffix_withrvs = "gpicruncher_joint_16_512_10000_2_True_coplanar"
# suffix_norv = "sherlock_16_1024_200000_50_False_coplanar"
# suffix_withrvs = "sherlock_16_1024_200000_50_True_coplanar"
# suffix_norv = "sherlock_2ndrun_16_1024_250000_50_False_coplanar"
# suffix_norv = "sherlock_2ndrun_2_1024_1250000_100_False_coplanar"
# suffix = "sherlock"
# suffix = "sherlock_ptemceefix_12_100_300000_50"
suffix_withrvs = "sherlock_restrictOme_16_1024_200000_50_True_coplanar"
suffix_norv = "sherlock_restrictOme_16_1024_200000_50_False_coplanar"
# suffix_withrvs = "sherlock_restrictOme_notcoplanar_16_1024_200000_50_True"
# suffix_norv = "sherlock_restrictOme_notcoplanar_16_1024_200000_50_False"



from scipy.interpolate import interp1d
def get_err_from_posterior(x,posterior):
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    # import matplotlib.pyplot as plt
    # plt.plot(x,cum_posterior)
    # plt.plot(x,posterior/np.nanmax(posterior))
    # plt.show()
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
    # return x[argmax_post],(rx-lx)/2.,argmax_post


def get_upperlim_from_posterior(x,posterior):
    cum_posterior = np.cumsum(posterior)
    cum_posterior = cum_posterior/np.max(cum_posterior)
    # import matplotlib.pyplot as plt
    # plt.plot(x,cum_posterior)
    # plt.plot(x,posterior/np.nanmax(posterior))
    # plt.show()
    lf = interp1d(cum_posterior,x,bounds_error=False,fill_value=np.nan)
    upplim = lf(0.6827)
    return upplim

if 1:
    # mu = 5
    # x = np.linspace(-7+mu,7+mu,1400)
    # post = np.exp(-0.5*(x-mu)**2/1.5**2)
    # print(get_err_from_posterior(x,post))
    # exit()

    import matplotlib.pyplot as plt
    filename = "{0}/HR8799{1}_rvs.csv".format(astrometry_DATADIR,planet)
    data_table_withrvs = orbitize.read_input.read_file(filename)
    filename = "{0}/HR8799{1}.csv".format(astrometry_DATADIR,planet)
    data_table_norv = orbitize.read_input.read_file(filename)
    if 0:
        hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format("withrvs",planet,suffix_withrvs))
        print(hdf5_filename)
        # print("/data/osiris_data/astrometry/figures/HR_8799_bc/posterior_withrvs_bc_test_joint_16_512_1000_2_True_coplanar.hdf5")
        # exit()
        loaded_results_withrvs = results.Results() # Create blank results object for loading
        loaded_results_withrvs.load_results(hdf5_filename)
        param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","plx","sysrv","mtot"]
        corner_plot_fig = loaded_results_withrvs.plot_corner(param_list=param_list)
        corner_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"corner_plot_withrvs_{0}_{1}.png".format(planet,suffix_withrvs)))
        plt.show()


    suffix_norv2 = "test_joint_16_512_1000_2_False_coplanar"
    suffix_withrvs2 = "test_joint_16_512_1000_2_True_coplanar"
    # suffix_norv2 = "test_fixomegabug_notcoplanar_16_100_100_2_False"
    # suffix_withrvs2 = "test_fixomegabug_notcoplanar_16_100_100_2_True"
    hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format("norv",planet,suffix_norv2))
    print(hdf5_filename)
    loaded_results_norv = results.Results() # Create blank results object for loading
    loaded_results_norv.load_results(hdf5_filename)
    hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format("withrvs",planet,suffix_withrvs2))
    print(hdf5_filename)
    loaded_results_withrvs = results.Results() # Create blank results object for loading
    loaded_results_withrvs.load_results(hdf5_filename)


    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("norv",planet,suffix_norv))) as hdulist:
        myshape = hdulist[0].data.shape
        print(myshape)
        if myshape[3] == 14:
            chains_norv = hdulist[0].data
        else:
            chains_norv = np.zeros((myshape[0],myshape[1],myshape[2],14))
            chains_norv[:,:,:,0:(2+6)] = hdulist[0].data[:,:,:,0:(2+6)]
            chains_norv[:,:,:,3+6] = hdulist[0].data[:,:,:,2+6]
            chains_norv[:,:,:,(5+6)::] = hdulist[0].data[:,:,:,(3+6)::]
        if 0:
            choose = np.random.randint(0, high=chains_norv.shape[2], size=chains_norv.shape[2]//2)
            chains_norv[:,:,choose,4] -= np.pi
            chains_norv[:,:,choose,4] = np.mod(chains_norv[:,:,choose,4],2*np.pi)
            chains_norv[:,:,choose,3] -= np.pi
            chains_norv[:,:,choose,3+6] -= np.pi
            chains_norv[:,:,choose,3] = np.mod(chains_norv[:,:,choose,3],2*np.pi)
            chains_norv[:,:,choose,3+6] = np.mod(chains_norv[:,:,choose,3+6],2*np.pi)
        if not (myshape[3] == 14):
            chains_norv[:,:,:,2+6] = chains_norv[:,:,:,2]
            chains_norv[:,:,:,4+6] = chains_norv[:,:,:,4]
        print(chains_norv.shape)
        chains_norv = chains_norv[0,:,2*chains_norv.shape[2]//4::,:]
        # chains_norv = chains_norv[0,:,:,:]
        # chains_norv = chains_norv[0,:,0:10,:]
        # chains_norv = chains_norv[0,:,0:1,:]
    print(chains_norv.shape)
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs))) as hdulist:
        myshape = hdulist[0].data.shape
        print(myshape)
        if myshape[3] == 15:
            chains_withrvs = hdulist[0].data
        else:
            chains_withrvs = np.zeros((myshape[0],myshape[1],myshape[2],myshape[3]+2))
            chains_withrvs[:,:,:,0:(2+6)] = hdulist[0].data[:,:,:,0:(2+6)]
            chains_withrvs[:,:,:,3+6] = hdulist[0].data[:,:,:,2+6]
            chains_withrvs[:,:,:,(5+6)::] = hdulist[0].data[:,:,:,(3+6)::]
        if not (myshape[3] == 15):
            chains_withrvs[:,:,:,2+6] = chains_withrvs[:,:,:,2]
            chains_withrvs[:,:,:,4+6] = chains_withrvs[:,:,:,4]
        print(chains_withrvs.shape)
        chains_withrvs = chains_withrvs[0,:,2*chains_withrvs.shape[2]//4::,:]
        # chains_withrvs = chains_withrvs[0,:,:,:]
        # chains_withrvs = chains_withrvs[0,:,0:10,:]
        # chains_withrvs = chains_withrvs[0,:,0:1,:]
    print(chains_withrvs.shape)
    # exit()

    sysrv_chain = np.ravel(chains_withrvs[:,:,-2])
    ome_chain = np.ravel(chains_withrvs[:,:,4])
    inc_chain = np.ravel(chains_withrvs[:,:,2])

    if 0:
        #mutual inclination with disk
        i_herschel = 26
        ierr_herschel =3
        Omega_herschel =62
        Omegaerr_herschel =3
        # SMA+ALMA
        # i=32.8+5.6-9.6
        # Omega = 35.6+9.4-10.1
        i_alma = 32.8
        ierr_alma =(5.6+9.6)/2
        Omega_alma =35.6
        Omegaerr_alma =(9.4+10.1)/2
        # i_alma = 20.8
        # ierr_alma =0.01
        # Omega_alma =89
        # Omegaerr_alma =0.01

        plt.figure(10)

        mean = [i_herschel,Omega_herschel]
        cov=np.diag([ierr_herschel**2,Omegaerr_herschel**2])
        iOme_samples = np.deg2rad(np.random.multivariate_normal(mean,cov,size=np.size(inc_chain)))
        idisk,Omedisk = iOme_samples[:,0],iOme_samples[:,1]
        im_samples = np.rad2deg(np.arccos(np.cos(inc_chain)*np.cos(idisk) + np.sin(inc_chain)*np.sin(idisk)*np.cos(ome_chain-Omedisk)))
        im_post,xedges = np.histogram(im_samples,bins=100,range=[np.min(im_samples),np.max(im_samples)])
        x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
        im_mod, _,_,im_merr, im_perr,_ = get_err_from_posterior(x_centers,im_post)
        print("herschel",im_mod,im_merr, im_perr)
        plt.plot(x_centers,im_post/np.max(im_post),label="herschel")

        mean = [i_alma,Omega_alma]
        cov=np.diag([ierr_alma**2,Omegaerr_alma**2])
        iOme_samples = np.deg2rad(np.random.multivariate_normal(mean,cov,size=np.size(inc_chain)))
        idisk,Omedisk = iOme_samples[:,0],iOme_samples[:,1]
        im_samples = np.rad2deg(np.arccos(np.cos(inc_chain)*np.cos(idisk) + np.sin(inc_chain)*np.sin(idisk)*np.cos(ome_chain-Omedisk)))
        im_post,xedges = np.histogram(im_samples,bins=100,range=[np.min(im_samples),np.max(im_samples)])
        x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
        im_mod, _,_,im_merr, im_perr,_ = get_err_from_posterior(x_centers,im_post)
        print("SMA_ALMA",im_mod,im_merr, im_perr)
        plt.plot(x_centers,im_post/np.max(im_post),label="SMA_ALMA")

        plt.legend()
        plt.xlabel("Mutual Inclination")
        plt.ylabel("PDF")
        plt.show()



    inc_post,xedges = np.histogram(np.rad2deg(inc_chain),bins=2*60,range=[0,60])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    inc_mod, _,_,inc_merr, inc_perr,_ = get_err_from_posterior(x_centers,inc_post)

    ome_post,xedges = np.histogram(np.rad2deg(ome_chain),bins=2*180,range=[0,180])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    ome_mod, _,_,ome_merr, ome_perr,_ = get_err_from_posterior(x_centers,ome_post)

    sysrv_post,xedges = np.histogram(sysrv_chain,bins=10*20,range=[-20,0])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    sysrv_mod, _,_,sysrv_merr, sysrv_perr,_ = get_err_from_posterior(x_centers,sysrv_post)
    print("inclination: {0},{1},{2}".format(inc_mod,inc_merr,inc_perr))
    print("Omega: {0},{1},{2}".format(ome_mod,ome_merr,ome_perr))
    print("sysrv: {0},{1},{2}".format(sysrv_mod,sysrv_merr,sysrv_perr))
    # exit()

    post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,0]),bins=200,range=[np.min(chains_withrvs[:,:,0]),np.max(chains_withrvs[:,:,0])])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    print("a_b : {0},{1},{2}".format(mod,merr,perr))
    post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,1]),bins=2*60,range=[np.min(chains_withrvs[:,:,1]),np.max(chains_withrvs[:,:,1])])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    upplim = get_upperlim_from_posterior(x_centers,post)
    print("e_b : uplim={0}".format(upplim))
    tmp = np.rad2deg(np.ravel(chains_withrvs[:,:,3]))
    post,xedges = np.histogram(tmp,bins=2*60,range=[np.min(tmp),np.max(tmp)])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    print("omega_b : {0},{1},{2}".format(mod,merr,perr))
    post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,5]),bins=2*60,range=[np.min(chains_withrvs[:,:,5]),np.max(chains_withrvs[:,:,5])])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    print("tau_b : {0},{1},{2}".format(mod,merr,perr))

    post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,0+6]),bins=2*60,range=[np.min(chains_withrvs[:,:,0+6]),np.max(chains_withrvs[:,:,0+6])])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    print("a_c : {0},{1},{2}".format(mod,merr,perr))
    post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,1+6]),bins=2*60,range=[np.min(chains_withrvs[:,:,1+6]),np.max(chains_withrvs[:,:,1+6])])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    upplim = get_upperlim_from_posterior(x_centers,post)
    print("e_c : uplim={0}".format(upplim))
    tmp = np.rad2deg(np.ravel(chains_withrvs[:,:,3+6]))
    post,xedges = np.histogram(tmp,bins=2*60,range=[np.min(tmp),np.max(tmp)])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    print("omega_c : {0},{1},{2}".format(mod,merr,perr))
    post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,5+6]),bins=2*60,range=[np.min(chains_withrvs[:,:,5+6]),np.max(chains_withrvs[:,:,5+6])])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    print("tau_c : {0},{1},{2}".format(mod,merr,perr))

    post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,-3]),bins=2*60,range=[np.min(chains_withrvs[:,:,-3]),np.max(chains_withrvs[:,:,-3])])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    print("plx : {0},{1},{2}".format(mod,merr,perr))
    post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,-1]),bins=2*60,range=[np.min(chains_withrvs[:,:,-1]),np.max(chains_withrvs[:,:,-1])])
    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    print("mtot : {0},{1},{2}".format(mod,merr,perr))
    # exit()

    # plt.figure(1)
    # plt.subplot(2,1,1)
    # plt.plot(chains_norv[::100,::1,4].T,color="grey",alpha =0.1 )
    # plt.subplot(2,1,2)
    # plt.plot(chains_withrvs[::100,::1,4].T,color="grey",alpha =0.1 )
    # plt.show()
    # #
    # all_mycorr = []
    # plt.figure(2)
    # for k in np.arange(0,chains_norv.shape[0],50):
    #     print(k)
    #     chain = chains_norv[k,:,4] - np.mean(chains_norv[k,:,4])
    #     mycorr = np.correlate(chain,chain,mode="full")
    #     mycorr = mycorr/np.max(mycorr)
    #     all_mycorr.append(mycorr)
    #     plt.plot(mycorr,alpha=0.1)
    #     # l = chains_norv.shape[1]
    #     # # print(l)
    #     # print(mycorr[l-1:l+1],np.sum(mycorr[l-1:l+1]))
    #     # print(np.sum(mycorr[l-l//2:l+l//2]))
    #
    # plt.plot(np.mean(all_mycorr,axis=0),alpha=1,linewidth = 3)
    # # print(np.sum(all_mycorr)/len(all_mycorr))
    # plt.show()

    chains_norv = np.reshape(chains_norv,(chains_norv.shape[0]*chains_norv.shape[1],chains_norv.shape[2]))
    chains_withrvs = np.reshape(chains_withrvs,(chains_withrvs.shape[0]*chains_withrvs.shape[1],chains_withrvs.shape[2]))

    loaded_results_norv.post = chains_norv
    loaded_results_withrvs.post = chains_withrvs

    post_norv = loaded_results_norv.post
    post_withrvs = loaded_results_withrvs.post

    if 1:
        num_orbits_to_plot = 10000
        num_secondary_bodies = len(planet)
        system_mass = 1.52#1.47 # [Msol] (Jason)
        plx = 24.2175#25.38 # [mas]
        mass_err = 0.15#0.3 # [Msol] (Jason)
        plx_err = 0.0881#0.7 # [mas]
        sysrv=-12.6
        sysrv_err=1.4
        from orbitize import priors, sampler
        out_pngs = os.path.join(astrometry_DATADIR,"figures")
        filename = "{0}/HR8799bc_rvs_4RVcalc.csv".format(astrometry_DATADIR)
        # userv = "includingrvdata"
        userv = "norvdata"
        if userv == "includingrvdata":
            restrict_angle_ranges = True
            choose = np.random.randint(0, high=chains_withrvs.shape[0], size=num_orbits_to_plot)
            chains = chains_withrvs[choose,:].T
        else:
            restrict_angle_ranges = False
            choose = np.random.randint(0, high=chains_norv.shape[0], size=num_orbits_to_plot)
            chains = chains_norv[choose,:].T
        my_driver = driver.Driver(
            filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
            sysrv=sysrv,sysrv_err=sysrv_err,
            mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers, 'num_threads': num_threads},system_kwargs = {"restrict_angle_ranges":restrict_angle_ranges},
        )
        if "bc" in planet:
            my_driver.system.coplanar = True
        if my_driver.system.coplanar and len(planet) >=2:
            my_driver.system.sys_priors[2+6] = -2
            my_driver.system.sys_priors[4+6] = -3
        if len(planet) == 1:
            my_driver.system.sys_priors[0] = priors.JeffreysPrior(1, 1e2)

        print(chains.shape)

        print(my_driver.system.data_table)
        # my_driver.system.data_table = my_driver.system.data_table[0:1]

        out_model = my_driver.system.compute_model(chains)
        print(out_model)
        print(out_model.shape)
        inclination = np.rad2deg(chains[2,:])
        Omega = np.rad2deg(chains[4,:])
        rvs_b =  out_model[0,0,:]
        rvs_c =  out_model[1,0,:]
        diffrvs =  rvs_b-rvs_c

        plt.scatter(inclination,diffrvs)
        # inclvsdiffrvs,xedges,yedges = np.histogram2d(inclination,diffrvs,bins=[50,100],range=[[0,np.pi],[-5,5]])
        inclvsdiffrvs,xedges,yedges = np.histogram2d(inclination,diffrvs,bins=[50,100],range=[[0,180],[-5,5]])
        inclvsdiffrvs = inclvsdiffrvs.T
        x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
        y_centers = [(y1+y2)/2. for y1,y2 in zip(yedges[0:len(yedges)-1],yedges[1:len(yedges)])]
        levels = [0.6827,0.9545]
        ravel_H = np.ravel(inclvsdiffrvs)
        ind = np.argsort(ravel_H)
        cum_ravel_H = np.zeros(np.shape(ravel_H))
        cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
        cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inclvsdiffrvs))
        image = copy(inclvsdiffrvs)
        image[np.where(cum_H>0.9545)] = np.nan
        xx,yy = np.meshgrid(x_centers,y_centers)
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles=["-","--"],linewidths=[2],colors=("black",),zorder=15,label="With RVs")
        # plt.xlabel("Inclination")
        plt.xlabel("Omega")
        plt.ylabel("rvb-rvc")

        where0 = np.where(np.abs(diffrvs)<0.1)
        plt.figure(3)
        plt.scatter(Omega[where0],inclination[where0])
        plt.xlabel("Omega")
        plt.ylabel("Inclination")

        plt.show()

        # hdulist = pyfits.HDUList()
        # hdulist.append(pyfits.PrimaryHDU(data=rvs_b))
        # try:
        #     hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'rvs_b_55392_{0}.fits'.format(userv)), overwrite=True)
        # except TypeError:
        #     hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'rvs_b_55392_{0}.fits'.format(suffix)), clobber=True)
        # hdulist.close()
        # hdulist = pyfits.HDUList()
        # hdulist.append(pyfits.PrimaryHDU(data=rvs_c))
        # try:
        #     hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'rvs_c_55392_{0}.fits'.format(userv)), overwrite=True)
        # except TypeError:
        #     hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'rvs_c_55392_{0}.fits'.format(suffix)), clobber=True)
        # hdulist.close()
        # hdulist = pyfits.HDUList()
        # hdulist.append(pyfits.PrimaryHDU(data=diffrvs))
        # try:
        #     hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'rvs_diffbc_55392_{0}.fits'.format(userv)), overwrite=True)
        # except TypeError:
        #     hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'rvs_diffbc_55392_{0}.fits'.format(suffix)), clobber=True)
        # hdulist.close()

        diffrvs_post,xedges = np.histogram(diffrvs,bins=20*10,range=[-10,10])
        x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
        # drv_mod, _,_,drv_merr, drv_perr,_ = get_err_from_posterior(x_centers,diffrvs_post)

        plt.plot(x_centers,diffrvs_post)
        plt.show()
        exit()



    if 0:
        param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","plx","mtot"]
        corner_plot_fig = loaded_results_norv.plot_corner(param_list=param_list)
        corner_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"corner_plot_norv_{0}_{1}.png".format(planet,suffix_norv)))
        param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","plx","sysrv","mtot"]
        corner_plot_fig = loaded_results_withrvs.plot_corner(param_list=param_list)
        corner_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"corner_plot_withrvs_{0}_{1}.png".format(planet,suffix_withrvs)))
        # exit()
        fig = loaded_results_norv.plot_orbits(
            object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
            num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
            start_mjd=data_table_norv['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
            data_table=data_table_norv,
            cbar_param="sma1",
            total_mass=system_mass,
            parallax=plx,
            system_rv=sysrv
        )
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_plot_{0}_{1}_{2}_obj1.png'.format("norv",planet,suffix_norv))) # This is matplotlib.figure.Figure.savefig()
        fig = loaded_results_norv.plot_orbits(
            object_to_plot = 2, # Plot orbits for the first (and only, in this case) companion
            num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
            start_mjd=data_table_norv['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
            data_table=data_table_norv,
            cbar_param="sma2",
            total_mass=system_mass,
            parallax=plx,
            system_rv=sysrv
        )
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_plot_{0}_{1}_{2}_obj2.png'.format("norv",planet,suffix_norv))) # This is matplotlib.figure.Figure.savefig()

        fig = loaded_results_norv.plot_rvs(
            object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
            num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
            start_mjd=data_table_norv['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
            data_table=data_table_norv,
            total_mass=system_mass,
            parallax=plx,
            system_rv=sysrv
        )
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'rv_plot_{0}_{1}_{2}_obj1.png'.format("norv",planet,suffix_norv))) # This is matplotlib.figure.Figure.savefig()
        fig = loaded_results_norv.plot_rvs(
            object_to_plot = 2, # Plot orbits for the first (and only, in this case) companion
            num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
            start_mjd=data_table_norv['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
            data_table=data_table_norv,
            system_rv=sysrv
        )
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'rv_plot_{0}_{1}_{2}_obj2.png'.format("norv",planet,suffix_norv))) # This is matplotlib.figure.Figure.savefig()


        fig = loaded_results_withrvs.plot_orbits(
            object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
            num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
            start_mjd=data_table_withrvs['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
            data_table=data_table_withrvs,
            cbar_param="sma1",
            total_mass=system_mass,
            parallax=plx,
            system_rv=sysrv
        )
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_plot_{0}_{1}_{2}_obj1.png'.format("withrvs",planet,suffix_withrvs))) # This is matplotlib.figure.Figure.savefig()
        fig = loaded_results_withrvs.plot_orbits(
            object_to_plot = 2, # Plot orbits for the first (and only, in this case) companion
            num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
            start_mjd=data_table_withrvs['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
            data_table=data_table_withrvs,
            cbar_param="sma2",
            total_mass=system_mass,
            parallax=plx,
            system_rv=sysrv
        )
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_plot_{0}_{1}_{2}_obj2.png'.format("withrvs",planet,suffix_withrvs))) # This is matplotlib.figure.Figure.savefig()

        fig = loaded_results_withrvs.plot_rvs(
            object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
            num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
            start_mjd=data_table_withrvs['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
            data_table=data_table_withrvs,
            total_mass=system_mass,
            parallax=plx,
            system_rv=sysrv
        )
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'rv_plot_{0}_{1}_{2}_obj1.png'.format("withrvs",planet,suffix_withrvs))) # This is matplotlib.figure.Figure.savefig()
        fig = loaded_results_withrvs.plot_rvs(
            object_to_plot = 2, # Plot orbits for the first (and only, in this case) companion
            num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
            start_mjd=data_table_withrvs['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
            data_table=data_table_withrvs,
            system_rv=sysrv
        )
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'rv_plot_{0}_{1}_{2}_obj2.png'.format("withrvs",planet,suffix_withrvs))) # This is matplotlib.figure.Figure.savefig()
        # plt.show()
        exit()

    if 0:
        fig = plt.figure(1,figsize=(6,4))
        Ome_bounds = [0,360]
        inc_bounds = [0,45]

        cmaps = ["cool","hot"]
        colors=["#006699","#ff9900","#6600ff","grey"]
        inc_Ome_norv_b,xedges,yedges = np.histogram2d(np.rad2deg(np.concatenate([post_norv[:,4],post_norv[:,4]+np.pi])),np.rad2deg(np.concatenate([post_norv[:,2],post_norv[:,2]])),bins=[360/4,45//2],range=[Ome_bounds,inc_bounds])
        inc_Ome_withrvs_b,xedges,yedges = np.histogram2d(np.rad2deg(post_withrvs[:,4]),np.rad2deg(post_withrvs[:,2]),bins=[360/4,45//2],range=[Ome_bounds,inc_bounds])
        # inc_Ome_norv_c,xedges,yedges = np.histogram2d(np.rad2deg(post_norv[:,4+6]),np.rad2deg(post_norv[:,2+6]),bins=[40,20],range=[Ome_bounds,inc_bounds])
        # inc_Ome_withrvs_c,xedges,yedges = np.histogram2d(np.rad2deg(post_withrvs[:,4+6]),np.rad2deg(post_withrvs[:,2+6]),bins=[40,20],range=[Ome_bounds,inc_bounds])
        inc_Ome_norv_b = inc_Ome_norv_b.T
        inc_Ome_withrvs_b = inc_Ome_withrvs_b.T
        # inc_Ome_norv_c = inc_Ome_norv_c.T
        # inc_Ome_withrvs_c = inc_Ome_withrvs_c.T
        x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
        y_centers = [(y1+y2)/2. for y1,y2 in zip(yedges[0:len(yedges)-1],yedges[1:len(yedges)])]


        ravel_H = np.ravel(inc_Ome_norv_b)
        ind = np.argsort(ravel_H)
        cum_ravel_H = np.zeros(np.shape(ravel_H))
        cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
        cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inc_Ome_norv_b))
        image = copy(inc_Ome_norv_b)
        image[np.where(cum_H>0.9545)] = np.nan

        # plt.imshow(image.T,origin ="lower",
        #            extent=[Ome_bounds[0],Ome_bounds[1],inc_bounds[0],inc_bounds[1]],
        #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
        plt.xlim(Ome_bounds)
        plt.ylim(inc_bounds)
        levels = [0.6827]
        xx,yy = np.meshgrid(x_centers,y_centers)
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles="--",linewidths=[2],colors=(colors[2],),zorder=15)
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)


        ravel_H = np.ravel(inc_Ome_withrvs_b)
        print(ravel_H.shape)
        ind = np.argsort(ravel_H)
        cum_ravel_H = np.zeros(np.shape(ravel_H))
        cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
        cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inc_Ome_withrvs_b))
        cum_H.shape = inc_Ome_withrvs_b.shape
        image = copy(inc_Ome_withrvs_b)
        image[np.where(cum_H>0.9545)] = np.nan

        # plt.imshow(image,origin ="lower",
        #            extent=[Ome_bounds[0],Ome_bounds[1],inc_bounds[0],inc_bounds[1]],
        #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
        plt.xlim(Ome_bounds)
        plt.ylim(inc_bounds)
        levels = [0.6827]
        xx,yy = np.meshgrid(x_centers,y_centers)
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles="-",linewidths=[2],colors=(colors[2],),zorder=15,label="With RVs")
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)

        plt.xlabel(r"Longitude of Ascending Node (deg)",fontsize=fontsize)
        plt.ylabel(r"Inclination (deg)",fontsize=fontsize)
        plt.plot([-1,-2],[-1,-2],linestyle="--",linewidth=2,color=colors[2],label="No RV")
        plt.plot([-1,-2],[-1,-2],linestyle="-",linewidth=2,color=colors[2],label="With RVs")
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'ome_vs_inc.pdf'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'ome_vs_inc.png'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        plt.show()


    if 0:
        fig = plt.figure(2,figsize=(6,4))
        sma_bounds = [30,80]
        ecc_bounds = [0,0.4]

        cmaps = ["cool","hot"]
        colors=["#006699","#ff9900","#6600ff","grey"]
        inc_Ome_norv_b,xedges,yedges = np.histogram2d(post_norv[:,0],post_norv[:,1],bins=[50*2,4*10],range=[sma_bounds,ecc_bounds])
        inc_Ome_withrvs_b,xedges,yedges = np.histogram2d(post_withrvs[:,0],post_withrvs[:,1],bins=[50*2,4*10],range=[sma_bounds,ecc_bounds])
        inc_Ome_norv_c,xedges,yedges = np.histogram2d(post_norv[:,0+6],post_norv[:,1+6],bins=[50*2,4*10],range=[sma_bounds,ecc_bounds])
        inc_Ome_withrvs_c,xedges,yedges = np.histogram2d(post_withrvs[:,0+6],post_withrvs[:,1+6],bins=[50*2,4*10],range=[sma_bounds,ecc_bounds])
        inc_Ome_norv_b = inc_Ome_norv_b.T
        inc_Ome_withrvs_b = inc_Ome_withrvs_b.T
        inc_Ome_norv_c = inc_Ome_norv_c.T
        inc_Ome_withrvs_c = inc_Ome_withrvs_c.T
        x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
        y_centers = [(y1+y2)/2. for y1,y2 in zip(yedges[0:len(yedges)-1],yedges[1:len(yedges)])]


        ravel_H = np.ravel(inc_Ome_norv_b)
        ind = np.argsort(ravel_H)
        cum_ravel_H = np.zeros(np.shape(ravel_H))
        cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
        cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inc_Ome_norv_b))
        image = copy(inc_Ome_norv_b)
        image[np.where(cum_H>0.9545)] = np.nan

        # plt.imshow(image.T,origin ="lower",
        #            extent=[sma_bounds[0],sma_bounds[1],ecc_bounds[0],ecc_bounds[1]],
        #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
        plt.xlim(sma_bounds)
        plt.ylim(ecc_bounds)
        levels = [0.6827]
        xx,yy = np.meshgrid(x_centers,y_centers)
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles="--",linewidths=[2],colors=(colors[0],),zorder=15)
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)


        ravel_H = np.ravel(inc_Ome_withrvs_b)
        print(ravel_H.shape)
        ind = np.argsort(ravel_H)
        cum_ravel_H = np.zeros(np.shape(ravel_H))
        cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
        cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inc_Ome_withrvs_b))
        cum_H.shape = inc_Ome_withrvs_b.shape
        image = copy(inc_Ome_withrvs_b)
        image[np.where(cum_H>0.9545)] = np.nan

        # plt.imshow(image,origin ="lower",
        #            extent=[sma_bounds[0],sma_bounds[1],ecc_bounds[0],ecc_bounds[1]],
        #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
        plt.xlim(sma_bounds)
        plt.ylim(ecc_bounds)
        levels = [0.6827]
        xx,yy = np.meshgrid(x_centers,y_centers)
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15,label="With RVs")
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)


        ravel_H = np.ravel(inc_Ome_norv_c)
        ind = np.argsort(ravel_H)
        cum_ravel_H = np.zeros(np.shape(ravel_H))
        cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
        cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inc_Ome_norv_c))
        image = copy(inc_Ome_norv_c)
        image[np.where(cum_H>0.9545)] = np.nan

        # plt.imshow(image.T,origin ="lower",
        #            extent=[sma_bounds[0],sma_bounds[1],ecc_bounds[0],ecc_bounds[1]],
        #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
        plt.xlim(sma_bounds)
        plt.ylim(ecc_bounds)
        levels = [0.6827]
        xx,yy = np.meshgrid(x_centers,y_centers)
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles="--",linewidths=[2],colors=(colors[1],),zorder=15)
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)


        ravel_H = np.ravel(inc_Ome_withrvs_c)
        print(ravel_H.shape)
        ind = np.argsort(ravel_H)
        cum_ravel_H = np.zeros(np.shape(ravel_H))
        cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
        cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inc_Ome_withrvs_c))
        cum_H.shape = inc_Ome_withrvs_c.shape
        image = copy(inc_Ome_withrvs_c)
        image[np.where(cum_H>0.9545)] = np.nan

        # plt.imshow(image,origin ="lower",
        #            extent=[sma_bounds[0],sma_bounds[1],ecc_bounds[0],ecc_bounds[1]],
        #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
        plt.xlim(sma_bounds)
        plt.ylim(ecc_bounds)
        levels = [0.6827]
        xx,yy = np.meshgrid(x_centers,y_centers)
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles="-",linewidths=[2],colors=(colors[1],),zorder=15,label="With RVs")
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)

        plt.xlabel(r"Semi-Major Axis (au)",fontsize=fontsize)
        plt.ylabel(r"Eccentricity",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.plot([-1,-2],[-1,-2],linestyle="--",linewidth=2,color=colors[0],label="b: No RV")
        plt.plot([-1,-2],[-1,-2],linestyle="-",linewidth=2,color=colors[0],label="b: With RVs")
        plt.plot([-1,-2],[-1,-2],linestyle="--",linewidth=2,color=colors[1],label="c: No RV")
        plt.plot([-1,-2],[-1,-2],linestyle="-",linewidth=2,color=colors[1],label="c: With RVs")
        plt.legend(loc="upper left",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'sma_vs_ecc.pdf'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'sma_vs_ecc.png'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        plt.show()

    if 1:
        # Create figure for orbit plots
        fig = plt.figure(figsize=(12,5.5))
        post = post_withrvs

        num_orbits_to_plot= 100 # Will plot 100 randomly selected orbits of this companion
        start_mjd=data_table_withrvs['epoch'][0] # Minimum MJD for colorbar (here we choose first data epoch)
        data_table=data_table_withrvs
        cbar_param='epochs'
        total_mass=system_mass
        parallax=plx
        system_rv=sysrv
        num_epochs_to_plot=50
        object_mass = 0
        square_plot=True
        tau_ref_epoch = loaded_results_withrvs.tau_ref_epoch
        sep_pa_end_year=2025.0
        show_colorbar=True

        start_yr = Time(start_mjd,format='mjd').decimalyear

        ax0 = plt.subplot2grid((6, 14), (0, 0), rowspan=6, colspan=6)
        ax11 = plt.subplot2grid((6, 14), (0, 9), colspan=6)
        ax21 = plt.subplot2grid((6, 14), (2, 9), colspan=6)
        ax12 = plt.subplot2grid((6, 14), (1, 9), colspan=6)
        ax22 = plt.subplot2grid((6, 14), (3, 9), colspan=6)
        ax3 = plt.subplot2grid((6, 14), (4, 9), rowspan=2, colspan=6)
        # cmap = mpl.cm.Purples_r
        # exit()
        for object_to_plot,cmap,pl_color,ax1,ax2 in zip([1,2],[mpl.cm.Blues_r,mpl.cm.Oranges_r],["#006699","#ff9900"],[ax11,ax12],[ax21,ax22]): # Plot orbits for the first (and only, in this case) companion
            print(object_to_plot,cmap,pl_color)
            print(data_table)

            if data_table is not None:
                radec_indices = np.where((data_table['quant_type']=='radec')*(data_table['object']==object_to_plot))
                seppa_indices = np.where((data_table['quant_type']=='seppa')*(data_table['object']==object_to_plot))
                rv_indices = np.where((data_table['quant_type']=='rv')*(data_table['object']==object_to_plot))

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ErfaWarning)

                dict_of_indices = {
                    'sma': 0,
                    'ecc': 1,
                    'inc': 2,
                    'aop': 3,
                    'pan': 4,
                    'tau': 5
                }

                if cbar_param == 'epochs':
                    pass
                elif cbar_param[0:3] in dict_of_indices:
                    try:
                        object_id = np.int(cbar_param[3:])
                    except ValueError:
                        object_id = 1

                    index = dict_of_indices[cbar_param[0:3]] + 6*(object_id-1)
                else:
                    raise Exception('Invalid input; acceptable inputs include epochs, sma1, ecc1, inc1, aop1, pan1, tau1, sma2, ecc2, ...')


                # Split the 2-D post array into series of 1-D arrays for each orbital parameter
                num_objects, remainder = np.divmod(post.shape[1],6)

                sma = post[:,dict_of_indices['sma']+(object_to_plot-1)*6]
                ecc = post[:,dict_of_indices['ecc']+(object_to_plot-1)*6]
                inc = post[:,dict_of_indices['inc']+(object_to_plot-1)*6]
                aop = post[:,dict_of_indices['aop']+(object_to_plot-1)*6]
                pan = post[:,dict_of_indices['pan']+(object_to_plot-1)*6]
                tau = post[:,dict_of_indices['tau']+(object_to_plot-1)*6]
                sysrv = post[:,-2]

                # Then, get the other parameters
                if remainder == 3: # have samples for parallax, system rv, and mtot
                    plx = post[:,-3]
                    sysrv = post[:,-2]
                    mtot = post[:,-1]
                elif remainder == 2: # have samples for parallax, system rv, and mtot
                    plx = post[:,-2]
                    if system_rv is not None:
                        sysrv = np.ones(len(sma))*system_rv
                    else:
                        raise Exception('results.Results.plot_orbits(): system radial velocity must be provided if not part of samples')
                    mtot = post[:,-1]
                else: # otherwise make arrays out of user provided value
                    if total_mass is not None:
                        mtot = np.ones(len(sma))*total_mass
                    else:
                        raise Exception('results.Results.plot_orbits(): total mass must be provided if not part of samples')
                    if parallax is not None:
                        plx = np.ones(len(sma))*parallax
                    else:
                        raise Exception('results.Results.plot_orbits(): parallax must be provided if not part of samples')
                mplanet = np.ones(len(sma))*object_mass

                # Select random indices for plotted orbit
                if num_orbits_to_plot > len(sma):
                    num_orbits_to_plot = len(sma)
                choose = np.random.randint(0, high=len(sma), size=num_orbits_to_plot)

                raoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
                deoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
                seps = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
                pas = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
                rvs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
                epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
                yr_epochs = np.zeros((num_orbits_to_plot, num_epochs_to_plot))

                # Compute period (from Kepler's third law)
                period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
                period = period.to(u.day).value
                # Loop through each orbit to plot and calcualte ra/dec offsets for all points in orbit
                # Need this loops since epochs[] vary for each orbit, unless we want to just plot the same time period for all orbits
                for i in np.arange(num_orbits_to_plot):
                    orb_ind = choose[i]
                    # Create an epochs array to plot num_epochs_to_plot points over one orbital period
                    epochs[i,:] = np.linspace(start_mjd, float(start_mjd+period[orb_ind]), num_epochs_to_plot)

                    # Calculate ra/dec offsets for all epochs of this orbit
                    raoff0, deoff0, relrv0 = kepler.calc_orbit(
                        epochs[i,:], sma[orb_ind], ecc[orb_ind], inc[orb_ind], aop[orb_ind], pan[orb_ind],
                        tau[orb_ind], plx[orb_ind], mtot[orb_ind], mass=mplanet[orb_ind], tau_ref_epoch=tau_ref_epoch
                    )

                    raoff[i,:] = raoff0
                    deoff[i,:] = deoff0
                    rvs[i,:] = relrv0+sysrv[orb_ind]

                    seps[i,:], pas[i,:] = orbitize.system.radec2seppa(raoff[i,:], deoff[i,:])

                    yr_epochs[i,:] = Time(epochs[i,:],format='mjd').decimalyear
                    # plot_epochs = np.where(yr_epochs[i,:] <= sep_pa_end_year)[0]
                    # yr_epochs = yr_epochs[plot_epochs]

                # Create a linearly increasing colormap for our range of epochs
                if cbar_param != 'epochs':
                    cbar_param_arr = post[:,index]
                    norm = mpl.colors.Normalize(vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr))
                    norm_yr = mpl.colors.Normalize(vmin=np.min(cbar_param_arr), vmax=np.max(cbar_param_arr))

                elif cbar_param == 'epochs':
                    norm = mpl.colors.Normalize(vmin=np.min(epochs), vmax=np.max(epochs[-1,:]))

                    norm_yr = mpl.colors.Normalize(
                    vmin=np.min(Time(epochs,format='mjd').decimalyear),
                    vmax=np.max(Time(epochs,format='mjd').decimalyear)
                    )



                plt.sca(ax0)
                # Plot each orbit (each segment between two points coloured using colormap)
                for i in np.arange(num_orbits_to_plot):
                    points = np.array([raoff[i,:], deoff[i,:]]).T.reshape(-1,1,2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(
                        segments, cmap=cmap, norm=norm, linewidth=1.0
                    )
                    if cbar_param != 'epochs':
                        lc.set_array(np.ones(len(epochs[0]))*cbar_param_arr[i])
                    elif cbar_param == 'epochs':
                        lc.set_array(epochs[i,:])
                    ax0.add_collection(lc)

                # modify the axes
                if square_plot:
                    adjustable_param='datalim'
                else:
                    adjustable_param='box'
                ax0.set_aspect('equal', adjustable=adjustable_param)
                ax0.set_xlabel('$\Delta$RA (mas)',fontsize=fontsize)
                ax0.set_ylabel('$\Delta$Dec (mas)',fontsize=fontsize)
                ax0.tick_params(axis='x', labelsize=fontsize)
                ax0.tick_params(axis='y', labelsize=fontsize)
                plt.sca(ax0)
                plt.xlim([-2000,2000])
                plt.ylim([-2000,2000])
                plt.xticks([2000,1000,0,-1000,-2000])
                plt.yticks([-2000,-1000,0,1000,2000])

                if data_table is not None:
                    plt.errorbar(data_table["quant1"][radec_indices],data_table["quant2"][radec_indices],
                                 xerr=data_table["quant1_err"][radec_indices],
                                 yerr=data_table["quant2_err"][radec_indices],fmt="x",color=pl_color)

                    for seppa_index in seppa_indices[0]:
                        ra_from_seppa = data_table["quant1"][seppa_index]*np.sin(np.deg2rad(data_table["quant2"][seppa_index]))
                        dec_from_seppa = data_table["quant1"][seppa_index]*np.cos(np.deg2rad(data_table["quant2"][seppa_index]))
                        dra_from_seppa = data_table["quant1_err"][seppa_index]*np.sin(np.deg2rad(data_table["quant2"][seppa_index]))
                        ddec_from_seppa = data_table["quant1_err"][seppa_index]*np.cos(np.deg2rad(data_table["quant2"][seppa_index]))
                        plt.plot(ra_from_seppa,dec_from_seppa,"o",color=pl_color)
                        plt.plot([ra_from_seppa-dra_from_seppa,ra_from_seppa+dra_from_seppa],
                                 [dec_from_seppa-ddec_from_seppa,dec_from_seppa+ddec_from_seppa],color=pl_color, linestyle ="--")
                        e1 = mpl.patches.Arc((0,0),2*data_table["quant1"][seppa_index],2*data_table["quant1"][seppa_index],0,
                                             theta2=90-(data_table["quant2"][seppa_index]-data_table["quant2_err"][seppa_index]),
                                             theta1=90-(data_table["quant2"][seppa_index]+data_table["quant2_err"][seppa_index]),
                                             color=pl_color, linestyle ="--")
                        ax0.add_patch(e1)
                ax0.invert_xaxis()

                # add colorbar
                if object_to_plot == 1:
                    cbar_ax = fig.add_axes([0.47, 0.10, 0.015, 0.4]) # xpos, ypos, width, height, in fraction of figure size
                    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm_yr, orientation='vertical')
                    cbar.set_label("b: "+cbar_param,fontsize=fontsize)
                if object_to_plot == 2:
                    cbar_ax = fig.add_axes([0.47, 0.5, 0.015, 0.4]) # xpos, ypos, width, height, in fraction of figure size
                    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm_yr, orientation='vertical')
                    cbar.set_label("c: "+cbar_param,fontsize=fontsize)
                cbar.ax.tick_params(labelsize=fontsize)


                plt.sca(ax1)
                for i in np.arange(num_orbits_to_plot):
                    points = np.array([yr_epochs[i,:], seps[i,:]]).T.reshape(-1,1,2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(
                        segments, cmap=cmap, norm=norm, linewidth=1.0,alpha=0.5
                    )
                    if cbar_param != 'epochs':
                        lc.set_array(np.ones(len(epochs[0]))*cbar_param_arr[i])
                    elif cbar_param == 'epochs':
                        lc.set_array(epochs[i,:])
                    ax1.add_collection(lc)
                plt.xlim([start_yr,sep_pa_end_year])


                plt.sca(ax2)
                for i in np.arange(num_orbits_to_plot):
                    points = np.array([yr_epochs[i,:], pas[i,:]]).T.reshape(-1,1,2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(
                        segments, cmap=cmap, norm=norm, linewidth=1.0,alpha=0.5
                    )
                    if cbar_param != 'epochs':
                        lc.set_array(np.ones(len(epochs[0]))*cbar_param_arr[i])
                    elif cbar_param == 'epochs':
                        lc.set_array(epochs[i,:])
                    ax2.add_collection(lc)
                plt.xlim([start_yr,sep_pa_end_year])

                plt.sca(ax3)
                for i in np.arange(num_orbits_to_plot):
                    points = np.array([yr_epochs[i,:], rvs[i,:]]).T.reshape(-1,1,2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(
                        segments, cmap=cmap, norm=norm, linewidth=1.0,alpha=0.5
                    )
                    if cbar_param != 'epochs':
                        lc.set_array(np.ones(len(epochs[0]))*cbar_param_arr[i])
                    elif cbar_param == 'epochs':
                        lc.set_array(epochs[i,:])
                    ax3.add_collection(lc)
                plt.xlim([start_yr,sep_pa_end_year])


                if data_table is not None:
                    plt.sca(ax1)
                    eb1 = plt.errorbar(Time(data_table["epoch"][seppa_indices],format='mjd').decimalyear,
                                 data_table["quant1"][seppa_indices],
                                 yerr=data_table["quant1_err"][seppa_indices],fmt="x",color=pl_color,linestyle="",zorder=10)
                    eb1[-1][0].set_linestyle("-")
                    plt.xticks([],[])
                    plt.sca(ax2)
                    eb2 = plt.errorbar(Time(data_table["epoch"][seppa_indices],format='mjd').decimalyear,
                                 data_table["quant2"][seppa_indices],
                                 yerr=data_table["quant2_err"][seppa_indices],fmt="x",color=pl_color,linestyle="",zorder=10)
                    eb2[-1][0].set_linestyle("-")
                    plt.xticks([],[])
                    plt.sca(ax3)
                    eb3 = plt.errorbar(Time(data_table["epoch"][rv_indices],format='mjd').decimalyear,
                                 data_table["quant1"][rv_indices],
                                 yerr=data_table["quant1_err"][rv_indices],fmt="x",color=pl_color,linestyle="",zorder=10)
                    eb3[-1][0].set_linestyle("-")

                    #Monte Carlo error for radec
                    ra_list = data_table["quant1"][radec_indices]
                    dec_list = data_table["quant2"][radec_indices]
                    ra_err_list = data_table["quant1_err"][radec_indices]
                    dec_err_list = data_table["quant2_err"][radec_indices]
                    # sep_list,pa_list = orbitize.system.radec2seppa(ra_list,dec_list)
                    sep_merr_list = np.zeros(ra_list.shape)
                    pa_merr_list = np.zeros(ra_list.shape)
                    sep_perr_list = np.zeros(ra_list.shape)
                    pa_perr_list = np.zeros(ra_list.shape)
                    sep_list = np.zeros(ra_list.shape)
                    pa_list = np.zeros(ra_list.shape)
                    for myid,(ra,dec,ra_err,dec_err) in enumerate(zip(ra_list,dec_list,ra_err_list,dec_err_list)):
                        mean = [ra,dec]
                        cov=np.diag([ra_err**2,dec_err**2])
                        radec_samples = np.random.multivariate_normal(mean,cov,size=1000)
                        sep_samples,pa_samples = orbitize.system.radec2seppa(radec_samples[:,0],radec_samples[:,1])
                        seppost,xedges = np.histogram(sep_samples,bins=25,range=[np.min(sep_samples),np.max(sep_samples)])
                        x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
                        sep_mod, _,_,sep_merr, sep_perr,_ = get_err_from_posterior(x_centers,seppost)
                        # plt.figure(10)
                        # plt.subplot(1,2,1)
                        # plt.plot(x_centers,seppost)
                        # print(np.min(sep_samples),np.max(sep_samples))
                        # print(sep_mod,sep_merr, sep_perr,np.std(sep_samples))
                        papost,xedges = np.histogram(pa_samples,bins=25,range=[np.min(pa_samples),np.max(pa_samples)])
                        x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
                        pa_mod, _,_,pa_merr, pa_perr,_ = get_err_from_posterior(x_centers,papost)
                        # plt.subplot(1,2,2)
                        # plt.plot(x_centers,papost)
                        # print(np.min(papost),np.max(papost))
                        # print(pa_mod,pa_merr, pa_perr,np.std(pa_samples))
                        # plt.show()
                        # exit()
                        sep_merr_list[myid] = sep_merr
                        pa_merr_list[myid] = pa_merr
                        sep_perr_list[myid] = sep_perr
                        pa_perr_list[myid] = pa_perr
                        sep_list[myid] = sep_mod
                        pa_list[myid] = pa_mod

                    plt.sca(ax1)
                    plt.errorbar(Time(data_table["epoch"][radec_indices],format='mjd').decimalyear,
                                 sep_list,
                                 yerr=[-sep_merr_list,sep_perr_list],fmt="x",color=pl_color,linestyle="",zorder=10)
                    plt.sca(ax2)
                    plt.errorbar(Time(data_table["epoch"][radec_indices],format='mjd').decimalyear,
                                 pa_list,
                                 yerr=[-pa_merr_list,pa_perr_list],fmt="x",color=pl_color,linestyle="",zorder=10)

        plt.sca(ax11)
        plt.ylim([1700,1730])
        ax11.set_ylabel('$\\rho_b$ (mas)',fontsize=fontsize)
        plt.yticks([1700,1710,1720,1730])
        ax11.tick_params(axis='x', labelsize=fontsize)
        ax11.tick_params(axis='y', labelsize=fontsize)

        plt.sca(ax12)
        plt.ylim([930,970])
        ax12.set_ylabel('$\\rho_c$ (mas)',fontsize=fontsize)
        plt.yticks([930,950,970])
        ax12.yaxis.tick_right()
        ax12.yaxis.set_label_position("right")
        ax12.tick_params(axis='x', labelsize=fontsize)
        ax12.tick_params(axis='y', labelsize=fontsize)

        plt.sca(ax21)
        plt.ylim([55,70])
        ax21.set_ylabel('PA$_b$ (deg)',fontsize=fontsize)
        plt.yticks([55,65,75])
        ax21.tick_params(axis='x', labelsize=fontsize)
        ax21.tick_params(axis='y', labelsize=fontsize)

        plt.sca(ax22)
        plt.ylim([300,340])
        ax22.set_ylabel('PA$_c$ (deg)',fontsize=fontsize)
        plt.yticks([300,320,340])
        ax22.yaxis.tick_right()
        ax22.yaxis.set_label_position("right")
        ax22.tick_params(axis='x', labelsize=fontsize)
        ax22.tick_params(axis='y', labelsize=fontsize)

        plt.sca(ax3)
        plt.ylim([-20,0])
        plt.yticks([-20,-15,-10,-5,0])
        ax3.set_ylabel('RV (km/s)',fontsize=fontsize)
        ax3.set_xlabel('Epoch (yr)',fontsize=fontsize)
        ax3.tick_params(axis='x', labelsize=fontsize)
        ax3.tick_params(axis='y', labelsize=fontsize)

        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_bc_withrvs.pdf')) # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_bc_withrvs.png')) # This is matplotlib.figure.Figure.savefig()
        plt.show()


    # exit()
    # print(loaded_results.post.shape)
    # # exit()
    # # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1"]
    # # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","plx","mtot"]
    # # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","plx","sysrv","mtot"]
    # # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","plx","mtot"]
    # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","plx","sysrv","mtot"]
    # loaded_results.post = loaded_results.post[::10,:]
    # corner_plot_fig = loaded_results.plot_corner(param_list=param_list)
    # corner_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"corner_plot_{0}_{1}_{2}.png".format(rv_str,planet,suffix)))
    # exit()

    # fig = loaded_results.plot_orbits(
    #     object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
    #     num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
    #     start_mjd=data_table['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
    #     data_table=data_table,
    #     cbar_param="sma2"
    #     # total_mass=system_mass,
    #     # parallax=plx
    # )
    # fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_plot_{0}_{1}_{2}_obj1.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
    # fig = loaded_results.plot_orbits(
    #     object_to_plot = 2, # Plot orbits for the first (and only, in this case) companion
    #     num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
    #     start_mjd=data_table['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
    #     data_table=data_table,
    #     cbar_param="sma2"
    #     # total_mass=system_mass,
    #     # parallax=plx
    # )
    # fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_plot_{0}_{1}_{2}_obj2.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
    #
    # fig = loaded_results.plot_rvs(
    #     object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
    #     num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
    #     start_mjd=data_table['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
    #     data_table=data_table
    # )
    # fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'rv_plot_{0}_{1}_{2}_obj1.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
    # fig = loaded_results.plot_rvs(
    #     object_to_plot = 2, # Plot orbits for the first (and only, in this case) companion
    #     num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
    #     start_mjd=data_table['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
    #     data_table=data_table
    # )
    # fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'rv_plot_{0}_{1}_{2}_obj2.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
    # plt.show()


