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
from orbitize import results
import matplotlib.pyplot as plt


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
    return lf(1-0.6827),lf(0.6827)

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

# osiris_data_dir = "/data/osiris_data"
osiris_data_dir = "/scr3/jruffio/data/osiris_data"
astrometry_DATADIR = os.path.join(osiris_data_dir,"astrometry")
#out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
out_pngs = os.path.join(osiris_data_dir,"astrometry","figures")
sysrv=-12.6
sysrv_err=1.4
fontsize = 12
planet = "bcd"


if 0: # mutual inclination vs step
    N_steps = 0
    for it in range(5):
        suffix_withrvs = "restriOme_it{0}_16_512_100000_50_True".format(it)
        with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs))) as hdulist:
            chains_withrvs = hdulist[0].data
            print(chains_withrvs.shape)
            chains_withrvs = chains_withrvs[0,::10,::10,:]
            steps = np.arange(N_steps,N_steps+chains_withrvs.shape[1]*50*10,50*10)
            N_steps+=chains_withrvs.shape[1]*50*10
            # print(steps)
            # exit()


        im_bc_samples = np.rad2deg(np.arccos(np.cos(chains_withrvs[:,:,2]) * np.cos(chains_withrvs[:,:,2+6]) + np.sin(chains_withrvs[:,:,2]) * np.sin(chains_withrvs[:,:,2+6]) * np.cos(chains_withrvs[:,:,4] - chains_withrvs[:,:,4+6])))
        plt.subplot(3,1,1)
        plt.plot(np.tile(steps[:,None],(1,im_bc_samples.shape[0])),im_bc_samples.T,alpha=0.02)
        plt.ylabel("i_m (b-c)")
        plt.xlabel("steps")

        im_cd_samples = np.rad2deg(np.arccos(np.cos(chains_withrvs[:,:,2+6]) * np.cos(chains_withrvs[:,:,2+12]) + np.sin(chains_withrvs[:,:,2+6]) * np.sin(chains_withrvs[:,:,2+12]) * np.cos(chains_withrvs[:,:,4+6] - chains_withrvs[:,:,4+12])))
        plt.subplot(3,1,2)
        plt.plot(np.tile(steps[:,None],(1,im_cd_samples.shape[0])),im_cd_samples.T,alpha=0.02)
        plt.ylabel("i_m (c-d)")
        plt.xlabel("steps")

        im_bd_samples = np.rad2deg(np.arccos(np.cos(chains_withrvs[:,:,2]) * np.cos(chains_withrvs[:,:,2+12]) + np.sin(chains_withrvs[:,:,2]) * np.sin(chains_withrvs[:,:,2+12]) * np.cos(chains_withrvs[:,:,4] - chains_withrvs[:,:,4+12])))
        plt.subplot(3,1,3)
        plt.plot(np.tile(steps[:,None],(1,im_bd_samples.shape[0])),im_bd_samples.T,alpha=0.02)
        plt.ylabel("i_m (b-d)")
        plt.xlabel("steps")
    plt.show()



if 1: # plot inclination and Omega
    # suffix_withrvs =  "sherlock_16_512_100000_50_False"
    suffix_withrvs =  "single_planet2_16_512_100000_50_False"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+"b",'chain_{0}_{1}_{2}.fits'.format("norv","b",suffix_withrvs))) as hdulist:
        chains_withrvs = hdulist[0].data
        chains_withrvs = chains_withrvs[0,:,chains_withrvs.shape[2]-1000::,:]
        post_b_norv = np.reshape(chains_withrvs,(chains_withrvs.shape[0]*chains_withrvs.shape[1],chains_withrvs.shape[2]))
    # suffix_withrvs =  "sherlock_16_512_100000_50_False"
    suffix_withrvs =  "single_planet2_16_512_100000_50_False"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+"c",'chain_{0}_{1}_{2}.fits'.format("norv","c",suffix_withrvs))) as hdulist:
        chains_withrvs = hdulist[0].data
        chains_withrvs = chains_withrvs[0,:,chains_withrvs.shape[2]-1000::,:]
        post_c_norv = np.reshape(chains_withrvs,(chains_withrvs.shape[0]*chains_withrvs.shape[1],chains_withrvs.shape[2]))
    # suffix_withrvs =  "sherlock_16_512_100000_50_False"
    suffix_withrvs =  "single_planet2_16_512_100000_50_False"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+"d",'chain_{0}_{1}_{2}.fits'.format("norv","d",suffix_withrvs))) as hdulist:
        chains_withrvs = hdulist[0].data
        chains_withrvs = chains_withrvs[0,:,chains_withrvs.shape[2]-1000::,:]
        post_d_norv = np.reshape(chains_withrvs,(chains_withrvs.shape[0]*chains_withrvs.shape[1],chains_withrvs.shape[2]))

    # suffix_withrvs =  "it8_16_512_100000_50_True"
    # suffix_withrvs = "from_scratch_it1_16_512_100000_50_True"
    suffix_withrvs = "restriOme_it4_16_512_100000_50_True"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs))) as hdulist:
        chains_withrvs = hdulist[0].data
        print(chains_withrvs.shape)
        # chains_withrvs = chains_withrvs[0,:,chains_withrvs.shape[2]-25::,:]
        # chains_withrvs = chains_withrvs[0,:,0:25,:]
        chains_withrvs = chains_withrvs[0,:,:,:]
        print(chains_withrvs.shape)
        post_withrvs = np.reshape(chains_withrvs,(chains_withrvs.shape[0]*chains_withrvs.shape[1],chains_withrvs.shape[2]))
    # hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format("withrvs",planet,suffix_withrvs))
    # print(hdf5_filename)
    # loaded_results_withrvs = results.Results() # Create blank results object for loading
    # loaded_results_withrvs.load_results(hdf5_filename)

    # suffix_withrvs_copl =  "it2_16_512_100000_50_True_coplanar"
    # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs_copl))) as hdulist:
    #     chains_withrvs_copl = hdulist[0].data
    #     print(chains_withrvs_copl.shape)
    # suffix_withrvs_copl =  "it3_16_512_100000_50_True_coplanar"
    # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs_copl))) as hdulist:
    #     chains_withrvs_copl = hdulist[0].data
    #     print(chains_withrvs_copl.shape)
    # suffix_withrvs_copl =  "it4_16_512_100000_50_True_coplanar"
    # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs_copl))) as hdulist:
    #     chains_withrvs_copl = hdulist[0].data
    #     print(chains_withrvs_copl.shape)
    # suffix_withrvs_copl =  "it5_16_512_100000_50_True_coplanar"
    # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs_copl))) as hdulist:
    #     chains_withrvs_copl = hdulist[0].data
    #     print(chains_withrvs_copl.shape)
    suffix_withrvs_copl =  "it6_16_512_100000_50_True_coplanar"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs_copl))) as hdulist:
        chains_withrvs_copl = hdulist[0].data
        print(chains_withrvs_copl.shape)
    suffix_withrvs_copl =  "it7_16_512_100000_50_True_coplanar"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs_copl))) as hdulist:
        chains_withrvs_copl = np.concatenate([hdulist[0].data,chains_withrvs_copl],axis=2)
        print(chains_withrvs_copl.shape)
        # exit()
        # chains_withrvs_copl = chains_withrvs_copl[0,:,chains_withrvs_copl.shape[2]-25::,:]
        # chains_withrvs_copl = chains_withrvs_copl[0,:,0:25,:]
        chains_withrvs_copl = chains_withrvs_copl[0,:,:,:]

        _chains_withrvs_copl = np.zeros((chains_withrvs_copl.shape[0],chains_withrvs_copl.shape[1],chains_withrvs_copl.shape[2]+4))
        a_list = [0,1,2,3,4,5, 6,7,2,8,4,9, 10,11,2,12,4,13, 14,15,16]
        b_list = np.arange(21)
        for a,b in zip(a_list,b_list):
            _chains_withrvs_copl[:,:,b] = chains_withrvs_copl[:,:,a]
        chains_withrvs_copl =_chains_withrvs_copl
        print(chains_withrvs_copl.shape)
        post_withrvs_copl = np.reshape(chains_withrvs_copl,(chains_withrvs_copl.shape[0]*chains_withrvs_copl.shape[1],chains_withrvs_copl.shape[2]))


    # suffix_norv_copl =  "it7_16_512_100000_50_False_coplanar"
    suffix_norv_copl = "from_scratch_it1_16_512_100000_50_False_coplanar"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("norv",planet,suffix_norv_copl))) as hdulist:
        chains_norv_copl = hdulist[0].data
        # chains_norv_copl = chains_norv_copl[0,:,chains_norv_copl.shape[2]-100::,:]
        # chains_norv_copl = chains_norv_copl[0,:,500:525,:]
        chains_norv_copl = chains_norv_copl[0,:,:,:]
        _chains_norv_copl = np.zeros((chains_norv_copl.shape[0],chains_norv_copl.shape[1],chains_norv_copl.shape[2]+5))
        a_list = [0,1,2,3,4,5, 6,7,2,8,4,9, 10,11,2,12,4,13, 14,None,15]
        b_list = np.arange(21)
        for a,b in zip(a_list,b_list):
            if a is not None:
                _chains_norv_copl[:,:,b] = chains_norv_copl[:,:,a]
            else:
                _chains_norv_copl[:,:,b] = -10.5
        chains_norv_copl =_chains_norv_copl
        print(chains_norv_copl.shape)

        post_norv_copl = np.reshape(chains_norv_copl,(chains_norv_copl.shape[0]*chains_norv_copl.shape[1],chains_norv_copl.shape[2]))


    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'Samples_nw512_lo-0566.fits')) as hdulist:
        chains_rv_Rob = hdulist[0].data
        post_rv_Rob = np.reshape(chains_rv_Rob,(chains_rv_Rob.shape[0]*chains_rv_Rob.shape[1],chains_rv_Rob.shape[2]))
        print(chains_rv_Rob.shape)
    # plt.subplot(1,2,1)
    # plt.plot(np.rad2deg(chains_withrvs[:,:,4+6].T),alpha=0.05)
    # plt.subplot(1,2,2)
    # plt.plot(np.rad2deg(chains_withrvs[:,:,4].T)-np.rad2deg(chains_withrvs[:,:,4+6].T),alpha=0.05)
    # plt.ylabel("inc")
    # plt.xlabel("it*50")
    # # plt.plot(chains_withrvs_copl[:,:,19].T,alpha=0.05)
    # print(np.rad2deg(chains_withrvs[:,:10,4].T)-np.rad2deg(chains_withrvs[:,1:11,4+6].T))
    # print(np.nanmedian(np.rad2deg(chains_withrvs[:,:,2].T)-np.rad2deg(chains_withrvs[:,:,2+6].T),axis=1))
    # plt.show()

    Ome_bounds = [0,360]
    inc_bounds = [0,60]
    # Ome_inc_bins = [360 / 40, 60 // 5]
    Ome_inc_bins = [360 / 10, 60 // 1]
    # print(np.rad2deg(post_rv_Rob[:,4]))
    # print(post_rv_Rob[:,1])
    # print((np.arccos(post_rv_Rob[:,1])))
    # print(np.rad2deg(np.arccos(post_rv_Rob[:,1])))
    # exit()
    inc_Ome_rv_Rob_b,xedges,yedges = np.histogram2d(np.rad2deg(post_rv_Rob[:,4]),np.rad2deg(np.arccos(post_rv_Rob[:,1])),
                                                  bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_rv_Rob_c,xedges,yedges = np.histogram2d(np.rad2deg(post_rv_Rob[:,4+6]),np.rad2deg(np.arccos(post_rv_Rob[:,1+6])),
                                                  bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_rv_Rob_d,xedges,yedges = np.histogram2d(np.rad2deg(post_rv_Rob[:,4+12]),np.rad2deg(np.arccos(post_rv_Rob[:,1+12])),
                                                  bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])

    im_bc_norv_samples = np.rad2deg(np.arccos(np.cos(post_b_norv[:,2]) * np.cos(post_c_norv[:,2]) + np.sin(post_b_norv[:,2]) * np.sin(post_c_norv[:,2]) * np.cos(post_b_norv[:,4] - post_c_norv[:,4])))
    im_bc_norv_post, imutual_edges = np.histogram(im_bc_norv_samples, bins=500, range=[-360,360])
    im_cd_norv_samples = np.rad2deg(np.arccos(np.cos(post_c_norv[:,2]) * np.cos(post_d_norv[:,2]) + np.sin(post_c_norv[:,2]) * np.sin(post_d_norv[:,2]) * np.cos(post_c_norv[:,4] - post_d_norv[:,4])))
    im_cd_norv_post, imutual_edges = np.histogram(im_cd_norv_samples, bins=500, range=[-360,360])
    im_bd_norv_samples = np.rad2deg(np.arccos(np.cos(post_b_norv[:,2]) * np.cos(post_d_norv[:,2]) + np.sin(post_b_norv[:,2]) * np.sin(post_d_norv[:,2]) * np.cos(post_b_norv[:,4] - post_d_norv[:,4])))
    im_bd_norv_post, imutual_edges = np.histogram(im_bd_norv_samples, bins=500, range=[-360,360])

    im_bc_samples = np.rad2deg(
        np.arccos(np.cos(post_withrvs[:,2]) * np.cos(post_withrvs[:,2+6]) + np.sin(post_withrvs[:,2]) * np.sin(post_withrvs[:,2+6]) * np.cos(post_withrvs[:,4] - post_withrvs[:,4+6])))
    im_bc_post, imutual_edges = np.histogram(im_bc_samples, bins=500, range=[-360,360])
    im_cd_samples = np.rad2deg(
        np.arccos(np.cos(post_withrvs[:,2+6]) * np.cos(post_withrvs[:,2+12]) + np.sin(post_withrvs[:,2+6]) * np.sin(post_withrvs[:,2+12]) * np.cos(post_withrvs[:,4+6] - post_withrvs[:,4+12])))
    im_cd_post, imutual_edges = np.histogram(im_cd_samples, bins=500, range=[-360,360])
    im_bd_samples = np.rad2deg(
        np.arccos(np.cos(post_withrvs[:,2]) * np.cos(post_withrvs[:,2+12]) + np.sin(post_withrvs[:,2]) * np.sin(post_withrvs[:,2+12]) * np.cos(post_withrvs[:,4] - post_withrvs[:,4+12])))
    im_bd_post, imutual_edges = np.histogram(im_bd_samples, bins=500, range=[-360,360])

    imutual_centers = [(x1+x2)/2. for x1,x2 in zip(imutual_edges[0:len(imutual_edges)-1],imutual_edges[1:len(imutual_edges)])]

    post_withrvs = post_withrvs[np.where(im_cd_samples<20)[0],:]
    im_bc_samples = np.rad2deg(
        np.arccos(np.cos(post_withrvs[:,2]) * np.cos(post_withrvs[:,2+6]) + np.sin(post_withrvs[:,2]) * np.sin(post_withrvs[:,2+6]) * np.cos(post_withrvs[:,4] - post_withrvs[:,4+6])))
    im_bc_post, imutual_edges = np.histogram(im_bc_samples, bins=500, range=[-360,360])
    im_cd_samples = np.rad2deg(
        np.arccos(np.cos(post_withrvs[:,2+6]) * np.cos(post_withrvs[:,2+12]) + np.sin(post_withrvs[:,2+6]) * np.sin(post_withrvs[:,2+12]) * np.cos(post_withrvs[:,4+6] - post_withrvs[:,4+12])))
    im_cd_post, imutual_edges = np.histogram(im_cd_samples, bins=500, range=[-360,360])
    im_bd_samples = np.rad2deg(
        np.arccos(np.cos(post_withrvs[:,2]) * np.cos(post_withrvs[:,2+12]) + np.sin(post_withrvs[:,2]) * np.sin(post_withrvs[:,2+12]) * np.cos(post_withrvs[:,4] - post_withrvs[:,4+12])))
    im_bd_post, imutual_edges = np.histogram(im_bd_samples, bins=500, range=[-360,360])

    cmaps = ["cool","hot","hot"]
    # inc_Ome_norv_b,xedges,yedges = np.histogram2d(np.rad2deg(np.concatenate([post_norv[:,4],post_norv[:,4]+np.pi])),np.rad2deg(np.concatenate([post_norv[:,2],post_norv[:,2]])),bins=[360/4,45//2],range=[Ome_bounds,inc_bounds])
    inc_Ome_norv_b,xedges,yedges = np.histogram2d(np.rad2deg(np.concatenate([post_b_norv[:,4],np.mod(post_b_norv[:,4]+np.pi,2*np.pi)])),np.rad2deg(np.concatenate([post_b_norv[:,2],post_b_norv[:,2]])),#np.rad2deg(post_b_norv[:,4]),np.rad2deg(post_b_norv[:,2]),
                                                  bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_norv_c,xedges,yedges = np.histogram2d(np.rad2deg(np.concatenate([post_c_norv[:,4],np.mod(post_c_norv[:,4]+np.pi,2*np.pi)])),np.rad2deg(np.concatenate([post_c_norv[:,2],post_c_norv[:,2]])),#np.rad2deg(post_c_norv[:,4]),np.rad2deg(post_c_norv[:,2]),
                                                  bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_norv_d,xedges,yedges = np.histogram2d(np.rad2deg(np.concatenate([post_d_norv[:,4],np.mod(post_d_norv[:,4]+np.pi,2*np.pi)])),np.rad2deg(np.concatenate([post_d_norv[:,2],post_d_norv[:,2]])),#np.rad2deg(post_d_norv[:,4]),np.rad2deg(post_d_norv[:,2]),
                                                  bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    # inc_Ome_norv_b,xedges,yedges = np.histogram2d(np.rad2deg(post_b_norv[:,4]),np.rad2deg(post_b_norv[:,2]),
    #                                               bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    # inc_Ome_norv_c,xedges,yedges = np.histogram2d(np.rad2deg(post_c_norv[:,4]),np.rad2deg(post_c_norv[:,2]),
    #                                               bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    # inc_Ome_norv_d,xedges,yedges = np.histogram2d(np.rad2deg(post_d_norv[:,4]),np.rad2deg(post_d_norv[:,2]),
    #                                               bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_withrvs_b,xedges,yedges = np.histogram2d(np.rad2deg(post_withrvs[:,4]),np.rad2deg(post_withrvs[:,2]),bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_withrvs_c,xedges,yedges = np.histogram2d(np.rad2deg(post_withrvs[:,4+6]),np.rad2deg(post_withrvs[:,2+6]),bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_withrvs_d,xedges,yedges = np.histogram2d(np.rad2deg(post_withrvs[:,4+12]),np.rad2deg(post_withrvs[:,2+12]),bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_withrvs_copl,xedges,yedges = np.histogram2d(np.rad2deg(post_withrvs_copl[:,4]),np.rad2deg(post_withrvs_copl[:,2]),bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_norv_copl,xedges,yedges = np.histogram2d(np.rad2deg(np.concatenate([post_norv_copl[:,4],np.mod(post_norv_copl[:,4]+np.pi,2*np.pi)])),np.rad2deg(np.concatenate([post_norv_copl[:,2],post_norv_copl[:,2]])),#np.rad2deg(post_norv_copl[:,4]),np.rad2deg(post_norv_copl[:,2]),
                                                     bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    sma_ecc_withrvs_copl_b,aedges,eedges = np.histogram2d(post_withrvs_copl[:,0],post_withrvs_copl[:,1],bins=[50,20],range=[[0,80],[0,0.4]])
    sma_ecc_norvs_copl_b,aedges,eedges = np.histogram2d(post_norv_copl[:,0],post_norv_copl[:,1],bins=[50,20],range=[[0,80],[0,0.4]])
    sma_ecc_withrvs_copl_c,aedges,eedges = np.histogram2d(post_withrvs_copl[:,0+6],post_withrvs_copl[:,1+6],bins=[50,20],range=[[0,80],[0,0.4]])
    sma_ecc_norvs_copl_c,aedges,eedges = np.histogram2d(post_norv_copl[:,0+6],post_norv_copl[:,1+6],bins=[50,20],range=[[0,80],[0,0.4]])
    sma_ecc_withrvs_copl_d,aedges,eedges = np.histogram2d(post_withrvs_copl[:,0+12],post_withrvs_copl[:,1+12],bins=[50,20],range=[[0,80],[0,0.4]])
    sma_ecc_norvs_copl_d,aedges,eedges = np.histogram2d(post_norv_copl[:,0+12],post_norv_copl[:,1+12],bins=[50,20],range=[[0,80],[0,0.4]])
    # print(post_norv_copl[:,0],post_norv_copl[:,1])
    # exit()
    sma_ecc_withrvs_b,aedges,eedges = np.histogram2d(post_withrvs[:,0],post_withrvs[:,1],bins=[50,20],range=[[0,80],[0,0.4]])
    sma_ecc_norvs_b,aedges,eedges = np.histogram2d(post_b_norv[:,0],post_b_norv[:,1],bins=[50,20],range=[[0,80],[0,0.4]])
    sma_ecc_withrvs_c,aedges,eedges = np.histogram2d(post_withrvs[:,0+6],post_withrvs[:,1+6],bins=[50,20],range=[[0,80],[0,0.4]])
    sma_ecc_norvs_c,aedges,eedges = np.histogram2d(post_c_norv[:,0],post_c_norv[:,1],bins=[50,20],range=[[0,80],[0,0.4]])
    sma_ecc_withrvs_d,aedges,eedges = np.histogram2d(post_withrvs[:,0+12],post_withrvs[:,1+12],bins=[50,20],range=[[0,80],[0,0.4]])
    sma_ecc_norvs_d,aedges,eedges = np.histogram2d(post_d_norv[:,0],post_d_norv[:,1],bins=[50,20],range=[[0,80],[0,0.4]])


    if 0: #corner
        param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","sma3","ecc3","inc3","aop3","pan3","epp3","plx","sysrv","mtot"]
        # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","plx","sysrv","mtot"]
        # corner_plot_fig = loaded_results_withrvs.plot_corner(param_list=param_list)
        # corner_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"corner_plot_withrvs_{0}_{1}.png".format(planet,suffix_withrvs)))

        import corner
        corner_kwargs = {}
        corner_kwargs['labels'] = ["inc1","pan1","inc2","pan2","inc3","pan3","imbc","imcd","imbd"]
        post4corner = post_withrvs[:,[2,4,2+6,4+6,2+12,4+12]]
        post4corner = np.concatenate([np.rad2deg(post4corner),im_bc_samples[:,None],im_cd_samples[:,None],im_bd_samples[:,None]],axis=1)
        corner_plot_fig = corner.corner(post4corner, **corner_kwargs)
        corner_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"corner_mutualincl_withrvs_{0}_{1}.png".format(planet,suffix_withrvs)))
        # plt.show()
        exit()


    fig = plt.figure(30)
    plt.plot(imutual_centers,im_bc_post/np.nansum(im_bc_post),label="b-c w/ rv")
    plt.plot(imutual_centers,im_cd_post/np.nansum(im_cd_post),label="c-d w/ rv")
    plt.plot(imutual_centers,im_bd_post/np.nansum(im_bd_post),label="b-d w/ rv")
    plt.plot(imutual_centers,im_bc_norv_post/np.nansum(im_bc_norv_post),label="b-c w/o rv",linestyle="--")
    plt.plot(imutual_centers,im_cd_norv_post/np.nansum(im_cd_norv_post),label="c-d w/o rv",linestyle="--")
    plt.plot(imutual_centers,im_bd_norv_post/np.nansum(im_bd_norv_post),label="b-d w/o rv",linestyle="--")
    plt.xlabel("Mutual inclination (deg)",fontsize=fontsize)
    # plt.ylabel(r"Eccentricity",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.legend(loc="upper left",frameon=True,fontsize=fontsize)
    plt.tight_layout()
    fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'mutual_incl.pdf'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
    fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'mutual_incl.png'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
    # plt.show()


    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    y_centers = [(y1+y2)/2. for y1,y2 in zip(yedges[0:len(yedges)-1],yedges[1:len(yedges)])]
    a_centers = [(x1+x2)/2. for x1,x2 in zip(aedges[0:len(aedges)-1],aedges[1:len(aedges)])]
    e_centers = [(y1+y2)/2. for y1,y2 in zip(eedges[0:len(eedges)-1],eedges[1:len(eedges)])]

    #
    fig = plt.figure(2,figsize=(5,3))
    hist_list = [inc_Ome_norv_b,inc_Ome_norv_c,inc_Ome_norv_d,inc_Ome_withrvs_b,inc_Ome_withrvs_c,inc_Ome_withrvs_d]
    label_list = ["b w/o rv","c w/o rv","d w/o rv","b w/ rv","c w/ rv","d w/ rv"]
    linestyle_list = ["-","--",":","-","--",":"]
    linewidth_list = [1,1,1,3,3,3]
    alpha_list = [0.5,0.5,0.5,1,1,1]
    colors=["#006699","#ff9900","#6600ff","#006699","#ff9900","#6600ff"]
    for inc_Ome_hist,mylabel,ls,mycolor,lw,alpha in zip(hist_list,label_list,linestyle_list,colors,linewidth_list,alpha_list):
        inc_Ome_hist_T = inc_Ome_hist.T
        ravel_H = np.ravel(inc_Ome_hist_T)
        print(ravel_H.shape)
        ind = np.argsort(ravel_H)
        cum_ravel_H = np.zeros(np.shape(ravel_H))
        cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
        cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inc_Ome_hist_T))
        cum_H.shape = inc_Ome_hist_T.shape
        image = copy(inc_Ome_hist_T)
        image[np.where(cum_H>0.9545)] = np.nan

        # plt.imshow(image,origin ="lower",
        #            extent=[Ome_bounds[0],Ome_bounds[1],inc_bounds[0],inc_bounds[1]],
        #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
        plt.xlim(Ome_bounds)
        plt.ylim(inc_bounds)
        levels = [0.6827]
        xx,yy = np.meshgrid(x_centers,y_centers)
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles=ls,linewidths=[lw],colors=(mycolor,),zorder=15,label=mylabel,alpha=alpha)
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)
        plt.plot([-1,-2],[-1,-2],linestyle=ls,linewidth=lw,color=mycolor,label=mylabel)
    plt.xlabel(r"Longitude of Ascending Node (deg)",fontsize=fontsize)
    plt.ylabel(r"Inclination (deg)",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
    plt.tight_layout()


    plt.show()
