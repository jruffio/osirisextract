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

osiris_data_dir = "/data/osiris_data"
# osiris_data_dir = "/scr3/jruffio/data/osiris_data"
astrometry_DATADIR = os.path.join(osiris_data_dir,"astrometry")
#out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
out_pngs = os.path.join(osiris_data_dir,"astrometry","figures")
sysrv=-12.6
sysrv_err=1.4
fontsize = 12
planet = "bcd"




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
    suffix_withrvs = "from_scratch_it1_16_512_100000_50_True"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs))) as hdulist:
        chains_withrvs = hdulist[0].data
        print(chains_withrvs.shape)
        chains_withrvs = chains_withrvs[0,:,chains_withrvs.shape[2]-100::,:]
        # chains_withrvs = chains_withrvs[0,:,0:25,:]
        # chains_withrvs = chains_withrvs[0,:,:,:]
        print(chains_withrvs.shape)
        post_withrvs = np.reshape(chains_withrvs,(chains_withrvs.shape[0]*chains_withrvs.shape[1],chains_withrvs.shape[2]))
    suffix_withrvs_copl =  "it7_16_512_100000_50_True_coplanar"
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs_copl))) as hdulist:
        chains_withrvs_copl = hdulist[0].data
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

    cmaps = ["cool","hot"]
    # inc_Ome_norv_b,xedges,yedges = np.histogram2d(np.rad2deg(np.concatenate([post_norv[:,4],post_norv[:,4]+np.pi])),np.rad2deg(np.concatenate([post_norv[:,2],post_norv[:,2]])),bins=[360/4,45//2],range=[Ome_bounds,inc_bounds])
    # inc_Ome_norv_b,xedges,yedges = np.histogram2d(np.rad2deg(np.concatenate([post_b_norv[:,4],np.mod(post_b_norv[:,4]+np.pi,2*np.pi)])),np.rad2deg(np.concatenate([post_b_norv[:,2],post_b_norv[:,2]])),#np.rad2deg(post_b_norv[:,4]),np.rad2deg(post_b_norv[:,2]),
    #                                               bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    # inc_Ome_norv_c,xedges,yedges = np.histogram2d(np.rad2deg(np.concatenate([post_c_norv[:,4],np.mod(post_c_norv[:,4]+np.pi,2*np.pi)])),np.rad2deg(np.concatenate([post_c_norv[:,2],post_c_norv[:,2]])),#np.rad2deg(post_c_norv[:,4]),np.rad2deg(post_c_norv[:,2]),
    #                                               bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    # inc_Ome_norv_d,xedges,yedges = np.histogram2d(np.rad2deg(np.concatenate([post_d_norv[:,4],np.mod(post_d_norv[:,4]+np.pi,2*np.pi)])),np.rad2deg(np.concatenate([post_d_norv[:,2],post_d_norv[:,2]])),#np.rad2deg(post_d_norv[:,4]),np.rad2deg(post_d_norv[:,2]),
    #                                               bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_norv_b,xedges,yedges = np.histogram2d(np.rad2deg(post_b_norv[:,4]),np.rad2deg(post_b_norv[:,2]),
                                                  bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_norv_c,xedges,yedges = np.histogram2d(np.rad2deg(post_c_norv[:,4]),np.rad2deg(post_c_norv[:,2]),
                                                  bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
    inc_Ome_norv_d,xedges,yedges = np.histogram2d(np.rad2deg(post_d_norv[:,4]),np.rad2deg(post_d_norv[:,2]),
                                                  bins=Ome_inc_bins,range=[Ome_bounds,inc_bounds])
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

    x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    y_centers = [(y1+y2)/2. for y1,y2 in zip(yedges[0:len(yedges)-1],yedges[1:len(yedges)])]
    a_centers = [(x1+x2)/2. for x1,x2 in zip(aedges[0:len(aedges)-1],aedges[1:len(aedges)])]
    e_centers = [(y1+y2)/2. for y1,y2 in zip(eedges[0:len(eedges)-1],eedges[1:len(eedges)])]

    fig = plt.figure(11,figsize=(6,4))
    hist_list = [sma_ecc_withrvs_copl_b,sma_ecc_withrvs_copl_c,sma_ecc_withrvs_copl_d,sma_ecc_norvs_copl_b,sma_ecc_norvs_copl_c,sma_ecc_norvs_copl_d]
    label_list = ["b w/ rv","c w/ rv","d w/ rv","b w/o rv","c w/o rv","d w/o rv"]
    linestyle_list = ["-","--",":","-","--",":"]
    linewidth_list = [3,3,3,1,1,1]
    colors=["#006699","#ff9900","#6600ff","#006699","#ff9900","#6600ff"]
    for inc_Ome_hist,mylabel,ls,mycolor,lw in zip(hist_list,label_list,linestyle_list,colors,linewidth_list):
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
        plt.xlim([0,80])
        plt.ylim([0,0.4])
        levels = [0.6827]
        xx,yy = np.meshgrid(a_centers,e_centers)
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles=ls,linewidths=[lw],colors=(mycolor,),zorder=15,label=mylabel)
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)
        plt.plot([-1,-2],[-1,-2],linestyle=ls,linewidth=lw,color=mycolor,label=mylabel)

    plt.xlabel(r"Semi-Major Axis (au)",fontsize=fontsize)
    plt.ylabel(r"Eccentricity",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
    plt.tight_layout()
    fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'sma_vs_ecc_coplanar.pdf'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
    fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'sma_vs_ecc_coplanar.png'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

    fig = plt.figure(1,figsize=(6,4))
    hist_list = [inc_Ome_withrvs_copl,inc_Ome_norv_copl]
    label_list = ["coplanar w/ rv","coplanar w/o rv"]
    linestyle_list = ["-","-."]
    linewidth_list = [3,1]
    colors=["grey","black"]
    for inc_Ome_hist,mylabel,ls,mycolor,lw in zip(hist_list,label_list,linestyle_list,colors,linewidth_list):
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
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles=ls,linewidths=[lw],colors=(mycolor,),zorder=15,label=mylabel)
        levels = [0.954]
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles=ls,linewidths=[lw/2.],colors=(mycolor,),zorder=15,label=mylabel)
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)
        plt.plot([-1,-2],[-1,-2],linestyle=ls,linewidth=lw,color=mycolor,label=mylabel)

    plt.xlabel(r"Longitude of Ascending Node (deg)",fontsize=fontsize)
    plt.ylabel(r"Inclination (deg)",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
    plt.tight_layout()
    plt.xlim([0,360])
    plt.ylim([10,40])
    fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'ome_vs_inc_coplanar.pdf'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
    fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'ome_vs_inc_coplanar.png'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

    #
    fig = plt.figure(2,figsize=(6,4))
    hist_list = [inc_Ome_norv_b,inc_Ome_norv_c,inc_Ome_norv_d,inc_Ome_withrvs_b,inc_Ome_withrvs_c,inc_Ome_withrvs_d]
    label_list = ["b w/o rv","c w/o rv","d w/o rv","b w/ rv","c w/ rv","d w/ rv"]
    linestyle_list = ["-","--",":","-","--",":"]
    linewidth_list = [1,1,1,3,3,3]
    colors=["#006699","#ff9900","#6600ff","#006699","#ff9900","#6600ff"]
    for inc_Ome_hist,mylabel,ls,mycolor,lw in zip(hist_list,label_list,linestyle_list,colors,linewidth_list):
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
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles=ls,linewidths=[lw],colors=(mycolor,),zorder=15,label=mylabel)
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)
        plt.plot([-1,-2],[-1,-2],linestyle=ls,linewidth=lw,color=mycolor,label=mylabel)
    plt.xlabel(r"Longitude of Ascending Node (deg)",fontsize=fontsize)
    plt.ylabel(r"Inclination (deg)",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
    plt.tight_layout()

    # fig = plt.figure(2,figsize=(6,4))
    # hist_list = [inc_Ome_norv_b,inc_Ome_norv_c,inc_Ome_norv_d,inc_Ome_rv_Rob_b,inc_Ome_rv_Rob_c,inc_Ome_rv_Rob_d,]
    # label_list = ["b w/o rv","c w/o rv","d w/o rv","b w/ rv (Rob)","c w/ rv (Rob)","d w/ rv (Rob)"]
    # linestyle_list = ["-","--",":","-","--",":"]
    # linewidth_list = [1,1,1,3,3,3]
    # colors=["#006699","#ff9900","#6600ff","#006699","#ff9900","#6600ff"]
    # for inc_Ome_hist,mylabel,ls,mycolor,lw in zip(hist_list,label_list,linestyle_list,colors,linewidth_list):
    #     inc_Ome_hist_T = inc_Ome_hist.T
    #     ravel_H = np.ravel(inc_Ome_hist_T)
    #     print(ravel_H.shape)
    #     ind = np.argsort(ravel_H)
    #     cum_ravel_H = np.zeros(np.shape(ravel_H))
    #     cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
    #     cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inc_Ome_hist_T))
    #     cum_H.shape = inc_Ome_hist_T.shape
    #     image = copy(inc_Ome_hist_T)
    #     image[np.where(cum_H>0.9545)] = np.nan
    #
    #     # plt.imshow(image,origin ="lower",
    #     #            extent=[Ome_bounds[0],Ome_bounds[1],inc_bounds[0],inc_bounds[1]],
    #     #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
    #     plt.xlim(Ome_bounds)
    #     plt.ylim(inc_bounds)
    #     levels = [0.6827]
    #     xx,yy = np.meshgrid(x_centers,y_centers)
    #     CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles=ls,linewidths=[lw],colors=(mycolor,),zorder=15,label=mylabel)
    #     # levels = [0.9545,0.9973]
    #     # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)
    #     plt.plot([-1,-2],[-1,-2],linestyle=ls,linewidth=lw,color=mycolor,label=mylabel)
    # plt.xlabel(r"Longitude of Ascending Node (deg)",fontsize=fontsize)
    # plt.ylabel(r"Inclination (deg)",fontsize=fontsize)
    # plt.gca().tick_params(axis='x', labelsize=fontsize)
    # plt.gca().tick_params(axis='y', labelsize=fontsize)
    # plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
    # plt.tight_layout()

    fig = plt.figure(3,figsize=(6,4))
    hist_list = [inc_Ome_rv_Rob_b,inc_Ome_rv_Rob_c,inc_Ome_rv_Rob_d,inc_Ome_withrvs_b,inc_Ome_withrvs_c,inc_Ome_withrvs_d]
    label_list = ["b w/ rv Rob","c w/ rv Rob","d w/ rv Rob","b w/ rv","c w/ rv","d w/ rv"]
    linestyle_list = ["-","--",":","-","--",":"]
    linewidth_list = [1,1,1,3,3,3]
    colors=["#006699","#ff9900","#6600ff","#006699","#ff9900","#6600ff"]
    for inc_Ome_hist,mylabel,ls,mycolor,lw in zip(hist_list,label_list,linestyle_list,colors,linewidth_list):
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
        CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles=ls,linewidths=[lw],colors=(mycolor,),zorder=15,label=mylabel)
        # levels = [0.9545,0.9973]
        # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)
        plt.plot([-1,-2],[-1,-2],linestyle=ls,linewidth=lw,color=mycolor,label=mylabel)
    plt.xlabel(r"Longitude of Ascending Node (deg)",fontsize=fontsize)
    plt.ylabel(r"Inclination (deg)",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
    plt.tight_layout()



    colors=["#006699","#ff9900","#6600ff","#006699","#ff9900","#6600ff"]
    param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","sma3","ecc3","inc3","aop3","pan3","epp3","plx","sysrv","mtot"]
    xlabel_list = ["a (au)","e","i (deg)",r"$\omega$ (deg)",r"$\Omega$ (deg)",r"$\tau$",
                   "a (au)","e","i (deg)",r"$\omega$ (deg)",r"$\Omega$ (deg)",r"$\tau$",
                   "a (au)","e","i (deg)",r"$\omega$ (deg)",r"$\Omega$ (deg)",r"$\tau$",
                   "Paral. (mas)","$\mathrm{RV}_{\mathrm{sys}}$(km/s)",r"$M_{\mathrm{tot}}$ ($M_{\mathrm{Sun}}$)"]
    color_list = ["#006699","#006699","black","#006699","black","#006699",
                  "#ff9900","#ff9900","black","#ff9900","black","#ff9900",
                  "#6600ff","#6600ff","black","#6600ff","black","#6600ff",
                  "black","black","black"]
    axis_list = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,7,8,9]
    bins_list = [50,50,50,50,200,50,
                 50,50,50,50,200,50,
                 50,50,50,50,200,50,
                 50,50,50]
    xticks_list = [[25,50],[0,0.2],[0,30],[0,180],[0,180],[0,0.5],
                 [25,50],[0,0.2],[0,30],[0,180],[0,180],[0,0.5],
                 [25,50],[0,0.2],[0,30],[0,180],[0,180],[0,0.5],
                 [24,24.5],[-12,-10],[1.5,1.8]]
    planet_list = ["b","b","b","b","b","b",
                 "c","c","c","c","c","c",
                 "d","d","d","d","d","d",
                 "","",""]
    fig = plt.figure(20,figsize=(12,2))
    for paraid, (param, chain,color,axis,pl,xlabel,xticks) in enumerate(zip(param_list, chains_withrvs_copl.T,color_list,axis_list,planet_list,xlabel_list,xticks_list)):
        if pl == "b":
            ls = "-"
        elif pl == "c":
            ls = "--"
        elif pl == "d":
            ls = ":"
        else:
            ls = "-"

        plt.subplot(1,9,axis)
        # print(param, np.nanmedian(chain), np.nanstd(chain))
        if "inc" in param or "aop" in param or "pan" in param:
            chain = np.rad2deg(chain)
        post, xedges = np.histogram(chain, bins=100, range=[np.min(chain), np.max(chain)],density=True)
        post /= np.max(post)
        x_centers = [(x1 + x2) / 2. for x1, x2 in zip(xedges[0:len(xedges) - 1], xedges[1:len(xedges)])]
        mode,_,_,merr,perr,_ = get_err_from_posterior(x_centers, post)
        print(param,mode,merr,perr,get_upperlim_from_posterior(x_centers, post))
        plt.plot(x_centers, post,color=color,label=pl,linestyle=ls)
        plt.ylim([0,1.1])
        plt.xlabel(xlabel,fontsize=fontsize)
        ax0 = plt.gca()
        ax0.tick_params(axis='x', labelsize=fontsize)
        ax0.tick_params(axis='y', labelsize=fontsize)
        plt.xticks(xticks)
        plt.yticks([])

    plt.subplot(1,9,1)
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
    plt.subplot(1,9,1)
    plt.yticks([0,0.5,1])
    plt.ylabel("Posterior",fontsize=fontsize)

    fig.subplots_adjust(wspace=0,hspace=0)
    # plt.tight_layout()
    fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'all_paras_coplanar.pdf'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
    fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'all_paras_coplanar.png'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
    # plt.show()
    # exit()




    plt.show()


if 1: # v2 PLot orbits
    # mu = 5
    # x = np.linspace(-7+mu,7+mu,1400)
    # post = np.exp(-0.5*(x-mu)**2/1.5**2)
    # print(get_err_from_posterior(x,post))
    # exit()

    # system parameters
    num_secondary_bodies = len(planet)
    system_mass = 1.47 # [Msol]
    plx = None#25.38 # [mas]
    mass_err = 0.3#0.3 # [Msol]
    plx_err = None#0.7#0.7 # [mas]
    # suffix_withrvs =  "sherlock_restrictOme_16_1024_200000_50_True_coplanar"
    # suffix_norv = "sherlock_restrictOme_16_1024_200000_50_False_coplanar"
    # suffix_withrvs1 =  "test_bcd_16_100_1000_2_True"
    # suffix_withrvs2 =  "it1_16_512_10000_50_True"
    # suffix_withrvs1 =  "it1_16_512_10000_50_True"
    # suffix_withrvs2 =  "it2_16_512_100000_50_True"
    suffix_withrvs1 =  "it7_16_512_100000_50_True_coplanar"
    suffix_withrvs2 =  "it7_16_512_100000_50_True_coplanar"
    # suffix_withrvs1 =  "it8_16_512_100000_50_True"
    # suffix_withrvs2 =  "it8_16_512_100000_50_True"

    filename = "{0}/HR8799{1}_rvs.csv".format(astrometry_DATADIR,planet)
    data_table_withrvs = orbitize.read_input.read_file(filename)
    filename = "{0}/HR8799{1}.csv".format(astrometry_DATADIR,planet)
    data_table_norv = orbitize.read_input.read_file(filename)

    hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format("withrvs",planet,suffix_withrvs1))
    print(hdf5_filename)
    loaded_results_withrvs = results.Results() # Create blank results object for loading
    loaded_results_withrvs.load_results(hdf5_filename)

    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs2))) as hdulist:
        myshape = hdulist[0].data.shape
        print(myshape)
        # exit()
        chains_withrvs = hdulist[0].data
        chains_withrvs = chains_withrvs[0,:,chains_withrvs.shape[2]-25::,:]
        if chains_withrvs.shape[2] == 21-4:
            _chains_withrvs = np.zeros((chains_withrvs.shape[0],chains_withrvs.shape[1],chains_withrvs.shape[2]+4))
            a_list = [0,1,2,3,4,5, 6,7,2,8,4,9, 10,11,2,12,4,13, 14,15,16]
            b_list = np.arange(21)
            for a,b in zip(a_list,b_list):
                _chains_withrvs[:,:,b] = chains_withrvs[:,:,a]
            chains_withrvs =_chains_withrvs
    print(chains_withrvs.shape)
    sysrv_med = np.median(chains_withrvs[:,:,-2],axis=(0,1))
    sysrv_err = np.std(chains_withrvs[:,:,-2],axis=(0,1))

    if 1:
        # Create figure for orbit plots
        fig = plt.figure(figsize=(12,5.5*10./6.*13./10.))
        post = np.reshape(chains_withrvs,(chains_withrvs.shape[0]*chains_withrvs.shape[1],chains_withrvs.shape[2]))

        num_orbits_to_plot= 50 # Will plot 100 randomly selected orbits of this companion
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

        # ax0 = plt.subplot2grid((6, 14), (0, 0), rowspan=6, colspan=6)
        # ax11 = plt.subplot2grid((6, 14), (0, 9), colspan=6)
        # ax21 = plt.subplot2grid((6, 14), (2, 9), colspan=6)
        # ax12 = plt.subplot2grid((6, 14), (1, 9), colspan=6)
        # ax22 = plt.subplot2grid((6, 14), (3, 9), colspan=6)
        # ax3 = plt.subplot2grid((6, 14), (4, 9), rowspan=2, colspan=6)


        ax0 = plt.subplot2grid((13, 14), (0, 0), rowspan=6, colspan=6)
        ax11 = plt.subplot2grid((13, 14), (0, 9), colspan=6)
        ax12 = plt.subplot2grid((13, 14), (1, 9), colspan=6)
        ax13 = plt.subplot2grid((13, 14), (2, 9), colspan=6)
        ax21 = plt.subplot2grid((13, 14), (3, 9), colspan=6)
        ax22 = plt.subplot2grid((13, 14), (4, 9), colspan=6)
        ax23 = plt.subplot2grid((13, 14), (5, 9), colspan=6)
        ax31 = plt.subplot2grid((13, 14), (7, 0), rowspan=2, colspan=14)
        ax32 = plt.subplot2grid((13, 14), (9, 0), rowspan=2, colspan=14)
        ax33 = plt.subplot2grid((13, 14), (11, 0), rowspan=2, colspan=14)

        # cmap = mpl.cm.Purples_r
        # exit()
        color_list = ["#006699","#ff9900","#6600ff"]
        for object_to_plot,cmap,pl_linestyle,pl_color,ax1,ax2,ax3 in zip([1,2,3],
                                                        [mpl.cm.Blues_r,mpl.cm.Oranges_r,mpl.cm.Purples_r],
                                                        ["-", "--", ":"],
                                                        color_list,
                                                        [ax11,ax12,ax13],[ax21,ax22,ax23],[ax31,ax32,ax33]): # Plot orbits for the first (and only, in this case) companion
            # print(object_to_plot,cmap,pl_color)
            # print(data_table)

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
                    cbar_ax = fig.add_axes([0.48, 0.52, 0.015, 0.37/3]) # xpos, ypos, width, height, in fraction of figure size
                    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm_yr, orientation='vertical')
                    cbar.set_label("b: (yr)",fontsize=fontsize)
                if object_to_plot == 2:
                    cbar_ax = fig.add_axes([0.48, 0.52+0.37/3, 0.015, 0.37/3]) # xpos, ypos, width, height, in fraction of figure size
                    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm_yr, orientation='vertical')
                    cbar.set_label("c: (yr)",fontsize=fontsize)
                if object_to_plot == 3:
                    cbar_ax = fig.add_axes([0.48, 0.52+2*0.37/3, 0.015,  0.37/3]) # xpos, ypos, width, height, in fraction of figure size
                    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm_yr, orientation='vertical')
                    cbar.set_label("d: "+"Calendar year",fontsize=fontsize)
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
                        segments, cmap=cmap, norm=norm, linewidth=1.0,alpha=0.2
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
                    eb1[-1][0].set_linestyle(pl_linestyle)
                    plt.xticks([],[])
                    plt.sca(ax2)
                    eb2 = plt.errorbar(Time(data_table["epoch"][seppa_indices],format='mjd').decimalyear,
                                 data_table["quant2"][seppa_indices],
                                 yerr=data_table["quant2_err"][seppa_indices],fmt="x",color=pl_color,linestyle="",zorder=10)
                    eb2[-1][0].set_linestyle(pl_linestyle)
                    plt.xticks([],[])
                    for _ax3 in [ax31,ax32,ax33]:
                        plt.sca(_ax3)
                        # print(np.array(data_table["quant1"][rv_indices]))
                        # exit()
                        eb3 = plt.errorbar(Time(data_table["epoch"][rv_indices],format='mjd').decimalyear,
                                     np.array(data_table["quant1"][rv_indices]),
                                     yerr=np.array(data_table["quant1_err"][rv_indices]),fmt="x",color=pl_color,linestyle="",zorder=10)
                        eb3[-1][0].set_linestyle(pl_linestyle)
                    #
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
                    eb = plt.errorbar(Time(data_table["epoch"][radec_indices],format='mjd').decimalyear,
                                 sep_list,
                                 yerr=[-sep_merr_list,sep_perr_list],fmt="x",color=pl_color,linestyle="",zorder=10)
                    eb[-1][0].set_linestyle(pl_linestyle)
                    plt.sca(ax2)
                    eb = plt.errorbar(Time(data_table["epoch"][radec_indices],format='mjd').decimalyear,
                                 pa_list,
                                 yerr=[-pa_merr_list,pa_perr_list],fmt="x",color=pl_color,linestyle="",zorder=10)
                    eb[-1][0].set_linestyle(pl_linestyle)
        # plt.show()
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

        plt.sca(ax13)
        plt.ylim([600,750])
        ax13.set_ylabel('$\\rho_d$ (mas)',fontsize=fontsize)
        plt.yticks([600,650,700,750])
        ax13.tick_params(axis='x', labelsize=fontsize)
        ax13.tick_params(axis='y', labelsize=fontsize)

        plt.sca(ax21)
        plt.ylim([55,70])
        ax21.set_ylabel('PA$_b$ (deg)',fontsize=fontsize)
        plt.yticks([55,65,75])
        ax21.yaxis.tick_right()
        ax21.yaxis.set_label_position("right")
        ax21.tick_params(axis='x', labelsize=fontsize)
        ax21.tick_params(axis='y', labelsize=fontsize)

        plt.sca(ax22)
        plt.ylim([300,340])
        ax22.set_ylabel('PA$_c$ (deg)',fontsize=fontsize)
        plt.yticks([300,320,340])
        ax22.tick_params(axis='x', labelsize=fontsize)
        ax22.tick_params(axis='y', labelsize=fontsize)

        plt.sca(ax23)
        plt.ylim([180,260])
        ax23.set_ylabel('PA$_d$ (deg)',fontsize=fontsize)
        plt.yticks([180,220,260])
        ax23.yaxis.tick_right()
        ax23.yaxis.set_label_position("right")
        ax23.set_xlabel('Epoch (yr)',fontsize=fontsize)
        plt.xticks([2005,2010,2015,2020,2025],[2005,2010,2015,2020,2025])
        ax23.tick_params(axis='x', labelsize=fontsize)
        ax23.tick_params(axis='y', labelsize=fontsize)

        plt.sca(ax31)
        plt.fill_between([2000,2050],np.zeros(2)+sysrv_med+sysrv_err,np.zeros(2)+sysrv_med-sysrv_err,facecolor="none",edgecolor="black",alpha=1,hatch="\\",label="System RV")
        #alpha=0.4,facecolor="none",edgecolor="#006699",label="Wang et al. 2018 ($RV_b-RV_c$)",hatch="\\"
        plt.gca().annotate("HR 8799 " + "b", xy=(2005,-1), va="top", ha="left", fontsize=fontsize, color=color_list[0])
        plt.xticks([],[])
        plt.ylim([-20,0])
        plt.yticks([-15,-10,-5,0])
        ax31.tick_params(axis='y', labelsize=fontsize)
        plt.sca(ax32)
        plt.fill_between([2000,2050],np.zeros(2)+sysrv_med+sysrv_err,np.zeros(2)+sysrv_med-sysrv_err,facecolor="none",edgecolor="black",alpha=1,hatch="\\",label="System RV")
        plt.gca().annotate("HR 8799 " + "c", xy=(2005,-1), va="top", ha="left", fontsize=fontsize, color=color_list[1])
        plt.xticks([],[])
        plt.ylim([-20,0])
        plt.yticks([-15,-10,-5,0])
        ax32.tick_params(axis='y', labelsize=fontsize)
        plt.sca(ax33)
        plt.fill_between([2000,2050],np.zeros(2)+sysrv_med+sysrv_err,np.zeros(2)+sysrv_med-sysrv_err,facecolor="none",edgecolor="black",alpha=1,hatch="\\",label="System RV")
        plt.legend(loc="upper right",frameon=True,fontsize=12)#
        plt.gca().annotate("HR 8799 " + "d", xy=(2005,-1), va="top", ha="left", fontsize=fontsize, color=color_list[2])
        plt.ylim([-20,0])
        plt.yticks([-20,-15,-10,-5,0])
        ax33.set_ylabel('RV (km/s)',fontsize=fontsize)
        ax33.set_xlabel('Epoch (yr)',fontsize=fontsize)
        ax33.tick_params(axis='x', labelsize=fontsize)
        ax33.tick_params(axis='y', labelsize=fontsize)

        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_bcd_withrvs.pdf'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_bcd_withrvs.png'),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        plt.show()
    exit()

    # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("norv",planet,suffix_norv))) as hdulist:
    #     myshape = hdulist[0].data.shape
    #     print(myshape)
    #     if myshape[3] == 14:
    #         chains_norv = hdulist[0].data
    #     else:
    #         chains_norv = np.zeros((myshape[0],myshape[1],myshape[2],14))
    #         chains_norv[:,:,:,0:(2+6)] = hdulist[0].data[:,:,:,0:(2+6)]
    #         chains_norv[:,:,:,3+6] = hdulist[0].data[:,:,:,2+6]
    #         chains_norv[:,:,:,(5+6)::] = hdulist[0].data[:,:,:,(3+6)::]
    #     if 0:
    #         choose = np.random.randint(0, high=chains_norv.shape[2], size=chains_norv.shape[2]//2)
    #         chains_norv[:,:,choose,4] -= np.pi
    #         chains_norv[:,:,choose,4] = np.mod(chains_norv[:,:,choose,4],2*np.pi)
    #         chains_norv[:,:,choose,3] -= np.pi
    #         chains_norv[:,:,choose,3+6] -= np.pi
    #         chains_norv[:,:,choose,3] = np.mod(chains_norv[:,:,choose,3],2*np.pi)
    #         chains_norv[:,:,choose,3+6] = np.mod(chains_norv[:,:,choose,3+6],2*np.pi)
    #     if not (myshape[3] == 14):
    #         chains_norv[:,:,:,2+6] = chains_norv[:,:,:,2]
    #         chains_norv[:,:,:,4+6] = chains_norv[:,:,:,4]
    #     print(chains_norv.shape)
    #     chains_norv = chains_norv[0,:,2*chains_norv.shape[2]//4::,:]
    #     # chains_norv = chains_norv[0,:,:,:]
    #     # chains_norv = chains_norv[0,:,0:10,:]
    #     # chains_norv = chains_norv[0,:,0:1,:]
    # print(chains_norv.shape)
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs))) as hdulist:
        myshape = hdulist[0].data.shape
        chains_withrvs = hdulist[0].data
        # print(myshape)
        # if myshape[3] == 15:
        #     chains_withrvs = hdulist[0].data
        # else:
        #     chains_withrvs = np.zeros((myshape[0],myshape[1],myshape[2],myshape[3]+2))
        #     chains_withrvs[:,:,:,0:(2+6)] = hdulist[0].data[:,:,:,0:(2+6)]
        #     chains_withrvs[:,:,:,3+6] = hdulist[0].data[:,:,:,2+6]
        #     chains_withrvs[:,:,:,(5+6)::] = hdulist[0].data[:,:,:,(3+6)::]
        # if not (myshape[3] == 15):
        #     chains_withrvs[:,:,:,2+6] = chains_withrvs[:,:,:,2]
        #     chains_withrvs[:,:,:,4+6] = chains_withrvs[:,:,:,4]
        # print(chains_withrvs.shape)
        # chains_withrvs = chains_withrvs[0,:,2*chains_withrvs.shape[2]//4::,:]
        # # chains_withrvs = chains_withrvs[0,:,:,:]
        # # chains_withrvs = chains_withrvs[0,:,0:10,:]
        # # chains_withrvs = chains_withrvs[0,:,0:1,:]
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



    # inc_post,xedges = np.histogram(np.rad2deg(inc_chain),bins=2*60,range=[0,60])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # inc_mod, _,_,inc_merr, inc_perr,_ = get_err_from_posterior(x_centers,inc_post)
    #
    # ome_post,xedges = np.histogram(np.rad2deg(ome_chain),bins=2*180,range=[0,180])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # ome_mod, _,_,ome_merr, ome_perr,_ = get_err_from_posterior(x_centers,ome_post)
    #
    # sysrv_post,xedges = np.histogram(sysrv_chain,bins=10*20,range=[-20,0])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # sysrv_mod, _,_,sysrv_merr, sysrv_perr,_ = get_err_from_posterior(x_centers,sysrv_post)
    # print("inclination: {0},{1},{2}".format(inc_mod,inc_merr,inc_perr))
    # print("Omega: {0},{1},{2}".format(ome_mod,ome_merr,ome_perr))
    # print("sysrv: {0},{1},{2}".format(sysrv_mod,sysrv_merr,sysrv_perr))
    # # exit()
    #
    # post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,0]),bins=200,range=[np.min(chains_withrvs[:,:,0]),np.max(chains_withrvs[:,:,0])])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    # print("a_b : {0},{1},{2}".format(mod,merr,perr))
    # post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,1]),bins=2*60,range=[np.min(chains_withrvs[:,:,1]),np.max(chains_withrvs[:,:,1])])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # upplim = get_upperlim_from_posterior(x_centers,post)
    # print("e_b : uplim={0}".format(upplim))
    # tmp = np.rad2deg(np.ravel(chains_withrvs[:,:,3]))
    # post,xedges = np.histogram(tmp,bins=2*60,range=[np.min(tmp),np.max(tmp)])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    # print("omega_b : {0},{1},{2}".format(mod,merr,perr))
    # post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,5]),bins=2*60,range=[np.min(chains_withrvs[:,:,5]),np.max(chains_withrvs[:,:,5])])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    # print("tau_b : {0},{1},{2}".format(mod,merr,perr))
    #
    # post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,0+6]),bins=2*60,range=[np.min(chains_withrvs[:,:,0+6]),np.max(chains_withrvs[:,:,0+6])])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    # print("a_c : {0},{1},{2}".format(mod,merr,perr))
    # post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,1+6]),bins=2*60,range=[np.min(chains_withrvs[:,:,1+6]),np.max(chains_withrvs[:,:,1+6])])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # upplim = get_upperlim_from_posterior(x_centers,post)
    # print("e_c : uplim={0}".format(upplim))
    # tmp = np.rad2deg(np.ravel(chains_withrvs[:,:,3+6]))
    # post,xedges = np.histogram(tmp,bins=2*60,range=[np.min(tmp),np.max(tmp)])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    # print("omega_c : {0},{1},{2}".format(mod,merr,perr))
    # post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,5+6]),bins=2*60,range=[np.min(chains_withrvs[:,:,5+6]),np.max(chains_withrvs[:,:,5+6])])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    # print("tau_c : {0},{1},{2}".format(mod,merr,perr))
    #
    # post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,-3]),bins=2*60,range=[np.min(chains_withrvs[:,:,-3]),np.max(chains_withrvs[:,:,-3])])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    # print("plx : {0},{1},{2}".format(mod,merr,perr))
    # post,xedges = np.histogram(np.ravel(chains_withrvs[:,:,-1]),bins=2*60,range=[np.min(chains_withrvs[:,:,-1]),np.max(chains_withrvs[:,:,-1])])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # mod, _,_,merr, perr,_ = get_err_from_posterior(x_centers,post)
    # print("mtot : {0},{1},{2}".format(mod,merr,perr))
    # # exit()

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

    # chains_norv = np.reshape(chains_norv,(chains_norv.shape[0]*chains_norv.shape[1],chains_norv.shape[2]))
    chains_withrvs = np.reshape(chains_withrvs,(chains_withrvs.shape[0]*chains_withrvs.shape[1],chains_withrvs.shape[2]))

    # loaded_results_norv.post = chains_norv
    loaded_results_withrvs.post = chains_withrvs

    # post_norv = loaded_results_norv.post
    post_withrvs = loaded_results_withrvs.post

    if 0:
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


