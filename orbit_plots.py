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
plx = 25.38 # [mas]
mass_err = 0.3#0.3 # [Msol]
plx_err = 0.7#0.7 # [mas]
# suffix = "test4_coplanar"
# suffix_norv = "test_joint_16_512_1000_2_False_coplanar"
# suffix_withrvs = "test_joint_16_512_1000_2_True_coplanar"
# suffix_norv = "gpicruncher_joint_16_512_10000_2_False_coplanar"
# suffix_withrvs = "gpicruncher_joint_16_512_10000_2_True_coplanar"
suffix_norv = "sherlock_16_1024_200000_50_False_coplanar"
suffix_withrvs = "sherlock_16_1024_200000_50_True_coplanar"
# suffix = "sherlock"
# suffix = "sherlock_ptemceefix_12_100_300000_50"



if 1:
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
    hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format("norv",planet,suffix_norv2))
    print(hdf5_filename)
    loaded_results_norv = results.Results() # Create blank results object for loading
    loaded_results_norv.load_results(hdf5_filename)
    hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format("withrvs",planet,suffix_withrvs2))
    print(hdf5_filename)
    loaded_results_withrvs = results.Results() # Create blank results object for loading
    loaded_results_withrvs.load_results(hdf5_filename)

    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("norv",planet,suffix_norv))) as hdulist:
        chains_norv = hdulist[0].data
        if "coplanar" in suffix_norv:
            chains_norv[:,:,:,2+6] = chains_norv[:,:,:,2]
            chains_norv[:,:,:,4+6] = chains_norv[:,:,:,4]
        print(chains_norv.shape)
        chains_norv = chains_norv[0,:,3*chains_norv.shape[2]//4::,:]
        # chains_norv = chains_norv[0,:,:,:]
        # chains_norv = chains_norv[0,:,0:10,:]
        # chains_norv = chains_norv[0,:,0:1,:]
    print(chains_norv.shape)
    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format("withrvs",planet,suffix_withrvs))) as hdulist:
        chains_withrvs = hdulist[0].data
        if "coplanar" in suffix_withrvs:
            chains_withrvs[:,:,:,2+6] = chains_withrvs[:,:,:,2]
            chains_withrvs[:,:,:,4+6] = chains_withrvs[:,:,:,4]
        print(chains_withrvs.shape)
        chains_withrvs = chains_withrvs[0,:,3*chains_withrvs.shape[2]//4::,:]
        # chains_withrvs = chains_withrvs[0,:,:,:]
        # chains_withrvs = chains_withrvs[0,:,0:10,:]
        # chains_withrvs = chains_withrvs[0,:,0:1,:]
    print(chains_withrvs.shape)
    # exit()

    # plt.subplot(2,1,1)
    # plt.plot(chains_norv[:,:,6].T,color="grey",alpha =0.1 )
    # plt.subplot(2,1,2)
    # plt.plot(chains_withrvs[:,:,6].T,color="grey",alpha =0.1 )
    # plt.show()

    chains_norv = np.reshape(chains_norv,(chains_norv.shape[0]*chains_norv.shape[1],chains_norv.shape[2]))
    chains_withrvs = np.reshape(chains_withrvs,(chains_withrvs.shape[0]*chains_withrvs.shape[1],chains_withrvs.shape[2]))

    loaded_results_norv.post = chains_norv
    loaded_results_withrvs.post = chains_withrvs

    post_norv = loaded_results_norv.post
    post_withrvs = loaded_results_withrvs.post

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
        inc_Ome_norv_b,xedges,yedges = np.histogram2d(np.rad2deg(post_norv[:,4]),np.rad2deg(post_norv[:,2]),bins=[40,20],range=[Ome_bounds,inc_bounds])
        inc_Ome_withrvs_b,xedges,yedges = np.histogram2d(np.rad2deg(post_withrvs[:,4]),np.rad2deg(post_withrvs[:,2]),bins=[40,20],range=[Ome_bounds,inc_bounds])
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
        inc_Ome_norv_b,xedges,yedges = np.histogram2d(post_norv[:,0],post_norv[:,1],bins=[40,20],range=[sma_bounds,ecc_bounds])
        inc_Ome_withrvs_b,xedges,yedges = np.histogram2d(post_withrvs[:,0],post_withrvs[:,1],bins=[40,20],range=[sma_bounds,ecc_bounds])
        inc_Ome_norv_c,xedges,yedges = np.histogram2d(post_norv[:,0+6],post_norv[:,1+6],bins=[40,20],range=[sma_bounds,ecc_bounds])
        inc_Ome_withrvs_c,xedges,yedges = np.histogram2d(post_withrvs[:,0+6],post_withrvs[:,1+6],bins=[40,20],range=[sma_bounds,ecc_bounds])
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

    if 0:
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
                ax0.set_xlabel('$\Delta$RA (mas)')
                ax0.set_ylabel('$\Delta$Dec (mas)')
                ax0.locator_params(axis='x', nbins=6)
                ax0.locator_params(axis='y', nbins=6)
                plt.xlim([-2000,2000])
                plt.ylim([-2000,2000])

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
                    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm_yr, orientation='vertical', label="b: "+cbar_param)
                if object_to_plot == 2:
                    cbar_ax = fig.add_axes([0.47, 0.5, 0.015, 0.4]) # xpos, ypos, width, height, in fraction of figure size
                    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm_yr, orientation='vertical', label="c: "+cbar_param)


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

                ax1.locator_params(axis='x', nbins=6)
                ax1.locator_params(axis='y', nbins=3)
                ax2.locator_params(axis='x', nbins=6)
                ax2.locator_params(axis='y', nbins=3)
                ax3.locator_params(axis='x', nbins=6)
                ax3.locator_params(axis='y', nbins=5)


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
                    sep_list,pa_list = orbitize.system.radec2seppa(ra_list,dec_list)
                    sep_err_list = np.zeros(ra_list.shape)
                    pa_err_list = np.zeros(ra_list.shape)
                    for myid,(ra,dec,ra_err,dec_err) in enumerate(zip(ra_list,dec_list,ra_err_list,dec_err_list)):
                        mean = [ra,dec]
                        cov=np.diag([ra_err**2,dec_err**2])
                        radec_samples = np.random.multivariate_normal(mean,cov,size=200)
                        sep_samples,pa_samples = orbitize.system.radec2seppa(radec_samples[:,0],radec_samples[:,1])
                        sep_err_list[myid] = np.std(sep_samples)
                        pa_err_list[myid] = np.std(pa_samples)

                    plt.sca(ax1)
                    plt.errorbar(Time(data_table["epoch"][radec_indices],format='mjd').decimalyear,
                                 sep_list,
                                 yerr=sep_err_list,fmt="x",color=pl_color,linestyle="",zorder=10)
                    plt.sca(ax2)
                    plt.errorbar(Time(data_table["epoch"][radec_indices],format='mjd').decimalyear,
                                 pa_list,
                                 yerr=pa_err_list,fmt="x",color=pl_color,linestyle="",zorder=10)

        plt.sca(ax11)
        plt.ylim([1700,1730])
        ax11.set_ylabel('$\\rho_b$ (mas)')
        plt.sca(ax12)
        plt.ylim([930,970])
        ax12.set_ylabel('$\\rho_c$ (mas)')
        plt.sca(ax21)
        plt.ylim([55,70])
        ax21.set_ylabel('PA$_b$ (deg)')
        plt.sca(ax22)
        plt.ylim([300,340])
        ax22.set_ylabel('PA$_c$ (deg)')
        plt.sca(ax3)
        plt.ylim([-20,0])
        # plot sep/PA zoom-in panels
        ax3.set_ylabel('RV (km/s)')
        ax3.set_xlabel('Epoch (yr)')

        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_bc_withrvs.pdf')) # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_bc_withrvs.png')) # This is matplotlib.figure.Figure.savefig()
        plt.show()
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


