__author__ = 'jruffio'

import numpy as np
import os
import sys
import multiprocessing as mp
from astropy.time import Time
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
    if uservs and (planet == "b" or planet =="c" or planet == "bc"):
        filename = "{0}/HR8799{1}_rvs.csv".format(astrometry_DATADIR,planet)
    else:
        filename = "{0}/HR8799{1}.csv".format(astrometry_DATADIR,planet)
    # MCMC parameters
    num_temps = 20
    num_walkers = 100
    total_orbits = 10000 # number of steps x number of walkers (at lowest temperature)
    burn_steps = 100 # steps to burn in per walker
    thin = 2 # only save every 2nd step
    num_threads = mp.cpu_count() # or a different number if you prefer
    # suffix = "test2"
    suffix = "sherlock"
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

if "_rvs" in filename:
    rv_str = "withrvs"
    sysrv=-12.6
    sysrv_err=1.4
else:
    rv_str = "norv"
    sysrv=0
    sysrv_err=0

# print(rv_str,sysrv,sysrv_err)
# exit()

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

# system parameters
num_secondary_bodies = len(planet)
system_mass = 1.47 # [Msol]
plx = 25.38 # [mas]
mass_err = 0.3#0.3 # [Msol]
plx_err = 0.7#0.7 # [mas]

if 1:
    import matplotlib.pyplot as plt
    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
    data_table = orbitize.read_input.read_file(filename)
    print(data_table)
    print(filename)
    # exit()

    from orbitize import results
    hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format("norv",planet,suffix))
    print(hdf5_filename)
    loaded_results_norv = results.Results() # Create blank results object for loading
    loaded_results_norv.load_results(hdf5_filename)
    hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format("withrvs",planet,suffix))
    print(hdf5_filename)
    loaded_results_withrvs = results.Results() # Create blank results object for loading
    loaded_results_withrvs.load_results(hdf5_filename)

    post_norv = loaded_results_norv.post[:,:]
    post_withrvs = loaded_results_withrvs.post[:,:]



    # N_walkers = 1000
    # chain_size = 5000
    # post_norv = np.reshape(post_norv,(N_walkers,chain_size,post_norv.shape[1]))

    # plt.plot(post_norv[::10,:,4+6].T)
    # plt.show()

    Ome_bounds = [0,360]
    inc_bounds = [0,180]

    cmaps = ["cool","hot"]
    colors=["#006699","#ff9900","grey"]
    inc_Ome_norv_b,xedges,yedges = np.histogram2d(np.rad2deg(post_norv[:,4]),np.rad2deg(post_norv[:,2]),bins=[40,20],range=[Ome_bounds,inc_bounds])
    inc_Ome_withrvs_b,xedges,yedges = np.histogram2d(np.rad2deg(post_withrvs[:,4]),np.rad2deg(post_withrvs[:,2]),bins=[40,20],range=[Ome_bounds,inc_bounds])
    inc_Ome_norv_c,xedges,yedges = np.histogram2d(np.rad2deg(post_norv[:,4+6]),np.rad2deg(post_norv[:,2+6]),bins=[40,20],range=[Ome_bounds,inc_bounds])
    inc_Ome_withrvs_c,xedges,yedges = np.histogram2d(np.rad2deg(post_withrvs[:,4+6]),np.rad2deg(post_withrvs[:,2+6]),bins=[40,20],range=[Ome_bounds,inc_bounds])
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
    #            extent=[Ome_bounds[0],Ome_bounds[1],inc_bounds[0],inc_bounds[1]],
    #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
    plt.xlim(Ome_bounds)
    plt.ylim(inc_bounds)
    levels = [0.6827]
    xx,yy = np.meshgrid(x_centers,y_centers)
    CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)
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
    CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles="--",linewidths=[2],colors=(colors[0],),zorder=15)
    # levels = [0.9545,0.9973]
    # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)


    ravel_H = np.ravel(inc_Ome_norv_c)
    ind = np.argsort(ravel_H)
    cum_ravel_H = np.zeros(np.shape(ravel_H))
    cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
    cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inc_Ome_norv_c))
    cum_H.shape = inc_Ome_norv_c.shape
    image = copy(inc_Ome_norv_c)
    image[np.where(cum_H>0.9545)] = np.nan

    # plt.imshow(image,origin ="lower",
    #            extent=[Ome_bounds[0],Ome_bounds[1],inc_bounds[0],inc_bounds[1]],
    #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
    plt.xlim(Ome_bounds)
    plt.ylim(inc_bounds)
    levels = [0.6827]
    xx,yy = np.meshgrid(x_centers,y_centers)
    CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles="-",linewidths=[2],colors=(colors[1],),zorder=15)
    # levels = [0.9545,0.9973]
    # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)


    ravel_H = np.ravel(inc_Ome_withrvs_c)
    ind = np.argsort(ravel_H)
    cum_ravel_H = np.zeros(np.shape(ravel_H))
    cum_ravel_H[ind] = np.cumsum(ravel_H[ind])
    cum_H = 1-np.reshape(cum_ravel_H/np.nanmax(cum_ravel_H),np.shape(inc_Ome_withrvs_c))
    cum_H.shape = inc_Ome_withrvs_c.shape
    image = copy(inc_Ome_withrvs_c)
    image[np.where(cum_H>0.9545)] = np.nan

    # plt.imshow(image,origin ="lower",
    #            extent=[Ome_bounds[0],Ome_bounds[1],inc_bounds[0],inc_bounds[1]],
    #            aspect="auto",zorder=10,cmap=cmaps[0],alpha=0.5)#,alpha = 0.5,interpolation="spline16",,alpha = 0.75
    plt.xlim(Ome_bounds)
    plt.ylim(inc_bounds)
    levels = [0.6827]
    xx,yy = np.meshgrid(x_centers,y_centers)
    CS = plt.contour(xx,yy,cum_H,levels = levels,linestyles="--",linewidths=[2],colors=(colors[1],),zorder=15)
    # levels = [0.9545,0.9973]
    # CS = plt.contour(xx,yy,cum_H.T,levels = levels,linestyles="-",linewidths=[2],colors=(colors[0],),zorder=15)

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


