__author__ = 'jruffio'
__author__ = 'jruffio'

from orbitize.kepler import _calc_ecc_anom





import warnings
import h5py
import astropy.units as u
import astropy.constants as consts
from astropy.io import fits
from astropy.time import Time
from astropy._erfa.core import ErfaWarning

import matplotlib as mpl
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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

def calc_orbit_3d(epochs, sma, ecc, inc, argp, lan, tau, plx, mtot, mass=None, tau_ref_epoch=0, tolerance=1e-9, max_iter=100):
    """
    Returns the separation and radial velocity of the body given array of
    orbital parameters (size n_orbs) at given epochs (array of size n_dates)

    Based on orbit solvers from James Graham and Rob De Rosa. Adapted by Jason Wang and Henry Ngo.

    Args:
        epochs (np.array): MJD times for which we want the positions of the planet
        sma (np.array): semi-major axis of orbit [au]
        ecc (np.array): eccentricity of the orbit [0,1]
        inc (np.array): inclination [radians]
        argp (np.array): argument of periastron [radians]
        lan (np.array): longitude of the ascending node [radians]
        tau (np.array): epoch of periastron passage in fraction of orbital period past MJD=0 [0,1]
        plx (np.array): parallax [mas]
        mtot (np.array): total mass [Solar masses]
        mass (np.array, optional): mass of the body [Solar masses]. For planets mass ~ 0 (default)
        tau_ref_epoch (float, optional): reference date that tau is defined with respect to (i.e., tau=0)
        tolerance (float, optional): absolute tolerance of iterative computation. Defaults to 1e-9.
        max_iter (int, optional): maximum number of iterations before switching. Defaults to 100.

    Return:
        3-tuple:

            raoff (np.array): array-like (n_dates x n_orbs) of RA offsets between the bodies
            (origin is at the other body) [mas]

            deoff (np.array): array-like (n_dates x n_orbs) of Dec offsets between the bodies [mas]

            vz (np.array): array-like (n_dates x n_orbs) of radial velocity offset between the bodies  [km/s]

    Written: Jason Wang, Henry Ngo, 2018
    """

    n_orbs  = np.size(sma)  # num sets of input orbital parameters
    n_dates = np.size(epochs) # number of dates to compute offsets and vz

    # Necessary for _calc_ecc_anom, for now
    if np.isscalar(epochs): # just in case epochs is given as a scalar
        epochs = np.array([epochs])
    ecc_arr = np.tile(ecc, (n_dates, 1))

    # If mass not given, assume test particle case
    if mass is None:
        mass = np.zeros(n_orbs)

    # Compute period (from Kepler's third law) and mean motion
    period = np.sqrt(4*np.pi**2.0*(sma*u.AU)**3/(consts.G*(mtot*u.Msun)))
    period = period.to(u.day).value
    mean_motion = 2*np.pi/(period) # in rad/day

    # # compute mean anomaly (size: n_orbs x n_dates)
    manom = (mean_motion*(epochs[:, None] - tau_ref_epoch) - 2*np.pi*tau) % (2.0*np.pi)

    # compute eccentric anomalies (size: n_orbs x n_dates)
    eanom = _calc_ecc_anom(manom, ecc_arr, tolerance=tolerance, max_iter=max_iter)

    # compute the true anomalies (size: n_orbs x n_dates)
    # Note: matrix multiplication makes the shapes work out here and below
    tanom = 2.*np.arctan(np.sqrt( (1.0 + ecc)/(1.0 - ecc))*np.tan(0.5*eanom) )
    # compute 3-D orbital radius of second body (size: n_orbs x n_dates)
    radius = sma * (1.0 - ecc * np.cos(eanom))

    # compute ra/dec offsets (size: n_orbs x n_dates)
    # math from James Graham. Lots of trig
    c2i2 = np.cos(0.5*inc)**2
    s2i2 = np.sin(0.5*inc)**2
    arg1 = tanom + argp + lan
    arg2 = tanom + argp - lan
    c1 = np.cos(arg1)
    c2 = np.cos(arg2)
    s1 = np.sin(arg1)
    s2 = np.sin(arg2)

    # updated sign convention for Green Eq. 19.4-19.7
    raoff = radius * (c2i2*s1 - s2i2*s2) * plx
    deoff = radius * (c2i2*c1 + s2i2*c2) * plx
    # zoff = np.ones(raoff.shape)
    zoff = radius*np.sin(tanom+argp) * np.sin(inc) * plx

    # compute the radial velocity (vz) of the body (size: n_orbs x n_dates)
    # first comptue the RV semi-amplitude (size: n_orbs x n_dates)
    m1 = mtot - mass # mass of the primary star
    Kv = np.sqrt(consts.G / (1.0 - ecc**2)) * (m1 * u.Msun * np.sin(inc)) / np.sqrt(mtot * u.Msun) / np.sqrt(sma * u.au)
    # Convert to km/s
    Kv = Kv.to(u.km/u.s)
    # compute the vz
    vz =  Kv.value * ( ecc*np.cos(argp) + np.cos(argp + tanom) )

    # Squeeze out extra dimension (useful if n_orbs = 1, does nothing if n_orbs > 1)
    # [()] used to convert 1-element arrays into scalars, has no effect for larger arrays
    # raoff = np.transpose(np.squeeze(raoff)[()])
    # deoff = np.transpose(np.squeeze(deoff)[()])
    # vz = np.transpose(np.squeeze(vz)[()])
    vz = np.squeeze(vz)[()]

    return raoff, deoff, zoff, vz


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
        if 1:
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

    chains_norv = np.reshape(chains_norv,(chains_norv.shape[0]*chains_norv.shape[1],chains_norv.shape[2]))
    chains_withrvs = np.reshape(chains_withrvs,(chains_withrvs.shape[0]*chains_withrvs.shape[1],chains_withrvs.shape[2]))

    loaded_results_norv.post = chains_norv
    loaded_results_withrvs.post = chains_withrvs

    post_norv = loaded_results_norv.post
    post_withrvs = loaded_results_withrvs.post

    if 1:
        # Create figure for orbit plots
        fig = plt.figure(figsize=(5,5))
        post = post_norv
        # post = post_withrvs

        num_orbits_to_plot= 500 # Will plot 100 randomly selected orbits of this companion
        start_mjd=data_table_withrvs['epoch'][0] # Minimum MJD for colorbar (here we choose first data epoch)
        data_table=data_table_withrvs
        cbar_param='epochs'
        total_mass=system_mass
        parallax=plx
        system_rv=sysrv
        num_epochs_to_plot=100
        object_mass = 0
        square_plot=True
        tau_ref_epoch = loaded_results_withrvs.tau_ref_epoch
        sep_pa_end_year=2025.0
        show_colorbar=True

        start_yr = Time(start_mjd,format='mjd').decimalyear

        # ax0 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
        ax0 = fig.gca(projection="3d")
        # ax0 = plt.subplot2grid((6, 14), (0, 0), rowspan=6, colspan=6)
        # ax11 = plt.subplot2grid((6, 14), (0, 9), colspan=6)
        # ax21 = plt.subplot2grid((6, 14), (2, 9), colspan=6)
        # ax12 = plt.subplot2grid((6, 14), (1, 9), colspan=6)
        # ax22 = plt.subplot2grid((6, 14), (3, 9), colspan=6)
        # ax3 = plt.subplot2grid((6, 14), (4, 9), rowspan=2, colspan=6)
        # cmap = mpl.cm.Purples_r
        # exit()
        for object_to_plot,cmap,pl_color in zip([1,2],[mpl.cm.Blues_r,mpl.cm.Oranges_r],["#006699","#ff9900"]): # Plot orbits for the first (and only, in this case) companion
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
                zoff = np.zeros((num_orbits_to_plot, num_epochs_to_plot))
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
                    raoff0, deoff0,zoff0, relrv0 = calc_orbit_3d(
                        epochs[i,:], sma[orb_ind], ecc[orb_ind], inc[orb_ind], aop[orb_ind], pan[orb_ind],
                        tau[orb_ind], plx[orb_ind], mtot[orb_ind], mass=mplanet[orb_ind], tau_ref_epoch=tau_ref_epoch
                    )

                    raoff[i,:] = raoff0
                    deoff[i,:] = deoff0
                    zoff[i,:] = zoff0
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
                    points = np.array([raoff[i,:], deoff[i,:],zoff[i,:]]).T.reshape(-1,1,3)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = Line3DCollection(
                        segments, cmap=cmap, norm=norm, linewidth=1.0,alpha=0.5
                    )
                    if cbar_param != 'epochs':
                        lc.set_array(np.ones(len(epochs[0]))*cbar_param_arr[i])
                    elif cbar_param == 'epochs':
                        lc.set_array(epochs[i,:])
                    ax0.add_collection3d(lc)

                # modify the axes
                if square_plot:
                    adjustable_param='datalim'
                else:
                    adjustable_param='box'
                ax0.set_aspect('equal', adjustable=adjustable_param)
                ax0.set_xlabel('$\Delta$RA (mas)',fontsize=fontsize)
                ax0.set_ylabel('$\Delta$Dec (mas)',fontsize=fontsize)
                ax0.set_zlabel('z',fontsize=fontsize)
                ax0.tick_params(axis='x', labelsize=fontsize)
                ax0.tick_params(axis='y', labelsize=fontsize)
                ax0.tick_params(axis='z', labelsize=fontsize)
                plt.sca(ax0)
                plt.xlim([-2000,2000])
                plt.ylim([-2000,2000])
                ax0.set_zlim([-2000,2000])
                plt.xticks([2000,1000,0,-1000,-2000])
                plt.yticks([-2000,-1000,0,1000,2000])
                # plt.zticks([-2000,-1000,0,1000,2000])

                # if data_table is not None:
                #     plt.errorbar(data_table["quant1"][radec_indices],data_table["quant2"][radec_indices],
                #                  xerr=data_table["quant1_err"][radec_indices],
                #                  yerr=data_table["quant2_err"][radec_indices],fmt="x",color=pl_color)
                #
                #     for seppa_index in seppa_indices[0]:
                #         ra_from_seppa = data_table["quant1"][seppa_index]*np.sin(np.deg2rad(data_table["quant2"][seppa_index]))
                #         dec_from_seppa = data_table["quant1"][seppa_index]*np.cos(np.deg2rad(data_table["quant2"][seppa_index]))
                #         dra_from_seppa = data_table["quant1_err"][seppa_index]*np.sin(np.deg2rad(data_table["quant2"][seppa_index]))
                #         ddec_from_seppa = data_table["quant1_err"][seppa_index]*np.cos(np.deg2rad(data_table["quant2"][seppa_index]))
                #         plt.plot(ra_from_seppa,dec_from_seppa,"o",color=pl_color)
                #         plt.plot([ra_from_seppa-dra_from_seppa,ra_from_seppa+dra_from_seppa],
                #                  [dec_from_seppa-ddec_from_seppa,dec_from_seppa+ddec_from_seppa],color=pl_color, linestyle ="--")
                #         e1 = mpl.patches.Arc((0,0),2*data_table["quant1"][seppa_index],2*data_table["quant1"][seppa_index],0,
                #                              theta2=90-(data_table["quant2"][seppa_index]-data_table["quant2_err"][seppa_index]),
                #                              theta1=90-(data_table["quant2"][seppa_index]+data_table["quant2_err"][seppa_index]),
                #                              color=pl_color, linestyle ="--")
                #         ax0.add_patch(e1)
                ax0.invert_xaxis()


            arr2save = np.concatenate([raoff[None,:,:],deoff[None,:,:],zoff[None,:,:],yr_epochs[None,:,:]],axis=0)
            print(arr2save.shape)

            # hdulist = pyfits.HDUList()
            # hdulist.append(pyfits.PrimaryHDU(data=arr2save))
            # try:
            #     hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_3d_including_RV_planet{0}.fits'.format(object_to_plot)), overwrite=True)
            # except TypeError:
            #     hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_3d_including_RV_planet{0}.fits'.format(object_to_plot)), clobber=True)
            # hdulist.close()
            hdulist = pyfits.HDUList()
            hdulist.append(pyfits.PrimaryHDU(data=arr2save))
            try:
                hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_3d_no_RV_planet{0}.fits'.format(object_to_plot)), overwrite=True)
            except TypeError:
                hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_3d_no_RV_planet{0}.fits'.format(object_to_plot)), clobber=True)
            hdulist.close()
        exit()



        # fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_bc_withrvs.pdf')) # This is matplotlib.figure.Figure.savefig()
        # fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_bc_withrvs.png')) # This is matplotlib.figure.Figure.savefig()
        # plt.show()
