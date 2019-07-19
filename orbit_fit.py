__author__ = 'jruffio'

import numpy as np
import os
import sys
import multiprocessing as mp
import astropy.io.fits as pyfits
from astropy.time import Time
if len(sys.argv) == 1:
    osiris_data_dir = "/data/osiris_data"
    astrometry_DATADIR = os.path.join(osiris_data_dir,"astrometry")
    uservs = False
    planet = "b"
    # planet = "bc"
    if uservs and (planet == "b" or planet =="c" or planet == "bc"):
        filename = "{0}/HR8799{1}_rvs.csv".format(astrometry_DATADIR,planet)
    else:
        filename = "{0}/HR8799{1}.csv".format(astrometry_DATADIR,planet)
    # MCMC parameters
    num_temps = 16
    num_walkers = 100
    total_orbits = 100*15 # number of steps x number of walkers (at lowest temperature)
    burn_steps = 0 # steps to burn in per walker
    thin = 2 # only save every 2nd step
    num_threads = 1#mp.cpu_count() # or a different number if you prefer
    suffix = "test_joint_Jason"
    # suffix = "sherlock"
    # suffix = "sherlock_ptemceefix_16_512_78125_50"
    suffix = suffix+"_{0}_{1}_{2}_{3}_{4}".format(num_temps,num_walkers,total_orbits//num_walkers,thin,uservs)
else:
    import matplotlib
    matplotlib.use("Agg")

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
# nice -n 15 /home/anaconda3/bin/python3 /home/sda/jruffio/pyOSIRIS/osirisextract/orbit_fit.py /data/osiris_data /data/osiris_data/astrometry/HR8799bc_rvs.csv bc 16 512 51200 0 2 16 test2_joint_16_512_100_2_True
# nice -n 16 /home/anaconda3/bin/python3 /home/sda/jruffio/pyOSIRIS/osirisextract/orbit_fit.py /data/osiris_data /data/osiris_data/astrometry/HR8799bc.csv bc 16 512 51200 0 2 16 test2_joint_16_512_100_2_False
# nice -n 15 /home/anaconda3/bin/python3 /home/sda/jruffio/pyOSIRIS/osirisextract/orbit_fit.py /data/osiris_data /data/osiris_data/astrometry/HR8799bc_rvs.csv bc 16 512 5120000 0 2 16 gpicruncher_joint_16_512_10000_2_True
# nice -n 16 /home/anaconda3/bin/python3 /home/sda/jruffio/pyOSIRIS/osirisextract/orbit_fit.py /data/osiris_data /data/osiris_data/astrometry/HR8799bc.csv bc 16 512 5120000 0 2 16 gpicruncher_joint_16_512_10000_2_False

import orbitize
from orbitize import priors, sampler
from orbitize import driver

print("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}".format(osiris_data_dir,
                                                       filename,
                                                       planet,
                                                       num_temps,
                                                       num_walkers,
                                                       total_orbits,
                                                       burn_steps,
                                                       thin,
                                                       num_threads,
                                                       suffix))
# exit()

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
system_mass = 1.52#1.47 # [Msol] (Jason)
plx = 24.2175#25.38 # [mas]
mass_err = 0.15#0.3 # [Msol] (Jason)
plx_err = 0.0881#0.7 # [mas]

if "Jason" in suffix:
    plx = 24.76#Jason
    plx_err = 0.64#Jason
print(plx,plx_err)

if 1:
    out_pngs = os.path.join(astrometry_DATADIR,"figures")
    my_driver = driver.Driver(
        filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
        sysrv=sysrv,sysrv_err=sysrv_err,
        mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers, 'num_threads': num_threads}
    )

    # force unrestricted longitude of ascending node
    my_driver.system.angle_upperlim = 2.*np.pi
    if "bc" in planet:
        my_driver.system.coplanar = True

    if my_driver.system.coplanar and len(planet) >=2:
        suffix = suffix + "_coplanar"
        fake_inc_prior = priors.GaussianPrior(-2, 0.01,no_negatives=False)
        fake_lan_prior = priors.GaussianPrior(-3, 0.01,no_negatives=False)
        my_driver.system.sys_priors[2+6] = fake_inc_prior
        my_driver.system.sys_priors[4+6] = fake_lan_prior
        print(my_driver.system.param_idx)
        print(my_driver.system.param_idx['inc2'])
        print(my_driver.system.param_idx['pan2'])

    if len(planet) == 1:
        my_driver.system.sys_priors[0] = priors.JeffreysPrior(1, 1e2)
    if len(planet) == 2:
        my_driver.system.sys_priors[0] = priors.JeffreysPrior(1, 1e2)
        my_driver.system.sys_priors[6] = priors.JeffreysPrior(1, 1e2)
    my_driver.sampler = sampler.MCMC(my_driver.system, num_temps=num_temps, num_walkers=num_walkers, num_threads=num_threads, like='chi2_lnlike', custom_lnlike=None)


    if planet == "bc":
        tmpplanet = "d"
        # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+tmpplanet,'chain_norv_'+tmpplanet+'_sherlock_ptemceefix_16_512_78125_50.hdf5')) as hdulist:
        #     chainspos_b = hdulist[0].data[:,:,-1,:]
        # print(chainspos_b.shape)
        # tmpplanet = "e"
        # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+tmpplanet,'chain_norv_'+tmpplanet+'_sherlock_ptemceefix_16_512_78125_50.hdf5')) as hdulist:
        #     chainspos_c = hdulist[0].data[:,:,-1,:]

        # change init points
        tmpplanet = "b"
        with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+tmpplanet,'chain_norv_'+tmpplanet+'_sherlock_ptemceefix_16_512_78125_50.hdf5')) as hdulist:
            chainspos_b = hdulist[0].data[:,:,-1,:]
        print(chainspos_b.shape)
        tmpplanet = "c"
        with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+tmpplanet,'chain_norv_'+tmpplanet+'_sherlock_ptemceefix_16_512_78125_50.hdf5')) as hdulist:
            chainspos_c = hdulist[0].data[:,:,-1,:]
        # import matplotlib.pyplot as plt
        # plt.scatter(np.ravel(chainspos_b[:,:,4]),np.ravel(chainspos_c[:,:,4]),s=5)
        plxpos = np.concatenate([chainspos_b[:,:,6],chainspos_c[:,:,6]],axis=1)
        mtotpos = np.concatenate([chainspos_b[:,:,7],chainspos_c[:,:,7]],axis=1)
        # lan_b = np.concatenate([chainspos_b[:,:,4],chainspos_b[:,:,4]],axis=1)
        # lan_c = np.concatenate([chainspos_c[:,:,4],chainspos_c[:,:,4]],axis=1)
        chainspos_b = np.tile(chainspos_b,(1,2,1))
        chainspos_c = np.tile(chainspos_c,(1,2,1))
        print(chainspos_c.shape)
        print(my_driver.sampler.curr_pos.shape)
        #HR 8799 b
        my_driver.sampler.curr_pos[:,:,0:6] = np.copy(chainspos_b[0:num_temps,0:num_walkers,0:6])
        #HR 8799 c
        my_driver.sampler.curr_pos[:,:,6:12] = np.copy(chainspos_c[0:num_temps,0:num_walkers,0:6])
        #plx
        my_driver.sampler.curr_pos[:,:,12] = np.copy(plxpos[0:num_temps,0:num_walkers])
        if "_rvs" in filename:
            #sysrv
            # my_driver.sampler.curr_pos[:,:,13] = np.11
            # mtot
            my_driver.sampler.curr_pos[:,:,14] = np.copy(mtotpos[0:num_temps,0:num_walkers])
        else:
            my_driver.sampler.curr_pos[:,:,13] = np.copy(mtotpos[0:num_temps,0:num_walkers])

        # my_driver.sampler.curr_pos[:,:,4] = np.random.uniform(0,2*np.pi,size=(num_temps,num_walkers))
        # my_driver.sampler.curr_pos[:,:,4+6] = np.random.uniform(0,2*np.pi,size=(num_temps,num_walkers))
        my_driver.sampler.curr_pos[:,:,2+6] = np.reshape(fake_inc_prior.draw_samples(np.size(my_driver.sampler.curr_pos[:,:,2+6])),(num_temps,num_walkers))
        my_driver.sampler.curr_pos[:,:,4+6] = np.reshape(fake_lan_prior.draw_samples(np.size(my_driver.sampler.curr_pos[:,:,4+6])),(num_temps,num_walkers))
        # exit()


    # plt.figure(2)
    # plt.scatter(np.ravel(my_driver.sampler.curr_pos[:,:,4]),np.ravel(my_driver.sampler.curr_pos[:,:,4+6]))
    my_driver.sampler.save_intermediate = os.path.join(out_pngs,"HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format(rv_str,planet,suffix))

    # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}_5steps.fits'.format(rv_str,planet,suffix))) as hdulist:
    #     chains = hdulist[0].data
    #     print(chains.shape)
    #     print(chains[0,0,:,0])
    #     exit()

    my_driver.sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)

    if my_driver.system.coplanar and len(planet) >=2:
        my_driver.sampler.chain[:,:,:,2+6] = my_driver.sampler.chain[:,:,:,2]
        my_driver.sampler.chain[:,:,:,4+6] = my_driver.sampler.chain[:,:,:,4]
        my_driver.sampler.results.post[:,2+6] = my_driver.sampler.results.post[:,2]
        my_driver.sampler.results.post[:,4+6] = my_driver.sampler.results.post[:,4]

    # plt.figure(3)
    # plt.scatter(np.ravel(my_driver.sampler.curr_pos[:,:,4]),np.ravel(my_driver.sampler.curr_pos[:,:,4+6]))
    # plt.show()

    hdulist = pyfits.HDUList()
    print(my_driver.sampler.chain.shape)
    hdulist.append(pyfits.PrimaryHDU(data=my_driver.sampler.chain))
    try:
        hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format(rv_str,planet,suffix)), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(out_pngs,"HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format(rv_str,planet,suffix)), clobber=True)
    hdulist.close()

    hdf5_filename=os.path.join(out_pngs,"HR_8799_"+planet,'posterior_{0}_{1}_{2}.fits'.format(rv_str,planet,suffix))
    # To avoid weird behaviours, delete saved file if it already exists from a previous run of this notebook
    if os.path.isfile(hdf5_filename):
        os.remove(hdf5_filename)
    my_driver.sampler.results.save_results(hdf5_filename)
    # # sma1: semimajor axis
    # # ecc1: eccentricity
    # # inc1: inclination
    # # aop1: argument of periastron
    # # pan1: position angle of nodes
    # # epp1: epoch of periastron passage
    # # [repeat for 2, 3, 4, etc if multiple objects]
    # # mtot: total mass
    # # plx:  parallax
    # corner_plot_fig = my_driver.sampler.results.plot_corner(param_list=["sma1","ecc1","inc1","aop1","pan1","epp1"]) # Creates a corner plot and returns Figure object
    # corner_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_b","corner_plot_{0}_{1}_{2}.png".format(rv_str,planet,suffix)))
    #
    #
    # epochs = my_driver.system.data_table['epoch']
    #
    # orbit_plot_fig = my_driver.sampler.results.plot_orbits(
    #     object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
    #     num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
    #     start_mjd=epochs[0] # Minimum MJD for colorbar (here we choose first data epoch)
    # )
    # orbit_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_b",'orbit_plot_{0}_{1}_{2}.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
    #
    # rv_plot_fig = my_driver.sampler.results.plot_rvs(
    #     object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
    #     num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
    #     start_mjd=epochs[0] # Minimum MJD for colorbar (here we choose first data epoch)
    # )
    # rv_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_b",'rv_plot_{0}_{1}_{2}.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
    # print("coucou")
else:
    import matplotlib.pyplot as plt

    osiris_data_dir = "/data/osiris_data"
    astrometry_DATADIR = os.path.join(osiris_data_dir,"astrometry")
    uservs = False
    planet = "b"
    # planet = "bc"
    if uservs and (planet == "b" or planet =="c" or planet == "bc"):
        filename = "{0}/HR8799{1}_rvs.csv".format(astrometry_DATADIR,planet)
    else:
        filename = "{0}/HR8799{1}.csv".format(astrometry_DATADIR,planet)
    # MCMC parameters
    num_temps = 16
    num_walkers = 512
    total_orbits = 512*78125 # number of steps x number of walkers (at lowest temperature)
    burn_steps = 0 # steps to burn in per walker
    thin = 50 # only save every 2nd step
    # suffix = "test_joint"
    # suffix = "sherlock"
    suffix = "sherlock_ptemceefix"
    suffix = suffix+"_{0}_{1}_{2}_{3}".format(num_temps,num_walkers,total_orbits//num_walkers,thin)
    # suffix = suffix+"_{0}_{1}_{2}_{3}_{4}".format(num_temps,num_walkers,total_orbits//num_walkers,thin,uservs)

    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
    data_table = orbitize.read_input.read_file(filename)
    print(data_table)
    print(filename)
    #
    # plt.figure()
    # from matplotlib import patches
    # ax = plt.gca()
    # e1 = patches.Arc((0,0),1,2,0,theta1=0,theta2=45)
    # ax.add_patch(e1)
    # plt.show()

    hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format(rv_str,planet,suffix))
    print(hdf5_filename)
    from orbitize import results
    loaded_results = results.Results() # Create blank results object for loading
    loaded_results.load_results(hdf5_filename)

    with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.hdf5'.format(rv_str,planet,suffix))) as hdulist:
        chains = hdulist[0].data[0,:,1000::,:]

    print(chains.shape)
    chains = np.reshape(chains,(chains.shape[0]*chains.shape[1],chains.shape[2]))
    print(chains.shape)
    chains_a = chains[:,0]
    where_lt90 = np.where(chains_a<100)
    chains = chains[where_lt90[0],:]
    print(chains.shape)
    # exit()
    # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1"]
    # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","plx","mtot"]
    param_list = ["sma1","pan1"]
    # param_list = ["sma1","ecc1"]
    # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","plx","sysrv","mtot"]
    # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","plx","mtot"]
    # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","plx","sysrv","mtot"]
    loaded_results.post = chains#loaded_results.post[,:]
    corner_plot_fig = loaded_results.plot_corner(param_list=param_list)
    corner_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"corner_plot_{0}_{1}_{2}.png".format(rv_str,planet,suffix)))
    exit()

    fig = loaded_results.plot_orbits(
        object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
        num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
        start_mjd=data_table['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
        data_table=data_table,
        cbar_param="sma2",
        total_mass=system_mass,
        parallax=plx,
        system_rv=sysrv
    )
    fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'orbits_plot_{0}_{1}_{2}_obj1.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
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

    fig = loaded_results.plot_rvs(
        object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
        num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
        start_mjd=data_table['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
        data_table=data_table,
        total_mass=system_mass,
        parallax=plx,
        system_rv=sysrv
    )
    fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'rv_plot_{0}_{1}_{2}_obj1.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
    # fig = loaded_results.plot_rvs(
    #     object_to_plot = 2, # Plot orbits for the first (and only, in this case) companion
    #     num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
    #     start_mjd=data_table['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
    #     data_table=data_table
    # )
    # fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'rv_plot_{0}_{1}_{2}_obj2.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
    # plt.show()



# print(orbitize.DATADIR)
# exit(0)
# testDATADIR = "/home/sda/jruffio/orbitize/tests/"

# # orbitize.read_input.read_file('{0}/GJ504.csv'.format(testDATADIR))
# orbitize.read_input.read_file('{0}/HR8799b.csv'.format(testDATADIR))
#
# myDriver = orbitize.driver.Driver('{0}/HR8799b.csv'.format(testDATADIR), # path to data file
#                                   'MCMC', # name of algorithm for orbit-fitting
#                                   1, # number of secondary bodies in system
#                                   1.47, # total system mass [M_sun]
#                                   25.38, # total parallax of system [mas]
#                                   mass_err=0.3, # mass error [M_sun]
#                                   plx_err=0.7) # parallax error [mas]
#
# s = myDriver.sampler
# orbits = s.run_sampler(5)
# print(s.system.param_idx)
# print(orbits[0])

# if 1:
#     try:
#         import mkl
#         mkl.set_num_threads(1)
#     except:
#         pass
#
#     testDATADIR = "/home/sda/jruffio/orbitize/tests/"
#     filename = "{}/HR8799e.csv".format(testDATADIR)
#
#     # system parameters
#     num_secondary_bodies = 1
#     system_mass = 1.47 # [Msol]
#     plx = 25.38 # [mas]
#     mass_err = 0#0.00003 # [Msol]
#     plx_err = 0#0.7 # [mas]
#
#     # MCMC parameters
#     num_temps = 20
#     num_walkers = 100
#     num_threads = mp.cpu_count() # or a different number if you prefer
#
#     my_driver = driver.Driver(
#         filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
#         mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers, 'num_threads': num_threads}
#     )
#
#     total_orbits = 2000000 # number of steps x number of walkers (at lowest temperature)
#     burn_steps = 10000 # steps to burn in per walker
#     thin = 2 # only save every 2nd step
#     my_driver.sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
#
#     hdf5_filename=os.path.join(out_pngs,"HR_8799_b",'my_posterior_oricode_e.hdf5')
#     # To avoid weird behaviours, delete saved file if it already exists from a previous run of this notebook
#     if os.path.isfile(hdf5_filename):
#         os.remove(hdf5_filename)
#     my_driver.sampler.results.save_results(hdf5_filename)
#     corner_plot_fig = my_driver.sampler.results.plot_corner(param_list=["sma1","ecc1","inc1","aop1","pan1","epp1"]) # Creates a corner plot and returns Figure object
#     corner_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_b","for_jason_corner_hr8799e.png"))
#
#     epochs = my_driver.system.data_table['epoch']
#     orbit_plot_fig = my_driver.sampler.results.plot_orbits(
#         object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
#         num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
#         start_mjd=epochs[0] # Minimum MJD for colorbar (here we choose first data epoch)
#     )
#     orbit_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_b",'for_jason_orbits_hr8799e.png')) # This is matplotlib.figure.Figure.savefig()
#
#     print("coucou")
#     exit()

# samples = np.random.uniform(-1,1,size=(100000000,3))
# radii = np.sqrt(np.sum(samples**2,axis=1))
# ball_samples = samples[np.where(radii<1)[0],:]
#
# projected_radii = np.sqrt(np.sum(ball_samples[:,0:2]**2,axis=1))
# hist,bin_edges = np.histogram(projected_radii,bins=100,range=[0,1])
# bin_centers = (bin_edges[0:(np.size(bin_edges)-1)]+bin_edges[1:np.size(bin_edges)])/2
# plt.plot(bin_centers,(hist/bin_centers)/(hist[0]/bin_centers[0]))
# plt.plot(bin_centers,np.sqrt(1-bin_centers**2))
# plt.show()
# exit()
#
# import numpy as np
# samples = np.random.uniform(-1,1,size=(1000000,3))
# radii = np.sqrt(np.sum(samples**2,axis=1))
# sphere_samples = samples[np.where(radii<1)[0],:]/radii[np.where(radii<1)][:,None]
#
# projected_radii = np.sqrt(np.sum(sphere_samples[:,0:2]**2,axis=1))
# hist,bin_edges = np.histogram(projected_radii,bins=10,range=[0,1])
# bin_centers = (bin_edges[0:(np.size(bin_edges)-1)]+bin_edges[1:np.size(bin_edges)])/2
# plt.plot(bin_centers,(hist/bin_centers)/(hist[0]/bin_centers[0]))
# plt.show()
# exit()

# if 1:
#     # utc data,ra(as),raerr(as),dec(as),decerr(as),sep(mas),seperr(mas),PA(deg),PAerr(deg)
#     hr8799b_data =[["2004-07-14T00:00:00",1.471,0.006,0.884,0.006,"","","","","",""],
#                    ["2007-08-02T00:00:00",1.504,0.003,0.837,0.003,"","","","","",""],
#                    ["2007-10-25T00:00:00",1.500,0.007,0.836,0.007,"","","","","",""],
#                    ["2008-09-19T00:00:00",1.516,0.004,0.818,0.004,"","","","","",""],
#                    ["2009-07-30T00:00:00",1.526,0.004,0.797,0.004,"","","","","",""],
#                    ["2009-08-01T00:00:00",1.531,0.007,0.794,0.007,"","","","","",""],
#                    ["2009-11-01T00:00:00",1.524,0.010,0.795,0.010,"","","","","",""],
#                    ["2010-07-13T00:00:00",1.532,0.005,0.783,0.005,"","","","","",""],
#                    ["2010-10-30T00:00:00",1.535,0.015,0.766,0.015,"","","","","",""],
#                    ["2011-07-21T00:00:00",1.541,0.005,0.762,0.005,"","","","","",""],
#                    ["2012-07-22T00:00:00",1.545,0.005,0.747,0.005,"","","","","",""],
#                    ["2012-10-26T00:00:00",1.549,0.004,0.743,0.004,"","","","","",""],
#                    ["2013-10-16T00:00:00",1.545,0.022,0.724,0.022,"","","","","",""],
#                    ["2014-07-17T00:00:00",1.560,0.013,0.725,0.013,"","","","","",""],
#                    ["2014-09-12T00:00:00","","","","",1721.2,1.4,65.46,0.14,"",""],
#                    ["2010-07-11T00:00:00","","","","","","","","",-10.2,1.2],
#                    ["2010-07-12T00:00:00","","","","","","","","",-10.1,1.1],
#                    ["2010-07-13T00:00:00","","","","","","","","",-4.0,2.4],
#                    ["2013-07-25T00:00:00","","","","","","","","",-8.9,1.0],
#                    ["2013-07-26T00:00:00","","","","","","","","",-9.4,1.4],
#                    ["2013-07-27T00:00:00","","","","","","","","",-6.5,2.3],
#                    ["2016-11-06T00:00:00","","","","","","","","",-10.8,2.2],
#                    ["2016-11-07T00:00:00","","","","","","","","",-11.8,1.5],
#                    ["2018-07-22T00:00:00","","","","","","","","",-6.7,1.3]]
#
#     hr8799c_data =[["2004-07-14T00:00:00",-0.739,0.006,0.612,0.006,"","","","","",""],
#                    ["2007-08-02T00:00:00",-0.683,0.004,0.671,0.004,"","","","","",""],
#                    ["2007-10-25T00:00:00",-0.678,0.007,0.676,0.007,"","","","","",""],
#                    ["2008-09-19T00:00:00",-0.663,0.003,0.693,0.003,"","","","","",""],
#                    ["2009-07-30T00:00:00",-0.639,0.004,0.712,0.004,"","","","","",""],
#                    ["2009-08-01T00:00:00",-0.635,0.009,0.722,0.009,"","","","","",""],
#                    ["2009-11-01T00:00:00",-0.636,0.009,0.720,0.009,"","","","","",""],
#                    ["2010-07-13T00:00:00",-0.619,0.004,0.728,0.004,"","","","","",""],
#                    ["2010-10-30T00:00:00",-0.607,0.012,0.744,0.012,"","","","","",""],
#                    ["2011-07-21T00:00:00",-0.595,0.004,0.747,0.004,"","","","","",""],
#                    ["2012-07-22T00:00:00",-0.578,0.005,0.761,0.005,"","","","","",""],
#                    ["2012-10-26T00:00:00",-0.572,0.003,0.768,0.003,"","","","","",""],
#                    ["2013-10-16T00:00:00",-0.542,0.022,0.784,0.022,"","","","","",""],
#                    ["2014-07-17T00:00:00",-0.540,0.013,0.799,0.013,"","","","","",""],
#                    ["2013-11-17T00:00:00","","","","",949.5,0.9,325.18,0.14,"",""],
#                    ["2014-09-12T00:00:00","","","","",949.0,1.1,326.53,0.14,"",""],
#                    ["2016-09-19T00:00:00","","","","",944.2,1,330.01,0.14,"",""],
#                    ["2010-07-15T00:00:00","","","","","","","","",-11.8,0.7],
#                    ["2010-11-04T00:00:00","","","","","","","","",-11.5,0.8],
#                    ["2011-07-23T00:00:00","","","","","","","","",-10.8,1.3],
#                    ["2011-07-25T00:00:00","","","","","","","","",-18.8,5.7],]
#
#
#     hr8799d_data =[
#                    ["2007-08-02T00:00:00",-0.179,0.005,-0.588,0.005,"","","","","",""],
#                    ["2007-10-25T00:00:00",-0.175,0.010,-0.589,0.010,"","","","","",""],
#                    ["2008-09-19T00:00:00",-0.202,0.004,-0.588,0.005,"","","","","",""],
#                    ["2009-07-30T00:00:00",-0.237,0.003,-0.577,0.003,"","","","","",""],
#                    ["2009-08-01T00:00:00",-0.250,0.007,-0.570,0.007,"","","","","",""],
#                    ["2009-11-01T00:00:00",-0.251,0.007,-0.573,0.007,"","","","","",""],
#                    ["2010-07-13T00:00:00",-0.265,0.004,-0.576,0.004,"","","","","",""],
#                    ["2010-10-30T00:00:00",-0.296,0.013,-0.561,0.013,"","","","","",""],
#                    ["2011-07-21T00:00:00",-0.303,0.005,-0.562,0.005,"","","","","",""],
#                    ["2012-07-22T00:00:00",-0.339,0.005,-0.555,0.005,"","","","","",""],
#                    ["2012-10-26T00:00:00",-0.346,0.004,-0.548,0.004,"","","","","",""],
#                    ["2013-10-16T00:00:00",-0.382,0.016,-0.522,0.016,"","","","","",""],
#                    ["2014-07-17T00:00:00",-0.400,0.011,-0.534,0.011,"","","","","",""],
#                    ["2013-11-17T00:00:00","","","","",654.6,0.9,214.15,0.15,"",""],
#                    ["2014-09-12T00:00:00","","","","",662.5,1.3,216.57,0.17,"",""],
#                    ["2016-09-19T00:00:00","","","","",674.5,1.0,221.81,0.15,"",""]]
#
#     hr8799e_data =[
#                    ["2009-07-30T00:00:00",-0.306,0.007,-0.211,0.007,"","","","","",""],
#                    ["2009-08-01T00:00:00",-0.318,0.010,-0.195,0.010,"","","","","",""],
#                    ["2009-11-01T00:00:00",-0.310,0.009,-0.187,0.009,"","","","","",""],
#                    ["2010-07-13T00:00:00",-0.323,0.007,-0.166,0.006,"","","","","",""],
#                    ["2010-10-30T00:00:00",-0.341,0.016,-0.143,0.016,"","","","","",""],
#                    ["2011-07-21T00:00:00",-0.352,0.008,-0.130,0.008,"","","","","",""],
#                    ["2012-07-22T00:00:00",-0.373,0.008,-0.084,0.008,"","","","","",""],
#                    ["2012-10-26T00:00:00",-0.370,0.009,-0.076,0.009,"","","","","",""],
#                    ["2013-10-16T00:00:00",-0.373,0.013,-0.017,0.013,"","","","","",""],
#                    ["2014-07-17T00:00:00",-0.387,0.011,0.003,0.011,"","","","","",""],
#                    ["2013-11-17T00:00:00","","","","",382.6,2.1,265.13,0.24,"",""],
#                    ["2016-09-19T00:00:00","","","","",384.8,1.7,281.68,0.25,"",""]]
#
#     # print("epoch,object,sep,sep_err,pa,pa_err,rv,rv_err")
#     print("epoch,object,sep,sep_err,pa,pa_err,raoff,raoff_err,decoff,decoff_err,rv,rv_err")
#     for row in hr8799d_data:
#         print("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}".format(Time(row[0],format="isot",scale="utc").mjd,
#                                                1,
#                                                row[5],
#                                                row[6],
#                                                row[7],
#                                                row[8],
#                                                1000*row[1],
#                                                1000*row[2],
#                                                1000*row[3],
#                                                1000*row[4],
#                                                row[9],
#                                                row[10]))
#     exit()
