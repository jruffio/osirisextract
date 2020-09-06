__author__ = 'jruffio'

import numpy as np
import os
import sys
import multiprocessing as mp
import astropy.io.fits as pyfits
from astropy.time import Time
import orbitize
from orbitize import priors, sampler
from orbitize import driver

if __name__ == "__main__":

    if len(sys.argv) == 1:
        # osiris_data_dir = "/data/osiris_data"
        osiris_data_dir = "/scr3/jruffio/data/osiris_data"
        astrometry_DATADIR = os.path.join(osiris_data_dir,"astrometry")
        uservs = True
        # planet = "b"
        # planet = "bc"
        planet = "bcd"
        # coplanar = False
        coplanar = True
        if uservs:
            filename = "{0}/HR8799{1}_rvs.csv".format(astrometry_DATADIR,planet)
        else:
            filename = "{0}/HR8799{1}.csv".format(astrometry_DATADIR,planet)
        # MCMC parameters
        num_temps = 16
        num_walkers = 512
        total_orbits = 512*100000 # number of steps x number of walkers (at lowest temperature)
        burn_steps = 0 # steps to burn in per walker
        thin = 50 # only save every 2nd step
        num_threads = 16#mp.cpu_count() # or a different number if you prefer
        # suffix = "test_fixomegabug_notcoplanar"
        # suffix = "sherlock"
        # suffix = "sherlock_ptemceefix_16_512_78125_50"
        # suffix = "test_bcd"
        itnum = 4
        suffix = "it{0}".format(itnum)
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
        coplanar = bool(int(sys.argv[11]))
    #     sbatch --partition=hns,owners,iric --qos=normal --time=2-00:00:00 --mem=20G --output=/scratch/groups/bmacint/osiris_data/astrometry/logs/20190703_203155_orbit_fit_HR8799b.csv --error=/scratch/groups/bmacint/osiris_data/astrometry/logs/20190703_203155_orbit_fit_HR8799b.csv --nodes=1 --ntasks-per-node=10 --mail-type=END,FAIL,BEGIN --mail-user=jruffio@stanford.edu --wrap="
    # nice -n 15 /home/anaconda3/bin/python3 /home/sda/jruffio/pyOSIRIS/osirisextract/orbit_fit.py /data/osiris_data /data/osiris_data/astrometry/HR8799bc_rvs.csv bc 16 512 51200 0 2 16 test2_joint_16_512_100_2_True
    # nice -n 16 /home/anaconda3/bin/python3 /home/sda/jruffio/pyOSIRIS/osirisextract/orbit_fit.py /data/osiris_data /data/osiris_data/astrometry/HR8799bc.csv bc 16 512 51200 0 2 16 test2_joint_16_512_100_2_False
    # nice -n 15 /home/anaconda3/bin/python3 /home/sda/jruffio/pyOSIRIS/osirisextract/orbit_fit.py /data/osiris_data /data/osiris_data/astrometry/HR8799bc_rvs.csv bc 16 1024 204800000 0 2 16 gpicruncher_joint_16_1024_200000_2_True
    # nice -n 16 /home/anaconda3/bin/python3 /home/sda/jruffio/pyOSIRIS/osirisextract/orbit_fit.py /data/osiris_data /data/osiris_data/astrometry/HR8799bc.csv bc 16 1024 204800000 0 2 16 gpicruncher_joint_16_1024_200000_2_False

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
        # restrict_angle_ranges = True
        restrict_angle_ranges = False
    else:
        rv_str = "norv"
        sysrv=0
        sysrv_err=0
        restrict_angle_ranges = True

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

    # if "Jason" in suffix:
    #     plx = 24.76#Jason
    #     plx_err = 0.64#Jason
    # print(plx,plx_err)


    if 0:
        out_pngs = os.path.join(astrometry_DATADIR,"figures")
        my_driver = driver.Driver(
            filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
            sysrv=sysrv,sysrv_err=sysrv_err,
            mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers, 'num_threads': num_threads},system_kwargs = {"restrict_angle_ranges":restrict_angle_ranges},
        )
        # force unrestricted longitude of ascending node
        # if "_rvs" in filename:
        #     my_driver.system.angle_upperlim = 2.*np.pi
        # else:
        #     my_driver.system.angle_upperlim = np.pi

        # if "bc" == planet:
        #     my_driver.system.coplanar = coplanar
        my_driver.system.coplanar = coplanar

        if my_driver.system.coplanar and len(planet) >=2:
            suffix = suffix + "_coplanar"
            # fake_inc_prior = priors.GaussianPrior(-2, 0.01,no_negatives=False)
            # fake_lan_prior = priors.GaussianPrior(-3, 0.01,no_negatives=False)
            # my_driver.system.sys_priors[2+6] = fake_inc_prior
            # my_driver.system.sys_priors[4+6] = fake_lan_prior
            my_driver.system.sys_priors[2+6] = -2
            my_driver.system.sys_priors[4+6] = -3
            my_driver.system.sys_priors[2+6+6] = -4
            my_driver.system.sys_priors[4+6+6] = -5
            print(my_driver.system.param_idx)
            print(my_driver.system.param_idx['inc2'])
            print(my_driver.system.param_idx['pan2'])

        # # exit()
        # if len(planet) == 1:
        #     my_driver.system.sys_priors[0] = priors.JeffreysPrior(1, 1e2)

        my_driver.sampler = sampler.MCMC(my_driver.system, num_temps=num_temps, num_walkers=num_walkers, num_threads=num_threads, like='chi2_lnlike', custom_lnlike=None)

        print(my_driver.sampler.curr_pos[0,0,:])
        print(my_driver.sampler.curr_pos.shape)
        # exit()

        if not my_driver.system.coplanar:

            # tmpplanet = "b"
            # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+tmpplanet,'chain_norv_'+tmpplanet+'_sherlock_ptemceefix_16_512_78125_50.hdf5')) as hdulist:
            #     chainspos_b = hdulist[0].data[:,:,-1,:]
            # tmpplanet = "c"
            # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+tmpplanet,'chain_norv_'+tmpplanet+'_sherlock_ptemceefix_16_512_78125_50.hdf5')) as hdulist:
            #     chainspos_c = hdulist[0].data[:,:,-1,:]
            # tmpplanet = "d"
            # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+tmpplanet,'chain_norv_'+tmpplanet+'_sherlock_ptemceefix_16_512_78125_50.hdf5')) as hdulist:
            #     chainspos_d = hdulist[0].data[:,:,-1,:]
            #
            # my_driver.sampler.curr_pos[:,:,0:6] = np.copy(chainspos_b[0:num_temps,0:num_walkers,0:6])
            # my_driver.sampler.curr_pos[:,:,6:12] = np.copy(chainspos_c[0:num_temps,0:num_walkers,0:6])
            # my_driver.sampler.curr_pos[:,:,12:18] = np.copy(chainspos_d[0:num_temps,0:num_walkers,0:6])
            # my_driver.sampler.curr_pos[:,:,18] = np.copy(chainspos_d[0:num_temps,0:num_walkers,6])
            # my_driver.sampler.curr_pos[:,:,20] = np.copy(chainspos_d[0:num_temps,0:num_walkers,7])
            _filename = os.path.join(astrometry_DATADIR,"figures","HR_8799_bcd",
                                          'chain_'+rv_str+'_bcd_it{0}_{1}_{2}_{3}_{4}_{5}.fits'.format(
                                              itnum-1,num_temps,num_walkers,100000,thin,uservs))
            print(_filename)
            with pyfits.open(_filename) as hdulist:#total_orbits//num_walkers
                print(hdulist[0].data.shape)
                my_driver.sampler.curr_pos = hdulist[0].data[:,:,-1,:]
        else:
            _filename = os.path.join(astrometry_DATADIR,"figures","HR_8799_bcd",
                                          'chain_'+rv_str+'_bcd_it{0}_{1}_{2}_{3}_{4}_{5}_coplanar.fits'.format(
                                              itnum-1,num_temps,num_walkers,100000,thin,uservs))
            print(_filename)
            with pyfits.open(_filename) as hdulist: #total_orbits//num_walkers
                print(hdulist[0].data.shape)
                # print(my_driver.sampler.curr_pos.shape)
                # exit()
                my_driver.sampler.curr_pos = hdulist[0].data[:,:,-1,:]
        # exit()


        print(my_driver.sampler.curr_pos[0,0,:])
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

        # if my_driver.system.coplanar and len(planet) >=2:
        #     my_driver.sampler.chain[:,:,:,2+6] = my_driver.sampler.chain[:,:,:,2]
        #     my_driver.sampler.chain[:,:,:,4+6] = my_driver.sampler.chain[:,:,:,4]
        #     my_driver.sampler.results.post[:,2+6] = my_driver.sampler.results.post[:,2]
        #     my_driver.sampler.results.post[:,4+6] = my_driver.sampler.results.post[:,4]

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

        hdf5_filename=os.path.join(out_pngs,"HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format(rv_str,planet,suffix))
        # To avoid weird behaviours, delete saved file if it already exists from a previous run of this notebook
        print(hdf5_filename)
        # if os.path.isfile(hdf5_filename):
        #     os.remove(hdf5_filename)
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
        # osiris_data_dir = "/data/osiris_data"
        osiris_data_dir = "/scr3/jruffio/data/osiris_data"
        astrometry_DATADIR = os.path.join(osiris_data_dir,"astrometry")
        uservs = True
        object_to_plot = 3
        # planet = "b"
        # planet = "bc"
        planet = "bcd"
        # coplanar = False
        coplanar = True
        if uservs:
            filename = "{0}/HR8799{1}_rvs.csv".format(astrometry_DATADIR,planet)
        else:
            filename = "{0}/HR8799{1}.csv".format(astrometry_DATADIR,planet)
        # MCMC parameters
        num_temps = 16
        num_walkers = 512
        total_orbits = 512*100000 # number of steps x number of walkers (at lowest temperature)
        burn_steps = 0 # steps to burn in per walker
        thin = 50 # only save every 2nd step
        num_threads = 16#mp.cpu_count() # or a different number if you prefer
        # suffix = "test_fixomegabug_notcoplanar"
        # suffix = "sherlock"
        # suffix = "sherlock_ptemceefix_16_512_78125_50"
        # suffix = "test_bcd"
        suffix = "it3"
        suffix = suffix+"_{0}_{1}_{2}_{3}_{4}".format(num_temps,num_walkers,total_orbits//num_walkers,thin,uservs)
        if coplanar:
            suffix = suffix + "_coplanar"

        # out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
        out_pngs = os.path.join(astrometry_DATADIR,"figures")
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

        suffix2 = "it3"
        suffix2 = suffix2+"_{0}_{1}_{2}_{3}_{4}".format(num_temps,num_walkers,100000,thin,uservs)
        if coplanar:
            suffix2 = suffix2 + "_coplanar"
        hdf5_filename=os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'posterior_{0}_{1}_{2}.hdf5'.format(rv_str,planet,suffix2))
        print(hdf5_filename)
        from orbitize import results
        loaded_results = results.Results() # Create blank results object for loading
        loaded_results.load_results(hdf5_filename)

        with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_"+planet,'chain_{0}_{1}_{2}.fits'.format(rv_str,planet,suffix))) as hdulist:
            if hdulist[0].data[0,:,:,:].shape[2] == 21:
                chains = hdulist[0].data[0,:,:,:]
            else:
                chains = hdulist[0].data[0,:,:,:]
                chains_withrvs = np.zeros((chains.shape[0],chains.shape[1],chains.shape[2]+4))
                a_list = [0,1,2,3,4,5, 6,7,2,8,4,9, 10,11,2,12,4,13, 14,15,16]
                b_list = np.arange(21)
                for a,b in zip(a_list,b_list):
                    chains_withrvs[:,:,b] = chains[:,:,a]
                chains =chains_withrvs
        # chains = chains[:,250::,:]
        print(chains.shape)
        # plt.plot(chains[:,:,0].T)
        # plt.show()

        print(chains.shape)
        chains = np.reshape(chains,(chains.shape[0]*chains.shape[1],chains.shape[2]))
        print(chains.shape)
        # chains_a = chains[:,0]
        # where_lt90 = np.where(chains_a<100)
        # chains = chains[where_lt90[0],:]
        # print(chains.shape)
        # exit()
        # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1"]
        # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","plx","mtot"]
        # param_list = ["sma1","pan1"]
        # param_list = ["sma1","ecc1"]
        # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","plx","sysrv","mtot"]
        # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","plx","mtot"]
        # param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","plx","sysrv","mtot"]
        param_list = ["sma1","ecc1","inc1","aop1","pan1","epp1","sma2","ecc2","inc2","aop2","pan2","epp2","sma3","ecc3","inc3","aop3","pan3","epp3","plx","sysrv","mtot"]
        loaded_results.post = chains # loaded_results.post[,:]

        if 1: # corner
            pass
            corner_plot_fig = loaded_results.plot_corner(param_list=param_list)
            corner_plot_fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"corner_plot_{0}_{1}_{2}.png".format(rv_str,planet,suffix)))
            exit()

        fig = loaded_results.plot_orbits(
            object_to_plot = object_to_plot, # Plot orbits for the first (and only, in this case) companion
            num_orbits_to_plot= 10, # Will plot 100 randomly selected orbits of this companion
            start_mjd=data_table['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
            data_table=data_table,
            cbar_param="sma2",
            total_mass=system_mass,
            parallax=plx,
            system_rv=sysrv
        )
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

        fig = loaded_results.plot_rvs(
            object_to_plot = object_to_plot, # Plot orbits for the first (and only, in this case) companion
            num_orbits_to_plot= 10, # Will plot 100 randomly selected orbits of this companion
            start_mjd=data_table['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
            data_table=data_table,
            total_mass=system_mass,
            parallax=plx,
            system_rv=sysrv
        )
        # fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'rv_plot_{0}_{1}_{2}_obj1.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
        # fig = loaded_results.plot_rvs(
        #     object_to_plot = 2, # Plot orbits for the first (and only, in this case) companion
        #     num_orbits_to_plot= 100, # Will plot 100 randomly selected orbits of this companion
        #     start_mjd=data_table['epoch'][0], # Minimum MJD for colorbar (here we choose first data epoch)
        #     data_table=data_table
        # )
        # fig.savefig(os.path.join(out_pngs,"HR_8799_"+planet,'rv_plot_{0}_{1}_{2}_obj2.png'.format(rv_str,planet,suffix))) # This is matplotlib.figure.Figure.savefig()
        plt.show()


