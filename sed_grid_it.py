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
import matplotlib.pyplot as plt

def get_data_hr8799c(gridname):
    inputdir = "/data/osiris_data/low_res/HR_8799_c/"

    data = []
    err = []
    model = []

    #import photometry first
    t5 = np.genfromtxt(inputdir+'marois2008_hr8799c.txt')
    M08fluxes = t5[:,0]
    M08err = t5[:,1]
    M08microns = t5[:,2] #[1.248 1.633 1.592 1.681 2.146 3.776]
    filters = ["Keck_NIRC2.J","Keck_NIRC2.H","Gemini_NIRI.CH4short-G0228.dat","Gemini_NIRI.CH4long-G0229.dat","Keck_NIRC2.Ks","Keck_NIRC2.Lp"]
    # J nirc2; H nirc2; CH4s niri, CH4l niri; Ks nirc2; L' nirc2

    print(M08fluxes,M08err,M08microns)
    return 0

    t1 = np.genfromtxt(inputdir+'currie2011_hr8799c.txt')
    C11fluxes = t1[:,0]
    C11err = t1[:,1]
    C11microns = t1[:,2]

    print(C11fluxes,C11err,C11microns)

    t2 = np.genfromtxt(inputdir+'currie2014_hr8799c.txt')
    C14fluxes = t2[:,0]
    C14err = t2[:,1]
    C14microns = t2[:,2]

    # t3 = np.genfromtxt(inputdir+'galicher2011_hr8799d_phot.txt')
    # G11fluxes = t3[:,0]
    # G11err = t3[:,1]
    # G11microns = t3[:,2]

    #galicher 2011
    G11fluxes = 3.36E-16
    G11err = 4.33E-17
    G11microns = 4.67
    #3.36E-16	4.33E-17	4.67	galicher 2011

    t4 = np.genfromtxt(inputdir+'zurlo2016_phot_hr8799c.txt')
    Z16fluxes = t4[:,0]
    Z16err = t4[:,1]
    Z16microns = t4[:,2]

    # t6 = np.genfromtxt(inputdir+'skemer2012_hr8799d.txt')
    # S12fluxes = t6[:,0]
    # S12err = t6[:,1]
    # S12microns = t6[:,2]

    #Skemer 2012
    S12fluxes = 2.47E-15
    S12err = 3.18E-16
    S12microns = 1.633
    #2.47E-15	3.18E-16	1.633	skemer 2012

    t7 = np.genfromtxt(inputdir+'skemer2014_hr8799c.txt')
    S14fluxes = t7[:,0]
    S14err = t7[:,1]
    S14microns = t7[:,2]


    #import spectra next
    t8 = np.genfromtxt(inputdir+'greenbaum2018_hr8799c_H.txt')
    G18Hfluxes = t8[:,1]
    G18Herr = t8[:,2]
    G18Hmicrons = t8[:,0]
    t8 = np.genfromtxt(inputdir+'greenbaum2018_hr8799c_K.txt')
    G18Kfluxes = t8[:,1]
    G18Kerr = t8[:,2]
    G18Kmicrons = t8[:,0]
    t8 = np.genfromtxt(inputdir+'greenbaum2018_hr8799c_K2.txt')
    G18K2fluxes = t8[:,1]
    G18K2err = t8[:,2]
    G18K2microns = t8[:,0]



    #project 1640
    t9 = np.genfromtxt(inputdir+'hr8799c_p1640_spec.txt')
    Pfluxes = t9[:,1] *1.1e-15#(41.3/10)**2
    Perr = t9[:,2] *1.1e-15#(41.3/10)**2
    Pmicrons = t9[:,0]/1000

    #Konopacky 2013
    t10 = np.genfromtxt(inputdir+'hr8799c_kbb_medres_Konopacky2013.txt')
    K13fluxes = t10[:,1]
    K13err = t10[:,2]
    K13microns = t10[:,0]
    K13fluxes = 10**K13fluxes
    K13err = 10**K13err


    return 0

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
        planet = "c"
        # gridname = os.path.join("/data/osiris_data/","hr8799b_modelgrid")
        gridname = os.path.join("/data/osiris_data/","clouds_modelgrid")
        numthreads=32
    else:
        pass

    # tmpfilename = os.path.join(gridname,"hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
    # hdulist = pyfits.open(tmpfilename)
    # planet_model_grid =  hdulist[0].data
    # oriplanet_spec_wvs =  hdulist[1].data
    # Tlistunique =  hdulist[2].data
    # logglistunique =  hdulist[3].data
    # paralistunique =  hdulist[4].data
    # # Tlistunique =  hdulist[1].data
    # # logglistunique =  hdulist[2].data
    # # paralistunique =  hdulist[3].data
    # hdulist.close()
    # 
    # print(planet_model_grid.shape,np.size(Tlistunique),np.size(logglistunique),np.size(paralistunique),np.size(oriplanet_spec_wvs))
    # from scipy.interpolate import RegularGridInterpolator
    # myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,paralistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

    ## generate model grids for each filter
    filters = ["Keck_NIRC2.J.dat","Keck_NIRC2.H.dat","Gemini_NIRI.CH4short-G0228.dat","Gemini_NIRI.CH4long-G0229.dat","Keck_NIRC2.Ks.dat","Keck_NIRC2.Lp.dat"]
        
        
    if "hr8799b_modelgrid" in gridname:
        planet_model_list = []
        grid_filelist = glob.glob(os.path.join(gridname,"lte*-*-0.0.aces_hr8799b_pgs=4d6_Kzz=1d8_C=*_O=*_gs=5um.exoCH4_hiresHK.7.D2e.sorted"))
        # grid_filelist = glob.glob(os.path.join(gridname,"lte12-4.5-0.0.aces_hr8799b_pgs=4d6_Kzz=1d8_C=8.38_O=*_gs=5um.exoCH4_hiresHK.7.D2e.sorted"))
        grid_filelist.sort()

        Tlist = np.array([int(float(os.path.basename(grid_filename).split("lte")[-1].split("-")[0])*100) for grid_filename in grid_filelist])
        logglist = np.array([float(os.path.basename(grid_filename).split("-")[1]) for grid_filename in grid_filelist])
        Clist = np.array([float(os.path.basename(grid_filename).split("C=")[-1].split("_O")[0]) for grid_filename in grid_filelist])
        Olist = np.array([float(os.path.basename(grid_filename).split("O=")[-1].split("_gs")[0]) for grid_filename in grid_filelist])
        CtoOlist = 10**(Clist-Olist)
        Tlistunique = np.unique(Tlist)
        logglistunique = np.unique(logglist)
        paralistunique = np.unique(CtoOlist)
    if "clouds_modelgrid" in gridname:
        planet_model_list = []
        #lte0800-3.0-0.0.aces_pgs=4d6_Kzz=1d8_gs=1um_4osiris.7
        grid_filelist = glob.glob(os.path.join(gridname,"lte*-*-0.0.aces_pgs=*_Kzz=1d8_gs=1um_4osiris.7"))
        # grid_filelist = glob.glob(os.path.join(gridname,"lte1200-4.5-0.0.aces_pgs=4d6_Kzz=1d8_gs=1um_4osiris.7"))
        grid_filelist.sort()

        Tlist = np.array([int(float(os.path.basename(grid_filename).split("lte")[-1].split("-")[0])) for grid_filename in grid_filelist])
        logglist = np.array([float(os.path.basename(grid_filename).split("-")[1]) for grid_filename in grid_filelist])
        pgslist = np.array([float(os.path.basename(grid_filename).split("pgs=")[-1].split("_Kzz")[0].replace("d","e")) for grid_filename in grid_filelist])
        Tlistunique = np.unique(Tlist)
        logglistunique = np.unique(logglist)
        paralistunique = np.unique(pgslist)
    print(Tlistunique,len(Tlistunique))
    print(logglistunique,len(logglistunique))
    print(paralistunique,len(paralistunique))
    print(len(Tlist),len(Tlistunique)*len(logglistunique)*len(paralistunique))
    print(os.path.basename(grid_filelist[0]))
    # exit()

    #print(gridname)
    specpool = mp.Pool(processes=numthreads)
    for file_id,grid_filename in enumerate(grid_filelist):
        # if os.path.basename(grid_filename) == "lte0800-3.0-0.0.aces_pgs=1d6_Kzz=1d8_gs=1um_4osiris.7":
        #     continue
        print(os.path.basename(grid_filename))

        print(grid_filename.replace(".7",".7.D2e.sorted"))
        with open(grid_filename.replace(".7",".7.D2e.sorted"), 'r') as csvfile:
            out = np.loadtxt(grid_filename.replace(".7",".7.D2e.sorted"),skiprows=0)
            # print(np.size(oriplanet_spec_wvs))
            oriplanet_spec_wvs = out[:,0]/1e4
            dwvs = oriplanet_spec_wvs[1::]-oriplanet_spec_wvs[0:np.size(oriplanet_spec_wvs)-1]
            dwvs = np.insert(dwvs,0,dwvs[0])
            print(dwvs)
            oriplanet_spec = 10**(out[:,1]-np.max(out[:,1]))
            # oriplanet_spec = out[:,1]
            oriplanet_spec /= np.nanmean(oriplanet_spec)

            print("convolving: "+grid_filename)

        phot_list = []
        for photfilter in filters:
            print(photfilter)
            filter_arr = np.loadtxt(os.path.join(osiris_data_dir,"filters",photfilter))
            wvs = filter_arr[:,0]/1e4
            trans = filter_arr[:,1]

            wvs_firsthalf = interp1d(trans[0:np.size(wvs)//2],wvs[0:np.size(wvs)//2])
            wvs_secondhalf = interp1d(trans[np.size(wvs)//2::],wvs[np.size(wvs)//2::])
            print(wvs_firsthalf(0.5))
            print(wvs_secondhalf(0.5))

            trans_f = interp1d(wvs,trans,bounds_error=False,fill_value=0)

            # plt.plot(wvs,trans,label=photfilter)
            # plt.plot([wvs_firsthalf(0.5),wvs_secondhalf(0.5)],[0.5,0.5],label=photfilter)

            phot_list.append(np.sum(oriplanet_spec*trans_f(oriplanet_spec_wvs)*dwvs)/np.sum(trans_f(oriplanet_spec_wvs)*dwvs))
            plt.plot(oriplanet_spec_wvs,oriplanet_spec)
            plt.plot([wvs_firsthalf(0.5),wvs_secondhalf(0.5)],[phot_list[-1],phot_list[-1]],label=photfilter)
        plt.legend()
        plt.show()

            # planet_convspec = convolve_spectrum(oriplanet_spec_wvs,oriplanet_spec,R,specpool)
            planet_model_list.append(phot_list)
                    
        planet_model_grid = np.zeros((np.size(Tlistunique),np.size(logglistunique),np.size(paralistunique),np.size(oriplanet_spec_wvs)))
        for T_id,T in enumerate(Tlistunique):
            for logg_id,logg in enumerate(logglistunique):
                for pgs_id,pgs in enumerate(paralistunique):
                    planet_model_grid[T_id,logg_id,pgs_id,:] = planet_model_list[np.where((Tlist==T)*(logglist==logg)*(pgslist==pgs))[0][0]]
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=planet_model_grid))
        hdulist.append(pyfits.ImageHDU(data=filters))
        hdulist.append(pyfits.ImageHDU(data=Tlistunique))
        hdulist.append(pyfits.ImageHDU(data=logglistunique))
        hdulist.append(pyfits.ImageHDU(data=paralistunique))
        try:
            hdulist.writeto(os.path.join(gridname,"hr8799b_modelgrid_{0}.fits".format(photfilter.replace(".dat",""))), overwrite=True)
        except TypeError:
            hdulist.writeto(os.path.join(gridname,"hr8799b_modelgrid_{0}.fits".format(photfilter.replace(".dat",""))), clobber=True)
        # try:
        #     hdulist.writeto(os.path.join(gridname,"hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter)), overwrite=True)
        # except TypeError:
        #     hdulist.writeto(os.path.join(gridname,"hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter)), clobber=True)
        hdulist.close()
        exit()



    plt.legend()
    plt.show()
    exit()

    get_data_hr8799c(gridname)