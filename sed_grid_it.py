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
        IFSfilter = "Kbb"
        # gridname = os.path.join("/data/osiris_data/","hr8799b_modelgrid")
        gridname = os.path.join("/data/osiris_data/","clouds_modelgrid")
    else:
        pass

    get_data_hr8799c(gridname)