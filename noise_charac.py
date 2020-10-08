__author__ = 'jruffio'

# import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
import pyklip.klip as klip
import glob
import os
import pyklip.instruments.osiris as osi
import scipy.linalg as linalg
from scipy.interpolate import UnivariateSpline
import multiprocessing as mp
import itertools
from scipy import interpolate
import random
import scipy.ndimage as ndimage
from scipy.signal import correlate2d
import ctypes
from scipy.ndimage.filters import median_filter
# from scipy.signal import medfilt2d
from copy import copy
from astropy.stats import mad_std
import scipy.io as scio
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import sys
import xml.etree.ElementTree as ET
import csv
import time
from PyAstronomy import pyasl
import scipy.linalg as la
from scipy.sparse.linalg import lsqr
from scipy.sparse import csc_matrix
from scipy.sparse import bsr_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dia_matrix
from scipy.optimize import lsq_linear
#Logic to test mkl exists
from reduce_HPFonly_diagcov_resmodel_v2 import return_64x19,_spline_psf_model,LPFvsHPF,_arraytonumpy
import matplotlib.pyplot as plt
try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

import numpy
import sys
#import asciitable
from astropy.io import ascii
import bisect
# import ipdb
from scipy.interpolate import interp1d


def mass_model(model_name,deltaMag,dist,Smag,age,band):
    """returns a mass model from the cond, btsettl, or dusty
    model_name: 'cond', 'bt_settl', 'dusty'
    deltaMag: the delta magnitude of the planetary companion
    dist: the distance(in parsecs) to the target star
    Smag: the stellar magnitude
    age:the age in Myrs
    band: 'l', 'ks', 'h', 'j', 'm'
    """
    if band.lower()=='l': col='col11'
    if band.lower()=='ks': col='col10'
    if band.lower()=='h': col='col9'
    if band.lower()=='j': col='col8'
    if band.lower()=='m': col='col12'
    if model_name.lower()=='cond':
        age_below, ML_below, mass_below, age_above, ML_above, mass_above=read_model('/data/osiris_data/evol_grids/model.AMES-Cond-2000.M-0.0.NaCo.txt',col,age/1000.)
    if model_name.lower()=='btsettl' or model_name.lower()=='bt-settl' :
        age_below, ML_below, mass_below, age_above, ML_above, mass_above=read_model('/data/osiris_data/evol_grids/model.BT-Settl.M-0.0.NaCo.txt',col,age/1000.)
    if model_name.lower()=='dusty':
        age_below, ML_below, mass_below, age_above, ML_above, mass_above=read_model('/data/osiris_data/evol_grids/model.AMES-dusty.M-0.0.NaCo.txt',col,age/1000.)

    ##convert to age to Myr
    age_below*=1000.
    age_above*=1000.

    ###Corresponding masses
    mass=mass_below
    #This doesn't seem right to me

    ###Adjust for distance (in pc)
    dist_adjust=5.*numpy.log10(dist)-5.
    ML_below+=dist_adjust
    ML_above+=dist_adjust

    ####adjust for stellar magnitude
    ML_below-=Smag
    ML_above-=Smag

    ###3. Need to create L' magnitudes and masses for the age of the star
    fraction=(age-age_below)/(age_above-age_below)

    ML_sub=(ML_above-ML_below)*fraction
    ML_adjust=ML_below+ML_sub

    ##input new magnitude to determine masses
    f=interp1d(ML_adjust[::-1],mass[::-1],bounds_error=False,fill_value=np.nan)
    # print(ML_adjust[::-1],mass[::-1])
    # print(deltaMag)
    # mass_new=np.zeros(deltaMag.shape)+np.nan
    # wherevalid = np.where(np.isfinite(deltaMag)*np.isfinite(deltaMag))
    mass_new=f(deltaMag)

    ### Now convert mass from m/Msun to jupiter masses
    mass_new=mass_new*1047.9

    return mass_new

def dmag_model(model_name,mass2,Smass,age,band):
    """returns a deltaMag model from the cond, btsettl, or dusty
    model_name: 'cond', 'bt_settl', 'dusty'
    mass: the mass of the planetary companion (Solar)
    dist: the distance(in parsecs) to the target star
    Smag: the stellar magnitude
    age:the age in Myrs
    band: 'l', 'ks', 'h', 'j', 'm'
    """
    if band.lower()=='l': col='col11'
    if band.lower()=='ks': col='col10'
    if band.lower()=='h': col='col9'
    if band.lower()=='j': col='col8'
    if band.lower()=='m': col='col12'
    if model_name.lower()=='cond':
        age_below, ML_below, mass_below, age_above, ML_above, mass_above=read_model('/data/osiris_data/evol_grids/model.AMES-Cond-2000.M-0.0.NaCo.txt',col,age/1000.)
    if model_name.lower()=='btsettl' or model_name.lower()=='bt-settl' :
        age_below, ML_below, mass_below, age_above, ML_above, mass_above=read_model('/data/osiris_data/evol_grids/model.BT-Settl.M-0.0.NaCo.txt',col,age/1000.)
    if model_name.lower()=='dusty':
        age_below, ML_below, mass_below, age_above, ML_above, mass_above=read_model('/data/osiris_data/evol_grids/model.AMES-dusty.M-0.0.NaCo.txt',col,age/1000.)

    ##convert to age to Myr
    age_below*=1000.
    age_above*=1000.

    ###Corresponding mass
    mass=mass_below

    ### Need to create L' magnitudes for the age of the star
    fraction=(age-age_below)/(age_above-age_below)

    ML_sub=(ML_above-ML_below)*fraction
    ML_adjust=ML_below+ML_sub

    ##input new mass to determine mag
    f=interp1d(mass[::-1], ML_adjust[::-1])
    dmag_new=f(mass2) - f(Smass)

    return dmag_new


###input the model name, the filter and age you want and the magnitudes and masses will be returned.
def read_model(model_name, filterColNum, exact_age_wanted):

    ###this will return the upper and lower bounds for the age requested, unless the age exactly matches one listed
    ###age in Gyrs

    #model_name='model.BT-Settl.NaCo.txt'
    #filterColNum='col11'
    #age_wanted=0.001####temporarily
    with open(model_name) as f:
        content = f.readlines()

    content=numpy.array(content)
    headers=ascii.read(content[6]) ##can be read, column by column. for example headers['col11'][0] which is L'
    #print "Returning filter ",headers[filterColNum][0]

    ###determine where all the bars are, assuming the appear in threes, the header is between the 1st and 2nd bar,
    ### above the 1st bar is the age and between the 2nd and 3rd bar is the data.

    bars=numpy.where(content==content[5])[0]
    bars=numpy.reshape(bars,(bars.size//3,3)) ###now it is reshaped so that the three bars are associated together
    age=[] ###append the ages since we do not know the length of the file


    ##first read in all the ages and find the one the user has requested
    for bar in bars:
        age.append(float(content[bar[0]-1].split('=')[1])) ###Read the line above the first bar, and store just the part after the equal sign. Convert to float and this is the age.
    age=numpy.array(age)
    if numpy.size(age[age>exact_age_wanted]) > 0:
        age_above=min(age[age>exact_age_wanted])
    else:
        age_above=max(age)
        print("Warning: age greater than max age in grid")

    if numpy.size(age[age<=exact_age_wanted]) > 0:
        age_below=max(age[age<=exact_age_wanted])
    else:
        age_below=min(age)
        print("Warning: age less than min age in grid")

    #age_below = age[age <= exact_age_wanted].max()
    #age_above = age[age >exact_age_wanted].min()
    ########### first for the nearest age
    #age_row=bars[numpy.where(age==nearest_age)[0],0] ##find the correct row to begin in the list, based on the input age
    nums_start0=bars[numpy.where(age==age_below)[0],1]+1
    nums_end0=bars[numpy.where(age==age_below)[0],2]

    ###add from mike
    nums_start0= int(nums_start0[0])
    nums_end0  = int(nums_end0[0])

    nums0=ascii.read(content[nums_start0:nums_end0])

    ########then for the next nearest age

    nums_start1=bars[numpy.where(age==age_above)[0],1]+1
    nums_end1=bars[numpy.where(age==age_above)[0],2]
    ###add from mike
    nums_start1= int(nums_start1[0])
    nums_end1  = int(nums_end1[0])
    nums1=ascii.read(content[nums_start1:nums_end1])

    ########Finally check if the mass column matches in both cases
    mass_below=nums0['col1']
    mass_above=nums1['col1']
    ML_below=nums0[filterColNum]
    ML_above=nums1[filterColNum]


    if numpy.array_equal(mass_below,mass_above)==0:
        #print "in if statement1"
        overlap=numpy.in1d(mass_below,mass_above)
        ML_below=ML_below[overlap]
        mass_below=mass_below[overlap]
    if numpy.array_equal(mass_above,mass_below)==0:
        #print "in if statement2"
        overlap=numpy.in1d(mass_above,mass_below)
        ML_above=ML_above[overlap]
        mass_above=mass_above[overlap]


    ###return the ages, magnitudes and the corresponding masses

    return age_below, ML_below, mass_below, age_above, ML_above, mass_above

def age_mass_to_mag_Sonora_hotstart(ageMyr,mass_list,band,tefflooginterpgrid= None):
    Msun = 1.989e30 #kg
    Mjup = 1.898e27 #kg
    if tefflooginterpgrid is None:
        mags_filename = "/data/osiris_data/sonora//Sonora_Bobcat_Tables/photometry_tables/mag_table+0.0_nostar"
        mags_data = np.loadtxt(mags_filename, skiprows=11)
        mags_data_Teff_logg = mags_data[:, 0:2]
        uniqueTeff = np.unique(mags_data[:, 0])
        uniquelogg = np.unique(mags_data[:, 1])
        # exit()
        mags_data_Rnorm = mags_data[:, 3]
        # col 17 Keck Lp
        if band == "Lp":
            mags_data_Lp = mags_data[:, 17]  # Lp
        elif band == "Ms":
            mags_data_Lp = mags_data[:,18] #M
        elif band == "Ks":
            mags_data_Lp = mags_data[:,16] #K
        from scipy.interpolate import RegularGridInterpolator
        tefflooginterpgrid = RegularGridInterpolator((uniqueTeff, uniquelogg),
                                               np.concatenate([np.reshape(mags_data_Rnorm,(len(uniquelogg), len(uniqueTeff))).T[:, :, None],
                                                               np.reshape(mags_data_Lp,(len(uniquelogg), len(uniqueTeff))).T[:, :,None]], axis=2),
                                               bounds_error=False,
                                               fill_value=np.nan)

    evol_filename = "/data/osiris_data/sonora//Sonora_Bobcat_Tables/evolution_tables/evo_tables+0.0/nc+0.0_co1.0_mass"
    data_chunks_lines = []
    with open(evol_filename) as openfileobject:
        for linenum, line in enumerate(openfileobject):
            if "  " == line[0:2]:
                data_chunks_lines.append((linenum, int(line)))
    evol_data = np.zeros((len(data_chunks_lines), 6))
    for k,(firstline, nline) in enumerate(data_chunks_lines):
        evol_data_chunk = np.loadtxt(evol_filename, skiprows=firstline + 1, max_rows=nline)
        for l in np.arange(6):
            if l == 1:
                evol_data[k, l] = ageMyr
            else:
                f = interp1d(np.log10(evol_data_chunk[:,1])+3,evol_data_chunk[:,l],kind="cubic",bounds_error=False, fill_value=np.nan)
                evol_data[k,l] = f(np.log10(ageMyr))
    # converting mass to Mjup
    evol_data[:, 0] = evol_data[:, 0] * Msun / Mjup
    evol_uniquemass = np.unique(evol_data[:, 0])

    evol_Lpmag = np.zeros(evol_data.shape[0])
    for k, row in enumerate(evol_data):
        Teff = row[3]
        logg = row[4]
        Rnorm0 = row[5]
        Rnorm1, Lpmag1 = tefflooginterpgrid([Teff, logg])[0]
        evol_Lpmag[k] = Lpmag1 - 5 * np.log10(Rnorm1 / Rnorm0)

    whereLpmagfinite = np.where(np.isfinite(evol_Lpmag))
    f = interp1d(evol_data[:, 0][whereLpmagfinite],evol_Lpmag[whereLpmagfinite], kind="cubic", bounds_error=False,fill_value=np.nan)
    maxmass = np.nanmax(evol_data[:, 0][whereLpmagfinite])
    minmass = np.nanmin(evol_data[:, 0][whereLpmagfinite])

    outLpMag = f(mass_list)
    outLpMag[np.where(mass_list>maxmass)] = -np.inf
    outLpMag[np.where(mass_list<minmass)] = np.inf
    return outLpMag

def age_mag_to_mass_Sonora_hotstart(ageMyr,mag_list,band,tefflooginterpgrid= None):
    Msun = 1.989e30 #kg
    Mjup = 1.898e27 #kg
    if tefflooginterpgrid is None:
        mags_filename = "/data/osiris_data/sonora//Sonora_Bobcat_Tables/photometry_tables/mag_table+0.0_nostar"
        mags_data = np.loadtxt(mags_filename, skiprows=11)
        mags_data_Teff_logg = mags_data[:, 0:2]
        uniqueTeff = np.unique(mags_data[:, 0])
        uniquelogg = np.unique(mags_data[:, 1])
        # exit()
        mags_data_Rnorm = mags_data[:, 3]
        # col 17 Keck Lp
        if band == "Lp":
            mags_data_Lp = mags_data[:, 17]  # Lp
        elif band == "Ms":
            mags_data_Lp = mags_data[:,18] #M
        elif band == "Ks":
            mags_data_Lp = mags_data[:,16] #K
        from scipy.interpolate import RegularGridInterpolator
        tefflooginterpgrid = RegularGridInterpolator((uniqueTeff, uniquelogg),
                                               np.concatenate([np.reshape(mags_data_Rnorm,(len(uniquelogg), len(uniqueTeff))).T[:, :, None],
                                                               np.reshape(mags_data_Lp,(len(uniquelogg), len(uniqueTeff))).T[:, :,None]], axis=2),
                                               bounds_error=False,
                                               fill_value=np.nan)

    evol_filename = "/data/osiris_data/sonora//Sonora_Bobcat_Tables/evolution_tables/evo_tables+0.0/nc+0.0_co1.0_mass"
    data_chunks_lines = []
    with open(evol_filename) as openfileobject:
        for linenum, line in enumerate(openfileobject):
            if "  " == line[0:2]:
                data_chunks_lines.append((linenum, int(line)))
    evol_data = np.zeros((len(data_chunks_lines), 6))
    for k,(firstline, nline) in enumerate(data_chunks_lines):
        evol_data_chunk = np.loadtxt(evol_filename, skiprows=firstline + 1, max_rows=nline)
        for l in np.arange(6):
            if l == 1:
                evol_data[k, l] = ageMyr
            else:
                f = interp1d(np.log10(evol_data_chunk[:,1])+3,evol_data_chunk[:,l],kind="cubic",bounds_error=False, fill_value=np.nan)
                evol_data[k,l] = f(np.log10(ageMyr))
    # converting mass to Mjup
    evol_data[:, 0] = evol_data[:, 0] * Msun / Mjup
    evol_uniquemass = np.unique(evol_data[:, 0])

    evol_Lpmag = np.zeros(evol_data.shape[0])
    for k, row in enumerate(evol_data):
        Teff = row[3]
        logg = row[4]
        Rnorm0 = row[5]
        Rnorm1, Lpmag1 = tefflooginterpgrid([Teff, logg])[0]
        evol_Lpmag[k] = Lpmag1 - 5 * np.log10(Rnorm1 / Rnorm0)

    whereLpmagfinite = np.where(np.isfinite(evol_Lpmag))
    f = interp1d(evol_Lpmag[whereLpmagfinite],evol_data[:, 0][whereLpmagfinite], kind="cubic", bounds_error=False,fill_value=np.nan)
    maxmass = np.nanmax(evol_data[:, 0][whereLpmagfinite])
    minmass = np.nanmin(evol_data[:, 0][whereLpmagfinite])

    outmass = f(mag_list)
    outmass[np.where(outmass>maxmass)] = -np.inf
    outmass[np.where(outmass<minmass)] = np.inf
    return outmass
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

    planet = "HR_8799_d"
    # date = "200729"
    # date = "200730"
    # date = "200731"
    date = "200803"
    # IFSfilter = "Kbb"
    IFSfilter = "Kbb"
    # IFSfilter = "Jbb" # "Kbb" or "Hbb"
    scale = "020"
    # scale = "035"
    imnum,ply,plx = 16,40,11
    cutoff=40
    fontsize = 12

    inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/"
    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"

    if 1:
        rot = -42
        fov_ravec = np.array([np.cos(np.deg2rad(rot)),-np.sin(np.deg2rad(rot))])
        fov_decvec= np.array([np.sin(np.deg2rad(rot)),np.cos(np.deg2rad(rot))])
        print(fov_ravec)
        print(fov_decvec)
        print(fov_ravec*1076*10+fov_decvec*544*10)
        print(np.sqrt(np.sum((fov_ravec*1076*10)**2+(fov_decvec*544*10)**2)))
        exit()

        e_ra = -308
        e_dec = 232
        d_ra = -546.5
        d_dec = -428.8
        c_ra =  -373
        c_dec = 865.5
        print(np.rad2deg(np.arctan2(0,0)))
        print(np.rad2deg(np.arctan2(1,0)))

        f_ra =  -225
        f_dec = 52.5
        # f_sep = 250
        # f_pa = 270
        # f_ra = f_sep * np.sin(np.deg2rad(f_pa))
        # f_dec = f_sep * np.cos(np.deg2rad(f_pa))

        print(np.rad2deg(np.arctan2(c_ra,c_dec)) % 360)
        print(np.rad2deg(np.arctan2(d_ra,d_dec)) % 360)
        # print(np.rad2deg(np.arctan2(e_ra,e_dec)) % 360)
        # print(np.rad2deg(np.arctan2(f_ra,f_dec)) % 360)
        #
        # print(np.sqrt(c_ra**2+c_dec**2))
        # print(np.sqrt(d_ra**2+d_dec**2))
        # print(np.sqrt(e_ra**2+e_dec**2))
        # print(np.sqrt(f_ra**2+f_dec**2))

        print(np.rad2deg(np.arctan2((c_ra+d_ra)/2.,(c_dec+d_dec)/2.)) % 360)
        print(np.rad2deg(np.arctan2((c_ra-d_ra)/2.,(c_dec-d_dec)/2.)) % 360)
        # print(np.rad2deg(np.arctan2(4,64)) % 360)
        # exit()
        # cen_ra = (f_ra+d_ra)/2.
        # cen_dec = 0
        # rot = 0

        rot = 7.6-(np.rad2deg(np.arctan2(4,64)) % 360)
        fov_ravec = np.array([np.cos(np.deg2rad(rot)),-np.sin(np.deg2rad(rot))])
        fov_decvec= np.array([np.sin(np.deg2rad(rot)),np.cos(np.deg2rad(rot))])
        cen_ra = (c_ra+d_ra)/2. + 65*fov_ravec[0]+ 8*35*fov_decvec[0]
        cen_dec = (c_dec+d_dec)/2. + 65*fov_ravec[1] + 8*35*fov_decvec[1]
        # cen_ra = (f_ra+d_ra)/2.
        # cen_dec = 0

        # cen_ra = c_ra
        # cen_dec = c_dec
        # rot = 66
        # cen_ra = d_ra
        # cen_dec = d_dec
        # rot = -39


        plt.scatter([0],[0])
        plt.scatter([c_ra,d_ra,e_ra,f_ra],[c_dec,d_dec,e_dec,f_dec])
        plt.xlim([-1500,1500])
        plt.ylim([-1500,1500])

        w = 18*35
        l = 63*35
        print(fov_ravec)
        print(fov_decvec)
        plt.plot([cen_ra+fov_ravec[0]*w/2.+fov_decvec[0]*l/2.,cen_ra+fov_ravec[0]*w/2.-fov_decvec[0]*l/2],
                 [cen_dec+fov_ravec[1]*w/2.+fov_decvec[1]*l/2.,cen_dec+fov_ravec[1]*w/2.-fov_decvec[1]*l/2],color="red",linestyle="--")
        plt.plot([cen_ra-fov_ravec[0]*w/2.+fov_decvec[0]*l/2.,cen_ra-fov_ravec[0]*w/2.-fov_decvec[0]*l/2],
                 [cen_dec-fov_ravec[1]*w/2.+fov_decvec[1]*l/2.,cen_dec-fov_ravec[1]*w/2.-fov_decvec[1]*l/2],color="red",linestyle="--")
        plt.plot([cen_ra+fov_ravec[0]*w/2.+fov_decvec[0]*l/2.,cen_ra+fov_ravec[0]*(w/2.-3*35)-fov_decvec[0]*l/2],
                 [cen_dec+fov_ravec[1]*w/2.+fov_decvec[1]*l/2.,cen_dec+fov_ravec[1]*(w/2.-3*35)-fov_decvec[1]*l/2],color="pink",linestyle="-")
        plt.plot([cen_ra-fov_ravec[0]*(w/2.-4*35)+fov_decvec[0]*l/2.,cen_ra-fov_ravec[0]*w/2.-fov_decvec[0]*l/2],
                 [cen_dec-fov_ravec[1]*(w/2.-4*35)+fov_decvec[1]*l/2.,cen_dec-fov_ravec[1]*w/2.-fov_decvec[1]*l/2],color="pink",linestyle="-")

        plt.gca().annotate("FOV: dra={0:.0f} ddec={1:.0f} rot={2:.1f}".format(cen_ra,cen_dec,rot),
                           xy=(1000,-1000), va="bottom", ha="left", fontsize=fontsize, color="black")
        plt.gca().invert_xaxis()
        plt.gca().set_aspect("equal")
        plt.show()
        exit()


    if 0: # JWST spectra
        # filename = "/data/JWST/nirspec/HR2562_G395H:F290LP_R2700/cube/cube_reconstructed.fits"
        # filename = "/data/JWST/nirspec/HR2562_G395H:F290LP_R2700/cube/cube_reconstructed_snr.fits"
        filename = "/data/JWST/nirspec/eps_Eri/eps_Eri_wb52793/cube/cube_reconstructed_signal.fits"
        with pyfits.open(filename) as hdulist:
            cube = hdulist[0].data
        filename = "/data/JWST/nirspec/eps_Eri/eps_Eri_satspot_wb52793/cube/cube_reconstructed_signal.fits"
        with pyfits.open(filename) as hdulist:
            cube_satspot = hdulist[0].data
        cal_cube = cube_satspot-cube
        filename = "/data/JWST/nirspec/eps_Eri/eps_Eri_wb52793/cube/cube_reconstructed_snr.fits"
        with pyfits.open(filename) as hdulist:
            cube_snr = hdulist[0].data
        filename = "/data/JWST/nirspec/eps_Eri/eps_Eri_wb52793/cube/cube_reconstructed_noise.fits"
        with pyfits.open(filename) as hdulist:
            cube_noise = hdulist[0].data
        filename = "/data/JWST/nirspec/eps_Eri/eps_Eri_wb52793/cube/cube_reconstructed_saturation.fits"
        with pyfits.open(filename) as hdulist:
            cube_satur = hdulist[0].data
        im_satur = np.nanmax(cube_satur,axis=0)
        where_satur = np.where(im_satur==2)
        filename = "/data/JWST/nirspec/eps_Eri/eps_Eri_wb52793/lineplot/lineplot_wave_pix.fits"
        wvs = pyfits.getdata(filename).WAVELENGTH
        id = np.argmin(np.abs(wvs-4.5))
        cal_im = np.nanmedian(cal_cube,axis=0)
        ny,nx = cal_im.shape
        yc,xc = ny//2,nx//2
        x_psf_vec, y_psf_vec = np.arange(nx)-xc, np.arange(ny)-yc
        x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
        r_grid = np.sqrt(x_psf_grid**2+y_psf_grid**2)*0.1
        where_out =np.where(r_grid>1)

        ypsfc,xpsfc = np.unravel_index(np.argmax(cal_im),cal_im.shape)
        psf_cube = cal_cube[:,ypsfc-1:ypsfc+2,xpsfc-1:xpsfc+2] * 10**((1.67-9.5)/-2.5)

        cube_wpl = copy(cube)
        plt.subplot(1,2,1)
        tmp = np.nanmedian(psf_cube,axis=(1,2))
        plt.plot(wvs,tmp/np.nanmean(tmp),label="psf")
        tmp = np.nanmedian(cube[:,where_out[0],where_out[1]],axis=1)
        plt.plot(wvs,tmp/np.nanmedian(tmp),label="data")
        plt.legend()
        plt.subplot(1,2,2)
        plt.imshow(im_satur)
        plt.show()

        epserib_Mcont = 2.5e-6
        numthreads = 10
        specpool = mp.Pool(processes=numthreads)
        filename = "/data/osiris_data/sonora/sp_t250g31nc_m0.0"
        spec_arr = np.loadtxt(filename, skiprows=2)
        wvs,spec = spec_arr[::-1,0],spec_arr[::-1,1]/1e-8
        wherewvs = np.where((wvs>2.87-1)*(wvs<5.27+1))
        wvs,spec = wvs[wherewvs],spec[wherewvs]
        plt.plot(wvs, spec,label="From Mark original")
        from reduce_HPFonly_diagcov_resmodel_v2 import convolve_spectrum
        R = 2700
        planet_convspec = convolve_spectrum(wvs, spec,R,specpool)

        pl_cube = psf_cube/np.nansum(psf_cube,axis=(1,2))[:,None,None]*planet_convspec
        pl_cube = pl_cube/np.nansum(pl_cube)*np.nansum(psf_cube)*epserib_Mcont

        # print(yc,xc)
        # exit()
        print(cube.shape)
        im = cube[id,:,:]#np.nanmedian(cube,axis=0)
        # sig = im/cube_snr[id,:,:]
        sig = cube_noise[id,:,:]
        sig = sig/np.nanmax(im)
        im/=np.nanmax(im)
        im[np.where(im<=0)] = np.nan
        im[np.where(r_grid>1.5)] = np.nan
        sig[np.where(im<=0)] = np.nan
        sig[np.where(r_grid>1.5)] = np.nan

        psf_sep_list = np.array(np.ravel(r_grid))
        psf_value_list = np.array(np.ravel(im))
        sig_value_list = np.array(np.ravel(sig))
        dsep=0.1
        binnedpsf_sep = np.arange(0,2.,dsep)
        binnedpsf_value = np.zeros(binnedpsf_sep.shape)
        binnedpsf_sig = np.zeros(binnedpsf_sep.shape)
        for sepid,sep in enumerate(binnedpsf_sep):
            wherebin = np.where((psf_sep_list<sep+dsep/2)*(psf_sep_list>sep-dsep/2))
            if len(wherebin[0]) != 0:
                binnedpsf_value[sepid]=np.nanmedian(psf_value_list[wherebin])
                binnedpsf_sig[sepid]=np.nanmedian(sig_value_list[wherebin])
            else:
                binnedpsf_value[sepid] = np.nan
                binnedpsf_sig[sepid] = np.nan
        plt.plot(binnedpsf_sep,binnedpsf_value,linestyle="-",linewidth=3,label="4.5mum PSF profile HR2562",color="red")
        plt.plot(binnedpsf_sep,binnedpsf_sig,linestyle="-",linewidth=3,label="1-sigma",color="blue")
        plt.yscale("log")
        plt.xlim([0,2])
        plt.ylim([1e-7,1e-0])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Planet to star flux ratio",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize*0.75)

        plt.figure(2)
        filename = "/data/osiris_data/sonora/sp_t250g31nc_m0.0"
        spec_arr = np.loadtxt(filename, skiprows=2)
        wvs,spec = spec_arr[::-1,0],spec_arr[::-1,1]/1e-8
        wherewvs = np.where((wvs>2.87-1)*(wvs<5.27+1))
        wvs,spec = wvs[wherewvs],spec[wherewvs]
        plt.plot(wvs, spec,label="From Mark original")
        from reduce_HPFonly_diagcov_resmodel_v2 import convolve_spectrum
        R = 2700
        planet_convspec = convolve_spectrum(wvs, spec,R,specpool)
        plt.plot(wvs, planet_convspec,label="R=2700")
        R = 1000
        planet_convspec = convolve_spectrum(wvs, spec,R,specpool)
        print(np.mean(planet_convspec))
        plt.plot(wvs, planet_convspec,label="R=1000")
        R = 100
        planet_convspec = convolve_spectrum(wvs, spec,R,specpool)
        print(np.mean(planet_convspec))
        plt.plot(wvs, planet_convspec,label="R=100")
        # plt.xlim([1,30])
        plt.legend()
        plt.show()
        print(wvs,spec )
        exit()



    if 0: # residuals
        planet = "HR_8799_d"
        date = "200731"
        inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/"
        ply,plx = 39,9
        # filename = "/data/osiris_data/HR_8799_b/20200803/reduced_jb/20200914_res/s200803_a034002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl10_res.fits"
        filename = "/data/osiris_data/HR_8799_b/20200803/reduced_jb/20200914_res/s200803_a034002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_rescalc_res.fits"
        with pyfits.open(filename) as hdulist:
            res = np.nanstd(hdulist[0].data[0,0,2,:,10:64+10-10,10:19+10-10],axis=0)
            fl = np.nanmedian(hdulist[0].data[0,0,3,:,10:64+10-10,10:19+10-10]+hdulist[0].data[0,0,4,:,10:64+10-10,10:19+10-10],axis=0)
            plt.scatter(fl,res,s = 2)
        # filename = "/data/osiris_data/HR_8799_c/20200729/reduced_jb/20200914_res/s200729_a036002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl10_res.fits"
        filename = "/data/osiris_data/HR_8799_c/20200729/reduced_jb/20200914_res/s200729_a036002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_rescalc_res.fits"
        with pyfits.open(filename) as hdulist:
            res = np.nanstd(hdulist[0].data[0,0,2,:,10:64+10-10,10:19+10-10],axis=0)
            fl = np.nanmedian(hdulist[0].data[0,0,3,:,10:64+10-10,10:19+10-10]+hdulist[0].data[0,0,4,:,10:64+10-10,10:19+10-10],axis=0)
            plt.scatter(fl,res,s = 2)
        # filename = "/data/osiris_data/HR_8799_d/20200731/reduced_jb/20200914_res/s200731_a028002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl10_res.fits"
        filename = "/data/osiris_data/HR_8799_d/20200731/reduced_jb/20200914_res/s200731_a028002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_rescalc_res.fits"
        with pyfits.open(filename) as hdulist:
            res = np.nanstd(hdulist[0].data[0,0,2,:,10:64+10-10,10:19+10-10],axis=0)
            fl = np.nanmedian(hdulist[0].data[0,0,3,:,10:64+10-10,10:19+10-10]+hdulist[0].data[0,0,4,:,10:64+10-10,10:19+10-10],axis=0)
            plt.scatter(fl,res,s = 2)


        # sky_nodarksub_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_nosub","s"+date+"*{0}*".format(9)+IFSfilter+"_"+scale+".fits"))
        # sky2_nodarksub_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_nosub","s"+date+"*{0}*".format(22)+IFSfilter+"_"+scale+".fits"))
        # filelist = [sky_nodarksub_filelist[0],
        #             sky2_nodarksub_filelist[0]]
        # # label_list = ["sky w/o dark sub","sky w/ dark sub","science w/o sky sub","science w/ sky sub"]
        # mycube_list = []
        # myvec_list = []
        # for filename in filelist:
        #     print(filename)
        #     with pyfits.open(filename) as hdulist:
        #         prihdr = hdulist[0].header
        #         curr_mjdobs = prihdr["MJD-OBS"]
        #         imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        #         imgs = return_64x19(imgs)
        #         imgs = np.moveaxis(imgs,0,2)
        #         imgs_hdrbadpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
        #         imgs_hdrbadpix = return_64x19(imgs_hdrbadpix)
        #         imgs_hdrbadpix = np.moveaxis(imgs_hdrbadpix,0,2)
        #         imgs_hdrbadpix = imgs_hdrbadpix.astype(dtype=ctypes.c_double)
        #         imgs_hdrbadpix[np.where(imgs_hdrbadpix==0)] = np.nan
        #         imgs[np.where(imgs_hdrbadpix==0)] = 0
        #     mycube_list.append(imgs)
        # delta_sky = mycube_list[1]-mycube_list[0]
        # window_size=100
        # threshold=5
        # for k in range(imgs.shape[0]):
        #     print(k)
        #     for l in range(imgs.shape[1]):
        #         delta_sky_nodarksub = delta_sky[k,l,:]
        #         smooth_vec = median_filter(delta_sky_nodarksub,footprint=np.ones(window_size),mode="reflect")
        #         _myvec = delta_sky_nodarksub - smooth_vec
        #         wherefinite = np.where(np.isfinite(_myvec))
        #         mad = mad_std(_myvec[wherefinite])
        #         whereoutliers = np.where(np.abs(_myvec)>threshold*mad)[0]
        #         delta_sky_nodarksub[whereoutliers] = np.nan
        #         delta_sky[k,l,:] = LPFvsHPF(delta_sky_nodarksub,cutoff=cutoff)[1]
        #
        # delta_sky_std = np.nanstd(delta_sky,axis=2)
        # med_sky = np.nanmedian(delta_sky_std)
        # print(med_sky)

        med_stdsky = 0.016679049
        x = np.linspace(0,10,100)
        plt.plot(x,np.sqrt(med_stdsky**2+0.025**2*x**2),"--",color="black",label="calib limited")
        plt.plot(x,np.sqrt(med_stdsky**2+0.037**2*x),"-",color="black",label="photon limited")
        plt.plot(x,0*x+med_stdsky,":",color="gray",label="sigma_background")
        # plt.yscale("log")
        plt.xlim([0,5])
        plt.ylim([0,0.1])
        plt.legend(loc="upper left")

        plt.xlabel("star flux (arb. unit)")
        plt.ylabel("sigma_res")

        plt.gca().annotate("calib limited: sigma_res $\propto$ sqrt(sigma_background**2+ g0*star_flux**2)",
                           xy=(0,0), va="bottom", ha="left", fontsize=fontsize, color="black")
        plt.gca().annotate("photon limited: sigma_res $\propto$ sqrt(sigma_background**2+ g1*star_flux)",
                           xy=(0,0.005), va="bottom", ha="left", fontsize=fontsize, color="black")
        plt.show()


    if 0: # SNR vs narrow band filters
        # ["#006699","#ff9900","#6600ff"]
        ply,plx = 32,11
        filter_list = ["Kbb","Kn5","Kn4","Kn3","Kn2","Kn1"]
        for k,whichfilter in enumerate(filter_list):
            filename = "/data/osiris_data/HR_8799_c/20100715/reduced_jb/20200910_narrowfilters/s100715_a010001_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl10_{0}.fits".format(whichfilter)
            with pyfits.open(filename) as hdulist:
                arr = hdulist[0].data[0,0,13-3,0,ply,plx]
                plt.plot(k,arr,"x",color="#ff9900")
        ply,plx = 39,9
        for k,whichfilter in enumerate(filter_list):
            filename = "/data/osiris_data/HR_8799_d/20200731/reduced_jb/20200910_narrowfilters/s200731_a028002_Kbb_020_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl10_{0}.fits".format(whichfilter)
            with pyfits.open(filename) as hdulist:
                arr = hdulist[0].data[0,0,13-3,0,ply,plx]
                plt.plot(k,arr,"o",color="#6600ff")
        plt.plot(0,-100,"x",color="#ff9900",label="HR 8799 c")
        plt.plot(0,-100,"o",color="#6600ff",label="HR 8799 d")
        plt.xticks(np.arange(len(filter_list)),filter_list)
        plt.ylim([0,25])
        plt.ylabel("S/N")
        plt.legend()
        plt.show()



    if 0:
        fig = plt.figure(6,figsize=(6,4))
        psf_filelist = glob.glob(os.path.join(inputDir,"..","reduced_telluric_jb","HR_8799","s"+date+"*"+IFSfilter+"_"+scale+".fits"))
        print(psf_filelist)
        myim_list = []
        psf_sep_list = []
        value_sep_list = []
        for filename in psf_filelist:
            print(filename)
            with pyfits.open(filename) as hdulist:
                prihdr = hdulist[0].header
                curr_mjdobs = prihdr["MJD-OBS"]
                imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                imgs = return_64x19(imgs)
                imgs = np.moveaxis(imgs,0,2)
                imgs_hdrbadpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
                imgs_hdrbadpix = return_64x19(imgs_hdrbadpix)
                imgs_hdrbadpix = np.moveaxis(imgs_hdrbadpix,0,2)
                imgs_hdrbadpix = imgs_hdrbadpix.astype(dtype=ctypes.c_double)
                imgs_hdrbadpix[np.where(imgs_hdrbadpix==0)] = np.nan
                imgs[np.where(imgs_hdrbadpix==0)] = 0
            ny,nx,nz = imgs.shape
            init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
            dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
            wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

            wvid = np.argmin(np.abs(wvs-2.3))

            im = np.nanmedian(imgs,axis=2)
            im/=np.nanmax(im)
            ny,nx = im.shape
            ymax,xmax = np.unravel_index(np.argmax(im),im.shape)
            x_psf_vec, y_psf_vec = np.arange(nx)-xmax, np.arange(ny)-ymax
            x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
            r_grid = np.sqrt(x_psf_grid**2+y_psf_grid**2)*0.02
            im[np.where(im<=0)] = np.nan
            im[np.where(r_grid>0.8)] = np.nan
            im_ravel = np.ravel(im)
            where_fin = np.where(np.isfinite(im_ravel))
            im_ravel = im_ravel[where_fin]
            r_ravel = np.ravel(r_grid)[where_fin]
            myim_list.append(im)
            plt.scatter(r_ravel,im_ravel,s=5,alpha=0.1,c="black")
            psf_sep_list.extend(r_ravel)
            value_sep_list.extend(im_ravel)
        plt.yscale("log")
        plt.ylim([1e-4,1])
        plt.xlim([0,0.6])
        plt.ylabel("Normalized PSF")
        plt.xlabel(r"Separation (arcsec)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        # plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        # fig.savefig(os.path.join(out_pngs,"OSIRIS_PSF.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        # fig.savefig(os.path.join(out_pngs,"OSIRIS_PSF.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        plt.show()
        exit()

    if 1:# contrast curves
        HR8799_Lmag = 5.194
        HR8799_cde_dLmag = -2.5*np.log10(2e-4)
        HR8799_Kmag = 5.24
        HR8799_cde_dKmag = 10.86
        HR8799_plx = 24.2175#25.38 # [mas]
        HR8799_dist = 1/(HR8799_plx/1000)
        HR8799_LMag = HR8799_Lmag-5*np.log10(HR8799_dist)+5
        HR8799_KMag = HR8799_Kmag-5*np.log10(HR8799_dist)+5
        HR8799_cde_LMag = HR8799_LMag+HR8799_cde_dLmag
        HR8799_cde_KMag = HR8799_KMag+HR8799_cde_dKmag
        age = 30
        print(-2.5*np.log10(2e-4)+HR8799_LMag)
        print(mass_model("btsettl",-2.5*np.log10(2e-4),10,HR8799_LMag,age,"l"))
        print(-2.5*np.log10(4.5e-5)+HR8799_KMag)
        print(mass_model("btsettl",-2.5*np.log10(4.5e-5),10,HR8799_KMag,age,"ks"))
        # exit()

        psf_sep_list = []
        psf_value_list = []
        if 1: #HR 8799 d
            sc_filelist = glob.glob("/data/osiris_data/HR_8799_d/20200731/reduced_jb/s200731*Kbb_020.fits")
            sc_sep_list = []
            sc_value_list = []
            fileinfos_filename = os.path.join("/data/osiris_data/HR_8799_d","fileinfos_Kbb_jb_kl{0}.csv".format(10))
            if 1:
                with open(fileinfos_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=';')
                    list_table = list(csv_reader)
                    colnames = list_table[0]
                    N_col = len(colnames)
                    list_data = list_table[1::]
                    N_lines =  len(list_data)
                kcen_id = colnames.index("kcen")
                lcen_id = colnames.index("lcen")
                filename_id = colnames.index("filename")
                status_id = colnames.index("status")
                filelist = [os.path.basename(item[filename_id]) for item in list_data]

            if 1:
                refstar_name_filter = "*"
                ref_star_folder = os.path.join(os.path.dirname(sc_filelist[0]),"..","reduced_telluric_jb")
                fileinfos_refstars_filename = os.path.join("/data/osiris_data","fileinfos_refstars_jb.csv")
                with open(fileinfos_refstars_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=';')
                    refstarsinfo_list_table = list(csv_reader)
                    refstarsinfo_colnames = refstarsinfo_list_table[0]
                    refstarsinfo_list_data = refstarsinfo_list_table[1::]
                refstarsinfo_filename_id = refstarsinfo_colnames.index("filename")
                refstarsinfo_filelist = [os.path.basename(item[refstarsinfo_filename_id]) for item in refstarsinfo_list_data]
                type_id = refstarsinfo_colnames.index("type")
                Jmag_id = refstarsinfo_colnames.index("Jmag")
                Hmag_id = refstarsinfo_colnames.index("Hmag")
                Kmag_id = refstarsinfo_colnames.index("Kmag")
                rv_simbad_id = refstarsinfo_colnames.index("RV Simbad")
                starname_id = refstarsinfo_colnames.index("star name")
                psfs_rep4flux_filelist = glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"s*"+IFSfilter+"_"+scale+"_psfs_repaired_v2.fits"))
                psfs_rep4flux_filelist.sort()
                hr8799_flux_list = []
                for psfs_rep4flux_filename in psfs_rep4flux_filelist:
                    for refstar_fileid,refstarsinfo_file in enumerate(refstarsinfo_filelist):
                        if os.path.basename(refstarsinfo_file).replace(".fits","") in psfs_rep4flux_filename:
                            fileitem = refstarsinfo_list_data[refstar_fileid]
                            break
                    refstar_RV = float(fileitem[rv_simbad_id])
                    ref_star_type = fileitem[type_id]
                    if IFSfilter == "Jbb":
                        refstar_mag = float(fileitem[Jmag_id])
                    elif IFSfilter == "Hbb":
                        refstar_mag = float(fileitem[Hmag_id])
                    elif IFSfilter == "Kbb":
                        refstar_mag = float(fileitem[Kmag_id])
                        host_mag = 4.34
                    with pyfits.open(psfs_rep4flux_filename) as hdulist:
                        psfs_repaired = hdulist[0].data
                        bbflux = np.nanmedian(np.nanmax(psfs_repaired,axis=(1,2)))
                    hr8799_flux_list.append(bbflux* 10**(-1./2.5*(host_mag-refstar_mag)))

                print(hr8799_flux_list)
                hr8799_flux = np.mean(hr8799_flux_list)
                print(hr8799_flux)

            for filename in sc_filelist:
                print(filename)
                fileid = filelist.index(os.path.basename(filename))
                fileitem = list_data[fileid]
                status,plcen_k,plcen_l = int(fileitem[status_id]),float(fileitem[kcen_id]),float(fileitem[lcen_id])
                if status != 1:
                    continue
                print(status,plcen_k,plcen_l)


                # exit()
                with pyfits.open(filename) as hdulist:
                    prihdr = hdulist[0].header
                    curr_mjdobs = prihdr["MJD-OBS"]
                    imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                    imgs = return_64x19(imgs)
                    imgs = np.moveaxis(imgs,0,2)
                    imgs_hdrbadpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
                    imgs_hdrbadpix = return_64x19(imgs_hdrbadpix)
                    imgs_hdrbadpix = np.moveaxis(imgs_hdrbadpix,0,2)
                    imgs_hdrbadpix = imgs_hdrbadpix.astype(dtype=ctypes.c_double)
                    imgs_hdrbadpix[np.where(imgs_hdrbadpix==0)] = np.nan
                    imgs[np.where(imgs_hdrbadpix==0)] = 0
                ny,nx,nz = imgs.shape
                init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
                dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
                wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

                wvid = np.argmin(np.abs(wvs-2.3))

                im = np.nanmedian(imgs,axis=2)/hr8799_flux
                ny,nx = im.shape
                x_psf_vec, y_psf_vec = np.arange(nx)-(plcen_l-34.65), np.arange(ny)-plcen_k
                x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
                r_grid = np.sqrt(x_psf_grid**2+y_psf_grid**2)*0.02
                im[np.where(im<=0)] = np.nan
                im[np.where(r_grid>0.75)] = np.nan
                im[np.where(r_grid<0.5)] = np.nan
                im_ravel = np.ravel(im)
                where_fin = np.where(np.isfinite(im_ravel))
                im_ravel = im_ravel[where_fin]
                r_ravel = np.ravel(r_grid)[where_fin]
                psf_sep_list.extend(r_ravel)
                psf_value_list.extend(im_ravel)

        if 1: #HR 8799 c
            sc_filelist = glob.glob("/data/osiris_data/HR_8799_c/20200729/reduced_jb/s200729*Kbb_020.fits")
            sc_sep_list = []
            sc_value_list = []
            fileinfos_filename = os.path.join("/data/osiris_data/HR_8799_c","fileinfos_Kbb_jb_kl{0}.csv".format(10))
            if 1:
                with open(fileinfos_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=';')
                    list_table = list(csv_reader)
                    colnames = list_table[0]
                    N_col = len(colnames)
                    list_data = list_table[1::]
                    N_lines =  len(list_data)
                kcen_id = colnames.index("kcen")
                lcen_id = colnames.index("lcen")
                filename_id = colnames.index("filename")
                status_id = colnames.index("status")
                filelist = [os.path.basename(item[filename_id]) for item in list_data]

            if 1:
                refstar_name_filter = "*"
                ref_star_folder = os.path.join(os.path.dirname(sc_filelist[0]),"..","reduced_telluric_jb")
                fileinfos_refstars_filename = os.path.join("/data/osiris_data","fileinfos_refstars_jb.csv")
                with open(fileinfos_refstars_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=';')
                    refstarsinfo_list_table = list(csv_reader)
                    refstarsinfo_colnames = refstarsinfo_list_table[0]
                    refstarsinfo_list_data = refstarsinfo_list_table[1::]
                refstarsinfo_filename_id = refstarsinfo_colnames.index("filename")
                refstarsinfo_filelist = [os.path.basename(item[refstarsinfo_filename_id]) for item in refstarsinfo_list_data]
                type_id = refstarsinfo_colnames.index("type")
                Jmag_id = refstarsinfo_colnames.index("Jmag")
                Hmag_id = refstarsinfo_colnames.index("Hmag")
                Kmag_id = refstarsinfo_colnames.index("Kmag")
                rv_simbad_id = refstarsinfo_colnames.index("RV Simbad")
                starname_id = refstarsinfo_colnames.index("star name")
                psfs_rep4flux_filelist = glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"s*"+IFSfilter+"_"+scale+"_psfs_repaired_v2.fits"))
                psfs_rep4flux_filelist.sort()
                hr8799_flux_list = []
                for psfs_rep4flux_filename in psfs_rep4flux_filelist:
                    for refstar_fileid,refstarsinfo_file in enumerate(refstarsinfo_filelist):
                        if os.path.basename(refstarsinfo_file).replace(".fits","") in psfs_rep4flux_filename:
                            fileitem = refstarsinfo_list_data[refstar_fileid]
                            break
                    refstar_RV = float(fileitem[rv_simbad_id])
                    ref_star_type = fileitem[type_id]
                    if IFSfilter == "Jbb":
                        refstar_mag = float(fileitem[Jmag_id])
                    elif IFSfilter == "Hbb":
                        refstar_mag = float(fileitem[Hmag_id])
                    elif IFSfilter == "Kbb":
                        refstar_mag = float(fileitem[Kmag_id])
                        host_mag = 4.34
                    with pyfits.open(psfs_rep4flux_filename) as hdulist:
                        psfs_repaired = hdulist[0].data
                        bbflux = np.nanmedian(np.nanmax(psfs_repaired,axis=(1,2)))
                    hr8799_flux_list.append(bbflux* 10**(-1./2.5*(host_mag-refstar_mag)))

                print(hr8799_flux_list)
                hr8799_flux = np.mean(hr8799_flux_list)
                print(hr8799_flux)

            for filename in sc_filelist:
                print(filename)
                fileid = filelist.index(os.path.basename(filename))
                fileitem = list_data[fileid]
                status,plcen_k,plcen_l = int(fileitem[status_id]),float(fileitem[kcen_id]),float(fileitem[lcen_id])
                if status != 1:
                    continue
                print(status,plcen_k,plcen_l)


                # exit()
                with pyfits.open(filename) as hdulist:
                    prihdr = hdulist[0].header
                    curr_mjdobs = prihdr["MJD-OBS"]
                    imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                    imgs = return_64x19(imgs)
                    imgs = np.moveaxis(imgs,0,2)
                    imgs_hdrbadpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
                    imgs_hdrbadpix = return_64x19(imgs_hdrbadpix)
                    imgs_hdrbadpix = np.moveaxis(imgs_hdrbadpix,0,2)
                    imgs_hdrbadpix = imgs_hdrbadpix.astype(dtype=ctypes.c_double)
                    imgs_hdrbadpix[np.where(imgs_hdrbadpix==0)] = np.nan
                    imgs[np.where(imgs_hdrbadpix==0)] = 0
                ny,nx,nz = imgs.shape
                init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
                dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
                wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

                wvid = np.argmin(np.abs(wvs-2.3))

                im = np.nanmedian(imgs,axis=2)/hr8799_flux
                ny,nx = im.shape
                x_psf_vec, y_psf_vec = np.arange(nx)-(plcen_l-47.15), np.arange(ny)-plcen_k
                x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
                r_grid = np.sqrt(x_psf_grid**2+y_psf_grid**2)*0.02
                im[np.where(im<=0)] = np.nan
                im[np.where(r_grid>1.5)] = np.nan
                # im[np.where(r_grid<0.5)] = np.nan
                im_ravel = np.ravel(im)
                where_fin = np.where(np.isfinite(im_ravel))
                im_ravel = im_ravel[where_fin]
                r_ravel = np.ravel(r_grid)[where_fin]
                psf_sep_list.extend(r_ravel)
                psf_value_list.extend(im_ravel)

        if 1: #HR 8799 b
            sc_filelist = glob.glob("/data/osiris_data/HR_8799_b/20200803/reduced_jb/s200803*Kbb_020.fits")
            sc_sep_list = []
            sc_value_list = []
            fileinfos_filename = os.path.join("/data/osiris_data/HR_8799_b","fileinfos_Kbb_jb_kl{0}.csv".format(10))
            if 1:
                with open(fileinfos_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=';')
                    list_table = list(csv_reader)
                    colnames = list_table[0]
                    N_col = len(colnames)
                    list_data = list_table[1::]
                    N_lines =  len(list_data)
                kcen_id = colnames.index("kcen")
                lcen_id = colnames.index("lcen")
                filename_id = colnames.index("filename")
                status_id = colnames.index("status")
                filelist = [os.path.basename(item[filename_id]) for item in list_data]

            if 1:
                refstar_name_filter = "*"
                ref_star_folder = os.path.join(os.path.dirname(sc_filelist[0]),"..","reduced_telluric_jb")
                fileinfos_refstars_filename = os.path.join("/data/osiris_data","fileinfos_refstars_jb.csv")
                with open(fileinfos_refstars_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=';')
                    refstarsinfo_list_table = list(csv_reader)
                    refstarsinfo_colnames = refstarsinfo_list_table[0]
                    refstarsinfo_list_data = refstarsinfo_list_table[1::]
                refstarsinfo_filename_id = refstarsinfo_colnames.index("filename")
                refstarsinfo_filelist = [os.path.basename(item[refstarsinfo_filename_id]) for item in refstarsinfo_list_data]
                type_id = refstarsinfo_colnames.index("type")
                Jmag_id = refstarsinfo_colnames.index("Jmag")
                Hmag_id = refstarsinfo_colnames.index("Hmag")
                Kmag_id = refstarsinfo_colnames.index("Kmag")
                rv_simbad_id = refstarsinfo_colnames.index("RV Simbad")
                starname_id = refstarsinfo_colnames.index("star name")
                psfs_rep4flux_filelist = glob.glob(os.path.join(ref_star_folder,refstar_name_filter,"s*"+IFSfilter+"_"+scale+"_psfs_repaired_v2.fits"))
                psfs_rep4flux_filelist.sort()
                hr8799_flux_list = []
                for psfs_rep4flux_filename in psfs_rep4flux_filelist:
                    for refstar_fileid,refstarsinfo_file in enumerate(refstarsinfo_filelist):
                        if os.path.basename(refstarsinfo_file).replace(".fits","") in psfs_rep4flux_filename:
                            fileitem = refstarsinfo_list_data[refstar_fileid]
                            break
                    refstar_RV = float(fileitem[rv_simbad_id])
                    ref_star_type = fileitem[type_id]
                    if IFSfilter == "Jbb":
                        refstar_mag = float(fileitem[Jmag_id])
                    elif IFSfilter == "Hbb":
                        refstar_mag = float(fileitem[Hmag_id])
                    elif IFSfilter == "Kbb":
                        refstar_mag = float(fileitem[Kmag_id])
                        host_mag = 4.34
                    with pyfits.open(psfs_rep4flux_filename) as hdulist:
                        psfs_repaired = hdulist[0].data
                        bbflux = np.nanmedian(np.nanmax(psfs_repaired,axis=(1,2)))
                    hr8799_flux_list.append(bbflux* 10**(-1./2.5*(host_mag-refstar_mag)))

                print(hr8799_flux_list)
                hr8799_flux = np.mean(hr8799_flux_list)
                print(hr8799_flux)

            for filename in sc_filelist:
                print(filename)
                fileid = filelist.index(os.path.basename(filename))
                fileitem = list_data[fileid]
                status,plcen_k,plcen_l = int(fileitem[status_id]),float(fileitem[kcen_id]),float(fileitem[lcen_id])
                if status != 1:
                    continue
                print(status,plcen_k,plcen_l)


                # exit()
                with pyfits.open(filename) as hdulist:
                    prihdr = hdulist[0].header
                    curr_mjdobs = prihdr["MJD-OBS"]
                    imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                    imgs = return_64x19(imgs)
                    imgs = np.moveaxis(imgs,0,2)
                    imgs_hdrbadpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
                    imgs_hdrbadpix = return_64x19(imgs_hdrbadpix)
                    imgs_hdrbadpix = np.moveaxis(imgs_hdrbadpix,0,2)
                    imgs_hdrbadpix = imgs_hdrbadpix.astype(dtype=ctypes.c_double)
                    imgs_hdrbadpix[np.where(imgs_hdrbadpix==0)] = np.nan
                    imgs[np.where(imgs_hdrbadpix==0)] = 0
                ny,nx,nz = imgs.shape
                init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
                dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
                wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

                wvid = np.argmin(np.abs(wvs-2.3))

                im = np.nanmedian(imgs,axis=2)/hr8799_flux
                ny,nx = im.shape
                x_psf_vec, y_psf_vec = np.arange(nx)-(plcen_l+86), np.arange(ny)-plcen_k
                x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
                r_grid = np.sqrt(x_psf_grid**2+y_psf_grid**2)*0.02
                im[np.where(im<=0)] = np.nan
                # im[np.where(r_grid>0.75)] = np.nan
                im[np.where(r_grid<1.65)] = np.nan
                im_ravel = np.ravel(im)
                where_fin = np.where(np.isfinite(im_ravel))
                im_ravel = im_ravel[where_fin]
                r_ravel = np.ravel(r_grid)[where_fin]
                psf_sep_list.extend(r_ravel)
                psf_value_list.extend(im_ravel)
        # dsep=0.02
        # psf_sep_list = np.array(psf_sep_list)
        # psf_value_list = np.array(psf_value_list)
        # binnedpsf_sep = np.arange(0,0.75,dsep)
        # binnedpsf_value = np.zeros(binnedpsf_sep.shape)
        # for sepid,sep in enumerate(binnedpsf_sep):
        #     wherebin = np.where((psf_sep_list<sep+dsep/2)*(psf_sep_list>sep-dsep/2))
        #     if len(wherebin[0]) != 0:
        #         binnedpsf_value[sepid]=np.nanmedian(psf_value_list[wherebin])
        #     else:
        #         binnedpsf_value[sepid] = np.nan
        #
        # plt.title("K-band")
        # plt.plot(binnedpsf_sep,binnedpsf_value,linestyle="-",linewidth=3,label="OSIRIS unocculted PSF K",color="red")
        # plt.yscale("log")
        # plt.xlim([0,1])
        # plt.ylim([1e-5,1e-0])
        # plt.xlabel("Separation (as)",fontsize=fontsize)
        # plt.ylabel("Planet to star flux ratio",fontsize=fontsize)
        # plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)
        # plt.legend(loc="upper right",frameon=True,fontsize=fontsize*0.75)
        # plt.show()


        if 1: #PSF
            psf_filelist = glob.glob("/data/osiris_data/HR_8799_d/20200731/reduced_telluric_jb/HR_8799/s200731*Kbb_020.fits")
            # psf_filelist.extend(glob.glob("/data/osiris_data/HR_8799_d/20200803/reduced_telluric_jb/HR_8799/s200803*Kbb_020.fits"))
            # psf_filelist = glob.glob("/data/osiris_data/HR_8799_d/20200730/reduced_telluric_jb/HR_8799/s200730*Kbb_020.fits")
            # psf_filelist = glob.glob("/data/osiris_data/HR_8799_d/20200729/reduced_telluric_jb/HR_8799/s200729*Kbb_020.fits")
            for filename in psf_filelist:
                print(filename)
                with pyfits.open(filename) as hdulist:
                    prihdr = hdulist[0].header
                    curr_mjdobs = prihdr["MJD-OBS"]
                    imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                    imgs = return_64x19(imgs)
                    imgs = np.moveaxis(imgs,0,2)
                    imgs_hdrbadpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
                    imgs_hdrbadpix = return_64x19(imgs_hdrbadpix)
                    imgs_hdrbadpix = np.moveaxis(imgs_hdrbadpix,0,2)
                    imgs_hdrbadpix = imgs_hdrbadpix.astype(dtype=ctypes.c_double)
                    imgs_hdrbadpix[np.where(imgs_hdrbadpix==0)] = np.nan
                    imgs[np.where(imgs_hdrbadpix==0)] = 0
                ny,nx,nz = imgs.shape
                init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
                dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
                wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

                wvid = np.argmin(np.abs(wvs-2.3))

                im = np.nanmedian(imgs,axis=2)
                im/=np.nanmax(im)
                ny,nx = im.shape
                ymax,xmax = np.unravel_index(np.argmax(im),im.shape)
                x_psf_vec, y_psf_vec = np.arange(nx)-xmax, np.arange(ny)-ymax
                x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)
                r_grid = np.sqrt(x_psf_grid**2+y_psf_grid**2)*0.02
                im[np.where(im<=0)] = np.nan
                im[np.where(r_grid>0.4)] = np.nan
                im_ravel = np.ravel(im)
                where_fin = np.where(np.isfinite(im_ravel))
                im_ravel = im_ravel[where_fin]
                r_ravel = np.ravel(r_grid)[where_fin]
                psf_sep_list.extend(r_ravel)
                psf_value_list.extend(im_ravel)

        psf_sep_list = np.array(psf_sep_list)
        psf_value_list = np.array(psf_value_list)
        dsep=0.02
        binnedpsf_sep = np.arange(0,2.,dsep)
        binnedpsf_value = np.zeros(binnedpsf_sep.shape)
        for sepid,sep in enumerate(binnedpsf_sep):
            wherebin = np.where((psf_sep_list<sep+dsep/2)*(psf_sep_list>sep-dsep/2))
            if len(wherebin[0]) != 0:
                binnedpsf_value[sepid]=np.nanmedian(psf_value_list[wherebin])
            else:
                binnedpsf_value[sepid] = np.nan

        # ["#006699","#ff9900","#6600ff"]
        loD2as = 0.079
        nirc2_L_raw = np.loadtxt('/data/osiris_data/noise_charac/4jb-raw-phot.txt')
        nirc2_L_raw_sep = nirc2_L_raw[:,0]*loD2as
        nirc2_L_raw_cont = nirc2_L_raw[:,1]
        nirc2_L_raw_cont[np.where(nirc2_L_raw_sep<0.079)] = np.nan
        nirc2_L_raw_label = "NIRC2 L Raw - 25s"
        nirc2_L_raw_plotparas = ["--","#006699"]
        nirc2_L_nocoro = np.loadtxt('/data/osiris_data/noise_charac/4jb-nocoro.txt')
        nirc2_L_nocoro_sep = nirc2_L_nocoro[:,0]*loD2as
        nirc2_L_nocoro_cont = nirc2_L_nocoro[:,1]
        nirc2_L_nocoro_cont[np.where(nirc2_L_nocoro_sep<0.079)] = np.nan
        nirc2_L_nocoro_label = "NIRC2 L No coro (best) - 90 min"
        nirc2_L_nocoro_plotparas = [":","#ff9900"]
        nirc2_L_coro = np.loadtxt('/data/osiris_data/noise_charac/4jb-coro400.txt')
        nirc2_L_coro_sep = nirc2_L_coro[:,0]*loD2as
        nirc2_L_coro_cont = nirc2_L_coro[:,1]
        nirc2_L_coro_cont[np.where(nirc2_L_coro_sep<0.079)] = np.nan
        nirc2_L_coro_label = "NIRC2 L 90min Coro (avg+) - 90 min"
        nirc2_L_coro_plotparas = [":","#6600ff"]

        plt.figure(10,figsize=(12,12))
        plt.subplot(2,2,1)
        plt.title("K-band")
        plt.plot(binnedpsf_sep,binnedpsf_value,linestyle=nirc2_L_raw_plotparas[0],linewidth=3,label="OSIRIS unocculted PSF K",color=nirc2_L_raw_plotparas[1])
        whered=np.argmin(np.abs(binnedpsf_sep-0.7))
        OSIRIS_FMnoise_cont = binnedpsf_value/30
        OSIRIS_FMnoise_cont = OSIRIS_FMnoise_cont/OSIRIS_FMnoise_cont[whered]*4.5e-5
        OSIRIS_FMnoise_label = "OSIRIS HR8799d modeling noise limited - 10 min"
        OSIRIS_FMnoise_plotparas = ["-","red"]
        plt.plot(binnedpsf_sep,OSIRIS_FMnoise_cont,linestyle=OSIRIS_FMnoise_plotparas[0],linewidth=3,label=OSIRIS_FMnoise_label,color=OSIRIS_FMnoise_plotparas[1])
        binnedpsf_value_at_700mas = binnedpsf_value[whered]
        OSIRIS_photnoise_cont = np.sqrt(binnedpsf_value/binnedpsf_value_at_700mas)*(binnedpsf_value_at_700mas/30)
        OSIRIS_photnoise_cont = OSIRIS_photnoise_cont/OSIRIS_photnoise_cont[whered]*4.5e-5
        OSIRIS_photnoise_label = "OSIRIS HR8799d photon noise limited - 10 min"
        OSIRIS_photnoise_plotparas = ["-","blue"]
        OSIRIS_photnoise90_cont = OSIRIS_photnoise_cont/np.sqrt(9)
        OSIRIS_photnoise90_label = "OSIRIS HR8799d photon noise limited - 90 min"
        OSIRIS_photnoise90_plotparas = ["--","cyan"]
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_cont,linestyle=OSIRIS_photnoise_plotparas[0],linewidth=3,label=OSIRIS_photnoise_label,color=OSIRIS_photnoise_plotparas[1])
        plt.plot(binnedpsf_sep,OSIRIS_photnoise90_cont,linestyle=OSIRIS_photnoise90_plotparas[0],linewidth=3,label=OSIRIS_photnoise90_label,color=OSIRIS_photnoise90_plotparas[1])
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim([0.05,2])
        plt.ylim([1e-6,1e-0])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Planet to star flux ratio",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize*0.75)
        # plt.show()


        plt.subplot(2,2,2)
        plt.title("L-band")
        plt.plot(nirc2_L_raw_sep,nirc2_L_raw_cont,linestyle=nirc2_L_raw_plotparas[0],linewidth=3,label=nirc2_L_raw_label,color=nirc2_L_raw_plotparas[1])
        plt.plot(nirc2_L_nocoro_sep,nirc2_L_nocoro_cont,linestyle=nirc2_L_nocoro_plotparas[0],linewidth=3,label=nirc2_L_nocoro_label,color=nirc2_L_nocoro_plotparas[1])
        plt.plot(nirc2_L_coro_sep,nirc2_L_coro_cont,linestyle=nirc2_L_coro_plotparas[0],linewidth=3,label=nirc2_L_coro_label,color=nirc2_L_coro_plotparas[1])
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim([0.05,2])
        plt.ylim([1e-5,1e-2])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Planet to star flux ratio",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize*0.75)
        plt.gca().annotate("From Thompson W. et al.", xy=(0,1e-5), va="bottom", ha="left", fontsize=fontsize, color="black")
        # plt.tight_layout()


        plt.subplot(2,2,3)
        # plt.figure(10,figsize=(6,6))
        plt.title("2 Myr")
        # plt.plot(nirc2_L_raw_sep,age_mag_to_mass_Sonora_hotstart(30,nirc2_L_raw_cont,"Lp",tefflooginterpgrid= None),linestyle="-",linewidth=3,label=nirc2_L_raw_label)
        gridname = "btsettl"
        # gridname = "cond"
        age = 2
        if 1:
            nirc2_L_nocoro_mass = mass_model(gridname,-2.5*np.log10(nirc2_L_nocoro_cont),10,HR8799_LMag,age,"l")
            nirc2_L_coro_mass = mass_model(gridname,-2.5*np.log10(nirc2_L_coro_cont),10,HR8799_LMag,age,"l")
            OSIRIS_FMnoise_mass = mass_model(gridname,-2.5*np.log10(OSIRIS_FMnoise_cont),10,HR8799_KMag,age,"ks")
            OSIRIS_photnoise_mass = mass_model(gridname,-2.5*np.log10(OSIRIS_photnoise_cont),10,HR8799_KMag,age,"ks")
            OSIRIS_photnoise90_mass = mass_model(gridname,-2.5*np.log10(OSIRIS_photnoise90_cont),10,HR8799_KMag,age,"ks")
            plt.gca().annotate(gridname, xy=(0.1,0), va="bottom", ha="left", fontsize=fontsize, color="black")
        if 0:
            nirc2_L_nocoro_mass = age_mag_to_mass_Sonora_hotstart(age,HR8799_LMag-2.5*np.log10(nirc2_L_nocoro_cont),"Lp",tefflooginterpgrid= None)
            nirc2_L_coro_mass = age_mag_to_mass_Sonora_hotstart(age,HR8799_LMag-2.5*np.log10(nirc2_L_coro_cont),"Lp",tefflooginterpgrid= None)
            OSIRIS_FMnoise_mass = age_mag_to_mass_Sonora_hotstart(age,HR8799_KMag-2.5*np.log10(OSIRIS_FMnoise_cont),"Ks",tefflooginterpgrid= None)
            OSIRIS_photnoise_mass = age_mag_to_mass_Sonora_hotstart(age,HR8799_KMag-2.5*np.log10(OSIRIS_photnoise_cont),"Ks",tefflooginterpgrid= None)
            OSIRIS_photnoise90_mass = age_mag_to_mass_Sonora_hotstart(age,HR8799_KMag-2.5*np.log10(OSIRIS_photnoise90_cont),"Ks",tefflooginterpgrid= None)
            plt.gca().annotate("Sonora Bobcat - Marley et al.", xy=(0.1,0), va="bottom", ha="left", fontsize=fontsize, color="black")
        plt.plot(nirc2_L_nocoro_sep,nirc2_L_nocoro_mass,
                 linestyle=nirc2_L_nocoro_plotparas[0],linewidth=3,label=nirc2_L_nocoro_label,color=nirc2_L_nocoro_plotparas[1])
        plt.plot(nirc2_L_coro_sep,nirc2_L_coro_mass,
                 linestyle=nirc2_L_coro_plotparas[0],linewidth=3,label=nirc2_L_coro_label,color=nirc2_L_coro_plotparas[1])
        plt.plot(binnedpsf_sep,OSIRIS_FMnoise_mass,
                 linestyle=OSIRIS_FMnoise_plotparas[0],linewidth=3,label=OSIRIS_FMnoise_label,color=OSIRIS_FMnoise_plotparas[1])
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_mass,
                 linestyle=OSIRIS_photnoise_plotparas[0],linewidth=3,label=OSIRIS_photnoise_label,color=OSIRIS_photnoise_plotparas[1])
        plt.plot(binnedpsf_sep,OSIRIS_photnoise90_mass,
                 linestyle=OSIRIS_photnoise90_plotparas[0],linewidth=3,label=OSIRIS_photnoise90_label,color=OSIRIS_photnoise90_plotparas[1])
        plt.xscale("log")
        plt.xlim([0.05,2])
        plt.ylim([0,12])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Mass ($M_{Jup}$)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize*0.75)

        plt.subplot(2,2,4)
        plt.title("30 Myr")
        # plt.plot(nirc2_L_raw_sep,age_mag_to_mass_Sonora_hotstart(30,nirc2_L_raw_cont,"Lp",tefflooginterpgrid= None),linestyle="-",linewidth=3,label=nirc2_L_raw_label)
        age = 30
        if 1:
            nirc2_L_nocoro_mass = mass_model(gridname,-2.5*np.log10(nirc2_L_nocoro_cont),10,HR8799_LMag,age,"l")
            nirc2_L_coro_mass = mass_model(gridname,-2.5*np.log10(nirc2_L_coro_cont),10,HR8799_LMag,age,"l")
            OSIRIS_FMnoise_mass = mass_model(gridname,-2.5*np.log10(OSIRIS_FMnoise_cont),10,HR8799_KMag,age,"ks")
            OSIRIS_photnoise_mass = mass_model(gridname,-2.5*np.log10(OSIRIS_photnoise_cont),10,HR8799_KMag,age,"ks")
            OSIRIS_photnoise90_mass = mass_model(gridname,-2.5*np.log10(OSIRIS_photnoise90_cont),10,HR8799_KMag,age,"ks")
            plt.gca().annotate(gridname, xy=(0.1,0), va="bottom", ha="left", fontsize=fontsize, color="black")
        if 0:
            nirc2_L_nocoro_mass = age_mag_to_mass_Sonora_hotstart(age,HR8799_LMag-2.5*np.log10(nirc2_L_nocoro_cont),"Lp",tefflooginterpgrid= None)
            nirc2_L_coro_mass = age_mag_to_mass_Sonora_hotstart(age,HR8799_LMag-2.5*np.log10(nirc2_L_coro_cont),"Lp",tefflooginterpgrid= None)
            OSIRIS_FMnoise_mass = age_mag_to_mass_Sonora_hotstart(age,HR8799_KMag-2.5*np.log10(OSIRIS_FMnoise_cont),"Ks",tefflooginterpgrid= None)
            OSIRIS_photnoise_mass = age_mag_to_mass_Sonora_hotstart(age,HR8799_KMag-2.5*np.log10(OSIRIS_photnoise_cont),"Ks",tefflooginterpgrid= None)
            OSIRIS_photnoise90_mass = age_mag_to_mass_Sonora_hotstart(age,HR8799_KMag-2.5*np.log10(OSIRIS_photnoise90_cont),"Ks",tefflooginterpgrid= None)
            plt.gca().annotate("Sonora Bobcat - Marley et al.", xy=(0.1,0), va="bottom", ha="left", fontsize=fontsize, color="black")
        plt.plot(nirc2_L_nocoro_sep,nirc2_L_nocoro_mass,
                 linestyle=nirc2_L_nocoro_plotparas[0],linewidth=3,label=nirc2_L_nocoro_label,color=nirc2_L_nocoro_plotparas[1])
        plt.plot(nirc2_L_coro_sep,nirc2_L_coro_mass,
                 linestyle=nirc2_L_coro_plotparas[0],linewidth=3,label=nirc2_L_coro_label,color=nirc2_L_coro_plotparas[1])
        plt.plot(binnedpsf_sep,OSIRIS_FMnoise_mass,
                 linestyle=OSIRIS_FMnoise_plotparas[0],linewidth=3,label=OSIRIS_FMnoise_label,color=OSIRIS_FMnoise_plotparas[1])
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_mass,
                 linestyle=OSIRIS_photnoise_plotparas[0],linewidth=3,label=OSIRIS_photnoise_label,color=OSIRIS_photnoise_plotparas[1])
        plt.plot(binnedpsf_sep,OSIRIS_photnoise90_mass,
                 linestyle=OSIRIS_photnoise90_plotparas[0],linewidth=3,label=OSIRIS_photnoise90_label,color=OSIRIS_photnoise90_plotparas[1])

        plt.xscale("log")
        # plt.yscale("log")
        plt.xlim([0.05,2])
        plt.ylim([0,20])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Mass ($M_{Jup}$)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize*0.75)


        plt.figure(11,figsize=(6,4))
        #"#006699","#ff9900"
        plt.plot(nirc2_L_nocoro_sep,nirc2_L_nocoro_cont,linestyle="-",linewidth=3,label="NIRC2 L-band (90 min; Thompson W.)",color="#006699")
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_cont,linestyle="-",linewidth=1,label="OSIRIS K-band (10 min)",color="#ff9900")
        plt.plot(binnedpsf_sep,OSIRIS_photnoise90_cont,linestyle="--",linewidth=1,label="OSIRIS K-band (predicted 90 min)",color="#ff9900")
        plt.plot(binnedpsf_sep,OSIRIS_photnoise90_cont/np.sqrt(30)*np.sqrt(9),linestyle=":",linewidth=1,label="OSIRIS K-band (predicted 300 min)",color="#ff9900")
        # Hr 8799
        sep_list = [0.25,0.384,0.693,0.942,0.27,0.7]
        contK_list = [2.5e-5,4.5e-5,4.5e-5,4.5e-5,1.7e-4,4.3e-3]
        contL_list = [8e-5,2e-4,2e-4,2e-4,6.9e-4,5.5e-3]
        name_list = ["HR 8799 f","HR 8799 e","HR 8799 d","HR 8799 c","HD 206893 b","GQ Lup b"]
        for sep,contL,contK,name in zip(sep_list,contL_list,contK_list,name_list):
            plt.plot([sep], [contK],marker="x",linestyle="",markeredgecolor="#ff9900",markerfacecolor="#ff9900")
            plt.plot([sep], [contL],marker="o",linestyle="",markeredgecolor="#006699",markerfacecolor="#006699")
            plt.plot([sep,sep], [contK,contL],marker="",linestyle="-",linewidth=0.5,color="gray")
            plt.gca().annotate(name, xy=(sep+0.01,contL), va="bottom", ha="left", fontsize=fontsize*0.5, color="black")

        # plt.plot([0.1720],)
        plt.yscale("log")
        plt.xlim([0.05,2])
        plt.ylim([1e-6,1e-2])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Planet to star flux ratio",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="lower right",frameon=True,fontsize=fontsize*0.75)
        # plt.show()



        plt.figure(12,figsize=(6,4))
        #"#006699","#ff9900"
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_cont/np.sqrt(10**((0-5.2)/-2.5)),linestyle="--",linewidth=2,label="Kmag = 0 (scaled)",color="#ff9900")
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_cont,linestyle="-",linewidth=2,label="Kmag = 5.2; OSIRIS K-band (10 min)",color="#ff9900")
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_cont/np.sqrt(10**((10-5.2)/-2.5)),linestyle=":",linewidth=2,label="Kmag = 10 (scaled)",color="#ff9900")
        # plt.plot([0.1720],)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim([0.05,2])
        plt.ylim([1e-6,1e-2])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Planet to star flux ratio",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize*0.75)
        plt.tight_layout()


        plt.figure(13,figsize=(12,5))
        #"#006699","#ff9900"
        plt.subplot(1,2,1)
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_cont/np.sqrt(10**((6.5-5.2)/-2.5)),linestyle="-",linewidth=2,label="Kmag = 6.5 (scaled; 10 min)",color="#ff9900")
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_cont/np.sqrt(10**((6.5-5.2)/-2.5))/np.sqrt(6),linestyle="--",linewidth=2,label="Kmag = 6.5 (scaled; 60 min)",color="#ff9900")
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_cont/np.sqrt(10**((6.5-5.2)/-2.5))/np.sqrt(30),linestyle=":",linewidth=2,label="Kmag = 6.5 (scaled; 300 min)",color="#ff9900")
        # plt.plot(binnedpsf_sep,OSIRIS_photnoise_cont,linestyle="-",linewidth=2,label="Kmag = 5.2; OSIRIS K-band (10 min)",color="#ff9900")
        # plt.plot(binnedpsf_sep,OSIRIS_photnoise_cont/np.sqrt(10**((10-5.2)/-2.5)),linestyle=":",linewidth=2,label="Kmag = 10 (scaled)",color="#ff9900")
        # plt.plot([0.1720],)
        plt.yscale("log")
        plt.xscale("log")
        plt.xlim([0.05,2])
        plt.ylim([1e-6,1e-2])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Planet to star flux ratio",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize*0.75)
        plt.subplot(1,2,2)
        tmp = mass_model(gridname,-2.5*np.log10(OSIRIS_photnoise_cont/np.sqrt(10**((0-5.2)/-2.5))),7.68,0,455,"ks")
        plt.plot(binnedpsf_sep,tmp,linestyle="-",linewidth=2,label="btsettl; Kmag = 0 (scaled; 10 min)",color="#ff9900")
        tmp = mass_model(gridname,-2.5*np.log10(OSIRIS_photnoise_cont/np.sqrt(10**((0-5.2)/-2.5))/np.sqrt(30)),7.68,0,455,"ks")
        plt.plot(binnedpsf_sep,tmp,linestyle="--",linewidth=2,label="btsettl; Kmag = 0 (scaled; 300 min)",color="#ff9900")
        plt.xscale("log")
        plt.xlim([0.05,2])
        plt.ylim([0,80])
        plt.xlabel("Separation (as)",fontsize=fontsize)
        plt.ylabel("Planet mass ($M_\mathrm{Jup}$)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize*0.75)

        plt.tight_layout()

        # plt.subplot(2,2,3)
        plt.figure(14,figsize=(8,6))
        # ["#006699","#ff9900","#6600ff"]
        # plt.title("2 Myr")
        # plt.plot(nirc2_L_raw_sep,age_mag_to_mass_Sonora_hotstart(30,nirc2_L_raw_cont,"Lp",tefflooginterpgrid= None),linestyle="-",linewidth=3,label=nirc2_L_raw_label)
        gridname = "btsettl"
        # gridname = "cond"
        age = 2
        if 1:
            nirc2_L_nocoro_mass = mass_model(gridname,-2.5*np.log10(nirc2_L_nocoro_cont),10,HR8799_LMag,age,"l")
            nirc2_L_coro_mass = mass_model(gridname,-2.5*np.log10(nirc2_L_coro_cont),10,HR8799_LMag,age,"l")
            OSIRIS_FMnoise_mass = mass_model(gridname,-2.5*np.log10(OSIRIS_FMnoise_cont),10,HR8799_KMag,age,"ks")
            OSIRIS_photnoise_mass = mass_model(gridname,-2.5*np.log10(OSIRIS_photnoise_cont),10,HR8799_KMag,age,"ks")
            OSIRIS_photnoise60_mass = mass_model(gridname,-2.5*np.log10(OSIRIS_photnoise_cont/np.sqrt(6)),10,HR8799_KMag,age,"ks")
        plt.plot(nirc2_L_nocoro_sep,nirc2_L_nocoro_mass,
                 linestyle="--",linewidth=3,label="NIRC2 L (90 min; no coro; best)",color="#006699")
        plt.plot(nirc2_L_coro_sep,nirc2_L_coro_mass,
                 linestyle="-",linewidth=3,label="NIRC2 L (90 min; coro; avg+)",color="#006699")
        plt.plot(binnedpsf_sep,OSIRIS_FMnoise_mass,
                 linestyle=":",linewidth=3,label="OSIRIS K (10 min; model limited)",color="#ff9900")
        plt.plot(binnedpsf_sep,OSIRIS_photnoise_mass,
                 linestyle="--",linewidth=3,label="OSIRIS K (10 min; photon limited)",color="#ff9900")
        plt.plot(binnedpsf_sep,OSIRIS_photnoise60_mass,
                 linestyle="-",linewidth=3,label="OSIRIS K (60 min; photon limited)",color="#ff9900")

        np.savetxt("/home/sda/jruffio/pyOSIRIS/figures/contrast4Eric.txt",
            np.concatenate([binnedpsf_sep[:,None],OSIRIS_photnoise_cont[:,None]/np.sqrt(6)],axis=1))
        plt.xscale("log")
        plt.xlim([0.05,2])
        plt.ylim([0,12])
        plt.xlabel("Separation (arcsec)",fontsize=fontsize)
        plt.ylabel("BTSettl - 2 Myr - Mass ($M_{Jup}$)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize*0.75)
        plt.tight_layout()

        plt.show()

        exit()

    if 1:
        sky_nodarksub_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_nosub","s"+date+"*{0}*".format(9)+IFSfilter+"_"+scale+".fits"))
        sky2_nodarksub_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_nosub","s"+date+"*{0}*".format(22)+IFSfilter+"_"+scale+".fits"))
        sky_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_simpledarksub","s"+date+"*{0}*".format(9)+IFSfilter+"_"+scale+".fits"))
        sky2_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_simpledarksub","s"+date+"*{0}*".format(22)+IFSfilter+"_"+scale+".fits"))
        sc_noskysub_filelist = glob.glob(os.path.join(inputDir,"..","noise_test_nosub","s"+date+"*{0}*".format(imnum)+IFSfilter+"_"+scale+".fits"))
        sc_filelist = glob.glob(os.path.join(inputDir,"..","reduced_jb","s"+date+"*{0}*".format(imnum)+IFSfilter+"_"+scale+".fits"))
        filelist = [sky_nodarksub_filelist[0],
                    sky2_nodarksub_filelist[0],
                    sky_filelist[0],
                    sky2_filelist[0],
                    sc_noskysub_filelist[0],sc_filelist[0]]
        # label_list = ["sky w/o dark sub","sky w/ dark sub","science w/o sky sub","science w/ sky sub"]
        myvec_list = []
        for filename in filelist:
            print(filename)
            with pyfits.open(filename) as hdulist:
                prihdr = hdulist[0].header
                curr_mjdobs = prihdr["MJD-OBS"]
                imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
                imgs = return_64x19(imgs)
                imgs = np.moveaxis(imgs,0,2)
                imgs_hdrbadpix = np.rollaxis(np.rollaxis(hdulist[2].data,2),2,1)
                imgs_hdrbadpix = return_64x19(imgs_hdrbadpix)
                imgs_hdrbadpix = np.moveaxis(imgs_hdrbadpix,0,2)
                imgs_hdrbadpix = imgs_hdrbadpix.astype(dtype=ctypes.c_double)
                imgs_hdrbadpix[np.where(imgs_hdrbadpix==0)] = np.nan
                imgs[np.where(imgs_hdrbadpix==0)] = 0
            ny,nx,nz = imgs.shape
            init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
            dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
            wvs=np.linspace(init_wv,init_wv+dwv*nz,nz,endpoint=False)

            myvec_list.append(imgs[ply,plx,:])
            myvec = copy(imgs[ply,plx,:])
            window_size=100
            threshold=7
            smooth_vec = median_filter(myvec,footprint=np.ones(window_size),mode="reflect")
            _myvec = myvec - smooth_vec
            wherefinite = np.where(np.isfinite(_myvec))
            mad = mad_std(_myvec[wherefinite])
            whereoutliers = np.where(np.abs(_myvec)>threshold*mad)[0]
            myvec[whereoutliers] = np.nan

            # plt.plot(wvs,myvec,label=label)


        dark = myvec_list[0]-myvec_list[2]
        dark2 = myvec_list[1]-myvec_list[3]
        sky = myvec_list[0]
        delta_sky_nodarksub = myvec_list[0]-myvec_list[1]
        dark_pix2pix = dark-dark2
        window_size=100
        threshold=5
        smooth_vec = median_filter(dark,footprint=np.ones(window_size),mode="reflect")
        _myvec = dark - smooth_vec
        wherefinite = np.where(np.isfinite(_myvec))
        mad = mad_std(_myvec[wherefinite])
        whereoutliers = np.where(np.abs(_myvec)>threshold*mad)[0]
        dark[whereoutliers] = np.nan
        dark_HPF = LPFvsHPF(dark,cutoff=cutoff)[1]
        smooth_vec = median_filter(dark_pix2pix,footprint=np.ones(window_size),mode="reflect")
        _myvec = dark_pix2pix - smooth_vec
        wherefinite = np.where(np.isfinite(_myvec))
        mad = mad_std(_myvec[wherefinite])
        whereoutliers = np.where(np.abs(_myvec)>threshold*mad)[0]
        dark_pix2pix[whereoutliers] = np.nan
        dark_HPF = LPFvsHPF(dark,cutoff=cutoff)[1]
        smooth_vec = median_filter(delta_sky_nodarksub,footprint=np.ones(window_size),mode="reflect")
        _myvec = delta_sky_nodarksub - smooth_vec
        wherefinite = np.where(np.isfinite(_myvec))
        mad = mad_std(_myvec[wherefinite])
        whereoutliers = np.where(np.abs(_myvec)>threshold*mad)[0]
        delta_sky_nodarksub[whereoutliers] = np.nan
        delta_sky_nodarksub = LPFvsHPF(delta_sky_nodarksub,cutoff=cutoff)[1]
        sky[whereoutliers] = np.nan

        # print(np.nanmean(dark),np.nanstd(dark))
        # plt.plot(wvs,dark,label='"dark"')
        # # plt.plot(dark_pix2pix)
        # plt.plot(wvs,sky,label="sky")
        # plt.ylabel("Data number")
        # plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        # plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)
        # plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        # plt.tight_layout()
        # # fig.savefig(os.path.join(out_pngs,"noise_analysis_4.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        # # fig.savefig(os.path.join(out_pngs,"noise_analysis_4.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        # plt.show()
        # exit()

        # res_filelist = glob.glob(os.path.join(inputDir,"sherlock","20191205_RV",os.path.basename(sc_filelist[0]).replace(".fits","")+"*_outputHPF_cutoff40_sherlock_v1_search_rescalc_res.fits"))
        # print(res_filelist[0])
        # with pyfits.open(res_filelist[0]) as hdulist:
        #     hpf = hdulist[0].data[0,0,0,:,:,:]
        #     lpf = hdulist[0].data[0,0,5,:,:,:]
        #     hpfres = hdulist[0].data[0,0,6,:,:,:]
        #     print(lpf.shape)
        #     # exit()
        #     # myvec = (hpf+lpf)[:,ply+5,plx+5]()
        #     myvec = hpfres[:,ply+5,plx+5]
        #     plt.plot(wvs,myvec,linestyle="--",label="Mean 5x5 Residuals H0")
        #



        # plt.legend()
        # plt.show()

    if 1:
        modelfolder = "20200309_model"
        gridname = os.path.join("/data/osiris_data/","hr8799b_modelgrid")
        N_kl = 0#10
        numthreads = 32#16
        small = True
        inj_fake_str=""

        Tfk,loggfk,ctoOfk = 1200,3.7,0.55

        c_kms = 299792.458
        cutoff = 40
        R= 4000

        tmpfilename = os.path.join(gridname,"hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
        hdulist = pyfits.open(tmpfilename)
        planet_model_grid =  hdulist[0].data
        oriplanet_spec_wvs =  hdulist[1].data
        Tlistunique =  hdulist[2].data
        logglistunique =  hdulist[3].data
        CtoOlistunique =  hdulist[4].data
        hdulist.close()

        from scipy.interpolate import RegularGridInterpolator
        myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

        filename = glob.glob(os.path.join(inputDir,"..","reduced_jb","s"+date+"*{0}*".format(imnum)+IFSfilter+"_"+scale+".fits"))[0]
        print(filename)

        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_corrwvs"+inj_fake_str+".fits"))
        if len(glob.glob(tmpfilename))!=1:
            print("No data on "+filename)
            exit()
        hdulist = pyfits.open(tmpfilename)
        # wvs =  hdulist[0].data
        if small:
            wvs =  hdulist[0].data[2:7,2:7,:]
        else:
            wvs =  hdulist[0].data
        hdulist.close()

        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_LPFdata"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            LPFdata =  hdulist[0].data[2:7,2:7,:]
        else:
            LPFdata =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_HPFdata"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        print(hdulist[0].data.shape)
        if small:
            HPFdata =  hdulist[0].data[2:7,2:7,:]
        else:
            HPFdata =  hdulist[0].data
        hdulist.close()

        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_badpix"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            data_badpix =  hdulist[0].data[2:7,2:7,:]
        else:
            data_badpix =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_sigmas"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            data_sigmas =  hdulist[0].data[2:7,2:7,:]
        else:
            data_sigmas =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_trans"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        transmission_vec =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_starspec"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if small:
            star_obsspec =  hdulist[0].data[2:7,2:7,:]
        else:
            star_obsspec =  hdulist[0].data
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_reskl"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        if N_kl == 0:
            res4model_kl = None
        else:
            res4model_kl =  hdulist[0].data[:,0:N_kl]
        hdulist.close()
        tmpfilename = os.path.join(os.path.dirname(filename),modelfolder,os.path.basename(filename).replace(".fits","_plrv0"+inj_fake_str+".fits"))
        hdulist = pyfits.open(tmpfilename)
        plrv0 =  hdulist[0].data
        data_ny,data_nx,data_nz = HPFdata.shape
        w = int((data_nx-1)//2)
        star_flux = np.nansum(star_obsspec[w,w,:])
        hdulist.close()

        ##############################
        ## Create PSF model
        ##############################
        ref_star_folder = os.path.join(os.path.dirname(filename),"..","reduced_telluric_jb")
        with pyfits.open(glob.glob(os.path.join(ref_star_folder,"*"+IFSfilter+"_hdpsfs_v2.fits"))[0]) as hdulist:
            psfs_refstar_arr = hdulist[0].data[None,:,:,:]
        with pyfits.open(glob.glob(os.path.join(ref_star_folder,"*"+IFSfilter+"_hdpsfs_xy_v2.fits"))[0]) as hdulist:
            hdpsfs_xy = hdulist[0].data
            hdpsfs_x,hdpsfs_y = hdpsfs_xy

        nx_psf,ny_psf = 15,15
        nz_psf = psfs_refstar_arr.shape[1]
        x_psf_vec, y_psf_vec = np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2
        x_psf_grid, y_psf_grid = np.meshgrid(x_psf_vec, y_psf_vec)

        print("starting psf")
        specpool = mp.Pool(processes=numthreads)
        chunk_size=20
        N_chunks = nz_psf//chunk_size
        psfs_chunks = []
        for k in range(N_chunks-1):
            psfs_chunks.append(psfs_refstar_arr[:,k*chunk_size:(k+1)*chunk_size,:,:])
        psfs_chunks.append(psfs_refstar_arr[:,(N_chunks-1)*chunk_size:nz_psf,:,:])
        outputs_list = specpool.map(_spline_psf_model, zip(psfs_chunks,
                                                           itertools.repeat(hdpsfs_x[None,:,:]),
                                                           itertools.repeat(hdpsfs_y[None,:,:]),
                                                           itertools.repeat(x_psf_grid[0,0:nx_psf-1]+0.5),
                                                           itertools.repeat(y_psf_grid[0:ny_psf-1,0]+0.5),
                                                           np.arange(len(psfs_chunks))))

        normalized_psfs_func_list = []
        chunks_ids = []
        for out in outputs_list:
            normalized_psfs_func_list.extend(out[1])
            chunks_ids.append(out[0])
        print("finish psf")
        specpool.close()
        specpool.join()
        print("closed psf")

        dx,dy = 0,0
        nospec_planet_model = np.zeros(HPFdata.shape)
        pl_x_vec = np.arange(-w,w+1) + dx
        pl_y_vec = np.arange(-w,w+1) + dy
        for z in range(data_nz):
            nospec_planet_model[:,:,z] = normalized_psfs_func_list[z](pl_x_vec,pl_y_vec).transpose()


        ravelHPFdata = np.ravel(HPFdata)
        ravelLPFdata = np.ravel(LPFdata)
        ravelsigmas = np.ravel(data_sigmas)
        where_finite_data = np.where(np.isfinite(np.ravel(data_badpix)))
        where_bad_data = np.where(~(np.isfinite(np.ravel(data_badpix))))
        ravelLPFdata = ravelLPFdata[where_finite_data]
        sigmas_vec = ravelsigmas[where_finite_data]#np.ones(ravelLPFdata.shape)#np.sqrt(np.abs(ravelLPFdata))
        ravelHPFdata = ravelHPFdata[where_finite_data]
        ravelHPFdata = ravelHPFdata/sigmas_vec
        logdet_Sigma = np.sum(2*np.log(sigmas_vec))

        planetRV_array = np.array([plrv0])

        HPFmodelH0_list = []
        if 1:
            bkg_model = np.zeros((2*w+1,2*w+1,2*w+1,2*w+1,data_nz))
            for bkg_k in range(2*w+1):
                for bkg_l in range(2*w+1):
                    if 1:
                        star_obsspec_tmp = star_obsspec[bkg_k,bkg_l]
                        smooth_model = median_filter(star_obsspec_tmp,footprint=np.ones(50),mode="reflect")
                        where_bad_pix_4model = np.where(np.isnan(data_badpix[bkg_k,bkg_l,:]))
                        star_obsspec_tmp[where_bad_pix_4model] = smooth_model[where_bad_pix_4model]
                        star_obsspec_tmp[np.where(np.isnan(HPFdata[bkg_k,bkg_l,:]))] = np.nan
                    LPF_star_obsspec_tmp,_ = LPFvsHPF(star_obsspec_tmp,cutoff)

                    myspec = LPFdata[bkg_k,bkg_l,:]*star_obsspec_tmp/LPF_star_obsspec_tmp
                    if 1:
                        smooth_model = median_filter(myspec,footprint=np.ones(50),mode="reflect")
                        where_bad_pix_4model = np.where(np.isnan(data_badpix[bkg_k,bkg_l,:]))
                        myspec[where_bad_pix_4model] = smooth_model[where_bad_pix_4model]
                        myspec[np.where(np.isnan(HPFdata[bkg_k,bkg_l,:]))] = np.nan
                    _,myspec = LPFvsHPF(myspec,cutoff)

                    bkg_model[bkg_k,bkg_l,bkg_k,bkg_l,:] = myspec
            HPFmodelH0_list.append(np.reshape(bkg_model,((2*w+1)**2,(2*w+1)**2*data_nz)).transpose())
        if res4model_kl is not None:
            for kid in range(res4model_kl.shape[1]):
                res4model = res4model_kl[:,kid]
                LPF4resmodel = np.nansum(LPFdata*nospec_planet_model,axis=(0,1))/np.nansum(nospec_planet_model**2,axis=(0,1))
                resmodel = nospec_planet_model*LPF4resmodel[None,None,:]*res4model[None,None,:]
                HPFmodelH0_list.append(np.ravel(resmodel)[:,None])


        HPFmodel_H0 = np.concatenate(HPFmodelH0_list,axis=1)

        HPFmodel_H0 = HPFmodel_H0[where_finite_data[0],:]/sigmas_vec[:,None]


        cp_HPFmodel_H0 = copy(HPFmodel_H0)
        cp_HPFmodel_H0[np.where(np.isnan(HPFmodel_H0))] = 0

        w = int((nospec_planet_model.shape[0]-1)/2)
        c_kms = 299792.458
        # print(temp,fitlogg,CtoO)
        planet_template_func = interp1d(oriplanet_spec_wvs,myinterpgrid([Tfk,loggfk,ctoOfk])[0],bounds_error=False,fill_value=np.nan)

        planet_model = copy(nospec_planet_model)
        for bkg_k in range(2*w+1):
            for bkg_l in range(2*w+1):
                # print(wvs.shape,plrv,c_kms)
                wvs4planet_model = wvs[bkg_k,bkg_l,:]*(1-(plrv0)/c_kms)
                planet_model[bkg_k,bkg_l,:] *= planet_template_func(wvs4planet_model) * transmission_vec

        star_model = copy(nospec_planet_model)
        for bkg_k in range(2*w+1):
            for bkg_l in range(2*w+1):
                # print(wvs.shape,plrv,c_kms)
                star_model[bkg_k,bkg_l,:] *= star_obsspec[3,3,:]
        star_model = star_model/np.nansum(star_model)*star_flux


        planet_model = planet_model/np.nansum(planet_model)*star_flux*1e-5
        HPF_planet_model = np.zeros(planet_model.shape)
        for bkg_k in range(2*w+1):
            for bkg_l in range(2*w+1):
                HPF_planet_model[bkg_k,bkg_l,:]  = LPFvsHPF(planet_model[bkg_k,bkg_l,:] ,cutoff)[1]


        HPFmodel_H1only = (HPF_planet_model.ravel())[:,None]
        HPFmodel_H1only = HPFmodel_H1only[where_finite_data[0],:]/sigmas_vec[:,None] # where_finite_data[0]
        HPFmodel_H1only[np.where(np.isnan(HPFmodel_H1only))] = 0

        HPFmodel = np.concatenate([HPFmodel_H1only,cp_HPFmodel_H0],axis=1)
        HPFmodel_H1_cp_4res = copy(HPFmodel)

        where_valid_parameters = np.where(np.nansum(np.abs(HPFmodel)>0,axis=0)>=50)
        HPFmodel = HPFmodel[:,where_valid_parameters[0]]

        HPFparas,HPFchi2,rank,s = np.linalg.lstsq(HPFmodel,ravelHPFdata,rcond=None)
        data_model = np.dot(HPFmodel,HPFparas)
        ravelresiduals = ravelHPFdata-data_model
        # HPFchi2 = np.nansum((ravelresiduals)**2)
        # Npixs_HPFdata = HPFmodel.shape[0]
        # covphi =  HPFchi2/Npixs_HPFdata*np.linalg.inv(np.dot(HPFmodel.T,HPFmodel))
        # slogdet_icovphi0 = np.linalg.slogdet(np.dot(HPFmodel.T,HPFmodel))
        # logpost_rv[temp_id,plrv_id] = -0.5*logdet_Sigma-0.5*slogdet_icovphi0[1]- (Npixs_HPFdata-HPFmodel.shape[-1]+2-1)/(2)*np.log(HPFchi2)


        canvas_res= np.zeros(HPFdata.shape) + np.nan
        canvas_res = np.reshape(canvas_res,((2*w+1)**2*data_nz))
        canvas_res[where_finite_data[0]] = ravelresiduals
        canvas_res = np.reshape(canvas_res,((2*w+1),(2*w+1),data_nz))
        canvas_res *= data_sigmas
        canvas_model= np.zeros(HPFdata.shape) + np.nan
        canvas_model = np.reshape(canvas_model,((2*w+1)**2*data_nz))
        canvas_model[where_finite_data[0]] = data_model
        canvas_model = np.reshape(canvas_model,((2*w+1),(2*w+1),data_nz))
        canvas_model *= data_sigmas

        _res = canvas_res[2,2,:]
        _res[np.where(np.isnan(_res))] = 0
        # _res = np.ones(_res.shape)
        res_ccf = np.correlate(_res,_res,mode="same")/np.size(_res)
        res_ccf_argmax = np.argmax(res_ccf)


        fig = plt.figure(1,figsize=(12,4))
        # colors=["#006699","#ff9900","#6600ff","grey"]
        plt.plot(wvs[2,2,:],star_model[2,2,:],linestyle="--",linewidth=3,color="#006699",label="On-axis star")
        plt.plot(wvs[2,2,:],LPFdata[2,2,:]+HPFdata[2,2,:],linestyle="-",linewidth=2,color="#ff9900",label="Starlight + planet")
        plt.plot(wvs[2,2,:],planet_model[2,2,:]*HPFparas[0],linestyle="-",linewidth=1,color="#6600ff",label="Scaled planet (fit)")
        plt.plot(wvs[2,2,:],sky,linestyle="-.",linewidth=0.5,color="blue",label="Sky")
        # plt.plot(wvs[2,2,:],dark,linestyle="--",linewidth=0.5,color="grey",label="Irreducible noise")
        plt.yscale("log")
        plt.ylim([1e-2,1e5])
        plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        plt.ylabel(r"Data Number",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_1.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_1.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

        fig = plt.figure(2,figsize=(12,3))
        # colors=["#006699","#ff9900","#6600ff","grey"]
        print(np.nansum(planet_model[2,2,:]*HPFparas[0])/np.nansum((LPFdata[2,2,:]+HPFdata[2,2,:])))
        # exit()
        plt.plot(wvs[2,2,:],planet_model[2,2,:]*HPFparas[0]/(LPFdata[2,2,:]+HPFdata[2,2,:]),linestyle="--",linewidth=1,color="#6600ff",label="Scaled planet / Data")
        plt.plot(wvs[2,2,:],(LPFdata[2,2,:]+HPFdata[2,2,:])/star_model[2,2,:],linestyle="-",linewidth=2,color="#ff9900",label="(Starlight + planet) / On-axis star")
        plt.plot(wvs[2,2,:],planet_model[2,2,:]*HPFparas[0]/star_model[2,2,:],linestyle="-",linewidth=1,color="#6600ff",label="Scaled planet / On-axis star")
        plt.yscale("log")
        plt.ylim([1e-6,1e-1])
        plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        plt.ylabel(r"Flux ratio",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_2.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_2.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

        fig = plt.figure(3,figsize=(12,3))
        plt.plot(wvs[2,2,:],HPFdata[2,2,:],linestyle="-",linewidth=2,color="#ff9900",label="Data (HPF; Starlight + planet)")
        plt.plot(wvs[2,2,:],canvas_model[2,2,:],linestyle="--",linewidth=0.5,color="black",label="Forward Model")
        plt.plot(wvs[2,2,:],HPF_planet_model[2,2,:]*HPFparas[0],linestyle="-",linewidth=1,color="#6600ff",label="Scaled planet (HPF)")
        plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        plt.ylabel(r"Data Number",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_3.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_3.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

        fig = plt.figure(4,figsize=(12,2))
        # plt.plot(wvs,dark_pix2pix/np.sqrt(2),linestyle="--",label="dark_pix2pix: std={0}".format()
        # plt.plot(wvs,delta_sky_nodarksub/np.sqrt(2),linestyle="-.",label="sky + dark pix2pix: std={0}".format(np.sqrt((np.nanvar(delta_sky_nodarksub)-np.nanvar(dark_pix2pix))/2)))
        # plt.plot(wvs[2,2,:],canvas_res[2,2,:],label="res: std_pix2pix={0} ; std_corr={1}".format(np.sqrt(res_ccf[res_ccf_argmax]-res_ccf[res_ccf_argmax-1]),
        #                                                                                          np.sqrt(res_ccf[res_ccf_argmax-1])))
        # plt.subplot(1,2,1)
        # plt.plot(wvs[2,2,:],dark_pix2pix/np.sqrt(2),linestyle="--",linewidth=0.5,color="grey",label="$\Delta$(Irr. noise)/$\sqrt{2}$ (HPF)")
        plt.plot(wvs[2,2,:],delta_sky_nodarksub/np.sqrt(2),linestyle="-.",linewidth=0.5,color="blue",label="$\Delta$Sky (HPF)")
        plt.plot(wvs[2,2,:],canvas_res[2,2,:],linestyle="-",linewidth=1,color="#6600ff",label="Residuals",alpha=0.5)
        plt.ylim([-0.2,0.2])
        plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        plt.ylabel(r"Data Number",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        # plt.subplot(1,2,2)
        # plt.plot(wvs[2,2,:],dark_pix2pix/np.sqrt(2),linestyle="--",linewidth=0.5,color="grey",label="$\Delta$(Irr. noise)/$\sqrt{2}$ (HPF)")
        # plt.plot(wvs[2,2,:],delta_sky_nodarksub/np.sqrt(2),linestyle="-.",linewidth=0.5,color="blue",label="$\Delta$Sky/$\sqrt{2}$ (HPF)")
        # plt.plot(wvs[2,2,:],canvas_res[2,2,:],linestyle="-",linewidth=1,color="#6600ff",label="Residuals",alpha=0.5)
        # plt.ylim([-0.2,0.2])
        # plt.xlim([2.28,2.38])
        # plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        # plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_4.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_4.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

        fig = plt.figure(5,figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(np.arange(-100,101),res_ccf[(res_ccf_argmax-100):(res_ccf_argmax+101)])
        plt.xlabel(r"Spectral pixel",fontsize=fontsize)
        plt.ylabel(r"Auto-correlation (Variance)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.subplot(1,2,2)
        plt.plot(np.arange(-5,6),res_ccf[(res_ccf_argmax-5):(res_ccf_argmax+6)])
        plt.xlabel(r"Spectral pixel",fontsize=fontsize)
        plt.ylabel(r"Auto-correlation (Variance)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.tight_layout()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_5.pdf"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()
        fig.savefig(os.path.join(out_pngs,"noise_analysis_5.png"),bbox_inches='tight') # This is matplotlib.figure.Figure.savefig()

        print(res_ccf[res_ccf_argmax-1],res_ccf[res_ccf_argmax],res_ccf[res_ccf_argmax+1])
        # print("var res pix2pix - var res corr",res_ccf[res_ccf_argmax]-res_ccf[res_ccf_argmax-1])
        # print("var res corr",res_ccf[res_ccf_argmax-1])
        # print("var dark pix2pix",np.nanvar(dark_pix2pix)/2)
        # # print("var dark HPF",np.nanvar(dark_HPF))
        # print("var sky pix2pix",(np.nanvar(delta_sky_nodarksub)-np.nanvar(dark_pix2pix))/2)
        #
        # print("dark_pix2pix: var={0}".format((np.nanvar(dark_pix2pix)/2)))
        # print("sky - dark pix2pix: var={0}".format(((np.nanvar(delta_sky_nodarksub)-np.nanvar(dark_pix2pix))/2)))
        # print("res: var_pix2pix={0} ; var_corr={1} ; var_tot={2}".format((res_ccf[res_ccf_argmax]-res_ccf[res_ccf_argmax-1]),
        #                                                                                          (res_ccf[res_ccf_argmax-1]),
        #                                                                                          (res_ccf[res_ccf_argmax])))
        # print("dark_pix2pix: std={0}".format(np.sqrt(np.nanvar(dark_pix2pix)/2)))
        # print("sky - dark pix2pix: std={0}".format(np.sqrt((np.nanvar(delta_sky_nodarksub)-np.nanvar(dark_pix2pix))/2)))
        # print("res: std_pix2pix={0} ; std_corr={1} ; var_tot={2}".format(np.sqrt(res_ccf[res_ccf_argmax]-res_ccf[res_ccf_argmax-1]),
        #                                                                                          np.sqrt(res_ccf[res_ccf_argmax-1]),
        #                                                                                          np.sqrt(res_ccf[res_ccf_argmax])))


        print("sky std", np.nanstd(delta_sky_nodarksub))
        print("res std", np.nanstd(canvas_res[2,2,:]))
        print("photon std", np.sqrt(np.nanmedian((LPFdata[2,2,:]+HPFdata[2,2,:]))*600*2.15)/600)
        print(np.nanmedian(LPFdata[2,2,:]+HPFdata[2,2,:]))

        print("bin planet SNR",np.median(planet_model[2,2,:]*HPFparas[0])/np.nanstd(canvas_res[2,2,:]))
        print("bin star SNR",np.nanmedian(HPFdata[2,2,:]+LPFdata[2,2,:])/np.nanstd(canvas_res[2,2,:]))
        print("starlight",np.nanmedian((LPFdata[2,2,:]+HPFdata[2,2,:])/star_model[2,2,:]))
        print(np.median(np.nanmax(nospec_planet_model,axis=(0,1))/np.nansum(nospec_planet_model,axis=(0,1))))


    plt.show()
    exit()

    exit()


    # inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb_pairsub/"
    # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb_pairsub/20190228_HPF_only/"

    print(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
    filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
    filelist.sort()
    filelist = [filelist[7],]
    print(filelist)