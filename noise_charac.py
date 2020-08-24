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
try:
    import mkl
    mkl_exists = True
except ImportError:
    mkl_exists = False

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
    date = "200731"
    # date = "200803"
    # IFSfilter = "Kbb"
    IFSfilter = "Kbb"
    # IFSfilter = "Jbb" # "Kbb" or "Hbb"
    scale = "020"
    # scale = "035"

    inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb/"

    # inputDir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb_pairsub/"
    # outputdir = "/data/osiris_data/"+planet+"/20"+date+"/reduced_jb_pairsub/20190228_HPF_only/"

    print(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
    filelist = glob.glob(os.path.join(inputDir,"s"+date+"*"+IFSfilter+"_"+scale+".fits"))
    filelist.sort()
    filelist = [filelist[7],]
    print(filelist)