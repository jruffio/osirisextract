__author__ = 'jruffio'



import os
import sys
import glob
import time
import astropy.io.fits as pyfits
import numpy as np

print("coucou")

# OSIRISDATA = "/scratch/groups/bmacint/osiris_data/"
OSIRISDATA = "/data/osiris_data/"
if 1:
    # foldername = "HR_8799_b"
    foldername = "HR_8799_c"
    # foldername = "HR_8799_d"

if 1:
    year = "*"
    # reductionname = "reduced_quinn"
    # filenamefilter = "s*_a*001_tlc_Kbb_020.fits"
    # filenamefilter = "s*_a*001_tlc_Hbb_020.fits"
    reductionname = "reduced_jb"
    filenamefilter = "s*_a*_Hbb_020.fits"

    filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
    filelist.sort()
    for filename in filelist:
        print(filename)
        #continue

        inputdir = os.path.dirname(filename)
        with pyfits.open(filename) as hdulist:
            # imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
            prihdr = hdulist[0].header

            for k,line in enumerate(prihdr["COMMENT"]):
                #
                if line.startswith('DRFC  <module Name="Subtract Frame"'):
                    print(line)
                    print(prihdr["COMMENT"][k+1])
                    print(prihdr["COMMENT"][k+2])
                    break
            # exit()
    exit()

if 0:
    year = "*"
    reductionname = "reduced_jb"
    filenamefilter = "s*_a*001_Kbb_020.fits"

    filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
    filelist.sort()
    for filename in filelist:
        print(filename)
        #continue

        inputdir = os.path.dirname(filename)
        with pyfits.open(filename) as hdulist:
            # imgs = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
            prihdr = hdulist[0].header

            for k,line in enumerate(prihdr["COMMENT"]):
                #
                if line.startswith('DRFC  <module Name="Subtract Frame"'):
                    print(line)
                    print(prihdr["COMMENT"][k+1])
                    print(prihdr["COMMENT"][k+2])
                    break
            # exit()