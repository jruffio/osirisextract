__author__ = 'jruffio'



import os
import sys
import glob
import time
import astropy.io.fits as pyfits
import numpy as np

print("coucou")

# OSIRISDATA = "/scratch/groups/bmacint/osiris_data/"
OSIRISDATA = "/home/sda/jruffio/osiris_data/"
if 1:
    foldername = "HR_8799_c"
    sep = 0.950
    telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
    template_spec = os.path.join(OSIRISDATA,"hr8799c_osiris_template.save")

if 0:
    year = "*"
    reductionname = "reduced_quinn"
    filenamefilter = "s*_a*001_tlc_Kbb_020.fits"

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

if 1:
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