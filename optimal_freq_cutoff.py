__author__ = 'jruffio'

import numpy as np
import os
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import glob

#------------------------------------------------
if __name__ == "__main__":

    inputDir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/"
    outputdir = "/home/sda/jruffio/osiris_data/HR_8799_c/20100715/reduced_jb/20181120_out/"
    filelist = glob.glob(os.path.join(inputDir,"s100715*20.fits"))
    filelist.sort()
    filename = filelist[0]

    out_list = []
    # for cutoff in np.arange(10,200,10):
    for cutoff in np.arange(10,200,20):

        # suffix = "polyfit_"+centermode+"cen"+"_testmaskbadpix"
        # suffix = "polyfit_"+centermode+"cen"+"_resmask_maskbadpix"
        # suffix = "polyfit_"+centermode+"cen"+"_resmask_norma"
        # suffix = "polyfit_"+centermode+"cen"+"_resmask_norma_bkg"
        # suffix = "polyfit_"+centermode+"cen"+"_cov_all"
        suffix = "test_splitLPFHPF_all_LPFbinned_cutoff{0}".format(cutoff)


        out_file = os.path.join(outputdir,os.path.basename(filename).replace(".fits","_output"+suffix+".fits"))
        with pyfits.open(out_file) as hdulist:
            cube = hdulist[0].data
            out_list.append(cube)


        plt.subplot(1,2,1)
        plt.title("LPF")
        plt.plot((cube[2,:,16]-cube[1,:,16])/((cube[2,37,16]-cube[1,37,16])),label="{0}".format(cutoff))
        # plt.subplot(1,2,2)
        # plt.plot(cube[3,:,16],label="{0}".format(cutoff))
        plt.subplot(1,2,2)
        plt.title("HPF")
        plt.plot(cube[7,:,16]/cube[7,37,16],label="{0}".format(cutoff))


    plt.legend()
    plt.show()

