__author__ = 'jruffio'

import glob
import os
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np

fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos.xml"
out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
tree = ET.parse(fileinfos_filename)
root = tree.getroot()


OSIRISDATA = "/home/sda/jruffio/osiris_data/"
if 1:
    foldername = "HR_8799_c"
    sep = 0.950
    telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
    template_spec = os.path.join(OSIRISDATA,"hr8799c_osiris_template.save")
year = "*"#"20101104"
reductionname = "reduced_quinn"
filenamefilter = "s*_a*001_tlc_Kbb_020.fits"

from astropy.stats import mad_std

# HPF only
if 1:
    suffix = "_outputHPF_cutoff80_new_sig_phoenix_wvshift_centroid"
    planet = "c"
    # planet = "d"
    IFSfilter = "Kbb"
    # IFSfilter = "Hbb"
    fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos_jb.xml"
    out_pngs = os.path.join("/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/")#"/home/sda/jruffio/pyOSIRIS/figures/"
    tree = ET.parse(fileinfos_filename)
    root = tree.getroot()
    reductionname = "reduced_jb"
    filenamefilter = "s*_a*001_"+IFSfilter+"_020.fits"
    filelist = glob.glob(os.path.join(OSIRISDATA,"HR_8799_"+planet,year,reductionname,filenamefilter))
    filelist.sort()
    planet_c = root.find("c")
    print(len(filelist))

    if IFSfilter=="Kbb": #Kbb 1965.0 0.25
        CRVAL1 = 1965.
        CDELT1 = 0.25
        nl=1665
        R=4000
    elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1473.
        CDELT1 = 0.2
        nl=1651
        R=5000
    dwv = CDELT1/1000.

    # f,ax1_list = plt.subplots(4,len(filelist)//4+1,sharey="row",sharex="col",figsize=(18,0.59*18))
    # ax1_list = [myax for rowax in ax1_list for myax in rowax ]
    # f,ax2_list = plt.subplots(4,len(filelist)//4+1,sharey="row",sharex="col",figsize=(18,0.59*18))
    # ax2_list = [myax for rowax in ax2_list for myax in rowax ]
    ax1_list = [plt.subplots(1,1,sharey="row",sharex="col",figsize=(8,8))[1]]
    ax2_list = [plt.subplots(1,1,sharey="row",sharex="col",figsize=(8,8))[1]]
    for ax1,ax2,filename in zip(ax1_list,ax2_list,filelist):
        print(filename)
        # filebasename = os.path.basename(filename)
        # fileelement = planet_c.find(filebasename)
        # print(fileelement.attrib["xADIcen"],fileelement.attrib["yADIcen"])

        # try:
        if 1:
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"20181205_HPF_only",
                                               os.path.basename(filename).replace(".fits",suffix+".fits")))
            logposterior = hdulist[0].data[11,:,:,:]
            prihdr = hdulist[0].header
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"20181205_HPF_only",
                                               os.path.basename(filename).replace(".fits",suffix+"_wvshifts.fits")))
            wvshifts = hdulist[0].data
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"20181205_HPF_only",
                                               os.path.basename(filename).replace(".fits",suffix+"_klgrids.fits")))
            klgrids = hdulist[0].data
            real_k_grid = klgrids[0,:,:]
            real_l_grid = klgrids[1,:,:]

            logposterior = np.exp(logposterior-np.nanmax(logposterior))
            logpost_pos = np.nansum(logposterior,axis=0)
            logpost_wvs = np.nansum(logposterior,axis=(1,2))


            # plt.figure(1)
            plt.sca(ax1)
            plt.plot(wvshifts/dwv,logpost_wvs)
            # plt.imshow(logpost_pos,interpolation="nearest")
            # plt.colorbar()

            # plt.figure(2)
            plt.sca(ax2)
            ny,nx = logpost_pos.shape
            plt.imshow(logpost_pos,interpolation="nearest")
            plt.colorbar()
            plt.show()

            logpost_pos[np.where(~np.isfinite(logpost_pos))] = 0
            maxind = np.unravel_index(np.argmax(logpost_pos),logpost_pos.shape)
            circle = plt.Circle(maxind[::-1],5,color="red", fill=False)
            ax2.add_artist(circle)
            plt.show()
            # exit()
        # except:
        #     pass

    # f.subplots_adjust(wspace=0,hspace=0)
    # print("Saving "+os.path.join(out_pngs,"HR8799"+planet+suffix+"_"+IFSfilter+".pdf"))
    # plt.savefig(os.path.join(out_pngs,"HR8799"+planet+suffix+"_"+IFSfilter+".pdf"),bbox_inches='tight')
    # plt.savefig(os.path.join(out_pngs,"HR8799"+planet+suffix+"_"+IFSfilter+".png"),bbox_inches='tight')
