__author__ = 'jruffio'

import glob
import os
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
import csv


out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
OSIRISDATA = "/data/osiris_data/"
# planet = "b"
planet = "c"
date = "100715"
# date = "101104"
# date = "110723"
# planet = "d"
# date = "150720"
# date = "150722"
# date = "150723"
# date = "150828"
IFSfilter = "Kbb"
# IFSfilter = "Hbb" # "Kbb" or "Hbb"

from astropy.stats import mad_std

# HPF only
if 1:
    suffix = "_outputHPF_cutoff40_sherlock_v1_search"
    fileinfos_filename = "/data/osiris_data/HR_8799_"+planet+"/fileinfos_"+IFSfilter+"_jb.csv"
    out_pngs = os.path.join("/data/osiris_data/HR_8799_"+planet+"/")#"/home/sda/jruffio/pyOSIRIS/figures/"
    reductionname = "reduced_jb"
    foldername = "20190307_HPF_only_full3"
    filenamefilter = "s*_a*001_"+IFSfilter+"_020.fits"
    filelist = glob.glob(os.path.join(OSIRISDATA,"HR_8799_"+planet,"20"+date,reductionname,filenamefilter))
    filelist.sort()

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

    ## file specific info
    with open(fileinfos_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        list_table = list(csv_reader)
        colnames = list_table[0]
        N_col = len(colnames)
        list_data = list_table[1::]
        N_lines =  len(list_data)

    filename_id = colnames.index("filename")
    filelist_infos = [os.path.basename(item[filename_id]) for item in list_data]

    # f,ax1_list = plt.subplots(4,len(filelist)//4+1,sharey="row",sharex="col",figsize=(18,0.59*18))
    # ax1_list = [myax for rowax in ax1_list for myax in rowax ]
    # f,ax2_list = plt.subplots(4,len(filelist)//4+1,sharey="row",sharex="col",figsize=(18,0.59*18))
    # ax2_list = [myax for rowax in ax2_list for myax in rowax ]
    ax1_list = [plt.subplots(1,1,sharey="row",sharex="col",figsize=(8,8))[1]]
    ax2_list = [plt.subplots(1,1,sharey="row",sharex="col",figsize=(8,8))[1]]
    for ax1,ax2,filename in zip(ax1_list,ax2_list,filelist):
        print(filename)
        fileid = filelist_infos.index(os.path.basename(filename))
        fileitem = list_data[fileid]
        for colname,it in zip(colnames,fileitem):
            print(colname+": "+it)
        cen_filename_id = colnames.index("cen filename")
        kcen_id = colnames.index("kcen")
        lcen_id = colnames.index("lcen")
        rvcen_id = colnames.index("RVcen")
        baryrv_id = colnames.index("barycenter rv")
        hr8799_bary_rv = -float(fileitem[baryrv_id])/1000
        hr8799_rv = -12.6 #+-1.4
        plcen_k,plcen_l = int(round(float(fileitem[kcen_id]))),int(round(float(fileitem[lcen_id])))

        # try:
        if 1:
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),foldername,
                                               os.path.basename(filename).replace(".fits",suffix+".fits")))
            print(hdulist[0].data.shape)
            # # exit()
            # plt.figure(3)
            # from copy import copy
            # tmp = copy(hdulist[0].data[0,0,11,400::,plcen_k,plcen_l])
            # tmp[100-10:100+10] = np.nan
            # print(np.nanmean(tmp))
            # print(np.nanstd(tmp))
            # plt.plot((hdulist[0].data[0,0,11,400::,plcen_k,plcen_l]-np.nanmean(tmp))/np.nanstd(tmp))
            # plt.show()
            logposterior = hdulist[0].data[0,0,9,0:400,plcen_k-1:plcen_k+1,plcen_l-1:plcen_l+1 ]
            # logposterior = hdulist[0].data[0,0,9,400::,plcen_k-1:plcen_k+1,plcen_l-1:plcen_l+1 ]
            ny,nx = hdulist[0].data.shape[4::]
            prihdr = hdulist[0].header
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),foldername,
                                               os.path.basename(filename).replace(".fits",suffix+"_planetRV.fits")))
            planetRVs = hdulist[0].data[0:400]
            # planetRVs = hdulist[0].data[400::]
            try:
                hdulist = pyfits.open(os.path.join(os.path.dirname(filename),foldername,
                                                   os.path.basename(filename).replace(".fits",suffix+"_klgrids.fits")))
                klgrids = hdulist[0].data
                real_k_grid = klgrids[0,:,:]
                real_l_grid = klgrids[1,:,:]
            except:
                real_l_grid,real_k_grid = np.meshgrid(np.arange(nx),np.arange(ny))

            # plt.figure(3)
            # plt.plot(planetRVs,logposterior[:,1,1])

            logposterior = np.exp(logposterior-np.nanmax(logposterior))
            logpost_pos = np.nansum(logposterior,axis=0)
            logpost_wvs = np.nansum(logposterior,axis=(1,2))


            # plt.figure(1)
            plt.sca(ax1)
            plt.plot(planetRVs-hr8799_bary_rv-hr8799_rv,logpost_wvs)
            plt.show()
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
