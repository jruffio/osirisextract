__author__ = 'jruffio'

import glob
import os
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
import csv
from copy import copy

# planet = "c"
IFSfilter = "Kbb"
planet = "d"
# IFSfilter = "Hbb"

fileinfos_filename = "/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/fileinfos_"+IFSfilter+"_jb.csv"

# create file if none exists
if len(glob.glob(fileinfos_filename)) == 0:
    print("Creating new file")
    with open(fileinfos_filename, 'w+') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=';')
        csvwriter.writerows([["filename","MJD-OBS"]])
# exit()

#read file
with open(fileinfos_filename, 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=';')
    old_list_table = list(csv_reader)
    old_colnames = old_list_table[0]
    N_col = len(old_colnames)
    try:
        old_list_data = old_list_table[1::]
    except:
        old_list_data = []
    # np_table_str = np.array(list_table, dtype=np.str)
    # col_names = np_table_str[0]
    # np_tableval_str = np_table_str[1::,1]

# #Save a backup file
# with open(fileinfos_filename.replace(".csv","_backup.csv"), 'w+') as csvfile:
#     csvwriter = csv.writer(csvfile, delimiter=';')
#     csvwriter.writerows(old_list_table)
# exit()

new_colnames = old_colnames
new_list_data = copy(old_list_data)

#print file
for item in old_list_table:
    print(item)

if 0: # add filename
    filename_id = old_colnames.index("filename")
    old_filelist = [item[filename_id] for item in old_list_data]

    reductionname = "reduced_jb"
    filenamefilter = "s*_a*001_"+IFSfilter+"_020.fits"
    filelist = glob.glob(os.path.join("/home/sda/jruffio/osiris_data/HR_8799_"+planet,"*",reductionname,filenamefilter))
    for filename in filelist:
        if filename not in old_filelist:
            new_list_data.append([filename,]+[np.nan,]*(N_col-1))
    print(new_list_data)

if 0: # add MJD-OBS
    filename_id = old_colnames.index("filename")
    MJDOBS_id = old_colnames.index("MJD-OBS")

    for k,item in enumerate(old_list_data):
        hdulist = pyfits.open(item[filename_id])
        prihdr0 = hdulist[0].header
        new_list_data[k][MJDOBS_id] = prihdr0["MJD-OBS"]

if 0: # add barycenter RV
    from barycorrpy import get_BC_vel
    filename_id = old_colnames.index("filename")
    MJDOBS_id = old_colnames.index("MJD-OBS")
    try:
        bary_rv_id = old_colnames.index("barycenter rv")
    except:
        new_colnames.append("barycenter rv")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        bary_rv_id = new_colnames.index("barycenter rv")

    for k,item in enumerate(old_list_data):
        MJDOBS = float(item[MJDOBS_id])
        result = get_BC_vel(MJDOBS+2400000.5,hip_id=114189,obsname="Keck Observatory",ephemeris="de430")
        new_list_data[k][bary_rv_id] = result[0][0]

if 0: # add filename
    if 0:
        filename_id = old_colnames.index("filename")
        filelist = [item[filename_id] for item in old_list_data]
        filelist.sort()
        for filename in filelist:
            print("[\"{0}\",0,0],".format(filename) )
        exit()
    else:
        sequence_list =[["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a091001_Kbb_020.fits",1,0],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a092001_Kbb_020.fits",1,1],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a093001_Kbb_020.fits",1,2],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a094001_Kbb_020.fits",1,3],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a095001_Kbb_020.fits",1,4],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a096001_Kbb_020.fits",1,5],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a097001_Kbb_020.fits",1,6],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a098001_Kbb_020.fits",1,7],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a099001_Kbb_020.fits",1,8],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a100001_Kbb_020.fits",1,9],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a107001_Kbb_020.fits",2,0],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a108001_Kbb_020.fits",2,1],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a109001_Kbb_020.fits",2,2],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a110001_Kbb_020.fits",2,3],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150720/reduced_jb/s150720_a111001_Kbb_020.fits",2,4],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a046001_Kbb_020.fits",3,0],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a047001_Kbb_020.fits",3,1],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a048001_Kbb_020.fits",3,2],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a049001_Kbb_020.fits",3,3],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a050001_Kbb_020.fits",3,4],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a051001_Kbb_020.fits",3,5],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a052001_Kbb_020.fits",3,6],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a053001_Kbb_020.fits",3,7],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a054001_Kbb_020.fits",3,8],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a055001_Kbb_020.fits",3,9],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a056001_Kbb_020.fits",3,10],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a066001_Kbb_020.fits",4,0],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a067001_Kbb_020.fits",4,1],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a068001_Kbb_020.fits",4,2],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150722/reduced_jb/s150722_a069001_Kbb_020.fits",4,3],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a033001_Kbb_020.fits",5,4],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a034001_Kbb_020.fits",5,5],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a035001_Kbb_020.fits",5,6],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a036001_Kbb_020.fits",5,7],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a037001_Kbb_020.fits",5,8],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a038001_Kbb_020.fits",5,9],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a039001_Kbb_020.fits",5,10],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a040001_Kbb_020.fits",5,11],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a041001_Kbb_020.fits",5,12],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a049001_Kbb_020.fits",6,1],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a050001_Kbb_020.fits",6,2],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a051001_Kbb_020.fits",6,3],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a052001_Kbb_020.fits",6,4],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a053001_Kbb_020.fits",6,5],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150723/reduced_jb/s150723_a054001_Kbb_020.fits",6,6],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a017001_Kbb_020.fits",7,0],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a018001_Kbb_020.fits",7,1],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a019001_Kbb_020.fits",7,2],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a020001_Kbb_020.fits",7,3],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a021001_Kbb_020.fits",7,4],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a022001_Kbb_020.fits",7,5],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a023001_Kbb_020.fits",7,6],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a024001_Kbb_020.fits",7,7],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a025001_Kbb_020.fits",7,8],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a026001_Kbb_020.fits",7,9],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a027001_Kbb_020.fits",7,10],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a028001_Kbb_020.fits",7,11],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a029001_Kbb_020.fits",7,12],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a030001_Kbb_020.fits",7,13],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a031001_Kbb_020.fits",7,15],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a032001_Kbb_020.fits",7,16],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a033001_Kbb_020.fits",7,17],
                        ["/home/sda/jruffio/osiris_data/HR_8799_d/20150828/reduced_jb/s150828_a034001_Kbb_020.fits",7,18]]

    try:
        sequence_id = old_colnames.index("sequence")
    except:
        new_colnames.append("sequence")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        sequence_id = new_colnames.index("sequence")
    try:
        sequence_it_id = old_colnames.index("sequence it")
    except:
        new_colnames.append("sequence it")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        sequence_it_id = new_colnames.index("sequence it")
    filename_id = old_colnames.index("filename")

    for k,item in enumerate(old_list_data):
        filename = item[filename_id]
        for seq_filename, sec_num,sec_it in sequence_list:
            if filename == seq_filename:
                new_list_data[k][sequence_id] = sec_num
                new_list_data[k][sequence_it_id] = sec_it

if 0:
    def determine_mosaic_offsets_from_header(prihdr_list):
        OBFMXIM_list = []
        OBFMYIM_list = []
        parang_list = []
        vd_InstAngl_list = []
        for k,prihdr in enumerate(prihdr_list):
            OBFMXIM_list.append(float(prihdr["OBFMXIM"]))
            OBFMYIM_list.append(float(prihdr["OBFMYIM"]))
            parang_list.append(float(prihdr["PARANG"]))
            vd_InstAngl_list.append(float(prihdr["INSTANGL"]))

        vd_C0 = OBFMXIM_list
        vd_C1 = OBFMYIM_list
        md_Coords = np.array([vd_C0,vd_C1])
        vd_InstAngl = np.array(vd_InstAngl_list)

        if "0.02" in prihdr["SSCALE"]:
            d_Scale = 0.0203
        elif "0.035" in prihdr["SSCALE"]:
            d_Scale = 0.0350
        elif "0.05" in prihdr["SSCALE"]:
            d_Scale = 0.0500
        elif "0.1" in prihdr["SSCALE"]:
            d_Scale = 0.1009
        else:
            d_Scale = 0.0203

        vd_CoordsNX =   (md_Coords[0,0] - md_Coords[0,:]) * (35.6 * (0.0397/d_Scale))
        vd_CoordsNY  =  (md_Coords[1,0] - md_Coords[1,:]) * (35.6 * (0.0397/d_Scale))

        vd_InstAngl = np.deg2rad(vd_InstAngl)
        md_Offsets = np.array([vd_CoordsNX * np.cos(vd_InstAngl) + vd_CoordsNY * np.sin(vd_InstAngl),
                       (-1.)*vd_CoordsNX * np.sin(vd_InstAngl) + vd_CoordsNY * np.cos(vd_InstAngl)])

        delta_x = -(md_Offsets[1,:]-md_Offsets[1,0])
        delta_y = -(md_Offsets[0,:]-md_Offsets[0,0])

        return delta_x,delta_y

    try:
        xoffset_id = old_colnames.index("header offset x")
    except:
        new_colnames.append("header offset x")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        xoffset_id = new_colnames.index("header offset x")
    try:
        yoffset_id = old_colnames.index("header offset y")
    except:
        new_colnames.append("header offset y")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        yoffset_id = new_colnames.index("header offset y")
    sequence_id = old_colnames.index("sequence")
    sequence_it_id = old_colnames.index("sequence it")
    filename_id = old_colnames.index("filename")

    old_sequence_num = [int(item[sequence_id]) for item in old_list_data]
    unique_sequence_num = np.unique(old_sequence_num)
    print(unique_sequence_num)
    for seq_num in unique_sequence_num:
        seq_indices = np.where(old_sequence_num == seq_num)[0]
        print(seq_indices)
        if len(seq_indices)<=1:
            new_list_data[seq_indices[0]][xoffset_id] = 0
            new_list_data[seq_indices[0]][yoffset_id] = 0

        seq_it_list = [int(old_list_data[seq_ind][sequence_it_id]) for seq_ind in seq_indices]
        seq_indices=seq_indices[np.argsort(seq_it_list)]

        prihdr_list = []
        for seq_ind in seq_indices:
            item = old_list_data[seq_ind]
            hdulist = pyfits.open(item[filename_id])
            prihdr0 = hdulist[0].header
            prihdr_list.append(prihdr0)

        delta_x,delta_y = determine_mosaic_offsets_from_header(prihdr_list)

        for seq_ind,dx,dy in zip(seq_indices,delta_x,delta_y):
            new_list_data[seq_ind][xoffset_id] = dx
            new_list_data[seq_ind][yoffset_id] = dy

if 1:
    from scipy.signal import correlate2d
    try:
        cen_filename_id = old_colnames.index("cen filename")
    except:
        new_colnames.append("cen filename")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        cen_filename_id = new_colnames.index("cen filename")
    try:
        kcen_id = old_colnames.index("kcen")
    except:
        new_colnames.append("kcen")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        kcen_id = new_colnames.index("kcen")
    try:
        lcen_id = old_colnames.index("lcen")
    except:
        new_colnames.append("lcen")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        lcen_id = new_colnames.index("lcen")
    try:
        rvcen_id = old_colnames.index("RVcen")
    except:
        new_colnames.append("RVcen")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        rvcen_id = new_colnames.index("RVcen")

    filename_id = old_colnames.index("filename")
    bary_rv_id = new_colnames.index("barycenter rv")

    xoffset_id = new_colnames.index("header offset x")
    yoffset_id = new_colnames.index("header offset y")


    filelist = [item[filename_id] for item in old_list_data]
    filelist_sorted = copy(filelist)
    filelist_sorted.sort()
    tmp_list_data = []
    for filename in filelist_sorted:
        tmp_list_data.append(old_list_data[filelist.index(filename)])
    old_list_data=tmp_list_data

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
    init_wv = CRVAL1/1000. # wv for first slice in mum

    f,ax_list = plt.subplots(5,len(old_list_data)//5+1,sharey="row",sharex="col",figsize=(18,0.59*18))
    ax_list = [myax for rowax in ax_list for myax in rowax ]

    suffix = "_outputHPF_cutoff80_sherlock_v0"
    myfolder = "sherlock/20190117_HPFonly"
    for k,item in enumerate(old_list_data):
        filename = item[filename_id]
        # if filename == '/home/sda/jruffio/osiris_data/HR_8799_c/20101104/reduced_jb/s101104_a034001_Kbb_020.fits':
        #     continue
        hdulist = pyfits.open(os.path.join(os.path.dirname(filename),myfolder,
                                           os.path.basename(filename).replace(".fits",suffix+"_wvshifts.fits")))
        wvshifts = hdulist[0].data
        Nwvshifts_hd = np.where((wvshifts[1::]-wvshifts[0:(np.size(wvshifts)-1)]) < 0)[0][0]+1
        wvshifts_hd = hdulist[0].data[0:Nwvshifts_hd]
        wvshifts = hdulist[0].data[Nwvshifts_hd::]
        rv_per_pix = 3e5*dwv/(init_wv+dwv*nl//2) # 38.167938931297705
        rvshifts_hd = wvshifts_hd/dwv*rv_per_pix
        rvshifts = hdulist[0].data[Nwvshifts_hd::]

        new_list_data[k][cen_filename_id] = os.path.join(os.path.dirname(filename),myfolder,
                                           os.path.basename(filename).replace(".fits",suffix+".fits"))
        hdulist = pyfits.open(os.path.join(os.path.dirname(filename),myfolder,
                                           os.path.basename(filename).replace(".fits",suffix+".fits")))
        cube_hd = hdulist[0].data[2,0:Nwvshifts_hd,:,:]
        cube = hdulist[0].data[2,Nwvshifts_hd::,:,:]

        bary_rv = -float(item[bary_rv_id])/1000. # RV in km/s
        rv_star = -12.6#-12.6+-1.4km/s HR 8799 Rob and Simbad

        # print(bary_rv+rv_star)
        guess_rv_id = np.argmin(np.abs(rvshifts_hd-(bary_rv+rv_star)))
        guess_rv_im = copy(cube_hd[guess_rv_id,:,:])
        ny,nx = guess_rv_im.shape
        nan_mask_boxsize = 5
        guess_rv_im[np.where(np.isnan(correlate2d(guess_rv_im,np.ones((nan_mask_boxsize,nan_mask_boxsize)),mode="same")))] = np.nan
        guess_rv_im[0:nan_mask_boxsize//2,:] = np.nan
        guess_rv_im[-nan_mask_boxsize//2+1::,:] = np.nan
        guess_rv_im[:,0:nan_mask_boxsize//2] = np.nan
        guess_rv_im[:,-nan_mask_boxsize//2+1::] = np.nan

        # plt.imshow(guess_rv_im)
        # plt.show()
        guesspos = np.unravel_index(np.nanargmax(guess_rv_im),guess_rv_im.shape)
        guess_y,guess_x = guesspos

        cube_hd_cp = copy(cube_hd)
        cube_hd_cp[:,0:np.max([0,(guess_y-5)]),:] = np.nan
        cube_hd_cp[:,np.min([ny,(guess_y+5)])::,:] = np.nan
        cube_hd_cp[:,:,0:np.max([0,(guess_x-5)])] = np.nan
        cube_hd_cp[:,:,np.min([nx,(guess_x+5)])::] = np.nan

        # plt.imshow(cube_hd_cp[100,:,:])
        # plt.show()

        zmax,ymax,xmax = np.unravel_index(np.nanargmax(cube_hd_cp),cube_hd.shape)

        new_list_data[k][kcen_id] = ymax
        new_list_data[k][lcen_id] = xmax
        new_list_data[k][rvcen_id] = rvshifts_hd[zmax]


        ax = ax_list[k]
        image = copy(cube_hd[guess_rv_id,:,:])

        plt.sca(ax)
        ny,nx = image.shape
        plt.imshow(image,interpolation="nearest")
        plt.clim([0,50])

        xoffset = float(item[xoffset_id])
        yoffset = float(item[yoffset_id])
        print(xoffset,yoffset)
        plt.title("{0}".format(k))
        print(k,filename)
        if xoffset != 0 or yoffset != 0:
            arrow = plt.arrow(10,32,xoffset,yoffset,color="red")
            ax.add_artist(arrow)
        else:
            circle = plt.Circle((10,32),5,color="red", fill=False)
            ax.add_artist(circle)
    plt.show()


print("NEW")
for item in new_list_data:
    print(item)
print(new_colnames)

exit()


#Save NEW file
with open(fileinfos_filename, 'w+') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerows([new_colnames])
    csvwriter.writerows(new_list_data)
exit()




with open(fileinfos_filename, 'w+') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerows([col_names])
    csvwriter.writerows(np_tableval_str)


#
# if 0:
#     root = ET.Element("HR8799")
#     userelement = ET.Element("c")
#     root.append(userelement)
#     tree = ET.ElementTree(root)
#     with open(fileinfos_filename, "wb") as fh:
#         tree.write(fh)
#     exit()
# else:
#     tree = ET.parse(fileinfos_filename)
#     root = tree.getroot()
#     root_children = root.getchildren()
#     planet_c = root.find("c")
#     # exit()
#
#
# # planet separation
# if 1:
#     OSIRISDATA = "/home/sda/jruffio/osiris_data/"
#     if 1:
#         foldername = "HR_8799_c"
#         sep = 0.950
#         telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
#         template_spec = os.path.join(OSIRISDATA,"hr8799c_osiris_template.save")
#     year = "*"
#     reductionname = "reduced_jb"
#     filenamefilter = "s*_a*001_Kbb_020.fits"
#
#     filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
#     filelist.sort()
#     for filename in filelist:
#         print(filename)
#         filebasename = os.path.basename(filename)
#         if planet_c.find(filebasename) is None:
#             fileelement = ET.Element(filebasename)
#             planet_c.append(fileelement)
#         else:
#             fileelement = planet_c.find(filebasename)
#
#         print(fileelement.tag)
#         print(fileelement.attrib)
#
#         fileelement.set("sep","0.950")
#
#
# if 1:
#     tree = ET.ElementTree(root)
#     with open(fileinfos_filename, "wb") as fh:
#         tree.write(fh)
# exit()
#
#
# # HR_8799_c/20110723/reduced_quinn/sherlock/logs/parallelized_osiris_s110723_a025001_tlc_Kbb_020.out
# # HR_8799_c/20110723/reduced_quinn/sherlock/polyfit_ADIcenter/
# # HR_8799_c/20110723/reduced_quinn/sherlock/polyfit_ADIcenter/s110723_a017001_tlc_Kbb_020_output_centerADI.fits
