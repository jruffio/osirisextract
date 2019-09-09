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

fileinfos_filename = "/data/osiris_data/fileinfos_refstars_jb.csv"

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

if 1: # add filename
    filename_id = new_colnames.index("filename")
    old_filelist = [item[filename_id] for item in new_list_data]

    reductionname = "reduced_telluric_jb"
    filenamefilter = "s*_a*_*_[0-9][0-9][0-9].fits"
    filelist = glob.glob(os.path.join("/data/osiris_data/HR_8799_*","*",reductionname,"*",filenamefilter))
    filelist.extend(glob.glob(os.path.join("/data/osiris_data/51_Eri_*","*",reductionname,"*",filenamefilter)))
    filelist.sort()
    for filename in filelist:
        if filename not in old_filelist:
            new_list_data.append([filename,]+[np.nan,]*(N_col-1))
    print(new_list_data)
    # exit()

if 1: # add filename for ao off
    filename_id = new_colnames.index("filename")
    old_filelist = [item[filename_id] for item in new_list_data]

    reductionname = "reduced_telluric_jb"
    filenamefilter = "ao_off_s*_a*_*_020.fits"
    filelist = glob.glob(os.path.join("/data/osiris_data/HR_8799_*","*",reductionname,"*",filenamefilter))
    filelist.extend(glob.glob(os.path.join("/data/osiris_data/51_Eri_*","*",reductionname,"*",filenamefilter)))
    filelist.sort()
    for filename in filelist:
        if filename not in old_filelist:
            new_list_data.append([filename,]+[np.nan,]*(N_col-1))
    print(new_list_data)

#sort files
if 1:
    filename_id = new_colnames.index("filename")
    filelist = [item[filename_id] for item in new_list_data]
    filelist_sorted = copy(filelist)
    filelist_sorted.sort()
    print(len(filelist_sorted)) #37
    # exit()
    new_new_list_data = []
    for filename in filelist_sorted:
        new_new_list_data.append(new_list_data[filelist.index(filename)])

    new_list_data = new_new_list_data

if 1: # add MJD-OBS
    filename_id = new_colnames.index("filename")
    MJDOBS_id = new_colnames.index("MJD-OBS")

    for k,item in enumerate(new_list_data):
        hdulist = pyfits.open(item[filename_id])
        prihdr0 = hdulist[0].header
        new_list_data[k][MJDOBS_id] = prihdr0["MJD-OBS"]

if 1: # add barycenter RV
    from barycorrpy import get_BC_vel
    filename_id = new_colnames.index("filename")
    MJDOBS_id = new_colnames.index("MJD-OBS")
    try:
        bary_rv_id = new_colnames.index("barycenter rv")
    except:
        new_colnames.append("barycenter rv")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        bary_rv_id = new_colnames.index("barycenter rv")

    for k,item in enumerate(new_list_data):
        MJDOBS = float(item[MJDOBS_id])
        result = get_BC_vel(MJDOBS+2400000.5,hip_id=114189,obsname="Keck Observatory",ephemeris="de430")
        new_list_data[k][bary_rv_id] = result[0][0]

if 1: # add spectral band
    filename_id = new_colnames.index("filename")
    try:
        ifs_filter_id = new_colnames.index("IFS filter")
    except:
        new_colnames.append("IFS filter")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        ifs_filter_id = new_colnames.index("IFS filter")

    for k,item in enumerate(new_list_data):
        if "_Jbb_" in os.path.basename(item[filename_id]):
            new_list_data[k][ifs_filter_id] = "Jbb"
        if "_Hbb_" in os.path.basename(item[filename_id]):
            new_list_data[k][ifs_filter_id] = "Hbb"
        if "_Kbb_" in os.path.basename(item[filename_id]):
            new_list_data[k][ifs_filter_id] = "Kbb"

# Add star Simbad info
if 1:
    try:
        type_id = old_colnames.index("type")
    except:
        new_colnames.append("type")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        type_id = new_colnames.index("type")
    try:
        Jmag_id = old_colnames.index("Jmag")
    except:
        new_colnames.append("Jmag")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        Jmag_id = new_colnames.index("Jmag")
    try:
        Hmag_id = old_colnames.index("Hmag")
    except:
        new_colnames.append("Hmag")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        Hmag_id = new_colnames.index("Hmag")
    try:
        Kmag_id = old_colnames.index("Kmag")
    except:
        new_colnames.append("Kmag")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        Kmag_id = new_colnames.index("Kmag")
    try:
        rv_simbad_id = old_colnames.index("RV Simbad")
    except:
        new_colnames.append("RV Simbad")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        rv_simbad_id = new_colnames.index("RV Simbad")
    try:
        vsini_fixed_id = old_colnames.index("vsini fixed")
    except:
        new_colnames.append("vsini fixed")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        vsini_fixed_id = new_colnames.index("vsini fixed")
    try:
        starname_id = old_colnames.index("star name")
    except:
        new_colnames.append("star name")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        starname_id = new_colnames.index("star name")

    filename_id = new_colnames.index("filename")

    for k,item in enumerate(new_list_data):
        filename = item[filename_id]

        refstar_name = os.path.dirname(filename).split(os.path.sep)[-1]
        new_list_data[k][starname_id] = refstar_name

        if refstar_name == "HD_210501":
            new_list_data[k][rv_simbad_id] = -20.20 #+-2.5
            new_list_data[k][vsini_fixed_id] = 100 #+-2.5
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 7.615
            new_list_data[k][Hmag_id] = 7.606
            new_list_data[k][Kmag_id] = 7.597
        elif refstar_name == "HIP_1123":
            new_list_data[k][rv_simbad_id] = 0.9 #+-2
            new_list_data[k][vsini_fixed_id] = 75 #+-2.5
            new_list_data[k][type_id] = "A1"
            new_list_data[k][Jmag_id] = 6.186
            new_list_data[k][Hmag_id] = 6.219
            new_list_data[k][Kmag_id] = 6.189
        elif refstar_name == "HIP_116886":
            new_list_data[k][rv_simbad_id] = 0#actually dunno np.nan
            new_list_data[k][vsini_fixed_id] = 50#actually dunno np.nan
            new_list_data[k][type_id] = "A5"
            new_list_data[k][Jmag_id] = 9.375
            new_list_data[k][Hmag_id] = 9.212
            new_list_data[k][Kmag_id] = 9.189
        elif refstar_name == "HR_8799":
            new_list_data[k][rv_simbad_id] = -12.6 #
            new_list_data[k][vsini_fixed_id] = 49 #+-2.5
            new_list_data[k][type_id] = "F0"
            new_list_data[k][Jmag_id] = 5.383
            new_list_data[k][Hmag_id] = 5.280
            new_list_data[k][Kmag_id] = 5.240
        elif refstar_name == "BD+14_4774":
            new_list_data[k][rv_simbad_id] = 0#actually dunno np.nan
            new_list_data[k][vsini_fixed_id] = 50#actually dunno np.nan
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 9.291
            new_list_data[k][Hmag_id] = 9.655
            new_list_data[k][Kmag_id] = 9.613
        elif refstar_name == "HD_7215":
            new_list_data[k][rv_simbad_id] =  -2.1
            new_list_data[k][vsini_fixed_id] = 81.3
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 6.906
            new_list_data[k][Hmag_id] = 6.910
            new_list_data[k][Kmag_id] = 6.945
        elif refstar_name == "HIP_18717":
            new_list_data[k][rv_simbad_id] =  28.5
            new_list_data[k][vsini_fixed_id] = np.nan
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 6.064
            new_list_data[k][Hmag_id] = 6.090
            new_list_data[k][Kmag_id] = 6.074
        elif refstar_name == "HIP_111538":
            new_list_data[k][rv_simbad_id] =  1.6
            new_list_data[k][vsini_fixed_id] = np.nan
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 9.393
            new_list_data[k][Hmag_id] = 9.431
            new_list_data[k][Kmag_id] = 9.406
        elif refstar_name == "HIP_25453":
            new_list_data[k][rv_simbad_id] = 13.10
            new_list_data[k][vsini_fixed_id] = 133
            new_list_data[k][type_id] = "B9"
            new_list_data[k][Jmag_id] = 6.394
            new_list_data[k][Hmag_id] = 6.418
            new_list_data[k][Kmag_id] = 6.438
        else:
            new_list_data[k][rv_simbad_id] =  np.nan
            new_list_data[k][vsini_fixed_id] = np.nan
            new_list_data[k][type_id] = np.nan
            new_list_data[k][Jmag_id] = np.nan
            new_list_data[k][Hmag_id] = np.nan
            new_list_data[k][Kmag_id] = np.nan

from scipy.interpolate import interp1d
def get_err_from_posterior(x,posterior):
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    lf = interp1d(cum_posterior[0:argmax_post],x[0:argmax_post],bounds_error=False,fill_value=np.nan)
    rf = interp1d(cum_posterior[argmax_post::],x[argmax_post::],bounds_error=False,fill_value=np.nan)
    return x[argmax_post],(rf(1-0.6827)-lf(1-0.6827))/2.
if 0:
    try:
        post_filename_id = old_colnames.index("posterior filename")
    except:
        new_colnames.append("posterior filename")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        post_filename_id = new_colnames.index("posterior filename")

    try:
        vsini_id = old_colnames.index("vsini")
    except:
        new_colnames.append("vsini")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        vsini_id = new_colnames.index("vsini")
    try:
        vsini_err_id = old_colnames.index("vsini err")
    except:
        new_colnames.append("vsini err")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        vsini_err_id = new_colnames.index("vsini err")

    try:
        rv_id = old_colnames.index("rv")
    except:
        new_colnames.append("rv")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        rv_id = new_colnames.index("rv")
    try:
        rv_err_id = old_colnames.index("rv err")
    except:
        new_colnames.append("rv err")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        rv_err_id = new_colnames.index("rv err")

    try:
        limbdark_id = old_colnames.index("limb dark")
    except:
        new_colnames.append("limb dark")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        limbdark_id = new_colnames.index("limb dark")
    try:
        limbdark_err_id = old_colnames.index("limb dark err")
    except:
        new_colnames.append("limb dark err")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        limbdark_err_id = new_colnames.index("limb dark err")

    print("coucou")
    cutoff=20
    filename_id = new_colnames.index("filename")
    filelist = [item[filename_id] for item in new_list_data]
    for item,filename in zip(new_list_data,filelist):
        print(filename)
        if "ao_off" in filename:
            spec_filename = filename.replace(".fits","_spec_v2_cutoff{0}_logpost.fits".format(cutoff))
        else:
            spec_filename = filename.replace(".fits","_psfs_repaired_spec_v2_cutoff{0}_logpost.fits".format(cutoff))
        if len(glob.glob(spec_filename))>0:
            with pyfits.open(spec_filename) as hdulist:
                logposterior = hdulist[0].data
            with pyfits.open(spec_filename.replace(".fits","_vsini.fits")) as hdulist:
                vsini_vec = hdulist[0].data
            with pyfits.open(spec_filename.replace(".fits","_limbdar.fits")) as hdulist:
                limbdark_vec = hdulist[0].data
            with pyfits.open(spec_filename.replace(".fits","_rv.fits")) as hdulist:
                RV_vec = hdulist[0].data

            posterior = np.exp(logposterior-np.nanmax(logposterior))
            print(posterior.shape)

            vsini_post = np.sum(posterior,axis=(0,2))
            item[vsini_id],item[vsini_err_id] = get_err_from_posterior(vsini_vec,vsini_post)
            limbdark_post = np.sum(posterior,axis=(1,2))
            item[limbdark_id],item[limbdark_err_id] = limbdark_vec[np.argmax(limbdark_post)],np.nan
            rv_post = np.sum(posterior,axis=(0,1))
            item[rv_id],item[rv_err_id] = get_err_from_posterior(RV_vec,rv_post)

            item[post_filename_id] = spec_filename
            print(item)
        else:
            item[post_filename_id] = np.nan
            item[vsini_id] = np.nan
            item[vsini_err_id] = np.nan
            item[rv_id] = np.nan
            item[rv_err_id] = np.nan
            item[limbdark_id] = np.nan
            item[limbdark_err_id] = np.nan
print("NEW")
for item in new_list_data:
    print(item)
print(new_colnames)
# exit()

#Save NEW file
with open(fileinfos_filename, 'w+') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerows([new_colnames])
    csvwriter.writerows(new_list_data)
exit()

