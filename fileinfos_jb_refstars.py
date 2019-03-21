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

if 0: # add filename
    filename_id = new_colnames.index("filename")
    old_filelist = [item[filename_id] for item in new_list_data]

    reductionname = "reduced_telluric_jb"
    filenamefilter = "s*_a*_*_020.fits"
    filelist = glob.glob(os.path.join("/data/osiris_data/HR_8799_*","*",reductionname,"*",filenamefilter))
    for filename in filelist:
        if filename not in old_filelist:
            new_list_data.append([filename,]+[np.nan,]*(N_col-1))
    print(new_list_data)

if 0: # add filename for ao off
    filename_id = new_colnames.index("filename")
    old_filelist = [item[filename_id] for item in new_list_data]

    reductionname = "reduced_telluric_jb"
    filenamefilter = "ao_off_s*_a*_*_020.fits"
    filelist = glob.glob(os.path.join("/data/osiris_data/HR_8799_*","*",reductionname,"*",filenamefilter))
    for filename in filelist:
        if filename not in old_filelist:
            new_list_data.append([filename,]+[np.nan,]*(N_col-1))
    print(new_list_data)

#sort files
if 0:
    filename_id = new_colnames.index("filename")
    filelist = [item[filename_id] for item in new_list_data]
    filelist_sorted = copy(filelist)
    filelist_sorted.sort()
    if 0:
        for filename in filelist_sorted:
            print('["'+os.path.join("/data/osiris_data/HR_8799_b/20090722/reduced_jb",filename)+'",,],')
        exit()
    print(len(filelist_sorted)) #37
    # exit()
    new_new_list_data = []
    for filename in filelist_sorted:
        new_new_list_data.append(new_list_data[filelist.index(filename)])

    new_list_data = new_new_list_data

if 0: # add MJD-OBS
    filename_id = new_colnames.index("filename")
    MJDOBS_id = new_colnames.index("MJD-OBS")

    for k,item in enumerate(new_list_data):
        hdulist = pyfits.open(item[filename_id])
        prihdr0 = hdulist[0].header
        new_list_data[k][MJDOBS_id] = prihdr0["MJD-OBS"]

if 0: # add barycenter RV
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
if 1:
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
exit()

#Save NEW file
with open(fileinfos_filename, 'w+') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerows([new_colnames])
    csvwriter.writerows(new_list_data)
exit()

