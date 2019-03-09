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

fileinfos_filename = "/home/sda/jruffio/osiris_data/fileinfos_refstars_jb.csv"

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

    reductionname = "reduced_telluric_jb"
    filenamefilter = "s*_a*_*_020.fits"
    filelist = glob.glob(os.path.join("/home/sda/jruffio/osiris_data/HR_8799_*","*",reductionname,"*",filenamefilter))
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

