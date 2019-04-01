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

planet = "b"
# planet = "c"
# planet = "d"
# IFSfilter = "Kbb"
# IFSfilter = "Hbb"



fileinfos_filename = "/data/osiris_data/HR_8799_"+planet+"/fileinfos_Kbb_jb.csv"

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

    reductionname = "reduced_jb"
    filenamefilter = "s*_a*_[0-9][0-9][0-9].fits"
    filelist = glob.glob(os.path.join("/data/osiris_data/HR_8799_"+planet,"*",reductionname,filenamefilter))
    for filename in filelist:
        if filename not in old_filelist:
            new_list_data.append([filename,]+[np.nan,]*(N_col-1))
    print(new_list_data)

if 0: # add spectral band
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

if 0: # add filename
    if 0:
        filename_id = new_colnames.index("filename")
        ifs_filter_id = new_colnames.index("IFS filter")
        filelist = [item[filename_id] for item in new_list_data]
        filelist.sort()
        seqid = 0
        imid = 0
        pastnum = 0
        for k,filename in enumerate(filelist):
            # if "Hbb" in new_list_data[k][ifs_filter_id]:
            #     print("[\"{0}\",,],".format(filename) )
            currnum = int(os.path.basename(filename).split("_a")[1][0:3])
            if currnum ==  pastnum or currnum == pastnum+1:
                imid += 1
            else:
                seqid += 1
                imid = 0
            print("[\"{0}\",{1},{2}],".format(filename,seqid,imid) )
            pastnum = currnum
        exit()
    else:
        sequence_list =[["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a023001_Kbb_020.fits",1,0],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a023002_Kbb_020.fits",1,1],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a023003_Kbb_020.fits",1,2],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a024001_Kbb_020.fits",1,3],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a024002_Kbb_020.fits",1,4],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a024003_Kbb_020.fits",1,5],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a025001_Kbb_020.fits",1,6],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a036001_Kbb_020.fits",2,0],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a037001_Kbb_020.fits",2,1],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a037002_Kbb_020.fits",2,2],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a037003_Kbb_020.fits",2,3],
                        ["/data/osiris_data/HR_8799_b/20090722/reduced_jb/s090722_a038001_Kbb_020.fits",2,4],
                        ["/data/osiris_data/HR_8799_b/20090723/reduced_jb/s090723_a019001_Hbb_020.fits",3,0],
                        ["/data/osiris_data/HR_8799_b/20090723/reduced_jb/s090723_a020001_Hbb_020.fits",3,1],
                        ["/data/osiris_data/HR_8799_b/20090723/reduced_jb/s090723_a021001_Hbb_020.fits",3,2],
                        ["/data/osiris_data/HR_8799_b/20090723/reduced_jb/s090723_a021002_Hbb_020.fits",3,3],
                        ["/data/osiris_data/HR_8799_b/20090723/reduced_jb/s090723_a021003_Hbb_020.fits",3,4],
                        ["/data/osiris_data/HR_8799_b/20090723/reduced_jb/s090723_a029001_Hbb_020.fits",4,0],
                        ["/data/osiris_data/HR_8799_b/20090730/reduced_jb/s090730_a021001_Hbb_020.fits",5,0],
                        ["/data/osiris_data/HR_8799_b/20090730/reduced_jb/s090730_a022001_Hbb_020.fits",5,1],
                        ["/data/osiris_data/HR_8799_b/20090730/reduced_jb/s090730_a022002_Hbb_020.fits",5,2],
                        ["/data/osiris_data/HR_8799_b/20090730/reduced_jb/s090730_a024001_Hbb_020.fits",6,0],
                        ["/data/osiris_data/HR_8799_b/20090730/reduced_jb/s090730_a026001_Hbb_020.fits",7,0],
                        ["/data/osiris_data/HR_8799_b/20090730/reduced_jb/s090730_a027001_Hbb_020.fits",7,1],
                        ["/data/osiris_data/HR_8799_b/20090730/reduced_jb/s090730_a027002_Hbb_020.fits",7,2],
                        ["/data/osiris_data/HR_8799_b/20090730/reduced_jb/s090730_a029001_Hbb_020.fits",8,0],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a016001_Hbb_020.fits",9,0],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a016002_Hbb_020.fits",9,1],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a016003_Hbb_020.fits",9,2],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a020001_Hbb_020.fits",10,0],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a022001_Hbb_020.fits",11,0],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a023001_Hbb_020.fits",11,1],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a024001_Hbb_020.fits",11,2],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a025001_Hbb_020.fits",11,3],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a026001_Hbb_020.fits",11,4],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a027001_Hbb_020.fits",11,5],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a028001_Hbb_020.fits",11,6],
                        ["/data/osiris_data/HR_8799_b/20090903/reduced_jb/s090903_a029001_Hbb_020.fits",11,7],
                        ["/data/osiris_data/HR_8799_b/20100711/reduced_jb/s100711_a016001_Kbb_020.fits",12,0],
                        ["/data/osiris_data/HR_8799_b/20100711/reduced_jb/s100711_a016002_Kbb_020.fits",12,1],
                        ["/data/osiris_data/HR_8799_b/20100711/reduced_jb/s100711_a017001_Kbb_020.fits",12,2],
                        ["/data/osiris_data/HR_8799_b/20100711/reduced_jb/s100711_a018001_Kbb_020.fits",12,3],
                        ["/data/osiris_data/HR_8799_b/20100711/reduced_jb/s100711_a024001_Kbb_020.fits",13,0],
                        ["/data/osiris_data/HR_8799_b/20100711/reduced_jb/s100711_a025001_Kbb_020.fits",13,1],
                        ["/data/osiris_data/HR_8799_b/20100711/reduced_jb/s100711_a026001_Kbb_020.fits",13,2],
                        ["/data/osiris_data/HR_8799_b/20100711/reduced_jb/s100711_a028001_Kbb_020.fits",14,0],
                        ["/data/osiris_data/HR_8799_b/20100711/reduced_jb/s100711_a033001_Kbb_020.fits",15,0],
                        ["/data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a011001_Kbb_020.fits",16,0],
                        ["/data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a012001_Kbb_020.fits",16,1],
                        ["/data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a017001_Kbb_020.fits",17,0],
                        ["/data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a018001_Kbb_020.fits",17,1],
                        ["/data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a019001_Kbb_020.fits",17,2],
                        ["/data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a020001_Kbb_020.fits",17,3],
                        ["/data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a021001_Kbb_020.fits",17,4],
                        ["/data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a028001_Kbb_020.fits",18,0],
                        ["/data/osiris_data/HR_8799_b/20100712/reduced_jb/s100712_a029001_Kbb_020.fits",18,1],
                        ["/data/osiris_data/HR_8799_b/20100713/reduced_jb/s100713_a020001_Hbb_020.fits",19,0],
                        ["/data/osiris_data/HR_8799_b/20100713/reduced_jb/s100713_a021001_Hbb_020.fits",19,1],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a035001_Kbb_020.fits",20,0],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a039001_Kbb_020.fits",21,0],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a040001_Kbb_020.fits",21,1],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a041001_Kbb_020.fits",21,2],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a042001_Kbb_020.fits",21,3],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a043001_Kbb_020.fits",21,4],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a044001_Kbb_020.fits",21,5],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a045001_Kbb_020.fits",21,6],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a053001_Kbb_020.fits",22,0],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a054001_Kbb_020.fits",22,1],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a055001_Kbb_020.fits",22,2],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a056001_Kbb_020.fits",22,3],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a057001_Kbb_020.fits",22,4],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a058001_Kbb_020.fits",22,5],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a059001_Kbb_020.fits",22,6],
                        ["/data/osiris_data/HR_8799_b/20130725/reduced_jb/s130725_a060001_Kbb_020.fits",22,7],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a034001_Kbb_020.fits",23,0],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a035001_Kbb_020.fits",23,1],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a036001_Kbb_020.fits",23,2],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a037001_Kbb_020.fits",23,3],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a038001_Kbb_020.fits",23,4],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a039001_Kbb_020.fits",23,5],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a040001_Kbb_020.fits",23,6],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a041001_Kbb_020.fits",23,7],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a042001_Kbb_020.fits",23,8],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a057001_Kbb_020.fits",24,0],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a058001_Jbb_020.fits",24,1],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a059001_Jbb_020.fits",24,2],
                        ["/data/osiris_data/HR_8799_b/20130726/reduced_jb/s130726_a060001_Jbb_020.fits",24,3],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a036001_Kbb_020.fits",25,0],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a037001_Kbb_020.fits",25,1],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a038001_Kbb_020.fits",25,2],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a039001_Kbb_020.fits",25,3],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a040001_Kbb_020.fits",25,4],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a041001_Kbb_020.fits",25,5],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a042001_Jbb_020.fits",25,6],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a051001_Kbb_020.fits",26,0],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a052001_Jbb_020.fits",26,1],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a053001_Jbb_020.fits",26,2],
                        ["/data/osiris_data/HR_8799_b/20130727/reduced_jb/s130727_a054001_Jbb_020.fits",26,3],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a039002_Kbb_020.fits",27,0],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a040002_Kbb_020.fits",27,1],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a041002_Jbb_020.fits",27,2],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a042002_Jbb_020.fits",27,3],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a043002_Jbb_020.fits",27,4],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a044002_Jbb_020.fits",27,5],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a045002_Jbb_020.fits",27,6],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a046002_Jbb_020.fits",27,7],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a047002_Jbb_020.fits",27,8],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a048002_Jbb_020.fits",27,9],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a049002_Jbb_020.fits",27,10],
                        ["/data/osiris_data/HR_8799_b/20161106/reduced_jb/s161106_a050002_Jbb_020.fits",27,11],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a038002_Kbb_020.fits",28,0],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a039002_Kbb_020.fits",28,1],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a040002_Kbb_020.fits",28,2],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a041002_Jbb_020.fits",28,3],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a042002_Jbb_020.fits",28,4],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a043002_Jbb_020.fits",28,5],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a044002_Jbb_020.fits",28,6],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a045002_Jbb_020.fits",28,7],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a046002_Jbb_020.fits",28,8],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a047002_Jbb_020.fits",28,9],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a048002_Jbb_020.fits",28,10],
                        ["/data/osiris_data/HR_8799_b/20161107/reduced_jb/s161107_a049002_Jbb_020.fits",28,11],
                        ["/data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a043002_Kbb_020.fits",29,0],
                        ["/data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a044002_Jbb_020.fits",29,1],
                        ["/data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a045002_Jbb_020.fits",29,2],
                        ["/data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a046002_Jbb_020.fits",29,3],
                        ["/data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a047002_Jbb_020.fits",29,4],
                        ["/data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a048002_Jbb_020.fits",29,5],
                        ["/data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a049002_Jbb_020.fits",29,6],
                        ["/data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a050002_Jbb_020.fits",29,7],
                        ["/data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a051002_Jbb_020.fits",29,8],
                        ["/data/osiris_data/HR_8799_b/20161108/reduced_jb/s161108_a052002_Jbb_020.fits",29,9],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a032002_Kbb_035.fits",30,0],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a033002_Kbb_035.fits",30,1],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a034002_Jbb_035.fits",30,2],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a035002_Jbb_035.fits",30,3],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a036002_Jbb_035.fits",30,4],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a037002_Kbb_035.fits",30,5],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a038002_Kbb_035.fits",30,6],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a039002_Jbb_035.fits",30,7],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a040002_Jbb_035.fits",30,8],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a041002_Kbb_035.fits",30,9],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a042002_Kbb_035.fits",30,10],
                        ["/data/osiris_data/HR_8799_b/20180722/reduced_jb/s180722_a043002_Jbb_035.fits",30,11]]

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

from scipy.interpolate import interp1d
def get_err_from_posterior(x,posterior):
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    if len(x[0:argmax_post]) < 2:
        lx = np.nan
    else:
        lf = interp1d(cum_posterior[0:argmax_post],x[0:argmax_post],bounds_error=False,fill_value=np.nan)
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx = np.nan
    else:
        rf = interp1d(cum_posterior[argmax_post::],x[argmax_post::],bounds_error=False,fill_value=np.nan)
        rx = rf(1-0.6827)
    return x[argmax_post],(rx-lx)/2.

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
    try:
        rvcensig_id = old_colnames.index("RVcensig")
    except:
        new_colnames.append("RVcensig")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        rvcensig_id = new_colnames.index("RVcensig")

    filename_id = new_colnames.index("filename")
    bary_rv_id = new_colnames.index("barycenter rv")
    ifs_filter_id_id = new_colnames.index("IFS filter")

    # if IFSfilter=="Kbb": #Kbb 1965.0 0.25
    #     CRVAL1 = 1965.
    #     CDELT1 = 0.25
    #     nl=1665
    #     R=4000
    # elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
    #     CRVAL1 = 1473.
    #     CDELT1 = 0.2
    #     nl=1651
    #     R=5000
    # dwv = CDELT1/1000.
    # init_wv = CRVAL1/1000. # wv for first slice in mum

    suffix = "_outputHPF_cutoff40_sherlock_v1_search"
    myfolder = "sherlock/20190324_HPF_only"
    for k,item in enumerate(old_list_data):
        filename = item[filename_id]
        print(filename)
        # if filename == '/data/osiris_data/HR_8799_c/20101104/reduced_jb/s101104_a034001_Kbb_020.fits':
        #     continue
        try:
            print(os.path.join(os.path.dirname(filename),myfolder,
                                           os.path.basename(filename).replace(".fits",suffix+"_planetRV.fits")))
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),myfolder,
                                           os.path.basename(filename).replace(".fits",suffix+"_planetRV.fits")))
        except:
            continue
        planetRV = hdulist[0].data
        print(hdulist[0].data.shape)
        NplanetRV_hd = np.where((planetRV[1::]-planetRV[0:(np.size(planetRV)-1)]) < 0)[0][0]+1
        planetRV_hd = hdulist[0].data[0:NplanetRV_hd]
        planetRV = hdulist[0].data[NplanetRV_hd::]
        # rv_per_pix = 3e5*dwv/(init_wv+dwv*nl//2) # 38.167938931297705

        hdulist = pyfits.open(os.path.join(os.path.dirname(filename),myfolder,
                                           os.path.basename(filename).replace(".fits",suffix+".fits")))
        cube_hd = hdulist[0].data[0,0,0,0:NplanetRV_hd,:,:]
        cube = hdulist[0].data[0,0,0,NplanetRV_hd::,:,:]

        bary_rv = -float(item[bary_rv_id])/1000. # RV in km/s
        rv_star = -12.6#-12.6+-1.4km/s HR 8799 Rob and Simbad

        # print(bary_rv+rv_star)
        guess_rv_id = np.argmin(np.abs(planetRV_hd-(bary_rv+rv_star)))
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
        try:
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


            logposterior = hdulist[0].data[0,0,9,0:NplanetRV_hd,ymax,xmax]
            posterior = np.exp(logposterior-np.nanmax(logposterior))

            new_list_data[k][kcen_id] = ymax
            new_list_data[k][lcen_id] = xmax
            new_list_data[k][rvcen_id],new_list_data[k][rvcensig_id] = get_err_from_posterior(planetRV_hd,posterior)
            new_list_data[k][cen_filename_id] = os.path.join(os.path.dirname(filename),myfolder,
                                               os.path.basename(filename).replace(".fits",suffix+".fits"))
            print(new_list_data[k][rvcen_id],planetRV_hd[zmax],planetRV_hd[zmax]-(bary_rv+rv_star))
        except:
            new_list_data[k][kcen_id] = np.nan
            new_list_data[k][lcen_id] = np.nan
            new_list_data[k][rvcen_id],new_list_data[k][rvcensig_id] = np.nan,np.nan
            new_list_data[k][cen_filename_id] = np.nan


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