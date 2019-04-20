__author__ = 'jruffio'

import csv
import os
import glob
import numpy as np


for planet_id,planet in enumerate(["b","c","d"]):
    fileinfos_filename = "/data/osiris_data/HR_8799_"+planet+"/fileinfos_Kbb_jb.csv"

    #read file
    with open(fileinfos_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        list_table = list(csv_reader)
        colnames = list_table[0]
        N_col = len(colnames)
        list_data = list_table[1::]

        try:
            cen_filename_id = colnames.index("cen filename")
            kcen_id = colnames.index("kcen")
            lcen_id = colnames.index("lcen")
            rvcen_id = colnames.index("RVcen")
            rvcensig_id = colnames.index("RVcensig")
        except:
            pass
        filename_id = colnames.index("filename")
        mjdobs_id = colnames.index("MJD-OBS")
        bary_rv_id = colnames.index("barycenter rv")
        ifs_filter_id = colnames.index("IFS filter")
        xoffset_id = colnames.index("header offset x")
        yoffset_id = colnames.index("header offset y")
        sequence_id = colnames.index("sequence")
        status_id = colnames.index("status")
        wvsolerr_id = colnames.index("wv sol err")
        snr_id = colnames.index("snr")
        itime_id = colnames.index("itime")
    filelist = np.array([item[filename_id] for item in list_data])
    out_filelist = np.array([item[cen_filename_id] for item in list_data])
    kcen_list = np.array([int(item[kcen_id]) if item[kcen_id] != "nan" else 0 for item in list_data])
    lcen_list = np.array([int(item[lcen_id]) if item[lcen_id] != "nan" else 0  for item in list_data])
    date_list = np.array([item[filename_id].split(os.path.sep)[4] for item in list_data])
    scale_list = np.array([int(list_data[0][filename_id].split(os.path.sep)[-1].split("_")[-1].replace(".fits","")) for item in list_data])
    ifs_fitler_list = np.array([item[ifs_filter_id] for item in list_data])
    snr_list = np.array([float(item[snr_id]) if item[snr_id] != "nan" else 0  for item in list_data])
    itime_list = np.array([float(item[itime_id]) if item[itime_id] != "nan" else 0  for item in list_data])

    # print("\\hline")
    # for myfilter in ["Jbb","Hbb","Kbb"]:
    #     print("{0} & {1} & {2} & {3} & {4:0.1f} &  \\\\ ".format("","all",myfilter,np.size(np.where(ifs_fitler_list==myfilter)[0]),np.sum(itime_list[np.where(ifs_fitler_list==myfilter)])/3600) )

    print("\\hline")
    print("\\hline")
    for date in np.unique(date_list):
        if date == "20180722":
            mynotes = "35mas platescale"
        elif date == "20090730":
            mynotes = "Cooling issue"
        else:
            mynotes = ""
        first_date = True
        for myfilter in ["Jbb","Hbb","Kbb"]:
            formated_date =  date[0:4]+"-"+date[4:6]+"-"+date[6:8]
            where_data = np.where((date == date_list)*(myfilter==ifs_fitler_list))
            if np.size(where_data[0]) != 0:
                if first_date:
                    print("{0} & {1} & {2} & {3} & {4:0.1f} & {5}  \\\\ ".format("",formated_date,myfilter,np.size(where_data[0]),np.sum(itime_list[where_data[0]])/3600,mynotes) )
                    first_date = False
                else:
                    print("{0} & {1} & {2} & {3} & {4:0.1f} & {5}  \\\\ ".format("","",myfilter,np.size(where_data[0]),np.sum(itime_list[where_data[0]])/3600,mynotes) )
        # print("\\cline{2-6}")
