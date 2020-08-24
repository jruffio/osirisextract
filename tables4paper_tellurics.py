__author__ = 'jruffio'

import csv
import os
import glob
import numpy as np
import astropy.io.fits as pyfits


# for planet_id,planet in enumerate(["b","c","d"]):
if 1:
    fileinfos_filename = "/data/osiris_data/fileinfos_refstars_jb.csv"
    # filename;MJD-OBS;IFS filter;type;Jmag;Hmag;Kmag;RV Simbad;vsini fixed;star name;hip num;barycenter rv;posterior filename;model filename;vsini;vsini err;rv;rv err;limb dark;limb dark err
    #read file
    with open(fileinfos_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        list_table = list(csv_reader)
        colnames = list_table[0]
        N_col = len(colnames)
        list_data = list_table[1::]

        filename_id = colnames.index("filename")
        mjdobs_id = colnames.index("MJD-OBS")
        bary_rv_id = colnames.index("barycenter rv")
        ifs_filter_id = colnames.index("IFS filter")
        itime_id = colnames.index("itime")
    filelist = np.array([item[filename_id] for item in list_data])
    basenamefilelist = np.array([os.path.basename(filename)  for filename in filelist])
    rmdouble_list_data = []
    for basenamefile in np.unique(basenamefilelist):
        myindex = np.where(basenamefilelist == basenamefile )[0][0]
        rmdouble_list_data.append(list_data[myindex])
    list_data = rmdouble_list_data


    filelist = np.array([item[filename_id] for item in list_data])
    aooff_list =  np.array(["flux only" if "ao_off" in os.path.basename(filename) else "PSF" for filename in filelist])
    date_list = np.array([item[filename_id].split(os.path.sep)[4] for item in list_data])
    scale_list = np.array([int(list_data[0][filename_id].split(os.path.sep)[-1].split("_")[-1].replace(".fits","")) for item in list_data])
    ifs_fitler_list = np.array([item[ifs_filter_id] for item in list_data])
    itime_list = np.array([float(item[itime_id]) if item[itime_id] != "nan" else 0  for item in list_data])

    starname_list =  np.array([filename.split(os.path.sep)[6].replace("_"," ") for filename in filelist])
    isHR8799_list =  np.array(["HR_8799" in filename.split(os.path.sep)[3] for filename in filelist])
    basenamefilelist = np.array([os.path.basename(filename)  for filename in filelist])
    # print(len(basenamefilelist))
    # print(len(np.unique(basenamefilelist)))
    # exit()
    # print(aooff_list)
    # exit()

    # print("\\hline")
    # for myfilter in ["Jbb","Hbb","Kbb"]:
    #     print("{0} & {1} & {2} & {3} & {4:0.1f} &  \\\\ ".format("","all",myfilter,np.size(np.where(ifs_fitler_list==myfilter)[0]),np.sum(itime_list[np.where(ifs_fitler_list==myfilter)])/3600) )

    print("\\hline")
    print("\\hline")

    for date in np.unique(date_list):
        # if date == "20180722":
        #     mynotes = "35mas platescale"
        # elif date == "20090730":
        #     mynotes = "Cooling issue"
        # else:
        #     mynotes = ""
        first_date = True
        for myfilter in ["Hbb","Kbb"]:#["Jbb","Hbb","Kbb"]:
            if myfilter=="Kbb": #Kbb 1965.0 0.25
                CRVAL1 = 1965.
                CDELT1 = 0.25
                nl=1665
                R=4000
            elif myfilter=="Hbb": #Hbb 1651 1473.0 0.2
                CRVAL1 = 1473.
                CDELT1 = 0.2
                nl=1651
                R=4000
            elif myfilter=="Jbb": #Hbb 1651 1473.0 0.2
                CRVAL1 = 1180.
                CDELT1 = 0.15
                nl=1574
                R0=4000
            init_wv = CRVAL1/1000.
            dwv = CDELT1/1000.
            wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)
            # print(wvs[0],(wvs[-1]-wvs[0])/40,dwv)
            # exit()
            dprv = 3e5*dwv/(init_wv+dwv*nl//2)

            formated_date =  date[0:4]+"-"+date[4:6]+"-"+date[6:8]
            for starname in np.unique(starname_list):
                for aooff in np.unique(aooff_list):
                    where_data = np.where((date == date_list)*(myfilter==ifs_fitler_list)*isHR8799_list*(starname_list==starname)*(aooff_list == aooff))

                    if np.size(where_data[0]) != 0:
                        if first_date:
                            print("{0} & {1} & {2} & {3} & {4} & {5}  \\\\ ".format(formated_date,starname,myfilter,np.size(where_data[0]),itime_list[where_data[0][0]],aooff ))
                            first_date = False
                        else:
                            print("{0} & {1} & {2} & {3} & {4} & {5} \\\\ ".format(" ",starname,myfilter,np.size(where_data[0]),itime_list[where_data[0][0]],aooff ))
        # print("\\cline{2-6}")
