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

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units as u
from astropy.utils import iers
from astropy.utils.iers import conf as iers_conf
print(iers_conf.iers_auto_url)
#default_iers = iers_conf.iers_auto_url
#print(default_iers)
iers_conf.iers_auto_url = 'https://datacenter.iers.org/data/9/finals2000A.all'
iers_conf.iers_auto_url_mirror = 'ftp://cddis.gsfc.nasa.gov/pub/products/iers/finals2000A.all'
iers.IERS_Auto.open()  # Note the URL

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
    filelist.extend(glob.glob(os.path.join("/data/osiris_data/kap_And","*",reductionname,"*",filenamefilter)))
    filelist.extend(glob.glob(os.path.join("/data/osiris_data/GJ_504_b","*",reductionname,"*",filenamefilter)))
    filelist.extend(glob.glob(os.path.join("/data/osiris_data/HD_1160","*",reductionname,"*",filenamefilter)))
    filelist.sort()
    for filename in filelist:
        if filename not in old_filelist:
            new_list_data.append([filename,]+[np.nan,]*(N_col-1))
    # print(new_list_data)
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
    # print(new_list_data)

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

if 1: # add exposure time
    filename_id = new_colnames.index("filename")
    try:
        itime_id = new_colnames.index("itime")
    except:
        new_colnames.append("itime")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        itime_id = new_colnames.index("itime")

    for k,item in enumerate(new_list_data):
        hdulist = pyfits.open(item[filename_id])
        prihdr0 = hdulist[0].header
        if prihdr0["MJD-OBS"]>57698:
            new_list_data[k][itime_id] = float(prihdr0["ITIME"])/1000
        else:
            new_list_data[k][itime_id] = float(prihdr0["ITIME"])

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
    try:
        hipnum_id = old_colnames.index("hip num")
    except:
        new_colnames.append("hip num")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        hipnum_id = new_colnames.index("hip num")

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
            new_list_data[k][hipnum_id] = 109452
        elif refstar_name == "HIP_1123":
            new_list_data[k][rv_simbad_id] = 0.9 #+-2
            new_list_data[k][vsini_fixed_id] = 75 #+-2.5
            new_list_data[k][type_id] = "A1"
            new_list_data[k][Jmag_id] = 6.186
            new_list_data[k][Hmag_id] = 6.219
            new_list_data[k][Kmag_id] = 6.189
            new_list_data[k][hipnum_id] =1123
        elif refstar_name == "HIP_116886":
            new_list_data[k][rv_simbad_id] = 0#actually dunno np.nan
            new_list_data[k][vsini_fixed_id] = 50#actually dunno np.nan
            new_list_data[k][type_id] = "A5"
            new_list_data[k][Jmag_id] = 9.375
            new_list_data[k][Hmag_id] = 9.212
            new_list_data[k][Kmag_id] = 9.189
            new_list_data[k][hipnum_id] = 116886
        elif refstar_name == "HR_8799":
            new_list_data[k][rv_simbad_id] = -12.6 #
            new_list_data[k][vsini_fixed_id] = 49 #+-2.5
            new_list_data[k][type_id] = "F0"
            new_list_data[k][Jmag_id] = 5.383
            new_list_data[k][Hmag_id] = 5.280
            new_list_data[k][Kmag_id] = 5.240
            new_list_data[k][hipnum_id] = 114189
        elif refstar_name == "BD+14_4774":
            new_list_data[k][rv_simbad_id] = 0#actually dunno np.nan
            new_list_data[k][vsini_fixed_id] = 50#actually dunno np.nan
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 9.291
            new_list_data[k][Hmag_id] = 9.655
            new_list_data[k][Kmag_id] = 9.613
            new_list_data[k][hipnum_id] = np.nan
        elif refstar_name == "HD_7215":
            new_list_data[k][rv_simbad_id] =  -2.1
            new_list_data[k][vsini_fixed_id] = 81.3
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 6.906
            new_list_data[k][Hmag_id] = 6.910
            new_list_data[k][Kmag_id] = 6.945
            new_list_data[k][hipnum_id] = 5671
        elif refstar_name == "HIP_18717":
            new_list_data[k][rv_simbad_id] =  28.5
            new_list_data[k][vsini_fixed_id] = np.nan
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 6.064
            new_list_data[k][Hmag_id] = 6.090
            new_list_data[k][Kmag_id] = 6.074
            new_list_data[k][hipnum_id] = 18717
        elif refstar_name == "HIP_111538":
            new_list_data[k][rv_simbad_id] =  1.6
            new_list_data[k][vsini_fixed_id] = 100#actually dunno np.nan
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 9.393
            new_list_data[k][Hmag_id] = 9.431
            new_list_data[k][Kmag_id] = 9.406
            new_list_data[k][hipnum_id] =111538
        elif refstar_name == "HIP_25453":
            new_list_data[k][rv_simbad_id] = 13.10
            new_list_data[k][vsini_fixed_id] = 133
            new_list_data[k][type_id] = "B9"
            new_list_data[k][Jmag_id] = 6.394
            new_list_data[k][Hmag_id] = 6.418
            new_list_data[k][Kmag_id] = 6.438
            new_list_data[k][hipnum_id] = 25453
        elif refstar_name == "HIP_65599":
            new_list_data[k][rv_simbad_id] =-18
            new_list_data[k][vsini_fixed_id] = 100#actually dunno np.nan
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 7.929
            new_list_data[k][Hmag_id] = 7.975
            new_list_data[k][Kmag_id] = 7.938
            new_list_data[k][hipnum_id] = 65599
        elif refstar_name == "HD_1160":
            new_list_data[k][rv_simbad_id] =12.6
            new_list_data[k][vsini_fixed_id] = np.nan#actually dunno np.nan
            new_list_data[k][type_id] = "A0"
            new_list_data[k][Jmag_id] = 6.983
            new_list_data[k][Hmag_id] = 7.013
            new_list_data[k][Kmag_id] = 7.040
            new_list_data[k][hipnum_id] = 1272
        else:
            new_list_data[k][rv_simbad_id] =  np.nan
            new_list_data[k][vsini_fixed_id] = np.nan
            new_list_data[k][type_id] = np.nan
            new_list_data[k][Jmag_id] = np.nan
            new_list_data[k][Hmag_id] = np.nan
            new_list_data[k][Kmag_id] = np.nan
            new_list_data[k][hipnum_id] = np.nan

if 1: # add barycenter RV
        # hip_id : Hipparcos Catalog ID. (Integer). Epoch will be taken to be Catalogue Epoch or J1991.25
        #         If specified then ra,dec,pmra,pmdec,px, and epoch need not be specified.
        #                         OR / AND
        # ra, dec : RA and Dec of star [degrees].
        # epoch : Epoch of coordinates in Julian Date. Default is J2000 or 2451545.
        # pmra : Proper motion in RA [mas/year]. Eg. PMRA = d(RA)/dt * cos(dec). Default is 0.
        # pmdec : Proper motion in Dec [mas/year]. Default is 0.
        # px : Parallax of target [mas]. Default is 0.
    # from barycorrpy import get_BC_vel
    filename_id = new_colnames.index("filename")
    MJDOBS_id = new_colnames.index("MJD-OBS")
    try:
        bary_rv_id = new_colnames.index("barycenter rv")
    except:
        new_colnames.append("barycenter rv")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        bary_rv_id = new_colnames.index("barycenter rv")

    for k,item in enumerate(new_list_data):
        print(item[bary_rv_id])
        if np.isfinite(float(item[bary_rv_id])):
            continue
        MJDOBS = float(item[MJDOBS_id])
        # print(MJDOBS)
        # if item[starname_id] == "BD+14_4774":
        #     result = get_BC_vel(MJDOBS+2400000.5,ra=334.9042083,dec=14.7468861,pmra=13.3,pmdec=2.370,px=2.3269,obsname="Keck Observatory",ephemeris="de430")
        # else:
        #     print(item[hipnum_id])
        #     result = get_BC_vel(MJDOBS+2400000.5,hip_id=int(item[hipnum_id]),obsname="Keck Observatory",ephemeris="de430")
        # new_list_data[k][bary_rv_id] = result[0][0]

        hdulist = pyfits.open(item[filename_id])
        prihdr0 = hdulist[0].header
        print(float(prihdr0["RA"]),float(prihdr0["DEC"]))
        keck = EarthLocation.from_geodetic(lat=19.8283*u.deg, lon=-155.4783*u.deg, height=4160*u.m)
        sc = SkyCoord(float(prihdr0["RA"]) * u.deg, float(prihdr0["DEC"]) * u.deg)
        barycorr = sc.radial_velocity_correction(obstime=Time(MJDOBS, format="mjd", scale="utc"), location=keck)
        new_list_data[k][bary_rv_id] = barycorr.to(u.m/u.s).value



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
    return x[argmax_post],lx,rx,lx-x[argmax_post],rx-x[argmax_post],argmax_post


if 1:
    try:
        post_filename_id = old_colnames.index("posterior filename")
    except:
        new_colnames.append("posterior filename")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        post_filename_id = new_colnames.index("posterior filename")
    try:
        model_filename_id = old_colnames.index("model filename")
    except:
        new_colnames.append("model filename")
        new_list_data = [item+[np.nan,] for item in new_list_data]
        model_filename_id = new_colnames.index("model filename")

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

    type_id = new_colnames.index("type")
    rv_simbad_id = new_colnames.index("RV Simbad")
    vsini_fixed_id = new_colnames.index("vsini fixed")
    starname_id = new_colnames.index("star name")
    baryrv_id = new_colnames.index("barycenter rv")

    OSIRISDATA = os.path.dirname(fileinfos_filename)

    # ## Combining all the posteriors
    # all_starnames = np.unique([item[starname_id] for item in new_list_data]+["kap_And","51_Eri"])
    # print(all_starnames)
    # for refstar_name in all_starnames:
    #     filelist = glob.glob(os.path.join(OSIRISDATA,"stellar_fits","{0}_*_*_*_posterior.fits".format(refstar_name)))
    #     all_posterior = []
    #     all_logpost_arr = []
    #     all_chi2_arr = []
    #     for myfilename in filelist:
    #         hdulist = pyfits.open(myfilename.replace("_posterior","_rv_samples"))
    #         rv_samples = hdulist[0].data
    #         hdulist = pyfits.open(myfilename.replace("_posterior","_vsini_samples"))
    #         vsini_samples = hdulist[0].data
    #         hdulist = pyfits.open(myfilename)
    #         posterior = hdulist[0].data[0]
    #         posterior /=np.nanmax(posterior)
    #         logpost_arr = hdulist[0].data[1]
    #         chi2_arr = hdulist[0].data[2]
    #         print(posterior.shape,rv_samples.shape,vsini_samples.shape)
    #         all_posterior.append(posterior)
    #         all_logpost_arr.append(logpost_arr)
    #         all_chi2_arr.append(chi2_arr)
    #     combined_posterior = np.prod(all_posterior,axis=0)
    #     combined_posterior/=np.nanmax(combined_posterior)
    #     combined_logpost = np.mean(all_logpost_arr,axis=0)
    #     combined_chi2 = np.sum(all_chi2_arr,axis=0)
    #     # print(combined_posterior.shape)
    #     hdulist = pyfits.HDUList()
    #     hdulist.append(pyfits.PrimaryHDU(data=np.concatenate((combined_posterior[None,:,:,:],combined_logpost[None,:,:,:],combined_chi2[None,:,:,:]))))
    #     try:
    #         hdulist.writeto(os.path.join(OSIRISDATA,"stellar_fits","{0}_combined_posterior.fits".format(refstar_name)), overwrite=True)
    #     except TypeError:
    #         hdulist.writeto(os.path.join(OSIRISDATA,"stellar_fits","{0}_combined_posterior.fits".format(refstar_name)), clobber=True)
    #     hdulist.close()
    #     # plt.imshow(np.nansum(combined_posterior,axis=0))
    #     # plt.show()
    # exit()

    print("coucou")
    cutoff=20
    R0 = 4000
    filename_id = new_colnames.index("filename")
    filelist = [item[filename_id] for item in new_list_data]
    for item,filename in zip(new_list_data,filelist):
        refstar_RV = float(item[rv_simbad_id])
        vsini_fixed = float(item[vsini_fixed_id])
        ref_star_type = item[type_id]
        refstar_name = item[starname_id]
        baryrv = -float(item[baryrv_id])/1000

        splitpostfilename = os.path.basename(filename).split("_")
        # print(splitpostfilename)
        if "ao_off" in filename:
            imtype = "aooff"
            IFSfilter,date = splitpostfilename[2+2],splitpostfilename[0+2][1::]
        else:
            imtype = "psf"
            IFSfilter,date = splitpostfilename[2],splitpostfilename[0][1::]
        if len(glob.glob(os.path.join(OSIRISDATA,"stellar_fits","{0}_{1}_{2}_{3}_posterior.fits".format(refstar_name,IFSfilter,date,imtype))))>0:
            phoenix_folder = os.path.join(OSIRISDATA,"phoenix","PHOENIX-ACES-AGSS-COND-2011")
            phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011_R{0}.fits".format(R0))
            with pyfits.open(phoenix_wv_filename) as hdulist:
                phoenix_wvs = hdulist[0].data
            hdulist = pyfits.open(os.path.join(OSIRISDATA,"stellar_fits","{0}_{1}_{2}_{3}_rv_samples.fits".format(refstar_name,IFSfilter,date,imtype)))
            rv_samples = hdulist[0].data
            hdulist = pyfits.open(os.path.join(OSIRISDATA,"stellar_fits","{0}_{1}_{2}_{3}_vsini_samples.fits".format(refstar_name,IFSfilter,date,imtype)))
            vsini_samples = hdulist[0].data
            hdulist = pyfits.open(os.path.join(OSIRISDATA,"stellar_fits","{0}_{1}_{2}_{3}_posterior.fits".format(refstar_name,IFSfilter,date,imtype)))
            # hdulist = pyfits.open(os.path.join(OSIRISDATA,"stellar_fits","{0}_combined_posterior.fits".format(refstar_name,IFSfilter,date,imtype)))
            posterior = hdulist[0].data[0]
            logpost_arr = hdulist[0].data[1]
            chi2_arr = hdulist[0].data[2]
            posterior_rv_vsini = np.nansum(posterior,axis=0)
            posterior_model = np.nansum(posterior,axis=(1,2))
            with open(os.path.join(OSIRISDATA,"stellar_fits","{0}_{1}_{2}_{3}_models.txt".format(refstar_name,IFSfilter,date,imtype)), 'r') as txtfile:
                grid_refstar_filelist = [s.strip().replace("/scratch/groups/bmacint","/data") for s in txtfile.readlines()]
            rv_posterior = np.nansum(posterior_rv_vsini,axis=1)
            vsini_posterior = np.nansum(posterior_rv_vsini,axis=0)
            bestrv,_,_,bestrv_merr,bestrv_perr,_ = get_err_from_posterior(rv_samples,rv_posterior)
            bestrv_merr = np.abs(bestrv_merr)
            bestvsini,_,_,bestvsini_merr,bestvsini_perr,_ = get_err_from_posterior(vsini_samples,vsini_posterior)
            bestvsini_merr = np.abs(bestvsini_merr)
            best_model_id = np.argmax(posterior_model)

            item[vsini_id],item[vsini_err_id] = bestvsini,np.nanmax([bestvsini_merr,bestvsini_perr])
            item[rv_id],item[rv_err_id] = bestrv,np.nanmax([bestrv_merr,bestrv_perr])
            item[limbdark_id],item[limbdark_err_id] = 0.5,np.nan
            item[post_filename_id] = os.path.join(OSIRISDATA,"stellar_fits","{0}_{1}_{2}_{3}_posterior.fits".format(refstar_name,IFSfilter,date,imtype))
            item[model_filename_id] = grid_refstar_filelist[best_model_id]

            # if os.path.basename(item[post_filename_id]) == "HR_8799_Kbb_130727_aooff_posterior.fits":
            #     print("haha")
            #     exit()

            # print(item)
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

