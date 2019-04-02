__author__ = 'jruffio'

import astropy.io.fits as pyfits
import os

#20150726 to Jbb
if 0: # Hack to include a Jbb header into a Kbb raw file
    hdr_filename = "/data/osiris_data/HR_8799_c/20130726/raw_telluric_Jbb/HR_8799/s130726_a052001.fits"

    data_filelist = ["/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a022003.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a022004.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a023003.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a023004.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a024003.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a024004.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a025001.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a026003.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a026004.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a027003.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a027004.fits"]

    for data_filename in data_filelist:
        with pyfits.open(hdr_filename) as hdulist1:
            tmphdr0 = hdulist1[0].header
            tmphdr1 = hdulist1[1].header
            tmphdr2 = hdulist1[2].header
        with pyfits.open(data_filename) as hdulist:
            data0 = hdulist[0].data
            data1 = hdulist[1].data
            data2 = hdulist[2].data
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=data0,header=tmphdr0))
        hdulist.append(pyfits.PrimaryHDU(data=data1,header=tmphdr1))
        hdulist.append(pyfits.PrimaryHDU(data=data2,header=tmphdr2))
        try:
            hdulist.writeto("/data/osiris_data/HR_8799_c/20130726/gc2Jbb/"+os.path.basename(data_filename).replace(".fits","_fakeJbb.fits"), overwrite=True)
        except TypeError:
            hdulist.writeto("/data/osiris_data/HR_8799_c/20130726/gc2Jbb/"+os.path.basename(data_filename).replace(".fits","_fakeJbb.fits"), clobber=True)
        hdulist.close()
    exit()
#20150726 to Kbb
if 0: # Hack to include a Jbb header into a Kbb raw file
    hdr_filename = "/data/osiris_data/HR_8799_c/20130726/raw_telluric_Jbb/HR_8799/s130726_a031001.fits"

    data_filelist = ["/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a022003.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a022004.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a023003.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a023004.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a024003.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a024004.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a025001.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a026003.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a026004.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a027003.fits",
                     "/data/osiris_data/HR_8799_c/20130726/raw_galactic_center/s130726_a027004.fits"]

    for data_filename in data_filelist:
        with pyfits.open(hdr_filename) as hdulist1:
            tmphdr0 = hdulist1[0].header
            tmphdr1 = hdulist1[1].header
            tmphdr2 = hdulist1[2].header
        with pyfits.open(data_filename) as hdulist:
            data0 = hdulist[0].data
            data1 = hdulist[1].data
            data2 = hdulist[2].data
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=data0,header=tmphdr0))
        hdulist.append(pyfits.PrimaryHDU(data=data1,header=tmphdr1))
        hdulist.append(pyfits.PrimaryHDU(data=data2,header=tmphdr2))
        try:
            hdulist.writeto("/data/osiris_data/HR_8799_c/20130726/gc2Jbb/"+os.path.basename(data_filename).replace(".fits","_fakeJbb.fits"), overwrite=True)
        except TypeError:
            hdulist.writeto("/data/osiris_data/HR_8799_c/20130726/gc2Jbb/"+os.path.basename(data_filename).replace(".fits","_fakeJbb.fits"), clobber=True)
        hdulist.close()
    exit()


#20110724 Hbb to Kbb
if 1: # Hack to include a Jbb header into a Kbb raw file
    hdr_filename = "/data/osiris_data/HR_8799_c/20110724/raw_science_Kbb/s110724_a030001.fits"

    data_filelist = ["/data/osiris_data/HR_8799_c/20110724/raw_telluric_Hbb/HD_210501/s110724_a016001.fits",
                     "/data/osiris_data/HR_8799_c/20110724/raw_telluric_Hbb/HD_210501/bad/s110724_a017001.fits",
                     "/data/osiris_data/HR_8799_c/20110724/raw_telluric_Hbb/HD_210501/bad/s110724_a017002.fits",
                     "/data/osiris_data/HR_8799_c/20110724/raw_telluric_Hbb/HD_210501/s110724_a018001.fits"]

    for data_filename in data_filelist:
        with pyfits.open(hdr_filename) as hdulist1:
            tmphdr0 = hdulist1[0].header
            tmphdr1 = hdulist1[1].header
            tmphdr2 = hdulist1[2].header
            tmphdr0["MJD-OBS"] = 55766.4409846203
        with pyfits.open(data_filename) as hdulist:
            data0 = hdulist[0].data
            data1 = hdulist[1].data
            data2 = hdulist[2].data
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=data0,header=tmphdr0))
        hdulist.append(pyfits.PrimaryHDU(data=data1,header=tmphdr1))
        hdulist.append(pyfits.PrimaryHDU(data=data2,header=tmphdr2))
        try:
            hdulist.writeto("/data/osiris_data/HR_8799_c/20110724/test/"+os.path.basename(data_filename).replace(".fits","_fakeKbb.fits"), overwrite=True)
        except TypeError:
            hdulist.writeto("/data/osiris_data/HR_8799_c/20110724/test/"+os.path.basename(data_filename).replace(".fits","_fakeKbb.fits"), clobber=True)
        hdulist.close()
    exit()

# #20110724 to Kbb
# if 0: # Hack to include a Jbb header into a Kbb raw file
#     hdr_filename = "/data/osiris_data/HR_8799_c/20130726/raw_telluric_Jbb/HR_8799/s130726_a031001.fits"
#
#     data_filelist = ["/data/osiris_data/HR_8799_c/20110724/raw_galactic_center/s110724_a010001.fits",
#                      "/data/osiris_data/HR_8799_c/20110724/raw_galactic_center/s110724_a011001.fits",
#                      "/data/osiris_data/HR_8799_c/20110724/raw_galactic_center/s110724_a011002.fits",
#                      "/data/osiris_data/HR_8799_c/20110724/raw_galactic_center/s110724_a011003.fits",
#                      "/data/osiris_data/HR_8799_c/20110724/raw_galactic_center/s110724_a011004.fits",
#                      "/data/osiris_data/HR_8799_c/20110724/raw_galactic_center/s110724_a012001.fits",
#                      "/data/osiris_data/HR_8799_c/20110724/raw_galactic_center/s110724_a012002.fits",
#                      "/data/osiris_data/HR_8799_c/20110724/raw_galactic_center/s110724_a013001.fits",
#                      "/data/osiris_data/HR_8799_c/20110724/raw_galactic_center/s110724_a013002.fits"]
#
#     for data_filename in data_filelist:
#         with pyfits.open(hdr_filename) as hdulist1:
#             tmphdr0 = hdulist1[0].header
#             tmphdr1 = hdulist1[1].header
#             tmphdr2 = hdulist1[2].header
#             tmphdr0["MJD-OBS"] = 0
#         with pyfits.open(data_filename) as hdulist:
#             data0 = hdulist[0].data
#             data1 = hdulist[1].data
#             data2 = hdulist[2].data
#         hdulist = pyfits.HDUList()
#         hdulist.append(pyfits.PrimaryHDU(data=data0,header=tmphdr0))
#         hdulist.append(pyfits.PrimaryHDU(data=data1,header=tmphdr1))
#         hdulist.append(pyfits.PrimaryHDU(data=data2,header=tmphdr2))
#         try:
#             hdulist.writeto("/data/osiris_data/HR_8799_c/20130726/gc2Jbb/"+os.path.basename(data_filename).replace(".fits","_fakeKbb.fits"), overwrite=True)
#         except TypeError:
#             hdulist.writeto("/data/osiris_data/HR_8799_c/20130726/gc2Jbb/"+os.path.basename(data_filename).replace(".fits","_fakeKbb.fits"), clobber=True)
#         hdulist.close()
#     exit()