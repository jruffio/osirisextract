__author__ = 'jruffio'

import glob
import os
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np

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

fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos_jb.xml"
if 0:
    root = ET.Element("HR8799")
    userelement = ET.Element("c")
    root.append(userelement)
    tree = ET.ElementTree(root)
    with open(fileinfos_filename, "wb") as fh:
        tree.write(fh)
    exit()
else:
    tree = ET.parse(fileinfos_filename)
    root = tree.getroot()
    root_children = root.getchildren()
    planet_c = root.find("c")
    # exit()


# planet separation
if 1:
    OSIRISDATA = "/home/sda/jruffio/osiris_data/"
    if 1:
        foldername = "HR_8799_c"
        sep = 0.950
        telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
        template_spec = os.path.join(OSIRISDATA,"hr8799c_osiris_template.save")
    year = "*"
    reductionname = "reduced_jb"
    filenamefilter = "s*_a*001_Kbb_020.fits"

    filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
    filelist.sort()
    for filename in filelist:
        print(filename)
        filebasename = os.path.basename(filename)
        if planet_c.find(filebasename) is None:
            fileelement = ET.Element(filebasename)
            planet_c.append(fileelement)
        else:
            fileelement = planet_c.find(filebasename)

        print(fileelement.tag)
        print(fileelement.attrib)

        fileelement.set("sep","0.950")


if 1:
    tree = ET.ElementTree(root)
    with open(fileinfos_filename, "wb") as fh:
        tree.write(fh)
exit()


# HR_8799_c/20110723/reduced_quinn/sherlock/logs/parallelized_osiris_s110723_a025001_tlc_Kbb_020.out
# HR_8799_c/20110723/reduced_quinn/sherlock/polyfit_ADIcenter/
# HR_8799_c/20110723/reduced_quinn/sherlock/polyfit_ADIcenter/s110723_a017001_tlc_Kbb_020_output_centerADI.fits
