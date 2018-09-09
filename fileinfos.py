__author__ = 'jruffio'

import glob
import os
import re
import xml.etree.ElementTree as ET

fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos.xml"
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


if 0:
    OSIRISDATA = "/home/sda/jruffio/osiris_data/"
    if 1:
        foldername = "HR_8799_c"
        sep = 0.950
        telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
        template_spec = os.path.join(OSIRISDATA,"hr8799c_osiris_template.save")
    year = "*"
    reductionname = "reduced_quinn"
    filenamefilter = "s*_a*001_tlc_Kbb_020.fits"

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

        if 0:
            # Sorry hardcoded
            radialfile_list = ["s100715_a025001_tlc_Kbb_020.fits",
                               "s100715_a026001_tlc_Kbb_020.fits",
                               "s100715_a027001_tlc_Kbb_020.fits",
                               "s100715_a028001_tlc_Kbb_020.fits",
                               "s100715_a029001_tlc_Kbb_020.fits"]
            if os.path.basename(filename) in radialfile_list:
                fileelement.set("stardir","down")
            else:
                fileelement.set("stardir","left")

        if 0:
            logfile = os.path.join(os.path.dirname(filename),"sherlock","logs","parallelized_osiris_"+filebasename.replace(".fits",".out"))
            fileobj = open(logfile,"r")
            for line in fileobj:
                if line.startswith("('Center=',"):
                    print(line)
                    # center = re.findall("\d+\.\d+",line)
                    center = line.replace("('Center=', array([","").replace("]))","").split(",")
                    fileelement.set("xADIcen",center[0].strip())
                    fileelement.set("yADIcen",center[1].strip())
                    if fileelement.attrib["stardir"] == "left":
                        fileelement.set("xdefcen",str(19//2-float(fileelement.attrib["sep"])/ 0.0203))
                        fileelement.set("ydefcen",str(64//2))
                    elif fileelement.attrib["stardir"] == "down":
                        fileelement.set("xdefcen",str(19//2))
                        fileelement.set("ydefcen",str(64//2+float(fileelement.attrib["sep"])/ 0.0203))
                    print(fileelement.attrib)
                    break
                    # exit()

        if 0:
            fileelement.set("sep",str(sep))


if 1:
    #hr 8799 c, 20100715
    # science files: s100715_a0[10-21,25-29]001_tlc_Kbb_020.fits
    # telluric hd 210501: s100715_a0[4]001_tlc_Kbb020.fits s100715_a0[5]00[1,2]_tlc_Kbb020.fits
    # sky: s100715_a0[22]00[1,2]_tlc_Kbb020.fits
    visual_planet_coords_hor = [[11,32],[12,27],[12,33],[12,39],[10,33],[9,28],[8,38],[10,32.5],[9,32],[10,33],[10,35],[10,33]] #in order: image 10 to 21
    visual_planet_coords_ver = [[7,34],[5,35],[8,35],[7.5,33],[9.5,34.5]]
    visual_planet_coords=visual_planet_coords_hor+visual_planet_coords_ver

    OSIRISDATA = "/home/sda/jruffio/osiris_data/"
    if 1:
        foldername = "HR_8799_c"
        sep = 0.950
        telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
        template_spec = os.path.join(OSIRISDATA,"hr8799c_osiris_template.save")
    year = "20100715"
    reductionname = "reduced_quinn"
    filenamefilter = "s*_a*001_tlc_Kbb_020.fits"

    filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
    filelist.sort()
    print(filelist)
    print(len(filelist),len(visual_planet_coords))
    # exit()
    for filename,guess_planet in zip(filelist,visual_planet_coords):
        print(filename)
        filebasename = os.path.basename(filename)
        fileelement = planet_c.find(filebasename)

        if fileelement.attrib["stardir"] == "left":
            fileelement.set("xvisucen",str(guess_planet[0]-float(fileelement.attrib["sep"])/ 0.0203))
            fileelement.set("yvisucen",str(guess_planet[1]))
        elif fileelement.attrib["stardir"] == "down":
            fileelement.set("xvisucen",str(guess_planet[0]))
            fileelement.set("yvisucen",str(guess_planet[1]+float(fileelement.attrib["sep"])/ 0.0203))

if 1:
    tree = ET.ElementTree(root)
    with open(fileinfos_filename, "wb") as fh:
        tree.write(fh)
exit()


# HR_8799_c/20110723/reduced_quinn/sherlock/logs/parallelized_osiris_s110723_a025001_tlc_Kbb_020.out
# HR_8799_c/20110723/reduced_quinn/sherlock/polyfit_ADIcenter/
# HR_8799_c/20110723/reduced_quinn/sherlock/polyfit_ADIcenter/s110723_a017001_tlc_Kbb_020_output_centerADI.fits
