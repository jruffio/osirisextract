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
    print(root.find("c").tag)
    # exit()


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
planet_c = root.find("c")
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

    if 1:
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
    tree = ET.ElementTree(root)
    with open(fileinfos_filename, "wb") as fh:
        tree.write(fh)
exit()


# HR_8799_c/20110723/reduced_quinn/sherlock/logs/parallelized_osiris_s110723_a025001_tlc_Kbb_020.out
# HR_8799_c/20110723/reduced_quinn/sherlock/polyfit_ADIcenter/
# HR_8799_c/20110723/reduced_quinn/sherlock/polyfit_ADIcenter/s110723_a017001_tlc_Kbb_020_output_centerADI.fits
