__author__ = 'jruffio'

import glob
import os
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np

fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos.xml"
out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
tree = ET.parse(fileinfos_filename)
root = tree.getroot()


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
print(len(filelist))
f,ax_list = plt.subplots(4,len(filelist)//4+1,sharey="row",sharex="col",figsize=(18,0.59*18))
print(len(ax_list))
ax_list = [myax for rowax in ax_list for myax in rowax ]
print(len(ax_list))
for ax,filename in zip(ax_list,filelist):
    print(filename)
    filebasename = os.path.basename(filename)
    fileelement = planet_c.find(filebasename)
    print(fileelement.attrib["xADIcen"],fileelement.attrib["yADIcen"])

    try:
        hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"sherlock","polyfit_ADIcenter",
                                           os.path.basename(filename).replace(".fits","_output_centerADI.fits")))
        image = hdulist[0].data[2,:,:]
        prihdr = hdulist[0].header

        # hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"sherlock","polyfit_ADIcenter",
        #                                    os.path.basename(filename).replace(".fits","_output_defcen.fits")))
        # image = hdulist[0].data[2,:,:]
        # prihdr = hdulist[0].header

        # hdulist = pyfits.open(filename)
        # cube = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        # image = np.nansum(cube,axis=0)
        # prihdr = hdulist[0].header

        plt.sca(ax)
        ny,nx = image.shape
        plt.imshow(image,interpolation="nearest")

        xcen,ycen = float(fileelement.attrib["xADIcen"]),float(fileelement.attrib["yADIcen"])
        import matplotlib.patches as mpatches
        myarrow = mpatches.Arrow(xcen,ycen,nx//2-xcen,ny//2-ycen,color="pink",linestyle="--",linewidth=0.5)
        myarrow.set_clip_on(False)
        # myarrow.set_clip_box(ax.bbox)
        ax.add_artist(myarrow)
        if fileelement.attrib["stardir"] == "left":
            myarrow = mpatches.Arrow(xcen,ycen,float(fileelement.attrib["sep"])/ 0.0203,0,color="red",linestyle="-",linewidth=1)
        elif fileelement.attrib["stardir"] == "down":
            myarrow = mpatches.Arrow(xcen,ycen,0,-float(fileelement.attrib["sep"])/ 0.0203,color="red",linestyle="-",linewidth=1)
        myarrow.set_clip_on(False)
        # myarrow.set_clip_box(ax.bbox)
        ax.add_artist(myarrow)
    except:
        pass

f.subplots_adjust(wspace=0,hspace=0)
plt.show()
exit()
plt.savefig(os.path.join(out_pngs,"HR8799c_raw_ADIcenter.pdf"),bbox_inches='tight')
plt.show()