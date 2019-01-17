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
year = "*"#"20101104"
reductionname = "reduced_quinn"
filenamefilter = "s*_a*001_tlc_Kbb_020.fits"

from astropy.stats import mad_std

# HPF only
if 1:
    suffix = "_outputHPF_cutoff80_new_sig_phoenix"
    planet = "c"
    # planet = "d"
    IFSfilter = "Kbb"
    # IFSfilter = "Hbb"
    fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos_jb.xml"
    out_pngs = os.path.join("/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/")#"/home/sda/jruffio/pyOSIRIS/figures/"
    tree = ET.parse(fileinfos_filename)
    root = tree.getroot()
    reductionname = "reduced_jb"
    filenamefilter = "s*_a*001_"+IFSfilter+"_020.fits"
    filelist = glob.glob(os.path.join(OSIRISDATA,"HR_8799_"+planet,year,reductionname,filenamefilter))
    filelist.sort()
    planet_c = root.find("c")
    print(len(filelist))
    f,ax_list = plt.subplots(4,len(filelist)//4+1,sharey="row",sharex="col",figsize=(18,0.59*18))
    ax_list = [myax for rowax in ax_list for myax in rowax ]
    # f,ax_list = plt.subplots(3,8,sharey="row",sharex="col",figsize=(2*8,15))
    # ax_list = [myax for rowax in ax_list for myax in rowax ]
    for ax,filename in zip(ax_list,filelist):
        print(filename)
        # filebasename = os.path.basename(filename)
        # fileelement = planet_c.find(filebasename)
        # print(fileelement.attrib["xADIcen"],fileelement.attrib["yADIcen"])

        try:
        # if 1:
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"20181205_HPF_only",
                                               os.path.basename(filename).replace(".fits",suffix+".fits")))
            image = hdulist[0].data[3,:,:]
            prihdr = hdulist[0].header

            plt.sca(ax)
            ny,nx = image.shape
            plt.imshow(image,interpolation="nearest")
            plt.clim([np.nanmedian(image),np.nanmax(image[np.where(np.isfinite(image))])/2.0])
            # plt.clim([np.nanmedian(image),np.nanmax(image[np.where(np.isfinite(image))])])

            image[np.where(~np.isfinite(image))] = 0
            maxind = np.unravel_index(np.argmax(image),image.shape)
            circle = plt.Circle(maxind[::-1],5,color="red", fill=False)
            ax.add_artist(circle)
            # exit()
        except:
            pass

    f.subplots_adjust(wspace=0,hspace=0)
    print("Saving "+os.path.join(out_pngs,"HR8799"+planet+suffix+"_"+IFSfilter+".pdf"))
    plt.savefig(os.path.join(out_pngs,"HR8799"+planet+suffix+"_"+IFSfilter+".pdf"),bbox_inches='tight')
    plt.savefig(os.path.join(out_pngs,"HR8799"+planet+suffix+"_"+IFSfilter+".png"),bbox_inches='tight')

    plt.figure(2)
    tmp = np.zeros(300)
    for filename in filelist:
        print(filename)
        # filebasename = os.path.basename(filename)
        # fileelement = planet_c.find(filebasename)
        # print(fileelement.attrib["xADIcen"],fileelement.attrib["yADIcen"])

        try:
        # if 1:
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"20181205_HPF_only",
                                               os.path.basename(filename).replace(".fits",suffix+".fits")))
            image = hdulist[0].data[3,:,:]
            image[np.where(~np.isfinite(image))] = 0
            maxind = np.unravel_index(np.argmax(image),image.shape)
            print(1,maxind)
            image[np.where(~np.isfinite(image))] = 0
            ny,nx = image.shape
            tmp[(150-maxind[0]):(150-maxind[0]+ny)] = tmp[(150-maxind[0]):(150-maxind[0]+ny)] + image[:,maxind[1]]
            print(2,np.argmax(tmp))
        except:
            pass
    tmp2 = np.sqrt(tmp/np.max(tmp))
    tmp2[140:155]=np.nan
    tmp2[0:130]=np.nan
    tmp2[180::]=np.nan
    plt.plot(np.sqrt(tmp/np.max(tmp)))
    plt.title(1/np.nanstd(tmp2))
    plt.xlim([120,180])
    plt.ylim([-0.1,1])
    plt.ylabel(r"$\sqrt{\Delta log(L)}$ ")
    plt.xlabel("vertical axis (pixels)")
    # plt.ylim([0,1100])
    plt.savefig(os.path.join(out_pngs,"HR8799"+planet+suffix+"_CCF_"+IFSfilter+".pdf"),bbox_inches='tight')
    plt.savefig(os.path.join(out_pngs,"HR8799"+planet+suffix+"_CCF_"+IFSfilter+".png"),bbox_inches='tight')
    plt.show()

# medHPF + CCF
if 0:
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
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"sherlock","medfilt_ccmap",
                                               os.path.basename(filename).replace(".fits","_output_medfilt_ccmapconvo.fits")))
            image = hdulist[0].data
            prihdr = hdulist[0].header


            plt.sca(ax)
            ny,nx = image.shape
            plt.imshow(image,interpolation="nearest")
        except:
            pass

    f.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(out_pngs,"HR8799c_medfilt_ccmapconvo.pdf"),bbox_inches='tight')
    plt.savefig(os.path.join(out_pngs,"HR8799c_medfilt_ccmapconvo.png"),bbox_inches='tight')


# polyfit + visual center
if 0:
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
            #polyfit_visucen/s101104_a016001_tlc_Kbb_020_outputpolyfit_visucen.fit
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"sherlock","polyfit_visucen",
                                               os.path.basename(filename).replace(".fits","_outputpolyfit_visucen.fits")))
            image = hdulist[0].data[2,:,:]
            prihdr = hdulist[0].header


            plt.sca(ax)
            ny,nx = image.shape
            plt.imshow(image,interpolation="nearest")

            xcen,ycen = float(fileelement.attrib["xvisucen"])+5,float(fileelement.attrib["yvisucen"])+5
            import matplotlib.patches as mpatches
            # myarrow = mpatches.Arrow(xcen,ycen,nx//2-xcen,ny//2-ycen,color="pink",linestyle="--",linewidth=0.5)
            # myarrow.set_clip_on(False)
            # # myarrow.set_clip_box(ax.bbox)
            # ax.add_artist(myarrow)
            if fileelement.attrib["stardir"] == "left":
                myarrow = mpatches.Arrow(xcen,ycen,float(fileelement.attrib["sep"])/ 0.0203-2,0,color="pink",linestyle="--",linewidth=1)
            elif fileelement.attrib["stardir"] == "down":
                myarrow = mpatches.Arrow(xcen,ycen,0,-float(fileelement.attrib["sep"])/ 0.0203-2,color="pink",linestyle="--",linewidth=1)
            myarrow.set_clip_on(False)
            # myarrow.set_clip_box(ax.bbox)
            ax.add_artist(myarrow)
        except:
            pass

    f.subplots_adjust(wspace=0,hspace=0)
    plt.savefig(os.path.join(out_pngs,"HR8799c_polyfit_visucen.pdf"),bbox_inches='tight')
    plt.savefig(os.path.join(out_pngs,"HR8799c_polyfit_visucen.png"),bbox_inches='tight')
    # exit()

# polyfit + default center
if 0:
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

            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"sherlock","medfilt_ccmap",
                                               os.path.basename(filename).replace(".fits","_output_defcen.fits")))
            image = hdulist[0].data[2,:,:]
            prihdr = hdulist[0].header


            plt.sca(ax)
            ny,nx = image.shape
            plt.imshow(image,interpolation="nearest")

            xcen,ycen = float(fileelement.attrib["xdefcen"])+5,float(fileelement.attrib["ydefcen"])+5
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
    plt.savefig(os.path.join(out_pngs,"HR8799c_2ndpolyfit_defcen.pdf"),bbox_inches='tight')
    plt.savefig(os.path.join(out_pngs,"HR8799c_2ndpolyfit_defcen.png"),bbox_inches='tight')
    # exit()


# polyfit + default center
if 0:
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
                                               os.path.basename(filename).replace(".fits","_output_defcen.fits")))
            image = hdulist[0].data[2,:,:]
            prihdr = hdulist[0].header


            plt.sca(ax)
            ny,nx = image.shape
            plt.imshow(image,interpolation="nearest")

            xcen,ycen = float(fileelement.attrib["xdefcen"])+5,float(fileelement.attrib["ydefcen"])+5
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
    plt.savefig(os.path.join(out_pngs,"HR8799c_polyfit_defcen.pdf"),bbox_inches='tight')
    plt.savefig(os.path.join(out_pngs,"HR8799c_polyfit_defcen.png"),bbox_inches='tight')

# Raw frames + ADI center
if 0:
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

            hdulist = pyfits.open(filename)
            cube = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
            image = np.nansum(cube,axis=0)
            prihdr = hdulist[0].header

            plt.sca(ax)
            ny,nx = image.shape
            plt.imshow(image,interpolation="nearest")

            xcen,ycen = (float(fileelement.attrib["xADIcen"])+5),(float(fileelement.attrib["yADIcen"])+5)
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
    plt.savefig(os.path.join(out_pngs,"HR8799c_raw_ADIcenter.pdf"),bbox_inches='tight')
    plt.savefig(os.path.join(out_pngs,"HR8799c_raw_ADIcenter.png"),bbox_inches='tight')

#ADI center + polyfit
if 0:
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

            plt.sca(ax)
            ny,nx = image.shape
            plt.imshow(image,interpolation="nearest")

            xcen,ycen = (float(fileelement.attrib["xADIcen"])+5),(float(fileelement.attrib["yADIcen"])+5)
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
    plt.savefig(os.path.join(out_pngs,"HR8799c_polyfit_ADIcenter.pdf"),bbox_inches='tight')
    plt.savefig(os.path.join(out_pngs,"HR8799c_polyfit_ADIcenter.png"),bbox_inches='tight')