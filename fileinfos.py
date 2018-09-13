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

OSIRISDATA = "/home/sda/jruffio/osiris_data/"
if 1:
    foldername = "HR_8799_c"
    sep = 0.950
    telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
    template_spec = os.path.join(OSIRISDATA,"hr8799c_osiris_template.save")
year = "*"
reductionname = "reduced_quinn"
filenamefilter = "s*_a*001_tlc_Kbb_020.fits"

# estimate visual locationof hr8799c
if 0:

    visual_planet_coords_hor = [[11,32],[12,27],[12,33],[12,39],[10,33],[9,28],[8,38],[10,32.5],[9,32],[10,33],[10,35],[10,33]] #in order: image 10 to 21
    visual_planet_coords_ver = [[7,34],[5,35],[8,35],[7.5,33],[9.5,34.5]]
    visual_planet_coords=visual_planet_coords_hor+visual_planet_coords_ver
    visual_planet_coords = visual_planet_coords +\
                           [[19//2,64//2]]+[[19//2,64//2]]+[[9,31]]+\
                           [[19//2,64//2],[19//2,64//2],[19//2,64//2],[19//2,64//2],[19//2,64//2]]+\
                           [[9,28],[9,22]]+\
                           [[9,32],[9,28],[9,28],[9,24]]+\
                           [[9,29],[8,40],[8,21]]+\
                           [[19//2,64//2]]+[[9,31]]+[[9,31]]
    sequences = [12,5,1,1,1,5,2,4,3,1,1,1]
    cumseq = np.roll(np.cumsum(sequences),1)
    cumseq[0] = 0
    filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
    filelist.sort()
    deltas = []
    hdrs_deltas_coords = []
    for nfiles,ind0 in zip(sequences,cumseq):
        print(nfiles,ind0)
        print(filelist[ind0:(ind0+nfiles)])
        if nfiles > 1:
            prihdr_list=[]
            for filename in filelist[ind0:(ind0+nfiles)]:
                hdulist = pyfits.open(filename)
                prihdr = hdulist[0].header
                prihdr_list.append(prihdr)
            delta_x,delta_y = determine_mosaic_offsets_from_header(prihdr_list)
            deltas.extend([[dx,dy] for dx,dy in zip(delta_x,delta_y)])
            hdrs_deltas_coords.extend([[dx+visual_planet_coords[ind0][0],dy+visual_planet_coords[ind0][1]] for dx,dy in zip(delta_x,delta_y)])
        else:
            deltas.append([np.nan,np.nan])
            hdrs_deltas_coords.append([0,0])
    # plt.figure(10)
    # plt.plot(np.array(deltas)[:,0],color="blue")
    # plt.plot(np.array(deltas)[:,1],color="red")
    # plt.show()

    planet_c = root.find("c")
    print(len(filelist))
    f,ax_list = plt.subplots(4,len(filelist)//4+1,figsize=(18*0.75,0.59*18*0.75))
    ax_list = [myax for rowax in ax_list for myax in rowax ]
    for k,(ax,filename) in enumerate(zip(ax_list,filelist)):
        print(filename)
        filebasename = os.path.basename(filename)
        fileelement = planet_c.find(filebasename)

        try:
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),"sherlock","polyfit_ADIcenter",
                                               os.path.basename(filename).replace(".fits","_output_defcen.fits")))
            image = hdulist[0].data[2,:,:]
            prihdr = hdulist[0].header

            plt.sca(ax)
            ny,nx = image.shape
            plt.imshow(image[5:69,5:24],interpolation="nearest")
            plt.ylabel(filebasename,fontsize=8)

            if k <= len(visual_planet_coords):
                xcen,ycen = visual_planet_coords[k][0],visual_planet_coords[k][1]
                import matplotlib.patches as mpatches
                if fileelement.attrib["stardir"] == "left":
                    myarrow = mpatches.Arrow(xcen,ycen,float(fileelement.attrib["sep"])/ 0.0203,0,color="pink",linestyle="--",linewidth=1)
                elif fileelement.attrib["stardir"] == "down":
                    myarrow = mpatches.Arrow(xcen,ycen,0,-float(fileelement.attrib["sep"])/ 0.0203,color="pink",linestyle="--",linewidth=1)
                myarrow.set_clip_on(False)
                ax.add_artist(myarrow)

                xcen,ycen = hdrs_deltas_coords[k][0],hdrs_deltas_coords[k][1]
                dx,dy = deltas[k][0],deltas[k][1]
                print(xcen,ycen)
                import matplotlib.patches as mpatches
                if fileelement.attrib["stardir"] == "left":
                    myarrow = mpatches.Arrow(xcen,ycen,-dx,-dy,color="red",linestyle="--",linewidth=1)
                elif fileelement.attrib["stardir"] == "down":
                    myarrow = mpatches.Arrow(xcen,ycen,-dx,-dy,color="red",linestyle="--",linewidth=1)
                myarrow.set_clip_on(False)
                ax.add_artist(myarrow)
        except:
            pass

    f.subplots_adjust(wspace=0,hspace=0)
    plt.show()

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


if 0:
    #hr 8799 c, 20100715
    # science files: s100715_a0[10-21,25-29]001_tlc_Kbb_020.fits
    # telluric hd 210501: s100715_a0[4]001_tlc_Kbb020.fits s100715_a0[5]00[1,2]_tlc_Kbb020.fits
    # sky: s100715_a0[22]00[1,2]_tlc_Kbb020.fits
    visual_planet_coords_hor = [[11,32],[12,27],[12,33],[12,39],[10,33],[9,28],[8,38],[10,32.5],[9,32],[10,33],[10,35],[10,33]] #in order: image 10 to 21
    visual_planet_coords_ver = [[7,34],[5,35],[8,35],[7.5,33],[9.5,34.5]]
    visual_planet_coords=visual_planet_coords_hor+visual_planet_coords_ver
    visual_planet_coords = visual_planet_coords +\
                           [[19//2,64//2]]+[[19//2,64//2]]+[[9,31]]+\
                           [[19//2,64//2],[19//2,64//2],[19//2,64//2],[19//2,64//2],[19//2,64//2]]+\
                           [[9,28],[9,22]]+\
                           [[9,32],[9,28],[9,28],[9,24]]+\
                           [[9,29],[8,40],[8,21]]+\
                           [[19//2,64//2]]+[[9,31]]+[[9,31]]

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
