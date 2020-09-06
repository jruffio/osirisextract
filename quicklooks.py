__author__ = 'jruffio'

import glob
import os
import csv
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
from copy import copy


out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"

# planet = "HR_8799_b"
planet = "HR_8799_c"
# planet = "HR_8799_d"
# planet = "kap_And"
# planet = "51_Eri_b"

# IFSfilter = "Kbb"
# IFSfilter = "Hbb"
# IFSfilter = "all"
suffix = "KbbHbb"
# suffix = "all"
fontsize = 12
resnumbasis = 10
# fileinfos_filename = "/data/osiris_data/"+planet+"/fileinfos_Kbb_jb.csv"
if resnumbasis ==0:
    fileinfos_filename = "/data/osiris_data/"+planet+"/fileinfos_Kbb_jb.csv"
else:
    fileinfos_filename = "/data/osiris_data/"+planet+"/fileinfos_Kbb_jb_kl{0}.csv".format(resnumbasis)


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
    ifs_filter_id = colnames.index("IFS filter")

filelist = [item[filename_id] for item in list_data]
filelist_sorted = copy(filelist)
filelist_sorted.sort()
print(len(filelist_sorted)) #37
# exit()
new_list_data = []
for filename in filelist_sorted:
    if 0 or "Kbb" in list_data[filelist.index(filename)][ifs_filter_id] or \
       "Hbb" in list_data[filelist.index(filename)][ifs_filter_id]:
        if 1:#"20190324_HPF_only" in list_data[filelist.index(filename)][cen_filename_id]:
            new_list_data.append(list_data[filelist.index(filename)])
list_data=new_list_data
# print(new_list_data)
# exit()
# valid_d_files = ["s150720_a091001_Kbb_020.fits",
#                  "s150720_a092001_Kbb_020.fits",
#                  "s150720_a093001_Kbb_020.fits",
#                  "s150720_a095001_Kbb_020.fits",
#                  "s150720_a096001_Kbb_020.fits",
#                  "s150720_a097001_Kbb_020.fits",
#                  "s150720_a098001_Kbb_020.fits",
#                  "s150720_a099001_Kbb_020.fits",
#                  "s150720_a100001_Kbb_020.fits",
#                  "s150722_a052001_Kbb_020.fits",
#                  "s150722_a054001_Kbb_020.fits",
#                  "s150723_a036001_Kbb_020.fits",
#                  "s150723_a037001_Kbb_020.fits",
#                  "s150723_a038001_Kbb_020.fits",
#                  ]
# if "d" in planet:
#     new_list_data = []
#     for k,item in enumerate(list_data):
#         for dfile in valid_d_files:
#             if dfile in item[filename_id]:
#                 new_list_data.append(item)
#     list_data = new_list_data

N_lines =  len(list_data)

# plot 2D images
if 1:
    # if IFSfilter=="Kbb": #Kbb 1965.0 0.25
    #     CRVAL1 = 1965.
    #     CDELT1 = 0.25
    #     nl=1665
    #     R=4000
    # elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
    #     CRVAL1 = 1473.
    #     CDELT1 = 0.2
    #     nl=1651
    #     R=5000
    # dwv = CDELT1/1000.
    # init_wv = CRVAL1/1000. # wv for first slice in mum

    seqref = -1
    Ninarow = 20
    f,ax_list = plt.subplots(int(np.ceil(N_lines/Ninarow)),Ninarow,sharey="row",sharex="col",figsize=(12,12./Ninarow*64./19.*(N_lines//Ninarow+1)))#figsize=(12,8)
    try:
        ax_list = [myax for rowax in ax_list for myax in rowax ]
    except:
        pass

    # test = []
    # test2 = []
    cube_hd_list = []
    for k,item in enumerate(list_data):
        # if item[rvcen_id] == "nan":
        #     continue
        ax = ax_list[k]
        # if item[ifs_filter_id] != IFSfilter:
        #     continue

        # reducfilename = os.path.join(os.path.dirname(item[filename_id]),"sherlock","20190309_HPF_only",os.path.basename(item[filename_id]).replace(".fits","_outputHPF_cutoff40_sherlock_v1_search.fits"))
        # reducfilename = item[cen_filename_id].replace("20190324_HPF_only","20190401_HPF_only")
        reducfilename = item[cen_filename_id]
        # print(reducfilename)
        # exit()
        plt.sca(ax)
        # plt.ylabel(os.path.basename(item[filename_id]).split("bb_")[0],fontsize=10)
        # if "20190324_HPF_only" not in reducfilename:
        #     continue
        print(k,item)
        # if "20171104" not in reducfilename:
        #     continue
        # print(reducfilename)
        # reducfilename = item[cen_filename_id].replace("20190117_HPFonly","20190125_HPFonly").replace("sherlock_v0","sherlock_v1_search")
        # reducfilename = item[cen_filename_id].replace("20190117_HPFonly","20190125_HPFonly_cov").replace("sherlock_v0","sherlock_v1_search_empcov")
        # print(reducfilename)
        # exit()

        # if "HR_8799" in planet:
        #     if resnumbasis == 0 :
        #         reducfilename = reducfilename.replace("20190416_no_persis_corr","20190920_resH0model_detec").replace("_outputHPF_cutoff40_sherlock_v1_search","_outputHPF_cutoff40_sherlock_v1_search_rescalc")
        #     else:
        #         reducfilename = reducfilename.replace("20190416_no_persis_corr","20190920_resH0model_detec").replace("_outputHPF_cutoff40_sherlock_v1_search","_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl{0}".format(resnumbasis))
        # exit()

        try:
            hdulist = pyfits.open(reducfilename.replace(".fits","_planetRV.fits"))
        except:
            print("Not there")
            continue
        planetRV = hdulist[0].data

        if np.size(planetRV) > 1:
            NplanetRV_hd = np.where((planetRV[1::]-planetRV[0:(np.size(planetRV)-1)]) < 0)[0][0]+1
            planetRV_hd = hdulist[0].data[0:NplanetRV_hd]
            planetRV = hdulist[0].data[NplanetRV_hd::]
            # rv_per_pix = 3e5*dwv/(init_wv+dwv*nl//2) # 38.167938931297705

            hdulist = pyfits.open(reducfilename)
            print(hdulist[0].data.shape)
            # cube_hd = hdulist[0].data[-1,0,0,0:NplanetRV_hd,:,:]
            # cube = hdulist[0].data[-1,0,0,NplanetRV_hd::,:,:]

            cube_hd = hdulist[0].data[-1,0,2,0:NplanetRV_hd,:,:]-hdulist[0].data[-1,0,1,0:NplanetRV_hd,:,:]
            cube = hdulist[0].data[-1,0,2,NplanetRV_hd::,:,:]-hdulist[0].data[-1,0,1,NplanetRV_hd::,:,:]
            cube_cp = copy(cube)
            cube_cp[np.where(np.abs(planetRV)<500)[0],:,:] = np.nan
            offsets = np.nanmedian(cube_cp,axis=0)[None,:,:]
            cube_hd = cube_hd - offsets
            cube = cube - offsets
            cube_hd_list.append(cube_hd)

            bary_rv = -float(item[bary_rv_id])/1000. # RV in km/s
            if "HR_8799" in reducfilename:
                rv_star = -12.6#-12.6+-1.4km/s HR 8799 Rob and Simbad
            if "kap_And" in reducfilename:
                rv_star = -12.7#-12.6+-1.4km/s HR 8799 Rob and Simbad
            if "51_Eri_b" in reducfilename:
                rv_star = +12.6#-12.6+-1.4km/s HR 8799 Rob and Simbad
            print(rv_star)
            # exit()

            try:
                kcen = int(item[kcen_id])
                lcen = int(item[lcen_id])
                rvcen = float(item[rvcen_id])
            except:
                rvcen = bary_rv + rv_star
            zcen = np.argmin(np.abs(planetRV_hd-rvcen))
            image = copy(cube_hd[zcen,:,:])
            delta_AIC = cube_hd[zcen,kcen,lcen]
            # if "171103" in reducfilename:
            #     test.append(image)
            # if "171103" not in reducfilename:
            #     test2.append(image)
        else:
            try:
                kcen = int(item[kcen_id])
                lcen = int(item[lcen_id])
            except:
                pass
            hdulist = pyfits.open(reducfilename)
            image = hdulist[0].data[-1,0,0,0,:,:]
            delta_AIC = image[kcen,lcen]

        ny,nx = image.shape
        if 1:#delta_AIC>50:
            plt.imshow(image,interpolation="nearest",origin="lower")
            # plt.clim([0,cube_hd[zcen,kcen,lcen]/2.0])
            # plt.clim([0,np.nanstd(cube_hd)*10])
            # plt.clim([0,30])
            try:
                plt.clim([0,np.max([np.nanstd(cube_hd)*10,30])])
            except:
                plt.clim([0,np.max([np.nanstd(image)*10,30])])
            plt.clim([0,40])
            plt.xticks([0,10])


        if int(item[status_id]) == 1:
            color = "white"
            circlelinestyle = "-"
        elif int(item[status_id]) == 2:
            color = "grey"
            circlelinestyle = "--"
        else:
            color = "grey"
            circlelinestyle = "--"
        plt.gca().text(3,10,os.path.basename(item[filename_id]).split("bb_")[0],ha="left",va="bottom",rotation=90,size=fontsize/3*2,color=color)
        try:
            circle = plt.Circle((lcen,kcen),5,color=color, fill=False,linestyle=circlelinestyle)
            ax.add_artist(circle)
            # print(hdulist[0].data[0,0,11,zcen,kcen,lcen])
        except:
            pass
        # plt.title(os.path.basename(item[filename_id]).split(IFSfilter)[0])


        xoffset = float(item[xoffset_id])
        yoffset = float(item[yoffset_id])
        sequence = float(item[sequence_id])
        print(seqref,sequence)
        if seqref == sequence:
            arrow = plt.arrow(lcenref,kcenref,xoffset-xoffset0,yoffset-yoffset0,color="#ff9900",linestyle =":",alpha=0.5)
            ax.add_artist(arrow)
            circle = plt.Circle((xoffset-xoffset0+lcenref,kcenref+yoffset-yoffset0),1,color="#ff9900", fill=False)
            ax.add_artist(circle)
        elif int(item[status_id]) >= 1:
            xoffset0 = xoffset
            yoffset0 = yoffset
            lcenref = lcen
            kcenref = kcen
            seqref = sequence
            circle = plt.Circle((lcenref,kcenref),3,color="#ff9900", fill=False)
            ax.add_artist(circle)
            circle = plt.Circle((lcenref,kcenref),1,color="#ff9900", fill=False)
            ax.add_artist(circle)
        else:
            pass
            # xoffset0 = xoffset
            # yoffset0 = yoffset
            # lcenref = 9
            # kcenref = 30
            # seqref = sequence
            # circle = plt.Circle((lcenref,kcenref),3,color="#ff9900", fill=False)
            # ax.add_artist(circle)
            # circle = plt.Circle((lcenref,kcenref),1,color="#ff9900", fill=False)
            # ax.add_artist(circle)
            # # xoffset0 = 0
            # # yoffset0 = 0

    for ax in ax_list:
        plt.sca(ax)
        plt.tick_params(axis="x",which="both",bottom=False,top=False,labelbottom=False)
        plt.tick_params(axis="y",which="both",left=False,right=False,labelleft=False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
    #
    # plt.figure(2)
    # plt.imshow(np.nanmean(test2,axis=0),interpolation="nearest",origin="lower")
    # plt.show()

    f.subplots_adjust(wspace=0,hspace=0)
    # plt.show()
    if not os.path.exists(os.path.join(out_pngs,planet)):
        os.makedirs(os.path.join(out_pngs,planet))
    print("Saving "+os.path.join(out_pngs,planet,planet+"_"+suffix+"_images_kl{0}.pdf".format(resnumbasis)))
    plt.savefig(os.path.join(out_pngs,planet,planet+"_"+suffix+"_images_kl{0}.png".format(resnumbasis)),bbox_inches='tight')
    plt.savefig(os.path.join(out_pngs,planet,planet+"_"+suffix+"_images_kl{0}.pdf".format(resnumbasis)),bbox_inches='tight')

    # print("Saving "+os.path.join(out_pngs,"HR8799"+planet+"_"+suffix+"_images_tentativedetec.pdf"))
    # plt.savefig(os.path.join(out_pngs,"HR8799"+planet+"_"+suffix+"_images_tentativedetec.pdf"),bbox_inches='tight')
    # plt.savefig(os.path.join(out_pngs,"HR8799"+planet+"_"+suffix+"_images_tentativedetec.png"),bbox_inches='tight')


    # plt.figure(10)
    # rvcen = bary_rv + rv_star
    # cube_hd = np.nansum(np.array(cube_hd_list),axis=0)
    # zcen = np.argmin(np.abs(planetRV_hd-rvcen))
    # image = copy(cube_hd[zcen,:,:])
    # plt.imshow(image,interpolation="nearest",origin="lower")
    # # plt.clim([0,0])
    plt.show()
    exit()

# plot CCF
if 0:
    # molecule_list = ["_H2O"]#["","_CH4","_CO","_CO2","_H2O"]
    # molecule_str_list = ["H2O"]#["Atmospheric model","CH4","CO","CO2","H2O"]
    # molecule_list = ["","_CH4","_CO","_CO2","_H2O"]
    # molecule_str_list = ["Atmospheric model","CH4","CO","CO2","H2O"]
    # molecule_list = [""]
    # molecule_str_list = ["Atmospheric model"]
    # molecule_list = ["_CO"]
    # molecule_str_list = ["CO"]
    # molecule_list = ["_CO2"]
    # molecule_str_list = ["CO2"]
    # molecule_list = ["_CH4"]
    # molecule_str_list = ["CH4"]
    molecule_list = ["_CH4","_CO","_CO2"]
    molecule_str_list = ["CH4","CO","CO2"]
    for molecule,molecule_str in zip(molecule_list,molecule_str_list):
    # molecule = ""
    # molecule_str="Atmospheric model"
    # molecule = "_CH4"
    # molecule_str = "CH4"
    # # molecule = "_CO"
    # # molecule_str = "CO"
    # # molecule = "_CO2"
    # # molecule_str = "CO2"
    # # molecule = "_H2O"
    # # molecule_str = "H20"
        for IFSfilter in ["Kbb"]:#["Kbb","Hbb"]:
            if IFSfilter=="Kbb": #Kbb 1965.0 0.25
                CRVAL1 = 1965.
                CDELT1 = 0.25
                nl=1665
                R=4000
            elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
                CRVAL1 = 1473.
                CDELT1 = 0.2
                nl=1651
                R=5000
            dwv = CDELT1/1000.
            init_wv = CRVAL1/1000. # wv for first slice in mum

            summed_wideRV = np.zeros((200*3,64*3,19*3))
            Nvalid_wideRV = np.zeros((200*3,64*3,19*3))
            summed_hdRV = np.zeros((400*3,64*3,19*3))
            Nvalid_hdRV = np.zeros((400*3,64*3,19*3))

            filtered_list_data = []
            for k,item in enumerate(list_data):
                if item[rvcen_id] == "nan":
                    continue
                if int(item[status_id]) != 1 or item[ifs_filter_id] != IFSfilter:
                    continue
                filtered_list_data.append(item)
            print(len(filtered_list_data))
            # exit()

            plt.figure(2,figsize=(10*2,7*2))
            for k,item in enumerate(filtered_list_data):
                reducfilename = item[cen_filename_id].replace("search","search"+molecule)
                # reducfilename = item[cen_filename_id].replace("20190117_HPFonly","20190125_HPFonly").replace("sherlock_v0","sherlock_v1_search")
                # reducfilename = item[cen_filename_id].replace("20190117_HPFonly","20190125_HPFonly_cov").replace("sherlock_v0","sherlock_v1_search_empcov")
                # print(len(glob.glob(reducfilename.replace(".fits","_planetRV.fits"))) == 0,reducfilename.replace(".fits","_planetRV.fits"))
                if len(glob.glob(reducfilename.replace(".fits","_planetRV.fits"))) == 0:
                    continue
                hdulist = pyfits.open(reducfilename.replace(".fits","_planetRV.fits"))
                planetRV = hdulist[0].data
                NplanetRV_hd = np.where((planetRV[1::]-planetRV[0:(np.size(planetRV)-1)]) < 0)[0][0]+1
                planetRV_hd = hdulist[0].data[0:NplanetRV_hd]
                planetRV = hdulist[0].data[NplanetRV_hd::]
                NplanetRV = np.size(planetRV)
                rv_per_pix = 3e5*dwv/(init_wv+dwv*nl//2) # 38.167938931297705

                hdulist = pyfits.open(reducfilename)
                cube_hd = hdulist[0].data[0,0,0,0:NplanetRV_hd,:,:]
                cube = hdulist[0].data[0,0,0,NplanetRV_hd::,:,:]
                _,ny,nx = cube.shape

                # print(cube_hd.shape)
                # plt.imshow(cube_hd[100,:,:])
                # plt.show()
                # exit()

                bary_rv = -float(item[bary_rv_id])/1000. # RV in km/s
                rv_star = -12.6#-12.6+-1.4km/s HR 8799 Rob and Simbad

                kcen = int(item[kcen_id])
                lcen = int(item[lcen_id])
                rvcen = float(item[rvcen_id])
                zcenhd = np.argmin(np.abs(planetRV_hd-rvcen))
                zcen = np.argmin(np.abs(planetRV-rvcen))

                SNR_data = hdulist[0].data[0,0,10,NplanetRV_hd::,:,:]
                SNR_data_cp = copy(SNR_data)
                SNR_data_cp[np.where(np.abs(SNR_data)>100)] = np.nan
                # if "20100712" in item[filename_id]:
                #     print("skipping",item[filename_id])
                #     continue
                #     y,x = 46,8
                #     SNR_data_cp[:,y-1:y+2,x-1:x+2] = np.nan
                #     y,x = 21,9
                #     SNR_data_cp[:,y-1:y+2,x-1:x+2] = np.nan

                SNR_data_cp[98:103,:,:] = np.nan
                # print(SNR_data_cp.shape)
                # exit()
                meanSNR1 = np.nanmean(SNR_data_cp,axis=0)[None,:,:]
                SNR_data_calib = (SNR_data-meanSNR1)
                meanSNR2 = np.nanmedian(SNR_data_calib,axis=(1,2))[:,None,None]
                SNR_data_calib = (SNR_data_calib-meanSNR2)

                stdSNR = np.nanstd(SNR_data_cp-meanSNR1-meanSNR2)
                SNR_data_calib = SNR_data_calib/stdSNR

                plt.figure(2)
                plt.subplot(7,10,k+1)
                plt.title(os.path.basename(item[filename_id]).split("bb_")[0],fontsize=5)
                plt.hist(np.ravel(SNR_data_calib)[np.where(np.isfinite(np.ravel(SNR_data_calib)))],bins=100,density=True)
                plt.plot(np.linspace(-7,7,200),1/np.sqrt(2*np.pi)*np.exp(-0.5*np.linspace(-7,7,200)**2))
                plt.yscale("log")

                # plt.figure(2)
                # plt.subplot(7,10,k+1)
                # # plt.plot(np.nanmean((SNR_data_cp-meanSNR)/stdSNR,axis=0))
                # plt.plot(np.nanmean((SNR_data_cp-meanSNR)/stdSNR,axis=(1,2)))

                canvas = np.zeros(SNR_data_calib.shape)
                canvas[np.where(np.isfinite(SNR_data_calib))] = 1
                Nvalid_wideRV[(300-zcen):(300+NplanetRV-zcen),
                ((64*3)//2-kcen):((64*3)//2+ny-kcen),
                ((19*3)//2-lcen):((19*3)//2+nx-lcen)] += canvas
                SNR_data_calib[np.where(np.isnan(SNR_data_calib))] = 0

                SNR_data_calib_hd = hdulist[0].data[0,0,10,0:NplanetRV_hd,:,:]-meanSNR1
                meanSNR3 = np.nanmedian(SNR_data_calib_hd,axis=(1,2))[:,None,None]
                SNR_data_calib_hd = (SNR_data_calib_hd-meanSNR3)/stdSNR

                canvas = np.zeros(hdulist[0].data[0,0,10,0:NplanetRV_hd,:,:].shape)
                canvas[np.where(np.isfinite(hdulist[0].data[0,0,10,0:NplanetRV_hd,:,:]))] = 1
                Nvalid_hdRV[((400*3)//2-zcenhd):((400*3)//2+NplanetRV_hd-zcenhd),
                ((64*3)//2-kcen):((64*3)//2+ny-kcen),
                ((19*3)//2-lcen):((19*3)//2+nx-lcen)] += canvas
                SNR_data_calib_hd[np.where(np.isnan(SNR_data_calib_hd))] = 0

                # plt.figure(3)
                # print(item[filename_id])
                # plt.imshow(np.nansum(SNR_data_calib,axis=0),interpolation="nearest")
                # plt.colorbar()
                # plt.show()

                summed_wideRV[(300-zcen):(300+NplanetRV-zcen),
                ((64*3)//2-kcen):((64*3)//2+ny-kcen),
                ((19*3)//2-lcen):((19*3)//2+nx-lcen)] += copy(SNR_data_calib)
                summed_hdRV[((400*3)//2-zcenhd):((400*3)//2+NplanetRV_hd-zcenhd),
                ((64*3)//2-kcen):((64*3)//2+ny-kcen),
                ((19*3)//2-lcen):((19*3)//2+nx-lcen)] += copy(SNR_data_calib_hd)

            # plt.show()
            print("Saving "+os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_SNRhistmozaic"+molecule+".png"))
            plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_SNRhistmozaic"+molecule+".png"),bbox_inches='tight')
            # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_SNRhistmozaic"+molecule+".pdf"),bbox_inches='tight')
            plt.close(2)
            # exit()

            # noise1 = copy(summed_wideRV[200:400,((64*3)//2+5):((64*3)//2+15),((19*3)//2-3):((19*3)//2+3)])
            summed_wideRV = summed_wideRV/Nvalid_wideRV
            summed_hdRV = summed_hdRV/Nvalid_hdRV
            Nvalid_wideRV = np.sum(Nvalid_wideRV,axis=0)
            where_valid = np.where(Nvalid_wideRV>0.8*len(filtered_list_data))
            where_notvalid = np.where(Nvalid_wideRV<=0.8*len(filtered_list_data))
            noise1 = copy(summed_wideRV[200:400,:,:])
            noise1[:,((64*3)//2-5):((64*3)//2+5),((19*3)//2-5):((19*3)//2+5)] = np.nan
            noise1[:,where_notvalid[0],where_notvalid[1]] = np.nan
            sigma = np.nanstd(noise1)
            summed_wideRV = summed_wideRV/sigma
            summed_hdRV = summed_hdRV/sigma
            noise1 = noise1/sigma

            plt.figure(1)
            # noise1 = summed_wideRV[200:400,((64*3)//2+5):((64*3)//2+15),((19*3)//2-3):((19*3)//2+3)]
            # noise2 = summed_wideRV[200:400,((64*3)//2-15):((64*3)//2-5),((19*3)//2-3):((19*3)//2+3)]
            # plt.subplot(1,2,1)
            # plt.imshow(np.nansum(summed_wideRV,axis=0),interpolation="nearest")
            # plt.subplot(1,2,2)
            # plt.imshow(noise1[100,:,:],interpolation="nearest")
            # plt.show() # 28,96
            # for k in range(3):
            #     for l in range(3):
            #         plt.plot(planetRV,summed_wideRV[200:400,(64*3)//2-1+k,(19*3)//2-1+l],linestyle="-",linewidth=3,color="red")
            #         plt.plot(planetRV_hd,summed_hdRV[400:800,(64*3)//2-1+k,(19*3)//2-1+l],linestyle="--",linewidth=3,color="pink")
            plt.plot(planetRV,summed_wideRV[200:400,(64*3)//2,(19*3)//2],linestyle="-",linewidth=2,color="red")
            plt.plot(planetRV_hd,summed_hdRV[400:800,(64*3)//2,(19*3)//2],linestyle="--",linewidth=2,color="pink") #"black","#ff9900","#006699","grey"
            for k,l in zip(where_valid[0],where_valid[1]):
                plt.plot(planetRV,noise1[:,k,l],alpha=0.1,linestyle="--",linewidth=0.2,color="#006699") #006699
                # plt.plot(planetRV,noise2[:,k,l],alpha=0.5,linestyle="--",linewidth=1,color="cyan")
            plt.ylabel("SNR",fontsize=15)
            plt.xlabel(r"$\Delta V$ (km/s)",fontsize=15)
            plt.gca().tick_params(axis='x', labelsize=15)
            plt.gca().tick_params(axis='y', labelsize=15)
            plt.gca().annotate(molecule_str,xy=(-4000,24.5),va="top",ha="left",fontsize=15,color="black")
            plt.ylim([-5,25])
            # plt.show()
            # exit()
            plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+".png"),bbox_inches='tight')
            print("Saving "+os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+".pdf"))
            plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+".pdf"),bbox_inches='tight')

            plt.gca().annotate(molecule_str,xy=(-1450,24.5),va="top",ha="left",fontsize=15,color="black")
            plt.xlim([-1500,1500])
            plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+"_zoomed.png"),bbox_inches='tight')
            print("Saving "+os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+"_zoomed.pdf"))
            plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+"_zoomed.pdf"),bbox_inches='tight')
            plt.close(1)
    exit()

# plot RVs
if 0:
    rv_star = -12.6#-12.6+-1.4km/s HR 8799 Rob and Simbad
    bary_star_list = np.array([-float(item[bary_rv_id])/1000+rv_star for item in list_data])
    mjdobs_list = np.array([float(item[mjdobs_id]) for item in list_data])
    mjdobs_list = np.arange(np.size(mjdobs_list))
    baryrv_list = np.array([-float(item[bary_rv_id])/1000 for item in list_data])
    rv_list = np.array([float(item[rvcen_id]) for item in list_data])
    status_list = np.array([int(item[status_id]) for item in list_data])
    # rv_list[37] = np.nan
    # rv_list[31] = np.nan
    rvsig_list = np.array([float(item[rvcensig_id]) for item in list_data])
    wvsolerr_list = np.array([float(item[wvsolerr_id]) for item in list_data])
    # valid_list = np.array([not ("201007" in item[filename_id]) for item in list_data])
    good_rv_list = copy(rv_list)
    good_rv_list[np.where(status_list!=1)] = np.nan
    bad_rv_list= copy(rv_list)
    bad_rv_list[np.where(status_list>0)] = np.nan
    plt.figure(1,figsize=(9,0.75*9))
    plt.subplot(2,1,1)
    plt.errorbar(mjdobs_list,good_rv_list,yerr=rvsig_list,fmt="x",color="red",label="Measured raw RV ($1\sigma$)")
    plt.plot(mjdobs_list,bad_rv_list,linestyle="",marker="o",color="blue",label="Bad Data")
    plt.plot(mjdobs_list,baryrv_list,color="#006699",label="Barycentric RV")
    plt.plot(mjdobs_list,bary_star_list,color="#ff9900",label="Barycentric + HR8799 RV")
    plt.xlabel("Exposure Index",fontsize=15)
    plt.ylabel("RV (km/s)",fontsize=15)
    plt.legend(fontsize=10)
    plt.legend(fontsize=10,loc="upper left")
    plt.subplot(2,1,2)
    # plt.plot(rv_list-bary_star_list,"x",color="red",label="Estimated Planet RV")
    plt.plot(mjdobs_list,np.zeros(rv_list.shape)+np.nanmean(rv_list-bary_star_list),linestyle=":",color="pink",label="Mean Planet RV")
    plt.errorbar(mjdobs_list,good_rv_list-bary_star_list,yerr=np.sqrt(rvsig_list**2+wvsolerr_list**2),fmt="x",color="red",label="Estimated Planet RV ($1\sigma$)")
    plt.errorbar(mjdobs_list,good_rv_list-bary_star_list,yerr=rvsig_list,fmt="x",color="pink",label="Estimated Planet RV ($1\sigma$)")
    plt.plot(mjdobs_list,bad_rv_list-bary_star_list,linestyle="",marker="o",color="blue",label="Bad data")
    plt.xlabel("Exposure Index",fontsize=15)
    plt.ylabel("RV (km/s)",fontsize=15)
    print(np.nansum((good_rv_list-bary_star_list)/rvsig_list)/(np.sum(1./rvsig_list[np.where(np.isfinite(good_rv_list))])),
        np.sqrt(np.size(rvsig_list[np.where(np.isfinite(good_rv_list))]))/(np.sum(1./rvsig_list[np.where(np.isfinite(good_rv_list))])))
    print(np.nanmean(good_rv_list[10:50]-bary_star_list[10:50]))
    # plt.ylim([-20,20])
    plt.legend(fontsize=10,loc="upper left")
    plt.show()
    print("Saving "+os.path.join(out_pngs,"HR8799"+planet+"_"+IFSfilter+"_RV.pdf"))
    plt.savefig(os.path.join(out_pngs,"HR8799"+planet+"_"+IFSfilter+"_RV.pdf"),bbox_inches='tight')
    plt.savefig(os.path.join(out_pngs,"HR8799"+planet+"_"+IFSfilter+"_RV.png"),bbox_inches='tight')

plt.show()
exit()





OSIRISDATA = "/data/osiris_data/"
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
if 0:
    # suffix = "_outputHPF_cutoff80_new_sig_phoenix"
    # myfolder = "20181205_HPF_only"
    suffix = "_outputHPF_cutoff80_sherlock_v0"
    myfolder = "sherlock/20190117_HPFonly"
    # planet = "c"
    planet = "d"
    IFSfilter = "Kbb"
    # IFSfilter = "Hbb"
    fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos_jb.xml"
    out_pngs = os.path.join("/data/osiris_data/HR_8799_"+planet+"/")#"/home/sda/jruffio/pyOSIRIS/figures/"
    # tree = ET.parse(fileinfos_filename)
    # root = tree.getroot()
    reductionname = "reduced_jb"
    filenamefilter = "s*_a*001_"+IFSfilter+"_020.fits"
    filelist = glob.glob(os.path.join(OSIRISDATA,"HR_8799_"+planet,year,reductionname,filenamefilter))
    filelist.sort()
    # planet_c = root.find("c")
    print(len(filelist))

    if IFSfilter=="Kbb": #Kbb 1965.0 0.25
        CRVAL1 = 1965.
        CDELT1 = 0.25
        nl=1665
        R=4000
    elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1473.
        CDELT1 = 0.2
        nl=1651
        R=5000
    dwv = CDELT1/1000.

    f,ax_list = plt.subplots(4,len(filelist)//4+1,sharey="row",sharex="col",figsize=(18,0.59*18))
    ax_list = [myax for rowax in ax_list for myax in rowax ]
    # f,ax_list = plt.subplots(3,8,sharey="row",sharex="col",figsize=(2*8,15))
    # ax_list = [myax for rowax in ax_list for myax in rowax ]
    logpostcube_doppler_list = []
    dAICcube_doppler_list = []
    wvshiftmax_list = []
    realfilelist = []
    mjdobs_list = []
    row_list = []
    summed_dAIC = np.zeros((600,64*3,19*3))
    summed_dAIC2 = np.zeros((400*3,64*3,19*3))
    for ax,filename in zip(ax_list,filelist):
        print(filename)
        hdulist = pyfits.open(filename)
        prihdr0 = hdulist[0].header
        mjdobs_list.append(prihdr0["MJD-OBS"])
        # filebasename = os.path.basename(filename)
        # fileelement = planet_c.find(filebasename)
        # print(fileelement.attrib["xADIcen"],fileelement.attrib["yADIcen"])

        # try:
        if 1:
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),myfolder,
                                               os.path.basename(filename).replace(".fits",suffix+"_wvshifts.fits")))
            wvshifts = hdulist[0].data
            wvshifts_doppler = wvshifts[0:400]
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),myfolder,
                                               os.path.basename(filename).replace(".fits",suffix+".fits")))
            cube = hdulist[0].data[2,:,:,:]
            cube[np.where(cube>2000)] = np.nan
            realfilelist.append(filename)

            logpostcube_doppler_list.append(hdulist[0].data[11,0:400,:,:])
            dAICcube_doppler_list.append(hdulist[0].data[2,0:400,:,:])
            max_vec = np.nanmax(cube[0:400,:,:],axis=(1,2))
            wvid_max = np.nanargmax(max_vec)
            wvshiftmax_list.append(wvshifts[wvid_max]/dwv) #3e5*dwv/(init_wv+dwv*nz//2)
            print(wvid_max,wvshifts[wvid_max],np.nanmax(max_vec))
            image = cube[wvid_max,:,:]

            plt.sca(ax)
            ny,nx = image.shape
            plt.imshow(image,interpolation="nearest")
            plt.clim([np.nanmedian(image),np.nanmax(image[np.where(np.isfinite(image))])/2.0])
            # plt.clim([np.nanmedian(image),np.nanmax(image[np.where(np.isfinite(image))])])

            image[np.where(~np.isfinite(image))] = 0
            maxind = np.unravel_index(np.argmax(image),image.shape)
            row_list.append(maxind[0])
            circle = plt.Circle(maxind[::-1],5,color="red", fill=False)
            ax.add_artist(circle)

            tmpcube = copy(hdulist[0].data[0,400::,:,:]/hdulist[0].data[1,400::,:,:])
            tmpcube2 = copy(hdulist[0].data[0,400::,:,:]/hdulist[0].data[1,400::,:,:])
            nz,ny,nx = tmpcube.shape
            max_vec = np.nanmax(tmpcube,axis=(1,2))
            wvid_max = np.nanargmax(max_vec)
            tmpcube[np.max([0,(wvid_max-5)]):np.min([(wvid_max+5),nz]),
            np.max([0,(maxind[0]-5)]):np.min([(maxind[0]+5),ny]),
            np.max([0,(maxind[1]-5)]):np.min([(maxind[1]+5),nx])] = np.nan
            tmpcube2 -= np.nanmedian(tmpcube)
            tmpcube2[np.where(np.isnan(tmpcube2))] = 0
            summed_dAIC[(300-wvid_max):(300+nz-wvid_max),
            ((64*3)//2-maxind[0]):((64*3)//2+ny-maxind[0]),
            ((19*3)//2-maxind[1]):((19*3)//2+nx-maxind[1])] += tmpcube2

            tmpcube22 = copy(hdulist[0].data[0,0:400,:,:]/hdulist[0].data[1,0:400,:,:])
            nz,ny,nx = tmpcube22.shape
            max_vec = np.nanmax(tmpcube22,axis=(1,2))
            wvid_max = np.nanargmax(max_vec)
            tmpcube22 -= np.nanmedian(tmpcube)
            tmpcube22[np.where(np.isnan(tmpcube22))] = 0
            summed_dAIC2[((400*3)//2-wvid_max):((400*3)//2+nz-wvid_max),
            ((64*3)//2-maxind[0]):((64*3)//2+ny-maxind[0]),
            ((19*3)//2-maxind[1]):((19*3)//2+nx-maxind[1])] += tmpcube22


            # exit()
        # except:
        #     pass

    # # print(mjdobs_list)
    # # # from astropy.coordinates import EarthLocation
    # # # EarthLocation.get_site_names() #"Keck Observatory"
    # # #HR 8799 = HIP 114189
    # from barycorrpy import get_BC_vel
    # result = get_BC_vel(np.array(mjdobs_list)+2400000.5,hip_id=114189,obsname="Keck Observatory",ephemeris="de430")
    # print(result)
    # print([res,filename] for filename,res in zip(filelist,result))
    # exit()
    # BC_vel = np.array([ 24197.25199165,  24174.90007515,  24156.78279151,  24137.22541644,
    #     24116.44636105,  24096.21104372,  24075.60004385,  24054.39517929,
    #     24032.65609066,  24011.37223287,  23989.63120374,  23967.16914106,
    #     23884.28592087,  23862.55438303,  23840.84193603,  23819.57695393,
    #     23797.89401162, -19343.85636064, -19390.33422923, -19560.77119117,
    #    -19766.54845987, -19785.8996881 , -19804.53677882, -19822.95537198,
    #     22437.70912121,  22416.49498461,  22362.831724  ,  22341.75539097,
    #     22320.13500181,  22297.93592043,  22166.54731677,
    #     22143.54408699,  22121.66079362])/1000. # HR 8799 c
    BC_vel = np.array([23164.86503798, 23144.75954252, 23124.03319351, 23101.86042246,
       23079.39997125, 23057.88612099, 23035.71551717, 23010.99002571,
       22988.83032754, 22965.75444462, 22874.27060698, 22850.59140727,
       22827.96543939, 22806.37688705, 22785.10306701, 22689.68272808,
       22670.75180415, 22651.07085841, 22631.24246311, 22611.04120447,
       22590.16430639, 22569.03816678, 22547.92785873, 22526.5195444 ,
       22503.33670587, 22481.31745378, 22363.84548024, 22340.57998809,
       22317.61434054, 22295.63753363, 22413.05761464, 22389.59443951,
       22367.82844697, 22345.32309399, 22321.66889475, 22299.38325312,
       22276.91402527, 22254.24357567, 22231.40699259, 22145.32093106,
       22122.25329363, 22099.41355866, 22076.34309571, 22053.87560401,
       22032.83206775,  9748.01130965,  9721.65748163,  9695.90248584,
        9670.7554549 ,  9645.99775605,  9621.49685952,  9597.47115864,
        9573.98100073,  9551.02039266,  9528.74291465,  9507.03532965,
        9485.4159476 ,  9465.21634848,  9445.4207218 ,  9426.76734994,
        9408.92172425,  9391.9611914 ,  9375.97321369])/1000.  # HR 8799 d

    # f.subplots_adjust(wspace=0,hspace=0)
    # print("Saving "+os.path.join(out_pngs,"HR8799"+planet+suffix+"_"+IFSfilter+".pdf"))
    # plt.savefig(os.path.join(out_pngs,"HR8799"+planet+suffix+"_"+IFSfilter+".pdf"),bbox_inches='tight')
    # plt.savefig(os.path.join(out_pngs,"HR8799"+planet+suffix+"_"+IFSfilter+".png"),bbox_inches='tight')

    plt.figure(3)
    # plt.subplot(1,3,1)
    # plt.imshow(summed_dAIC[299,:,:],interpolation="nearest")
    # plt.colorbar()
    # plt.subplot(1,3,2)
    # plt.imshow(summed_dAIC[300,:,:],interpolation="nearest")
    # plt.colorbar()
    # plt.subplot(1,3,3)
    # plt.imshow(summed_dAIC[301,:,:],interpolation="nearest")
    # plt.colorbar()

    # plt.subplot(1,3,1)
    # plt.plot(summed_dAIC[300,(64*3)//2,:])
    # plt.subplot(1,3,2)
    # plt.plot(summed_dAIC[300,:,(19*3)//2])
    # plt.subplot(1,3,3)

    noise1 = summed_dAIC[200:400,((64*3)//2+5):((64*3)//2+15),((19*3)//2-3):((19*3)//2+3)]
    offset = np.nanmean(noise1)
    summed_dAIC = summed_dAIC-offset
    summed_dAIC2 = summed_dAIC2-offset
    # summed_dAIC = np.sign(summed_dAIC)*np.sqrt(np.abs(summed_dAIC))
    noise1 = summed_dAIC[200:400,((64*3)//2+5):((64*3)//2+15),((19*3)//2-3):((19*3)//2+3)]
    sigma = np.nanstd(noise1)
    summed_dAIC = summed_dAIC/sigma
    summed_dAIC2 = summed_dAIC2/sigma
    noise1 = summed_dAIC[200:400,((64*3)//2+5):((64*3)//2+15),((19*3)//2-3):((19*3)//2+3)]
    noise2 = summed_dAIC[200:400,((64*3)//2-15):((64*3)//2-5),((19*3)//2-3):((19*3)//2+3)]
    plt.plot((np.arange(200)-100)*38.167938931297705,summed_dAIC[200:400,(64*3)//2,(19*3)//2],linestyle="-",linewidth=3,color="red")
    plt.plot((np.arange(-2,2,0.01))*38.167938931297705,summed_dAIC2[400:800,(64*3)//2,(19*3)//2],linestyle="--",linewidth=3,color="pink")
    for k in range(noise1.shape[1]):
        for l in range(noise1.shape[2]):
            plt.plot((np.arange(200)-100)*38.167938931297705,noise1[:,k,l],alpha=0.5,linestyle="--",linewidth=1,color="blue")
            plt.plot((np.arange(200)-100)*38.167938931297705,noise2[:,k,l],alpha=0.5,linestyle="--",linewidth=1,color="cyan")
    plt.ylabel("SNR")
    plt.xlabel(r"$\Delta V$ (km/s)")
    plt.xlim([-1000,1000])
    plt.show()


    plt.figure(2)
    plt.subplot(2,1,1)
    plt.plot(np.array(wvshiftmax_list)*38.167938931297705,label="measured")
    plt.plot(-BC_vel,label="barycentric")
    plt.xlabel("Exposure #")
    plt.ylabel(r"$\Delta V$ (km/s)")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(np.array(wvshiftmax_list)*38.167938931297705+BC_vel,label="bary corrected")
    plt.plot(np.zeros(len(wvshiftmax_list))-12.6,label="RV star")
    plt.plot(np.array(wvshiftmax_list)*38.167938931297705+BC_vel+12.6,label="residual RV")
    # plt.plot(-(np.array(row_list)-np.mean(np.array(row_list)))/np.max(row_list)*30,label="row")
    plt.xlabel("Exposure #")
    plt.ylabel(r"$\Delta V$ (km/s)")
    plt.legend()


    # for logpostcube,dAICcube in zip(logpostcube_doppler_list[0:3],dAICcube_doppler_list):
    #     max_vec = np.nanmax(dAICcube,axis=(1,2))
    #     wvid_max = np.nanargmax(max_vec)
    #     image = copy(cube[wvid_max,:,:])
    #     image[np.where(~np.isfinite(image))] = 0
    #     maxind = np.unravel_index(np.argmax(image),image.shape)
    #     logpoststamp_cube = logpostcube[:,(maxind[0]-3):(maxind[0]+4),(maxind[1]-3):(maxind[1]+4)]
    #     print(logpoststamp_cube.shape)
    #     posterior = np.exp(logpoststamp_cube-np.nanmax(logpoststamp_cube))
    #     post_pos = np.nansum(posterior,axis=0)
    #     post_wvs = np.nansum(posterior,axis=(1,2))
    #     plt.plot(wvshifts_doppler/dwv,post_wvs)
    #     plt.plot(wvshifts_doppler/dwv,np.nanmax(dAICcube,axis=(1,2)),"--")

    plt.show()

    plt.figure(2)
    tmp = np.zeros(300)
    for filename in filelist:
        print(filename)
        # filebasename = os.path.basename(filename)
        # fileelement = planet_c.find(filebasename)
        # print(fileelement.attrib["xADIcen"],fileelement.attrib["yADIcen"])

        try:
        # if 1:
            hdulist = pyfits.open(os.path.join(os.path.dirname(filename),myfolder,
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