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
from scipy.interpolate import interp1d
from reduce_HPFonly_diagcov_resmodel_v2 import LPFvsHPF

out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"

# planet = "b"
# planet = "c"
planet = "d"

# IFSfilter = "Kbb"
# IFSfilter = "Hbb"
# IFSfilter = "all"
suffix = "KbbHbb"
# suffix = "all"
fontsize = 12


# plot for Travis:
if 0:
    osiris_data_dir = "/data/osiris_data"
    molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
    planet_template_folder = os.path.join(osiris_data_dir,"planets_templates")
    fontsize = 12
    IFSfilter = "Kbb"
    cutoff= 40

    if IFSfilter=="Kbb": #Kbb 1965.0 0.25
        CRVAL1 = 1965.
        CDELT1 = 0.25
        nl=1665
        R=4000
    elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1473.
        CDELT1 = 0.2
        nl=1651
        R=4000
    init_wv = CRVAL1/1000.
    dwv = CDELT1/1000.
    wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)
    dprv = 3e5*dwv/(init_wv+dwv*nl//2)

    template_func_list = []
    template_name_list = []
    travis_b_spec_filename=os.path.join(molecular_template_folder,
                                  "hr8799b_kbb_fullsub_bc_25jul13_flam_cal_b.dat")
    data = np.loadtxt(travis_b_spec_filename)
    func = interp1d(data[:,0],data[:,1]-np.mean(data[:,1]),bounds_error=False,fill_value=np.nan)
    template_func_list.append(func)
    template_name_list.append("Travis b OSIRIS")

    # for molecule in ["CO","H2O","CH4"]:
    for molecule in ["CH4"]:#,"H2O","CO"]:
        suffix = suffix+"_"+molecule
        print(molecule)
        travis_mol_filename=os.path.join(molecular_template_folder,
                                      "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7")
        travis_mol_filename_D2E=os.path.join(molecular_template_folder,
                                      "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7_D2E")
        mol_template_filename=travis_mol_filename+"_gaussconv_R{0}_{1}.csv".format(R,IFSfilter)

        with open(mol_template_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            list_starspec = list(csv_reader)
            oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
            col_names = oriplanet_spec_str_arr[0]
            oriplanet_spec = oriplanet_spec_str_arr[1::3,1].astype(np.float)
            oriplanet_spec_wvs = oriplanet_spec_str_arr[1::3,0].astype(np.float)
            where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
            oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
            # plt.figure(3)
            # plt.plot(oriplanet_spec_wvs,oriplanet_spec)
            # plt.show()
            planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
            template_func_list.append(planet_spec_func)
            template_name_list.append(molecule)


    plt.figure(1,figsize=(12,8))
    plt.subplot(3,1,1)
    for k,(name,func) in enumerate(zip(template_name_list[0:2],template_func_list[0:2])):
        myLPFspec,myHPFspec = LPFvsHPF(func(wvs),cutoff)
        if k==0:
            plt.plot(wvs,myHPFspec/np.nanmax(np.abs(myHPFspec)),color="red",linestyle="-",label=name,linewidth=1)
        else:
            plt.plot(wvs,myHPFspec/np.nanmax(np.abs(myHPFspec)),color="black",linestyle="--",label=name,linewidth=1)
    plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.xlim([2.15,2.4])
    plt.legend()



    c_kms = 299792.458
    rvshifts = np.arange(-100*dprv,100*dprv,dprv)
    argnullrv = np.argmin(np.abs(rvshifts))
    ccf = np.zeros(rvshifts.shape)
    where_ch4 = np.where(wvs>2.15)
    # wvs = wvs[where_ch4]

    func1 = template_func_list[0]
    func2 = template_func_list[1]
    name1 = template_name_list[0]
    name2 = template_name_list[1]

    myLPFspec1,myHPFspec1 = LPFvsHPF(func1(wvs),cutoff)
    myHPFspec1 = myHPFspec1/np.sqrt(np.nansum(myHPFspec1**2))

    for plrv_id in range(np.size(rvshifts)):
        wvs4planet_model = wvs*(1-(rvshifts[plrv_id])/c_kms)
        myLPFspec2,myHPFspec2 = LPFvsHPF(func2(wvs4planet_model),cutoff)
        myHPFspec2 = myHPFspec2/np.sqrt(np.nansum(myHPFspec2**2))
        # myLPFspec3,myHPFspec3 = LPFvsHPF(template_func_list[2](wvs4planet_model),cutoff)
        # myHPFspec3 = myHPFspec3/np.sqrt(np.nansum(myHPFspec3**2))
        # myLPFspec4,myHPFspec4 = LPFvsHPF(template_func_list[3](wvs4planet_model),cutoff)
        # myHPFspec4 = myHPFspec4/np.sqrt(np.nansum(myHPFspec4**2))

        # myHPFspec1_tmp = myHPFspec1-myHPFspec3*np.nansum(myHPFspec1*myHPFspec3)
        # myHPFspec1_tmp = myHPFspec1-myHPFspec4*np.nansum(myHPFspec1*myHPFspec4)
        # ccf[plrv_id] = np.nansum(myHPFspec1*myHPFspec2)

        ccf[plrv_id] = np.nansum(myHPFspec1[where_ch4]*myHPFspec2[where_ch4])

    plt.subplot(3,1,2)
    plt.plot(rvshifts,ccf/np.max(ccf),color="black",linestyle="-",linewidth=2)#"#ff9900"
    plt.plot([-4000,4000],[0,0],color="black",linewidth=0.5)
    plt.gca().annotate(name1+" vs. "+name2,xy=(-1750,1),va="top",ha="left",fontsize=fontsize,color="black")
    # plt.annotate("{0:0.2f}".format(ccf[argnullrv]),xy=(0,ccf[argnullrv]),xytext=(0,ccf[argnullrv]+0.1),xycoords="data",fontsize=fontsize,color="grey")
    # plt.annotate("{0:0.2f}".format(ccf[argnullrv]),xy=(0,ccf[argnullrv]),xytext=(500,ccf[argnullrv]+0.2),xycoords="data",fontsize=fontsize,color="grey",arrowprops={"headwidth":5,"width":1,"facecolor":"grey","shrink":0.1})
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.xlim([-2000,2000])
    plt.ylim([-0.5,1.2])
    plt.xlabel(r"$\Delta V$ (km/s)",fontsize=fontsize)
    # plt.xticks([-4000,0,4000])
    # plt.yticks([-0.5,0,0.5,1])

    plt.subplot(3,1,3)
    plt.plot(rvshifts,ccf/np.max(ccf),color="black",linestyle="-",linewidth=2)#"#ff9900"
    plt.plot([-4000,4000],[0,0],color="black",linewidth=0.5)
    plt.gca().annotate(name1+" vs. "+name2,xy=(-3500,1),va="top",ha="left",fontsize=fontsize,color="black")
    # plt.annotate("{0:0.2f}".format(ccf[argnullrv]),xy=(0,ccf[argnullrv]),xytext=(0,ccf[argnullrv]+0.1),xycoords="data",fontsize=fontsize,color="grey")
    # plt.annotate("{0:0.2f}".format(ccf[argnullrv]),xy=(0,ccf[argnullrv]),xytext=(500,ccf[argnullrv]+0.2),xycoords="data",fontsize=fontsize,color="grey",arrowprops={"headwidth":5,"width":1,"facecolor":"grey","shrink":0.1})
    plt.gca().tick_params(axis='x', labelsize=fontsize)
    plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.xlim([-4000,4000])
    plt.ylim([-0.5,1.2])
    plt.xlabel(r"$\Delta V$ (km/s)",fontsize=fontsize)

    plt.show()


# plot molecular templates
if 0:
    osiris_data_dir = "/data/osiris_data"
    molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
    planet_template_folder = os.path.join(osiris_data_dir,"planets_templates")
    fontsize = 12
    for IFSfilter in ["Kbb","Hbb"]:
        if IFSfilter=="Kbb": #Kbb 1965.0 0.25
            CRVAL1 = 1965.
            CDELT1 = 0.25
            nl=1665
            R=4000
        elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
            CRVAL1 = 1473.
            CDELT1 = 0.2
            nl=1651
            R=4000
        init_wv = CRVAL1/1000.
        dwv = CDELT1/1000.
        wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)
        dprv = 3e5*dwv/(init_wv+dwv*nl//2)

        template_func_list = []
        template_name_list = []

        for planet in ["b","c"]:
            travis_spec_filename=os.path.join(planet_template_folder,
                                              "HR8799"+planet+"_"+IFSfilter[0:1]+"_3Oct2018.save")
            planet_template_filename=travis_spec_filename.replace(".save",
                                                                  "_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))

            # if len(glob.glob(planet_template_filename)) == 0:
            #     travis_spectrum = scio.readsav(travis_spec_filename)
            #     ori_planet_spec = np.array(travis_spectrum["fmod"])
            #     ori_planet_convspec = np.array(travis_spectrum["fmods"])
            #     wmod = np.array(travis_spectrum["wmod"])/1.e4
            #     print("convolving: "+planet_template_filename)
            #     planet_convspec = convolve_spectrum(wmod,ori_planet_spec,R,specpool)
            #
            #     with open(planet_template_filename, 'w+') as csvfile:
            #         csvwriter = csv.writer(csvfile, delimiter=' ')
            #         csvwriter.writerows([["wvs","spectrum"]])
            #         csvwriter.writerows([[a,b] for a,b in zip(wmod,planet_convspec)])

            print(planet_template_filename)
            with open(planet_template_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                list_starspec = list(csv_reader)
                oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                col_names = oriplanet_spec_str_arr[0]
                oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
                oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
                where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
                template_func_list.append(planet_spec_func)
                template_name_list.append("model "+planet)

        for molecule in ["CO","H2O"]:#,"CH4"]:
        # for molecule in ["CH4"]:
            suffix = suffix+"_"+molecule
            print(molecule)
            travis_mol_filename=os.path.join(molecular_template_folder,
                                          "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7")
            travis_mol_filename_D2E=os.path.join(molecular_template_folder,
                                          "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7_D2E")
            mol_template_filename=travis_mol_filename+"_gaussconv_R{0}_{1}.csv".format(R,IFSfilter)

            # if len(glob.glob(mol_template_filename)) == 0:
            #     data = np.loadtxt(travis_mol_filename_D2E)
            #     print(data.shape)
            #     wmod = data[:,0]/10000.
            #     wmod_argsort = np.argsort(wmod)
            #     wmod= wmod[wmod_argsort]
            #     crop_moltemp = np.where((wmod>wvs[0]-(wvs[-1]-wvs[0])/2)*(wmod<wvs[-1]+(wvs[-1]-wvs[0])/2))
            #     wmod = wmod[crop_moltemp]
            #     mol_temp = data[wmod_argsort,1][crop_moltemp]
            #     plt.figure(3)
            #     plt.plot(wmod,mol_temp)
            #     plt.show()
            #
            #     # import matplotlib.pyplot as plt
            #     # plt.plot(wmod,mol_temp)#,data[::100,1])
            #     # print(mol_temp.shape)
            #     # plt.show()
            #     # exit()
            #     print("convolving: "+mol_template_filename)
            #     planet_convspec = convolve_spectrum(wmod,mol_temp,R,specpool)
            #
            #     with open(mol_template_filename, 'w+') as csvfile:
            #         csvwriter = csv.writer(csvfile, delimiter=' ')
            #         csvwriter.writerows([["wvs","spectrum"]])
            #         csvwriter.writerows([[a,b] for a,b in zip(wmod,planet_convspec)])

            with open(mol_template_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                list_starspec = list(csv_reader)
                oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                col_names = oriplanet_spec_str_arr[0]
                oriplanet_spec = oriplanet_spec_str_arr[1::3,1].astype(np.float)
                oriplanet_spec_wvs = oriplanet_spec_str_arr[1::3,0].astype(np.float)
                where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                # plt.figure(3)
                # plt.plot(oriplanet_spec_wvs,oriplanet_spec)
                # plt.show()
                planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
                template_func_list.append(planet_spec_func)
                template_name_list.append(molecule)

        # travis_newCH4_spec_filename=os.path.join(molecular_template_folder,
        #                               "hr8799b_kbb_fullsub_bc_25jul13_flam_cal_b.dat")
        # data = np.loadtxt(travis_newCH4_spec_filename)
        # # from reduce_HPFonly_diagcov import convolve_spectrum
        # # planet_convspec = convolve_spectrum(data[:,0],data[:,1],R)
        # func = interp1d(data[:,0],data[:,1]-np.mean(data[:,1]),bounds_error=False,fill_value=np.nan)
        # myLPFspec,myHPFspec = LPFvsHPF(func(wvs),40)
        # template_func_list.append(func)
        # template_name_list.append("Travis b OSIRIS")
        # # # # print(data.shape)
        # # # plt.plot(data[:,0],func(wvs),label="col 2")
        # # # plt.plot(data[:,0],myHPFspec,label="col 2")
        # # # # plt.plot(data[:,0],data[:,2],label="col 3")
        # # # # plt.plot(data[:,0],data[:,3],label="col 4")
        # # # # plt.legend()
        # # # plt.show()
        # # # # exit(0)


        color_list = ["#0099cc","#ff9900","black","black","black","black"]
        plt.figure(1,figsize=(12,5))
        for k,(name,func,color) in enumerate(zip(template_name_list,template_func_list,color_list)):
            myLPFspec,myHPFspec = LPFvsHPF(func(wvs),40)
            plt.plot(wvs,-k+0.5*myHPFspec/np.nanmax(np.abs(myHPFspec)),color=color,linestyle="-")
            if name =="CH4":
                plt.fill_between([2.315,2.324],[-k-0.5,-k-0.5],[-k+0.5,-k+0.5],alpha=0.5,color="grey")
                plt.fill_between([2.369,2.374],[-k-0.5,-k-0.5],[-k+0.5,-k+0.5],alpha=0.5,color="grey")
            if IFSfilter == "Kbb":
                plt.gca().text(1.96,-k,name,ha="right",va="center",rotation=0,size=fontsize*1.25,color=color)
                # plt.gca().text(2.2,-k,name,ha="right",va="center",rotation=0,size=fontsize*1.25,color=color)
            if IFSfilter == "Hbb":
                plt.gca().text(1.47,-k,name,ha="right",va="center",rotation=0,size=fontsize*1.25,color=color)
        if IFSfilter == "Kbb":
            plt.xlim([1.90,2.4])
            # plt.xlim([2.15,2.4])
        elif IFSfilter == "Hbb":
            plt.xlim([1.42,1.81])
        plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        plt.tick_params(axis="y",which="both",labelleft=False,right=False,left=False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        # plt.legend()
        # plt.show()
        print("Saving "+os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_templates.png"))
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_templates.png"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_templates.pdf"),bbox_inches='tight')
        plt.close(1)
    exit()

#template CCF
if 0:
    cutoff = 40
    osiris_data_dir = "/data/osiris_data"
    molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
    planet_template_folder = os.path.join(osiris_data_dir,"planets_templates")
    fontsize = 12
    for IFSfilter in ["Kbb","Hbb"]:
    # for IFSfilter in ["Hbb"]:
        if IFSfilter=="Kbb": #Kbb 1965.0 0.25
            CRVAL1 = 1965.
            CDELT1 = 0.25
            nl=1665
            R=4000
        elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
            CRVAL1 = 1473.
            CDELT1 = 0.2
            nl=1651
            R=4000
        init_wv = CRVAL1/1000.
        dwv = CDELT1/1000.
        wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)
        dprv = 3e5*dwv/(init_wv+dwv*nl//2)

        template_func_list = []
        template_name_list = []
        color_list = ["#0099cc","#ff9900","black","black","black"]


        # planet = "c"
        for planet in ["b","c"]:
            travis_spec_filename=os.path.join(planet_template_folder,
                                              "HR8799"+planet+"_"+IFSfilter[0:1]+"_3Oct2018.save")
            planet_template_filename=travis_spec_filename.replace(".save",
                                                                  "_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))

            # if len(glob.glob(planet_template_filename)) == 0:
            #     travis_spectrum = scio.readsav(travis_spec_filename)
            #     ori_planet_spec = np.array(travis_spectrum["fmod"])
            #     ori_planet_convspec = np.array(travis_spectrum["fmods"])
            #     wmod = np.array(travis_spectrum["wmod"])/1.e4
            #     print("convolving: "+planet_template_filename)
            #     planet_convspec = convolve_spectrum(wmod,ori_planet_spec,R,specpool)
            #
            #     with open(planet_template_filename, 'w+') as csvfile:
            #         csvwriter = csv.writer(csvfile, delimiter=' ')
            #         csvwriter.writerows([["wvs","spectrum"]])
            #         csvwriter.writerows([[a,b] for a,b in zip(wmod,planet_convspec)])

            print(planet_template_filename)
            with open(planet_template_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                list_starspec = list(csv_reader)
                oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                col_names = oriplanet_spec_str_arr[0]
                oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
                oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
                where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
                template_func_list.append(planet_spec_func)
                template_name_list.append("model "+planet)

        # travis_newCH4_spec_filename=os.path.join(molecular_template_folder,
        #                               "hr8799b_kbb_fullsub_bc_25jul13_flam_cal_b.dat")
        # data = np.loadtxt(travis_newCH4_spec_filename)
        # # from reduce_HPFonly_diagcov import convolve_spectrum
        # # planet_convspec = convolve_spectrum(data[:,0],data[:,1],R)
        # func = interp1d(data[:,0],data[:,1]-np.mean(data[:,1]),bounds_error=False,fill_value=np.nan)
        # myLPFspec,myHPFspec = LPFvsHPF(func(wvs),40)
        # template_func_list.append(func)
        # template_name_list.append("Travis b")
        # # # # print(data.shape)
        # # # plt.plot(data[:,0],func(wvs),label="col 2")
        # # # plt.plot(data[:,0],myHPFspec,label="col 2")
        # # # # plt.plot(data[:,0],data[:,2],label="col 3")
        # # # # plt.plot(data[:,0],data[:,3],label="col 4")
        # # # # plt.legend()
        # # # plt.show()
        # # # # exit(0)

        for molecule in ["CO","H2O","CH4"]:
            suffix = suffix+"_"+molecule
            print(molecule)
            travis_mol_filename=os.path.join(molecular_template_folder,
                                          "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7")
            travis_mol_filename_D2E=os.path.join(molecular_template_folder,
                                          "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7_D2E")
            mol_template_filename=travis_mol_filename+"_gaussconv_R{0}_{1}.csv".format(R,IFSfilter)

            # if len(glob.glob(mol_template_filename)) == 0:
            #     data = np.loadtxt(travis_mol_filename_D2E)
            #     print(data.shape)
            #     wmod = data[:,0]/10000.
            #     wmod_argsort = np.argsort(wmod)
            #     wmod= wmod[wmod_argsort]
            #     crop_moltemp = np.where((wmod>wvs[0]-(wvs[-1]-wvs[0])/2)*(wmod<wvs[-1]+(wvs[-1]-wvs[0])/2))
            #     wmod = wmod[crop_moltemp]
            #     mol_temp = data[wmod_argsort,1][crop_moltemp]
            #
            #     # import matplotlib.pyplot as plt
            #     # plt.plot(wmod,mol_temp)#,data[::100,1])
            #     # print(mol_temp.shape)
            #     # plt.show()
            #     # exit()
            #     print("convolving: "+mol_template_filename)
            #     planet_convspec = convolve_spectrum(wmod,mol_temp,R,specpool)
            #
            #     with open(mol_template_filename, 'w+') as csvfile:
            #         csvwriter = csv.writer(csvfile, delimiter=' ')
            #         csvwriter.writerows([["wvs","spectrum"]])
            #         csvwriter.writerows([[a,b] for a,b in zip(wmod,planet_convspec)])

            with open(mol_template_filename, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                list_starspec = list(csv_reader)
                oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                col_names = oriplanet_spec_str_arr[0]
                oriplanet_spec = oriplanet_spec_str_arr[1::3,1].astype(np.float)
                oriplanet_spec_wvs = oriplanet_spec_str_arr[1::3,0].astype(np.float)
                where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
                template_func_list.append(planet_spec_func)
                template_name_list.append(molecule)


        c_kms = 299792.458
        rvshifts = np.arange(-100*dprv,100*dprv,dprv)
        argnullrv = np.argmin(np.abs(rvshifts))
        f1,ax_CCF_list = plt.subplots(len(template_func_list),len(template_func_list),sharey="row",sharex="col",figsize=(12,12))
        # where_ch4 = np.where(wvs>2.15)
        # wvs = wvs[where_ch4]
        for k,(name1,func1) in enumerate(zip(template_name_list,template_func_list)):
            for l,(name2,func2,color) in enumerate(zip(template_name_list,template_func_list,color_list)):
                print(name1,name2)
                ccf = np.zeros(rvshifts.shape)
                # plt.plot(wvs,func1(wvs))
                # plt.show()
                myLPFspec1,myHPFspec1 = LPFvsHPF(func1(wvs),cutoff)
                myHPFspec1 = myHPFspec1/np.sqrt(np.nansum(myHPFspec1**2))
                for plrv_id in range(np.size(rvshifts)):
                    wvs4planet_model = wvs*(1-(rvshifts[plrv_id])/c_kms)
                    myLPFspec2,myHPFspec2 = LPFvsHPF(func2(wvs4planet_model),cutoff)
                    myHPFspec2 = myHPFspec2/np.sqrt(np.nansum(myHPFspec2**2))

                    ccf[plrv_id] = np.nansum(myHPFspec1*myHPFspec2)

                plt.sca(ax_CCF_list[k][l])
                if k ==l:
                    plt.plot(rvshifts,ccf,color=color,linestyle="-",linewidth=2)#"#ff9900"
                    plt.gca().annotate(name1,xy=(-3500,1.2),va="bottom",ha="left",fontsize=fontsize,color=color)
                else:
                    plt.plot(rvshifts,ccf,color="grey",linestyle="-",linewidth=2)#"#ff9900"
                    plt.gca().annotate(name1+" vs. "+name2,xy=(-3500,1.2),va="bottom",ha="left",fontsize=fontsize,color="black")
                    plt.annotate("{0:0.2f}".format(ccf[argnullrv]),xy=(0,ccf[argnullrv]),xytext=(0,ccf[argnullrv]+0.1),xycoords="data",fontsize=fontsize,color="grey")
                    # plt.annotate("{0:0.2f}".format(ccf[argnullrv]),xy=(0,ccf[argnullrv]),xytext=(500,ccf[argnullrv]+0.2),xycoords="data",fontsize=fontsize,color="grey",arrowprops={"headwidth":5,"width":1,"facecolor":"grey","shrink":0.1})
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                plt.xlim([-4000,4000])
                plt.ylim([-0.5,1.4])
                plt.xticks([0,4000])
                plt.yticks([0,0.5,1])

        plt.figure(f1.number)
        plt.sca(ax_CCF_list[-1][0])
        # plt.ylabel("$\propto S/N$",fontsize=15)
        plt.ylabel("CCF",fontsize=15)
        plt.xlabel(r"$\Delta V$ (km/s)",fontsize=fontsize)
        # plt.tick_params(axis="y",which="both",labelleft=False,bottom=False,top=False)
        plt.xticks([-4000,0,4000])
        plt.yticks([-0.5,0,0.5,1])
        # plt.xticks([-2000,-1000,0,1000,2000])
        # plt.yticks([-10,0,10,20,30,40])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        f1.subplots_adjust(wspace=0,hspace=0)
        # plt.show()
        print("Saving "+os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_templates_CCF_v2.png"))
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_templates_CCF_v2.png"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_templates_CCF_v2.pdf"),bbox_inches='tight')


    # plt.show()

    exit()



# plot CCF
if 0:
    fontsize = 12
    for IFSfilter in ["Hbb"]:#["Kbb","Hbb"]:
        if IFSfilter=="Kbb": #Kbb 1965.0 0.25
            CRVAL1 = 1965.
            CDELT1 = 0.25
            nl=1665
            R=4000
            f1,ax_CCF_list = plt.subplots(4,3,sharey="row",sharex="col",figsize=(12,12))#figsize=(12,8)
            f2,ax_histo_list = plt.subplots(4,3,sharey="row",sharex="col",figsize=(12,12))#figsize=(12,8)
            f4,ax_CCFsummary_list = plt.subplots(3,1,sharex="col",figsize=(6,9))#figsize=(12,8)
            planet_list = ["b","c","d"]
            # planet_list = ["c"]
        elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
            CRVAL1 = 1473.
            CDELT1 = 0.2
            nl=1651
            R=5000
            f1,ax_CCF_list = plt.subplots(4,2,sharey="row",sharex="col",figsize=(8,12))#figsize=(12,8)
            f2,ax_histo_list = plt.subplots(4,2,sharey="row",sharex="col",figsize=(8,12))#figsize=(12,8)
            f4,ax_CCFsummary_list = plt.subplots(2,1,sharex="col",figsize=(6,6))#figsize=(12,8)
            planet_list = ["b","c"]
        linestyle_list = ["-","-","--","-.",":"]
        dwv = CDELT1/1000.
        init_wv = CRVAL1/1000. # wv for first slice in mum
        for plid,(planet,color) in enumerate(zip(planet_list,["#0099cc","#ff9900","#6600ff"])):

            fileinfos_filename = "/data/osiris_data/HR_8799_"+planet+"/fileinfos_Kbb_jb.csv"

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
            molecule_list = ["","_CO","_H2O","_CH4"]
            molecule_str_list = ["model","CO","H2O","CH4"]
            for molid,(molecule,molecule_str) in enumerate(zip(molecule_list,molecule_str_list)):
                # plt.sca(ax_CCF_list[molid][plid])
                # plt.sca(ax_histo_list[molid][plid])

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

                SNR_hist_list = []
                f3,ax_histoperfile_list = plt.subplots(10,10,sharey="row",sharex="col",figsize=(20,20))#figsize=(12,8)
                ax_histoperfile_list = [ax for ax_list in ax_histoperfile_list for ax in ax_list]
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

                    SNR_data_cp[zcen-5:zcen+5,kcen-5:kcen+5,lcen-5:lcen+5] = np.nan
                    # print(SNR_data_cp.shape)
                    # exit()
                    meanSNR1 = np.nanmedian(SNR_data_cp,axis=0)[None,:,:]
                    SNR_data_cp = (SNR_data_cp-meanSNR1)
                    meanSNR2 = np.nanmedian(SNR_data_cp,axis=(1,2))[:,None,None]
                    SNR_data_cp = (SNR_data_cp-meanSNR2)
                    stdSNR2 = np.nanstd(SNR_data_cp,axis=(1,2))[:,None,None]
                    SNR_data_cp = SNR_data_cp/stdSNR2
                    stdSNR1 = np.nanstd(SNR_data_cp,axis=0)[None,:,:]
                    SNR_data_cp = SNR_data_cp/stdSNR1

                    SNR_data_calib = (SNR_data-meanSNR2-meanSNR1)/stdSNR1/stdSNR2

                    # print(len(ax_histoperfile_list),k)
                    SNR_hist,bin_edges = np.histogram(np.ravel(SNR_data_cp)[np.where(np.isfinite(np.ravel(SNR_data_cp)))],bins=800,range=[-40,40])
                    SNR_hist_list.append(SNR_hist)
                    bin_center = (bin_edges[1::]+bin_edges[0:np.size(bin_edges)-1])/2

                    plt.sca(ax_histoperfile_list[k])
                    plt.title(os.path.basename(item[filename_id]).split("bb_")[0],fontsize=5)
                    plt.plot(bin_center,SNR_hist/np.max(SNR_hist),linewidth=3,color=color,label="hist")
                    plt.plot(np.linspace(-7,7,200),1/np.sqrt(2*np.pi)*np.exp(-0.5*np.linspace(-7,7,200)**2),linestyle="--",linewidth=1,color="black",label="Gaussian")
                    plt.xlim([-10,10])
                    plt.ylim([1e-10,10])
                    plt.yscale("log")
                    # plt.show()

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
                    SNR_data_calib_hd_cp = copy(SNR_data_calib_hd)
                    SNR_data_calib_hd_cp[:,kcen-5:kcen+5,lcen-5:lcen+5] = np.nan
                    # plt.figure(4)
                    # plt.plot(np.nanmedian(SNR_data_calib_hd_cp,axis=(1,2)))
                    # plt.show()
                    meanSNR3 = np.nanmedian(SNR_data_calib_hd_cp,axis=(1,2))[:,None,None]
                    stdSNR2_hd = np.nanstd(SNR_data_calib_hd_cp,axis=(1,2))[:,None,None]
                    SNR_data_calib_hd = (SNR_data_calib_hd-meanSNR3)/stdSNR1/stdSNR2_hd

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
                    # print("zcenhd",zcenhd)
                    summed_hdRV[((400*3)//2-zcenhd):((400*3)//2+NplanetRV_hd-zcenhd),
                    ((64*3)//2-kcen):((64*3)//2+ny-kcen),
                    ((19*3)//2-lcen):((19*3)//2+nx-lcen)] += copy(SNR_data_calib_hd)


                # f3.subplots_adjust(wspace=0,hspace=0)
                # print("Saving "+os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_SNRhistmozaic"+molecule+".png"))
                # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_SNRhistmozaic"+molecule+".png"),bbox_inches='tight')
                # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_SNRhistmozaic"+molecule+".pdf"),bbox_inches='tight')
                plt.close(f3.number)
                # exit()


                # noise1 = copy(summed_wideRV[200:400,((64*3)//2+5):((64*3)//2+15),((19*3)//2-3):((19*3)//2+3)])
                # plt.figure(6)
                # plt.plot(np.nansum(Nvalid_hdRV,axis=(1,2))/np.nanmax(np.nansum(Nvalid_hdRV,axis=(1,2))))
                # plt.plot(summed_hdRV[:,(64*3)//2,(19*3)//2]/np.nanmax(summed_hdRV[:,(64*3)//2,(19*3)//2]))
                summed_wideRV = summed_wideRV/Nvalid_wideRV
                summed_hdRV = summed_hdRV/Nvalid_hdRV
                Nvalid_wideRV = np.sum(Nvalid_wideRV,axis=0)
                where_valid = np.where(Nvalid_wideRV>0.8*np.nanmax(Nvalid_wideRV))
                where_notvalid = np.where(Nvalid_wideRV<=0.8*np.nanmax(Nvalid_wideRV))
                noise1 = copy(summed_wideRV[200:400,:,:])
                noise1[:,((64*3)//2-7):((64*7)//2+7),((19*3)//2-7):((19*3)//2+7)] = np.nan
                noise1[:,where_notvalid[0],where_notvalid[1]] = np.nan
                sigma = np.nanstd(noise1)
                summed_wideRV = summed_wideRV/sigma
                summed_hdRV = summed_hdRV/sigma
                noise1 = noise1/sigma

                plt.figure(f2.number)
                plt.sca(ax_histo_list[molid][plid])
                # master_SNR_hist = np.sum(SNR_hist_list,axis=0)
                master_SNR_hist,bin_edges = np.histogram(np.ravel(noise1)[np.where(np.isfinite(np.ravel(noise1)))],bins=800,range=[-40,40])
                bin_center = (bin_edges[1::]+bin_edges[0:np.size(bin_edges)-1])/2
                plt.plot(bin_center,master_SNR_hist/np.max(master_SNR_hist),linestyle="-",linewidth=2,color=color,label="S/N histogram")
                plt.plot(np.linspace(-10,10,200),np.exp(-0.5*np.linspace(-10,10,200)**2),linestyle="--",linewidth=1,color="black",label="Gaussian")#1/np.sqrt(2*np.pi)*
                plt.xlim([-6,6])
                plt.ylim([1e-4,10])
                plt.xticks([-4,-2,0,2,4,6])
                plt.yticks([1e-4,1e-3,1e-2,1e-1,1e-0])
                plt.yscale("log")
                plt.gca().tick_params(axis='x', labelsize=fontsize,which="both")
                plt.gca().tick_params(axis='y', labelsize=fontsize,which="both")
                plt.gca().annotate(planet+": "+molecule_str,xy=(-5,1),va="top",ha="left",fontsize=fontsize,color="black")
                # plt.show()


                plt.figure(f1.number)
                plt.sca(ax_CCF_list[molid][plid])
                colids_choice = np.random.choice(where_valid[0],size = 100,replace=False)
                rowids_choice = np.random.choice(where_valid[1],size = 100,replace=False)
                for k,l in zip(colids_choice,rowids_choice):
                    plt.plot(planetRV,noise1[:,k,l],alpha=0.1,linestyle="--",linewidth=0.2,color="grey") #006699
                    # plt.plot(planetRV,noise2[:,k,l],alpha=0.5,linestyle="--",linewidth=1,color="cyan")
                plt.plot(planetRV,summed_wideRV[200:400,(64*3)//2,(19*3)//2],linestyle="-",linewidth=2,color=color)
                Nvalid_hdRV = np.sum(Nvalid_hdRV[400:800,:,:],axis=(1,2))
                where_validhd = np.where(Nvalid_hdRV>0.9*np.nanmax(Nvalid_hdRV))
                where_notvalidhd = np.where(Nvalid_hdRV<=0.9*np.nanmax(Nvalid_hdRV))
                plt.plot(planetRV_hd[where_validhd],summed_hdRV[400:800,(64*3)//2,(19*3)//2][where_validhd],linestyle="--",linewidth=1,color="black") #"black","#ff9900","#006699","grey"
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                plt.xlim([-4000,4000])
                plt.xticks([-2000,0,2000,4000])
                if IFSfilter =="Kbb":
                    plt.ylim([-10,50])
                    plt.yticks([0,10,20,30,40,50])
                    plt.gca().annotate(planet+": "+molecule_str,xy=(-3750,45),va="top",ha="left",fontsize=fontsize,color="black")
                    plt.gca().annotate("S/N={0:0.1f}".format(summed_hdRV[(400*3)//2,(64*3)//2,(19*3)//2]),xy=(3750,45),va="top",ha="right",fontsize=fontsize,color="black")
                elif IFSfilter =="Hbb":
                    plt.ylim([-4,20])
                    plt.yticks([0,5,10,15,20])
                    plt.gca().annotate(planet+": "+molecule_str,xy=(-3750,18),va="top",ha="left",fontsize=fontsize,color="black")
                    plt.gca().annotate("S/N={0:0.1f}".format(summed_hdRV[(400*3)//2,(64*3)//2,(19*3)//2]),xy=(3750,18),va="top",ha="right",fontsize=fontsize,color="black")
                # plt.ylim([-2000,2000])
                # plt.ylim([-10,40])
                # plt.xticks([-1000,0,1000,2000])
                # plt.yticks([0,10,20,30,40])


                plt.figure(f4.number)
                plt.sca(ax_CCFsummary_list[plid])
                if plid != 2:
                    if IFSfilter == "Kbb":
                        plt.gca().annotate("HR 8799 "+planet,xy=(-900,45),va="top",ha="left",fontsize=fontsize,color=color)
                    elif IFSfilter == "Hbb":
                        plt.gca().annotate("HR 8799 "+planet,xy=(-900,18),va="top",ha="left",fontsize=fontsize,color=color)
                else:
                    plt.gca().annotate("HR 8799 "+planet,xy=(-900,13),va="top",ha="left",fontsize=fontsize,color=color)
                if molid == 0:
                    plt.plot(planetRV,summed_wideRV[200:400,(64*3)//2,(19*3)//2],linestyle=linestyle_list[molid],linewidth=3,color=color,label=molecule_str+": S/N={0:0.1f}".format(summed_hdRV[(400*3)//2,(64*3)//2,(19*3)//2]))
                else:
                    plt.plot(planetRV,summed_wideRV[200:400,(64*3)//2,(19*3)//2],linestyle=linestyle_list[molid],linewidth=2,color="black",label=molecule_str+": S/N={0:0.1f}".format(summed_hdRV[(400*3)//2,(64*3)//2,(19*3)//2]))
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                plt.xlim([-1000,1000])
                if planet == "b" or planet == "c":
                    if IFSfilter == "Kbb":
                        plt.ylim([-10,50])
                        plt.yticks([0,10,20,30,40,50])
                    elif IFSfilter == "Hbb":
                        plt.ylim([-5,20])
                        plt.yticks([0,5,10,15,20])
                elif planet == "d":
                    plt.ylim([-3,15])
                    plt.yticks([0,5,10,15])
                plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#
                # plt.show()

        plt.figure(f4.number)
        plt.sca(ax_CCFsummary_list[-1])
        plt.ylabel("S/N",fontsize=15)
        plt.xlabel(r"$\Delta V$ (km/s)",fontsize=fontsize)
        plt.xticks([-1000,-500,0,500,1000])
        # plt.xticks([-2000,-1000,0,1000,2000])
        # plt.yticks([-10,0,10,20,30,40])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        f4.subplots_adjust(wspace=0,hspace=0)

        print("Saving "+os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCFsummary2.png"))
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCFsummary2.png"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCFsummary2.pdf"),bbox_inches='tight')
        plt.close(f4.number)

        plt.figure(f1.number)
        plt.sca(ax_CCF_list[-1][0])
        plt.ylabel("S/N",fontsize=15)
        plt.xlabel(r"$\Delta V$ (km/s)",fontsize=fontsize)
        plt.xticks([-4000,-2000,0,2000,4000])
        if IFSfilter =="Kbb":
            plt.yticks([-10,0,10,20,30,40,50])
        elif IFSfilter =="Hbb":
            plt.yticks([-5,0,5,10,15,20])
        # plt.xticks([-2000,-1000,0,1000,2000])
        # plt.yticks([-10,0,10,20,30,40])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        f1.subplots_adjust(wspace=0,hspace=0)
        # plt.show()

        print("Saving "+os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF2.png"))
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF2.png"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF2.pdf"),bbox_inches='tight')
        plt.close(f1.number)


        plt.figure(f2.number)
        for m in range(len(planet_list)):
            plt.sca(ax_histo_list[0][m])
            plt.legend(loc="lower center",frameon=True,fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize,which="both")
        plt.gca().tick_params(axis='y', labelsize=fontsize,which="both")
        plt.sca(ax_histo_list[0][0])
        plt.xticks([-6,-4,-2,0,2,4,6])
        for m in range(len(molecule_list)):
            plt.sca(ax_histo_list[m][0])
            plt.yticks([1e-4,1e-3,1e-2,1e-1,1e-0])
        f2.subplots_adjust(wspace=0,hspace=0)

        print("Saving "+os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF_histo2.png"))
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF_histo2.png"),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF_histo2.pdf"),bbox_inches='tight')
        plt.close(f2.number)
        # plt.show()
        # exit()


            # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+".png"),bbox_inches='tight')
            # print("Saving "+os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+".pdf"))
            # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+".pdf"),bbox_inches='tight')
            #
            # plt.gca().annotate(molecule_str,xy=(-1450,24.5),va="top",ha="left",fontsize=15,color="black")
            # plt.xlim([-1500,1500])
            # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+"_zoomed.png"),bbox_inches='tight')
            # print("Saving "+os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+"_zoomed.pdf"))
            # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_CCF"+molecule+"_zoomed.pdf"),bbox_inches='tight')
            # plt.close(1)
    exit()




# plot CCF
if 1:
    resnumbasis = 10
    fontsize = 12
    for IFSfilter in ["Kbb"]:#,"Hbb"]:
        if IFSfilter=="Kbb": #Kbb 1965.0 0.25
            CRVAL1 = 1965.
            CDELT1 = 0.25
            nl=1665
            R=4000
            f1,ax_CCF_list = plt.subplots(4,3,sharey="row",sharex="col",figsize=(12,12*4/5))#figsize=(12,8)
            f2,ax_histo_list = plt.subplots(4,3,sharey="row",sharex="col",figsize=(12,12*4/5))#figsize=(12,8)
            f4,ax_CCFsummary_list = plt.subplots(3,1,sharex="col",figsize=(6,9))#figsize=(12,8)
            planet_list = ["b","c","d"]
            # planet_list = ["c","d"]
        elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
            CRVAL1 = 1473.
            CDELT1 = 0.2
            nl=1651
            R=4000
            f1,ax_CCF_list = plt.subplots(4,2,sharey="row",sharex="col",figsize=(8,12*4/5))#figsize=(12,8)
            f2,ax_histo_list = plt.subplots(4,2,sharey="row",sharex="col",figsize=(8,12*4/5))#figsize=(12,8)
            f4,ax_CCFsummary_list = plt.subplots(2,1,sharex="col",figsize=(6,6))#figsize=(12,8)
            planet_list = ["b","c"]
        linestyle_list = ["-","-","--","-.",":"]
        dwv = CDELT1/1000.
        init_wv = CRVAL1/1000. # wv for first slice in mum
        for plid,(planet,color) in enumerate(zip(planet_list,["#0099cc","#ff9900","#6600ff"])):

            if resnumbasis == 0:
                fileinfos_filename = "/data/osiris_data/HR_8799_"+planet+"/fileinfos_Kbb_jb.csv"
            else:
                fileinfos_filename = "/data/osiris_data/HR_8799_"+planet+"/fileinfos_Kbb_jb_kl{0}.csv".format(resnumbasis)

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

            # molecule_list = ["_H2O"]#["","_CH4","_CO","_CO2","_H2O"]
            # molecule_str_list = ["H2O"]#["Atmospheric model","CH4","CO","CO2","H2O"]
            molecule_list = ["","_CO","_H2O","_CH4"]
            molecule_str_list = ["Model","CO","H2O","CH4"]
            # molecule_list = [""]
            # molecule_str_list = ["Atmospheric model"]
            # molecule_list = ["_CO"]
            # molecule_str_list = ["CO"]
            # molecule_list = ["_CO2"]
            # molecule_str_list = ["CO2"]
            # molecule_list = ["_CH4"]
            # molecule_str_list = ["CH4"]
            # molecule_list = ["_CO","_H2O","_CH4","_CO2"]
            # molecule_str_list = ["CO","H2O","CH4","CO2"]
            # molecule_list = ["_CO","_H2O","_CH4"]
            # molecule_str_list = ["CO","H2O","CH4"]
            for molid,(molecule,molecule_str) in enumerate(zip(molecule_list,molecule_str_list)):
                # plt.sca(ax_CCF_list[molid][plid])
                # plt.sca(ax_histo_list[molid][plid])

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

                SNR_hist_list = []
                f3,ax_histoperfile_list = plt.subplots(10,10,sharey="row",sharex="col",figsize=(20,20))#figsize=(12,8)
                ax_histoperfile_list = [ax for ax_list in ax_histoperfile_list for ax in ax_list]
                for k,item in enumerate(filtered_list_data):
                    # reducfilename = item[cen_filename_id].replace("search","search_CO2_CO_H2O_CH4")
                    # reducfilename = item[cen_filename_id].replace("search","search_CO_H2O_CH4_joint")
                    # print(item[cen_filename_id].replace("search","search_CO2_CO_H2O_CH4"))
                    if molecule != "":
                        # reducfilename = item[cen_filename_id].replace("20191205_RV","20200213_molecules").replace("search","search_"+molecule_str)
                        reducfilename = item[cen_filename_id].replace("20191205_RV","20200518_molecules").replace("search","search_"+molecule_str)
                    else:
                        reducfilename = item[cen_filename_id]
                    # print(reducfilename)
                    # exit()
                    if len(glob.glob(reducfilename.replace(".fits","_planetRV.fits"))) == 0:
                        continue
                    print(reducfilename)
                    # exit()
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

                    bary_rv = -float(item[bary_rv_id])/1000. # RV in km/s
                    rv_star = -12.6#-12.6+-1.4km/s HR 8799 Rob and Simbad

                    kcen = int(item[kcen_id])
                    lcen = int(item[lcen_id])
                    rvcen = float(item[rvcen_id])
                    zcenhd = np.argmin(np.abs(planetRV_hd-rvcen))
                    zcen = np.argmin(np.abs(planetRV-rvcen))

                    # # if molecule_str == "CO2":
                    # #     SNR_data = hdulist[0].data[0,0,13,NplanetRV_hd::,:,:]
                    # # elif molecule_str == "CO":
                    # #     SNR_data = hdulist[0].data[0,0,16,NplanetRV_hd::,:,:]
                    # # elif molecule_str == "H2O":
                    # #     SNR_data = hdulist[0].data[0,0,19,NplanetRV_hd::,:,:]
                    # # elif molecule_str == "CH4":
                    # #     SNR_data = hdulist[0].data[0,0,22,NplanetRV_hd::,:,:]
                    # # else:
                    # #     exit()
                    # if molecule_str == "CO":
                    #     SNR_data = hdulist[0].data[0,0,13,NplanetRV_hd::,:,:]
                    # elif molecule_str == "H2O":
                    #     SNR_data = hdulist[0].data[0,0,16,NplanetRV_hd::,:,:]
                    # elif molecule_str == "CH4":
                    #     SNR_data = hdulist[0].data[0,0,10,NplanetRV_hd::,:,:]
                    # else:
                    #     exit()
                    SNR_data = hdulist[0].data[0,0,10,NplanetRV_hd::,:,:]
                    SNR_data_cp = copy(SNR_data)
                    SNR_data_cp[np.where(np.abs(SNR_data)>100)] = np.nan

                    SNR_data_cp[zcen-3:zcen+3,kcen-3:kcen+3,lcen-3:lcen+3] = np.nan
                    # print(SNR_data_cp.shape)
                    # exit()
                    meanSNR1 = np.nanmedian(SNR_data_cp,axis=0)[None,:,:]
                    SNR_data_cp = (SNR_data_cp-meanSNR1)
                    meanSNR2 = np.nanmedian(SNR_data_cp,axis=(1,2))[:,None,None]
                    SNR_data_cp = (SNR_data_cp-meanSNR2)
                    stdSNR = np.nanstd(SNR_data_cp,axis=0)[None,:,:]
                    SNR_data_cp = SNR_data_cp/stdSNR

                    SNR_data_calib = (SNR_data-meanSNR2-meanSNR1)/stdSNR

                    # print(len(ax_histoperfile_list),k)
                    SNR_hist,bin_edges = np.histogram(np.ravel(SNR_data_cp)[np.where(np.isfinite(np.ravel(SNR_data_cp)))],bins=800,range=[-40,40])
                    SNR_hist_list.append(SNR_hist)
                    bin_center = (bin_edges[1::]+bin_edges[0:np.size(bin_edges)-1])/2

                    plt.sca(ax_histoperfile_list[k])
                    plt.title(os.path.basename(item[filename_id]).split("bb_")[0],fontsize=5)
                    plt.plot(bin_center,SNR_hist/np.max(SNR_hist),linewidth=3,color=color,label="hist")
                    plt.plot(np.linspace(-7,7,200),1/np.sqrt(2*np.pi)*np.exp(-0.5*np.linspace(-7,7,200)**2),linestyle="--",linewidth=1,color="black",label="Gaussian")
                    plt.xlim([-10,10])
                    plt.ylim([1e-10,10])
                    plt.yscale("log")
                    # plt.show()

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

                    # # if molecule_str == "CO2":
                    # #     SNR_data_hd = hdulist[0].data[0,0,13,0:NplanetRV_hd,:,:]
                    # # elif molecule_str == "CO":
                    # #     SNR_data_hd = hdulist[0].data[0,0,16,0:NplanetRV_hd,:,:]
                    # # elif molecule_str == "H2O":
                    # #     SNR_data_hd = hdulist[0].data[0,0,19,0:NplanetRV_hd,:,:]
                    # # elif molecule_str == "CH4":
                    # #     SNR_data_hd = hdulist[0].data[0,0,22,0:NplanetRV_hd,:,:]
                    # # else:
                    # #     exit()
                    # if molecule_str == "CO":
                    #     SNR_data_hd = hdulist[0].data[0,0,13,0:NplanetRV_hd,:,:]
                    # elif molecule_str == "H2O":
                    #     SNR_data_hd = hdulist[0].data[0,0,16,0:NplanetRV_hd,:,:]
                    # elif molecule_str == "CH4":
                    #     SNR_data_hd = hdulist[0].data[0,0,10,0:NplanetRV_hd,:,:]
                    # else:
                    #     exit()
                    SNR_data_hd = hdulist[0].data[0,0,10,0:NplanetRV_hd,:,:]
                    SNR_data_calib_hd = SNR_data_hd-meanSNR1
                    SNR_data_calib_hd_cp = copy(SNR_data_calib_hd)
                    SNR_data_calib_hd_cp[:,kcen-3:kcen+3,lcen-3:lcen+3] = np.nan
                    # plt.figure(4)
                    # plt.plot(np.nanmedian(SNR_data_calib_hd_cp,axis=(1,2)))
                    # plt.show()
                    meanSNR3 = np.nanmedian(SNR_data_calib_hd_cp,axis=(1,2))[:,None,None]
                    SNR_data_calib_hd = (SNR_data_calib_hd-meanSNR3)/stdSNR

                    canvas = np.zeros(SNR_data_hd.shape)
                    canvas[np.where(np.isfinite(SNR_data_hd))] = 1
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
                    # print("zcenhd",zcenhd)
                    summed_hdRV[((400*3)//2-zcenhd):((400*3)//2+NplanetRV_hd-zcenhd),
                    ((64*3)//2-kcen):((64*3)//2+ny-kcen),
                    ((19*3)//2-lcen):((19*3)//2+nx-lcen)] += copy(SNR_data_calib_hd)


                # f3.subplots_adjust(wspace=0,hspace=0)
                # print("Saving "+os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_SNRhistmozaic"+molecule+".png"))
                # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_SNRhistmozaic"+molecule+".png"),bbox_inches='tight')
                # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR8799"+planet+"_"+IFSfilter+"_SNRhistmozaic"+molecule+".pdf"),bbox_inches='tight')
                plt.close(f3.number)
                # exit()

                plt.figure(f2.number)
                plt.sca(ax_histo_list[molid][plid])
                master_SNR_hist = np.sum(SNR_hist_list,axis=0)
                plt.plot(bin_center,master_SNR_hist/np.max(master_SNR_hist),linestyle="-",linewidth=3,color=color,label="S/N histogram")
                plt.plot(np.linspace(-10,10,200),np.exp(-0.5*np.linspace(-10,10,200)**2),linestyle="--",linewidth=1,color="black",label="Gaussian")#1/np.sqrt(2*np.pi)*
                plt.xlim([-6,6])
                plt.ylim([1e-6,10])
                plt.xticks([-4,-2,0,2,4,6])
                plt.yticks([1e-6,1e-4,1e-2,1e-0])
                plt.yscale("log")
                plt.gca().tick_params(axis='x', labelsize=fontsize,which="both")
                plt.gca().tick_params(axis='y', labelsize=fontsize,which="both")
                plt.gca().annotate(planet+": "+molecule_str,xy=(-5,1),va="top",ha="left",fontsize=fontsize,color="black")

                # noise1 = copy(summed_wideRV[200:400,((64*3)//2+5):((64*3)//2+15),((19*3)//2-3):((19*3)//2+3)])
                # plt.figure(6)
                # plt.plot(np.nansum(Nvalid_hdRV,axis=(1,2))/np.nanmax(np.nansum(Nvalid_hdRV,axis=(1,2))))
                # plt.plot(summed_hdRV[:,(64*3)//2,(19*3)//2]/np.nanmax(summed_hdRV[:,(64*3)//2,(19*3)//2]))
                summed_wideRV = summed_wideRV/Nvalid_wideRV
                summed_hdRV = summed_hdRV/Nvalid_hdRV
                Nvalid_wideRV = np.sum(Nvalid_wideRV,axis=0)
                where_valid = np.where(Nvalid_wideRV>0.8*np.nanmax(Nvalid_wideRV))
                where_notvalid = np.where(Nvalid_wideRV<=0.8*np.nanmax(Nvalid_wideRV))
                noise1 = copy(summed_wideRV[200:400,:,:])
                noise1[:,((64*3)//2-5):((64*3)//2+5),((19*3)//2-5):((19*3)//2+5)] = np.nan
                noise1[:,where_notvalid[0],where_notvalid[1]] = np.nan
                sigma = np.nanstd(noise1)
                summed_wideRV = summed_wideRV/sigma
                summed_hdRV = summed_hdRV/sigma
                noise1 = noise1/sigma

                plt.figure(f1.number)
                plt.sca(ax_CCF_list[molid][plid])
                colids_choice = np.random.choice(where_valid[0],size = 50,replace=False)
                rowids_choice = np.random.choice(where_valid[1],size = 50,replace=False)
                for k,l in zip(colids_choice,rowids_choice):
                    plt.plot(planetRV,noise1[:,k,l],alpha=0.5,linestyle="--",linewidth=0.5,color="grey") #006699
                    # plt.plot(planetRV,noise2[:,k,l],alpha=0.5,linestyle="--",linewidth=1,color="cyan")
                plt.plot(planetRV,summed_wideRV[200:400,(64*3)//2,(19*3)//2],linestyle="-",linewidth=2,color=color)
                Nvalid_hdRV = np.sum(Nvalid_hdRV[400:800,:,:],axis=(1,2))
                where_validhd = np.where(Nvalid_hdRV>0.9*np.nanmax(Nvalid_hdRV))
                where_notvalidhd = np.where(Nvalid_hdRV<=0.9*np.nanmax(Nvalid_hdRV))
                # plt.plot(planetRV_hd[where_validhd],summed_hdRV[400:800,(64*3)//2,(19*3)//2][where_validhd],linestyle="--",linewidth=1,color="black") #"black","#ff9900","#006699","grey"
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                plt.xlim([-4000,4000])
                plt.xticks([-2000,0,2000,4000])
                if IFSfilter =="Kbb":
                    plt.ylim([-10,50])
                    plt.yticks([0,10,20,30,40,50])
                    plt.gca().annotate(planet+": "+molecule_str,xy=(-3750,45),va="top",ha="left",fontsize=fontsize,color="black")
                    plt.gca().annotate("S/N={0:0.1f}".format(summed_hdRV[(400*3)//2,(64*3)//2,(19*3)//2]),xy=(3750,45),va="top",ha="right",fontsize=fontsize,color="black")
                elif IFSfilter =="Hbb":
                    plt.ylim([-4,20])
                    plt.yticks([0,5,10,15,20])
                    plt.gca().annotate(planet+": "+molecule_str,xy=(-3750,18),va="top",ha="left",fontsize=fontsize,color="black")
                    plt.gca().annotate("S/N={0:0.1f}".format(summed_hdRV[(400*3)//2,(64*3)//2,(19*3)//2]),xy=(3750,18),va="top",ha="right",fontsize=fontsize,color="black")
                # plt.ylim([-2000,2000])
                # plt.ylim([-10,40])
                # plt.xticks([-1000,0,1000,2000])
                # plt.yticks([0,10,20,30,40])


                plt.figure(f4.number)
                plt.sca(ax_CCFsummary_list[plid])
                if plid != 2:
                    if IFSfilter == "Kbb":
                        plt.gca().annotate("HR 8799 "+planet,xy=(-900,45),va="top",ha="left",fontsize=fontsize,color=color)
                    elif IFSfilter == "Hbb":
                        plt.gca().annotate("HR 8799 "+planet,xy=(-900,18),va="top",ha="left",fontsize=fontsize,color=color)
                else:
                    plt.gca().annotate("HR 8799 "+planet,xy=(-900,13),va="top",ha="left",fontsize=fontsize,color=color)
                # croppedsummed_hdRV = summed_hdRV[400:800,(64*3)//2,(19*3)//2]
                # finitehdRV=np.where(np.isfinite(croppedsummed_hdRV))
                # croppedsummed_hdRV = croppedsummed_hdRV[finitehdRV]
                # croppedplanetRV_hd = planetRV_hd[finitehdRV]
                # cut1 = np.where(planetRV<=croppedplanetRV_hd[0])[0][-1]
                # cut2 = np.where(planetRV>=croppedplanetRV_hd[-1])[0][0]
                # # print(np.size(planetRV_hd))
                # # print(cut1,cut2)
                # # exit()
                # concaplanetRV2plot = np.concatenate([planetRV[0:cut1+1],croppedplanetRV_hd,planetRV[cut2::]])
                # concaCCF2plot = np.concatenate([summed_wideRV[200:400,(64*3)//2,(19*3)//2][0:cut1+1],
                #                                 croppedsummed_hdRV,
                #                                 summed_wideRV[200:400,(64*3)//2,(19*3)//2][cut2::]])
                # if molid == 0:
                #     plt.plot(concaplanetRV2plot,concaCCF2plot,linestyle=linestyle_list[molid],linewidth=3,color=color,label=molecule_str+": S/N={0:0.1f}".format(summed_hdRV[(400*3)//2,(64*3)//2,(19*3)//2]))
                # else:
                #     plt.plot(concaplanetRV2plot,concaCCF2plot,linestyle=linestyle_list[molid],linewidth=2,color="black",label=molecule_str+": S/N={0:0.1f}".format(summed_hdRV[(400*3)//2,(64*3)//2,(19*3)//2]))
                if molid == 0:
                    plt.plot(planetRV,summed_wideRV[200:400,(64*3)//2,(19*3)//2],linestyle=linestyle_list[molid],linewidth=3,color=color,label=molecule_str+": S/N={0:0.1f}".format(summed_hdRV[(400*3)//2,(64*3)//2,(19*3)//2]))
                else:
                    plt.plot(planetRV,summed_wideRV[200:400,(64*3)//2,(19*3)//2],linestyle=linestyle_list[molid],linewidth=2,color="black",label=molecule_str+": S/N={0:0.1f}".format(summed_hdRV[(400*3)//2,(64*3)//2,(19*3)//2]))
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                plt.xlim([-1000,1000])
                if IFSfilter == "Kbb":
                    plt.ylim([-10,50])
                    plt.yticks([0,10,20,30,40,50])
                elif IFSfilter == "Hbb":
                    plt.ylim([-5,20])
                    plt.yticks([0,5,10,15,20])
                # if planet == "b" or planet == "c":
                # elif planet == "d":
                #     plt.ylim([-3,15])
                #     plt.yticks([0,5,10,15])
                plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#
                # plt.show()

        plt.figure(f4.number)
        plt.sca(ax_CCFsummary_list[-1])
        plt.ylabel("S/N",fontsize=15)
        plt.xlabel(r"$\Delta V$ (km/s)",fontsize=fontsize)
        plt.xticks([-1000,-500,0,500,1000])
        # plt.xticks([-2000,-1000,0,1000,2000])
        # plt.yticks([-10,0,10,20,30,40])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        f4.subplots_adjust(wspace=0,hspace=0)

        print("Saving "+os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCFsummary_kl{0}.png".format(resnumbasis)))
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCFsummary_kl{0}.png".format(resnumbasis)),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCFsummary_kl{0}.pdf".format(resnumbasis)),bbox_inches='tight')
        plt.close(f4.number)

        plt.figure(f1.number)
        plt.sca(ax_CCF_list[-1][0])
        plt.ylabel("S/N",fontsize=15)
        plt.xlabel(r"$\Delta V$ (km/s)",fontsize=fontsize)
        plt.xticks([-4000,-2000,0,2000,4000])
        if IFSfilter =="Kbb":
            plt.yticks([-10,0,10,20,30,40,50])
        elif IFSfilter =="Hbb":
            plt.yticks([-5,0,5,10,15,20])
        # plt.xticks([-2000,-1000,0,1000,2000])
        # plt.yticks([-10,0,10,20,30,40])
        plt.gca().tick_params(axis='x', labelsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)
        f1.subplots_adjust(wspace=0,hspace=0)

        print("Saving "+os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF_kl{0}.png".format(resnumbasis)))
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF_kl{0}.png".format(resnumbasis)),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF_kl{0}.pdf".format(resnumbasis)),bbox_inches='tight')
        plt.close(f1.number)


        plt.figure(f2.number)
        for m in range(len(planet_list)):
            plt.sca(ax_histo_list[0][m])
            plt.legend(loc="lower center",frameon=True,fontsize=fontsize)
        plt.gca().tick_params(axis='x', labelsize=fontsize,which="both")
        plt.gca().tick_params(axis='y', labelsize=fontsize,which="both")
        plt.xticks([-6,-4,-2,0,2,4,6])
        plt.yticks([1e-6,1e-4,1e-2,1e-0])
        f2.subplots_adjust(wspace=0,hspace=0)

        print("Saving "+os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF_histo_kl{0}.png".format(resnumbasis)))
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF_histo_kl{0}.png".format(resnumbasis)),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,"HR_8799_"+IFSfilter+"_CCF_histo_kl{0}.pdf".format(resnumbasis)),bbox_inches='tight')
        plt.close(f2.number)
        plt.show()
        # exit()

    exit()
