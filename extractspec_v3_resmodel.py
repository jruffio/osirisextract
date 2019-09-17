__author__ = 'jruffio'

import glob
import os
import csv
import astropy.io.fits as pyfits
import numpy as np
import multiprocessing as mp
from reduce_HPFonly_diagcov import convolve_spectrum
from reduce_HPFonly_diagcov import LPFvsHPF
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from copy import copy

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
    date = "*"
    cutoff = 40
    fontsize = 12

    if 1:
        planetcolor_list = ["#0099cc","#ff9900","#6600ff"]
        # for planet,planetcolor in zip(["b","c","d"],planetcolor_list):
            # for IFSfilter in ["Kbb","Hbb"]:
        for planet,planetcolor in zip(["c"],["#0099cc"]):
            for IFSfilter in ["Kbb"]:

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

                ## file specific info
                fileinfos_filename = os.path.join("/data/osiris_data/HR_8799_"+planet,"fileinfos_Kbb_jb.csv")
                with open(fileinfos_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=';')
                    list_table = list(csv_reader)
                    colnames = list_table[0]
                    N_col = len(colnames)
                    list_data = list_table[1::]
                    N_lines =  len(list_data)

                myspec_list = []
                myspecwvs_list = []
                myspec_std_list = []
                bias_myspec_list = []

                filename_id = colnames.index("filename")
                kcen_id = colnames.index("kcen")
                lcen_id = colnames.index("lcen")
                baryrv_id = colnames.index("barycenter rv")

                for resnumbasis in np.array([0,1,3,10]):
                    for fileitem in list_data:
                        filename = fileitem[filename_id]
                        if resnumbasis == 0:
                            data_filename = filename.replace("reduced_jb","reduced_jb/sherlock/20190911_resmodel").replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_rescalc_estispec.fits")
                        else:
                            data_filename = filename.replace("reduced_jb","reduced_jb/sherlock/20190911_resmodel").replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl{0}_estispec.fits".format(resnumbasis))
                        if len(glob.glob(data_filename)) == 0:
                            continue
                        print(data_filename)


                        with pyfits.open(data_filename) as hdulist:
                            esti_spec_arr = hdulist[0].data
                        print(esti_spec_arr.shape)

                        if fileitem[kcen_id] == "nan":
                            continue
                        plcen_k,plcen_l = int(fileitem[kcen_id]),int(fileitem[lcen_id])
                        host_bary_rv = -float(fileitem[baryrv_id])/1000
                        c_kms = 299792.458
                        myspecwvs_list.append(copy(esti_spec_arr[0,:,plcen_k,plcen_l])*(1-host_bary_rv/c_kms) )
                        myspec_list.append(copy(esti_spec_arr[1,:,plcen_k,plcen_l]))


                        esti_spec_arr[:,:,plcen_k-5:plcen_k+5,plcen_l-5:plcen_l+5] = np.nan
                        bias_myspec = np.nanmean(esti_spec_arr[1,:,:,:],axis=(1,2))
                        std_myspec = np.nanstd(esti_spec_arr[1,:,:,:],axis=(1,2))

                        bias_myspec_list.append(bias_myspec)
                        myspec_std_list.append(std_myspec)



                    myspecwvs_conca = np.concatenate(myspecwvs_list)
                    myspec_conca = np.concatenate(myspec_list)
                    myspec_std_conca = np.concatenate(myspec_std_list)
                    myspec_bias_conca = np.concatenate(bias_myspec_list)
                    nbins = nl
                    binedges = np.linspace(wvs[0]-dwv/4,wvs[-1]+dwv/4,nbins+1,endpoint=True)
                    bincenter = np.linspace(wvs[0],wvs[-1],nbins,endpoint=True)
                    digitized = np.digitize(myspecwvs_conca,binedges)-1
                    final_spec = np.zeros(nbins)+np.nan
                    final_spec_biascorr = np.zeros(nbins)+np.nan
                    final_spec_std = np.zeros(nbins)+np.nan
                    for k in np.arange(2,nbins):
                        where_digit = np.where((k==digitized)*(np.isfinite(myspec_conca)))
                        if np.size(where_digit[0]) > 0.2*len(myspecwvs_list):
                            final_spec[k]=np.nansum((myspec_conca[where_digit])/myspec_std_conca[where_digit]**2)/np.nansum(1/myspec_std_conca[where_digit]**2)
                            final_spec_biascorr[k]=np.nansum((myspec_conca[where_digit]-myspec_bias_conca[where_digit])/myspec_std_conca[where_digit]**2)/np.nansum(1/myspec_std_conca[where_digit]**2)
                            final_spec_std[k]=np.sqrt(1/np.nansum(1/myspec_std_conca[where_digit]**2))
                        else:
                            final_spec[k]=np.nan
                            final_spec_biascorr[k]=np.nan
                            final_spec_std[k]=np.nan


                    plt.plot(bincenter,final_spec/np.nanstd(final_spec),linestyle="--",label="{0}: final_spec".format(resnumbasis))
                    plt.plot(bincenter,final_spec_biascorr/np.nanstd(final_spec_biascorr),linestyle="-",label="{0} final_spec - bias".format(resnumbasis))



                osiris_data_dir = "/data/osiris_data"
                molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
                mol_linestyle_list = ["-","-","-"]#["-","--",":"]
                # for molid,(molecule,mol_linestyle) in enumerate(zip(["CO","H2O","CH4"],mol_linestyle_list)):
                for molid,(molecule,mol_linestyle) in enumerate(zip(["CH4"],mol_linestyle_list)):
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
                        planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)

                    molec_spec = planet_spec_func(bincenter)
                    from reduce_HPFonly_diagcov import LPFvsHPF
                    HPF_molec_spec = LPFvsHPF(molec_spec,cutoff)[1]

                    plt.plot(bincenter,HPF_molec_spec/np.nanstd(HPF_molec_spec),color="black",linestyle="--",label=molecule)

                plt.legend()
                plt.show()
                exit()
        if 1:
            if 1:
                if 1:
                    exit()

                    with pyfits.open(data_filename) as hdulist:
                        data = hdulist[0].data
                    with pyfits.open(data_filename.replace("data.fits","HPFdata.fits")) as hdulist:
                        HPFdata = hdulist[0].data
                    with pyfits.open(data_filename.replace("data.fits","transmission.fits")) as hdulist:
                        transmission = hdulist[0].data
                    with pyfits.open(data_filename.replace("data.fits","wvs.fits")) as hdulist:
                        datawvs = hdulist[0].data
                    with pyfits.open(data_filename.replace("data.fits","planetwvs.fits")) as hdulist:
                        planetwvs = hdulist[0].data
                    with pyfits.open(data_filename.replace("data.fits","star.fits")) as hdulist:
                        star = hdulist[0].data
                    with pyfits.open(data_filename.replace("data.fits","PSF.fits")) as hdulist:
                        PSF = hdulist[0].data
                    with pyfits.open(data_filename.replace("data.fits","sigmas.fits")) as hdulist:
                        sigmas = hdulist[0].data
                    with pyfits.open(data_filename.replace("data.fits","badpix.fits")) as hdulist:
                        badpix = hdulist[0].data
                    with pyfits.open(data_filename.replace("data.fits","HPFbkg_model.fits")) as hdulist:
                        HPFbkg_model = hdulist[0].data

                    # plt.imshow(np.nansum(data,axis=2),interpolation="nearest")
                    # plt.show()
                    w=2
                    where_bad_data = np.where(np.isnan(badpix))
                    where_finite_data = np.where(np.isfinite(badpix))
                    where_finite_raveldata = np.where(np.isfinite(np.ravel(badpix)))
                    HPFmodel_H0 = np.reshape(HPFbkg_model,((2*w+1)**2,(2*w+1)**2*nl)).transpose()[where_finite_raveldata[0],:]
                    where_valid_parameters = np.where(np.sum(np.abs(HPFmodel_H0),axis=0)!=0)
                    HPFmodel_H0 = HPFmodel_H0[:,where_valid_parameters[0]]
                    ravelHPFdata = np.ravel(HPFdata)[where_finite_raveldata]
                    HPFmodel_H0[np.where(np.isnan(HPFmodel_H0))]=0
                    sigmas_ravel = np.ravel(sigmas)[where_finite_raveldata]

                    HPFparas_H0,HPFchi2_H0,rank,s = np.linalg.lstsq(HPFmodel_H0/sigmas_ravel[:,None],ravelHPFdata/sigmas_ravel,rcond=None)

                    data_model_H0 = np.dot(HPFmodel_H0,HPFparas_H0)
                    ravelresiduals_H0 = ravelHPFdata-data_model_H0

                    canvas = np.zeros((5,5,nl))+np.nan
                    canvas[where_finite_data] = ravelresiduals_H0

                    PSF[where_bad_data] = np.nan
                    myspec = np.nansum(canvas*PSF,axis=(0,1))/np.nansum(PSF,axis=(0,1))/transmission

                    esti_spec_arr[:,:,plcen_k-5:plcen_k+5,plcen_l-5:plcen_l+5] = np.nan
                    bias_myspec = np.nanmean(esti_spec_arr[1,:,:,:],axis=(1,2))
                    std_myspec = np.nanstd(esti_spec_arr[1,:,:,:],axis=(1,2))

                    bias_myspec_list.append(bias_myspec)
                    myspec_std_list.append(std_myspec)
                    # bias_myspec_wvs = np.nanmean(esti_spec_arr[0,:,:,:],axis=(1,2))
                    # plt.plot(np.nanmean(planetwvs,axis=(0,1)),myspec,"r")
                    # plt.plot(bias_myspec_wvs,bias_myspec,"b")
                    # plt.fill_between(bias_myspec_wvs,bias_myspec-std_myspec,bias_myspec+std_myspec,color="cyan")
                    # plt.show()

                    esti_spec_arr_list.append(esti_spec_arr[1,:,:]-bias_myspec[:,None,None])
                    myspec_list.append(myspec-bias_myspec)
                    bias_myspec_list.append(bias_myspec)
                    myspec_std_list.append(std_myspec)
                    myspecwvs_list.append(np.nanmean(planetwvs,axis=(0,1)))

                mymeanspec = np.nansum((np.array(myspec_list)/np.array(myspec_std_list)**2),axis=0)/np.nansum((1/np.array(myspec_std_list)**2),axis=0)
                myspecwvs_conca = np.concatenate(myspecwvs_list)
                myspec_conca = np.concatenate(myspec_list)
                myspec_std_conca = np.concatenate(myspec_std_list)
                myspec_bias_conca = np.concatenate(bias_myspec_list)
                nbins = nl
                binedges = np.linspace(wvs[0]-dwv/4,wvs[-1]+dwv/4,nbins+1,endpoint=True)
                bincenter = np.linspace(wvs[0],wvs[-1],nbins,endpoint=True)
                digitized = np.digitize(myspecwvs_conca,binedges)-1
                final_spec = np.zeros(nbins)+np.nan
                final_spec_std = np.zeros(nbins)+np.nan
                for k in np.arange(2,nbins):
                    where_digit = np.where((k==digitized)*(np.isfinite(myspec_conca)))
                    if np.size(where_digit[0]) > 0.2*len(myspecwvs_list):
                        final_spec[k]=np.nansum((myspec_conca[where_digit]-myspec_bias_conca[where_digit])/myspec_std_conca[where_digit]**2)/np.nansum(1/myspec_std_conca[where_digit]**2)
                        final_spec_std[k]=np.sqrt(1/np.nansum(1/myspec_std_conca[where_digit]**2))
                    else:
                        final_spec[k]=np.nan
                        final_spec_std[k]=np.nan

                f1 = plt.figure(1,figsize=(12,9))
                f2,ax_CCF_list = plt.subplots(1,3,sharey="row",sharex="col",figsize=(12,3))
                osiris_data_dir = "/data/osiris_data"
                molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
                mol_linestyle_list = ["-","-","-"]#["-","--",":"]
                for molid,(molecule,mol_linestyle) in enumerate(zip(["CO","H2O","CH4"],mol_linestyle_list)):
                # for molid,(molecule,mol_linestyle) in enumerate(zip(["CH4"],mol_linestyle_list)):
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
                        planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)

                    molec_spec = planet_spec_func(bincenter)
                    from reduce_HPFonly_diagcov import LPFvsHPF
                    HPF_molec_spec = LPFvsHPF(molec_spec,cutoff)[1]

                    plt.figure(f1.number)
                    if IFSfilter == "Kbb":
                        if molecule == "CO":
                            plt.subplot(4,1,molid+1)
                            plt.xlim([2.28,2.38])
                            plt.gca().text(2.28,0.01,planet+": "+molecule,ha="left",va="center",rotation=0,size=fontsize*1.25,color="black")
                        elif molecule == "CH4":
                            plt.subplot(4,1,molid+1)
                            plt.xlim([2.2,2.38])
                            plt.gca().text(2.2,0.01,planet+": "+molecule,ha="left",va="center",rotation=0,size=fontsize*1.25,color="black")
                        elif molecule == "H2O":
                            plt.subplot(4,2,(molid*2)+1)
                            plt.xlim([1.95,2.1])
                            plt.gca().text(1.95,0.01,planet+": "+molecule,ha="left",va="center",rotation=0,size=fontsize*1.25,color="black")
                            plt.plot(bincenter,final_spec,color=planetcolor,label="Data")
                            plt.plot(bincenter,HPF_molec_spec*np.nansum(final_spec*HPF_molec_spec/final_spec_std**2)/np.nansum(HPF_molec_spec*HPF_molec_spec/final_spec_std**2),color="black",linestyle=mol_linestyle,label=molecule+" model (best fit)")
                            plt.ylim([-0.01,0.02])
                            plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
                            plt.gca().tick_params(axis='x', labelsize=fontsize)
                            plt.gca().tick_params(axis='y', labelsize=fontsize)
                            plt.tick_params(axis="y",which="both",labelleft=False,right=False,left=False)
                            plt.gca().spines["right"].set_visible(False)
                            plt.gca().spines["left"].set_visible(False)
                            plt.gca().spines["top"].set_visible(False)

                            plt.subplot(4,2,(molid*2)+2)
                            # plt.subplot(4,1,molid+1)
                            plt.xlim([2.25,2.38])
                            plt.gca().text(2.25,0.01,planet+": "+molecule,ha="left",va="center",rotation=0,size=fontsize*1.25,color="black")
                        # plt.xlim([2.15,2.4])
                    elif IFSfilter == "Hbb":
                        if molecule == "CO":
                            plt.subplot(4,1,molid+1)
                            plt.xlim([1.54,1.62])
                            plt.gca().text(1.54,0.01,planet+": "+molecule,ha="left",va="center",rotation=0,size=fontsize*1.25,color="black")
                        elif molecule == "CH4":
                            plt.subplot(4,1,molid+1)
                            plt.xlim([1.6,1.72])
                            plt.gca().text(1.6,0.01,planet+": "+molecule,ha="left",va="center",rotation=0,size=fontsize*1.25,color="black")
                        elif molecule == "H2O":
                            plt.subplot(4,2,(molid*2)+1)
                            plt.xlim([1.47,1.55])
                            plt.gca().text(1.47,0.01,planet+": "+molecule,ha="left",va="center",rotation=0,size=fontsize*1.25,color="black")
                            plt.plot(bincenter,final_spec,color=planetcolor,label="Data")
                            plt.plot(bincenter,HPF_molec_spec*np.nansum(final_spec*HPF_molec_spec/final_spec_std**2)/np.nansum(HPF_molec_spec*HPF_molec_spec/final_spec_std**2),color="black",linestyle=mol_linestyle,label=molecule+" model (best fit)")
                            plt.ylim([-0.01,0.02])
                            plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
                            plt.gca().tick_params(axis='x', labelsize=fontsize)
                            plt.gca().tick_params(axis='y', labelsize=fontsize)
                            plt.tick_params(axis="y",which="both",labelleft=False,right=False,left=False)
                            plt.gca().spines["right"].set_visible(False)
                            plt.gca().spines["left"].set_visible(False)
                            plt.gca().spines["top"].set_visible(False)

                            plt.subplot(4,2,(molid*2)+2)
                            plt.xlim([1.72,1.8])
                            plt.gca().text(1.72,0.01,planet+": "+molecule,ha="left",va="center",rotation=0,size=fontsize*1.25,color="black")
                    plt.fill_between(bincenter,final_spec-final_spec_std,final_spec+final_spec_std,color=planetcolor,alpha=0.5)
                    plt.plot(bincenter,final_spec,color=planetcolor,label="Data")
                    if molecule == "CH4" or (molecule == "CO" and IFSfilter =="Hbb"):
                        plt.plot(bincenter,HPF_molec_spec/np.nanmax(np.abs(HPF_molec_spec))*0.01,color="black",linestyle="--",label=molecule+" model (normalized)")
                    else:
                        plt.plot(bincenter,HPF_molec_spec*np.nansum(final_spec*HPF_molec_spec/final_spec_std**2)/np.nansum(HPF_molec_spec*HPF_molec_spec/final_spec_std**2),color="black",linestyle=mol_linestyle,label=molecule+" model (best fit)")
                    plt.ylim([-0.01,0.02])
                    plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
                    plt.gca().tick_params(axis='x', labelsize=fontsize)
                    plt.gca().tick_params(axis='y', labelsize=fontsize)
                    plt.tick_params(axis="y",which="both",labelleft=False,right=False,left=False)
                    plt.gca().spines["right"].set_visible(False)
                    plt.gca().spines["left"].set_visible(False)
                    plt.gca().spines["top"].set_visible(False)
                    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)
                    # plt.show()

                    plt.figure(f2.number)
                    plt.sca(ax_CCF_list[molid])
                    c_kms = 299792.458
                    rvshifts = np.arange(-100*dprv,100*dprv,dprv)
                    argnullrv = np.argmin(np.abs(rvshifts))
                    ccf = np.zeros(rvshifts.shape)
                    ccf_noise_rows = np.arange(10,54)
                    ccf_noise_cols = np.arange(4,15)
                    # ccf_noise_rows = np.arange(10,20)
                    # ccf_noise_cols = np.arange(4,15)
                    ccf_noise = np.zeros((len(ccf_noise_rows),len(ccf_noise_cols),np.size(rvshifts)))
                    for plrv_id in range(np.size(rvshifts)):
                        wvs4planet_model = bincenter*(1-(rvshifts[plrv_id])/c_kms)
                        myLPFspec2,myHPFspec2 = LPFvsHPF(planet_spec_func(wvs4planet_model),cutoff)
                        myHPFspec2 = myHPFspec2/np.sqrt(np.nansum(myHPFspec2**2))
                        ccf[plrv_id] = np.nansum(final_spec*myHPFspec2/final_spec_std**2)/np.sqrt(np.nansum(myHPFspec2**2/final_spec_std**2))
                        for rowid,row in enumerate(ccf_noise_rows):
                            for colid,col in enumerate(ccf_noise_cols):
                                sample_myspec_list = []
                                for esti_spec_arr in esti_spec_arr_list:
                                    sample_myspec_list.append(esti_spec_arr[:,row,col])
                                sample_myspec_arr = np.array(sample_myspec_list)
                                myspec_std_arr = copy(np.array(myspec_std_list))
                                myspec_std_arr[np.where(np.isnan(sample_myspec_arr))] = np.nan
                                sample_myspec = np.nansum((sample_myspec_arr/myspec_std_arr**2),axis=0)/np.nansum((1/myspec_std_arr**2),axis=0)
                                ccf_noise[rowid,colid,plrv_id] = np.nansum(sample_myspec*myHPFspec2/final_spec_std**2)/np.sqrt(np.nansum(myHPFspec2**2/final_spec_std**2))

                    ccf_noise_norma = np.nanstd(ccf_noise,axis=(0,1))
                    ccf_norma = np.nanstd(ccf_noise/ccf_noise_norma[None,None,:])
                    ccf_noise = ccf_noise/ccf_noise_norma[None,None,:]/ccf_norma
                    # ccf_norma = 1#np.nanmax(ccf)
                    ccf_snr = ccf/ccf_noise_norma/ccf_norma
                    nullrvarg = np.argmin(np.abs(rvshifts))
                    plt.plot(rvshifts,ccf_noise[0,0,:],color="grey",linewidth=1,linestyle="--",label="Noise samples",alpha=0.1)
                    for rowid in np.arange(0,len(ccf_noise_rows),2):
                        for colid in np.arange(0,len(ccf_noise_cols),2):
                            plt.plot(rvshifts,ccf_noise[rowid,colid,:],color="grey",linewidth=1,linestyle="--",alpha=0.1)
                    plt.plot(rvshifts,ccf_snr,color=planetcolor,linewidth=2,linestyle=mol_linestyle,label="Planet signal")
                    if IFSfilter == "Kbb":
                        plt.gca().text(-3500,16,planet+": "+molecule,ha="left",va="bottom",rotation=0,size=fontsize,color="black")
                        plt.gca().annotate("S/N={0:0.1f}".format(ccf_snr[nullrvarg]),xy=(3750,16),va="bottom",ha="right",fontsize=fontsize,color="black")
                        # plt.ylim([-0.5,1.1])
                        plt.ylim([-5,20])
                    elif IFSfilter == "Hbb":
                        plt.gca().text(-3500,8,planet+": "+molecule,ha="left",va="bottom",rotation=0,size=fontsize,color="black")
                        plt.gca().annotate("S/N={0:0.1f}".format(ccf_snr[nullrvarg]),xy=(3750,8),va="bottom",ha="right",fontsize=fontsize,color="black")
                        plt.ylim([-2,10])
                    plt.xlabel(r"$\Delta V$ (km/s)",fontsize=fontsize)
                    plt.xticks([-2000,0,2000,4000])
                    plt.gca().tick_params(axis='x', labelsize=fontsize)
                    plt.gca().tick_params(axis='y', labelsize=fontsize)
                    # plt.show()
                plt.figure(f2.number)
                plt.sca(ax_CCF_list[0])
                plt.ylabel("$S/N$",fontsize=15)
                plt.xticks([-4000,-2000,0,2000,4000])
                plt.sca(ax_CCF_list[2])
                plt.legend(loc="center right",frameon=True,fontsize=fontsize)
                # plt.yticks([-0.5,0,0.5,1])
                # plt.show()

                plt.figure(f2.number)
                plt.tight_layout()
                f2.subplots_adjust(wspace=0)
                print("Saving "+os.path.join(out_pngs,"HR_8799_"+planet,"HR_8799_"+planet+"_"+IFSfilter+"_estispec_ccf.png"))
                plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR_8799_"+planet+"_"+IFSfilter+"_estispec_ccf.png"),bbox_inches='tight')
                plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR_8799_"+planet+"_"+IFSfilter+"_estispec_ccf.pdf"),bbox_inches='tight')
                plt.close(f2.number)

                plt.figure(f1.number)
                plt.tight_layout()
                print("Saving "+os.path.join(out_pngs,"HR_8799_"+planet,"HR_8799_"+planet+"_"+IFSfilter+"_estispec.png"))
                plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR_8799_"+planet+"_"+IFSfilter+"_estispec.png"),bbox_inches='tight')
                plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR_8799_"+planet+"_"+IFSfilter+"_estispec.pdf"),bbox_inches='tight')
                # plt.show()
                plt.close(f1.number)
            # print(k,np.size(where_digit[0]),bincenter[k],final_spec[k],final_spec_std[k])
        exit()
        # # final_spec /= np.std(final_spec)
        # # final_spec_std /= np.std(final_spec)
        # # print(myspecwvs_conca.shape)
        # # for tmpwvs,tmpspec in zip(myspecwvs_list,myspec_list):
        # # plt.scatter(myspecwvs_conca,myspec_conca,s=1,alpha=0.3,color="grey")
        # # plt.fill_between(bincenter,final_spec-final_spec_std,final_spec+final_spec_std,color="pink",alpha=0.5)
        # # plt.plot(bincenter,final_spec+final_spec_std,linestyle="--",color="blue",alpha=0.5)
        #
        #
        # template_func_list = []
        # template_name_list = []
        # # plt.show()
        #
        # plt.subplot(2,1,2)
        # c_kms = 299792.458
        # rvshifts = np.arange(-100*dprv,100*dprv,dprv)
        # argnullrv = np.argmin(np.abs(rvshifts))
        # ccf = np.zeros(rvshifts.shape)
        # for plrv_id in range(np.size(rvshifts)):
        #     wvs4planet_model = bincenter*(1-(rvshifts[plrv_id])/c_kms)
        #     myLPFspec2,myHPFspec2 = LPFvsHPF(template_func_list[0](wvs4planet_model),cutoff)
        #     myHPFspec2 = myHPFspec2/np.sqrt(np.nansum(myHPFspec2**2))
        #     ccf[plrv_id] = np.nansum(final_spec*myHPFspec2/final_spec_std**2)/np.nansum(myHPFspec2**2/final_spec_std**2)
        #
        #     # wvs4planet_model = wvs*(1-(rvshifts[plrv_id])/c_kms)
        #     # myLPFspec2,myHPFspec2 = LPFvsHPF(template_func_list[0](wvs4planet_model),cutoff)
        #     # myHPFspec2 = myHPFspec2/np.sqrt(np.nansum(myHPFspec2**2))
        #     # ccf[plrv_id] = np.nansum(myspec*myHPFspec2)
        #
        # plt.plot(rvshifts,ccf,color="black",linestyle="-",linewidth=2)#"#ff9900"
        # plt.plot([-4000,4000],[0,0],color="black",linewidth=0.5)
        # plt.gca().annotate("JB spec"+" vs. "+molecule,xy=(-1750,1),va="top",ha="left",fontsize=fontsize,color="black")
        # plt.gca().tick_params(axis='x', labelsize=fontsize)
        # plt.gca().tick_params(axis='y', labelsize=fontsize)
        # plt.xlim([-4000,4000])
        # # plt.ylim([-0.5,1.2])
        # plt.xlabel(r"$\Delta V$ (km/s)",fontsize=fontsize)
        #
        # plt.show()
        # exit()
        #
        #     # HPFmodel_H0 = np.concatenate(HPFmodelH0_list,axis=1)
        #
        #     # data_div = PSF*data/(transmission*star)[None,None,:]
        #     # data_div[np.where(np.isnan(badpix))] = np.nan
        #     # from reduce_HPFonly_diagcov import LPFvsHPF
        #     # HPF_data_div = LPFvsHPF(np.nansum(data_div,axis=(0,1)),40)[1]
        #     #
        #     # plt.plot(wvs,HPF_data_div/np.nanstd(HPF_data_div))
        #     # HPFtrans = LPFvsHPF(transmission,40)[1]
        #     # plt.plot(wvs,HPFtrans/np.nanstd(HPFtrans))
        #     # plt.show()
        #     # exit()
        #
        #
        #
        #     # # digitize_wvs = np.digitize(planetwvs,bin_edges,right=True)
        #     # # digitize_list.append(digitize_wvs)
        #     #
        #     # # tmp = PSF*HPFdata/np.nansum(PSF_transmission)
        #     # myspec = np.array([np.nansum(HPFdata[np.where(digitize_wvs==k)]) for k in range(nl)])
        #     # mytrans = np.array([np.nansum((PSF*transmission)[np.where(digitize_wvs==k)]) for k in range(nl)])
        #     # final_spec += myspec
        #     # final_spec2 += mytrans
        #
        # # import matplotlib.pyplot as plt
        # # plt.figure(1)
        # # plt.plot(wvs,final_spec)
        # # plt.figure(2)
        # # plt.plot(wvs,final_spec2)
        # # plt.show()
        #
        #     # print(HPFdata.shape)
        #     # print(datawvs.shape)
        #     # print(HPFmodel_H0.shape)
        #     # print(transmission.shape)
        #

    if 0:
        planetcolor_list = ["#0099cc","#ff9900","#6600ff"]
        # for planet,planetcolor in zip(["b","c","d"],planetcolor_list):
        #     for IFSfilter in ["Kbb","Hbb"]:
        for planet,planetcolor in zip(["b"],planetcolor_list):
            for IFSfilter in ["Kbb"]:
                # IFSfilter= "Hbb"
                numthreads = 10
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

                f1,ax_CCF_list = plt.subplots(1,3,sharey="row",sharex="col",figsize=(12,5))
                for colordate,labeldate,filenamefilter in zip(["#0099cc","#ff9900","#6600ff","black"],["201611??","20180722","20130725","20100712"],["s1611*"+IFSfilter+"*_data.fits","s1807*"+IFSfilter+"*_data.fits","s130725*"+IFSfilter+"*_data.fits","s100712*"+IFSfilter+"*_data.fits"]):
                # for colordate,labeldate,filenamefilter in zip(["#0099cc"],["all"],["s*"+IFSfilter+"*_data.fits"]):
                    inputdir = "/data/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb/sherlock/20190508_models_off/"

                    data_filelist = glob.glob(os.path.join(inputdir,filenamefilter))
                    data_filelist.sort()
                    print(data_filelist)
                    print(len(data_filelist))
                    if len(data_filelist) == 0:
                        continue

                    myspec_list = []
                    myspecwvs_list = []
                    for data_filename in data_filelist:
                        # print(data_filename)

                        with pyfits.open(data_filename) as hdulist:
                            data = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","HPFdata.fits")) as hdulist:
                            HPFdata = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","transmission.fits")) as hdulist:
                            transmission = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","wvs.fits")) as hdulist:
                            datawvs = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","planetwvs.fits")) as hdulist:
                            planetwvs = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","star.fits")) as hdulist:
                            star = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","PSF.fits")) as hdulist:
                            PSF = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","sigmas.fits")) as hdulist:
                            sigmas = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","badpix.fits")) as hdulist:
                            badpix = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","HPFbkg_model.fits")) as hdulist:
                            HPFbkg_model = hdulist[0].data

                        # plt.imshow(np.nansum(data,axis=2),interpolation="nearest")
                        # plt.show()
                        w=2
                        where_bad_data = np.where(np.isnan(badpix))
                        where_finite_data = np.where(np.isfinite(badpix))
                        where_finite_raveldata = np.where(np.isfinite(np.ravel(badpix)))
                        HPFmodel_H0 = np.reshape(HPFbkg_model,((2*w+1)**2,(2*w+1)**2*nl)).transpose()[where_finite_raveldata[0],:]
                        where_valid_parameters = np.where(np.sum(np.abs(HPFmodel_H0),axis=0)!=0)
                        HPFmodel_H0 = HPFmodel_H0[:,where_valid_parameters[0]]
                        ravelHPFdata = np.ravel(HPFdata)[where_finite_raveldata]
                        HPFmodel_H0[np.where(np.isnan(HPFmodel_H0))]=0
                        sigmas_ravel = np.ravel(sigmas)[where_finite_raveldata]

                        HPFparas_H0,HPFchi2_H0,rank,s = np.linalg.lstsq(HPFmodel_H0/sigmas_ravel[:,None],ravelHPFdata/sigmas_ravel,rcond=None)

                        data_model_H0 = np.dot(HPFmodel_H0,HPFparas_H0)
                        ravelresiduals_H0 = ravelHPFdata-data_model_H0

                        canvas = np.zeros((5,5,nl))+np.nan
                        canvas[where_finite_data] = ravelresiduals_H0

                        PSF[where_bad_data] = np.nan
                        myspec = np.nansum(canvas*PSF,axis=(0,1))/np.nansum(PSF,axis=(0,1))/transmission
                        myspec_list.append(myspec)
                        myspecwvs_list.append(np.nanmean(planetwvs,axis=(0,1)))
                    # exit()
                    mymeanspec = np.mean(np.array(myspec_list),axis=0)
                    myspecwvs_conca = np.concatenate(myspecwvs_list)
                    myspec_conca = np.concatenate(myspec_list)
                    nbins = nl
                    binedges = np.linspace(wvs[0]-dwv/4,wvs[-1]+dwv/4,nbins+1,endpoint=True)
                    bincenter = np.linspace(wvs[0],wvs[-1],nbins,endpoint=True)
                    digitized = np.digitize(myspecwvs_conca,binedges)-1
                    final_spec_noise = np.zeros(nbins)+np.nan
                    final_spec_noise_std = np.zeros(nbins)+np.nan
                    for k in np.arange(2,nbins):
                        where_digit = np.where((k==digitized)*(np.isfinite(myspec_conca)))
                        if np.size(where_digit[0]) > 0.2*len(myspecwvs_list):
                            final_spec_noise[k]=np.nanmean(myspec_conca[where_digit])
                            final_spec_noise_std[k]=np.nanstd(myspec_conca[where_digit]/np.sqrt(np.size(where_digit[0])))
                        else:
                            final_spec_noise[k]=np.nan
                            final_spec_noise_std[k]=np.nan





                    inputdir = "/data/osiris_data/HR_8799_"+planet+"/20"+date+"/reduced_jb/sherlock/20190508_models2/"

                    data_filelist = glob.glob(os.path.join(inputdir,filenamefilter))
                    data_filelist.sort()
                    print(data_filelist)
                    print(len(data_filelist))
                    if len(data_filelist) == 0:
                        continue

                    myspec_list = []
                    myspecwvs_list = []
                    for data_filename in data_filelist:
                        print(data_filename)
                        with pyfits.open(data_filename) as hdulist:
                            data = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","HPFdata.fits")) as hdulist:
                            HPFdata = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","transmission.fits")) as hdulist:
                            transmission = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","wvs.fits")) as hdulist:
                            datawvs = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","planetwvs.fits")) as hdulist:
                            planetwvs = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","star.fits")) as hdulist:
                            star = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","PSF.fits")) as hdulist:
                            PSF = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","sigmas.fits")) as hdulist:
                            sigmas = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","badpix.fits")) as hdulist:
                            badpix = hdulist[0].data
                        with pyfits.open(data_filename.replace("data.fits","HPFbkg_model.fits")) as hdulist:
                            HPFbkg_model = hdulist[0].data

                        # plt.imshow(np.nansum(data,axis=2),interpolation="nearest")
                        # plt.show()
                        w=2
                        where_bad_data = np.where(np.isnan(badpix))
                        where_finite_data = np.where(np.isfinite(badpix))
                        where_finite_raveldata = np.where(np.isfinite(np.ravel(badpix)))
                        HPFmodel_H0 = np.reshape(HPFbkg_model,((2*w+1)**2,(2*w+1)**2*nl)).transpose()[where_finite_raveldata[0],:]
                        where_valid_parameters = np.where(np.sum(np.abs(HPFmodel_H0),axis=0)!=0)
                        HPFmodel_H0 = HPFmodel_H0[:,where_valid_parameters[0]]
                        ravelHPFdata = np.ravel(HPFdata)[where_finite_raveldata]
                        HPFmodel_H0[np.where(np.isnan(HPFmodel_H0))]=0
                        sigmas_ravel = np.ravel(sigmas)[where_finite_raveldata]

                        HPFparas_H0,HPFchi2_H0,rank,s = np.linalg.lstsq(HPFmodel_H0/sigmas_ravel[:,None],ravelHPFdata/sigmas_ravel,rcond=None)

                        data_model_H0 = np.dot(HPFmodel_H0,HPFparas_H0)
                        ravelresiduals_H0 = ravelHPFdata-data_model_H0

                        canvas = np.zeros((5,5,nl))+np.nan
                        canvas[where_finite_data] = ravelresiduals_H0

                        PSF[where_bad_data] = np.nan
                        myspec = np.nansum(canvas*PSF,axis=(0,1))/np.nansum(PSF,axis=(0,1))/transmission
                        myspec_list.append(myspec)
                        # myspecwvs_list.append(np.nanmean(planetwvs,axis=(0,1)))
                        myspecwvs_list.append(wvs)

                    mymeanspec = np.mean(np.array(myspec_list),axis=0)
                    myspecwvs_conca = np.concatenate(myspecwvs_list)
                    myspec_conca = np.concatenate(myspec_list)
                    nbins = nl
                    binedges = np.linspace(wvs[0]-dwv/4,wvs[-1]+dwv/4,nbins+1,endpoint=True)
                    bincenter = np.linspace(wvs[0],wvs[-1],nbins,endpoint=True)
                    digitized = np.digitize(myspecwvs_conca,binedges)-1
                    final_spec = np.zeros(nbins)+np.nan
                    final_spec_std = np.zeros(nbins)+np.nan
                    for k in np.arange(2,nbins):
                        where_digit = np.where((k==digitized)*(np.isfinite(myspec_conca)))
                        if np.size(where_digit[0]) > 0.2*len(myspecwvs_list):
                            final_spec[k]=np.nanmean(myspec_conca[where_digit])
                            final_spec_std[k]=np.nanstd(myspec_conca[where_digit]/np.sqrt(np.size(where_digit[0])))
                        else:
                            final_spec[k]=np.nan
                            final_spec_std[k]=np.nan


                    osiris_data_dir = "/data/osiris_data"
                    molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
                    mol_linestyle_list = ["-","-","-"]#["-","--",":"]
                    for molid,(molecule,mol_linestyle) in enumerate(zip(["CO","H2O","CH4"],mol_linestyle_list)):
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
                            planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)

                        molec_spec = planet_spec_func(bincenter)
                        from reduce_HPFonly_diagcov import LPFvsHPF
                        HPF_molec_spec = LPFvsHPF(molec_spec,cutoff)[1]

                        plt.sca(ax_CCF_list[molid])
                        c_kms = 299792.458
                        rvshifts = np.arange(-100*dprv,100*dprv,dprv)
                        argnullrv = np.argmin(np.abs(rvshifts))
                        ccf = np.zeros(rvshifts.shape)
                        ccf_noise = np.zeros(rvshifts.shape)
                        for plrv_id in range(np.size(rvshifts)):
                            wvs4planet_model = bincenter*(1-(rvshifts[plrv_id])/c_kms)
                            myLPFspec2,myHPFspec2 = LPFvsHPF(planet_spec_func(wvs4planet_model),cutoff)
                            myHPFspec2 = myHPFspec2/np.sqrt(np.nansum(myHPFspec2**2))
                            ccf[plrv_id] = np.nansum(final_spec*myHPFspec2/final_spec_std**2)/np.nansum(myHPFspec2**2/final_spec_std**2)
                            ccf_noise[plrv_id] = np.nansum(final_spec_noise*myHPFspec2/final_spec_noise_std**2)/np.nansum(myHPFspec2**2/final_spec_noise_std**2)

                        ccf_snr = ccf/np.nanstd(ccf_noise)
                        if labeldate == "201611??":
                            plt.plot(rvshifts,ccf_snr/np.max(ccf_snr),color=colordate,linewidth=3,linestyle=mol_linestyle,label="Planet signal "+labeldate)
                        else:
                            plt.plot(rvshifts,ccf_snr/np.max(ccf_snr),color=colordate,linewidth=1,linestyle=mol_linestyle,label="Planet signal "+labeldate)
                        plt.plot(rvshifts,ccf_noise/np.nanstd(ccf_noise)/np.max(ccf_snr),color=colordate,linewidth=1,linestyle="--")
                        if IFSfilter == "Kbb":
                            plt.gca().text(-300,0.8,planet+": "+molecule,ha="left",va="bottom",rotation=0,size=fontsize,color="black")
                            # plt.gca().annotate("S/N={0:0.1f}".format(np.nanmax(ccf_snr)),xy=(300,8),va="bottom",ha="right",fontsize=fontsize,color="black")
                            plt.ylim([-0.5,1])
                        elif IFSfilter == "Hbb":
                            plt.gca().text(-300,8,planet+": "+molecule,ha="left",va="bottom",rotation=0,size=fontsize,color="black")
                            # plt.gca().annotate("S/N={0:0.1f}".format(np.nanmax(ccf_snr)),xy=(3750,8),va="bottom",ha="right",fontsize=fontsize,color="black")
                            plt.ylim([-2,10])
                        plt.xlabel(r"$\Delta V$ (km/s)",fontsize=fontsize)
                        plt.xlim([-300,300])
                        # plt.xlim([-4000,4000])
                        plt.xticks([-150,0,150,300])
                        plt.gca().tick_params(axis='x', labelsize=fontsize)
                        plt.gca().tick_params(axis='y', labelsize=fontsize)
                        # plt.show()
                plt.sca(ax_CCF_list[0])
                plt.ylabel("$S/N$",fontsize=15)
                plt.xticks([-300,-150,0,150,300])
                plt.sca(ax_CCF_list[2])
                plt.legend(loc="center left",frameon=True,fontsize=fontsize)
                # plt.yticks([-0.5,0,0.5,1])
                plt.show()
                plt.tight_layout()
                f1.subplots_adjust(wspace=0)
                # print("Saving "+os.path.join(out_pngs,"HR_8799_"+planet,"HR_8799_"+planet+"_"+IFSfilter+"_estispec_ccf.png"))
                # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR_8799_"+planet+"_"+IFSfilter+"_estispec_ccf.png"),bbox_inches='tight')
                # plt.savefig(os.path.join(out_pngs,"HR_8799_"+planet,"HR_8799_"+planet+"_"+IFSfilter+"_estispec_ccf.pdf"),bbox_inches='tight')
                # plt.close(f1.number)
            # print(k,np.size(where_digit[0]),bincenter[k],final_spec[k],final_spec_std[k])
    exit()