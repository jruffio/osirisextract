__author__ = 'jruffio'

import glob
import os
import csv
import astropy.io.fits as pyfits
import numpy as np
import multiprocessing as mp
from reduce_HPFonly_diagcov import convolve_spectrum
# from reduce_HPFonly_diagcov import LPFvsHPF
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from copy import copy


def LPFvsHPF(myvec,cutoff):
    myvec_cp = copy(myvec)
    #handling nans:
    wherenans = np.where(np.isnan(myvec_cp))
    for k in wherenans[0]:
        myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-20]):np.min([np.size(myvec_cp),k+20])])

    fftmyvec = np.fft.fft(np.concatenate([myvec_cp,myvec_cp[::-1]],axis=0))
    LPF_fftmyvec = copy(fftmyvec)
    LPF_fftmyvec[cutoff:(2*np.size(myvec_cp)-cutoff+1)] = 0
    LPF_myvec = np.real(np.fft.ifft(LPF_fftmyvec))[0:np.size(myvec_cp)]
    HPF_myvec = myvec_cp - LPF_myvec

    LPF_myvec[wherenans] = np.nan
    HPF_myvec[wherenans] = np.nan
    return LPF_myvec,HPF_myvec

if __name__ == "__main__":
    try:
        import mkl
        mkl.set_num_threads(1)
    except:
        pass

    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
    planet = "kap_And"
    # planet = "HR_8799_b"
    # planet = "HR_8799_c"
    # planet = "HR_8799_d"
    # date = "2010*"
    cutoff = 40
    fontsize = 12
    fakes = True
    R=4000
    IFSfilter = "Kbb"
    test = False
    c_kms = 299792.458


    if 0:
        planetcolor_list = ["#0099cc","#ff9900","#6600ff"]
        # for planet,planetcolor in zip(["b","c","d"],planetcolor_list):
            # for IFSfilter in ["Kbb","Hbb"]:
        for planetcolor in ["#0099cc"]:
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
                if 1:
                    osiris_data_dir = "/data/osiris_data/"
                    phoenix_folder = os.path.join(osiris_data_dir,"phoenix")
                    planet_template_folder = os.path.join(osiris_data_dir,"planets_templates")
                    molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
                    sky_transmission_folder = os.path.join(osiris_data_dir,"sky_transmission")

                    if planet == "HR_8799_b":
                        travis_spec_filename=os.path.join(planet_template_folder,
                                                      "HR8799b_"+IFSfilter[0:1]+"_3Oct2018.save")
                    if planet == "HR_8799_c":
                        travis_spec_filename=os.path.join(planet_template_folder,
                                                      "HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save")
                    if planet == "HR_8799_d":
                        travis_spec_filename=os.path.join(planet_template_folder,
                                                      "HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save")
                    if "HR_8799" in planet:
                        phoenix_model_host_filename = glob.glob(os.path.join(phoenix_folder,"HR_8799"+"*.fits"))[0]
                        host_rv = -12.6 #+-1.4
                        host_limbdark = 0.5
                        host_vsini = 49 # true = 49
                        star_name = "HR_8799"
                    if planet == "51_Eri_b":
                        phoenix_model_host_filename = glob.glob(os.path.join(phoenix_folder,"51_Eri"+"*.fits"))[0]
                        travis_spec_filename=os.path.join(planet_template_folder,
                                                      "51Eri_b_highres_template.save")
                        host_rv = 12.6 #+-0.3
                        host_limbdark = 0.5
                        host_vsini = 80
                        star_name = "51_Eri"
                    if planet == "kap_And":
                        phoenix_model_host_filename = glob.glob(os.path.join(phoenix_folder,"kap_And"+"*.fits"))[0]
                        travis_spec_filename=os.path.join(planet_template_folder,
                                                      "KapAnd_lte19-3.50-0.0.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019.7.save")
                        host_rv = -12.7 #+-0.8
                        host_limbdark = 0.5
                        host_vsini = 150 #unknown
                        star_name = "kap_And"
                    planet_template_filename=travis_spec_filename.replace(".save",
                                                                          "_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))

                    with open(planet_template_filename, 'r') as csvfile:
                        csv_reader = csv.reader(csvfile, delimiter=' ')
                        list_starspec = list(csv_reader)
                        oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                        col_names = oriplanet_spec_str_arr[0]
                        oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
                        oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
                        planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)


                    # planet_template_filename2 = "/data/osiris_data/hr8799b_modelgrid/lte10-3.0-0.0.aces_hr8799b_pgs=4d6_Kzz=1d8_C=8.48_O=8.82_gs=5um.exoCH4_hiresHK.7.D2e.sorted_gaussconv_R4000_Kbb.csv"
                    # # planet_template_filename2 = "/data/osiris_data/hr8799b_modelgrid/lte11-3.5-0.0.aces_hr8799b_pgs=4d6_Kzz=1d8_C=8.33_O=8.51_gs=5um.exoCH4_hiresHK.7.D2e.sorted_gaussconv_R4000_Kbb.csv"
                    # with open(planet_template_filename2, 'r') as csvfile:
                    #     csv_reader = csv.reader(csvfile, delimiter=' ')
                    #     list_starspec = list(csv_reader)
                    #     oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                    #     col_names = oriplanet_spec_str_arr[0]
                    #     oriplanet_spec2 = oriplanet_spec_str_arr[1::,1].astype(np.float)
                    #     oriplanet_spec_wvs2 = oriplanet_spec_str_arr[1::,0].astype(np.float)
                    #     planet_spec_func2 = interp1d(oriplanet_spec_wvs2,oriplanet_spec2,bounds_error=False,fill_value=np.nan)

                # plt.plot(oriplanet_spec_wvs,oriplanet_spec/np.max(oriplanet_spec),label="1")
                # plt.plot(oriplanet_spec_wvs2,oriplanet_spec2/np.max(oriplanet_spec2),label="2")
                # plt.show()

                # gridconv_filename = "/data/osiris_data/sonora/spectra/sp_t1700g562nc_m0.0_gaussconv_R4000_"+IFSfilter+".csv"
                # with open(gridconv_filename, 'r') as csvfile:
                #     csv_reader = csv.reader(csvfile, delimiter=' ')
                #     list_starspec = list(csv_reader)
                #     oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                #     col_names = oriplanet_spec_str_arr[0]
                #     oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
                #     oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
                #     where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                #     oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                #     planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)


                for resnumbasis in [10]:#np.arange(0,20):#np.array([0,1,5]):
                    ## file specific info
                    if resnumbasis ==0:
                        fileinfos_filename = "/data/osiris_data/"+planet+"/fileinfos_Kbb_jb.csv"
                    else:
                        if test:
                            fileinfos_filename = "/data/osiris_data/"+planet+"/fileinfos_Kbb_jb.csv"
                        else:
                            fileinfos_filename = "/data/osiris_data/"+planet+"/fileinfos_Kbb_jb_kl{0}.csv".format(resnumbasis)
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
                    fakes_myspec_list = []
                    fakes_myspec_std_list = []
                    fakes_myspeccorr_list = []
                    mycorrspec_list = []
                    fakes_myspec_list = []
                    fakes_myspec_std_list = []
                    model_list = []
                    model_notrans_list = []
                    trans_list = []

                    cen_filename_id = colnames.index("cen filename")
                    filename_id = colnames.index("filename")
                    kcen_id = colnames.index("kcen")
                    lcen_id = colnames.index("lcen")
                    baryrv_id = colnames.index("barycenter rv")

                    for fileitem in list_data:
                        filename = fileitem[filename_id]
                        if test:
                            # print(fileitem[cen_filename_id])
                            # reduc_filename = fileitem[cen_filename_id].replace("20191104_RVsearch","20191202_newresmodel").replace("kl0",'kl{0}'.format(resnumbasis))
                            # reduc_filename = fileitem[cen_filename_id].replace("20191104_RVsearch","20191120_newresmodel").replace("kl0",'kl{0}'.format(resnumbasis))
                            # reduc_filename = fileitem[cen_filename_id].replace("20191104_RVsearch","20191204_grid").replace("kl0",'kl{0}'.format(resnumbasis))
                            reduc_filename = fileitem[cen_filename_id].replace("20191104_RVsearch","20191205_spec").replace("20191120_newres_RV","20191205_spec").replace("kl0",'kl{0}'.format(resnumbasis))
                            # print(reduc_filename)
                            # print(glob.glob(reduc_filename))
                            # exit()
                        else:
                            reduc_filename = fileitem[cen_filename_id]
                        if "Kbb" not in os.path.basename(filename):
                            continue
                        # if resnumbasis == 0:
                        #     data_filename = filename.replace("reduced_jb","reduced_jb/sherlock/20190920_resH0model_detec").replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_rescalc_estispec.fits")
                        # else:
                        #     data_filename = filename.replace("reduced_jb","reduced_jb/sherlock/20190920_resH0model_detec").replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl{0}_estispec.fits".format(resnumbasis))
                        # if resnumbasis == 0:
                        #     data_filename = filename.replace("reduced_jb","reduced_jb/20190923_HPF_restest4").replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_rescalc_estispec.fits")
                        # else:
                        #     data_filename = filename.replace("reduced_jb","reduced_jb/20190923_HPF_restest4").replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl{0}_estispec.fits".format(resnumbasis))
                        # if resnumbasis == 0:
                        #     data_filename = filename.replace("reduced_jb","reduced_jb/sherlock/20190925_resH0model_RV").replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_rescalc_estispec.fits")
                        # else:
                        #     data_filename = filename.replace("reduced_jb","reduced_jb/sherlock/20190925_resH0model_RV").replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl{0}_estispec.fits".format(resnumbasis))
                        # data_filename = filename.replace("reduced_jb","reduced_jb/20191018_HPF_faketest").replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl{0}_estispec.fits".format(resnumbasis))
                        data_filename = reduc_filename.replace(".fits","_estispec.fits")
                        # print(glob.glob(data_filename))
                        # exit()
                        # if "20110723" not in data_filename:
                        #     continue
                        # if "20100715" not in data_filename:
                        #     continue
                        if len(glob.glob(data_filename)) == 0:
                            continue

                        print(data_filename)
                        with pyfits.open(data_filename) as hdulist:
                            esti_spec_arr = hdulist[0].data
                            # estispec_np[0,row,col,:] = np.nanmean(wvs[None,None,:]-data_wvsol_offsets[:,:,None],axis=(0,1))
                            # estispec_np[1,row,col,:] = np.nansum(canvas_residuals_with_nans*PSF,axis=(0,1))/np.nansum(PSF**2,axis=(0,1))
                            # estispec_np[2,row,col,:] = np.nansum(canvas_residuals_with_nans*PSF,axis=(0,1))/np.nansum(PSF**2,axis=(0,1))/tr4planet(wvs)
                            # if fake_paras is not None:
                            #     estispec_np[3,row,col,:] = np.nansum(HPF_fake*fake_paras["contrast"],axis=(0,1))
                            #     estispec_np[4,row,col,:] = np.nansum(HPF_fake*fake_paras["contrast"],axis=(0,1))/tr4planet(wvs)
                            # else:
                            #     estispec_np[3,row,col,:] = np.nan
                            #     estispec_np[4,row,col,:] = np.nan

                        print(esti_spec_arr.shape)

                        if fileitem[kcen_id] == "nan":
                            continue
                        plcen_k,plcen_l = int(fileitem[kcen_id]),int(fileitem[lcen_id])
                        host_bary_rv = -float(fileitem[baryrv_id])/1000
                        myspecwvs_list.append(copy(esti_spec_arr[0,:,plcen_k,plcen_l])*(1-host_bary_rv/c_kms) )
                        myspec_list.append(copy(esti_spec_arr[1,:,plcen_k,plcen_l]))

                        esti_spec_arr_cp = copy(esti_spec_arr)
                        esti_spec_arr_cp[:,:,plcen_k-7:plcen_k+8,plcen_l-7:plcen_l+8] = np.nan
                        # esti_spec_arr_cp[1,:,:,:] = esti_spec_arr_cp[1,:,:,:]#/np.nanstd(esti_spec_arr_cp[1,:,:,:],axis=0)[None,:,:]
                        # esti_spec_arr[:,:,plcen_k-5:plcen_k+5,0:plcen_l] = np.nan
                        bias_myspec = np.nanmean(esti_spec_arr_cp[1,:,:,:],axis=(1,2))
                        std_myspec = np.nanstd(esti_spec_arr_cp[1,:,:,:],axis=(1,2))#/30
                        perfile_transmission = np.nanmean(esti_spec_arr_cp[2,:,:,:],axis=(1,2))
                        bias_myspec_list.append(bias_myspec)
                        myspec_std_list.append(std_myspec)
                        trans_list.append(perfile_transmission)
                        model_notrans_list.append(LPFvsHPF(planet_spec_func(esti_spec_arr[0,:,plcen_k,plcen_l]*(1-host_bary_rv/c_kms)),cutoff)[1])
                        model_list.append(LPFvsHPF(perfile_transmission*planet_spec_func(esti_spec_arr[0,:,plcen_k,plcen_l]*(1-host_bary_rv/c_kms)),cutoff)[1])

                        if fakes:
                            try:
                                fakes_filename = data_filename.replace("_estispec.fits","_fakes_estispec.fits")
                                with pyfits.open(fakes_filename) as hdulist:
                                    fakes_esti_spec_arr = hdulist[0].data
                                fakes_esti_spec_arr[:,:,plcen_k-7:plcen_k+8,plcen_l-7:plcen_l+8] = np.nan
                                fakes_myspec = np.nanmean(fakes_esti_spec_arr[1,:,:,:],axis=(1,2))
                                fakes_std_myspec = np.nanstd(fakes_esti_spec_arr[1,:,:,:],axis=(1,2))#/30
                                fakes_myspec_list.append(fakes_myspec)
                                fakes_myspec_std_list.append(fakes_std_myspec)
                            except:
                                fakes_myspec_list.append(bias_myspec+np.nan)
                                fakes_myspec_std_list.append(bias_myspec+np.nan)

                        # myspec_list.append(copy(esti_spec_fakes_arr[2,:,plcen_k+10,plcen_l]))
                        # fakes_myspec_list.append(copy(esti_spec_fakes_arr[4,:,plcen_k+10,plcen_l]))

                        # # import matplotlib.pyplot as plt
                        # for a in np.arange(0,esti_spec_arr_cp.shape[2],3):
                        #     for b in np.arange(0,esti_spec_arr_cp.shape[3],3):
                        #         plt.plot(esti_spec_arr_cp[1,:,a,b],alpha=0.5)
                        # plt.plot(bias_myspec-std_myspec,"--",linewidth=5)
                        # plt.plot(bias_myspec,linewidth=5)
                        # plt.plot(bias_myspec+std_myspec,"--",linewidth=5)
                        # plt.show()


                        a = copy(esti_spec_arr[1,:,plcen_k,plcen_l])
                        b = bias_myspec
                        a = a - b*np.nansum(a*b)/np.nansum(b*b)
                        mycorrspec_list.append(a)

                        # datafakes_filename = filename.replace("reduced_jb","reduced_jb/20191018_HPF_faketest").replace(".fits","_outputHPF_cutoff40_sherlock_v1_search_resinmodel_kl{0}_fakes_estispec.fits".format(resnumbasis))
                        # with pyfits.open(datafakes_filename) as hdulist:
                        #     esti_spec_fakes_arr = hdulist[0].data
                        # esti_spec_fakes_arr[:,:,plcen_k-5:plcen_k+6,plcen_l-5:plcen_l+6] = np.nan
                        # esti_spec_fakes_arr[:,:,:,0:plcen_l-5] = np.nan
                        # esti_spec_fakes_arr[:,:,:,plcen_l+6::] = np.nan
                        # fake_myspec = np.nanmean(esti_spec_fakes_arr[1,:,:,:],axis=(1,2))
                        # fake_std_myspec = np.nanstd(esti_spec_fakes_arr[1,:,:,:],axis=(1,2))
                        # fakes_myspec_list.append(fake_myspec)
                        # fakes_myspec_std_list.append(fake_std_myspec)
                        # a = copy(fake_myspec)
                        # b = bias_myspec
                        # a = a - b*np.nansum(a*b)/np.nansum(b*b)
                        # fakes_myspeccorr_list.append(a)


                        # print(plcen_k,plcen_l)
                        # plt.imshow(np.nanmean(esti_spec_arr[1,:,:,:]-bias_myspec[:,None,None],axis=0),interpolation="nearest")
                        # plt.show()

                        # exit()



                    myspecwvs_conca = np.concatenate(myspecwvs_list)
                    myspec_conca = np.concatenate(myspec_list)
                    mymodel_conca = np.concatenate(model_list)
                    mymodel_notrans_conca = np.concatenate(model_notrans_list)
                    trans_conca = np.concatenate(trans_list)
                    myspec_std_conca = np.concatenate(myspec_std_list)
                    myspec_bias_conca = np.concatenate(bias_myspec_list)
                    mycorrspec_conca = np.concatenate(mycorrspec_list)
                    fakes_spec_conca = np.concatenate(fakes_myspec_list)
                    fakes_spec_std_conca = np.concatenate(fakes_myspec_std_list)
                    # fakes_myspec_conca = np.concatenate(fakes_myspec_list)
                    # fakes_myspec_std_conca = np.concatenate(fakes_myspec_std_list)
                    # fakes_myspeccorr_conca = np.concatenate(fakes_myspeccorr_list)
                    nbins = nl
                    binedges = np.linspace(wvs[0]-dwv/4,wvs[-1]+dwv/4,nbins+1,endpoint=True)
                    bincenter = np.linspace(wvs[0],wvs[-1],nbins,endpoint=True)
                    digitized = np.digitize(myspecwvs_conca,binedges)-1
                    final_spec = np.zeros(nbins)+np.nan
                    final_model = np.zeros(nbins)+np.nan
                    final_model_notrans = np.zeros(nbins)+np.nan
                    final_trans = np.zeros(nbins)+np.nan
                    final_spec_biascorr = np.zeros(nbins)+np.nan
                    final_spec_std = np.zeros(nbins)+np.nan
                    finalfakes_spec = np.zeros(nbins)+np.nan
                    finalfakes_spec_biascorr = np.zeros(nbins)+np.nan
                    finalfakes_spec_std = np.zeros(nbins)+np.nan
                    final_spec_fakes = np.zeros(nbins)+np.nan
                    final_spec_std_fakes = np.zeros(nbins)+np.nan
                    for k in np.arange(2,nbins):
                        where_digit = np.where((k==digitized)*(np.isfinite(myspec_conca)))
                        if np.size(where_digit[0]) > 0.2*len(myspecwvs_list):
                            final_spec[k]=np.nansum((myspec_conca[where_digit])/myspec_std_conca[where_digit]**2)/np.nansum(1/myspec_std_conca[where_digit]**2)
                            final_model[k]=np.nansum((mymodel_conca[where_digit])/myspec_std_conca[where_digit]**2)/np.nansum(1/myspec_std_conca[where_digit]**2)
                            final_model_notrans[k]=np.nansum((mymodel_notrans_conca[where_digit])/myspec_std_conca[where_digit]**2)/np.nansum(1/myspec_std_conca[where_digit]**2)
                            # final_spec_biascorr[k]=np.nansum((myspec_conca[where_digit]-myspec_bias_conca[where_digit])/myspec_std_conca[where_digit]**2)/np.nansum(1/myspec_std_conca[where_digit]**2)
                            final_spec_biascorr[k]=np.nansum((mycorrspec_conca[where_digit])/myspec_std_conca[where_digit]**2)/np.nansum(1/myspec_std_conca[where_digit]**2)
                            final_spec_std[k]=np.sqrt(1/np.nansum(1/myspec_std_conca[where_digit]**2))

                            final_spec_fakes[k]=np.nansum((fakes_spec_conca[where_digit])/fakes_spec_std_conca[where_digit]**2)/np.nansum(1/fakes_spec_std_conca[where_digit]**2)
                            final_spec_std_fakes[k]=np.sqrt(1/np.nansum(1/fakes_spec_std_conca[where_digit]**2))
                            final_trans[k] = np.nanmean(trans_conca[where_digit])
                            # finalfakes_spec[k]=np.nansum((fakes_myspec_conca[where_digit])/fakes_myspec_std_conca[where_digit]**2)/np.nansum(1/fakes_myspec_std_conca[where_digit]**2)
                            # finalfakes_spec_biascorr[k]=np.nansum((fakes_myspeccorr_conca[where_digit])/fakes_myspec_std_conca[where_digit]**2)/np.nansum(1/fakes_myspec_std_conca[where_digit]**2)
                            # finalfakes_spec_std[k]=np.sqrt(1/np.nansum(1/fakes_myspec_std_conca[where_digit]**2))
                        else:
                            final_spec[k]=np.nan
                            final_model[k]=np.nan
                            final_model_notrans[k]=np.nan
                            final_spec_biascorr[k]=np.nan
                            final_spec_std[k]=np.nan
                            final_spec_fakes[k]=np.nan
                            final_spec_std_fakes[k]=np.nan
                            final_trans[k] = np.nan

                            # finalfakes_spec[k]=np.nan
                            # finalfakes_spec_biascorr[k]=np.nan
                            # finalfakes_spec_std[k]=np.nan


                    # plt.plot(bincenter,final_spec,linestyle="--",label="{0}: final_spec".format(resnumbasis))
                    # plt.plot(bincenter,final_spec_biascorr,linestyle="-",label="{0} final_spec - bias".format(resnumbasis))
                    # plt.plot(bincenter,final_spec-final_spec_biascorr,linestyle="-",label="{0} bias".format(resnumbasis))
                    # plt.plot(bincenter,final_spec_std,linestyle="-",alpha=0.5,label="{0} std".format(resnumbasis))
                    # plt.plot(bincenter,final_spec_fakes,linestyle="-",label="{0} final_spec_fakes".format(resnumbasis))
                    # plt.plot(bincenter,final_spec_std_fakes,linestyle="-",label="{0} final_spec_std_fakes".format(resnumbasis))

                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_trans))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec"+"_trans"+"_kl{0}.fits".format(resnumbasis)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec"+"_trans"+"_kl{0}.fits".format(resnumbasis)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=bincenter))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec"+"_wvs"+"_kl{0}.fits".format(resnumbasis)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec"+"_wvs"+"_kl{0}.fits".format(resnumbasis)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_spec))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec"+"_kl{0}.fits".format(resnumbasis)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec"+"_kl{0}.fits".format(resnumbasis)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_model))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec_"+"model"+"_kl{0}.fits".format(resnumbasis)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec_"+"model"+"_kl{0}.fits".format(resnumbasis)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_model_notrans))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec_"+"model_notrans"+"_kl{0}.fits".format(resnumbasis)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec_"+"model_notrans"+"_kl{0}.fits".format(resnumbasis)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_spec_biascorr))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec_"+"biascorr"+"_kl{0}.fits".format(resnumbasis)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec_"+"biascorr"+"_kl{0}.fits".format(resnumbasis)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_spec_std))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_specstd"+"_kl{0}.fits".format(resnumbasis)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_specstd"+"_kl{0}.fits".format(resnumbasis)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_spec_fakes))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec_"+"fakes"+"_kl{0}.fits".format(resnumbasis)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec_"+"fakes"+"_kl{0}.fits".format(resnumbasis)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_spec_fakes))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_specstd_"+"fakes"+"_kl{0}.fits".format(resnumbasis)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_specstd_"+"fakes"+"_kl{0}.fits".format(resnumbasis)), clobber=True)
                    hdulist.close()

                    # plt.plot(bincenter,finalfakes_spec,linestyle="--",label="{0}: fakes final_spec".format(resnumbasis))
                    # plt.plot(bincenter,finalfakes_spec_biascorr,linestyle="-",label="{0} fakes final_spec - bias".format(resnumbasis))
                    # plt.plot(bincenter,finalfakes_spec-finalfakes_spec_biascorr,linestyle="-",label="{0} fakes bias".format(resnumbasis))
                    # plt.plot(bincenter,finalfakes_spec_std,linestyle="-",alpha=0.5,label="{0} fakes std".format(resnumbasis))



                # osiris_data_dir = "/data/osiris_data"
                # molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
                # mol_linestyle_list = ["-","-","-"]#["-","--",":"]
                # # for molid,(molecule,mol_linestyle) in enumerate(zip(["CO","H2O","CH4"],mol_linestyle_list)):
                # for molid,(molecule,mol_linestyle) in enumerate(zip(["CO","H2O"],mol_linestyle_list)):
                #     print(molecule)
                #     travis_mol_filename=os.path.join(molecular_template_folder,
                #                                   "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7")
                #     travis_mol_filename_D2E=os.path.join(molecular_template_folder,
                #                                   "lte11-4.0_hr8799c_pgs=4d6_Kzz=1d8_gs=5um."+molecule+"only.7_D2E")
                #     mol_template_filename=travis_mol_filename+"_gaussconv_R{0}_{1}.csv".format(R,IFSfilter)
                #
                #     with open(mol_template_filename, 'r') as csvfile:
                #         csv_reader = csv.reader(csvfile, delimiter=' ')
                #         list_starspec = list(csv_reader)
                #         oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                #         col_names = oriplanet_spec_str_arr[0]
                #         oriplanet_spec = oriplanet_spec_str_arr[1::3,1].astype(np.float)
                #         oriplanet_spec_wvs = oriplanet_spec_str_arr[1::3,0].astype(np.float)
                #         where_IFSfilter = np.where((oriplanet_spec_wvs>wvs[0])*(oriplanet_spec_wvs<wvs[-1]))
                #         oriplanet_spec = oriplanet_spec/np.mean(oriplanet_spec[where_IFSfilter])
                #         planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
                #
                #     molec_spec = planet_spec_func(bincenter)
                #     from reduce_HPFonly_diagcov import LPFvsHPF
                #     HPF_molec_spec = LPFvsHPF(molec_spec,cutoff)[1]
                #
                #     plt.plot(bincenter,HPF_molec_spec/np.nanstd(HPF_molec_spec),color="black",linestyle="--",label=molecule)

                # plt.legend()
                # plt.show()
                # exit()


    if 1:
        osiris_data_dir = "/data/osiris_data/"
        phoenix_folder = os.path.join(osiris_data_dir,"phoenix")
        planet_template_folder = os.path.join(osiris_data_dir,"planets_templates")
        molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
        sky_transmission_folder = os.path.join(osiris_data_dir,"sky_transmission")
        fileinfos_refstars_filename = os.path.join(osiris_data_dir,"fileinfos_refstars_jb.csv")

        if planet == "HR_8799_b":
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "HR8799b_"+IFSfilter[0:1]+"_3Oct2018.save")
            pl_rv = -9.0
            color = "#0099cc"
        if planet == "HR_8799_c":
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save")
            pl_rv = -11.1
            color = "#ff9900"
        if planet == "HR_8799_d":
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save")
            pl_rv = -15.7
            color = "#6600ff"
        if "HR_8799" in planet:
            phoenix_model_host_filename = glob.glob(os.path.join(phoenix_folder,"HR_8799"+"*.fits"))[0]
            host_rv = -12.6 #+-1.4
            host_limbdark = 0.5
            host_vsini = 49 # true = 49
            star_name = "HR_8799"
        if planet == "51_Eri_b":
            phoenix_model_host_filename = glob.glob(os.path.join(phoenix_folder,"51_Eri"+"*.fits"))[0]
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "51Eri_b_highres_template.save")
            host_rv = 12.6 #+-0.3
            host_limbdark = 0.5
            host_vsini = 80
            star_name = "51_Eri"
        if planet == "kap_And":
            phoenix_model_host_filename = glob.glob(os.path.join(phoenix_folder,"kap_And"+"*.fits"))[0]
            travis_spec_filename=os.path.join(planet_template_folder,
                                          "KapAnd_lte19-3.50-0.0.AGSS09.Dusty.Kzz=0.0.PHOENIX-ACES-2019.7.save")
            host_rv = -12.7 #+-0.8
            host_limbdark = 0.5
            host_vsini = 150 #unknown
            star_name = "kap_And"
            color ="#6600ff"
            pl_rv = -13.9
        planet_template_filename=travis_spec_filename.replace(".save",
                                                              "_gaussconv_R{0}_{1}.csv".format(R,IFSfilter))

        with open(planet_template_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            list_starspec = list(csv_reader)
            oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
            col_names = oriplanet_spec_str_arr[0]
            oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
            oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
            planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)

        f1 = plt.figure(1,figsize=(12,6))
        filename = os.path.join(out_pngs,planet+"_spec_"+"wvs"+"_kl{0}.fits".format(0))
        with pyfits.open(filename) as hdulist:
            bincenter = hdulist[0].data
        filename = os.path.join(out_pngs,planet+"_spec"+"_kl{0}.fits".format(0))
        with pyfits.open(filename) as hdulist:
            final_spec = hdulist[0].data
        planet_model = planet_spec_func(bincenter*(1-pl_rv/c_kms) )
        planet_model[np.where(np.isnan(final_spec))] = np.nan
        _,model_spec = LPFvsHPF(planet_model,40)
        # plt.plot(bincenter,model_spec*np.nansum(final_spec*model_spec)/np.nansum(model_spec*model_spec),color="black",linestyle="-",label="model")

        pca_list = [10]
        ax1 = []
        ax2 = []
        for it, resnumbasis in enumerate(pca_list):#np.arange(0,10):#[0,1,2]:
            # filename = os.path.join(out_pngs,planet+"_spec_"+"trans"+"_kl{0}.fits".format(resnumbasis))
            # with pyfits.open(filename) as hdulist:
            #     transmission = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_spec_"+"wvs"+"_kl{0}.fits".format(resnumbasis))
            with pyfits.open(filename) as hdulist:
                bincenter = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_spec"+"_kl{0}.fits".format(resnumbasis))
            with pyfits.open(filename) as hdulist:
                final_spec = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_spec_"+"model"+"_kl{0}.fits".format(resnumbasis))
            with pyfits.open(filename) as hdulist:
                final_model = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_spec_"+"model_notrans"+"_kl{0}.fits".format(resnumbasis))
            with pyfits.open(filename) as hdulist:
                final_model_notrans = hdulist[0].data
            # filename = os.path.join(out_pngs,planet+"_spec_"+"modeljb"+"_kl{0}.fits".format(resnumbasis))
            # with pyfits.open(filename) as hdulist:
            #     final_model2 = hdulist[0].data
            # filename = os.path.join(out_pngs,planet+"_spec_"+"modeljb_notrans"+"_kl{0}.fits".format(resnumbasis))
            # with pyfits.open(filename) as hdulist:
            #     final_model2_notrans = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_spec_"+"biascorr"+"_kl{0}.fits".format(resnumbasis))
            with pyfits.open(filename) as hdulist:
                final_spec_biascorr = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_specstd"+"_kl{0}.fits".format(resnumbasis))
            with pyfits.open(filename) as hdulist:
                final_spec_std = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_spec_"+"fakes"+"_kl{0}.fits".format(resnumbasis))
            with pyfits.open(filename) as hdulist:
                final_spec_fakes = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_specstd_"+"fakes"+"_kl{0}.fits".format(resnumbasis))
            with pyfits.open(filename) as hdulist:
                final_spec_std_fakes = hdulist[0].data

            # plt.figure(2)
            # plt.plot(transmission)
            # plt.show()

            nl = np.size(final_spec)
            for sp_id in range(2):
                if it ==0:
                    plt.subplot2grid((5*2,1),(5*sp_id,0),rowspan=2)
                    ax1.append(plt.gca())
                else:
                    plt.sca(ax1[sp_id])
                plt.plot(bincenter[sp_id*(nl//2):(sp_id+1)*(nl//2)],
                         final_spec[sp_id*(nl//2):(sp_id+1)*(nl//2)],
                         linestyle="-",label="data ({0} PCs)".format(resnumbasis),color=color) #["#0099cc","#ff9900","#6600ff"]
                # plt.plot(bincenter,final_spec_biascorr,linestyle="-",label="{0} final_spec - bias".format(resnumbasis))
                # plt.plot(bincenter,final_spec-final_spec_biascorr,linestyle="-",label="{0} bias".format(resnumbasis))
                # plt.plot(bincenter,final_spec_std,linestyle="-",alpha=0.5,label="{0} std".format(resnumbasis))
                # plt.plot(bincenter,final_spec_fakes,linestyle="-",label="{0} final_spec_fakes".format(resnumbasis))
                # plt.plot(bincenter,final_spec_std_fakes,linestyle="-",label="{0} final_spec_std_fakes".format(resnumbasis))

                final_model = (final_model*np.nansum(final_model*final_spec)/np.nansum(final_model**2))
                # final_model2 = (final_model2*np.nansum(final_model2*final_spec)/np.nansum(final_model2**2))
                if it+1 ==len(pca_list):
                    plt.plot(bincenter[sp_id*(nl//2):(sp_id+1)*(nl//2)],
                             final_model[sp_id*(nl//2):(sp_id+1)*(nl//2)],
                             linestyle="-",color="black",label="model")
                    # plt.plot(bincenter[sp_id*(nl//4):(sp_id+1)*(nl//4)],
                    #          final_model2[sp_id*(nl//4):(sp_id+1)*(nl//4)],
                    #          linestyle="--",color="grey",label="model from grid")
                    plt.gca().tick_params(axis='y', labelsize=fontsize)
                    # plt.ylim([-0.1,0.1])
                    plt.ylim([-1,1])
                    if sp_id==0:
                        plt.legend(loc="lower right",frameon=True,fontsize=fontsize)


                if it ==0:
                    plt.subplot2grid((5*2,1),(5*sp_id+2,0),rowspan=2)
                    ax2.append(plt.gca())
                else:
                    plt.sca(ax2[sp_id])
                plt.plot(bincenter[sp_id*(nl//2):(sp_id+1)*(nl//2)],
                         final_spec[sp_id*(nl//2):(sp_id+1)*(nl//2)]-final_model[sp_id*(nl//2):(sp_id+1)*(nl//2)],
                         linestyle="-",label="residuals",color="grey")
                # plt.plot(bincenter[sp_id*(nl//4):(sp_id+1)*(nl//4)],
                #          final_spec[sp_id*(nl//4):(sp_id+1)*(nl//4)]-final_model2[sp_id*(nl//4):(sp_id+1)*(nl//4)],
                #          linestyle="--",color="grey")
                plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                # plt.ylim([-0.1,0.1])
                plt.ylim([-0.5,0.5])
                if sp_id==0:
                    plt.legend(loc="lower right",frameon=True,fontsize=fontsize)

                plt.subplot2grid((5*2,1),(5*sp_id+4,0))
                plt.plot([-1],[-1])
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
                plt.tick_params(axis="y",which="both",labelleft=False,right=False,left=False)
                plt.xticks([])
                plt.yticks([])
                plt.gca().spines["right"].set_visible(False)
                plt.gca().spines["left"].set_visible(False)
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["bottom"].set_visible(False)
                plt.gca().patch.set_alpha(0)
        # plt.show()
        # plt.plot(bincenter,final_model_notrans,linestyle="-",color="grey",label="{0}: final_model_notrans".format(resnumbasis))
        plt.tight_layout()
        f1.subplots_adjust(wspace=0,hspace=0)
        plt.legend()
        print("Saving "+os.path.join(out_pngs,planet,planet+"_"+IFSfilter+"_spec_kl{0}.png".format(resnumbasis)))
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+IFSfilter+"_spec_kl{0}.png".format(resnumbasis)),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+IFSfilter+"_spec_kl{0}.pdf".format(resnumbasis)),bbox_inches='tight')
        plt.show()


