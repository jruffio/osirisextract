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
import pandas as pd


def LPFvsHPF(myvec,cutoff,nansmooth=10):
    myvec_cp = np.zeros(myvec.shape)
    myvec_cp[:] = copy(myvec[:])
    wherenans = np.where(np.isnan(myvec_cp))
    myvec_cp = np.array(pd.DataFrame(myvec_cp).interpolate(method="linear").fillna(method="bfill").fillna(method="ffill"))[:,0]
    # for k in wherenans[0]:
    #     myvec_cp[k] = np.nanmedian(myvec_cp[np.max([0,k-nansmooth]):np.min([np.size(myvec_cp),k+nansmooth])])

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
    # planet = "kap_And"
    # planet = "HR_8799_b"
    planet = "HR_8799_c"
    # planet = "HR_8799_d"
    # date = "2010*"
    cutoff = 40
    fontsize = 12
    fakes = True
    R=4000
    IFSfilter = "Kbb"
    c_kms = 299792.458


    if 0:
        planetcolor_list = ["#0099cc","#ff9900","#6600ff"]
        # for planet,planetcolor in zip(["b","c","d"],planetcolor_list):
            # for IFSfilter in ["Kbb","Hbb"]:
        for planetcolor in ["#0099cc"]:
            # for IFSfilter in ["Hbb"]:
            if 1:

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
                # print(wvs[0],wvs[-1])
                # exit()
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
                        plT,pllogg,plCtoO = 1200,3.8,0.56
                        plrv = -9.
                    if planet == "HR_8799_c":
                        travis_spec_filename=os.path.join(planet_template_folder,
                                                      "HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save")
                        plT,pllogg,plCtoO = 1200,3.8,0.56
                        plrv = -11.1
                    if planet == "HR_8799_d":
                        travis_spec_filename=os.path.join(planet_template_folder,
                                                      "HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save")
                        plT,pllogg,plCtoO = 1200,3.8,0.56
                        plrv = -15.7
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

                    tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R,IFSfilter))
                    hdulist = pyfits.open(tmpfilename)
                    planet_model_grid =  hdulist[0].data
                    oriplanet_spec_wvs =  hdulist[1].data
                    Tlistunique =  hdulist[2].data
                    logglistunique =  hdulist[3].data
                    CtoOlistunique =  hdulist[4].data
                    hdulist.close()

                    planet_spec_func_list = []
                    from scipy.interpolate import RegularGridInterpolator
                    myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)
                    # plT,pllogg,plCtoO =1160.0, 4.266666666666667,  0.5724985412658228
                    plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
                    planet_spec_func = interp1d(oriplanet_spec_wvs,myinterpgrid([plT,pllogg,plCtoO])[0],bounds_error=False,fill_value=np.nan)
                    planet_spec_func_list.append(planet_spec_func)
                    # plT,pllogg,plCtoO = 1200.0, 3.8, 0.5615070792405064
                    plT,pllogg,plCtoO = 1000.0, 3.8, 0.5615070792405064
                    planet_spec_func = interp1d(oriplanet_spec_wvs,myinterpgrid([plT,pllogg,plCtoO])[0],bounds_error=False,fill_value=np.nan)
                    planet_spec_func_list.append(planet_spec_func)
                    # plT,pllogg,plCtoO = 1200.0, 3.0, 0.5450198862025316
                    plT,pllogg,plCtoO = 800.0, 3.8, 0.5615070792405064
                    planet_spec_func = interp1d(oriplanet_spec_wvs,myinterpgrid([plT,pllogg,plCtoO])[0],bounds_error=False,fill_value=np.nan)
                    planet_spec_func_list.append(planet_spec_func)


                    with open(planet_template_filename, 'r') as csvfile:
                        csv_reader = csv.reader(csvfile, delimiter=' ')
                        list_starspec = list(csv_reader)
                        oriplanet_spec_str_arr = np.array(list_starspec, dtype=np.str)
                        col_names = oriplanet_spec_str_arr[0]
                        oriplanet_spec = oriplanet_spec_str_arr[1::,1].astype(np.float)
                        oriplanet_spec_wvs = oriplanet_spec_str_arr[1::,0].astype(np.float)
                        planet_spec_func = interp1d(oriplanet_spec_wvs,oriplanet_spec,bounds_error=False,fill_value=np.nan)
                    planet_spec_func_list.append(planet_spec_func)

                for resnumbasis in [10]:#np.arange(0,20):#np.array([0,1,5]):
                    ## file specific info
                    if resnumbasis ==0:
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
                    status_id = colnames.index("status")

                    for fileitem in list_data:
                        filename = fileitem[filename_id]
                        print(filename)
                        reduc_filename = fileitem[cen_filename_id]
                        if int(fileitem[status_id]) != 1:
                            continue
                        if IFSfilter not in os.path.basename(filename):
                            continue
                        # if "100715" not in os.path.basename(filename):
                        #     continue
                        data_filename = reduc_filename.replace(".fits","_estispec.fits")
                        if len(glob.glob(data_filename)) == 0:
                            continue

                        print(data_filename)
                        with pyfits.open(data_filename) as hdulist:
                            esti_spec_arr = hdulist[0].data

                        print(esti_spec_arr.shape)

                        plcen_k,plcen_l = int(fileitem[kcen_id]),int(fileitem[lcen_id])
                        host_bary_rv = -float(fileitem[baryrv_id])/1000
                        myspecwvs_list.append(copy(esti_spec_arr[0,:,plcen_k,plcen_l])*(1-host_bary_rv/c_kms) )
                        myspec_list.append(copy(esti_spec_arr[1,:,plcen_k,plcen_l]))

                        esti_spec_arr_cp = copy(esti_spec_arr)
                        esti_spec_arr_cp[:,:,plcen_k-7:plcen_k+8,plcen_l-7:plcen_l+8] = np.nan

                        std_myspec = np.nanstd(esti_spec_arr_cp[1,:,:,:],axis=(1,2))#/30
                        perfile_transmission = np.nanmean(esti_spec_arr_cp[2,:,:,:],axis=(1,2))
                        myspec_std_list.append(std_myspec)
                        trans_list.append(perfile_transmission)
                        model_sublist = []
                        for modid,planet_spec_func in enumerate(planet_spec_func_list):
                            model_sublist.append(LPFvsHPF(perfile_transmission*planet_spec_func(esti_spec_arr[0,:,plcen_k,plcen_l]*(1-(plrv+host_bary_rv)/c_kms)),cutoff)[1])
                        # print(model_sublist[0])
                        # exit()
                        model_list.append(model_sublist)

                    model_list = np.array(model_list)
                    model_list = np.moveaxis(model_list,1,0)
                    mymodel_conca = np.reshape(model_list,(len(planet_spec_func_list),len(trans_list)*np.size(perfile_transmission)))

                    myspecwvs_conca = np.concatenate(myspecwvs_list)
                    myspec_conca = np.concatenate(myspec_list)
                    myspec_std_conca = np.concatenate(myspec_std_list)
                    nbins = nl
                    binedges = np.linspace(wvs[0]-dwv/4,wvs[-1]+dwv/4,nbins+1,endpoint=True)
                    bincenter = np.linspace(wvs[0],wvs[-1],nbins,endpoint=True)
                    digitized = np.digitize(myspecwvs_conca,binedges)-1
                    final_spec = np.zeros(nbins)+np.nan
                    final_model = np.zeros((len(planet_spec_func_list),nbins))+np.nan
                    final_spec_std = np.zeros(nbins)+np.nan
                    for k in np.arange(2,nbins):
                        where_digit = np.where((k==digitized)*(np.isfinite(myspec_conca)))
                        if np.size(where_digit[0]) > 0.2*len(myspecwvs_list):
                            sumvar = np.nansum(1/myspec_std_conca[where_digit]**2)
                            final_spec[k]=np.nansum((myspec_conca[where_digit])/myspec_std_conca[where_digit]**2)/sumvar
                            for modid in range(len(planet_spec_func_list)):
                                final_model[modid,k]=np.nansum((mymodel_conca[modid,where_digit[0]])/myspec_std_conca[where_digit]**2)/sumvar
                            final_spec_std[k]=np.sqrt(1/sumvar)
                        else:
                            final_spec[k]=np.nan
                            final_model[:,k]=np.nan
                            final_spec_std[k]=np.nan

                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=bincenter))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec"+"_wvs"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec"+"_wvs"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_spec))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_model))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec_"+"model"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_spec_"+"model"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter)), clobber=True)
                    hdulist.close()
                    hdulist = pyfits.HDUList()
                    hdulist.append(pyfits.PrimaryHDU(data=final_spec_std))
                    try:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_specstd"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter)), overwrite=True)
                    except TypeError:
                        hdulist.writeto(os.path.join(out_pngs,planet+"_specstd"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter)), clobber=True)
                    hdulist.close()



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
        # filename = os.path.join(out_pngs,planet+"_spec_"+"wvs"+"_kl{0}.fits".format(0))
        # with pyfits.open(filename) as hdulist:
        #     bincenter = hdulist[0].data
        # filename = os.path.join(out_pngs,planet+"_spec"+"_kl{0}.fits".format(0))
        # with pyfits.open(filename) as hdulist:
        #     final_spec = hdulist[0].data
        # planet_model = planet_spec_func(bincenter*(1-pl_rv/c_kms) )
        # planet_model[np.where(np.isnan(final_spec))] = np.nan
        # _,model_spec = LPFvsHPF(planet_model,40)
        # # plt.plot(bincenter,model_spec*np.nansum(final_spec*model_spec)/np.nansum(model_spec*model_spec),color="black",linestyle="-",label="model")

        pca_list = [10]
        ax1 = []
        ax2 = []
        for it, resnumbasis in enumerate(pca_list):
            filename = os.path.join(out_pngs,planet+"_spec_"+"wvs"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
            with pyfits.open(filename) as hdulist:
                bincenter = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_spec"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
            with pyfits.open(filename) as hdulist:
                final_spec = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_spec_"+"model"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
            with pyfits.open(filename) as hdulist:
                final_model_arr = hdulist[0].data
            # print(np.nanmean(final_model_arr[0]))
            # exit()
            filename = os.path.join(out_pngs,planet+"_specstd"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
            with pyfits.open(filename) as hdulist:
                final_spec_std = hdulist[0].data

            # mylabel_list= ["T=1200K","T=1000K","T=800K","model RV"]
            # mylim=0.1
            print(final_model_arr.shape)
            print(planet == "HR_8799_b")
            if planet == "HR_8799_b":
                final_model_arr = final_model_arr[0,:][None,:]
                mylabel_list = ["Best fit model"]
                mylim=0.05
            if planet == "HR_8799_c":
                final_model_arr = final_model_arr[0,:][None,:]
                mylabel_list = ["Best fit model"]
                mylim=0.1
            if planet == "HR_8799_d":
                final_model_arr = final_model_arr[2,:][None,:]
                mylabel_list = ["Best fit model"]
                mylim=0.2

            nl = np.size(final_spec)
            for sp_id in range(2):
                if it ==0:
                    plt.subplot2grid((5*2,1),(5*sp_id,0),rowspan=2)
                    ax1.append(plt.gca())
                else:
                    plt.sca(ax1[sp_id])
                plt.plot(bincenter[sp_id*(nl//2):(sp_id+1)*(nl//2)],
                         final_spec[sp_id*(nl//2):(sp_id+1)*(nl//2)],
                         linestyle="-",label="Data ({0} PCs)".format(resnumbasis),color=color,linewidth=1.5) #["#0099cc","#ff9900","#6600ff"]

                for modid,(final_model,linestyle,mylabel ) in enumerate(zip(final_model_arr,["--","-",":"],mylabel_list)):#,"-."
                        final_model = (final_model*np.nansum(final_model*final_spec)/np.nansum(final_model**2))
                        if it+1 ==len(pca_list):
                            plt.plot(bincenter[sp_id*(nl//2):(sp_id+1)*(nl//2)],
                                     final_model[sp_id*(nl//2):(sp_id+1)*(nl//2)],
                                     linestyle=linestyle,color="black",label=mylabel,linewidth=1)
                            plt.gca().tick_params(axis='y', labelsize=fontsize)
                            # plt.ylim([-0.1,0.1])
                            plt.ylim([-mylim,mylim])
                            # plt.ylim([-1,1])
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
                plt.gca().yaxis.set_ticks([-mylim,0])
                # plt.plot(bincenter[sp_id*(nl//4):(sp_id+1)*(nl//4)],
                #          final_spec[sp_id*(nl//4):(sp_id+1)*(nl//4)]-final_model2[sp_id*(nl//4):(sp_id+1)*(nl//4)],
                #          linestyle="--",color="grey")
                plt.xlabel(r"$\lambda$ ($\mu$m)",fontsize=fontsize)
                plt.gca().tick_params(axis='x', labelsize=fontsize)
                plt.gca().tick_params(axis='y', labelsize=fontsize)
                # plt.ylim([-0.1,0.1])
                plt.ylim([-mylim,mylim])
                # plt.ylim([-0.5,0.5])
                # if sp_id==0:
                #     plt.legend(loc="lower right",frameon=True,fontsize=fontsize)

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
        print("Saving "+os.path.join(out_pngs,planet,planet+"_"+IFSfilter+"_spec_kl{0}_{1}.png".format(resnumbasis,IFSfilter)))
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+IFSfilter+"_spec_kl{0}_{1}.png".format(resnumbasis,IFSfilter)),bbox_inches='tight')
        plt.savefig(os.path.join(out_pngs,planet,planet+"_"+IFSfilter+"_spec_kl{0}_{1}.pdf".format(resnumbasis,IFSfilter)),bbox_inches='tight')
        plt.show()


