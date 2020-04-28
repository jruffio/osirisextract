__author__ = 'jruffio'


import csv
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import astropy.io.fits as pyfits


def get_err_from_posterior(x,posterior):
    ind = np.argsort(posterior)
    cum_posterior = np.zeros(np.shape(posterior))
    cum_posterior[ind] = np.cumsum(posterior[ind])
    cum_posterior = cum_posterior/np.max(cum_posterior)
    argmax_post = np.argmax(cum_posterior)
    # plt.figure(10)
    # plt.plot(x,posterior)
    # plt.plot(x,cum_posterior)
    # plt.show()
    if len(x[0:np.min([argmax_post+1,len(x)])]) < 2:
        lx = x[0]
    else:
        lf = interp1d(cum_posterior[0:np.min([argmax_post+1,len(x)])],
                      x[0:np.min([argmax_post+1,len(x)])],bounds_error=False,fill_value=x[0])
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx=x[-1]
    else:
        rf = interp1d(cum_posterior[argmax_post::],x[argmax_post::],bounds_error=False,fill_value=x[-1])
        rx = rf(1-0.6827)
    return lx,x[argmax_post],rx,argmax_post

if 1:
    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
    fontsize=12
    numbasis_list = np.array([10])
    IFSfilter = "HK"
    useprior = False
    if useprior:
        priorsuffix = "_prior"
    else:
        priorsuffix = ""
    # rv_b = [0.4770881896148752]
    # rvmerr_b = [0.009366900980685822]
    # rvperr_b = [0.007481767555257368]
    # rv_c = [0.5520881896148753  ]
    # rvmerr_c = [0.008513334049441879]
    # rvperr_c = [0.005541985372088609]
    # rv_d = [0.5870881896148753]
    # rvmerr_d = [0.03407766692871694]
    # rvperr_d = [0.03155199331507741]





    myoutfilename = "CtoO_HR_8799_b_measurements_kl{0}_HK".format(numbasis_list[0])+priorsuffix+".pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_b",myoutfilename.replace(".pdf","_logposterior.fits")))
    print(hdulist[0].data.shape)
    T_sampling,logg_sampling, CtoO_sampling, posterior_b = hdulist[1].data,hdulist[2].data,hdulist[3].data,hdulist[0].data

    myoutfilename = "CtoO_HR_8799_c_measurements_kl{0}_HK".format(numbasis_list[0])+priorsuffix+".pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_c",myoutfilename.replace(".pdf","_logposterior.fits")))
    print(hdulist[0].data.shape)
    T_sampling,logg_sampling, CtoO_sampling, posterior_c = hdulist[1].data,hdulist[2].data,hdulist[3].data,hdulist[0].data

    myoutfilename = "CtoO_HR_8799_d_measurements_kl{0}_HK".format(numbasis_list[0])+priorsuffix+".pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_d",myoutfilename.replace(".pdf","_logposterior.fits")))
    print(hdulist[0].data.shape)
    T_sampling,logg_sampling, CtoO_sampling, posterior_d = hdulist[1].data,hdulist[2].data,hdulist[3].data,hdulist[0].data

    # plt.plot(T_sampling,CtoO_sampling[np.argmax(max(posterior_b,axis=1))])
    # plt.plot(T_sampling,np.max(posterior_c,axis=1))
    # plt.plot(T_sampling,np.max(posterior_d,axis=1))
    # plt.show()

    dCtoO_sampling = CtoO_sampling - CtoO_sampling[len(CtoO_sampling)//2]
    dCtoO_bc = np.zeros((np.size(logg_sampling),np.size(logg_sampling)))
    dCtoO_dc = np.zeros((np.size(logg_sampling),np.size(logg_sampling)))
    CtoO_b = np.zeros((np.size(logg_sampling),3))
    CtoO_c = np.zeros((np.size(logg_sampling),3))
    CtoO_d = np.zeros((np.size(logg_sampling),3))
    for k,Tb in enumerate(logg_sampling):
        prof_posterior_b = np.sum(np.exp(posterior_b[:,k,:,:] - np.max(posterior_b[:,k,:,:])),axis=(0,2))
        prof_posterior_d = np.sum(np.exp(posterior_d[:,k,:,:] - np.max(posterior_d[:,k,:,:])),axis=(0,2))
        for l,Tc in enumerate(logg_sampling):
            prof_posterior_c = np.sum(np.exp(posterior_c[:,l,:,:] - np.max(posterior_c[:,l,:,:])),axis=(0,2))
            if l==0:
                CtoO_b[k,0],CtoO_b[k,1],CtoO_b[k,2],_ = get_err_from_posterior(CtoO_sampling,prof_posterior_b)
                CtoO_d[k,0],CtoO_d[k,1],CtoO_d[k,2],_ = get_err_from_posterior(CtoO_sampling,prof_posterior_d)
            if k==0:
                CtoO_c[l,0],CtoO_c[l,1],CtoO_c[l,2],_ = get_err_from_posterior(CtoO_sampling,prof_posterior_c)
            # exit()
    #
            delta_bc_posterior = np.correlate(prof_posterior_b,prof_posterior_c,mode="same")
            delta_bc_posterior = delta_bc_posterior/np.max(delta_bc_posterior)
            deltaRV_bc_lCI,deltaCtoO_bc,deltaRV_bc_rCI,_ = get_err_from_posterior(dCtoO_sampling,delta_bc_posterior)
            dCtoO_bc[k,l] = deltaCtoO_bc

            delta_dc_posterior = np.correlate(prof_posterior_d,prof_posterior_c,mode="same")
            delta_dc_posterior = delta_dc_posterior/np.max(delta_dc_posterior)
            deltaRV_dc_lCI,deltaCtoO_dc,deltaRV_dc_rCI,_ = get_err_from_posterior(dCtoO_sampling,delta_dc_posterior)
            dCtoO_dc[k,l] = deltaCtoO_dc
    #
    plt.figure(3,figsize=(12,6))

    plt.subplot(1,2,1)
    plt.fill_between(logg_sampling, CtoO_b[:,0],CtoO_b[:,2],color="#0099cc",alpha=0.5)
    plt.fill_between(logg_sampling, CtoO_c[:,0],CtoO_c[:,2],color="#ff9900",alpha=0.5)
    plt.fill_between(logg_sampling, CtoO_d[:,0],CtoO_d[:,2],color="#6600ff",alpha=0.5)
    plt.plot(logg_sampling, CtoO_b[:,1],linestyle="-",linewidth=3,color="#0099cc",label="b")
    plt.plot(logg_sampling, CtoO_c[:,1],linestyle="--",linewidth=3,color="#ff9900",label="c")
    plt.plot(logg_sampling, CtoO_d[:,1],linestyle=":",linewidth=3,color="#6600ff",label="d")
    plt.legend(loc='upper left',frameon=True,fontsize=fontsize)
    plt.ylabel("C/O",fontsize=fontsize)
    plt.xlabel(r"log(g [cm/s]) (K)",fontsize=fontsize)

    plt.subplot(1,2,2)
    for k,Tb in enumerate(logg_sampling):
        plt.plot(logg_sampling,dCtoO_bc[k,:],linestyle="-",linewidth=3,color="#0099cc",alpha=0.5)
        plt.plot(logg_sampling,dCtoO_dc[k,:],linestyle=":",linewidth=3,color="#6600ff",alpha=0.5)
    plt.ylabel(r"$[C/O]_{[b,d]}-[C/O]_c$",fontsize=fontsize)
    plt.xlabel(r"log(g$_c$ [cm/s]) (K)",fontsize=fontsize)
    if 1:
        print("Saving "+os.path.join(out_pngs,"CtoO_HR_8799_bcd_loggvariability"+priorsuffix+".pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd_loggvariability"+priorsuffix+".pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd_loggvariability"+priorsuffix+".png"))

    dCtoO_sampling = CtoO_sampling - CtoO_sampling[len(CtoO_sampling)//2]
    dCtoO_bc = np.zeros((np.size(T_sampling),np.size(T_sampling)))
    dCtoO_dc = np.zeros((np.size(T_sampling),np.size(T_sampling)))
    CtoO_b = np.zeros((np.size(T_sampling),3))
    CtoO_c = np.zeros((np.size(T_sampling),3))
    CtoO_d = np.zeros((np.size(T_sampling),3))
    for k,Tb in enumerate(T_sampling):
        prof_posterior_b = np.sum(np.exp(posterior_b[k,:,:,:] - np.max(posterior_b[k,:,:,:])),axis=(0,2))
        prof_posterior_d = np.sum(np.exp(posterior_d[k,:,:,:] - np.max(posterior_d[k,:,:,:])),axis=(0,2))
        for l,Tc in enumerate(T_sampling):
            prof_posterior_c = np.sum(np.exp(posterior_c[l,:,:,:] - np.max(posterior_c[l,:,:,:])),axis=(0,2))
            if l==0:
                CtoO_b[k,0],CtoO_b[k,1],CtoO_b[k,2],_ = get_err_from_posterior(CtoO_sampling,prof_posterior_b)
                CtoO_d[k,0],CtoO_d[k,1],CtoO_d[k,2],_ = get_err_from_posterior(CtoO_sampling,prof_posterior_d)
            if k==0:
                CtoO_c[l,0],CtoO_c[l,1],CtoO_c[l,2],_ = get_err_from_posterior(CtoO_sampling,prof_posterior_c)
            # exit()
    #
            delta_bc_posterior = np.correlate(prof_posterior_b,prof_posterior_c,mode="same")
            delta_bc_posterior = delta_bc_posterior/np.max(delta_bc_posterior)
            deltaRV_bc_lCI,deltaCtoO_bc,deltaRV_bc_rCI,_ = get_err_from_posterior(dCtoO_sampling,delta_bc_posterior)
            dCtoO_bc[k,l] = deltaCtoO_bc

            delta_dc_posterior = np.correlate(prof_posterior_d,prof_posterior_c,mode="same")
            delta_dc_posterior = delta_dc_posterior/np.max(delta_dc_posterior)
            deltaRV_dc_lCI,deltaCtoO_dc,deltaRV_dc_rCI,_ = get_err_from_posterior(dCtoO_sampling,delta_dc_posterior)
            dCtoO_dc[k,l] = deltaCtoO_dc
    #
    plt.figure(4,figsize=(12,6))

    plt.subplot(1,2,1)
    plt.fill_between(T_sampling, CtoO_b[:,0],CtoO_b[:,2],color="#0099cc",alpha=0.5)
    plt.fill_between(T_sampling, CtoO_c[:,0],CtoO_c[:,2],color="#ff9900",alpha=0.5)
    plt.fill_between(T_sampling, CtoO_d[:,0],CtoO_d[:,2],color="#6600ff",alpha=0.5)
    plt.plot(T_sampling, CtoO_b[:,1],linestyle="-",linewidth=3,color="#0099cc",label="b")
    plt.plot(T_sampling, CtoO_c[:,1],linestyle="--",linewidth=3,color="#ff9900",label="c")
    plt.plot(T_sampling, CtoO_d[:,1],linestyle=":",linewidth=3,color="#6600ff",label="d")
    plt.legend(loc='upper left',frameon=True,fontsize=fontsize)
    plt.ylabel("C/O",fontsize=fontsize)
    plt.xlabel("T (K)",fontsize=fontsize)

    plt.subplot(1,2,2)
    for k,Tb in enumerate(T_sampling):
        plt.plot(T_sampling,dCtoO_bc[k,:],linestyle="-",linewidth=3,color="#0099cc",alpha=0.5)
        plt.plot(T_sampling,dCtoO_dc[k,:],linestyle=":",linewidth=3,color="#6600ff",alpha=0.5)
    plt.ylabel(r"$[C/O]_{[b,d]}-[C/O]_c$",fontsize=fontsize)
    plt.xlabel(r"T$_c$ (K)",fontsize=fontsize)
    if 1:
        print("Saving "+os.path.join(out_pngs,"CtoO_HR_8799_bcd_Tvariability"+priorsuffix+".pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd_Tvariability"+priorsuffix+".pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd_Tvariability"+priorsuffix+".png"))
    # plt.show()
    # exit()





    # hdulist.append(pyfits.PrimaryHDU(data=np.array(["20"+date for date in  unique_dates]).astype(np.int)))
    # hdulist.append(pyfits.ImageHDU(data=nightly_yval))
    # hdulist.append(pyfits.ImageHDU(data=nightly_m_yerr))
    # hdulist.append(pyfits.ImageHDU(data=nightly_p_yerr))
    # hdulist.append(pyfits.ImageHDU(data=nightly_Nexp))
    # hdulist.append(pyfits.ImageHDU(data=np.array([final_yval,final_m_yerr,final_p_yerr])))
    myoutfilename = "CtoO_HR_8799_b_measurements_kl{0}_{1}".format(numbasis_list[0],IFSfilter)+priorsuffix+".pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_b",myoutfilename.replace(".pdf","_data4plotting.fits")))
    epochs_b = hdulist[0].data.astype(np.str)
    epochs_rv_b = hdulist[1].data
    epochs_rvmerr_b = hdulist[2].data
    epochs_rvperr_b = hdulist[3].data
    epochs_Nexp_b = hdulist[4].data
    tmp = hdulist[5].data
    rv_b,rvmerr_b,rvperr_b = [tmp[0]],[tmp[1]],[tmp[2]]
    myoutfilename = "CtoO_HR_8799_c_measurements_kl{0}_{1}".format(numbasis_list[0],IFSfilter)+priorsuffix+".pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_c",myoutfilename.replace(".pdf","_data4plotting.fits")))
    epochs_c = hdulist[0].data.astype(np.str)
    epochs_rv_c = hdulist[1].data
    epochs_rvmerr_c = hdulist[2].data
    epochs_rvperr_c = hdulist[3].data
    epochs_Nexp_c = hdulist[4].data
    tmp = hdulist[5].data
    rv_c,rvmerr_c,rvperr_c = [tmp[0]],[tmp[1]],[tmp[2]]
    myoutfilename = "CtoO_HR_8799_d_measurements_kl{0}_{1}".format(numbasis_list[0],"HK")+priorsuffix+".pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_d",myoutfilename.replace(".pdf","_data4plotting.fits")))
    epochs_d = hdulist[0].data.astype(np.str)
    epochs_rv_d = hdulist[1].data
    epochs_rvmerr_d = hdulist[2].data
    epochs_rvperr_d = hdulist[3].data
    epochs_Nexp_d = hdulist[4].data
    tmp = hdulist[5].data
    rv_d,rvmerr_d,rvperr_d = [tmp[0]],[tmp[1]],[tmp[2]]
    # print( rv_b,rvmerr_b,rvperr_b)
    # print( rv_c,rvmerr_c,rvperr_c)
    # print( rv_d,rvmerr_d,rvperr_d)
    # #HK
    # # [0.5724985412658228] [0.007010956574370453] [0.008206830254938624]
    # # [0.5615070792405064] [0.007112094805131641] [0.0027126839461255603]
    # # [0.5505156172151899] [0.030442390162036426] [0.020825385568305488]
    # #Kbb
    # # [0.577994272278481] [0.007142601073822252] [0.0057130950403451175]
    # # [0.5670028102531646] [0.007719166550491652] [0.002812072013038791]
    # #Hbb
    # # [0.5505156172151899] [0.02549277339729694] [0.01772715262689173]
    # # [0.528532693164557] [0.020075260819417573] [0.021420610027287457]
    # exit()

    #kl 10
    # epochs_b =  ['20090722', '20100711', '20100712', '20130725', '20130726', '20130727', '20161106', '20161107', '20161108', '20180722']
    # epochs_rv_b =  [0.4570881896148752, 0.4570881896148752, 0.6020881896148753, 0.4770881896148752, 0.4720881896148752, 0.4870881896148752, 0.4870881896148752, 0.6420881896148753, 0.4570881896148752, 0.4570881896148752]
    # epochs_rvmerr_b = [0.0, 0.0, 0.012454921819535558, 0.015606546516245823, 0.015000000000000013, 0.017696403491876733, 0.024967236691255446, 0.030439740254334824, 0.0, 0.0]
    # epochs_rvperr_b = [0.006990640796310399, 0.01606401669871238, 0.019622255922465026, 0.018159846275942082, 0.0189906348634597, 0.05279881717897561, 0.042474981531926304, 0.015000000000000013, 0.01632177317917255, 0.015156153909095471]
    # epochs_Nexp_b = [6, 9, 9, 16, 9, 5, 2, 3, 1, 6]
    #
    # #kl 10
    # epochs_c = ['20100715', '20101104', '20110723', '20110724' ,'20110725' ,'20130726', '20171103']
    # epochs_rv_c = [0.5570881896148753, 0.5470881896148753, 0.5320881896148753, 0.5470881896148753, 0.5020881896148752, 0.4870881896148752, 0.5670881896148753]
    # epochs_rvmerr_c = [0.008845547722997837, 0.023882125665470988, 0.04023163621912473, 0.05677346393306559, 0.03813009080943314, 0.030000000000000027, 0.017422119852433915]
    # epochs_rvperr_c = [0.009547044243143232, 0.01648039746198937, 0.020437072773469178, 0.011416430549343382, 0.018144881385531497, 0.09520265512307635, 0.011870988646839553]
    # epochs_Nexp_c = [17, 7, 10, 2, 2, 1, 3]
    #
    # #kl 10
    # epochs_d =  ['20150720', '20150723', '20150828']
    # epochs_rv_d = [0.5520881896148753, 0.6220881896148753, 0.5470881896148753]
    # epochs_rvmerr_d = [0.027692237516661833, 0.02644814646131144, 0.06944956188925638]
    # epochs_rvperr_d = [0.07042022207853338, 0.03500000000000003, 0.04099991814166137]
    # epochs_Nexp_d = [8, 5, 3]

    allepochs = np.unique(np.concatenate([epochs_b,epochs_c,epochs_d]))
    allepochs_rv_b = np.zeros(allepochs.shape) + np.nan
    allepochs_rvmerr_b = np.zeros(allepochs.shape) + np.nan
    allepochs_rvperr_b = np.zeros(allepochs.shape) + np.nan
    allepochs_Nexp_b = np.zeros(allepochs.shape) + np.nan
    allepochs_rv_c = np.zeros(allepochs.shape) + np.nan
    allepochs_rvmerr_c = np.zeros(allepochs.shape) + np.nan
    allepochs_rvperr_c = np.zeros(allepochs.shape) + np.nan
    allepochs_Nexp_c = np.zeros(allepochs.shape) + np.nan
    allepochs_rv_d = np.zeros(allepochs.shape) + np.nan
    allepochs_rvmerr_d = np.zeros(allepochs.shape) + np.nan
    allepochs_rvperr_d = np.zeros(allepochs.shape) + np.nan
    allepochs_Nexp_d = np.zeros(allepochs.shape) + np.nan
    for epochid,epoch in enumerate(allepochs):
        if epoch in epochs_d:
            wheretmp = np.where(int(epoch)==np.array(epochs_d).astype(np.int))
            if len(wheretmp[0]) != 0:
                wheretmp = wheretmp[0][0]
                allepochs_rv_d[epochid] = epochs_rv_d[wheretmp]
                allepochs_rvmerr_d[epochid] = epochs_rvmerr_d[wheretmp]
                allepochs_rvperr_d[epochid] = epochs_rvperr_d[wheretmp]
                allepochs_Nexp_d[epochid] = epochs_Nexp_d[wheretmp]
        if epoch in epochs_c:
            wheretmp = np.where(int(epoch)==np.array(epochs_c).astype(np.int))
            print(epoch,wheretmp)
            if len(wheretmp[0]) != 0:
                wheretmp = wheretmp[0][0]
                allepochs_rv_c[epochid] = epochs_rv_c[wheretmp]
                allepochs_rvmerr_c[epochid] = epochs_rvmerr_c[wheretmp]
                allepochs_rvperr_c[epochid] = epochs_rvperr_c[wheretmp]
                allepochs_Nexp_c[epochid] = epochs_Nexp_c[wheretmp]
        if epoch in epochs_b:
            wheretmp = np.where(int(epoch)==np.array(epochs_b).astype(np.int))
            if len(wheretmp[0]) != 0:
                wheretmp = wheretmp[0][0]
                allepochs_rv_b[epochid] = epochs_rv_b[wheretmp]
                allepochs_rvmerr_b[epochid] = epochs_rvmerr_b[wheretmp]
                allepochs_rvperr_b[epochid] = epochs_rvperr_b[wheretmp]
                allepochs_Nexp_b[epochid] = epochs_Nexp_b[wheretmp]
    # exit()

    def convertdates(date_list):
        return [date[4:6]+"-"+date[6:8]+"-"+date[0:4] for date in date_list ]

    plt.figure(1,figsize=(6,6))
    if 1:
        plt.plot(numbasis_list,rv_b,linestyle="",color="#0099cc",marker="x") #"#ff9900" "#0099cc" "#6600ff"
        plt.errorbar(numbasis_list,rv_b,yerr=(rvmerr_b,rvperr_b),color="#0099cc",label="b: RV")
        # plt.plot(numbasis_list+0.02,rv_fake_b,linestyle="",color="#0099cc",marker="x")
        # eb = plt.errorbar(numbasis_list+0.02,rv_fake_b,yerr=rverr_fake_b,color="#0099cc",fmt="",linestyle="--",label="c: corrected RV")
        # eb[-1][0].set_linestyle("--")
    if 1:
        plt.plot(numbasis_list,rv_c,linestyle="",color="#ff9900",marker="x") #"#ff9900" "#0099cc" "#6600ff"
        plt.errorbar(numbasis_list,rv_c,yerr=(rvmerr_c,rvperr_c),color="#ff9900",label="c: RV")
        # plt.plot(numbasis_list+0.02,rv_fake_c,linestyle="",color="#ff9900",marker="x")
        # eb = plt.errorbar(numbasis_list+0.02,rv_fake_c,yerr=rverr_fake_c,color="#ff9900",fmt="",linestyle="--",label="c: corrected RV")
        # eb[-1][0].set_linestyle("--")
    if 1:
        plt.plot(numbasis_list,rv_d,linestyle="",color="#6600ff",marker="x") #"#ff9900" "#0099cc" "#6600ff"
        plt.errorbar(numbasis_list,rv_d,yerr=(rvmerr_d,rvperr_d),color="#6600ff",label="d: RV")
        # plt.plot(numbasis_list+0.02,rv_fake_d,linestyle="",color="#6600ff",marker="x")
        # eb = plt.errorbar(numbasis_list+0.02,rv_fake_d,yerr=rverr_fake_d,color="#6600ff",fmt="",linestyle="--",label="c: corrected RV")
        # eb[-1][0].set_linestyle("--")


    plt.legend(loc="lower left",frameon=True,fontsize=fontsize)
    plt.xlabel("# PCA modes",fontsize=fontsize)
    plt.xticks([0,1,2],[0,1,10])
    plt.ylabel("C/O",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    # plt.gca().spines["right"].set_visible(False)
    # plt.gca().spines["top"].set_visible(False)
    if 1:
        print("Saving "+os.path.join(out_pngs,"CtoO_HR_8799_bcd_PCA"+priorsuffix+".pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd_PCA"+priorsuffix+".pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd_PCA"+priorsuffix+".png"))


    plt.figure(2,figsize=(12,12))
    plt.subplot2grid((4,1),(0,0),rowspan=2)
    # print([0,np.size(allepochs)],rv_d[-1]-rverr_d[-1],rv_d[-1]+rverr_d[-1])
    # plt.fill_betweenx([0,np.size(allepochs)],rv_d[-1]-rverr_d[-1],rv_d[-1]+rverr_d[-1],alpha=0.5,color="6600cc")
    # plt.show()
    plt.fill_betweenx([0,np.size(allepochs)],rv_b[-1]-rvmerr_b[-1],rv_b[-1]+rvperr_b[-1],alpha=0.2,color="#006699")
    plt.fill_betweenx([0,np.size(allepochs)],rv_c[-1]-rvmerr_c[-1],rv_c[-1]+rvperr_c[-1],alpha=0.2,color="#cc6600")
    plt.fill_betweenx([0,np.size(allepochs)],rv_d[-1]-rvmerr_d[-1],rv_d[-1]+rvperr_d[-1],alpha=0.2,color="#6600cc")

    # print("bonjour",epochs_rv_b,epochs_rverr_b,epochs_b,epochs_Nexp_b)
    # print("Planet & Date & RV & N cubes \\\\")
    # for a,b,c,d in zip(epochs_rv_b,epochs_rverr_b,epochs_b,epochs_Nexp_b):
    #     if np.isnan(a):
    #         continue
    #     formated_date =  c[0:4]+"-"+c[4:6]+"-"+c[6:8]
    #     print("& {0} & ${1:.1f} \\pm {2:.1f}$ & {3} \\\\".format(formated_date,a,b,int(d)))
    eb = plt.errorbar(allepochs_rv_b,np.arange(np.size(allepochs)),xerr=(allepochs_rvmerr_b,allepochs_rvperr_b),fmt="none",color="#0099cc",alpha=1,label="b (# exposures)")
    eb[-1][0].set_linestyle("-")
    plt.plot(allepochs_rv_b,np.arange(np.size(allepochs)),"x",color="#0099cc")
    wherenotnans = np.where(np.isfinite(allepochs_rv_b))
    for y,(x,date,num) in enumerate(zip(allepochs_rv_b,allepochs,allepochs_Nexp_b)):
        if np.isfinite(x):
            plt.gca().text(x,y,"{0}".format(int(num)),ha="center",va="bottom",rotation=0,size=fontsize,color="#003366",alpha=1)
    # plt.plot([rv_b[-1]-rverr_b[-1],rv_b[-1]-rverr_b[-1]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#006699",alpha=0.4)
    plt.plot([rv_b[-1],rv_b[-1]],[0,np.size(allepochs)],linestyle="-",linewidth=2,color="#003366",alpha=0.4)
    # plt.plot([rv_b[-1]+rverr_b[-1],rv_b[-1]+rverr_b[-1]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#006699",alpha=0.4)

    # print("bonjour",epochs_rv_c,epochs_rverr_c,epochs_c,epochs_Nexp_c)
    # print("Planet & Date & RV & N cubes \\\\")
    # for a,b,c,d in zip(epochs_rv_c,epochs_rverr_c,epochs_c,epochs_Nexp_c):
    #     if np.isnan(a):
    #         continue
    #     formated_date =  c[0:4]+"-"+c[4:6]+"-"+c[6:8]
    #     print("& {0} & ${1:.1f} \\pm {2:.1f}$ & {3} \\\\".format(formated_date,a,b,int(d)))
    eb = plt.errorbar(allepochs_rv_c,np.arange(np.size(allepochs)),xerr=(allepochs_rvmerr_c,allepochs_rvperr_c),fmt="none",color="#ff9900",alpha=1,label="c (# exposures)")
    eb[-1][0].set_linestyle("--")
    plt.plot(allepochs_rv_c,np.arange(np.size(allepochs)),"x",color="#ff9900")
    wherenotnans = np.where(np.isfinite(allepochs_rv_c))
    for y,(x,date,num) in enumerate(zip(allepochs_rv_c,allepochs,allepochs_Nexp_c)):
        if np.isfinite(x):
            plt.gca().text(x,y,"{0}".format(int(num)),ha="center",va="bottom",rotation=0,size=fontsize,color="#cc3300",alpha=1)
    # plt.plot([rv_c[-1]-rverr_c[-1],rv_c[-1]-rverr_c[-1]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#cc6600",alpha=0.4)
    plt.plot([rv_c[-1],rv_c[-1]],[0,np.size(allepochs)],linestyle="-",linewidth=2,color="#cc3300",alpha=0.4)
    # plt.plot([rv_c[-1]+rverr_c[-1],rv_c[-1]+rverr_c[-1]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#cc6600",alpha=0.4)

    # print("bonjour",epochs_rv_d,epochs_rverr_d,epochs_d,epochs_Nexp_d)
    # print("Planet & Date & RV & N cubes \\\\")
    # for a,b,c,d in zip(epochs_rv_d,epochs_rverr_d,epochs_d,epochs_Nexp_d):
    #     if np.isnan(a):
    #         continue
    #     formated_date =  c[0:4]+"-"+c[4:6]+"-"+c[6:8]
    #     print("& {0} & ${1:.1f} \\pm {2:.1f}$ & {3} \\\\".format(formated_date,a,b,int(d)))
    eb = plt.errorbar(allepochs_rv_d,np.arange(np.size(allepochs)),xerr=(allepochs_rvmerr_d,allepochs_rvperr_d),fmt="none",color="#6600ff",alpha=1,label="d (# exposures)")
    eb[-1][0].set_linestyle(":")
    plt.plot(allepochs_rv_d,np.arange(np.size(allepochs)),"x",color="#6600ff")
    wherenotnans = np.where(np.isfinite(allepochs_rv_d))
    for y,(x,date,num) in enumerate(zip(allepochs_rv_d,allepochs,allepochs_Nexp_d)):
        if np.isfinite(x):
            plt.gca().text(x,y,"{0}".format(int(num)),ha="center",va="bottom",rotation=0,size=fontsize,color="#330099",alpha=1)
    # plt.plot([rv_d[-1]-rverr_d[-1],rv_d[-1]-rverr_d[-1]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#6600cc",alpha=0.4)
    plt.plot([rv_d[-1],rv_d[-1]],[0,np.size(allepochs)],linestyle="-",linewidth=2,color="#330099",alpha=0.4)
    # plt.plot([rv_d[-1]+rverr_d[-1],rv_d[-1]+rverr_d[-1]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#6600cc",alpha=0.4)

    plt.xlim([0.45, 0.75])
    plt.xlabel("C/O",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.yticks(np.arange(0,np.size(allepochs)),convertdates(allepochs))
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",0.45))
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#


    # plt.figure(3,figsize=(12,3))
    plt.subplot2grid((4,1),(2,0),rowspan=1)
    myoutfilename = "CtoO_HR_8799_b_measurements_kl{0}_HK".format(numbasis_list[0])+priorsuffix+".pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_b",myoutfilename.replace(".pdf","_posterior.fits")))
    print(hdulist[0].data.shape)
    rvsampling, posterior_b = hdulist[3].data,np.sum(hdulist[0].data,axis=(0,1))
    posterior_b /=np.max(posterior_b)
    plt.gca().text(rv_b[-1]+0.003,1,"${0:.3f}".format(rv_b[-1])+"^{+"+"{0:.3f}".format(rvperr_b[-1])+"}"+"_{-"+"{0:.3f}".format(rvmerr_b[-1])+"}$",ha="left",va="bottom",rotation=0,size=fontsize,color="#003366")
    plt.plot(rvsampling, posterior_b,linestyle="-",linewidth=3,color="#0099cc",label="b")

    myoutfilename = "CtoO_HR_8799_c_measurements_kl{0}_HK".format(numbasis_list[0])+priorsuffix+".pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_c",myoutfilename.replace(".pdf","_posterior.fits")))
    print(hdulist[0].data.shape)
    rvsampling, posterior_c = hdulist[3].data,np.sum(hdulist[0].data,axis=(0,1))
    posterior_c /=np.max(posterior_c)
    plt.gca().text(rv_c[-1]+0.001,1,"${0:.3f}".format(rv_c[-1])+"^{+"+"{0:.3f}".format(rvperr_c[-1])+"}"+"_{-"+"{0:.3f}".format(rvmerr_c[-1])+"}$",ha="center",va="bottom",rotation=0,size=fontsize,color="#cc3300")
    plt.plot(rvsampling, posterior_c,linestyle="--",linewidth=3,color="#ff9900",label="c")

    myoutfilename = "CtoO_HR_8799_d_measurements_kl{0}_HK".format(numbasis_list[0])+priorsuffix+".pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_d",myoutfilename.replace(".pdf","_posterior.fits")))
    print(hdulist[0].data.shape)
    rvsampling, posterior_d = hdulist[3].data,np.sum(hdulist[0].data,axis=(0,1))
    posterior_d /=np.max(posterior_d)
    plt.gca().text(rv_d[-1]-0.002,1,"${0:.3f}".format(rv_d[-1])+"^{+"+"{0:.3f}".format(rvperr_d[-1])+"}"+"_{-"+"{0:.3f}".format(rvmerr_d[-1])+"}$",ha="right",va="bottom",rotation=0,size=fontsize,color="#330099")
    plt.plot(rvsampling, posterior_d,linestyle=":",linewidth=3,color="#6600ff",label="d")

    plt.xlim([0.45, 0.75])
    plt.ylim([0,1.1])
    plt.xlabel("C/O",fontsize=fontsize)
    plt.ylabel("$\propto \mathcal{P}(C/O|d)$",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",0.45))
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = handles#handles[0:3]+[handles[6]]+handles[3:6]+[handles[7]]
    new_labels = labels#labels[0:3]+[labels[6]]+labels[3:6]+[labels[7]]
    # print(handles)
    # print(labels)
    # exit()
    plt.legend(new_handles,new_labels,loc="upper right",frameon=True,fontsize=fontsize)#



    # plt.figure(4,figsize=(12,3))
    plt.subplot2grid((4,1),(3,0),rowspan=1)

    drvsampling = rvsampling - rvsampling[len(rvsampling)//2]
    delta_bc_posterior = np.correlate(posterior_b,posterior_c,mode="same")
    delta_bc_posterior = delta_bc_posterior/np.max(delta_bc_posterior)
    deltaRV_bc_lCI,deltaRV_bc,deltaRV_bc_rCI,_ = get_err_from_posterior(drvsampling,delta_bc_posterior)
    confidence_interval = (1-np.cumsum(delta_bc_posterior)[np.argmin(np.abs(drvsampling))]/np.sum(delta_bc_posterior))

    plt.gca().text(deltaRV_bc+0.001,1,"${0:.3f}".format(deltaRV_bc)+"^{+"+"{0:.3f}".format(deltaRV_bc_rCI-deltaRV_bc)+"}"+"_{-"+"{0:.3f}".format(deltaRV_bc-deltaRV_bc_lCI)+"}$",ha="center",va="bottom",rotation=0,size=fontsize,color="#003366")
    # plt.gca().text(deltaRV_bc-1,1.0,"${0:.1f}\pm {1:.1f}$ km/s".format(deltaRV_bc,deltaRV_bc_sig),ha="left",va="bottom",rotation=0,size=fontsize,color="#003366")
    plt.plot(drvsampling,delta_bc_posterior,linestyle="-",linewidth=3,color="#0099cc",label="$[C/O]_b-[C/O]_c$")


    delta_dc_posterior = np.correlate(posterior_d,posterior_c,mode="same")
    delta_dc_posterior = delta_dc_posterior/np.max(delta_dc_posterior)
    deltaRV_dc_lCI,deltaRV_dc,deltaRV_dc_rCI,_ = get_err_from_posterior(drvsampling,delta_dc_posterior)
    confidence_interval = (1-np.cumsum(delta_dc_posterior)[np.argmin(np.abs(drvsampling))]/np.sum(delta_dc_posterior))

    plt.gca().text(deltaRV_dc+0.001,1,"${0:.3f}".format(deltaRV_dc)+"^{+"+"{0:.3f}".format(deltaRV_dc_rCI-deltaRV_dc)+"}"+"_{-"+"{0:.3f}".format(deltaRV_dc-deltaRV_dc_lCI)+"}$",ha="right",va="bottom",rotation=0,size=fontsize,color="#330099")
    # plt.gca().text(deltaRV_dc-1,1.0,"${0:.1f}\pm {1:.1f}$ km/s".format(deltaRV_dc,deltaRV_dc_sig),ha="left",va="bottom",rotation=0,size=fontsize,color="#660066")
    plt.plot(drvsampling,delta_dc_posterior,linestyle=":",linewidth=3,color="#6600ff",label="$[C/O]_d-[C/O]_c$")

    plt.xlim([-0.1,0.1])
    plt.ylim([0,1.1])
    plt.xlabel(r"$[C/O]_{[b,d]}-[C/O]_c$",fontsize=fontsize)
    plt.ylabel("$\propto \mathcal{P}([C/O]_b-[C/O]_c|d)$",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",0))
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#

    if 1:
        plt.tight_layout()
        print("Saving "+os.path.join(out_pngs,"CtoO_HR_8799_bcd"+priorsuffix+".pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd"+priorsuffix+".pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd"+priorsuffix+".png"))




    plt.show()
    exit()

