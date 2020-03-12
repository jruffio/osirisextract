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
    numbasis_list = np.array([0])
    rv_b = [0.4920881896148752]
    rvmerr_b = [0.013720486405409349]
    rvperr_b = [0.008466184363451479]
    rv_c = [0.5720881896148753  ]
    rvmerr_c = [0.007555074480420365]
    rvperr_c = [0.004273973788900376]
    rv_d = [0.5870881896148753]
    rvmerr_d = [0.037483488514273855]
    rvperr_d = [0.02700830863970982]

    #kl 10
    epochs_b =  ['20100711', '20100712', '20130725']
    epochs_rv_b =  [ 0.4570881896148752, 0.6020881896148753, 0.5470881896148753]
    epochs_rvmerr_b = [ 0.0, 0.012443490035948135, 0.022231000707946924]
    epochs_rvperr_b = [ 0.015040172364527316, 0.019625311449477212, 0.019595703217906935]
    epochs_Nexp_b = [6, 9, 9, 8]

    #kl 10
    epochs_c = ['20100715', '20101104', '20110723', '20110724' ,'20110725' ,'20130726', '20171103']
    epochs_rv_c = [0.5770881896148753, 0.6120881896148753, 0.5620881896148753, 0.5270881896148752, 0.5020881896148752, 0.5820881896148753, 0.5420881896148753]
    epochs_rvmerr_c = [0.00873566738438003, 0.012394674307188769, 0.01621109063021342, 0.020544220520430256, 0.0226948414896761, 0.06448905330747357, 0.00969140949731373]
    epochs_rvperr_c = [0.006128291869357483, 0.014823259324531213, 0.01246934796403465, 0.02202295195539039, 0.020185919585072543, 0.06202795456009802, 0.01702541758372611]
    epochs_Nexp_c = [17, 7, 10, 2, 2, 1, 3]

    #kl 10
    epochs_d =  ['20150720', '20150723', '20150828']
    epochs_rv_d = [0.5520881896148753, 0.6220881896148753, 0.5470881896148753]
    epochs_rvmerr_d = [0.029599372546365244, 0.027418757298598995, 0.07184297251288135]
    epochs_rvperr_d = [0.06663649790036053, 0.03500000000000003, 0.03981527313195121]
    epochs_Nexp_d = [8, 5, 3]

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
        plt.plot(rv_b,linestyle="",color="#0099cc",marker="x") #"#ff9900" "#0099cc" "#6600ff"
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
        print("Saving "+os.path.join(out_pngs,"CtoO_HR_8799_bcd_PCA.pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd_PCA.pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd_PCA.png"))


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
    plt.errorbar(allepochs_rv_b,np.arange(np.size(allepochs)),xerr=(allepochs_rvmerr_b,allepochs_rvperr_b),fmt="none",color="#0099cc")
    plt.plot(allepochs_rv_b,np.arange(np.size(allepochs)),"x",color="#0099cc",label="b (# exposures)")
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
    plt.errorbar(allepochs_rv_c,np.arange(np.size(allepochs)),xerr=(allepochs_rvmerr_c,allepochs_rvperr_c),fmt="none",color="#ff9900")
    plt.plot(allepochs_rv_c,np.arange(np.size(allepochs)),"x",color="#ff9900",label="c (# exposures)")
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
    plt.errorbar(allepochs_rv_d,np.arange(np.size(allepochs)),xerr=(allepochs_rvmerr_d,allepochs_rvperr_d),fmt="none",color="#6600ff")
    plt.plot(allepochs_rv_d,np.arange(np.size(allepochs)),"x",color="#6600ff",label="d (# exposures)")
    wherenotnans = np.where(np.isfinite(allepochs_rv_d))
    for y,(x,date,num) in enumerate(zip(allepochs_rv_d,allepochs,allepochs_Nexp_d)):
        if np.isfinite(x):
            plt.gca().text(x,y,"{0}".format(int(num)),ha="center",va="bottom",rotation=0,size=fontsize,color="#330099",alpha=1)
    # plt.plot([rv_d[-1]-rverr_d[-1],rv_d[-1]-rverr_d[-1]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#6600cc",alpha=0.4)
    plt.plot([rv_d[-1],rv_d[-1]],[0,np.size(allepochs)],linestyle="-",linewidth=2,color="#330099",alpha=0.4)
    # plt.plot([rv_d[-1]+rverr_d[-1],rv_d[-1]+rverr_d[-1]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#6600cc",alpha=0.4)

    plt.xlim([0.4570881896148752, 0.6570881896148754])
    plt.xlabel("C/O",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.yticks(np.arange(0,np.size(allepochs)),convertdates(allepochs))
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",0.4570881896148752))
    plt.legend(loc="center right",frameon=True,fontsize=fontsize)#


    # plt.figure(3,figsize=(12,3))
    plt.subplot2grid((4,1),(2,0),rowspan=1)
    myoutfilename = "CtoO_HR_8799_b_measurements.pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_b",myoutfilename.replace(".pdf","_CtoOposterior.fits")))
    print(hdulist[0].data.shape)
    rvsampling, posterior_b = hdulist[0].data[0,:],hdulist[0].data[1,:]
    plt.gca().text(rv_b[-1]+0.001,1,"${0:.3f}".format(rv_b[-1])+"^{+"+"{0:.3f}".format(rvperr_b[-1])+"}"+"_{-"+"{0:.3f}".format(rvmerr_b[-1])+"}$",ha="center",va="bottom",rotation=0,size=fontsize,color="#003366")
    plt.plot(rvsampling, posterior_b,linestyle="-",linewidth=3,color="#0099cc",label="b: Posterior")

    myoutfilename = "CtoO_HR_8799_c_measurements.pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_c",myoutfilename.replace(".pdf","_CtoOposterior.fits")))
    print(hdulist[0].data.shape)
    rvsampling, posterior_c = hdulist[0].data[0,:],hdulist[0].data[1,:]

    plt.gca().text(rv_c[-1]+0.001,1,"${0:.3f}".format(rv_c[-1])+"^{+"+"{0:.3f}".format(rvperr_c[-1])+"}"+"_{-"+"{0:.3f}".format(rvmerr_c[-1])+"}$",ha="center",va="bottom",rotation=0,size=fontsize,color="#cc3300")
    plt.plot(rvsampling, posterior_c,linestyle="-",linewidth=3,color="#ff9900",label="c: Posterior")

    myoutfilename = "CtoO_HR_8799_d_measurements.pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_d",myoutfilename.replace(".pdf","_CtoOposterior.fits")))
    print(hdulist[0].data.shape)
    rvsampling, posterior_d = hdulist[0].data[0,:],hdulist[0].data[1,:]
    plt.gca().text(rv_d[-1]+0.001,1,"${0:.3f}".format(rv_d[-1])+"^{+"+"{0:.3f}".format(rvperr_d[-1])+"}"+"_{-"+"{0:.3f}".format(rvmerr_d[-1])+"}$",ha="center",va="bottom",rotation=0,size=fontsize,color="#330099")
    plt.plot(rvsampling, posterior_d,linestyle="-",linewidth=3,color="#6600ff",label="d: Posterior")

    plt.xlim([0.4570881896148752, 0.6570881896148754])
    plt.ylim([0,1.1])
    plt.xlabel("C/O",fontsize=fontsize)
    plt.ylabel("$\propto \mathcal{P}(C/O|d)$",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",0.4570881896148752))
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
    plt.plot(drvsampling,delta_bc_posterior,linestyle="-",linewidth=3,color="#0099cc",label="Data ($RV_b-RV_c$)")


    delta_dc_posterior = np.correlate(posterior_d,posterior_c,mode="same")
    delta_dc_posterior = delta_dc_posterior/np.max(delta_dc_posterior)
    deltaRV_dc_lCI,deltaRV_dc,deltaRV_dc_rCI,_ = get_err_from_posterior(drvsampling,delta_dc_posterior)
    confidence_interval = (1-np.cumsum(delta_dc_posterior)[np.argmin(np.abs(drvsampling))]/np.sum(delta_dc_posterior))

    plt.gca().text(deltaRV_dc+0.001,1,"${0:.3f}".format(deltaRV_dc)+"^{+"+"{0:.3f}".format(deltaRV_dc_rCI-deltaRV_dc)+"}"+"_{-"+"{0:.3f}".format(deltaRV_dc-deltaRV_dc_lCI)+"}$",ha="center",va="bottom",rotation=0,size=fontsize,color="#660066")
    # plt.gca().text(deltaRV_dc-1,1.0,"${0:.1f}\pm {1:.1f}$ km/s".format(deltaRV_dc,deltaRV_dc_sig),ha="left",va="bottom",rotation=0,size=fontsize,color="#660066")
    plt.plot(drvsampling,delta_dc_posterior,linestyle="-",linewidth=3,color="#6600ff",label="Data ($RV_d-RV_c$)")

    plt.xlim([-0.1,0.1])
    plt.ylim([0,1.1])
    plt.xlabel(r"$RV_{[b,d]}-RV_c$ (km/s)",fontsize=fontsize)
    plt.ylabel("$\propto \mathcal{P}(RV_b-RV_c|d)$",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",0))
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#

    if 1:
        plt.tight_layout()
        print("Saving "+os.path.join(out_pngs,"CtoO_HR_8799_bcd.pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd.pdf"))
        plt.savefig(os.path.join(out_pngs,"CtoO_HR_8799_bcd.png"))
    plt.show()
    exit()

