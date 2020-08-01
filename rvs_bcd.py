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
    if len(x[0:argmax_post]) < 2:
        lx = np.nan
    else:
        lf = interp1d(cum_posterior[0:argmax_post],x[0:argmax_post],bounds_error=False,fill_value=np.nan)
        lx = lf(1-0.6827)
    if len(x[argmax_post::]) < 2:
        rx = np.nan
    else:
        rf = interp1d(cum_posterior[argmax_post::],x[argmax_post::],bounds_error=False,fill_value=np.nan)
        rx = rf(1-0.6827)
    return x[argmax_post],(rx-lx)/2.,argmax_post

if 1:
    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
    fontsize=15
    numbasis_list = np.array([0,1,2])
    # rv_c = [-11.608945710803633, -11.207188499440162, -11.130158122796846]
    # rverr_c = [0.39995957140655475, 0.4005435461700829, 0.40263553957115733]
    # rv_fake_c = [-12.087383546897385, -11.304705751066765, -11.193931216436319]
    # rverr_fake_c =[0.39995957140655475, 0.4005435461700829, 0.40263553957115733]
    # rv_b  =[-9.02651489473455, -9.300812415620943, -9.057521618776493]
    # rverr_b = [0.43534461134695857, 0.43419128603868534, 0.43769025062075956]
    # rv_fake_b = [-9.31206462355225, -9.256584652721518, -9.058596258967563]
    # rverr_fake_b = [0.43534461134695857, 0.43419128603868534, 0.43769025062075956]
    # rv_d = [-19.85512596989514, -13.77195024165619, -15.538716007518406]
    # rverr_d = [1.5544543012391918, 1.6443821722825074, 1.6940156886081097]
    # rv_fake_d = [-16.964360807639505, -13.809828903996385, -15.479359543199587]
    # rverr_fake_d = [1.5544543012391918, 1.6443821722825074, 1.6940156886081097]
    rv_c = [-11.606934489081514, -11.206867811301883, -11.126854475745958]
    rverr_c = [0.391234743860962, 0.3929428817820666, 0.3954140633322645]
    rv_fake_c = [-12.08701450241707, -11.306884480746792, -11.1868644774129]
    rverr_fake_c =[0.41213882476395547, 0.39792877298272966, 0.4059180296709144]
    rv_b = [-8.966494415735955, -9.266544424070677, -9.0265044174029]
    rverr_b = [0.6933120101775065, 0.6719185206191662, 0.6637374898511119]
    rv_fake_b = [-9.266544424070677, -9.226537756292714, -9.0265044174029]
    rverr_fake_b = [0.6636727841071384, 0.6788268543640683, 0.6728046132388119]
    rv_d = [-19.688281380230038, -13.897316219369895, -15.69761626937823]
    rverr_d = [1.7946953690541445, 1.8628739788951396, 1.9279952452381766]
    rv_fake_d = [-16.877812968828138, -13.937322887147857, -15.637606267711284]
    rverr_fake_d = [1.5994389329638912, 1.9054168236452247, 1.9561937931198887]

    rv_b = []
    rv_c = []
    rv_d = []
    rverr_b = []
    rverr_c = []
    rverr_d = []
    for kl in [0,1,10]:
        myoutfilename = "RV_HR_8799_b_measurements_kl10.pdf"
        hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_b",myoutfilename.replace(".pdf","_posterior.fits")))
        rvsampling, posterior_b = hdulist[0].data[0,:],hdulist[0].data[1,:]
        rv,rverr,_ = get_err_from_posterior(rvsampling, posterior_b)
        rv_b.append(rv)
        rverr_b.append(rverr)

        myoutfilename = "RV_HR_8799_c_measurements_kl10.pdf"
        hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_c",myoutfilename.replace(".pdf","_posterior.fits")))
        rvsampling, posterior_c = hdulist[0].data[0,:],hdulist[0].data[1,:]
        rv,rverr,_ = get_err_from_posterior(rvsampling, posterior_c)
        rv_c.append(rv)
        rverr_c.append(rverr)

        myoutfilename = "RV_HR_8799_d_measurements_kl10.pdf"
        hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_d",myoutfilename.replace(".pdf","_posterior.fits")))
        rvsampling, posterior_d = hdulist[0].data[0,:],hdulist[0].data[1,:]
        rv,rverr,_ = get_err_from_posterior(rvsampling, posterior_d)
        rv_d.append(rv)
        rverr_d.append(rverr)


    # b
    # 0
    # yrs_b = ['2009' '2010' '2013' '2016' '2018']
    # [ 1.56573989 -9.64086128 -9.45143865 -8.48559063 -6.2237735 ]
    # [2.21078689 0.73938039 0.71169818 1.02203032 1.39278113]
    # 1
    # ['2009' '2010' '2013' '2016' '2018']
    # [  0.64309642 -10.72815957  -9.57082563  -8.07248319  -6.02885303]
    # [2.213628   0.74100775 0.71718292 1.03405558 1.28690558]
    # 10
    # ['2009' '2010' '2013' '2016' '2018']
    # [  1.53187338 -10.55965046  -9.14796198  -7.74766766  -6.21223367]
    # [2.21987898 0.74417127 0.72278583 1.05386561 1.29136631]

    # 0
    # ['20090722' '20090723' '20090730' '20090903' '20100711' '20100712'
    #  '20100713' '20130725' '20130726' '20130727' '20161106' '20161107'
    #  '20161108' '20180722']
    # [  2.99382078  -9.60521813          nan          nan -10.99510914
    #   -9.4289564   -4.11849687  -9.04536411  -9.41895135 -11.03944654
    #  -10.02574529 -11.61340736  -0.94722271  -6.2237735 ]
    # [2.34785017 6.56657917        nan        nan 1.08834663 1.11180927
    #  2.38372322 0.94274584 1.34820838 1.82888127 2.03924522 1.45565281
    #  2.0205268  1.39278113]
    # 1
    # ['20090722' '20090723' '20090730' '20090903' '20100711' '20100712'
    #  '20100713' '20130725' '20130726' '20130727' '20161106' '20161107'
    #  '20161108' '20180722']
    # [  2.13474987 -10.01255011          nan          nan -10.86880893
    #  -12.2997317   -4.15436169  -9.76136812  -9.19790887  -9.57163102
    #   -7.6423368  -10.67930232  -2.85561965  -6.02885303]
    # [2.3634946  6.31699524        nan        nan 1.08520109 1.13972661
    #  2.22381737 0.96031668 1.34266108 1.81032644 2.18358091 1.42092456
    #  2.0842469  1.28690558]
    # 10
    epochs_b =  ['20100711', '20100712',  '20100713', '20130725', '20130726', '20130727', '20161106', '20161107' , '20161108', '20180722']
    epochs_rv_b =  [  -10.57920756, -12.35982049 , -3.71445932 , -9.53545184,  -8.44582097,  -9.05172061,   -8.23000863,  -9.71148301,  -3.23729904,  -6.21223367]
    epochs_rverr_b = [ 1.09404719, 1.14231799,  2.21445212 ,0.96492429, 1.34422894, 1.86751255, 2.23673811 ,1.45381564,  2.09713865, 1.29136631]
    epochs_Nexp_b = [8100.0, 8100.0, 1800.0, 9600.0, 5400.0, 3000.0, 1200.0, 1800.0, 600.0, 1800.0]#[9, 9, 2, 16, 9, 5, 2, 3, 1, 6]

    # c
    # 0
    # ['2010' '2011' '2013' '2017']
    # [-11.60550704 -10.25192305 -16.57530511 -12.39141603]
    # [0.51252946 0.98597127 3.72706011 0.86264761]
    # 1
    # ['2010' '2011' '2013' '2017']
    # [-11.42815183 -10.45910842 -15.81194633 -10.84169197]
    # [0.50433363 0.98101345 3.66514447 0.91748099]
    # 10
    # ['2010' '2011' '2013' '2017']
    # [-11.4376885  -10.40621871 -15.81194633 -10.47611131]
    # [0.50947933 0.98869494 3.64578072 0.90622151]

    # 0
    # ['20100715' '20101028' '20101104' '20110723' '20110724' '20110725'
    #  '20130726' '20131029' '20131030' '20131031' '20171103']
    # [-12.1241233           nan -10.96171609 -10.66641296  -9.97563245
    #   -9.42989149 -16.57530511          nan          nan          nan
    #  -12.39141603]
    # [0.68869281        nan 0.76731717 1.30977175 2.0911748  2.14637722
    #  3.72706011        nan        nan        nan 0.86264761]
    # 1
    # ['20100715' '20101028' '20101104' '20110723' '20110724' '20110725'
    #  '20130726' '20131029' '20131030' '20131031' '20171103']
    # [-11.83626976          nan -10.95106544 -10.25391634 -10.69299638
    #  -10.74618039 -15.81194633          nan          nan          nan
    #  -10.84169197]
    # [0.68697538        nan 0.74275716 1.31189968 2.08184032 2.09739146
    #  3.66514447        nan        nan        nan 0.91748099]
    # 10
    epochs_c =  ['20100715' , '20101104' ,'20110723' ,'20110724', '20110725','20130726', '20171103','20200729']
    epochs_rv_c = [-11.83538564, -10.95940131, -10.04625721, -10.75684941,-11.01553684 ,-15.81194633,-10.47611131,-10.59403037]
    epochs_rverr_c = [0.68949379 ,   0.75613354 ,1.31100169 ,2.08722546, 2.17375194, 3.64578072,  0.90622151, 1.50326013]
    epochs_Nexp_c = [10200.0,10800.0, 6000.0, 1800.0, 3000.0, 600.0, 3000.0, 2400.0]#[17, 18, 10, 3, 5, 1, 5]


    # d
    # 0
    # ['2015']
    # [-19.85512597]
    # [1.5544543]
    # 1
    # ['2015']
    # [-13.77195024]
    # [1.64438217]
    # 10
    # ['2015']
    # [-15.53871601]
    # [1.69401569]

    # 0
    # ['20130727' '20150720' '20150722' '20150723' '20150828']
    # [         nan -24.18339925          nan -19.07261404 -14.07602613]
    # [       nan 2.51736928        nan 2.50424453 3.21755884]
    # 1
    # ['20130727' '20150720' '20150722' '20150723' '20150828']
    # [         nan -10.15767679          nan -17.01463235 -14.33700864]
    # [       nan 2.65915811        nan 2.64729949 3.41567384]
    # 10
    epochs_d =  ['20150720', '20150723', '20150828','20200729', '20200730', '20200731']
    epochs_rv_d = [-13.81186852, -18.30012132, -13.60937274, -14.42853693 ,-17.69367862, -13.55751892]
    epochs_rverr_d = [2.79074779,2.69400271, 3.48607747,  1.81554366, 2.04721357, 1.17071597]
    epochs_Nexp_d = [ 4800.0, 3000.0, 1800.0, 4800.0, 4800.0, 9600.0]



    allepochs = np.unique(np.concatenate([epochs_b,epochs_c,epochs_d]))
    allepochs_rv_b = np.zeros(allepochs.shape) + np.nan
    allepochs_rverr_b = np.zeros(allepochs.shape) + np.nan
    allepochs_Nexp_b = np.zeros(allepochs.shape) + np.nan
    allepochs_rv_c = np.zeros(allepochs.shape) + np.nan
    allepochs_rverr_c = np.zeros(allepochs.shape) + np.nan
    allepochs_Nexp_c = np.zeros(allepochs.shape) + np.nan
    allepochs_rv_d = np.zeros(allepochs.shape) + np.nan
    allepochs_rverr_d = np.zeros(allepochs.shape) + np.nan
    allepochs_Nexp_d = np.zeros(allepochs.shape) + np.nan
    for epochid,epoch in enumerate(allepochs):
        if epoch in epochs_d:
            wheretmp = np.where(int(epoch)==np.array(epochs_d).astype(np.int))
            if len(wheretmp[0]) != 0:
                wheretmp = wheretmp[0][0]
                allepochs_rv_d[epochid] = epochs_rv_d[wheretmp]
                allepochs_rverr_d[epochid] = epochs_rverr_d[wheretmp]
                allepochs_Nexp_d[epochid] = epochs_Nexp_d[wheretmp]
        if epoch in epochs_c:
            wheretmp = np.where(int(epoch)==np.array(epochs_c).astype(np.int))
            print(epoch,wheretmp)
            if len(wheretmp[0]) != 0:
                wheretmp = wheretmp[0][0]
                allepochs_rv_c[epochid] = epochs_rv_c[wheretmp]
                allepochs_rverr_c[epochid] = epochs_rverr_c[wheretmp]
                allepochs_Nexp_c[epochid] = epochs_Nexp_c[wheretmp]
        if epoch in epochs_b:
            wheretmp = np.where(int(epoch)==np.array(epochs_b).astype(np.int))
            if len(wheretmp[0]) != 0:
                wheretmp = wheretmp[0][0]
                allepochs_rv_b[epochid] = epochs_rv_b[wheretmp]
                allepochs_rverr_b[epochid] = epochs_rverr_b[wheretmp]
                allepochs_Nexp_b[epochid] = epochs_Nexp_b[wheretmp]
    # exit()

    def convertdates(date_list):
        return [date[4:6]+"-"+date[6:8]+"-"+date[0:4] for date in date_list ]

    plt.figure(1,figsize=(4,3.7))
    if 1:
        plt.plot(rv_b,linestyle="",color="#0099cc",marker="x") #"#ff9900" "#0099cc" "#6600ff"
        plt.errorbar(numbasis_list,rv_b,yerr=rverr_b,color="#0099cc",label="b: RV")
        # plt.plot(numbasis_list+0.02,rv_fake_b,linestyle="",color="#0099cc",marker="x")
        # eb = plt.errorbar(numbasis_list+0.02,rv_fake_b,yerr=rverr_fake_b,color="#0099cc",fmt="",linestyle="--",label="c: corrected RV")
        # eb[-1][0].set_linestyle("--")
    if 1:
        plt.plot(numbasis_list,rv_c,linestyle="",color="#ff9900",marker="x") #"#ff9900" "#0099cc" "#6600ff"
        plt.errorbar(numbasis_list,rv_c,yerr=rverr_c,color="#ff9900",label="c: RV")
        # plt.plot(numbasis_list+0.02,rv_fake_c,linestyle="",color="#ff9900",marker="x")
        # eb = plt.errorbar(numbasis_list+0.02,rv_fake_c,yerr=rverr_fake_c,color="#ff9900",fmt="",linestyle="--",label="c: corrected RV")
        # eb[-1][0].set_linestyle("--")
    if 1:
        plt.plot(numbasis_list,rv_d,linestyle="",color="#6600ff",marker="x") #"#ff9900" "#0099cc" "#6600ff"
        plt.errorbar(numbasis_list,rv_d,yerr=rverr_d,color="#6600ff",label="d: RV")
        # plt.plot(numbasis_list+0.02,rv_fake_d,linestyle="",color="#6600ff",marker="x")
        # eb = plt.errorbar(numbasis_list+0.02,rv_fake_d,yerr=rverr_fake_d,color="#6600ff",fmt="",linestyle="--",label="c: corrected RV")
        # eb[-1][0].set_linestyle("--")


    plt.legend(loc="lower right",frameon=True,fontsize=fontsize)
    plt.xlabel("# PCA modes",fontsize=fontsize)
    plt.xticks([0,1,2],[0,1,10])
    plt.ylabel("RV (km/s)",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    # plt.gca().spines["right"].set_visible(False)
    # plt.gca().spines["top"].set_visible(False)
    plt.tight_layout()
    if 1:
        print("Saving "+os.path.join(out_pngs,"RV_HR_8799_bcd_PCA.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bcd_PCA.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bcd_PCA.png"))
    # plt.show()

    plt.figure(2,figsize=(12,12))
    plt.subplot2grid((4,1),(0,0),rowspan=2)
    # print([0,np.size(allepochs)],rv_d[2]-rverr_d[2],rv_d[2]+rverr_d[2])
    # plt.fill_betweenx([0,np.size(allepochs)],rv_d[2]-rverr_d[2],rv_d[2]+rverr_d[2],alpha=0.5,color="6600cc")
    # plt.show()
    plt.fill_betweenx([0,np.size(allepochs)],rv_b[2]-rverr_b[2],rv_b[2]+rverr_b[2],alpha=0.2,color="#006699")
    plt.fill_betweenx([0,np.size(allepochs)],rv_c[2]-rverr_c[2],rv_c[2]+rverr_c[2],alpha=0.2,color="#cc6600")
    plt.fill_betweenx([0,np.size(allepochs)],rv_d[2]-rverr_d[2],rv_d[2]+rverr_d[2],alpha=0.2,color="#6600cc")

    print("bonjour",epochs_rv_b,epochs_rverr_b,epochs_b,epochs_Nexp_b)
    print("Planet & Date & RV & N cubes \\\\")
    for a,b,c,d in zip(epochs_rv_b,epochs_rverr_b,epochs_b,epochs_Nexp_b):
        if np.isnan(a):
            continue
        formated_date =  c[0:4]+"-"+c[4:6]+"-"+c[6:8]
        print("& {0} & ${1:.1f} \\pm {2:.1f}$ & {3} \\\\".format(formated_date,a,b,int(d)//60))
    eb=plt.errorbar(allepochs_rv_b,np.arange(np.size(allepochs)),xerr=allepochs_rverr_b,fmt="none",color="#0099cc",label="b w/ exp. time (min)")
    eb[-1][0].set_linestyle("-")
    plt.plot(allepochs_rv_b,np.arange(np.size(allepochs)),"x",color="#0099cc")
    wherenotnans = np.where(np.isfinite(allepochs_rv_b))
    for y,(x,date,num) in enumerate(zip(allepochs_rv_b,allepochs,allepochs_Nexp_b)):
        if np.isfinite(x):
            plt.gca().text(x,y,"{0}".format(int(num)//60),ha="center",va="bottom",rotation=0,size=fontsize,color="#003366",alpha=1)
    # plt.plot([rv_b[2]-rverr_b[2],rv_b[2]-rverr_b[2]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#006699",alpha=0.4)
    plt.plot([rv_b[2],rv_b[2]],[0,np.size(allepochs)],linestyle="-",linewidth=2,color="#003366",alpha=0.4)
    # plt.plot([rv_b[2]+rverr_b[2],rv_b[2]+rverr_b[2]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#006699",alpha=0.4)

    print("bonjour",epochs_rv_c,epochs_rverr_c,epochs_c,epochs_Nexp_c)
    print("Planet & Date & RV & N cubes \\\\")
    for a,b,c,d in zip(epochs_rv_c,epochs_rverr_c,epochs_c,epochs_Nexp_c):
        if np.isnan(a):
            continue
        formated_date =  c[0:4]+"-"+c[4:6]+"-"+c[6:8]
        print("& {0} & ${1:.1f} \\pm {2:.1f}$ & {3} \\\\".format(formated_date,a,b,int(d)//60))
    eb=plt.errorbar(allepochs_rv_c,np.arange(np.size(allepochs)),xerr=allepochs_rverr_c,fmt="none",color="#ff9900",label="c")
    eb[-1][0].set_linestyle("--")
    plt.plot(allepochs_rv_c,np.arange(np.size(allepochs)),"x",color="#ff9900")
    wherenotnans = np.where(np.isfinite(allepochs_rv_c))
    for y,(x,date,num) in enumerate(zip(allepochs_rv_c,allepochs,allepochs_Nexp_c)):
        if np.isfinite(x):
            plt.gca().text(x,y,"{0}".format(int(num)//60),ha="center",va="bottom",rotation=0,size=fontsize,color="#cc3300",alpha=1)
    # plt.plot([rv_c[2]-rverr_c[2],rv_c[2]-rverr_c[2]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#cc6600",alpha=0.4)
    plt.plot([rv_c[2],rv_c[2]],[0,np.size(allepochs)],linestyle="-",linewidth=2,color="#cc3300",alpha=0.4)
    # plt.plot([rv_c[2]+rverr_c[2],rv_c[2]+rverr_c[2]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#cc6600",alpha=0.4)

    print("bonjour",epochs_rv_d,epochs_rverr_d,epochs_d,epochs_Nexp_d)
    print("Planet & Date & RV & N cubes \\\\")
    for a,b,c,d in zip(epochs_rv_d,epochs_rverr_d,epochs_d,epochs_Nexp_d):
        if np.isnan(a):
            continue
        formated_date =  c[0:4]+"-"+c[4:6]+"-"+c[6:8]
        print("& {0} & ${1:.1f} \\pm {2:.1f}$ & {3} \\\\".format(formated_date,a,b,int(d)//60))
    eb=plt.errorbar(allepochs_rv_d,np.arange(np.size(allepochs)),xerr=allepochs_rverr_d,fmt="none",color="#6600ff",label="d")
    eb[-1][0].set_linestyle(":")
    plt.plot(allepochs_rv_d,np.arange(np.size(allepochs)),"x",color="#6600ff")
    wherenotnans = np.where(np.isfinite(allepochs_rv_d))
    for y,(x,date,num) in enumerate(zip(allepochs_rv_d,allepochs,allepochs_Nexp_d)):
        if np.isfinite(x):
            plt.gca().text(x,y,"{0}".format(int(num)//60),ha="center",va="bottom",rotation=0,size=fontsize,color="#330099",alpha=1)
    # plt.plot([rv_d[2]-rverr_d[2],rv_d[2]-rverr_d[2]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#6600cc",alpha=0.4)
    plt.plot([rv_d[2],rv_d[2]],[0,np.size(allepochs)],linestyle="-",linewidth=2,color="#330099",alpha=0.4)
    # plt.plot([rv_d[2]+rverr_d[2],rv_d[2]+rverr_d[2]],[0,np.size(allepochs)],linestyle="--",linewidth=2,color="#6600cc",alpha=0.4)

    plt.xlim([-20,0])
    plt.xlabel("RV (km/s)",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.yticks(np.arange(0,np.size(allepochs)),convertdates(allepochs))
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",-20))
    plt.legend(loc="center right",frameon=True,fontsize=fontsize)#


    if 0:
        print("Saving "+os.path.join(out_pngs,"RV_HR_8799_bcd_epochs.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bcd_epochs.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bcd_epochs.png"))


    # plt.figure(3,figsize=(12,3))
    plt.subplot2grid((4,1),(2,0),rowspan=1)
    myoutfilename = "RV_HR_8799_b_measurements_kl10.pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_b",myoutfilename.replace(".pdf","_posterior.fits")))
    print(hdulist[0].data.shape)
    rvsampling, posterior_b = hdulist[0].data[0,:],hdulist[0].data[1,:]
    plt.gca().text(rv_b[2]+0.25,1,"${0:.1f}\pm {1:.1f}$ km/s".format(rv_b[2],rverr_b[2]),ha="left",va="bottom",rotation=0,size=fontsize,color="#003366")
    plt.plot(rvsampling, posterior_b,linestyle="-",linewidth=3,color="#0099cc",label="b")

    myoutfilename = "RV_HR_8799_c_measurements_kl10.pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_c",myoutfilename.replace(".pdf","_posterior.fits")))
    print(hdulist[0].data.shape)
    rvsampling, posterior_c = hdulist[0].data[0,:],hdulist[0].data[1,:]
    plt.gca().text(rv_c[2]+0.25,1,"${0:.1f}\pm {1:.1f}$ km/s".format(rv_c[2],rverr_c[2]),ha="center",va="bottom",rotation=0,size=fontsize,color="#cc3300")
    plt.plot(rvsampling, posterior_c,linestyle="--",linewidth=3,color="#ff9900",label="c")

    myoutfilename = "RV_HR_8799_d_measurements_kl10.pdf"
    hdulist = pyfits.open(os.path.join(out_pngs,"HR_8799_d",myoutfilename.replace(".pdf","_posterior.fits")))
    print(hdulist[0].data.shape)
    rvsampling, posterior_d = hdulist[0].data[0,:],hdulist[0].data[1,:]
    plt.gca().text(rv_d[2]+0.25,1,"${0:.1f}\pm {1:.1f}$ km/s".format(rv_d[2],rverr_d[2]),ha="center",va="bottom",rotation=0,size=fontsize,color="#330099")
    plt.plot(rvsampling, posterior_d,linestyle=":",linewidth=3,color="#6600ff",label="d")

    plt.xlim([-20,0])
    plt.ylim([0,1.1])
    plt.xlabel("RV (km/s)",fontsize=fontsize)
    plt.ylabel("$\propto \mathcal{P}(RV|d)$",fontsize=fontsize)
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",-20))
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = handles#handles[0:3]+[handles[6]]+handles[3:6]+[handles[7]]
    new_labels = labels#labels[0:3]+[labels[6]]+labels[3:6]+[labels[7]]
    # print(handles)
    # print(labels)
    # exit()
    plt.legend(new_handles,new_labels,loc="upper right",frameon=True,fontsize=fontsize)#

    if 0:
        print("Saving "+os.path.join(out_pngs,"RV_HR_8799_bcd_posterior.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bcd_posterior.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bcd_posterior.png"))


    # plt.figure(4,figsize=(12,3))
    plt.subplot2grid((4,1),(3,0),rowspan=1)
    # with open("/data/osiris_data/jason_rv_bc_2011.csv", 'r') as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=';')
    #     dRV_list_table = list(csv_reader)
    #     dRV_list_data = np.array(dRV_list_table[1::]).astype(np.float)
    #     print("Jason",np.mean(np.abs(dRV_list_data)),np.std(np.abs(dRV_list_data)))
    #     dRV_list_data = np.concatenate([dRV_list_data,-dRV_list_data])
    with open("/data/osiris_data/rv_bcd_2015.csv", 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        rv_bcd_list_table = np.array(list(csv_reader)[1::]).astype(np.float)
        rv_bcd_list_table = np.concatenate([rv_bcd_list_table,-rv_bcd_list_table],axis=0)
        rv_bcd_list_table = rv_bcd_list_table[np.where(rv_bcd_list_table[:,0]>0)[0],:]
        rv_bmc_list_data = rv_bcd_list_table[:,0]-rv_bcd_list_table[:,1]
        rv_dmc_list_data = rv_bcd_list_table[:,2]-rv_bcd_list_table[:,1]
    # rv_bmc_list_data = np.concatenate([rv_bmc_list_data,-rv_bmc_list_data])
    rv_bmc_hist,bin_edges = np.histogram(rv_bmc_list_data,bins=400,range=[-20,20])
    rv_bmc_hist = rv_bmc_hist/np.max(rv_bmc_hist)
    # plt.figure(10)
    # plt.plot(bin_edges[1::],rv_bmc_hist)
    # plt.show()
    bin_center = (bin_edges[1::]+bin_edges[0:np.size(bin_edges)-1])/2
    rv_bmc_post = interp1d(bin_center,rv_bmc_hist,bounds_error=False,fill_value=0)(rvsampling)
    # plt.plot(final_planetRV_hd,dRV_posterior,linestyle="-",linewidth=1,color="black") #9966ff
    plt.fill_between(rvsampling,
                     rv_bmc_post*0,
                     rv_bmc_post,alpha=0.4,facecolor="none",edgecolor="#006699",label="Wang et al. 2018 ($RV_b-RV_c$)",hatch="\\") # (Astrometry; bcde coplanar & stable)
    # rv_dmc_list_data = np.concatenate([rv_dmc_list_data,-rv_dmc_list_data])
    rv_dmc_hist,bin_edges = np.histogram(rv_dmc_list_data,bins=200,range=[-20,20])
    rv_dmc_hist = rv_dmc_hist/np.max(rv_dmc_hist)
    bin_center = (bin_edges[1::]+bin_edges[0:np.size(bin_edges)-1])/2
    rv_dmc_post = interp1d(bin_center,rv_dmc_hist,bounds_error=False,fill_value=0)(rvsampling)
    # plt.plot(final_planetRV_hd,dRV_posterior,linestyle="-",linewidth=1,color="black") #9966ff
    plt.fill_between(rvsampling,
                     rv_dmc_post*0,
                     rv_dmc_post,alpha=0.4,facecolor="none",edgecolor="#6600ff",label="Wang et al. 2018 ($RV_d-RV_c$)",hatch="/") # (Astrometry; bcde coplanar & stable)


    delta_bc_posterior = np.correlate(posterior_b,posterior_c,mode="same")
    delta_bc_posterior = delta_bc_posterior/np.max(delta_bc_posterior)
    deltaRV_bc,deltaRV_bc_sig,_ = get_err_from_posterior(rvsampling,delta_bc_posterior)
    confidence_interval = (1-np.cumsum(delta_bc_posterior)[np.argmin(np.abs(rvsampling))]/np.sum(delta_bc_posterior))

    plt.gca().text(deltaRV_bc-1,1.0,"${0:.1f}\pm {1:.1f}$ km/s".format(deltaRV_bc,deltaRV_bc_sig),ha="left",va="bottom",rotation=0,size=fontsize,color="#003366")
    plt.plot(rvsampling,delta_bc_posterior,linestyle="-",linewidth=3,color="#0099cc",label="This work ($RV_b-RV_c$)")


    delta_dc_posterior = np.correlate(posterior_d,posterior_c,mode="same")
    delta_dc_posterior = delta_dc_posterior/np.max(delta_dc_posterior)
    deltaRV_dc,deltaRV_dc_sig,_ = get_err_from_posterior(rvsampling,delta_dc_posterior)
    confidence_interval = (1-np.cumsum(delta_dc_posterior)[np.argmin(np.abs(rvsampling))]/np.sum(delta_dc_posterior))

    plt.gca().text(deltaRV_dc-1,1.0,"${0:.1f}\pm {1:.1f}$ km/s".format(deltaRV_dc,deltaRV_dc_sig),ha="left",va="bottom",rotation=0,size=fontsize,color="#660066")
    plt.plot(rvsampling,delta_dc_posterior,linestyle=":",linewidth=3,color="#6600ff",label="This work ($RV_d-RV_c$)")

    plt.xlim([-10,10])
    plt.ylim([0,1.1])
    plt.yticks([0.25,0.5,0.75,1.0])
    plt.xlabel(r"$RV_{[b,d]}-RV_c$ (km/s)",fontsize=fontsize)
    plt.ylabel("$\propto \mathcal{P}(RV_b-RV_c|d)$",fontsize=fontsize)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_position(("data",0))
    plt.tick_params(axis="x",labelsize=fontsize)
    plt.tick_params(axis="y",labelsize=fontsize)
    plt.legend(loc="center right",frameon=True,fontsize=12)#

    if 0:
        print("Saving "+os.path.join(out_pngs,"RV_HR_8799_bcd_relRVpost.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bcd_relRVpost.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bcd_relRVpost.png"))
    if 1:
        plt.tight_layout()
        print("Saving "+os.path.join(out_pngs,"RV_HR_8799_bcd.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bcd.pdf"))
        plt.savefig(os.path.join(out_pngs,"RV_HR_8799_bcd.png"))

    plt.show()
    exit()

    # plt.subplot(3,1,2)
    # plt.gca().text(c_combined_avg+0.25,1,"${0:.1f}\pm {1:.1f}$ km/s".format(c_combined_avg,c_combined_sig),ha="center",va="bottom",rotation=0,size=fontsize,color="#cc3300")
    # plt.plot(final_planetRV_hd,c_posterior,linestyle="-",linewidth=3,color="#ff9900",label="c: Posterior")
    #
    # plt.gca().text(b_combined_avg-0.7,1,"${0:.1f}\pm {1:.1f}$ km/s".format(b_combined_avg,b_combined_sig),ha="left",va="bottom",rotation=0,size=fontsize,color="#003366")
    # plt.plot(final_planetRV_hd,b_posterior,linestyle="-",linewidth=3,color="#0099cc",label="b: Posterior")
    #
    # plt.xlim([rv_star-4,rv_star+12])
    # plt.ylim([0,1.1])
    # plt.xlabel("RV (km/s)",fontsize=fontsize)
    # plt.ylabel("$\propto \mathcal{P}(RV|d)$",fontsize=fontsize)
    # plt.tick_params(axis="x",labelsize=fontsize)
    # plt.tick_params(axis="y",labelsize=fontsize)
    # plt.gca().spines["right"].set_visible(False)
    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["left"].set_position(("data",rv_star))
    # handles, labels = plt.gca().get_legend_handles_labels()
    # new_handles = handles#handles[0:3]+[handles[6]]+handles[3:6]+[handles[7]]
    # new_labels = labels#labels[0:3]+[labels[6]]+labels[3:6]+[labels[7]]
    # # print(handles)
    # # print(labels)
    # # exit()
    # plt.legend(new_handles,new_labels,loc="upper right",frameon=True,fontsize=fontsize)#

    # plt.subplot(3,1,3)
    # with open("/data/osiris_data/jason_rv_bc_2011.csv", 'r') as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=';')
    #     dRV_list_table = list(csv_reader)
    #     dRV_list_data = np.array(dRV_list_table[1::]).astype(np.float)
    #     print("Jason",np.mean(np.abs(dRV_list_data)),np.std(np.abs(dRV_list_data)))
    #     dRV_list_data = np.concatenate([dRV_list_data,-dRV_list_data])
    # dRV_hist,bin_edges = np.histogram(dRV_list_data,bins=400,range=[-20,20])
    # dRV_hist = dRV_hist/np.max(dRV_hist)
    # bin_center = (bin_edges[1::]+bin_edges[0:np.size(bin_edges)-1])/2
    # dRV_posterior = interp1d(bin_center,dRV_hist,bounds_error=False,fill_value=0)(final_planetRV_hd)
    # # plt.plot(final_planetRV_hd,dRV_posterior,linestyle="-",linewidth=1,color="black") #9966ff
    # plt.fill_between(final_planetRV_hd,
    #                  dRV_posterior*0,
    #                  dRV_posterior,alpha=0.2,color="grey",label="Wang 2018 (Astrometry; bcde coplanar & stable)")
    #
    #
    # astrometry_DATADIR = os.path.join("/data/osiris_data","astrometry")
    # userv = "includingrvdata"
    # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_bc",'rvs_diffbc_55392_{0}.fits'.format(userv))) as hdulist:
    #     diffrvs = hdulist[0].data
    # diffrvs_post,xedges = np.histogram(diffrvs,bins=10*5,range=[-5,5])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # plt.plot(x_centers,diffrvs_post/np.max(diffrvs_post),linestyle="--",linewidth=1,color="#6600ff",label="Orbit fit (Astrometry & RV; bc coplanar)") #9966ff
    # userv = "norvdata"
    # with pyfits.open(os.path.join(astrometry_DATADIR,"figures","HR_8799_bc",'rvs_diffbc_55392_{0}.fits'.format(userv))) as hdulist:
    #     diffrvs = hdulist[0].data
    #     diffrvs = np.concatenate([diffrvs,-diffrvs])
    # diffrvs_post,xedges = np.histogram(diffrvs,bins=10*5,range=[-5,5])
    # x_centers = [(x1+x2)/2. for x1,x2 in zip(xedges[0:len(xedges)-1],xedges[1:len(xedges)])]
    # plt.plot(x_centers,diffrvs_post/np.max(diffrvs_post),linestyle=":",linewidth=1,color="black",label="Orbit fit (Astrometry; bc coplanar)") #9966ff
    #
    # delta_posterior = np.correlate(b_posterior,c_posterior,mode="same")
    # delta_posterior = delta_posterior/np.max(delta_posterior)
    # delta_pessimistic_posterior = np.correlate(b_pessimistic_posterior,c_pessimistic_posterior,mode="same")
    # delta_pessimistic_posterior = delta_pessimistic_posterior/np.max(delta_pessimistic_posterior)
    # deltaRV,deltaRV_sig,_ = get_err_from_posterior(final_planetRV_hd,delta_posterior)
    # deltaRV_pess,deltaRV_sig_pess,_ = get_err_from_posterior(final_planetRV_hd,delta_pessimistic_posterior)
    # confidence_interval = (1-np.cumsum(delta_posterior)[np.argmin(np.abs(final_planetRV_hd))]/np.sum(delta_posterior))
    # plt.gca().text(deltaRV-1,1.0,"${0:.1f}\pm {1:.1f}$ km/s".format(deltaRV,deltaRV_sig),ha="left",va="bottom",rotation=0,size=fontsize,color="#660066")
    # plt.plot(final_planetRV_hd,delta_posterior,linestyle="-",linewidth=3,color="#6600ff",label="Data only (RV)") #9966ff
    #
    #
    # plt.xlim([-4,12])
    # plt.ylim([0,1.1])
    # plt.xlabel(r"$RV_b-RV_c$ (km/s)",fontsize=fontsize)
    # plt.ylabel("$\propto \mathcal{P}(RV_b-RV_c|d)$",fontsize=fontsize)
    # plt.gca().spines["right"].set_visible(False)
    # plt.gca().spines["top"].set_visible(False)
    # plt.gca().spines["left"].set_position(("data",0))
    # plt.tick_params(axis="x",labelsize=fontsize)
    # plt.tick_params(axis="y",labelsize=fontsize)
    # plt.legend(loc="upper right",frameon=True,fontsize=fontsize)#

    plt.show()
    exit()

