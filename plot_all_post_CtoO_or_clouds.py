__author__ = 'jruffio'

import matplotlib.pyplot as plt
import os
import astropy.io.fits as pyfits
import numpy as np

#------------------------------------------------
if __name__ == "__main__":

    fontsize = 12
    gridname = "hr8799b_modelgrid"
    # gridname = "clouds_modelgrid"
    out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"

    if "hr8799b_modelgrid" in gridname:
        xlabel_list = ["T [K]", "log(g/[1 cm/$\mathrm{s}^2$])","C/O"]
        xticks_list = [[800,900,1000,1100,1200], [3.0,3.5,4.0,4.5],[0.4,0.5,0.6,0.7,0.8,0.9]]
    elif "clouds_modelgrid" in gridname:
        xlabel_list = ["T [K]", "log(g/[1 cm/$\mathrm{s}^2$])","pgs"]
        xticks_list = [[800,900,1000,1100,1200,1300], [3.0,3.5,4.0,4.5,5.0],[5e5,1e6,4e6]]

    f1,ax_post_list = plt.subplots(3,3,sharey="row",figsize=(12,12*4/5))#,sharex="col"
    planet_list = ["b","c","d"]
    linestyle_list = [":","--","-","-.",":",(0,(3,5,1,5,1,5))]

    plt.figure(f1.number)
    for plid,(planet,color) in enumerate(zip(planet_list,["#0099cc","#ff9900","#6600ff"])):
        post_filelist = []
        label_list = []
        post_filelist.append("/home/sda/jruffio/pyOSIRIS/figures/HR_8799_"+planet+"/HR_8799_"+planet+"_"+gridname+"_fit_photo_mags_posterior.fits")
        label_list.append("Photometry")
        post_filelist.append("/home/sda/jruffio/pyOSIRIS/figures/HR_8799_"+planet+"/HR_8799_"+planet+"_"+gridname+"_fit_lowresspec_posterior.fits")
        label_list.append("Low resolution spectra")
        # post_filelist.append("/home/sda/jruffio/pyOSIRIS/figures/HR_8799_"+planet+"/HR_8799_"+planet+"_"+gridname+"_fit_lowresspec_scalefac_posterior.fits")
        # label_list.append("Low resolution spectra w\ scaling factor for the noise")
        if "hr8799b_modelgrid" in gridname:
            post_filelist.append("/home/sda/jruffio/pyOSIRIS/figures/HR_8799_"+planet+"/CtoO_HR_8799_"+planet+"_measurements_kl10_HK_posterior.fits")
            label_list.append("Forward model OSIRIS")
            # post_filelist.append("/home/sda/jruffio/pyOSIRIS/figures/HR_8799_"+planet+"/CtoO_HR_8799_"+planet+"_measurements_kl10_HK_best10SNR_posterior.fits")
            # label_list.append("Forward model OSIRIS (best 10 exposures)")
        elif "clouds_modelgrid" in gridname:
            post_filelist.append("/home/sda/jruffio/pyOSIRIS/figures/HR_8799_"+planet+"/clouds_HR_8799_"+planet+"_measurements_kl10_HK_posterior.fits")
            label_list.append("Forward model OSIRIS")
            # post_filelist.append("/home/sda/jruffio/pyOSIRIS/figures/HR_8799_"+planet+"/clouds_HR_8799_"+planet+"_measurements_kl10_HK_best10SNR_posterior.fits")
            # label_list.append("Forward model OSIRIS (best 10 exposures)")
        # post_filelist.append("/home/sda/jruffio/pyOSIRIS/figures/HR_8799_"+planet+"/HR_8799_"+planet+"_"+gridname+"_fit_OSIRISspec_Kbb_posterior.fits")
        # label_list.append("Extracted spectrum OSIRIS high-pass filtered")

        # color_list =  ["black","black","grey",color,color]
        color_list =  ["grey","black",color]
        for postid,(linestyle,postfilename,label,_color) in enumerate(zip(linestyle_list,post_filelist,label_list,color_list)):

            hdulist = pyfits.open(postfilename)
            post = hdulist[0].data
            fitT_list = hdulist[1].data
            fitlogg_list = hdulist[2].data
            fitpara_list = hdulist[3].data
            # print(post.shape,fitlogg_list)
            # continue

            # maxids = np.unravel_index(np.argmax(post),post.shape)
            # if postid <= 1:
            #     fluxes = hdulist[4].data
            #     if "Photometry" == label:
            #         print(planet,label,
            #               fitT_list[maxids[0]],fitlogg_list[maxids[1]],fitpara_list[maxids[2]],
            #               fluxes[maxids],10**(fluxes[maxids]/-2.5))
            #     else:
            #         print(planet,label,
            #               fitT_list[maxids[0]],fitlogg_list[maxids[1]],fitpara_list[maxids[2]],
            #               fluxes[maxids])
            # else:
            #     print(planet,label,fitT_list[maxids[0]],fitlogg_list[maxids[1]],fitpara_list[maxids[2]],fluxes[maxids])
            # # exit()

            for paraid,(xvec,xticks) in enumerate(zip([fitT_list,fitlogg_list,fitpara_list],xticks_list)):
                plt.sca(ax_post_list[paraid][plid])

                Nparas = 3
                dims = np.arange(Nparas).tolist()
                dims.pop(paraid)
                tmppost = np.sum(post,axis=(*dims,))
                tmppost /= np.max(tmppost)

                plt.plot(xvec,tmppost,linestyle=linestyle,color=_color,label=label)
                if postid == 0:
                    if plid == 0:
                        plt.xticks(xticks)
                    else:
                        plt.xticks(xticks[1::])
                    plt.gca().annotate(planet,xy=(xvec[0],0.95),va="top",ha="left",fontsize=fontsize,color=color)
                    # plt.gca().tick_params(axis='c', labelsize=fontsize)
                    plt.gca().tick_params(axis='x', labelsize=fontsize)
                    plt.gca().tick_params(axis='y', labelsize=fontsize)

    for paraid, xlabel in enumerate(xlabel_list):
        plt.sca(ax_post_list[paraid][0])
        plt.xlabel(xlabel,fontsize=fontsize)
        plt.ylabel("Posterior ",fontsize=fontsize)
        plt.gca().tick_params(axis='y', labelsize=fontsize)

    # plt.sca(ax_post_list[-1][0])
    # plt.ylabel("Posteriors",fontsize=15)
    # plt.gca().tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    f1.subplots_adjust(wspace=0)#,hspace=0

    plt.sca(ax_post_list[0][0])
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0],[0],color="black",lw=1,linestyle=linestyle) for linestyle in linestyle_list]
    legend_list = []
    lgd = plt.legend(custom_lines,label_list,loc="lower left",bbox_to_anchor=(0,1),frameon=False,fontsize=fontsize*0.9,ncol=1)#loc="lower right"
    legend_list.append(lgd)

    if "clouds_modelgrid" in gridname:
        for plid in range(3):
            plt.sca(ax_post_list[2][plid])
            plt.xticks(xticks,["5e5","1e6","4e6"])

    print("Saving "+os.path.join(out_pngs,"HR8799bcd_"+gridname+"_all_post.png"))
    plt.savefig(os.path.join(out_pngs,"HR8799bcd_"+gridname+"_all_post.png"),bbox_inches='tight',bbox_extra_artists=legend_list)
    plt.savefig(os.path.join(out_pngs,"HR8799bcd_"+gridname+"_all_post.pdf"),bbox_inches='tight',bbox_extra_artists=legend_list)

    plt.show()

