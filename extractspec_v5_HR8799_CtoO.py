__author__ = 'jruffio'

import glob
import os
import csv
import astropy.io.fits as pyfits
import numpy as np
import multiprocessing as mp
from reduce_HPFonly_diagcov_resmodel_v2 import convolve_spectrum
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
    planet = "HR_8799_b"
    # planet = "HR_8799_c"
    # planet = "HR_8799_d"
    # date = "2010*"
    cutoff = 40
    fontsize = 15
    fakes = True
    R=4000
    c_kms = 299792.458
    resnumbasis = 10

    if 1: #"Kbb"
        IFSfilter = "Kbb"
        osiris_data_dir = "/data/osiris_data/"
        phoenix_folder = os.path.join(osiris_data_dir,"phoenix")
        planet_template_folder = os.path.join(osiris_data_dir,"planets_templates")
        molecular_template_folder = os.path.join(osiris_data_dir,"molecular_templates")
        sky_transmission_folder = os.path.join(osiris_data_dir,"sky_transmission")
        fileinfos_refstars_filename = os.path.join(osiris_data_dir,"fileinfos_refstars_jb.csv")

        f1 = plt.figure(1,figsize=(18,12))
        legend_list = []

        for planet_id,(color,planet,mylim) in enumerate(zip(["#0099cc","#ff9900","#6600ff"],["HR_8799_b","HR_8799_c","HR_8799_d"],[0.05,0.1,0.2])):
            filename = os.path.join(out_pngs,planet+"_spec_"+"wvs"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
            with pyfits.open(filename) as hdulist:
                bincenter = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_spec"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
            with pyfits.open(filename) as hdulist:
                final_spec = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_spec_"+"model"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
            with pyfits.open(filename) as hdulist:
                final_model_arr = hdulist[0].data
            filename = os.path.join(out_pngs,planet+"_specstd"+"_kl{0}_{1}.fits".format(resnumbasis,IFSfilter))
            with pyfits.open(filename) as hdulist:
                final_spec_std = hdulist[0].data
            if planet_id == 0:
                xmin = np.min(bincenter[np.where(np.isfinite(final_spec))])
                xmax = np.max(bincenter[np.where(np.isfinite(final_spec))])

            if planet == "HR_8799_d":
                dlogg = "-0.5"
            else:
                dlogg = "+0.5"
            for model_id,model_label in zip(range(4),["Best fit","T-200K","logg"+dlogg,"C/O+0.2"]):
                plt.subplot2grid((12,1),(planet_id*4+model_id,0),rowspan=1)
                # if planet == "HR_8799_d":
                #     from scipy.signal import medfilt
                #     final_spec = medfilt(final_spec,kernel_size=3)
                plt.plot(bincenter,final_spec, linestyle="-",label="{0}".format(planet.replace("_"," ")),color=color,linewidth=1.5) #["#0099cc","#ff9900","#6600ff"]
                final_model = final_model_arr[model_id,:]
                final_model = (final_model*np.nansum(final_model*final_spec)/np.nansum(final_model**2))
                plt.plot(bincenter,final_model,linestyle="--",color="black",label=model_label,linewidth=1)
                plt.ylim([-mylim,mylim])
                lgd = plt.legend(loc="upper left",bbox_to_anchor=(1,1),frameon=False,fontsize=fontsize*0.9,ncol=1)
                legend_list.append(lgd)
                plt.xlim([xmin,xmax])
                plt.yticks([0,mylim])
                plt.tick_params(axis="x",labelsize=fontsize)
                plt.tick_params(axis="y",labelsize=fontsize)

        # plt.tight_layout()
        f1.subplots_adjust(wspace=0,hspace=0)#
        print("Saving "+os.path.join(out_pngs,"HR8799bcd_modelcomparison_kl{0}.png".format(resnumbasis)))
        plt.savefig(os.path.join(out_pngs,"HR8799bcd_modelcomparison_kl{0}.png".format(resnumbasis)),bbox_inches='tight',bbox_extra_artists=legend_list)
        plt.savefig(os.path.join(out_pngs,"HR8799bcd_modelcomparison_kl{0}.pdf".format(resnumbasis)),bbox_inches='tight',bbox_extra_artists=legend_list)
        plt.show()


        #         plt.gca().tick_params(axis='x', labelsize=fontsize)
        #         plt.gca().tick_params(axis='y', labelsize=fontsize)
        #         # plt.ylim([-0.1,0.1])
        #         plt.ylim([-mylim,mylim])
        #         # plt.ylim([-0.5,0.5])
        #         # if sp_id==0:
        #         #     plt.legend(loc="lower right",frameon=True,fontsize=fontsize)
        #
        #         plt.subplot2grid((7*2,1),(7*sp_id+6,0))
        #         plt.plot([-1],[-1])
        #         plt.xlim([0,1])
        #         plt.ylim([0,1])
        #         plt.tick_params(axis="x",which="both",labelleft=False,right=False,left=False)
        #         plt.tick_params(axis="y",which="both",labelleft=False,right=False,left=False)
        #         plt.xticks([])
        #         plt.yticks([])
        #         plt.gca().spines["right"].set_visible(False)
        #         plt.gca().spines["left"].set_visible(False)
        #         plt.gca().spines["top"].set_visible(False)
        #         plt.gca().spines["bottom"].set_visible(False)
        #         plt.gca().patch.set_alpha(0)
        # # plt.show()
        # # plt.plot(bincenter,final_model_notrans,linestyle="-",color="grey",label="{0}: final_model_notrans".format(resnumbasis))
        # plt.tight_layout()
        # f1.subplots_adjust(wspace=0,hspace=0)
        # # plt.show()
        # print("Saving "+os.path.join(out_pngs,planet,planet+"_"+IFSfilter+"_spec_kl{0}_{1}.png".format(resnumbasis,IFSfilter)))
        # plt.savefig(os.path.join(out_pngs,planet,planet+"_"+IFSfilter+"_spec_kl{0}_{1}.png".format(resnumbasis,IFSfilter)),bbox_inches='tight')
        # plt.savefig(os.path.join(out_pngs,planet,planet+"_"+IFSfilter+"_spec_kl{0}_{1}.pdf".format(resnumbasis,IFSfilter)),bbox_inches='tight')


