__author__ = 'jruffio'


import os
import glob
import astropy.io.fits as pyfits
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from reduce_HPFonly_diagcov import _remove_bad_pixels_z, _remove_edges,_tpool_init,_arraytonumpy,_spline_psf_model,convolve_spectrum

# Calculate transmission
if 1:
    import csv
    from PyAstronomy import pyasl
    from scipy.interpolate import interp1d
    numthreads=28
    specpool = mp.Pool(processes=numthreads)

    cutoff = 20
    R0 = 4000

    # star_name = "HR_8799"
    # star_name = "HD_210501"
    # star_name = "HIP_1123"
    star_name = "HIP_116886"
    OSIRISDATA = "/data/osiris_data/"

    fileinfos_refstars_filename = os.path.join(OSIRISDATA,"fileinfos_refstars_jb.csv")
    with open(fileinfos_refstars_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')
        refstarsinfo_list_table = list(csv_reader)
        refstarsinfo_colnames = refstarsinfo_list_table[0]
        refstarsinfo_list_data = refstarsinfo_list_table[1::]
    refstarsinfo_filename_id = refstarsinfo_colnames.index("filename")
    refstarsinfo_filelist = [os.path.basename(item[refstarsinfo_filename_id]) for item in refstarsinfo_list_data]

    phoenix_folder = os.path.join(OSIRISDATA,"phoenix")
    phoenix_model_refstar_filelist = glob.glob(os.path.join(phoenix_folder,"fitting",star_name,"*.fits"))

    for phoenix_model_refstar_filename in phoenix_model_refstar_filelist:
        plt.figure(1,figsize=(16,16))
        plt.title(phoenix_model_refstar_filename)
        for IFSfilter in ["Kbb","Hbb"]:
            filelist=[]
            filename_filter = "ao_off_s*"+IFSfilter+"*020_spec_v2.fits"
            filelist.extend(glob.glob(os.path.join(OSIRISDATA,"HR_8799_*","*","reduced_telluric_jb",star_name,filename_filter)))
            filename_filter = "s*"+IFSfilter+"*020_psfs_repaired_spec_v2.fits"
            filelist.extend(glob.glob(os.path.join(OSIRISDATA,"HR_8799_*","*","reduced_telluric_jb",star_name,filename_filter)))
            filelist_new = []
            for spec_filename in filelist:
                if "2009" not in spec_filename:
                    filelist_new.append(spec_filename)

            for specid,spec_filename in enumerate(filelist_new[0:20]):
                print(spec_filename)
                with pyfits.open(spec_filename) as hdulist:
                    wvs = hdulist[0].data[0,:]
                    spec = hdulist[0].data[1,:]/np.nanmean(hdulist[0].data[1,:])


                for fileid,refstarsinfo_file in enumerate(refstarsinfo_filelist):
                    if os.path.basename(refstarsinfo_file).replace(".fits","") in spec_filename:
                        fileitem = refstarsinfo_list_data[fileid]
                        break

                type_id = refstarsinfo_colnames.index("type")
                rv_simbad_id = refstarsinfo_colnames.index("RV Simbad")
                vsini_fixed_id = refstarsinfo_colnames.index("vsini fixed")
                starname_id = refstarsinfo_colnames.index("star name")

                refstar_RV = float(fileitem[rv_simbad_id])
                vsini_fixed = float(fileitem[vsini_fixed_id])
                ref_star_type = fileitem[type_id]
                refstar_name = fileitem[starname_id]

                phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
                with pyfits.open(phoenix_wv_filename) as hdulist:
                    phoenix_wvs = hdulist[0].data/1.e4
                crop_phoenix = np.where((phoenix_wvs>wvs[0]-(wvs[-1]-wvs[0])/2)*(phoenix_wvs<wvs[-1]+(wvs[-1]-wvs[0])/2))
                phoenix_wvs = phoenix_wvs[crop_phoenix]


                phoenix_refstar_filename=phoenix_model_refstar_filename.replace(".fits","_gaussconv_R{0}_{1}.csv".format(R0,IFSfilter))
                print(phoenix_refstar_filename)
                if len(glob.glob(phoenix_refstar_filename)) == 0:
                    with pyfits.open(phoenix_model_refstar_filename) as hdulist:
                        phoenix_refstar = hdulist[0].data[crop_phoenix]
                    print("convolving: "+phoenix_model_refstar_filename)

                    phoenix_refstar_conv = convolve_spectrum(phoenix_wvs,phoenix_refstar,R0,specpool)

                    with open(phoenix_refstar_filename, 'w+') as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=' ')
                        csvwriter.writerows([["wvs","spectrum"]])
                        csvwriter.writerows([[a,b] for a,b in zip(phoenix_wvs,phoenix_refstar_conv)])

                with open(phoenix_refstar_filename, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=' ')
                    list_starspec = list(csv_reader)
                    refstarpho_spec_str_arr = np.array(list_starspec, dtype=np.str)
                    col_names = refstarpho_spec_str_arr[0]
                    refstarpho_spec = refstarpho_spec_str_arr[1::,1].astype(np.float)
                    refstarpho_spec_wvs = refstarpho_spec_str_arr[1::,0].astype(np.float)
                    where_IFSfilter = np.where((refstarpho_spec_wvs>wvs[0])*(refstarpho_spec_wvs<wvs[-1]))
                    refstarpho_spec = refstarpho_spec/np.mean(refstarpho_spec[where_IFSfilter])
                    refstarpho_spec_func = interp1d(refstarpho_spec_wvs,refstarpho_spec,bounds_error=False,fill_value=np.nan)


                c_kms = 299792.458
                refstarpho_spec = refstarpho_spec_func(wvs*(1-refstar_RV/c_kms))
                broadened_refstarpho = pyasl.rotBroad(wvs, refstarpho_spec, 0.5, vsini_fixed_id)

                transmission_model = spec/broadened_refstarpho
                transmission_model = transmission_model/np.nanmean(transmission_model)

                plt.plot(wvs,transmission_model+specid*0.2,label=os.path.basename(spec_filename).split("bb_")[0])
        # plt.show()

        plt.savefig(os.path.join(phoenix_folder,"fitting",star_name,star_name+"_"+os.path.basename(phoenix_model_refstar_filename).split(".fits")[0]+".png"),bbox_inches='tight')
        plt.close(1)