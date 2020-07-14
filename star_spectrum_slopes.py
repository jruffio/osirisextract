__author__ = 'jruffio'

import os
import astropy.io.fits as pyfits
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

if __name__ == "__main__":
    IFSfilter = "Kbb"
    if IFSfilter=="Kbb": #Kbb 1965.0 0.25
        CRVAL1 = 1965.
        CDELT1 = 0.25
        nl=1665
        R0=4000
    elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1473.
        CDELT1 = 0.2
        nl=1651
        R0=4000#5000
    elif IFSfilter=="Jbb": #Hbb 1651 1473.0 0.2
        CRVAL1 = 1180.
        CDELT1 = 0.15
        nl=1574
        R0=4000
    init_wv = CRVAL1/1000. # wv for first slice in mum
    dwv = CDELT1/1000. # wv interval between 2 slices in mum
    wvs=np.linspace(init_wv,init_wv+dwv*nl,nl,endpoint=False)

    osiris_data_dir = "/data/osiris_data/"
    phoenix_db_folder = os.path.join(osiris_data_dir,"phoenix","PHOENIX-ACES-AGSS-COND-2011")

    phoenix_wv_filename = os.path.join(phoenix_db_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011_R{0}.fits".format(R0))
    with pyfits.open(phoenix_wv_filename) as hdulist:
        phospec_wvs = hdulist[0].data

    phoenix_model_host_filename = os.path.join(phoenix_db_folder,"Z-0.0/lte09000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes_R4000.fits")
    with pyfits.open(phoenix_model_host_filename) as hdulist:
        pho9000K_spec = hdulist[0].data
    pho9000K_spec = pho9000K_spec/np.mean(pho9000K_spec)
    pho9000K_func = interp1d(phospec_wvs,pho9000K_spec,bounds_error=False,fill_value=np.nan)

    phoenix_model_host_filename = os.path.join(phoenix_db_folder,"Z-0.0/lte10000-4.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes_R4000.fits")
    with pyfits.open(phoenix_model_host_filename) as hdulist:
        pho9400K_spec = hdulist[0].data
    pho9400K_spec = pho9400K_spec/np.mean(pho9400K_spec)
    pho9400K_func = interp1d(phospec_wvs,pho9400K_spec,bounds_error=False,fill_value=np.nan)


    spec1 = pho9000K_func(wvs)
    spec1 = spec1/spec1[0]
    spec2 = pho9400K_func(wvs)
    spec2 = spec2/spec2[0]


    tmpfilename = os.path.join(osiris_data_dir,"hr8799b_modelgrid/","hr8799b_modelgrid_R{0}_{1}.fits".format(R0,IFSfilter))
    hdulist = pyfits.open(tmpfilename)
    planet_model_grid =  hdulist[0].data
    oriplanet_spec_wvs =  hdulist[1].data
    Tlistunique =  hdulist[2].data
    logglistunique =  hdulist[3].data
    CtoOlistunique =  hdulist[4].data
    hdulist.close()

    print(planet_model_grid.shape,np.size(Tlistunique),np.size(logglistunique),np.size(CtoOlistunique),np.size(oriplanet_spec_wvs))
    from scipy.interpolate import RegularGridInterpolator
    myinterpgrid = RegularGridInterpolator((Tlistunique,logglistunique,CtoOlistunique),planet_model_grid,method="linear",bounds_error=False,fill_value=0.0)

    plspec1 = myinterpgrid([900,4,0.55])[0]
    plspec1_func = interp1d(oriplanet_spec_wvs,plspec1,bounds_error=False,fill_value=np.nan)
    plspec1 = plspec1_func(wvs)
    plspec1 = plspec1/plspec1[0]
    plspec2 = myinterpgrid([910,4,0.55])[0]
    plspec2_func = interp1d(oriplanet_spec_wvs,plspec2,bounds_error=False,fill_value=np.nan)
    plspec2 = plspec2_func(wvs)
    plspec2 = plspec2/plspec2[0]


    plt.plot(wvs,(spec2-spec1)/spec2,label="star")
    plt.plot(wvs,(plspec2-plspec1)/plspec2,label="planet")
    plt.legend()
    plt.show()