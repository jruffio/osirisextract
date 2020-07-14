__author__ = 'jruffio'
import glob
import os
import re
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import numpy as np
from scipy.ndimage.filters import median_filter

fileinfos_filename = "/home/sda/jruffio/pyOSIRIS/osirisextract/fileinfos.xml"
out_pngs = "/home/sda/jruffio/pyOSIRIS/figures/"
tree = ET.parse(fileinfos_filename)
root = tree.getroot()

OSIRISDATA = "/home/sda/jruffio/osiris_data/"
if 1:
    foldername = "HR_8799_c"
    sep = 0.950
    telluric = os.path.join(OSIRISDATA,"HR_8799_c/20100715/reduced_telluric/HD_210501","s100715_a005001_Kbb_020.fits")
    template_spec = os.path.join(OSIRISDATA,"hr8799c_osiris_template.save")
year = "*"
reductionname = "reduced_quinn"
filenamefilter = "s*_a*001_tlc_Kbb_020.fits"

bad_list = ["s101104_a014001_tlc_Kbb_020.fits","s101104_a016001_tlc_Kbb_020.fits","s101104_a034001_tlc_Kbb_020.fits","s101104_a035001_tlc_Kbb_020.fits","s101104_a036001_tlc_Kbb_020.fits","s101104_a037001_tlc_Kbb_020.fits","s101104_a038001_tlc_Kbb_020.fits","s110724_a030001_tlc_Kbb_020.fits"]


with pyfits.open(glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))[0]) as hdulist:
    input0 = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
    nl,ny,nx = input0.shape
    prihdr = hdulist[0].header
init_wv = prihdr["CRVAL1"]/1000. # wv for first slice in mum
dwv = prihdr["CDELT1"]/1000. # wv interval between 2 slices in mum
cube_wvs = np.arange(init_wv,init_wv+dwv*nl,dwv)
ext_cube_wvs = np.arange(init_wv-(nl//4)*dwv,init_wv+dwv*nl+(nl//4)*dwv,dwv)

# Load planet template spectrum
import scipy.io as scio
travis_spectrum = scio.readsav(template_spec)
hr8799c_spec = np.array(travis_spectrum["fk_bin"])
lambdas = np.array(travis_spectrum["kwave"])

template_spec2 = os.path.join(OSIRISDATA,"HRbc_templates","HR8799c_K_3Oct2018.save")
travis_spectrum2 = scio.readsav(template_spec2)
print(travis_spectrum2.keys())
wmod = np.array(travis_spectrum2["wmod"])*1e-4
fmods = np.array(travis_spectrum2["fmods"])
from  scipy.interpolate import interp1d
fmods_interp1d = interp1d(wmod,fmods,bounds_error=True)
fmod = np.array(travis_spectrum2["fmod"])
fmod_interp1d = interp1d(wmod,fmod,bounds_error=True)

# plt.plot(wmod,fmods,label="fmods",alpha=0.5,linestyle="-")
# plt.plot(ext_cube_wvs,fmods_interp1d(ext_cube_wvs),label="fmods ext interp",alpha=0.5,linestyle="--")
# plt.plot(cube_wvs,fmods_interp1d(cube_wvs),label="fmods interp",alpha=0.5,linestyle=":")
# plt.plot(wmod,fmod,label="fmod",alpha=0.5,linestyle="-")
# plt.plot(ext_cube_wvs,fmod_interp1d(ext_cube_wvs),label="fmod ext interp",alpha=0.5,linestyle="--")
# plt.plot(cube_wvs,fmod_interp1d(cube_wvs),label="fmod interp",alpha=0.5,linestyle=":")
# plt.plot(lambdas,hr8799c_spec,label="fk_bin")
# plt.legend()
# plt.show()
# exit()
hr8799c_spec = fmods_interp1d(cube_wvs)
suffix="water"


# generate hpf data
if 0:
    filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
    filelist.sort()
    planet_c = root.find("c")
    for filename in filelist:
        print(filename)
        filebasename = os.path.basename(filename)
        filedirname = os.path.dirname(filename)
        if filebasename in bad_list:
            continue
        fileelement = planet_c.find(filebasename)
        center = (fileelement.attrib["xvisucen"],fileelement.attrib["yvisucen"])

        with pyfits.open(filename) as hdulist:
            input0 = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
            prihdr = hdulist[0].header
        nl,ny,nx = input0.shape
        #########
        # compute spectrum of HR 8799 (the star)
        hr8799_spec = np.nanmean(input0,axis=(1,2))
        hr8799_spec = hr8799_spec/np.mean(hr8799_spec)

        mn_input = np.nanmean(input0,axis=(1,2))
        input0 = input0/hr8799_spec[:,None,None]
        hpf_input0 = np.zeros(input0.shape)
        for k in range(ny):
            for l in range(nx):
                # print(k,l)
                data = input0[:, k, l]/np.mean(input0[:, k, l])
                if np.sum(np.isnan(data)) >0:
                    continue
                hpf_data = data - median_filter(data,footprint=np.ones(200),mode="constant",cval=0.0)
                hpf_data = hpf_data[100:(nl-100)]
                data[(np.where(np.abs(hpf_data)>3*np.nanstd(hpf_data))[0]+100,)] = 0
                # smooth_data = np.correlate(data,template,mode="same")[100:(nl-100)]
                # data = data[100:(nl-100)]
                # hpf_data = data-smooth_data*(np.sum(smooth_data*data)/np.sum(smooth_data*smooth_data))
                hpf_data = data - median_filter(data,footprint=np.ones(200),mode="constant",cval=0.0)
                hpf_input0[:,k,l] = hpf_data

        if not os.path.exists(os.path.join(filedirname,"medfilt")):
            os.makedirs(os.path.join(filedirname,"medfilt"))
        print(os.path.join(filedirname,"medfilt",filebasename))
        hdulist = pyfits.HDUList()
        hdulist.append(pyfits.PrimaryHDU(data=hpf_input0))
        try:
            hdulist.writeto(os.path.join(filedirname,"medfilt",filebasename), overwrite=True)#.replace(".fits",suffix+".fits")
        except TypeError:
            hdulist.writeto(os.path.join(filedirname,"medfilt",filebasename), clobber=True)
        hdulist.close()


# do CC
if 1:
    # tmpselec = np.linspace(0,1664,100,dtype=np.int)
    tmpselec = np.arange(100,1565)
    filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
    filelist.sort()
    planet_c = root.find("c")

    centers = []
    hpfinputs = []
    specmodels = []

    for filename in filelist:
        print(filename)
        filebasename = os.path.basename(filename)
        filedirname = os.path.dirname(filename)
        if filebasename in bad_list:
            continue
        fileelement = planet_c.find(filebasename)
        center = (float(fileelement.attrib["xvisucen"]),float(fileelement.attrib["yvisucen"]))
        if fileelement.attrib["stardir"] == "left":
            center = (center[0]+float(fileelement.attrib["sep"])/ 0.0203, center[1])
        elif fileelement.attrib["stardir"] == "down":
            center = (center[0], center[1]-float(fileelement.attrib["sep"])/ 0.0203)
        centers.append(center)

        with pyfits.open(filename) as hdulist:
            input0 = np.rollaxis(np.rollaxis(hdulist[0].data,2),2,1)
        nl,ny,nx = input0.shape
        with pyfits.open(filename.replace(filebasename,"medfilt"+os.path.sep+filebasename)) as hdulist:
            hpf_input0 = hdulist[0].data

        hpfinputs.append(hpf_input0[tmpselec,:,:])

        hr8799_spec = np.nanmean(input0,axis=(1,2))
        hr8799_spec = hr8799_spec/np.mean(hr8799_spec)
        model = hr8799c_spec/hr8799_spec/np.nanmean(hr8799c_spec/hr8799_spec)
        smooth_model = median_filter(model,footprint=np.ones(200),mode="constant",cval=0.0)
        hpf_model = model - smooth_model
        specmodels.append(hpf_model[tmpselec])

    npspecmodels = np.concatenate(specmodels,axis=0)
    nphpfinputs = np.concatenate(hpfinputs,axis=0)
    nlall,nyall,nxall = nphpfinputs.shape

    # psfs = psfs*(hr8799c_spec[:,None,None]/np.nansum(psfs,axis=(1,2))[:,None,None])
    with pyfits.open(os.path.join("/home/sda/jruffio/osiris_data/","s100715_a005001_Kbb_020_medcombinedpsfs.fits")) as hdulist:
        psfs = hdulist[0].data[tmpselec,:,:]
        hdulist.close()
    psfs = psfs/np.nansum(psfs,axis=(1,2))[:,None,None]
    nl_psf,ny_psf,nx_psf = psfs.shape

    x_psf_grid, y_psf_grid = np.meshgrid(np.arange(nx_psf * 1.)-nx_psf//2,np.arange(ny_psf* 1.)-ny_psf//2)
    psfs_func_list = []
    psfs[np.where(np.isnan(psfs))] = 0
    import warnings
    from scipy import interpolate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for wv_index in range(nl_psf):
            print(wv_index)
            model_psf = psfs[wv_index, :, :]
            psf_func = interpolate.LSQBivariateSpline(x_psf_grid.ravel(),y_psf_grid.ravel(),model_psf.ravel(),x_psf_grid[0,0:nx_psf-1]+0.5,y_psf_grid[0:ny_psf-1,0]+0.5)
            psfs_func_list.append(psf_func)

    ccmap = np.zeros((30,10))
    nycc,nxcc = ccmap.shape
    for k in range(nxcc):
        for l in range(nycc):
            print(k,l)
            #build model
            psf_model = np.zeros(nphpfinputs.shape)
            for z in range(nlall):
                center = centers[z//nl_psf]
                # x_grid, y_grid = np.meshgrid(np.arange(nxall * 1.)-center[0],np.arange(nyall* 1.)-center[1])
                x_vec_centered = np.arange(nxall * 1.)-(center[0]+k-nxcc//2)
                y_vec_centered = np.arange(nyall* 1.)-(center[1]+l-nycc//2)
                psf_model[z,:,:] = psfs_func_list[z%nl_psf](x_vec_centered,y_vec_centered).transpose()*npspecmodels[z]

            # plt.plot(psf_model[:,32,11],color="blue")
            # plt.plot(nphpfinputs[:,32,11],color="red")
            # plt.show()
            if 1:
                ccmap[l,k] = np.nansum(np.nansum(psf_model[:,5:60,2:17]*nphpfinputs[:,5:60,2:17],axis=(1,2))/np.nanvar(nphpfinputs[:,5:60,2:17]))
                # plt.subplot(1,2,1)
                # print(center)
                # plt.imshow(tmpmod,interpolation="nearest")
                # plt.subplot(1,2,2)
                # plt.imshow(nphpfinputs[z,:,:],interpolation="nearest")
                # plt.show()

    hdulist = pyfits.HDUList()
    hdulist.append(pyfits.PrimaryHDU(data=ccmap))
    try:
        hdulist.writeto(os.path.join(filedirname,"medfilt","ccmap_allcubes"+suffix+".fits"), overwrite=True)
    except TypeError:
        hdulist.writeto(os.path.join(filedirname,"medfilt","ccmap_allcubes"+suffix+".fits"), clobber=True)
    hdulist.close()

    plt.imshow(ccmap,interpolation="nearest")
    plt.show()


if 1:
    filelist = glob.glob(os.path.join(OSIRISDATA,foldername,year,reductionname,filenamefilter))
    filelist.sort()
    filename = filelist[-1]
    filebasename = os.path.basename(filename)
    filedirname = os.path.dirname(filename)
    with pyfits.open(os.path.join(filedirname,"medfilt","ccmap_allcubes"+suffix+".fits")) as hdulist:
        ccmap = hdulist[0].data

    # plt.imshow(ccmap,interpolation="nearest")
    plt.plot(ccmap[:,5])
    plt.xlabel("distance (Pix)")
    plt.ylabel("correlation")
    plt.show()
    exit()
