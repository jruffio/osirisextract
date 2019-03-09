__author__ = 'jruffio'

from pyklip.spectra_management import *
import scipy.io as scio
import matplotlib.pyplot as plt


if 1: # PLanet model
    phoenix_folder = "/home/sda/jruffio/phoenix/"
    for planet in ["b"]:#["c","b"]
        for IFSfilter in ["Kbb","Hbb"]:
            print(planet,IFSfilter)

            if IFSfilter=="Kbb": #Kbb 1965.0 0.25
                CRVAL1 = 1965.
                CDELT1 = 0.25
                nl=1665
                R=4000
            elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
                CRVAL1 = 1473.
                CDELT1 = 0.2
                nl=1651
                R=5000

            init_wv = CRVAL1/1000. # wv for first slice in mum
            dwv = CDELT1/1000. # wv interval between 2 slices in mum
            wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)

            template_spec_filename="/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/HR8799"+planet+"_"+IFSfilter[0:1]+"_3Oct2018.save"
            travis_spectrum = scio.readsav(template_spec_filename)
            ori_planet_spec = np.array(travis_spectrum["fmod"])
            ori_planet_convspec = np.array(travis_spectrum["fmods"])
            wmod = np.array(travis_spectrum["wmod"])/1.e4

            print(wmod.shape)
            planet_convspec = np.zeros(ori_planet_spec.shape)
            dwvs = wmod[1::]-wmod[0:(np.size(wmod)-1)]
            med_dwv = np.median(dwvs)
            for k,pwv in enumerate(wmod):
                # if k!=190000:
                #     continue
                # print(k)
                FWHM = pwv/R
                sig = FWHM/(2*np.sqrt(2*np.log(2)))
                w = int(np.round(sig/med_dwv*10.))
                stamp_spec = ori_planet_spec[np.max([0,k-w]):np.min([np.size(ori_planet_spec),k+w])]
                stamp_wvs = wmod[np.max([0,k-w]):np.min([np.size(wmod),k+w])]
                stamp_dwvs = stamp_wvs[1::]-stamp_wvs[0:(np.size(stamp_spec)-1)]
                gausskernel = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(stamp_wvs-pwv)**2/sig**2)
                if 0:
                    plt.plot(stamp_wvs,stamp_spec/np.max(stamp_spec),label="tlc")
                    plt.plot(stamp_wvs,gausskernel/np.max(gausskernel),label="ker")
                    plt.xlabel("mum")
                    plt.legend()
                    plt.show()
                planet_convspec[k] = np.sum(gausskernel[1::]*stamp_spec[1::]*stamp_dwvs)

            out_filename = template_spec_filename.replace(".save","_conv"+IFSfilter+".csv")
            with open(out_filename, 'w+') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ')
                csvwriter.writerows([["wvs","spectrum"]])
                csvwriter.writerows([[a,b] for a,b in zip(wmod,planet_convspec)])

            # plt.plot(wmod,ori_planet_spec,label="ori")
            # plt.plot(wmod,planet_convspec,label="conv")
            # plt.plot(wmod,ori_planet_convspec,label="ori conv")
            # plt.xlabel("mum")
            # plt.legend()
            # plt.show()


if 0: # Phoenix
    phoenix_folder = "/home/sda/jruffio/phoenix/"
    for ref_star_name in ["HR_8799","HD_210501","HIP_1123","HIP_116886"]:
        for IFSfilter in ["Kbb","Hbb"]:
            print(ref_star_name,IFSfilter)

            if IFSfilter=="Kbb": #Kbb 1965.0 0.25
                CRVAL1 = 1965.
                CDELT1 = 0.25
                nl=1665
                R=4000
            elif IFSfilter=="Hbb": #Hbb 1651 1473.0 0.2
                CRVAL1 = 1473.
                CDELT1 = 0.2
                nl=1651
                R=5000

            init_wv = CRVAL1/1000. # wv for first slice in mum
            dwv = CDELT1/1000. # wv interval between 2 slices in mum
            wvs=np.arange(init_wv,init_wv+dwv*nl-1e-6,dwv)

            template_spec_filename="/home/sda/jruffio/osiris_data/HR_8799_c/HR8799c_"+IFSfilter[0:1]+"_3Oct2018.save"
            travis_spectrum = scio.readsav(template_spec_filename)
            ori_planet_spec = np.array(travis_spectrum["fmods"])
            wmod = np.array(travis_spectrum["wmod"])/1.e4
            ori_planet_spec = ori_planet_spec/np.mean(ori_planet_spec)
            z = interp1d(wmod,ori_planet_spec)
            planet_spec = z(wvs)
            planet_spec = planet_spec/np.mean(planet_spec)

            phoenix_wv_filename = os.path.join(phoenix_folder,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
            with pyfits.open(phoenix_wv_filename) as hdulist:
                phoenix_wvs = hdulist[0].data/1.e4
            crop_phoenix = np.where((phoenix_wvs>wmod[0])*(phoenix_wvs<wmod[-1]))
            phoenix_wvs = phoenix_wvs[crop_phoenix]

            phoenix_tlc_filename = glob.glob(os.path.join(phoenix_folder,ref_star_name+"*.fits"))[0]
            with pyfits.open(phoenix_tlc_filename) as hdulist:
                phoenix_tlc = hdulist[0].data[crop_phoenix]

            print(phoenix_tlc.shape)
            conv_phoenix_tlc = np.zeros(phoenix_tlc.shape)
            dwvs = phoenix_wvs[1::]-phoenix_wvs[0:(np.size(phoenix_wvs)-1)]
            med_dwv = np.median(dwvs)
            for k,pwv in enumerate(phoenix_wvs):
                # if k!=190000:
                #     continue
                # print(k)
                FWHM = pwv/R
                sig = FWHM/(2*np.sqrt(2*np.log(2)))
                w = int(np.round(sig/med_dwv*10.))
                stamp_spec = phoenix_tlc[np.max([0,k-w]):np.min([np.size(phoenix_tlc),k+w])]
                stamp_wvs = phoenix_wvs[np.max([0,k-w]):np.min([np.size(phoenix_tlc),k+w])]
                stamp_dwvs = stamp_wvs[1::]-stamp_wvs[0:(np.size(stamp_spec)-1)]
                gausskernel = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*(stamp_wvs-pwv)**2/sig**2)
                if 0:
                    plt.plot(stamp_wvs,stamp_spec/np.max(stamp_spec),label="tlc")
                    plt.plot(stamp_wvs,gausskernel/np.max(gausskernel),label="ker")
                    plt.xlabel("mum")
                    plt.legend()
                    plt.show()
                conv_phoenix_tlc[k] = np.sum(gausskernel[1::]*stamp_spec[1::]*stamp_dwvs)

            out_filename = phoenix_tlc_filename.replace(".fits","_conv"+IFSfilter+".csv")
            with open(out_filename, 'w+') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ')
                csvwriter.writerows([["wvs","spectrum"]])
                csvwriter.writerows([[a,b] for a,b in zip(phoenix_wvs,conv_phoenix_tlc)])

            # plt.plot(phoenix_wvs,phoenix_tlc,label="tlc")
            # plt.plot(phoenix_wvs,conv_phoenix_tlc,label="conv tlc")
            # plt.xlabel("mum")
            # plt.legend()
            # plt.show()


if 0: # Pickles
    for planet in ["b"]:#["c","b"]:
        for IFSfilter in ["Jbb","Hbb","Kbb"]:
            print(planet,IFSfilter)

            template_spec_filename="/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/HR8799"+planet+"_"+IFSfilter[0:1]+"_3Oct2018.save"
            travis_spectrum = scio.readsav(template_spec_filename)
            ori_planet_spec = np.array(travis_spectrum["fmods"])
            wmod = np.array(travis_spectrum["wmod"])/1.e4
            ori_planet_spec = ori_planet_spec/np.mean(ori_planet_spec)
            # from scipy.interpolate import interp1d
            # z = interp1d(wmod,ori_planet_spec)
            # planet_spec = z(wvs)
            # planet_spec = planet_spec/np.mean(planet_spec)

            print("coucou")
            wvs, starspec = get_star_spectrum(wmod,star_type = "F0", temperature = None,mute = None)
            print("bonjour")
            with open("/home/sda/jruffio/osiris_data/HR_8799_"+planet+"/hr_8799_"+planet+"_pickles_spectrum_"+IFSfilter+".csv", 'w+') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=' ')
                csvwriter.writerows([["wvs","spectrum"]])
                csvwriter.writerows([[a,b] for a,b in zip(wvs,starspec)])
            # import matplotlib.pyplot as plt
            # plt.plot(wvs, starspec)
            # plt.show()

            # init_wv = CRVAL1/1000. # wv for first slice in mum
            # dwv = CDELT1/1000. # wv interval between 2 slices in mum
            # wvs = np.arange(init_wv,init_wv+dwv*nl,dwv)
            #
            # wvs, starspec = get_star_spectrum(wvs,star_type = "F0", temperature = None,mute = None)
            #
            # with open("/home/sda/jruffio/osiris_data/HR_8799_c/hr_8799_c_pickles_spectrum_Kbb.csv", 'w+') as csvfile:
            #     csvwriter = csv.writer(csvfile, delimiter=' ')
            #     csvwriter.writerows([["wvs","spectrum"]])
            #     csvwriter.writerows([[a,b] for a,b in zip(wvs,starspec)])
            #
            # import matplotlib.pyplot as plt
            # plt.plot(wvs, starspec)
            # plt.show()