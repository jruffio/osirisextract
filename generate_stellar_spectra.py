__author__ = 'jruffio'

from pyklip.spectra_management import *


if "Kbb":
    CRVAL1 = 1965.
    CDELT1 = 0.25
    nl=1665

init_wv = CRVAL1/1000. # wv for first slice in mum
dwv = CDELT1/1000. # wv interval between 2 slices in mum
wvs = np.arange(init_wv,init_wv+dwv*nl,dwv)

wvs, starspec = get_star_spectrum(wvs,star_type = "F0", temperature = None,mute = None)

with open("/home/sda/jruffio/osiris_data/HR_8799_c/hr_8799_c_pickles_spectrum_Kbb.csv", 'w+') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=' ')
    csvwriter.writerows([["wvs","spectrum"]])
    csvwriter.writerows([[a,b] for a,b in zip(wvs,starspec)])

import matplotlib.pyplot as plt
plt.plot(wvs, starspec)
plt.show()