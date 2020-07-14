__author__ = 'jruffio'

import numpy as np
import matplotlib.pyplot as plt

gravcst = 6.67408e-11 #(m3kg-1s-2)
Mjup = 1.898e27 # kg
a_Io = 421700 #km

planet_Mjupmass_list = np.logspace(-1,2,100)
moon_Mjupmass_list = np.logspace(-1,1,10)
planet_mass_list = planet_Mjupmass_list*Mjup
moon_mass_list = moon_Mjupmass_list*Mjup

planet_mass_grid,moon_mass_grid = np.meshgrid(planet_mass_list,moon_mass_list)
print(moon_mass_grid)
print(moon_mass_grid.shape)

P_grid = np.sqrt(4*np.pi**2/(gravcst*(planet_mass_grid+moon_mass_grid))*(a_Io**3))
KP_grid = (2*np.pi*gravcst/P_grid*(moon_mass_grid**3)/(planet_mass_grid+moon_mass_grid)**2)**(1/3)/1000

plt.imshow(KP_grid,origin="lower",extent=[np.min(planet_Mjupmass_list),np.max(planet_Mjupmass_list),np.min(moon_Mjupmass_list),np.max(moon_Mjupmass_list)])
plt.colorbar()
plt.xscale("log")
plt.yscale("log")
plt.contour(planet_Mjupmass_list,moon_Mjupmass_list,KP_grid)
plt.show()