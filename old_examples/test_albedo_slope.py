import matplotlib.pyplot as plt
import numpy as np

import full_physics
import full_physics_swig

#
# the custom_config.lua mainly just sets absco files etc, 
# for paths on UWisc machine
#
# sounding id was selected to be over-land, so the 
# lambertian land surface model is setup.
#
arg_list = ['custom_config.lua', 
            '2016011219402771', 
            'oco2_ECMWFND_08148a_160112_B7302_160805205818s.h5', 
            'oco2_L1bScND_08148a_160112_B7302_160805205904s.h5']

kw = dict(merradir='/data/OCO2/L2_datasets/merra_composite', 
          abscodir='/data/OCO2/dat_share/absco')

l2_obj = full_physics.L2Run(*tuple(arg_list), **kw)

x_a = l2_obj.state_vector.state.copy()
S_a = l2_obj.state_vector.state_covariance.copy()
names = l2_obj.state_vector.state_vector_name

##############
### Base state
ref_A = 0.1
x_a[35] = ref_A   # O2 Albedo
x_a[36] = 0.0     # O2 Albedo slope

l2_obj.state_vector.update_state(x_a, S_a)
X_0 = l2_obj.forward_model.radiance_all(True)
wl = X_0.spectral_domain.data.copy()
L_0 = X_0.spectral_range.data.copy()

# s = index into O2 part of spectrum
s = (wl<1.0).nonzero()[0]

# compute albedo slope needed to reduce the ref Albedo 
# to zero at the left edge of the band. multiply it by 0.95 
# to prevent actually going below zero (LIDORT error)
ref_wavelen = 0.770

Dwl = ref_wavelen - wl[s][0]
Dwn = 1e4/ref_wavelen - 1e4/wl[s][0]

# albedo slope (DA) calculation:
# DA = Delta_A / Delta Spectral Unit (wl or wnum)
DA_wavelen = ref_A / Dwl * 0.95 # guess - really in nm-1?
DA_wavenum = ref_A / Dwn * 0.95

##############
### perturb 1 - attempt in wavelength
### this will crash l2_fp (negative albedo.)
### mutliply by 1e-4 arbitrarily to prevent negative albedo
x_a[35] = ref_A
x_a[36] = DA_wavelen * 1e-4

l2_obj.state_vector.update_state(x_a, S_a)
X_p1 = l2_obj.forward_model.radiance_all(True)
L_p1 = X_p1.spectral_range.data.copy()

##############
### perturb 2 - attempt in wavenumber
x_a[25] = ref_A
x_a[36] = DA_wavenum

l2_obj.state_vector.update_state(x_a, S_a)
X_p2 = l2_obj.forward_model.radiance_all(True)
L_p2 = X_p2.spectral_range.data.copy()

fig = plt.figure(22)
fig.clf()
ax = fig.add_subplot(111)
l1, = ax.plot(wl[s], L_0[s])
l2, = ax.plot(wl[s], L_p2[s]) 
fig.legend(
    [l1,l2], 
    ['O2 Alb={0:5.2f}, O2 alb slope=0.0'.format(ref_A), 
     'O2 Alb={0:5.2f}, O2 Alb slope={1:f}'.format(ref_A, DA_wavenum)], 
    loc = 'upper center')

ax.set_xticklabels(
    ['{0:5.3f}\n{1:5.0f}'.format(x,1e4/x)
     for x in ax.get_xticks()] )
ax.set_ylabel('Radiance')
ax.set_xlabel('Wavelength [um] or Wavenumber [1/cm]')
ax.set_ylim(0,4.9e19)
ax.grid(1)

ax.text(wl[s][0], L_0[s][0]*1.15, 
        '{0:5.3f} um\n{1:5.0f} 1/cm'.format(wl[s][0],1e4/wl[s][0]), 
        ha='center')
ax.text(wl[s][-1], L_0[s][-1]*1.15, 
        '{0:5.3f} um\n{1:5.0f} 1/cm'.format(wl[s][-1],1e4/wl[s][-1]),
        ha='center')

plt.tight_layout()
plt.draw()

plt.savefig('l2_slope_test.png')
np.savez('l2_slope_test.npz', wl=wl, L_0=L_0, L_p2=L_p2, s=s)
