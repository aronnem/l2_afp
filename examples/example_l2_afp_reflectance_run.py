import os, shutil

import l2_afp

ddir = '/data/merrelli/l2_afp_sandbox'
L1bfile = os.path.join(ddir, 'oco2_L1bScND_10246a_160604_B8000r_170630100507.h5')
Metfile = os.path.join(ddir, 'oco2_L2MetND_10246a_160604_B8000r_170630042307.h5')
IDPfile = os.path.join(ddir, 'oco2_L2IDPND_10246a_160604_B8100r_170721105543.h5')
sounding_id = '2016060421174703'

config_file = l2_afp.utils.get_lua_config_files()['unit_solar']
merradir = '/data/OCO2/L2_datasets/merra_composite'
abscodir = '/data/OCO2/absco'

l2_obj = l2_afp.wrapped_fp(
    L1bfile, Metfile, config_file, abscodir, 
    merradir = merradir, sounding_id = sounding_id, imap_file = IDPfile,
    enable_console_log=False)

# to get true reflectance, the fluorescence contribution must be 
# zeroed out. That state vector is directly added to the radiance in
# physical radiance units.
x = l2_obj.get_x()
x[59] = 0.0
l2_obj.set_x(x)

wl_lo, L_lo = l2_obj.forward_run(band=1)
wl_hi, R_hi = l2_obj.forward_run_highres(band=1)

# to get reflectance from the unit solar radiance run,
# the solar distance scaling needs to be removed.
Dsolar = l2_obj.get_solar_distance()
R_lo = L_lo * Dsolar ** 2

# now, both need to be scaled by 2 pi / mu; factor of 2 is due to the
# action of the linear polarizer. In essence, computing the equivalent
# unpolarized radiance (I-only).
geom = l2_obj.get_geometry(band=1)
mu = np.cos(np.deg2rad(geom['solar_zenith']))

R_hi *= (2*np.pi/mu)
R_lo *= (2*np.pi/mu)
