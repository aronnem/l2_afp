import os.path
import full_physics

ddir = '/data/merrelli/OCO2_L2_workarea/sandbox_B8_pytest'
#ddir = './'
L1bfile = os.path.join(ddir, 'oco2_L1bScTG_06280a_150906_B7000r_151030071317.h5')
ECfile = os.path.join(ddir, 'oco2_ECMWFTG_06280a_150906_B7000_150906183644.h5')
IDPfile = os.path.join(ddir, 'oco2_L2IDPTG_06280a_150906_B7000r_151030124259.h5')
sounding_id = '2015090613050738'
#config_file = 'custom_config_default.lua'
config_file = 'custom_config.lua'
merradir = '/data/OCO2/L2_datasets/merra_composite'
abscodir = '/data/OCO2/absco'
#merradir = '/data/OCO2/L2_datasets/merra_composite'
#abscodir = '/data/OCO2/absco'

arg_list = (config_file, sounding_id, ECfile, L1bfile)
kw_dict = dict(merradir=merradir, abscodir=abscodir, 
               imap_file=IDPfile)

l2_obj = full_physics.L2Run(*arg_list, **kw_dict)

out, outerr = l2_obj.config.output()

fm_result = l2_obj.forward_model.radiance_all(False)
wl = fm_result.wavelength.copy()
I = fm_result.value.copy()
K = fm_result.spectral_range.data_ad.jacobian.copy()

# any of these will cause a segfault.
# I think the write_best_attempt() is the desired one?

#out.write()
#outerr.write()
outerr.write_best_attempt()
