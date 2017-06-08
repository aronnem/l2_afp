import os.path

import l2_full_physics_wrapper
import l2_afp_retrieval

ddir = '/data/merrelli/OCO2_L2_workarea/sandbox_B8_pytest'
L1bfile = os.path.join(ddir, 'oco2_L1bScTG_06280a_150906_B7000r_151030071317.h5')
ECfile = os.path.join(ddir, 'oco2_ECMWFTG_06280a_150906_B7000_150906183644.h5')
IDPfile = os.path.join(ddir, 'oco2_L2IDPTG_06280a_150906_B7000r_151030124259.h5')
sounding_id = '2015090613050738'
#config_file = 'custom_config_default.lua'
config_file = 'custom_config.lua'
merradir = '/data/OCO2/L2_datasets/merra_composite'
abscodir = '/data/OCO2/absco'

l2_obj = l2_full_physics_wrapper.wrapped_l2_fp(
    L1bfile, ECfile, config_file, merradir, abscodir, 
    sounding_id = sounding_id, imap_file = IDPfile)

Sa = l2_obj.get_Sa()
Se_diag = l2_obj.get_Se_diag()
Kupdate = l2_obj.Kupdate
y = l2_obj.get_y()
x0 = l2_obj.get_x()

Se = l2_afp_retrieval.diagcmatrix(Se_diag)

#hatx = l2_afp_retrieval.bayesian_nonlinear_l2fp(
#    Se, Sa, y, x0, Kupdate, start_gamma=10.0,
#    max_iteration_ct = 8, debug_write=True, 
#    debug_write_prefix='test_l2_afp_run', 
#    match_l2_fp_costfunc=False)
#    debug_write_prefix='test_l2_afp_match_run', 
#    match_l2_fp_costfunc=True)
