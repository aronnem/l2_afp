import os.path

import h5py
import numpy as np

import l2_afp

ddir = '/data/merrelli/OCO2_cld_reff_test'
L1bfile = os.path.join(ddir,'oco2_L1bScND_07216a_151109_B8000r_170703121429.h5')
IDPfile = os.path.join(ddir,'oco2_L2IDPND_07216a_151109_B8100r_170730204402.h5')
Metfile = os.path.join(ddir,'oco2_L2MetND_07216a_151109_B8000r_170703000620.h5')

# a cloudy FOV over Gulf of Mexico
sounding_id = '2015110919351871'
# a probably clear FOV over OK or KS
sounding_id = '2015110919384031'

merradir = '/data/OCO2/L2_datasets/merra_composite'
abscodir = '/data/OCO2/absco'

wrkdir = './'
sprop_variable_file = 'l2_aerosol_combined_base.h5'
sprop_grid_file = 'DU_sproperty_grid.h5'
l2_obj = l2_afp.fp_wrapper.wrapped_fp_DU_reff(
    wrkdir, sprop_variable_file, 
    L1bfile, Metfile, merradir, abscodir, sprop_grid_file,
    sounding_id = sounding_id, imap_file = IDPfile)

Sa = l2_obj.get_Sa()
Se_diag = l2_obj.get_Se_diag()
Kupdate = l2_obj.Kupdate
y = l2_obj.get_y()
x0 = l2_obj.get_x()

Se = l2_afp.utils.diagcmatrix(Se_diag)

x_i_list, Fx_i_list, S_i_list = l2_afp.bayesian_nonlinear_solver(
     Se, Sa, y, x0, Kupdate, start_gamma=10.0,
     max_iteration_ct = 5, debug_write=False, 
     debug_write_prefix='l2_afp_DU_reff_run_debug_', 
     match_l2_fp_costfunc=True)
