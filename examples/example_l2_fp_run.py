import os, shutil

#import l2_full_physics_wrapper
#import retrieval
import full_physics
from l2_afp.utils.misc import get_lua_config_files

ddir = '/data/merrelli/OCO2_L2_workarea/sandbox_B8_pytest'
L1bfile = os.path.join(ddir, 'oco2_L1bScTG_06280a_150906_B7000r_151030071317.h5')
ECfile = os.path.join(ddir, 'oco2_ECMWFTG_06280a_150906_B7000_150906183644.h5')
IDPfile = os.path.join(ddir, 'oco2_L2IDPTG_06280a_150906_B7000r_151030124259.h5')
sounding_id = '2015090613050738'

config_file = get_lua_config_files()['default']
merradir = '/data/OCO2/L2_datasets/merra_composite'
abscodir = '/data/OCO2/absco'

#l2_obj = l2_full_physics_wrapper.wrapped_l2_fp(
#    L1bfile, ECfile, config_file, merradir, abscodir, 
#    sounding_id = sounding_id, imap_file = IDPfile)

#Sa = l2_obj.get_Sa()
#Se_diag = l2_obj.get_Se_diag()
#Kupdate = l2_obj.Kupdate
#y = l2_obj.get_y()
#x0 = l2_obj.get_x()

# copying output method from l2_special_run.py
L2run = full_physics.L2Run(
    config_file, sounding_id, ECfile, L1bfile, 
    abscodir = abscodir, merradir = merradir, 
    imap_file = IDPfile)

x_a = L2run.state_vector.state.copy()
x_i = x_a.copy()
cov_a = L2run.state_vector.state_covariance.copy()

out, outerr = L2run.config.output()
l2_solver_res = L2run.solver.solve(x_i, x_a, cov_a)
out.write()
out = None
L2run = None

output_filename = 'l2fp_test_'+sounding_id+'.h5' 
try:
    os.rename('out.h5', output_filename)
except OSError:
    print('out.h5 not found, copying the generating file')
    shutil.copyfile('out.h5.generating', output_filename)
