import os, shutil

#import l2_full_physics_wrapper
#import retrieval
import full_physics
from l2_afp.utils.misc import get_lua_config_files

ddir = '/data/merrelli/OCO2_cld_reff_test'
L1bfile = os.path.join(ddir,'oco2_L1bScND_07216a_151109_B8000r_170703121429.h5')
IDPfile = os.path.join(ddir,'oco2_L2IDPND_07216a_151109_B8100r_170730204402.h5')
Metfile = os.path.join(ddir,'oco2_L2MetND_07216a_151109_B8000r_170703000620.h5')
# a probably clear FOV over OK or KS
sounding_id = '2015110919384031'

config_file = get_lua_config_files()['default']
merradir = '/data/OCO2/L2_datasets/merra_composite'
abscodir = '/data/OCO2/absco'

# copying output method from l2_special_run.py
L2run = full_physics.L2Run(
    config_file, sounding_id, Metfile, L1bfile, 
    abscodir = abscodir, merradir = merradir, 
    imap_file = IDPfile)

x_a = L2run.state_vector.state.copy()
x_i = x_a.copy()
cov_a = L2run.state_vector.state_covariance.copy()

out, outerr = L2run.config.output()
l2_solver_res = L2run.solver.solve(x_i, x_a, cov_a)

# update state vector, so the output is correctly synced
L2run.state_vector.update_state(L2run.solver.x_solution, 
                                L2run.solver.aposteriori_covariance)

out.write()
out = None
L2run = None

output_filename = 'l2fp_test_'+sounding_id+'.h5' 
try:
    os.rename('out.h5', output_filename)
except OSError:
    print('out.h5 not found, copying the generating file')
    shutil.copyfile('out.h5.generating', output_filename)
