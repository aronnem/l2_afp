import os.path

import h5py
import numpy as np

import l2_afp

ddir = '/data/merrelli/l2_afp_sandbox'
L1bfile = os.path.join(ddir, 'oco2_L1bScND_10246a_160604_B8000r_170630100507.h5')
Metfile = os.path.join(ddir, 'oco2_L2MetND_10246a_160604_B8000r_170630042307.h5')
IDPfile = os.path.join(ddir, 'oco2_L2IDPND_10246a_160604_B8100r_170721105543.h5')
sounding_id = '2016060421174703'

config_file = l2_afp.utils.get_lua_config_files()['default']
merradir = '/data/OCO2/L2_datasets/merra_composite'
abscodir = '/data/OCO2/absco'

l2_obj = l2_afp.wrapped_fp(
    L1bfile, Metfile, config_file, abscodir, 
    merradir = merradir, sounding_id = sounding_id, imap_file = IDPfile)

# run the afp solver.
Sa = l2_obj.get_Sa()
Se_diag = l2_obj.get_Se_diag()
Kupdate = l2_obj.Kupdate
y = l2_obj.get_y()
x0 = l2_obj.get_x()

Se = l2_afp.utils.diagcmatrix(Se_diag)

x_i_list, Fx_i_list, S_i_list = l2_afp.bayesian_nonlinear_solver(
    Se, Sa, y, x0, Kupdate, start_gamma=10.0,
    max_iteration_ct = 5, debug_write=False, 
    debug_write_prefix='test_l2_afp_match_run', 
    match_l2_fp_costfunc=True)

# change these into ndarray that match the normal l2 single sounding output, 
# with iteration output turned on.
# shapes are [1, N_iter, N_var] for state var, uncertainty;
# and [1, N_iter, N_var, N_var] for posterior covar.
x_i = np.array(x_i_list)[np.newaxis, ...]
Fx_i = np.array(Fx_i_list)[np.newaxis, ...]
S_i = np.array(S_i_list)[np.newaxis, ...]
x_unc_i = np.zeros_like(x_i)
for n in range(S_i.shape[0]):
    x_unc_i[0,n,:] = np.sqrt(np.diag(S_i_list[n]))

output_filename = 'l2_afp_test_'+sounding_id+'.h5'
l2_obj.write_h5_output_file(
    output_filename, final_state = x_i_list[-1], 
    final_uncert = np.sqrt(np.diag(S_i_list[-1])), 
    modeled_rad = Fx_i_list[-1])

# now append in the Iteration data.
# note that h5py will nicely create the needed groups for you, 
# so we can do this easily in one loop.
h = h5py.File(output_filename, 'a')
vnames = [
    '/Iteration/RetrievedStateVector/state_vector_result', 
    '/Iteration/RetrievalResults/aposteriori_covariance_matrix',
    '/Iteration/RetrievedStateVector/state_vector_aposteriori_uncert',
    '/Iteration/SpectralParameters/modeled_radiance']
vdata = [x_i, S_i, x_unc_i, Fx_i]
for vname, v in zip(vnames, vdata):
    h.create_dataset(vname, data = v)
h.close()
