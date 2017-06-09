import os.path

import h5py
import numpy as np

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

x_i_list, S_i_list = l2_afp_retrieval.bayesian_nonlinear_l2fp(
    Se, Sa, y, x0, Kupdate, start_gamma=10.0,
    max_iteration_ct = 2, debug_write=False, 
    debug_write_prefix='test_l2_afp_match_run', 
    match_l2_fp_costfunc=True)


# change these into ndarray that match the normal l2 single sounding output, 
# with iteration output turned on.
# shapes are [1, N_iter, N_var] for state var, uncertainty;
# and [1, N_iter, N_var, N_var] for posterior covar.
x_i = np.array(x_i_list)[np.newaxis, ...]
S_i = np.array(S_i_list)[np.newaxis, ...]
x_unc_i = np.zeros_like(x_i)
for n in range(S_i.shape[0]):
    x_unc_i[0,n,:] = np.sqrt(np.diag(S_i_list[n]))

h = h5py.File('l2afp_'+sounding_id+'.h5', 'w')
# note that h5py will nicely create the needed groups for you, 
# so we can do this easily in one loop.
vnames = [
    '/Iteration/RetrievedStateVector/state_vector_result', 
    '/Iteration/RetrievalResults/aposteriori_covariance_matrix',
    '/Iteration/RetrievedStateVector/state_vector_aposteriori_uncert' ]
vdata = [x_i, S_i, x_unc_i]
for vname, v in zip(vnames, vdata):
    h.create_dataset(vname, data = v)
h.close()
