import os.path

import numpy as np

import l2_full_physics_wrapper
import l2_afp_retrieval
import retrieval_utils

ddir = '/data/merrelli/OCO2_L2_workarea/sandbox_B8_pytest'
L1bfile = os.path.join(ddir, 'oco2_L1bScTG_06280a_150906_B7000r_151030071317.h5')
ECfile = os.path.join(ddir, 'oco2_ECMWFTG_06280a_150906_B7000_150906183644.h5')
IDPfile = os.path.join(ddir, 'oco2_L2IDPTG_06280a_150906_B7000r_151030124259.h5')
sounding_id = '2015090613050738'
#config_file = 'custom_config_default.lua'
config_file = 'custom_config.lua'
merradir = '/data/OCO2/L2_datasets/merra_composite'
abscodir = '/data/OCO2/absco'

L2_obj = l2_full_physics_wrapper.wrapped_l2_fp(
    L1bfile, ECfile, config_file, merradir, abscodir, 
    sounding_id = sounding_id, imap_file = IDPfile)

Sa_L2 = L2_obj.get_Sa()
Se_diag_L2 = L2_obj.get_Se_diag()
y_L2 = L2_obj.get_y()
x0_L2 = L2_obj.get_x()
n_state_L2 = len(L2_obj.get_state_variable_names())
_, Fhatx0_L2, K0_L2 = L2_obj.jacobian_run()

# augmented retrieval. 
# for now, trying a sort of dummy variable:
# measurement, y = surface_P x 1e13
# (e.g., like direct measurement; except makes noises ~ order.)
# K = 1e15 (direct proportionality)
# meas (in this case) = 1005 x 1e13 (change below, to experiment)
# prior variance = 10 ** 2 (loose constraint from prior)
# meas variance = 0.5 ** 2 (tight constraint from meas)

n_state_aug = 0
n_meas_aug = 1
x0_aug = np.zeros(n_state_aug)
Sa_aug = np.zeros((n_state_aug, n_state_aug))
Fhatx0_aug = np.zeros(n_meas_aug)
y_aug = np.zeros(n_meas_aug)
Se_diag_aug = np.zeros(n_meas_aug)
K0_aug = np.zeros((n_meas_aug, n_state_L2 + n_state_aug))

Fhatx0_aug[0] = x0_L2[21] * 1e13
y_aug[0] = 1005e2 * 1e13 # what's the measurement?
# soft constraint - this should not change retrieval much.
#Se_diag_aug[0] = (10.0e2 * 1e13) ** 2
# harder constraint
Se_diag_aug[0] = (0.1e2 * 1e13) ** 2

# augment the covariances
Sa = retrieval_utils.blk_diag([Sa_L2, Sa_aug])
Se_diag = np.concatenate([Se_diag_L2, Se_diag_aug])

# concat x, y and K.
y = np.concatenate([y_L2, y_aug])
Fhatx0 = np.concatenate([Fhatx0_L2, Fhatx0_aug])
x0 = np.concatenate([x0_L2, x0_aug])
K0 = np.concatenate([K0_L2, K0_aug], axis=0)

Se = l2_afp_retrieval.diagcmatrix(Se_diag)

def Kupdate(hatx, model_params):
    L2_obj = model_params['L2_obj']
    Fhatx_L2, K_L2 = L2_obj.Kupdate(hatx, None)
    Fhatx_aug = hatx[[21]] * 1e13
    K_aug = np.zeros((1, model_params['n_state_L2']))
    K_aug[0,-1] = 1e13
    Fhatx = np.concatenate([Fhatx_L2, Fhatx_aug])
    K = np.concatenate([K_L2, K_aug], axis=0)

    return Fhatx, K

model_params = {'L2_obj':L2_obj, 'n_state_L2':n_state_L2, }

input_args = (Se, Sa, y, x0, Fhatx0, K0, Kupdate)
input_kws = dict(start_gamma=10.0, model_params = model_params,
                 max_iteration_ct = 8, debug_write=True, 
                 debug_write_prefix='test_l2_aug_run')

hatx = l2_afp_retrieval.bayesian_nonlinear_l2fp(
    *input_args, **input_kws)

#hatx = l2_afp_retrieval.bayesian_nonlinear_l2fp(
#    Se, Sa, y0, x0, y0, K0, Kupdate, 
#    start_gamma=10.0, model_params = model_params, 
#    max_iteration_ct = 8, debug_write=True)
