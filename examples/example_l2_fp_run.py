import os, shutil

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

# run the operational solver.
solve_result, return_x, return_S = l2_obj.solve()

output_filename = 'l2_fp_test_'+sounding_id+'.h5'

l2_obj.write_h5_output_file_FPformat(output_filename)
