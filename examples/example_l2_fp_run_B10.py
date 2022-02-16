import os

import l2_afp

ddir = '/data/merrelli/OCO2_L2_workarea/sandbox_B10'
L1bfile = os.path.join(ddir, 'oco2_L1bScGL_07977a_160101_B10003r_191121204308.h5')
Metfile = os.path.join(ddir, 'oco2_L2MetGL_07977a_160101_B10003r_191121204336.h5')
IDPfile = os.path.join(ddir, 'oco2_L2IDPGL_07977a_160101_B10003r_191122012156.h5')
CPrfile = os.path.join(ddir, 'oco2_L2CPrGL_07977a_160101_B10003r_191121205252.h5')
sounding_id = '2016010101221538'

config_file = l2_afp.utils.get_lua_config_files()['operational']
abscodir = '/data/OCO2/absco'

l2_obj = l2_afp.wrapped_fp(
    L1bfile, Metfile, config_file, abscodir, 
    sounding_id = sounding_id, imap_file = IDPfile,
    co2_pr_file = CPrfile,
    enable_console_log=False)

# run the operational solver.
solve_result, return_x, return_S = l2_obj.solve()

output_filename = 'l2_fp_test_'+sounding_id+'.h5'

l2_obj.write_h5_output_file_FPformat(output_filename)
