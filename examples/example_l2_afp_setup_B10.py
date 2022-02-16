import os

import l2_afp

ddir = '/data/merrelli/OCO2_L2_workarea/sandbox_B10'
L1bfile = os.path.join(ddir, 'oco2_L1bScGL_07977a_160101_B10003r_191121204308.h5')
Metfile = os.path.join(ddir, 'oco2_L2MetGL_07977a_160101_B10003r_191121204336.h5')
IDPfile = os.path.join(ddir, 'oco2_L2IDPGL_07977a_160101_B10003r_191122012156.h5')
CPrfile = os.path.join(ddir, 'oco2_L2CPrGL_07977a_160101_B10003r_191121205252.h5')
sounding_id = '2016010101221538'

config_file = l2_afp.utils.get_lua_config_files()['operational']
merradir = '/data/OCO2/L2_datasets/merra_composite'
abscodir = '/data/OCO2/absco'

l2_obj = l2_afp.wrapped_fp(
    L1bfile, Metfile, config_file, abscodir, 
    sounding_id = sounding_id, imap_file = IDPfile,
    co2_pr_file = CPrfile,
    enable_console_log=True)
