#!/usr/bin/env python
# This is a template for a special Level 2 retrieval run. You can use the
# run manager tool or populate.py to set up the full run, this code here
# then goes in as a replacement for the l2_binary_filename.  You may also
# need to update the various paths for running, in examples in the past
# I've needed to update PYTHONPATH, PATH, LD_LIBRARY_PATH, and LUA_PATH.
# I find it easiest to generate the scripts as normal, and then just edit
# the l2_fp_job.sh program that is created.

from full_physics import *
import os, shutil
import h5py

version = "1.0"
usage='''Usage:
  l2_special_run.py [options] <config_file> <output_file>
  l2_special_run.py -h | --help
  l2_special_run.py -v | --version

  This runs a Level 2 retrieval, but before running changing the apriori
  value

Options:
  -h --help         
       Print this message

  -v --version      
       Print program version
'''

args = docopt_simple(usage, version=version)


# Because it is convenient in the wrapper shell script, we get most
# of the arguments from environment variables
# AJM - some of these keywords are not valid - maybe Rob was using a newer L2?
l2run = L2Run(args.config_file, 
              os.environ.get("sounding_id"),
              os.environ.get("ecmwf_file"),
              os.environ.get("spectrum_file"),
#              scene_file = os.environ.get("scene_file"),
              abscodir = os.environ.get("abscodir"),
              merradir = os.environ.get("merradir"), )
#              imap_file = os.environ.get("imap_file"),
#              )

# Get a copy of the original apriori and initial guess
x_a = np.copy(l2run.state_vector.state)
x_i = np.copy(x_a)
cov_a = np.copy(l2run.state_vector.state_covariance)

# Update whatever
for i, svname in enumerate(l2run.state_vector.state_vector_name):
    if(re.match(r'Aerosol Shape Ice Logarithmic Gaussian for Coefficient 1',
                svname)):
        #x_a[i] = -10.0
        pass

# Want initial guess same apriori for this particular test
x_i = np.copy(x_a)

# Setup output. We need to make sure the state is whatever the apriori is
# so all the various 'apriori' fields are correct
l2run.state_vector.update_state(x_a, cov_a)
l2run.config.output_name = args.output_file
out, outerr = l2run.config.output()

# Run sounding
log_timing = fp.LogTiming()
l2run.forward_model.setup_grid()
print((l2run.forward_model))
log_timing.write_to_log("Initialization")
l2run.solver.add_observer(log_timing)
success = False
try:
    res = l2run.solver.solve(x_i, x_a, cov_a)
    l2run.state_vector.update_state(l2run.solver.x_solution, 
                                    l2run.solver.aposteriori_covariance);
    out.write()
    log_timing.write_to_log("Final");
    if(res):
        print("Found Solution")
    else:
        print("Failed to find solution")
    print("Bye bye")
    success = True
except:
    #print("Caught error", file=sys.stderr)
    import traceback
    #print("-" * 25, file=sys.stderr)
    traceback.print_exc()
    #print("-" * 25, file=sys.stderr)
    outerr.write_best_attempt()

# Force flush of output
out = None
l2run = None

# handle apparent bug, file is written to 'out.h5' or 'out.h5.error'
# it also doesn't seem to be related to 'res' - so maybe just attempt 
# this for both files?
out_path, out_file_only = os.path.split(args.output_file)
try:
    os.rename('out.h5', out_file_only)
    shutil.move('./'+out_file_only, out_path)
except OSError:
    pass

try:
    os.rename('out.h5.error', out_file_only+'.error')
    shutil.move('./'+out_file_only+'.error', out_path)
except OSError:
    pass

# Update the initial guess. This is only needed if we use a different
# initial guess. It turns out that the L2 code *always* fill this in with
# the apriori value, not the initial guess. If we have a different initial
# guess, then the generated file will have the wrong value. Easiest thing to
# do is just to change this in place. We should perhaps fix the L2 code at
# some point, but this isn't a huge priority.


# AJM this seems to not work - that variable does not exist (?)
#if success:
#    f = h5py.File(args.output_file, "r+")
#    f["/RetrievedStateVector/state_vector_initial"][0,:] = x_i
#    f.close()
