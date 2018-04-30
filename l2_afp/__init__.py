# this import needs to happen first, for reasons I don't understand.
# just stash this in a hidden underscore variable, since it has no 
# use interactively.
import full_physics_swig as _full_physics_swig

import full_physics

import os.path

_pkg_home_dir = os.path.abspath(os.path.dirname(__file__))
_lua_config_dir = os.path.join(_pkg_home_dir, 'lua_configs')

from . import l2_afp_retrieval
from .l2_afp_retrieval import bayesian_nonlinear_solver

from . import fp_wrapper
from .fp_wrapper import wrapped_fp

from . import utils
