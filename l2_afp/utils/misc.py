import pkg_resources
import numpy as np
import scipy.sparse
from .. import _lua_config_dir
import os.path


class diagcmatrix(scipy.sparse.dia_matrix):
    """
    A wrapped scipy.sparse.dia_matrix, with a slightly nicer argument 
    format for creation (this assumes the input is a single vector, and we 
    create a diagonal matrix with that as the center diagonal.), 
    and an added, cached getI() method.

    NOTE: the set methods are not monitored, so the getI will break if those 
    are used. (ToDo list)
    """
    def __init__(self, diag, dtype=None, copy=False):
        self._cachedI = None
        diag = np.asarray(diag)
        if diag.ndim > 2:
            raise ValueError('input diagonal must have shape ' + \
                ' (1,N), (N,1) or (N,)')
        if diag.ndim == 2:
            if (diag.shape[0] > 1) and (diag.shape[1] > 1):
                raise ValueError('input diagonal must have shape ' + \
                    ' (1,N), (N,1) or (N,)')
            if diag.shape[1] > 1:
                diag = diag.reshape((diag.shape[1],))
            else:
                diag = diag.reshape((diag.shape[0],))
                
        scipy.sparse.dia_matrix.__init__(
            self, (diag, 0), shape = (diag.shape[0], diag.shape[0]), 
            dtype = dtype, copy = copy)

    def getI(self):
        if self._cachedI is None:
            self._computeI()
        return self._cachedI

    def _computeI(self):
        Idiag = 1.0/self.diagonal()
        self._cachedI = scipy.sparse.dia_matrix( (Idiag,0), self.shape )


def blk_diag(S_list):
    """
    mostly brute force block diagonal creation.
    assumes a list of S matrices, all must be 2D
    no type checking is done - output will be float native.

    Parameters:
    S_list: a list of 2D ndarrays that will be blocked into a single 2D array.

    Returns:
    SD, a simple 2D array with shape chosen to contain all arrays from 
        the input S_list.
    """
    j_max = 0
    k_max = 0
    for S in S_list:
        if S.ndim != 2:
            raise ValueError('All input arrays must be 2D')
        j_max += S.shape[0]
        k_max += S.shape[1]

    SD = np.zeros((j_max, k_max))
    j = 0
    k = 0
    for S in S_list:
        j_S = S.shape[0]
        k_S = S.shape[1]
        SD[j:j+j_S,k:k+k_S] = S
        j += j_S
        k += k_S
    return SD


def get_lua_config_files():
    """
    finds the abs paths to the lua configs that are included 
    in l2_afp. Uses pkg_resources.

    Returns a python dictionary with several Lua configs:
    'default' - the default V8 config
    'default_ABSCOv4.2' - V8, but reverting to ABSCO version 4.2
    'watercloud_reff' - V8, but with added config for water cloud 
        r_eff retrieval.
    """

    # This is a hardcoded lists of the Lua files that are present, 
    # so this is less than idea - needs some fixing.

    lua_configs = {}

    # method via pkg_resources leaves us with a relative dirname, if 
    # we did the import of the package inside the dir (which may be done 
    # during testing/devel)...
    #lua_configs['default'] = pkg_resources.resource_filename(
    #    'l2_afp', 'lua_configs/custom_config_default.lua')
    #lua_configs['default_ABSCOv4.2'] = pkg_resources.resource_filename(
    #    'l2_afp', 'lua_configs/custom_config_absco42.lua')
    #lua_configs['watercloud_reff'] = pkg_resources.resource_filename(
    #    'l2_afp', 'lua_configs/custom_config_watercloud_reff.lua')

    # ... other method, using the abs path derived in the package init
    # should work in either case.
    lua_configs['default'] = os.path.join(
        _lua_config_dir, 'custom_config_default.lua')
    lua_configs['default_ABSCOv4.2'] = os.path.join(
        _lua_config_dir, 'custom_config_absco42.lua')
    lua_configs['fixed_aerosol'] = os.path.join(
        _lua_config_dir, 'custom_config_fixed_aerosol.lua')
    lua_configs['watercloud_reff'] = os.path.join(
        _lua_config_dir, 'custom_config_watercloud_reff.lua')
    lua_configs['watercloud_only'] = os.path.join(
        _lua_config_dir, 'custom_config_watercloud_only.lua')

    return lua_configs

    
