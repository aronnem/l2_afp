import numpy as np
from scipy.spatial.distance import mahalanobis as mahal
import scipy.sparse

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
            raise ValueError, 'input diagonal must have shape ' + \
                ' (1,N), (N,1) or (N,)'
        if diag.ndim == 2:
            if (diag.shape[0] > 1) and (diag.shape[1] > 1):
                raise ValueError, 'input diagonal must have shape ' + \
                    ' (1,N), (N,1) or (N,)'
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


def bayesian_nonlinear(
    Se, Sa, y, x0, 
    Kupdate, 
    y0 = None, K0 = None, 
    model_params = None, 
    get_matrices = False, 
    max_iteration_ct = 10, 
    convergence_limit = 0.1):

    convergence_met = False
    iter_ct = 1

    if get_matrices:
        mat_save = []

    hatx = x0
    Fhatx = y0
    K = K0

    inv_Sa = np.linalg.inv(Sa)

    while (not convergence_met) & (iter_ct <= max_iteration_ct):

        if (iter_ct > 1) or (K is None):
            Fhatx, K = Kupdate(hatx, model_params)

        # Note - there is some unfortunate interaction between the 
        # sparse matrix and ndarray, if you use np.dot. Plain multiplication 
        # seems to do the right thing (at least with numpy 1.11.3), and 
        # returns a plain ndarray.
        # so, use * for Se, np.dot for the others.

        inv_hatS = np.dot(K.T, Se.getI() * K) + inv_Sa
        # note to self - check if solve is better here?
        # G = linalg.solve(inv_hatS, K.T) * Se.getI()
        hatS = np.linalg.inv(inv_hatS)
        G = np.dot(hatS, K.T * Se.getI())
        A = np.dot(G, K)

        ylocal = y - Fhatx + np.dot(K, (hatx - x0))

        hatx_new = np.dot(G, ylocal) + x0

        if get_matrices:
            mat_save.append({'K':K, 'G':G, 'A':A,
                             'hatx':hatx_new, 'Fhatx':Fhatx})

        if iter_ct > 1:
            # mahal function needs these to be row vectors, 
            # unfortunately.
            conv_test = mahal(hatx.T, hatx_new.T, inv_hatS)
            conv_test = conv_test**2 / x0.shape[0]
            convergence_met = conv_test < convergence_limit

        iter_ct = iter_ct + 1
        hatx = hatx_new

    if get_matrices:
        return hatx, mat_save
    else:
        return hatx
