import numpy as np
from scipy.spatial.distance import mahalanobis as mahal
import scipy.sparse
import h5py

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

def cost_function(y, Fx, inv_Se, xa, x, inv_Sa):
    """
    Evaulates the standard cost function:
    C = (y - F(x))^T Se^{-1} (y - F(x)) + (x - x_a)^T Sa^{-1} (x - x_a)
    
    inputs:
    y, Fx - measurement vector and forward model result at state vector x
    x, x_a - state vector x, and the a priori mean
    inv_Se, inv_Sa - the inverse Measurement and state covariances.
    the inverses are input, for efficiency reasons - since these do not 
    change with iteration, better to have these computed and stored in the 
    caller.

    returns scalar value for Cost function.
    """

    d1 = y - Fx
    d2 = xa - x

    # remember the issue with the inv_Se - this is a matrix so it wants 
    # a normal multiply here, but the result is then an ndarray.
    term1 = np.dot(d1, inv_Se * d1)
    term2 = np.dot(d2, np.dot(inv_Sa, d2))

    return term1 + term2

def _check_convergence(fs_gamma2, fs_last_gamma2, fs_last_gamma2_fc, gamma, iter_ct):
    """
    see connor_convergence.cc, ConnorConvergence::convergence_check()
    from l2_fp lib/Implementation.
    lines 58 - 78 roughly

    this computes the C ratio from ATBD, and tests to see if this is a 
    diverging step. Computes the new gamma parameter (L-M param)
    Returns new gamma, and boolean specifying whether this was diverging step.
    """

    r1 = fs_gamma2 - fs_last_gamma2
    r2 = fs_last_gamma2_fc - fs_last_gamma2
    
    if r1 >= 0.0:
        R = 0.0
    elif abs(r2) < (-r1*1e-20):
        R = 1.0
    else:
        R = r1/r2
        
    gamma_p1 = gamma

    if (R < 0.25) and (iter_ct > 1):
        if gamma > 1e-8:
            gamma_p1 = gamma * 10.0
        else:
            gamma_p1 = 1.0
    elif (R > 0.75) and (iter_ct > 1):
        gamma_p1 = gamma / 2.0

    if (R < 0.0001) and (iter_ct > 1):
        diverged = True
    else:
        diverged = False

    return gamma_p1, diverged


def bayesian_nonlinear_l2fp(
    Se, Sa, y, x0, y0, K0, Kupdate, 
    start_gamma = 10.0, 
    model_params = None, 
    max_iteration_ct = 10, 
    convergence_limit = 0.1, 
    max_num_diverged = 2,     
    debug_write=True):
    """
    Main function that replicated l2_fp solver.
    See connor_solver.cc in lib/Implementation.

    """

    # startup counters.
    convergence_met = False
    iter_ct = 0
    file_ct = 1
    num_diverged = 0
    gamma = start_gamma

    # prepare lists (using append on these)
    cost_function_values = []
    gamma_values = []
    divergence_status = []

    # precompute matrix inversions, etc.
    Sa_sigma = np.sqrt(np.diag(Sa))
    N = np.diag(Sa_sigma)
    inv_N = np.diag(1.0/np.sqrt(np.diag(Sa)))

    inv_Sa = np.linalg.inv(Sa)
    Sa_scaled = np.dot(inv_N, np.dot(Sa, inv_N))
    inv_Sa_scaled = np.linalg.inv(Sa_scaled)

    # values for the initial iteration.
    Fhatx = y0
    K = K0
    hatx = x0
    cost_function_values.append(
        cost_function(y, Fhatx, Se.getI(), x0, x0, inv_Sa))
    gamma_values.append(gamma)

    # main convergence loop. Note the 3 possible exit criteria
    while ( (not convergence_met) & 
            (iter_ct <= max_iteration_ct) & 
            (num_diverged <= max_num_diverged) ):

        # Note - there is some unfortunate interaction between the 
        # sparse matrix and ndarray, if you use np.dot. Plain multiplication 
        # seems to do the right thing (at least with numpy 1.11.3), and 
        # returns a plain ndarray.
        # so, use * for Se, np.dot for the others.

        # the layout here is copied from the connor_solver cpp code.
        KtSeK = np.dot(K.T, Se.getI() * K)
        residual = y - Fhatx

        lhs = (1+gamma) * inv_Sa_scaled + np.dot(N, np.dot(KtSeK,N))
        rhs = np.dot(N, np.dot(K.T, Se.getI() * residual))

        dx_scaled = np.linalg.solve(lhs, rhs)
        dx = dx_scaled * Sa_sigma

        hatx_p1 = hatx + dx
        Fhatx_p1, K_p1 = Kupdate(hatx_p1, model_params)

        # cost function forecast - uses Fhatx + linear extrapolation using 
        # K at this step.
        Fhatx_forecast = Fhatx + np.dot(K, dx)
        cost_function_forecast = \
            cost_function(y, Fhatx_forecast, Se.getI(), 
                          x0, hatx_p1, inv_Sa)

        # convergence testing.
        cost_function_values.append(
            cost_function(y, Fhatx_p1, Se.getI(), 
                          x0, hatx_p1, inv_Sa))

        gamma_p1, diverged = _check_convergence(
            cost_function_values[iter_ct], 
            cost_function_values[iter_ct-1], 
            cost_function_forecast, gamma, iter_ct)

        gamma_values.append(gamma_p1)
        divergence_status.append(diverged)

        if debug_write:
            # painfully writing out every matrix, for testing purposes.
            print('Writing K, Fhatx for iter ' + str(iter_ct)) + ' file ' + str(file_ct)
            fname = 'l2ret_debug_file{0:03d}.h5'.format(file_ct)
            h = h5py.File(fname, 'w')
            h.create_dataset('N', data=N)
            h.create_dataset('KtSeK', data=KtSeK)
            h.create_dataset('Sa', data=Sa)
            h.create_dataset('Sa_scaled', data=Sa_scaled)
            h.create_dataset('K', data=K)
            h.create_dataset('K_p1', data=K_p1)
            h.create_dataset('residual', data=residual)
            h.create_dataset('lhs', data=lhs)
            h.create_dataset('rhs', data=rhs)
            h.create_dataset('dx', data=dx)
            h.create_dataset('Fhatx', data=Fhatx)
            h.create_dataset('Fhatx_p1', data=Fhatx_p1)
            h.create_dataset('hatx', data=hatx)
            h.create_dataset('hatx_p1', data=hatx_p1)
            h.create_dataset('cost_func', data=cost_function_values)
            h.create_dataset('gamma', data=gamma_values)
            h.create_dataset('divergence', data=divergence_status)
            h.create_dataset('iter_ct', data=iter_ct)
            h.create_dataset('num_diverged', data=num_diverged)
            h.close()
            file_ct += 1

        gamma = gamma_p1
        if diverged:
            num_diverged += 1            
        else:
            iter_ct += 1
            Fhatx = Fhatx_p1
            hatx = hatx_p1
            K = K_p1

        print ('Iter Stat: ', iter_ct,  ' <= ', max_iteration_ct, ' ; ', 
               num_diverged, ' <= ', max_num_diverged)

    return hatx
