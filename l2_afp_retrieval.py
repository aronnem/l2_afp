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

def _check_convergence(fs_gamma2, fs_last_gamma2, 
                       fs_last_gamma2_fc, gamma, iter_ct, 
                       debug_print = False):
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

    if (R < 0.25) and (iter_ct > 0):
        if gamma > 1e-8:
            gamma_p1 = gamma * 10.0
        else:
            gamma_p1 = 1.0
    elif (R > 0.75) and (iter_ct > 0):
        gamma_p1 = gamma / 2.0

    if (R < 0.0001) and (iter_ct > 0):
        diverged = True
    else:
        diverged = False

    if debug_print:
        print('  -> fs_last.gamma2 = ' + str(fs_last_gamma2))
        print('  -> fs_last.gamma2_fc = ' + str(fs_last_gamma2_fc))
        print('  -> fs.gamma2 = ' + str(fs_gamma2))
        print('  -> r1, r2, ratio = ' + str(r1) + ',' + str(r2) + ',' + str(R))
        print('  -> gamma last = ' + str(gamma))
        print('  -> gamma next = ' + str(gamma_p1))

    return gamma_p1, diverged


def _debug_write_file(fname, 
                      N, Sa, Sa_scaled, x0, y, 
                      Fx_i, K_i, x_i, x_prev, residual_i, residual_prev, 
                      cost_function_values, cost_function_value_forecasts, 
                      gamma_values, divergence_status, 
                      iter_ct, num_diverged):
    """
    helper to just dump out the contents of the various arrays
    """
    h = h5py.File(fname, 'w')
    h.create_dataset('N', data=N)
    h.create_dataset('Sa', data=Sa)
    h.create_dataset('Sa_scaled', data=Sa_scaled)
    h.create_dataset('x0', data=x0)
    h.create_dataset('y', data=y)
    h.create_dataset('Fx_i', data=Fx_i)
    h.create_dataset('K_i', data=K_i)
    h.create_dataset('x_i', data=x_i)
    h.create_dataset('x_prev', data=x_prev)
    h.create_dataset('residual_i', data=residual_i)
    h.create_dataset('residual_prev', data=residual_prev)
    h.create_dataset('cost_func', data=cost_function_values)
    h.create_dataset('cost_func_fc', data=cost_function_value_forecasts)
    h.create_dataset('gamma', data=gamma_values)
    h.create_dataset('divergence', data=divergence_status)
    h.create_dataset('iter_ct', data=iter_ct)
    h.create_dataset('num_diverged', data=num_diverged)
    h.close()


def _do_inversion(residual, K, Se, N, inv_Sa_scaled, Sa_sigma, gamma):
    """
    mimics the do_inversion in connor_solver.
    Useful to have this as a helper function, since we basically need to 
    run it in two different ways at one iteration, depending on the 
    divergence status
    """

    # Note - there is some unfortunate interaction between the 
    # sparse matrix and ndarray, if you use np.dot. Plain multiplication 
    # seems to do the right thing (at least with numpy 1.11.3), and 
    # returns a plain ndarray.
    # so, use * for Se, np.dot for the others.

    KtSeK = np.dot(K.T, Se.getI() * K)
    
    lhs = (1+gamma) * inv_Sa_scaled + np.dot(N, np.dot(KtSeK,N))
    rhs = np.dot(N, np.dot(K.T, Se.getI() * residual))
    
    dx_scaled = np.linalg.solve(lhs, rhs)
    dx = dx_scaled * Sa_sigma

    d_sigma_sq = np.dot(rhs, dx_scaled)
    d_sigma_sq_scaled = d_sigma_sq / K.shape[1]

    S_i = np.linalg.inv(lhs)

    return dx, d_sigma_sq_scaled, S_i


def bayesian_nonlinear_l2fp(
    Se, Sa, y, x0, Kupdate, 
    start_gamma = 10.0, 
    model_params = None, 
    max_iteration_ct = 10, 
    convergence_thresh = 0.2, 
    max_num_diverged = 2,     
    debug_write=False, 
    debug_write_prefix='l2ret_debug', 
    match_l2_fp_costfunc=True):
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

    # initialize lists (using append on these)
    cost_function_values = []
    cost_function_value_forecasts = []
    gamma_values = []
    divergence_status = []

    # precompute matrix inversions, etc.
    Sa_sigma = np.sqrt(np.diag(Sa))
    N = np.diag(Sa_sigma)
    inv_N = np.diag(1.0/np.sqrt(np.diag(Sa)))

    inv_Sa = np.linalg.inv(Sa)
    Sa_scaled = np.dot(inv_N, np.dot(Sa, inv_N))
    inv_Sa_scaled = np.linalg.inv(Sa_scaled)

    # prep list values.
    cost_function_values.append(0.0)
    cost_function_value_forecasts.append(0.0)
    gamma_values.append(gamma)

    last_cf = cost_function_values[0]
    last_cf_forecast = cost_function_value_forecasts[0]

    x_i = x0
    x_prev = x0

    # make a list of the x_i at each step; make a copy just to make 
    # sure nothing gets modified later.
    x_i_list = []
    x_i_list.append(x_i.copy())
    S_i_list = []
    S_i_list.append(Sa.copy())

    # main convergence loop. Note the 3 possible exit criteria
    #
    # The important thing to realize here, is that the Kupdate is the 
    # expensive step, but that we need it to evaluate the true value 
    # of the cost function at any candidate state vector value.
    #
    # for the initial iteration (iter_ct = 0), this is really just 
    # evaluating the forward model at the prior mean.
    #
    while ( (not convergence_met) & 
            (iter_ct <= max_iteration_ct) & 
            (num_diverged <= max_num_diverged) ):

        if debug_write:
            print('  -> Ps = x_i[21] ' + str(x_i[21]/100))

        # evaluate forward model, and update the residual.
        Fx_i, K_i = Kupdate(x_i, model_params)
        residual_i = y - Fx_i

        # this allows the true cost function to be computed at the 
        # at newly evaluated F(x_i)
        new_cf = cost_function(y, Fx_i, Se.getI(), x0, x_i, inv_Sa)

        # we can now evaluate the divergence, with this new cost function value.
        gamma_p1, diverged = _check_convergence(
            new_cf, last_cf, last_cf_forecast, gamma, iter_ct, 
            debug_print=debug_write)

        gamma = gamma_p1

        if diverged:
            # This means the previous dx increment, that brought us to x_i, 
            # increased the cost function. So, it needs to be removed, and then a 
            # new x_i is recomputed with the new gamma.
            #
            # note, this should not happen on first iteration (assuming 
            # _check_convergence() is implemented such that iter_ct = 0 means 
            # it cannot produce a convergence.)

            num_diverged += 1
            dx_i, d_sigma_sq_scaled, S_i = _do_inversion(
                residual_prev, K_prev, Se, N, inv_Sa_scaled, Sa_sigma, gamma)

            # cost function forecast - uses Fhatx + linear extrapolation using 
            # K at this step.
            # note the forecast isn't used during this iteration, but will be used on 
            # the next iteration, to assess the linearity of this state update
            x_i = x_prev + dx_i
            if match_l2_fp_costfunc:
                # fpr l2_fp, the last forecast isn't updated, so just a pass here.
                # I think that is maybe wrong, but this is perhaps very minor?
                pass
            else:
                Fx_i_fc = Fx_prev + np.dot(K_prev, dx_i)
                last_cf_forecast = cost_function(y, Fx_i_fc, Se.getI(), x0, x_i, inv_Sa)
            # does not update last_cf - since we are restarting from the x_prev, 
            # the last_cf value does not change.

        else:

            # not diverging, so proceed normally, by updating to i+1
            iter_ct += 1

            # stash successful state update into "previous"
            x_prev = x_i
            residual_prev = residual_i
            K_prev = K_i
            Fx_prev = Fx_i

            dx_ip1, d_sigma_sq_scaled, S_i = _do_inversion(
                residual_i, K_i, Se, N, inv_Sa_scaled, Sa_sigma, gamma)
            
            # cost function forecast - uses Fhatx + linear extrapolation using 
            # K at this step.
            # note the forecast isn't used during this iteration, but will be used on 
            # the next iteration, to assess the linearity of the state update
            x_ip1 = x_i + dx_ip1
            Fx_ip1_fc = Fx_i + np.dot(K_i, dx_ip1)

            last_cf_forecast = cost_function(y, Fx_ip1_fc, Se.getI(), x0, x_ip1, inv_Sa)
            last_cf = new_cf
            x_i = x_ip1

            if d_sigma_sq_scaled < convergence_thresh:
                convergence_met = True

        x_i_list.append(x_i.copy())
        S_i_list.append(S_i.copy())

        if debug_write:
            print('  -> D_sigma_sq_scaled: ' + str(d_sigma_sq_scaled))
        cost_function_values.append(new_cf)
        cost_function_value_forecasts.append(last_cf_forecast)
        gamma_values.append(gamma_p1)
        divergence_status.append(diverged)

        

        if debug_write:
            # painfully writing out every matrix, for testing purposes.
            print('Writing K, Fhatx for iter ' + str(iter_ct)) + ' file ' + str(file_ct)
            fname = debug_write_prefix+'_file{0:03d}.h5'.format(file_ct)
            _debug_write_file(fname, 
                              N, Sa, Sa_scaled, x0, y, 
                              Fx_i, K_i, x_i, x_prev, residual_i, residual_prev, 
                              cost_function_values, cost_function_value_forecasts, 
                              gamma_values, divergence_status, 
                              iter_ct, num_diverged)
            file_ct += 1

            
            print (
                'Iter Stat: {0:d}, {1:d} max iter'.format(iter_ct,max_iteration_ct)+
                ', divergences: {0:d}, {1:d} max'.format(num_diverged, max_num_diverged))

    return x_i_list, S_i_list
