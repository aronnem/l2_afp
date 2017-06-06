import numpy as np

def blk_diag(S_list):
    """
    mostly brute force block diagonal creation.
    assumes a list of S matrices, all must be 2D
    no type checking is done - output will be float native.
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
        SD[j:j+j_S,k:+k_S] = S
        j += j_S
        k += k_S
    return SD
