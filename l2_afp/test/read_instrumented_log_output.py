import collections
import numpy as np

def _extract_array(alines, n):

    n2 = n+1
    x = []

    if 'x' in alines[n2]:
        aline_toks = alines[n2].split()
        array_shape = int(aline_toks[0]),int(aline_toks[2])
    else:
        array_shape = (int(alines[n2].strip()),)

    array_len = np.prod(array_shape)
    x = np.zeros(array_len)
    i1 = 0
    n2 += 1
    while n2 < len(alines):
        if alines[n2].startswith('AJM_instr'):
            break
        else:
            aline_clean = alines[n2].replace('[','').replace(']','')
            aline_toks = aline_clean.split()
            i2 = i1 + len(aline_toks)
            x[i1:i2] =  list(map(float,aline_toks))
            i1 = i2
        n2 += 1

    nshift = n2 - n
    x = np.reshape(x,array_shape)
    
    return nshift, x

def read_l2_log(logfile):

    with open(logfile, 'r') as f:
        alines = [l for l in f]

    vardata = collections.defaultdict(dict)

    n_max = len(alines)
    n = 0

    while n < n_max:

        a = alines[n]
        atoks = a.split()

        if a.startswith('AJM_instr'):
            func_name = atoks[1]
            if atoks[2] == 'begin':
                var_name = atoks[3]
                tmp_n, tmp_array = _extract_array(alines, n)
                if var_name in vardata[func_name]:
                    vardata[func_name][var_name].append(tmp_array)
                else:
                    vardata[func_name][var_name] = [tmp_array]
                n += tmp_n
            else:
                var_name = atoks[2]
                tmp_value = float(atoks[3])
                if var_name in vardata[func_name]:
                    vardata[func_name][var_name].append(tmp_value)
                else:
                    vardata[func_name][var_name] = [tmp_value]

        n += 1


    return vardata
