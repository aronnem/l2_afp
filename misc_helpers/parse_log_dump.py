#
# Parse the log dump - the console output (via c++ cout operator), 
# that was prefixed with "V:"
# write the parsed ndarray into a npz file.
#
import numpy as np

vard = {}

f = open(logfilename, 'r')

# just run to EOF I guess
done = False

while not done:

    try:
        line = f.next().strip()
    except StopIteration:
        break

    if line.startswith('V:'):

        toks = line.split('=')[1].split('x')
        if len(toks) == 2:
            N = int(toks[0]) * int(toks[1])
            shape = int(toks[0]) , int(toks[1])
        else:
            N = int(toks[0])
            shape = (N,)
        varname = line.split('=')[0].strip()[2:]

        if varname in vard:
            pass
        else:
            vard[varname] = np.zeros((shape) + (0,))

        tmp_list = []
        while ']' not in line:
            line = f.next().strip()
            if line.startswith('['):
                nums = map(float, line.split()[1:])
            elif line.endswith(']'):
                nums = map(float, line.split()[:-1])
            else:
                nums = map(float, line.split())
            tmp_list += nums

        tmp_var = np.array(tmp_list)

        tmp_var_reshaped = np.reshape(
            tmp_var, shape + (1,))
        vard[varname] = np.concatenate(
            [vard[varname], tmp_var_reshaped], axis=-1)


np.savez('log_variables.npz', **vard)
