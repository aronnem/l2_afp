import os.path
import collections
import h5py
import numpy as np

# note this is already sorted.
_expected_table_vars = (
    'Phase Function Moment Convention',
    'Source', 'extinction_coefficient',
    'phase_function_moment', 'scattering_coefficient',
    'wave_number')

def _parse_mom_chunk(llist, ctr):
    """
    helper to parse a chunk of a phase function momemt (mom) file.
    This reads the set of all moments. This assumed to be a chunk of 
    data formatted as:

    wavelength N_moments
    6 matrix elements for N = 1
    ...
    6 matrix elements for N = N_moments

    Parameters
    ----------
    llist: list of string lines, loaded from the mom file (should 
        be the entire file loaded into a single list).
    ctr: integer containing the index into llist, for where a chunk of 
        mom data is located.

    Returns
    -------
    wlen:  wavelength (a scalar float)
    Pnn: phase function moment array for the chunk, shaped (N_moments, 6).
    """
    toks = llist[ctr].split()
    if len(toks) != 2:
        raise ValueError('line format mismatch')
    wlen = float(toks[0])
    npts = int(toks[1])
    Pnn = np.zeros((npts,6))
    for n in range(npts):
        Pnn[n,:] = map(float,llist[ctr+1+n].split())
    return wlen, Pnn


def read_mie_mom_file(fileroot):
    """
    reads the Mie and Moments files (extensions .mie, .mom), written 
    from the Mie_moments_program. Note the fileroot should not contain 
    the file extensions, which are automatically added.

    This does skip past the comment lines that contain the metadata
    (though those values should probably be included at some point.)
    
    Parameters
    ----------
    fileroot: a string containing path to mie/mom files.


    Returns
    -------
    dat: a python dictionary with fields:
        wlen, Q_ext, Q_sca, SSA, sigma_ext, Reff: float arrays of shape Nwave
        Pnn: object array of shape Nwave.
        Each element of Pnn is a float array with shape (Nmoment, 6), which 
        forces Pnn itself to be a ragged array since Nmoment changes with 
        wavelength.
    """
    miefile = fileroot+'.mie'
    momfile = fileroot+'.mom'

    tmpdat = np.loadtxt(miefile)

    dat = {}
    dat['wlen'] = tmpdat[:,0]
    dat['Q_ext'] = tmpdat[:,1]
    dat['Q_sca'] = tmpdat[:,2]
    dat['SSA'] = tmpdat[:,3]
    dat['sigma_ext'] = tmpdat[:,4]
    dat['Reff'] = tmpdat[:,5]

    f = open(momfile, 'r')
    llist = [l for l in f]
    f.close()

    ctr = 0
    while llist[ctr].startswith('#'):
        ctr += 1

    dat['Pnn'] = np.zeros(6, dtype=np.object)
    for n in range(6):
        wlen, Pnn = _parse_mom_chunk(llist, ctr)
        if wlen != dat['wlen'][n]:
            print('Wavelen mismatch: {0:f} {1:f}'.format(wlen,dat['slen'][n]))
        dat['Pnn'][n] = Pnn
        ctr = ctr + 1 + Pnn.shape[0]

    return dat


def write_aggregated_properties(out_file, prop_name_out, adat):
    """
    writes the aggregated properties to an h5 file, which contains 
    the scattering properties as a function of r_eff.

    Parameters
    ----------
    out_file: string containing path and filename for the output file.
        it will be opened with mode 'w', implying the file is overwritten
    prop_name_out: string with the name of the property group in the h5 
        file. A subgroup 'Properties' is created with this group, and 
        then the various arrays are created within the subgroup.
        This matches the layout of the l2_aerosol property files.
        Each field has an additional dimension (the r_eff), and one 
        additional variable is written into the file with the r_eff values.
    adat: data dictionary containing property data. 
        see aggregate_data()
    """
    gpath = '/{0:s}/Properties'.format(prop_name_out)
    with h5py.File(out_file, 'w') as h:
        dset = h.create_dataset(
            name = gpath+"/phase_function_moment", 
            shape = adat["phase_function_moment"].shape,
            dtype = np.float64, 
            compression = "gzip",
            compression_opts = 1)
        dset[:] = adat["phase_function_moment"]
        for var in _expected_table_vars + ('r_eff',):
            vpath = gpath+'/'+var
            if var == "phase_function_moment":
                pass
            elif var == "Source":
                # have to convert the Source field from a dtype np.object, 
                # to a dtype S (null-term) string array, as the latter is the 
                # only one that h5py is able to write into HDF5.
                # find the S# dtype for the Source strings; the number 
                # here is the longest len() string in the Source array
                dt = 'S'+str(max([len(s) for s in adat['Source'][0,:]]))
                tmp_array = np.zeros(adat['Source'].shape, dtype=dt)
                tmp_array[:] = adat['Source']
                h[vpath] = tmp_array
            else:
                h[vpath] = adat[var]


def aggregate_properties(mmfile_list, r_eff_list, out_file, prop_name_out):
    """
    aggregate a list of mie, mom files into a single dictionary, 
    to be inserted into h5 combined file. This runs read_mie_mom_file() 
    on each file in the mmfile_list.

    Note we switch the spectral ordering, to make it easier to compare 
    to the existing l2_static_aerosol. The mie/mom are sorted increasing 
    in wavelength, l2_static is increasing in wavenumber.

    Parameters
    ----------
    mmfile_list: a list of mie/mom files. It is assumed to be a list of 
        filenames without the file extensions (.mie, .mom), 
        that are added automatically.
    r_eff_list: a list of the effective radii.
        TODO: I think this could be read directly from the mie file.
    out_file: path and filename for the output aggregated file.
        see write_aggregated_properties().
    prop_name_out: property name in output file.
        see write_aggregated_properties().

    Returns
    -------
    adat: the aggregated properties, with names that match the l2_aerosol 
        h5 file: Source, Phase Function Moment Convention, 
        wave_number, extinction_coefficient, scattering_coefficient, 
        phase_function_moment.
        In addition, adds a key 'r_eff' with the input r_eff_list.
    """

    mmdata = map(read_mie_mom_file, mmfile_list)
    nprops = len(mmfile_list)

    adat = {}
    source_list = [os.path.split(f)[1] for f in mmfile_list]
    adat['Source'] = np.array(source_list).reshape((1,nprops))
    # I am not 100% sure this is right - needs double checking
    adat['Phase Function Moment Convention'] = ["de Rooij"]
    adat['wave_number'] = 1e4/mmdata[0]['wlen'][::-1]

    nwave = mmdata[0]['wlen'].shape[0]
    nmom = []
    for n, w in np.ndindex((nprops, nwave)):
        nmom.append(mmdata[n]['Pnn'][w].shape[0])
    max_nmom = np.max(nmom)

    adat['extinction_coefficient'] = np.zeros((nwave, nprops))
    adat['scattering_coefficient'] = np.zeros((nwave, nprops))
    adat['phase_function_moment'] = np.zeros((nwave, max_nmom, 6, nprops))
    adat['r_eff'] = r_eff_list
    for n, mdat in enumerate(mmdata):
        adat['extinction_coefficient'][:,n] = mdat['Q_ext'][::-1]
        adat['scattering_coefficient'][:,n] = mdat['Q_sca'][::-1]
        for w in range(nwave):
            k = mdat['Pnn'][w].shape[0]
            adat['phase_function_moment'][nwave-w-1,:k,:,n] = mdat['Pnn'][w]

    write_aggregated_properties(out_file, prop_name_out, adat)

    return adat

    

def aggregate_V8_properties(
    V8_static_file, prop_name_list, r_eff_list, 
    out_file, prop_name_out, check_for_mismatch=True):
    """
    Aggregate V8 properties into a single H5 file.
    This is just for comparison to what is aggregated from 
    mie/mom files (the aggregate_properties() function), and it 
    makes an essentially equivalent file.

    Parameters
    ----------
    V8_static_file: path and filename to the V8 l2_aerosol_combined.h5 file
    prop_name_list: list of the string names property groups to aggregate. 
        For example, ['wc_004', 'wc_005', ...]
    r_eff_list: the list of r_eff values for the properties. (this could 
        be derived from the property names themselves, but I'm not sure 
        all contents of the l2_aerosol file are named as the wc/ic groups.)
    out_file: path and filename for the output aggregated file.
        see write_aggregated_properties().
    prop_name_out: property name in output file.
        see write_aggregated_properties().

    check_for_mismatch: specify whether the wave_number and the 
        phase function convention are checked within each entry in the 
        prop_name_list, to make sure these are consistent.
        The intention here was to make sure the list of properties 
        are all similar in a way that it makes sense to aggregate them 
        in a single array. In practice, this does not work due to small 
        changes in the wave_number values (probably floating point roundoff 
        related). Thus, turn this to False to prevent that checking, if 
        needed.

    Returns
    -------
    adat: python dictionary with the aggregated data, same contents as
        aggregate_properties()
    """
    
    dat_list = []
    with h5py.File(V8_static_file, 'r') as h:
        for prop_name in prop_name_list:
            gpath = '/{0:s}/Properties'.format(prop_name)
            pdata = {}
            for var in _expected_table_vars:
                vpath = gpath+'/'+var
                pdata[var] = h.get(vpath)[:]
            dat_list.append(pdata)

    # aggregate data, and double check that things are consistent.
    fixed_vars = ('Phase Function Moment Convention', 'wave_number')
    variable_vars = ('phase_function_moment', 'scattering_coefficient', 
                     'extinction_coefficient', 'Source')
    nmom = [d['phase_function_moment'].shape[1] for d in dat_list]
    max_nmom = np.max(nmom)
    nprops = len(dat_list)
    adat = {}
    for v in fixed_vars:
        adat[var] = dat_list[0][var]
    adat['Source'] = np.zeros((1,nprops), dtype=np.object)
    adat['extinction_coefficient'] = np.zeros((6, nprops))
    adat['scattering_coefficient'] = np.zeros((6, nprops))
    adat['phase_function_moment'] = np.zeros((6, max_nmom, 6, nprops))
    adat['r_eff'] = r_eff_list
    
    for n, dat in enumerate(dat_list):
        if n == 0:
            for v in fixed_vars:
                adat[v] = dat[v]
        else:
            if check_for_mismatch:
                for v in fixed_vars:
                    # double check that the fixed vars are not changing 
                    # within the list of property tables.
                    if np.any(dat[v] != adat[v]):
                        raise ValueError('Mismatch in '+v)
        for v in variable_vars:
            if v == 'phase_function_moment':
                adat[v][:,:nmom[n],:,n] = dat[v]
            else:
                adat[v][...,n] = dat[v]

    write_aggregated_properties(out_file, prop_name_out, adat)

    return adat


def _calc_grid_interpvals(reff_grid, reff):
    """
    helper for interpolation - compute the index and weight value for 
    linear interpolation between reff values in a reff_grid.
    Prints a warning if the limits are hit.
    (TODO - this needs to be revisited, I am not sure how often this 
    issue may occur, and whether it should raise a ValueError)

    Parameters
    ----------
    reff_grid: array like with the effective radii grid points
    reff: scalar float with the desired interpolation value.

    Returns
    -------
    k: scalar integer index into reff_grid
    f: weighting value

    The k and f values should be used to linearly interpolate between 
    any other data value that corresponds to the reff_grid:

    z_interp = f * z[k] + (1-f) * z[k+1]
    """
    nreff = reff_grid.shape[0]
    if reff <= reff_grid[0]:
        print("Warning, reff interp value is out of range (too low)")
        k, f = 0, 1.0
    elif reff >= reff_grid[-1]:
        print("Warning, reff interp value is out of range (too high)")
        k, f = nreff-2, 0.0
    else:
        # in range, and don't have to worry about edge cases.
        k = reff_grid.searchsorted(reff) - 1
        f = (reff_grid[k+1] - reff) / (reff_grid[k+1] - reff_grid[k])
    return k, f


def interpolate_by_reff(h5file, prop_name, reff):
    """
    load a h5 file that contains the grid over r_eff, and interpolate 
    the variable properties to that r_eff value. This is the key function 
    for deriving the scattering properties as a function of r_eff.

    Parameters
    ----------
    h5file: the path and filename to the h5 file containing the scattering 
        property grid aggregated over a range of r_eff values.
    prop_name: string property name for the desired property grid.
    reff: scalar float within the desired reff interpolation value.

    Returns
    -------
    dat: a python dictionary with the standard fields for l2_aerosol 
        scattering properties.
    """

    dat = {}

    with h5py.File(h5file, 'r') as h:
        gpath = '/{0:s}/Properties'.format(prop_name)
        reff_grid = h[gpath+'/r_eff'][:]
        k, f = _calc_grid_interpvals(reff_grid, reff)
        for var in ('phase_function_moment', 'scattering_coefficient',
                    'extinction_coefficient'):
            hvar = h[gpath+'/'+var]
            dat[var] = f*hvar[...,k] + (1-f)*hvar[...,k+1]
        dat['wave_number'] = h[gpath+'/wave_number'][:]
    # note the two strings here need to be lists with 1 element, 
    # to match the layout in the h5 property file.
    # I am not 100% sure this is right - needs double checking
    dat['Phase Function Moment Convention'] = ["de Rooij"]
    dat['Source'] = ['interp_r={0:5.2f}'.format(reff)]
    # trims maybe unneeded zero from end of pf moment array
    max_nmom = np.max( (dat['phase_function_moment'] != 0).sum(1) )
    dat['phase_function_moment'] = dat['phase_function_moment'][:,:max_nmom,:]
    return dat


def write_variable_properties(h5file, prop_name, prop_data):
    """
    writes a set of scattering properties into an l2_aerosol table.

    Parameters
    ----------
    h5file: path and filename for the l2_aerosol file that will actually 
        be referenced in the L2_FP lua config, and utilized by the L2_FP 
        code for scattering property definition
        This file is opened in mode 'a', and the property data in the group 
        prop_name will be overwritten.
    prop_name: the name of the property group in the h5 file.
    prop_data: python dictionary with property data (e.g., output from 
        interpolate_by_reff()
    """

    if tuple(sorted(prop_data.keys())) != _expected_table_vars:
        raise ValueError('input prop_data does not have correct keys')

    # Why does this make a create error?
    with h5py.File(h5file, 'a') as h:
        gpath = '/{0:s}/Properties'.format(prop_name)
        if gpath in h:
            # the group already exists, so we can just write into 
            # existing variables, except for ...
            for var in _expected_table_vars:
                vpath = gpath+'/'+var
                if var == 'phase_function_moment':
                    # ... the phase function, because the array 
                    # can change size. In this case, delete the variable 
                    # and then remake a new variable.
                    del h[vpath]
                    h[vpath] = prop_data[var]
                else:
                    h[vpath][:] = prop_data[var]
        else:
            # creating new variables
            for var in _expected_table_vars:
                vpath = gpath+'/'+var
                # getting error here for 'Phase Function Moment Convention'
                # TypeError: Invalid index for scalar dataset (only ..., () allowed)
                h[vpath] = prop_data[var]


def thin_aerosol_table(in_file, out_file):
    """
    takes the default V8 file and simply thins it down to what we 
    actually need to use. This is basically just the MERRA aerosol 
    types, wc_008, the MODIS ice cloud properties, and the SO 
    (strat aerosol)

    This is just a one shot function to simply the l2_aerosol_combined 
    file down into what we actually need.
    """

    prop_names = ("wc_008", "ice_cloud_MODIS6_deltaM_1000", "strat",
                  "DU", "SS", "BC", "OC", "SO")

    d = collections.OrderedDict()

    with h5py.File(in_file,'r') as h:
        for name in prop_names:
            gpath = '/{0:s}/Properties'.format(name)
            for var in h[gpath]:
                vpath = gpath+'/'+var
                d[vpath] = h.get(vpath)[:]
    
    with h5py.File(out_file,'w') as h:
        for vpath in d:
            h[vpath] = d[vpath]
