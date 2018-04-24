import tables
import numpy as np

def state_vector_splitting(x0, hatx, hatx_u, svnames):
    """
    helper function to do the tedious work of splitting apart the 
    various state vector components into individual variables to be written 
    to the output h5's.

    Note this output should match the normal 'l2_aggregate.h5' but not 
    the 'l2_plus_more' that includes RetrievalGeometry and regroups variables 
    into subgroups 'AerosolResults', 'AlbedoResults', etc.

    Parameters
    ----------
    x0: the a priori state, with N elements
    hatx: the retrieved state, with N elements
    hatx_u: the uncertainty (computed from the retrieval a posteriori
        covariance)
    svnames: array like with state vector names

    All inputs must have the same length and indexing, so the nth 
    values all correspond to the same actual variable.

    Returns
    -------
    sdat: the python data dictionary containing the split up state variables.
        The key names are now the paths into the single sounding h5 file, 
        and the values are the corresponding state variable values (or 
        a priori values or uncertainties).
    """

    if (x0.shape[0] != len(svnames)) or (hatx.shape[0] != len(svnames)):
        raise ValueError('length mismatch in state vector or variable name list')

    sdat = {}

    # assumes sv elements [:20] are the co2 profile
    g = '/RetrievalResults/'
    sdat[g+'co2_profile'] = hatx[:20]
    sdat[g+'co2_profile_apriori'] = x0[:20]
    sdat[g+'co2_vertical_gradient_delta'] = [(x0[13]-hatx[13])-(x0[-1]-hatx[-1])]

    # assumes this ordering is always fixed
    aerosol_types = ('DU', 'SS', 'BC', 'OC', 'SO', 'Ice', 'Water', 'ST')

    # a handy mapping from svnames to the h5 names
    bandname_map = {'A-Band':'_o2', 'WC-Band':'_weak_co2', 
                    'SC-Band':'_strong_co2'}

    # loop over remaining variables, and convert to dictionary key/values 
    # with hardcoded mappings from the state vector.
    # This cannot be automated more cleanly because there are various 
    # inconsistencies in naming (the fph, and where uncert is inserted 
    # in the var name - sometimes suffix, some not)

    for k in range(20, len(svnames)):

        # note, the seemingly extra [] on the indexing into the state 
        # vector is to ensure we have 1-element arrays rather than 
        # scalar elements.
        n = svnames[k]

        # atmosphere
        g = '/RetrievalResults/'
        if n == 'H2O Scaling factor':
            sdat[g+'h2o_scale_factor'] = hatx[[k]]
            sdat[g+'h2o_scale_factor_uncert'] = hatx_u[[k]]
            sdat[g+'h2o_scale_factor_apriori'] = x0[[k]]
        if n == 'Surface Pressure (Pascals)':
            sdat[g+'surface_pressure_fph'] = hatx[[k]]
            sdat[g+'surface_pressure_uncert_fph'] = hatx_u[[k]]
            sdat[g+'surface_pressure_apriori_fph'] = x0[[k]]
        if n == 'Temperature Offset (Kelvin)':
            sdat[g+'temperature_offset_fph'] = hatx[[k]]
            sdat[g+'temperature_offset_uncert_fph'] = hatx_u[[k]]
            sdat[g+'temperature_offset_apriori_fph'] = x0[[k]]

        # EOFs, with some automation
        if n.startswith('EOF order'):
            eof_N = n.split()[2]
            suffix = bandname_map[n.split()[-1]]
            sdat[g+'eof_'+eof_N+'_scale'+suffix] = hatx[[k]]
            sdat[g+'eof_'+eof_N+'_scale_uncert'+suffix] = hatx_u[[k]]
            sdat[g+'eof_'+eof_N+'_scale_apriori'+suffix] = x0[[k]]

        # C-M wind speed
        if n == 'Ground Coxmunk Windspeed':
            sdat[g+'wind_speed'] = hatx[[k]]
            sdat[g+'wind_speed_uncertainty'] = hatx_u[[k]]
            sdat[g+'wind_speed_apriori'] = x0[[k]]

        # fluor
        if n == 'Fluorescence Surface Coefficient 1':
            sdat[g+'fluorescence_at_reference'] = hatx[[k]]
            sdat[g+'fluorescence_at_reference_uncert'] = hatx_u[[k]]
            sdat[g+'fluorescence_at_reference_apriori'] = x0[[k]]
        if n == 'Fluorescence Surface Coefficient 2':
            sdat[g+'fluorescence_slope'] = hatx[[k]]
            sdat[g+'fluorescence_slope_uncert'] = hatx_u[[k]]
            sdat[g+'fluorescence_slope_apriori'] = x0[[k]]

        # aerosols are a bit complicated, skip for now
        g = '/AerosolResults/'

        # BRDF params
        g = '/BRDFResults/'
        if n == 'Ground BRDF Soil A-Band BRDF Weight Intercept':
            sdat[g+'brdf_weight_o2'] = hatx[[k]]
            sdat[g+'brdf_weight_uncert_o2'] = hatx_u[[k]]
            sdat[g+'brdf_weight_apriori_o2'] = x0[[k]]
        if n == 'Ground BRDF Soil A-Band BRDF Weight Slope':
            sdat[g+'brdf_weight_slope_o2'] = hatx[[k]]
            sdat[g+'brdf_weight_slope_uncert_o2'] = hatx_u[[k]]
            sdat[g+'brdf_weight_slope_apriori_o2'] = x0[[k]]     
        if n == 'Ground BRDF Soil WC-Band BRDF Weight Intercept':
            sdat[g+'brdf_weight_weak_co2'] = hatx[[k]]
            sdat[g+'brdf_weight_uncert_weak_co2'] = hatx_u[[k]]
            sdat[g+'brdf_weight_apriori_weak_co2'] = x0[[k]]
        if n == 'Ground BRDF Soil WC-Band BRDF Weight Slope':
            sdat[g+'brdf_weight_slope_weak_co2'] = hatx[[k]]
            sdat[g+'brdf_weight_slope_uncert_weak_co2'] = hatx_u[[k]]
            sdat[g+'brdf_weight_slope_apriori_weak_co2'] = x0[[k]]
        if n == 'Ground BRDF Soil SC-Band BRDF Weight Intercept':
            sdat[g+'brdf_weight_strong_co2'] = hatx[[k]]
            sdat[g+'brdf_weight_uncert_strong_co2'] = hatx_u[[k]]
            sdat[g+'brdf_weight_apriori_strong_co2'] = x0[[k]]
        if n == 'Ground BRDF Soil SC-Band BRDF Weight Slope':
            sdat[g+'brdf_weight_slope_strong_co2'] = hatx[[k]]
            sdat[g+'brdf_weight_slope_uncert_strong_co2'] = hatx_u[[k]]
            sdat[g+'brdf_weight_slope_apriori_strong_co2'] = x0[[k]]

        # Possible ToDo:
        # compute brdf_reflectance

        # albedo (in use with ocean surface model)
        # can use some automation here
        g = '/AlbedoResults/'
        if n.startswith('Ground Lambertian') and n.endswith('1'):
            prefix = 'albedo_'
            suffix = bandname_map[n.split()[2]]
            sdat[g+prefix+suffix+'_fph'] = hatx[[k]]
            sdat[g+prefix+'_uncert'+suffix+'_fph'] = hatx_u[[k]]
            sdat[g+prefix+'_apriori'+suffix+'_fph'] = x0[[k]]

        if n.startswith('Ground Lambertian') and n.endswith('2'):
            prefix = 'albedo_slope_'
            suffix = bandname_map[n.split()[2]]
            sdat[g+prefix+suffix] = hatx[[k]]
            sdat[g+prefix+'uncert'+suffix] = hatx_u[[k]]
            sdat[g+prefix+'apriori'+suffix] = x0[[k]]

        # dispersions
        g = '/DispersionResults/'
        if n == 'Instrument Dispersion A-Band Offset':
            sdat[g+'dispersion_offset_o2'] = hatx[[k]]
            sdat[g+'dispersion_offset_uncert_o2'] = hatx_u[[k]]
            sdat[g+'dispersion_offset_apriori_o2'] = x0[[k]]
        if n == 'Instrument Dispersion A-Band Scale':
            sdat[g+'dispersion_spacing_o2'] = hatx[[k]]
            sdat[g+'dispersion_spacing_uncert_o2'] = hatx_u[[k]]
            sdat[g+'dispersion_spacing_apriori_o2'] = x0[[k]]
        if n == 'Instrument Dispersion WC-Band Offset':
            sdat[g+'dispersion_offset_weak_co2'] = hatx[[k]]
            sdat[g+'dispersion_offset_uncert_weak_co2'] = hatx_u[[k]]
            sdat[g+'dispersion_offset_apriori_weak_co2'] = x0[[k]]
        if n == 'Instrument Dispersion WC-Band Scale':
            sdat[g+'dispersion_spacing_weak_co2'] = hatx[[k]]
            sdat[g+'dispersion_spacing_uncert_weak_co2'] = hatx_u[[k]]
            sdat[g+'dispersion_spacing_apriori_weak_co2'] = x0[[k]]
        if n == 'Instrument Dispersion SC-Band Offset':
            sdat[g+'dispersion_offset_strong_co2'] = hatx[[k]]
            sdat[g+'dispersion_offset_uncert_strong_co2'] = hatx_u[[k]]
            sdat[g+'dispersion_offset_apriori_strong_co2'] = x0[[k]]
        if n == 'Instrument Dispersion SC-Band Scale':
            sdat[g+'dispersion_spacing_strong_co2'] = hatx[[k]]
            sdat[g+'dispersion_spacing_uncert_strong_co2'] = hatx_u[[k]]
            sdat[g+'dispersion_spacing_apriori_strong_co2'] = x0[[k]]

    return sdat


def l2_varname_compare(l2_file1, l2_file2):
    """
    helper to compare the list of variables within two h5 files. this is 
    intended to help track what variables are still lacking from the 
    l2_afp's single sounding output, for example.

    Parameters
    ----------
    l2_file1, l2_file2: two path+filenames to h5 files to compare. 
        Should be l2 single sounding output, but this is actually written 
        pretty generically so it might work on general h5 files.
        But, note that it does not recurse deeper than 1 group level.

    Returns
    -------
    both_item_list: items in both files
    file1_itemlist: items that reside in only file1
    file2_itemlist: items that reside in only file2

    the item lists contain strings that are formatted like:
    ['G /Groupname', 'V '/Groupname/variable'], with the G and V to 
    specify the groups and variables (e.g. Array objects).
    """

    h1 = tables.open_file(l2_file1, 'r')
    h2 = tables.open_file(l2_file2, 'r')


    def list_union(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        setu = set1.union(set2)
        return list(setu)

    groupnames1 = [g._v_name for g in h1.list_nodes('/', 'Group')]
    groupnames2 = [g._v_name for g in h2.list_nodes('/', 'Group')]
    groupnames = list_union(groupnames1, groupnames2)

    both_item_list = []
    file1_item_list = []
    file2_item_list = []

    for g in groupnames:
        if (g in h1.root) and (g in h2.root):
            both_item_list.append('G /'+g)
            varnames1 = [v._v_name for v in h1.list_nodes('/'+g,'Array')]
            varnames2 = [v._v_name for v in h2.list_nodes('/'+g,'Array')]
            varnames = list_union(varnames1, varnames2)
            for v in varnames:
                if (v in h1.get_node('/'+g)) and (v in h2.get_node('/'+g)):
                    both_item_list.append('V /'+g+'/'+v)
                elif v in h1.get_node('/'+g):
                    file1_item_list.append('V /'+g+'/'+v)
                elif v in h2.get_node('/'+g):
                    file2_item_list.append('V /'+g+'/'+v)
        elif g in h1.root:
            file1_item_list.append('G /'+g)
        elif g in h2.root:
            file2_item_list.append('G /'+g)

    h1.close()
    h2.close()

    return both_item_list, file1_item_list, file2_item_list

