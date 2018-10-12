# I don't think we use the _swig module directly, but it needs to be 
# importable for the full_physics to function properly.
import full_physics_swig
import full_physics
import h5py

from .utils import blk_diag, output_translation, get_lua_config_files
from .utils import scattering_properties

import numpy as np

from builtins import str

# these imports are for some file management done by the r_eff derivative 
# variant. ideally, try to remove these, if we can find a cleaner method.
import shutil, os.path

def get_sounding_id_from_framefp(L1bfile, frame_number, footprint):
    """
    convenience function to get sounding_id from frame_number and footprint.
    note that frame_number and footprint are assumed to be 1-ordered.

    assumes scalar inputs.
    """
    varpath = '/SoundingGeometry/sounding_id'
    with h5py.File(L1bfile) as hfile:
        sounding_id = hfile.get(varpath)[frame_number-1, footprint-1]
    sounding_id = str(sounding_id)

    return sounding_id

def get_framefp_from_sounding_id(L1bfile, sounding_id):
    varpath = '/SoundingGeometry/sounding_id'
    with h5py.File(L1bfile) as hfile:
        sounding_id_array = hfile.get(varpath)[:]
    frame_number_a, footprint_a = (sounding_id == sounding_id_array).nonzero()
    if frame_number_a.shape[0] == 0:
        raise ValueError('sounding_id not found in file')
    if frame_number_a.shape[0] > 1:
        raise ValueError('More than one matching sounding_id? aborting...')

    frame_number = frame_number_a[0] + 1
    footprint = footprint_a[0] + 1

    return frame_number, footprint


class wrapped_fp(object):
    """
    create a wrapped full_physics.L2Run object. Contains convenience 
    methods to make this easier to run with retrieval.py

    In general, throughout the class code, when we extract data out of 
    the full_physics.L2Run object, 
    """

    def __init__(self, L1bfile, ECMWFfile, 
                 config_file, 
                 merradir, abscodir,
                 imap_file = None, 
                 sounding_id = None, 
                 frame_number = None, 
                 footprint = None,
                 enable_console_log = True):

        if (frame_number is None or footprint is None) and \
                (sounding_id is None):
            raise ValueError("Either sounding_id, or footprint "+
                             "and frame_number must be specified")

        if sounding_id is None:
            sounding_id = get_sounding_id_from_framefp(
                L1bfile, frame_number, footprint)

        arg_list = (config_file, sounding_id, 
                    ECMWFfile, L1bfile, )
        kw_dict = dict(merradir=merradir, abscodir=abscodir, 
                       imap_file=imap_file)

        # store the input args and kw - mostly as a debug tool later.
        self._arg_list = arg_list
        self._kw_dict = kw_dict
        self._sounding_id = sounding_id

        # create the full physics L2Run object. This is the main 
        # interface to the l2_fp application.
        self._L2Run = full_physics.L2Run(*arg_list, **kw_dict)

        self._enable_console_log = enable_console_log
        if enable_console_log is False:
            self._L2Run.config.logger.turn_off_logger()

        # so - the grid may depend on the prior data (I think).
        # unknown at this point, whether a state vector change would require 
        # the grid to be setup again. For now, assuming no.
        self._L2Run.forward_model.setup_grid()

        # Extract some parameters:
        #

        # 0) the apriori covariance and first guess
        self._state_first_guess = \
            self._L2Run.state_vector.state.copy()
        self._apriori_covariance = \
            self._L2Run.state_vector.state_covariance.copy()

        # 1) channel ranges (start/stop channel numbers) for each band;
        #   (channels outside this range are not used.
        self._channel_ranges = \
            self._L2Run.spectral_window.range_array.value[:,0,:].astype(int)

        # 2) convert the channel ranges to python slice objects (for ease 
        # of use layer on.)
        self._band_slices = []
        for b in range(3):
            start = self._channel_ranges[b,0]
            stop = self._channel_ranges[b,1]
            self._band_slices.append(slice(start, stop))

        # 3) bad sample masks.
        # this essentially merges the bad sample mask with the channel 
        # ranges. The result is a single bool mask that will extract 
        # only those samples actually used in the retrieval.
        self._sample_masks = np.zeros((1016,3), np.bool)
        for b in range(3):
            bad_sample_mask = np.ones(1016, np.bool)
            s = self._band_slices[b]
            bad_sample_mask[s] = \
                self._L2Run.spectral_window.bad_sample_mask[b,s]
            self._sample_masks[:,b] = bad_sample_mask == 0

        # 4) attempt to get the sample indexes from other internal object.
        # turns out that step 3) doesn't really work - the actual sample lists 
        # have another criteria outside of the bad sample masks, it seems. 
        # often step 3) would be one sample off.
        self._sample_indexes = []
        # copy grid_obj to shorten syntax.
        grid_obj = self._L2Run.forward_model.spectral_grid
        self.num_samples = 0
        for b in range(3):
            index_b = grid_obj.low_resolution_grid(b).sample_index.copy()-1
            self.num_samples += index_b.shape[0]
            self._sample_indexes.append(index_b)
            

        output_objs = self._L2Run.config.output()
        self._L2Run_output = output_objs[0]
        self._L2Run_error_output = output_objs[1]

        self._solve_called = False
        self._solve_completed = False

        # TBD - there are other 'derived' values that could be usefully 
        # extracted from the L1b file during setup:
        # retrieval times (string and tai93), operation_mode, 
        # surface_type (?), some contents of L1bSc spectral parameters.
        # mostly these would need to be added if we want the output 
        # in the mocked-up l2 files that are created  - 
        # see write_h5_output_file().


    def _get_L2Run(self):
        """
        internal helper to get the L2Run object. This only exists to 
        allow a more useful Exception string to be printed, if something 
        attempts to use the object after a single-sounding format output 
        file is written, which has to delete the L2Run as a side effect.

        After the __init__() finishes, all references to the L2Run object 
        (even internal methods) should use the L2Run property (which 
        calls this function) rather than referencing the _L2Run attribute.
        """

        if self._L2Run:
            return self._L2Run
        else:
            raise ValueError(
                'L2Run object has been closed, and this operation '+
                'is no longer possible.')

    L2Run = property(_get_L2Run, None, None,
                     'Access full_physics.L2Run object')

    def get_sample_indexes(self, band='all'):
        """
        gets the sample indexes, per band, for this observation.
        band must be equal to : 1,2,3, or 'all', to get all 
        bands concatenated together.

        this is equivalent to the 'sample_indexes' array in the L2Dia 
        SpectralParameters group, or in single sounding input.
        These are 1-ordered indexes (This needs verification! - have not 
        checked that the official L2 output is 1-ordered.)
        
        Returns the sample indexes in a 1D int array with shape (w,)
        """
        if band == 'all':
            sample_indexes = [self.get_sample_indexes(b) for b in range(1,4)]
            sample_index = np.concatenate(sample_indexes)
        else:
            b = band-1
            sample_index = self._sample_indexes[b].copy()
        return sample_index


    def get_noise(self, band='all'):
        """
        gets the sensor noise, per band, for this observation.
        band must be equal to : 1,2,3, or 'all', to get all 
        bands concatenated together.

        Returns the noise at the used spectral samples, in a 1D array 
        with shape (w,)
        """
        if band == 'all':
            noise_per_band = [self.get_noise(b) for b in range(1,4)]
            noise = np.concatenate(noise_per_band)
        else:
            b = band-1
            radiance_all = self.L2Run.level_1b.radiance(b).data.copy()
            noise_all = self.L2Run.level_1b.noise_model.uncertainty(
                b, radiance_all)
            noise = noise_all[self._sample_indexes[b]]

        return noise


    def get_y(self, band='all'):
        """
        gets the observation, per band (e.g. the measured radiance)
        band must be equal to : 1,2,3, or 'all', to get all 
        bands concatenated together.

        Returns the radiance at the used spectral samples, in a 1D  
        array with shape (w,).
        """

        if band == 'all':
            y_per_band = [self.get_y(b) for b in range(1,4)]
            y = np.concatenate(y_per_band)
        else:
            b = band-1
            radiance_all = self.L2Run.level_1b.radiance(b).data.copy()
            y = radiance_all[self._sample_indexes[b]]
        return y


    def get_Se_diag(self, band='all'):
        """
        convenience function to get the get S_e (measurement noise) diagonal - 
        this is just the noise level squared.

        returns a 1D array with shape (w,), for w used samples.
        """

        noise = self.get_noise(band=band)
        Se_diag = noise ** 2

        return Se_diag


    def set_x(self, x_new):
        """
        update the state vector with new values.

        the caller must know the correct layout for the state vector. This must be 
        a 1D numpy ndarray with the correct shape and content.

        see get_state_variable_names, in order to get the variable list.
        """
        x = self.L2Run.state_vector.state.copy()
        S = self.L2Run.state_vector.state_covariance.copy()
        
        if x_new.shape != x.shape:
            raise ValueError('shape mismatch, x_new.shape = ' + 
                             str(x_new.shape) + ' , x.shape = ' + 
                             str(x.shape))
        # I think we need to carry this through a copy of S ?
        self.L2Run.state_vector.update_state(x_new, S)


    def get_state_variable_names(self):
        """
        return the state variable names, in order, as a python tuple 
        of strings.
        """
        svnames = self.L2Run.state_vector.state_vector_name
        return svnames


    def set_Sa(self, S_new):
        """
        update the state covariance with new values.

        the caller must know the correct layout for the state vector (and thus 
        the covariance matrix.) This must be a 2D numpy ndarray with the 
        correct shape and content.

        see get_state_variable_names, in order to get the variable list.
        """
        x = self.L2Run.state_vector.state.copy()
        S = self.L2Run.state_vector.state_covariance.copy()
        if S_new.shape != S.shape:
            raise ValueError('shape mismatch, S_new.shape = ' + 
                             str(S_new.shape) + ' , S.shape = ' + 
                             str(S.shape))
        # I think we need to carry this through a copy of x ?
        self.L2Run.state_vector.update_state(x, S_new)


    def get_x(self):
        """
        get the current state vector

        return is a 1D array with shape (v,) for v state variables.
        """
        return self.L2Run.state_vector.state.copy()


    def get_Sa(self):
        """
        get the current state covariance.

        return is a 2D array with shape (v, v) for v state variables.
        """
        return self.L2Run.state_vector.state_covariance.copy()


    def forward_run(self, band='all'):
        """
        do a forward run, for one band (1,2,3), or all bands (band='all'), 
        using the current configuration and state vector.
        
        returns:
        wl - wavelength at the used spectral samples, in micrometers.
           1D array with shape (w,), for w used samples.
        I - modeled radiance at the used spectral channels, in L1b units
           1D array with shape (w,), for w used samples.
        """
        if self.L2Run is None:
            raise ValueError('L2 object is closed - no further runs possible')
        if band == 'all':
            fm_result = self.L2Run.forward_model.radiance_all(True)
        else:
            b = band-1
            fm_result = self.L2Run.forward_model.radiance(b,True)
        wl = fm_result.wavelength.copy()
        I = fm_result.value.copy()
        return wl, I


    def jacobian_run(self, band='all'):
        """
        do a jacobian run, for one band (1,2,3), or all bands (band='all')

        returns:
        wl - wavelength at the used spectral samples, in micrometers.
           1D array with shape (w,), for w used samples.
        I - modeled radiance at the used spectral channels, in L1b units.
           1D array with shape (w,), for w used samples.
        K - modeled jacobian at the used spectral channels, in L1b radiance 
           1D array with units (same as I) per state vector unit.
           shape (w, v) for w used samples, and v state variables
        """ 
        if self.L2Run is None:
            raise ValueError('L2 object is closed - no further runs possible')
        if band == 'all':
            fm_result = self.L2Run.forward_model.radiance_all(False)
        else:
            b = band-1
            fm_result = self.L2Run.forward_model.radiance(b,False)
        wl = fm_result.wavelength.copy()
        I = fm_result.value.copy()
        K = fm_result.spectral_range.data_ad.jacobian.copy()
        return wl, I, K


    def Kupdate(self, hatx, model_params):
        """
        API - aligned function

        Right now, no model params are used.

        intended to be used inside 'retrieval', where Kupdate will be 
        this method from an instantiated wrapped_fp object.

        K, Fhatx = Kupdate(hatx, model_params)

        """
        self.set_x(hatx)
        wl, I, K = self.jacobian_run()
        return I, K

    def solve(self, x_i=None, x_a=None, cov_a=None):
        """
        Exposes the operational solver. this will run the l2_fp in the 
        same way that it runs operationally, but allows the prior, FG, or 
        covariance to be changed.
        Note if any of the three are used, the stored values are used 
        (FG is the prior)

        return values are:
        solve_result: this will be None if the retrieval crashed; 
            otherwise it will be equal to True/False depending on 
            whether the converge criteria were met.
        hatx: state vector at the final iteration.
        hatS: posterior covariance at the final iteration.

        note, if solve_result is None (meaning a crash), then hatx and 
        hatS are also None.
        """

        if x_i is None:
            x_i = self._state_first_guess
        if x_a is None:
            x_a = self._state_first_guess
        if cov_a is None:
            cov_a = self._apriori_covariance

        # some strange construct here:
        # if an exception occurs, that likely means a forward model 
        # crash, and thus no reliable output. use solve_completed
        # to signify this.
        # whether or not the retrieval converged, is noted by the solve 
        # result. this should be None (if crashed), or True/False.
        # those values all need to be stored, as it affects how the 
        # FPformat write functions are run.
        solve_result = None
        self._solve_called = True
        try:
            solve_result = self._L2Run.solver.solve(x_i, x_a, cov_a)
            self._solve_completed = True
        except:
            self._solve_completed = False

        self._solve_converged = solve_result

        if solve_result is not None:

            # I think, the solver does not update itself with the 
            # state estimate. we need to do that ourselves
            hatx = self._L2Run.solver.x_solution
            hatS = self._L2Run.solver.aposteriori_covariance
            self._L2Run.state_vector.update_state(hatx, hatS)

            # copy these for output
            return_x = hatx.copy()
            return_S = hatS.copy()

        else:
            solve_result = None
            return_x = None
            return_S = None

        return solve_result, return_x, return_S


    def write_h5_output_file_FPformat(self, filename):
        """
        write output file using operational "Full Physics" 
        single-sounding output method. This is the best option, since 
        it includes additional variables related to the solver steps, 
        that would not otherwise be included in the file, since they 
        are not exposed to python
        (at least, I don't know how to get the data)

        Note:
        ** This method only works if the operational solve() is called.
        ** It also must necessarily delete the internal L2Run object 
           to actually write and close the file. This will render most 
           of the methods inoperable.
        """

        if not self._solve_called:
            raise ValueError('solve() method must be called first '+
                             'before FPformat data can be written')

        cwd = os.getcwd()
        if self._solve_completed:
            self._L2Run_output.write()
            output_filename = filename
            written_filename = os.path.join(cwd,'out.h5')
        else:
            self._L2Run_error_output.write_best_attempt()
            output_filename = filename+'.error'
            written_filename = os.path.join(cwd,'out.h5.error')
            written_filename = 'out.h5.error'

        self._close_L2Run()

        # if the the write() fails, or doesn't close, we might have 
        # either no file or a 'generating' file. in either case, 
        # this would cause a mysterious file not found error; catch 
        # the except and re-raise a better error message.
        try:
            shutil.move(written_filename, output_filename)
        except:
            raise RuntimeError(
                'L2_FP object appears to have failed to write or close '+
                'the expected h5 file: '+written_filename)


    def write_h5_output_file(self, filename, 
                             final_state=None, final_uncert=None,
                             modeled_rad=None):
        """
        writes a h5 file, similar to the format for the RetrievedStateVector 
        and SpectralParameters groups of the offline l2-aggregate

        note that the final state and modeled_rad are optionally set via 
        keyword, for cases where instead of the solve() method, the 
        caller uses the afp to find the state estimate.
        """

        if final_state is None:
            if self.L2Run.solver.x_solution.shape[0] > 0:
                final_state = self.L2Run.solver.x_solution.copy()
            else:
                final_state = np.zeros_like(self._state_first_guess)
                final_state[:] = np.nan

        if final_uncert is None:
            if self.L2Run.solver.x_solution.shape[0] > 0:
                final_uncert = self.L2Run.solver.x_solution.copy()
            else:
                final_uncert = np.zeros_like(self._state_first_guess)
                final_uncert[:] = np.nan

        # Note this might have a mismatch with the state vector, since there 
        # are some dispersion related elements. that might shift the wavelength 
        # grid slightly?
        wavelength = []
        for b in range(3):
            tmp = self.L2Run.forward_model.spectral_grid.low_resolution_grid(b)
            wavelength.append(tmp.wavelength())
        wavelength = np.concatenate(wavelength)

        if modeled_rad is None:
            modeled_rad = np.zeros_like(wavelength)
            modeled_rad[:] = np.nan

        dat = {}

        # FG is equal to a priori state
        dat['/RetrievedStateVector/state_vector_apriori'] = \
            self._state_first_guess
        dat['/RetrievedStateVector/state_vector_apriori_uncert'] = \
            np.sqrt(np.diag(self._apriori_covariance))
        dat['/RetrievedStateVector/state_vector_result'] = final_state
        # convert python strings (that are unicode by default) into 
        # ascii strings; h5py can only write the simple ascii-encoding.
        # I think we can rely on numpy's dtype to do the encoding, 
        # instead of python's string.encode()
        svnames = self.get_state_variable_names()
        # not sure which is better. these might be identical.
        #svnames = np.array(svnames, dtype='S')
        svnames = np.array([n.encode('ascii') for n in svnames])
        dat['/RetrievedStateVector/state_vector_names'] = svnames

        sounding_id = np.zeros(1, dtype=np.int64)
        sounding_id[0] = int(self._sounding_id)
        dat['/RetrievalHeader/sounding_id_reference'] = sounding_id
        dat['/RetrievalHeader/sounding_id'] = sounding_id

        dat['/SpectralParameters/wavelength'] = wavelength
        dat['/SpectralParameters/measured_radiance'] = self.get_y()
        dat['/SpectralParameters/measured_radiance_uncert'] = self.get_noise()
        dat['/SpectralParameters/modeled_radiance'] = modeled_rad
        dat['/SpectralParameters/sample_indexes'] = self.get_sample_indexes()

        dat['/RetrievalResults/xco2'] = [self.L2Run.xco2]

        s = np.newaxis, Ellipsis

        split_dat = output_translation.state_vector_splitting(
            self._state_first_guess, final_state, final_uncert, 
            self.get_state_variable_names())

        dat.update(split_dat)

        with h5py.File(filename, 'w') as h:
            for vname in dat:
                if len(dat[vname]) > 1:
                    h.create_dataset(vname, data=dat[vname][s])
                else:
                    h.create_dataset(vname, data=dat[vname])


    def _close_L2Run(self):
        """
        deletes the L2Run object. This is required to get the 
        operational output file to be finalized and closed correctly.
        """

        self._L2Run_output = None
        self._L2Run_error_output = None

        # put this in a try/finally block, just in case something odd happens 
        # on the delete. This will make sure the attribute it set back to 
        # None, so the now-missing or broken L2Run will correctly cause
        # an exception later through _get_L2Run()
        try:
            del self._L2Run
        finally:
            self._L2Run = None


class wrapped_fp_aerosol_reff(wrapped_fp):
    """
    creates a wrapped full_physics.L2Run object, but includes 
    extra code to implement r_eff derivative. This is 
    automatically appended to the state dimension.
    """

    @staticmethod
    def _read_reff_from_static_file(hfile, grp_name):
        # note - assumes the 'Source' variable is a 1-element array of string
        with h5py.File(hfile,'r') as h:
            if grp_name in h:
                sprop_name = h.get(grp_name+'/Properties/Source')[0]
                if not isinstance(sprop_name, str):
                    sprop_name = sprop_name.decode()
                reff = float(sprop_name.split('=')[-1])
            else:
                # Note To Self: fix this part,
                # use negative one as a value to force a recalc - 
                # this should happen if the properties have not been inserted 
                # into file yet.
                reff = -1.0
        return reff

    @staticmethod
    def _prepare_lua_config(infile, outfile, static_input_file,
                            aerosol_variable_defs):

        linefmt1 = 'config.static_aerosol_file = "{0:s}"' + os.linesep
        linefmt2 = (
            'config.fm.atmosphere.aerosol.{0:s}.property = '+
            'ConfigCommon.hdf_aerosol_property("{1:s}")' + os.linesep )

        with open(infile, 'r') as f:
            config_lines = [line for line in f]

        final_line = config_lines.index('config:do_config()'+os.linesep)

        for vardef in aerosol_variable_defs:
            config_lines.insert(
                final_line, 
                linefmt2.format(vardef['lua_name'], vardef['grp_name']))
        config_lines.insert(final_line, linefmt1.format(static_input_file))

        with open(outfile, 'w') as f:
            f.writelines(config_lines)

                
    def __init__(self,
                 base_aerosol_property_file,
                 aerosol_variable_defs,
                 L1bfile, ECMWFfile, config_file, 
                 merradir, abscodir,
                 reff_prior_mean = 10.0, reff_prior_stdv = 4.0, 
                 reff_increment = 0.05, wrkdir = '.',  **kwarg):
        """
        see wrapped_fp init

        aerosol_variable_defs:
        svname: string name for the state variable name list.
        lua_name: 'Water'  (must match name defined in Lua - can't 
             easily get this automatically)
        grp_name: 'wc_variable' name of group in l2_static_aerosol file.
             arbitrary, code will write/create this group on the fly.
        grid_grp_name: 'wc_grid' name of group in the property grid file
        grid_filename: property grid file.
        """

        src_file = base_aerosol_property_file
        dst_file = os.path.join(
            wrkdir, os.path.split(base_aerosol_property_file)[1])
        dst_file = dst_file.replace('.h5', '_variable.h5')
        if os.access(dst_file, os.R_OK):
            raise ValueError(
                'Aborting, aerosol property file already exists in wrkdir:'
                +dst_file)

        shutil.copy(src_file, dst_file)
        self._aerosol_property_file = dst_file
        
        src_file = config_file
        dst_file = os.path.join(wrkdir, os.path.split(config_file)[1])
        if os.access(dst_file, os.R_OK):
            raise ValueError(
                'Aborting, config file already exists in wrkdir:'+dst_file)

        self._prepare_lua_config(
            src_file, dst_file, self._aerosol_property_file,
            aerosol_variable_defs)
        lua_config_file = dst_file

        self._aerosol_vardefs = aerosol_variable_defs
        self._n_aerosol_reff = len(aerosol_variable_defs)

        self._reff = [reff_prior_mean] * self._n_aerosol_reff
        self._reff_var = [reff_prior_stdv**2] * self._n_aerosol_reff
        self._reff_incr = reff_increment
        
        # The L2Run object will initially be set to the same value 
        # as existed in the file, so read the value
        self._staticprops_reff = []
        initialize_props = False
        for n in range(self._n_aerosol_reff):
            vardef = self._aerosol_vardefs[n]
            current_reff = self._read_reff_from_static_file(
                self._aerosol_property_file, vardef['grp_name'])
            if current_reff < 0:
                initialize_props = True
            self._staticprops_reff.append(current_reff)
        if initialize_props:
            self._update_aerosol_props(self._reff)
        self._L2Run_reff = list(self._staticprops_reff)

        # I think this needs to happen first?
        # _update_aerosol_props(reff_prior_mean)

        arg = (L1bfile, ECMWFfile, lua_config_file, merradir, abscodir)

        super(wrapped_fp_aerosol_reff, self).__init__(*arg, **kwarg)

        # most methods will fall to super class, as desired:
        # get_sample_indexes(), get_noise(), get_y(), get_Se_diag()
        #
        # ones involving state variables are intercepted here and 
        # altered: get_x(), set_x(), get_Sa(), set_Sa(), 
        # get_state_variable_names(), jacobian_run(), forward_run(), 
        # write_h5_output_file.
        # Kupdate() is the most important since that 
        # will be used by the retrieval.

        # udpate the state and prior covars to inclue the 
        # prior mean and stdv.
        reff_var_diag = np.zeros(self._n_aerosol_reff) + self._reff_var
        self._Sa_augmented = blk_diag(
            [self._apriori_covariance, np.diag(reff_var_diag)])

        self._state_first_guess = np.concatenate(
            [self._state_first_guess, self._reff])
        self._apriori_covariance = self._Sa_augmented.copy()

        # this also needs a copy stored separately, because it will need 
        # to be restored during a L2Run refresh.
        self._x = self.get_x()


    def _refresh_L2Run(self):
        # this shortcuts super.__init__(), because there are a lot 
        # of calcs related to the sample index, band slices, etc, 
        # that we do not need to redo.
        print('** refreshing L2Run to reff = '+
              '{0:s} **'.format(str(self._reff)))
        self._L2Run = full_physics.L2Run(*self._arg_list, **self._kw_dict)
        self._L2Run.forward_model.setup_grid()
        output_objs = self._L2Run.config.output()
        self._L2Run_output = output_objs[0]
        self._L2Run_error_output = output_objs[1]
        if self._enable_console_log is False:
            self._L2Run.config.logger.turn_off_logger()
        # assumes we now have this reff (should be in synch, but this might 
        # be a potential trouble spot...)
        self._L2Run_reff = list(self._reff)
        # restore the state vector into L2Run. need to do this through super(),
        # for the part of the state vector used in L2Run.
        x_no_reff = self._x[:-self._n_aerosol_reff]
        super(wrapped_fp_aerosol_reff, self).set_x(x_no_reff)
        
        
    def _update_aerosol_props(self, reff):
        # 1) get aerosol properties by interpolation at reff
        # 2) store into l2_static file

        for reff_i, vardef in zip(reff, self._aerosol_vardefs):
            print('interpolating props to reff_i = {0:9.5f} for {1:s}'.format(
                reff_i, vardef['svname']))
            pdata = scattering_properties.interpolate_by_reff(
                vardef['grid_filename'], vardef['grid_grp_name'], reff_i)
            scattering_properties.write_variable_properties(
                self._aerosol_property_file, vardef['grp_name'], pdata)
        self._staticprops_reff = list(reff)


    def get_x(self):
        x0 = super(wrapped_fp_aerosol_reff, self).get_x()
        x0 = np.concatenate([x0, self._reff])
        return x0

    def set_x(self, x_new):
        x_no_reff = self._x[:-self._n_aerosol_reff]
        super(wrapped_fp_aerosol_reff, self).set_x(x_no_reff)
        self._reff = list(x_new[-self._n_aerosol_reff:])
        self._x = x_new.copy()

    def get_Sa(self):
        reff_var_diag = np.zeros(self._n_aerosol_reff) + self._reff_var
        Sa = super(wrapped_fp_aerosol_reff, self).get_Sa()
        Sa = blk_diag([Sa, np.diag(reff_var_diag)])
        return Sa

    def set_Sa(self, S_new):
        super(wrapped_fp_aerosol_reff, self).set_Sa(
            S_new[:-self._n_aerosol_reff,:-self._n_aerosol_reff])
        self._reff_var = list(np.diag(
            S_new[-self._n_aerosol_reff:,-self._n_aerosol_reff:]))
        self._Sa_augmented = S_new.copy()

    def get_state_variable_names(self):
        svnames = \
            super(wrapped_fp_aerosol_reff, self).get_state_variable_names()
        svnames = list(svnames)
        svnames += [v['svname'] for v in self._aerosol_vardefs]
        return svnames

    def _ensure_synched_reff(self):
        if self._staticprops_reff != self._reff:
            self._update_aerosol_props(self._reff)
        if self._L2Run_reff != self._reff:
            self._refresh_L2Run()

    def forward_run(self, band='all'):
        # if the reff has changed, need to refresh the static data and L2Run.
        self._ensure_synched_reff()
        return super(wrapped_fp_aerosol_reff, self).forward_run(band=band)

    
    def jacobian_run(self, band='all'):
        # if the reff has changed, need to refresh the static data and L2Run.
        self._ensure_synched_reff()
        wl,I,K = super(wrapped_fp_aerosol_reff, self).jacobian_run(band=band)

        K_reff = self._reff_derivative(band=band)
        K_aug = np.concatenate([K, K_reff], axis=1)

        return wl, I, K_aug


    def _reff_derivative(self, band='all'):

        # store base value, before we perturb it.
        base_reff = list(self._reff)

        # since the call can be 1 or all 3 bands, we don't know 
        # the needed shape for dIdR until the end. hence the use of list()
        dIdR_list = []

        for n in range(self._n_aerosol_reff):
            # perturbed values.
            reff_d1 = base_reff[n] * (1 - 0.5*self._reff_incr)
            reff_d2 = base_reff[n] * (1 + 0.5*self._reff_incr)

            self._reff[n] = reff_d1
            wl, I1 = self.forward_run(band=band)
            self._reff[n] = reff_d2
            wl, I2 = self.forward_run(band=band)

            dIdR_list.append( (I2 - I1) / (self._reff_incr*base_reff[n]) )

            # restore original value
            self._reff[n] = base_reff[n]

        dIdR = np.array(dIdR_list).T

        return dIdR


    def write_h5_output_file_test(self):
        # I think this needs update, because the state has changed
        raise NotImplementedError()
