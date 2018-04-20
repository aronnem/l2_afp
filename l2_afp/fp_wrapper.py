# I don't think we use the _swig module directly, but it needs to be 
# importable for the full_physics to function properly.
import full_physics_swig
import full_physics
import h5py

from .utils import output_translation

import numpy as np

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
    create a wrapped full_physics.L2run object. Contains convenience 
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
                 footprint = None):

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

        # create the full physics L2Eun object. This is the main 
        # interface to the l2_fp application.
        self._L2run = full_physics.L2Run(*arg_list, **kw_dict)

        # so - the grid may depend on the prior data (I think).
        # unknown at this point, whether a state vector change would require 
        # the grid to be setup again. For now, assuming no.
        self._L2run.forward_model.setup_grid()

        # Extract some parameters:
        #

        # 0) the apriori covariance and first guess
        self._state_first_guess = \
            self._L2run.state_vector.state.copy()
        self._apriori_covariance = \
            self._L2run.state_vector.state_covariance.copy()

        # 1) channel ranges (start/stop channel numbers) for each band;
        #   (channels outside this range are not used.
        self._channel_ranges = \
            self._L2run.spectral_window.range_array.value[:,0,:].astype(int)

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
                self._L2run.spectral_window.bad_sample_mask[b,s]
            self._sample_masks[:,b] = bad_sample_mask == 0

        # 4) attempt to get the sample indexes from other internal object.
        # turns out that step 3) doesn't really work - the actual sample lists 
        # have another criteria outside of the bad sample masks, it seems. 
        # often step 3) would be one sample off.
        self._sample_indexes = []
        # copy grid_obj to shorten syntax.
        grid_obj = self._L2run.forward_model.spectral_grid
        for b in range(3):
            self._sample_indexes.append( 
                grid_obj.low_resolution_grid(b).sample_index.copy()-1 )

        # TBD - there are other 'derived' values that could be usefully 
        # extracted from the L1b file during setup:
        # retrieval times (string and tai93), operation_mode, 
        # surface_type (?), some contents of L1bSc spectral parameters.
        # mostly these would need to be added if we want the output 
        # in the mocked-up l2 files that are created  - 
        # see write_h5_output_file().


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
            radiance_all = self._L2run.level_1b.radiance(b).data.copy()
            noise_all = self._L2run.level_1b.noise_model.uncertainty(
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
            radiance_all = self._L2run.level_1b.radiance(b).data.copy()
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
        x = self._L2run.state_vector.state.copy()
        S = self._L2run.state_vector.state_covariance.copy()
        
        if x_new.shape != x.shape:
            raise ValueError('shape mismatch, x_new.shape = ' + 
                             str(x_new.shape) + ' , x.shape = ' + 
                             str(x.shape))
        # I think we need to carry this through a copy of S ?
        self._L2run.state_vector.update_state(x_new, S)


    def get_state_variable_names(self):
        """
        return the state variable names, in order, as a python tuple 
        of strings.
        """
        svnames = self._L2run.state_vector.state_vector_name
        return svnames


    def set_Sa(self, S_new):
        """
        update the state covariance with new values.

        the caller must know the correct layout for the state vector (and thus 
        the covariance matrix.) This must be a 2D numpy ndarray with the 
        correct shape and content.

        see get_state_variable_names, in order to get the variable list.
        """
        x = self._L2run.state_vector.state.copy()
        S = self._L2run.state_vector.state_covariance.copy()
        if S_new.shape != S.shape:
            raise ValueError('shape mismatch, S_new.shape = ' + 
                             str(S_new.shape) + ' , S.shape = ' + 
                             str(S.shape))
        # I think we need to carry this through a copy of x ?
        self._L2run.state_vector.update_state(x, S_new)


    def get_x(self):
        """
        get the current state vector

        return is a 1D array with shape (v,) for v state variables.
        """
        return self._L2run.state_vector.state.copy()


    def get_Sa(self):
        """
        get the current state covariance.

        return is a 2D array with shape (v, v) for v state variables.
        """
        return self._L2run.state_vector.state_covariance.copy()


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
        if self._L2run is None:
            raise ValueError('L2 object is closed - no further runs possible')
        if band == 'all':
            fm_result = self._L2run.forward_model.radiance_all(True)
        else:
            b = band-1
            fm_result = self._L2run.forward_model.radiance(b,True)
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
        if self._L2run is None:
            raise ValueError('L2 object is closed - no further runs possible')
        if band == 'all':
            fm_result = self._L2run.forward_model.radiance_all(False)
        else:
            b = band-1
            fm_result = self._L2run.forward_model.radiance(b,False)
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
        """

        if x_i is None:
            x_i = self._state_first_guess
        if x_a is None:
            x_a = self._state_first_guess
        if cov_a is None:
            cov_a = self._apriori_covariance

        self._L2run.solver.solve(x_i, x_a, cov_a)

        hatx = self._L2run.solver.x_solution.copy()
        

        return 


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
            if self._L2run.solver.x_solution.shape[0] > 0:
                final_state = self._L2run.solver.x_solution.copy()
            else:
                final_state = np.zeros_like(self._state_first_guess)
                final_state[:] = np.nan

        if final_uncert is None:
            if self._L2run.solver.x_solution.shape[0] > 0:
                final_uncert = self._L2run.solver.x_solution.copy()
            else:
                final_uncert = np.zeros_like(self._state_first_guess)
                final_uncert[:] = np.nan

        # Note this might have a mismatch with the state vector, since there 
        # are some dispersion related elements. that might shift the wavelength 
        # grid slightly?
        wavelength = []
        for b in range(3):
            tmp = self._L2run.forward_model.spectral_grid.low_resolution_grid(b)
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
        dat['/RetrievedStateVector/state_vector_names'] = \
            np.array(self.get_state_variable_names())

        sounding_id = np.zeros(1, dtype=np.int64)
        sounding_id[0] = int(self._sounding_id)
        dat['/RetrievalHeader/sounding_id_reference'] = sounding_id
        dat['/RetrievalHeader/sounding_id'] = sounding_id

        dat['/SpectralParameters/wavelength'] = wavelength
        dat['/SpectralParameters/measured_radiance'] = self.get_y()
        dat['/SpectralParameters/measured_radiance_uncert'] = self.get_noise()
        dat['/SpectralParameters/modeled_radiance'] = modeled_rad
        dat['/SpectralParameters/sample_indexes'] = self.get_sample_indexes()

        dat['/RetrievalResults/xco2'] = [self._L2run.xco2]

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


    def close_obj(self):
        """
        This attempts to "close" the object - I am not sure this really works - 
        but this is how the l2_special_run.py tries to do it.
        """
        self._L2run = None
