# I don't think we use the _swig module directly, but it needs to be 
# importable for the full_physics to function properly.
import full_physics_swig
import full_physics
import h5py

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


class wrapped_l2_fp(object):
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
            self._L2run.spectral_window.range_array.value[:,0,:].copy()

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
            sample_indexes = [self._sample_indexes]
            sample_index = np.concatenate(sample_index)
        else:
            b = band-1
            sample_index = np._sample_indexes[b].copy()
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
        if band is None:
            fm_result = self._L2run.forward_model.radiance_all(True)
        else:
            b = band-1
            fm_result = self._L2run.forward_model.radiance(b,True)
        wl = fm_result.wavelength.copy()
        I = fm_result.value.copy()
        return wl, I


    def jacobian_run(self, band=None):
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
        if band is None:
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
        this method from an instantiated wrapped_l2_fp object.

        K, Fhatx = Kupdate(hatx, model_params)

        """
        self.set_x(hatx)
        wl, I, K = self.jacobian_run()
        return I, K


    def close_obj(self):
        """
        This attempts to "close" the object - I am not sure this really works - 
        but this is how the l2_special_run.py tries to do it.
        """
        self._L2run = None
