# I don't think we use the _swig module directly, but it needs to be 
# importable for the full_physics to function properly.
import full_physics_swig
import full_physics
import h5py

import numpy as np

def _get_sounding_id_from_framefp(L1bfile, frame_number, footprint):
    varpath = '/SoundingGeometry/sounding_id'
    with h5py.File(L1bfile) as hfile:
        sounding_id = hfile.get(varpath)[frame_number-1, footprint-1]
    sounding_id = str(sounding_id)


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
                 sounding_id = None, 
                 frame_number = None, 
                 footprint = None):

        if (frame_number is None or footprint is None) and \
                (sounding_id is None):
            raise ValueError("Either sounding_id, or footprint "+
                             "and frame_number must be specified")

        arg_list = (config_file, sounding_id, 
                    ECMWFfile, L1bfile, )
        kw_dict = dict(merradir=merradir, abscodir=abscodir)

        # store the input args and kw - mostly as a debug tool later.
        self._arg_list = arg_list
        self._kw_dict = kw_dict

        # create the full physics L2Eun object. This is the main 
        # interface to the l2_fp application.
        self._L2run = full_physics.L2Run(*arg_list, **kw_dict)

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
            # hack, for unknown reason, the last O2AB sample is not included
            #if b == 0:
            #    stop -= 1
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
        # turns out that 3) doesn't really work - the actual sample lists 
        # have another criteria outside of the bad sample masks.
        self._sample_indexes = []
        # copy grid_obj to shorten syntax.
        grid_obj = self._L2run.forward_model.spectral_grid
        for b in range(3):
            self._sample_indexes.append( 
                grid_obj.low_resolution_grid(b).sample_index.copy() )


    def get_noise(self, band=None):
        """
        gets the sensor noise, per band, for this observation.
        band must be equal to : 1,2,3, or None, to get all 
        bands concatenated together.
        Returns the noise at only the used samples.
        """
        if band is None:
            noise_per_band = [self.get_noise(b) for b in range(1,4)]
            noise = np.concatenate(noise_per_band)
        else:
            b = band-1
            radiance_all = self._L2run.level_1b.radiance(b).data.copy()
            noise_all = self._L2run.level_1b.noise_model.uncertainty(
                b, radiance_all)
            noise = noise_all[self._sample_indexes[b]]

        return noise


    def get_y(self, band=None):

        if band is None:
            y_per_band = [self.get_y(b) for b in range(1,4)]
            y = np.concatenate(y_per_band)
        else:
            b = band-1
            radiance_all = self._L2run.level_1b.radiance(b).data.copy()
            y = radiance_all[self._sample_indexes[b]]
        return y


    def get_Se_diag(self, band=None):

        noise = self.get_noise(band=band)
        Se_diag = noise ** 2

        return Se_diag


    def set_x(self, x_new):
        """
        update the state vector with new values.
        """
        x = self._L2run.state_vector.state.copy()
        S = self._L2run.state_vector.state_covariance.copy()
        if x_new.shape != x.shape:
            raise ValueError('shape mismatch, x_new.shape = ' + 
                             str(x_new.shape) + ' , x.shape = ' + 
                             str(x.shape))
        # I think we need to carry this through a copy of S ?
        self._L2run.state_vector.update_state(x_new, S)


    def set_Sa(self, S_new):
        """
        update the state covariance with new values.
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
        """
        return self._L2run.state_vector.state.copy()


    def get_Sa(self):
        """
        get the current state covariance
        """
        return self._L2run.state_vector.state_covariance.copy()


    def forward_run(self, band=None):
        if band is None:
            fm_result = self._L2run.forward_model.radiance_all(True)
        else:
            b = band-1
            fm_result = self._L2run.forward_model.radiance(b,True)
        wl = fm_result.wavelength.copy()
        I = fm_result.value.copy()
        return wl, I


    def jacobian_run(self, band=None):
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
        self._L2run = None
