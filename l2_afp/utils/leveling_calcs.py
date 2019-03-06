import numpy as np

def plevel_covariance_matrix(Plevel, corr_scale, var):
    """
    compute a synthetic covariance with exp(-|p[j] - p[k]|/corr_scale)
    correlations.
    Uses Plevel array to determine the strength of the off diagonal
    correlation numbers.
    """

    n_levels = Plevel.shape[0]

    var_diag = np.asarray(var)
    if var_diag.ndim == 0:
        var_diag = np.zeros(n_levels)
        var_diag[:] = var
    elif var_diag.shape[0] == 1:
        var_diag = np.zeros(n_levels)
        var_diag[:] = var
    else:
        if var_diag.shape[0] != n_levels:
            raise ValueError('variance and Plevel have different shape')

    R = np.eye(n_levels)

    # simple inefficient loop, but this is easy
    for j,k in np.ndindex((n_levels, n_levels)):
        scaled_diff = np.abs((Plevel[j]-Plevel[k])/corr_scale)
        R[j,k] = np.exp(-scaled_diff)

    std_diag = np.sqrt(var_diag)
    C = R * np.outer(std_diag, std_diag)

    return C, R


class pleveler(object):
    """
    pleveler: automate Cloud-Center Pressure (CCP) and Cloud pressure 
    Thickness (CT) calculations, for a set of cloud levels  inserted
    in a set of fixed pressure levels for the clear atmosphere
    above and below the cloud.
    """

    @classmethod
    def _merge_plevels(cls, clear_plevels, cloud_plevels, npad):

        ctop = cloud_plevels[0]
        cbas = cloud_plevels[-1]

        if ctop <= clear_plevels[0]:
            raise ValueError(
                'Cloud Top {0:f} is above clear level grid {1:f}'.format(
                    ctop, clear_plevels[0]))
        if cbas >= clear_plevels[-1]:
            raise ValueError(
                'Cloud Base {0:f} is below clear level grid {1:f}'.format(
                    cbas, clear_plevels[-1]))

        ktop, kbot = clear_plevels.searchsorted([ctop, cbas])

        plevel_blocks = collections.OrderedDict()
        plevel_blocks['clear_upper'] = clear_plevels[:ktop-npad]
        plevel_blocks['pad_upper'] = clear_plevels[ktop-npad:ktop]
        plevel_blocks['cloud'] = cloud_plevels
        
        if cbas == clear_plevels[kbot]:
            kbot += 1
        if (kbot + npad) < clear_plevels.shape:
            plevel_blocks['pad_lower'] = clear_plevels[kbot:kbot+npad]
            plevel_blocks['clear_lower'] = clear_plevels[kbot+npad:]
        else:
            # here, we need to adjust to make the clear_lower contain
            # at least the surface level
            plevel_blocks['pad_lower'] = clear_plevels[kbot:-1]
            plevel_blocks['clear_lower'] = clear_plevels[-1:]
            # if the cloud is low enough that there were no plevels 
            # between cbase and the lowest plevel (surface), add one
            # level in between.
            if len(plevel_blocks['pad_lower']) == 0:
                pad_plevel = 0.5 * (cbas + clear_plevels[-1])
                plevel_blocks['pad_lower'] = np.array([pad_plevel])

        return plevel_blocks


    def __init__(self, clear_plevels, cloud_plevels, n_padding):

        # ensure sorted
        self._cloud_plevels_input = np.sort(cloud_plevels)
        self._clear_plevels_input = np.sort(clear_plevels)

        # merge & block
        self._plevel_blocks = self._merge_plevels(
            self._clear_plevels_input, self._cloud_plevels_input, n_padding)

        # derive CCP/CT
        self._update_Cparams()

        self._update_padding_levels()

    def _update_padding_levels(self):
        pad_upper = np.linspace(
            self._plevel_blocks['clear_upper'][-1], 
            self._plevel_blocks['cloud'][0],
            self._plevel_blocks['pad_upper'].shape[0]+2)
        pad_lower = np.linspace(
            self._plevel_blocks['cloud'][-1], 
            self._plevel_blocks['clear_lower'][0],
            self._plevel_blocks['pad_lower'].shape[0]+2)
        self._plevel_blocks['pad_upper'][:] = pad_upper[1:-1]
        self._plevel_blocks['pad_lower'][:] = pad_lower[1:-1]

    def _update_Cparams(self):
        cloud_plevels = self._plevel_blocks['cloud']
        self._CTP = cloud_plevels[0]
        self._CBP = cloud_plevels[-1]
        self._CCP = 0.5 * (self._CBP + self._CTP)
        self._CT = self._CBP - self._CTP

    def get_plevels(self):
        plevel_blocks = list(self._plevel_blocks.values())
        return np.concatenate(plevel_blocks)

    def get_level_types(self):
        block_types = []
        for n,block in enumerate(self._plevel_blocks.values()):
            block_types.append(np.zeros(block.shape[0], int) + n)
        return np.concatenate(block_types)

    def update_CCP(self, CCP_delta):
        # move cloud by delta
        new_cloud_levels = self._plevel_blocks['cloud'] + CCP_delta
        # make sure not OOR (e.g. the update doesn't collapse the pad layers
        # to nothing, or negatives.)
        if new_cloud_levels[0] <= self._plevel_blocks['clear_upper'][-1]:
            raise ValueError(
                'Invalid CCP update, cloud top moves into clear upper section')
        if new_cloud_levels[-1] >= self._plevel_blocks['clear_lower'][0]:
            raise ValueError(
                'Invalid CCP update, cloud base moves into clear lower section')
        self._plevel_blocks['cloud'][:] = new_cloud_levels

        self._update_Cparams()
        self._update_padding_levels()

    def update_CT(self, CT_delta):
        # change cloud thickess by delta.
        scaling = CT_delta / self._CT + 1
        relative_cloud_levels = self._plevel_blocks['cloud'] - self._CCP
        new_cloud_levels = scaling * relative_cloud_levels + self._CCP
        # make sure not OOR (e.g. the update doesn't collapse the pad layers
        # to nothing, or negatives.)
        if new_cloud_levels[0] <= self._plevel_blocks['clear_upper'][-1]:
            raise ValueError(
                'Invalid CT update, cloud top moves into clear upper section')
        if new_cloud_levels[-1] >= self._plevel_blocks['clear_lower'][0]:
            raise ValueError(
                'Invalid CT update, cloud base moves into clear lower section')
        self._plevel_blocks['cloud'][:] = new_cloud_levels

        self._update_Cparams()
        self._update_padding_levels()

    def _get_CCP(self):
        return self._CCP
    def _get_CT(self):
        return self._CT
    def _get_CTP(self):
        return self._CTP
    def _get_CBP(self):
        return self._CBP

    CCP = property(_get_CCP)
    CT = property(_get_CT)
    CTP = property(_get_CTP)
    CBP = property(_get_CBP)

    # mainly for debugging: hides the MPL import here
    def plot(self, ax=None, fignum=10, X=None, XP=None, Xlabel='',
             legend=False):
        """
        plot the Pleveler object data in a convenient vertical grid.
        This function will import matplotlib.
        """
        import matplotlib.pyplot as plt

        P = self.get_plevels()

        if X is None:
            X_list = [np.zeros_like(P_b)
                      for P_b in self._plevel_blocks.values()]
        else:
            X_list = [np.interp(P_b, XP, X)
                      for P_b in self._plevel_blocks.values()]
        X = np.concatenate(X_list)

        if ax is None:
            fig = plt.figure(fignum, figsize=(5,8))
            fig.clf()
            ax = fig.add_subplot(111)

        ax.plot(X, P, '-', label='_nolabel_')

        for n, (k, v) in enumerate(self._plevel_blocks.items()):
            ax.plot(X_list[n], v, '_', label=k, ms=12, mew=2)


        ax.set_ylim(1050, -50)
        ax.set_yticks(np.arange(0,1001,50))
        #ax.set_yticks(P)
        ax.grid(1)
        ax.set_ylabel('Pressure [hPa]')
        ax.set_xlabel(Xlabel)
        ax.tick_params(labelsize=9)
        if legend:
            ax.legend(loc='best')

        return ax
