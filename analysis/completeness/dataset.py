from os.path import join
import numpy as np

from .config import path0, filters


def get_mact_data(ff):
    # Read in EW and fluxes for H-alpha NB emitter sample
    npz_NB_file = join(path0, 'Completeness/ew_flux_Ha-' + filters[ff] + '.npz')
    npz_NB = np.load(npz_NB_file)

    dict_NB = dict(npz_NB)

    for r_item in ['NB_Flux', 'NB_ID', 'Ha_EW']:
        dict_NB.pop(r_item)

    spec_flag = npz_NB['spec_flag']
    dict_NB['w_spec'] = np.where(spec_flag)[0]
    dict_NB['wo_spec'] = np.where(spec_flag == 0)[0]

    return dict_NB
