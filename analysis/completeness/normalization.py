from os.path import exists, join
from scipy.interpolate import interp1d
import numpy as np

from .config import path0, npz_path0, filters
from .config import npz_NBnames
from .select import get_sigma
from .config import m_NB

# Number density for normalization
npz_slope = np.load(join(path0, 'Completeness/NB_numbers.npz'), allow_pickle=True)


def get_normalization(ff, Nmock, NB, Nsim, NB_bin, mylog, redo=False):

    npz_NBfile = join(npz_path0, filters[ff] + '_init.npz')

    norm_dict = dict()

    if not exists(npz_NBfile) or redo:
        N_mag_mock = npz_slope['N_norm0'][ff] * Nsim * NB_bin
        N_interp = interp1d(npz_slope['mag_arr'][ff], N_mag_mock)
        Ndist_mock = np.int_(np.round(N_interp(NB)))
        NB_ref = np.repeat(NB, Ndist_mock)

        Ngal = NB_ref.size  # Number of galaxies, populate with eval

        NB_sig = get_sigma(NB, m_NB[ff], sigma=3.0)
        NB_sig_ref = np.repeat(NB_sig, Ndist_mock)  # populate with eval

        for name in npz_NBnames:
            norm_dict[name] = eval(name)

        if exists(npz_NBfile):
            mylog.info("Overwriting : " + npz_NBfile)
        else:
            mylog.info("Writing : " + npz_NBfile)
        np.savez(npz_NBfile, **norm_dict)
    else:
        if not redo:
            mylog.info("File found : " + npz_NBfile)
            npz_NB = np.load(npz_NBfile)

            # for key0 in npz_NB.keys():
            #    cmd1 = key0 + " = npz_NB['" + key0 + "']"
            #    exec (cmd1)

            norm_dict = dict(npz_NB)

    return norm_dict
