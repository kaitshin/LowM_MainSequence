import numpy as np

from .config import m_NB, cont_lim, minthres
from .properties import compute_EW

from scipy.interpolate import interp1d

m_AB = 48.6


def get_sigma(x, lim1, sigma=3.0):
    """
    Purpose:
      Magnitude errors based on limiting magnitude

    :param x: numpy array of magnitudes
    :param lim1: 3-sigma limiting magnitude for corresponding x (float)
    :param sigma: Sigma threshold (float).  Default: 3.0

    :return dmag: numpy array of magnitude errors
    """

    SNR = sigma * 10 ** (-0.4 * (x - lim1))
    dmag = 2.5 * np.log10(1 + 1 / SNR)

    return dmag


def color_cut(x, lim1, lim2, mean=0.0, sigma=3.0):
    """
    Purpose:
      NB excess color selection based on limiting magnitudes

    :param x: numpy array of NB magnitudes
    :param lim1: 3-sigma NB limiting magnitude (float)
    :param lim2: 3-sigma BB limiting magnitude (float)
    :param mean: mean of excess (float). Default: 0
    :param sigma: Sigma threshold (float).  Default: 3.0

    :return val: numpy array of 3-sigma allowed BB-NB excess color
    """

    f1 = (sigma / 3.0) * 10 ** (-0.4 * (m_AB + lim1))
    f2 = (sigma / 3.0) * 10 ** (-0.4 * (m_AB + lim2))

    f = 10 ** (-0.4 * (m_AB + x))

    val = mean - 2.5 * np.log10(1 - np.sqrt(f1 ** 2 + f2 ** 2) / f)

    return val


def NB_select(ff, NB_mag, x_mag):
    """
    Purpose:
      NB excess color selection

    :param ff: integer for filter
    :param NB_mag: numpy array of NB magnitudes
    :param x_mag: numpy array of NB excess colors, continuum - NB

    :return NB_sel: numpy index for NB excess selection
    :return NB_nosel: numpy index for non NB excess selection
    :return sig_limit: numpy array providing 3-sig limit for NB_mag input
    """

    sig_limit = color_cut(NB_mag, m_NB[ff], cont_lim[ff])

    NB_sel = np.where((x_mag >= minthres[ff]) & (x_mag >= sig_limit))
    NB_nosel = np.where((x_mag < minthres[ff]) | (x_mag < sig_limit))

    return NB_sel, NB_nosel, sig_limit


def get_EW(ff, mylog):
    """
    Purpose:
      Retrieve an interpolated grid for EW to NB excess mapping

    :param ff: integer input for filter
    :param mylog: logger class

    :return EW_int: scipy.interp1d object
    """
    x = np.arange(0.01, 10.00, 0.01)
    EW_ref = compute_EW(x, ff)

    good = np.where(np.isfinite(EW_ref))[0]
    mylog.info('EW_ref (min/max): %f %f ' % (min(EW_ref[good]),
                                             max(EW_ref[good])))
    EW_int = interp1d(EW_ref[good], x[good], bounds_error=False,
                      fill_value=(-3.0, np.max(EW_ref[good])))

    return EW_int
