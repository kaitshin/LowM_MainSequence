from os.path import join
import numpy as np

from scipy.interpolate import interp1d

from . import filter_dict  # dBB, dNB
from ...mainseq_corrections import niiha_oh_determine
from ..NB_errors import ew_flux_dual
from .config import path0


def get_mag_vs_mass_interp(prefix_ff):
    """
    Purpose:
      Define interpolation function between continuum magnitude and stellar mass

    :param prefix_ff: filter prefix (str)
      Either 'Ha-NB7', 'Ha-NB816', 'Ha-NB921', or 'Ha-NB973'

    :return mass_int: interp1d object for logarithm of stellar mass, logM
    :return std_mass_int: interp1d object for dispersion in logM
    """

    npz_mass_file = join(path0, 'Completeness/mag_vs_mass_' + prefix_ff + '.npz')
    npz_mass = np.load(npz_mass_file, allow_pickle=True)
    cont_arr = npz_mass['cont_arr']
    dmag = cont_arr[1] - cont_arr[0]
    mgood = np.where(npz_mass['N_logM'] != 0)[0]

    x_temp = cont_arr + dmag / 2.0
    mass_int = interp1d(x_temp[mgood], npz_mass['avg_logM'][mgood],
                        bounds_error=False, fill_value='extrapolate',
                        kind='linear')

    m_bad = np.where(npz_mass['N_logM'] <= 1)[0]
    std0 = npz_mass['std_logM']
    if len(m_bad) > 0:
        std0[m_bad] = 0.30

    std_mass_int = interp1d(x_temp, std0, fill_value=0.3, bounds_error=False,
                            kind='nearest')
    return mass_int, std_mass_int


def compute_EW(x0, ff):
    y_temp = 10 ** (-0.4 * x0)
    EW_ref = np.log10(filter_dict['dNB'][ff] * (1 - y_temp) / (y_temp - filter_dict['dNB'][ff] / filter_dict['dBB'][ff]))
    return EW_ref


def correct_NII(log_flux, NIIHa):
    """
    Purpose:
      Provide H-alpha fluxes from F_NB using NII/Ha flux ratios for
      correction

    :param log_flux: array containing logarithm of flux
    :param NIIHa: array containing NII/Ha flux ratios
    """

    return log_flux - np.log10(1 + NIIHa)


def get_NIIHa_logOH(logM):
    """
    Purpose:
      Get [NII]6548,6583/H-alpha flux ratios and oxygen abundance based on
      stellar mass.  Metallicity is from PP04

    :param logM: numpy array of logarithm of stellar mass

    :return NIIHa: array of NII/Ha flux ratios
    :return logOH: log(O/H) abundances from PP04 formula
    """

    NIIHa = np.zeros(logM.shape)

    low_mass = np.where(logM <= 8.0)
    if len(low_mass[0]) > 0:
        NIIHa[low_mass] = 0.0624396766589

    high_mass = np.where(logM > 8.0)
    if len(high_mass[0]) > 0:
        NIIHa[high_mass] = 0.169429547993 * logM[high_mass] - 1.29299670728

    # Compute metallicity
    NII6583_Ha = NIIHa * 1 / (1 + 1 / 2.96)

    NII6583_Ha_resize = np.reshape(NII6583_Ha, NII6583_Ha.size)
    logOH = niiha_oh_determine(np.log10(NII6583_Ha_resize), 'PP04_N2') - 12.0
    logOH = np.reshape(logOH, logM.shape)

    return NIIHa, logOH


def HaSFR_metal_dep(logOH, orig_lum):
    """
    Purpose:
      Determine H-alpha SFR using metallicity and luminosity to follow
      Ly+ 2016 metallicity-dependent SFR conversion

    :param logOH: log(O/H) abundance
    :param orig_lum: logarithm of H-alpha luminosity
    """

    y = logOH + 3.31

    # metallicity-dependent SFR conversion
    log_SFR_LHa = -41.34 + 0.39 * y + 0.127 * y ** 2

    log_SFR = log_SFR_LHa + orig_lum

    return log_SFR


def dict_phot_maker(NB, BB, x, filt_dict, filt_corr, mass_int, lum_dist):
    dict_phot = {'NB': NB, 'BB': BB, 'x': x, 'filt_dict': filt_dict,
                 'filt_corr': filt_corr, 'mass_int': mass_int,
                 'lum_dist': lum_dist}
    return dict_phot


def derived_properties(NB, BB, x, filt_dict, filt_corr, mass_int, lum_dist,
                       std_mass_int=None, suffix=''):
    EW, NB_flux = ew_flux_dual(NB, BB, x, filt_dict)

    der_prop_dict = dict()
    der_prop_dict['logEW'+suffix] = np.log10(EW)

    # Apply NB filter correction from beginning
    der_prop_dict['NB_flux'+suffix] = np.log10(NB_flux * filt_corr)

    logM = mass_int(BB)
    if not isinstance(std_mass_int, type(None)):
        std_ref = std_mass_int(BB)
        np.random.seed(348)
        rtemp = np.random.normal(size=NB.shape)
        logM += std_ref * rtemp
    der_prop_dict['logM'+suffix] = logM

    NIIHa, logOH = get_NIIHa_logOH(logM)
    der_prop_dict['NIIHa'+suffix] = NIIHa
    der_prop_dict['logOH'+suffix] = logOH

    der_prop_dict['Ha_Flux'+suffix] = correct_NII(der_prop_dict['NB_flux'+suffix], NIIHa)
    der_prop_dict['Ha_Lum'+suffix] = der_prop_dict['Ha_Flux'+suffix] + \
                                     np.log10(4 * np.pi) + 2 * np.log10(lum_dist)

    der_prop_dict['logSFR'+suffix] = HaSFR_metal_dep(der_prop_dict['logOH'+suffix],
                                                     der_prop_dict['Ha_Lum'+suffix])

    return der_prop_dict
