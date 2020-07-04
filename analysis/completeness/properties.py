import numpy as np

from . import dBB, dNB
from ...mainseq_corrections import niiha_oh_determine
from ..NB_errors import ew_flux_dual


def compute_EW(x0, ff):
    y_temp = 10 ** (-0.4 * x0)
    EW_ref = np.log10(dNB[ff] * (1 - y_temp) / (y_temp - dNB[ff] / dBB[ff]))
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


def dict_prop_maker(NB, BB, x, filt_dict, filt_corr, mass_int, lum_dist):
    dict_prop = {'NB': NB, 'BB': BB, 'x': x, 'filt_dict': filt_dict,
                 'filt_corr': filt_corr, 'mass_int': mass_int,
                 'lum_dist': lum_dist}
    return dict_prop


def derived_properties(NB, BB, x, filt_dict, filt_corr, mass_int, lum_dist,
                       std_mass_int=None):
    EW, NB_flux = ew_flux_dual(NB, BB, x, filt_dict)

    # Apply NB filter correction from beginning
    NB_flux = np.log10(NB_flux * filt_corr)

    logM = mass_int(BB)
    if not isinstance(std_mass_int, type(None)):
        std_ref = std_mass_int(BB)
        np.random.seed(348)
        rtemp = np.random.normal(size=NB.shape)
        logM += std_ref * rtemp

    NIIHa, logOH = get_NIIHa_logOH(logM)

    Ha_Flux = correct_NII(NB_flux, NIIHa)
    Ha_Lum = Ha_Flux + np.log10(4 * np.pi) + 2 * np.log10(lum_dist)

    logSFR = HaSFR_metal_dep(logOH, Ha_Lum)

    return np.log10(EW), NB_flux, logM, NIIHa, logOH, Ha_Flux, Ha_Lum, logSFR
