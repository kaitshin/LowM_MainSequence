from os.path import exists

import numpy as np

from .stats import stats_log
from .config import filt_corr, cont_lim
from .select import NB_select, get_sigma
from .properties import dict_prop_maker, derived_properties


npz_MCnames = ['EW_seed', 'logEW_MC_ref', 'x_MC0_ref', 'BB_MC0_ref',
               'BB_sig_ref', 'sig_limit_ref', 'NB_sel_ref', 'NB_nosel_ref',
               'EW_flag_ref']
'''
               'flux_ref', 'logM_ref', 'NIIHa_ref',
               'logOH_ref', 'HaFlux_ref', 'HaLum_ref', 'logSFR_ref']
'''


def mock_ones(arr0, Nmock):
    """
    Generate (Nmock,Ngal) array using np.ones() to repeat
    """

    return np.ones((Nmock, 1)) * arr0


def random_mags(t_seed, rand_shape, mag_ref, sig_ref):
    """
    Generate randomized array of magnitudes based on ref values and sigma
    """

    N_rep = rand_shape[0]

    np.random.seed(t_seed)
    return mock_ones(mag_ref, N_rep) + np.random.normal(size=rand_shape) * \
           mock_ones(sig_ref, N_rep)


def main(int_dict, npz_MC_file, mock_sz, ss_range, mass_dict, norm_dict,
         filt_dict, EW_dict, NB_MC, lum_dist, mylog, redo=False):

    ff = int_dict['ff']
    mm = int_dict['mm']
    ss = int_dict['ss']
    mass_int = mass_dict['mass_int']
    std_mass_int = mass_dict['std_mass_int']
    logEW_mean = EW_dict['logEW_mean']
    logEW_sig = EW_dict['logEW_sig']
    EW_int = EW_dict['EW_int']

    if not exists(npz_MC_file) or redo:
        EW_seed = mm * len(ss_range) + ss
        mylog.info("seed for mm=%i ss=%i : %i" % (mm, ss, EW_seed))
        np.random.seed(EW_seed)
        rand0 = np.random.normal(0.0, 1.0, size=norm_dict['Ngal'])

        # Randomize based on log-normal EW distribution for Ngal (ref). Not H-alpha EW.
        logEW_MC_ref = logEW_mean[mm] + logEW_sig[ss] * rand0
        stats_log(logEW_MC_ref, "logEW_MC_ref", mylog)

        x_MC0_ref = EW_int(logEW_MC_ref)  # NB color excess
        negs = np.where(x_MC0_ref < 0)
        if len(negs[0]) > 0:
            x_MC0_ref[negs] = 0.0
        stats_log(x_MC0_ref, "x_MC0_ref", mylog)

        # Selection based on 'true' magnitudes
        NB_sel_ref, NB_nosel_ref, \
            sig_limit_ref = NB_select(ff, norm_dict['NB_ref'], x_MC0_ref)

        # Flag array to indicate true galaxies meet selection requirements
        EW_flag_ref = np.zeros(norm_dict['Ngal'])
        EW_flag_ref[NB_sel_ref] = 1

        # Broad-band magnitudes for input sample
        BB_MC0_ref = norm_dict['NB_ref'] + x_MC0_ref
        BB_sig_ref = get_sigma(BB_MC0_ref, cont_lim[ff], sigma=3.0)

        # Combine photometry to get dictionary for derived properties input
        dict_prop_ref = dict_prop_maker(norm_dict['NB_ref'], BB_MC0_ref,
                                        x_MC0_ref, filt_dict, filt_corr[ff],
                                        mass_int, lum_dist)
        der_prop_dict_ref = derived_properties(suffix='_ref', **dict_prop_ref)

        if exists(npz_MC_file):
            mylog.info("Overwriting : " + npz_MC_file)
        else:
            mylog.info("Writing : " + npz_MC_file)

        npz_MCdict = {}
        for name in npz_MCnames:
            npz_MCdict[name] = eval(name)
        npz_MCdict.update(der_prop_dict_ref)  # Add the derived properties dictionary
        np.savez(npz_MC_file, **npz_MCdict)

        der_prop_dict_ref = derived_properties(**dict_prop_ref)
    else:
        if not redo:
            mylog.info("File found : " + npz_MC_file)
            npz_MC = np.load(npz_MC_file)

            npz_MCdict = dict(npz_MC)

            dict_prop_ref = dict_prop_maker(norm_dict['NB_ref'],
                                            npz_MCdict['BB_MC0_ref'],
                                            npz_MCdict['x_MC0_ref'],
                                            filt_dict, filt_corr[ff],
                                            mass_int, lum_dist)

    BB_seed = ff + 5
    mylog.info("seed for broadband, mm=%i ss=%i : %i" % (mm, ss, BB_seed))

    # Broad-band mocked magnitudes
    BB_MC = random_mags(BB_seed, mock_sz, npz_MCdict['BB_MC0_ref'],
                        npz_MCdict['BB_sig_ref'])
    stats_log(BB_MC, "BB_MC", mylog)

    x_MC = BB_MC - NB_MC  # NB color excess (mocked)
    stats_log(x_MC, "x_MC", mylog)

    # Selection based on mocked magnitudes
    NB_sel, NB_nosel, sig_limit = NB_select(ff, NB_MC, x_MC)

    # Flag array to indicate if mock galaxies meet selection requirements
    EW_flag0 = np.zeros(mock_sz)
    EW_flag0[NB_sel[0], NB_sel[1]] = 1

    # Not sure if we should use true logEW or the mocked values
    # Currently using mocked values
    # logEW_MC = mock_ones(logEW_MC_ref, Nmock)

    # Replace NB, BB and x with mocked values
    dict_prop_MC = dict_prop_ref.copy()

    dict_prop_MC['NB'] = NB_MC
    dict_prop_MC['BB'] = BB_MC
    dict_prop_MC['x'] = x_MC

    der_prop_dict_MC = derived_properties(std_mass_int=std_mass_int,
                                          **dict_prop_MC)
    stats_log(der_prop_dict_MC['logEW'], "logEW_MC", mylog)
    stats_log(der_prop_dict_MC['NB_flux'], "flux_MC", mylog)
    stats_log(der_prop_dict_MC['Ha_Flux'], "HaFlux_MC", mylog)

    return dict_prop_ref, der_prop_dict_ref, npz_MCdict, dict_prop_MC,\
           der_prop_dict_MC, EW_flag0, NB_sel, NB_nosel
