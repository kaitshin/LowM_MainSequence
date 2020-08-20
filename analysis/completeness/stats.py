from os.path import join
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import ascii as asc

from . import avg_sig_ctype, M_lab
from .config import npz_path0, filters, path0
from .dataset import get_mact_data


def stats_log(input_arr, arr_type, mylog):
    """
    Purpose:
      Computes, min, max, median, and average value of input array and
      output to log

    :param input_arr: input array (numpy.ndarray)
    :param arr_type: string
    :param mylog: MLog class
    :return None. mylog is called
    """

    min0 = np.nanmin(input_arr)
    max0 = np.nanmax(input_arr)
    mean0 = np.nanmean(input_arr)
    med0 = np.nanmedian(input_arr)

    str0 = "%s: min=%f max=%f mean=%f median=%f" % (arr_type, min0, max0,
                                                    mean0, med0)
    mylog.info(str0)


def avg_sig_label(str0, avg, sigma, panel_type=''):
    """
    Purpose:
        Generate raw strings that contain proper formatting for average and
        sigma EW and fluxes

    :param str0: Input string variable to append
    :param avg: Average value (float)
    :param sigma: Dispersion (float)
    :param panel_type: String describing panel type . Either 'EW' or 'Flux'

    :return: str0: Revised string
    """

    if panel_type == 'EW':
        str0 += r'$\langle\log({\rm EW})\rangle$ = %.2f' % avg
        str0 += '\n' + r'$\sigma[\log({\rm EW})]$ = %.2f' % sigma

    if panel_type == 'Flux':
        str0 += r'$\langle\log(F_{{\rm H}\alpha})\rangle$ = %.2f' % avg
        str0 += '\n' + r'$\sigma[\log(F_{{\rm H}\alpha})]$ = %.2f' % sigma

    return str0


def N_avg_sig_label(x0, avg, sigma):
    """
    String containing average and sigma for ax.legend() labels
    """

    return r'N: %i  $\langle x\rangle$: %.2f  $\sigma$: %.2f' % \
           (x0.size, avg, sigma)


def stats_plot(type0, ax2, ax3, ax, s_row, Ng, No, binso, EW_mean, EW_sig, ss):
    """
    Plot statistics (chi^2, model vs data comparison) for each model

    type0: str
       Either 'EW' or 'Flux'

    ax2 : matplotlib.axes._subplots.AxesSubplot
       matplotlib axes for stats plot

    ax3 : matplotlib.axes._subplots.AxesSubplot
       matplotlib axes for avg_sigma plot

    ax : matplotlib.axes._subplots.AxesSubplot
       matplotlib axes for main plot

    s_row: int
       Integer for row for ax2

    EW_mean: float
       Value of median logEW in model

    EW_sig: float
       Value of sigma logEW in model

    ss: int
       Integer indicating index for sigma
    """

    delta = (Ng - No) / np.sqrt(Ng + No)

    if type0 == 'EW':
        pn = 0
    if type0 == 'Flux':
        pn = 1

    ax2[s_row][pn].axhline(0.0, linestyle='dashed')  # horizontal line at zero

    ax2[s_row][pn].scatter(binso[:-1], delta)
    no_use = np.where((Ng == 0) | (No == 0))[0]

    if len(no_use) > 0:
        ax2[s_row][pn].scatter(binso[:-1][no_use], delta[no_use], marker='x',
                               color='r', s=20)

    # ax2[s_row][pn].set_ylabel(r'1 - $N_{\rm mock}/N_{\rm data}$')
    if type0 == 'EW':
        ax2[s_row][pn].set_ylabel(r'$(N_{\rm mock} - N_{\rm data})/\sigma$')

        annot_txt = r'$\langle\log({\rm EW})\rangle = %.2f$  ' % EW_mean
        annot_txt += r'$\sigma[\log({\rm EW})] = %.2f$' % EW_sig
        ax2[s_row][pn].set_title(annot_txt, fontdict={'fontsize': 10}, loc='left')

    # Compute chi^2
    use_bins = np.where((Ng != 0) & (No != 0))[0]
    if len(use_bins) > 2:
        fit_chi2 = np.sum(delta[use_bins] ** 2) / (len(use_bins) - 2)
        c_txt = r'$\chi^2_{\nu}$ = %.2f' % fit_chi2

        ax3.scatter([EW_mean + 0.005 * (ss - 3 / 2.)], [fit_chi2],
                    marker='o', s=40, color=avg_sig_ctype[ss],
                    edgecolor='none')
    else:
        print("Too few bins")
        c_txt = r'$\chi^2_{\nu}$ = Unavailable'
        fit_chi2 = np.nan

    ax.annotate(c_txt, [0.025, 0.975], xycoords='axes fraction',
                ha='left', va='top')
    c_txt += '\n' + r'N = %i' % len(use_bins)
    ax2[s_row][pn].annotate(c_txt, [0.975, 0.975], ha='right',
                            xycoords='axes fraction', va='top')

    return fit_chi2


def bin_MACT(x_cen, dict_NB):

    bin_size = x_cen[1] - x_cen[0]

    N_bins = np.zeros(x_cen.shape)
    logMstar = dict_NB['logMstar']

    for bb in range(len(N_bins)):
        idx = np.where((logMstar >= x_cen[bb]-bin_size/2.0) &
                       (logMstar < x_cen[bb]+bin_size/2.0))[0]
        N_bins[bb] = len(idx)

    return N_bins


def lowM_cutoff_sigma(logMstar):
    "Return low-mass cutoff.  If lower than 6.0 return 6.0"
    avg0 = np.average(logMstar)
    sig0 = np.std(logMstar)

    lowM_cutoff = avg0 - 1.5 * sig0
    if lowM_cutoff < 6.0:
        lowM_cutoff = 6.0

    return lowM_cutoff


def compute_weighted_dispersion(best_fit_file, mylog):
    """
    Purpose:
      Computes a weighted dispersion of the main sequence vs stellar mass.
      Here the best fit for each MC set for each filter is used, and weighting
      is determined based on the MACT sample size in each stellar mass bin

    :return:
    """

    fig, ax = plt.subplots()

    mylog.info("Reading: "+best_fit_file)
    comp_tab0 = asc.read(best_fit_file)

    best_EWmean = comp_tab0['log_EWmean'].data
    best_EWsig  = comp_tab0['log_EWsig'].data

    # ctype = ['b', 'b', 'orange', 'g', 'r']
    for filt, ff in zip(filters, range(len(filters))):
        # Read in MACT sample
        dict_NB = get_mact_data(ff)

        # Read in dispersion
        infile = join(npz_path0,
                      '%s_SFR_bin_%.2f_%0.2f.npz' % (filt, best_EWmean[ff], best_EWsig[ff]))
        npz0 = np.load(infile)

        x_cen = npz0['x_cen']
        std_full = npz0['y_std_full']
        std_sel = npz0['y_std_sel']

        N_bins = bin_MACT(x_cen, dict_NB)

        if ff == 0:
            set_shape = (len(filt), len(x_cen))
            wht_sig_full = np.zeros(set_shape)
            wht_sig_sel = np.zeros(set_shape)
            N_bins_filt = np.zeros(set_shape)

        wht_sig_full[ff] = std_full
        wht_sig_sel[ff] = std_sel
        N_bins_filt[ff] = N_bins

        # ax.plot(x_cen, std_full, color=ctype[ff], linestyle='dotted', label=filt)
        # ax.plot(x_cen, std_sel, color=ctype[ff], linestyle='dashed')

    sig_full_sq = wht_sig_full**2 * N_bins_filt
    sig_full = np.sqrt(np.sum(sig_full_sq, axis=0) / np.sum(N_bins_filt, axis=0))

    sig_sel_sq = wht_sig_sel**2 * N_bins_filt
    sig_sel = np.sqrt(np.sum(sig_sel_sq, axis=0) / np.sum(N_bins_filt, axis=0))

    ax.plot(x_cen, sig_full, linestyle='dotted', label='Weighted (full)')
    ax.plot(x_cen, sig_sel, linestyle='dashed', label='Weighted (selected)')

    ax.legend(loc='upper left')

    ax.set_xlabel(M_lab)
    ax.set_ylabel(r'$\sigma$ [dex]')

    ax.set_xlim([6.0, 9.95])

    ax.tick_params(axis='both', direction='in')

    plt.subplots_adjust(left=0.08, right=0.99, bottom=0.09, top=0.99)
    out_pdf = join(path0, 'Completeness/compute_weighted_dispersion.pdf')
    mylog.info("Writing : "+out_pdf)
    fig.savefig(out_pdf, format='pdf')
