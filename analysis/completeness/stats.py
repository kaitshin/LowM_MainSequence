import numpy as np


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