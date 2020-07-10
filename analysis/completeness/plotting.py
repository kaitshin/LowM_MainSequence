import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from chun_codes import intersect_ndim

from . import EW_lab, Flux_lab, avg_sig_ctype
from . import cmap_sel, cmap_nosel
from .stats import avg_sig_label, N_avg_sig_label


def avg_sig_plot_init(t_filt, logEW_mean, avg_NB, sig_NB, avg_NB_flux,
                      sig_NB_flux):
    """
    Initialize fig and axes objects for avg_sigma plot and set matplotlib
    aesthetics
    """

    xlim = [min(logEW_mean) - 0.05, max(logEW_mean) + 0.05]
    ylim1 = [avg_NB - sig_NB - 0.05, avg_NB + sig_NB + 0.15]
    ylim2 = [avg_NB_flux - sig_NB_flux - 0.05, avg_NB_flux + sig_NB_flux + 0.15]

    xticks = np.arange(xlim[0], xlim[1], 0.1)
    fig3, ax3 = plt.subplots(ncols=2, nrows=2)

    ax3[0][0].axhline(y=avg_NB, color='black', linestyle='dashed')
    ax3[0][0].axhspan(avg_NB - sig_NB, avg_NB + sig_NB, alpha=0.5, color='black')
    ax3[0][0].set_xlim(xlim)
    ax3[0][0].set_xticks(xticks)
    # ax3[0][0].set_ylim(ylim1)
    ax3[0][0].set_ylabel(EW_lab)
    ax3[0][0].set_xticklabels([])
    ax3_txt = avg_sig_label(t_filt + '\n', avg_NB, sig_NB, panel_type='EW')
    ax3[0][0].annotate(ax3_txt, (0.025, 0.975), xycoords='axes fraction',
                       ha='left', va='top', fontsize=11)

    ax3[1][0].axhline(y=avg_NB_flux, color='black', linestyle='dashed')
    ax3[1][0].axhspan(avg_NB_flux - sig_NB_flux, avg_NB_flux + sig_NB_flux,
                      alpha=0.5, color='black')
    ax3[1][0].set_xlim(xlim)
    ax3[1][0].set_xticks(xticks)
    # ax3[1][0].set_ylim(ylim2)
    ax3[1][0].set_xlabel(EW_lab)
    ax3[1][0].set_ylabel(Flux_lab)
    ax3_txt = avg_sig_label('', avg_NB_flux, sig_NB_flux, panel_type='Flux')
    ax3[1][0].annotate(ax3_txt, (0.025, 0.975), xycoords='axes fraction',
                       ha='left', va='top', fontsize=11)

    ax3[0][1].set_xlim(xlim)
    ax3[0][1].set_xticks(xticks)
    ax3[0][1].set_ylabel(r'$\chi^2_{\nu}$')
    ax3[0][1].set_xticklabels([])
    ax3[0][1].set_ylim([0.11, 100])
    ax3[0][1].set_yscale('log')

    ax3[1][1].set_xlim(xlim)
    ax3[1][1].set_xticks(xticks)
    ax3[1][1].set_ylabel(r'$\chi^2_{\nu}$')
    ax3[1][1].set_xlabel(EW_lab)
    ax3[1][1].set_ylim([0.11, 100])
    ax3[1][1].set_yscale('log')

    return fig3, ax3


def plot_MACT(ax, dict_NB, x0, y0):
    """
    Plot MACT spectroscopic and photometric sample in various sub-panel

    :param ax: matplotlib.axes._subplots.AxesSubplot
       sub-Axis to plot

    :param dict_NB: dictionary containing NB data

    :param x0: list or numpy.array or string corresponding to dict_NB key
       Array to plot on x-axis

    :param y0: list or numpy.array or string corresponding to dict_NB key
       Array to plot on y-axis
    """

    w_spec = dict_NB['w_spec']
    wo_spec = dict_NB['wo_spec']

    if isinstance(x0, str):
        x0 = dict_NB[x0]

    if isinstance(y0, str):
        y0 = dict_NB[y0]

    ax.scatter(x0[w_spec], y0[w_spec], color='k', edgecolor='none',
               alpha=0.5, s=5)
    ax.scatter(x0[wo_spec], y0[wo_spec], facecolor='none', edgecolor='k',
               alpha=0.5, s=5)


def plot_mock(ax, x0, y0, NB_sel, NB_nosel, xlabel, ylabel):
    """
    Plot mocked galaxies in various sub-panel

    ax : matplotlib.axes._subplots.AxesSubplot
       sub-Axis to plot

    x0 : list or numpy.array
       Array to plot on x-axis

    y0 : list or numpy.array
       Array to plot on y-axis

    NB_sel: numpy.array
       Index array indicating which sources are NB selected

    wo_spec: numpy.array
       Index array indicating which sources are not NB selected

    xlabel: str
       String for x-axis.  Set to '' to not show a label

    xlabel: str
       String for y-axis.  Set to '' to not show a label
    """

    is1, is2 = NB_sel[0], NB_sel[1]
    in1, in2 = NB_nosel[0], NB_nosel[1]

    ax.hexbin(x0[is1, is2], y0[is1, is2], gridsize=100, mincnt=1, cmap=cmap_sel,
              linewidth=0.2)
    ax.hexbin(x0[in1, in2], y0[in1, in2], gridsize=100, mincnt=1, cmap=cmap_nosel,
              linewidth=0.2)
    # ax.scatter(x0[is1,is2], y0[is1,is2], alpha=0.25, s=2, edgecolor='none')
    # ax.scatter(x0[in1,in2], y0[in1,in2], alpha=0.25, s=2, edgecolor='red',
    #            linewidth=0.25, facecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlabel == '':
        ax.set_xticklabels([])


def get_completeness(hist_bins, hist_data):
    """
    Determine 50% completeness for various quantities (sSFR, EW, Flux)
    """

    i_hist = interp1d(hist_data, hist_bins)

    try:
        comp_50 = i_hist(0.50)
    except ValueError:
        print("ValueError. Setting to zero")
        comp_50 = 0.0

    return comp_50


def plot_completeness(t_ax, arr0, NB_sel0, bins, ref_arr0=None, above_break=None, annotate=True):
    finite = np.where(np.isfinite(arr0))
    if not isinstance(above_break, type(None)):
        finite = intersect_ndim(above_break, finite, arr0.shape)
        NB_sel0 = intersect_ndim(above_break, NB_sel0, arr0.shape)

    orig, bins_edges0 = np.histogram(arr0[finite], bins)

    NB_sel = intersect_ndim(NB_sel0, finite, arr0.shape)

    sel, bins_edges1 = np.histogram(arr0[NB_sel], bins)

    if len(bins_edges0) != len(bins_edges1):
        print("bin_edges0 != bin_edges1")

    x0 = bins_edges0[:-1]
    y0 = sel / np.float_(orig)
    label0 = 'mocked' if annotate else ''
    t_ax.step(x0, y0, 'b--', where='mid', label=label0)

    comp_50 = get_completeness(x0, y0)
    if annotate:
        t_ax.annotate('%.2f' % comp_50, [0.975, 0.025], xycoords='axes fraction',
                      ha='right', va='bottom', fontsize=8, color='blue')

    if not isinstance(ref_arr0, type(None)):
        arr1 = np.ones((arr0.shape[0], 1)) * ref_arr0
        finite = np.where(np.isfinite(arr1))
        if not isinstance(above_break, type(None)):
            finite = intersect_ndim(above_break, finite, arr0.shape)

        orig1, bins_edges01 = np.histogram(arr1[finite], bins)

        NB_sel = intersect_ndim(NB_sel0, finite, arr1.shape)

        sel1, bins_edges11 = np.histogram(arr1[NB_sel], bins)

        if len(bins_edges01) != len(bins_edges11):
            print("bin_edges01 != bin_edges11")

        x1 = bins_edges01[:-1]
        y1 = sel1 / np.float_(orig1)
        label0 = 'true' if annotate else ''
        t_ax.step(x1, y1, 'k--', where='mid', label=label0)

        comp_50_ref = get_completeness(x1, y1)
        if annotate:
            t_ax.annotate('%.2f' % comp_50_ref, [0.975, 0.06], fontsize=8,
                          xycoords='axes fraction', ha='right', va='bottom',
                          color='black')

    t_ax.legend(loc='upper left', fancybox=True, fontsize=8, framealpha=0.75)

    if isinstance(ref_arr0, type(None)):
        return comp_50
    else:
        return comp_50, comp_50_ref


def ew_flux_hist(type0, mm, ss, t2_ax, x0, avg_x0, sig_x0, x0_bins, logEW_mean,
                 logEW_sig, EW_flag0, x0_arr0, ax3=None):
    """
    Generate histogram plots for EW or flux
    """

    if type0 == 'EW':
        x0_lab = EW_lab
    if type0 == 'Flux':
        x0_lab = Flux_lab

    label_x0 = N_avg_sig_label(x0, avg_x0, sig_x0)
    No, binso, _ = t2_ax.hist(x0, bins=x0_bins, align='mid', color='black',
                              alpha=0.5, linestyle='solid', edgecolor='none',
                              histtype='stepfilled', label=label_x0)
    t2_ax.axvline(x=avg_x0, color='black', linestyle='solid', linewidth=1.5)

    good = np.where((EW_flag0 == 1) & (np.isfinite(x0_arr0)))

    # Normalize relative to selected sample
    if len(good[0]) > 0:
        finite = np.where(np.isfinite(x0_arr0))

        norm0 = float(len(x0)) / len(good[0])
        wht0 = np.repeat(norm0, x0_arr0.size)
        wht0 = np.reshape(wht0, x0_arr0.shape)

        avg_MC = np.average(x0_arr0[finite])
        sig_MC = np.std(x0_arr0[finite])
        label0 = N_avg_sig_label(x0_arr0, avg_MC, sig_MC)

        N, bins, _ = t2_ax.hist(x0_arr0[finite], bins=x0_bins, weights=wht0[finite],
                                align='mid', color='black', linestyle='dashed',
                                edgecolor='black', histtype='step', label=label0)
        t2_ax.axvline(x=avg_MC, color='black', linestyle='dashed', linewidth=1.5)

        avg_gd = np.average(x0_arr0[good[0], good[1]])
        sig_gd = np.std(x0_arr0[good[0], good[1]])
        label1 = N_avg_sig_label(good[0], avg_gd, sig_gd)
        Ng, binsg, _ = t2_ax.hist(x0_arr0[good], bins=x0_bins, weights=wht0[good],
                                  align='mid', alpha=0.5, color='blue',
                                  edgecolor='blue', linestyle='solid',
                                  histtype='stepfilled', label=label1)
        t2_ax.axvline(x=avg_gd, color='blue', linestyle='solid', linewidth=1.5)

        t2_ax.legend(loc='upper right', fancybox=True, fontsize=6, framealpha=0.75)
        t2_ax.set_xlabel(x0_lab)
        t2_ax.set_yscale('log')
        t2_ax.set_ylim([0.1, 1e3])
        if type0 == 'EW':
            t2_ax.set_ylabel(r'$N$')
            t2_ax.set_xlim([0.0, 2.95])
        if type0 == 'Flux':
            t2_ax.set_ylabel('')
            t2_ax.set_yticklabels([''] * 5)
            t2_ax.set_xlim([-18.0, -14.0])
            t2_ax.set_xticks(np.arange(-17.5, -13.5, 1.0))
            t2_ax.set_xticks(np.arange(-17.5, -13.5, 1.0))

        as_label = ''
        if mm == 0:
            as_label = '%.2f' % logEW_sig[ss]

        if not isinstance(ax3, type(None)):
            temp_x = [logEW_mean[mm] + 0.005 * (ss - 3 / 2.)]
            ax3.scatter(temp_x, [avg_gd], marker='o', s=40, edgecolor='none',
                        color=avg_sig_ctype[ss], label=as_label)
            ax3.errorbar(temp_x, [avg_gd], yerr=[sig_gd], capsize=0,
                         elinewidth=1.5, ecolor=avg_sig_ctype[ss], fmt='none')

    return No, Ng, binso, wht0
