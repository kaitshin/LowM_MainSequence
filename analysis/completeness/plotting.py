import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from chun_codes import intersect_ndim

from . import EW_lab, Flux_lab, avg_sig_ctype, M_lab
from . import cmap_sel, cmap_nosel
from .stats import avg_sig_label, N_avg_sig_label
from .fitting import fit_sequence, draw_linear_fit, linear

M_xlimit = (4.0, 10.0)
disp_limit = (-0.05, 1.0)


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


def plot_MACT(ax, dict_NB, x0, y0, size=5):
    """
    Plot MACT spectroscopic and photometric sample in various sub-panel

    :param ax: matplotlib.axes._subplots.AxesSubplot
       sub-Axis to plot

    :param dict_NB: dictionary containing NB data

    :param x0: list or numpy.array or string corresponding to dict_NB key
       Array to plot on x-axis

    :param y0: list or numpy.array or string corresponding to dict_NB key
       Array to plot on y-axis

    :param size: integer for matplotlib size. Default: 5
    """

    w_spec = dict_NB['w_spec']
    wo_spec = dict_NB['wo_spec']

    if isinstance(x0, str):
        x0 = dict_NB[x0]

    if isinstance(y0, str):
        y0 = dict_NB[y0]

    ax.scatter(x0[w_spec], y0[w_spec], color='k', edgecolor='none',
               alpha=0.5, s=size)
    ax.scatter(x0[wo_spec], y0[wo_spec], facecolor='none', edgecolor='k',
               alpha=0.5, s=size)


def plot_mock(ax, dict_MC, x0, y0, x_limit=None, y_limit=None,
              xlabel='', ylabel=''):
    """
    Plot mocked galaxies in various sub-panel

    ax : matplotlib.axes._subplots.AxesSubplot
       sub-Axis to plot

    dict_MC: dictionary containing photometry and NB selection

    x0 : list or numpy.array
       Array to plot on x-axis

    y0 : list or numpy.array
       Array to plot on y-axis

    xlabel: str
       String for x-axis.  Set to '' to not show a label

    ylabel: str
       String for y-axis.  Set to '' to not show a label
    """

    NB_sel = dict_MC['NB_sel']
    NB_nosel = dict_MC['NB_nosel']

    if isinstance(x0, str):
        x0 = dict_MC[x0]

    if isinstance(y0, str):
        y0 = dict_MC[y0]

    is1, is2 = NB_sel[0], NB_sel[1]
    in1, in2 = NB_nosel[0], NB_nosel[1]

    if not isinstance(x_limit, type(None)):
        ax.set_xlim(x_limit)
    else:
        ax.set_xlim(np.nanmin(x0), np.nanmax(x0))

    if not isinstance(y_limit, type(None)):
        ax.set_ylim(y_limit)
    else:
        ax.set_ylim(np.nanmin(y0), np.nanmax(y0))

    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1])
    ax.hexbin(x0[in1, in2], y0[in1, in2], gridsize=100, mincnt=1,
              extent=extent, cmap=cmap_nosel, linewidth=0.2)
    ax.hexbin(x0[is1, is2], y0[is1, is2], gridsize=100, mincnt=1,
              extent=extent, cmap=cmap_sel, linewidth=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlabel == '':
        ax.set_xticklabels([])


def overlay_mock_average_dispersion(ax, dict_MC, x0, y0, lowM_cutoff):

    NB_sel = dict_MC['NB_sel']

    if isinstance(x0, str):
        x0 = dict_MC[x0]

    if isinstance(y0, str):
        y0 = dict_MC[y0]

    bin_size = 0.5
    x_bins = np.arange(4.0, 10.5, bin_size)
    x_cen = x_bins + bin_size/2.0

    N_full = np.zeros(x_bins.shape)
    N_sel = np.zeros(x_bins.shape)
    y_avg_full = np.zeros(x_bins.shape)
    y_std_full = np.zeros(x_bins.shape)
    y_avg_sel = np.zeros(x_bins.shape)
    y_std_sel = np.zeros(x_bins.shape)

    for ii in range(x_bins.shape[0]):
        idx = np.where((x0 >= x_bins[ii]) & (x0 < x_bins[ii]+bin_size))
        N_full[ii] = len(idx[0])

        if N_full[ii] > 0:
            y_avg_full[ii] = np.nanmean(y0[idx[0], idx[1]])
            y_std_full[ii] = np.nanstd(y0[idx[0], idx[1]])

            NB_sel0 = intersect_ndim(idx, NB_sel, y0.shape)
            if type(NB_sel0) == tuple:
                N_sel[ii] = len(NB_sel0[0])

                if N_full[ii] > 0:
                    y_avg_sel[ii] = np.nanmean(y0[NB_sel0])
                    y_std_sel[ii] = np.nanstd(y0[NB_sel0])
            else:
                print("Empty idx or NB_sel")

    nonzero_sel = np.where(N_sel > 0)

    ax.errorbar(x_cen[nonzero_sel], y_avg_sel[nonzero_sel],
                yerr=y_std_sel[nonzero_sel], marker='s', fmt='s', color='m',
                markeredgecolor='none', alpha=0.5, label='NB-selected mock sample')

    # Draw best fit for selection
    nonzero_masscut_sel = np.where((N_sel > 0) & (x_cen >= 6.0))
    sel_fit = fit_sequence(x_cen, y_avg_sel, nonzero_masscut_sel)
    draw_linear_fit(ax, x_cen[nonzero_masscut_sel], sel_fit, 0.30, color='m')

    nonzero_full = np.where(N_full > 0)
    ax.errorbar(x_cen[nonzero_full], y_avg_full[nonzero_full],
                yerr=y_std_full[nonzero_full], marker='s', fmt='s', color='k',
                markeredgecolor='none', alpha=0.75, label='Full mock sample')

    # Draw line for lowM_cutoff
    ax.axvline(x=lowM_cutoff, color='r', linestyle='dashed', linewidth=1.5)

    # Draw best fit for full mock sample
    nonzero_masscut_full = np.where((N_full > 0) & (x_cen >= lowM_cutoff))
    full_fit = fit_sequence(x_cen, y_avg_full, nonzero_masscut_full)
    draw_linear_fit(ax, x_cen[nonzero_masscut_full], full_fit, 0.25, color='k')

    ax.legend(loc='lower left', frameon=False, fontsize=10)

    ax.set_xlim(M_xlimit)

    bin_MC = dict()
    bin_MC['x_bins'] = x_bins
    bin_MC['x_cen'] = x_cen
    bin_MC['y_avg_full'] = y_avg_full
    bin_MC['y_std_full'] = y_std_full
    bin_MC['y_avg_sel'] = y_avg_sel
    bin_MC['y_std_sel'] = y_std_sel
    bin_MC['nonzero_full'] = nonzero_full
    bin_MC['nonzero_sel'] = nonzero_sel
    bin_MC['sel_fit'] = sel_fit
    bin_MC['full_fit'] = full_fit

    # Add stellar mass and SFRs for random selection
    bin_MC['logM_MC']   = x0[NB_sel]
    bin_MC['logSFR_MC'] = y0[NB_sel]
    bin_MC['offset_MC'] = y0[NB_sel] - linear(x0[NB_sel], *sel_fit)  # This is offset from best fit

    return bin_MC


def plot_dispersion(ax, bin_MC, xlabel=M_lab):

    x_cen = bin_MC['x_cen']
    nonzero_full = bin_MC['nonzero_full']
    nonzero_sel = bin_MC['nonzero_sel']

    # Plot full set
    ax.scatter(x_cen[nonzero_full], bin_MC['y_std_full'][nonzero_full],
               marker='s', color='k', alpha=0.75)

    # Plot selected set
    ax.scatter(x_cen[nonzero_sel], bin_MC['y_std_sel'][nonzero_sel],
               marker='s', color='m', alpha=0.75)

    ax.set_xlim(M_xlimit)
    ax.set_ylim(disp_limit)
    ax.set_xlabel(xlabel)

    ax.set_ylabel(r'$\sigma$ [dex]')


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


def plot_completeness(t_ax, dict_MC, arr0, bins, ref_arr0=None,
                      above_break=None, annotate=True):

    if isinstance(ref_arr0, dict):
        ref0 = ref_arr0[arr0]
    else:
        ref0 = ref_arr0

    if isinstance(arr0, str):
        arr0 = dict_MC[arr0]

    NB_sel0 = dict_MC['NB_sel']

    # Plot mocked set
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
        t_ax.axvline(comp_50, linestyle='dashed', color='blue', linewidth=1.5)
        t_ax.annotate('%.2f' % comp_50, [0.975, 0.025], xycoords='axes fraction',
                      ha='right', va='bottom', fontsize=8, color='blue')

    # Plot modeled/"true" set
    if not isinstance(ref_arr0, type(None)):
        arr1 = np.ones((arr0.shape[0], 1)) * ref0
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
            t_ax.axvline(comp_50_ref, linestyle='dashed', color='black', linewidth=1.5)
            t_ax.annotate('%.2f' % comp_50_ref, [0.975, 0.06], fontsize=8,
                          xycoords='axes fraction', ha='right', va='bottom',
                          color='black')

    if annotate:
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
