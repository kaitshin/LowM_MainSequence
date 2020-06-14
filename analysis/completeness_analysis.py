"""
completeness_analysis
====

A set of Python 2.7 codes for completeness analysis of NB-selected galaxies
in the M*-SFR plot
"""

import os

from chun_codes import intersect_ndim, TimerClass

from datetime import date

from os.path import exists

from astropy.table import Table, vstack

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.interpolate import interp1d

from NB_errors import ew_flux_dual, mag_combine

from NB_errors import filt_ref, dNB, lambdac, dBB, epsilon

from ..mainseq_corrections import niiha_oh_determine

import logging

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)

formatter = logging.Formatter('%(asctime)s - %(module)12s.%(funcName)20s - %(levelname)s: %(message)s')

"""
Pass through ew_MC() call
Nsim  = 5000. # Number of modelled galaxies
Nmock = 10    # Number of mocked galaxies
"""

filters = ['NB704', 'NB711', 'NB816', 'NB921', 'NB973']
cont0 = [r'$R_Ci^{\prime}$', r'$R_Ci^{\prime}$', r'$i^{\prime}z^{\prime}$',
         r'$z^{\prime}$', r'$z^{\prime}$']
NB_filt = np.array([xx for xx in range(len(filt_ref)) if 'NB' in filt_ref[xx]])
for arr in ['filt_ref', 'dNB', 'lambdac', 'dBB', 'epsilon']:
    cmd1 = arr + ' = np.array('+arr+')'
    exec(cmd1)
    cmd2 = arr + ' = '+arr+'[NB_filt]'
    exec(cmd2)

# Limiting magnitudes for NB and BB data
m_NB = np.array([26.7134 - 0.047, 26.0684, 26.9016 + 0.057, 26.7088 - 0.109, 25.6917 - 0.051])
m_BB1 = np.array([28.0829, 28.0829, 27.7568, 26.8250, 26.8250])
m_BB2 = np.array([27.7568, 27.7568, 26.8250, 00.0000, 00.0000])
cont_lim = mag_combine(m_BB1, m_BB2, epsilon)

# Minimum NB excess color for selection
minthres = [0.15, 0.15, 0.15, 0.2, 0.25]

if exists('/Users/cly/GoogleDrive'):
    path0 = '/Users/cly/GoogleDrive/Research/NASA_Summer2015/'
if exists('/Users/cly/Google Drive'):
    path0 = '/Users/cly/Google Drive/Research/NASA_Summer2015/'

npz_path0 = '/Users/cly/data/SDF/MACT/LowM_MainSequence_npz/'
if not exists(npz_path0):
    os.mkdir(npz_path0)

m_AB = 48.6

# Common text for labels
EW_lab = r'$\log({\rm EW}/\AA)$'
Flux_lab = r'$\log(F_{{\rm H}\alpha})$'
M_lab = r'$\log(M_{\star}/M_{\odot})$'
SFR_lab = r'$\log({\rm SFR}[{\rm H}\alpha]/M_{\odot}\,{\rm yr}^{-1})$'

EW_bins = np.arange(0.2, 3.0, 0.2)
Flux_bins = np.arange(-17.75, -14.00, 0.25)
sSFR_bins = np.arange(-11.0, -6.0, 0.2)
SFR_bins = np.arange(-5.0, 2.0, 0.2)
# Colors for each separate points on avg_sigma plots
avg_sig_ctype = ['m', 'r', 'g', 'b', 'k']

cmap_sel = plt.cm.Blues
cmap_nosel = plt.cm.Reds

# Dictionary names
npz_NBnames = ['N_mag_mock', 'Ndist_mock', 'Ngal', 'Nmock', 'NB_ref', 'NB_sig_ref']

npz_MCnames = ['EW_seed', 'logEW_MC_ref', 'x_MC0_ref', 'BB_MC0_ref',
               'BB_sig_ref', 'sig_limit_ref', 'NB_sel_ref', 'NB_nosel_ref',
               'EW_flag_ref', 'flux_ref', 'logM_ref', 'NIIHa_ref',
               'logOH_ref', 'HaFlux_ref', 'HaLum_ref', 'logSFR_ref']


class MLog:
    """
    Main class to log information to stdout and ASCII file

    To execute:
    mylog = MLog(dir0)._get_logger()

    Parameters
    ----------
    dir0 : str
      Full path for where log files should be placed

    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 2 October 2019
    """

    def __init__(self, dir0, str_date):
        self.LOG_FILENAME = dir0 + 'completeness_analysis.' + str_date + '.log'
        self._log = self._get_logger()

    def _get_logger(self):
        loglevel = logging.INFO
        log = logging.getLogger(self.LOG_FILENAME)
        if not getattr(log, 'handler_set', None):
            log.setLevel(logging.INFO)
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            log.addHandler(sh)

            fh = logging.FileHandler(self.LOG_FILENAME)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)

            log.setLevel(loglevel)
            log.handler_set = True
        return log


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


def compute_EW(x0, ff):
    y_temp = 10 ** (-0.4 * x0)
    EW_ref = np.log10(dNB[ff] * (1 - y_temp) / (y_temp - dNB[ff] / dBB[ff]))
    return EW_ref


def plot_NB_select(ff, t_ax, NB, ctype, linewidth=1, plot4=True):
    t_ax.axhline(y=minthres[ff], linestyle='dashed', color=ctype)

    y3 = color_cut(NB, m_NB[ff], cont_lim[ff])
    t_ax.plot(NB, y3, ctype + '--', linewidth=linewidth)

    y3_int = interp1d(y3, NB)
    NB_break = y3_int(minthres[ff])

    if plot4:
        y4 = color_cut(NB, m_NB[ff], cont_lim[ff], sigma=4.0)
        t_ax.plot(NB, y4, ctype + ':', linewidth=linewidth)

    return NB_break


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


def get_mag_vs_mass_interp(prefix_ff):
    """
    Purpose:
      Define interpolation function between continuum magnitude and stellar mass

    :param prefix_ff: filter prefix (str)
      Either 'Ha-NB7', 'Ha-NB816', 'Ha-NB921', or 'Ha-NB973'

    :return mass_int: interp1d object for logarithm of stellar mass, logM
    :return std_mass_int: interp1d object for dispersion in logM
    """

    npz_mass_file = path0 + 'Completeness/mag_vs_mass_' + prefix_ff + '.npz'
    npz_mass = np.load(npz_mass_file, allow_pickle=True)
    cont_arr = npz_mass['cont_arr']
    dmag = cont_arr[1] - cont_arr[0]
    mgood = np.where(npz_mass['N_logM'] != 0)[0]

    x_temp = cont_arr + dmag / 2.0
    mass_int = interp1d(x_temp[mgood], npz_mass['avg_logM'][mgood],
                        bounds_error=False, fill_value='extrapolate',
                        kind='linear')

    mbad = np.where(npz_mass['N_logM'] <= 1)[0]
    std0 = npz_mass['std_logM']
    if len(mbad) > 0:
        std0[mbad] = 0.30

    std_mass_int = interp1d(x_temp, std0, fill_value=0.3, bounds_error=False,
                            kind='nearest')
    return mass_int, std_mass_int


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


def plot_MACT(ax, x0, y0, w_spec, wo_spec):
    """
    Plot MACT spectroscopic and photometric sample in various sub-panel

    ax : matplotlib.axes._subplots.AxesSubplot
       sub-Axis to plot

    x0 : list or numpy.array
       Array to plot on x-axis

    y0 : list or numpy.array
       Array to plot on y-axis

    w_spec: numpy.array
       Index array indicating which sources with spectra

    wo_spec: numpy.array
       Index array indicating which sources without spectra (i.e., photometric)
    """

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


def ew_MC(Nsim=5000., Nmock=10, debug=False, redo=False):
    """
    Main function for Monte Carlo realization.  Adopts log-normal
    EW distribution to determine survey sensitivity and impact on
    M*-SFR relation

    Parameters
    ----------
    Nsim: Number of modelled galaxies (int)
    Nmock: Number of mock galaxies for each modelled galaxy (int)
    debug : boolean
      If enabled, a quicker version is executed for test-driven development.
      Default: False
    redo : boolean
      Re-run mock galaxy generation even if file exists. Default: False
    """

    today0 = date.today()
    str_date = "%02i%02i" % (today0.month, today0.day)
    if debug:
        str_date += ".debug"
    mylog = MLog(path0 + 'Completeness/', str_date)._get_logger()

    t0 = TimerClass()
    t0._start()

    prefixes = ['Ha-NB7', 'Ha-NB7', 'Ha-NB816', 'Ha-NB921', 'Ha-NB973']

    # NB statistical filter correction
    filt_corr = [1.289439104, 1.41022358406, 1.29344789854,
                 1.32817034288, 1.29673596942]

    z_NB = lambdac / 6562.8 - 1.0

    npz_slope = np.load(path0 + 'Completeness/NB_numbers.npz',
                        allow_pickle=True)

    logEW_mean_start = np.array([1.25, 1.25, 1.25, 1.25, 0.90])
    logEW_sig_start = np.array([0.15, 0.55, 0.25, 0.35, 0.55])
    n_mean = 4
    n_sigma = 4

    mylog.info('Nsim : ', Nsim)

    NBbin = 0.25

    nrow_stats = 4

    # One file written for all avg and sigma comparisons
    if not debug:
        out_pdf3 = path0 + 'Completeness/ew_MC.avg_sigma.pdf'
        pp3 = PdfPages(out_pdf3)

    ff_range = [0] if debug else range(len(filt_ref))
    mm_range = [0] if debug else range(n_mean)
    ss_range = [0] if debug else range(n_sigma)

    for ff in ff_range:  # loop over filter
        t_ff = TimerClass()
        t_ff._start()
        mylog.info("Working on : " + filters[ff])

        logEW_mean = logEW_mean_start[ff] + 0.1 * np.arange(n_mean)
        logEW_sig = logEW_sig_start[ff] + 0.1 * np.arange(n_sigma)

        comp_shape = (len(mm_range), len(ss_range))
        comp_sSFR = np.zeros(comp_shape)
        # comp_EW   = np.zeros(comp_shape)
        comp_SFR = np.zeros(comp_shape)
        comp_flux = np.zeros(comp_shape)
        comp_EWmean = np.zeros(comp_shape)
        comp_EWsig = np.zeros(comp_shape)

        out_pdf = path0 + 'Completeness/ew_MC_' + filters[ff] + '.pdf'
        if debug:
            out_pdf = out_pdf.replace('.pdf', '.debug.pdf')
        pp = PdfPages(out_pdf)

        # This is cropped to fit
        out_pdf0 = path0 + 'Completeness/ew_MC_' + filters[ff] + '.crop.pdf'
        if debug:
            out_pdf0 = out_pdf0.replace('.pdf', '.debug.pdf')
        pp0 = PdfPages(out_pdf0)

        out_pdf2 = path0 + 'Completeness/ew_MC_' + filters[ff] + '.stats.pdf'
        if debug:
            out_pdf2 = out_pdf2.replace('.pdf', '.debug.pdf')
        pp2 = PdfPages(out_pdf2)

        out_pdf4 = path0 + 'Completeness/ew_MC_' + filters[ff] + '.comp.pdf'
        if debug:
            out_pdf4 = out_pdf4.replace('.pdf', '.debug.pdf')
        pp4 = PdfPages(out_pdf4)

        filt_dict = {'dNB': dNB[ff], 'dBB': dBB[ff], 'lambdac': lambdac[ff]}

        x = np.arange(0.01, 10.00, 0.01)
        EW_ref = compute_EW(x, ff)

        good = np.where(np.isfinite(EW_ref))[0]
        mylog.info('EW_ref (min/max): %f %f ' % (min(EW_ref[good]),
                                                 max(EW_ref[good])))
        EW_int = interp1d(EW_ref[good], x[good], bounds_error=False,
                          fill_value=(-3.0, np.max(EW_ref[good])))

        NBmin = 20.0
        NBmax = m_NB[ff] - 0.25
        NB = np.arange(NBmin, NBmax + NBbin, NBbin)
        mylog.info('NB (min/max): %f %f ' % (min(NB), max(NB)))

        npz_NBfile = npz_path0 + filters[ff] + '_init.npz'

        if not exists(npz_NBfile) or redo:
            N_mag_mock = npz_slope['N_norm0'][ff] * Nsim * NBbin
            N_interp = interp1d(npz_slope['mag_arr'][ff], N_mag_mock)
            Ndist_mock = np.int_(np.round(N_interp(NB)))
            NB_ref = np.repeat(NB, Ndist_mock)

            Ngal = NB_ref.size  # Number of galaxies

            NB_sig = get_sigma(NB, m_NB[ff], sigma=3.0)
            NB_sig_ref = np.repeat(NB_sig, Ndist_mock)

            npz_NBdict = {}
            for name in npz_NBnames:
                npz_NBdict[name] = eval(name)

            if exists(npz_NBfile):
                mylog.info("Overwriting : " + npz_NBfile)
            else:
                mylog.info("Writing : " + npz_NBfile)
            np.savez(npz_NBfile, **npz_NBdict)
        else:
            if not redo:
                mylog.info("File found : " + npz_NBfile)
                npz_NB = np.load(npz_NBfile)

                for key0 in npz_NB.keys():
                    cmd1 = key0 + " = npz_NB['" + key0 + "']"
                    exec (cmd1)

        mock_sz = (Nmock, Ngal)

        # Randomize NB magnitudes. First get relative sigma, then scale by size
        NB_seed = ff
        mylog.info("seed for %s : %i" % (filters[ff], NB_seed))
        NB_MC = random_mags(NB_seed, mock_sz, NB_ref, NB_sig_ref)
        stats_log(NB_MC, "NB_MC", mylog)

        # Read in mag vs mass extrapolation
        mass_int, std_mass_int = get_mag_vs_mass_interp(prefixes[ff])

        lum_dist = cosmo.luminosity_distance(z_NB[ff]).to(u.cm).value

        # Read in EW and fluxes for H-alpha NB emitter sample
        npz_NB_file = path0 + 'Completeness/ew_flux_Ha-' + filters[ff] + '.npz'
        npz_NB = np.load(npz_NB_file)
        NB_EW = npz_NB['NB_EW']
        Ha_Flux = npz_NB['Ha_Flux']

        NBmag = npz_NB['NBmag']
        contmag = npz_NB['contmag']
        logMstar = npz_NB['logMstar']
        Ha_SFR = npz_NB['Ha_SFR']  # metallicity-dependent observed SFR
        Ha_Lum = npz_NB['Ha_Lum']  # filter and [NII] corrected

        spec_flag = npz_NB['spec_flag']
        w_spec = np.where(spec_flag)[0]
        wo_spec = np.where(spec_flag == 0)[0]

        # Statistics for comparisons
        avg_NB = np.average(NB_EW)
        sig_NB = np.std(NB_EW)

        avg_NB_flux = np.average(Ha_Flux)
        sig_NB_flux = np.std(Ha_Flux)

        # Plot sigma and average
        fig3, ax3 = avg_sig_plot_init(filters[ff], logEW_mean, avg_NB, sig_NB,
                                      avg_NB_flux, sig_NB_flux)
        ax3ul = ax3[0][0]
        ax3ll = ax3[1][0]
        ax3ur = ax3[0][1]
        ax3lr = ax3[1][1]

        chi2_EW0 = np.zeros((n_mean, n_sigma))
        chi2_Fl0 = np.zeros((n_mean, n_sigma))

        count = 0
        for mm in mm_range:  # loop over median of EW dist
            comp_EWmean[mm] = logEW_mean[mm]
            for ss in ss_range:  # loop over sigma of EW dist
                comp_EWsig[mm, ss] = logEW_sig[ss]

                npz_MCfile = npz_path0 + filters[ff] + ('_%.2f_%.2f.npz') % (logEW_mean[mm],
                                                                             logEW_sig[ss])

                fig, ax = plt.subplots(ncols=2, nrows=3)
                [[ax00, ax01], [ax10, ax11], [ax20, ax21]] = ax

                plt.subplots_adjust(left=0.105, right=0.98, bottom=0.05,
                                    top=0.98, wspace=0.25, hspace=0.05)

                # This is for statistics plot
                if count % nrow_stats == 0:
                    fig2, ax2 = plt.subplots(ncols=2, nrows=nrow_stats)
                s_row = count % nrow_stats  # For statistics plot

                if not exists(npz_MCfile) or redo:
                    EW_seed = mm * len(ss_range) + ss
                    mylog.info("seed for mm=%i ss=%i : %i" % (mm, ss, EW_seed))
                    np.random.seed(EW_seed)
                    rand0 = np.random.normal(0.0, 1.0, size=Ngal)
                    # This is not H-alpha
                    logEW_MC_ref = logEW_mean[mm] + logEW_sig[ss] * rand0
                    stats_log(logEW_MC_ref, "logEW_MC_ref", mylog)

                    x_MC0_ref = EW_int(logEW_MC_ref)  # NB color excess
                    negs = np.where(x_MC0_ref < 0)
                    if len(negs[0]) > 0:
                        x_MC0_ref[negs] = 0.0
                    stats_log(x_MC0_ref, "x_MC0_ref", mylog)

                    # Selection based on 'true' magnitudes
                    NB_sel_ref, NB_nosel_ref, sig_limit_ref = NB_select(ff, NB_ref, x_MC0_ref)

                    EW_flag_ref = np.zeros(Ngal)
                    EW_flag_ref[NB_sel_ref] = 1

                    BB_MC0_ref = NB_ref + x_MC0_ref
                    BB_sig_ref = get_sigma(BB_MC0_ref, cont_lim[ff], sigma=3.0)

                    dict_prop = dict_prop_maker(NB_ref, BB_MC0_ref, x_MC0_ref,
                                                filt_dict, filt_corr[ff], mass_int,
                                                lum_dist)
                    _, flux_ref, logM_ref, NIIHa_ref, logOH_ref, HaFlux_ref, \
                        HaLum_ref, logSFR_ref = derived_properties(**dict_prop)

                    if exists(npz_MCfile):
                        mylog.info("Overwriting : " + npz_MCfile)
                    else:
                        mylog.info("Writing : " + npz_MCfile)

                    npz_MCdict = {}
                    for name in npz_MCnames:
                        npz_MCdict[name] = eval(name)
                    np.savez(npz_MCfile, **npz_MCdict)
                else:
                    if not redo:
                        mylog.info("File found : " + npz_MCfile)
                        npz_MC = np.load(npz_MCfile)

                        for key0 in npz_MC.keys():
                            cmd1 = key0 + " = npz_MC['" + key0 + "']"
                            exec (cmd1)

                        dict_prop = dict_prop_maker(NB_ref, BB_MC0_ref, x_MC0_ref,
                                                    filt_dict, filt_corr[ff], mass_int,
                                                    lum_dist)

                BB_seed = ff + 5
                mylog.info("seed for broadband, mm=%i ss=%i : %i" % (mm, ss, BB_seed))
                BB_MC = random_mags(BB_seed, mock_sz, BB_MC0_ref, BB_sig_ref)
                stats_log(BB_MC, "BB_MC", mylog)

                x_MC = BB_MC - NB_MC
                stats_log(x_MC, "x_MC", mylog)

                NB_sel, NB_nosel, sig_limit = NB_select(ff, NB_MC, x_MC)

                EW_flag0 = np.zeros(mock_sz)
                EW_flag0[NB_sel[0], NB_sel[1]] = 1

                # Not sure if we should use true logEW or the mocked values
                # logEW_MC = mock_ones(logEW_MC_ref, Nmock)

                dict_prop['NB'] = NB_MC
                dict_prop['BB'] = BB_MC
                dict_prop['x'] = x_MC
                logEW_MC, flux_MC, logM_MC, NIIHa, logOH, HaFlux_MC, HaLum_MC, \
                    logSFR_MC = derived_properties(std_mass_int=std_mass_int,
                                                   **dict_prop)
                stats_log(logEW_MC, "logEW_MC", mylog)
                stats_log(flux_MC, "flux_MC", mylog)
                stats_log(HaFlux_MC, "HaFlux_MC", mylog)

                # Panel (0,0) - NB excess selection plot

                plot_mock(ax00, NB_MC, x_MC, NB_sel, NB_nosel, '', cont0[ff] + ' - ' + filters[ff])

                ax00.axvline(m_NB[ff], linestyle='dashed', color='b')

                temp_x = contmag - NBmag
                plot_MACT(ax00, NBmag, temp_x, w_spec, wo_spec)

                NB_break = plot_NB_select(ff, ax00, NB, 'b')

                N_annot_txt = avg_sig_label('', logEW_mean[mm], logEW_sig[ss],
                                            panel_type='EW')
                N_annot_txt += '\n' + r'$N$ = %i' % NB_MC.size
                ax00.annotate(N_annot_txt, [0.05, 0.95], va='top',
                              ha='left', xycoords='axes fraction')

                # Plot cropped version
                fig0, ax0 = plt.subplots()
                plt.subplots_adjust(left=0.1, right=0.98, bottom=0.10,
                                    top=0.98, wspace=0.25, hspace=0.05)

                plot_mock(ax0, NB_MC, x_MC, NB_sel, NB_nosel, filters[ff], cont0[ff] + ' - ' + filters[ff])
                ax0.axvline(m_NB[ff], linestyle='dashed', color='b')

                temp_x = contmag - NBmag
                plot_MACT(ax0, NBmag, temp_x, w_spec, wo_spec)

                plot_NB_select(ff, ax0, NB, 'b', plot4=False)

                N_annot_txt = avg_sig_label('', logEW_mean[mm], logEW_sig[ss],
                                            panel_type='EW')
                N_annot_txt += '\n' + r'$N$ = %i' % NB_MC.size
                ax0.annotate(N_annot_txt, [0.025, 0.975], va='top',
                             ha='left', xycoords='axes fraction')
                fig0.savefig(pp0, format='pdf')

                # Panel (1,0) - NB mag vs H-alpha flux
                plot_mock(ax10, NB_MC, HaFlux_MC, NB_sel, NB_nosel, filters[ff],
                          Flux_lab)

                plot_MACT(ax10, NBmag, Ha_Flux, w_spec, wo_spec)

                # Panel (0,1) - stellar mass vs H-alpha luminosity

                plot_mock(ax01, logM_MC, HaLum_MC, NB_sel, NB_nosel, '',
                          r'$\log(L_{{\rm H}\alpha})$')

                plot_MACT(ax01, logMstar, Ha_Lum, w_spec, wo_spec)

                # Panel (1,1) - stellar mass vs H-alpha SFR

                plot_mock(ax11, logM_MC, logSFR_MC, NB_sel, NB_nosel, M_lab, SFR_lab)

                plot_MACT(ax11, logMstar, Ha_SFR, w_spec, wo_spec)

                # Plot cropped version
                fig0, ax0 = plt.subplots()
                plt.subplots_adjust(left=0.1, right=0.98, bottom=0.10,
                                    top=0.98, wspace=0.25, hspace=0.05)

                plot_mock(ax0, logM_MC, logSFR_MC, NB_sel, NB_nosel, M_lab, SFR_lab)

                plot_MACT(ax0, logMstar, Ha_SFR, w_spec, wo_spec)
                # ax0.set_ylim([-5,-1])
                fig0.savefig(pp0, format='pdf')

                # Panel (2,0) - histogram of EW
                min_EW = compute_EW(minthres[ff], ff)
                mylog.info("minimum EW : %f " % min_EW)
                ax20.axvline(x=min_EW, color='red')

                No, Ng, binso, \
                    wht0 = ew_flux_hist('EW', mm, ss, ax20, NB_EW, avg_NB,
                                        sig_NB, EW_bins, logEW_mean, logEW_sig,
                                        EW_flag0, logEW_MC, ax3=ax3ul)
                ax20.set_position([0.085, 0.05, 0.44, 0.265])

                good = np.where(EW_flag0)[0]

                # Model comparison plots
                if len(good) > 0:
                    chi2 = stats_plot('EW', ax2, ax3ur, ax20, s_row, Ng, No,
                                      binso, logEW_mean[mm], logEW_sig[ss], ss)
                    chi2_EW0[mm, ss] = chi2

                # Panel (2,1) - histogram of H-alpha fluxes
                No, Ng, binso, \
                    wht0 = ew_flux_hist('Flux', mm, ss, ax21, Ha_Flux,
                                        avg_NB_flux, sig_NB_flux, Flux_bins,
                                        logEW_mean, logEW_sig,
                                        EW_flag0, HaFlux_MC, ax3=ax3ll)
                ax21.set_position([0.53, 0.05, 0.44, 0.265])

                ax21.legend(loc='upper right', fancybox=True, fontsize=6,
                            framealpha=0.75)

                # Model comparison plots
                if len(good) > 0:
                    chi2 = stats_plot('Flux', ax2, ax3lr, ax21, s_row, Ng, No,
                                      binso, logEW_mean[mm], logEW_sig[ss], ss)
                    chi2_Fl0[mm, ss] = chi2

                if s_row != nrow_stats - 1:
                    ax2[s_row][0].set_xticklabels([])
                    ax2[s_row][1].set_xticklabels([])
                else:
                    ax2[s_row][0].set_xlabel(EW_lab)
                    ax2[s_row][1].set_xlabel(Flux_lab)

                # Save each page after each model iteration
                fig.set_size_inches(8, 10)
                fig.savefig(pp, format='pdf')
                plt.close(fig)

                # Save figure for each full page completed
                if s_row == nrow_stats - 1 or count == len(mm_range) * len(ss_range) - 1:
                    fig2.subplots_adjust(left=0.1, right=0.97, bottom=0.08,
                                         top=0.97, wspace=0.13)

                    fig2.set_size_inches(8, 10)
                    fig2.savefig(pp2, format='pdf')
                    plt.close(fig2)
                count += 1

                # Compute and plot completeness
                # Combine over modelled galaxies
                comp_arr = np.sum(EW_flag0, axis=0) / float(Nmock)

                # Plot Type 1 and 2 errors
                cticks = np.arange(0, 1.2, 0.2)

                fig4, ax4 = plt.subplots(nrows=2, ncols=2)
                [[ax400, ax401], [ax410, ax411]] = ax4

                ax4ins0 = inset_axes(ax400, width="40%", height="15%", loc=3,
                                     bbox_to_anchor=(0.025, 0.1, 0.95, 0.25),
                                     bbox_transform=ax400.transAxes)  # LL
                ax4ins1 = inset_axes(ax400, width="40%", height="15%", loc=4,
                                     bbox_to_anchor=(0.025, 0.1, 0.95, 0.25),
                                     bbox_transform=ax400.transAxes)  # LR

                ax4ins0.xaxis.set_ticks_position("top")
                ax4ins1.xaxis.set_ticks_position("top")

                idx0 = [NB_sel_ref, NB_nosel_ref]
                cmap0 = [cmap_sel, cmap_nosel]
                lab0 = ['Type 1', 'Type 2']
                for idx, cmap, ins, lab in zip(idx0, cmap0, [ax4ins0, ax4ins1], lab0):
                    cs = ax400.scatter(NB_ref[idx], x_MC0_ref[idx], edgecolor='none',
                                       vmin=0, vmax=1.0, s=15, c=comp_arr[idx],
                                       cmap=cmap)
                    cb = fig4.colorbar(cs, cax=ins, orientation="horizontal",
                                       ticks=cticks)
                    cb.ax.tick_params(labelsize=8)
                    cb.set_label(lab)

                plot_NB_select(ff, ax400, NB, 'k', linewidth=2)

                ax400.set_xlabel(filters[ff])
                ax400.set_ylim([-0.5, 2.0])
                ax400.set_ylabel(cont0[ff] + ' - ' + filters[ff])

                ax400.annotate(N_annot_txt, [0.025, 0.975], va='top',
                               ha='left', xycoords='axes fraction')

                logsSFR_ref = logSFR_ref - logM_ref
                logsSFR_MC = logSFR_MC - logM_MC

                above_break = np.where(NB_MC <= NB_break)

                t_comp_sSFR, \
                    t_comp_sSFR_ref = plot_completeness(ax401, logsSFR_MC, NB_sel, sSFR_bins,
                                                        ref_arr0=logsSFR_ref,
                                                        above_break=above_break)

                '''t_comp_EW, \
                    t_comp_EW_ref = plot_completeness(ax410, logEW_MC, NB_sel,
                                                      EW_bins, ref_arr0=logEW_MC_ref)
                '''
                t_comp_Fl, \
                    t_comp_Fl_ref = plot_completeness(ax410, HaFlux_MC, NB_sel,
                                                      Flux_bins, ref_arr0=HaFlux_ref)

                t_comp_SFR, \
                    t_comp_SFR_ref = plot_completeness(ax411, logSFR_MC, NB_sel,
                                                       SFR_bins, ref_arr0=logSFR_ref)
                comp_sSFR[mm, ss] = t_comp_sSFR
                comp_SFR[mm, ss] = t_comp_SFR
                comp_flux[mm, ss] = t_comp_Fl

                fig0, ax0 = plt.subplots()
                plt.subplots_adjust(left=0.1, right=0.97, bottom=0.10,
                                    top=0.98, wspace=0.25, hspace=0.05)
                t_comp_SFR = plot_completeness(ax0, logSFR_MC, NB_sel, SFR_bins,
                                               ref_arr0=logSFR_ref, annotate=False)
                ax0.set_ylabel('Completeness')
                ax0.set_xlabel(SFR_lab)
                ax0.set_ylim([0.0, 1.05])
                fig0.savefig(pp0, format='pdf')

                xlabels = [r'$\log({\rm sSFR})$', Flux_lab, SFR_lab]
                for t_ax, xlabel in zip([ax401, ax410, ax411], xlabels):
                    t_ax.set_ylabel('Completeness')
                    t_ax.set_xlabel(xlabel)
                    t_ax.set_ylim([0.0, 1.05])

                # ax410.axvline(x=compute_EW(minthres[ff], ff), color='red')

                plt.subplots_adjust(left=0.09, right=0.98, bottom=0.065,
                                    top=0.98, wspace=0.20, hspace=0.15)
                fig4.set_size_inches(8, 8)
                fig4.savefig(pp4, format='pdf')

                # Plot sSFR vs stellar mass
                fig5, ax5 = plt.subplots()
                plot_mock(ax5, logM_MC, logSFR_MC - logM_MC, NB_sel, NB_nosel, M_lab,
                          r'$\log({\rm sSFR})$')
                plt.subplots_adjust(left=0.09, right=0.98, bottom=0.1, top=0.98)
                fig5.set_size_inches(8, 8)
                fig5.savefig(pp4, format='pdf')

        pp.close()
        pp0.close()
        pp2.close()
        pp4.close()

        ax3ul.legend(loc='upper right', title=r'$\sigma[\log({\rm EW})]$',
                     fancybox=True, fontsize=8, framealpha=0.75, scatterpoints=1)

        # Compute best fit using weighted chi^2
        chi2_wht = np.sqrt(chi2_EW0 ** 2 / 2 + chi2_Fl0 ** 2 / 2)
        b_chi2 = np.where(chi2_wht == np.min(chi2_wht))
        mylog.info("Best chi2 : " + str(b_chi2))
        mylog.info("Best chi2 : (%s, %s) " % (logEW_mean[b_chi2[0]][0],
                                              logEW_sig[b_chi2[1]][0]))
        ax3ur.scatter(logEW_mean[b_chi2[0]] + 0.005 * (b_chi2[1] - 3 / 2.),
                      chi2_EW0[b_chi2], edgecolor='k', facecolor='none',
                      s=100, linewidth=2)
        ax3lr.scatter(logEW_mean[b_chi2[0]] + 0.005 * (b_chi2[1] - 3 / 2.),
                      chi2_Fl0[b_chi2], edgecolor='k', facecolor='none',
                      s=100, linewidth=2)

        fig3.set_size_inches(8, 8)
        fig3.subplots_adjust(left=0.105, right=0.97, bottom=0.065, top=0.98,
                             wspace=0.25, hspace=0.01)

        out_pdf3_each = path0 + 'Completeness/ew_MC_' + filters[ff] + '.avg_sigma.pdf'
        if debug:
            out_pdf3_each = out_pdf3_each.replace('.pdf', '.debug.pdf')
        fig3.savefig(out_pdf3_each, format='pdf')

        if not debug:
            fig3.savefig(pp3, format='pdf')
        plt.close(fig3)

        table_outfile = path0 + 'Completeness/' + filters[ff] + '_completeness_50.tbl'
        c_size = comp_shape[0] * comp_shape[1]
        comp_arr0 = [comp_EWmean.reshape(c_size), comp_EWsig.reshape(c_size),
                     comp_sSFR.reshape(c_size), comp_SFR.reshape(c_size),
                     comp_flux.reshape(c_size)]
        c_names = ('log_EWmean', 'log_EWsig', 'comp_50_sSFR', 'comp_50_SFR',
                   'comp_50_flux')

        mylog.info("Writing : " + table_outfile)
        comp_tab = Table(comp_arr0, names=c_names)
        comp_tab.write(table_outfile, format='ascii.fixed_width_two_line',
                       overwrite=True)

        # Generate table containing best fit results
        if not debug:
            best_tab0 = comp_tab[b_chi2[0] * len(ss_range) + b_chi2[1]]
            if ff == 0:
                comp_tab0 = best_tab0
            else:
                comp_tab0 = vstack([comp_tab0, best_tab0])

        t_ff._stop()
        mylog.info("ew_MC completed for " + filters[ff] + " in : " + t_ff.format)

    if not debug:
        table_outfile0 = path0 + 'Completeness/best_fit_completeness_50.tbl'
        comp_tab0.write(table_outfile0, format='ascii.fixed_width_two_line',
                        overwrite=True)

    if not debug:
        pp3.close()

    t0._stop()
    mylog.info("ew_MC completed in : " + t0.format)


'''
THIS IS CODE THAT WAS NOT USED FOR CROPPING PDF.  DECIDED TO GENERATE NEW PLOTSX
def crop_pdf(infile, outfile, pp_page):
    with open(infile, "rb") as in_f:
        input1 = PdfFileReader(in_f)
        output = PdfFileWriter()

        numPages = input1.getNumPages()
        print("document has %s pages." % numPages)

        page = input1.getPage(pp_page)
        print(page.mediaBox.getUpperRight_x(), page.mediaBox.getUpperRight_y())
        page.trimBox.lowerLeft = (20, 25)
        page.trimBox.upperRight = (225, 225)
        page.cropBox.lowerLeft = (50, 50)
        page.cropBox.upperRight = (200, 200)
        output.addPage(page)

    with open(outfile, "wb") as out_f:
        output.write(out_f)
#enddef
'''
