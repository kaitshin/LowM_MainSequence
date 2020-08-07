"""
completeness_analysis
====

A set of Python 2.7 codes for completeness analysis of NB-selected galaxies
in the M*-SFR plot
"""

import os

from chun_codes import TimerClass

from os.path import exists

from astropy.table import Table, vstack

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.interpolate import interp1d

from . import MLog, get_date  # for logging
from . import filter_dict  # For EW and flux calculations

# Variable definitions
from .config import filters, cont0  # Filter name and corresponding broad-band for NB color excess plot
from .config import prefixes, z_NB  # Prefix for mag-to-mass interpolation files
from .config import logEW_mean_start, logEW_sig_start, n_mean, n_sigma  # Grid definition for log-normal distribution
from .config import NB_bin  # Bin size for NB magnitude
from . import cmap_sel, cmap_nosel
from . import EW_lab, Flux_lab, M_lab, SFR_lab
from . import EW_bins, Flux_bins, sSFR_bins, SFR_bins
from .config import m_NB, cont_lim, minthres
from .config import path0, npz_path0


# Import separate functions
from .config import pdf_filename
from .stats import stats_log, avg_sig_label, stats_plot
from .monte_carlo import random_mags
from .monte_carlo import main as mc_main
from .select import color_cut, get_EW
from .dataset import get_mact_data
from .plotting import avg_sig_plot_init, plot_MACT, plot_mock, plot_completeness
from .plotting import overlay_mock_average_dispersion, plot_dispersion, ew_flux_hist
from .properties import get_mag_vs_mass_interp, compute_EW
from .normalization import get_normalization

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)

if not exists(npz_path0):
    os.mkdir(npz_path0)


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

    str_date = get_date(debug=debug)
    mylog = MLog(path0 + 'Completeness/', str_date)._get_logger()

    t0 = TimerClass()
    t0._start()

    mylog.info('Nsim : ', Nsim)

    nrow_stats = 4

    # One file written for all avg and sigma comparisons
    if not debug:
        out_pdf3 = path0 + 'Completeness/ew_MC.avg_sigma.pdf'
        pp3 = PdfPages(out_pdf3)

    ff_range = [0] if debug else range(len(filter_dict['filt_ref']))
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

        pdf_dict = pdf_filename(ff, debug=debug)

        pp = PdfPages(pdf_dict['main'])    # Main plots
        pp0 = PdfPages(pdf_dict['crop'])   # Contains cropped plots
        pp2 = PdfPages(pdf_dict['stats'])  # This contains stats plots
        pp4 = PdfPages(pdf_dict['comp'])   # Completeness plots

        filt_dict = {'dNB': filter_dict['dNB'][ff],
                     'dBB': filter_dict['dBB'][ff],
                     'lambdac': filter_dict['lambdac'][ff]}

        # Retrieve EW interpolated grid
        EW_int = get_EW(ff, mylog)

        NBmin = 20.0
        NBmax = m_NB[ff] - 0.25
        NB = np.arange(NBmin, NBmax + NB_bin, NB_bin)
        mylog.info('NB (min/max): %f %f ' % (min(NB), max(NB)))

        # Get number distribution for normalization
        norm_dict = get_normalization(ff, Nmock, NB, Nsim, NB_bin, mylog, redo=redo)

        mock_sz = (Nmock, norm_dict['Ngal'])

        # Randomize NB magnitudes. First get relative sigma, then scale by size
        NB_seed = ff
        mylog.info("seed for %s : %i" % (filters[ff], NB_seed))
        NB_MC = random_mags(NB_seed, mock_sz, norm_dict['NB_ref'],
                            norm_dict['NB_sig_ref'])
        stats_log(NB_MC, "NB_MC", mylog)

        # Read in mag vs mass extrapolation
        mass_int, std_mass_int = get_mag_vs_mass_interp(prefixes[ff])

        lum_dist = cosmo.luminosity_distance(z_NB[ff]).to(u.cm).value

        # Read in EW and fluxes for H-alpha NB emitter sample
        dict_NB = get_mact_data(ff)

        # Statistics for comparisons
        avg_NB = np.average(dict_NB['NB_EW'])
        sig_NB = np.std(dict_NB['NB_EW'])

        avg_NB_flux = np.average(dict_NB['Ha_Flux'])
        sig_NB_flux = np.std(dict_NB['Ha_Flux'])

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

                int_dict = {'ff': ff, 'mm': mm, 'ss': ss}
                mass_dict = {'mass_int': mass_int, 'std_mass_int': std_mass_int}
                EW_dict = {'logEW_mean': logEW_mean, 'logEW_sig': logEW_sig,
                           'EW_int': EW_int}

                dict_phot_ref, der_prop_dict_ref, npz_MCdict, \
                    dict_MC = mc_main(int_dict, npz_MCfile, mock_sz, ss_range,
                                      mass_dict, norm_dict, filt_dict, EW_dict,
                                      NB_MC, lum_dist, mylog, redo=redo)

                # Panel (0,0) - NB excess selection plot
                NB_limit = [20.0, max(NB)+1]

                ylabel_excess = cont0[ff] + ' - ' + filters[ff]
                plot_mock(ax00, dict_MC, 'NB', 'x', x_limit=NB_limit,
                          ylabel=ylabel_excess)

                ax00.axvline(m_NB[ff], linestyle='dashed', color='b')

                temp_x = dict_NB['contmag'] - dict_NB['NBmag']
                plot_MACT(ax00, dict_NB, 'NBmag', temp_x)

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

                plot_mock(ax0, dict_MC, 'NB', 'x', x_limit=NB_limit,
                          xlabel=filters[ff], ylabel=ylabel_excess)
                ax0.axvline(m_NB[ff], linestyle='dashed', color='b')

                temp_x = dict_NB['contmag'] - dict_NB['NBmag']
                plot_MACT(ax0, dict_NB, 'NBmag', temp_x)

                plot_NB_select(ff, ax0, NB, 'b', plot4=False)

                N_annot_txt = avg_sig_label('', logEW_mean[mm], logEW_sig[ss],
                                            panel_type='EW')
                N_annot_txt += '\n' + r'$N$ = %i' % NB_MC.size
                ax0.annotate(N_annot_txt, [0.025, 0.975], va='top',
                             ha='left', xycoords='axes fraction')
                fig0.savefig(pp0, format='pdf')

                # Panel (1,0) - NB mag vs H-alpha flux
                plot_mock(ax10, dict_MC, 'NB', 'Ha_Flux', x_limit=NB_limit,
                          xlabel=filters[ff], ylabel=Flux_lab)

                plot_MACT(ax10, dict_NB, 'NBmag', 'Ha_Flux')

                # Panel (0,1) - stellar mass vs H-alpha luminosity

                plot_mock(ax01, dict_MC, 'logM', 'Ha_Lum',
                          ylabel=r'$\log(L_{{\rm H}\alpha})$')

                plot_MACT(ax01, dict_NB, 'logMstar', 'Ha_Lum')

                # Panel (1,1) - stellar mass vs H-alpha SFR

                plot_mock(ax11, dict_MC, 'logM', 'logSFR',
                          xlabel=M_lab, ylabel=SFR_lab)

                plot_MACT(ax11, dict_NB, 'logMstar', 'Ha_SFR')

                # Plot cropped version
                fig0, ax0 = plt.subplots()
                plt.subplots_adjust(left=0.1, right=0.98, bottom=0.10,
                                    top=0.98, wspace=0.25, hspace=0.05)

                plot_mock(ax0, dict_MC, 'logM', 'logSFR', xlabel=M_lab,
                          ylabel=SFR_lab)

                plot_MACT(ax0, dict_NB, 'logMstar', 'Ha_SFR')
                # ax0.set_ylim([-5,-1])
                fig0.savefig(pp0, format='pdf')

                # Panel (2,0) - histogram of EW
                min_EW = compute_EW(minthres[ff], ff)
                mylog.info("minimum EW : %f " % min_EW)
                ax20.axvline(x=min_EW, color='red')

                No, Ng, binso, \
                    wht0 = ew_flux_hist('EW', mm, ss, ax20, dict_NB['NB_EW'], avg_NB,
                                        sig_NB, EW_bins, logEW_mean, logEW_sig,
                                        dict_MC['EW_flag0'], dict_MC['logEW'], ax3=ax3ul)
                ax20.set_position([0.085, 0.05, 0.44, 0.265])

                good = np.where(dict_MC['EW_flag0'])[0]

                # Model comparison plots
                if len(good) > 0:
                    chi2 = stats_plot('EW', ax2, ax3ur, ax20, s_row, Ng, No,
                                      binso, logEW_mean[mm], logEW_sig[ss], ss)
                    chi2_EW0[mm, ss] = chi2

                # Panel (2,1) - histogram of H-alpha fluxes
                No, Ng, binso, \
                    wht0 = ew_flux_hist('Flux', mm, ss, ax21, dict_NB['Ha_Flux'],
                                        avg_NB_flux, sig_NB_flux, Flux_bins,
                                        logEW_mean, logEW_sig,
                                        dict_MC['EW_flag0'], dict_MC['Ha_Flux'], ax3=ax3ll)
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
                comp_arr = np.sum(dict_MC['EW_flag0'], axis=0) / float(Nmock)

                # Plot Type 1 and 2 errors
                cticks = np.arange(0, 1.2, 0.2)

                fig4, ax4 = plt.subplots(nrows=2, ncols=2)
                [[ax400, ax401], [ax410, ax411]] = ax4

                plt.subplots_adjust(left=0.09, right=0.98, bottom=0.065,
                                    top=0.98, wspace=0.20, hspace=0.15)
                for t_ax in [ax400, ax401, ax410, ax411]:
                    t_ax.tick_params(axis='both', direction='in')

                ax4ins0 = inset_axes(ax400, width="40%", height="15%", loc=3,
                                     bbox_to_anchor=(0.025, 0.1, 0.95, 0.25),
                                     bbox_transform=ax400.transAxes)  # LL
                ax4ins1 = inset_axes(ax400, width="40%", height="15%", loc=4,
                                     bbox_to_anchor=(0.025, 0.1, 0.95, 0.25),
                                     bbox_transform=ax400.transAxes)  # LR

                ax4ins0.xaxis.set_ticks_position("top")
                ax4ins1.xaxis.set_ticks_position("top")

                idx0 = [npz_MCdict['NB_sel_ref'], npz_MCdict['NB_nosel_ref']]
                cmap0 = [cmap_sel, cmap_nosel]
                lab0 = ['Type 1', 'Type 2']
                for idx, cmap, ins, lab in zip(idx0, cmap0, [ax4ins0, ax4ins1], lab0):
                    cs = ax400.scatter(norm_dict['NB_ref'][idx], dict_phot_ref['x'][idx],
                                       edgecolor='none', vmin=0, vmax=1.0, s=15,
                                       c=comp_arr[idx], cmap=cmap)
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

                logsSFR_ref = der_prop_dict_ref['logSFR'] - der_prop_dict_ref['logM']
                logsSFR_MC = dict_MC['logSFR'] - dict_MC['logM']

                above_break = np.where(NB_MC <= NB_break)

                t_comp_sSFR, \
                    t_comp_sSFR_ref = plot_completeness(ax401, dict_MC, logsSFR_MC,
                                                        sSFR_bins, ref_arr0=logsSFR_ref,
                                                        above_break=above_break)

                t_comp_Fl, \
                    t_comp_Fl_ref = plot_completeness(ax410, dict_MC, 'Ha_Flux',
                                                      Flux_bins, ref_arr0=der_prop_dict_ref['Ha_Flux'])

                t_comp_SFR, \
                    t_comp_SFR_ref = plot_completeness(ax411, dict_MC, 'logSFR',
                                                       SFR_bins, ref_arr0=der_prop_dict_ref['logSFR'])
                comp_sSFR[mm, ss] = t_comp_sSFR
                comp_SFR[mm, ss] = t_comp_SFR
                comp_flux[mm, ss] = t_comp_Fl

                fig0, ax0 = plt.subplots()
                plt.subplots_adjust(left=0.1, right=0.97, bottom=0.10,
                                    top=0.98, wspace=0.25, hspace=0.05)
                plot_completeness(ax0, dict_MC, 'logSFR', SFR_bins, annotate=False,
                                  ref_arr0=der_prop_dict_ref['logSFR'])

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

                fig4.set_size_inches(8, 8)
                fig4.savefig(pp4, format='pdf')

                # Plot SFR/sSFR vs stellar mass and dispersion
                fig5, ax5 = plt.subplots(nrows=2, ncols=2)

                # SFR vs stellar mass
                plot_mock(ax5[0][0], dict_MC, 'logM', 'logSFR', ylabel=SFR_lab)
                ax5[0][0].set_xticklabels([])
                ax5[0][0].tick_params(axis='both', direction='in')

                SFR_bin_MC = overlay_mock_average_dispersion(ax5[0][0], dict_MC,
                                                             'logM', 'logSFR')
                SFR_bin_MCfile = npz_MCfile.replace(filters[ff], filters[ff]+'_SFR_bin')
                mylog.info("Writing : " + SFR_bin_MCfile)
                np.savez(SFR_bin_MCfile, **SFR_bin_MC)

                plot_MACT(ax5[0][0], dict_NB, 'logMstar', 'Ha_SFR', size=15)

                # Dispersion: SFR vs stellar mass
                plot_dispersion(ax5[1][0], SFR_bin_MC)
                ax5[1][0].tick_params(axis='both', direction='in')

                # sSFR vs stellar mass
                plot_mock(ax5[0][1], dict_MC, 'logM', logsSFR_MC, ylabel=r'$\log({\rm sSFR})$')
                ax5[0][1].set_xticklabels([])
                ax5[0][1].tick_params(axis='both', direction='in')

                sSFR_bin_MC = overlay_mock_average_dispersion(ax5[0][1], dict_MC,
                                                              'logM', logsSFR_MC)
                sSFR_bin_MCfile = npz_MCfile.replace(filters[ff], filters[ff]+'_sSFR_bin')
                mylog.info("Writing : " + sSFR_bin_MCfile)
                np.savez(sSFR_bin_MCfile, **sSFR_bin_MC)

                logsSFR = dict_NB['Ha_SFR'] - dict_NB['logMstar']
                plot_MACT(ax5[0][1], dict_NB, 'logMstar', logsSFR, size=15)

                # Dispersion: sSFR vs stellar mass
                plot_dispersion(ax5[1][1], sSFR_bin_MC)
                ax5[1][1].tick_params(axis='both', direction='in')

                plt.subplots_adjust(left=0.07, right=0.98, bottom=0.05, top=0.98,
                                    hspace=0.025)
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
        if debug:
            table_outfile = table_outfile.replace('.tbl', '.debug.tbl')
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
