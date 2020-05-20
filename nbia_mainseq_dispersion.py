"""
NAME:
    nbia_mainseq_dispersion.py

PURPOSE:
    Generates main sequence dispersion plots for the MACT dataset.

    Depends on mainseq_corrections.py.

    Contains functions that are called by plot_mact_with_newha.py.

INPUTS:
    config.FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt'
    config.FULL_PATH+'Main_Sequence/Noeske07_fig1_z1.txt'

OUTPUTS:
    config.FULL_PATH+'Plots/main_sequence/mainseq_dispersion.pdf'
    config.FULL_PATH+'Main_Sequence/dispersion_tbl.txt'
"""
from __future__ import print_function

import numpy as np, matplotlib.pyplot as plt
import scipy.optimize as optimize
from astropy.io import ascii as asc
from astropy.table import Table

import config
from MACT_utils import get_mainseq_fit_params, get_filt_index, get_z_arr, approximated_zspec0, get_func0_eqn0


def add_legends(ax, withnewha):
    '''
    adds two legends to the plot
    '''
    from matplotlib.patches import Patch

    # first legend
    legend1 = ax.legend(loc='upper left', frameon=False)
    ax.add_artist(legend1)

    # second legend
    noeske, = ax.plot(-100, 100, color='orange', marker='+',
        label='Noeske+07 (0.20<z<0.40)',mew=2,markersize=11)
    salim = Patch(facecolor='gray', edgecolor='None', alpha=0.4,
        label='Salim+07 (z~0)')

    if not withnewha:
        delosreyes = ax.scatter(-100, 100, color='deepskyblue', marker='s',
            label='de los Reyes+15 (z~0.8)', zorder=2)
        labelarr2 = np.array([delosreyes, noeske, salim])
    else:
        labelarr2 = np.array([noeske, salim])

    legend2 = ax.legend(handles=list(labelarr2), loc='lower right',
        frameon=False, fontsize=11, scatterpoints=1, numpoints=1)
    ax.add_artist(legend2)


def create_disp_tbl(smass0, sfrs00, sfrs00_nofuv, sfrs_resid, meas_errs):
    '''
    creates & returns a dispersion table with (1) stlrmass bins,
    (2) avg sfr (avg sfr w/o FUV corr),
    (3) observed dispersion (per bin), (4) systematic dispersion, and
    (5) the intrinsic dispersion (obtained by subtracting (4) from (3)
    in quadrature)
    '''
    stlrmass_bins = []
    avg_sfr = []
    observed_disp = []
    systematic_disp = []
    intrinsic_disp = []

    # defining mass bins
    mbins0 = np.arange(6.25, 10.75, .5)
    bin_ii = np.digitize(smass0, mbins0+0.25)

    for i in range(len(mbins0)):
        bin_match = np.where(bin_ii == i)[0]
        
        mass_str = str(mbins0[i]-0.25)+'--'+str(mbins0[i]+0.25)
        stlrmass_bins.append(mass_str)
        
        avgsfr = np.mean(sfrs00[bin_match])
        sfr_entry = ''
        if avgsfr < 0:
            sfr_entry += '-%.3f'%avgsfr
        else:
            sfr_entry += '%.3f'%avgsfr

        avgsfr_nofuv = np.mean(sfrs00_nofuv[bin_match])
        if avgsfr_nofuv < 0:
            sfr_entry += ' (-%.3f)'%avgsfr_nofuv
        else:
            sfr_entry += ' (%.3f)'%avgsfr_nofuv

        avg_sfr.append(sfr_entry)
        
        obs_disp = np.std(sfrs_resid[bin_match])
        observed_disp.append('%.3f'%obs_disp)
        
        syst_disp = np.mean(meas_errs[bin_match])
        systematic_disp.append('%.3f'%syst_disp)
        
        intr_disp = np.sqrt(obs_disp**2 - syst_disp**2)
        if np.isnan(intr_disp):
            intrinsic_disp.append('\\ldots')
        else:
            intrinsic_disp.append('%.3f'%intr_disp)

    tt = Table([stlrmass_bins, avg_sfr, observed_disp, systematic_disp,
        intrinsic_disp], names=['(1)','(2)','(3)','(4)','(5)'])
    
    return tt


def delosreyes_2015(ax):
    '''
    Plots the residuals of the z~0.8 data points from de Los Reyes+15 in cyan.
    '''
    def delosreyes_fit(mass):
        return 0.75*mass - 6.73

    dlr_xarr = np.array([9.27, 9.52, 9.76, 10.01, 10.29, 10.59, 10.81, 11.15])
    dlr_yarr = np.array([0.06, 0.27, 0.43, 0.83, 1.05, 1.18, 1.50, 1.54])
    dlr_yerr = np.array([0.454, 0.313, 0.373, 0.329, 0.419, 0.379, 0.337,
        0.424])
    ax.errorbar(dlr_xarr, dlr_yarr - delosreyes_fit(dlr_xarr), dlr_yerr,
        fmt='none', ecolor='deepskyblue', zorder=2) 
    ax.scatter(dlr_xarr, dlr_yarr - delosreyes_fit(dlr_xarr),
        color='deepskyblue', marker='s', zorder=2)


def noeske_2007(ax):
    '''
    Plots the residuals of the data points from Noeske+07 in orange. 
    (ASCII file provided by Chun Ly)
    '''
    def line(mass, a, b):
        return a*mass + b

    noeske = asc.read(config.FULL_PATH+'Main_Sequence/Noeske07_fig1_z1.txt',
        guess=False, Reader=asc.NoHeader)
    logM   = np.array(noeske['col1'])
    logSFR = np.array(noeske['col2'])
    logSFR_low  = np.array(noeske['col3'])
    logSFR_high = np.array(noeske['col4'])

    params, pcov = optimize.curve_fit(line, logM, logSFR)
    ax.fill_between(logM, logSFR_low-line(logM, *params),
        logSFR_high-line(logM, *params), facecolor='none', hatch=3*'.',
        edgecolor='orange', linewidth=0.0, zorder=1)
    ax.plot(logM, logSFR - line(logM, *params), color='orange', marker='+',
        lw=0, mew=2, markersize=11)


def plot_avg_resids(ax, smass0, sfrs_resid, withnewha):
    '''
    plots the average residuals in each 0.5M mass bin
    '''
    # defining mass bins
    if withnewha:
        mbins0 = np.arange(6.25, 12.25, .5)
    else:
        mbins0 = np.arange(6.25, 10.75, .5)
    bin_ii = np.digitize(smass0, mbins0+0.25)

    for i in range(len(mbins0)):
        bin_match = np.where(bin_ii == i)[0]
        ax.plot(mbins0[i], np.mean(sfrs_resid[bin_match]), 'ko')
        ax.errorbar(mbins0[i], np.mean(sfrs_resid[bin_match]), xerr=0.25,
            yerr=np.std(sfrs_resid[bin_match]), fmt='none', color='k')
        # print('sigma =',str(np.round(np.std(sfrs_resid[bin_match]), 4)))


def plot_resids(ax, markarr, sizearr, z_arr, no_spectra, yes_spectra, smass0,
    sfrs_resid, filts00, ffarr, llarr):
    '''
    plots residuals of the ha galaxies w/ good sigma,mass params
    same scheme as plot_nbia_mainseq.py

    residuals are
        data - model
    where the model is described in the function func0
    '''
    check_nums = []
    for ff,mm,ll,size,avg_z in zip(ffarr, markarr, llarr, sizearr, z_arr):
        filt_index_n = get_filt_index(no_spectra, ff, filts00)
        filt_index_y = get_filt_index(yes_spectra, ff, filts00)

        check_nums.append(len(filt_index_y)+len(filt_index_n))

        ax.scatter(smass0[yes_spectra][filt_index_y],
            sfrs_resid[yes_spectra][filt_index_y], marker=mm,
            facecolors='blue', edgecolors='none', alpha=0.2,
            label='z~'+np.str(avg_z)+' ('+ll+')', s=size)

        if ff != 'NEWHA':
            ax.scatter(smass0[no_spectra][filt_index_n], 
                sfrs_resid[no_spectra][filt_index_n], marker=mm,
                facecolors='none', edgecolors='blue', alpha=0.2, 
                linewidth=0.5, zorder=3, s=size)

    assert np.sum(check_nums)==len(smass0)


def salim_2007(ax):
    '''
    Plots the residuals of the log(M*)-log(SFR) relation from Salim+07
    in black.
    '''
    xarr = np.arange(8.5, 11.2, 0.01)
    ax.fill_between(xarr, -np.array([0.2]*len(xarr)),
        np.array([0.2]*len(xarr)), color='gray', alpha=0.4)


def plot_all_dispersion(f, ax, data00, corr_sfrs, delta_sfrs, stlr_mass, filts,
    no_spectra, yes_spectra, z_arr, 
    markarr = np.array(['o','^','D','*']), 
    sizearr = np.array([6.0,6.0,6.0,9.0])**2,
    ffarr=['NB7', 'NB816', 'NB921', 'NB973'],
    llarr=['NB704,NB711', 'NB816', 'NB921', 'NB973'],
    ytype='SFR', fittype = 'first_order', withnewha=False):
    '''
    Makes the dispersion figure by calling plot_resides(), plot_avg_resids(),
    and other functions to add final touches (add_legends(), salim_2007(),
    noeske_2007(), and possibly delosreyes_2015()). Shapes are iterated
    through by filter (proxy for redshift) for each source.

    The appropriately modified figure is then returned. If this is just the
    MACT dataset, then the residuals are returned as well so that they could
    be used in create_disp_tbl().
    '''
    func0, eqn0 = get_func0_eqn0(fittype)

    params_arr = get_mainseq_fit_params(corr_sfrs, delta_sfrs, data00, num_params=3)
    params = [np.mean(params_arr[i]) for i in range(len(params_arr))]

    sfrs_resid = corr_sfrs - func0(data00, *params)
    ax.axhline(0, color='k', ls='--', zorder=1)

    plot_resids(ax, markarr, sizearr, z_arr, no_spectra, yes_spectra,
        stlr_mass, sfrs_resid, filts, ffarr, llarr)
    plot_avg_resids(ax, stlr_mass, sfrs_resid, withnewha)

    # overlaying results from other studies
    salim_2007(ax)
    noeske_2007(ax)

    # final touches
    add_legends(ax, withnewha)
    ax.set_xlabel('log(M'+r'$_\bigstar$'+'/M'+r'$_{\odot}$'+')', size=14)
    ax.set_ylabel(r'$\Delta$'+ytype+' (dex)', size=14)

    [a.tick_params(axis='both', labelsize='10', which='both', direction='in')
        for a in f.axes[:]]

    if withnewha:
        ax.set_xlim([5.5,12.5])
        ax.set_ylim([-1.9,2.3])
        f.set_size_inches(10,8)
    else:
        delosreyes_2015(ax)
        ax.set_xlim([5.5,11.5])
        ax.set_ylim([-1.1,2.0])
        f.set_size_inches(7,6)
        plt.subplots_adjust(hspace=0.01, wspace=0.01, right=0.99, top=0.98,
            left=0.1, bottom=0.09)
        return sfrs_resid


def main():
    '''
    Reads in data from the MACT dataset, and obtains the useful data
    (defined by good_sig_iis). Then, plot_all_dispersion() is called.

    If considering the MACT dataset only, a dispersion is also created and
    either written or printed (for ease of copy+paste into the paper).
    '''
    # reading in data generated by EBV_corrections.py
    corr_tbl = asc.read(config.FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt',
        guess=False, Reader=asc.FixedWidthTwoLine)

    # defining a flux sigma and mass config.CUTOFF
    good_sig_iis = np.where((corr_tbl['flux_sigma'] >= config.CUTOFF_SIGMA) & 
        (corr_tbl['stlr_mass'] >= config.CUTOFF_MASS))[0]
    corr_tbl = corr_tbl[good_sig_iis]

    # getting/storing useful data
    zspec0 = np.array(corr_tbl['zspec0'])
    no_spectra  = np.where((zspec0 <= 0) | (zspec0 > 9))[0]
    yes_spectra = np.where((zspec0 >= 0) & (zspec0 < 9))[0]

    stlr_mass = corr_tbl['stlr_mass'].data
    filts = corr_tbl['filt'].data
    obs_sfr = corr_tbl['met_dep_sfr'].data
    delta_sfrs = corr_tbl['meas_errs'].data
    dust_corr_factor = corr_tbl['dust_corr_factor'].data
    filt_corr_factor = corr_tbl['filt_corr_factor'].data
    nii_ha_corr_factor = corr_tbl['nii_ha_corr_factor'].data
    corr_sfrs = obs_sfr+filt_corr_factor+nii_ha_corr_factor+dust_corr_factor
    from MACT_utils import get_FUV_corrs
    FUV_corr_factor = get_FUV_corrs(corr_tbl)

    zspec00 = approximated_zspec0(zspec0, filts)
    data00 = np.vstack([stlr_mass, zspec00]).T

    z_arr = get_z_arr() # for visualization

    # plotting
    f, ax = plt.subplots()
    sfrs_resid = plot_all_dispersion(f, ax, data00, corr_sfrs+FUV_corr_factor,
        delta_sfrs, stlr_mass,
        filts, no_spectra, yes_spectra, z_arr)
    plt.savefig(config.FULL_PATH+'Plots/main_sequence/mainseq_dispersion.pdf')
    plt.close()

    # creating a dispersion table
    meas_errs = corr_tbl['meas_errs'].data # = delta_sfrs
    tt = create_disp_tbl(stlr_mass, corr_sfrs+FUV_corr_factor,
        corr_sfrs, sfrs_resid, meas_errs)
    asc.write(tt, config.FULL_PATH+'Tables/4.txt', format='latex', overwrite=True)
    # asc.write(tt, config.FULL_PATH+'Main_Sequence/dispersion_tbl.txt', 
    #     format='latex', overwrite=True)
    # print(asc.write(tt, format='latex'))


if __name__ == '__main__':
    main()