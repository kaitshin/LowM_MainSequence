"""
NAME:
    plot_mainseq_UV_Ha_comparison.py

PURPOSE:
    This code plots a comparison between Halpha and UV luminosities (the
    latter of which is 'nu_L_nu'). Then, the ratio of the two is plotted as a
    function of stellar mass.
    With the GALEX command line option, if GALEX is typed, then GALEX files
    files used/output. Otherwise, the files without GALEX photometry are used.

INPUTS:
    'Catalogs/nb_ia_zspec.txt'
    'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+fileend+'.fout'
    'Catalogs/NB_IA_emitters.nodup.colorrev.fix.fits'
    'Main_Sequence/mainseq_corrections_tbl.txt'
    'FAST/outputs/BEST_FITS/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'
         +fileend+'_'+str(ID[ii])+'.fit'

CALLING SEQUENCE:
    main body -> get_nu_lnu -> get_flux
              -> make_scatter_plot, make_ratio_plot
              -> make_all_ratio_plot -> (make_all_ratio_legend,
                                         get_binned_stats)

OUTPUTS:
    'Plots/main_sequence_UV_Ha/'+ff+'_'+ltype+fileend+'.pdf'
    'Plots/main_sequence_UV_Ha/ratios/'+ff+'_'+ltype+fileend+'.pdf'
    'Plots/main_sequence_UV_Ha/ratios/all_filt_'+ltype+fileend+'.pdf'
    
REVISION HISTORY:
    Created by Kaitlyn Shin 13 August 2015
"""

import numpy as np, astropy.units as u, matplotlib.pyplot as plt, sys
from scipy import interpolate
from astropy import constants
from astropy.io import fits as pyfits, ascii as asc
from astropy.cosmology import FlatLambdaCDM
from mainseq_corrections import niiha_oh_determine
cosmo = FlatLambdaCDM(H0 = 70 * u.km / u.s / u.Mpc, Om0=0.3)

# emission line wavelengths (air)
HA = 6562.80

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'
CUTOFF_SIGMA = 4.0
CUTOFF_MASS = 6.0
fileend='.GALEX'


def get_flux(ID, lambda_arr):
    '''
    Reads in the relevant SED spectrum file and then interpolates the
    function to obtain a flux, the array of which is then returned.
    '''
    newflux = np.zeros(len(ID))
    for ii in range(len(ID)):
        tempfile = asc.read(FULL_PATH+
            'FAST/outputs/BEST_FITS/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+
            fileend+'_'+str(ID[ii])+'.fit', guess=False,Reader=asc.NoHeader)
        wavelength = np.array(tempfile['col1'])
        flux = np.array(tempfile['col2'])
        f = interpolate.interp1d(wavelength, flux)
        newflux[ii] = f(lambda_arr[ii])

    return newflux


def get_LUV(corrID, corrzspec0, centr_filts, filt_index_haii, ff):
    '''
    get FUV luminosity (at 1500 AA) by converting the flux (get_flux())

    sources without spectroscopic z are estimated by the center of the
    filter profile

    returns the log of the luminosity log_L_nu (nu=1500AA)
    '''
    ID = corrID[filt_index_haii]
    zspec = corrzspec0[filt_index_haii]

    goodz = np.where((zspec >= 0) & (zspec < 9))[0]
    badz  = np.where((zspec <= 0) | (zspec > 9))[0]

    tempz = np.zeros(len(filt_index_haii))
    tempz[goodz] = zspec[goodz]
    tempz[badz] = centr_filts[ff]

    lambda_arr = (1+tempz)*1500

    f_lambda = get_flux(ID, lambda_arr)
    f_nu = f_lambda*(1E-19*(lambda_arr**2*1E-10)/(constants.c.value))
    log_L_nu = np.log10(f_nu*4*np.pi) + \
        2*np.log10(cosmo.luminosity_distance(tempz).to(u.cm).value)

    return log_L_nu


def plot_ff_zz_color_filled(ax, xvals, yvals, corr_tbl,
    ff_arr=['NB7', 'NB816', 'NB921', 'NB973'],
    ll_arr=['NB704,NB711', 'NB816', 'NB921', 'NB973'],
    color_arr = ['r', 'orange', 'g', 'b']):
    '''
    '''
    from plot_nbia_mainseq import get_z_arr
    z_arr = get_z_arr()

    zspec0 = corr_tbl['zspec0'].data
    for (ff, cc, zz, ll) in zip(ff_arr, color_arr, z_arr, ll_arr):
        filt_index_haii = np.array([x for x in range(len(corr_tbl)) if ff in
            corr_tbl['filt'].data[x]])
        
        zspec = zspec0[filt_index_haii]
        good_z = np.array([x for x in range(len(zspec)) if zspec[x] > 0. and
                           zspec[x] < 9.])
        bad_z  = np.array([x for x in range(len(zspec)) if zspec[x] <= 0. or
                           zspec[x] >= 9.])
        
        ax.scatter(xvals[filt_index_haii][good_z], yvals[filt_index_haii][good_z],
            facecolor=cc, edgecolor='none', alpha=0.3, s=30,
            label='z~'+zz+' ('+ll+')')
        ax.scatter(xvals[filt_index_haii][bad_z], yvals[filt_index_haii][bad_z],
            facecolor='none', edgecolor=cc, linewidth=0.5, alpha=0.3, s=30)

    return ax


def plot_zz_shapes_filled(ax, xvals, yvals, corr_tbl, color, legend_on=False,
    ff_arr=['NB7', 'NB816', 'NB921', 'NB973'],
    ll_arr=['NB704,NB711', 'NB816', 'NB921', 'NB973'],
    mark_arr=['o', '^', 'D', '*'], size_arr=np.array([6.0, 6.0, 6.0, 9.0])**2):
    '''
    '''
    from plot_nbia_mainseq import get_z_arr
    z_arr = get_z_arr()

    labelarr = np.array([])
    zspec0 = corr_tbl['zspec0'].data
    for (ff, mark, avg_z, size, ll) in zip(ff_arr, mark_arr, z_arr, size_arr, ll_arr):
        filt_index_haii = np.array([x for x in range(len(corr_tbl)) if ff in
            corr_tbl['filt'].data[x]])
        
        zspec = zspec0[filt_index_haii]
        good_z = np.array([x for x in range(len(zspec)) if zspec[x] > 0. and
                           zspec[x] < 9.])
        bad_z  = np.array([x for x in range(len(zspec)) if zspec[x] <= 0. or
                           zspec[x] >= 9.])

        temp = ax.scatter(xvals[filt_index_haii][good_z], yvals[filt_index_haii][good_z],
            marker=mark, facecolors=color, edgecolors='none', alpha=0.2,
            zorder=1, s=size, label='z~'+np.str(avg_z)+' ('+ll+')')

        ax.scatter(xvals[filt_index_haii][bad_z], yvals[filt_index_haii][bad_z],
            marker=mark, facecolors='none', edgecolors=color, alpha=0.2,
            linewidth=0.5, zorder=1, s=size)

        labelarr = np.append(labelarr, temp)

    if legend_on:
        leg1 = ax.legend(handles=list(labelarr), loc='upper right', frameon=False)
        ax.add_artist(leg1)

    return ax


def plot_binned_with_yesz(ax, xvals, yvals, plot_bins, zspec0=None):
    '''
    plots bins, where the bin edges are passed in as the param `plot_bins`

    if there are less than `min_yesz_per_bin` number of sources with
    spectroscopic confirmation in that bin, then the binned avg point
    is plotted with an open shape

    plots mean(x), mean(y), and 1 stddev y-error bars (std(y))
    '''
    iis = np.digitize(xvals, bins=plot_bins)
    range_iis = np.arange(len(plot_bins[:-1])) + 1
    xvals_binned = np.array([np.mean([plot_bins[ii],plot_bins[ii+1]]) 
        for ii in range_iis-1])

    yvals_binned = np.array([np.mean(yvals[np.where(iis==ii)])
        for ii in range_iis])
    yerrs = np.array([np.std(yvals[np.where(iis==ii)])
            for ii in range_iis])

    if zspec0 is None:
        ax.plot(xvals_binned, yvals_binned, 'md')
    else:
        min_yesz_per_bin = 10
        yesz_num = np.array([len(np.where((zspec0[np.where(iis==ii)] >= 0) & 
            (zspec0[np.where(iis==ii)] < 9))[0]) for ii in range_iis])
        
        for yesz, xval, yval in zip(yesz_num, xvals_binned, yvals_binned):
            if yesz > min_yesz_per_bin:
                ax.plot(xval, yval, 'md')
            else:
                ax.scatter(xval, yval, edgecolors='m', facecolors='none',
                    marker='d', zorder=10)

    ax.errorbar(xvals_binned, yvals_binned, fmt='none', ecolor='m', lw=1,
        yerr=yerrs, zorder=11)

    return ax


def plot_binned_percbins(ax, xvals, yvals, corr_tbl, yesz=True, num_bins=8):
    '''
    if yesz (default=True), then only sources with spectroscopic confirmation
    are considered in the binning

    splits the data into num_bins bins (default=8) and plots the avg(z) and
    avg(y) per bin.

    yerr bars are 1stddev in y (std(y)), and xerr bars are the xrange
    of the particular bin.
    '''
    if yesz:
        # limiting to yesz only
        zspec0 = corr_tbl['zspec0'].data
        good_z = np.array([x for x in range(len(zspec0)) if zspec0[x] > 0. and
            zspec0[x] < 9.])
        xvals = xvals[good_z]
        yvals = yvals[good_z]

    bin_edges = np.linspace(0, 100, num_bins+1)
    step_size = bin_edges[1] - bin_edges[0]

    xvals_perc_ii = np.array([[i for i in range(len(xvals)) if 
        (xvals[i] <= np.percentile(xvals,NUM) 
            and xvals[i] > np.percentile(xvals,NUM-step_size))] 
        for NUM in bin_edges[1:]])
    # edge case: add in 0th percentile value which isn't accounted for above
    xvals_perc_ii[0] = np.insert(xvals_perc_ii[0], 0, np.argmin(xvals))

    for i in range(xvals_perc_ii.shape[0]):
        xarr = xvals[xvals_perc_ii[i]]
        xval = np.mean(xarr)

        yarr = yvals[xvals_perc_ii[i]]
        yval = np.mean(yarr)

        ax.plot(xval, yval, 'kd', zorder=12)
        ax.errorbar(xval, yval, fmt='none', ecolor='k', lw=1,
            zorder=11, yerr=np.std(yarr),
            xerr=np.array([[xval - min(xarr)],
                [max(xarr) - xval]]))
    return ax


def plot_SFR_comparison(log_SFR_HA, log_SFR_UV, corr_tbl):
    '''
    comparing with Lee+09 fig 1 relation
    without dust correction
    '''
    f, ax = plt.subplots()

    # plotting data
    ax = plot_zz_shapes_filled(ax, log_SFR_HA, log_SFR_UV, corr_tbl,
        color='blue', legend_on=True)

    # plotting 1-1 correspondence
    xlims = [min(log_SFR_HA)-0.2, max(log_SFR_HA)+0.2]
    ylims = [min(log_SFR_UV)-0.4, max(log_SFR_UV)+0.4]
    ax.plot(xlims, xlims, 'k')

    # plotting relation from Lee+09
    ax = lee_09(ax, xlims, lee_fig_num='1')

    # final touches
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xlabel('log SFR(Ha)')
    ax.set_ylabel('log SFR(UV)')
    ax.tick_params(axis='both', labelsize='10', which='both',
        direction='in')
    ax.minorticks_on()
    f.set_size_inches(6,6)
    ax.legend(frameon=False, loc='best')
    plt.savefig(FULL_PATH+'Plots/main_sequence_UV_Ha/SFR_UV_vs_HA.pdf')


def get_UV_SFR(corr_tbl):
    '''
    deriving log UV SFR from the 1500AA measurements
    '''
    npz_files = np.load(FULL_PATH+'Plots/sfr_metallicity_plot_fit.npz')

    Lnu_fit_ch = npz_files['Lnu_fit_ch']

    P2, P1, P0 = -1*Lnu_fit_ch
    def log_SFR_from_L(zz):
        '''zz is metallicity: log(Z/Z_sol)'''
        return P0 + P1*zz + P2*zz**2

    # niiha_oh_determine estimates log(O/H)+12
    NII6583_Ha = corr_tbl['NII_Ha_ratio'].data * 2.96/(1+2.96)
    logOH = niiha_oh_determine(np.log10(NII6583_Ha), 'PP04_N2') - 12
    y = logOH + 3.31

    log_SFR_LUV = log_SFR_from_L(y)

    return log_SFR_LUV


def lee_09(ax, xlims0, lee_fig_num):
    '''
    overlays data points and theoretical relations from Lee+09
    figures 1 and 2, depending on which one is being compared
    '''
    if lee_fig_num=='1':
        xtmparr = np.linspace(xlims0[0], xlims0[1], 10)
        ax.plot(xtmparr, 0.79*xtmparr-0.2, 'k--')

    elif lee_fig_num=='2A':
        jlee_logSFRHa = np.array([0.25,-0.25,-0.75,-1.25,-1.75,-2.25,-2.75,-3.5,-4.5])
        jlee_logSFR_ratio = np.array([0.2,0.17,0.07,-0.02,-.1,-.23,-.46,-.49,-1.29])
        jlee_logSFR_ratio_errs = np.array([0.37,0.30,0.26,0.25,0.22,0.22,0.26,0.58,0.57])
        ax.plot(jlee_logSFRHa, jlee_logSFR_ratio, 'cs', alpha=0.7)
        ax.errorbar(jlee_logSFRHa, jlee_logSFR_ratio, fmt='none', ecolor='c', lw=2,
            yerr=jlee_logSFR_ratio_errs, alpha=0.7)
        xtmparr0 = np.linspace(min(jlee_logSFRHa)-0.1, xlims0[1], 10)
        jlee09, = ax.plot(xtmparr0, 0.26*xtmparr0+0.3, 'c--', alpha=0.7, 
            label='Lee+09: '+r'$\log(\rm SFR(H\alpha)/SFR(FUV)) = 0.26 \log(SFR(H\alpha))+0.30$')
        legend_jlee09 = ax.legend(handles=[jlee09], loc='lower right', frameon=False)
        ax.add_artist(legend_jlee09)

    else:
        raise ValueError('Invalid fig_num. So far only Lee+09 figs 1 and 2A are\
            valid comparisons')

    return ax


def plot_SFR_ratios_final_touches(f, ax0, ax1):
    '''
    does final touches for the SFR ratios, SFR_ratio.pdf
    incl. x labels, ticks, sizing, and saving
    '''
    # labels
    ax0.set_xlabel(r'$\log \rm SFR(H\alpha)$', fontsize=12)
    ax1.set_xlabel('log(M'+r'$_\bigstar$'+'/M'+r'$_{\odot}$'+')', fontsize=12)
    [ax.set_ylabel(r'$\log(\rm SFR(H\alpha)/SFR(FUV))$',
        fontsize=12) for ax in [ax0,ax1]]

    # ticks
    [ax.tick_params(axis='both', labelsize='10', which='both',
        direction='in') for ax in [ax0,ax1]]
    [ax.minorticks_on() for ax in [ax0,ax1]]

    # sizing+saving
    f.set_size_inches(12,5)
    plt.tight_layout()
    plt.savefig(FULL_PATH+'Plots/main_sequence_UV_Ha/SFR_ratio.pdf')


def plot_SFR_ratios(log_SFR_HA, log_SFR_UV, corr_tbl):
    '''
    without dust correction
    '''
    log_SFR_ratio = log_SFR_HA - log_SFR_UV
    stlr_mass = corr_tbl['stlr_mass'].data

    f, axarr = plt.subplots(1,2)
    ax0 = axarr[0]
    ax1 = axarr[1]

    # plotting data
    ax0 = plot_zz_shapes_filled(ax0, log_SFR_HA, log_SFR_ratio, corr_tbl,
        color='gray', legend_on=True)
    ax1 = plot_zz_shapes_filled(ax1, stlr_mass, log_SFR_ratio, corr_tbl,
        color='gray')

    xlims0 = [min(log_SFR_HA)-0.2, max(log_SFR_HA)+0.2]
    xlims1 = [min(stlr_mass)-0.2, max(stlr_mass)+0.2]
    ylims = [min(log_SFR_ratio)-0.4, max(log_SFR_ratio)+0.4]
    [ax.axhline(0, color='k') for ax in [ax0,ax1]]

    # plotting relation from Lee+09
    ax0 = lee_09(ax0, xlims0, lee_fig_num='2A')

    # plotting our own avgs
    plot_bins = np.array([-4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5])
    ax0 = plot_binned_percbins(ax0, log_SFR_HA, log_SFR_ratio, corr_tbl)
    ax1 = plot_binned_percbins(ax1, stlr_mass, log_SFR_ratio, corr_tbl)

    # final touches
    plot_SFR_ratios_final_touches(f, ax0, ax1)


def main():
    '''
    o Reads relevant inputs
    o Iterating by filter, calls nu_lnu, make_scatter_plot, and
      make_ratio_plot
    o After the filter iteration, make_all_ratio_plot is called.
    o For each of the functions to make a plot, they're called twice - once for
      plotting the nii/ha corrected version, and one for plotting the dust
      corrected version.

    +190531: only GALEX files will be used
    '''
    # reading input files
    fout  = asc.read(FULL_PATH+
        'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+fileend+'.fout',
        guess=False, Reader=asc.NoHeader)

    corr_tbl = asc.read(FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt',
        guess=False, Reader=asc.FixedWidthTwoLine)
    good_sig_iis = np.where((corr_tbl['flux_sigma'] >= CUTOFF_SIGMA) & 
        (corr_tbl['stlr_mass'] >= CUTOFF_MASS))[0]
    corr_tbl = corr_tbl[good_sig_iis]
    print '### done reading input files'


    # defining useful things
    corrID = corr_tbl['ID'].data
    corrzspec0 = corr_tbl['zspec0'].data
    corrfilts = corr_tbl['filt'].data

    color_arr = ['r', 'orange', 'g', 'b']
    centr_filts = {'NB7':((7045.0/HA - 1) + (7126.0/HA - 1))/2.0, 
        'NB816':8152.0/HA - 1, 'NB921':9193.0/HA - 1, 'NB973':9749.0/HA - 1}


    # getting SFR values
    log_SFR_LUV = get_UV_SFR(corr_tbl)

    LUV = np.zeros(len(corr_tbl))
    for ff in ['NB7','NB816','NB921','NB973']:
        print ff

        filt_index_haii = np.array([x for x in range(len(corr_tbl)) if ff in
            corrfilts[x]])

        lnu = get_LUV(corrID, corrzspec0, centr_filts, filt_index_haii, ff)
        LUV[filt_index_haii] = lnu

    log_SFR_UV = log_SFR_LUV + LUV
    log_SFR_HA = corr_tbl['met_dep_sfr'].data


    # plotting
    plot_SFR_comparison(log_SFR_HA, log_SFR_UV, corr_tbl)
    plot_SFR_ratios(log_SFR_HA, log_SFR_UV, corr_tbl)


if __name__ == '__main__':
    main()