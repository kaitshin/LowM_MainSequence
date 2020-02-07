"""
NAME:
    plot_nbia_mainseq.py

PURPOSE:
    Generates main sequence plots (incl. the 4-panel fig., a fig. w/ the best-
    fit relation, and a sSFR fig.) for the MACT dataset.
    
    Depends on mainseq_corrections.py.
    
    Contains functions that are called by nbia_mainseq_dispersion.py
    and plot_mact_with_newha.py.

INPUTS:
    FULL_PATH+'Main_Sequence/Berg2012_table.clean.txt'
    FULL_PATH+'Main_Sequence/Noeske07_fig1_z1.txt'
    FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt'

OUTPUTS:
    FULL_PATH+'Plots/main_sequence/mainseq.pdf'
    if mainseq_fig4_only = True:
        FULL_PATH+'Plots/main_sequence/mainseq_allcorrs.pdf'
    FULL_PATH+'Plots/main_sequence/zdep_mainseq.pdf'
    FULL_PATH+'Plots/main_sequence/mainseq_sSFRs.pdf'
"""
from __future__ import print_function

import numpy as np, matplotlib.pyplot as plt
import scipy.optimize as optimize
import matplotlib as mpl
from scipy.optimize import curve_fit
from astropy.io import ascii as asc
from MACT_utils import get_z_arr, get_mainseq_fit_params, compute_onesig_pdf

# emission line wavelengths (air)
HA = 6562.80

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'
CUTOFF_SIGMA = 4.0
CUTOFF_MASS = 6.0

mainseq_fig4_only = False

def whitaker_2014(ax):
    '''
    Plots the log(M*)-log(SFR) relation from Whitaker+14 in red.
    '''
    xmin = 8.4
    xmax = 11.2
    xarr1 = np.arange(xmin, 10.2+0.01, 0.01)
    xarr2 = np.arange(10.2, xmax+0.01, 0.01)

    ax.plot(xarr1, 0.94 * (xarr1 - 10.2) + 1.11, 'r--', lw=2)
    whitaker, = ax.plot(xarr2, 0.14 * (xarr2 - 10.2) + 1.11, 'r--',
        label='Whitaker+14 (0.5<z<1.0)', zorder=6, lw=2)

    return whitaker


def delosreyes_2015(ax, fuv=False, corr_tbl=None):
    '''
    Plots the z~0.8 data points from de Los Reyes+15 in cyan.
    '''
    xarr = np.array([9.27, 9.52, 9.76, 10.01, 10.29, 10.59, 10.81, 11.15])
    yarr = np.array([0.06, 0.27, 0.43, 0.83, 1.05, 1.18, 1.50, 1.54])
    yerr = np.array([0.454, 0.313, 0.373, 0.329, 0.419, 0.379, 0.337, 0.424])
    if fuv:
        from MACT_utils import get_FUV_corrs
        m, b, const = get_FUV_corrs(corr_tbl, ret_coeffs_const=True)
        yarr = yarr-const

    ax.errorbar(xarr, yarr, yerr, fmt='deepskyblue', ecolor='deepskyblue',
        zorder=2) 
    delosreyes = ax.scatter(xarr, yarr, color='deepskyblue', marker='s',
        label='de los Reyes+15 (z~0.8)', zorder=2)

    return delosreyes


def salim_2007(ax):
    '''
    Plots the log(M*)-log(SFR) relation from Salim+07 in black.
    '''
    xmin = 8.5
    xmax = 11.2
    xarr = np.arange(xmin, xmax+0.01, 0.01)
    
    def salim_line(xarr):
        return (-0.35 * (xarr - 10.0) - 9.83) + xarr

    lowlim = salim_line(xarr) - np.array([0.2]*len(xarr))
    uplim = salim_line(xarr) + np.array([0.2]*len(xarr))

    ax.plot(xarr, lowlim, 'k--', zorder=1, lw=0.5)
    ax.plot(xarr, uplim, 'k--', zorder=1, lw=0.5)
    ax.fill_between(xarr, lowlim, uplim, color='gray', alpha=0.2)
    salim, = ax.plot(xarr, salim_line(xarr), 'k-',
        label='Salim+07 (z~0.1)', zorder=1, lw=0.5)

    return salim


def berg_2012(ax):
    '''
    Plots the log(M*)-log(SFR) relation from Berg+12 in green. (ASCII file
    provided by Chun Ly)
    '''
    berg = asc.read(FULL_PATH+'Main_Sequence/Berg2012_table.clean.txt',
        guess=False, Reader=asc.CommentedHeader, delimiter='\t')
    berg_stlr = np.array(berg['log(M*)'])
    berg_sfr  = np.log10(np.array(berg['SFR']))

    berg = ax.scatter(berg_stlr, berg_sfr, color='g', marker='x',
        label='Berg+12 (z<0.01)', zorder=4)

    return berg


def noeske_2007(ax):
    '''
    Plots the data points from Noeske+07 in orange. (ASCII file provided by
    Chun Ly)
    '''
    noeske = asc.read(FULL_PATH+'Main_Sequence/Noeske07_fig1_z1.txt',
        guess=False, Reader=asc.NoHeader)
    logM   = np.array(noeske['col1'])
    logSFR = np.array(noeske['col2'])
    logSFR_low  = np.array(noeske['col3'])
    logSFR_high = np.array(noeske['col4'])

    ax.plot(logM, logSFR_low, color='orange', marker='', linestyle='',
        zorder=1)
    ax.plot(logM, logSFR_high, color='orange', marker='', linestyle='',
        zorder=1)
    ax.fill_between(logM, logSFR_low, logSFR_high, facecolor='none',
        hatch=3*'.', edgecolor='orange', linewidth=0.0, zorder=1)
    noeske, = ax.plot(logM, logSFR, color='orange', marker='+',
        label='Noeske+07 (0.20<z<0.45)',zorder=1, mew=2, markersize=11)

    return noeske


def sSFR_lines(ax, xlim):
    '''
    Creates the four dotted sSFR^-1 lines: 0.1, 1, 10, and 100 Gyr
    '''
    xmin = min(xlim)
    xmax = max(xlim)
    xarr = np.arange(xmin, xmax, 0.01)

    ax.plot(xarr, xarr - 8, 'k:',zorder=8)
    ax.text(5.85, -1.4, 'sSFR=(0.1 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)

    ax.plot(xarr, xarr - 9, 'k:',zorder=8)
    ax.text(5.85, -2.4, 'sSFR=(1.0 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)

    ax.plot(xarr, xarr - 10, 'k:')
    ax.text(6.17, -3.04, 'sSFR=(10.0 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)

    ax.plot(xarr, xarr - 11, 'k:')
    ax.text(7.15, -3.0, 'sSFR=(100.0 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)


def modify_all_graph(ax, labelarr, xlim, ylim, title, i, corr_tbl):
    '''
    Modifies the 4-panel graph (gives it limits, labels, a title, etc.)
    
    Low mass sources are overlaid with a '+' by calling lowmass().
    
    Information from previous literature is plotted by calling the various
    methods above (usually named after primary author_year).

    Two legends are created - one for the shapes and one for the different
    literature/data points. Then minor ticks were added.
    '''
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    if i>1:
        ax.set_xlabel('log(M'+r'$_\bigstar$'+'/M'+r'$_{\odot}$'+')', size=14)
        noeske = noeske_2007(ax)
        whitaker = whitaker_2014(ax)
        salim = salim_2007(ax)
        berg = berg_2012(ax)
        if i==3:
            delosreyes = delosreyes_2015(ax, fuv=True, corr_tbl=corr_tbl)
        else:
            delosreyes = delosreyes_2015(ax)
    if i%2==0 or i==5:
        ax.set_ylabel('log(SFR[H'+r'$\alpha$'+']/M'+r'$_{\odot}$'+
            ' yr'+r'$^{-1}$'+')', size=14)
    if i%2==1 and i!=5:
        ax.set_yticklabels([])
    if i<2:
        ax.set_xticklabels([])
    ax.text(0.02, 0.95, title, transform=ax.transAxes, color='k', fontsize=14,
        fontweight='bold')

    if i==3 or i==5:
        labelarr2 = np.array([whitaker, delosreyes, noeske, salim, berg])
        legend2 = ax.legend(handles=list(labelarr2), loc='lower right',
            frameon=False, fontsize=11, scatterpoints=1, numpoints=1)
        ax.add_artist(legend2)
    if i==0 or i==5:
        legend1 = ax.legend(handles=list(labelarr), loc=(0.01, 0.78),
            frameon=False, fontsize=11, scatterpoints=1, numpoints=1)
        ax.add_artist(legend1)

    ax.minorticks_on()
    sSFR_lines(ax, xlim)


def plot_avg_sfrs(ax, stlr_mass, sfrs, newha=False, openz=False,
    cc='k', aa=1.0, zz=1):
    '''
    assumes the lowest mass is at m=6
    plots the mean sfr in each mass bin of width 0.5
    '''
    if newha:
        mbins0 = np.arange(6.25, 11.75, .5)
    else:
        mbins0 = np.arange(6.25, 10.75, .5)
    bin_ii = np.digitize(stlr_mass, mbins0+0.25)

    for i in range(len(mbins0)):
        bin_match = np.where(bin_ii == i)[0]
        sfrs_matched = sfrs[bin_match]
        # if openz:
        #     # ax.plot(mbins0[i], np.mean(sfrs_matched), 'o', 
        #     #     markerfacecolor='none', markeredgecolor='k',
        #     #     alpha=0.8, ms=8)
        #     ax.plot(mbins0[i], np.mean(sfrs_matched), 'o', color=cc,
        #         alpha=aa, ms=8, zorder=1)
        # else:
        #     ax.plot(mbins0[i], np.mean(sfrs_matched), 'ko',
        #         alpha=0.8, ms=8, zorder=4)

        ax.plot(mbins0[i], np.mean(sfrs_matched), 'o', color=cc,
                alpha=aa, ms=8, zorder=zz)
        ax.errorbar(mbins0[i], np.mean(sfrs_matched), xerr=0.25, fmt='none',
            ecolor=cc, alpha=aa, lw=2)

        if not newha:
            # calculating yerr assuming a uniform distribution
            np.random.seed(213078)
            num_iterations = 1000
            len0 = len(sfrs_matched)

            MC_arr = np.random.choice(sfrs_matched, size=(len0, num_iterations))
            avg_dist = np.average(MC_arr, axis=0)
            avg_dist = np.reshape(avg_dist,(1,num_iterations))

            #x_pdf, x_val
            ysfrerr, xpeak = compute_onesig_pdf(avg_dist, [np.mean(sfrs_matched)])
            ax.errorbar(mbins0[i], np.mean(sfrs_matched), yerr=ysfrerr, fmt='none',
                ecolor='black', alpha=0.8, lw=2)


def get_filt_index(spectra, ff, filts):
    '''
    returns the indexes at which the sources are in the filter
    (and handles the special case of 'NB704+NB711')

    compatible with both spectra == no_spectra and spectra == yes_spectra

    called in make_all_graph() and make_redshift_graph()
    '''
    if 'NB7' in ff:
        filt_index = np.array([x for x in range(len(spectra)) if
            ff[:3] in filts[spectra][x]])
    else:
        filt_index = np.array([x for x in range(len(spectra)) if
            ff==filts[spectra][x]])

    return filt_index


def make_all_graph(stlr_mass, sfr, filtarr, markarr, z_arr, sizearr, title,
    no_spectra, yes_spectra, filts, ax, i, corr_tbl):
    '''
    Makes the 4-panel main sequence figure with varying levels of correction
    applied. Shapes are iterated through by filter (proxy for redshift). 
    Average SFRs are plotted in 0.5dex mass bins. The plot is then modified
    and returned.
    '''
    color='blue'
    xlim = [5.80, 11.20]
    ylim = [-3.75, 2]

    labelarr = np.array([])
    check_nums = []
    for (ff, mark, avg_z, size) in zip(filtarr, markarr, z_arr, sizearr):
        filt_index_n = get_filt_index(no_spectra, ff, filts)
        filt_index_y = get_filt_index(yes_spectra, ff, filts)

        print('>>>', ff, avg_z)
        check_nums.append(len(filt_index_y)+len(filt_index_n))

        temp = ax.scatter(stlr_mass[yes_spectra][filt_index_y],
            sfr[yes_spectra][filt_index_y], marker=mark,
            facecolors=color, edgecolors='none', alpha=0.2, zorder=3, s=size,
            label='z~'+np.str(avg_z)+' ('+ff+')')

        ax.scatter(stlr_mass[no_spectra][filt_index_n],
            sfr[no_spectra][filt_index_n], marker=mark,
            facecolors='none', edgecolors=color, alpha=0.2, linewidth=0.5,
            zorder=3, s=size, label='z~'+np.str(avg_z)+' ('+ff+')')
        
        labelarr = np.append(labelarr, temp)

    assert np.sum(check_nums)==len(sfr)
    plot_avg_sfrs(ax, stlr_mass, sfr)
    modify_all_graph(ax, labelarr, xlim, ylim, title, i, corr_tbl)


def plot_redshift_avg_sfrs(ax, stlr_mass, sfrs, cc):
    '''
    Plots the average SFRs for the redshift-dependent relation mainseq figure
    in bins of 0.5dex mass. xerr bars denote the mass range.
    '''
    mbins0 = np.arange(6.25, 12.25, .5)
    bin_ii = np.digitize(stlr_mass, mbins0+0.25)
    
    for i in set(bin_ii):
        bin_match = np.where(bin_ii == i)[0]
        avg_sfr = np.mean(sfrs[bin_match])
        avg_mass = np.mean(stlr_mass[bin_match])
        
        min_per_bin = 5
        if len(bin_match) < min_per_bin:
            ax.scatter(avg_mass, avg_sfr, edgecolors=cc, facecolors='none',
                marker='s', alpha=0.4, s=15**2, linewidth=1)
        elif len(bin_match) >= min_per_bin:
            ax.plot(avg_mass, avg_sfr, color=cc, marker='s', alpha=0.6,
                ms=15, mew=0)
        
        ax.errorbar(avg_mass, avg_sfr, fmt='none', ecolor=cc, alpha=0.6, lw=2,
            xerr=np.array([[avg_mass - (mbins0[i]-0.25)],
                [(mbins0[i]+0.25) - avg_mass]]))


def get_func0_eqn0(fittype):
    '''
    returns functions and equation strings based on the fit type (either
    first or second order)

    these functions are the 'model's that are subtracted from the data
    to calculate the residuals
    '''
    if fittype=='first_order':
        eqn0 = r'$log(SFR) = \alpha log(M) + \beta z + \gamma$'
        def func0(data, a, b, c):
            return a*data[:,0] + b*data[:,1] + c

    elif fittype=='second_order':
        eqn0 = r'$log(SFR) = \alpha ^l log(M)^2 + \alpha log(M) + \beta z + \gamma$'
        def func0(data, aprime, a, b, c):
            return aprime*data[:,0]**2 + a*data[:,0] + b*data[:,1] + c

    else:
        raise ValueError('invalid fit type')

    return func0, eqn0


def modify_redshift_graph(f, ax, fittype, eqn0, params, ytype, withnewha):
    '''
    Modifies the redshift-dependent graph to add labels, legend, text, and
    adjust limits according to ytype and withnewha. 
    '''
    ax.set_xlabel('log(M'+r'$_\bigstar$'+'/M'+r'$_{\odot}$'+')', size=14)
    if withnewha:
        xpos = 0.40
    else:
        xpos = 0.50
    ypos = 0.12

    if ytype == 'SFR':
        ax.legend(loc='upper left', fontsize=14, frameon=False)
        ax.set_ylabel('log(SFR[H'+r'$\alpha$'+']/M'+r'$_{\odot}$'+
            ' yr'+r'$^{-1}$'+')', size=14)
    elif ytype == 'sSFR':
        xpos = 0.02
        ax.legend(loc='upper right', fontsize=14, frameon=False)
        ax.set_ylabel('log(sSFR[H'+r'$\alpha$'+']'+
            ' yr'+r'$^{-1}$'+')', size=14)
    else:
        raise ValueError('invalid ytype')

    if fittype=='first_order':
        ax.text(xpos, ypos, eqn0+
            '\n'+r'$\alpha=$'+'%.2f'%(params[0])+', '+r'$\beta=$'+
            '%.2f'%(params[1])+', '+r'$\gamma=$'+'%.2f'%(params[2]),
            transform=ax.transAxes, fontsize=15, ha='left', va='top')
    elif fittype=='second_order':
        ax.text(xpos, ypos, eqn0+
            '\n'+r'$\alpha ^l =$'+'%.2f'%(params[0])+', '+r'$\alpha=$'+
            '%.2f'%(params[1])+', '+r'$\beta=$'+'%.2f'%(params[2])+
            ', '+r'$\gamma=$'+'%.2f'%(params[3]),
            transform=ax.transAxes, fontsize=15, ha='left', va='top')
    else:
        raise ValueError('invalid fit type')

    [a.tick_params(axis='both', labelsize='10', which='both', direction='in')
        for a in f.axes[:]]
    if withnewha:
        f.set_size_inches(10,8)
    else:
        f.set_size_inches(7,6)


def make_redshift_graph(f, ax, z_arr, corr_sfrs, delta_sfrs, stlr_mass, zspec0, filts,
    no_spectra, yes_spectra, cwheel, ffarr=['NB7', 'NB816', 'NB921', 'NB973'],
    llarr=['NB704,NB711', 'NB816', 'NB921', 'NB973'], ytype='SFR',
    fittype='first_order', withnewha=False):
    '''
    Makes a main sequence figure with all corrections applied and the derived
    best-fit line shown as well. Colors are iterated through by filter (proxy
    for redshift).

    Calls plot_redshift_avg_sfrs() to plot the average SFRs in 0.5dex mass
    bins, and calls modify_redshift_graph() to modify the plot. 
    '''
    func0, eqn0 = get_func0_eqn0(fittype)

    centr_filts = {'NB7':((7045.0/HA - 1) + (7126.0/HA - 1))/2.0, 
        'NB816':8152.0/HA - 1, 'NB921':9193.0/HA - 1, 'NB973':9749.0/HA - 1,
        'NEWHA':0.8031674}


    data00 = np.vstack([stlr_mass, zspec0]).T

    params_arr = get_mainseq_fit_params(corr_sfrs, delta_sfrs, data00, num_params=3)
    params = [np.mean(params_arr[i]) for i in range(len(params_arr))]


    for ff, cc, ll, zz in zip(ffarr[::-1], cwheel[::-1],
        llarr[::-1], z_arr[::-1]):

        filt_index_n = get_filt_index(no_spectra, ff, filts)
        filt_index_y = get_filt_index(yes_spectra, ff, filts)

        # scattering
        ax.scatter(stlr_mass[yes_spectra][filt_index_y],
            corr_sfrs[yes_spectra][filt_index_y], facecolors=cc,
            edgecolors='none', alpha=0.3, zorder=3, label='z~'+zz+' ('+ll+')')
        if ff != 'NEWHA':
            ax.scatter(stlr_mass[no_spectra][filt_index_n],
                corr_sfrs[no_spectra][filt_index_n], facecolors='none',
                edgecolors=cc, alpha=0.3, linewidth=0.5, zorder=3)

        # plotting the best-fit lines
        filt_match = np.array([x for x in range(len(filts)) if ff in filts[x]])
        mrange = np.arange(min(stlr_mass[filt_match]),
            max(stlr_mass[filt_match]), 0.1)
        avgz = np.array([centr_filts[ff]]*len(mrange))
        tmpdata = np.vstack([mrange, avgz]).T
        ax.plot(mrange, func0(tmpdata, *params), color=cc, lw=2)

        plot_redshift_avg_sfrs(ax, stlr_mass[filt_match], corr_sfrs[filt_match],
            cc)

    modify_redshift_graph(f, ax, fittype, eqn0, params, ytype, withnewha)


def bestfit_zssfr(ax, filtbins_1z, filtbins_ssfr, delta_sfrs):
    '''
    plots and returns the parameters for the best-fit linear relation to
    the sSFR as a function of redshift.

    called by make_ssfr_graph()
    '''
    def line(x, m, b):
        return m*x + b
    

    num_iters = 10000
    np.random.seed(12376)

    a_arr = np.zeros(num_iters)
    c_arr = np.zeros(num_iters)

    for i in range(num_iters):
        zmeans, smeans = [], []
        for j, zbin in enumerate(filtbins_1z):
            sbin = filtbins_ssfr[j]
            assert len(zbin) == len(sbin)
            n_gal = len(zbin)

            z_arr = np.random.choice(zbin, n_gal)
            s_arr = np.random.choice(sbin, n_gal)

            zmeans.append(np.mean(z_arr))
            smeans.append(np.mean(s_arr))

        params, pcov = curve_fit(line, zmeans, smeans)
        a_arr[i] = params[0]
        c_arr[i] = params[1]

    assert (0 not in a_arr) and (0 not in c_arr)
    params_arr = [a_arr, c_arr]
    params = [np.mean(params_arr[i]) for i in range(len(params_arr))]

    errs_arr = []
    for i, arr in enumerate(params_arr):
        errs_arr.append(compute_onesig_pdf(arr.reshape(len(arr),1).T,
            [np.mean(arr)])[0][0])

    print('sSFR a*log(1+z)+b params:', params)
    print('+/-', [errs_arr[0][0], errs_arr[0][1]])

    stp = 0.02
    xrange_tmp = np.linspace(min(filtbins_1z[0])-stp, max(filtbins_1z[-1])+stp, 100)
    ax.plot(xrange_tmp, line(xrange_tmp, *params), 'k--')


def make_ssfr_graph_old(f, axes, sfrs00, delta_sfrs, smass0, filts00, zspec00, cwheel, z_arr,
    ffarr=['NB7', 'NB816', 'NB921', 'NB973'],
    llarr=['NB704,NB711', 'NB816', 'NB921', 'NB973']):
    '''
    plots a two-panel plot of sSFR as a function of mass (LHS) and redshift
    (RHS). colors differ depending on the associated filter of the source.
        note: currently plots log(1+z) rather than z for the RHS panel

    calls bestfit_zssfr() to plot the best-fit line of sSFR as a function of
    redshift and return those parameters as well.
    '''
    ssfr = sfrs00-smass0
    filtbins_1z, filtbins_ssfr = [], []
    for i, ax in enumerate(axes):
        for ff,cc,ll,zz in zip(ffarr, cwheel, llarr, z_arr):
            filt_match = np.array([x for x in range(len(filts00)) if
                ff in filts00[x]])
            
            if i==0:
                ax.scatter(smass0[filt_match], ssfr[filt_match],
                    facecolors='none', edgecolors=cc, linewidth=0.5,
                    label='z~'+zz+' ('+ll+')')
                ax.set_xlabel('log(M'+r'$_\bigstar$'+'/M'+r'$_{\odot}$'+')',
                    size=14)
                ax.set_ylabel('log(sSFR[H'+r'$\alpha$'+']'+' yr'+
                    r'$^{-1}$'+')', size=14)
            else: #i==1
                filtbins_ssfr.append(ssfr[filt_match])
                ax.scatter(np.log10(1+zspec00[filt_match]), ssfr[filt_match],
                           facecolors='none', edgecolors=cc, linewidth=0.5)
                filtbins_1z.append(np.log10(1+zspec00[filt_match]))

                ax.plot(np.mean(np.log10(1+zspec00[filt_match])), 
                    np.mean(ssfr[filt_match]),'ko', ms=10)
                ax.set_xlabel(r'$\log(1+z)$', size=14)

                # ax.scatter(zspec00[filt_match], ssfr[filt_match],
                #            facecolors='none', edgecolors=cc, linewidth=0.5)
                # filtbins_1z.append(zspec00[filt_match])
                
                # ax.plot(np.mean(zspec00[filt_match]), 
                #     np.mean(ssfr[filt_match]),'ko', ms=10)
                # ax.set_xlabel(r'$z$', size=14)

    axes[0].legend(loc='upper left', fontsize=12, frameon=False)
    axes[0].set_ylim(ymax=-6.9)
    
    bestfit_zssfr(axes[1], filtbins_1z, filtbins_ssfr, delta_sfrs)
    f.subplots_adjust(wspace=0.01)
    [a.tick_params(axis='both', labelsize='10', which='both', direction='in')
        for a in f.axes[:]]
    f.set_size_inches(16,6)


def make_ssfr_graph_newha(f, ax, corr_sfrs, stlr_mass, filts, zspec0, zspec00,
    cwheel, z_arr, corr_tbl,
    ffarr=['NB7', 'NB816', 'NB921', 'NB973', 'NEWHA'],
    llarr=['NB704,NB711', 'NB816', 'NB921', 'NB973', 'NewH'+r'$\alpha$']):
    '''
    plots a two-panel plot of sSFR as a function of mass (LHS) and redshift
    (RHS). colors differ depending on the associated filter of the source.
        note: currently plots log(1+z) rather than z for the RHS panel

    calls bestfit_zssfr() to plot the best-fit line of sSFR as a function of
    redshift and return those parameters as well.
    '''
    # getting MACT+NewHa data
    from MACT_utils import combine_mact_newha
    (sfrs_with_newha, mass_with_newha, zspec_with_newha,
        zspec_with_newha00, filts_with_newha, mz_data_with_newha,
        no_spectra, yes_spectra, z_arr, cwheel) = combine_mact_newha(corr_tbl)

    ssfrs_with_newha = sfrs_with_newha - mass_with_newha
    
    labelarr = []
    mact_masses, mact_ssfrs, mact_mean_ssfr = [], [], []
    m_turnover = 8.5
    for ff,cc,ll,zz,zo,al in zip(ffarr, cwheel, llarr, z_arr,
        [2,2,2,2,1], [0.3,0.3,0.3,0.3,0.2]):
        filt_match = np.array([x for x in range(len(filts_with_newha)) if
            ((ff in filts_with_newha[x]) and (mass_with_newha[x] <= m_turnover)) ])
        filt_index_n = get_filt_index(no_spectra, ff, filts_with_newha)
        filt_index_y = get_filt_index(yes_spectra, ff, filts_with_newha)

        temp = ax.scatter(mass_with_newha[yes_spectra][filt_index_y],
            ssfrs_with_newha[yes_spectra][filt_index_y], 
            facecolors=cc, edgecolors='none', alpha=al, zorder=zo,
            label='z~'+np.str(zz)+' ('+ll+')')

        if ff!='NEWHA':
            mean_mass = np.mean(mass_with_newha[filt_match])
            mean_ssfr = np.mean(ssfrs_with_newha[filt_match])
            # if ff != 'NB973':
            #     ax.plot(mean_mass, mean_ssfr, 'k*', ms=15.0)
            mact_masses += list(mass_with_newha[filt_match])
            mact_ssfrs += list(ssfrs_with_newha[filt_match])
            mact_mean_ssfr.append(mean_ssfr)

            ax.scatter(mass_with_newha[no_spectra][filt_index_n],
                ssfrs_with_newha[no_spectra][filt_index_n], 
                facecolors='none', edgecolors=cc, alpha=al, linewidth=0.5,
                zorder=zo, label='z~'+np.str(zz)+' ('+ll+')')

        labelarr.append(temp)

    plot_avg_sfrs(ax, mass_with_newha[yes_spectra], ssfrs_with_newha[yes_spectra],
        newha=True, cc='k', aa=0.8, zz=4)
    plot_avg_sfrs(ax, mass_with_newha, ssfrs_with_newha,
        newha=True, openz=True, cc='gray', aa=0.7, zz=1)
    # fitting a constant line to the mact sources below 10^9 M*
    # const = np.mean(mact_mean_ssfr[:-1])
    # print('C =', const)
    # tmp_xarr = np.linspace(6.0,m_turnover,30)
    # ax.plot(tmp_xarr, np.array([const]*len(tmp_xarr)), 'k--', lw=2)
    # a linear line to the composites above 10^8 M*
    # def line2(x, m):
    #     return m*(x-m_turnover)+const
    # def salim07(mass):
    #     return -0.35*(mass - 10) - 9.83
    # highm_ii = np.array([x for x in range(len(mass_with_newha)) 
    #     if mass_with_newha[x] >= m_turnover])
    # coeffs1, covar = curve_fit(line2,
    #     mass_with_newha[highm_ii], ssfrs_with_newha[highm_ii])
    # print('m =',coeffs1[0], '& b =', coeffs1[0]*-8+const)
    # tmp_xarr2 = np.linspace(m_turnover,12.0,30)
    # ax.plot(tmp_xarr2, line2(tmp_xarr2, *coeffs1), 'k--', lw=2)
    # ax.plot(tmp_xarr2, salim07(tmp_xarr2), 'c', ls='--', lw=2)

    # finishing touches
    ax.legend(handles=labelarr, loc='lower left', fontsize=14, frameon=False)
    ax.set_xlabel('log(M'+r'$_\bigstar$'+'/M'+r'$_{\odot}$'+')',
        size=14)
    ax.set_ylabel('log(sSFR[H'+r'$\alpha$'+']'+' yr'+
        r'$^{-1}$'+')', size=14)
    
    ax.set_ylim(ymax=-6.9)    
    ax.tick_params(axis='both', labelsize='10', which='both', direction='in')
    f.set_size_inches(8,6.5)


def approximated_zspec0(zspec0, filts):
    '''
    modifying zspec0 such that all invalid zspec vals are replaced by
    approximate values from the filters

    this is for obtaining the best-fit line params for 
    make_redshift_graph() and make_ssfr_graph()
    '''
    zspec00 = np.copy(zspec0)

    badz_iis = np.array([x for x in range(len(zspec00)) if zspec00[x] < 0
        or zspec00[x] > 9])
    filt_lambda_list = {'NB704':7045.0, 'NB711':7126.0, 'NB816':8152.0,
        'NB921':9193.0, 'NB973':9749.0}

    for ff in filt_lambda_list.keys():
        badf_match = np.where(filts[badz_iis] == ff)[0]
        zspec00[badz_iis[badf_match]] = (filt_lambda_list[ff]/HA) - 1

    return zspec00


def main():
    '''
    Reads in data from the MACT dataset, and obtains the useful data
    (defined by good_sig_iis). Then, various plotting functions are called.
    Figures are then saved and closed.
    '''
    # reading in data generated by EBV_corrections.py
    corr_tbl = asc.read(FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt',
        guess=False, Reader=asc.FixedWidthTwoLine)

    # defining a flux sigma and mass cutoff
    good_sig_iis = np.where((corr_tbl['flux_sigma'] >= CUTOFF_SIGMA) & 
        (corr_tbl['stlr_mass'] >= CUTOFF_MASS))[0]
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

    # defining useful data structs for plotting
    filtarr = np.array(['NB704,NB711', 'NB816', 'NB921', 'NB973'])
    markarr = np.array(['o', '^', 'D', '*'])
    sizearr = np.array([6.0, 6.0, 6.0, 9.0])**2
    z_arr = get_z_arr()
    cwheel = [np.array(mpl.rcParams['axes.prop_cycle'])[x]['color']
        for x in range(4)] # getting colorwheel
    # for obtaining the best-fit line params
    zspec00 = approximated_zspec0(zspec0, filts)

    from MACT_utils import get_FUV_corrs
    FUV_corr_factor = get_FUV_corrs(corr_tbl)

    # print('making 4-panel mainseq plot now' # (with 'all' types of corrs))
    # f_all, ax_all = plt.subplots(2,2)
    # axarr = np.ndarray.flatten(ax_all)
    # f_all.set_size_inches(14,14)
    # for title, corrs, ax, i in zip(['(a) Observed', '(b) Filter+[N II]',
    #     '(c) Filter+[N II]+Dust Attenuation', '(d) Filter+[N II]+Dust Attenuation+FUV'],
    #     [np.zeros(len(good_sig_iis)),
    #         filt_corr_factor+nii_ha_corr_factor,
    #         filt_corr_factor+nii_ha_corr_factor+dust_corr_factor,
    #         filt_corr_factor+nii_ha_corr_factor+dust_corr_factor+FUV_corr_factor],
    #     axarr, range(4)):

    #     #  should pass in e.g., "obs_sfr + corrs" to plot applied corrs
    #     make_all_graph(stlr_mass, obs_sfr+corrs, filtarr, markarr, z_arr, sizearr,
    #         title, no_spectra, yes_spectra, filts, ax, i, corr_tbl)
    #     print('done plotting', title)

    # [a.tick_params(axis='both', labelsize='10', which='both', direction='in')
    #     for a in f_all.axes[:]]
    # plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.04, right=0.99,
    #     top=0.99, bottom=0.04)
    # plt.savefig(FULL_PATH+'Plots/main_sequence/mainseq.pdf')
    # plt.close()


    # if mainseq_fig4_only:
    #     print('making 1-panel mainseq plot now (with only \'all\' corrs)')
    #     i=5
    #     corrs = filt_corr_factor+nii_ha_corr_factor+dust_corr_factor+FUV_corr_factor
    #     f, ax = plt.subplots()
    #     make_all_graph(stlr_mass, obs_sfr+corrs, filtarr, markarr, z_arr, sizearr,
    #         title, no_spectra, yes_spectra, filts, ax, i)
    #     ax.tick_params(axis='both', labelsize='10', which='both',
    #         direction='in')
    #     f.set_size_inches(8,8)
    #     plt.tight_layout()
    #     plt.savefig(FULL_PATH+'Plots/main_sequence/mainseq_allcorrs.pdf')
    #     plt.close()


    NEWHA = True
    
    # print('making zdep plot')
    # f, ax = plt.subplots()
    # make_redshift_graph(f, ax, z_arr, corr_sfrs+FUV_corr_factor, delta_sfrs,
    #     stlr_mass, zspec00, filts, no_spectra, yes_spectra, cwheel)
    # plt.subplots_adjust(hspace=0.01, wspace=0.01, right=0.99, top=0.98,
    #     left=0.1, bottom=0.09)
    # plt.ylim([-3.8,1.8])
    # plt.savefig(FULL_PATH+'Plots/main_sequence/zdep_mainseq.pdf')
    # plt.close()

    print('making sSFR plot now')
    if NEWHA:
        f, ax = plt.subplots()
        make_ssfr_graph_newha(f, ax, corr_sfrs+FUV_corr_factor,
            stlr_mass, filts, zspec0, zspec00, cwheel, z_arr, corr_tbl=corr_tbl)
        plt.subplots_adjust(right=0.98, top=0.98, left=0.08, bottom=0.08)
        plt.savefig(FULL_PATH+'Plots/main_sequence/mainseq_sSFRs_FUV_corrs.pdf')

    # print('making old sSFR plot now')
    # lowm_ii = np.arange(len(corr_sfrs))
    # f, axes = plt.subplots(1,2, sharey=True)
    # make_ssfr_graph_old(f, axes, corr_sfrs[lowm_ii]+FUV_corr_factor[lowm_ii],
    #     delta_sfrs[lowm_ii], stlr_mass[lowm_ii], filts[lowm_ii], zspec00[lowm_ii], cwheel, z_arr)
    # plt.subplots_adjust(right=0.99, top=0.98, left=0.05, bottom=0.09)
    # plt.savefig(FULL_PATH+'Plots/main_sequence/mainseq_sSFRs_fuvz_tmp.pdf')


if __name__ == '__main__':
    main()
