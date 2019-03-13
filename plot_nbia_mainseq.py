"""
NAME:
    plot_nbia_mainseq.py

PURPOSE:

    depends on 

INPUTS:
    FULL_PATH+'Main_Sequence/Berg2012_table.clean.txt'
    FULL_PATH+'Main_Sequence/Noeske07_fig1_z1.txt'
    FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt'

OUTPUTS:
    FULL_PATH+'Plots/main_sequence/ALL.pdf'
"""

import numpy as np, matplotlib.pyplot as plt
from astropy.io import ascii as asc

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'
CUTOFF_SIGMA = 4.0
CUTOFF_MASS = 6.0

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


def delosreyes_2015(ax):
    '''
    Plots the z~0.8 data points from de Los Reyes+15 in cyan.
    '''
    xarr = np.array([9.27, 9.52, 9.76, 10.01, 10.29, 10.59, 10.81, 11.15])
    yarr = np.array([0.06, 0.27, 0.43, 0.83, 1.05, 1.18, 1.50, 1.54])
    yerr = np.array([0.454, 0.313, 0.373, 0.329, 0.419, 0.379, 0.337, 0.424])

    ax.errorbar(xarr, yarr, yerr, fmt='c', ecolor='c', zorder=2) 
    delosreyes = ax.scatter(xarr, yarr, color='c', marker='s',
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

    ax.plot(xarr, lowlim, 'k--', zorder=1)
    ax.plot(xarr, uplim, 'k--', zorder=1)
    ax.fill_between(xarr, lowlim, uplim, color='gray', alpha=0.4)
    salim, = ax.plot(xarr, salim_line(xarr), 'k-',
                      label='Salim+07 (z~0)', zorder=1)
    return salim


def berg_2012(ax):
    '''
    Plots the log(M*)-log(SFR) relation from Berg+12 in green. (ASCII file
    provided by Chun Ly)
    '''
    berg = asc.read(FULL_PATH+'Main_Sequence/Berg2012_table.clean.txt',guess=False,
                    Reader=asc.CommentedHeader,delimiter='\t')
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
    noeske = asc.read(FULL_PATH+'Main_Sequence/Noeske07_fig1_z1.txt',guess=False,
                      Reader=asc.NoHeader)
    logM   = np.array(noeske['col1'])
    logSFR = np.array(noeske['col2'])
    logSFR_low  = np.array(noeske['col3'])
    logSFR_high = np.array(noeske['col4'])

    ax.plot(logM, logSFR_low, color='orange', marker='', linestyle='',
             zorder=1)
    ax.plot(logM, logSFR_high, color='orange', marker='', linestyle='',
             zorder=1)
    ax.fill_between(logM, logSFR_low, logSFR_high, facecolor='none',
                     hatch=3*'.', edgecolor='orange', linewidth=0.0,
                     zorder=1)
    noeske, = ax.plot(logM, logSFR, color='orange', marker='+', #linestyle='', 
                       label='Noeske+07 (0.20<z<0.45)',zorder=1, mew=2, markersize=11)
    return noeske


def sSFR_lines(ax, xlim):
    '''
    Creates the three dotted sSFR^-1 lines: 1, 10, and 100 Gyr
    '''
    xmin = min(xlim)
    xmax = max(xlim)
    xarr = np.arange(xmin, xmax, 0.01)
    ax.plot(xarr, xarr - 9, 'k:',zorder=8)#, alpha=.6)
    ax.text(5.85, -2.4, 'sSFR=(1.0 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)
    ax.plot(xarr, xarr - 10, 'k:')#, alpha=.5)
    ax.text(6.17, -3.04, 'sSFR=(10.0 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)
    ax.plot(xarr, xarr - 11, 'k:')#, alpha=.5)
    ax.text(7.15, -3.0, 'sSFR=(100.0 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)
    

def modify_graph(ax, labelarr, xlim, ylim, title, i):
    '''
    Modifies the graph (gives it limits, labels, a title, etc.)
    
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
    if i%2==0:
        ax.set_ylabel('log(SFR[H'+r'$\alpha$'+']/M'+r'$_{\odot}$'+' yr'+r'$^{-1}$'+')', size=14)
    if i%2==1:
        ax.set_yticklabels([])
    if i<2:
        ax.set_xticklabels([])
    ax.text(0.02, 0.95, title, transform=ax.transAxes, color='k', fontsize=14, fontweight='bold')
    
    noeske = noeske_2007(ax)
    delosreyes = delosreyes_2015(ax)
    whitaker = whitaker_2014(ax)
    salim = salim_2007(ax)
    berg = berg_2012(ax)
    
    labelarr2 = np.array([whitaker, delosreyes, noeske, salim, berg])

    if i==0:
        legend1 = ax.legend(handles=list(labelarr), loc=(0.01, 0.78), frameon=False,
                         fontsize=11, scatterpoints=1, numpoints=1)
        ax.add_artist(legend1)
        legend2 = ax.legend(handles=list(labelarr2), loc='lower right', frameon=False,
                         fontsize=11, scatterpoints=1, numpoints=1)
        ax.add_artist(legend2)

    ax.minorticks_on()
    sSFR_lines(ax, xlim)


def plot_avg_sfrs(ax, stlr_mass, sfrs):
    '''
    assumes the lowest mass is at m=6
    plots the mean sfr in each mass bin of width 0.5
    '''
    mbins0 = np.arange(6.25, 10.75, .5)
    bin_ii = np.digitize(stlr_mass, mbins0+0.25)

    for i in range(len(mbins0)):
        bin_match = np.where(bin_ii == i)[0]
        ax.plot(mbins0[i], np.mean(sfrs[bin_match]), 'ko', alpha=0.8, ms=8)
        ax.errorbar(mbins0[i], np.mean(sfrs[bin_match]), xerr=0.25, fmt='none', ecolor='black', alpha=0.8, lw=2)


def make_all_graph(stlr_mass, sfr, filtarr, markarr, z_arr, sizearr, title,
    no_spectra, yes_spectra, filts, good_sig_iis, ax, i):
    '''
    '''
    color='blue'
    xlim = [5.80, 11.20]
    ylim = [-3.75, 2]

    labelarr = np.array([])
    for (ff, mark, avg_z, size) in zip(filtarr, markarr, z_arr, sizearr):
        if 'NB7' in ff:
            filt_index_n = np.array([x for x in range(len(no_spectra)) if ff[:3] in filts[no_spectra][x] 
                and no_spectra[x] in good_sig_iis])
            filt_index_y = np.array([x for x in range(len(yes_spectra)) if ff[:3] in filts[yes_spectra][x]
                and yes_spectra[x] in good_sig_iis])
        else:
            filt_index_n = np.array([x for x in range(len(no_spectra)) if ff==filts[no_spectra][x] 
                and no_spectra[x] in good_sig_iis])
            filt_index_y = np.array([x for x in range(len(yes_spectra)) if ff==filts[yes_spectra][x]
                and yes_spectra[x] in good_sig_iis])

        print '>>>', ff, avg_z

        temp = ax.scatter(stlr_mass[yes_spectra][filt_index_y],
                       sfr[yes_spectra][filt_index_y], marker=mark,
                       facecolors=color, edgecolors='none', alpha=0.3,
                       zorder=3, label='z~'+np.str(avg_z)+' ('+ff+')', s=size)

        ax.scatter(stlr_mass[no_spectra][filt_index_n], 
                        sfr[no_spectra][filt_index_n],
                        marker=mark, facecolors='none', edgecolors=color, alpha=0.3, 
                        linewidth=0.5, zorder=3, label='z~'+np.str(avg_z)+' ('+ff+')', s=size)
        
        labelarr = np.append(labelarr, temp)
    #endfor
    plot_avg_sfrs(ax, stlr_mass[good_sig_iis], sfr[good_sig_iis])

    modify_graph(ax, labelarr, xlim, ylim, title, i)


def main():
    '''
    '''
    # reading in data generated by EBV_corrections.py
    corr_tbl = asc.read(FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt',guess=False,
                    Reader=asc.FixedWidthTwoLine)
    zspec0 = np.array(corr_tbl['zspec0'])
    no_spectra  = np.where((zspec0 <= 0) | (zspec0 > 9))[0]
    yes_spectra = np.where((zspec0 >= 0) & (zspec0 < 9))[0]

    stlr_mass = np.array(corr_tbl['stlr_mass'])
    filts = np.array(corr_tbl['filt'])
    sfr = np.array(corr_tbl['met_dep_sfr'])
    dust_corr_factor = np.array(corr_tbl['dust_corr_factor'])
    filt_corr_factor = np.array(corr_tbl['filt_corr_factor'])
    nii_ha_corr_factor = np.array(corr_tbl['nii_ha_corr_factor'])


    # defining useful data structs for plotting
    filtarr = np.array(['NB704,NB711', 'NB816', 'NB921', 'NB973'])
    markarr = np.array(['o', '^', 'D', '*'])
    sizearr = np.array([6.0, 6.0, 6.0, 9.0])**2

    # defining an approximate redshift array for plot visualization
    z_arr0 = np.array([7045.0, 7126.0, 8152.0, 9193.0, 9749.0])/6563.0 - 1
    z_arr0 = np.around(z_arr0, 2)
    z_arr  = np.array(z_arr0, dtype='|S9')
    z_arr[0] = ",".join(z_arr[:2])
    z_arr = np.delete(z_arr, 1)
    z_arr  = np.array([x+'0' if len(x)==3 else x for x in z_arr])

    # defining a flux sigma and mass cutoff
    good_sig_iis = np.where((corr_tbl['flux_sigma'] >= CUTOFF_SIGMA) & (stlr_mass >= CUTOFF_MASS))[0]

    f_all, ax_all = plt.subplots(2,2)
    axarr = np.ndarray.flatten(ax_all)
    f_all.set_size_inches(14,14)
    for title, corrs, ax, i in zip(['(a) Observed', '(b) Filter-corrected', 
                             '(c) Filter+[N II]', '(d) Filter+[N II]+Dust Attenuation'], 
                            [np.zeros(len(corr_tbl)), filt_corr_factor, filt_corr_factor+nii_ha_corr_factor, 
                             filt_corr_factor+nii_ha_corr_factor+dust_corr_factor], axarr, range(4)):
        #  should pass in e.g., "sfr + corrs" to plot applied corrs
        make_all_graph(stlr_mass, sfr+corrs, filtarr, markarr, z_arr, sizearr, title, 
            no_spectra, yes_spectra, filts, good_sig_iis, ax, i)
        print 'done plotting', title

    [a.tick_params(axis='both', labelsize='10', which='both', direction='in') for a in f_all.axes[:]]
    plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.04, right=0.99, top=0.99, bottom=0.04)
    plt.savefig(FULL_PATH+'Plots/main_sequence/mainseq.pdf')
    plt.close()


if __name__ == '__main__':
    main()
