"""
NAME:
    plot_nbia_mainseq.py

PURPOSE:

    depends on mainseq_corrections.py, 

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

import numpy as np, matplotlib.pyplot as plt
import scipy.optimize as optimize
import matplotlib as mpl
from astropy.io import ascii as asc
from analysis.composite_errors import compute_onesig_pdf

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


def delosreyes_2015(ax):
    '''
    Plots the z~0.8 data points from de Los Reyes+15 in cyan.
    '''
    xarr = np.array([9.27, 9.52, 9.76, 10.01, 10.29, 10.59, 10.81, 11.15])
    yarr = np.array([0.06, 0.27, 0.43, 0.83, 1.05, 1.18, 1.50, 1.54])
    yerr = np.array([0.454, 0.313, 0.373, 0.329, 0.419, 0.379, 0.337, 0.424])

    ax.errorbar(xarr, yarr, yerr, fmt='deepskyblue', ecolor='deepskyblue', zorder=2) 
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
    ax.fill_between(xarr, lowlim, uplim, color='gray', alpha=0.4)
    salim, = ax.plot(xarr, salim_line(xarr), 'k-',
                      label='Salim+07 (z~0)', zorder=1, lw=0.5)
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
    ax.plot(xarr, xarr - 8, 'k:',zorder=8)#, alpha=.6)
    ax.text(5.85, -1.4, 'sSFR=(0.1 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)
    ax.plot(xarr, xarr - 9, 'k:',zorder=8)#, alpha=.6)
    ax.text(5.85, -2.4, 'sSFR=(1.0 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)
    ax.plot(xarr, xarr - 10, 'k:')#, alpha=.5)
    ax.text(6.17, -3.04, 'sSFR=(10.0 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)
    ax.plot(xarr, xarr - 11, 'k:')#, alpha=.5)
    ax.text(7.15, -3.0, 'sSFR=(100.0 Gyr)'+r'$^{-1}$', rotation=42, color='k',
             alpha=1, fontsize=9)


def modify_all_graph(ax, labelarr, xlim, ylim, title, i):
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
    if i%2==0 or i==5:
        ax.set_ylabel('log(SFR[H'+r'$\alpha$'+']/M'+r'$_{\odot}$'+
            ' yr'+r'$^{-1}$'+')', size=14)
    if i%2==1 and i!=5:
        ax.set_yticklabels([])
    if i<2:
        ax.set_xticklabels([])
    ax.text(0.02, 0.95, title, transform=ax.transAxes, color='k', fontsize=14,
        fontweight='bold')
    
    noeske = noeske_2007(ax)
    delosreyes = delosreyes_2015(ax)
    whitaker = whitaker_2014(ax)
    salim = salim_2007(ax)
    berg = berg_2012(ax)

    if i==0 or i==5:
        legend1 = ax.legend(handles=list(labelarr), loc=(0.01, 0.78),
            frameon=False, fontsize=11, scatterpoints=1, numpoints=1)
        ax.add_artist(legend1)
        
        labelarr2 = np.array([whitaker, delosreyes, noeske, salim, berg])
        legend2 = ax.legend(handles=list(labelarr2), loc='lower right',
            frameon=False, fontsize=11, scatterpoints=1, numpoints=1)
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
        sfrs_matched = sfrs[bin_match]
        ax.plot(mbins0[i], np.mean(sfrs_matched), 'ko', alpha=0.8, ms=8)
        ax.errorbar(mbins0[i], np.mean(sfrs_matched), xerr=0.25, fmt='none',
            ecolor='black', alpha=0.8, lw=2)

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
    '''
    if 'NB7' in ff:
        filt_index = np.array([x for x in range(len(spectra)) if
            ff[:3] in filts[spectra][x]])
    else:
        filt_index = np.array([x for x in range(len(spectra)) if
            ff==filts[spectra][x]])

    return filt_index


def make_all_graph(stlr_mass, sfr, filtarr, markarr, z_arr, sizearr, title,
    no_spectra, yes_spectra, filts, ax, i):
    '''
    '''
    color='blue'
    xlim = [5.80, 11.20]
    ylim = [-3.75, 2]

    labelarr = np.array([])
    check_nums = []
    for (ff, mark, avg_z, size) in zip(filtarr, markarr, z_arr, sizearr):
        filt_index_n = get_filt_index(no_spectra, ff, filts)
        filt_index_y = get_filt_index(yes_spectra, ff, filts)

        print '>>>', ff, avg_z
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
    #endfor
    assert np.sum(check_nums)==len(sfr)
    plot_avg_sfrs(ax, stlr_mass, sfr)

    modify_all_graph(ax, labelarr, xlim, ylim, title, i)


def plot_zdep_avg_sfrs(ax, stlr_mass, sfrs, cc):
    '''
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
    '''
    if fittype=='first_order':
        eqn0 = r'$log(SFR) = \alpha log(M) + \beta z + \gamma$'
        def func0(data, a, b, c):
            return a*data[:,0] + b*data[:,1] + c

    elif fittype=='second_order':
        eqn0 = r'$log(SFR) = \alpha\' log(M)^2 + \alpha log(M) + \beta z + \gamma$'
        def func0(data, aprime, a, b, c):
            return aprime*data[:,0]**2 + a*data[:,0] + b*data[:,1] + c

    else:
        raise ValueError('invalid fit type')

    return func0, eqn0


def modify_redshift_graph(f, ax, fittype, eqn0, params):
    '''
    '''
    ax.set_xlabel('log(M'+r'$_\bigstar$'+'/M'+r'$_{\odot}$'+')', size=14)
    ax.set_ylabel('log(SFR[H'+r'$\alpha$'+']/M'+r'$_{\odot}$'+
        ' yr'+r'$^{-1}$'+')', size=14)
    ax.legend(loc='upper left', fontsize=14, frameon=False)

    if fittype=='first_order':
        ax.text(0.50,0.12,eqn0+
            '\n'+r'$\alpha=$'+'%.2f'%(params[0])+', '+r'$\beta=$'+
            '%.2f'%(params[1])+', '+r'$\gamma=$'+'%.2f'%(params[2]),
            transform=ax.transAxes, fontsize=15, ha='left', va='top')

    elif fittype=='second_order':
        ax.text(0.50,0.12,eqn0+
            '\n'+r'$\alpha\'=$'+'%.2f'%(params[0])+', '+r'$\alpha=$'+
            '%.2f'%(params[1])+', '+r'$\beta=$'+'%.2f'%(params[2])+
            ', '+r'$\gamma=$'+'%.2f'%(params[3]),
            transform=ax.transAxes,fontsize=15, ha='left', va='top')

    else:
        raise ValueError('invalid fit type')

    [a.tick_params(axis='both', labelsize='10', which='both', direction='in')
        for a in f.axes[:]]
    f.set_size_inches(7,6)


def make_redshift_graph(f, ax, z_arr, corr_sfrs, stlr_mass, zspec0, filts,
    good_sig_iis, cwheel, ffarr=['NB7', 'NB816', 'NB921', 'NB973'],
    llarr=['NB704,NB711', 'NB816', 'NB921', 'NB973'], fittype='first_order'):
    '''
    '''
    func0, eqn0 = get_func0_eqn0(fittype)

    # getting relevant data in a good format
    ffs = filts[good_sig_iis]
    sfrs00 = corr_sfrs[good_sig_iis]
    smass0 = stlr_mass[good_sig_iis]
    zspec0 = zspec0[good_sig_iis]
    no_spectra  = np.where((zspec0 <= 0) | (zspec0 > 9))[0]
    yes_spectra = np.where((zspec0 >= 0) & (zspec0 < 9))[0]

    centr_filts = {'NB7':((7045.0/HA - 1) + (7126.0/HA - 1))/2.0, 
        'NB816':8152.0/HA - 1, 'NB921':9193.0/HA - 1, 'NB973':9749.0/HA - 1,
        'NEWHA':0.8031674}


    # for obtaining the best-fit line params
    badz_iis = np.array([x for x in range(len(zspec0)) if zspec0[x] < 0 or zspec0[x] > 9])
    filt_lambda_list = {'NB704':7045.0, 'NB711':7126.0, 'NB816':8152.0, 'NB921':9193.0, 'NB973':9749.0}
    for ff in filt_lambda_list.keys():
        badf_match = np.where(ffs[badz_iis] == ff)[0]
        zspec0[badz_iis[badf_match]] = (filt_lambda_list[ff]/HA) - 1
    
    data00 = np.vstack([smass0, zspec0]).T

    # ----- no longer need good_sig_iis

    params, pcov = optimize.curve_fit(func0, data00, sfrs00, method='lm')
    perr = np.sqrt(np.diag(pcov))


    for ff,cc,ll,zz in zip(ffarr[::-1], cwheel[::-1], llarr[::-1], z_arr[::-1]):
        if 'NB7' in ff:
            filt_index_n = np.array([x for x in range(len(no_spectra)) if ff[:3] in ffs[no_spectra][x]])
            filt_index_y = np.array([x for x in range(len(yes_spectra)) if ff[:3] in ffs[yes_spectra][x]])
        else:
            filt_index_n = np.array([x for x in range(len(no_spectra)) if ff==ffs[no_spectra][x]])
            filt_index_y = np.array([x for x in range(len(yes_spectra)) if ff==ffs[yes_spectra][x]])

            # if ff=='NEWHA': print filt_index_y, filt_index_n

        # scattering
        ax.scatter(smass0[yes_spectra][filt_index_y], sfrs00[yes_spectra][filt_index_y],
            facecolors=cc, edgecolors='none', alpha=0.3,
            zorder=3, label='z~'+zz+' ('+ll+')')
        if ff != 'NEWHA':
            ax.scatter(smass0[no_spectra][filt_index_n], sfrs00[no_spectra][filt_index_n],
                facecolors='none', edgecolors=cc, alpha=0.3, 
                linewidth=0.5, zorder=3)

        # plotting the best-fit lines
        filt_match = np.array([x for x in range(len(ffs)) if ff in ffs[x]])
        mrange = np.arange(min(smass0[filt_match]), max(smass0[filt_match]), 0.1)
        avgz = np.array([centr_filts[ff]]*len(mrange))
        tmpdata = np.vstack([mrange, avgz]).T
        ax.plot(mrange, func0(tmpdata, *params), color=cc, lw=2)

        plot_zdep_avg_sfrs(ax, smass0[filt_match], sfrs00[filt_match], cc)

    modify_redshift_graph(f, ax, fittype, eqn0, params)


def bestfit_zssfr(ax, tmpzarr0, tmpsarr0):
    '''
    '''
    def line(x, m, b):
        return m*x + b
    tempparams, temppcov = optimize.curve_fit(line, tmpzarr0, tmpsarr0)
    print 'sSFR a*log(1+z)+b params:', tempparams

    stp = 0.02
    xrange_tmp = np.arange(min(tmpzarr0)-stp, max(tmpzarr0)+2*stp, stp)
    ax.plot(xrange_tmp,  line(xrange_tmp, *tempparams), 'k--')


def make_ssfr_graph(f, axes, sfrs00, smass0, filts00, zspec00, cwheel, z_arr,
    ffarr=['NB7', 'NB816', 'NB921', 'NB973'],
    llarr=['NB704,NB711', 'NB816', 'NB921', 'NB973']):
    '''
    '''
    ssfr = sfrs00-smass0
    tmpzarr0, tmpsarr0 = [], []
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
                ax.set_ylabel(r'sSFR', size=14)
            else: #i==1
                # ax.scatter(zspec00[filt_match], ssfr[filt_match],
                #            facecolors='none', edgecolors=cc, linewidth=0.5)
                ax.scatter(np.log10(1+zspec00[filt_match]), ssfr[filt_match],
                           facecolors='none', edgecolors=cc, linewidth=0.5)
                zmean = np.mean(np.log10(1+zspec00[filt_match]))
                smean = np.mean(ssfr[filt_match])
                tmpzarr0.append(zmean)
                tmpsarr0.append(smean)
                ax.plot(zmean, smean,'ko', ms=10)
                ax.set_xlabel(r'$\log(1+z)$', size=14)

    axes[0].legend(loc='upper left', fontsize=12, frameon=False)
    axes[0].set_ylim(ymax=-6.9)
    bestfit_zssfr(axes[1], tmpzarr0, tmpsarr0)
    f.subplots_adjust(wspace=0.01)
    [a.tick_params(axis='both', labelsize='10', which='both', direction='in')
        for a in f.axes[:]]
    f.set_size_inches(16,6)


def get_z_arr():
    '''
    '''
    # defining an approximate redshift array for plot visualization
    z_arr0 = np.array([7045.0, 7126.0, 8152.0, 9193.0, 9749.0])/HA - 1
    z_arr0 = np.around(z_arr0, 2)
    z_arr  = np.array(z_arr0, dtype='|S9')
    z_arr[0] = ",".join(z_arr[:2])
    z_arr = np.delete(z_arr, 1)
    z_arr  = np.array([x+'0' if len(x)==3 else x for x in z_arr])
    return z_arr


def main():
    '''
    '''
    # reading in data generated by EBV_corrections.py
    corr_tbl = asc.read(FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt',
        guess=False, Reader=asc.FixedWidthTwoLine)

    # defining a flux sigma and mass cutoff
    good_sig_iis = np.where((corr_tbl['flux_sigma'] >= CUTOFF_SIGMA) & 
        (corr_tbl['stlr_mass'] >= CUTOFF_MASS))[0]

    # getting/storing useful data
    zspec0 = np.array(corr_tbl['zspec0'])[good_sig_iis]
    no_spectra  = np.where((zspec0 <= 0) | (zspec0 > 9))[0]
    yes_spectra = np.where((zspec0 >= 0) & (zspec0 < 9))[0]

    stlr_mass = np.array(corr_tbl['stlr_mass'])[good_sig_iis]
    filts = np.array(corr_tbl['filt'])[good_sig_iis]
    sfr = np.array(corr_tbl['met_dep_sfr'])[good_sig_iis]
    dust_corr_factor = np.array(corr_tbl['dust_corr_factor'])[good_sig_iis]
    filt_corr_factor = np.array(corr_tbl['filt_corr_factor'])[good_sig_iis]
    nii_ha_corr_factor = np.array(corr_tbl['nii_ha_corr_factor'])[good_sig_iis]

    # defining useful data structs for plotting
    filtarr = np.array(['NB704,NB711', 'NB816', 'NB921', 'NB973'])
    markarr = np.array(['o', '^', 'D', '*'])
    sizearr = np.array([6.0, 6.0, 6.0, 9.0])**2
    z_arr = get_z_arr()


    print 'making 4-panel mainseq plot now' # (with 'all' types of corrs)
    f_all, ax_all = plt.subplots(2,2)
    axarr = np.ndarray.flatten(ax_all)
    f_all.set_size_inches(14,14)
    for title, corrs, ax, i in zip(['(a) Observed', '(b) Filter-corrected',
        '(c) Filter+[N II]', '(d) Filter+[N II]+Dust Attenuation'], 
        [np.zeros(len(good_sig_iis)), filt_corr_factor,
            filt_corr_factor+nii_ha_corr_factor,
            filt_corr_factor+nii_ha_corr_factor+dust_corr_factor],
        axarr, range(4)):

        #  should pass in e.g., "sfr + corrs" to plot applied corrs
        make_all_graph(stlr_mass, sfr+corrs, filtarr, markarr, z_arr, sizearr,
            title, no_spectra, yes_spectra, filts, ax, i)
        print 'done plotting', title

    [a.tick_params(axis='both', labelsize='10', which='both', direction='in')
        for a in f_all.axes[:]]
    plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.04, right=0.99,
        top=0.99, bottom=0.04)
    plt.savefig(FULL_PATH+'Plots/main_sequence/mainseq.pdf')
    plt.close()


    if mainseq_fig4_only:
        print 'making 1-panel mainseq plot now (with only \'all\' corrs)'
        i=5
        f, ax = plt.subplots()
        make_all_graph(stlr_mass, sfr+corrs, filtarr, markarr, z_arr, sizearr,
            title, no_spectra, yes_spectra, filts, ax, i)
        ax.tick_params(axis='both', labelsize='10', which='both',
            direction='in')
        f.set_size_inches(8,8)
        plt.tight_layout()
        plt.savefig(FULL_PATH+'Plots/main_sequence/mainseq_allcorrs.pdf')
        plt.close()


    # getting colorwheel
    cwheel = [np.array(mpl.rcParams['axes.prop_cycle'])[x]['color']
        for x in range(4)]

    print 'making redshift dependent plot now'
    f, ax = plt.subplots()
    corr_sfrs = sfr+filt_corr_factor+nii_ha_corr_factor+dust_corr_factor

    # variable00 means we've down-selected to good_sig_iis only
    # TODO: make this consistent across all plotting types...
    sfrs00 = corr_sfrs[good_sig_iis]
    smass00 = stlr_mass[good_sig_iis]
    filts00 = filts[good_sig_iis]
    zspec00 = zspec0[good_sig_iis]

    make_redshift_graph(f, ax, z_arr, corr_sfrs, stlr_mass, zspec0, filts, good_sig_iis, cwheel)
    plt.subplots_adjust(hspace=0.01, wspace=0.01, right=0.99, top=0.98, left=0.1, bottom=0.09)
    plt.savefig(FULL_PATH+'Plots/main_sequence/zdep_mainseq.pdf')
    plt.close()

    print 'making sSFR plot now'
    f, axes = plt.subplots(1,2, sharey=True)
    sfrs00 = corr_sfrs[good_sig_iis]
    smass0 = corr_tbl['stlr_mass'][good_sig_iis].data
    filts00 = corr_tbl['filt'][good_sig_iis].data
    zspec00 = corr_tbl['zspec0'][good_sig_iis].data
    no_spectra  = np.where((zspec00 <= 0) | (zspec00 > 9))[0]
    yes_spectra = np.where((zspec00 >= 0) & (zspec00 < 9))[0]
    badz_iis = np.array([x for x in range(len(zspec00)) if zspec00[x] < 0 or zspec00[x] > 9])
    filt_lambda_list = {'NB704':7045.0, 'NB711':7126.0, 'NB816':8152.0, 'NB921':9193.0, 'NB973':9749.0}
    ffs = filts[good_sig_iis]
    for ff in filt_lambda_list.keys():
        badf_match = np.where(ffs[badz_iis] == ff)[0]
        zspec00[badz_iis[badf_match]] = (filt_lambda_list[ff]/HA) - 1

    make_ssfr_graph(f, axes, sfrs00, smass0, filts00, zspec00, cwheel, z_arr)
    plt.subplots_adjust(right=0.99, top=0.98, left=0.05, bottom=0.09)
    plt.savefig(FULL_PATH+'Plots/main_sequence/mainseq_sSFRs.pdf')


if __name__ == '__main__':
    main()
