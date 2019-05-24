"""
NAME:
    plotting_NII_Ha_ratios.py

PURPOSE:

    depends on mainseq_corrections.py, stack_spectral_data.py

INPUTS:
    FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt'
    FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt'
    FULL_PATH+'Main_Sequence/mainseq_corrections_tbl_ref.txt'

OUTPUTS:
    FULL_PATH+'Plots/main_sequence/NII_Ha_scatter.pdf'
    FULL_PATH+'Plots/main_sequence/NII_Ha_scatter_log.pdf'
"""

from analysis.composite_errors import composite_errors
from astropy.io import fits as pyfits, ascii as asc
from create_ordered_AP_arrays import create_ordered_AP_arrays
from scipy.optimize import curve_fit
import numpy as np, matplotlib.pyplot as plt
FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'


def niiha_oh_determine(x0, type, index=None, silent=None, linear=None):
    '''
    Adapted from Chun Ly 
    
    PURPOSE:
       This code estimates 12+log(O/H) based on strong-line diagnostics. It uses
       emission-line that use [NII]6583, such as [NII]6583/Halpha.

    CALLING SEQUENCE:
       niiha_oh_determine(x0, type, index=index, silent=1)

    INPUTS:
       x0   -- Array of log([NII]6583/Halpha)
       type -- The type of diagnostics to use. The options are:
         'PP04_N2'    -- N2 index calibration of Pettini & Pagel (2004), MNRAS, 348, 59
           - Specify linear keyword to use linear instead of 3rd-order function

    OPTIONAL KEYWORD INPUT:
       index   -- Index of array to determine metallicity
       silent  -- If set, this means that nothing will be printed out
    '''

    if index is None: index = range(len(x0))

    ## Default sets those without metallicity at -1.0
    OH_gas = np.repeat(-1.000, len(x0))


    ######################################
    ## Empirical, PP04                  ##
    ## ---------------------------------##
    ## See Pettini & Pagel (2004)       ##
    ## Eq. A10 of Kewley & Ellison 2008 ##
    ## + on 04/03/2016                  ##
    ## Mod on 14/06/2016                ##
    ######################################
    if type == 'PP04_N2':
        if linear == None:
            OH_gas[index] = 9.37 + 2.03*x0[index] + 1.26*(x0[index])**2 + 0.32*(x0[index])**3
        else:
            print '## Using linear relation!'
            # Bug found. Mod on 30/06/2016 OH_gas -> OH_gas[index]
            OH_gas[index] = 8.90 + 0.57 * x0[index] #xt0
    #endif

    return OH_gas


def main():
    '''
    '''
    # latex backend for mpl
    # import matplotlib
    # matplotlib.rcParams['text.usetex'] = True
    # matplotlib.rcParams['text.latex.unicode'] = True

    # reading data in
    mmt_mz  = asc.read(FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt',
        guess=False, format='fixed_width_two_line', delimiter=' ')
    keck_mz = asc.read(FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt',
        guess=False, format='fixed_width_two_line', delimiter=' ')
    mainseq_corrs = asc.read(FULL_PATH+'Main_Sequence/mainseq_corrections_tbl_ref.txt',
        guess=False, format='fixed_width_two_line', delimiter=' ')

    stlr_mass = np.array(mainseq_corrs['stlr_mass'])
    inst_str0 = np.array(mainseq_corrs['inst_str0'])
    ha_ii = np.array(mainseq_corrs['ID'])-1

    data_dict = create_ordered_AP_arrays()
    HA_FLUX   = data_dict['HA_FLUX'][ha_ii]
    HA_SNR    = data_dict['HA_SNR'][ha_ii]
    NIIB_FLUX = data_dict['NIIB_FLUX'][ha_ii]
    NIIB_SNR  = data_dict['NIIB_SNR'][ha_ii]

    # using only the highest mmt_mz nb921 m bin
    aa = np.array([x for x in range(len(mmt_mz)) 
                   if (mmt_mz['stlrmass_bin'][x] != 'N/A' and mmt_mz['filter'][x] != 'NB973')])
    badnb921 = np.where((mmt_mz['filter'][aa]=='NB921') & (mmt_mz['max_stlrmass'][aa] < 9.78))[0]
    aa = np.delete(aa, badnb921)

    # getting more info from the data
    avgm_arr = np.concatenate((mmt_mz['avg_stlrmass'][aa], keck_mz['avg_stlrmass']))
    minm_arr = np.concatenate((mmt_mz['min_stlrmass'][aa], keck_mz['min_stlrmass']))
    maxm_arr = np.concatenate((mmt_mz['max_stlrmass'][aa], keck_mz['max_stlrmass']))
    nii_flux_arr = np.concatenate((mmt_mz['NII_6583_flux'][aa], keck_mz['NII_6583_flux']))
    ha_flux_arr  = np.concatenate((mmt_mz['HA_flux'][aa], keck_mz['HA_flux']))


    ## SNR >= 2 detections
    # getting indexes
    big_good_nii = np.array([x for x in range(len(NIIB_SNR)) 
        if (NIIB_SNR[x] >= 2 and NIIB_FLUX[x] != 0 and HA_FLUX[x] < 99)])
    i0 = [x for x in range(len(big_good_nii)) if 'MMT' in inst_str0[big_good_nii][x]]
    i1 = [x for x in range(len(big_good_nii)) if 'Keck' in inst_str0[big_good_nii][x]]
    i2 = [x for x in range(len(big_good_nii)) if 'merged' in inst_str0[big_good_nii][x]]

    # ratios for individual sources
    NII_BOTH_FLUX = NIIB_FLUX*((1+2.96)/2.96)
    BIG_FLUX_RAT = NII_BOTH_FLUX[big_good_nii]/HA_FLUX[big_good_nii]

    # errors for individual sources
    niiha_errs_tmp = composite_errors([NII_BOTH_FLUX[big_good_nii], HA_FLUX[big_good_nii]], 
        [NII_BOTH_FLUX[big_good_nii]/NIIB_SNR[big_good_nii], HA_FLUX[big_good_nii]/HA_SNR[big_good_nii]], 
        seed_i=8679, label='NII_BOTH/HA')
    niiha_errs_neg = niiha_errs_tmp[:,0]
    niiha_errs_pos = niiha_errs_tmp[:,1]
    niiha_errs = np.sqrt(niiha_errs_neg**2/2 + niiha_errs_pos**2/2)

    # plotting
    plt.plot(stlr_mass[big_good_nii][i0], BIG_FLUX_RAT[i0], 
             color='blue', mec='blue', marker='*', lw=0, label='MMT', ms=10, alpha=0.8)
    plt.errorbar(stlr_mass[big_good_nii][i0], BIG_FLUX_RAT[i0],
             yerr=niiha_errs[i0], fmt='none', mew=0, ecolor='blue', alpha=0.8)
    plt.plot(stlr_mass[big_good_nii][i1], BIG_FLUX_RAT[i1], 
             color='lightblue', mec='lightblue', marker='s', lw=0, label='Keck', alpha=0.8)
    plt.errorbar(stlr_mass[big_good_nii][i1], BIG_FLUX_RAT[i1],
             yerr=niiha_errs[i1], fmt='none', mew=0, ecolor='lightblue', alpha=0.8)
    plt.plot(stlr_mass[big_good_nii][i2], BIG_FLUX_RAT[i2], 
             color='purple', mec='purple', marker='o', lw=0, label='MMT+Keck', alpha=0.8)
    plt.errorbar(stlr_mass[big_good_nii][i2], BIG_FLUX_RAT[i2],
             yerr=niiha_errs[i2], fmt='none', mew=0, ecolor='purple', alpha=0.8)
    print 'are there zeros in individuals?', [x for x in BIG_FLUX_RAT[i0] if x==0], [x for x in BIG_FLUX_RAT[i1] if x==0], [x for x in BIG_FLUX_RAT[i2] if x==0]

    ## SNR < 2 limits
    # getting indexes
    lil_good_nii = np.array([x for x in range(len(NIIB_SNR)) 
        if (NIIB_SNR[x] < 2 and NIIB_FLUX[x] > 0 and HA_FLUX[x] < 99 
            and HA_SNR[x] > 0 and HA_SNR[x] < 99)])
    j0 = [x for x in range(len(lil_good_nii)) if 'MMT' in inst_str0[lil_good_nii][x]]
    j1 = [x for x in range(len(lil_good_nii)) if 'Keck' in inst_str0[lil_good_nii][x]]
    j2 = [x for x in range(len(lil_good_nii)) if 'merged' in inst_str0[lil_good_nii][x]]

    # limits for individual sources
    LIMIT_arr = 2/HA_SNR[lil_good_nii] * ((1+2.96)/2.96)

    # plotting
    plt.plot(stlr_mass[lil_good_nii][j0], LIMIT_arr[j0],
             linestyle='none', marker=u'$\u2193$', markersize=10, color='blue', mec='blue', mew=2, alpha=0.8)
    plt.plot(stlr_mass[lil_good_nii][j1], LIMIT_arr[j1],
             linestyle='none', marker=u'$\u2193$', markersize=10, color='lightblue', mec='lightblue', mew=2, alpha=0.8)
    plt.plot(stlr_mass[lil_good_nii][j2], LIMIT_arr[j2],
             linestyle='none', marker=u'$\u2193$', markersize=10, color='purple', mec='purple', mew=2, alpha=0.8)

    ## composites
    # ratios for composites
    flux_rat_arr = nii_flux_arr*((1+2.96)/2.96)/ha_flux_arr

    # errors for composites
    

    # plotting
    plt.plot(avgm_arr[:9], flux_rat_arr[:9], color='limegreen', mec='limegreen', lw=0, label='MMT composites', marker='*', ms=10, alpha=0.8)
    plt.plot(avgm_arr[9:], flux_rat_arr[9:], color='darkgreen', mec='darkgreen', lw=0, label='Keck composites', marker='s', alpha=0.8)
    plt.errorbar(avgm_arr[:9], flux_rat_arr[:9], xerr=np.array([avgm_arr[:9]-minm_arr[:9], maxm_arr[:9]-avgm_arr[:9]]), fmt='none', ecolor='limegreen', alpha=0.8)
    plt.errorbar(avgm_arr[9:], flux_rat_arr[9:], xerr=np.array([avgm_arr[9:]-minm_arr[9:], maxm_arr[9:]-avgm_arr[9:]]), fmt='none', ecolor='darkgreen', alpha=0.8)
    print 'are there zeros in composites?', [x for x in flux_rat_arr[:9] if x==0]

    # legend 1
    legendAA = plt.legend(loc='best', fontsize=11)
    ax = plt.gca().add_artist(legendAA)

    ## fitting 
    lowm_ii = np.array([x for x in range(len(avgm_arr)) if avgm_arr[x]<8])
    highm_ii = np.array([x for x in range(len(avgm_arr)) if avgm_arr[x]>=8])

    # a constant line to the composites below 10^8 M*
    const = np.mean(flux_rat_arr[lowm_ii])
    print 'C =', const

    # a linear line to the composites above 10^8 M*
    def line2(x, m):
        return m*(x-8.0)+const
    coeffs1, covar = curve_fit(line2, avgm_arr[highm_ii], flux_rat_arr[highm_ii])
    print 'm =',coeffs1[0], '& b =', coeffs1[0]*-8+const

    # plotting
    lineA, = plt.plot(np.arange(6.0,8.1,0.1), np.array([const]*len(np.arange(6.0,8.1,0.1))), 
        'r--', lw=2, label='C = '+str(np.around(const,4)))
    lineB, = plt.plot(np.arange(8.0,10.6,0.1), line2(np.arange(8.0,10.6,0.1), *coeffs1), 
        'r--', lw=2, label='m = '+str(np.around(coeffs1[0],4))+', b = '+str(np.around(coeffs1[0]*-8+const,4)))

    # legend2
    # legendAB = plt.gca().legend(handles=[lineA, lineB], loc='lower right', fontsize=11)
    # plt.gca().add_artist(legendAB)

    ## finishing touches
    plt.xlabel(r'$\log(M_\bigstar/M_\odot)$', size=16)
    plt.ylabel('[N II]'+r'$\lambda\lambda$6548,6583/H$\alpha$', size=16)
    # plt.ylabel(r'$[N \textsc{II}]$'+r'$\lambda\lambda$6548,6583/H$\alpha$', size=16)
    # plt.ylabel('['+r'\textsc{N ii}]$\lambda\lambda$6548,6583/H$\alpha$', size=16)
    a = [tick.label.set_fontsize(14) for tick in plt.gca().xaxis.get_major_ticks()]
    b = [tick.label.set_fontsize(14) for tick in plt.gca().yaxis.get_major_ticks()]
    plt.gca().tick_params(axis='both', which='both', direction='in')
    plt.ylim(ymax=1.2)
    plt.gcf().set_size_inches(8,7)
    plt.axhline(0.54 * (1+2.96)/2.96, color='k', ls='--', alpha=0.8)  # line marking NII6583/Ha > 0.54, Kennicut+08
    ax1 = plt.gca()
    ylims = ax1.get_ylim()

    # ax2 = ax1.twiny().twinx()
    # ax2.set_xticks([])
    # ax2.set_ylabel('log(O/H)+12', size=16)
    # ax2.tick_params(axis='y', which='both', direction='in')
    # ax2.yaxis.set_tick_params(labelsize=14)
    # ax2.set_ylim(ylims)
    # print ax1.get_yticks()[2:]
    # x0 = np.log10(ax1.get_yticks() * (2.96)/(1+2.96))
    # ax2.set_yticklabels(np.round(niiha_oh_determine(x0, 'PP04_N2'),2))
    # ax2.set_yticks(ax1.get_yticks())
    

    plt.tight_layout()
    plt.savefig(FULL_PATH+'Plots/main_sequence/NII_Ha_scatter.pdf')
    plt.close()


    # BELOW: making the y axis a log10 scale as well

    # testing plotting
    # individ
    plt.plot(stlr_mass[big_good_nii][i0], BIG_FLUX_RAT[i0], 
             color='blue', mec='blue', marker='*', lw=0, label='MMT', ms=10, alpha=0.8)
    plt.errorbar(stlr_mass[big_good_nii][i0], BIG_FLUX_RAT[i0],
             yerr=niiha_errs[i0], fmt='none', mew=0, ecolor='blue', alpha=0.8)
    plt.plot(stlr_mass[big_good_nii][i1], BIG_FLUX_RAT[i1], 
             color='lightblue', mec='lightblue', marker='s', lw=0, label='Keck', alpha=0.8)
    plt.errorbar(stlr_mass[big_good_nii][i1], BIG_FLUX_RAT[i1],
             yerr=niiha_errs[i1], fmt='none', mew=0, ecolor='lightblue', alpha=0.8)
    plt.plot(stlr_mass[big_good_nii][i2], BIG_FLUX_RAT[i2], 
             color='purple', mec='purple', marker='o', lw=0, label='MMT+Keck', alpha=0.8)
    plt.errorbar(stlr_mass[big_good_nii][i2], BIG_FLUX_RAT[i2],
             yerr=niiha_errs[i2], fmt='none', mew=0, ecolor='purple', alpha=0.8)

    # individ limits
    plt.plot(stlr_mass[lil_good_nii][j0], LIMIT_arr[j0],
             linestyle='none', marker=u'$\u2193$', markersize=10, color='blue', mec='blue', mew=2, alpha=0.8)
    plt.plot(stlr_mass[lil_good_nii][j1], LIMIT_arr[j1],
             linestyle='none', marker=u'$\u2193$', markersize=10, color='lightblue', mec='lightblue', mew=2, alpha=0.8)
    plt.plot(stlr_mass[lil_good_nii][j2], LIMIT_arr[j2],
             linestyle='none', marker=u'$\u2193$', markersize=10, color='purple', mec='purple', mew=2, alpha=0.8)

    # composites
    plt.plot(avgm_arr[:9], flux_rat_arr[:9], color='limegreen', mec='limegreen', lw=0, label='MMT composites', marker='*', ms=10, alpha=0.8)
    plt.plot(avgm_arr[9:], flux_rat_arr[9:], color='darkgreen', mec='darkgreen', lw=0, label='Keck composites', marker='s', alpha=0.8)
    plt.errorbar(avgm_arr[:9], flux_rat_arr[:9], xerr=np.array([avgm_arr[:9]-minm_arr[:9], maxm_arr[:9]-avgm_arr[:9]]), fmt='none', ecolor='limegreen', alpha=0.8)
    plt.errorbar(avgm_arr[9:], flux_rat_arr[9:], xerr=np.array([avgm_arr[9:]-minm_arr[9:], maxm_arr[9:]-avgm_arr[9:]]), fmt='none', ecolor='darkgreen', alpha=0.8)

    # lines
    lineA, = plt.plot(np.arange(6.0,8.1,0.1), np.array([const]*len(np.arange(6.0,8.1,0.1))), 
        'r--', lw=2)
    lineB, = plt.plot(np.arange(8.0,10.6,0.1), line2(np.arange(8.0,10.6,0.1), *coeffs1), 
        'r--', lw=2)
    plt.axhline(0.54 * (1+2.96)/2.96, color='k', ls='--', alpha=0.8)  # line marking NII6583/Ha > 0.54, Kennicut+08


    # setting the y axis to a log scale
    plt.yscale('log')


    # aesthetic touches
    plt.xlabel(r'$\log(M_\bigstar/M_\odot)$', size=16)
    plt.ylabel('log([N II]'+r'$\lambda\lambda$6548,6583/H$\alpha$'+')', size=16)
    plt.legend(loc='lower right', fontsize=14)

    a = [tick.label.set_fontsize(14) for tick in plt.gca().xaxis.get_major_ticks()]
    b = [tick.label.set_fontsize(14) for tick in plt.gca().yaxis.get_major_ticks()]
    plt.gca().tick_params(axis='both', which='both', direction='in')
    plt.gca().tick_params(axis='y', which='major', length=6)
    plt.gca().tick_params(axis='y', which='minor', length=3)
    plt.gcf().set_size_inches(8,7)


    # adding second axis
    ax1 = plt.gca()
    ylims = ax1.get_ylim()
    ax2 = ax1.twiny().twinx()
    ax2.set_xticks([])
    ax2.set_yscale('log')
    ax2.set_ylabel('12+log(O/H) [PP04]', size=16)
    ax2.tick_params(axis='y', which='both', direction='in')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.set_ylim(ylims)
    ax2.minorticks_off()

    x0 = np.log10(ax1.get_yticks()) + np.log10((2.96)/(1+2.96))
    ax2.set_yticklabels(np.round(niiha_oh_determine(x0, 'PP04_N2'),2))


    # saving plot
    plt.tight_layout()
    plt.gcf().set_size_inches(10,8)
    plt.savefig(FULL_PATH+'Plots/main_sequence/NII_Ha_scatter_log.pdf')
    plt.close()


    return {'C':const, 'm':coeffs1[0], 'b':coeffs1[0]*-8+const}


if __name__ == '__main__':
    main()