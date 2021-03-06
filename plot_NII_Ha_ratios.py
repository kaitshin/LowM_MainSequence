"""
NAME:
    plotting_NII_Ha_ratios.py

PURPOSE:

    depends on mainseq_corrections.py, stack_spectral_data.py

INPUTS:
    FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt'
    FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt'

OUTPUTS:
    FULL_PATH+'Plots/main_sequence/NII_Ha_scatter.pdf'
    FULL_PATH+'Plots/main_sequence/NII_Ha_scatter_log.pdf'
"""
from __future__ import print_function

import numpy as np, matplotlib.pyplot as plt
from astropy.io import fits as pyfits, ascii as asc
from scipy.optimize import curve_fit

import config
from MACT_utils import composite_errors, exclude_bad_sources


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
            print('## Using linear relation!')
            # Bug found. Mod on 30/06/2016 OH_gas -> OH_gas[index]
            OH_gas[index] = 8.90 + 0.57 * x0[index] #xt0
    #endif

    return OH_gas


def get_ref_arrs():
    # read in stuff
    nbia = pyfits.open(config.FULL_PATH+config.NB_IA_emitters_cat)
    nbiadata = nbia[1].data
    fout  = asc.read(config.FULL_PATH+'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX.fout',
                     guess=False,Reader=asc.NoHeader)
    zspec = asc.read(config.FULL_PATH+'Catalogs/nb_ia_zspec.txt',guess=False,
                     Reader=asc.CommentedHeader)

    
    NAME0 = np.array(nbiadata['NAME'])
    ID0   = np.array(nbiadata['ID'])
    stlr_mass = np.array(fout['col7'])
    inst_str0 = zspec['inst_str0'].data
    
    # limit all data to Halpha emitters only
    ha_ii = np.array([x for x in range(len(NAME0)) if 'Ha-NB' in NAME0[x]])
    NAME0       = NAME0[ha_ii]

    # getting rid of unreliable galaxies:
    ha_ii, NAME0 = exclude_bad_sources(ha_ii, NAME0)
    ID0         = ID0[ha_ii]
    stlr_mass   = stlr_mass[ha_ii]
    inst_str0   = inst_str0[ha_ii]

    ret_ha_ii = ID0 - 1
    assert(np.all(ha_ii ==ret_ha_ii))

    return ha_ii, stlr_mass, inst_str0


def main():
    '''
    '''
    # latex backend for mpl
    # matplotlib.rcParams['text.usetex'] = True
    # matplotlib.rcParams['text.latex.unicode'] = True

    # reading data in
    mmt_mz  = asc.read(config.FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt',
        guess=False, format='fixed_width_two_line', delimiter=' ')
    keck_mz = asc.read(config.FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt',
        guess=False, format='fixed_width_two_line', delimiter=' ')
    ha_ii, stlr_mass, inst_str0 = get_ref_arrs()

    data_dict = config.data_dict
    HA_FLUX   = data_dict['HA_FLUX'][ha_ii]
    HA_SNR    = data_dict['HA_SNR'][ha_ii]
    NIIB_FLUX = data_dict['NIIB_FLUX'][ha_ii]
    NIIB_SNR  = data_dict['NIIB_SNR'][ha_ii]

    # using only the highest mmt_mz nb921 m bin
    aa = np.array([x for x in range(len(mmt_mz)) 
                   if (mmt_mz['stlrmass_bin'][x] != 'N/A' and mmt_mz['filter'][x] != 'NB973')])
    badnb921 = np.where((mmt_mz['filter'][aa]=='NB921') & (mmt_mz['max_stlrmass'][aa] < 9.72))[0]
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
    print('are there zeros in individuals?', [x for x in BIG_FLUX_RAT[i0] if x==0],
        [x for x in BIG_FLUX_RAT[i1] if x==0], [x for x in BIG_FLUX_RAT[i2] if x==0])

    # errors for individual sources
    niiha_errs_tmp = composite_errors([NII_BOTH_FLUX[big_good_nii], HA_FLUX[big_good_nii]], 
        [NII_BOTH_FLUX[big_good_nii]/NIIB_SNR[big_good_nii], HA_FLUX[big_good_nii]/HA_SNR[big_good_nii]], 
        seed_i=8679, label='NII_BOTH/HA')
    niiha_errs_neg = niiha_errs_tmp[:,0]
    niiha_errs_pos = niiha_errs_tmp[:,1]
    niiha_errs = np.sqrt(niiha_errs_neg**2/2 + niiha_errs_pos**2/2)


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


    ## Composites
    # ratios for composites
    flux_rat_arr = nii_flux_arr*((1+2.96)/2.96)/ha_flux_arr
    print('are there zeros in composites?', [x for x in flux_rat_arr[:9] if x==0])

    ## fitting 
    lowm_ii = np.array([x for x in range(len(avgm_arr)) if avgm_arr[x]<8])
    highm_ii = np.array([x for x in range(len(avgm_arr)) if avgm_arr[x]>=8])

    # a constant line to the composites below 10^8 M*
    const = np.mean(flux_rat_arr[lowm_ii])
    print('C =', const)

    # a linear line to the composites above 10^8 M*
    def line2(x, m):
        return m*(x-8.0)+const
    coeffs1, covar = curve_fit(line2, avgm_arr[highm_ii], flux_rat_arr[highm_ii])
    print('m =',coeffs1[0], '& b =', coeffs1[0]*-8+const)


    ## plotting: making the y axis a log10 scale
    # individ
    plt.plot(stlr_mass[big_good_nii][i0], BIG_FLUX_RAT[i0], 
             color='blue', mec='blue', marker='o', lw=0, label='MMT', alpha=0.8)
    plt.errorbar(stlr_mass[big_good_nii][i0], BIG_FLUX_RAT[i0],
             yerr=niiha_errs[i0], fmt='none', mew=0, ecolor='blue', alpha=0.8)
    plt.plot(stlr_mass[big_good_nii][i1], BIG_FLUX_RAT[i1], 
             color='lightblue', mec='lightblue', marker='*', ms=10, lw=0, label='Keck', alpha=0.8)
    plt.errorbar(stlr_mass[big_good_nii][i1], BIG_FLUX_RAT[i1],
             yerr=niiha_errs[i1], fmt='none', mew=0, ecolor='lightblue', alpha=0.8)
    plt.plot(stlr_mass[big_good_nii][i2], BIG_FLUX_RAT[i2], 
             color='purple', mec='purple', marker='s', lw=0, label='MMT+Keck', alpha=0.8)
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
    plt.plot(avgm_arr[:9], flux_rat_arr[:9], color='limegreen', mec='limegreen', lw=0,
        label='MMT composites', marker='o', ms=10, alpha=0.5)
    plt.plot(avgm_arr[9:], flux_rat_arr[9:], color='darkgreen', mec='darkgreen', lw=0,
        label='Keck composites', marker='*', ms=15, alpha=0.5)

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
    plt.savefig(config.FULL_PATH+'Plots/main_sequence/NII_Ha_scatter_log.pdf')
    plt.close()


    return {'C':const, 'm':coeffs1[0], 'b':coeffs1[0]*-8+const}


if __name__ == '__main__':
    main()