"""
NAME:
    plot_mstar_vs_ebv.py

PURPOSE:

    depends on 

INPUTS:
    FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt'
    FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt'
    FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt'

OUTPUTS:
    FULL_PATH+'Plots/main_sequence/mstar_vs_ebv.pdf'

NOTES:
    for the garn & best scaling, the following was done.
    1. we plotted mass vs reddening as usual
    2. for the RHS subplot, we created a twinx().twiny() axis called "ax2"
    3. for ax2, we manually relabeled the ticks such that A(Ha) = k_ha * E(B-V)
    4. we plotted the garn & best line according to the equation given in the paper.
       however, plotting that line as-is shows the g&b line on the E(B-V) scale (i.e., 
       at M=8.5, E(B-V) shows as 0.3 when it should be A(Ha) that is 0.3). therefore,
       we divided the g&b line by k_ha such that the g&b line is plotted on the A(Ha) scale.
       (for further clarification, look at troubleshooting notes 1/8)
"""

from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table
from create_ordered_AP_arrays import create_ordered_AP_arrays
import numpy as np, matplotlib.pyplot as plt
import plotting.general_plotting as general_plotting
import plot_NII_Ha_ratios
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from analysis.cardelli import *
from astropy.cosmology import FlatLambdaCDM
from scipy import stats
from scipy.interpolate import interp1d
cosmo = FlatLambdaCDM(H0 = 70 * u.km / u.s / u.Mpc, Om0=0.3)


FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'




### starting here
def main():
    # reading in data
    corr_tbl = asc.read(FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt',guess=False,
        Reader=asc.FixedWidthTwoLine)

    ha_ii = np.array(corr_tbl['ID'])-1
    zspec0 = corr_tbl['zspec0'].data

    data_dict = create_ordered_AP_arrays()
    HA_FLUX   = data_dict['HA_FLUX'][ha_ii]
    HB_FLUX   = data_dict['HB_FLUX'][ha_ii]
    HA_SNR    = data_dict['HA_SNR'][ha_ii]
    HB_SNR    = data_dict['HB_SNR'][ha_ii]

    no_spectra  = np.where((zspec0 <= 0) | (zspec0 > 9))[0]
    yes_spectra = np.where((zspec0 >= 0) & (zspec0 < 9))[0]

    k_ha = cardelli(6563 * u.Angstrom)



    gooddata_iis = np.where((HB_SNR[yes_spectra] >= 5) & (HA_FLUX[yes_spectra] > 1e-20) & (HA_FLUX[yes_spectra] < 99))[0]
    good_EBV_iis = yes_spectra[gooddata_iis]

    hahb = HA_FLUX[good_EBV_iis]/HB_FLUX[good_EBV_iis]
    sigma_hahb = hahb * np.sqrt((1/HA_SNR[good_EBV_iis])**2 + (1/HB_SNR[good_EBV_iis])**2)
    sigma_ebv = sigma_hahb/(hahb * np.log(10))



    mmt_mz  = asc.read(FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt',
        guess=False, format='fixed_width_two_line', delimiter=' ')
    keck_mz = asc.read(FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt',
        guess=False, format='fixed_width_two_line', delimiter=' ')
    # using only valid mmt_mz m bins
    aa = np.array([x for x in range(len(mmt_mz)) if mmt_mz['stlrmass_bin'][x] != 'N/A'])

    # getting more info from the data
    m = len(mmt_mz[aa])
    k = len(keck_mz)
    filt_arr = np.concatenate((mmt_mz['filter'][aa], keck_mz['filter']))
    inst_arr = np.concatenate((['MMT']*m, ['Keck']*k))
    avgm_arr = np.concatenate((mmt_mz['avg_stlrmass'][aa], keck_mz['avg_stlrmass']))
    minm_arr = np.concatenate((mmt_mz['min_stlrmass'][aa], keck_mz['min_stlrmass']))
    maxm_arr = np.concatenate((mmt_mz['max_stlrmass'][aa], keck_mz['max_stlrmass']))
    EBV = np.concatenate((mmt_mz['E(B-V)_hahb'][aa], keck_mz['E(B-V)_hahb']))
    EBV_rms = np.concatenate((mmt_mz['E(B-V)_hahb_rms'][aa], keck_mz['E(B-V)_hahb_rms']))

    # replacing invalid MMT NB973 EBV_hahb w/ EBV_hghb
    h = [x for x in range(len(aa)) if mmt_mz['filter'][aa][x]=='NB973'][0]
    EBV[h:h+5] = mmt_mz['E(B-V)_hghb'][-5:]
    EBV_rms[h:h+5] = mmt_mz['E(B-V)_hghb_rms'][-5:]

    # replacing invalid lowest two m bins MMT NB921 EBV_hahb w/ EBV_hghb
    i = np.where(mmt_mz['filter']=='NB921')[0][0]
    j = [x for x in range(len(aa)) if mmt_mz['filter'][aa][x]=='NB921'][0]
    EBV[j:j+2] = mmt_mz['E(B-V)_hghb'][i:i+2]
    EBV_rms[j:j+2] = mmt_mz['E(B-V)_hghb_rms'][i:i+2]

    x_arr = np.arange(8.5, 10.6, 0.1)
    gb2010 = 0.91 + 0.77*(x_arr-10) + 0.11*(x_arr-10)**2 - 0.09*(x_arr-10)**3
    f, axarr = plt.subplots(1,2, sharex=True, sharey=True)

    for ff, cc in zip(['NB704+NB711','NB816','NB921','NB973'], ['blue','green','orange','red']):
        yz_fmatch = np.array([x for x in range(len(corr_tbl)) if corr_tbl['filt'][x] in ff and 
                              corr_tbl['zspec0'][x]>0 and corr_tbl['zspec0'][x]<9])
        for inst, shape, ax_ii in zip(['MMT','Keck','merged'], ['o','*','s'], [0, 1, 0]):
            inst_match = np.array([x for x in range(len(yz_fmatch)) if inst in corr_tbl['inst_str0'][yz_fmatch][x]])

            if len(inst_match) > 0:
                mstar = corr_tbl['stlr_mass'][yz_fmatch[inst_match]]
                ebv00 = corr_tbl['EBV'][yz_fmatch[inst_match]]
                
                axarr[ax_ii].plot(mstar, ebv00, color=cc, marker=shape, lw=0, markersize=8, alpha=0.9, label=ff+'-'+inst)

                has_errs = np.array([x for x in yz_fmatch[inst_match] if x in good_EBV_iis])
                if len(has_errs) > 0:
                    sig_iis = np.array([x for x in range(len(good_EBV_iis)) if good_EBV_iis[x] in has_errs])
                    axarr[ax_ii].errorbar(corr_tbl['stlr_mass'][has_errs], corr_tbl['EBV'][has_errs], yerr=sigma_ebv[sig_iis],
                                fmt='none', mew=0, ecolor=cc, alpha=0.9)
                
                if inst=='merged':
                    for ax_ii in range(2):
                        axarr[ax_ii].plot(mstar, ebv00, color=cc, marker=shape, lw=0, markersize=8, alpha=0.9, label=ff+'-'+inst)

                        has_errs = np.array([x for x in yz_fmatch[inst_match] if x in good_EBV_iis])
                        if len(has_errs) > 0:
                            sig_iis = np.array([x for x in range(len(good_EBV_iis)) if good_EBV_iis[x] in has_errs])
                            axarr[ax_ii].errorbar(corr_tbl['stlr_mass'][has_errs], corr_tbl['EBV'][has_errs], yerr=sigma_ebv[sig_iis],
                                        fmt='none', mew=0, ecolor=cc, alpha=0.9)

    # plotting composites
    for ff, cc in zip(['NB704+NB711','NB816','NB921','NB973'], ['blue','green','orange','red']):
        yz_fmatch = np.array([x for x in range(len(filt_arr)) if filt_arr[x] in ff])
        
        for inst, shape, ax_ii in zip(['MMT','Keck'], ['o','*'], [0,1]):
            inst_match = np.array([x for x in range(len(yz_fmatch)) if inst_arr[yz_fmatch][x]==inst])
            
            if len(inst_match) > 0:
                mstar = avgm_arr[yz_fmatch[inst_match]]
                ebv00 = EBV[yz_fmatch[inst_match]]
                
                axarr[ax_ii].plot(mstar, ebv00, color=cc, marker=shape, lw=0, markersize=15, alpha=0.5, label=ff+'-'+inst)
                axarr[ax_ii].errorbar(mstar, ebv00, 
                             xerr=np.array([mstar-minm_arr[yz_fmatch[inst_match]], 
                                            maxm_arr[yz_fmatch[inst_match]]-mstar]),
                             yerr=EBV_rms[yz_fmatch[inst_match]],
                             fmt='none', ecolor=cc, alpha=0.5)
            axarr[ax_ii].tick_params(axis='both', which='both', direction='in')

    # set axis props
    for ax, ii in zip(axarr, range(2)):
        ax.set_xlabel('log10(M'+r'$_\bigstar$'+'/M'+r'$_\odot$'+')', size=14)

        ax.plot(x_arr, gb2010/k_ha, 'k--', lw=3, label='Garn & Best (2010)')
        ax.text(9.78, 0.48, 'Garn & Best (2010)', rotation=33, color='k',
             alpha=1, fontsize=10, fontweight='bold')

        if ii==0:
            ax.set_ylabel('E(B-V)', size=14)
        else: #ii==1
            ax.yaxis.set_tick_params(size=0)


    # creating a twin axis for the second subplot
    ax2 = axarr[1].twiny().twinx()
    axarr[1].get_shared_x_axes().join(axarr[0], axarr[1], ax2)
    axarr[1].get_shared_y_axes().join(axarr[0], axarr[1], ax2)
    ax2.set_ylabel(r'A(H$\alpha$)', size=14)
    ax2.tick_params(axis='y', which='both', direction='in')
    ax2.set_yticks(axarr[0].get_yticks())
    ax2.set_yticklabels(np.round(k_ha*axarr[0].get_yticks(),2))
    ax2.set_xticks([])

    axarr[0].set_ylim(-0.1, 1.45)




    b_patch = mpatches.Patch(color='b', label='NB704,NB711')
    g_patch = mpatches.Patch(color='g', label='NB816')
    o_patch = mpatches.Patch(color='orange', label='NB921')
    r_patch = mpatches.Patch(color='r', label='NB973')

    mmt = mlines.Line2D([], [], color='white', mec='k', marker='o', markersize=15, label='MMT')
    keck = mlines.Line2D([], [], color='white', mec='k', marker='*', markersize=15, label='Keck')
    mrgd = mlines.Line2D([], [], color='white', mec='k', marker='s', markersize=15, label='MMT+Keck')

    legend1 = axarr[1].legend(handles=[b_patch, g_patch, o_patch, r_patch], ncol=4, fontsize=13,
                         loc='upper left', bbox_to_anchor=(-0.45, 1.0), facecolor='white', frameon=1)
    axarr[1].add_artist(legend1)
    legend2 = axarr[0].legend(handles=[mmt, mrgd], ncol=3, fontsize=13, 
                         loc='upper left', bbox_to_anchor=(0, 0.935))
    axarr[0].add_artist(legend2)
    legend3 = axarr[1].legend(handles=[keck, mrgd], ncol=3, fontsize=13, 
                         loc='upper left', bbox_to_anchor=(0, 0.935))
    axarr[1].add_artist(legend3)

    # need to label composites

    # f.set_size_inches(20,8)
    f.set_size_inches(17,6)
    f.subplots_adjust(wspace=0, left=0.04, right=0.95, top=0.99, bottom=0.09)

    plt.savefig(FULL_PATH+'Plots/main_sequence/mstar_vs_ebv.pdf')


if __name__ == '__main__':
    main()