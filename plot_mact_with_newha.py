"""
NAME:
    plot_mact_with_newha.py

PURPOSE:
    Generates main sequence figures with fits and dispersion, and sSFR figs,
    with both the MACT and the NEWHA dataset.

    Depends on plot_nbia_mainseq, nbia_mainseq_dispersion, and
    mainseq_corrections.py

INPUTS:
    FULL_PATH+'NewHa/NewHa.fits'
    FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt'

OUTPUTS:
    FULL_PATH+'Plots/NewHa/zdep_mainseq_'+newha_sfr_type+
        '_'+ytype+'_'+fittype+'.pdf'
    FULL_PATH+'Plots/NewHa/mainseq_dispersion_'+newha_sfr_type+
        '_'+ytype+'_'+fittype+'.pdf'
    FULL_PATH+'Plots/NewHa/mainseq_sSFRs_'+newha_sfr_type+'.pdf'
"""
from __future__ import print_function

import numpy as np, matplotlib.pyplot as plt
import plot_nbia_mainseq, nbia_mainseq_dispersion
import matplotlib as mpl
from astropy.io import ascii as asc, fits as pyfits

from MACT_utils import get_good_newha_ii, get_newha_logsfrha

# emission line wavelengths (air)
HA = 6562.80

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'
CUTOFF_SIGMA = 4.0
CUTOFF_MASS = 6.0

# newha_sfr_type = 'orig_sfr'
newha_sfr_type = 'met_dep_sfr'


def main():
    '''
    Reads in data from NewHa and MACT datasets, and obtains the useful data
    (defined by good_sig_iis for MACT). Then, iterating through appropriate
    combinations of fittype and ytype, various plotting functions from
    plot_nbia_mainseq and nbia_mainseq_dispersion are called. Figures are
    then saved and closed.
    '''
    # reading in NewHa data
    newha = pyfits.open(FULL_PATH+'NewHa/NewHa.fits')
    newhadata_tmp = newha[1].data

    good_newha_ii = get_good_newha_ii(newhadata_tmp)
    newhadata = newhadata_tmp[good_newha_ii]

    newha_logm = newhadata['LOGM']
    newha_zspec = newhadata['Z_SPEC']
    newha_mzdata = np.vstack([newha_logm, newha_zspec]).T
    newha_logsfrha = get_newha_logsfrha(newhadata, newha_sfr_type)

    # reading in data generated by EBV_corrections.py
    corr_tbl = asc.read(FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt',
        guess=False, Reader=asc.FixedWidthTwoLine)

    # defining a flux sigma and mass cutoff
    good_sig_iis = np.where((corr_tbl['flux_sigma'] >= CUTOFF_SIGMA) & 
        (corr_tbl['stlr_mass'] >= CUTOFF_MASS))[0]

    # getting/storing useful data
    zspec0 = np.array(corr_tbl['zspec0'])[good_sig_iis]
    stlr_mass = np.array(corr_tbl['stlr_mass'])[good_sig_iis]
    filts = np.array(corr_tbl['filt'])[good_sig_iis]
    sfr = np.array(corr_tbl['met_dep_sfr'])[good_sig_iis]
    dust_corr_factor = np.array(corr_tbl['dust_corr_factor'])[good_sig_iis]
    filt_corr_factor = np.array(corr_tbl['filt_corr_factor'])[good_sig_iis]
    nii_ha_corr_factor = np.array(corr_tbl['nii_ha_corr_factor'])[good_sig_iis]
    corr_sfrs = sfr+filt_corr_factor+nii_ha_corr_factor+dust_corr_factor
    zspec00 = plot_nbia_mainseq.approximated_zspec0(zspec0, filts)

    # defining useful data structs for plotting
    nh_markarr = np.array(['o','^','D','*','X'])
    nh_sizearr = np.array([6.0,6.0,6.0,9.0,6.0])**2
    nh_ffarr = np.array(['NB7', 'NB816', 'NB921', 'NB973', 'NEWHA'])
    nh_llarr = np.array(['NB704,NB711', 'NB816', 'NB921', 'NB973', 'NEWHA'])
    z_arr = plot_nbia_mainseq.get_z_arr()
    nh_z_arr = np.append(z_arr, '%.2f'%np.mean(newha_zspec))
    nh_cwheel = [np.array(mpl.rcParams['axes.prop_cycle'])[x]['color']
        for x in range(5)]

    # combining datasets
    sfrs_with_newha  = np.concatenate((corr_sfrs, newha_logsfrha))
    mass_with_newha  = np.concatenate((stlr_mass, newha_logm))
    zspec_with_newha = np.concatenate((zspec0, newha_zspec))
    zspec_with_newha00 = np.concatenate((zspec00, newha_zspec))
    filts_with_newha = np.concatenate((filts,
        np.array(['NEWHA']*len(newha_logsfrha))))
    data_with_newha = np.vstack([mass_with_newha, zspec_with_newha00]).T

    no_spectra  = np.where((zspec_with_newha <= 0) | (zspec_with_newha > 9))[0]
    yes_spectra = np.where((zspec_with_newha >= 0) & (zspec_with_newha < 9))[0]


    # plotting
    ssfrs_with_newha = sfrs_with_newha - mass_with_newha
    for ytype, ydata_sfrs in zip(['SFR', 'sSFR'], 
        [sfrs_with_newha, ssfrs_with_newha]):
        for fittype in ['first_order', 'second_order']:
            print('making redshift dependent plot (y-axis: '+
                ytype+'; '+fittype+' fit)')

            f, ax = plt.subplots()
            plot_nbia_mainseq.make_redshift_graph(f, ax, nh_z_arr, ydata_sfrs,
                mass_with_newha, zspec_with_newha00, filts_with_newha,
                no_spectra, yes_spectra, nh_cwheel, ffarr=nh_ffarr,
                llarr=nh_llarr, ytype=ytype, fittype=fittype, withnewha=True)

            plt.savefig(FULL_PATH+'Plots/NewHa/zdep_mainseq_'+newha_sfr_type+
                '_'+ytype+'_'+fittype+'.pdf')
            plt.close()


            print('making dispersion plots (y-axis: '+
                ytype+'; '+fittype+' fit)')

            f, ax = plt.subplots()
            nbia_mainseq_dispersion.plot_all_dispersion(f, ax, data_with_newha,
                ydata_sfrs, mass_with_newha, filts_with_newha, no_spectra,
                yes_spectra, nh_z_arr, markarr=nh_markarr, sizearr=nh_sizearr,
                ffarr=nh_ffarr, llarr=nh_llarr, ytype=ytype, fittype=fittype,
                withnewha=True)

            plt.savefig(FULL_PATH+'Plots/NewHa/mainseq_dispersion_'+newha_sfr_type+
                '_'+ytype+'_'+fittype+'.pdf')
            plt.close()

            print


    print('making sSFR plot')
    f, axes = plt.subplots(1,2, sharey=True)
    plot_nbia_mainseq.make_ssfr_graph(f, axes, sfrs_with_newha,
        mass_with_newha, filts_with_newha, zspec_with_newha00, nh_cwheel,
        nh_z_arr, ffarr=nh_ffarr, llarr=nh_llarr)
    plt.savefig(FULL_PATH+'Plots/NewHa/mainseq_sSFRs_'+newha_sfr_type+'.pdf')
    plt.close()


    newha.close()


if __name__ == '__main__':
    main()