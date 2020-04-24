"""
NAME:
    mainseq_corrections.py

PURPOSE:
    Applies filter, nii, and dust corrections to all valid Ha emitting 
    sources and outputs corrections to a table

    Table is then used to graph the main sequence plots in
    plot_nbia_mainseq.py

INPUTS:
    'Catalogs/NB_IA_emitters.nodup.colorrev.fix.fits'
    'Catalogs/NB_IA_emitters.allcols.colorrev.fits'
    'Catalogs/nb_ia_zspec.txt'
    'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX.fout'
    'Composite_Spectra/StellarMass/MMT_all_five_data.txt'
    'Composite_Spectra/StellarMass/Keck_all_five_data.txt'
    'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt'
    'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt'
    'Main_Sequence/Catalogs/Keck/DEIMOS_single_line_fit.fits'
    'Filters/'+ff+'response.dat'

OUTPUTS:
    FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt'
"""
from __future__ import print_function

from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table
from create_ordered_AP_arrays import create_ordered_AP_arrays
import numpy as np, matplotlib.pyplot as plt
import plotting.general_plotting as general_plotting
import plot_NII_Ha_ratios

from MACT_utils import niiha_oh_determine, composite_errors
from analysis.cardelli import *
from astropy.cosmology import FlatLambdaCDM
from scipy import stats
from scipy.interpolate import interp1d
cosmo = FlatLambdaCDM(H0 = 70 * u.km / u.s / u.Mpc, Om0=0.3)

# emission line wavelengths (air)
HB = 4861.32
HA = 6562.80

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'
SEED_ORIG = 57842


def apply_filt_corrs_interp(ff, filt_corrs, zspec0, bad_z, good_z, AP, allcolsdata):
    '''
    Applies the filter-based correction.

    Accepts only no/yes_spectra sources with filter matches.

    For the sources with no_spectra (bad_z), a statistical correction based on
    the filter response profile was provided.
    For the ones with a yes_spectra (good_z), the relevant filter response .dat file
    was read in and interpolated to try and find what kind of flux a source 
    with that zspec would have in the filter response. the flux filter correction
    factor was returned.

    (filt_corrs = filter_corrected_fluxes - orig_NB_flux_from_allcolsdata)
    '''
    print(ff, len(bad_z)+len(good_z))
    # reading data
    response = asc.read(FULL_PATH+'Filters/'+ff+'response.dat',guess=False,
                    Reader=asc.NoHeader)
    z_filt = response['col1'].data/HA - 1
    yresponse = np.array(response['col2'])

    filter_stat_corr_dict = {'NB704':1.289439104, 'NB711':1.41022358406, 'NB816':1.29344789854, 'NB921':1.32817034288, 'NB973':1.29673596942}
    filt_corrs[bad_z] = np.log10(filter_stat_corr_dict[ff]) # np.log10(1.28)
    
    f = interp1d(z_filt, response['col2'])

    if max(zspec0[good_z]) > max(z_filt):
        print('Source above filter profile range (', ff, '|| z =', max(zspec0[good_z]), ')')
        bad_z0 = np.array([x for x in good_z if zspec0[x] == max(zspec0[good_z])])
        DEIMOS = pyfits.open('/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/Main_Sequence/Catalogs/Keck/DEIMOS_single_line_fit.fits')
        DEIMOSdata = DEIMOS[1].data
        print('Source name:', AP[bad_z0])
        ii = np.where(DEIMOSdata['AP']==AP[bad_z0])[0]
        keck_flux = np.log10(DEIMOSdata['HA_FLUX_MOD'][ii])
        filtcorr_bflux = np.log10(10**keck_flux + 0.8e-17*1.33)
        print('>>>>>>> Filter-corrected excess flux:', filtcorr_bflux)
        jj = np.where(zspec0==max(zspec0[good_z]))[0]
        filt_corrs[bad_z0] = filtcorr_bflux - allcolsdata[ff+'_FLUX'][jj]

        good_z = np.array([x for x in good_z if zspec0[x] != max(zspec0[good_z])])

        DEIMOS.close()

    filt_corrs[good_z] = np.log10(1.0/(f(zspec0[good_z])/max(yresponse)))

    # these are the filter correction factors
    return filt_corrs


def consolidate_ns_ys(orig_fluxes, no_spectra, yes_spectra, data_ns, data_ys, datatype='num'):
    '''
    consolidates no_spectra and yes_spectra data into a single data_array
    of shape orig_fluxes (same as allcolsdata)
    '''
    consod = np.zeros(len(orig_fluxes))
    if datatype == 'str':
        consod = np.array(['']*len(orig_fluxes), dtype='|S10')

    consod[no_spectra] = data_ns
    consod[yes_spectra] = data_ys

    return consod


def get_bins(masslist_MMT, masslist_Keck, massZlist_MMT, massZlist_Keck):
    '''
    Helper function for main. Flattens ndarrays of bins into a bins array

    Doesn't include the max_mass as an upper bound to the mass bins so that
    sources that have m > m_max_bin will automatically fall into the highest
    bin.
    Ex:
      >>> x = np.array([5.81, 6.78, 7.12, 7.94, 9.31])
      >>> bins = np.array([6.2, 7.53, 9.31]) 
      >>> np.digitize(x, bins, right=True)
      array([0, 1, 1, 2, 2]) # instead of array([0, 1, 1, 2, 3])
    '''
    massbins_MMT = np.append(masslist_MMT[:,0], masslist_MMT[-1,-1])
    massbins_Keck = np.append(masslist_Keck[:,0], masslist_Keck[-1,-1])
    massZbins_MMT = np.append(massZlist_MMT[:,0], massZlist_MMT[-1,-1])
    massZbins_Keck = np.append(massZlist_Keck[:,0], massZlist_Keck[-1,-1])

    return massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck


def fix_masses_out_of_range(masses_MMT_ii, masses_Keck_ii, masstype):
    '''
    Fixes masses that were not put into bins because they were too low.
    Reassigns them to the lowest mass bin. 

    Fixes masses that were not put into bins because they were too high.
    Reassigns them to the highest mass bin.  (only relevant for nb973 keck_m/z)

    Does this for both stlrmass bins and stlrmassZ bins.
    '''
    if masstype=='stlrmass':
        masses_MMT_ii = np.array([1 if x==0 else x for x in masses_MMT_ii])
        masses_Keck_ii = np.array([1 if x==0 else x for x in masses_Keck_ii])
        masses_Keck_ii = np.array([5 if x==6 else x for x in masses_Keck_ii])
    else: #=='stlrmassZ'
        masses_MMT_ii = np.array(['1'+x[1:] if x[0]=='0' else x for x in masses_MMT_ii])
        masses_Keck_ii = np.array(['1'+x[1:] if x[0]=='0' else x for x in masses_Keck_ii])
        masses_Keck_ii = np.array(['5'+x[1:] if x[0]=='6' else x for x in masses_Keck_ii])

    assert (6 not in masses_MMT_ii) and (6 not in masses_Keck_ii)
    return masses_MMT_ii, masses_Keck_ii


def handle_unusual_dual_emitters(names):
    """
    Purpose:
      This is intended to refactor handling of dual emitters and how to classify them

    :param names: list of full NB-IA identifying names

    :return filts:
    :return dual_iis:
    :return dual_ii2:
    """

    # get redshift bins
    #  ensures that dual NB704+NB711 emitters are treated as NB704-only emitters
    #  purely for the purpose of more convenient filter corrections
    filts = np.array([x[x.find('Ha-')+3:x.find('Ha-')+8] for x in names])

    # this ensures that NB704+NB921 dual emitters will be placed in Ha-NB921 bins
    #  since these sources are likely OIII-NB704 and Ha-NB921
    dual_iis = [x for x in range(len(names)) if 'Ha-NB704' in names[x] and 'Ha-NB921' in names[x]]

    # this ensures that the NB816+NB921 dual emitter will be placed in the NB921 bin
    #  we decided this based on visual inspection of the photometry
    dual_ii2 = [x for x in range(len(names)) if 'Ha-NB816' in names[x] and 'Ha-NB921' in names[x]]

    return filts, dual_iis, dual_ii2


def bins_table(indexes, NAME0, AP, stlr_mass, massbins_MMT, massbins_Keck,
               massZbins_MMT, massZbins_Keck, massZlist_filts_MMT, massZlist_filts_Keck):
    '''
    Creates and returns a table of bins as such:

    NAME             stlrmass filter stlrmassbin_MMT stlrmassbin_Keck stlrmassZbin_MMT stlrmassZbin_Keck
    ---------------- -------- ------ --------------- ---------------- ---------------- -----------------
    Ha-NB973_178201  8.4      NB973  4               3                2-NB973          2-NB973

    WARNINGS: 
    - dual Ha-NB704/Ha-NB711 sources are being treated as Ha-NB704 sources for now, but
      we may have to use photometry to better determine which EBV values to use
    - Ha-NB704+OII-NB973 sources are being treated as Ha-NB704 sources for now, but 
      they may be excluded as they may be something else given the NB973 excess
    '''
    names = NAME0[indexes]
    masses = stlr_mass[indexes]

    # Handle dual NB704+NB711 emitters, NB704+NB921 emitters, and
    # NB816+NB921 emitters
    filts, dual_iis, dual_ii2 = handle_unusual_dual_emitters(names)

    filts[dual_iis] = 'NB921'
    
    filts[dual_ii2] = 'NB921'

    # get stlrmass bins
    masses_MMT_ii = np.digitize(masses, massbins_MMT, right=True)
    masses_Keck_ii = np.digitize(masses, massbins_Keck, right=True)
    masses_MMT, masses_Keck = fix_masses_out_of_range(masses_MMT_ii, masses_Keck_ii, 'stlrmass')
    
    # get stlrmassZ bins
    ii0 = 0
    ii1 = 0
    massZs_MMT = np.array(['UNFILLED']*len(indexes))
    massZs_Keck = np.array(['UNFILLED']*len(indexes))
    for ff in ['NB704+NB711','NB816','NB921','NB973']:
        jj = len(np.where(ff==massZlist_filts_MMT)[0])
        
        good_filt_iis = np.array([x for x in range(len(filts)) if filts[x] in ff])
        
        if ff=='NB973':
            jj += 1
        
        mass_MMT_iis  = np.digitize(masses[good_filt_iis], massZbins_MMT[ii0:ii0+jj], right=True)
        massZ_MMT_iis = np.array([str(x)+'-'+ff for x in mass_MMT_iis])
        massZs_MMT[good_filt_iis] = massZ_MMT_iis
        
        ii0+=jj
        if 'NB9' in ff:
            kk = len(np.where(ff==massZlist_filts_Keck)[0])

            if ff=='NB973':
                kk += 1

            mass_Keck_iis  = np.digitize(masses[good_filt_iis], massZbins_Keck[ii1:ii1+kk], right=True)
            massZ_Keck_iis = np.array([str(x)+'-'+ff for x in mass_Keck_iis])
            massZs_Keck[good_filt_iis] = massZ_Keck_iis
            
            ii1+=kk
        else:
            massZs_Keck[good_filt_iis] = np.array(['N/A']*len(good_filt_iis))
            masses_Keck[good_filt_iis] = -99
    
    # putting sources with m < m_min_bin in the lowest-massZ mass bin
    massZs_MMT, massZs_Keck = fix_masses_out_of_range(massZs_MMT, massZs_Keck, 'stlrmassZ')

        
    tab0 = Table([names, masses, filts, masses_MMT, masses_Keck, massZs_MMT, massZs_Keck], 
        names=['NAME', 'stlrmass', 'filter', 'stlrmassbin_MMT', 'stlrmassbin_Keck', 
               'stlrmassZbin_MMT', 'stlrmassZbin_Keck'])

    return tab0


def EBV_corrs_no_spectra(tab_no_spectra, mmt_mz, mmt_mz_EBV_hahb, mmt_mz_EBV_hghb, keck_mz, keck_mz_EBV_hahb):
    '''
    '''
    EBV_corrs = np.array([-100.0]*len(tab_no_spectra))

    # loop based on filter
    for ff in ['NB704', 'NB711', 'NB816', 'NB921', 'NB973']:
        bin_filt_iis = np.array([x for x in range(len(tab_no_spectra)) if tab_no_spectra['filter'][x]==ff])
        print('num in '+ff+':', len(bin_filt_iis))

        if ff=='NB704' or ff=='NB711' or ff=='NB816':
            # using MMT hahb vals for all of these sources
            tab_filt_iis = np.array([x for x in range(len(mmt_mz)) if 
                (ff in mmt_mz['filter'][x] and mmt_mz['stlrmass_bin'][x] != 'N/A')])
            m_bin = np.array([int(x[0])-1 for x in tab_no_spectra['stlrmassZbin_MMT'][bin_filt_iis]])

            for ii, m_i in enumerate(m_bin):
                EBV_corrs[bin_filt_iis[ii]] = mmt_mz_EBV_hahb[tab_filt_iis[m_i]]

        elif ff == 'NB921':
            # for sources that fall within the keck mass range, we use keck hahb vals
            #  max_keck_mass should be 9.72
            max_keck_mass = float(keck_mz[np.where(keck_mz['filter']=='NB921')[0]]['stlrmass_bin'][-1]
                [keck_mz[np.where(keck_mz['filter']=='NB921')[0]]['stlrmass_bin'][-1].find('-')+1:])
            bin_filt_iis_keck = np.array([x for x in range(len(tab_no_spectra)) if 
                (tab_no_spectra['filter'][x]==ff and tab_no_spectra['stlrmass'][x]<=max_keck_mass)])
            tab_filt_iis_keck = np.array([x for x in range(len(keck_mz)) if 
                (keck_mz['filter'][x]==ff and keck_mz['stlrmass_bin'][x] != 'N/A')])
            m_bin_keck = np.array([int(x[0])-1 for x in tab_no_spectra['stlrmassZbin_Keck'][bin_filt_iis_keck]])

            for ii, m_i in enumerate(m_bin_keck):
                EBV_corrs[bin_filt_iis_keck[ii]] = keck_mz_EBV_hahb[tab_filt_iis_keck[m_i]]

            # for sources that fall above the keck mass range (so we use mmt hahb vals)
            bin_filt_iis_mmt = np.array([x for x in range(len(tab_no_spectra)) if 
                (tab_no_spectra['filter'][x]==ff and tab_no_spectra['stlrmass'][x]>max_keck_mass)])
            tab_filt_iis_mmt = np.array([x for x in range(len(mmt_mz)) if 
                (mmt_mz['filter'][x]==ff and mmt_mz['stlrmass_bin'][x] != 'N/A')])
            m_bin_mmt = np.array([int(x[0])-1 for x in tab_no_spectra['stlrmassZbin_MMT'][bin_filt_iis_mmt]])

            for ii, m_i in enumerate(m_bin_mmt):
                EBV_corrs[bin_filt_iis_mmt[ii]] = mmt_mz_EBV_hghb[tab_filt_iis_mmt[m_i]]
            
        elif ff == 'NB973':
            # using keck hahb vals instead of mmt hahb vals
            tab_filt_iis = np.array([x for x in range(len(keck_mz)) if 
                (keck_mz['filter'][x]==ff and keck_mz['stlrmass_bin'][x] != 'N/A')])
            m_bin = np.array([int(x[0])-1 for x in tab_no_spectra['stlrmassZbin_Keck'][bin_filt_iis]])

            for ii, m_i in enumerate(m_bin):
                EBV_corrs[bin_filt_iis[ii]] = keck_mz_EBV_hahb[tab_filt_iis[m_i]]
    #endfor

    assert len([x for x in EBV_corrs if x==-100.0]) == 0
    return EBV_corrs


def EBV_corrs_yes_spectra(EBV_corrs_ys, yes_spectra, HA_FLUX, HB_FLUX, HB_SNR, HA_SNR):
    '''
    '''
    k_hb = cardelli(HB * u.Angstrom)
    k_ha = cardelli(HA * u.Angstrom)
    
    gooddata_iis = np.where((HB_SNR[yes_spectra] >= 5) & (HA_SNR[yes_spectra] > 0) & (HA_FLUX[yes_spectra] > 1e-20) & (HA_FLUX[yes_spectra] < 99))[0]
    good_EBV_iis = yes_spectra[gooddata_iis]

    hahb = HA_FLUX[good_EBV_iis]/HB_FLUX[good_EBV_iis]
    hahb = np.array([2.86 if (x < 2.86 and x > 0) else x for x in hahb])
    EBV_hahb = np.log10((hahb)/2.86)/(-0.4*(k_ha - k_hb))
    
    EBV_corrs_ys[gooddata_iis] = EBV_hahb
    
    return EBV_corrs_ys


def EBV_errs_no_spectra(tab_no_spectra, mmt_mz, mmt_mz_EBV_hahb_errs_neg, mmt_mz_EBV_hahb_errs_pos,
    mmt_mz_EBV_hghb_errs_neg, mmt_mz_EBV_hghb_errs_pos,
    keck_mz, keck_mz_EBV_hahb_errs_neg, keck_mz_EBV_hahb_errs_pos):
    '''
    ## using composites

    Almost identical to EBV_corrs_no_spectra, except uses <instr>_mz_EBV_hahb_errs_<neg/pos> rather than
    <instr>_mz_EBV_hahb
    '''
    # loop based on filter
    EBV_errs_neg = np.array([-100.0]*len(tab_no_spectra))
    EBV_errs_pos = np.array([-100.0]*len(tab_no_spectra))
    for ff in ['NB704', 'NB711', 'NB816', 'NB921', 'NB973']:
        bin_filt_iis = np.array([x for x in range(len(tab_no_spectra)) if tab_no_spectra['filter'][x]==ff])
        # print('num in '+ff+':', len(bin_filt_iis))

        if ff=='NB704' or ff=='NB711' or ff=='NB816':
            # using MMT hahb errs for all of these sources
            tab_filt_iis = np.array([x for x in range(len(mmt_mz)) if 
                (ff in mmt_mz['filter'][x] and mmt_mz['stlrmass_bin'][x] != 'N/A')])
            m_bin = np.array([int(x[0])-1 for x in tab_no_spectra['stlrmassZbin_MMT'][bin_filt_iis]])

            for ii, m_i in enumerate(m_bin):
                EBV_errs_neg[bin_filt_iis[ii]] = mmt_mz_EBV_hahb_errs_neg[tab_filt_iis[m_i]]
                EBV_errs_pos[bin_filt_iis[ii]] = mmt_mz_EBV_hahb_errs_pos[tab_filt_iis[m_i]]

        elif ff == 'NB921':
            # for sources that fall within the keck mass range, we use keck hahb errs
            #  max_keck_mass should be 9.72
            max_keck_mass = float(keck_mz[np.where(keck_mz['filter']=='NB921')[0]]['stlrmass_bin'][-1]
                [keck_mz[np.where(keck_mz['filter']=='NB921')[0]]['stlrmass_bin'][-1].find('-')+1:])
            bin_filt_iis_keck = np.array([x for x in range(len(tab_no_spectra)) if 
                (tab_no_spectra['filter'][x]==ff and tab_no_spectra['stlrmass'][x]<=max_keck_mass)])
            tab_filt_iis_keck = np.array([x for x in range(len(keck_mz)) if 
                (keck_mz['filter'][x]==ff and keck_mz['stlrmass_bin'][x] != 'N/A')])
            m_bin_keck = np.array([int(x[0])-1 for x in tab_no_spectra['stlrmassZbin_Keck'][bin_filt_iis_keck]])

            for ii, m_i in enumerate(m_bin_keck):
                EBV_errs_neg[bin_filt_iis_keck[ii]] = keck_mz_EBV_hahb_errs_neg[tab_filt_iis_keck[m_i]]
                EBV_errs_pos[bin_filt_iis_keck[ii]] = keck_mz_EBV_hahb_errs_pos[tab_filt_iis_keck[m_i]]

            # for sources that fall above the keck mass range (so we use mmt hahb errs)
            bin_filt_iis_mmt = np.array([x for x in range(len(tab_no_spectra)) if 
                (tab_no_spectra['filter'][x]==ff and tab_no_spectra['stlrmass'][x]>max_keck_mass)])
            tab_filt_iis_mmt = np.array([x for x in range(len(mmt_mz)) if 
                (mmt_mz['filter'][x]==ff and mmt_mz['stlrmass_bin'][x] != 'N/A')])
            m_bin_mmt = np.array([int(x[0])-1 for x in tab_no_spectra['stlrmassZbin_MMT'][bin_filt_iis_mmt]])

            for ii, m_i in enumerate(m_bin_mmt):
                EBV_errs_neg[bin_filt_iis_mmt[ii]] = mmt_mz_EBV_hahb_errs_neg[tab_filt_iis_mmt[m_i]]
                EBV_errs_pos[bin_filt_iis_mmt[ii]] = mmt_mz_EBV_hahb_errs_pos[tab_filt_iis_mmt[m_i]]
            
        elif ff == 'NB973':
            # using Keck hahb errs instead of MMT hahb errs 
            tab_filt_iis = np.array([x for x in range(len(keck_mz)) if 
                (keck_mz['filter'][x]==ff and keck_mz['stlrmass_bin'][x] != 'N/A')])
            m_bin = np.array([int(x[0])-1 for x in tab_no_spectra['stlrmassZbin_Keck'][bin_filt_iis]])

            for ii, m_i in enumerate(m_bin):
                EBV_errs_neg[bin_filt_iis[ii]] = keck_mz_EBV_hahb_errs_neg[tab_filt_iis[m_i]]
                EBV_errs_pos[bin_filt_iis[ii]] = keck_mz_EBV_hahb_errs_pos[tab_filt_iis[m_i]]
    #endfor

    assert len([x for x in EBV_errs_neg if x==-100.0]) == 0
    assert len([x for x in EBV_errs_pos if x==-100.0]) == 0
    return EBV_errs_neg, EBV_errs_pos


def EBV_errs_yes_spectra(EBV_errs_ys_neg, EBV_errs_ys_pos, yes_spectra, HA_SNR, HB_SNR, HA_FLUX, HB_FLUX):
    '''
    HA_RMS    = HA_FLUX/HA_SNR
    HB_RMS    = HB_FLUX/HB_SNR
    '''
    gooddata_iis = np.where((HB_SNR[yes_spectra] >= 5) & (HA_SNR[yes_spectra] > 0) & (HA_FLUX[yes_spectra] > 1e-20) & (HA_FLUX[yes_spectra] < 99))[0]
    good_EBV_iis = yes_spectra[gooddata_iis]
    ebv_hahb_errs = composite_errors([HA_FLUX[good_EBV_iis], HB_FLUX[good_EBV_iis]], 
        [HA_FLUX[good_EBV_iis]/HA_SNR[good_EBV_iis], HB_FLUX[good_EBV_iis]/HB_SNR[good_EBV_iis]], seed_i=SEED_ORIG, label='HA/HB')
    EBV_errs_ys_neg[gooddata_iis] = ebv_hahb_errs[:,0]
    EBV_errs_ys_pos[gooddata_iis] = ebv_hahb_errs[:,1]

    return EBV_errs_ys_neg, EBV_errs_ys_pos


def get_NBIA_errs(nbiaerrsdata, filtarr, FILT):
    '''
    '''
    NBIA_errs_neg = np.array([-100.0]*len(nbiaerrsdata))
    NBIA_errs_pos = np.array([-100.0]*len(nbiaerrsdata))
    for ff in filtarr:
        filt_iis = np.where(FILT==ff)[0]  # where 1080-indexing == relevant filter
        NBIA_errs_neg[filt_iis] = nbiaerrsdata[ff+'_FLUX_LOERROR'][filt_iis]
        NBIA_errs_pos[filt_iis] = nbiaerrsdata[ff+'_FLUX_UPERROR'][filt_iis]

    return NBIA_errs_neg, NBIA_errs_pos


def exclude_bad_sources(ha_ii, NAME0):
    '''
    excludes sources from the mainseq sample depending on their unreliable properties
    '''
    # getting rid of special cases (no_spectra):
    bad_highz_gal = np.where(NAME0=='Ha-NB816_174829_Ha-NB921_187439_Lya-IA598_163379')[0]

    bad_HbNB704_SIINB973_gals = np.array([x for x in range(len(ha_ii)) if 
            (NAME0[x]=='Ha-NB704_028405_OII-NB973_056979' or 
             NAME0[x]=='Ha-NB704_090945_OII-NB973_116533')])

    # getting rid of a source w/o flux (yes_spectra):
    no_flux_gal = np.where(NAME0=='Ha-NB921_069950')[0]

    # getting rid of a source w/ atypical SFR behavior we don't understand
    weird_SFR_gal = np.where(NAME0=='OIII-NB704_063543_Ha-NB816_086540')[0]

    # getting rid of AGN candidate galaxies
    possibly_AGNs = np.array([x for x in range(len(ha_ii)) if 
            (NAME0[x]=='Ha-NB921_063859' or NAME0[x]=='Ha-NB973_054540' or
             NAME0[x]=='Ha-NB973_064347' or NAME0[x]=='Ha-NB973_084633')])

    bad_sources = np.concatenate([bad_highz_gal, bad_HbNB704_SIINB973_gals, no_flux_gal, weird_SFR_gal, possibly_AGNs])
    print("bad_sources: {}".format(len(bad_sources)))

    ha_ii = np.delete(ha_ii, bad_sources)
    NAME0 = np.delete(NAME0, bad_sources)

    return ha_ii, NAME0


def main():
    # reading in data
    nbia = pyfits.open(FULL_PATH+'Catalogs/NB_IA_emitters.nodup.colorrev.fix.fits')
    nbiadata = nbia[1].data
    allcols = pyfits.open(FULL_PATH+'Catalogs/NB_IA_emitters.allcols.colorrev.fits')
    allcolsdata0 = allcols[1].data
    NAME0 = np.array(nbiadata['NAME'])
    ID0   = np.array(nbiadata['ID'])
    zspec = asc.read(FULL_PATH+'Catalogs/nb_ia_zspec.txt',guess=False,
                     Reader=asc.CommentedHeader)
    zspec0 = np.array(zspec['zspec0'])
    inst_str0 = np.array(zspec['inst_str0'])
    fout  = asc.read(FULL_PATH+'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX.fout',
                     guess=False,Reader=asc.NoHeader)
    stlr_mass = np.array(fout['col7'])
    data_dict = create_ordered_AP_arrays()
    AP = data_dict['AP']
    HA_FLUX   = data_dict['HA_FLUX']
    HB_FLUX   = data_dict['HB_FLUX']
    HA_SNR    = data_dict['HA_SNR']
    HB_SNR    = data_dict['HB_SNR']
    NIIB_FLUX = data_dict['NIIB_FLUX']
    NIIB_SNR  = data_dict['NIIB_SNR']
    nbiaerrs = pyfits.open(FULL_PATH+'Catalogs/NB_IA_emitters.allcols.colorrev.fix.errors.fits')
    nbiaerrsdata0 = nbiaerrs[1].data


    # defining other useful data structs
    filtarr = np.array(['NB704', 'NB711', 'NB816', 'NB921', 'NB973'])
    inst_dict = {}
    inst_dict['MMT']  = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,']
    inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,','Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,']


    # limit all data to Halpha emitters only
    ha_ii = np.array([x for x in range(len(NAME0)) if 'Ha-NB' in NAME0[x]])
    NAME0       = NAME0[ha_ii]

    # getting rid of unreliable galaxies:
    ha_ii, NAME0 = exclude_bad_sources(ha_ii, NAME0)

    ID0         = ID0[ha_ii]
    zspec0      = zspec0[ha_ii]
    inst_str0   = inst_str0[ha_ii]
    stlr_mass   = stlr_mass[ha_ii]
    AP          = AP[ha_ii]
    HA_FLUX     = HA_FLUX[ha_ii]
    HB_FLUX     = HB_FLUX[ha_ii]
    HA_SNR      = HA_SNR[ha_ii]
    HB_SNR      = HB_SNR[ha_ii]
    NIIB_FLUX   = NIIB_FLUX[ha_ii]
    NIIB_SNR    = NIIB_SNR[ha_ii]
    allcolsdata = allcolsdata0[ha_ii]
    nbiaerrsdata = nbiaerrsdata0[ha_ii]

    no_spectra  = np.where((zspec0 <= 0) | (zspec0 > 9))[0]
    yes_spectra = np.where((zspec0 >= 0) & (zspec0 < 9))[0]


    # reading in EBV data tables & getting relevant EBV cols
    mmt_m   = asc.read(FULL_PATH+'Composite_Spectra/StellarMass/MMT_all_five_data.txt')
    mmt_m_EBV_hahb = np.array(mmt_m['E(B-V)_hahb'])
    mmt_m_EBV_hghb = np.array(mmt_m['E(B-V)_hghb'])
    
    keck_m  = asc.read(FULL_PATH+'Composite_Spectra/StellarMass/Keck_all_five_data.txt')
    keck_m_EBV_hahb = np.array(keck_m['E(B-V)_hahb'])
    
    mmt_mz  = asc.read(FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt')
    mmt_mz_EBV_hahb = np.array(mmt_mz['E(B-V)_hahb'])
    mmt_mz_EBV_hahb_errs_neg = np.array(mmt_mz['E(B-V)_hahb_errs_neg'])
    mmt_mz_EBV_hahb_errs_pos = np.array(mmt_mz['E(B-V)_hahb_errs_pos'])
    mmt_mz_EBV_hghb = np.array(mmt_mz['E(B-V)_hghb'])
    mmt_mz_EBV_hghb_errs_neg = np.array(mmt_mz['E(B-V)_hghb_errs_neg'])
    mmt_mz_EBV_hghb_errs_pos = np.array(mmt_mz['E(B-V)_hghb_errs_pos'])
    
    keck_mz = asc.read(FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt')
    keck_mz_EBV_hahb = np.array(keck_mz['E(B-V)_hahb'])
    keck_mz_EBV_hahb_errs_neg = np.array(keck_mz['E(B-V)_hahb_errs_neg'])
    keck_mz_EBV_hahb_errs_pos = np.array(keck_mz['E(B-V)_hahb_errs_pos'])


    # mass 'bin' lists made by reading in from files generated by the stack plots
    masslist_MMT  = np.array([x.split('-') for x in mmt_m['stlrmass_bin']], dtype=float) 
    masslist_Keck = np.array([x.split('-') for x in keck_m['stlrmass_bin']], dtype=float)

    # same with massZ 'bin' lists (also getting the filts)
    massZlist_MMT  = np.array([x.split('-') for x in mmt_mz['stlrmass_bin'] if x!='N/A'], dtype=float)
    massZlist_Keck = np.array([x.split('-') for x in keck_mz['stlrmass_bin']], dtype=float)
    massZlist_filts_MMT  = np.array([mmt_mz['filter'][x] for x in range(len(mmt_mz)) if mmt_mz['stlrmass_bin'][x]!='N/A'])
    massZlist_filts_Keck = np.array([keck_mz['filter'][x] for x in range(len(keck_mz))])

    # splitting the 'bin' lists from above into a flattened bins array
    massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck = get_bins(masslist_MMT, 
        masslist_Keck, massZlist_MMT, massZlist_Keck)

    # getting tables of which bins the sources fall into (for eventual EBV corrections)
    #  populate w/ no_spectra sources first, then populate w/ yes_spectra sources, then consolidate
    tab_no_spectra = bins_table(no_spectra, NAME0, AP, stlr_mass, 
        massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck, massZlist_filts_MMT, massZlist_filts_Keck)
    tab_yes_spectra = bins_table(yes_spectra, NAME0, AP, stlr_mass, 
        massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck, massZlist_filts_MMT, massZlist_filts_Keck)
    FILT = consolidate_ns_ys(allcolsdata, no_spectra, yes_spectra,
        np.array(tab_no_spectra['filter']), np.array(tab_yes_spectra['filter']), datatype='str')


    print('### obtaining filter corrections')
    # dual nb704/nb711 emitters are corrected w/ only nb704 as the dual emitters
    #  fall more centrally within the nb704 filter profile
    orig_fluxes = np.zeros(len(allcolsdata))
    filt_corr_factor = np.zeros(len(allcolsdata))
    sigma = np.zeros(len(allcolsdata))
    flux_3sigcutoffs = {'NB704':np.log10(5.453739e-18), 'NB711':np.log10(6.303345e-18), 
         'NB816':np.log10(4.403077e-18), 'NB921':np.log10(5.823993e-18), 'NB973':np.log10(1.692677e-17)}
    for filt in filtarr:
        filt_ii = np.array([x for x in range(len(FILT)) if filt in FILT[x]])

        no_spectra_temp  = np.array([x for x in filt_ii if x in no_spectra])
        yes_spectra_temp = np.array([x for x in filt_ii if x in yes_spectra])

        orig_fluxes[filt_ii] = allcolsdata[filt+'_FLUX'][filt_ii]
        sigma[filt_ii] = 3*10**(orig_fluxes[filt_ii] - flux_3sigcutoffs[filt])

        filt_corr_factor = apply_filt_corrs_interp(filt, filt_corr_factor, zspec0, 
            no_spectra_temp, yes_spectra_temp, AP, allcolsdata)
    #endfor


    print('### obtaining nii_ha corrections')
    linedict = plot_NII_Ha_ratios.main()
    C = linedict['C']
    b = linedict['b']
    m = linedict['m']

    nii_ha_ratio = np.zeros(len(allcolsdata))
    ratio_vs_line = np.array(['']*len(allcolsdata), dtype='|S5')

    # NIIB_SNR >= 2
    highSNR = np.array([x for x in range(len(NIIB_SNR)) 
        if (NIIB_SNR[x] >= 2 and NIIB_FLUX[x] != 0 and HA_FLUX[x] < 99)])
    nii_ha_ratios_tmp = (1+2.96)/2.96*NIIB_FLUX[highSNR]/HA_FLUX[highSNR]
    agns_ii = np.where(nii_ha_ratios_tmp > 0.54*(1+2.96)/2.96)[0]
    highSNR = np.delete(highSNR, agns_ii)
    nii_ha_ratio[highSNR] = (1+2.96)/2.96*NIIB_FLUX[highSNR]/HA_FLUX[highSNR]
    ratio_vs_line[highSNR] = 'ratio'

    # NIIB_SNR < 2
    lowSNR = np.array([x for x in range(len(NIIB_SNR)) if x not in highSNR])
    ratio_vs_line[lowSNR] = 'line'
    llow_m  = np.array([x for x in range(len(lowSNR)) if stlr_mass[lowSNR][x] <= 8])
    lhigh_m = np.array([x for x in range(len(lowSNR)) if stlr_mass[lowSNR][x] > 8])
    nii_ha_ratio[lowSNR[llow_m]] = C
    nii_ha_ratio[lowSNR[lhigh_m]] = m*(stlr_mass[lowSNR[lhigh_m]])+b

    nii_ha_corr_factor = np.log10(1/(1+nii_ha_ratio))


    # getting the corrected EBVs to use
    #  yes_spectra originally gets the no_spectra and then individual ones are applied 
    #  if S/N is large enough
    EBV_corrs_ns = EBV_corrs_no_spectra(tab_no_spectra, mmt_mz, mmt_mz_EBV_hahb,
        mmt_mz_EBV_hghb, keck_mz, keck_mz_EBV_hahb)
    EBV_corrs_ys = EBV_corrs_no_spectra(tab_yes_spectra, mmt_mz, mmt_mz_EBV_hahb, 
        mmt_mz_EBV_hghb, keck_mz, keck_mz_EBV_hahb)
    EBV_corrs_ys = EBV_corrs_yes_spectra(EBV_corrs_ys, yes_spectra, HA_FLUX, HB_FLUX, HB_SNR, HA_SNR)
    
    EBV_corrs = consolidate_ns_ys(orig_fluxes, no_spectra, yes_spectra,
        EBV_corrs_ns, EBV_corrs_ys)


    # getting the EBV errors
    #  yes_spectra originally gets the no_spectra and then individual ones are applied
    #  if S/N is large enough
    EBV_errs_ns_neg, EBV_errs_ns_pos = EBV_errs_no_spectra(tab_no_spectra, 
        mmt_mz, mmt_mz_EBV_hahb_errs_neg, mmt_mz_EBV_hahb_errs_pos,
        mmt_mz_EBV_hghb_errs_neg, mmt_mz_EBV_hghb_errs_pos,
        keck_mz, keck_mz_EBV_hahb_errs_neg, keck_mz_EBV_hahb_errs_pos)
    EBV_errs_ys_neg, EBV_errs_ys_pos = EBV_errs_no_spectra(tab_yes_spectra, 
        mmt_mz, mmt_mz_EBV_hahb_errs_neg, mmt_mz_EBV_hahb_errs_pos,
        mmt_mz_EBV_hghb_errs_neg, mmt_mz_EBV_hghb_errs_pos,
        keck_mz, keck_mz_EBV_hahb_errs_neg, keck_mz_EBV_hahb_errs_pos)
    EBV_errs_ys_neg, EBV_errs_ys_pos = EBV_errs_yes_spectra(EBV_errs_ys_neg, EBV_errs_ys_pos, 
        yes_spectra, HA_SNR, HB_SNR, HA_FLUX, HB_FLUX)

    EBV_errs_neg = consolidate_ns_ys(orig_fluxes, no_spectra, yes_spectra,
        EBV_errs_ns_neg, EBV_errs_ys_neg)
    EBV_errs_pos = consolidate_ns_ys(orig_fluxes, no_spectra, yes_spectra,
        EBV_errs_ns_pos, EBV_errs_ys_pos)
    EBV_errs = np.sqrt(EBV_errs_neg**2/2 + EBV_errs_pos**2/2)


    # getting the NBIA errors
    NBIA_errs_neg, NBIA_errs_pos = get_NBIA_errs(nbiaerrsdata, filtarr, FILT)
    NBIA_errs = np.sqrt(NBIA_errs_neg**2/2 + NBIA_errs_pos**2/2)


    # getting dust extinction factors and corrections
    k_ha = cardelli(HA * u.Angstrom)
    A_V = k_ha * EBV_corrs
    dustcorr_fluxes = orig_fluxes + 0.4*A_V # A_V = A(Ha) = extinction at Ha
    dust_corr_factor = dustcorr_fluxes - orig_fluxes
    dust_errs = 0.4*(k_ha * EBV_errs)

    # getting the errors associated w/ measurements by adding in quadrature
    meas_errs = np.sqrt(NBIA_errs**2 + dust_errs**2)


    # getting luminosities
    filt_cen = [7045, 7126, 8152, 9193, 9749] # angstroms, from the response files
    tempz = np.array([-100.0]*len(no_spectra))
    for ff, ii in zip(filtarr, range(len(filt_cen))):
        filt_match = np.array([x for x in range(len(no_spectra)) if tab_no_spectra['filter'][x]==ff])
        tempz[filt_match] = filt_cen[ii]/HA - 1
    #endfor
    lum_dist_ns = (cosmo.luminosity_distance(tempz).to(u.cm).value)
    lum_factor_ns = np.log10(4*np.pi)+2*np.log10(lum_dist_ns) # = log10(L[Ha])
    lum_dist_ys = (cosmo.luminosity_distance(zspec0[yes_spectra]).to(u.cm).value)
    lum_factor_ys = np.log10(4*np.pi)+2*np.log10(lum_dist_ys)
    lum_factors = consolidate_ns_ys(orig_fluxes, no_spectra, yes_spectra,
        lum_factor_ns, lum_factor_ys)

    orig_lums = orig_fluxes + lum_factors


    # # getting SFR
    #  7.9E-42 is conversion btwn L and SFR based on Kennicutt 1998 for Salpeter IMF. 
    #  We use 1.8 to convert to Chabrier IMF.
    orig_sfr = np.log10(7.9/1.8) - 42 + orig_lums

    # # getting metallicity-dependent SFR corrections based on Ly+16 (MACT 1)
    # nii_ha_ratio is NII_6548,6583/Ha = (1+2.96)/2.96 * NII_6583/Ha
    NII6583_Ha = nii_ha_ratio * 2.96/(1+2.96)
    logOH = niiha_oh_determine(np.log10(NII6583_Ha), 'PP04_N2') - 12   # since this code estimates log(O/H)+12
    y = logOH + 3.31 
    log_SFR_LHa = -41.34 + 0.39*y + 0.127*y**2

    # # metallicity-dep SFRs
    log_SFR = log_SFR_LHa + orig_lums

    
    # write some table so that plot_nbia_mainseq.py can read this in
    tab00 = Table([ID0, NAME0, FILT, inst_str0, zspec0, stlr_mass, sigma, 
        orig_fluxes, orig_lums, orig_sfr, log_SFR,
        filt_corr_factor, nii_ha_corr_factor, nii_ha_ratio, ratio_vs_line,
        A_V, EBV_corrs, dust_corr_factor, EBV_errs, dust_errs,  NBIA_errs, meas_errs], 
        names=['ID', 'NAME0', 'filt', 'inst_str0', 'zspec0', 'stlr_mass', 'flux_sigma', 
        'obs_fluxes', 'obs_lumin', 'obs_sfr', 'met_dep_sfr',
        'filt_corr_factor', 'nii_ha_corr_factor', 'NII_Ha_ratio', 'ratio_vs_line', 
        'A(Ha)', 'EBV', 'dust_corr_factor', 'EBV_errs', 'dust_errs', 'NBIA_errs', 'meas_errs'])
    asc.write(tab00, FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt',
        format='fixed_width_two_line', delimiter=' ', overwrite=True)


if __name__ == '__main__':
    main()