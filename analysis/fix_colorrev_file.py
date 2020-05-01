"""
fix_colorrev_file
====

This code will modify the colorrev.fits file to use updated spectroscopic
redshifts from MMT/Hectospec and Keck/DEIMOS.  Some SDF NB excess emitters
were mis-classified using color information
"""

from __future__ import print_function

from chun_codes import systime, intersect, match_nosort

from astropy.io import ascii as asc
from astropy.io import fits
from astropy import log
from astropy.table import Table

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from os.path import join
import pandas as pd

from ..mainseq_corrections import exclude_bad_sources, handle_unusual_dual_emitters

dir0 = '/Users/cly/GoogleDrive/Research/NASA_Summer2015/'


def NB_spec_redshift(filt):
    """
    Redshift for NB excess emitter selection for various emission lines

    Parameters
    ----------
    filt : str
      Name of filter. Either 'NB704', 'NB711', 'NB816', 'NB921', 'NB973'

    Returns
    -------
    z1, z2, z3, z4, z5, z6, z7, z8, z9, z10 : float
      Minimum and maximum redshift for various emission lines

    Notes
    -----
    Created by Chun Ly, 27 January 2018
     - This is a Python version of IDL's code, NB_spec_redshift.pro
    Modified by Chun Ly, 29 January 2018
     - Add IA598 and IA679 redshift limits
     - Redshift limit change for OII-NB921 and Ha-NB921
    Modified by Chun Ly, 31 January 2018
     - Include and return ltype
     - Add MgII for NB704 and NB711
    Modified by Chun Ly, 1 February 2018
     - Add MgII for IA679
     - Define and return z_vals
    """

    if filt == 'NB704':
        z1,   z2 = 0.050, 0.100  # H-alpha
        z3,   z4 = 0.370, 0.475  # OIII
        z5,   z6 = 0.870, 0.910  # OII
        z7,   z8 = 4.600, 4.900  # Ly-alpha
        z9,  z10 = 0.800, 0.850  # NeIII
        z11, z12 = 1.460, 1.560  # MgII | + on 31/01/2018
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII', 'MgII']

    if filt == 'NB711':
        z1,   z2 = 0.050, 0.100  # H-alpha
        z3,   z4 = 0.375, 0.475  # OIII
        z5,   z6 = 0.875, 0.940  # OII
        z7,   z8 = 4.650, 4.900  # Ly-alpha
        z9,  z10 = 0.800, 0.870  # NeIII
        z11, z12 = 1.460, 1.560  # MgII | + on 31/01/2018
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII', 'MgII']

    if filt == 'NB816':
        z1,  z2 = 0.210, 0.260  # H-alpha
        z3,  z4 = 0.600, 0.700  # OIII
        z5,  z6 = 1.150, 1.225  # OII
        z7,  z8 = 5.600, 5.800  # Ly-alpha
        z9, z10 = 1.075, 1.150  # NeIII
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII']

    if filt == 'NB921':
        z1,  z2 = 0.385, 0.429  # H-alpha
        z3,  z4 = 0.810, 0.910  # OIII
        z5,  z6 = 1.445, 1.492  # OII
        z7,  z8 = 6.520, 6.630  # Ly-alpha
        z9, z10 = 0.000, 0.000  # NeIII
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII']

    if filt == 'NB973':
        z1,  z2 = 0.450, 0.520  # H-alpha
        z3,  z4 = 0.940, 0.975  # OIII
        z5,  z6 = 1.585, 1.620  # OII
        z7,  z8 = 6.950, 7.100  # Ly-alpha
        z9, z10 = 0.000, 0.000  # NeIII
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII']

    # + on 29/01/2018
    if filt == 'IA598':
        z1,  z2 = 0.000, 0.000  # Don't really use this
        z3,  z4 = 0.150, 0.300  # [OIII]/H-beta
        z5,  z6 = 0.550, 0.650  # [OII]
        z7,  z8 = 3.600, 4.100  # Ly-alpha
        z9, z10 = 0.000, 0.000  # NeIII
        # Also, we have 1 CIII] 1909. No [NeIII]
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII']

    if filt == 'IA679':
        z1,   z2 = 0.000, 0.080  # Ha. Don't really use this
        z3,   z4 = 0.300, 0.450  # [OIII]/H-beta
        z5,   z6 = 0.750, 0.950  # [OII]
        z7,   z8 = 4.200, 4.900  # Ly-alpha
        z9,  z10 = 0.000, 0.000  # NeIII
        z11, z12 = 1.500, 1.510  # MgII
        # Also, we have 1 CIII] 1909 and 1 or 2 CIV 1549. No [NeIII]
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII', 'MgII']

    return z_vals, ltype


def read_nb_catalog(filt='', ID=None, use_fix=False):
    """
    Purpose:
      Read in NB/IA excess photometric catalogs

    :param filt: Optional input.  Use for main_color() to restrict sample to
                 a specific NB sample. Options are:
                   'NB704', 'NB711', 'NB816', 'NB921', or 'NB973'
    :param ID: Optional input.  Use for cross-matching to specific filt
    :param use_fix: bool to indicate whether to use revised colorrev file
                    (see main()) based on new spec-z or the old one
    :return:
    """

    orig_file   = dir0+'Catalogs/NB_IA_emitters.nodup.fits'
    colorrev_file = orig_file.replace('.fits', '.colorrev.fits')
    if use_fix:
        colorrev_file = colorrev_file.replace('.fits', '.fix.fits')

    log.info('### Reading : '+orig_file)
    raw_data = fits.getdata(orig_file)

    log.info('### Reading : '+colorrev_file)
    c_data, c_hdr = fits.getdata(colorrev_file, header=True)

    raw_Name = np.array([str0.replace(' ', '') for str0 in raw_data.NAME],
                        dtype='|S67')
    rev_Name = np.array([str0.replace(' ', '') for str0 in c_data.NAME])

    if not isinstance(ID, type(None)):
        allcol_file = dir0+'Catalogs/NB_IA_emitters.allcols.fits'
        log.info('### Reading : '+allcol_file)
        allcol_data = fits.getdata(allcol_file)

        idx1, NBIA_idx = match_nosort(ID, allcol_data[filt+'_ID'])

        raw_data = raw_data[NBIA_idx]
        c_data = c_data[NBIA_idx]
        raw_Name = raw_Name[NBIA_idx]
        rev_Name = rev_Name[NBIA_idx]

    corr_Name = raw_Name.copy()

    return colorrev_file, raw_data, c_data, c_hdr, raw_Name, rev_Name, \
        corr_Name


def read_zspec_data():
    """
    Purpose:
      Read in z-spec data

    :return:
    z_data: astropy table
    z_spec0: spec-z data
    with_z : numpy index array with spec-z
    without_z : numpy index array without spec-z
    """

    zspec_file = dir0+'Catalogs/nb_ia_zspec.txt'
    log.info('### Reading : '+zspec_file)
    z_data = asc.read(zspec_file)

    z_spec0 = z_data['zspec0']

    # Note: This should give 1989. Which matches the number of spectra
    # in spec_match/1014/NB_IA_emitters.spec.fits (current 'current')
    # in_z_cat = np.where((z_spec0 != -10))[0]

    # Note: This yields 1519 galaxies
    with_z = np.where((z_spec0 != -10) & (z_spec0 < 9.999) &
                      (z_spec0 != -1.0))[0]

    # + on 30/01/2018
    without_z = np.where((z_spec0 == -10) | (z_spec0 >= 9.999) |
                         (z_spec0 == -1.0))[0]

    return z_data, z_spec0, with_z, without_z


def handle_bad_sources_dual_emitters(Ha_index, rev_Name, filt):

    Ha_index_exclude, rev_Name_exclude = exclude_bad_sources(Ha_index, rev_Name)

    filts, dual_iis, dual_ii2 = handle_unusual_dual_emitters(rev_Name_exclude)

    final_Ha_index = np.copy(Ha_index_exclude)
    final_Ha_rev_Name = np.copy(rev_Name_exclude)

    if filt != 'NB921':
        if len(dual_iis) > 0:
            print("dual_iis: {}".format(len(dual_iis)))
            final_Ha_index = np.delete(final_Ha_index, dual_iis)
            final_Ha_rev_Name = np.delete(final_Ha_rev_Name, dual_iis)

        if len(dual_ii2) > 0:
            print("dual_ii2: {}".format(len(dual_ii2)))
            final_Ha_index = np.delete(final_Ha_index, dual_ii2)
            final_Ha_rev_Name = np.delete(final_Ha_rev_Name, dual_ii2)

    return final_Ha_index, final_Ha_rev_Name


def main(silent=False):
    """
    Main function for fix_colorrev_file.py

    Parameters
    ----------

    silent : boolean
      Turns off stdout messages. Default: False


    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 26 January 2018
    Modified by Chun Ly, 27 January 2018
     - Add inset plot that zooms in for H-alpha to [OII]
     - Plot aesthetics
    Modified by Chun Ly, 29 January 2018
     - Call NB_spec_redshift()
     - Draw vertical lines for redshift selection
     - Read in original NBIA no-dup FITS file
     - Add IA598 and IA679 filters for update
     - Plot aesthetics - Avoid when min/max redshift if both equal to zero
     - Update galaxy name with redshift info
    Modified by Chun Ly, 30 January 2018
     - Change Name of source to include emission line info
     - Write changes to source name to file
     - Transfer colorrev info from rev_Name to corr_Name for those without redshift
     - Write redshift info (ID, redshift) to outfile
     - Bug fix: On without_z definition
     - Bug fix: Address string loss with np.array dtype default settings
     - Logging via print statements
     - QA check - Get spec-z cases that don't have em-line info updated
     - Write nochange ASCII table
    Modified by Chun Ly, 31 January 2018
     - Get ltype from NB_spec_redshift()
     - Handle more than the usual sets of emission lines
     - Bug fix: Incorrect boolean statement. Allow for zmin = 0.0
    Modified by Chun Ly,  1 February 2018
     - Write updated colorrev file
    """
    
    if not silent:
        log.info('### Begin main : '+systime())

    # Read in NB/IA photometric data
    colorrev_file, raw_data, c_data, c_hdr, raw_Name, rev_Name, corr_Name = \
        read_nb_catalog()

    # Read in spec-z dataset
    z_data, z_spec0, with_z, without_z = read_zspec_data()

    corr_Name[without_z] = rev_Name[without_z]

    filt0 = ['NB704', 'NB711', 'NB816', 'NB921', 'NB973', 'IA598', 'IA679']

    out_pdf = dir0+'Plots/NB_IA_zspec.pdf'
    pp = PdfPages(out_pdf)

    z_max = [1.05, 1.05, 1.3, 1.55, 1.70, 0.70, 0.96]
    for ff in range(len(filt0)):
        idx = [xx for xx in range(len(c_data)) if filt0[ff] in c_data.NAME[xx]]
        idx_z = intersect(idx, with_z)

        z_vals, ltype = NB_spec_redshift(filt0[ff])

        fig, ax = plt.subplots()
        ax.hist(z_spec0[idx_z], bins=500, alpha=0.5, edgecolor='none',
                histtype='bar', align='mid')
        ax.set_xlabel('Spectroscopic Redshift')
        ax.set_ylabel('Number of Spectra')
        ax.minorticks_on()
        ax.annotate(filt0[ff], [0.025, 0.975], xycoords='axes fraction',
                    ha='left', va='top')

        # zoom-in inset panel | + on 12/12/2016
        axins = inset_axes(ax, width=4., height=4., loc=1)
        axins.hist(z_spec0[idx_z], bins=500, alpha=0.5, edgecolor='none',
                   histtype='bar', align='mid')
        axins.set_xlim([0.0, z_max[ff]])
        axins.set_xlabel(r'$z_{\rm spec}$')
        # axins.set_ylim([0.0,max(N)])
        axins.minorticks_on()

        # Draw vertical lines for selection | + on 29/01/2018
        ctype = ['red', 'green', 'blue', 'black', 'purple', 'magenta']
        # ltype = [ 'Ha', 'OIII', 'OII',  'Lya', 'NeIII']

        for zz in range(len(z_vals)/2):
            if z_vals[2*zz] != z_vals[2*zz+1]:
                ax.axvline(x=z_vals[2*zz],   color=ctype[zz], linestyle='dashed')
                ax.axvline(x=z_vals[2*zz+1], color=ctype[zz], linestyle='dashed')

                axins.axvline(x=z_vals[2*zz],   color=ctype[zz], linestyle='dashed')
                axins.axvline(x=z_vals[2*zz+1], color=ctype[zz], linestyle='dashed')

                # Update galaxy name with redshift info | + on 29/01/2018
                z_line = np.array([xx for xx in range(len(idx_z)) if
                                   (z_spec0[idx_z][xx] >= z_vals[2*zz]) &
                                   (z_spec0[idx_z][xx] <= z_vals[2*zz+1])])
                print('%s %.2f %.2f %03i' % ((ltype[zz]+'-'+filt0[ff]).rjust(11), z_vals[2*zz],
                                             z_vals[2*zz+1], len(z_line)))

                # + on 30/01/2018
                if len(z_line) > 0:
                    z_temp = np.array(idx_z)[z_line]
                    new_Name = [str0.replace(filt0[ff], ltype[zz]+'-'+filt0[ff]) for
                                str0 in corr_Name[z_temp]]
                    corr_Name[z_temp] = new_Name
            # endif
        # end for

        fig.set_size_inches(8, 8)
        fig.savefig(pp, format='pdf', bbox_inches='tight')

        # Get spec-z cases that don't have color info updated | + on 30/01/2018
        no_change = [cc for cc in range(len(idx_z))
                     if '-'+filt0[ff] not in corr_Name[idx_z[cc]]]
        if len(no_change) == 0:
            log.info('## All sources updated')
        else:
            tab_temp = z_data[idx_z[no_change]]
            tab_temp.sort('zspec0')
            outfile2 = dir0+'Catalogs/'+filt0[ff]+'_nochange.tbl'
            log.info('### Writing : '+outfile2)
            tab_temp.write(outfile2, format='ascii.fixed_width_two_line',
                           overwrite=True)
    # end for

    pp.close()

    # Write changes to source name to file | + on 30/01/2018
    change = [xx for xx in range(len(raw_Name)) if corr_Name[xx] != rev_Name[xx]]
    print('## len of [change] : ', len(change))

    # Mod on 30/01/2018
    z_data_ch = z_data[change]
    arr0 = zip(z_data_ch['ID0'], z_data_ch['zspec0'], z_data_ch['slit_str0'],
               rev_Name[change], corr_Name[change])
    change_str0 = [str(a)+' '+str(b)+' '+c+' '+d+' -> '+e+'\n' for
                   a, b, c, d, e in arr0]

    outfile = dir0+'Catalogs/fix_colorrev_file.dat'
    log.info('## Writing : '+outfile)
    f0 = open(outfile, 'w')
    f0.writelines(change_str0)
    f0.close()

    # + on 01/02/2018
    c_data.NAME = corr_Name
    fits.writeto(colorrev_file.replace('.fits', '.fix.fits'), c_data, c_hdr)
    if not silent:
        log.info('### End main : '+systime())


def main_color(old_selection=False):
    """
    Purpose:
      Update selection using revised color selection for non spec-z

    :param old_selection: boolean to indicate whether to use old selection or new. Default: False
    :return:
    """

    # Read in z-spec data
    # z_data, z_spec0, with_z, without_z = read_zspec_data()

    filters = ['NB704', 'NB711', 'NB816', 'NB921', 'NB973']

    for filt in filters:
        # Read in photometric data
        phot_file = join(dir0, 'Plots/color_plots/{}_phot.csv'.format(filt))
        phot_df = pd.read_csv(phot_file)
        good_phot = phot_df['good_phot']
        NB_zspec = phot_df['zspec']

        # Read in NB/IA photometric data
        colorrev_file, raw_data, c_data, c_hdr, raw_Name, rev_Name, corr_Name = \
            read_nb_catalog(filt=filt, ID=phot_df['ID'], use_fix=True)

        N_NB = len(phot_df)
        print("N ({}) : {}".format(filt, N_NB))

        with_specz = np.where(NB_zspec != -10.0)[0]
        print("with spec-z ({}) : {}".format(filt, len(with_specz)))

        Ha_orig_full = np.array([xx for xx in range(N_NB) if 'Ha-'+filt in rev_Name[xx]])

        final_Ha_index, final_Ha_rev_Name = \
            handle_bad_sources_dual_emitters(Ha_orig_full, rev_Name[Ha_orig_full], filt)

        test_tab = Table([final_Ha_rev_Name], names=['final_Ha_ref_name'])
        exclude_filename = 'Plots/color_plots/{}_Ha_exclude_names.txt'.format(filt)
        test_tab.write(join(dir0, exclude_filename), format='ascii.fixed_width_two_line')

        print("N(H-alpha) original ({}) : {} -> {}".format(filt, len(Ha_orig_full),
                                                           len(final_Ha_index)))

        # Mark those with spec-z in H-alpha
        z_vals, _ = NB_spec_redshift(filt)
        Ha_zspec = np.where((NB_zspec >= z_vals[0]) & (NB_zspec <= z_vals[1]))[0]
        print("N(H-alpha) with spec-z ({}) : {}".format(filt, len(Ha_zspec)))
        Ha_zspec_file = join(dir0, 'Plots/color_plots/{}_Ha_zspec.txt'.format(filt))
        print("Writing : "+Ha_zspec_file)
        f0 = open(Ha_zspec_file, 'w')
        f0.writelines([str0+'\n' for str0 in rev_Name[Ha_zspec]])
        f0.close()

        if 'NB7' in filt:
            # NB704 and NB711 selection
            VR = phot_df['VR']
            Ri = phot_df['Ri']

            # Old selection
            if old_selection:
                Ha_sel = np.where((VR <= 0.82 * Ri + 0.264) & (VR >= 2.5 * Ri - 0.24) &
                                  good_phot & ((NB_zspec == -10) | (NB_zspec == -1) | (NB_zspec >= 9.9)))[0]
            else:
                Ha_sel = np.where((VR <= 0.84 * Ri + 0.125) & (VR >= 2.5 * Ri - 0.24) &
                                  good_phot & ((NB_zspec == -10) | (NB_zspec == -1) | (NB_zspec >= 9.9)))[0]

        if filt == 'NB816':
            # NB816 selection
            BV = phot_df['BV']
            Ri = phot_df['Ri']

            if old_selection:
                Ha_sel = np.where((Ri <= 0.45) & (BV >= 2 * Ri - 0.1) & good_phot &
                                  ((NB_zspec == -10) | (NB_zspec == -1) | (NB_zspec >= 9.9)))[0]
            else:
                Ha_sel = np.where((Ri <= 0.45) & (BV >= 2 * Ri) & good_phot &
                                  ((NB_zspec == -10) | (NB_zspec == -1) | (NB_zspec >= 9.9)))[0]

        if filt == 'NB921':
            # NB921 selection
            BR = phot_df['BR']
            Ri = phot_df['Ri']

            Ha_sel = np.where((Ri <= 0.45) & (BR >= 1.46 * Ri + 0.58) & good_phot &
                              ((NB_zspec == -10) | (NB_zspec == -1) | (NB_zspec >= 9.9)))[0]

        if filt == 'NB973':
            # NB973 selection
            BR = phot_df['BR']
            Ri = phot_df['Ri']

            Ha_sel = np.where((Ri >= -0.4) & (Ri <= 0.55) & (BR >= 2.423 * Ri + 0.06386) &
                              (BR >= 0.5) & (BR <= 3.0) & good_phot &
                              ((NB_zspec == -10) | (NB_zspec == -1) | (NB_zspec >= 9.9)))[0]

        print("N(H-alpha) phot ({}) : {}".format(filt, len(Ha_sel)))

        new_Ha_index, new_Ha_rev_Name = \
            handle_bad_sources_dual_emitters(Ha_sel, rev_Name[Ha_sel], filt)
        print("N(H-alpha) phot ({}) : {}".format(filt, len(new_Ha_index)))

        Ha_sel_orig_phot = np.where((NB_zspec[final_Ha_index] == -10) |
                                    (NB_zspec[final_Ha_index] == -1) |
                                    (NB_zspec[final_Ha_index] >= 9.9))[0]
        print("N(H-alpha) original phot ({}) : {} ".format(filt, len(Ha_sel_orig_phot)))
        Ha_sel_orig_phot = final_Ha_index[Ha_sel_orig_phot]

        # Ha_sel_orig = np.array([xx for xx in range(N_NB) if
        #                         ('Ha-'+filt in rev_Name[xx]) and
        #                        (NB_zspec[xx] == -10 or NB_zspec[xx] >= 9.9)])
        # print("N(H-alpha) original ({}) : {}".format(filt, len(Ha_sel_orig)))

        # Identify those that were previously selected as H-alpha photometrically
        # that should not be included, and fix those using set logic
        # non_Ha = set(Ha_sel_orig) - set(Ha_sel)

    # Write new FITS file
    # c_data.NAME = corr_Name
    colorrev2_file = colorrev_file.replace('colorrev', 'colorrev2')
    print("Writing : "+colorrev2_file)
    # fits.writeto(colorrev_file.replace('colorrev', 'colorrev2'), c_data, c_hdr)
