"""
fix_colorrev_file
====

This code will modify the colorrev.fits file to use updated spectroscopic
redshifts from MMT/Hectospec and Keck/DEIMOS.  Some SDF NB excess emitters
were mis-classified using color information
"""

import sys, os

from chun_codes import systime, intersect

from os.path import exists
from astropy.io import ascii as asc
from astropy.io import fits

import numpy as np

import glob

from astropy.table import Table
from astropy import log

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def NB_spec_redshift(filt):
    '''
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
    '''

    if filt == 'NB704':
        z1, z2  = 0.050, 0.100 # H-alpha
        z3, z4  = 0.370, 0.475 # OIII
        z5, z6  = 0.870, 0.910 # OII
        z7, z8  = 4.600, 4.900 # Ly-alpha
        z9, z10 = 0.800, 0.850 # NeIII
        z11,z12 = 1.460, 1.560 # MgII | + on 31/01/2018
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII', 'MgII']
    #endif
    if filt == 'NB711':
        z1, z2  = 0.050, 0.100 # H-alpha
        z3, z4  = 0.375, 0.475 # OIII
        z5, z6  = 0.875, 0.940 # OII
        z7, z8  = 4.650, 4.900 # Ly-alpha
        z9, z10 = 0.800, 0.870 # NeIII
        z11,z12 = 1.460, 1.560 # MgII | + on 31/01/2018
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII', 'MgII']
    #endif
    if filt == 'NB816':
        z1, z2  = 0.210, 0.260 # H-alpha
        z3, z4  = 0.600, 0.700 # OIII
        z5, z6  = 1.150, 1.225 # OII
        z7, z8  = 5.600, 5.800 # Ly-alpha
        z9, z10 = 1.075, 1.150 # NeIII
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII']
    #endif
    if filt == 'NB921':
        z1, z2  = 0.385, 0.429 # H-alpha
        z3, z4  = 0.810, 0.910 # OIII
        z5, z6  = 1.445, 1.492 # OII
        z7, z8  = 6.520, 6.630 # Ly-alpha
        z9, z10 = 0.000, 0.000 # NeIII
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII']
    #endif
    if filt == 'NB973':
        z1, z2  = 0.450, 0.520 # H-alpha
        z3, z4  = 0.940, 0.975 # OIII
        z5, z6  = 1.585, 1.620 # OII
        z7, z8  = 6.950, 7.100 # Ly-alpha
        z9, z10 = 0.000, 0.000 # NeIII
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII']
    #endif

    # + on 29/01/2018
    if filt == 'IA598':
        z1, z2  = 0.000, 0.000 # Don't really use this
        z3, z4  = 0.150, 0.300 # [OIII]/H-beta
        z5, z6  = 0.550, 0.650 # [OII]
        z7, z8  = 3.600, 4.100 # Ly-alpha
        z9, z10 = 0.000, 0.000 # NeIII
        #Also, we have 1 CIII] 1909. No [NeIII]
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII']
    #endif
    if filt == 'IA679':
        z1, z2  = 0.000, 0.080 # Ha. Don't really use this
        z3, z4  = 0.300, 0.450 # [OIII]/H-beta
        z5, z6  = 0.750, 0.950 # [OII]
        z7, z8  = 4.200, 4.900 # Ly-alpha
        z9, z10 = 0.000, 0.000 # NeIII
        z11,z12 = 1.500, 1.510 # MgII
        #Also, we have 1 CIII] 1909 and 1 or 2 CIV 1549. No [NeIII]
        z_vals = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12)
        ltype = ['Ha', 'OIII', 'OII',  'Lya', 'NeIII', 'MgII']
    #endif

    return z_vals, ltype
#enddef

def main(silent=False, verbose=True):

    '''
    Main function for fix_colorrev_file.py

    Parameters
    ----------

    silent : boolean
      Turns off stdout messages. Default: False

    verbose : boolean
      Turns on additional stdout messages. Default: True

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
    '''
    
    if silent == False: log.info('### Begin main : '+systime())

    dir0 = '/Users/cly/Google Drive/NASA_Summer2015/'

    # Mod on 29/01/2018
    orig_file     = dir0+'Catalogs/NB_IA_emitters.nodup.fits'
    colorrev_file = orig_file.replace('.fits','.colorrev.fits')

    # + on 29/01/2018
    log.info('### Reading : '+orig_file)
    raw_data = fits.getdata(orig_file)

    log.info('### Reading : '+colorrev_file)
    c_data, c_hdr = fits.getdata(colorrev_file, header=True) # Mod on 01/02/2018

    # + on 29/01/2018
    raw_Name = np.array([str0.replace(' ','') for str0 in raw_data.NAME],
                        dtype='|S67')
    rev_Name = np.array([str0.replace(' ','') for str0 in c_data.NAME])

    corr_Name = raw_Name.copy() # + on 29/01/2018

    zspec_file = dir0+'Catalogs/nb_ia_zspec.txt'
    log.info('### Reading : '+zspec_file)
    z_data = asc.read(zspec_file)

    z_spec0 = z_data['zspec0']

    # Note: This should give 1989. Which matches the number of spectra
    # in spec_match/1014/NB_IA_emitters.spec.fits (current 'current')
    in_z_cat = np.where((z_spec0 != -10))[0]

    # Note: This yields 1519 galaxies
    with_z = np.where((z_spec0 != -10) & (z_spec0 < 9.999) &
                      (z_spec0 != -1.0))[0]

    # + on 30/01/2018
    without_z = np.where((z_spec0 == -10) | (z_spec0 >= 9.999) |
                         (z_spec0 == -1.0))[0]

    corr_Name[without_z] = rev_Name[without_z]

    filt0 = ['NB704','NB711','NB816','NB921','NB973','IA598','IA679']

    out_pdf = dir0+'Plots/NB_IA_zspec.pdf'
    pp = PdfPages(out_pdf)

    zmax = [1.05, 1.05, 1.3, 1.55, 1.70, 0.70, 0.96]
    for ff in range(len(filt0)):
        idx = [xx for xx in range(len(c_data)) if filt0[ff] in c_data.NAME[xx]]
        idx_z = intersect(idx, with_z)

        z_vals, ltype = NB_spec_redshift(filt0[ff])

        fig, ax = plt.subplots()
        N, bins, patch = ax.hist(z_spec0[idx_z], bins=500, alpha=0.5,
                                 edgecolor='none', histtype='bar',
                                 align='mid')
        ax.set_xlabel('Spectroscopic Redshift')
        ax.set_ylabel('Number of Spectra')
        ax.minorticks_on()
        ax.annotate(filt0[ff], [0.025,0.975], xycoords='axes fraction',
                    ha='left', va='top')

        # zoom-in inset panel | + on 12/12/2016
        axins = inset_axes(ax, width=4., height=4., loc=1)
        axins.hist(z_spec0[idx_z], bins=500, alpha=0.5, edgecolor='none',
                   histtype='bar', align='mid')
        axins.set_xlim([0.0,zmax[ff]])
        axins.set_xlabel(r'$z_{\rm spec}$')
        # axins.set_ylim([0.0,max(N)])
        axins.minorticks_on()

        # Draw vertical lines for selection | + on 29/01/2018
        ctype = ['red','green','blue','black','purple', 'magenta']
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
                print '%s %.2f %.2f %03i' % ((ltype[zz]+'-'+filt0[ff]).rjust(11), z_vals[2*zz],
                                             z_vals[2*zz+1], len(z_line))

                # + on 30/01/2018
                if len(z_line) > 0:
                    z_temp = np.array(idx_z)[z_line]
                    new_Name = [str0.replace(filt0[ff],ltype[zz]+'-'+filt0[ff]) for
                                str0 in corr_Name[z_temp]]
                    corr_Name[z_temp] = new_Name
            #endif
        #endfor

        fig.set_size_inches(8,8)
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
            tab_temp.write(outfile2, format='ascii.fixed_width_two_line', overwrite=True)
    #endfor

    pp.close()

    # Write changes to source name to file | + on 30/01/2018
    change = [xx for xx in range(len(raw_Name)) if corr_Name[xx] != rev_Name[xx]]
    print '## len of [change] : ', len(change)

    # Mod on 30/01/2018
    z_data_ch = z_data[change]
    arr0 = zip(z_data_ch['ID0'], z_data_ch['zspec0'], z_data_ch['slit_str0'],
               rev_Name[change], corr_Name[change])
    change_str0 = [str(a)+' '+str(b)+' '+c+' '+d+' -> '+e+'\n' for a,b,c,d,e in arr0]

    outfile = dir0+'Catalogs/fix_colorrev_file.dat'
    log.info('## Writing : '+outfile)
    f0 = open(outfile, 'w')
    f0.writelines(change_str0)
    f0.close()

    # + on 01/02/2018
    c_data.NAME = corr_Name
    fits.writeto(colorrev_file.replace('.fits','.fix.fits'), c_data, c_hdr)
    if silent == False: log.info('### End main : '+systime())
#enddef

