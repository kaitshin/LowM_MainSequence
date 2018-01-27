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
    '''

    if filt == 'NB704':
        z1, z2  = 0.050, 0.100 # H-alpha
        z3, z4  = 0.370, 0.475 # OIII
        z5, z6  = 0.870, 0.910 # OII
        z7, z8  = 4.600, 4.900 # Ly-alpha
        z9, z10 = 0.800, 0.850 # NeIII
    #endif
    if filt == 'NB711':
        z1, z2  = 0.050, 0.100 # H-alpha
        z3, z4  = 0.375, 0.475 # OIII
        z5, z6  = 0.875, 0.940 # OII
        z7, z8  = 4.650, 4.900 # Ly-alpha
        z9, z10 = 0.800, 0.870 # NeIII
    #endif
    if filt == 'NB816':
        z1, z2  = 0.210, 0.260 # H-alpha
        z3, z4  = 0.600, 0.700 # OIII
        z5, z6  = 1.150, 1.225 # OII
        z7, z8  = 5.600, 5.800 # Ly-alpha
        z9, z10 = 1.075, 1.150 # NeIII
    #endif
    if filt == 'NB921':
        z1, z2  = 0.385, 0.420 # H-alpha
        z3, z4  = 0.810, 0.910 # OIII
        z5, z6  = 1.460, 1.480 # OII
        z7, z8  = 6.520, 6.630 # Ly-alpha
        z9, z10 = 0.000, 0.000 # NeIII
    #endif
    if filt == 'NB973':
        z1, z2  = 0.450, 0.520 # H-alpha
        z3, z4  = 0.940, 0.975 # OIII
        z5, z6  = 1.585, 1.620 # OII
        z7, z8  = 6.950, 7.100 # Ly-alpha
        z9, z10 = 0.000, 0.000 # NeIII
    #endif
    return z1, z2, z3, z4, z5, z6, z7, z8, z9, z10
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
    '''
    
    if silent == False: log.info('### Begin main : '+systime())

    dir0 = '/Users/cly/Google Drive/NASA_Summer2015/'

    colorrev_file = dir0+'Catalogs/NB_IA_emitters.nodup.colorrev.fits'

    log.info('### Reading : '+colorrev_file)
    c_data = fits.getdata(colorrev_file)

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

    filt0 = ['NB704','NB711','NB816','NB921','NB973']

    out_pdf = dir0+'Plots/NB_IA_zspec.pdf'
    pp = PdfPages(out_pdf)

    zmax = [1.05, 1.05, 1.3, 1.55, 1.70]
    for ff in range(len(filt0)):
        idx = [xx for xx in range(len(c_data)) if filt0[ff] in c_data.NAME[xx]]
        idx_z = intersect(idx, with_z)

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

        fig.set_size_inches(8,8)
        fig.savefig(pp, format='pdf', bbox_inches='tight')

    #endfor

    pp.close()
    
    if silent == False: log.info('### End main : '+systime())
#enddef

