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

    for filt in filt0:
        idx = [xx for xx in range(len(c_data)) if filt in c_data.NAME[xx]]
        idx_z = intersect(idx, with_z)

        fig, ax = plt.subplots()
        ax.hist(z_spec0[idx_z], bins=500, alpha=0.5)
        ax.set_xlabel('Spectroscopic Redshift')
        ax.set_ylabel('Number of Spectra')
        ax.annotate(filt, [0.05,0.95], xycoords='axes fraction', ha='left',
                    va='top')
        fig.set_size_inches(8,8)
        fig.savefig(pp, format='pdf', bbox_inches='tight')

    #endfor

    pp.close()
    
    if silent == False: log.info('### End main : '+systime())
#enddef

