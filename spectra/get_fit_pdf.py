"""
get_fit_pdf
===========

Extract specific pages from IDL MPFIT fits results
"""

import sys, os
import getpass

from chun_codes import systime
from chun_codes import exec_pdfmerge

from os.path import exists
from astropy.io import ascii as asc
from astropy.io import fits

import numpy as np

from astropy import log

def main(MMT=False, Keck=False, silent=False, verbose=True):

    '''
    Main function to extract specific pages using exec_pdfmerge

    Parameters
    ----------
    MMT : boolean
      Set to True if extracting MMT plots. Default: False

    Keck : boolean
      Set to True if extracting Keck plots. Default: False

    silent : boolean
      Turns off stdout messages. Default: False

    verbose : boolean
      Turns on additional stdout messages. Default: True

    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 22 January 2018
    '''
    
    if silent == False: log.info('### Begin main : '+systime())

    if getpass.getuser() == 'cly':
        g_dir0 = '/Users/cly/Google Drive/NASA_Summer2015/Spectra/'
    else:
        g_dir0 = ''

    if MMT:
        pdf_files = glob.glob(g_dir0 + 'MMT/Plots/MMT?_all_line_fit.pdf')
        cat_files = glob.glob(g_dir0 + 'MMT/Plots/MMT?_all_line_fit.fits')

    if Keck:
        pdf_files = glob.glob(g_dir0 + 'Keck/Plots/DEIMOS_*_line_fit.pdf')
        cat_files = glob.glob(g_dir0 + 'Keck/Plots/DEIMOS_*_line_fit.fits')

    n_files = len(pdf_files)

    if silent == False: log.info('### End main : '+systime())
#enddef

