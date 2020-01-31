"""
get_fit_pdf
===========

Extract specific pages from IDL MPFIT fits results
"""
from __future__ import print_function

import sys, os
import getpass

from chun_codes import systime
from chun_codes import match_nosort_str # + on 23/01/2018
from chun_codes import exec_pdfmerge

from os.path import exists
from astropy.io import ascii as asc
from astropy.io import fits

from astropy.table import Table

import numpy as np

import glob # + on 23/01/2018
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
     - Modified input MMT files
    Created by Chun Ly, 23 January 2018
     - Read in FITS catalog files
     - Use match_nosort_str to crossmatch to get pages
     - Get pages and call exec_pdfmerge to generate PDF
     - Bug fix: Fix issue when FITS catalog is different from number of pages
       in PDF (i.e., some spectra were not fitted by IDL)
    '''
    
    if silent == False: log.info('### Begin main : '+systime())

    if getpass.getuser() == 'cly':
        g_dir0 = '/Users/cly/Google Drive/NASA_Summer2015/'
    else:
        g_dir0 = ''

    s_dir0 = g_dir0 + 'Spectra/'
    if MMT:
        pdf_files = glob.glob(s_dir0+'MMT/Plots/MMT*_line_fit.pdf')
        cat_files = glob.glob(s_dir0+'MMT/Plots/MMT*_line_fit.fits')

        # Handle spectra without PDF fitting plots | + on 23/01/2018
        spec_fit_file = glob.glob(s_dir0+'MMT/Plots/spec_fit*txt')
        spec_fit_file.reverse() # For consistency with pdf_files and cat_files

        # Accidently removed before last commit, cc34fdf
        spec_cov_file = g_dir0+'Composite_Spectra/MMT_spectral_coverage.txt'

    if Keck:
        pdf_files = glob.glob(s_dir0+'Keck/Plots/DEIMOS_*_line_fit.pdf')
        cat_files = glob.glob(s_dir0+'Keck/Plots/DEIMOS_*_line_fit.fits')

        # Handle spectra without PDF fitting plots | + on 23/01/2018
        spec_fit_file = glob.glob(s_dir0+'Keck/Plots/spec_fit*txt')

        spec_cov_file = g_dir0+'Composite_Spectra/Keck_spectral_coverage.txt'

    if silent == False: log.info('## cat_files : ')
    print(cat_files)

    if silent == False: log.info('## pdf_files : ')
    print(pdf_files)

    # Read in FITS catalogs | + on 23/01/2018
    cat_data = []
    for cc in range(len(cat_files)):
        if silent == False: log.info('### Reading : '+cat_files[cc])
        tmp = fits.getdata(cat_files[cc])

        # Fix issue when FITS catalog is different from PDF | + on 23/01/2018
        if silent == False: log.info('### Reading : '+spec_fit_file[cc])
        tmp1 = asc.read(spec_fit_file[cc])
        good = np.where(tmp1['col2'] == 1)[0]

        # Only reduce if FITS catalog contain more entries than PDF pages
        if len(tmp) > len(good): tmp = tmp[good]

        cat_data.append(tmp)
    #endfor

    n_files = len(pdf_files)

    # Read in MMT/Keck spectral coverage file | + on 23/01/2018
    if silent == False: log.info('### Reading : '+spec_cov_file)
    spec_data = asc.read(spec_cov_file)

    # Crossmatch to get samples | + on 23/01/2018
    filt0 = ['NB704', 'NB711', 'NB816', 'NB921', 'NB973']
    for filt in filt0:
        idx     = np.where(spec_data['filter'] == filt)[0]
        spec_ID = spec_data['AP'][idx]

        pages = []
        for cc in range(len(cat_files)):
            m_idxa, m_idxb = match_nosort_str(spec_ID, cat_data[cc].AP)
            str_page = ",".join([str(test+1) for test in m_idxb])

            t_tab = Table([spec_ID[m_idxa],cat_data[cc].AP[m_idxb]])
            t_tab.pprint(max_lines=-1)
            pages.append(str_page)

        outfile = g_dir0+'Plots/Line_Fits/'+filt+'.pdf'
        exec_pdfmerge(pdf_files, pages, outfile, merge=True)

    if silent == False: log.info('### End main : '+systime())
#enddef
