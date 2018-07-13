"""
filter_stat_corr
================

Provide description for code here.
"""

import sys, os

from chun_codes import systime

from astropy.io import ascii as asc

import numpy as np

from astropy.table import Table
from astropy import log

def main(silent=False, verbose=True):

    '''
    Provide explanation for function here.

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
    Created by Chun Ly, 13 July 2018
    '''
    
    if silent == False: log.info('### Begin main : '+systime())

    path0 = '/Users/cly/Google Drive/NASA_Summer2015/Filters/'

    filters = ['NB704','NB711','NB816','NB921','NB973']
    files = [path0 + filt + 'response.dat' for filt in filters]

    filt_corr = np.zeros(len(files))
    for ff in range(len(files)):
        log.info('Reading : '+files[ff])
        tab1   = asc.read(files[ff], format='no_header')
        y_val  = tab1['col2']
        y_val /= max(y_val)

        good = np.where(y_val >= 0.05)[0]

        # Compute statistical correction by Sum [filter_amp * filter_amp] / Sum [filter_amp]
        weight_sum = np.sum(y_val[good]**2)/np.sum(y_val[good])
        filt_corr[ff] = 1/weight_sum

    filt_tab = Table([filters, filt_corr], names=('Filter','Filt_Stat_Corr'))
    filt_tab.pprint()
        
    if silent == False: log.info('### End main : '+systime())
#enddef

