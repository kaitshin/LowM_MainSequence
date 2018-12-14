"""
NB_errors
====

Compute flux and EW errors based on NB and broadband photometry
"""

import sys, os

from chun_codes import systime

from astropy.io import ascii as asc
from astropy.io import fits

import numpy as np

from astropy import log

def main(filter, NB, sig_NB, excess, sig_excess, silent=False, verbose=True):

    '''
    Main function to derive errors from NB photometry

    Parameters
    ----------
    filter : str
      Name of filter: 'NB704', 'NB711', 'NB816', 'NB921', 'NB973'

    NB : array
      NB magnitudes on AB system

    sig_NB : array
      error on [NB]

    excess : array
      BB - NB color on AB system

    sig_excess : array
      error on [excess]

    silent : boolean
      Turns off stdout messages. Default: False

    verbose : boolean
      Turns on additional stdout messages. Default: True

    Returns
    -------
      flux
      sig_flux
      EW
      sig_EW

    Notes
    -----
    Created by Chun Ly, 13 December 2018
    '''

    if silent == False: log.info('### Begin main : '+systime())

    
    if silent == False: log.info('### End main : '+systime())
#enddef

