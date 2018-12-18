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

m_AB = 48.6
def fluxline(y, fNB, dNB, dBB):
  '''RETURNS emission-line flux in erg/s/cm^2'''

  return dNB*fNB*(1.0 - y)/(1.0-dNB/dBB)
#enddef

def mag_combine(m1, m2, epsilon):
  cont_flux = epsilon * 10**(-0.4*(m1+m_AB)) + (1-epsilon)*10**(-0.4*(m2+m_AB))
  cont_mag  = -2.5*np.log10(cont_flux) - m_AB
  return cont_mag
#enddef

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

    filt_ref= ['NB704', 'NB711', 'NB816', 'IA598', 'IA679','NB921','NB973']
    dNB     = [  100.0,    72.0,   120.0,   303.0,   340.0,  132.0,  200.0]
    lambdac = [ 7046.0,  7111.0,  8150.0,  6007.0,  6780.0, 9196.0, 9755.0]
    dBB     = [ 1110.0,  1110.0,  1419.0,   885.0,  1110.0,  956.0,  956.0] # R R i, V, R

    
    if silent == False: log.info('### End main : '+systime())
#enddef

