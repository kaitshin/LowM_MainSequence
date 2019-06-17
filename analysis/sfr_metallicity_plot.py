"""
sfr_metallicity_plot
====

Compute conversion between nuLnu(1500) to SFR
"""

import sys, os

from chun_codes import systime

from os.path import exists
from astropy.io import ascii as asc
from astropy.io import fits

import numpy as np

import matplotlib.pyplot as plt
import glob

from astropy.table import Table
from astropy import log
import astropy.units as u

def main(silent=False, verbose=True):

    '''
    Main function to read in Starburst99 models and compute UV lum

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
    Created by Chun Ly, 16 June 2019
    '''
    
    if silent == False: log.info('### Begin main : '+systime())

    Z   = np.array([    0.05,     0.02,    0.008,    0.004,   0.0004])
    Llambda = [40.12892, 40.21840, 40.25546, 40.27597, 40.30982]
    '''
    From Starburst99 CSF 1 Msun/yr model with Kroupa IMF and
    Padova stellar tracks.  Log units of erg/s/Ang
    '''

    lambda0 = 1500.0 * u.Angstrom

    # nu Lnu = lambda Llambda
    nuLnu = Llambda + np.log10(lambda0.value)

    fit0   = np.polyfit(np.log10(Z/0.02), nuLnu, 2)
    p_fit0 = np.poly1d(fit0)
    
    print(fit0)
    Z_arr = np.arange(-2,0.5,0.001)
    y_arr = p_fit0(Z_arr)
    
    fig, ax = plt.subplots()

    ax.scatter(Z/0.02, nuLnu, color='red', marker='o', s=50,
               edgecolor='none', alpha=0.5, label='Kroupa IMF')
    ax.plot(10**Z_arr, y_arr, 'r--')

    #Kroupa to Chabrier offset
    imf_offset = -np.log10(4.4e-42) - 41.257
    print(imf_offset)

    fit1   = np.polyfit(np.log10(Z/0.02), nuLnu+imf_offset, 2)
    p_fit1 = np.poly1d(fit1)
    print(fit1)

    y_arr1 = p_fit1(Z_arr)
    
    ax.scatter(Z/0.02, nuLnu+imf_offset, color='blue', marker='o', s=50,
               edgecolor='none', alpha=0.5, label='Chabrier IMF')
    ax.plot(10**Z_arr, y_arr1, 'b--')

    ax.legend(loc='upper right')
    ax.set_xlim([1e-2,3])
    ax.set_xscale('log')
    ax.set_xlabel(r'$Z/Z_{\odot}$')
    ax.set_ylabel(r'$\nu L_{\nu}(1500\AA)$/SFR [erg s$^{-1}$/$M_{\odot}$ yr$^{-1}$]')

    out_pdf = '/Users/cly/Google Drive/NASA_Summer2015/Plots/sfr_metallicity_plot.pdf'
    fig.savefig(out_pdf)

    if silent == False: log.info('### End main : '+systime())
#enddef

