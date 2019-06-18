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
from astropy.constants import c as c0

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

    Z       = [    0.05,     0.02,    0.008,    0.004,   0.0004]
    Z       = np.array(Z)
    Llambda = [40.12892, 40.21840, 40.25546, 40.27597, 40.30982]
    '''
    From Starburst99 CSF 1 Msun/yr model with Kroupa IMF and
    Padova stellar tracks.  Log units of erg/s/Ang. Using
    age of 0.3E9 (except 0.1E9 for 2.5xZ_solar)
    '''

    lambda0 = 1500.0 * u.Angstrom

    #Kroupa to Chabrier offset
    imf_offset = -np.log10(4.4e-42) - 41.257
    print(imf_offset)

    fig, ax = plt.subplots(ncols=2)

    # nuL_nu in ax[0]

    # nu Lnu = lambda Llambda
    nuLnu = Llambda + np.log10(lambda0.value)

    fit0   = np.polyfit(np.log10(Z/0.02), nuLnu, 2)
    print(fit0)
    p_fit0 = np.poly1d(fit0)

    Z_arr = np.arange(-2,0.5,0.001)
    y_arr = p_fit0(Z_arr)
    
    ax[0].scatter(Z/0.02, nuLnu, color='red', marker='o', s=50,
                  edgecolor='none', alpha=0.5, label='Kroupa IMF')

    ax[0].plot(10**Z_arr, y_arr, 'r--')

    ann_str0  = r'$y = P_0 + P_1\cdot\log(Z/Z_{\odot}) + P_2\cdot\log(Z/Z_{\odot})^2$' + '\n'
    ann_str0 += r'Kroupa:   $P_0$=%.3f $P_1$=%.3f $P_2$=%.3f' % (fit0[2], fit0[1], fit0[0])
    ann_str0 += '\n'

    fit1   = np.polyfit(np.log10(Z/0.02), nuLnu+imf_offset, 2)
    p_fit1 = np.poly1d(fit1)
    print(fit1)

    y_arr1 = p_fit1(Z_arr)
    
    ax[0].scatter(Z/0.02, nuLnu+imf_offset, color='blue', marker='o', s=50,
                  edgecolor='none', alpha=0.5, label='Chabrier IMF')
    ax[0].plot(10**Z_arr, y_arr1, 'b--')

    ax[0].legend(loc='upper right', fancybox=True, fontsize=12, framealpha=0.5)
    ax[0].set_xlim([1e-2,3])
    ax[0].set_xscale('log')
    ax[0].minorticks_on()
    ax[0].set_xlabel(r'$Z/Z_{\odot}$')
    ax[0].set_ylabel(r'$\nu L_{\nu}(1500\AA)$/SFR [erg s$^{-1}$/$M_{\odot}$ yr$^{-1}$]')

    ann_str0 += r'Chabrier: $P_0$=%.3f $P_1$=%.3f $P_2$=%.3f' % (fit1[2], fit1[1], fit1[0])

    ax[0].annotate(ann_str0, [0.025,0.025], fontsize=10, xycoords='axes fraction',
                   ha='left', va='bottom')

    plt.subplots_adjust(left=0.105, right=0.99, bottom=0.07, top=0.99)
    out_pdf = '/Users/cly/Google Drive/NASA_Summer2015/Plots/sfr_metallicity_plot.pdf'
    fig.set_size_inches(8,4)
    fig.savefig(out_pdf)

    if silent == False: log.info('### End main : '+systime())
#enddef

