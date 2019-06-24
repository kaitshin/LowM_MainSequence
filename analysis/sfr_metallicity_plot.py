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

Z_arr = np.arange(-2,0.5,0.001)

#Kroupa to Chabrier offset
imf_offset = -np.log10(4.4e-42) - 41.257

def plot_panel(t_ax, z_metal, sfr_convs, ylabel):
    fit0   = np.polyfit(np.log10(z_metal/0.02), sfr_convs, 2)
    print(fit0)
    p_fit0 = np.poly1d(fit0)

    arr0 = p_fit0(Z_arr)

    # Kroupa IMF
    t_ax.scatter(z_metal/0.02, sfr_convs, color='red', marker='o', s=50,
                 edgecolor='none', alpha=0.5, label='Kroupa IMF')

    t_ax.plot(10**Z_arr, nuLnu_arr0, 'r--')

    ann_str0  = r'$y = P_0 + P_1\cdot\log(Z/Z_{\odot}) + P_2\cdot\log(Z/Z_{\odot})^2$' + '\n'
    ann_str0 += r'Kroupa:   $P_0$=%.3f $P_1$=%.3f $P_2$=%.3f' % \
                (fit0[2], fit0[1], fit0[0])
    ann_str0 += '\n'

    sfr_convs1 = sfr_convs + imf_offset
    fit1   = np.polyfit(np.log10(z_metal/0.02), sfr_convs1, 2)
    p_fit1 = np.poly1d(fit1)
    print(fit1)

    # Chabrier IMF

    arr1 = p_fit1(Z_arr)

    t_ax.scatter(z_metal/0.02, sfr_convs1, color='blue', marker='o', s=50,
                 edgecolor='none', alpha=0.5, label='Chabrier IMF')
    t_ax.plot(10**Z_arr, arr1, 'b--')

    t_ax.legend(loc='upper right', fancybox=True, fontsize=12, framealpha=0.5)
    t_ax.set_xlim([1e-2,3])
    t_ax.set_xscale('log')
    t_ax.minorticks_on()
    t_ax.set_xlabel(r'$Z/Z_{\odot}$')
    t_ax.set_ylabel(ylabel)

    ann_str0 += r'Chabrier: $P_0$=%.3f $P_1$=%.3f $P_2$=%.3f' % \
                (fit1[2], fit1[1], fit1[0])

    t_ax.annotate(ann_str0, [0.025,0.025], fontsize=10,
                  xycoords='axes fraction', ha='left', va='bottom')

#enddef

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

    fig, ax = plt.subplots(ncols=2)

    # nuL_nu in ax[0]

    # nu Lnu = lambda Llambda
    nuLnu = Llambda + np.log10(lambda0.value)

    nuLnu_fit0   = np.polyfit(np.log10(Z/0.02), nuLnu, 2)
    print(nuLnu_fit0)
    p_nuLnu_fit0 = np.poly1d(nuLnu_fit0)

    Z_arr = np.arange(-2,0.5,0.001)
    nuLnu_arr0 = p_nuLnu_fit0(Z_arr)
    
    ax[0].scatter(Z/0.02, nuLnu, color='red', marker='o', s=50,
                  edgecolor='none', alpha=0.5, label='Kroupa IMF')

    ax[0].plot(10**Z_arr, nuLnu_arr0, 'r--')

    ann_str0  = r'$y = P_0 + P_1\cdot\log(Z/Z_{\odot}) + P_2\cdot\log(Z/Z_{\odot})^2$' + '\n'
    ann_str0 += r'Kroupa:   $P_0$=%.3f $P_1$=%.3f $P_2$=%.3f' % \
                (nuLnu_fit0[2], nuLnu_fit0[1], nuLnu_fit0[0])
    ann_str0 += '\n'

    nuLnu_fit1   = np.polyfit(np.log10(Z/0.02), nuLnu+imf_offset, 2)
    p_nuLnu_fit1 = np.poly1d(nuLnu_fit1)
    print(nuLnu_fit1)

    nuLnu_arr1 = p_nuLnu_fit1(Z_arr)

    ax[0].scatter(Z/0.02, nuLnu+imf_offset, color='blue', marker='o', s=50,
                  edgecolor='none', alpha=0.5, label='Chabrier IMF')
    ax[0].plot(10**Z_arr, nuLnu_arr1, 'b--')

    ax[0].legend(loc='upper right', fancybox=True, fontsize=12, framealpha=0.5)
    ax[0].set_xlim([1e-2,3])
    ax[0].set_xscale('log')
    ax[0].minorticks_on()
    ax[0].set_xlabel(r'$Z/Z_{\odot}$')
    ax[0].set_ylabel(r'$\nu L_{\nu}(1500\AA)$/SFR [erg s$^{-1}$/$M_{\odot}$ yr$^{-1}$]')

    ann_str0 += r'Chabrier: $P_0$=%.3f $P_1$=%.3f $P_2$=%.3f' % \
                (nuLnu_fit1[2], nuLnu_fit1[1], nuLnu_fit1[0])

    ax[0].annotate(ann_str0, [0.025,0.025], fontsize=10,
                   xycoords='axes fraction', ha='left', va='bottom')

    # L_nu in ax[1]
    nu_offset = np.log10(c0.to(u.m/u.s).value/lambda0.to(u.m).value)
    Lnu = nuLnu - nu_offset

    Lnu_fit0 = nuLnu_fit0.copy()
    Lnu_fit0[2] -= nu_offset
    p_Lnu_fit0 = np.poly1d(Lnu_fit0)

    Lnu_arr0 = p_Lnu_fit0(Z_arr)

    ax[1].scatter(Z/0.02, Lnu, color='red', marker='o', s=50,
                  edgecolor='none', alpha=0.5, label='Kroupa IMF')

    ax[1].plot(10**Z_arr, Lnu_arr0, 'r--')

    ann_str1  = r'$y = P_0 + P_1\cdot\log(Z/Z_{\odot}) + P_2\cdot\log(Z/Z_{\odot})^2$' + '\n'
    ann_str1 += r'Kroupa:   $P_0$=%.3f $P_1$=%.3f $P_2$=%.3f' % \
                (Lnu_fit0[2], Lnu_fit0[1], Lnu_fit0[0])
    ann_str1 += '\n'


    Lnu_fit1 = nuLnu_fit1.copy()
    Lnu_fit1[2] -= nu_offset
    p_Lnu_fit1 = np.poly1d(Lnu_fit1)

    Lnu_arr1 = p_Lnu_fit1(Z_arr)

    ax[1].scatter(Z/0.02, Lnu+imf_offset, color='blue', marker='o', s=50,
                  edgecolor='none', alpha=0.5, label='Kroupa IMF')

    ax[1].plot(10**Z_arr, Lnu_arr1, 'b--')

    ax[1].scatter([1.0], -1*np.log10(1.4e-28), color='green', marker='o',
                  s=50, edgecolor='none', alpha=0.5)
    ax[1].annotate('K98', [1.05,-1*np.log10(1.4e-28*0.98)], xycoords='data',
                   fontsize=8, ha='left', va='bottom')

    ann_str1 += r'Chabrier: $P_0$=%.3f $P_1$=%.3f $P_2$=%.3f' % \
                (Lnu_fit1[2], Lnu_fit1[1], Lnu_fit1[0])

    ax[1].annotate(ann_str1, [0.025,0.025], fontsize=10,
                   xycoords='axes fraction', ha='left', va='bottom')

    ax[1].set_xlim([1e-2,3])
    ax[1].set_xscale('log')
    ax[1].minorticks_on()
    ax[1].set_xlabel(r'$Z/Z_{\odot}$')
    ax[1].set_ylabel(r'$L_{\nu}(1500\AA)$/SFR [erg s$^{-1}$ $\AA^{-1}$/$M_{\odot}$ yr$^{-1}$]')


    plt.subplots_adjust(left=0.085, right=0.995, bottom=0.11, top=0.98, wspace=0.225)
    out_pdf = '/Users/cly/Google Drive/NASA_Summer2015/Plots/sfr_metallicity_plot.pdf'
    fig.set_size_inches(10,5)
    fig.savefig(out_pdf)

    if silent == False: log.info('### End main : '+systime())
#enddef

