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

def plot_panel(t_ax, z_metal, sfr_convs, ylabel, showlegend=False, labelx=False):
    fit0   = np.polyfit(np.log10(z_metal/0.02), sfr_convs, 2)
    print(fit0)
    p_fit0 = np.poly1d(fit0)

    arr0 = p_fit0(Z_arr)

    # Kroupa IMF
    t_ax.scatter(z_metal/0.02, sfr_convs, color='red', marker='o', s=50,
                 edgecolor='none', alpha=0.5, label='Kroupa IMF')

    t_ax.plot(10**Z_arr, arr0, 'r--')

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

    if showlegend:
        t_ax.legend(loc='upper right', fancybox=True, fontsize=12, framealpha=0.5)
    t_ax.set_xlim([1e-2,3])
    t_ax.set_xscale('log')
    t_ax.minorticks_on()
    if labelx:
        t_ax.set_xlabel(r'$Z/Z_{\odot}$')
    else:
        t_ax.set_xticklabels([])
    t_ax.set_ylabel(ylabel)

    ann_str0 += r'Chabrier: $P_0$=%.3f $P_1$=%.3f $P_2$=%.3f' % \
                (fit1[2], fit1[1], fit1[0])

    t_ax.annotate(ann_str0, [0.025,0.025], fontsize=10,
                  xycoords='axes fraction', ha='left', va='bottom')

    return fit0, fit1
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
    Llambda = np.array([40.12892, 40.21840, 40.25546, 40.27597, 40.30982])
    '''
    From Starburst99 CSF 1 Msun/yr model with Kroupa IMF and
    Padova stellar tracks.  Log units of erg/s/Ang. Using
    age of 0.3E9 (except 0.1E9 for 2.5xZ_solar)
    '''

    lambda0 = 1500.0 * u.Angstrom

    fig, ax = plt.subplots(ncols=2, nrows=2)

    # nuL_nu in ax[0][0]

    # nu Lnu = lambda Llambda
    nuLnu = Llambda + np.log10(lambda0.value)

    ylabel = r'$\nu L_{\nu}(1500\AA)$/SFR [erg s$^{-1}$/$M_{\odot}$ yr$^{-1}$]'
    nuLnu_fit_kr, nuLnu_fit_ch = plot_panel(ax[0][0], Z, nuLnu, ylabel, showlegend=True)

    # L_nu in ax[0][1]
    nu_offset = np.log10(c0.to(u.m/u.s).value/lambda0.to(u.m).value)
    Lnu = nuLnu - nu_offset

    ylabel = r'$L_{\nu}(1500\AA)$/SFR [erg s$^{-1}$ Hz$^{-1}$/$M_{\odot}$ yr$^{-1}$]'
    Lnu_fit_kr, Lnu_fit_ch = plot_panel(ax[0][1], Z, Lnu, ylabel)

    # Plot K98 relation
    ax[0][1].scatter([1.0], -1*np.log10(1.4e-28), color='green', marker='o',
                  s=50, edgecolor='none', alpha=0.5)
    ax[0][1].annotate('K98', [1.05,-1*np.log10(1.4e-28*0.98)], xycoords='data',
                   fontsize=8, ha='left', va='bottom')


    # Plot H-alpha in ax[1][0]
    LHa = np.array([41.061, 41.257, 41.381, 41.439, 41.536])

    ylabel = r'$L({\rm H}\alpha)$/SFR [erg s$^{-1}$/$M_{\odot}$ yr$^{-1}$]'
    LHa_fit_kr, LHa_fit_ch = plot_panel(ax[1][0], Z, LHa, ylabel, labelx=True)


    # Plot nuLnu vs LHa in ax[1][1]
    ylabel = r'$\nu L_{\nu}(1500\AA)/L({\rm H}\alpha)$'
    nuLnu_LHa_fit_kr, nuLnu_LHa_fit_ch = plot_panel(ax[1][1], Z, nuLnu-LHa, ylabel,
                                                    labelx=True)


    plt.subplots_adjust(left=0.085, right=0.995, bottom=0.07, top=0.98, wspace=0.225,
                        hspace=0.04)
    out_pdf = '/Users/cly/Google Drive/NASA_Summer2015/Plots/sfr_metallicity_plot.pdf'
    fig.set_size_inches(10,8)
    fig.savefig(out_pdf)

    out_npzfile = out_pdf.replace('.pdf', '_fit.npz')
    np.savez(out_npzfile, nuLnu_fit_kr=nuLnu_fit_kr, nuLnu_fit_ch=nuLnu_fit_ch,
             Lnu_fit_kr=Lnu_fit_kr, Lnu_fit_ch=Lnu_fit_ch,
             LHa_fit_kr=LHa_fit_kr, LHa_fit_ch=LHa_fit_ch,
             nuLnu_LHa_fit_kr=nuLnu_LHa_fit_kr,
             nuLnu_LHa_fit_ch=nuLnu_LHa_fit_ch)

    if silent == False: log.info('### End main : '+systime())
#enddef
