"""
completeness_analysis
====

A set of Python 2.7 codes for completeness analysis of NB-selected galaxies
in the M*-SFR plots
"""

import sys, os

from chun_codes import systime

from os.path import exists

from astropy.io import fits
from astropy.io import ascii as asc

import numpy as np

import matplotlib.pyplot as plt

from astropy import log

def mag_vs_mass(silent=False, verbose=True):

    '''
    Compares optical photometry against stellar masses to get relationship

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
    Created by Chun Ly, 1 May 2019
    '''

    if silent == False: log.info('### Begin mag_vs_mass : '+systime())

    path0 = '/Users/cly/Google Drive/NASA_Summer2015/'

    # NB Ha emitter sample for ID
    NB_file = path0 + 'Main_Sequence/mainseq_corrections_tbl (1).txt'
    log.info("Reading : "+NB_file)
    NB_tab     = asc.read(NB_file)
    NB_HA_Name = NB_tab['NAME0'].data
    NB_Ha_ID   = NB_tab['ID'].data - 1 # Relative to 0 --> indexing

    # Read in stellar mass results table
    FAST_file = path0 + \
                'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX.fout'
    log.info("Reading : "+FAST_file)
    FAST_tab = asc.read(FAST_file)

    logM = FAST_tab['col7'].data

    logM_NB_Ha = logM[NB_Ha_ID]

    NB_catfile = path0 + 'Catalogs/NB_IA_emitters.allcols.colorrev.fix.errors.fits'
    log.info("Reading : "+NB_catfile)
    NB_catdata = fits.getdata(NB_catfile)
    NB_catdata = NB_catdata[NB_Ha_ID]

    cont_mag = np.zeros(len(NB_catdata))

    fig, ax = plt.subplots(ncols=2, nrows=2)

    filts = ['NB704','NB711','NB816','NB921','NB973']

    for filt in filts:
        log.info('### Working on : '+filt)
        NB_idx = [ii for ii in range(len(NB_tab)) if 'Ha-'+filt in NB_HA_Name[ii]]
        print(" Size : ", len(NB_idx))
        #print(NB_catdata[filt+'_CONT_MAG'])[NB_idx]
        cont_mag[NB_idx] = NB_catdata[filt+'_CONT_MAG'][NB_idx]


    NB704711_idx = [ii for ii in range(len(NB_tab)) if 'Ha-NB7' in NB_HA_Name[ii]]
    ax[0][0].scatter(cont_mag[NB704711_idx], logM_NB_Ha[NB704711_idx])

    NB816_idx = [ii for ii in range(len(NB_tab)) if 'Ha-NB816' in NB_HA_Name[ii]]
    ax[0][1].scatter(cont_mag[NB816_idx], logM_NB_Ha[NB816_idx])

    NB921_idx = [ii for ii in range(len(NB_tab)) if 'Ha-NB921' in NB_HA_Name[ii]]
    ax[1][0].scatter(cont_mag[NB921_idx], logM_NB_Ha[NB921_idx])

    NB973_idx = [ii for ii in range(len(NB_tab)) if 'Ha-NB973' in NB_HA_Name[ii]]
    ax[1][1].scatter(cont_mag[NB973_idx], logM_NB_Ha[NB973_idx])
    
    if silent == False: log.info('### End mag_vs_mass : '+systime())
#enddef

