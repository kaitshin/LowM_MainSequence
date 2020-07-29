"""
completeness_analysis_extras
====

A set of Python 2.7 codes for completeness analysis of NB-selected galaxies in
the M*-SFR plot.  These functions were for generating ancillary products.
They are not executed within completeness_analysis.ew_MC(), but are needed
in advance
"""

from os.path import exists
from chun_codes import systime

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii as asc
from astropy.io import fits

from .config import m_NB

from .config import path0, filters
from . import MLog


def mag_vs_mass(silent=False):  # verbose=True):
    """
    Compares optical photometry against stellar masses to get relationship

    Parameters
    ----------

    silent : boolean
      Turns off stdout messages. Default: False

    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 1 May 2019
    """

    log = MLog(path0 + 'Completeness/', '', prefix='mag_vs_mass')._get_logger()

    if not silent:
        log.info('### Begin mag_vs_mass : '+systime())

    # NB Ha emitter sample for ID
    NB_file = path0 + 'Main_Sequence/mainseq_corrections_tbl (1).txt'
    if not exists(NB_file):
        NB_file = NB_file.replace(' (1)', '')

    log.info("Reading : "+NB_file)
    NB_tab     = asc.read(NB_file)
    NB_HA_Name = NB_tab['NAME0'].data
    NB_Ha_ID   = NB_tab['ID'].data - 1  # Relative to 0 --> indexing

    # Read in stellar mass results table
    FAST_file = path0 + \
        'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX.fout'
    log.info("Reading : " + FAST_file)
    FAST_tab = asc.read(FAST_file)

    logM = FAST_tab['col7'].data

    logM_NB_Ha = logM[NB_Ha_ID]

    NB_catfile = path0 + 'Catalogs/NB_IA_emitters.allcols.colorrev.fix.errors.fits'
    log.info("Reading : "+NB_catfile)
    NB_catdata = fits.getdata(NB_catfile)
    NB_catdata = NB_catdata[NB_Ha_ID]

    cont_mag = np.zeros(len(NB_catdata))

    fig, ax = plt.subplots(ncols=2, nrows=2)

    for filt in filters:
        log.info('### Working on : '+filt)
        NB_idx = [ii for ii in range(len(NB_tab)) if 'Ha-'+filt in
                  NB_HA_Name[ii]]
        print(" Size : ", len(NB_idx))
        cont_mag[NB_idx] = NB_catdata[filt+'_CONT_MAG'][NB_idx]

    for rr in range(2):
        for cc in range(2):
            if cc == 0:
                ax[rr][cc].set_ylabel(r'$\log(M/M_{\odot})$')
            if cc == 1:
                ax[rr][cc].set_yticklabels([])
            ax[rr][cc].set_xlim(19.5, 28.5)
            ax[rr][cc].set_ylim(4.0, 11.0)

    prefixes = ['Ha-NB7', 'Ha-NB816', 'Ha-NB921', 'Ha-NB973']
    xlabels  = [r"$R_Ci$'", r"$i$'$z$'", "$z$'", "$z$'"]
    annot    = ['NB704,NB711', 'NB816', 'NB921', 'NB973']

    dmag = 0.4

    for ff in range(len(prefixes)):
        col = ff % 2
        row = ff / 2
        NB_idx = np.array([ii for ii in range(len(NB_tab)) if prefixes[ff] in
                           NB_HA_Name[ii]])
        t_ax = ax[row][col]
        t_ax.scatter(cont_mag[NB_idx], logM_NB_Ha[NB_idx], edgecolor='blue',
                     color='none', alpha=0.5)
        t_ax.set_xlabel(xlabels[ff])
        t_ax.annotate(annot[ff], [0.975, 0.975], xycoords='axes fraction',
                      ha='right', va='top')

        x_min    = np.min(cont_mag[NB_idx])
        x_max    = np.max(cont_mag[NB_idx])
        cont_arr = np.arange(x_min, x_max+dmag, dmag)
        avg_logM = np.zeros(len(cont_arr))
        std_logM = np.zeros(len(cont_arr))
        N_logM   = np.zeros(len(cont_arr))
        for cc in range(len(cont_arr)):
            cc_idx = np.where((cont_mag[NB_idx] >= cont_arr[cc]) &
                              (cont_mag[NB_idx] < cont_arr[cc]+dmag))[0]
            if len(cc_idx) > 0:
                avg_logM[cc] = np.average(logM_NB_Ha[NB_idx[cc_idx]])
                std_logM[cc] = np.std(logM_NB_Ha[NB_idx[cc_idx]])
                N_logM[cc]   = len(cc_idx)

        t_ax.scatter(cont_arr+dmag/2, avg_logM, marker='o', color='black',
                     edgecolor='none')
        t_ax.errorbar(cont_arr+dmag/2, avg_logM, yerr=std_logM, capsize=0,
                      linestyle='none', color='black')

        out_npz = path0 + 'Completeness/mag_vs_mass_'+prefixes[ff]+'.npz'
        log.info("Writing : "+out_npz)
        np.savez(out_npz, x_min=x_min, x_max=x_max, cont_arr=cont_arr,
                 avg_logM=avg_logM, std_logM=std_logM, N_logM=N_logM)

    plt.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.97,
                        wspace=0.01)

    out_pdf = path0 + 'Completeness/mag_vs_mass.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    if not silent:
        log.info('### End mag_vs_mass : '+systime())


def get_EW_Flux_distribution():
    '''
    Retrieve NB excess emission-line EW and fluxes from existing tables
    '''

    log = MLog(path0 + 'Completeness/', '', prefix='get_EW_Flux_distribution')._get_logger()

    # NB Ha emitter sample for ID
    NB_file = path0 + 'Main_Sequence/mainseq_corrections_tbl.txt'
    log.info("Reading : "+NB_file)
    NB_tab      = asc.read(NB_file)
    NB_HA_Name  = NB_tab['NAME0'].data
    NB_Ha_ID    = NB_tab['ID'].data - 1 # Relative to 0 --> indexing
    NII_Ha_corr = NB_tab['nii_ha_corr_factor'].data # This is log(1+NII/Ha)
    filt_corr   = NB_tab['filt_corr_factor'].data # This is log(f_filt)
    zspec0      = NB_tab['zspec0'].data

    NB_catfile = path0 + 'Catalogs/NB_IA_emitters.allcols.colorrev.fix.errors.fits'
    log.info("Reading : "+NB_catfile)
    NB_catdata = fits.getdata(NB_catfile)
    NB_catdata = NB_catdata[NB_Ha_ID]

    # These are the raw measurements
    NB_EW   = np.zeros(len(NB_catdata))
    NB_Flux = np.zeros(len(NB_catdata))

    Ha_EW   = np.zeros(len(NB_catdata))
    Ha_Flux = np.zeros(len(NB_catdata))

    logMstar = np.zeros(len(NB_catdata))
    Ha_SFR   = np.zeros(len(NB_catdata))
    Ha_Lum   = np.zeros(len(NB_catdata))

    NBmag   = np.zeros(len(NB_catdata))
    contmag = np.zeros(len(NB_catdata))

    spec_flag = np.zeros(len(NB_catdata))

    for filt in filters:
        log.info('### Working on : '+filt)
        NB_idx = np.array([ii for ii in range(len(NB_tab)) if 'Ha-'+filt in \
                           NB_HA_Name[ii]])
        print(" Size : ", len(NB_idx))
        NB_EW[NB_idx]   = np.log10(NB_catdata[filt+'_EW'][NB_idx])
        NB_Flux[NB_idx] = NB_catdata[filt+'_FLUX'][NB_idx]

        Ha_EW[NB_idx]   = (NB_EW   + NII_Ha_corr + filt_corr)[NB_idx]
        Ha_Flux[NB_idx] = (NB_Flux + NII_Ha_corr + filt_corr)[NB_idx]

        NBmag[NB_idx]   = NB_catdata[filt+'_MAG'][NB_idx]
        contmag[NB_idx] = NB_catdata[filt+'_CONT_MAG'][NB_idx]

        logMstar[NB_idx] = NB_tab['stlr_mass'][NB_idx]
        Ha_SFR[NB_idx]   = NB_tab['met_dep_sfr'][NB_idx]
        Ha_Lum[NB_idx]   = NB_tab['obs_lumin'][NB_idx] + \
                           (NII_Ha_corr + filt_corr)[NB_idx]

        with_spec = np.where((zspec0[NB_idx] > 0) & (zspec0[NB_idx] < 9))[0]
        with_spec = NB_idx[with_spec]
        spec_flag[with_spec] = 1

        out_npz = path0 + 'Completeness/ew_flux_Ha-'+filt+'.npz'
        log.info("Writing : "+out_npz)
        np.savez(out_npz, NB_ID=NB_catdata['ID'][NB_idx], NB_EW=NB_EW[NB_idx],
                 NB_Flux=NB_Flux[NB_idx], Ha_EW=Ha_EW[NB_idx],
                 Ha_Flux=Ha_Flux[NB_idx], NBmag=NBmag[NB_idx],
                 contmag=contmag[NB_idx], spec_flag=spec_flag[NB_idx],
                 logMstar=logMstar[NB_idx], Ha_SFR=Ha_SFR[NB_idx],
                 Ha_Lum=Ha_Lum[NB_idx])
    #endfor

#enddef


def NB_numbers():
    '''
    Uses SExtractor catalog to look at number of NB galaxies vs magnitude
    '''

    NB_path = '/Users/cly/data/SDF/NBcat/'
    NB_phot_files = [NB_path+filt+'/sdf_pub2_'+filt+'.cat.mask' for \
                     filt in filters]

    out_pdf = path0 + 'Completeness/NB_numbers.pdf'
    fig, ax_arr = plt.subplots(nrows=3, ncols=2)

    fig0, ax0 = plt.subplots()
    out_pdf0 = path0 + 'Completeness/NB_numbers_all.pdf'

    bin_size = 0.25
    bins = np.arange(17.0,28,bin_size)

    NB_slope0 = np.zeros(len(filters))

    N_norm0 = []
    mag_arr = []

    ctype = ['blue','green','black','red','magenta']
    for ff in range(len(filters)):
        print('Reading : '+NB_phot_files[ff])
        phot_tab = asc.read(NB_phot_files[ff])
        MAG_APER = phot_tab['col13'].data

        row = int(ff / 2)
        col = ff % 2

        ax = ax_arr[row][col]

        ax0.hist(MAG_APER, bins=bins, align='mid', color=ctype[ff],
                 linestyle='solid', histtype='step', label=filters[ff])

        N, m_bins, _ = ax.hist(MAG_APER, bins=bins, align='mid', color='black',
                               linestyle='solid', histtype='step',
                               label='N = '+str(len(MAG_APER)))

        det0  = np.where((m_bins >= 18.0) & (m_bins <= m_NB[ff]))[0]
        p_fit = np.polyfit(m_bins[det0], np.log10(N[det0]), 1)

        det1    = np.where((m_bins >= 20.0) & (m_bins <= m_NB[ff]))[0]
        mag_arr += [m_bins[det1]]
        N_norm0 += [N[det1] / bin_size / np.sum(N[det1])]

        NB_slope0[ff] = p_fit[0]

        fit   = np.poly1d(p_fit)
        fit_lab = 'P[0] = %.3f  P[1] = %.3f' % (p_fit[1], p_fit[0])
        ax.plot(m_bins, 10**(fit(m_bins)), 'r--', label=fit_lab)

        ax.axvline(m_NB[ff], linestyle='dashed', linewidth=1.5)
        ax.legend(loc='lower right', fancybox=True, fontsize=8, framealpha=0.75)
        ax.annotate(filters[ff], [0.025,0.975], xycoords='axes fraction',
                    ha='left', va='top', fontsize=10)
        ax.set_yscale('log')
        ax.set_ylim([1,5e4])
        ax.set_xlim([16.5,28.0])

        if col == 0:
            ax.set_ylabel(r'$N$')

        if col == 1:
            ax.yaxis.set_ticklabels([])

        if row == 0 or (row == 1 and col == 0):
            ax.xaxis.set_ticklabels([])
        else:
            ax.set_xlabel('NB [mag]')
    #endfor

    ax_arr[2][1].axis('off')

    plt.subplots_adjust(left=0.105, right=0.98, bottom=0.05, top=0.98,
                        wspace=0.025, hspace=0.025)
    fig.savefig(out_pdf, bbox_inches='tight')

    ax0.legend(loc='upper left', fancybox=True, fontsize=8, framealpha=0.75)
    ax0.set_yscale('log')
    ax0.set_ylim([1,5e4])
    ax0.set_xlim([16.5,28.0])
    ax0.set_xlabel('NB [mag]')
    ax0.set_ylabel(r'$N$')

    plt.subplots_adjust(left=0.105, right=0.98, bottom=0.05, top=0.98,
                        wspace=0.025, hspace=0.025)
    fig0.savefig(out_pdf0, bbox_inches='tight')

    out_npz = out_pdf.replace('.pdf', '.npz')
    np.savez(out_npz, filters=filters, bin_size=bin_size, NB_slope0=NB_slope0,
             mag_arr=mag_arr, N_norm0=N_norm0)

#enddef
