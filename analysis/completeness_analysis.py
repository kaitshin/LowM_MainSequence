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
from matplotlib.backends.backend_pdf import PdfPages

from scipy.interpolate import interp1d

from NB_errors import ew_flux_dual, fluxline, mag_combine

from NB_errors import filt_ref, dNB, lambdac, dBB, epsilon

NB_filt = np.array([xx for xx in range(len(filt_ref)) if 'NB' in filt_ref[xx]])
for arr in ['filt_ref','dNB','lambdac','dBB','epsilon']:
    cmd1 = arr + ' = np.array('+arr+')'
    exec(cmd1)
    cmd2 = arr + ' = '+arr+'[NB_filt]'
    exec(cmd2)

#Limiting magnitudes for NB data (only)
m_NB  = np.array([26.7134-0.047, 26.0684, 26.9016+0.057, 26.7088-0.109, 25.6917-0.051])
m_BB1 = np.array([28.0829, 28.0829, 27.7568, 26.8250, 26.8250])
m_BB2 = np.array([27.7568, 27.7568, 26.8250, 00.0000, 00.0000])
cont_lim = mag_combine(m_BB1, m_BB2, epsilon)

#Minimum NB excess color
minthres = [0.15, 0.15, 0.15, 0.2, 0.25]

from astropy import log

path0 = '/Users/cly/Google Drive/NASA_Summer2015/'

m_AB = 48.6
def color_cut(x, lim1, lim2, mean=0.0):
    f1 = 10**(-0.4*(m_AB+lim1))
    f2 = 10**(-0.4*(m_AB+lim2))

    f = 10**(-0.4*(m_AB+x))

    val = mean -2.5*np.log10(1 - np.sqrt(f1**2+f2**2)/f)

    return val
#enddef

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

    for rr in range(2):
        for cc in range(2):
            if cc == 0: ax[rr][cc].set_ylabel(r'$\log(M/M_{\odot})$')
            if cc == 1: ax[rr][cc].set_yticklabels([])
            ax[rr][cc].set_xlim(19.5,28.5)
            ax[rr][cc].set_ylim(4.0,11.0)

    prefixes = ['Ha-NB7','Ha-NB816','Ha-NB921','Ha-NB973']
    xlabels  = [r"$R_Ci$'", r"$i$'$z$'", "$z$'", "$z$'"]
    annot    = ['NB704,NB711', 'NB816', 'NB921', 'NB973']

    dmag = 0.4

    for ff in range(len(prefixes)):
        col = ff % 2
        row = ff / 2
        NB_idx = np.array([ii for ii in range(len(NB_tab)) if prefixes[ff] in NB_HA_Name[ii]])
        t_ax = ax[row][col]
        t_ax.scatter(cont_mag[NB_idx], logM_NB_Ha[NB_idx], edgecolor='blue',
                     color='none', alpha=0.5)
        t_ax.set_xlabel(xlabels[ff])
        t_ax.annotate(annot[ff], [0.975,0.975], xycoords='axes fraction',
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
    #endfor
    plt.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.97,
                        wspace=0.01)

    out_pdf = path0 + 'Completeness/mag_vs_mass.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    if silent == False: log.info('### End mag_vs_mass : '+systime())
#enddef

def ew_MC():

    prefixes = ['Ha-NB7','Ha-NB7','Ha-NB816','Ha-NB921','Ha-NB973']

    logEW_mean = np.arange(1.15,1.60,0.05)
    logEW_sig  = np.arange(0.15,0.45,0.05)

    Nsim = 200
    print('Nsim : ', Nsim)

    out_pdf = path0 + 'Completeness/ew_MC.pdf'
    pp = PdfPages(out_pdf)

    NBbin = 0.25

    for ff in range(len(filt_ref)): # loop over filter
        filt_dict = {'dNB': dNB[ff], 'dBB': dBB[ff], 'lambdac': lambdac[ff]}

        x      = np.arange(0.01,10.00,0.01)
        y_temp = 10**(-0.4 * x)
        EW_ref = np.log10(dNB[ff]*(1 - y_temp)/(y_temp - dNB[ff]/dBB[ff]))

        good = np.where(np.isfinite(EW_ref))[0]
        EW_int = interp1d(EW_ref[good], x[good], bounds_error=False,
                          fill_value=(-3.0, np.max(EW_ref[good])))

        #print np.max(EW_ref[good])
        NBmin = 20.0
        NBmax = m_NB[ff]-0.5
        NB = np.arange(NBmin,NBmax+NBbin,NBbin)
        print('NB (min/max)', min(NB), max(NB))

        # Read in mag vs mass extrapolation
        npz_mass_file = path0 + 'Completeness/mag_vs_mass_'+prefixes[ff]+'.npz'
        npz_mass = np.loadz(npz_mass_file)

        fig, ax = plt.subplots(ncols=2, nrows=2)
        for nn in range(len(NB)):
            for mm in [len(logEW_mean)-1]: #range(len(logEW_mean)): # loop over median of EW dist
                for ss in [0]: #range(len(logEW_sig)): # loop over sigma of EW dist
                    np.random.seed = mm*ss
                    rand0    = np.random.normal(0.0, 1.0, size=100)
                    logEW_MC = logEW_mean[mm] + logEW_sig[ss]*rand0

                    #print max(logEW_MC)
                    x_MC = EW_int(logEW_MC)

                    #ax.hist(x_MC, bins=50)
                    t_NB = np.repeat(NB[nn], len(x_MC))
                    ax[0][0].scatter(t_NB, x_MC, marker=',', s=1)

                    t_EW, t_flux = ew_flux_dual(t_NB, t_NB + x_MC, x_MC, filt_dict)
                    ax[0][1].scatter(t_NB, np.log10(t_flux))

                    ax[0][0].axhline(y=minthres[ff], linestyle='dashed', color='blue')
                    ax[0][0].plot(NB, color_cut(NB, m_NB[ff], cont_lim[ff]), 'b--')

                    annot_txt  = r'$\langle\log({\rm EW}_0)\rangle = %.2f$' % logEW_mean[mm] + '\n'
                    annot_txt += r'$\sigma[\log({\rm EW}_0)] = %.2f$' % logEW_sig[ss] + '\n'
                    ax[0][0].annotate(annot_txt, [0.05,0.95], xycoords='axes fraction',
                                      va='top', ha='left')

        fig.savefig(pp, format='pdf')

    pp.close()
