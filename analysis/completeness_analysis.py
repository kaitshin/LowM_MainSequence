"""
completeness_analysis
====

A set of Python 2.7 codes for completeness analysis of NB-selected galaxies
in the M*-SFR plot
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

from ..mainseq_corrections import niiha_oh_determine

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 70 * u.km / u.s / u.Mpc, Om0=0.3)

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

filters = ['NB704','NB711','NB816','NB921','NB973']

def color_cut(x, lim1, lim2, mean=0.0):
    '''
    NB excess color selection based on limiting magnitudes

    Parameters
    ----------

    x : Array of NB magnitudes

    lim1 : float
      3-sigma NB limiting magnitude

    lim2 : float
      3-sigma BB limiting magnitude

    Returns
    -------

    val : array of 3-sigma allowed BB - NB excess color
    '''

    f1 = 10**(-0.4*(m_AB+lim1))
    f2 = 10**(-0.4*(m_AB+lim2))

    f = 10**(-0.4*(m_AB+x))

    val = mean -2.5*np.log10(1 - np.sqrt(f1**2+f2**2)/f)

    return val
#enddef

def correct_NII(log_flux, NIIHa):
    '''
    This returns Halpha fluxes from F_NB using NII/Ha flux ratios for
    correction
    '''
    return log_flux - np.log10(1+NIIHa)
#enddef

def get_NIIHa_logOH(logM):
    '''
    Get [NII]6548,6583/H-alpha flux ratios and oxygen abundance based on
    stellar mass.  Metallicity is from PP04
    '''

    NIIHa = np.zeros(len(logM))

    low_mass = np.where(logM <= 8.0)[0]
    if len(low_mass) > 0:
        NIIHa[low_mass] = 0.0624396766589

    high_mass = np.where(logM > 8.0)[0]
    if len(high_mass) > 0:
        NIIHa[high_mass] = 0.169429547993*logM[high_mass] - 1.29299670728


    # Compute metallicity
    NII6583_Ha = NIIHa * 1/(1+1/2.96)
    logOH = niiha_oh_determine(np.log10(NII6583_Ha), 'PP04_N2') - 12.0

    return NIIHa, logOH
#enddef

def HaSFR_metal_dep(logOH, orig_lums):
    '''
    Determine H-alpha SFR using metallicity and luminosity to follow
    Ly+ 2016 metallicity-dependent SFR conversion
    '''

    y = logOH + 3.31
    log_SFR_LHa = -41.34 + 0.39 * y + 0.127 * y**2

    log_SFR = log_SFR_LHa + orig_lums

    return log_SFR
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

    for filt in filters:
        log.info('### Working on : '+filt)
        NB_idx = [ii for ii in range(len(NB_tab)) if 'Ha-'+filt in \
                  NB_HA_Name[ii]]
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
        NB_idx = np.array([ii for ii in range(len(NB_tab)) if prefixes[ff] in \
                           NB_HA_Name[ii]])
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

def get_EW_Flux_distribution():
    '''
    Retrieve NB excess emission-line EW and fluxes from existing tables
    '''

    # NB Ha emitter sample for ID
    NB_file = path0 + 'Main_Sequence/mainseq_corrections_tbl (1).txt'
    log.info("Reading : "+NB_file)
    NB_tab     = asc.read(NB_file)
    NB_HA_Name = NB_tab['NAME0'].data
    NB_Ha_ID   = NB_tab['ID'].data - 1 # Relative to 0 --> indexing

    NB_catfile = path0 + 'Catalogs/NB_IA_emitters.allcols.colorrev.fix.errors.fits'
    log.info("Reading : "+NB_catfile)
    NB_catdata = fits.getdata(NB_catfile)
    NB_catdata = NB_catdata[NB_Ha_ID]

    NB_EW   = np.zeros(len(NB_catdata))
    NB_Flux = np.zeros(len(NB_catdata))

    for filt in filters:
        log.info('### Working on : '+filt)
        NB_idx = [ii for ii in range(len(NB_tab)) if 'Ha-'+filt in \
                  NB_HA_Name[ii]]
        print(" Size : ", len(NB_idx))
        NB_EW[NB_idx]   = np.log10(NB_catdata[filt+'_EW'][NB_idx])
        NB_Flux[NB_idx] = NB_catdata[filt+'_FLUX'][NB_idx]

        out_npz = path0 + 'Completeness/ew_flux_Ha-'+filt+'.npz'
        log.info("Writing : "+out_npz)
        np.savez(out_npz, NB_EW=NB_EW[NB_idx], NB_Flux=NB_Flux[NB_idx])
    #endfor

#enddef

def NB_numbers():
    '''
    Uses SExtractor catalog to look at number of NB galaxies vs magnitude
    '''

    NB_path = '/Users/cly/data/SDF/NBcat/'
    NB_phot_files = [NB_path+filt+'/sdf_pub2_'+filt+'.cat.mask' for filt in filters]

    out_pdf = path0 + 'Completeness/NB_numbers.pdf'

    fig, ax_arr = plt.subplots(nrows=3, ncols=2)

    bins = np.arange(17.0,28,0.25)

    for ff in range(len(filters)):
        print('Reading : '+NB_phot_files[ff])
        phot_tab = asc.read(NB_phot_files[ff])
        MAG_APER = phot_tab['col13'].data

        row = int(ff / 2)
        col = ff % 2

        ax = ax_arr[row][col]

        N, m_bins, _ = ax.hist(MAG_APER, bins=bins, align='mid', color='black',
                               linestyle='solid', histtype='step')

        ax.axvline(m_NB[ff], linestyle='dashed', linewidth=1.5)
        ax.annotate(filters[ff], [0.025,0.975], xycoords='axes fraction',
                    ha='left', va='top', fontsize=12)
        ax.set_yscale('log')

        #ax.xaxis.set_ticklabels([])

    fig.savefig(out_pdf, bbox_inches='tight')

#enddef

def ew_MC():
    '''
    Main function for Monte Carlo realization.  Adopts log-normal
    EW distribution to determine survey sensitivity and impact on
    M*-SFR relation
    '''

    prefixes = ['Ha-NB7','Ha-NB7','Ha-NB816','Ha-NB921','Ha-NB973']

    # NB statistical filter correction
    filt_corr = [1.289439104,   1.41022358406, 1.29344789854,
                 1.32817034288, 1.29673596942]

    z_NB     = lambdac/6562.8 - 1.0

    logEW_mean = np.arange(1.25,1.55,0.1)
    logEW_sig  = np.arange(0.15,0.45,0.1)

    Nsim = 200
    print('Nsim : ', Nsim)

    NBbin = 0.25

    for ff in range(len(filt_ref)): # loop over filter
        out_pdf = path0 + 'Completeness/ew_MC_'+filters[ff]+'.pdf'
        print("Working on : "+filters[ff])

        pp = PdfPages(out_pdf)

        filt_dict = {'dNB': dNB[ff], 'dBB': dBB[ff], 'lambdac': lambdac[ff]}

        x      = np.arange(0.01,10.00,0.01)
        y_temp = 10**(-0.4 * x)
        EW_ref = np.log10(dNB[ff]*(1 - y_temp)/(y_temp - dNB[ff]/dBB[ff]))

        good = np.where(np.isfinite(EW_ref))[0]
        EW_int = interp1d(EW_ref[good], x[good], bounds_error=False,
                          fill_value=(-3.0, np.max(EW_ref[good])))

        NBmin = 20.0
        NBmax = m_NB[ff]-0.5
        NB = np.arange(NBmin,NBmax+NBbin,NBbin)
        print('NB (min/max)', min(NB), max(NB))

        # Read in mag vs mass extrapolation
        npz_mass_file = path0 + 'Completeness/mag_vs_mass_'+prefixes[ff]+'.npz'
        npz_mass = np.load(npz_mass_file)
        cont_arr = npz_mass['cont_arr']
        dmag = cont_arr[1]-cont_arr[0]
        mass_int = interp1d(cont_arr+dmag/2.0, npz_mass['avg_logM'],
                            bounds_error=False, fill_value=(0.0,15.0))

        lum_dist = cosmo.luminosity_distance(z_NB[ff]).to(u.cm).value

        for mm in range(len(logEW_mean)): # loop over median of EW dist
            for ss in range(len(logEW_sig)): # loop over sigma of EW dist
                fig, ax = plt.subplots(ncols=2, nrows=3)
                plt.subplots_adjust(left=0.105, right=0.98, bottom=0.05,
                                    top=0.98, wspace=0.25, hspace=0.05)

                EW_arr0  = np.array([])
                EW_flag0 = np.array([])

                Flux_arr0  = np.array([])

                for nn in range(len(NB)):
                    np.random.seed = mm*ss
                    rand0    = np.random.normal(0.0, 1.0, size=100)
                    logEW_MC = logEW_mean[mm] + logEW_sig[ss]*rand0

                    EW_arr0 = np.append(EW_arr0, logEW_MC)
                    EW_flag = np.zeros(len(logEW_MC))

                    x_MC = EW_int(logEW_MC)

                    t_NB = np.repeat(NB[nn], len(x_MC))

                    # Panel (0,0) - NB excess selection plot
                    ax[0][0].scatter(t_NB, x_MC, marker=',', s=1)

                    ax[0][0].axhline(y=minthres[ff], linestyle='dashed',
                                     color='blue')
                    ax[0][0].plot(NB, color_cut(NB, m_NB[ff], cont_lim[ff]),
                                  'b--')
                    ax[0][0].set_xticklabels([])
                    ax[0][0].set_ylabel('cont - NB')

                    sig_limit = color_cut(t_NB, m_NB[ff], cont_lim[ff])
                    NB_sel   = np.where((x_MC >= minthres[ff]) &
                                        (x_MC >= sig_limit))[0]
                    NB_nosel = np.where((x_MC < minthres[ff]) |
                                        (x_MC < sig_limit))[0]

                    EW_flag[NB_sel] = 1
                    EW_flag0 = np.append(EW_flag0, EW_flag)

                    t_EW, t_flux = ew_flux_dual(t_NB, t_NB + x_MC, x_MC,
                                                filt_dict)

                    # Apply NB filter correction from beginning
                    t_flux = np.log10(t_flux * filt_corr[ff])

                    logM_MC = mass_int(t_NB + x_MC)
                    NIIHa, logOH = get_NIIHa_logOH(logM_MC)

                    t_Haflux = correct_NII(t_flux, NIIHa)

                    Flux_arr0 = np.append(Flux_arr0, t_Haflux)

                    # Panel (1,0) - NB mag vs H-alpha flux
                    ax[1][0].scatter(t_NB[NB_sel], t_Haflux[NB_sel], alpha=0.25,
                                     s=2, edgecolor='none')
                    ax[1][0].scatter(t_NB[NB_nosel], t_Haflux[NB_nosel],
                                     alpha=0.25, s=2, edgecolor='blue',
                                     linewidth=0.25, facecolor='none')
                    ax[1][0].set_xlabel('NB')
                    ax[1][0].set_ylabel(r'$\log(F_{H\alpha})$')

                    t_HaLum = t_Haflux +np.log10(4*np.pi) +2*np.log10(lum_dist)


                    # Panel (0,1) - stellar mass vs H-alpha luminosity
                    ax[0][1].scatter(logM_MC[NB_sel], t_HaLum[NB_sel],
                                     alpha=0.25, s=2, edgecolor='none')
                    ax[0][1].scatter(logM_MC[NB_nosel], t_HaLum[NB_nosel],
                                     alpha=0.25, s=2, edgecolor='blue',
                                     linewidth=0.25, facecolor='none')
                    ax[0][1].set_xticklabels([])
                    ax[0][1].set_ylabel(r'$\log(L_{{\rm H}\alpha})$')
                    #ax[1][1].set_ylim([37.5,43.0])

                    # Panel (1,1) - stellar mass vs H-alpha SFR
                    logSFR_MC = HaSFR_metal_dep(logOH, t_HaLum)
                    ax[1][1].scatter(logM_MC[NB_sel], logSFR_MC[NB_sel],
                                     alpha=0.25, s=2, edgecolor='none')
                    ax[1][1].scatter(logM_MC[NB_nosel], logSFR_MC[NB_nosel],
                                     alpha=0.25, s=2, edgecolor='blue',
                                     linewidth=0.25, facecolor='none')
                    ax[1][1].set_xlabel(r'$\log(M_{\star}/M_{\odot})$')
                    ax[1][1].set_ylabel(r'$\log({\rm SFR}({\rm H}\alpha))$')


                    annot_txt  = r'$\langle\log({\rm EW}_0)\rangle = %.2f$' % logEW_mean[mm] + '\n'
                    annot_txt += r'$\sigma[\log({\rm EW}_0)] = %.2f$' % logEW_sig[ss] + '\n'
                    ax[0][0].annotate(annot_txt, [0.05,0.95], va='top',
                                      ha='left', xycoords='axes fraction')
                #endfor

                # Read in EW and fluxes for H-alpha NB emitter sample
                npz_NB_file = path0 + 'Completeness/ew_flux_Ha-'+filters[ff]+'.npz'
                npz_NB      = np.load(npz_NB_file)
                NB_EW   = npz_NB['NB_EW']
                NB_Flux = npz_NB['NB_Flux']

                avg_NB = np.average(NB_EW)
                sig_NB = np.std(NB_EW)

                EW_bins = np.arange(0.2,3.0,0.2)

                # Panel (2,0) - histogram of EW

                # NB_counts, NB_bins = np.histogram(NB_EW, np.arange(0.5,3.0,0.2))
                label_EW = r'N: %i  $\langle x\rangle$: %.2f  $\sigma$: %.2f' % \
                           (len(NB_EW), avg_NB, sig_NB)
                ax[2][0].hist(NB_EW, bins=EW_bins, align='mid', color='blue',
                              linestyle='solid', edgecolor='none',
                              histtype='stepfilled', label=label_EW)
                ax[2][0].axvline(x=avg_NB, color='blue', linestyle='dashed',
                                 linewidth=1.5)

                good = np.where(EW_flag0)[0]

                # Normalize relative to selected sample
                norm0 = float(len(NB_EW))/len(good)
                wht0  = np.repeat(norm0, len(EW_arr0))

                avg_MC = np.average(EW_arr0)
                sig_MC = np.std(EW_arr0)
                label0 = r'N: %i  $\langle x\rangle$: %.2f  $\sigma$: %.2f ' % \
                         (len(EW_arr0), avg_MC, sig_MC)
                N, bins, _ = ax[2][0].hist(EW_arr0, bins=EW_bins, weights=wht0,
                                           align='mid', color='black',
                                           linestyle='solid', edgecolor='black',
                                           histtype='step', label=label0)
                ax[2][0].axvline(x=avg_MC, color='black', linestyle='dashed',
                                 linewidth=1.5)

                avg_gd = np.average(EW_arr0[good])
                sig_gd = np.std(EW_arr0[good])
                label1 = r'N: %i  $\langle x\rangle$: %.2f  $\sigma$: %.2f ' % \
                         (len(good), avg_gd, sig_gd)
                ax[2][0].hist(EW_arr0[good], bins=EW_bins, weights=wht0[good],
                              align='mid', alpha=0.5, color='red', edgecolor='red',
                              linestyle='solid', histtype='stepfilled', label=label1)
                ax[2][0].axvline(x=avg_gd, color='red', linestyle='dashed',
                                 linewidth=1.5)

                #nonzero = np.where(N > 0)[0]
                #ax[2][0].set_xlim(bins[nonzero[0]],bins[nonzero[-1]])

                ax[2][0].legend(loc='upper right', fancybox=True, fontsize=6,
                                framealpha=0.75)
                ax[2][0].set_xlabel(r'$\log({\rm EW}/\AA)$')
                ax[2][0].set_ylabel(r'$N$')
                ax[2][0].set_position([0.105,0.05,0.389,0.265])


                # Panel (2,1) - histogram of H-alpha fluxes

                Flux_bins = np.arange(-17.75,-14.75,0.25)

                ax[2][1].hist(NB_Flux, bins=Flux_bins, align='mid',
                              color='blue', linestyle='solid', edgecolor='none',
                              histtype='stepfilled')

                finite = np.where(np.isfinite(Flux_arr0))
                N, bins, _ = ax[2][1].hist(Flux_arr0[finite], bins=Flux_bins,
                                           weights=wht0[finite], align='mid',
                                           color='black', linestyle='solid',
                                           edgecolor='black', histtype='step')

                ax[2][1].hist(Flux_arr0[good], bins=Flux_bins, alpha=0.5,
                              weights=wht0[good], align='mid', color='red',
                              edgecolor='red', linestyle='solid',
                              histtype='stepfilled')

                ax[2][1].set_xlabel(r'$\log(F_{\rm NB})$')
                ax[2][1].set_ylabel(r'$N$')
                ax[2][1].set_position([0.591,0.05,0.389,0.265])

                fig.set_size_inches(8,10)
                fig.savefig(pp, format='pdf')
            #endfor
        #endfor

        pp.close()
