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

def color_cut(x, lim1, lim2, mean=0.0, sigma=3.0):
    '''
    NB excess color selection based on limiting magnitudes

    Parameters
    ----------

    x : Array of NB magnitudes

    lim1 : float
      3-sigma NB limiting magnitude

    lim2 : float
      3-sigma BB limiting magnitude

    sigma : float
      Sigma threshold.  Default: 3.0

    Returns
    -------

    val : array of 3-sigma allowed BB - NB excess color
    '''

    f1 = (sigma/3.0) * 10**(-0.4*(m_AB+lim1))
    f2 = (sigma/3.0) * 10**(-0.4*(m_AB+lim2))

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
    NB_tab      = asc.read(NB_file)
    NB_HA_Name  = NB_tab['NAME0'].data
    NB_Ha_ID    = NB_tab['ID'].data - 1 # Relative to 0 --> indexing

    NII_Ha_corr = NB_tab['nii_ha_corr_factor'].data # This is log(1+NII/Ha)
    filt_corr   = NB_tab['filt_corr_factor'].data # This is log(f_filt)

    NB_catfile = path0 + 'Catalogs/NB_IA_emitters.allcols.colorrev.fix.errors.fits'
    log.info("Reading : "+NB_catfile)
    NB_catdata = fits.getdata(NB_catfile)
    NB_catdata = NB_catdata[NB_Ha_ID]

    # These are the raw measurements
    NB_EW   = np.zeros(len(NB_catdata))
    NB_Flux = np.zeros(len(NB_catdata))

    Ha_EW   = np.zeros(len(NB_catdata))
    Ha_Flux = np.zeros(len(NB_catdata))

    for filt in filters:
        log.info('### Working on : '+filt)
        NB_idx = [ii for ii in range(len(NB_tab)) if 'Ha-'+filt in \
                  NB_HA_Name[ii]]
        print(" Size : ", len(NB_idx))
        NB_EW[NB_idx]   = np.log10(NB_catdata[filt+'_EW'][NB_idx])
        NB_Flux[NB_idx] = NB_catdata[filt+'_FLUX'][NB_idx]

        Ha_EW[NB_idx]   = (NB_EW   + NII_Ha_corr + filt_corr)[NB_idx]
        Ha_Flux[NB_idx] = (NB_Flux + NII_Ha_corr + filt_corr)[NB_idx]

        out_npz = path0 + 'Completeness/ew_flux_Ha-'+filt+'.npz'
        log.info("Writing : "+out_npz)
        np.savez(out_npz, NB_ID=NB_catdata['ID'][NB_idx], NB_EW=NB_EW[NB_idx],
                 NB_Flux=NB_Flux[NB_idx], Ha_EW=Ha_EW[NB_idx], Ha_Flux=Ha_Flux[NB_idx])
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

    bin_size = 0.25
    bins = np.arange(17.0,28,bin_size)

    NB_slope0 = np.zeros(len(filters))

    N_norm0 = []
    mag_arr = []

    for ff in range(len(filters)):
        print('Reading : '+NB_phot_files[ff])
        phot_tab = asc.read(NB_phot_files[ff])
        MAG_APER = phot_tab['col13'].data

        row = int(ff / 2)
        col = ff % 2

        ax = ax_arr[row][col]

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

        ax.axvline(m_NB[ff], linestyle='dashed', linewidth=1.5) #, label=r'3$\sigma$')
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

    out_npz = out_pdf.replace('.pdf', '.npz')
    np.savez(out_npz, filters=filters, bin_size=bin_size, NB_slope0=NB_slope0,
             mag_arr=mag_arr, N_norm0=N_norm0)
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

    npz_slope = np.load(path0 + 'Completeness/NB_numbers.npz')
    NB_slope0 = npz_slope['NB_slope0']

    logEW_mean = np.arange(1.25,1.55,0.1)
    logEW_sig  = np.arange(0.15,0.45,0.1)

    Nsim = 2000.
    print('Nsim : ', Nsim)

    NBbin = 0.25

    nrow_stats = 4

    for ff in range(len(filt_ref)): # loop over filter
        print("Working on : "+filters[ff])

        out_pdf = path0 + 'Completeness/ew_MC_'+filters[ff]+'.pdf'
        pp = PdfPages(out_pdf)

        out_pdf2 = path0 + 'Completeness/ew_MC_'+filters[ff]+'.stats.pdf'
        pp2 = PdfPages(out_pdf2)

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

        N_mag_mock = npz_slope['N_norm0'][ff] * Nsim * NBbin
        N_interp   = interp1d(npz_slope['mag_arr'][ff], N_mag_mock)
        Ndist_mock = np.int_(np.round(N_interp(NB)))
        NB_MC = np.repeat(NB, Ndist_mock)

        # Read in mag vs mass extrapolation
        npz_mass_file = path0 + 'Completeness/mag_vs_mass_'+prefixes[ff]+'.npz'
        npz_mass = np.load(npz_mass_file)
        cont_arr = npz_mass['cont_arr']
        dmag     = cont_arr[1]-cont_arr[0]
        mgood    = np.where(npz_mass['N_logM'] != 0)[0]
        mass_int = interp1d(cont_arr[mgood]+dmag/2.0, npz_mass['avg_logM'][mgood],
                            bounds_error=False, fill_value='extrapolate',
                            kind='linear')

        # Read in EW and fluxes for H-alpha NB emitter sample
        npz_NB_file = path0 + 'Completeness/ew_flux_Ha-'+filters[ff]+'.npz'
        npz_NB      = np.load(npz_NB_file)
        NB_EW   = npz_NB['NB_EW']
        Ha_Flux = npz_NB['Ha_Flux']

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

        N_mag_mock = npz_slope['N_norm0'][ff] * Nsim * NBbin
        N_interp   = interp1d(npz_slope['mag_arr'][ff], N_mag_mock)
        Ndist_mock = np.int_(np.round(N_interp(NB)))
        NB_MC = np.repeat(NB, Ndist_mock)
        
        # Read in mag vs mass extrapolation
        npz_mass_file = path0 + 'Completeness/mag_vs_mass_'+prefixes[ff]+'.npz'
        npz_mass = np.load(npz_mass_file)
        cont_arr = npz_mass['cont_arr']
        dmag     = cont_arr[1]-cont_arr[0]
        mgood    = np.where(npz_mass['N_logM'] != 0)[0]
        mass_int = interp1d(cont_arr[mgood]+dmag/2.0, npz_mass['avg_logM'][mgood],
                            bounds_error=False, fill_value='extrapolate',
                            kind='linear')

        lum_dist = cosmo.luminosity_distance(z_NB[ff]).to(u.cm).value

        # Read in EW and fluxes for H-alpha NB emitter sample
        npz_NB_file = path0 + 'Completeness/ew_flux_Ha-'+filters[ff]+'.npz'
        npz_NB      = np.load(npz_NB_file)
        NB_EW   = npz_NB['NB_EW']
        Ha_Flux = npz_NB['Ha_Flux']

        avg_NB = np.average(NB_EW)
        sig_NB = np.std(NB_EW)

        fig3, ax3 = plt.subplots(ncols=2, nrows=1)
        ax3[0].axhline(y=avg_NB, color='black', linestyle='dashed')
        ax3[0].axhspan(avg_NB-sig_NB, avg_NB+sig_NB, alpha=0.5, color='black')
        ax3[0].set_xlim([min(logEW_mean)-0.05,max(logEW_mean)+0.05])
        ax3[0].set_xlabel(r'$\log({\rm EW}/\AA)$')

        count = 0
        for mm in range(len(logEW_mean)): # loop over median of EW dist
            for ss in range(len(logEW_sig)): # loop over sigma of EW dist
                fig, ax = plt.subplots(ncols=2, nrows=3)
                plt.subplots_adjust(left=0.105, right=0.98, bottom=0.05,
                                    top=0.98, wspace=0.25, hspace=0.05)

                EW_arr0  = np.array([])
                EW_flag0 = np.array([])

                Flux_arr0  = np.array([])

                np.random.seed = mm*ss
                rand0 = np.random.normal(0.0, 1.0, size=len(NB_MC))
                logEW_MC = logEW_mean[mm] + logEW_sig[ss]*rand0 # This is NB EW (not H-alpha)

                EW_arr0 = np.append(EW_arr0, logEW_MC)
                EW_flag = np.zeros(len(logEW_MC))

                x_MC = EW_int(logEW_MC) # NB color excess
                negs = np.where(x_MC < 0)[0]
                if len(negs) > 0:
                    x_MC[negs] = 0.0

                # t_NB = np.repeat(NB_MC, len(x_MC))

                # Panel (0,0) - NB excess selection plot
                ax[0][0].scatter(NB_MC, x_MC, marker=',', s=1)

                ax[0][0].axhline(y=minthres[ff], linestyle='dashed',
                                 color='blue')

                y3 = color_cut(NB, m_NB[ff], cont_lim[ff])
                ax[0][0].plot(NB, y3, 'b--')
                y4 = color_cut(NB, m_NB[ff], cont_lim[ff], sigma=4.0)
                ax[0][0].plot(NB, y4, 'b:')
                ax[0][0].set_xticklabels([])
                ax[0][0].set_ylabel('cont - NB')

                annot_txt  = r'$\langle\log({\rm EW}_0)\rangle = %.2f$' % logEW_mean[mm] + '\n'
                annot_txt += r'$\sigma[\log({\rm EW}_0)] = %.2f$' % logEW_sig[ss] + '\n'
                annot_txt += r'$N$ = %i' % len(NB_MC)
                ax[0][0].annotate(annot_txt, [0.05,0.95], va='top',
                                  ha='left', xycoords='axes fraction')


                sig_limit = color_cut(NB_MC, m_NB[ff], cont_lim[ff]) #, sigma=4.0)
                NB_sel   = np.where((x_MC >= minthres[ff]) &
                                    (x_MC >= sig_limit))[0]
                NB_nosel = np.where((x_MC < minthres[ff]) |
                                    (x_MC < sig_limit))[0]

                EW_flag[NB_sel] = 1
                EW_flag0 = np.append(EW_flag0, EW_flag)

                t_EW, t_flux = ew_flux_dual(NB_MC, NB_MC + x_MC, x_MC,
                                            filt_dict)

                # Apply NB filter correction from beginning
                t_flux = np.log10(t_flux * filt_corr[ff])

                cont_MC = NB_MC + x_MC
                logM_MC = mass_int(cont_MC)
                NIIHa, logOH = get_NIIHa_logOH(logM_MC)

                t_Haflux = correct_NII(t_flux, NIIHa)

                Flux_arr0 = np.append(Flux_arr0, t_Haflux)

                # Panel (1,0) - NB mag vs H-alpha flux
                ax[1][0].scatter(NB_MC[NB_sel], t_Haflux[NB_sel], alpha=0.25,
                                 s=2, edgecolor='none')
                ax[1][0].scatter(NB_MC[NB_nosel], t_Haflux[NB_nosel],
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

                EW_bins = np.arange(0.2,3.0,0.2)

                # This is for statistics plot
                if count % nrow_stats == 0:
                    fig2, ax2 = plt.subplots(ncols=2, nrows=nrow_stats)
                s_row = count % nrow_stats

                # Panel (2,0) - histogram of EW

                # NB_counts, NB_bins = np.histogram(NB_EW, np.arange(0.5,3.0,0.2))
                label_EW = r'N: %i  $\langle x\rangle$: %.2f  $\sigma$: %.2f' % \
                           (len(NB_EW), avg_NB, sig_NB)
                No, binso, _ = ax[2][0].hist(NB_EW, bins=EW_bins, align='mid', color='blue',
                                             linestyle='solid', edgecolor='none',
                                             histtype='stepfilled', label=label_EW)
                ax[2][0].axvline(x=avg_NB, color='blue', linestyle='dashed',
                                 linewidth=1.5)

                good = np.where(EW_flag0)[0]

                # Normalize relative to selected sample
                if len(good) > 0:
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
                    Ng, binsg, _ = ax[2][0].hist(EW_arr0[good], bins=EW_bins, weights=wht0[good],
                                                 align='mid', alpha=0.5, color='red', edgecolor='red',
                                                 linestyle='solid', histtype='stepfilled', label=label1)
                    ax[2][0].axvline(x=avg_gd, color='red', linestyle='dashed',
                                     linewidth=1.5)

                ax[2][0].legend(loc='upper right', fancybox=True, fontsize=6,
                                framealpha=0.75)
                ax[2][0].set_xlabel(r'$\log({\rm EW}/\AA)$')
                ax[2][0].set_ylabel(r'$N$')
                ax[2][0].set_yscale('log')
                ax[2][0].set_position([0.105,0.05,0.389,0.265])

                if len(good) > 0:
                    delta    = (Ng-No)/np.sqrt(Ng**0.5 + No**0.5)
                    ax2[s_row][0].scatter(binso[:-1], delta)

                    ax2[s_row][0].axhline(0.0, linestyle='dashed')
                    ax2[s_row][0].set_ylabel(r'1 - $N_{\rm mock}/N_{\rm data}$')
                    ax2[s_row][0].set_ylabel(r'$(N_{\rm mock} - N_{\rm data})/\sigma$')

                    annot_txt  = r'$\langle\log({\rm EW}_0)\rangle = %.2f$  ' % logEW_mean[mm]
                    annot_txt += r'$\sigma[\log({\rm EW}_0)] = %.2f$' % logEW_sig[ss]
                    ax2[s_row][0].set_title(annot_txt, fontdict={'fontsize': 10}, loc='left')

                    # Compute chi^2
                    use_bins = np.where((Ng != 0) & (No != 0))[0]
                    fit_chi2 = np.sum(delta[use_bins]**2)/(len(use_bins)-2)
                    ax2[s_row][0].annotate(r'$\chi^2_{\nu}$ = %.2f' % fit_chi2, [0.975,0.975],
                                           xycoords='axes fraction', ha='right', va='top')

                # Panel (2,1) - histogram of H-alpha fluxes

                Flux_bins = np.arange(-17.75,-14.75,0.25)

                No, binso, _ = ax[2][1].hist(Ha_Flux, bins=Flux_bins, align='mid',
                                             color='blue', linestyle='solid', edgecolor='none',
                                             histtype='stepfilled')

                if len(good) > 0:
                    finite = np.where(np.isfinite(Flux_arr0))
                    N, bins, _ = ax[2][1].hist(Flux_arr0[finite], bins=Flux_bins,
                                               weights=wht0[finite], align='mid',
                                               color='black', linestyle='solid',
                                               edgecolor='black', histtype='step')

                    Ng, binsg, _ = ax[2][1].hist(Flux_arr0[good], bins=Flux_bins, alpha=0.5,
                                                 weights=wht0[good], align='mid', color='red',
                                                 edgecolor='red', linestyle='solid',
                                                 histtype='stepfilled')

                ax[2][1].set_xlabel(r'$\log(F_{{\rm H}\alpha})$')
                ax[2][1].set_ylabel(r'$N$')
                ax[2][1].set_yscale('log')
                ax[2][1].set_position([0.591,0.05,0.389,0.265])

                if len(good) > 0:
                    delta = (Ng-No)/np.sqrt(Ng**0.5+No**0.5)
                    ax2[s_row][1].scatter(binso[:-1], delta)
                    ax2[s_row][1].axhline(0.0, linestyle='dashed')

                    # Compute chi^2
                    use_bins = np.where((Ng != 0) & (No != 0))[0]
                    fit_chi2 = np.sum(delta[use_bins]**2)/(len(use_bins)-2)
                    ax2[s_row][1].annotate(r'$\chi^2_{\nu}$ = %.2f' % fit_chi2, [0.975,0.975],
                                           xycoords='axes fraction', ha='right', va='top')

                if s_row != nrow_stats-1:
                    ax2[s_row][0].set_xticklabels([])
                    ax2[s_row][1].set_xticklabels([])
                else:
                    ax2[s_row][0].set_xlabel(r'$\log({\rm EW}/\AA)$')
                    ax2[s_row][1].set_xlabel(r'$\log(F_{{\rm H}\alpha})$')

                fig.set_size_inches(8,10)
                fig.savefig(pp, format='pdf')

                if s_row == nrow_stats-1:
                    fig2.subplots_adjust(left=0.1, right=0.97, bottom=0.08, top=0.97,
                                         wspace=0.13)

                    fig2.set_size_inches(8,10)
                    fig2.savefig(pp2, format='pdf')

                count += 1
            #endfor

        #endfor


        pp.close()
        pp2.close()

        out_pdf3 = path0 + 'Completeness/ew_MC_'+filters[ff]+'.avg_sigma.pdf'
        fig3.savefig(out_pdf3, format='pdf')

