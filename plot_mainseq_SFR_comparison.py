"""
NAME:
    plot_mainseq_SFR_comparison.py

PURPOSE:
    This code plots comparisons of SFR (through luminosity):
    * SED SFR vs. Ha luminosity
    * UV photometry luminosity vs. Ha luminosity.
    For each plot, there are 4 subplots for 4 different types of corrections.
    Then, one plot is created with only the sources that had Balmer
    corrections applied.
    With the GALEX command line option, if GALEX is typed, then GALEX files
    files used/output. Otherwise, the files without GALEX photometry are used.

INPUTS:
    'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+fileend+'.fout'
    'FAST/outputs/BEST_FITS/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'
         +fileend+'_'+str(ID[ii])+'.fit'
    'Main_Sequence/Catalogs/mainseq_Ha_corrections'+fileend+'.fits'
    'Main_Sequence/Catalogs/EBV_both_HAHB_HGHB_no_overlap.txt'
    'Catalogs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+fileend+'.cat'

CALLING SEQUENCE:
    main body -> plot_SFR_Ha_vs_SED -> get_bad_FUV_NUV
              -> plot_SFR_Ha_vs_UV -> get_lnu -> get_flux
                                   -> get_corr0, get_corr12
                                   -> get_bad_FUV_NUV
              -> plot_just_balmer_decrements -> get_lnu -> get_flux
                                             -> get_corr12

OUTPUTS:
    'Plots/SFR_comparisons/SFR_Ha_vs_SED'+fileend+'.pdf'
    'Plots/SFR_comparisons/SFR_Ha_vs_UV'+fileend+'.pdf'
    'Plots/SFR_comparisons/SFR_Ha_vs_UV_balmer_only'+fileend+'.pdf'

REVISION HISTORY:
    Created by Kaitlyn Shin 17 August 2015
    Revised by Kaitlyn Shin 20 August 2015
    * Added get_bad_FUV_NUV
"""

import numpy as np, matplotlib.pyplot as plt, astropy.units as u, sys
from astropy.io import fits as pyfits, ascii as asc
from scipy import interpolate
from astropy import constants
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 70 * u.km / u.s / u.Mpc, Om0=0.3)

fileend='.GALEX'

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'


def get_flux(ID, lambda_arr):
    '''
    Reads in the relevant SED spectrum file and then interpolates the
    function to obtain a flux, the array of which is then returned.
    '''
    newflux = np.zeros(len(ID))
    for ii in range(len(ID)):
        tempfile = asc.read(FULL_PATH+
            'FAST/outputs/BEST_FITS/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+
            fileend+'_'+str(ID[ii])+'.fit', guess=False,Reader=asc.NoHeader)
        wavelength = np.array(tempfile['col1'])
        flux = np.array(tempfile['col2'])
        f = interpolate.interp1d(wavelength, flux)
        newflux[ii] = f(lambda_arr[ii])

    return newflux


def get_lnu(filt_index):
    '''
    Calls get_flux with an array of redshifted wavelengths in order to get
    the corresponding flux values. Those f_lambda values are then converted
    into f_nu values, which is in turn converted into L_nu, the log of
    which is returned as nu_lnu.
    '''
    ID    = ha_id[filt_index]
    zspec = ha_zspec[filt_index]
    zphot = ha_zphot[filt_index]
    goodz = np.array([x for x in range(len(zspec)) if zspec[x] < 9. and
                      zspec[x] > 0.])
    badz  = np.array([x for x in range(len(zspec)) if zspec[x] >= 9. or
                      zspec[x] <= 0.])
    tempz = np.zeros(len(filt_index))
    if len(goodz) > 0:
        tempz[goodz] = zspec[goodz]
    if len(badz) > 0:
        tempz[badz]  = zphot[badz]

    lambda_arr = (1+tempz)*1500

    f_lambda = get_flux(ID, lambda_arr)
    f_nu = f_lambda*(1E-19*(lambda_arr**2*1E-10)/(constants.c.value))
    L_nu = f_nu*4*np.pi*(cosmo.luminosity_distance(tempz).to(u.cm).value)**2
    return np.log10(L_nu)
#endef


def get_bad_FUV_NUV(ff, filt_match):
    '''
    This code first looks at the filter to determine whether it'll look at
    the FUV or the NUV flux. Then, the indexes where that flux doesn't
    exist (==-99) are returned as well as a color (for plotting).

    Note: the indexes are returned as 'the indexes of filt_match'
    '''
    if ff=='NB704' or ff=='NB711':
        flux_filt = f_FUV[filt_match]
        tempcolor='teal'
    else:
        flux_filt = f_NUV[filt_match]
        tempcolor='pink'
    #endif
    neg_flux = np.where(flux_filt==-99)[0]
    return neg_flux, tempcolor
#enddef


def plot_SFR_Ha_vs_SED():
    '''
    Plots the SFR from the SED fits against the Ha luminosity (observed, but
    corrected for filter and nii/ha). There are four different subplots:
    * (a) SED SFR vs. observed Ha luminosity
    * (b) SED SFR vs. Ha luminosity+0.4dex (corrected for 1 mag. extinction)
    * (c) SED SFR vs. Hopkins corrected Ha luminosity (N/A as of 170815)
    * (d) SED SFR vs. dust-corrected Ha luminosity

    Iterates by filter to plot the points in different colors. Good zspec
    sources are filled, while bad zspec sources are empty.

    The 'ideal' line is plotted in red, with the median line (obtained from
    yarr_values-xarr_values) plotted in a dashed black. The last plot
    has cyan dotted lines marking a 'cutoff' above which SED plots were
    generated.

    The plot is modified, with labels added before the figure is saved and
    closed.

    If using GALEX files, get_bad_FUV_NUV is called to find where there is
    no UV photometry (filter-dependent whether code looks at FUV or NUV).
    '''
    print '### plotting SFR_Ha vs. SFR_SED'
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    f.subplots_adjust(hspace=0.05)
    f.subplots_adjust(wspace=0.05)
    f.set_size_inches(10, 10)
    for (ff, cc) in zip(filt_arr, color_arr):
        print ff
        filt_match = np.array([x for x in range(len(names)) if 'Ha-'+ff in
                               names[x]])
        filtz = ha_zspec[filt_match]
        bad_UV, tempcolor = get_bad_FUV_NUV(ff, filt_match)

        goodz = np.array([x for x in range(len(filtz)) if filtz[x]>0 and
                          filtz[x] < 9])
        badz  = np.array([x for x in range(len(filtz)) if filtz[x]<=0 or
                          filtz[x] >= 9])
        ax1.scatter(sed_sfr[filt_match][goodz], nii_lumin[filt_match][goodz],
                    facecolor=cc, edgecolor='none', alpha=0.5)
        ax1.scatter(sed_sfr[filt_match][badz], nii_lumin[filt_match][badz],
                    facecolor='none', edgecolor=cc, linewidth=0.5, alpha=0.5)
        ax1.scatter(sed_sfr[filt_match][bad_UV],
                    nii_lumin[filt_match][bad_UV], marker='x',
                    color=tempcolor,alpha=0.5,linewidth=0.5)

        
        ax2.scatter(sed_sfr[filt_match][goodz],
                    nii_lumin[filt_match][goodz]+0.4, facecolor=cc,
                    edgecolor='none', alpha=0.5)
        ax2.scatter(sed_sfr[filt_match][badz], nii_lumin[filt_match][badz]+0.4,
                    facecolor='none', edgecolor=cc, linewidth=0.5, alpha=0.5)
        ax2.scatter(sed_sfr[filt_match][bad_UV],
                    nii_lumin[filt_match][bad_UV]+0.4, marker='x',
                    color=tempcolor,alpha=0.5,linewidth=0.5)        
    
    
        ax4.scatter(sed_sfr[filt_match][goodz], dust_lumin[filt_match][goodz],
                    facecolor=cc, edgecolor='none', alpha=0.5)
        ax4.scatter(sed_sfr[filt_match][badz], dust_lumin[filt_match][badz],
                    facecolor='none', edgecolor=cc, linewidth=0.5, alpha=0.5)
        ax4.scatter(sed_sfr[filt_match][bad_UV],
                    dust_lumin[filt_match][bad_UV], marker='x',
                    color=tempcolor,alpha=0.5,linewidth=0.5)

    #endfor
    plt.setp([a.minorticks_on() for a in f.axes[:]])
    plt.setp([a.set_xlim(-6, 4) and a.set_ylim(37, 44) for a in f.axes[:]])

    ax1.text(min(ax1.get_xlim())+0.5, max(ax1.get_ylim())-0.5, '(a) Observed',
             fontsize=9, ha='left', va='top')
    ax2.text(min(ax2.get_xlim())+0.5, max(ax2.get_ylim())-0.5,
             '(b) Observed + 0.4 dex', fontsize=9, ha='left', va='top')
    ax3.text(min(ax3.get_xlim())+0.5, max(ax3.get_ylim())-0.5,
             '(c) Observed + H01 Ext.-Corr.', fontsize=9, ha='left', va='top')
    ax4.text(min(ax4.get_xlim())+0.5, max(ax4.get_ylim())-0.5,
             '(d) Observed + SED Ext.-Corr.', fontsize=9, ha='left', va='top')

    ax1.set_ylabel('log(H'+r'$\alpha$'+' Luminosity) [erg s'+r'$\^{-1}$'+']',
                   fontsize=9)
    ax3.set_ylabel('log(H'+r'$\alpha$'+' Luminosity) [erg s'+r'$\^{-1}$'+']',
                   fontsize=9)
    ax3.set_xlabel('log(SED SFR) [M'+r'$_{\odot}$'+' yr'+r'$\^{-1}$'+']',
                   fontsize=9)
    ax4.set_xlabel('log(SED SFR) [M'+r'$_{\odot}$'+' yr'+r'$\^{-1}$'+']',
                   fontsize=9)

    line=np.log10(7.9/1.8)-42
    ax4.plot([-3.158, 100], np.array([-3.158, 100])-line+1, 'c:', linewidth=0.5)
    ax4.plot((-6, -3.158), [39.2, 39.2], 'c:', linewidth=0.5)

    f0 = plt.scatter(100,100,marker='x',color='teal',alpha=0.5,label='no FUV')
    n0 = plt.scatter(100,100,marker='x',color='pink',alpha=0.5,label='no NUV')

    for (ax, num) in zip(f.axes[:], [1, 2, 3, 4]):
        if num==1:
            median = np.median(nii_lumin - sed_sfr)
            stddev = np.std(nii_lumin - sed_sfr)
        elif num==2:
            median = np.median((nii_lumin+0.4) - sed_sfr)
            stddev = np.std((nii_lumin+0.4) - sed_sfr)
        elif num==4:
            median = np.median(dust_lumin - sed_sfr)
            stddev = np.std(dust_lumin - sed_sfr)
        #endif
        if num != 3:
            ax.plot([-100, 100],
                    np.array([-100., 100.])-(np.log10(7.9/1.8)-42), 'r-',
                    linewidth=0.5)
            line, = ax.plot([-100, 100], np.array([-100, 100])+median, 'k--',
                            linewidth=0.5, label='median='
                            +str(np.around(median, 3))+'\nstd dev='
                            +str(np.around(stddev, 3)))

            legend0 = ax.legend(handles=[line,f0,n0],fontsize=9,
                                loc='lower right',frameon=False,
                                numpoints=1)

            ax.add_artist(legend0)
        #endif
    #endfor

    plt.savefig('Plots/SFR_comparisons/SFR_Ha_vs_SED'+fileend+'.pdf')
    plt.close()
#enddef


def get_corr0():
    '''
    Returns the correction factor for the UV luminosity based on 1 magnitude
    of Halpha extinction in the UV band.
    '''
    Ha_lambda = 0.6563
    K_Ha      = (2.659*(-1.857 + 1.040/Ha_lambda) + 4.05)
    Estar     = 1.0 * 0.44 / K_Ha
    
    UV_lambda  = 0.15
    K_UV       = (2.659*(-2.156 + 1.509/UV_lambda - 0.198/UV_lambda**2
                         + 0.011/UV_lambda**3)+ 4.05)
    corr_factor0 = 0.4*Estar*K_UV
    return corr_factor0
#enddef


def get_corr12(filt_match):
    '''
    Method is function is called on a filter-by-filter case.

    Reads in the EBV file and then finds the indexes at which there are the
    Ha+ff emitters in both cases. At those indexes, the 2nd correciton
    factor uses those Egas values*0.44 instead of the Estar values obtained
    from the .fout file.

    Returns the 1st correction factor for the UV luminosity

    Returns the 2nd correction factor for the Ha luminosity

    Returns the relevant index matches from the original 9264-source-long
    file. Important really only for the Balmer decrements only plot.
    '''
    UV_lambda  = 0.15
    K_UV       = (2.659*(-2.156 + 1.509/UV_lambda - 0.198/UV_lambda**2
                         + 0.011/UV_lambda**3)+ 4.05)

    Ha_lambda = 0.6563
    K_Ha      = (2.659*(-1.857 + 1.040/Ha_lambda) + 4.05)
    
    V_lambda  = 0.55
    K_V       = (2.659*(-2.156 + 1.509/V_lambda - 0.198/V_lambda**2
                        + 0.011/V_lambda**3)+ 4.05)

    
    tempfile = asc.read('Main_Sequence/Catalogs/EBV_both_HAHB_HGHB_no_overlap.txt',guess=False)
    tempfilenames = np.array(tempfile['name'])
    EBV_HaHB = np.array(tempfile['Egas'])

    filtnames = names[filt_match]
    index_match_a = np.array([x for x in range(len(filtnames)) if filtnames[x]
                              in tempfilenames]) #index of filtmatch
    index_match_b = np.array([x for x in range(len(tempfilenames)) if
                              tempfilenames[x] in filtnames]) #index of asc file
    print len(index_match_a), len(index_match_b)
    
    
    Estar1    = A_V[filt_match]/K_V
    corr_factor1 = 0.4*Estar1*K_UV

    Estar2    = A_V[filt_match]/K_V
    Estar2[index_match_a] = EBV_HaHB[index_match_b]*0.44
    Egas2 = Estar2/0.44
    corr_factor2 = 0.4*Egas2*K_Ha
    
    return corr_factor1, corr_factor2, index_match_a
#enddef

    
def plot_SFR_Ha_vs_UV():
    '''
    Plots the luminosity from the UV photometry (at 1500 angstroms) against
    the Ha luminosity (observed, but corrected for filter and nii/ha). There
    are four different subplots:
    * (a) L_nu vs. observed Ha luminosity
    * (b) L_nu+corr0 vs. Ha lumin+0.4dex (corrected for 1 mag. extinction)
    * (c) L_nu+(??) vs. Hopkins corrected Ha luminosity (N/A as of 170815)
    * (d) L_nu+corr1 vs. dust-corrected Ha luminosity+corr2

    Iterates by filter to plot the points in different colors. Good zspec
    sources are filled, while bad zspec sources are empty.

    The median line (obtained from yarr_values-xarr_values) is plotted in a
    dashed black.

    The plot is modified, with labels added before the figure is saved and
    closed.

    If using GALEX files, get_bad_FUV_NUV is called to find where there is
    no UV photometry (filter-dependent whether code looks at FUV or NUV).
    '''
    print '### plotting SFR_Ha vs. SFR_UV'
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
    f.subplots_adjust(hspace=0.05)
    f.subplots_adjust(wspace=0.05)
    f.set_size_inches(10, 10)
    for (ff, cc) in zip(filt_arr, color_arr):
        print ff
        filt_match = np.array([x for x in range(len(names)) if 'Ha-'+ff in
                               names[x]])
        filtz = ha_zspec[filt_match]
        lnu = get_lnu(filt_match)
        bad_UV, tempcolor = get_bad_FUV_NUV(ff, filt_match)

        goodz = np.array([x for x in range(len(filtz)) if filtz[x]>0 and
                          filtz[x] < 9])
        badz  = np.array([x for x in range(len(filtz)) if filtz[x]<=0 or
                          filtz[x] >= 9])
        ax1.scatter(lnu[goodz], nii_lumin[filt_match][goodz], facecolor=cc,
                    edgecolor='none', alpha=0.5)
        ax1.scatter(lnu[badz], nii_lumin[filt_match][badz], facecolor='none',
                    edgecolor=cc, linewidth=0.5, alpha=0.5)
        ax1.scatter(lnu[bad_UV], nii_lumin[filt_match][bad_UV], marker='x',
                    color=tempcolor,alpha=0.5,linewidth=0.5)

        corr0 = get_corr0()        
        ax2.scatter(lnu[goodz]+corr0, nii_lumin[filt_match][goodz]+0.4,
                    facecolor=cc, edgecolor='none', alpha=0.5)
        ax2.scatter(lnu[badz]+corr0, nii_lumin[filt_match][badz]+0.4,
                    facecolor='none', edgecolor=cc, linewidth=0.5, alpha=0.5)
        ax2.scatter(lnu[bad_UV]+corr0, nii_lumin[filt_match][bad_UV]+0.4,
                    marker='x', color=tempcolor,alpha=0.5,linewidth=0.5)
        
        corr1, corr2, indexa = get_corr12(filt_match)
        ax4.scatter(lnu[goodz]+corr1[goodz],
                    dust_lumin[filt_match][goodz]+corr2[goodz],
                    facecolor=cc, edgecolor='none', alpha=0.5)
        ax4.scatter(lnu[badz]+corr1[badz],
                    dust_lumin[filt_match][badz]+corr2[badz], facecolor='none',
                    edgecolor=cc, linewidth=0.5, alpha=0.5)
        ax4.scatter(lnu[bad_UV]+corr1[bad_UV],
                    dust_lumin[filt_match][bad_UV]+corr2[bad_UV],
                    marker='x',color=tempcolor,alpha=0.5,linewidth=0.5)

    #endfor
    plt.setp([a.minorticks_on() for a in f.axes[:]])
    plt.setp([a.set_xlim(23, 32) and a.set_ylim(37, 47) for a in f.axes[:]])

    f0 = plt.scatter(100,100,marker='x',color='teal',alpha=0.5,label='no FUV')
    n0 = plt.scatter(100,100,marker='x',color='pink',alpha=0.5,label='no NUV')

    ax1.text(min(ax1.get_xlim())+0.5, max(ax1.get_ylim())-0.5, '(a) Observed',
             fontsize=9, ha='left', va='top')
    ax2.text(min(ax2.get_xlim())+0.5, max(ax2.get_ylim())-0.5,
             '(b) Observed + 0.4 dex', fontsize=9, ha='left', va='top')
    ax3.text(min(ax3.get_xlim())+0.5, max(ax3.get_ylim())-0.5,
             '(c) Observed + H01 Ext.-Corr.', fontsize=9, ha='left', va='top')
    ax4.text(min(ax4.get_xlim())+0.5, max(ax4.get_ylim())-0.5,
             '(d) Observed + SED Ext.-Corr.', fontsize=9, ha='left', va='top')

    ax1.set_ylabel('log(H'+r'$\alpha$'+' Luminosity) [erg s'+r'$\^{-1}$'+']',
                   fontsize=9)
    ax3.set_ylabel('log(H'+r'$\alpha$'+' Luminosity) [erg s'+r'$\^{-1}$'+']',
                   fontsize=9)
    ax3.set_xlabel('log[L'+r'$_{\nu}$'+'(1500 '+r'$\AA$'+')]', fontsize=9)
    ax4.set_xlabel('log[L'+r'$_{\nu}$'+'(1500 '+r'$\AA$'+')]', fontsize=9)

    lnu = get_lnu(range(len(nii_lumin)))
    for (ax, num) in zip(f.axes[:], [1, 2, 3, 4]):
        print num
        if num==1:
            median = np.median(nii_lumin - lnu)
            stddev = np.std(nii_lumin - lnu)
        elif num==2:
            corr0 = get_corr0()
            median = np.median((nii_lumin+0.4) - (lnu+corr0))
            stddev = np.std((nii_lumin+0.4) - (lnu+corr0))
        elif num==4:
            corr1, corr2, indexa = get_corr12(range(len(lnu)))
            median = np.median((dust_lumin[indexa]+corr2[indexa]) -
                               (lnu[indexa]+corr1[indexa]))
            stddev = np.std((dust_lumin[indexa]+corr2[indexa]) -
                            (lnu[indexa]+corr1[indexa]))
        #endif
        if num != 3:
            line, = ax.plot([-100, 100], np.array([-100, 100])+median, 'k--',
                            linewidth=0.5, label='median='
                            +str(np.around(median, 3))+'\nstd dev='
                            +str(np.around(stddev, 3)))

            legend0 = ax.legend(handles=[line,f0,n0],fontsize=9,
                                loc='lower right',frameon=False,
                                numpoints=1)

            ax.add_artist(legend0)
        #endif
    #endfor
    
    plt.savefig('Plots/SFR_comparisons/SFR_Ha_vs_UV'+fileend+'.pdf')
    plt.close()
#enddef


def plot_just_balmer_decrements():
    '''
    Iterating by filter, plots only those sources that had a Balmer
    correction applied to it (different filter->different color). Good zspec
    sources are filled; bad zspec sources are empty.

    By subtracting the xarr values from the yarr values, the median is
    obtained; that line is then plotted (assuming a slope of 1) and the
    value of that and the standard deviation are in the lower right corner.

    The plot is then saved and closed.
    '''
    print '### plotting just_balmer_decrements'
    f, ax = plt.subplots()
    for (ff, cc) in zip(filt_arr, color_arr):
        filt_match = np.array([x for x in range(len(names)) if 'Ha-'+ff in
                               names[x]])
        corr1, corr2, index_match = get_corr12(filt_match)
        filtz = ha_zspec[filt_match][index_match]
        lnu = get_lnu(filt_match[index_match])
        
        goodz = np.array([x for x in range(len(filtz)) if filtz[x]>0 and
                          filtz[x] < 9])
        badz  = np.array([x for x in range(len(filtz)) if filtz[x]<=0 or
                          filtz[x] >= 9])

        try:
            ax.scatter(lnu[goodz]+corr1[index_match][goodz],
                        dust_lumin[filt_match[index_match[goodz]]]+
                        corr2[index_match][goodz], facecolor=cc,
                        edgecolor='none', alpha=0.5)
        except IndexError:
            pass
        try:
            ax.scatter(lnu[badz]+corr1[index_match][badz],
                        dust_lumin[filt_match[index_match[badz]]]+
                        corr2[index_match][badz], facecolor='none',
                        edgecolor=cc, linewidth=0.5, alpha=0.5)
        except IndexError:
            pass
    #endfor
    ax.minorticks_on()

    corr1, corr2, index_match = get_corr12(range(len(nii_lumin)))
    lnu = get_lnu(index_match)
    median = np.median((dust_lumin[index_match]+corr2[index_match]) -
                       (lnu+corr1[index_match]))
    stddev = np.std((dust_lumin[index_match]+corr2[index_match]) -
                    (lnu+corr1[index_match]))

    line, = ax.plot([-100, 100], np.array([-100, 100])+median, 'k--',
                     linewidth=0.5, label='median='
                     +str(np.around(median, 3))+'\nstd dev='
                     +str(np.around(stddev, 3)))
    legend0 = ax.legend(handles=[line],fontsize=9,loc='lower right',
                         frameon=False, numpoints=1)
    ax.add_artist(legend0)

    ax.set_xlim(23, 32)
    ax.set_ylim(37, 47)
    ax.set_xlabel('log[L'+r'$_{\nu}$'+'(1500 '+r'$\AA$'+')]')
    ax.set_ylabel('log(H'+r'$\alpha$'+' Luminosity) [erg s'+r'$\^{-1}$'+']')
    
    plt.savefig('Plots/SFR_comparisons/SFR_Ha_vs_UV_balmer_only'+fileend+'.pdf')
    plt.close()
#enddef


#----main body---------------------------------------------------------------#
# o Reads relevant inputs
# o Trims all relevant arrays so that, while maintaining the same indexing
#   as each other, they only contain ha emitters and 'good' sed_sfr fits.
# o Then calls plot_SFR_Ha_vs_SED, plot_SFR_Ha_vs_UV, and
#   plot_just_balmer_decrements
#----------------------------------------------------------------------------#
Ha_corrs = pyfits.open('Main_Sequence/Catalogs/mainseq_Ha_corrections'+fileend
                       +'.fits')
corrdata = Ha_corrs[1].data
names = corrdata['name']
ha_id = corrdata['ID']
ha_zspec = corrdata['zspec']
nii_lumin = corrdata['nii_ha_corr_lumin']
nii_sfr  = corrdata['nii_ha_corr_sfr']
dust_lumin = corrdata['dust_corr_lumin']
dust_sfr = corrdata['dust_corr_sfr']


catfile = asc.read('Catalogs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'
    +fileend+'.cat',guess=False,Reader=asc.CommentedHeader)
f_FUV = np.array(catfile['f_FUV'])
f_NUV = np.array(catfile['f_NUV'])

fout = asc.read('FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'
                +fileend+'.fout',guess=False,Reader=asc.NoHeader)
fout_ID = np.array(fout['col1'])
sed_sfr    = np.array(fout['col8'])
fout_zphot = np.array(fout['col2'])
A_V   = np.array(fout['col6'])
print '### done reading input files'

#getting just ha emitters
id_match = np.array([x for x in range(len(fout_ID)) if fout_ID[x] in ha_id])
sed_sfr = sed_sfr[id_match]
ha_zphot = fout_zphot[id_match]
A_V = A_V[id_match]
f_FUV = f_FUV[id_match]
f_NUV = f_NUV[id_match]


#getting rid of the 'bad' sed_sfr fits
bad_index = np.array([x for x in range(len(sed_sfr)) 
    if (sed_sfr[x]==-99. or sed_sfr[x]==-1. or sed_sfr[x] <=-5)])
bad_index = np.array(bad_index,dtype=np.int32)
names = np.delete(names, bad_index)
sed_sfr = np.delete(sed_sfr, bad_index)
nii_lumin = np.delete(nii_lumin, bad_index)
nii_sfr = np.delete(nii_sfr, bad_index)
dust_lumin = np.delete(dust_lumin, bad_index)
dust_sfr = np.delete(dust_sfr, bad_index)
ha_zspec = np.delete(ha_zspec, bad_index)
ha_id = np.delete(ha_id, bad_index)
ha_zphot = np.delete(ha_zphot, bad_index)
A_V = np.delete(A_V, bad_index)
f_FUV = np.delete(f_FUV, bad_index)
f_NUV = np.delete(f_NUV, bad_index)

filt_arr = ['NB704','NB711','NB816','NB921','NB973']
color_arr = ['red','orange','green','blue','purple']

plot_SFR_Ha_vs_SED()

plot_SFR_Ha_vs_UV()

plot_just_balmer_decrements()

Ha_corrs.close()
