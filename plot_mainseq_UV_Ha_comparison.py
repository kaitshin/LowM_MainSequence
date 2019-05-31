"""
NAME:
    plot_mainseq_UV_Ha_comparison.py

PURPOSE:
    This code plots a comparison between Halpha and UV luminosities (the
    latter of which is 'nu_L_nu'). Then, the ratio of the two is plotted as a
    function of stellar mass.
    With the GALEX command line option, if GALEX is typed, then GALEX files
    files used/output. Otherwise, the files without GALEX photometry are used.

INPUTS:
    'Catalogs/nb_ia_zspec.txt'
    'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+fileend+'.fout'
    'Main_Sequence/Catalogs/line_emission_ratios_table.dat'
    'Main_Sequence/Catalogs/mainseq_Ha_corrections'+fileend+'.fits'
    'FAST/outputs/BEST_FITS/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'
         +fileend+'_'+str(ID[ii])+'.fit'

CALLING SEQUENCE:
    main body -> get_nu_lnu -> get_flux
              -> make_scatter_plot, make_ratio_plot
              -> make_all_ratio_plot -> (make_all_ratio_legend,
                                         get_binned_stats)

OUTPUTS:
    'Plots/main_sequence_UV_Ha/'+ff+'_'+ltype+fileend+'.pdf'
    'Plots/main_sequence_UV_Ha/ratios/'+ff+'_'+ltype+fileend+'.pdf'
    'Plots/main_sequence_UV_Ha/ratios/all_filt_'+ltype+fileend+'.pdf'
    
REVISION HISTORY:
    Created by Kaitlyn Shin 13 August 2015
"""

import numpy as np, astropy.units as u, matplotlib.pyplot as plt, sys
from scipy import interpolate
from astropy import constants
from astropy.io import fits as pyfits, ascii as asc
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0 = 70 * u.km / u.s / u.Mpc, Om0=0.3)

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'


def get_flux(ID, lambda_arr):
    '''
    Reads in the relevant SED spectrum file and then interpolates the
    function to obtain a flux, the array of which is then returned.
    '''
    newflux = np.zeros(len(ID))
    for ii in range(len(ID)):
        tempfile = asc.read(FULL_PATH+'FAST/outputs/BEST_FITS/NB_IA_emitters\
            _allphot.emagcorr.ACpsf_fast'+fileend+'_'+str(ID[ii])+'.fit',
            guess=False,Reader=asc.NoHeader)
        wavelength = np.array(tempfile['col1'])
        flux = np.array(tempfile['col2'])
        f = interpolate.interp1d(wavelength, flux)
        newflux[ii] = f(lambda_arr[ii])

    return newflux


def get_nu_lnu(filt_index):
    '''
    Calls get_flux with an array of redshifted wavelengths in order to get
    the corresponding flux values. Those f_lambda values are then converted
    into f_nu values, which is in turn converted into L_nu and multiplied by
    nu, the log of which is returned as nu_lnu.
    '''
    ID    = ID0[filt_index]
    zspec = zspec0[filt_index]
    zphot = zphot0[filt_index]
    goodz = np.array([x for x in range(len(zspec)) if zspec[x] < 9. and
                      zspec[x] > 0.])
    badz  = np.array([x for x in range(len(zspec)) if zspec[x] >= 9. or
                      zspec[x] <= 0.])
    tempz = np.zeros(len(filt_index))
    tempz[goodz] = zspec[goodz]
    tempz[badz]  = zphot[badz]

    lambda_arr = (1+tempz)*1500

    f_lambda = get_flux(ID, lambda_arr)
    f_nu = f_lambda*(1E-19*(lambda_arr**2*1E-10)/(constants.c.value))
    L_nu = f_nu*4*np.pi*(cosmo.luminosity_distance(tempz).to(u.cm).value)**2
    return np.log10(L_nu*((constants.c.value)/1.5E-7))


def make_scatter_plot(filt_index, nu_lnu, l_ha, ff, ltype):
    '''
    Makes a scatter plot (by filter) of nu_lnu vs l_ha, then saves and
    closes the figure.

    There is a value in the filter NB921 which has flux=='NAN'. That is
    ignored.
    '''
    if ff=='NB921':
        zero = np.where(l_ha == 0.)[0]
        nu_lnu = np.concatenate((nu_lnu[:zero], nu_lnu[zero+1:]))
        l_ha = np.concatenate((l_ha[:zero], l_ha[zero+1:]))

    plt.scatter(nu_lnu, l_ha)
    plt.gca().minorticks_on()
    plt.xlabel('log['+r'$\nu$'+'L'+r'$_{\nu}$'+'(1500 '+r'$\AA$'+')]')
    plt.ylabel('log[L'+r'$_{H\alpha}$'+']')    
    plt.xlim(36.0, 48.0)
    plt.ylim(37.0, 44.0)
    plt.savefig(FULL_PATH+'Plots/main_sequence_UV_Ha/'+ff+'_'+ltype+
        fileend+'.pdf')
    plt.close()


def make_ratio_plot(filt_index, nu_lnu, l_ha, stlr, ff, ltype):
    '''
    Makes a ratio plot (by filter) of stellar mass vs. (nu_lnu/l_ha), then
    saves and closes the figure.

    There is a value in the filter NB921 which has flux=='NAN'. That is
    ignored.
    '''
    if ff=='NB921':
        zero = np.where(l_ha == 0.)[0]
        nu_lnu = np.concatenate((nu_lnu[:zero], nu_lnu[zero+1:]))
        l_ha = np.concatenate((l_ha[:zero], l_ha[zero+1:]))
        stlr = np.concatenate((stlr[:zero], stlr[zero+1:]))

    ratio = nu_lnu-l_ha
    plt.scatter(stlr, ratio)
    plt.gca().minorticks_on()
    plt.xlabel('log[M/M'+r'$_{\odot}$'+']')
    plt.ylabel('log['+r'$\nu$'+'L'+r'$_{\nu}$'+'/L(H'+r'$\alpha$'+')'+']')
    plt.savefig(FULL_PATH+'Plots/main_sequence_UV_Ha/ratios/'+ff+'_'+ltype+
        fileend+'.pdf')
    plt.close()


def make_all_ratio_legend(filtlabel):
    '''
    Makes a legend for the plot with all ratios (as a function of mass) of
    all the filters on the same plot.

    red='NB704', orange='NB711', green='NB816', blue='NB921', purple='NB973'
    '''
    import matplotlib.patches as mpatches

    red_patch = mpatches.Patch(color='red', label='H'+r'$\alpha$'+'-NB704 '
                               +filtlabel['NB704'], alpha=0.5)
    orange_patch = mpatches.Patch(color='orange', label='H'+r'$\alpha$'
                                  +'-NB711 '+filtlabel['NB711'], alpha=0.5)
    green_patch = mpatches.Patch(color='green', label='H'+r'$\alpha$'+'-NB816 '
                                 +filtlabel['NB816'], alpha=0.5)
    blue_patch = mpatches.Patch(color='blue', label='H'+r'$\alpha$'+'-NB921 '
                                +filtlabel['NB921'], alpha=0.5)
    purple_patch = mpatches.Patch(color='purple', label='H'+r'$\alpha$'
                                  +'-NB973 '+filtlabel['NB973'], alpha=0.5)
    legend0 = plt.legend(handles=[red_patch,orange_patch,green_patch,
                                  blue_patch,purple_patch],fontsize=9,
                         loc='lower right', frameon=False)
    plt.gca().add_artist(legend0)


def get_binned_stats(xposdata, yposdata):
    '''
    From the min to the max x-values of the graph, 'bins' of interval 0.25
    are created. If there are more than 3 x values in a particular bin, the
    average of the corresponding y values are plotted, and their std dev
    values are plotted as error bars as well.

    In order to clearly stand out from the rainbow scatter plot points, these
    'binned' points are black.
    '''
    bins = np.arange(min(plt.xlim()), max(plt.xlim())+0.25, 0.25)
    for bb in range(len(bins)-1):
        valid_index = np.array([x for x in range(len(xposdata)) if
                                xposdata[x] >= bins[bb] and xposdata[x] <
                                bins[bb+1]])
        if len(valid_index) > 3:
            xpos = np.mean((bins[bb], bins[bb+1]))
            ypos = np.mean(yposdata[valid_index])
            yerr = np.std(yposdata[valid_index])
            plt.scatter(xpos, ypos, facecolor='k', edgecolor='none', alpha=0.7)
            plt.errorbar(xpos, ypos, yerr=yerr, ecolor='k', alpha=0.7,
                         fmt='none')


def make_all_ratio_plot(L_ha, ltype):
    '''
    Similar as make_ratio_plot, except each filter is plotted on the graph,
    and sources with good zspec are filled points while those w/o good zspec
    are plotted as empty points. get_binned_stats and make_all_ratio_legend
    are called, before the plot is duly modified, saved, and closed.
    '''
    print ltype
    xposdata = np.array([])
    yposdata = np.array([])
    filtlabel = {}
    for (ff, cc) in zip(['NB704','NB711','NB816','NB921','NB973'], color_arr):
        print ff
        filt_index = np.array([x for x in range(len(names0)) if 'Ha-'+ff in
                               names0[x]])
        filt_index = np.array(filt_index,dtype=np.int32)
        
        zspec = zspec0[filt_index]
        stlr = stlr0[filt_index]
        nu_lnu = get_nu_lnu(filt_index)
        l_ha = L_ha[filt_index]
        if ff=='NB921':
            zero_index = np.where(l_ha == 0.)[0]
            l_ha = np.delete(l_ha, zero_index)
            zspec = np.delete(zspec, zero_index)
            stlr = np.delete(stlr, zero_index)
            nu_lnu = np.delete(nu_lnu, zero_index)
        
        ratio = nu_lnu-l_ha
        xposdata = np.append(xposdata, stlr)
        yposdata = np.append(yposdata, ratio)

        good_z = np.array([x for x in range(len(zspec)) if zspec[x] > 0. and
                           zspec[x] < 9.])
        bad_z  = np.array([x for x in range(len(zspec)) if zspec[x] <= 0. or
                           zspec[x] >= 9.])
        filtlabel[ff] = '('+str(len(good_z))+', '+str(len(bad_z))+')'

        plt.scatter(stlr[good_z], ratio[good_z], facecolor=cc, edgecolor='none',
                    alpha=0.5)
        plt.scatter(stlr[bad_z], ratio[bad_z], facecolor='none', edgecolor=cc,
                    linewidth=0.5, alpha=0.5)


    get_binned_stats(xposdata, yposdata)
    plt.gca().minorticks_on()
    plt.xlabel('log[M/M'+r'$_{\odot}$'+']')
    plt.ylabel('log['+r'$\nu$'+'L'+r'$_{\nu}$'+'(1500 '+r'$\AA$'+')/L'
               +r'$_{H\alpha}$'+']')
    plt.xlim(4, 11)
    plt.ylim(-2.5, 4)
    make_all_ratio_legend(filtlabel)
    plt.plot(plt.xlim(), [2.05, 2.05], 'k--', alpha=0.3, linewidth=3.0)
    plt.savefig(FULL_PATH+'Plots/main_sequence_UV_Ha/ratios/all_filt_'+ltype+
                fileend+'.pdf')
    plt.close()


#----main body---------------------------------------------------------------#
# o Reads relevant inputs
# o Iterating by filter, calls nu_lnu, make_scatter_plot, and
#   make_ratio_plot
# o After the filter iteration, make_all_ratio_plot is called.
# o For each of the functions to make a plot, they're called twice - once for
#   plotting the nii/ha corrected version, and one for plotting the dust
#   corrected version.
#----------------------------------------------------------------------------#
# +190531: only GALEX files will be used
fileend='.GALEX'

zspec_file = asc.read(FULL_PATH+'Catalogs/nb_ia_zspec.txt',guess=False,
                      Reader=asc.CommentedHeader)
ID0    = np.array(zspec_file['ID0'])
zspec0 = np.array(zspec_file['zspec0'])

fout  = asc.read(FULL_PATH+'FAST/outputs/NB_IA_emitters_allphot.emagcorr\
    .ACpsf_fast'+fileend+'.fout',guess=False,Reader=asc.NoHeader)
zphot0 = np.array(fout['col2'])
stlr0 = np.array(fout['col7'])

NIIB_Ha_ratios = asc.read(FULL_PATH+'Main_Sequence/Catalogs/line_emission_\
    ratios_table.dat',guess=False,Reader=asc.CommentedHeader)
names0 = np.array(NIIB_Ha_ratios['NAME'])

Ha_corrs = pyfits.open(FULL_PATH+'Main_Sequence/Catalogs/mainseq_Ha_\
    corrections'+fileend+'.fits')
corrdata = Ha_corrs[1].data
corrID = corrdata['ID']
print '### done reading input files'

nii_ha_corr_lumin = np.zeros(len(ID0))
dust_corr_lumin = np.zeros(len(ID0))
ID_match = np.array([x for x in range(len(ID0)) if ID0[x] in corrID])
nii_ha_corr_lumin[ID_match] = corrdata['nii_ha_corr_lumin']
dust_corr_lumin[ID_match] = corrdata['dust_corr_lumin']
color_arr = ['r', 'orange', 'g', 'b', 'purple']

print '### making scatter_plots and ratio_plots'
for (ff, cc) in zip(['NB704','NB711','NB816','NB921','NB973'], color_arr):
    print ff
    filt_index = np.array([x for x in range(len(names0)) if 'Ha-'+ff in
                           names0[x]])
    filt_index = np.array(filt_index,dtype=np.int32)

    nu_lnu = get_nu_lnu(filt_index)
    
    make_scatter_plot(filt_index, nu_lnu, nii_ha_corr_lumin[filt_index], ff,
                      'nii_ha_corr')
    make_scatter_plot(filt_index, nu_lnu, dust_corr_lumin[filt_index], ff,
                      'dust_corr')
    make_ratio_plot(filt_index, nu_lnu, nii_ha_corr_lumin[filt_index],
                    stlr0[filt_index], ff, 'nii_ha_corr')
    make_ratio_plot(filt_index, nu_lnu, dust_corr_lumin[filt_index],
                    stlr0[filt_index], ff, 'dust_corr')

print '### making all_ratio_plots'
make_all_ratio_plot(nii_ha_corr_lumin, 'nii_ha_corr')
make_all_ratio_plot(dust_corr_lumin, 'dust_corr')

Ha_corrs.close()