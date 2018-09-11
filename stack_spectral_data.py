"""
NAME:
    stack_spectral_data.py

PURPOSE:
    This code creates a PDF file with 15 subplots, filter-emission line
    row-major order, to show all the MMT and Keck spectral data stacked and
    plotted in a 'de-redshifted' frame.
    Specific to SDF data.

    Depends on combine_spectral_data.py and create_ordered_AP_arrays.py

INPUTS:
    'Catalogs/NB_IA_emitters.nodup.colorrev.fix.fits'
    'Catalogs/nb_ia_zspec.txt'
    'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.fout'
    'Spectra/spectral_MMT_grid_data.txt'
    'Spectra/spectral_MMT_grid.fits'
    'Spectra/spectral_Keck_grid_data.txt'
    'Spectra/spectral_Keck_grid.fits'

OUTPUTS:
    'Composite_Spectra/Redshift/MMT_spectra_vals/'+subtitle+'.txt''
    'Composite_Spectra/Redshift/MMT_stacked_spectra_data.txt'
    'Composite_Spectra/StellarMass/MMT_spectra_vals/'+subtitle[10:]+'.txt'
    'Composite_Spectra/StellarMass/MMT_all_five_data.txt'
    'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt'
    'Composite_Spectra/Redshift/Keck_spectra_vals/'+subtitle+'.txt'
    'Composite_Spectra/Redshift/Keck_stacked_spectra_data.txt'
    'Composite_Spectra/StellarMass/Keck_spectra_vals/'+subtitle[10:]+'.txt'
    'Composite_Spectra/StellarMass/Keck_all_five_data.txt'
    'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt'
    'Composite_Spectra/Redshift/MMT_stacked_spectra.pdf'
    'Composite_Spectra/StellarMass/MMT_all_five.pdf'
    'Composite_Spectra/StellarMassZ/MMT_stlrmassZ.pdf'
    'Composite_Spectra/Redshift/Keck_stacked_spectra.pdf'
    'Composite_Spectra/StellarMass/Keck_all_five.pdf'
    'Composite_Spectra/StellarMassZ/Keck_stlrmassZ.pdf'
"""

import numpy as np, numpy.ma as ma, matplotlib.pyplot as plt
import plotting.hg_hb_ha_plotting as MMT_plotting
import plotting.hb_ha_plotting as Keck_plotting
import plotting.general_plotting as general_plotting
import writing_tables.hg_hb_ha_tables as MMT_twriting
import writing_tables.hb_ha_tables as Keck_twriting
import writing_tables.general_tables as general_twriting
from analysis.balmer_fit import get_best_fit3
from analysis.sdf_stack_data import stack_data
from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table, vstack
from create_ordered_AP_arrays import create_ordered_AP_arrays
from matplotlib.backends.backend_pdf import PdfPages
from analysis.cardelli import *   # k = cardelli(lambda0, R=3.1)
from astropy import units as u

MIN_NUM_PER_BIN = 10
MAX_NUM_OF_BINS = 5

def correct_instr_AP(indexed_AP, indexed_inst_str0, instr):
    '''
    Returns the indexed AP_match array based on the 'matctot = 3h_index' from
    plot_MMT/Keck_Ha
    '''
    for ii in range(len(indexed_inst_str0)):
        if instr == 'MMT': # indexed_inst_str0[ii]=='MMT,' is fine
            if (indexed_inst_str0[ii]=='merged,FOCAS,' or indexed_inst_str0[ii] == 'MMT,FOCAS,' 
                or indexed_inst_str0[ii] == 'MMT,Keck,' or indexed_inst_str0[ii] == 'merged,'):
                indexed_AP[ii] = indexed_AP[ii][:5]
        elif instr=='Keck': # indexed_inst_str0[ii]=='Keck,' is fine
            if (indexed_inst_str0[ii] == 'merged,' or indexed_inst_str0[ii]=='merged,FOCAS,'):
                indexed_AP[ii] = indexed_AP[ii][6:]
        #endif
    #endfor
    return indexed_AP
#enddef

def HG_HB_EBV(hg, hb):
    '''
    instr is always MMT
    '''
    hg = np.array(hg)
    hb = np.array(hb)
    hghb = np.array([0.468 if x > 0.468 else x for x in hg/hb])
    EBV_hghb = np.log10((hghb)/0.468)/(-0.4*(k_hg - k_hb))
    EBV_hghb = np.array([-99.0 if np.isnan(x) else x for x in EBV_hghb])
    return EBV_hghb
#enddef

def get_HB_NB921_flux(bintype='redshift'):
    '''
    '''
    cvg = asc.read(full_path+'Composite_Spectra/MMT_spectral_coverage.txt')
    nb921 = np.array([x for x in range(len(cvg)) if cvg['filter'][x]=='NB921' and cvg['HB_cvg'][x]=='YES'])
    nb921_ha = np.array([x for x in range(len(nb921)) if cvg['HA_cvg'][nb921][x] == 'YES'])

    # print '>>>>>>>HB IDs:', cvg['ID'][nb921].data ##PRINT STATEMENT

    if bintype=='redshift':
        flux_arr = np.array([-99.0])
    else:
        flux_arr = np.array([-99.0]*5)

    for i in range(len(flux_arr)):
        # i = index of array (0-indexed)
        i0 = np.array([])
        if bintype == 'StellarMassZ':
            # i+1 = index of bin (1-indexed)
            bin_i = np.array([x for x in range(len(nb921_ha)) if str(i+1)+'-' in cvg['stlrmassZbin'][nb921[nb921_ha]].data[x]])
            if len(bin_i) < 2:
                continue
            i0 = np.array([x for x in range(len(gridap)) if gridap[x] in cvg['AP'][nb921[nb921_ha[bin_i]]]])
        elif bintype == 'StlrMass':
            bin_i = np.array([x for x in range(len(nb921_ha)) if i+1==cvg['stlrmassbin'][nb921[nb921_ha]].data[x]])
            if len(bin_i) < 2:
                continue
            i0 = np.array([x for x in range(len(gridap)) if gridap[x] in cvg['AP'][nb921[nb921_ha[bin_i]]]])
        else:
            i0 = np.array([x for x in range(len(gridap)) if gridap[x] in cvg['AP'][nb921[nb921_ha]]])
            # print '>>>>>>>GRIDAP INDICES:', i0 ##PRINT STATEMENT
            # print '>>>>>>>NAMES:', cvg['NAME'][nb921[nb921_ha]].data ##PRINT STATEMENT
            # print '>>>>>>>IDs:', cvg['ID'][nb921[nb921_ha]].data ##PRINT STATEMENT

        # print '>>>>>>>LENGTH OF HB_NB921_flux ARRAY:', len(i0) ##PRINT STATEMENT
        # print '>>>>>>>APs:', gridap[i0] ##PRINT STATEMENT
        
        zs = np.array(gridz[i0])
        good_z2 = np.where((zs >= 0.385) & (zs <= 0.429))[0]
        zs = np.average(zs[good_z2])
        dlambda = (x0[1]-x0[0])/(1+zs)

        xval, yval, len_input_index, stacked_indexes, avgz, minz, maxz = stack_data(grid_ndarr, gridz, i0,
            x0, 3700, 6700, dlambda, ff='NB921', instr='MMT')

        # calculating flux for subtitle=='NB921' emissions
        xmin0 = 4801
        xmax0 = 4921

        good_ii = np.array([x for x in range(len(xval)) if xval[x] >= xmin0 and xval[x] <= xmax0])
        xval = xval[good_ii]
        yval = yval[good_ii]

        good_ii = [ii for ii in range(len(yval)) if not np.isnan(yval[ii])] # not NaN
        xval = xval[good_ii]
        yval = yval[good_ii]

        o1 = get_best_fit3(xval, yval, r'H$\beta$')
        pos0 = o1[6]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)
        neg0 = o1[3]*np.exp(-0.5*((xval-o1[4])/o1[5])**2)
        idx_small = np.where(np.absolute(xval - o1[1]) <= 2.5*o1[2])[0]
        flux = np.sum(dlambda * (pos0[idx_small] - o1[6] - neg0[idx_small]))
        
        flux_arr[i] = flux

    return flux_arr
#enddef

def HA_HB_EBV(ha, hb, instr, bintype='redshift', filt='N/A'):
    '''
    '''
    ha = np.array(ha)
    hb = np.array(hb)

    hahb = np.array([2.86 if (x < 2.86 and x > 0) else x for x in ha/hb])
    EBV_hahb = np.log10((hahb)/2.86)/(-0.4*(k_ha - k_hb))

    if instr=='MMT' and bintype=='redshift':
        EBV_hahb[-1] = -99.0 #no nb973 halpha
    elif instr=='MMT' and bintype=='StellarMassZ' and filt=='NB973':
        EBV_hahb[:] = -99.0 #no nb973 halpha
    elif instr=='Keck' and bintype=='redshift':
        EBV_hahb[0] = -99.0 #no nb816 hbeta

    EBV_hahb = np.array([-99.0 if np.isnan(x) else x for x in EBV_hahb])

    return EBV_hahb
#enddef

def split_into_bins(masses, n):
    '''
    '''
    perc_arrs = np.array([np.percentile(masses, x) for x in np.arange(0, 100+100.0/n, 100.0/n)[1:]])
    
    index_arrs = []
    for i in range(n):
        if i == 0:
            index = [x for x in range(len(masses)) if (masses[x]>0 and masses[x]<=perc_arrs[i])]
        else:
            index = [x for x in range(len(masses)) if (masses[x]>perc_arrs[i-1] and masses[x]<=perc_arrs[i])]
        index_arrs.append(index)
    
    if min([len(i) for i in index_arrs]) < MIN_NUM_PER_BIN and n != 2:
        return 'TOO SMALL'
    else:
        return index_arrs
#enddef

def plot_MMT_Ha():
    '''
    Creates a pdf (8"x11") with 5x3 subplots for different lines and filter
    combinations.

    Then, the code iterates through every subplot in row-major filter-line
    order. Using only the 'good' indexes, finds 'match_index'. With those
    indexes of AP and inst_str0, calls AP_match.

    For NB921 Halpha, does a cross-match to ensure no 'invalid' point is
    plotted.

    Except for NB973 Halpha, the graph is 'de-redshifted' in order to have
    the spectral line appear in the subplot. The values to plot are called
    from sdf_stack_data.stack_data

    get_best_fit is called to obtain the best-fit spectra, overlay the
    best fit, and then calculate the flux

    Additionally, a line is plotted at the value at which the emission line
    should theoretically be (based on which emission line it is).

    The yscale is fixed for each filter type (usually the yscale values of
    the Halpha subplot).

    Minor ticks are set on, lines and filters are labeled, and with the
    line label is another label for the number of stacked sources that were
    used to produce the emission graph.

    At the end of all the iterations, the plot is saved and closed.
    
    The fluxes are also output to a separate .txt file.
    '''
    print '>MMT REDSHIFT STACKING'
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    (HG_flux, HB_flux, HA_flux, NII_6548_flux, NII_6583_flux,
        HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr,
        HG_EW_abs, HB_EW_abs, HG_continuum, HB_continuum, HA_continuum,
        HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude) = table_arrays
    (HB_NB921_flux, num_sources, num_stack_HG, num_stack_HB, num_stack_HA, avgz_arr, minz_arr, maxz_arr,
        IDs_arr) = ([], [], [], [], [], [], [], [], [])
    index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'MMT')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('MMT')
    EBV_hahb = np.array([])

    f, axarr = plt.subplots(5, 3)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)
    
    subplot_index=0
    # this for-loop stacks by filter
    for (match_index, subtitle) in zip(index_list, subtitle_list):
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)

        zs = np.array(gridz[input_index])
        if subtitle=='NB704' or subtitle=='NB711':
            good_z = np.where((zs >= 0.05) & (zs <= 0.1))[0]
        elif subtitle=='NB816':
            good_z = np.where((zs >= 0.21) & (zs <= 0.26))[0]
        elif subtitle=='NB921':
            good_z = np.where((zs >= 0.385) & (zs <= 0.429))[0]
        else:
            good_z = np.where((zs >= 0.45) & (zs <= 0.52))[0]
        #endif
        zs = np.average(zs[good_z])
        dlambda = (x0[1]-x0[0])/(1+zs)

        if len(input_index) < 2: 
            print 'Not enough sources to stack (less than two)'
            [arr.append(0) for arr in table_arrays]
            HB_NB921_flux.append(0)
            num_sources.append(0)
            num_stack_HG.append(0)
            num_stack_HB.append(0)
            num_stack_HA.append(0)
            avgz_arr.append(0)
            minz_arr.append(0)
            maxz_arr.append(0)
            IDs_arr.append('N/A')
            for i in range(3):
                ax = ax_list[subplot_index]
                label = label_list[i]
                MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
                subplot_index += 1
            continue
        #endif

        try:
            xval, yval, len_input_index, stacked_indexes, avgz, minz, maxz = stack_data(grid_ndarr, gridz, input_index,
                x0, 3700, 6700, dlambda, ff=subtitle, instr='MMT')
        except AttributeError:
            print 'Not enough sources to stack (less than two)'
            [arr.append(0) for arr in table_arrays]
            num_sources.append(0)
            num_stack_HG.append(0)
            num_stack_HB.append(0)
            num_stack_HA.append(0)
            avgz_arr.append(0)
            minz_arr.append(0)
            maxz_arr.append(0)
            IDs_arr.append('N/A')
            for i in range(3):
                ax = ax_list[subplot_index]
                label = label_list[i]
                MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
                subplot_index += 1
            continue
        #endtry

        num_sources.append(len_input_index[0])
        avgz_arr.append(avgz)
        minz_arr.append(minz)
        maxz_arr.append(maxz)

        # appending to the ID columns
        # print '#####STACKED INDEXES???', stacked_indexes ##PRINT STATEMENT
        mm0 = [x for x in range(len(AP)) if any(y in AP[x][:5] for y in gridap[stacked_indexes])] # gridap ordering -> NBIA ordering
        IDs_arr.append(','.join(NAME0[mm0]))

        # writing the spectra table
        table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
        spectra_file_path = full_path+'Composite_Spectra/Redshift/MMT_spectra_vals/'+subtitle+'.txt'
        asc.write(table0, spectra_file_path, format='fixed_width', delimiter=' ')

        # calculating flux for NII emissions
        pos_flux_list = []
        flux_list = []
        pos_amplitude_list = []
        neg_amplitude_list = []
        pos_sigma_list = []
        neg_sigma_list = []
        median_list = []
        for i in range(3):
            xmin0 = xmin_list[i]
            xmax0 = xmax_list[i]
            ax = ax_list[subplot_index+i]
            label = label_list[i]
            len_ii = len_input_index[i]

            try:
                ax, flux, flux2, flux3, pos_flux, o1 = MMT_plotting.subplots_plotting(
                    ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol, len_ii=len_ii)
                pos_flux_list.append(pos_flux)
                flux_list.append(flux)
            except IndexError:
                print len_ii, 'IndexError (too few sources?)'
                continue
            finally:
                (ew, ew_emission, ew_absorption, median, pos_amplitude, 
                    neg_amplitude) = MMT_twriting.Hg_Hb_Ha_tables(label, flux, 
                    o1, xval, pos_flux, dlambda)
                table_arrays = general_twriting.table_arr_appends(i, subtitle,
                    table_arrays, flux, flux2, flux3, ew, ew_emission, ew_absorption, 
                    median, pos_amplitude, neg_amplitude, 'MMT', len_ii=len_ii)
                if len_ii > 2 and (not (subtitle=='NB973' and i==2)):
                    pos_amplitude_list.append(pos_amplitude)
                    neg_amplitude_list.append(neg_amplitude)
                    median_list.append(median)
                    pos_sigma_list.append(o1[2])
                    if i==2:
                        neg_sigma_list.append(0)
                    else:
                        neg_sigma_list.append(o1[5])
                else:
                    pos_amplitude_list.append(0)
                    neg_amplitude_list.append(0)
                    pos_sigma_list.append(0)
                    neg_sigma_list.append(0)
                    median_list.append(0)
                #endif
        #endfor
        
        for i in range(3):
            label = label_list[i] + ' ('+str(len_input_index[i])+')'
            if i == 0:
                num_stack_HG.append(int(len_input_index[i]))
            elif i == 1:
                num_stack_HB.append(int(len_input_index[i]))
            else: # i == 2:
                num_stack_HA.append(int(len_input_index[i]))

            ax = ax_list[subplot_index]
            try:
                pos_flux = pos_flux_list[i]
                flux = flux_list[i]
                if not (subtitle=='NB973' and i==2):
                    pos_amplitude = pos_amplitude_list[i]
                    neg_amplitude = neg_amplitude_list[i]
                    pos_sigma = pos_sigma_list[i]
                    neg_sigma = neg_sigma_list[i]
                    median = median_list[i]
                    ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, pos_flux, flux,
                        pos_amplitude, neg_amplitude, pos_sigma, neg_sigma, median)
                else:
                    ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, pos_flux, flux)
            except IndexError: # assuming there's no pos_flux or flux value
                ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
            subplot_index+=1
        #endfor 
    #endfor

    f = general_plotting.final_plot_setup(f, r'MMT detections of H$\alpha$ emitters')
    plt.savefig(full_path+'Composite_Spectra/Redshift/MMT_stacked_spectra.pdf')
    plt.close()

    EBV_hghb = HG_HB_EBV(HG_flux, HB_flux)
    
    HB_NB921_flux = np.copy(HB_flux)
    HB_NB921_flux[-2] = get_HB_NB921_flux()
    EBV_hahb = HA_HB_EBV(HA_flux, HB_NB921_flux, 'MMT')

    table00 = Table([subtitle_list, num_sources, num_stack_HG, num_stack_HB, num_stack_HA,
        avgz_arr, minz_arr, maxz_arr, 
        HG_flux, HB_flux, HB_NB921_flux, HA_flux, NII_6548_flux, 
        NII_6583_flux, HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr, HG_EW_abs, HB_EW_abs,
        HG_continuum, HB_continuum, HA_continuum, HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude, EBV_hghb, EBV_hahb], # IDs_arr
        names=['filter', 'num_sources', 'num_stack_HG', 'num_stack_HB', 'num_stack_HA',
        'avgz', 'minz', 'maxz',
        'HG_flux', 'HB_flux', 'HB_NB921_flux', 'HA_flux', 'NII_6548_flux', 
        'NII_6583_flux', 'HG_EW', 'HB_EW', 'HA_EW', 'HG_EW_corr', 'HB_EW_corr', 'HA_EW_corr', 'HG_EW_abs', 'HB_EW_abs',
        'HG_continuum', 'HB_continuum', 'HA_continuum', 'HG_pos_amplitude', 'HB_pos_amplitude', 'HA_pos_amplitude',
        'HG_neg_amplitude', 'HB_neg_amplitude', 'E(B-V)_hghb', 'E(B-V)_hahb']) # IDs

    asc.write(table00, full_path+'Composite_Spectra/Redshift/MMT_stacked_spectra_data.txt',
        format='fixed_width_two_line', delimiter=' ')
#enddef

def plot_MMT_Ha_stlrmass(index_list=[], pp=None, title='', bintype='StlrMass'):
    '''
    TODO(document)
    TODO(implement flexible stellar mass bin-readings)
    TODO(implement flexible file-naming)
        (nothing from the command line -- default into 5 bins by percentile)
        (number n from the command line -- make n bins by percentile)
        (file name from the command line -- flag to read the stellar mass bins from that ASCII file)
    TODO(get rid of assumption that there's only one page)
    '''
    if index_list == []:
        print '>MMT STELLARMASS STACKING'
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    (HG_flux, HB_flux, HA_flux, NII_6548_flux, NII_6583_flux,
        HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr,
        HG_EW_abs, HB_EW_abs, HG_continuum, HB_continuum, HA_continuum,
        HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude) = table_arrays
    (num_sources, num_stack_HG, num_stack_HB, num_stack_HA, avgz_arr, minz_arr, maxz_arr,
        stlrmass_bin_arr, avg_stlrmass_arr, min_stlrmass_arr, max_stlrmass_arr,
        IDs_arr) = ([], [], [], [], [], [], [], [], [], [], [], [])
    if index_list == []:
        index_list = general_plotting.get_index_list2(NAME0, stlr_mass, inst_str0, zspec0, inst_dict, 'MMT')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('MMT')

    f, axarr = plt.subplots(5, 3)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)

    subplot_index=0
    # this for-loop stacks by stlr mass
    for (match_index) in (index_list):
        if len(match_index) == 0:
            print 'this is an empty bin'
            [arr.append(0) for arr in table_arrays]
            num_sources.append(0)
            num_stack_HG.append(0)
            num_stack_HB.append(0)
            num_stack_HA.append(0)
            avgz_arr.append(0)
            minz_arr.append(0)
            maxz_arr.append(0)
            IDs_arr.append('N/A')
            stlrmass_bin_arr.append('N/A')
            avg_stlrmass_arr.append(0)
            min_stlrmass_arr.append(0)
            max_stlrmass_arr.append(0)
            for i in range(2):
                ax = ax_list[subplot_index]
                label = label_list[i]
                Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
                subplot_index += 1
            continue
        #endif

        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)

        subtitle='stlrmass: '+str(min(stlr_mass[match_index]))+'-'+str(max(stlr_mass[match_index]))
        print '>>>', subtitle
        avg_stlrmass_arr.append(np.mean(stlr_mass[match_index]))
        min_stlrmass_arr.append(np.min(stlr_mass[match_index]))
        max_stlrmass_arr.append(np.max(stlr_mass[match_index]))

        dlambda = 0.1 # xval[1] - xval[0]

        xval, yval, len_input_index, stacked_indexes, avgz, minz, maxz = stack_data(grid_ndarr, gridz, input_index,
            x0, 3700, 6700, dlambda, instr='MMT')
        num_sources.append(len_input_index[0])
        avgz_arr.append(avgz)
        minz_arr.append(minz)
        maxz_arr.append(maxz)
        stlrmass_bin_arr.append(subtitle[10:])

        # appending to the ID columns
        mm0 = [x for x in range(len(AP)) if any(y in AP[x][:5] for y in gridap[stacked_indexes])] # gridap ordering -> NBIA ordering
        IDs_arr.append(','.join(NAME0[mm0]))

        # writing the spectra table
        table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
        spectra_file_path = full_path+'Composite_Spectra/StellarMass/MMT_spectra_vals/'+subtitle[10:]+'.txt'
        asc.write(table0, spectra_file_path,
            format='fixed_width', delimiter=' ')

        # calculating flux for NII emissions
        pos_flux_list = []
        flux_list = []
        pos_amplitude_list = []
        neg_amplitude_list = []
        pos_sigma_list = []
        neg_sigma_list = []
        median_list = []
        for i in range(3):
            xmin0 = xmin_list[i]
            xmax0 = xmax_list[i]
            ax = ax_list[subplot_index+i]
            label = label_list[i]
            try:
                ax, flux, flux2, flux3, pos_flux, o1 = MMT_plotting.subplots_plotting(
                    ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol, len_input_index[i])
                pos_flux_list.append(pos_flux)
                flux_list.append(flux)
            except IndexError:
                print 'There\'s an exception'
                continue
            finally:
                (ew, ew_emission, ew_absorption, median, pos_amplitude, 
                    neg_amplitude) = MMT_twriting.Hg_Hb_Ha_tables(label, flux, 
                    o1, xval, pos_flux, dlambda)
                table_arrays = general_twriting.table_arr_appends(i, subtitle,
                    table_arrays, flux, flux2, flux3, ew, ew_emission, ew_absorption, 
                    median, pos_amplitude, neg_amplitude, 'MMT')
                if not (subtitle=='NB973' and i==2):
                    pos_amplitude_list.append(pos_amplitude)
                    neg_amplitude_list.append(neg_amplitude)
                    pos_sigma_list.append(o1[2])
                    if i==2:
                        neg_sigma_list.append(0)
                    else:
                        neg_sigma_list.append(o1[5])
                    median_list.append(median)
                else:
                    pos_amplitude_list.append(0)
                    neg_amplitude_list.append(0)
                    pos_sigma_list.append(0)
                    neg_sigma_list.append(0)
                    median_list.append(0)
                #endif
            #endtry
        #endfor
        
        for i in range(3):
            label = label_list[i] + ' ('+str(len_input_index[i])+')'
            if i == 0:
                num_stack_HG.append(int(len_input_index[i]))
            elif i == 1:
                num_stack_HB.append(int(len_input_index[i]))
            else: # i == 2:
                num_stack_HA.append(int(len_input_index[i]))

            ax = ax_list[subplot_index]
            try:
                pos_flux = pos_flux_list[i]
                flux = flux_list[i]
                if not (subtitle=='NB973' and i==2) and len_input_index[i] > 1:
                    pos_amplitude = pos_amplitude_list[i]
                    neg_amplitude = neg_amplitude_list[i]
                    pos_sigma = pos_sigma_list[i]
                    neg_sigma = neg_sigma_list[i]
                    median = median_list[i]
                    ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, pos_flux, flux,
                        pos_amplitude, neg_amplitude, pos_sigma, neg_sigma, median)
                else:
                    ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, pos_flux, flux)
            except IndexError: # assuming there's no pos_flux or flux value
                ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
            subplot_index+=1
        #endfor
    #endfor
    if title=='':
        f = general_plotting.final_plot_setup(f, r'MMT detections of H$\alpha$ emitters')
    else:
        f = general_plotting.final_plot_setup(f, title)

    if pp == None:
        plt.savefig(full_path+'Composite_Spectra/StellarMass/MMT_all_five.pdf')
        subtitle_list = np.array(['all']*len(stlrmass_bin_arr))
    else:
        pp.savefig()
        subtitle_list = np.array([title]*len(stlrmass_bin_arr))
    plt.close()

    EBV_hghb = HG_HB_EBV(HG_flux, HB_flux)

    HB_NB921_flux = np.copy(HB_flux)
    if title=='NB921' or bintype=='StlrMass':
        HB_NB921_flux = get_HB_NB921_flux(bintype=bintype)

    EBV_hahb = HA_HB_EBV(HA_flux, HB_NB921_flux, 'MMT', bintype, title)

    table00 = Table([subtitle_list, stlrmass_bin_arr, num_sources, num_stack_HG, num_stack_HB, num_stack_HA,
        avgz_arr, minz_arr, maxz_arr, 
        avg_stlrmass_arr, min_stlrmass_arr, max_stlrmass_arr, HG_flux, HB_flux, HB_NB921_flux, HA_flux, NII_6548_flux, 
        NII_6583_flux, HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr, HG_EW_abs, HB_EW_abs,
        HG_continuum, HB_continuum, HA_continuum, HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude, EBV_hghb, EBV_hahb], # IDs_arr
        names=['filter', 'stlrmass_bin', 'num_sources', 'num_stack_HG', 'num_stack_HB', 'num_stack_HA',
        'avgz', 'minz', 'maxz',
        'avg_stlrmass', 'min_stlrmass', 'max_stlrmass', 'HG_flux', 'HB_flux', 'HB_NB921_flux', 'HA_flux', 'NII_6548_flux', 
        'NII_6583_flux', 'HG_EW', 'HB_EW', 'HA_EW', 'HG_EW_corr', 'HB_EW_corr', 'HA_EW_corr', 'HG_EW_abs', 'HB_EW_abs',
        'HG_continuum', 'HB_continuum', 'HA_continuum', 'HG_pos_amplitude', 'HB_pos_amplitude', 'HA_pos_amplitude',
        'HG_neg_amplitude', 'HB_neg_amplitude', 'E(B-V)_hghb', 'E(B-V)_hahb']) # IDs

    if pp != None: return pp, table00

    asc.write(table00, full_path+'Composite_Spectra/StellarMass/MMT_all_five_data.txt',
        format='fixed_width_two_line', delimiter=' ')
#enddef

def plot_MMT_Ha_stlrmass_z():
    '''
    TODO(document)
    TODO(generalize stellar mass binning functionality?)
    TODO(implement flexible file-naming)
    '''
    print '>MMT STELLARMASS+REDSHIFT STACKING'
    pp = PdfPages(full_path+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ.pdf')
    table00 = None

    mmt_ii = np.array([x for x in range(len(NAME0)) if 
        ('Ha-NB' in NAME0[x] and inst_str0[x] in inst_dict['MMT'] 
            and stlr_mass[x] > 0 and (zspec0[x] > 0 and zspec0[x] < 9))])
    bins_ii_tbl = np.ndarray((5,5), dtype=object)

    bins_ii_tbl_temp = np.ndarray((5,5), dtype=object)
    for ff, ii in zip(['NB704', 'NB711', 'NB816', 'NB921', 'NB973'], [0,1,2,3,4]):
        filt_ii = np.array([x for x in range(len(mmt_ii)) if 'Ha-'+ff in NAME0[mmt_ii][x]])
        filt_masses = stlr_mass[mmt_ii][filt_ii]
        for n in [5, 4, 3, 2]:
            bins_ii = split_into_bins(filt_masses, n)
            if bins_ii != 'TOO SMALL': break
        for x in range(5 - len(bins_ii)):
            bins_ii.append([])
        bins_ii_tbl[ii] = bins_ii

        for jj in range(len(bins_ii)):
            bins_ii_tbl_temp[ii][jj] = mmt_ii[filt_ii][bins_ii_tbl[ii][jj]]

        title=ff
        print '>>>', title

        pp, table_data = plot_MMT_Ha_stlrmass(bins_ii_tbl_temp[ii], pp, title, 'StellarMassZ')
        if table00 == None:
            table00 = table_data
        else:
            table00 = vstack([table00, table_data])
        #endif
    #endfor

    asc.write(table00, full_path+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt',
        format='fixed_width_two_line', delimiter=' ')
    pp.close()
#enddef

def plot_Keck_Ha():
    '''
    Creates a pdf (8"x11") with 3x2 subplots for different lines and filter
    combinations.

    Then, the code iterates through every subplot in row-major filter-line
    order. Using only the 'good' indexes, finds 'match_index'. With those
    indexes of AP and inst_str0, calls AP_match.

    For NB921 Halpha, does a cross-match to ensure no 'invalid' point is
    plotted.

    Except for NB973 Halpha, the graph is 'de-redshifted' in order to have
    the spectral line appear in the subplot. The values to plot are called
    from sdf_stack_data.stack_data

    get_best_fit is called to obtain the best-fit spectra, overlay the
    best fit, and then calculate the flux

    Additionally, a line is plotted at the value at which the emission line
    should theoretically be (based on which emission line it is).

    The yscale is fixed for each filter type (usually the yscale values of
    the Halpha subplot).

    Minor ticks are set on, lines and filters are labeled, and with the
    line label is another label for the number of stacked sources that were
    used to produce the emission graph.

    At the end of all the iterations, the plot is saved and closed.

    The fluxes are also output to a separate .txt file.
    '''
    print '>KECK REDSHIFT STACKING'
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])
    (HB_flux, HA_flux, NII_6548_flux, NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr,
        HB_EW_abs, HB_continuum, HA_continuum, HB_pos_amplitude, HA_pos_amplitude,
        HB_neg_amplitude) = table_arrays
    (num_sources, num_stack_HG, num_stack_HB, num_stack_HA, avgz_arr, minz_arr, maxz_arr,
        IDs_arr) = ([], [], [], [], [], [], [], [])
    index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'Keck')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('Keck')

    f, axarr = plt.subplots(3, 2)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)

    subplot_index=0
    for (match_index,subtitle) in zip(index_list,subtitle_list):
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'Keck')
        AP_match = np.array([x for x in AP_match if x != 'INVALID_KECK'], dtype=np.float32)

        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match and gridz[x] != 0],dtype=np.int32)

        zs = np.array(gridz[input_index])
        if subtitle=='NB816':
            good_z = np.where((zs >= 0.21) & (zs <= 0.26))[0]
        elif subtitle=='NB921':
            good_z = np.where((zs >= 0.385) & (zs <= 0.429))[0]
        else:
            good_z = np.where((zs >= 0.45) & (zs <= 0.52))[0]
        #endif
        zs = np.average(zs[good_z])
        dlambda = (x0[1]-x0[0])/(1+zs)

        xval, yval, len_input_index, stacked_indexes, avgz, minz, maxz = stack_data(grid_ndarr, gridz, input_index,
            x0, 3800, 6700, dlambda, ff=subtitle, instr='Keck')

        num_sources.append(len_input_index[0])
        avgz_arr.append(avgz)
        minz_arr.append(minz)
        maxz_arr.append(maxz)
        
        # appending to the ID columns
        tempgridapstacked_ii = [str(y) for y in gridap[stacked_indexes]]

        mm0 = []
        for x in range(len(AP)):
            for y in tempgridapstacked_ii:
                if len(y)==5: 
                    y = '0'+y
                if y in AP[x]:
                    mm0.append(x)
        #endfor
        IDs_arr.append(','.join(NAME0[mm0]))
        
        # writing the spectra table
        table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
        spectra_file_path = full_path+'Composite_Spectra/Redshift/Keck_spectra_vals/'+subtitle+'.txt'
        asc.write(table0, spectra_file_path, format='fixed_width', delimiter=' ')

        # calculating flux for NII emissions
        pos_flux_list = []
        flux_list = []
        pos_amplitude_list = []
        neg_amplitude_list = []
        pos_sigma_list = []
        neg_sigma_list = []
        median_list = []
        for i in range(2):
            xmin0 = xmin_list[i]
            xmax0 = xmax_list[i]
            ax = ax_list[subplot_index+i]
            label = label_list[i]

            ax, flux, flux2, flux3, pos_flux, o1 = Keck_plotting.subplots_plotting(
                ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol, subplot_index+i)
            pos_flux_list.append(pos_flux)
            flux_list.append(flux)

            (ew, ew_emission, ew_absorption, median, pos_amplitude, 
                neg_amplitude) = Keck_twriting.Hb_Ha_tables(label, subtitle, flux, 
                o1, xval, pos_flux, dlambda)
            table_arrays = general_twriting.table_arr_appends(i, subtitle,
                table_arrays, flux, flux2, flux3, ew, ew_emission, ew_absorption, 
                median, pos_amplitude, neg_amplitude, 'Keck')
            if not (subtitle=='NB816' and i==0):
                pos_amplitude_list.append(pos_amplitude)
                neg_amplitude_list.append(neg_amplitude)
                pos_sigma_list.append(o1[2])
                if i==0:
                    neg_sigma_list.append(o1[5])
                else:
                    neg_sigma_list.append(0)
                median_list.append(median)
            else:
                pos_amplitude_list.append(0)
                neg_amplitude_list.append(0)
                pos_sigma_list.append(0)
                neg_sigma_list.append(0)
                median_list.append(0)
            #endif
        #endfor
        
        for i in range(2):
            label = label_list[i] + ' ('+str(len_input_index[i])+')'
            if i == 0:
                num_stack_HB.append(int(len_input_index[i]))
            else: # i == 1:
                num_stack_HA.append(int(len_input_index[i]))

            ax = ax_list[subplot_index]
            try:
                pos_flux = pos_flux_list[i]
                flux = flux_list[i]
                if not (subtitle=='NB816' and i==0):
                    pos_amplitude = pos_amplitude_list[i]
                    neg_amplitude = neg_amplitude_list[i]
                    pos_sigma = pos_sigma_list[i]
                    neg_sigma = neg_sigma_list[i]
                    median = median_list[i]
                    ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, pos_flux, flux,
                        pos_amplitude, neg_amplitude, pos_sigma, neg_sigma, median)
                else:
                    ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, pos_flux, flux)
            except SyntaxError: # assuming there's no pos_flux or flux value
                print 'no pos_flux or flux value'
                ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
            subplot_index+=1
        #endfor
    #endfor
    f = general_plotting.final_plot_setup(f, r'Keck detections of H$\alpha$ emitters')    
    plt.savefig(full_path+'Composite_Spectra/Redshift/Keck_stacked_spectra.pdf')
    plt.close()

    EBV_hahb = HA_HB_EBV(HA_flux, HB_flux, 'Keck')

    table00 = Table([subtitle_list, num_sources, num_stack_HB, num_stack_HA,
        avgz_arr, minz_arr, maxz_arr, 
        HB_flux, HA_flux, NII_6548_flux, 
        NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr, HB_EW_abs,
        HB_continuum, HA_continuum, HB_pos_amplitude, HA_pos_amplitude,
        HB_neg_amplitude, EBV_hahb], # IDs_arr
        names=['filter', 'num_sources', 'num_stack_HB', 'num_stack_HA',
        'avgz', 'minz', 'maxz',
        'HB_flux', 'HA_flux', 'NII_6548_flux', 
        'NII_6583_flux', 'HB_EW', 'HA_EW', 'HB_EW_corr', 'HA_EW_corr', 'HB_EW_abs',
        'HB_continuum', 'HA_continuum', 'HB_pos_amplitude', 'HA_pos_amplitude',
        'HB_neg_amplitude', 'E(B-V)_hahb']) # IDs

    asc.write(table00, full_path+'Composite_Spectra/Redshift/Keck_stacked_spectra_data.txt',
            format='fixed_width_two_line', delimiter=' ')
#enddef

def plot_Keck_Ha_stlrmass(index_list=[], pp=None, title='', bintype='StlrMass'):
    '''
    TODO(document)
    TODO(implement flexible stellar mass bin-readings)
    TODO(implement flexible file-naming)
        (nothing from the command line -- default into 5 bins by percentile)
        (number n from the command line -- make n bins by percentile)
        (file name from the command line -- flag to read the stellar mass bins from that ASCII file)
    TODO(get rid of assumption that there's only one page)
    '''
    if index_list == []:
        print '>KECK STELLARMASS STACKING'
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])
    (HB_flux, HA_flux, NII_6548_flux, NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr,
        HB_EW_abs, HB_continuum, HA_continuum, HB_pos_amplitude, HA_pos_amplitude,
        HB_neg_amplitude) = table_arrays
    (num_sources, num_stack_HG, num_stack_HB, num_stack_HA, avgz_arr, minz_arr, maxz_arr,
        stlrmass_bin_arr, avg_stlrmass_arr, min_stlrmass_arr, max_stlrmass_arr,
        IDs_arr) = ([], [], [], [], [], [], [], [], [], [], [], [])
    if index_list == []:
        index_list = general_plotting.get_index_list2(NAME0, stlr_mass, inst_str0, zspec0, inst_dict, 'Keck')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('Keck', stlr=True)

    f, axarr = plt.subplots(5, 2)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)

    subplot_index=0
    # this for-loop stacks by stlr mass
    for (match_index) in zip(index_list):
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'Keck')
        AP_match = np.array([x for x in AP_match if x != 'INVALID_KECK'], dtype=np.float32)
        
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)

        subtitle='stlrmass: '+str(min(stlr_mass[match_index]))+'-'+str(max(stlr_mass[match_index]))
        print '>>>', subtitle
        avg_stlrmass_arr.append(np.mean(stlr_mass[match_index]))
     	min_stlrmass_arr.append(np.min(stlr_mass[match_index]))
     	max_stlrmass_arr.append(np.max(stlr_mass[match_index]))

        dlambda = 0.1 # xval[1] - xval[0]

        xval, yval, len_input_index, stacked_indexes, avgz, minz, maxz = stack_data(grid_ndarr, gridz, input_index,
            x0, 3800, 6700, dlambda, instr='Keck')
        num_sources.append(len_input_index[0])
        avgz_arr.append(avgz)
        minz_arr.append(minz)
        maxz_arr.append(maxz)
        stlrmass_bin_arr.append(subtitle[10:])

        # appending to the ID columns
        tempgridapstacked_ii = [str(y) for y in gridap[stacked_indexes]]
        mm0 = []
        for x in range(len(AP)):
            for y in tempgridapstacked_ii:
                if len(y)==5: 
                    y = '0'+y
                if y in AP[x]:
                    mm0.append(x)
        #endfor
        IDs_arr.append(','.join(NAME0[mm0]))

        # writing the spectra table
        table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
        spectra_file_path = full_path+'Composite_Spectra/StellarMass/Keck_spectra_vals/'+subtitle[10:]+'.txt'
        asc.write(table0, spectra_file_path,
            format='fixed_width', delimiter=' ')

        # calculating flux for NII emissions
        pos_flux_list = []
        flux_list = []
        pos_amplitude_list = []
        neg_amplitude_list = []
        pos_sigma_list = []
        neg_sigma_list = []
        median_list = []
        for i in range(2):
            xmin0 = xmin_list[i]
            xmax0 = xmax_list[i]
            ax = ax_list[subplot_index+i]
            label = label_list[i]
            try:
                ax, flux, flux2, flux3, pos_flux, o1 = Keck_plotting.subplots_plotting(
                    ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol, subplot_index+i)
                pos_flux_list.append(pos_flux)
                flux_list.append(flux)
            except IndexError:
                print '(!!) There\'s some unexpected exception or another.'
                continue
            finally:
                (ew, ew_emission, ew_absorption, median, pos_amplitude, 
                  neg_amplitude) = Keck_twriting.Hb_Ha_tables(label, subtitle, flux, 
                  o1, xval, pos_flux, dlambda)
                table_arrays = general_twriting.table_arr_appends(i, subtitle,
                  table_arrays, flux, flux2, flux3, ew, ew_emission, ew_absorption, 
                  median, pos_amplitude, neg_amplitude, 'Keck')
                
                pos_amplitude_list.append(pos_amplitude)
                neg_amplitude_list.append(neg_amplitude)
                pos_sigma_list.append(o1[2])
                if i==0:
                    neg_sigma_list.append(o1[5])
                else:
                    neg_sigma_list.append(0)
                median_list.append(median)
            #endtry
        #endfor

        for i in range(2):
            label = label_list[i] + ' ('+str(len_input_index[i])+')'
            if i == 0:
                num_stack_HB.append(int(len_input_index[i]))
            else: # i == 1:
                num_stack_HA.append(int(len_input_index[i]))

            ax = ax_list[subplot_index]
            try:
                pos_flux = pos_flux_list[i]
                flux = flux_list[i]

                pos_amplitude = pos_amplitude_list[i]
                neg_amplitude = neg_amplitude_list[i]
                pos_sigma = pos_sigma_list[i]
                neg_sigma = neg_sigma_list[i]
                median = median_list[i]
                ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, pos_flux, flux,
                    pos_amplitude, neg_amplitude, pos_sigma, neg_sigma, median)
            except IndexError: # assuming there's no pos_flux or flux value
                ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
            subplot_index+=1
        #endfor
    #endfor
    if title=='':
        f = general_plotting.final_plot_setup(f, r'Keck detections of H$\alpha$ emitters')
    else:
        f = general_plotting.final_plot_setup(f, title)

    if pp == None:
        plt.savefig(full_path+'Composite_Spectra/StellarMass/Keck_all_five.pdf')
        subtitle_list = np.array(['all']*len(stlrmass_bin_arr))
    else:
        pp.savefig()
        subtitle_list = np.array([title]*len(stlrmass_bin_arr))
    plt.close()

    EBV_hahb = HA_HB_EBV(HA_flux, HB_flux, 'Keck', 'stlrmass')

    table00 = Table([subtitle_list, stlrmass_bin_arr, num_sources, num_stack_HB, num_stack_HA,
        avgz_arr, minz_arr, maxz_arr, 
        avg_stlrmass_arr, min_stlrmass_arr, max_stlrmass_arr, HB_flux, HA_flux, NII_6548_flux, 
        NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr, HB_EW_abs,
        HB_continuum, HA_continuum, HB_pos_amplitude, HA_pos_amplitude,
        HB_neg_amplitude, EBV_hahb], # IDs_arr
        names=['filter', 'stlrmass_bin', 'num_sources', 'num_stack_HB', 'num_stack_HA',
        'avgz', 'minz', 'maxz',
        'avg_stlrmass', 'min_stlrmass', 'max_stlrmass', 'HB_flux', 'HA_flux', 'NII_6548_flux', 
        'NII_6583_flux', 'HB_EW', 'HA_EW', 'HB_EW_corr', 'HA_EW_corr', 'HB_EW_abs',
        'HB_continuum', 'HA_continuum', 'HB_pos_amplitude', 'HA_pos_amplitude',
        'HB_neg_amplitude', 'E(B-V)_hahb']) # 'IDs'

    if pp != None: return pp, table00

    asc.write(table00, full_path+'Composite_Spectra/StellarMass/Keck_all_five_data.txt',
            format='fixed_width_two_line', delimiter=' ')
#enddef

def plot_Keck_Ha_stlrmass_z():
    '''
    TODO(document)
    TODO(generalize stellar mass binning functionality?)
    TODO(implement flexible file-naming)
    '''
    print '>KECK STELLARMASS+REDSHIFT STACKING'
    pp = PdfPages(full_path+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ.pdf')
    table00 = None
    
    keck_ii = np.array([x for x in range(len(NAME0)) if 
                        ('Ha-NB9' in NAME0[x] and inst_str0[x] in inst_dict['Keck'] 
                         and stlr_mass[x] > 0 and (zspec0[x] > 0 and zspec0[x] < 9))])
    bins_ii_tbl = np.ndarray((2,5), dtype=object)

    bins_ii_tbl_temp = np.ndarray((2,5), dtype=object)
    for ff, ii in zip(['NB921', 'NB973'], [0,1]):
        filt_ii = np.array([x for x in range(len(keck_ii)) if 'Ha-'+ff in NAME0[keck_ii][x]])
        filt_masses = stlr_mass[keck_ii][filt_ii]
        for n in [5, 4, 3, 2]:
            bins_ii = split_into_bins(filt_masses, n)
            if bins_ii != 'TOO SMALL': break
        for x in range(5 - len(bins_ii)):
            bins_ii.append([])
        bins_ii_tbl[ii] = bins_ii

        for jj in range(len(bins_ii)):
            bins_ii_tbl_temp[ii][jj] = keck_ii[filt_ii][bins_ii_tbl[ii][jj]]
    
        title=ff
        print '>>>', title

        pp, table_data = plot_Keck_Ha_stlrmass(bins_ii_tbl_temp[ii], pp, title, 'StellarMassZ')
        if table00 == None:
            table00 = table_data
        else:
            table00 = vstack([table00, table_data])
        #endif
    #endfor
    
    asc.write(table00, full_path+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt',
        format='fixed_width_two_line', delimiter=' ')
    pp.close()
#enddef


#----main body---------------------------------------------------------------#
# o Reads relevant inputs, combining all of the input data into one ordered
#   array for AP by calling make_AP_arr_MMT, make_AP_arr_DEIMOS,
#   make_AP_arr_merged, and make_AP_arr_FOCAS. 
# o Using the AP order, then creates HA, HB, HG_Y0 arrays by calling get_Y0
# o Then looks at the grid stored in 'Spectra/spectral_*_grid_data.txt'
#   and 'Spectra/spectral_*_grid.fits' created from
#   combine_spectral_data.py in order to read in relevant data columns.
# o Then calls plot_*_Ha.
# o Done for both MMT and Keck data.
#----------------------------------------------------------------------------#
full_path = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'
inst_dict = {} ##used
inst_dict['MMT'] = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,','merged,FOCAS,']
inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,',
                     'Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,','merged,FOCAS,']
tol = 3 #in angstroms, used for NII emission flux calculations ##used

k_hg = cardelli(4341 * u.Angstrom)
k_hb = cardelli(4861 * u.Angstrom)
k_ha = cardelli(6563 * u.Angstrom)

nbia = pyfits.open(full_path+'Catalogs/NB_IA_emitters.nodup.colorrev.fix.fits')
nbiadata = nbia[1].data
NAME0 = nbiadata['NAME']

zspec = asc.read(full_path+'Catalogs/nb_ia_zspec.txt',guess=False,
                 Reader=asc.CommentedHeader)
zspec0 = np.array(zspec['zspec0'])
inst_str0 = np.array(zspec['inst_str0']) ##used

fout  = asc.read(full_path+'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.fout',
                 guess=False,Reader=asc.NoHeader)
stlr_mass = np.array(fout['col7']) ##used
nan_stlr_mass = np.copy(stlr_mass)
nan_stlr_mass[nan_stlr_mass < 0] = np.nan

data_dict = create_ordered_AP_arrays(AP_only = True)
AP = data_dict['AP'] ##used

print '### looking at the MMT grid'
griddata = asc.read(full_path+'Spectra/spectral_MMT_grid_data.txt',guess=False)
gridz  = np.array(griddata['ZSPEC']) ##used
gridap = np.array(griddata['AP']) ##used
grid   = pyfits.open(full_path+'Spectra/spectral_MMT_grid.fits')
grid_ndarr = grid[0].data ##used
grid_hdr   = grid[0].header
CRVAL1 = grid_hdr['CRVAL1']
CDELT1 = grid_hdr['CDELT1']
NAXIS1 = grid_hdr['NAXIS1']
x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1) ##used
# mask spectra that doesn't exist or lacks coverage in certain areas
ndarr_zeros = np.where(grid_ndarr == 0)
mask_ndarr = np.zeros_like(grid_ndarr)
mask_ndarr[ndarr_zeros] = 1
# mask spectra with unreliable redshift
bad_zspec = [x for x in range(len(gridz)) if gridz[x] > 9 or gridz[x] < 0]
mask_ndarr[bad_zspec,:] = 1
grid_ndarr = ma.masked_array(grid_ndarr, mask=mask_ndarr, fill_value=np.nan)

print '### plotting MMT_Ha'
plot_MMT_Ha()
plot_MMT_Ha_stlrmass()
plot_MMT_Ha_stlrmass_z()
grid.close()

print '### looking at the Keck grid'
griddata = asc.read(full_path+'Spectra/spectral_Keck_grid_data.txt',guess=False)
gridz  = np.array(griddata['ZSPEC']) ##used
gridap = np.array(griddata['AP']) ##used
grid   = pyfits.open(full_path+'Spectra/spectral_Keck_grid.fits')
grid_ndarr = grid[0].data ##used
grid_hdr   = grid[0].header
CRVAL1 = grid_hdr['CRVAL1']
CDELT1 = grid_hdr['CDELT1']
NAXIS1 = grid_hdr['NAXIS1']
x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1) ##used
# mask spectra that doesn't exist or lacks coverage in certain areas
ndarr_zeros = np.where(grid_ndarr == 0)
mask_ndarr = np.zeros_like(grid_ndarr)
mask_ndarr[ndarr_zeros] = 1
# mask spectra with unreliable redshift
bad_zspec = [x for x in range(len(gridz)) if gridz[x] > 9 or gridz[x] < 0]
mask_ndarr[bad_zspec,:] = 1
grid_ndarr = ma.masked_array(grid_ndarr, mask=mask_ndarr)

print '### plotting Keck_Ha'
plot_Keck_Ha()
plot_Keck_Ha_stlrmass()
plot_Keck_Ha_stlrmass_z()
grid.close()

nbia.close()
print '### done'
#endmain