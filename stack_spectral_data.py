"""
NAME:
    stack_spectral_data.py

PURPOSE:
    This code creates a PDF file with 15 subplots, filter-emission line
    row-major order, to show all the MMT and Keck spectral data stacked and
    plotted in a 'de-redshifted' frame.
    Specific to SDF data.

INPUTS:
    'Catalogs/python_outputs/nbia_all_nsource.fits'
    'Catalogs/nb_ia_zspec.txt'
    'Spectra/spectral_MMT_grid_data.txt'
    'Spectra/spectral_MMT_grid.fits'

OUTPUTS:
    'Spectra/Ha_MMT_stacked_ew.txt'
    'Spectra/Ha_MMT_stacked_fluxes.txt'
    'Spectra/Ha_MMT_stacked.pdf'
    'Spectra/Ha_Keck_stacked_ew.txt'
    'Spectra/Ha_Keck_stacked_fluxes.txt'
    'Spectra/Ha_Keck_stacked.pdf'
"""

import numpy as np, numpy.ma as ma, matplotlib.pyplot as plt
import plotting.hg_hb_ha_plotting as MMT_plotting
import plotting.hb_ha_plotting as Keck_plotting
import plotting.general_plotting as general_plotting
import writing_tables.hg_hb_ha_tables as MMT_twriting
import writing_tables.hb_ha_tables as Keck_twriting
import writing_tables.general_tables as general_twriting
from analysis.sdf_stack_data import stack_data
from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table, vstack
from create_ordered_AP_arrays import create_ordered_AP_arrays
from matplotlib.backends.backend_pdf import PdfPages

def correct_instr_AP(indexed_AP, indexed_inst_str0, instr):
    '''
    Returns the indexed AP_match array based on the 'match_index' from
    plot_MMT/Keck_Ha
    '''
    for ii in range(len(indexed_inst_str0)):
        if indexed_inst_str0[ii]=='merged,':
            if instr=='MMT':
                indexed_AP[ii] = indexed_AP[ii][:5]
            elif instr=='Keck':
                indexed_AP[ii] = indexed_AP[ii][6:]
        #endif
    #endfor
    return indexed_AP
#enddef

def plot_MMT_Ha(index_list=[], pp=None, title='', bintype='Redshift', stlrmassindex0=[]):
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
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    (HG_flux, HB_flux, HA_flux, NII_6548_flux, NII_6583_flux,
        HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr,
        HG_EW_abs, HB_EW_abs, HG_continuum, HB_continuum, HA_continuum,
        HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude) = table_arrays
    (num_sources, num_bad_NB921_sources, minz_arr, maxz_arr,
        spectra_file_path_arr, stlrmass_bin_arr, avg_stlrmass_arr,
        IDs_arr, IDs_bad_NB921_sources) = ([], [], [], [], [], [], [], [], [])
    if index_list == []:
        index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'MMT')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('MMT')
    
    f, axarr = plt.subplots(5, 3)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)
    
    subplot_index=0
    # this for-loop stacks by filter
    for (match_index, subtitle) in zip(index_list, subtitle_list):
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)
        if len(input_index) < 2: 
            print 'Not enough sources to stack (less than two)'
            [arr.append(0) for arr in table_arrays]
            num_sources.append(0)
            num_bad_NB921_sources.append(0)
            minz_arr.append(0)
            maxz_arr.append(0)
            spectra_file_path_arr.append('N/A')
            IDs_arr.append('N/A')
            IDs_bad_NB921_sources.append('N/A')
            if bintype=='Redshift': 
                stlrmass_bin_arr.append('N/A')
                avg_stlrmass_arr.append(0)
            elif bintype=='StellarMassZ': 
                stlrmass_bin_arr.append(title[10:])
                avg_stlrmass_arr.append(np.mean(stlr_mass[stlrmassindex0]))
            for i in range(3):
                ax = ax_list[subplot_index]
                label = label_list[i]
                MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
                subplot_index += 1
            continue
        #endif

        try:
            xval, yval, len_input_index, stacked_indexes, minz, maxz = stack_data(grid_ndarr, gridz, input_index,
                x0, 3700, 6700, ff=subtitle, instr='MMT', AP_rows=halpha_maskarr)
            num_sources.append(len_input_index[0])
            num_bad_NB921_sources.append(len_input_index[1])
            minz_arr.append(minz)
            maxz_arr.append(maxz)

            # appending to the ID columns
            mm0 = [x for x in range(len(AP)) if any(y in AP[x][:5] for y in gridap[stacked_indexes[0]])] # gridap ordering -> NBIA ordering
            IDs_arr.append(','.join(NAME0[mm0]))
            mm1 = [x for x in range(len(AP)) if any(y in AP[x][:5] for y in gridap[stacked_indexes[1]])] # gridap ordering -> NBIA ordering
            if len(mm1)==0:
                IDs_bad_NB921_sources.append('N/A')
            else:
                IDs_bad_NB921_sources.append(','.join(NAME0[mm1]))
            #endif

            # writing the spectra table
            table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
            if bintype=='Redshift':
                spectra_file_path = full_path+'Composite_Spectra/'+bintype+'/MMT_spectra_vals/'+subtitle+'.txt'
                stlrmass_bin_arr.append('N/A')
                avg_stlrmass_arr.append(0)
            elif bintype=='StellarMassZ':
                spectra_file_path = full_path+'Composite_Spectra/'+bintype+'/MMT_spectra_vals/'+title[10:]+'_'+subtitle+'.txt'
                stlrmass_bin_arr.append(title[10:])
                avg_stlrmass_arr.append(np.mean(stlr_mass[stlrmassindex0]))
            #endif
            asc.write(table0, spectra_file_path, format='fixed_width', delimiter=' ')
            spectra_file_path_arr.append(spectra_file_path)
            
            # calculating flux for NII emissions
            zs = np.array(gridz[input_index])
            if subtitle=='NB704' or subtitle=='NB711':
                good_z = np.where(zs < 0.1)[0]
            elif subtitle=='NB816':
                good_z = np.where(zs < 0.3)[0]
            elif subtitle=='NB921':
                good_z = np.where(zs < 0.6)[0]
            else:
                good_z = np.where(zs < 0.6)[0]
            #endif
            zs = np.average(zs[good_z])
            dlambda = (x0[1]-x0[0])/(1+zs)

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
                        ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol)
                    pos_flux_list.append(pos_flux)
                    flux_list.append(flux)
                except SyntaxError:
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
            #endfor

        except ValueError:
            print 'ValueError: none exist'
        #endtry
        
        for i in range(3):
            if subplot_index==11:
                label = label_list[i] + ' ('+str(len_input_index[0]-len_input_index[1])+')'
            else:
                label = label_list[i] + ' ('+str(len_input_index[0])+')'
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
    if title=='':
        f = general_plotting.final_plot_setup(f, r'MMT detections of H$\alpha$ emitters')
    else:
        f = general_plotting.final_plot_setup(f, title)
    if pp == None:
        plt.savefig(full_path+'Composite_Spectra/Redshift/MMT_stacked_spectra.pdf')
    else:
        pp.savefig()
    plt.close()

    table00 = Table([subtitle_list, stlrmass_bin_arr, num_sources, num_bad_NB921_sources, minz_arr, maxz_arr, 
        avg_stlrmass_arr, IDs_arr, IDs_bad_NB921_sources, spectra_file_path_arr, HG_flux, HB_flux, HA_flux, NII_6548_flux, 
        NII_6583_flux, HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr, HG_EW_abs, HB_EW_abs,
        HG_continuum, HB_continuum, HA_continuum, HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude], 
        names=['filter', 'stlrmass_bin', 'num_sources', 'num_bad_MMT_Halpha_NB921', 'minz', 'maxz',
        'avg_stlrmass', 'IDs', 'IDs_bad_NB921_sources', 'spectra_file_path', 'HG_flux', 'HB_flux', 'HA_flux', 'NII_6548_flux', 
        'NII_6583_flux', 'HG_EW', 'HB_EW', 'HA_EW', 'HG_EW_corr', 'HB_EW_corr', 'HA_EW_corr', 'HG_EW_abs', 'HB_EW_abs',
        'HG_continuum', 'HB_continuum', 'HA_continuum', 'HG_pos_amplitude', 'HB_pos_amplitude', 'HA_pos_amplitude',
        'HG_neg_amplitude', 'HB_neg_amplitude'])
    if pp != None: return pp, table00

    asc.write(table00, full_path+'Composite_Spectra/Redshift/MMT_stacked_spectra_data.txt',
        format='fixed_width_two_line', delimiter=' ')
#enddef

def plot_MMT_Ha_stlrmass():
    '''
    TODO(document)
    TODO(implement flexible stellar mass bin-readings)
    TODO(implement flexible file-naming)
        (nothing from the command line -- default into 5 bins by percentile)
        (number n from the command line -- make n bins by percentile)
        (file name from the command line -- flag to read the stellar mass bins from that ASCII file)
    TODO(get rid of assumption that there's only one page)
    TODO(add in the flux/EW/full spectra ASCII table writing code)
    '''
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    (HG_flux, HB_flux, HA_flux, NII_6548_flux, NII_6583_flux,
        HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr,
        HG_EW_abs, HB_EW_abs, HG_continuum, HB_continuum, HA_continuum,
        HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude) = table_arrays
    (num_sources, num_bad_NB921_sources, minz_arr, maxz_arr,
        spectra_file_path_arr, stlrmass_bin_arr, avg_stlrmass_arr,
        IDs_arr, IDs_bad_NB921_sources) = ([], [], [], [], [], [], [], [], [])
    index_list = general_plotting.get_index_list2(stlr_mass, inst_str0, inst_dict, 'MMT')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('MMT')

    pp = PdfPages(full_path+'Composite_Spectra/StellarMass/MMT_all_five.pdf')

    f, axarr = plt.subplots(5, 3)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)

    subplot_index=0
    # this for-loop stacks by stlr mass
    for (match_index) in (index_list):
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)
        try:
            subtitle='stlrmass: '+str(min(stlr_mass[match_index]))+'-'+str(max(stlr_mass[match_index]))
            print '>>>', subtitle
            avg_stlrmass_arr.append(np.mean(stlr_mass[match_index]))
            xval, yval, len_input_index, stacked_indexes, minz, maxz = stack_data(grid_ndarr, gridz, input_index,
                x0, 3700, 6700)
            num_sources.append(len_input_index[0])
            num_bad_NB921_sources.append(len_input_index[1])
            minz_arr.append(minz)
            maxz_arr.append(maxz)
            stlrmass_bin_arr.append(subtitle[10:])

            # appending to the ID columns
            mm0 = [x for x in range(len(AP)) if any(y in AP[x][:5] for y in gridap[stacked_indexes[0]])] # gridap ordering -> NBIA ordering
            IDs_arr.append(','.join(NAME0[mm0]))
            mm1 = [x for x in range(len(AP)) if any(y in AP[x][:5] for y in gridap[stacked_indexes[1]])] # gridap ordering -> NBIA ordering
            if len(mm1)==0:
                IDs_bad_NB921_sources.append('N/A')
            else:
                IDs_bad_NB921_sources.append(','.join(NAME0[mm1]))
            #endif

            # writing the spectra table
            table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
            spectra_file_path = full_path+'Composite_Spectra/StellarMass/MMT_spectra_vals/'+subtitle[10:]+'.txt'
            asc.write(table0, spectra_file_path,
                format='fixed_width', delimiter=' ')
            spectra_file_path_arr.append(spectra_file_path)

            # calculating flux for NII emissions
            dlambda = xval[1] - xval[0]

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
                        ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol)
                    pos_flux_list.append(pos_flux)
                    flux_list.append(flux)
                except ValueError:
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
            
        except ValueError:
            print 'ValueError: none exist'
        #endtry
        
        for i in range(3):
            label = label_list[i] + ' ('+str(len_input_index[0])+')'
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

    subtitle_list = np.array(['all']*len(subtitle_list))

    table00 = Table([subtitle_list, stlrmass_bin_arr, num_sources, num_bad_NB921_sources, minz_arr, maxz_arr, 
        avg_stlrmass_arr, IDs_arr, IDs_bad_NB921_sources, spectra_file_path_arr, HG_flux, HB_flux, HA_flux, NII_6548_flux, 
        NII_6583_flux, HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr, HG_EW_abs, HB_EW_abs,
        HG_continuum, HB_continuum, HA_continuum, HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude], 
        names=['filter', 'stlrmass_bin', 'num_sources', 'num_bad_MMT_Halpha_NB921', 'minz', 'maxz',
        'avg_stlrmass', 'IDs', 'IDs_bad_NB921_sources', 'spectra_file_path', 'HG_flux', 'HB_flux', 'HA_flux', 'NII_6548_flux', 
        'NII_6583_flux', 'HG_EW', 'HB_EW', 'HA_EW', 'HG_EW_corr', 'HB_EW_corr', 'HA_EW_corr', 'HG_EW_abs', 'HB_EW_abs',
        'HG_continuum', 'HB_continuum', 'HA_continuum', 'HG_pos_amplitude', 'HB_pos_amplitude', 'HA_pos_amplitude',
        'HG_neg_amplitude', 'HB_neg_amplitude'])
    asc.write(table00, full_path+'Composite_Spectra/StellarMass/MMT_all_five_data.txt',
        format='fixed_width_two_line', delimiter=' ')

    pp.savefig()
    plt.close()
    pp.close()
#enddef

def plot_MMT_Ha_stlrmass_z():
    '''
    TODO(document)
    TODO(generalize stellar mass binning functionality?)
    TODO(implement flexible file-naming)
    '''
    stlrmass_index_list = general_plotting.get_index_list2(stlr_mass, inst_str0, inst_dict, 'MMT')
    pp = PdfPages(full_path+'Composite_Spectra/StellarMassZ/MMT_two_percbins.pdf')
    table00 = None
    n = 2 # how many redshifts we want to take into account (max 5, TODO(generalize this?))
    for stlrmassindex0 in stlrmass_index_list[:n]:        
        title='stlrmass: '+str(min(stlr_mass[stlrmassindex0]))+'-'+str(max(stlr_mass[stlrmassindex0]))
        print '>>>', title

        # get one stlrmass bin per page
        temp_index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'MMT')
        index_list = []
        for index0 in temp_index_list:
            templist = [x for x in index0 if x in stlrmassindex0]
            index_list.append(templist)
        #endfor

        pp, table_data = plot_MMT_Ha(index_list, pp, title, 'StellarMassZ', stlrmassindex0)
        if table00 == None:
            table00 = table_data
        else:
            table00 = vstack([table00, table_data])
        #endif
    #endfor
    asc.write(table00, full_path+'Composite_Spectra/StellarMassZ/MMT_two_percbins_data.txt',
        format='fixed_width_two_line', delimiter=' ')
    pp.close()
#enddef

def plot_Keck_Ha(index_list=[], pp=None, title='', bintype='Redshift', stlrmassindex0=[]):
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
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])
    (HB_flux, HA_flux, NII_6548_flux, NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr,
        HB_EW_abs, HB_continuum, HA_continuum, HB_pos_amplitude, HA_pos_amplitude,
        HB_neg_amplitude) = table_arrays
    (num_sources, minz_arr, maxz_arr, spectra_file_path_arr, stlrmass_bin_arr, avg_stlrmass_arr,
        IDs_arr) = ([], [], [], [], [], [], [])
    if index_list == []:
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
        if len(input_index) < 2: 
            print 'Not enough sources to stack (less than two)'
            [arr.append(0) for arr in table_arrays]
            num_sources.append(0)
            minz_arr.append(0)
            maxz_arr.append(0)
            spectra_file_path_arr.append('N/A')
            IDs_arr.append('N/A')
            if bintype=='Redshift': 
                stlrmass_bin_arr.append('N/A')
                avg_stlrmass_arr.append(0)
            elif bintype=='StellarMassZ': 
                stlrmass_bin_arr.append(title[10:])
                avg_stlrmass_arr.append(np.mean(stlr_mass[stlrmassindex0]))
            for i in range(2):
                ax = ax_list[subplot_index]
                label = label_list[i]
                Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
                subplot_index += 1
            continue
        #endif

        try:
            xval, yval, len_input_index, stacked_indexes, minz, maxz = stack_data(grid_ndarr, gridz, input_index,
                x0, 3800, 6700, ff=subtitle)
            num_sources.append(len_input_index[0])
            minz_arr.append(minz)
            maxz_arr.append(maxz)
            
            # appending to the ID columns
            tempgridapstacked_ii = [str(y) for y in gridap[stacked_indexes[0]]]
            mm0 = []
            for x in range(len(AP)):
                for y in tempgridapstacked_ii:
                    if len(y)==5: 
                        y = '0'+y
                    if y in AP[x][6:]:
                        mm0.append(x)
            #endfor
            IDs_arr.append(','.join(NAME0[mm0]))
            
            # writing the spectra table
            table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
            if bintype=='Redshift':
                spectra_file_path = full_path+'Composite_Spectra/'+bintype+'/Keck_spectra_vals/'+subtitle+'.txt'
                stlrmass_bin_arr.append('N/A')
                avg_stlrmass_arr.append(0)
            elif bintype=='StellarMassZ':
                spectra_file_path = full_path+'Composite_Spectra/'+bintype+'/Keck_spectra_vals/'+title[10:]+'_'+subtitle+'.txt'
                stlrmass_bin_arr.append(title[10:])
                avg_stlrmass_arr.append(np.mean(stlr_mass[stlrmassindex0]))
            #endif
            asc.write(table0, spectra_file_path, format='fixed_width', delimiter=' ')
            spectra_file_path_arr.append(spectra_file_path)

            # calculating flux for NII emissions
            zs = np.array(gridz[input_index])
            if subtitle=='NB816':
                good_z = np.where(zs < 0.3)[0]
            elif subtitle=='NB921':
                good_z = np.where(zs < 0.6)[0]
            else:
                good_z = np.where(zs < 0.6)[0]
            #endif
            zs = np.average(zs[good_z])
            dlambda = (x0[1]-x0[0])/(1+zs)

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
                except ValueError:
                    print 'ValueError??'
                    continue
                else:
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
            
        except SyntaxError:
            print 'ValueError: none exist'
        
        for i in range(2):
            label = label_list[i] + ' ('+str(len_input_index[0])+')'
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
    if title=='':
        f = general_plotting.final_plot_setup(f, r'Keck detections of H$\alpha$ emitters')
    else:
        f = general_plotting.final_plot_setup(f, title)
    if pp == None:
        plt.savefig(full_path+'Composite_Spectra/Redshift/Keck_stacked_spectra.pdf')
    else:
        pp.savefig()
    plt.close()

    table00 = Table([subtitle_list, stlrmass_bin_arr, num_sources, minz_arr, maxz_arr, 
        avg_stlrmass_arr, IDs_arr, spectra_file_path_arr, HB_flux, HA_flux, NII_6548_flux, 
        NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr, HB_EW_abs,
        HB_continuum, HA_continuum, HB_pos_amplitude, HA_pos_amplitude,
        HB_neg_amplitude], 
        names=['filter', 'stlrmass_bin', 'num_sources', 'minz', 'maxz',
        'avg_stlrmass', 'IDs', 'spectra_file_path', 'HB_flux', 'HA_flux', 'NII_6548_flux', 
        'NII_6583_flux', 'HB_EW', 'HA_EW', 'HB_EW_corr', 'HA_EW_corr', 'HB_EW_abs',
        'HB_continuum', 'HA_continuum', 'HB_pos_amplitude', 'HA_pos_amplitude',
        'HB_neg_amplitude'])

    if pp != None: return pp, table00
    asc.write(table00, full_path+'Composite_Spectra/Redshift/Keck_stacked_spectra_data.txt',
            format='fixed_width_two_line', delimiter=' ')
#enddef

def plot_Keck_Ha_stlrmass():
    '''
    TODO(document)
    TODO(implement flexible stellar mass bin-readings)
    TODO(implement flexible file-naming)
        (nothing from the command line -- default into 5 bins by percentile)
        (number n from the command line -- make n bins by percentile)
        (file name from the command line -- flag to read the stellar mass bins from that ASCII file)
    TODO(get rid of assumption that there's only one page)
    TODO(add in the flux/EW/full spectra ASCII table writing code)
    '''
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])
    (HB_flux, HA_flux, NII_6548_flux, NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr,
        HB_EW_abs, HB_continuum, HA_continuum, HB_pos_amplitude, HA_pos_amplitude,
        HB_neg_amplitude) = table_arrays
    (num_sources, minz_arr, maxz_arr, spectra_file_path_arr, stlrmass_bin_arr, avg_stlrmass_arr,
        IDs_arr) = ([], [], [], [], [], [], [])
    index_list = general_plotting.get_index_list2(nan_stlr_mass, inst_str0, inst_dict, 'Keck')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('Keck', stlr=True)

    pp = PdfPages(full_path+'Composite_Spectra/StellarMass/Keck_all_five.pdf')

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
        try:
            subtitle='stlrmass: '+str(min(stlr_mass[match_index]))+'-'+str(max(stlr_mass[match_index]))
            print '>>>', subtitle
            avg_stlrmass_arr.append(np.mean(stlr_mass[match_index]))
            xval, yval, len_input_index, stacked_indexes, minz, maxz = stack_data(grid_ndarr, gridz, input_index,
                x0, 3800, 6700)
            num_sources.append(len_input_index[0])
            minz_arr.append(minz)
            maxz_arr.append(maxz)
            stlrmass_bin_arr.append(subtitle[10:])

            # appending to the ID columns
            tempgridapstacked_ii = [str(y) for y in gridap[stacked_indexes[0]]]
            mm0 = []
            for x in range(len(AP)):
                for y in tempgridapstacked_ii:
                    if len(y)==5: 
                        y = '0'+y
                    if y in AP[x][6:]:
                        mm0.append(x)
            #endfor
            IDs_arr.append(','.join(NAME0[mm0]))

            # writing the spectra table
            table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
            spectra_file_path = full_path+'Composite_Spectra/StellarMass/Keck_spectra_vals/'+subtitle[10:]+'.txt'
            asc.write(table0, spectra_file_path,
                format='fixed_width', delimiter=' ')
            spectra_file_path_arr.append(spectra_file_path)

            # calculating flux for NII emissions
            dlambda = xval[1] - xval[0]

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
                except ValueError:
                    continue
                finally:
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
                #endtry
            #endfor

        except ValueError:
            print 'ValueError: none exist'
        #endtry

        for i in range(2):
            label = label_list[i] + ' ('+str(len_input_index[0])+')'
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
            except IndexError: # assuming there's no pos_flux or flux value
                ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
            subplot_index+=1
        #endfor
    #endfor
    f = general_plotting.final_plot_setup(f, r'Keck detections of H$\alpha$ emitters')

    subtitle_list = np.array(['all']*len(stlrmass_bin_arr))

    table00 = Table([subtitle_list, stlrmass_bin_arr, num_sources, minz_arr, maxz_arr, 
        avg_stlrmass_arr, IDs_arr, spectra_file_path_arr, HB_flux, HA_flux, NII_6548_flux, 
        NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr, HB_EW_abs,
        HB_continuum, HA_continuum, HB_pos_amplitude, HA_pos_amplitude,
        HB_neg_amplitude], 
        names=['filter', 'stlrmass_bin', 'num_sources', 'minz', 'maxz',
        'avg_stlrmass', 'IDs', 'spectra_file_path', 'HB_flux', 'HA_flux', 'NII_6548_flux', 
        'NII_6583_flux', 'HB_EW', 'HA_EW', 'HB_EW_corr', 'HA_EW_corr', 'HB_EW_abs',
        'HB_continuum', 'HA_continuum', 'HB_pos_amplitude', 'HA_pos_amplitude',
        'HB_neg_amplitude'])
    asc.write(table00, full_path+'Composite_Spectra/StellarMass/Keck_all_five_data.txt',
            format='fixed_width_two_line', delimiter=' ')

    pp.savefig()
    plt.close()
    pp.close()
#enddef

def plot_Keck_Ha_stlrmass_z():
    '''
    TODO(document)
    TODO(generalize stellar mass binning functionality?)
    TODO(implement flexible file-naming)
    '''
    stlrmass_index_list = general_plotting.get_index_list2(stlr_mass, inst_str0, inst_dict, 'Keck')
    pp = PdfPages(full_path+'Composite_Spectra/StellarMassZ/Keck_five_percbins.pdf')
    table00 = None
    n = 5 # how many redshifts we want to take into account (max 5, TODO(generalize this?))
    for stlrmassindex0 in stlrmass_index_list[:n]:
        title='stlrmass: '+str(min(stlr_mass[stlrmassindex0]))+'-'+str(max(stlr_mass[stlrmassindex0]))
        print '>>>', title

        # get one stlrmass bin per page
        temp_index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'Keck')
        index_list = []
        for index0 in temp_index_list:
            templist = [x for x in index0 if x in stlrmassindex0]
            index_list.append(templist)
        #endfor

        pp, table_data = plot_Keck_Ha(index_list, pp, title, 'StellarMassZ', stlrmassindex0)
        if table00 == None:
            table00 = table_data
        else:
            table00 = vstack([table00, table_data])
        #endif
    #endfor
    asc.write(table00, full_path+'Composite_Spectra/StellarMassZ/Keck_five_percbins_data.txt',
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
good_NB921_Halpha = ['S.245','S.278','S.291','S.306','S.308','S.333','S.334',
                     'S.350','S.364','A.134','D.076','D.099','D.123','D.125',
                     'D.127','D.135','D.140','D.215','D.237','D.298'] ##used
inst_dict = {} ##used
inst_dict['MMT'] = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,']
inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,',
                     'Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,']
tol = 3 #in angstroms, used for NII emission flux calculations ##used

nbia = pyfits.open(full_path+'Catalogs/python_outputs/nbia_all_nsource.fits')
nbiadata = nbia[1].data
NAME0 = nbiadata['source_name'] ##used

zspec = asc.read(full_path+'Catalogs/nb_ia_zspec.txt',guess=False,
                 Reader=asc.CommentedHeader)
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
grid_ndarr = ma.masked_array(grid_ndarr, mask=mask_ndarr)

halpha_maskarr = np.array([x for x in range(len(gridap)) if gridap[x] not in good_NB921_Halpha]) 

print '### plotting MMT_Ha'
plot_MMT_Ha()
plot_MMT_Ha_stlrmass()
# plot_MMT_Ha_stlrmass_z()
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
# plot_Keck_Ha_stlrmass_z()
grid.close()

nbia.close()
print '### done'
#endmain