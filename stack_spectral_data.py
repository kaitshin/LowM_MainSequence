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
from astropy.table import Table
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

def write_spectral_table(instr, grid_ndarr, gridz, input_index, x0, name, full_path, shortlabel, bintype):
    '''
    Writes a table of spectra for the entire wavelength range of each stacked 
    galaxy data set. The resulting ASCII files are saved in a folder in 'Composite_Spectra/'
    depending on binning type (redshift, stlrmass, stlrmass+z) and instrument of detection
    (MMT, Keck).

    The 'name' is either:
    	o The filter (if bintype='Redshift')
    	o The stellar mass range (if bintype='StellarMass')
    	o The stellar mass range + filter (if bintype='StellarMassZ')
    '''
    if bintype=='Redshift':
    	xval, yval, len_input_index = stack_data(grid_ndarr, gridz, input_index, x0, 3700, 6700, ff=name)
    else:
    	xval, yval, len_input_index = stack_data(grid_ndarr, gridz, input_index, x0, 3700, 6700)
    table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
    asc.write(table0, full_path+'Composite_Spectra/'+bintype+'/'+instr+'_spectra_vals/'+name+'.txt',
        format='fixed_width', delimiter=' ')
#enddef

def plot_MMT_Ha(index_list=[], pp=None, title='', bintype='Redshift'):
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
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [])
    (tablenames, tablefluxes, nii6548fluxes, nii6583fluxes, ewlist, 
        ewposlist , ewneglist, ewchecklist, medianlist, pos_amplitudelist, 
        neg_amplitudelist) = table_arrays
    if index_list == []:
        index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'MMT')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('MMT')
    
    f, axarr = plt.subplots(5, 3)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)
    
    num=0
    # this for-loop stacks by filter
    for (match_index,ax,xmin0,xmax0,label,subtitle) in zip(index_list,ax_list,
                                                            xmin_list,xmax_list,
                                                            label_list, 
                                                            subtitle_list):
        shortlabel = ''
        if 'gamma' in label:
        	shortlabel = 'Hg'
        elif 'beta' in label:
            shortlabel = 'Hb'
        elif 'alpha' in label:
            shortlabel = 'Ha'
        #endif

        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)
        if len(input_index) < 2: 
            MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, num)
            print 'Not enough sources to stack (less than two)'
            num += 1 
            continue
        #endif

        try:
            print label, subtitle
            xval, yval, len_input_index = stack_data(grid_ndarr, gridz, input_index,
                x0, xmin0, xmax0, ff=subtitle, instr='MMT', AP_rows=halpha_maskarr)
            # xval, yval, len_input_index = stack_data(grid_ndarr, gridz, input_index,
            #     x0, xmin0, xmax0, ff=subtitle)
            if shortlabel=='Ha':
                label += ' ('+str(len_input_index[0]-len_input_index[1])+')'
            else:
                label += ' ('+str(len_input_index[0])+')'
            
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

            ax, flux, flux2, flux3, pos_flux, o1, o2, o3 = MMT_plotting.subplots_plotting(
                ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol)

            (ew, ew_emission, ew_absorption, ew_check, median, pos_amplitude, 
            	neg_amplitude) = MMT_twriting.Hg_Hb_Ha_tables(label, flux, 
            	o1, xval, pos_flux, dlambda)

            table_arrays = general_twriting.table_arr_appends(num, table_arrays, label, 
            	subtitle, flux, flux2, flux3, ew, ew_emission, ew_absorption, ew_check, 
            	median, pos_amplitude, neg_amplitude, 'MMT')
            
            #writing the spectra table
            if (num%3==0):
                if bintype=='Redshift':
                    write_spectral_table('MMT', grid_ndarr, gridz, input_index, x0, 
                        subtitle, full_path, shortlabel, bintype)
                elif bintype=='StellarMassZ':
                    write_spectral_table('MMT', grid_ndarr, gridz, input_index, x0, 
                        title[10:]+'_'+subtitle, full_path, shortlabel, bintype)

        except ValueError:
            print 'ValueError: none exist'
        #endtry
        
        if pos_flux and flux:
            ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, num, pos_flux, flux)
        elif not pos_flux and not flux:
            ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, num)
        else:
            print '>>>something\'s not right...'
        #endif

        num+=1
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
    if pp != None: return pp

    #writing the flux table
    table1 = Table([tablenames,tablefluxes,nii6548fluxes,nii6583fluxes],
        names=['type','flux','NII6548 flux','NII6583 flux'])
    asc.write(table1, full_path+'Composite_Spectra/Redshift/MMT_stacked_fluxes.txt',
        format='fixed_width', delimiter=' ')  

    #writing the EW table
    table2 = Table([tablenames,ewlist,ewposlist,ewneglist,ewchecklist,medianlist,pos_amplitudelist,neg_amplitudelist],
        names=['type','EW','EW_corr','EW_abs','ew check','median','pos_amplitude','neg_amplitude'])
    asc.write(table2, full_path+'Composite_Spectra/Redshift/MMT_stacked_ew.txt',
        format='fixed_width', delimiter=' ')  
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
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [])
    (tablenames, tablefluxes, nii6548fluxes, nii6583fluxes, ewlist, 
        ewposlist , ewneglist, ewchecklist, medianlist, pos_amplitudelist, 
        neg_amplitudelist) = table_arrays
    index_list = general_plotting.get_index_list2(stlr_mass, inst_str0, inst_dict, 'MMT')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('MMT')

    pp = PdfPages(full_path+'Composite_Spectra/StellarMass/MMT_all_20percbins.pdf')

    f, axarr = plt.subplots(5, 3)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)

    num=0
    # this for-loop stacks by stlr mass
    for (match_index,ax,xmin0,xmax0,label) in zip(index_list,ax_list,xmin_list,xmax_list,label_list):
        shortlabel = ''
        if 'gamma' in label:
            shortlabel = 'Hg'
        elif 'beta' in label:
            shortlabel = 'Hb'
        elif 'alpha' in label:
            shortlabel = 'Ha'
        #endif

        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)
        try:
            subtitle='stlrmass: '+str(min(stlr_mass[match_index]))+'-'+str(max(stlr_mass[match_index]))
            print label, subtitle
            xval, yval, len_input_index = stack_data(grid_ndarr, gridz, input_index,
                                                     x0, xmin0, xmax0)
            label += ' ('+str(len_input_index[0])+')'

            # calculating flux for NII emissions
            dlambda = xval[1] - xval[0]

            ax, flux, flux2, flux3, pos_flux, o1, o2, o3 = MMT_plotting.subplots_plotting(
                ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol)

            if (num%3==0):
                write_spectral_table('MMT', grid_ndarr, gridz, input_index, x0, 
                    subtitle[10:], full_path, shortlabel, 'StellarMass')

        except ValueError:
            print 'ValueError: none exist'
        #endtry

        if pos_flux and flux:
            ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, num, pos_flux, flux)
        elif not pos_flux and not flux:
            ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, num)
        else:
            print '>>>something\'s not right...'
        #endif

        num+=1
    #endfor
    f = general_plotting.final_plot_setup(f, r'MMT detections of H$\alpha$ emitters')
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
    pp = PdfPages(full_path+'Composite_Spectra/StellarMassZ/MMT_20_40_percbins.pdf')
    num=0
    n = 2 # how many redshifts we want to take into account (max 5, TODO(generalize this?))
    for stlrmassindex0 in stlrmass_index_list[:n*3]:
        if num%3 != 0:
            num += 1
            continue
        #endif
        
        title='stlrmass: '+str(min(stlr_mass[stlrmassindex0]))+'-'+str(max(stlr_mass[stlrmassindex0]))
        print '>>>', title

        # get one stlrmass bin per page
        temp_index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'MMT')
        index_list = []
        for index0 in temp_index_list:
            templist = [x for x in index0 if x in stlrmassindex0]
            index_list.append(templist)
        #endfor

        pp = plot_MMT_Ha(index_list, pp, title, 'StellarMassZ')
        num += 1
    #endfor
    pp.close()
#enddef

def plot_Keck_Ha(index_list=[], pp=None, title='', bintype='Redshift'):
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
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [])
    (tablenames, tablefluxes, nii6548fluxes, nii6583fluxes, ewlist, 
        ewposlist , ewneglist, ewchecklist, medianlist, pos_amplitudelist, 
        neg_amplitudelist) = table_arrays
    if index_list == []:
        index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'Keck')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('Keck')
    
    f, axarr = plt.subplots(3, 2)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)

    num=0
    for (match_index,ax,xmin0,xmax0,label,subtitle) in zip(index_list,ax_list,
                                                            xmin_list,xmax_list,
                                                            label_list, 
                                                            subtitle_list):
        shortlabel = ''
        if 'beta' in label:
            shortlabel = 'Hb'
        elif 'alpha' in label:
            shortlabel = 'Ha'
        #endif

        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'Keck')
        AP_match = np.array(AP_match, dtype=np.float32)
        
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match and gridz[x] != 0],dtype=np.int32)
        if len(input_index) < 2: 
            Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, num)
            print 'Not enough sources to stack (less than two)'
            num += 1 
            continue
        #endif

        try:
            print label, subtitle
            xval, yval, len_input_index = stack_data(grid_ndarr, gridz, input_index,
                x0, xmin0, xmax0, ff=subtitle)
            label += ' ('+str(len_input_index[0])+')'

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

            ax, flux, flux2, flux3, pos_flux, o1, o2, o3 = Keck_plotting.subplots_plotting(
                ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol, num)

            (ew, ew_emission, ew_absorption, ew_check, median, pos_amplitude, 
            	neg_amplitude) = Keck_twriting.Hb_Ha_tables(label, subtitle, flux, 
            	o1, xval, pos_flux, dlambda)
            table_arrays = general_twriting.table_arr_appends(num, table_arrays, label, 
            	subtitle, flux, flux2, flux3, ew, ew_emission, ew_absorption, ew_check, 
            	median, pos_amplitude, neg_amplitude, 'Keck')

            #writing the spectra table
            if (num%2==1):
                if bintype=='Redshift':
                    write_spectral_table('Keck', grid_ndarr, gridz, input_index, x0, 
                        subtitle, full_path, shortlabel, bintype)
                elif bintype=='StellarMassZ':
                    write_spectral_table('Keck', grid_ndarr, gridz, input_index, x0, 
                        title[10:]+'_'+subtitle, full_path, shortlabel, bintype)
        except ValueError:
            print 'ValueError: none exist'
        except RuntimeError:
            print 'Error - curve_fit failed'
        #endtry

        if pos_flux and flux:
            ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, num, pos_flux, flux)
        elif not pos_flux and not flux:
            ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, num)
        else:
            print '>>>something\'s not right...'
        #endif
        
        num+=1
    #endfor

    if title=='':
        f = general_plotting.final_plot_setup(f, r'Keck detections of H$\alpha$ emitters')

        #writing the flux table
        table1 = Table([tablenames,tablefluxes,nii6548fluxes,nii6583fluxes],
            names=['type','flux','NII6548 flux','NII6583 flux'])
        asc.write(table1, full_path+'Composite_Spectra/Redshift/Keck_stacked_fluxes.txt', 
            format='fixed_width', delimiter=' ')

        #writing the EW table
        table2 = Table([tablenames,ewlist,ewposlist,ewneglist,ewchecklist,medianlist,pos_amplitudelist,neg_amplitudelist], 
            names=['type','EW','EW_corr','EW_abs','ew check','median','pos_amplitude','neg_amplitude'])
        asc.write(table2, full_path+'Composite_Spectra/Redshift/Keck_stacked_ew.txt', 
            format='fixed_width', delimiter=' ')
    else:
        f = general_plotting.final_plot_setup(f, title)
    #endif

    if pp == None:
        plt.savefig(full_path+'Composite_Spectra/Redshift/Keck_stacked_spectra.pdf')
    else:
        pp.savefig()
    #endif
    plt.close()

    if pp != None: return pp

    #writing the flux table
    table1 = Table([tablenames,tablefluxes,nii6548fluxes,nii6583fluxes],
        names=['type','flux','NII6548 flux','NII6583 flux'])
    asc.write(table1, full_path+'Composite_Spectra/'+bintype+'/Keck_stacked_fluxes.txt', 
        format='fixed_width', delimiter=' ')

    #writing the EW table
    table2 = Table([tablenames,ewlist,ewposlist,ewneglist,ewchecklist,medianlist,pos_amplitudelist,neg_amplitudelist], 
        names=['type','EW','EW_corr','EW_abs','ew check','median','pos_amplitude','neg_amplitude'])
    asc.write(table2, full_path+'Composite_Spectra/'+bintype+'/Keck_stacked_ew.txt',
        format='fixed_width', delimiter=' ')
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
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [])
    (tablenames, tablefluxes, nii6548fluxes, nii6583fluxes, ewlist, 
        ewposlist , ewneglist, ewchecklist, medianlist, pos_amplitudelist, 
        neg_amplitudelist) = table_arrays
    index_list = general_plotting.get_index_list2(nan_stlr_mass, inst_str0, inst_dict, 'Keck')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('Keck', stlr=True)

    pp = PdfPages(full_path+'Composite_Spectra/StellarMass/Keck_all_20percbins.pdf')

    f, axarr = plt.subplots(5, 2)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)

    num=0
    # this for-loop stacks by stlr mass
    for (match_index,ax,xmin0,xmax0,label) in zip(index_list,ax_list,xmin_list,xmax_list,label_list):
        shortlabel = ''
        if 'beta' in label:
            shortlabel = 'Hb'
        elif 'alpha' in label:
            shortlabel = 'Ha'
        #endif

        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'Keck')
        AP_match = np.array([x for x in AP_match if x != 'INVALID_KECK'], dtype=np.float32)
        
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)
        try:
            subtitle='stlrmass: '+str(min(stlr_mass[match_index]))+'-'+str(max(stlr_mass[match_index]))
            print label, subtitle
            xval, yval, len_input_index = stack_data(grid_ndarr, gridz, input_index,
                                                     x0, xmin0, xmax0)
            label += ' ('+str(len_input_index[0])+')'

            # calculating flux for NII emissions
            dlambda = xval[1] - xval[0]

            ax, flux, flux2, flux3, pos_flux, o1, o2, o3 = Keck_plotting.subplots_plotting(
                ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol, num)

            if (num%2==1):
                write_spectral_table('Keck', grid_ndarr, gridz, input_index, x0, 
                        subtitle[10:], full_path, shortlabel, 'StellarMass')

        except ValueError:
            print 'ValueError: none exist'
        #endtry

        if pos_flux and flux:
            ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, num, pos_flux, flux)
        elif not pos_flux and not flux:
            ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, num)
        else:
            print '>>>something\'s not right...'
        #endif

        num+=1
    #endfor
    f = general_plotting.final_plot_setup(f, r'Keck detections of H$\alpha$ emitters')
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
    pp = PdfPages(full_path+'Composite_Spectra/StellarMassZ/Keck_20_40_60_80_100_percbins.pdf')
    num=0
    n = 5 # how many redshifts we want to take into account (max 5, TODO(generalize this?))
    for stlrmassindex0 in stlrmass_index_list[:n*2]:
        if num%2 != 0:
            num += 1
            continue
        #endif
        
        title='stlrmass: '+str(min(stlr_mass[stlrmassindex0]))+'-'+str(max(stlr_mass[stlrmassindex0]))
        print '>>>', title

        # get one stlrmass bin per page
        temp_index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'Keck')
        index_list = []
        for index0 in temp_index_list:
            templist = [x for x in index0 if x in stlrmassindex0]
            index_list.append(templist)
        #endfor

        pp = plot_Keck_Ha(index_list, pp, title, 'StellarMassZ')
        num += 1
    #endfor
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