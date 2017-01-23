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

import numpy as np, matplotlib.pyplot as plt
import plotting.hg_hb_ha_plotting as MMT_plotting
import plotting.hb_ha_plotting as Keck_plotting
import plotting.general_plotting as general_plotting
import writing_tables.hg_hb_ha_tables as MMT_twriting
import writing_tables.hb_ha_tables as Keck_twriting
import writing_tables.general_tables as general_twriting
from analysis.sdf_stack_data import stack_data
from create_ordered_AP_arrays import create_ordered_AP_arrays
from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table

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
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [])
    (tablenames, tablefluxes, nii6548fluxes, nii6583fluxes, ewlist, 
        ewposlist , ewneglist, ewchecklist, medianlist, pos_amplitudelist, 
        neg_amplitudelist) = table_arrays
    index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'MMT')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('MMT')
    
    f, axarr = plt.subplots(5, 3)
    f.set_size_inches(8, 11)
    ax_list = [axarr[0,0],axarr[0,1],axarr[0,2],axarr[1,0],axarr[1,1],
               axarr[1,2],axarr[2,0],axarr[2,1],axarr[2,2],axarr[3,0],
               axarr[3,1],axarr[3,2],axarr[4,0],axarr[4,1],axarr[4,2]]
    
    num=0
    for (match_index0,ax,xmin0,xmax0,label,subtitle) in zip(index_list,ax_list,
                                                            xmin_list,xmax_list,
                                                            label_list, 
                                                            subtitle_list):
        if 'gamma' in label:
            input_norm = HG_Y0[match_index0]
        elif 'beta' in label:
            input_norm = HB_Y0[match_index0]
        elif 'alpha' in label:
            input_norm = HA_Y0[match_index0]
        #endif

        good_index = [x for x in range(len(input_norm)) if
                      input_norm[x]!=-99.99999 and input_norm[x]!=-1
                      and input_norm[x]!=0]
        match_index = match_index0[good_index]
        
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
        if subtitle=='NB921' and 'alpha' in label:
            good_AP_match = np.array([x for x in range(len(AP_match)) if
                                      AP_match[x] in good_NB921_Halpha])
            AP_match = AP_match[good_AP_match]
        #endif
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)
        try:
            label += ' ('+str(len(input_index))+')'
            print label, subtitle
            xval, yval = stack_data(grid_ndarr, gridz, input_index,
                x0, xmin0, xmax0, subtitle)
            
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
            
        except ValueError:
            print 'ValueError: none exist'
        #endtry
        
        if pos_flux and flux:
            ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, num, pos_flux, flux)
        elif not pos_flux and not flux:
            ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, num)
        else:
            print 'something\'s not right...'
        #endif

        num+=1
    #endfor
    f = general_plotting.final_plot_setup(f, r'MMT detections of H$\alpha$ emitters')
    plt.savefig(full_path+'Spectra/Ha_MMT_stacked.pdf')
    plt.close()

    #writing the table
    table = Table([tablenames,tablefluxes,nii6548fluxes,nii6583fluxes],
                  names=['type','flux','NII6548 flux','NII6583 flux'])
    asc.write(table, full_path+'Spectra/Ha_MMT_stacked_fluxes.txt',
              format='fixed_width', delimiter=' ')  

    #writing the EW table
    table0 = Table([tablenames,ewlist,ewposlist,ewneglist,ewchecklist,medianlist,pos_amplitudelist,neg_amplitudelist], 
        names=['type','EW','EW_corr','EW_abs','ew check','median','pos_amplitude','neg_amplitude'])
    asc.write(table0, full_path+'Spectra/Ha_MMT_stacked_ew.txt', format='fixed_width', delimiter=' ')  
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
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [])
    (tablenames, tablefluxes, nii6548fluxes, nii6583fluxes, ewlist, 
        ewposlist , ewneglist, ewchecklist, medianlist, pos_amplitudelist, 
        neg_amplitudelist) = table_arrays
    index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'Keck')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('Keck')
    
    f, axarr = plt.subplots(3, 2)
    f.set_size_inches(8, 11)
    ax_list = [axarr[0,0],axarr[0,1],axarr[1,0],
               axarr[1,1],axarr[2,0],axarr[2,1]]

    num=0
    for (match_index0,ax,xmin0,xmax0,label,subtitle) in zip(index_list,ax_list,
                                                            xmin_list,xmax_list,
                                                            label_list, 
                                                            subtitle_list):
        if 'beta' in label:
            input_norm = HB_Y0[match_index0]
        elif 'alpha' in label:
            input_norm = HA_Y0[match_index0]
        #endif

        good_index = [x for x in range(len(input_norm)) if
                      input_norm[x]!=-99.99999 and input_norm[x]!=-1
                      and input_norm[x]!=0]
        match_index = match_index0[good_index]
        
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'Keck')
        AP_match = np.array(AP_match, dtype=np.float32)
        
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match and gridz[x] != 0],dtype=np.int32)
        try:
            label += ' ('+str(len(input_index))+')'
            print label, subtitle
            xval, yval = stack_data(grid_ndarr, gridz, input_index,
                x0, xmin0, xmax0, subtitle)

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
    plt.savefig(full_path+'Spectra/Ha_Keck_stacked.pdf')
    plt.close()

    #writing the table
    table = Table([tablenames,tablefluxes,nii6548fluxes,nii6583fluxes],
                  names=['type','flux','NII6548 flux','NII6583 flux'])
    asc.write(table, full_path+'Spectra/Ha_Keck_stacked_fluxes.txt', format='fixed_width', delimiter=' ')

    #writing the EW table
    table0 = Table([tablenames,ewlist,ewposlist,ewneglist,ewchecklist,medianlist,pos_amplitudelist,neg_amplitudelist], 
        names=['type','EW','EW_corr','EW_abs','ew check','median','pos_amplitude','neg_amplitude'])
    asc.write(table0, full_path+'Spectra/Ha_Keck_stacked_ew.txt', format='fixed_width', delimiter=' ')
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
                     'D.127','D.135','D.140','D.215','D.237','D.298']
inst_dict = {}
inst_dict['MMT'] = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,']
inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,',
                     'Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,']
tol = 3 #in angstroms, used for NII emission flux calculations

nbia = pyfits.open(full_path+'Catalogs/python_outputs/nbia_all_nsource.fits')
nbiadata = nbia[1].data
NAME0 = nbiadata['source_name']

zspec = asc.read(full_path+'Catalogs/nb_ia_zspec.txt',guess=False,
                 Reader=asc.CommentedHeader)
slit_str0 = np.array(zspec['slit_str0'])
inst_str0 = np.array(zspec['inst_str0'])

fout  = asc.read(full_path+'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.fout',
                 guess=False,Reader=asc.NoHeader)
stlr_mass = np.array(fout['col7'])

data_dict = create_ordered_AP_arrays()
AP = data_dict['AP']
HA_Y0 = data_dict['HA_Y0']
HB_Y0 = data_dict['HB_Y0']
HG_Y0 = data_dict['HG_Y0']

print '### looking at the MMT grid'
griddata = asc.read(full_path+'Spectra/spectral_MMT_grid_data.txt',guess=False)
gridz  = np.array(griddata['ZSPEC'])
gridap = np.array(griddata['AP'])
grid   = pyfits.open(full_path+'Spectra/spectral_MMT_grid.fits')
grid_ndarr = grid[0].data
grid_hdr   = grid[0].header
CRVAL1 = grid_hdr['CRVAL1']
CDELT1 = grid_hdr['CDELT1']
NAXIS1 = grid_hdr['NAXIS1']
x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1)

print '### plotting MMT_Ha'
plot_MMT_Ha()
grid.close()

print '### looking at the Keck grid'
griddata = asc.read(full_path+'Spectra/spectral_Keck_grid_data.txt',guess=False)
gridz  = np.array(griddata['ZSPEC'])
gridap = np.array(griddata['AP'])
grid   = pyfits.open(full_path+'Spectra/spectral_Keck_grid.fits')
grid_ndarr = grid[0].data
grid_hdr   = grid[0].header
CRVAL1 = grid_hdr['CRVAL1']
CDELT1 = grid_hdr['CDELT1']
NAXIS1 = grid_hdr['NAXIS1']
x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1)

print '### plotting Keck_Ha'
plot_Keck_Ha()
grid.close()

nbia.close()
print '### done'
#endmain