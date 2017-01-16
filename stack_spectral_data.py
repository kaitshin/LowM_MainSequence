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
from analysis.sdf_spectra_fit import find_nearest, get_best_fit, get_best_fit2, get_best_fit3
from analysis.sdf_stack_data import stack_data
import plotting.hg_hb_ha_plotting as MMT_plotting
import plotting.general_plotting as general_plotting
import writing_tables.general_tables as general_twriting
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


#----plot_MMT_Ha-------------------------------------------------------------#
# o Calls get_name_index_matches in order to get the indexes at which
#   there is the particular name match and instrument and then creates a
#   master index list.
# o Creates a pdf (8"x11") with 5x3 subplots for different lines and filter
#   combinations.
# o Then, the code iterates through every subplot in row-major filter-line
#   order. Using only the 'good' indexes, finds 'match_index'. With those
#   indexes of AP and inst_str0, calls AP_match.
# o For NB921 Halpha, does a cross-match to ensure no 'invalid' point is
#   plotted.
# o Except for NB973 Halpha, the graph is 'de-redshifted' in order to have
#   the spectral line appear in the subplot. The values to plot are called
#   from sdf_stack_data.stack_data
# o get_best_fit is called to obtain the best-fit spectra, overlay the
#   best fit, and then calculate the flux
# o Additionally, a line is plotted at the value at which the emission line
#   should theoretically be (based on which emission line it is).
# o The yscale is fixed for each filter type (usually the yscale values of
#   the Halpha subplot).
# o Minor ticks are set on, lines and filters are labeled, and with the
#   line label is another label for the number of stacked sources that were
#   used to produce the emission graph.
# o At the end of all the iterations, the plot is saved and closed.
# o The fluxes are also output to a separate .txt file.
#----------------------------------------------------------------------------#
def plot_MMT_Ha():
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

            ew_emission = 0
            ew_absorption = 0
            ew_check = 0
            median = 0
            pos_amplitude = 0
            neg_amplitude = 0
            if 'alpha' in label:
                ew = flux/o1[3]
                ew_emission = ew
                ew_check = ew
                median = o1[3]
                pos_amplitude = o1[0]
                neg_amplitude = 0
            else:
                pos0 = o1[6]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)
                neg0 = o1[3]*np.exp(-0.5*((xval-o1[4])/o1[5])**2)

                ew = flux/o1[6]
                median = o1[6]
                pos_amplitude = o1[0]
                neg_amplitude = o1[3]

                if (neg_amplitude > 0): 
                    neg_amplitude = 0
                    ew = pos_flux/o1[6]
                    ew_emission = ew
                    ew_check = ew
                else:
                    idx_small = np.where(np.absolute(xval - o1[1]) <= 2.5*o1[2])[0]
                    pos_corr = np.sum(dlambda * (pos0[idx_small] - o1[6]))
                    ew_emission = pos_corr / o1[6]
                    neg_corr = np.sum(dlambda * neg0[idx_small])
                    ew_absorption = neg_corr / o1[6]
                    ew_check = ew_emission + ew_absorption
            #endif
            
        except ValueError:
            print 'ValueError: none exist'
        #endtry
        
        table_arrays = general_twriting.table_arr_appends(num, table_arrays, label, subtitle, flux, flux2, flux3, ew, ew_emission, ew_absorption, ew_check, median, pos_amplitude, neg_amplitude, 'MMT')
        
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
    Calls get_name_index_matches in order to get the indexes at which
    there is the particular name match and instrument and then creates a
    master index list.
    
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
            o1 = get_best_fit(xval, yval, label)
            if not (subtitle=='NB816' and num%2==0):
                ax.plot(xval, yval/1E-17, zorder=2)
            #endif

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

            (flux, flux2, flux3, ew, ew_emission, ew_absorption, ew_check, 
                median, pos_amplitude, neg_amplitude) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            if 'alpha' in label:
                ax.plot(xval, (o1[3]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2))/1E-17, 'r--', zorder=3)
                flux = np.sum(dlambda * (o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)))
                pos_flux = flux
                ew = flux/o1[3]
                median = o1[3]
                pos_amplitude = o1[0]
                neg_amplitude = 0
                ew_emission = ew
                ew_check = ew
                
                peak_idx2_left  = find_nearest(xval, 6548.1-tol)
                peak_idx2_right = find_nearest(xval, 6548.1+tol)
                xval2=xval[peak_idx2_left:peak_idx2_right]
                yval2=yval[peak_idx2_left:peak_idx2_right]
                o2 = get_best_fit2(xval2, yval2, 6548.1, label)
                flux2 = np.sum(dlambda * (o2[0]*np.exp(-0.5*((xval2-o2[1])/o2[2])**2)))
                ax.plot(xval2, (o2[3]+o2[0]*np.exp(-0.5*((xval2-o2[1])/o2[2])**2))/1E-17, 'g,', zorder=3)

                peak_idx3_left = find_nearest(xval, 6583.6-tol)
                peak_idx3_right = find_nearest(xval, 6583.6+tol)
                xval3=xval[peak_idx3_left:peak_idx3_right]
                yval3=yval[peak_idx3_left:peak_idx3_right]
                o3 = get_best_fit2(xval3, yval3, 6583.6, label)
                flux3 = np.sum(dlambda * (o3[0]*np.exp(-0.5*((xval3-o3[1])/o3[2])**2)))
                ax.plot(xval3, (o3[3]+o3[0]*np.exp(-0.5*((xval3-o3[1])/o3[2])**2))/1E-17, 'g,', zorder=3)
            elif 'beta' in label and subtitle!='NB816':
                o1 = get_best_fit3(xval, yval, label)
                pos0 = o1[6]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)
                neg0 = o1[3]*np.exp(-0.5*((xval-o1[4])/o1[5])**2)
                func0 = pos0 + neg0
                ax.plot(xval, func0/1E-17, 'r--', zorder=3)

                idx_small = np.where(np.absolute(xval - o1[1]) <= 2.5*o1[2])[0]

                pos_flux = np.sum(dlambda * (pos0[idx_small] - o1[6]))
                flux = np.sum(dlambda * (func0[idx_small] - o1[6]))
                flux2 = 0
                flux3 = 0

                ew = flux/o1[6]
                median = o1[6]
                pos_amplitude = o1[0]
                neg_amplitude = o1[3]
                if (neg_amplitude > 0): 
                    neg_amplitude = 0
                    ew = pos_flux/o1[6]
                    ew_emission = ew
                    ew_check = ew
                else:
                    pos_corr = np.sum(dlambda * (pos0[idx_small] - o1[6]))
                    ew_emission = pos_corr / o1[6]
                    neg_corr = np.sum(dlambda * neg0[idx_small])
                    ew_absorption = neg_corr / o1[6]
                    ew_check = ew_emission + ew_absorption
                #endif
            #endif

            ax.set_xlim(xmin0, xmax0)
            ax.set_ylim(ymin=0)
        except ValueError:
            print 'ValueError: none exist'
        #endtry

        table_arrays = general_twriting.table_arr_appends(num, table_arrays, label, subtitle, flux, flux2, flux3, ew, ew_emission, ew_absorption, ew_check, median, pos_amplitude, neg_amplitude, 'Keck')

        ax.text(0.03,0.97,label,transform=ax.transAxes,fontsize=7,ha='left',
                va='top')

        if not (subtitle=='NB816' and num%2==0):
            ax.text(0.97,0.97,'flux_before='+'{:.4e}'.format((pos_flux))+
                '\nflux='+'{:.4e}'.format((flux)),transform=ax.transAxes,fontsize=7,ha='right',va='top')
        if num%2==0:
            ax.set_title(subtitle,fontsize=8,loc='left')
        elif num%2==1:
            ymaxval = max(ax.get_ylim())
            plt.setp([a.set_ylim(ymax=ymaxval) for a in ax_list[num-1:num]])
            if subtitle != 'NB816':
                ax_list[num-1].plot([4861,4861],[0,ymaxval],'k',alpha=0.7,zorder=1)
            ax_list[num].plot([6563,6563], [0,ymaxval],'k',alpha=0.7,zorder=1)
            ax_list[num].plot([6548,6548],[0,ymaxval], 'k:',alpha=0.4,zorder=1)
            ax_list[num].plot([6583,6583],[0,ymaxval], 'k:',alpha=0.4,zorder=1)
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