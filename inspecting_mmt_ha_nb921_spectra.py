"""
NAME:
    inspecting_mmt_ha_nb921_spectra.py

PURPOSE:
    This code creates a PDF file with 9 subplots per page, inspecting
    the behavior of MMT Halpha NB921 sources around the Halpha emission
    line. Useful for manually masking unreliable spectra due to 
    faulty sky subtraction.

INPUTS:
    'Spectra/spectral_MMT_grid_data.txt'
    'Spectra/spectral_MMT_grid.fits'
    'Spectra/spectral_MMT_grid_data.unmasked.txt'
    'Spectra/spectral_MMT_grid.unmasked.fits'
    'Catalogs/python_outputs/nbia_all_nsource.fits'
    'Catalogs/nb_ia_zspec.txt'
    'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.fout'

OUTPUTS:
    'Composite_Spectra/all_MMT_HaNB921_spectra.pdf'
"""

import numpy as np, numpy.ma as ma, matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import plotting.general_plotting as general_plotting
from astropy.io import fits as pyfits, ascii as asc
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


def remove_empty_plots(tempdiff, f, current_axis):
    '''
    Removes empty plots at the end of the PDF
    '''
    if tempdiff != 0:
        for ii in np.arange(8, 4-tempdiff, -1):
            f.get_axes()[ii].axis('off')
        #endfor
    #endif
#enddef


def masked_MMT_grid(full_path, NAME0, inst_str0, inst_dict, AP):
    '''
    Obtains the masked MMT grid data
    '''
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

    
    index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'MMT')
    match_index = index_list[3]

    AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
    input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                            AP_match],dtype=np.int32)

    ff='NB921'
    ndarr = grid_ndarr[input_index]
    zspec = gridz[input_index]

    ndarr[np.where(ndarr.mask==True)] = np.nan

    x_rest   = np.arange(3700, 6700, 0.1)
    new_grid = np.ndarray(shape=(len(ndarr), len(x_rest)))
    #deshifting to rest-frame wavelength
    for (row_num, ii) in zip(range(len(ndarr)), input_index):
        #normalizing
        spec_test = ndarr[row_num]

        #interpolating a function for rest-frame wavelength and normalized y
        x_test = x0/(1.0+zspec[row_num])
        f = interp1d(x_test, spec_test, bounds_error=False, fill_value=np.nan)

        #finding the new rest-frame wavelength values from the interpolation
        #and putting them into the 'new grid'
        spec_interp = f(x_rest)
        new_grid[row_num] = spec_interp
    #endfor
    return new_grid, gridap, input_index, x_rest, match_index
#enddef


def unmasked_MMT_grid(full_path, NAME0, inst_str0, inst_dict, AP):
    '''
    Obtains the unmasked MMT grid data
    '''
    print '### looking at the unmasked (_um) MMT grid'
    griddata = asc.read(full_path+'Spectra/spectral_MMT_grid_data.unmasked.txt',guess=False)
    gridz  = np.array(griddata['ZSPEC']) ##used
    gridap = np.array(griddata['AP']) ##used
    grid   = pyfits.open(full_path+'Spectra/spectral_MMT_grid.unmasked.fits')
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

    
    index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'MMT')
    match_index = index_list[3]

    AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
    input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                            AP_match],dtype=np.int32)

    ff='NB921'
    ndarr = grid_ndarr[input_index]
    zspec = gridz[input_index]

    ndarr[np.where(ndarr.mask==True)] = np.nan

    x_rest   = np.arange(3700, 6700, 0.1)
    new_grid_um = np.ndarray(shape=(len(ndarr), len(x_rest)))
    #deshifting to rest-frame wavelength
    for (row_num, ii) in zip(range(len(ndarr)), input_index):
        #normalizing
        spec_test = ndarr[row_num]

        #interpolating a function for rest-frame wavelength and normalized y
        x_test = x0/(1.0+zspec[row_num])
        f = interp1d(x_test, spec_test, bounds_error=False, fill_value=np.nan)

        #finding the new rest-frame wavelength values from the interpolation
        #and putting them into the 'new grid'
        spec_interp = f(x_rest)
        new_grid_um[row_num] = spec_interp
    #endfor
    return new_grid_um
#enddef


def plotting_spectra(full_path, new_grid, new_grid_um, gridap, input_index, x_rest, match_index):
    '''
    Plots the behavior of MMT Halpha NB921 sources around the Halpha emission
    line.
    
    Red dashed lines denote unmasked data; blue solid lines denote masked
    data (what will be used).
    '''
    pp = PdfPages(full_path+'Composite_Spectra/all_MMT_HaNB921_spectra.pdf')
    # get 9 per page
    for row, row_ii, AP in zip(new_grid, range(len(new_grid)), gridap[input_index]):
        if row_ii %9 == 0: # new page
            f, axarr = plt.subplots(3, 3)
            f.set_size_inches(12, 12)
            ax_list = np.ndarray.flatten(axarr)
        #endif
        
        axnum = row_ii%9
        current_axis = f.get_axes()[axnum]
        
        current_axis.plot(x_rest, row/1e-17, 'b', label='masked')
        current_axis.plot(x_rest, new_grid_um[row_ii]/1e-17, 'r--', alpha=0.6, label='unmasked')
        current_axis.set_xlim(6503, 6623)
        ymaxval = max(current_axis.get_ylim())
        current_axis.plot([6563,6563], [0,ymaxval],'k',alpha=0.7,zorder=1)
        current_axis.text(0.03,0.97,AP,transform=current_axis.transAxes,fontsize=7,ha='left',
                va='top')

        if axnum==8 or row_ii==len(new_grid)-1: #last on page
            if row_ii==len(match_index)-1: #last on last page of file
                remove_empty_plots((8-axnum), f, current_axis)
            #endif
            pp.savefig()
            plt.close()
        #endif
    #endfor
    pp.close()
#enddef


def main():
    full_path = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'

    nbia = pyfits.open(full_path+'Catalogs/python_outputs/nbia_all_nsource.fits')
    nbiadata = nbia[1].data
    NAME0 = nbiadata['source_NAME']

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

    inst_dict = {} ##used
    inst_dict['MMT'] = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,']
    inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,',
                         'Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,']
    tol = 3 #in angstroms, used for NII emission flux calculations ##used

    new_grid, gridap, input_index, x_rest, match_index = masked_MMT_grid(full_path, NAME0, inst_str0, inst_dict, AP)
    new_grid_um = unmasked_MMT_grid(full_path, NAME0, inst_str0, inst_dict, AP)

    plotting_spectra(full_path, new_grid, new_grid_um, gridap, input_index, x_rest, match_index)
    
    print 'done'
#enddef


if __name__ == '__main__':
    main()