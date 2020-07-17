"""
NAME:
    write_spectral_coverage_table.py

PURPOSE:
    This code 

    Depends on combine_spectral_data.py and stack_spectral_data.py

INPUTS:
    FULL_PATH+'Catalogs/NB_IA_emitters.nodup.colorrev.fix.fits'
    FULL_PATH+'Catalogs/nb_ia_zspec.txt'
    FULL_PATH+'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX.fout'
    FULL_PATH+'Spectra/spectral_MMT_grid_data.txt'
    FULL_PATH+'Spectra/spectral_MMT_grid.fits'
    FULL_PATH+'Composite_Spectra/StellarMass/MMT_all_five_data.txt'
    FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt'
    FULL_PATH+'Spectra/spectral_Keck_grid_data.txt'
    FULL_PATH+'Spectra/spectral_Keck_grid.fits'
    FULL_PATH+'Composite_Spectra/StellarMass/Keck_all_five_data.txt'
    FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt'

OUTPUTS:
    FULL_PATH+'Composite_Spectra/MMT_spectral_coverage.txt'
    FULL_PATH+'Composite_Spectra/Keck_spectral_coverage.txt'
"""
from __future__ import print_function

import numpy as np, re
from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table

import config
import plotting.general_plotting as general_plotting
from MACT_utils import exclude_AGN, get_filt_arr, get_stlrmassbinZ_arr, get_spectral_cvg_MMT, find_nearest_iis


def get_stlrmassbin_arr(stlr_mass, min_mass, max_mass):
    '''
    '''
    stlrmassbin = np.array([], dtype=int)
    for m in stlr_mass:
        for ii in range(len(min_mass)):
            if m >= min_mass[ii] and m <= max_mass[ii]:
                stlrmassbin = np.append(stlrmassbin, ii+1)
                break
    #endfor

    return stlrmassbin


def write_MMT_table(inst_str0, ID, zspec0, NAME0, AP, stlr_mass, filt_arr, 
    stlr_mass_orig, inst_str0_orig, inst_dict, MMT_LMIN0, MMT_LMAX0, NAME0_orig):
    '''
    '''
    # reading in grid tables
    griddata = asc.read(config.FULL_PATH+'Spectra/spectral_MMT_grid_data.txt',guess=False)
    gridz  = np.array(griddata['ZSPEC']) ##used
    gridap = np.array(griddata['AP']) ##used

    grid   = pyfits.open(config.FULL_PATH+'Spectra/spectral_MMT_grid.fits')
    grid_ndarr = grid[0].data ##used
    grid_hdr   = grid[0].header
    CRVAL1 = grid_hdr['CRVAL1']
    CDELT1 = grid_hdr['CDELT1']
    NAXIS1 = grid_hdr['NAXIS1']
    x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1)

    # getting indices relevant to MMT
    mmt_ii = [x for x in range(len(inst_str0)) if 'MMT' in inst_str0[x] or 'merged' in inst_str0[x]]
    ID = ID[mmt_ii]
    zspec0 = zspec0[mmt_ii]
    NAME0 = NAME0[mmt_ii]
    AP = AP[mmt_ii]
    stlr_mass = stlr_mass[mmt_ii]
    filt_arr = filt_arr[mmt_ii]
    AP = np.array([ap[:5] for ap in AP])
    MMT_LMIN0 = MMT_LMIN0[mmt_ii]
    MMT_LMAX0 = MMT_LMAX0[mmt_ii]

    # getting stlrmassbin cols for the table
    tab0 = asc.read(config.FULL_PATH+'Composite_Spectra/StellarMass/MMT_all_five_data.txt')
    min_mass = np.array(tab0['min_stlrmass'])
    max_mass = np.array(tab0['max_stlrmass'])
    stlrmassbin = get_stlrmassbin_arr(stlr_mass, min_mass, max_mass)

    # getting stlrmassZbin cols for the table
    tab1 = asc.read(config.FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt')
    stlrmassbinZ = get_stlrmassbinZ_arr(filt_arr, stlr_mass, tab1['filter'],
        tab1['min_stlrmass'], tab1['max_stlrmass'], 'MMT')
    
    # setting 'YES' and 'NO' and 'MASK' coverage values
    match_ii = np.array([])
    for ii in range(len(AP)):
        match_ii = np.append(match_ii, np.where(gridap == AP[ii])[0])
    #endfor
    match_ii = np.array(match_ii, dtype=np.int32)
    HG_cvg, HB_cvg, HA_cvg = get_spectral_cvg_MMT(MMT_LMIN0, MMT_LMAX0, zspec0, grid_ndarr[match_ii], x0)

    tt_mmt = Table([ID, NAME0, AP, zspec0, filt_arr, stlr_mass, stlrmassbin, stlrmassbinZ,
        HG_cvg, HB_cvg, HA_cvg, MMT_LMIN0, MMT_LMAX0], 
        names=['ID', 'NAME', 'AP', 'z', 'filter', 'stlrmass', 'stlrmassbin', 'stlrmassZbin',
        'HG_cvg', 'HB_cvg', 'HA_cvg', 'LMIN0', 'LMAX0']) 
    asc.write(tt_mmt, config.FULL_PATH+'Composite_Spectra/MMT_spectral_coverage.txt', format='fixed_width', delimiter=' ', overwrite=True)


def get_spectral_cvg_Keck(KECK_LMIN0, KECK_LMAX0):
    '''
    '''
    HB = np.array([])
    HA = np.array([])
    for lmin0, lmax0 in zip(KECK_LMIN0, KECK_LMAX0):
        if lmin0 < 0:
            HB = np.append(HB, 'NO')
            HA = np.append(HA, 'NO')
        else:
            if lmin0 < config.HB_VAL and lmax0 > config.HB_VAL:
                HB = np.append(HB, 'YES')
            else:
                HB = np.append(HB, 'NO')
                
            if lmin0 < config.HA_VAL and lmax0 > config.HA_VAL:
                HA = np.append(HA, 'YES')
            else:
                HA = np.append(HA, 'NO')
    #endfor
    return HB, HA


def write_Keck_table(inst_str0, ID, zspec0, NAME0, AP, stlr_mass, filt_arr, 
    stlr_mass_orig, inst_str0_orig, inst_dict, KECK_LMIN0, KECK_LMAX0, NAME0_orig):
    '''
    '''
    # reading in grid tables
    griddata = asc.read(config.FULL_PATH+'Spectra/spectral_Keck_grid_data.txt',guess=False)
    gridz  = np.array(griddata['ZSPEC']) ##used
    gridap = np.array(griddata['AP']) ##used

    grid   = pyfits.open(config.FULL_PATH+'Spectra/spectral_Keck_grid.fits')
    grid_ndarr = grid[0].data ##used
    grid_hdr   = grid[0].header
    CRVAL1 = grid_hdr['CRVAL1']
    CDELT1 = grid_hdr['CDELT1']
    NAXIS1 = grid_hdr['NAXIS1']
    x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1)

    # getting indices relevant to Keck
    keck_ii = [x for x in range(len(inst_str0)) if ('Keck' in inst_str0[x] or 'merged' in inst_str0[x]) and 'MMT' not in inst_str0[x]] # MMT,Keck, means MMT
    ID = ID[keck_ii]
    zspec0 = zspec0[keck_ii]
    NAME0 = NAME0[keck_ii]
    AP = AP[keck_ii]
    stlr_mass = stlr_mass[keck_ii]
    filt_arr = filt_arr[keck_ii]
    KECK_LMIN0 = KECK_LMIN0[keck_ii]
    KECK_LMAX0 = KECK_LMAX0[keck_ii]
    AP = np.array([x if len(x) == 6 else x[6:] for x in AP], dtype=np.float64)

    # getting stlrmassbin cols for the table
    tab0 = asc.read(config.FULL_PATH+'Composite_Spectra/StellarMass/Keck_all_five_data.txt')
    min_mass = np.array(tab0['min_stlrmass'])
    max_mass = np.array(tab0['max_stlrmass'])
    stlrmassbin = get_stlrmassbin_arr(stlr_mass, min_mass, max_mass)

    # getting stlrmassZbin cols for the table
    tab1 = asc.read(config.FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt')
    stlrmassbinZ = get_stlrmassbinZ_arr(filt_arr, stlr_mass, tab1['filter'],
        tab1['min_stlrmass'], tab1['max_stlrmass'], 'Keck')

    # setting 'YES' and 'NO' and 'MASK' coverage values
    HB_cvg, HA_cvg = get_spectral_cvg_Keck(KECK_LMIN0, KECK_LMAX0)

    # creating/writing the table
    tt_keck = Table([ID, NAME0, AP, zspec0, filt_arr, stlr_mass, stlrmassbin, stlrmassbinZ, HB_cvg, HA_cvg, KECK_LMIN0, KECK_LMAX0], 
        names=['ID', 'NAME', 'AP', 'z', 'filter', 'stlrmass', 'stlrmassbin', 'stlrmassZbin', 'HB_cvg', 'HA_cvg', 'LMIN0', 'LMAX0']) 
    asc.write(tt_keck, config.FULL_PATH+'Composite_Spectra/Keck_spectral_coverage.txt', format='fixed_width', delimiter=' ', overwrite=True)


def main():
    inst_dict = config.inst_dict

    nbia = pyfits.open(config.FULL_PATH+config.NB_IA_emitters_cat)
    nbiadata = nbia[1].data
    NAME0_orig = np.array(nbiadata['NAME'])

    zspec = asc.read(config.FULL_PATH+'Catalogs/nb_ia_zspec.txt',guess=False,
                     Reader=asc.CommentedHeader)
    zspec0 = np.array(zspec['zspec0'])

    # limit all data to valid Halpha NB emitters only
    ha_ii = np.array([x for x in range(len(NAME0_orig)) if 'Ha-NB' in NAME0_orig[x] and (zspec0[x] < 9.0 and zspec0[x] > 0.0)])
    ha_ii = exclude_AGN(ha_ii, NAME0_orig)
    NAME0 = NAME0_orig[ha_ii]
    zspec0 = zspec0[ha_ii]
    
    ID        = np.array(zspec['ID0'][ha_ii])
    inst_str0_orig = np.array(zspec['inst_str0'])
    inst_str0 = inst_str0_orig[ha_ii]

    fout  = asc.read(config.FULL_PATH+'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX.fout',
                     guess=False,Reader=asc.NoHeader)
    stlr_mass_orig = np.array(fout['col7'])
    stlr_mass = stlr_mass_orig[ha_ii]

    data_dict = config.data_dict
    AP = data_dict['AP'][ha_ii]
    MMT_LMIN0 = data_dict['MMT_LMIN0'][ha_ii]
    MMT_LMAX0 = data_dict['MMT_LMAX0'][ha_ii]
    KECK_LMIN0 = data_dict['KECK_LMIN0'][ha_ii]
    KECK_LMAX0 = data_dict['KECK_LMAX0'][ha_ii]
    filt_arr = get_filt_arr(NAME0)

    print('writing MMT table...')
    write_MMT_table(inst_str0, ID, zspec0, NAME0, AP, stlr_mass, filt_arr, 
        stlr_mass_orig, inst_str0_orig, inst_dict, MMT_LMIN0, MMT_LMAX0, NAME0_orig)
    print('finished writing MMT table')

    print('writing Keck table...')
    bad_iis    = np.array([x for x in range(len(NAME0)) if ('Ha-NB816' not in NAME0[x] 
        and 'Ha-NB921' not in NAME0[x] and 'Ha-NB973' not in NAME0[x])])
    NAME0      = np.array([NAME0[x] for x in range(len(NAME0)) if x not in bad_iis])
    zspec0     = np.array([zspec0[x] for x in range(len(zspec0)) if x not in bad_iis])
    ID         = np.array([ID[x] for x in range(len(ID)) if x not in bad_iis])
    inst_str0  = np.array([inst_str0[x] for x in range(len(inst_str0)) if x not in bad_iis])
    AP         = np.array([AP[x] for x in range(len(AP)) if x not in bad_iis])
    stlr_mass  = np.array([stlr_mass[x] for x in range(len(stlr_mass)) if x not in bad_iis])
    KECK_LMIN0 = np.array([KECK_LMIN0[x] for x in range(len(KECK_LMIN0)) if x not in bad_iis])
    KECK_LMAX0 = np.array([KECK_LMAX0[x] for x in range(len(KECK_LMAX0)) if x not in bad_iis])
    filt_arr   = np.array([filt_arr[x] for x in range(len(filt_arr)) if x not in bad_iis])
    write_Keck_table(inst_str0, ID, zspec0, NAME0, AP, stlr_mass, filt_arr, 
        stlr_mass_orig, inst_str0_orig, inst_dict, KECK_LMIN0, KECK_LMAX0, NAME0_orig)
    print('finished writing Keck table')

if __name__ == '__main__':
    main()