"""
NAME:
    write_spectral_coverage_table.py

PURPOSE:
    This code 

    Depends on combine_spectral_data.py and stack_spectral_data.py

INPUTS:
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

import numpy as np, re
import plotting.general_plotting as general_plotting
from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table
from create_ordered_AP_arrays import create_ordered_AP_arrays

def get_filt_arr(NAME0):
    '''
    '''
    filt_arr = np.array([])
    for name in NAME0:
        if name.count('Ha-NB') > 1: 
            tempname = ''
            for m in re.finditer('Ha', name):
                tempname += name[m.start()+3:m.start()+8]
                tempname += ','
            filt_arr = np.append(filt_arr, tempname)
        else:
            i = name.find('Ha-NB')
            filt_arr = np.append(filt_arr, name[i+3:i+8])

    return filt_arr


def find_nearest_iis(array, value):
    '''
    '''
    idx_closest = (np.abs(array-value)).argmin()
    if array[idx_closest] > value and idx_closest != 0:
        return [idx_closest-1, idx_closest]
    else:
        return [idx_closest, idx_closest+1]


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


def get_stlrmassbinZ_arr(filt_arr, stlr_mass, tab1, instr):
    '''
    '''
    stlrmassZbin = np.array([])
    for ff, m in zip(filt_arr, stlr_mass):
        if instr=='Keck' and ff=='NB816':
            stlrmassZbin = np.append(stlrmassZbin, 'N/A')
        elif instr=='MMT' and ff=='NB704,NB711,':
            assigned_bin = ''
            for ff0 in ['NB704', 'NB711']:
                good_filt_iis = np.array([x for x in range(len(tab1)) if tab1['filter'][x]==ff0])
                min_mass = np.array(tab1['min_stlrmass'][good_filt_iis])
                max_mass = np.array(tab1['max_stlrmass'][good_filt_iis])
                for ii in range(len(good_filt_iis)):
                    if m >= min_mass[ii] and m <= max_mass[ii]:
                        assigned_bin += str(ii+1)+'-'+ff0+','
            stlrmassZbin = np.append(stlrmassZbin, assigned_bin)
        else:
            good_filt_iis = np.array([x for x in range(len(tab1)) if tab1['filter'][x]==ff])
            min_mass = np.array(tab1['min_stlrmass'][good_filt_iis])
            max_mass = np.array(tab1['max_stlrmass'][good_filt_iis])
            for ii in range(len(good_filt_iis)):
                if m >= min_mass[ii] and m <= max_mass[ii]:
                    stlrmassZbin = np.append(stlrmassZbin, str(ii+1)+'-'+ff)
    #endfor

    return stlrmassZbin


def get_spectral_cvg_MMT(MMT_LMIN0, MMT_LMAX0, zspec0, grid_ndarr_match_ii, x0):
    '''
    '''
    HG = np.array([])
    HB = np.array([])
    HA = np.array([])
    for lmin0, lmax0, row, z in zip(MMT_LMIN0, MMT_LMAX0, grid_ndarr_match_ii, zspec0):
        hg_near_iis = find_nearest_iis(x0, 4341*(1+z))
        hb_near_iis = find_nearest_iis(x0, 4861*(1+z))
        ha_near_iis = find_nearest_iis(x0, 6563*(1+z))

        if lmin0 < 0:
            HG = np.append(HG, 'NO')
            HB = np.append(HB, 'NO')
            HA = np.append(HA, 'NO')
        else:
            if lmin0 < HG_VAL and lmax0 > HG_VAL:
                if np.average(row[hg_near_iis])==0:
                    HG = np.append(HG, 'MASK')
                else:
                    HG = np.append(HG, 'YES')
            else:
                HG = np.append(HG, 'NO')
            
            if lmin0 < HB_VAL and lmax0 > HB_VAL:
                if np.average(row[hb_near_iis])==0:
                    HB = np.append(HB, 'MASK')
                else:
                    HB = np.append(HB, 'YES')
            else:
                HB = np.append(HB, 'NO')
                
            if lmin0 < HA_VAL and lmax0 > HA_VAL:
                if np.average(row[ha_near_iis])==0:
                    HA = np.append(HA, 'MASK')
                else:
                    HA = np.append(HA, 'YES')
            else:
                HA = np.append(HA, 'NO')
    #endfor
    return HG, HB, HA


def write_MMT_table(inst_str0, ID, zspec0, NAME0, AP, stlr_mass, filt_arr, 
    stlr_mass_orig, inst_str0_orig, inst_dict, MMT_LMIN0, MMT_LMAX0, NAME0_orig):
    '''
    '''
    # reading in grid tables
    griddata = asc.read(FULL_PATH+'Spectra/spectral_MMT_grid_data.txt',guess=False)
    gridz  = np.array(griddata['ZSPEC']) ##used
    gridap = np.array(griddata['AP']) ##used

    grid   = pyfits.open(FULL_PATH+'Spectra/spectral_MMT_grid.fits')
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
    tab0 = asc.read(FULL_PATH+'Composite_Spectra/StellarMass/MMT_all_five_data.txt')
    min_mass = np.array(tab0['min_stlrmass'])
    max_mass = np.array(tab0['max_stlrmass'])
    stlrmassbin = get_stlrmassbin_arr(stlr_mass, min_mass, max_mass)

    # getting stlrmassZbin cols for the table
    tab1 = asc.read(FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt')
    stlrmassbinZ = stlrmassbinZ = get_stlrmassbinZ_arr(filt_arr, stlr_mass, tab1, 'MMT')
    
    # setting 'YES' and 'NO' and 'MASK' coverage values
    match_ii = np.array([])
    for ii in range(len(AP)):
        match_ii = np.append(match_ii, np.where(gridap == AP[ii])[0])
    #endfor
    match_ii = np.array(match_ii, dtype=np.int32)
    HG_cvg, HB_cvg, HA_cvg = get_spectral_cvg_MMT(MMT_LMIN0, MMT_LMAX0, zspec0, grid_ndarr[match_ii], x0)

    # creating/writing the table
    tt_mmt = Table([ID, NAME0, AP, zspec0, filt_arr, stlr_mass, stlrmassbin, stlrmassbinZ, HG_cvg, HB_cvg, HA_cvg, MMT_LMIN0, MMT_LMAX0], 
        names=['ID', 'NAME', 'AP', 'z', 'filter', 'stlrmass', 'stlrmassbin', 'stlrmassZbin', 'HG_cvg', 'HB_cvg', 'HA_cvg', 'LMIN0', 'LMAX0']) 
    asc.write(tt_mmt, FULL_PATH+'Composite_Spectra/MMT_spectral_coverage.txt', format='fixed_width', delimiter=' ')


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
            if lmin0 < HB_VAL and lmax0 > HB_VAL:
                HB = np.append(HB, 'YES')
            else:
                HB = np.append(HB, 'NO')
                
            if lmin0 < HA_VAL and lmax0 > HA_VAL:
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
    griddata = asc.read(FULL_PATH+'Spectra/spectral_Keck_grid_data.txt',guess=False)
    gridz  = np.array(griddata['ZSPEC']) ##used
    gridap = np.array(griddata['AP']) ##used

    grid   = pyfits.open(FULL_PATH+'Spectra/spectral_Keck_grid.fits')
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
    tab0 = asc.read(FULL_PATH+'Composite_Spectra/StellarMass/Keck_all_five_data.txt')
    min_mass = np.array(tab0['min_stlrmass'])
    max_mass = np.array(tab0['max_stlrmass'])
    stlrmassbin = get_stlrmassbin_arr(stlr_mass, min_mass, max_mass)

    # getting stlrmassZbin cols for the table
    tab1 = asc.read(FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt')
    stlrmassbinZ = get_stlrmassbinZ_arr(filt_arr, stlr_mass, tab1, 'Keck')

    # setting 'YES' and 'NO' and 'MASK' coverage values
    HB_cvg, HA_cvg = get_spectral_cvg_Keck(KECK_LMIN0, KECK_LMAX0)

    # creating/writing the table
    tt_keck = Table([ID, NAME0, AP, zspec0, filt_arr, stlr_mass, stlrmassbin, stlrmassbinZ, HB_cvg, HA_cvg, KECK_LMIN0, KECK_LMAX0], 
        names=['ID', 'NAME', 'AP', 'z', 'filter', 'stlrmass', 'stlrmassbin', 'stlrmassZbin', 'HB_cvg', 'HA_cvg', 'LMIN0', 'LMAX0']) 
    asc.write(tt_keck, FULL_PATH+'Composite_Spectra/Keck_spectral_coverage.txt', format='fixed_width', delimiter=' ')


def main():
    global FULL_PATH, HG_VAL, HB_VAL, HA_VAL
    FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'
    HG_VAL = 4341
    HB_VAL = 4861
    HA_VAL = 6563

    inst_dict = {} ##used
    inst_dict['MMT'] = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,']
    inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,',
                         'Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,']

    nbia = pyfits.open(FULL_PATH+'Catalogs/NB_IA_emitters.nodup.colorrev.fix.fits')
    nbiadata = nbia[1].data
    NAME0_orig = np.array(nbiadata['NAME'])

    zspec = asc.read(FULL_PATH+'Catalogs/nb_ia_zspec.txt',guess=False,
                     Reader=asc.CommentedHeader)
    zspec0 = np.array(zspec['zspec0'])

    # limit all data to valid Halpha NB emitters only
    ha_ii = np.array([x for x in range(len(NAME0_orig)) if 'Ha-NB' in NAME0_orig[x] and (zspec0[x] < 9.0 and zspec0[x] > 0.0)])
    NAME0 = NAME0_orig[ha_ii]
    zspec0 = zspec0[ha_ii]
    
    ID        = np.array(zspec['ID0'][ha_ii])
    inst_str0_orig = np.array(zspec['inst_str0'])
    inst_str0 = inst_str0_orig[ha_ii]

    fout  = asc.read(FULL_PATH+'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.fout',
                     guess=False,Reader=asc.NoHeader)
    stlr_mass_orig = np.array(fout['col7'])
    stlr_mass = stlr_mass_orig[ha_ii]

    data_dict = create_ordered_AP_arrays()
    AP = data_dict['AP'][ha_ii]
    MMT_LMIN0 = data_dict['MMT_LMIN0'][ha_ii]
    MMT_LMAX0 = data_dict['MMT_LMAX0'][ha_ii]
    KECK_LMIN0 = data_dict['KECK_LMIN0'][ha_ii]
    KECK_LMAX0 = data_dict['KECK_LMAX0'][ha_ii]
    filt_arr = get_filt_arr(NAME0)

    print 'writing MMT table...'
    write_MMT_table(inst_str0, ID, zspec0, NAME0, AP, stlr_mass, filt_arr, 
        stlr_mass_orig, inst_str0_orig, inst_dict, MMT_LMIN0, MMT_LMAX0, NAME0_orig)
    print 'finished writing MMT table'

    print 'writing Keck table...'
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
    print 'finished writing Keck table'

if __name__ == '__main__':
    main()