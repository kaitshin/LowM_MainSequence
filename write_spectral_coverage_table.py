import numpy as np, numpy.ma as ma, matplotlib.pyplot as plt, re, copy
import plotting.general_plotting as general_plotting
from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table
from create_ordered_AP_arrays import create_ordered_AP_arrays

def get_filt_arr(NAME0, ha_ii):
    '''
    '''
    filt_arr = np.array([])
    for name in NAME0[ha_ii]:
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


def get_indexed_arrs(ha_ii, arrs):
    '''
    '''
    copyarrs = copy.deepcopy(arrs)
    for i in range(len(copyarrs)):
        copyarrs[i] = copyarrs[i][ha_ii]
    return copyarrs


def get_init_NO_YES_coverage(arr):
    '''
    '''
    covg_arrs = [[], [], []] # assuming len(arr) <= 3
    
    for i in range(len(arr)):
        covg_arrs[i] = np.array(['NO']*len(arr[i]), dtype='|S4')
        good_ii = np.array([x for x in range(len(arr[i])) if (arr[i][x] > -99 and arr[i][x] != -1.0 and arr[i][x] != 0.0)])
        covg_arrs[i][good_ii] = 'YES'

    return covg_arrs


def get_stlrmassbin_arr(stlr_mass, inst_str0, inst_dict, stlr_mass_ii_instr, instr):
    '''
    '''
    index_list = general_plotting.get_index_list2(stlr_mass, inst_str0, inst_dict, instr)

    masslist = []
    for match_index in index_list:
        subtitle = 'stlrmass: '+str(min(stlr_mass[match_index]))+'-'+str(max(stlr_mass[match_index]))
        masslist.append(subtitle[10:].split('-'))
    #endfor

    stlrmassbin = np.array([], dtype='int32')
    for mass in stlr_mass_ii_instr:
        bin_num = -1
        if mass >= np.float(masslist[-1][0]):
            bin_num = len(masslist)
        elif mass >= np.float(masslist[-2][0]):
            bin_num = len(masslist)-1
        elif mass >= np.float(masslist[-3][0]):
            bin_num = len(masslist)-2
        elif mass >= np.float(masslist[-4][0]):
            bin_num = len(masslist)-3
        elif mass >= np.float(masslist[-5][0]):
            bin_num = len(masslist)-4
            
        stlrmassbin = np.append(stlrmassbin, bin_num)
    #endfor
    return stlrmassbin


def get_stlrmassbinZ_arr(stlr_mass, inst_str0, inst_dict, stlr_mass_ii_instr, filt_arr_instr, instr):
    '''
    '''
    index_list = []
    if instr=='MMT':
        index_list = general_plotting.get_index_list3(stlr_mass, inst_str0, inst_dict, instr)
    elif instr=='Keck':
        index_list = general_plotting.get_index_list2(stlr_mass, inst_str0, inst_dict, instr)

    masslist = []
    for match_index in index_list:
        subtitle = 'stlrmass: '+str(min(stlr_mass[match_index]))+'-'+str(max(stlr_mass[match_index]))
        masslist.append(subtitle[10:].split('-'))
    #endfor

    stlrmassZbin = np.array([], dtype='int32')
    if instr == 'MMT':
        for mass, filt in zip(stlr_mass_ii_instr, filt_arr_instr):
            bin_num = -1
            if mass >= np.float(masslist[-1][0]):
                bin_num = len(masslist)
            elif mass >= np.float(masslist[-2][0]):
                bin_num = len(masslist)-1
                
            stlrmassZbin = np.append(stlrmassZbin, str(bin_num)+'-'+filt)
        #endfor
        return stlrmassZbin
    #endif
    elif instr == 'Keck':
        for mass, filt in zip(stlr_mass_ii_instr, filt_arr_instr):
            bin_num = -1
            if mass >= np.float(masslist[-1][0]):
                bin_num = len(masslist)
            elif mass >= np.float(masslist[-2][0]):
                bin_num = len(masslist)-1
            elif mass >= np.float(masslist[-3][0]):
                bin_num = len(masslist)-2
            elif mass >= np.float(masslist[-4][0]):
                bin_num = len(masslist)-3
            elif mass >= np.float(masslist[-5][0]):
                bin_num = len(masslist)-4
                
            stlrmassZbin = np.append(stlrmassZbin, str(bin_num)+'-'+filt)
        #endfor
        return stlrmassZbin
    #endif


def get_spectral_cvg_MMT(mmt_ii, MMT_LMIN0_ii, MMT_LMAX0_ii):
    '''
    '''
    MMT_LMIN0_ii_m, MMT_LMAX0_ii_m = get_indexed_arrs(mmt_ii, [MMT_LMIN0_ii, MMT_LMAX0_ii])

    HG = np.array([])
    HB = np.array([])
    HA = np.array([])
    for lmin, lmax in zip(MMT_LMIN0_ii_m, MMT_LMAX0_ii_m):
        if lmin < 0:
            HG = np.append(HG, 'NO')
            HB = np.append(HB, 'NO')
            HA = np.append(HA, 'NO')
        else:
            if lmin < HG_VAL and lmax > HG_VAL:
                HG = np.append(HG, 'YES')
            else:
                HG = np.append(HG, 'NO')
            
            if lmin < HB_VAL and lmax > HB_VAL:
                HB = np.append(HB, 'YES')
            else:
                HB = np.append(HB, 'NO')
                
            if lmin < HA_VAL and lmax > HA_VAL:
                HA = np.append(HA, 'YES')
            else:
                HA = np.append(HA, 'NO')
    #endfor
    return HG, HB, HA


def write_MMT_table(inst_str0_ii, ID_ii, z_ii, NAME0_ii, AP_ii, stlr_mass_ii, filt_arr, stlr_mass, inst_str0, inst_dict, MMT_LMIN0_ii, MMT_LMAX0_ii):
    '''
    '''
    mmt_ii = [x for x in range(len(inst_str0_ii)) if 'MMT' in inst_str0_ii[x] or 'merged' in inst_str0_ii[x]]
    ID_ii_m, z_ii_m, NAME0_ii_m, AP_ii_m, inst_str0_ii_m, stlr_mass_ii_m, filt_arr_m = get_indexed_arrs(mmt_ii, [ID_ii, z_ii, NAME0_ii, AP_ii, inst_str0_ii, stlr_mass_ii, filt_arr])

    stlrmassbin_ii_m = get_stlrmassbin_arr(stlr_mass, inst_str0, inst_dict, stlr_mass_ii_m, 'MMT')
    stlrmassbinZ_ii_m = get_stlrmassbinZ_arr(stlr_mass, inst_str0, inst_dict, stlr_mass_ii_m, filt_arr_m, 'MMT')
    HG_cvg, HB_cvg, HA_cvg = get_spectral_cvg_MMT(mmt_ii, MMT_LMIN0_ii, MMT_LMAX0_ii)

    AP_ii_m = np.array([ap[:5] for ap in AP_ii_m])

    tt_mmt = Table([ID_ii_m, NAME0_ii_m, AP_ii_m, z_ii_m, filt_arr_m, 
        stlrmassbin_ii_m, stlrmassbinZ_ii_m, HG_cvg, HB_cvg, HA_cvg], 
        names=['ID', 'NAME', 'AP', 'z', 'filter', 'stlrmassbin', 'stlrmassZbin', 'HG_cvg', 'HB_cvg', 'HA_cvg']) 
    asc.write(tt_mmt, FULL_PATH+'Composite_Spectra/MMT_spectral_coverage.txt', format='fixed_width', delimiter=' ')


def get_spectral_cvg_Keck(keck_ii, KECK_LMIN0_ii, KECK_LMAX0_ii):
    '''
    '''
    KECK_LMIN0_ii_k, KECK_LMAX0_ii_k = get_indexed_arrs(keck_ii, [KECK_LMIN0_ii, KECK_LMAX0_ii])

    HB = np.array([])
    HA = np.array([])
    for lmin, lmax in zip(KECK_LMIN0_ii_k, KECK_LMAX0_ii_k):
        if lmin < 0:
            HB = np.append(HB, 'NO')
            HA = np.append(HA, 'NO')
        else:
            if lmin < HB_VAL and lmax > HB_VAL:
                HB = np.append(HB, 'YES')
            else:
                HB = np.append(HB, 'NO')
                
            if lmin < HA_VAL and lmax > HA_VAL:
                HA = np.append(HA, 'YES')
            else:
                HA = np.append(HA, 'NO')
    #endfor
    return HB, HA


def write_Keck_table(inst_str0_ii, ID_ii, z_ii, NAME0_ii, AP_ii, stlr_mass_ii, filt_arr, stlr_mass, inst_str0, inst_dict, KECK_LMIN0_ii, KECK_LMAX0_ii):
    '''
    '''
    keck_ii = [x for x in range(len(inst_str0_ii)) if ('Keck' in inst_str0_ii[x] or 'merged' in inst_str0_ii[x]) and 'MMT' not in inst_str0_ii[x]] # MMT,Keck, means MMT
    ID_ii_k, z_ii_k, NAME0_ii_k, AP_ii_k, inst_str0_ii_k, stlr_mass_ii_k, filt_arr_k = get_indexed_arrs(keck_ii, [ID_ii, z_ii, NAME0_ii, AP_ii, inst_str0_ii, stlr_mass_ii, filt_arr])

    stlrmassbin_ii_k = get_stlrmassbin_arr(stlr_mass, inst_str0, inst_dict, stlr_mass_ii_k, 'Keck')
    stlrmassbinZ_ii_k = get_stlrmassbinZ_arr(stlr_mass, inst_str0, inst_dict, stlr_mass_ii_k, filt_arr_k, 'Keck')
    HB_cvg, HA_cvg = get_spectral_cvg_Keck(keck_ii, KECK_LMIN0_ii, KECK_LMAX0_ii)

    AP_ii_k = np.array([x if len(x) == 6 else x[6:] for x in AP_ii_k])

    tt_keck = Table([ID_ii_k, NAME0_ii_k, AP_ii_k, z_ii_k, filt_arr_k, 
        stlrmassbin_ii_k, stlrmassbinZ_ii_k, HB_cvg, HA_cvg], 
        names=['ID', 'NAME', 'AP', 'z', 'filter', 'stlrmassbin', 'stlrmassZbin', 'HB_cvg', 'HA_cvg']) 
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

    nbia = pyfits.open(FULL_PATH+'Catalogs/python_outputs/nbia_all_nsource.fits')
    nbiadata = nbia[1].data
    NAME0 = nbiadata['source_name'] ##used

    zspec = asc.read(FULL_PATH+'Catalogs/nb_ia_zspec.txt',guess=False,
                     Reader=asc.CommentedHeader)
    inst_str0 = np.array(zspec['inst_str0'])

    fout  = asc.read(FULL_PATH+'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.fout',
                     guess=False,Reader=asc.NoHeader)
    stlr_mass = np.array(fout['col7']) ##used
    nan_stlr_mass = np.copy(stlr_mass)
    nan_stlr_mass[nan_stlr_mass < 0] = np.nan

    data_dict = create_ordered_AP_arrays()
    AP = data_dict['AP'] ##used
    HA_Y0 = data_dict['HA_Y0'] ##used
    HB_Y0 = data_dict['HB_Y0'] ##used
    HG_Y0 = data_dict['HG_Y0'] ##used
    MMT_LMIN0 = data_dict['MMT_LMIN0']
    MMT_LMAX0 = data_dict['MMT_LMAX0']
    KECK_LMIN0 = data_dict['KECK_LMIN0']
    KECK_LMAX0 = data_dict['KECK_LMAX0']

    ha_ii = [x for x in range(len(NAME0)) if 'Ha' in NAME0[x]]
    filt_arr = get_filt_arr(NAME0, ha_ii)
    ID_ii, z_ii, NAME0_ii, AP_ii, inst_str0_ii, stlr_mass_ii = get_indexed_arrs(ha_ii, [zspec['ID0'], zspec['zspec0'], NAME0, AP, inst_str0, stlr_mass])
    MMT_LMIN0_ii, MMT_LMAX0_ii, KECK_LMIN0_ii, KECK_LMAX0_ii = get_indexed_arrs(ha_ii, [MMT_LMIN0, MMT_LMAX0, KECK_LMIN0, KECK_LMAX0])
    HG_Y0_i, HB_Y0_i, HA_Y0_i = get_indexed_arrs(ha_ii, [HG_Y0, HB_Y0, HA_Y0])
    HG_Y0_ii, HB_Y0_ii, HA_Y0_ii = get_init_NO_YES_coverage([HG_Y0_i, HB_Y0_i, HA_Y0_i])

    print 'writing MMT table...'
    write_MMT_table(inst_str0_ii, ID_ii, z_ii, NAME0_ii, AP_ii, stlr_mass_ii, filt_arr, stlr_mass, inst_str0, inst_dict, MMT_LMIN0_ii, MMT_LMAX0_ii)
    print 'finished writing MMT table'

    print 'writing Keck table...'
    write_Keck_table(inst_str0_ii, ID_ii, z_ii, NAME0_ii, AP_ii, stlr_mass_ii, filt_arr, stlr_mass, inst_str0, inst_dict, KECK_LMIN0_ii, KECK_LMAX0_ii)
    print 'finished writing Keck table'    

if __name__ == '__main__':
    main()