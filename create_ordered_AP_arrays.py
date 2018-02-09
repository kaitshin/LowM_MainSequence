"""
NAME:
    create_ordered_AP_arrays

PURPOSE:
    This code cross-references all of the data in different
    fits files and orders the relevant data in the standard
    9264 order. Meant to be used as a module.

INPUTS:
    'Catalogs/nb_ia_zspec.txt'
    'Main_Sequence/Catalogs/MMT/MMTS_all_line_fit.fits'
    'Main_Sequence/Catalogs/MMT/MMT_single_line_fit.fits'
    'Main_Sequence/Catalogs/Keck/DEIMOS_single_line_fit.fits'
    'Main_Sequence/Catalogs/Keck/DEIMOS_00_all_line_fit.fits'
    'Main_Sequence/Catalogs/merged/MMT_Keck_line_fit.fits'

OUTPUTS:
    A dictionary with ordered AP, HA_Y0, HB_Y0, HG_Y0 arrays as values
"""

import numpy as np
from astropy.io import fits as pyfits, ascii as asc

def make_AP_arr_MMT(slit_str0):
    '''
    Creates an AP (aperture) array with the slit_str0 input from
    nb_ia_zspec.txt. At the indexes of the slit_str0 where there existed a
    type of AP ('N/A', 'S./A./D./1./2./3./4.###'), the indexes of the new
    AP array were replaced with those values with the remaining values
    renamed as 'not_MMT' before the array was returned.

    Takes care of N/A and MMT,
    '''
    AP = np.array(['x.xxx']*len(slit_str0))

    #N/A
    slit_NA_index = np.array([x for x in range(len(slit_str0))
                              if slit_str0[x] == 'N/A'])
    AP[slit_NA_index] = slit_str0[slit_NA_index]

    #MMT,
    slit_S_index  = np.array([x for x in range(len(slit_str0))
                              if 'S.' == slit_str0[x][:2] and
                              len(slit_str0[x])==6], dtype=np.int32)
    AP[slit_S_index] = slit_str0[slit_S_index]

    slit_A_index  = np.array([x for x in range(len(slit_str0))
                              if 'A.' == slit_str0[x][:2] and
                              len(slit_str0[x])==6], dtype=np.int32)
    AP[slit_A_index] = slit_str0[slit_A_index]
    
    slit_D_index  = np.array([x for x in range(len(slit_str0))
                              if 'D.' == slit_str0[x][:2] and
                              len(slit_str0[x])==6], dtype=np.int32)
    AP[slit_D_index] = slit_str0[slit_D_index]
    
    slit_1_index  = np.array([x for x in range(len(slit_str0))
                              if '1.' == slit_str0[x][:2] and
                              len(slit_str0[x])==6], dtype=np.int32)
    AP[slit_1_index] = slit_str0[slit_1_index]
    
    slit_2_index  = np.array([x for x in range(len(slit_str0))
                              if '2.' == slit_str0[x][:2] and
                              len(slit_str0[x])==6], dtype=np.int32)
    AP[slit_2_index] = slit_str0[slit_2_index]
    
    slit_3_index  = np.array([x for x in range(len(slit_str0))
                              if '3.' == slit_str0[x][:2] and
                              len(slit_str0[x])==6], dtype=np.int32)
    AP[slit_3_index] = slit_str0[slit_3_index]
    
    slit_4_index  = np.array([x for x in range(len(slit_str0))
                              if '4.' == slit_str0[x][:2] and
                              len(slit_str0[x])==6], dtype=np.int32)
    AP[slit_4_index] = slit_str0[slit_4_index]

    bad_index = np.array([x for x in range(len(AP)) if AP[x] == 'x.xxx'])
    APgood = np.array(AP, dtype='|S20')
    APgood[bad_index] = 'not_MMT'
    
    return APgood
#enddef


def make_AP_arr_DEIMOS(AP, slit_str0):
    '''
    Accepts the AP array made by make_AP_arr_MMT and the slit_str0 array.
    Then, at the indices of slit_str0 where '##.###' exists (that's not a
    FOCAS detection), those indices of the AP array are replaced and then
    after modification is done, returned.

    Those with '08.' as a detection were ignored for now.

    Takes care of Keck, and Keck,Keck,
    '''

    #Keck,
    temp_index1 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x]) == 7 and 'f' not in slit_str0[x]
                            and '08.'!=slit_str0[x][:3]])
    AP[temp_index1] = slit_str0[temp_index1]
    AP[temp_index1] = np.array([x[:6] for x in AP[temp_index1]])

    temp_index2 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x]) == 7 and 'f' not in slit_str0[x]
                            and '08.'==slit_str0[x][:3]])
    AP[temp_index2] = 'INVALID_KECK'

    #Keck,Keck,
    temp_index3 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x])==14 and 'f' not in slit_str0[x]
                            and '08.'==slit_str0[x][7:10] and
                            '08.'==slit_str0[x][:3]])
    AP[temp_index3] = 'INVALID_KECK'

    temp_index4 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x])==14 and 'f' not in slit_str0[x]
                            and '08.'==slit_str0[x][:3] and
                            '08.'!=slit_str0[x][7:10]])
    AP[temp_index4] = slit_str0[temp_index4]
    AP[temp_index4] = np.array([x[7:13] for x in AP[temp_index4]])
    
    return AP
#enddef


def make_AP_arr_merged(AP, slit_str0):
    '''
    Accepts the AP array made by make_AP_arr_DEIMOS and the slit_str0 array.
    Then, at the indices where there were multiple detections (not including
    a FOCAS detection), those indices were replaced and returned.

    Takes care of merged, and MMT,Keck, and merged,FOCAS,
    '''

    #merged,
    temp_index1 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x]) == 13 and slit_str0[x][5] == ','
                            and 'f' not in slit_str0[x] and
                            '08.'!= slit_str0[x][6:9]])
    AP[temp_index1] = slit_str0[temp_index1]
    AP[temp_index1] = np.array([x[:12] for x in AP[temp_index1]])

    #MMT,Keck, means MMT,
    temp_index2 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x]) == 13 and slit_str0[x][5] == ','
                            and 'f' not in slit_str0[x] and
                            '08.'== slit_str0[x][6:9]])
    AP[temp_index2] = slit_str0[temp_index2]
    AP[temp_index2] = np.array([x[:5] for x in AP[temp_index2]])

    #merged,FOCAS,
    temp_index3 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x]) > 13 and 'f'==slit_str0[x][13]
                            and '08.' not in slit_str0[x][:13]], dtype=np.int32)
    AP[temp_index3] = slit_str0[temp_index3]
    AP[temp_index3] = np.array([x[:12] for x in AP[temp_index3]])
    
    return AP
#enddef


def make_AP_arr_FOCAS(AP, slit_str0):
    '''
    Accepts the AP array made by make_AP_arr_DEIMOS and the slit_str0 array.
    Same idea as the other make_AP_arr functions.

    Takes care of FOCAS, and FOCAS,FOCAS,FOCAS, and FOCAS,FOCAS, and
    MMT,FOCAS, and Keck,FOCAS, and Keck,Keck,FOCAS, and Keck,FOCAS,FOCAS,
    '''

    #FOCAS,
    temp_index1 = np.array([x for x in range(len(slit_str0)) if
                            'f'==slit_str0[x][0] and len(slit_str0[x])==7],
                           dtype=np.int32)
    AP[temp_index1] = 'FOCAS'

    #FOCAS,FOCAS,FOCAS,
    temp_index2 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x])==21 and 'f'==slit_str0[x][0] and
                            'f'==slit_str0[x][7] and 'f'==slit_str0[x][14]])
    AP[temp_index2] = 'FOCAS'

    #FOCAS,FOCAS,
    temp_index3 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x])==14 and 'f'==slit_str0[x][0] and
                            'f'==slit_str0[x][7]])
    AP[temp_index3] = 'FOCAS'

    #MMT,FOCAS,
    temp_index4 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x])==13 and 'f'==slit_str0[x][6]])
    AP[temp_index4] = slit_str0[temp_index4]
    AP[temp_index4] = np.array([x[:5] for x in AP[temp_index4]])

    #Keck,FOCAS,
    temp_index5 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x])==14 and 'f'==slit_str0[x][7] and
                            'f' not in slit_str0[x][:6]])
    AP[temp_index5] = slit_str0[temp_index5]
    AP[temp_index5] = np.array([x[:6] for x in AP[temp_index5]])

    #Keck,Keck,FOCAS,
    temp_index6 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x])==21 and 'f'==slit_str0[x][14] and
                            'f' not in slit_str0[x][:13] and
                            '08.'==slit_str0[x][:3]])
    AP[temp_index6] = slit_str0[temp_index6]
    AP[temp_index6] = np.array([x[7:13] for x in AP[temp_index6]])

    #Keck,FOCAS,FOCAS,
    temp_index7 = np.array([x for x in range(len(slit_str0)) if
                            len(slit_str0[x])==21 and 'f' not in
                            slit_str0[x][:6] and 'f'==slit_str0[x][7] and
                            'f'==slit_str0[x][14]])
    AP[temp_index7] = slit_str0[temp_index7]
    AP[temp_index7] = np.array([x[:6] for x in AP[temp_index7]])
    
    return AP
#enddef


def get_Y0(all_AP, detect_AP, all_HA_Y0, detect_HA_Y0, all_HB_Y0,
           detect_HB_Y0, all_HG_Y0, detect_HG_Y0):
    '''
    Accepts, modifies, and returns 'HA_Y0' (passed in as 'all_HA_Y0').
    'all_AP' is the complete AP column with all the information, while
    'detect_AP' and every input subsequent until the last four are the arrays
    specific to the Main_Sequence catalog.

    There are 5 different types of catalogs, so this method is called 5 times.

    This method looks at the indices where the detect_AP is in the all_AP and
    appends the overlapping indices of the all_AP array. Then, at those
    overlapping indices, the zero values in all_AP are replaced by the
    corresponding detected values.
    '''

    index1 = np.array([x for x in range(len(detect_AP)) if detect_AP[x]
                       in all_AP], dtype=np.int32)
    index2 = np.array([])

    for mm in range(len(detect_AP)):
        index2 = np.append(index2, [x for x in range(len(all_AP))
                                    if all_AP[x] == detect_AP[mm]])
    #endfor
    index2 = np.array(index2, dtype=np.int32)

    all_HA_Y0[index2] = detect_HA_Y0[index1]
    all_HB_Y0[index2] = detect_HB_Y0[index1]
    all_HG_Y0[index2] = detect_HG_Y0[index1]

    return all_HA_Y0,all_HB_Y0,all_HG_Y0
#enddef


def get_LMIN0_LMAX0(all_AP, detect_AP, all_MMT_LMIN0, detect_MMT_LMIN0, 
    all_MMT_LMAX0, detect_MMT_LMAX0, all_KECK_LMIN0, detect_KECK_LMIN0,
    all_KECK_LMAX0, detect_KECK_LMAX0):
    '''
    Accepts, modifies, and returns 'HA_Y0' (passed in as 'all_HA_Y0').
    'all_AP' is the complete AP column with all the information, while
    'detect_AP' and every input subsequent until the last four are the arrays
    specific to the Main_Sequence catalog.

    There are 5 different types of catalogs, so this method is called 5 times.

    This method looks at the indices where the detect_AP is in the all_AP and
    appends the overlapping indices of the all_AP array. Then, at those
    overlapping indices, the zero values in all_AP are replaced by the
    corresponding detected values.
    '''

    index1 = np.array([x for x in range(len(detect_AP)) if detect_AP[x]
                       in all_AP], dtype=np.int32)
    index2 = np.array([])

    for mm in range(len(detect_AP)):
        index2 = np.append(index2, [x for x in range(len(all_AP))
                                    if all_AP[x] == detect_AP[mm]])
    #endfor
    index2 = np.array(index2, dtype=np.int32)

    all_MMT_LMIN0[index2] = detect_MMT_LMIN0[index1]
    all_MMT_LMAX0[index2] = detect_MMT_LMAX0[index1]
    all_KECK_LMIN0[index2] = detect_KECK_LMIN0[index1]
    all_KECK_LMAX0[index2] = detect_KECK_LMAX0[index1]

    return all_MMT_LMIN0,all_MMT_LMAX0,all_KECK_LMIN0,all_KECK_LMAX0
#enddef


def create_ordered_AP_arrays(AP_only=False):
    '''
    Reads relevant inputs, combining all of the input data into one ordered
    array for AP by calling make_AP_arr_MMT, make_AP_arr_DEIMOS,
    make_AP_arr_merged, and make_AP_arr_FOCAS. 

    Using the AP order, then creates HA, HB, HG_Y0 arrays by calling get_Y0
    '''

    zspec = asc.read('/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/Catalogs/nb_ia_zspec.txt',guess=False,
                     Reader=asc.CommentedHeader)
    slit_str0 = np.array(zspec['slit_str0'])

    MMTall = pyfits.open('/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/Main_Sequence/Catalogs/MMT/MMTS_all_line_fit.fits')
    MMTalldata = MMTall[1].data
    MMTallAP = MMTalldata['AP']
    MMTallHAY0 = MMTalldata['HA_Y0']
    MMTallHBY0 = MMTalldata['HB_Y0']
    MMTallHGY0 = MMTalldata['HG_Y0']
    MMTallLMIN0 = MMTalldata['LMIN0']
    MMTallLMAX0 = MMTalldata['LMAX0']

    MMTsingle = pyfits.open('/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/Main_Sequence/Catalogs/MMT/MMT_single_line_fit.fits')
    MMTsingledata = MMTsingle[1].data
    MMTsingleAP = MMTsingledata['AP']
    MMTsingleHAY0 = MMTsingledata['HA_Y0']
    MMTsingleHBY0 = MMTsingledata['HB_Y0']
    MMTsingleHGY0 = MMTsingledata['HG_Y0']
    MMTsingleLMIN0 = MMTsingledata['LMIN0']
    MMTsingleLMAX0 = MMTsingledata['LMAX0']

    DEIMOS = pyfits.open('/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/Main_Sequence/Catalogs/Keck/DEIMOS_single_line_fit.fits')
    DEIMOSdata = DEIMOS[1].data
    DEIMOSAP = DEIMOSdata['AP']
    DEIMOSHAY0 = DEIMOSdata['HA_Y0']
    DEIMOSHBY0 = DEIMOSdata['HB_Y0']
    DEIMOSHGY0 = DEIMOSdata['HG_Y0']
    DEIMOSLMIN0 = DEIMOSdata['LMIN0']
    DEIMOSLMAX0 = DEIMOSdata['LMAX0']

    DEIMOS00=pyfits.open('/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/Main_Sequence/Catalogs/Keck/DEIMOS_00_all_line_fit.fits')
    DEIMOS00data = DEIMOS00[1].data
    DEIMOS00AP = DEIMOS00data['AP']
    DEIMOS00HAY0 = DEIMOS00data['HA_Y0']
    DEIMOS00HBY0 = DEIMOS00data['HB_Y0']
    DEIMOS00HGY0 = DEIMOS00data['HG_Y0']
    DEIMOS00LMIN0 = DEIMOS00data['LMIN0']
    DEIMOS00LMAX0 = DEIMOS00data['LMAX0']

    merged = pyfits.open('/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/Main_Sequence/Catalogs/merged/MMT_Keck_line_fit.fits')
    mergeddata = merged[1].data
    mergedAP = mergeddata['AP']
    mergedHAY0 = mergeddata['HA_Y0']
    mergedHBY0 = mergeddata['HB_Y0']
    mergedHGY0 = mergeddata['HG_Y0']
    mergedLMIN0_MMT = mergeddata['MMT_LMIN0']
    mergedLMAX0_MMT = mergeddata['MMT_LMAX0']
    mergedLMIN0_KECK = mergeddata['KECK_LMIN0']
    mergedLMAX0_KECK = mergeddata['KECK_LMAX0']
    #end inputs
    print '### done reading input files'

    print '### creating ordered AP arr'
    AP0 = make_AP_arr_MMT(slit_str0)
    AP1 = make_AP_arr_DEIMOS(AP0, slit_str0)
    AP2 = make_AP_arr_merged(AP1, slit_str0)
    AP  = make_AP_arr_FOCAS(AP2, slit_str0)
    print '### done creating ordered AP arr'

    if (AP_only == True):
        MMTall.close()
        MMTsingle.close()
        DEIMOS.close()
        DEIMOS00.close()
        merged.close()

        return {'AP': AP}
    #endif 

    print '### creating ordered HA,HB,HG_Y0 arr'
    HA_Y0 = np.array([-99.99999]*len(AP))
    HB_Y0 = np.array([-99.99999]*len(AP))
    HG_Y0 = np.array([-99.99999]*len(AP))
    HA_Y0, HB_Y0, HG_Y0 = get_Y0(AP, MMTallAP, HA_Y0, MMTallHAY0, HB_Y0,
                                 MMTallHBY0, HG_Y0, MMTallHGY0)
    HA_Y0, HB_Y0, HG_Y0 = get_Y0(AP, MMTsingleAP, HA_Y0, MMTsingleHAY0, HB_Y0,
                                 MMTsingleHBY0, HG_Y0, MMTsingleHGY0)
    HA_Y0, HB_Y0, HG_Y0 = get_Y0(AP, DEIMOSAP, HA_Y0, DEIMOSHAY0, HB_Y0,
                                 DEIMOSHBY0, HG_Y0, DEIMOSHGY0)
    HA_Y0, HB_Y0, HG_Y0 = get_Y0(AP, DEIMOS00AP, HA_Y0, DEIMOS00HAY0, HB_Y0,
                                 DEIMOS00HBY0, HG_Y0, DEIMOS00HGY0)
    HA_Y0, HB_Y0, HG_Y0 = get_Y0(AP, mergedAP, HA_Y0, mergedHAY0, HB_Y0,
                                 mergedHBY0, HG_Y0, mergedHGY0)
    print '### done creating ordered HA,HB,HG_Y0 arrays'

    print '### creating ordered LMIN0/LMAX0 arr'
    MMT_LMIN0 = np.array([-99.99999]*len(AP))
    MMT_LMAX0 = np.array([-99.99999]*len(AP))
    KECK_LMIN0 = np.array([-99.99999]*len(AP))
    KECK_LMAX0 = np.array([-99.99999]*len(AP))
    MMT_LMIN0, MMT_LMAX0, KECK_LMIN0, KECK_LMAX0 = get_LMIN0_LMAX0(AP, MMTallAP, MMT_LMIN0, MMTallLMIN0, 
        MMT_LMAX0, MMTallLMAX0, KECK_LMIN0, MMTallLMIN0, KECK_LMAX0, MMTallLMAX0)
    MMT_LMIN0, MMT_LMAX0, KECK_LMIN0, KECK_LMAX0 = get_LMIN0_LMAX0(AP, MMTsingleAP, MMT_LMIN0, MMTsingleLMIN0, 
        MMT_LMAX0, MMTsingleLMAX0, KECK_LMIN0, MMTsingleLMIN0, KECK_LMAX0, MMTsingleLMAX0)
    MMT_LMIN0, MMT_LMAX0, KECK_LMIN0, KECK_LMAX0 = get_LMIN0_LMAX0(AP, DEIMOSAP, MMT_LMIN0, DEIMOSLMIN0, 
        MMT_LMAX0, DEIMOSLMAX0, KECK_LMIN0, DEIMOSLMIN0, KECK_LMAX0, DEIMOSLMAX0)
    MMT_LMIN0, MMT_LMAX0, KECK_LMIN0, KECK_LMAX0 = get_LMIN0_LMAX0(AP, DEIMOS00AP, MMT_LMIN0, DEIMOS00LMIN0, 
        MMT_LMAX0, DEIMOS00LMAX0, KECK_LMIN0, DEIMOS00LMIN0, KECK_LMAX0, DEIMOS00LMAX0)
    MMT_LMIN0, MMT_LMAX0, KECK_LMIN0, KECK_LMAX0 = get_LMIN0_LMAX0(AP, mergedAP, MMT_LMIN0, mergedLMIN0_MMT, 
        MMT_LMAX0, mergedLMAX0_MMT, KECK_LMIN0, mergedLMIN0_KECK, KECK_LMAX0, mergedLMAX0_KECK)
    print '### done creating ordered LMIN0/LMAX0 arr'

    MMTall.close()
    MMTsingle.close()
    DEIMOS.close()
    DEIMOS00.close()
    merged.close()

    return {'AP': AP, 'HA_Y0': HA_Y0, 'HB_Y0': HB_Y0, 'HG_Y0': HG_Y0, 'MMT_LMIN0': MMT_LMIN0, 'MMT_LMAX0': MMT_LMAX0, 'KECK_LMIN0': KECK_LMIN0, 'KECK_LMAX0': KECK_LMAX0}


def main():
    AP_dict = create_ordered_AP_arrays()


if __name__ == '__main__':
    main()