import numpy as np, matplotlib.pyplot as plt
import plotting.general_plotting as general_plotting
from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table
from create_ordered_AP_arrays import create_ordered_AP_arrays

global FULL_PATH

def get_stlrmass_lists(stlr_mass, inst_str0, inst_dict):
    '''
    Helper function for main. Returns the MMT+Keck 'stlrmass' bins
    '''
    masslist_MMT, masslist_Keck = np.array([]), np.array([])
    for instr in ['MMT', 'Keck']:
        index_list = general_plotting.get_index_list2(stlr_mass, inst_str0, inst_dict, instr)

        masslist = []
        for match_index in index_list:
            subtitle = [min(stlr_mass[match_index]), max(stlr_mass[match_index])] # USE NUMBERS NOT STRINGS
            masslist.append(subtitle)
        #endfor

        if instr == 'MMT':
            masslist_MMT = np.array(masslist)
        else:
            masslist_Keck = np.array(masslist)
    #endfor
    return masslist_MMT, masslist_Keck


def get_stlrmassZ_lists(stlr_mass, inst_str0, inst_dict):
    '''
    Helper function for main. Returns the MMT+Keck 'stlrmassZ' bins
    '''
    masslistZ_MMT, masslistZ_Keck = np.array([]), np.array([])
    for instr in ['MMT', 'Keck']:
        if instr=='MMT':
            index_list = general_plotting.get_index_list3(stlr_mass, inst_str0, inst_dict, instr)
        elif instr=='Keck':
            index_list = general_plotting.get_index_list2(stlr_mass, inst_str0, inst_dict, instr)
        
        masslist = []
        for match_index in index_list:
            subtitle = [min(stlr_mass[match_index]), max(stlr_mass[match_index])]
            masslist.append(subtitle)
        #endfor

        if instr == 'MMT':
            masslistZ_MMT = np.array(masslist)
        else:
            masslistZ_Keck = np.array(masslist)
    #endfor
    return masslistZ_MMT, masslistZ_Keck


def get_bins(masslist_MMT, masslist_Keck, massZlist_MMT, massZlist_Keck):
    '''
    Helper function for main. Flattens ndarrays of bins into a bins array
    '''
    massbins_MMT = np.append(masslist_MMT[:,0], masslist_MMT[-1,-1])
    massbins_Keck = np.append(masslist_Keck[:,0], masslist_Keck[-1,-1])
    massZbins_MMT = np.append(massZlist_MMT[:,0], massZlist_MMT[-1,-1])
    massZbins_Keck = np.append(massZlist_Keck[:,0], massZlist_Keck[-1,-1])

    return massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck


def fix_masses_out_of_range(masses_MMT, masses_Keck, massZs_MMT, massZs_Keck):
    '''
    Fixes masses that were not put into bins because they were too low.
    Reassigns them to the lowest mass bin.
    '''
    masses_MMT = np.array([1 if x==0 else x for x in masses_MMT])
    masses_Keck = np.array([1 if x==0 else x for x in masses_Keck])
    massZs_MMT = np.array([1 if x==0 else x for x in massZs_MMT])
    massZs_Keck = np.array([1 if x==0 else x for x in massZs_Keck])

    return masses_MMT, masses_Keck, massZs_MMT, massZs_Keck


def add_Zs_to_bins(massZs_MMT, massZs_Keck, filts):
    '''
    Adds the Z componenet to the stlrrmassZ bin lists
    '''
    massZs_MMT = np.array([str(massZs_MMT[x])+'-'+filts[x] for x in range(len(filts))])
    massZs_Keck = np.array([str(massZs_Keck[x])+'-'+filts[x] for x in range(len(filts))])

    return massZs_MMT, massZs_Keck


def bins_table(indexes, NAME0, AP, stlr_mass, massbins_MMT, massbins_Keck, 
    massZbins_MMT, massZbins_Keck):
    '''
    Creates and returns a table of bins as such:

    NAME             stlrmass filter stlrmassbin_MMT stlrmassbin_Keck stlrmassZbin_MMT stlrmassZbin_Keck
    ---------------- -------- ------ --------------- ---------------- ---------------- -----------------
    Ha-NB704_005332  6.78     NB704  1               1                1-NB704           1-NB704

    TODO: for 'indexes == yes_spectra', include S/N arrays etc.?
    '''
    names = NAME0[indexes]
    masses = stlr_mass[indexes]

    # get redshift bins
    filts = np.array([x[x.find('Ha-')+3:x.find('Ha-')+8] for x in names])

    # get stlrmass and stlrmassZ bins
    masses_MMT = np.digitize(masses, massbins_MMT)
    masses_Keck = np.digitize(masses, massbins_Keck)
    massZs_MMT = np.digitize(masses, massZbins_MMT)
    massZs_Keck = np.digitize(masses, massZbins_Keck)
    masses_MMT, masses_Keck, massZs_MMT, massZs_Keck = fix_masses_out_of_range(masses_MMT, 
        masses_Keck, massZs_MMT, massZs_Keck)
    massZs_MMT, massZs_Keck = add_Zs_to_bins(massZs_MMT, massZs_Keck, filts)

    tab0 = Table([names, masses, filts, masses_MMT, masses_Keck, massZs_MMT, massZs_Keck], 
        names=['NAME', 'stlrmass', 'filter', 'stlrmassbin_MMT', 'stlrmassbin_Keck', 
        'stlrmassZbin_MMT', 'stlrmassZbin_Keck'])

    return tab0


def EBV_corrs_no_spectra():
    '''
    '''
    # if there's no stlrmass, use filts for EBV corrections
    # if filts == 704, 711, or 816, use stlrmassZ MMT Hg/Hb lines for EBV corrs
    # if filts == 921, 973 use stlrmassZ MMT Hg/Hb or Keck Ha/Hb for EBV corrs ... ?


def main():
    FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'

    # reading in data
    nbia = pyfits.open(FULL_PATH+'Catalogs/NB_IA_emitters.nodup.colorrev.fix.fits')
    nbiadata = nbia[1].data
    NAME0 = np.array(nbiadata['NAME'])
    zspec = asc.read(FULL_PATH+'Catalogs/nb_ia_zspec.txt',guess=False,
                     Reader=asc.CommentedHeader)
    zspec0 = np.array(zspec['zspec0'])
    inst_str0 = np.array(zspec['inst_str0'])
    fout  = asc.read(FULL_PATH+'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.fout',
                     guess=False,Reader=asc.NoHeader)
    stlr_mass = np.array(fout['col7'])
    data_dict = create_ordered_AP_arrays(AP_only = True)
    AP = data_dict['AP']

    # limit all data to Halpha emitters only
    ha_ii = np.array([x for x in range(len(NAME0)) if 'Ha' in NAME0[x]])
    NAME0     = NAME0[ha_ii]
    zspec0    = zspec0[ha_ii]
    inst_str0 = inst_str0[ha_ii]
    stlr_mass = stlr_mass[ha_ii]
    AP        = AP[ha_ii]

    # defining other useful data structs
    inst_dict = {}
    inst_dict['MMT'] = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,']
    inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,','Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,']
    masslist_MMT, masslist_Keck = get_stlrmass_lists(stlr_mass, inst_str0, inst_dict)
    massZlist_MMT, massZlist_Keck = get_stlrmassZ_lists(stlr_mass, inst_str0, inst_dict)
    massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck = get_bins(masslist_MMT, 
        masslist_Keck, massZlist_MMT, massZlist_Keck)
    
    # getting indexes for sources with and without spectra
    no_spectra  = np.where((zspec0 <= 0) | (zspec0 > 9))[0]
    yes_spectra = np.where((zspec0 >= 0) & (zspec0 < 9))[0]

    # getting tables of which bins the sources fall into (for eventual EBV corrections)
    tab_no_spectra = bins_table(no_spectra, NAME0, AP, stlr_mass, 
        massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck)
    tab_yes_spectra = bins_table(yes_spectra, NAME0, AP, stlr_mass, 
        massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck)

    # start getting those corrections!
    # EBV_corrs_no_spectra(no_spectra, NAME0, AP, stlr_mass)
    # def EBV_corrs_for_no_spectra_sources(Pass in the identified sources w/o spectra): # pass in the instrument(s) as well?
    #     apply EBV correction based on bin (consider Keck vs. MMT, which lines, etc)
    #     return dust-corrected fluxes for the sources

    # def EBV_corrs_for_yes_spectra_sources(Pass in the identified sources w/ spectra):
    #     for each source:
    #         what is its S/N (for all lines?)?

    #         (think of handling keck vs. mmt measurements when both spectra are available)
    #             Will use Keck if Ha and Hb available (NB921, NB973) and MMT for NB816 (Ha/Hb)

    #         apply EBV correction

    #     return dust-corrected fluxes for the sources

    
if __name__ == '__main__':
    main()