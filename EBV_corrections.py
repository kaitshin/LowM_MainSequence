from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table
from create_ordered_AP_arrays import create_ordered_AP_arrays
import numpy as np, matplotlib.pyplot as plt
import plotting.general_plotting as general_plotting

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'


def get_bins(masslist_MMT, masslist_Keck, massZlist_MMT, massZlist_Keck):
    '''
    Helper function for main. Flattens ndarrays of bins into a bins array
    '''
    massbins_MMT = np.append(masslist_MMT[:,0], masslist_MMT[-1,-1])
    massbins_Keck = np.append(masslist_Keck[:,0], masslist_Keck[-1,-1])
    massZbins_MMT = np.append(massZlist_MMT[:,0], massZlist_MMT[-1,-1])
    massZbins_Keck = np.append(massZlist_Keck[:,0], massZlist_Keck[-1,-1])

    return massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck


def fix_masses_out_of_range(masses_MMT_ii, masses_Keck_ii, masstype):
    '''
    Fixes masses that were not put into bins because they were too low.
    Reassigns them to the lowest mass bin. 

    Does this for both stlrmass bins and stlrmassZ bins.
    '''
    if masstype=='stlrmass':
        masses_MMT_ii = np.array([1 if x==0 else x for x in masses_MMT_ii])
        masses_Keck_ii = np.array([1 if x==0 else x for x in masses_Keck_ii])
    else: #=='stlrmassZ'
        masses_MMT_ii = np.array(['1'+x[1:] if x[0]=='0' else x for x in masses_MMT_ii])
        masses_Keck_ii = np.array(['1'+x[1:] if x[0]=='0' else x for x in masses_Keck_ii])

    return masses_MMT_ii, masses_Keck_ii


def bins_table_no_spectra(indexes, NAME0, AP, stlr_mass, massbins_MMT, massbins_Keck, 
    massZbins_MMT, massZbins_Keck):
    '''
    Creates and returns a table of bins as such:

    NAME             stlrmass filter stlrmassbin_MMT stlrmassbin_Keck stlrmassZbin_MMT stlrmassZbin_Keck
    ---------------- -------- ------ --------------- ---------------- ---------------- -----------------
    Ha-NB973_178201  8.4      NB973  4               3                2-NB973          2-NB973

    TODO: for 'indexes == yes_spectra', include S/N arrays etc.?
    '''
    names = NAME0[indexes]
    masses = stlr_mass[indexes]

    # get redshift bins
    filts = np.array([x[x.find('Ha-')+3:x.find('Ha-')+8] for x in names])
    
    # this ensures that NB704+NB921 dual emitters will be placed in Ha-NB704 bins
    #  since these sources are likely OIII-NB704 and Ha-NB921
    dual_iis = [x for x in range(len(names)) if 'Ha-NB704' in names[x] and 'Ha-NB921' in names[x]]
    filts[dual_iis] = 'NB704'
    
    # this ensures that the NB816+NB921 dual emitter will be placed in the NB921 bin
    #  we decided this based on visual inspection of the photometry
    dual_ii2 = [x for x in range(len(names)) if 'Ha-NB816' in names[x] and 'Ha-NB921' in names[x]]
    filts[dual_ii2] = 'NB921'

    # get stlrmass bins
    masses_MMT_ii = np.digitize(masses, massbins_MMT)
    masses_Keck_ii = np.digitize(masses, massbins_Keck)
    masses_MMT, masses_Keck = fix_masses_out_of_range(masses_MMT_ii, masses_Keck_ii, 'stlrmass')
    
    # get stlrmassZ bins
    ii0 = 0
    ii1 = 0
    massZs_MMT = np.array([])
    massZs_Keck = np.array([])
    for ff in ['NB704','NB711','NB816','NB921','NB973']:
        jj = len(np.where(ff==massZlist_filts_MMT)[0])        
        
        good_filt_iis = np.array([x for x in range(len(filts)) if filts[x]==ff])
        
        if ff=='NB973':
            jj += 1
        
        mass_MMT_iis  = np.digitize(masses[good_filt_iis], massZbins_MMT[ii0:ii0+jj])
        massZ_MMT_iis = np.array([str(x)+'-'+ff for x in mass_MMT_iis])
        massZs_MMT  = np.append(massZs_MMT, massZ_MMT_iis)
        
        ii0+=jj
        if 'NB9' in ff:
            kk = len(np.where(ff==massZlist_filts_Keck)[0])

            if ff=='NB973':
                kk += 1
                
            mass_Keck_iis  = np.digitize(masses[good_filt_iis], massZbins_Keck[ii1:ii1+kk])
            massZ_Keck_iis = np.array([str(x)+'-'+ff for x in mass_Keck_iis])
            massZs_Keck = np.append(massZs_Keck, massZ_Keck_iis)
            
            ii1+=kk
        else:
            massZs_Keck = np.append(massZs_Keck, np.array(['N/A']*len(good_filt_iis)))
            masses_Keck[good_filt_iis] = -99
    
    # putting sources with m < m_min_bin in the lowest-massZ mass bin
    massZs_MMT, massZs_Keck = fix_masses_out_of_range(massZs_MMT, massZs_Keck, 'stlrmassZ')

        
    tab0 = Table([names, masses, filts, masses_MMT, masses_Keck, massZs_MMT, massZs_Keck], 
        names=['NAME', 'stlrmass', 'filter', 'stlrmassbin_MMT', 'stlrmassbin_Keck', 
               'stlrmassZbin_MMT', 'stlrmassZbin_Keck'])

    return tab0


def EBV_corrs_no_spectra(tab_no_spectra):
    '''
    '''
    EBV_corrs = np.array([-100.0]*len(tab_no_spectra))

    # if there's no mass:
    no_mass_iis = np.array([x for x in range(len(tab_no_spectra)) if tab_no_spectra['stlrmass'][x] < 0])
    if len(no_mass_iis) > 0:
        print 'TODO: implement using redshift bins to apply EBV corrections'
        break

    # if there is mass:
    #  if filt == 'NB704', 'NB711', or 'NB816':
    for filt in ['NB704', 'NB711', 'NB816']:
        filt_ii = np.array([x for x in range(len(tab_no_spectra)) if tab_no_spectra['filter'][x]==filt])

        # use stlrmassZ MMT ha/hb if possible
        temp_ii = np.array([x for x in range(len(filt_ii)) if '0-' not in tab_no_spectra['stlrmassZbin_MMT'][filt_ii][x]])

        # tab_filt_ii = [x for x in range(len(mmt_mz_EBV_hahb[:5])) if mmt_mz['filter'][x]==filt][0]
        # EBV_corrs[filt_ii][temp_ii] = mmt_mz_EBV_hahb[tab_filt_ii]

        # else use stlrmass MMT ha/hb
        temp_ii = np.array([x for x in range(len(filt_ii)) if '0-' in tab_no_spectra['stlrmassZbin_MMT'][filt_ii][x]])
        for ii in temp_ii:
            new_ii = filt_ii[ii]
            tab_filt_ii = tab_no_spectra['stlrmassbin_MMT'][new_ii] - 1
            EBV_corrs[new_ii] = mmt_m_EBV_hahb[tab_filt_ii]
    
    #  elif filt == 'NB921':
    filt_ii = np.array([x for x in range(len(tab_no_spectra)) if tab_no_spectra['filter'][x]=='NB921'])
    #  use stlrmassZ Keck ha/hb if possible
    tab_filt = [x for x in range(len(keck_mz_EBV_hahb)) if keck_mz['filter'][x]=='NB921']
    temp_ii = np.array([x for x in range(len(filt_ii)) if '0-' not in tab_no_spectra['stlrmassZbin_Keck'][filt_ii][x]])
    sub_tab = tab_no_spectra['stlrmassZbin_Keck'][filt_ii][temp_ii]
    for ii in range(len(tab_filt)):
        matched = np.array([x for x in range(len(sub_tab)) if sub_tab[x] == str(ii+1)+'-'+filt])
        EBV_corrs[filt_ii[matched]] = keck_mz_EBV_hahb[ii]
    if len(temp_ii) != len(filt_ii):
        print 'TODO: implement the following condititions'
    #         else use stlrmassZ MMT hg/hb
    #         else use stlrmass Keck ha/hb
    #         else use stlrmass MMT hg/hb
        break

    #  elif filt == 'NB973':
    filt_ii = np.array([x for x in range(len(tab_no_spectra)) if tab_no_spectra['filter'][x]=='NB973'])
    #  use stlrmassZ Keck ha/hb if possible
    tab_filt = [x for x in range(len(keck_mz_EBV_hahb)) if keck_mz['filter'][x]=='NB973']
    temp_ii = np.array([x for x in range(len(filt_ii)) if '0-' not in tab_no_spectra['stlrmassZbin_Keck'][filt_ii][x]])
    sub_tab = tab_no_spectra['stlrmassZbin_Keck'][filt_ii][temp_ii]
    for ii in range(len(tab_filt)):
        matched = np.array([x for x in range(len(sub_tab)) if sub_tab[x] == str(ii+1)+'-'+filt])
        EBV_corrs[filt_ii[matched]] = keck_mz_EBV_hahb[ii]
    if len(temp_ii) != len(filt_ii):
        print 'TODO: implement the following condititions'
    #         else use stlrmassZ MMT hg/hb
    #         else use stlrmass Keck ha/hb
    #         else use stlrmass MMT hg/hb
        break

    return EBV_corrs


def main():
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

    # defining other useful data structs
    inst_dict = {}
    inst_dict['MMT']  = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,']
    inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,','Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,']


    # limit all data to Halpha emitters only
    ha_ii = np.array([x for x in range(len(NAME0)) if 'Ha' in NAME0[x]])
    NAME0     = NAME0[ha_ii]
    zspec0    = zspec0[ha_ii]
    inst_str0 = inst_str0[ha_ii]
    stlr_mass = stlr_mass[ha_ii]
    AP        = AP[ha_ii]


    # getting indexes for sources with and without spectra
    no_spectra  = np.where((zspec0 <= 0) | (zspec0 > 9))[0]
    yes_spectra = np.where((zspec0 >= 0) & (zspec0 < 9))[0]

    # getting rid of special case:
    bad_highz_gal = np.where(NAME0[no_spectra]=='Ha-NB816_174829_Ha-NB921_187439_Lya-IA598_163379')[0]
    no_spectra = np.delete(no_spectra, bad_highz_gal)


    # reading in EBV data tables & getting relevant EBV cols
    mmt_z   = asc.read(FULL_PATH+'Composite_Spectra/Redshift/MMT_stacked_spectra_data.txt')
    mmt_z_EBV_hahb = np.array(mmt_z['E(B-V)_hahb'])
    mmt_z_EBV_hghb = np.array(mmt_z['E(B-V)_hghb'])
    
    keck_z  = asc.read(FULL_PATH+'Composite_Spectra/Redshift/Keck_stacked_spectra_data.txt')
    keck_z_EBV_hahb = np.array(keck_z['E(B-V)_hahb'])
    
    mmt_m   = asc.read(FULL_PATH+'Composite_Spectra/StellarMass/MMT_all_five_data.txt')
    mmt_m_EBV_hahb = np.array(mmt_m['E(B-V)_hahb'])
    mmt_m_EBV_hghb = np.array(mmt_m['E(B-V)_hghb'])
    
    keck_m  = asc.read(FULL_PATH+'Composite_Spectra/StellarMass/Keck_all_five_data.txt')
    keck_m_EBV_hahb = np.array(keck_m['E(B-V)_hahb'])
    
    mmt_mz  = asc.read(FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt')
    mmt_mz_EBV_hahb = np.array(mmt_mz['E(B-V)_hahb'])
    mmt_mz_EBV_hghb = np.array(mmt_mz['E(B-V)_hghb'])
    
    keck_mz = asc.read(FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt')
    keck_mz_EBV_hahb = np.array(keck_mz['E(B-V)_hahb'])


    # mass 'bin' lists made by reading in from files generated by the stack plots
    masslist_MMT  = np.array([x.split('-') for x in mmt_m['stlrmass_bin']], dtype=float) 
    masslist_Keck = np.array([x.split('-') for x in keck_m['stlrmass_bin']], dtype=float)

    # same with massZ 'bin' lists (also getting the filts)
    massZlist_MMT  = np.array([x.split('-') for x in mmt_mz['stlrmass_bin'] if x!='N/A'], dtype=float)
    massZlist_Keck = np.array([x.split('-') for x in keck_mz['stlrmass_bin']], dtype=float)
    massZlist_filts_MMT  = np.array([mmt_mz['filter'][x] for x in range(len(mmt_mz)) if mmt_mz['stlrmass_bin'][x]!='N/A'])
    massZlist_filts_Keck = np.array([keck_mz['filter'][x] for x in range(len(keck_mz))])

    # splitting the 'bin' lists from above into a flattened bins array
    massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck = get_bins(masslist_MMT, 
        masslist_Keck, massZlist_MMT, massZlist_Keck)
    

    # getting tables of which bins the sources fall into (for eventual EBV corrections)
    tab_no_spectra = bins_table_no_spectra(no_spectra, NAME0, AP, stlr_mass, 
        massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck)
    
    # tab_yes_spectra = bins_table(yes_spectra, NAME0, AP, stlr_mass, 
    #     massbins_MMT, massbins_Keck, massZbins_MMT, massZbins_Keck)

    # start getting those corrections!
    # EBV_corrs_no_spectra(no_spectra)
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

    # MMT_z_data = asc.read()

    
if __name__ == '__main__':
    main()