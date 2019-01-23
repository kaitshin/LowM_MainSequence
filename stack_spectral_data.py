"""
NAME:
    stack_spectral_data.py

PURPOSE:
    This code creates a PDF file with 15 subplots, filter-emission line
    row-major order, to show all the MMT and Keck spectral data stacked and
    plotted in a 'de-redshifted' frame.
    Specific to SDF data.

    Depends on combine_spectral_data.py and create_ordered_AP_arrays.py
    Depends on write_spectral_coverage_table.py for easy MMT Ha-NB921 coverage info

INPUTS:
    'Catalogs/NB_IA_emitters.nodup.colorrev.fix.fits'
    'Catalogs/nb_ia_zspec.txt'
    'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX.fout'
    'Spectra/spectral_MMT_grid_data.txt'
    'Spectra/spectral_MMT_grid.fits'
    'Spectra/spectral_Keck_grid_data.txt'
    'Spectra/spectral_Keck_grid.fits'

OUTPUTS:
    'Composite_Spectra/Redshift/MMT_spectra_vals/'+subtitle+'.txt''
    'Composite_Spectra/Redshift/MMT_stacked_spectra_data.txt'
    'Composite_Spectra/StellarMass/MMT_spectra_vals/'+subtitle[10:]+'.txt'
    'Composite_Spectra/StellarMass/MMT_all_five_data.txt'
    'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt'
    'Composite_Spectra/Redshift/Keck_spectra_vals/'+subtitle+'.txt'
    'Composite_Spectra/Redshift/Keck_stacked_spectra_data.txt'
    'Composite_Spectra/StellarMass/Keck_spectra_vals/'+subtitle[10:]+'.txt'
    'Composite_Spectra/StellarMass/Keck_all_five_data.txt'
    'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt'
    'Composite_Spectra/Redshift/MMT_stacked_spectra.pdf'
    'Composite_Spectra/StellarMass/MMT_all_five.pdf'
    'Composite_Spectra/StellarMassZ/MMT_stlrmassZ.pdf'
    'Composite_Spectra/Redshift/Keck_stacked_spectra.pdf'
    'Composite_Spectra/StellarMass/Keck_all_five.pdf'
    'Composite_Spectra/StellarMassZ/Keck_stlrmassZ.pdf'
"""

import numpy as np, numpy.ma as ma, matplotlib.pyplot as plt
import plotting.hg_hb_ha_plotting as MMT_plotting
import plotting.hb_ha_plotting as Keck_plotting
import plotting.general_plotting as general_plotting
import writing_tables.hg_hb_ha_tables as MMT_twriting
import writing_tables.hb_ha_tables as Keck_twriting
import writing_tables.general_tables as general_twriting
from analysis.balmer_fit import get_best_fit3, get_baseline_median
from analysis.composite_errors import composite_errors
from analysis.sdf_stack_data import stack_data
from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table, vstack
from create_ordered_AP_arrays import create_ordered_AP_arrays
from matplotlib.backends.backend_pdf import PdfPages
from analysis.cardelli import *   # k = cardelli(lambda0, R=3.1)
from astropy import units as u

MIN_NUM_PER_BIN = 10
MAX_NUM_OF_BINS = 5
SEED_ORIG = 19823

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'

def correct_instr_AP(indexed_AP, indexed_inst_str0, instr):
    '''
    Returns the indexed AP_match array based on the 'matctot = 3h_index' from
    plot_MMT/Keck_Ha
    '''
    for ii in range(len(indexed_inst_str0)):
        if instr == 'MMT': # indexed_inst_str0[ii]=='MMT,' is fine
            if (indexed_inst_str0[ii]=='merged,FOCAS,' or indexed_inst_str0[ii] == 'MMT,FOCAS,' 
                or indexed_inst_str0[ii] == 'MMT,Keck,' or indexed_inst_str0[ii] == 'merged,'):
                indexed_AP[ii] = indexed_AP[ii][:5]
        elif instr=='Keck': # indexed_inst_str0[ii]=='Keck,' is fine
            if (indexed_inst_str0[ii] == 'merged,' or indexed_inst_str0[ii]=='merged,FOCAS,'):
                indexed_AP[ii] = indexed_AP[ii][6:]
        #endif
    #endfor
    return indexed_AP
#enddef

def HG_HB_EBV(hg, hb):
    '''
    instr is always MMT
    '''
    hg = np.array(hg)
    hb = np.array(hb)
    hghb = np.array([0.468 if x > 0.468 else x for x in hg/hb])
    EBV_hghb = np.log10((hghb)/0.468)/(-0.4*(k_hg - k_hb))
    EBV_hghb = np.array([-99.0 if np.isnan(x) else x for x in EBV_hghb])
    return EBV_hghb
#enddef

def get_HB_NB921_flux(bintype='redshift'):
    '''
    '''
    cvg = asc.read(FULL_PATH+'Composite_Spectra/MMT_spectral_coverage.txt')
    nb921 = np.array([x for x in range(len(cvg)) if cvg['filter'][x]=='NB921' and cvg['HB_cvg'][x]=='YES'])
    nb921_ha = np.array([x for x in range(len(nb921)) if cvg['HA_cvg'][nb921][x] == 'YES'])

    # print '>>>>>>>HB IDs:', cvg['ID'][nb921].data ##PRINT STATEMENT

    if bintype=='redshift':
        flux_arr = np.array([-99.0])
    else:
        flux_arr = np.array([-99.0]*5)

    for i in range(len(flux_arr)):
        # i = index of array (0-indexed)
        i0 = np.array([])
        if bintype == 'StellarMassZ':
            # i+1 = index of bin (1-indexed)
            bin_i = np.array([x for x in range(len(nb921_ha)) if str(i+1)+'-' in cvg['stlrmassZbin'][nb921[nb921_ha]].data[x]])
            if len(bin_i) < 2:
                continue
            i0 = np.array([x for x in range(len(gridap)) if gridap[x] in cvg['AP'][nb921[nb921_ha[bin_i]]]])
        elif bintype == 'StlrMass':
            bin_i = np.array([x for x in range(len(nb921_ha)) if i+1==cvg['stlrmassbin'][nb921[nb921_ha]].data[x]])
            if len(bin_i) < 2:
                continue
            i0 = np.array([x for x in range(len(gridap)) if gridap[x] in cvg['AP'][nb921[nb921_ha[bin_i]]]])
        else:
            i0 = np.array([x for x in range(len(gridap)) if gridap[x] in cvg['AP'][nb921[nb921_ha]]])
        
        zs = np.array(gridz[i0])
        good_z2 = np.where((zs >= 0.385) & (zs <= 0.429))[0]
        zs = np.average(zs[good_z2])
        dlambda = (x0[1]-x0[0])/(1+zs)

        xval, yval, len_input_index, stacked_indexes, avgz, minz, maxz = stack_data(grid_ndarr, gridz, i0,
            x0, 3700, 6700, dlambda, ff='NB921', instr='MMT')

        # calculating flux for subtitle=='NB921' emissions
        xmin0 = 4801
        xmax0 = 4921

        good_ii = np.array([x for x in range(len(xval)) if xval[x] >= xmin0 and xval[x] <= xmax0])
        xval = xval[good_ii]
        yval = yval[good_ii]

        good_ii = [ii for ii in range(len(yval)) if not np.isnan(yval[ii])] # not NaN
        xval = xval[good_ii]
        yval = yval[good_ii]

        o1 = get_best_fit3(xval, yval, r'H$\beta$')
        pos0 = o1[5]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)
        neg0 = o1[3]*np.exp(-0.5*((xval-o1[1])/o1[4])**2)
        idx_small = np.where(np.absolute(xval - o1[1]) <= 2.5*o1[2])[0]
        flux = np.sum(dlambda * (pos0[idx_small] - o1[5] - neg0[idx_small]))
        
        flux_arr[i] = flux

    return flux_arr
#enddef

def HA_HB_EBV(ha, hb, instr, bintype='redshift', filt='N/A'):
    '''
    '''
    ha = np.array(ha)
    hb = np.array(hb)

    hahb = np.array([2.86 if (x < 2.86 and x > 0) else x for x in ha/hb])
    EBV_hahb = np.log10((hahb)/2.86)/(-0.4*(k_ha - k_hb))

    if instr=='MMT' and bintype=='redshift':
        EBV_hahb[-1] = -99.0 #no nb973 halpha
    elif instr=='MMT' and bintype=='StellarMassZ' and filt=='NB973':
        EBV_hahb[:] = -99.0 #no nb973 halpha
    elif instr=='MMT' and bintype=='StellarMassZ' and filt=='NB921':
        EBV_hahb[:2] = -99.0 #no nb921 halpha for the lowest two bins
    elif instr=='Keck' and bintype=='redshift':
        EBV_hahb[0] = -99.0 #no nb816 hbeta

    EBV_hahb = np.array([-99.0 if np.isnan(x) else x for x in EBV_hahb])

    return EBV_hahb
#enddef

def split_into_bins(masses, n):
    '''
    '''
    perc_arrs = np.array([np.percentile(masses, x) for x in np.arange(0, 100+100.0/n, 100.0/n)[1:]])
    
    index_arrs = []
    for i in range(n):
        if i == 0:
            index = [x for x in range(len(masses)) if (masses[x]>0 and masses[x]<=perc_arrs[i])]
        else:
            index = [x for x in range(len(masses)) if (masses[x]>perc_arrs[i-1] and masses[x]<=perc_arrs[i])]
        index_arrs.append(index)
    
    if min([len(i) for i in index_arrs]) < MIN_NUM_PER_BIN and n != 2:
        return 'TOO SMALL'
    else:
        return index_arrs
#enddef

def plot_MMT_stlrmass(index_list=[], pp=None, title='', bintype='StlrMass', publ=True, instr00='MMT'):
    '''
    TODO(document)
    TODO(implement flexible stellar mass bin-readings)
    TODO(implement flexible file-naming)
        (nothing from the command line -- default into 5 bins by percentile)
        (number n from the command line -- make n bins by percentile)
        (file name from the command line -- flag to read the stellar mass bins from that ASCII file)
    TODO(get rid of assumption that there's only one page)
    '''
    if index_list == []:
        print '>MMT STELLARMASS STACKING'
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    (HG_flux, HB_flux, HA_flux, NII_6548_flux, NII_6583_flux,
        HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr,
        HG_EW_abs, HB_EW_abs, HG_continuum, HB_continuum, HA_continuum,
        HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude) = table_arrays
    (num_sources, num_stack_HG, num_stack_HB, num_stack_HA, avgz_arr, minz_arr, maxz_arr,
        stlrmass_bin_arr, avg_stlrmass_arr, min_stlrmass_arr, max_stlrmass_arr,
        IDs_arr, HA_RMS, HB_RMS, HG_RMS, 
        HA_ERR, HB_ERR, HG_ERR) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    if index_list == []:
        index_list = general_plotting.get_index_list2(NAME0, stlr_mass, inst_str0, zspec0, inst_dict, 'MMT')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('MMT')

    f, axarr = plt.subplots(5, 3)
    f.set_size_inches(8, 11)
    ax_list = np.ndarray.flatten(axarr)

    subplot_index=0
    # this for-loop stacks by stlr mass
    for (match_index) in (index_list):
        if len(match_index) == 0:
            print 'this is an empty bin'
            [arr.append(0) for arr in table_arrays]
            num_sources.append(0)
            num_stack_HG.append(0)
            num_stack_HB.append(0)
            num_stack_HA.append(0)
            avgz_arr.append(0)
            minz_arr.append(0)
            maxz_arr.append(0)
            IDs_arr.append('N/A')
            stlrmass_bin_arr.append('N/A')
            avg_stlrmass_arr.append(0)
            min_stlrmass_arr.append(0)
            max_stlrmass_arr.append(0)
            HA_RMS.append(0)
            HB_RMS.append(0)
            HG_RMS.append(0)
            HA_ERR.append(np.zeros((1,2))[0])
            HB_ERR.append(np.zeros((1,2))[0])
            HG_ERR.append(np.zeros((1,2))[0])
            for i in range(3):
                ax = ax_list[subplot_index]
                label = label_list[i]
                MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
                subplot_index += 1
            continue
        #endif

        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)

        subtitle='stlrmass: '+str(min(stlr_mass[match_index]))+'-'+str(max(stlr_mass[match_index]))
        print '>>>', subtitle
        avg_stlrmass_arr.append(np.mean(stlr_mass[match_index]))
        min_stlrmass_arr.append(np.min(stlr_mass[match_index]))
        max_stlrmass_arr.append(np.max(stlr_mass[match_index]))

        if bintype=='StellarMassZ':
            zs = np.mean(zspec0[match_index]) # avgz
            dlambda = (x0[1]-x0[0])/(1+zs)  #  < -- avgz changes for ea. redshift sample
        else: # bintype = 'StlrMass'
            dlambda = 0.1

        xval, yval, len_input_index, stacked_indexes, avgz, minz, maxz = stack_data(grid_ndarr, gridz, input_index,
            x0, 3700, 6700, dlambda, instr='MMT')
        num_sources.append(len_input_index[0])
        avgz_arr.append(avgz)
        minz_arr.append(minz)
        maxz_arr.append(maxz)
        stlrmass_bin_arr.append(subtitle[10:])

        # appending to the ID columns
        mm0 = [x for x in range(len(AP)) if any(y in AP[x][:5] for y in gridap[stacked_indexes])] # gridap ordering -> NBIA ordering
        IDs_arr.append(','.join(NAME0[mm0]))

        # writing the spectra table
        table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
        spectra_file_path = FULL_PATH+'Composite_Spectra/StellarMass/MMT_spectra_vals/'+subtitle[10:]+'.txt'
        asc.write(table0, spectra_file_path, format='fixed_width', delimiter=' ', overwrite=True)

        pos_flux_list = []
        flux_list = []
        flux_niib_list = []
        ew_list = []
        ew_abs_list = []
        pos_amplitude_list = []
        neg_amplitude_list = []
        pos_sigma_list = []
        neg_sigma_list = []
        median_list = []
        # calculating flux for NII emissions & rms of the emission lines
        for i, rms_arr, err_arr in zip(range(3), [HG_RMS, HB_RMS, HA_RMS], [HG_ERR, HB_ERR, HA_ERR]):
            xmin0 = xmin_list[i]
            xmax0 = xmax_list[i]
            ax = ax_list[subplot_index+i]
            label = label_list[i]
            try:
                ax, flux, flux2, flux3, pos_flux, o1 = MMT_plotting.subplots_plotting(
                    ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol, len_input_index[i])
                pos_flux_list.append(pos_flux)
                flux_list.append(flux)
                flux_niib_list.append(flux3)

                # rms calculations for the flux for +/- 2.5sigma
                good_ii = np.array([x for x in range(len(xval)) if xval[x] >= xmin0 and xval[x] <= xmax0
                    and not np.isnan(yval[x])])
                med0, std0 = get_baseline_median(xval[good_ii], yval[good_ii], label)
                npix = 5*o1[2]/dlambda  # o1[2] is the positive emission gaussian
                rms = std0 * dlambda * np.sqrt(npix) 
                rms_arr.append(rms)                

                ## calculating composites error bars
                flux_err = composite_errors(flux, rms, seed_i=SEED_ORIG+subplot_index, label=label)
                err_arr.append(flux_err[0])
                # print label, 'errs/1E-18:', flux_err[0]/1E-18
            except IndexError:
                print 'Not enough sources to stack (less than two)'
                rms_arr.append(0)
                err_arr.append(np.zeros((1,2))[0])
                continue
            finally:
                (ew, ew_emission, ew_absorption, median, pos_amplitude, 
                    neg_amplitude) = MMT_twriting.Hg_Hb_Ha_tables(label, flux, 
                    o1, xval, pos_flux, dlambda)
                table_arrays = general_twriting.table_arr_appends(i, subtitle,
                    table_arrays, flux, flux2, flux3, ew, ew_emission, ew_absorption, 
                    median, pos_amplitude, neg_amplitude, 'MMT')
                ew_list.append(ew)
                ew_abs_list.append(ew_absorption)
                if not (subtitle=='NB973' and i==2):
                    pos_amplitude_list.append(pos_amplitude)
                    neg_amplitude_list.append(neg_amplitude)
                    pos_sigma_list.append(o1[2])
                    if i==2:
                        neg_sigma_list.append(0)
                    else:
                        neg_sigma_list.append(o1[5])
                    median_list.append(median)
                else:
                    pos_amplitude_list.append(0)
                    neg_amplitude_list.append(0)
                    pos_sigma_list.append(0)
                    neg_sigma_list.append(0)
                    median_list.append(0)
                #endif
            #endtry
        #endfor
        
        # plotting the emission line spectra (w/ relevant information) for the mass bin
        for i, arr in zip(range(3), [HG_RMS, HB_RMS, HA_RMS]):
            label = label_list[i] + ' ('+str(len_input_index[i])+')'
            if i == 0:
                num_stack_HG.append(int(len_input_index[i]))
            elif i == 1:
                num_stack_HB.append(int(len_input_index[i]))
            else: # i == 2:
                num_stack_HA.append(int(len_input_index[i]))

            ax = ax_list[subplot_index]
            try:
                pos_flux = pos_flux_list[i]
                flux = flux_list[i]
                rms = arr[subplot_index/3]
                flux_niib = flux_niib_list[i]
                ew = ew_list[i]
                ew_abs = ew_abs_list[i]

                if not (subtitle=='NB973' and i==2) and len_input_index[i] > 1:
                    pos_amplitude = pos_amplitude_list[i]
                    neg_amplitude = neg_amplitude_list[i]
                    pos_sigma = pos_sigma_list[i]
                    neg_sigma = neg_sigma_list[i]
                    median = median_list[i]
                    if title=='NB921' and i==1:
                        hb_nb921_flux = get_HB_NB921_flux(bintype=bintype)[subplot_index/3]
                    else:
                        hb_nb921_flux = 0
                    ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, pos_flux, flux,
                        pos_amplitude, neg_amplitude, pos_sigma, neg_sigma, median, hb_nb921_flux, 
                        ew=ew, ew_abs=ew_abs, flux_niib=flux_niib, rms=rms, bintype=bintype, ftitle=title)
                else:
                    ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, pos_flux, flux, 
                        ew=ew, ew_abs=ew_abs, flux_niib=flux_niib, rms=rms, bintype=bintype, ftitle=title)
            except IndexError: # assuming there's no pos_flux or flux value
                ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, bintype=bintype, ftitle=title)
            subplot_index+=1
        #endfor
    #endfor
    if title=='':
        f = general_plotting.final_plot_setup(f, r'MMT detections of H$\alpha$ emitters')
    else:
        f = general_plotting.final_plot_setup(f, title, publ, instr00)

    plt.subplots_adjust(hspace=0, wspace=0.05)

    if pp == None:
        plt.savefig(FULL_PATH+'Composite_Spectra/StellarMass/MMT_all_five.pdf')
        subtitle_list = np.array(['all']*len(stlrmass_bin_arr))
    else:
        pp.savefig()
        subtitle_list = np.array([title]*len(stlrmass_bin_arr))
    plt.close()

    if title=='NB921':
        HA_flux = np.array(HA_flux)
        HA_flux[:2] = -99
    elif title=='NB973':
        HA_flux = np.array([-99]*5)

    # assigning pos/neg errs to separate cols
    HG_errs_neg = np.array(HG_ERR)[:,0]
    HG_errs_pos = np.array(HG_ERR)[:,1]
    HB_errs_neg = np.array(HB_ERR)[:,0]
    HB_errs_pos = np.array(HB_ERR)[:,1]
    HA_errs_neg = np.array(HA_ERR)[:,0]
    HA_errs_pos = np.array(HA_ERR)[:,1]


    HB_NB921_flux = np.copy(HB_flux)
    if title=='NB921' or bintype=='StlrMass':
        HB_NB921_flux = get_HB_NB921_flux(bintype=bintype)

    # getting flux ratio errs
    FLUX_hghb_errs = composite_errors([HG_flux, HB_flux], [HG_RMS, HB_RMS], seed_i=SEED_ORIG+subplot_index, label='HG/HB_flux_rat_errs')
    FLUX_hghb_errs_neg = FLUX_hghb_errs[:,0]
    FLUX_hghb_errs_pos = FLUX_hghb_errs[:,1]

    FLUX_hahb_errs = composite_errors([HA_flux, HB_NB921_flux], [HA_RMS, HB_RMS], seed_i=SEED_ORIG+subplot_index, label='HA/HB_flux_rat_errs')
    FLUX_hahb_errs_neg = FLUX_hahb_errs[:,0]
    FLUX_hahb_errs_pos = FLUX_hahb_errs[:,1]


    # getting EBV and EBV errs
    EBV_hghb = HG_HB_EBV(HG_flux, HB_flux)
    EBV_hghb_errs = composite_errors([HG_flux, HB_flux], [HG_RMS, HB_RMS], seed_i=SEED_ORIG+subplot_index, label='HG/HB')
    EBV_hghb_errs_neg = EBV_hghb_errs[:,0]
    EBV_hghb_errs_pos = EBV_hghb_errs[:,1]

    EBV_hahb = HA_HB_EBV(HA_flux, HB_NB921_flux, 'MMT', bintype, title)
    EBV_hahb_errs = composite_errors([HA_flux, HB_NB921_flux], [HA_RMS, HB_RMS], seed_i=SEED_ORIG+subplot_index, label='HA/HB')
    EBV_hahb_errs_neg = EBV_hahb_errs[:,0]
    EBV_hahb_errs_pos = EBV_hahb_errs[:,1]


    table00 = Table([subtitle_list, stlrmass_bin_arr, num_sources, num_stack_HG, num_stack_HB, num_stack_HA,
        avgz_arr, minz_arr, maxz_arr, 
        avg_stlrmass_arr, min_stlrmass_arr, max_stlrmass_arr, HG_flux, HB_flux, HB_NB921_flux, HA_flux, NII_6548_flux, 
        NII_6583_flux, HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr, HG_EW_abs, HB_EW_abs,
        HG_continuum, HB_continuum, HA_continuum, HG_RMS, HB_RMS, HA_RMS, 
        HG_errs_neg, HG_errs_pos, HB_errs_neg, HB_errs_pos, HA_errs_neg, HA_errs_pos,
        HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude, EBV_hghb, EBV_hahb, 
        EBV_hghb_errs_neg, EBV_hghb_errs_pos, EBV_hahb_errs_neg, EBV_hahb_errs_pos], # IDs_arr
        names=['filter', 'stlrmass_bin', 'num_sources', 'num_stack_HG', 'num_stack_HB', 'num_stack_HA',
        'avgz', 'minz', 'maxz',
        'avg_stlrmass', 'min_stlrmass', 'max_stlrmass', 'HG_flux', 'HB_flux', 'HB_NB921_flux', 'HA_flux', 'NII_6548_flux', 
        'NII_6583_flux', 'HG_EW', 'HB_EW', 'HA_EW', 'HG_EW_corr', 'HB_EW_corr', 'HA_EW_corr', 'HG_EW_abs', 'HB_EW_abs',
        'HG_continuum', 'HB_continuum', 'HA_continuum', 'HG_RMS', 'HB_RMS', 'HA_RMS', 
        'HG_flux_errs_neg', 'HG_flux_errs_pos', 'HB_flux_errs_neg', 'HB_flux_errs_pos', 'HA_flux_errs_neg', 'HA_flux_errs_pos',
        'HG_pos_amplitude', 'HB_pos_amplitude', 'HA_pos_amplitude',
        'HG_neg_amplitude', 'HB_neg_amplitude', 'E(B-V)_hghb', 'E(B-V)_hahb', 
        'E(B-V)_hghb_errs_neg', 'E(B-V)_hghb_errs_pos', 'E(B-V)_hahb_errs_neg', 'E(B-V)_hahb_errs_pos']) # IDs

    if pp != None: return pp, table00

    asc.write(table00, FULL_PATH+'Composite_Spectra/StellarMass/MMT_all_five_data.txt',
        format='fixed_width_two_line', delimiter=' ', overwrite=True)
#enddef

def plot_MMT_stlrmass_z():
    '''
    TODO(document)
    TODO(generalize stellar mass binning functionality?)
    TODO(implement flexible file-naming)
    '''
    print '>MMT STELLARMASS+REDSHIFT STACKING'
    pp = PdfPages(FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ.pdf')
    table00 = None

    mmt_ii = np.array([x for x in range(len(NAME0)) if 
        ('Ha-NB' in NAME0[x] and inst_str0[x] in inst_dict['MMT'] 
            and stlr_mass[x] > 0 and (zspec0[x] > 0 and zspec0[x] < 9))])
    bins_ii_tbl = np.ndarray((5,5), dtype=object)

    bins_ii_tbl_temp = np.ndarray((5,5), dtype=object)
    for ff, ii in zip(['NB7', 'NB816', 'NB921', 'NB973'], [0,1,2,3]):
        filt_ii = np.array([x for x in range(len(mmt_ii)) if 'Ha-'+ff in NAME0[mmt_ii][x]])
        filt_masses = stlr_mass[mmt_ii][filt_ii]
        for n in [5, 4, 3, 2]:
            bins_ii = split_into_bins(filt_masses, n)
            if bins_ii != 'TOO SMALL': break
        for x in range(5 - len(bins_ii)):
            bins_ii.append([])
        bins_ii_tbl[ii] = bins_ii

        for jj in range(len(bins_ii)):
            bins_ii_tbl_temp[ii][jj] = mmt_ii[filt_ii][bins_ii_tbl[ii][jj]]

        if ff=='NB7':
            title='NB704+NB711'
        else:
            title=ff
        print '>>>', title

        pp, table_data = plot_MMT_stlrmass(bins_ii_tbl_temp[ii], pp, title, 'StellarMassZ')
        if table00 == None:
            table00 = table_data
        else:
            table00 = vstack([table00, table_data])
        #endif
    #endfor

    asc.write(table00, FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt',
        format='fixed_width_two_line', delimiter=' ', overwrite=True)
    pp.close()
#enddef

def plot_Keck_stlrmass(index_list=[], pp=None, title='', bintype='StlrMass', publ=True, instr00='Keck'):
    '''
    TODO(document)
    TODO(implement flexible stellar mass bin-readings)
    TODO(implement flexible file-naming)
        (nothing from the command line -- default into 5 bins by percentile)
        (number n from the command line -- make n bins by percentile)
        (file name from the command line -- flag to read the stellar mass bins from that ASCII file)
    TODO(get rid of assumption that there's only one page)
    '''
    if index_list == []:
        print '>KECK STELLARMASS STACKING'
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [], [], [], [])
    (HB_flux, HA_flux, NII_6548_flux, NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr,
        HB_EW_abs, HB_continuum, HA_continuum, HB_pos_amplitude, HA_pos_amplitude,
        HB_neg_amplitude) = table_arrays
    (num_sources, num_stack_HG, num_stack_HB, num_stack_HA, avgz_arr, minz_arr, maxz_arr,
        stlrmass_bin_arr, avg_stlrmass_arr, min_stlrmass_arr, max_stlrmass_arr,
        IDs_arr, HA_RMS, HB_RMS, HA_ERR, HB_ERR) = ([], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [])
    if index_list == []:
        index_list = general_plotting.get_index_list2(NAME0, stlr_mass, inst_str0, zspec0, inst_dict, 'Keck')
    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('Keck', stlr=True)

    f, axarr = plt.subplots(5, 2)
    f.set_size_inches(5, 11)
    ax_list = np.ndarray.flatten(axarr)

    subplot_index=0
    # this for-loop stacks by stlr mass
    for (match_index) in zip(index_list):
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'Keck')
        AP_match = np.array([x for x in AP_match if x != 'INVALID_KECK'], dtype=np.float32)
        
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)

        subtitle='stlrmass: '+str(min(stlr_mass[match_index]))+'-'+str(max(stlr_mass[match_index]))
        print '>>>', subtitle
        avg_stlrmass_arr.append(np.mean(stlr_mass[match_index]))
     	min_stlrmass_arr.append(np.min(stlr_mass[match_index]))
     	max_stlrmass_arr.append(np.max(stlr_mass[match_index]))

        if bintype=='StellarMassZ':
            zs = np.mean(zspec0[match_index]) # avgz
            dlambda = (x0[1]-x0[0])/(1+zs)  #  < -- avgz changes for ea. redshift sample
        else:
            dlambda = 0.1 # xval[1] - xval[0]

        xval, yval, len_input_index, stacked_indexes, avgz, minz, maxz = stack_data(grid_ndarr, gridz, input_index,
            x0, 3800, 6700, dlambda, instr='Keck')
        num_sources.append(len_input_index[0])
        avgz_arr.append(avgz)
        minz_arr.append(minz)
        maxz_arr.append(maxz)
        stlrmass_bin_arr.append(subtitle[10:])

        # appending to the ID columns
        tempgridapstacked_ii = [str(y) for y in gridap[stacked_indexes]]
        mm0 = []
        for x in range(len(AP)):
            for y in tempgridapstacked_ii:
                if len(y)==5: 
                    y = '0'+y
                if y in AP[x]:
                    mm0.append(x)
        #endfor
        IDs_arr.append(','.join(NAME0[mm0]))

        # writing the spectra table
        table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
        spectra_file_path = FULL_PATH+'Composite_Spectra/StellarMass/Keck_spectra_vals/'+subtitle[10:]+'.txt'
        asc.write(table0, spectra_file_path, format='fixed_width', delimiter=' ', overwrite=True)

        pos_flux_list = []
        flux_list = []
        flux_niib_list = []
        ew_list = []
        ew_abs_list = []
        pos_amplitude_list = []
        neg_amplitude_list = []
        pos_sigma_list = []
        neg_sigma_list = []
        median_list = []
        # calculating flux for NII emissions & rms of the emission lines
        for i, rms_arr, err_arr in zip(range(2), [HB_RMS, HA_RMS], [HB_ERR, HA_ERR]):
            xmin0 = xmin_list[i]
            xmax0 = xmax_list[i]
            ax = ax_list[subplot_index+i]
            label = label_list[i]
            try:
                ax, flux, flux2, flux3, pos_flux, o1 = Keck_plotting.subplots_plotting(
                    ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol, subplot_index+i)
                pos_flux_list.append(pos_flux)
                flux_list.append(flux)
                flux_niib_list.append(flux3)

                # rms calculations
                good_ii = np.array([x for x in range(len(xval)) if xval[x] >= xmin0 and xval[x] <= xmax0
                    and not np.isnan(yval[x])])
                med0, std0 = get_baseline_median(xval[good_ii], yval[good_ii], label)
                npix = 5*o1[2]/dlambda  # o1[2] is the positive emission gaussian
                rms = std0 * dlambda * np.sqrt(npix)
                rms_arr.append(rms)

                ## calcluating composites error bars
                flux_err = composite_errors(flux, rms, seed_i=SEED_ORIG+subplot_index, label=label)
                err_arr.append(flux_err[0])
            except IndexError:
                print '(!!) There\'s some unexpected exception or another.'
                rms_arr.append(0)
                err_arr.append(np.zeros((1,2))[0])
                continue
            finally:
                (ew, ew_emission, ew_absorption, median, pos_amplitude, 
                  neg_amplitude) = Keck_twriting.Hb_Ha_tables(label, subtitle, flux, 
                  o1, xval, pos_flux, dlambda)
                table_arrays = general_twriting.table_arr_appends(i, subtitle,
                  table_arrays, flux, flux2, flux3, ew, ew_emission, ew_absorption, 
                  median, pos_amplitude, neg_amplitude, 'Keck')
                ew_list.append(ew)
                ew_abs_list.append(ew_absorption)

                pos_amplitude_list.append(pos_amplitude)
                neg_amplitude_list.append(neg_amplitude)
                pos_sigma_list.append(o1[2])
                if i==0:
                    neg_sigma_list.append(o1[5])
                else:
                    neg_sigma_list.append(0)
                median_list.append(median)
            #endtry
        #endfor

        for i, arr in zip(range(2), [HB_RMS, HA_RMS]):
            label = label_list[i] + ' ('+str(len_input_index[i])+')'
            if i == 0:
                num_stack_HB.append(int(len_input_index[i]))
            else: # i == 1:
                num_stack_HA.append(int(len_input_index[i]))

            ax = ax_list[subplot_index]
            try:
                pos_flux = pos_flux_list[i]
                flux = flux_list[i]
                rms = arr[subplot_index/2]
                flux_niib = flux_niib_list[i]
                ew = ew_list[i]
                ew_abs = ew_abs_list[i]

                pos_amplitude = pos_amplitude_list[i]
                neg_amplitude = neg_amplitude_list[i]
                pos_sigma = pos_sigma_list[i]
                neg_sigma = neg_sigma_list[i]
                median = median_list[i]
                ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index, pos_flux, flux,
                    pos_amplitude, neg_amplitude, pos_sigma, neg_sigma, median, 
                    ew=ew, ew_abs=ew_abs, flux_niib=flux_niib, rms=rms)
            except IndexError: # assuming there's no pos_flux or flux value
                ax = Keck_plotting.subplots_setup(ax, ax_list, label, subtitle, subplot_index)
            subplot_index+=1
        #endfor
    #endfor
    if title=='':
        f = general_plotting.final_plot_setup(f, r'Keck detections of H$\alpha$ emitters')
    else:
        f = general_plotting.final_plot_setup(f, title, publ, instr00)

    plt.subplots_adjust(hspace=0, wspace=0.05)

    if pp == None:
        plt.savefig(FULL_PATH+'Composite_Spectra/StellarMass/Keck_all_five.pdf')
        subtitle_list = np.array(['all']*len(stlrmass_bin_arr))
    else:
        pp.savefig()
        subtitle_list = np.array([title]*len(stlrmass_bin_arr))
    plt.close()

    # assigning pos/neg errs to separate cols
    HB_errs_neg = np.array(HB_ERR)[:,0]
    HB_errs_pos = np.array(HB_ERR)[:,1]
    HA_errs_neg = np.array(HA_ERR)[:,0]
    HA_errs_pos = np.array(HA_ERR)[:,1]


    # getting flux ratio errs
    FLUX_hahb_errs = composite_errors([HA_flux, HB_flux], [HA_RMS, HB_RMS], seed_i=SEED_ORIG+subplot_index, label='HA/HB_flux_rat_errs')
    FLUX_hahb_errs_neg = FLUX_hahb_errs[:,0]
    FLUX_hahb_errs_pos = FLUX_hahb_errs[:,1]

    # getting EBV and EBV errs
    EBV_hahb = HA_HB_EBV(HA_flux, HB_flux, 'Keck', 'stlrmass')
    EBV_hahb_errs = composite_errors([HA_flux, HB_flux], [HA_RMS, HB_RMS], seed_i=SEED_ORIG+subplot_index, label='HA/HB')
    EBV_hahb_errs_neg = EBV_hahb_errs[:,0]
    EBV_hahb_errs_pos = EBV_hahb_errs[:,1]

    table00 = Table([subtitle_list, stlrmass_bin_arr, num_sources, num_stack_HB, num_stack_HA,
        avgz_arr, minz_arr, maxz_arr, 
        avg_stlrmass_arr, min_stlrmass_arr, max_stlrmass_arr, HB_flux, HA_flux, NII_6548_flux, 
        NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr, HB_EW_abs,
        HB_continuum, HA_continuum, HB_RMS, HA_RMS, 
        HB_errs_neg, HB_errs_pos, HA_errs_neg, HA_errs_pos,
        HB_pos_amplitude, HA_pos_amplitude, HB_neg_amplitude, 
        EBV_hahb, EBV_hahb_errs_neg, EBV_hahb_errs_pos], # IDs_arr
        names=['filter', 'stlrmass_bin', 'num_sources', 'num_stack_HB', 'num_stack_HA',
        'avgz', 'minz', 'maxz',
        'avg_stlrmass', 'min_stlrmass', 'max_stlrmass', 'HB_flux', 'HA_flux', 'NII_6548_flux', 
        'NII_6583_flux', 'HB_EW', 'HA_EW', 'HB_EW_corr', 'HA_EW_corr', 'HB_EW_abs',
        'HB_continuum', 'HA_continuum', 'HB_RMS', 'HA_RMS', 
        'HB_flux_errs_neg', 'HB_flux_errs_pos', 'HA_flux_errs_neg', 'HA_flux_errs_pos', 
        'HB_pos_amplitude', 'HA_pos_amplitude', 'HB_neg_amplitude', 
        'E(B-V)_hahb', 'E(B-V)_hahb_errs_neg', 'E(B-V)_hahb_errs_pos']) # 'IDs'

    if pp != None: return pp, table00

    asc.write(table00, FULL_PATH+'Composite_Spectra/StellarMass/Keck_all_five_data.txt',
            format='fixed_width_two_line', delimiter=' ', overwrite=True)
#enddef

def plot_Keck_stlrmass_z():
    '''
    TODO(document)
    TODO(generalize stellar mass binning functionality?)
    TODO(implement flexible file-naming)
    '''
    print '>KECK STELLARMASS+REDSHIFT STACKING'
    pp = PdfPages(FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ.pdf')
    table00 = None
    
    keck_ii = np.array([x for x in range(len(NAME0)) if 
                        ('Ha-NB9' in NAME0[x] and inst_str0[x] in inst_dict['Keck'] 
                         and stlr_mass[x] > 0 and (zspec0[x] > 0 and zspec0[x] < 9))])
    bins_ii_tbl = np.ndarray((2,5), dtype=object)

    bins_ii_tbl_temp = np.ndarray((2,5), dtype=object)
    for ff, ii in zip(['NB921', 'NB973'], [0,1]):
        filt_ii = np.array([x for x in range(len(keck_ii)) if 'Ha-'+ff in NAME0[keck_ii][x]])
        filt_masses = stlr_mass[keck_ii][filt_ii]
        for n in [5, 4, 3, 2]:
            bins_ii = split_into_bins(filt_masses, n)
            if bins_ii != 'TOO SMALL': break
        for x in range(5 - len(bins_ii)):
            bins_ii.append([])
        bins_ii_tbl[ii] = bins_ii

        for jj in range(len(bins_ii)):
            bins_ii_tbl_temp[ii][jj] = keck_ii[filt_ii][bins_ii_tbl[ii][jj]]
    
        title=ff
        print '>>>', title

        pp, table_data = plot_Keck_stlrmass(bins_ii_tbl_temp[ii], pp, title, 'StellarMassZ')
        if table00 == None:
            table00 = table_data
        else:
            table00 = vstack([table00, table_data])
        #endif
    #endfor
    
    asc.write(table00, FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt',
        format='fixed_width_two_line', delimiter=' ', overwrite=True)
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
inst_dict = {} ##used
inst_dict['MMT'] = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,','merged,FOCAS,']
inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,',
                     'Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,','merged,FOCAS,']
tol = 3 #in angstroms, used for NII emission flux calculations ##used

k_hg = cardelli(4341 * u.Angstrom)
k_hb = cardelli(4861 * u.Angstrom)
k_ha = cardelli(6563 * u.Angstrom)

nbia = pyfits.open(FULL_PATH+'Catalogs/NB_IA_emitters.nodup.colorrev.fix.fits')
nbiadata = nbia[1].data
NAME0 = nbiadata['NAME']

zspec = asc.read(FULL_PATH+'Catalogs/nb_ia_zspec.txt',guess=False,
                 Reader=asc.CommentedHeader)
zspec0 = np.array(zspec['zspec0'])
inst_str0 = np.array(zspec['inst_str0']) ##used

fout  = asc.read(FULL_PATH+'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast.GALEX.fout',
                 guess=False,Reader=asc.NoHeader)
stlr_mass = np.array(fout['col7']) ##used
nan_stlr_mass = np.copy(stlr_mass)
nan_stlr_mass[nan_stlr_mass < 0] = np.nan

data_dict = create_ordered_AP_arrays(AP_only = True)
AP = data_dict['AP'] ##used

print '### looking at the MMT grid'
griddata = asc.read(FULL_PATH+'Spectra/spectral_MMT_grid_data.txt',guess=False)
gridz  = np.array(griddata['ZSPEC']) ##used
gridap = np.array(griddata['AP']) ##used
grid   = pyfits.open(FULL_PATH+'Spectra/spectral_MMT_grid.fits')
grid_ndarr = grid[0].data ##used
grid_hdr   = grid[0].header
CRVAL1 = grid_hdr['CRVAL1']
CDELT1 = grid_hdr['CDELT1']
NAXIS1 = grid_hdr['NAXIS1'] #npix
x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1) ##used
# mask spectra that doesn't exist or lacks coverage in certain areas
ndarr_zeros = np.where(grid_ndarr == 0)
mask_ndarr = np.zeros_like(grid_ndarr)
mask_ndarr[ndarr_zeros] = 1
# mask spectra with unreliable redshift
bad_zspec = [x for x in range(len(gridz)) if gridz[x] > 9 or gridz[x] < 0]
mask_ndarr[bad_zspec,:] = 1
grid_ndarr = ma.masked_array(grid_ndarr, mask=mask_ndarr, fill_value=np.nan)

print '### plotting MMT'
plot_MMT_stlrmass_z()
grid.close()

print '### looking at the Keck grid'
griddata = asc.read(FULL_PATH+'Spectra/spectral_Keck_grid_data.txt',guess=False)
gridz  = np.array(griddata['ZSPEC']) ##used
gridap = np.array(griddata['AP']) ##used
grid   = pyfits.open(FULL_PATH+'Spectra/spectral_Keck_grid.fits')
grid_ndarr = grid[0].data ##used
grid_hdr   = grid[0].header
CRVAL1 = grid_hdr['CRVAL1']
CDELT1 = grid_hdr['CDELT1']
NAXIS1 = grid_hdr['NAXIS1'] #npix
x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1) ##used
# mask spectra that doesn't exist or lacks coverage in certain areas
ndarr_zeros = np.where(grid_ndarr == 0)
mask_ndarr = np.zeros_like(grid_ndarr)
mask_ndarr[ndarr_zeros] = 1
# mask spectra with unreliable redshift
bad_zspec = [x for x in range(len(gridz)) if gridz[x] > 9 or gridz[x] < 0]
mask_ndarr[bad_zspec,:] = 1
grid_ndarr = ma.masked_array(grid_ndarr, mask=mask_ndarr)

print '### plotting Keck'
plot_Keck_stlrmass_z()
grid.close()

nbia.close()
print '### done'
#endmain