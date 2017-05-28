"""
NAME:
    general_tables.py

PURPOSE:
    Provides a module where table-writing functions can be called from 
    stack_spectral_data.py for both Hg/Hb/Ha and Hb/Ha (MMT,Keck) plots
"""

# def table_arr_appends(num, table_arrays, label, subtitle, flux, flux2, flux3, ew, ew_emission, ew_absorption, median, pos_amplitude, neg_amplitude, instr):
def table_arr_appends(i, subtitle, table_arrays, flux, flux2, flux3, ew, ew_emission, ew_absorption, median, pos_amplitude, neg_amplitude, instr):
    '''
    i denotes which column it is. 
    i==0: Hgamma
    i==1: Hbeta
    i==2: Halpha
    
    Appends all the passed in values to the respective arrays in table_arrays
    based on which instr keyword is passed in.

    table_arrays is then returned
    '''
    (HG_flux, HB_flux, HA_flux, NII_6548_flux, NII_6583_flux,
        HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr,
        HG_EW_abs, HB_EW_abs, HG_continuum, HB_continuum, HA_continuum,
        HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
        HG_neg_amplitude, HB_neg_amplitude) = table_arrays

    if instr == 'MMT':
        if i==0:
            HG_flux.append(flux)
            HG_EW.append(ew)
            HG_EW_corr.append(ew_emission)
            HG_EW_abs.append(ew_absorption)
            HG_continuum.append(median)
            HG_pos_amplitude.append(pos_amplitude)
            HG_neg_amplitude.append(neg_amplitude)
        elif i==1:
            HB_flux.append(flux)
            HB_EW.append(ew)
            HB_EW_corr.append(ew_emission)
            HB_EW_abs.append(ew_absorption)
            HB_continuum.append(median)
            HB_pos_amplitude.append(pos_amplitude)
            HB_neg_amplitude.append(neg_amplitude)
        elif i==2 and subtitle!='NB973':
            HA_flux.append(flux)
            NII_6548_flux.append(flux2)
            NII_6583_flux.append(flux3)
            HA_EW.append(ew)
            HA_EW_corr.append(ew_emission)
            HA_continuum.append(median)
            HA_pos_amplitude.append(pos_amplitude)
        elif i==2 and subtitle=='NB973':
            HA_flux.append(0)
            NII_6548_flux.append(0)
            NII_6583_flux.append(0)
            HA_EW.append(0)
            HA_EW_corr.append(0)
            HA_continuum.append(0)
            HA_pos_amplitude.append(0)
        #endif
    elif instr == 'Keck':
        (tablenames, tablefluxes, nii6548fluxes, nii6583fluxes, ewlist, 
            ewposlist , ewneglist, ewchecklist, medianlist, pos_amplitudelist, 
            neg_amplitudelist) = table_arrays
        if not (subtitle=='NB816' and num%2==0):
            tablenames.append(label+'_'+subtitle)
            tablefluxes.append(flux)
            nii6548fluxes.append(flux2)
            nii6583fluxes.append(flux3)
            ewlist.append(ew)
            ewposlist.append(ew_emission)
            ewneglist.append(ew_absorption)
            medianlist.append(median)
            pos_amplitudelist.append(pos_amplitude)
            neg_amplitudelist.append(neg_amplitude)
        #endif
    #endif
    return table_arrays
#enddef