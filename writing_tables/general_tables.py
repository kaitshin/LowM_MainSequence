"""
NAME:
    general_tables.py

PURPOSE:
    Provides a module where table-writing functions can be called from 
    stack_spectral_data.py for both Hg/Hb/Ha and Hb/Ha (MMT,Keck) plots
"""

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
    if instr == 'MMT':
        (HG_flux, HB_flux, HA_flux, NII_6548_flux, NII_6583_flux,
            HG_EW, HB_EW, HA_EW, HG_EW_corr, HB_EW_corr, HA_EW_corr,
            HG_EW_abs, HB_EW_abs, HG_continuum, HB_continuum, HA_continuum,
            HG_pos_amplitude, HB_pos_amplitude, HA_pos_amplitude,
            HG_neg_amplitude, HB_neg_amplitude) = table_arrays
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
        (HB_flux, HA_flux, NII_6548_flux, NII_6583_flux, HB_EW, HA_EW, HB_EW_corr, HA_EW_corr,
            HB_EW_abs, HB_continuum, HA_continuum, HB_pos_amplitude, HA_pos_amplitude,
            HB_neg_amplitude) = table_arrays
        if i==0 and subtitle!='NB816':
            HB_flux.append(flux)
            HB_EW.append(ew)
            HB_EW_corr.append(ew_emission)
            HB_EW_abs.append(ew_absorption)
            HB_continuum.append(median)
            HB_pos_amplitude.append(pos_amplitude)
            HB_neg_amplitude.append(neg_amplitude)
        if i==0 and subtitle=='NB816':
            HB_flux.append(0)
            HB_EW.append(0)
            HB_EW_corr.append(0)
            HB_EW_abs.append(0)
            HB_continuum.append(0)
            HB_pos_amplitude.append(0)
            HB_neg_amplitude.append(0)
        elif i==1:
            HA_flux.append(flux)
            NII_6548_flux.append(flux2)
            NII_6583_flux.append(flux3)
            HA_EW.append(ew)
            HA_EW_corr.append(ew_emission)
            HA_continuum.append(median)
            HA_pos_amplitude.append(pos_amplitude)
        #endif
    #endif
    return table_arrays
#enddef