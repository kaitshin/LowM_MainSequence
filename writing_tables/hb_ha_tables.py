"""
NAME:
    hb_ha_tables.py

PURPOSE:
    Provides a module where table-writing functions can be called from 
    stack_spectral_data.py for Hb/Ha (Keck) plots
"""

import numpy as np

def Hb_Ha_tables(label, subtitle, flux, o1, xval, pos_flux, dlambda):
    '''
    '''
    ew = 0
    ew_emission = 0
    ew_absorption = 0
    ew_check = 0
    median = 0
    pos_amplitude = 0
    neg_amplitude = 0
    if 'alpha' in label:
        ew = flux/o1[3]
        ew_emission = ew
        ew_check = ew
        median = o1[3]
        pos_amplitude = o1[0]
        neg_amplitude = 0
    elif 'beta' in label and subtitle!='NB816':
        pos0 = o1[6]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)
        neg0 = o1[3]*np.exp(-0.5*((xval-o1[4])/o1[5])**2)

        idx_small = np.where(np.absolute(xval - o1[1]) <= 2.5*o1[2])[0]

        ew = flux/o1[6]
        median = o1[6]
        pos_amplitude = o1[0]
        neg_amplitude = o1[3]

        if (neg_amplitude > 0): 
            neg_amplitude = 0
            ew = pos_flux/o1[6]
            ew_emission = ew
            ew_check = ew
        else:
            pos_corr = np.sum(dlambda * (pos0[idx_small] - o1[6]))
            ew_emission = pos_corr / o1[6]
            neg_corr = np.sum(dlambda * neg0[idx_small])
            ew_absorption = neg_corr / o1[6]
            ew_check = ew_emission + ew_absorption
    #endif

    return (ew, ew_emission, ew_absorption, ew_check, median, pos_amplitude, neg_amplitude)
#enddef