"""
NAME:
    MACT_utils.py

PURPOSE:
    this code has functions that are used across multiple codes
"""

import numpy as np

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'

def get_flux_from_FAST(ID, lambda_arr, byarr=True, fileend='GALEX'):
    '''
    Reads in the relevant SED spectrum file and then interpolates the
    function to obtain a flux, the array of which is then returned.
    '''
    from scipy import interpolate
    from astropy.io import ascii as asc

    if byarr:
        newflux = np.zeros(len(ID))
        for ii in range(len(ID)):
            tempfile = asc.read(FULL_PATH+
                'FAST/outputs/BEST_FITS/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+
                fileend+'_'+str(ID[ii])+'.fit', guess=False,Reader=asc.NoHeader)
            wavelength = np.array(tempfile['col1'])
            flux = np.array(tempfile['col2'])
            f = interpolate.interp1d(wavelength, flux)
            newflux[ii] = f(lambda_arr[ii])

    else:
        tempfile = asc.read(FULL_PATH+
            'FAST/outputs/BEST_FITS/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+
            fileend+'_'+str(ID)+'.fit', guess=False,Reader=asc.NoHeader)
        wavelength = np.array(tempfile['col1'])
        flux = np.array(tempfile['col2'])
        f = interpolate.interp1d(wavelength, flux)
        newflux = f(lambda_arr)

    return newflux


def get_z_arr():
    '''
    defining an approximate redshift array for plot visualization
    '''
    z_arr0 = np.array([7045.0, 7126.0, 8152.0, 9193.0, 9749.0])/HA - 1
    z_arr0 = np.around(z_arr0, 2)
    z_arr  = np.array(z_arr0, dtype='|S9')
    z_arr[0] = ",".join(z_arr[:2])
    z_arr = np.delete(z_arr, 1)
    z_arr  = np.array([x+'0' if len(x)==3 else x for x in z_arr])

    return z_arr


def niiha_oh_determine(x0, type, index=None, silent=None, linear=None):
    '''
    Adapted from Chun Ly 
    
    PURPOSE:
       This code estimates 12+log(O/H) based on strong-line diagnostics. It uses
       emission-line that use [NII]6583, such as [NII]6583/Halpha.

    CALLING SEQUENCE:
       niiha_oh_determine(x0, type, index=index, silent=1)

    INPUTS:
       x0   -- Array of log([NII]6583/Halpha)
       type -- The type of diagnostics to use. The options are:
         'PP04_N2'    -- N2 index calibration of Pettini & Pagel (2004), MNRAS, 348, 59
           - Specify linear keyword to use linear instead of 3rd-order function

    OPTIONAL KEYWORD INPUT:
       index   -- Index of array to determine metallicity
       silent  -- If set, this means that nothing will be printed out
    '''

    if index is None: index = range(len(x0))

    ## Default sets those without metallicity at -1.0
    OH_gas = np.repeat(-1.000, len(x0))


    ######################################
    ## Empirical, PP04                  ##
    ## ---------------------------------##
    ## See Pettini & Pagel (2004)       ##
    ## Eq. A10 of Kewley & Ellison 2008 ##
    ## + on 04/03/2016                  ##
    ## Mod on 14/06/2016                ##
    ######################################
    if type == 'PP04_N2':
        if linear == None:
            OH_gas[index] = 9.37 + 2.03*x0[index] + 1.26*(x0[index])**2 + 0.32*(x0[index])**3
        else:
            print '## Using linear relation!'
            # Bug found. Mod on 30/06/2016 OH_gas -> OH_gas[index]
            OH_gas[index] = 8.90 + 0.57 * x0[index] #xt0
    #endif

    return OH_gas
