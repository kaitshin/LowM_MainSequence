"""
NAME:
    MACT_utils.py

PURPOSE:
    this code has functions that are used across multiple codes
"""

import numpy as np
from analysis.cardelli import *   # k = cardelli(lambda0, R=3.1)

FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'

# emission line wavelengths (air)
HA = 6562.80

def random_pdf(x, dx, seed_i, n_iter=1000):
    '''
    adapted from https://github.com/astrochun/chun_codes/blob/master/__init__.py
    '''
    if type(x)==np.float64:
        len0 = 1
    else:
        len0 = len(x)
    x_pdf  = np.zeros((len0, n_iter), dtype=np.float64)

    seed0 = seed_i + np.arange(len0)

    for ii in range(len0):
        np.random.seed(seed0[ii])
        temp = np.random.normal(0.0, 1.0, size=n_iter)
        if len0==1:
            x_pdf[ii]  = x + dx*temp
        else:
            x_pdf[ii]  = x[ii] + dx[ii]*temp

    return x_pdf


def get_flux_from_FAST(ID, lambda_arr, byarr=True, fileend='.GALEX'):
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


def get_mainseq_fit_params(sfrs, delta_sfrs, otherdata, num_params=0,
    num_iters=10000, seed=132089):
    '''
    num_params == 2 is for mainseq w/o zspec,
        where 'otherdata' = mass
    num_params == 3 is for mainseq with zspec,
        where 'otherdata' = mass and redshift

    - returns a list of len(num_params) where each element is a parameter
      array of length num_iters
    - the parameters themselves will be [np.mean(params_arr[i]) for i in range(num_params)]
    '''
    from scipy.optimize import curve_fit

    sfrs_pdf = random_pdf(sfrs, delta_sfrs, seed_i=seed, n_iter=num_iters)
    np.random.seed(12376)

    alpha_arr = np.zeros(num_iters)
    gamma_arr = np.zeros(num_iters)

    if num_params == 2:
        def func(data, a, b):
            ''' r'$\log(SFR) = \alpha \log(M) + \beta z + \gamma$' '''
            return a*data + b

        mass = otherdata
        for i in range(num_iters):
            s_arr = sfrs_pdf[:,i]

            params, pcov = curve_fit(func, mass, s_arr)
            alpha_arr[i] = params[0]
            gamma_arr[i] = params[1]
        params_arr = [alpha_arr, gamma_arr]

    elif num_params == 3:
        def func(data, a, b, c):
            ''' r'$\log(SFR) = \alpha \log(M) + \beta z + \gamma$' '''
            return a*data[:,0] + b*data[:,1] + c

        mz_data = otherdata
        beta_arr = np.zeros(num_iters)
        for i in range(num_iters):
            s_arr = sfrs_pdf[:,i]

            params, pcov = curve_fit(func, mz_data, s_arr)
            alpha_arr[i] = params[0]
            beta_arr[i] = params[1]
            gamma_arr[i] = params[2]
        params_arr = [alpha_arr, beta_arr, gamma_arr]

    else:
        raise ValueError('num_params should be 2 or 3')

    return params_arr


def get_tempz(zspec0, filt_arr):
    '''
    gets tempz which returns a redshift array
    sources with spectroscopically confirmed redshifts use that spec_z
    otherwise, estimated redshifts based on the center of the filter are used
    '''
    HA = 6562.80
    centr_filts = {'NB7':((7045.0/HA - 1) + (7126.0/HA - 1))/2.0,  
        'NB704':7045.0/HA - 1, 'NB711':7126.0/HA - 1, 
        'NB816':8152.0/HA - 1, 'NB921':9193.0/HA - 1, 'NB973':9749.0/HA - 1}

    tempz = np.zeros(len(zspec0))
    for ii, zspec in enumerate(zspec0):
        if (zspec > 0 and zspec < 9):
            tempz[ii] = zspec
        elif (zspec <= 0 or zspec > 9):
            tempz[ii] = centr_filts[filt_arr[ii]]
        else:
            raise ValueError('something went wrong with zspecs?')

    return tempz


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


def compute_onesig_pdf(arr0, x_val):
    '''
    adapted from https://github.com/astrochun/chun_codes/blob/master/__init__.py

    whereas np.std assumes that the distribution is normal, this assumes nothing
    about the distribution.

    requires arr0 to be of shape (len0, N_random), where len0 is the number of
    parameters you want to find the onesig of

    x_val must be of shape (len0,)
    '''
    len0 = arr0.shape[0] # arr0.shape[1] # Mod on 29/06/2016

    err   = np.zeros((len0,2)) # np.zeros((2,len0)) # Mod on 29/06/2016
    xpeak = np.zeros(len0)

    conf = 0.68269 # 1-sigma

    for ii in range(len0):
        test = arr0[ii] # arr0[:,ii] # Mod on 29/06/2016
        good = np.where(np.isfinite(test) == True)[0]
        if len(good) > 0:
            v_low  = np.percentile(test[good],15.8655)
            v_high = np.percentile(test[good],84.1345)

            xpeak[ii] = np.percentile(test[good],50.0)
            if len0==1:
                t_ref = x_val
            else:
                t_ref = x_val[ii]

        err[ii,0]  = t_ref - v_low
        err[ii,1]  = v_high - t_ref

    return err, xpeak


def composite_errors(x, dx, seed_i, label=''):
    '''
    '''
    # emission line wavelengths (air)
    HG = 4340.46
    HB = 4861.32
    HA = 6562.80

    k_hg = cardelli(HG * u.Angstrom)
    k_hb = cardelli(HB * u.Angstrom)
    k_ha = cardelli(HA * u.Angstrom)

    if '/HB' in label or label=='NII_BOTH/HA':
        hn_flux = np.array(x[0])
        hn_rms = np.array(dx[0])
        hb_flux = np.array(x[1])
        hb_rms = np.array(dx[1])

        good_iis = np.where((hn_flux > 0) & (hb_flux > 0))[0]
        onesig_errs = np.ones((len(hn_flux), 2))*np.nan
        if len(good_iis) > 0:
            hn_pdf = random_pdf(hn_flux[good_iis], hn_rms[good_iis], seed_i)
            hb_pdf = random_pdf(hb_flux[good_iis], hb_rms[good_iis], seed_i)
            x_pdf = hn_pdf/hb_pdf

            if label=='HA/HB' or label=='HG/HB':
                if label=='HA/HB':
                    ebv_pdf = np.log10((x_pdf)/2.86)/(-0.4*(k_ha-k_hb))
                    ebv_guess = np.log10((hn_flux[good_iis]/hb_flux[good_iis])/2.86)/(-0.4*(k_ha-k_hb))
                else: #label=='HG/HB'
                    ebv_pdf = np.log10((x_pdf)/0.468)/(-0.4*(k_hg-k_hb))
                    ebv_guess = np.log10((hn_flux[good_iis]/hb_flux[good_iis])/0.468)/(-0.4*(k_hg-k_hb))

                err, xpeak = compute_onesig_pdf(ebv_pdf, ebv_guess)
            else: # label == 'Hn/HB_flux_rat_errs'
                err, xpeak = compute_onesig_pdf(x_pdf, hn_flux[good_iis]/hb_flux[good_iis])

            onesig_errs[good_iis] = err

    else:
        x_pdf = random_pdf(x, dx, seed_i)
        onesig_errs, xpeak = compute_onesig_pdf(x_pdf, x)

    return onesig_errs


def get_FUV_corrs(corr_tbl):
    '''
    '''
    from plot_mainseq_UV_Ha_comparison import get_UV_SFR
    from scipy.optimize import curve_fit
    def line(x, m, b):
        return m*x+b

    zspec0 = corr_tbl['zspec0'].data
    yes_spectra = np.where((zspec0 >= 0) & (zspec0 < 9))[0]

    # getting FUV_corr_factor
    log_SFR_UV = get_UV_SFR(corr_tbl)
    log_SFR_HA = corr_tbl['met_dep_sfr'].data
    log_SFR_ratio = log_SFR_HA - log_SFR_UV
    coeffs, covar = curve_fit(line, log_SFR_HA[yes_spectra],
        log_SFR_ratio[yes_spectra])
    m, b = coeffs[0], coeffs[1]
    FUV_corr_factor = -(m*log_SFR_HA + b)

    return FUV_corr_factor

