"""
NAME:
    MACT_utils.py

PURPOSE:
    this code has functions that are used across multiple codes
"""
from __future__ import print_function

import numpy as np

import config


def get_func0_eqn0(fittype):
    '''
    returns functions and equation strings based on the fit type (either
    first or second order)

    these functions are the 'model's that are subtracted from the data
    to calculate the residuals
    '''
    if fittype=='first_order':
        eqn0 = r'$log(SFR) = \alpha log(M) + \beta z + \gamma$'
        def func0(data, a, b, c):
            return a*data[:,0] + b*data[:,1] + c

    elif fittype=='second_order':
        eqn0 = r'$log(SFR) = \alpha ^l log(M)^2 + \alpha log(M) + \beta z + \gamma$'
        def func0(data, aprime, a, b, c):
            return aprime*data[:,0]**2 + a*data[:,0] + b*data[:,1] + c

    else:
        raise ValueError('invalid fit type')

    return func0, eqn0


def approximated_zspec0(zspec0, filts):
    '''
    modifying zspec0 such that all invalid zspec vals are replaced by
    approximate values from the filters

    this is for obtaining the best-fit line params for 
    make_redshift_graph() and make_ssfr_graph()
    '''
    zspec00 = np.copy(zspec0)

    badz_iis = np.array([x for x in range(len(zspec00)) if zspec00[x] < 0
        or zspec00[x] > 9])
    filt_lambda_list = {'NB704':7045.0, 'NB711':7126.0, 'NB816':8152.0,
        'NB921':9193.0, 'NB973':9749.0}

    for ff in filt_lambda_list.keys():
        badf_match = np.where(filts[badz_iis] == ff)[0]
        zspec00[badz_iis[badf_match]] = (filt_lambda_list[ff]/config.HA_VAL) - 1

    return zspec00


def get_filt_index(spectra, ff, filts):
    '''
    returns the indexes at which the sources are in the filter
    (and handles the special case of 'NB704+NB711')

    compatible with both spectra == no_spectra and spectra == yes_spectra

    called in make_all_graph() and make_redshift_graph()
    '''
    if 'NB7' in ff:
        filt_index = np.array([x for x in range(len(spectra)) if
            ff[:3] in filts[spectra][x]])
    else:
        filt_index = np.array([x for x in range(len(spectra)) if
            ff==filts[spectra][x]])

    return filt_index


def exclude_bad_sources(ha_ii, NAME0):
    '''
    excludes sources from the mainseq sample depending on their unreliable properties
    '''
    # getting rid of special cases (no_spectra):
    bad_highz_gal = np.where(NAME0=='Ha-NB816_174829_Ha-NB921_187439_Lya-IA598_163379')[0]

    bad_HbNB704_SIINB973_gals = np.array([x for x in range(len(ha_ii)) if 
            (NAME0[x]=='Ha-NB704_028405_OII-NB973_056979' or 
             NAME0[x]=='Ha-NB704_090945_OII-NB973_116533')])

    # getting rid of a source w/o flux (yes_spectra):
    no_flux_gal = np.where(NAME0=='Ha-NB921_069950')[0]

    # getting rid of a source w/ atypical SFR behavior we don't understand
    weird_SFR_gal = np.where(NAME0=='OIII-NB704_063543_Ha-NB816_086540')[0]

    # getting rid of AGN candidate galaxies
    possibly_AGNs = np.array([x for x in range(len(ha_ii)) if 
            (NAME0[x]=='Ha-NB921_063859' or NAME0[x]=='Ha-NB973_054540' or
             NAME0[x]=='Ha-NB973_064347' or NAME0[x]=='Ha-NB973_084633')])

    bad_sources = np.concatenate([bad_highz_gal, bad_HbNB704_SIINB973_gals, no_flux_gal, weird_SFR_gal, possibly_AGNs])
    print("bad_sources: {}".format(len(bad_sources)))

    ha_ii = np.delete(ha_ii, bad_sources)
    NAME0 = np.delete(NAME0, bad_sources)

    return ha_ii, NAME0


def exclude_AGN(index, NAME0):
    '''
    excludes AGNs based on manual inspection whre NII6583/Ha > 0.54 (Kennicutt+08)
    '''
    # affected by skyline. don't use individ NII measurements
    index = np.delete(index, np.where(index==np.where(NAME0=='Ha-NB921_063859')[0])[0])

    # affected by skyline. don't use individ NII measurements
    index = np.delete(index, np.where(index==np.where(NAME0=='Ha-NB973_054540')[0])[0])

    # lines too broad b/c of poor sky subtraction (bad fit).
    # redo IRAF fit?  otherwise don't use individ. NII measurements.
    index = np.delete(index, np.where(index==np.where(NAME0=='Ha-NB973_064347')[0])[0])

    # definitely an AGN (seifert 2 galaxy) --> narrow hb line. def exclude!
    index = np.delete(index, np.where(index==np.where(NAME0=='Ha-NB973_084633')[0])[0])

    return index


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
            tempfile = asc.read(config.FULL_PATH+
                'FAST/outputs/BEST_FITS/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+
                fileend+'_'+str(ID[ii])+'.fit', guess=False,Reader=asc.NoHeader)
            wavelength = np.array(tempfile['col1'])
            flux = np.array(tempfile['col2'])
            f = interpolate.interp1d(wavelength, flux)
            newflux[ii] = f(lambda_arr[ii])

    else:
        tempfile = asc.read(config.FULL_PATH+
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
    tempz = np.zeros(len(zspec0))
    for ii, zspec in enumerate(zspec0):
        if (zspec > 0 and zspec < 9):
            tempz[ii] = zspec
        elif (zspec <= 0 or zspec > 9):
            tempz[ii] = config.centr_filts[filt_arr[ii]]
        else:
            raise ValueError('something went wrong with zspecs?')

    return tempz


def get_z_arr():
    '''
    defining an approximate redshift array for plot visualization
    '''
    z_arr0 = np.array([7045.0, 7126.0, 8152.0, 9193.0, 9749.0])/config.HA_VAL - 1
    z_arr0 = np.around(z_arr0, 2)
    # z_arr  = np.array(z_arr0, dtype='|S9')
    z_arr = z_arr0.astype(str)
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
            print('## Using linear relation!')
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
                    ebv_pdf = np.log10((x_pdf)/2.86)/(-0.4*(config.k_ha-config.k_hb))
                    ebv_guess = np.log10((hn_flux[good_iis]/hb_flux[good_iis])/2.86)/(-0.4*(config.k_ha-config.k_hb))
                else: #label=='HG/HB'
                    ebv_pdf = np.log10((x_pdf)/0.468)/(-0.4*(config.k_hg-config.k_hb))
                    ebv_guess = np.log10((hn_flux[good_iis]/hb_flux[good_iis])/0.468)/(-0.4*(config.k_hg-config.k_hb))

                err, xpeak = compute_onesig_pdf(ebv_pdf, ebv_guess)
            else: # label == 'Hn/HB_flux_rat_errs'
                err, xpeak = compute_onesig_pdf(x_pdf, hn_flux[good_iis]/hb_flux[good_iis])

            onesig_errs[good_iis] = err

    else:
        x_pdf = random_pdf(x, dx, seed_i)
        onesig_errs, xpeak = compute_onesig_pdf(x_pdf, x)

    return onesig_errs


def get_LUV(corrID, corrzspec0, centr_filts, filt_index_haii, ff):
    '''
    get FUV luminosity (at 1500 AA) by converting the flux (get_flux_from_FAST())

    sources without spectroscopic z are estimated by the center of the
    filter profile

    returns the log of the luminosity log_L_nu (nu=1500AA)
    '''
    import astropy.units as u
    from astropy import constants
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0 = 70 * u.km / u.s / u.Mpc, Om0=0.3)

    ID = corrID[filt_index_haii]
    zspec = corrzspec0[filt_index_haii]

    goodz = np.where((zspec >= 0) & (zspec < 9))[0]
    badz  = np.where((zspec <= 0) | (zspec > 9))[0]

    tempz = np.zeros(len(filt_index_haii))
    tempz[goodz] = zspec[goodz]
    tempz[badz] = centr_filts[ff]

    lambda_arr = (1+tempz)*1500

    f_lambda = get_flux_from_FAST(ID, lambda_arr)
    f_nu = f_lambda*(1E-19*(lambda_arr**2*1E-10)/(constants.c.value))
    log_L_nu = np.log10(f_nu*4*np.pi) + \
        2*np.log10(cosmo.luminosity_distance(tempz).to(u.cm).value)

    return log_L_nu


def get_UV_SFR(corr_tbl):
    '''
    deriving log UV SFR from the 1500AA measurements
    '''
    # defining useful things
    corrID = corr_tbl['ID'].data
    corrfilts = corr_tbl['filt'].data
    corrzspec0 = corr_tbl['zspec0'].data

    npz_files = np.load(config.FULL_PATH+'Plots/sfr_metallicity_plot_fit.npz')

    Lnu_fit_ch = npz_files['Lnu_fit_ch']

    P2, P1, P0 = -1*Lnu_fit_ch
    def log_SFR_from_L(zz):
        '''zz is metallicity: log(Z/Z_sol)'''
        return P0 + P1*zz + P2*zz**2

    # niiha_oh_determine estimates log(O/H)+12
    NII6583_Ha = corr_tbl['NII_Ha_ratio'].data * 2.96/(1+2.96)
    logOH = niiha_oh_determine(np.log10(NII6583_Ha), 'PP04_N2') - 12
    y = logOH + 3.31

    # this is luminosity
    log_SFR_LUV = log_SFR_from_L(y)

    # this is luminosity correction
    LUV = np.zeros(len(corr_tbl))
    for ff in ['NB7','NB816','NB921','NB973']:
        filt_index_haii = np.array([x for x in range(len(corr_tbl)) if ff in
            corrfilts[x]])

        lnu = get_LUV(corrID, corrzspec0, config.centr_filts, filt_index_haii, ff)
        LUV[filt_index_haii] = lnu

    log_SFR_UV = log_SFR_LUV + LUV

    return log_SFR_UV


def get_FUV_corrs(corr_tbl, ret_coeffs_const=False, ret_coeffs=False, old=False):
    '''
    '''
    from scipy.optimize import curve_fit
    def line(x, m, b):
        return m*x+b

    zspec0 = corr_tbl['zspec0'].data
    yesz_ii = np.where((zspec0 >= 0.) & (zspec0 < 9.))[0]

    # getting dustcorr UV SFR
    log_SFR_UV = get_UV_SFR(corr_tbl)
    EBV_HA = corr_tbl['EBV'].data
    UV_lambda  = 0.15 # units of micron
    K_UV       = (2.659*(-2.156 + 1.509/UV_lambda - 0.198/UV_lambda**2
                        + 0.011/UV_lambda**3)+ 4.05)
    A_UV = K_UV*0.44*EBV_HA
    log_SFR_UV_dustcorr = log_SFR_UV + 0.4*A_UV

    # getting dustcorr HA SFR
    log_SFR_HA = corr_tbl['met_dep_sfr'].data
    dust_corr_factor = corr_tbl['dust_corr_factor'].data
    filt_corr_factor = corr_tbl['filt_corr_factor'].data
    nii_ha_corr_factor = corr_tbl['nii_ha_corr_factor'].data
    log_SFR_HA_dustcorr = log_SFR_HA+filt_corr_factor+nii_ha_corr_factor+dust_corr_factor

    # getting ratio
    log_SFR_ratio_dustcorr = log_SFR_HA_dustcorr - log_SFR_UV_dustcorr

    # getting turnover mass
    turnover = -1.5
    salpeter_to_chabrier = np.log10(7.9/4.55)
    turnover -= salpeter_to_chabrier

    # fitting piecewise fn
    low_SFRHA_yesz_ii = np.where(log_SFR_HA_dustcorr[yesz_ii] <= turnover)[0]
    high_SFRHA_yesz_ii = np.where(log_SFR_HA_dustcorr[yesz_ii] >= turnover)[0]
    const = np.mean(log_SFR_ratio_dustcorr[yesz_ii][high_SFRHA_yesz_ii])
    def line(x, m):
        return m*(x-turnover)+const
    coeffs, covar = curve_fit(line, log_SFR_HA_dustcorr[yesz_ii][low_SFRHA_yesz_ii],
        log_SFR_ratio_dustcorr[yesz_ii][low_SFRHA_yesz_ii])
    m = coeffs[0]
    b = const-m*turnover

    low_SFRHA_ii = np.where(log_SFR_HA_dustcorr <= turnover)[0]
    high_SFRHA_ii = np.where(log_SFR_HA_dustcorr >= turnover)[0]    
    FUV_corr_factor = np.zeros(len(log_SFR_HA_dustcorr))
    FUV_corr_factor[low_SFRHA_ii] = line(log_SFR_HA_dustcorr[low_SFRHA_ii], *coeffs)
    FUV_corr_factor[high_SFRHA_ii] = const
    assert 0 not in FUV_corr_factor
    FUV_corr_factor = -FUV_corr_factor

    if ret_coeffs:
        return m, b
    elif ret_coeffs_const:
        return m, b, const
    elif old:
        # getting FUV_corr_factor
        def line(x, m, b):
            return m*x+b
        log_SFR_ratio = log_SFR_HA - log_SFR_UV
        coeffs, covar = curve_fit(line, log_SFR_HA[yesz_ii],
            log_SFR_ratio[yesz_ii])
        m, b = coeffs[0], coeffs[1]
        FUV_corr_factor = -(m*log_SFR_HA + b)
        return FUV_corr_factor
    else:
        return FUV_corr_factor


def get_good_newha_ii(newhadata):
    '''
    returns indices of the newha data that have valid mass, sfrs, lha, and
    are not agns
    '''
    tempm = newhadata['LOGM']
    tempsfr = newhadata['LOGSFR_HA']
    templha = newhadata['L_HA']
    tempagn = newhadata['AGN']
    good_newha_ii = np.where((tempm > 0) & (tempsfr != 0) & (templha != 0)
        & (tempagn != 1))[0]

    return good_newha_ii


def get_newha_logsfrha(newhadata, newha_sfr_type):
    '''
    returns sfrs from the newha dataset depending on newha_sfr_type
    
    if the type is 'orig_sfr', then the sfr is returned directly from the tbl
    
    if the type is 'met_dep_sfr', then a metallicity-dependent sfr is derived
    from the luminosity and nii/ha ratios given in the table. this is to be
    consistent with the way the MACT dataset was analyzed
    '''
    if newha_sfr_type == 'orig_sfr':
        newha_logsfrha = newhadata['LOGSFR_HA']
    elif newha_sfr_type == 'met_dep_sfr':
        from mainseq_corrections import niiha_oh_determine

        l_ha = np.log10(newhadata['L_HA']) + 41 # ha luminosity
        nii_ha_best = newhadata['NII_HALPHA_BEST']
        nii6583_ha = nii_ha_best * 2.96/(1+2.96)

        # since this code estimates log(O/H)+12
        logOH = niiha_oh_determine(np.log10(nii6583_ha), 'PP04_N2') - 12
        y = logOH + 3.31 
        log_SFR_LHa = -41.34 + 0.39*y + 0.127*y**2 # metallicity-dependent

        newha_logsfrha = log_SFR_LHa + l_ha
    else:
        raise ValueError('invalid newha_sfr_type')

    return newha_logsfrha


def combine_mact_newha(corr_tbl, FUV_corr=True, errs=False):
    '''
    '''
    import matplotlib as mpl
    # getting relevant corr_tbl data
    sfrs = corr_tbl['met_dep_sfr'].data
    dust_corr_factor = corr_tbl['dust_corr_factor'].data
    filt_corr_factor = corr_tbl['filt_corr_factor'].data
    nii_ha_corr_factor = corr_tbl['nii_ha_corr_factor'].data
    corr_sfrs = sfrs+filt_corr_factor+nii_ha_corr_factor+dust_corr_factor
    if FUV_corr:
        FUV_corr_factor = get_FUV_corrs(corr_tbl)
        corr_sfrs+=FUV_corr_factor
    stlr_mass = corr_tbl['stlr_mass'].data
    filts = corr_tbl['filt'].data
    zspec0 = np.array(corr_tbl['zspec0'])
    zspec00 = approximated_zspec0(zspec0, filts)
    z_arr = get_z_arr()
    

    # reading in newha data
    from astropy.io import fits as pyfits
    newha = pyfits.open(config.FULL_PATH+'NewHa/NewHa.fits')
    newhadata_tmp = newha[1].data

    good_newha_ii = get_good_newha_ii(newhadata_tmp)
    newhadata = newhadata_tmp[good_newha_ii]

    newha_logm = newhadata['LOGM']
    newha_zspec = newhadata['Z_SPEC']
    newha_mzdata = np.vstack([newha_logm, newha_zspec]).T
    newha_logsfrha = get_newha_logsfrha(newhadata, newha_sfr_type='met_dep_sfr')
    if FUV_corr:
        m, b, const = get_FUV_corrs(corr_tbl, ret_coeffs_const=True)
        newha_logsfrha = newha_logsfrha-const
    if errs:
        delta_sfrs = corr_tbl['meas_errs'].data
        newha_logsfrha_uperr = newhadata['LOGSFR_HA_UPERR']
        newha_logsfrha_lowerr = newhadata['LOGSFR_HA_LOWERR']
        newha_logsfr_err = np.sqrt(newha_logsfrha_uperr**2/2 + newha_logsfrha_lowerr**2/2)
        errs_with_newha = np.concatenate((delta_sfrs, newha_logsfr_err))


    # combining datasets
    sfrs_with_newha  = np.concatenate((corr_sfrs, newha_logsfrha))
    mass_with_newha  = np.concatenate((stlr_mass, newha_logm))
    zspec_with_newha = np.concatenate((zspec0, newha_zspec))
    zspec_with_newha00 = np.concatenate((zspec00, newha_zspec))
    filts_with_newha = np.concatenate((filts,
        np.array(['NEWHA']*len(newha_logsfrha))))
    mz_data_with_newha = np.vstack([mass_with_newha, zspec_with_newha00]).T

    no_spectra  = np.where((zspec_with_newha <= 0) | (zspec_with_newha > 9))[0]
    yes_spectra = np.where((zspec_with_newha >= 0) & (zspec_with_newha < 9))[0]


    # misc things
    nh_z_arr = np.append(z_arr, '%.2f'%np.mean(newha_zspec))
    nh_cwheel = [np.array(mpl.rcParams['axes.prop_cycle'])[x]['color']
        for x in range(5)]

    newha.close()
    if errs:
        return (sfrs_with_newha, mass_with_newha, zspec_with_newha,
            zspec_with_newha00, filts_with_newha, mz_data_with_newha,
            no_spectra, yes_spectra, nh_z_arr, nh_cwheel, errs_with_newha)
    else:
        return (sfrs_with_newha, mass_with_newha, zspec_with_newha,
            zspec_with_newha00, filts_with_newha, mz_data_with_newha,
            no_spectra, yes_spectra, nh_z_arr, nh_cwheel)


def get_filt_arr(NAME0):
    '''
    '''
    import re
    filt_arr = np.array([])
    for name in NAME0:
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


def get_stlrmassbinZ_arr(filt_arr, stlr_mass, tabfiltarr, min_mass_arr, max_mass_arr, instr):
    '''
    '''
    stlrmassZbin = np.array([])
    for ff, m in zip(filt_arr, stlr_mass):
        if instr=='Keck' and ff=='NB816':
            stlrmassZbin = np.append(stlrmassZbin, 'N/A')
        elif instr=='MMT' and 'NB7' in ff:
            assigned_bin = ''
            good_filt_iis = np.array([x for x in range(len(tabfiltarr)) if tabfiltarr[x]=='NB704+NB711'])
            min_mass = np.array(min_mass_arr[good_filt_iis])
            max_mass = np.array(max_mass_arr[good_filt_iis])
            for ii in range(len(good_filt_iis)):
                if m >= min_mass[ii] and m <= max_mass[ii]:
                    assigned_bin += str(ii+1)+'-NB704+NB711'
            stlrmassZbin = np.append(stlrmassZbin, assigned_bin)
        else:
            good_filt_iis = np.array([x for x in range(len(tabfiltarr)) if tabfiltarr[x]==ff])
            min_mass = np.array(min_mass_arr[good_filt_iis])
            max_mass = np.array(max_mass_arr[good_filt_iis])
            for ii in range(len(good_filt_iis)):
                if m >= min_mass[ii] and m <= max_mass[ii]:
                    stlrmassZbin = np.append(stlrmassZbin, str(ii+1)+'-'+ff)
    #endfor

    return stlrmassZbin


def find_nearest_iis(array, value):
    '''
    '''
    idx_closest = (np.abs(array-value)).argmin()
    if array[idx_closest] > value and idx_closest != 0:
        return [idx_closest-1, idx_closest]
    else:
        return [idx_closest, idx_closest+1]


def get_spectral_cvg_MMT(MMT_LMIN0, MMT_LMAX0, zspec0, grid_ndarr_match_ii, x0):
    '''
    '''
    HG = np.array([])
    HB = np.array([])
    HA = np.array([])
    for lmin0, lmax0, row, z in zip(MMT_LMIN0, MMT_LMAX0, grid_ndarr_match_ii, zspec0):
        hg_near_iis = find_nearest_iis(x0, config.HG_VAL*(1+z))
        hb_near_iis = find_nearest_iis(x0, config.HB_VAL*(1+z))
        ha_near_iis = find_nearest_iis(x0, config.HA_VAL*(1+z))

        if lmin0 < 0:
            HG = np.append(HG, 'NO')
            HB = np.append(HB, 'NO')
            HA = np.append(HA, 'NO')
        else:
            if lmin0 <= config.HG_VAL and lmax0 >= config.HG_VAL:
                if np.average(row[hg_near_iis])==0:
                    HG = np.append(HG, 'MASK')
                else:
                    HG = np.append(HG, 'YES')
            else:
                HG = np.append(HG, 'NO')
            
            if lmin0 <= config.HB_VAL and lmax0 >= config.HB_VAL:
                if np.average(row[hb_near_iis])==0:
                    HB = np.append(HB, 'MASK')
                else:
                    HB = np.append(HB, 'YES')
            else:
                HB = np.append(HB, 'NO')
                
            if lmin0 <= config.HA_VAL and lmax0 >= config.HA_VAL:
                if np.average(row[ha_near_iis])==0:
                    HA = np.append(HA, 'MASK')
                else:
                    HA = np.append(HA, 'YES')
            else:
                HA = np.append(HA, 'NO')
    #endfor
    return HG, HB, HA


