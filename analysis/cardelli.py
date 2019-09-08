import numpy as np

from astropy import units as u

def uv_func_mw(x, R):
  
    Fa = -0.04473*(x-5.9)**2 - 0.009779*(x-5.9)**3
    Fb = 0.2130*(x-5.9)**2 - 0.1207*(x-5.9)**3
    
    mark1 = np.where(x < 5.9)[0]
    if len(mark1) > 0:
        Fa[mark1] = 0.0
        Fb[mark1] = 0.0
        
    a = 1.752 - 0.316 * x - 0.104/((x-4.67)**2 + 0.341) + Fa
    b = -3.090 + 1.825 * x + 1.206/((x-4.62)**2+0.263) + Fb

    return a + b/R
#enddef

def fuv_func_mw(x, R):

    a = -1.073 - 0.628 * (x-8) + 0.137*(x-8)**2 - 0.070*(x-8)**3
    b = 13.670 + 4.257*(x-8) - 0.420*(x-8)**2 + 0.374*(x-8)**3
    return a + b/R
#enddef

def ir_func_mw(x, R):

    a = 0.574 * x**1.61
    b = -0.527 * x**1.61
    return a + b/R
#enddef

def opt_func_mw(x, R):
    y = x - 1.82

    a = 1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 + 0.72085 * y**4 + \
        0.01979 * y**5 - 0.77530 * y**6 + 0.32999 * y**7
    b = 1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 - 5.38434 * y**4 - \
        0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7
    return a + b/R
#enddef

def cardelli(lambda0, R=3.1): #, extrapolate=False):
    '''
    NAME:
       cardelli

    PURPOSE:
       Determine extinction coefficient at lambda0 following
       Cardelli, Clayton & Mathis (1989), ApJ, 345, 245

    CALLING SEQUENCE:
       from cardelli import *
       k = cardelli(lambda0, R=3.1)

    INPUTS:
       lambda0 -- The rest-frame wavelength. Can be a single value or
                  an array. Must specify units

    OPTIONAL KEYWORD INPUT:
       R = R_V \equiv A(V)/E(B-V). Default: 3.1

    OUTPUTS:
       k : Is A(lambda0)/A_V
           A_V = E(B-V) * R_V
           So k' (Calzetti's definition) is k' = A(lambda0)/A_V * R_V


    OPTIONAL OUTPUT KEYWORD:
       None.

    PROCEDURES USED:
       np.where
       astropy.units
       uv_func_mw()
       fuv_func_mw()
       ir_func_mw()
       opt_func_mw()

    NOTES:

    REVISON HISTORY:
       Created by Chun Ly, 28 June 2016
    '''
    # Specify units of lambda0 so that code can convert
    # Default is R=3.1

    t_lam = lambda0.to(u.nm).value

    ## Handles individual values, x
    if type(t_lam) == 'list':
        t_lam = np.array(t_lam)
    else:
        if isinstance(t_lam, (np.ndarray, np.generic)) == False:
            t_lam = np.array([t_lam])

    x = 1.0/(t_lam/1000.0) #in micron^-1

    k = np.zeros(np.size(t_lam), dtype=np.float64)

    mark = np.where((x <= 1.10) & (x >= 0.30))[0]
    if len(mark) > 0: k[mark] = ir_func_mw(x[mark], R)

    mark = np.where((x <= 3.30) & (x > 1.10))[0]
    if len(mark) > 0: k[mark] = opt_func_mw(x[mark], R)

    mark = np.where((x <= 8.00) & (x > 3.3))[0]
    if len(mark) > 0: k[mark] = uv_func_mw(x[mark], R)

    mark = np.where((x <= 10.00) & (x > 8.0))[0]
    if len(mark) > 0: k[mark] = fuv_func_mw(x[mark], R)

    k = k * R
    if np.size(x) == 1: k = k[0]
    return k 
#enddef
