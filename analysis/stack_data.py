"""
NAME:
    stack_data.py

PURPOSE:
    Meant to be called as a subroutine, this code takes the input data,
    'stacks' it, and then returns it. 

INPUTS:
    Reads in (ndarr, zspec, index, x0, xmin, xmax, ff).

OUTPUTS:
    Returns (x_rest, plot_grid_avg)
"""

import numpy as np, numpy.ma as ma
from scipy.interpolate import interp1d
def stack(ndarr_in, zspec_in, index, x0, xmin, xmax, ff='', AP_rows=[]):
    ndarr = ndarr_in[index]
    zspec = zspec_in[index]
    
    x_rest   = np.arange(xmin, xmax, 0.1)
    new_grid = np.ndarray(shape=(len(ndarr), len(x_rest)))
    
    num_maskednb921 = 0
    minz = min(x for x in zspec if x > 0)
    maxz = max(x for x in zspec if x < 9)
    print '### zspec:', minz, maxz
    #deshifting to rest-frame wavelength
    for (row_num, ii) in zip(range(len(ndarr)), index):
        #normalizing
        spec_test = ndarr[row_num]

        #interpolating a function for rest-frame wavelength and normalized y
        x_test = x0/(1.0+zspec[row_num])
        f = interp1d(x_test, spec_test, bounds_error=False, fill_value=np.nan)

        #finding the new rest-frame wavelength values from the interpolation
        #and putting them into the 'new grid'
        spec_interp = f(x_rest)
        
        # masking MMT NB921 Halpha stacking values if they're part of the 'bad' APs
        if (ff and ff=='NB921' and ii in AP_rows):
            mask_ii = np.where(x_rest > 6503)[0]
            spec_interp[mask_ii] = np.nan
            num_maskednb921 += 1
        #endif

        new_grid[row_num] = spec_interp
    #endfor

    #taking the average, column by column
    plot_grid_avg = np.nanmean(new_grid, axis=0)

    return x_rest, plot_grid_avg, [len(index), num_maskednb921], minz, maxz
#enddef
