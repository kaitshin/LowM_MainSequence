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

import numpy as np
from scipy.interpolate import interp1d
def stack(ndarr, zspec, index, x0, xmin, xmax):
    ndarr = ndarr[index]
    zspec = zspec[index]
    
    x_rest   = np.arange(xmin, xmax, 0.1)
    new_grid = np.ndarray(shape=(len(ndarr), len(x_rest)))
    
    print xmin, xmax
    print '## zspec : ', np.min(zspec), np.max(zspec)
    #deshifting to rest-frame wavelength
    for row_num in range(len(ndarr)):
        #normalizing
        spec_test = ndarr[row_num]

        #interpolating a function for rest-frame wavelength and normalized y
        x_test = x0/(1.0+zspec[row_num])
        f = interp1d(x_test, spec_test, bounds_error=False, fill_value=np.nan)

        #finding the new rest-frame wavelength values from the interpolation
        #and putting them into the 'new grid'
        spec_interp = f(x_rest)
        new_grid[row_num] = spec_interp
    #endfor

    #taking the average, column by column
    print np.nansum(new_grid, axis=1)
    plot_grid_avg = np.nanmean(new_grid, axis=0)

    return x_rest, plot_grid_avg
#enddef
