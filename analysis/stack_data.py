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
def stack(ndarr_in, zspec_in, index, x0, xmin, xmax, ff=''):
    ndarr = ndarr_in[index]
    zspec = zspec_in[index]

    ndarr[np.where(ndarr.mask==True)] = np.nan
    
    x_rest   = np.arange(xmin, xmax, 0.1)
    new_grid = np.ndarray(shape=(len(ndarr), len(x_rest)))
    
    masked_spectra_len = 0
    num_maskednb921 = 0
    masked_index = []
    minz = min(x for x in zspec if x > 0)
    maxz = max(x for x in zspec if x < 9)
    print '### zspec:', minz, maxz
    # deshifting to rest-frame wavelength
    for (row_num, ii) in zip(range(len(ndarr)), index):
        # normalizing
        spec_test = ndarr[row_num]

        # counts the completely masked elements of the array
        if len(np.where(np.isnan(spec_test))[0]) == len(ndarr[row_num]):
            masked_spectra_len += 1

        # interpolating a function for rest-frame wavelength and normalized y
        x_test = x0/(1.0+zspec[row_num])
        f = interp1d(x_test, spec_test, bounds_error=False, fill_value=np.nan)

        # counts the completely masked elements of the nb921 spectrum
        if ff == 'NB921':
            nb921ii = np.where(x_test > 6500)[0]
            if len(np.where(np.isnan(spec_test[nb921ii]))[0]) == len(nb921ii):
                num_maskednb921 += 1

        # finding the new rest-frame wavelength values from the interpolation
        # and putting them into the 'new grid'
        spec_interp = f(x_rest)
        new_grid[row_num] = spec_interp
    #endfor

    # taking the average, column by column to 'stack'
    plot_grid_avg = np.nanmean(new_grid, axis=0)

    return x_rest, plot_grid_avg, [len(index)-masked_spectra_len, num_maskednb921], [index, masked_index], minz, maxz
#enddef
