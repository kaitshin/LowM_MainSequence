"""
NAME:
    sdf_stack_data.py

PURPOSE:
    Meant to be called as a subroutine, this code takes the input data,
    passes it to stack_data.py, and then returns it. Specific to the SDF data.

INPUTS:
    Reads in (ndarr, zspec, index, x0, xmin, xmax, ff).

OUTPUTS:
    Returns (x_rest, plot_grid_avg)
"""

import numpy as np
from scipy.interpolate import interp1d
from stack_data import stack
def stack_data(ndarr, zspec, index, x0, xmin, xmax, ff='', stlr_mass=-1):
    plot_grid = ndarr[index]
    plot_zspec = zspec[index]
    
    good_z = []
    if ff=='NB704' or ff=='NB711':
        good_z = np.where(plot_zspec < 0.1)[0]
    elif ff=='NB816':
        good_z = np.where(plot_zspec < 0.3)[0]
    elif ff=='NB921':
        good_z = np.where(plot_zspec < 0.4)[0]
    elif ff=='NB973':
        good_z = np.where(plot_zspec < 0.6)[0]
    #endif
    plot_grid = plot_grid[good_z]
    plot_zspec = plot_zspec[good_z]

    return stack(plot_grid, plot_zspec, x0, xmin, xmax)
#enddef
