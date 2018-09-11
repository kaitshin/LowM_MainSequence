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
from stack_data import stack
def stack_data(ndarr, zspec, index, x0, xmin, xmax, dlambda, ff='', instr=''):
    '''
    TODO(document)
    '''
    plot_grid = ndarr[index]
    plot_zspec = zspec[index]
    
    # this is stacking data in filter
    if ff!='':
        good_z = []
        if ff=='NB704' or ff=='NB711':
            good_z = np.where((plot_zspec >= 0.05) & (plot_zspec <= 0.1))[0]
        elif ff=='NB816':
            good_z = np.where((plot_zspec >= 0.21) & (plot_zspec <= 0.26))[0]
        elif ff=='NB921':
            good_z = np.where((plot_zspec >= 0.385) & (plot_zspec <= 0.429))[0]
        elif ff=='NB973':
            good_z = np.where((plot_zspec >= 0.45) & (plot_zspec <= 0.52))[0]
        #endif

        if len(good_z) < 2:
            raise AttributeError('There are less than 2 good sources')

        if instr=='MMT':
            # to help mask MMT NB921 Halpha sources
            # return stack(ndarr, zspec, index[good_z], x0, xmin, xmax, ff=ff)
            # print 'sdf_stack_data goodz length:', len(good_z), good_z
            x_rest, plot_grid_avg, index, avgz, minz, maxz, new_grid = stack(ndarr, 
                zspec, index[good_z], x0, xmin, xmax, dlambda)

            # looks for # sources stacked @ nearest emission line by finding nearest idx 
            ma = np.isnan(new_grid)
            new_grid2 = np.ma.array(new_grid, mask=ma)
            x_rest = np.arange(xmin, xmax, dlambda)
            idx0 = (np.abs(x_rest-4341.0)).argmin()
            idx1 = (np.abs(x_rest-4861.0)).argmin()
            idx2 = (np.abs(x_rest-6563.0)).argmin()
            good_hg_num = np.ma.count(new_grid2, axis=0)[idx0]
            good_hb_num = np.ma.count(new_grid2, axis=0)[idx1]
            good_ha_num = np.ma.count(new_grid2, axis=0)[idx2]

            return x_rest, plot_grid_avg, [good_hg_num, good_hb_num, good_ha_num], index, avgz, minz, maxz
        else: # instr == 'Keck'
            x_rest, plot_grid_avg, index, avgz, minz, maxz, new_grid = stack(ndarr, 
                zspec, index[good_z], x0, xmin, xmax, dlambda)

            # looks for # sources stacked @ nearest emission line by finding nearest idx 
            ma = np.isnan(new_grid)
            new_grid2 = np.ma.array(new_grid, mask=ma)
            idx0 = (np.abs(x_rest-4861.0)).argmin()
            idx1 = (np.abs(x_rest-6563.0)).argmin()
            good_hb_num = np.ma.count(new_grid2, axis=0)[idx0]
            good_ha_num = np.ma.count(new_grid2, axis=0)[idx1]

            return x_rest, plot_grid_avg, [good_hb_num, good_ha_num], index, avgz, minz, maxz
        #endif
    # this is stacking data in stlrmass
    else: # instr == 'Keck'
        good_z = np.where((plot_zspec >= 0) & (plot_zspec <= 9))[0]
        x_rest, plot_grid_avg, index, avgz, minz, maxz, new_grid = stack(ndarr, 
            zspec, index[good_z], x0, xmin, xmax, dlambda)

        # looks for # sources stacked @ nearest emission line by finding nearest idx 
        ma = np.isnan(new_grid)
        new_grid2 = np.ma.array(new_grid, mask=ma)
        x_rest = np.arange(xmin, xmax, dlambda)
        idx0 = (np.abs(x_rest-4861.0)).argmin()
        idx1 = (np.abs(x_rest-6563.0)).argmin()
        good_hb_num = np.ma.count(new_grid2, axis=0)[idx0]
        good_ha_num = np.ma.count(new_grid2, axis=0)[idx1]

        if instr=='MMT':
            idx2 = (np.abs(x_rest-4341.0)).argmin()
            good_hg_num = np.ma.count(new_grid2, axis=0)[idx2]
            return x_rest, plot_grid_avg, [good_hg_num, good_hb_num, good_ha_num], index, avgz, minz, maxz
        else:
            return x_rest, plot_grid_avg, [good_hb_num, good_ha_num], index, avgz, minz, maxz
#enddef
