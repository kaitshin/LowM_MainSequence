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
def stack_data(ndarr, zspec, index, x0, xmin, xmax, ff='', instr=''):
    '''
    TODO(document)
    '''
    plot_grid = ndarr[index]
    plot_zspec = zspec[index]
    
    # this is stacking data in filter
    if ff!='':
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

        if instr=='MMT':
            # to help mask MMT NB921 Halpha sources
            # return stack(ndarr, zspec, index[good_z], x0, xmin, xmax, ff=ff)
            x_rest, plot_grid_avg, [index, masked_index], minz, maxz, new_grid = stack(ndarr, 
                zspec, index[good_z], x0, xmin, xmax, ff=ff)

            # looks for # sources stacked @ nearest emission line by finding nearest idx 
            ma = np.isnan(new_grid)
            new_grid2 = np.ma.array(new_grid, mask=ma)
            x_rest = np.arange(xmin, xmax, 0.1)
            idx0 = (np.abs(x_rest-4341.0)).argmin()
            idx1 = (np.abs(x_rest-4861.0)).argmin()
            idx2 = (np.abs(x_rest-6563.0)).argmin()
            good_hg_num = np.ma.count(new_grid2, axis=0)[idx0]
            good_hb_num = np.ma.count(new_grid2, axis=0)[idx1]
            good_ha_num = np.ma.count(new_grid2, axis=0)[idx2]

            return x_rest, plot_grid_avg, [good_hg_num, good_hb_num, good_ha_num], [index, masked_index], minz, maxz
        else:
            x_rest, plot_grid_avg, [index, masked_index], minz, maxz, new_grid = stack(ndarr, 
                zspec, index[good_z], x0, xmin, xmax)

            # looks for # sources stacked @ nearest emission line by finding nearest idx 
            ma = np.isnan(new_grid)
            new_grid2 = np.ma.array(new_grid, mask=ma)
            x_rest = np.arange(xmin, xmax, 0.1)
            idx0 = (np.abs(x_rest-4861.0)).argmin()
            idx1 = (np.abs(x_rest-6563.0)).argmin()
            good_hb_num = np.ma.count(new_grid2, axis=0)[idx0]
            good_ha_num = np.ma.count(new_grid2, axis=0)[idx1]

            return x_rest, plot_grid_avg, [good_hb_num, good_ha_num], [index, masked_index], minz, maxz
        #endif
    # this is stacking data in stlrmass
    else: # instr == 'Keck'
        x_rest, plot_grid_avg, [index, masked_index], minz, maxz, new_grid = stack(ndarr, 
            zspec, index, x0, xmin, xmax)

        # looks for # sources stacked @ nearest emission line by finding nearest idx 
        ma = np.isnan(new_grid)
        new_grid2 = np.ma.array(new_grid, mask=ma)
        x_rest = np.arange(xmin, xmax, 0.1)
        idx0 = (np.abs(x_rest-4861.0)).argmin()
        idx1 = (np.abs(x_rest-6563.0)).argmin()
        good_hb_num = np.ma.count(new_grid2, axis=0)[idx0]
        good_ha_num = np.ma.count(new_grid2, axis=0)[idx1]

        if instr=='MMT':
            idx2 = (np.abs(x_rest-4341.0)).argmin()
            good_hg_num = np.ma.count(new_grid2, axis=0)[idx2]
            return x_rest, plot_grid_avg, [good_hg_num, good_hb_num, good_ha_num], [index, masked_index], minz, maxz
        else:
            return x_rest, plot_grid_avg, [good_hb_num, good_ha_num], [index, masked_index], minz, maxz
#enddef
