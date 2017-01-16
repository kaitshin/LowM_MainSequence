"""
NAME:
    general_plotting.py

PURPOSE:
    Provides a module where plotting functions can be called from 
    stack_spectral_data.py for both Hg/Hb/Ha and Hb/Ha (MMT, Keck)
    plots
"""

import numpy as np

def initialize_table_arrays():
    '''
    '''

    return ([], [], [], [], [], [], [], [], [], [], [])

def get_name_index_matches(NAME0, inst_str0, inst_dict, *args, **kwargs):
    '''
    Returns the indexes from which the kwargs name is in the ordered NAME0
    array and the kwargs instr is in the ordered inst_dict dict.
    '''
    namematch = kwargs['namematch']
    instr     = kwargs['instr']
    index = np.array([x for x in range(len(NAME0)) if namematch in NAME0[x] and
                      inst_str0[x] in inst_dict[instr]])
    return index
#enddef

def get_index_list(NAME0, inst_str0, inst_dict, instr):
    '''
    Helper function. Returns an index_list for either MMT or Keck 
    based on instr keyword
    '''
    if instr=='MMT':
        index_0 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB704',instr=instr)
        index_1 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB711',instr=instr)
        index_2 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB816',instr=instr)
        index_3 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB921',instr=instr)
        index_4 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB973',instr=instr)
        return [index_0]*3+[index_1]*3+[index_2]*3+[index_3]*3+[index_4]*3
    elif instr=='Keck':
        index_0 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB816',instr=instr)
        index_1 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB921',instr=instr)
        index_2 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB973',instr=instr)
        return [index_0]*2+[index_1]*2+[index_2]*2
    else:
        return 0
#enddef

def final_plot_setup(f, title):
    '''
    For the passed in figure, adjusts tick parameter sizes,
    adds minorticks and a suptitle, and adjusts spacing

    Works for both Hg/Hb/Ha and Hb/Ha plots
    '''
    [a.tick_params(axis='both', labelsize='6') for a in f.axes[:]]
    [a.minorticks_on() for a in f.axes[:]]

    f.suptitle(title, size=15)    
    f.subplots_adjust(wspace=0.2)
    f.subplots_adjust(hspace=0.2)
    
    return f
#enddef