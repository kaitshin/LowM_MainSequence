"""
NAME:
    general_plotting.py

PURPOSE:
    Provides a module where plotting functions can be called from 
    stack_spectral_data.py for both Hg/Hb/Ha and Hb/Ha (MMT, Keck)
    plots
"""

import numpy as np

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

    Considers by-filter
    '''
    if instr=='MMT':
        index_0 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB7',instr=instr)
        index_1 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB816',instr=instr)
        index_2 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB921',instr=instr)
        index_3 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB973',instr=instr)        
        return [index_0]+[index_1]+[index_2]+[index_3]
    elif instr=='Keck':
        index_0 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB816',instr=instr)
        index_1 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB921',instr=instr)
        index_2 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB973',instr=instr)
        return [index_0]+[index_1]+[index_2]
    else:
        print 'error'
        return 0
#enddef

def get_index_list2(NAME0, stlr_mass, inst_str0, zspec0, inst_dict, instr):
    '''
    Helper function. Returns an index_list for either MMT or Keck 
    based on instr keyword

    Considers by-stellarmass. For now, goes in 20% percentile bins.
    Only considers valid redshifts.

    Note: stlrmass+z composites use dynamic stlrmass binning such 
    that the bins have as even a number of sources as possible 
    '''
    if instr=='Keck':
        good_iis = np.array([x for x in range(len(stlr_mass)) if 
            (stlr_mass[x] > 0 and inst_str0[x] in inst_dict[instr]) 
            and ('Ha-NB8' in NAME0[x] or 'Ha-NB9' in NAME0[x])
            and (zspec0[x] > 0 and zspec0[x] < 9)])
    else: #=='MMT'
        good_iis = np.array([x for x in range(len(stlr_mass)) if 
            (stlr_mass[x] > 0 and inst_str0[x] in inst_dict[instr]) 
            and ('Ha-NB' in NAME0[x])
            and (zspec0[x] > 0 and zspec0[x] < 9)])
    good_stlr_mass = stlr_mass[good_iis]

    # 20% percentile bins
    perc20 = np.percentile(good_stlr_mass, 20)
    if instr=='Keck':
        index_0 = np.array([x for x in range(len(stlr_mass)) 
            if (stlr_mass[x]>0 and stlr_mass[x]<=perc20) 
            and (inst_str0[x] in inst_dict[instr]) 
            and ('Ha-NB8' in NAME0[x] or 'Ha-NB9' in NAME0[x])
            and (zspec0[x] > 0 and zspec0[x] < 9)])
    else: #=='MMT'
        index_0 = np.array([x for x in range(len(stlr_mass)) 
            if (stlr_mass[x]>0 and stlr_mass[x]<=perc20) 
            and (inst_str0[x] in inst_dict[instr] and 'Ha-NB' in NAME0[x])
            and (zspec0[x] > 0 and zspec0[x] < 9)])

    # the below don't have problems for keck accidentally including ha-nb704/711 yet...
    perc40 = np.percentile(good_stlr_mass, 40)
    index_1 = np.array([x for x in range(len(stlr_mass)) 
        if (stlr_mass[x]>perc20 and stlr_mass[x]<=perc40) 
        and (inst_str0[x] in inst_dict[instr] and 'Ha-NB' in NAME0[x])
        and (zspec0[x] > 0 and zspec0[x] < 9)])
    perc60 = np.percentile(good_stlr_mass, 60)
    index_2 = np.array([x for x in range(len(stlr_mass)) 
        if (stlr_mass[x]>perc40 and stlr_mass[x]<=perc60) 
        and (inst_str0[x] in inst_dict[instr] and 'Ha-NB' in NAME0[x])
        and (zspec0[x] > 0 and zspec0[x] < 9)])
    perc80 = np.percentile(good_stlr_mass, 80)
    index_3 = np.array([x for x in range(len(stlr_mass)) 
        if (stlr_mass[x]>perc60 and stlr_mass[x]<=perc80) 
        and (inst_str0[x] in inst_dict[instr] and 'Ha-NB' in NAME0[x])
        and (zspec0[x] > 0 and zspec0[x] < 9)])
    perc100 = np.percentile(good_stlr_mass, 100)        
    index_4 = np.array([x for x in range(len(stlr_mass)) 
        if (stlr_mass[x]>perc80 and stlr_mass[x]<=perc100) 
        and (inst_str0[x] in inst_dict[instr] and 'Ha-NB' in NAME0[x])
        and (zspec0[x] > 0 and zspec0[x] < 9)])
    if instr=='MMT':
        return [index_0]+[index_1]+[index_2]+[index_3]+[index_4]
    if instr=='Keck':
        return [index_0]+[index_1]+[index_2]+[index_3]+[index_4]
#enddef

def get_iter_lists(instr, stlr=False):
    '''
    Helper function. Returns  for either MMT or Keck
    based on instr keyword
    '''
    if instr=='MMT':
        xmin_list = np.array([4341,4861,6563])-60
        xmax_list = np.array([4341,4861,6563])+60
        label_list=[r'H$\gamma$',r'H$\beta$',r'H$\alpha$']
        subtitle_list = ['NB704+NB711']+['NB816']+['NB921']+['NB973']
        return (xmin_list, xmax_list, label_list, subtitle_list)
    elif instr=='Keck':
        xmin_list = np.array([4861,6563])-30
        xmax_list = np.array([4861,6563])+30
        label_list=[r'H$\beta$',r'H$\alpha$']
        subtitle_list = ['NB816']+['NB921']+['NB973']
        return (xmin_list, xmax_list, label_list, subtitle_list)
    else:
        print 'error'
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
    
    return f
#enddef