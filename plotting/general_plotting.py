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
            namematch='Ha-NB704',instr=instr)
        index_1 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB711',instr=instr)
        index_2 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB816',instr=instr)
        index_3 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB921',instr=instr)
        index_4 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB973',instr=instr)
        return [index_0]+[index_1]+[index_2]+[index_3]+[index_4]
    elif instr=='Keck':
        index_0 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB816',instr=instr)
        index_1 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB921',instr=instr)
        index_2 = get_name_index_matches(NAME0, inst_str0, inst_dict, 
            namematch='Ha-NB973',instr=instr)
        return [index_0]*2+[index_1]*2+[index_2]*2
    else:
        print 'error'
        return 0
#enddef

def get_index_list2(stlr_mass, inst_str0, inst_dict, instr):
    '''
    Helper function. Returns an index_list for either MMT or Keck 
    based on instr keyword

    Considers by-stellarmass. For now, goes in 20% percentile bins.
    TODO(generalize?)
    '''
    good_stlr_mass = np.array([x for x in stlr_mass if x > 0])

    # 20% percentile bins
    perc20 = np.percentile(good_stlr_mass, 20)
    index_0 = np.array([x for x in range(len(stlr_mass)) 
        if (stlr_mass[x]>0 and stlr_mass[x]<=perc20) 
        and inst_str0[x] in inst_dict[instr]])
    perc40 = np.percentile(good_stlr_mass, 40)
    index_1 = np.array([x for x in range(len(stlr_mass)) 
        if (stlr_mass[x]>perc20 and stlr_mass[x]<=perc40) 
        and inst_str0[x] in inst_dict[instr]])
    perc60 = np.percentile(good_stlr_mass, 60)
    index_2 = np.array([x for x in range(len(stlr_mass)) 
        if (stlr_mass[x]>perc40 and stlr_mass[x]<=perc60) 
        and inst_str0[x] in inst_dict[instr]])
    perc80 = np.percentile(good_stlr_mass, 80)
    index_3 = np.array([x for x in range(len(stlr_mass)) 
        if (stlr_mass[x]>perc60 and stlr_mass[x]<=perc80) 
        and inst_str0[x] in inst_dict[instr]])
    perc100 = np.percentile(good_stlr_mass, 100)        
    index_4 = np.array([x for x in range(len(stlr_mass)) 
        if (stlr_mass[x]>perc80 and stlr_mass[x]<=perc100) 
        and inst_str0[x] in inst_dict[instr]])
    if instr=='MMT':
        return [index_0]*3+[index_1]*3+[index_2]*3+[index_3]*3+[index_4]*3
    if instr=='Keck':
        return [index_0]*2+[index_1]*2+[index_2]*2+[index_3]*2+[index_4]*2
#enddef

def get_iter_lists(instr, stlr=False):
    '''
    Helper function. Returns  for either MMT or Keck
    based on instr keyword

    Returns different subtitle_list based on stlr param
    '''
    if instr=='MMT':
        xmin_list = np.array([4341,4861,6563])-60
        xmax_list = np.array([4341,4861,6563])+60
        label_list=[r'H$\gamma$',r'H$\beta$',r'H$\alpha$']
        subtitle_list = ['NB704']+['NB711']+['NB816']+['NB921']+['NB973']
        return (xmin_list, xmax_list, label_list, subtitle_list)
    elif instr=='Keck':
        if stlr==False:
            xmin_list = np.array([4861,6563]*3)-60
            xmax_list = np.array([4861,6563]*3)+60
            label_list=[r'H$\beta$',r'H$\alpha$']*3
            subtitle_list = ['NB816']*2+['NB921']*2+['NB973']*2
        else: #stlr==True
            xmin_list = np.array([4861,6563]*5)-60
            xmax_list = np.array([4861,6563]*5)+60
            label_list=[r'H$\beta$',r'H$\alpha$']*5
            subtitle_list = []
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
    f.subplots_adjust(wspace=0.2)
    f.subplots_adjust(hspace=0.2)
    
    return f
#enddef