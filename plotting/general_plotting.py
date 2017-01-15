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