def subplots_setup(ax, ax_list, label, subtitle, num, pos_flux=0, flux=0):
    '''
    Sets up the subplots for Hg/Hb/Ha. Adds emission lines for each subplot
    and sets the ylimits for each row. Also adds flux labels to each subplot.
    '''
    ax.text(0.03,0.97,label,transform=ax.transAxes,fontsize=7,ha='left',
            va='top')
    
    if not (subtitle=='NB973' and num%3==2):
        ax.text(0.97,0.97,'flux_before='+'{:.4e}'.format((pos_flux))+
            '\nflux='+'{:.4e}'.format((flux)),transform=ax.transAxes,fontsize=7,ha='right',va='top')
    if num%3==0:
        ax.set_title(subtitle,fontsize=8,loc='left')
    elif num%3==2 and subtitle!='NB973':
        ymaxval = max(ax.get_ylim())
        # plt.setp([a.set_ylim(ymax=ymaxval) for a in ax_list[num-2:num]])
        [a.set_ylim(ymax=ymaxval) for a in ax_list[num-2:num]]
        ax_list[num-2].plot([4341,4341],[0,ymaxval],'k',alpha=0.7,zorder=1)
        ax_list[num-2].plot([4363,4363],[0,ymaxval],'k:',alpha=0.4,zorder=1)
        ax_list[num-1].plot([4861,4861],[0,ymaxval],'k',alpha=0.7,zorder=1)
        ax_list[num].plot([6563,6563], [0,ymaxval],'k',alpha=0.7,zorder=1)
        ax_list[num].plot([6548,6548],[0,ymaxval], 'k:',alpha=0.4,zorder=1)
        ax_list[num].plot([6583,6583],[0,ymaxval], 'k:',alpha=0.4,zorder=1)
    elif subtitle=='NB973' and num%3==1:
        ymaxval = max(ax.get_ylim())
        # plt.setp([a.set_ylim(ymax=ymaxval) for a in ax_list[num-1:num]])
        [a.set_ylim(ymax=ymaxval) for a in ax_list[num-1:num]]
        ax_list[num-1].plot([4341,4341],[0,ymaxval],'k',alpha=0.7,zorder=1)
        ax_list[num-1].plot([4363,4363],[0,ymaxval],'k:',alpha=0.4,zorder=1)
        ax_list[num].plot([4861,4861],[0,ymaxval],'k',alpha=0.7,zorder=1)
    #endif

    return ax
#enddef