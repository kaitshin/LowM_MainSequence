"""
NAME:
    hg_hb_ha_plotting.py

PURPOSE:
    Provides a module where plotting functions can be called from 
    stack_spectral_data.py for Hg/Hb/Ha (MMT) plots
"""

from analysis.balmer_fit import find_nearest, get_best_fit, get_best_fit2, get_best_fit3
import numpy as np

def subplots_setup(ax, ax_list, label, subtitle, num, pos_flux=0, flux=0):
    '''
    Sets up the subplots for Hg/Hb/Ha. Adds emission lines for each subplot
    and sets the ylimits for each row. Also adds flux labels to each subplot.
    '''
    ax = ax_list[num]
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


def subplots_plotting(ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol):
    '''
    Plots all the spectra for the subplots for Hg/Hb/Ha. Also calculates
    and returns fluxes and equations of best fit.
    '''
    ax.plot(xval, yval/1E-17, zorder=2)
    
    flux = 0
    flux2 = 0
    flux3 = 0
    pos_flux = 0
    o1 = 0
    o2 = 0
    o3 = 0

    if 'alpha' in label:
        o1 = get_best_fit(xval, yval, label)
        ax.plot(xval, (o1[3]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2))/1E-17,
                'r--', zorder=3)
        
        flux = np.sum(dlambda * (o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)))
        pos_flux = flux

        if subtitle=='NB973':
            flux2 = 0
            flux3 = 0
        else: #elif subtitle!='NB973':
            peak_idx2_left  = find_nearest(xval, 6548.1-tol)
            peak_idx2_right = find_nearest(xval, 6548.1+tol)
            xval2=xval[peak_idx2_left:peak_idx2_right]
            yval2=yval[peak_idx2_left:peak_idx2_right]
            o2 = get_best_fit2(xval2, yval2, 6548.1, label)
            flux2 = np.sum(dlambda * (o2[0]*np.exp(-0.5*((xval2-o2[1])/o2[2])**2)))
            ax.plot(xval2, (o2[3]+o2[0]*np.exp(-0.5*((xval2-o2[1])/o2[2])**2))/1E-17, 'g,', zorder=3)

            peak_idx3_left  = find_nearest(xval, 6583.6-tol)
            peak_idx3_right = find_nearest(xval, 6583.6+tol)
            xval3=xval[peak_idx3_left:peak_idx3_right]
            yval3=yval[peak_idx3_left:peak_idx3_right]
            o3 = get_best_fit2(xval3, yval3, 6583.6, label)
            flux3 = np.sum(dlambda * (o3[0]*np.exp(-0.5*((xval3-o3[1])/o3[2])**2)))
            ax.plot(xval3, (o3[3]+o3[0]*np.exp(-0.5*((xval3-o3[1])/o3[2])**2))/1E-17, 'g,', zorder=3)
    else:
        o1 = get_best_fit3(xval, yval, label)
        pos0 = o1[6]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)
        neg0 = o1[3]*np.exp(-0.5*((xval-o1[4])/o1[5])**2)
        func0 = pos0 + neg0
        ax.plot(xval, func0/1E-17, 'r--', zorder=3)
        
        idx_small = np.where(np.absolute(xval - o1[1]) <= 2.5*o1[2])[0]

        pos_flux = np.sum(dlambda * (pos0[idx_small] - o1[6]))
        flux = np.sum(dlambda * (func0[idx_small] - o1[6]))
        flux2 = 0
        flux3 = 0
    #endif

    ax.set_xlim(xmin0, xmax0)
    ax.set_ylim(ymin=0)
    
    return ax, flux, flux2, flux3, pos_flux, o1, o2, o3
#enddef