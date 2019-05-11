"""
NAME:
    hg_hb_ha_plotting.py

PURPOSE:
    Provides a module where plotting functions can be called from 
    stack_spectral_data.py for Hg/Hb/Ha (MMT) plots
"""

from analysis.balmer_fit import find_nearest, get_best_fit, get_best_fit2, get_best_fit3
from matplotlib.ticker import MaxNLocator
import numpy as np

# newt.phys.unsw.edu.au/~jkw/alpha/useful_lines.pdf
HG = 4340.47
HB = 4861.33
HA = 6562.80
OIII4363 = 4363.21
NII6548 = 6548.03
NII6583 = 6583.41

def subplots_setup(ax, ax_list, label, subtitle, num, pos_flux=0, flux=0, 
    pos_amp=0, neg_amp=0, pos_sigma=0, neg_sigma=0, continuum=0, hb_nb921_flux=0, 
    ew=0, ew_abs=0, flux_niib=0, rms=0, bintype='N/A', ftitle='', publ=True):
    '''
    Sets up the subplots for Hg/Hb/Ha. Adds emission lines for each subplot
    and sets the ylimits for each row. Also adds flux labels to each subplot.
    '''
    if flux==0 and pos_flux==0 and pos_amp==0 and neg_amp==0 and pos_sigma==0 and neg_sigma==0 and continuum==0:
        # this part only really works for mmt m+z nb921,973
        ax.set_axis_off()
        return ax

    ax.text(0.03,0.97,label,transform=ax.transAxes,fontsize=7,ha='left',
            va='top')
    if num%3!=2: #Hg or Hb
        if (subtitle=='NB921' or ftitle=='NB921') and num%3==1 and hb_nb921_flux > 0:
            if publ==True:
                ax.text(0.97,0.97,'Flux='+'{:.3f}'.format((flux/1E-17))+
                    # '\nRMS='+'{:.3f}'.format((rms/1E-17))+
                    '\nS/N='+'{:.3f}'.format(flux/rms)+
                    '\nEW='+'{:.3f}'.format(ew)[:4]+
                    '\nEW_abs='+'{:.3f}'.format(ew_abs)[:4],
                    # '\nEW='+'{:.3f}'.format(flux/continuum)+
                    # '\nEW_abs='+'{:.3f}'.format((flux - pos_flux)/continuum),
                    transform=ax.transAxes,fontsize=5,ha='right',va='top')
            else:
                ax.text(0.97,0.97,'flux='+'{:.3f}'.format((pos_flux/1E-17))+
                    '\nflux_corr='+'{:.3f}'.format((flux/1E-17))+
                    '\nflux_yes_ha='+'{:.3f}'.format((hb_nb921_flux/1E-17))+
                    '\nA'+r'$+$'+'='+'{:.3f}'.format((pos_amp/1E-17))+
                    '\nA'+r'$-$'+'='+'{:.3f}'.format((neg_amp/1E-17))+
                    '\n'+r'$\sigma+$'+'='+'{:.3f}'.format((pos_sigma))+
                    '\n'+r'$\sigma-$'+'='+'{:.3f}'.format((neg_sigma))+
                    '\ncontinuum='+'{:.3f}'.format((continuum/1E-17)),transform=ax.transAxes,fontsize=5,ha='right',va='top')
        else:
            if publ==True:
                ax.text(0.97,0.97,'Flux='+'{:.3f}'.format((flux/1E-17))+
                    # '\nRMS='+'{:.3f}'.format((rms/1E-17))+
                    '\nS/N='+'{:.3f}'.format(flux/rms)+
                    '\nEW='+'{:.3f}'.format(ew)[:4]+
                    '\nEW_abs='+'{:.3f}'.format(ew_abs)[:4],
                    # '\nEW='+'{:.3f}'.format(flux/continuum)+
                    # '\nEW_abs='+'{:.3f}'.format((flux - pos_flux)/continuum),
                    transform=ax.transAxes,fontsize=5,ha='right',va='top')
            else:
                ax.text(0.97,0.97,'flux='+'{:.3f}'.format((pos_flux/1E-17))+
                    '\nflux_corr='+'{:.3f}'.format((flux/1E-17))+
                    '\nA'+r'$+$'+'='+'{:.3f}'.format((pos_amp/1E-17))+
                    '\nA'+r'$-$'+'='+'{:.3f}'.format((neg_amp/1E-17))+
                    '\n'+r'$\sigma+$'+'='+'{:.3f}'.format((pos_sigma))+
                    '\n'+r'$\sigma-$'+'='+'{:.3f}'.format((neg_sigma))+
                    '\ncontinuum='+'{:.3f}'.format((continuum/1E-17)),transform=ax.transAxes,fontsize=5,ha='right',va='top')
    if num%3==0:
        ax.set_ylabel(r'M$_*$'+subtitle[8:],fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    elif ftitle=='NB921' and num%3==1 and subtitle[10] != '8':
        ymaxval = max(ax.get_ylim())
        ax_list[num-1].plot([HG,HG],[-1,ymaxval],'k',alpha=0.7,zorder=1)
        ax_list[num-1].plot([OIII4363,OIII4363],[-1,ymaxval],'k:',alpha=0.4,zorder=1)
        ax_list[num].plot([HB,HB],[-1,ymaxval],'k',alpha=0.7,zorder=1)
        if subtitle[10]=='6':
            [a.set_ylim([-0.05, ymaxval]) for a in ax_list[num-1:num+1]]
        elif subtitle[10]=='7':
            [a.set_ylim([0, ymaxval]) for a in ax_list[num-1:num+1]]
    elif num%3==2 and subtitle!='NB973': #Ha
        ymaxval = max(ax.get_ylim())
        if bintype!='StellarMassZ' or ftitle!='NB973':
            [a.set_ylim(ymax=ymaxval) for a in ax_list[num-2:num]]
        ax_list[num-2].plot([HG,HG],[0,ymaxval],'k',alpha=0.7,zorder=1)
        ax_list[num-2].plot([OIII4363,OIII4363],[0,ymaxval],'k:',alpha=0.4,zorder=1)
        ax_list[num-1].plot([HB,HB],[0,ymaxval],'k',alpha=0.7,zorder=1)
        ax_list[num].plot([HA,HA], [0,ymaxval],'k',alpha=0.7,zorder=1)
        ax_list[num].plot([NII6548,NII6548],[0,ymaxval], 'k:',alpha=0.4,zorder=1)
        ax_list[num].plot([NII6583,NII6583],[0,ymaxval], 'k:',alpha=0.4,zorder=1)
        if publ==True:
            ax.text(0.97,0.97,'Flux='+'{:.3f}'.format((pos_flux/1E-17))+
                '\n[N II]='+'{:.3f}'.format((flux_niib/pos_flux))+ # [N II] is the flux of [N II]lambda6583/Ha
                # '\nRMS='+'{:.3f}'.format((rms/1E-17))+
                '\nS/N='+'{:.3f}'.format(pos_flux/rms)+
                '\nEW='+'{:.3f}'.format(ew)[:4],
                # '\nEW='+'{:.3f}'.format(flux/continuum),
                transform=ax.transAxes,fontsize=5,ha='right',va='top')
        else:
            ax.text(0.97,0.97,'flux='+'{:.3f}'.format((pos_flux/1E-17))+
                '\nA'+r'$+$'+'='+'{:.3f}'.format((pos_amp/1E-17))+
                '\n'+r'$\sigma+$'+'='+'{:.3f}'.format((pos_sigma))+
                '\ncontinuum='+'{:.3f}'.format((continuum/1E-17)),transform=ax.transAxes,fontsize=5,ha='right',va='top')
    elif subtitle=='NB973' and num%3==1:
        ymaxval = max(ax.get_ylim())
        [a.set_ylim(ymax=ymaxval) for a in ax_list[num-1:num]]
        ax_list[num-1].plot([HG,HG],[0,ymaxval],'k',alpha=0.7,zorder=1)
        ax_list[num-1].plot([OIII4363,OIII4363],[0,ymaxval],'k:',alpha=0.4,zorder=1)
        ax_list[num].plot([HB,HB],[0,ymaxval],'k',alpha=0.7,zorder=1)
    elif bintype=='StellarMassZ' and ftitle=='NB973' and num%3==1:
        ymaxval = max(ax.get_ylim())
        [a.set_ylim(ymax=ymaxval) for a in ax_list[num-1:num]]
        ax_list[num-1].plot([HG,HG],[0,ymaxval],'k',alpha=0.7,zorder=1)
        ax_list[num-1].plot([OIII4363,OIII4363],[0,ymaxval],'k:',alpha=0.4,zorder=1)
        ax_list[num].plot([HB,HB],[0,ymaxval],'k',alpha=0.7,zorder=1)
    if num%3!=0:
        ax.set_yticklabels([])
    if num<12:
        ax.set_xticklabels([])
    if num%3==2 and flux==0:
        if num<3: 
            ymaxval = 0.7
        else: 
            ymaxval = 0.8
            ax_list[num-2].yaxis.set_major_locator(MaxNLocator(prune='upper', nbins=5))
        if bintype!='StellarMassZ' or ftitle!='NB973':
            [a.set_ylim(ymax=ymaxval) for a in ax_list[num-2:num]]
    #endif
    return ax
#enddef


def subplots_plotting(ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol, len_ii=999):
    '''
    Plots all the spectra for the subplots for Hg/Hb/Ha. Also calculates
    and returns fluxes and equations of best fit.
    '''
    # within range
    if len_ii < 2:
        ax.set_xlim(xmin0, xmax0)
        # ax.set_axis_off()
        raise IndexError('Not enough sources to stack (less than two)')
        
    good_ii = np.array([x for x in range(len(xval)) if xval[x] >= xmin0 and xval[x] <= xmax0])
    xval = xval[good_ii]
    yval = yval[good_ii]

    # not NaN
    good_ii = [ii for ii in range(len(yval)) if not np.isnan(yval[ii])]
    xval = xval[good_ii]
    yval = yval[good_ii]

    if len(yval) == 0:
        return ax, 0, 0, 0, 0, np.array([0]*7)

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
        
        peak_idx1_left  = find_nearest(xval, HA-tol)
        peak_idx1_right = find_nearest(xval, HA+tol)
        xval1=xval[peak_idx1_left:peak_idx1_right]
        flux = np.sum(dlambda * (o1[0]*np.exp(-0.5*((xval1-o1[1])/o1[2])**2)))
        # flux = np.sum(dlambda * (o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)))
        pos_flux = flux

        if subtitle=='NB973':
            flux2 = 0
            flux3 = 0
        else: #elif subtitle!='NB973':
            print 'label|subtitle', label, subtitle
            peak_idx2_left  = find_nearest(xval, NII6548-tol)
            peak_idx2_right = find_nearest(xval, NII6548+tol)
            xval2=xval[peak_idx2_left:peak_idx2_right]
            yval2=yval[peak_idx2_left:peak_idx2_right]
            o2 = get_best_fit2(xval2, yval2, NII6548, label)
            flux2 = np.sum(dlambda * (o2[0]*np.exp(-0.5*((xval2-o2[1])/o2[2])**2)))
            ax.plot(xval2, (o2[3]+o2[0]*np.exp(-0.5*((xval2-o2[1])/o2[2])**2))/1E-17, 'g,', zorder=3)

            peak_idx3_left  = find_nearest(xval, NII6583-tol)
            peak_idx3_right = find_nearest(xval, NII6583+tol)
            xval3=xval[peak_idx3_left:peak_idx3_right]
            yval3=yval[peak_idx3_left:peak_idx3_right]
            o3 = get_best_fit2(xval3, yval3, NII6583, label)
            flux3 = np.sum(dlambda * (o3[0]*np.exp(-0.5*((xval3-o3[1])/o3[2])**2)))
            ax.plot(xval3, (o3[3]+o3[0]*np.exp(-0.5*((xval3-o3[1])/o3[2])**2))/1E-17, 'g,', zorder=3)

        idx_small = np.where(np.absolute(xval - o1[1]) <= 2.5*o1[2])[0]
        func0 = o1[3]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)
        ax.plot(xval[idx_small], (yval[idx_small] - func0[idx_small] + o1[3])/1E-17, c='#1f77b4', ls='--', alpha=0.5)
    else:
        o1 = get_best_fit3(xval, yval, label)
        pos0 = o1[5]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)
        neg0 = o1[3]*np.exp(-0.5*((xval-o1[1])/o1[4])**2)
        func0 = pos0 + neg0
        ax.plot(xval, func0/1E-17, 'r--', zorder=3)
        # ax.plot(xval, pos0/1E-17, 'c--', zorder=2)
        ax.plot(xval, (o1[5]+neg0)/1E-17, 'orange', ls='--', zorder=2)
        
        idx_small = np.where(np.absolute(xval - o1[1]) <= 2.5*o1[2])[0]

        pos_flux = np.sum(dlambda * (pos0[idx_small] - o1[5]))
        flux = np.sum(dlambda * (pos0[idx_small] - o1[5] - neg0[idx_small]))
        flux2 = 0
        flux3 = 0

        ax.plot(xval[idx_small], (yval[idx_small] - func0[idx_small] + o1[5])/1E-17, color='#1f77b4', ls='--', alpha=0.5)
    #endif

    ax.set_xlim(xmin0, xmax0)
    ax.set_ylim(ymin=0)
    
    return ax, flux, flux2, flux3, pos_flux, o1
#enddef