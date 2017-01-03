"""
NAME:
    stack_spectral_data.py
    (previously stack_spectral_MMT_data.py, stack_spectral_Keck_data.py)

PURPOSE:
    This code creates a PDF file with 15 subplots, filter-emission line
    row-major order, to show all the MMT and Keck spectral data stacked and
    plotted in a 'de-redshifted' frame.

INPUTS:
    'Catalogs/python_outputs/nbia_all_nsource.fits'
    'Catalogs/nb_ia_zspec.txt'
    'Spectra/spectral_MMT_grid_data.txt'
    'Spectra/spectral_MMT_grid.fits'

CALLING SEQUENCE:
    main body -> create_ordered_AP_arrays, 
                 plot_MMT_Ha / plot_Keck_Ha -> get_name_index_matches, 
                                correct_instr_AP, get_best_fit -> func (etc)

OUTPUTS:
    'Spectra/Ha_MMT_stacked_ew.txt'
    'Spectra/Ha_MMT_stacked_fluxes.txt'
    'Spectra/Ha_MMT_stacked.pdf'
    'Spectra/Ha_Keck_stacked_ew.txt'
    'Spectra/Ha_Keck_stacked_fluxes.txt'
    'Spectra/Ha_Keck_stacked.pdf'

REVISION HISTORY:
    Created by Kaitlyn Shin 25 July 2016
    Revised by Kaitlyn Shin 04 August 2016
    o Added NII fluxes to the tables and fit the lines on the plots
    Revised by Kaitlyn Shin 09 October 2016
    o Added HG/HB stellar absorption
    o Re-calculated EW values
    o Refactored creating the ordered AP arrays into a separate code
"""

import numpy as np, matplotlib.pyplot as plt, sys, math
sys.path.append('/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/codes')
import stack_data
reload(stack_data)
import create_ordered_AP_arrays
reload(create_ordered_AP_arrays)
from astropy.io import fits as pyfits, ascii as asc
from scipy.interpolate import interp1d
import scipy.optimize as optimization
from astropy.table import Table


#----get_name_index_matches--------------------------------------------------#
# o Returns the indexes from which the kwargs name is in the ordered NAME0
#   array and the kwargs instr is in the ordered inst_dict dict.
#----------------------------------------------------------------------------#
def get_name_index_matches(*args, **kwargs):
    namematch = kwargs['namematch']
    instr     = kwargs['instr']
    index = np.array([x for x in range(len(NAME0)) if namematch in NAME0[x] and
                      inst_str0[x] in inst_dict[instr]])         ##### add in stellar mass check??????
    return index
#enddef


#----correct_instr_AP--------------------------------------------------------#
# o Returns the indexed AP_match array based on the 'match_index' from
#   plot_MMT/Keck_Ha
#----------------------------------------------------------------------------#
def correct_instr_AP(indexed_AP, indexed_inst_str0, instr):
    for ii in range(len(indexed_inst_str0)):
        if indexed_inst_str0[ii]=='merged,':
            if instr=='MMT':
                indexed_AP[ii] = indexed_AP[ii][:5]
            elif instr=='Keck':
                indexed_AP[ii] = indexed_AP[ii][6:]
        #endif
    #endfor
    return indexed_AP
#enddef


#----find_nearest------------------------------------------------------------#
# o Uses np.searchsorted to find the array index closest to the input
#   numerical value
#----------------------------------------------------------------------------#
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
#enddef


#----get_baseline_median-----------------------------------------------------#
# o Returns the median of the baseline of the spectrum, masking out the
#   emission peaks
#----------------------------------------------------------------------------#
def get_baseline_median(xval, yval, label):
    if 'gamma' in label:
        peak_l = find_nearest(xval, 4341-5)
        peak_r = find_nearest(xval, 4341+5)

        temparr = np.concatenate([yval[:peak_l], yval[peak_r:]], axis=0)
        return np.median(temparr)
    if 'beta' in label:
        peak_l = find_nearest(xval, 4861-5)
        peak_r = find_nearest(xval, 4861+5)

        temparr = np.concatenate([yval[:peak_l], yval[peak_r:]], axis=0)
        return np.median(temparr)
    if 'alpha' in label:
        nii_1l = find_nearest(xval, 6548.1-3)
        nii_1r = find_nearest(xval, 6548.1+3)

        peak_l = find_nearest(xval, 6563-5)
        peak_r = find_nearest(xval, 6563+5)

        nii_2l = find_nearest(xval, 6583.6-3)
        nii_2r = find_nearest(xval, 6583.6+3)

        temparr = np.concatenate([yval[:nii_1l], yval[nii_1r:peak_l],
                                  yval[peak_r:nii_2l], yval[nii_2r:]], axis=0)
        return np.median(temparr)
    else:
        print 'error!'
        return 0
#enddef


#----func--------------------------------------------------------------------#
# o Is the passed-in model function for optimization.curve_fit
#----------------------------------------------------------------------------#
def func(x, a, b, c, d):
    u = (x-b)/c
    return a * np.exp(-0.5*u*u) + d
#enddef


#----func3-------------------------------------------------------------------#
# o Is the passed-in model function for optimization.curve_fit
#----------------------------------------------------------------------------#
def func3(x, a1, b1, c1, a2, b2, c2, d):
    u = (x-b1)/c1
    v = (x-b2)/c2
    return a1*np.exp(-0.5*u*u) + a2*np.exp(-0.5*v*v) + d
#enddef


#----get_best_fit------------------------------------------------------------#
# o Uses scipy.optimize.curve_fit() to obtain the best fit of the spectra
#   which is then returned
#----------------------------------------------------------------------------#
def get_best_fit(xval, yval, label):
    med0 = get_baseline_median(xval, yval, label)
    err = np.repeat(1.0e-18, len(xval))
    p0 = [np.max(yval)-med0, xval[np.argmax(yval)], 1.10, med0]

    o1,o2 = optimization.curve_fit(func, xval, yval, p0, err)
    return o1
#enddef


#----get_best_fit2-----------------------------------------------------------#
# o Uses scipy.optimize.curve_fit() to obtain the best fit of the spectra
#   which is then returned
# o NII 6548 (6548.1 A)
# o NII 6583 (6583.6 A)
#----------------------------------------------------------------------------#
def get_best_fit2(xval, yval, peakxval, label):
    med0 = np.median(yval)
    err = np.repeat(1.0e-18, len(xval))
    p0 = [yval[find_nearest(xval, peakxval)], peakxval, 1.10, med0]

    o1,o2 = optimization.curve_fit(func, xval, yval, p0, err)
    return o1
#enddef


#----get_best_fit3-----------------------------------------------------------#
# o Uses scipy.optimize.curve_fit() to obtain the best fit of the spectra
#   which is then returned
# o Hg and Hb absorption spectra
#----------------------------------------------------------------------------#
def get_best_fit3(xval, yval, label):
    med0 = get_baseline_median(xval, yval, label)
    err = np.repeat(1.0e-18, len(xval))
    p0 = [np.max(yval)-med0, xval[np.argmax(yval)], 1.10,
          -0.05*(np.max(yval)-med0), xval[np.argmax(yval)], 2.20, med0]

    o1,o2 = optimization.curve_fit(func3, xval, yval, p0, err)
    return o1
#enddef


#----plot_MMT_Ha-------------------------------------------------------------#
# o Calls get_name_index_matches in order to get the indexes at which
#   there is the particular name match and instrument and then creates a
#   master index list.
# o Creates a pdf (8"x11") with 5x3 subplots for different lines and filter
#   combinations.
# o Then, the code iterates through every subplot in row-major filter-line
#   order. Using only the 'good' indexes, finds 'match_index'. With those
#   indexes of AP and inst_str0, calls AP_match.
# o For NB921 Halpha, does a cross-match to ensure no 'invalid' point is
#   plotted.
# o Except for NB973 Halpha, the graph is 'de-redshifted' in order to have
#   the spectral line appear in the subplot. The values to plot are called
#   from stack_data.stack_data
# o get_best_fit is called to obtain the best-fit spectra, overlay the
#   best fit, and then calculate the flux
# o Additionally, a line is plotted at the value at which the emission line
#   should theoretically be (based on which emission line it is).
# o The yscale is fixed for each filter type (usually the yscale values of
#   the Halpha subplot).
# o Minor ticks are set on, lines and filters are labeled, and with the
#   line label is another label for the number of stacked sources that were
#   used to produce the emission graph.
# o At the end of all the iterations, the plot is saved and closed.
# o The fluxes are also output to a separate .txt file.
#----------------------------------------------------------------------------#
def plot_MMT_Ha():
    tablenames  = []
    tablefluxes = []
    nii6548fluxes = []
    nii6583fluxes = []
    ewlist = []
    ewposlist = []
    ewneglist = []
    ewchecklist = []
    medianlist = []
    pos_amplitudelist = []
    neg_amplitudelist = []
    index_0 = get_name_index_matches(namematch='Ha-NB704',instr='MMT')
    index_1 = get_name_index_matches(namematch='Ha-NB711',instr='MMT')
    index_2 = get_name_index_matches(namematch='Ha-NB816',instr='MMT')
    index_3 = get_name_index_matches(namematch='Ha-NB921',instr='MMT')
    index_4 = get_name_index_matches(namematch='Ha-NB973',instr='MMT')
    index_list = [index_0]*3+[index_1]*3+[index_2]*3+[index_3]*3+[index_4]*3
    xmin_list = np.array([4341,4861,6563]*5)-60
    xmax_list = np.array([4341,4861,6563]*5)+60
    label_list=[r'H$\gamma$',r'H$\beta$',r'H$\alpha$']*5

    subtitle_list = ['NB704']*3+['NB711']*3+['NB816']*3+['NB921']*3+['NB973']*3
    f, axarr = plt.subplots(5, 3)
    f.set_size_inches(8, 11)
    ax_list = [axarr[0,0],axarr[0,1],axarr[0,2],axarr[1,0],axarr[1,1],
               axarr[1,2],axarr[2,0],axarr[2,1],axarr[2,2],axarr[3,0],
               axarr[3,1],axarr[3,2],axarr[4,0],axarr[4,1],axarr[4,2]]
    num=0
    for (match_index0,ax,xmin0,xmax0,label,subtitle) in zip(index_list,ax_list,
                                                            xmin_list,xmax_list,
                                                            label_list, 
                                                            subtitle_list):
        if 'gamma' in label:
            input_norm = HG_Y0[match_index0]
        elif 'beta' in label:
            input_norm = HB_Y0[match_index0]
        elif 'alpha' in label:
            input_norm = HA_Y0[match_index0]
        #endif

        good_index = [x for x in range(len(input_norm)) if
                      input_norm[x]!=-99.99999 and input_norm[x]!=-1
                      and input_norm[x]!=0]
        match_index = match_index0[good_index]
        
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
        if subtitle=='NB921' and 'alpha' in label:
            good_AP_match = np.array([x for x in range(len(AP_match)) if
                                      AP_match[x] in good_NB921_Halpha])
            AP_match = AP_match[good_AP_match]
        #endif
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)
        try:
            label += ' ('+str(len(input_index))+')'
            print label, subtitle
            xval, yval = stack_data.stack_data(grid_ndarr, gridz, input_index,
                                               x0, xmin0, xmax0, subtitle)
            ax.plot(xval, yval/1E-17, zorder=2)
            
            # calculating flux for NII emissions
            zs = np.array(gridz[input_index])
            if subtitle=='NB704' or subtitle=='NB711':
                good_z = np.where(zs < 0.1)[0]
            elif subtitle=='NB816':
                good_z = np.where(zs < 0.3)[0]
            elif subtitle=='NB921':
                good_z = np.where(zs < 0.6)[0]
            else:
                good_z = np.where(zs < 0.6)[0]
            #endif
            zs = np.average(zs[good_z])
            dlambda = (x0[1]-x0[0])/(1+zs)

            ew_emission = 0
            ew_absorption = 0
            ew_check = 0
            median = 0
            pos_amplitude = 0
            neg_amplitude = 0
            if 'alpha' in label:
                o1 = get_best_fit(xval, yval, label)
                ax.plot(xval, (o1[3]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2))/1E-17,
                        'r--', zorder=3)

                flux = np.sum(dlambda * (o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)))
                pos_flux = flux

                ew = flux/o1[3]
                ew_emission = ew
                ew_check = ew
                median = o1[3]
                pos_amplitude = o1[0]
                neg_amplitude = 0
                if subtitle=='NB973':
                    flux2 = 0
                    flux3 = 0
                else:
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

                ew = flux/o1[6]
                median = o1[6]
                pos_amplitude = o1[0]
                neg_amplitude = o1[3]

                if (neg_amplitude > 0): 
                    neg_amplitude = 0
                    ew = pos_flux/o1[6]
                    ew_emission = ew
                    ew_check = ew
                else:
                    pos_corr = np.sum(dlambda * (pos0[idx_small] - o1[6]))
                    ew_emission = pos_corr / o1[6]
                    neg_corr = np.sum(dlambda * neg0[idx_small])
                    ew_absorption = neg_corr / o1[6]
                    ew_check = ew_emission + ew_absorption
            #endif
            
            ax.set_xlim(xmin0, xmax0)
            ax.set_ylim(ymin=0)
        except ValueError:
            print 'ValueError: none exist'
        #endtry
        
        ax.text(0.03,0.97,label,transform=ax.transAxes,fontsize=7,ha='left',
                va='top')
        
        if not (subtitle=='NB973' and num%3==2):
            ax.text(0.97,0.97,'flux_before='+'{:.4e}'.format((pos_flux))+
                '\nflux='+'{:.4e}'.format((flux)),transform=ax.transAxes,fontsize=7,ha='right',va='top')
            tablenames.append(label+'_'+subtitle)
            tablefluxes.append(flux)
            nii6548fluxes.append(flux2)
            nii6583fluxes.append(flux3)
            ewlist.append(ew)
            ewposlist.append(ew_emission)
            ewneglist.append(ew_absorption)
            ewchecklist.append(ew_check)
            medianlist.append(median)
            pos_amplitudelist.append(pos_amplitude)
            neg_amplitudelist.append(neg_amplitude)
        if num%3==0:
            ax.set_title(subtitle,fontsize=8,loc='left')
        elif num%3==2 and subtitle!='NB973':
            ymaxval = max(ax.get_ylim())
            plt.setp([a.set_ylim(ymax=ymaxval) for a in ax_list[num-2:num]])
            ax_list[num-2].plot([4341,4341],[0,ymaxval],'k',alpha=0.7,zorder=1)
            ax_list[num-2].plot([4363,4363],[0,ymaxval],'k:',alpha=0.4,zorder=1)
            ax_list[num-1].plot([4861,4861],[0,ymaxval],'k',alpha=0.7,zorder=1)
            ax_list[num].plot([6563,6563], [0,ymaxval],'k',alpha=0.7,zorder=1)
            ax_list[num].plot([6548,6548],[0,ymaxval], 'k:',alpha=0.4,zorder=1)
            ax_list[num].plot([6583,6583],[0,ymaxval], 'k:',alpha=0.4,zorder=1)
        elif subtitle=='NB973' and num%3==1:
            ymaxval = max(ax.get_ylim())
            plt.setp([a.set_ylim(ymax=ymaxval) for a in ax_list[num-1:num]])
            ax_list[num-1].plot([4341,4341],[0,ymaxval],'k',alpha=0.7,zorder=1)
            ax_list[num-1].plot([4363,4363],[0,ymaxval],'k:',alpha=0.4,zorder=1)
            ax_list[num].plot([4861,4861],[0,ymaxval],'k',alpha=0.7,zorder=1)
        #endif
        num+=1
    #endfor
    f.suptitle(r'MMT detections of H$\alpha$ emitters', size=15)
    plt.setp([a.get_xticklabels() for a in f.axes[:]], size='6')
    plt.setp([a.get_yticklabels() for a in f.axes[:]], size='6')
    plt.setp([a.minorticks_on() for a in f.axes[:]])
    f.subplots_adjust(wspace=0.2)
    f.subplots_adjust(hspace=0.2)
    plt.savefig('Spectra/Ha_MMT_stacked.pdf')
    plt.close()

    #writing the table
    table = Table([tablenames,tablefluxes,nii6548fluxes,nii6583fluxes],
                  names=['type','flux','NII6548 flux','NII6583 flux'])
    asc.write(table, 'Spectra/Ha_MMT_stacked_fluxes.txt',
              format='fixed_width', delimiter=' ')  

    #writing the EW table
    table0 = Table([tablenames,ewlist,ewposlist,ewneglist,ewchecklist,medianlist,pos_amplitudelist,neg_amplitudelist], 
        names=['type','EW','EW_corr','EW_abs','ew check','median','pos_amplitude','neg_amplitude'])
    asc.write(table0, 'Spectra/Ha_MMT_stacked_ew.txt', format='fixed_width', delimiter=' ')  
#enddef


#----plot_Keck_Ha------------------------------------------------------------#
# o Calls get_name_index_matches in order to get the indexes at which
#   there is the particular name match and instrument and then creates a
#   master index list.
# o Creates a pdf (8"x11") with 3x2 subplots for different lines and filter
#   combinations.
# o Then, the code iterates through every subplot in row-major filter-line
#   order. Using only the 'good' indexes, finds 'match_index'. With those
#   indexes of AP and inst_str0, calls AP_match.
# o For NB921 Halpha, does a cross-match to ensure no 'invalid' point is
#   plotted.
# o Except for NB973 Halpha, the graph is 'de-redshifted' in order to have
#   the spectral line appear in the subplot. The values to plot are called
#   from stack_data.stack_data
# o get_best_fit is called to obtain the best-fit spectra, overlay the
#   best fit, and then calculate the flux
# o Additionally, a line is plotted at the value at which the emission line
#   should theoretically be (based on which emission line it is).
# o The yscale is fixed for each filter type (usually the yscale values of
#   the Halpha subplot).
# o Minor ticks are set on, lines and filters are labeled, and with the
#   line label is another label for the number of stacked sources that were
#   used to produce the emission graph.
# o At the end of all the iterations, the plot is saved and closed.
# o The fluxes are also output to a separate .txt file.
#----------------------------------------------------------------------------#
def plot_Keck_Ha():
    tablenames  = []
    tablefluxes = []
    nii6548fluxes = []
    nii6583fluxes = []
    ewlist = []
    ewposlist = []
    ewneglist = []
    ewchecklist = []
    medianlist = []
    pos_amplitudelist = []
    neg_amplitudelist = []
    index_0 = get_name_index_matches(namematch='Ha-NB816',instr='Keck')
    index_1 = get_name_index_matches(namematch='Ha-NB921',instr='Keck')
    index_2 = get_name_index_matches(namematch='Ha-NB973',instr='Keck')
    index_list = [index_0]*2+[index_1]*2+[index_2]*2
    xmin_list = np.array([4861,6563]*3)-60
    xmax_list = np.array([4861,6563]*3)+60
    label_list=[r'H$\beta$',r'H$\alpha$']*3

    subtitle_list = ['NB816']*2+['NB921']*2+['NB973']*2
    f, axarr = plt.subplots(3, 2)
    f.set_size_inches(8, 11)
    ax_list = [axarr[0,0],axarr[0,1],axarr[1,0],
               axarr[1,1],axarr[2,0],axarr[2,1]]
    num=0
    for (match_index0,ax,xmin0,xmax0,label,subtitle) in zip(index_list,ax_list,
                                                            xmin_list,xmax_list,
                                                            label_list, 
                                                            subtitle_list):
        if 'beta' in label:
            input_norm = HB_Y0[match_index0]
        elif 'alpha' in label:
            input_norm = HA_Y0[match_index0]
        #endif

        good_index = [x for x in range(len(input_norm)) if
                      input_norm[x]!=-99.99999 and input_norm[x]!=-1
                      and input_norm[x]!=0]
        match_index = match_index0[good_index]
        
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'Keck')
        AP_match = np.array(AP_match, dtype=np.float32)
        
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match and gridz[x] != 0],dtype=np.int32)
        try:
            label += ' ('+str(len(input_index))+')'
            print label, subtitle
            xval, yval = stack_data.stack_data(grid_ndarr, gridz, input_index,
                                               x0, xmin0, xmax0, subtitle)
            o1 = get_best_fit(xval, yval, label)
            if not (subtitle=='NB816' and num%2==0):
                ax.plot(xval, yval/1E-17, zorder=2)
            #endif

            # calculating flux for NII emissions
            zs = np.array(gridz[input_index])
            if subtitle=='NB816':
                good_z = np.where(zs < 0.3)[0]
            elif subtitle=='NB921':
                good_z = np.where(zs < 0.6)[0]
            else:
                good_z = np.where(zs < 0.6)[0]
            #endif
            zs = np.average(zs[good_z])
            dlambda = (x0[1]-x0[0])/(1+zs)

            ew_emission = 0
            ew_absorption = 0
            ew_check = 0
            median = 0
            pos_amplitude = 0
            neg_amplitude = 0
            if 'alpha' in label:
                ax.plot(xval, (o1[3]+o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2))/1E-17, 'r--', zorder=3)
                flux = np.sum(dlambda * (o1[0]*np.exp(-0.5*((xval-o1[1])/o1[2])**2)))
                pos_flux = flux
                ew = flux/o1[3]
                median = o1[3]
                pos_amplitude = o1[0]
                neg_amplitude = 0
                ew_emission = ew
                ew_check = ew
                
                peak_idx2_left  = find_nearest(xval, 6548.1-tol)
                peak_idx2_right = find_nearest(xval, 6548.1+tol)
                xval2=xval[peak_idx2_left:peak_idx2_right]
                yval2=yval[peak_idx2_left:peak_idx2_right]
                o2 = get_best_fit2(xval2, yval2, 6548.1, label)
                flux2 = np.sum(dlambda * (o2[0]*np.exp(-0.5*((xval2-o2[1])/o2[2])**2)))
                ax.plot(xval2, (o2[3]+o2[0]*np.exp(-0.5*((xval2-o2[1])/o2[2])**2))/1E-17, 'g,', zorder=3)

                peak_idx3_left = find_nearest(xval, 6583.6-tol)
                peak_idx3_right = find_nearest(xval, 6583.6+tol)
                xval3=xval[peak_idx3_left:peak_idx3_right]
                yval3=yval[peak_idx3_left:peak_idx3_right]
                o3 = get_best_fit2(xval3, yval3, 6583.6, label)
                flux3 = np.sum(dlambda * (o3[0]*np.exp(-0.5*((xval3-o3[1])/o3[2])**2)))
                ax.plot(xval3, (o3[3]+o3[0]*np.exp(-0.5*((xval3-o3[1])/o3[2])**2))/1E-17, 'g,', zorder=3)
            elif 'beta' in label and subtitle!='NB816':
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

                ew = flux/o1[6]
                median = o1[6]
                pos_amplitude = o1[0]
                neg_amplitude = o1[3]

                if (neg_amplitude > 0): 
                    neg_amplitude = 0
                    ew = pos_flux/o1[6]
                    ew_emission = ew
                    ew_check = ew
                else:
                    pos_corr = np.sum(dlambda * (pos0[idx_small] - o1[6]))
                    ew_emission = pos_corr / o1[6]
                    neg_corr = np.sum(dlambda * neg0[idx_small])
                    ew_absorption = neg_corr / o1[6]
                    ew_check = ew_emission + ew_absorption
            #endif

            ax.set_xlim(xmin0, xmax0)
            ax.set_ylim(ymin=0)
        except ValueError:
            print 'ValueError: none exist'
        #endtry

        ax.text(0.03,0.97,label,transform=ax.transAxes,fontsize=7,ha='left',
                va='top')

        if not (subtitle=='NB816' and num%2==0):
            ax.text(0.97,0.97,'flux_before='+'{:.4e}'.format((pos_flux))+
                '\nflux='+'{:.4e}'.format((flux)),transform=ax.transAxes,fontsize=7,ha='right',va='top')
            tablenames.append(label+'_'+subtitle)
            tablefluxes.append(flux)
            nii6548fluxes.append(flux2)
            nii6583fluxes.append(flux3)
            ewlist.append(ew)
            ewposlist.append(ew_emission)
            ewneglist.append(ew_absorption)
            ewchecklist.append(ew_check)
            medianlist.append(median)
            pos_amplitudelist.append(pos_amplitude)
            neg_amplitudelist.append(neg_amplitude)
        if num%2==0:
            ax.set_title(subtitle,fontsize=8,loc='left')
        elif num%2==1:
            ymaxval = max(ax.get_ylim())
            plt.setp([a.set_ylim(ymax=ymaxval) for a in ax_list[num-1:num]])
            if subtitle != 'NB816':
                ax_list[num-1].plot([4861,4861],[0,ymaxval],'k',alpha=0.7,zorder=1)
            ax_list[num].plot([6563,6563], [0,ymaxval],'k',alpha=0.7,zorder=1)
            ax_list[num].plot([6548,6548],[0,ymaxval], 'k:',alpha=0.4,zorder=1)
            ax_list[num].plot([6583,6583],[0,ymaxval], 'k:',alpha=0.4,zorder=1)
        #endif
        num+=1
    #endfor
    f.suptitle(r'Keck detections of H$\alpha$ emitters', size=15)
    plt.setp([a.get_xticklabels() for a in f.axes[:]], size='6')
    plt.setp([a.get_yticklabels() for a in f.axes[:]], size='6')
    plt.setp([a.minorticks_on() for a in f.axes[:]])
    f.subplots_adjust(wspace=0.2)
    f.subplots_adjust(hspace=0.2)
    plt.savefig('Spectra/Ha_Keck_stacked.pdf')
    plt.close()

    #writing the table
    table = Table([tablenames,tablefluxes,nii6548fluxes,nii6583fluxes],
                  names=['type','flux','NII6548 flux','NII6583 flux'])
    asc.write(table, 'Spectra/Ha_Keck_stacked_fluxes.txt', format='fixed_width', delimiter=' ')

    #writing the EW table
    table0 = Table([tablenames,ewlist,ewposlist,ewneglist,ewchecklist,medianlist,pos_amplitudelist,neg_amplitudelist], 
        names=['type','EW','EW_corr','EW_abs','ew check','median','pos_amplitude','neg_amplitude'])
    asc.write(table0, 'Spectra/Ha_Keck_stacked_ew.txt', format='fixed_width', delimiter=' ')  
#enddef


#----main body---------------------------------------------------------------#
# o Reads relevant inputs, combining all of the input data into one ordered
#   array for AP by calling make_AP_arr_MMT, make_AP_arr_DEIMOS,
#   make_AP_arr_merged, and make_AP_arr_FOCAS. 
# o Using the AP order, then creates HA, HB, HG_Y0 arrays by calling get_Y0
# o Then looks at the grid stored in 'Spectra/spectral_*_grid_data.txt'
#   and 'Spectra/spectral_*_grid.fits' created from
#   combine_spectral_data.py in order to read in relevant data columns.
# o Then calls plot_*_Ha.
# o Done for both MMT and Keck data.
#----------------------------------------------------------------------------#
good_NB921_Halpha = ['S.245','S.278','S.291','S.306','S.308','S.333','S.334',
                     'S.350','S.364','A.134','D.076','D.099','D.123','D.125',
                     'D.127','D.135','D.140','D.215','D.237','D.298']
inst_dict = {}
inst_dict['MMT'] = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,']
inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,',
                     'Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,']
tol = 3 #in angstroms, used for NII emission flux calculations

nbia = pyfits.open('Catalogs/python_outputs/nbia_all_nsource.fits')
nbiadata = nbia[1].data
NAME0 = nbiadata['source_name']

zspec = asc.read('Catalogs/nb_ia_zspec.txt',guess=False,
                 Reader=asc.CommentedHeader)
slit_str0 = np.array(zspec['slit_str0'])
inst_str0 = np.array(zspec['inst_str0'])

data_dict = create_ordered_AP_arrays.create_ordered_AP_arrays()
AP = data_dict['AP']
HA_Y0 = data_dict['HA_Y0']
HB_Y0 = data_dict['HB_Y0']
HG_Y0 = data_dict['HG_Y0']

print '### looking at the MMT grid'
griddata = asc.read('Spectra/spectral_MMT_grid_data.txt',guess=False)
gridz  = np.array(griddata['ZSPEC'])
gridap = np.array(griddata['AP'])
grid   = pyfits.open('Spectra/spectral_MMT_grid.fits')
grid_ndarr = grid[0].data
grid_hdr   = grid[0].header
CRVAL1 = grid_hdr['CRVAL1']
CDELT1 = grid_hdr['CDELT1']
NAXIS1 = grid_hdr['NAXIS1']
x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1)

print '### plotting MMT_Ha'
plot_MMT_Ha()
grid.close()

print '### looking at the Keck grid'
griddata = asc.read('Spectra/spectral_Keck_grid_data.txt',guess=False)
gridz  = np.array(griddata['ZSPEC'])
gridap = np.array(griddata['AP'])
grid   = pyfits.open('Spectra/spectral_Keck_grid.fits')
grid_ndarr = grid[0].data
grid_hdr   = grid[0].header
CRVAL1 = grid_hdr['CRVAL1']
CDELT1 = grid_hdr['CDELT1']
NAXIS1 = grid_hdr['NAXIS1']
x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1)

print '### plotting Keck_Ha'
plot_Keck_Ha()
grid.close()

nbia.close()
print '### done'
#endmain
