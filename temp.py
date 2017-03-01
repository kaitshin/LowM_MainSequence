# out of the, say, the first bin of the <9264 sample w/ valid stlr mass (say ~2000)
# a smaller subset of them will have valid spectra. find them by AP

def plot_MMT_Ha_stlrmass():
    '''
    '''
    table_arrays = ([], [], [], [], [], [], [], [], [], [], [])
    (tablenames, tablefluxes, nii6548fluxes, nii6583fluxes, ewlist, 
        ewposlist , ewneglist, ewchecklist, medianlist, pos_amplitudelist, 
        neg_amplitudelist) = table_arrays
    # index_list = general_plotting.get_index_list(NAME0, inst_str0, inst_dict, 'MMT')

    #below index_list goes for the (20,40,60,80,100) percentiles
    index_list = general_plotting.get_index_list2(stlr_mass, inst_str0, inst_dict, 'MMT')

    (xmin_list, xmax_list, label_list, 
        subtitle_list) = general_plotting.get_iter_lists('MMT')
    
    f, axarr = plt.subplots(5, 3)
    f.set_size_inches(8, 11)
    ax_list = [axarr[0,0],axarr[0,1],axarr[0,2],axarr[1,0],axarr[1,1],
               axarr[1,2],axarr[2,0],axarr[2,1],axarr[2,2],axarr[3,0],
               axarr[3,1],axarr[3,2],axarr[4,0],axarr[4,1],axarr[4,2]]
    
    num=0
    # this for-loop stacks by stlrmass
    for (match_index0,ax,xmin0,xmax0,label,subtitle) in zip(index_list,ax_list,
                                                            xmin_list,xmax_list,
                                                            label_list, 
                                                            subtitle_list):
        shortlabel = ''
        if 'gamma' in label:
            input_norm = HG_Y0[match_index0]
            shortlabel = 'Hg'
        elif 'beta' in label:
            input_norm = HB_Y0[match_index0]
            shortlabel = 'Hb'
        elif 'alpha' in label:
            input_norm = HA_Y0[match_index0]
            shortlabel = 'Ha'
        #endif

        good_index = [x for x in range(len(input_norm)) if
                      input_norm[x]!=-99.99999 and input_norm[x]!=-1
                      and input_norm[x]!=0]
        match_index = match_index0[good_index]
        
        AP_match = correct_instr_AP(AP[match_index], inst_str0[match_index], 'MMT')
        # if subtitle=='NB921' and 'alpha' in label:
        #     good_AP_match = np.array([x for x in range(len(AP_match)) if
        #                               AP_match[x] in good_NB921_Halpha])
        #     AP_match = AP_match[good_AP_match]
        # #endif
        input_index = np.array([x for x in range(len(gridap)) if gridap[x] in
                                AP_match],dtype=np.int32)
        try:
            label += ' ('+str(len(input_index))+')'
            print label, subtitle
            xval, yval = stack_data(grid_ndarr, gridz, input_index,
                x0, xmin0, xmax0, ff=subtitle)
            
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

            ax, flux, flux2, flux3, pos_flux, o1, o2, o3 = MMT_plotting.subplots_plotting(
                ax, xval, yval, label, subtitle, dlambda, xmin0, xmax0, tol)

            (ew, ew_emission, ew_absorption, ew_check, median, pos_amplitude, 
            	neg_amplitude) = MMT_twriting.Hg_Hb_Ha_tables(label, flux, 
            	o1, xval, pos_flux, dlambda)

            table_arrays = general_twriting.table_arr_appends(num, table_arrays, label, 
            	subtitle, flux, flux2, flux3, ew, ew_emission, ew_absorption, ew_check, 
            	median, pos_amplitude, neg_amplitude, 'MMT')
            
            #writing the spectra table
            table0 = Table([xval, yval/1E-17], names=['xval','yval/1E-17'])
            asc.write(table0, full_path+'Spectra/Ha_MMT_spectra_vals/'+shortlabel+'_'+subtitle+'.txt',
                format='fixed_width', delimiter=' ')
        except ValueError:
            print 'ValueError: none exist'
        #endtry
        
        if pos_flux and flux:
            ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, num, pos_flux, flux)
        elif not pos_flux and not flux:
            ax = MMT_plotting.subplots_setup(ax, ax_list, label, subtitle, num)
        else:
            print '>>>something\'s not right...'
        #endif

        num+=1
    #endfor
    f = general_plotting.final_plot_setup(f, r'MMT detections of H$\alpha$ emitters')
    plt.savefig(full_path+'Spectra/Ha_MMT_stacked.pdf')
    plt.close()

    #writing the flux table
    table1 = Table([tablenames,tablefluxes,nii6548fluxes,nii6583fluxes],
        names=['type','flux','NII6548 flux','NII6583 flux'])
    asc.write(table1, full_path+'Spectra/Ha_MMT_stacked_fluxes.txt',
        format='fixed_width', delimiter=' ')  

    #writing the EW table
    table2 = Table([tablenames,ewlist,ewposlist,ewneglist,ewchecklist,medianlist,pos_amplitudelist,neg_amplitudelist],
        names=['type','EW','EW_corr','EW_abs','ew check','median','pos_amplitude','neg_amplitude'])
    asc.write(table2, full_path+'Spectra/Ha_MMT_stacked_ew.txt',
        format='fixed_width', delimiter=' ')  
#enddef