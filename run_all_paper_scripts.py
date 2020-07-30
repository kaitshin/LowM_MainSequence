"""
NAME:
    run_all_paper_scripts.py

PURPOSE:
    run all the scripts in order to generate figures 1–11, B1, and B2 in the paper
    as well as tables 2–4, B1, and B2

CODE FLOW (incl. outputs):
    1. stack_spectral_data.py
        - directly calls colorrev2.fix.fits
        * figs B1, B2
            - 'Composite_Spectra/StellarMassZ/MMT_stlrmassZ.pdf'
            - 'Composite_Spectra/StellarMassZ/Keck_stlrmassZ.pdf'
        * tabs B1, B2
            - 'Tables/B1.txt'
            - 'Tables/B2.txt'
    2. write_spectral_coverage_table.py
        - directly calls colorrev2.fix.fits
        - depends on file(s) from (1)
    3. mainseq_corrections.py
        - directly calls colorrev2.fix.fits
        - depends on file(s) from (1), (2)
            - (tabs 2,C1 depend on file from (3))
        - calls plot_NII_Ha_ratios.py (which also depends on file(s) from (1))
        * fig 1
            - 'Plots/main_sequence/NII_Ha_scatter_log.pdf'
        * tabs 2, C1
            - 'Tables/2.txt'
            - 'Tables/2_footnote.txt'
            - 'Tables/C1.txt'
    4. plot_mstar_vs_ebv.py
        - depends on file(s) from (1), (2), (3)
        * fig 2
            - 'Plots/main_sequence/mstar_vs_ebv.pdf'
    5. SED_fits.py
        - directly calls colorrev2.fix.fits
        * fig 3
            - 'Plots/SED_fits/allfilt.galex.PDF'
    6. plot_mainseq_UV_Ha_comparison.py
        - depends on file(s) from (3)
        * figs 4, 5
            - 'Plots/main_sequence/UV_Ha/SFR_ratio.pdf'
            - 'Plots/main_sequence/UV_Ha/SFR_ratio_dustcorr.pdf'
        * tab 3
            - 'Tables/3.txt'
    7. plot_nbia_mainseq.py
        - depends on file(s) from (3)
        * figs 6, 7, 10
            - 'Plots/main_sequence/mainseq_sSFRs_FUV_corrs.pdf'
            - 'Plots/main_sequence/mainseq.pdf'
            - 'Plots/main_sequence/zdep_mainseq.pdf'
    8. MC_contours.py
        - depends on file(s) from (3)
        * figs 8, 9
            - 'Plots/main_sequence/MC_regr_contours_noz.pdf'
            - 'Plots/main_sequence/MC_regr_contours.pdf'
    9. nbia_mainseq_dispersion.py
        - depends on file(s) from (3)
        * fig 11
            - 'Plots/main_sequence/mainseq_dispersion.pdf'
        * tab 4
            - 'Tables/4.txt'

INPUTS:


"""
from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table, Column, vstack

import numpy as np, numpy.ma as ma
import matplotlib.style
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.0

import config



def generate_ref_tab_helper(AP, inst_str0, gridap, stlr_mass, title, index_list):
    min_stlrmass_arr, max_stlrmass_arr = [],[]
    
    subplot_index=0
    for (match_index) in (index_list):
        if len(match_index) == 0:
            min_stlrmass_arr.append(0)
            max_stlrmass_arr.append(0)
            continue

        min_stlrmass_arr.append(np.min(stlr_mass[match_index]))
        max_stlrmass_arr.append(np.max(stlr_mass[match_index]))

    subtitle_list = np.array([title]*len(min_stlrmass_arr))

    table00 = Table([subtitle_list, min_stlrmass_arr, max_stlrmass_arr],
        names=['filter', 'min_stlrmass', 'max_stlrmass'])

    return table00


def generate_ref_tab(NAME0, inst_str0, inst_dict, stlr_mass, zspec0, AP):
    import stack_spectral_data

    griddata = asc.read(config.FULL_PATH+'Spectra/spectral_MMT_grid_data.txt',guess=False)
    gridz  = np.array(griddata['ZSPEC']) ##used
    gridap = np.array(griddata['AP']) ##used
    grid   = pyfits.open(config.FULL_PATH+'Spectra/spectral_MMT_grid.fits')
    grid_ndarr = grid[0].data ##used
    grid_hdr   = grid[0].header
    CRVAL1 = grid_hdr['CRVAL1']
    CDELT1 = grid_hdr['CDELT1']
    NAXIS1 = grid_hdr['NAXIS1'] #npix
    x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1) ##used
    # mask spectra that doesn't exist or lacks coverage in certain areas
    ndarr_zeros = np.where(grid_ndarr == 0)
    mask_ndarr = np.zeros_like(grid_ndarr)
    mask_ndarr[ndarr_zeros] = 1
    # mask spectra with unreliable redshift
    bad_zspec = [x for x in range(len(gridz)) if gridz[x] > 9 or gridz[x] < 0]
    mask_ndarr[bad_zspec,:] = 1
    grid_ndarr = ma.masked_array(grid_ndarr, mask=mask_ndarr, fill_value=np.nan)


    table00 = None

    mmt_ii = np.array([x for x in range(len(NAME0)) if 
        ('Ha-NB' in NAME0[x] and inst_str0[x] in inst_dict['MMT'] 
            and stlr_mass[x] > 0 and (zspec0[x] > 0 and zspec0[x] < 9))])
    mmt_ii = stack_spectral_data.exclude_AGN(mmt_ii, NAME0)
    bins_ii_tbl = np.ndarray((5,5), dtype=object)

    bins_ii_tbl_temp = np.ndarray((5,5), dtype=object)
    for ff, ii in zip(['NB7', 'NB816', 'NB921', 'NB973'], [0,1,2,3]):
        filt_ii = np.array([x for x in range(len(mmt_ii)) if 'Ha-'+ff in NAME0[mmt_ii][x]])
        filt_masses = stlr_mass[mmt_ii][filt_ii]
        for n in [5, 4, 3, 2]:
            bins_ii = stack_spectral_data.split_into_bins(filt_masses, n)
            if bins_ii != 'TOO SMALL': break
        for x in range(5 - len(bins_ii)):
            bins_ii.append([])
        bins_ii_tbl[ii] = bins_ii

        for jj in range(len(bins_ii)):
            bins_ii_tbl_temp[ii][jj] = mmt_ii[filt_ii][bins_ii_tbl[ii][jj]]

        if ff=='NB7':
            title='NB704+NB711'
        else:
            title=ff
        print('>>>', title)

        table_data = generate_ref_tab_helper(AP=AP, inst_str0=inst_str0, gridap=gridap,
            stlr_mass=stlr_mass, title=title, index_list=bins_ii_tbl_temp[ii])
        if table00 == None:
            table00 = table_data
        else:
            table00 = vstack([table00, table_data])
        #endif
    #endfor

    return table00


def call_plot_MMT_stlrmass_z(data_dict, NAME0, AP, inst_str0, inst_dict, stlr_mass, zspec0, tol, cvg_ref):
    '''part of step 1'''
    print('### looking at the MMT grid')
    import stack_spectral_data

    griddata = asc.read(config.FULL_PATH+'Spectra/spectral_MMT_grid_data.txt',guess=False)
    gridz  = np.array(griddata['ZSPEC']) ##used
    gridap = np.array(griddata['AP']) ##used
    grid   = pyfits.open(config.FULL_PATH+'Spectra/spectral_MMT_grid.fits')
    grid_ndarr = grid[0].data ##used
    grid_hdr   = grid[0].header
    CRVAL1 = grid_hdr['CRVAL1']
    CDELT1 = grid_hdr['CDELT1']
    NAXIS1 = grid_hdr['NAXIS1'] #npix
    x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1) ##used
    # mask spectra that doesn't exist or lacks coverage in certain areas
    ndarr_zeros = np.where(grid_ndarr == 0)
    mask_ndarr = np.zeros_like(grid_ndarr)
    mask_ndarr[ndarr_zeros] = 1
    # mask spectra with unreliable redshift
    bad_zspec = [x for x in range(len(gridz)) if gridz[x] > 9 or gridz[x] < 0]
    mask_ndarr[bad_zspec,:] = 1
    grid_ndarr = ma.masked_array(grid_ndarr, mask=mask_ndarr, fill_value=np.nan)

    stack_spectral_data.plot_MMT_stlrmass_z(data_dict, NAME0, AP, inst_str0, inst_dict, stlr_mass, zspec0,
        gridap, grid_ndarr, gridz, x0, tol, cvg_ref)


def call_plot_Keck_stlrmass_z(data_dict, NAME0, AP, inst_str0, inst_dict, stlr_mass, zspec0, tol):
    '''part of step 1'''
    print('### looking at the Keck grid')
    import stack_spectral_data

    griddata = asc.read(config.FULL_PATH+'Spectra/spectral_Keck_grid_data.txt',guess=False)
    gridz  = np.array(griddata['ZSPEC']) ##used
    gridap = np.array(griddata['AP']) ##used
    grid   = pyfits.open(config.FULL_PATH+'Spectra/spectral_Keck_grid.fits')
    grid_ndarr = grid[0].data ##used
    grid_hdr   = grid[0].header
    CRVAL1 = grid_hdr['CRVAL1']
    CDELT1 = grid_hdr['CDELT1']
    NAXIS1 = grid_hdr['NAXIS1'] #npix
    x0 = np.arange(CRVAL1, CDELT1*NAXIS1+CRVAL1, CDELT1) ##used
    # mask spectra that doesn't exist or lacks coverage in certain areas
    ndarr_zeros = np.where(grid_ndarr == 0)
    mask_ndarr = np.zeros_like(grid_ndarr)
    mask_ndarr[ndarr_zeros] = 1
    # mask spectra with unreliable redshift
    bad_zspec = [x for x in range(len(gridz)) if gridz[x] > 9 or gridz[x] < 0]
    mask_ndarr[bad_zspec,:] = 1
    grid_ndarr = ma.masked_array(grid_ndarr, mask=mask_ndarr)

    stack_spectral_data.plot_Keck_stlrmass_z(data_dict, NAME0, AP, inst_str0, inst_dict, stlr_mass, zspec0,
        gridap, grid_ndarr, gridz, x0, tol)


def tab_B1():
    mmtmz = asc.read(config.FULL_PATH+'Composite_Spectra/StellarMassZ/MMT_stlrmassZ_data.txt',
        format='fixed_width_two_line', delimiter=' ')
    g=np.array([x for x in range(len(mmtmz)) if mmtmz['stlrmass_bin'][x] != 'N/A'])
    temparr = np.array([x.split('-') for x in mmtmz[g]['stlrmass_bin']])
    zz = np.vstack((np.array([x+'0' if len(x)==3 else x for x in temparr[:,0]]),np.array([x+'0' if len(x)==3 else x for x in temparr[:,1]]))).T
    stlrmass = np.array(['--'.join(y) for y in zz])
    zrange = []
    for i in range(len(g)):
        minz = '%.3f'%mmtmz[g][i]['minz']
        maxz = '%.3f'%mmtmz[g][i]['maxz']
        zrange.append(str(minz)+'-'+str(maxz))

    flux_hb = np.array(['%.2f'%x for x in mmtmz[g]['HB_flux']/1e-18])
    flux_hb_errs_pos = np.array(['%.2f'%x for x in mmtmz[g]['HB_flux_errs_pos']/1e-18])
    flux_hb_errs_neg = np.array(['%.2f'%x for x in mmtmz[g]['HB_flux_errs_neg']/1e-18])
    flux_hb_col = []
    for x, y, z in zip(flux_hb, flux_hb_errs_neg, flux_hb_errs_pos):
        flux_hb_col.append(x+'$_{-'+y+'}^{+'+z+'}$')

    flux_hg__flux_hb = np.array(['%.2f'%x for x in mmtmz[g]['HG_flux']/mmtmz[g]['HB_flux']])
    flux_hg__flux_hb_errs_pos = np.array(['%.2f'%x for x in mmtmz[g]['FLUX_hghb_errs_pos']])
    flux_hg__flux_hb_errs_neg = np.array(['%.2f'%x for x in mmtmz[g]['FLUX_hghb_errs_neg']])
    flux_hg__flux_hb_col = []
    for x, y, z in zip(flux_hg__flux_hb, flux_hg__flux_hb_errs_neg, flux_hg__flux_hb_errs_pos):
        flux_hg__flux_hb_col.append(x+'$_{-'+y+'}^{+'+z+'}$')

    flux_ha__flux_hb = np.array(['%.2f'%(mmtmz[g][x]['HA_flux']/mmtmz[g][x]['HB_NB921_flux']) if mmtmz[g][x]['HA_flux'] > 0 else '\ldots' for x in range(len(g))])
    flux_ha__flux_hb_errs_pos = np.array(['%.2f'%x for x in mmtmz[g]['FLUX_hahb_errs_pos']])
    flux_ha__flux_hb_errs_neg = np.array(['%.2f'%x for x in mmtmz[g]['FLUX_hahb_errs_neg']])
    flux_ha__flux_hb_col = []
    for x, y, z in zip(flux_ha__flux_hb, flux_ha__flux_hb_errs_neg, flux_ha__flux_hb_errs_pos):
        if x=='\\ldots':
            flux_ha__flux_hb_col.append(x)
        else:
            flux_ha__flux_hb_col.append(x+'$_{-'+y+'}^{+'+z+'}$')

    nb921 = np.where((mmtmz[g]['filter']=='NB921') & (mmtmz[g]['HB_NB921_flux'] > 0))[0]
    flux_ha__flux_hb[nb921] = np.array(['%.2f'%x for x in mmtmz[g][nb921]['HA_flux']/mmtmz[g][nb921]['HB_NB921_flux']])
    flux_nii__flux_ha = np.array(['\ldots' if x==0 else '%.2f'%x for x in np.abs((1+2.96)/2.96 * mmtmz[g]['NII_6583_flux'] / mmtmz[g]['HA_flux'])])
    ebv_hghb = np.array(['%.3f'%np.abs(x) for x in mmtmz[g]['E(B-V)_hghb']])
    ebv_hghb_errs_pos = np.array(['%.3f'%np.abs(x) for x in mmtmz[g]['E(B-V)_hghb_errs_pos']])
    ebv_hghb_errs_neg = np.array(['%.3f'%np.abs(x) for x in mmtmz[g]['E(B-V)_hghb_errs_neg']])
    ebv_hahb = np.array(['%.3f'%x if x>=0 else '\ldots' for x in mmtmz[g]['E(B-V)_hahb']])
    ebv_hahb_errs_pos = np.array(['%.3f'%np.abs(x) for x in mmtmz[g]['E(B-V)_hahb_errs_pos']])
    ebv_hahb_errs_neg = np.array(['%.3f'%np.abs(x) for x in mmtmz[g]['E(B-V)_hahb_errs_neg']])

    ebv_hghb_col = []
    for x, y, z in zip(ebv_hghb, ebv_hghb_errs_neg, ebv_hghb_errs_pos):
        ebv_hghb_col.append(x+'$_{-'+y+'}^{+'+z+'}$')
    ebv_hahb_col = []
    for x, y, z in zip(ebv_hahb, ebv_hahb_errs_neg, ebv_hahb_errs_pos):
        if y=='nan':
            ebv_hahb_col.append(x)
        else:
            ebv_hahb_col.append(x+'$_{-'+y+'}^{+'+z+'}$')

    tt = Table([stlrmass, mmtmz[g]['num_stack_HG'], mmtmz[g]['num_stack_HB'], mmtmz[g]['num_stack_HA'], flux_hb_col, flux_hg__flux_hb_col, flux_ha__flux_hb_col, ebv_hghb_col, ebv_hahb_col, flux_nii__flux_ha], names=['(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)','(9)','(10)'])
    asc.write(tt, config.FULL_PATH+'Tables/B1.txt', format='latex', overwrite=True)


def tab_B2():
    keckmz = asc.read(config.FULL_PATH+'Composite_Spectra/StellarMassZ/Keck_stlrmassZ_data.txt', format='fixed_width_two_line', delimiter=' ')
    temparr = np.array([x.split('-') for x in keckmz['stlrmass_bin']])
    zz = np.vstack((np.array([x+'0' if len(x)==3 else x for x in temparr[:,0]]),np.array([x+'0' if len(x)==3 else x for x in temparr[:,1]]))).T
    stlrmass = np.array(['--'.join(y) for y in zz])
    zrange = []
    for i in range(len(keckmz)):
        minz = '%.3f'%keckmz[i]['minz']
        maxz = '%.3f'%keckmz[i]['maxz']
        zrange.append(str(minz)+'-'+str(maxz))

    flux_hb = np.array(['%.2f'%x for x in keckmz['HB_flux']/1e-18])
    flux_hb_errs_pos = np.array(['%.2f'%x for x in keckmz['HB_flux_errs_pos']/1e-18])
    flux_hb_errs_neg = np.array(['%.2f'%x for x in keckmz['HB_flux_errs_neg']/1e-18])
    flux_hb_col = []
    for x, y, z in zip(flux_hb, flux_hb_errs_neg, flux_hb_errs_pos):
        flux_hb_col.append(x+'$_{-'+y+'}^{+'+z+'}$')

    flux_ha__flux_hb = np.array(['%.2f'%x for x in keckmz['HA_flux']/keckmz['HB_flux']])
    flux_ha__flux_hb_errs_pos = np.array(['%.2f'%x for x in keckmz['FLUX_hahb_errs_pos']])
    flux_ha__flux_hb_errs_neg = np.array(['%.2f'%x for x in keckmz['FLUX_hahb_errs_neg']])
    flux_ha__flux_hb_col = []
    for x, y, z in zip(flux_ha__flux_hb, flux_ha__flux_hb_errs_neg, flux_ha__flux_hb_errs_pos):
        flux_ha__flux_hb_col.append(x+'$_{-'+y+'}^{+'+z+'}$')

    flux_nii__flux_ha = np.array(['%.2f'%x for x in np.abs((1+2.96)/2.96 * keckmz['NII_6583_flux'] / keckmz['HA_flux'])])

    ebv_hahb = np.array(['%.3f'%np.abs(x) for x in keckmz['E(B-V)_hahb']])
    ebv_hahb_errs_pos = np.array(['%.3f'%np.abs(x) for x in keckmz['E(B-V)_hahb_errs_pos']])
    ebv_hahb_errs_neg = np.array(['%.3f'%np.abs(x) for x in keckmz['E(B-V)_hahb_errs_neg']])
    ebv_hahb_col = []
    for x, y, z in zip(ebv_hahb, ebv_hahb_errs_neg, ebv_hahb_errs_pos):
        ebv_hahb_col.append(x+'$_{-'+y+'}^{+'+z+'}$')

    tt = Table([stlrmass, keckmz['num_stack_HB'], keckmz['num_stack_HA'], flux_hb_col, flux_ha__flux_hb_col, ebv_hahb_col, flux_nii__flux_ha], names=['(1)','(2)','(3)','(4)','(5)','(6)','(7)'])
    asc.write(tt, config.FULL_PATH+'Tables/B2.txt', format='latex', overwrite=True)


def tab_2():
    mainseq = asc.read(config.FULL_PATH+config.mainseq_corrs_tbl,
        format='fixed_width_two_line', delimiter=' ')
    col1 = ['NB704', 'NB711', 'NB816', 'NB921', 'NB973']
    col2,col3,col4,col5,col6,col7,col8,col9,col10 = [],[],[],[],[],[],[],[],[]
    for ff in col1: 
        col2.append(len([x for x in range(len(mainseq)) if mainseq['filt'][x]==ff]))
        col3.append(len([x for x in range(len(mainseq)) if mainseq['filt'][x]==ff and ('MMT' in mainseq['inst_str0'][x] or 'merged' in mainseq['inst_str0'][x] or 'Keck' in mainseq['inst_str0'][x])]))
        col4.append(len([x for x in range(len(mainseq)) if mainseq['filt'][x]==ff and ('MMT' in mainseq['inst_str0'][x] or 'merged' in mainseq['inst_str0'][x] or 'Keck' in mainseq['inst_str0'][x]) and mainseq['zspec0'][x]>0 and mainseq['zspec0'][x]<9]))
        col5.append(len([x for x in range(len(mainseq)) if mainseq['filt'][x]==ff and ('MMT' in mainseq['inst_str0'][x] or 'merged' in mainseq['inst_str0'][x])]))
        col6.append(len([x for x in range(len(mainseq)) if mainseq['filt'][x]==ff and ('MMT' in mainseq['inst_str0'][x] or 'merged' in mainseq['inst_str0'][x]) and mainseq['zspec0'][x]>0 and mainseq['zspec0'][x]<9]))
        ztemp = [mainseq['zspec0'][x] for x in range(len(mainseq)) if mainseq['filt'][x]==ff and ('MMT' in mainseq['inst_str0'][x] or 'merged' in mainseq['inst_str0'][x]) and mainseq['zspec0'][x]>0 and mainseq['zspec0'][x]<9]
        tmpstr = str(np.round(min(ztemp),3))+'--'+str(np.round(max(ztemp),3))+' &'
        col7.append(tmpstr)
        
        if 'NB9' in ff:
            col8.append(len([x for x in range(len(mainseq)) if mainseq['filt'][x]==ff and ('Keck' in mainseq['inst_str0'][x] or 'merged' in mainseq['inst_str0'][x])]))
            col9.append(len([x for x in range(len(mainseq)) if mainseq['filt'][x]==ff and ('Keck' in mainseq['inst_str0'][x] or 'merged' in mainseq['inst_str0'][x]) and mainseq['zspec0'][x]>0 and mainseq['zspec0'][x]<9]))
            ztemp = [mainseq['zspec0'][x] for x in range(len(mainseq)) if mainseq['filt'][x]==ff and ('Keck' in mainseq['inst_str0'][x] or 'merged' in mainseq['inst_str0'][x]) and mainseq['zspec0'][x]>0 and mainseq['zspec0'][x]<9]
            tmpstr = str(np.round(min(ztemp),3))+'--'+str(np.round(max(ztemp),3))
            col10.append(tmpstr)
        else:
            col8.append('\ldots')
            col9.append('\ldots')
            col10.append('\ldots')


    tt = Table([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10], 
               names=['(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)','(9)','(10)'])
    asc.write(tt, config.FULL_PATH+'Tables/2.txt', format='latex', overwrite=True)

    # table 2 footnote
    cvg = asc.read(config.FULL_PATH+'Composite_Spectra/MMT_spectral_coverage.txt')
    num1 = len([x for x in range(len(cvg)) if cvg['filter'][x]=='NB921' and (cvg['HA_cvg'][x]=='NO' or cvg['HA_cvg'][x]=='MASK')])
    num2 = len([x for x in range(len(cvg)) if cvg['filter'][x]=='NB921'])
    
    ff = open(config.FULL_PATH+'Tables/2_footnote.txt', 'w')
    ff.write(f'{num1} of {num2} Ha NB921 emitters had their MMT Ha measurements excluded')
    ff.close()


def tab_C1():
    from MACT_utils import get_FUV_corrs

    corr_tbl = asc.read(config.FULL_PATH+config.mainseq_corrs_tbl,
        guess=False, Reader=asc.FixedWidthTwoLine)
    good_sig_iis = np.where((corr_tbl['flux_sigma'] >= config.CUTOFF_SIGMA) 
        & (corr_tbl['stlr_mass'] >= config.CUTOFF_MASS))[0]
    corr_tbl = corr_tbl[good_sig_iis]
    ffs = corr_tbl['filt'].data
    smass0 = corr_tbl['stlr_mass'].data
    sfr = corr_tbl['met_dep_sfr'].data
    dust_corr_factor = corr_tbl['dust_corr_factor'].data
    filt_corr_factor = corr_tbl['filt_corr_factor'].data
    nii_ha_corr_factor = corr_tbl['nii_ha_corr_factor'].data
    sfrs00 = sfr+filt_corr_factor+nii_ha_corr_factor+dust_corr_factor
    FUV_corr_factor = get_FUV_corrs(corr_tbl)
    mbins0 = np.arange(6.25, 10.75, .5)

    col1 = ['%.1f'%(m-0.25)+'--'+'%.1f'%(m+0.25) for m in mbins0]
    tt = Table([col1])
    names_arr = [['(2)','(3)','(4)'],['(5)','(6)','(7)'],['(8)','(9)','(10)'],['(11)','(12)','(13)']]
    for idx, ff in enumerate(['NB7', 'NB816', 'NB921', 'NB973']):
        colN_tmp = np.array(['\\ldots']*len(mbins0))
        colM_tmp = np.array(['\\ldots']*len(mbins0))
        # colSFR_tmp = np.char.decode(np.array(['\\ldots']*len(mbins0), dtype='S20'))
        # colSFR_tmp = colSFR_tmp.astype('S20')
        colSFR_tmp = np.array(['\\ldots']*len(mbins0), dtype='S20')

        filt_match = np.array([x for x in range(len(ffs)) if ff in ffs[x]])
        
        mass = smass0[filt_match]
        sfrs = sfrs00[filt_match]
        sfrs_with_fuv = sfrs + FUV_corr_factor[filt_match]
        
        bin_ii = np.digitize(mass, mbins0+0.25)
        for i in set(bin_ii):
            bin_match = np.where(bin_ii == i)[0]

            colN_tmp[i] = len(bin_match)
            colM_tmp[i] = '%.2f'%(np.mean(mass[bin_match]))

            if np.mean(sfrs_with_fuv[bin_match]) < 0:
                colSFR_tmp[i] = b'-%.2f'%(np.mean(sfrs_with_fuv[bin_match]))
                # colSFR_tmp[i] = f'-{np.mean(sfrs_with_fuv[bin_match]):.2f}'
            else:
                colSFR_tmp[i] = b'+%.2f'%(np.mean(sfrs_with_fuv[bin_match]))
                # colSFR_tmp[i] = f'{np.mean(sfrs_with_fuv[bin_match]):.2f}'

            if np.mean(sfrs[bin_match]) < 0:
                colSFR_tmp[i] += b' (-%.2f)'%(np.mean(sfrs[bin_match]))
                # colSFR_tmp[i] += f' (-{np.mean(sfrs[bin_match]):.2f})'
            else:
                colSFR_tmp[i] += b' (+%.2f)'%(np.mean(sfrs[bin_match]))
                # colSFR_tmp[i] += f' ({np.mean(sfrs[bin_match]):.2f})'


        if ff != 'NB973':
            colSFR_tmp = np.array([x+b' &' for x in colSFR_tmp])

        tt.add_columns([Column(data=colN_tmp), 
                        Column(data=colM_tmp), 
                        Column(data=colSFR_tmp)])

    asc.write(tt, config.FULL_PATH+'Tables/C1.txt', format='latex', overwrite=True)


def section_3_2_numbers(corr_tbl):
    mmt_cvg = asc.read(config.FULL_PATH+'Composite_Spectra/MMT_spectral_coverage.txt')
    keck_cvg = asc.read(config.FULL_PATH+'Composite_Spectra/Keck_spectral_coverage.txt')

    print('\n\nThere are...')
    len_nonb816 = len(keck_cvg) - len(np.where(keck_cvg['filter']=='NB816')[0])
    len_joint = len([x for x in keck_cvg['NAME'].data if x in mmt_cvg['NAME'].data and 'Ha-NB816' not in x])
    print(f'\t{len(mmt_cvg)} MMT galaxies with spec_z')
    print(f'\t{len_nonb816} Keck galaxies with spec_z (no NB816 included)')
    print(f'\t{len_joint} joint galaxies')

    print('\t---')

    len_mmt_yeshg = len([x for x in range(len(mmt_cvg)) if mmt_cvg['HG_cvg'][x]=='YES'])
    len_keck_yeshb = len([x for x in range(len(keck_cvg)) if keck_cvg['HB_cvg'][x]=='YES' and 'NB816' not in keck_cvg['NAME'][x]])
    overlap_mmt_check = len([x for x in range(len(mmt_cvg)) if ((mmt_cvg['NAME'][x] in keck_cvg['NAME'].data)
        and ('Ha-NB816' not in mmt_cvg['NAME'][x]) and (mmt_cvg['HG_cvg'][x]=='YES'))])
    overlap_keck_check = len([x for x in range(len(keck_cvg)) if ((keck_cvg['NAME'][x] in mmt_cvg['NAME'].data)
        and ('Ha-NB816' not in keck_cvg['NAME'][x]) and (keck_cvg['HB_cvg'][x]=='YES'))])
    print(f'\t{len_mmt_yeshg} MMT galaxies w/ Hg cvg')
    print(f'\t{len_keck_yeshb} Keck galaxies w/ Hb cvg')
    print(f'\t{min(overlap_keck_check, overlap_mmt_check)} joint galaxies w/ good cvg')

    ha_ii = np.array(corr_tbl['ID'])-1
    zspec0 = corr_tbl['zspec0'].data
    yes_spectra = np.where((zspec0 >= 0) & (zspec0 < 9))[0]

    data_dict = config.data_dict
    HA_FLUX   = data_dict['HA_FLUX'][ha_ii]
    HB_FLUX   = data_dict['HB_FLUX'][ha_ii]
    HA_SNR    = data_dict['HA_SNR'][ha_ii]
    HB_SNR    = data_dict['HB_SNR'][ha_ii]
    # getting indices where the valid-redshift (yes_spectra) data has appropriate HB SNR as well as valid HA_FLUX
    gooddata_iis = np.where((HB_SNR[yes_spectra] >= 5) & (HA_SNR[yes_spectra] > 0) & (HA_FLUX[yes_spectra] > 1e-20) & (HA_FLUX[yes_spectra] < 99))[0]
    print(f'\n{len(gooddata_iis)} out of {len(corr_tbl)} Ha emitting galaxies have reliable enough emission line measurements.')


def run_stack_spectral_data(inst_dict, nbiadata, zspec, fout, data_dict):
    '''
    calls functions from stack_spectral_data.py
    '''
    # defined variables
    tol = 3 #in angstroms, used for NII emission flux calculations

    # arrays from read-in files
    NAME0 = nbiadata['NAME']
    zspec0 = zspec['zspec0'].data
    inst_str0 = zspec['inst_str0'].data
    stlr_mass = fout['col7'].data
    nan_stlr_mass = np.copy(stlr_mass)
    nan_stlr_mass[nan_stlr_mass < 0] = np.nan
    AP = data_dict['AP']

    # generating hb_nb921 reference table
    cvg_ref = generate_ref_tab(NAME0, inst_str0, inst_dict, stlr_mass, zspec0, AP)

    # generates figure B1
    print('\n generating fig B1...')
    call_plot_MMT_stlrmass_z(data_dict, NAME0, AP, inst_str0, inst_dict, stlr_mass, zspec0, tol, cvg_ref)
    
    # generates figure B2
    print('\n generating fig B2...')
    call_plot_Keck_stlrmass_z(data_dict, NAME0, AP, inst_str0, inst_dict, stlr_mass, zspec0, tol)

    # generates table B1
    print('\n generating tab B1...')
    tab_B1()

    # generates table B2
    print('\n generating tab B2...')
    tab_B2()


def run_write_spectral_coverage_table():
    import write_spectral_coverage_table

    write_spectral_coverage_table.main()


def run_mainseq_corrections():
    import mainseq_corrections

    print('\n generating fig 1...')
    mainseq_corrections.main()

    # generates table 2
    print('\n generating tab 2...')
    tab_2()

    # generates table C1
    print('\n generating tab C1...')
    tab_C1()


def run_plot_mstar_vs_ebv():
    import plot_mstar_vs_ebv

    print('\n generating fig 2...')
    plot_mstar_vs_ebv.main()


def run_SED_fits():
    import SED_fits

    print('\n generating fig 3...')
    SED_fits.main()


def run_plot_mainseq_UV_Ha_comparison():
    import plot_mainseq_UV_Ha_comparison

    print('\n generating figs 4 and 5 and tab 3...')
    plot_mainseq_UV_Ha_comparison.main()


def run_plot_nbia_mainseq():
    import plot_nbia_mainseq

    print('\n generating figs 6, 7, and 10...')
    plot_nbia_mainseq.main()


def run_MC_contours():
    import MC_contours

    print('\n generating figs 8 and 9...')
    MC_contours.main()


def run_nbia_mainseq_dispersion():
    import nbia_mainseq_dispersion

    print('\n generating fig 11 and tab 4...')
    nbia_mainseq_dispersion.main()


def main():
    # READ-IN FILES, DEFINED DATA STRUCTS
    nbia = pyfits.open(config.FULL_PATH+config.NB_IA_emitters_cat)
    nbiadata = nbia[1].data
    zspec = asc.read(config.FULL_PATH+config.zspec_cat, guess=False, Reader=asc.CommentedHeader)
    fout  = asc.read(config.FULL_PATH+config.fout_cat, guess=False, Reader=asc.NoHeader)


    ## some number checks
    num_emgal = len(np.array([x for x in range(len(nbiadata['NAME'])) if 'NB' in nbiadata['NAME'][x]]))
    num_ha = len(np.array([x for x in range(len(nbiadata['NAME'])) if 'Ha-NB' in nbiadata['NAME'][x]]))
    print(f"\nFrom the sample of {num_emgal} emission-line galaxies, \n\
        {num_ha} of them were identified as Ha emitting galaxies in the NB filters.")
    print("(However, 9 of those Ha emitting galaxies have been excluded from the analysis.)\n")


    # calling stack_spectral_data.py
    run_stack_spectral_data(config.inst_dict, nbiadata, zspec, fout, config.data_dict)

    # calling write_spectral_coverage_table.py
    run_write_spectral_coverage_table()

    # calling mainseq_corrections.py
    run_mainseq_corrections()

    ## some more number checks
    corr_tbl = asc.read(config.FULL_PATH+config.mainseq_corrs_tbl,
        guess=False, Reader=asc.FixedWidthTwoLine)
    section_3_2_numbers(corr_tbl)
    good_sig_iis = np.where((corr_tbl['flux_sigma'] >= config.CUTOFF_SIGMA)
        & (corr_tbl['stlr_mass'] >= config.CUTOFF_MASS))[0]
    print(f"\nApplying the flux and mass cutoff, \n\
        from the sample of {len(corr_tbl)} Ha emitting galaxies, we now have {len(good_sig_iis)} galaxies.\n")

    # calling plot_mstar_vs_ebv.py
    run_plot_mstar_vs_ebv()

    # calling SED_fits.py
    run_SED_fits()

    # calling plot_mainseq_UV_Ha_comparison.py
    run_plot_mainseq_UV_Ha_comparison()

    # calling plot_nbia_mainseq.py
    run_plot_nbia_mainseq()

    # calling MC_contours.py
    run_MC_contours()

    # calling nbia_mainseq_dispersion.py
    run_nbia_mainseq_dispersion()

    print('\ndone!\n')


if __name__ == '__main__':
    main()
