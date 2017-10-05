"""
NAME:
    combine_spectral_data.py
      (previously combine_spectral_MMT_data.py, combine_spectral_Keck_data.py)

PURPOSE:
    This code combines all the raw spectral data such that they all have the
    same spacing and range (as grid_col) and writes them to a .fits file.
    Does this for both MMT data and Keck data.

INPUTS:
    'Spectra/MMT/Single_Data/f6_mmtlist*fits'
    'Spectra/MMT/Single_Data/f6_2014*fits'
    'Spectra/MMT/Combine_Data/f6_MMT_combine_spec.sigma.fits'
    'Spectra/MMT/Single_Data/MMT_2008_targets.0*fits'
    'Spectra/MMT/Single_Data/MMT_2014*fits'
    'Spectra/MMT/Combine_Data/f6_MMT_combine_spec.sigma.redshift.all.fits'
    'Spectra/Keck/Single_Data/DEIMOS*f2.fits'
    'Spectra/Keck/Single_Data/DEIMOS*f3.fits'
    'Spectra/Keck/Single_Data/DEIMOS*f4.fits'
    'Spectra/Keck/Combine_Data/f3_DEIMOS_combine_spec.sigma.fits'
    'Spectra/Keck/Single_Data/DEIMOS*ID.fits'
    'Spectra/Keck/Combine_Data/f3_DEIMOS_combine_spec.sigma.redshift.all.fits'

CALLING SEQUENCE:
    Reads relevant inputs and creates two arrays, vers_spec_list_all (from the
    image fits files) and vers_table_list_all (from the binary hdu fits files).
    Then iterates through each source (in proper order) and cuts out the 'bad'
    parts of the images ('badnum', counted manually from the images
    themselves). Relevant, 'good' data is then added to arrays and dicts.
    At the end of the iteration, averages cdelt_all to get cdelt_avg, creates
    a col (from x0min_all to x0max_all, increments of cdelt_avg), and also
    creates a ndarr originally filled with zeros.
    At this point,  there is another iteration, this time going through all
    the keys and values in type_len_dict and img_data_dict (2d for-loop).
    Using the data, creates an interpolation function. With the interpolated
    data, fills the appropriate indexes of grid_ndarr with the same column
    'spacing' as in grid_col. Then after the iteration, saves grid_ndarr,
    cdelt_avg, and crval1 to a fits file. Also saves AP_all, ZSPEC_all to
    a .txt file.
    Does this for both MMT and Keck data.

OUTPUTS:
    'Spectra/spectral_MMT_grid.fits'
    'Spectra/spectral_MMT_grid_data.txt'
    'Spectra/spectral_Keck_grid.fits'
    'Spectra/spectral_Keck_grid_data.txt'

REVISION HISTORY:
    Created by Kaitlyn Shin 23 July 2016
"""

import numpy as np, matplotlib.pyplot as plt, glob, copy
from astropy.io import fits as pyfits, ascii as asc
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from scipy.interpolate import interp1d

def fix_AP(aa):
	empty_str_ii = np.array([x for x in range(len(aa)) if aa[x] == '      '])
	if len(empty_str_ii) > 0:
		for j in range(len(empty_str_ii)):
			aa[empty_str_ii[j]] = '0'+str(np.float(aa[empty_str_ii[j]-1])+0.001)
			if len(aa[empty_str_ii[j]])==5:
				aa[empty_str_ii[j]] += '0'
	return aa


full_path = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'
# reading in MMT data
mmt_spec_list_1_4 = glob.glob(full_path+'Spectra/MMT/Single_Data/f6_mmtlist*fits')
mmt_spec_list_A_H = glob.glob(full_path+'Spectra/MMT/Single_Data/f6_2014*fits')
mmt_spec_list_combined = glob.glob(full_path+'Spectra/MMT/Combine_Data/f6_MMT_combine_spec.sigma.fits')
mmt_spec_list_all = (mmt_spec_list_1_4 + mmt_spec_list_A_H +
                     mmt_spec_list_combined)

mmt_table_list_1_4_unordered = glob.glob(full_path+'Spectra/MMT/Single_Data/MMT_2008_targets.0*fits')
mmt_table_list_1_4 = [mmt_table_list_1_4_unordered[3],
                      mmt_table_list_1_4_unordered[0],
                      mmt_table_list_1_4_unordered[1],
                      mmt_table_list_1_4_unordered[2]]
mmt_table_list_A_H = glob.glob(full_path+'Spectra/MMT/Single_Data/MMT_2014*fits')
mmt_table_list_combined = glob.glob(full_path+'Spectra/MMT/Combine_Data/f6_MMT_combine_spec.sigma.redshift.all.fits')
mmt_table_list_all = (mmt_table_list_1_4 + mmt_table_list_A_H +
                      mmt_table_list_combined)


# reading in Keck data
keck_spec_list_2 = glob.glob(full_path+'Spectra/Keck/Single_Data/DEIMOS*f2.fits')
keck_spec_list_3 = glob.glob(full_path+'Spectra/Keck/Single_Data/DEIMOS*f3.fits')
keck_spec_list_4 = glob.glob(full_path+'Spectra/Keck/Single_Data/DEIMOS*f4.fits')
keck_spec_list_2_4 = (keck_spec_list_2 + keck_spec_list_3 + keck_spec_list_4)
keck_spec_list_2_4.sort()
keck_spec_list_combined = glob.glob(full_path+'Spectra/Keck/Combine_Data/f3_DEIMOS_combine_spec.sigma.fits')
keck_spec_list_all = (keck_spec_list_combined + keck_spec_list_2_4)

keck_table_list = glob.glob(full_path+'Spectra/Keck/Single_Data/DEIMOS*ID.fits')
keck_table_list_combined = glob.glob(full_path+'Spectra/Keck/Combine_Data/f3_DEIMOS_combine_spec.sigma.redshift.all.fits')
keck_table_list_all = (keck_table_list_combined + keck_table_list)
# end reading in data


speclist  = {'MMT': mmt_spec_list_all, 'Keck': keck_spec_list_all}
tablelist = {'MMT': mmt_table_list_all, 'Keck': keck_table_list_all}
vlist = {'MMT': ['1','2','3','4','A','B','C','D','E','F','G','H','S'],
         'Keck': ['00','01','02','04','06','07','10','11','22','24','25','27','28',
              '30','31','32','33','34','35','36','37','38','39']}
badnumlist = {'MMT': [0,0,0,0,82,40,40,82,82,82,82,82,0],
              'Keck': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}

for version in ['MMT','Keck']:
    AP_all = np.array([], dtype='|S6')
    ZSPEC_all = np.array([], dtype=np.float32)
    cdelt_all = np.array([], dtype=np.float32)
    x0min_all = np.array([], dtype=np.float32)
    x0max_all = np.array([], dtype=np.float32)
    img_data_dict = {}
    type_len_dict = {}
    x0_dict = {}
    for (spec, table, v, badnum) in zip(speclist[version], tablelist[version],
                                        vlist[version], badnumlist[version]):
        spec_data = pyfits.open(spec)
        crval1 = spec_data[0].header['CRVAL1']
        cdelt1 = spec_data[0].header['CDELT1']
        cdelt_all = np.append(cdelt_all, cdelt1)
        npix = spec_data[0].header['NAXIS1']
        x0 = np.arange(crval1, npix*cdelt1 + crval1, cdelt1)
        x0_dict[v] = x0
        x0min_all = np.append(x0min_all, min(x0))
        x0max_all = np.append(x0max_all, max(x0))
        image_data = spec_data[0].data
        img_data_dict['img_'+v] = image_data
        image_data[:,0:badnum]=0.0

        if version=='MMT':
            # masking the right-most indices of continuously negative spectra
            for row in image_data:
                num_red_neg_ii = 0
                index_marker = 0
                for ii in range(len(row)-1,-1,-1):
                    if ii == len(row) - 1:
                        index_marker = ii
                    if row[ii] < 0:
                        if index_marker != ii: break
                        row[ii] = 0
                        index_marker -= 1
                        num_red_neg_ii += 1
                    #endif
                #endfor
            #endfor

            # masking the right-most indices of continuously abnormally positive spectra
            tempdata = copy.deepcopy(image_data)
            temp2 = copy.deepcopy(image_data)
            idx = np.where((x0 > 6000) & (x0 < 6650))[0]
            for row, ii in zip(tempdata, range(len(tempdata))):
                tempmed = sigma_clipped_stats(row[idx])[1]
                row/=tempmed
            #endfor
            col_nanmean = np.nanmean(tempdata, axis=0)
            bad_init = np.where((np.abs(col_nanmean) >= 1.5) | (np.isnan((col_nanmean)) == True))[0]
            if len(bad_init) > 0:
                for ii in range(len(bad_init)-1, -1, -1):
                    if (bad_init[ii] - bad_init[ii-1]) > 1:
                        break
                #endfor
                bad_init = bad_init[ii:]
                image_data[:,bad_init] = 0.0

                # special cases
                if v == 'S':
                    image_data[59] = temp2[59] # 'S.060'
                    image_data[272] = temp2[272] # 'S.273'
                    image_data[284] = temp2[284] # 'S.285'
                    image_data[312] = temp2[312] # 'S.313'
                #endif
            #endif
        #endif

        
        table0 = pyfits.open(table)
        table_data = table0[1].data
        if version=='MMT':
            if 'targets.0329.fits' in table:
                table_data = table_data[:300]
            #endif
            line = table_data['LINE']
            all_index = np.arange(len(line))
            bad_index = np.where(line=='---')[0]
            good_index = np.delete(all_index, bad_index)
            AP = 'AP'
        elif version=='Keck':
            good_index = np.arange(len(table_data['ID']))
            AP = 'SLIT'
        #endif
        AP0 = np.array(table_data[AP][good_index])
        AP = fix_AP(AP0)

        ZSPEC = np.array(table_data['ZSPEC'][good_index])
        
        AP_all = np.append(AP_all, AP)
        ZSPEC_all = np.append(ZSPEC_all, ZSPEC)
        type_len_dict[v] = len(good_index)
        
        spec_data.close()
        table0.close()
    #endfor
    cdelt_avg = np.average(cdelt_all)
    
    grid_col = np.arange(min(x0min_all), max(x0max_all)+cdelt_avg, cdelt_avg)
    grid_ndarr = np.zeros((len(AP_all), len(grid_col)))

    mark = 0
    masknum = 0
    for key in np.sort(type_len_dict.keys()):
        print key,len(img_data_dict['img_'+key]),len(img_data_dict['img_'+key][0])
        for (arr, arr_index) in zip(img_data_dict['img_'+key],
                                    range(len(img_data_dict['img_'+key]))):
            xarr = x0_dict[key]
            if len(xarr) != len(arr):
                goodlen = min(len(xarr), len(arr))
                xarr = xarr[:goodlen]
                arr = arr[:goodlen]
            #endif
            f = interp1d(xarr, arr)
            good_range = np.where((grid_col >= min(xarr)) &
                                  (grid_col <= max(xarr)))[0]
            temp = grid_col[good_range]
            grid_ndarr[arr_index + mark][good_range] = f(temp) #uses interpolation
        #endfor

        if (key=='1' or key=='2' or key=='3' or key=='4'):
            masknum += len(img_data_dict['img_'+key])
        #endif
        
        mark += (arr_index+1)
    #endfor

    grid_ndarr[:masknum,4296:] = 0
    if version=='MMT':
        # special cases
        AP_names = np.concatenate((np.array(['D.028', 'D.069', 'D.075', 'D.076', 'D.081', 'D.086', 'D.088', 'D.093',
            'D.099', 'D.103', 'D.106', 'D.108', 'D.112', 'D.123', 'D.127', 'D.131',
            'D.132', 'D.140', 'D.147', 'D.150', 'D.195', 'D.213', 'D.215', 'D.231',
            'D.237', 'D.248', 'D.258', 'D.280', 'D.288', 'D.298', 'S.004', 'S.018',
            'S.062', 'S.063', 'S.067', 'S.068', 'S.070', 'S.071', 'S.222', 'S.223',
            'S.224', 'S.225', 'S.226', 'S.228', 'S.229', 'S.231', 'S.232', 'S.234',
            'S.235', 'S.236', 'S.237', 'S.239', 'S.244', 'S.246', 'S.247', 'S.248',
            'S.251', 'S.253', 'S.255', 'S.256', 'S.257', 'S.258', 'S.259', 'S.260',
            'S.261', 'S.267', 'S.268', 'S.269', 'S.270', 'S.271', 'S.272', 'S.275',
            'S.276', 'S.277', 'S.279', 'S.280', 'S.281', 'S.282', 'S.286', 'S.287',
            'S.289', 'S.290', 'S.293', 'S.294', 'S.296', 'S.297', 'S.298', 'S.299',
            'S.300', 'S.301', 'S.302', 'S.303', 'S.304', 'S.305', 'S.316', 'S.315',
            'S.312', 'S.311', 'S.310', 'S.308', 'S.307', 'S.327', 'S.326', 'S.325',
            'S.323', 'S.322', 'S.321', 'S.318', 'S.332', 'S.330', 'S.354', 'S.349',
            'S.348', 'S.347', 'S.346', 'S.345', 'S.344', 'S.342', 'S.336', 'S.335',
            'S.366', 'S.363', 'S.361', 'S.360', 'S.359', 'S.358', 'S.357', 'S.356',
            'S.353', 'S.341', 'S.338', 'S.331']), np.array(['S.367', 'S.365', 'S.329', 'S.317', 'S.328'])))
        mask_cutoffs = np.concatenate((np.array([6500]*132), np.array([6566, 6569, 6598, 6566, 6597])))
        for ap_name, mask_ii in zip(AP_names, mask_cutoffs):
            ii = np.where(ap_name == AP_all)[0][0]
            x_test = x0/(1.0 + ZSPEC_all[ii])
            tmpidx = np.where((x_test >= mask_ii))[0]
            grid_ndarr[ii][tmpidx] = np.zeros(np.shape(grid_ndarr[ii][tmpidx]))
        #endfor
    #endif

    pyfits.writeto(full_path+'Spectra/spectral_'+version+'_grid.fits', grid_ndarr,
                   clobber=True)
    hdr = pyfits.getheader(full_path+'Spectra/spectral_'+version+'_grid.fits')
    hdr.append(('CDELT1', cdelt_avg))
    hdr.append(('CRVAL1', min(grid_col)))
    hdr.append(('CRPIX1', 1))
    hdr.append(('CTYPE1', 'LINEAR'))
    pyfits.writeto(full_path+'Spectra/spectral_'+version+'_grid.fits', grid_ndarr,
                   header=hdr,clobber=True)
    
    t = Table([AP_all, ZSPEC_all],names=['AP','ZSPEC'])
    asc.write(t,full_path+'Spectra/spectral_'+version+'_grid_data.txt',
              format='fixed_width',delimiter='\t')
#endfor
