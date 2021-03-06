"""
NAME:
    MC_contours.py

PURPOSE:
    

INPUTS:
    config.FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt'

OUTPUTS:
    config.FULL_PATH+'Plots/main_sequence/MC_regr_contours_noz.pdf'
    config.FULL_PATH+'Plots/main_sequence/MC_regr_contours.pdf'
"""
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
from astropy.stats import sigma_clip
from scipy import interpolate
from astropy.convolution import convolve, Box2DKernel
from astropy.io import ascii as asc

import config
from MACT_utils import get_tempz, get_mainseq_fit_params, compute_onesig_pdf, get_FUV_corrs

num_iters = 10000


def contours(x, y, xsize=0.01, ysize=0.01, three_sig=False):
    '''
    adapted from code provided to me by Chun Ly

    creating 2D confidence contours of a parameter
    Input: x, y, xname, yname. x and y are arrays
        containing the values of given parameter.
    Return: x_final, y_final, hist2d, sig_levels
    '''
    #removing outliers
    xc = sigma_clip(x, sigma=3, maxiters=60)
    yc = sigma_clip(y, sigma=3, maxiters=60)
    
    #caculating the number of bins along each axis    
    x_nbins = int(np.ceil((max(x)-min(x))/xsize))
    y_nbins = int(np.ceil((max(y)-min(y))/ysize))

    #creating grids
    x_grid = np.linspace(min(x), max(x), x_nbins)
    y_grid = np.linspace(min(y), max(y), y_nbins)
    
    hist2d, xedges, yedges = np.histogram2d(x, y, bins=(x_grid, y_grid))

    #smoothing hist2d
    box_kernel = Box2DKernel(4)
    hist2d = convolve(hist2d, box_kernel, normalize_kernel=True)

    xedges  = xedges[:-1]
    yedges  = yedges[:-1]

    x_final, y_final = np.meshgrid(xedges, yedges)
    
    x_final = x_final.transpose()
    y_final = y_final.transpose()

    #finding peak of 2d distribution
    max0 = np.max(hist2d)
    idx_max = np.where(hist2d == np.max(hist2d))

    #stepping down from peak to zero
    levels = range(int(max0), -1, -1)

    #empty array for percent in a given level
    percent = np.zeros(len(levels))

    #loop to fill percent array
    for jj in range(int(max0)+1):
        #define a level
        level = max0 - jj

        #finding points in hist2d above that level
        where = np.where(hist2d > level)
        ugh = np.unique(where[0])

        #calculating percent
        percent[jj] = (np.sum(hist2d[ugh]))/(np.sum(hist2d))

    f = interpolate.interp1d(percent, levels, bounds_error=False)
    if three_sig:
        sigs = [0.68269, 0.95449, 0.9973002]
    else:
        sigs = [0.68269, 0.95449]

    sig_levels = f(sigs)
    sig_levels.sort()

    return x_final, y_final, hist2d, sig_levels


def confidence(x):
    '''
    adapted from code provided to me by Chun Ly

    finding unequal error in a value
    Input: x. x is an arrray containing the values of given parameter
    Return: low_limit, high_limit
    '''
    #removing outliers from data set
    xc = sigma_clip(x, sigma=3, maxiters=40)

    #calculating number of bins to use
    nbins = np.ceil((max(x)-min(x))/(0.02*np.std(xc)))

    #creating histogram of parameter
    hist, bin_edges = np.histogram(x, bins=int(nbins))

    #calculating size of a bin
    size = (np.max(x)-np.min(x))/nbins

    #finding peak of distribution
    max0 = max(hist)

    #stepping down from peak to zero
    levels = range(int(max0), -1, -1)

    #empty array for percent in a given level
    percent = np.zeros(len(levels))

    #filling percent array
    for jj in range(int(max0)+1):
        level = max0 - jj

        #finding points in hist above that level
        ugh = np.where(hist > level)

        #calculating precent
        percent[jj] = (np.sum(hist[ugh])/(np.sum(hist)+0.0))

    f = interpolate.interp1d(percent, levels)
    sigs = [0.68269]

    sig_levels = f(sigs)

    idx_max = np.where(hist == np.max(hist))[0]

    left = np.arange(idx_max[0]+1)
    right = np.array(np.arange(idx_max[-1], len(hist)))

    f_left = interpolate.interp1d(hist[left], left)
    f_right = interpolate.interp1d(hist[right], right)

    low1 = f_left(sig_levels)
    high1 = f_right(sig_levels)

    low = bin_edges[0]+low1*size
    high = bin_edges[0]+high1*size

    low_limit = np.median(x)-low
    high_limit = high-np.median(x)

    return low_limit, high_limit


def plot_contours_two_params(sfrs, delta_sfrs, mass):
    '''
    '''
    params_arr = get_mainseq_fit_params(sfrs, delta_sfrs, mass, num_params=2)
    errs_arr = []
    for i, param_arr in enumerate(params_arr):
        errs_arr.append(compute_onesig_pdf(param_arr.reshape(num_iters,1).T,
            [np.mean(param_arr)])[0][0])

    # plotting
    f, ax = plt.subplots(1,1)
    lbl_arr = [r'$\alpha$', r'$\gamma$']

    i, j = 0, 1
    xsize=0.001 if np.std(params_arr[i]) < 0.01 else 0.01
    ysize=0.001 if np.std(params_arr[j]) < 0.01 else 0.01

    x_final, y_final, hist2d, sig_levels = contours(params_arr[i],
        params_arr[j], xsize, ysize, three_sig=False)

    ax.contour(x_final, y_final, hist2d, levels=sig_levels, colors='black', linewidths=1)
    ax.scatter(np.mean(params_arr[i]), np.mean(params_arr[j]), zorder=2)

    ax.text(0.05, 0.06,
        lbl_arr[i]+r'$=%.3f \pm %.3f$'%(np.mean(params_arr[i]), np.mean([errs_arr[i][0], errs_arr[i][1]]))+'\n'+
        lbl_arr[j]+r'$=%.2f \pm %.2f$'%(np.mean(params_arr[j]), np.mean([errs_arr[j][0], errs_arr[j][1]])),
        transform=ax.transAxes, fontsize=13)
    ax.set_xlabel(lbl_arr[i], fontsize=12)
    ax.set_ylabel(lbl_arr[j], fontsize=12)
    ax.set_xlim([np.mean(params_arr[i])-0.02, np.mean(params_arr[i])+0.02])
    ax.set_ylim([np.mean(params_arr[j])-0.2, np.mean(params_arr[j])+0.2])

    ax.tick_params(axis='both', labelsize='10', which='both', direction='in')
    f.set_size_inches(5,4)
    plt.tight_layout()
    plt.savefig(config.FULL_PATH+'Plots/main_sequence/MC_regr_contours_noz.pdf')


def plot_contours_three_params(sfrs, delta_sfrs, mz_data):
    '''
    '''
    params_arr = get_mainseq_fit_params(sfrs, delta_sfrs, mz_data, num_params=3)
    errs_arr = []
    for i, param_arr in enumerate(params_arr):
        errs_arr.append(compute_onesig_pdf(param_arr.reshape(num_iters,1).T,
            [np.mean(param_arr)])[0][0])

    # plotting
    f, axes = plt.subplots(1,3)
    lbl_arr = [r'$\alpha$', r'$\beta$', r'$\gamma$']

    for i, ax, lbl in zip(range(3), axes, lbl_arr):
        j = (i+1)%3

        print(np.std(params_arr[i]))
        mult=2
        if i>0: # 0.0025 ?
            xsize=mult*0.001 if np.std(params_arr[i]) < mult*0.01 else mult*0.01
            ysize=mult*0.001 if np.std(params_arr[j]) < mult*0.01 else mult*0.01
        else:
            xsize=mult*0.001 if np.std(params_arr[i]) < mult*0.01 else mult*0.01
            ysize=mult*0.001 if np.std(params_arr[j]) < mult*0.01 else mult*0.011

        x_final, y_final, hist2d, sig_levels = contours(params_arr[i],
            params_arr[j], xsize, ysize, three_sig=False)

        ax.contour(x_final, y_final, hist2d, levels=sig_levels, colors='black', linewidths=1)
        ax.scatter(np.mean(params_arr[i]), np.mean(params_arr[j]), zorder=2)
        
        ax.text(0.05, 0.06,
            lbl_arr[i]+r'$=%.2f \pm %.2f$'%(np.mean(params_arr[i]), np.mean([errs_arr[i][0], errs_arr[i][1]]))+'\n'+
            lbl_arr[j]+r'$=%.2f \pm %.2f$'%(np.mean(params_arr[j]), np.mean([errs_arr[j][0], errs_arr[j][1]])),
            transform=ax.transAxes, fontsize=13)
        ax.set_xlabel(lbl_arr[i], fontsize=12)
        ax.set_ylabel(lbl_arr[j], fontsize=12)
        
    [ax.tick_params(axis='both', labelsize='10', which='both',
        direction='in') for ax in axes]
    f.set_size_inches(15,5)
    plt.tight_layout()
    plt.savefig(config.FULL_PATH+'Plots/main_sequence/MC_regr_contours.pdf')


def main():
    # reading input files
    corr_tbl = asc.read(config.FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt',
        guess=False, Reader=asc.FixedWidthTwoLine)
    good_sig_iis = np.where((corr_tbl['flux_sigma'] >= config.CUTOFF_SIGMA) & 
        (corr_tbl['stlr_mass'] >= config.CUTOFF_MASS))[0]
    corr_tbl = corr_tbl[good_sig_iis]

    filts = corr_tbl['filt'].data
    delta_sfrs = corr_tbl['meas_errs'].data
    mass = corr_tbl['stlr_mass'].data
    obs_sfr = corr_tbl['met_dep_sfr'].data
    dust_corr_factor = corr_tbl['dust_corr_factor'].data
    filt_corr_factor = corr_tbl['filt_corr_factor'].data
    nii_ha_corr_factor = corr_tbl['nii_ha_corr_factor'].data
    FUV_corr_factor = get_FUV_corrs(corr_tbl)
    sfrs = obs_sfr+filt_corr_factor+nii_ha_corr_factor+dust_corr_factor+FUV_corr_factor

    zspec0 = corr_tbl['zspec0'].data
    no_spectra  = np.where((zspec0 <= 0) | (zspec0 > 9))[0]
    yes_spectra = np.where((zspec0 > 0) & (zspec0 < 9))[0]

    tempz = get_tempz(zspec0, filts)
    mz_data = np.vstack([mass, tempz]).T

    plot_contours_two_params(sfrs, delta_sfrs, mass)
    plot_contours_three_params(sfrs, delta_sfrs, mz_data)


if __name__ == '__main__':
    main()