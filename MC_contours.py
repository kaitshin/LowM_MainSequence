import numpy as np
from matplotlib import pyplot as plt
from astropy.stats import sigma_clip
from scipy import interpolate
from astropy.convolution import convolve, Box2DKernel
from astropy.io import ascii as asc
from MACT_utils import get_tempz
from scipy.optimize import curve_fit
from analysis.composite_errors import random_pdf


FULL_PATH = '/Users/kaitlynshin/GoogleDrive/NASA_Summer2015/'
CUTOFF_SIGMA = 4.0
CUTOFF_MASS = 6.0

n_repeats = 50000


def contours(x, y, xsize=0.01, ysize=0.01, three_sig=False):
    '''
    adapted from code provided to me by Chun Ly
    '''
    #creating 2D confidence contours of a parameter
    #Input: x, y, xname, yname. x and y are arrrays containing the values of given parameter. 
    #Return: x_final, y_final, hist2d, sig_levels

    #removing outliers
    xc = sigma_clip(x, sigma=3, iters=60)
    yc = sigma_clip(y, sigma=3, iters=60)

    #setting x and y size based on which parameters are being plotted. For data that is tighter a smaller size is used.
    
    #caculating the number of bins along each axis    
    x_nbins = np.ceil((max(x)-min(x))/xsize)     
    y_nbins = np.ceil((max(y)-min(y))/ysize)        

    #creating girds
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
    #print levels

    #empty array for percent in a given level
    percent = np.zeros(len(levels))

    #loop to fill percetn array
    for jj in range(int(max0)+1):
        #define a level
        level = max0 - jj
        #print level

        #finding points in hist2d above that level
        where = np.where(hist2d > level)
        ugh = np.unique(where[0])

        #calucalting percent
        percent[jj] = (np.sum(hist2d[ugh]))/(np.sum(hist2d))

    #print percent
    f = interpolate.interp1d(percent, levels, bounds_error=False)
    if three_sig:
        sigs = [0.68269, 0.95449, 0.9973002]
    else:
        sigs = [0.68269, 0.95449]

    sig_levels = f(sigs)
    sig_levels.sort()
    #print sig_levels

    return x_final, y_final, hist2d, sig_levels


def confidence(x):
    '''
    adapted from code provided to me by Chun Ly
    '''
    #finding unequal error in a value
    #Input: x. x is an arrray containing the values of given parameter
    #Return: low_limit, high_limit

    #removing outliers from data set
    xc = sigma_clip(x, sigma=3, iters=40)

    #calculating number of bins to use
    nbins = np.ceil((max(x)-min(x))/(0.02*np.std(xc)))

    #creating histogram of parameter
    hist, bin_edges = np.histogram(x, bins=int(nbins))

    #calculating size of a bin
    size = (np.max(x)-np.min(x))/nbins

    #finding peak of distribution
    max0 = max(hist)
    #print bin_edges[np.where(hist == max0)[0][0]]

    #stepping down from peak to zero
    levels = range(int(max0), -1, -1)
    #print levels

    #empty array for percent in a given level
    percent = np.zeros(len(levels))

    #filling percent array
    for jj in range(int(max0)+1):
        temp = []
        level = max0 - jj

        #finding points in hist above that level
        ugh = np.where(hist > level)

        #calculating precent
        percent[jj] = (np.sum(hist[ugh])/(np.sum(hist)+0.0))
        #print level, np.sum(temp)

    #print percent
    f = interpolate.interp1d(percent, levels)
    sigs = [0.68269]

    sig_levels = f(sigs)
    #print sig_levels

    idx_max = np.where(hist == np.max(hist))[0]

    left = np.arange(idx_max[0]+1)
    right = np.array(np.arange(idx_max[-1], len(hist)))

    f_left = interpolate.interp1d(hist[left], left)
    f_right = interpolate.interp1d(hist[right], right)

    low1 = f_left(sig_levels)
    high1 = f_right(sig_levels)

    low = bin_edges[0]+low1*size
    high = bin_edges[0]+high1*size

    #n,bins,patches = plt.hist(x,nbins, facecolor='green',align='mid')
    #plt.plot((np.min(x),np.max(x)),(sig_levels, sig_levels))
    #plt.show()

    low_limit = np.median(x)-low
    high_limit = high-np.median(x)

    return low_limit, high_limit


def contours_three_params(sfrs, delta_sfrs, mass, mz_data):
    '''
    '''
    eqn_str = r'$\log(SFR) = \alpha \log(M) + \beta z + \gamma$'
    def func(data, a, b, c):
        return a*data[:,0] + b*data[:,1] + c

    params00, pcov = curve_fit(func, mz_data, sfrs)

    num_iters = 10000
    seed = 132089
    sfrs_pdf = random_pdf(sfrs, delta_sfrs, seed_i=seed, n_iter=num_iters)

    np.random.seed(12376)

    alpha_arr = np.array([])
    beta_arr = np.array([])
    gamma_arr = np.array([])

    for i in range(num_iters):
        s_arr = sfrs_pdf[:,i]

        params, pcov = curve_fit(func, mz_data, s_arr)
        alpha_arr = np.append(alpha_arr, params[0])
        beta_arr = np.append(beta_arr, params[1])
        gamma_arr = np.append(gamma_arr, params[2])

    # plotting
    f, axes = plt.subplots(1,3)
    params_arr = [alpha_arr, beta_arr, gamma_arr]
    lbl_arr = [r'$\alpha$', r'$\beta$', r'$\gamma$']

    for i, ax, const_arr, lbl in zip(range(3), axes, params_arr, lbl_arr):
        j = (i+1)%3

        if i>0: # 0.0025 ?
            xsize=0.001 if np.std(params_arr[i]) < 0.01 else 0.01
            ysize=0.001 if np.std(params_arr[j]) < 0.01 else 0.01
        else:
            xsize=0.001 if np.std(params_arr[i]) < 0.01 else 0.01
            ysize=0.001 if np.std(params_arr[j]) < 0.01 else 0.011

        x_final, y_final, hist2d, sig_levels = contours(params_arr[i],
            params_arr[j], xsize, ysize, three_sig=True)
        # ax.contour(x_final, y_final, hist2d, levels=sig_levels, colors=('0.75','0.75','0.75'))
        ax.contour(x_final, y_final, hist2d, levels=sig_levels, colors=('0.90','0.75','0.75'))
        ax.scatter(np.mean(params_arr[i]), np.mean(params_arr[j]), zorder=2)
        ax.scatter(params00[i], params00[j], zorder=1) # 'true' value
        
        ax.text(0.05, 0.06,
            lbl_arr[i]+r'$=%.2f \pm(%.2f,%.2f)$'%(np.median(params[i]),99,99)+'\n'+
            lbl_arr[j]+r'$=%.2f \pm(%.2f,%.2f)$'%(np.median(params[j]),99,99),
            transform=ax.transAxes)
        ax.set_xlabel(lbl_arr[i])
        ax.set_ylabel(lbl_arr[j])
        # ax.set_title(lbl_arr[j]+' (%.3f)'%params00[j]+' vs. '+lbl_arr[i]+' (%.3f)'%params00[i])
        
    [ax.tick_params(axis='both', labelsize='10', which='both',
        direction='in') for ax in axes]
    f.set_size_inches(15,5)
    # plt.subplots_adjust(wspace=0.2, left=0.04, bottom=0.09)
    plt.tight_layout()
    plt.savefig(FULL_PATH+'Plots/main_sequence/MC_regr_contours.pdf')


def main():
    # reading input files
    corr_tbl = asc.read(FULL_PATH+'Main_Sequence/mainseq_corrections_tbl.txt',
        guess=False, Reader=asc.FixedWidthTwoLine)
    good_sig_iis = np.where((corr_tbl['flux_sigma'] >= CUTOFF_SIGMA) & 
        (corr_tbl['stlr_mass'] >= CUTOFF_MASS))[0]
    corr_tbl = corr_tbl[good_sig_iis]

    filts = corr_tbl['filt'].data
    delta_sfrs = corr_tbl['meas_errs'].data
    mass = corr_tbl['stlr_mass'].data
    obs_sfr = corr_tbl['met_dep_sfr'].data
    dust_corr_factor = corr_tbl['dust_corr_factor'].data
    filt_corr_factor = corr_tbl['filt_corr_factor'].data
    nii_ha_corr_factor = corr_tbl['nii_ha_corr_factor'].data
    sfrs = obs_sfr+filt_corr_factor+nii_ha_corr_factor+dust_corr_factor

    zspec0 = corr_tbl['zspec0'].data
    no_spectra  = np.where((zspec0 <= 0) | (zspec0 > 9))[0]
    yes_spectra = np.where((zspec0 > 0) & (zspec0 < 9))[0]

    tempz = get_tempz(zspec0, filts)
    mz_data = np.vstack([mass, tempz]).T

    contours_three_params(sfrs, delta_sfrs, mass, mz_data)


if __name__ == '__main__':
    main()