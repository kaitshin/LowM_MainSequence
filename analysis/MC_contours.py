import numpy as np
from matplotlib import pyplot as plt
from astropy.stats import sigma_clip
from scipy import interpolate
from astropy.convolution import convolve, Box2DKernel

n_repeats = 50000

def contours(x, y, xsize=0.01, ysize=0.01, three_sig=False):
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
