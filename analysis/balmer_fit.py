"""
NAME:
    balmer_fit.py

PURPOSE:
    Provides a module where fitting functions can be called from 
    stack_spectral_data.py
"""

import numpy as np, math
import scipy.optimize as optimization

def find_nearest(array,value):
    '''
    Uses np.searchsorted to find the array index closest to the input
    numerical value
    '''
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx
#enddef


def get_baseline_median(xval, yval, label):
    '''
    Returns the median of the baseline of the spectrum, masking out the
    emission peaks
    '''
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


def func(x, a, b, c, d):
    '''
    Is the passed-in model function for optimization.curve_fit
    '''
    u = (x-b)/c
    return a * np.exp(-0.5*u*u) + d
#enddef


def func3(x, a1, b1, c1, a2, b2, c2, d):
    '''
    Is the passed-in model function for optimization.curve_fit
    '''
    u = (x-b1)/c1
    v = (x-b2)/c2
    return a1*np.exp(-0.5*u*u) + a2*np.exp(-0.5*v*v) + d
#enddef


def get_best_fit(xval, yval, label):
    '''
    Uses scipy.optimize.curve_fit() to obtain the best fit of the spectra
    which is then returned

    Ha absorption spectra
    '''
    med0 = get_baseline_median(xval, yval, label)
    err = np.repeat(1.0e-18, len(xval))
    p0 = [np.max(yval)-med0, xval[np.argmax(yval)], 1.10, med0]

    o1,o2 = optimization.curve_fit(func, xval, yval, p0, err)
    return o1
#enddef


def get_best_fit2(xval, yval, peakxval, label):
    '''
    Uses scipy.optimize.curve_fit() to obtain the best fit of the spectra
    which is then returned

    NII 6548 (6548.1 A)

    NII 6583 (6583.6 A)
    '''
    med0 = np.median(yval)
    err = np.repeat(1.0e-18, len(xval))
    p0 = [yval[find_nearest(xval, peakxval)], peakxval, 1.10, med0]

    o1,o2 = optimization.curve_fit(func, xval, yval, p0, err)
    return o1
#enddef


def get_best_fit3(xval, yval, label):
    '''
    Uses scipy.optimize.curve_fit() to obtain the best fit of the spectra
    which is then returned

    Hg and Hb absorption spectra
    '''
    med0 = get_baseline_median(xval, yval, label)
    err = np.repeat(1.0e-18, len(xval))
    p0 = [np.max(yval)-med0, xval[np.argmax(yval)], 1.10,
          -0.05*(np.max(yval)-med0), xval[np.argmax(yval)], 2.20, med0]
    # print 'p0[3]:', p0[3]

    # param_bounds = ([0,xval[np.argmax(yval)]-3,0,-1,xval[np.argmax(yval)]-3,0,0],
    #     [1,xval[np.argmax(yval)]+3,10,0,xval[np.argmax(yval)]+3,10,1])
    # param_bounds = ([0,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],
    #     [np.inf,np.inf,np.inf,0,np.inf,np.inf,np.inf])

    # o1,o2 = optimization.curve_fit(func3, xval, yval, p0, err, bounds=param_bounds)
    o1,o2 = optimization.curve_fit(func3, xval, yval, p0, err)
    # print 'o1[3]:', o1[3]
    return o1
#enddef