"""
NAME:
    

PURPOSE:

    depends on 

INPUTS:
    

OUTPUTS:
    

NOTES:
    
"""

import numpy as np

def compute_onesig_pdf(arr0, x_val):
    '''https://github.com/astrochun/chun_codes/blob/master/__init__.py
    '''
    len0 = arr0.shape[0] # arr0.shape[1] # Mod on 29/06/2016

    err   = np.zeros((len0,2)) # np.zeros((2,len0)) # Mod on 29/06/2016
    xpeak = np.zeros(len0)

    conf = 0.68269 # 1-sigma

    for ii in range(len0):
        test = arr0[ii] # arr0[:,ii] # Mod on 29/06/2016
        good = np.where(np.isfinite(test) == True)[0]
        if len(good) > 0:
            v_low  = np.percentile(test[good],15.8655)
            v_high = np.percentile(test[good],84.1345)

            xpeak[ii] = np.percentile(test[good],50.0)
            if len0==1:
                t_ref = x_val
            else:
                t_ref = x_val[ii]

        err[ii,0]  = t_ref - v_low
        err[ii,1]  = v_high - t_ref

    return err, xpeak


def random_pdf(x, dx, seed_i, n_iter=1000):
    '''
    adapted from https://github.com/astrochun/chun_codes/blob/master/__init__.py
    '''
    if type(x)==np.float64:
        len0 = 1
    else:
        len0 = len(x)
    x_pdf  = np.zeros((len0, n_iter), dtype=np.float64)

    seed0 = seed_i + np.arange(len0)

    for ii in range(len0):
        np.random.seed(seed0[ii])
        temp = np.random.normal(0.0, 1.0, size=n_iter)
        if len0==1:
            x_pdf[ii]  = x + dx*temp
        else:
            x_pdf[ii]  = x[ii] + dx[ii]*temp

    return x_pdf


def composite_errors(x, dx, label, seed_i):
    if label=='HA/HB' or label=='HG/HB':
        hn_flux = np.array(x[0])
        hn_rms = np.array(dx[0])
        hb_flux = np.array(x[1])
        hb_rms = np.array(dx[1])

        good_iis = np.where((hn_flux > 0) & (hb_flux > 0))[0]
        onesig_errs = np.ones((len(hn_flux), 2))*np.nan
        if len(good_iis) > 0:
            hn_pdf = random_pdf(hn_flux[good_iis], hn_rms[good_iis], seed_i)
            hb_pdf = random_pdf(hb_flux[good_iis], hb_rms[good_iis], seed_i)
            x_pdf = hn_pdf/hb_pdf

            if label=='HA/HB':
                import matplotlib.pyplot as plt, datetime
                f, axes = plt.subplots(len(good_iis),1, sharex=True)
                for i, ax in zip(range(len(good_iis)), axes):
                    ax.hist(x_pdf[i])
                plt.show()
            #     # plt.savefig(str(datetime.datetime.now().time())+'.pdf')

            err, xpeak = compute_onesig_pdf(x_pdf, hn_flux[good_iis]/hb_flux[good_iis])
            onesig_errs[good_iis] = err

    else:
        x_pdf = random_pdf(x, dx, seed_i)
        onesig_errs, xpeak = compute_onesig_pdf(x_pdf, x)

    # print 'onesig_errs:::::::', onesig_errs
    return onesig_errs