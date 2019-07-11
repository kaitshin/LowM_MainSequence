from astropy.io import fits
from astropy.io import ascii
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import readsav
from matplotlib.backends.backend_pdf import PdfPages
from scipy import asarray as ar,exp
from astropy.stats import sigma_clip
from astropy.table import hstack, Column, Table
from scipy import interpolate

def confidence(x):
    xc = sigma_clip(x, sigma=3, iters=40)
    nbins = np.ceil((max(x)-min(x))/(0.02*np.std(xc)))
    
    hist, bin_edges = np.histogram(x, bins=nbins)
    size = (np.max(x)-np.min(x))/nbins
    max0 = max(hist)
    #print bin_edges[np.where(hist == max0)[0][0]]
    levels = range(int(max0), -1, -1)
    #print levels

    percent = np.zeros(len(levels))
    for jj in range(int(max0)+1):
        temp = []
        level = max0 - jj
        ugh = np.where(hist > level)
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


def main():
	return 0


if __name__ == '__main__':
	main()

# def run():
    # tab0 = readsav('/Users/weldon119/Google Drive/DEEP2_Spitzer/Monte Carlo/MPFIT_fit.EQ.save')
    # tab1 = readsav('/Users/weldon119/Google Drive/DEEP2_Spitzer/Ly2016/bin_MZ_z2.save')
    # tab2 = '/Users/weldon119/Google Drive/DEEP2_Spitzer/Monte Carlo/Monte_Carlo_raw_MACT_nondet_EQ_binned_data'
    
    # pp = PdfPages('/Users/weldon119/Google Drive/DEEP2_Spitzer/Monte Carlo/IDL_EQ_contours_Jan22.pdf')
    # pp2 = PdfPages('/Users/weldon119/Google Drive/DEEP2_Spitzer/Monte Carlo/IDL_EQ_analysis_Jan22.pdf')
    # outfile ='/Users/weldon119/Google Drive/DEEP2_Spitzer/Monte Carlo/IDL_EQ_binned_parameters_Jan22.tbl'

    # temp = tab0.z_fit0

    # OH = np.array([temp[ii][0] for ii in range(len(temp))])
    # MTO = np.array([temp[ii][1] for ii in range(len(temp))])
    # gamma = np.array([temp[ii][2] for ii in range(len(temp))])

    # #loading in fixed parameters
    # raw = np.load(tab2+'.npz')     
    # OH_fixed = raw['OH_fixed']
    # gamma_fixed = raw['gamma_fixed']
    
    # #OHc = OH[np.where((OH <= 10) & (OH >=8))] 
    # #MTOc = MTO[np.where((MTO <= 11) & (MTO >=7.25))]
    # #gammac = gamma[np.where((gamma <=2) & (gamma >=0))]

    # idx = np.where((OH <= 10) & (OH >=8) & (MTO <= 11) & (MTO >=7.25) & (gamma <=2) & (gamma >=0))[0]

    # OH_fixedc = sigma_clip(OH_fixed, sigma=3, iters=80)
    # gamma_fixedc = sigma_clip(gamma_fixed, sigma=3, iters=80)

    # OH_med = np.median(OH[idx])
    # MTO_med = np.median(MTO[idx])
    # gamma_med = np.median(gamma[idx])
    # print OH_med, gamma_med, MTO_med

    # #histograms
    # order = [OH, gamma, MTO, OH_fixed, gamma_fixed]
    # title = ['OH', 'gamma', 'Turnover Mass','OH with fixed MTO', 'gamma with fixed MTO']
    # x_range = [[8,20], [0,2.5], [6,60]]
    # hist_max = np.zeros(5)

    # fig_hist = plt.figure(figsize=(8,8))
    # for kk in range(len(order)):
    #     x = order[kk]

    #     if title[kk] == 'gamma':
    #         size = 0.01
    #     elif title[kk] == 'OH with fixed MTO':
    #         size = 0.01
    #     elif title[kk] == 'gamma with fixed MTO':
    #         size = 0.01
    #     else:
    #         size = 0.05
            
    #     nbins = np.ceil((max(x)-min(x))/(size))       #(0.02*np.std(xc)))
    #     n,bins,patches = plt.hist(x, nbins, facecolor='green', align='mid')
    #     hist_max[kk] = bins[np.where(n == np.max(n))[0][0]]+(0.5*0.05)
    #     plt.axvline(x=hist_max[kk], label='max')
    #     plt.axvline(x=np.median(x), color='r', label='median')
    #     if kk < 3:
    #         plt.xlim(x_range[kk][0], x_range[kk][1])
    #     if kk == 0:
    #         plt.legend(prop={'size':7})
    #     plt.title(title[kk])
    #     fig_hist = plt.gcf(); fig_hist.savefig(pp2, format='pdf')
    #     fig_hist.clear()
    
    # pp2.close()

    # #Making table
    # parameters = [OH_med, gamma_med, MTO_med]
    # table = Table([parameters], names=['binned parameters'])
    # fixed = Column([hist_max[3], hist_max[4], 8.901], name='binned fixed MTO parameters')
    # ID = Column(['OH asy','gamma','MTO'],name='ID')

    # OH_low, OH_high = confidence(OH[idx])
    # gamma_low, gamma_high = confidence(gamma[idx])
    # MTO_low, MTO_high = confidence(MTO[idx])
    # low = Column([OH_low, gamma_low, MTO_low], name='low bound')
    # high = Column([OH_high, gamma_high, MTO_high], name='high bound')

    # OH_fixed_low, OH_fixed_high = confidence(OH_fixed)
    # gamma_fixed_low, gamma_fixed_high = confidence(gamma_fixed)
    # low_fixed = Column([OH_fixed_low, gamma_fixed_low, 0], name='low fixed bound')
    # high_fixed = Column([OH_fixed_high, gamma_fixed_high, 0], name='high fixed bound')

    # table.add_column(ID, index=0)
    # table.add_columns([low, high, fixed, low_fixed, high_fixed])
    # table.write(outfile, format='ascii.fixed_width_two_line',overwrite=True)
    
    # gs_top = plt.GridSpec(3,1,hspace=0.02)
    # gs_bottom = plt.GridSpec(3,1)
    # fig = plt.figure(figsize=(5,15))

    # ax = fig.add_subplot(gs_top[0,:])
    # axes = [ax] + [fig.add_subplot(gs_top[1], sharex=ax)]
    # axes.append(fig.add_subplot(gs_bottom[2]))
    
    # cp = axes[0].contour(tab0.cx2, tab0.cy2, tab0.prob2s, levels=tab0.levels2s, colors=('k','k','k'))
    # cp_ly16 = axes[0].contour(tab1.cx2, tab1.cy2, tab1.prob2s, levels=tab1.levels2s, colors=('#9e3623','#9e3623','#9e3623'), linestyles='dashed', label='Ly2015 contour')
    # axes[0].set_ylabel('$12+log(O/H)_{asm}$')
    # axes[0].set_ylim(8, 10)
    # #axes[0].scatter(hist_max[1], hist_max[0], color='k', label='Weldon2017')
    # axes[0].scatter(gamma_med, OH_med, color='k', label='Weldon2017')
    # axes[0].scatter(0.67, 8.46, facecolors='none', edgecolors='#9e3623', label='Ly et al 2015')
    # axes[0].scatter(0.64, 8.798, facecolors='none', edgecolors='b', label='A&M13')

    # cp = axes[1].contour(tab0.cx1, tab0.cy1, tab0.prob1s, levels=tab0.levels1s, colors=('k','k','k'))
    # cp_ly16 = axes[1].contour(tab1.cx1, tab1.cy1, tab1.prob1s, levels=tab1.levels1s, colors=('#9e3623','#9e3623','#9e3623'), linestyles='dashed', label='Ly2015 contour')
    # axes[1].set_xlabel('$\gamma$')
    # axes[1].set_ylabel('$log(M_{TO}/M_{\odot})$')
    # axes[1].set_xlim(0,3)
    # axes[1].set_ylim(7.25,11.25)
    # #axes[1].scatter(hist_max[1], hist_max[2], color='k')
    # axes[1].scatter(gamma_med, MTO_med, color='k')
    # axes[1].scatter(0.67, 8.61, facecolors='none', edgecolors='#9e3623')
    # axes[1].scatter(0.64, 8.901, facecolors='none', edgecolors='b')

    # cp = axes[2].contour(tab0.cx0, tab0.cy0, tab0.prob0s, levels=tab0.levels0s, colors=('k','k','k'))
    # cp_ly16 = axes[2].contour(tab1.cx0, tab1.cy0, tab1.prob0s, levels=tab1.levels0s, colors=('#9e3623','#9e3623','#9e3623'), linestyles='dashed',label='Ly2015 contour')
    # axes[2].set_xlabel('$12+log(O/H)_{asm}$')
    # axes[2].set_ylabel('$log(M_{TO}/M_{\odot})$')
    # axes[2].set_xlim(8,10)
    # axes[2].set_ylim(7.25,11.25)
    # #axes[2].scatter(hist_max[0], hist_max[2], color='k')
    # axes[2].scatter(OH_med, MTO_med, color='k')
    # axes[2].scatter(8.46, 8.61, facecolors='none', edgecolors='#9e3623')
    # axes[2].scatter(8.798, 8.901, facecolors='none', edgecolors='b')

    # plt.setp(axes[0].get_xticklabels(), visible=False)
    # axes[0].tick_params(axis='both',which='both',direction='in',right='on',bottom='on',top='on')
    # axes[1].tick_params(axis='both',which='both',direction='in',right='on',top='on')
    # axes[2].tick_params(axis='both',which='both',direction='in',right='on',top='on')
    # axes[0].minorticks_on()
    # axes[1].minorticks_on()
    # axes[2].minorticks_on()
    # axes[0].legend(prop={'size':7})
    # fig.subplots_adjust(left=0.14, bottom=0.04, right=0.95, top=0.99)

    # fig.savefig(pp, format='pdf')
    # pp.close()