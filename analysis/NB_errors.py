"""
NB_errors
====

Compute flux and EW errors based on NB and broadband photometry
"""

import sys, os

from chun_codes import systime, match_nosort, random_pdf, compute_onesig_pdf

from astropy.io import ascii as asc
from astropy.io import fits
from astropy.table import Table, Column

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import numpy as np

from glob import glob
from astropy import log

path0 = '/Users/cly/Google Drive/NASA_Summer2015/Catalogs/'

filt_ref= ['NB704', 'NB711', 'NB816', 'IA598', 'IA679','NB921','NB973']
dNB     = [  100.0,    72.0,   120.0,   303.0,   340.0,  132.0,  200.0]
lambdac = [ 7046.0,  7111.0,  8150.0,  6007.0,  6780.0, 9196.0, 9755.0]
dBB     = [ 1110.0,  1110.0,  1419.0,   885.0,  1110.0,  956.0,  956.0] # R R i, V, R

epsilon = np.array([0.5, 0.5, 0.6, 0.45, 0.75, 1.0, 1.0])
BB_filt = {'one': ['R','R','i','V','R','z', 'z'], 'two':['i','i','z','R','i','', '']}

filt_dict0 = {'filter': filt_ref, 'dNB': dNB, 'dBB': dBB, 'lambdac': lambdac}

m_AB = 48.6

def error_from_limit(mag, lim_mag):
  f     = 10**(-0.4*(m_AB+mag + np.log10(3)))
  f_lim = 10**(-0.4*(m_AB + lim_mag + np.log10(3)))

  error = - 2.5*np.log10(1 - f_lim/f)
  return error
#enddef

def fluxline(y, fNB, dNB, dBB):
  '''RETURNS emission-line flux in erg/s/cm^2'''

  return dNB*fNB*(1.0 - y)/(1.0-dNB/dBB)
#enddef

def ew_flux_dual(NB, BB, x, filt_dict):

  y_temp = 10**(-0.4 * x)

  dNB = filt_dict['dNB']
  dBB = filt_dict['dBB']
  lambdac = filt_dict['lambdac']

  EW  = dNB*(1 - y_temp)/(y_temp - dNB/dBB)

  fNB = 10**(-0.4*(NB + m_AB)) # in erg/s/cm2/Hz
  fNB = fNB*(3.0e8/(filt_dict['lambdac']**2*1.0e-10)) # in erg/s/cm2/Ang

  flux = fluxline(y_temp, fNB, dNB, dBB) # in erg/s/cm2

  return EW, flux
#enddef

def mag_combine(m1, m2, epsilon):
  cont_flux = epsilon * 10**(-0.4*(m1+m_AB)) + (1-epsilon)*10**(-0.4*(m2+m_AB))
  cont_mag  = -2.5*np.log10(cont_flux) - m_AB
  return cont_mag
#enddef

def get_data():

  infile = path0 + 'NB_IA_emitters.allcols.colorrev.fits'
  print("Reading : "+infile)

  data0 = fits.getdata(infile)

  tab0 = Table(data0)

  return tab0, infile
#enddef

def get_errors(tab0, filt_dict0, BB_filt, epsilon, limit_dict=None):

  NB_path = '/Users/cly/data/SDF/NBcat/'

  filt_ref = filt_dict0['filter']

  NB_phot_files = [NB_path+filt+'/sdf_pub2_'+filt+'.cat.mask' for filt in filt_ref]

  BB_phot_files1 = [NB_path+filt+'/sdf_pub2_'+BBfilt+'_'+filt+'.cat.mask' for
                    BBfilt,filt in zip(BB_filt['one'],filt_ref)]
  BB_phot_files2 = [NB_path+filt+'/sdf_pub2_'+BBfilt+'_'+filt+'.cat.mask' if
                    BBfilt != '' else '' for BBfilt,filt in
                    zip(BB_filt['two'],filt_ref)]

  n_gal = len(tab0)

  # Add columns
  for filt,ff in zip(filt_ref,range(len(filt_ref))):
    c0 = Column(np.zeros(n_gal), name=filt+'_MAG_ERROR')
    c0b= Column(np.zeros(n_gal), name=filt+'_MAG_ERROR_RAW')
    c1 = Column(np.zeros(n_gal), name=filt+'_CONT_MAG')
    c2 = Column(np.zeros(n_gal), name=filt+'_CONT_ERROR')
    c2b= Column(np.zeros(n_gal), name=filt+'_CONT_ERROR_RAW')
    c3 = Column(np.zeros(n_gal), name=filt+'_EW_UPERROR')
    c3b= Column(np.zeros(n_gal), name=filt+'_EW_LOERROR')
    c4 = Column(np.zeros(n_gal), name=filt+'_FLUX_UPERROR')
    c4b= Column(np.zeros(n_gal), name=filt+'_FLUX_LOERROR')

    colnames = tab0.colnames
    idx_end = [xx+1 for xx in range(len(colnames)) if colnames[xx] == filt+'_MAG']
    # +1 to add at end
    tab0.add_columns([c0,c0b,c1,c2,c2b,c3,c3b,c4,c4b], indexes=idx_end * 9)

    print("Reading : "+NB_phot_files[ff])
    phot_tab    = asc.read(NB_phot_files[ff])
    NB_id       = phot_tab['col1'].data
    MAG_APER    = phot_tab['col13'].data
    MAGERR_APER = phot_tab['col15'].data

    NBem = np.where(tab0[filt+'_ID'] != 0)[0]
    idx1, idx2  = match_nosort(tab0[filt+'_ID'][NBem], NB_id)
    idx1 = NBem[idx1]
    print('index size : '+str(len(NBem))+', '+str(len(idx2)))
    tab0[filt+'_MAG_ERROR_RAW'][idx1] = MAGERR_APER[idx2]

    if limit_dict == None:
      tab0[filt+'_MAG_ERROR'][idx1] = MAGERR_APER[idx2]
    else:
      NB_err = error_from_limit(MAG_APER[idx2], limit_dict['m_NB'][ff])
      tab0[filt+'_MAG_ERROR'][idx1] = NB_err

    print("Reading : "+BB_phot_files1[ff])
    phot_tab1       = asc.read(BB_phot_files1[ff])
    BB_MAG_APER1    = phot_tab1['col13'].data
    BB_MAGERR_APER1 = phot_tab1['col15'].data

    cont_mag = tab0[filt+'_MAG'] + tab0[filt+'_EXCESS']
    tab0[filt+'_CONT_MAG'][idx1] = cont_mag[idx1]

    if BB_filt['two'][ff] != '':
      print("Reading : "+BB_phot_files2[ff])
      phot_tab2       = asc.read(BB_phot_files2[ff])
      BB_MAG_APER2    = phot_tab2['col13'].data
      BB_MAGERR_APER2 = phot_tab2['col15'].data

      m1_dist = random_pdf(BB_MAG_APER1[idx2], BB_MAGERR_APER1[idx2], seed_i = ff,
                           n_iter=1000)
      m2_dist = random_pdf(BB_MAG_APER2[idx2], BB_MAGERR_APER2[idx2], seed_i = 2*ff+1,
                           n_iter=1000)

      cont_mag_dist = mag_combine(m1_dist, m2_dist, epsilon[ff])
      err, xpeak = compute_onesig_pdf(cont_mag_dist, cont_mag[idx1])
      g_err = np.sqrt(err[:,0]**2 + err[:,1]**2)
    else:
      g_err = BB_MAGERR_APER1[idx2]
      cont_mag_dist = random_pdf(BB_MAG_APER1[idx2], BB_MAGERR_APER1[idx2], seed_i = ff)

    tab0[filt+'_CONT_ERROR_RAW'][idx1] = g_err

    if limit_dict == None:
      tab0[filt+'_CONT_ERROR'][idx1] = g_err
    else:
      BB_err = error_from_limit(cont_mag[idx1], limit_dict['m_BB'][ff])
      tab0[filt+'_CONT_ERROR'][idx1] = BB_err
      cont_mag_dist = random_pdf(cont_mag[idx1], BB_err, seed_i = ff)

    NB_mag_dist = random_pdf(tab0[filt+'_MAG'][idx1], tab0[filt+'_MAG_ERROR'][idx1],
                             seed_i = ff+1)
    x_dist = cont_mag_dist - NB_mag_dist

    filt_dict = {'dNB': filt_dict0['dNB'][ff], 'dBB': filt_dict0['dBB'][ff],
                 'lambdac': filt_dict0['lambdac'][ff]}

    ew_dist, flux_dist = ew_flux_dual(NB_mag_dist, cont_mag_dist, x_dist, filt_dict)

    flux_err, flux_xpeak = compute_onesig_pdf(np.log10(flux_dist), tab0[filt+'_FLUX'][idx1])

    ew_err, ew_xpeak = compute_onesig_pdf(ew_dist, tab0[filt+'_EW'][idx1])

    tab0[filt+'_FLUX_UPERROR'][idx1] = flux_err[:,0]
    tab0[filt+'_FLUX_LOERROR'][idx1] = flux_err[:,1]

    tab0[filt+'_EW_UPERROR'][idx1] = ew_err[:,0]
    tab0[filt+'_EW_LOERROR'][idx1] = ew_err[:,1]

  return tab0
#enddef

def plot_flux_ew_errors(l_type, filt_ref, tab0):
  pp = PdfPages(path0+'NB_IA_emitters_'+l_type+'_ew_flux_errors.pdf')

  for filt,ff in zip(filt_ref,range(len(filt_ref))):
    idx = [xx for xx in range(len(tab0)) if l_type+'-'+filt in tab0['NAME'][xx]]

    if len(idx) > 0:
      fig, ax = plt.subplots(ncols=2)

      t_tab = tab0[idx]
      x0 = t_tab[filt+'_FLUX']
      y0 = np.sqrt((t_tab[filt+'_FLUX_UPERROR']**2 +
                    t_tab[filt+'_FLUX_LOERROR']**2)/2.0)
      ax[0].scatter(x0, y0, marker='o', color='blue', facecolor='none', s=10)

      ax[0].annotate(l_type+'-'+filt, [0.025,0.975], xycoords='axes fraction',
                     ha='left', va='top')
      ax[0].set_xlabel(r'$F_{\rm NB}$ [dex]')
      ax[0].set_ylabel(r'$\sigma$ [dex]')

      x1 = t_tab[filt+'_EW']
      y1 = np.sqrt((t_tab[filt+'_EW_UPERROR']**2 +
                    t_tab[filt+'_EW_LOERROR']**2)/2.0)
      ax[1].scatter(x1, y1, marker='o', color='blue', facecolor='none', s=10)

      ax[1].set_xlabel(r'EW [$\AA$]')
      ax[1].set_ylabel(r'$\sigma$')

      plt.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.98)
      fig.savefig(pp, format='pdf')
    #endif
  #endfor
  pp.close()
#enddef

def plot_errors(l_type, filt_ref, tab0, limit_dict):
  pp = PdfPages(path0+'NB_IA_emitters_'+l_type+'_photometric_errors.pdf')

  for filt,ff in zip(filt_ref,range(len(filt_ref))):
    idx = [xx for xx in range(len(tab0)) if l_type+'-'+filt in tab0['NAME'][xx]]

    if len(idx) > 0:
      fig, ax = plt.subplots()
      x0 = tab0[filt+'_MAG'][idx]
      y0 = tab0[filt+'_MAG_ERROR'][idx]
      ax.scatter(x0, y0, marker='o', color='blue', facecolor='none', s=10,
                 label='NB phot')

      y0_raw = tab0[filt+'_MAG_ERROR_RAW'][idx]
      ax.scatter(x0, y0_raw, marker='o', color='blue', facecolor='none',
                 s=5, label = 'NB phot (raw)')

      x1 = tab0[filt+'_CONT_MAG'][idx]
      y1 = tab0[filt+'_CONT_ERROR'][idx]
      ax.scatter(x1, y1, marker='o', color='green', facecolor='none', s=10,
                 label='Cont. phot')

      y1_raw = tab0[filt+'_CONT_ERROR_RAW'][idx]
      ax.scatter(x1, y1_raw, marker='o', color='green', facecolor='none', s=5,
                 label='Cont. phot (raw)')

      ax.annotate(l_type+'-'+filt, [0.025,0.975], xycoords='axes fraction',
                  ha='left', va='top')
      ax.set_xlabel('magnitude')
      ax.set_ylabel(r'$\Delta$ magnitude')

      max_y = np.max([max(y0),max(y1)])
      ax.set_ylim([0,max_y*1.05])

      x = np.arange(19,28,0.01)

      NB_err = error_from_limit(x, limit_dict['m_NB'][ff])
      ax.plot(x, NB_err, 'b--')

      BB_err = error_from_limit(x, limit_dict['m_BB'][ff])
      ax.plot(x, BB_err, 'g--')

      ax.legend(loc='lower right')
      plt.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.98)
      fig.savefig(pp, format='pdf')

  pp.close()
#enddef

def main(silent=False, verbose=True):
  '''
  Main function to derive errors from NB photometry

  Parameters
  ----------
  filter : str
    Name of filter: 'NB704', 'NB711', 'NB816', 'NB921', 'NB973'

  NB : array
    NB magnitudes on AB system

  sig_NB : array
    error on [NB]

  excess : array
    BB - NB color on AB system

  sig_excess : array
    error on [excess]

  silent : boolean
    Turns off stdout messages. Default: False

  verbose : boolean
    Turns on additional stdout messages. Default: True

  Returns
  -------
    flux
    sig_flux
    EW
    sig_EW

  Notes
  -----
  Created by Chun Ly, 13 December 2018
  '''

  if silent == False: log.info('### Begin main : '+systime())

  #Limiting magnitudes
  m_NB  = np.array([26.7134-0.047, 26.0684, 26.9016+0.057, 26.7894+0.041,
                    27.3928+0.032, 26.7088-0.109, 25.6917-0.051])
  m_BB1 = np.array([28.0829, 28.0829, 27.7568, 27.8933, 28.0829, 26.8250,
                    26.8250])
  m_BB2 = np.array([27.7568, 27.7568, 26.8250, 28.0829, 27.7568, 00.0000,
                    00.0000])

  cont_lim  = mag_combine(m_BB1, m_BB2, epsilon)

  limit_dict = {'m_NB': m_NB, 'm_BB': cont_lim}

  tab0, infile = get_data()
  tab0 = get_errors(tab0, filt_dict0, BB_filt, epsilon, limit_dict=limit_dict)

  outfile = infile.replace('.fits','.errors.fits')
  tab0.write(outfile, format='fits', overwrite=True)

  plot_errors('Ha', filt_ref, tab0, limit_dict)

  if silent == False: log.info('### End main : '+systime())
#enddef

def test_ew_flux():
  '''
  Code to check IDL NB excess flux/EW against Python NB excess flux/EW
  '''

  err_file = path0+'NB_IA_emitters.allcols.colorrev.errors.fits'
  data = fits.getdata(err_file)

  for ff in range(len(filt_ref)):
    filt0 = filt_ref[ff]
    NB = data[filt0+'_MAG']
    BB = data[filt0+'_CONT_MAG']
    x  = data[filt0+'_EXCESS']

    filt_dict = {'dNB': dNB[ff], 'dBB': dBB[ff], 'lambdac': lambdac[ff]}

    idx= [xx for xx in range(len(data)) if 'Ha-'+filt0 in data['NAME'][xx]]

    if len(idx) > 0:
      ew, flux = ew_flux_dual(NB[idx], BB[idx], x[idx], filt_dict)

      val1 = data[filt0+'_FLUX'][idx] - np.log10(flux)
      val2 = np.log10(data[filt0+'_EW'][idx]) - np.log10(ew)

      print(filt0, np.nanmin(val1), np.nanmax(val1), np.nanmin(val2), np.nanmax(val2))
  #endfor
#endfor
