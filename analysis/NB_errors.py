"""
NB_errors
====

Compute flux and EW errors based on NB and broadband photometry
"""

import sys, os

from chun_codes import systime, match_nosort

from astropy.io import ascii as asc
from astropy.io import fits
from astropy.table import Table, Column

import numpy as np

from glob import glob
from astropy import log

m_AB = 48.6
def fluxline(y, fNB, dNB, dBB):
  '''RETURNS emission-line flux in erg/s/cm^2'''

  return dNB*fNB*(1.0 - y)/(1.0-dNB/dBB)
#enddef

def mag_combine(m1, m2, epsilon):
  cont_flux = epsilon * 10**(-0.4*(m1+m_AB)) + (1-epsilon)*10**(-0.4*(m2+m_AB))
  cont_mag  = -2.5*np.log10(cont_flux) - m_AB
  return cont_mag
#enddef

def get_data():
  path0 = '/Users/cly/Google Drive/NASA_Summer2015/Catalogs/'

  infile = path0 + 'NB_IA_emitters.allcols.colorrev.fits'
  print("Reading : "+infile)

  data0 = fits.getdata(infile)

  tab0 = Table(data0)

  return tab0
#enddef

def get_errors(tab0, filt_ref, BB_filt):

  NB_path = '/Users/cly/data/SDF/NBcat/'

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
    c1 = Column(np.zeros(n_gal), name=filt+'_CONT_ERROR')
    c2 = Column(np.zeros(n_gal), name=filt+'_EW_ERROR')
    c3 = Column(np.zeros(n_gal), name=filt+'_FLUX_ERROR')

    colnames = tab0.colnames
    idx_end = [xx+1 for xx in range(len(colnames)) if colnames[xx] == filt+'_MAG']
    # +1 to add at end
    tab0.add_columns([c0,c1,c2,c3], indexes=idx_end * 4)

    print("Reading : "+NB_phot_files[ff])
    phot_tab    = asc.read(NB_phot_files[ff])
    NB_id       = phot_tab['col1']
    MAGERR_APER = phot_tab['col15']

    NBem = np.where(tab0[filt+'_ID'] != 0)[0]
    idx1, idx2  = match_nosort(tab0[filt+'_ID'][NBem], NB_id)
    print('index size : '+str(len(NBem))+', '+str(len(idx2)))
    tab0[filt+'_MAG_ERROR'][NBem[idx1]] = MAGERR_APER[idx2]

    print("Reading : "+BB_phot_files1[ff])
    phot_tab    = asc.read(BB_phot_files1[ff])
    BB_MAGERR_APER = phot_tab['col15']

    tab0[filt+'_CONT_ERROR'][NBem[idx1]] = BB_MAGERR_APER[idx2]

  return tab0
#enddef

def main(filter, NB, sig_NB, excess, sig_excess, silent=False, verbose=True):
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

  filt_ref= ['NB704', 'NB711', 'NB816', 'IA598', 'IA679','NB921','NB973']
  dNB     = [  100.0,    72.0,   120.0,   303.0,   340.0,  132.0,  200.0]
  lambdac = [ 7046.0,  7111.0,  8150.0,  6007.0,  6780.0, 9196.0, 9755.0]
  dBB     = [ 1110.0,  1110.0,  1419.0,   885.0,  1110.0,  956.0,  956.0] # R R i, V, R

  BB_filt = {'one': ['R','R','i','V','R','z', 'z'], 'two':['i','i','z','R','i','', '']}
  tab0 = get_data()

  tab0 = get_errors(tab0, filt_ref, BB_filt)

  if silent == False: log.info('### End main : '+systime())
#enddef

