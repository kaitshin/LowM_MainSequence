from analysis.cardelli import *  # k = cardelli(lambda0, R=3.1)
from create_ordered_AP_arrays import create_ordered_AP_arrays as create_ordered_AP_arrays_fn

FULL_PATH = '/Users/kaitlynshin/Google Drive/NASA_Summer2015/'
fileend='.GALEX'

NB_IA_emitters_cat = 'Catalogs/NB_IA_emitters.nodup.colorrev2.fix.fits'
allcols_cat        = 'Catalogs/NB_IA_emitters.allcols.colorrev.fits'
allcols_errs_cat   = 'Catalogs/NB_IA_emitters.allcols.colorrev.fix.errors.fits'

# emission line wavelengths (air)
HG_VAL = 4340.46
HB_VAL = 4861.32
HA_VAL = 6562.80
OII3727_VAL = (3726.16+3728.91)/2.0
OIII5007_VAL = 5006.84

CUTOFF_SIGMA = 4.0
CUTOFF_MASS = 6.0

k_hg = cardelli(HG_VAL * u.Angstrom)
k_hb = cardelli(HB_VAL * u.Angstrom)
k_ha = cardelli(HA_VAL * u.Angstrom)

inst_dict = {}
inst_dict['MMT'] = ['MMT,FOCAS,','MMT,','merged,','MMT,Keck,','merged,FOCAS,']
inst_dict['Keck'] = ['merged,','Keck,','Keck,Keck,','Keck,FOCAS,',
                     'Keck,FOCAS,FOCAS,','Keck,Keck,FOCAS,','merged,FOCAS,']

data_dict = create_ordered_AP_arrays_fn()