from analysis.cardelli import *  # k = cardelli(lambda0, R=3.1)
from create_ordered_AP_arrays import create_ordered_AP_arrays as create_ordered_AP_arrays_fn

FULL_PATH = '/Users/kaitlynshin/Google Drive/NASA_Summer2015/'
fileend='.GALEX'

# pre-existing files to read in
NB_IA_emitters_cat = 'Catalogs/NB_IA_emitters.nodup.colorrev2.fix.fits'
allcols_cat        = 'Catalogs/NB_IA_emitters.allcols.colorrev.fits' # -> colorrev2.fix.fits
allcols_errs_cat   = 'Catalogs/NB_IA_emitters.allcols.colorrev.fix.errors.fits'

emagcorr_cat       = 'Catalogs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+fileend+'.cat'  ##
zspec_cat          = 'Catalogs/nb_ia_zspec.txt'  ##
fout_cat           = 'FAST/outputs/NB_IA_emitters_allphot.emagcorr.ACpsf_fast'+fileend+'.fout'  ##
Ha_corrs_cat       = 'Main_Sequence/Catalogs/mainseq_Ha_corrections'+fileend+'.fits'  ##

# generated files read in by multiple scripts
mainseq_corrs_tbl  = 'Main_Sequence/mainseq_corrections_tbl.txt'  ##

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

centr_filts = {'NB7':((7045.0/HA_VAL - 1) + (7126.0/HA_VAL - 1))/2.0, 
	'NB816':8152.0/config.HA_VAL - 1, 'NB921':9193.0/HA_VAL - 1, 'NB973':9749.0/HA_VAL - 1,
    'NEWHA':0.8031674}

# also reads in 'Main_Sequence/Catalogs/{MMT,Keck,merged}/{*}_line_fit.fits'
data_dict = create_ordered_AP_arrays_fn()
