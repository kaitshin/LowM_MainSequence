from os.path import join

from astropy.io import fits
from astropy.io import ascii as asc

# local paths. These may require changes
root_path = '/Users/cly/GoogleDrive/Research/NASA_Summer2015'

# Main catalog containing NB information
catalog_file = 'Main_Sequence/mainseq_corrections_tbl.txt'

# This is the output catalog from catalog_infile
catalog_outfile = 'Distribution/Shin2020_catalog_primary.tbl'


def main():
    """
    Purpose:
      This is the primary code to aggregate data for easier dissemination
    """

    # Get catalog-level H-alpha fluxes and uncertainties, and SFRs
    infile = join(root_path, catalog_file)
    print("Reading : %s ".format(infile))
    main_tab = asc.read(infile)

    # Rename columns, key pair set
    # These columns are set/defined in the mainseq_corrections.py code
    rename_columns = {'ID': 'NB_ID',
                      'NAME0': 'NB_Name',
                      'filt': 'Ha_filt',
                      'zspec0': 'zspec',
                      'stlr_mass': 'log(Mstar)_FAST',
                      'flux_sigma': 'NB_excess_flux_sigma',  # Excess flux sigma selection. Selection >= 3.0
                      'obs_fluxes': 'log(F[obs]_NB)',  # NB observed flux
                      'obs_lumin': 'log(L[obs]_NB)',   # NB observed luminosity using spec or phot redshifts
                      'obs_sfr': 'log(SFR[obs]_NB)',   # NB observed SFRs assuming Chabrier IMF with 1.8 conversion
                      'met_dep_sfr': 'log(SFR[obs]_metal)',  # Metallicity-dependent SFR. See Eq. 13 in Shin+ 2020
                      'filt_corr_factor': 'log(NB_filt_corr)',  # NB filter correction. These are to be added
                      'nii_ha_corr_factor': 'log(NII_corr)',  # log(1/(1+NII/Ha). Neg. values means subtract
                      'NII_Ha_ratio': 'NII/Ha',  # NII/Ha flux ratio (not logarithmic)
                      'ratio_vs_line': 'NII_source',  # Either 'ratio' (from spectra) or 'line' (from best fit)
                      'EBV': 'EBV_gas',  # Assumes Cardelli+ 1989
                      'EBV_errs': 'EBV_gas_err',
                      'dust_errs': 'dust_corr_factor_err',  # dust_corr_factor = 0.4*A(Ha)
                      'NBIA_errs': 'L(F[obs_NB]_err'}

    for col_key in rename_columns.keys():
        main_tab[col_key].name = rename_columns[col_key]

    # Remove unused/unnecessary columns

    # Write file
    main_tab.write(join(root_path, catalog_outfile), overwrite=True,
                   format='ascii.fixed_width_two_line')

    # Get photometric data

    # Get spectroscopic H-beta fluxes and uncertainties where available

