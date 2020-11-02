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
    print("Reading : " + infile)
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
                      'EBV': 'EBV_gas',  # Case B: 2.86 and uses Cardelli+ 1989
                      'EBV_errs': 'EBV_gas_err',  # Assumes Case B: 2.86 and uses Cardelli+ 1989
                      'dust_errs': 'dust_corr_factor_err',  # dust_corr_factor = 0.4*A(Ha)
                      'NBIA_errs': 'log(F[obs]_NB)_err'}

    for col_key in rename_columns.keys():
        main_tab[col_key].name = rename_columns[col_key]

    # Remove unused/unnecessary columns and re-order
    out_columns = ['NB_ID', 'NB_Name', 'Ha_filt', 'zspec', 'log(Mstar)_FAST',
                   'NB_excess_flux_sigma', 'log(F[obs]_NB)', 'log(F[obs]_NB)_err',
                   'log(L[obs]_NB)', 'log(SFR[obs]_NB)', 'log(SFR[obs]_metal)',
                   'log(NB_filt_corr)', 'NII/Ha', 'NII_source', 'log(NII_corr)',
                   'EBV_gas', 'EBV_gas_err', 'A(Ha)', 'dust_corr_factor',
                   'dust_corr_factor_err', 'meas_errs']
    main_tab = main_tab[out_columns]

    # Write file
    outfile = join(root_path, catalog_outfile)
    print("Writing : " + outfile)
    main_tab.write(outfile, overwrite=True, format='ascii.fixed_width_two_line')

    # Get photometric data

    # Get spectroscopic H-beta fluxes and uncertainties where available


if __name__ == '__main__':
    main()
