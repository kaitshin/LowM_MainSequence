---
# README Documentation for Distributed Data

Distribution of data for "Metal Abundances across Cosmic Time (MACT) Survey.
III. The Relationship between Stellar Mass and Star Formation Rate in
Extremely Low-Mass Galaxies", is provided by direct requests to either the
first (Shin) or second author (Ly). This README.md provides metadata and other
information that describes and document the data in this distribution.

---
## Citation

Use of any data in this distribution *requires* the citation of the following:

Shin, K.; Ly, C.; Malkan, M. A.; Malhotra, S.; de los Reyes; Mithi; Rhoads, J. E.;
["The Metal Abundance across Cosmic Time (MACT) Survey. III. The Relationship between Stellar Mass and Star Formation Rate in Extremely Low-Mass Galaxies"](https://doi.org/10.1093/mnras/staa3307),
2020, _Monthly Notices of the Royal Astronomical Society_, in press, 
[arXiv](https://arxiv.org/abs/1910.10735)

---
## Overview of Data

There are three separate files provided with this data distribution:
 1. Primary catalog data, `Shin2020_catalog_primary.tbl`
 2. Broad-band photometric data, `Shin2020_catalog_photometry.tbl`
 3. Ancillary spectroscopic data, `Shin2020_catalog_spec.tbl`


### Primary Catalog Data

The primary catalog data (PCD) include galaxy identification information,
narrow-band (NB) excess flux measurements, and properties (e.g., luminosity,
SFRs) derived from these observables. It includes spectroscopic information,
dust attenuation corrections based on spectroscopic measurements, and any
corrections used to derive H-alpha measurements from NB measurements.

The PCD is provided as an [Astropy](https://astropy.org) ASCII table
(`fixed_width_two_line` format).

The PCD table has 953 entries (galaxies) and contains 21 columns:

| No. | Column Name            | Description  |
|----:|:-----------------------|:-------------|
|  1  | `NB_ID`                | NB numerical identifier based on [Ly et al. (2014)](https://iopscience.iop.org/article/10.1088/0004-637X/780/2/122) sample of 9,264 emission-line galaxies |
|  2  | `NB_Name`              | Name of source based on NB catalog information |
|  3  | `Ha_filt`              | Filter with H-alpha emission |
|  4  | `zspec`                | Spectroscopic redshift. Set to -10 when spectroscopy is not available |
|  5  | `log(Mstar)_FAST`      | Logarithm of stellar mass in solar units from FAST fitting |
|  6  | `NB_excess_flux_sigma` | NB excess flux signal-to-noise. Selection is >= 3.0 |
|  7  | `log(F[obs]_NB)`       | Logarithm of observed NB excess emission-line flux in erg/s/cm^2 |
|  8  | `log(F[obs]_NB)_err`   | Error on No. 7 | 
|  9  | `log(L[obs]_NB)`       | Logarithm of observed NB excess emission-line luminosity in erg/s |
| 10  | `log(SFR[obs]_NB)`     | Logarithm of observed SFR derived from No. 9 using Chabrier (2003) IMF in M_sun/yr |
| 11  | `log(SFR[obs]_metal)`  | Logarithm of observed SFR derived using Chabrier (2003) IMF and metallicity-dependent conversion (Eq. 13, Shin+ 2020) |
| 12  | `log(NB_filt_corr)`    | Logarithm of NB non-tophat filter correction, Positive values mean addition |
| 13  | `NII/Ha`               | Logarithm of [NII]/H-alpha flux ratio |
| 14  | `NII_source`           | source of No. 13. 'ratio' = spectra, 'line' = best fit line |
| 15  | `log(NII_corr)`        | Logarithm of 1/(1+[NII]/H-alpha). Negative values means subtraction |
| 16  | `EBV_gas`              | Nebular gas color excess reddening assuming Case B (2.86) and Cardelli+ (1989) reddening curve |
| 17  | `EBV_gas_err`          | Error on No. 16 |
| 18  | `A(Ha)`                | Nebular gas attenuation at H-alpha in magnitude|
| 19  | `dust_corr_factor`     | 0.4 * A(Ha) |
| 20  | `dust_corr_factor_err` | Error on No. 19 |
| 21  | `meas_errs`            | ... |

To determine luminosities and SFRs, `zspec` values were used where available.
For instances without spectroscopic redshifts, the central redshift for the NB
filters were used. They are: 7045 (NB704), 7126 (NB711), 8152 (NB816), 9193 (NB921),
and 9749 Angstroms (NB973). These correspond to H-alpha redshift of 0.073 (NB704),
0.086 (NB711), 0.242 (NB816), 0.401 (NB921), and 0.485 (NB973).

### Photometric data

The photometry for the sample is described in Shin et al. (2020) and consists
of Subaru Deep Field broad-band imaging data from Subaru/Suprime-Cam, XX
