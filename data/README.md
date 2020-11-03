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

The photometry for the sample is described in Shin et al. (2020) and a number
of papers, including [Ly et al. (2011)](https://doi.org/10.1088/0004-637X/735/2/91),
[Ly et al. (2014)](https://doi.org/10.1088/0004-637X/780/2/122), and
[Ly et al. (2016)](https://doi.org/10.3847/0067-0049/226/1/5).

It consists of Subaru Deep Field broad-band (_BVRi'z'_) and intermediate-band
(IA598, IA679, _zb_, _zr_) optical imaging data from Subaru/Suprime-Cam,
_U_-band imaging from KPNO/MOSAIC, near-infrared imaging from UKIRT/WFCAM
(_K_) and KPNO/NEWFIRM (_J_, _H_), and _GALEX_ imaging in FUV and NUV bands.

The photometric data consist of 15 bands and fluxes and uncertainties are
reported as micro-Janskies (10^{-29} erg/s/cm2/Hz).

Flux columns are preceded with the "f_" prefix and their associated uncertainties
have a "e_" prefix.

For instances without photometric data, the `f_` and `e_` values are -99.0.
For instances where the galaxy is not detected at 3-sigma for a given band,
we use the 1.5-sigma flux limit for both the flux and its uncertainty.

The `redshift` column indicate what redshift is used in the FAST SED fitting.
It uses spectroscopic redshift where available and the NB-based redshift otherwise.

| No.    | Column Name            | Description  |
|-------:|:-----------------------|:-------------|
|      1 | `NB_ID`                | NB numerical identifier based on [Ly et al. (2014)](https://iopscience.iop.org/article/10.1088/0004-637X/780/2/122) sample of 9,264 emission-line galaxies |
|  2,  3 | `f_U`, `e_U`           | _U_-band fluxes and uncertainties from KPNO 4-m Mayall MOSAIC imaging |
|  4,  5 | `f_B`, `e_B`           | _B_-band fluxes and uncertainties from Subaru/Suprime-Cam imaging |
|  6,  7 | `f_V`, `e_V`           | _V_-band fluxes and uncertainties from Subaru/Suprime-Cam imaging |
|  8,  9 | `f_R`, `e_R`           | _R_c_-band fluxes and uncertainties from Subaru/Suprime-Cam imaging |
| 10, 11 | `f_I`, `e_I`           | SDSS _i'_-band fluxes and uncertainties from Subaru/Suprime-Cam imaging |
| 12, 13 | `f_Z`, `e_Z`           | SDSS _z'_-band fluxes and uncertainties from Subaru/Suprime-Cam imaging |
| 14, 15 | `f_IA598`, `e_IA598`   | IA598-band fluxes and uncertainties from Subaru/Suprime-Cam imaging |
| 16, 17 | `f_IA679`, `e_IA679`   | IA679-band fluxes and uncertainties from Subaru/Suprime-Cam imaging |
| 18, 19 | `f_K`, `e_K`           | _K_-band fluxes and uncertainties from UKIRT/WFCAM imaging |
| 20, 21 | `f_ZB`, `e_ZR`         | _zb_-band fluxes and uncertainties from Subaru/Suprime-Cam imaging |
| 22, 23 | `f_ZR`, `e_ZR`         | _zr_-band fluxes and uncertainties from Subaru/Suprime-Cam imaging |
| 24, 25 | `f_J`, `e_J`           | _J_-band fluxes and uncertainties from UKIRT/WFCAM imaging |
| 26, 27 | `f_H`, `e_H`           | _H_-band fluxes and uncertainties from KPNO 4-m Mayall NEWFIRM imaging |
| 28, 29 | `f_NUV`, `e_NUV`       | _NUV_-band fluxes and uncertainties from _GALEX_ imaging |
| 30, 31 | `f_FUV`, `e_FUV`       | _FUV_-band fluxes and uncertainties from _GALEX_ imaging |
|     32 | redshift               | Redshift used in FAST SED fitting. Either spectra or NB-based redshift |
