# Analysis scripts (reference)

One-off analysis and figure scripts used to characterize the Alcor OMEA 8C
photometry and derive its calibration. These are **not** part of the installed
package and are not on `PYTHONPATH`; they are kept here for reference so the
analyses behind the calibration decisions can be re-run or revisited.

They have hardcoded local data paths (the two calibration nights
`/Users/tim/MMT/skycam_data/2024-09-04` and `/Volumes/Samsung_4TB/skycam/2026-05-18`,
each holding combined `*_phot.csv` from `alcor_star_photometry --both`) and write
figures to `/tmp/gplots/`. Adjust the path constants at the top of each before
re-running elsewhere.

## Calibration

- **`zeropoint_calib.py`** — the adopted calibration. Fits per-band zeropoint +
  B−V color term (aperture mags, single 0.40 mag/airmass term, −11.5 bright cut)
  mapping instrument R/G/B → catalog Johnson R/V/B. Confirms G→V is color-flat
  while R and B need color terms; zeropoints stable ~0.03 mag across both nights.
- **`ensemble_extinction.py`** — per-star ensemble extinction `k` (slope of
  instrument mag vs Kasten-Young airmass, 3σ-clipped); writes
  `ensemble_extinction_<night>.csv` per night dir.

## CMOS non-linearity characterization

- **`make_fig1_both.py`** — extinction `k` vs peak instrumental mag, aperture vs
  Gaussian on identical frames; locates the (estimator-independent) −11.5 knee.
- **`make_fig1clean.py`** — clean binned `k` vs peak instrumental mag (per band),
  showing the R/G/B collapse onto a common knee.
- **`make_fig1redo.py`** — `k` vs instrumental mag per band, aperture and Gaussian
  overlaid.
- **`gaussian_vs_aperture.py`** — direct aperture-vs-Gaussian extinction comparison
  (established Gaussian does not fix the bright-star suppression).
- **`radius_test.py`** — aperture-radius experiment that refuted the fit-window
  truncation hypothesis (plateau insensitive to radius 4→10).
- **`nonlin_beta.py`, `nonlin_correction.py`, `nonlin_deficit2.py`** — attempts at
  an explicit non-linearity correction (hyperbolic β(flux) etc.); parked, see the
  calibration note — bright-star non-linearity is not cleanly recoverable from
  aperture-sum photometry.

## General visualization

- **`make_plots.py`, `make_plots2.py`** — the original four-figure visualization set
  (k-vs-V, FWHM-vs-airmass, light curves, radius/undersampling).
