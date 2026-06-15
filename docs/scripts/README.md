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

### Bright-star deficit / linear-regime investigation (2026-06)

Characterized the bright-star magnitude *deficit* (measured − catalog-true, both
top-of-atmosphere instrumental mags) as a function of signal level, to find how
far the calibration stays usable into the bright regime. Pooled both calibration
nights against the adopted `ALCOR_ZEROPOINTS`. Plots in `../gplots/`. The deficit
is computed as `(mag_inst − k·X) − (catalog_band − zp − color·(B−V))`.

The runs below are the investigation trail; **`nonlin_binned.py` is the
definitive one** — the others established that per-frame data is too noisy to fit
directly.

- **`nonlin_linear.py`** — deficit vs **measured** instrumental mag. Confirms the
  deficit is ≈0 fainter than the knee and rises toward bright, but per-frame data
  shows a huge star-dependent fan (the intra-pixel-sensitivity × undersampled-PSF
  jitter).
- **`nonlin_peak.py`** — deficit vs an estimated **peak** mag `flux/(2π σ²)` from
  the luminance FWHM. Fails: the undersampled FWHM is too noisy per frame and
  *increases* the scatter (rms 0.38–0.59). The peak proxy is not recoverable
  per-frame.
- **`nonlin_truemag.py`** — deficit vs **catalog-true** mag (errors-in-variables
  fix) plus a per-night FWHM diagnostic. Per-frame fits are still unstable; the
  scatter, not a band/night law, dominates.
- **`nonlin_combined.py`** — pools all three bands and both nights for a single
  signal-level slope (the effect is band-independent: at a given instrumental mag
  the aperture holds the same counts regardless of Bayer channel). Confirms the
  bands track together, but the raw per-frame fan (rms ≈ 0.8) and a persistent
  blend-contamination tail still swamp a clean fit.
- **`nonlin_binned.py`** — **the result.** Beats the intra-pixel jitter down by
  taking a robust **15-minute per-star median** (a star drifts across several
  pixels in 15 min, averaging the sub-pixel-phase response, while airmass — hence
  signal level — barely moves). Pools all bands + both nights, clips the blend
  tail (deficit < −0.35), pins the knee. Result: the deficit collapses onto a
  single, band- and night-consistent curve with **MAD ≈ 0.11–0.14 mag** — flat
  fainter than −11.5, **gently linear (≈0.15–0.20 mag/mag) out to ≈ −12.5**, then
  accelerating steeply (≈0.4, 0.9, 1.4 mag by −13.1/−13.3) into a sparse,
  non-monotonic regime fed by only a handful of stars. **Practical outcome:
  `ALCOR_BRIGHT_CUT` widened from −11.5 to −12.5** (the linear-regime cutoff);
  brighter than −12.5 is dropped pending more calibration nights to pin the steep
  shoulder. No explicit deficit-correction function is applied within the
  −11.5…−12.5 range — it is small enough to treat as linear.

## General visualization

- **`make_plots.py`, `make_plots2.py`** — the original four-figure visualization set
  (k-vs-V, FWHM-vs-airmass, light curves, radius/undersampling).
