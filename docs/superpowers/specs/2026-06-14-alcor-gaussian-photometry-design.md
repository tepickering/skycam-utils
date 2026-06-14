# Alcor Gaussian-fit star photometry — design

**Date:** 2026-06-14
**Module:** `skycam_utils/alcor.py`
**Status:** approved design, ready for implementation plan

## Motivation

Ensemble extinction analysis established that the Alcor CMOS sensor is
signal-dependent non-linear: the brightest pixels of a star read low *before*
any pixel reaches the 15-bit saturation ceiling (32767), so aperture-sum flux is
suppressed for bright stars and the per-pixel `sat_*` flag does not catch it.
(See the `alcor-cmos-nonlinearity-extinction` memory note.) Aperture-sum
photometry cannot recover the true flux because the suppressed core is summed in
directly.

The fix is **profile fitting that lets the linear wings set the amplitude**. We
fit a Gaussian whose shape (center, width) is pinned from a high-S/N luminance
frame, mask out the non-linear core, and recover each channel's amplitude from
the unmasked linear wings only. The reported flux is the analytic integral of
the fitted model, so it integrates the *model* rather than the suppressed data.

## Approach (chosen: A — method switch)

Add a Gaussian path *inside* `alcor_star_photometry`, selected by a flag. The
catalog projection, bias subtraction, Sun-rejection, CSV writing, and check-plot
remain shared; only the per-star measurement branches. The new fitting logic
lives in a single testable helper parallel to the existing
`_aperture_annulus_photometry`. The CSV schema is unchanged except for one added
column (`fwhm`), so `collect_alcor_photometry` and all downstream extinction
scripts keep working — the Gaussian path is just a different way to fill
`flux_*`/`mag_*`.

Rejected alternatives: a separate `alcor_star_psf_photometry()` function (forces
duplicating or refactoring all the shared scaffolding, two entry points); a
post-processing pass over an existing `_phot.csv` (still needs the FITS, redoes
the WCS/catalog work, not what was asked).

## New module constant

```python
ALCOR_NONLINEAR_THRESHOLD = 15000   # raw ADU; per-pixel non-linearity onset
```

Defined next to `ALCOR_SATURATION = 32767`. Used as the default mask cutoff for
the Gaussian fit. Distinct from `ALCOR_SATURATION` because non-linearity sets in
well below the hard ceiling.

## Algorithm (per catalog star, `gaussian=True`)

The frame is loaded and bias-subtracted exactly as today (`data` = raw cube minus
per-channel corner bias). The catalog stars are projected to predicted pixel
positions `(x, y)` via the WCS as today. Then, replacing the per-star body of the
measurement loop:

### 1. Luminance shape fit

- Luminance frame `L = data[0] + data[1] + data[2]` (the three *bias-subtracted*
  channels summed, for maximum S/N on the shape).
- Subtract the annulus-median background of `L`, measured at the predicted center
  `(x, y)` with the existing aperture/annulus geometry.
- **Linearity mask:** a pixel is usable for the luminance fit only if *none* of
  the three **raw** channel values at that pixel reaches `mask_threshold`
  (default `ALCOR_NONLINEAR_THRESHOLD`). Because `L` sums the channels, a single
  saturated/non-linear channel corrupts the luminance core, so the mask is the
  logical-AND of all three channels being linear. Masking is computed on the
  **raw** cube (the threshold is a raw-ADU level), consistent with
  `_aperture_saturated`.
- Fit a circular 2-D Gaussian
  `f(x, y) = A · exp(-((x-x0)² + (y-y0)²) / (2σ²))`
  over the **unmasked pixels within `aperture_radius`** of the predicted center.
  Free parameters `[A, x0, y0, σ]`, seeded `A = max(L in aperture)`,
  `(x0, y0) = (x, y)`, `σ = aperture_radius / 2`. Solved with
  `scipy.optimize.least_squares` (already imported), bounds `A > 0`,
  `0 < σ < aperture_radius`, center within `aperture_radius` of the seed.
- Outputs the shared center `(x0, y0)` and width `σ`.

### 2. Per-channel amplitude (linear, closed form)

For each channel, with the shape `(x0, y0, σ)` fixed:

- Recenter to `(x0, y0)`; subtract that channel's annulus-median background
  measured at the new center.
- The amplitude is a linear least-squares projection — no iteration:
  `A_ch = Σ(g · d) / Σ(g²)`
  over that channel's **unmasked** pixels within `aperture_radius`, where
  `g = exp(-r²/2σ²)` is the unit profile and `d` is the bg-subtracted channel
  data. For a bright star the only unmasked pixels are the linear wings, so they
  alone set `A_ch`; the suppressed core is excluded.
- Per-channel mask: that channel's **raw** value `< mask_threshold` within the
  aperture.

### 3. Flux and magnitude

- `flux_ch = A_ch · 2π σ²` — the analytic Gaussian integral over the plane
  (aperture-independent total flux; integrates the model, not the suppressed
  data). `σ` is shared across channels.
- `mag_ch = -2.5 · log10(flux_ch)`.

### Derived/stored values

- `xcen`, `ycen` = the fitted luminance center `(x0, y0)` (more accurate than the
  WCS prediction).
- `fwhm` = `2.3548 · σ` (pixels), shared across channels.
- `background_ch` = the per-channel annulus median subtracted at the recentered
  position.
- `sat_ch` = the raw saturation flag (`_aperture_saturated` on the raw channel at
  the fitted center, raw pixel `>= saturation`), unchanged — now informational,
  since the Gaussian recovers flux even when the core is saturated.

## Interface

### `alcor_star_photometry` parameters (additions)

- `gaussian=False` — select the Gaussian path.
- `mask_threshold=ALCOR_NONLINEAR_THRESHOLD` — raw-ADU cutoff above which a pixel
  is excluded from the fit. Lower it to mask deeper into the non-linear regime;
  raise it toward `saturation` to fit more of the core.

All existing parameters keep their current defaults and meaning. In Gaussian
mode, `aperture_radius` additionally defines the fit window.

### CLI (`alcor_star_photometry_cli`)

- `--gaussian` (store_true) — use Gaussian-fit photometry instead of aperture.
- `--mask-threshold` (float, default `ALCOR_NONLINEAR_THRESHOLD`) — raw-ADU mask
  cutoff for the Gaussian fit.
- `--aperture-radius` help text notes it also sets the Gaussian fit window.

## Output schema

Unchanged column set plus one new column `fwhm`:

```
altitude, azimuth, xcen, ycen, fwhm,
flux_r, mag_r, background_r, sat_r,
flux_g, mag_g, background_g, sat_g,
flux_b, mag_b, background_b, sat_b
```

`fwhm` is populated in Gaussian mode and `NaN` in aperture mode, giving a single
stable schema for both methods. Rows are still sorted by descending `flux_g`;
the CSV is always written (per the existing contract). `collect_alcor_photometry`
concatenates columns, so the added `fwhm` carries through transparently and
downstream column-selecting code is unaffected.

## Error handling

A star is dropped (`continue`, leaving it out of the output — same contract as
the existing "non-finite or non-positive flux" rule) when:

- the luminance fit fails to converge, or returns σ non-finite, `σ <= 0`, or
  `σ >= aperture_radius`;
- the fitted center drifts more than `aperture_radius` from the WCS prediction
  (blend / noise spike);
- there are too few unmasked pixels for a fit — luminance `< 8` (4-parameter
  fit), or any single channel `< 3`;
- any channel yields `A_ch <= 0` or non-finite flux.

These mirror and extend the aperture path's existing drop rule, so a rejected
star simply does not appear in the CSV.

## Testing

TDD, following the existing monkeypatched-synthetic-frame pattern in
`skycam_utils/tests/test_alcor.py` (patch `load_alcor_fits` and the named
catalog, build a small synthetic cube).

1. **Clean recovery.** A synthetic clean Gaussian star (no clipping): recovered
   `flux_*` within a few percent of the injected integral, center within 0.1 px,
   `fwhm` within ~5% of truth.
2. **Non-linearity recovery (the point of the feature).** Inject a star, then
   clamp its core pixels to a ceiling to mimic per-pixel non-linearity. Assert
   the Gaussian-recovered flux is closer to the true *unclipped* integral than
   the aperture-sum flux is.
3. **Schema/flag.** `gaussian=True` returns the expected columns including
   `fwhm`, and writes the CSV.
4. **Drop on failure.** A signal-free / fully-masked star is absent from the
   output (not emitted with junk values).
5. **Per-channel amplitude helper.** The closed-form linear projection recovers a
   known amplitude given a fixed unit profile.

## Documentation

Update `CLAUDE.md` (which `AGENTS.md` symlinks to):

- The Alcor paragraph: describe the Gaussian-fit path, the luminance shape fit,
  core masking at `ALCOR_NONLINEAR_THRESHOLD`, wing-driven per-channel amplitude,
  analytic-integral flux, and the new `fwhm` column.
- The `alcor_star_photometry` CLI block: add `--gaussian` and `--mask-threshold`,
  and note that `--aperture-radius` sets the Gaussian fit window.

## Out of scope

- Elliptical / rotated PSF (the design fits a single circular width, as
  specified). A future extension if field-dependent elongation proves important.
- Any change to the aperture path's numbers or the existing CSV values.
- Recalibrating extinction with the new fluxes — a separate analysis step.
