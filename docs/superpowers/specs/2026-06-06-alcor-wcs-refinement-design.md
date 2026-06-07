# Alcor WCS refinement — design

**Date:** 2026-06-06
**Component:** `skycam_utils/alcor.py`

## Goal

The WCS built in `load_alcor_fits` is an idealized ARC (zenith-equidistant)
projection: pixel radius from zenith is exactly linear in zenith angle
(`cdelt = 90 / horizon_radius`). Real fisheye lenses deviate from that, and the
deviation grows with zenith angle. Replace the idealized mapping with a fitted
lens model — refined center, rotation, and a **non-linear radial term in zenith
angle** — calibrated against the `bright_star_sloan.fits` catalog (Vmag ≤ 3).

The returned object stays a standard `astropy.wcs.WCS`, so all downstream
consumers (`plot_alcor_fits`, `alcor_proc_fits`, `alcor_keogram`) are unaffected
and benefit transparently. This mirrors the calibration approach used for the
old stellacam system (`astrometry.initial_wcs_fit` → `astrometry.wcs_sip_fit`).

## The model

For a star at zenith angle `z = 90° − alt` and azimuth `A`:

```
rho   = r_pix / horizon_radius                # normalized detector radius, 0 at zenith
z     = 90·(k1·rho + k3·rho³ + k5·rho⁵)        # zenith angle from detector radius (plate solution)
x = center_x + xshift - r_pix·sin(A + rot)
y = center_y + yshift + r_pix·cos(A + rot)
```

The radial polynomial is **odd-power only** and written in SIP's native
direction — **detector radius → zenith angle** (the plate solution):
`radial_coeffs = (k1, k3, k5)` are the coefficients of ρ, ρ³, ρ⁵. Because the
detector→sky map is itself an odd polynomial, SIP reproduces it to numerical
precision everywhere (degree 5 captures the ρ⁵ term exactly). Predicting a
pixel from a known alt/az (for star matching) inverts the polynomial with a few
vectorized Newton steps. Fit parameters: `xshift, yshift, rot, k3, k5` (`k1`
held at 1.0, since the zenith plate scale is carried by `horizon_radius`). These
few interpretable numbers are the baked-in constants — consistent with the
"coefficients in code" delivery choice.

Sign/orientation conventions follow the existing north-up, zenith-centered
processed frame produced by `load_alcor_fits`.

## Calibration pipeline (new, offline)

New function `fit_alcor_wcs(...)` in `alcor.py`, with a CLI entry point.

1. **Frame selection.** Given a directory of alcor FITS, read each frame's
   `DATE-OBS`, compute the Sun's altitude at `MMT_LOCATION`, and keep only
   frames with **Sun altitude < −18°** (astronomical twilight / dark sky). This
   selects the usable dark-sky window from a noon→dawn directory automatically.
2. **Star detection.** For each selected frame, load via `load_alcor_fits`
   (processed: zenith-centered, north-up, RGB). Detect stars on a luminance
   combination of the RGB planes, reusing the existing `photometry` detection
   path (`make_background` → `make_segmentation_image` → `make_catalog`) or
   `DAOStarFinder`. Centroids are in the processed-frame pixel coordinates that
   the WCS describes.
3. **Catalog prep.** Load `bright_star_sloan.fits`, filter to Vmag ≤ 3, and
   transform RA/Dec → Alt/Az at the frame's `DATE-OBS` from `MMT_LOCATION`,
   **including atmospheric refraction** (astropy `AltAz` with pressure /
   temperature). The frame headers leave `PRESSURE`/`TEMPEXT` at `0.0`, so use a
   nominal MMT-site pressure (~0.75 atm for the 2600 m elevation) and a
   temperature of 10 °C rather than the header values. Refraction matters because
   it is largest at high zenith angle —
   exactly where the radial distortion is largest — so omitting it would
   contaminate the radial term.
4. **Matching (bootstrap from zenith outward).** A fixed pixel tolerance is not
   safe at large zenith angle: the idealized equidistant model can mispredict
   positions by tens of pixels there — exactly the stars needed to constrain the
   non-linear term — so a loose global tolerance would invite mismatches in the
   sparse Vmag ≤ 3 field (~103 stars over the whole sky).

   Instead, exploit that the model is accurate near zenith and degrades smoothly
   with `z`:
   - Match only stars within a small zenith-angle cutoff (e.g. `z < 20°`), where
     a tight tolerance is safe, and fit the low-order terms.
   - Expand the `z` cutoff in steps, re-predicting and re-matching with the
     progressively improved model each round; the match radius grows only as far
     as the current model's residuals require.
   - Use detected flux rank vs. catalog Vmag rank as a **brightness tie-break**
     when more than one candidate falls within tolerance.
   - Iterate match → fit → re-match with outlier rejection.

   **Fallback.** If residuals show the bootstrap still mismatches at the edge,
   escalate to local asterism / triangle (distortion-invariant) pattern matching
   (astrometry.net style). Heavier to implement and less reliable in this sparse
   field, so only if needed.
5. **Fit.** `scipy.optimize.least_squares` over the six parameters, minimizing
   pixel residuals over the matched stars.
6. **Aggregate across the night.** Pool matched stars from all dark-sky frames
   so the fit sees full azimuth and zenith-angle coverage, then fit once on the
   combined set.
7. **Diagnostics.** Report residual RMS and a residual-vs-zenith-angle plot
   (before/after), and print the baked-in constants.

## WCS representation

The returned WCS stays ARC with `crpix` at the array center. The radial
polynomial is encoded as **SIP distortion** generated from the fitted constants:
a one-time fit of SIP terms (via `fit_wcs_from_points` over a synthetic grid of
the forward model). This generation is `@lru_cache`d on the geometry parameters,
so it runs once per process — never per image. Encoding as SIP keeps
`world_to_pixel_values()` and `to_header()` working unchanged, which is what the
downstream consumers rely on.

The fitted center offset and rotation are applied in the image itself rather
than the WCS: `load_alcor_fits` rotates by the refined `rotation` and recenters
the zenith onto the array center with a `scipy.ndimage` shift (`xshift`/`yshift`).
Keeping the zenith at the array center preserves the assumptions that the keogram
center column is the zenith meridian and that the plot's horizon circle is
centered — while the WCS `crpix` stays at the center and only the radial
distortion lives in SIP.

## Integration into `load_alcor_fits`

The fitted values become the new **defaults** for `load_alcor_fits`: existing
`xcen`/`ycen`/`radius`/`horizon_radius` kwargs stay; refined `rotation` and the
new radial coefficients (and refined `crpix`) become defaults, all overridable.
Calling `load_alcor_fits` on current data Just Works with the improved WCS.

## Validation & testing

- Residual-vs-zenith-angle plot and RMS before/after, demonstrating the
  non-linear term is captured (RMS should drop, and the systematic trend with
  zenith angle should flatten).
- Unit test (using the provided frame
  `2024-09-04/2024_09_05__00_00_36.fits.bz2`) asserting `world_to_pixel` →
  `pixel_to_world` round-trip consistency and that matched-star residuals stay
  under a threshold.

## Files changed

- `skycam_utils/alcor.py` — new `fit_alcor_wcs()` + CLI (`fit_alcor_wcs_cli`),
  radial→SIP helper, updated `load_alcor_fits` defaults.
- `pyproject.toml` — new `[project.scripts]` entry for the CLI.
- `skycam_utils/tests/` — new test for the WCS round-trip / residuals.

## Out of scope

- No change to the keogram, plotting, or FITS-output APIs themselves — they
  inherit the improved WCS automatically.
- No per-image re-fitting at runtime; calibration is offline, constants are
  baked.
