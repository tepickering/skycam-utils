# Alcor WCS-Driven `load_alcor_fits` Refactor — Design

**Date:** 2026-06-10
**Status:** Design (pending review)

## Motivation

`load_alcor_fits` currently resamples pixels: it transposes the raw `(3, ny, nx)`
cube to `(ny, nx, 3)`, trims a square around the illuminated region, rotates
(`scipy.ndimage.rotate`) to remove camera tilt, shifts (`ndimage_shift`) to put
the zenith at the array center, and `flipud`s to make north up. It then returns a
clean array-centered ARC WCS. The geometry (rotation, zenith offset, radial
distortion) is spread across loose function parameters and an in-code
`ALCOR_CALIBRATIONS` table whose constants are expressed in that resampled frame.

This resampling alters the pixel data, complicates every consumer, and means the
WCS only describes the *post-resampling* frame. We want the **WCS to be the single
source of geometry** and the pixel data left untouched.

Key facts that make this clean:

- The raw `.fits.bz2` files already display correctly as image cubes in DS9, in
  native orientation, with each color channel (0=R, 1=G, 2=B) in its own plane.
- The `flipud` existed only because matplotlib defaults to `origin="upper"`; using
  `origin="lower"` in the plotter reproduces the orientation with no array flip.

So `load_alcor_fits` can return the raw cube as-is plus a WCS that carries the
zenith offset (CRPIX), rotation (CD/PC matrix), and radial distortion (SIP).

## Scope

Full sweep, one coherent change (nothing left half-broken):

- `load_alcor_fits` (thin, WCS-driven, no resampling).
- `build_alcor_wcs` + `ALCOR_CALIBRATIONS` schema (raw-frame geometry).
- Forward model (`_predict_pixels`), matcher (`assign_alcor_matches`),
  fitter (`fit_alcor_wcs`, `_detect_alcor_frame`) — all in the raw frame.
- Re-fit and replace the single 2024-09-04 calibration epoch in the new schema.
- `plot_alcor_fits` and `alcor_proc_fits` (presentation transform moves into the
  plotter; proc writes the raw cube + WCS).
- CLIs, tests, and the CLAUDE.md alcor paragraph.

### Out of scope (deferred)

- Per-channel bias subtraction. `load_alcor_fits` no longer subtracts the `-2000`
  pedestal or clips negatives; it returns raw counts. Bias will become a separate
  standalone function that measures it per-channel from the dark image corners.
- The structure mask (buildings + rotating observatory) and the combined
  photometry mask.

## Architecture

### 1. `load_alcor_fits` — thin, WCS is the geometry

```python
load_alcor_fits(filename, wcs=None, badpix="repair", masks_dir=None)
    -> (cube, wcs, mask)
```

Behavior:

1. Read the raw `(3, ny, nx)` cube. **No transpose, trim, rotate, shift, or
   flipud.** Cast to `float32` (values unchanged; convenient for repair /
   photometry). No bias subtraction, no negative clip.
2. Resolve the bad-pixel mask whenever a source is available:
   - explicit `badpix=path` or `badpix=ndarray` → that mask (cast to bool);
   - otherwise resolve nearest-in-date via `load_alcor_badpix_mask(frame_time,
     masks_dir)` (frame time from the `YYYY_MM_DD__HH_MM_SS` filename, MST→UT,
     DATE-header fallback).
   - If shape mismatches the cube or nothing is found, the mask is `None`.
3. Bad-pixel repair: if `badpix != None` (i.e. `"repair"`, a path, or an array)
   and a mask was resolved, replace flagged pixels per channel with their local
   5×5 median on the raw cube (`_apply_badpix_repair`, unchanged). `badpix=None`
   leaves the cube untouched but still resolves+returns the mask.
4. WCS: if `wcs` is passed, use it verbatim. If `None`, resolve the nearest
   calibration epoch by the frame's time and build the WCS from it
   (`build_alcor_wcs(**epoch)`).
5. **Always** return the 3-tuple `(cube, wcs, mask)`. `mask` is the `(3, ny, nx)`
   bool bad-pixel mask in native orientation, or `None` when unavailable. No
   geometric transform is applied to the mask (the cube is not resampled), so it
   stays aligned to `cube` trivially.

Removed parameters: `rotation, xcen, ycen, radius, horizon_radius, xshift,
yshift, radial_coeffs, sip_degree, return_mask`. All geometry now lives in the
WCS / calibration epoch.

Usage patterns:
- Pipeline: `cube, wcs, _ = load_alcor_fits(f)` → repaired cube + WCS.
- Photometry: `cube, wcs, mask = load_alcor_fits(f, badpix=None)` → untouched
  pixels + the mask to OR with the future horizon/structure mask.

### 2. Raw-frame WCS builder + calibration schema

`build_alcor_wcs` is rewritten to produce a full alt/az WCS over the **raw**
`(ny, nx)` image plane from absolute geometry:

```python
build_alcor_wcs(xcen, ycen, rotation, radial_coeffs,
                horizon_radius=ALCOR_HORIZON_RADIUS, sip_degree=5)
```

- `CRPIX` = the zenith pixel `(xcen, ycen)` in raw coords (1-based FITS).
- `CRVAL = (0, 90)`, `CTYPE = ["RA---ARC", "DEC--ARC"]` (azimuth as RA-analog,
  altitude as Dec-analog), `lonpole = 0` as today.
- `CD`/`PC` matrix carries the **rotation** and the axis **parity** that makes
  az/alt come out correct in native (un-flipped) orientation. This is what the
  old `flipud` + `scipy rotate` used to do to the pixels. The plate scale is
  `cdelt = 90 * k1 / horizon_radius` as today.
- SIP `A`/`B`, centered at this `CRPIX`, encode the radial `k3`/`k5` distortion.
  The analytic construction (Cartesian displacement of
  `z = 90*(k1*rho + k3*rho**3 + k5*rho**5)`, `rho = r/horizon_radius`) is the same
  as the current `_build_alcor_wcs_cached`; only the SIP reference pixel moves
  from the array center to the zenith. Inverse SIP (`AP`/`BP`) fit as today.
- Cached on its hashable arguments; returns a fresh `deepcopy` per call.

The parity/rotation sign is derived by composing the known old→raw coordinate
transform (invert transpose → flipud → rotate → trim) so the new WCS reproduces
the old behavior, then confirmed against detections (see §6 round-trip test).

`ALCOR_CALIBRATIONS` epoch dicts change schema to raw-frame absolutes:

```python
ALCOR_CALIBRATIONS = [
    {"epoch": "2024-09-04", "xcen": <px>, "ycen": <px>,
     "rotation": <deg>, "radial_coeffs": (1.0, <k3>, <k5>),
     "horizon_radius": 662},
]
```

`xshift`/`yshift` (offsets from the resampled array center) are replaced by
`xcen`/`ycen` (the absolute zenith pixel in the raw frame). `alcor_calibration(time)`
nearest-epoch resolution is unchanged. The module-level convenience constants
(`ALCOR_ROTATION`, etc.) are updated to the new schema; `ALCOR_XSHIFT`/
`ALCOR_YSHIFT` are removed and references updated.

### 3. Forward model, matcher, fitter (raw frame)

- `_predict_pixels(alt, az, xcen, ycen, rotation, radial_coeffs, horizon_radius)`
  maps alt/az → raw `(x=col, y=row)` pixel coordinates. Same radial inversion
  (`_invert_radial`); the linear part is recentered on `(xcen, ycen)` with the
  raw-frame rotation/parity (matching §2's WCS). It must agree with
  `wcs.world_to_pixel` for the same epoch.

- `assign_alcor_matches` — algorithm unchanged (kd-tree candidates, connected
  components, brightness tie-break, asterism verification). It just consumes
  raw-frame predictions and raw detection centroids. Its `params` dict carries
  `xcen, ycen, rotation, radial_coeffs` instead of `xshift, yshift, ...`.

- `_fit_params` fits `(xcen, ycen, rotation, k3[, k5])` (k1 fixed at 1.0) via the
  same `soft_l1` robust least-squares against `_predict_pixels`.

- `detect_alcor_stars(image, ...)` accepts either a 3D cube or a 2D frame:
  - 3D `(3, ny, nx)` → collapse to a **luminance** frame (mean of R, G, B) and
    detect on it.
  - 2D `(ny, nx)` → detect as-is (lets a caller pass `cube[1]` for green-only).
  Detection centroids are raw-frame `(xcentroid, ycentroid)`.

- `_detect_alcor_frame` loads the raw cube (`load_alcor_fits` with `badpix=None`,
  no resampling), detects on luminance, builds the reference catalog, and returns
  raw-frame detections. The fitter no longer loads a "neutral" resampled frame.

- `fit_alcor_wcs` is unchanged in structure (dark-frame selection, pooled
  multi-round tightening, 3·MAD rejection, parallel detect). It seeds from the
  nearest epoch, fits raw-frame geometry, and **prints an `ALCOR_CALIBRATIONS`
  dict in the new schema** (`xcen, ycen, rotation, radial_coeffs,
  horizon_radius`, stamped with the night date). `save_alcor_residual_plot`
  follows the raw frame.

- **Re-fit 2024-09-04**: run `fit_alcor_wcs` on the 2024-09-04 night and paste the
  resulting new-schema dict as the shipped epoch, so the library resolves a
  correct WCS out of the box.

### 4. `plot_alcor_fits` & `alcor_proc_fits`

- `plot_alcor_fits` does its own presentation transform now:
  1. `cube, wcs, _ = load_alcor_fits(filename)`.
  2. Build the RGB display image: stack channels to `(ny, nx, 3)`, apply the
     empirical `gscale`/`bscale` channel factors and the power + ZScale stretch.
  3. Trim a square of half-width `radius` around the WCS zenith
     (`wcs.wcs.crpix`) for display, render with `origin="lower"` (no flipud).
  4. Center the horizon circle and the polar alt/az overlay on the trimmed
     zenith. Altitude ticks still come from `wcs.world_to_pixel_values`.
  Display flags (`gscale`, `bscale`, `powerstretch`, `contrast`, `radius`,
  `figsize`) stay; geometry flags are gone.

- `alcor_proc_fits` writes the **raw cube as-is** plus the WCS header (no
  `np.flipud`, no resampling). The previous flipud-into-FITS-lower-origin dance
  is removed.

### 5. CLIs & docs

- `alcor_proc_fits` CLI loses `--rotation/--xcen/--ycen/--xshift/--yshift/
  --radial.../--radius/--horizon-radius` geometry flags (geometry comes from the
  resolved epoch or an explicit WCS).
- `plot_alcor_fits` CLI loses the geometry flags; keeps display flags.
- `fit_alcor_wcs` CLI is unchanged in spirit (it now prints the new-schema dict).
- Update CLAUDE.md's alcor paragraph: describe the WCS-driven, no-resample model
  (raw cube returned; WCS carries zenith/rotation/distortion; visualization does
  its own trim + `origin="lower"`; no bias subtraction in `load_alcor_fits`).

### 6. Testing

- `build_alcor_wcs` round-trip: for the 2024 epoch, `wcs.world_to_pixel(az, alt)`
  of several catalog stars lands within a small tolerance of their known raw
  pixel positions; `_predict_pixels` agrees with `wcs.world_to_pixel` for the
  same epoch.
- `load_alcor_fits`: returns `(cube, wcs, mask)` 3-tuple; `cube` is `(3, ny, nx)`
  raw (no resampling, no bias subtraction); `badpix="repair"` removes a known hot
  pixel while `badpix=None` leaves it but still returns the mask; explicit
  `wcs=` is returned verbatim.
- `detect_alcor_stars`: 3D input detects on luminance; 2D input detects as-is;
  a known star is found in both.
- Update `test_alcor_badpix.py` and any WCS/load tests for the new signature and
  raw-frame mask orientation.

## Data flow

```
load_alcor_fits(frame)
  read raw (3, ny, nx) float32                     [no transpose/trim/flip]
   └ resolve badpix mask (nearest-date / explicit)
   └ badpix='repair'? repair raw cube in place
   └ wcs given? use it : build_alcor_wcs(nearest epoch)   [CRPIX=zenith, PC=rot+parity, SIP=radial]
   └ return (cube, wcs, mask)

fit_alcor_wcs(night)                               plot_alcor_fits(frame)
  select dark frames                                 cube,wcs,_ = load_alcor_fits
  detect on luminance (raw)                          stretch RGB; trim around wcs zenith
  pool + match (raw-frame _predict_pixels)           render origin='lower'; polar overlay
  fit (xcen,ycen,rotation,k3[,k5])                 alcor_proc_fits(frame)
  print new-schema ALCOR_CALIBRATIONS dict           write raw cube + wcs header
```

## Error handling

- `wcs=None` and no calibration epoch resolvable → fall back to the most recent
  epoch (existing `alcor_calibration(None)` behavior).
- Frame shape ≠ mask shape, or no mask found → `mask=None`, repair is a no-op.
- Per-channel / 2D detection input that is neither 2D nor 3D → `ValueError`.
