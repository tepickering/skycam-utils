# Alcor WCS: Optical-Axis Tilt (World-Side Pole Offset) — Design

**Date:** 2026-06-11
**Status:** Approved design, pending implementation plan
**Follows:** `2026-06-10-alcor-tangential-distortion-design.md` (Brown–Conrady
P1/P2, implemented)

## Problem

After adding Brown–Conrady tangential terms, the 2026-05-18 night fit improved
only modestly (RMS 1.537 → 1.321 px) and the residuals
(`~/MMT/skycam_data/20260518_k5_p1p2.png`) retain coherent once-per-revolution
structure with a zenith-angle dependence that Brown–Conrady cannot produce:

- radial component: ~3 px peak-to-peak near the zenith, *decreasing* to
  ~1 px peak-to-peak at large zenith angle;
- tangential component: ~2 px peak-to-peak near the zenith, growing to
  ~6 px peak-to-peak at large zenith angle with the *opposite sign*
  (a 180° phase flip).

Brown–Conrady decentering grows strictly as r² in both components, in phase,
with the radial part 3x the tangential — no (P1, P2) produces the observed
shape. The matching physical model is a **tilt of the optical axis in world
space**: the lens pointing at zenith distance ε toward azimuth A₀. Its
signature:

- tangential displacement ∝ ε·r/tan(z): largest near the zenith (position
  angle about the axis swings fastest near the pole), falling toward the
  horizon — after the fit absorbs the best constant translation, the leftover
  flips sign between small and large z;
- radial displacement ∝ ε·(dr/dz), growing only ~70% center-to-edge for this
  lens — after the center shift absorbs the mean, the leftover is modest.

Matching the ~3 px near-zenith tangential amplitude requires ε ≈ 0.3–0.4°,
plausible for a lens mounted plumb-by-eye. The camera is mechanically stable
(2024 and 2026 fits nearly identical before these terms), so the tilt is fixed
installation geometry — and refitting the 2024 night should recover the same
tilt, a built-in cross-check.

## Decision

Model the axis tilt exactly in world coordinates — no new distortion
machinery. The lens model becomes azimuthally symmetric about the **axis**
instead of the zenith. Fit the tilt **always** (no CLI flag), parametrized as
Cartesian components for conditioning.

## Model definition

### Parametrization

Per-epoch optional key `"axis_tilt": (t_n, t_e)` — tilt components of the
optical axis toward north (az=0) and east (az=90), in **degrees**. Default
`(0.0, 0.0)` for epochs that omit it. Derived quantities:

```
eps = hypot(t_n, t_e)          # tilt magnitude, deg
A0  = atan2(t_e, t_n)          # azimuth the axis leans toward, deg
```

The axis points at altitude `90 - eps`, azimuth `A0`. Cartesian components are
the fitted parameters because polar (eps, A0) is singular at eps=0 (A0
undefined), exactly where the fit starts.

### Axis-centered coordinates

A star at `(alt, az)` is transformed to axis-centered polar coordinates
`(z', A')` by the exact **minimal rotation**: rotate the sky by `eps` about
the horizontal unit vector at azimuth `A0 + 90°`. Properties:

- exact spherical trig (vectorized rotation matrix), no small-angle
  approximation;
- `A' -> az` continuously as `eps -> 0`, so zero tilt reproduces today's model
  bit-for-bit;
- `z'` is the true angular distance from the axis:
  `cos z' = sin(alt)·sin(alt0) + cos(alt)·cos(alt0)·cos(az - A0)` with
  `alt0 = 90 - eps`.

Everything downstream of `(z', A')` is unchanged: the radial polynomial is
inverted in `z'`, the pixel angle is `rotation - A'`, and the Brown–Conrady
tangential terms continue to act in pixel space about `(xcen, ycen)`.

### Semantics change: xcen/ycen

`xcen`/`ycen` becomes the **optical-axis pixel** (the center of radial
symmetry / distortion center), no longer the zenith pixel. With eps ≈ 0.35°
the two differ by ~3 px. The zenith pixel is obtained through the WCS
(`world_to_pixel` of alt=90), not from CRPIX.

## Changes by component

### 1. Forward model — `_predict_pixels`

- New keyword `axis_tilt=(0.0, 0.0)`.
- Nonzero tilt computes `(z', A')` via the minimal-rotation matrix
  (numpy-vectorized), then proceeds through the existing radial inversion and
  tangential fixed-point machinery with `z'` and `A'` in place of `90 - alt`
  and `az`.
- Zero tilt short-circuits to the current code path exactly (bit-identical).

### 2. WCS — `build_alcor_wcs` / `_build_alcor_wcs_cached`

- New keyword `axis_tilt=ALCOR_AXIS_TILT`; cache key extended with the
  (float, float) tuple.
- The tilt is pure FITS-WCS geometry: `CRVAL` moves from `(0, 90)` to
  `(A0, 90 - eps)`; `CRPIX` stays the axis pixel; the analytic SIP (radial +
  Brown–Conrady) is unchanged and stays centered on CRPIX.
- `LONPOLE` is set to the value that makes the WCS native frame coincide with
  the forward model's minimal-rotation frame. The exact value is derived
  during implementation and **pinned by round-trip tests**: WCS vs
  `_predict_pixels` to < 1e-3 px over the full FOV with tilt, k5, and P1/P2
  all nonzero simultaneously. astropy/WCSLIB performs the spherical rotation
  exactly.
- Zero tilt produces CRVAL `(0, 90)` and today's WCS unchanged.

### 3. Fitter — `_fit_params`

- `t_n`, `t_e` join the parameter vector in both branches:
  - default: `(xcen, ycen, rotation, k3, P1, P2, t_n, t_e)` — 8 params;
  - `fit_k5`: 9 params (k5 inserted as today).
- Initialized from `init_params.get("axis_tilt", (0.0, 0.0))`.
- Returned dict gains `axis_tilt=(t_n, t_e)`.
- Always fit; no CLI flag. Conditioning: the tilt's `1/tan(z)` tangential
  signature is linearly independent of translation (constant), rotation
  (∝ r, azimuth-uniform), and Brown–Conrady (∝ r²); its radial part separates
  from the center shift through the dr/dz growth.

### 4. Calibration schema

- Optional epoch key `"axis_tilt": (t_n, t_e)`; `alcor_calibration()` fills
  `(0.0, 0.0)` when absent (existing epochs untouched).
- New module default `ALCOR_AXIS_TILT = _LATEST_CALIBRATION["axis_tilt"]`.
- `_format_calibration_entry` includes `"axis_tilt": (t_n, t_e)` (with a
  `.get` zero default for old result dicts).
- The `ALCOR_CALIBRATIONS` block comment documents the key, its units
  (degrees), and the north/east component convention.

### 5. Threading

Same plumbing class as `tangential_coeffs` (a geometry parameter silently
reverting to a default mid-fit corrupts the pool):

- `assign_alcor_matches` passes `params.get("axis_tilt", (0.0, 0.0))` into its
  `_predict_pixels` seeding.
- `fit_alcor_wcs`: warm-start init carries `base.get("axis_tilt", (0,0))`;
  the two final-pool `_predict_pixels` calls pass
  `tuple(params["axis_tilt"])`; the returned dict includes it via `**params`.
- `save_alcor_residual_plot`: the refined prediction uses
  `params.get("axis_tilt", (0.0, 0.0))`; the idealized baseline stays
  zenith-centered equidistant by design.
- `load_alcor_fits` passes the resolved epoch's `cal["axis_tilt"]` to
  `build_alcor_wcs`.
- The CLI prints, in addition to the paste-ready entry, one informational
  line with the equivalent polar tilt:
  `# axis tilt: eps=X.XXX deg toward az=YYY.Y deg` (Cartesian components are
  not intuitive on their own).

### 6. Zenith consumers (CRPIX no longer the zenith)

Two call sites read CRPIX as the zenith pixel and must switch to a WCS
lookup of alt=90 (`wcs.world_to_pixel_values(0.0, 90.0)`; the azimuth value
is irrelevant at the pole):

- the keogram center-column selection (`alcor.py:1530`);
- the plotter crop center (`alcor.py:1792`).

With zero tilt these return CRPIX exactly, so existing outputs are unchanged
for current epochs.

## Testing

1. **Zero-tilt no-op:** `_predict_pixels` and `build_alcor_wcs` with
   `axis_tilt=(0, 0)` are bit-identical to current behavior; an epoch dict
   without the key resolves to `(0, 0)`.
2. **Spherical exactness:** for nonzero tilt, the angular distance between
   the world point and the axis, pushed through the plate solution, equals
   the pixel radius about `(xcen, ycen)` (independent re-derivation via the
   `cos z'` identity above).
3. **Continuity:** A' -> az as eps -> 0 (compare eps=1e-9 against eps=0).
4. **WCS round-trip:** `build_alcor_wcs` vs `_predict_pixels` to < 1e-3 px in
   both directions over the FOV, with tilt + k5 + P1/P2 all nonzero.
5. **Parameter recovery:** `_fit_params` recovers synthetic
   `(t_n, t_e) ≈ (0.3°, -0.2°)` alongside the other parameters, both branches.
6. **End-to-end:** synthetic night fit (mirroring the existing monkeypatched
   tests) recovers injected tilt with a seed calibration lacking the key.
7. **Zenith lookup:** keogram and plotter consume the WCS zenith; with zero
   tilt the selected column/crop center equals today's.

## Validation (operational)

Refit 2026-05-18 with `--fit-k5` on the remote machine. Success:

- RMS well below 1.321 px;
- the near-zenith tangential sinusoid and the large-z phase-flipped structure
  collapse in the residual plot;
- fitted eps ≈ 0.3–0.4°.

Cross-check: refit the 2024-09-04 night — the camera has not moved, so it
should recover the same `(t_n, t_e)` within uncertainty. Agreement confirms
real installation geometry; disagreement means the term is absorbing
something else and the model needs rethinking before baking entries.

## Out of scope

- Removing or gating the Brown–Conrady terms — they stay, always fit; with
  the tilt present they relax to genuine sensor tilt (the two are separable:
  r² vs dr/dz radial dependence).
- Per-frame or time-dependent tilt (the camera is stable; one tilt per epoch).
- Any change to detection, matching logic, or the residual-plot layout.
