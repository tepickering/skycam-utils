# Alcor WCS: Brown–Conrady Tangential (Decentering) Distortion — Design

**Date:** 2026-06-10
**Status:** Approved design, pending implementation plan

## Problem

`fit_alcor_wcs` residuals (e.g. `~/MMT/skycam_data/20260518_k5.png`, 2026-05-18
night, pooled RMS 1.54 px) are dominated by a once-per-revolution sinusoid in
both the radial and tangential residual components whose amplitude grows with
zenith angle. The current model — zenith center, rotation, and an
azimuthally-symmetric odd radial polynomial (k1/k3/k5) — has no parameter that
can absorb an azimuth-dependent residual, which is why adding `--fit-k5`
barely moved the RMS.

A residual growing as r², once per revolution, is the classic signature of
sensor-plane tilt / lens decentering. (A pure optical-axis tilt for a
near-equidistant fisheye produces a nearly uniform image shift to first order,
which `xcen`/`ycen` already absorb — it is not the dominant term here.)

## Decision

Add the standard 2-parameter Brown–Conrady tangential (decentering) distortion
to the Alcor lens model, fit it **always** (no opt-in flag — unlike k3/k5 the
terms are well-conditioned against the existing parameters), and encode it
exactly in the WCS SIP.

## Model definition

The distortion is defined on the **pixel→world side**, matching the existing
convention (the radial plate solution maps detector radius to zenith angle;
`_predict_pixels` inverts it numerically).

Let `u, v` be the raw-pixel offsets from the zenith pixel `(xcen, ycen)`,
`H = horizon_radius`, `ū = u/H`, `v̄ = v/H`, `ρ² = ū² + v̄²`. The SIP-applied
distortion (added to `(u, v)` before the linear ARC part) becomes:

```
Δū = (radial k-terms) + P1·(ρ² + 2ū²) + 2·P2·ū·v̄
Δv̄ = (radial k-terms) + P2·(ρ² + 2v̄²) + 2·P1·ū·v̄
```

`P1`, `P2` are dimensionless (normalized by `H`), like the k coefficients.
In raw pixel units the tangential correction is an exact degree-2 polynomial:

```
Δu = (P1/H)·(3u² + v²) + (2·P2/H)·u·v
Δv = (P2/H)·(u² + 3v²) + (2·P1/H)·u·v
```

Expected magnitude for the observed ~1–1.5 px swing at the horizon:
|P1|, |P2| ≲ a few ×10⁻³.

## Changes by component

### 1. `build_alcor_wcs` / `_build_alcor_wcs_cached` (alcor.py:914)

- New keyword `tangential_coeffs=(0.0, 0.0)` (module default
  `ALCOR_TANGENTIAL_COEFFS`).
- Add to the analytic SIP A/B matrices (exact, no approximation):
  `a[2,0] += 3·P1/H`, `a[0,2] += P1/H`, `a[1,1] += 2·P2/H`;
  `b[0,2] += 3·P2/H`, `b[2,0] += P2/H`, `b[1,1] += 2·P1/H`.
- The cache key gains the (float, float) tuple.
- The "all distortion zero → plain linear ARC" early return also requires
  `|P1| < 1e-12 and |P2| < 1e-12`.
- `_fit_sip_inverse` (AP/BP) needs no change — it already fits whatever A/B
  contain.

### 2. `_predict_pixels` (alcor.py:112)

- New keyword `tangential_coeffs=(0.0, 0.0)`.
- The radial Newton inversion runs unchanged, producing radial-only offsets
  `(u₀, v₀)`. If the tangential coefficients are nonzero, refine with 2–3
  fixed-point iterations of the full 2D inversion
  `(u, v) ← (u_target, v_target) − Δ_tang(u, v)` where `(u_target, v_target)`
  is the intermediate-coordinate target the SIP must reach. The correction is
  px-scale (≪ H), so fixed-point convergence is immediate.
- Zero coefficients short-circuit to the current code path exactly
  (bit-identical results).

### 3. `_fit_params` (alcor.py:143)

- `P1`, `P2` join the parameter vector in **both** branches:
  - default branch: `(xcen, ycen, rotation, k3, P1, P2)` — 6 params;
  - `fit_k5` branch: `(xcen, ycen, rotation, k3, k5, P1, P2)` — 7 params.
- Initialized from `init_params.get("tangential_coeffs", (0.0, 0.0))`.
- Returned dict gains `tangential_coeffs=(P1, P2)`.
- Loss/f_scale unchanged (`soft_l1`, 3.0).

### 4. `ALCOR_CALIBRATIONS` schema (alcor.py:49)

- Epochs gain an **optional** `"tangential_coeffs": (P1, P2)` key.
- `alcor_calibration()` fills `(0.0, 0.0)` when the key is absent, so the two
  existing epochs are untouched and previously printed dicts still paste
  cleanly.
- New module default `ALCOR_TANGENTIAL_COEFFS = _LATEST_CALIBRATION["tangential_coeffs"]`
  alongside the existing constants.
- The table's explanatory comment documents the new key and its default.

### 5. Threading through the night fit

`tangential_coeffs` must travel inside the params dict everywhere the geometry
travels — this is the same plumbing class as the `horizon_radius` bug (a
parameter silently reverting to the module default mid-fit collapses the match
pool). Specifically:

- `assign_alcor_matches` passes `params.get("tangential_coeffs", (0.0, 0.0))`
  into its `_predict_pixels` seeding.
- `fit_alcor_wcs`'s round-by-round refit carries the fitted values into the
  next round's matching.
- `load_alcor_fits` → `build_alcor_wcs` passes the resolved epoch's value.
- The ready-to-paste epoch dict printed by the CLI includes
  `"tangential_coeffs": (P1, P2)`.

### 6. No CLI flag

The fit always solves for P1/P2. No `--fit-tilt` flag. `--fit-k5` semantics
are unchanged (it still only toggles the quintic radial term).

## Testing

1. **Round-trip exactness:** `build_alcor_wcs` with nonzero P1/P2 agrees with
   `_predict_pixels` to < 0.01 px in both directions (world→pix and pix→world)
   on a grid spanning the full FOV.
2. **Parameter recovery:** synthetic stars generated from known
   `(xcen, ycen, rotation, k3, P1, P2)` plus Gaussian pixel noise →
   `_fit_params` recovers P1/P2 (and leaves the other params unbiased).
3. **Backward compatibility:** an epoch dict without `tangential_coeffs`
   resolves to `(0.0, 0.0)` and produces a WCS identical to today's; the
   existing fit/residual-plot tests pass unchanged.
4. **Zero short-circuit:** `_predict_pixels` with `tangential_coeffs=(0, 0)`
   is bit-identical to the current implementation.

## Validation (operational)

Refit the 2026-05-18 night on the remote machine with the new code. Success:

- the once-per-rev sinusoid in the radial/tangential-vs-azimuth panels of the
  residual plot collapses;
- pooled RMS drops meaningfully below 1.54 px;
- |P1|, |P2| come out small (≲ a few ×10⁻³) with the other parameters stable
  relative to the current epoch values.

## Out of scope

- Physical axis-tilt (θ, φ) parametrization — mostly degenerate with
  `xcen`/`ycen` for this lens; rejected during brainstorming.
- General degree-2 SIP residual surfaces beyond P1/P2 — YAGNI unless P1/P2
  leaves structure behind.
- Any change to the residual-plot layout (the recent `extent` fix stands).
