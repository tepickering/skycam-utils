# Alcor star-matcher hardening — design

**Date:** 2026-06-07
**Component:** `skycam_utils/alcor.py`

## Goal

`fit_alcor_wcs` pools per-frame star matches across a night and fits one global
lens geometry. On 2026 data it returns ~14 px residual RMS with strong
structure: a well-fit population near 0 plus a broad upward scatter that grows
toward the horizon. Diagnosis (residual-vs-zenith, residual-vs-azimuth, and the
binned residual vector field) shows the RMS is **dominated by match
contamination** — the all-sky projection compresses stars near the horizon until
the fixed-epoch seed error rivals inter-star spacing, so nearest-neighbor
matching grabs the wrong star. A smaller, genuine azimuthal asymmetry sits
underneath but cannot be assessed until the matches are clean.

Harden the matcher so the pooled residual reflects real geometry error, not
mispairings: limit detections to the brightest few hundred, and match with
kd-tree pattern (asterism) verification plus a local relative-brightness
tie-break, all seeded by the existing epoch geometry.

## Decisions (from brainstorming)

- **Seeded, not seed-free.** Keep the neutral-frame + nearest-epoch geometry as
  the starting guess; the fit already converges, so the problem is assignment
  quality, not initial alignment. No blind asterism hashing.
- **Detection cap: top-N by flux.** Keep DAOStarFinder's threshold modest but
  keep only the N brightest detections per frame (default 200). Predictable,
  clean bright list regardless of per-frame noise/transparency.
- **Catalog depth Vmag ≤ ~4** (default changes from 3), so the catalog yields a
  couple hundred stars above the horizon — comparable to the detection cap.
- **No per-frame transform fitting.** The seed error is *global* (one fixed
  epoch applied to every frame), not per-frame, so a single global fit each
  round suffices; the failure to fix is *local* crowding. (Rules out RANSAC /
  per-frame similarity fits as unnecessary — YAGNI.)
- **Drop the per-frame bootstrap refit.** The current `match_alcor_stars` refits
  the 4 geometry params *per frame* during matching; pooling those under one
  global model is a primary source of the residual structure. The new matcher
  assigns against a *fixed* geometry and never refits internally.
- **Brightness is a *local* tie-break, not a global gate.** Atmospheric
  extinction (zenith-angle) and, more importantly, **thin-cloud extinction**
  (patchy in space and time) break any global/zenith brightness model: a bright
  star behind a cloud or near the horizon can be fainter than a mid star in
  clear sky at zenith. Clouds extinct a *local* patch roughly in common, so only
  the **relative** flux ordering among nearby contested stars is trustworthy.
  Brightness therefore disambiguates *only* when one detection is contested by
  several catalog candidates within tolerance.

## Components (all in `skycam_utils/alcor.py`)

### `detect_alcor_stars(im, fwhm=3.0, threshold_sigma=5.0, max_detections=200)`

After DAOStarFinder, sort detections by `flux` descending and keep the top
`max_detections` (no cap when `None`). Column set unchanged
(`xcentroid`, `ycentroid`, `flux`).

### `assign_alcor_matches(cat, det, params, tolerance, radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS, n_neighbors=5, min_corroborating=2, pattern_tol=3.0, brightness=True)`

Per-frame assignment against a **fixed** geometry. Replaces `match_alcor_stars`.

1. Predict catalog pixels `(px, py)` via `_predict_pixels(cat["Alt"], cat["Az"],
   **params, radius=radius, horizon_radius=horizon_radius)`. Build
   `scipy.spatial.cKDTree` of detection `(x, y)` and of predicted catalog
   `(px, py)`.
2. **Mutual nearest-neighbor** within `tolerance`: keep pair (i, j) iff j is the
   nearest detection to catalog i *and* i is the nearest catalog to detection j,
   with separation ≤ `tolerance`.
3. **Local-pattern (asterism) verification.** For each surviving pair i→j, query
   catalog i's `n_neighbors` nearest catalog neighbors (predicted-pixel kd-tree).
   The pair is accepted iff at least `min_corroborating` of those neighbors have
   their own surviving pair whose relative offset agrees:
   `‖(det_jn − det_j) − (pred_in − pred_i)‖ ≤ pattern_tol`. This rejects
   crowded-region mispairs, which lack a self-consistent local constellation.
4. **Local brightness tie-break.** Applied during step 2 when a detection falls
   within `tolerance` of more than one catalog candidate (or a catalog star of
   more than one detection): among that contested set only, prefer the
   assignment whose `flux` ordering matches the candidates' `Vmag` ordering,
   rather than blindly taking the nearest. Skipped when `brightness=False` or the
   `flux`/`Vmag` columns are absent (falls back to nearest). It never compares
   brightness across non-contesting stars, so spatially/temporally patchy
   cloud extinction never enters.

Returns an `hstack`ed table of accepted catalog+detection rows (same shape
contract as today's `match_alcor_stars` output), empty table if none.

### `fit_alcor_wcs(...)` outer loop

Warm-start `params` from the nearest epoch (unchanged). Then iterate a
**tightening tolerance schedule** (`match_rounds` rounds, default 4, geometric
from `tolerance_start` (default ~12 px) down to `tolerance` (default ~3 px)):

```
for tol in schedule:
    pool assign_alcor_matches(cat, det, params, tolerance=tol) over all frames
    if pooled matches >= 3: params = _fit_params(pooled, init=params)   # k3-only, soft_l1
final pooled fit at the tightest tol; keep a light global 3*MAD safety cut
```

Report `n_matched`, `residual_rms`, and the **matched fraction**
(matches / available catalog-star-frames) so contamination is visible. The
returned dict and the printed `ALCOR_CALIBRATIONS` entry are unchanged in shape.

## Parameters & CLI

Function defaults carry the tuning. Exposed on `fit_alcor_wcs_cli`:
`--max-detections` (200), `--vmag-limit` (default 4), `--tolerance` (final, ~3).
Pattern knobs (`n_neighbors`, `min_corroborating`, `pattern_tol`) and the
brightness flag stay as function defaults to avoid CLI sprawl. `detect_alcor_stars`
gains `--max-detections` plumbing through the existing task tuple.

## Error handling

- Frames with too few detections/catalog skip with a reason (as today).
- Empty kd-tree / empty candidate set returns an empty match table.
- A round yielding < 3 pooled matches keeps the previous `params` (no fit on
  noise); if the very first round finds nothing, raise as today.

## Testing

- `detect_alcor_stars(max_detections=N)` returns exactly the N brightest by flux.
- `assign_alcor_matches`:
  - recovers all true pairs on a clean synthetic frame;
  - with a **planted crowding mispair** (a decoy detection near a catalog star),
    the pattern check rejects the decoy and keeps the true pair;
  - with one detection contested by two catalog stars, the **brightness
    tie-break** selects the Vmag-consistent pair; with `brightness=False` it
    falls back to nearest.
- `fit_alcor_wcs` integration: synthetic multi-frame with **injected mismatch
  detections** recovers the true geometry at low RMS, where the old
  nearest-neighbor matcher would not. Existing synthetic fit tests are updated
  for the new matcher and to supply `flux`/`Vmag` columns.

## Out of scope

- The genuine azimuthal-asymmetry distortion model (2-D / both-axis distortion).
  Decide whether it is warranted *after* clean matches expose its true
  magnitude; that is a separate spec.
- Any change to `load_alcor_fits`, the WCS math, the calibration library, or the
  residual-plot diagnostic.

## Files changed

- `skycam_utils/alcor.py` — `detect_alcor_stars` cap; new `assign_alcor_matches`
  replacing `match_alcor_stars`; `fit_alcor_wcs` tightening outer loop +
  matched-fraction reporting; CLI flags; `vmag_limit` default 3 → 4.
- `skycam_utils/tests/test_alcor_wcs.py` — detection-cap, matcher, and
  integration tests; updates to existing synthetic fit tests.
- `CLAUDE.md` — document the hardened matcher and the new flags.
