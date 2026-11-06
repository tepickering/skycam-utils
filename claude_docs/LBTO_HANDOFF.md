# LBTO/VATT OMEA support — handoff

Status as of **2026-11-05**. Phase 0 is done and passed; nothing in
`skycam_utils/` or `../omea_linux/` has been changed. Resume at Phase 1.

## What was decided

| Question | Answer |
| --- | --- |
| Scope | Full science parity, not display-only |
| omea's role | Ingest their Skywatch `.fits.fz` first, replace Skywatch later |
| Binning | 2x2 (omea's existing convention), not 4x4 |

## What was established (don't re-derive)

**The data.** ZWO ASI6200MC Pro, `(6388, 9576)` raw Bayer mosaic in a
`.fits.fz` `CompImageHDU` (extension 1) — *not* Skywatch's debayered
`(3, 1411, 1422)` cube. 60 s at f/4.0, ~2.2 min cadence, ~70 MB/frame,
~22 GB/night. Site Mt Graham: `+32:42:01`, `-109:53:31`, ~3190 m.
`BAYERIND = 4` (RGGB), `BITCAMPX = 15`, so a 2x2 mean stays inside int16 and
`ALCOR_SATURATION = 32767` still holds.

**`DATE-OBS` is UT here and local MST at the MMT.** The rule is "exposure
start in the acquisition host's timezone", a property of the machine, not the
site. `_filename_ut_datetime`'s fixed +7 h is wrong for LBTO, and their
filenames are `allsky-YYYY-MM-DD-HH-MM-SS-utc`.

**Phase 0 result: the MMT lens model fits, at the same 0.36 px floor.** Fitted
on the 2x2-binned cube **after mirroring in x**:

```python
LBTO_GEOMETRY = dict(
    xcen=2334.130, ycen=1611.948, rotation=171.6477,
    radial_coeffs=(1.0, 0.119277, 0.016480),
    tangential_coeffs=(0.000581, 0.0006389),
    axis_tilt=(3.02652, -3.67126),
    horizon_radius=1825.6,
)   # parity -1 vs the MMT: north is at -y in the raw frame
```

700 stars, rms 0.357 px. The 2026-10-15 night converges to the same values
(centre 0.11 px, rotation 0.004 deg). Trailing is not the limit
(residual-vs-trail correlation -0.10). Altitude 0 falls at mean radius 1647 px
against an illuminated circle of ~1685, so the camera reaches the horizon on
most azimuths.

**Display orientation.** The camera's native frame has north at DECREASING y,
so figures are rotated 180 degrees to put north up and east left, matching how
the MMT presents its frames (`lbto_render.py`, `--no-north-up` opts out). That
is a rotation, not a mirror: it preserves the handedness an all-sky camera
actually has, and the geometry constants above always refer to the UNROTATED
(mirrored-in-x) frame. Do not bake the rotation into the WCS.

## Three traps, all paid for already

1. **A blind cold-start WCS search does not work here, and fails
   *convincingly*.** `fit_alcor_wcs` always warm-starts from a known epoch. A
   grid search over (parity, rotation, plate scale, k3) is dominated by chance
   coincidence long before the grid is fine enough, and converges reproducibly
   onto aliases that put Sirius 107 px from its own image. Always print the
   expected chance rate `ndet * pi * tol**2 / (nx*ny) * ncat` beside any match
   count; if matches are not many times chance, the number means nothing.
   **The fix is ground truth**: a handful of objects identified by eye. The
   seven used are in `claude_docs/scripts/lbto_wcs_groundtruth.py`. This
   should become a small packaged bootstrap tool — it is needed once per new
   camera, and it is what makes multi-instrument support tractable.
2. **The radial term must be in the seed, not left to the fit.** A seed with
   `k3 = k5 = 0` is an equidistant projection; any other fisheye family
   departs from it by ~66 px at mid-radius here, so only a thin annulus can
   ever match.
3. **`_fit_params` always frees the tangential and axis-tilt terms.** That is
   right for a warm start and ruinous on a partial match set — those four
   absorb the seed error and lock the solution into whichever corner it
   started in (this produced an absurd 4.8 deg tilt before it was caught).
   Open the model in stages.

## The proposed design (presented, NOT yet approved)

**Core idea:** key calibration on `(instrument, time)` instead of time alone.
A new `alcor/instruments.py` holds an `AlcorInstrument` dataclass bundling
what are now ~40 module globals: site `EarthLocation`, filename-time
convention, cube shape, `saturation`, `calib_exptime`, `field_radius`, the
`calibrations` / `zeropoints` / `altitude_flat` tables, asset directory names,
`sb_targets`, refraction constants (elevation-dependent), blend PSF sigma.

Resolution in three layers so nothing outside the package changes:
`ALCOR_INSTRUMENT` defaults to the MMT and every existing constant becomes an
alias into it; `alcor_instrument_for(header)` resolves from `INSTRUME` + site
so `load_alcor_fits` picks it up; `instrument=` on public entry points plus an
`$ALCOR_INSTRUMENT` env override.

**The main risk** is the one that bit the `alcor.py` -> `alcor/` split: dozens
of signatures read `def f(..., horizon_radius=ALCOR_HORIZON_RADIUS)`, and
**default arguments bind at import time**. The profile must be threaded
through call arguments; rebinding a global leaves the real consumers on MMT
numbers and the tests pass for the wrong reason. This is the bulk of the work.

**`raw_to_cube` should move from omea into `skycam_utils.alcor.mosaic`.**
omea depends on skycam_utils, not the reverse, so the library that defines
every other part of the frame format should own the demosaic too. Then
`load_alcor_fits` can read a mosaic instrument on the fly from its profile
(~0.95 s a frame, against the MMT's 0.72 s bz2 load) and
`lbto_alcor/allsky2skycam.py` becomes unnecessary.

**Audit for the axis tilt.** LBTO's is 4.76 deg against the MMT's ~1.0, which
puts the zenith 96 px off the optical axis. `CLAUDE.md` already warns that the
zenith must be found via the WCS rather than CRPIX; anything measuring a
radius "from CRPIX" (notably `ALCOR_FIELD_RADIUS`) needs checking.

## Phases

| Phase | Work | Gate |
| --- | --- | --- |
| 0 | Does the lens model fit? | **DONE — passed** |
| 1 | `AlcorInstrument` profile refactor, MMT-only | 257 tests green, no behaviour change |
| 2 | `alcor/mosaic.py` + the LBTO profile + mosaic reading | reads their frames end to end |
| 3 | LBTO calibration assets | **blocked on data from LBTO** |
| 4 | omea at LBTO | needs answers to the open questions |

Phase 3 is the long pole and it is **data, not code**: zeropoints need one
clear moonless night, the horizon mask one overcast night, badpix consensus a
night a month, the altitude flat several clear nights across seasons. Also
note `ALCOR_BLEND_PSF_SIGMA = 0.68` px is an MMT number; at LBTO's 2.4x finer
sampling the blend model must be re-derived, not scaled. (Plate scale: 20.3
px/deg at the zenith against the MMT's 8.3, both on the 2x2-binned frames;
`horizon_radius / 90` IS the zenith plate scale, so do not divide it by
`1 + k3 + k5` as an earlier pass wrongly did.)

## Open questions for LBTO

- Does this OMEA carry the Alcor sensor/heater board? `omea/board.py` assumes one.
- Which light domes replace Tucson and Nogales as fixed SB targets?
- Is the acquisition host clock disciplined? Everything rests on `DATE-OBS`.

## Files

Scripts, none packaged, all in `claude_docs/scripts/`:

| File | What it does |
| --- | --- |
| `lbto_quicklook.py` | coarse render + image-circle measurement |
| `lbto_wcs_spike.py` | the fitting pipeline; `--seed-from` skips the grid search |
| `lbto_wcs_groundtruth.py` | solves geometry from hand-identified objects; holds the seven |
| `lbto_render.py` | colour all-sky renders with the alt/az grid and labels |

Figures in `claude_docs/gplots/`: `lbto_quicklook.png`,
`lbto_wcs_residuals.png`, `lbto_wcs_residuals_1015.png`,
`lbto_groundtruth.png`, `lbto_allsky_labelled.png`, `lbto_allsky_zoom.png`.

Sample data: `~/MMT/lbto_alcor` (20 frames, 2026-10-15 and 2026-11-04).

To reproduce the fit:

```bash
cd /Users/tim/MMT/skycam_utils
/Users/tim/conda/envs/skycam/bin/python claude_docs/scripts/lbto_wcs_spike.py \
    ~/MMT/lbto_alcor/allsky-2026-11-04-0*.fits.fz \
    -o claude_docs/gplots/lbto_wcs_residuals.png \
    --cache <scratch>/lbto_det --max-detections 1200 --tolerance 6 \
    --seed-from="-1,2385.283,1662.127,171.8782,1876.50,0.269411,-0.141487"
```

The report sent to LBTO:
<https://claude.ai/artifact/3CUiaJtWFk5HuxQ83ZrsuZ>
