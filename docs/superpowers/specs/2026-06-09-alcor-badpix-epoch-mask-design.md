# Alcor Bad-Pixel Epoch-Mask Subsystem — Design

**Date:** 2026-06-09
**Status:** Design (pending review)

## Motivation

The alcor (OMEA 8C) sensor has a large hot/warm-pixel population that corrupts
both source finding (false detections) and sky-background modeling (spurious
peaks). They are glaringly obvious in a co-added night stack. We want to detect
them automatically and remove them before any downstream processing.

This spec covers the **bad-pixel mask subsystem only**. It is independent of two
larger efforts that remain deferred to a separate combined spec:

- Refactoring `load_alcor_fits` to stop resampling pixels (move zenith
  offset + rotation into the WCS).
- The **structure mask** (buildings + the rotating observatory) and the eventual
  combined photometry mask (bad-pixel ∪ horizon/structure).

Repairing hot pixels is an in-place pixel replacement, so it does not depend on
the resampling refactor.

## Empirical findings that drive the design

All measured on full-night dark-frame stacks (Sun < −18°, Moon < −6°):
2026-05-18 (726 frames) and 2024-09-04 (1161 frames), identical sensor geometry
`(3, 1411, 1422)`.

1. **Detection.** A per-channel small-kernel (5×5) median high-pass cleanly
   isolates hot pixels: residual robust-σ ≈ 5 counts while hot pixels reach
   residuals ≈ 25,000 (a ~4000σ dynamic range). A trail-free **per-pixel median**
   stack (not a mean) is required as input — a 3-hour mean leaves smeared
   star-trail cores that contaminate the low-z tail; the night median removes
   them (z>8 count dropped ~3.3× while the z>30–250 core was unchanged).

2. **Channel-multiplicity discriminant.** A real broadband source lights up all
   three channels; a sensor defect fires in only one or two. Excluding spikes
   present in all three channels removes ~600–960 real features per epoch from
   the mask. At z>25, bad pixels are overwhelmingly single-channel (~3,470) with
   fewer two-channel (~995).

3. **The mask is NOT constant over time.** 2024 vs 2026 overlap is flat across
   the entire threshold range (Jaccard ≈ 0.30, recall ≈ 0.45 from z>25 to z>250).
   Flatness rules out threshold flicker (would rise with z) and a global
   registration shift (would drop isolated-pixel overlap to ~0). ~Half the hot
   pixels — even the brightest — turn over between epochs: textbook CMOS aging.

4. **Central region is more stable than the edges.** Within 500 px of image
   center, 2024→2026 persistence is ~68% (Jaccard ≈ 0.47); outside, ~41%
   (Jaccard ≈ 0.25). Hot pixels are ~3× denser in the vignetted outer annulus,
   most of which is at/below the horizon (illuminated radius ≈ 662 px) and will
   be covered by the future structure mask anyway.

**Conclusion:** bad-pixel masks must be **epoch-specific**, regenerated regularly
(daily) and resolved nearest-in-date to each frame — analogous to the
time-indexed `ALCOR_CALIBRATIONS` geometry, but stored as files rather than an
in-code table because they are large arrays.

## Validated parameters

- High-pass kernel: 5×5 median.
- Threshold: **z > 25** robust-σ (`σ = 1.4826 · MAD` of the residual).
- Channel rule: keep a spike in channel *c* iff it fires in *c* **and** the total
  number of channels firing at that pixel is **≤ 2**.
- Dark-frame selection: Sun < −18°, Moon < −6° (same as `fit_alcor_wcs`).
- Repair: replace each flagged pixel, per channel, with the local 5×5 median.

## Architecture

Five units, each independently testable:

### 1. `build_alcor_badpix_mask(median_cube, ksize=5, z_thresh=25)` → `(3, ny, nx)` bool

Pure function. Input is a per-channel median stack `(3, ny, nx)`. For each
channel: `resid = img − median_filter(img, ksize)`, `z = (resid − median(resid)) /
(1.4826·MAD(resid))`. `hot = z > z_thresh`. Apply the multiplicity rule:
`bad[c] = hot[c] & (hot.sum(axis=0) <= 2)`. No I/O.

### 2. `build_alcor_median_stack(dark_files, max_frames=None, scratch_dir=None)` → `(3, ny, nx)` float32

Per-pixel median over the dark frames. RAM-bounded: write each frame's
`(3, ny, nx)` uint16 cube to a disk memmap (in `scratch_dir`, default system
temp), then median in row tiles (50 rows) so peak RAM stays well under the full
~14 GB stack. Frames whose shape differs from the first are skipped (logged).
`max_frames` strided-subsamples to cap cron runtime/scratch (median quality
saturates well below 1000 frames; default None = use all).

### 3. `create_badpix_mask` CLI

```
create_badpix_mask <day-dir> [--out-dir DIR] [--min-frames 500]
    [--z-thresh 25] [--ksize 5] [--sun-alt-max -18] [--moon-alt-max -6]
    [--max-frames N] [--scratch-dir DIR] [--quiet]
```

Flow:
1. Glob `<day-dir>/*.fits.bz2`; `select_dark_frames(sun_alt_max, moon_alt_max)`.
2. If `len(dark) < --min-frames` (default **500**): print reason, exit 0 without
   writing. The existing nearest mask continues to apply.
3. `build_alcor_median_stack` → `build_alcor_badpix_mask`.
4. Determine the mask date from the day directory name (`YYYY-MM-DD`), falling
   back to the median dark-frame time.
5. Write **gzipped** per-channel `(3, ny, nx)` uint8 mask to
   `<out-dir>/alcor_badpix_YYYY-MM-DD.fits.gz` (~30 kB). Header records
   `NSTACK`, `ZTHRESH`, `KSIZE`, `CHRULE`, and per-channel counts.

Intended to run from the existing daily movie/keogram cron on the skycam host,
once per previous day. The cron line lives in operational `scripts/` (not the
package); the spec only specifies the CLI it calls.

### 4. Storage & resolution

Masks are date-stamped files resolved **nearest-in-date** to a frame's time.
`load_alcor_badpix_mask(time, masks_dir=None)`:

- `masks_dir` resolution order: explicit arg → `$ALCOR_BADPIX_DIR` → packaged
  `skycam_utils/data/badpix/` (via `files(__package__)`, mirroring `load_wcs`).
- Glob `alcor_badpix_*.fits.gz`, parse each date, pick the one with minimum
  absolute date difference from `time`. Returns the `(3, ny, nx)` bool array
  (and its date), or `None` if the directory holds no masks.
- Packaged baseline: commit the validated **2024-09-04** and **2026-05-18** masks
  into `data/badpix/` so the library resolves correctly out of the box; the host
  cron accumulates daily masks into its operational `$ALCOR_BADPIX_DIR`.

### 5. `load_alcor_fits` integration

Two new orthogonal parameters; default behavior and return signature unchanged.

- `badpix='repair'` (default) | `None`:
  - `'repair'`: replace flagged pixels per channel with the local 5×5 median,
    applied **first thing on the raw `(3, ny, nx)` data, before the trim**.
  - `None`: leave pixels untouched.
- `return_mask=False` (default) | `True`:
  - `False`: return `(im, wcs)` (unchanged).
  - `True`: return `(im, mask, wcs)`, where `mask` is the bad-pixel mask carried
    through the **same** geometric operations the image undergoes (transpose,
    trim, `flipud`, and — in the current display path — `rotate`/`shift` with
    nearest-neighbor `order=0` so it stays boolean), aligned to `im`.

Resolution of which mask to use: when `badpix='repair'` or `return_mask=True` and
no explicit mask/array is supplied, resolve via `load_alcor_badpix_mask` using the
frame's time (filename `YYYY_MM_DD__HH_MM_SS` MST, DATE-header fallback). An
explicit `badpix=<path or array>` overrides resolution. If no mask is found,
repair is a no-op and `mask` is all-False (logged once).

Usage patterns:
- Current pipeline: `badpix='repair'`, `return_mask=False` → repaired `(im, wcs)`.
- Future photometry: `badpix=None`, `return_mask=True` → untouched pixels + the
  bad-pixel mask to OR with the horizon/structure mask into the combined mask
  passed to detection/background.
- QA: `badpix='repair'`, `return_mask=True` → repaired image + the mask repaired.

## Data flow

```
daily cron (host)                       load_alcor_fits(frame)
  <day-dir>/*.fits.bz2                    read raw (3,ny,nx)
   └ select_dark_frames                    └ load_alcor_badpix_mask(frame time)
   └ (>= min_frames?)                          └ nearest alcor_badpix_DATE.fits.gz
   └ build_alcor_median_stack                └ repair (local 5x5 median)  [badpix='repair']
   └ build_alcor_badpix_mask                 └ trim / flipud / (rotate,shift)
   └ write alcor_badpix_DATE.fits.gz         └ return (im[,mask],wcs)
        -> $ALCOR_BADPIX_DIR
```

## Error handling

- Fewer than `--min-frames` dark frames → no mask written, exit 0, message to
  stderr. Nearest existing mask keeps applying.
- Frame shape ≠ mask shape in `load_alcor_fits` → skip repair, all-False mask,
  log once (avoids crashing on odd-sized frames; the night-stack builder already
  skips mismatched frames).
- Empty / missing masks directory → repair no-op, all-False mask, logged once.
- Memmap scratch file is always removed in a `finally`.

## Testing (TDD)

1. `build_alcor_badpix_mask`: synthetic `(3, 5, 5)` cube with a 1-channel spike
   (flagged in that channel), a 2-channel spike (flagged in both), and a
   3-channel spike (excluded from all). Assert exact mask.
2. Repair helper: frame with a known hot pixel + mask → repaired value equals the
   local median; unmasked neighbors unchanged; other channels untouched.
3. `load_alcor_badpix_mask` nearest-date resolution: a temp dir with masks dated
   D1 and D2; a query time nearer D2 returns the D2 mask; empty dir → `None`.
4. `create_badpix_mask` min-frames gate: a directory with < min dark frames
   writes nothing and exits 0.
5. `load_alcor_fits` integration: `badpix='repair'` removes a known hot pixel;
   `return_mask=True` returns a mask aligned to `im` (shape matches, masked
   location maps correctly through trim/flip); default call still returns
   `(im, wcs)`.

## Deferred (separate combined spec)

- `load_alcor_fits` resampling refactor (rotate/shift → WCS).
- Structure mask (buildings + rotating observatory) from the `lum_std` product.
- Combined photometry mask (bad-pixel ∪ structure) and its use in detection /
  background.
