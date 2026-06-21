# Alcor Horizon-Mask Regeneration CLIs — Design

**Date:** 2026-06-20
**Status:** Design (pending review)

## Motivation

The date-indexed Alcor horizon mask (`load_alcor_horizon_mask`) ships as package
data and is resolved automatically from the installed package. But its
**regeneration path** is currently a pair of unpackaged reference scripts under
`claude_docs/scripts/` (`alcor_median_stack.py` → `horizon_floodfill.py`) with
hardcoded local paths. The documentation points at those repo-relative scripts,
which do not exist for anyone who `pip install`ed the package, and the algorithm
lives only in a script, not in the library.

This spec promotes that regeneration path to two packaged CLIs that follow the
existing `create_badpix_mask` pattern (pure builder + I/O wrapper + CLI), so the
horizon mask is rebuilt with installed tools and the algorithm has a single home
in `skycam_utils.alcor`. It does **not** change the mask format, the loader, or
the shipped mask asset.

Like `create_badpix_mask`, these CLIs require **local raw observational data**
(a cloudy night's frames; optionally several nights of per-frame photometry).
That is consistent with the project's existing philosophy — `create_badpix_mask`
and `fit_alcor_wcs` are packaged CLIs that also need local data and are not
reproducible from a bare pip install.

## Background: the existing algorithm

`horizon_floodfill.py` builds the mask as a Sobel-edge flood-fill of a
cloudy-night luminance median:

1. **Median input.** A 2-D luminance (R+G+B) per-pixel median of a
   cloudy/overcast night. The smooth overcast sky leaves only sharp obstruction
   edges. Produced today by `alcor_median_stack.py`, written as
   `<tag>_median.fits` with the raw-frame WCS in the header.
2. **Sobel walls + flood-fill.** Strong `sobel(log10 median)` edges (above the
   `EDGE_PCT=96` percentile within the FOV, dilated once) are treated as walls;
   the sky is flood-filled outward from the WCS zenith. Everything the fill
   cannot reach (terrain, buildings, the fully enclosed lightning-rod spike) is
   not-sky.
3. **Undetected-star patch.** In the SW→W building sector (az 225–270°) the Sobel
   edges break up, so the wall there comes from an az/alt **undetected-fraction
   grid** accumulated from fixed-position per-frame photometry (`*_phot.csv`,
   `flux_g_ap == 0` ⇒ undetected) over several nights. A cell with undetected
   fraction `≥ UND_THR=0.5` and `≥ UND_MINCOUNT=15` transits is obstruction.
4. **Cleanup.** A morphological opening (`OPEN_RADIUS=3`) severs thin necks; a
   connected-component pass keeps a not-sky blob only if it touches the rim
   (`min_alt < RIM_ALT=1.5`) or is rod-sized (`size ≥ ROD_AREA_MIN=400`).
5. **Output.** `horizon_mask = (~in_fov) | notsky`, written as
   `alcor_horizon_<EPOCH>.fits.gz` (uint8) with method/parameter header cards.

The WCS is rebuilt from the epoch via `alcor_calibration` + `build_alcor_wcs`,
not read from the median header.

## Design decisions

Two forks were settled with the user before this spec:

- **Median input — prebuilt FITS.** Choosing a *cloudy* night is a human
  judgment, so median-building stays its own step. `alcor_median_stack.py` is
  promoted to a CLI; `create_horizon_mask` consumes the median FITS it produces.
- **Undetected patch — optional `--phot-nights`, Sobel fallback.** No new shipped
  asset. When `--phot-nights` is given, the grid is accumulated on demand; when
  omitted, the SW→W sector is Sobel-only and the CLI warns. Re-run with fresh
  photometry after a camera move.

## New public API (all in `skycam_utils/alcor.py`)

### Median stack

```python
def build_alcor_luminance_median(dark_files, badpix=True, max_frames=None,
                                 scratch_dir=None, tile=50, log=None):
    """Per-pixel 2-D luminance (R+G+B) median over raw frames -> float32 (ny, nx)."""

def alcor_median_stack(night_dir, out_path=None, sun_alt_max=-18.0,
                       moon_alt_max=90.0, max_frames=None, scratch_dir=None,
                       pattern="*.fits.bz2", log=None):
    """Select dark frames from one (cloudy) night, build the luminance median,
    and write <tag>_median.fits with the raw-frame WCS header. Returns the path."""

def alcor_median_stack_cli():
    """CLI entry point for alcor_median_stack."""
```

- A **new** luminance builder is required: the existing `build_alcor_median_stack`
  returns the `(3, ny, nx)` RGB cube. The luminance variant collapses channels
  (`sum(axis=0)`), reusing the same RAM-bounded memmap + row-tile median
  approach and the same fast raw `fits` read (not the heavier
  `load_alcor_fits` per-frame path). The per-pixel median is already
  hot-pixel-robust; `badpix=True` additionally zeroes out the resolved
  per-channel mask before collapsing (faithful to `alcor_median_stack.py`), and
  can be disabled.
- `moon_alt_max` defaults to `90` (disabled): a cloudy night is selected by the
  human, and overcast frames should not be filtered on moon altitude.
- The WCS is resolved for the night's epoch (`alcor_calibration` →
  `build_alcor_wcs`) and written into the header so the median FITS is
  self-describing in DS9.

### Horizon mask

```python
def build_alcor_horizon_mask(median_img, wcs, undetected=None, *,
                             edge_pct=96.0, edge_dilate=1, open_radius=3,
                             sector=(225.0, 270.0), und_thr=0.5, und_mincount=15,
                             rim_alt=1.5, rod_area_min=400):
    """Pure builder: 2-D luminance median + WCS -> bool not-sky mask.
    `undetected` is an optional (fraction, count) az/alt sampler for the SW->W
    sector; None -> that sector is Sobel-only."""

def create_horizon_mask(median_path, epoch=None, out_dir=None, phot_nights=None,
                        edge_pct=96.0, edge_dilate=1, open_radius=3,
                        sector=(225.0, 270.0), und_thr=0.5, und_mincount=15,
                        rim_alt=1.5, rod_area_min=400, log=None):
    """Load the median, resolve the epoch WCS, optionally accumulate the
    undetected-fraction grid from phot_nights, build the mask, and write
    alcor_horizon_YYYY-MM-DD.fits.gz to out_dir. Returns the output Path."""

def create_horizon_mask_cli():
    """CLI entry point for create_horizon_mask."""
```

- All of the script's tuning constants become keyword arguments at their current
  values, so the committed `2026-02-18` mask is reproducible.
- **`epoch` resolution:** explicit `epoch` wins; else parse `YYYY-MM-DD` (or
  `YYYY_MM_DD`) from the median filename; else error asking for `--epoch`. The
  epoch both selects the calibration (WCS) and date-stamps the output.
- **Undetected grid** is factored into a private helper
  `_alcor_undetected_fraction(phot_nights, wcs, sector, ...)` that reads
  `xcen`/`ycen`/`flux_g_ap` from each night's `*_phot.csv`, projects to az/alt,
  and returns `RegularGridInterpolator`s for fraction and count (the script's
  inline logic). Returned to `build_alcor_horizon_mask` as `undetected=`.
- **No figures** in the CLI. It prints the output path and a one-line summary
  (`sky px … / not-sky … (…% of FOV); dropped N pockets`), mirroring
  `create_badpix_mask`. `--quiet` silences the per-step `log`.
- Output dir defaults to the resolved horizon dir (`$ALCOR_HORIZON_DIR` →
  packaged `data/horizon/`), exactly like `create_badpix_mask`'s badpix dir.

## CLI surface

```
alcor_median_stack <night-dir> [-o OUT.fits] [--sun-alt-max -18] [--moon-alt-max 90]
                   [--max-frames N] [--scratch-dir DIR] [--pattern *.fits.bz2] [--quiet]

create_horizon_mask <median.fits> [--epoch YYYY-MM-DD] [--out-dir DIR]
                    [--phot-nights DIR [DIR ...]] [--edge-pct 96] [--edge-dilate 1]
                    [--open-radius 3] [--sector 225 270] [--und-thr 0.5]
                    [--und-mincount 15] [--rim-alt 1.5] [--rod-area-min 400] [--quiet]
```

`[project.scripts]` in `pyproject.toml` gains:

```
alcor_median_stack  = "skycam_utils.alcor:alcor_median_stack_cli"
create_horizon_mask = "skycam_utils.alcor:create_horizon_mask_cli"
```

## Reference script

`claude_docs/scripts/horizon_floodfill.py` is slimmed to **import**
`build_alcor_horizon_mask` (and `_alcor_undetected_fraction`) from the package
and keep only the two-panel diagnostic plotting that produces
`horizon_floodfill.png` / `horizon_floodfill_south.png`. The algorithm then lives
in exactly one place. `alcor_median_stack.py` can be reduced to a thin call to
the packaged function or left as-is; it is not load-bearing once the CLI exists.

## Documentation

- **`docs/horizon_mask.rst`** (covers the package-data ask):
  - Reword the "Date-resolved" property so it states the mask **ships as package
    data** and is resolved automatically from the installed location by
    `load_alcor_horizon_mask` — drop the implication that a repo path
    (`skycam_utils/data/horizon/...`) is what callers use. Keep the loader
    snippet.
  - Replace "There is **no packaged build CLI**…" with the two-command
    regeneration path (`alcor_median_stack` → `create_horizon_mask
    [--phot-nights …]`), and note the reference script now only renders the
    diagnostic figures.
- **`CLAUDE.md`**: add `alcor_median_stack` and `create_horizon_mask` to the CLI
  block; update the horizon-mask paragraph that currently says "There is **no
  packaged build CLI**".

## Testing

Extend `skycam_utils/tests/test_alcor_horizon.py` with a synthetic build test
(mirroring the existing synthetic-geometry tests, no real data):

- Construct a small synthetic raw-frame WCS (as the other horizon/WCS tests do)
  and a smooth bright luminance disk over the FOV.
- Carve a **dark obstruction wedge** that reaches the rim and a small **enclosed
  dark spike** (lightning-rod analog).
- `build_alcor_horizon_mask(median, wcs)` (no `undetected`) must: mask the wedge,
  mask the enclosed spike, mask everything at/below alt 0, and leave open sky
  (`~mask`) for the clear region.
- A small assertion that an isolated sub-`rod_area_min` open-sky pocket is *not*
  retained as not-sky (pocket cleanup) and that a rim-touching blob *is*.

`create_horizon_mask` I/O (epoch parsing, output path, header cards) is covered
by a thin test that writes the synthetic median to a temp FITS and checks the
returned path name and `NMASK`/`NSKY` header cards — analogous to the badpix
mask-writing test.

## Out of scope

- No change to `load_alcor_horizon_mask`, the mask file format, or the shipped
  `alcor_horizon_2026-02-18.fits.gz`.
- No committed undetected-fraction asset (the SW→W patch is accumulated on
  demand from `--phot-nights`).
- No new epoch / camera-move handling beyond what date-stamped output already
  provides.
