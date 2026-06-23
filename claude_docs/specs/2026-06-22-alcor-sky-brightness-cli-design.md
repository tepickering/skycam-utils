# `alcor_sky_brightness` CLI — design

2026-06-22

## Goal

A new CLI, `alcor_sky_brightness`, that turns one raw Alcor OMEA 8C
`.fits.bz2` frame into a calibrated **V mag/arcsec²** 2-D FITS data product with
the raw-frame alt/az WCS attached. It is the FITS-output sibling of
`plot_alcor_sky_brightness`, which only renders an annotated figure.

## Pipeline

Reuses the already-validated surface-brightness calibration chain (no new
photometry):

1. `load_alcor_fits(filename, badpix="repair")` → raw `(3, ny, nx)` cube + the
   epoch-resolved WCS, with bad pixels repaired ("fixes bad pixels").
2. Green channel only: subtract the per-channel corner bias (`_corner_bias`),
   then scale counts to the 20 s reference exposure (`ALCOR_CALIB_EXPTIME`) via
   the frame's `EXPOSURE` header (counts are linear in exposure).
3. Divide by the per-pixel solid angle from the WCS
   (`_alcor_pixel_solid_angle`, exact for the ARC plate-scale change + SIP).
4. Apply the epoch G→V zeropoint (`alcor_zeropoint(time)["g"]["zp"]`), **no
   airmass term** (this is the *observed* sky brightness):

   ```
   mu = -2.5 * log10(g20 / omega_arcsec2) + zp_g
   ```

   NaN where `g20/omega` is non-positive.

## Output

- Full frame, **native orientation**, `float32` (matches `alcor_proc_fits`, so
  the attached WCS resolves correctly in DS9). No cropping/rotation.
- NaN blanking, by default: off-frame (WCS does not project) ∪
  `raw G ≥ saturation` (`ALCOR_SB_SATURATION = 25000`, clipped/non-linear).
  Every other on-sky pixel keeps its calibrated value, including below the
  geometric horizon.
- `--horizon-mask`: additionally NaN the not-sky region from
  `load_alcor_horizon_mask(time)` (terrain/buildings/lightning-rod + below
  horizon). Masking is a clean opt-in; there is no default altitude floor.
- Header: full WCS via `wcs.to_header(relax=True)`, plus `BUNIT='mag/arcsec2'`
  and provenance keywords: `ZP_G` (zeropoint used), `ZP_EPOCH` (zeropoint epoch
  date), `EXPOSURE` (frame exposure, s), `CALIBEXP` (`ALCOR_CALIB_EXPTIME`),
  `SATLEVEL` (`saturation`), `HORIZMSK` (bool: horizon mask applied).

## Default output path

Strip the first of `.fits.bz2` / `.fits.gz` / `.fits` from the input and append
`_sb.fits` (same stem logic as `alcor_proc_fits`'s `_proc.fits`). Example:
`2024_09_04__19_00_00.fits.bz2` → `2024_09_04__19_00_00_sb.fits`. `-o/--output`
overrides; `--overwrite` clobbers an existing file.

## Refactor (shared core)

Extract calibration steps 2–4 into a small helper so the new function and
`plot_alcor_sky_brightness` stop duplicating the math:

```python
def _alcor_sky_brightness_map(cube, wcs, time, exposure,
                              saturation=ALCOR_SB_SATURATION):
    """Full-frame (mu, alt): V mag/arcsec^2 with off-frame + saturated pixels
    blanked to NaN. Geometric (horizon/altitude) masking is left to callers."""
```

Geometric masking stays in each caller because their default policy differs:
`plot_alcor_sky_brightness` keeps its `fov_altitude=-2` floor; the FITS product
applies no floor (only `--horizon-mask`). This refactor must not change the
existing plot output.

## Public API

```python
def alcor_sky_brightness_fits(filename, output_file=None, horizon_mask=False,
                              saturation=ALCOR_SB_SATURATION, overwrite=False,
                              **kwargs):
    """Write the calibrated V mag/arcsec^2 FITS; return the output Path.
    **kwargs forwarded to load_alcor_fits (wcs, masks_dir, ...); badpix defaults
    to "repair"."""
```

A `main()` parses the CLI and registers in `pyproject.toml [project.scripts]`
as `alcor_sky_brightness`.

## CLI

```
alcor_sky_brightness <input.fits[.bz2|.gz]> [-o OUT.fits] [--horizon-mask]
                     [--saturation 25000] [--overwrite]
```

Minimal by design: no `vmin`/`vmax`/`cmap`/`radius`/`fov-altitude` (those are
display-only and irrelevant to a data FITS).

## Tests (`test_alcor_skybright.py`)

- Writes `<input>_sb.fits`; default path derivation correct.
- Data is 2-D `float32` with shape `(ny, nx)`; `BUNIT == 'mag/arcsec2'`.
- Header WCS round-trips (CTYPE `RA---ARC`/`DEC--ARC`; CRPIX matches the loaded
  WCS, mirroring `test_alcor_proc_fits_writes_processed_cube_and_header`).
- Finite zenith pixels (alt > 85°) land in 20–23 mag/arcsec² (dark-sky sanity).
- `--horizon-mask` path runs and produces additional NaNs.
- The refactor leaves `plot_alcor_sky_brightness`'s existing tests green.

## Docs

Add the `alcor_sky_brightness` block to the CLI list in `CLAUDE.md` (and the
mirrored `AGENTS.md` symlink updates automatically).
