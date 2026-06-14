# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

`skycam_utils` is an MMT-Observatory utility package for analyzing all-sky camera images. It supports three distinct camera systems whose data formats and calibration assets differ:

- **Stellacam** (the main pipeline target — `pipeline.py`). Header conventions changed across eras, so `get_ut()` parses `UT`/`DATE` differently for `year < 2013` vs. later years.
- **ASI** (ASI1600). Handled by `process_asi_image()`; runs astrometry.net `solve-field` on a central cutout because the full all-sky image is unsolvable directly.
- **Alcor OMEA 8C** (`alcor.py`). RGB FITS. The WCS is the single source of geometry: `load_alcor_fits()` returns the raw `(3, ny, nx)` RGB cube **untouched** (no transpose/trim/rotate/shift/flipud, and no bias subtraction — only optional bad-pixel repair) plus a **raw-frame** ARC-projection WCS, as the 3-tuple `(cube, wcs, mask)` (`mask` is the bad-pixel mask aligned to the cube, or `None`). The WCS maps raw pixel ↔ (azimuth, altitude) with altitude=0 at `horizon_radius` pixels from the zenith, encoding all geometry: the zenith offset in CRPIX, the camera rotation in the PC matrix (a pure rotation, det=+1; the sky/sensor handedness lives in the `rotation - az` azimuth convention, since an all-sky camera images the sky from below — north lands toward +y), and the fitted odd-power radial (k3/k5) lens distortion as an exact analytic SIP, plus Brown–Conrady tangential decentering terms (P1/P2, fit always, exact degree-2 SIP) that absorb the sensor-tilt signature (once-per-azimuth residual growing as r²), plus a world-side optical-axis tilt (`axis_tilt`, fit always, encoded as the WCS pole: CRVAL=(A0, 90−ε), LONPOLE=A0) — with nonzero tilt `xcen`/`ycen` is the optical-axis pixel and the zenith must be located via the WCS (alt=90), not CRPIX. It is calibrated by `fit_alcor_wcs()` against `bright_star_sloan.fits` (Vmag <= 4) over dark-sky frames (Sun < -18 deg, Moon < -6 deg); the camera's `DATE` header is the true UT (`DATE-OBS` is local time despite its label). The fitted geometry — absolute raw-frame constants `xcen`/`ycen`/`rotation`/`radial_coeffs`/`tangential_coeffs`/`axis_tilt`/`horizon_radius` (`tangential_coeffs` and `axis_tilt` are optional, `(0, 0)` when absent; `axis_tilt` is in degrees toward north/east) — lives in a time-indexed `ALCOR_CALIBRATIONS` table; when `wcs=None`, `load_alcor_fits` resolves the epoch nearest a frame's time (filename timestamp, DATE-header fallback) and builds the WCS, so frames from different eras (e.g. 2024 vs 2026) each get the right geometry; pass an explicit `wcs=` to override. (`fit_alcor_wcs` aggregates matches across a whole night; on clean dark frames it matches ~80 stars/frame; with the full model (k5 + P1/P2 + axis tilt) a healthy fit has a matched fraction near 0.7 and a pooled RMS of ~0.35px — the residual floor is a smooth, azimuthally-symmetric ~0.6px p-p radial wiggle from k3/k5 polynomial truncation, not worth chasing. Two gotchas that silently wreck the fit, both fixed: detection MUST run on the bad-pixel-repaired cube — `_detect_alcor_frame` loads `badpix="repair"` — or hot pixels masquerade as stars; and `_fit_params` must carry `horizon_radius` through its returned dict, or the matcher reverts to the module-default `ALCOR_HORIZON_RADIUS` mid-fit and the pool collapses while k3 runs away.) Visualization routines do their own crop and use `origin="lower"`: `plot_alcor_fits()` builds the display RGB (empirically-tuned per-channel `gscale`/`bscale` + power+ZScale stretch), crops a `radius`-pixel square around the WCS zenith, and renders north-up; `alcor_proc_fits()` writes the raw cube + WCS header directly (native orientation, so DS9 shows the camera's native frame while the WCS resolves correctly). The matcher (`assign_alcor_matches`) is seeded by the resolved epoch geometry and uses a kd-tree with local-asterism pattern verification plus a local relative-brightness tie-break (cloud extinction is patchy, so only relative flux among nearby contested stars is trusted) rather than per-frame nearest-neighbor refitting. Star photometry: `alcor_star_photometry()` measures fixed-position per-channel (R/G/B) aperture+annulus photometry at WCS-predicted positions — no source detection — for named bright stars from `bright_star_sloan_named.fits` (the named variant of the calibration catalog; `alcor_named_reference_altaz()` loads it, `lookup_sloan_photometry(star_name)` returns one star's Sloan row by NAME). Frames with the Sun above `sun_alt_max` (-12 deg default) are rejected (empty DataFrame, no CSV). The per-channel bias (median of the four 10x10 image corners, `_corner_bias`) is subtracted before measuring — and the same corner-bias subtraction now feeds the display stretch via `_alcor_display_rgb` (otherwise the raw pedestal gets color-scaled and the image turns purple); the raw cube from `load_alcor_fits` stays untouched. Results are a pandas DataFrame (pandas is a core dependency) indexed by star name (HD-number fallback, duplicate labels suffixed), with instrumental mags `-2.5*log10(flux)` and a per-channel `sat_*` boolean (True when any raw aperture pixel reaches `saturation`, default `ALCOR_SATURATION = 32767`, the camera's 15-bit ceiling — checked on the raw cube, not the bias-subtracted data, so bright stars saturated in one channel can be filtered per-channel downstream without losing the others); stars with non-finite or non-positive flux in any channel are dropped, rows sort by descending `flux_g`, and a CSV is always written (`<input>_phot.csv` default). A Gaussian-fit path (`gaussian=True`, CLI `--gaussian`) handles the bright-star CMOS non-linearity that suppresses aperture-sum flux before saturation: it fits a circular Gaussian whose center and width are pinned from a luminance (R+G+B) fit with the non-linear core masked (raw pixels >= `mask_threshold`, default `ALCOR_NONLINEAR_THRESHOLD = 15000`, excluded), recovers each channel's amplitude as the linear projection of the masked, background-subtracted aperture onto the fixed unit-Gaussian profile (so the linear wings set the amplitude), and reports the analytic Gaussian integral `2*pi*A*sigma^2` as `flux_*`; the shared luminance FWHM is added as an `fwhm` column (NaN in aperture mode), `xcen`/`ycen` then hold the fitted center, and `sat_*` still reflects raw-core saturation (now informational). `collect_alcor_photometry(inputs)` gathers a set of those per-frame CSVs (a directory globbed for `*_phot.csv`, or an explicit path list) into one combined DataFrame with a UT `OBSTIME` column (parsed from each `YYYY_MM_DD__HH_MM_SS` filename, MST + 7h) and `name` as a regular column, sorted by `name` then `OBSTIME` so `df.groupby("name")` yields each star's time-ordered light curve for calibration against catalog photometry; unparseable filenames and malformed CSVs are skipped with stderr warnings, and an empty result raises `ValueError`.

Packaging is PEP 621 / `pyproject.toml`-only — there is no `setup.py`, `setup.cfg`, `MANIFEST.in`, or `tox.ini`. The version is generated by `setuptools_scm` into `skycam_utils/_version.py` at build/install time (gitignored). `AGENTS.md` is a symlink to this file — edit `CLAUDE.md` only.

## Common commands

```bash
# Install for development (editable, with test extras)
pip install -e ".[test]"

# Run the test suite
pytest
# Run a single test file:
pytest skycam_utils/tests/<file>.py::<test>

# Build the package
python -m build

# Build docs
pip install ".[docs]" && sphinx-build -W -b html docs docs/_build/html

# CLI entry points (installed via [project.scripts])
process_stellacam_dir <YYYYMMDD-dir> [--writefits] [--zeropoint Z] [--nproc N] [-z] [-s]
#   -z       : process *.fits.gz instead of *.fits
#   -s       : produce strip image + plot instead of per-frame photometry
#   The directory NAME must start with YYYY — that's how `year` is derived,
#   which selects the WCS and mask files via load_wcs()/load_mask().

alcor_proc_fits <input.fits> [-o OUT] [--overwrite]
#   Writes the raw (3, ny, nx) RGB cube (native orientation) with the raw-frame
#   alt/az WCS in the header. Geometry comes from the WCS, so there are no
#   geometry flags. Default output: <input>_proc.fits.

plot_alcor_fits <input.fits> [-o OUT.pdf] [--outimage RAW] [--radius 680] [--gscale ...] ...
#   Renders an annotated all-sky figure, cropping a --radius-pixel square around
#   the WCS zenith and rendering north-up (origin="lower"). --radius is the
#   display crop only. Default output: <input>.pdf (extension drives the backend).

alcor_star_photometry <input.fits> [-o OUT.csv] [--aperture-radius 4] [--annulus-width 1] [--min-altitude 20] [--vmag-limit 5.5] [--no-refraction] [--sun-alt-max -12] [--saturation 32767] [--gaussian] [--mask-threshold 15000] [--check-plot] [--check-radius 680]
#   Fixed-position RGB aperture photometry of named bright stars at their
#   WCS-predicted pixels (no detection step). Writes <input>_phot.csv indexed
#   by star name, with a per-channel sat_* flag (raw aperture pixel >=
#   --saturation); --check-plot overlays the apertures on the plot_alcor_fits
#   rendering as <input>_phot.pdf. Prints the warning and writes nothing when
#   the Sun is above --sun-alt-max.
#   --gaussian switches to constrained-Gaussian PSF photometry (luminance-pinned
#   center/width, non-linear core masked at --mask-threshold, analytic-integral
#   flux) to recover bright-star flux lost to CMOS non-linearity; it adds an fwhm
#   column. --aperture-radius also sets the Gaussian fit window.

fit_alcor_wcs <night-dir> [--pattern ...] [--vmag-limit 4] [--tolerance 3] [--max-detections 200] [--sun-alt-max -18] [--moon-alt-max -6] [--residual-plot OUT.png] [--max-frames N] [--workers N] [--quiet]
#   Aggregates bright-star matches across dark frames across a night and prints
#   a ready-to-paste ALCOR_CALIBRATIONS epoch dict (absolute raw-frame constants —
#   xcen, ycen, rotation, radial_coeffs, tangential_coeffs, axis_tilt, horizon_radius — stamped with the night's
#   UT date) to add to alcor.py and commit.
#   Detections are capped to the brightest --max-detections per frame; matching is
#   seeded by the nearest epoch and done with assign_alcor_matches (cKDTree candidate
#   search, local-asterism pattern verification, and a local relative-brightness
#   tie-break for contested detections), with no per-frame geometry refit. The match
#   tolerance tightens over several rounds from ~12px to --tolerance (~3px). Also
#   prints the matched fraction so contamination/coverage is visible.
#   Dark-frame selection keeps frames with Sun < --sun-alt-max AND Moon < --moon-alt-max
#   (-6deg default; moonlight scatter swamps the faint star field and corrupts detection;
#   pass --moon-alt-max 90 to disable). It parses the UT from each YYYY_MM_DD__HH_MM_SS
#   filename (local MST = UT-7), so it never opens files; oddly-named files fall back to
#   the DATE header.
#   Per-frame load/detect is parallelized across processes (--workers; default: one per core).
#   Prints each file's disposition to stderr (Sun-rejected / no stars / used + count); --quiet silences it.
```

Test coverage is concentrated on the Alcor module (`test_alcor.py`, `test_alcor_wcs.py`, `test_alcor_badpix.py`, run against the bundled `test.fits.bz2` frame and synthetic geometry). The Stellacam pipeline and photometry/astrometry modules have no tests — don't assume coverage exists for code you change there.

## Pipeline architecture

`pipeline.py` is the orchestrator; the photometry/astrometry modules are libraries it composes:

```
process_stellacam_image(fitsfile, year)
  ├─ get_ut(hdr, year)                       # year-dependent header parsing
  ├─ Filters frames: only FRAME='256 Frames' AND GAIN=106 are processed
  │                  (the dark-sky steady-state config — other configs are skipped, not failed)
  ├─ load_mask(year)        / load_wcs(year) # year → packaged FITS in skycam_utils/data/
  ├─ load_skycam_catalog()  → update_altaz() # apply current-time AltAz to the curated star catalog
  ├─ photometry.make_background()            # photutils Background2D, optional source masking
  ├─ photometry.make_segmentation_image()    # detect_sources + deblend
  ├─ photometry.make_catalog()               # source_properties → table with obs_mag
  ├─ photometry.match_stars()                # WCS pix→AltAz, match against skycat by 2.5° sep
  └─ writes .cat.csv (always); .bkg/.subt/.sky FITS only when --writefits
```

`process_stellacam_dir()` then groups all per-frame `.cat.csv` outputs by `Star Name` and writes one `star_<name>.csv` per matched star. Per-frame work is parallelized via `multiprocessing.Pool`.

### Year-keyed calibration assets

`load_wcs()` and `load_mask()` map year → file. Currently supported buckets: 2011–2012, 2015–2016, 2017–2021. **Adding a new year requires extending both functions** and shipping the corresponding FITS in `skycam_utils/data/` (covered by the `skycam_utils = ["data/*"]` glob in `[tool.setuptools.package-data]`).

### WCS fitting (`fit_wcs.py`)

Borrowed from LSST. Provides `wcs_zea` / `wcs_azp` callable classes used as objective functions for `scipy.optimize.minimize` in `astrometry.initial_wcs_fit()`. `wcs_sip_fit()` then refines with SIP distortion via `astropy.wcs.utils.fit_wcs_from_points`. SIP terms only survive a write if you use `write_sip()` — `to_fits()` drops them by default.

## `scripts/` is operational, not packaged

These run on the live skycam host (Windows WSL — paths like `/mnt/d/skycam/...`, `/mnt/c/Users/skycam/...`) and on `ops.mmto.arizona.edu`. They are **not** part of the installed Python package and are not on PYTHONPATH:

- `make_movies.sh` / `daily_movies.sh` — ffmpeg concat-demuxer pipelines that produce `allsky.mp4` and `unwrap.mp4`. The `pad=ceil(iw/2)*2:ceil(ih/2)*2` filter is required because H.264 needs even dimensions; do not remove it. Bitrate/framerate constants in these scripts have been hand-tuned over many commits — change them only when explicitly asked.
- `latest_image.sh` / `latest_movie.sh` — scp/ffmpeg jobs that publish the most-recent image and a rolling 150-frame movie to the public web host.
- `mmt_position.py` / `tcs_logger.py` — Redis publishers reading TCS state and weather telemetry. Default Redis host is `redis.mmto.arizona.edu`; override via `REDISHOST` / `REDISPORT` / `REDISPW` env vars. `mmt_position.py` writes the alcor-format Dublin-JD timestamp (`jd - 2415020 + 1.5`) plus RA/Dec/LST in radians to `/mnt/d/skycam/mmt_position.txt`.

When editing scripts under `scripts/`, assume the absolute paths and the cron/systemd context they run under are load-bearing.

## External dependencies

- **astrometry.net** — `astrometry.solve_field()` shells out to `solve-field` with hardcoded `-L 100 -H 150 -u app` (arcsec-per-pixel scale bounds) and `--no-background-subtraction`. Requires the binary on `PATH` plus appropriate index files installed; it's not a Python dependency.
- **redis** — only needed by the `scripts/` operational tools, not the photometry pipeline.
