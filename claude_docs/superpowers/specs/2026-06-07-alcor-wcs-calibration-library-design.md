# Alcor WCS calibration library — design

**Date:** 2026-06-07
**Component:** `skycam_utils/alcor.py`

## Goal

The alcor lens geometry drifts over time (mount/focus changes): the 2024-09
calibration gives ~16px residuals on 2026-05 data, while a 2026 fit returns a
different center and a larger radial term. A single set of baked-in constants
cannot serve both eras. Replace the single calibration with a **time-indexed
library of calibration epochs**, and resolve the epoch **nearest in time** to
each image automatically, so `load_alcor_fits` (and everything built on it)
uses the right geometry for any frame, past or future.

## Decisions

- **Storage:** an in-code table in `alcor.py`. New calibrations are added as
  code and committed — consistent with the project's "constants in code"
  convention, version-controlled, and reviewable in PRs.
- **Selection:** nearest epoch in absolute time (before or after the image).
- **Per-epoch fields:** only the four fitted quantities vary per epoch
  (`xshift`, `yshift`, `rotation`, `radial_coeffs`). The trim/scale geometry
  (`radius`, `horizon_radius`, `xcen`, `ycen`) stays fixed as module constants:
  the fit holds them constant and center drift is absorbed by `xshift`/`yshift`.
  If the optical scale ever changes, an epoch dict can grow a `horizon_radius`
  key later without restructuring.
- **Image time source:** parse the `YYYY_MM_DD__HH_MM_SS` filename first (fast,
  no file I/O — see `_filename_ut_datetime`); fall back to the `DATE` header
  only if the filename does not parse.

## Data model

```python
ALCOR_CALIBRATIONS = [
    {"epoch": "2024-09-04", "xshift": -4.570, "yshift": 4.413,
     "rotation": 0.3886, "radial_coeffs": (1.0, 0.01383, 0.0)},
    {"epoch": "2026-05-19", "xshift": <fit>, "yshift": <fit>,
     "rotation": <fit>, "radial_coeffs": (1.0, <fit k3>, 0.0)},
]
```

`epoch` is an ISO date string parsed once to an astropy `Time`. The list is the
single source of truth; the existing module constants (`ALCOR_ROTATION`,
`ALCOR_XSHIFT`, `ALCOR_YSHIFT`, `ALCOR_RADIAL_COEFFS`) are redefined as the
**most-recent epoch's** values so that `_predict_pixels` / `build_alcor_wcs`
defaults and any external references keep working unchanged.

## Resolution

`alcor_calibration(time=None)` returns the calibration dict whose `epoch` is
nearest in absolute time to `time` (an astropy `Time`). An exact tie (an image
equidistant between two epochs) resolves to the more recent epoch. `time=None`
returns the most-recent epoch — the sensible default for one-off, time-agnostic
calls. Epoch `Time`s are computed once (module load or first call) and reused.

A helper resolves an image's time for selection:

```
_alcor_frame_calibration(filename):
    t = _filename_ut_datetime(filename)        # filename first
    if t is None:
        t = Time(fits.getheader(filename)["DATE"], ...)   # header fallback
    return alcor_calibration(Time(t))
```

## `load_alcor_fits` integration

The calibration-bearing kwargs (`rotation`, `xshift`, `yshift`, and the radial
coefficients forwarded to `build_alcor_wcs`) change their **defaults from the
fixed constants to `None`**. When a value is left `None`, `load_alcor_fits`
resolves it from the frame's time via `_alcor_frame_calibration`. Any
explicitly-passed value wins, so callers can pin a calibration or override a
single field. `radius`/`horizon_radius`/`xcen`/`ycen` keep their fixed defaults.

Net effect: `load_alcor_fits("…2026….fits.bz2")` uses the 2026 epoch
automatically, and a keogram spanning two epochs calibrates each frame
correctly because resolution happens per file.

## Adding a calibration (`fit_alcor_wcs`)

`fit_alcor_wcs_cli` currently prints `ALCOR_XSHIFT = …` lines. It changes to
print a **ready-to-paste epoch dict**, stamped with the night's date (derived
from the median resolved frame time):

```python
    {"epoch": "2026-05-19", "xshift": -3.10, "yshift": 1.10,
     "rotation": 0.3013, "radial_coeffs": (1.0, 0.0840, 0.0)},
```

The user pastes it into `ALCOR_CALIBRATIONS` and commits. No automatic
source-file rewriting (fragile, and the in-code table is meant to be reviewed).
The composed-residual reporting (baked + fit residual) is unchanged in spirit:
the printed values are absolute epoch constants ready to drop in.

## Downstream CLIs & back-compat

`plot_alcor_fits`, `alcor_proc_fits`, and `alcor_keogram` reach calibration
only through `load_alcor_fits`, so they inherit per-frame resolution for free.
Their CLI `--rotation` / `--xshift` / `--yshift` (and radial) flags change to
**default `None` (auto-resolve)**, with help text noting that omitting them
resolves by frame date; passing a flag overrides. The keogram/plot/FITS output
APIs are otherwise unchanged.

## Testing

- `alcor_calibration` nearest-in-time selection: a time between two epochs picks
  the closer one; a time before the first epoch picks the first; a time after
  the last picks the last; `time=None` returns the most recent.
- `load_alcor_fits` resolves the correct epoch from a frame's filename
  (synthetic frames named in each epoch), and an explicit kwarg overrides
  resolution.
- Time-source fallback: a frame whose name does not parse falls back to its
  `DATE` header.
- `fit_alcor_wcs` prints a parseable epoch dict whose `epoch` matches the input
  night.

## Out of scope

- No change to the WCS math, the k3-only robust fit, or the keogram/plot/FITS
  output formats.
- No automatic editing of `ALCOR_CALIBRATIONS` by the fitter.
- No per-epoch trim/scale geometry (deferred until a real scale change appears).

## Files changed

- `skycam_utils/alcor.py` — `ALCOR_CALIBRATIONS` table, `alcor_calibration()`,
  `_alcor_frame_calibration()`, `load_alcor_fits` default/resolution changes,
  `fit_alcor_wcs_cli` output change, downstream CLI flag defaults.
- `skycam_utils/tests/test_alcor_wcs.py` — selection, resolution, fallback, and
  fit-output tests.
- `CLAUDE.md` — document the calibration library and per-frame resolution.
