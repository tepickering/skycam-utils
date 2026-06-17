# Alcor: Collect Per-Star Photometry Across Frames — Design

**Date:** 2026-06-12
**Status:** Approved design, pending implementation plan

## Problem

`alcor_star_photometry` writes one `*_phot.csv` per frame (name-indexed rows of
per-channel flux/mag/background). A night produces ~1300 of them
(e.g. `~/MMT/skycam_data/2024-09-04`). To calibrate camera flux/magnitude
against catalog photometry we need each star's measurements gathered across
all frames as a time series.

## Decision

A single library function — no CLI, no per-star output files. The product is
an in-memory DataFrame for interactive calibration work; `df.to_csv()` covers
any export need.

## `collect_alcor_photometry(inputs)`

Lives in `alcor.py`. Returns one combined pandas DataFrame.

- **`inputs`**: a directory (globbed for `*_phot.csv`) or an iterable of CSV
  paths.
- **Columns**: the original photometry columns (`altitude`, `azimuth`, `xcen`,
  `ycen`, `flux_*`, `mag_*`, `background_*`) plus:
  - `name` — the star label, kept as a regular column (not the index);
  - `OBSTIME` — pandas datetime in UT, parsed from each file's
    `YYYY_MM_DD__HH_MM_SS` filename stamp (camera local MST, so UT =
    stamp + 7 h). Reuses the existing filename-timestamp parsing in
    `alcor.py`.
- **Order**: rows sorted by `name` then `OBSTIME`, so
  `df.groupby("name")` yields each star's time-ordered light curve directly.
- **Errors**:
  - a file whose name has no parseable timestamp is skipped with a stderr
    warning (the CSVs carry no internal time information, so there is no
    fallback);
  - a malformed/unreadable CSV is skipped with a stderr warning;
  - if nothing usable remains (empty input set or all files skipped), raise
    `ValueError` with a clear message.

## Downstream (out of scope)

Calibration against catalog photometry: iterate `df.groupby("name")` and join
with `lookup_sloan_photometry(name)` per channel to derive instrumental →
catalog magnitude transformations. Deliberately not part of this change.

## Testing

Unit tests in `test_alcor.py` against synthetic `*_phot.csv` files in
`tmp_path`:

1. multi-file collection groups rows per star with correct columns;
2. MST→UT conversion of the filename stamp;
3. sort order (`name`, then `OBSTIME`);
4. unparseable filename skipped with a warning, rest still collected;
5. empty/no-usable-input raises `ValueError`;
6. directory input and explicit-file-list input are equivalent.
