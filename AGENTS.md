# Repository Guidelines

## Project Structure & Module Organization

`skycam_utils/` contains the installable Python package for all-sky camera processing. Core modules include `pipeline.py` for Stellacam orchestration, `photometry.py` and `astrometry.py` for library routines, `alcor.py` for Alcor FITS processing and plotting, and `fit_wcs.py` for WCS fitting helpers. Tests live under `skycam_utils/tests/`, though coverage is currently sparse. Operational shell and telemetry tools are in `scripts/`; these are not packaged and often depend on observatory host paths. Sphinx documentation is in `docs/`, exploratory notebooks are in `notebooks/`, and license materials are in `licenses/`.

## Build, Test, and Development Commands

Use `rtk` before shell commands in this repository.

```bash
rtk pip install -e ".[test]"
rtk pytest
rtk pytest skycam_utils/tests/<file>.py::<test>
rtk pip install -e ".[docs]"
rtk sphinx-build -W -b html docs docs/_build/html
rtk python -m build
```

`pip install -e ".[test]"` installs the package in editable mode with pytest tooling. `pytest` runs package tests plus configured doctests in `docs/` and `.rst` files. `sphinx-build -W` treats documentation warnings as failures. `python -m build` creates source and wheel distributions using `setuptools` and `setuptools_scm`.

## Coding Style & Naming Conventions

Target Python 3.9+. Follow PEP 8 conventions with 4-space indentation, `snake_case` functions and variables, and descriptive module-level helpers. Keep CLI functions thin and put reusable behavior in package modules. Preserve existing FITS/WCS conventions, especially lower-origin FITS handling and year-keyed calibration behavior. Generated version files such as `skycam_utils/_version.py` should not be edited manually.

## Testing Guidelines

Use pytest for unit tests and doctests. Add tests near the code under `skycam_utils/tests/` with names like `test_alcor.py` and functions named `test_<behavior>`. For changes involving FITS output, WCS transforms, or date parsing, include focused regression tests where practical. Do not assume existing coverage protects modified behavior; add explicit tests for new edge cases.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, lowercase commit subjects such as `fix output path derivation for compressed alcor inputs` and `add conda-forge environment file`. Keep commits focused and describe user-visible behavior. Pull requests should include a concise summary, test commands run, linked issues when applicable, and before/after artifacts for plotting or image-output changes.

## Operational Notes

Be careful with `scripts/`: absolute `/mnt/...` paths, ffmpeg options, Redis defaults, and cron-style assumptions are load-bearing. `astrometry.solve_field()` requires the external `solve-field` binary and installed astrometry.net indexes; this is not provided by Python dependencies.
