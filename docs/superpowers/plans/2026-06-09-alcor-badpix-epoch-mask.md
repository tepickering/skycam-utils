# Alcor Bad-Pixel Epoch-Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect alcor sensor hot pixels per channel, regenerate the mask daily as a date-stamped epoch product, and repair them in `load_alcor_fits` (with an option to return the mask instead).

**Architecture:** Five units in `skycam_utils/alcor.py` — a pure detector (`build_alcor_badpix_mask`), a RAM-bounded median-stack builder (`build_alcor_median_stack`), file storage + nearest-date resolution (`load_alcor_badpix_mask`), an in-place repair helper (`_apply_badpix_repair`), and `load_alcor_fits` integration via `badpix`/`return_mask` params — plus a `create_badpix_mask` CLI run from the daily cron. Masks are gzipped per-channel `(3, ny, nx)` uint8 FITS resolved nearest-in-date, mirroring the time-indexed `ALCOR_CALIBRATIONS` pattern but stored as files.

**Tech Stack:** Python, numpy, scipy.ndimage (`median_filter`), astropy.io.fits, argparse; pytest.

**Reference:** `docs/superpowers/specs/2026-06-09-alcor-badpix-epoch-mask-design.md`

**Working on `main`** (this repo is immature and deployed in one place; no feature branch).

---

## File Structure

- `skycam_utils/alcor.py` — all new functions/CLI (follow the existing `fit_alcor_wcs` / `*_cli` patterns).
- `skycam_utils/tests/test_alcor_badpix.py` — new test module (mirrors `test_alcor.py` style).
- `skycam_utils/data/badpix/` — committed baseline masks `alcor_badpix_2024-09-04.fits.gz`, `alcor_badpix_2026-05-18.fits.gz`.
- `pyproject.toml` — register `create_badpix_mask` script; extend `package-data` to ship `data/badpix/*`.

---

### Task 1: Detector `build_alcor_badpix_mask` + imports

**Files:**
- Modify: `skycam_utils/alcor.py` (imports near top, lines 1–14; new function after `detect_alcor_stars`, ~line 994)
- Test: `skycam_utils/tests/test_alcor_badpix.py`

- [ ] **Step 1: Write the failing test**

Create `skycam_utils/tests/test_alcor_badpix.py`:

```python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from skycam_utils.alcor import build_alcor_badpix_mask


def test_build_badpix_mask_channel_multiplicity():
    # flat zero background; isolated spikes spaced > kernel apart
    cube = np.zeros((3, 15, 15), dtype=float)
    cube[0, 2, 2] = 1000.0                      # 1-channel spike -> flagged in R only
    cube[0, 7, 7] = 1000.0; cube[1, 7, 7] = 1000.0   # 2-channel -> flagged R and G
    cube[:, 12, 12] = 1000.0                    # 3-channel -> real source, excluded

    mask = build_alcor_badpix_mask(cube, ksize=5, z_thresh=25)

    assert mask.shape == (3, 15, 15)
    assert mask.dtype == bool
    # 1-channel
    assert mask[0, 2, 2] and not mask[1, 2, 2] and not mask[2, 2, 2]
    # 2-channel
    assert mask[0, 7, 7] and mask[1, 7, 7] and not mask[2, 7, 7]
    # 3-channel excluded everywhere
    assert not mask[:, 12, 12].any()
    # nothing else flagged (1-ch contributes 1 + 2-ch contributes 2 = 3)
    assert mask.sum() == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py::test_build_badpix_mask_channel_multiplicity -v`
Expected: FAIL with `ImportError: cannot import name 'build_alcor_badpix_mask'`

- [ ] **Step 3: Add imports**

At the top of `skycam_utils/alcor.py`, ensure these imports are present (add any missing; `re`, `sys`, `datetime`, `Path`, `np`, `fits`, `Time` already exist):

```python
import os
import tempfile
from datetime import date
from importlib.resources import files
from scipy.ndimage import median_filter
```

(`from scipy.ndimage import rotate` and `shift as ndimage_shift` already exist on lines 11/13 — add `median_filter` alongside them.)

- [ ] **Step 4: Implement `build_alcor_badpix_mask`**

Add after `detect_alcor_stars` (around line 994):

```python
def build_alcor_badpix_mask(median_cube, ksize=5, z_thresh=25.0):
    """
    Detect per-channel hot pixels in a night-median stack.

    For each channel a small-kernel median high-pass isolates sharp spikes:
    ``resid = img - median_filter(img, ksize)``; a pixel is hot where its robust
    z-score ``(resid - median) / (1.4826 * MAD)`` exceeds ``z_thresh``. A spike is
    a sensor defect only if it fires in AT MOST TWO channels -- one present in all
    three is a real broadband source and is excluded from every plane.

    Parameters
    ----------
    median_cube : ndarray
        Per-pixel median stack of shape ``(3, ny, nx)`` (see
        :func:`build_alcor_median_stack`).
    ksize : int (default=5)
        Local-background median-filter kernel (pixels).
    z_thresh : float (default=25.0)
        Robust-sigma threshold for a hot pixel.

    Returns
    -------
    mask : ndarray of bool, shape ``(3, ny, nx)``
        True where a pixel is a per-channel bad pixel.
    """
    cube = np.asarray(median_cube, dtype=float)
    if cube.ndim != 3 or cube.shape[0] != 3:
        raise ValueError(f"expected a (3, ny, nx) cube, got {cube.shape}")
    z = np.empty_like(cube)
    for c in range(3):
        resid = cube[c] - median_filter(cube[c], size=ksize)
        med = np.median(resid)
        sigma = 1.4826 * np.median(np.abs(resid - med)) + 1e-9
        z[c] = (resid - med) / sigma
    hot = z > z_thresh
    keep = hot.sum(axis=0) <= 2
    return hot & keep[None, :, :]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py::test_build_badpix_mask_channel_multiplicity -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_badpix.py
git commit -m "add build_alcor_badpix_mask hot-pixel detector"
```

---

### Task 2: `build_alcor_median_stack`

**Files:**
- Modify: `skycam_utils/alcor.py` (new function after `build_alcor_badpix_mask`)
- Test: `skycam_utils/tests/test_alcor_badpix.py`

- [ ] **Step 1: Write the failing test**

Append to `test_alcor_badpix.py`:

```python
def _write_fake_frame(path, cube):
    fits.PrimaryHDU(data=cube.astype(np.int16)).writeto(path, overwrite=True)


def test_build_median_stack_rejects_outliers(tmp_path):
    from skycam_utils.alcor import build_alcor_median_stack
    # 5 frames, shape (3, 4, 4); pixel (0,0,0) has an outlier in one frame
    base = np.full((3, 4, 4), 10, dtype=np.int16)
    files = []
    for i, val in enumerate([10, 10, 10, 10, 1000]):
        frame = base.copy()
        frame[0, 0, 0] = val
        p = tmp_path / f"f{i}.fits"
        _write_fake_frame(p, frame)
        files.append(p)

    median = build_alcor_median_stack(files, scratch_dir=str(tmp_path))

    assert median.shape == (3, 4, 4)
    assert median.dtype == np.float32
    assert median[0, 0, 0] == 10.0          # outlier rejected
    assert median[1, 2, 3] == 10.0          # unchanged elsewhere
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py::test_build_median_stack_rejects_outliers -v`
Expected: FAIL with `ImportError: cannot import name 'build_alcor_median_stack'`

- [ ] **Step 3: Implement `build_alcor_median_stack`**

Add after `build_alcor_badpix_mask`:

```python
def build_alcor_median_stack(dark_files, max_frames=None, scratch_dir=None,
                             tile=50, log=None):
    """
    Per-pixel median over a set of raw alcor frames, trail-free for hot-pixel
    detection.

    RAM-bounded: each frame's ``(3, ny, nx)`` uint16 cube is written to a disk
    memmap (in ``scratch_dir``), then the median is taken in row tiles so peak
    memory stays small even for ~1000 frames. Frames whose shape differs from the
    first are skipped. ``max_frames`` strided-subsamples to cap runtime/scratch.

    Returns the median as ``(3, ny, nx)`` float32.
    """
    files_ = list(dark_files)
    if max_frames is not None and len(files_) > max_frames:
        stride = len(files_) // max_frames
        files_ = files_[::stride][:max_frames]
    if not files_:
        raise ValueError("no frames provided")

    with fits.open(files_[0]) as hdul:
        shp = np.asarray(hdul[0].data).shape       # (3, ny, nx)
    nch, ny, nx = shp

    tmp = tempfile.NamedTemporaryFile(
        prefix="alcor_stack_", suffix=".dat",
        dir=scratch_dir or tempfile.gettempdir(), delete=False)
    tmp.close()
    memmap_path = Path(tmp.name)
    cube_mm = None
    try:
        cube_mm = np.memmap(memmap_path, dtype=np.uint16, mode="w+",
                            shape=(len(files_), nch, ny, nx))
        n = 0
        for f in files_:
            with fits.open(f) as hdul:
                data = np.asarray(hdul[0].data)
            if data.shape != shp:
                if log:
                    log(f"skip {Path(f).name}: shape {data.shape}")
                continue
            cube_mm[n] = np.clip(data, 0, 65535).astype(np.uint16)
            n += 1
        cube_mm.flush()
        if n == 0:
            raise ValueError("no frames matched the reference shape")

        median = np.empty((nch, ny, nx), dtype=np.float32)
        for c in range(nch):
            for r0 in range(0, ny, tile):
                r1 = min(r0 + tile, ny)
                slab = np.asarray(cube_mm[:n, c, r0:r1, :], dtype=np.float32)
                median[c, r0:r1, :] = np.median(slab, axis=0)
        return median
    finally:
        del cube_mm
        memmap_path.unlink(missing_ok=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py::test_build_median_stack_rejects_outliers -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_badpix.py
git commit -m "add build_alcor_median_stack (memmap-tiled per-pixel median)"
```

---

### Task 3: Storage + nearest-date resolution

**Files:**
- Modify: `skycam_utils/alcor.py` (new helpers after `build_alcor_median_stack`)
- Test: `skycam_utils/tests/test_alcor_badpix.py`

- [ ] **Step 1: Write the failing test**

Append to `test_alcor_badpix.py`:

```python
def test_load_badpix_mask_nearest_date(tmp_path):
    from skycam_utils.alcor import load_alcor_badpix_mask
    early = (np.zeros((3, 4, 4), dtype=np.uint8))
    late = (np.ones((3, 4, 4), dtype=np.uint8))
    fits.PrimaryHDU(data=early).writeto(tmp_path / "alcor_badpix_2026-05-10.fits.gz")
    fits.PrimaryHDU(data=late).writeto(tmp_path / "alcor_badpix_2026-05-18.fits.gz")

    mask, mdate = load_alcor_badpix_mask(Time("2026-05-17T00:00:00"), masks_dir=str(tmp_path))
    assert mdate == date(2026, 5, 18)
    assert mask.dtype == bool
    assert mask.all()                          # picked the 'late' (all ones) mask


def test_load_badpix_mask_empty_dir(tmp_path):
    from skycam_utils.alcor import load_alcor_badpix_mask
    mask, mdate = load_alcor_badpix_mask(Time("2026-05-17T00:00:00"), masks_dir=str(tmp_path))
    assert mask is None and mdate is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py -k load_badpix_mask -v`
Expected: FAIL with `ImportError: cannot import name 'load_alcor_badpix_mask'`

- [ ] **Step 3: Implement resolution helpers**

Add after `build_alcor_median_stack`:

```python
_BADPIX_DATE_RE = re.compile(r"alcor_badpix_(\d{4})-(\d{2})-(\d{2})\.fits")


def _resolve_badpix_dir(masks_dir=None):
    """Directory holding the date-stamped bad-pixel masks.

    Resolution order: explicit ``masks_dir`` -> ``$ALCOR_BADPIX_DIR`` -> the
    packaged ``skycam_utils/data/badpix/`` (mirroring :func:`load_wcs`).
    """
    if masks_dir is not None:
        return Path(masks_dir)
    env = os.environ.get("ALCOR_BADPIX_DIR")
    if env:
        return Path(env)
    return Path(str(files(__package__) / "data" / "badpix"))


def _badpix_date_from_dir(day_dir, dark_files):
    """Mask date: the ``YYYY-MM-DD`` in the day-directory name, else the median
    dark-frame time's date."""
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(day_dir).name)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    dts = sorted(d for d in (_filename_ut_datetime(f) for f in dark_files)
                 if d is not None)
    return dts[len(dts) // 2].date()


def load_alcor_badpix_mask(time, masks_dir=None):
    """
    Return ``(mask, date)`` for the bad-pixel mask nearest in date to ``time``.

    ``mask`` is a ``(3, ny, nx)`` bool array; ``(None, None)`` if no masks are
    found. ``time`` may be a `~astropy.time.Time`, ``datetime``, or ``date``.
    """
    directory = Path(str(_resolve_badpix_dir(masks_dir)))
    if not directory.is_dir():
        return None, None
    candidates = []
    for p in directory.iterdir():
        m = _BADPIX_DATE_RE.match(p.name)
        if m:
            candidates.append(
                (date(int(m.group(1)), int(m.group(2)), int(m.group(3))), p))
    if not candidates:
        return None, None

    if isinstance(time, Time):
        target = time.to_datetime().date()
    elif isinstance(time, datetime):
        target = time.date()
    else:
        target = time
    best_date, best_path = min(candidates,
                               key=lambda dp: abs((dp[0] - target).days))
    mask = np.asarray(fits.getdata(best_path)).astype(bool)
    return mask, best_date
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py -k load_badpix_mask -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_badpix.py
git commit -m "add bad-pixel mask storage and nearest-date resolution"
```

---

### Task 4: Repair helper `_apply_badpix_repair`

**Files:**
- Modify: `skycam_utils/alcor.py` (new helper after `load_alcor_badpix_mask`)
- Test: `skycam_utils/tests/test_alcor_badpix.py`

- [ ] **Step 1: Write the failing test**

Append to `test_alcor_badpix.py`:

```python
def test_apply_badpix_repair_local_median():
    from skycam_utils.alcor import _apply_badpix_repair
    data = np.full((3, 5, 5), 10, dtype=np.int16)
    data[0, 2, 2] = 1000                       # hot pixel in R
    mask = np.zeros((3, 5, 5), dtype=bool)
    mask[0, 2, 2] = True

    out = _apply_badpix_repair(data, mask, ksize=3)

    assert out[0, 2, 2] == 10                   # replaced with local median
    assert out[0, 1, 1] == 10                   # neighbor untouched
    assert (out[1] == 10).all() and (out[2] == 10).all()   # other channels untouched
    assert data[0, 2, 2] == 1000                # input not mutated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py::test_apply_badpix_repair_local_median -v`
Expected: FAIL with `ImportError: cannot import name '_apply_badpix_repair'`

- [ ] **Step 3: Implement `_apply_badpix_repair`**

Add after `load_alcor_badpix_mask`:

```python
def _apply_badpix_repair(data, mask, ksize=5):
    """
    Replace masked pixels with their local median, per channel.

    ``data`` and ``mask`` are ``(3, ny, nx)``. Returns a repaired copy; the input
    is not mutated. The local median (computed over the surrounding ``ksize``
    window) is robust to the spike itself, so it recovers the underlying sky.
    """
    out = np.array(data, copy=True)
    for c in range(data.shape[0]):
        if not mask[c].any():
            continue
        # filter on a native float copy (FITS data is big-endian int16, which
        # median_filter can choke on), then cast back to the frame's dtype.
        local = median_filter(np.asarray(out[c], dtype=np.float32), size=ksize)
        out[c][mask[c]] = local[mask[c]].astype(out.dtype)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py::test_apply_badpix_repair_local_median -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_badpix.py
git commit -m "add _apply_badpix_repair local-median repair helper"
```

---

### Task 5: `load_alcor_fits` integration (`badpix` + `return_mask`)

**Files:**
- Modify: `skycam_utils/alcor.py` — `load_alcor_fits` signature (line 1050) and body (lines 1105–1138)
- Test: `skycam_utils/tests/test_alcor_badpix.py`

- [ ] **Step 1: Write the failing test**

Append to `test_alcor_badpix.py`:

```python
def _write_alcor_raw(path, cube):
    # alcor raw layout is (3, ny, nx) int16
    fits.PrimaryHDU(data=cube.astype(np.int16)).writeto(path, overwrite=True)


def test_load_alcor_fits_repairs_and_returns_mask(tmp_path):
    from skycam_utils.alcor import load_alcor_fits
    ny = nx = 60
    cube = np.full((3, ny, nx), 2100, dtype=np.int16)   # ~100 above bias
    cube[0, 30, 30] = 30000                              # hot pixel in R at center
    raw = tmp_path / "raw.fits"
    _write_alcor_raw(raw, cube)

    badpix = np.zeros((3, ny, nx), dtype=bool)
    badpix[0, 30, 30] = True

    # explicit mask array -> repair; neutral geometry so no resampling moves pixels
    im, mask, wcs = load_alcor_fits(
        raw, xcen=30, ycen=30, radius=25, horizon_radius=25,
        rotation=0.0, xshift=0.0, yshift=0.0, radial_coeffs=(1.0, 0.0, 0.0),
        badpix=badpix, return_mask=True)

    assert im.shape == (50, 50, 3)
    assert mask.shape == im.shape
    assert im.max() < 1000          # hot pixel repaired (would be ~28000 otherwise)
    assert mask[:, :, 0].sum() == 1 # exactly the one R bad pixel survives transforms
    assert mask[:, :, 1].sum() == 0 and mask[:, :, 2].sum() == 0


def test_load_alcor_fits_default_two_tuple(tmp_path):
    from skycam_utils.alcor import load_alcor_fits
    cube = np.full((3, 60, 60), 2100, dtype=np.int16)
    raw = tmp_path / "raw.fits"
    _write_alcor_raw(raw, cube)
    result = load_alcor_fits(
        raw, xcen=30, ycen=30, radius=25, horizon_radius=25,
        rotation=0.0, xshift=0.0, yshift=0.0, radial_coeffs=(1.0, 0.0, 0.0),
        badpix=None)
    assert len(result) == 2          # (im, wcs) unchanged for default callers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py -k load_alcor_fits -v`
Expected: FAIL with `TypeError: load_alcor_fits() got an unexpected keyword argument 'badpix'`

- [ ] **Step 3: Update the signature**

Change the `load_alcor_fits` signature (line 1050) to add the two parameters:

```python
def load_alcor_fits(filename, rotation=None, xcen=696, ycen=698,
                    radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                    xshift=None, yshift=None,
                    radial_coeffs=None, sip_degree=5,
                    badpix="repair", return_mask=False, masks_dir=None):
```

- [ ] **Step 4: Resolve + repair on the raw frame, return aligned mask**

Replace the body from the `with fits.open(filename)` block through the final `return` (currently lines 1116–1138) with:

```python
    with fits.open(filename) as hdul:
        data = np.asarray(hdul[0].data)        # (3, ny, nx), raw sensor layout

    # --- bad-pixel handling on the raw frame, before any trim/resample ---
    raw_mask = None
    if return_mask or badpix is not None:
        if isinstance(badpix, np.ndarray):
            cand = badpix.astype(bool)
        elif isinstance(badpix, (str, Path)):
            cand = np.asarray(fits.getdata(badpix)).astype(bool)
        else:                                   # "repair" or None -> resolve by time
            cand = None
            try:
                dt = _filename_ut_datetime(filename)
                t = (Time(dt) if dt is not None
                     else Time(_read_frame_date(filename), format="isot", scale="utc"))
                cand, _ = load_alcor_badpix_mask(t, masks_dir=masks_dir)
            except (KeyError, OSError, ValueError):
                cand = None
        if cand is not None and cand.shape == data.shape:
            raw_mask = cand

    if badpix is not None and raw_mask is not None:
        data = _apply_badpix_repair(data, raw_mask)

    im = np.transpose(data, axes=(1, 2, 0)) - 2000  # 2000 is a bit above the normal bias level of the camera.
    im[im < 0] = 0
    im = im * 1.0
    xl = xcen - radius
    xu = xcen + radius
    yl = ycen - radius
    yu = ycen + radius
    im = im[yl:yu, xl:xu, :]
    im = np.flipud(rotate(im, rotation, reshape=False))
    if xshift != 0.0 or yshift != 0.0:
        # Recenter the zenith onto the array center (rows=y, cols=x, channels untouched).
        im = ndimage_shift(im, shift=(-yshift, -xshift, 0.0), order=1, mode="constant", cval=0.0)

    wcs = build_alcor_wcs(
        radius=radius,
        horizon_radius=horizon_radius,
        radial_coeffs=tuple(float(c) for c in radial_coeffs),
        sip_degree=sip_degree,
    )

    if return_mask:
        if raw_mask is not None:
            mk = np.transpose(raw_mask, axes=(1, 2, 0)).astype(np.uint8)
            mk = mk[yl:yu, xl:xu, :]
            mk = np.flipud(rotate(mk, rotation, reshape=False, order=0))
            if xshift != 0.0 or yshift != 0.0:
                mk = ndimage_shift(mk, shift=(-yshift, -xshift, 0.0), order=0,
                                   mode="constant", cval=0)
            mask_out = mk.astype(bool)
        else:
            mask_out = np.zeros(im.shape, dtype=bool)
        return im, mask_out, wcs

    return im, wcs
```

Also add the two new parameters to the docstring Parameters block (after `sip_degree`):

```
    badpix : str or None or path or ndarray (default="repair")
        Bad-pixel handling on the raw frame, before trim/resample. "repair"
        resolves the nearest-date epoch mask and replaces flagged pixels per
        channel with their local 5x5 median; None disables repair; a path or
        (3, ny, nx) bool array uses that mask explicitly (and repairs).
    return_mask : bool (default=False)
        When True, also return the bad-pixel mask aligned to the returned image
        frame, i.e. (im, mask, wcs). Pair with badpix=None to get untouched
        pixels plus the mask (e.g. to OR with a horizon mask for photometry).
    masks_dir : str or None (default=None)
        Override the bad-pixel masks directory (else $ALCOR_BADPIX_DIR, else the
        packaged data/badpix/).
```

- [ ] **Step 5: Run the new tests and the existing alcor suite**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py -k load_alcor_fits -v`
Expected: PASS (2 passed)

Run: `pytest skycam_utils/tests/test_alcor.py skycam_utils/tests/test_alcor_wcs.py -q`
Expected: PASS — existing callers default to `badpix="repair"`, which is a no-op when no mask resolves (the fixtures' frame times don't match a packaged mask, or resolution fails gracefully), so behavior is unchanged. If any existing test asserts a 2-tuple return it still holds (default `return_mask=False`).

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_badpix.py
git commit -m "apply bad-pixel repair in load_alcor_fits; add return_mask option"
```

---

### Task 6: `create_badpix_mask` function + CLI + script registration

**Files:**
- Modify: `skycam_utils/alcor.py` (new `create_badpix_mask` and `create_badpix_mask_cli`, near the other `*_cli` functions, after `fit_alcor_wcs_cli` ~line 1862)
- Modify: `pyproject.toml` (line 52 area — `[project.scripts]`)
- Test: `skycam_utils/tests/test_alcor_badpix.py`

- [ ] **Step 1: Write the failing test**

Append to `test_alcor_badpix.py`:

```python
def test_create_badpix_mask_min_frames_gate(tmp_path, monkeypatch):
    from skycam_utils import alcor
    day = tmp_path / "2026-05-18"
    day.mkdir()
    # make select_dark_frames return a tiny set regardless of contents
    monkeypatch.setattr(alcor, "select_dark_frames",
                        lambda files, **kw: list(files)[:3])
    for i in range(3):
        (day / f"2026_05_18__0{i}_00_00.fits.bz2").write_bytes(b"x")

    out = alcor.create_badpix_mask(day, out_dir=str(tmp_path), min_frames=500)
    assert out is None                          # below the gate: nothing written
    assert not list(tmp_path.glob("alcor_badpix_*.fits.gz"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py::test_create_badpix_mask_min_frames_gate -v`
Expected: FAIL with `AttributeError: module 'skycam_utils.alcor' has no attribute 'create_badpix_mask'`

- [ ] **Step 3: Implement `create_badpix_mask` and its CLI**

Add after `fit_alcor_wcs_cli`:

```python
def create_badpix_mask(day_dir, out_dir=None, min_frames=500, z_thresh=25.0,
                        ksize=5, sun_alt_max=-18.0, moon_alt_max=-6.0,
                        max_frames=None, scratch_dir=None, pattern="*.fits.bz2",
                        log=None):
    """
    Build and write a date-stamped per-channel bad-pixel mask for one night.

    Selects dark frames (Sun < ``sun_alt_max``, Moon < ``moon_alt_max``), and if
    at least ``min_frames`` are available builds the night-median stack, detects
    hot pixels, and writes a gzipped ``alcor_badpix_YYYY-MM-DD.fits.gz`` to
    ``out_dir`` (default: the resolved bad-pixel masks directory). Returns the
    output `~pathlib.Path`, or ``None`` if there were too few dark frames.
    """
    day_dir = Path(day_dir)
    frames = sorted(day_dir.glob(pattern))
    dark = select_dark_frames(frames, sun_alt_max=sun_alt_max,
                              moon_alt_max=moon_alt_max, log=None)
    if log:
        log(f"{len(dark)} dark frames of {len(frames)}")
    if len(dark) < min_frames:
        if log:
            log(f"only {len(dark)} dark frames (< {min_frames}); no mask written")
        return None

    median = build_alcor_median_stack(dark, max_frames=max_frames,
                                      scratch_dir=scratch_dir, log=log)
    mask = build_alcor_badpix_mask(median, ksize=ksize, z_thresh=z_thresh)
    mask_date = _badpix_date_from_dir(day_dir, dark)

    out_dir = Path(str(out_dir)) if out_dir is not None else _resolve_badpix_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alcor_badpix_{mask_date.isoformat()}.fits.gz"

    hdu = fits.PrimaryHDU(data=mask.astype(np.uint8))
    hdu.header["NSTACK"] = (len(dark), "dark frames used")
    hdu.header["ZTHRESH"] = (z_thresh, "robust-sigma threshold")
    hdu.header["KSIZE"] = (ksize, "high-pass kernel (px)")
    hdu.header["CHRULE"] = ("1-2 of 3", "channels flagged for a bad pixel")
    for c, name in enumerate("RGB"):
        hdu.header[f"NBAD{name}"] = (int(mask[c].sum()), f"{name} bad pixels")
    hdu.writeto(out_path, overwrite=True)
    if log:
        log(f"wrote {out_path}")
    return out_path


def create_badpix_mask_cli():
    """CLI entry point for :func:`create_badpix_mask` (run daily from cron)."""
    parser = argparse.ArgumentParser(
        description="Build a date-stamped alcor bad-pixel mask from a night of frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("day_dir", help="Directory of one night's alcor frames.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: $ALCOR_BADPIX_DIR or packaged data/badpix).")
    parser.add_argument("--min-frames", type=int, default=500,
                        help="Minimum dark frames required to generate a mask.")
    parser.add_argument("--z-thresh", type=float, default=25.0,
                        help="Robust-sigma threshold for a hot pixel.")
    parser.add_argument("--ksize", type=int, default=5, help="High-pass median kernel (px).")
    parser.add_argument("--sun-alt-max", type=float, default=-18.0,
                        help="Use frames with Sun altitude below this (deg).")
    parser.add_argument("--moon-alt-max", type=float, default=-6.0,
                        help="Use frames with Moon altitude below this (deg); pass 90 to disable.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap frames used (strided) to bound runtime/scratch.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Directory for the temporary memmap (default: system temp).")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob for input frames.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print per-step progress messages.")
    args = parser.parse_args()

    log = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    out = create_badpix_mask(
        args.day_dir, out_dir=args.out_dir, min_frames=args.min_frames,
        z_thresh=args.z_thresh, ksize=args.ksize, sun_alt_max=args.sun_alt_max,
        moon_alt_max=args.moon_alt_max, max_frames=args.max_frames,
        scratch_dir=args.scratch_dir, pattern=args.pattern, log=log)
    if out is None:
        print("# no mask written (insufficient dark frames)")
    else:
        print(out)
```

- [ ] **Step 4: Register the CLI in `pyproject.toml`**

In `[project.scripts]` (after the `fit_alcor_wcs` line, ~line 52) add:

```toml
create_badpix_mask = "skycam_utils.alcor:create_badpix_mask_cli"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py::test_create_badpix_mask_min_frames_gate -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py pyproject.toml skycam_utils/tests/test_alcor_badpix.py
git commit -m "add create_badpix_mask CLI for daily epoch-mask generation"
```

---

### Task 7: Package the baseline masks (2024-09-04, 2026-05-18) and ship them

**Files:**
- Create: `skycam_utils/data/badpix/alcor_badpix_2024-09-04.fits.gz`, `skycam_utils/data/badpix/alcor_badpix_2026-05-18.fits.gz`
- Modify: `pyproject.toml` (`[tool.setuptools.package-data]`, line 57–58)

- [ ] **Step 1: Extend `package-data` to ship the badpix subdir**

In `pyproject.toml` change:

```toml
[tool.setuptools.package-data]
skycam_utils = ["data/*"]
```
to:
```toml
[tool.setuptools.package-data]
skycam_utils = ["data/*", "data/badpix/*"]
```

- [ ] **Step 2: Generate the two baseline masks**

Canonical, reproducible (re-stacks the raw nights — takes a few minutes each):

```bash
mkdir -p skycam_utils/data/badpix
create_badpix_mask "$HOME/MMT/skycam_data/2024-09-04" --out-dir skycam_utils/data/badpix
create_badpix_mask /Volumes/Samsung_4TB/skycam/2026-05-18 --out-dir skycam_utils/data/badpix
```

Faster shortcut **if** the validated `*_night_median3.fits` stacks from the design session still exist (identical output):

```bash
python - <<'PY'
import numpy as np
from astropy.io import fits
from skycam_utils.alcor import build_alcor_badpix_mask
srcs = {
    "2024-09-04": "/Users/tim/MMT/skycam_data/2024-09-04/2024-09-04_night_median3.fits",
    "2026-05-18": "/Volumes/Samsung_4TB/skycam/2026-05-18/2026-05-18_night_median3.fits",
}
for d, p in srcs.items():
    cube = np.asarray(fits.getdata(p), dtype=float)
    mask = build_alcor_badpix_mask(cube, ksize=5, z_thresh=25).astype(np.uint8)
    out = f"skycam_utils/data/badpix/alcor_badpix_{d}.fits.gz"
    h = fits.PrimaryHDU(data=mask)
    h.header["ZTHRESH"] = 25; h.header["KSIZE"] = 5; h.header["CHRULE"] = "1-2 of 3"
    for c, name in enumerate("RGB"):
        h.header[f"NBAD{name}"] = int(mask[c].sum())
    h.writeto(out, overwrite=True)
    print("wrote", out)
PY
```

- [ ] **Step 3: Verify the masks load and resolve**

Run:
```bash
python -c "
from astropy.time import Time
from skycam_utils.alcor import load_alcor_badpix_mask
import skycam_utils
m24, d24 = load_alcor_badpix_mask(Time('2024-09-05T08:00:00'))
m26, d26 = load_alcor_badpix_mask(Time('2026-05-19T08:00:00'))
print(d24, m24.shape, m24.sum())
print(d26, m26.shape, m26.sum())
"
```
Expected: `2024-09-04 (3, 1411, 1422) <~5000>` and `2026-05-18 (3, 1411, 1422) <~5463>` (the validated counts).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml skycam_utils/data/badpix/alcor_badpix_2024-09-04.fits.gz skycam_utils/data/badpix/alcor_badpix_2026-05-18.fits.gz
git commit -m "ship baseline alcor bad-pixel masks (2024-09-04, 2026-05-18)"
```

---

### Task 8: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `pytest skycam_utils/tests/ -q`
Expected: PASS — all new `test_alcor_badpix.py` tests plus the pre-existing alcor/WCS suites green.

Note: now that Task 7 has shipped the packaged masks, a pre-existing test that loads the bundled alcor fixture (`tests/` contains 2024-09-04 artifacts) may resolve its epoch to a packaged mask and get hot-pixel-repaired by the new `badpix="repair"` default. If such a test fails on *exact pixel values*, that's expected — the remedy is to either assert on structure (shape/WCS) or pass `badpix=None` in that specific test to preserve its original intent. Do NOT change the `load_alcor_fits` default.

- [ ] **Step 2: Confirm the CLI is installed**

Run: `create_badpix_mask --help`
Expected: usage text listing `--min-frames`, `--z-thresh`, `--ksize`, `--sun-alt-max`, `--moon-alt-max`, `--max-frames`, `--scratch-dir`, `--out-dir`, `--pattern`, `--quiet`.

(Requires `pip install -e .` to have picked up the new entry point; rerun if needed.)

---

## Deferred (NOT in this plan)

- Wiring `create_badpix_mask` into `scripts/make_movies.sh` (the daily cron) — done post-implementation per the spec; it's operational, not packaged.
- `load_alcor_fits` resampling refactor (rotate/shift → WCS), the structure mask, and the combined photometry mask — separate combined spec.
