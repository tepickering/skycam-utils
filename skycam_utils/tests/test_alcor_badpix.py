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
    # nothing else flagged
    assert mask.sum() == 3


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


def test_load_badpix_mask_tie_breaks_to_earlier(tmp_path):
    from skycam_utils.alcor import load_alcor_badpix_mask
    # two masks equidistant from the query date -> earlier date wins, deterministically
    fits.PrimaryHDU(data=np.zeros((3, 4, 4), dtype=np.uint8)).writeto(
        tmp_path / "alcor_badpix_2026-05-10.fits.gz")
    fits.PrimaryHDU(data=np.ones((3, 4, 4), dtype=np.uint8)).writeto(
        tmp_path / "alcor_badpix_2026-05-20.fits.gz")
    # also drop a non-mask file that must be ignored by the anchored regex
    (tmp_path / "alcor_badpix_2026-05-10.fits.bak").write_bytes(b"junk")
    mask, mdate = load_alcor_badpix_mask(Time("2026-05-15T00:00:00"), masks_dir=str(tmp_path))
    assert mdate == date(2026, 5, 10)      # equidistant tie -> earlier date
    assert not mask.any()                  # the 'early' all-zeros mask


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
