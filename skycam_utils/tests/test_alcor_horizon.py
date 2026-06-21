# Licensed under a 3-clause BSD style license - see LICENSE.rst
from datetime import date

import numpy as np
from astropy.io import fits
from astropy.time import Time

from skycam_utils.alcor import load_alcor_horizon_mask


def test_load_horizon_mask_nearest_date(tmp_path):
    early = np.zeros((4, 4), dtype=np.uint8)
    late = np.ones((4, 4), dtype=np.uint8)
    fits.PrimaryHDU(data=early).writeto(tmp_path / "alcor_horizon_2026-02-18.fits.gz")
    fits.PrimaryHDU(data=late).writeto(tmp_path / "alcor_horizon_2026-05-18.fits.gz")

    mask, mdate = load_alcor_horizon_mask(Time("2026-05-17T00:00:00"),
                                          masks_dir=str(tmp_path))
    assert mdate == date(2026, 5, 18)
    assert mask.dtype == bool
    assert mask.ndim == 2                       # achromatic single plane
    assert mask.all()                           # picked the 'late' (all ones) mask


def test_load_horizon_mask_empty_dir(tmp_path):
    mask, mdate = load_alcor_horizon_mask(Time("2026-05-17T00:00:00"),
                                          masks_dir=str(tmp_path))
    assert mask is None and mdate is None


def test_load_horizon_mask_tie_breaks_to_earlier(tmp_path):
    # equidistant masks -> earlier date wins, deterministically
    fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.uint8)).writeto(
        tmp_path / "alcor_horizon_2026-05-10.fits.gz")
    fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.uint8)).writeto(
        tmp_path / "alcor_horizon_2026-05-20.fits.gz")
    # a non-mask file must be ignored by the anchored regex
    (tmp_path / "alcor_horizon_2026-05-10.fits.bak").write_bytes(b"junk")

    mask, mdate = load_alcor_horizon_mask(Time("2026-05-15T00:00:00"),
                                          masks_dir=str(tmp_path))
    assert mdate == date(2026, 5, 10)           # equidistant tie -> earlier date
    assert not mask.any()                       # the 'early' all-zeros mask


def test_load_horizon_mask_env_override(tmp_path, monkeypatch):
    fits.PrimaryHDU(data=np.ones((4, 4), dtype=np.uint8)).writeto(
        tmp_path / "alcor_horizon_2026-02-18.fits.gz")
    monkeypatch.setenv("ALCOR_HORIZON_DIR", str(tmp_path))
    mask, mdate = load_alcor_horizon_mask(Time("2026-02-18T00:00:00"))
    assert mdate == date(2026, 2, 18)
    assert mask.shape == (4, 4)


def test_packaged_horizon_mask_loads():
    # the shipped 2026-02-18 asset resolves from the packaged data/horizon dir
    mask, mdate = load_alcor_horizon_mask(Time("2026-02-18T12:00:00"))
    assert mdate == date(2026, 2, 18)
    assert mask.ndim == 2 and mask.dtype == bool
    assert mask.shape == (1411, 1422)           # raw alcor luminance frame
    # a real mask keeps most of the sky but masks a meaningful obstruction+below
    # horizon fraction; both senses must be present
    frac = mask.mean()
    assert 0.2 < frac < 0.8
    assert (~mask).any() and mask.any()


def test_build_luminance_median_collapses_and_rejects(tmp_path):
    from skycam_utils.alcor import build_alcor_luminance_median
    # 5 frames, shape (3, 4, 4); luminance is 30 everywhere (10+10+10).
    # One frame has an R-channel outlier at (0,0) -> luminance 1020 there.
    base = np.full((3, 4, 4), 10, dtype=np.int16)
    files = []
    for i, val in enumerate([10, 10, 10, 10, 1000]):
        frame = base.copy()
        frame[0, 0, 0] = val
        p = tmp_path / f"f{i}.fits"
        fits.PrimaryHDU(data=frame).writeto(p)
        files.append(p)

    med = build_alcor_luminance_median(files, scratch_dir=str(tmp_path))

    assert med.shape == (4, 4)               # channels collapsed
    assert med.dtype == np.float32
    assert med[0, 0] == 30.0                 # per-pixel median rejects the outlier frame
    assert med[2, 3] == 30.0                 # unchanged elsewhere
