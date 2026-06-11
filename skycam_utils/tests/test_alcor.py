# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import shutil
from io import StringIO
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import (
    _timestamp_edges,
    alcor_keogram,
    alcor_proc_fits,
    build_alcor_wcs,
    load_alcor_keogram_fits,
    load_alcor_fits,
    plot_alcor_keogram_fits,
    plot_alcor_fits,
    save_alcor_keogram_fits,
    save_alcor_keogram_plot,
)


TEST_FITS = Path(__file__).with_name("test.fits.bz2")


@pytest.fixture(scope="module")
def alcor_cube_wcs_mask():
    return load_alcor_fits(TEST_FITS)


def test_load_alcor_fits_returns_raw_cube_wcs_mask(alcor_cube_wcs_mask):
    cube, wcs, mask = alcor_cube_wcs_mask
    assert cube.ndim == 3 and cube.shape[0] == 3          # (3, ny, nx), no transpose
    assert cube.dtype == np.float32
    assert wcs.wcs.ctype[0].startswith("RA---ARC")
    assert mask is None or mask.shape == cube.shape       # native-orientation mask


def test_load_alcor_fits_no_bias_subtraction():
    cube, _, _ = load_alcor_fits(TEST_FITS, badpix=None)
    with fits.open(TEST_FITS) as hdul:
        raw = np.asarray(hdul[0].data, dtype=np.float32)
    np.testing.assert_array_equal(cube, raw)              # untouched: no -2000, no clip


def test_load_alcor_fits_accepts_explicit_wcs():
    w = build_alcor_wcs(xcen=10.0, ycen=20.0, rotation=0.0,
                        radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=30.0)
    _, wcs, _ = load_alcor_fits(TEST_FITS, wcs=w)
    assert list(wcs.wcs.crpix) == [11.0, 21.0]


def test_load_alcor_fits_wcs_maps_zenith_and_horizon(alcor_cube_wcs_mask):
    _, wcs, _ = alcor_cube_wcs_mask

    # The zenith pixel (looked up via the WCS -- with a tilted optical axis it
    # is NOT crpix) round-trips to alt=90.
    xz, yz = wcs.world_to_pixel_values(0.0, 90.0)
    _, zenith_alt = wcs.pixel_to_world_values(xz, yz)
    np.testing.assert_allclose(zenith_alt, 90.0, atol=0.02)

    # The SIP-encoded radial model round-trips world -> pixel -> world.
    az = np.array([10.0, 100.0, 200.0, 300.0])
    alt = np.array([15.0, 35.0, 55.0, 75.0])
    px, py = wcs.world_to_pixel_values(az, alt)
    az2, alt2 = wcs.pixel_to_world_values(px, py)
    np.testing.assert_allclose(alt2, alt, atol=0.02)
    np.testing.assert_allclose(az2 % 360.0, az % 360.0, atol=0.02)


def test_alcor_proc_fits_writes_processed_cube_and_header(tmp_path):
    input_file = tmp_path / "sample.fits.bz2"
    shutil.copyfile(TEST_FITS, input_file)

    output_file = alcor_proc_fits(input_file, overwrite=False)

    assert output_file == tmp_path / "sample_proc.fits"
    assert output_file.exists()

    cube, wcs, _ = load_alcor_fits(input_file)
    with fits.open(output_file) as hdul:
        # the raw (3, ny, nx) cube is written untouched (native orientation)
        assert hdul[0].data.shape == cube.shape
        assert hdul[0].data.dtype.kind == "f"
        assert hdul[0].data.dtype.itemsize == np.dtype(np.float32).itemsize
        np.testing.assert_array_equal(hdul[0].data, cube.astype(np.float32))
        assert hdul[0].header["CTYPE1"].startswith("RA---ARC")
        assert hdul[0].header["CTYPE2"].startswith("DEC--ARC")
        # header WCS round-trips to the same reference pixel as the loaded WCS
        np.testing.assert_allclose(hdul[0].header["CRPIX1"], wcs.wcs.crpix[0])
        np.testing.assert_allclose(hdul[0].header["CRPIX2"], wcs.wcs.crpix[1])


def test_alcor_keogram_uses_center_columns_and_date_headers(tmp_path):
    for index in range(3):
        shutil.copyfile(TEST_FITS, tmp_path / f"alcor_{index:03d}.fits.bz2")

    keogram, timestamps, files = alcor_keogram(tmp_path)
    cube, wcs, _ = load_alcor_fits(TEST_FITS)
    zx, _ = wcs.world_to_pixel_values(0.0, 90.0)
    zcol = int(round(float(zx)))                          # 0-based zenith column
    ny = cube.shape[1]

    assert keogram.shape == (ny, 3, 3)
    assert len(timestamps) == 3
    assert all(timestamp == "2024-09-05T06:51:31.224500" for timestamp in timestamps)
    assert [file.name for file in files] == [
        "alcor_000.fits.bz2",
        "alcor_001.fits.bz2",
        "alcor_002.fits.bz2",
    ]
    np.testing.assert_allclose(keogram[:, 0, :], cube[:, :, zcol].T)


def test_alcor_keogram_can_report_progress(tmp_path):
    for index in range(2):
        shutil.copyfile(TEST_FITS, tmp_path / f"alcor_{index:03d}.fits.bz2")

    progress_file = StringIO()
    alcor_keogram(
        tmp_path,
        progress=True,
        progress_file=progress_file,
    )

    progress_output = progress_file.getvalue()
    assert "1/2" in progress_output
    assert "2/2" in progress_output
    assert "100.0%" in progress_output
    assert progress_output.endswith("\n")


def test_alcor_keogram_dispatches_center_columns_to_workers(tmp_path, monkeypatch):
    for index in range(3):
        shutil.copyfile(TEST_FITS, tmp_path / f"alcor_{index:03d}.fits.bz2")

    serial_keogram, serial_timestamps, serial_files = alcor_keogram(
        tmp_path,
        workers=1,
    )

    submitted_tasks = []

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def submit(self, func, task):
            submitted_tasks.append(task)
            return FakeFuture(func(task))

    monkeypatch.setattr("skycam_utils.alcor.ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr("skycam_utils.alcor.as_completed", lambda futures: futures)

    parallel_keogram, parallel_timestamps, parallel_files = alcor_keogram(
        tmp_path,
        workers=2,
    )

    assert len(submitted_tasks) == 3
    np.testing.assert_allclose(parallel_keogram, serial_keogram)
    assert parallel_timestamps == serial_timestamps
    assert parallel_files == serial_files


def test_save_alcor_keogram_plot_writes_output(tmp_path):
    for index in range(2):
        shutil.copyfile(TEST_FITS, tmp_path / f"alcor_{index:03d}.fits.bz2")

    keogram, timestamps, _ = alcor_keogram(tmp_path)
    output_file = save_alcor_keogram_plot(
        keogram,
        timestamps,
        tmp_path / "keogram.png",
        figsize=(3, 2),
    )

    assert output_file == tmp_path / "keogram.png"
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_timestamp_edges_preserve_irregular_cadence():
    edges = _timestamp_edges(np.array([1.0, 1.5, 3.0]))

    np.testing.assert_allclose(edges, [0.75, 1.25, 2.25, 3.75])


def test_save_alcor_keogram_plot_accepts_irregular_timestamps(tmp_path):
    keogram = np.ones((4, 3, 3))
    timestamps = [
        "2024-09-04T19:00:00",
        "2024-09-04T19:01:00",
        "2024-09-04T19:05:00",
    ]

    output_file = save_alcor_keogram_plot(
        keogram,
        timestamps,
        tmp_path / "irregular_keogram.png",
        figsize=(3, 2),
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_save_alcor_keogram_fits_writes_image_and_timestamp_table(tmp_path):
    for index in range(2):
        shutil.copyfile(TEST_FITS, tmp_path / f"alcor_{index:03d}.fits.bz2")

    keogram, timestamps, _ = alcor_keogram(tmp_path)
    output_file = save_alcor_keogram_fits(
        keogram,
        timestamps,
        tmp_path / "keogram.fits",
    )

    ny = keogram.shape[0]
    assert output_file == tmp_path / "keogram.fits"
    assert output_file.exists()
    with fits.open(output_file) as hdul:
        assert hdul[0].data.shape == (3, ny, 2)
        assert hdul[0].data.dtype.kind == "f"
        assert hdul[1].name == "TIMESTAMPS"
        assert list(hdul[1].data["DATE"]) == timestamps


def test_load_alcor_keogram_fits_round_trips_saved_keogram(tmp_path):
    keogram = np.arange(4 * 3 * 3).reshape(4, 3, 3)
    timestamps = [
        "2024-09-04T19:00:00",
        "2024-09-04T19:01:00",
        "2024-09-04T19:05:00",
    ]
    fits_file = save_alcor_keogram_fits(
        keogram,
        timestamps,
        tmp_path / "keogram.fits",
    )

    loaded_keogram, loaded_timestamps = load_alcor_keogram_fits(fits_file)

    np.testing.assert_allclose(loaded_keogram, keogram)
    assert loaded_timestamps == timestamps


def test_plot_alcor_keogram_fits_writes_output(tmp_path):
    keogram = np.ones((4, 3, 3))
    timestamps = [
        "2024-09-04T19:00:00",
        "2024-09-04T19:01:00",
        "2024-09-04T19:05:00",
    ]
    fits_file = save_alcor_keogram_fits(
        keogram,
        timestamps,
        tmp_path / "keogram.fits",
    )
    output_file = plot_alcor_keogram_fits(
        fits_file,
        output_file=tmp_path / "keogram.png",
        figsize=(3, 2),
    )

    assert output_file == tmp_path / "keogram.png"
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_alcor_fits_writes_outputs_and_returns_figure(tmp_path):
    outimage = tmp_path / "alcor.png"
    outfig = tmp_path / "alcor.pdf"

    fig = plot_alcor_fits(
        TEST_FITS,
        outimage=outimage,
        outfig=outfig,
        radius=600,
        figsize=2,
    )

    try:
        assert outimage.exists()
        assert outimage.stat().st_size > 0
        assert outfig.exists()
        assert outfig.stat().st_size > 0
        assert len(fig.axes) == 2
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
