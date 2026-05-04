# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import alcor_proc_fits, load_alcor_fits, plot_alcor_fits


TEST_FITS = Path(__file__).with_name("test.fits.bz2")


@pytest.fixture(scope="module")
def alcor_image_and_wcs():
    return load_alcor_fits(TEST_FITS)


def test_load_alcor_fits_returns_centered_rgb_image(alcor_image_and_wcs):
    im, wcs = alcor_image_and_wcs

    assert im.shape == (1360, 1360, 3)
    assert np.issubdtype(im.dtype, np.floating)
    assert np.isfinite(im).all()
    assert im.max() > im.min()

    assert list(wcs.wcs.ctype) == ["RA---ARC", "DEC--ARC"]
    np.testing.assert_allclose(wcs.wcs.crpix, [680.5, 680.5])
    np.testing.assert_allclose(wcs.wcs.crval, [0.0, 90.0])
    np.testing.assert_allclose(wcs.wcs.cdelt, [90.0 / 662.0, 90.0 / 662.0])
    assert wcs.wcs.lonpole == 0.0


def test_load_alcor_fits_wcs_maps_zenith_and_horizon(alcor_image_and_wcs):
    _, wcs = alcor_image_and_wcs

    _, zenith_alt = wcs.pixel_to_world_values(679.5, 679.5)
    np.testing.assert_allclose(zenith_alt, 90.0)

    azimuths = np.array([0.0, 90.0, 180.0, 270.0])
    px, py = wcs.world_to_pixel_values(azimuths, np.zeros_like(azimuths))
    radii = np.hypot(px - 679.5, py - 679.5)

    np.testing.assert_allclose(radii, 662.0, atol=1e-9)
    np.testing.assert_allclose(px, [679.5, 17.5, 679.5, 1341.5], atol=1e-9)
    np.testing.assert_allclose(py, [1341.5, 679.5, 17.5, 679.5], atol=1e-9)


def test_alcor_proc_fits_writes_processed_cube_and_header(tmp_path):
    input_file = tmp_path / "sample.fits.bz2"
    shutil.copyfile(TEST_FITS, input_file)

    output_file = alcor_proc_fits(
        input_file,
        overwrite=False,
        radius=32,
        horizon_radius=30,
    )

    assert output_file == tmp_path / "sample_proc.fits"
    assert output_file.exists()

    with fits.open(output_file) as hdul:
        assert hdul[0].data.shape == (3, 64, 64)
        assert hdul[0].data.dtype.kind == "f"
        assert hdul[0].data.dtype.itemsize == np.dtype(np.float32).itemsize
        assert np.isfinite(hdul[0].data).all()
        assert hdul[0].header["CTYPE1"] == "RA---ARC"
        assert hdul[0].header["CTYPE2"] == "DEC--ARC"
        np.testing.assert_allclose(hdul[0].header["CRPIX1"], 32.5)
        np.testing.assert_allclose(hdul[0].header["CRPIX2"], 32.5)
        np.testing.assert_allclose(hdul[0].header["CDELT1"], 3.0)
        np.testing.assert_allclose(hdul[0].header["CDELT2"], 3.0)


def test_plot_alcor_fits_writes_outputs_and_returns_figure(tmp_path):
    outimage = tmp_path / "alcor.png"
    outfig = tmp_path / "alcor.pdf"

    fig = plot_alcor_fits(
        TEST_FITS,
        outimage=outimage,
        outfig=outfig,
        radius=32,
        horizon_radius=30,
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
