# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import (
    ALCOR_CALIB_EXPTIME,
    _alcor_pixel_solid_angle,
    _read_frame_exposure,
    plot_alcor_sky_brightness,
)

TEST_FITS = Path(__file__).with_name("test.fits.bz2")


def _flat_altaz_grid(cdelt_deg, lat0_deg=0.0, n=21):
    """(az, alt) grids for a regular ``cdelt_deg`` patch centred at lat0."""
    offset = (np.arange(n) - n // 2) * cdelt_deg
    az = np.broadcast_to(offset[None, :], (n, n))
    alt = np.broadcast_to(lat0_deg + offset[:, None], (n, n))
    return np.array(az), np.array(alt)


def test_pixel_solid_angle_flat_field_matches_plate_scale():
    # At the equator a cdelt-spaced patch subtends (cdelt*3600)^2 arcsec^2/pixel.
    cdelt = 1.0e-3
    az, alt = _flat_altaz_grid(cdelt, lat0_deg=0.0)
    omega = _alcor_pixel_solid_angle(az, alt)
    expected = (cdelt * 3600.0) ** 2
    assert np.isclose(omega[10, 10], expected, rtol=1e-3)


def test_pixel_solid_angle_follows_cosine_latitude_law():
    # The solid angle is a real spherical element, so it scales as cos(lat).
    cdelt = 1.0e-3
    lat0 = 60.0
    az, alt = _flat_altaz_grid(cdelt, lat0_deg=lat0)
    omega = _alcor_pixel_solid_angle(az, alt)
    expected = (cdelt * 3600.0) ** 2 * np.cos(np.radians(lat0))
    assert np.isclose(omega[10, 10], expected, rtol=1e-3)


def test_read_frame_exposure_from_header_and_fallback(tmp_path):
    # the bundled frame carries EXPOSURE = 20 s
    assert _read_frame_exposure(TEST_FITS) == 20.0

    explicit = tmp_path / "exp10.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.int16))
    hdu.header["EXPOSURE"] = 10.0
    hdu.writeto(explicit)
    assert _read_frame_exposure(explicit) == 10.0

    missing = tmp_path / "noexp.fits"
    fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.int16)).writeto(missing)
    assert _read_frame_exposure(missing) == ALCOR_CALIB_EXPTIME

    nonpos = tmp_path / "zeroexp.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.int16))
    hdu.header["EXPOSURE"] = 0.0
    hdu.writeto(nonpos)
    assert _read_frame_exposure(nonpos) == ALCOR_CALIB_EXPTIME


def test_plot_alcor_sky_brightness_writes_figure_and_zenith(tmp_path):
    import matplotlib.pyplot as plt

    outfig = tmp_path / "skybright.pdf"
    fig = plot_alcor_sky_brightness(TEST_FITS, outfig=outfig, radius=600,
                                    figsize=3)
    try:
        assert outfig.exists() and outfig.stat().st_size > 0
        # image axes + colorbar axes + polar overlay
        assert len(fig.axes) == 3
        # the zenith annotation carries a dark-sky surface brightness
        texts = [t.get_text() for t in fig.texts if "zenith" in t.get_text()]
        assert texts, "expected a zenith-brightness annotation"
        match = re.search(r"=\s*([0-9.]+)", texts[0])
        assert match is not None
        zenith = float(match.group(1))
        assert 20.0 < zenith < 23.0          # MMT dark-sky V mag/arcsec^2
    finally:
        plt.close(fig)


def test_plot_alcor_sky_brightness_horizon_mask_runs(tmp_path):
    import matplotlib.pyplot as plt

    outfig = tmp_path / "skybright_hmask.png"
    fig = plot_alcor_sky_brightness(TEST_FITS, outfig=outfig, radius=600,
                                    figsize=3, horizon_mask=True)
    try:
        assert outfig.exists() and outfig.stat().st_size > 0
    finally:
        plt.close(fig)
