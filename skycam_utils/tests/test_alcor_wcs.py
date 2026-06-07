# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import tempfile
from pathlib import Path

import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import (
    ALCOR_HORIZON_RADIUS,
    ALCOR_RADIUS,
    ALCOR_RADIAL_COEFFS,
    _predict_pixels,
)


def test_predict_pixels_idealized_reproduces_zenith_and_horizon():
    # Zenith maps to the array center.
    x, y = _predict_pixels(90.0, 0.0)
    np.testing.assert_allclose([x, y], [ALCOR_RADIUS, ALCOR_RADIUS], atol=1e-9)

    # alt=0 maps to a circle of horizon_radius, with the existing azimuth layout:
    # az=0 -> +y (top), az=90 -> -x (left), az=180 -> -y, az=270 -> +x.
    az = np.array([0.0, 90.0, 180.0, 270.0])
    x, y = _predict_pixels(np.zeros_like(az), az)
    radii = np.hypot(x - ALCOR_RADIUS, y - ALCOR_RADIUS)
    np.testing.assert_allclose(radii, ALCOR_HORIZON_RADIUS, atol=1e-9)
    np.testing.assert_allclose(x, ALCOR_RADIUS - ALCOR_HORIZON_RADIUS * np.sin(np.radians(az)), atol=1e-9)
    np.testing.assert_allclose(y, ALCOR_RADIUS + ALCOR_HORIZON_RADIUS * np.cos(np.radians(az)), atol=1e-9)


def test_predict_pixels_radial_term_pushes_stars_outward():
    # A positive cubic term increases pixel radius at large zenith angle.
    base_x, base_y = _predict_pixels(10.0, 45.0)
    bent_x, bent_y = _predict_pixels(10.0, 45.0, radial_coeffs=(1.0, 0.0, 0.1))
    base_r = np.hypot(base_x - ALCOR_RADIUS, base_y - ALCOR_RADIUS)
    bent_r = np.hypot(bent_x - ALCOR_RADIUS, bent_y - ALCOR_RADIUS)
    assert bent_r > base_r


def test_predict_pixels_default_coeffs_are_idealized():
    assert ALCOR_RADIAL_COEFFS == (1.0, 0.0, 0.0)


from astropy.time import Time

from skycam_utils.alcor import _sun_altitude, select_dark_frames


def test_sun_altitude_night_vs_day():
    # 2024-09-05 07:00 UT at MMT is local ~midnight -> Sun well below horizon.
    night = _sun_altitude(Time("2024-09-05T07:00:00", format="isot", scale="utc"))
    # 2024-09-05 20:00 UT is local ~13:00 -> Sun high.
    day = _sun_altitude(Time("2024-09-05T20:00:00", format="isot", scale="utc"))
    assert night < -18.0
    assert day > 18.0


def test_select_dark_frames_filters_by_sun_altitude(tmp_path):
    from astropy.io import fits

    def write(name, date_obs):
        hdu = fits.PrimaryHDU(data=np.zeros((3, 4, 4), dtype=np.int16))
        hdu.header["DATE-OBS"] = date_obs
        path = tmp_path / name
        hdu.writeto(path)
        return path

    dark = write("dark.fits", "2024-09-05T07:00:00")
    twilight = write("twi.fits", "2024-09-05T02:30:00")  # near sunset, Sun above -18
    day = write("day.fits", "2024-09-05T20:00:00")

    selected = select_dark_frames([day, dark, twilight], sun_alt_max=-18.0)
    assert selected == [dark]
