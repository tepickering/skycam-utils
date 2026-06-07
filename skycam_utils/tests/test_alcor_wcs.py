# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import tempfile
from pathlib import Path

import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from astropy.time import Time

from skycam_utils.alcor import (
    ALCOR_HORIZON_RADIUS,
    ALCOR_RADIUS,
    ALCOR_RADIAL_COEFFS,
    _predict_pixels,
    _sun_altitude,
    build_alcor_wcs,
    load_alcor_fits,
    select_dark_frames,
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
    # A positive higher-order (quintic) term increases pixel radius at large zenith angle.
    base_x, base_y = _predict_pixels(10.0, 45.0)
    bent_x, bent_y = _predict_pixels(10.0, 45.0, radial_coeffs=(1.0, 0.0, 0.1))
    base_r = np.hypot(base_x - ALCOR_RADIUS, base_y - ALCOR_RADIUS)
    bent_r = np.hypot(bent_x - ALCOR_RADIUS, bent_y - ALCOR_RADIUS)
    assert bent_r > base_r


def test_predict_pixels_default_coeffs_are_idealized():
    assert ALCOR_RADIAL_COEFFS == (1.0, 0.0, 0.0)


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


def test_build_alcor_wcs_idealized_matches_equidistant():
    wcs = build_alcor_wcs(radius=680, horizon_radius=662, radial_coeffs=(1.0, 0.0, 0.0))
    assert list(wcs.wcs.ctype) == ["RA---ARC", "DEC--ARC"]
    np.testing.assert_allclose(wcs.wcs.crpix, [680.5, 680.5])
    np.testing.assert_allclose(wcs.wcs.cdelt, [90.0 / 662.0, 90.0 / 662.0])
    assert wcs.sip is None  # no SIP attached when the model is purely linear

    az = np.array([0.0, 90.0, 180.0, 270.0])
    px, py = wcs.world_to_pixel_values(az, np.zeros_like(az))
    radii = np.hypot(px - 679.5, py - 679.5)
    np.testing.assert_allclose(radii, 662.0, atol=1e-6)


def test_build_alcor_wcs_with_radial_term_reproduces_forward_model():
    coeffs = (1.0, 0.02, 0.05)
    wcs = build_alcor_wcs(radius=680, horizon_radius=662, radial_coeffs=coeffs)
    assert wcs.sip is not None

    alt = np.array([80.0, 60.0, 40.0, 20.0, 5.0])
    az = np.array([10.0, 100.0, 190.0, 280.0, 350.0])
    model_x, model_y = _predict_pixels(alt, az, radial_coeffs=coeffs)
    wcs_x, wcs_y = wcs.world_to_pixel_values(az, alt)
    # The odd-power radial model (k3*zeta**3 + k5*zeta**5) defines the
    # distortion as a polynomial in the *undistorted* pixel radius, whereas SIP
    # applies its forward polynomial to the *observed* (distorted) pixels; the
    # inverse of an odd polynomial is not itself a polynomial, so a finite SIP
    # fit cannot reproduce it exactly. With degree-5 SIP the residual is
    # dominated by the near-horizon point (~0.30 px at alt=5 deg); it falls well
    # below 0.1 px above alt~20 deg. So we require reproduction to better than
    # 0.31 px across the FOV.
    np.testing.assert_allclose(wcs_x, model_x, atol=0.31)
    np.testing.assert_allclose(wcs_y, model_y, atol=0.31)


def test_load_alcor_fits_idealized_defaults_unchanged():
    test_fits = Path(__file__).with_name("test.fits.bz2")
    im, wcs = load_alcor_fits(test_fits)
    assert im.shape == (1360, 1360, 3)
    np.testing.assert_allclose(wcs.wcs.crpix, [680.5, 680.5])
    np.testing.assert_allclose(wcs.wcs.cdelt, [90.0 / 662.0, 90.0 / 662.0])
