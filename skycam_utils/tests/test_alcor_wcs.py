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
    cen = ALCOR_RADIUS - 0.5
    x, y = _predict_pixels(90.0, 0.0)
    np.testing.assert_allclose([x, y], [cen, cen], atol=1e-9)

    az = np.array([0.0, 90.0, 180.0, 270.0])
    x, y = _predict_pixels(np.zeros_like(az), az)
    radii = np.hypot(x - cen, y - cen)
    np.testing.assert_allclose(radii, ALCOR_HORIZON_RADIUS, atol=1e-9)
    np.testing.assert_allclose(x, cen - ALCOR_HORIZON_RADIUS * np.sin(np.radians(az)), atol=1e-9)
    np.testing.assert_allclose(y, cen + ALCOR_HORIZON_RADIUS * np.cos(np.radians(az)), atol=1e-9)


def test_predict_pixels_radial_term_changes_radius():
    # Under the plate-solution parametrization z = 90*(k1*rho + ... + k5*rho**5),
    # a positive rho**5 coefficient makes z grow faster with rho, so a star at a
    # fixed altitude (fixed z) is reached at a SMALLER detector radius: it is
    # pulled inward relative to the equidistant mapping.
    cen = ALCOR_RADIUS - 0.5
    base_x, base_y = _predict_pixels(10.0, 45.0)
    bent_x, bent_y = _predict_pixels(10.0, 45.0, radial_coeffs=(1.0, 0.0, 0.1))
    base_r = np.hypot(base_x - cen, base_y - cen)
    bent_r = np.hypot(bent_x - cen, bent_y - cen)
    assert bent_r < base_r


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
    assert list(wcs.wcs.ctype) == ["RA---ARC-SIP", "DEC--ARC-SIP"]
    model_x, model_y = _predict_pixels(alt, az, radial_coeffs=coeffs)
    wcs_x, wcs_y = wcs.world_to_pixel_values(az, alt)
    # The plate-solution radial model maps the detector to the sky as an
    # odd-power polynomial in the detector pixel radius (k1*rho + k3*rho**3 +
    # k5*rho**5). Its Cartesian displacement is an exact degree-5 polynomial, so
    # the analytic SIP reproduces the plate solution to numerical precision in
    # both directions (world->pixel via astropy's iterative refinement of A/B).
    np.testing.assert_allclose(wcs_x, model_x, atol=1e-3)
    np.testing.assert_allclose(wcs_y, model_y, atol=1e-3)


def test_load_alcor_fits_idealized_defaults_unchanged():
    test_fits = Path(__file__).with_name("test.fits.bz2")
    im, wcs = load_alcor_fits(test_fits)
    assert im.shape == (1360, 1360, 3)
    np.testing.assert_allclose(wcs.wcs.crpix, [680.5, 680.5])
    np.testing.assert_allclose(wcs.wcs.cdelt, [90.0 / 662.0, 90.0 / 662.0])
