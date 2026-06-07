# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

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
