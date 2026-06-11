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

from astropy.table import Table as _Table
from astropy.time import Time

from skycam_utils.alcor import (
    ALCOR_HORIZON_RADIUS,
    ALCOR_RADIUS,
    ALCOR_RADIAL_COEFFS,
    ALCOR_XCEN,
    ALCOR_YCEN,
    _fit_params,
    _moon_altitude,
    _predict_pixels,
    _sun_altitude,
    alcor_calibration,
    alcor_reference_altaz,
    assign_alcor_matches,
    build_alcor_wcs,
    detect_alcor_stars,
    fit_alcor_wcs,
    load_alcor_fits,
    save_alcor_residual_plot,
    select_dark_frames,
)


def test_predict_pixels_zenith_maps_to_center():
    x, y = _predict_pixels(90.0, 0.0, xcen=696.0, ycen=698.0, rotation=0.0,
                           radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=662.0)
    assert abs(x - 696.0) < 1e-6
    assert abs(y - 698.0) < 1e-6


def test_predict_pixels_north_is_plus_y():
    # az=0 (north), alt below zenith -> +y (larger row); verified against
    # ground-truth Polaris, which sits well above the image center.
    x, y = _predict_pixels(80.0, 0.0, xcen=696.0, ycen=698.0, rotation=0.0,
                           radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=662.0)
    assert abs(x - 696.0) < 1e-6
    assert y > 698.0


def test_predict_pixels_horizon_radius_at_alt_zero():
    # At alt=0 the detector radius equals horizon_radius about the zenith.
    az = np.array([0.0, 90.0, 180.0, 270.0])
    x, y = _predict_pixels(np.zeros_like(az), az, xcen=696.0, ycen=698.0,
                           rotation=0.0, radial_coeffs=(1.0, 0.0, 0.0),
                           horizon_radius=662.0)
    radii = np.hypot(x - 696.0, y - 698.0)
    np.testing.assert_allclose(radii, 662.0, atol=1e-9)


def test_predict_pixels_radial_term_changes_radius():
    # Under the plate-solution parametrization z = 90*(k1*rho + ... + k5*rho**5),
    # a positive rho**5 coefficient makes z grow faster with rho, so a star at a
    # fixed altitude (fixed z) is reached at a SMALLER detector radius: it is
    # pulled inward relative to the equidistant mapping.
    base_x, base_y = _predict_pixels(10.0, 45.0, radial_coeffs=(1.0, 0.0, 0.0))
    bent_x, bent_y = _predict_pixels(10.0, 45.0, radial_coeffs=(1.0, 0.0, 0.1))
    base_r = np.hypot(base_x - ALCOR_XCEN, base_y - ALCOR_YCEN)
    bent_r = np.hypot(bent_x - ALCOR_XCEN, bent_y - ALCOR_YCEN)
    assert bent_r < base_r


def test_predict_pixels_default_coeffs_are_baked_calibration():
    # The module defaults carry the full-night fitted calibration: k1=1 by
    # construction plus the fitted odd-power k3/k5 terms.
    assert ALCOR_RADIAL_COEFFS[0] == 1.0
    assert len(ALCOR_RADIAL_COEFFS) == 3


def test_sun_altitude_night_vs_day():
    # 2024-09-05 07:00 UT at MMT is local ~midnight -> Sun well below horizon.
    night = _sun_altitude(Time("2024-09-05T07:00:00", format="isot", scale="utc"))
    # 2024-09-05 20:00 UT is local ~13:00 -> Sun high.
    day = _sun_altitude(Time("2024-09-05T20:00:00", format="isot", scale="utc"))
    assert night < -18.0
    assert day > 18.0


def test_moon_altitude_up_vs_down():
    # 2026-05-18 04:00 UT at MMT: waning crescent still up (~ -2 deg).
    up = _moon_altitude(Time("2026-05-18T04:00:00", format="isot", scale="utc"))
    # one hour later it has set (~ -12 deg).
    down = _moon_altitude(Time("2026-05-18T05:00:00", format="isot", scale="utc"))
    assert up > -6.0
    assert down < -6.0


def test_select_dark_frames_rejects_moon_up(tmp_path):
    from astropy.io import fits

    def write(name, date_obs):
        hdu = fits.PrimaryHDU(data=np.zeros((3, 4, 4), dtype=np.int16))
        hdu.header["DATE"] = date_obs
        path = tmp_path / name
        hdu.writeto(path)
        return path

    # Both frames have the Sun well below -18 deg (astronomical night). They
    # differ only in the Moon: at 04:00 UT it is still up (~ -2 deg), at 05:00 UT
    # it has set (~ -12 deg), so only the moonless frame should survive.
    moon_up = write("moonup.fits", "2026-05-18T04:00:00")
    moonless = write("moonless.fits", "2026-05-18T05:00:00")

    selected = select_dark_frames([moon_up, moonless], sun_alt_max=-18.0,
                                  moon_alt_max=-6.0)
    assert selected == [moonless]


def test_select_dark_frames_filters_by_sun_altitude(tmp_path):
    from astropy.io import fits

    def write(name, date_obs):
        hdu = fits.PrimaryHDU(data=np.zeros((3, 4, 4), dtype=np.int16))
        hdu.header["DATE"] = date_obs
        path = tmp_path / name
        hdu.writeto(path)
        return path

    # These names don't match the timestamp pattern, so selection falls back to
    # the DATE header (the UT values written above).
    dark = write("dark.fits", "2024-09-05T07:00:00")
    twilight = write("twi.fits", "2024-09-05T02:30:00")  # near sunset, Sun above -18
    day = write("day.fits", "2024-09-05T20:00:00")

    messages = []
    selected = select_dark_frames([day, dark, twilight], sun_alt_max=-18.0,
                                  log=messages.append)
    assert selected == [dark]
    joined = "\n".join(messages)
    assert "selecting dark frames from 3 files" in joined
    assert "1 of 3 frames are dark" in joined


def test_select_dark_frames_parses_time_from_filename(tmp_path):
    from astropy.io import fits

    def write(name):
        # No DATE header: selection must rely on the filename, and would raise
        # KeyError if it fell back to reading the header.
        path = tmp_path / name
        fits.PrimaryHDU(data=np.zeros((3, 4, 4), dtype=np.int16)).writeto(path)
        return path

    # Local MST: midnight -> UT 07:00 (dark); 13:00 -> UT 20:00 (daylight).
    night = write("2024_09_05__00_00_36.fits.bz2")
    day = write("2024_09_04__13_00_00.fits.bz2")

    selected = select_dark_frames([day, night], sun_alt_max=-18.0)
    assert selected == [night]


def test_filename_ut_datetime_applies_mst_offset():
    from skycam_utils.alcor import _filename_ut_datetime

    ut = _filename_ut_datetime("2024_09_05__00_00_36.fits.bz2")
    assert (ut.year, ut.month, ut.day, ut.hour, ut.minute, ut.second) == \
        (2024, 9, 5, 7, 0, 36)
    assert _filename_ut_datetime("master_flat.fits") is None


def test_build_alcor_wcs_idealized_matches_equidistant():
    wcs = build_alcor_wcs(xcen=696.0, ycen=698.0, rotation=0.0,
                          radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=662.0)
    assert list(wcs.wcs.ctype) == ["RA---ARC", "DEC--ARC"]
    np.testing.assert_allclose(wcs.wcs.crpix, [697.0, 699.0])
    np.testing.assert_allclose(wcs.wcs.cdelt, [90.0 / 662.0, 90.0 / 662.0])
    assert wcs.sip is None  # no SIP attached when the model is purely linear

    az = np.array([0.0, 90.0, 180.0, 270.0])
    px, py = wcs.world_to_pixel_values(az, np.zeros_like(az))
    radii = np.hypot(px - 696.0, py - 698.0)
    np.testing.assert_allclose(radii, 662.0, atol=1e-6)


def test_build_alcor_wcs_reproduces_raw_forward_model():
    xcen, ycen, rot, coeffs, hr = 696.0, 698.0, 0.4, (1.0, 0.0138, 0.0), 662.0
    wcs = build_alcor_wcs(xcen=xcen, ycen=ycen, rotation=rot,
                          radial_coeffs=coeffs, horizon_radius=hr)
    az = np.array([0.0, 90.0, 180.0, 270.0, 45.0])
    alt = np.array([85.0, 60.0, 30.0, 10.0, 0.0])
    mx, my = _predict_pixels(alt, az, xcen=xcen, ycen=ycen, rotation=rot,
                             radial_coeffs=coeffs, horizon_radius=hr)
    # world_to_pixel_values runs astropy's iterative SIP inversion (refining the
    # approximate AP/BP with the exact A/B); wcs_world2pix would only apply the
    # approximate inverse and drift radially.
    wx, wy = wcs.world_to_pixel_values(az, alt)
    np.testing.assert_allclose(wx, mx, atol=1e-3)
    np.testing.assert_allclose(wy, my, atol=1e-3)


def test_build_alcor_wcs_with_radial_term_reproduces_forward_model():
    coeffs = (1.0, 0.02, 0.05)
    xcen, ycen, hr = 696.0, 698.0, 662.0
    wcs = build_alcor_wcs(xcen=xcen, ycen=ycen, rotation=0.0,
                          radial_coeffs=coeffs, horizon_radius=hr)
    assert wcs.sip is not None

    alt = np.array([80.0, 60.0, 40.0, 20.0, 5.0])
    az = np.array([10.0, 100.0, 190.0, 280.0, 350.0])
    assert list(wcs.wcs.ctype) == ["RA---ARC-SIP", "DEC--ARC-SIP"]
    model_x, model_y = _predict_pixels(
        alt, az, xcen=xcen, ycen=ycen, rotation=0.0, radial_coeffs=coeffs,
        horizon_radius=hr,
    )
    wcs_x, wcs_y = wcs.world_to_pixel_values(az, alt)
    # The plate-solution radial model maps the detector to the sky as an
    # odd-power polynomial in the detector pixel radius (k1*rho + k3*rho**3 +
    # k5*rho**5). Its Cartesian displacement is an exact degree-5 polynomial, so
    # the analytic SIP reproduces the plate solution to numerical precision in
    # both directions (world->pixel via astropy's iterative refinement of A/B).
    np.testing.assert_allclose(wcs_x, model_x, atol=1e-4)
    np.testing.assert_allclose(wcs_y, model_y, atol=1e-4)


def test_load_alcor_fits_returns_raw_cube():
    from skycam_utils.alcor import alcor_calibration
    test_fits = Path(__file__).with_name("test.fits.bz2")
    cube, wcs, _ = load_alcor_fits(test_fits)
    assert cube.ndim == 3 and cube.shape[0] == 3
    assert cube.dtype == np.float32
    # cdelt is the zenith plate scale set by the resolved epoch's horizon_radius
    hr = alcor_calibration()["horizon_radius"]
    np.testing.assert_allclose(wcs.wcs.cdelt, [90.0 / hr, 90.0 / hr])


def test_detect_alcor_stars_accepts_cube_and_2d():
    yy, xx = np.mgrid[0:40, 0:40]
    img = 500.0 * np.exp(-((xx - 25) ** 2 + (yy - 20) ** 2) / (2 * 2.0**2))
    cube = np.stack([img, img, img], axis=0)              # (3, ny, nx)
    det_cube = detect_alcor_stars(cube, fwhm=2.0, threshold_sigma=3.0)
    det_2d = detect_alcor_stars(img, fwhm=2.0, threshold_sigma=3.0)
    assert len(det_cube) >= 1 and len(det_2d) >= 1
    assert abs(det_cube["xcentroid"][0] - 25) < 1.5
    assert abs(det_cube["ycentroid"][0] - 20) < 1.5


def test_detect_alcor_stars_on_synthetic_image():
    rng = np.random.default_rng(0)
    im = np.zeros((3, 200, 200), dtype=float)
    im += rng.normal(0.0, 1.0, im.shape)
    truth = [(50.0, 60.0), (120.0, 140.0), (160.0, 30.0)]
    yy, xx = np.mgrid[0:200, 0:200]
    for cx, cy in truth:
        g = 500.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0**2))
        im += g[None, :, :]

    sources = detect_alcor_stars(im, fwhm=2.5, threshold_sigma=5.0)
    assert len(sources) >= 3
    assert {"xcentroid", "ycentroid", "flux"}.issubset(sources.colnames)
    for cx, cy in truth:
        d = np.hypot(np.asarray(sources["xcentroid"]) - cx,
                     np.asarray(sources["ycentroid"]) - cy)
        assert d.min() < 2.0


def test_detect_alcor_stars_caps_to_brightest():
    rng = np.random.default_rng(11)
    im = np.zeros((3, 200, 200), dtype=float)
    im += rng.normal(0.0, 1.0, im.shape)
    yy, xx = np.mgrid[0:200, 0:200]
    # 8 stars of decreasing brightness at distinct positions
    centers = [(20, 20), (60, 40), (100, 30), (140, 60),
               (40, 120), (90, 150), (150, 130), (170, 170)]
    amps = np.linspace(2000.0, 400.0, len(centers))
    for (cx, cy), amp in zip(centers, amps):
        im += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0**2))[None, :, :]

    capped = detect_alcor_stars(im, fwhm=2.5, threshold_sigma=5.0, max_detections=3)
    assert len(capped) == 3
    # The three kept must be the three brightest detected.
    full = detect_alcor_stars(im, fwhm=2.5, threshold_sigma=5.0, max_detections=None)
    top3 = np.sort(np.asarray(full["flux"]))[::-1][:3]
    np.testing.assert_allclose(np.sort(np.asarray(capped["flux"]))[::-1], top3)


def test_detect_alcor_stars_on_real_fixture():
    test_fits = Path(__file__).with_name("test.fits.bz2")
    cube, _, _ = load_alcor_fits(test_fits)
    sources = detect_alcor_stars(cube)
    assert len(sources) > 10


def test_alcor_reference_altaz_filters_and_refracts():
    time = Time("2024-09-05T07:00:00", format="isot", scale="utc")
    cat = alcor_reference_altaz(time, vmag_limit=3.0, min_alt=5.0)

    assert {"Alt", "Az", "Vmag"}.issubset(cat.colnames)
    assert len(cat) > 0
    assert (cat["Vmag"] <= 3.0).all()
    assert (cat["Alt"] >= 5.0).all()

    # Refraction lifts stars: with refraction, altitudes are >= the airless value.
    no_refr = alcor_reference_altaz(time, vmag_limit=3.0, min_alt=5.0, refraction=False)
    common = set(cat["HD"]) & set(no_refr["HD"])
    hd = sorted(common)[0]
    a_refr = float(cat["Alt"][list(cat["HD"]).index(hd)])
    a_none = float(no_refr["Alt"][list(no_refr["HD"]).index(hd)])
    assert a_refr >= a_none - 1e-6


def _clean_frame(true_params, n=12, seed=0):
    """A catalog + detection pair with detections exactly at the true pixels."""
    rng = np.random.default_rng(seed)
    alt = rng.uniform(15.0, 85.0, n)
    az = rng.uniform(0.0, 360.0, n)
    cat = _Table({"Alt": alt, "Az": az, "Vmag": rng.uniform(0.5, 3.5, n)})
    x, y = _predict_pixels(alt, az, **true_params)
    det = _Table({"xcentroid": np.asarray(x), "ycentroid": np.asarray(y),
                  "flux": rng.uniform(200.0, 2000.0, n)})
    return cat, det, np.asarray(x), np.asarray(y)


def test_assign_alcor_matches_recovers_clean_frame():
    true = dict(xcen=4.0, ycen=-3.0, rotation=0.6, radial_coeffs=(1.0, 0.03, 0.0))
    cat, det, tx, ty = _clean_frame(true, n=12, seed=1)
    matched = assign_alcor_matches(cat, det, params=true, tolerance=3.0)
    assert len(matched) == 12
    for row in matched:
        ex, ey = _predict_pixels(row["Alt"], row["Az"], **true)
        assert np.hypot(row["xcentroid"] - ex, row["ycentroid"] - ey) < 1e-6


def test_assign_alcor_matches_pattern_rejects_decoy():
    true = dict(xcen=4.0, ycen=-3.0, rotation=0.6, radial_coeffs=(1.0, 0.03, 0.0))
    cat, det, tx, ty = _clean_frame(true, n=12, seed=2)
    # Plant a decoy detection near catalog star 0's predicted pixel, nearer than
    # its true detection, but displaced so it breaks the local constellation.
    decoy_x = tx[0] + 3.5
    decoy_y = ty[0]
    det_x = np.asarray(det["xcentroid"], dtype=float)
    det_y = np.asarray(det["ycentroid"], dtype=float)
    # remove star 0's true detection so the decoy is the only candidate for it
    keep = np.ones(len(det), dtype=bool)
    keep[0] = False
    det2 = _Table({"xcentroid": np.append(det_x[keep], decoy_x),
                   "ycentroid": np.append(det_y[keep], decoy_y),
                   "flux": np.append(np.asarray(det["flux"])[keep], 1500.0)})
    matched = assign_alcor_matches(cat, det2, params=true, tolerance=4.0)
    # The decoy must not be matched to catalog star 0; the genuine pairs survive.
    for row in matched:
        assert not (abs(row["xcentroid"] - decoy_x) < 1e-6
                    and abs(row["ycentroid"] - decoy_y) < 1e-6)
    assert len(matched) >= 9  # the clean stars still match


def test_fit_params_recovers_k5_when_enabled():
    """With fit_k5=True the quintic radial term is recovered on clean data."""
    rng = np.random.default_rng(9)
    alt = rng.uniform(5.0, 88.0, 400)
    az = rng.uniform(0.0, 360.0, 400)
    true = dict(xcen=5.0, ycen=-4.0, rotation=0.7, radial_coeffs=(1.0, 0.03, 0.02))
    x, y = _predict_pixels(alt, az, **true)

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)),
                      fit_k5=True)
    assert abs(fit["radial_coeffs"][1] - 0.03) < 5e-3
    assert abs(fit["radial_coeffs"][2] - 0.02) < 5e-3
    # default (k3-only) still pins k5 at zero
    k3_only = _fit_params(alt, az, x, y,
                          init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                           radial_coeffs=(1.0, 0.0, 0.0)))
    assert k3_only["radial_coeffs"][2] == 0.0


def test_assign_alcor_matches_brightness_tiebreak():
    # One detection contested by two catalog stars within tolerance.
    # Catalog A is bright (Vmag 1.0) and farther; B is faint (Vmag 4.0) and nearer.
    params = dict(xcen=0.0, ycen=0.0, rotation=0.0, radial_coeffs=(1.0, 0.0, 0.0))
    ax, ay = _predict_pixels(60.0, 10.0, **params)
    bx, by = _predict_pixels(60.5, 10.0, **params)  # close neighbor
    cat = _Table({"Alt": [60.0, 60.5], "Az": [10.0, 10.0], "Vmag": [1.0, 4.0]})
    # detection sits 1px from B's pixel, ~ a few px from A's pixel
    det = _Table({"xcentroid": [float(bx) + 1.0], "ycentroid": [float(by)],
                  "flux": [1000.0]})

    bright = assign_alcor_matches(cat, det, params=params, tolerance=20.0,
                                  brightness=True, min_corroborating=2)
    assert len(bright) == 1
    assert float(bright["Vmag"][0]) == 1.0          # bright catalog star A wins

    nearest = assign_alcor_matches(cat, det, params=params, tolerance=20.0,
                                   brightness=False, min_corroborating=2)
    assert len(nearest) == 1
    assert float(nearest["Vmag"][0]) == 4.0         # nearest catalog star B wins


def test_assign_alcor_matches_empty_inputs():
    params = dict(xcen=0.0, ycen=0.0, rotation=0.0, radial_coeffs=(1.0, 0.0, 0.0))
    cat = _Table({"Alt": [45.0], "Az": [10.0], "Vmag": [2.0]})
    empty_det = _Table({"xcentroid": [], "ycentroid": [], "flux": []})
    out = assign_alcor_matches(cat, empty_det, params=params, tolerance=3.0)
    assert len(out) == 0
    assert {"Alt", "Az", "xcentroid", "ycentroid"}.issubset(out.colnames)


def test_fit_params_recovers_known_geometry():
    rng = np.random.default_rng(1)
    alt = rng.uniform(5.0, 88.0, 200)
    az = rng.uniform(0.0, 360.0, 200)
    # The fitter is k3-only (k5 held at 0), so generate the truth with k5=0.
    true = dict(xcen=5.0, ycen=-4.0, rotation=0.7, radial_coeffs=(1.0, 0.03, 0.0))
    x, y = _predict_pixels(alt, az, **true)

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)))
    assert abs(fit["xcen"] - 5.0) < 1e-3
    assert abs(fit["ycen"] + 4.0) < 1e-3
    assert abs(fit["rotation"] - 0.7) < 1e-3
    np.testing.assert_allclose(fit["radial_coeffs"], (1.0, 0.03, 0.0), atol=1e-4)


def test_fit_params_stays_physical_with_mismatches():
    """k3-only + robust loss must not run away on data polluted by mismatches.

    Fitting the collinear k3 and k5 with a plain loss produced unphysical
    cancelling coefficients (e.g. k3=-0.58, k5=3.6) on real pooled data; this
    guards the regression.
    """
    rng = np.random.default_rng(7)
    alt = rng.uniform(5.0, 88.0, 200)
    az = rng.uniform(0.0, 360.0, 200)
    true = dict(xcen=5.0, ycen=-4.0, rotation=0.7, radial_coeffs=(1.0, 0.08, 0.0))
    x, y = _predict_pixels(alt, az, **true)
    # 15% of the points are gross mismatches to other detections.
    bad = rng.choice(len(x), size=30, replace=False)
    x[bad] += 45.0
    y[bad] -= 45.0

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)))
    assert fit["radial_coeffs"][2] == 0.0           # k5 held at zero, no runaway
    assert abs(fit["radial_coeffs"][1] - 0.08) < 0.03
    assert abs(fit["xcen"] - 5.0) < 3.0
    assert abs(fit["ycen"] + 4.0) < 3.0


def test_fit_alcor_wcs_aggregates_synthetic_frames(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xcen=6.0, ycen=-5.0, rotation=0.8, radial_coeffs=(1.0, 0.04, 0.0))
    rng = np.random.default_rng(2)

    frame_alt = [rng.uniform(10.0, 88.0, 30), rng.uniform(10.0, 88.0, 30)]
    frame_az = [rng.uniform(0.0, 360.0, 30), rng.uniform(0.0, 360.0, 30)]
    files = [tmp_path / "2024_09_05__00_00_00.fits",
             tmp_path / "2024_09_05__00_10_00.fits"]
    for f in files:
        f.write_bytes(b"stub")

    def fake_select_dark_frames(fs, **kw):
        return list(files)

    calls = {"i": 0}

    def fake_load_alcor_fits(path, **kw):
        return np.zeros((3, 2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS)), None, None

    def fake_reference_altaz(time, **kw):
        i = calls["i"]
        return Table({"Alt": frame_alt[i], "Az": frame_az[i],
                      "Vmag": rng.uniform(0.5, 3.0, 30), "HD": np.arange(30)})

    def fake_detect(im, **kw):
        i = calls["i"]
        calls["i"] += 1
        x, y = _predict_pixels(frame_alt[i], frame_az[i], **true)
        return Table({"xcentroid": x, "ycentroid": y, "flux": np.linspace(1e3, 1e2, 30)})

    def fake_frame_time(path):
        return Time("2024-09-05T07:00:00", format="isot", scale="utc")

    monkeypatch.setattr(alcor_mod, "select_dark_frames", fake_select_dark_frames)
    monkeypatch.setattr(alcor_mod, "load_alcor_fits", fake_load_alcor_fits)
    monkeypatch.setattr(alcor_mod, "alcor_reference_altaz", fake_reference_altaz)
    monkeypatch.setattr(alcor_mod, "detect_alcor_stars", fake_detect)
    monkeypatch.setattr(alcor_mod, "_frame_time", fake_frame_time)
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xcen": 0.0,
                                           "ycen": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0), "horizon_radius": 747.0})

    result = fit_alcor_wcs(tmp_path, pattern="*.fits")
    assert abs(result["xcen"] - 6.0) < 0.05
    assert abs(result["ycen"] + 5.0) < 0.05
    assert abs(result["rotation"] - 0.8) < 0.05
    np.testing.assert_allclose(result["radial_coeffs"], (1.0, 0.04, 0.0), atol=2e-3)
    assert result["n_matched"] >= 40
    assert result["residual_rms"] < 0.1
    assert result["epoch"] == "2024-09-05"


def test_fit_alcor_wcs_survives_injected_mismatches(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xcen=6.0, ycen=-5.0, rotation=0.8, radial_coeffs=(1.0, 0.04, 0.0))
    rng = np.random.default_rng(5)

    frame_alt = [rng.uniform(15.0, 85.0, 25), rng.uniform(15.0, 85.0, 25)]
    frame_az = [rng.uniform(0.0, 360.0, 25), rng.uniform(0.0, 360.0, 25)]
    files = [tmp_path / "2024_09_05__00_00_00.fits",
             tmp_path / "2024_09_05__00_10_00.fits"]
    for f in files:
        f.write_bytes(b"stub")

    def fake_select_dark_frames(fs, **kw):
        return list(files)

    calls = {"i": 0}

    def fake_load_alcor_fits(path, **kw):
        return np.zeros((3, 2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS)), None, None

    def fake_reference_altaz(time, **kw):
        i = calls["i"]
        return Table({"Alt": frame_alt[i], "Az": frame_az[i],
                      "Vmag": rng.uniform(0.5, 3.5, 25), "HD": np.arange(25)})

    def fake_detect(im, **kw):
        i = calls["i"]
        calls["i"] += 1
        x, y = _predict_pixels(frame_alt[i], frame_az[i], **true)
        x = np.asarray(x)
        y = np.asarray(y)
        # 10 spurious detections scattered across the frame (mismatches)
        sx = rng.uniform(50, 2 * ALCOR_RADIUS - 50, 10)
        sy = rng.uniform(50, 2 * ALCOR_RADIUS - 50, 10)
        # 5 faint near-catalog decoys: offset +5px from real stars so they
        # contest those stars in the loose early rounds (and must be rejected by
        # the asterism/brightness logic) but fall outside the final tight gate.
        dx = x[:5] + 5.0
        dy = y[:5] + 5.0
        xx = np.concatenate([x, sx, dx])
        yy = np.concatenate([y, sy, dy])
        flux = np.concatenate([rng.uniform(500, 2000, 25),
                               rng.uniform(100, 400, 10),
                               rng.uniform(50, 150, 5)])
        return Table({"xcentroid": xx, "ycentroid": yy, "flux": flux})

    def fake_frame_time(path):
        return Time("2024-09-05T07:00:00", format="isot", scale="utc")

    monkeypatch.setattr(alcor_mod, "select_dark_frames", fake_select_dark_frames)
    monkeypatch.setattr(alcor_mod, "load_alcor_fits", fake_load_alcor_fits)
    monkeypatch.setattr(alcor_mod, "alcor_reference_altaz", fake_reference_altaz)
    monkeypatch.setattr(alcor_mod, "detect_alcor_stars", fake_detect)
    monkeypatch.setattr(alcor_mod, "_frame_time", fake_frame_time)
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xcen": 0.0,
                                           "ycen": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0), "horizon_radius": 747.0})

    result = fit_alcor_wcs(tmp_path, pattern="*.fits")
    assert abs(result["xcen"] - 6.0) < 0.1
    assert abs(result["ycen"] + 5.0) < 0.1
    assert abs(result["rotation"] - 0.8) < 0.05
    np.testing.assert_allclose(result["radial_coeffs"], (1.0, 0.04, 0.0), atol=3e-3)
    assert result["residual_rms"] < 0.5
    assert 0.0 < result["matched_fraction"] <= 1.0


def test_fit_alcor_wcs_parallel_matches_serial(monkeypatch, tmp_path):
    """The workers>1 path must recover the same geometry as the serial path.

    Real subprocesses can't see monkeypatched module functions, so the
    ProcessPoolExecutor is replaced with a synchronous stand-in that still
    drives the parallel dispatch/collection code (futures dict + as_completed).
    """
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xcen=6.0, ycen=-5.0, rotation=0.8, radial_coeffs=(1.0, 0.04, 0.0))
    rng = np.random.default_rng(2)

    frame_alt = [rng.uniform(10.0, 88.0, 30), rng.uniform(10.0, 88.0, 30)]
    frame_az = [rng.uniform(0.0, 360.0, 30), rng.uniform(0.0, 360.0, 30)]
    files = [tmp_path / "2024_09_05__00_00_00.fits",
             tmp_path / "2024_09_05__00_10_00.fits"]
    for f in files:
        f.write_bytes(b"stub")

    def fake_select_dark_frames(fs, **kw):
        return list(files)

    def fake_load_alcor_fits(path, **kw):
        return np.zeros((3, 2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS)), None, None

    # _detect_alcor_frame processes one frame's calls atomically, so a shared
    # "active frame index" keyed off the filename keeps reference/detect in sync
    # regardless of which order the frames are dispatched in.
    index_by_name = {f.name: i for i, f in enumerate(files)}
    state = {"i": 0}

    def fake_frame_time(path):
        state["i"] = index_by_name[Path(path).name]
        return Time("2024-09-05T07:00:00", format="isot", scale="utc")

    def fake_reference_altaz(time, **kw):
        i = state["i"]
        return Table({"Alt": frame_alt[i], "Az": frame_az[i],
                      "Vmag": rng.uniform(0.5, 3.0, 30), "HD": np.arange(30)})

    def fake_detect(im, **kw):
        i = state["i"]
        x, y = _predict_pixels(frame_alt[i], frame_az[i], **true)
        return Table({"xcentroid": x, "ycentroid": y, "flux": np.linspace(1e3, 1e2, 30)})

    class _SyncFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _SyncExecutor:
        def __init__(self, max_workers=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, fn, task):
            return _SyncFuture(fn(task))

    monkeypatch.setattr(alcor_mod, "select_dark_frames", fake_select_dark_frames)
    monkeypatch.setattr(alcor_mod, "load_alcor_fits", fake_load_alcor_fits)
    monkeypatch.setattr(alcor_mod, "alcor_reference_altaz", fake_reference_altaz)
    monkeypatch.setattr(alcor_mod, "detect_alcor_stars", fake_detect)
    monkeypatch.setattr(alcor_mod, "_frame_time", fake_frame_time)
    monkeypatch.setattr(alcor_mod, "ProcessPoolExecutor", _SyncExecutor)
    monkeypatch.setattr(alcor_mod, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xcen": 0.0,
                                           "ycen": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0), "horizon_radius": 747.0})

    result = fit_alcor_wcs(tmp_path, pattern="*.fits", workers=2)
    assert abs(result["xcen"] - 6.0) < 0.05
    assert abs(result["ycen"] + 5.0) < 0.05
    assert abs(result["rotation"] - 0.8) < 0.05
    np.testing.assert_allclose(result["radial_coeffs"], (1.0, 0.04, 0.0), atol=2e-3)
    assert result["n_matched"] >= 40
    assert result["epoch"] == "2024-09-05"


def test_fit_alcor_wcs_rejects_invalid_workers(tmp_path):
    (tmp_path / "f0.fits").write_bytes(b"stub")
    with pytest.raises(ValueError, match="workers"):
        fit_alcor_wcs(tmp_path, pattern="*.fits", workers=0)


def test_fit_alcor_wcs_log_reports_dispositions(monkeypatch, tmp_path):
    """The log callback receives one line per file: Sun-rejected, no-stars, used."""
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xcen=2.0, ycen=-1.0, rotation=0.3, radial_coeffs=(1.0, 0.0, 0.0))
    rng = np.random.default_rng(4)

    # f_day is dropped by the Sun filter; f_empty yields no detections; f_good is used.
    files = [tmp_path / "f_day.fits", tmp_path / "f_empty.fits", tmp_path / "f_good.fits"]
    for f in files:
        f.write_bytes(b"stub")

    good_alt = rng.uniform(10.0, 88.0, 30)
    good_az = rng.uniform(0.0, 360.0, 30)

    def fake_select_dark_frames(fs, **kw):
        # Sun too high for f_day; the other two are dark.
        return [tmp_path / "f_empty.fits", tmp_path / "f_good.fits"]

    def fake_load_alcor_fits(path, **kw):
        return np.zeros((3, 2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS)), None, None

    def fake_reference_altaz(time, **kw):
        return Table({"Alt": good_alt, "Az": good_az,
                      "Vmag": rng.uniform(0.5, 3.0, 30), "HD": np.arange(30)})

    def fake_detect(im, **kw):
        if fake_detect.empty:
            return Table({"xcentroid": [], "ycentroid": [], "flux": []})
        x, y = _predict_pixels(good_alt, good_az, **true)
        return Table({"xcentroid": x, "ycentroid": y, "flux": np.linspace(1e3, 1e2, 30)})
    fake_detect.empty = False

    def fake_frame_time(path):
        fake_detect.empty = Path(path).name == "f_empty.fits"
        return Time("2024-09-05T07:00:00", format="isot", scale="utc")

    monkeypatch.setattr(alcor_mod, "select_dark_frames", fake_select_dark_frames)
    monkeypatch.setattr(alcor_mod, "load_alcor_fits", fake_load_alcor_fits)
    monkeypatch.setattr(alcor_mod, "alcor_reference_altaz", fake_reference_altaz)
    monkeypatch.setattr(alcor_mod, "detect_alcor_stars", fake_detect)
    monkeypatch.setattr(alcor_mod, "_frame_time", fake_frame_time)
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xcen": 0.0,
                                           "ycen": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0),
                                           "horizon_radius": 662.0})

    messages = []
    fit_alcor_wcs(tmp_path, pattern="*.fits", log=messages.append)

    joined = "\n".join(messages)
    assert "f_day.fits: skipped (Sun above" in joined
    assert "f_empty.fits: skipped (no stars detected" in joined
    assert "f_good.fits: 30 stars detected" in joined


def test_save_alcor_residual_plot_writes_output(tmp_path):
    rng = np.random.default_rng(3)
    alt = rng.uniform(5.0, 88.0, 100)
    az = rng.uniform(0.0, 360.0, 100)
    params = dict(xcen=2.0, ycen=-1.0, rotation=0.3, radial_coeffs=(1.0, 0.02, 0.03))
    x, y = _predict_pixels(alt, az, **params)
    x = x + rng.normal(0, 0.2, x.shape)
    y = y + rng.normal(0, 0.2, y.shape)

    out = save_alcor_residual_plot(alt, az, x, y, params, tmp_path / "resid.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_load_alcor_fits_world_pixel_round_trip():
    test_fits = Path(__file__).with_name("test.fits.bz2")
    _, wcs, _ = load_alcor_fits(test_fits)
    az = np.array([10.0, 100.0, 200.0, 300.0])
    alt = np.array([15.0, 35.0, 55.0, 75.0])
    px, py = wcs.world_to_pixel_values(az, alt)
    az2, alt2 = wcs.pixel_to_world_values(px, py)
    np.testing.assert_allclose(az2 % 360.0, az % 360.0, atol=0.02)
    np.testing.assert_allclose(alt2, alt, atol=0.02)


def test_alcor_calibration_nearest_in_time(monkeypatch):
    import skycam_utils.alcor as alcor_mod
    table = [
        {"epoch": "2024-09-04", "xcen": -4.5, "ycen": 4.4,
         "rotation": 0.39, "radial_coeffs": (1.0, 0.014, 0.0)},
        {"epoch": "2026-05-19", "xcen": -12.0, "ycen": 9.9,
         "rotation": 0.31, "radial_coeffs": (1.0, 0.084, 0.0)},
    ]
    monkeypatch.setattr(alcor_mod, "ALCOR_CALIBRATIONS", table)

    # well inside the 2024 side
    c = alcor_mod.alcor_calibration(Time("2024-10-01T00:00:00"))
    assert c["epoch"] == "2024-09-04"
    # well inside the 2026 side
    c = alcor_mod.alcor_calibration(Time("2026-04-01T00:00:00"))
    assert c["epoch"] == "2026-05-19"
    # before the first epoch -> first
    c = alcor_mod.alcor_calibration(Time("2020-01-01T00:00:00"))
    assert c["epoch"] == "2024-09-04"
    # after the last epoch -> last
    c = alcor_mod.alcor_calibration(Time("2030-01-01T00:00:00"))
    assert c["epoch"] == "2026-05-19"
    # time=None -> most recent
    assert alcor_mod.alcor_calibration()["epoch"] == "2026-05-19"

    # exact midpoint -> tie breaks to the more recent epoch
    midpoint = Time((Time("2024-09-04").jd + Time("2026-05-19").jd) / 2.0, format="jd")
    assert alcor_mod.alcor_calibration(midpoint)["epoch"] == "2026-05-19"

    # returns a copy, not the stored dict
    c = alcor_mod.alcor_calibration(Time("2024-10-01T00:00:00"))
    c["xcen"] = 999.0
    assert table[0]["xcen"] == -4.5


def test_load_alcor_fits_resolves_and_overrides(monkeypatch):
    import skycam_utils.alcor as alcor_mod
    test_fits = Path(__file__).with_name("test.fits.bz2")

    calls = {"n": 0}
    real = alcor_mod.alcor_calibration

    def spy(time=None):
        calls["n"] += 1
        return real(time)

    monkeypatch.setattr(alcor_mod, "alcor_calibration", spy)

    # wcs=None -> the calibration epoch resolver is consulted to build the WCS
    _, wcs, _ = alcor_mod.load_alcor_fits(test_fits)
    assert calls["n"] == 1
    assert "ARC" in wcs.wcs.ctype[0]

    # explicit wcs -> resolver NOT consulted; the passed WCS is returned as-is
    calls["n"] = 0
    w = alcor_mod.build_alcor_wcs(xcen=0.0, ycen=0.0, rotation=0.0,
                                  radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=30.0)
    _, wcs, _ = alcor_mod.load_alcor_fits(test_fits, wcs=w)
    assert calls["n"] == 0
    assert wcs.sip is None
    assert list(wcs.wcs.crpix) == [1.0, 1.0]


def test_alcor_frame_calibration_uses_filename_then_header(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.io import fits

    table = [
        {"epoch": "2024-09-04", "xcen": -4.5, "ycen": 4.4,
         "rotation": 0.39, "radial_coeffs": (1.0, 0.014, 0.0)},
        {"epoch": "2026-05-19", "xcen": -12.0, "ycen": 9.9,
         "rotation": 0.31, "radial_coeffs": (1.0, 0.084, 0.0)},
    ]
    monkeypatch.setattr(alcor_mod, "ALCOR_CALIBRATIONS", table)

    # filename parses -> no file access needed, resolves by filename time
    assert alcor_mod._alcor_frame_calibration(
        "2026_05_19__03_37_18.fits.bz2")["epoch"] == "2026-05-19"
    assert alcor_mod._alcor_frame_calibration(
        "2024_09_04__22_00_00.fits.bz2")["epoch"] == "2024-09-04"

    # filename does not parse -> fall back to the DATE header
    path = tmp_path / "master.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((3, 4, 4), dtype=np.int16))
    hdu.header["DATE"] = "2026-05-19T10:37:18"
    hdu.writeto(path)
    assert alcor_mod._alcor_frame_calibration(path)["epoch"] == "2026-05-19"


def test_format_calibration_entry_is_parseable():
    import ast
    from skycam_utils.alcor import _format_calibration_entry

    line = _format_calibration_entry(dict(
        epoch="2026-05-19", xcen=-12.0, ycen=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0), horizon_radius=662.0))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["epoch"] == "2026-05-19"
    assert parsed["radial_coeffs"] == (1.0, 0.084, 0.0)
    assert abs(parsed["xcen"] + 12.0) < 1e-9
    assert parsed["horizon_radius"] == 662.0


def test_cli_geometry_flags_removed(monkeypatch):
    """The geometry flags (--rotation/--xcen/--ycen/--horizon-radius) are gone
    from the consumer CLIs now that the WCS is the single source of geometry."""
    import argparse
    import skycam_utils.alcor as alcor_mod

    seen = {}
    orig_parse = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        ns = orig_parse(self, *a, **k)
        seen["ns"] = ns
        raise SystemExit  # stop before the CLI does real work
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)

    for cli, argv in [
        (alcor_mod.plot_alcor_fits_cli, ["prog", "in.fits"]),
        (alcor_mod.alcor_proc_fits_cli, ["prog", "in.fits"]),
        (alcor_mod.alcor_keogram_cli, ["prog", "in_dir"]),
    ]:
        seen.clear()
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit):
            cli()
        for gone in ("rotation", "xcen", "ycen", "horizon_radius"):
            assert not hasattr(seen["ns"], gone), f"{cli.__name__} still has --{gone}"


def test_fit_alcor_wcs_cli_passes_new_flags(monkeypatch, capsys):
    import sys
    import skycam_utils.alcor as alcor_mod

    captured = {}

    def fake_fit(input_dir, **kwargs):
        captured.update(kwargs)
        return {"xcen": 1.0, "ycen": 2.0, "rotation": 0.3,
                "radial_coeffs": (1.0, 0.02, 0.0), "horizon_radius": 662.0,
                "epoch": "2026-05-19",
                "n_matched": 123, "residual_rms": 2.5, "matched_fraction": 0.42,
                "alt": np.array([]), "az": np.array([]),
                "x": np.array([]), "y": np.array([])}

    monkeypatch.setattr(alcor_mod, "fit_alcor_wcs", fake_fit)
    monkeypatch.setattr(sys, "argv",
                        ["fit_alcor_wcs", "/tmp/night", "--max-detections", "150",
                         "--pattern-tol", "5", "--min-corroborating", "1",
                         "--n-neighbors", "8", "--tolerance", "6",
                         "--tolerance-start", "15", "--match-rounds", "3"])
    alcor_mod.fit_alcor_wcs_cli()

    assert captured["vmag_limit"] == 4.0       # new default
    assert captured["tolerance"] == 6.0
    assert captured["max_detections"] == 150
    assert captured["pattern_tol"] == 5.0
    assert captured["min_corroborating"] == 1
    assert captured["n_neighbors"] == 8
    assert captured["tolerance_start"] == 15.0
    assert captured["match_rounds"] == 3
    out = capsys.readouterr().out
    assert "matched fraction" in out.lower()


def test_fit_alcor_wcs_forwards_matcher_knobs(monkeypatch, tmp_path):
    """fit_alcor_wcs must thread the asterism knobs into assign_alcor_matches."""
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table, hstack

    files = [tmp_path / "2024_09_05__00_00_00.fits"]
    files[0].write_bytes(b"stub")
    alt = np.array([60.0, 50.0, 40.0, 30.0])
    az = np.array([10.0, 100.0, 200.0, 300.0])
    x, y = _predict_pixels(alt, az, xcen=0.0, ycen=0.0, rotation=0.0,
                           radial_coeffs=(1.0, 0.0, 0.0))

    monkeypatch.setattr(alcor_mod, "select_dark_frames", lambda fs, **kw: list(files))
    monkeypatch.setattr(alcor_mod, "load_alcor_fits",
                        lambda p, **kw: (np.zeros((3, 2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS)), None, None))
    monkeypatch.setattr(alcor_mod, "alcor_reference_altaz",
                        lambda t, **kw: Table({"Alt": alt, "Az": az,
                                               "Vmag": [1.0, 2.0, 3.0, 4.0],
                                               "HD": np.arange(4)}))
    monkeypatch.setattr(alcor_mod, "detect_alcor_stars",
                        lambda im, **kw: Table({"xcentroid": np.asarray(x),
                                                "ycentroid": np.asarray(y),
                                                "flux": [4.0, 3.0, 2.0, 1.0]}))
    monkeypatch.setattr(alcor_mod, "_frame_time",
                        lambda p: Time("2024-09-05T07:00:00", format="isot", scale="utc"))
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xcen": 0.0,
                                           "ycen": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0), "horizon_radius": 747.0})

    seen = {}

    def fake_assign(cat, det, params, tolerance, **kw):
        seen.update(kw)
        return hstack([Table(cat), Table(det)])

    monkeypatch.setattr(alcor_mod, "assign_alcor_matches", fake_assign)

    fit_alcor_wcs(tmp_path, pattern="*.fits",
                  n_neighbors=8, min_corroborating=1, pattern_tol=4.5)
    assert seen["n_neighbors"] == 8
    assert seen["min_corroborating"] == 1
    assert seen["pattern_tol"] == 4.5


def test_alcor_calibration_defaults_tangential_coeffs():
    # Shipped epochs have no tangential_coeffs key; the resolver must fill it.
    cal = alcor_calibration()
    assert cal["tangential_coeffs"] == (0.0, 0.0)
    cal = alcor_calibration(Time("2024-09-05", scale="utc"))
    assert cal["tangential_coeffs"] == (0.0, 0.0)


def test_format_calibration_entry_includes_tangential():
    import ast
    from skycam_utils.alcor import _format_calibration_entry

    line = _format_calibration_entry(dict(
        epoch="2026-05-19", xcen=-12.0, ycen=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0), tangential_coeffs=(0.004, -0.003),
        horizon_radius=662.0))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["tangential_coeffs"] == (0.004, -0.003)
    # results without the key (old callers) format with the zero default
    line = _format_calibration_entry(dict(
        epoch="2026-05-19", xcen=-12.0, ycen=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0), horizon_radius=662.0))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["tangential_coeffs"] == (0.0, 0.0)


def test_predict_pixels_tangential_zero_is_noop():
    alt = np.array([80.0, 45.0, 10.0])
    az = np.array([15.0, 120.0, 300.0])
    bx, by = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                             radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0)
    tx, ty = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                             radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0,
                             tangential_coeffs=(0.0, 0.0))
    np.testing.assert_array_equal(tx, bx)
    np.testing.assert_array_equal(ty, by)


def test_predict_pixels_tangential_satisfies_plate_solution():
    """The forward model must invert the exact pix->world plate solution:
    (u, v) + D_radial(u, v) + D_tangential(u, v) lands on the linear ARC target,
    whose radius is H * z / (90 * k1) along the (rotation - az) direction."""
    k1, k3, k5 = 1.0, 0.09, 0.02
    H = 747.0
    p1, p2 = 3e-3, -2e-3
    rng = np.random.default_rng(11)
    alt = rng.uniform(2.0, 88.0, 200)
    az = rng.uniform(0.0, 360.0, 200)
    x, y = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                           radial_coeffs=(k1, k3, k5), horizon_radius=H,
                           tangential_coeffs=(p1, p2))
    u = x - 700.0
    v = y - 710.0
    rho2 = (u**2 + v**2) / H**2
    drad = (k3 * rho2 + k5 * rho2**2) / k1
    du = (p1 / H) * (3.0 * u**2 + v**2) + (2.0 * p2 / H) * u * v
    dv = (p2 / H) * (u**2 + 3.0 * v**2) + (2.0 * p1 / H) * u * v
    fu = u * (1.0 + drad) + du
    fv = v * (1.0 + drad) + dv
    s = H * (90.0 - alt) / (90.0 * k1)
    ang = np.radians(-1.0 - az)
    np.testing.assert_allclose(fu, s * np.sin(ang), atol=1e-3)
    np.testing.assert_allclose(fv, s * np.cos(ang), atol=1e-3)


def test_build_alcor_wcs_with_tangential_reproduces_forward_model():
    coeffs = (1.0, 0.02, 0.05)
    tc = (0.004, -0.003)
    xcen, ycen, hr = 696.0, 698.0, 662.0
    wcs = build_alcor_wcs(xcen=xcen, ycen=ycen, rotation=0.4,
                          radial_coeffs=coeffs, horizon_radius=hr,
                          tangential_coeffs=tc)
    assert wcs.sip is not None

    alt = np.array([80.0, 60.0, 40.0, 20.0, 5.0])
    az = np.array([10.0, 100.0, 190.0, 280.0, 350.0])
    mx, my = _predict_pixels(alt, az, xcen=xcen, ycen=ycen, rotation=0.4,
                             radial_coeffs=coeffs, horizon_radius=hr,
                             tangential_coeffs=tc)
    wx, wy = wcs.world_to_pixel_values(az, alt)
    np.testing.assert_allclose(wx, mx, atol=1e-3)
    np.testing.assert_allclose(wy, my, atol=1e-3)
    # pixel->world applies the exact A/B; round-trip back to the inputs
    waz, walt = wcs.pixel_to_world_values(mx, my)
    np.testing.assert_allclose(walt, alt, atol=1e-4)
    daz = (waz - az + 180.0) % 360.0 - 180.0   # wrap-safe angular difference
    np.testing.assert_allclose(daz, 0.0, atol=1e-3)


def test_build_alcor_wcs_tangential_only_attaches_sip():
    # No radial distortion but nonzero tangential terms must still get a SIP.
    wcs = build_alcor_wcs(xcen=696.0, ycen=698.0, rotation=0.0,
                          radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=662.0,
                          tangential_coeffs=(0.003, 0.0))
    assert wcs.sip is not None
    assert list(wcs.wcs.ctype) == ["RA---ARC-SIP", "DEC--ARC-SIP"]


def test_fit_params_recovers_tangential_coeffs():
    rng = np.random.default_rng(13)
    alt = rng.uniform(5.0, 88.0, 400)
    az = rng.uniform(0.0, 360.0, 400)
    true = dict(xcen=5.0, ycen=-4.0, rotation=0.7,
                radial_coeffs=(1.0, 0.08, 0.0),
                tangential_coeffs=(0.004, -0.003))
    x, y = _predict_pixels(alt, az, **true)

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)))
    assert abs(fit["tangential_coeffs"][0] - 0.004) < 2e-4
    assert abs(fit["tangential_coeffs"][1] + 0.003) < 2e-4
    assert abs(fit["xcen"] - 5.0) < 0.05
    assert abs(fit["ycen"] + 4.0) < 0.05
    assert abs(fit["radial_coeffs"][1] - 0.08) < 1e-3
    # k3-only mode still pins k5 at zero with tangential terms free
    assert fit["radial_coeffs"][2] == 0.0

    # fit_k5 branch also carries the tangential terms
    fit5 = _fit_params(alt, az, x, y,
                       init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                        radial_coeffs=(1.0, 0.0, 0.0)),
                       fit_k5=True)
    assert abs(fit5["tangential_coeffs"][0] - 0.004) < 5e-4
    assert abs(fit5["tangential_coeffs"][1] + 0.003) < 5e-4


def test_fit_alcor_wcs_recovers_tangential_terms(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xcen=6.0, ycen=-5.0, rotation=0.8,
                radial_coeffs=(1.0, 0.09, 0.0),
                tangential_coeffs=(0.004, -0.003))
    rng = np.random.default_rng(3)

    frame_alt = [rng.uniform(10.0, 88.0, 30), rng.uniform(10.0, 88.0, 30)]
    frame_az = [rng.uniform(0.0, 360.0, 30), rng.uniform(0.0, 360.0, 30)]
    files = [tmp_path / "2024_09_05__00_00_00.fits",
             tmp_path / "2024_09_05__00_10_00.fits"]
    for f in files:
        f.write_bytes(b"stub")

    calls = {"i": 0}

    def fake_load_alcor_fits(path, **kw):
        return np.zeros((3, 2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS)), None, None

    def fake_reference_altaz(time, **kw):
        i = calls["i"]
        return Table({"Alt": frame_alt[i], "Az": frame_az[i],
                      "Vmag": rng.uniform(0.5, 3.0, 30), "HD": np.arange(30)})

    def fake_detect(im, **kw):
        i = calls["i"]
        calls["i"] += 1
        x, y = _predict_pixels(frame_alt[i], frame_az[i], **true)
        return Table({"xcentroid": x, "ycentroid": y,
                      "flux": np.linspace(1e3, 1e2, 30)})

    monkeypatch.setattr(alcor_mod, "select_dark_frames",
                        lambda fs, **kw: list(files))
    monkeypatch.setattr(alcor_mod, "load_alcor_fits", fake_load_alcor_fits)
    monkeypatch.setattr(alcor_mod, "alcor_reference_altaz", fake_reference_altaz)
    monkeypatch.setattr(alcor_mod, "detect_alcor_stars", fake_detect)
    monkeypatch.setattr(alcor_mod, "_frame_time",
                        lambda path: Time("2024-09-05T07:00:00", format="isot",
                                          scale="utc"))
    # No tangential_coeffs key: the night fit must default it to (0, 0).
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xcen": 0.0,
                                           "ycen": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0),
                                           "horizon_radius": 747.0})

    result = fit_alcor_wcs(tmp_path, pattern="*.fits")
    assert abs(result["tangential_coeffs"][0] - 0.004) < 5e-4
    assert abs(result["tangential_coeffs"][1] + 0.003) < 5e-4
    assert abs(result["xcen"] - 6.0) < 0.1
    assert abs(result["ycen"] + 5.0) < 0.1
    assert result["residual_rms"] < 0.1


def test_alcor_calibration_defaults_axis_tilt():
    # Shipped epochs have no axis_tilt key; the resolver must fill it.
    cal = alcor_calibration()
    assert cal["axis_tilt"] == (0.0, 0.0)
    cal = alcor_calibration(Time("2024-09-05", scale="utc"))
    assert cal["axis_tilt"] == (0.0, 0.0)


def test_format_calibration_entry_includes_axis_tilt():
    import ast
    from skycam_utils.alcor import _format_calibration_entry

    line = _format_calibration_entry(dict(
        epoch="2026-05-19", xcen=-12.0, ycen=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0), tangential_coeffs=(0.004, -0.003),
        axis_tilt=(0.31, -0.22), horizon_radius=662.0))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["axis_tilt"] == (0.31, -0.22)
    # results without the key (old callers) format with the zero default
    line = _format_calibration_entry(dict(
        epoch="2026-05-19", xcen=-12.0, ycen=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0), horizon_radius=662.0))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["axis_tilt"] == (0.0, 0.0)


def test_predict_pixels_axis_tilt_zero_is_noop():
    alt = np.array([80.0, 45.0, 10.0])
    az = np.array([15.0, 120.0, 300.0])
    bx, by = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                             radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0,
                             tangential_coeffs=(0.003, -0.002))
    tx, ty = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                             radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0,
                             tangential_coeffs=(0.003, -0.002),
                             axis_tilt=(0.0, 0.0))
    np.testing.assert_array_equal(tx, bx)
    np.testing.assert_array_equal(ty, by)


def test_predict_pixels_axis_tilt_spherical_geometry():
    """With tilt (and no pixel-space tangential terms), the pixel radius about
    (xcen, ycen) must map through the plate solution to the TRUE angular
    distance from the tilted axis, computed independently from the spherical
    cosine identity."""
    tn, te = 0.3, -0.2
    eps = np.hypot(tn, te)
    a0 = np.degrees(np.arctan2(te, tn))
    k1, k3, k5 = 1.0, 0.09, 0.02
    H = 747.0
    rng = np.random.default_rng(17)
    alt = rng.uniform(5.0, 89.5, 300)
    az = rng.uniform(0.0, 360.0, 300)
    x, y = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                           radial_coeffs=(k1, k3, k5), horizon_radius=H,
                           axis_tilt=(tn, te))
    rho = np.hypot(x - 700.0, y - 710.0) / H
    z_model = 90.0 * (k1 * rho + k3 * rho**3 + k5 * rho**5)
    alt0 = np.radians(90.0 - eps)
    cos_zp = (np.sin(np.radians(alt)) * np.sin(alt0)
              + np.cos(np.radians(alt)) * np.cos(alt0)
              * np.cos(np.radians(az - a0)))
    z_true = np.degrees(np.arccos(np.clip(cos_zp, -1.0, 1.0)))
    np.testing.assert_allclose(z_model, z_true, atol=1e-6)


def test_predict_pixels_axis_tilt_continuous_at_zero():
    alt = np.array([80.0, 45.0, 10.0])
    az = np.array([15.0, 120.0, 300.0])
    kw = dict(xcen=700.0, ycen=710.0, rotation=-1.0,
              radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0)
    x0, y0 = _predict_pixels(alt, az, axis_tilt=(0.0, 0.0), **kw)
    x1, y1 = _predict_pixels(alt, az, axis_tilt=(1e-9, 0.0), **kw)
    np.testing.assert_allclose(x1, x0, atol=1e-5)
    np.testing.assert_allclose(y1, y0, atol=1e-5)


def test_build_alcor_wcs_with_axis_tilt_reproduces_forward_model():
    """All model terms nonzero simultaneously: tilt + k3 + k5 + P1/P2."""
    coeffs = (1.0, 0.05, 0.10)
    tc = (0.002, -0.001)
    at = (0.3, -0.2)
    xcen, ycen, hr = 699.0, 710.0, 747.0
    wcs = build_alcor_wcs(xcen=xcen, ycen=ycen, rotation=-1.0,
                          radial_coeffs=coeffs, horizon_radius=hr,
                          tangential_coeffs=tc, axis_tilt=at)
    eps = np.hypot(*at)
    a0 = np.degrees(np.arctan2(at[1], at[0])) % 360.0
    np.testing.assert_allclose(wcs.wcs.crval, [a0, 90.0 - eps])

    rng = np.random.default_rng(19)
    alt = rng.uniform(5.0, 89.0, 100)
    az = rng.uniform(0.0, 360.0, 100)
    mx, my = _predict_pixels(alt, az, xcen=xcen, ycen=ycen, rotation=-1.0,
                             radial_coeffs=coeffs, horizon_radius=hr,
                             tangential_coeffs=tc, axis_tilt=at)
    wx, wy = wcs.world_to_pixel_values(az, alt)
    np.testing.assert_allclose(wx, mx, atol=1e-3)
    np.testing.assert_allclose(wy, my, atol=1e-3)
    waz, walt = wcs.pixel_to_world_values(mx, my)
    np.testing.assert_allclose(walt, alt, atol=1e-4)
    daz = (waz - az + 180.0) % 360.0 - 180.0   # wrap-safe angular difference
    np.testing.assert_allclose(daz, 0.0, atol=1e-3)


def test_build_alcor_wcs_zero_tilt_keeps_zenith_pole():
    wcs = build_alcor_wcs(xcen=696.0, ycen=698.0, rotation=0.4,
                          radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0,
                          axis_tilt=(0.0, 0.0))
    np.testing.assert_allclose(wcs.wcs.crval, [0.0, 90.0])
    # at zero tilt the zenith pixel IS crpix
    zx, zy = wcs.world_to_pixel_values(0.0, 90.0)
    np.testing.assert_allclose([zx, zy], np.asarray(wcs.wcs.crpix) - 1.0,
                               atol=1e-6)


def test_fit_params_recovers_axis_tilt():
    rng = np.random.default_rng(23)
    alt = rng.uniform(5.0, 88.0, 400)
    az = rng.uniform(0.0, 360.0, 400)
    true = dict(xcen=5.0, ycen=-4.0, rotation=0.7,
                radial_coeffs=(1.0, 0.08, 0.0),
                tangential_coeffs=(0.002, -0.001),
                axis_tilt=(0.3, -0.2))
    x, y = _predict_pixels(alt, az, **true)

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)))
    assert abs(fit["axis_tilt"][0] - 0.3) < 0.01
    assert abs(fit["axis_tilt"][1] + 0.2) < 0.01
    assert abs(fit["xcen"] - 5.0) < 0.1
    assert abs(fit["ycen"] + 4.0) < 0.1
    assert abs(fit["radial_coeffs"][1] - 0.08) < 2e-3
    assert fit["radial_coeffs"][2] == 0.0

    # fit_k5 branch also carries the tilt
    fit5 = _fit_params(alt, az, x, y,
                       init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                        radial_coeffs=(1.0, 0.0, 0.0)),
                       fit_k5=True)
    assert abs(fit5["axis_tilt"][0] - 0.3) < 0.01
    assert abs(fit5["axis_tilt"][1] + 0.2) < 0.01


def test_fit_alcor_wcs_recovers_axis_tilt(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xcen=6.0, ycen=-5.0, rotation=0.8,
                radial_coeffs=(1.0, 0.09, 0.0),
                tangential_coeffs=(0.002, -0.001),
                axis_tilt=(0.3, -0.2))
    rng = np.random.default_rng(5)

    frame_alt = [rng.uniform(10.0, 88.0, 30), rng.uniform(10.0, 88.0, 30)]
    frame_az = [rng.uniform(0.0, 360.0, 30), rng.uniform(0.0, 360.0, 30)]
    files = [tmp_path / "2024_09_05__00_00_00.fits",
             tmp_path / "2024_09_05__00_10_00.fits"]
    for f in files:
        f.write_bytes(b"stub")

    calls = {"i": 0}

    def fake_load_alcor_fits(path, **kw):
        return np.zeros((3, 2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS)), None, None

    def fake_reference_altaz(time, **kw):
        i = calls["i"]
        return Table({"Alt": frame_alt[i], "Az": frame_az[i],
                      "Vmag": rng.uniform(0.5, 3.0, 30), "HD": np.arange(30)})

    def fake_detect(im, **kw):
        i = calls["i"]
        calls["i"] += 1
        x, y = _predict_pixels(frame_alt[i], frame_az[i], **true)
        return Table({"xcentroid": x, "ycentroid": y,
                      "flux": np.linspace(1e3, 1e2, 30)})

    monkeypatch.setattr(alcor_mod, "select_dark_frames",
                        lambda fs, **kw: list(files))
    monkeypatch.setattr(alcor_mod, "load_alcor_fits", fake_load_alcor_fits)
    monkeypatch.setattr(alcor_mod, "alcor_reference_altaz", fake_reference_altaz)
    monkeypatch.setattr(alcor_mod, "detect_alcor_stars", fake_detect)
    monkeypatch.setattr(alcor_mod, "_frame_time",
                        lambda path: Time("2024-09-05T07:00:00", format="isot",
                                          scale="utc"))
    # No axis_tilt key: the night fit must default it to (0, 0).
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xcen": 0.0,
                                           "ycen": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0),
                                           "horizon_radius": 747.0})

    result = fit_alcor_wcs(tmp_path, pattern="*.fits")
    assert abs(result["axis_tilt"][0] - 0.3) < 0.01
    assert abs(result["axis_tilt"][1] + 0.2) < 0.01
    assert abs(result["xcen"] - 6.0) < 0.1
    assert abs(result["ycen"] + 5.0) < 0.1
    assert result["residual_rms"] < 0.1


def test_wcs_zenith_lookup_with_and_without_tilt():
    kw = dict(xcen=699.0, ycen=710.0, rotation=-1.0,
              radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0)
    # zero tilt: zenith pixel == crpix (0-based)
    w0 = build_alcor_wcs(axis_tilt=(0.0, 0.0), **kw)
    zx, zy = w0.world_to_pixel_values(0.0, 90.0)
    np.testing.assert_allclose([zx, zy], [699.0, 710.0], atol=1e-6)
    # tilted: zenith pixel moves off crpix by ~eps * dr/dz ~ 8.3 px/deg
    wt = build_alcor_wcs(axis_tilt=(0.3, -0.2), **kw)
    zx, zy = wt.world_to_pixel_values(0.0, 90.0)
    offset = np.hypot(zx - 699.0, zy - 710.0)
    eps = np.hypot(0.3, -0.2)
    assert 0.5 * eps * 747.0 / 90.0 < offset < 2.0 * eps * 747.0 / 90.0
