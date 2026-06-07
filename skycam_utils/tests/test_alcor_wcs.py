# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import tempfile
from pathlib import Path

import numpy as np
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
    _fit_params,
    _predict_pixels,
    _sun_altitude,
    alcor_reference_altaz,
    build_alcor_wcs,
    detect_alcor_stars,
    fit_alcor_wcs,
    load_alcor_fits,
    match_alcor_stars,
    save_alcor_residual_plot,
    select_dark_frames,
)


def test_predict_pixels_idealized_reproduces_zenith_and_horizon():
    # Pass the idealized geometry explicitly: the module defaults are now the
    # fitted (non-zero) calibration constants, so we override them here to test
    # the equidistant mapping itself.
    ideal = dict(xshift=0.0, yshift=0.0, rotation=0.0, radial_coeffs=(1.0, 0.0, 0.0))
    cen = ALCOR_RADIUS - 0.5
    x, y = _predict_pixels(90.0, 0.0, **ideal)
    np.testing.assert_allclose([x, y], [cen, cen], atol=1e-9)

    az = np.array([0.0, 90.0, 180.0, 270.0])
    x, y = _predict_pixels(np.zeros_like(az), az, **ideal)
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


def test_predict_pixels_default_coeffs_are_baked_calibration():
    # The module defaults carry the full-night fitted calibration (k1=1 by
    # construction, fitted odd-power k3, k5 unused).
    assert ALCOR_RADIAL_COEFFS == (1.0, 0.01383, 0.0)


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
        hdu.header["DATE"] = date_obs
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
    # build_alcor_wcs encodes only the radial plate solution about the array
    # center; compare against the forward model with no shift/rotation so the
    # geometries match (the module defaults now carry non-zero shifts).
    model_x, model_y = _predict_pixels(
        alt, az, xshift=0.0, yshift=0.0, rotation=0.0, radial_coeffs=coeffs
    )
    wcs_x, wcs_y = wcs.world_to_pixel_values(az, alt)
    # The plate-solution radial model maps the detector to the sky as an
    # odd-power polynomial in the detector pixel radius (k1*rho + k3*rho**3 +
    # k5*rho**5). Its Cartesian displacement is an exact degree-5 polynomial, so
    # the analytic SIP reproduces the plate solution to numerical precision in
    # both directions (world->pixel via astropy's iterative refinement of A/B).
    np.testing.assert_allclose(wcs_x, model_x, atol=1e-4)
    np.testing.assert_allclose(wcs_y, model_y, atol=1e-4)


def test_load_alcor_fits_idealized_defaults_unchanged():
    test_fits = Path(__file__).with_name("test.fits.bz2")
    im, wcs = load_alcor_fits(test_fits)
    assert im.shape == (1360, 1360, 3)
    np.testing.assert_allclose(wcs.wcs.crpix, [680.5, 680.5])
    np.testing.assert_allclose(wcs.wcs.cdelt, [90.0 / 662.0, 90.0 / 662.0])


def test_detect_alcor_stars_on_synthetic_image():
    rng = np.random.default_rng(0)
    im = np.zeros((200, 200, 3), dtype=float)
    im += rng.normal(0.0, 1.0, im.shape)
    truth = [(50.0, 60.0), (120.0, 140.0), (160.0, 30.0)]
    yy, xx = np.mgrid[0:200, 0:200]
    for cx, cy in truth:
        g = 500.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0**2))
        im += g[:, :, None]

    sources = detect_alcor_stars(im, fwhm=2.5, threshold_sigma=5.0)
    assert len(sources) >= 3
    assert {"xcentroid", "ycentroid", "flux"}.issubset(sources.colnames)
    for cx, cy in truth:
        d = np.hypot(np.asarray(sources["xcentroid"]) - cx,
                     np.asarray(sources["ycentroid"]) - cy)
        assert d.min() < 2.0


def test_detect_alcor_stars_on_real_fixture():
    test_fits = Path(__file__).with_name("test.fits.bz2")
    im, _ = load_alcor_fits(test_fits)
    sources = detect_alcor_stars(im)
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


def _fake_detections(alt, az, params):
    x, y = _predict_pixels(alt, az, **params)
    return _Table({"xcentroid": np.asarray(x), "ycentroid": np.asarray(y),
                   "flux": np.linspace(1000, 100, len(np.atleast_1d(x)))})


def test_match_alcor_stars_recovers_correspondences():
    alt = np.array([85.0, 70.0, 55.0, 40.0, 25.0, 10.0])
    az = np.array([15.0, 80.0, 150.0, 210.0, 290.0, 340.0])
    cat = _Table({"Alt": alt, "Az": az, "Vmag": np.linspace(0.5, 3.0, len(alt))})

    true_params = dict(xshift=4.0, yshift=-3.0, rotation=0.6, radial_coeffs=(1.0, 0.03, 0.04))
    det = _fake_detections(alt, az, true_params)

    matched = match_alcor_stars(
        cat, det,
        init_params=dict(xshift=0.0, yshift=0.0, rotation=0.0, radial_coeffs=(1.0, 0.0, 0.0)),
        z_steps=(20.0, 45.0, 70.0, 90.0), tolerance=12.0,
    )
    assert len(matched) >= 5
    assert {"Alt", "Az", "xcentroid", "ycentroid"}.issubset(matched.colnames)
    for row in matched:
        ex, ey = _predict_pixels(row["Alt"], row["Az"], **true_params)
        assert np.hypot(row["xcentroid"] - ex, row["ycentroid"] - ey) < 1e-6


def test_fit_params_recovers_known_geometry():
    rng = np.random.default_rng(1)
    alt = rng.uniform(5.0, 88.0, 200)
    az = rng.uniform(0.0, 360.0, 200)
    true = dict(xshift=5.0, yshift=-4.0, rotation=0.7, radial_coeffs=(1.0, 0.03, 0.05))
    x, y = _predict_pixels(alt, az, **true)

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xshift=0.0, yshift=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)))
    assert abs(fit["xshift"] - 5.0) < 1e-3
    assert abs(fit["yshift"] + 4.0) < 1e-3
    assert abs(fit["rotation"] - 0.7) < 1e-3
    np.testing.assert_allclose(fit["radial_coeffs"], (1.0, 0.03, 0.05), atol=1e-4)


def test_fit_alcor_wcs_aggregates_synthetic_frames(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xshift=6.0, yshift=-5.0, rotation=0.8, radial_coeffs=(1.0, 0.04, 0.06))
    rng = np.random.default_rng(2)

    frame_alt = [rng.uniform(10.0, 88.0, 30), rng.uniform(10.0, 88.0, 30)]
    frame_az = [rng.uniform(0.0, 360.0, 30), rng.uniform(0.0, 360.0, 30)]
    files = [tmp_path / "f0.fits", tmp_path / "f1.fits"]
    for f in files:
        f.write_bytes(b"stub")

    def fake_select_dark_frames(fs, **kw):
        return list(files)

    calls = {"i": 0}

    def fake_load_alcor_fits(path, **kw):
        return np.zeros((2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS, 3)), None

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

    result = fit_alcor_wcs(tmp_path, pattern="*.fits")
    assert abs(result["xshift"] - 6.0) < 0.05
    assert abs(result["yshift"] + 5.0) < 0.05
    assert abs(result["rotation"] - 0.8) < 0.05
    np.testing.assert_allclose(result["radial_coeffs"], (1.0, 0.04, 0.06), atol=2e-3)
    assert result["n_matched"] >= 40
    assert result["residual_rms"] < 0.1


def test_save_alcor_residual_plot_writes_output(tmp_path):
    rng = np.random.default_rng(3)
    alt = rng.uniform(5.0, 88.0, 100)
    az = rng.uniform(0.0, 360.0, 100)
    params = dict(xshift=2.0, yshift=-1.0, rotation=0.3, radial_coeffs=(1.0, 0.02, 0.03))
    x, y = _predict_pixels(alt, az, **params)
    x = x + rng.normal(0, 0.2, x.shape)
    y = y + rng.normal(0, 0.2, y.shape)

    out = save_alcor_residual_plot(alt, az, x, y, params, tmp_path / "resid.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_load_alcor_fits_world_pixel_round_trip():
    test_fits = Path(__file__).with_name("test.fits.bz2")
    _, wcs = load_alcor_fits(test_fits)
    az = np.array([10.0, 100.0, 200.0, 300.0])
    alt = np.array([15.0, 35.0, 55.0, 75.0])
    px, py = wcs.world_to_pixel_values(az, alt)
    az2, alt2 = wcs.pixel_to_world_values(px, py)
    np.testing.assert_allclose(az2 % 360.0, az % 360.0, atol=0.02)
    np.testing.assert_allclose(alt2, alt, atol=0.02)
