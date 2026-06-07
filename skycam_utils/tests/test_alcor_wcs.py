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
    _fit_params,
    _predict_pixels,
    _sun_altitude,
    alcor_calibration,
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
    # The fitter is k3-only (k5 held at 0), so generate the truth with k5=0.
    true = dict(xshift=5.0, yshift=-4.0, rotation=0.7, radial_coeffs=(1.0, 0.03, 0.0))
    x, y = _predict_pixels(alt, az, **true)

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xshift=0.0, yshift=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)))
    assert abs(fit["xshift"] - 5.0) < 1e-3
    assert abs(fit["yshift"] + 4.0) < 1e-3
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
    true = dict(xshift=5.0, yshift=-4.0, rotation=0.7, radial_coeffs=(1.0, 0.08, 0.0))
    x, y = _predict_pixels(alt, az, **true)
    # 15% of the points are gross mismatches to other detections.
    bad = rng.choice(len(x), size=30, replace=False)
    x[bad] += 45.0
    y[bad] -= 45.0

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xshift=0.0, yshift=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)))
    assert fit["radial_coeffs"][2] == 0.0           # k5 held at zero, no runaway
    assert abs(fit["radial_coeffs"][1] - 0.08) < 0.03
    assert abs(fit["xshift"] - 5.0) < 3.0
    assert abs(fit["yshift"] + 4.0) < 3.0


def test_fit_alcor_wcs_aggregates_synthetic_frames(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xshift=6.0, yshift=-5.0, rotation=0.8, radial_coeffs=(1.0, 0.04, 0.0))
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
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xshift": 0.0,
                                           "yshift": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0)})

    result = fit_alcor_wcs(tmp_path, pattern="*.fits")
    assert abs(result["xshift"] - 6.0) < 0.05
    assert abs(result["yshift"] + 5.0) < 0.05
    assert abs(result["rotation"] - 0.8) < 0.05
    np.testing.assert_allclose(result["radial_coeffs"], (1.0, 0.04, 0.0), atol=2e-3)
    assert result["n_matched"] >= 40
    assert result["residual_rms"] < 0.1
    assert result["epoch"] == "2024-09-05"


def test_fit_alcor_wcs_parallel_matches_serial(monkeypatch, tmp_path):
    """The workers>1 path must recover the same geometry as the serial path.

    Real subprocesses can't see monkeypatched module functions, so the
    ProcessPoolExecutor is replaced with a synchronous stand-in that still
    drives the parallel dispatch/collection code (futures dict + as_completed).
    """
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xshift=6.0, yshift=-5.0, rotation=0.8, radial_coeffs=(1.0, 0.04, 0.0))
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
        return np.zeros((2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS, 3)), None

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
                        lambda time=None: {"epoch": "2024-09-05", "xshift": 0.0,
                                           "yshift": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0)})

    result = fit_alcor_wcs(tmp_path, pattern="*.fits", workers=2)
    assert abs(result["xshift"] - 6.0) < 0.05
    assert abs(result["yshift"] + 5.0) < 0.05
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

    true = dict(xshift=2.0, yshift=-1.0, rotation=0.3, radial_coeffs=(1.0, 0.0, 0.0))
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
        return np.zeros((2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS, 3)), None

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


def test_alcor_calibration_nearest_in_time(monkeypatch):
    import skycam_utils.alcor as alcor_mod
    table = [
        {"epoch": "2024-09-04", "xshift": -4.5, "yshift": 4.4,
         "rotation": 0.39, "radial_coeffs": (1.0, 0.014, 0.0)},
        {"epoch": "2026-05-19", "xshift": -12.0, "yshift": 9.9,
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
    c["xshift"] = 999.0
    assert table[0]["xshift"] == -4.5


def test_load_alcor_fits_resolves_and_overrides(monkeypatch):
    import skycam_utils.alcor as alcor_mod
    test_fits = Path(__file__).with_name("test.fits.bz2")

    calls = {"n": 0}
    real = alcor_mod._alcor_frame_calibration

    def spy(filename):
        calls["n"] += 1
        return real(filename)

    monkeypatch.setattr(alcor_mod, "_alcor_frame_calibration", spy)

    # defaults -> resolver is consulted
    _, wcs = alcor_mod.load_alcor_fits(test_fits)
    assert calls["n"] == 1
    assert "ARC" in wcs.wcs.ctype[0]

    # all calibration kwargs explicit -> resolver NOT consulted, and an
    # idealized radial term yields a plain ARC WCS (no SIP).
    calls["n"] = 0
    _, wcs = alcor_mod.load_alcor_fits(
        test_fits, rotation=0.0, xshift=0.0, yshift=0.0,
        radial_coeffs=(1.0, 0.0, 0.0))
    assert calls["n"] == 0
    assert wcs.sip is None


def test_alcor_frame_calibration_uses_filename_then_header(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.io import fits

    table = [
        {"epoch": "2024-09-04", "xshift": -4.5, "yshift": 4.4,
         "rotation": 0.39, "radial_coeffs": (1.0, 0.014, 0.0)},
        {"epoch": "2026-05-19", "xshift": -12.0, "yshift": 9.9,
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
        epoch="2026-05-19", xshift=-12.0, yshift=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0)))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["epoch"] == "2026-05-19"
    assert parsed["radial_coeffs"] == (1.0, 0.084, 0.0)
    assert abs(parsed["xshift"] + 12.0) < 1e-9


def test_plot_alcor_fits_defaults_rotation_to_none():
    import inspect
    from skycam_utils.alcor import plot_alcor_fits
    assert inspect.signature(plot_alcor_fits).parameters["rotation"].default is None


def test_cli_rotation_flags_default_to_none(monkeypatch):
    """Each CLI's --rotation default is None so load_alcor_fits auto-resolves.

    The parsers are constructed inside the *_cli functions, so capture the
    parsed namespace by patching parse_args and stop before the body runs.
    """
    import argparse
    import skycam_utils.alcor as alcor_mod

    captured = {}
    orig_parse = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        ns = orig_parse(self, *a, **k)
        captured["rotation"] = getattr(ns, "rotation", "MISSING")
        raise SystemExit  # stop before the CLI does real work
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)

    for cli, argv in [
        (alcor_mod.plot_alcor_fits_cli, ["prog", "in.fits"]),
        (alcor_mod.alcor_proc_fits_cli, ["prog", "in.fits"]),
        (alcor_mod.alcor_keogram_cli, ["prog", "in_dir"]),
    ]:
        captured.clear()
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit):
            cli()
        assert captured["rotation"] is None
