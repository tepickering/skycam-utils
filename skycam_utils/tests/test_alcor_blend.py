# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
import pandas as pd
import pytest
import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from skycam_utils.alcor import (
    ALCOR_BLEND_EFFICIENCY,
    MMT_LOCATION,
    _catalog_calibration_map,
    _frame_registration,
    _neighbor_table,
    _parallactic_angle,
    alcor_blend_terms,
    alcor_calibrate_photometry,
    alcor_extinction_stars,
)


def _sep_arcmin(rows):
    return np.hypot(rows["DE"], rows["DN"]).to_numpy() * 60.0


def test_neighbor_table_keeps_real_pairs_and_drops_self():
    nb = _neighbor_table()
    # nu Dra: 24 and 25 Dra are an equal pair 1' apart. Each is the other's
    # neighbour; matching "self" by magnitude alone would swallow the twin.
    for star in ("24 Dra", "25 Dra"):
        sep = _sep_arcmin(nb[nb["LABEL"] == star])
        assert np.isclose(sep.min(), 1.0, atol=0.1)
    # 30 Cas's own Hipparcos entry is 33" from the catalog position; it must not
    # be listed as a neighbour of itself (that predicted a -1.4 mag blend).
    assert _sep_arcmin(nb[nb["LABEL"] == "30 Cas"]).min() > 5.0


def test_parallactic_angle_puts_offsets_in_the_horizontal_frame():
    t = Time("2026-10-15T04:00:00")
    frame = AltAz(obstime=t, location=MMT_LOCATION)
    rng = np.random.default_rng(3)
    stars = SkyCoord(rng.uniform(0, 360, 200) * u.deg,
                     rng.uniform(-20, 85, 200) * u.deg)
    de, dn = 0.3, -0.2
    aa = stars.transform_to(frame)
    moved = stars.spherical_offsets_by(de * u.deg, dn * u.deg).transform_to(frame)
    up_ok = aa.alt.deg > 20
    q = _parallactic_angle(aa.az.deg, aa.alt.deg, MMT_LOCATION.lat.deg)
    up = de * np.sin(q) + dn * np.cos(q)
    # second-order curvature of a 0.36 deg offset reaches ~0.011 deg near the
    # zenith; a sign or axis error would be ~0.3 deg
    assert np.allclose(up[up_ok], (moved.alt.deg - aa.alt.deg)[up_ok], atol=0.015)


def _rows(names, az, alt):
    return pd.DataFrame({"name": names, "azimuth": az, "altitude": alt})


def test_blend_terms_isolated_blended_and_unknown():
    cmap = _catalog_calibration_map()
    names = ["10 PsA", "HD 166229", "Vega"]
    df = _rows(names, [200.0, 292.0, 80.0], [30.0, 42.0, 60.0])
    mags = {"g": np.array([cmap[n]["V"] for n in names[:2]] + [np.nan])}
    terms = alcor_blend_terms(df, mags, time=Time("2026-10-15T04:00:00"))["g"]
    assert terms[0] == 0.0                     # no Hipparcos neighbour at all
    # HD 166229 + HIP 89008 (V 5.57, 24'): measured -0.79 on 2026-10-14
    assert -0.85 < terms[1] < -0.55
    assert np.isnan(terms[2])                  # no catalog magnitude (variable)


def test_blend_terms_scale_with_efficiency():
    cmap = _catalog_calibration_map()
    df = _rows(["HD 166229"], [292.0], [42.0])
    mags = {"g": np.array([cmap["HD 166229"]["V"]])}
    full = alcor_blend_terms(df, mags, efficiency=1.0)["g"][0]
    used = alcor_blend_terms(df, mags)["g"][0]
    ratio = (10 ** (-0.4 * used) - 1) / (10 ** (-0.4 * full) - 1)
    assert np.isclose(ratio, ALCOR_BLEND_EFFICIENCY)


def test_blend_depends_on_where_the_star_is():
    """
    The 4 px aperture is a 0.61 x 0.39 deg ellipse at alt 20 and a 0.48 deg
    circle at the zenith, so a neighbour ~0.45 deg away is in or out of the
    aperture depending on the star's altitude and on its direction.
    """
    cmap = _catalog_calibration_map()
    df = _rows(["HD 166229"] * 3, [292.0, 300.0, 0.0], [20.0, 50.0, 85.0])
    mags = {"g": np.full(3, cmap["HD 166229"]["V"])}
    terms = alcor_blend_terms(df, mags)["g"]
    assert len(set(np.round(terms, 3))) > 1


def test_calibrate_writes_blend_and_subtracts_it_from_the_reference():
    df = pd.DataFrame({"altitude": [42.0], "azimuth": [292.0],
                       "mag_g": [-10.0]}, index=["HD 166229"])
    out = alcor_calibrate_photometry(df, time=Time("2026-10-15T04:00:00"))
    cat_v = _catalog_calibration_map()["HD 166229"]["V"]
    assert out["blend_g"].iloc[0] < -0.5
    assert np.isclose(out["ext_g"].iloc[0],
                      out["cal_g"].iloc[0] - (cat_v + out["blend_g"].iloc[0]))

    off = alcor_calibrate_photometry(df, time=Time("2026-10-15T04:00:00"),
                                     blend=False)
    assert off["blend_g"].iloc[0] == 0.0
    assert np.isclose(off["cal_g"].iloc[0], out["cal_g"].iloc[0])


def _star_image(xs, ys, shift, shape=(200, 200), sigma=0.7, peak=5000.0,
                seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(800.0, 15.0, shape)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for x, y in zip(xs, ys):
        img += peak * np.exp(-0.5 * (((xx - x - shift[0]) ** 2
                                      + (yy - y - shift[1]) ** 2) / sigma ** 2))
    return img


def test_frame_registration_recovers_a_translation():
    rng = np.random.default_rng(1)
    xs, ys = rng.uniform(15, 185, 30), rng.uniform(15, 185, 30)
    img = _star_image(xs, ys, shift=(-0.23, 1.41))
    dx, dy, n = _frame_registration(img, xs, ys, np.full(30, 2.0),
                                    np.full(30, 60.0))
    assert n == 30
    assert abs(dx + 0.23) < 0.1 and abs(dy - 1.41) < 0.1


def test_frame_registration_declines_when_too_few_stars():
    """Cloud: no registration beats a wrong one."""
    xs, ys = np.array([50.0, 100.0, 150.0]), np.array([50.0, 100.0, 150.0])
    img = _star_image(xs, ys, shift=(1.0, 1.0))
    assert _frame_registration(img, xs, ys, np.full(3, 2.0),
                               np.full(3, 60.0)) == (0.0, 0.0, 3)


def test_extinction_stars_recalibrates_legacy_photometry():
    """
    A CSV written before the blend correction has old-zeropoint ext_* and no
    blend_*; the map must recalibrate it rather than mix the two.
    """
    names = ["HD 166229", "Vega", "Deneb"]
    rows = []
    for k in range(5):
        rows.append(pd.DataFrame({
            "name": names, "OBSTIME": pd.Timestamp("2026-10-15T04:00:00")
            + pd.Timedelta(seconds=30 * k),
            "altitude": [42.0, 60.0, 70.0], "azimuth": [292.0, 280.0, 40.0],
            "variable": False, "xcen": 700.0, "ycen": 700.0,
            "mag_g": -10.2, "cal_g": 99.0, "ext_g": 99.0,
            "flux_g": 1000.0, "sat_g": False}))
    df = pd.concat(rows, ignore_index=True)
    measured, _ = alcor_extinction_stars(df, horizon_margin=None)
    assert len(measured)
    assert (measured["ext"] < 50).all()
