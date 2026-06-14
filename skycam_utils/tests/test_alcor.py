# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import shutil
from io import StringIO
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import (
    _annulus_background,
    _aperture_annulus_photometry,
    _aperture_saturated,
    _alcor_display_rgb,
    _corner_bias,
    _gaussian_channel_amplitude,
    _gaussian_psf_photometry,
    _timestamp_edges,
    alcor_keogram,
    alcor_proc_fits,
    alcor_star_photometry,
    build_alcor_wcs,
    load_alcor_keogram_fits,
    load_alcor_fits,
    collect_alcor_photometry,
    lookup_sloan_photometry,
    plot_alcor_keogram_fits,
    plot_alcor_fits,
    save_alcor_keogram_fits,
    save_alcor_keogram_plot,
)
from skycam_utils.alcor import ALCOR_NONLINEAR_THRESHOLD


TEST_FITS = Path(__file__).with_name("test.fits.bz2")


@pytest.fixture(scope="module")
def alcor_cube_wcs_mask():
    return load_alcor_fits(TEST_FITS)


def test_load_alcor_fits_returns_raw_cube_wcs_mask(alcor_cube_wcs_mask):
    cube, wcs, mask = alcor_cube_wcs_mask
    assert cube.ndim == 3 and cube.shape[0] == 3          # (3, ny, nx), no transpose
    assert cube.dtype == np.float32
    assert wcs.wcs.ctype[0].startswith("RA---ARC")
    assert mask is None or mask.shape == cube.shape       # native-orientation mask


def test_load_alcor_fits_no_bias_subtraction():
    cube, _, _ = load_alcor_fits(TEST_FITS, badpix=None)
    with fits.open(TEST_FITS) as hdul:
        raw = np.asarray(hdul[0].data, dtype=np.float32)
    np.testing.assert_array_equal(cube, raw)              # untouched: no -2000, no clip


def test_load_alcor_fits_accepts_explicit_wcs():
    w = build_alcor_wcs(xcen=10.0, ycen=20.0, rotation=0.0,
                        radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=30.0)
    _, wcs, _ = load_alcor_fits(TEST_FITS, wcs=w)
    assert list(wcs.wcs.crpix) == [11.0, 21.0]


def test_lookup_sloan_photometry_returns_catalog_dict():
    info = lookup_sloan_photometry(" vEgA ")

    assert info["NAME"] == "Vega"
    assert info["HD"] == 172167
    np.testing.assert_allclose(info["Vmag"], 0.03)
    np.testing.assert_allclose(info["g_mag"], -0.06)
    assert isinstance(info, dict)

    jabbah = lookup_sloan_photometry("Jabbah")
    assert jabbah["NAME"] == "Jabbah"
    assert jabbah["HD"] == 145502


def test_lookup_sloan_photometry_errors_for_missing_and_ambiguous(monkeypatch):
    from skycam_utils import alcor

    with pytest.raises(KeyError, match="not found"):
        lookup_sloan_photometry("definitely not a star")

    duplicate = Table({
        "NAME": ["dupe", "dupe"],
        "HD": [1, 2],
        "Vmag": [1.0, 2.0],
    })
    monkeypatch.setattr(alcor.Table, "read", lambda *args, **kwargs: duplicate)
    with pytest.raises(ValueError, match="ambiguous"):
        lookup_sloan_photometry("dupe")


def test_corner_bias_uses_four_corners_per_channel():
    cube = np.zeros((3, 20, 20), dtype=float)
    cube[0] += 100.0
    cube[1] += 200.0
    cube[2] += 300.0
    cube[:, 10, 10] = 10000.0

    bias = _corner_bias(cube, size=10)

    np.testing.assert_allclose(bias, [100.0, 200.0, 300.0])


def test_aperture_annulus_photometry_subtracts_local_background():
    yy, xx = np.mgrid[0:21, 0:21]
    image = np.full((21, 21), 5.0)
    aperture = np.hypot(xx - 10.0, yy - 10.0) <= 3.0
    gap = (np.hypot(xx - 10.0, yy - 10.0) > 3.0) & (
        np.hypot(xx - 10.0, yy - 10.0) <= 4.0)
    annulus = (np.hypot(xx - 10.0, yy - 10.0) > 4.0) & (
        np.hypot(xx - 10.0, yy - 10.0) <= 6.0)
    image[aperture] += 10.0
    image[gap] = 100.0
    image[annulus] = 5.0
    image[10, 15] = 500.0

    flux, background = _aperture_annulus_photometry(
        image, 10.0, 10.0, aperture_radius=3.0, annulus_width=2.0)

    assert background == 5.0  # median background ignores the annulus outlier
    np.testing.assert_allclose(flux, 10.0 * aperture.sum())


def test_annulus_background_is_median_of_annulus():
    yy, xx = np.mgrid[0:21, 0:21]
    image = np.full((21, 21), 5.0)
    rr = np.hypot(xx - 10.0, yy - 10.0)
    annulus = (rr > 4.0) & (rr <= 6.0)   # inner = ar+1 = 4, outer = 4+2 = 6
    image[annulus] = 5.0
    image[10, 16] = 500.0                # annulus outlier, killed by the median

    bg = _annulus_background(image, 10.0, 10.0, aperture_radius=3.0,
                             annulus_width=2.0)

    assert bg == 5.0


def test_annulus_background_returns_nan_when_off_image():
    image = np.zeros((10, 10))
    assert np.isnan(_annulus_background(image, -50.0, -50.0,
                                        aperture_radius=3.0, annulus_width=2.0))


def test_nonlinear_threshold_default_is_15000():
    assert ALCOR_NONLINEAR_THRESHOLD == 15000


def test_gaussian_channel_amplitude_recovers_known_amplitude():
    yy, xx = np.mgrid[0:11, 0:11]
    sigma = 1.5
    profile = np.exp(-((xx - 5.0) ** 2 + (yy - 5.0) ** 2) / (2.0 * sigma ** 2))
    amp_true = 1234.0
    background = 50.0
    data = amp_true * profile + background
    fit_mask = np.hypot(xx - 5.0, yy - 5.0) <= 4.0

    amp = _gaussian_channel_amplitude(data, background, profile, fit_mask)

    np.testing.assert_allclose(amp, amp_true, rtol=1e-6)


def test_gaussian_channel_amplitude_returns_nan_on_degenerate_profile():
    data = np.ones((5, 5))
    profile = np.zeros((5, 5))
    fit_mask = np.ones((5, 5), dtype=bool)
    assert np.isnan(_gaussian_channel_amplitude(data, 0.0, profile, fit_mask))


def test_alcor_display_rgb_subtracts_channel_bias_before_scaling():
    cube = np.zeros((3, 20, 20), dtype=float)
    cube[0] = 1000.0 + 100.0
    cube[1] = 2000.0 + 100.0 / 0.7
    cube[2] = 3000.0 + 100.0 / 1.7
    cube[0, 10, 10] += 1000.0
    cube[1, 10, 10] += 1000.0 / 0.7
    cube[2, 10, 10] += 1000.0 / 1.7

    rgb = _alcor_display_rgb(cube, gscale=0.7, bscale=1.7)

    np.testing.assert_allclose(rgb[10, 10, 0], rgb[10, 10, 1])
    np.testing.assert_allclose(rgb[10, 10, 0], rgb[10, 10, 2])


def test_alcor_star_photometry_writes_named_csv(tmp_path):
    input_file = tmp_path / TEST_FITS.name
    shutil.copyfile(TEST_FITS, input_file)
    output = tmp_path / "stars.csv"
    check_plot = tmp_path / "test_phot.pdf"

    phot, output_file = alcor_star_photometry(
        input_file,
        output_file=output,
        check_plot=True,
        aperture_radius=3.0,
        annulus_width=1.0,
        min_altitude=80.0,
        vmag_limit=5.0,
    )

    assert output_file == output
    assert output.exists()
    assert check_plot.exists()
    assert check_plot.stat().st_size > 0
    assert phot.index.name == "name"
    assert len(phot) > 0
    assert phot.index.is_unique
    for column in [
        "altitude", "azimuth", "xcen", "ycen",
        "flux_r", "mag_r", "background_r",
        "flux_g", "mag_g", "background_g",
        "flux_b", "mag_b", "background_b",
    ]:
        assert column in phot.columns
    flux_g = phot["flux_g"].to_numpy()
    finite = np.isfinite(flux_g)
    np.testing.assert_array_less(np.diff(flux_g[finite]), 1e-9)
    assert np.all(np.isfinite(phot["xcen"]))
    assert np.all(np.isfinite(phot["ycen"]))


def test_alcor_star_photometry_skips_nonpositive_channel_flux(tmp_path, monkeypatch):
    from astropy.time import Time
    from skycam_utils import alcor

    cube = np.zeros((3, 60, 60), dtype=float)
    yy, xx = np.mgrid[0:60, 0:60]
    first = np.hypot(xx - 15.0, yy - 15.0) <= 3.0
    second = np.hypot(xx - 40.0, yy - 40.0) <= 3.0
    cube[:, first] = 10.0
    cube[0, second] = 10.0
    cube[2, second] = 10.0

    class FakeWCS:
        def world_to_pixel_values(self, az, alt):
            return np.array([15.0, 40.0]), np.array([15.0, 40.0])

    cat = Table({
        "NAME": ["keep", "skip"],
        "HD": [1, 2],
        "Alt": [80.0, 81.0],
        "Az": [10.0, 20.0],
    })
    monkeypatch.setattr(alcor, "_alcor_frame_time",
                        lambda filename: Time("2024-09-05T07:00:00"))
    monkeypatch.setattr(alcor, "load_alcor_fits",
                        lambda *args, **kwargs: (cube, FakeWCS(), None))
    monkeypatch.setattr(alcor, "alcor_named_reference_altaz",
                        lambda *args, **kwargs: cat)

    phot, output_file = alcor_star_photometry(
        tmp_path / "synthetic.fits",
        output_file=tmp_path / "synthetic.csv",
        aperture_radius=3.0,
        annulus_width=1.0,
    )

    assert output_file.exists()
    text = output_file.read_text()
    assert "keep" in text
    assert "skip" not in text
    assert list(phot.index) == ["keep"]
    assert phot.loc["keep", "flux_g"] > 0.0
    assert np.isfinite(phot.loc["keep", "mag_r"])
    assert np.isfinite(phot.loc["keep", "mag_g"])
    assert np.isfinite(phot.loc["keep", "mag_b"])


def test_aperture_saturated_detects_ceiling_pixel():
    image = np.zeros((20, 20), dtype=float)
    image[10, 10] = 32767.0
    assert _aperture_saturated(image, 10.0, 10.0, 3.0, 32767.0) is True
    assert _aperture_saturated(image, 10.0, 10.0, 3.0, 40000.0) is False
    # a ceiling pixel outside the aperture does not count
    assert _aperture_saturated(image, 2.0, 2.0, 3.0, 32767.0) is False


def test_alcor_star_photometry_flags_saturation(tmp_path, monkeypatch):
    from astropy.time import Time
    from skycam_utils import alcor

    cube = np.zeros((3, 60, 60), dtype=float)
    yy, xx = np.mgrid[0:60, 0:60]
    bright = np.hypot(xx - 15.0, yy - 15.0) <= 3.0
    faint = np.hypot(xx - 40.0, yy - 40.0) <= 3.0
    cube[:, bright] = 1000.0
    cube[:, faint] = 1000.0
    # saturate only the green channel of the bright star
    cube[1, 15, 15] = 32767.0

    class FakeWCS:
        def world_to_pixel_values(self, az, alt):
            return np.array([15.0, 40.0]), np.array([15.0, 40.0])

    cat = Table({
        "NAME": ["bright", "faint"],
        "HD": [1, 2],
        "Alt": [80.0, 81.0],
        "Az": [10.0, 20.0],
    })
    monkeypatch.setattr(alcor, "_alcor_frame_time",
                        lambda filename: Time("2024-09-05T07:00:00"))
    monkeypatch.setattr(alcor, "load_alcor_fits",
                        lambda *args, **kwargs: (cube, FakeWCS(), None))
    monkeypatch.setattr(alcor, "alcor_named_reference_altaz",
                        lambda *args, **kwargs: cat)

    phot, _ = alcor_star_photometry(
        tmp_path / "synthetic.fits",
        output_file=tmp_path / "synthetic.csv",
        aperture_radius=3.0,
        annulus_width=1.0,
    )

    for channel in ("r", "g", "b"):
        assert f"sat_{channel}" in phot.columns
    assert bool(phot.loc["bright", "sat_g"]) is True
    assert bool(phot.loc["bright", "sat_r"]) is False
    assert bool(phot.loc["bright", "sat_b"]) is False
    assert bool(phot.loc["faint", "sat_g"]) is False


def _gaussian_star_cube(ny=80, nx=80, cx=40.3, cy=39.7, sigma=1.3,
                        amps=(4000.0, 6000.0, 3000.0), base=100.0):
    yy, xx = np.mgrid[0:ny, 0:nx]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    cube = np.full((3, ny, nx), base, dtype=float)
    for idx, amp in enumerate(amps):
        cube[idx] += amp * g
    return cube


def _patch_single_star(monkeypatch, cube, px, py):
    from astropy.time import Time
    from skycam_utils import alcor

    class FakeWCS:
        def world_to_pixel_values(self, az, alt):
            return np.array([float(px)]), np.array([float(py)])

    cat = Table({"NAME": ["star"], "HD": [1], "Alt": [80.0], "Az": [10.0]})
    monkeypatch.setattr(alcor, "_alcor_frame_time",
                        lambda filename: Time("2024-09-05T07:00:00"))
    monkeypatch.setattr(alcor, "load_alcor_fits",
                        lambda *a, **k: (cube, FakeWCS(), None))
    monkeypatch.setattr(alcor, "alcor_named_reference_altaz",
                        lambda *a, **k: cat)


def test_alcor_star_photometry_gaussian_recovers_clean_star(tmp_path, monkeypatch):
    cx, cy, sigma = 40.3, 39.7, 1.3
    amps = (4000.0, 6000.0, 3000.0)
    cube = _gaussian_star_cube(cx=cx, cy=cy, sigma=sigma, amps=amps)
    _patch_single_star(monkeypatch, cube, px=40.0, py=40.0)

    phot, _ = alcor_star_photometry(
        tmp_path / "synthetic.fits", output_file=tmp_path / "out.csv",
        gaussian=True, aperture_radius=5.0, annulus_width=2.0)

    assert "fwhm" in phot.columns
    np.testing.assert_allclose(phot.loc["star", "xcen"], cx, atol=0.1)
    np.testing.assert_allclose(phot.loc["star", "ycen"], cy, atol=0.1)
    np.testing.assert_allclose(
        phot.loc["star", "fwhm"],
        2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma, rtol=0.05)
    for idx, channel in enumerate("rgb"):
        expected = amps[idx] * 2.0 * np.pi * sigma ** 2
        np.testing.assert_allclose(
            phot.loc["star", f"flux_{channel}"], expected, rtol=0.05)


def test_alcor_star_photometry_gaussian_beats_aperture_under_nonlinearity(
        tmp_path, monkeypatch):
    cx, cy, sigma = 40.0, 40.0, 1.3
    amp = 40000.0
    ceiling = 16000.0
    cube = _gaussian_star_cube(cx=cx, cy=cy, sigma=sigma,
                               amps=(amp, amp, amp))
    cube = np.minimum(cube, ceiling)        # mimic per-pixel non-linearity
    true_flux = amp * 2.0 * np.pi * sigma ** 2

    _patch_single_star(monkeypatch, cube, px=cx, py=cy)
    phot_g, _ = alcor_star_photometry(
        tmp_path / "g.fits", output_file=tmp_path / "g.csv",
        gaussian=True, aperture_radius=5.0, annulus_width=2.0,
        mask_threshold=15000.0)

    _patch_single_star(monkeypatch, cube, px=cx, py=cy)
    phot_a, _ = alcor_star_photometry(
        tmp_path / "a.fits", output_file=tmp_path / "a.csv",
        gaussian=False, aperture_radius=5.0, annulus_width=2.0)

    rg = float(phot_g.loc["star", "flux_g"])
    ra = float(phot_a.loc["star", "flux_g"])
    # the clamped core was never at the 32767 ceiling, so sat does not catch it
    assert bool(phot_g.loc["star", "sat_g"]) is False
    assert ra < true_flux                            # aperture underestimates
    assert abs(rg - true_flux) < abs(ra - true_flux) # gaussian is closer
    np.testing.assert_allclose(rg, true_flux, rtol=0.1)


def test_alcor_star_photometry_aperture_mode_fwhm_is_nan(tmp_path, monkeypatch):
    cube = _gaussian_star_cube()
    _patch_single_star(monkeypatch, cube, px=40.0, py=40.0)

    phot, _ = alcor_star_photometry(
        tmp_path / "a.fits", output_file=tmp_path / "a.csv",
        gaussian=False, aperture_radius=5.0, annulus_width=2.0)

    assert "fwhm" in phot.columns
    assert phot["fwhm"].isna().all()


def test_alcor_star_photometry_gaussian_drops_signal_free_star(
        tmp_path, monkeypatch):
    cube = np.full((3, 60, 60), 100.0)      # flat field, no star
    _patch_single_star(monkeypatch, cube, px=30.0, py=30.0)

    phot, output_file = alcor_star_photometry(
        tmp_path / "flat.fits", output_file=tmp_path / "flat.csv",
        gaussian=True, aperture_radius=5.0, annulus_width=2.0)

    assert len(phot) == 0
    assert output_file.exists()


def test_alcor_star_photometry_default_vmag_limit_is_5p5(monkeypatch, tmp_path):
    from astropy.time import Time
    from skycam_utils import alcor

    class FakeWCS:
        def world_to_pixel_values(self, az, alt):
            return np.array([10.0]), np.array([10.0])

    seen = {}
    monkeypatch.setattr(alcor, "_alcor_frame_time",
                        lambda filename: Time("2024-09-05T07:00:00"))
    monkeypatch.setattr(alcor, "load_alcor_fits",
                        lambda *args, **kwargs: (
                            np.zeros((3, 20, 20), dtype=float), FakeWCS(), None))

    def fake_catalog(time, vmag_limit, min_alt, refraction):
        seen["vmag_limit"] = vmag_limit
        return Table({"NAME": ["default"], "HD": [1], "Alt": [80.0], "Az": [10.0]})

    monkeypatch.setattr(alcor, "alcor_named_reference_altaz", fake_catalog)

    def fake_photometry(image, xcen, ycen, aperture_radius, annulus_width):
        seen["aperture_radius"] = aperture_radius
        return 1.0, 0.0

    monkeypatch.setattr(alcor, "_aperture_annulus_photometry", fake_photometry)

    alcor_star_photometry(tmp_path / "synthetic.fits",
                          output_file=tmp_path / "synthetic.csv")

    assert seen["vmag_limit"] == 5.5
    assert seen["aperture_radius"] == 4.0


def test_alcor_star_photometry_rejects_sunlit_images(monkeypatch, tmp_path, capsys):
    from astropy.time import Time
    from skycam_utils import alcor

    output = tmp_path / "sunlit.csv"
    loaded = {"called": False}
    monkeypatch.setattr(alcor, "_alcor_frame_time",
                        lambda filename: Time("2024-09-05T20:00:00"))
    monkeypatch.setattr(alcor, "_sun_altitude", lambda time: -5.0)

    def fake_load(*args, **kwargs):
        loaded["called"] = True
        raise AssertionError("sunlit image should not be loaded")

    monkeypatch.setattr(alcor, "load_alcor_fits", fake_load)

    phot, output_file = alcor_star_photometry(
        tmp_path / "sunlit.fits",
        output_file=output,
    )

    captured = capsys.readouterr()
    assert "Warning: rejecting" in captured.err
    assert "Sun altitude -5.0 deg" in captured.err
    assert output_file is None
    assert phot.empty
    assert not output.exists()
    assert not loaded["called"]


_PHOT_COLUMNS = ("name,altitude,azimuth,xcen,ycen,"
                 "flux_r,mag_r,background_r,"
                 "flux_g,mag_g,background_g,"
                 "flux_b,mag_b,background_b")


def _write_phot_csv(path, rows):
    lines = [_PHOT_COLUMNS]
    for name, flux in rows:
        lines.append(f"{name},45.0,180.0,100.0,200.0,"
                     f"{flux},-10.0,5.0,{flux},-10.0,5.0,{flux},-10.0,5.0")
    path.write_text("\n".join(lines) + "\n")


def test_collect_alcor_photometry_groups_and_sorts(tmp_path):
    _write_phot_csv(tmp_path / "2024_09_04__20_00_00_phot.csv",
                    [("Vega", 100.0), ("Arcturus", 50.0)])
    _write_phot_csv(tmp_path / "2024_09_04__19_36_32_phot.csv",
                    [("Vega", 200.0)])

    df = collect_alcor_photometry(tmp_path)

    assert list(df["name"]) == ["Arcturus", "Vega", "Vega"]
    vega = df[df["name"] == "Vega"]
    # sorted by OBSTIME within each star; filename MST + 7h = UT
    assert list(vega["OBSTIME"]) == [
        np.datetime64("2024-09-05T02:36:32"),
        np.datetime64("2024-09-05T03:00:00"),
    ]
    assert list(vega["flux_g"]) == [200.0, 100.0]
    assert "altitude" in df.columns and "background_b" in df.columns


def test_collect_alcor_photometry_accepts_file_list(tmp_path):
    f1 = tmp_path / "2024_09_04__19_36_32_phot.csv"
    f2 = tmp_path / "2024_09_04__20_00_00_phot.csv"
    _write_phot_csv(f1, [("Vega", 1.0)])
    _write_phot_csv(f2, [("Vega", 2.0)])

    from_dir = collect_alcor_photometry(tmp_path)
    from_list = collect_alcor_photometry([f1, f2])

    assert from_dir.equals(from_list)


def test_collect_alcor_photometry_skips_unparseable_filename(tmp_path, capsys):
    _write_phot_csv(tmp_path / "2024_09_04__19_36_32_phot.csv",
                    [("Vega", 1.0)])
    _write_phot_csv(tmp_path / "oddly_named_phot.csv", [("Deneb", 1.0)])

    df = collect_alcor_photometry(tmp_path)

    assert list(df["name"]) == ["Vega"]
    assert "oddly_named_phot.csv" in capsys.readouterr().err


def test_collect_alcor_photometry_skips_malformed_csv(tmp_path, capsys):
    _write_phot_csv(tmp_path / "2024_09_04__19_36_32_phot.csv",
                    [("Vega", 1.0)])
    (tmp_path / "2024_09_04__20_00_00_phot.csv").write_bytes(b"\x00\x01\x02")

    df = collect_alcor_photometry(tmp_path)

    assert list(df["name"]) == ["Vega"]
    assert "2024_09_04__20_00_00_phot.csv" in capsys.readouterr().err


def test_collect_alcor_photometry_raises_when_nothing_usable(tmp_path):
    with pytest.raises(ValueError):
        collect_alcor_photometry(tmp_path)

    _write_phot_csv(tmp_path / "oddly_named_phot.csv", [("Vega", 1.0)])
    with pytest.raises(ValueError):
        collect_alcor_photometry(tmp_path)


def test_load_alcor_fits_wcs_maps_zenith_and_horizon(alcor_cube_wcs_mask):
    _, wcs, _ = alcor_cube_wcs_mask

    # The zenith pixel (looked up via the WCS -- with a tilted optical axis it
    # is NOT crpix) round-trips to alt=90.
    xz, yz = wcs.world_to_pixel_values(0.0, 90.0)
    _, zenith_alt = wcs.pixel_to_world_values(xz, yz)
    np.testing.assert_allclose(zenith_alt, 90.0, atol=0.02)

    # The SIP-encoded radial model round-trips world -> pixel -> world.
    az = np.array([10.0, 100.0, 200.0, 300.0])
    alt = np.array([15.0, 35.0, 55.0, 75.0])
    px, py = wcs.world_to_pixel_values(az, alt)
    az2, alt2 = wcs.pixel_to_world_values(px, py)
    np.testing.assert_allclose(alt2, alt, atol=0.02)
    np.testing.assert_allclose(az2 % 360.0, az % 360.0, atol=0.02)


def test_alcor_proc_fits_writes_processed_cube_and_header(tmp_path):
    input_file = tmp_path / "sample.fits.bz2"
    shutil.copyfile(TEST_FITS, input_file)

    output_file = alcor_proc_fits(input_file, overwrite=False)

    assert output_file == tmp_path / "sample_proc.fits"
    assert output_file.exists()

    cube, wcs, _ = load_alcor_fits(input_file)
    with fits.open(output_file) as hdul:
        # the raw (3, ny, nx) cube is written untouched (native orientation)
        assert hdul[0].data.shape == cube.shape
        assert hdul[0].data.dtype.kind == "f"
        assert hdul[0].data.dtype.itemsize == np.dtype(np.float32).itemsize
        np.testing.assert_array_equal(hdul[0].data, cube.astype(np.float32))
        assert hdul[0].header["CTYPE1"].startswith("RA---ARC")
        assert hdul[0].header["CTYPE2"].startswith("DEC--ARC")
        # header WCS round-trips to the same reference pixel as the loaded WCS
        np.testing.assert_allclose(hdul[0].header["CRPIX1"], wcs.wcs.crpix[0])
        np.testing.assert_allclose(hdul[0].header["CRPIX2"], wcs.wcs.crpix[1])


def test_alcor_keogram_uses_center_columns_and_date_headers(tmp_path):
    for index in range(3):
        shutil.copyfile(TEST_FITS, tmp_path / f"alcor_{index:03d}.fits.bz2")

    keogram, timestamps, files = alcor_keogram(tmp_path)
    cube, wcs, _ = load_alcor_fits(TEST_FITS)
    zx, _ = wcs.world_to_pixel_values(0.0, 90.0)
    zcol = int(round(float(zx)))                          # 0-based zenith column
    ny = cube.shape[1]

    assert keogram.shape == (ny, 3, 3)
    assert len(timestamps) == 3
    assert all(timestamp == "2024-09-05T06:51:31.224500" for timestamp in timestamps)
    assert [file.name for file in files] == [
        "alcor_000.fits.bz2",
        "alcor_001.fits.bz2",
        "alcor_002.fits.bz2",
    ]
    np.testing.assert_allclose(keogram[:, 0, :], cube[:, :, zcol].T)


def test_alcor_keogram_can_report_progress(tmp_path):
    for index in range(2):
        shutil.copyfile(TEST_FITS, tmp_path / f"alcor_{index:03d}.fits.bz2")

    progress_file = StringIO()
    alcor_keogram(
        tmp_path,
        progress=True,
        progress_file=progress_file,
    )

    progress_output = progress_file.getvalue()
    assert "1/2" in progress_output
    assert "2/2" in progress_output
    assert "100.0%" in progress_output
    assert progress_output.endswith("\n")


def test_alcor_keogram_dispatches_center_columns_to_workers(tmp_path, monkeypatch):
    for index in range(3):
        shutil.copyfile(TEST_FITS, tmp_path / f"alcor_{index:03d}.fits.bz2")

    serial_keogram, serial_timestamps, serial_files = alcor_keogram(
        tmp_path,
        workers=1,
    )

    submitted_tasks = []

    class FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class FakeExecutor:
        def __init__(self, max_workers=None):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def submit(self, func, task):
            submitted_tasks.append(task)
            return FakeFuture(func(task))

    monkeypatch.setattr("skycam_utils.alcor.ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr("skycam_utils.alcor.as_completed", lambda futures: futures)

    parallel_keogram, parallel_timestamps, parallel_files = alcor_keogram(
        tmp_path,
        workers=2,
    )

    assert len(submitted_tasks) == 3
    np.testing.assert_allclose(parallel_keogram, serial_keogram)
    assert parallel_timestamps == serial_timestamps
    assert parallel_files == serial_files


def test_save_alcor_keogram_plot_writes_output(tmp_path):
    for index in range(2):
        shutil.copyfile(TEST_FITS, tmp_path / f"alcor_{index:03d}.fits.bz2")

    keogram, timestamps, _ = alcor_keogram(tmp_path)
    output_file = save_alcor_keogram_plot(
        keogram,
        timestamps,
        tmp_path / "keogram.png",
        figsize=(3, 2),
    )

    assert output_file == tmp_path / "keogram.png"
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_timestamp_edges_preserve_irregular_cadence():
    edges = _timestamp_edges(np.array([1.0, 1.5, 3.0]))

    np.testing.assert_allclose(edges, [0.75, 1.25, 2.25, 3.75])


def test_save_alcor_keogram_plot_accepts_irregular_timestamps(tmp_path):
    keogram = np.ones((4, 3, 3))
    timestamps = [
        "2024-09-04T19:00:00",
        "2024-09-04T19:01:00",
        "2024-09-04T19:05:00",
    ]

    output_file = save_alcor_keogram_plot(
        keogram,
        timestamps,
        tmp_path / "irregular_keogram.png",
        figsize=(3, 2),
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_save_alcor_keogram_fits_writes_image_and_timestamp_table(tmp_path):
    for index in range(2):
        shutil.copyfile(TEST_FITS, tmp_path / f"alcor_{index:03d}.fits.bz2")

    keogram, timestamps, _ = alcor_keogram(tmp_path)
    output_file = save_alcor_keogram_fits(
        keogram,
        timestamps,
        tmp_path / "keogram.fits",
    )

    ny = keogram.shape[0]
    assert output_file == tmp_path / "keogram.fits"
    assert output_file.exists()
    with fits.open(output_file) as hdul:
        assert hdul[0].data.shape == (3, ny, 2)
        assert hdul[0].data.dtype.kind == "f"
        assert hdul[1].name == "TIMESTAMPS"
        assert list(hdul[1].data["DATE"]) == timestamps


def test_load_alcor_keogram_fits_round_trips_saved_keogram(tmp_path):
    keogram = np.arange(4 * 3 * 3).reshape(4, 3, 3)
    timestamps = [
        "2024-09-04T19:00:00",
        "2024-09-04T19:01:00",
        "2024-09-04T19:05:00",
    ]
    fits_file = save_alcor_keogram_fits(
        keogram,
        timestamps,
        tmp_path / "keogram.fits",
    )

    loaded_keogram, loaded_timestamps = load_alcor_keogram_fits(fits_file)

    np.testing.assert_allclose(loaded_keogram, keogram)
    assert loaded_timestamps == timestamps


def test_plot_alcor_keogram_fits_writes_output(tmp_path):
    keogram = np.ones((4, 3, 3))
    timestamps = [
        "2024-09-04T19:00:00",
        "2024-09-04T19:01:00",
        "2024-09-04T19:05:00",
    ]
    fits_file = save_alcor_keogram_fits(
        keogram,
        timestamps,
        tmp_path / "keogram.fits",
    )
    output_file = plot_alcor_keogram_fits(
        fits_file,
        output_file=tmp_path / "keogram.png",
        figsize=(3, 2),
    )

    assert output_file == tmp_path / "keogram.png"
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_alcor_fits_writes_outputs_and_returns_figure(tmp_path):
    outimage = tmp_path / "alcor.png"
    outfig = tmp_path / "alcor.pdf"

    fig = plot_alcor_fits(
        TEST_FITS,
        outimage=outimage,
        outfig=outfig,
        radius=600,
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
