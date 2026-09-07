# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import shutil
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import (
    ALCOR_SB_APERTURE_RADIUS,
    ALCOR_SB_BEST_MIN_ALTITUDE,
    ALCOR_SB_TARGETS,
    _alcor_best_cone_targets,
    _alcor_cone_indices,
    _cone_median,
    _photometry_is_done,
    _set_keogram_yaxis,
    alcor_process_night,
    alcor_star_photometry,
    build_alcor_wcs,
    load_alcor_fits,
    load_alcor_keogram_fits,
    load_alcor_sb_keogram_fits,
    save_alcor_sb_keogram_fits,
)

TEST_FITS = Path(__file__).with_name("test.fits.bz2")
# The bundled frame is 2024-09-04 23:51 local (MST); alcor filenames carry the
# local timestamp, so these names put the copies on a real dark-sky night.
FRAME_STAMPS = ("2024_09_04__23_51_09", "2024_09_04__23_55_09")
FRAME_SHAPE = (3, 1411, 1422)


def _make_night(directory, stamps=FRAME_STAMPS):
    """Populate ``directory`` with copies of the bundled frame, night-stamped."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stamp in stamps:
        shutil.copy(TEST_FITS, directory / f"{stamp}.fits.bz2")
    return directory


@pytest.fixture(scope="module")
def night_run(tmp_path_factory):
    """One processed night, shared by the tests that only inspect its products."""
    root = tmp_path_factory.mktemp("night")
    night_dir = _make_night(root / "2024-09-04")
    out_dir = root / "out"
    result = alcor_process_night(night_dir, out_dir=out_dir, workers=1,
                                 median_stack=True,
                                 scratch_dir=str(root))
    return result, night_dir, out_dir


def test_cone_indices_select_only_pixels_inside_the_cone():
    # Every selected pixel must lie within the angular radius of its target, and
    # the target direction itself must be selected.
    wcs = build_alcor_wcs()
    ny, nx = FRAME_SHAPE[1:]
    cones = _alcor_cone_indices(wcs, (ny, nx))
    assert set(cones) == set(ALCOR_SB_TARGETS)

    for name, (az0, alt0) in ALCOR_SB_TARGETS.items():
        idx = cones[name]
        assert idx.size > 0
        yy, xx = np.unravel_index(idx, (ny, nx))
        az, alt = wcs.pixel_to_world_values(xx.astype(float), yy.astype(float))
        sep = np.degrees(np.arccos(np.clip(
            np.sin(np.radians(alt)) * np.sin(np.radians(alt0))
            + np.cos(np.radians(alt)) * np.cos(np.radians(alt0))
            * np.cos(np.radians(az - az0)), -1.0, 1.0)))
        assert sep.max() <= ALCOR_SB_APERTURE_RADIUS + 1e-6

        # the target's own pixel is in the cone
        tx, ty = wcs.all_world2pix(az0, alt0, 0, quiet=True)
        centre = int(round(float(ty))) * nx + int(round(float(tx)))
        assert centre in set(idx.tolist())


def test_cone_indices_area_matches_the_plate_scale():
    # A 5 deg cone covers pi*r^2 sq deg; at the ARC plate scale that is a
    # predictable pixel count, so a geometry error shows up as a size error.
    wcs = build_alcor_wcs()
    ny, nx = FRAME_SHAPE[1:]
    cones = _alcor_cone_indices(wcs, (ny, nx))
    cdelt = abs(float(wcs.wcs.cdelt[1]))
    expected = np.pi * ALCOR_SB_APERTURE_RADIUS ** 2 / cdelt ** 2
    # the zenith is where the ARC projection is closest to flat
    assert cones["allsky_mv_zenith"].size == pytest.approx(expected, rel=0.05)


def test_best_cone_targets_tile_the_sky_above_the_floor():
    radius = ALCOR_SB_APERTURE_RADIUS
    targets = _alcor_best_cone_targets(radius_deg=radius,
                                       min_altitude=ALCOR_SB_BEST_MIN_ALTITUDE)
    assert targets

    alts = sorted({round(alt, 6) for _, alt in targets.values()})
    # every cone lies wholly above the floor ...
    assert min(alts) >= ALCOR_SB_BEST_MIN_ALTITUDE + radius - 1e-9
    # ... rings are one cone diameter apart, and the zenith is a candidate so
    # allsky_mv_best can never come out brighter than allsky_mv_zenith
    assert 90.0 in alts
    ring = [a for a in alts if a < 90.0]
    assert np.allclose(np.diff(ring), 2 * radius)

    for az, alt in targets.values():
        assert 0.0 <= az < 360.0
        assert ALCOR_SB_BEST_MIN_ALTITUDE <= alt <= 90.0

    # azimuth sampling widens with altitude so cones do not bunch at the pole
    counts = {}
    for az, alt in targets.values():
        counts[round(alt)] = counts.get(round(alt), 0) + 1
    lows = [counts[a] for a in sorted(counts) if a < 90]
    assert lows == sorted(lows, reverse=True)


def test_cone_indices_drop_excluded_pixels():
    wcs = build_alcor_wcs()
    ny, nx = FRAME_SHAPE[1:]
    exclude = np.zeros((ny, nx), dtype=bool)
    zx, zy = wcs.world_to_pixel_values(0.0, 90.0)
    zx, zy = int(round(float(zx))), int(round(float(zy)))
    exclude[zy - 100:zy + 100, zx - 100:zx + 100] = True

    cones = _alcor_cone_indices(wcs, (ny, nx), exclude=exclude)
    assert cones["allsky_mv_zenith"].size == 0
    # the low-altitude cones are far from the excluded block and survive
    assert cones["allsky_mv_tucson"].size > 0


def test_cone_median_ignores_nans_and_empty_cones():
    mu = np.array([[1.0, 2.0], [np.nan, 4.0]])
    assert _cone_median(mu, np.array([0, 1, 2, 3])) == 2.0
    assert np.isnan(_cone_median(mu, np.array([2])))
    assert np.isnan(_cone_median(mu, np.array([], dtype=int)))


def test_star_photometry_frame_passthrough_matches_internal_load():
    # The batch driver hands alcor_star_photometry an already-loaded frame to
    # avoid a second decompress; that shortcut must not change the numbers.
    frame = load_alcor_fits(TEST_FITS, badpix="repair")
    direct, _ = alcor_star_photometry(TEST_FITS, output_file=os.devnull)
    passed, _ = alcor_star_photometry(TEST_FITS, output_file=os.devnull,
                                      frame=frame)
    assert list(direct.index) == list(passed.index)
    np.testing.assert_allclose(direct["flux_g"].to_numpy(),
                               passed["flux_g"].to_numpy(), rtol=0, atol=0)


def test_process_night_writes_the_expected_products(night_run):
    result, night_dir, out_dir = night_run
    assert result["errors"] == []
    assert len(result["files"]) == len(FRAME_STAMPS)

    for key in ("summary_file", "keogram_file", "keogram_plot",
                "photometry_file", "median_file"):
        assert result[key].exists(), key
    assert result["summary_file"] == out_dir / "sky_brightness.csv"
    # nothing is written into the (possibly read-only) archive directory
    assert sorted(p.name for p in night_dir.iterdir()) == [
        f"{stamp}.fits.bz2" for stamp in FRAME_STAMPS]
    # the per-frame surface-brightness maps are opt-in
    assert not list(out_dir.glob("*_sb.fits"))
    assert len(list(out_dir.glob("*_phot.csv"))) == len(FRAME_STAMPS) + 1


def test_process_night_summary_columns_and_values(night_run):
    result, _, _ = night_run
    summary = result["summary"]

    assert list(summary.columns) == [
        "filename", "OBSTIME", "exposure", "sun_alt", "moon_alt", "moon_az",
        "allsky_mv_zenith", "allsky_mv_tucson", "allsky_mv_nogales",
        "allsky_mv_best", "best_az", "best_alt"]
    assert len(summary) == len(FRAME_STAMPS)
    assert summary["OBSTIME"].is_monotonic_increasing
    assert (summary["exposure"] == 20.0).all()
    assert (summary["sun_alt"] < -12.0).all()

    # A clear dark frame sits near 21.5 mag/arcsec^2 at the zenith, and the two
    # low-altitude light domes are brighter (numerically smaller) than that.
    zenith = summary["allsky_mv_zenith"]
    assert ((zenith > 20.5) & (zenith < 22.5)).all()
    assert (summary["allsky_mv_tucson"] < zenith).all()
    assert (summary["allsky_mv_nogales"] < zenith).all()

    # The zenith is one of the darkest-cone candidates, so the darkest patch is
    # always at least as dark as the zenith -- and it must come from above the
    # altitude floor. This is the invariant that makes allsky_mv_best a valid
    # darkness measure when the Milky Way is sitting on the zenith.
    best = summary["allsky_mv_best"]
    assert (best >= zenith - 1e-9).all()
    assert ((best > 20.5) & (best < 22.5)).all()
    assert (summary["best_alt"] >= ALCOR_SB_BEST_MIN_ALTITUDE - 1e-9).all()
    assert ((summary["best_az"] >= 0) & (summary["best_az"] < 360)).all()


def test_process_night_keogram_matches_the_summary(night_run):
    result, _, _ = night_run
    keogram = result["keogram"]

    assert keogram.shape == (FRAME_SHAPE[1], len(FRAME_STAMPS))
    assert keogram.dtype == np.float32
    assert np.isfinite(keogram).any()
    # the horizon mask blanks the ground, so a full column is never all-finite
    assert np.isnan(keogram).any()

    loaded, timestamps = load_alcor_sb_keogram_fits(result["keogram_file"])
    np.testing.assert_allclose(loaded, keogram, equal_nan=True)
    assert list(timestamps) == list(result["timestamps"])
    with fits.open(result["keogram_file"]) as hdul:
        assert hdul[0].header["BUNIT"] == "mag/arcsec2"


def test_process_night_median_stack_tracks_raw_pixels(night_run):
    result, _, _ = night_run
    with fits.open(result["median_file"]) as hdul:
        median = np.asarray(hdul[0].data)
        header = hdul[0].header

    assert median.shape == FRAME_SHAPE
    assert header["NSTACK"] == len(FRAME_STAMPS)
    for channel in "RGB":
        assert header[f"NBAD{channel}"] > 0

    # The stack must be built from RAW frames: repaired copies of the same frame
    # would median to the repaired values, hiding the very pixels it tracks.
    raw = np.asarray(fits.getdata(TEST_FITS), dtype=np.float32)
    repaired, _, mask = load_alcor_fits(TEST_FITS, badpix="repair")
    np.testing.assert_allclose(median, raw)
    assert mask is not None and mask.any()
    assert not np.allclose(median[mask], repaired[mask])


def test_photometry_is_done_only_for_a_non_empty_existing_csv(tmp_path):
    missing = tmp_path / "missing_phot.csv"
    assert not _photometry_is_done(missing, reprocess=False)

    empty = tmp_path / "empty_phot.csv"
    empty.touch()
    assert not _photometry_is_done(empty, reprocess=False)

    written = tmp_path / "written_phot.csv"
    written.write_text("name,flux_g\nVega,1.0\n")
    assert _photometry_is_done(written, reprocess=False)
    assert not _photometry_is_done(written, reprocess=True)


def test_process_night_reuses_existing_photometry(tmp_path):
    # The eventual real-time ingest writes <frame>_phot.csv as images arrive; a
    # later night run must not overwrite those, but must still produce complete
    # sky-brightness and keogram products, which need the frame's pixels.
    night_dir = _make_night(tmp_path / "2024-09-04")
    out_dir = tmp_path / "out"
    first = alcor_process_night(night_dir, out_dir=out_dir, workers=1)

    per_frame = sorted(out_dir.glob("2024_09_04__*_phot.csv"))
    assert len(per_frame) == len(FRAME_STAMPS)
    sentinel = per_frame[0]
    sentinel.write_text("name,flux_g\nSENTINEL,1.0\n")
    stamped = sentinel.stat().st_mtime_ns

    second = alcor_process_night(night_dir, out_dir=out_dir, workers=1)

    # the existing CSV is left exactly as it was ...
    assert sentinel.stat().st_mtime_ns == stamped
    assert "SENTINEL" in sentinel.read_text()
    # ... while every pixel-derived product is still complete and unchanged
    np.testing.assert_allclose(second["summary"]["allsky_mv_zenith"].to_numpy(),
                               first["summary"]["allsky_mv_zenith"].to_numpy())
    np.testing.assert_allclose(second["keogram"], first["keogram"], equal_nan=True)

    # --reprocess overwrites the stale CSV
    third = alcor_process_night(night_dir, out_dir=out_dir, workers=1,
                                reprocess=True)
    assert "SENTINEL" not in sentinel.read_text()
    np.testing.assert_allclose(third["keogram"], first["keogram"], equal_nan=True)


def test_process_night_day_keogram_covers_daylight_frames(tmp_path):
    # A day directory spans local noon to the next morning. The night frames'
    # columns are reused from the main pass; the daylight ones are loaded by the
    # second pass, and both end up in one time-ordered keogram.
    night_dir = _make_night(tmp_path / "2024-09-04")
    shutil.copy(TEST_FITS, night_dir / "2024_09_04__13_00_00.fits.bz2")   # daylight
    out_dir = tmp_path / "out"

    result = alcor_process_night(night_dir, out_dir=out_dir, workers=1,
                                 day_keogram=True)

    day = result["day_keogram"]
    assert day.shape == (FRAME_SHAPE[1], len(FRAME_STAMPS) + 1, 3)
    assert day.dtype == np.float32
    assert np.isfinite(day).all()
    assert result["day_keogram_file"].exists()
    assert result["day_keogram_plot"].exists()

    # the daylight frame sorts first, ahead of the two night frames
    loaded, timestamps = load_alcor_keogram_fits(result["day_keogram_file"])
    assert timestamps == sorted(timestamps)
    assert timestamps[0].startswith("2024-09-04T20:00")
    np.testing.assert_allclose(loaded, day)

    # the night half of the day keogram is the same zenith column the
    # sky-brightness keogram samples, so the two have equal height
    assert day.shape[0] == result["keogram"].shape[0]

    # the day keogram is off unless asked for
    plain = alcor_process_night(night_dir, out_dir=tmp_path / "out2", workers=1)
    assert plain["day_keogram"] is None
    assert plain["day_keogram_file"] is None


def test_wcs_puts_north_at_increasing_y():
    # The premise the keogram y axis rests on: along the zenith column, the high
    # row index is north. If this ever flips, the S/Z/N labels flip with it.
    wcs = build_alcor_wcs()
    zx, zy = wcs.world_to_pixel_values(0.0, 90.0)
    az_high, _ = wcs.pixel_to_world_values(float(zx), float(zy) + 200)
    az_low, _ = wcs.pixel_to_world_values(float(zx), float(zy) - 200)
    assert float(az_high) % 360 < 10 or float(az_high) % 360 > 350
    assert 170 < float(az_low) % 360 < 190


@pytest.mark.parametrize("edges", [False, True])
def test_keogram_yaxis_is_north_up(edges):
    # Guards both keogram plotters: row 0 is the south end, so the axis must run
    # bottom-to-top as S/Z/N. Inverting it renders the keogram upside down.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _set_keogram_yaxis(ax, 100, edges=edges)
    bottom, top = ax.get_ylim()
    assert top > bottom, "the y axis must not be inverted"

    ticks = np.asarray(ax.get_yticks())
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels[int(np.argmax(ticks))] == "N"
    assert labels[int(np.argmin(ticks))] == "S"
    assert ticks.max() == (100 if edges else 99)
    plt.close(fig)


def test_sb_keogram_fits_roundtrip_preserves_nans(tmp_path):
    keogram = np.array([[21.0, np.nan], [20.0, 19.5]], dtype=np.float32)
    timestamps = ["2024-09-05T06:51:09.000", "2024-09-05T06:55:09.000"]
    path = save_alcor_sb_keogram_fits(keogram, timestamps,
                                      tmp_path / "sb_keogram.fits")

    loaded, loaded_timestamps = load_alcor_sb_keogram_fits(path)
    np.testing.assert_allclose(loaded, keogram, equal_nan=True)
    assert list(loaded_timestamps) == timestamps


def test_process_night_parallel_matches_serial(tmp_path):
    # The worker pool gets the cones and horizon mask through an initializer and
    # writes the stack memmap from several processes; check that path agrees
    # with the serial one it shortcuts.
    night_dir = _make_night(tmp_path / "2024-09-04")
    serial = alcor_process_night(night_dir, out_dir=tmp_path / "serial",
                                 workers=1, median_stack=True,
                                 scratch_dir=str(tmp_path))
    parallel = alcor_process_night(night_dir, out_dir=tmp_path / "parallel",
                                   workers=2, median_stack=True,
                                   scratch_dir=str(tmp_path))

    assert parallel["errors"] == []
    np.testing.assert_allclose(parallel["keogram"], serial["keogram"],
                               equal_nan=True)
    for name in ALCOR_SB_TARGETS:
        np.testing.assert_allclose(parallel["summary"][name].to_numpy(),
                                   serial["summary"][name].to_numpy())
    np.testing.assert_allclose(fits.getdata(parallel["median_file"]),
                               fits.getdata(serial["median_file"]))


def test_process_night_rejects_a_night_with_no_dark_frames(tmp_path):
    night_dir = _make_night(tmp_path / "2024-09-04")
    with pytest.raises(ValueError, match="Sun below"):
        alcor_process_night(night_dir, out_dir=tmp_path / "out",
                            sun_alt_max=-70.0, workers=1)


def test_process_night_requires_matching_files(tmp_path):
    night_dir = _make_night(tmp_path / "2024-09-04")
    with pytest.raises(FileNotFoundError):
        alcor_process_night(night_dir, out_dir=tmp_path / "out",
                            pattern="*.fits.gz", workers=1)


def test_keogram_row_altitude_spans_below_the_horizon():
    """The keogram column runs past alt 0 at BOTH ends, inside the lit field."""
    from skycam_utils.alcor import (_keogram_row_altitude, alcor_calibration,
                                    build_alcor_wcs)

    cal = alcor_calibration(Time("2026-06-09"))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])
    zx, zy = wcs.world_to_pixel_values(0.0, 90.0)
    nrows = 1411
    alt = _keogram_row_altitude(wcs, nrows, int(round(float(zx))))

    assert alt.shape == (nrows,)
    assert alt[0] < 0 and alt[-1] < 0          # both ends below the horizon
    assert alt.max() > 88                       # and the zenith in between
    # the whole column is inside the illuminated field, so nothing is clipped
    assert max(float(zy), nrows - 1 - float(zy)) < cal["horizon_radius"]


def test_keogram_horizon_rows_finds_both_crossings():
    from skycam_utils.alcor import _keogram_horizon_rows

    alt = np.concatenate([np.linspace(-5, -0.1, 10),
                          np.linspace(0.1, 89, 30),
                          np.linspace(89, -4, 20)])
    rows = _keogram_horizon_rows(alt)
    assert len(rows) == 2
    assert rows[0] < rows[1]
    assert _keogram_horizon_rows(None) == []
    assert _keogram_horizon_rows(np.full(10, -3.0)) == []


def test_sb_keogram_scale_uses_sky_rows_only():
    """Bright below-horizon rows must not drag the colour scale."""
    from skycam_utils.alcor import _sb_keogram_limits

    alt = np.concatenate([np.full(10, -3.0), np.full(20, 45.0), np.full(10, -3.0)])
    keo = np.full((40, 12), 21.5)
    keo[alt < 0] = 17.0                      # terrain / light domes, much brighter

    vmin, vmax = _sb_keogram_limits(keo, alt, None, None)
    assert 21.0 < vmin <= vmax < 22.0        # scaled on sky, not on terrain

    # without altitudes the terrain drags the range wide open, which is the
    # behaviour older keograms (no ROWALT extension) keep
    wide_min, wide_max = _sb_keogram_limits(keo, None, None, None)
    assert wide_min < 18.0

    # explicit limits always win
    assert _sb_keogram_limits(keo, alt, 19.0, 22.0) == (19.0, 22.0)


def test_keogram_fits_round_trips_row_altitude(tmp_path):
    from skycam_utils.alcor import (save_alcor_sb_keogram_fits,
                                    save_alcor_keogram_fits, _load_row_altitude)

    alt = np.linspace(-6.0, -5.0, 25)
    times = [(Time("2026-06-09T04:00:00") + i * u.min).isot for i in range(4)]

    sb = tmp_path / "sb_keogram.fits"
    save_alcor_sb_keogram_fits(np.zeros((25, 4)), times, sb, altitude=alt)
    np.testing.assert_allclose(_load_row_altitude(sb), alt, rtol=1e-6)

    rgb = tmp_path / "keogram.fits"
    save_alcor_keogram_fits(np.zeros((25, 4, 3)), times, rgb, altitude=alt)
    np.testing.assert_allclose(_load_row_altitude(rgb), alt, rtol=1e-6)

    # a file written without it stays loadable
    plain = tmp_path / "plain.fits"
    save_alcor_sb_keogram_fits(np.zeros((25, 4)), times, plain)
    assert _load_row_altitude(plain) is None
