# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import (
    ALCOR_EXT_FRAME_SHAPE,
    ALCOR_EXT_KERNEL_SIGMA,
    _resolve_column,
    alcor_calibration,
    alcor_extinction_at,
    alcor_extinction_fits,
    alcor_extinction_grid,
    alcor_extinction_map,
    alcor_extinction_stars,
    build_alcor_wcs,
    plot_alcor_extinction_fits,
)

BLOB_AZ, BLOB_ALT, BLOB_EXT, BLOB_SIGMA = 120.0, 50.0, 1.0, 10.0


def _wcs():
    cal = alcor_calibration(None)
    return build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                           rotation=cal["rotation"],
                           radial_coeffs=cal["radial_coeffs"],
                           horizon_radius=cal["horizon_radius"],
                           tangential_coeffs=cal["tangential_coeffs"],
                           axis_tilt=cal["axis_tilt"])


def _separation(az1, alt1, az2, alt2):
    def vec(az, alt):
        az, alt = np.radians(az), np.radians(alt)
        return np.array([np.cos(alt) * np.cos(az),
                         np.cos(alt) * np.sin(az),
                         np.sin(alt)])
    return np.degrees(np.arccos(np.clip(vec(az1, alt1) @ vec(az2, alt2),
                                        -1.0, 1.0)))


def _synthetic_frames(nframes=10, nstars=400, lost_names=(), seed=1):
    """
    A synthetic night block: stars spread over the dome, each carrying a
    Gaussian extinction blob centred on (BLOB_AZ, BLOB_ALT), measured in every
    frame. Stars named in ``lost_names`` are never detected.
    """
    rng = np.random.default_rng(seed)
    az = rng.uniform(0.0, 360.0, nstars)
    # uniform on the cap, so stars are not piled up at the zenith
    alt = np.degrees(np.arcsin(rng.uniform(np.sin(np.radians(21.0)), 1.0,
                                           nstars)))
    names = np.array([f"star{i:04d}" for i in range(nstars)])

    rows = []
    start = pd.Timestamp("2025-01-02T05:00:00")
    for frame in range(nframes):
        sep = np.array([_separation(a, h, BLOB_AZ, BLOB_ALT)
                        for a, h in zip(az, alt)])
        ext = BLOB_EXT * np.exp(-0.5 * (sep / BLOB_SIGMA) ** 2)
        lost = np.isin(names, list(lost_names))
        flux = np.where(lost, 0.0, 1000.0)
        rows.append(pd.DataFrame({
            "name": names,
            "OBSTIME": start + pd.Timedelta(seconds=30 * frame),
            "altitude": alt,
            "azimuth": az,
            "variable": False,
            "mag_g_ap": -10.2,
            "ext_g_ap": np.where(lost, np.nan, ext),
            "flux_g_ap": flux,
            "sat_g_ap": False,
        }))
    return pd.concat(rows, ignore_index=True)


def test_extinction_stars_reduces_a_block_to_one_row_per_star():
    df = _synthetic_frames(nframes=10, nstars=50)
    measured, lost = alcor_extinction_stars(df)
    assert len(measured) == 50
    assert not len(lost)
    assert (measured["n"] == 10).all()
    assert {"az", "alt", "ext", "n"} <= set(measured.columns)


def test_extinction_stars_keeps_undetected_stars_as_lower_limits():
    """
    A star behind thick cloud has flux 0 and no ext value at all. It must not
    silently vanish -- it carries the strongest signal in the frame.
    """
    df = _synthetic_frames(nframes=10, nstars=60, lost_names=["star0000",
                                                              "star0001"])
    measured, lost = alcor_extinction_stars(df)
    assert set(lost.index) == {"star0000", "star0001"}
    assert not set(lost.index) & set(measured.index)
    assert "ext" not in lost.columns          # a limit, not a measurement


def test_extinction_stars_excludes_variables_and_saturated():
    df = _synthetic_frames(nframes=10, nstars=40)
    df.loc[df["name"] == "star0000", "variable"] = True
    df.loc[df["name"] == "star0001", "sat_g_ap"] = True
    measured, _ = alcor_extinction_stars(df)
    assert "star0000" not in measured.index
    assert "star0001" not in measured.index


def test_extinction_stars_applies_the_magnitude_window():
    df = _synthetic_frames(nframes=10, nstars=40)
    df.loc[df["name"] == "star0000", "mag_g_ap"] = -12.0   # non-linear regime
    df.loc[df["name"] == "star0001", "mag_g_ap"] = -8.0    # faint-end bias
    measured, _ = alcor_extinction_stars(df)
    assert "star0000" not in measured.index
    assert "star0001" not in measured.index


def test_resolve_column_accepts_both_schemas():
    both = pd.DataFrame(columns=["ext_g_ap", "ext_g_gauss"])
    plain = pd.DataFrame(columns=["ext_g"])
    assert _resolve_column(both, "ext", "g", "ap") == "ext_g_ap"
    assert _resolve_column(both, "ext", "g", "gauss") == "ext_g_gauss"
    assert _resolve_column(plain, "ext", "g", "ap") == "ext_g"
    with pytest.raises(KeyError):
        _resolve_column(plain, "ext", "r", "ap")


def test_extinction_grid_recovers_a_synthetic_blob():
    """
    The kernel average must put the peak where the blob is. Smoothing with an
    8 deg kernel broadens a 10 deg blob, so the recovered amplitude is lower
    than the input -- the position is the thing being tested here.
    """
    df = _synthetic_frames(nframes=10, nstars=600)
    measured, _ = alcor_extinction_stars(df)
    grid, az_values, alt_values = alcor_extinction_grid(measured)

    peak = np.unravel_index(np.nanargmax(grid), grid.shape)
    peak_az = az_values[peak[1]]
    peak_alt = alt_values[peak[0]]
    assert _separation(peak_az, peak_alt, BLOB_AZ, BLOB_ALT) < ALCOR_EXT_KERNEL_SIGMA

    # Far from the blob the map must return to zero.
    far = np.nanmin(grid)
    assert far < 0.05
    assert 0.2 < np.nanmax(grid) < BLOB_EXT


def test_extinction_grid_blanks_where_no_stars_are_near():
    """
    Blanking rather than extrapolating is the point: an unsampled region is the
    one most likely to be the most extinguished.
    """
    df = _synthetic_frames(nframes=10, nstars=300)
    measured, _ = alcor_extinction_stars(df)
    # Delete every star in a wide patch, then require the map to blank there.
    keep = np.array([_separation(a, h, 300.0, 60.0) > 30.0
                     for a, h in zip(measured["az"], measured["alt"])])
    grid, az_values, alt_values = alcor_extinction_grid(measured[keep])
    ai = int(np.argmin(np.abs(az_values - 300.0)))
    hi = int(np.argmin(np.abs(alt_values - 60.0)))
    assert not np.isfinite(grid[hi, ai])


def test_extinction_grid_survives_the_azimuth_seam():
    """A blob straddling 0/360 must not be split by the grid's wrap."""
    df = _synthetic_frames(nframes=6, nstars=500)
    measured, _ = alcor_extinction_stars(df)
    measured["ext"] = [
        np.exp(-0.5 * (_separation(a, h, 0.0, 45.0) / 10.0) ** 2)
        for a, h in zip(measured["az"], measured["alt"])
    ]
    grid, az_values, alt_values = alcor_extinction_grid(measured)
    hi = int(np.argmin(np.abs(alt_values - 45.0)))
    left = grid[hi, int(np.argmin(np.abs(az_values - 355.0)))]
    right = grid[hi, int(np.argmin(np.abs(az_values - 5.0)))]
    assert abs(left - right) < 0.1        # continuous across the seam


def test_extinction_map_is_on_the_raw_frame_and_blanks_low_altitude():
    df = _synthetic_frames(nframes=6, nstars=400)
    measured, _ = alcor_extinction_stars(df)
    ext_map, wcs, grid = alcor_extinction_map(measured)

    assert ext_map.shape == ALCOR_EXT_FRAME_SHAPE
    assert ext_map.dtype == np.float32
    assert np.isfinite(ext_map).any()

    # The corners of the sensor are far outside the mapped altitude range.
    assert not np.isfinite(ext_map[0, 0])
    # The zenith pixel is inside it.
    x, y = wcs.world_to_pixel_values(0.0, 90.0)
    assert np.isfinite(ext_map[int(round(float(y))), int(round(float(x)))])


def test_extinction_fits_round_trips_through_the_wcs(tmp_path):
    """
    The whole point of the raw-frame + ARC WCS choice: a pointing goes straight
    from (az, alt) to a value with one world_to_pixel call, and the value must
    match the grid the map was built from.
    """
    df = _synthetic_frames(nframes=10, nstars=600)
    out = tmp_path / "ext.fits"
    path, ext_map, measured, lost = alcor_extinction_fits(
        df, output_file=out, return_products=True)
    assert path.exists()

    peak = alcor_extinction_at(out, BLOB_AZ, BLOB_ALT)
    edge = alcor_extinction_at(out, BLOB_AZ + 180.0, BLOB_ALT)
    assert peak > edge + 0.2
    assert np.isfinite(peak)

    # Below the mapped altitude there is nothing to report.
    assert not np.isfinite(alcor_extinction_at(out, 0.0, 5.0))


def test_extinction_at_radius_is_in_degrees(tmp_path):
    """
    A degrees-vs-pixels mix-up here would be invisible on a smooth map, so pin
    it: a 1 deg cone and a 20 deg cone must differ on a 10 deg blob, and a 20
    deg cone must pull the peak value down toward the surroundings.
    """
    df = _synthetic_frames(nframes=8, nstars=600)
    out = tmp_path / "ext.fits"
    alcor_extinction_fits(df, output_file=out)

    single = alcor_extinction_at(out, BLOB_AZ, BLOB_ALT, radius=0.0)
    small = alcor_extinction_at(out, BLOB_AZ, BLOB_ALT, radius=1.0)
    wide = alcor_extinction_at(out, BLOB_AZ, BLOB_ALT, radius=20.0)

    assert abs(single - small) < 0.02      # 1 deg changes almost nothing
    assert wide < small - 0.05             # 20 deg averages the blob away


def test_extinction_fits_header_and_stars_table(tmp_path):
    df = _synthetic_frames(nframes=10, nstars=200, lost_names=["star0000"])
    out = tmp_path / "ext.fits"
    alcor_extinction_fits(df, output_file=out)

    with fits.open(out) as hdul:
        header = hdul[0].header
        rows = hdul["STARS"].data
        names = list(rows["name"])
        detected = np.asarray(rows["detected"])

    assert header["BUNIT"] == "mag"
    assert header["NFRAMES"] == 10
    assert header["EXTBAND"] == "g"
    assert header["EXTMETH"] == "ap"
    assert header["KERNSIG"] == ALCOR_EXT_KERNEL_SIGMA
    assert header["EXTFLAT"] == "none"     # reserved, not yet applied
    assert header["PIXSCALE"] > 0
    assert "TSTART" in header and "TEND" in header and "TMID" in header
    assert header["NSTARS"] == int(detected.sum())
    assert header["NLOST"] == int((~detected).sum())

    # The lost star must be in the table, flagged, with no extinction value.
    assert "star0000" in names
    lost_ext = np.asarray(rows["ext"])[~detected]
    assert not np.isfinite(lost_ext).any()


def test_extinction_fits_refuses_frames_with_no_usable_stars(tmp_path):
    df = _synthetic_frames(nframes=4, nstars=20)
    df["ext_g_ap"] = np.nan
    with pytest.raises(ValueError, match="no usable star measurements"):
        alcor_extinction_fits(df, output_file=tmp_path / "ext.fits")


def test_plot_alcor_extinction_fits_replots_from_the_stars_table(tmp_path):
    df = _synthetic_frames(nframes=8, nstars=300, lost_names=["star0000"])
    out = tmp_path / "ext.fits"
    alcor_extinction_fits(df, output_file=out)

    png = plot_alcor_extinction_fits(out)
    assert Path(png) == tmp_path / "ext.png"
    assert Path(png).exists() and Path(png).stat().st_size > 0
