# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import (
    ALCOR_CALIB_EXPTIME,
    _alcor_pixel_solid_angle,
    _read_frame_exposure,
    alcor_sky_brightness_fits,
    load_alcor_fits,
    plot_alcor_sky_brightness,
)

TEST_FITS = Path(__file__).with_name("test.fits.bz2")


def _flat_altaz_grid(cdelt_deg, lat0_deg=0.0, n=21):
    """(az, alt) grids for a regular ``cdelt_deg`` patch centred at lat0."""
    offset = (np.arange(n) - n // 2) * cdelt_deg
    az = np.broadcast_to(offset[None, :], (n, n))
    alt = np.broadcast_to(lat0_deg + offset[:, None], (n, n))
    return np.array(az), np.array(alt)


def test_pixel_solid_angle_flat_field_matches_plate_scale():
    # At the equator a cdelt-spaced patch subtends (cdelt*3600)^2 arcsec^2/pixel.
    cdelt = 1.0e-3
    az, alt = _flat_altaz_grid(cdelt, lat0_deg=0.0)
    omega = _alcor_pixel_solid_angle(az, alt)
    expected = (cdelt * 3600.0) ** 2
    assert np.isclose(omega[10, 10], expected, rtol=1e-3)


def test_pixel_solid_angle_follows_cosine_latitude_law():
    # The solid angle is a real spherical element, so it scales as cos(lat).
    cdelt = 1.0e-3
    lat0 = 60.0
    az, alt = _flat_altaz_grid(cdelt, lat0_deg=lat0)
    omega = _alcor_pixel_solid_angle(az, alt)
    expected = (cdelt * 3600.0) ** 2 * np.cos(np.radians(lat0))
    assert np.isclose(omega[10, 10], expected, rtol=1e-3)


def test_read_frame_exposure_from_header_and_fallback(tmp_path):
    # the bundled frame carries EXPOSURE = 20 s
    assert _read_frame_exposure(TEST_FITS) == 20.0

    explicit = tmp_path / "exp10.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.int16))
    hdu.header["EXPOSURE"] = 10.0
    hdu.writeto(explicit)
    assert _read_frame_exposure(explicit) == 10.0

    missing = tmp_path / "noexp.fits"
    fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.int16)).writeto(missing)
    assert _read_frame_exposure(missing) == ALCOR_CALIB_EXPTIME

    nonpos = tmp_path / "zeroexp.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.int16))
    hdu.header["EXPOSURE"] = 0.0
    hdu.writeto(nonpos)
    assert _read_frame_exposure(nonpos) == ALCOR_CALIB_EXPTIME


def test_plot_alcor_sky_brightness_writes_figure_and_zenith(tmp_path):
    import matplotlib.pyplot as plt

    outfig = tmp_path / "skybright.pdf"
    fig = plot_alcor_sky_brightness(TEST_FITS, outfig=outfig, radius=600,
                                    figsize=3)
    try:
        assert outfig.exists() and outfig.stat().st_size > 0
        # image axes + colorbar axes + polar overlay
        assert len(fig.axes) == 3
        # the zenith annotation carries a dark-sky surface brightness
        texts = [t.get_text() for t in fig.texts if "zenith" in t.get_text()]
        assert texts, "expected a zenith-brightness annotation"
        match = re.search(r"=\s*([0-9.]+)", texts[0])
        assert match is not None
        zenith = float(match.group(1))
        assert 20.0 < zenith < 23.0          # MMT dark-sky V mag/arcsec^2
    finally:
        plt.close(fig)


def test_plot_alcor_sky_brightness_horizon_mask_runs(tmp_path):
    import matplotlib.pyplot as plt

    outfig = tmp_path / "skybright_hmask.png"
    fig = plot_alcor_sky_brightness(TEST_FITS, outfig=outfig, radius=600,
                                    figsize=3, horizon_mask=True)
    try:
        assert outfig.exists() and outfig.stat().st_size > 0
    finally:
        plt.close(fig)


def _zenith_altitudes(cube, wcs):
    """Full-frame altitude grid (deg) for the green channel of ``cube``."""
    ny, nx = cube[1].shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    _, alt = wcs.pixel_to_world_values(xx.astype(float), yy.astype(float))
    return alt


def test_alcor_sky_brightness_fits_writes_calibrated_map(tmp_path):
    input_file = tmp_path / "sample.fits.bz2"
    shutil.copyfile(TEST_FITS, input_file)

    out = alcor_sky_brightness_fits(input_file)

    assert out == tmp_path / "sample_sb.fits"
    assert out.exists()

    cube, wcs, _ = load_alcor_fits(input_file)
    with fits.open(out) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
        # full-frame, native-orientation 2-D float32 matching the G channel
        assert data.shape == cube[1].shape
        assert data.dtype.kind == "f"
        assert data.dtype.itemsize == np.dtype(np.float32).itemsize
        # calibrated surface-brightness units + provenance
        assert hdr["BUNIT"] == "mag/arcsec2"
        for key in ("ZP_G", "ZP_EPOCH", "EXPOSURE", "CALIBEXP", "SATLEVEL", "HORIZMSK"):
            assert key in hdr
        assert hdr["HORIZMSK"] is False
        # raw-frame alt/az WCS encoded as an ARC projection, CRPIX preserved
        assert hdr["CTYPE1"].startswith("RA---ARC")
        assert hdr["CTYPE2"].startswith("DEC--ARC")
        np.testing.assert_allclose(hdr["CRPIX1"], wcs.wcs.crpix[0])
        np.testing.assert_allclose(hdr["CRPIX2"], wcs.wcs.crpix[1])

        # off-frame pixels are blanked; on-sky pixels are finite
        assert np.isnan(data).any()
        assert np.isfinite(data).any()
        # dark-sky zenith cap lands in the expected V mag/arcsec^2 range
        alt = _zenith_altitudes(cube, wcs)
        cap = data[np.isfinite(data) & (alt > 85.0)]
        assert cap.size
        assert 20.0 < float(np.median(cap)) < 23.0


def test_alcor_sky_brightness_fits_horizon_mask_adds_blanks(tmp_path):
    input_file = tmp_path / "sample.fits.bz2"
    shutil.copyfile(TEST_FITS, input_file)

    plain = alcor_sky_brightness_fits(input_file, output_file=tmp_path / "plain.fits")
    masked = alcor_sky_brightness_fits(
        input_file, output_file=tmp_path / "masked.fits", horizon_mask=True
    )

    with fits.open(plain) as hdul:
        plain_nan = int(np.isnan(hdul[0].data).sum())
    with fits.open(masked) as hdul:
        masked_nan = int(np.isnan(hdul[0].data).sum())
        assert hdul[0].header["HORIZMSK"] is True

    # the horizon mask blanks the not-sky region on top of the default blanking
    assert masked_nan > plain_nan


def test_alcor_sky_brightness_fits_requires_overwrite(tmp_path):
    input_file = tmp_path / "sample.fits.bz2"
    shutil.copyfile(TEST_FITS, input_file)

    out = alcor_sky_brightness_fits(input_file)
    with pytest.raises(OSError):
        alcor_sky_brightness_fits(input_file, output_file=out)
    # overwrite succeeds
    assert alcor_sky_brightness_fits(input_file, output_file=out, overwrite=True) == out


def test_sb_map_blanks_outside_the_illuminated_field():
    """Beyond the image circle the fisheye delivers no light; that must be NaN."""
    from skycam_utils.alcor import (ALCOR_FIELD_RADIUS, _alcor_sky_brightness_map,
                                    alcor_calibration, build_alcor_wcs)

    cal = alcor_calibration(Time("2026-06-09"))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])
    ny, nx = 1411, 1422
    cube = np.full((3, ny, nx), 3000.0)
    # _corner_bias reads the four 10x10 image corners; leave them at a pedestal
    # so the bias-subtracted signal is positive and the map is finite on-field.
    for ys in (slice(0, 10), slice(ny - 10, ny)):
        for xs in (slice(0, 10), slice(nx - 10, nx)):
            cube[:, ys, xs] = 1900.0

    mu, _ = _alcor_sky_brightness_map(cube, wcs, Time("2026-06-09T08:00:00"), 20.0)
    kept, _ = _alcor_sky_brightness_map(cube, wcs, Time("2026-06-09T08:00:00"), 20.0,
                                        field_radius=None)

    ax, ay = (float(c) - 1.0 for c in wcs.wcs.crpix[:2])
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - ax, yy - ay)

    assert not np.isfinite(mu[r > ALCOR_FIELD_RADIUS + 1]).any()
    assert np.isfinite(mu[r < ALCOR_FIELD_RADIUS - 1]).any()
    # opting out restores pixels the field cut removed
    assert np.isfinite(kept[(r > ALCOR_FIELD_RADIUS + 1) & np.isfinite(kept)]).any()


def test_field_radius_sits_inside_the_nominal_horizon_radius():
    """The sensor reaches a couple of degrees below the horizon, no further."""
    from skycam_utils.alcor import (ALCOR_FIELD_RADIUS, ALCOR_HORIZON_RADIUS,
                                    alcor_calibration, build_alcor_wcs)

    assert ALCOR_FIELD_RADIUS < ALCOR_HORIZON_RADIUS

    cal = alcor_calibration(Time("2026-06-09"))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])
    _, alt = wcs.pixel_to_world_values(cal["xcen"],
                                       cal["ycen"] - ALCOR_FIELD_RADIUS)
    assert -4.0 < float(alt) < -1.0


def test_sb_map_keeps_saturated_pixels_by_default():
    """A blanked pixel renders as background, so the brightest sources would
    appear as dark holes. Keeping the (slightly too faint) value is the lesser
    error -- see ALCOR_SB_SATURATION."""
    from skycam_utils.alcor import (ALCOR_SB_SATURATION, _alcor_sky_brightness_map,
                                    alcor_calibration, build_alcor_wcs)

    cal = alcor_calibration(Time("2026-06-09"))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])
    ny, nx = 1411, 1422
    cube = np.full((3, ny, nx), 3000.0)
    for ys in (slice(0, 10), slice(ny - 10, ny)):
        for xs in (slice(0, 10), slice(nx - 10, nx)):
            cube[:, ys, xs] = 1900.0
    # a saturated core near the optical axis
    yc, xc = int(round(cal["ycen"])), int(round(cal["xcen"]))
    cube[1, yc, xc] = ALCOR_SB_SATURATION + 5000

    kept, _ = _alcor_sky_brightness_map(cube, wcs, Time("2026-06-09T08:00:00"), 20.0)
    masked, _ = _alcor_sky_brightness_map(cube, wcs, Time("2026-06-09T08:00:00"),
                                          20.0, saturation=ALCOR_SB_SATURATION)

    # default keeps it, and it reads BRIGHTER (lower mag) than the surround
    assert np.isfinite(kept[yc, xc])
    assert kept[yc, xc] < kept[yc, xc + 20]
    # opting in still blanks it
    assert np.isnan(masked[yc, xc])
    assert np.isfinite(masked[yc, xc + 20])


def test_sb_fits_header_records_that_nothing_was_masked():
    from skycam_utils.alcor import _alcor_sb_fits_header, alcor_calibration, build_alcor_wcs

    cal = alcor_calibration(Time("2026-06-09"))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])
    off = _alcor_sb_fits_header(wcs, Time("2026-06-09T08:00:00"), 20.0, None, False)
    on = _alcor_sb_fits_header(wcs, Time("2026-06-09T08:00:00"), 20.0, 25000, False)
    assert off["SATLEVEL"] == "none"
    assert on["SATLEVEL"] == 25000
