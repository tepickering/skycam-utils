"""Apply the illuminated-field cut to already-written SB keograms.

ALCOR_FIELD_RADIUS was added after the nights were processed. The cut only
blanks pixels -- it changes no value that stays -- so re-running the whole night
would reproduce exactly this, and re-rendering from the saved FITS is equivalent
and ~1000x cheaper. sky_brightness.csv and the median stacks are unaffected: the
cones sit far inside the field.
"""
import sys
import numpy as np
from astropy.io import fits
from astropy.time import Time
from pathlib import Path
from skycam_utils.alcor import (ALCOR_FIELD_RADIUS, alcor_calibration,
                                build_alcor_wcs, plot_alcor_sb_keogram_fits)

for night_dir in sys.argv[1:]:
    night_dir = Path(night_dir)
    night = night_dir.name
    path = night_dir / f"{night}_sb_keogram.fits"
    if not path.exists():
        print(f"{night}: no sb keogram, skipped")
        continue

    cal = alcor_calibration(Time(night))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])
    zx, _ = wcs.world_to_pixel_values(0.0, 90.0)
    zcol = int(round(float(zx)))
    ax, ay = (float(c) - 1.0 for c in wcs.wcs.crpix[:2])

    with fits.open(path, mode="update") as hdul:
        data = hdul[0].data
        rows = np.arange(data.shape[0], dtype=float)
        radius = np.hypot(zcol - ax, rows - ay)
        outside = radius > ALCOR_FIELD_RADIUS
        was = int(np.isfinite(data).any(axis=1).sum())
        data[outside] = np.nan
        hdul[0].header["FIELDRAD"] = (ALCOR_FIELD_RADIUS,
                                      "illuminated field radius (px)")
        hdul.flush()
        now = int(np.isfinite(data).any(axis=1).sum())

    png = plot_alcor_sb_keogram_fits(path)
    print(f"{night}: blanked {int(outside.sum())} rows outside r={ALCOR_FIELD_RADIUS} "
          f"({was} -> {now} live rows); re-rendered {png}")
