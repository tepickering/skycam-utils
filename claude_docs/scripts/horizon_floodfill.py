#!/usr/bin/env python
"""Diagnostic figures for the Alcor horizon mask.

The mask algorithm now lives in the package
(``skycam_utils.alcor.build_alcor_horizon_mask`` + ``_alcor_undetected_fraction``);
regenerate the shipped mask with the ``create_horizon_mask`` CLI. This script
only re-renders the diagnostic figure used in the docs from local data.
Not part of the installed package.
"""
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skycam_utils.alcor import (alcor_calibration, build_alcor_wcs,
                                build_alcor_horizon_mask,
                                _alcor_undetected_fraction)

GP = Path(__file__).resolve().parent.parent / "gplots"
MEDIAN = GP / "2026-02-18_median.fits"
EPOCH = "2026-02-18"
SEC_LO, SEC_HI = 225.0, 270.0

# nights whose per-frame photometry feeds the undetected-star patch
NIGHTS = [
    os.path.expanduser("~/MMT/skycam_data/2024-09-04"),
    "/Volumes/Samsung_4TB/skycam/2026-01-11",
    "/Volumes/Samsung_4TB/skycam/2026-03-11",
    "/Volumes/Samsung_4TB/skycam/2026-05-18",
    "/Volumes/Samsung_4TB/skycam/2026-06-09",
]


def main():
    cal = alcor_calibration(Time(f"{EPOCH}T12:00:00"))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"], radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])
    img = fits.getdata(MEDIAN).astype(float)
    ny, nx = img.shape

    undetected = _alcor_undetected_fraction(NIGHTS, wcs)
    horizon_mask = build_alcor_horizon_mask(
        img, wcs, undetected=undetected, sector=(SEC_LO, SEC_HI))

    vmin, vmax = ZScaleInterval().get_limits(img)
    fig, ax = plt.subplots(1, 1, figsize=(11, 10))
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ov = np.zeros((ny, nx, 4))
    ov[horizon_mask] = (1, 0, 0, 0.32)
    ax.imshow(ov, origin="lower")
    for a in (SEC_LO, SEC_HI):
        ra = np.linspace(0.5, 25, 60)
        rx, ry = wcs.world_to_pixel_values(np.full_like(ra, a), ra)
        ax.plot(rx, ry, ":", color="cyan", lw=1.0, alpha=0.8)
    ax.set_title("complete horizon mask (red=not-sky) on 2026-02-18 median; "
                 "dotted = SW->W undetected sector")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = GP / "horizon_floodfill.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
