"""Overlay undetected catalog-star positions on the night luminance co-add.

A new probe for the Alcor horizon map: instead of inferring the obstruction
boundary from the star-trail / coverage field (which follows sky structure, not
the building), use the catalog stars themselves. `alcor_star_photometry` predicts
every bright star's raw pixel from the WCS and measures fixed-position photometry
down to alt 0; a star is "undetected" when its aperture flux is 0 (blocked by the
building/terrain, below the horizon, or lost in cloud). Accumulated over a whole
night, the undetected positions trace the obstruction silhouette directly, because
the building blocks each star as its arc sweeps behind the roofline.

All nights' non-detections are combined onto one co-add. The WCS is stable to
~1 px across the 2024 and 2026 epochs, so the raw pixel positions are stacked
directly (no reprojection). For definition the points are rendered as a smoothed
2D-density heatmap with a low threshold, which sharpens the obstruction edges and
suppresses the sparse interior speckle (faint-limit / transient-cloud misses).

In the raw frame as displayed (origin="lower"): N up, E left, S down, W right.

Writes one figure to ../gplots. Not part of the installed package.
"""
import glob
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import (AsinhStretch, ImageNormalize,
                                    PercentileInterval)
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from skycam_utils import alcor

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "gplots"))

NIGHTS = [
    ("2024-09-04", os.path.expanduser("~/MMT/skycam_data/2024-09-04")),
    ("2026-01-11", "/Volumes/Samsung_4TB/skycam/2026-01-11"),
    ("2026-03-11", "/Volumes/Samsung_4TB/skycam/2026-03-11"),
    ("2026-05-18", "/Volumes/Samsung_4TB/skycam/2026-05-18"),
    ("2026-06-09", "/Volumes/Samsung_4TB/skycam/2026-06-09"),
]
TARGET = "2026-05-18"   # co-add to display (WCS ~1 px stable between epochs)

USECOLS = ["xcen", "ycen", "flux_g_ap"]

# ---- tunables ----
BIN_PX = 2.5      # density bin size, native px
SMOOTH = 1.0      # gaussian smoothing of the density, in bins
THR = 2.0         # mask density below this (counts) -> kills interior speckle


def collect_undetected(night_dir):
    """WCS-predicted pixel positions of undetected stars (G aperture flux <= 0)."""
    frames = sorted(glob.glob(os.path.join(night_dir, "*_phot.csv")))
    parts = []
    for f in frames:
        try:
            parts.append(pd.read_csv(f, usecols=USECOLS))
        except Exception as exc:
            print(f"  skip {os.path.basename(f)}: {exc}")
    df = pd.concat(parts, ignore_index=True)
    undet = df["flux_g_ap"] <= 0
    print(f"  {len(frames)} frames, {len(df)} measurements, "
          f"{undet.sum()} undetected ({100 * undet.mean():.1f}%)")
    sub = df.loc[undet, ["xcen", "ycen"]]
    return sub["xcen"].to_numpy(), sub["ycen"].to_numpy()


def main():
    target_dir = dict(NIGHTS)[TARGET]
    coadd = fits.getdata(f"{target_dir}/{TARGET}_night_lum_coadd.fits").astype(float)
    ny, nx = coadd.shape

    cal = alcor.alcor_calibration(Time(f"{TARGET}T12:00:00", scale="utc"))
    wcs = alcor.build_alcor_wcs(
        xcen=cal["xcen"], ycen=cal["ycen"], rotation=cal["rotation"],
        radial_coeffs=cal["radial_coeffs"], horizon_radius=cal["horizon_radius"],
        tangential_coeffs=cal["tangential_coeffs"], axis_tilt=cal["axis_tilt"])

    yy, xx = np.mgrid[0:ny, 0:nx]
    world = wcs.all_pix2world(np.column_stack([xx.ravel(), yy.ravel()]), 0)
    alt = world[:, 1].reshape(ny, nx)

    xs_all, ys_all = [], []
    for night, ddir in NIGHTS:
        print(night)
        ux, uy = collect_undetected(ddir)
        xs_all.append(ux)
        ys_all.append(uy)
    ux = np.concatenate(xs_all)
    uy = np.concatenate(ys_all)
    print(f"combined undetected positions: {len(ux)}")

    zp = wcs.all_world2pix([[0.0, 90.0]], 0)[0]
    half = int(cal["horizon_radius"] * 1.02)
    x0, y0 = int(round(zp[0])), int(round(zp[1]))
    xs, ys = max(0, x0 - half), max(0, y0 - half)
    xe, ye = min(nx, x0 + half), min(ny, y0 + half)
    sub, al = coadd[ys:ye, xs:xe], alt[ys:ye, xs:xe]
    w, h = xe - xs, ye - ys

    nbx, nby = int(w / BIN_PX), int(h / BIN_PX)
    H, _, _ = np.histogram2d(ux, uy, bins=[nbx, nby],
                             range=[[xs, xe], [ys, ye]])
    H = ndi.gaussian_filter(H, SMOOTH)
    dens = np.ma.masked_less(H.T, THR)   # transpose: histogram2d is [x, y]

    fig, ax = plt.subplots(figsize=(13, 13))
    norm = ImageNormalize(sub, interval=PercentileInterval(99.3),
                          stretch=AsinhStretch(a=0.02))
    ax.imshow(sub, origin="lower", cmap="gray", norm=norm,
              extent=[0, w, 0, h])
    im = ax.imshow(dens, origin="lower", cmap="inferno", alpha=0.78,
                   norm=PowerNorm(0.5), extent=[0, w, 0, h])
    ax.contour(al, levels=[0, 10, 20], colors="cyan", linewidths=0.4, alpha=0.6)
    ax.contour(al, levels=[25], colors="yellow", linewidths=0.8, alpha=0.8)
    for fx, fy, lab in [(0.5, 0.985, "N"), (0.5, 0.015, "S"),
                        (0.015, 0.5, "E"), (0.985, 0.5, "W")]:
        ax.text(fx, fy, lab, transform=ax.transAxes, color="yellow",
                fontsize=15, ha="center", va="center", weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("undetected-star density (both nights)")
    nights_label = " + ".join(n for n, _ in NIGHTS)
    ax.set_title(f"Undetected catalog stars, {nights_label} combined "
                 f"({len(ux)} positions)  on {TARGET} co-add")
    ax.set_xticks([])
    ax.set_yticks([])
    out = f"{OUT}/alcor_undetected_stars.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
