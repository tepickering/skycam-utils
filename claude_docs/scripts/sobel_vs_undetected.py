#!/usr/bin/env python
"""Compare the cloudy-night Sobel(log) edge map against the undetected-star
density probe, in a common raw-pixel crop.

The bet: undetected-star density flags BOTH real obstructions AND open sky too
bright for reliable detection; the Sobel edge map only fires on sharp obstruction
boundaries. So density present with no Sobel edge == bright open sky (a star-
method false positive that Sobel correctly ignores).

Renders three panels (undetected density, Sobel edges, Sobel edges as contours
over the density). Not part of the installed package.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import (AsinhStretch, ImageNormalize,
                                    PercentileInterval, ZScaleInterval)
from scipy import ndimage as ndi
from skimage.filters import sobel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from skycam_utils import alcor

OUT = str(Path(__file__).resolve().parent.parent / "gplots")
MEDIAN = f"{OUT}/2026-02-18_median.fits"
CACHE = "/tmp/horizon_cache/undet_xy.npz"

NIGHTS = [
    ("2024-09-04", os.path.expanduser("~/MMT/skycam_data/2024-09-04")),
    ("2026-01-11", "/Volumes/Samsung_4TB/skycam/2026-01-11"),
    ("2026-03-11", "/Volumes/Samsung_4TB/skycam/2026-03-11"),
    ("2026-05-18", "/Volumes/Samsung_4TB/skycam/2026-05-18"),
    ("2026-06-09", "/Volumes/Samsung_4TB/skycam/2026-06-09"),
]
TARGET = "2026-05-18"
USECOLS = ["xcen", "ycen", "flux_g_ap"]

BIN_PX, SMOOTH, THR = 2.5, 1.0, 2.0


def accumulate():
    """x, y, ap_det across all nights (cached)."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        print("loaded cache", CACHE, len(z["x"]), "measurements")
        return z["x"], z["y"], z["ap_det"]
    xs, ys, det = [], [], []
    for night, ddir in NIGHTS:
        frames = sorted(glob.glob(os.path.join(ddir, "*_phot.csv")))
        print(f"{night}: {len(frames)} frames")
        for f in frames:
            try:
                d = pd.read_csv(f, usecols=USECOLS)
            except Exception as exc:
                print("  skip", os.path.basename(f), exc)
                continue
            xs.append(d["xcen"].to_numpy())
            ys.append(d["ycen"].to_numpy())
            det.append((d["flux_g_ap"] > 0).to_numpy())
    x = np.concatenate(xs); y = np.concatenate(ys)
    ap_det = np.concatenate(det)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, x=x, y=y, ap_det=ap_det)
    print("cached", len(x), "measurements ->", CACHE)
    return x, y, ap_det


def main():
    target_dir = dict(NIGHTS)[TARGET]
    coadd = fits.getdata(f"{target_dir}/{TARGET}_night_lum_coadd.fits").astype(float)
    ny, nx = coadd.shape

    cal = alcor.alcor_calibration(Time(f"{TARGET}T12:00:00", scale="utc"))
    wcs = alcor.build_alcor_wcs(
        xcen=cal["xcen"], ycen=cal["ycen"], rotation=cal["rotation"],
        radial_coeffs=cal["radial_coeffs"], horizon_radius=cal["horizon_radius"],
        tangential_coeffs=cal["tangential_coeffs"], axis_tilt=cal["axis_tilt"])

    # same crop the probe uses: square of horizon_radius*1.02 about the WCS zenith
    zp = wcs.all_world2pix([[0.0, 90.0]], 0)[0]
    half = int(cal["horizon_radius"] * 1.02)
    x0, y0 = int(round(zp[0])), int(round(zp[1]))
    xs0, ys0 = max(0, x0 - half), max(0, y0 - half)
    xe0, ye0 = min(nx, x0 + half), min(ny, y0 + half)
    w, h = xe0 - xs0, ye0 - ys0
    sub = coadd[ys0:ye0, xs0:xe0]

    # undetected-star density in that crop
    x, y, ap_det = accumulate()
    undet = ~ap_det
    nbx, nby = int(w / BIN_PX), int(h / BIN_PX)
    H, _, _ = np.histogram2d(x[undet], y[undet], bins=[nbx, nby],
                             range=[[xs0, xs0 + w], [ys0, ys0 + h]])
    H = ndi.gaussian_filter(H, SMOOTH)
    dens = np.ma.masked_less(H.T, THR)

    # Sobel(log) edges on the cloudy-night median, same crop
    med = fits.getdata(MEDIAN).astype(float)
    sob_log = sobel(ndi.gaussian_filter(np.log10(np.clip(med, 1, None)), 1.0))
    sob = sob_log[ys0:ye0, xs0:xe0]

    fig, ax = plt.subplots(1, 3, figsize=(21, 7.2))

    norm = ImageNormalize(sub, interval=PercentileInterval(99.3),
                          stretch=AsinhStretch(a=0.02))
    ax[0].imshow(sub, origin="lower", cmap="gray", norm=norm)
    ax[0].imshow(dens, origin="lower", cmap="inferno", alpha=0.78,
                 norm=PowerNorm(0.5), extent=[0, w, 0, h])
    ax[0].set_title("undetected-star density (5 nights)")

    p = np.percentile(sob, 99.5)
    ax[1].imshow(sob, origin="lower", cmap="inferno", vmin=0, vmax=p)
    ax[1].set_title("Sobel(log) edges, 2026-02-18 cloudy median")

    vlo, vhi = ZScaleInterval().get_limits(sub)
    ax[2].imshow(sub, origin="lower", cmap="gray", vmin=vlo, vmax=vhi)
    ax[2].imshow(dens, origin="lower", cmap="inferno", alpha=0.7,
                 norm=PowerNorm(0.5), extent=[0, w, 0, h])
    edge_thr = np.percentile(sob, 99.0)
    ax[2].contour(sob, levels=[edge_thr], colors="cyan", linewidths=0.6,
                  extent=[0, w, 0, h])
    ax[2].set_title("Sobel edges (cyan) over undetected density")

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
        for fx, fy, lab in [(0.5, 0.98, "N"), (0.5, 0.02, "S"),
                            (0.02, 0.5, "E"), (0.98, 0.5, "W")]:
            a.text(fx, fy, lab, transform=a.transAxes, color="yellow",
                   fontsize=13, ha="center", va="center", weight="bold")

    fig.tight_layout()
    out = f"{OUT}/sobel_vs_undetected.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
