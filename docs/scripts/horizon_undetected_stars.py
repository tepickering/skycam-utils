"""Overlay detected/undetected catalog-star density on the night luminance co-add.

A probe for the Alcor horizon map: instead of inferring the obstruction boundary
from the star-trail / coverage field (which follows sky structure, not the
building), use the catalog stars themselves. `alcor_star_photometry` predicts every
bright star's raw pixel from the WCS and measures fixed-position photometry down to
alt 0; accumulated over a night the positions trace the geometry directly, because
each star's arc sweeps behind the roofline.

Two photometry methods give two discriminants (the CSVs are produced with --both):

- *aperture* (``flux_g_ap``): a detection is any positive aperture sum. Sensitive,
  but reflected light off the buildings leaks into the aperture and registers as a
  spurious detection, so the obstruction holes in the detected map are not fully
  dark.
- *Gaussian* (``flux_g_gauss``): a detection requires a compact PSF to fit within
  bounds and project to positive amplitude. Diffuse reflected building light fails
  this (the width hits the aperture bound, or the amplitude is ~0/negative), so a
  successful Gaussian fit is a stronger "real star" flag and the obstruction holes
  come out cleaner.

For each method both senses are rendered: *undetected* density traces where stars
go missing (the obstruction silhouette, contaminated by transient cloud and the
faint limit), and *detected* density traces where stars are seen (a fixed
obstruction is a hole; an intermittently-blocked area still accumulates detections,
so it appears in both). Comparing senses and methods discriminates real obstruction
from "sometimes seen, sometimes not" and from aperture reflection artifacts.

All nights' positions are combined onto one co-add. The WCS is stable to ~1 px
across the 2024 and 2026 epochs, so the raw pixel positions are stacked directly
(no reprojection). Points are rendered as a smoothed 2D-density heatmap with a low
threshold, which sharpens the edges and suppresses sparse interior speckle. Density
uses the shared WCS-predicted ``xcen``/``ycen`` for every map, so the geometry is
identical across methods.

In the raw frame as displayed (origin="lower"): N up, E left, S down, W right.

Writes four figures to ../gplots. Not part of the installed package.
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

USECOLS = ["xcen", "ycen", "flux_g_ap", "flux_g_gauss"]

# ---- tunables ----
BIN_PX = 2.5      # density bin size, native px
SMOOTH = 1.0      # gaussian smoothing of the density, in bins
THR = 2.0         # mask density below this (counts) -> kills interior speckle


def collect_night(night_dir):
    """Concatenated photometry (USECOLS) for one night, with a detection summary."""
    frames = sorted(glob.glob(os.path.join(night_dir, "*_phot.csv")))
    parts = []
    for f in frames:
        try:
            parts.append(pd.read_csv(f, usecols=USECOLS))
        except Exception as exc:
            print(f"  skip {os.path.basename(f)}: {exc}")
    df = pd.concat(parts, ignore_index=True)
    n = len(df)
    ap = int((df["flux_g_ap"] > 0).sum())
    gz = int((df["flux_g_gauss"] > 0).sum())
    print(f"  {len(frames)} frames, {n} measurements: "
          f"aperture {ap} det ({100*(n-ap)/n:.1f}% undet), "
          f"Gaussian {gz} det ({100*(n-gz)/n:.1f}% undet)")
    return df


def render_density(px, py, sub, al, geom, label, title, out):
    """Render one smoothed-density heatmap over the co-add crop and save it."""
    xs, ys, w, h = geom
    nbx, nby = int(w / BIN_PX), int(h / BIN_PX)
    H, _, _ = np.histogram2d(px, py, bins=[nbx, nby],
                             range=[[xs, xs + w], [ys, ys + h]])
    H = ndi.gaussian_filter(H, SMOOTH)
    dens = np.ma.masked_less(H.T, THR)   # transpose: histogram2d is [x, y]

    fig, ax = plt.subplots(figsize=(13, 13))
    norm = ImageNormalize(sub, interval=PercentileInterval(99.3),
                          stretch=AsinhStretch(a=0.02))
    ax.imshow(sub, origin="lower", cmap="gray", norm=norm, extent=[0, w, 0, h])
    im = ax.imshow(dens, origin="lower", cmap="inferno", alpha=0.78,
                   norm=PowerNorm(0.5), extent=[0, w, 0, h])
    ax.contour(al, levels=[0, 10, 20], colors="cyan", linewidths=0.4, alpha=0.6)
    ax.contour(al, levels=[25], colors="yellow", linewidths=0.8, alpha=0.8)
    for fx, fy, lab in [(0.5, 0.985, "N"), (0.5, 0.015, "S"),
                        (0.015, 0.5, "E"), (0.985, 0.5, "W")]:
        ax.text(fx, fy, lab, transform=ax.transAxes, color="yellow",
                fontsize=15, ha="center", va="center", weight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


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

    dfs = []
    for night, ddir in NIGHTS:
        print(night)
        dfs.append(collect_night(ddir))
    df = pd.concat(dfs, ignore_index=True)
    x, y = df["xcen"].to_numpy(), df["ycen"].to_numpy()
    ap_det = (df["flux_g_ap"] > 0).to_numpy()
    gauss_det = (df["flux_g_gauss"] > 0).to_numpy()   # NaN (fit failed) -> False
    print(f"combined {len(df)} measurements: "
          f"aperture {ap_det.sum()} det, Gaussian {gauss_det.sum()} det")

    zp = wcs.all_world2pix([[0.0, 90.0]], 0)[0]
    half = int(cal["horizon_radius"] * 1.02)
    x0, y0 = int(round(zp[0])), int(round(zp[1]))
    xs, ys = max(0, x0 - half), max(0, y0 - half)
    xe, ye = min(nx, x0 + half), min(ny, y0 + half)
    sub, al = coadd[ys:ye, xs:xe], alt[ys:ye, xs:xe]
    geom = (xs, ys, xe - xs, ye - ys)

    nights_label = " + ".join(n for n, _ in NIGHTS)
    nn = len(NIGHTS)
    combos = [
        ("undetected", "aperture", ~ap_det, "alcor_undetected_stars.png"),
        ("detected", "aperture", ap_det, "alcor_detected_stars.png"),
        ("undetected", "Gaussian", ~gauss_det, "alcor_undetected_stars_gauss.png"),
        ("detected", "Gaussian", gauss_det, "alcor_detected_stars_gauss.png"),
    ]
    for sense, method, mask, fname in combos:
        render_density(
            x[mask], y[mask], sub, al, geom,
            label=f"{sense}-star density, {method} ({nn} nights)",
            title=f"{sense.capitalize()} catalog stars ({method}), {nights_label} "
                  f"combined ({int(mask.sum())} positions)  on {TARGET} co-add",
            out=f"{OUT}/{fname}")


if __name__ == "__main__":
    main()
