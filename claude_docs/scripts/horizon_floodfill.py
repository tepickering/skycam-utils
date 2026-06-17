#!/usr/bin/env python
"""Alcor horizon mask by flood-filling the sky against the 2D Sobel edge map.

The cloudy-night median is a smooth bright sky disk; obstructions are dark with
sharp Sobel edges. Treat strong edges as walls and flood-fill the sky outward
from the zenith: whatever the fill can't reach (blocked by a roofline edge, or
enclosed -- like the thin lightning-rod spike) is masked as not-sky.

Where the Sobel wall is ill-defined -- the SW->W building sector (az 225-270) --
plug the gap with the undetected-star patch so the fill can't leak through.
Everywhere else (W -> N -> E -> S -> SW) the boundary is the Sobel edge.

The undetected-star patch is accumulated from per-frame ``*_phot.csv`` fixed-
position photometry over several nights (a star is "undetected" when its G
aperture flux is 0). The 4.7M-row accumulation is cached to a temp .npz; delete
it to rebuild. Not part of the installed package.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import sobel
from skimage.morphology import opening, disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skycam_utils.alcor import build_alcor_wcs, alcor_calibration

GP = Path(__file__).resolve().parent.parent / "gplots"
MEDIAN = GP / "2026-02-18_median.fits"
CACHE = "/tmp/horizon_cache/undet_xy.npz"   # derived cache (~80 MB); rebuilt if absent
EPOCH = "2026-02-18"

# nights whose per-frame photometry feeds the undetected-star patch
NIGHTS = [
    os.path.expanduser("~/MMT/skycam_data/2024-09-04"),
    "/Volumes/Samsung_4TB/skycam/2026-01-11",
    "/Volumes/Samsung_4TB/skycam/2026-03-11",
    "/Volumes/Samsung_4TB/skycam/2026-05-18",
    "/Volumes/Samsung_4TB/skycam/2026-06-09",
]

SEC_LO, SEC_HI = 225.0, 270.0      # undetected-star sector (SW -> W)
EDGE_PCT = 96.0                    # Sobel-edge wall threshold (pct within FOV)
EDGE_DILATE = 1                    # close small gaps in the edge walls
UND_THR = 0.5                      # undetected fraction = obstructed
UND_MINCOUNT = 15                  # min star transits per (az,alt) cell
RIM_ALT = 1.5                      # a not-sky blob reaching below this alt is rim-connected
ROD_AREA_MIN = 400                 # keep isolated blobs >= this (px): the lightning rod
OPEN_RADIUS = 3                    # morphological opening (px) to sever thin necks / specks


def load_undetected():
    """x, y, ap_det for every measurement across NIGHTS (cached to CACHE)."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["x"], z["y"], z["ap_det"]
    xs, ys, det = [], [], []
    for ddir in NIGHTS:
        for f in sorted(glob.glob(os.path.join(ddir, "*_phot.csv"))):
            try:
                d = pd.read_csv(f, usecols=["xcen", "ycen", "flux_g_ap"])
            except Exception as exc:
                print("  skip", os.path.basename(f), exc)
                continue
            xs.append(d["xcen"].to_numpy())
            ys.append(d["ycen"].to_numpy())
            det.append((d["flux_g_ap"] > 0).to_numpy())
    x, y = np.concatenate(xs), np.concatenate(ys)
    ap_det = np.concatenate(det)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, x=x, y=y, ap_det=ap_det)
    print(f"cached {len(x)} measurements -> {CACHE}")
    return x, y, ap_det


def main():
    cal = alcor_calibration(Time(f"{EPOCH}T12:00:00"))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"], radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])

    img = fits.getdata(MEDIAN).astype(float)
    ny, nx = img.shape

    # per-pixel az/alt
    yy, xx = np.mgrid[0:ny, 0:nx]
    w = wcs.all_pix2world(np.column_stack([xx.ravel(), yy.ravel()]), 0)
    az = (w[:, 0] % 360.0).reshape(ny, nx)
    alt = w[:, 1].reshape(ny, nx)
    in_fov = alt > 0.0

    # 2D Sobel edges on the log image (suppresses the smooth vignette gradient)
    E = sobel(ndi.gaussian_filter(np.log10(np.clip(img, 1, None)), 1.0))
    thr = np.percentile(E[in_fov], EDGE_PCT)
    wall = (E > thr) & in_fov
    if EDGE_DILATE:
        wall = ndi.binary_dilation(wall, iterations=EDGE_DILATE)

    # undetected-star patch -> obstruction in the SW->W sector
    px, py, ap_det = load_undetected()
    sw = wcs.all_pix2world(np.column_stack([px, py]), 0)
    saz, salt = sw[:, 0] % 360.0, sw[:, 1]
    undet = ~ap_det
    ok = np.isfinite(saz) & np.isfinite(salt)
    saz, salt, undet = saz[ok], salt[ok], undet[ok]
    az_e = np.arange(-0.25, 360.0, 0.5)
    alt_e = np.arange(-6.25, 30.26, 0.5)
    az_c = 0.5 * (az_e[:-1] + az_e[1:])
    alt_c = 0.5 * (alt_e[:-1] + alt_e[1:])
    Htot, _, _ = np.histogram2d(saz, salt, bins=[az_e, alt_e])
    Hund, _, _ = np.histogram2d(saz[undet], salt[undet], bins=[az_e, alt_e])
    frac = ndi.gaussian_filter(Hund / np.maximum(Htot, 1), (1.0, 1.0))
    fr_i = RegularGridInterpolator((az_c, alt_c), frac, bounds_error=False, fill_value=0.0)
    ct_i = RegularGridInterpolator((az_c, alt_c), Htot, bounds_error=False, fill_value=0.0)
    pts = np.column_stack([az.ravel(), np.clip(alt.ravel(), alt_c[0], alt_c[-1])])
    fr_pix = fr_i(pts).reshape(ny, nx)
    ct_pix = ct_i(pts).reshape(ny, nx)
    sector = (az >= SEC_LO) & (az <= SEC_HI)
    undet_obstruction = sector & in_fov & (fr_pix >= UND_THR) & (ct_pix >= UND_MINCOUNT)

    # flood-fill the sky: free pixels = inside FOV, not a wall, not undetected-blocked
    free = in_fov & ~wall & ~undet_obstruction
    lbl, _ = ndi.label(free, structure=np.ones((3, 3)))
    zp = wcs.all_world2pix([[0.0, 90.0]], 0)[0]
    zx, zy = int(round(zp[0])), int(round(zp[1]))
    if not free[zy, zx]:                      # nudge to nearest free pixel
        fy, fx = np.where(free)
        k = np.argmin((fx - zx) ** 2 + (fy - zy) ** 2)
        zx, zy = fx[k], fy[k]
    sky = lbl == lbl[zy, zx]
    notsky_raw = in_fov & ~sky                # obstruction intruding above horizon

    # morphological opening: severs thin necks (so loosely-attached open-sky
    # protrusions detach from the rim band) and erases sub-disk specks, while
    # large structures regrow to ~their original extent.
    notsky_open = opening(notsky_raw, disk(OPEN_RADIUS)) if OPEN_RADIUS else notsky_raw

    # --- drop spurious open-sky pockets ---------------------------------
    # Real obstructions (terrain, buildings) reach the horizon rim; the one
    # genuine isolated feature is the lightning rod. So keep a not-sky blob
    # only if it touches the rim (min altitude < RIM_ALT) OR it is large
    # enough to be the rod; small floating pockets in open sky are dropped.
    nlab, nn = ndi.label(notsky_open, structure=np.ones((3, 3)))
    idx = np.arange(1, nn + 1)
    size = np.bincount(nlab.ravel())[1:]
    min_alt = ndi.minimum(alt, nlab, index=idx)
    keep = (min_alt < RIM_ALT) | (size >= ROD_AREA_MIN)
    print("largest not-sky components (size, min_alt, kept):")
    for k in np.argsort(size)[::-1][:12]:
        print(f"  size {int(size[k]):7d}  min_alt {min_alt[k]:6.1f}  "
              f"{'keep' if keep[k] else 'DROP'}")
    keep_full = np.concatenate([[False], keep])
    notsky = keep_full[nlab]

    # same pocket cleanup WITHOUT the opening, for the before/after comparison
    nl0, nn0 = ndi.label(notsky_raw, structure=np.ones((3, 3)))
    sz0 = np.bincount(nl0.ravel())[1:]
    ma0 = ndi.minimum(alt, nl0, index=np.arange(1, nn0 + 1))
    k0 = np.concatenate([[False], (ma0 < RIM_ALT) | (sz0 >= ROD_AREA_MIN)])
    notsky_noopen = k0[nl0]

    # --- complete pixel mask: obstructions above the horizon, PLUS everything
    # at/below altitude 0 (and outside the projection). 1 = not-sky, 0 = sky.
    horizon_mask = (~in_fov) | notsky

    mask_dir = Path(__file__).resolve().parents[2] / "skycam_utils" / "data" / "horizon"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"alcor_horizon_{EPOCH}.fits.gz"
    mh = fits.PrimaryHDU(data=horizon_mask.astype(np.uint8))
    mh.header["METHOD"] = ("sobel-floodfill", "horizon mask construction")
    mh.header["SRCMED"] = (os.path.basename(str(MEDIAN)), "source median image")
    mh.header["EDGEPCT"] = (EDGE_PCT, "Sobel-edge wall percentile")
    mh.header["OPENR"] = (OPEN_RADIUS, "morphological opening radius (px)")
    mh.header["ALTCUT"] = (0.0, "altitude cutoff (deg); <= is masked")
    mh.header["NMASK"] = (int(horizon_mask.sum()), "masked (not-sky) pixels")
    mh.header["NSKY"] = (int((~horizon_mask).sum()), "valid-sky pixels")
    mh.writeto(mask_path, overwrite=True)
    print("wrote", mask_path)

    print(f"edge thr {thr:.4g}  wall px {wall.sum()}  "
          f"undet-obstruction px {undet_obstruction.sum()}")
    print(f"sky px {sky.sum()}  not-sky raw {notsky_raw.sum()} -> "
          f"opened {notsky_open.sum()} -> cleaned {notsky.sum()} px "
          f"({100*notsky.sum()/in_fov.sum():.1f}% of FOV); "
          f"dropped {nn - int(keep.sum())} pockets")

    # ============================ figure ============================
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # panel A: Sobel edges + final sky boundary
    p = np.percentile(E[in_fov], 99.6)
    ax[0].imshow(np.where(in_fov, E, np.nan), origin="lower", cmap="inferno",
                 vmin=0, vmax=p)
    ax[0].contour(sky.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    ax[0].set_title("Sobel(log) edges + flood-fill sky boundary (cyan)")

    # panel B: complete pixel mask on the median (red = not-sky: obstructions
    # above the horizon plus everything at/below alt 0)
    vmin, vmax = ZScaleInterval().get_limits(img)
    ax[1].imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ov = np.zeros((ny, nx, 4))
    ov[horizon_mask] = (1, 0, 0, 0.32)
    ax[1].imshow(ov, origin="lower")
    # mark the undetected sector seams (az 225 and 270 rays)
    for a in (SEC_LO, SEC_HI):
        ra = np.linspace(0.5, 25, 60)
        rx, ry = wcs.world_to_pixel_values(np.full_like(ra, a), ra)
        ax[1].plot(rx, ry, ":", color="cyan", lw=1.0, alpha=0.8)
    ax[1].set_title("complete horizon mask (red=not-sky) on 2026-02-18 median; "
                    "dotted = SW->W undetected sector")

    for a in ax:
        a.set_xlim(0, nx); a.set_ylim(0, ny)
        a.set_xticks([]); a.set_yticks([])
        for fx, fy, lab in [(0.5, 0.975, "N"), (0.5, 0.025, "S"),
                            (0.025, 0.5, "E"), (0.975, 0.5, "W")]:
            a.text(fx, fy, lab, transform=a.transAxes, color="yellow",
                   fontsize=13, ha="center", va="center", weight="bold")

    fig.tight_layout()
    out = GP / "horizon_floodfill.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)

    # --- south-region (S/SSW/SW) before/after opening zoom ---------------
    hr = cal["horizon_radius"]
    sx0, sx1 = max(0, int(zx - 380)), min(nx, int(zx + 700))
    sy0, sy1 = max(0, int(zy - hr - 25)), min(ny, int(zy - 150))
    crop = img[sy0:sy1, sx0:sx1]
    figs, axs = plt.subplots(1, 2, figsize=(18, 7.5))
    for a, m, t in [(axs[0], notsky_noopen, "no opening"),
                    (axs[1], notsky, f"opening r={OPEN_RADIUS}")]:
        a.imshow(crop, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ov2 = np.zeros((sy1 - sy0, sx1 - sx0, 4))
        ov2[m[sy0:sy1, sx0:sx1]] = (1, 0, 0, 0.35)
        a.imshow(ov2, origin="lower")
        a.set_title(f"south (E left -> W right) — {t}")
        a.set_xticks([]); a.set_yticks([])
    figs.tight_layout()
    out2 = GP / "horizon_floodfill_south.png"
    figs.savefig(out2, dpi=120, bbox_inches="tight")
    print("wrote", out2)


if __name__ == "__main__":
    main()
