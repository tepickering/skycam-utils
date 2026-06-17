"""Prototype auto-masker for the Alcor horizon map (exploration / record).

Derives an initial "what is sky vs not-sky" mask for the Alcor all-sky camera
from a whole-night luminance co-add and std-dev image, in raw pixel space, using
the calibrated alt/az WCS for geometry. This is the auto-propose half of the
planned horizon-map feature (az/alt-canonical static asset); the algorithm here
is being tuned against the 2026-05-18 night before it is packaged.

Approach (per user guidance)
----------------------------
The obstruction boundary is the TRAIL-TERMINATION edge: in the std image the star
trails stop at a clear edge (a building roofline, the terrain), cross-checked
against the dark edge in the co-add. So in each azimuth the ground obstruction is
everything BELOW the altitude at which trail coverage drops off, and that region
is filled down to the horizon (generous). Two firm rules from the user:
  * smooth, edgeless sky must NOT be masked -- so masking is driven by a coverage
    *drop* relative to the clear sky above, never by a low absolute level;
  * alt < 0 deg is masked by default.

Coverage = peak temporal variance (max-filtered std) in a small neighborhood:
high where star trails pass, low over a motionless obstruction. The clear-sky
reference for each azimuth is taken from ABOVE the 25 deg zone (SKY_LO..SKY_HI),
where -- apart from the rod -- there is only sky, so a large building cannot
absorb its own reference (the bug that left building bodies hollow).

Two obstructions are handled separately from the per-azimuth fill:
  * the rotating SW dome MOVES (high coverage, no trail-termination drop), so it
    is caught as a compact bright residual (bright_blob);
  * the lightning rod is the one obstruction ABOVE 25 deg, caught as a thin dark
    residual there.

The mask is deliberately *generous* (MARGIN + dilation).

In the raw frame as displayed (origin="lower"): N up, E left, S down, W right.

Hardcoded local data paths (adjust to re-run elsewhere); writes overlays to
../gplots. Not part of the installed package.
"""
import os

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import (ZScaleInterval, AsinhStretch, ImageNormalize,
                                    PercentileInterval)
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skycam_utils import alcor

DATA = "/Volumes/Samsung_4TB/skycam/2026-05-18"
NIGHT = "2026-05-18"
OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "gplots"))

# ---- tunables ----
ALT_MAX = 25.0        # deg, obstruction-zone ceiling (rod excepted)
SKY_LO, SKY_HI = 25.0, 33.0   # deg, clear-sky coverage reference band (per azimuth)
COV_SIZE = 15         # px neighborhood for the trail-coverage (max-variance) map
AZ_STEP = 1.0         # deg, azimuth bin
ALT_STEP = 0.5        # deg, altitude bin
FRAC = 0.30           # trail-terminated where coverage < FRAC * clear-sky coverage
GAP_DEG = 2.0         # deg, bridge coverage gaps up to this when climbing from rim
AZ_SMOOTH = 5         # az bins, circular median smoothing of the horizon
MARGIN = 1.0          # deg, raise the horizon this much (generous)
RING = 0.5            # deg, altitude ring for the co-add background fill
BG_DECIM = 8          # decimation factor for the fast 2D median background
BG_MED = 9            # median-filter size on the decimated image (~72 px native)
KB = 6.0              # bright-structure threshold (robust sigma) for the dome
BLOB_OPEN = 6         # px disk-opening radius: keep the bright dome blob
KD = 5.0              # dark-silhouette (rod) threshold (robust sigma)
MIN_AREA = 40         # px, drop smaller not-sky components (speckle)
CLOSE = 3             # px closing iterations
DILATE = 6            # px generous dilation of not-sky


def disk(radius):
    yk, xk = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xk ** 2 + yk ** 2) <= radius * radius


def ring_stat(values, idx, mask, func, nring):
    out = np.full(nring, np.nan)
    for r in range(nring):
        sel = mask & (idx == r)
        if sel.sum() >= 30:
            out[r] = func(values[sel])
    good = np.isfinite(out)
    if good.sum() >= 2:
        out = np.interp(np.arange(nring), np.where(good)[0], out[good])
    return out


def smooth_bg(field, within, ai, nring, ny, nx):
    ring = ring_stat(field, ai, within, np.median, nring)
    filled = np.where(within, field, ring[ai])
    dec = filled[::BG_DECIM, ::BG_DECIM]
    smooth = ndi.median_filter(dec, size=BG_MED)
    return ndi.zoom(smooth, (ny / dec.shape[0], nx / dec.shape[1]), order=1)[:ny, :nx]


def per_azimuth_horizon(cov, az, alt, within):
    """Per-azimuth trail-termination horizon: climbing from the rim, the
    obstruction is the contiguous run where coverage < FRAC * clear-sky coverage
    (small gaps bridged); fill everything below its top. The clear-sky reference
    is taken from SKY_LO..SKY_HI (above the zone), so a tall building cannot
    absorb its own reference. Smooth sky (coverage ~ reference) yields no run."""
    naz = int(round(360 / AZ_STEP))
    nfull = int(round(SKY_HI / ALT_STEP))
    nzone = int(round(ALT_MAX / ALT_STEP))
    sky0 = int(round(SKY_LO / ALT_STEP))
    azb = np.clip((az / AZ_STEP).astype(int), 0, naz - 1)
    altb = np.clip((alt / ALT_STEP).astype(int), 0, nfull - 1)
    m = within & (alt >= 0) & (alt < SKY_HI)
    flat = azb * nfull + altb
    cnt = np.bincount(flat[m], minlength=naz * nfull).astype(float)
    ssum = np.bincount(flat[m], weights=cov[m], minlength=naz * nfull)
    prof = np.full(naz * nfull, np.nan)
    nz = cnt > 0
    prof[nz] = ssum[nz] / cnt[nz]
    prof = prof.reshape(naz, nfull)

    gapb = int(round(GAP_DEG / ALT_STEP))
    horizon = np.zeros(naz)
    for a in range(naz):
        skyref = np.nanmedian(prof[a, sky0:nfull])
        if not np.isfinite(skyref) or skyref <= 0:
            continue
        thr = FRAC * skyref
        top, gap = 0.0, 0
        for k in range(nzone):
            v = prof[a, k]
            if (not np.isfinite(v)) or v < thr:
                top = (k + 1) * ALT_STEP
                gap = 0
            else:
                gap += 1
                if gap > gapb:
                    break
        horizon[a] = top
    horizon = ndi.median_filter(horizon, size=AZ_SMOOTH, mode="wrap")
    horizon = np.where(horizon > 0, horizon + MARGIN, 0.0)  # generous, but leave open azimuths open
    return within & (alt < horizon[azb]), horizon


def main():
    coadd = fits.getdata(f"{DATA}/{NIGHT}_night_lum_coadd.fits").astype(float)
    std = fits.getdata(f"{DATA}/{NIGHT}_night_lum_std.fits").astype(float)
    ny, nx = coadd.shape

    cal = alcor.alcor_calibration(Time(f"{NIGHT}T12:00:00", scale="utc"))
    wcs = alcor.build_alcor_wcs(
        xcen=cal["xcen"], ycen=cal["ycen"], rotation=cal["rotation"],
        radial_coeffs=cal["radial_coeffs"], horizon_radius=cal["horizon_radius"],
        tangential_coeffs=cal["tangential_coeffs"], axis_tilt=cal["axis_tilt"])

    yy, xx = np.mgrid[0:ny, 0:nx]
    world = wcs.all_pix2world(np.column_stack([xx.ravel(), yy.ravel()]), 0)
    az = world[:, 0].reshape(ny, nx)
    alt = world[:, 1].reshape(ny, nx)
    within = np.isfinite(alt) & (alt >= 0.0)

    nring = int(90 / RING)
    ai = np.clip((alt / RING).astype(int), 0, nring - 1)

    # co-add residual (for the bright dome and the dark rod)
    bg2d = smooth_bg(coadd, within, ai, nring, ny, nx)
    resid = coadd - bg2d
    sig = 1.4826 * np.median(np.abs(resid[within] - np.median(resid[within])))

    # trail coverage and the per-azimuth trail-termination horizon
    cov = ndi.maximum_filter(std, size=COV_SIZE)
    ground, horizon = per_azimuth_horizon(cov, az, alt, within)

    # rotating dome: compact bright residual in the zone (it moves -> not in `ground`)
    bright = within & (alt < ALT_MAX) & (resid > KB * sig)
    bright_blob = ndi.binary_opening(bright, structure=disk(BLOB_OPEN))

    # lightning rod: thin dark residual above the zone
    rod = within & (alt >= ALT_MAX) & (resid < -KD * sig)

    core = within & (ground | bright_blob | rod)
    lab, n = ndi.label(core)
    if n:
        areas = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = areas >= MIN_AREA
        core = keep[lab]
    core = ndi.binary_closing(core, ndi.generate_binary_structure(2, 2), iterations=CLOSE)
    core = ndi.binary_dilation(core, iterations=DILATE) & within
    not_sky = core | (~within)   # alt < 0 / outside FOV masked by default

    pct = 100 * core.sum() / within.sum()
    print(f"horizon deg: med={np.median(horizon):.1f} max={horizon.max():.1f}  "
          f"not-sky {core.sum()} px ({pct:.1f}% of FOV)")

    zp = wcs.all_world2pix([[0.0, 90.0]], 0)[0]
    half = int(cal["horizon_radius"] * 1.02)
    x0, y0 = int(round(zp[0])), int(round(zp[1]))
    sl = (slice(max(0, y0 - half), min(ny, y0 + half)),
          slice(max(0, x0 - half), min(nx, x0 + half)))
    compass = [(0.5, 0.985, "N"), (0.5, 0.015, "S"),
               (0.015, 0.5, "E"), (0.985, 0.5, "W")]
    for name, img, mode in [("coadd", coadd, "asinh"), ("std", std, "zscale")]:
        sub, ns, al = img[sl], not_sky[sl], alt[sl]
        fig, ax = plt.subplots(figsize=(12, 12))
        if mode == "asinh":
            norm = ImageNormalize(sub, interval=PercentileInterval(99.3),
                                  stretch=AsinhStretch(a=0.02))
        else:
            norm = ImageNormalize(sub, interval=ZScaleInterval(contrast=0.15))
        ax.imshow(sub, origin="lower", cmap="gray", norm=norm)
        ax.contour(ns.astype(float), levels=[0.5], colors="red", linewidths=1.0)
        ax.contourf(ns.astype(float), levels=[0.5, 1.5], colors="red", alpha=0.18)
        ax.contour(al, levels=[0, 10, 20], colors="cyan", linewidths=0.4, alpha=0.5)
        ax.contour(al, levels=[ALT_MAX], colors="yellow", linewidths=0.8, alpha=0.8)
        for fx, fy, lab_txt in compass:
            ax.text(fx, fy, lab_txt, transform=ax.transAxes, color="yellow",
                    fontsize=15, ha="center", va="center", weight="bold")
        ax.set_title(f"auto not-sky (red); alt<{ALT_MAX:.0f} zone (yellow)  "
                     f"{name}  {pct:.1f}% masked")
        ax.set_xticks([])
        ax.set_yticks([])
        out = f"{OUT}/alcor_horizon_auto_{name}.png"
        fig.savefig(out, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
