#!/usr/bin/env python
"""Extract the horizon outline from a median-stacked alcor luminance image.

The median of a cloudy/foggy night is a smooth bright sky disk interrupted by
the static, sharp-edged silhouettes of horizon obstructions (terrain, domes,
buildings). This traces that silhouette as altitude vs azimuth.

Method: using the calibrated raw-frame WCS, walk a radial ray outward (from the
zenith toward the horizon) at each azimuth, sampling the median brightness vs
altitude. The horizon for that azimuth is the altitude of the steepest
brightness *rise* (dark obstruction below -> bright sky above) within the
search band -- for an unobstructed ray this lands at the optical rim near
alt~0, and for an obstructed ray it lands at the top of the silhouette.

Outputs (claude_docs/gplots/):
  <tag>_horizon.csv  -- azimuth, altitude, edge strength, raw-frame x/y
  <tag>_horizon_profile.png -- altitude vs azimuth (the horizon profile)
  <tag>_horizon_overlay.png -- the traced outline drawn on the median image

Usage:
    python alcor_horizon_extract.py [MEDIAN_FITS] [--epoch 2026-02-18]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from scipy.ndimage import map_coordinates, gaussian_filter1d

from skycam_utils.alcor import build_alcor_wcs, alcor_calibration

GPLOTS = Path(__file__).resolve().parent.parent / "gplots"

# search band and sampling for the radial rays
ALT_LO, ALT_HI, ALT_STEP = -6.0, 25.0, 0.05   # deg
AZ_STEP = 0.5                                  # deg
SMOOTH_ALT = 0.4                               # gaussian smoothing along ray (deg)
SMOOTH_AZ = 1.5                                # gaussian smoothing of az profile (deg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("median", nargs="?",
                   default=str(GPLOTS / "2026-02-18_median.fits"))
    p.add_argument("--epoch", default=None,
                   help="UT date (YYYY-MM-DD) for the WCS calibration epoch; "
                        "default: parsed from the filename tag")
    args = p.parse_args()

    median = Path(args.median)
    tag = median.name.replace("_median.fits", "")
    img = fits.getdata(median).astype(np.float32)
    ny, nx = img.shape

    epoch = args.epoch or tag
    cal = alcor_calibration(Time(f"{epoch}T12:00:00"))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])

    az = np.arange(0.0, 360.0, AZ_STEP)
    alt = np.arange(ALT_HI, ALT_LO - 1e-9, -ALT_STEP)   # zenith -> horizon
    nalt = alt.size

    horizon_alt = np.full(az.size, np.nan)
    edge_strength = np.zeros(az.size)
    hx = np.full(az.size, np.nan)
    hy = np.full(az.size, np.nan)

    for i, a in enumerate(az):
        # pixel track of this azimuth ray, from high altitude down to ALT_LO
        px, py = wcs.world_to_pixel_values(np.full(nalt, a), alt)
        on = (px >= 0) & (px < nx - 1) & (py >= 0) & (py < ny - 1)
        prof = np.full(nalt, np.nan)
        prof[on] = map_coordinates(img, [py[on], px[on]], order=1, mode="nearest")
        if np.count_nonzero(on) < 10:
            continue
        # interpolate small gaps, then smooth along the ray
        idx = np.arange(nalt)
        good = np.isfinite(prof)
        prof = np.interp(idx, idx[good], prof[good])
        prof = gaussian_filter1d(prof, SMOOTH_ALT / ALT_STEP)
        # gradient wrt increasing altitude (alt array is descending): a sky-ward
        # rise (dark below -> bright above) is a positive d(brightness)/d(alt).
        grad = -np.gradient(prof)              # >0 where brightness rises with alt
        # ignore the outermost samples (off-frame ringing) when picking the edge
        j = int(np.argmax(grad))
        horizon_alt[i] = alt[j]
        edge_strength[i] = grad[j]
        hx[i], hy[i] = float(px[j]), float(py[j])

    # lightly smooth the azimuthal profile (it is periodic)
    sm = SMOOTH_AZ / AZ_STEP
    pad = int(np.ceil(3 * sm))
    wrapped = np.concatenate([horizon_alt[-pad:], horizon_alt, horizon_alt[:pad]])
    horizon_smooth = gaussian_filter1d(wrapped, sm)[pad:pad + az.size]

    # --- CSV ---
    csv = GPLOTS / f"{tag}_horizon.csv"
    np.savetxt(
        csv,
        np.column_stack([az, horizon_alt, horizon_smooth, edge_strength, hx, hy]),
        delimiter=",", comments="",
        header="azimuth_deg,altitude_deg,altitude_smooth_deg,edge_strength,x_pix,y_pix",
        fmt=["%.2f", "%.4f", "%.4f", "%.2f", "%.3f", "%.3f"],
    )
    print(f"wrote {csv}")
    print(f"horizon altitude: min {np.nanmin(horizon_alt):.2f} "
          f"max {np.nanmax(horizon_alt):.2f} "
          f"median {np.nanmedian(horizon_alt):.2f} deg")

    # --- profile plot ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(az, horizon_alt, ".", ms=2, color="0.6", label="raw edge")
    ax.plot(az, horizon_smooth, "-", color="C3", lw=1.5,
            label=f"smoothed ({SMOOTH_AZ}deg)")
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("azimuth (deg, N=0, E=90)")
    ax.set_ylabel("horizon altitude (deg)")
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(0, 361, 30))
    ax.set_title(f"{tag}  extracted horizon outline")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    prof_png = GPLOTS / f"{tag}_horizon_profile.png"
    fig.savefig(prof_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {prof_png}")

    # --- overlay on the median image ---
    vmin, vmax = ZScaleInterval().get_limits(img)
    fig, ax = plt.subplots(figsize=(10, 10 * ny / nx))
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    # draw the smoothed outline (recompute its pixel track for a clean closed curve)
    opx, opy = wcs.world_to_pixel_values(az, horizon_smooth)
    opx = np.append(opx, opx[0])
    opy = np.append(opy, opy[0])
    ax.plot(opx, opy, "-", color="C1", lw=1.2, label="extracted horizon")
    # reference: nominal alt=0 circle
    zpx, zpy = wcs.world_to_pixel_values(az, np.zeros_like(az))
    ax.plot(np.append(zpx, zpx[0]), np.append(zpy, zpy[0]),
            "--", color="C0", lw=0.8, alpha=0.7, label="alt = 0 deg")
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title(f"{tag}  horizon outline on median luminance")
    ax.legend(loc="lower left")
    ov_png = GPLOTS / f"{tag}_horizon_overlay.png"
    fig.savefig(ov_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ov_png}")


if __name__ == "__main__":
    main()
