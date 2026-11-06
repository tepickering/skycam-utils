"""
Quick look at the LBTO/VATT OMEA frames (ZWO ASI6200MC Pro, 1x1 binning).

Exploratory: renders one raw Bayer frame as a coarse luminance image and
measures where the fisheye image circle actually falls on the 9576 x 6388
sensor, which is the thing that decides whether the MMT geometry pipeline can
be reused as-is or has to be re-fit.

Usage: lbto_quicklook.py <allsky-*.fits.fz> [-o OUT.png]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval

BIN = 8          # coarse binning for the quick look
EDGE_FRAC = 0.05  # illuminated where the binned signal clears this x the median


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("frame")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args()

    with fits.open(args.frame) as hdul:
        hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        raw = hdu.data.astype(np.float32)
        hdr = hdu.header
    ny, nx = raw.shape

    corners = np.concatenate([raw[:30, :30].ravel(), raw[:30, -30:].ravel(),
                              raw[-30:, :30].ravel(), raw[-30:, -30:].ravel()])
    bias = float(np.median(corners))

    # Bin by BIN x BIN; that also averages the Bayer mosaic to luminance.
    byy, bxx = ny // BIN, nx // BIN
    lum = (raw[:byy * BIN, :bxx * BIN] - bias).reshape(byy, BIN, bxx, BIN)
    lum = lum.mean(axis=(1, 3))

    med = float(np.median(lum[lum > 0.5 * np.percentile(lum, 90)]))
    mask = lum > EDGE_FRAC * med
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cx, cy = 0.5 * (x0 + x1) * BIN, 0.5 * (y0 + y1) * BIN
    width, height = (x1 - x0 + 1) * BIN, (y1 - y0 + 1) * BIN

    print(f"{args.frame}")
    print(f"  raw {nx} x {ny}, bias {bias:.1f} ADU, sky median {med:.1f} ADU "
          f"above bias, EXPOSURE {hdr.get('EXPOSURE')} s")
    print(f"  illuminated x {x0 * BIN}..{x1 * BIN}  y {y0 * BIN}..{y1 * BIN}")
    print(f"  centre ({cx:.0f}, {cy:.0f})  extent {width} x {height}")
    clipped = []
    if y0 * BIN <= BIN:
        clipped.append("bottom")
    if y1 * BIN >= ny - 2 * BIN:
        clipped.append("top")
    if x0 * BIN <= BIN:
        clipped.append("left")
    if x1 * BIN >= nx - 2 * BIN:
        clipped.append("right")
    print(f"  circle clipped at: {', '.join(clipped) if clipped else 'nowhere'}")

    vmin, vmax = ZScaleInterval(contrast=0.15).get_limits(lum)
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.imshow(np.power(np.clip((lum - vmin) / (vmax - vmin), 0, 1), 0.5),
              origin="lower", cmap="gray",
              extent=(0, bxx * BIN, 0, byy * BIN))
    ax.plot([cx], [cy], "r+", ms=14, mew=1.6)
    ax.add_patch(plt.Rectangle((x0 * BIN, y0 * BIN), width, height,
                               ec="#ff5500", fc="none", lw=1.2, ls="--"))
    ax.set_xlabel("raw x [px]")
    ax.set_ylabel("raw y [px]")
    ax.set_title(f"{hdr.get('INSTRUME', '?')} -- {hdr.get('DATE-OBS', '?')}  "
                 f"({hdr.get('EXPOSURE')} s)\n"
                 f"illuminated {width} x {height} px about "
                 f"({cx:.0f}, {cy:.0f}) in a {nx} x {ny} frame", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
