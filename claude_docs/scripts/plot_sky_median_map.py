"""
Render the median all-sky surface-brightness map from ``sky_median_map.py``.

Two panels, because the second is what makes the first trustworthy. The map
itself is the median site sky with the Milky Way masked out; the coverage panel
is how many frames survived that mask in each direction. A dark patch in the
map means something only if the coverage panel shows it was well sampled --
otherwise it is a thin median, not a dark sky.

Rendering follows ``plot_alcor_sky_brightness``: the zenith crop, north up,
``origin="lower"``, a ``cividis_r`` colorbar so bright sky reads bright, and
the same shared alt/az grid helpers.

``--map-only`` drops the coverage panel AND the provenance suptitle, for when
the map is going into a document rather than being checked: the run parameters
belong on the diagnostic version, and at single-panel width the line is too
long to fit anyway. The panel keeps its own title, so the figure still says
what it is. The output extension drives the backend, so
``-o something.pdf`` gives vector output.

Usage: plot_sky_median_map.py <map.fits> [-o OUT.png] [--map-only]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from skycam_utils.alcor.display import _alcor_zenith_crop_bounds


def _alt_az_grid(ax, wcs, xz, yz, alts=(0, 15, 30, 45, 60, 75),
                 azs=range(0, 360, 45), color="0.55"):
    """
    Alt/az overlay drawn in DATA coordinates on one axes.

    ``_add_alcor_alt_az_grid`` in ``alcor.display`` is figure-scoped -- it adds
    its own overlay axes positioned against the figure -- which is right for a
    single-panel plot and lands on top of the colorbar here. The ARC projection
    is equidistant in radius, so an altitude circle is a true circle about the
    zenith and the radius comes straight from the WCS.
    """
    th = np.radians(np.arange(0, 361, 2))
    for alt in alts:
        x, y = wcs.world_to_pixel_values(0.0, float(alt))
        r = float(np.hypot(x - xz, y - yz))
        ax.plot(xz + r * np.cos(th), yz + r * np.sin(th), lw=0.6,
                color=color, alpha=0.6, zorder=3)
        # Offset from straight up: the azimuth spokes label 0 deg there too.
        ax.annotate(f"{alt}\u00b0",
                    (xz + r * np.cos(np.radians(65)), yz + r * np.sin(np.radians(65))),
                    color=color, fontsize=7, ha="center", va="bottom", zorder=4)
    x0_, y0_ = wcs.world_to_pixel_values(0.0, float(min(alts)))
    rmax = float(np.hypot(x0_ - xz, y0_ - yz))
    for az in azs:
        x, y = wcs.world_to_pixel_values(float(az), float(min(alts)))
        ax.plot([xz, x], [yz, y], lw=0.5, color=color, alpha=0.5, zorder=3)
        f = 1.06
        ax.annotate(f"{az}\u00b0", (xz + f * (x - xz), yz + f * (y - yz)),
                    color=color, fontsize=8, ha="center", va="center", zorder=4)
    return rmax


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("map", help="output of sky_median_map.py")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--radius", type=int, default=700, help="crop half-size, px")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--map-only", action="store_true",
                   help="just the brightness map, without the coverage panel")
    p.add_argument("--cmap", default="cividis_r")
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args()

    out = args.output or os.path.splitext(args.map)[0] + ".png"

    with fits.open(args.map) as hdul:
        sky = hdul[0].data.astype(float)
        hdr = hdul[0].header
        nframes = hdul["NFRAMES"].data.astype(float)
        nights = list(hdul["NIGHTS"].data["NIGHT"])
        wcs = WCS(hdr)

    ny, nx = sky.shape
    xz, yz, x0, x1, y0, y1 = _alcor_zenith_crop_bounds(wcs, ny, nx, args.radius)

    sub = sky[y0:y1, x0:x1]
    cov = np.where(nframes[y0:y1, x0:x1] > 0, nframes[y0:y1, x0:x1], np.nan)

    finite = np.isfinite(sub)
    vmin = args.vmin if args.vmin is not None else np.percentile(sub[finite], 1)
    vmax = args.vmax if args.vmax is not None else np.percentile(sub[finite], 99)

    if args.map_only:
        fig, ax0 = plt.subplots(figsize=(8.4, 7.8))
        axes = [ax0]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6))
    extent = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)

    im = axes[0].imshow(sub, origin="lower", extent=extent, cmap=args.cmap,
                        vmin=vmin, vmax=vmax)
    axes[0].set_title(f"median clear-sky brightness, {len(nights)} nights "
                      f"(|b| > {hdr['BMIN']:.0f}$\\degree$ masked)")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)
    cb.set_label("V mag / arcsec$^2$")
    cb.ax.invert_yaxis()

    if not args.map_only:
        im2 = axes[1].imshow(cov, origin="lower", extent=extent, cmap="magma")
        axes[1].set_title("frames contributing (after the galactic-plane cut)")
        fig.colorbar(im2, ax=axes[1], fraction=0.046,
                     pad=0.02).set_label("frames")

    for ax in axes:
        rmax = _alt_az_grid(ax, wcs, float(xz), float(yz))
        pad = 1.12 * rmax
        ax.set_xlim(xz - pad, xz + pad)
        ax.set_ylim(yz - pad, yz + pad)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    if args.map_only:
        fig.tight_layout()
    else:
        span = f"{min(nights)} to {max(nights)}" if nights else "?"
        thru = "throughput-corrected" if hdr.get("THRUCORR") else "uncorrected"
        fig.suptitle(f"Alcor all-sky median surface brightness   "
                     f"{span}   stride {hdr.get('STRIDE', '?')}, "
                     f"{hdr.get('SPBIN', '?')}x{hdr.get('SPBIN', '?')} "
                     f"superpixels, {thru}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")

    ok = np.isfinite(sky)
    print(f"  {ok.sum():,} pixels mapped, "
          f"{np.nanmin(sky):.2f} to {np.nanmax(sky):.2f} mag/arcsec2")


if __name__ == "__main__":
    main()
