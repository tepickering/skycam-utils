"""
Eight-panel overview of a night's cloud extinction, for review at a glance.

A convenience wrapper over the packaged API (``alcor_extinction_stars`` /
``alcor_extinction_grid`` / ``_draw_extinction``), not a second implementation:
the per-panel maps are exactly what ``alcor_extinction_map`` and
``alcor_extinction_movie`` produce. It stays a script rather than a CLI because
a montage is a way of looking at a night, not a data product.

Usage: extinction_montage.py <night-dir-or-phot-csvs> -o OUT.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from skycam_utils.alcor import (
    ALCOR_EXT_CADENCE, ALCOR_EXT_MIN_ALTITUDE, ALCOR_EXT_NFRAMES,
    ALCOR_EXT_VMAX, _blocks, _draw_extinction, alcor_extinction_grid,
    alcor_extinction_stars, collect_alcor_photometry,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--nframes", type=int, default=ALCOR_EXT_NFRAMES)
    p.add_argument("--panels", type=int, default=8)
    p.add_argument("--vmax", type=float, default=ALCOR_EXT_VMAX)
    p.add_argument("--cmap", default="inferno_r")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    source = args.inputs[0] if len(args.inputs) == 1 else args.inputs
    df = collect_alcor_photometry(source).sort_values("OBSTIME")
    title = args.title or (Path(args.inputs[0]).name
                           if len(args.inputs) == 1 else "alcor extinction")

    all_blocks = list(_blocks(df, args.nframes))
    pick = np.linspace(0, len(all_blocks) - 1, args.panels).astype(int)
    print(f"{title}: {len(all_blocks)} blocks of {args.nframes}; "
          f"showing {len(pick)}")

    norm = Normalize(vmin=0.0, vmax=args.vmax)
    ncol = 4
    nrow = int(np.ceil(len(pick) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.4 * nrow),
                             subplot_kw={"projection": "polar"})
    axes = np.atleast_1d(axes).ravel()

    mesh = None
    for ax, idx in zip(axes, pick):
        stamp, sub = all_blocks[idx]
        measured, lost = alcor_extinction_stars(sub)
        if not len(measured):
            ax.set_visible(False)
            continue
        grid, az_values, alt_values = alcor_extinction_grid(measured)
        mesh = _draw_extinction(ax, measured, lost, grid, az_values,
                                alt_values, norm, args.cmap)
        ax.set_xticklabels(["N", "E", "S", "W"], fontsize=8)
        med = np.nanmedian(measured["ext"])
        ax.set_title(f"{pd.Timestamp(stamp).strftime('%H:%M')} UT\n"
                     f"{len(measured)} stars, med {med:+.2f}"
                     + (f", {len(lost)} lost" if len(lost) else ""),
                     fontsize=8.5, pad=8)
    for ax in axes[len(pick):]:
        ax.set_visible(False)

    fig.suptitle(f"{title} -- G-band cloud extinction, {args.nframes}-frame "
                 f"averages (~{args.nframes * ALCOR_EXT_CADENCE / 60:.0f} min)",
                 fontsize=12, y=0.99)
    cax = fig.add_axes([0.25, 0.075, 0.5, 0.014])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label("extinction [mag]      open circles: stars lost entirely "
                   "(lower limit)", fontsize=9, labelpad=4)
    cbar.ax.tick_params(labelsize=8)
    # wspace has to be generous or each panel's W label sits against the next
    # panel's E label.
    fig.subplots_adjust(top=0.90, bottom=0.13, hspace=0.35, wspace=0.30)

    fig.savefig(args.output, dpi=140)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
