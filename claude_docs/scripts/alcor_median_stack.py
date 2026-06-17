#!/usr/bin/env python
"""Median-stack the dark (Sun < -18 deg) alcor frames from one night.

Motivation: median-combining a whole cloudy/foggy night accentuates the
contrast between the smooth, time-variable sky and the static, sharp-edged
horizon obstructions (buildings, terrain), giving a cleaner horizon outline
than star photometry alone.

Each frame is decompressed, bad-pixel-repaired (per-frame, using the epoch's
mask) as it is read, and collapsed to a luminance image (R+G+B), then written
into a disk-backed int32 memmap. The per-pixel median is computed in row chunks
to keep peak RAM bounded. Output (in claude_docs/gplots/) is a (ny, nx)
luminance median FITS image carrying the raw-frame alt/az WCS, plus a quicklook
PNG.

Usage:
    python alcor_median_stack.py [NIGHT_DIR] [--sun-alt-max -18] [--workers 10]
"""
import argparse
import glob
import sys
import time as _time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
from multiprocessing import Pool

from skycam_utils.alcor import (
    select_dark_frames,
    load_alcor_badpix_mask,
    _apply_badpix_repair,
    _filename_ut_datetime,
    build_alcor_wcs,
    alcor_calibration,
)

# claude_docs/gplots, relative to this script (claude_docs/scripts/).
GPLOTS = Path(__file__).resolve().parent.parent / "gplots"

_MASK = None    # bad-pixel mask, populated per-worker via initializer
_MEMMAP = None  # scratch memmap path, set in main and inherited by workers
_SHAPE = None   # (n, ny, nx), set in main and inherited by workers


def _init(mask, memmap, shape):
    global _MASK, _MEMMAP, _SHAPE
    _MASK, _MEMMAP, _SHAPE = mask, memmap, shape


def _read_repair(args):
    idx, path = args
    data = np.asarray(fits.getdata(path))               # (3, ny, nx) int16
    if _MASK is not None and _MASK.shape == data.shape:
        data = _apply_badpix_repair(data, _MASK)        # repair as read
    lum = data.astype(np.int32).sum(axis=0)             # (ny, nx) luminance
    mm = np.memmap(_MEMMAP, dtype=np.int32, mode="r+", shape=_SHAPE)
    mm[idx] = lum
    mm.flush()
    del mm
    return idx


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("night", nargs="?",
                   default="/Volumes/Seagate_24TB/skycam/2026-02-18",
                   help="night directory of *.fits.bz2 frames")
    p.add_argument("--sun-alt-max", type=float, default=-18.0)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--row-chunk", type=int, default=100)
    args = p.parse_args()

    night = Path(args.night)
    tag = night.name
    GPLOTS.mkdir(parents=True, exist_ok=True)
    out_fits = GPLOTS / f"{tag}_median.fits"
    out_png = GPLOTS / f"{tag}_median.png"
    # scratch memmap stays on the (large) data volume, not in the repo
    memmap = night.parent / f"{tag}_stack.dat"

    files = sorted(glob.glob(f"{night}/*.fits.bz2"))
    dark = [str(f) for f in select_dark_frames(
        files, sun_alt_max=args.sun_alt_max, moon_alt_max=90, log=print)]
    n = len(dark)
    if n == 0:
        sys.exit("no dark frames")

    # epoch / mask / wcs resolved once from the first dark frame's time
    t = Time(_filename_ut_datetime(dark[0]))
    mask, mask_date = load_alcor_badpix_mask(t)
    print(f"bad-pixel mask: {mask_date} "
          f"({0 if mask is None else int(mask.sum())} flagged pixels)")
    cal = alcor_calibration(t)
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])

    hdr0 = fits.getheader(dark[0])
    ny, nx = int(hdr0["NAXIS2"]), int(hdr0["NAXIS1"])
    shape = (n, ny, nx)
    print(f"stacking {n} luminance frames into memmap {shape} "
          f"({np.prod(shape) * 4 / 1e9:.1f} GB)")

    # allocate the memmap on disk
    np.memmap(memmap, dtype=np.int32, mode="w+", shape=shape).flush()

    t0 = _time.time()
    done = 0
    with Pool(args.workers, initializer=_init,
              initargs=(mask, str(memmap), shape)) as pool:
        for _ in pool.imap_unordered(_read_repair, list(enumerate(dark)),
                                     chunksize=4):
            done += 1
            if done % 100 == 0 or done == n:
                el = _time.time() - t0
                print(f"  read+repaired {done}/{n} "
                      f"({el:.0f}s, {done/el:.1f} fr/s)")

    # per-pixel median in row chunks (bounds peak RAM)
    print("computing median...")
    stack = np.memmap(memmap, dtype=np.int32, mode="r", shape=shape)
    med = np.empty((ny, nx), dtype=np.float32)
    for r0 in range(0, ny, args.row_chunk):
        r1 = min(r0 + args.row_chunk, ny)
        med[r0:r1, :] = np.median(
            stack[:, r0:r1, :].astype(np.float32), axis=0)
        print(f"  rows {r0}-{r1}/{ny}")
    del stack

    hdu = fits.PrimaryHDU(data=med, header=wcs.to_header(relax=True))
    hdu.header["NSTACK"] = (n, "frames median-combined")
    hdu.header["SUNALTMX"] = (args.sun_alt_max, "max Sun alt (deg) for inclusion")
    hdu.header["BADPXMSK"] = (str(mask_date), "bad-pixel mask epoch")
    hdu.writeto(out_fits, overwrite=True)
    print(f"wrote {out_fits}")

    # quicklook PNG (ZScale luminance, native orientation, origin=lower)
    vmin, vmax = ZScaleInterval().get_limits(med)
    fig, ax = plt.subplots(figsize=(10, 10 * ny / nx))
    ax.imshow(med, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(f"{tag}  median luminance  (N={n}, Sun<{args.sun_alt_max:g})")
    ax.set_xlabel("x (pix)")
    ax.set_ylabel("y (pix)")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")

    memmap.unlink()
    print("removed scratch memmap")


if __name__ == "__main__":
    main()
