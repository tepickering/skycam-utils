"""
Hot-pixel persistence across an archive of night medians.

The analysis behind ``create_badpix_consensus_masks``. Stage 1 re-runs the
bad-pixel detector on every ``<night>/<night>_median.fits`` under a products
tree and saves each night's candidates sparsely (channel, y, x, z, ADU excess,
plus the night's per-channel noise sigma); stage 2 reports:

* persistence -- the fraction of nights each (c, y, x) is flagged. Over 644
  nights (2025-01..2026-10): 99k ever flagged, ~1000 on more than half the
  nights, ~98k on fewer than 5%;
* where the transients sit -- median altitude ~5 deg, piled toward the Tucson
  (N) and Nogales/building (S-SW) azimuths: sky structure above the dilated
  horizon mask, not sensor;
* moonlight sensitivity -- nightly hits on the long-lived core vs the G noise
  sigma, which is why only dark nights vote;
* monthly consensus size and churn, i.e. the aging rate.

Usage::

    python badpix_persistence.py extract <products-dir> <cand-dir> [--workers 8]
    python badpix_persistence.py report <cand-dir>
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from skycam_utils.alcor import (ALCOR_BADPIX_SIGMA_MAX, alcor_badpix_search_region,
                                alcor_calibration, build_alcor_wcs)
from skycam_utils.alcor.badpix import _badpix_zscores

ZMIN = 8.0          # keep candidates down to the footprint-growth threshold


def _wcs_for(night):
    cal = alcor_calibration(Time(night.isoformat()))
    return build_alcor_wcs(**{k: cal[k] for k in (
        "xcen", "ycen", "rotation", "radial_coeffs", "horizon_radius",
        "tangential_coeffs", "axis_tilt")})


def _extract_one(task):
    median_path, out = task
    if out.exists():
        return out.name
    night = date.fromisoformat(median_path.parent.name)
    cube = np.asarray(fits.getdata(median_path), dtype=float)
    valid = alcor_badpix_search_region(cube.shape[1:], time=night, wcs=_wcs_for(night))
    z, sig = _badpix_zscores(cube)
    c, y, x = np.nonzero((z > ZMIN) & valid[None])
    np.savez_compressed(out, c=c.astype(np.uint8), y=y.astype(np.int16),
                        x=x.astype(np.int16), z=z[c, y, x].astype(np.float32),
                        sig=sig, shape=np.array(cube.shape))
    return out.name


def extract(products, cand_dir, workers):
    cand_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(p, cand_dir / f"{p.parent.name}.npz")
             for p in sorted(products.glob("*/*_median.fits"))]
    with ProcessPoolExecutor(workers) as ex:
        for i, name in enumerate(ex.map(_extract_one, tasks)):
            print(i, name, flush=True)


def _seed_flags(d):
    """Production rule: z > 25 in at most two of three channels."""
    shape = tuple(d["shape"])
    c, y, x, z = d["c"], d["y"], d["x"], d["z"]
    hot = z > 25
    pix = np.ravel_multi_index((y, x), shape[1:])
    u, n = np.unique(pix[hot], return_counts=True)
    keep = hot & ~np.isin(pix, u[n == 3])
    return np.ravel_multi_index((c[keep], y[keep], x[keep]), shape)


def report(cand_dir):
    files = sorted(cand_dir.glob("*.npz"))
    nights = pd.to_datetime([f.stem for f in files])
    flags, sig = [], []
    for f in files:
        d = np.load(f)
        flags.append(_seed_flags(d))
        sig.append(float(d["sig"][1]))
        shape = tuple(d["shape"])
    sig = np.array(sig)
    allpix = np.unique(np.concatenate(flags))
    M = np.zeros((allpix.size, len(files)), dtype=bool)
    for j, fl in enumerate(flags):
        M[np.searchsorted(allpix, fl), j] = True
    frac = M.mean(1)

    print(f"{len(files)} nights, {allpix.size} (c,y,x) ever flagged")
    for lo, hi in [(0, .05), (.05, .2), (.2, .5), (.5, .8), (.8, 1.01)]:
        print(f"  flagged on {lo:4.2f}-{hi:4.2f} of nights: "
              f"{((frac >= lo) & (frac < hi)).sum()}")

    c, y, x = np.unravel_index(allpix, shape)
    az, alt = _wcs_for(nights[len(nights) // 2].date()).pixel_to_world_values(x, y)
    for name, sel in (("transient (<5%)", frac < .05), ("persistent (>50%)", frac > .5)):
        h, _ = np.histogram(az[sel] % 360, bins=12, range=(0, 360))
        print(f"{name}: alt p5/50/95 {np.percentile(alt[sel], [5, 50, 95]).round(1)}, "
              f"RGB {np.bincount(c[sel], minlength=3)}, az/30deg {h}")

    core = frac > 0.8
    hits = M[core].sum(0)
    print(f"\ncore ({core.sum()} px): nightly hits vs G sigma")
    for lo, hi in [(0, 7), (7, 9), (9, 12), (12, 16), (16, 99)]:
        s = (sig > lo) & (sig <= hi)
        if s.any():
            print(f"  sigma {lo:2d}-{hi:2d}: {s.sum():3d} nights, median hits {np.median(hits[s]):.0f}")

    month = nights.to_period("M")
    dark = sig <= ALCOR_BADPIX_SIGMA_MAX
    print("\nmonthly consensus (>= 0.5), all nights vs dark nights only:")
    prev = None
    for m in month.unique():
        row = []
        for sel in ((month == m), (month == m) & dark):
            j = np.flatnonzero(sel)
            row.append((j.size, int((M[:, j].mean(1) >= 0.5).sum()) if j.size else 0))
        on = M[:, (month == m) & dark].mean(1) >= 0.5
        churn = "" if prev is None else f"  lost {(prev & ~on).sum():3d} new {(on & ~prev).sum():3d}"
        prev = on
        print(f"  {m}: all {row[0][0]:2d} n -> {row[0][1]:5d}   dark {row[1][0]:2d} n -> {row[1][1]:5d}{churn}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("products", type=Path)
    e.add_argument("cand_dir", type=Path)
    e.add_argument("--workers", type=int, default=8)
    r = sub.add_parser("report")
    r.add_argument("cand_dir", type=Path)
    args = ap.parse_args()
    if args.cmd == "extract":
        extract(args.products, args.cand_dir, args.workers)
    else:
        report(args.cand_dir)
