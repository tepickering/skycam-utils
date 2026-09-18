"""
Per-night throughput offset, for correcting the median sky-brightness map.

The acrylic dome soils and is washed, so the system throughput wanders by up
to ~0.11 mag over a year (see the throughput-excursion note in CLAUDE.md).
That lands in surface brightness exactly as it lands in stellar photometry:

    mu_measured = mu_true + ext_med      =>      mu_true = mu_measured - ext_med

with ``ext_med`` the night's median stellar extinction. Extinction cannot
really be negative, so a negative ``ext_med`` means the throughput was higher
than the zeropoint assumes and the sky reads brighter than it was.

``night_clarity.py`` computes the same statistic for nights that have a
``sky_brightness.csv``; this one also works on a bare raw night directory, so
the hand-picked calibration nights can be corrected too. It accepts both
photometry schemas -- ``ext_g_ap`` from ``--both`` runs, and plain ``ext_g``
from aperture-only runs.

Output is a CSV of ``night,ext_med,ext_scatter,n_frames``, which
``sky_median_map.py --grades`` consumes directly. ``night_clarity.py`` output
works there too, since it carries the same two columns.

Usage: night_throughput.py <night-dir> [...] -o OUT.csv [--stride 3]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, get_body, get_sun
from astropy.time import Time

from skycam_utils.alcor.config import ALCOR_EXT_MAG_WINDOW, ALCOR_EXT_MIN_ALTITUDE
from skycam_utils.alcor.timeutils import _filename_ut_datetime
from skycam_utils.astrometry import MMT_LOCATION

SUN_DARK = -18.0
MOON_DOWN = 0.0
MIN_STARS = 10          # per frame, before its median extinction is trusted

SCHEMAS = (("mag_g_ap", "ext_g_ap", "flux_g_ap"), ("mag_g", "ext_g", "flux_g"))


def robust_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _schema(path):
    head = pd.read_csv(path, nrows=0).columns
    for cols in SCHEMAS:
        if all(c in head for c in cols):
            return cols
    raise SystemExit(f"{path}: no recognised photometry columns ({list(head)})")


def night_offset(night_dir, stride):
    night = os.path.basename(os.path.normpath(night_dir))
    files = sorted(glob.glob(os.path.join(night_dir, "*_phot.csv")))
    files = [f for f in files if not os.path.basename(f).startswith(night)]
    if not files:
        print(f"WARNING: {night}: no per-frame photometry", file=sys.stderr)
        return None

    stamps = [_filename_ut_datetime(os.path.basename(f).replace("_phot.csv", ""))
              for f in files]
    good = [i for i, s in enumerate(stamps) if s is not None]
    t = Time([stamps[i] for i in good], scale="utc")
    frame = AltAz(obstime=t, location=MMT_LOCATION)
    dark = ((get_sun(t).transform_to(frame).alt.deg < SUN_DARK)
            & (get_body("moon", t).transform_to(frame).alt.deg < MOON_DOWN))
    keep = [files[i] for i, d in zip(good, dark) if d]
    if not keep:
        print(f"WARNING: {night}: no dark moonless frames", file=sys.stderr)
        return None

    mag_c, ext_c, _ = _schema(keep[0])
    lo, hi = ALCOR_EXT_MAG_WINDOW
    per_frame = []
    for f in keep[::stride]:
        d = pd.read_csv(f, usecols=["altitude", "variable", mag_c, ext_c])
        # A variable has no catalog magnitude, so its extinction is NaN by
        # construction; the altitude floor is the same one the maps use.
        u = d[(~d["variable"].astype(bool))
              & (d["altitude"] >= ALCOR_EXT_MIN_ALTITUDE)]
        s = u[(u[mag_c] > lo) & (u[mag_c] < hi) & np.isfinite(u[ext_c])]
        if len(s) >= MIN_STARS:
            per_frame.append(float(s[ext_c].median()))

    if not per_frame:
        print(f"WARNING: {night}: no usable frames", file=sys.stderr)
        return None

    e = np.asarray(per_frame)
    return {"night": night, "ext_med": float(np.median(e)),
            "ext_scatter": float(robust_std(e)), "n_frames": len(e)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("nights", nargs="+", help="night directories with *_phot.csv")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--stride", type=int, default=3, help="every Nth dark frame")
    args = p.parse_args()

    rows = [r for d in args.nights if (r := night_offset(d, args.stride))]
    if not rows:
        raise SystemExit("no nights measured")

    tab = pd.DataFrame(rows).sort_values("night")
    tab.to_csv(args.output, index=False)
    for r in tab.itertuples():
        print(f"{r.night}: ext_med {r.ext_med:+.3f}  scatter {r.ext_scatter:.4f}  "
              f"({r.n_frames} frames)  ->  map corrected by {-r.ext_med:+.3f} mag",
              file=sys.stderr)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
