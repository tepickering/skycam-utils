"""
Per-night clarity from stellar extinction, for selecting a clear-dark sample.

Sky brightness alone cannot grade a night: the darkest nights of 2025 are
overcast, because thick cloud at a dark site blocks starlight and airglow along
with the city light (see sb_year_summary.py). The stellar photometry is the
independent handle -- ``ext_g_ap`` = calibrated - catalog is a direct
line-of-sight extinction in mag, so a night's typical extinction says how clear
it was regardless of how bright or dark it read.

For each night this reduces the ``<night>_phot.csv`` rollup to a per-frame
median extinction (over the unbiased magnitude window, above the altitude
floor, variables excluded -- a variable has no catalog mag so its ext is NaN by
construction) and then to per-night statistics, and joins the sky-brightness
flicker alongside. Writes one tidy CSV; plotting lives in its caller.

Usage: night_clarity.py <products-dir> -o OUT.csv [--year 2025] [--workers N]
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.csv as pc

from skycam_utils.alcor import (
    ALCOR_EXT_MAG_WINDOW, ALCOR_EXT_MIN_ALTITUDE,
)

SUN_DARK = -18.0
MOON_DOWN = 0.0
MIN_FRAMES = 20

# Only these five columns are parsed out of a ~256 MB, 46-column rollup; the
# whole year is ~146 GB and a naive full-width read is the difference between
# minutes and hours.
COLS = ["OBSTIME", "altitude", "variable", "mag_g_ap", "ext_g_ap", "flux_g_ap"]


def robust_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def reduce_night(args):
    """One night -> a dict of clarity statistics (runs in a worker process)."""
    night, phot_path, sb_path = args
    try:
        tbl = pc.read_csv(phot_path, convert_options=pc.ConvertOptions(
            include_columns=COLS))
        df = tbl.to_pandas()
    except Exception as exc:
        return {"night": night, "error": f"{type(exc).__name__}: {exc}"}

    df["OBSTIME"] = pd.to_datetime(df["OBSTIME"])

    # Restrict to dark moonless frames, using the night's own sky_brightness.csv
    # for the Sun/Moon altitudes rather than recomputing them.
    try:
        sb = pd.read_csv(sb_path, usecols=["OBSTIME", "sun_alt", "moon_alt"])
        sb["OBSTIME"] = pd.to_datetime(sb["OBSTIME"])
        keep = sb.loc[(sb["sun_alt"] < SUN_DARK) & (sb["moon_alt"] < MOON_DOWN),
                      "OBSTIME"]
        df = df[df["OBSTIME"].isin(set(keep))]
    except Exception as exc:
        return {"night": night, "error": f"sb: {type(exc).__name__}: {exc}"}

    if not len(df):
        return {"night": night, "n_frames": 0}

    # Stars that could carry an extinction measurement at all: non-variable
    # (a variable has no catalog mag), above the altitude floor.
    usable = df[(~df["variable"].astype(bool))
                & (df["altitude"] >= ALCOR_EXT_MIN_ALTITUDE)]
    if not len(usable):
        return {"night": night, "n_frames": 0}

    # Lost = measured as a hard non-detection (flux exactly 0), which is the
    # strong-extinction signal the map carries as a lower limit.
    lost_frac = float((usable["flux_g_ap"] <= 0).mean())

    lo, hi = ALCOR_EXT_MAG_WINDOW
    sel = usable[(usable["mag_g_ap"] > lo) & (usable["mag_g_ap"] < hi)
                 & np.isfinite(usable["ext_g_ap"])]
    if not len(sel):
        return {"night": night, "n_frames": 0, "lost_frac": lost_frac}

    per_frame = sel.groupby("OBSTIME")["ext_g_ap"].agg(["median", "size"])
    per_frame = per_frame[per_frame["size"] >= 10]
    if len(per_frame) < MIN_FRAMES:
        return {"night": night, "n_frames": len(per_frame),
                "lost_frac": lost_frac}

    ext = per_frame["median"].to_numpy()
    return {
        "night": night,
        "n_frames": len(per_frame),
        "n_stars_med": float(per_frame["size"].median()),
        "ext_med": float(np.median(ext)),
        "ext_mean": float(np.mean(ext)),
        "ext_p90": float(np.percentile(ext, 90)),
        "ext_scatter": float(robust_std(ext)),
        "frac_frames_clear": float(np.mean(ext <= 0.05)),
        "lost_frac": lost_frac,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("products")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--year", default="2025")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    jobs = []
    for sb_path in sorted(glob.glob(os.path.join(
            args.products, f"{args.year}-*", "sky_brightness.csv"))):
        d = os.path.dirname(sb_path)
        night = os.path.basename(d)
        phot = os.path.join(d, f"{night}_phot.csv")
        if os.path.exists(phot) and os.path.getsize(phot) > 0:
            jobs.append((night, phot, sb_path))
    print(f"{len(jobs)} nights to reduce")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(reduce_night, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    tab = pd.DataFrame(rows).sort_values("night").reset_index(drop=True)
    bad = tab[tab.get("error").notna()] if "error" in tab else tab.iloc[:0]
    for _, r in bad.iterrows():
        print(f"  ERROR {r['night']}: {r['error']}")
    tab.to_csv(args.output, index=False)
    print(f"wrote {args.output}")

    ok = tab.dropna(subset=["ext_med"]) if "ext_med" in tab else tab.iloc[:0]
    if len(ok):
        clear = ok[ok["ext_med"] <= 0.05]
        print(f"\n{len(ok)} nights reduced; "
              f"{len(clear)} with median extinction <= 0.05 mag "
              f"({100 * len(clear) / len(ok):.0f}%)")
        print(f"  median of night medians: {ok['ext_med'].median():+.3f} mag")


if __name__ == "__main__":
    main()
