"""
Frame list for the clear-night median sky-brightness map.

``clear_dark_sample.py`` names the nights that graded clear; this expands them
into the individual frames the map is built from, reading each night's
``sky_brightness.csv`` (which already carries the per-frame filename, exposure
and Sun/Moon altitudes, so no raw frame is opened here).

The list is written **self-contained** on purpose: the machine that builds the
map needs only the raw archive and this file, not the products tree. Paths are
stored as ``night`` + ``filename`` rather than absolute paths, since the raw
archive is mounted somewhere different on every host.

Frames are kept when Sun < -18 and Moon < 0, the same dark-moonless cut the
sky-brightness statistics use. Sub-sampling is deliberately NOT done here --
the list is written at full density and ``sky_median_map.py --stride`` thins
it, so the stride can be changed without regenerating the list.

Usage: clear_frame_list.py <sample.csv> <products-dir> -o OUT.csv.gz
"""

import argparse
import os
import sys

import pandas as pd

SUN_DARK = -18.0
MOON_DOWN = 0.0

COLS = ["filename", "OBSTIME", "exposure", "sun_alt", "moon_alt"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sample", help="clear_dark_sample.csv")
    p.add_argument("products", help="products directory (<night>/sky_brightness.csv)")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    nights = sorted(set(pd.read_csv(args.sample)["night"].astype(str)))
    print(f"{len(nights)} clear nights", file=sys.stderr)

    out = []
    missing = []
    for night in nights:
        path = os.path.join(args.products, night, "sky_brightness.csv")
        if not os.path.exists(path):
            missing.append(night)
            continue
        tab = pd.read_csv(path, usecols=COLS)
        dark = tab[(tab["sun_alt"] < SUN_DARK) & (tab["moon_alt"] < MOON_DOWN)].copy()
        dark.insert(0, "night", night)
        out.append(dark)

    if missing:
        print(f"WARNING: no sky_brightness.csv for {len(missing)} nights: "
              f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}",
              file=sys.stderr)
    if not out:
        raise SystemExit("no frames found")

    frames = pd.concat(out, ignore_index=True)
    frames.to_csv(args.output, index=False, compression="infer")

    per_night = frames.groupby("night").size()
    print(f"wrote {args.output}: {len(frames):,} dark moonless frames "
          f"over {len(per_night)} nights "
          f"({per_night.min()}-{per_night.max()} per night, "
          f"median {int(per_night.median())})", file=sys.stderr)


if __name__ == "__main__":
    main()
