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

With ``--from-raw`` the list is instead built straight from raw night
directories, for nights that were never run through ``alcor_process_night`` and
so have no ``sky_brightness.csv`` -- the hand-picked calibration nights, for
instance. Times come from the filenames and the Sun/Moon altitudes are
recomputed. Exposure is SAMPLED rather than assumed: the camera auto-exposes
through twilight, but it is pinned at its 20 s ceiling once the sky is dark, and
a wrong exposure is a direct error in surface brightness, so the script checks a
sample of each night's dark frames and refuses to guess if they disagree.

Usage:
  clear_frame_list.py <sample.csv> <products-dir> -o OUT.csv.gz
  clear_frame_list.py --from-raw <night-dir> [...] -o OUT.csv.gz
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

SUN_DARK = -18.0
MOON_DOWN = 0.0

COLS = ["filename", "OBSTIME", "exposure", "sun_alt", "moon_alt"]

EXPOSURE_SAMPLE = 25          # every Nth dark frame, when checking exposure


def _from_raw(night_dirs, pattern="*.fits.bz2"):
    """Frame list for nights with no products, straight off the raw archive."""
    import glob

    import astropy.units as u
    from astropy.coordinates import AltAz, get_body, get_sun
    from astropy.time import Time

    from skycam_utils.alcor.timeutils import (
        _filename_ut_datetime, _read_frame_exposure,
    )
    from skycam_utils.astrometry import MMT_LOCATION

    out = []
    for d in night_dirs:
        night = os.path.basename(os.path.normpath(d))
        files = sorted(glob.glob(os.path.join(d, pattern)))
        stamps = [_filename_ut_datetime(os.path.basename(f).replace(".fits.bz2", ""))
                  for f in files]
        good = [i for i, s in enumerate(stamps) if s is not None]
        if not good:
            print(f"WARNING: {night}: no parseable filenames", file=sys.stderr)
            continue

        t = Time([stamps[i] for i in good], scale="utc")
        frame = AltAz(obstime=t, location=MMT_LOCATION)
        sun = get_sun(t).transform_to(frame).alt.deg
        moon = get_body("moon", t).transform_to(frame).alt.deg
        dark = (sun < SUN_DARK) & (moon < MOON_DOWN)
        if not dark.any():
            print(f"WARNING: {night}: no dark moonless frames", file=sys.stderr)
            continue

        keep = [(files[i], t[j], sun[j], moon[j])
                for j, i in enumerate(good) if dark[j]]

        # The camera auto-exposes through twilight and pins at 20 s once dark.
        # Sample rather than trust it: surface brightness scales directly with
        # exposure, so a wrong value is a straight magnitude error.
        sampled = sorted({_read_frame_exposure(f)
                          for f, *_ in keep[::EXPOSURE_SAMPLE]})
        if len(sampled) != 1:
            raise SystemExit(
                f"{night}: dark-frame exposure is not constant ({sampled}); "
                "build this night's list from its sky_brightness.csv instead")
        exposure = sampled[0]
        print(f"{night}: {len(keep)} dark moonless frames, exposure "
              f"{exposure} s ({len(keep[::EXPOSURE_SAMPLE])} sampled)",
              file=sys.stderr)

        out.append(pd.DataFrame({
            "night": night,
            "filename": [os.path.basename(f) for f, *_ in keep],
            "OBSTIME": [tt.isot.replace("T", " ") for _, tt, _, _ in keep],
            "exposure": exposure,
            "sun_alt": [s for *_, s, _ in keep],
            "moon_alt": [m for *_, m in keep],
        }))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sample", nargs="?", help="clear_dark_sample.csv")
    p.add_argument("products", nargs="?",
                   help="products directory (<night>/sky_brightness.csv)")
    p.add_argument("--from-raw", nargs="+", metavar="NIGHT_DIR", default=None,
                   help="build from raw night directories instead")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args()

    if args.from_raw:
        out = _from_raw(args.from_raw)
        if not out:
            raise SystemExit("no frames found")
        frames = pd.concat(out, ignore_index=True)
        frames.to_csv(args.output, index=False, compression="infer")
        print(f"wrote {args.output}: {len(frames):,} dark moonless frames "
              f"over {frames['night'].nunique()} nights", file=sys.stderr)
        return

    if not (args.sample and args.products):
        raise SystemExit("need <sample.csv> <products-dir>, or --from-raw")

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
