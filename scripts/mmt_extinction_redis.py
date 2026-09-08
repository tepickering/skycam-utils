#!/usr/bin/env python
"""
Publish the cloud extinction at the MMT's current pointing to redis.

Reads the most recent alcor extinction map (written by ``alcor_extinction_map``,
normally from the same cron that runs every five minutes), looks up the
telescope's current (az, alt) in it, and publishes the result.

This is an operational script, not part of the installed package: it assumes the
skycam host's paths and the MMT redis conventions, and it is the only place
``redis`` is imported. Run it from cron AFTER the map is built.

Published keys:

    allsky_extinction         mag of G-band extinction at the pointing
    allsky_extinction_az      azimuth used (deg)
    allsky_extinction_alt     altitude used (deg)
    allsky_extinction_age     seconds between the map's midpoint and now
    allsky_extinction_zenith  extinction at the zenith, as a sky-wide reference

A value that cannot be measured -- the telescope below the mapped altitude, or
pointing into a region the map had to blank -- is published as the string
"nan" rather than omitted, so a consumer can tell "no data" from "stale data".
"""

import argparse
import json
import os
import sys
from pathlib import Path

import redis

import astropy.units as u
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time

from skycam_utils.alcor import ALCOR_EXT_LOOKUP_RADIUS, alcor_extinction_at

MMT = EarthLocation.from_geodetic("-110:53:04.4", "31:41:19.6", 2600 * u.m)

# Where the cron drops the rolling map. A single file that is overwritten each
# run, rather than one file per map: a raw-frame map is ~8 MB, so keeping every
# five-minute map would be ~2.3 GB/day. The night's history lives in the
# animation instead.
DEFAULT_MAP = "/mnt/d/skycam/alcor_extinction_latest.fits"

MAX_AGE_DEFAULT = 1800.0     # s; publish nothing from a map older than this


def open_redis():
    """Open the MMT redis, honouring the usual environment overrides."""
    host = os.environ.get("REDISHOST", "redis.mmto.arizona.edu")
    port = int(os.environ.get("REDISPORT", 6379))   # env vars are strings
    password = os.environ.get("REDISPW")
    try:
        if password:
            return redis.StrictRedis(host=host, port=port, password=password,
                                     db=0, decode_responses=True)
        return redis.StrictRedis(host=host, port=port, db=0,
                                 decode_responses=True)
    except Exception as err:                        # noqa: BLE001
        print(f"could not open redis: {err}", file=sys.stderr)
        return None


def mmt_altaz(redis_server, time=None):
    """
    The telescope's current horizontal coordinates, from the TCS RA/Dec in
    redis. Returns ``(az, alt)`` in degrees, or ``None`` when the mount is not
    reporting a real catalog position (``cat_ra2000 < 0``), which is how
    mmt_position.py already detects a parked/undefined pointing.
    """
    time = Time.now() if time is None else time
    try:
        cat_ra = float(json.loads(
            redis_server.get("mount_mini_cat_ra2000"))["value"])
        if cat_ra < 0:
            return None
        ra = Angle(json.loads(redis_server.get("mount_mini_ra"))["value"],
                   u.hourangle)
        dec = Angle(json.loads(
            redis_server.get("mount_mini_declination"))["value"], u.deg)
    except (TypeError, ValueError, KeyError) as err:
        print(f"could not read the mount position: {err}", file=sys.stderr)
        return None

    altaz = SkyCoord(ra=ra, dec=dec).transform_to(
        AltAz(obstime=time, location=MMT))
    return float(altaz.az.deg), float(altaz.alt.deg)


def publish(redis_server, values):
    for key, value in values.items():
        redis_server.set(key, value)
        redis_server.publish(key, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=DEFAULT_MAP,
                        help="Extinction map FITS written by alcor_extinction_map.")
    parser.add_argument("--radius", type=float, default=ALCOR_EXT_LOOKUP_RADIUS,
                        help="Great-circle lookup radius in DEGREES.")
    parser.add_argument("--max-age", type=float, default=MAX_AGE_DEFAULT,
                        help="Refuse to publish from a map older than this (s).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be published and exit.")
    args = parser.parse_args()

    path = Path(args.map)
    if not path.exists():
        print(f"no extinction map at {path}", file=sys.stderr)
        return 1

    from astropy.io import fits
    mid = Time(fits.getval(path, "TMID"))
    age = (Time.now() - mid).sec
    if age > args.max_age:
        print(f"map is {age:.0f} s old (limit {args.max_age:g}); not publishing",
              file=sys.stderr)
        return 1

    redis_server = None if args.dry_run else open_redis()
    if redis_server is None and not args.dry_run:
        return 1

    zenith = alcor_extinction_at(path, 0.0, 90.0, radius=args.radius)
    values = {
        "allsky_extinction_age": round(age, 1),
        "allsky_extinction_zenith": f"{zenith:.3f}",
    }

    pointing = mmt_altaz(redis_server) if redis_server is not None else None
    if pointing is None:
        # Not an error: the telescope may simply be parked. Publish the zenith
        # reference and an explicit nan so consumers can distinguish this from
        # a stale map.
        values["allsky_extinction"] = "nan"
    else:
        az, alt = pointing
        value = alcor_extinction_at(path, az, alt, radius=args.radius)
        values["allsky_extinction"] = f"{value:.3f}"
        values["allsky_extinction_az"] = f"{az:.2f}"
        values["allsky_extinction_alt"] = f"{alt:.2f}"

    if args.dry_run:
        for key, value in values.items():
            print(f"{key} = {value}")
        return 0

    publish(redis_server, values)
    return 0


if __name__ == "__main__":
    sys.exit(main())
