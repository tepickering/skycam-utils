#!/usr/bin/env python

import json
import os
import time

import redis

from astropy.time import Time
from astropy.coordinates import Angle, EarthLocation
import astropy.units as u


if 'REDISHOST' in os.environ:
    redis_host = os.environ['REDISHOST']
else:
    redis_host = 'redis.mmto.arizona.edu'

if 'REDISPORT' in os.environ:
    redis_port = os.environ['REDISPORT']
else:
    redis_port = 6379

if 'REDISPW' in os.environ:
    redis_server = redis.StrictRedis(
        host=redis_host,
        port=redis_port,
        password=os.environ['REDISPW'],
        db=0,
        decode_responses=True
    )
else:
    redis_server = redis.StrictRedis(host=redis_host, port=redis_port, db=0, decode_responses=True)

mmt = EarthLocation.from_geodetic("-110:53:04.4", "31:41:19.6", 2600 * u.m)

mmt_lat = mmt.lat.to(u.radian).value

while True:
    now = Time(Time.now(), location=mmt)

    # alcor uses the dublin JD, which is weird, but needs a 1.5 day offset to match example in the docs
    now_alcor = now.jd - 2415020 + 1.5

    mmt_ra = Angle(json.loads(redis_server.get('mount_mini_ra'))['value'], u.hourangle).to(u.radian).value
    mmt_dec = Angle(json.loads(redis_server.get('mount_mini_declination'))['value'], u.deg).to(u.radian).value
    mmt_lst = now.sidereal_time('apparent').to(u.radian).value
    mmt_catra = Angle(float(json.loads(redis_server.get('mount_mini_cat_ra2000'))['value']) * u.hourangle).to(u.radian).value
    mmt_catdec = Angle(float(json.loads(redis_server.get('mount_mini_cat_dec2000'))['value']) * u.deg).to(u.radian).value

    if mmt_catra < 0:
        mmt_ra = mmt_lst
        mmt_dec = mmt_lat

    try:
        with open("/mnt/d/skycam/mmt_position.txt", "w") as f:
            f.write(f"{now_alcor}\n{mmt_ra}\n{mmt_dec}\n")
    except Exception as e:
        print(e)

    time.sleep(5)
