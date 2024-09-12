#!/usr/bin/env python

import time
import os
import redis


def read_telemetry(filename="/mnt/c/Users/skycam/Documents/skywatch/Heater/DataOut_Weather.csv"):
    with open(filename, "rb") as f:
        last = f.readlines()[-1].decode().split(';')

    data = {
        'allsky_external_temp': float(last[2]),
        'allsky_internal_temp': float(last[3]),
        'allsky_hotside_temp': float(last[4]),
        'allsky_ccd_temp': float(last[5]),
        'allsky_external_rh': float(last[6]),
        'allsky_internal_rh': float(last[7])
    }

    return data


def publish_to_redis(redis_server, data):
    for key, value in data.items():
        redis_server.set(key, value)
        redis_server.publish(key, value)


if 'REDISHOST' in os.environ:
    redis_host = os.environ['REDISHOST']
else:
    redis_host = 'redis.mmto.arizona.edu'

if 'REDISPORT' in os.environ:
    redis_port = os.environ['REDISPORT']
else:
    redis_port = 6379

if 'REDISPW' in os.environ:
    redis_server = redis.StrictRedis(host=redis_host, port=redis_port, password=os.environ['REDISPW'], db=0)
else:
    redis_server = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

data = read_telemetry()
publish_to_redis(redis_server, data)

time.sleep(30)

data = read_telemetry()
publish_to_redis(redis_server, data)