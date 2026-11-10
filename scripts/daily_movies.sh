#!/bin/bash

# SKYCAM_DATA_DIR: the archive's top directory (default /mnt/d/skycam, the
# WSL host; /data/skycam on Debian). Cron does not read shell profiles, so set
# it in the crontab:  SKYCAM_DATA_DIR=/data/skycam
D="${SKYCAM_DATA_DIR:-/mnt/d/skycam}"

cur=`ls -d $D/20* | tail -2 | head -1 | xargs basename`

/home/skycam/skycam-utils/scripts/make_movies.sh $cur
