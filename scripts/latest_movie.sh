#!/bin/bash

# SKYCAM_DATA_DIR: the archive's top directory (default /mnt/d/skycam, the
# WSL host; /data/skycam on Debian). Cron does not read shell profiles, so set
# it in the crontab:  SKYCAM_DATA_DIR=/data/skycam
D="${SKYCAM_DATA_DIR:-/mnt/d/skycam}"

cur=`ls -d $D/20* | tail -1`

for i in `ls $cur/20*.jpg | tail -150`; do echo "file '$i'"; done > $D/ff_input.txt

ffmpeg -r 10 -f concat -safe 0 -i $D/ff_input.txt -c:v h264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -b:v 18000k /var/www/html/skycam/tmp.mp4

mv /var/www/html/skycam/tmp.mp4 /var/www/html/skycam/latest_movie.mp4
