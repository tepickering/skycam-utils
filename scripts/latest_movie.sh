#!/bin/bash

cur=`ls -d /mnt/d/skycam/20* | tail -1`

for i in `ls $cur/20*.jpg | tail -150`; do echo "file '$i'"; done > /mnt/d/skycam/ff_input.txt

ffmpeg -f concat -safe 0 -i /mnt/d/skycam/ff_input.txt -vcodec mpeg4 /var/www/html/skycam/tmp.mp4

mv /var/www/html/skycam/tmp.mp4 /var/www/html/skycam/latest_movie.mp4

