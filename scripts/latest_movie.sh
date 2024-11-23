#!/bin/bash

cur=`ls -d /mnt/d/skycam/20* | tail -1`

for i in `ls $cur/20*.jpg | tail -150`; do echo "file '$i'"; done > /mnt/d/skycam/ff_input.txt

ffmpeg -f concat -safe 0 -i /mnt/d/skycam/ff_input.txt -c:v h264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -b:v 18000k fps=20 /var/www/html/skycam/tmp.mp4

mv /var/www/html/skycam/tmp.mp4 /var/www/html/skycam/latest_movie.mp4
