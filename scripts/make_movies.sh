#!/bin/bash

datedir="$1"

mkdir -p /mnt/d/skycam/movies/$datedir

for i in `ls /mnt/d/skycam/$datedir/20*.jpg`; do echo "file '$i'"; done > /mnt/d/skycam/movie_input.txt
for i in `ls /mnt/d/skycam/$datedir/Unwrap*.jpg`; do echo "file '$i'"; done > /mnt/d/skycam/unwrap_input.txt

ffmpeg -f concat -safe 0 -i /mnt/d/skycam/movie_input.txt -c:v h264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -b:v 18000k /mnt/d/skycam/movies/$datedir/allsky.mp4
ffmpeg -f concat -safe 0 -i /mnt/d/skycam/unwrap_input.txt -c:v h264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -b:v 1200k /mnt/d/skycam/movies/$datedir/unwrap.mp4

rm /mnt/d/skycam/$datedir/20*.jpg
rm /mnt/d/skycam/$datedir/Unwrap*.jpg
