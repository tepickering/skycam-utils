#!/bin/bash

export PATH="/home/skycam/conda/envs/skycam/bin:$PATH"

datedir="$1"

mkdir -p /mnt/d/skycam/movies/$datedir

for i in `ls /mnt/d/skycam/$datedir/20*.jpg`; do echo "file '$i'"; done > /mnt/d/skycam/movie_input.txt
for i in `ls /mnt/d/skycam/$datedir/Unwrap*.jpg`; do echo "file '$i'"; done > /mnt/d/skycam/unwrap_input.txt

ffmpeg -f concat -safe 0 -i /mnt/d/skycam/movie_input.txt -c:v h264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -b:v 18000k /mnt/d/skycam/movies/$datedir/allsky.mp4
ffmpeg -f concat -safe 0 -i /mnt/d/skycam/unwrap_input.txt -c:v h264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -b:v 1200k /mnt/d/skycam/movies/$datedir/unwrap.mp4

# Process the night's FITS images: per-frame star photometry, the sky-brightness
# summary, the calibrated nighttime keogram, and the full-day raw RGB keogram
# that alcor_keogram used to build on its own. Products land in the day
# directory; per-frame *_phot.csv files that already exist (written by the
# real-time ingest) are reused rather than re-measured.
alcor_process_night /mnt/d/skycam/$datedir --pattern "*.fits.bz2" --day-keogram

# publish both keograms under keograms/<year>/{png,fits}/
year="${datedir:0:4}"
mkdir -p /mnt/d/skycam/keograms/$year/png /mnt/d/skycam/keograms/$year/fits

for kind in keogram sb_keogram; do
    cp /mnt/d/skycam/$datedir/${datedir}_${kind}.png /mnt/d/skycam/keograms/$year/png/
    cp /mnt/d/skycam/$datedir/${datedir}_${kind}.fits /mnt/d/skycam/keograms/$year/fits/
done

#rm /mnt/d/skycam/$datedir/20*.jpg
#rm /mnt/d/skycam/$datedir/Unwrap*.jpg
