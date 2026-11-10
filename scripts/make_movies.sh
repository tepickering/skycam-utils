#!/bin/bash

export PATH="/home/skycam/conda/envs/skycam/bin:$PATH"

# SKYCAM_DATA_DIR: the archive's top directory (default /mnt/d/skycam, the
# WSL host; /data/skycam on Debian). Cron does not read shell profiles, so set
# it in the crontab:  SKYCAM_DATA_DIR=/data/skycam
D="${SKYCAM_DATA_DIR:-/mnt/d/skycam}"

datedir="$1"

mkdir -p $D/movies/$datedir

for i in `ls $D/$datedir/20*.jpg`; do echo "file '$i'"; done > $D/movie_input.txt
for i in `ls $D/$datedir/Unwrap*.jpg`; do echo "file '$i'"; done > $D/unwrap_input.txt

ffmpeg -f concat -safe 0 -i $D/movie_input.txt -c:v h264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -b:v 18000k $D/movies/$datedir/allsky.mp4
ffmpeg -f concat -safe 0 -i $D/unwrap_input.txt -c:v h264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -b:v 1200k $D/movies/$datedir/unwrap.mp4

# Process the night's FITS images: per-frame star photometry, the sky-brightness
# summary, the calibrated nighttime keogram, and the full-day raw RGB keogram
# that alcor_keogram used to build on its own. Products land in the day
# directory; per-frame *_phot.csv files that already exist (written by the
# real-time ingest) are reused rather than re-measured.
alcor_process_night $D/$datedir --pattern "*.fits.bz2" --day-keogram

# publish both keograms under keograms/<year>/{png,fits}/
year="${datedir:0:4}"
mkdir -p $D/keograms/$year/png $D/keograms/$year/fits

for kind in keogram sb_keogram; do
    cp $D/$datedir/${datedir}_${kind}.png $D/keograms/$year/png/
    cp $D/$datedir/${datedir}_${kind}.fits $D/keograms/$year/fits/
done

#rm $D/$datedir/20*.jpg
#rm $D/$datedir/Unwrap*.jpg
