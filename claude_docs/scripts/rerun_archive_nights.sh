#!/bin/bash
#
# Overnight re-run of the five archived alcor nights.
#
# Why a re-run is needed:
#   1. sky_brightness.csv gained the allsky_mv_best / best_az / best_alt columns
#      (darkest 5 deg cone above alt 30), so the existing CSVs are stale.
#   2. 2026-01-11, 2026-03-11 and 2026-06-09 have no bad-pixel mask of their own
#      and fall back to the 2026-05-18 mask, which does not flag a pixel that was
#      hot months earlier. It shows up as a bright horizontal line at alt ~31.7 N
#      in those nights' calibrated keograms (0.19 mag above the local sky), and
#      as a stronger one near the S horizon on 2026-06-09 (0.66 mag). Building
#      each night its own mask removes it. All three have enough Sun<-18/Moon<-6
#      dark frames (877 / 850 / 548, need >=500) -- verified.
#
# Masks MUST be built before the re-processing pass, since load_alcor_fits
# resolves the nearest-in-date mask at load time.
#
# Runtime: ~30 min for the three masks, then ~2 h for the five nights (the
# per-frame *_phot.csv files all exist, so photometry is skipped throughout).
# Uses all cores. ~11 GB of scratch at a time, deleted after each mask.
#
# New masks land in the packaged skycam_utils/data/badpix/ and will show up as
# untracked files to review and commit.
#
# Usage:  claude_docs/scripts/rerun_archive_nights.sh
#         tail -f /private/tmp/alcor_rerun/rerun.log

set -u

export PATH="/Users/tim/conda/envs/skycam/bin:$PATH"

SCRATCH="${SCRATCH:-/private/tmp/alcor_rerun}"
LOG="$SCRATCH/rerun.log"
mkdir -p "$SCRATCH"

LOCAL_ROOT="$HOME/MMT/skycam_data"
EXT_ROOT="/Volumes/Samsung_4TB/skycam"

NIGHTS=(
    "$LOCAL_ROOT/2024-09-04"
    "$EXT_ROOT/2026-01-11"
    "$EXT_ROOT/2026-03-11"
    "$EXT_ROOT/2026-05-18"
    "$EXT_ROOT/2026-06-09"
)

# nights lacking a bad-pixel mask of their own
NEED_MASK=(
    "$EXT_ROOT/2026-01-11"
    "$EXT_ROOT/2026-03-11"
    "$EXT_ROOT/2026-06-09"
)

{
    echo "###### started $(date) ######"

    if [ ! -d "$EXT_ROOT" ]; then
        echo "FATAL: $EXT_ROOT is not mounted; plug in the Samsung_4TB drive."
        exit 1
    fi

    echo "###### phase 1: bad-pixel masks ######"
    for d in "${NEED_MASK[@]}"; do
        echo "--- create_badpix_mask $(basename "$d")  $(date +%H:%M:%S) ---"
        create_badpix_mask "$d" --scratch-dir "$SCRATCH"
    done

    echo "###### phase 2: re-process nights ######"
    for d in "${NIGHTS[@]}"; do
        echo "--- $(basename "$d")  $(date +%H:%M:%S) ---"
        alcor_process_night "$d" --day-keogram 2>&1 | grep -v '^\['
        echo "--- done $(basename "$d")  $(date +%H:%M:%S) ---"
    done

    echo "###### finished $(date) ######"
} > "$LOG" 2>&1

echo "done; log at $LOG"
