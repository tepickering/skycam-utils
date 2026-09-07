#!/bin/bash
#
# Rebuild all five bad-pixel masks with the sky-restricted detector.
#
# create_badpix_mask now searches only pixels that alcor_badpix_search_region
# accepts: outside the (dilated) horizon mask, and outside a 15 px disc at the
# north celestial pole where Polaris' trail survives the night median. The
# masks in skycam_utils/data/badpix/ predate that and are ~78% skyline.
#
# ~10 min per night. Uses ~11 GB of scratch at a time, deleted after each.
#
# Usage:  claude_docs/scripts/rebuild_badpix_masks.sh
#         tail -f /private/tmp/alcor_rerun/badpix.log

set -u
export PATH="/Users/tim/conda/envs/skycam/bin:$PATH"

SCRATCH="${SCRATCH:-/private/tmp/alcor_rerun}"
LOG="$SCRATCH/badpix.log"
mkdir -p "$SCRATCH"

NIGHTS=(
    "$HOME/MMT/skycam_data/2024-09-04"
    "/Volumes/Samsung_4TB/skycam/2026-01-11"
    "/Volumes/Samsung_4TB/skycam/2026-03-11"
    "/Volumes/Samsung_4TB/skycam/2026-05-18"
    "/Volumes/Samsung_4TB/skycam/2026-06-09"
)

{
    echo "###### started $(date) ######"
    for d in "${NIGHTS[@]}"; do
        echo "--- $(basename "$d")  $(date +%H:%M:%S) ---"
        create_badpix_mask "$d" --scratch-dir "$SCRATCH"
    done
    echo "###### finished $(date) ######"
} > "$LOG" 2>&1

echo "done; log at $LOG"
