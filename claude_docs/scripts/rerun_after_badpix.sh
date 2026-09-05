#!/bin/bash
#
# Re-process the five archived nights after the bad-pixel masks were rebuilt
# with the sky-restricted detector. Waits for rebuild_badpix_masks.sh to finish
# first, since load_alcor_fits resolves the mask at load time.
#
# Why --reprocess: the per-frame *_phot.csv files were measured against the OLD
# masks, so the resume path would leave the photometry inconsistent with the
# frames the SB maps and keograms are built from. The new masks are a subset of
# the old (only terrain and the celestial pole were dropped) so stars above
# alt 20 barely move, but the products should be self-consistent, not nearly so.
#
# Why --median-stack: NBAD*/NSTACK in <night>_median.fits are now a meaningful
# CMOS-aging metric rather than a count of horizon edges, and the medians are
# the diagnostic we keep coming back to. ~19 GB of scratch per night, one at a
# time (76 GB free), ~24 MB of output each.
#
# Also picks up the keogram change: the SB keogram column now comes from the
# un-horizon-masked map, so both keograms span the full sensor column down to
# alt ~-6 at each end, with a ROWALT extension and horizon marks.
#
# Usage:  claude_docs/scripts/rerun_after_badpix.sh
#         tail -f /private/tmp/alcor_rerun/rerun2.log

set -u
export PATH="/Users/tim/conda/envs/skycam/bin:$PATH"

SCRATCH="${SCRATCH:-/private/tmp/alcor_rerun}"
LOG="$SCRATCH/rerun2.log"
MASKLOG="$SCRATCH/badpix.log"
mkdir -p "$SCRATCH"

NIGHTS=(
    "$HOME/MMT/skycam_data/2024-09-04"
    "/Volumes/Samsung_4TB/skycam/2026-01-11"
    "/Volumes/Samsung_4TB/skycam/2026-03-11"
    "/Volumes/Samsung_4TB/skycam/2026-05-18"
    "/Volumes/Samsung_4TB/skycam/2026-06-09"
)

{
    echo "###### waiting for the mask rebuild ######"
    while ! grep -q '^###### finished' "$MASKLOG" 2>/dev/null; do sleep 60; done
    echo "masks done $(date); shipped set:"
    ls -la "$(dirname "$0")"/../../skycam_utils/data/badpix/

    echo "###### re-processing nights ######"
    for d in "${NIGHTS[@]}"; do
        echo "--- $(basename "$d")  $(date +%H:%M:%S) ---"
        alcor_process_night "$d" --day-keogram --median-stack --reprocess \
            --scratch-dir "$SCRATCH" 2>&1 | grep -v '^\['
        echo "--- done $(basename "$d")  $(date +%H:%M:%S) ---"
    done
    echo "###### finished $(date) ######"
} > "$LOG" 2>&1

echo "done; log at $LOG"
