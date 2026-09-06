#!/bin/bash
#
# Re-process the five archived alcor nights. This is the single canonical
# re-run script; it supersedes the earlier rerun_after_badpix.sh (which chained
# off a bad-pixel mask rebuild that has since been done).
#
# Why this run:
#   The per-frame *_phot.csv files predate the bright-variable catalog
#   (bright_variable_vsx.fits, 637 stars, measured by default since 0203e94),
#   so they carry no `variable` column and no variable rows.
#
# Why --reprocess is REQUIRED, not optional:
#   alcor_process_night's resume path reuses any existing non-empty
#   <frame>_phot.csv without re-measuring. Those files are all present, so
#   without --reprocess this run would change nothing in the photometry, and
#   collect_alcor_photometry would pool the old schema. --reprocess forces
#   re-measurement, which is the whole point.
#
# Why NOT --median-stack:
#   <night>_median.fits already exists for every night and depends only on the
#   raw frames, which have not changed. Rebuilding it would cost ~19 GB of
#   scratch and a full memmap pass per night for a byte-identical result.
#
# --day-keogram is kept: the night columns come free from the main pass, and the
# sky-brightness keograms are regenerated natively with the ALCOR_FIELD_RADIUS
# cut (9cf7069), superseding the in-place fixup that
# claude_docs/scripts/blank_keogram_field.py applied to the previous run.
#
# Runtime: ~25 min per night, so ~2 h total. Needs the Samsung_4TB drive.
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

{
    echo "###### started $(date) ######"

    if [ ! -d "$EXT_ROOT" ]; then
        echo "FATAL: $EXT_ROOT is not mounted; plug in the Samsung_4TB drive."
        exit 1
    fi

    for d in "${NIGHTS[@]}"; do
        echo "--- $(basename "$d")  $(date +%H:%M:%S) ---"
        alcor_process_night "$d" --day-keogram --reprocess \
            --scratch-dir "$SCRATCH" 2>&1 | grep -v '^\['
        echo "--- done $(basename "$d")  $(date +%H:%M:%S) ---"
    done

    echo "###### finished $(date) ######"
} > "$LOG" 2>&1

echo "done; log at $LOG"
