# Archive-wide alcor photometry driver (`alcor_process_archive`)

*Design, 2026-09-07*

## Problem

`alcor_process_night` turns one night directory into the standard data products, and
`scripts/make_movies.sh` calls it once per night as each night completes on the ops
host. What does not exist is a way to run it across a whole archive.

`/Volumes/Seagate_24TB/skycam` holds **613 night directories** spanning 2025-01-01 to
2026-09-05, ~3900 `.fits.bz2` per day (~2000 of them with the Sun below -12 deg), 31 GB
per night, and no derived products at all. Processing it is a multi-day job, so the
driver is mostly an exercise in state, safety, and progress reporting; the science is
already implemented one level down.

Measured on the target machine (10 cores, archive on USB): ~2.3 frames/s steady state
with the full product set, i.e. **~30 min per night, ~13 days** for the archive.
It is CPU-bound once the worker pool is full, not I/O-bound.

## Goals

* Process every night in an archive into the full product set, unattended, over days.
* Leave alone any night that is still arriving from the camera host.
* Resume exactly where it left off after a crash, a reboot, or a deliberate stop.
* Report progress and a credible ETA without drowning the log in per-frame lines.
* Pause and resume a backgrounded run without finding its PID.
* Never lose the archive: derived products go to a separate tree, and the one
  destructive operation is opt-in, gated, and dry-runnable.

## Non-goals

* Cross-night parallelism. `alcor_process_night` already saturates the cores.
* Re-implementing any per-frame science. This driver only iterates and records.
* Replacing `scripts/make_movies.sh` in the nightly ops path.

## Decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Placement | Packaged: `skycam_utils/alcor/archive.py` + `alcor_process_archive` entry point | Sibling to `alcor_process_night`; importable and testable, unlike `scripts/` |
| Output | Separate products tree, `<out-dir>/<night>/` | Archive stays read-only except for pruning |
| Per-frame CSVs | Kept | They are the frame-level resume mechanism and stay inspectable |
| Products | Baseline + `--day-keogram` + `--median-stack` + `--both` | Complete archive; ~30 min and ~1.4 GB per night |
| Ledger | JSON, atomically replaced | 613 rows; inspectable and hand-editable beats SQLite here |
| Control | Ledger + a `PAUSE` sentinel at `<out-dir>/.archive_state/PAUSE` | Pausing a detached 13-day run must not require a PID |
| JPEG pruning | Opt-in `--prune-jpegs` flag on the driver | Frees space as the run proceeds so the drive never fills |
| In-flight nights | Skip any night modified within `--min-age` hours (default 24) | The archive syncs from the camera host; a half-arrived night would be recorded `done` and never revisited |

## Architecture

New module `skycam_utils/alcor/archive.py`, sitting between `night` and `cli` in the
package's one-way dependency chain (`... -> keogram -> night -> archive -> cli`). It
imports `alcor_process_night` and adds iteration, state, and safety; it introduces no
new science and no new dependency edges.

```
alcor_process_archive /Volumes/Seagate_24TB/skycam \
    -o /Volumes/Seagate_24TB/skycam_products \
    --day-keogram --median-stack --both --prune-jpegs \
    --scratch-dir ~/scratch
```

### Night discovery

Directories in the archive root whose names match `YYYY-MM-DD` or `YYYY_MM_DD`, which
naturally excludes the sibling `keograms/` and `movies/` trees. `--start` / `--end`
bound the range, `--nights` takes an explicit list, `--reverse` runs newest-first.
Default order is oldest-first.

### Skipping nights that are still arriving

**The archive is not static.** It is synced from the camera host's computer, so at any
moment one or more night directories may be mid-download, and the most recent night is
additionally still being observed. This was seen directly while preparing this design:
`2026-09-06` listed as an empty directory, and 45 minutes later held 12,449 files.

Processing such a night is worse than useless. The products would cover only the frames
that had landed at the time, the ledger would record the night `done`, and it would be
skipped by every subsequent run — silently and permanently incomplete, with nothing in
the output to indicate it. With `--prune-jpegs` it is actively destructive: JPEGs would
be deleted for a night whose FITS frames have not all arrived.

So the driver applies a **quiet-period rule**: a night whose contents changed within
`--min-age` hours (default 24) is skipped, logged, and left `pending` for a later run.

The test is on **modification time, not on the timestamps in the filenames**, because the
two failure modes differ:

* the current night is still being *observed*, so its newest filename timestamp is recent;
* an older night being *back-filled* from the camera host has old filename timestamps but
  new mtimes.

Only mtime catches both. The check is deliberately cheap — two `stat` calls, not a walk
of 12,000 files: the night directory's own mtime (which updates on every file creation
or rename, including rsync's temp-file rename) and the mtime of the last frame in sorted
order, whichever is later.

`--min-age 0` disables the rule for a night known to be complete. Nights skipped this way
are counted and named in the run summary and in `--status`, so an incomplete archive is
visible rather than silent; they are never recorded `failed`, since nothing failed.

### Output tree

Each night writes to `<out-dir>/<night>/`, holding exactly what `alcor_process_night`
produces: per-frame `<frame>_phot.csv`, `sky_brightness.csv`, `<night>_phot.csv`,
`<night>_sb_keogram.{fits,png}`, `<night>_keogram.{fits,png}`, `<night>_median.fits`.
Because `alcor_process_night` already writes per-frame CSVs into its `out_dir`, this
needs no change to the night driver.

The median stack needs ~24 GB of scratch per night. `--scratch-dir` should point at
local storage rather than the USB archive.

### Ledger and resume

All run state lives in one directory, `<out-dir>/.archive_state/` — that is, inside the
products tree named by `-o/--out-dir`, never in the archive and never in the current
working directory. It is created on the first run and holds exactly three things:

| Path | Written by | Purpose |
| --- | --- | --- |
| `<out-dir>/.archive_state/ledger.json` | the driver | Per-night state; rewritten atomically (temp file + `os.replace`) after each night |
| `<out-dir>/.archive_state/PAUSE` | **the operator** | Presence of this file pauses the run. Contents are ignored |
| `<out-dir>/.archive_state/archive.log` | the driver | The appended run log |

The driver prints the absolute path of its state directory as the first line of every
run, so the pause switch never has to be guessed at from a detached session.

Each ledger entry records the night name, state (`pending` / `running` / `done` /
`failed`), start and finish timestamps, elapsed seconds, frame count, per-frame error
count, product paths, JPEGs pruned and bytes freed, last error message, and attempt
count. Top-level metadata records the archive directory, the output directory, the
schema version, and the options fingerprint.

Resume works at two levels:

* **Night level**, from the ledger. `done` nights are skipped. A night left `running`
  by a crash is reset to `pending` at startup.
* **Frame level**, free of charge. `alcor_process_night` already reuses a non-empty
  `<frame>_phot.csv` instead of re-measuring, and those CSVs now live in the products
  tree. The frame is still *read* — the surface-brightness map and both keogram columns
  need its pixels — so an interrupted night recovers roughly the 17% of per-frame cost
  that is measurement, not the whole thing. `--median-stack` and `--day-keogram` have no
  frame-level resume and redo in full.

**Options fingerprint.** The ledger stores the photometry-affecting options (`both`,
`gaussian`, `aperture_radius`, `annulus_width`, `min_altitude`, `vmag_limit`,
`variables`). Resuming with a different set is refused, with a printed diff, unless
`--force-options` is given. This is the archive-level form of the hazard `--reprocess`
exists for one level down: a rollup assembled from per-frame CSVs written in two
different modes is a silently mixed schema, and nothing downstream would notice.

### Pause and stop

**Pausing is done by creating `<out-dir>/.archive_state/PAUSE`.** Any means of creating
the file works — `touch`, an editor, `scp` — and its contents are ignored; only its
existence matters. For a run started with `-o /Volumes/Seagate_24TB/skycam_products`:

```bash
touch /Volumes/Seagate_24TB/skycam_products/.archive_state/PAUSE   # pause
rm    /Volumes/Seagate_24TB/skycam_products/.archive_state/PAUSE   # resume
```

The driver checks for the file **between nights, not during one**, so a pause takes
effect at the next night boundary — up to ~30 min after the file appears, and the night
in flight always runs to completion and is recorded `done`. Once paused it logs
`paused — waiting on <path>` and polls every 30 s until the file is removed, at which
point the same process picks up at the next night. Nothing needs relaunching, and no PID
needs finding: this is the entire reason the mechanism is a file rather than a signal.

Pausing is *not* required before killing the run. `SIGINT`/`SIGTERM` (Ctrl-C) makes the
driver finish the current night and exit cleanly; a second one aborts immediately,
leaving that night `pending` to be resumed, with its already-written per-frame CSVs
reused. A `PAUSE` file left in place will pause the *next* run at its very first night,
which is a deliberate property — it lets a run be staged and held — but it means the file
must be removed before restarting.

### Progress reporting

Progress goes to stderr and to `archive.log`. The night driver emits one line per frame
— about 1.2M lines over the archive — so the archive driver wraps the `log` callable it
passes down, collapsing `[i/n] frame` lines into a throttled progress line (every 60 s)
and passing everything else through unchanged. `--verbose` disables the throttle.

```
[ 12/613] 2025-01-12  start · 2013 night frames
[ 12/613] 2025-01-12  frames 500/2013 · 1.9 f/s · 13m left
[ 12/613] 2025-01-12  done · 28m14s · 0 frame errors · 1.4 GB · pruned 7834 jpegs (7.3 GB)
           overall  12 done / 601 left · elapsed 5h32m · ETA 2026-09-20 04:11 (13.2 d)
```

`alcor_process_archive --status -o <out-dir>` reads the ledger, prints that summary plus
the failure list, and exits. `-o/--out-dir` is required in every mode, but `--status`
makes the `archive_dir` positional optional, so status works with the archive
unmounted.

### Error handling

A night that raises is recorded `failed` with its message and the run continues — one
bad night must not cost the other 612. `--retry-failed` re-attempts failed nights on a
later pass. `alcor_process_night` already collects per-frame failures into `errors[]`
without aborting; that count is recorded per night.

The one systemic guard is `--max-consecutive-failures` (default 10). If the USB archive
unmounts mid-run, every remaining night fails instantly; stopping after 10 is better
than writing 600 meaningless ledger entries.

### JPEG pruning

The vendor software writes two JPEGs per frame (`<ts>.jpg` and `Unwrap_<ts>.jpg`),
~7.3 GB per night, ~4.5 TB across the archive. Better renderings are now derivable
on demand from the raw FITS, so these are redundant — and the archive volume is 92%
full with 1.9 TB free, which the products themselves would eat a large fraction of.

`--prune-jpegs` deletes a night's JPEGs *after* that night reaches `done`, and only if:

* `sky_brightness.csv` and `<night>_phot.csv` exist and are non-empty, and
* the per-frame error fraction is below 10%.

`--prune-dry-run` evaluates the same gates and reports what would be freed, deleting
nothing; it does not require `--prune-jpegs` and overrides it if both are given. Pruning
also applies to nights the ledger already marks `done` and skipped this run, provided
their ledger entry shows they were never pruned — so pruning can be turned on partway
through a run, or in a later pass over an already-processed archive. Every prune is logged
with file count and bytes and recorded in the ledger. Nights with no rendered
`movies/<night>/` are logged as warnings when pruned, since for those the JPEGs are the
only existing rendering. (As of this design, movies for the 25 such nights that had
JPEGs have been backfilled; `2026-09-06` was an empty directory.)

## Testing

`skycam_utils/tests/test_alcor_archive.py`, using the `patch_alcor` fixture to stub
`alcor_process_night` over a synthetic tree of empty night directories — the driver's
logic is state management, so no real frames are needed.

1. Discovery selects only date-shaped directories, skipping `keograms` and `movies`.
2. `--start` / `--end` / `--nights` / `--reverse` select and order correctly.
3. A completed night is recorded `done`; a second run skips it.
4. A night left `running` by a simulated crash is reset to `pending` and reprocessed.
5. A raising night is recorded `failed` and the run continues to the next night.
6. `--retry-failed` re-attempts a failed night; without it, the night stays skipped.
7. An options-fingerprint mismatch refuses to run; `--force-options` overrides.
8. Creating `<out-dir>/.archive_state/PAUSE` halts the run before the next night;
    removing it resumes in the same process. A `PAUSE` present at startup pauses before
    the first night. The night already in flight when the file appears still completes
    and is recorded `done`.
9. `--max-consecutive-failures` aborts the run after the configured streak.
10. `--prune-jpegs` deletes JPEGs for a `done` night, and does not for a `failed` one
    or one whose error fraction is too high; `--prune-dry-run` deletes nothing.
11. A night whose directory mtime is inside `--min-age` is skipped, left `pending`, and
    named in the summary; `--min-age 0` processes it. A night with old filename
    timestamps but a fresh mtime is skipped too — the rule is mtime, not filename.
12. `--prune-jpegs` never prunes a night skipped for being too recent.
13. `--status` reads a ledger and reports without an archive directory present.
14. The log throttle collapses per-frame lines but passes other messages through.

## Expected run

| | |
| --- | --- |
| Nights | 613 |
| Wall clock | ~30 min/night, ~13 days |
| Products written | ~1.4 GB/night, ~860 GB |
| JPEGs freed | ~7.3 GB/night, ~4.5 TB |
| Net disk | ~3.6 TB recovered; the volume goes from 92% to ~72% full |
