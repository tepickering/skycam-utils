"""The archive-wide driver: every night of an archive, resumably.

`alcor_process_night` turns one night directory into the standard data products.
This module runs it across a whole archive -- hundreds of nights, days of wall
clock -- and adds only what that scale requires: a ledger so the run resumes, a
pause switch that works on a detached process, throttled progress reporting, a
rule for leaving alone the nights that are still arriving from the camera host,
and an opt-in prune of the redundant vendor JPEGs.  It contains no science.
"""

import datetime
import re
import signal
import time
from pathlib import Path

from .ledger import ArchiveLedger, options_fingerprint
from .night import alcor_process_night

#: Night directories are named YYYY-MM-DD (or YYYY_MM_DD in older archives).
#: Matching on the name is what keeps the sibling ``keograms/`` and ``movies/``
#: trees, which live in the same root, out of a run.
NIGHT_NAME_RE = re.compile(r"^(\d{4})[-_](\d{2})[-_](\d{2})$")

#: Hours a night directory must have been unchanged before it is processed.
#: The archive is synced from the camera host, so a directory can be
#: mid-download while the driver reads it; processing one would write partial
#: products under a ``done`` ledger entry that no later run revisits.
ALCOR_ARCHIVE_MIN_AGE = 24.0


def _night_date(name):
    """The date encoded in a night-directory name, or None if it is not one."""
    match = NIGHT_NAME_RE.match(name)
    if match is None:
        return None
    try:
        return datetime.date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _as_date(value):
    """Coerce a `~datetime.date` or a ``YYYY-MM-DD`` string to a date."""
    if isinstance(value, datetime.date):
        return value
    return datetime.date.fromisoformat(str(value).replace("_", "-"))


def discover_nights(archive_dir, start=None, end=None, nights=None, reverse=False):
    """
    The night directories of an archive, in date order.

    Parameters
    ----------
    archive_dir : str or `~pathlib.Path`
        Archive root, holding one directory per night.
    start, end : str or `~datetime.date` or None (default=None)
        Inclusive date bounds.
    nights : list of str or None (default=None)
        Explicit night directory names, instead of scanning `archive_dir`.
    reverse : bool (default=False)
        Newest first instead of oldest first.

    Returns
    -------
    list of `~pathlib.Path`
        The selected night directories.

    Raises
    ------
    FileNotFoundError
        If a name in `nights` is not a directory of `archive_dir`.
    ValueError
        If a name in `nights` is not a ``YYYY-MM-DD`` night name.
    """
    archive_dir = Path(archive_dir)
    if nights:
        dated = []
        for name in nights:
            path = archive_dir / name
            if not path.is_dir():
                raise FileNotFoundError(f"no such night directory: {path}")
            date = _night_date(path.name)
            if date is None:
                raise ValueError(f"not a YYYY-MM-DD night name: {path.name}")
            dated.append((date, path))
    else:
        dated = [(_night_date(path.name), path)
                 for path in archive_dir.iterdir() if path.is_dir()]
        dated = [item for item in dated if item[0] is not None]
    if start is not None:
        start = _as_date(start)
        dated = [item for item in dated if item[0] >= start]
    if end is not None:
        end = _as_date(end)
        dated = [item for item in dated if item[0] <= end]
    dated.sort(key=lambda item: (item[0], item[1].name), reverse=reverse)
    return [path for _, path in dated]


def night_last_modified(night_dir, pattern="*.fits.bz2"):
    """
    When a night directory last changed, as a Unix timestamp.

    One directory listing plus two `stat` calls, rather than stat-ing all twelve
    thousand files a night holds. The directory's own mtime moves whenever a
    file is created or renamed inside it -- which is what rsync does when it
    finishes a transfer -- and the last frame in sorted order catches one still
    being written in place.

    Parameters
    ----------
    night_dir : str or `~pathlib.Path`
        The night directory.
    pattern : str (default="*.fits.bz2")
        Glob selecting the frames.

    Returns
    -------
    float
        The later of the two mtimes.
    """
    night_dir = Path(night_dir)
    mtimes = [night_dir.stat().st_mtime]
    frames = sorted(night_dir.glob(pattern))
    if frames:
        mtimes.append(frames[-1].stat().st_mtime)
    return max(mtimes)


def is_too_recent(night_dir, min_age_hours, pattern="*.fits.bz2", now=None):
    """
    True when `night_dir` changed recently enough that it may still be arriving.

    The test is on modification time, never on the timestamps in the filenames,
    because the two ways a night can be incomplete look opposite from the names:
    the current night is still being observed, so its newest filename stamp is
    recent, while an older night being back-filled from the camera host has old
    filename stamps and fresh mtimes. Only mtime catches both.

    Parameters
    ----------
    night_dir : str or `~pathlib.Path`
        The night directory.
    min_age_hours : float
        Quiet period in hours. 0 (or None) disables the rule.
    pattern : str (default="*.fits.bz2")
        Glob selecting the frames.
    now : float or None (default=None)
        Unix timestamp to compare against; None uses the current time.

    Returns
    -------
    bool
        True when the night should be left alone for now.
    """
    if not min_age_hours:
        return False
    now = time.time() if now is None else now
    age_hours = (now - night_last_modified(night_dir, pattern)) / 3600.0
    return age_hours < min_age_hours
#: Seconds between the progress lines a night's frame log is collapsed into.
ALCOR_ARCHIVE_LOG_INTERVAL = 60.0

#: `alcor_process_night` reports each frame as ``[done/total] filename``.
_FRAME_LINE_RE = re.compile(r"^\[(\d+)/(\d+)\]\s")


def _duration(seconds):
    """
    Format a duration compactly: ``45s``, ``15m30s``, ``5h32m``, ``13.2d``.

    Returns ``"?"`` for a non-finite value, which is what an unknown rate gives.
    """
    if seconds is None or seconds != seconds or seconds in (float("inf"),
                                                            float("-inf")):
        return "?"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"
    return f"{seconds / 86400:.1f}d"


class ThrottledLog:
    """
    Collapse a night's per-frame log lines into periodic progress updates.

    `alcor_process_night` logs one line per frame. Over a whole archive that is
    of order a million lines, so frame lines are swallowed and re-emitted at
    most once per `interval` seconds as a rate-and-ETA summary. Every other
    message -- the ones that say what was written, or what went wrong -- passes
    through untouched, which is the point: the throttle must not hide anything
    that only happens once.

    Parameters
    ----------
    emit : callable
        Called with each line that survives the throttle.
    prefix : str (default="")
        Prepended to every emitted line, e.g. ``"[ 12/613] 2025-01-12  "``.
    interval : float (default ALCOR_ARCHIVE_LOG_INTERVAL)
        Minimum seconds between progress lines.
    verbose : bool (default=False)
        Pass frame lines through instead of throttling them.
    clock : callable (default `time.monotonic`)
        Returns elapsed seconds; injectable for tests.

    Attributes
    ----------
    done, total : int
        The most recent frame counts seen.
    """

    def __init__(self, emit, prefix="", interval=ALCOR_ARCHIVE_LOG_INTERVAL,
                 verbose=False, clock=time.monotonic):
        self.emit = emit
        self.prefix = prefix
        self.interval = interval
        self.verbose = verbose
        self.clock = clock
        self.started = clock()
        self.last = self.started
        self.done = 0
        self.total = 0

    def __call__(self, message):
        match = _FRAME_LINE_RE.match(str(message))
        if match is None:
            self.emit(f"{self.prefix}{message}")
            return
        self.done, self.total = int(match.group(1)), int(match.group(2))
        if self.verbose:
            self.emit(f"{self.prefix}{message}")
            return
        now = self.clock()
        if now - self.last < self.interval:
            return
        self.last = now
        self.emit(f"{self.prefix}{self.progress()}")

    def progress(self):
        """A ``frames 61/2000 · 1.9 f/s · 13m left`` summary of this night."""
        elapsed = self.clock() - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        left = (self.total - self.done) / rate if rate > 0 else float("nan")
        return (f"frames {self.done}/{self.total} · {rate:.2f} f/s · "
                f"{_duration(left)} left")
#: Fraction of a night's frames that may fail and still allow its JPEGs to go.
ALCOR_ARCHIVE_PRUNE_MAX_ERROR_FRACTION = 0.10


def prune_night_jpegs(night_dir, products, n_frames, n_errors, dry_run=False,
                      max_error_fraction=ALCOR_ARCHIVE_PRUNE_MAX_ERROR_FRACTION):
    """
    Delete a completed night's vendor JPEGs, or report what would go.

    The camera's vendor software writes two JPEGs per frame, ``<stamp>.jpg`` and
    ``Unwrap_<stamp>.jpg`` -- about 7.3 GB a night. Better renderings are
    derivable from the raw FITS on demand, so they are redundant, and deleting
    them is what makes an archive-wide run fit on the volume. It is also the one
    irreversible thing the driver does, so it is gated: the night's products
    must exist and be non-empty, and its frames must have mostly succeeded.

    Parameters
    ----------
    night_dir : str or `~pathlib.Path`
        The archive night directory to prune.
    products : dict
        The night's product paths, as returned by :func:`alcor_process_night`.
        ``summary_file`` and ``photometry_file`` are the two that are checked.
    n_frames, n_errors : int
        Frames processed, and frames that failed.
    dry_run : bool (default=False)
        Report what would be deleted without deleting it.
    max_error_fraction : float (default ALCOR_ARCHIVE_PRUNE_MAX_ERROR_FRACTION)
        Refuse to prune when more than this fraction of frames failed.

    Returns
    -------
    tuple
        ``(n_files, n_bytes, reason)``. `reason` is None when the prune ran, or
        would have under `dry_run`, and otherwise says why the night was left
        alone.
    """
    checks = (("sky_brightness.csv", products.get("summary_file")),
              ("the photometry rollup", products.get("photometry_file")))
    for label, value in checks:
        path = Path(value) if value else None
        if path is None or not path.exists() or path.stat().st_size == 0:
            return 0, 0, f"{label} is missing or empty"
    if n_frames and n_errors / n_frames > max_error_fraction:
        return 0, 0, (f"{n_errors} of {n_frames} frames failed, over the "
                      f"{max_error_fraction:.0%} limit")
    # One glob: "*.jpg" matches the Unwrap_ series too.
    jpegs = sorted(Path(night_dir).glob("*.jpg"))
    n_bytes = sum(path.stat().st_size for path in jpegs)
    if not dry_run:
        for path in jpegs:
            path.unlink()
    return len(jpegs), n_bytes, None
#: Consecutive night failures that abort a run. An unmounted archive fails every
#: remaining night instantly, and stopping beats writing hundreds of bogus
#: ledger entries.
ALCOR_ARCHIVE_MAX_CONSECUTIVE_FAILURES = 10

#: Seconds between checks of the PAUSE sentinel while a run is paused.
ALCOR_ARCHIVE_PAUSE_POLL = 30.0


def _wait_while_paused(pause_file, poll, log, stop, sleep=time.sleep):
    """
    Block while `pause_file` exists.

    Pausing is a file rather than a signal precisely so it works on a detached
    run whose PID nobody has: the operator creates the file, and the driver
    stops at the next night boundary until it goes away.

    Parameters
    ----------
    pause_file : `~pathlib.Path`
        The sentinel. Its contents are ignored; only existence matters.
    poll : float
        Seconds between checks.
    log : callable
        Progress reporter.
    stop : dict
        Shared flag; ``stop["requested"]`` breaks the wait.
    sleep : callable (default `time.sleep`)
        Injectable for tests.

    Returns
    -------
    bool
        True if the wait ended because a stop was requested.
    """
    if not pause_file.exists():
        return False
    log(f"paused — waiting on {pause_file}")
    while pause_file.exists() and not stop["requested"]:
        sleep(poll)
    if stop["requested"]:
        return True
    log("resumed")
    return False


def alcor_process_archive(archive_dir, out_dir, start=None, end=None,
                          nights=None, reverse=False, pattern="*.fits.bz2",
                          min_age=ALCOR_ARCHIVE_MIN_AGE, prune_jpegs=False,
                          prune_dry_run=False, retry_failed=False,
                          force_options=False,
                          max_consecutive_failures=ALCOR_ARCHIVE_MAX_CONSECUTIVE_FAILURES,
                          pause_poll=ALCOR_ARCHIVE_PAUSE_POLL,
                          log_interval=ALCOR_ARCHIVE_LOG_INTERVAL, verbose=False,
                          log=None, pause_sleep=time.sleep, install_signals=False,
                          **night_kwargs):
    """
    Run :func:`alcor_process_night` over every night of an archive, resumably.

    An archive is hundreds of night directories and days of wall clock, so this
    adds what that scale needs and nothing else: a ledger so a killed run picks
    up where it stopped, a ``PAUSE`` file so a detached run can be held without
    its PID, throttled progress with an ETA, a quiet period that leaves alone the
    nights still arriving from the camera host, and an opt-in prune of the
    redundant vendor JPEGs.

    All state lives in ``<out_dir>/.archive_state/``: ``ledger.json``, the
    ``PAUSE`` sentinel the operator creates, and ``archive.log``. Each night's
    products go to ``<out_dir>/<night>/``; the archive itself is only read from,
    except by `prune_jpegs`.

    Parameters
    ----------
    archive_dir : str or `~pathlib.Path`
        Archive root holding one directory per night.
    out_dir : str or `~pathlib.Path`
        Products tree. Created if absent.
    start, end : str or `~datetime.date` or None (default=None)
        Inclusive date bounds on the nights to run.
    nights : list of str or None (default=None)
        Explicit night names instead of scanning `archive_dir`.
    reverse : bool (default=False)
        Newest first.
    pattern : str (default="*.fits.bz2")
        Glob selecting frames, passed to `alcor_process_night`.
    min_age : float (default ALCOR_ARCHIVE_MIN_AGE)
        Hours a night must have been unchanged before it is processed. 0
        disables the rule.
    prune_jpegs : bool (default=False)
        Delete each night's vendor JPEGs once its products are written.
    prune_dry_run : bool (default=False)
        Report what pruning would free without deleting. Implies the pruning
        pass and overrides `prune_jpegs`.
    retry_failed : bool (default=False)
        Re-attempt nights previously recorded ``failed``.
    force_options : bool (default=False)
        Continue despite a changed photometry-options fingerprint.
    max_consecutive_failures : int (default ALCOR_ARCHIVE_MAX_CONSECUTIVE_FAILURES)
        Abort after this many nights fail in a row.
    pause_poll : float (default ALCOR_ARCHIVE_PAUSE_POLL)
        Seconds between checks of the ``PAUSE`` file while paused.
    log_interval : float (default ALCOR_ARCHIVE_LOG_INTERVAL)
        Seconds between per-frame progress lines.
    verbose : bool (default=False)
        Pass every per-frame line through instead of throttling.
    log : callable or None (default=None)
        Called with each progress message, in addition to ``archive.log``.
    pause_sleep : callable (default `time.sleep`)
        Injectable sleep for the pause loop.
    install_signals : bool (default=False)
        Install SIGINT/SIGTERM handlers so the first interrupt finishes the
        current night and the second aborts. Only valid in the main thread; the
        CLI sets it, tests do not.
    **night_kwargs
        Forwarded to :func:`alcor_process_night` (``day_keogram``,
        ``median_stack``, ``both``, ``workers``, ``scratch_dir``, ...).

    Returns
    -------
    dict
        ``ledger`` (the `ArchiveLedger`), ``processed``, ``failed``,
        ``skipped_recent``, ``skipped_failed`` (night-name lists),
        ``pruned_files``, ``pruned_bytes``, and ``stopped`` (True if the run
        ended early via a signal or the failure limit).

    Raises
    ------
    ValueError
        If the photometry options differ from those recorded in the ledger and
        `force_options` is not set.
    """
    archive_dir = Path(archive_dir)
    out_dir = Path(out_dir)
    state_dir = out_dir / ".archive_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    pause_file = state_dir / "PAUSE"
    log_file = state_dir / "archive.log"

    def _log(message):
        line = str(message)
        with log_file.open("a") as handle:
            handle.write(f"{line}\n")
        if log is not None:
            log(line)

    _log(f"state directory: {state_dir}")

    ledger = ArchiveLedger(state_dir)
    ledger.check_fingerprint(options_fingerprint(night_kwargs),
                             force=force_options)
    ledger.save()
    for night in ledger.reset_stale_running():
        _log(f"{night}: was left running by an earlier run; will reprocess")

    candidates = discover_nights(archive_dir, start=start, end=end,
                                 nights=nights, reverse=reverse)
    total = len(candidates)
    _log(f"{total} night directories selected under {archive_dir}")

    stop = {"requested": False}
    previous_handlers = {}

    def _handle_signal(signum, frame):
        if stop["requested"]:
            raise KeyboardInterrupt("second interrupt: aborting now")
        stop["requested"] = True
        _log("interrupt received — finishing the current night, then stopping "
             "(interrupt again to abort immediately)")

    if install_signals:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, _handle_signal)

    processed, failed, skipped_recent, skipped_failed = [], [], [], []
    pruned_files = pruned_bytes = 0
    consecutive = 0
    run_started = time.monotonic()

    def _maybe_prune(night, night_dir, products, n_frames, n_errors, prefix):
        nonlocal pruned_files, pruned_bytes
        if not (prune_jpegs or prune_dry_run) or ledger.was_pruned(night):
            return
        n_files, n_bytes, reason = prune_night_jpegs(
            night_dir, products, n_frames, n_errors, dry_run=prune_dry_run)
        if reason is not None:
            _log(f"{prefix}not pruned: {reason}")
            return
        pruned_files += n_files
        pruned_bytes += n_bytes
        verb = "would free" if prune_dry_run else "pruned"
        _log(f"{prefix}{verb} {n_files} jpegs ({n_bytes / 1e9:.1f} GB)")
        if not prune_dry_run:
            ledger.mark_pruned(night, n_files, n_bytes)

    try:
        for index, night_dir in enumerate(candidates, start=1):
            night = night_dir.name
            prefix = f"[{index:>4}/{total}] {night}  "
            if stop["requested"]:
                break
            if _wait_while_paused(pause_file, pause_poll, _log, stop,
                                  sleep=pause_sleep):
                break

            state = ledger.state(night)
            if state == "done":
                entry = ledger.entry(night) or {}
                _maybe_prune(night, night_dir,
                             {key: value for key, value
                              in (entry.get("products") or {}).items()},
                             entry.get("n_frames", 0), entry.get("n_errors", 0),
                             prefix)
                continue
            if state == "failed" and not retry_failed:
                skipped_failed.append(night)
                continue
            if is_too_recent(night_dir, min_age, pattern):
                skipped_recent.append(night)
                _log(f"{prefix}skipped: changed within {min_age:g} h, may still "
                     "be arriving from the camera host")
                continue
            if consecutive >= max_consecutive_failures:
                _log(f"aborting: {consecutive} nights failed in a row")
                stop["requested"] = True
                break

            ledger.mark_running(night)
            night_out = out_dir / night
            night_log = ThrottledLog(_log, prefix=prefix, interval=log_interval,
                                     verbose=verbose)
            started = time.monotonic()
            _log(f"{prefix}start")
            try:
                result = alcor_process_night(night_dir, out_dir=night_out,
                                             pattern=pattern, log=night_log,
                                             **night_kwargs)
            except Exception as exc:  # one bad night must not cost the rest
                elapsed = time.monotonic() - started
                ledger.mark_failed(night, exc, elapsed=elapsed)
                failed.append(night)
                consecutive += 1
                _log(f"{prefix}FAILED after {_duration(elapsed)}: {exc}")
                continue

            elapsed = time.monotonic() - started
            n_frames = len(result.get("files") or [])
            n_errors = len(result.get("errors") or [])
            products = {key: result.get(key) for key
                        in ("summary_file", "photometry_file", "keogram_file",
                            "keogram_plot", "day_keogram_file",
                            "day_keogram_plot", "median_file")}
            ledger.mark_done(night, elapsed, n_frames, n_errors, products)
            processed.append(night)
            consecutive = 0
            _log(f"{prefix}done · {_duration(elapsed)} · {n_frames} frames · "
                 f"{n_errors} frame errors")
            _maybe_prune(night, night_dir, products, n_frames, n_errors, prefix)

            done_count, mean = ledger.timing()
            left = total - index
            eta = _duration(left * mean) if mean else "?"
            _log(f"           overall  {done_count} done / {left} left · "
                 f"elapsed {_duration(time.monotonic() - run_started)} · "
                 f"ETA {eta}")
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    _log(f"run finished: {len(processed)} processed, {len(failed)} failed, "
         f"{len(skipped_recent)} skipped as too recent, "
         f"{len(skipped_failed)} skipped as previously failed")
    if skipped_recent:
        _log(f"still arriving, left pending: {', '.join(skipped_recent)}")
    if pruned_files:
        verb = "would free" if prune_dry_run else "freed"
        _log(f"jpeg prune {verb} {pruned_files} files "
             f"({pruned_bytes / 1e9:.1f} GB)")

    return {"ledger": ledger, "processed": processed, "failed": failed,
            "skipped_recent": skipped_recent, "skipped_failed": skipped_failed,
            "pruned_files": pruned_files, "pruned_bytes": pruned_bytes,
            "stopped": stop["requested"]}
def alcor_archive_status(out_dir, log=print):
    """
    Summarise an archive run's ledger without disturbing it.

    Reads only ``<out_dir>/.archive_state/ledger.json``, so it works while a run
    is in flight and on a machine where the archive itself is not mounted.

    Parameters
    ----------
    out_dir : str or `~pathlib.Path`
        The products tree given to :func:`alcor_process_archive`.
    log : callable (default `print`)
        Called with each line of the report.

    Returns
    -------
    dict
        ``counts`` (nights by state), ``elapsed_mean`` (seconds per completed
        night, or None), ``eta`` (seconds to finish the pending nights, or
        None), ``failed`` (night names), and ``pruned_bytes``.

    Raises
    ------
    FileNotFoundError
        If no ledger exists under `out_dir`.
    """
    state_dir = Path(out_dir) / ".archive_state"
    ledger = ArchiveLedger(state_dir)
    if not ledger.path.exists():
        raise FileNotFoundError(f"no ledger.json under {state_dir}")

    counts = ledger.counts()
    done, mean = ledger.timing()
    remaining = counts["pending"] + counts["running"]
    eta = remaining * mean if mean else None
    failed = sorted(night for night, entry in ledger.data["nights"].items()
                    if entry.get("state") == "failed")
    pruned_bytes = sum(entry.get("bytes_freed") or 0
                       for entry in ledger.data["nights"].values())

    log(f"ledger: {ledger.path}")
    log(f"  done {counts['done']} · failed {counts['failed']} · "
        f"pending {counts['pending']} · running {counts['running']}")
    if mean:
        log(f"  {_duration(mean)} per night · {remaining} left · "
            f"ETA {_duration(eta)}")
    if pruned_bytes:
        log(f"  jpegs pruned: {pruned_bytes / 1e9:.1f} GB freed")
    for night in failed:
        log(f"  FAILED {night}: {(ledger.entry(night) or {}).get('error')}")

    return {"counts": counts, "elapsed_mean": mean, "eta": eta,
            "failed": failed, "pruned_bytes": pruned_bytes}
