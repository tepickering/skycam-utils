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
import time
from pathlib import Path

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
