# Archive-wide alcor photometry driver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `alcor_process_archive`, a resumable driver that runs `alcor_process_night` over every night directory of a skycam archive, unattended, over multiple days.

**Architecture:** Two new modules in the existing `skycam_utils/alcor/` package. `ledger.py` is a standalone JSON state store with no package dependencies. `archive.py` holds night discovery, the quiet-period rule, log throttling, JPEG pruning, and the driver loop; it imports `alcor_process_night` and adds no new science. The dependency chain gains `... -> night -> archive -> cli`, with `ledger` as a leaf that only `archive` imports.

**Tech Stack:** Python 3, stdlib only for the new code (`json`, `os`, `re`, `signal`, `time`, `datetime`, `pathlib`). pytest for tests. No new dependencies.

**Design spec:** `claude_docs/specs/2026-09-07-alcor-archive-driver-design.md`. Read it before starting.

## Global Constraints

- **Packaging is `pyproject.toml`-only.** There is no `setup.py`, `setup.cfg`, `MANIFEST.in`, or `tox.ini`. Add the console script to `[project.scripts]`.
- **`AGENTS.md` is a symlink to `CLAUDE.md`.** Edit `CLAUDE.md` only.
- **Tests must patch with the `patch_alcor` fixture** (`skycam_utils/tests/conftest.py`), never `monkeypatch.setattr(alcor, ...)`. Each submodule binds its own name, so patching the package re-export leaves the real consumers running the real function and the test passes for the wrong reason.
- **`alcor/__init__.py` re-exports the entire namespace, private names included.** Every new public *and* private name goes into the `from .ledger import (...)` / `from .archive import (...)` blocks and into `__all__`.
- **No new third-party dependencies.** The new modules use the standard library only.
- **The dependency chain runs one way.** `archive` may import from `ledger` and `night`; neither may import from `archive`. `cli` imports from `archive`.
- **Docstring style:** numpydoc, matching the surrounding package (`Parameters` / `Returns` / `Raises` sections, types as ``str or `~pathlib.Path` ``).
- **Commit after every task.** Do not batch commits.

---

### Task 1: The run ledger

The ledger is the resume mechanism: it records what each night's state is, survives a crash, and refuses a resume that would mix photometry schemas. It has no package dependencies, so it is built and tested first.

**Files:**
- Create: `skycam_utils/alcor/ledger.py`
- Create: `skycam_utils/tests/test_alcor_archive.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `LEDGER_VERSION: int`, `FINGERPRINT_KEYS: tuple[str, ...]`, `options_fingerprint(options: dict) -> dict`, and class `ArchiveLedger(state_dir)` with methods `save()`, `check_fingerprint(fingerprint: dict, force: bool = False) -> None`, `entry(night: str) -> dict | None`, `state(night: str) -> str`, `mark_running(night)`, `mark_done(night, elapsed, n_frames, n_errors, products)`, `mark_failed(night, error, elapsed=None)`, `mark_pruned(night, n_files, n_bytes)`, `reset_stale_running() -> list[str]`, `was_pruned(night: str) -> bool`, `counts() -> dict[str, int]`, `timing() -> tuple[int, float | None]`, and attributes `path: Path`, `data: dict`.

- [ ] **Step 1: Write the failing tests**

Create `skycam_utils/tests/test_alcor_archive.py`:

```python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import json
import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import ArchiveLedger, options_fingerprint


def test_ledger_round_trips_night_state(tmp_path):
    """A night marked done is still done when the ledger is re-read."""
    ledger = ArchiveLedger(tmp_path)
    assert ledger.state("2025-01-01") == "pending"

    ledger.mark_running("2025-01-01")
    assert ledger.state("2025-01-01") == "running"

    ledger.mark_done("2025-01-01", elapsed=1800.0, n_frames=2000, n_errors=0,
                     products={"summary_file": "/x/sky_brightness.csv"})

    reloaded = ArchiveLedger(tmp_path)
    assert reloaded.state("2025-01-01") == "done"
    assert reloaded.entry("2025-01-01")["n_frames"] == 2000
    assert reloaded.entry("2025-01-01")["attempts"] == 1
    assert reloaded.counts()["done"] == 1


def test_ledger_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    """The ledger is replaced, not truncated in place, so a crash cannot shred it."""
    ledger = ArchiveLedger(tmp_path)
    ledger.mark_done("2025-01-01", elapsed=1.0, n_frames=1, n_errors=0, products={})

    assert json.loads(ledger.path.read_text())["nights"]["2025-01-01"]["state"] == "done"
    assert list(tmp_path.glob("*.tmp")) == []


def test_reset_stale_running_returns_a_crashed_night_to_pending(tmp_path):
    """A run killed mid-night leaves `running`; the next run must retry it."""
    ledger = ArchiveLedger(tmp_path)
    ledger.mark_running("2025-01-01")

    reloaded = ArchiveLedger(tmp_path)
    assert reloaded.reset_stale_running() == ["2025-01-01"]
    assert reloaded.state("2025-01-01") == "pending"
    assert ArchiveLedger(tmp_path).state("2025-01-01") == "pending"


def test_failed_night_records_its_error(tmp_path):
    ledger = ArchiveLedger(tmp_path)
    ledger.mark_failed("2025-01-02", ValueError("no dark frames"))

    assert ledger.state("2025-01-02") == "failed"
    assert "no dark frames" in ledger.entry("2025-01-02")["error"]


def test_mark_pruned_is_visible_to_was_pruned(tmp_path):
    ledger = ArchiveLedger(tmp_path)
    ledger.mark_done("2025-01-01", elapsed=1.0, n_frames=1, n_errors=0, products={})
    assert not ledger.was_pruned("2025-01-01")

    ledger.mark_pruned("2025-01-01", 7834, 7_800_000_000)
    assert ledger.was_pruned("2025-01-01")
    assert ArchiveLedger(tmp_path).entry("2025-01-01")["bytes_freed"] == 7_800_000_000


def test_options_fingerprint_keeps_only_photometry_affecting_options():
    fingerprint = options_fingerprint(
        {"both": True, "vmag_limit": 5.5, "workers": 8, "overwrite": False}
    )
    assert fingerprint == {"both": True, "vmag_limit": 5.5}


def test_changed_photometry_options_are_refused_on_resume(tmp_path):
    """A rollup built from CSVs written in two modes is a silently mixed schema."""
    ledger = ArchiveLedger(tmp_path)
    ledger.check_fingerprint({"both": True, "vmag_limit": 5.5})
    ledger.save()

    reloaded = ArchiveLedger(tmp_path)
    with pytest.raises(ValueError, match="both"):
        reloaded.check_fingerprint({"both": False, "vmag_limit": 5.5})


def test_force_options_accepts_a_changed_fingerprint(tmp_path):
    ledger = ArchiveLedger(tmp_path)
    ledger.check_fingerprint({"both": True})
    ledger.save()

    reloaded = ArchiveLedger(tmp_path)
    reloaded.check_fingerprint({"both": False}, force=True)
    assert reloaded.data["fingerprint"] == {"both": False}


def test_timing_averages_only_completed_nights(tmp_path):
    ledger = ArchiveLedger(tmp_path)
    ledger.mark_done("2025-01-01", elapsed=1000.0, n_frames=1, n_errors=0, products={})
    ledger.mark_done("2025-01-02", elapsed=2000.0, n_frames=1, n_errors=0, products={})
    ledger.mark_failed("2025-01-03", "boom")

    assert ledger.timing() == (2, 1500.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: FAIL — `ImportError: cannot import name 'ArchiveLedger' from 'skycam_utils.alcor'`

- [ ] **Step 3: Write the ledger**

Create `skycam_utils/alcor/ledger.py`:

```python
"""Persistent per-night run state for the archive-wide driver.

The ledger is what makes an archive run resumable: nights recorded ``done`` are
skipped by a later run, and a run killed mid-night comes back to a ``pending``
night whose already-written per-frame photometry is reused.  It is deliberately
a single JSON file rewritten in full after every night -- an archive has a few
hundred nights, which is small enough that atomic replacement beats any
incremental format, and it keeps the state readable and hand-editable when a
run needs rescuing.
"""

import datetime
import json
import os
from pathlib import Path

#: Schema version of the ledger file, so a future format change is detectable.
LEDGER_VERSION = 1

#: Options that change the schema or the content of the per-frame photometry
#: CSVs.  Resuming a run with any of these altered would build a night rollup
#: out of CSVs written in two different modes, so the ledger stores them and
#: refuses a mismatched resume.
FINGERPRINT_KEYS = ("both", "gaussian", "aperture_radius", "annulus_width",
                    "min_altitude", "vmag_limit", "variables")
# Deliberately excluded: workers, scratch_dir, max_frames, day_keogram,
# median_stack and the like change how or how fast a night is processed, not
# what a per-frame CSV contains, so they must not block a resume.


def _now():
    """The current UTC time as an ISO-8601 string, to the second."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def options_fingerprint(options):
    """
    Reduce a set of driver options to the photometry-affecting subset.

    Parameters
    ----------
    options : dict
        Any option mapping; keys outside :data:`FINGERPRINT_KEYS` are dropped.

    Returns
    -------
    dict
        The subset of `options` whose keys are in :data:`FINGERPRINT_KEYS`.
    """
    return {key: options[key] for key in FINGERPRINT_KEYS if key in options}


class ArchiveLedger:
    """
    The per-night state of one archive run, persisted as JSON.

    Parameters
    ----------
    state_dir : str or `~pathlib.Path`
        Directory holding ``ledger.json``. Created on the first write.

    Attributes
    ----------
    path : `~pathlib.Path`
        The ledger file.
    data : dict
        The whole ledger: ``version``, ``fingerprint``, and ``nights``, a
        mapping of night name to its entry.
    """

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / "ledger.json"
        self.data = {"version": LEDGER_VERSION, "fingerprint": None, "nights": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        self.data.setdefault("version", LEDGER_VERSION)
        self.data.setdefault("fingerprint", None)
        self.data.setdefault("nights", {})

    def save(self):
        """
        Write the ledger atomically.

        The temp file is created in the same directory so `os.replace` is a
        rename within one filesystem, which is atomic. A crash therefore leaves
        either the old ledger or the new one, never a half-written file.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.parent / f"{self.path.name}.tmp"
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True))
        os.replace(tmp, self.path)

    def check_fingerprint(self, fingerprint, force=False):
        """
        Record, or verify, the photometry options this run was started with.

        Parameters
        ----------
        fingerprint : dict
            From :func:`options_fingerprint`.
        force : bool (default=False)
            Accept and overwrite a differing fingerprint instead of raising.

        Raises
        ------
        ValueError
            If the ledger holds a different fingerprint and `force` is False.
            Continuing would assemble a night rollup from per-frame CSVs written
            under different photometry options -- a silently mixed schema that
            nothing downstream would notice.
        """
        stored = self.data.get("fingerprint")
        if stored is None or stored == fingerprint:
            self.data["fingerprint"] = fingerprint
            return
        diff = "\n".join(
            f"  {key}: ledger={stored.get(key)!r} requested={fingerprint.get(key)!r}"
            for key in sorted(set(stored) | set(fingerprint))
            if stored.get(key) != fingerprint.get(key)
        )
        if not force:
            raise ValueError(
                "photometry options differ from the ones this ledger was "
                f"started with:\n{diff}\nPass --force-options to continue "
                "anyway; the combined photometry will mix schemas."
            )
        self.data["fingerprint"] = fingerprint

    def entry(self, night):
        """The ledger entry for `night`, or None if it has never been seen."""
        return self.data["nights"].get(night)

    def state(self, night):
        """The state of `night`: pending, running, done, or failed."""
        entry = self.entry(night)
        return "pending" if entry is None else entry.get("state", "pending")

    def _update(self, night, **fields):
        entry = self.data["nights"].setdefault(night, {"night": night, "attempts": 0})
        entry.update(fields)
        return entry

    def mark_running(self, night):
        """Record that `night` has started, and count the attempt."""
        entry = self._update(night, state="running", started=_now(),
                             finished=None, error=None)
        entry["attempts"] = entry.get("attempts", 0) + 1
        self.save()

    def mark_done(self, night, elapsed, n_frames, n_errors, products):
        """Record a completed night and the products it wrote."""
        self._update(night, state="done", finished=_now(), elapsed_s=elapsed,
                     n_frames=n_frames, n_errors=n_errors,
                     products={key: str(value) for key, value in products.items()
                               if value is not None},
                     error=None)
        self.save()

    def mark_failed(self, night, error, elapsed=None):
        """Record a night that raised, with its message."""
        self._update(night, state="failed", finished=_now(), elapsed_s=elapsed,
                     error=str(error))
        self.save()

    def mark_pruned(self, night, n_files, n_bytes):
        """Record that `night`'s vendor JPEGs were deleted."""
        self._update(night, jpegs_pruned=n_files, bytes_freed=n_bytes)
        self.save()

    def reset_stale_running(self):
        """
        Return any ``running`` night to ``pending`` and report which.

        A night is only ``running`` if a previous run died while processing it.
        Reprocessing is cheap relative to the risk of skipping it: the per-frame
        photometry it did write is reused by `alcor_process_night`.

        Returns
        -------
        list of str
            The night names that were reset, sorted.
        """
        stale = sorted(night for night, entry in self.data["nights"].items()
                       if entry.get("state") == "running")
        for night in stale:
            self._update(night, state="pending")
        if stale:
            self.save()
        return stale

    def was_pruned(self, night):
        """True when `night` has already had its JPEGs deleted."""
        entry = self.entry(night)
        return entry is not None and entry.get("jpegs_pruned") is not None

    def counts(self):
        """A count of nights by state, always containing all four keys."""
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        for entry in self.data["nights"].values():
            state = entry.get("state", "pending")
            counts[state] = counts.get(state, 0) + 1
        return counts

    def timing(self):
        """
        Completed-night timing, for the ETA.

        Returns
        -------
        tuple
            ``(n, mean_elapsed_seconds)`` over nights recorded ``done`` with a
            timing, or ``(0, None)`` when none have finished yet.
        """
        elapsed = [entry["elapsed_s"] for entry in self.data["nights"].values()
                   if entry.get("state") == "done" and entry.get("elapsed_s")]
        if not elapsed:
            return 0, None
        return len(elapsed), sum(elapsed) / len(elapsed)
```

- [ ] **Step 4: Re-export the new names**

In `skycam_utils/alcor/__init__.py`, add a new import block immediately **before** the `from .night import (...)` block (ledger is a leaf; it must not appear to depend on night):

```python
from .ledger import (  # noqa: F401
    LEDGER_VERSION,
    FINGERPRINT_KEYS,
    ArchiveLedger,
    options_fingerprint,
)
```

and add these four names to the `__all__` list, keeping the file's existing ordering style:

```python
    "LEDGER_VERSION",
    "FINGERPRINT_KEYS",
    "ArchiveLedger",
    "options_fingerprint",
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor/ledger.py skycam_utils/alcor/__init__.py skycam_utils/tests/test_alcor_archive.py
git commit -m "alcor: add the archive run ledger

Per-night state persisted as one atomically-replaced JSON file, so an
archive run resumes after a crash and refuses a resume that would change
the photometry options mid-archive."
```

---

### Task 2: Night discovery and the quiet-period rule

Selecting which directories are nights, in what order, and excluding the ones still arriving from the camera host.

**Files:**
- Create: `skycam_utils/alcor/archive.py`
- Modify: `skycam_utils/tests/test_alcor_archive.py`
- Modify: `skycam_utils/alcor/__init__.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `NIGHT_NAME_RE`, `ALCOR_ARCHIVE_MIN_AGE = 24.0`, `_night_date(name: str) -> datetime.date | None`, `_as_date(value) -> datetime.date`, `discover_nights(archive_dir, start=None, end=None, nights=None, reverse=False) -> list[Path]`, `night_last_modified(night_dir, pattern="*.fits.bz2") -> float`, `is_too_recent(night_dir, min_age_hours, pattern="*.fits.bz2", now=None) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_archive.py`:

```python
import time

from skycam_utils.alcor import (
    ALCOR_ARCHIVE_MIN_AGE,
    discover_nights,
    is_too_recent,
    night_last_modified,
)


def _make_archive(root, names=("2025-01-01", "2025-01-02", "2025-01-03")):
    """An archive tree of empty night directories plus the sibling product trees."""
    root = Path(root)
    for name in names:
        (root / name).mkdir(parents=True)
    (root / "keograms").mkdir(exist_ok=True)
    (root / "movies").mkdir(exist_ok=True)
    (root / "notes.txt").write_text("not a night")
    return root


def test_discover_nights_skips_the_sibling_product_trees(tmp_path):
    """keograms/ and movies/ live beside the nights and must never be processed."""
    archive = _make_archive(tmp_path)

    found = [p.name for p in discover_nights(archive)]

    assert found == ["2025-01-01", "2025-01-02", "2025-01-03"]


def test_discover_nights_accepts_underscored_names(tmp_path):
    archive = _make_archive(tmp_path, names=("2025-01-01", "2025_01_02"))

    assert [p.name for p in discover_nights(archive)] == ["2025-01-01", "2025_01_02"]


def test_discover_nights_orders_by_date_and_honours_reverse(tmp_path):
    archive = _make_archive(tmp_path)

    assert [p.name for p in discover_nights(archive, reverse=True)] == [
        "2025-01-03", "2025-01-02", "2025-01-01"
    ]


def test_discover_nights_bounds_the_range(tmp_path):
    archive = _make_archive(tmp_path)

    found = [p.name for p in discover_nights(archive, start="2025-01-02",
                                             end="2025-01-02")]

    assert found == ["2025-01-02"]


def test_discover_nights_takes_an_explicit_list(tmp_path):
    archive = _make_archive(tmp_path)

    found = [p.name for p in discover_nights(archive, nights=["2025-01-03",
                                                              "2025-01-01"])]

    assert found == ["2025-01-01", "2025-01-03"]


def test_discover_nights_raises_for_a_missing_explicit_night(tmp_path):
    archive = _make_archive(tmp_path)

    with pytest.raises(FileNotFoundError, match="2025-12-25"):
        discover_nights(archive, nights=["2025-12-25"])


def test_a_night_still_being_written_is_too_recent(tmp_path):
    """The archive syncs from the camera host; a half-arrived night must be left alone."""
    night = tmp_path / "2025-01-01"
    night.mkdir()
    (night / "2025_01_01__20_00_00.fits.bz2").write_bytes(b"x")

    assert is_too_recent(night, min_age_hours=24.0)


def test_a_quiet_night_is_not_too_recent(tmp_path):
    night = tmp_path / "2025-01-01"
    night.mkdir()
    (night / "2025_01_01__20_00_00.fits.bz2").write_bytes(b"x")

    old = time.time() - 48 * 3600
    os.utime(night / "2025_01_01__20_00_00.fits.bz2", (old, old))
    os.utime(night, (old, old))

    assert not is_too_recent(night, min_age_hours=24.0)


def test_min_age_zero_disables_the_quiet_period(tmp_path):
    night = tmp_path / "2025-01-01"
    night.mkdir()
    (night / "2025_01_01__20_00_00.fits.bz2").write_bytes(b"x")

    assert not is_too_recent(night, min_age_hours=0)


def test_old_filenames_do_not_excuse_a_fresh_mtime(tmp_path):
    """A back-filled old night has old filename stamps but new mtimes -- the
    rule is mtime, which is the only test that catches both failure modes."""
    night = tmp_path / "2024-03-01"
    night.mkdir()
    (night / "2024_03_01__20_00_00.fits.bz2").write_bytes(b"x")

    assert is_too_recent(night, min_age_hours=ALCOR_ARCHIVE_MIN_AGE)


def test_night_last_modified_uses_the_directory_when_it_is_newer(tmp_path):
    """rsync renames its temp file into place, which moves the directory mtime."""
    night = tmp_path / "2025-01-01"
    night.mkdir()
    frame = night / "2025_01_01__20_00_00.fits.bz2"
    frame.write_bytes(b"x")

    old = time.time() - 48 * 3600
    os.utime(frame, (old, old))

    assert night_last_modified(night) > old + 3600
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: FAIL — `ImportError: cannot import name 'discover_nights' from 'skycam_utils.alcor'`

- [ ] **Step 3: Write the discovery module**

Create `skycam_utils/alcor/archive.py`:

```python
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
```

- [ ] **Step 4: Re-export the new names**

In `skycam_utils/alcor/__init__.py`, add after the `from .night import (...)` block:

```python
from .archive import (  # noqa: F401
    ALCOR_ARCHIVE_MIN_AGE,
    NIGHT_NAME_RE,
    _as_date,
    _night_date,
    discover_nights,
    is_too_recent,
    night_last_modified,
)
```

and add those seven names to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: PASS, 20 tests.

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor/archive.py skycam_utils/alcor/__init__.py skycam_utils/tests/test_alcor_archive.py
git commit -m "alcor: discover archive nights and skip the ones still arriving

Night directories are selected by name, which excludes the sibling
keograms/ and movies/ trees. A night whose mtime is inside the quiet
period is left alone: the archive syncs from the camera host, and a
mtime test is the only one that catches both a night still being
observed and an old night being back-filled."
```

---

### Task 3: Throttled progress logging

`alcor_process_night` logs one line per frame. Over a 613-night archive that is about 1.2 million lines, which makes the log useless. This collapses frame lines into a periodic rate-and-ETA line and passes everything else through.

**Files:**
- Modify: `skycam_utils/alcor/archive.py`
- Modify: `skycam_utils/tests/test_alcor_archive.py`
- Modify: `skycam_utils/alcor/__init__.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `ALCOR_ARCHIVE_LOG_INTERVAL = 60.0`, `_duration(seconds: float) -> str`, class `ThrottledLog(emit, prefix="", interval=ALCOR_ARCHIVE_LOG_INTERVAL, verbose=False, clock=time.monotonic)` — callable with one message argument, with method `progress() -> str` and attributes `done: int`, `total: int`.

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_archive.py`:

```python
from skycam_utils.alcor import ThrottledLog, _duration


class _FakeClock:
    """A monotonic clock the test drives by hand."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def test_throttled_log_passes_non_frame_messages_through():
    lines = []
    log = ThrottledLog(lines.append, clock=_FakeClock())

    log("wrote /out/2025-01-01/sky_brightness.csv")

    assert lines == ["wrote /out/2025-01-01/sky_brightness.csv"]


def test_throttled_log_swallows_frame_lines_inside_the_interval():
    clock = _FakeClock()
    lines = []
    log = ThrottledLog(lines.append, interval=60.0, clock=clock)

    for i in range(1, 51):
        clock.advance(1.0)
        log(f"[{i}/2000] 2025_01_01__20_00_00.fits.bz2")

    assert lines == []
    assert log.done == 50
    assert log.total == 2000


def test_throttled_log_emits_once_the_interval_has_passed():
    clock = _FakeClock()
    lines = []
    log = ThrottledLog(lines.append, interval=60.0, clock=clock)

    clock.advance(30.0)
    log("[30/2000] a.fits.bz2")
    clock.advance(31.0)
    log("[61/2000] b.fits.bz2")

    assert len(lines) == 1
    assert "frames 61/2000" in lines[0]
    assert "f/s" in lines[0]


def test_throttled_log_applies_its_prefix():
    clock = _FakeClock()
    lines = []
    log = ThrottledLog(lines.append, prefix="[ 12/613] 2025-01-12  ", clock=clock)

    log("no horizon mask found; not applied")

    assert lines == ["[ 12/613] 2025-01-12  no horizon mask found; not applied"]


def test_verbose_throttled_log_passes_every_frame_line():
    clock = _FakeClock()
    lines = []
    log = ThrottledLog(lines.append, interval=60.0, verbose=True, clock=clock)

    log("[1/2000] a.fits.bz2")
    log("[2/2000] b.fits.bz2")

    assert lines == ["[1/2000] a.fits.bz2", "[2/2000] b.fits.bz2"]


@pytest.mark.parametrize("seconds,expected", [
    (45.0, "45s"),
    (930.0, "15m30s"),
    (19934.0, "5h32m"),
    (1140000.0, "13.2d"),
])
def test_duration_formats_readably(seconds, expected):
    assert _duration(seconds) == expected


def test_duration_handles_an_unknown_rate():
    assert _duration(float("nan")) == "?"
    assert _duration(float("inf")) == "?"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: FAIL — `ImportError: cannot import name 'ThrottledLog' from 'skycam_utils.alcor'`

- [ ] **Step 3: Implement the throttle**

Add to `skycam_utils/alcor/archive.py`, after `is_too_recent`:

```python
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
```

- [ ] **Step 4: Re-export the new names**

Add `ALCOR_ARCHIVE_LOG_INTERVAL`, `ThrottledLog`, `_duration`, and `_FRAME_LINE_RE` to the `from .archive import (...)` block in `skycam_utils/alcor/__init__.py`, and the first three to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: PASS, 30 tests.

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor/archive.py skycam_utils/alcor/__init__.py skycam_utils/tests/test_alcor_archive.py
git commit -m "alcor: throttle the archive driver's per-frame log lines

One line per frame is a million lines over an archive. Collapse them to
a rate-and-ETA line a minute, and pass every other message straight
through so nothing that happens once is hidden."
```

---

### Task 4: JPEG pruning

The vendor software writes two JPEGs per frame, ~7.3 GB a night and ~4.5 TB across the archive. Better renderings come from the raw FITS on demand, so they are redundant — but deleting them is the one irreversible thing this driver does, so the gates matter more than the mechanism.

**Files:**
- Modify: `skycam_utils/alcor/archive.py`
- Modify: `skycam_utils/tests/test_alcor_archive.py`
- Modify: `skycam_utils/alcor/__init__.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-3.
- Produces: `ALCOR_ARCHIVE_PRUNE_MAX_ERROR_FRACTION = 0.10`, `prune_night_jpegs(night_dir, products, n_frames, n_errors, dry_run=False, max_error_fraction=ALCOR_ARCHIVE_PRUNE_MAX_ERROR_FRACTION) -> tuple[int, int, str | None]` returning `(n_files, n_bytes, reason)` where `reason` is None when the prune ran and otherwise says why it did not.

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_archive.py`:

```python
from skycam_utils.alcor import prune_night_jpegs


def _make_night_with_jpegs(root, name="2025-01-01", n=3):
    """A night directory holding both JPEG series and the FITS frames."""
    night = Path(root) / name
    night.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        stamp = f"2025_01_01__20_00_{i:02d}"
        (night / f"{stamp}.jpg").write_bytes(b"j" * 100)
        (night / f"Unwrap_{stamp}.jpg").write_bytes(b"u" * 50)
        (night / f"{stamp}.fits.bz2").write_bytes(b"f" * 10)
    return night


def _good_products(out_dir):
    """The two product files pruning is gated on, written non-empty."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "sky_brightness.csv"
    photometry = out_dir / "2025-01-01_phot.csv"
    summary.write_text("filename,OBSTIME\n")
    photometry.write_text("name,OBSTIME\n")
    return {"summary_file": summary, "photometry_file": photometry}


def test_prune_deletes_both_jpeg_series_but_no_fits(tmp_path):
    night = _make_night_with_jpegs(tmp_path / "archive")
    products = _good_products(tmp_path / "out" / "2025-01-01")

    n_files, n_bytes, reason = prune_night_jpegs(night, products, n_frames=3,
                                                 n_errors=0)

    assert reason is None
    assert n_files == 6
    assert n_bytes == 3 * 150
    assert sorted(p.name for p in night.iterdir()) == [
        "2025_01_01__20_00_00.fits.bz2",
        "2025_01_01__20_00_01.fits.bz2",
        "2025_01_01__20_00_02.fits.bz2",
    ]


def test_prune_dry_run_reports_but_deletes_nothing(tmp_path):
    night = _make_night_with_jpegs(tmp_path / "archive")
    products = _good_products(tmp_path / "out" / "2025-01-01")

    n_files, n_bytes, reason = prune_night_jpegs(night, products, n_frames=3,
                                                 n_errors=0, dry_run=True)

    assert (n_files, n_bytes, reason) == (6, 450, None)
    assert len(list(night.glob("*.jpg"))) == 6


def test_prune_refuses_when_the_summary_is_missing(tmp_path):
    """The JPEGs are the only other copy of the imagery; no products, no prune."""
    night = _make_night_with_jpegs(tmp_path / "archive")
    products = _good_products(tmp_path / "out" / "2025-01-01")
    Path(products["summary_file"]).unlink()

    n_files, n_bytes, reason = prune_night_jpegs(night, products, n_frames=3,
                                                 n_errors=0)

    assert (n_files, n_bytes) == (0, 0)
    assert "sky_brightness.csv" in reason
    assert len(list(night.glob("*.jpg"))) == 6


def test_prune_refuses_when_the_photometry_rollup_is_empty(tmp_path):
    night = _make_night_with_jpegs(tmp_path / "archive")
    products = _good_products(tmp_path / "out" / "2025-01-01")
    Path(products["photometry_file"]).write_text("")

    n_files, _, reason = prune_night_jpegs(night, products, n_frames=3, n_errors=0)

    assert n_files == 0
    assert "photometry" in reason


def test_prune_refuses_when_too_many_frames_failed(tmp_path):
    night = _make_night_with_jpegs(tmp_path / "archive")
    products = _good_products(tmp_path / "out" / "2025-01-01")

    n_files, _, reason = prune_night_jpegs(night, products, n_frames=100,
                                           n_errors=20)

    assert n_files == 0
    assert "20 of 100" in reason
    assert len(list(night.glob("*.jpg"))) == 6


def test_prune_allows_a_few_frame_errors(tmp_path):
    night = _make_night_with_jpegs(tmp_path / "archive")
    products = _good_products(tmp_path / "out" / "2025-01-01")

    n_files, _, reason = prune_night_jpegs(night, products, n_frames=100,
                                           n_errors=5)

    assert reason is None
    assert n_files == 6
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: FAIL — `ImportError: cannot import name 'prune_night_jpegs' from 'skycam_utils.alcor'`

- [ ] **Step 3: Implement pruning**

Add to `skycam_utils/alcor/archive.py`, after `ThrottledLog`:

```python
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
```

- [ ] **Step 4: Re-export the new names**

Add `ALCOR_ARCHIVE_PRUNE_MAX_ERROR_FRACTION` and `prune_night_jpegs` to the `from .archive import (...)` block and to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: PASS, 36 tests.

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor/archive.py skycam_utils/alcor/__init__.py skycam_utils/tests/test_alcor_archive.py
git commit -m "alcor: prune a completed night's redundant vendor JPEGs

Gated on the night's products existing non-empty and on its frames
having mostly succeeded, with a dry run that reports without deleting.
This is the only irreversible operation in the archive driver."
```

---

### Task 5: The driver loop

Everything so far, tied together: iterate nights, honour the ledger, pause on the sentinel file, stop cleanly on a signal, and never let one bad night cost the rest.

**Files:**
- Modify: `skycam_utils/alcor/archive.py`
- Modify: `skycam_utils/tests/test_alcor_archive.py`
- Modify: `skycam_utils/alcor/__init__.py`

**Interfaces:**
- Consumes: `ArchiveLedger`, `options_fingerprint` (Task 1); `discover_nights`, `is_too_recent`, `ALCOR_ARCHIVE_MIN_AGE` (Task 2); `ThrottledLog`, `_duration` (Task 3); `prune_night_jpegs` (Task 4); `alcor_process_night` from `.night`.
- Produces: `ALCOR_ARCHIVE_MAX_CONSECUTIVE_FAILURES = 10`, `ALCOR_ARCHIVE_PAUSE_POLL = 30.0`, `_wait_while_paused(pause_file, poll, log, stop, sleep=time.sleep) -> bool`, and `alcor_process_archive(archive_dir, out_dir, ...) -> dict` with keys `ledger`, `processed`, `failed`, `skipped_recent`, `skipped_failed`, `pruned_files`, `pruned_bytes`, `stopped`.

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_archive.py`. These stub `alcor_process_night` with the `patch_alcor` fixture — the driver's job is state management, so no real frames are needed.

```python
from skycam_utils.alcor import _wait_while_paused, alcor_process_archive


def _stub_night(record, fail_on=(), out_root=None):
    """A stand-in for `alcor_process_night` that writes plausible products."""

    def _run(night_dir, out_dir=None, log=None, **kwargs):
        night = Path(night_dir).name
        record.append(night)
        if night in fail_on:
            raise ValueError(f"synthetic failure for {night}")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = out_dir / "sky_brightness.csv"
        photometry = out_dir / f"{night}_phot.csv"
        summary.write_text("filename,OBSTIME\n")
        photometry.write_text("name,OBSTIME\n")
        if log is not None:
            log("[1/2] a.fits.bz2")
            log(f"wrote {summary}")
        return {"summary_file": summary, "photometry_file": photometry,
                "keogram_file": None, "files": ["a", "b"], "errors": []}

    return _run


def _aged_archive(root, names=("2025-01-01", "2025-01-02", "2025-01-03")):
    """An archive whose nights are all old enough to process."""
    archive = _make_archive(root, names=names)
    old = time.time() - 72 * 3600
    for name in names:
        night = archive / name
        (night / f"{name.replace('-', '_')}__20_00_00.fits.bz2").write_bytes(b"x")
        for path in sorted(night.iterdir()) + [night]:
            os.utime(path, (old, old))
    return archive


def test_archive_run_processes_every_night_and_records_it(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    seen = []
    patch_alcor("alcor_process_night", _stub_night(seen))

    result = alcor_process_archive(archive, out_dir=tmp_path / "out")

    assert seen == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert result["ledger"].counts()["done"] == 3
    assert (tmp_path / "out" / "2025-01-01" / "sky_brightness.csv").exists()
    assert (tmp_path / "out" / ".archive_state" / "ledger.json").exists()
    assert (tmp_path / "out" / ".archive_state" / "archive.log").exists()


def test_a_second_run_skips_completed_nights(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    first = []
    patch_alcor("alcor_process_night", _stub_night(first))
    alcor_process_archive(archive, out_dir=tmp_path / "out")

    second = []
    patch_alcor("alcor_process_night", _stub_night(second))
    alcor_process_archive(archive, out_dir=tmp_path / "out")

    assert second == []


def test_a_failing_night_is_recorded_and_the_run_continues(tmp_path, patch_alcor):
    """One bad night must not cost the other 612."""
    archive = _aged_archive(tmp_path / "archive")
    seen = []
    patch_alcor("alcor_process_night", _stub_night(seen, fail_on=("2025-01-02",)))

    result = alcor_process_archive(archive, out_dir=tmp_path / "out")

    assert seen == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert result["failed"] == ["2025-01-02"]
    assert result["ledger"].state("2025-01-02") == "failed"
    assert result["ledger"].counts()["done"] == 2


def test_failed_nights_are_skipped_until_retry_failed(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    patch_alcor("alcor_process_night", _stub_night([], fail_on=("2025-01-02",)))
    alcor_process_archive(archive, out_dir=tmp_path / "out")

    seen = []
    patch_alcor("alcor_process_night", _stub_night(seen))
    alcor_process_archive(archive, out_dir=tmp_path / "out")
    assert seen == []

    retried = []
    patch_alcor("alcor_process_night", _stub_night(retried))
    alcor_process_archive(archive, out_dir=tmp_path / "out", retry_failed=True)
    assert retried == ["2025-01-02"]


def test_a_night_still_arriving_is_skipped_and_left_pending(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    fresh = archive / "2025-01-02" / "2025_01_02__21_00_00.fits.bz2"
    fresh.write_bytes(b"x")  # touches the file and the directory mtime
    seen = []
    patch_alcor("alcor_process_night", _stub_night(seen))

    result = alcor_process_archive(archive, out_dir=tmp_path / "out")

    assert seen == ["2025-01-01", "2025-01-03"]
    assert result["skipped_recent"] == ["2025-01-02"]
    assert result["ledger"].state("2025-01-02") == "pending"


def test_min_age_zero_processes_a_night_still_arriving(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    (archive / "2025-01-02" / "2025_01_02__21_00_00.fits.bz2").write_bytes(b"x")
    seen = []
    patch_alcor("alcor_process_night", _stub_night(seen))

    alcor_process_archive(archive, out_dir=tmp_path / "out", min_age=0)

    assert seen == ["2025-01-01", "2025-01-02", "2025-01-03"]


def test_a_crashed_night_is_reset_to_pending_and_reprocessed(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    state_dir = tmp_path / "out" / ".archive_state"
    state_dir.mkdir(parents=True)
    ledger = ArchiveLedger(state_dir)
    ledger.mark_running("2025-01-02")

    seen = []
    patch_alcor("alcor_process_night", _stub_night(seen))
    alcor_process_archive(archive, out_dir=tmp_path / "out")

    assert "2025-01-02" in seen


def test_changed_photometry_options_abort_the_run(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    patch_alcor("alcor_process_night", _stub_night([]))
    alcor_process_archive(archive, out_dir=tmp_path / "out", both=True)

    with pytest.raises(ValueError, match="both"):
        alcor_process_archive(archive, out_dir=tmp_path / "out", both=False)

    alcor_process_archive(archive, out_dir=tmp_path / "out", both=False,
                          force_options=True)


def test_the_run_aborts_after_too_many_consecutive_failures(tmp_path, patch_alcor):
    """An unmounted archive fails every night instantly; stop rather than burn through."""
    names = tuple(f"2025-02-{day:02d}" for day in range(1, 11))
    archive = _aged_archive(tmp_path / "archive", names=names)
    seen = []
    patch_alcor("alcor_process_night", _stub_night(seen, fail_on=names))

    result = alcor_process_archive(archive, out_dir=tmp_path / "out",
                                   max_consecutive_failures=3)

    assert len(seen) == 3
    assert result["stopped"] is True


def test_pause_file_halts_before_the_next_night(tmp_path, patch_alcor):
    """The night in flight completes; the pause lands at the boundary."""
    archive = _aged_archive(tmp_path / "archive")
    pause = tmp_path / "out" / ".archive_state" / "PAUSE"
    seen = []
    inner = _stub_night(seen)

    def _run(night_dir, **kwargs):
        result = inner(night_dir, **kwargs)
        if Path(night_dir).name == "2025-01-01":
            pause.parent.mkdir(parents=True, exist_ok=True)
            pause.write_text("")
        return result

    patch_alcor("alcor_process_night", _run)

    removals = []

    def _sleep(_seconds):
        removals.append(1)
        pause.unlink()  # simulate the operator removing it

    result = alcor_process_archive(archive, out_dir=tmp_path / "out",
                                   pause_sleep=_sleep)

    assert seen == ["2025-01-01", "2025-01-02", "2025-01-03"]
    assert removals == [1]
    assert result["ledger"].state("2025-01-01") == "done"


def test_a_pause_file_present_at_startup_holds_the_first_night(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    pause = tmp_path / "out" / ".archive_state" / "PAUSE"
    pause.parent.mkdir(parents=True)
    pause.write_text("")
    seen = []
    patch_alcor("alcor_process_night", _stub_night(seen))

    def _sleep(_seconds):
        pause.unlink()

    alcor_process_archive(archive, out_dir=tmp_path / "out", pause_sleep=_sleep)

    assert seen == ["2025-01-01", "2025-01-02", "2025-01-03"]


def test_wait_while_paused_returns_immediately_without_the_file(tmp_path):
    stop = {"requested": False}
    assert _wait_while_paused(tmp_path / "PAUSE", 30.0, lambda m: None,
                              stop) is False


def test_wait_while_paused_gives_up_when_stop_is_requested(tmp_path):
    pause = tmp_path / "PAUSE"
    pause.write_text("")
    stop = {"requested": False}

    def _sleep(_seconds):
        stop["requested"] = True

    assert _wait_while_paused(pause, 30.0, lambda m: None, stop,
                              sleep=_sleep) is True
    assert pause.exists()


def test_prune_jpegs_deletes_only_for_a_successful_night(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    for name in ("2025-01-01", "2025-01-02", "2025-01-03"):
        for i in range(2):
            (archive / name / f"{name}_{i}.jpg").write_bytes(b"j" * 10)
    patch_alcor("alcor_process_night", _stub_night([], fail_on=("2025-01-02",)))

    result = alcor_process_archive(archive, out_dir=tmp_path / "out",
                                   min_age=0, prune_jpegs=True)

    assert list((archive / "2025-01-01").glob("*.jpg")) == []
    assert len(list((archive / "2025-01-02").glob("*.jpg"))) == 2
    assert result["pruned_files"] == 4


def test_prune_dry_run_deletes_nothing(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    for i in range(2):
        (archive / "2025-01-01" / f"a_{i}.jpg").write_bytes(b"j" * 10)
    patch_alcor("alcor_process_night", _stub_night([]))

    result = alcor_process_archive(archive, out_dir=tmp_path / "out",
                                   min_age=0, prune_dry_run=True)

    assert len(list((archive / "2025-01-01").glob("*.jpg"))) == 2
    assert result["pruned_files"] == 2


def test_prune_never_touches_a_night_skipped_as_too_recent(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    fresh = archive / "2025-01-02"
    (fresh / "keep_me.jpg").write_bytes(b"j" * 10)
    patch_alcor("alcor_process_night", _stub_night([]))

    alcor_process_archive(archive, out_dir=tmp_path / "out", prune_jpegs=True)

    assert (fresh / "keep_me.jpg").exists()


def test_pruning_can_be_enabled_on_a_later_pass(tmp_path, patch_alcor):
    """A night already done but never pruned is still prunable."""
    archive = _aged_archive(tmp_path / "archive")
    (archive / "2025-01-01" / "a.jpg").write_bytes(b"j" * 10)
    patch_alcor("alcor_process_night", _stub_night([]))
    alcor_process_archive(archive, out_dir=tmp_path / "out", min_age=0)
    assert (archive / "2025-01-01" / "a.jpg").exists()

    alcor_process_archive(archive, out_dir=tmp_path / "out", min_age=0,
                          prune_jpegs=True)

    assert not (archive / "2025-01-01" / "a.jpg").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: FAIL — `ImportError: cannot import name 'alcor_process_archive' from 'skycam_utils.alcor'`

- [ ] **Step 3: Implement the driver**

Add the import at the top of `skycam_utils/alcor/archive.py`, below the stdlib imports:

```python
import signal

from .ledger import ArchiveLedger, options_fingerprint
from .night import alcor_process_night
```

Add to the end of `skycam_utils/alcor/archive.py`:

```python
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
```

- [ ] **Step 4: Re-export the new names**

Add `ALCOR_ARCHIVE_MAX_CONSECUTIVE_FAILURES`, `ALCOR_ARCHIVE_PAUSE_POLL`, `_wait_while_paused`, and `alcor_process_archive` to the `from .archive import (...)` block, and all but `_wait_while_paused` to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: PASS, 53 tests.

- [ ] **Step 6: Run the whole suite to check nothing regressed**

Run: `pytest -q`
Expected: PASS, no failures.

- [ ] **Step 7: Commit**

```bash
git add skycam_utils/alcor/archive.py skycam_utils/alcor/__init__.py skycam_utils/tests/test_alcor_archive.py
git commit -m "alcor: add the archive-wide driver loop

Iterates an archive's nights through alcor_process_night with the
ledger for resume, a PAUSE sentinel that holds a detached run without
its PID, a consecutive-failure abort for the unmounted-drive case, and
opt-in JPEG pruning gated on the night having succeeded."
```

---

### Task 6: Status reporting

A 13-day run needs to be inspectable without touching it, and from a machine where the archive is not even mounted.

**Files:**
- Modify: `skycam_utils/alcor/archive.py`
- Modify: `skycam_utils/tests/test_alcor_archive.py`
- Modify: `skycam_utils/alcor/__init__.py`

**Interfaces:**
- Consumes: `ArchiveLedger` (Task 1), `_duration` (Task 3).
- Produces: `alcor_archive_status(out_dir, log=print) -> dict` returning `counts`, `elapsed_mean`, `eta`, `failed`, `pruned_bytes`.

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_archive.py`:

```python
from skycam_utils.alcor import alcor_archive_status


def test_status_summarises_a_ledger_without_the_archive(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    patch_alcor("alcor_process_night", _stub_night([], fail_on=("2025-01-03",)))
    alcor_process_archive(archive, out_dir=tmp_path / "out")

    lines = []
    status = alcor_archive_status(tmp_path / "out", log=lines.append)

    assert status["counts"]["done"] == 2
    assert status["counts"]["failed"] == 1
    assert status["failed"] == ["2025-01-03"]
    assert any("done" in line for line in lines)


def test_status_reports_the_reason_a_night_failed(tmp_path, patch_alcor):
    archive = _aged_archive(tmp_path / "archive")
    patch_alcor("alcor_process_night", _stub_night([], fail_on=("2025-01-03",)))
    alcor_process_archive(archive, out_dir=tmp_path / "out")

    lines = []
    alcor_archive_status(tmp_path / "out", log=lines.append)

    assert any("synthetic failure" in line for line in lines)


def test_status_raises_without_a_ledger(tmp_path):
    with pytest.raises(FileNotFoundError, match="ledger.json"):
        alcor_archive_status(tmp_path / "nowhere")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: FAIL — `ImportError: cannot import name 'alcor_archive_status' from 'skycam_utils.alcor'`

- [ ] **Step 3: Implement the status reporter**

Add to the end of `skycam_utils/alcor/archive.py`:

```python
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
```

- [ ] **Step 4: Re-export the new name**

Add `alcor_archive_status` to the `from .archive import (...)` block and to `__all__`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: PASS, 56 tests.

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor/archive.py skycam_utils/alcor/__init__.py skycam_utils/tests/test_alcor_archive.py
git commit -m "alcor: report an archive run's progress from its ledger

Reads only the ledger, so it works while a run is in flight and with the
archive unmounted."
```

---

### Task 7: CLI entry point and documentation

**Files:**
- Modify: `skycam_utils/alcor/cli.py` (add `alcor_process_archive_cli` after `alcor_process_night_cli`)
- Modify: `skycam_utils/alcor/__init__.py`
- Modify: `pyproject.toml:[project.scripts]`
- Modify: `CLAUDE.md`
- Modify: `skycam_utils/tests/test_alcor_archive.py`

**Interfaces:**
- Consumes: `alcor_process_archive`, `alcor_archive_status` (Tasks 5-6).
- Produces: `alcor_process_archive_cli()`, and the `alcor_process_archive` console script.

- [ ] **Step 1: Write the failing test**

Append to `skycam_utils/tests/test_alcor_archive.py`:

```python
import sys

from skycam_utils.alcor import alcor_process_archive_cli


def test_cli_runs_an_archive(tmp_path, patch_alcor, monkeypatch):
    archive = _aged_archive(tmp_path / "archive")
    seen = []
    patch_alcor("alcor_process_night", _stub_night(seen))
    monkeypatch.setattr(sys, "argv", [
        "alcor_process_archive", str(archive), "-o", str(tmp_path / "out"),
        "--min-age", "0", "--workers", "1",
    ])

    alcor_process_archive_cli()

    assert seen == ["2025-01-01", "2025-01-02", "2025-01-03"]


def test_cli_status_mode_needs_no_archive_argument(tmp_path, patch_alcor,
                                                   monkeypatch, capsys):
    archive = _aged_archive(tmp_path / "archive")
    patch_alcor("alcor_process_night", _stub_night([]))
    alcor_process_archive(archive, out_dir=tmp_path / "out", min_age=0)

    monkeypatch.setattr(sys, "argv", [
        "alcor_process_archive", "--status", "-o", str(tmp_path / "out"),
    ])
    alcor_process_archive_cli()

    assert "done 3" in capsys.readouterr().out


def test_cli_requires_an_archive_dir_outside_status_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "alcor_process_archive", "-o", str(tmp_path / "out"),
    ])

    with pytest.raises(SystemExit):
        alcor_process_archive_cli()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: FAIL — `ImportError: cannot import name 'alcor_process_archive_cli' from 'skycam_utils.alcor'`

- [ ] **Step 3: Add the CLI**

In `skycam_utils/alcor/cli.py`, extend the existing night import:

```python
from .night import alcor_process_night
from .archive import alcor_archive_status, alcor_process_archive
```

and add this function immediately after `alcor_process_night_cli`:

```python
def alcor_process_archive_cli():
    """
    CLI entry point for :func:`alcor_process_archive`: run every night of an
    archive through the night driver, resumably.
    """
    parser = argparse.ArgumentParser(
        description="Process every night of a skycam archive into star "
                    "photometry, sky-brightness summaries, and keograms. "
                    "Resumable: rerun the same command to continue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("archive_dir", nargs="?", default=None,
                        help="Archive root holding one directory per night. "
                             "Optional only with --status.")
    parser.add_argument("-o", "--out-dir", required=True,
                        help="Products tree. All state lives in "
                             "<out-dir>/.archive_state/ (ledger.json, PAUSE, "
                             "archive.log); each night writes to <out-dir>/<night>/.")
    parser.add_argument("--status", action="store_true",
                        help="Report the run's progress from the ledger and exit. "
                             "Does not need the archive mounted.")
    parser.add_argument("--start", default=None,
                        help="Earliest night to process (YYYY-MM-DD, inclusive).")
    parser.add_argument("--end", default=None,
                        help="Latest night to process (YYYY-MM-DD, inclusive).")
    parser.add_argument("--nights", nargs="+", default=None,
                        help="Explicit night directory names instead of scanning.")
    parser.add_argument("--reverse", action="store_true",
                        help="Process newest nights first.")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob for input frames.")
    parser.add_argument("--min-age", type=float, default=ALCOR_ARCHIVE_MIN_AGE,
                        help="Hours a night must be unchanged before it is "
                             "processed. The archive syncs from the camera host, "
                             "so a night modified more recently may still be "
                             "arriving. 0 disables the rule.")
    parser.add_argument("--prune-jpegs", action="store_true",
                        help="DELETE each night's vendor JPEGs once its products "
                             "are written and verified. Irreversible; frees "
                             "~7.3 GB per night.")
    parser.add_argument("--prune-dry-run", action="store_true",
                        help="Report what --prune-jpegs would free, deleting "
                             "nothing. Overrides --prune-jpegs.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt nights previously recorded as failed.")
    parser.add_argument("--force-options", action="store_true",
                        help="Continue even though the photometry options differ "
                             "from the ones this ledger was started with. The "
                             "combined photometry will mix schemas.")
    parser.add_argument("--max-consecutive-failures", type=int,
                        default=ALCOR_ARCHIVE_MAX_CONSECUTIVE_FAILURES,
                        help="Abort after this many nights fail in a row (an "
                             "unmounted archive fails every one instantly).")
    parser.add_argument("--verbose", action="store_true",
                        help="Log every per-frame line instead of a throttled "
                             "progress line each minute.")
    # Forwarded to alcor_process_night.
    parser.add_argument("--sun-alt-max", type=float, default=-12.0,
                        help="Night is the Sun below this altitude (deg).")
    parser.add_argument("--day-keogram", action="store_true",
                        help="Also build each night's full-day raw RGB keogram.")
    parser.add_argument("--median-stack", action="store_true",
                        help="Also build each night's raw median stack.")
    parser.add_argument("--write-sb-fits", action="store_true",
                        help="Keep each frame's full surface-brightness map.")
    parser.add_argument("--no-horizon-mask", action="store_true",
                        help="Do not blank not-sky pixels.")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-measure star photometry even where "
                             "<frame>_phot.csv already exists.")
    parser.add_argument("--aperture-radius", type=float, default=4.0,
                        help="Star aperture radius in pixels.")
    parser.add_argument("--annulus-width", type=float, default=1.0,
                        help="Star background annulus width in pixels.")
    parser.add_argument("--min-altitude", type=float, default=20.0,
                        help="Minimum catalog-star altitude to measure (deg).")
    parser.add_argument("--vmag-limit", type=float, default=5.5,
                        help="Faintest catalog star Vmag to measure.")
    parser.add_argument("--no-variables", dest="variables", action="store_false",
                        help="Do not measure the bright-variable catalog.")
    parser.add_argument("--gaussian", action="store_true",
                        help="Use constrained-Gaussian PSF photometry.")
    parser.add_argument("--both", action="store_true",
                        help="Measure aperture AND Gaussian photometry.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Directory for the median-stack scratch memmap. "
                             "Point this at local storage, not the archive.")
    parser.add_argument("--masks-dir", default=None,
                        help="Override the bad-pixel masks directory.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes per night (default: one per core).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Strided-subsample each night to at most this many "
                             "frames. For smoke tests, not production runs.")
    args = parser.parse_args()

    if args.status:
        alcor_archive_status(args.out_dir)
        return
    if args.archive_dir is None:
        parser.error("archive_dir is required unless --status is given")

    alcor_process_archive(
        args.archive_dir, out_dir=args.out_dir, start=args.start, end=args.end,
        nights=args.nights, reverse=args.reverse, pattern=args.pattern,
        min_age=args.min_age, prune_jpegs=args.prune_jpegs,
        prune_dry_run=args.prune_dry_run, retry_failed=args.retry_failed,
        force_options=args.force_options,
        max_consecutive_failures=args.max_consecutive_failures,
        verbose=args.verbose, log=lambda message: print(message, file=sys.stderr),
        install_signals=True,
        sun_alt_max=args.sun_alt_max, day_keogram=args.day_keogram,
        median_stack=args.median_stack, write_sb_fits=args.write_sb_fits,
        horizon_mask=not args.no_horizon_mask, reprocess=args.reprocess,
        aperture_radius=args.aperture_radius, annulus_width=args.annulus_width,
        min_altitude=args.min_altitude, vmag_limit=args.vmag_limit,
        variables=args.variables, gaussian=args.gaussian, both=args.both,
        scratch_dir=args.scratch_dir, masks_dir=args.masks_dir,
        workers=args.workers, max_frames=args.max_frames,
    )
```

Add the two constants to the existing `from .config import (...)`-style import block at the top of `cli.py` by extending the archive import:

```python
from .archive import (
    ALCOR_ARCHIVE_MAX_CONSECUTIVE_FAILURES, ALCOR_ARCHIVE_MIN_AGE,
    alcor_archive_status, alcor_process_archive
)
```

(replacing the two-name version added above).

- [ ] **Step 4: Register the entry point**

In `skycam_utils/alcor/__init__.py`, add `alcor_process_archive_cli` to the `from .cli import (...)` block and to `__all__`.

In `pyproject.toml`, add to `[project.scripts]` after the `alcor_process_night` line:

```toml
alcor_process_archive = "skycam_utils.alcor:alcor_process_archive_cli"
```

- [ ] **Step 5: Reinstall so the console script exists**

Run: `pip install -e ".[test]"`
Expected: completes, and `which alcor_process_archive` prints a path.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_archive.py -v`
Expected: PASS, 59 tests.

Run: `pytest -q`
Expected: PASS, no failures.

- [ ] **Step 7: Document the CLI in CLAUDE.md**

In the `## Common commands` fenced block, add this entry after the `alcor_process_night` one:

```
alcor_process_archive <archive-dir> -o <out-dir> [--status] [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--nights N ...] [--reverse] [--min-age 24] [--prune-jpegs] [--prune-dry-run] [--retry-failed] [--force-options] [--max-consecutive-failures 10] [--verbose] [--day-keogram] [--median-stack] [--both] [--workers N] [--scratch-dir DIR]
#   Archive-wide driver: runs alcor_process_night over every YYYY-MM-DD night
#   directory under <archive-dir>, resumably. Hundreds of nights and days of wall
#   clock, so it adds only what that scale needs and no science of its own.
#   ALL STATE LIVES IN <out-dir>/.archive_state/, whose absolute path is the first
#   line of every run:
#     ledger.json   per-night state (pending/running/done/failed), timings, frame
#                   and error counts, product paths, jpegs pruned. Rewritten
#                   atomically after each night; `done` nights are skipped on a
#                   rerun, and a night left `running` by a crash is reset to
#                   `pending`. Frame-level resume then comes free, since
#                   alcor_process_night reuses a non-empty <frame>_phot.csv.
#                   It also stores an OPTIONS FINGERPRINT (both/gaussian/
#                   aperture_radius/annulus_width/min_altitude/vmag_limit/
#                   variables); resuming with any of them changed is REFUSED
#                   unless --force-options, because a rollup built from CSVs
#                   written in two modes is a silently mixed schema.
#     PAUSE         create this file to pause: `touch <out-dir>/.archive_state/PAUSE`
#                   and `rm` it to resume. Checked BETWEEN nights, so a pause lands
#                   at the next boundary (up to ~30 min) and the night in flight
#                   always completes. It is a file, not a signal, so a detached
#                   run can be held without finding its PID. One left in place
#                   holds the NEXT run at its first night.
#     archive.log   the run log, also echoed to stderr.
#   Each night's products go to <out-dir>/<night>/. The archive is read-only
#   except for --prune-jpegs.
#   SKIPS NIGHTS STILL ARRIVING: the archive syncs from the camera host, so a night
#   whose mtime is inside --min-age (24 h) is skipped, logged, and left `pending`
#   -- not `failed`, since nothing failed. The test is MTIME, never the filename
#   timestamps: the current night is still being observed (recent names), while an
#   old night being back-filled has old names and fresh mtimes, and only mtime
#   catches both. Without this a half-arrived night would be recorded `done` and
#   never revisited. --min-age 0 disables it.
#   --prune-jpegs DELETES a night's vendor JPEGs (~7.3 GB/night; "*.jpg" covers the
#   Unwrap_ series too) after its products are written, gated on sky_brightness.csv
#   and <night>_phot.csv existing non-empty and under 10% frame errors. Never
#   prunes a night skipped as too recent. It also prunes already-`done` nights that
#   were never pruned, so it can be enabled partway through or as a later pass.
#   --prune-dry-run reports and deletes nothing.
#   Per-frame log lines are throttled to one progress line a minute (--verbose
#   disables); other messages pass through. One bad night is recorded `failed` and
#   the run continues, but --max-consecutive-failures (10) aborts the run, which is
#   what catches the archive unmounting mid-run. --retry-failed re-attempts them.
#   --status reports counts, per-night timing, an ETA, and the failure list from the
#   ledger alone -- no archive needed, and safe while a run is in flight.
#   Reusable as alcor_process_archive(archive_dir, out_dir, ...) and
#   alcor_archive_status(out_dir).
```

Then update the `alcor` package paragraph's dependency chain, replacing:

```
(`config → timeutils → wcs → masks → {badpix, horizon, catalogs} → io → {photometry, display, skybright} → keogram → night → cli`)
```

with:

```
(`config → timeutils → wcs → masks → {badpix, horizon, catalogs} → io → {photometry, display, skybright} → keogram → night → archive → cli`, with `ledger` a dependency-free leaf that only `archive` imports)
```

and change "`cli` (all 14 entry points)" to "`cli` (all 15 entry points)", and add to the module list "`ledger` (the archive run ledger), `archive` (`alcor_process_archive`)".

- [ ] **Step 8: Verify the docs match reality**

Run: `alcor_process_archive --help`
Expected: the flags listed in CLAUDE.md all appear.

Run: `grep -c alcor_process_archive CLAUDE.md pyproject.toml`
Expected: non-zero for both.

- [ ] **Step 9: Commit**

```bash
git add skycam_utils/alcor/cli.py skycam_utils/alcor/__init__.py pyproject.toml CLAUDE.md skycam_utils/tests/test_alcor_archive.py
git commit -m "alcor: add the alcor_process_archive CLI entry point

Wires the archive driver up as a console script, documents its state
directory, pause switch, quiet-period rule, and JPEG prune in CLAUDE.md,
and records archive/ledger in the package dependency chain."
```

---

## Verification

After Task 7, confirm the whole feature end to end against real data — the unit tests all run against stubs, so this is the first time the driver meets `alcor_process_night`.

- [ ] **Full suite**

Run: `pytest -q`
Expected: PASS.

- [ ] **A real two-night run, subsampled**

```bash
alcor_process_archive /Volumes/Seagate_24TB/skycam \
    -o /tmp/archive-smoke --start 2025-01-01 --end 2025-01-02 \
    --min-age 0 --max-frames 20 --workers 4
```

Expected: both nights recorded `done`; `/tmp/archive-smoke/2025-01-01/sky_brightness.csv` and `2025-01-01_phot.csv` exist and are non-empty.

- [ ] **Resume does not redo work**

Run the same command again.
Expected: the log reports no nights processed; `alcor_process_archive --status -o /tmp/archive-smoke` shows `done 2`.

- [ ] **Pause works on a detached run**

```bash
alcor_process_archive /Volumes/Seagate_24TB/skycam -o /tmp/archive-smoke2 \
    --start 2025-01-01 --end 2025-01-05 --min-age 0 --max-frames 20 &
sleep 5 && touch /tmp/archive-smoke2/.archive_state/PAUSE
```

Expected: the log prints `paused — waiting on ...` after the current night finishes, and nothing further is processed until the file is removed.

- [ ] **Ctrl-C stops cleanly at the night boundary**

`install_signals=True` is set only by the CLI, so this is the one path the unit
tests cannot reach.

```bash
alcor_process_archive /Volumes/Seagate_24TB/skycam -o /tmp/archive-smoke3 \
    --start 2025-01-01 --end 2025-01-05 --min-age 0 --max-frames 20
```

Press Ctrl-C once partway through.
Expected: it logs `interrupt received — finishing the current night, then stopping`,
completes the night in flight, records it `done`, and exits 0. A second Ctrl-C aborts
at once. `alcor_process_archive --status -o /tmp/archive-smoke3` then shows the
remaining nights still `pending`, and rerunning the command resumes them.

- [ ] **Dry-run prune reports without deleting**

```bash
alcor_process_archive /Volumes/Seagate_24TB/skycam -o /tmp/archive-smoke \
    --start 2025-01-01 --end 2025-01-01 --min-age 0 --prune-dry-run
```

Expected: reports a jpeg count and GB figure; `ls /Volumes/Seagate_24TB/skycam/2025-01-01/*.jpg | wc -l` is unchanged.

- [ ] **Clean up the smoke-test output**

```bash
rm -rf /tmp/archive-smoke /tmp/archive-smoke2 /tmp/archive-smoke3
```
