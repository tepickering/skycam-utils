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
