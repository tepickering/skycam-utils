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
