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
