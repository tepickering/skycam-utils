# Alcor WCS Calibration Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single set of baked alcor WCS constants with a time-indexed library of calibration epochs, resolving the epoch nearest in time to each image automatically.

**Architecture:** An in-code list `ALCOR_CALIBRATIONS` of `{epoch, xshift, yshift, rotation, radial_coeffs}` dicts is the single source of truth. `alcor_calibration(time)` returns the nearest epoch (ties → more recent; `time=None` → most recent). `load_alcor_fits` resolves per-frame from the image's filename time (DATE-header fallback) when its calibration kwargs are left `None`. `fit_alcor_wcs` fits from a neutral (uncalibrated) frame seeded with the nearest epoch, so it returns absolute constants and prints a ready-to-paste epoch dict stamped with the night's date.

**Tech Stack:** Python, astropy (`Time`, `fits`), numpy, scipy, pytest.

---

## Background the implementer must know

- File under change: `skycam_utils/alcor.py`. Tests: `skycam_utils/tests/test_alcor_wcs.py`.
- Run a single test with: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::<test_name> -v`
- Run the suite with: `python -m pytest skycam_utils/tests/ -q` (currently 35 passing).
- Pyright will emit `float | tuple` argument warnings on `_predict_pixels(**dict)` calls in the test file. These are **pre-existing noise** unrelated to this work — ignore them.
- Current constants (`skycam_utils/alcor.py:38-43`):
  ```python
  ALCOR_RADIUS = 680
  ALCOR_HORIZON_RADIUS = 662
  ALCOR_ROTATION = 0.3886
  ALCOR_XSHIFT = -4.570
  ALCOR_YSHIFT = 4.413
  ALCOR_RADIAL_COEFFS = (1.0, 0.01383, 0.0)
  ```
- `_predict_pixels` (line 65) and `build_alcor_wcs` (line 516) use `ALCOR_XSHIFT`/`ALCOR_YSHIFT`/`ALCOR_RADIAL_COEFFS`/`ALCOR_RADIUS`/`ALCOR_HORIZON_RADIUS` as **default argument values**. These names must continue to exist after the refactor.
- `_filename_ut_datetime(filename)` already exists: it returns a UT `datetime` parsed from a `YYYY_MM_DD__HH_MM_SS` name, or `None`. `_read_frame_date(filename)` returns the `DATE` header string.
- **Key model fact:** in a frame loaded with `rotation=0, xshift=0, yshift=0` (no recentering/rotation applied — a "neutral" frame), `_predict_pixels(alt, az, xshift=Xabs, yshift=Yabs, rotation=Rabs, radial_coeffs=C)` maps a catalog star to its pixel using the **absolute** geometry. `load_alcor_fits` later uses those same absolute values to recenter/rotate the frame for display. So fitting on a neutral frame yields absolute constants directly.

---

## File Structure

All changes are within two existing files; no new files.

- `skycam_utils/alcor.py`
  - New: `ALCOR_CALIBRATIONS` table, `_calibration_epochs()`, `alcor_calibration()`, `_alcor_frame_calibration()`.
  - Changed: derive `ALCOR_ROTATION`/`ALCOR_XSHIFT`/`ALCOR_YSHIFT`/`ALCOR_RADIAL_COEFFS` from the latest epoch; `load_alcor_fits` `None`-defaults + resolution; `_detect_alcor_frame` neutral load; `fit_alcor_wcs` warm-start + epoch stamp; `fit_alcor_wcs_cli` epoch-dict output; `plot_alcor_fits` + three CLI `--rotation` defaults.
- `skycam_utils/tests/test_alcor_wcs.py` — new selection/resolution/fallback/output tests; minor updates to synthetic-fit tests (timestamped fake filenames).
- `CLAUDE.md` — document the library and per-frame resolution.

---

### Task 1: Calibration table and `alcor_calibration()`

**Files:**
- Modify: `skycam_utils/alcor.py:38-43` (constants block)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing test**

Add to `skycam_utils/tests/test_alcor_wcs.py` (import `alcor_calibration` in the existing `from skycam_utils.alcor import (...)` block, and `from astropy.time import Time` is already imported):

```python
def test_alcor_calibration_nearest_in_time(monkeypatch):
    import skycam_utils.alcor as alcor_mod
    table = [
        {"epoch": "2024-09-04", "xshift": -4.5, "yshift": 4.4,
         "rotation": 0.39, "radial_coeffs": (1.0, 0.014, 0.0)},
        {"epoch": "2026-05-19", "xshift": -12.0, "yshift": 9.9,
         "rotation": 0.31, "radial_coeffs": (1.0, 0.084, 0.0)},
    ]
    monkeypatch.setattr(alcor_mod, "ALCOR_CALIBRATIONS", table)

    # well inside the 2024 side
    c = alcor_mod.alcor_calibration(Time("2024-10-01T00:00:00"))
    assert c["epoch"] == "2024-09-04"
    # well inside the 2026 side
    c = alcor_mod.alcor_calibration(Time("2026-04-01T00:00:00"))
    assert c["epoch"] == "2026-05-19"
    # before the first epoch -> first
    c = alcor_mod.alcor_calibration(Time("2020-01-01T00:00:00"))
    assert c["epoch"] == "2024-09-04"
    # after the last epoch -> last
    c = alcor_mod.alcor_calibration(Time("2030-01-01T00:00:00"))
    assert c["epoch"] == "2026-05-19"
    # time=None -> most recent
    assert alcor_mod.alcor_calibration()["epoch"] == "2026-05-19"

    # exact midpoint -> tie breaks to the more recent epoch
    midpoint = Time((Time("2024-09-04").jd + Time("2026-05-19").jd) / 2.0, format="jd")
    assert alcor_mod.alcor_calibration(midpoint)["epoch"] == "2026-05-19"

    # returns a copy, not the stored dict
    c = alcor_mod.alcor_calibration(Time("2024-10-01T00:00:00"))
    c["xshift"] = 999.0
    assert table[0]["xshift"] == -4.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_alcor_calibration_nearest_in_time -v`
Expected: FAIL — `ImportError: cannot import name 'alcor_calibration'` (or AttributeError).

- [ ] **Step 3: Write minimal implementation**

Replace the constants block at `skycam_utils/alcor.py:38-43` with:

```python
ALCOR_RADIUS = 680
ALCOR_HORIZON_RADIUS = 662

# Time-indexed lens calibrations. Each epoch holds only the fitted geometry
# (xshift, yshift, rotation, radial_coeffs); the trim/scale geometry
# (ALCOR_RADIUS / ALCOR_HORIZON_RADIUS / xcen / ycen) is fixed. The camera
# geometry drifts over time (mount/focus), so the epoch nearest in time to an
# image is used (see alcor_calibration). Add a new epoch by pasting the dict
# that fit_alcor_wcs prints. `epoch` is the calibration night at day precision.
ALCOR_CALIBRATIONS = [
    {"epoch": "2024-09-04", "xshift": -4.570, "yshift": 4.413,
     "rotation": 0.3886, "radial_coeffs": (1.0, 0.01383, 0.0)},
]


def _calibration_epochs():
    """Return [(Time, calibration_dict), ...] for the configured epochs."""
    return [(Time(c["epoch"], scale="utc"), c) for c in ALCOR_CALIBRATIONS]


def alcor_calibration(time=None):
    """
    Return the calibration dict whose epoch is nearest in time to ``time``.

    ``time`` is an astropy ``Time``. An exact tie resolves to the more recent
    epoch. ``time=None`` returns the most recent epoch (the default for
    time-agnostic calls). The returned dict is a copy and may be mutated freely.
    """
    epochs = _calibration_epochs()
    if time is None:
        return dict(max(epochs, key=lambda e: e[0].jd)[1])
    jds = np.array([e[0].jd for e in epochs])
    dt = np.abs(jds - Time(time).jd)
    # primary: smallest |dt|; tie-break: largest jd (more recent)
    order = np.lexsort((-jds, dt))
    return dict(epochs[order[0]][1])


# Module-level defaults track the most-recent epoch so existing default-argument
# references (in _predict_pixels, build_alcor_wcs, etc.) keep working unchanged.
_LATEST_CALIBRATION = alcor_calibration()
ALCOR_ROTATION = _LATEST_CALIBRATION["rotation"]
ALCOR_XSHIFT = _LATEST_CALIBRATION["xshift"]
ALCOR_YSHIFT = _LATEST_CALIBRATION["yshift"]
ALCOR_RADIAL_COEFFS = _LATEST_CALIBRATION["radial_coeffs"]
```

Note: `np` and `Time` are already imported at the top of `alcor.py`. This block sits above `_predict_pixels`, so the derived constants exist before they are used as defaults.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_alcor_calibration_nearest_in_time -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `python -m pytest skycam_utils/tests/ -q`
Expected: all pass (the derived constants equal the previous literals, so existing tests are unaffected).

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add time-indexed alcor calibration table and alcor_calibration()"
```

---

### Task 2: Per-frame calibration resolver

**Files:**
- Modify: `skycam_utils/alcor.py` (add `_alcor_frame_calibration` near `_filename_ut_datetime`)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_alcor_frame_calibration_uses_filename_then_header(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.io import fits

    table = [
        {"epoch": "2024-09-04", "xshift": -4.5, "yshift": 4.4,
         "rotation": 0.39, "radial_coeffs": (1.0, 0.014, 0.0)},
        {"epoch": "2026-05-19", "xshift": -12.0, "yshift": 9.9,
         "rotation": 0.31, "radial_coeffs": (1.0, 0.084, 0.0)},
    ]
    monkeypatch.setattr(alcor_mod, "ALCOR_CALIBRATIONS", table)

    # filename parses -> no file access needed, resolves by filename time
    assert alcor_mod._alcor_frame_calibration(
        "2026_05_19__03_37_18.fits.bz2")["epoch"] == "2026-05-19"
    assert alcor_mod._alcor_frame_calibration(
        "2024_09_04__22_00_00.fits.bz2")["epoch"] == "2024-09-04"

    # filename does not parse -> fall back to the DATE header
    path = tmp_path / "master.fits"
    hdu = fits.PrimaryHDU(data=np.zeros((3, 4, 4), dtype=np.int16))
    hdu.header["DATE"] = "2026-05-19T10:37:18"
    hdu.writeto(path)
    assert alcor_mod._alcor_frame_calibration(path)["epoch"] == "2026-05-19"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_alcor_frame_calibration_uses_filename_then_header -v`
Expected: FAIL — `AttributeError: ... has no attribute '_alcor_frame_calibration'`.

- [ ] **Step 3: Write minimal implementation**

Add immediately after the `_read_frame_date` function in `skycam_utils/alcor.py`:

```python
def _alcor_frame_calibration(filename):
    """
    Resolve the calibration epoch nearest in time to a frame.

    The frame time is parsed from its YYYY_MM_DD__HH_MM_SS filename first (no
    file access); if the name does not parse, the DATE header is read instead.
    Returns the calibration dict from :func:`alcor_calibration`.
    """
    dt = _filename_ut_datetime(filename)
    if dt is None:
        time = Time(_read_frame_date(filename), format="isot", scale="utc")
    else:
        time = Time(dt)
    return alcor_calibration(time)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_alcor_frame_calibration_uses_filename_then_header -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add _alcor_frame_calibration: resolve epoch from frame time"
```

---

### Task 3: `load_alcor_fits` resolves calibration per frame

**Files:**
- Modify: `skycam_utils/alcor.py:671-757` (`load_alcor_fits`)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing test**

Uses the packaged `test.fits.bz2` already used by other tests.

```python
def test_load_alcor_fits_resolves_and_overrides(monkeypatch):
    import skycam_utils.alcor as alcor_mod
    test_fits = Path(__file__).with_name("test.fits.bz2")

    calls = {"n": 0}
    real = alcor_mod._alcor_frame_calibration

    def spy(filename):
        calls["n"] += 1
        return real(filename)

    monkeypatch.setattr(alcor_mod, "_alcor_frame_calibration", spy)

    # defaults -> resolver is consulted
    _, wcs = alcor_mod.load_alcor_fits(test_fits)
    assert calls["n"] == 1
    assert "ARC" in wcs.wcs.ctype[0]

    # all calibration kwargs explicit -> resolver NOT consulted, and an
    # idealized radial term yields a plain ARC WCS (no SIP).
    calls["n"] = 0
    _, wcs = alcor_mod.load_alcor_fits(
        test_fits, rotation=0.0, xshift=0.0, yshift=0.0,
        radial_coeffs=(1.0, 0.0, 0.0))
    assert calls["n"] == 0
    assert wcs.sip is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_load_alcor_fits_resolves_and_overrides -v`
Expected: FAIL — with the current signature the defaults are the fixed constants (not `None`), so `_alcor_frame_calibration` is never called and `calls["n"] == 0` on the first assertion.

- [ ] **Step 3: Write minimal implementation**

Change the signature at `skycam_utils/alcor.py:671-674` to:

```python
def load_alcor_fits(filename, rotation=None, xcen=696, ycen=698,
                    radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                    xshift=None, yshift=None,
                    radial_coeffs=None, sip_degree=5):
```

Insert resolution at the very start of the function body (immediately after the docstring, before `with fits.open(filename) as hdul:` at line 735):

```python
    if rotation is None or xshift is None or yshift is None or radial_coeffs is None:
        cal = _alcor_frame_calibration(filename)
        if rotation is None:
            rotation = cal["rotation"]
        if xshift is None:
            xshift = cal["xshift"]
        if yshift is None:
            yshift = cal["yshift"]
        if radial_coeffs is None:
            radial_coeffs = cal["radial_coeffs"]
```

Update the docstring lines for `rotation`, `xshift`, `yshift`, `radial_coeffs` to read (replace the existing `(default=ALCOR_*)` descriptions):

```
    rotation : float or None (default=None)
        Camera rotation w.r.t. true north, in degrees. When None, resolved from
        the calibration epoch nearest the frame's time (see alcor_calibration).
    ...
    xshift : float or None (default=None)
        Zenith offset from the array center in x (pixels). When None, resolved
        from the nearest calibration epoch. Applied via scipy.ndimage.shift.
    yshift : float or None (default=None)
        Zenith offset from the array center in y (pixels). When None, resolved
        from the nearest calibration epoch.
    radial_coeffs : tuple of float or None (default=None)
        The (k1, k3, k5) plate-solution coefficients. When None, resolved from
        the nearest calibration epoch. The idealized mapping is (1.0, 0.0, 0.0).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_load_alcor_fits_resolves_and_overrides -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest skycam_utils/tests/ -q`
Expected: all pass. (Existing `test_alcor.py` calls `load_alcor_fits(test_fits)` with no kwargs; it now resolves to the 2024 epoch — identical values to before — so SIP is still present and round-trips hold.)

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "load_alcor_fits resolves calibration from frame time when unset"
```

---

### Task 4: `fit_alcor_wcs` fits absolute constants and stamps the epoch

**Files:**
- Modify: `skycam_utils/alcor.py` — `_detect_alcor_frame` (~line 205), `fit_alcor_wcs` init (~line 258) and return (~line 352)
- Test: `skycam_utils/tests/test_alcor_wcs.py` (update three synthetic tests)

The synthetic fit tests fake `load_alcor_fits` (returns zeros, ignores transform args) and fake `detect` (returns absolute positions from `**true`). The fit therefore recovers the absolute params, and the existing assertions on `result["xshift"]` etc. still hold. Three adjustments are needed: (a) the fake frame filenames must be timestamped so `fit_alcor_wcs` can compute the night time; (b) `alcor_calibration` must be monkeypatched to a zero base, because the real warm-start would seed the bootstrap from the 2024 epoch (xshift≈-4.6) while the synthetic truth (xshift=6) is far away, making the first-step match borderline — real data is always near its nearest epoch, but the synthetic truth is deliberately not; (c) assert the new `result["epoch"]`.

- [ ] **Step 1: Update the synthetic tests (failing on epoch)**

In `test_fit_alcor_wcs_aggregates_synthetic_frames`, rename the fake files to timestamped names, monkeypatch the base to zero, and assert the epoch. Change:

```python
    files = [tmp_path / "f0.fits", tmp_path / "f1.fits"]
```
to:
```python
    files = [tmp_path / "2024_09_05__00_00_00.fits",
             tmp_path / "2024_09_05__00_10_00.fits"]
```
Add this monkeypatch alongside the other `monkeypatch.setattr` calls (so the fit starts from a neutral seed, isolating the test from whatever epochs the real table holds):
```python
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xshift": 0.0,
                                           "yshift": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0)})
```
and add at the end of the test:
```python
    assert result["epoch"] == "2024-09-05"
```

In `test_fit_alcor_wcs_parallel_matches_serial`, change:
```python
    files = [tmp_path / "f0.fits", tmp_path / "f1.fits"]
```
to:
```python
    files = [tmp_path / "2024_09_05__00_00_00.fits",
             tmp_path / "2024_09_05__00_10_00.fits"]
```
The `index_by_name` / `state` logic already keys off `Path(path).name`, so it keeps working with the new names. Add the same `alcor_calibration` monkeypatch as above alongside the other `monkeypatch.setattr` calls, and add at the end:
```python
    assert result["epoch"] == "2024-09-05"
```

The night time is parsed from the timestamped filenames via `_filename_ut_datetime` (no file read), so the stub-byte files are never opened. `result["epoch"]` is derived from the night time (the filenames), independent of the monkeypatched base. Note `_filename_ut_datetime` adds the +7h MST→UT offset, so local `2024_09_05__00:10` → UT `2024-09-05T07:10`, whose date is still `2024-09-05`.

(`test_fit_alcor_wcs_log_reports_dispositions` keeps its `f_day`/`f_empty`/`f_good` stub-byte names and is left unchanged. Its names do not parse as timestamps, and the stub bytes are not valid FITS, so the night-time loop's `_read_frame_date` fallback would raise — which is why Step 3 wraps that fallback in `try/except` and skips unreadable frames. With both files skipped, `night_time` is `None`, the epoch falls back to the latest table entry, and the test's Sun/no-star/used message assertions are unaffected. No change to that test.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_fit_alcor_wcs_aggregates_synthetic_frames -v`
Expected: FAIL — `KeyError: 'epoch'`.

- [ ] **Step 3: Implement the neutral load, warm start, and epoch stamp**

In `_detect_alcor_frame` (~line 215), change the load call so the fit sees an uncalibrated frame:

```python
    im, _ = load_alcor_fits(filename, rotation=0.0, xshift=0.0, yshift=0.0,
                            radial_coeffs=(1.0, 0.0, 0.0))
```

In `fit_alcor_wcs`, after `dark` is finalized (just before building `init`, currently around line 257), compute the night time and seed the fit from the nearest epoch:

```python
    # Representative night time (median dark-frame time), used both to seed the
    # fit from the nearest existing calibration and to stamp the new epoch.
    # Prefer the filename timestamp; fall back to the DATE header, and skip any
    # frame that can be timed by neither (corrupt/oddly-named) rather than fail.
    night_dts = []
    for f in dark:
        d = _filename_ut_datetime(f)
        if d is None:
            try:
                d = Time(_read_frame_date(f), format="isot", scale="utc").to_datetime()
            except Exception:
                continue
        night_dts.append(d)
    night_time = Time(sorted(night_dts)[len(night_dts) // 2]) if night_dts else None
    base = alcor_calibration(night_time)
    epoch = (night_time.datetime.date().isoformat()
             if night_time is not None else base["epoch"])
```

Replace the existing cold init line:

```python
    init = dict(xshift=0.0, yshift=0.0, rotation=0.0,
                radial_coeffs=(1.0, 0.0, 0.0))
```
with the warm start:
```python
    init = dict(xshift=base["xshift"], yshift=base["yshift"],
                rotation=base["rotation"], radial_coeffs=base["radial_coeffs"])
```

Add `"epoch": epoch,` to the returned dict (currently lines 352-357):

```python
    return {
        **params,
        "epoch": epoch,
        "n_matched": int(good.sum()),
        "residual_rms": rms,
        "alt": alt[good], "az": az[good], "x": x[good], "y": y[good],
    }
```

Update the `fit_alcor_wcs` docstring sentence that says it returns "(xshift, yshift, rotation, radial_coeffs)" to note these are now **absolute** constants for the night plus an `epoch` date string, and that the fit runs on a neutral (uncalibrated) frame seeded from the nearest existing epoch.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_fit_alcor_wcs_aggregates_synthetic_frames skycam_utils/tests/test_alcor_wcs.py::test_fit_alcor_wcs_parallel_matches_serial -v`
Expected: PASS (absolute params recovered ≈ `true`, `result["epoch"] == "2024-09-05"`).

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest skycam_utils/tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "fit_alcor_wcs: fit absolute constants on neutral frame, stamp epoch"
```

---

### Task 5: `fit_alcor_wcs_cli` prints a ready-to-paste epoch dict

**Files:**
- Modify: `skycam_utils/alcor.py:1427-1435` (CLI output block)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing test**

This test exercises the printed format via a small helper so it needs no real data. Add the helper and test:

```python
def test_format_calibration_entry_is_parseable():
    import ast
    from skycam_utils.alcor import _format_calibration_entry

    line = _format_calibration_entry(dict(
        epoch="2026-05-19", xshift=-12.0, yshift=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0)))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["epoch"] == "2026-05-19"
    assert parsed["radial_coeffs"] == (1.0, 0.084, 0.0)
    assert abs(parsed["xshift"] + 12.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_format_calibration_entry_is_parseable -v`
Expected: FAIL — `ImportError: cannot import name '_format_calibration_entry'`.

- [ ] **Step 3: Implement the formatter and rewire the CLI**

Add this module-level helper to `skycam_utils/alcor.py` (place it just above `fit_alcor_wcs_cli`):

```python
def _format_calibration_entry(result):
    """Format a calibration result as a paste-ready ALCOR_CALIBRATIONS entry."""
    rc = tuple(float(c) for c in result["radial_coeffs"])
    return (f'    {{"epoch": "{result["epoch"]}", '
            f'"xshift": {result["xshift"]:.3f}, '
            f'"yshift": {result["yshift"]:.3f}, '
            f'"rotation": {result["rotation"]:.4f}, '
            f'"radial_coeffs": {rc!r}}},')
```

Replace the CLI output block at `skycam_utils/alcor.py:1427-1435` (the four `ALCOR_* = ...` prints and their preceding comment) with:

```python
    print(f"# matched stars: {result['n_matched']}")
    print(f"# residual RMS (pix): {result['residual_rms']:.3f}")
    print("# add this entry to ALCOR_CALIBRATIONS in alcor.py:")
    print(_format_calibration_entry(result))
```

(The `--residual-plot` block immediately below stays unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_format_calibration_entry_is_parseable -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "fit_alcor_wcs CLI prints a paste-ready calibration epoch dict"
```

---

### Task 6: Downstream functions and CLIs default to auto-resolution

**Files:**
- Modify: `skycam_utils/alcor.py` — `plot_alcor_fits` signature (line 1101) and its `load_alcor_fits` call (lines 1136-1143); the three CLI `--rotation` arguments (the `parser.add_argument("--rotation", ... default=ALCOR_ROTATION ...)` lines at ~1192, ~1246, ~1356)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing test**

```python
def test_plot_alcor_fits_defaults_rotation_to_none():
    import inspect
    from skycam_utils.alcor import plot_alcor_fits
    assert inspect.signature(plot_alcor_fits).parameters["rotation"].default is None


def test_cli_rotation_flags_default_to_none(monkeypatch):
    """Each CLI's --rotation default is None so load_alcor_fits auto-resolves.

    The parsers are constructed inside the *_cli functions, so capture the
    parsed namespace by patching parse_args and stop before the body runs.
    """
    import argparse
    import skycam_utils.alcor as alcor_mod

    captured = {}
    orig_parse = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        ns = orig_parse(self, *a, **k)
        captured["rotation"] = getattr(ns, "rotation", "MISSING")
        raise SystemExit  # stop before the CLI does real work
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)

    for cli, argv in [
        (alcor_mod.plot_alcor_fits_cli, ["prog", "in.fits"]),
        (alcor_mod.alcor_proc_fits_cli, ["prog", "in.fits"]),
        (alcor_mod.alcor_keogram_cli, ["prog", "in_dir"]),
    ]:
        captured.clear()
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(SystemExit):
            cli()
        assert captured["rotation"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_plot_alcor_fits_defaults_rotation_to_none skycam_utils/tests/test_alcor_wcs.py::test_cli_rotation_flags_default_to_none -v`
Expected: FAIL — defaults are currently `ALCOR_ROTATION` (a float), not `None`.

- [ ] **Step 3: Implement the default changes**

Change `plot_alcor_fits` signature at line 1101 from `rotation=ALCOR_ROTATION` to `rotation=None`:

```python
def plot_alcor_fits(filename, outimage=None, outfig=None, rotation=None, xcen=696, ycen=698, radius=680,
                    horizon_radius=662, powerstretch=0.75, contrast=0.35, gscale=0.7, bscale=1.7, figsize=12):
```

Its body already passes `rotation=rotation` to `load_alcor_fits` (lines 1136-1143); leave that call as-is — `None` now triggers resolution. Update the `rotation` docstring line in `plot_alcor_fits` to:

```
    rotation : float or None (default=None)
        Camera rotation w.r.t. true north (deg). When None, resolved from the
        calibration epoch nearest the frame date.
```

For each of the three CLI parsers (`plot_alcor_fits_cli`, `alcor_proc_fits_cli`, `alcor_keogram_cli`), change the `--rotation` argument from:

```python
    parser.add_argument("--rotation", type=float, default=ALCOR_ROTATION, help="Camera rotation w.r.t. true north (deg).")
```
to:
```python
    parser.add_argument("--rotation", type=float, default=None,
                        help="Camera rotation w.r.t. true north (deg); "
                             "default resolves the calibration epoch nearest the frame date.")
```

Leave each CLI's `rotation=args.rotation` pass-through unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest skycam_utils/tests/test_alcor_wcs.py::test_plot_alcor_fits_defaults_rotation_to_none skycam_utils/tests/test_alcor_wcs.py::test_cli_rotation_flags_default_to_none -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest skycam_utils/tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "downstream alcor CLIs default --rotation to auto-resolution"
```

---

### Task 7: Document the calibration library

**Files:**
- Modify: `CLAUDE.md` (the alcor paragraph and the `fit_alcor_wcs` CLI note)

- [ ] **Step 1: Update the alcor description**

In `CLAUDE.md`, find the alcor bullet describing `load_alcor_fits`. Append a sentence:

```
The fitted geometry (center shift, rotation, radial k3) lives in a time-indexed
`ALCOR_CALIBRATIONS` table; `load_alcor_fits` resolves the epoch nearest a
frame's time (filename timestamp, DATE-header fallback) when its calibration
kwargs are left unset, so frames from different eras (e.g. 2024 vs 2026) each
get the right geometry. Explicit kwargs override resolution.
```

- [ ] **Step 2: Update the `fit_alcor_wcs` CLI note**

Find the `fit_alcor_wcs <night-dir> ...` block in `CLAUDE.md` and change the description of its output to:

```
#   Prints a ready-to-paste ALCOR_CALIBRATIONS epoch dict (absolute constants
#   stamped with the night date) to add to alcor.py and commit.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "document the alcor time-indexed calibration library"
```

---

## Final verification

- [ ] Run the full suite once more: `python -m pytest skycam_utils/tests/ -q` — expect all green.
- [ ] Sanity-check the CLI end-to-end locally against the one 2026 frame if available:
  `python -c "from skycam_utils.alcor import _format_calibration_entry; print(_format_calibration_entry(dict(epoch='2026-05-19', xshift=-12.0, yshift=9.9, rotation=0.3, radial_coeffs=(1.0,0.08,0.0))))"`
  — confirm the printed line is valid Python that drops straight into `ALCOR_CALIBRATIONS`.

## Notes for the operator (post-merge, not a code task)

After merging, re-run `fit_alcor_wcs` on the 2026-05 night (with the k3-only robust fitter already in place), then paste the printed epoch dict into `ALCOR_CALIBRATIONS` and commit. Until that entry is added, 2026 frames resolve to the 2024 epoch (nearest available), which is the current behavior.
