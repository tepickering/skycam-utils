# Alcor Star-Matcher Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-frame-refit nearest-neighbor matcher in `fit_alcor_wcs` with a seeded, kd-tree, pattern-verified assignment so the pooled residual reflects real lens geometry error rather than crowded-horizon mispairings.

**Architecture:** Cap detections to the brightest few hundred per frame; assign catalog↔detection against a *fixed* geometry using a cKDTree candidate search, connected-component cluster resolution (1:1 candidate = the mutual-nearest case; contested clusters resolved by relative-brightness rank pairing), and a local-asterism pattern check that rejects mispairs. `fit_alcor_wcs` drives a tightening-tolerance outer loop (no per-frame transform refit) and reports the matched fraction.

**Tech Stack:** Python, NumPy, SciPy (`scipy.spatial.cKDTree`, `scipy.optimize.least_squares`), Astropy tables, photutils `DAOStarFinder`, pytest.

---

## Background for the implementer

The component lives entirely in `skycam_utils/alcor.py`. Read these existing pieces first — the new code mirrors their conventions:

- `_predict_pixels(alt, az, xshift=, yshift=, rotation=, radial_coeffs=, radius=, horizon_radius=)` — forward lens model, catalog Alt/Az (deg) → frame pixels `(x, y)`. Returns arrays.
- `_fit_params(alt, az, obs_x, obs_y, init_params, radius=, horizon_radius=)` — robust k3-only `least_squares` fit, returns a params dict `{"xshift", "yshift", "rotation", "radial_coeffs": (1.0, k3, 0.0)}`.
- `match_alcor_stars(cat, detections, init_params, z_steps=, tolerance=, ...)` — the matcher being **replaced**. It refits geometry *per frame*; the new matcher must not.
- `detect_alcor_stars(im, fwhm=3.0, threshold_sigma=5.0)` — returns an Astropy `Table` with `xcentroid`, `ycentroid`, `flux` columns.
- `alcor_reference_altaz(time, vmag_limit=3.0, min_alt=5.0, ...)` — catalog `Table` with `Alt`, `Az`, `Vmag` (and `HD`, etc.).
- `fit_alcor_wcs(...)` — orchestrator. `_detect_alcor_frame(task)` is its per-frame worker; the task tuple is `(index, filename, vmag_limit, min_alt, fwhm, threshold_sigma)`.
- `ALCOR_RADIUS = 680`, `ALCOR_HORIZON_RADIUS = 662` are module constants.
- The output-table contract of the matcher is `hstack([Table(cat[cat_rows]), Table(det[det_rows])])` — catalog columns and detection columns side by side.

Convention notes:
- Catalog and detection tables never share column names, so `hstack` is safe.
- Brightness ordering: catalog **brighter ⇒ smaller `Vmag`**; detection **brighter ⇒ larger `flux`**.
- Keep the UT-date epoch convention untouched (see memory `alcor-epoch-ut-date-convention`).
- Pyright `float|tuple` warnings on `_predict_pixels` and `SkyCoord/None` warnings are pre-existing noise — ignore them.

Run the whole suite with `pytest skycam_utils/tests/test_alcor_wcs.py -q` from the repo root.

---

## Files changed

- **Modify** `skycam_utils/alcor.py`
  - Add `from scipy.spatial import cKDTree` to the imports.
  - `detect_alcor_stars` — add `max_detections=200`, keep the brightest N by flux.
  - Add `assign_alcor_matches(...)` — the new matcher.
  - Remove `match_alcor_stars`.
  - `_detect_alcor_frame` — plumb `max_detections` through the task tuple.
  - `fit_alcor_wcs` — replace the matcher + two-pass refit with a tightening-tolerance outer loop, drop `z_steps`, change `vmag_limit` default 3.0 → 4.0 and `tolerance` default 12.0 → 3.0, add `tolerance_start`/`match_rounds`/`max_detections`, report `matched_fraction`.
  - `fit_alcor_wcs_cli` — add `--max-detections`, change `--vmag-limit` default to 4.0 and `--tolerance` default to 3.0, print the matched fraction.
- **Modify** `skycam_utils/tests/test_alcor_wcs.py`
  - Remove `match_alcor_stars` import + `test_match_alcor_stars_recovers_correspondences`.
  - Add detection-cap, `assign_alcor_matches`, and injected-mismatch integration tests.
  - Update the existing synthetic `fit_alcor_wcs` tests for the new keyword surface.
- **Modify** `CLAUDE.md` — document the hardened matcher and the new flags.

---

### Task 1: Detection cap (`max_detections`) in `detect_alcor_stars`

**Files:**
- Modify: `skycam_utils/alcor.py` (imports near line 11; `detect_alcor_stars` near line 711)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing test**

Add to `skycam_utils/tests/test_alcor_wcs.py` (after `test_detect_alcor_stars_finds_synthetic_sources`, near line 195):

```python
def test_detect_alcor_stars_caps_to_brightest():
    rng = np.random.default_rng(11)
    im = np.zeros((200, 200, 3), dtype=float)
    im += rng.normal(0.0, 1.0, im.shape)
    yy, xx = np.mgrid[0:200, 0:200]
    # 8 stars of decreasing brightness at distinct positions
    centers = [(20, 20), (60, 40), (100, 30), (140, 60),
               (40, 120), (90, 150), (150, 130), (170, 170)]
    amps = np.linspace(2000.0, 400.0, len(centers))
    for (cx, cy), amp in zip(centers, amps):
        im += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0**2))[:, :, None]

    capped = detect_alcor_stars(im, fwhm=2.5, threshold_sigma=5.0, max_detections=3)
    assert len(capped) == 3
    # The three kept must be the three brightest detected.
    full = detect_alcor_stars(im, fwhm=2.5, threshold_sigma=5.0, max_detections=None)
    top3 = np.sort(np.asarray(full["flux"]))[::-1][:3]
    np.testing.assert_allclose(np.sort(np.asarray(capped["flux"]))[::-1], top3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py::test_detect_alcor_stars_caps_to_brightest -q`
Expected: FAIL — `detect_alcor_stars() got an unexpected keyword argument 'max_detections'`.

- [ ] **Step 3: Add the cKDTree import**

In `skycam_utils/alcor.py`, alongside the other scipy imports (currently lines 11-13):

```python
from scipy.ndimage import rotate
from scipy.optimize import least_squares
from scipy.ndimage import shift as ndimage_shift
from scipy.spatial import cKDTree
```

- [ ] **Step 4: Add the `max_detections` parameter and cap logic**

Change the `detect_alcor_stars` signature (line 711):

```python
def detect_alcor_stars(im, fwhm=3.0, threshold_sigma=5.0, max_detections=200):
```

Add to the docstring's Parameters section (after the `threshold_sigma` entry):

```
    max_detections : int or None (default=200)
        Keep only the brightest ``max_detections`` sources by ``flux``. ``None``
        keeps all. Bounding the list to the brightest few hundred keeps matching
        on the well-detected stars regardless of per-frame noise/transparency.
```

At the end of the function, immediately before `return out` (currently line 753), insert:

```python
    if max_detections is not None and len(out) > max_detections:
        order = np.argsort(np.asarray(out["flux"], dtype=float))[::-1]
        out = out[order[:max_detections]]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py::test_detect_alcor_stars_caps_to_brightest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add brightest-N detection cap to detect_alcor_stars"
```

---

### Task 2: `assign_alcor_matches` — seeded, kd-tree, pattern-verified matcher

**Files:**
- Modify: `skycam_utils/alcor.py` (add the new function directly above `match_alcor_stars`, near line 186)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

This task adds the new matcher. `match_alcor_stars` stays in place for now (removed in Task 3) so the suite keeps passing.

- [ ] **Step 1: Write the failing tests**

Add to `skycam_utils/tests/test_alcor_wcs.py`. First add the import (in the `from skycam_utils.alcor import (...)` block, keep alphabetical-ish grouping):

```python
    assign_alcor_matches,
```

Then add these tests after `test_match_alcor_stars_recovers_correspondences` (near line 246):

```python
def _clean_frame(true_params, n=12, seed=0):
    """A catalog + detection pair with detections exactly at the true pixels."""
    rng = np.random.default_rng(seed)
    alt = rng.uniform(15.0, 85.0, n)
    az = rng.uniform(0.0, 360.0, n)
    cat = _Table({"Alt": alt, "Az": az, "Vmag": rng.uniform(0.5, 3.5, n)})
    x, y = _predict_pixels(alt, az, **true_params)
    det = _Table({"xcentroid": np.asarray(x), "ycentroid": np.asarray(y),
                  "flux": rng.uniform(200.0, 2000.0, n)})
    return cat, det, np.asarray(x), np.asarray(y)


def test_assign_alcor_matches_recovers_clean_frame():
    true = dict(xshift=4.0, yshift=-3.0, rotation=0.6, radial_coeffs=(1.0, 0.03, 0.0))
    cat, det, tx, ty = _clean_frame(true, n=12, seed=1)
    matched = assign_alcor_matches(cat, det, params=true, tolerance=3.0)
    assert len(matched) == 12
    for row in matched:
        ex, ey = _predict_pixels(row["Alt"], row["Az"], **true)
        assert np.hypot(row["xcentroid"] - ex, row["ycentroid"] - ey) < 1e-6


def test_assign_alcor_matches_pattern_rejects_decoy():
    true = dict(xshift=4.0, yshift=-3.0, rotation=0.6, radial_coeffs=(1.0, 0.03, 0.0))
    cat, det, tx, ty = _clean_frame(true, n=12, seed=2)
    # Plant a decoy detection near catalog star 0's predicted pixel: close enough
    # to be the only candidate within tolerance (4.0 px), but displaced beyond
    # pattern_tol (3.0 px) so it breaks the local constellation and is rejected.
    decoy_x = tx[0] + 3.5
    decoy_y = ty[0]
    det_x = np.asarray(det["xcentroid"], dtype=float)
    det_y = np.asarray(det["ycentroid"], dtype=float)
    # remove star 0's true detection so the decoy is the only candidate for it
    keep = np.ones(len(det), dtype=bool)
    keep[0] = False
    det2 = _Table({"xcentroid": np.append(det_x[keep], decoy_x),
                   "ycentroid": np.append(det_y[keep], decoy_y),
                   "flux": np.append(np.asarray(det["flux"])[keep], 1500.0)})
    matched = assign_alcor_matches(cat, det2, params=true, tolerance=4.0)
    # The decoy must not be matched to catalog star 0; the genuine pairs survive.
    for row in matched:
        assert not (abs(row["xcentroid"] - decoy_x) < 1e-6
                    and abs(row["ycentroid"] - decoy_y) < 1e-6)
    assert len(matched) >= 9  # the clean stars still match


def test_assign_alcor_matches_brightness_tiebreak():
    # One detection contested by two catalog stars within tolerance.
    # Catalog A is bright (Vmag 1.0) and farther; B is faint (Vmag 4.0) and nearer.
    params = dict(xshift=0.0, yshift=0.0, rotation=0.0, radial_coeffs=(1.0, 0.0, 0.0))
    ax, ay = _predict_pixels(60.0, 10.0, **params)
    bx, by = _predict_pixels(60.5, 10.0, **params)  # close neighbor
    cat = _Table({"Alt": [60.0, 60.5], "Az": [10.0, 10.0], "Vmag": [1.0, 4.0]})
    # detection sits 1px from B's pixel, ~ a few px from A's pixel
    det = _Table({"xcentroid": [float(bx) + 1.0], "ycentroid": [float(by)],
                  "flux": [1000.0]})

    bright = assign_alcor_matches(cat, det, params=params, tolerance=20.0,
                                  brightness=True, min_corroborating=2)
    assert len(bright) == 1
    assert float(bright["Vmag"][0]) == 1.0          # bright catalog star A wins

    nearest = assign_alcor_matches(cat, det, params=params, tolerance=20.0,
                                   brightness=False, min_corroborating=2)
    assert len(nearest) == 1
    assert float(nearest["Vmag"][0]) == 4.0         # nearest catalog star B wins


def test_assign_alcor_matches_empty_inputs():
    params = dict(xshift=0.0, yshift=0.0, rotation=0.0, radial_coeffs=(1.0, 0.0, 0.0))
    cat = _Table({"Alt": [45.0], "Az": [10.0], "Vmag": [2.0]})
    empty_det = _Table({"xcentroid": [], "ycentroid": [], "flux": []})
    out = assign_alcor_matches(cat, empty_det, params=params, tolerance=3.0)
    assert len(out) == 0
    assert {"Alt", "Az", "xcentroid", "ycentroid"}.issubset(out.colnames)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k assign_alcor_matches -q`
Expected: FAIL — `cannot import name 'assign_alcor_matches'`.

- [ ] **Step 3: Implement `assign_alcor_matches`**

Insert this function in `skycam_utils/alcor.py` directly above `def match_alcor_stars(` (currently line 186):

```python
def assign_alcor_matches(cat, det, params, tolerance,
                         radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                         n_neighbors=5, min_corroborating=2, pattern_tol=3.0,
                         brightness=True):
    """
    Assign catalog stars to detected sources against a *fixed* geometry.

    Unlike :func:`match_alcor_stars`, this never refits the geometry internally;
    ``params`` (``xshift``, ``yshift``, ``rotation``, ``radial_coeffs``) are the
    seed used for the whole frame. The steps are:

    1. Predict each catalog star's pixel ``(px, py)`` with :func:`_predict_pixels`
       and build a `~scipy.spatial.cKDTree` of detections and of predicted
       catalog pixels.
    2. Form candidate edges (catalog i, detection j) with separation
       <= ``tolerance``, group them into connected components, and resolve each
       component. An isolated 1:1 candidate is the mutual-nearest case and is
       accepted directly. A contested cluster (several catalog stars and/or
       detections within tolerance) is resolved by **relative-brightness rank
       pairing**: detections sorted by ``flux`` descending are paired with catalog
       stars sorted by ``Vmag`` ascending, in order (within ``tolerance``). With
       ``brightness=False`` or missing ``flux``/``Vmag`` columns the cluster is
       resolved greedily by nearest separation instead. Because brightness is only
       consulted *within* a contested cluster of nearby stars, spatially or
       temporally patchy cloud extinction (which dims a local patch in common)
       never enters a global comparison.
    3. **Local-pattern (asterism) verification.** For each tentative pair i->j,
       look at catalog i's ``n_neighbors`` nearest catalog neighbors that also have
       a tentative pair. The pair is accepted iff at least ``min_corroborating`` of
       them corroborate the local constellation -- their detection offset matches
       the predicted offset to within ``pattern_tol``:
       ``||(det_jn - det_j) - (pred_in - pred_i)|| <= pattern_tol``. Pairs with
       fewer than ``min_corroborating`` paired neighbors are kept (too little local
       evidence to reject); crowded-region mispairs, which sit among well-matched
       neighbors yet break the constellation, are rejected.

    Returns an ``hstack`` of the accepted catalog and detection rows (same column
    contract as :func:`match_alcor_stars`); an empty table if nothing matches.
    """
    px, py = _predict_pixels(
        cat["Alt"], cat["Az"], xshift=params["xshift"], yshift=params["yshift"],
        rotation=params["rotation"], radial_coeffs=tuple(params["radial_coeffs"]),
        radius=radius, horizon_radius=horizon_radius,
    )
    px = np.atleast_1d(np.asarray(px, dtype=float))
    py = np.atleast_1d(np.asarray(py, dtype=float))
    det_x = np.asarray(det["xcentroid"], dtype=float)
    det_y = np.asarray(det["ycentroid"], dtype=float)

    n_cat = px.size
    n_det = det_x.size
    empty = hstack([Table(cat[[]]), Table(det[[]])])
    if n_cat == 0 or n_det == 0:
        return empty

    cat_xy = np.column_stack([px, py])
    det_xy = np.column_stack([det_x, det_y])
    det_tree = cKDTree(det_xy)
    cat_tree = cKDTree(cat_xy)

    has_bright = (brightness and "Vmag" in cat.colnames and "flux" in det.colnames)
    vmag = np.asarray(cat["Vmag"], dtype=float) if "Vmag" in cat.colnames else None
    flux = np.asarray(det["flux"], dtype=float) if "flux" in det.colnames else None

    # candidate detections within tolerance of each catalog star
    cat_cands = det_tree.query_ball_point(cat_xy, tolerance)

    # --- connected components over the bipartite candidate graph ---
    # nodes 0..n_cat-1 are catalog stars, n_cat..n_cat+n_det-1 are detections.
    parent = list(range(n_cat + n_det))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i, cands in enumerate(cat_cands):
        for j in cands:
            _union(i, n_cat + j)

    comp_cat = {}
    comp_det = {}
    for i in range(n_cat):
        if cat_cands[i]:
            comp_cat.setdefault(_find(i), []).append(i)
    for j in range(n_det):
        root = _find(n_cat + j)
        if root in comp_cat or any(_find(n_cat + j) == _find(i) for i in ()):  # placeholder
            pass
    # detections that belong to a catalog-bearing component
    for j in range(n_det):
        root = _find(n_cat + j)
        if root in comp_cat:
            comp_det.setdefault(root, []).append(j)

    # --- resolve each component into tentative (cat i, det j) pairs ---
    tentative = {}  # cat i -> det j
    for root, cis in comp_cat.items():
        djs = comp_det.get(root, [])
        if not djs:
            continue
        if len(cis) == 1 and len(djs) == 1:
            tentative[cis[0]] = djs[0]
            continue
        cis_arr = np.asarray(cis, dtype=int)
        djs_arr = np.asarray(djs, dtype=int)
        if has_bright:
            ci_order = cis_arr[np.argsort(vmag[cis_arr])]        # brightest catalog first
            dj_order = djs_arr[np.argsort(-flux[djs_arr])]       # brightest detection first
            for k in range(min(len(ci_order), len(dj_order))):
                i, j = int(ci_order[k]), int(dj_order[k])
                if np.hypot(det_x[j] - px[i], det_y[j] - py[i]) <= tolerance:
                    tentative[i] = j
        else:
            edges = []
            for i in cis_arr:
                for j in djs_arr:
                    d = np.hypot(det_x[j] - px[i], det_y[j] - py[i])
                    if d <= tolerance:
                        edges.append((d, int(i), int(j)))
            edges.sort()
            used_c, used_d = set(), set()
            for d, i, j in edges:
                if i in used_c or j in used_d:
                    continue
                tentative[i] = j
                used_c.add(i)
                used_d.add(j)

    if not tentative:
        return empty

    # --- local-pattern (asterism) verification ---
    k_query = min(n_neighbors + 1, n_cat)
    accepted_cat, accepted_det = [], []
    for i, j in tentative.items():
        _, idxs = cat_tree.query(cat_xy[i], k=k_query)
        neighbors = [int(n) for n in np.atleast_1d(idxs)
                     if int(n) != i and int(n) < n_cat]
        paired = [n for n in neighbors if n in tentative]
        if len(paired) < min_corroborating:
            accepted_cat.append(i)
            accepted_det.append(j)
            continue
        corro = 0
        for n in paired:
            jn = tentative[n]
            pred_off = cat_xy[n] - cat_xy[i]
            det_off = det_xy[jn] - det_xy[j]
            if np.hypot(*(det_off - pred_off)) <= pattern_tol:
                corro += 1
        if corro >= min_corroborating:
            accepted_cat.append(i)
            accepted_det.append(j)

    if not accepted_cat:
        return empty
    return hstack([Table(cat[accepted_cat]), Table(det[accepted_det])])
```

NOTE: delete the dead `placeholder` loop fragment above — it was only a scratch line. The clean body has exactly one `for j in range(n_det)` loop that fills `comp_det`. Make sure the committed code reads:

```python
    comp_cat = {}
    comp_det = {}
    for i in range(n_cat):
        if cat_cands[i]:
            comp_cat.setdefault(_find(i), []).append(i)
    for j in range(n_det):
        root = _find(n_cat + j)
        if root in comp_cat:
            comp_det.setdefault(root, []).append(j)
```

- [ ] **Step 4: Run the assign tests**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k assign_alcor_matches -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full suite to confirm nothing else broke**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -q`
Expected: all PASS (`match_alcor_stars` and its test still present).

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add assign_alcor_matches kd-tree pattern-verified matcher"
```

---

### Task 3: Rewire `fit_alcor_wcs` to the hardened matcher

**Files:**
- Modify: `skycam_utils/alcor.py` (`_detect_alcor_frame` near line 254; `fit_alcor_wcs` near line 276; remove `match_alcor_stars` near line 186)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing integration test**

Add to `skycam_utils/tests/test_alcor_wcs.py` after `test_fit_alcor_wcs_aggregates_synthetic_frames` (near line 345). It injects spurious mismatch detections per frame and asserts the geometry is still recovered at low RMS and the matched fraction is reported:

```python
def test_fit_alcor_wcs_survives_injected_mismatches(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xshift=6.0, yshift=-5.0, rotation=0.8, radial_coeffs=(1.0, 0.04, 0.0))
    rng = np.random.default_rng(5)

    frame_alt = [rng.uniform(15.0, 85.0, 25), rng.uniform(15.0, 85.0, 25)]
    frame_az = [rng.uniform(0.0, 360.0, 25), rng.uniform(0.0, 360.0, 25)]
    files = [tmp_path / "2024_09_05__00_00_00.fits",
             tmp_path / "2024_09_05__00_10_00.fits"]
    for f in files:
        f.write_bytes(b"stub")

    def fake_select_dark_frames(fs, **kw):
        return list(files)

    calls = {"i": 0}

    def fake_load_alcor_fits(path, **kw):
        return np.zeros((2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS, 3)), None

    def fake_reference_altaz(time, **kw):
        i = calls["i"]
        return Table({"Alt": frame_alt[i], "Az": frame_az[i],
                      "Vmag": rng.uniform(0.5, 3.5, 25), "HD": np.arange(25)})

    def fake_detect(im, **kw):
        i = calls["i"]
        calls["i"] += 1
        x, y = _predict_pixels(frame_alt[i], frame_az[i], **true)
        # 10 spurious detections scattered across the frame (mismatches)
        sx = rng.uniform(50, 2 * ALCOR_RADIUS - 50, 10)
        sy = rng.uniform(50, 2 * ALCOR_RADIUS - 50, 10)
        xx = np.append(np.asarray(x), sx)
        yy = np.append(np.asarray(y), sy)
        flux = np.append(rng.uniform(500, 2000, 25), rng.uniform(100, 400, 10))
        return Table({"xcentroid": xx, "ycentroid": yy, "flux": flux})

    def fake_frame_time(path):
        return Time("2024-09-05T07:00:00", format="isot", scale="utc")

    monkeypatch.setattr(alcor_mod, "select_dark_frames", fake_select_dark_frames)
    monkeypatch.setattr(alcor_mod, "load_alcor_fits", fake_load_alcor_fits)
    monkeypatch.setattr(alcor_mod, "alcor_reference_altaz", fake_reference_altaz)
    monkeypatch.setattr(alcor_mod, "detect_alcor_stars", fake_detect)
    monkeypatch.setattr(alcor_mod, "_frame_time", fake_frame_time)
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xshift": 0.0,
                                           "yshift": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0)})

    result = fit_alcor_wcs(tmp_path, pattern="*.fits")
    assert abs(result["xshift"] - 6.0) < 0.1
    assert abs(result["yshift"] + 5.0) < 0.1
    assert abs(result["rotation"] - 0.8) < 0.05
    np.testing.assert_allclose(result["radial_coeffs"], (1.0, 0.04, 0.0), atol=3e-3)
    assert result["residual_rms"] < 0.5
    assert 0.0 < result["matched_fraction"] <= 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py::test_fit_alcor_wcs_survives_injected_mismatches -q`
Expected: FAIL — `KeyError: 'matched_fraction'` (the key does not exist yet).

- [ ] **Step 3: Plumb `max_detections` through `_detect_alcor_frame`**

Replace the body of `_detect_alcor_frame` (lines 254-273). Change the task unpacking line and the `detect_alcor_stars` call:

```python
def _detect_alcor_frame(task):
    """
    Per-frame preprocessing for :func:`fit_alcor_wcs`, executed in worker
    processes. Loads a frame, builds its reference catalog and star detections,
    and returns ``(index, cat, det, reason)``. On success ``reason`` is ``None``;
    when the frame is unusable ``cat``/``det`` are ``None`` and ``reason`` is a
    short human-readable string (too few detections or catalog stars).
    """
    index, filename, vmag_limit, min_alt, fwhm, threshold_sigma, max_detections = task
    filename = Path(filename)
    time = _frame_time(filename)
    im, _ = load_alcor_fits(filename, rotation=0.0, xshift=0.0, yshift=0.0,
                            radial_coeffs=(1.0, 0.0, 0.0))
    cat = alcor_reference_altaz(time, vmag_limit=vmag_limit, min_alt=min_alt)
    det = detect_alcor_stars(im, fwhm=fwhm, threshold_sigma=threshold_sigma,
                             max_detections=max_detections)
    if len(det) < 3:
        return index, None, None, f"no stars detected ({len(det)} < 3)"
    if len(cat) < 3:
        return index, None, None, f"too few catalog stars ({len(cat)} < 3)"
    return index, cat, det, None
```

- [ ] **Step 4: Rewrite `fit_alcor_wcs`**

Replace the `fit_alcor_wcs` signature and body (lines 276-423). New signature, docstring, task tuple, tightening outer loop, and matched-fraction reporting:

```python
def fit_alcor_wcs(input_dir, pattern="*.fits.bz2", vmag_limit=4.0, sun_alt_max=-18.0,
                  min_alt=10.0, tolerance=3.0, tolerance_start=12.0, match_rounds=4,
                  fwhm=3.0, threshold_sigma=5.0, max_detections=200, max_frames=None,
                  workers=1, log=None):
    """
    Calibrate the alcor lens geometry by aggregating bright-star matches across
    all dark-sky frames in ``input_dir``.

    Frames are selected with :func:`select_dark_frames` (Sun below ``sun_alt_max``).
    Each frame's detections are capped to the brightest ``max_detections`` and
    matched against the current geometry with :func:`assign_alcor_matches` (kd-tree
    candidates, asterism pattern verification, local brightness tie-break). The
    matcher never refits per frame; instead the whole night is pooled and fit once
    per round under a single global geometry. The match tolerance tightens
    geometrically over ``match_rounds`` rounds from ``tolerance_start`` down to
    ``tolerance`` so that each round's better seed admits a cleaner pool. A final
    pool at the tightest tolerance is fit after 3*MAD outlier rejection.

    The fit runs on a neutral (uncalibrated) frame -- loaded with no recentering,
    rotation, or radial distortion -- so the recovered (xshift, yshift, rotation,
    radial_coeffs) are the ABSOLUTE geometry constants for the night, suitable for
    baking into ``ALCOR_CALIBRATIONS``. It is warm-started from the nearest
    existing epoch (via :func:`alcor_calibration` at the night's median time).

    Returns a dict with the fitted absolute parameters plus an ``epoch`` date
    string (the night's UT date, or the seed epoch when no frame can be timed),
    ``n_matched``, ``residual_rms``, ``matched_fraction`` (matched stars divided by
    the available catalog-star-frames, so contamination/coverage is visible), and
    per-match arrays (``alt``, ``az``, ``x``, ``y``) for diagnostics.

    The per-frame load/detect/catalog work is the expensive part and is
    independent across frames, so it is parallelized: ``workers=1`` runs
    serially, any larger value (or ``None`` for the process-pool default)
    distributes the frames over a `~concurrent.futures.ProcessPoolExecutor`.
    Pass a ``log`` callable (e.g. ``print``) to report each file's disposition:
    frames skipped because the Sun is above ``sun_alt_max``, frames skipped
    because no stars were detected, and frames used (with detected star count).
    """
    if workers is not None and workers < 1:
        raise ValueError("workers must be None or a positive integer")
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    dark = select_dark_frames(files, sun_alt_max=sun_alt_max, log=log)
    if log is not None:
        dark_set = set(dark)
        for f in files:
            if f not in dark_set:
                log(f"{Path(f).name}: skipped (Sun above {sun_alt_max:g} deg)")
    if max_frames is not None:
        dark = dark[:max_frames]

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

    init = dict(xshift=base["xshift"], yshift=base["yshift"],
                rotation=base["rotation"], radial_coeffs=base["radial_coeffs"])
    # (cat, detections) per usable frame, kept in frame order for reproducible
    # pooling regardless of worker completion order.
    detected = [None] * len(dark)
    tasks = [(index, f, vmag_limit, min_alt, fwhm, threshold_sigma, max_detections)
             for index, f in enumerate(dark)]

    def _store(result):
        index, cat, det, reason = result
        name = Path(dark[index]).name
        if cat is not None:
            detected[index] = (cat, det)
            if log is not None:
                log(f"{name}: {len(det)} stars detected")
        elif log is not None:
            log(f"{name}: skipped ({reason})")

    if workers == 1:
        for task in tasks:
            _store(_detect_alcor_frame(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_detect_alcor_frame, task) for task in tasks]
            for future in as_completed(futures):
                _store(future.result())

    frames = [d for d in detected if d is not None]
    available = sum(len(cat) for cat, _ in frames)

    def pool(seed_params, tol):
        a, z, xs, ys = [], [], [], []
        for cat, det in frames:
            matched = assign_alcor_matches(cat, det, params=seed_params, tolerance=tol)
            if len(matched) == 0:
                continue
            a.append(np.asarray(matched["Alt"], dtype=float))
            z.append(np.asarray(matched["Az"], dtype=float))
            xs.append(np.asarray(matched["xcentroid"], dtype=float))
            ys.append(np.asarray(matched["ycentroid"], dtype=float))
        if not a:
            return None
        return (np.concatenate(a), np.concatenate(z),
                np.concatenate(xs), np.concatenate(ys))

    # Tightening tolerance schedule: each round re-pools with the refined seed.
    schedule = np.geomspace(tolerance_start, tolerance, match_rounds)
    params = dict(init)
    for tol in schedule:
        pooled = pool(params, float(tol))
        if pooled is None:
            continue
        alt, az, x, y = pooled
        if len(alt) >= 3:
            params = _fit_params(alt, az, x, y, init_params=params)

    # Final pool at the tightest tolerance, then 3*MAD outlier rejection + refit.
    pooled = pool(params, float(tolerance))
    if pooled is None:
        raise RuntimeError("No matched stars across the selected frames.")
    alt, az, x, y = pooled

    px, py = _predict_pixels(alt, az, xshift=params["xshift"], yshift=params["yshift"],
                             rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]))
    resid = np.hypot(px - x, py - y)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    good = resid < np.median(resid) + 3.0 * 1.4826 * mad
    if good.sum() >= 3:
        params = _fit_params(alt[good], az[good], x[good], y[good], init_params=params)
    px, py = _predict_pixels(alt[good], az[good], xshift=params["xshift"],
                             yshift=params["yshift"], rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]))
    rms = float(np.sqrt(np.mean((px - x[good]) ** 2 + (py - y[good]) ** 2)))

    return {
        **params,
        "epoch": epoch,
        "n_matched": int(good.sum()),
        "residual_rms": rms,
        "matched_fraction": float(int(good.sum()) / available) if available else 0.0,
        "alt": alt[good], "az": az[good], "x": x[good], "y": y[good],
    }
```

- [ ] **Step 5: Remove `match_alcor_stars`**

Delete the entire `match_alcor_stars` function (currently lines 186-241, ending just before `def _frame_time`). It has no remaining callers after Step 4.

- [ ] **Step 6: Remove the now-obsolete `match_alcor_stars` test**

In `skycam_utils/tests/test_alcor_wcs.py`, delete `test_match_alcor_stars_recovers_correspondences` (near line 228) and remove `match_alcor_stars` from the import block (line 30). Keep the `_fake_detections` helper only if still referenced; if nothing else uses it, delete it too. (Search: `grep -n "_fake_detections\|match_alcor_stars" skycam_utils/tests/test_alcor_wcs.py`.)

- [ ] **Step 7: Run the integration test, then the full suite**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py::test_fit_alcor_wcs_survives_injected_mismatches -q`
Expected: PASS

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -q`
Expected: all PASS. If `test_fit_alcor_wcs_aggregates_synthetic_frames` or `test_fit_alcor_wcs_parallel_matches_serial` fail on the `result["residual_rms"] < 0.1` / `n_matched >= 40` assertions, do NOT loosen them blindly — first confirm the final tight-tolerance round produces exact matches (it should, since those mocks place detections exactly at the true pixels). The two existing tests need no edit beyond what Step 6 covers; they already supply `Vmag` and `flux`.

- [ ] **Step 8: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "drive fit_alcor_wcs with hardened tightening-tolerance matcher"
```

---

### Task 4: CLI flags

**Files:**
- Modify: `skycam_utils/alcor.py` (`fit_alcor_wcs_cli` near line 1552)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing test**

Add to `skycam_utils/tests/test_alcor_wcs.py` (near the other CLI/format tests; if none exist, append at end of file). It checks the new defaults and that `--max-detections` reaches `fit_alcor_wcs`:

```python
def test_fit_alcor_wcs_cli_passes_new_flags(monkeypatch, capsys):
    import sys
    import skycam_utils.alcor as alcor_mod

    captured = {}

    def fake_fit(input_dir, **kwargs):
        captured.update(kwargs)
        return {"xshift": 1.0, "yshift": 2.0, "rotation": 0.3,
                "radial_coeffs": (1.0, 0.02, 0.0), "epoch": "2026-05-19",
                "n_matched": 123, "residual_rms": 2.5, "matched_fraction": 0.42,
                "alt": np.array([]), "az": np.array([]),
                "x": np.array([]), "y": np.array([])}

    monkeypatch.setattr(alcor_mod, "fit_alcor_wcs", fake_fit)
    monkeypatch.setattr(sys, "argv",
                        ["fit_alcor_wcs", "/tmp/night", "--max-detections", "150"])
    alcor_mod.fit_alcor_wcs_cli()

    assert captured["vmag_limit"] == 4.0       # new default
    assert captured["tolerance"] == 3.0        # new default
    assert captured["max_detections"] == 150
    out = capsys.readouterr().out
    assert "matched fraction" in out.lower()
```

NOTE: the test's local `import sys` is sufficient — no change to the test file's top-level imports is required.

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py::test_fit_alcor_wcs_cli_passes_new_flags -q`
Expected: FAIL — either `KeyError: 'max_detections'` (not passed through) or the vmag default assertion.

- [ ] **Step 3: Update the CLI**

In `fit_alcor_wcs_cli` (lines 1558-1591), change the `--vmag-limit` and `--tolerance` defaults, add `--max-detections`, pass it through, and print the matched fraction.

Change these two argument lines:

```python
    parser.add_argument("--vmag-limit", type=float, default=4.0, help="Faintest Vmag to use.")
```
```python
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="Final (tightest) match tolerance (pixels).")
```

Add this argument (next to `--tolerance`):

```python
    parser.add_argument("--max-detections", type=int, default=200,
                        help="Keep only the brightest N detections per frame.")
```

Update the `fit_alcor_wcs(...)` call to pass `max_detections`:

```python
    result = fit_alcor_wcs(
        args.input_dir, pattern=args.pattern, vmag_limit=args.vmag_limit,
        sun_alt_max=args.sun_alt_max, min_alt=args.min_alt, tolerance=args.tolerance,
        max_detections=args.max_detections, max_frames=args.max_frames,
        workers=args.workers, log=log,
    )
```

Add a matched-fraction print after the residual RMS line (after line 1585):

```python
    print(f"# matched fraction: {result['matched_fraction']:.3f}")
```

- [ ] **Step 4: Run the test, then the suite**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py::test_fit_alcor_wcs_cli_passes_new_flags -q`
Expected: PASS

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "expose --max-detections and new vmag/tolerance defaults on fit_alcor_wcs CLI"
```

---

### Task 5: Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the alcor description and the `fit_alcor_wcs` CLI block**

In `CLAUDE.md`, find the `fit_alcor_wcs <night-dir> ...` usage block (in the Common commands section). Update its flag list and description to reflect the hardened matcher. Replace the existing `fit_alcor_wcs` block with:

```
fit_alcor_wcs <night-dir> [--pattern ...] [--vmag-limit 4] [--tolerance 3] [--max-detections 200] [--residual-plot OUT.png] [--max-frames N] [--workers N] [--quiet]
#   Aggregates bright-star matches across Sun<-18deg frames across a night and prints
#   a ready-to-paste ALCOR_CALIBRATIONS epoch dict (absolute constants — center,
#   rotation, radial k3 — stamped with the night date) to add to alcor.py and commit.
#   Detections are capped to the brightest --max-detections per frame; matching is
#   seeded by the nearest epoch and done with assign_alcor_matches (cKDTree candidate
#   search, local-asterism pattern verification, and a local relative-brightness
#   tie-break for contested detections), with no per-frame geometry refit. The match
#   tolerance tightens over several rounds from ~12px to --tolerance (~3px). Also
#   prints the matched fraction so contamination/coverage is visible.
#   Dark-frame selection parses the UT from each YYYY_MM_DD__HH_MM_SS filename (local
#   MST = UT-7), so it never opens files; oddly-named files fall back to the DATE header.
#   Per-frame load/detect is parallelized across processes (--workers; default: one per core).
#   Prints each file's disposition to stderr (Sun-rejected / no stars / used + count); --quiet silences it.
```

Then, in the project-context paragraph describing alcor calibration (the `## Project context` section, the sentence about `fit_alcor_wcs()`), append one sentence after the existing description of the calibration table:

```
The matcher (`assign_alcor_matches`) is seeded by the resolved epoch geometry and uses a kd-tree with local-asterism pattern verification plus a local relative-brightness tie-break (cloud extinction is patchy, so only relative flux among nearby contested stars is trusted) rather than per-frame nearest-neighbor refitting.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "document hardened alcor matcher and fit_alcor_wcs flags"
```

---

## Self-review (completed during planning)

- **Spec coverage:** detection cap → Task 1; `assign_alcor_matches` (mutual/cluster, pattern, brightness) → Task 2; fixed-geometry, no per-frame refit, tightening schedule, matched-fraction, vmag 3→4, tolerance→3 → Task 3; CLI flags → Task 4; docs → Task 5. The out-of-scope 2-D azimuthal distortion model is intentionally not planned.
- **Type/name consistency:** `assign_alcor_matches(cat, det, params, tolerance, ...)` is called identically in `fit_alcor_wcs.pool()` and in tests; the task tuple gains `max_detections` and is unpacked the same way in `_detect_alcor_frame`; the result dict adds `matched_fraction` (float), printed by the CLI.
- **Interpretation note for the spec reviewer:** the spec phrases step 2 as "mutual nearest-neighbor within tolerance." The implementation realizes this via connected-component cluster resolution: an isolated 1:1 candidate *is* the mutual-nearest pair (accepted directly), and contested clusters are resolved by relative-brightness rank pairing (or nearest when `brightness=False`). This is the agreed brightness-tie-break semantics, generalized to resolve conflicts deterministically; it is not a deviation from intent.
- **Pattern-check escape:** pairs with fewer than `min_corroborating` paired neighbors are kept (insufficient local evidence to reject), so sparse frames are not wiped; dense frames still reject crowded-region mispairs. This is exercised by both the brightness test (tiny cluster, escape path) and the decoy test (dense cluster, rejection path).
- **No placeholders:** every code/test step is complete. The one scratch fragment in Task 2 Step 3 is explicitly called out for deletion with the corrected replacement shown.
```

