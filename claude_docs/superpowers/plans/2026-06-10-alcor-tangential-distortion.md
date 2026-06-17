# Alcor Brown–Conrady Tangential Distortion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Repo conventions:** work directly on `main` (no feature branch — repo memory), commit per task.

**Goal:** Add 2-parameter Brown–Conrady tangential (decentering) distortion to the Alcor lens model — fit always, encoded exactly in the WCS SIP — per `docs/superpowers/specs/2026-06-10-alcor-tangential-distortion-design.md`.

**Architecture:** The distortion is defined on the pixel→world side (like the radial k-terms): with `u, v` raw-pixel offsets from the zenith and `H = horizon_radius`, the SIP-applied displacement gains `Δu = (P1/H)(3u² + v²) + (2P2/H)uv`, `Δv = (P2/H)(u² + 3v²) + (2P1/H)uv` — an exact degree-2 polynomial, so the analytic SIP stays exact. `_predict_pixels` inverts it with a fixed-point loop that re-solves the radial part exactly each pass. `_fit_params` always fits P1/P2. Epoch dicts gain an optional `tangential_coeffs` key defaulting to `(0.0, 0.0)`.

**Tech Stack:** numpy, scipy.optimize.least_squares, astropy.wcs (Sip), pytest.

**Files (all tasks):**
- Modify: `skycam_utils/alcor.py` (single module — schema, forward model, WCS, fitter, threading)
- Test: `skycam_utils/tests/test_alcor_wcs.py`
- Modify: `CLAUDE.md` (model description, Task 6)

Line numbers below are pre-change positions; they shift as tasks land. Locate by symbol name.

---

### Task 1: Calibration schema and defaults

**Files:**
- Modify: `skycam_utils/alcor.py` — `ALCOR_CALIBRATIONS` comment (~line 42), `alcor_calibration` (~line 66), module constants (~line 87), `_format_calibration_entry` (~line 1938)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py` (note `alcor_calibration` and `_format_calibration_entry` are imported inside existing tests; follow that pattern):

```python
def test_alcor_calibration_defaults_tangential_coeffs():
    from skycam_utils.alcor import alcor_calibration

    # Shipped epochs have no tangential_coeffs key; the resolver must fill it.
    cal = alcor_calibration()
    assert cal["tangential_coeffs"] == (0.0, 0.0)
    cal = alcor_calibration(Time("2024-09-05", scale="utc"))
    assert cal["tangential_coeffs"] == (0.0, 0.0)


def test_format_calibration_entry_includes_tangential():
    import ast
    from skycam_utils.alcor import _format_calibration_entry

    line = _format_calibration_entry(dict(
        epoch="2026-05-19", xcen=-12.0, ycen=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0), tangential_coeffs=(0.004, -0.003),
        horizon_radius=662.0))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["tangential_coeffs"] == (0.004, -0.003)
    # results without the key (old callers) format with the zero default
    line = _format_calibration_entry(dict(
        epoch="2026-05-19", xcen=-12.0, ycen=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0), horizon_radius=662.0))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["tangential_coeffs"] == (0.0, 0.0)
```

(`Time` is already imported at the top of the test module; check, and add `from astropy.time import Time` if not.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "tangential" -v`
Expected: 2 FAILED (KeyError `tangential_coeffs` / missing key in parsed dict).

- [ ] **Step 3: Implement**

In `skycam_utils/alcor.py`:

(a) Extend the `ALCOR_CALIBRATIONS` block comment (~line 42–48) — after the sentence about horizon_radius, add:

```python
# zenith to alt=0). An optional "tangential_coeffs": (P1, P2) holds the
# Brown-Conrady decentering (sensor-tilt) terms, dimensionless like the k's;
# epochs without the key mean (0.0, 0.0) (alcor_calibration fills the default).
```

(b) Rewrite `alcor_calibration` so both return paths fill the default:

```python
def alcor_calibration(time=None):
    """
    Return the calibration dict whose epoch is nearest in time to ``time``.

    ``time`` is an astropy ``Time``. An exact tie resolves to the more recent
    epoch. ``time=None`` returns the most recent epoch (the default for
    time-agnostic calls). The returned dict is a copy and may be mutated freely;
    ``tangential_coeffs`` is filled with ``(0.0, 0.0)`` for epochs that omit it.
    """
    epochs = _calibration_epochs()
    if time is None:
        cal = dict(max(epochs, key=lambda e: e[0].jd)[1])
    else:
        jds = np.array([e[0].jd for e in epochs])
        dt = np.abs(jds - Time(time).jd)
        # primary: smallest |dt|; tie-break: largest jd (more recent)
        order = np.lexsort((-jds, dt))
        cal = dict(epochs[order[0]][1])
    cal.setdefault("tangential_coeffs", (0.0, 0.0))
    return cal
```

(c) Add the module default after `ALCOR_RADIAL_COEFFS` (~line 90):

```python
ALCOR_TANGENTIAL_COEFFS = _LATEST_CALIBRATION["tangential_coeffs"]
```

(d) Add `tangential_coeffs` to `_format_calibration_entry` (between radial_coeffs and horizon_radius, with a `.get` default so old result dicts still format):

```python
def _format_calibration_entry(result):
    """Format a calibration result as a paste-ready ALCOR_CALIBRATIONS entry."""
    rc = tuple(float(c) for c in result["radial_coeffs"])
    tc = tuple(float(c) for c in result.get("tangential_coeffs", (0.0, 0.0)))
    return (f'    {{"epoch": "{result["epoch"]}", '
            f'"xcen": {result["xcen"]:.3f}, '
            f'"ycen": {result["ycen"]:.3f}, '
            f'"rotation": {result["rotation"]:.4f}, '
            f'"radial_coeffs": {rc!r}, '
            f'"tangential_coeffs": {tc!r}, '
            f'"horizon_radius": {result["horizon_radius"]:.1f}}},')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "tangential or calibration or format" -v`
Expected: all PASS (including the pre-existing `test_alcor_calibration_nearest_in_time` and `test_format_calibration_entry_is_parseable`).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: optional tangential_coeffs in calibration schema

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Forward model — `_predict_pixels` tangential support

**Files:**
- Modify: `skycam_utils/alcor.py` — `_predict_pixels` (~line 112); new helper `_tangential_delta` directly above it
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
def test_predict_pixels_tangential_zero_is_noop():
    alt = np.array([80.0, 45.0, 10.0])
    az = np.array([15.0, 120.0, 300.0])
    bx, by = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                             radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0)
    tx, ty = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                             radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0,
                             tangential_coeffs=(0.0, 0.0))
    np.testing.assert_array_equal(tx, bx)
    np.testing.assert_array_equal(ty, by)


def test_predict_pixels_tangential_satisfies_plate_solution():
    """The forward model must invert the exact pix->world plate solution:
    (u, v) + D_radial(u, v) + D_tangential(u, v) lands on the linear ARC target,
    whose radius is H * z / (90 * k1) along the (rotation - az) direction."""
    k1, k3, k5 = 1.0, 0.09, 0.02
    H = 747.0
    p1, p2 = 3e-3, -2e-3
    rng = np.random.default_rng(11)
    alt = rng.uniform(2.0, 88.0, 200)
    az = rng.uniform(0.0, 360.0, 200)
    x, y = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                           radial_coeffs=(k1, k3, k5), horizon_radius=H,
                           tangential_coeffs=(p1, p2))
    u = x - 700.0
    v = y - 710.0
    rho2 = (u**2 + v**2) / H**2
    drad = (k3 * rho2 + k5 * rho2**2) / k1
    du = (p1 / H) * (3.0 * u**2 + v**2) + (2.0 * p2 / H) * u * v
    dv = (p2 / H) * (u**2 + 3.0 * v**2) + (2.0 * p1 / H) * u * v
    fu = u * (1.0 + drad) + du
    fv = v * (1.0 + drad) + dv
    s = H * (90.0 - alt) / (90.0 * k1)
    ang = np.radians(-1.0 - az)
    np.testing.assert_allclose(fu, s * np.sin(ang), atol=1e-3)
    np.testing.assert_allclose(fv, s * np.cos(ang), atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "predict_pixels_tangential" -v`
Expected: 2 FAILED (`TypeError: ... unexpected keyword argument 'tangential_coeffs'`).

- [ ] **Step 3: Implement**

In `skycam_utils/alcor.py`, insert above `_predict_pixels`:

```python
def _tangential_delta(u, v, p1, p2, horizon_radius):
    """Brown-Conrady tangential (decentering) displacement in raw pixels.

    ``u``, ``v`` are pixel offsets from the zenith; ``p1``/``p2`` are
    dimensionless (normalized by ``horizon_radius``), like the radial k
    coefficients. This is the pix->world displacement the WCS SIP applies
    (see build_alcor_wcs); it is an exact degree-2 polynomial.
    """
    H = float(horizon_radius)
    du = (p1 / H) * (3.0 * u**2 + v**2) + (2.0 * p2 / H) * u * v
    dv = (p2 / H) * (u**2 + 3.0 * v**2) + (2.0 * p1 / H) * u * v
    return du, dv
```

Replace `_predict_pixels` with (docstring gains the tangential paragraph; the radial path is byte-identical math when P1 = P2 = 0):

```python
def _predict_pixels(
    alt,
    az,
    xcen=ALCOR_XCEN,
    ycen=ALCOR_YCEN,
    rotation=0.0,
    radial_coeffs=ALCOR_RADIAL_COEFFS,
    horizon_radius=ALCOR_HORIZON_RADIUS,
    tangential_coeffs=(0.0, 0.0),
):
    """Forward lens model: map altitude/azimuth (deg) to RAW-frame pixel
    coordinates (x=column, y=row, 0-based).

    The zenith sits at ``(xcen, ycen)``; ``rotation`` is the camera azimuth
    zero-point offset (deg). The lens plate solution
    ``z = 90*(k1*rho + k3*rho**3 + k5*rho**5)`` (``rho = r/horizon_radius``,
    ``z = 90 - alt``) is inverted for ``rho`` via Newton's method. The sky's
    azimuth runs opposite to the sensor's polar angle (an all-sky camera images
    the sky as seen from below), so the pixel angle is ``rotation - az``; north
    (az=0) lands toward +y. The matching WCS encodes the same mapping in its PC
    rotation matrix (see ``build_alcor_wcs``).

    ``tangential_coeffs`` (P1, P2) adds Brown-Conrady decentering, defined like
    the k's on the pix->world side (`_tangential_delta`). It is inverted with a
    fixed-point loop that re-solves the radial part exactly each pass, so the
    contraction is governed by the (tiny, ~4*P) tangential derivative rather
    than the O(k3, k5) radial one; three passes reach well below 1e-3 px for
    |P| up to ~1e-2.
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    coeffs = tuple(float(c) for c in radial_coeffs)
    rho = _invert_radial(90.0 - alt, coeffs)
    r = horizon_radius * rho
    ang = np.radians(rotation - az)
    u = r * np.sin(ang)
    v = r * np.cos(ang)

    p1, p2 = (float(c) for c in tangential_coeffs)
    if p1 != 0.0 or p2 != 0.0:
        k1 = coeffs[0]
        H = float(horizon_radius)
        # Linear-pixel target of the SIP equation t = (u,v) + D_rad + D_tan:
        # the radial displacement preserves direction, so |t| = H*z/(90*k1).
        s = H * (90.0 - alt) / (90.0 * k1)
        tu = s * np.sin(ang)
        tv = s * np.cos(ang)
        for _ in range(3):
            du, dv = _tangential_delta(u, v, p1, p2, H)
            wu = tu - du
            wv = tv - dv
            wr = np.hypot(wu, wv)
            safe = np.where(wr > 0.0, wr, 1.0)
            rho_w = _invert_radial(90.0 * k1 * wr / H, coeffs)
            scale = np.where(wr > 0.0, H * rho_w / safe, 0.0)
            u = wu * scale
            v = wv * scale

    x = xcen + u
    y = ycen + v
    return x, y
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "predict_pixels" -v`
Expected: all PASS (new 2 plus the 5 pre-existing `predict_pixels` tests).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: Brown-Conrady tangential terms in the forward lens model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: WCS — exact tangential SIP in `build_alcor_wcs`

**Files:**
- Modify: `skycam_utils/alcor.py` — `build_alcor_wcs` (~line 914), `_build_alcor_wcs_cached` (~line 943)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
def test_build_alcor_wcs_with_tangential_reproduces_forward_model():
    coeffs = (1.0, 0.02, 0.05)
    tc = (0.004, -0.003)
    xcen, ycen, hr = 696.0, 698.0, 662.0
    wcs = build_alcor_wcs(xcen=xcen, ycen=ycen, rotation=0.4,
                          radial_coeffs=coeffs, horizon_radius=hr,
                          tangential_coeffs=tc)
    assert wcs.sip is not None

    alt = np.array([80.0, 60.0, 40.0, 20.0, 5.0])
    az = np.array([10.0, 100.0, 190.0, 280.0, 350.0])
    mx, my = _predict_pixels(alt, az, xcen=xcen, ycen=ycen, rotation=0.4,
                             radial_coeffs=coeffs, horizon_radius=hr,
                             tangential_coeffs=tc)
    wx, wy = wcs.world_to_pixel_values(az, alt)
    np.testing.assert_allclose(wx, mx, atol=1e-3)
    np.testing.assert_allclose(wy, my, atol=1e-3)
    # pixel->world applies the exact A/B; round-trip back to the inputs
    waz, walt = wcs.pixel_to_world_values(mx, my)
    np.testing.assert_allclose(walt, alt, atol=1e-4)
    daz = (waz - az + 180.0) % 360.0 - 180.0   # wrap-safe angular difference
    np.testing.assert_allclose(daz, 0.0, atol=1e-3)


def test_build_alcor_wcs_tangential_only_attaches_sip():
    # No radial distortion but nonzero tangential terms must still get a SIP.
    wcs = build_alcor_wcs(xcen=696.0, ycen=698.0, rotation=0.0,
                          radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=662.0,
                          tangential_coeffs=(0.003, 0.0))
    assert wcs.sip is not None
    assert list(wcs.wcs.ctype) == ["RA---ARC-SIP", "DEC--ARC-SIP"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "build_alcor_wcs_with_tangential or tangential_only" -v`
Expected: 2 FAILED (`TypeError: ... unexpected keyword argument 'tangential_coeffs'`).

- [ ] **Step 3: Implement**

In `skycam_utils/alcor.py`, change `build_alcor_wcs`'s signature and cache call (docstring gains one sentence noting the exact degree-2 tangential terms):

```python
def build_alcor_wcs(xcen=ALCOR_XCEN, ycen=ALCOR_YCEN, rotation=ALCOR_ROTATION,
                    radial_coeffs=ALCOR_RADIAL_COEFFS,
                    horizon_radius=ALCOR_HORIZON_RADIUS, sip_degree=5,
                    tangential_coeffs=ALCOR_TANGENTIAL_COEFFS):
```

with the docstring sentence (append to the second paragraph):

```
    ``tangential_coeffs`` (P1, P2) adds Brown-Conrady decentering; its Cartesian
    displacement is an exact degree-2 polynomial (see ``_tangential_delta``), so
    it joins the analytic SIP without approximation.
```

and the return:

```python
    return _build_alcor_wcs_cached(
        float(xcen), float(ycen), float(rotation),
        tuple(float(c) for c in radial_coeffs),
        float(horizon_radius), int(sip_degree),
        tuple(float(c) for c in tangential_coeffs),
    ).deepcopy()
```

Replace `_build_alcor_wcs_cached`:

```python
@lru_cache(maxsize=32)
def _build_alcor_wcs_cached(xcen, ycen, rotation, radial_coeffs, horizon_radius,
                            sip_degree, tangential_coeffs=(0.0, 0.0)):
    k1, k3, k5 = radial_coeffs
    p1, p2 = tangential_coeffs
    base = _base_arc_wcs(xcen, ycen, rotation, k1, horizon_radius)
    if (abs(k3) < 1e-12 and abs(k5) < 1e-12
            and abs(p1) < 1e-12 and abs(p2) < 1e-12):
        return base

    # Analytic SIP for the radial plate solution. The Cartesian displacement of
    # z = 90*(k1*rho + k3*rho**3 + k5*rho**5) is A_u = u*(k3*rho**2 + k5*rho**4)/k1
    # with rho = sqrt(u**2 + v**2)/horizon_radius -- an exact degree-5 polynomial.
    H = float(horizon_radius)
    c3 = k3 / (k1 * H**2)
    c5 = k5 / (k1 * H**4)
    a = np.zeros((sip_degree + 1, sip_degree + 1))
    b = np.zeros((sip_degree + 1, sip_degree + 1))
    a[3, 0] = c3; a[1, 2] = c3
    a[5, 0] = c5; a[3, 2] = 2 * c5; a[1, 4] = c5
    b[0, 3] = c3; b[2, 1] = c3
    b[0, 5] = c5; b[2, 3] = 2 * c5; b[4, 1] = c5
    # Brown-Conrady tangential (decentering) terms: an exact degree-2 polynomial
    # in the pixel offsets (see _tangential_delta).
    a[2, 0] = 3.0 * p1 / H; a[0, 2] = p1 / H; a[1, 1] = 2.0 * p2 / H
    b[0, 2] = 3.0 * p2 / H; b[2, 0] = p2 / H; b[1, 1] = 2.0 * p1 / H
    ap, bp = _fit_sip_inverse(a, b, int(round(H)), sip_degree)

    wcs = base.deepcopy()
    wcs.wcs.ctype = ["RA---ARC-SIP", "DEC--ARC-SIP"]
    wcs.sip = Sip(a, b, ap, bp, [xcen + 1.0, ycen + 1.0])
    return wcs
```

(The radial terms occupy only odd-total-degree slots `(3,0),(1,2),(5,0),(3,2),(1,4)` and the tangential only even-total-degree slots `(2,0),(0,2),(1,1)`, so plain assignment is collision-free.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "build_alcor_wcs" -v`
Expected: all PASS (2 new + 3 pre-existing, confirming the radial-only and linear paths are unchanged).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: encode tangential distortion as exact degree-2 SIP terms

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Fitter — `_fit_params` always solves P1/P2

**Files:**
- Modify: `skycam_utils/alcor.py` — `_fit_params` (~line 143)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing test**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
def test_fit_params_recovers_tangential_coeffs():
    rng = np.random.default_rng(13)
    alt = rng.uniform(5.0, 88.0, 400)
    az = rng.uniform(0.0, 360.0, 400)
    true = dict(xcen=5.0, ycen=-4.0, rotation=0.7,
                radial_coeffs=(1.0, 0.08, 0.0),
                tangential_coeffs=(0.004, -0.003))
    x, y = _predict_pixels(alt, az, **true)

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)))
    assert abs(fit["tangential_coeffs"][0] - 0.004) < 2e-4
    assert abs(fit["tangential_coeffs"][1] + 0.003) < 2e-4
    assert abs(fit["xcen"] - 5.0) < 0.05
    assert abs(fit["ycen"] + 4.0) < 0.05
    assert abs(fit["radial_coeffs"][1] - 0.08) < 1e-3
    # k3-only mode still pins k5 at zero with tangential terms free
    assert fit["radial_coeffs"][2] == 0.0

    # fit_k5 branch also carries the tangential terms
    fit5 = _fit_params(alt, az, x, y,
                       init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                        radial_coeffs=(1.0, 0.0, 0.0)),
                       fit_k5=True)
    assert abs(fit5["tangential_coeffs"][0] - 0.004) < 5e-4
    assert abs(fit5["tangential_coeffs"][1] + 0.003) < 5e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py::test_fit_params_recovers_tangential_coeffs -v`
Expected: FAIL with `KeyError: 'tangential_coeffs'`.

- [ ] **Step 3: Implement**

Replace `_fit_params` entirely (the two near-duplicate branches collapse into one parametrization; behavior for the radial parameters is unchanged):

```python
def _fit_params(alt, az, obs_x, obs_y, init_params,
                horizon_radius=ALCOR_HORIZON_RADIUS, fit_k5=False):
    """
    Robust least-squares fit of the lens geometry to matched stars.

    By default fits (xcen, ycen, rotation, k3, P1, P2) with k1 held at 1.0 (the
    zenith plate scale is set by horizon_radius) and k5 at 0.0. k3 and k5 are
    nearly collinear in rho over [0, 1], so fitting both is ill-conditioned on
    *dirty* data and runs away to large cancelling values (e.g. k3=-0.58,
    k5=3.6) that are unphysical despite a tolerable RMS -- which is why k3
    alone is the default (the model the shipped 2024 constants used). With
    ``fit_k5=True`` the odd quintic term k5 is fit as well: appropriate only on
    a clean, well-distributed match set (asterism-verified, spanning the full
    zenith range), where the radial residual that k3 alone leaves near the
    horizon can be captured.

    The Brown-Conrady tangential terms (P1, P2) are always fit: unlike k3/k5
    they are well-conditioned against the other parameters (their displacement
    grows as r**2 and varies once per azimuth revolution, while a center shift
    is constant and rotation grows as r), and they capture the sensor-tilt /
    decentering signature the azimuthally-symmetric radial basis cannot.

    A ``soft_l1`` loss downweights mismatched/noise detections, common in this
    sparse bright-star field. Returns an updated params dict with
    radial_coeffs=(1.0, k3, k5) (k5=0.0 unless ``fit_k5``) and
    tangential_coeffs=(P1, P2).
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    obs_x = np.asarray(obs_x, dtype=float)
    obs_y = np.asarray(obs_y, dtype=float)
    init_k3 = init_params["radial_coeffs"][1]
    init_k5 = init_params["radial_coeffs"][2]
    init_p1, init_p2 = init_params.get("tangential_coeffs", (0.0, 0.0))

    p0 = [init_params["xcen"], init_params["ycen"],
          init_params["rotation"], init_k3]
    if fit_k5:
        p0.append(init_k5)
    p0 += [init_p1, init_p2]
    p0 = np.asarray(p0, dtype=float)

    def unpack(p):
        if fit_k5:
            xcen, ycen, rot, k3, k5, p1, p2 = p
        else:
            xcen, ycen, rot, k3, p1, p2 = p
            k5 = 0.0
        return xcen, ycen, rot, k3, k5, p1, p2

    def residuals(p):
        xcen, ycen, rot, k3, k5, p1, p2 = unpack(p)
        x, y = _predict_pixels(alt, az, xcen=xcen, ycen=ycen, rotation=rot,
                               radial_coeffs=(1.0, k3, k5),
                               horizon_radius=horizon_radius,
                               tangential_coeffs=(p1, p2))
        return np.concatenate([x - obs_x, y - obs_y])

    result = least_squares(residuals, p0, loss="soft_l1", f_scale=3.0)
    xcen, ycen, rot, k3, k5, p1, p2 = unpack(result.x)
    return dict(xcen=float(xcen), ycen=float(ycen), rotation=float(rot),
                radial_coeffs=(1.0, float(k3), float(k5)),
                tangential_coeffs=(float(p1), float(p2)),
                horizon_radius=float(horizon_radius))
```

- [ ] **Step 4: Run all fitter tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "fit_params" -v`
Expected: all PASS — the new test plus the 3 pre-existing ones (`recovers_known_geometry`, `recovers_k5_when_enabled`, `stays_physical_with_mismatches`). The pre-existing tests now run with P1/P2 free; on their clean or robustly-handled synthetic data the tangential terms fit to ~0 and the existing tolerances hold. If `stays_physical_with_mismatches` fails, that is a real conditioning regression — stop and investigate, do not loosen the test.

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: always fit Brown-Conrady tangential terms in _fit_params

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Thread `tangential_coeffs` through matching, night fit, residual plot, and load

**Files:**
- Modify: `skycam_utils/alcor.py` — `assign_alcor_matches` (~line 241), `fit_alcor_wcs` (~lines 476, 543, 553), `save_alcor_residual_plot` (~line 627), `load_alcor_fits` (~line 1323)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

The threading rule (learned from the `horizon_radius` bug): every place the geometry travels as a params dict must carry `tangential_coeffs`, with a `(0.0, 0.0)` `.get` default so bare dicts (tests, old callers) keep working.

- [ ] **Step 1: Write the failing end-to-end test**

Append to `skycam_utils/tests/test_alcor_wcs.py` (mirrors `test_fit_alcor_wcs_aggregates_synthetic_frames`, with tangential truth injected; note the monkeypatched `alcor_calibration` deliberately returns a dict WITHOUT `tangential_coeffs` to prove the `.get` defaults hold):

```python
def test_fit_alcor_wcs_recovers_tangential_terms(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xcen=6.0, ycen=-5.0, rotation=0.8,
                radial_coeffs=(1.0, 0.09, 0.0),
                tangential_coeffs=(0.004, -0.003))
    rng = np.random.default_rng(3)

    frame_alt = [rng.uniform(10.0, 88.0, 30), rng.uniform(10.0, 88.0, 30)]
    frame_az = [rng.uniform(0.0, 360.0, 30), rng.uniform(0.0, 360.0, 30)]
    files = [tmp_path / "2024_09_05__00_00_00.fits",
             tmp_path / "2024_09_05__00_10_00.fits"]
    for f in files:
        f.write_bytes(b"stub")

    calls = {"i": 0}

    def fake_load_alcor_fits(path, **kw):
        return np.zeros((3, 2 * ALCOR_RADIUS, 2 * ALCOR_RADIUS)), None, None

    def fake_reference_altaz(time, **kw):
        i = calls["i"]
        return Table({"Alt": frame_alt[i], "Az": frame_az[i],
                      "Vmag": rng.uniform(0.5, 3.0, 30), "HD": np.arange(30)})

    def fake_detect(im, **kw):
        i = calls["i"]
        calls["i"] += 1
        x, y = _predict_pixels(frame_alt[i], frame_az[i], **true)
        return Table({"xcentroid": x, "ycentroid": y,
                      "flux": np.linspace(1e3, 1e2, 30)})

    monkeypatch.setattr(alcor_mod, "select_dark_frames",
                        lambda fs, **kw: list(files))
    monkeypatch.setattr(alcor_mod, "load_alcor_fits", fake_load_alcor_fits)
    monkeypatch.setattr(alcor_mod, "alcor_reference_altaz", fake_reference_altaz)
    monkeypatch.setattr(alcor_mod, "detect_alcor_stars", fake_detect)
    monkeypatch.setattr(alcor_mod, "_frame_time",
                        lambda path: Time("2024-09-05T07:00:00", format="isot",
                                          scale="utc"))
    # No tangential_coeffs key: the night fit must default it to (0, 0).
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xcen": 0.0,
                                           "ycen": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0),
                                           "horizon_radius": 747.0})

    result = fit_alcor_wcs(tmp_path, pattern="*.fits")
    assert abs(result["tangential_coeffs"][0] - 0.004) < 5e-4
    assert abs(result["tangential_coeffs"][1] + 0.003) < 5e-4
    assert abs(result["xcen"] - 6.0) < 0.1
    assert abs(result["ycen"] + 5.0) < 0.1
    assert result["residual_rms"] < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py::test_fit_alcor_wcs_recovers_tangential_terms -v`
Expected: FAIL — `fit_alcor_wcs` already returns `tangential_coeffs` via `**params` (Task 4), but the final-pool `_predict_pixels` calls ignore it, so `residual_rms` stays large (or the matching pool degrades). The failure is on the RMS/coefficient assertions, not a KeyError.

- [ ] **Step 3: Implement the threading**

In `skycam_utils/alcor.py`:

(a) `assign_alcor_matches` — add to the `_predict_pixels` call (~line 241):

```python
    px, py = _predict_pixels(
        cat["Alt"], cat["Az"], xcen=params["xcen"], ycen=params["ycen"],
        rotation=params["rotation"], radial_coeffs=tuple(params["radial_coeffs"]),
        horizon_radius=params.get("horizon_radius", horizon_radius),
        tangential_coeffs=tuple(params.get("tangential_coeffs", (0.0, 0.0))),
    )
```

(b) `fit_alcor_wcs` — the warm-start init (~line 476) carries the seed epoch's value (defensive `.get` because tests/callers may inject bare dicts):

```python
    init = dict(xcen=base["xcen"], ycen=base["ycen"],
                rotation=base["rotation"], radial_coeffs=base["radial_coeffs"],
                tangential_coeffs=base.get("tangential_coeffs", (0.0, 0.0)),
                horizon_radius=base["horizon_radius"])
```

(c) `fit_alcor_wcs` — both final-pool `_predict_pixels` calls (~lines 543 and 553) gain:

```python
                             tangential_coeffs=tuple(params["tangential_coeffs"]),
```

(after Task 4, `params` past the first `_fit_params` round always has the key; the pre-fit pool seeding goes through `assign_alcor_matches`, covered by (a)). Note: `params = dict(init)` before the rounds already contains the key from (b), so the first `pool()` call is also seeded correctly.

(d) `save_alcor_residual_plot` — the refined prediction (~line 627) uses the full fitted model (the idealized baseline stays radial-only by design):

```python
    fx, fy = _predict_pixels(alt, az, xcen=cenx, ycen=ceny,
                             rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]),
                             horizon_radius=hr,
                             tangential_coeffs=tuple(
                                 params.get("tangential_coeffs", (0.0, 0.0))))
```

(e) `load_alcor_fits` — the epoch-resolved WCS build (~line 1323):

```python
        wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                              rotation=cal["rotation"],
                              radial_coeffs=cal["radial_coeffs"],
                              horizon_radius=cal["horizon_radius"],
                              tangential_coeffs=cal["tangential_coeffs"])
```

(`alcor_calibration` always fills the key — Task 1 — so no `.get` needed here.)

(f) `fit_alcor_wcs` docstring: in the sentence "the recovered (xcen, ycen, rotation, radial_coeffs) are the ABSOLUTE raw-frame geometry constants", extend the tuple to "(xcen, ycen, rotation, radial_coeffs, tangential_coeffs)".

- [ ] **Step 4: Run the new test plus the whole night-fit and plot test set**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "fit_alcor_wcs or residual_plot or assign_alcor or load_alcor" -v`
Expected: all PASS, including the new recovery test and every pre-existing test (bare params dicts without the key must keep working everywhere).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: thread tangential_coeffs through matching, night fit, and WCS load

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Docs + full-suite verification

**Files:**
- Modify: `CLAUDE.md` (Alcor paragraph in "Project context"; `fit_alcor_wcs` entry in "Common commands")
- Modify: `skycam_utils/alcor.py` — `fit_alcor_wcs_cli` docstring/epilog if it enumerates the printed constants (check; the `--fit-k5` help text stays as-is)

- [ ] **Step 1: Update CLAUDE.md**

In the Alcor bullet of "Project context": where the fitted geometry is enumerated ("absolute raw-frame constants `xcen`/`ycen`/`rotation`/`radial_coeffs`/`horizon_radius`"), add `tangential_coeffs` to the list, and append one sentence after the SIP description:

```
the fitted odd-power radial (k3/k5) lens distortion as an exact analytic SIP,
plus Brown–Conrady tangential decentering terms (P1/P2, fit always, exact
degree-2 SIP) that absorb the sensor-tilt signature (once-per-azimuth residual
growing as r²)
```

In the `fit_alcor_wcs` command block, update the printed-constants list the same way (add `tangential_coeffs`).

- [ ] **Step 2: Check the CLI help for stale constant lists**

Run: `grep -n "xcen, ycen, rotation" skycam_utils/alcor.py`
Update any docstring/help string that enumerates the fitted constants to include `tangential_coeffs` (at minimum the `fit_alcor_wcs_cli` docstring and the module-level `ALCOR_CALIBRATIONS` comment were handled in Tasks 1/5 — verify nothing else enumerates them).

- [ ] **Step 3: Run the full test suite**

Run: `pytest`
Expected: all tests pass (the suite spans `test_alcor.py`, `test_alcor_badpix.py`, `test_alcor_wcs.py` plus the stellacam tests).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md skycam_utils/alcor.py
git commit -m "alcor: document tangential decentering terms

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Post-implementation validation (operational, user-run)

Not part of this plan's automated steps — per the spec, the user refits the
2026-05-18 night on the remote machine:

```bash
fit_alcor_wcs <night-dir> --residual-plot 20260518_tangential.png
```

Success: the once-per-rev sinusoid in the radial/tangential-vs-azimuth panels
collapses; pooled RMS drops meaningfully below 1.54 px; |P1|, |P2| ≲ a few
×10⁻³ with the other parameters stable. The printed epoch entry (now including
`tangential_coeffs`) replaces the 2026-05-19 entry in `ALCOR_CALIBRATIONS`.
