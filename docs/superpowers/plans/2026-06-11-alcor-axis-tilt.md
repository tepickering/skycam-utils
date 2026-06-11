# Alcor Optical-Axis Tilt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Repo conventions:** work directly on `main` (no feature branch — repo memory), commit per task.

**Goal:** Model the lens optical axis pointing off-zenith — exact world-side pole offset, always fit, Cartesian `(t_n, t_e)` parametrization — per `docs/superpowers/specs/2026-06-11-alcor-axis-tilt-design.md`.

**Architecture:** A star's `(alt, az)` is transformed to axis-centered polar coordinates `(z', A')` by the exact minimal rotation (by ε about the horizontal axis at azimuth A₀+90°); the existing radial/tangential machinery runs on `(z', A')`. In the WCS the tilt is pure FITS-WCS geometry: `CRVAL = (A₀, 90−ε)`, `LONPOLE = A₀`, SIP unchanged. `xcen`/`ycen` becomes the optical-axis pixel; zenith consumers switch from CRPIX to `world_to_pixel_values(0, 90)`.

**Tech Stack:** numpy (vectorized Rodrigues rotation), astropy.wcs/WCSLIB (spherical rotation), scipy.optimize.least_squares, pytest.

**Derived math (used in Tasks 2–3; verify, don't re-derive):**
- Unit vectors: x→north (az=0), y→east (az=90), z→up; `v = (cos alt cos az, cos alt sin az, sin alt)`.
- Axis at `alt₀ = 90−ε`, `A₀ = atan2(t_e, t_n)`, `ε = hypot(t_n, t_e)`.
- Minimal rotation axis `n = (−sin A₀, cos A₀, 0)`; transform to axis frame is the rotation by **−ε** about `n` (Rodrigues): `v' = v cos ε − (n×v) sin ε + n (n·v)(1−cos ε)`; then `z' = arccos(v'_z)`, `A' = atan2(v'_y, v'_x)`. Checks: the axis itself maps to the pole; the zenith maps to `A' = A₀+180`; ε→0 gives `A' = az`.
- WCS native longitude relation in this codebase (from PC rotation + ARC conventions, confirmed by the C&G δ₀=90 special case with today's `lonpole=0`): `φ = A' + 180°`. The celestial pole (zenith) sits at `A' = A₀+180`, so `LONPOLE = φ(zenith) = A₀ (mod 360)`.

**Files (all tasks):**
- Modify: `skycam_utils/alcor.py`
- Test: `skycam_utils/tests/test_alcor_wcs.py`
- Modify: `CLAUDE.md` (Task 6)

Line numbers are pre-change positions; locate by symbol name.

---

### Task 1: Calibration schema and defaults

**Files:**
- Modify: `skycam_utils/alcor.py` — `ALCOR_CALIBRATIONS` comment (~line 45), `alcor_calibration` (~line 70), module constants (~line 95), `_format_calibration_entry` (~line 2020)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
def test_alcor_calibration_defaults_axis_tilt():
    # Shipped epochs have no axis_tilt key; the resolver must fill it.
    cal = alcor_calibration()
    assert cal["axis_tilt"] == (0.0, 0.0)
    cal = alcor_calibration(Time("2024-09-05", scale="utc"))
    assert cal["axis_tilt"] == (0.0, 0.0)


def test_format_calibration_entry_includes_axis_tilt():
    import ast
    from skycam_utils.alcor import _format_calibration_entry

    line = _format_calibration_entry(dict(
        epoch="2026-05-19", xcen=-12.0, ycen=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0), tangential_coeffs=(0.004, -0.003),
        axis_tilt=(0.31, -0.22), horizon_radius=662.0))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["axis_tilt"] == (0.31, -0.22)
    # results without the key (old callers) format with the zero default
    line = _format_calibration_entry(dict(
        epoch="2026-05-19", xcen=-12.0, ycen=9.9, rotation=0.3013,
        radial_coeffs=(1.0, 0.084, 0.0), horizon_radius=662.0))
    parsed = ast.literal_eval(line.strip().rstrip(","))
    assert parsed["axis_tilt"] == (0.0, 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "axis_tilt" -v`
Expected: 2 FAILED (`KeyError: 'axis_tilt'` / missing key in parsed dict).

- [ ] **Step 3: Implement**

In `skycam_utils/alcor.py`:

(a) Extend the `ALCOR_CALIBRATIONS` block comment — after the
`tangential_coeffs` sentence, add:

```python
# An optional "axis_tilt": (t_n, t_e) holds the optical-axis tilt from the
# zenith as components toward north and east, in DEGREES (the axis points at
# alt 90 - hypot(t_n, t_e), az atan2(t_e, t_n)); epochs without the key mean
# (0.0, 0.0). With nonzero tilt, xcen/ycen is the optical-axis pixel (the
# distortion center), not the zenith pixel.
```

(b) In `alcor_calibration`, add a second `setdefault` next to the existing
tangential one:

```python
    cal.setdefault("tangential_coeffs", (0.0, 0.0))
    cal.setdefault("axis_tilt", (0.0, 0.0))
    return cal
```

(c) Add the module default after `ALCOR_TANGENTIAL_COEFFS`:

```python
ALCOR_AXIS_TILT = _LATEST_CALIBRATION["axis_tilt"]
```

(d) Add `axis_tilt` to `_format_calibration_entry` between
`tangential_coeffs` and `horizon_radius`:

```python
def _format_calibration_entry(result):
    """Format a calibration result as a paste-ready ALCOR_CALIBRATIONS entry."""
    rc = tuple(float(c) for c in result["radial_coeffs"])
    tc = tuple(float(c) for c in result.get("tangential_coeffs", (0.0, 0.0)))
    at = tuple(float(c) for c in result.get("axis_tilt", (0.0, 0.0)))
    return (f'    {{"epoch": "{result["epoch"]}", '
            f'"xcen": {result["xcen"]:.3f}, '
            f'"ycen": {result["ycen"]:.3f}, '
            f'"rotation": {result["rotation"]:.4f}, '
            f'"radial_coeffs": {rc!r}, '
            f'"tangential_coeffs": {tc!r}, '
            f'"axis_tilt": {at!r}, '
            f'"horizon_radius": {result["horizon_radius"]:.1f}}},')
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "axis_tilt or calibration or format" -v`
Expected: all PASS (including the pre-existing calibration/format tests).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: optional axis_tilt in calibration schema

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Forward model — `_axis_frame` + `_predict_pixels` tilt support

**Files:**
- Modify: `skycam_utils/alcor.py` — new helper `_axis_frame` above `_tangential_delta` (~line 117); `_predict_pixels` (~line 145)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
def test_predict_pixels_axis_tilt_zero_is_noop():
    alt = np.array([80.0, 45.0, 10.0])
    az = np.array([15.0, 120.0, 300.0])
    bx, by = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                             radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0,
                             tangential_coeffs=(0.003, -0.002))
    tx, ty = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                             radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0,
                             tangential_coeffs=(0.003, -0.002),
                             axis_tilt=(0.0, 0.0))
    np.testing.assert_array_equal(tx, bx)
    np.testing.assert_array_equal(ty, by)


def test_predict_pixels_axis_tilt_spherical_geometry():
    """With tilt (and no pixel-space tangential terms), the pixel radius about
    (xcen, ycen) must map through the plate solution to the TRUE angular
    distance from the tilted axis, computed independently from the spherical
    cosine identity."""
    tn, te = 0.3, -0.2
    eps = np.hypot(tn, te)
    a0 = np.degrees(np.arctan2(te, tn))
    k1, k3, k5 = 1.0, 0.09, 0.02
    H = 747.0
    rng = np.random.default_rng(17)
    alt = rng.uniform(5.0, 89.5, 300)
    az = rng.uniform(0.0, 360.0, 300)
    x, y = _predict_pixels(alt, az, xcen=700.0, ycen=710.0, rotation=-1.0,
                           radial_coeffs=(k1, k3, k5), horizon_radius=H,
                           axis_tilt=(tn, te))
    rho = np.hypot(x - 700.0, y - 710.0) / H
    z_model = 90.0 * (k1 * rho + k3 * rho**3 + k5 * rho**5)
    alt0 = np.radians(90.0 - eps)
    cos_zp = (np.sin(np.radians(alt)) * np.sin(alt0)
              + np.cos(np.radians(alt)) * np.cos(alt0)
              * np.cos(np.radians(az - a0)))
    z_true = np.degrees(np.arccos(np.clip(cos_zp, -1.0, 1.0)))
    np.testing.assert_allclose(z_model, z_true, atol=1e-6)


def test_predict_pixels_axis_tilt_continuous_at_zero():
    alt = np.array([80.0, 45.0, 10.0])
    az = np.array([15.0, 120.0, 300.0])
    kw = dict(xcen=700.0, ycen=710.0, rotation=-1.0,
              radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0)
    x0, y0 = _predict_pixels(alt, az, axis_tilt=(0.0, 0.0), **kw)
    x1, y1 = _predict_pixels(alt, az, axis_tilt=(1e-9, 0.0), **kw)
    np.testing.assert_allclose(x1, x0, atol=1e-5)
    np.testing.assert_allclose(y1, y0, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "predict_pixels_axis_tilt" -v`
Expected: 3 FAILED (`TypeError: ... unexpected keyword argument 'axis_tilt'`).

- [ ] **Step 3: Implement**

In `skycam_utils/alcor.py`, insert above `_tangential_delta`:

```python
def _axis_frame(alt, az, t_n, t_e):
    """Axis-centered polar coordinates (z', A') [deg] of sky points (alt, az)
    [deg] for an optical axis tilted off the zenith.

    The axis leans eps = hypot(t_n, t_e) deg from the zenith toward azimuth
    A0 = atan2(t_e, t_n) (components toward north and east). The sky is
    rotated by the exact minimal rotation -- by eps about the horizontal axis
    at azimuth A0 + 90 -- that carries the optical axis to the pole, so
    A' -> az continuously as eps -> 0. z' is the true angular distance from
    the axis.
    """
    eps = np.radians(np.hypot(t_n, t_e))
    a0 = np.arctan2(t_e, t_n)
    alt_r = np.radians(np.asarray(alt, dtype=float))
    az_r = np.radians(np.asarray(az, dtype=float))
    # unit vectors: x toward north (az=0), y toward east (az=90), z up
    vx = np.cos(alt_r) * np.cos(az_r)
    vy = np.cos(alt_r) * np.sin(az_r)
    vz = np.sin(alt_r)
    # Rodrigues rotation by -eps about n = (-sin A0, cos A0, 0):
    # v' = v cos(eps) - (n x v) sin(eps) + n (n.v)(1 - cos(eps))
    nx, ny = -np.sin(a0), np.cos(a0)
    ndv = nx * vx + ny * vy
    c, s = np.cos(eps), np.sin(eps)
    wx = c * vx - s * (ny * vz) + (1.0 - c) * ndv * nx
    wy = c * vy - s * (-nx * vz) + (1.0 - c) * ndv * ny
    wz = c * vz - s * (nx * vy - ny * vx)
    zp = np.degrees(np.arccos(np.clip(wz, -1.0, 1.0)))
    ap = np.degrees(np.arctan2(wy, wx))
    return zp, ap
```

In `_predict_pixels`: add the keyword and route `(z', A')` through the
existing machinery. The full new body (docstring gains the tilt paragraph;
zero tilt is byte-identical math):

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
    axis_tilt=(0.0, 0.0),
):
```

with this docstring paragraph appended after the tangential one:

```
    ``axis_tilt`` (t_n, t_e) tilts the optical axis off the zenith (degrees
    toward north/east; see `_axis_frame`). The model is azimuthally symmetric
    about the AXIS: the radial inversion runs in the axis distance z' and the
    pixel angle is ``rotation - A'``. With nonzero tilt, (xcen, ycen) is the
    optical-axis pixel, not the zenith pixel.
```

and this body (replacing everything from `alt = np.asarray(...)` to the end):

```python
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    coeffs = tuple(float(c) for c in radial_coeffs)

    tn, te = (float(c) for c in axis_tilt)
    if tn != 0.0 or te != 0.0:
        zp, ap = _axis_frame(alt, az, tn, te)
    else:
        zp = 90.0 - alt
        ap = az

    rho = _invert_radial(zp, coeffs)
    r = horizon_radius * rho
    ang = np.radians(rotation - ap)
    u = r * np.sin(ang)
    v = r * np.cos(ang)

    p1, p2 = (float(c) for c in tangential_coeffs)
    if p1 != 0.0 or p2 != 0.0:
        k1 = coeffs[0]
        H = float(horizon_radius)
        # Linear-pixel target of the SIP equation t = (u,v) + D_rad + D_tan:
        # the radial displacement preserves direction, so |t| = H*z'/(90*k1).
        s = H * zp / (90.0 * k1)
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

(The only changes from the current body: the `zp`/`ap` block replaces the
inline `90.0 - alt` / `az`, and `s` uses `zp`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "predict_pixels" -v`
Expected: all PASS (3 new + all pre-existing `predict_pixels` tests,
confirming the zero-tilt path is unchanged).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: optical-axis tilt in the forward lens model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: WCS — pole offset in `build_alcor_wcs`

**Files:**
- Modify: `skycam_utils/alcor.py` — `_base_arc_wcs` (~line 990), `build_alcor_wcs` (~line 1000), `_build_alcor_wcs_cached` (~line 1030)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
def test_build_alcor_wcs_with_axis_tilt_reproduces_forward_model():
    """All model terms nonzero simultaneously: tilt + k3 + k5 + P1/P2."""
    coeffs = (1.0, 0.05, 0.10)
    tc = (0.002, -0.001)
    at = (0.3, -0.2)
    xcen, ycen, hr = 699.0, 710.0, 747.0
    wcs = build_alcor_wcs(xcen=xcen, ycen=ycen, rotation=-1.0,
                          radial_coeffs=coeffs, horizon_radius=hr,
                          tangential_coeffs=tc, axis_tilt=at)
    eps = np.hypot(*at)
    a0 = np.degrees(np.arctan2(at[1], at[0])) % 360.0
    np.testing.assert_allclose(wcs.wcs.crval, [a0, 90.0 - eps])

    rng = np.random.default_rng(19)
    alt = rng.uniform(5.0, 89.0, 100)
    az = rng.uniform(0.0, 360.0, 100)
    mx, my = _predict_pixels(alt, az, xcen=xcen, ycen=ycen, rotation=-1.0,
                             radial_coeffs=coeffs, horizon_radius=hr,
                             tangential_coeffs=tc, axis_tilt=at)
    wx, wy = wcs.world_to_pixel_values(az, alt)
    np.testing.assert_allclose(wx, mx, atol=1e-3)
    np.testing.assert_allclose(wy, my, atol=1e-3)
    waz, walt = wcs.pixel_to_world_values(mx, my)
    np.testing.assert_allclose(walt, alt, atol=1e-4)
    daz = (waz - az + 180.0) % 360.0 - 180.0   # wrap-safe angular difference
    np.testing.assert_allclose(daz, 0.0, atol=1e-3)


def test_build_alcor_wcs_zero_tilt_keeps_zenith_pole():
    wcs = build_alcor_wcs(xcen=696.0, ycen=698.0, rotation=0.4,
                          radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0,
                          axis_tilt=(0.0, 0.0))
    np.testing.assert_allclose(wcs.wcs.crval, [0.0, 90.0])
    # at zero tilt the zenith pixel IS crpix
    zx, zy = wcs.world_to_pixel_values(0.0, 90.0)
    np.testing.assert_allclose([zx, zy], np.asarray(wcs.wcs.crpix) - 1.0,
                               atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "wcs_with_axis_tilt or zero_tilt_keeps" -v`
Expected: 2 FAILED (`TypeError: ... unexpected keyword argument 'axis_tilt'`).

- [ ] **Step 3: Implement**

In `skycam_utils/alcor.py`:

(a) `_base_arc_wcs` gains the tilt and sets the pole (docstring gains one
sentence; the LONPOLE value comes from the derived relation `phi = A' + 180`
with the zenith at `A' = A0 + 180`, hence `LONPOLE = A0`):

```python
def _base_arc_wcs(xcen, ycen, rotation, k1, horizon_radius,
                  axis_tilt=(0.0, 0.0)):
    """Linear ARC WCS (no SIP) reproducing the raw forward model's linear part.

    crpix is the 1-based optical-axis pixel; the PC matrix is the pure rotation
    (det=+1) that matches ``_predict_pixels`` (the sky/sensor handedness lives in
    the ``rotation - az`` azimuth convention, encoded by the ARC longitude axis).
    A nonzero ``axis_tilt`` moves the projection pole to the tilted optical
    axis: CRVAL = (A0, 90 - eps) and LONPOLE = A0, which makes the WCS native
    frame coincide with the minimal-rotation frame of ``_axis_frame`` (native
    longitude phi = A' + 180; the celestial pole sits at A' = A0 + 180).
    """
    cdelt = 90.0 * k1 / horizon_radius
    rot = np.radians(rotation)
    c, s = np.cos(rot), np.sin(rot)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
    wcs.wcs.crpix = [xcen + 1.0, ycen + 1.0]
    wcs.wcs.cdelt = [cdelt, cdelt]
    wcs.wcs.pc = [[c, -s], [s, c]]
    tn, te = axis_tilt
    if tn != 0.0 or te != 0.0:
        eps = float(np.hypot(tn, te))
        a0 = float(np.degrees(np.arctan2(te, tn))) % 360.0
        wcs.wcs.crval = [a0, 90.0 - eps]
        wcs.wcs.lonpole = a0
    else:
        wcs.wcs.crval = [0.0, 90.0]
        wcs.wcs.lonpole = 0.0
    return wcs
```

(b) `build_alcor_wcs` signature gains `axis_tilt=ALCOR_AXIS_TILT`, the
docstring gains:

```
    ``axis_tilt`` (t_n, t_e) tilts the optical axis off the zenith. This is
    pure FITS-WCS geometry -- CRVAL moves to (A0, 90 - eps) with LONPOLE = A0
    -- so the SIP is untouched and the mapping stays exact; with nonzero tilt
    (xcen, ycen) is the optical-axis pixel, and the zenith pixel must be
    obtained via ``world_to_pixel`` of alt=90 rather than CRPIX.
```

and the return passes it through:

```python
    return _build_alcor_wcs_cached(
        float(xcen), float(ycen), float(rotation),
        tuple(float(c) for c in radial_coeffs),
        float(horizon_radius), int(sip_degree),
        tuple(float(c) for c in tangential_coeffs),
        tuple(float(c) for c in axis_tilt),
    ).deepcopy()
```

(c) `_build_alcor_wcs_cached` gains the parameter and forwards it to the base
(only these two lines change; the SIP branch deepcopies `base` and is
otherwise untouched — the early return must NOT test the tilt, since a tilted
but distortion-free model is still a plain linear ARC WCS about the moved
pole):

```python
@lru_cache(maxsize=32)
def _build_alcor_wcs_cached(xcen, ycen, rotation, radial_coeffs, horizon_radius,
                            sip_degree, tangential_coeffs=(0.0, 0.0),
                            axis_tilt=(0.0, 0.0)):
    k1, k3, k5 = radial_coeffs
    p1, p2 = tangential_coeffs
    base = _base_arc_wcs(xcen, ycen, rotation, k1, horizon_radius,
                         axis_tilt=axis_tilt)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "build_alcor_wcs" -v`
Expected: all PASS (2 new + all pre-existing, confirming zero-tilt WCSes are
unchanged). If the round-trip test fails with a systematic azimuth offset,
the LONPOLE sign/offset convention is wrong — check `LONPOLE = A0` against
`A0 + 180` and the `% 360.0` reduction before touching anything else.

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: encode axis tilt as WCS pole offset (CRVAL/LONPOLE)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Fitter — `_fit_params` always solves (t_n, t_e)

**Files:**
- Modify: `skycam_utils/alcor.py` — `_fit_params` (~line 195)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing test**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
def test_fit_params_recovers_axis_tilt():
    rng = np.random.default_rng(23)
    alt = rng.uniform(5.0, 88.0, 400)
    az = rng.uniform(0.0, 360.0, 400)
    true = dict(xcen=5.0, ycen=-4.0, rotation=0.7,
                radial_coeffs=(1.0, 0.08, 0.0),
                tangential_coeffs=(0.002, -0.001),
                axis_tilt=(0.3, -0.2))
    x, y = _predict_pixels(alt, az, **true)

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)))
    assert abs(fit["axis_tilt"][0] - 0.3) < 0.01
    assert abs(fit["axis_tilt"][1] + 0.2) < 0.01
    assert abs(fit["xcen"] - 5.0) < 0.1
    assert abs(fit["ycen"] + 4.0) < 0.1
    assert abs(fit["radial_coeffs"][1] - 0.08) < 2e-3
    assert fit["radial_coeffs"][2] == 0.0

    # fit_k5 branch also carries the tilt
    fit5 = _fit_params(alt, az, x, y,
                       init_params=dict(xcen=0.0, ycen=0.0, rotation=0.0,
                                        radial_coeffs=(1.0, 0.0, 0.0)),
                       fit_k5=True)
    assert abs(fit5["axis_tilt"][0] - 0.3) < 0.01
    assert abs(fit5["axis_tilt"][1] + 0.2) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py::test_fit_params_recovers_axis_tilt -v`
Expected: FAIL with `KeyError: 'axis_tilt'`.

- [ ] **Step 3: Implement**

In `_fit_params`:

(a) Append to the docstring's tangential paragraph:

```
    The axis-tilt components (t_n, t_e) are likewise always fit: their
    tangential signature falls off as 1/tan(z), which no other parameter can
    produce (translation is constant, rotation grows as r, Brown-Conrady as
    r**2), so the term is well-conditioned. The returned dict also carries
    axis_tilt=(t_n, t_e).
```

(b) Extend the initial vector (after the `init_p1, init_p2` line):

```python
    init_tn, init_te = init_params.get("axis_tilt", (0.0, 0.0))

    p0 = [init_params["xcen"], init_params["ycen"],
          init_params["rotation"], init_k3]
    if fit_k5:
        p0.append(init_k5)
    p0 += [init_p1, init_p2, init_tn, init_te]
    p0 = np.asarray(p0, dtype=float)
```

(c) Extend `unpack`, `residuals`, and the return:

```python
    def unpack(p):
        if fit_k5:
            xcen, ycen, rot, k3, k5, p1, p2, tn, te = p
        else:
            xcen, ycen, rot, k3, p1, p2, tn, te = p
            k5 = 0.0
        return xcen, ycen, rot, k3, k5, p1, p2, tn, te

    def residuals(p):
        xcen, ycen, rot, k3, k5, p1, p2, tn, te = unpack(p)
        x, y = _predict_pixels(alt, az, xcen=xcen, ycen=ycen, rotation=rot,
                               radial_coeffs=(1.0, k3, k5),
                               horizon_radius=horizon_radius,
                               tangential_coeffs=(p1, p2),
                               axis_tilt=(tn, te))
        return np.concatenate([x - obs_x, y - obs_y])

    result = least_squares(residuals, p0, loss="soft_l1", f_scale=3.0)
    xcen, ycen, rot, k3, k5, p1, p2, tn, te = unpack(result.x)
    return dict(xcen=float(xcen), ycen=float(ycen), rotation=float(rot),
                radial_coeffs=(1.0, float(k3), float(k5)),
                tangential_coeffs=(float(p1), float(p2)),
                axis_tilt=(float(tn), float(te)),
                horizon_radius=float(horizon_radius))
```

- [ ] **Step 4: Run all fitter tests**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "fit_params" -v`
Expected: all PASS — the new test plus the 4 pre-existing ones, now with
(t_n, t_e) free. If `stays_physical_with_mismatches` fails, that is a real
conditioning regression — stop and investigate, do not loosen the test.

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: always fit the optical-axis tilt in _fit_params

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Threading, zenith consumers, CLI info line

**Files:**
- Modify: `skycam_utils/alcor.py` — `assign_alcor_matches` (~line 295), `fit_alcor_wcs` (init ~line 545, final pools ~lines 615/625), `save_alcor_residual_plot` (~line 700), `load_alcor_fits` (~line 1400), `_load_alcor_center_column` (~line 1529), `plot_alcor_fits` zenith block (~line 1792), `fit_alcor_wcs_cli` print block (~line 2095)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py` (end-to-end mirrors the
tangential one; the zenith-lookup test pins the new WCS-based lookup):

```python
def test_fit_alcor_wcs_recovers_axis_tilt(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xcen=6.0, ycen=-5.0, rotation=0.8,
                radial_coeffs=(1.0, 0.09, 0.0),
                tangential_coeffs=(0.002, -0.001),
                axis_tilt=(0.3, -0.2))
    rng = np.random.default_rng(5)

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
    # No axis_tilt key: the night fit must default it to (0, 0).
    monkeypatch.setattr(alcor_mod, "alcor_calibration",
                        lambda time=None: {"epoch": "2024-09-05", "xcen": 0.0,
                                           "ycen": 0.0, "rotation": 0.0,
                                           "radial_coeffs": (1.0, 0.0, 0.0),
                                           "horizon_radius": 747.0})

    result = fit_alcor_wcs(tmp_path, pattern="*.fits")
    assert abs(result["axis_tilt"][0] - 0.3) < 0.01
    assert abs(result["axis_tilt"][1] + 0.2) < 0.01
    assert abs(result["xcen"] - 6.0) < 0.1
    assert abs(result["ycen"] + 5.0) < 0.1
    assert result["residual_rms"] < 0.1


def test_wcs_zenith_lookup_with_and_without_tilt():
    kw = dict(xcen=699.0, ycen=710.0, rotation=-1.0,
              radial_coeffs=(1.0, 0.09, 0.0), horizon_radius=747.0)
    # zero tilt: zenith pixel == crpix (0-based)
    w0 = build_alcor_wcs(axis_tilt=(0.0, 0.0), **kw)
    zx, zy = w0.world_to_pixel_values(0.0, 90.0)
    np.testing.assert_allclose([zx, zy], [699.0, 710.0], atol=1e-6)
    # tilted: zenith pixel moves off crpix by ~eps * dr/dz ~ 8.3 px/deg
    wt = build_alcor_wcs(axis_tilt=(0.3, -0.2), **kw)
    zx, zy = wt.world_to_pixel_values(0.0, 90.0)
    offset = np.hypot(zx - 699.0, zy - 710.0)
    eps = np.hypot(0.3, -0.2)
    assert 0.5 * eps * 747.0 / 90.0 < offset < 2.0 * eps * 747.0 / 90.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "recovers_axis_tilt or zenith_lookup" -v`
Expected: `test_fit_alcor_wcs_recovers_axis_tilt` FAILS on the RMS/tilt
assertions (the final-pool predictions ignore the tilt until threaded).
`test_wcs_zenith_lookup_with_and_without_tilt` PASSES already (it only uses
Task 3 machinery) — that is fine; it pins behavior the consumer edits below
rely on.

- [ ] **Step 3: Implement the threading**

In `skycam_utils/alcor.py`:

(a) `assign_alcor_matches` `_predict_pixels` call gains:

```python
        tangential_coeffs=tuple(params.get("tangential_coeffs", (0.0, 0.0))),
        axis_tilt=tuple(params.get("axis_tilt", (0.0, 0.0))),
```

(b) `fit_alcor_wcs` warm-start init gains:

```python
    init = dict(xcen=base["xcen"], ycen=base["ycen"],
                rotation=base["rotation"], radial_coeffs=base["radial_coeffs"],
                tangential_coeffs=base.get("tangential_coeffs", (0.0, 0.0)),
                axis_tilt=base.get("axis_tilt", (0.0, 0.0)),
                horizon_radius=base["horizon_radius"])
```

(c) both final-pool `_predict_pixels` calls in `fit_alcor_wcs` gain:

```python
                             axis_tilt=tuple(params["axis_tilt"]),
```

(d) `save_alcor_residual_plot` refined prediction gains (idealized baseline
stays zenith-centered equidistant by design):

```python
                             tangential_coeffs=tuple(
                                 params.get("tangential_coeffs", (0.0, 0.0))),
                             axis_tilt=tuple(
                                 params.get("axis_tilt", (0.0, 0.0))))
```

(e) `load_alcor_fits` WCS build gains:

```python
                              tangential_coeffs=cal["tangential_coeffs"],
                              axis_tilt=cal["axis_tilt"])
```

(f) `_load_alcor_center_column` (keogram) — replace the CRPIX read:

```python
    zx, _ = wcs.world_to_pixel_values(0.0, 90.0)
    zcol = int(round(float(zx)))                          # 0-based zenith column
```

(g) `plot_alcor_fits` — replace the CRPIX reads:

```python
    zx, zy = wcs.world_to_pixel_values(0.0, 90.0)
    xz = int(round(float(zx)))                            # 0-based zenith
    yz = int(round(float(zy)))
```

(h) `fit_alcor_wcs_cli` — before the "add this entry" print, add the
informational polar form:

```python
    tn, te = result["axis_tilt"]
    eps = float(np.hypot(tn, te))
    a0 = float(np.degrees(np.arctan2(te, tn))) % 360.0
    print(f"# axis tilt: eps={eps:.4f} deg toward az={a0:.1f} deg")
```

(i) `fit_alcor_wcs` docstring: extend the constants tuple to "(xcen, ycen,
rotation, radial_coeffs, tangential_coeffs, axis_tilt)".

- [ ] **Step 4: Run the affected test set**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "fit_alcor_wcs or residual_plot or assign_alcor or load_alcor or zenith" -v && pytest skycam_utils/tests/test_alcor.py -q`
Expected: all PASS — including the keogram tests in `test_alcor.py`, which
exercise the center-column path with zero-tilt epochs (lookup returns CRPIX
exactly).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: thread axis_tilt through fit, load, and zenith consumers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Docs + full-suite verification

**Files:**
- Modify: `CLAUDE.md` (Alcor paragraph; `fit_alcor_wcs` command block)

- [ ] **Step 1: Update CLAUDE.md**

In the Alcor bullet: extend the constants list to
`xcen`/`ycen`/`rotation`/`radial_coeffs`/`tangential_coeffs`/`axis_tilt`/`horizon_radius`
(noting `axis_tilt` is optional, `(0, 0)` when absent, in degrees toward
north/east), and after the Brown–Conrady sentence add:

```
plus a world-side optical-axis tilt (`axis_tilt`, fit always, encoded as the
WCS pole: CRVAL=(A0, 90−ε), LONPOLE=A0) — with nonzero tilt `xcen`/`ycen` is
the optical-axis pixel and the zenith must be located via the WCS (alt=90),
not CRPIX
```

In the `fit_alcor_wcs` command block, add `axis_tilt` to the printed-constants
list.

- [ ] **Step 2: Run the full test suite**

Run: `pytest`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "alcor: document the optical-axis tilt model

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Post-implementation validation (operational, user-run)

Per the spec, on the remote machine:

```bash
fit_alcor_wcs <2026-05-18-dir> --fit-k5 --residual-plot 20260518_tilt.png
fit_alcor_wcs <2024-09-04-dir> --fit-k5 --residual-plot 20240904_tilt.png
```

Success: RMS well below 1.321 px on the 2026 night; the near-zenith
tangential sinusoid and large-z phase-flip collapse; fitted eps ≈ 0.3–0.4°.
Cross-check: the 2024 night recovers the same (t_n, t_e) within uncertainty
(the camera has not moved). Agreement → bake both printed entries into
`ALCOR_CALIBRATIONS`; disagreement → the term is absorbing something else,
rethink before committing entries.
