# Alcor WCS-Driven `load_alcor_fits` Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task (this repo does NOT use subagent-driven
> development — see memory `no-subagent-development`). Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Make the WCS the single source of alcor geometry: `load_alcor_fits`
returns the raw `(3, ny, nx)` cube untouched (no transpose/trim/rotate/shift/
flipud, no bias subtraction) plus a raw-frame alt/az WCS, and every consumer
(matcher, fitter, plotter, proc, keogram) follows that frame.

**Architecture:** The raw forward model is `x = xcen − r·sin(az+rot)`,
`y = ycen − r·cos(az+rot)` with `r = horizon_radius·rho(90−alt)`. A WCS reproduces
it with `crpix=[xcen+1, ycen+1]`, `crval=[0,90]`, `cdelt=[90·k1/horizon_radius]×2`,
`lonpole=0`, and `PC=[[cos rot, −sin rot],[−sin rot, −cos rot]]` (det=−1: the
reflection replacing `flipud`, plus rotation). The isotropic radial SIP carries
over unchanged, re-centered on the zenith. Calibration epochs become raw-frame
absolutes `{xcen, ycen, rotation, radial_coeffs, horizon_radius}`.

**Tech Stack:** numpy, astropy.wcs (ARC + SIP), scipy.optimize, photutils,
pytest. All work in `skycam_utils/alcor.py` and `skycam_utils/tests/`.

**Spec:** `docs/superpowers/specs/2026-06-10-alcor-wcs-driven-load-design.md`

**Reference:** `~/MMT/skycam_data/2024-09-04` (2024 dark-frame night, for the
re-fit in Task 4) and the packaged fixture `skycam_utils/tests/test.fits.bz2`.

---

## Task 1: Raw-frame geometry core (`_predict_pixels`, `build_alcor_wcs`, `ALCOR_CALIBRATIONS`)

**Files:**
- Modify: `skycam_utils/alcor.py` — `ALCOR_CALIBRATIONS` (51-54), `alcor_calibration`
  / module constants (62-86), `_predict_pixels` (108-150), `_base_arc_wcs`
  (840-849), `build_alcor_wcs` + `_build_alcor_wcs_cached` (891-945)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Change the calibration schema and module constants**

In `ALCOR_CALIBRATIONS`, replace `xshift`/`yshift` with `xcen`/`ycen` and add
`horizon_radius`. Use a temporary placeholder geometry (Task 4 replaces it with
the real re-fit); seed `xcen`/`ycen` from the old trim center (696, 698):

```python
ALCOR_CALIBRATIONS = [
    {"epoch": "2024-09-04", "xcen": 696.0, "ycen": 698.0,
     "rotation": 0.3886, "radial_coeffs": (1.0, 0.01383, 0.0),
     "horizon_radius": 662.0},
]
```

Update the module-level convenience constants (80-86): drop `ALCOR_XSHIFT`/
`ALCOR_YSHIFT`, add `ALCOR_XCEN`/`ALCOR_YCEN`, keep `ALCOR_ROTATION`/
`ALCOR_RADIAL_COEFFS`:

```python
_LATEST_CALIBRATION = alcor_calibration()
ALCOR_ROTATION = _LATEST_CALIBRATION["rotation"]
ALCOR_XCEN = _LATEST_CALIBRATION["xcen"]
ALCOR_YCEN = _LATEST_CALIBRATION["ycen"]
ALCOR_RADIAL_COEFFS = _LATEST_CALIBRATION["radial_coeffs"]
```

`alcor_calibration(time)` (62-77) is unchanged.

- [ ] **Step 2: Write the failing round-trip test**

Replace `test_predict_pixels_idealized_reproduces_zenith_and_horizon`,
`test_predict_pixels_radial_term_changes_radius`,
`test_predict_pixels_default_coeffs_are_baked_calibration`,
`test_build_alcor_wcs_idealized_matches_equidistant`, and
`test_build_alcor_wcs_with_radial_term_reproduces_forward_model` in
`test_alcor_wcs.py` with raw-frame versions. The defining new test:

```python
def test_build_alcor_wcs_reproduces_raw_forward_model():
    xcen, ycen, rot, coeffs, hr = 696.0, 698.0, 0.4, (1.0, 0.0138, 0.0), 662.0
    wcs = build_alcor_wcs(xcen=xcen, ycen=ycen, rotation=rot,
                          radial_coeffs=coeffs, horizon_radius=hr)
    az = np.array([0.0, 90.0, 180.0, 270.0, 45.0])
    alt = np.array([85.0, 60.0, 30.0, 10.0, 0.0])
    mx, my = _predict_pixels(alt, az, xcen=xcen, ycen=ycen, rotation=rot,
                             radial_coeffs=coeffs, horizon_radius=hr)
    wx, wy = wcs.wcs_world2pix(az, alt, 0)
    np.testing.assert_allclose(wx, mx, atol=1e-3)
    np.testing.assert_allclose(wy, my, atol=1e-3)


def test_predict_pixels_zenith_maps_to_center():
    x, y = _predict_pixels(90.0, 0.0, xcen=696.0, ycen=698.0, rotation=0.0,
                           radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=662.0)
    assert abs(x - 696.0) < 1e-6
    assert abs(y - 698.0) < 1e-6


def test_predict_pixels_north_is_minus_y():
    # az=0, alt below zenith -> straight "up" the sensor = decreasing row (−y)
    x, y = _predict_pixels(80.0, 0.0, xcen=696.0, ycen=698.0, rotation=0.0,
                           radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=662.0)
    assert abs(x - 696.0) < 1e-6
    assert y < 698.0
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py::test_build_alcor_wcs_reproduces_raw_forward_model -v`
Expected: FAIL (`_predict_pixels` still takes `xshift`/`yshift`; `build_alcor_wcs`
still takes `radius`).

- [ ] **Step 4: Rewrite `_predict_pixels` to the raw frame**

Replace the body/signature (108-150). Keep `_invert_radial` (89-105) unchanged.

```python
def _predict_pixels(
    alt,
    az,
    xcen=ALCOR_XCEN,
    ycen=ALCOR_YCEN,
    rotation=0.0,
    radial_coeffs=ALCOR_RADIAL_COEFFS,
    horizon_radius=ALCOR_HORIZON_RADIUS,
):
    """Forward lens model: map altitude/azimuth (deg) to RAW-frame pixel
    coordinates (x=column, y=row, 0-based).

    The zenith sits at ``(xcen, ycen)``; ``rotation`` is the camera azimuth
    zero-point offset (deg). The lens plate solution
    ``z = 90*(k1*rho + k3*rho**3 + k5*rho**5)`` (``rho = r/horizon_radius``,
    ``z = 90 - alt``) is inverted for ``rho`` via Newton's method. The parity of
    the raw frame (north toward −y, the reflection the old ``flipud`` produced)
    is baked into the −sin/−cos signs; the matching WCS encodes the same parity
    in its PC matrix (see ``build_alcor_wcs``).
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    rho = _invert_radial(90.0 - alt, tuple(float(c) for c in radial_coeffs))
    r = horizon_radius * rho
    ang = np.radians(az + rotation)
    x = xcen - r * np.sin(ang)
    y = ycen - r * np.cos(ang)
    return x, y
```

- [ ] **Step 5: Rewrite `_base_arc_wcs` and `build_alcor_wcs`**

Replace `_base_arc_wcs` (840-849):

```python
def _base_arc_wcs(xcen, ycen, rotation, k1, horizon_radius):
    """Linear ARC WCS (no SIP) reproducing the raw forward model's linear part.

    crpix is the 1-based zenith pixel; the PC matrix carries the rotation and the
    det=−1 parity that replaces the old flipud.
    """
    cdelt = 90.0 * k1 / horizon_radius
    rot = np.radians(rotation)
    c, s = np.cos(rot), np.sin(rot)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
    wcs.wcs.crpix = [xcen + 1.0, ycen + 1.0]
    wcs.wcs.crval = [0.0, 90.0]
    wcs.wcs.cdelt = [cdelt, cdelt]
    wcs.wcs.pc = [[c, -s], [-s, -c]]
    wcs.wcs.lonpole = 0.0
    return wcs
```

Replace `build_alcor_wcs` + `_build_alcor_wcs_cached` (891-945). The SIP block is
identical math to the old one (Cartesian displacement of the radial plate
solution), only the reference pixel moves to `[xcen+1, ycen+1]` and the inverse
SIP grid spans `±horizon_radius` around it:

```python
def build_alcor_wcs(xcen=ALCOR_XCEN, ycen=ALCOR_YCEN, rotation=ALCOR_ROTATION,
                    radial_coeffs=ALCOR_RADIAL_COEFFS,
                    horizon_radius=ALCOR_HORIZON_RADIUS, sip_degree=5):
    """Build the raw-frame alt/az ARC WCS for the alcor sensor.

    The zenith is at pixel ``(xcen, ycen)``; ``rotation`` and the frame parity are
    in the PC matrix; the radial ``k3``/``k5`` distortion is an exact analytic SIP
    centered on the zenith. Cached on its hashable args; returns a fresh copy.
    """
    return _build_alcor_wcs_cached(
        float(xcen), float(ycen), float(rotation),
        tuple(float(c) for c in radial_coeffs),
        float(horizon_radius), int(sip_degree),
    ).deepcopy()


@lru_cache(maxsize=32)
def _build_alcor_wcs_cached(xcen, ycen, rotation, radial_coeffs, horizon_radius,
                            sip_degree):
    k1, k3, k5 = radial_coeffs
    base = _base_arc_wcs(xcen, ycen, rotation, k1, horizon_radius)
    if abs(k3) < 1e-12 and abs(k5) < 1e-12:
        return base

    H = float(horizon_radius)
    c3 = k3 / (k1 * H**2)
    c5 = k5 / (k1 * H**4)
    a = np.zeros((sip_degree + 1, sip_degree + 1))
    b = np.zeros((sip_degree + 1, sip_degree + 1))
    a[3, 0] = c3; a[1, 2] = c3
    a[5, 0] = c5; a[3, 2] = 2 * c5; a[1, 4] = c5
    b[0, 3] = c3; b[2, 1] = c3
    b[0, 5] = c5; b[2, 3] = 2 * c5; b[4, 1] = c5
    ap, bp = _fit_sip_inverse(a, b, int(round(H)), sip_degree)

    wcs = base.deepcopy()
    wcs.wcs.ctype = ["RA---ARC-SIP", "DEC--ARC-SIP"]
    wcs.sip = Sip(a, b, ap, bp, [xcen + 1.0, ycen + 1.0])
    return wcs
```

`_fit_sip_inverse` (864-888) and `_sip_poly_eval` (852-861) are unchanged (the
first arg is the grid half-width, now passed `round(horizon_radius)`).

- [ ] **Step 6: Run the round-trip + parity tests**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "predict_pixels or build_alcor_wcs" -v`
Expected: PASS. **If `test_build_alcor_wcs_reproduces_raw_forward_model` fails on
sign**, flip the PC off-diagonal/diagonal signs (the reflection axis) until the
round-trip closes — the derivation is `PC=[[cos,−sin],[−sin,−cos]]`, but confirm
empirically.

- [ ] **Step 7: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: raw-frame _predict_pixels + build_alcor_wcs + epoch schema"
```

---

## Task 2: Thin `load_alcor_fits` + luminance-aware `detect_alcor_stars`

**Files:**
- Modify: `skycam_utils/alcor.py` — `load_alcor_fits` (1233-1370),
  `detect_alcor_stars` (948-997)
- Test: `skycam_utils/tests/test_alcor.py`, `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Write the failing tests**

In `test_alcor.py`, replace `test_load_alcor_fits_returns_centered_rgb_image` and
`test_load_alcor_fits_wcs_maps_zenith_and_horizon` with raw-frame versions:

```python
def test_load_alcor_fits_returns_raw_cube_wcs_mask():
    cube, wcs, mask = load_alcor_fits(TEST_FITS)
    assert cube.ndim == 3 and cube.shape[0] == 3          # (3, ny, nx), no transpose
    assert cube.dtype == np.float32
    assert wcs.wcs.ctype[0].startswith("RA---ARC")
    assert mask is None or mask.shape == cube.shape       # native-orientation mask

def test_load_alcor_fits_no_bias_subtraction():
    cube, _, _ = load_alcor_fits(TEST_FITS, badpix=None)
    with fits.open(TEST_FITS) as hdul:
        raw = np.asarray(hdul[0].data, dtype=np.float32)
    np.testing.assert_array_equal(cube, raw)              # untouched: no −2000, no clip

def test_load_alcor_fits_accepts_explicit_wcs():
    w = build_alcor_wcs(xcen=10.0, ycen=20.0, rotation=0.0,
                        radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=30.0)
    _, wcs, _ = load_alcor_fits(TEST_FITS, wcs=w)
    assert list(wcs.wcs.crpix) == [11.0, 21.0]
```

In `test_alcor_wcs.py`, replace `test_detect_alcor_stars_*` image setup to feed a
`(3, ny, nx)` cube and add:

```python
def test_detect_alcor_stars_accepts_cube_and_2d():
    img = np.zeros((40, 40))
    img[20, 25] = 500.0
    cube = np.stack([img, img, img], axis=0)              # (3, ny, nx)
    det_cube = detect_alcor_stars(cube, fwhm=2.0, threshold_sigma=3.0)
    det_2d = detect_alcor_stars(img, fwhm=2.0, threshold_sigma=3.0)
    assert len(det_cube) >= 1 and len(det_2d) >= 1
    assert abs(det_cube["xcentroid"][0] - 25) < 1.5
    assert abs(det_cube["ycentroid"][0] - 20) < 1.5
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor.py::test_load_alcor_fits_returns_raw_cube_wcs_mask -v`
Expected: FAIL (current `load_alcor_fits` returns a 2-tuple and `(ny, nx, 3)`).

- [ ] **Step 3: Rewrite `detect_alcor_stars` for the luminance rule**

Replace lines 948-997's docstring intro and the luminance line. Change the
signature docstring to "2D frame or (3, ny, nx) cube" and replace line 975:

```python
    arr = np.asarray(im, dtype=float)
    if arr.ndim == 3:
        lum = arr.mean(axis=0)            # (3, ny, nx) R,G,B -> luminance
    elif arr.ndim == 2:
        lum = arr
    else:
        raise ValueError(f"expected a 2D frame or (3, ny, nx) cube, got {arr.shape}")
```

The rest of the function (sigma-clipped stats, DAOStarFinder, column renaming,
`max_detections` cap) is unchanged.

- [ ] **Step 4: Rewrite `load_alcor_fits`**

Replace the whole function (1233-1370):

```python
def load_alcor_fits(filename, wcs=None, badpix="repair", masks_dir=None):
    """Load an alcor OMEA 8C FITS file and return ``(cube, wcs, mask)``.

    The raw ``(3, ny, nx)`` RGB cube is returned in native FITS orientation,
    unmodified except for optional bad-pixel repair: no transpose, trim, rotate,
    shift, or flipud, and no bias subtraction. Geometry lives entirely in the
    returned WCS.

    Parameters
    ----------
    filename : str or Path
        FITS file. Compressed (.gz, .bz2) inputs are supported.
    wcs : `astropy.wcs.WCS` or None (default=None)
        Geometry WCS. When None, the calibration epoch nearest the frame's time
        is resolved and ``build_alcor_wcs`` constructs the raw-frame ARC WCS.
    badpix : str or None or path or ndarray (default="repair")
        "repair" repairs flagged pixels per channel with their local 5x5 median;
        None leaves the cube untouched (the mask is still resolved and returned);
        a path or (3, ny, nx) bool array uses that mask explicitly (and repairs).
    masks_dir : str or None (default=None)
        Override the bad-pixel masks directory (else $ALCOR_BADPIX_DIR, else the
        packaged data/badpix/).

    Returns
    -------
    cube : ndarray
        Raw ``(3, ny, nx)`` float32 cube (channels 0,1,2 = R,G,B).
    wcs : `astropy.wcs.WCS`
        Raw-frame ARC WCS mapping pixel (x, y) <-> (azimuth, altitude).
    mask : ndarray or None
        ``(3, ny, nx)`` bool bad-pixel mask in native orientation, or None when
        no mask is available.
    """
    with fits.open(filename) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float32)   # (3, ny, nx)

    # --- resolve the bad-pixel mask (explicit, or nearest-date) ---
    mask = None
    if isinstance(badpix, np.ndarray):
        cand = badpix.astype(bool)
    elif isinstance(badpix, Path) or (isinstance(badpix, str) and badpix != "repair"):
        cand = np.asarray(fits.getdata(badpix)).astype(bool)
    else:                                                   # "repair" or None
        cand = None
        try:
            dt = _filename_ut_datetime(filename)
            t = (Time(dt) if dt is not None
                 else Time(_read_frame_date(filename), format="isot", scale="utc"))
            cand, _ = load_alcor_badpix_mask(t, masks_dir=masks_dir)
        except (KeyError, OSError, ValueError):
            cand = None
    if cand is not None and cand.shape == cube.shape:
        mask = cand

    if badpix is not None and mask is not None:
        cube = _apply_badpix_repair(cube, mask)

    if wcs is None:
        cal = alcor_calibration(_alcor_frame_time(filename))
        wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                              rotation=cal["rotation"],
                              radial_coeffs=cal["radial_coeffs"],
                              horizon_radius=cal["horizon_radius"])

    return cube, wcs, mask
```

Add a small time helper near `_alcor_frame_calibration` (782-796), reused by
`load_alcor_fits` and the keogram/proc paths:

```python
def _alcor_frame_time(filename):
    """Best-effort observation Time for a frame: filename timestamp, then DATE
    header, then None (so callers fall back to the latest epoch)."""
    dt = _filename_ut_datetime(filename)
    if dt is not None:
        return Time(dt)
    try:
        return Time(_read_frame_date(filename), format="isot", scale="utc")
    except (KeyError, OSError, ValueError):
        return None
```

- [ ] **Step 5: Run the load/detect tests**

Run: `pytest skycam_utils/tests/test_alcor.py -k load_alcor_fits skycam_utils/tests/test_alcor_wcs.py -k detect_alcor_stars -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: thin WCS-driven load_alcor_fits; luminance-aware detection"
```

---

## Task 3: Matcher + fitter to the raw frame

**Files:**
- Modify: `skycam_utils/alcor.py` — `_fit_params` (153-207), `assign_alcor_matches`
  (210-369, only the `params` keys and the `_predict_pixels` call), `_detect_alcor_frame`
  (381-401), `fit_alcor_wcs` (404-570), `save_alcor_residual_plot` (573-733)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

The matcher algorithm is unchanged; only the geometry parametrization changes from
`xshift`/`yshift` to `xcen`/`ycen` (+ `horizon_radius` threaded through). Apply this
substitution everywhere in these functions:

| Old | New |
|-----|-----|
| `params["xshift"]`, `params["yshift"]` | `params["xcen"]`, `params["ycen"]` |
| `_predict_pixels(..., xshift=, yshift=, radius=, horizon_radius=)` | `_predict_pixels(..., xcen=, ycen=, horizon_radius=)` (no `radius`) |
| `dict(xshift=, yshift=, rotation=, radial_coeffs=)` | `dict(xcen=, ycen=, rotation=, radial_coeffs=)` |

- [ ] **Step 1: Rewrite `_fit_params`**

The fit variables become `(xcen, ycen, rotation, k3[, k5])`. Replace `init_params`
reads (`init_params["xshift"]` → `["xcen"]`, etc.), the `p0` arrays, the `residuals`
closures (call `_predict_pixels(alt, az, xcen=, ycen=, rotation=rot,
radial_coeffs=, horizon_radius=horizon_radius)`), and both return dicts
(`dict(xcen=..., ycen=..., rotation=..., radial_coeffs=...)`). Drop the `radius`
parameter from the signature; keep `horizon_radius=ALCOR_HORIZON_RADIUS` and
`fit_k5`.

- [ ] **Step 2: Rewrite `assign_alcor_matches` geometry references**

Update the docstring (`xshift`/`yshift` → `xcen`/`ycen`), drop `radius` from the
signature, and change the single `_predict_pixels` call (249-253):

```python
    px, py = _predict_pixels(
        cat["Alt"], cat["Az"], xcen=params["xcen"], ycen=params["ycen"],
        rotation=params["rotation"], radial_coeffs=tuple(params["radial_coeffs"]),
        horizon_radius=params.get("horizon_radius", ALCOR_HORIZON_RADIUS),
    )
```

- [ ] **Step 3: Rewrite `_detect_alcor_frame` and `fit_alcor_wcs`**

`_detect_alcor_frame` (381-401): load the raw cube and detect on it directly.

```python
    index, filename, vmag_limit, min_alt, fwhm, threshold_sigma, max_detections = task
    filename = Path(filename)
    time = _frame_time(filename)
    cube, _, _ = load_alcor_fits(filename, badpix=None)
    cat = alcor_reference_altaz(time, vmag_limit=vmag_limit, min_alt=min_alt)
    det = detect_alcor_stars(cube, fwhm=fwhm, threshold_sigma=threshold_sigma,
                             max_detections=max_detections)
```

`fit_alcor_wcs` (404-570): change `init`, the final-pool `_predict_pixels` calls
(549, 558), and the printed/returned dict to the new keys. Seed `init` from the
nearest epoch:

```python
    init = dict(xcen=base["xcen"], ycen=base["ycen"],
                rotation=base["rotation"], radial_coeffs=base["radial_coeffs"])
```

Both diagnostic `_predict_pixels` calls become
`_predict_pixels(alt, az, xcen=params["xcen"], ycen=params["ycen"],
rotation=params["rotation"], radial_coeffs=tuple(params["radial_coeffs"]),
horizon_radius=base["horizon_radius"])`. Thread `horizon_radius=base["horizon_radius"]`
into the returned dict so the printed epoch is complete:

```python
    return {
        **params,
        "horizon_radius": base["horizon_radius"],
        "epoch": epoch,
        ...
    }
```

Update `_format_calibration_entry` (alcor.py:2015-2022) — the helper
`fit_alcor_wcs_cli` calls to print the paste-ready dict — to emit `xcen`, `ycen`,
`rotation`, `radial_coeffs`, `horizon_radius` (replacing `xshift`/`yshift`):

```python
def _format_calibration_entry(result):
    """Format a calibration result as a paste-ready ALCOR_CALIBRATIONS entry."""
    rc = tuple(float(c) for c in result["radial_coeffs"])
    return (f'    {{"epoch": "{result["epoch"]}", '
            f'"xcen": {result["xcen"]:.3f}, '
            f'"ycen": {result["ycen"]:.3f}, '
            f'"rotation": {result["rotation"]:.4f}, '
            f'"radial_coeffs": {rc!r}, '
            f'"horizon_radius": {result["horizon_radius"]:.1f}}},')
```

- [ ] **Step 4: Rewrite `save_alcor_residual_plot`**

Replace its internal `_predict_pixels(..., xshift=params["xshift"], ...)` calls and
any `params["xshift"]`/`["yshift"]` references with the `xcen`/`ycen` equivalents
(threading `horizon_radius`). The plot axes/labels stay; only the geometry source
changes.

- [ ] **Step 5: Migrate the matcher/fitter tests**

In `test_alcor_wcs.py`, every `dict(xshift=A, yshift=B, rotation=R, radial_coeffs=C)`
becomes `dict(xcen=696.0+A, ycen=698.0+B, rotation=R, radial_coeffs=C,
horizon_radius=662.0)` and every `_predict_pixels(..., **params)` keeps working
through the new kwargs. The synthetic-frame fits (`test_fit_alcor_wcs_*`) assert on
`result["xcen"]`/`["ycen"]` instead of `xshift`/`yshift`; the monkeypatched
`alcor_calibration` lambdas return the new schema (add `xcen`, `ycen`,
`horizon_radius`; drop `xshift`/`yshift`). The `_fit_params` recovery tests assert
the recovered `xcen`/`ycen` match the injected values.

- [ ] **Step 6: Run the matcher/fitter tests**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -v`
Expected: PASS (whole file).

- [ ] **Step 7: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "alcor: matcher + fitter operate in the raw frame"
```

---

## Task 4: Re-fit and replace the 2024-09-04 epoch

**Files:**
- Modify: `skycam_utils/alcor.py` — `ALCOR_CALIBRATIONS` (51-54)

- [ ] **Step 1: Run the fitter on the 2024 night**

```bash
fit_alcor_wcs ~/MMT/skycam_data/2024-09-04 --fit-k5 --residual-plot /tmp/alcor_2024_resid.png
```

Expected: prints an `ALCOR_CALIBRATIONS` epoch dict in the new schema (`xcen`,
`ycen`, `rotation`, `radial_coeffs`, `horizon_radius`) plus `n_matched`,
`residual_rms` (expect a few px or better), and `matched_fraction`. Inspect
`/tmp/alcor_2024_resid.png` for a flat residual field.

- [ ] **Step 2: Paste the fitted dict into `ALCOR_CALIBRATIONS`**

Replace the Task-1 placeholder epoch with the printed dict. Keep `epoch`
`"2024-09-04"`.

- [ ] **Step 3: Verify the real epoch round-trips against the fixture**

Run: `pytest skycam_utils/tests/test_alcor.py::test_load_alcor_fits_returns_raw_cube_wcs_mask -v`
Expected: PASS. Then sanity-check a bright star lands correctly:

```bash
python -c "
from skycam_utils.alcor import load_alcor_fits, detect_alcor_stars, alcor_reference_altaz, _frame_time
import numpy as np
f='skycam_utils/tests/test.fits.bz2'
cube,wcs,_=load_alcor_fits(f); det=detect_alcor_stars(cube)
cat=alcor_reference_altaz(_frame_time(f), vmag_limit=2.0, min_alt=30.0)
px,py=wcs.wcs_world2pix(np.asarray(cat['Az']),np.asarray(cat['Alt']),0)
print('catalog stars near a detection within 5px:',
      sum(np.min(np.hypot(det['xcentroid']-x,det['ycentroid']-y))<5 for x,y in zip(px,py)),'of',len(cat))
"
```

Expected: most bright catalog stars sit within ~5 px of a detection.

- [ ] **Step 4: Commit**

```bash
git add skycam_utils/alcor.py
git commit -m "alcor: re-fit 2024-09-04 calibration epoch in raw-frame schema"
```

---

## Task 5: Consumers — `plot_alcor_fits`, `alcor_proc_fits`, keogram, CLIs

**Files:**
- Modify: `skycam_utils/alcor.py` — `alcor_proc_fits` (1373-1410),
  `_load_alcor_center_column` (1484-1492), `plot_alcor_fits` (1714-1791),
  `alcor_proc_fits_cli` (1794-1825), `alcor_keogram_cli` (1828-1919),
  `plot_alcor_fits_cli` (1959-2012)
- Test: `skycam_utils/tests/test_alcor.py`

- [ ] **Step 1: Rewrite `alcor_proc_fits`**

It now writes the raw cube + WCS header directly (no flipud/transpose). Replace
1397 and 1407-1409:

```python
    cube, wcs, _ = load_alcor_fits(filename, **kwargs)
    ...
    hdu = fits.PrimaryHDU(data=cube.astype(np.float32),
                          header=wcs.to_header(relax=True))
    hdu.writeto(output_file, overwrite=overwrite)
```

Remove the `**kwargs` docstring references to `rotation, xcen, ...` (it now
forwards only `wcs`/`badpix`/`masks_dir`).

- [ ] **Step 2: Rewrite `_load_alcor_center_column` for the zenith column**

The keogram strip is the meridian column through the WCS zenith. Replace
1490-1492:

```python
    cube, wcs, _ = load_alcor_fits(filename, **kwargs)
    zcol = int(round(wcs.wcs.crpix[0] - 1.0))             # 0-based zenith column
    return index, timestamp, cube[:, :, zcol].T, filename.name  # (ny, 3) RGB strip
```

(`cube[:, :, zcol]` is `(3, ny)`; `.T` -> `(ny, 3)` to match the channel-last
strip the keogram stacker/plotter expects.)

- [ ] **Step 3: Rewrite `plot_alcor_fits`**

Do the presentation transform in the plotter. Replace the signature's geometry
params with `radius=680` (display crop half-width) and keep display flags; load
raw, build the display RGB, crop around the WCS zenith, render `origin="lower"`.

```python
def plot_alcor_fits(filename, outimage=None, outfig=None, radius=680,
                    powerstretch=0.75, contrast=0.35, gscale=0.7, bscale=1.7,
                    figsize=12):
    cube, wcs, _ = load_alcor_fits(filename)
    rgb = np.transpose(cube, (1, 2, 0)).astype(float)     # (ny, nx, 3)
    rgb[:, :, 1] *= gscale
    rgb[:, :, 2] *= bscale
    stretch = viz.PowerStretch(powerstretch) + viz.ZScaleInterval(contrast=contrast)
    rgb = stretch(rgb)

    xz = int(round(wcs.wcs.crpix[0] - 1.0))               # 0-based zenith
    yz = int(round(wcs.wcs.crpix[1] - 1.0))
    ny, nx = rgb.shape[:2]
    yl, yu = max(0, yz - radius), min(ny, yz + radius)
    xl, xu = max(0, xz - radius), min(nx, xz + radius)
    crop = rgb[yl:yu, xl:xu, :]
    cx, cy = xz - xl, yz - yl                              # zenith in crop coords

    if outimage is not None:
        plt.imsave(outimage, np.flipud(crop))             # imsave is origin-upper

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    circle = Circle((cx, cy), radius, facecolor='none', edgecolor=(0, 0, 0),
                    linewidth=1, alpha=0.5)
    ax.add_patch(circle)
    ax.axis("off")
    im_plot = ax.imshow(crop, origin="lower")
    im_plot.set_clip_path(circle)

    pax = fig.add_subplot(111, polar=True, label='polar')
    pax.set_facecolor("None")
    pax.set_theta_zero_location("N")
    tick_alts = np.array([75, 60, 45, 30, 15])
    px, py = wcs.wcs_world2pix(np.zeros_like(tick_alts, dtype=float),
                               tick_alts.astype(float), 0)
    yticks = np.hypot(px - xz, py - yz) / radius
    ylabels = [f" {a}°" for a in tick_alts]
    pax.set_yticks(yticks, labels=ylabels, color="white", alpha=0.5, fontsize=16)
    pax.set_rlabel_position(90)
    pax.tick_params(grid_alpha=0.5)
    pax.tick_params(axis='x', labelsize=16, labelcolor='silver', pad=10)

    if outfig is not None:
        plt.savefig(outfig, transparent=True, bbox_inches='tight', pad_inches=0)
    return fig
```

- [ ] **Step 4: Drop geometry flags from the three CLIs**

In `alcor_proc_fits_cli`, `alcor_keogram_cli`, and `plot_alcor_fits_cli`, remove
the `--rotation`, `--xcen`, `--ycen`, `--horizon-radius` arguments and the
`--radius` argument **except** in `plot_alcor_fits_cli` (keep `--radius` there as
the display crop). Remove those names from the corresponding call sites
(`alcor_proc_fits(...)`, `alcor_keogram(...)`, `plot_alcor_fits(...)`). Keep all
display flags (`--powerstretch`, `--contrast`, `--gscale`, `--bscale`,
`--figsize`, `--dpi`). Update the `alcor_proc_fits` CLI description (no longer
"zenith-centered, north-up").

- [ ] **Step 5: Migrate the consumer tests**

In `test_alcor.py`: `test_alcor_proc_fits_writes_processed_cube_and_header` asserts
the written data is `(3, ny, nx)` equal to the loaded `cube` and the header WCS
round-trips; `test_alcor_keogram_*` call sites that pass `radius=32,
horizon_radius=30` drop those kwargs (or pass an explicit `wcs=`); the keogram
strip height is now the full `ny`. `test_plot_alcor_fits_writes_outputs_and_returns_figure`
drops geometry kwargs and just checks the files/figure are produced. Update the
`load_alcor_fits(TEST_FITS, radius=32, horizon_radius=30)` call at test_alcor.py:107
to `load_alcor_fits(TEST_FITS)` and adjust the assertion to the raw shape.

- [ ] **Step 6: Run the full suite**

Run: `pytest skycam_utils/tests/ -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor.py
git commit -m "alcor: plotter/proc/keogram + CLIs follow the raw WCS frame"
```

---

## Task 6: Badpix tests + docs

**Files:**
- Modify: `skycam_utils/tests/test_alcor_badpix.py`, `CLAUDE.md`

- [ ] **Step 1: Migrate `test_alcor_badpix.py` to the new signature**

The `load_alcor_fits` integration tests there use `return_mask=True` /
`badpix='repair'` and expect `(im, mask, wcs)` in the resampled frame. Rewrite to
the 3-tuple `(cube, wcs, mask)` in raw orientation: `badpix='repair'` removes a
known hot pixel from `cube`; `badpix=None` leaves it but still returns a non-None
`mask` (when one resolves); the mask is `(3, ny, nx)` aligned to `cube` (no
geometric transform). Drop any assertions about trim/flip alignment.

- [ ] **Step 2: Run the badpix tests**

Run: `pytest skycam_utils/tests/test_alcor_badpix.py -v`
Expected: PASS.

- [ ] **Step 3: Update CLAUDE.md**

Rewrite the alcor paragraph (the `load_alcor_fits` description) to: returns the
raw `(3, ny, nx)` cube untouched (no transpose/trim/rotate/shift/flipud, no bias
subtraction) plus a raw-frame ARC WCS that carries zenith offset (CRPIX),
rotation+parity (PC), and radial distortion (SIP); the WCS is the single source of
geometry, resolved from the time-indexed `ALCOR_CALIBRATIONS` (now `xcen`/`ycen`/
`rotation`/`radial_coeffs`/`horizon_radius`) or passed explicitly; visualization
routines do their own crop and use `origin="lower"`. Update `alcor_proc_fits` note
(writes the raw cube + WCS, no flipud). Update the `fit_alcor_wcs` CLI note (prints
the new-schema dict). Remove the `--rotation/--xcen/...` geometry flags from the
documented `alcor_proc_fits`/`plot_alcor_fits` usage.

- [ ] **Step 4: Final full-suite run + commit**

Run: `pytest skycam_utils/tests/ -v`
Expected: PASS.

```bash
git add skycam_utils/tests/test_alcor_badpix.py CLAUDE.md
git commit -m "alcor: migrate badpix tests + document WCS-driven load"
```

---

## Self-review notes

- **Spec coverage:** §1 load (T2), §2 WCS+schema (T1), §3 model/matcher/fitter+refit
  (T3,T4), §4 plot/proc (T5), §5 CLIs+docs (T5,T6), §6 tests (interleaved + T6).
  Keogram (`_load_alcor_center_column`) was not in the spec's §-list but is a
  consumer of the old centered frame — covered in T5.
- **No bias subtraction / always 3-tuple / luminance=mean:** T2.
- **Parity risk:** T1 Step 6 verifies the PC signs against the round-trip rather
  than trusting the derivation; flip if needed.
- **Calibration realism:** T1 ships a placeholder epoch so the suite runs; T4
  replaces it with the real re-fit before anything user-facing depends on it.
