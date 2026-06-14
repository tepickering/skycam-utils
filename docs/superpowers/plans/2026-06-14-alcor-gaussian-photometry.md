# Alcor Gaussian-fit star photometry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `gaussian=True` path to `alcor_star_photometry` that recovers bright-star flux defeated by CMOS non-linearity, by pinning PSF shape from a luminance fit, masking the non-linear core, and integrating a constrained Gaussian fit to the linear wings.

**Architecture:** One method switch inside `alcor_star_photometry` (flag `gaussian`, CLI `--gaussian`). The catalog projection, bias subtraction, Sun-rejection, CSV writing, and check-plot stay shared; only the per-star measurement branches into a new helper `_gaussian_psf_photometry`. The CSV schema gains one column (`fwhm`), so `collect_alcor_photometry` and downstream scripts are unaffected.

**Tech Stack:** Python, numpy, pandas, `scipy.optimize.least_squares` (already imported in `alcor.py`), pytest.

**Spec:** `docs/superpowers/specs/2026-06-14-alcor-gaussian-photometry-design.md`

**Conventions:** Commit directly to `main` (this repo uses no feature branches). End each commit message with the trailer:
```
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```
All paths are relative to the repo root `/Users/tim/MMT/skycam_utils`. The package source lives in `skycam_utils/alcor.py`; tests in `skycam_utils/tests/test_alcor.py`. Run tests with `pytest`.

---

## File structure

- **Modify** `skycam_utils/alcor.py`:
  - Add module constant `ALCOR_NONLINEAR_THRESHOLD = 15000` next to `ALCOR_SATURATION` (line 44).
  - Extract `_annulus_background(...)` from `_aperture_annulus_photometry` (around line 1605) and have the latter call it.
  - Add `_gaussian_channel_amplitude(...)` and `_gaussian_psf_photometry(...)` helpers (after the aperture helpers, before `_default_alcor_photometry_output` ~line 1655).
  - Add `gaussian` and `mask_threshold` parameters to `alcor_star_photometry` (line 1691) and branch the per-star loop; add `fwhm` to the column list and row dict.
  - Add `--gaussian` and `--mask-threshold` to `alcor_star_photometry_cli` (line 2598) and pass them through; note the dual role of `--aperture-radius` in its help.
- **Modify** `skycam_utils/tests/test_alcor.py`: add `import sys`; import the new helpers and the CLI; add the new tests.
- **Modify** `CLAUDE.md`: document the Gaussian path and the new flags/column.

---

## Task 1: Extract `_annulus_background` (pure refactor, no behavior change)

**Files:**
- Modify: `skycam_utils/alcor.py:1605-1630` (`_aperture_annulus_photometry`)
- Test: `skycam_utils/tests/test_alcor.py`

- [ ] **Step 1: Write the failing test**

Add to `skycam_utils/tests/test_alcor.py`. First extend the import block (currently lines 19-37) to include `_annulus_background`:

```python
from skycam_utils.alcor import (
    _annulus_background,
    _aperture_annulus_photometry,
    _aperture_saturated,
    _alcor_display_rgb,
    _corner_bias,
    _timestamp_edges,
    alcor_keogram,
    alcor_proc_fits,
    alcor_star_photometry,
    build_alcor_wcs,
    load_alcor_keogram_fits,
    load_alcor_fits,
    collect_alcor_photometry,
    lookup_sloan_photometry,
    plot_alcor_keogram_fits,
    plot_alcor_fits,
    save_alcor_keogram_fits,
    save_alcor_keogram_plot,
)
```

Then add this test (place it right after `test_aperture_annulus_photometry_subtracts_local_background`):

```python
def test_annulus_background_is_median_of_annulus():
    yy, xx = np.mgrid[0:21, 0:21]
    image = np.full((21, 21), 5.0)
    rr = np.hypot(xx - 10.0, yy - 10.0)
    annulus = (rr > 4.0) & (rr <= 6.0)   # inner = ar+1 = 4, outer = 4+2 = 6
    image[annulus] = 5.0
    image[10, 16] = 500.0                # annulus outlier, killed by the median

    bg = _annulus_background(image, 10.0, 10.0, aperture_radius=3.0,
                             annulus_width=2.0)

    assert bg == 5.0


def test_annulus_background_returns_nan_when_off_image():
    image = np.zeros((10, 10))
    assert np.isnan(_annulus_background(image, -50.0, -50.0,
                                        aperture_radius=3.0, annulus_width=2.0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor.py::test_annulus_background_is_median_of_annulus -v`
Expected: FAIL — `ImportError: cannot import name '_annulus_background'`.

- [ ] **Step 3: Implement the extraction**

In `skycam_utils/alcor.py`, replace the whole body of `_aperture_annulus_photometry` (lines 1605-1630) with a new `_annulus_background` followed by a refactored `_aperture_annulus_photometry`:

```python
def _annulus_background(image, xcen, ycen, aperture_radius, annulus_width):
    """
    Median background in the circular annulus around ``(xcen, ycen)``.

    The annulus runs from ``aperture_radius + 1`` to
    ``aperture_radius + 1 + annulus_width``. Returns NaN when the annulus falls
    entirely outside the image.
    """
    ny, nx = image.shape
    annulus_inner = aperture_radius + 1.0
    outer = annulus_inner + annulus_width
    x0 = max(0, int(np.floor(xcen - outer)))
    x1 = min(nx, int(np.ceil(xcen + outer)) + 1)
    y0 = max(0, int(np.floor(ycen - outer)))
    y1 = min(ny, int(np.ceil(ycen + outer)) + 1)
    if x0 >= x1 or y0 >= y1:
        return np.nan
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - xcen, yy - ycen)
    annulus = (rr > annulus_inner) & (rr <= outer)
    if not annulus.any():
        return np.nan
    return float(np.median(image[y0:y1, x0:x1][annulus]))


def _aperture_annulus_photometry(image, xcen, ycen, aperture_radius,
                                 annulus_width):
    """
    Circular aperture flux with a local median annulus background.
    """
    background = _annulus_background(image, xcen, ycen, aperture_radius,
                                     annulus_width)
    if not np.isfinite(background):
        return np.nan, np.nan
    ny, nx = image.shape
    x0 = max(0, int(np.floor(xcen - aperture_radius)))
    x1 = min(nx, int(np.ceil(xcen + aperture_radius)) + 1)
    y0 = max(0, int(np.floor(ycen - aperture_radius)))
    y1 = min(ny, int(np.ceil(ycen + aperture_radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return np.nan, np.nan
    yy, xx = np.mgrid[y0:y1, x0:x1]
    aperture = np.hypot(xx - xcen, yy - ycen) <= aperture_radius
    if not aperture.any():
        return np.nan, np.nan
    flux = float(np.sum(image[y0:y1, x0:x1][aperture] - background))
    return flux, background
```

- [ ] **Step 4: Run the new and existing aperture tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor.py -k "annulus or aperture" -v`
Expected: PASS — including the pre-existing `test_aperture_annulus_photometry_subtracts_local_background` (guards the refactor).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor.py
git commit -m "$(cat <<'EOF'
alcor: extract _annulus_background helper from aperture photometry

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Gaussian fitting helpers

**Files:**
- Modify: `skycam_utils/alcor.py:44` (constant), and add helpers after `_aperture_saturated` (~line 1653, before `_default_alcor_photometry_output`)
- Test: `skycam_utils/tests/test_alcor.py`

- [ ] **Step 1: Write the failing test for the channel-amplitude helper**

Add the import to the test file's `from skycam_utils.alcor import (...)` block — add these two names (keep alphabetical-ish grouping with the other privates):

```python
    _gaussian_channel_amplitude,
    _gaussian_psf_photometry,
```

Also add the constant import; add this standalone import line below the big import block:

```python
from skycam_utils.alcor import ALCOR_NONLINEAR_THRESHOLD
```

Then add this test (after `test_annulus_background_returns_nan_when_off_image`):

```python
def test_nonlinear_threshold_default_is_15000():
    assert ALCOR_NONLINEAR_THRESHOLD == 15000


def test_gaussian_channel_amplitude_recovers_known_amplitude():
    yy, xx = np.mgrid[0:11, 0:11]
    sigma = 1.5
    profile = np.exp(-((xx - 5.0) ** 2 + (yy - 5.0) ** 2) / (2.0 * sigma ** 2))
    amp_true = 1234.0
    background = 50.0
    data = amp_true * profile + background
    fit_mask = np.hypot(xx - 5.0, yy - 5.0) <= 4.0

    amp = _gaussian_channel_amplitude(data, background, profile, fit_mask)

    np.testing.assert_allclose(amp, amp_true, rtol=1e-6)


def test_gaussian_channel_amplitude_returns_nan_on_degenerate_profile():
    data = np.ones((5, 5))
    profile = np.zeros((5, 5))
    fit_mask = np.ones((5, 5), dtype=bool)
    assert np.isnan(_gaussian_channel_amplitude(data, 0.0, profile, fit_mask))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor.py -k "nonlinear_threshold or channel_amplitude" -v`
Expected: FAIL — `ImportError` for `ALCOR_NONLINEAR_THRESHOLD` / `_gaussian_channel_amplitude`.

- [ ] **Step 3: Add the constant and the helpers**

In `skycam_utils/alcor.py`, find line 44:

```python
ALCOR_SATURATION = 32767
```

and add directly below it:

```python
ALCOR_NONLINEAR_THRESHOLD = 15000   # raw ADU; per-pixel non-linearity onset

# minimum unmasked pixel counts for the Gaussian fits
_GAUSS_MIN_LUM_PIXELS = 8            # 4-parameter luminance shape fit
_GAUSS_MIN_CHANNEL_PIXELS = 3        # 1-parameter per-channel amplitude
```

Then, immediately after `_aperture_saturated` (ends at line 1652) and before `_default_alcor_photometry_output`, insert:

```python
def _gaussian_channel_amplitude(data, background, profile, fit_mask):
    """
    Linear least-squares amplitude of ``data - background`` projected onto a
    fixed unit-Gaussian ``profile`` over ``fit_mask``.

    With the PSF shape held fixed, the best-fit amplitude is the closed-form
    projection ``sum(g*d) / sum(g*g)`` over the unmasked pixels, so the linear
    wings (the only unmasked pixels for a bright star) set the amplitude and the
    suppressed core does not bias it. Returns NaN when the projection is
    degenerate (no profile weight in the mask).
    """
    g = profile[fit_mask]
    denom = float(np.sum(g * g))
    if denom <= 0.0:
        return np.nan
    d = data[fit_mask] - background
    return float(np.sum(g * d) / denom)


def _gaussian_psf_photometry(data, cube, lum_frame, xcen, ycen,
                             aperture_radius, annulus_width, mask_threshold):
    """
    Constrained-Gaussian PSF photometry for one star.

    Pins the PSF center and width from a luminance (channel-summed) fit with the
    non-linear core masked, then recovers each channel's amplitude as the linear
    projection of the background-subtracted, masked aperture data onto the fixed
    unit-Gaussian profile. Flux is the analytic Gaussian integral
    ``2*pi*A*sigma**2``.

    Parameters
    ----------
    data : (3, ny, nx) float `~numpy.ndarray`
        Bias-subtracted RGB cube.
    cube : (3, ny, nx) `~numpy.ndarray`
        Raw RGB cube; the linearity mask is computed on it because the threshold
        is a raw-ADU level.
    lum_frame : (ny, nx) `~numpy.ndarray`
        Precomputed luminance frame ``data.sum(axis=0)``.
    xcen, ycen : float
        WCS-predicted pixel position; the fit seed.
    aperture_radius, annulus_width : float
    mask_threshold : float
        Raw-ADU level at/above which a pixel is excluded from the fit.

    Returns
    -------
    dict or None
        Keys ``xcen``, ``ycen``, ``fwhm`` and per-channel ``flux_<ch>`` and
        ``background_<ch>``. None if the fit cannot be trusted.
    """
    ny, nx = data.shape[1:]
    # Box generous enough to hold the aperture around any allowed fitted center
    # (the center may drift up to aperture_radius from the seed).
    box_r = int(np.ceil(2.0 * aperture_radius)) + 1
    xi = int(np.floor(xcen))
    yi = int(np.floor(ycen))
    x0 = max(0, xi - box_r)
    x1 = min(nx, xi + box_r + 1)
    y0 = max(0, yi - box_r)
    y1 = min(ny, yi + box_r + 1)
    if x0 >= x1 or y0 >= y1:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1]
    raw_box = cube[:, y0:y1, x0:x1]

    # --- luminance shape fit (amp, center, sigma) over the linear core+wings ---
    rr_seed = np.hypot(xx - xcen, yy - ycen)
    in_aperture = rr_seed <= aperture_radius
    lum_linear = np.all(raw_box < mask_threshold, axis=0)
    lum_mask = in_aperture & lum_linear
    if int(lum_mask.sum()) < _GAUSS_MIN_LUM_PIXELS:
        return None

    lum_bkg = _annulus_background(lum_frame, xcen, ycen, aperture_radius,
                                  annulus_width)
    if not np.isfinite(lum_bkg):
        return None
    lum_sub = lum_frame[y0:y1, x0:x1] - lum_bkg

    xf = xx[lum_mask].astype(float)
    yf = yy[lum_mask].astype(float)
    zf = lum_sub[lum_mask]
    amp0 = float(np.max(zf))
    if not np.isfinite(amp0) or amp0 <= 0.0:
        amp0 = 1.0
    sigma0 = aperture_radius / 2.0

    def residual(p):
        amp, cx, cy, sigma = p
        model = amp * np.exp(-((xf - cx) ** 2 + (yf - cy) ** 2)
                             / (2.0 * sigma ** 2))
        return model - zf

    lower = [0.0, xcen - aperture_radius, ycen - aperture_radius, 1e-3]
    upper = [np.inf, xcen + aperture_radius, ycen + aperture_radius,
             aperture_radius]
    try:
        result = least_squares(residual, [amp0, xcen, ycen, sigma0],
                               bounds=(lower, upper), max_nfev=200)
    except (ValueError, RuntimeError):
        return None
    if not result.success:
        return None
    _, cx, cy, sigma = result.x
    if not np.all(np.isfinite([cx, cy, sigma])):
        return None
    if sigma <= 0.0 or sigma >= aperture_radius:
        return None
    if np.hypot(cx - xcen, cy - ycen) > aperture_radius:
        return None

    # --- per-channel amplitude from the linear wings, shape fixed ------------
    rr_fit = np.hypot(xx - cx, yy - cy)
    in_aperture_fit = rr_fit <= aperture_radius
    profile = np.exp(-rr_fit ** 2 / (2.0 * sigma ** 2))
    norm = 2.0 * np.pi * sigma ** 2

    out = {
        "xcen": float(cx),
        "ycen": float(cy),
        "fwhm": float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma),
    }
    for idx, channel in enumerate(("r", "g", "b")):
        ch_bkg = _annulus_background(data[idx], cx, cy, aperture_radius,
                                     annulus_width)
        if not np.isfinite(ch_bkg):
            return None
        ch_mask = in_aperture_fit & (raw_box[idx] < mask_threshold)
        if int(ch_mask.sum()) < _GAUSS_MIN_CHANNEL_PIXELS:
            return None
        amp_ch = _gaussian_channel_amplitude(
            data[idx, y0:y1, x0:x1], ch_bkg, profile, ch_mask)
        if not np.isfinite(amp_ch):
            return None
        out[f"flux_{channel}"] = amp_ch * norm
        out[f"background_{channel}"] = float(ch_bkg)
    return out
```

- [ ] **Step 4: Run the helper tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor.py -k "nonlinear_threshold or channel_amplitude" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor.py
git commit -m "$(cat <<'EOF'
alcor: add constrained-Gaussian PSF photometry helpers

ALCOR_NONLINEAR_THRESHOLD, the linear channel-amplitude projection,
and the luminance-pinned per-star Gaussian fit that masks the
non-linear core and integrates the analytic Gaussian.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire the Gaussian path into `alcor_star_photometry`

**Files:**
- Modify: `skycam_utils/alcor.py:1691-1800` (`alcor_star_photometry`)
- Test: `skycam_utils/tests/test_alcor.py`

- [ ] **Step 1: Write the failing tests**

Add these four tests to `skycam_utils/tests/test_alcor.py` (after the saturation test block):

```python
def _gaussian_star_cube(ny=80, nx=80, cx=40.3, cy=39.7, sigma=1.3,
                        amps=(4000.0, 6000.0, 3000.0), base=100.0):
    yy, xx = np.mgrid[0:ny, 0:nx]
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    cube = np.full((3, ny, nx), base, dtype=float)
    for idx, amp in enumerate(amps):
        cube[idx] += amp * g
    return cube


def _patch_single_star(monkeypatch, cube, px, py):
    from astropy.time import Time
    from skycam_utils import alcor

    class FakeWCS:
        def world_to_pixel_values(self, az, alt):
            return np.array([float(px)]), np.array([float(py)])

    cat = Table({"NAME": ["star"], "HD": [1], "Alt": [80.0], "Az": [10.0]})
    monkeypatch.setattr(alcor, "_alcor_frame_time",
                        lambda filename: Time("2024-09-05T07:00:00"))
    monkeypatch.setattr(alcor, "load_alcor_fits",
                        lambda *a, **k: (cube, FakeWCS(), None))
    monkeypatch.setattr(alcor, "alcor_named_reference_altaz",
                        lambda *a, **k: cat)


def test_alcor_star_photometry_gaussian_recovers_clean_star(tmp_path, monkeypatch):
    cx, cy, sigma = 40.3, 39.7, 1.3
    amps = (4000.0, 6000.0, 3000.0)
    cube = _gaussian_star_cube(cx=cx, cy=cy, sigma=sigma, amps=amps)
    _patch_single_star(monkeypatch, cube, px=40.0, py=40.0)

    phot, _ = alcor_star_photometry(
        tmp_path / "synthetic.fits", output_file=tmp_path / "out.csv",
        gaussian=True, aperture_radius=5.0, annulus_width=2.0)

    assert "fwhm" in phot.columns
    np.testing.assert_allclose(phot.loc["star", "xcen"], cx, atol=0.1)
    np.testing.assert_allclose(phot.loc["star", "ycen"], cy, atol=0.1)
    np.testing.assert_allclose(
        phot.loc["star", "fwhm"],
        2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma, rtol=0.05)
    for idx, channel in enumerate("rgb"):
        expected = amps[idx] * 2.0 * np.pi * sigma ** 2
        np.testing.assert_allclose(
            phot.loc["star", f"flux_{channel}"], expected, rtol=0.05)


def test_alcor_star_photometry_gaussian_beats_aperture_under_nonlinearity(
        tmp_path, monkeypatch):
    cx, cy, sigma = 40.0, 40.0, 1.3
    amp = 40000.0
    ceiling = 16000.0
    cube = _gaussian_star_cube(cx=cx, cy=cy, sigma=sigma,
                               amps=(amp, amp, amp))
    cube = np.minimum(cube, ceiling)        # mimic per-pixel non-linearity
    true_flux = amp * 2.0 * np.pi * sigma ** 2

    _patch_single_star(monkeypatch, cube, px=cx, py=cy)
    phot_g, _ = alcor_star_photometry(
        tmp_path / "g.fits", output_file=tmp_path / "g.csv",
        gaussian=True, aperture_radius=5.0, annulus_width=2.0,
        mask_threshold=15000.0)

    _patch_single_star(monkeypatch, cube, px=cx, py=cy)
    phot_a, _ = alcor_star_photometry(
        tmp_path / "a.fits", output_file=tmp_path / "a.csv",
        gaussian=False, aperture_radius=5.0, annulus_width=2.0)

    rg = float(phot_g.loc["star", "flux_g"])
    ra = float(phot_a.loc["star", "flux_g"])
    # the clamped core was never at the 32767 ceiling, so sat does not catch it
    assert bool(phot_g.loc["star", "sat_g"]) is False
    assert ra < true_flux                            # aperture underestimates
    assert abs(rg - true_flux) < abs(ra - true_flux) # gaussian is closer
    np.testing.assert_allclose(rg, true_flux, rtol=0.1)


def test_alcor_star_photometry_aperture_mode_fwhm_is_nan(tmp_path, monkeypatch):
    cube = _gaussian_star_cube()
    _patch_single_star(monkeypatch, cube, px=40.0, py=40.0)

    phot, _ = alcor_star_photometry(
        tmp_path / "a.fits", output_file=tmp_path / "a.csv",
        gaussian=False, aperture_radius=5.0, annulus_width=2.0)

    assert "fwhm" in phot.columns
    assert phot["fwhm"].isna().all()


def test_alcor_star_photometry_gaussian_drops_signal_free_star(
        tmp_path, monkeypatch):
    cube = np.full((3, 60, 60), 100.0)      # flat field, no star
    _patch_single_star(monkeypatch, cube, px=30.0, py=30.0)

    phot, output_file = alcor_star_photometry(
        tmp_path / "flat.fits", output_file=tmp_path / "flat.csv",
        gaussian=True, aperture_radius=5.0, annulus_width=2.0)

    assert len(phot) == 0
    assert output_file.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor.py -k "gaussian_recovers or beats_aperture or fwhm_is_nan or drops_signal_free" -v`
Expected: FAIL — `TypeError: alcor_star_photometry() got an unexpected keyword argument 'gaussian'`.

- [ ] **Step 3: Add the parameters and branch the loop**

In `skycam_utils/alcor.py`, change the `alcor_star_photometry` signature (lines 1691-1695) from:

```python
def alcor_star_photometry(filename, output_file=None, aperture_radius=4.0,
                          annulus_width=1.0, min_altitude=20.0,
                          vmag_limit=5.5, refraction=True, masks_dir=None,
                          check_plot=False, check_radius=680,
                          sun_alt_max=-12.0, saturation=ALCOR_SATURATION):
```

to:

```python
def alcor_star_photometry(filename, output_file=None, aperture_radius=4.0,
                          annulus_width=1.0, min_altitude=20.0,
                          vmag_limit=5.5, refraction=True, masks_dir=None,
                          check_plot=False, check_radius=680,
                          sun_alt_max=-12.0, saturation=ALCOR_SATURATION,
                          gaussian=False,
                          mask_threshold=ALCOR_NONLINEAR_THRESHOLD):
```

Update the docstring: after the existing first paragraph of the body docstring (ending "...without discarding the unsaturated channels.", line 1707), add this paragraph:

```python
    When ``gaussian`` is True, photometry instead fits a circular Gaussian whose
    center and width are pinned from a luminance (channel-summed) fit with the
    non-linear core masked (raw pixels at/above ``mask_threshold`` excluded), and
    recovers each channel's amplitude from the linear wings. The reported flux is
    the analytic Gaussian integral, which is robust to the CMOS non-linearity
    that suppresses aperture-sum flux for bright stars before saturation. The
    luminance FWHM is reported in the ``fwhm`` column (NaN in aperture mode).
```

Change the column list (line 1724) from:

```python
    columns = ["altitude", "azimuth", "xcen", "ycen"]
```

to:

```python
    columns = ["altitude", "azimuth", "xcen", "ycen", "fwhm"]
```

Replace the per-star measurement loop (lines 1757-1781) — currently:

```python
    rows = []
    labels = []
    cat_labels = _alcor_star_labels(cat)
    for i, (x, y) in enumerate(zip(xcen, ycen)):
        row = {
            "altitude": float(cat["Alt"][i]),
            "azimuth": float(cat["Az"][i]),
            "xcen": float(x),
            "ycen": float(y),
        }
        fluxes = []
        for channel_index, channel in enumerate(channels):
            flux, background = _aperture_annulus_photometry(
                data[channel_index], x, y, aperture_radius, annulus_width)
            fluxes.append(flux)
            row[f"flux_{channel}"] = flux
            row[f"background_{channel}"] = background
            row[f"sat_{channel}"] = _aperture_saturated(
                cube[channel_index], x, y, aperture_radius, saturation)
        if any((not np.isfinite(flux)) or flux <= 0.0 for flux in fluxes):
            continue
        for channel, flux in zip(channels, fluxes):
            row[f"mag_{channel}"] = float(-2.5 * np.log10(flux))
        rows.append(row)
        labels.append(cat_labels[i])
```

with:

```python
    rows = []
    labels = []
    cat_labels = _alcor_star_labels(cat)
    lum_frame = data.sum(axis=0) if gaussian else None
    for i, (x, y) in enumerate(zip(xcen, ycen)):
        row = {
            "altitude": float(cat["Alt"][i]),
            "azimuth": float(cat["Az"][i]),
            "xcen": float(x),
            "ycen": float(y),
            "fwhm": np.nan,
        }
        if gaussian:
            fit = _gaussian_psf_photometry(
                data, cube, lum_frame, x, y, aperture_radius, annulus_width,
                mask_threshold)
            if fit is None:
                continue
            row["xcen"] = fit["xcen"]
            row["ycen"] = fit["ycen"]
            row["fwhm"] = fit["fwhm"]
            fluxes = []
            for channel_index, channel in enumerate(channels):
                flux = fit[f"flux_{channel}"]
                fluxes.append(flux)
                row[f"flux_{channel}"] = flux
                row[f"background_{channel}"] = fit[f"background_{channel}"]
                row[f"sat_{channel}"] = _aperture_saturated(
                    cube[channel_index], fit["xcen"], fit["ycen"],
                    aperture_radius, saturation)
        else:
            fluxes = []
            for channel_index, channel in enumerate(channels):
                flux, background = _aperture_annulus_photometry(
                    data[channel_index], x, y, aperture_radius, annulus_width)
                fluxes.append(flux)
                row[f"flux_{channel}"] = flux
                row[f"background_{channel}"] = background
                row[f"sat_{channel}"] = _aperture_saturated(
                    cube[channel_index], x, y, aperture_radius, saturation)
        if any((not np.isfinite(flux)) or flux <= 0.0 for flux in fluxes):
            continue
        for channel, flux in zip(channels, fluxes):
            row[f"mag_{channel}"] = float(-2.5 * np.log10(flux))
        rows.append(row)
        labels.append(cat_labels[i])
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor.py -k "gaussian_recovers or beats_aperture or fwhm_is_nan or drops_signal_free" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the full alcor test file to check for regressions**

Run: `pytest skycam_utils/tests/test_alcor.py -v`
Expected: PASS — all tests, including the pre-existing photometry tests (which run aperture mode by default and now also see the `fwhm` column).

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor.py
git commit -m "$(cat <<'EOF'
alcor: add gaussian-fit option to alcor_star_photometry

gaussian=True pins PSF shape from a luminance fit, masks the
non-linear core at mask_threshold (default ALCOR_NONLINEAR_THRESHOLD),
fits per-channel amplitude on the linear wings, and reports the
analytic Gaussian integral plus a shared fwhm column. Recovers
bright-star flux that aperture sums lose to CMOS non-linearity
before saturation.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CLI flags

**Files:**
- Modify: `skycam_utils/alcor.py:2598-2651` (`alcor_star_photometry_cli`)
- Test: `skycam_utils/tests/test_alcor.py`

- [ ] **Step 1: Write the failing test**

Add `import sys` near the top of `skycam_utils/tests/test_alcor.py` (with the other stdlib imports, after `import os`). Add `alcor_star_photometry_cli` to the `from skycam_utils.alcor import (...)` block. Then add this test:

```python
def test_alcor_star_photometry_cli_passes_gaussian_flags(monkeypatch):
    from skycam_utils import alcor

    captured = {}

    def fake_photometry(*args, **kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(alcor, "alcor_star_photometry", fake_photometry)
    monkeypatch.setattr(sys, "argv",
                        ["alcor_star_photometry", "frame.fits",
                         "--gaussian", "--mask-threshold", "12000"])

    alcor.alcor_star_photometry_cli()

    assert captured["gaussian"] is True
    assert captured["mask_threshold"] == 12000.0


def test_alcor_star_photometry_cli_defaults_to_aperture(monkeypatch):
    from skycam_utils import alcor

    captured = {}

    def fake_photometry(*args, **kwargs):
        captured.update(kwargs)
        return None, None

    monkeypatch.setattr(alcor, "alcor_star_photometry", fake_photometry)
    monkeypatch.setattr(sys, "argv",
                        ["alcor_star_photometry", "frame.fits"])

    alcor.alcor_star_photometry_cli()

    assert captured["gaussian"] is False
    assert captured["mask_threshold"] == alcor.ALCOR_NONLINEAR_THRESHOLD
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor.py -k "cli_passes_gaussian or cli_defaults_to_aperture" -v`
Expected: FAIL — the parser has no `--gaussian`, so `captured` lacks the key (`KeyError`), or argparse errors on the unknown flag.

- [ ] **Step 3: Add the CLI flags**

In `skycam_utils/alcor.py`, update the `--aperture-radius` help (line 2612-2613) to note its dual role:

```python
    parser.add_argument("--aperture-radius", type=float, default=4.0,
                        help="Circular aperture radius in pixels (also the Gaussian fit window).")
```

Add these two arguments right after the `--saturation` argument (after line 2627, before `--check-plot`):

```python
    parser.add_argument("--gaussian", action="store_true",
                        help="Use constrained-Gaussian PSF photometry instead of aperture sums.")
    parser.add_argument("--mask-threshold", type=float,
                        default=ALCOR_NONLINEAR_THRESHOLD,
                        help="Raw-ADU level at/above which a pixel is excluded from the Gaussian fit.")
```

Update the `alcor_star_photometry(...)` call (lines 2634-2647) to pass them through — add these two keyword arguments before the closing paren (after `saturation=args.saturation,`):

```python
        gaussian=args.gaussian,
        mask_threshold=args.mask_threshold,
```

- [ ] **Step 4: Run the CLI tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor.py -k "cli_passes_gaussian or cli_defaults_to_aperture" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor.py
git commit -m "$(cat <<'EOF'
alcor: add --gaussian and --mask-threshold to photometry CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Alcor prose paragraph**

In `CLAUDE.md`, find the sentence in the long Alcor paragraph that begins "Star photometry: `alcor_star_photometry()` measures fixed-position per-channel (R/G/B) aperture+annulus photometry...". After the sentence describing the `sat_*`/`saturation` behavior (ending "...without losing the others"), insert this sentence:

```
A Gaussian-fit path (`gaussian=True`, CLI `--gaussian`) handles the bright-star CMOS non-linearity that suppresses aperture-sum flux before saturation: it fits a circular Gaussian whose center and width are pinned from a luminance (R+G+B) fit with the non-linear core masked (raw pixels >= `mask_threshold`, default `ALCOR_NONLINEAR_THRESHOLD = 15000`, excluded), recovers each channel's amplitude as the linear projection of the masked, background-subtracted aperture onto the fixed unit-Gaussian profile (so the linear wings set the amplitude), and reports the analytic Gaussian integral `2*pi*A*sigma^2` as `flux_*`; the shared luminance FWHM is added as an `fwhm` column (NaN in aperture mode). `xcen`/`ycen` then hold the fitted center, and `sat_*` still reflects raw-core saturation (now informational).
```

- [ ] **Step 2: Update the `alcor_star_photometry` CLI block**

In `CLAUDE.md`, find the `alcor_star_photometry` usage line in the "Common commands" code block and add the two new flags. Change:

```
alcor_star_photometry <input.fits> [-o OUT.csv] [--aperture-radius 4] [--annulus-width 1] [--min-altitude 20] [--vmag-limit 5.5] [--no-refraction] [--sun-alt-max -12] [--saturation 32767] [--check-plot] [--check-radius 680]
```

to:

```
alcor_star_photometry <input.fits> [-o OUT.csv] [--aperture-radius 4] [--annulus-width 1] [--min-altitude 20] [--vmag-limit 5.5] [--no-refraction] [--sun-alt-max -12] [--saturation 32767] [--gaussian] [--mask-threshold 15000] [--check-plot] [--check-radius 680]
```

Then, after the existing description sentence for that command (ending "...writes nothing when the Sun is above --sun-alt-max."), append:

```
#   --gaussian switches to constrained-Gaussian PSF photometry (luminance-pinned
#   center/width, non-linear core masked at --mask-threshold, analytic-integral
#   flux) to recover bright-star flux lost to CMOS non-linearity; it adds an fwhm
#   column. --aperture-radius also sets the Gaussian fit window.
```

- [ ] **Step 3: Verify the full test suite still passes**

Run: `pytest`
Expected: PASS — entire suite green.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
CLAUDE.md: document gaussian-fit star photometry option

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review notes

**Spec coverage** — every spec section maps to a task:
- Module constant `ALCOR_NONLINEAR_THRESHOLD = 15000` → Task 2.
- Luminance shape fit (sum, annulus bg, AND-mask, 4-param `least_squares`, bounds) → Task 2 `_gaussian_psf_photometry`.
- Per-channel linear amplitude on masked wings → Task 2 `_gaussian_channel_amplitude` + Task 2 orchestrator.
- Analytic-integral flux `2*pi*A*sigma**2`, `mag = -2.5 log10(flux)` → Task 2 (flux) + Task 3 (mag, reusing the shared mag loop).
- Stored `xcen`/`ycen` = fitted center, `fwhm = 2.3548*sigma`, per-channel `background_*`, raw `sat_*` → Task 3 wiring.
- Params `gaussian`, `mask_threshold` + CLI flags → Task 3 + Task 4.
- Output schema (+`fwhm`, NaN in aperture mode) → Task 3 (column list + row init) with tests in Task 3.
- Error handling (drop on non-convergence / bad sigma / center drift / too few pixels / non-positive flux) → Task 2 (`return None`) + Task 3 (`if fit is None: continue` and the existing non-positive-flux skip).
- Tests 1-5 from the spec → Task 1 (annulus), Task 2 (amplitude helper), Task 3 (clean recovery, non-linearity-beats-aperture, schema, drop).
- Docs → Task 5.

**Placeholder scan** — no TBD/TODO; every code step shows complete code; every command shows expected output.

**Type/name consistency** — `_annulus_background` (Task 1) is consumed by `_gaussian_psf_photometry` (Task 2). `_gaussian_channel_amplitude(data, background, profile, fit_mask)` signature matches its call in the orchestrator and the unit test. `_gaussian_psf_photometry(data, cube, lum_frame, xcen, ycen, aperture_radius, annulus_width, mask_threshold)` signature matches its call in Task 3. Returned dict keys (`xcen`, `ycen`, `fwhm`, `flux_<ch>`, `background_<ch>`) match what the Task 3 loop reads. `ALCOR_NONLINEAR_THRESHOLD` default is consistent across the function signature, the CLI default, and the tests.

**Note on the check-plot:** no change needed — `save_alcor_photometry_check_plot` reads `xcen`/`ycen` from the DataFrame (the fitted centers in Gaussian mode) and draws the aperture/annulus geometry, which stays meaningful.
