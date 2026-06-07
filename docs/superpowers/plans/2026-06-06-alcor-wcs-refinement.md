# Alcor WCS Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the idealized equidistant ARC WCS in `load_alcor_fits` with a lens model fitted to bright catalog stars — refined center, rotation, and a non-linear radial term in zenith angle — calibrated by aggregating dark-sky frames across a night.

**Architecture:** A single pure forward model `_predict_pixels` maps (alt, az) → processed-image pixel for a 6-parameter geometry (center shift, rotation, cubic radial polynomial). The same model drives (a) the offline calibration fit (`fit_alcor_wcs`, detect → catalog AltAz with refraction → bootstrap match from zenith outward → `scipy.optimize.least_squares`, pooled over all Sun-below-−18° frames) and (b) `build_alcor_wcs`, which reproduces the radial model as SIP distortion on an ARC WCS via `astropy`'s `fit_wcs_from_points` (the stellacam pattern), cached per geometry. The fitted constants become baked-in defaults in `load_alcor_fits`; with the idealized defaults the new code is byte-for-byte equivalent to the current behavior, so existing tests keep passing until the real constants are baked in the final task.

**Tech Stack:** Python, numpy, scipy.optimize, astropy (WCS, coordinates, time, units), photutils 3.0 (`DAOStarFinder`).

---

## Conventions used throughout

- **Processed frame:** `load_alcor_fits` produces a `(2*radius, 2*radius, 3)` image, zenith near the array center, north up. Axis 0 = row = `y`, axis 1 = column = `x`.
- **Forward model** (validated against the existing `test_load_alcor_fits_wcs_maps_zenith_and_horizon` numbers):
  ```
  rho = r / horizon_radius                     # normalized detector radius, 0 at zenith
  z   = 90 * (k1*rho + k3*rho**3 + k5*rho**5)   # zenith angle from detector radius (plate solution)
  # _predict_pixels (alt/az -> pixel, used for matching) inverts the above for
  # rho via a few vectorized Newton steps, then:
  x   = radius + xshift - r*sin(radians(az + rotation))
  y   = radius + yshift + r*cos(radians(az + rotation))
  ```
  The radial polynomial is **odd-power only** and written in SIP's native
  direction — **detector radius → zenith angle** (the plate solution) — so
  `radial_coeffs = (k1, k3, k5)` are the coefficients of ρ, ρ³, ρ⁵. Because the
  map from detector pixels to sky is itself an odd polynomial, SIP reproduces it
  to numerical precision everywhere (a degree-5 SIP captures the ρ⁵ term exactly;
  hence `sip_degree=5`). Idealized coefficients `(k1, k3, k5) = (1.0, 0.0, 0.0)`
  reproduce the current equidistant mapping exactly (`z = 90*rho`, so
  horizon `r=horizon_radius`→z=90, zenith→0). Note the sign sense: with this
  parametrization a positive higher-order coefficient maps a star at fixed
  altitude to a *smaller* pixel radius.
- **Refinement is delivered as new defaults** on `load_alcor_fits`: `xshift`, `yshift` (recenter zenith to the array center via `scipy.ndimage.shift`), `rotation` (folds the residual rotation into the existing image rotate), and `radial_coeffs` (the cubic). All default to the idealized values so behavior is unchanged until Task 9 bakes the fitted numbers.
- **Module constants** (added in Task 1) hold the baked values: `ALCOR_RADIUS=680`, `ALCOR_HORIZON_RADIUS=662`, `ALCOR_ROTATION=0.4`, `ALCOR_XSHIFT=0.0`, `ALCOR_YSHIFT=0.0`, `ALCOR_RADIAL_COEFFS=(1.0, 0.0, 0.0)`.
- **Atmospheric refraction** for the catalog AltAz: nominal MMT pressure `760 * u.hPa` (~0.75 atm), temperature `10 * u.deg_C`, relative humidity `0.2`, `obswl = 0.55 * u.micron`.

## File structure

- `skycam_utils/alcor.py` — new constants, `_predict_pixels`, `_sun_altitude` / `select_dark_frames`, `build_alcor_wcs`, `detect_alcor_stars`, `alcor_reference_altaz`, `match_alcor_stars`, `fit_alcor_wcs`, `fit_alcor_wcs_cli`; `load_alcor_fits` refactored to use the new defaults + `build_alcor_wcs`.
- `pyproject.toml` — new `[project.scripts]` entry `fit_alcor_wcs`.
- `skycam_utils/tests/test_alcor_wcs.py` — new tests for the pure model, dark-frame selection, WCS construction, detection, catalog prep, matching, and the synthetic end-to-end fit.
- `skycam_utils/tests/test_alcor.py` — two existing WCS-value assertions updated in Task 9 once real constants are baked.
- `CLAUDE.md` — short note about `fit_alcor_wcs` (Task 9).

---

## Task 1: Pure forward model `_predict_pixels` + constants

**Files:**
- Modify: `skycam_utils/alcor.py` (add constants + function near the top, after the imports block ~line 16)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Create the test file with failing tests**

Create `skycam_utils/tests/test_alcor_wcs.py`:

```python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import (
    ALCOR_HORIZON_RADIUS,
    ALCOR_RADIUS,
    ALCOR_RADIAL_COEFFS,
    _predict_pixels,
)


def test_predict_pixels_idealized_reproduces_zenith_and_horizon():
    # Zenith maps to the array center.
    x, y = _predict_pixels(90.0, 0.0)
    np.testing.assert_allclose([x, y], [ALCOR_RADIUS, ALCOR_RADIUS], atol=1e-9)

    # alt=0 maps to a circle of horizon_radius, with the existing azimuth layout:
    # az=0 -> +y (top), az=90 -> -x (left), az=180 -> -y, az=270 -> +x.
    az = np.array([0.0, 90.0, 180.0, 270.0])
    x, y = _predict_pixels(np.zeros_like(az), az)
    radii = np.hypot(x - ALCOR_RADIUS, y - ALCOR_RADIUS)
    np.testing.assert_allclose(radii, ALCOR_HORIZON_RADIUS, atol=1e-9)
    np.testing.assert_allclose(x, ALCOR_RADIUS - ALCOR_HORIZON_RADIUS * np.sin(np.radians(az)), atol=1e-9)
    np.testing.assert_allclose(y, ALCOR_RADIUS + ALCOR_HORIZON_RADIUS * np.cos(np.radians(az)), atol=1e-9)


def test_predict_pixels_radial_term_pushes_stars_outward():
    # A positive cubic term increases pixel radius at large zenith angle.
    base_x, base_y = _predict_pixels(10.0, 45.0)
    bent_x, bent_y = _predict_pixels(10.0, 45.0, radial_coeffs=(1.0, 0.0, 0.1))
    base_r = np.hypot(base_x - ALCOR_RADIUS, base_y - ALCOR_RADIUS)
    bent_r = np.hypot(bent_x - ALCOR_RADIUS, bent_y - ALCOR_RADIUS)
    assert bent_r > base_r


def test_predict_pixels_default_coeffs_are_idealized():
    assert ALCOR_RADIAL_COEFFS == (1.0, 0.0, 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -v`
Expected: FAIL with `ImportError` (names not defined in `skycam_utils.alcor`).

- [ ] **Step 3: Add constants and `_predict_pixels` to `alcor.py`**

Insert after the import block (after line 16, before `def load_alcor_fits`):

```python
ALCOR_RADIUS = 680
ALCOR_HORIZON_RADIUS = 662
ALCOR_ROTATION = 0.4
ALCOR_XSHIFT = 0.0
ALCOR_YSHIFT = 0.0
ALCOR_RADIAL_COEFFS = (1.0, 0.0, 0.0)


def _predict_pixels(
    alt,
    az,
    xshift=ALCOR_XSHIFT,
    yshift=ALCOR_YSHIFT,
    rotation=0.0,
    radial_coeffs=ALCOR_RADIAL_COEFFS,
    radius=ALCOR_RADIUS,
    horizon_radius=ALCOR_HORIZON_RADIUS,
):
    """
    Forward lens model: map altitude/azimuth (degrees) to processed-frame
    pixel coordinates (x=column, y=row).

    The radial mapping is ``r = horizon_radius * (k1*zeta + k2*zeta**2 +
    k3*zeta**3)`` with ``zeta = (90 - alt)/90``. The idealized coefficients
    ``(1, 0, 0)`` give the equidistant ARC mapping; higher-order terms encode
    the lens's non-linear growth with zenith angle. ``rotation`` is the
    azimuth-frame rotation in degrees; ``xshift``/``yshift`` offset the zenith
    from the array center.
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    k1, k2, k3 = radial_coeffs
    zeta = (90.0 - alt) / 90.0
    r = horizon_radius * (k1 * zeta + k2 * zeta**2 + k3 * zeta**3)
    ang = np.radians(az + rotation)
    x = radius + xshift - r * np.sin(ang)
    y = radius + yshift + r * np.cos(ang)
    return x, y
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add alcor forward lens model and geometry constants"
```

---

## Task 2: Dark-frame selection by Sun altitude

**Files:**
- Modify: `skycam_utils/alcor.py`
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Add failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
from astropy.time import Time

from skycam_utils.alcor import _sun_altitude, select_dark_frames


def test_sun_altitude_night_vs_day():
    # 2024-09-05 07:00 UT at MMT is local ~midnight -> Sun well below horizon.
    night = _sun_altitude(Time("2024-09-05T07:00:00", format="isot", scale="utc"))
    # 2024-09-05 20:00 UT is local ~13:00 -> Sun high.
    day = _sun_altitude(Time("2024-09-05T20:00:00", format="isot", scale="utc"))
    assert night < -18.0
    assert day > 18.0


def test_select_dark_frames_filters_by_sun_altitude(tmp_path):
    from astropy.io import fits

    def write(name, date_obs):
        hdu = fits.PrimaryHDU(data=np.zeros((3, 4, 4), dtype=np.int16))
        hdu.header["DATE-OBS"] = date_obs
        path = tmp_path / name
        hdu.writeto(path)
        return path

    dark = write("dark.fits", "2024-09-05T07:00:00")
    twilight = write("twi.fits", "2024-09-05T02:30:00")  # near sunset, Sun above -18
    day = write("day.fits", "2024-09-05T20:00:00")

    selected = select_dark_frames([day, dark, twilight], sun_alt_max=-18.0)
    assert selected == [dark]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "sun_altitude or dark_frames" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add to the imports at the top of `alcor.py` (extend the existing astropy import lines):

```python
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, get_sun
```

Then add, after `_predict_pixels`:

```python
from .astrometry import MMT_LOCATION


def _sun_altitude(time, location=MMT_LOCATION):
    """Return the Sun's altitude in degrees at ``time`` and ``location``."""
    altaz = get_sun(time).transform_to(AltAz(obstime=time, location=location))
    return float(altaz.alt.deg)


def select_dark_frames(files, sun_alt_max=-18.0, location=MMT_LOCATION):
    """
    Return the subset of ``files`` whose DATE-OBS corresponds to a Sun
    altitude below ``sun_alt_max`` (default -18 deg, astronomical twilight).
    """
    files = [Path(f) for f in files]
    times = []
    for f in files:
        with fits.open(f) as hdul:
            times.append(hdul[0].header["DATE-OBS"])
    times = Time(times, format="isot", scale="utc")
    altaz = get_sun(times).transform_to(AltAz(obstime=times, location=location))
    keep = altaz.alt.deg < sun_alt_max
    return [f for f, k in zip(files, keep) if k]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "sun_altitude or dark_frames" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add Sun-altitude dark-frame selection for alcor calibration"
```

---

## Task 3: `build_alcor_wcs` + refactor `load_alcor_fits`

**Files:**
- Modify: `skycam_utils/alcor.py` (`load_alcor_fits` at lines 18-79)
- Test: `skycam_utils/tests/test_alcor_wcs.py` and existing `skycam_utils/tests/test_alcor.py` (must still pass unchanged)

- [ ] **Step 1: Add failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
from skycam_utils.alcor import build_alcor_wcs, load_alcor_fits


def test_build_alcor_wcs_idealized_matches_equidistant():
    wcs = build_alcor_wcs(radius=680, horizon_radius=662, radial_coeffs=(1.0, 0.0, 0.0))
    assert list(wcs.wcs.ctype) == ["RA---ARC", "DEC--ARC"]
    np.testing.assert_allclose(wcs.wcs.crpix, [680.5, 680.5])
    np.testing.assert_allclose(wcs.wcs.cdelt, [90.0 / 662.0, 90.0 / 662.0])
    assert wcs.sip is None  # no SIP attached when the model is purely linear

    az = np.array([0.0, 90.0, 180.0, 270.0])
    px, py = wcs.world_to_pixel_values(az, np.zeros_like(az))
    radii = np.hypot(px - 679.5, py - 679.5)
    np.testing.assert_allclose(radii, 662.0, atol=1e-6)


def test_build_alcor_wcs_with_radial_term_reproduces_forward_model():
    coeffs = (1.0, 0.02, 0.05)
    wcs = build_alcor_wcs(radius=680, horizon_radius=662, radial_coeffs=coeffs)
    assert wcs.sip is not None

    alt = np.array([80.0, 60.0, 40.0, 20.0, 5.0])
    az = np.array([10.0, 100.0, 190.0, 280.0, 350.0])
    model_x, model_y = _predict_pixels(alt, az, radial_coeffs=coeffs)
    wcs_x, wcs_y = wcs.world_to_pixel_values(az, alt)
    # SIP reproduces the radial model to better than 0.1 px across the FOV.
    np.testing.assert_allclose(wcs_x, model_x, atol=0.1)
    np.testing.assert_allclose(wcs_y, model_y, atol=0.1)


def test_load_alcor_fits_idealized_defaults_unchanged():
    test_fits = Path(__file__).with_name("test.fits.bz2")
    im, wcs = load_alcor_fits(test_fits)
    assert im.shape == (1360, 1360, 3)
    np.testing.assert_allclose(wcs.wcs.crpix, [680.5, 680.5])
    np.testing.assert_allclose(wcs.wcs.cdelt, [90.0 / 662.0, 90.0 / 662.0])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "build_alcor_wcs or idealized_defaults" -v`
Expected: FAIL with `ImportError` for `build_alcor_wcs`.

- [ ] **Step 3: Implement `build_alcor_wcs` and refactor `load_alcor_fits`**

Add these imports near the top of `alcor.py` (with the other astropy imports):

```python
from functools import lru_cache
from scipy.ndimage import shift as ndimage_shift
from astropy.wcs.utils import fit_wcs_from_points
```

Add `build_alcor_wcs` after `select_dark_frames`:

```python
def _base_arc_wcs(radius, horizon_radius, k1):
    """Construct the linear ARC WCS (no SIP) for the given geometry."""
    cdelt = 90.0 / (horizon_radius * k1)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
    wcs.wcs.crpix = [radius + 0.5, radius + 0.5]
    wcs.wcs.crval = [0.0, 90.0]
    wcs.wcs.cdelt = [cdelt, cdelt]
    wcs.wcs.lonpole = 0.0
    return wcs


@lru_cache(maxsize=32)
def build_alcor_wcs(radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                    radial_coeffs=ALCOR_RADIAL_COEFFS, sip_degree=4):
    """
    Build an ARC-projection WCS for the processed alcor frame. When the radial
    model has non-trivial higher-order terms, the deviation from the linear
    (equidistant) mapping is encoded as SIP distortion fitted from a synthetic
    grid of the forward model, so ``world_to_pixel``/``to_header`` reproduce the
    lens distortion. Cached because the geometry is fixed across images.
    """
    radial_coeffs = tuple(float(c) for c in radial_coeffs)
    k1, k2, k3 = radial_coeffs
    base = _base_arc_wcs(radius, horizon_radius, k1)
    if abs(k2) < 1e-12 and abs(k3) < 1e-12:
        return base

    # Synthetic grid spanning the FOV (avoid the alt=90 pole singularity).
    alt_grid = np.linspace(2.0, 89.5, 40)
    az_grid = np.linspace(0.0, 350.0, 36)
    alt_mesh, az_mesh = np.meshgrid(alt_grid, az_grid)
    alt_flat = alt_mesh.ravel()
    az_flat = az_mesh.ravel()
    x, y = _predict_pixels(
        alt_flat, az_flat,
        xshift=0.0, yshift=0.0, rotation=0.0,
        radial_coeffs=radial_coeffs, radius=radius, horizon_radius=horizon_radius,
    )
    world = SkyCoord(az_flat * u.deg, alt_flat * u.deg)
    wcs = fit_wcs_from_points(
        (x, y), world,
        projection=base,
        proj_point=SkyCoord(0 * u.deg, 90 * u.deg),
        sip_degree=sip_degree,
    )
    return wcs
```

Now replace the body of `load_alcor_fits` (lines 56-79, the part after the docstring) so it uses the new defaults and `build_alcor_wcs`. Change the signature line (18) to:

```python
def load_alcor_fits(filename, rotation=ALCOR_ROTATION, xcen=696, ycen=698,
                    radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                    xshift=ALCOR_XSHIFT, yshift=ALCOR_YSHIFT,
                    radial_coeffs=ALCOR_RADIAL_COEFFS, sip_degree=4):
```

Replace the post-docstring body (currently lines 56-79) with:

```python
    with fits.open(filename) as hdul:
        data = hdul[0].data
    im = np.transpose(data, axes=(1, 2, 0)) - 2000  # 2000 is a bit above the normal bias level of the camera.
    im[im < 0] = 0
    im = im * 1.0
    xl = xcen - radius
    xu = xcen + radius
    yl = ycen - radius
    yu = ycen + radius
    im = im[yl:yu, xl:xu, :]
    im = np.flipud(rotate(im, rotation, reshape=False))
    if xshift != 0.0 or yshift != 0.0:
        # Recenter the zenith onto the array center (rows=y, cols=x, channels untouched).
        im = ndimage_shift(im, shift=(-yshift, -xshift, 0.0), order=1, mode="constant", cval=0.0)

    wcs = build_alcor_wcs(
        radius=radius,
        horizon_radius=horizon_radius,
        radial_coeffs=tuple(float(c) for c in radial_coeffs),
        sip_degree=sip_degree,
    )

    return im, wcs
```

- [ ] **Step 4: Run new and existing tests**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py skycam_utils/tests/test_alcor.py -v`
Expected: PASS — the new `build_alcor_wcs`/`load_alcor_fits` tests pass, and **all existing `test_alcor.py` tests still pass** (idealized defaults are byte-equivalent: no shift, no SIP, same crpix/cdelt).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add build_alcor_wcs and route load_alcor_fits through it"
```

---

## Task 4: Star detection `detect_alcor_stars`

**Files:**
- Modify: `skycam_utils/alcor.py`
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Add failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
from skycam_utils.alcor import detect_alcor_stars


def test_detect_alcor_stars_on_synthetic_image():
    rng = np.random.default_rng(0)
    im = np.zeros((200, 200, 3), dtype=float)
    im += rng.normal(0.0, 1.0, im.shape)
    truth = [(50.0, 60.0), (120.0, 140.0), (160.0, 30.0)]
    yy, xx = np.mgrid[0:200, 0:200]
    for cx, cy in truth:
        g = 500.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0**2))
        im += g[:, :, None]

    sources = detect_alcor_stars(im, fwhm=2.5, threshold_sigma=5.0)
    assert len(sources) >= 3
    assert {"xcentroid", "ycentroid", "flux"}.issubset(sources.colnames)
    # Brightest detected source coordinates land near one of the injected stars.
    sources.sort("flux", reverse=True)
    bx, by = sources["xcentroid"][0], sources["ycentroid"][0]
    dists = [np.hypot(bx - cx, by - cy) for cx, cy in truth]
    assert min(dists) < 2.0


def test_detect_alcor_stars_on_real_fixture():
    test_fits = Path(__file__).with_name("test.fits.bz2")
    im, _ = load_alcor_fits(test_fits)
    sources = detect_alcor_stars(im)
    assert len(sources) > 10
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k detect -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add the import near the top of `alcor.py`:

```python
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
```

Add the function after `build_alcor_wcs`:

```python
def detect_alcor_stars(im, fwhm=3.0, threshold_sigma=5.0):
    """
    Detect point sources in a processed alcor RGB image.

    The three channels are summed into a luminance image, the background is
    estimated with a sigma-clipped median, and `~photutils.detection.DAOStarFinder`
    extracts sources above ``threshold_sigma`` times the background noise.

    Returns an astropy table with at least ``xcentroid``, ``ycentroid``, ``flux``.
    """
    lum = np.asarray(im, dtype=float).sum(axis=2)
    mean, median, std = sigma_clipped_stats(lum, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = finder(lum - median)
    if sources is None:
        from astropy.table import Table
        return Table(names=["xcentroid", "ycentroid", "flux"])
    return sources
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k detect -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add DAOStarFinder-based alcor star detection"
```

---

## Task 5: Catalog AltAz prep `alcor_reference_altaz`

**Files:**
- Modify: `skycam_utils/alcor.py`
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Add failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
from astropy.time import Time as _Time

from skycam_utils.alcor import alcor_reference_altaz


def test_alcor_reference_altaz_filters_and_refracts():
    time = _Time("2024-09-05T07:00:00", format="isot", scale="utc")
    cat = alcor_reference_altaz(time, vmag_limit=3.0, min_alt=5.0)

    assert {"Alt", "Az", "Vmag"}.issubset(cat.colnames)
    assert len(cat) > 0
    assert (cat["Vmag"] <= 3.0).all()
    assert (cat["Alt"] >= 5.0).all()

    # Refraction lifts stars: with refraction, altitudes are >= the airless value.
    no_refr = alcor_reference_altaz(time, vmag_limit=3.0, min_alt=5.0, refraction=False)
    # match a star present in both by HD number and compare its altitude.
    common = set(cat["HD"]) & set(no_refr["HD"])
    hd = sorted(common)[0]
    a_refr = float(cat["Alt"][list(cat["HD"]).index(hd)])
    a_none = float(no_refr["Alt"][list(no_refr["HD"]).index(hd)])
    assert a_refr >= a_none - 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k reference_altaz -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add the function after `detect_alcor_stars`. Note the catalog columns confirmed present in `bright_star_sloan.fits`: `_RAJ2000`, `_DEJ2000`, `HD`, `Vmag`.

```python
ALCOR_PRESSURE = 760 * u.hPa        # ~0.75 atm at the MMT 2600 m elevation
ALCOR_TEMPERATURE = 10 * u.deg_C
ALCOR_HUMIDITY = 0.2
ALCOR_OBSWL = 0.55 * u.micron


def alcor_reference_altaz(time, vmag_limit=3.0, min_alt=5.0, refraction=True,
                          location=MMT_LOCATION):
    """
    Load `bright_star_sloan.fits`, filter to ``Vmag <= vmag_limit``, and compute
    Alt/Az at ``time`` and ``location``. Stars below ``min_alt`` are dropped.

    When ``refraction`` is True the AltAz frame includes atmospheric refraction
    using nominal MMT pressure/temperature; this matters most at large zenith
    angle, where the radial distortion is also largest.
    """
    from importlib.resources import files
    from astropy.table import Table

    catpath = files("skycam_utils") / "data" / "bright_star_sloan.fits"
    cat = Table.read(str(catpath))
    cat = cat[cat["Vmag"] <= vmag_limit]

    coords = SkyCoord(cat["_RAJ2000"], cat["_DEJ2000"], unit="deg", frame="icrs")
    if refraction:
        frame = AltAz(obstime=time, location=location, pressure=ALCOR_PRESSURE,
                      temperature=ALCOR_TEMPERATURE, relative_humidity=ALCOR_HUMIDITY,
                      obswl=ALCOR_OBSWL)
    else:
        frame = AltAz(obstime=time, location=location)
    altaz = coords.transform_to(frame)
    cat["Alt"] = altaz.alt.deg
    cat["Az"] = altaz.az.deg
    cat = cat[cat["Alt"] >= min_alt]
    return cat
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k reference_altaz -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add bright-star catalog AltAz prep with refraction for alcor"
```

---

## Task 6: Bootstrap matching `match_alcor_stars`

**Files:**
- Modify: `skycam_utils/alcor.py`
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Add failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
from astropy.table import Table as _Table

from skycam_utils.alcor import match_alcor_stars


def _fake_detections(alt, az, params):
    x, y = _predict_pixels(alt, az, **params)
    return _Table({"xcentroid": np.asarray(x), "ycentroid": np.asarray(y),
                   "flux": np.linspace(1000, 100, len(np.atleast_1d(x)))})


def test_match_alcor_stars_recovers_correspondences():
    # Catalog of stars spread across zenith angles.
    alt = np.array([85.0, 70.0, 55.0, 40.0, 25.0, 10.0])
    az = np.array([15.0, 80.0, 150.0, 210.0, 290.0, 340.0])
    cat = _Table({"Alt": alt, "Az": az, "Vmag": np.linspace(0.5, 3.0, len(alt))})

    # Detections generated from a slightly perturbed geometry.
    true_params = dict(xshift=4.0, yshift=-3.0, rotation=0.6, radial_coeffs=(1.0, 0.03, 0.04))
    det = _fake_detections(alt, az, true_params)

    matched = match_alcor_stars(
        cat, det,
        init_params=dict(xshift=0.0, yshift=0.0, rotation=0.0, radial_coeffs=(1.0, 0.0, 0.0)),
        z_steps=(20.0, 45.0, 70.0, 90.0), tolerance=12.0,
    )
    assert len(matched) >= 5
    assert {"Alt", "Az", "xcentroid", "ycentroid"}.issubset(matched.colnames)
    # Every match links a catalog star to the detection generated from it.
    for row in matched:
        ex, ey = _predict_pixels(row["Alt"], row["Az"], **true_params)
        assert np.hypot(row["xcentroid"] - ex, row["ycentroid"] - ey) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k match_alcor -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Add after `alcor_reference_altaz`:

```python
from astropy.table import hstack as _hstack


def match_alcor_stars(cat, detections, init_params, z_steps=(20.0, 40.0, 60.0, 75.0, 90.0),
                      tolerance=12.0, radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS):
    """
    Match catalog stars (with Alt/Az/Vmag columns) to detected sources by
    bootstrapping outward from the zenith.

    For each zenith-angle cutoff in ``z_steps`` the catalog stars within that
    cutoff are projected to pixels with the current geometry parameters and
    matched to the nearest detection within ``tolerance`` pixels. When multiple
    catalog stars claim the same detection the brighter one (smaller Vmag) wins.
    After each step the geometry is refit (via :func:`_fit_params`) and the
    cutoff expands. Returns a table of matched (catalog + detection) rows.
    """
    det_x = np.asarray(detections["xcentroid"], dtype=float)
    det_y = np.asarray(detections["ycentroid"], dtype=float)
    params = dict(init_params)

    matched_idx = {}  # detection index -> (catalog row index, vmag)
    cat_z = 90.0 - np.asarray(cat["Alt"], dtype=float)
    vmag = np.asarray(cat["Vmag"], dtype=float)

    for z_cut in z_steps:
        within = np.where(cat_z <= z_cut)[0]
        if within.size == 0:
            continue
        px, py = _predict_pixels(
            cat["Alt"][within], cat["Az"][within],
            xshift=params["xshift"], yshift=params["yshift"], rotation=params["rotation"],
            radial_coeffs=tuple(params["radial_coeffs"]), radius=radius, horizon_radius=horizon_radius,
        )
        claims = {}  # detection index -> (cat row index, vmag, dist)
        for cat_i, x, y in zip(within, np.atleast_1d(px), np.atleast_1d(py)):
            d = np.hypot(det_x - x, det_y - y)
            j = int(np.argmin(d))
            if d[j] > tolerance:
                continue
            prev = claims.get(j)
            if prev is None or vmag[cat_i] < prev[1]:
                claims[j] = (cat_i, vmag[cat_i], d[j])
        matched_idx = {j: (ci, vm) for j, (ci, vm, _) in claims.items()}
        if len(matched_idx) >= 3:
            cat_rows = [ci for ci, _ in matched_idx.values()]
            det_rows = list(matched_idx.keys())
            params = _fit_params(
                cat["Alt"][cat_rows], cat["Az"][cat_rows],
                det_x[det_rows], det_y[det_rows],
                init_params=params, radius=radius, horizon_radius=horizon_radius,
            )

    if not matched_idx:
        return _hstack([cat[[]], detections[[]]])
    det_rows = list(matched_idx.keys())
    cat_rows = [ci for ci, _ in matched_idx.values()]
    out = _hstack([_Table(cat[cat_rows]), _Table(detections[det_rows])])
    return out
```

> `_fit_params` is implemented in Task 7. To keep this task self-contained for its test, add the **minimal** `_fit_params` now (Task 7 adds its own test and refines nothing — the signature is final):

```python
from scipy.optimize import least_squares


def _fit_params(alt, az, obs_x, obs_y, init_params, radius=ALCOR_RADIUS,
                horizon_radius=ALCOR_HORIZON_RADIUS):
    """
    Least-squares fit of (xshift, yshift, rotation, k3, k5) to matched stars.
    k1 is held at 1.0 (the zenith plate scale is set by horizon_radius/cdelt);
    k3, k5 are the odd-power radial distortion coefficients. Returns an updated
    params dict.
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    obs_x = np.asarray(obs_x, dtype=float)
    obs_y = np.asarray(obs_y, dtype=float)
    p0 = np.array([
        init_params["xshift"], init_params["yshift"], init_params["rotation"],
        init_params["radial_coeffs"][1], init_params["radial_coeffs"][2],
    ], dtype=float)

    def residuals(p):
        xshift, yshift, rot, k3, k5 = p
        x, y = _predict_pixels(alt, az, xshift=xshift, yshift=yshift, rotation=rot,
                               radial_coeffs=(1.0, k3, k5), radius=radius,
                               horizon_radius=horizon_radius)
        return np.concatenate([x - obs_x, y - obs_y])

    result = least_squares(residuals, p0)
    xshift, yshift, rot, k3, k5 = result.x
    return dict(xshift=float(xshift), yshift=float(yshift), rotation=float(rot),
                radial_coeffs=(1.0, float(k3), float(k5)))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "match_alcor" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add bootstrap star matching and least-squares param fit for alcor"
```

---

## Task 7: End-to-end fit `fit_alcor_wcs`

**Files:**
- Modify: `skycam_utils/alcor.py`
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Add failing tests**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
from skycam_utils.alcor import _fit_params, fit_alcor_wcs


def test_fit_params_recovers_known_geometry():
    rng = np.random.default_rng(1)
    alt = rng.uniform(5.0, 88.0, 200)
    az = rng.uniform(0.0, 360.0, 200)
    true = dict(xshift=5.0, yshift=-4.0, rotation=0.7, radial_coeffs=(1.0, 0.03, 0.05))
    x, y = _predict_pixels(alt, az, **true)

    fit = _fit_params(alt, az, x, y,
                      init_params=dict(xshift=0.0, yshift=0.0, rotation=0.0,
                                       radial_coeffs=(1.0, 0.0, 0.0)))
    assert abs(fit["xshift"] - 5.0) < 1e-3
    assert abs(fit["yshift"] + 4.0) < 1e-3
    assert abs(fit["rotation"] - 0.7) < 1e-3
    np.testing.assert_allclose(fit["radial_coeffs"], (1.0, 0.03, 0.05), atol=1e-4)


def test_fit_alcor_wcs_aggregates_synthetic_frames(monkeypatch, tmp_path):
    import skycam_utils.alcor as alcor_mod
    from astropy.table import Table

    true = dict(xshift=6.0, yshift=-5.0, rotation=0.8, radial_coeffs=(1.0, 0.04, 0.06))
    rng = np.random.default_rng(2)

    # Two synthetic "frames": each provides catalog stars and matching detections.
    frame_alt = [rng.uniform(10.0, 88.0, 30), rng.uniform(10.0, 88.0, 30)]
    frame_az = [rng.uniform(0.0, 360.0, 30), rng.uniform(0.0, 360.0, 30)]
    files = [tmp_path / "f0.fits", tmp_path / "f1.fits"]
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
                      "Vmag": rng.uniform(0.5, 3.0, 30), "HD": np.arange(30)})

    def fake_detect(im, **kw):
        i = calls["i"]
        calls["i"] += 1
        x, y = _predict_pixels(frame_alt[i], frame_az[i], **true)
        return Table({"xcentroid": x, "ycentroid": y, "flux": np.linspace(1e3, 1e2, 30)})

    def fake_frame_time(path):
        return _Time("2024-09-05T07:00:00", format="isot", scale="utc")

    monkeypatch.setattr(alcor_mod, "select_dark_frames", fake_select_dark_frames)
    monkeypatch.setattr(alcor_mod, "load_alcor_fits", fake_load_alcor_fits)
    monkeypatch.setattr(alcor_mod, "alcor_reference_altaz", fake_reference_altaz)
    monkeypatch.setattr(alcor_mod, "detect_alcor_stars", fake_detect)
    monkeypatch.setattr(alcor_mod, "_frame_time", fake_frame_time)

    result = fit_alcor_wcs(tmp_path, pattern="*.fits")
    assert abs(result["xshift"] - 6.0) < 0.05
    assert abs(result["yshift"] + 5.0) < 0.05
    assert abs(result["rotation"] - 0.8) < 0.05
    np.testing.assert_allclose(result["radial_coeffs"], (1.0, 0.04, 0.06), atol=2e-3)
    assert result["n_matched"] >= 40
    assert result["residual_rms"] < 0.1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "fit_params_recovers or aggregates_synthetic" -v`
Expected: FAIL with `ImportError` for `fit_alcor_wcs` (and `_frame_time`).

- [ ] **Step 3: Implement**

Add after `_fit_params`:

```python
def _frame_time(path):
    """Return the observation Time from a FITS file's DATE-OBS header."""
    with fits.open(path) as hdul:
        return Time(hdul[0].header["DATE-OBS"], format="isot", scale="utc")


def fit_alcor_wcs(input_dir, pattern="*.fits.bz2", vmag_limit=3.0, sun_alt_max=-18.0,
                  min_alt=10.0, tolerance=12.0, fwhm=3.0, threshold_sigma=5.0,
                  z_steps=(20.0, 40.0, 60.0, 75.0, 90.0), max_frames=None):
    """
    Calibrate the alcor lens geometry by aggregating bright-star matches across
    all dark-sky frames in ``input_dir``.

    Frames are selected with :func:`select_dark_frames` (Sun below ``sun_alt_max``).
    For each, stars (``Vmag <= vmag_limit``) are projected with the current
    geometry, matched via :func:`match_alcor_stars`, and the matched
    (Alt, Az, x, y) tuples are pooled. A single global least-squares fit over the
    pooled set yields the refined (xshift, yshift, rotation, radial_coeffs).

    Returns a dict with the fitted parameters plus ``n_matched``,
    ``residual_rms``, and per-match arrays (``alt``, ``az``, ``x``, ``y``) for
    diagnostics.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    dark = select_dark_frames(files, sun_alt_max=sun_alt_max)
    if max_frames is not None:
        dark = dark[:max_frames]

    init = dict(xshift=ALCOR_XSHIFT, yshift=ALCOR_YSHIFT, rotation=0.0,
                radial_coeffs=(1.0, 0.0, 0.0))
    pooled_alt, pooled_az, pooled_x, pooled_y = [], [], [], []

    for f in dark:
        time = _frame_time(f)
        im, _ = load_alcor_fits(f)
        cat = alcor_reference_altaz(time, vmag_limit=vmag_limit, min_alt=min_alt)
        det = detect_alcor_stars(im, fwhm=fwhm, threshold_sigma=threshold_sigma)
        if len(cat) < 3 or len(det) < 3:
            continue
        matched = match_alcor_stars(cat, det, init_params=init, z_steps=z_steps,
                                    tolerance=tolerance)
        if len(matched) == 0:
            continue
        pooled_alt.append(np.asarray(matched["Alt"], dtype=float))
        pooled_az.append(np.asarray(matched["Az"], dtype=float))
        pooled_x.append(np.asarray(matched["xcentroid"], dtype=float))
        pooled_y.append(np.asarray(matched["ycentroid"], dtype=float))

    if not pooled_alt:
        raise RuntimeError("No matched stars across the selected frames.")

    alt = np.concatenate(pooled_alt)
    az = np.concatenate(pooled_az)
    x = np.concatenate(pooled_x)
    y = np.concatenate(pooled_y)

    params = _fit_params(alt, az, x, y, init_params=init)

    # Final residuals (with outlier rejection at 3*MAD, then refit).
    px, py = _predict_pixels(alt, az, xshift=params["xshift"], yshift=params["yshift"],
                             rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]))
    resid = np.hypot(px - x, py - y)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    good = resid < np.median(resid) + 3.0 * 1.4826 * mad
    params = _fit_params(alt[good], az[good], x[good], y[good], init_params=params)
    px, py = _predict_pixels(alt[good], az[good], xshift=params["xshift"],
                             yshift=params["yshift"], rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]))
    rms = float(np.sqrt(np.mean((px - x[good]) ** 2 + (py - y[good]) ** 2)))

    return {
        **params,
        "n_matched": int(good.sum()),
        "residual_rms": rms,
        "alt": alt[good], "az": az[good], "x": x[good], "y": y[good],
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k "fit_params_recovers or aggregates_synthetic" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_wcs.py
git commit -m "add fit_alcor_wcs night-aggregating calibration"
```

---

## Task 8: CLI `fit_alcor_wcs_cli` + diagnostics + pyproject entry

**Files:**
- Modify: `skycam_utils/alcor.py` (add CLI at end of file, near the other `_cli` functions ~line 533+)
- Modify: `pyproject.toml` (line 46-51 `[project.scripts]`)
- Test: `skycam_utils/tests/test_alcor_wcs.py`

- [ ] **Step 1: Add failing test**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
from skycam_utils.alcor import save_alcor_residual_plot


def test_save_alcor_residual_plot_writes_output(tmp_path):
    rng = np.random.default_rng(3)
    alt = rng.uniform(5.0, 88.0, 100)
    az = rng.uniform(0.0, 360.0, 100)
    params = dict(xshift=2.0, yshift=-1.0, rotation=0.3, radial_coeffs=(1.0, 0.02, 0.03))
    x, y = _predict_pixels(alt, az, **params)
    x = x + rng.normal(0, 0.2, x.shape)
    y = y + rng.normal(0, 0.2, y.shape)

    out = save_alcor_residual_plot(alt, az, x, y, params, tmp_path / "resid.png")
    assert out.exists()
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k residual_plot -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement diagnostics + CLI**

Add `save_alcor_residual_plot` after `fit_alcor_wcs`:

```python
def save_alcor_residual_plot(alt, az, obs_x, obs_y, params, output_file,
                             radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                             figsize=(10, 5), dpi=150):
    """
    Plot pixel-residual magnitude versus zenith angle for matched stars, before
    (idealized) and after (fitted) the refinement. Returns the output path.
    """
    alt = np.asarray(alt, dtype=float)
    z = 90.0 - alt
    obs_x = np.asarray(obs_x, dtype=float)
    obs_y = np.asarray(obs_y, dtype=float)

    ix, iy = _predict_pixels(alt, az, xshift=0.0, yshift=0.0, rotation=0.0,
                             radial_coeffs=(1.0, 0.0, 0.0), radius=radius,
                             horizon_radius=horizon_radius)
    fx, fy = _predict_pixels(alt, az, xshift=params["xshift"], yshift=params["yshift"],
                             rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]),
                             radius=radius, horizon_radius=horizon_radius)
    before = np.hypot(ix - obs_x, iy - obs_y)
    after = np.hypot(fx - obs_x, fy - obs_y)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(z, before, s=8, alpha=0.5, label="idealized")
    ax.scatter(z, after, s=8, alpha=0.5, label="refined")
    ax.set_xlabel("zenith angle (deg)")
    ax.set_ylabel("pixel residual")
    ax.legend()
    output_file = Path(output_file)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_file


def fit_alcor_wcs_cli():
    """
    CLI entry point for ``fit_alcor_wcs``. Aggregates bright-star matches across
    the dark-sky frames of a night and prints the refined geometry constants
    ready to paste into the module defaults.
    """
    parser = argparse.ArgumentParser(
        description="Calibrate the alcor lens WCS from bright stars across a night."
    )
    parser.add_argument("input_dir", help="Directory containing alcor FITS images.")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob pattern for input files.")
    parser.add_argument("--vmag-limit", type=float, default=3.0, help="Faintest Vmag to use.")
    parser.add_argument("--sun-alt-max", type=float, default=-18.0,
                        help="Use frames with Sun altitude below this (deg).")
    parser.add_argument("--min-alt", type=float, default=10.0, help="Minimum star altitude (deg).")
    parser.add_argument("--tolerance", type=float, default=12.0, help="Match tolerance (pixels).")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap number of frames used.")
    parser.add_argument("--residual-plot", default=None, help="Optional residual-vs-zenith PNG path.")
    args = parser.parse_args()

    result = fit_alcor_wcs(
        args.input_dir, pattern=args.pattern, vmag_limit=args.vmag_limit,
        sun_alt_max=args.sun_alt_max, min_alt=args.min_alt, tolerance=args.tolerance,
        max_frames=args.max_frames,
    )
    print(f"# matched stars: {result['n_matched']}")
    print(f"# residual RMS (pix): {result['residual_rms']:.3f}")
    print(f"ALCOR_XSHIFT = {result['xshift']!r}")
    print(f"ALCOR_YSHIFT = {result['yshift']!r}")
    print(f"ALCOR_ROTATION = {ALCOR_ROTATION + result['rotation']!r}")
    print(f"ALCOR_RADIAL_COEFFS = {tuple(result['radial_coeffs'])!r}")
    if args.residual_plot is not None:
        out = save_alcor_residual_plot(result["alt"], result["az"], result["x"],
                                       result["y"], result, args.residual_plot)
        print(out)
```

> Note the rotation printout: `fit_alcor_wcs` fits the residual rotation relative to the image being rotated by the current `ALCOR_ROTATION`; the baked value is `ALCOR_ROTATION + fitted_rotation`.

- [ ] **Step 4: Add the pyproject script entry**

In `pyproject.toml`, under `[project.scripts]` (after line 51), add:

```toml
fit_alcor_wcs = "skycam_utils.alcor:fit_alcor_wcs_cli"
```

- [ ] **Step 5: Run to verify pass + confirm CLI imports**

Run: `pytest skycam_utils/tests/test_alcor_wcs.py -k residual_plot -v`
Expected: PASS.

Run: `python -c "from skycam_utils.alcor import fit_alcor_wcs_cli; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py pyproject.toml skycam_utils/tests/test_alcor_wcs.py
git commit -m "add fit_alcor_wcs CLI and residual diagnostics plot"
```

---

## Task 9: Run real calibration, bake constants, finalize tests & docs

**Files:**
- Modify: `skycam_utils/alcor.py` (constants `ALCOR_ROTATION`, `ALCOR_XSHIFT`, `ALCOR_YSHIFT`, `ALCOR_RADIAL_COEFFS`)
- Modify: `skycam_utils/tests/test_alcor.py` (two WCS-value assertions)
- Modify: `skycam_utils/tests/test_alcor_wcs.py` (add a fixture-based round-trip test)
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the calibration on the provided night**

Run:
```bash
pip install -e ".[test]"
fit_alcor_wcs /Users/tim/MMT/skycam_data/2024-09-04 --pattern "*.fits.bz2" \
    --residual-plot /tmp/alcor_resid.png --max-frames 60
```
Expected: prints `# matched stars: N` (expect dozens–hundreds), a residual RMS, and the four `ALCOR_*` constant lines. Inspect `/tmp/alcor_resid.png`: the "refined" residuals should be markedly smaller than "idealized" and should no longer trend upward with zenith angle.

**Sanity / sign check:** if `residual RMS` is large (> a few px) or the refined cloud is not clearly below the idealized one, the azimuth/shift sign conventions need flipping. Re-run on a single frame for debugging:
```bash
python -c "
from skycam_utils.alcor import fit_alcor_wcs
r = fit_alcor_wcs('/Users/tim/MMT/skycam_data/2024-09-04', max_frames=3)
print(r['n_matched'], r['residual_rms'], r['xshift'], r['yshift'], r['rotation'], r['radial_coeffs'])
"
```
If residuals are dominated by a constant offset, the `xshift`/`yshift` sign in `load_alcor_fits`'s `ndimage_shift` call (`shift=(-yshift, -xshift, 0.0)`) is the place to verify; if by a rotation, confirm the `ALCOR_ROTATION + fitted_rotation` combination. Adjust, re-run, and confirm RMS drops before baking.

Once the full night looks good, re-run without `--max-frames` for the final numbers.

- [ ] **Step 2: Bake the printed constants**

Edit `skycam_utils/alcor.py`, replacing the placeholder constant values with the printed ones, e.g.:

```python
ALCOR_ROTATION = <printed value>
ALCOR_XSHIFT = <printed value>
ALCOR_YSHIFT = <printed value>
ALCOR_RADIAL_COEFFS = (1.0, <printed k3>, <printed k5>)
```

- [ ] **Step 3: Update the two existing WCS-value tests**

In `skycam_utils/tests/test_alcor.py`:

- `test_load_alcor_fits_returns_centered_rgb_image` (lines 48-51): the `cdelt` is now `90/(horizon_radius*k1)` with the baked `k1=1.0` (so `cdelt` is unchanged), `crpix` may drift sub-pixel from the SIP fit, and `wcs.sip` is now not None. Update to:

```python
    np.testing.assert_allclose(wcs.wcs.crpix, [680.5, 680.5], atol=1.0)
    np.testing.assert_allclose(wcs.wcs.crval, [0.0, 90.0])
    np.testing.assert_allclose(wcs.wcs.cdelt, [90.0 / 662.0, 90.0 / 662.0], rtol=1e-6)
    assert wcs.wcs.lonpole == 0.0
    assert wcs.sip is not None
```

- `test_load_alcor_fits_wcs_maps_zenith_and_horizon` (lines 54-66): the equidistant `radii == 662` assertion no longer holds. Replace the body with a round-trip + model-consistency check:

```python
def test_load_alcor_fits_wcs_maps_zenith_and_horizon(alcor_image_and_wcs):
    from skycam_utils.alcor import _predict_pixels

    _, wcs = alcor_image_and_wcs

    _, zenith_alt = wcs.pixel_to_world_values(679.5, 679.5)
    np.testing.assert_allclose(zenith_alt, 90.0, atol=0.05)

    az = np.array([0.0, 90.0, 180.0, 270.0])
    alt = np.array([20.0, 40.0, 60.0, 5.0])
    px, py = wcs.world_to_pixel_values(az, alt)
    mx, my = _predict_pixels(alt, az)
    np.testing.assert_allclose(px, mx, atol=0.2)
    np.testing.assert_allclose(py, my, atol=0.2)
```

- [ ] **Step 4: Add a fixture round-trip test**

Append to `skycam_utils/tests/test_alcor_wcs.py`:

```python
def test_load_alcor_fits_world_pixel_round_trip():
    test_fits = Path(__file__).with_name("test.fits.bz2")
    _, wcs = load_alcor_fits(test_fits)
    az = np.array([10.0, 100.0, 200.0, 300.0])
    alt = np.array([15.0, 35.0, 55.0, 75.0])
    px, py = wcs.world_to_pixel_values(az, alt)
    az2, alt2 = wcs.pixel_to_world_values(px, py)
    np.testing.assert_allclose(az2 % 360, az % 360, atol=0.05)
    np.testing.assert_allclose(alt2, alt, atol=0.05)
```

- [ ] **Step 5: Run the full test suite**

Run: `pytest skycam_utils/tests/ -v`
Expected: PASS (all of `test_alcor.py` and `test_alcor_wcs.py`).

- [ ] **Step 6: Update CLAUDE.md**

In `CLAUDE.md`, in the alcor bullet of the "Project context" section, add a sentence noting that `load_alcor_fits`'s ARC WCS is refined by a fitted non-linear radial term (plus center/rotation) calibrated by `fit_alcor_wcs()` against `bright_star_sloan.fits` (Vmag ≤ 3) over dark-sky frames, and add the `fit_alcor_wcs` CLI to the "Common commands" list:

```bash
fit_alcor_wcs <night-dir> [--pattern ...] [--residual-plot OUT.png] [--max-frames N]
#   Aggregates bright-star matches across Sun-below-(-18 deg) frames and prints
#   refined ALCOR_* geometry constants to bake into alcor.py.
```

- [ ] **Step 7: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor.py skycam_utils/tests/test_alcor_wcs.py CLAUDE.md
git commit -m "bake fitted alcor WCS constants and finalize tests/docs"
```

---

## Self-review notes

- **Spec coverage:** non-linear radial term (Task 1 model, Task 3 SIP, Task 7 fit); refine center + rotation (model params + Task 9 baking); Vmag ≤ 3 catalog with refraction (Task 5); Sun < −18° aggregation across the night (Tasks 2, 7); bootstrap-from-zenith matching + brightness tie-break (Task 6); baked-in coefficients in code (constants, Task 9); `fit_alcor_wcs` in `alcor.py` + CLI (Tasks 7-8); residual-vs-zenith validation + round-trip test (Tasks 8-9). Asterism-matching fallback is documented in the spec and not implemented unless residuals demand it (flagged in Task 9 Step 1).
- **Backward compatibility:** idealized defaults make Tasks 1-8 non-breaking; only Task 9 changes WCS values, and it updates the two affected existing assertions in the same commit.
- **Naming consistency:** `_predict_pixels`, `_fit_params`, `_frame_time`, `build_alcor_wcs`, `detect_alcor_stars`, `alcor_reference_altaz`, `match_alcor_stars`, `fit_alcor_wcs`, `save_alcor_residual_plot`, `fit_alcor_wcs_cli` — signatures are consistent across the tasks that call them.
- **Note on the spec:** the spec described center refinement via "refined crpix"; this plan instead recenters the zenith to the array center via `ndimage` shift (baked into `xshift`/`yshift`) so the keogram center-column-as-zenith-meridian and plot circle-at-center assumptions stay valid. Functionally equivalent; worth a one-line spec update if desired.
