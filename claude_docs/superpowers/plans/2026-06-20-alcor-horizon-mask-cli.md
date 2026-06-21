# Alcor Horizon-Mask Regeneration CLIs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the unpackaged horizon-mask regeneration scripts to two packaged CLIs (`alcor_median_stack` and `create_horizon_mask`) so the date-indexed Alcor horizon mask can be rebuilt with installed tools.

**Architecture:** Follow the existing `create_badpix_mask` pattern in `skycam_utils/alcor.py` — a pure builder, a thin I/O wrapper, and an argparse CLI for each of the two stages. The cloudy-night luminance median is a prebuilt FITS input to the mask builder; the SW→W undetected-star patch is accumulated on demand from optional `--phot-nights`, with a Sobel-only fallback. The algorithm moves out of the reference script into the library, and the script is slimmed to import it and render only the diagnostic figures.

**Tech Stack:** Python, NumPy, SciPy (`ndimage`, `interpolate`), scikit-image (`filters.sobel`, `morphology`), Astropy (`io.fits`, `wcs`, `time`), pandas, pytest.

## Global Constraints

- All new library code goes in `skycam_utils/alcor.py` (single module, mirroring every other Alcor entry point). Copy these verbatim from the spec:
  - Mask output filename: `alcor_horizon_YYYY-MM-DD.fits.gz`, `uint8` data, in the dir resolved by `_resolve_horizon_dir` (`$ALCOR_HORIZON_DIR` → packaged `skycam_utils/data/horizon/`).
  - Median output filename default: `<night-name>_median.fits`, 2-D `float32` luminance (R+G+B), raw-frame WCS in the header.
  - Builder defaults (keyword args at these values): `edge_pct=96.0`, `edge_dilate=1`, `open_radius=3`, `sector=(225.0, 270.0)`, `und_thr=0.5`, `und_mincount=15`, `rim_alt=1.5`, `rod_area_min=400`.
  - `alcor_median_stack` default `moon_alt_max=90.0` (moon cut disabled — a cloudy night is chosen by hand).
  - No new shipped data asset; the undetected patch is accumulated from `--phot-nights` only.
  - Commit directly to `main` (this repo uses no feature branches).
- Mask format, the `load_alcor_horizon_mask` loader, and the shipped `alcor_horizon_2026-02-18.fits.gz` are **unchanged**.
- `AGENTS.md` is a symlink to `CLAUDE.md`; edit `CLAUDE.md` only.

---

## File Structure

- **`skycam_utils/alcor.py`** (modify): add 4 imports; add `build_alcor_luminance_median`, `alcor_median_stack`, `alcor_median_stack_cli`, `build_alcor_horizon_mask`, `_alcor_undetected_fraction`, `_horizon_epoch`, `create_horizon_mask`, `create_horizon_mask_cli` after `create_badpix_mask_cli` (currently ends at line 3724).
- **`skycam_utils/tests/test_alcor_horizon.py`** (modify): add unit tests for the luminance median, the horizon-mask builder, epoch parsing, and the `create_horizon_mask` I/O.
- **`pyproject.toml`** (modify): add two `[project.scripts]` entries.
- **`claude_docs/scripts/horizon_floodfill.py`** (rewrite): import the packaged builder; keep only the diagnostic plotting.
- **`docs/horizon_mask.rst`** (modify): package-data wording + the two-command regeneration path.
- **`CLAUDE.md`** (modify): add both CLIs; fix the "no packaged build CLI" line.

---

### Task 1: Luminance median builder + `alcor_median_stack` CLI

**Files:**
- Modify: `skycam_utils/alcor.py` (add after `create_badpix_mask_cli`, ~line 3724)
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `skycam_utils/tests/test_alcor_horizon.py`

**Interfaces:**
- Consumes: existing `select_dark_frames`, `_badpix_date_from_dir`, `load_alcor_badpix_mask`, `alcor_calibration`, `build_alcor_wcs` (all already in `alcor.py`).
- Produces:
  - `build_alcor_luminance_median(dark_files, badpix_mask=None, max_frames=None, scratch_dir=None, tile=50, log=None) -> np.ndarray` — 2-D `float32` `(ny, nx)` luminance median.
  - `alcor_median_stack(night_dir, out_path=None, sun_alt_max=-18.0, moon_alt_max=90.0, badpix=True, max_frames=None, scratch_dir=None, pattern="*.fits.bz2", log=None) -> pathlib.Path` — writes `<night-name>_median.fits`.
  - `alcor_median_stack_cli()` — entry point.

- [ ] **Step 1: Write the failing test**

Add to `skycam_utils/tests/test_alcor_horizon.py`:

```python
def test_build_luminance_median_collapses_and_rejects(tmp_path):
    from skycam_utils.alcor import build_alcor_luminance_median
    # 5 frames, shape (3, 4, 4); luminance is 30 everywhere (10+10+10).
    # One frame has an R-channel outlier at (0,0) -> luminance 1020 there.
    base = np.full((3, 4, 4), 10, dtype=np.int16)
    files = []
    for i, val in enumerate([10, 10, 10, 10, 1000]):
        frame = base.copy()
        frame[0, 0, 0] = val
        p = tmp_path / f"f{i}.fits"
        fits.PrimaryHDU(data=frame).writeto(p)
        files.append(p)

    med = build_alcor_luminance_median(files, scratch_dir=str(tmp_path))

    assert med.shape == (4, 4)               # channels collapsed
    assert med.dtype == np.float32
    assert med[0, 0] == 30.0                 # per-pixel median rejects the outlier frame
    assert med[2, 3] == 30.0                 # unchanged elsewhere
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_horizon.py::test_build_luminance_median_collapses_and_rejects -v`
Expected: FAIL with `ImportError: cannot import name 'build_alcor_luminance_median'`.

- [ ] **Step 3: Write the implementation**

Add to `skycam_utils/alcor.py` after `create_badpix_mask_cli` (the file's last function, ending ~line 3724):

```python
def build_alcor_luminance_median(dark_files, badpix_mask=None, max_frames=None,
                                 scratch_dir=None, tile=50, log=None):
    """
    Per-pixel 2-D luminance (R+G+B) median over raw alcor frames.

    Trail-free like :func:`build_alcor_median_stack`, but each frame's
    ``(3, ny, nx)`` cube is collapsed to an ``(ny, nx)`` int32 luminance plane
    (channel sum) before stacking, then the per-pixel median is taken in row
    tiles so peak memory stays small. When ``badpix_mask`` (a ``(3, ny, nx)``
    bool array) is given, its pixels are zeroed per channel before the sum so
    hot pixels don't survive into the median. ``max_frames`` strided-subsamples
    to cap runtime/scratch.

    Returns the luminance median as ``(ny, nx)`` float32.
    """
    files_ = list(dark_files)
    if max_frames is not None and len(files_) > max_frames:
        stride = len(files_) // max_frames
        files_ = files_[::stride][:max_frames]
    if not files_:
        raise ValueError("no frames provided")

    with fits.open(files_[0]) as hdul:
        shp = np.asarray(hdul[0].data).shape       # (3, ny, nx)
    nch, ny, nx = shp

    tmp = tempfile.NamedTemporaryFile(
        prefix="alcor_lum_", suffix=".dat",
        dir=scratch_dir or tempfile.gettempdir(), delete=False)
    tmp.close()
    memmap_path = Path(tmp.name)
    lum_mm = None
    try:
        lum_mm = np.memmap(memmap_path, dtype=np.int32, mode="w+",
                           shape=(len(files_), ny, nx))
        n = 0
        for f in files_:
            with fits.open(f) as hdul:
                data = np.asarray(hdul[0].data)
            if data.shape != shp:
                if log:
                    log(f"skip {Path(f).name}: shape {data.shape}")
                continue
            if badpix_mask is not None and badpix_mask.shape == data.shape:
                data = np.where(badpix_mask, 0, data)
            lum_mm[n] = data.astype(np.int32).sum(axis=0)
            n += 1
        lum_mm.flush()
        if n == 0:
            raise ValueError("no frames matched the reference shape")

        median = np.empty((ny, nx), dtype=np.float32)
        for r0 in range(0, ny, tile):
            r1 = min(r0 + tile, ny)
            slab = np.asarray(lum_mm[:n, r0:r1, :], dtype=np.float32)
            median[r0:r1, :] = np.median(slab, axis=0)
        return median
    finally:
        del lum_mm
        memmap_path.unlink(missing_ok=True)


def alcor_median_stack(night_dir, out_path=None, sun_alt_max=-18.0,
                       moon_alt_max=90.0, badpix=True, max_frames=None,
                       scratch_dir=None, pattern="*.fits.bz2", log=None):
    """
    Median-stack one (cloudy) night's frames into a luminance image.

    Selects dark frames (Sun < ``sun_alt_max``; ``moon_alt_max`` defaults to 90,
    i.e. the Moon cut is disabled because a cloudy night is chosen by hand),
    builds the luminance median (optionally zeroing the nearest bad-pixel mask),
    and writes ``<night-name>_median.fits`` carrying the raw-frame alt/az WCS for
    the night's epoch. Returns the output `~pathlib.Path`.
    """
    night_dir = Path(night_dir)
    frames = sorted(night_dir.glob(pattern))
    dark = select_dark_frames(frames, sun_alt_max=sun_alt_max,
                              moon_alt_max=moon_alt_max, log=log)
    if log:
        log(f"{len(dark)} dark frames of {len(frames)}")
    if not dark:
        raise ValueError("no dark frames selected")

    mdate = _badpix_date_from_dir(night_dir, dark)
    badpix_mask = None
    if badpix:
        badpix_mask, _ = load_alcor_badpix_mask(Time(mdate.isoformat() + "T12:00:00"))

    median = build_alcor_luminance_median(
        dark, badpix_mask=badpix_mask, max_frames=max_frames,
        scratch_dir=scratch_dir, log=log)

    cal = alcor_calibration(Time(mdate.isoformat() + "T12:00:00"))
    wcs = build_alcor_wcs(
        xcen=cal["xcen"], ycen=cal["ycen"], rotation=cal["rotation"],
        radial_coeffs=cal["radial_coeffs"], horizon_radius=cal["horizon_radius"],
        tangential_coeffs=cal["tangential_coeffs"], axis_tilt=cal["axis_tilt"])

    out_path = Path(str(out_path)) if out_path is not None \
        else Path(f"{night_dir.name}_median.fits")
    hdu = fits.PrimaryHDU(data=median, header=wcs.to_header(relax=True))
    hdu.header["NSTACK"] = (len(dark), "dark frames median-combined")
    hdu.header["LUM"] = ("R+G+B", "luminance sum of channels")
    hdu.writeto(out_path, overwrite=True)
    if log:
        log(f"wrote {out_path}")
    return out_path


def alcor_median_stack_cli():
    """CLI entry point for :func:`alcor_median_stack`."""
    parser = argparse.ArgumentParser(
        description="Median-stack a cloudy night's alcor frames into a luminance image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("night_dir", help="Directory of one (cloudy) night's frames.")
    parser.add_argument("-o", "--out", default=None,
                        help="Output FITS (default: <night-name>_median.fits).")
    parser.add_argument("--sun-alt-max", type=float, default=-18.0,
                        help="Use frames with Sun altitude below this (deg).")
    parser.add_argument("--moon-alt-max", type=float, default=90.0,
                        help="Use frames with Moon altitude below this (deg); 90 disables.")
    parser.add_argument("--no-badpix", action="store_true",
                        help="Do not zero the nearest bad-pixel mask before combining.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap frames used (strided) to bound runtime/scratch.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Directory for the temporary memmap (default: system temp).")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob for input frames.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print per-step progress messages.")
    args = parser.parse_args()

    log = None if args.quiet else (lambda m: print(m, file=sys.stderr))
    out = alcor_median_stack(
        args.night_dir, out_path=args.out, sun_alt_max=args.sun_alt_max,
        moon_alt_max=args.moon_alt_max, badpix=not args.no_badpix,
        max_frames=args.max_frames, scratch_dir=args.scratch_dir,
        pattern=args.pattern, log=log)
    print(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest skycam_utils/tests/test_alcor_horizon.py::test_build_luminance_median_collapses_and_rejects -v`
Expected: PASS.

- [ ] **Step 5: Register the CLI entry point**

In `pyproject.toml`, under `[project.scripts]`, add after the `create_badpix_mask` line:

```toml
alcor_median_stack = "skycam_utils.alcor:alcor_median_stack_cli"
```

- [ ] **Step 6: Verify the entry point and run the file's tests**

Run: `pip install -e ".[test]" -q && pytest skycam_utils/tests/test_alcor_horizon.py -v && alcor_median_stack --help`
Expected: all tests PASS; `--help` prints the usage with `--moon-alt-max` defaulting to `90.0`.

- [ ] **Step 7: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_horizon.py pyproject.toml
git commit -m "alcor: luminance median stack + alcor_median_stack CLI

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Horizon-mask pure builder

**Files:**
- Modify: `skycam_utils/alcor.py` (4 imports near the top; `build_alcor_horizon_mask` after the Task 1 functions)
- Test: `skycam_utils/tests/test_alcor_horizon.py`

**Interfaces:**
- Consumes: a built `astropy.wcs.WCS` (e.g. from `build_alcor_wcs`).
- Produces:
  - `build_alcor_horizon_mask(median_img, wcs, undetected=None, *, edge_pct=96.0, edge_dilate=1, open_radius=3, sector=(225.0, 270.0), und_thr=0.5, und_mincount=15, rim_alt=1.5, rod_area_min=400) -> np.ndarray` — 2-D bool `(ny, nx)` not-sky mask. `undetected`, when not `None`, is a callable `f(az, alt) -> (fraction, count)` (defined in Task 3); `None` means the SW→W sector is Sobel-only.

- [ ] **Step 1: Write the failing test**

Add to `skycam_utils/tests/test_alcor_horizon.py`:

```python
def test_build_horizon_mask_synthetic():
    from skycam_utils.alcor import build_alcor_horizon_mask, build_alcor_wcs

    nx = ny = 240
    xcen = ycen = 120.0
    hr = 110.0                                   # alt=0 at r=110 about the zenith
    wcs = build_alcor_wcs(xcen=xcen, ycen=ycen, rotation=0.0,
                          radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=hr)

    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - xcen, yy - ycen)
    img = np.full((ny, nx), 1000.0, dtype=np.float32)   # bright flat sky

    # Obstruction A: a dark finger from r~55 to the rim (reaches the horizon).
    finger = (xx - xcen > 55) & (np.abs(yy - ycen) < 14)
    img[finger] = 1.0
    # Obstruction B: a dark square fully enclosed in clear sky (lightning-rod analog).
    spike = (xx >= 60) & (xx <= 80) & (yy >= 110) & (yy <= 130)
    img[spike] = 1.0

    # open_radius=0 keeps the small synthetic features intact; rod_area_min low
    # so the enclosed square is retained by size.
    mask = build_alcor_horizon_mask(img, wcs, open_radius=0, rim_alt=1.5,
                                    rod_area_min=50)

    assert mask.shape == (ny, nx)
    assert mask.dtype == bool
    # everything at/below the horizon is masked
    assert mask[r > hr].all()
    # the open zenith and a clear-sky patch are NOT masked
    assert not mask[int(ycen), int(xcen)]
    assert not mask[80, 120]
    # both obstructions are masked
    assert mask[120, 200]                         # inside the rim finger
    assert mask[120, 70]                          # inside the enclosed square
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest skycam_utils/tests/test_alcor_horizon.py::test_build_horizon_mask_synthetic -v`
Expected: FAIL with `ImportError: cannot import name 'build_alcor_horizon_mask'`.

- [ ] **Step 3: Add the imports**

Near the top of `skycam_utils/alcor.py`, with the other scipy/scikit-image imports (after line 16, `from scipy.spatial import cKDTree`), add:

```python
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import sobel
from skimage.morphology import opening, disk
```

- [ ] **Step 4: Write the implementation**

Add to `skycam_utils/alcor.py` after `alcor_median_stack_cli`:

```python
def build_alcor_horizon_mask(median_img, wcs, undetected=None, *,
                             edge_pct=96.0, edge_dilate=1, open_radius=3,
                             sector=(225.0, 270.0), und_thr=0.5, und_mincount=15,
                             rim_alt=1.5, rod_area_min=400):
    """
    Build the raw-frame horizon (sky / not-sky) mask from a cloudy-night
    luminance median by flood-filling the sky against the 2-D Sobel edge map.

    Strong ``sobel(log10 median)`` edges (above the ``edge_pct`` percentile
    within the FOV, dilated ``edge_dilate`` times) are walls; the sky is
    flood-filled from the WCS zenith and everything unreachable is not-sky. In
    the ``sector`` azimuth range (the SW->W building sector, where the Sobel wall
    breaks up) an optional ``undetected`` sampler ``f(az, alt) -> (fraction,
    count)`` supplies extra walls where the undetected-star fraction is
    ``>= und_thr`` over ``>= und_mincount`` transits. A morphological opening of
    radius ``open_radius`` severs thin necks, then a connected-component pass
    keeps a not-sky blob only if it reaches the rim (``min_alt < rim_alt``) or is
    rod-sized (``size >= rod_area_min``). Returns a 2-D bool ``(ny, nx)`` array
    where ``True`` marks not-sky (obstructions above the horizon plus everything
    at/below altitude 0); valid sky is ``~mask``.
    """
    img = np.asarray(median_img, dtype=float)
    ny, nx = img.shape

    yy, xx = np.mgrid[0:ny, 0:nx]
    w = wcs.all_pix2world(np.column_stack([xx.ravel(), yy.ravel()]), 0)
    az = (w[:, 0] % 360.0).reshape(ny, nx)
    alt = w[:, 1].reshape(ny, nx)
    in_fov = alt > 0.0

    # 2-D Sobel edges on the log image (suppresses the smooth vignette gradient)
    E = sobel(ndi.gaussian_filter(np.log10(np.clip(img, 1, None)), 1.0))
    thr = np.percentile(E[in_fov], edge_pct)
    wall = (E > thr) & in_fov
    if edge_dilate:
        wall = ndi.binary_dilation(wall, iterations=edge_dilate)

    undet_obstruction = np.zeros((ny, nx), dtype=bool)
    if undetected is not None:
        fr_pix, ct_pix = undetected(az, alt)
        sec_lo, sec_hi = sector
        in_sector = (az >= sec_lo) & (az <= sec_hi)
        undet_obstruction = (in_sector & in_fov
                             & (fr_pix >= und_thr) & (ct_pix >= und_mincount))

    # flood-fill the sky: free = inside FOV, not a wall, not undetected-blocked
    free = in_fov & ~wall & ~undet_obstruction
    lbl, _ = ndi.label(free, structure=np.ones((3, 3)))
    zp = wcs.all_world2pix([[0.0, 90.0]], 0)[0]
    zx, zy = int(round(zp[0])), int(round(zp[1]))
    if not (0 <= zy < ny and 0 <= zx < nx and free[zy, zx]):  # nudge to nearest free
        fy, fx = np.where(free)
        k = np.argmin((fx - zx) ** 2 + (fy - zy) ** 2)
        zx, zy = int(fx[k]), int(fy[k])
    sky = lbl == lbl[zy, zx]
    notsky_raw = in_fov & ~sky

    notsky_open = opening(notsky_raw, disk(open_radius)) if open_radius else notsky_raw

    # drop spurious open-sky pockets: keep a not-sky blob only if it reaches the
    # rim or is large enough to be the lightning rod
    nlab, nn = ndi.label(notsky_open, structure=np.ones((3, 3)))
    if nn:
        size = np.bincount(nlab.ravel())[1:]
        min_alt = ndi.minimum(alt, nlab, index=np.arange(1, nn + 1))
        keep = (min_alt < rim_alt) | (size >= rod_area_min)
        notsky = np.concatenate([[False], keep])[nlab]
    else:
        notsky = notsky_open

    return (~in_fov) | notsky
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest skycam_utils/tests/test_alcor_horizon.py::test_build_horizon_mask_synthetic -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_horizon.py
git commit -m "alcor: horizon-mask flood-fill builder

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Undetected-fraction helper + `create_horizon_mask` CLI

**Files:**
- Modify: `skycam_utils/alcor.py` (`_alcor_undetected_fraction`, `_horizon_epoch`, `create_horizon_mask`, `create_horizon_mask_cli` after the Task 2 builder)
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `skycam_utils/tests/test_alcor_horizon.py`

**Interfaces:**
- Consumes: `build_alcor_horizon_mask` (Task 2), `build_alcor_wcs`, `alcor_calibration`, `_resolve_horizon_dir` (existing).
- Produces:
  - `_alcor_undetected_fraction(phot_nights, wcs, smooth=1.0) -> callable | None` — `f(az, alt) -> (fraction, count)` over identically-shaped arrays, or `None` if no `*_phot.csv` measurements were found.
  - `_horizon_epoch(epoch, median_path) -> datetime.date` — explicit `epoch` wins, else parse `YYYY-MM-DD`/`YYYY_MM_DD` from the median filename.
  - `create_horizon_mask(median_path, epoch=None, wcs=None, out_dir=None, phot_nights=None, edge_pct=96.0, edge_dilate=1, open_radius=3, sector=(225.0, 270.0), und_thr=0.5, und_mincount=15, rim_alt=1.5, rod_area_min=400, log=None) -> pathlib.Path` — writes `alcor_horizon_YYYY-MM-DD.fits.gz`. (`wcs=` overrides the epoch-built WCS; used for testing and for non-standard geometry.)
  - `create_horizon_mask_cli()` — entry point.

- [ ] **Step 1: Write the failing tests**

Add to `skycam_utils/tests/test_alcor_horizon.py`:

```python
def test_horizon_epoch_parsing():
    from skycam_utils.alcor import _horizon_epoch
    from pathlib import Path
    import pytest
    assert _horizon_epoch(None, Path("2026-02-18_median.fits")) == date(2026, 2, 18)
    assert _horizon_epoch(None, Path("2026_02_18_median.fits")) == date(2026, 2, 18)
    assert _horizon_epoch("2025-12-01", Path("anything.fits")) == date(2025, 12, 1)
    with pytest.raises(ValueError):
        _horizon_epoch(None, Path("median.fits"))


def test_create_horizon_mask_writes_dated_mask(tmp_path):
    from skycam_utils.alcor import create_horizon_mask, build_alcor_wcs
    nx = ny = 120
    img = np.full((ny, nx), 1000.0, dtype=np.float32)
    img[:, 95:] = 1.0                                  # a dark block to the +x rim
    median = tmp_path / "2026-02-18_median.fits"
    fits.PrimaryHDU(data=img).writeto(median)
    # pass an explicit small WCS so the build doesn't use the real 1411x1422 geometry
    wcs = build_alcor_wcs(xcen=60.0, ycen=60.0, rotation=0.0,
                          radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=55.0)

    out = create_horizon_mask(median, wcs=wcs, out_dir=str(tmp_path))

    assert out.name == "alcor_horizon_2026-02-18.fits.gz"   # epoch parsed from filename
    assert out.exists()
    data = fits.getdata(out)
    assert data.dtype == np.uint8
    hdr = fits.getheader(out)
    assert hdr["METHOD"] == "sobel-floodfill"
    assert hdr["NMASK"] + hdr["NSKY"] == ny * nx
    assert not hdr["UNDET"]                                  # no --phot-nights given
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest skycam_utils/tests/test_alcor_horizon.py::test_horizon_epoch_parsing skycam_utils/tests/test_alcor_horizon.py::test_create_horizon_mask_writes_dated_mask -v`
Expected: FAIL with `ImportError: cannot import name '_horizon_epoch'`.

- [ ] **Step 3: Write the implementation**

Add to `skycam_utils/alcor.py` after `build_alcor_horizon_mask`:

```python
def _alcor_undetected_fraction(phot_nights, wcs, smooth=1.0):
    """
    Accumulate the az/alt undetected-star fraction from fixed-position per-frame
    photometry over several nights, for the SW->W building sector wall.

    Reads ``xcen``/``ycen``/``flux_g_ap`` from every ``*_phot.csv`` under each
    directory in ``phot_nights`` (a star is "undetected" when ``flux_g_ap == 0``),
    projects to az/alt via ``wcs``, and bins into a 0.5-deg az/alt grid smoothed
    by ``smooth``. Returns a callable ``f(az, alt) -> (fraction, count)`` over
    identically-shaped arrays (altitude clipped into the grid), or ``None`` if no
    measurements were found.
    """
    xs, ys, det = [], [], []
    for ddir in phot_nights:
        for f in sorted(Path(ddir).glob("*_phot.csv")):
            try:
                d = pd.read_csv(f, usecols=["xcen", "ycen", "flux_g_ap"])
            except Exception:
                continue
            xs.append(d["xcen"].to_numpy())
            ys.append(d["ycen"].to_numpy())
            det.append((d["flux_g_ap"] > 0).to_numpy())
    if not xs:
        return None

    px, py = np.concatenate(xs), np.concatenate(ys)
    ap_det = np.concatenate(det)
    sw = wcs.all_pix2world(np.column_stack([px, py]), 0)
    saz, salt = sw[:, 0] % 360.0, sw[:, 1]
    undet = ~ap_det
    ok = np.isfinite(saz) & np.isfinite(salt)
    saz, salt, undet = saz[ok], salt[ok], undet[ok]

    az_e = np.arange(-0.25, 360.0, 0.5)
    alt_e = np.arange(-6.25, 30.26, 0.5)
    az_c = 0.5 * (az_e[:-1] + az_e[1:])
    alt_c = 0.5 * (alt_e[:-1] + alt_e[1:])
    Htot, _, _ = np.histogram2d(saz, salt, bins=[az_e, alt_e])
    Hund, _, _ = np.histogram2d(saz[undet], salt[undet], bins=[az_e, alt_e])
    frac = ndi.gaussian_filter(Hund / np.maximum(Htot, 1), (smooth, smooth))
    fr_i = RegularGridInterpolator((az_c, alt_c), frac,
                                   bounds_error=False, fill_value=0.0)
    ct_i = RegularGridInterpolator((az_c, alt_c), Htot,
                                   bounds_error=False, fill_value=0.0)

    def sample(az, alt):
        shape = np.shape(az)
        a = np.clip(np.asarray(alt, dtype=float), alt_c[0], alt_c[-1])
        pts = np.column_stack([np.asarray(az, dtype=float).ravel(), a.ravel()])
        return fr_i(pts).reshape(shape), ct_i(pts).reshape(shape)

    return sample


def _horizon_epoch(epoch, median_path):
    """
    Resolve the horizon-mask epoch date: explicit ``epoch`` (a ``date`` or a
    ``YYYY-MM-DD``/``YYYY_MM_DD`` string) wins, else parse the date from the
    median filename. Raises ``ValueError`` if neither yields a date.
    """
    if isinstance(epoch, date):
        return epoch
    if epoch is not None:
        m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", str(epoch))
        if not m:
            raise ValueError(f"cannot parse epoch {epoch!r}; use YYYY-MM-DD")
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{4})[-_](\d{2})[-_](\d{2})", Path(median_path).name)
    if not m:
        raise ValueError(
            f"cannot determine epoch from median filename "
            f"{Path(median_path).name!r}; pass epoch=YYYY-MM-DD")
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))


def create_horizon_mask(median_path, epoch=None, wcs=None, out_dir=None,
                        phot_nights=None, edge_pct=96.0, edge_dilate=1,
                        open_radius=3, sector=(225.0, 270.0), und_thr=0.5,
                        und_mincount=15, rim_alt=1.5, rod_area_min=400, log=None):
    """
    Build and write a date-stamped horizon mask from a cloudy-night luminance
    median.

    Resolves the epoch (``epoch`` or the median filename), builds the raw-frame
    WCS for that epoch (unless ``wcs`` is given), optionally accumulates the
    SW->W undetected-star patch from ``phot_nights`` (a list of dirs of
    ``*_phot.csv``; omitted -> that sector is Sobel-only), builds the mask, and
    writes ``alcor_horizon_YYYY-MM-DD.fits.gz`` to ``out_dir`` (default: the
    resolved horizon directory). Returns the output `~pathlib.Path`.
    """
    median_path = Path(median_path)
    img = np.asarray(fits.getdata(median_path), dtype=float)
    if img.ndim != 2:
        raise ValueError(f"median {median_path.name} is not a 2-D luminance image")

    mdate = _horizon_epoch(epoch, median_path)
    if wcs is None:
        cal = alcor_calibration(Time(mdate.isoformat() + "T12:00:00"))
        wcs = build_alcor_wcs(
            xcen=cal["xcen"], ycen=cal["ycen"], rotation=cal["rotation"],
            radial_coeffs=cal["radial_coeffs"], horizon_radius=cal["horizon_radius"],
            tangential_coeffs=cal["tangential_coeffs"], axis_tilt=cal["axis_tilt"])

    undetected = None
    if phot_nights:
        undetected = _alcor_undetected_fraction(phot_nights, wcs)
        if undetected is None and log:
            log("no *_phot.csv measurements found; SW->W sector uses Sobel edges only")
    elif log:
        log("no phot_nights given; SW->W sector uses Sobel edges only")

    mask = build_alcor_horizon_mask(
        img, wcs, undetected=undetected, edge_pct=edge_pct,
        edge_dilate=edge_dilate, open_radius=open_radius, sector=sector,
        und_thr=und_thr, und_mincount=und_mincount, rim_alt=rim_alt,
        rod_area_min=rod_area_min)

    out_dir = Path(str(out_dir)) if out_dir is not None else _resolve_horizon_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alcor_horizon_{mdate.isoformat()}.fits.gz"

    hdu = fits.PrimaryHDU(data=mask.astype(np.uint8))
    hdu.header["METHOD"] = ("sobel-floodfill", "horizon mask construction")
    hdu.header["SRCMED"] = (median_path.name, "source median image")
    hdu.header["EDGEPCT"] = (edge_pct, "Sobel-edge wall percentile")
    hdu.header["OPENR"] = (open_radius, "morphological opening radius (px)")
    hdu.header["ALTCUT"] = (0.0, "altitude cutoff (deg); <= is masked")
    hdu.header["UNDET"] = (bool(undetected is not None), "SW->W undetected patch used")
    hdu.header["NMASK"] = (int(mask.sum()), "masked (not-sky) pixels")
    hdu.header["NSKY"] = (int((~mask).sum()), "valid-sky pixels")
    hdu.writeto(out_path, overwrite=True)
    if log:
        log(f"sky px {int((~mask).sum())}  not-sky {int(mask.sum())} "
            f"({100 * mask.sum() / mask.size:.1f}% of frame)")
        log(f"wrote {out_path}")
    return out_path


def create_horizon_mask_cli():
    """CLI entry point for :func:`create_horizon_mask`."""
    parser = argparse.ArgumentParser(
        description="Build a date-stamped alcor horizon (sky/not-sky) mask from a cloudy-night median.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("median", help="Cloudy-night luminance median FITS (from alcor_median_stack).")
    parser.add_argument("--epoch", default=None,
                        help="Mask date YYYY-MM-DD (default: parsed from the median filename).")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: $ALCOR_HORIZON_DIR or packaged data/horizon).")
    parser.add_argument("--phot-nights", nargs="+", default=None,
                        help="Dirs of *_phot.csv for the SW->W undetected-star patch "
                             "(omitted: that sector uses Sobel edges only).")
    parser.add_argument("--edge-pct", type=float, default=96.0, help="Sobel-edge wall percentile.")
    parser.add_argument("--edge-dilate", type=int, default=1, help="Wall dilation iterations.")
    parser.add_argument("--open-radius", type=int, default=3, help="Morphological opening radius (px).")
    parser.add_argument("--sector", type=float, nargs=2, default=[225.0, 270.0],
                        metavar=("AZ_LO", "AZ_HI"), help="Undetected-patch azimuth sector (deg).")
    parser.add_argument("--und-thr", type=float, default=0.5, help="Undetected fraction = obstructed.")
    parser.add_argument("--und-mincount", type=int, default=15, help="Min star transits per cell.")
    parser.add_argument("--rim-alt", type=float, default=1.5,
                        help="A not-sky blob reaching below this alt is rim-connected (kept).")
    parser.add_argument("--rod-area-min", type=int, default=400,
                        help="Keep isolated not-sky blobs at least this size (px): the lightning rod.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print per-step progress messages.")
    args = parser.parse_args()

    log = None if args.quiet else (lambda m: print(m, file=sys.stderr))
    out = create_horizon_mask(
        args.median, epoch=args.epoch, out_dir=args.out_dir,
        phot_nights=args.phot_nights, edge_pct=args.edge_pct,
        edge_dilate=args.edge_dilate, open_radius=args.open_radius,
        sector=tuple(args.sector), und_thr=args.und_thr,
        und_mincount=args.und_mincount, rim_alt=args.rim_alt,
        rod_area_min=args.rod_area_min, log=log)
    print(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest skycam_utils/tests/test_alcor_horizon.py::test_horizon_epoch_parsing skycam_utils/tests/test_alcor_horizon.py::test_create_horizon_mask_writes_dated_mask -v`
Expected: PASS.

- [ ] **Step 5: Register the CLI entry point**

In `pyproject.toml`, under `[project.scripts]`, add after the `alcor_median_stack` line from Task 1:

```toml
create_horizon_mask = "skycam_utils.alcor:create_horizon_mask_cli"
```

- [ ] **Step 6: Verify the entry point and the whole horizon test file**

Run: `pip install -e ".[test]" -q && pytest skycam_utils/tests/test_alcor_horizon.py -v && create_horizon_mask --help`
Expected: all tests PASS; `--help` prints the usage including `--phot-nights`.

- [ ] **Step 7: Commit**

```bash
git add skycam_utils/alcor.py skycam_utils/tests/test_alcor_horizon.py pyproject.toml
git commit -m "alcor: create_horizon_mask CLI + undetected-star patch helper

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Slim the reference script + update docs and CLAUDE.md

**Files:**
- Rewrite: `claude_docs/scripts/horizon_floodfill.py`
- Modify: `docs/horizon_mask.rst`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `build_alcor_horizon_mask`, `_alcor_undetected_fraction`, `build_alcor_wcs`, `alcor_calibration` (all from Tasks 1–3).
- Produces: no new API. The script becomes a thin figure renderer; the docs reference the new CLIs.

- [ ] **Step 1: Rewrite the reference script to import the packaged builder**

Replace the body of `claude_docs/scripts/horizon_floodfill.py` so the algorithm is the packaged function and only the diagnostic plotting remains. The new file:

```python
#!/usr/bin/env python
"""Diagnostic figures for the Alcor horizon mask.

The mask algorithm now lives in the package
(``skycam_utils.alcor.build_alcor_horizon_mask`` + ``_alcor_undetected_fraction``);
regenerate the shipped mask with the ``create_horizon_mask`` CLI. This script
only re-renders the two diagnostic figures used in the docs from local data.
Not part of the installed package.
"""
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ZScaleInterval
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skycam_utils.alcor import (alcor_calibration, build_alcor_wcs,
                                build_alcor_horizon_mask,
                                _alcor_undetected_fraction)

GP = Path(__file__).resolve().parent.parent / "gplots"
MEDIAN = GP / "2026-02-18_median.fits"
EPOCH = "2026-02-18"
SEC_LO, SEC_HI = 225.0, 270.0

# nights whose per-frame photometry feeds the undetected-star patch
NIGHTS = [
    os.path.expanduser("~/MMT/skycam_data/2024-09-04"),
    "/Volumes/Samsung_4TB/skycam/2026-01-11",
    "/Volumes/Samsung_4TB/skycam/2026-03-11",
    "/Volumes/Samsung_4TB/skycam/2026-05-18",
    "/Volumes/Samsung_4TB/skycam/2026-06-09",
]


def main():
    cal = alcor_calibration(Time(f"{EPOCH}T12:00:00"))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"], radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])
    img = fits.getdata(MEDIAN).astype(float)
    ny, nx = img.shape

    undetected = _alcor_undetected_fraction(NIGHTS, wcs)
    horizon_mask = build_alcor_horizon_mask(
        img, wcs, undetected=undetected, sector=(SEC_LO, SEC_HI))

    vmin, vmax = ZScaleInterval().get_limits(img)
    fig, ax = plt.subplots(1, 1, figsize=(11, 10))
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ov = np.zeros((ny, nx, 4))
    ov[horizon_mask] = (1, 0, 0, 0.32)
    ax.imshow(ov, origin="lower")
    for a in (SEC_LO, SEC_HI):
        ra = np.linspace(0.5, 25, 60)
        rx, ry = wcs.world_to_pixel_values(np.full_like(ra, a), ra)
        ax.plot(rx, ry, ":", color="cyan", lw=1.0, alpha=0.8)
    ax.set_title("complete horizon mask (red=not-sky) on 2026-02-18 median; "
                 "dotted = SW->W undetected sector")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    out = GP / "horizon_floodfill.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Byte-compile the script to confirm it parses**

Run: `python -m py_compile claude_docs/scripts/horizon_floodfill.py && echo OK`
Expected: `OK` (the script needs local data to *run*, but must parse and import cleanly).

- [ ] **Step 3: Update `docs/horizon_mask.rst`**

In the **Properties** list, replace the "Date-resolved" bullet:

```rst
- **Date-resolved** — loaded by nearest date (overridable with
  ``$ALCOR_HORIZON_DIR``) from
  ``skycam_utils/data/horizon/alcor_horizon_YYYY-MM-DD.fits.gz``, exactly like
  the calibration and bad-pixel assets. It is stable across epochs for the same
  reason the geometry is (see :doc:`wcs_calibration`): one epoch covers
  2024–2026, and a new epoch is added only if the camera moves.
```

with:

```rst
- **Date-resolved** — the masks ship as **package data** and are resolved
  automatically from the installed package by
  :func:`~skycam_utils.alcor.load_alcor_horizon_mask` (nearest date, overridable
  with ``$ALCOR_HORIZON_DIR``); callers never reference a repository path. The
  mask is stable across epochs for the same reason the geometry is (see
  :doc:`wcs_calibration`): one epoch covers 2024–2026, and a new epoch is added
  only if the camera moves.
```

Then replace the entire **"How the mask is built"** intro paragraph:

```rst
There is **no packaged build CLI** — by design, the regeneration path is the
reference script ``claude_docs/scripts/horizon_floodfill.py``, run offline. The
method is a **Sobel-edge flood-fill of a cloudy-night median**:
```

with:

```rst
The mask is regenerated with two packaged CLIs (the reference script
``claude_docs/scripts/horizon_floodfill.py`` now only re-renders the diagnostic
figures below). First median-stack a **cloudy/overcast** night, then flood-fill
the obstructions:

.. code-block:: bash

   # 1. Build the cloudy-night luminance median (Sobel walls come from its edges)
   alcor_median_stack <skycam_datadir>/2026-02-18 -o 2026-02-18_median.fits

   # 2. Flood-fill the horizon mask; --phot-nights supplies the SW->W patch
   create_horizon_mask 2026-02-18_median.fits \
       --phot-nights <skycam_datadir>/2026-01-11 <skycam_datadir>/2026-05-18

Like ``create_badpix_mask``, both require local raw data and write to the
packaged ``data/horizon/`` directory (or ``$ALCOR_HORIZON_DIR``). The method is a
**Sobel-edge flood-fill of a cloudy-night median**:
```

- [ ] **Step 4: Update `CLAUDE.md`**

In the horizon-mask paragraph, replace:

```
There is **no packaged build CLI** — the design choice was that the scripts are the regeneration path. It is rebuilt offline by `claude_docs/scripts/horizon_floodfill.py`:
```

with:

```
It is rebuilt by the packaged `alcor_median_stack` + `create_horizon_mask` CLIs (the reference script `claude_docs/scripts/horizon_floodfill.py` now only re-renders the diagnostic figures). The method:
```

Then, in the `## Common commands` block, add after the `create_badpix_mask` entry:

```
alcor_median_stack <cloudy-night-dir> [-o OUT.fits] [--sun-alt-max -18] [--moon-alt-max 90] [--no-badpix] [--max-frames N] [--scratch-dir DIR] [--pattern *.fits.bz2] [--quiet]
#   Median-stacks one CLOUDY night's frames into a 2-D luminance (R+G+B) FITS with
#   the raw-frame alt/az WCS in the header — the smooth-overcast input whose Sobel
#   edges define the horizon obstructions. Dark-frame selection (Sun < --sun-alt-max);
#   the Moon cut defaults OFF (--moon-alt-max 90) since the cloudy night is chosen by
#   hand. Zeros the nearest bad-pixel mask unless --no-badpix. Default out:
#   <night-name>_median.fits.

create_horizon_mask <median.fits> [--epoch YYYY-MM-DD] [--out-dir DIR] [--phot-nights DIR ...] [--edge-pct 96] [--open-radius 3] [--sector 225 270] [--und-thr 0.5] [--und-mincount 15] [--rim-alt 1.5] [--rod-area-min 400] [--quiet]
#   Sobel-edge flood-fill of the cloudy-night median into alcor_horizon_YYYY-MM-DD.fits.gz
#   (written to --out-dir, default $ALCOR_HORIZON_DIR or packaged data/horizon). Epoch is
#   parsed from the median filename unless --epoch is given; the WCS is built from the
#   nearest calibration. --phot-nights DIRS accumulates the SW->W (az 225-270) undetected-
#   star patch from each dir's *_phot.csv; omit it and that sector falls back to Sobel-only.
#   Consumed by load_alcor_horizon_mask, resolved nearest-in-date. Like create_badpix_mask
#   it needs local raw data and is not reproducible from a bare pip install.
```

- [ ] **Step 5: Build the docs to confirm the RST is clean**

Run: `pip install -e ".[docs]" -q && sphinx-build -W -b html docs docs/_build/html 2>&1 | tail -5`
Expected: `build succeeded.` with no warnings promoted to errors.

- [ ] **Step 6: Commit**

```bash
git add claude_docs/scripts/horizon_floodfill.py docs/horizon_mask.rst CLAUDE.md
git commit -m "docs: packaged horizon-mask CLIs; slim the reference script

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Run the whole suite**

Run: `pytest -q`
Expected: all tests pass (the Alcor suite plus the new horizon tests).

- [ ] **Confirm both entry points resolve**

Run: `alcor_median_stack --help && create_horizon_mask --help`
Expected: both print usage without error.

---

## Self-Review notes (resolved during planning)

- **Spec coverage:** median CLI (Task 1), horizon builder (Task 2), undetected patch + `create_horizon_mask` + epoch parsing (Task 3), reference-script slim + docs + CLAUDE.md (Task 4), tests (Tasks 1–3), `pyproject` scripts (Tasks 1, 3). All spec sections map to a task.
- **One addition beyond the spec:** `create_horizon_mask` gains an optional `wcs=` override (defaults to the epoch-built WCS). It keeps the I/O test fast (a tiny synthetic WCS instead of the real 1411×1422 geometry) and is a natural parallel to `load_alcor_fits(wcs=...)`. No behavior change when omitted.
- **Type consistency:** the `undetected` value produced by `_alcor_undetected_fraction` (Task 3) is the callable `f(az, alt) -> (fraction, count)` consumed by `build_alcor_horizon_mask` (Task 2); names and shapes match across tasks.
