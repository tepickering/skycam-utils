"""Building bad-pixel masks from a night-median stack."""

import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy import ndimage
from astropy.io import fits
from astropy.time import Time

from ..astrometry import MMT_LOCATION

from .config import (
    ALCOR_BADPIX_POLE_RADIUS, ALCOR_BADPIX_RIM_DILATION, alcor_calibration
)
from .timeutils import select_dark_frames
from .wcs import build_alcor_wcs
from .masks import (
    _badpix_date_from_dir, _resolve_badpix_dir, load_alcor_horizon_mask
)


def alcor_badpix_search_region(shape, time=None, wcs=None, horizon_dir=None,
                               rim_dilation=ALCOR_BADPIX_RIM_DILATION,
                               pole_radius=ALCOR_BADPIX_POLE_RADIUS, log=None):
    """
    The sky region within which hot-pixel detection is trustworthy.

    Returns an ``(ny, nx)`` bool array, True where a pixel may be considered a
    bad-pixel candidate. Two regions are excluded, both for the same underlying
    reason -- the detector's premise is a sharp spike on a smooth, trail-free
    background, and neither region satisfies it:

    * **Not-sky**, from :func:`load_alcor_horizon_mask` for ``time``, dilated by
      ``rim_dilation`` pixels so the horizon edge itself goes too. Terrain,
      buildings and ground lights are sharp structure, and the rim is a step;
      the high-pass fires on all of them.
    * **The north celestial pole**, a disc of ``pole_radius`` pixels centered on
      the pole's pixel from ``wcs``. Stars there barely move, so a night median
      keeps their trails -- Polaris in particular.

    ``time`` selects the horizon-mask epoch (any `~astropy.time.Time`,
    ``datetime`` or ``date``); passing ``None`` skips the horizon cut.
    ``horizon_dir`` overrides where those masks are read from -- note this is the
    *horizon* mask directory, not the ``masks_dir`` (bad-pixel) argument used
    elsewhere in this module. ``wcs``
    is the raw-frame alt/az WCS; passing ``None`` skips the pole cut. Setting
    either radius to 0 disables that exclusion.
    """
    ny, nx = shape
    valid = np.ones((ny, nx), dtype=bool)

    if time is not None:
        horizon, horizon_date = load_alcor_horizon_mask(time, masks_dir=horizon_dir)
        if horizon is None:
            if log:
                log("no horizon mask found; searching the whole frame")
        elif horizon.shape != (ny, nx):
            if log:
                log(f"horizon mask shape {horizon.shape} does not match "
                    f"{(ny, nx)}; not applied")
        else:
            not_sky = horizon
            if rim_dilation > 0:
                not_sky = ndimage.binary_dilation(not_sky, iterations=int(rim_dilation))
            valid &= ~not_sky
            if log:
                log(f"horizon mask {horizon_date} (dilated {rim_dilation} px): "
                    f"{int(not_sky.sum())} of {ny * nx} pixels are not sky")

    if wcs is not None and pole_radius > 0:
        # The north celestial pole sits due true north at an altitude equal to
        # the site latitude; refraction moves it by ~0.2 px, far inside the disc.
        px, py = wcs.world_to_pixel_values(0.0, float(MMT_LOCATION.lat.deg))
        px, py = float(px), float(py)
        if np.isfinite(px) and np.isfinite(py):
            yy, xx = np.ogrid[:ny, :nx]
            pole = (xx - px) ** 2 + (yy - py) ** 2 <= float(pole_radius) ** 2
            valid &= ~pole
            if log:
                log(f"excluding a {pole_radius} px disc at the celestial pole "
                    f"({px:.1f}, {py:.1f})")

    return valid



def build_alcor_badpix_mask(median_cube, ksize=5, z_thresh=25.0, valid=None):
    """
    Detect per-channel hot pixels in a night-median stack.

    For each channel a small-kernel median high-pass isolates sharp spikes:
    ``resid = img - median_filter(img, ksize)``; a pixel is hot where its robust
    z-score ``(resid - median) / (1.4826 * MAD)`` exceeds ``z_thresh``. A spike is
    a sensor defect only if it fires in AT MOST TWO channels -- one present in all
    three is a real broadband source and is excluded from every plane.

    Parameters
    ----------
    median_cube : ndarray
        Per-pixel median stack of shape ``(3, ny, nx)`` (see
        :func:`build_alcor_median_stack`).
    ksize : int (default=5)
        Local-background median-filter kernel (pixels).
    z_thresh : float (default=25.0)
        Robust-sigma threshold for a hot pixel.
    valid : ndarray of bool, shape ``(ny, nx)``, optional
        Where a candidate may be accepted, from
        :func:`alcor_badpix_search_region`. Candidates outside it are dropped.
        The robust sigma is still measured over the whole frame, which is what
        we want: it is a read-noise scale, not a property of the search region.

    Returns
    -------
    mask : ndarray of bool, shape ``(3, ny, nx)``
        True where a pixel is a per-channel bad pixel.
    """
    cube = np.asarray(median_cube, dtype=float)
    if cube.ndim != 3 or cube.shape[0] != 3:
        raise ValueError(f"expected a (3, ny, nx) cube, got {cube.shape}")
    z = np.empty_like(cube)
    for c in range(3):
        resid = cube[c] - median_filter(cube[c], size=ksize)
        med = np.median(resid)
        sigma = 1.4826 * np.median(np.abs(resid - med)) + 1e-9
        z[c] = (resid - med) / sigma
    hot = z > z_thresh
    keep = hot.sum(axis=0) <= 2
    mask = hot & keep[None, :, :]
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != cube.shape[1:]:
            raise ValueError(f"valid has shape {valid.shape}, expected "
                             f"{cube.shape[1:]}")
        mask &= valid[None, :, :]
    return mask



def _median_stack_tiles(cube_mm, selection, shape, tile=50):
    """
    Per-pixel median over the selected frames of an ``(n, 3, ny, nx)`` memmap.

    ``selection`` indexes the frame axis: a slice for a contiguous run, or an
    index array when only some slots hold valid frames. The median is taken in
    ``tile``-row slabs so peak memory stays a small multiple of one row block
    rather than the whole stack. Returns ``(3, ny, nx)`` float32.
    """
    nch, ny, nx = shape
    median = np.empty((nch, ny, nx), dtype=np.float32)
    for c in range(nch):
        for r0 in range(0, ny, tile):
            r1 = min(r0 + tile, ny)
            slab = np.asarray(cube_mm[selection, c, r0:r1, :], dtype=np.float32)
            median[c, r0:r1, :] = np.median(slab, axis=0)
    return median



def build_alcor_median_stack(dark_files, max_frames=None, scratch_dir=None,
                             tile=50, log=None):
    """
    Per-pixel median over a set of raw alcor frames, trail-free for hot-pixel
    detection.

    RAM-bounded: each frame's ``(3, ny, nx)`` uint16 cube is written to a disk
    memmap (in ``scratch_dir``), then the median is taken in row tiles so peak
    memory stays small even for ~1000 frames. Frames whose shape differs from the
    first are skipped. ``max_frames`` strided-subsamples to cap runtime/scratch.

    Returns the median as ``(3, ny, nx)`` float32.
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
        prefix="alcor_stack_", suffix=".dat",
        dir=scratch_dir or tempfile.gettempdir(), delete=False)
    tmp.close()
    memmap_path = Path(tmp.name)
    cube_mm = None
    try:
        cube_mm = np.memmap(memmap_path, dtype=np.uint16, mode="w+",
                            shape=(len(files_), nch, ny, nx))
        n = 0
        for f in files_:
            with fits.open(f) as hdul:
                data = np.asarray(hdul[0].data)
            if data.shape != shp:
                if log:
                    log(f"skip {Path(f).name}: shape {data.shape}")
                continue
            cube_mm[n] = np.clip(data, 0, 65535).astype(np.uint16)
            n += 1
        cube_mm.flush()
        if n == 0:
            raise ValueError("no frames matched the reference shape")

        return _median_stack_tiles(cube_mm, slice(0, n), (nch, ny, nx), tile=tile)
    finally:
        del cube_mm
        memmap_path.unlink(missing_ok=True)



def create_badpix_mask(day_dir, out_dir=None, min_frames=500, z_thresh=25.0,
                        ksize=5, sun_alt_max=-18.0, moon_alt_max=-6.0,
                        rim_dilation=ALCOR_BADPIX_RIM_DILATION,
                        pole_radius=ALCOR_BADPIX_POLE_RADIUS,
                        max_frames=None, scratch_dir=None, pattern="*.fits.bz2",
                        horizon_dir=None, log=None):
    """
    Build and write a date-stamped per-channel bad-pixel mask for one night.

    Selects dark frames (Sun < ``sun_alt_max``, Moon < ``moon_alt_max``), and if
    at least ``min_frames`` are available builds the night-median stack, detects
    hot pixels within the sky region given by
    :func:`alcor_badpix_search_region`, and writes a gzipped
    ``alcor_badpix_YYYY-MM-DD.fits.gz`` to ``out_dir`` (default: the resolved
    bad-pixel masks directory). Returns the output `~pathlib.Path`, or ``None``
    if there were too few dark frames.
    """
    day_dir = Path(day_dir)
    frames = sorted(day_dir.glob(pattern))
    dark = select_dark_frames(frames, sun_alt_max=sun_alt_max,
                              moon_alt_max=moon_alt_max, log=None)
    if log:
        log(f"{len(dark)} dark frames of {len(frames)}")
    if len(dark) < min_frames:
        if log:
            log(f"only {len(dark)} dark frames (< {min_frames}); no mask written")
        return None

    median = build_alcor_median_stack(dark, max_frames=max_frames,
                                      scratch_dir=scratch_dir, log=log)
    mask_date = _badpix_date_from_dir(day_dir, dark)

    cal = alcor_calibration(Time(mask_date.isoformat()))
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])
    valid = alcor_badpix_search_region(
        median.shape[1:], time=mask_date, wcs=wcs, horizon_dir=horizon_dir,
        rim_dilation=rim_dilation, pole_radius=pole_radius, log=log)
    mask = build_alcor_badpix_mask(median, ksize=ksize, z_thresh=z_thresh,
                                   valid=valid)

    out_dir = Path(str(out_dir)) if out_dir is not None else _resolve_badpix_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alcor_badpix_{mask_date.isoformat()}.fits.gz"

    hdu = fits.PrimaryHDU(data=mask.astype(np.uint8))
    hdu.header["NSTACK"] = (len(dark), "dark frames used")
    hdu.header["ZTHRESH"] = (z_thresh, "robust-sigma threshold")
    hdu.header["KSIZE"] = (ksize, "high-pass kernel (px)")
    hdu.header["CHRULE"] = ("1-2 of 3", "channels flagged for a bad pixel")
    hdu.header["RIMDILAT"] = (rim_dilation, "horizon-mask dilation (px)")
    hdu.header["POLERAD"] = (pole_radius, "celestial-pole exclusion (px)")
    hdu.header["NSEARCH"] = (int(valid.sum()), "pixels searched for defects")
    for c, name in enumerate("RGB"):
        hdu.header[f"NBAD{name}"] = (int(mask[c].sum()), f"{name} bad pixels")
    hdu.writeto(out_path, overwrite=True)
    if log:
        log(f"wrote {out_path}")
    return out_path
