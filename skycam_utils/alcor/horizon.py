"""Building the sky/not-sky horizon mask from a cloudy-night median."""

import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from skimage.filters import sobel
from skimage.morphology import opening, disk
from astropy.io import fits
from astropy.time import Time

from .config import alcor_calibration
from .timeutils import select_dark_frames
from .wcs import build_alcor_wcs
from .masks import (
    _badpix_date_from_dir, _resolve_horizon_dir, load_alcor_badpix_mask
)


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
    E = sobel(ndimage.gaussian_filter(np.log10(np.clip(img, 1, None)), 1.0))
    thr = np.percentile(E[in_fov], edge_pct)
    wall = (E > thr) & in_fov
    if edge_dilate:
        wall = ndimage.binary_dilation(wall, iterations=edge_dilate)

    undet_obstruction = np.zeros((ny, nx), dtype=bool)
    if undetected is not None:
        fr_pix, ct_pix = undetected(az, alt)
        sec_lo, sec_hi = sector
        in_sector = (az >= sec_lo) & (az <= sec_hi)
        undet_obstruction = (in_sector & in_fov
                             & (fr_pix >= und_thr) & (ct_pix >= und_mincount))

    # flood-fill the sky: free = inside FOV, not a wall, not undetected-blocked
    free = in_fov & ~wall & ~undet_obstruction
    lbl, _ = ndimage.label(free, structure=np.ones((3, 3)))
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
    nlab, nn = ndimage.label(notsky_open, structure=np.ones((3, 3)))
    if nn:
        size = np.bincount(nlab.ravel())[1:]
        min_alt = ndimage.minimum(alt, nlab, index=np.arange(1, nn + 1))
        keep = (min_alt < rim_alt) | (size >= rod_area_min)
        notsky = np.concatenate([[False], keep])[nlab]
    else:
        notsky = notsky_open

    return (~in_fov) | notsky



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
    frac = ndimage.gaussian_filter(Hund / np.maximum(Htot, 1), (smooth, smooth))
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
