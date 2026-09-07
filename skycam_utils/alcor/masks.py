"""Loading the date-resolved bad-pixel and horizon mask assets."""

import os
import re
from datetime import datetime, date
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from astropy.io import fits
from astropy.time import Time

from .config import _DATA_ROOT
from .timeutils import _filename_ut_datetime


_BADPIX_DATE_RE = re.compile(r"alcor_badpix_(\d{4})-(\d{2})-(\d{2})\.fits(\.gz)?$")



def _resolve_badpix_dir(masks_dir=None):
    """
    Directory holding the date-stamped bad-pixel masks.

    Resolution order: explicit ``masks_dir`` -> ``$ALCOR_BADPIX_DIR`` -> the
    packaged ``skycam_utils/data/badpix/`` (mirroring :func:`load_wcs`).
    """
    if masks_dir is not None:
        return Path(masks_dir)
    env = os.environ.get("ALCOR_BADPIX_DIR")
    if env:
        return Path(env)
    return Path(str(_DATA_ROOT / "badpix"))



def _badpix_date_from_dir(day_dir, dark_files):
    """
    Mask date: the ``YYYY-MM-DD`` in the day-directory name, else the median
    dark-frame time's date.
    """
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(day_dir).name)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    dts = sorted(d for d in (_filename_ut_datetime(f) for f in dark_files)
                 if d is not None)
    if not dts:
        raise ValueError(
            f"cannot determine mask date: directory name {Path(day_dir).name!r} "
            "has no YYYY-MM-DD and no frame timestamps could be parsed")
    return dts[len(dts) // 2].date()



def load_alcor_badpix_mask(time, masks_dir=None):
    """
    Return ``(mask, date)`` for the bad-pixel mask nearest in date to ``time``.

    ``mask`` is a ``(3, ny, nx)`` bool array; ``(None, None)`` if no masks are
    found. ``time`` may be a `~astropy.time.Time`, ``datetime``, or ``date``.
    """
    directory = Path(str(_resolve_badpix_dir(masks_dir)))
    if not directory.is_dir():
        return None, None
    candidates = []
    for p in directory.iterdir():
        m = _BADPIX_DATE_RE.match(p.name)
        if m:
            candidates.append(
                (date(int(m.group(1)), int(m.group(2)), int(m.group(3))), p))
    if not candidates:
        return None, None

    if isinstance(time, Time):
        target = time.to_datetime().date()
    elif isinstance(time, datetime):
        target = time.date()
    else:
        target = time
    best_date, best_path = min(candidates,
                               key=lambda dp: (abs((dp[0] - target).days), dp[0]))
    mask = np.asarray(fits.getdata(best_path)).astype(bool)
    return mask, best_date



def _apply_badpix_repair(data, mask, ksize=5):
    """
    Replace masked pixels with their local median, per channel.

    ``data`` and ``mask`` are ``(3, ny, nx)``. Returns a repaired copy; the input
    is not mutated. The local median (computed over the surrounding ``ksize``
    window) is robust to the spike itself, so it recovers the underlying sky.
    """
    out = np.array(data, copy=True)
    for c in range(data.shape[0]):
        if not mask[c].any():
            continue
        # filter on a native float copy (FITS data is big-endian int16, which
        # median_filter can choke on), then cast back to the frame's dtype.
        local = median_filter(np.asarray(out[c], dtype=np.float32), size=ksize)
        out[c][mask[c]] = local[mask[c]].astype(out.dtype)
    return out



_HORIZON_DATE_RE = re.compile(r"alcor_horizon_(\d{4})-(\d{2})-(\d{2})\.fits(\.gz)?$")



def _resolve_horizon_dir(masks_dir=None):
    """
    Directory holding the date-stamped horizon (sky / not-sky) pixel masks.

    Resolution order: explicit ``masks_dir`` -> ``$ALCOR_HORIZON_DIR`` -> the
    packaged ``skycam_utils/data/horizon/`` (mirroring :func:`_resolve_badpix_dir`).
    """
    if masks_dir is not None:
        return Path(masks_dir)
    env = os.environ.get("ALCOR_HORIZON_DIR")
    if env:
        return Path(env)
    return Path(str(_DATA_ROOT / "horizon"))



def load_alcor_horizon_mask(time, masks_dir=None):
    """
    Return ``(mask, date)`` for the horizon mask nearest in date to ``time``.

    ``mask`` is a ``(ny, nx)`` bool array in the raw camera frame where ``True``
    marks **not-sky** pixels -- obstructions above the horizon (terrain, the
    buildings, the lightning rod) together with everything at or below altitude
    0 -- so the valid sky region is ``~mask``. ``(None, None)`` if no masks are
    found. ``time`` may be a `~astropy.time.Time`, ``datetime``, or ``date``.

    Unlike the per-channel bad-pixel mask this is a single achromatic 2-D plane
    (the horizon is the same for R/G/B), and it is an *exclusion* mask -- it is
    not used to repair pixels, only to select valid sky for sky-background /
    cloud-extinction maps. It is built offline by
    ``claude_docs/scripts/horizon_floodfill.py`` (a Sobel-edge flood-fill of a
    cloudy-night median, with the SW->W building sector filled from the
    undetected-star patch) and is resolved by date the same way as the
    calibration and bad-pixel assets, so a camera move is handled by adding a
    new epoch.
    """
    directory = Path(str(_resolve_horizon_dir(masks_dir)))
    if not directory.is_dir():
        return None, None
    candidates = []
    for p in directory.iterdir():
        m = _HORIZON_DATE_RE.match(p.name)
        if m:
            candidates.append(
                (date(int(m.group(1)), int(m.group(2)), int(m.group(3))), p))
    if not candidates:
        return None, None

    if isinstance(time, Time):
        target = time.to_datetime().date()
    elif isinstance(time, datetime):
        target = time.date()
    else:
        target = time
    best_date, best_path = min(candidates,
                               key=lambda dp: (abs((dp[0] - target).days), dp[0]))
    mask = np.asarray(fits.getdata(best_path)).astype(bool)
    return mask, best_date
