"""Frame timestamps, Sun/Moon altitude, and dark-frame selection."""

import re
from datetime import datetime, timedelta
from importlib.resources import files
from pathlib import Path

from astropy.io import fits
from astropy.time import Time

from astropy.coordinates import AltAz, get_sun, get_body

from ..astrometry import MMT_LOCATION

from .config import ALCOR_CALIB_EXPTIME, alcor_calibration


def _frame_time(path):
    """
    Return the observation Time (UT) from a FITS file's DATE (creation) header.

    For these alcor cameras DATE is the true UT timestamp; the DATE-OBS keyword
    is local time despite its 'UT' label.
    """
    with fits.open(path) as hdul:
        return Time(hdul[0].header["DATE"], format="isot", scale="utc")



def _sun_altitude(time, location=MMT_LOCATION):
    """
    Return the Sun's altitude in degrees at ``time`` and ``location``.

    ``time`` must be a scalar `~astropy.time.Time` (the result is returned as a
    Python float). Use :func:`select_dark_frames` for batched filtering.
    """
    altaz = get_sun(time).transform_to(AltAz(obstime=time, location=location))
    return float(altaz.alt.deg)



def _moon_altitude(time, location=MMT_LOCATION):
    """
    Return the Moon's altitude in degrees at ``time`` and ``location``.

    ``time`` must be a scalar `~astropy.time.Time` (the result is returned as a
    Python float). The Moon's position is computed topocentrically (parallax is
    ~1 deg, which matters near the rejection threshold). Use
    :func:`select_dark_frames` for batched filtering.
    """
    moon = get_body("moon", time, location)
    altaz = moon.transform_to(AltAz(obstime=time, location=location))
    return float(altaz.alt.deg)



# alcor filenames are YYYY_MM_DD__HH_MM_SS in local (MST) time.
_FILENAME_TIME_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})")

# Arizona observes Mountain Standard Time year-round (no DST): UT = local + 7h.
_MST_TO_UT = timedelta(hours=7)



def _filename_ut_datetime(filename):
    """
    Return the UT ``datetime`` parsed from an alcor filename, or ``None``.

    Filenames are ``YYYY_MM_DD__HH_MM_SS`` in local MST; converting to UT only
    needs the fixed +7h offset, so dark-frame selection can avoid opening (and
    decompressing) every file just to read its DATE header.
    """
    match = _FILENAME_TIME_RE.search(Path(filename).name)
    if match is None:
        return None
    year, month, day, hour, minute, second = (int(g) for g in match.groups())
    return datetime(year, month, day, hour, minute, second) + _MST_TO_UT



def _read_frame_date(filename):
    """
    Return a FITS file's DATE (UT) header string, for dark-frame selection.
    """
    return fits.getheader(filename)["DATE"]



def _read_frame_exposure(filename, default=ALCOR_CALIB_EXPTIME):
    """
    Return a frame's EXPOSURE (seconds) header value, ``default`` if absent or
    unreadable. Used to scale raw counts to the calibration reference exposure.
    """
    try:
        value = fits.getheader(filename).get("EXPOSURE", default)
    except (KeyError, OSError):
        return float(default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)



def _alcor_frame_calibration(filename):
    """
    Resolve the calibration epoch nearest in time to a frame.

    The frame time is parsed from its YYYY_MM_DD__HH_MM_SS filename first (no
    file access); if the name does not parse, the DATE header is read instead.
    Returns the calibration dict from :func:`alcor_calibration`.
    """
    dt = _filename_ut_datetime(filename)
    if dt is None:
        time = Time(_read_frame_date(filename), format="isot", scale="utc")
    else:
        time = Time(dt)
    return alcor_calibration(time)



def _alcor_frame_time(filename):
    """
    Best-effort observation Time for a frame: filename timestamp, then DATE
    header, then None (so callers fall back to the latest epoch).
    """
    dt = _filename_ut_datetime(filename)
    if dt is not None:
        return Time(dt)
    try:
        return Time(_read_frame_date(filename), format="isot", scale="utc")
    except (KeyError, OSError, ValueError):
        return None



def select_dark_frames(files, sun_alt_max=-18.0, moon_alt_max=-6.0,
                        location=MMT_LOCATION, log=None):
    """
    Return the subset of ``files`` whose timestamp corresponds to both the Sun
    below ``sun_alt_max`` (default -18 deg, astronomical twilight) and the Moon
    below ``moon_alt_max`` (default -6 deg). Moonlight scatter swamps the faint
    bright-star field and corrupts source detection, so moonlit frames are
    rejected even when the Sun is down. Pass ``moon_alt_max=90`` to disable the
    Moon cut.

    The UT timestamp is parsed directly from each ``YYYY_MM_DD__HH_MM_SS``
    filename (local MST, so UT = local + 7h), which avoids opening every file in
    a large archive just to read a header. Any file whose name does not match
    that pattern falls back to its DATE header (the true UT for these cameras;
    DATE-OBS is local time despite its label). Pass a ``log`` callable to report
    the start and the dark-frame count.
    """
    files = [Path(f) for f in files]
    n = len(files)
    if log is not None:
        log(f"selecting dark frames from {n} files "
            f"(Sun below {sun_alt_max:g} deg, Moon below {moon_alt_max:g} deg)...")

    dts = []
    for f in files:
        dt = _filename_ut_datetime(f)
        if dt is None:
            # Oddly-named file: fall back to the authoritative DATE header.
            dt = Time(_read_frame_date(f), format="isot", scale="utc").to_datetime()
        dts.append(dt)

    times = Time(dts, format="datetime", scale="utc")
    frame = AltAz(obstime=times, location=location)
    sun_alt = get_sun(times).transform_to(frame).alt.deg
    moon_alt = get_body("moon", times, location).transform_to(frame).alt.deg
    keep = (sun_alt < sun_alt_max) & (moon_alt < moon_alt_max)
    if log is not None:
        log(f"{int(keep.sum())} of {n} frames are dark "
            f"(Sun below {sun_alt_max:g} deg, Moon below {moon_alt_max:g} deg)")
    return [f for f, k in zip(files, keep) if k]
