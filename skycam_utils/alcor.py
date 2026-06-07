import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate
from scipy.optimize import least_squares
from scipy.ndimage import shift as ndimage_shift
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.dates as mdates
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, hstack
from astropy.time import Time
from astropy.wcs import WCS, Sip
from photutils.detection import DAOStarFinder

import astropy.units as u
import astropy.visualization as viz
from astropy.coordinates import SkyCoord, AltAz, get_sun

from .astrometry import MMT_LOCATION


# Geometry constants for load_alcor_fits, calibrated by fit_alcor_wcs against
# bright_star_sloan.fits (Vmag<=3) over a full night of dark-sky frames.
# XSHIFT/YSHIFT are absolute pixel offsets valid at the calibrated RADIUS/
# HORIZON_RADIUS; RADIAL_COEFFS=(k1,k3,k5) is the odd-power detector->sky plate
# solution. The per-star residual floor is ~4 px (centroid noise), which
# averages out for the fixed WCS; the fitted radial term removes a systematic
# that grows to ~5 px near the horizon.
ALCOR_RADIUS = 680
ALCOR_HORIZON_RADIUS = 662

# Time-indexed lens calibrations. Each epoch holds only the fitted geometry
# (xshift, yshift, rotation, radial_coeffs); the trim/scale geometry
# (ALCOR_RADIUS / ALCOR_HORIZON_RADIUS / xcen / ycen) is fixed. The camera
# geometry drifts over time (mount/focus), so the epoch nearest in time to an
# image is used (see alcor_calibration). Add a new epoch by pasting the dict
# that fit_alcor_wcs prints. `epoch` is the calibration night at day precision.
ALCOR_CALIBRATIONS = [
    {"epoch": "2024-09-04", "xshift": -4.570, "yshift": 4.413,
     "rotation": 0.3886, "radial_coeffs": (1.0, 0.01383, 0.0)},
]


def _calibration_epochs():
    """Return [(Time, calibration_dict), ...] for the configured epochs."""
    return [(Time(c["epoch"], scale="utc"), c) for c in ALCOR_CALIBRATIONS]


def alcor_calibration(time=None):
    """
    Return the calibration dict whose epoch is nearest in time to ``time``.

    ``time`` is an astropy ``Time``. An exact tie resolves to the more recent
    epoch. ``time=None`` returns the most recent epoch (the default for
    time-agnostic calls). The returned dict is a copy and may be mutated freely.
    """
    epochs = _calibration_epochs()
    if time is None:
        return dict(max(epochs, key=lambda e: e[0].jd)[1])
    jds = np.array([e[0].jd for e in epochs])
    dt = np.abs(jds - Time(time).jd)
    # primary: smallest |dt|; tie-break: largest jd (more recent)
    order = np.lexsort((-jds, dt))
    return dict(epochs[order[0]][1])


# Module-level defaults track the most-recent epoch so existing default-argument
# references (in _predict_pixels, build_alcor_wcs, etc.) keep working unchanged.
_LATEST_CALIBRATION = alcor_calibration()
ALCOR_ROTATION = _LATEST_CALIBRATION["rotation"]
ALCOR_XSHIFT = _LATEST_CALIBRATION["xshift"]
ALCOR_YSHIFT = _LATEST_CALIBRATION["yshift"]
ALCOR_RADIAL_COEFFS = _LATEST_CALIBRATION["radial_coeffs"]


def _invert_radial(z_deg, radial_coeffs, n_iter=8):
    """
    Invert ``z = 90*(k1*rho + k3*rho**3 + k5*rho**5)`` for the normalized
    detector radius ``rho`` using Newton's method. ``z_deg`` is the zenith angle
    in degrees. Assumes the polynomial is monotonic over the field of view
    (true for physical near-equidistant coefficients).
    """
    k1, k3, k5 = radial_coeffs
    if k1 <= 0:
        raise ValueError("radial coefficient k1 must be positive")
    t = np.asarray(z_deg, dtype=float) / 90.0
    rho = t / k1  # equidistant first guess
    for _ in range(n_iter):
        g = k1 * rho + k3 * rho**3 + k5 * rho**5
        gp = k1 + 3.0 * k3 * rho**2 + 5.0 * k5 * rho**4
        rho = rho - (g - t) / gp
    return rho


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
    Forward lens model: map altitude/azimuth (degrees) to processed-frame pixel
    coordinates (x=column, y=row).

    Pixel y increases upward (FITS/WCS convention), matching the coordinate
    system of the WCS returned by ``load_alcor_fits``; it is not a direct numpy
    row index.

    The lens is described by the plate solution ``z = 90*(k1*rho + k3*rho**3 +
    k5*rho**5)`` with ``rho = r / horizon_radius`` the normalized detector radius
    and ``z = 90 - alt`` the zenith angle (an odd-power, symmetric-fisheye
    polynomial in detector radius). Mapping alt/az to a pixel inverts this for
    ``rho`` via Newton's method. The idealized coefficients ``(1, 0, 0)`` give the
    equidistant ARC mapping; higher-order terms encode the lens's non-linear
    behavior with zenith angle. ``xshift``/``yshift`` offset the zenith from the
    array center.

    ``rotation`` is a residual azimuth-frame rotation (degrees) applied on top of
    the image rotation that ``load_alcor_fits`` already applies; it defaults to
    0.0 because the idealized/centered frame has no residual rotation.

    The zenith maps to the array geometric center (radius-0.5, radius-0.5),
    consistent with crpix=radius+0.5.
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    rho = _invert_radial(90.0 - alt, tuple(float(c) for c in radial_coeffs))
    r = horizon_radius * rho
    ang = np.radians(az + rotation)
    x = (radius - 0.5) + xshift - r * np.sin(ang)
    y = (radius - 0.5) + yshift + r * np.cos(ang)
    return x, y


def _fit_params(alt, az, obs_x, obs_y, init_params, radius=ALCOR_RADIUS,
                horizon_radius=ALCOR_HORIZON_RADIUS):
    """
    Robust least-squares fit of (xshift, yshift, rotation, k3) to matched stars.

    The lens nonlinearity is modeled by the single odd cubic term k3; k1 is held
    at 1.0 (the zenith plate scale is set by horizon_radius) and k5 at 0.0. k3 and
    k5 are nearly collinear in rho over [0, 1], so fitting both is ill-conditioned
    and on a large pooled dataset runs away to large cancelling values (e.g.
    k3=-0.58, k5=3.6) that are unphysical despite a tolerable RMS; k3 alone
    captures the distortion (this is the model the shipped 2024 constants used). A
    ``soft_l1`` loss downweights mismatched/noise detections, which are common in
    this sparse bright-star field. Returns an updated params dict with
    radial_coeffs=(1.0, k3, 0.0).
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    obs_x = np.asarray(obs_x, dtype=float)
    obs_y = np.asarray(obs_y, dtype=float)
    p0 = np.array([
        init_params["xshift"], init_params["yshift"], init_params["rotation"],
        init_params["radial_coeffs"][1],
    ], dtype=float)

    def residuals(p):
        xshift, yshift, rot, k3 = p
        x, y = _predict_pixels(alt, az, xshift=xshift, yshift=yshift, rotation=rot,
                               radial_coeffs=(1.0, k3, 0.0), radius=radius,
                               horizon_radius=horizon_radius)
        return np.concatenate([x - obs_x, y - obs_y])

    result = least_squares(residuals, p0, loss="soft_l1", f_scale=3.0)
    xshift, yshift, rot, k3 = result.x
    return dict(xshift=float(xshift), yshift=float(yshift), rotation=float(rot),
                radial_coeffs=(1.0, float(k3), 0.0))


def match_alcor_stars(cat, detections, init_params,
                      z_steps=(20.0, 40.0, 60.0, 75.0, 90.0), tolerance=12.0,
                      radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS):
    """
    Match catalog stars (with Alt/Az/Vmag columns) to detected sources by
    bootstrapping outward from the zenith.

    For each zenith-angle cutoff in ``z_steps`` the catalog stars within that
    cutoff are projected to pixels with the current geometry parameters and
    matched to the nearest detection within ``tolerance`` pixels. When multiple
    catalog stars claim the same detection the brighter one (smaller Vmag) wins.
    After each step the geometry is refit (:func:`_fit_params`) and the cutoff
    expands. Returns a table of matched (catalog + detection) rows.
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
        return hstack([Table(cat[[]]), Table(detections[[]])])
    det_rows = list(matched_idx.keys())
    cat_rows = [ci for ci, _ in matched_idx.values()]
    out = hstack([Table(cat[cat_rows]), Table(detections[det_rows])])
    return out


def _frame_time(path):
    """Return the observation Time (UT) from a FITS file's DATE (creation) header.

    For these alcor cameras DATE is the true UT timestamp; the DATE-OBS keyword
    is local time despite its 'UT' label.
    """
    with fits.open(path) as hdul:
        return Time(hdul[0].header["DATE"], format="isot", scale="utc")


def _detect_alcor_frame(task):
    """
    Per-frame preprocessing for :func:`fit_alcor_wcs`, executed in worker
    processes. Loads a frame, builds its reference catalog and star detections,
    and returns ``(index, cat, det, reason)``. On success ``reason`` is ``None``;
    when the frame is unusable ``cat``/``det`` are ``None`` and ``reason`` is a
    short human-readable string (too few detections or catalog stars).
    """
    index, filename, vmag_limit, min_alt, fwhm, threshold_sigma = task
    filename = Path(filename)
    time = _frame_time(filename)
    im, _ = load_alcor_fits(filename, rotation=0.0, xshift=0.0, yshift=0.0,
                            radial_coeffs=(1.0, 0.0, 0.0))
    cat = alcor_reference_altaz(time, vmag_limit=vmag_limit, min_alt=min_alt)
    det = detect_alcor_stars(im, fwhm=fwhm, threshold_sigma=threshold_sigma)
    if len(det) < 3:
        return index, None, None, f"no stars detected ({len(det)} < 3)"
    if len(cat) < 3:
        return index, None, None, f"too few catalog stars ({len(cat)} < 3)"
    return index, cat, det, None


def fit_alcor_wcs(input_dir, pattern="*.fits.bz2", vmag_limit=3.0, sun_alt_max=-18.0,
                  min_alt=10.0, tolerance=12.0, fwhm=3.0, threshold_sigma=5.0,
                  z_steps=(20.0, 40.0, 60.0, 75.0, 90.0), max_frames=None,
                  workers=1, log=None):
    """
    Calibrate the alcor lens geometry by aggregating bright-star matches across
    all dark-sky frames in ``input_dir``.

    Frames are selected with :func:`select_dark_frames` (Sun below ``sun_alt_max``).
    For each, stars (``Vmag <= vmag_limit``) are projected with the current
    geometry, matched via :func:`match_alcor_stars`, and the matched
    (Alt, Az, x, y) tuples are pooled. Matches are pooled and fit globally, then
    frames are re-matched with the refined geometry and refit, and a final fit is
    performed after 3*MAD outlier rejection.

    The fit runs on a neutral (uncalibrated) frame -- loaded with no recentering,
    rotation, or radial distortion -- so the recovered (xshift, yshift, rotation,
    radial_coeffs) are the ABSOLUTE geometry constants for the night, suitable for
    baking into ``ALCOR_CALIBRATIONS``. It is warm-started from the nearest
    existing epoch (via :func:`alcor_calibration` at the night's median time).

    Returns a dict with the fitted absolute parameters plus an ``epoch`` date
    string (the night's UT date, or the seed epoch when no frame can be timed),
    ``n_matched``, ``residual_rms``, and per-match arrays (``alt``, ``az``,
    ``x``, ``y``) for diagnostics.

    The per-frame load/detect/catalog work is the expensive part and is
    independent across frames, so it is parallelized: ``workers=1`` runs
    serially, any larger value (or ``None`` for the process-pool default)
    distributes the frames over a `~concurrent.futures.ProcessPoolExecutor`.
    Pass a ``log`` callable (e.g. ``print``) to report each file's disposition:
    frames skipped because the Sun is above ``sun_alt_max``, frames skipped
    because no stars were detected, and frames used (with detected star count).
    """
    if workers is not None and workers < 1:
        raise ValueError("workers must be None or a positive integer")
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    dark = select_dark_frames(files, sun_alt_max=sun_alt_max, log=log)
    if log is not None:
        dark_set = set(dark)
        for f in files:
            if f not in dark_set:
                log(f"{Path(f).name}: skipped (Sun above {sun_alt_max:g} deg)")
    if max_frames is not None:
        dark = dark[:max_frames]

    # Representative night time (median dark-frame time), used both to seed the
    # fit from the nearest existing calibration and to stamp the new epoch.
    # Prefer the filename timestamp; fall back to the DATE header, and skip any
    # frame that can be timed by neither (corrupt/oddly-named) rather than fail.
    night_dts = []
    for f in dark:
        d = _filename_ut_datetime(f)
        if d is None:
            try:
                d = Time(_read_frame_date(f), format="isot", scale="utc").to_datetime()
            except Exception:
                continue
        night_dts.append(d)
    night_time = Time(sorted(night_dts)[len(night_dts) // 2]) if night_dts else None
    base = alcor_calibration(night_time)
    epoch = (night_time.datetime.date().isoformat()
             if night_time is not None else base["epoch"])

    init = dict(xshift=base["xshift"], yshift=base["yshift"],
                rotation=base["rotation"], radial_coeffs=base["radial_coeffs"])
    # (cat, detections) per usable frame, kept in frame order for reproducible
    # pooling regardless of worker completion order.
    detected = [None] * len(dark)
    tasks = [(index, f, vmag_limit, min_alt, fwhm, threshold_sigma)
             for index, f in enumerate(dark)]

    def _store(result):
        index, cat, det, reason = result
        name = Path(dark[index]).name
        if cat is not None:
            detected[index] = (cat, det)
            if log is not None:
                log(f"{name}: {len(det)} stars detected")
        elif log is not None:
            log(f"{name}: skipped ({reason})")

    if workers == 1:
        for task in tasks:
            _store(_detect_alcor_frame(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_detect_alcor_frame, task) for task in tasks]
            for future in as_completed(futures):
                _store(future.result())

    frames = [d for d in detected if d is not None]

    def pool(seed_params):
        a, z, xs, ys = [], [], [], []
        for cat, det in frames:
            matched = match_alcor_stars(cat, det, init_params=seed_params,
                                        z_steps=z_steps, tolerance=tolerance)
            if len(matched) == 0:
                continue
            a.append(np.asarray(matched["Alt"], dtype=float))
            z.append(np.asarray(matched["Az"], dtype=float))
            xs.append(np.asarray(matched["xcentroid"], dtype=float))
            ys.append(np.asarray(matched["ycentroid"], dtype=float))
        return a, z, xs, ys

    pooled_alt, pooled_az, pooled_x, pooled_y = pool(init)
    if not pooled_alt:
        raise RuntimeError("No matched stars across the selected frames.")

    alt = np.concatenate(pooled_alt)
    az = np.concatenate(pooled_az)
    x = np.concatenate(pooled_x)
    y = np.concatenate(pooled_y)
    params = _fit_params(alt, az, x, y, init_params=init)

    # Re-match each frame seeded with the refined geometry. Frames whose
    # zenith-bootstrap stalled from the cold start now converge, recovering
    # additional matches before the final global fit.
    pooled_alt, pooled_az, pooled_x, pooled_y = pool(params)
    alt = np.concatenate(pooled_alt)
    az = np.concatenate(pooled_az)
    x = np.concatenate(pooled_x)
    y = np.concatenate(pooled_y)

    params = _fit_params(alt, az, x, y, init_params=params)

    # Final residuals with outlier rejection at 3*MAD, then refit.
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
        "epoch": epoch,
        "n_matched": int(good.sum()),
        "residual_rms": rms,
        "alt": alt[good], "az": az[good], "x": x[good], "y": y[good],
    }


def save_alcor_residual_plot(alt, az, obs_x, obs_y, params, output_file,
                             radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                             figsize=(10, 5), dpi=150):
    """
    Plot pixel-residual magnitude versus zenith angle for matched stars, before
    (idealized equidistant) and after (fitted) the refinement. Returns the
    output path.
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
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
    ax.set_title("Alcor WCS residuals")
    ax.set_xlabel("zenith angle (deg)")
    ax.set_ylabel("pixel residual")
    ax.legend()
    output_file = Path(output_file)
    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_file


def _sun_altitude(time, location=MMT_LOCATION):
    """Return the Sun's altitude in degrees at ``time`` and ``location``.

    ``time`` must be a scalar `~astropy.time.Time` (the result is returned as a
    Python float). Use :func:`select_dark_frames` for batched filtering.
    """
    altaz = get_sun(time).transform_to(AltAz(obstime=time, location=location))
    return float(altaz.alt.deg)


# alcor filenames are YYYY_MM_DD__HH_MM_SS in local (MST) time.
_FILENAME_TIME_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})")
# Arizona observes Mountain Standard Time year-round (no DST): UT = local + 7h.
_MST_TO_UT = timedelta(hours=7)


def _filename_ut_datetime(filename):
    """Return the UT ``datetime`` parsed from an alcor filename, or ``None``.

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
    """Return a FITS file's DATE (UT) header string, for dark-frame selection."""
    return fits.getheader(filename)["DATE"]


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


def select_dark_frames(files, sun_alt_max=-18.0, location=MMT_LOCATION, log=None):
    """
    Return the subset of ``files`` whose timestamp corresponds to a Sun altitude
    below ``sun_alt_max`` (default -18 deg, astronomical twilight).

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
        log(f"selecting dark frames from {n} files (Sun below {sun_alt_max:g} deg)...")

    dts = []
    for f in files:
        dt = _filename_ut_datetime(f)
        if dt is None:
            # Oddly-named file: fall back to the authoritative DATE header.
            dt = Time(_read_frame_date(f), format="isot", scale="utc").to_datetime()
        dts.append(dt)

    times = Time(dts, format="datetime", scale="utc")
    altaz = get_sun(times).transform_to(AltAz(obstime=times, location=location))
    keep = altaz.alt.deg < sun_alt_max
    if log is not None:
        log(f"{int(keep.sum())} of {n} frames are dark (Sun below {sun_alt_max:g} deg)")
    return [f for f, k in zip(files, keep) if k]


def _base_arc_wcs(radius, horizon_radius, k1):
    """Construct the linear ARC WCS (no SIP) for the given geometry."""
    cdelt = 90.0 * k1 / horizon_radius
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
    wcs.wcs.crpix = [radius + 0.5, radius + 0.5]
    wcs.wcs.crval = [0.0, 90.0]
    wcs.wcs.cdelt = [cdelt, cdelt]
    wcs.wcs.lonpole = 0.0
    return wcs


def _sip_poly_eval(coef, u, v):
    """Evaluate a SIP coefficient matrix (coef[p, q] * u**p * v**q) at (u, v)."""
    out = np.zeros_like(u, dtype=float)
    n = coef.shape[0]
    for p in range(n):
        for q in range(n):
            c = coef[p, q]
            if c != 0.0:
                out = out + c * u**p * v**q
    return out


def _fit_sip_inverse(a, b, radius, sip_degree):
    """
    Fit the approximate inverse SIP coefficients (AP, BP) for forward
    coefficients (A, B) over a pixel grid. The inverse of a radial polynomial is
    not itself polynomial, so AP/BP are a least-squares approximation (used by
    external tools and as the initial guess for astropy's iterative
    world->pixel solve, which refines to machine precision using A/B).
    """
    g = np.linspace(-radius, radius, 50)
    uu, vv = np.meshgrid(g, g)
    u = uu.ravel()
    v = vv.ravel()
    fu = u + _sip_poly_eval(a, u, v)
    fv = v + _sip_poly_eval(b, u, v)
    terms = [(p, q) for p in range(sip_degree + 1) for q in range(sip_degree + 1)
             if 1 <= p + q <= sip_degree]
    design = np.column_stack([fu**p * fv**q for (p, q) in terms])
    coef_u, _, _, _ = np.linalg.lstsq(design, u - fu, rcond=None)
    coef_v, _, _, _ = np.linalg.lstsq(design, v - fv, rcond=None)
    ap = np.zeros((sip_degree + 1, sip_degree + 1))
    bp = np.zeros((sip_degree + 1, sip_degree + 1))
    for (p, q), cu, cv in zip(terms, coef_u, coef_v):
        ap[p, q] = cu
        bp[p, q] = cv
    return ap, bp


def build_alcor_wcs(radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                    radial_coeffs=ALCOR_RADIAL_COEFFS, sip_degree=5):
    """
    Build an ARC-projection WCS for the processed alcor frame. When the radial
    model has non-trivial higher-order terms, the deviation from the linear
    (equidistant) mapping is encoded as an exact analytic SIP distortion, so
    ``world_to_pixel``/``to_header`` reproduce the lens distortion. Cached
    because the geometry is fixed across images.

    The lens is parametrized as a plate solution that maps the detector directly
    to the sky: ``z = 90*(k1*rho + k3*rho**3 + k5*rho**5)`` with ``rho =
    r / horizon_radius`` the normalized detector radius and ``z = 90 - alt`` the
    zenith angle (an odd-power, symmetric-fisheye polynomial in detector radius;
    the ``k5`` term needs degree 5, hence ``sip_degree=5``). The Cartesian
    displacement of this radial map is an exact degree-5 polynomial in the
    detector pixel offsets, so the SIP coefficients are constructed analytically
    (not fitted) and reproduce the plate solution to numerical precision over
    the whole FOV.

    Accepts any iterable of radial coefficients (coerced to a tuple for caching)
    and returns a fresh WCS copy on each call, so the cached canonical object is
    never mutated by callers.
    """
    wcs = _build_alcor_wcs_cached(
        int(radius), float(horizon_radius),
        tuple(float(c) for c in radial_coeffs), int(sip_degree),
    )
    return wcs.deepcopy()


@lru_cache(maxsize=32)
def _build_alcor_wcs_cached(radius, horizon_radius, radial_coeffs, sip_degree):
    k1, k3, k5 = radial_coeffs
    base = _base_arc_wcs(radius, horizon_radius, k1)
    if abs(k3) < 1e-12 and abs(k5) < 1e-12:
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
    ap, bp = _fit_sip_inverse(a, b, radius, sip_degree)

    wcs = base.deepcopy()
    wcs.wcs.ctype = ["RA---ARC-SIP", "DEC--ARC-SIP"]
    wcs.sip = Sip(a, b, ap, bp, [radius + 0.5, radius + 0.5])
    return wcs


def detect_alcor_stars(im, fwhm=3.0, threshold_sigma=5.0):
    """
    Detect point sources in a processed alcor RGB image.

    The three channels are summed into a luminance image, the background level is
    estimated with a sigma-clipped median, and `~photutils.detection.DAOStarFinder`
    extracts sources above ``threshold_sigma`` times the background noise.

    Parameters
    ----------
    im : ndarray
        Processed RGB image of shape (ny, nx, 3), as returned by ``load_alcor_fits``.
    fwhm : float (default=3.0)
        FWHM (pixels) of the Gaussian kernel used by the star finder.
    threshold_sigma : float (default=5.0)
        Detection threshold in multiples of the background noise.

    Returns
    -------
    sources : `~astropy.table.Table`
        Detected sources with at least ``xcentroid``, ``ycentroid``, ``flux``
        columns. Empty (with those columns) if nothing is found.
    """
    lum = np.asarray(im, dtype=float).sum(axis=2)
    _, median, std = sigma_clipped_stats(lum, sigma=3.0)
    finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = finder(lum - median)
    if sources is None:
        return Table(names=["xcentroid", "ycentroid", "flux"],
                     dtype=[float, float, float])
    # photutils 3.x uses x_centroid/y_centroid as the primary column names;
    # xcentroid/ycentroid are deprecated aliases scheduled for removal in 4.0.
    # Copy to a plain Table (which strips the deprecation-alias machinery) and
    # rename x_centroid/y_centroid to xcentroid/ycentroid — the names this
    # function's API exposes — so callers always see a consistent column set.
    xcol = "x_centroid" if "x_centroid" in sources.colnames else "xcentroid"
    ycol = "y_centroid" if "y_centroid" in sources.colnames else "ycentroid"
    out = Table(sources)
    if xcol != "xcentroid":
        out.rename_column(xcol, "xcentroid")
    if ycol != "ycentroid":
        out.rename_column(ycol, "ycentroid")
    return out


ALCOR_PRESSURE = 760 * u.hPa        # ~0.75 atm at the MMT 2600 m elevation
ALCOR_TEMPERATURE = 10 * u.deg_C
ALCOR_HUMIDITY = 0.2
ALCOR_OBSWL = 0.55 * u.micron


def alcor_reference_altaz(time, vmag_limit=3.0, min_alt=5.0, refraction=True,
                           location=MMT_LOCATION):
    """
    Load ``bright_star_sloan.fits``, filter to ``Vmag <= vmag_limit``, and compute
    Alt/Az at ``time`` and ``location``. Stars below ``min_alt`` are dropped.

    When ``refraction`` is True the AltAz frame includes atmospheric refraction
    using nominal MMT pressure/temperature; this matters most at large zenith
    angle, where the radial distortion is also largest.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Observation time (scalar).
    vmag_limit : float (default=3.0)
        Faintest V magnitude to keep.
    min_alt : float (default=5.0)
        Minimum altitude (deg) to keep.
    refraction : bool (default=True)
        If True, include atmospheric refraction in the AltAz transform.
    location : `~astropy.coordinates.EarthLocation` (default=MMT_LOCATION)
        Observatory location.

    Returns
    -------
    cat : `~astropy.table.Table`
        Catalog rows with added ``Alt`` and ``Az`` columns (degrees), filtered to
        ``Vmag <= vmag_limit`` and ``Alt >= min_alt``.
    """
    catpath = files(__package__) / "data" / "bright_star_sloan.fits"
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


def load_alcor_fits(filename, rotation=None, xcen=696, ycen=698,
                    radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                    xshift=None, yshift=None,
                    radial_coeffs=None, sip_degree=5):
    """
    Load a FITS image from the alcor OMEA 8C all-sky camera and return a
    zenith-centered, north-up RGB image along with a WCS that maps pixel
    coordinates to altitude/azimuth.

    The image is bias-subtracted, trimmed to a square centered on the
    illuminated region, rotated to remove the camera tilt, and flipped so
    north is at the top.

    The WCS is an ARC (zenith equidistant) projection with the pole placed
    at zenith (CRVAL=(0, 90)) so that altitude=0 sits on a circle of
    `horizon_radius` pixels. Azimuth is encoded as the RA-analog and
    altitude as the Dec-analog. The 185° lens FOV means usable pixels
    extend slightly past the horizon_radius circle (altitude ≲ -2.5°).

    Parameters
    ----------
    filename : str
        FITS filename. Compressed (.gz, .bz2) inputs are supported.
    rotation : float or None (default=None)
        Camera rotation w.r.t. true north, in degrees. When None, resolved from
        the calibration epoch nearest the frame's time (see alcor_calibration).
    xcen : int (default=696)
        X center of illuminated region in original image coordinates.
    ycen : int (default=698)
        Y center of illuminated region in original image coordinates.
    radius : int (default=680)
        Half-width of the trimmed square around (xcen, ycen).
    horizon_radius : float (default=662)
        Pixel radius from zenith at which altitude=0.
    xshift : float or None (default=None)
        Zenith offset from the array center in x (pixels). When None, resolved
        from the nearest calibration epoch. Applied via scipy.ndimage.shift.
    yshift : float or None (default=None)
        Zenith offset from the array center in y (pixels). When None, resolved
        from the nearest calibration epoch.
    radial_coeffs : tuple of float or None (default=None)
        The (k1, k3, k5) plate-solution coefficients. When None, resolved from
        the nearest calibration epoch. The idealized mapping is (1.0, 0.0, 0.0).
    sip_degree : int (default=5)
        Degree of the SIP polynomial used to encode lens distortion in the
        WCS. A degree of 5 is required to represent the ``k5`` term exactly;
        lower values may be used when only ``k3`` is non-zero.

    Returns
    -------
    im : ndarray
        Zenith-centered, north-up image of shape (2*radius, 2*radius, 3).
    wcs : `astropy.wcs.WCS`
        ARC-projection WCS mapping pixel (x, y) ↔ (azimuth, altitude).
    """
    if rotation is None or xshift is None or yshift is None or radial_coeffs is None:
        cal = _alcor_frame_calibration(filename)
        if rotation is None:
            rotation = cal["rotation"]
        if xshift is None:
            xshift = cal["xshift"]
        if yshift is None:
            yshift = cal["yshift"]
        if radial_coeffs is None:
            radial_coeffs = cal["radial_coeffs"]

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


def alcor_proc_fits(filename, output_file=None, overwrite=False, **kwargs):
    """
    Process an alcor OMEA 8C FITS file via `load_alcor_fits` and write a new
    FITS file containing the zenith-centered, north-up image with the
    alt/az WCS encoded in the header.

    Parameters
    ----------
    filename : str or `~pathlib.Path`
        Input FITS file.
    output_file : str or `~pathlib.Path` or None (default=None)
        Output path. If None, derived from `filename` by replacing the
        first `.fits` substring with `_proc.fits`.
    overwrite : bool (default=False)
        Passed through to `fits.PrimaryHDU.writeto`.
    **kwargs
        Forwarded to `load_alcor_fits` (rotation, xcen, ycen, radius,
        horizon_radius).

    Returns
    -------
    output_file : `~pathlib.Path`
        Path to the written FITS file.
    """
    im, wcs = load_alcor_fits(filename, **kwargs)
    if output_file is None:
        stem = str(filename)
        for ext in (".fits.bz2", ".fits.gz", ".fits"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        output_file = stem + "_proc.fits"
    output_file = Path(output_file)

    cube = np.transpose(np.flipud(im), axes=(2, 0, 1)).astype(np.float32)
    hdu = fits.PrimaryHDU(data=cube, header=wcs.to_header(relax=True))
    hdu.writeto(output_file, overwrite=overwrite)
    return output_file


def alcor_keogram(input_dir, pattern="*.fits.bz2", workers=1, progress=False, progress_file=None, **kwargs):
    """
    Build a keogram from a directory of alcor OMEA 8C FITS images.

    Each input image is loaded with `load_alcor_fits`, and the center column
    of the processed RGB image is copied into the next column of the keogram.
    The DATE header keyword from each FITS file is collected in the same order
    as the keogram columns.

    Parameters
    ----------
    input_dir : str or `~pathlib.Path`
        Directory containing alcor FITS images.
    pattern : str (default="*.fits.bz2")
        Glob pattern used to select files from `input_dir`.
    workers : int or None (default=1)
        Number of worker processes used to load center columns. A value of 1
        runs serially. A value of None uses the process pool default.
    progress : bool (default=False)
        If True, write a progress bar while images are loaded.
    progress_file : file-like or None (default=None)
        Output stream for the progress bar. Defaults to stderr when
        `progress` is True.
    **kwargs
        Forwarded to `load_alcor_fits` (rotation, xcen, ycen, radius,
        horizon_radius).

    Returns
    -------
    keogram : ndarray
        RGB keogram of shape (image_height, number_of_images, 3).
    timestamps : list of str
        DATE header values corresponding to the keogram columns.
    files : list of `~pathlib.Path`
        Input files used to build the keogram, in column order.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {input_dir}")
    if workers is not None and workers < 1:
        raise ValueError("workers must be None or a positive integer")

    strips = [None] * len(files)
    timestamps = [None] * len(files)
    tasks = [(index, filename, kwargs) for index, filename in enumerate(files)]
    completed = 0

    if workers == 1:
        for task in tasks:
            index, timestamp, strip, label = _load_alcor_center_column(task)
            strips[index] = strip
            timestamps[index] = timestamp
            completed += 1
            if progress:
                _print_progress(completed, len(files), label, file=progress_file)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_load_alcor_center_column, task) for task in tasks]
            for future in as_completed(futures):
                index, timestamp, strip, label = future.result()
                strips[index] = strip
                timestamps[index] = timestamp
                completed += 1
                if progress:
                    _print_progress(completed, len(files), label, file=progress_file)

    keogram = np.stack(strips, axis=1)
    return keogram, timestamps, files


def _load_alcor_center_column(task):
    index, filename, kwargs = task
    filename = Path(filename)
    with fits.open(filename) as hdul:
        timestamp = hdul[0].header.get("DATE", "")

    im, _ = load_alcor_fits(filename, **kwargs)
    center_column = im.shape[1] // 2
    return index, timestamp, im[:, center_column, :], filename.name


def _print_progress(current, total, label="", width=32, file=None):
    if file is None:
        file = sys.stderr

    fraction = current / total
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    message = f"\r[{bar}] {current}/{total} {fraction:>6.1%}"
    if label:
        message += f" {label}"
    if current == total:
        message += "\n"

    print(message, end="", file=file, flush=True)


def save_alcor_keogram_plot(
    keogram,
    timestamps,
    output_file,
    powerstretch=0.75,
    contrast=0.35,
    gscale=0.7,
    bscale=1.7,
    figsize=(12, 6),
    dpi=150,
):
    """
    Save a timestamp-labeled plot of an alcor keogram.

    Parameters
    ----------
    keogram : ndarray
        RGB keogram as returned by `alcor_keogram`.
    timestamps : sequence of str
        DATE header values corresponding to the keogram columns.
    output_file : str or `~pathlib.Path`
        Output figure filename. The format is inferred from the extension.
    powerstretch : float (default=0.75)
        Power-stretch exponent.
    contrast : float (default=0.35)
        ZScale contrast factor.
    gscale : float (default=0.7)
        Green channel scale factor.
    bscale : float (default=1.7)
        Blue channel scale factor.
    figsize : tuple (default=(12, 6))
        Matplotlib figure size in inches.
    dpi : int (default=150)
        Output figure resolution.

    Returns
    -------
    output_file : `~pathlib.Path`
        Path to the written plot.
    """
    output_file = Path(output_file)

    im = np.array(keogram, dtype=float, copy=True)
    im[:, :, 1] *= gscale
    im[:, :, 2] *= bscale
    stretch = viz.PowerStretch(powerstretch) + viz.ZScaleInterval(contrast=contrast)
    im = stretch(im)

    times = _parse_timestamps(timestamps)
    fig, ax = plt.subplots(figsize=figsize)
    if times is None:
        timestamp_edges = None
    else:
        xvalues = mdates.date2num(times)
        timestamp_edges = _timestamp_edges(xvalues)

    if timestamp_edges is None:
        ax.imshow(im, aspect="auto", origin="upper")
        ax.set_yticks([0, (keogram.shape[0] - 1) / 2.0, keogram.shape[0] - 1])
    else:
        yedges = np.arange(keogram.shape[0] + 1)
        ax.pcolormesh(timestamp_edges, yedges, im, shading="flat", rasterized=True)
        ax.invert_yaxis()
        ax.set_yticks([0, keogram.shape[0] / 2.0, keogram.shape[0]])
    ax.set_yticklabels(["N", "Z", "S"])
    ax.set_xlabel("UT")

    if times is None:
        ax.set_xlim(-0.5, keogram.shape[1] - 0.5)
    else:
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    return output_file


def _timestamp_edges(xvalues):
    xvalues = np.asarray(xvalues, dtype=float)
    if len(xvalues) == 0:
        return None
    if len(xvalues) == 1:
        dx = 1.0 / 24.0
        return np.array([xvalues[0] - dx / 2.0, xvalues[0] + dx / 2.0])

    dx = np.diff(xvalues)
    if not np.all(np.isfinite(dx)) or np.any(dx <= 0):
        return None

    edges = np.empty(len(xvalues) + 1, dtype=float)
    edges[1:-1] = xvalues[:-1] + dx / 2.0
    edges[0] = xvalues[0] - dx[0] / 2.0
    edges[-1] = xvalues[-1] + dx[-1] / 2.0
    return edges


def save_alcor_keogram_fits(keogram, timestamps, output_file="keogram.fits", overwrite=False):
    """
    Save an alcor keogram and its timestamps to a FITS file.

    Parameters
    ----------
    keogram : ndarray
        RGB keogram as returned by `alcor_keogram`, with shape
        (image_height, number_of_images, 3).
    timestamps : sequence of str
        DATE header values corresponding to the keogram columns.
    output_file : str or `~pathlib.Path` (default="keogram.fits")
        Output FITS filename.
    overwrite : bool (default=False)
        Passed through to `fits.HDUList.writeto`.

    Returns
    -------
    output_file : `~pathlib.Path`
        Path to the written FITS file.
    """
    output_file = Path(output_file)
    cube = np.transpose(keogram, axes=(2, 0, 1)).astype(np.float32)
    primary = fits.PrimaryHDU(data=cube)
    primary.header["CTYPE1"] = "TIME"
    primary.header["CTYPE2"] = "OFFSET"
    primary.header["CTYPE3"] = "COLOR"
    primary.header["BUNIT"] = "adu"

    timestamps = np.asarray(timestamps, dtype=str)
    width = max(1, max(len(timestamp) for timestamp in timestamps))
    columns = [fits.Column(name="DATE", format=f"{width}A", array=timestamps)]
    table = fits.BinTableHDU.from_columns(columns, name="TIMESTAMPS")

    hdul = fits.HDUList([primary, table])
    hdul.writeto(output_file, overwrite=overwrite)
    return output_file


def load_alcor_keogram_fits(filename):
    """
    Load an alcor keogram FITS file.

    Parameters
    ----------
    filename : str or `~pathlib.Path`
        Keogram FITS file written by `save_alcor_keogram_fits`.

    Returns
    -------
    keogram : ndarray
        RGB keogram with shape (image_height, number_of_images, 3).
    timestamps : list of str
        DATE values from the TIMESTAMPS table extension.
    """
    with fits.open(filename) as hdul:
        keogram = np.transpose(hdul[0].data, axes=(1, 2, 0))
        timestamps = list(hdul["TIMESTAMPS"].data["DATE"])

    return keogram, timestamps


def plot_alcor_keogram_fits(filename, output_file=None, **kwargs):
    """
    Create a keogram plot from an alcor keogram FITS file.

    Parameters
    ----------
    filename : str or `~pathlib.Path`
        Keogram FITS file written by `save_alcor_keogram_fits`.
    output_file : str or `~pathlib.Path` or None (default=None)
        Output plot path. If None, replaces the FITS suffix with `.png`.
    **kwargs
        Forwarded to `save_alcor_keogram_plot`.

    Returns
    -------
    output_file : `~pathlib.Path`
        Path to the written plot.
    """
    filename = Path(filename)
    if output_file is None:
        stem = str(filename)
        for ext in (".fits.gz", ".fits"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        output_file = stem + ".png"

    keogram, timestamps = load_alcor_keogram_fits(filename)
    return save_alcor_keogram_plot(keogram, timestamps, output_file, **kwargs)


def _parse_timestamps(timestamps):
    clean_timestamps = [timestamp for timestamp in timestamps if timestamp]
    if len(clean_timestamps) != len(timestamps):
        return None

    try:
        return Time(timestamps).datetime
    except ValueError:
        return None


def plot_alcor_fits(filename, outimage=None, outfig=None, rotation=None, xcen=696, ycen=698, radius=680,
                    horizon_radius=662, powerstretch=0.75, contrast=0.35, gscale=0.7, bscale=1.7, figsize=12):
    """
    Take a FITS file as produced by the alcor OMEA 8C and create a trimmed, rotated, and annotated figure
    file appropriate for display

    Parameters
    ----------
    filename : str
        FITS filename of image. Uses astropy.io.fits so gz and bz2 extentions are allowed.
    outimage : str (default=None)
        If not None, write out raw, unannotated image
    outfig : str (default=None)
        If not None, write out annotated image as produced by matplotlib
    rotation : float or None (default=None)
        Camera rotation w.r.t. true north (deg). When None, resolved from the
        calibration epoch nearest the frame date.
    xcen : int (default=696)
        X center of illuminated region in original image coordinates
    ycen : int (default=698)
        Y center of illuminated region in original image coordinates
    radius : float (default=680)
        Radius of illuminated region
    horizon_radius : float (default=662)
        Pixel radius from zenith at which altitude=0.
    powerstretch : float (default=0.75)
        Power of the stretch function to use
    contrast : float (default=0.35)
        ZScale contrast factor
    gscale : float (default=0.7)
        Scale factor to apply to green channel
    bscale : float (default=1.7)
        Scale factor to apply to blue channel
    figsize : float (default=12)
        Size of matplotlib figure in inches
    """
    im, wcs = load_alcor_fits(
        filename,
        rotation=rotation,
        xcen=xcen,
        ycen=ycen,
        radius=radius,
        horizon_radius=horizon_radius,
    )
    im[:, :, 1] *= gscale  # the factors to scale the green and blue channels were determined empirically and provide a
    im[:, :, 2] *= bscale  # reasonably good white/color balance for both day and night images.
    stretch = viz.PowerStretch(powerstretch) + viz.ZScaleInterval(contrast=contrast)  # apply a power-law stretch and
                                                                                      # zscale interval to the image data
    im = stretch(im)

    if outimage is not None:
        plt.imsave(outimage, im)

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    circle = Circle((radius, radius), radius, facecolor='none', edgecolor=(0, 0, 0), linewidth=1, alpha=0.5)
    ax.add_patch(circle)
    ax.axis("off")
    im_plot = plt.imshow(im)
    im_plot.set_clip_path(circle)

    pax = fig.add_subplot(111, polar=True, label='polar')
    pax.set_facecolor("None")
    pax.set_theta_zero_location("N")
    # Use the WCS to map altitude ticks to the correct radial fraction. The polar overlay
    # spans the figure region, so r=1 corresponds to a pixel distance of `radius` from zenith.
    tick_alts = np.array([75, 60, 45, 30, 15])
    px, py = wcs.world_to_pixel_values(np.zeros_like(tick_alts), tick_alts)
    yticks = np.hypot(px - (radius - 0.5), py - (radius - 0.5)) / radius
    ylabels = [f" {a}°" for a in tick_alts]
    pax.set_yticks(yticks, labels=ylabels, color="white", alpha=0.5, fontsize=16)
    pax.set_rlabel_position(90)
    pax.tick_params(grid_alpha=0.5)
    pax.tick_params(axis='x', labelsize=16, labelcolor='silver', pad=10)

    if outfig is not None:
        plt.savefig(outfig, transparent=True, bbox_inches='tight', pad_inches = 0)

    return fig


def alcor_proc_fits_cli():
    """
    CLI entry point for `alcor_proc_fits`. Writes a processed FITS file with
    the alt/az WCS encoded in the header.
    """
    parser = argparse.ArgumentParser(
        description="Process an alcor OMEA 8C FITS image into a zenith-centered, north-up FITS file with alt/az WCS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument("-o", "--output", default=None, help="Output FITS path (default: <input>_proc.fits).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")
    parser.add_argument("--rotation", type=float, default=None,
                        help="Camera rotation w.r.t. true north (deg); "
                             "default resolves the calibration epoch nearest the frame date.")
    parser.add_argument("--xcen", type=int, default=696, help="X center of illuminated region.")
    parser.add_argument("--ycen", type=int, default=698, help="Y center of illuminated region.")
    parser.add_argument("--radius", type=int, default=680, help="Half-width of trimmed square around (xcen, ycen).")
    parser.add_argument("--horizon-radius", type=float, default=662, help="Pixel radius from zenith at altitude=0.")
    args = parser.parse_args()

    out = alcor_proc_fits(
        args.filename,
        output_file=args.output,
        overwrite=args.overwrite,
        rotation=args.rotation,
        xcen=args.xcen,
        ycen=args.ycen,
        radius=args.radius,
        horizon_radius=args.horizon_radius,
    )
    print(out)


def alcor_keogram_cli():
    """
    CLI entry point for `alcor_keogram`. Writes a timestamp-labeled keogram
    figure and, optionally, the DATE header values used for the x-axis.
    """
    parser = argparse.ArgumentParser(
        description="Build a keogram from the center columns of alcor OMEA 8C FITS images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory containing alcor FITS images.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output keogram plot path (default: <input_dir_name>_keogram.png).",
    )
    parser.add_argument(
        "--fits-output",
        default=None,
        help="Output keogram FITS path (default: <input_dir_name>_keogram.fits).",
    )
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob pattern for input files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for FITS loading (default: one per process-pool default).",
    )
    parser.add_argument("--no-progress", action="store_true", help="Do not show progress while loading images.")
    parser.add_argument(
        "--timestamps-output",
        default=None,
        help="Optional text file to write DATE header values, one per line.",
    )
    parser.add_argument("--rotation", type=float, default=None,
                        help="Camera rotation w.r.t. true north (deg); "
                             "default resolves the calibration epoch nearest the frame date.")
    parser.add_argument("--xcen", type=int, default=696, help="X center of illuminated region.")
    parser.add_argument("--ycen", type=int, default=698, help="Y center of illuminated region.")
    parser.add_argument("--radius", type=int, default=680, help="Half-width of trimmed square around (xcen, ycen).")
    parser.add_argument("--horizon-radius", type=float, default=662, help="Pixel radius from zenith at altitude=0.")
    parser.add_argument("--powerstretch", type=float, default=0.75, help="Power-stretch exponent.")
    parser.add_argument("--contrast", type=float, default=0.35, help="ZScale contrast factor.")
    parser.add_argument("--gscale", type=float, default=0.7, help="Green channel scale factor.")
    parser.add_argument("--bscale", type=float, default=1.7, help="Blue channel scale factor.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(12, 6), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=150, help="Output figure resolution.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = args.output
    if output is None:
        output = f"{input_dir.name}_keogram.png"
    fits_output = args.fits_output
    if fits_output is None:
        fits_output = f"{input_dir.name}_keogram.fits"

    keogram, timestamps, _ = alcor_keogram(
        input_dir,
        pattern=args.pattern,
        workers=args.workers,
        progress=not args.no_progress,
        rotation=args.rotation,
        xcen=args.xcen,
        ycen=args.ycen,
        radius=args.radius,
        horizon_radius=args.horizon_radius,
    )
    output_file = save_alcor_keogram_plot(
        keogram,
        timestamps,
        output,
        powerstretch=args.powerstretch,
        contrast=args.contrast,
        gscale=args.gscale,
        bscale=args.bscale,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )
    fits_output = save_alcor_keogram_fits(
        keogram,
        timestamps,
        fits_output,
        overwrite=True,
    )

    if args.timestamps_output is not None:
        timestamps_output = Path(args.timestamps_output)
        timestamps_output.write_text("\n".join(timestamps) + "\n")

    print(output_file)
    print(fits_output)


def plot_alcor_keogram_fits_cli():
    """
    CLI entry point for `plot_alcor_keogram_fits`. Writes a PNG plot from an
    alcor keogram FITS file.
    """
    parser = argparse.ArgumentParser(
        description="Render a timestamp-labeled keogram plot from an alcor keogram FITS file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input alcor keogram FITS file.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output keogram plot path (default: <input>.png).",
    )
    parser.add_argument("--powerstretch", type=float, default=0.75, help="Power-stretch exponent.")
    parser.add_argument("--contrast", type=float, default=0.35, help="ZScale contrast factor.")
    parser.add_argument("--gscale", type=float, default=0.7, help="Green channel scale factor.")
    parser.add_argument("--bscale", type=float, default=1.7, help="Blue channel scale factor.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(12, 6), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=150, help="Output figure resolution.")
    args = parser.parse_args()

    output_file = plot_alcor_keogram_fits(
        args.filename,
        output_file=args.output,
        powerstretch=args.powerstretch,
        contrast=args.contrast,
        gscale=args.gscale,
        bscale=args.bscale,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )
    print(output_file)


def plot_alcor_fits_cli():
    """
    CLI entry point for `plot_alcor_fits`. Writes an annotated PDF figure by
    default, named after the input file with `.fits` replaced by `.pdf`.
    """
    parser = argparse.ArgumentParser(
        description="Render an annotated all-sky figure from an alcor OMEA 8C FITS image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument(
        "-o", "--outfig", default=None,
        help="Output figure path (default: <input>.pdf). Format inferred from extension."
    )
    parser.add_argument("--outimage", default=None, help="If set, also write the raw stretched image to this path.")
    parser.add_argument("--rotation", type=float, default=None,
                        help="Camera rotation w.r.t. true north (deg); "
                             "default resolves the calibration epoch nearest the frame date.")
    parser.add_argument("--xcen", type=int, default=696, help="X center of illuminated region.")
    parser.add_argument("--ycen", type=int, default=698, help="Y center of illuminated region.")
    parser.add_argument("--radius", type=int, default=680, help="Half-width of trimmed square around (xcen, ycen).")
    parser.add_argument("--horizon-radius", type=float, default=662, help="Pixel radius from zenith at altitude=0.")
    parser.add_argument("--powerstretch", type=float, default=0.75, help="Power-stretch exponent.")
    parser.add_argument("--contrast", type=float, default=0.35, help="ZScale contrast factor.")
    parser.add_argument("--gscale", type=float, default=0.7, help="Green channel scale factor.")
    parser.add_argument("--bscale", type=float, default=1.7, help="Blue channel scale factor.")
    parser.add_argument("--figsize", type=float, default=12, help="Matplotlib figure size in inches.")
    args = parser.parse_args()

    outfig = args.outfig
    if outfig is None:
        stem = str(args.filename)
        for ext in (".fits.bz2", ".fits.gz", ".fits"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        outfig = stem + ".pdf"

    plot_alcor_fits(
        args.filename,
        outimage=args.outimage,
        outfig=outfig,
        rotation=args.rotation,
        xcen=args.xcen,
        ycen=args.ycen,
        radius=args.radius,
        horizon_radius=args.horizon_radius,
        powerstretch=args.powerstretch,
        contrast=args.contrast,
        gscale=args.gscale,
        bscale=args.bscale,
        figsize=args.figsize,
    )
    print(outfig)


def _format_calibration_entry(result):
    """Format a calibration result as a paste-ready ALCOR_CALIBRATIONS entry."""
    rc = tuple(float(c) for c in result["radial_coeffs"])
    return (f'    {{"epoch": "{result["epoch"]}", '
            f'"xshift": {result["xshift"]:.3f}, '
            f'"yshift": {result["yshift"]:.3f}, '
            f'"rotation": {result["rotation"]:.4f}, '
            f'"radial_coeffs": {rc!r}}},')


def fit_alcor_wcs_cli():
    """
    CLI entry point for ``fit_alcor_wcs``. Aggregates bright-star matches across
    the dark-sky frames of a night and prints the refined geometry constants
    ready to paste into the module defaults.
    """
    parser = argparse.ArgumentParser(
        description="Calibrate the alcor lens WCS from bright stars across a night.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory containing alcor FITS images.")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob pattern for input files.")
    parser.add_argument("--vmag-limit", type=float, default=3.0, help="Faintest Vmag to use.")
    parser.add_argument("--sun-alt-max", type=float, default=-18.0,
                        help="Use frames with Sun altitude below this (deg).")
    parser.add_argument("--min-alt", type=float, default=10.0, help="Minimum star altitude (deg).")
    parser.add_argument("--tolerance", type=float, default=12.0, help="Match tolerance (pixels).")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap number of frames used.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes for per-frame detection "
                             "(default: one per available core).")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print per-file processing/rejection messages.")
    parser.add_argument("--residual-plot", default=None, help="Optional residual-vs-zenith PNG path.")
    args = parser.parse_args()

    log = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    result = fit_alcor_wcs(
        args.input_dir, pattern=args.pattern, vmag_limit=args.vmag_limit,
        sun_alt_max=args.sun_alt_max, min_alt=args.min_alt, tolerance=args.tolerance,
        max_frames=args.max_frames, workers=args.workers, log=log,
    )
    print(f"# matched stars: {result['n_matched']}")
    print(f"# residual RMS (pix): {result['residual_rms']:.3f}")
    print("# add this entry to ALCOR_CALIBRATIONS in alcor.py:")
    print(_format_calibration_entry(result))
    if args.residual_plot is not None:
        out = save_alcor_residual_plot(result["alt"], result["az"], result["x"],
                                       result["y"], result, args.residual_plot)
        print(out)
