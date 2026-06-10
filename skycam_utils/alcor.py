import argparse
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.ndimage import rotate
from scipy.optimize import least_squares
from scipy.ndimage import shift as ndimage_shift
from scipy.spatial import cKDTree
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
from astropy.coordinates import SkyCoord, AltAz, get_sun, get_body

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

    ``rotation`` is the camera's absolute azimuth-frame rotation (degrees).
    ``fit_alcor_wcs`` fits against a neutral (un-rotated, un-shifted) frame, so the
    recovered value is absolute; ``load_alcor_fits`` then applies exactly this
    rotation to the image. It defaults to 0.0, the idealized/centered frame.

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
                horizon_radius=ALCOR_HORIZON_RADIUS, fit_k5=False):
    """
    Robust least-squares fit of the lens geometry to matched stars.

    By default fits (xshift, yshift, rotation, k3) with k1 held at 1.0 (the zenith
    plate scale is set by horizon_radius) and k5 at 0.0. k3 and k5 are nearly
    collinear in rho over [0, 1], so fitting both is ill-conditioned on *dirty*
    data and runs away to large cancelling values (e.g. k3=-0.58, k5=3.6) that are
    unphysical despite a tolerable RMS -- which is why k3 alone is the default
    (the model the shipped 2024 constants used). With ``fit_k5=True`` the odd
    quintic term k5 is fit as well: appropriate only on a clean, well-distributed
    match set (asterism-verified, spanning the full zenith range), where the radial
    residual that k3 alone leaves near the horizon can be captured. A ``soft_l1``
    loss downweights mismatched/noise detections, common in this sparse
    bright-star field. Returns an updated params dict with
    radial_coeffs=(1.0, k3, k5) (k5=0.0 unless ``fit_k5``).
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    obs_x = np.asarray(obs_x, dtype=float)
    obs_y = np.asarray(obs_y, dtype=float)
    init_k3 = init_params["radial_coeffs"][1]
    init_k5 = init_params["radial_coeffs"][2]

    if fit_k5:
        p0 = np.array([init_params["xshift"], init_params["yshift"],
                       init_params["rotation"], init_k3, init_k5], dtype=float)

        def residuals(p):
            xshift, yshift, rot, k3, k5 = p
            x, y = _predict_pixels(alt, az, xshift=xshift, yshift=yshift, rotation=rot,
                                   radial_coeffs=(1.0, k3, k5), radius=radius,
                                   horizon_radius=horizon_radius)
            return np.concatenate([x - obs_x, y - obs_y])

        result = least_squares(residuals, p0, loss="soft_l1", f_scale=3.0)
        xshift, yshift, rot, k3, k5 = result.x
        return dict(xshift=float(xshift), yshift=float(yshift), rotation=float(rot),
                    radial_coeffs=(1.0, float(k3), float(k5)))

    p0 = np.array([init_params["xshift"], init_params["yshift"],
                   init_params["rotation"], init_k3], dtype=float)

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


def assign_alcor_matches(cat, det, params, tolerance,
                         radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                         n_neighbors=5, min_corroborating=2, pattern_tol=3.0,
                         brightness=True):
    """
    Assign catalog stars to detected sources against a *fixed* geometry.

    This never refits the geometry internally; ``params`` (``xshift``,
    ``yshift``, ``rotation``, ``radial_coeffs``) are the fixed geometry used for
    the whole frame. The steps are:

    1. Predict each catalog star's pixel ``(px, py)`` with :func:`_predict_pixels`
       and build a `~scipy.spatial.cKDTree` of detections and of predicted
       catalog pixels.
    2. Form candidate edges (catalog i, detection j) with separation
       <= ``tolerance``, group them into connected components, and resolve each
       component. An isolated 1:1 candidate is the mutual-nearest case and is
       accepted directly. A contested cluster (several catalog stars and/or
       detections within tolerance) is resolved by **relative-brightness rank
       pairing**: detections sorted by ``flux`` descending are paired with catalog
       stars sorted by ``Vmag`` ascending, in order (within ``tolerance``). With
       ``brightness=False`` or missing ``flux``/``Vmag`` columns the cluster is
       resolved greedily by nearest separation instead. Because brightness is only
       consulted *within* a contested cluster of nearby stars, spatially or
       temporally patchy cloud extinction (which dims a local patch in common)
       never enters a global comparison.
    3. **Local-pattern (asterism) verification.** For each tentative pair i->j,
       look at catalog i's ``n_neighbors`` nearest catalog neighbors that also have
       a tentative pair. The pair is accepted iff at least ``min_corroborating`` of
       them corroborate the local constellation -- their detection offset matches
       the predicted offset to within ``pattern_tol``:
       ``||(det_jn - det_j) - (pred_in - pred_i)|| <= pattern_tol``. Pairs with
       fewer than ``min_corroborating`` paired neighbors are kept (too little local
       evidence to reject); crowded-region mispairs, which sit among well-matched
       neighbors yet break the constellation, are rejected.

    Returns an ``hstack`` of the accepted catalog and detection rows (catalog
    columns then detection columns); an empty table if nothing matches.
    """
    px, py = _predict_pixels(
        cat["Alt"], cat["Az"], xshift=params["xshift"], yshift=params["yshift"],
        rotation=params["rotation"], radial_coeffs=tuple(params["radial_coeffs"]),
        radius=radius, horizon_radius=horizon_radius,
    )
    px = np.atleast_1d(np.asarray(px, dtype=float))
    py = np.atleast_1d(np.asarray(py, dtype=float))
    det_x = np.asarray(det["xcentroid"], dtype=float)
    det_y = np.asarray(det["ycentroid"], dtype=float)

    n_cat = px.size
    n_det = det_x.size
    empty = hstack([Table(cat[[]]), Table(det[[]])])
    if n_cat == 0 or n_det == 0:
        return empty

    cat_xy = np.column_stack([px, py])
    det_xy = np.column_stack([det_x, det_y])
    det_tree = cKDTree(det_xy)
    cat_tree = cKDTree(cat_xy)

    has_bright = (brightness and "Vmag" in cat.colnames and "flux" in det.colnames)
    vmag = np.asarray(cat["Vmag"], dtype=float) if "Vmag" in cat.colnames else None
    flux = np.asarray(det["flux"], dtype=float) if "flux" in det.colnames else None

    # candidate detections within tolerance of each catalog star
    cat_cands = det_tree.query_ball_point(cat_xy, tolerance)

    # --- connected components over the bipartite candidate graph ---
    # nodes 0..n_cat-1 are catalog stars, n_cat..n_cat+n_det-1 are detections.
    parent = list(range(n_cat + n_det))

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for i, cands in enumerate(cat_cands):
        for j in cands:
            _union(i, n_cat + j)

    comp_cat = {}
    comp_det = {}
    for i in range(n_cat):
        if cat_cands[i]:
            comp_cat.setdefault(_find(i), []).append(i)
    for j in range(n_det):
        root = _find(n_cat + j)
        if root in comp_cat:
            comp_det.setdefault(root, []).append(j)

    # --- resolve each component into tentative (cat i, det j) pairs ---
    tentative = {}  # cat i -> det j
    for root, cis in comp_cat.items():
        djs = comp_det.get(root, [])
        if not djs:
            continue
        if len(cis) == 1 and len(djs) == 1:
            tentative[cis[0]] = djs[0]
            continue
        cis_arr = np.asarray(cis, dtype=int)
        djs_arr = np.asarray(djs, dtype=int)
        if has_bright:
            ci_order = cis_arr[np.argsort(vmag[cis_arr])]        # brightest catalog first
            dj_order = djs_arr[np.argsort(-flux[djs_arr])]       # brightest detection first
            for k in range(min(len(ci_order), len(dj_order))):
                i, j = int(ci_order[k]), int(dj_order[k])
                if np.hypot(det_x[j] - px[i], det_y[j] - py[i]) <= tolerance:
                    tentative[i] = j
        else:
            edges = []
            for i in cis_arr:
                for j in djs_arr:
                    d = np.hypot(det_x[j] - px[i], det_y[j] - py[i])
                    if d <= tolerance:
                        edges.append((d, int(i), int(j)))
            edges.sort()
            used_c, used_d = set(), set()
            for d, i, j in edges:
                if i in used_c or j in used_d:
                    continue
                tentative[i] = j
                used_c.add(i)
                used_d.add(j)

    if not tentative:
        return empty

    # --- local-pattern (asterism) verification ---
    k_query = min(n_neighbors + 1, n_cat)
    accepted_cat, accepted_det = [], []
    for i, j in tentative.items():
        _, idxs = cat_tree.query(cat_xy[i], k=k_query)
        neighbors = [int(n) for n in np.atleast_1d(idxs)
                     if int(n) != i and int(n) < n_cat]
        paired = [n for n in neighbors if n in tentative]
        if len(paired) < min_corroborating:
            accepted_cat.append(i)
            accepted_det.append(j)
            continue
        corro = 0
        for n in paired:
            jn = tentative[n]
            pred_off = cat_xy[n] - cat_xy[i]
            det_off = det_xy[jn] - det_xy[j]
            if np.hypot(*(det_off - pred_off)) <= pattern_tol:
                corro += 1
        if corro >= min_corroborating:
            accepted_cat.append(i)
            accepted_det.append(j)

    if not accepted_cat:
        return empty
    return hstack([Table(cat[accepted_cat]), Table(det[accepted_det])])


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
    index, filename, vmag_limit, min_alt, fwhm, threshold_sigma, max_detections = task
    filename = Path(filename)
    time = _frame_time(filename)
    im, _ = load_alcor_fits(filename, rotation=0.0, xshift=0.0, yshift=0.0,
                            radial_coeffs=(1.0, 0.0, 0.0))
    cat = alcor_reference_altaz(time, vmag_limit=vmag_limit, min_alt=min_alt)
    det = detect_alcor_stars(im, fwhm=fwhm, threshold_sigma=threshold_sigma,
                             max_detections=max_detections)
    if len(det) < 3:
        return index, None, None, f"no stars detected ({len(det)} < 3)"
    if len(cat) < 3:
        return index, None, None, f"too few catalog stars ({len(cat)} < 3)"
    return index, cat, det, None


def fit_alcor_wcs(input_dir, pattern="*.fits.bz2", vmag_limit=4.0, sun_alt_max=-18.0,
                  moon_alt_max=-6.0,
                  min_alt=10.0, tolerance=3.0, tolerance_start=12.0, match_rounds=4,
                  n_neighbors=5, min_corroborating=2, pattern_tol=3.0,
                  fit_k5=False, fwhm=3.0, threshold_sigma=5.0, max_detections=200,
                  max_frames=None, workers=1, log=None):
    """
    Calibrate the alcor lens geometry by aggregating bright-star matches across
    all dark-sky frames in ``input_dir``.

    Frames are selected with :func:`select_dark_frames` (Sun below ``sun_alt_max``
    and Moon below ``moon_alt_max``, since moonlight scatter corrupts source
    detection).
    Each frame's detections are capped to the brightest ``max_detections`` and
    matched against the current geometry with :func:`assign_alcor_matches` (kd-tree
    candidates, asterism pattern verification, local brightness tie-break). The
    matcher never refits per frame; instead the whole night is pooled and fit once
    per round under a single global geometry. The match tolerance tightens
    geometrically over ``match_rounds`` rounds from ``tolerance_start`` down to
    ``tolerance`` so that each round's better seed admits a cleaner pool. A final
    pool at the tightest tolerance is fit after 3*MAD outlier rejection. The
    matcher's asterism knobs (``n_neighbors``, ``min_corroborating``,
    ``pattern_tol``) are forwarded to :func:`assign_alcor_matches`; loosening
    ``pattern_tol``/``tolerance`` recovers more (and higher-residual) matches when
    the seed geometry leaves real distortion unmodeled.

    The fit runs on a neutral (uncalibrated) frame -- loaded with no recentering,
    rotation, or radial distortion -- so the recovered (xshift, yshift, rotation,
    radial_coeffs) are the ABSOLUTE geometry constants for the night, suitable for
    baking into ``ALCOR_CALIBRATIONS``. It is warm-started from the nearest
    existing epoch (via :func:`alcor_calibration` at the night's median time).

    Returns a dict with the fitted absolute parameters plus an ``epoch`` date
    string (the night's UT date, or the seed epoch when no frame can be timed),
    ``n_matched``, ``residual_rms``, ``matched_fraction`` (matched stars divided by
    the available catalog-star-frames, so contamination/coverage is visible), and
    per-match arrays (``alt``, ``az``, ``x``, ``y``) for diagnostics.

    The per-frame load/detect/catalog work is the expensive part and is
    independent across frames, so it is parallelized: ``workers=1`` runs
    serially, any larger value (or ``None`` for the process-pool default)
    distributes the frames over a `~concurrent.futures.ProcessPoolExecutor`.
    Pass a ``log`` callable (e.g. ``print``) to report each file's disposition:
    frames skipped because the Sun is above ``sun_alt_max`` or the Moon is above
    ``moon_alt_max``, frames skipped because no stars were detected, and frames
    used (with detected star count).
    """
    if workers is not None and workers < 1:
        raise ValueError("workers must be None or a positive integer")
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    dark = select_dark_frames(files, sun_alt_max=sun_alt_max,
                              moon_alt_max=moon_alt_max, log=log)
    if log is not None:
        dark_set = set(dark)
        for f in files:
            if f not in dark_set:
                log(f"{Path(f).name}: skipped "
                    f"(Sun above {sun_alt_max:g} deg or Moon above {moon_alt_max:g} deg)")
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
    tasks = [(index, f, vmag_limit, min_alt, fwhm, threshold_sigma, max_detections)
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
    available = sum(len(cat) for cat, _ in frames)

    def pool(seed_params, tol):
        a, z, xs, ys = [], [], [], []
        for cat, det in frames:
            matched = assign_alcor_matches(cat, det, params=seed_params, tolerance=tol,
                                           n_neighbors=n_neighbors,
                                           min_corroborating=min_corroborating,
                                           pattern_tol=pattern_tol)
            if len(matched) == 0:
                continue
            a.append(np.asarray(matched["Alt"], dtype=float))
            z.append(np.asarray(matched["Az"], dtype=float))
            xs.append(np.asarray(matched["xcentroid"], dtype=float))
            ys.append(np.asarray(matched["ycentroid"], dtype=float))
        if not a:
            return None
        return (np.concatenate(a), np.concatenate(z),
                np.concatenate(xs), np.concatenate(ys))

    # Tightening tolerance schedule: each round re-pools with the refined seed.
    schedule = np.geomspace(tolerance_start, tolerance, match_rounds)
    params = dict(init)
    for tol in schedule:
        pooled = pool(params, float(tol))
        if pooled is None:
            continue
        alt, az, x, y = pooled
        if len(alt) >= 3:
            params = _fit_params(alt, az, x, y, init_params=params, fit_k5=fit_k5)

    # Final pool at the tightest tolerance, then 3*MAD outlier rejection + refit.
    pooled = pool(params, float(tolerance))
    if pooled is None:
        raise RuntimeError("No matched stars across the selected frames.")
    alt, az, x, y = pooled

    px, py = _predict_pixels(alt, az, xshift=params["xshift"], yshift=params["yshift"],
                             rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]))
    resid = np.hypot(px - x, py - y)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    good = resid < np.median(resid) + 3.0 * 1.4826 * mad
    if good.sum() >= 3:
        params = _fit_params(alt[good], az[good], x[good], y[good], init_params=params,
                             fit_k5=fit_k5)
    px, py = _predict_pixels(alt[good], az[good], xshift=params["xshift"],
                             yshift=params["yshift"], rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]))
    rms = float(np.sqrt(np.mean((px - x[good]) ** 2 + (py - y[good]) ** 2)))

    return {
        **params,
        "epoch": epoch,
        "n_matched": int(good.sum()),
        "residual_rms": rms,
        "matched_fraction": float(int(good.sum()) / available) if available else 0.0,
        "alt": alt[good], "az": az[good], "x": x[good], "y": y[good],
    }


def save_alcor_residual_plot(alt, az, obs_x, obs_y, params, output_file,
                             radius=ALCOR_RADIUS, horizon_radius=ALCOR_HORIZON_RADIUS,
                             figsize=(18, 10), dpi=150, nbins=30, min_per_cell=3):
    """
    Diagnostic plot of the fitted WCS residuals for matched stars, as six panels
    (two rows), and return the output path.

    Top row:

    1. Residual magnitude versus zenith angle, before (idealized equidistant)
       and after (fitted) the refinement -- shows whether the radial model order
       is adequate.
    2. Refined residual magnitude versus azimuth, colored by zenith angle -- a
       residual that varies with azimuth at fixed zenith indicates azimuthal
       asymmetry (lens/sensor decenter or tilt) that the azimuthally-symmetric
       radial model cannot capture.
    3. The refined residual vector field over the detector, binned onto an
       ``nbins`` x ``nbins`` grid (cells with at least ``min_per_cell`` matches)
       and shown as one mean arrow per cell, auto-scaled. Averaging cancels the
       incoherent scatter from mismatches so the coherent structure (radial /
       swirl / elliptical) stands out; per-star arrows would saturate the panel.

    Bottom row decomposes each residual into a **radial** component (along the
    zenith->star direction) and a **tangential** component (perpendicular), which
    discriminates the cause of a leftover residual that the radial model cannot
    remove:

    4. Radial component versus azimuth, colored by zenith. A sinusoid (one cycle
       per 360 deg) at fixed zenith is the signature of a sensor/lens decenter or
       tilt (a 2-D effect); flat scatter about zero is irreducible noise.
    5. Tangential component versus azimuth, colored by zenith. A nonzero
       tangential signal indicates a rotational/swirl term (e.g. residual sensor
       rotation that varies with zenith) the scalar ``rotation`` cannot capture.
    6. Binned mean radial and tangential component versus zenith. A mean radial
       curve that grows smoothly with zenith means the azimuthally-symmetric
       radial basis is itself inadequate (not merely truncated); a mean near zero
       with large per-azimuth spread (panels 4/5) points to a 2-D or noise term.

    Note: when the match tolerance is tight the residual magnitude (panels 1/2)
    is clamped below it; run with a loose ``--tolerance`` and a tight
    ``--pattern-tol`` (the asterism check is offset-invariant) to see the true,
    unclamped residual structure here.
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
    dx, dy = fx - obs_x, fy - obs_y
    after = np.hypot(dx, dy)

    fig, ((ax_z, ax_a, ax_v),
          (ax_rad, ax_tan, ax_prof)) = plt.subplots(2, 3, figsize=figsize)

    ax_z.scatter(z, before, s=8, alpha=0.5, label="idealized")
    ax_z.scatter(z, after, s=8, alpha=0.5, label="refined")
    rms = float(np.sqrt(np.mean(after ** 2))) if after.size else float("nan")
    ax_z.set_title(f"residual vs zenith  (RMS={rms:.2f} px, N={after.size})")
    ax_z.set_xlabel("zenith angle (deg)")
    ax_z.set_ylabel("pixel residual")
    ax_z.legend()

    sc = ax_a.scatter(az, after, s=8, alpha=0.6, c=z, cmap="plasma")
    ax_a.set_title("refined residual vs azimuth")
    ax_a.set_xlabel("azimuth (deg)")
    ax_a.set_ylabel("pixel residual")
    fig.colorbar(sc, ax=ax_a, label="zenith angle (deg)")

    cen = radius - 0.5
    # Bin the residual vectors onto a grid and average per cell, so coherent
    # structure survives while incoherent (mismatch) scatter cancels out.
    extent = 2.0 * radius
    cell = extent / nbins
    cx_i = np.clip((obs_x / cell).astype(int), 0, nbins - 1)
    cy_i = np.clip((obs_y / cell).astype(int), 0, nbins - 1)
    flat = cy_i * nbins + cx_i
    n = nbins * nbins
    count = np.bincount(flat, minlength=n).astype(float)
    sum_dx = np.bincount(flat, weights=dx, minlength=n)
    sum_dy = np.bincount(flat, weights=dy, minlength=n)
    keep = count >= min_per_cell
    cells = np.where(keep)[0]
    mean_dx = sum_dx[cells] / count[cells]
    mean_dy = sum_dy[cells] / count[cells]
    gx = (cells % nbins + 0.5) * cell
    gy = (cells // nbins + 0.5) * cell
    gmag = np.hypot(mean_dx, mean_dy)
    p90 = np.percentile(gmag, 90) if gmag.size else 1.0
    amp = (1.5 * cell) / (p90 + 1e-9)  # 90th-pct arrow spans ~1.5 cells
    q = ax_v.quiver(gx, gy, mean_dx, mean_dy, gmag, angles="xy",
                    scale_units="xy", scale=1.0 / amp, cmap="viridis", width=0.004)
    ax_v.plot(cen, cen, "r+", ms=14, label="zenith (array center)")
    ax_v.set_aspect("equal")
    ax_v.set_xlim(0, extent)
    ax_v.set_ylim(0, extent)
    ax_v.set_title(f"binned mean residual vectors ({nbins}x{nbins}, x{amp:.0f})")
    ax_v.set_xlabel("x (pix)")
    ax_v.set_ylabel("y (pix)")
    ax_v.legend(loc="upper right")
    fig.colorbar(q, ax=ax_v, label="mean |residual| (pix)")

    # Radial/tangential decomposition about the zenith (array center).
    vx = obs_x - cen
    vy = obs_y - cen
    rr = np.hypot(vx, vy)
    safe = rr > 1e-6
    denom = np.where(safe, rr, 1.0)
    rhx = np.where(safe, vx / denom, 0.0)
    rhy = np.where(safe, vy / denom, 0.0)
    rad_comp = dx * rhx + dy * rhy            # + = predicted outward of observed
    tan_comp = dx * (-rhy) + dy * rhx         # + = predicted CCW of observed

    sc_r = ax_rad.scatter(az, rad_comp, s=8, alpha=0.6, c=z, cmap="plasma")
    ax_rad.axhline(0.0, color="k", lw=0.5)
    ax_rad.set_title("radial residual component vs azimuth")
    ax_rad.set_xlabel("azimuth (deg)")
    ax_rad.set_ylabel("radial residual (pix)")
    fig.colorbar(sc_r, ax=ax_rad, label="zenith angle (deg)")

    sc_t = ax_tan.scatter(az, tan_comp, s=8, alpha=0.6, c=z, cmap="plasma")
    ax_tan.axhline(0.0, color="k", lw=0.5)
    ax_tan.set_title("tangential residual component vs azimuth")
    ax_tan.set_xlabel("azimuth (deg)")
    ax_tan.set_ylabel("tangential residual (pix)")
    fig.colorbar(sc_t, ax=ax_tan, label="zenith angle (deg)")

    # Binned mean radial/tangential component vs zenith (the decisive panel).
    if z.size:
        zb = np.linspace(float(z.min()), float(z.max()), 17)
        idx = np.clip(np.digitize(z, zb) - 1, 0, len(zb) - 2)
        centers, m_rad, m_tan = [], [], []
        for b in range(len(zb) - 1):
            m = idx == b
            if m.sum() >= min_per_cell:
                centers.append(0.5 * (zb[b] + zb[b + 1]))
                m_rad.append(float(rad_comp[m].mean()))
                m_tan.append(float(tan_comp[m].mean()))
        ax_prof.plot(centers, m_rad, "-o", ms=4, label="mean radial")
        ax_prof.plot(centers, m_tan, "-s", ms=4, label="mean tangential")
    ax_prof.axhline(0.0, color="k", lw=0.5)
    ax_prof.set_title("binned mean radial/tangential vs zenith")
    ax_prof.set_xlabel("zenith angle (deg)")
    ax_prof.set_ylabel("mean component (pix)")
    ax_prof.legend()

    fig.suptitle("Alcor WCS residuals")
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


def _moon_altitude(time, location=MMT_LOCATION):
    """Return the Moon's altitude in degrees at ``time`` and ``location``.

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


def detect_alcor_stars(im, fwhm=3.0, threshold_sigma=5.0, max_detections=200):
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
    max_detections : int or None (default=200)
        Keep only the brightest ``max_detections`` sources by ``flux``. ``None``
        keeps all. Bounding the list to the brightest few hundred keeps matching
        on the well-detected stars regardless of per-frame noise/transparency.

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
    if max_detections is not None and len(out) > max_detections:
        order = np.argsort(np.asarray(out["flux"], dtype=float))[::-1]
        out = out[order[:max_detections]]
    return out


def build_alcor_badpix_mask(median_cube, ksize=5, z_thresh=25.0):
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
    return hot & keep[None, :, :]


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

        median = np.empty((nch, ny, nx), dtype=np.float32)
        for c in range(nch):
            for r0 in range(0, ny, tile):
                r1 = min(r0 + tile, ny)
                slab = np.asarray(cube_mm[:n, c, r0:r1, :], dtype=np.float32)
                median[c, r0:r1, :] = np.median(slab, axis=0)
        return median
    finally:
        del cube_mm
        memmap_path.unlink(missing_ok=True)


_BADPIX_DATE_RE = re.compile(r"alcor_badpix_(\d{4})-(\d{2})-(\d{2})\.fits(\.gz)?$")


def _resolve_badpix_dir(masks_dir=None):
    """Directory holding the date-stamped bad-pixel masks.

    Resolution order: explicit ``masks_dir`` -> ``$ALCOR_BADPIX_DIR`` -> the
    packaged ``skycam_utils/data/badpix/`` (mirroring :func:`load_wcs`).
    """
    if masks_dir is not None:
        return Path(masks_dir)
    env = os.environ.get("ALCOR_BADPIX_DIR")
    if env:
        return Path(env)
    return Path(str(files(__package__) / "data" / "badpix"))


def _badpix_date_from_dir(day_dir, dark_files):
    """Mask date: the ``YYYY-MM-DD`` in the day-directory name, else the median
    dark-frame time's date."""
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", Path(day_dir).name)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    dts = sorted(d for d in (_filename_ut_datetime(f) for f in dark_files)
                 if d is not None)
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
                    radial_coeffs=None, sip_degree=5,
                    badpix="repair", return_mask=False, masks_dir=None):
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
    badpix : str or None or path or ndarray (default="repair")
        Bad-pixel handling on the raw frame, before trim/resample. "repair"
        resolves the nearest-date epoch mask and replaces flagged pixels per
        channel with their local 5x5 median; None disables repair; a path or
        (3, ny, nx) bool array uses that mask explicitly (and repairs).
    return_mask : bool (default=False)
        When True, also return the bad-pixel mask aligned to the returned image
        frame, i.e. (im, mask, wcs). Pair with badpix=None to get untouched
        pixels plus the mask (e.g. to OR with a horizon mask for photometry).
    masks_dir : str or None (default=None)
        Override the bad-pixel masks directory (else $ALCOR_BADPIX_DIR, else the
        packaged data/badpix/).

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
        data = np.asarray(hdul[0].data)        # (3, ny, nx), raw sensor layout

    # --- bad-pixel handling on the raw frame, before any trim/resample ---
    raw_mask = None
    if return_mask or badpix is not None:
        if isinstance(badpix, np.ndarray):
            cand = badpix.astype(bool)
        elif isinstance(badpix, Path) or (isinstance(badpix, str) and badpix != "repair"):
            cand = np.asarray(fits.getdata(badpix)).astype(bool)
        else:                                   # "repair" or None -> resolve by time
            cand = None
            try:
                dt = _filename_ut_datetime(filename)
                t = (Time(dt) if dt is not None
                     else Time(_read_frame_date(filename), format="isot", scale="utc"))
                cand, _ = load_alcor_badpix_mask(t, masks_dir=masks_dir)
            except (KeyError, OSError, ValueError):
                cand = None
        if cand is not None and cand.shape == data.shape:
            raw_mask = cand

    if badpix is not None and raw_mask is not None:
        data = _apply_badpix_repair(data, raw_mask)

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

    if return_mask:
        if raw_mask is not None:
            mk = np.transpose(raw_mask, axes=(1, 2, 0)).astype(np.uint8)
            mk = mk[yl:yu, xl:xu, :]
            mk = np.flipud(rotate(mk, rotation, reshape=False, order=0))
            if xshift != 0.0 or yshift != 0.0:
                mk = ndimage_shift(mk, shift=(-yshift, -xshift, 0.0), order=0,
                                   mode="constant", cval=0)
            mask_out = mk.astype(bool)
        else:
            mask_out = np.zeros(im.shape, dtype=bool)
        return im, mask_out, wcs

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
    parser.add_argument("--vmag-limit", type=float, default=4.0, help="Faintest Vmag to use.")
    parser.add_argument("--sun-alt-max", type=float, default=-18.0,
                        help="Use frames with Sun altitude below this (deg).")
    parser.add_argument("--moon-alt-max", type=float, default=-6.0,
                        help="Use frames with Moon altitude below this (deg); "
                             "moonlight scatter corrupts source detection. "
                             "Pass 90 to disable the Moon cut.")
    parser.add_argument("--min-alt", type=float, default=10.0, help="Minimum star altitude (deg).")
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="Final (tightest) match tolerance in pixels; the matcher "
                             "tightens to this from --tolerance-start over --match-rounds rounds.")
    parser.add_argument("--tolerance-start", type=float, default=12.0,
                        help="Initial (loosest) match tolerance in pixels.")
    parser.add_argument("--match-rounds", type=int, default=4,
                        help="Number of tightening rounds from --tolerance-start to --tolerance.")
    parser.add_argument("--pattern-tol", type=float, default=3.0,
                        help="Asterism corroboration tolerance in pixels: how closely a "
                             "neighbor's offset must match the local constellation. Loosen "
                             "to keep matches where real distortion bends the local pattern.")
    parser.add_argument("--min-corroborating", type=int, default=2,
                        help="Minimum neighbors that must corroborate a match's local pattern.")
    parser.add_argument("--n-neighbors", type=int, default=5,
                        help="Nearest catalog neighbors checked in asterism verification.")
    parser.add_argument("--fit-k5", action="store_true",
                        help="Also fit the odd quintic radial term k5 (richer radial "
                             "distortion model; use only on a clean, full-zenith match set).")
    parser.add_argument("--max-detections", type=int, default=200,
                        help="Keep only the brightest N detections per frame.")
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
        sun_alt_max=args.sun_alt_max, moon_alt_max=args.moon_alt_max,
        min_alt=args.min_alt, tolerance=args.tolerance,
        tolerance_start=args.tolerance_start, match_rounds=args.match_rounds,
        n_neighbors=args.n_neighbors, min_corroborating=args.min_corroborating,
        pattern_tol=args.pattern_tol, fit_k5=args.fit_k5,
        max_detections=args.max_detections,
        max_frames=args.max_frames, workers=args.workers, log=log,
    )
    print(f"# matched stars: {result['n_matched']}")
    print(f"# residual RMS (pix): {result['residual_rms']:.3f}")
    print(f"# matched fraction: {result['matched_fraction']:.3f}")
    print("# add this entry to ALCOR_CALIBRATIONS in alcor.py:")
    print(_format_calibration_entry(result))
    if args.residual_plot is not None:
        out = save_alcor_residual_plot(result["alt"], result["az"], result["x"],
                                       result["y"], result, args.residual_plot)
        print(out)


def create_badpix_mask(day_dir, out_dir=None, min_frames=500, z_thresh=25.0,
                        ksize=5, sun_alt_max=-18.0, moon_alt_max=-6.0,
                        max_frames=None, scratch_dir=None, pattern="*.fits.bz2",
                        log=None):
    """
    Build and write a date-stamped per-channel bad-pixel mask for one night.

    Selects dark frames (Sun < ``sun_alt_max``, Moon < ``moon_alt_max``), and if
    at least ``min_frames`` are available builds the night-median stack, detects
    hot pixels, and writes a gzipped ``alcor_badpix_YYYY-MM-DD.fits.gz`` to
    ``out_dir`` (default: the resolved bad-pixel masks directory). Returns the
    output `~pathlib.Path`, or ``None`` if there were too few dark frames.
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
    mask = build_alcor_badpix_mask(median, ksize=ksize, z_thresh=z_thresh)
    mask_date = _badpix_date_from_dir(day_dir, dark)

    out_dir = Path(str(out_dir)) if out_dir is not None else _resolve_badpix_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alcor_badpix_{mask_date.isoformat()}.fits.gz"

    hdu = fits.PrimaryHDU(data=mask.astype(np.uint8))
    hdu.header["NSTACK"] = (len(dark), "dark frames used")
    hdu.header["ZTHRESH"] = (z_thresh, "robust-sigma threshold")
    hdu.header["KSIZE"] = (ksize, "high-pass kernel (px)")
    hdu.header["CHRULE"] = ("1-2 of 3", "channels flagged for a bad pixel")
    for c, name in enumerate("RGB"):
        hdu.header[f"NBAD{name}"] = (int(mask[c].sum()), f"{name} bad pixels")
    hdu.writeto(out_path, overwrite=True)
    if log:
        log(f"wrote {out_path}")
    return out_path


def create_badpix_mask_cli():
    """CLI entry point for :func:`create_badpix_mask` (run daily from cron)."""
    parser = argparse.ArgumentParser(
        description="Build a date-stamped alcor bad-pixel mask from a night of frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("day_dir", help="Directory of one night's alcor frames.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: $ALCOR_BADPIX_DIR or packaged data/badpix).")
    parser.add_argument("--min-frames", type=int, default=500,
                        help="Minimum dark frames required to generate a mask.")
    parser.add_argument("--z-thresh", type=float, default=25.0,
                        help="Robust-sigma threshold for a hot pixel.")
    parser.add_argument("--ksize", type=int, default=5, help="High-pass median kernel (px).")
    parser.add_argument("--sun-alt-max", type=float, default=-18.0,
                        help="Use frames with Sun altitude below this (deg).")
    parser.add_argument("--moon-alt-max", type=float, default=-6.0,
                        help="Use frames with Moon altitude below this (deg); pass 90 to disable.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap frames used (strided) to bound runtime/scratch.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Directory for the temporary memmap (default: system temp).")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob for input frames.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print per-step progress messages.")
    args = parser.parse_args()

    log = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    out = create_badpix_mask(
        args.day_dir, out_dir=args.out_dir, min_frames=args.min_frames,
        z_thresh=args.z_thresh, ksize=args.ksize, sun_alt_max=args.sun_alt_max,
        moon_alt_max=args.moon_alt_max, max_frames=args.max_frames,
        scratch_dir=args.scratch_dir, pattern=args.pattern, log=log)
    if out is None:
        print("# no mask written (insufficient dark frames)")
    else:
        print(out)
