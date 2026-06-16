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
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
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


# Nominal raw-frame geometry defaults. The real geometry is per-epoch (see
# ALCOR_CALIBRATIONS) and is threaded explicitly through every call path;
# ALCOR_HORIZON_RADIUS is only a fallback default for the rare arg-less call.
# It is the plate scale: the pixel radius from the optical-axis pixel at which
# the angular distance from the axis reaches 90 deg (altitude=0 when axis_tilt
# is zero; cdelt = 90/horizon_radius deg/px). ALCOR_RADIUS is the default
# display-crop half-width for plot_alcor_fits.
ALCOR_RADIUS = 680
ALCOR_HORIZON_RADIUS = 747
# Raw-ADU ceiling: the OMEA 8C delivers 15-bit data, so pixels peg at 2**15 - 1.
ALCOR_SATURATION = 32767
ALCOR_NONLINEAR_THRESHOLD = 15000   # raw ADU; per-pixel non-linearity onset

# Adopted photometric calibration (see ALCOR_ZEROPOINTS). A single achromatic
# extinction term applies to all three bands, and instrument magnitudes brighter
# than the bright cut are in the CMOS non-linear regime where the calibration is
# invalid. The airmass term was established in docs/scripts/zeropoint_calib.py.
# The bright cut is set by docs/scripts/nonlin_binned.py: with intra-pixel jitter
# averaged out (15-min per-star medians), the bright-star magnitude deficit onsets
# near -11.5 but stays small and ~linear out to -12.5, then accelerates steeply.
# -12.5 is the linear-regime cutoff; brighter than that is sparse/inconclusive and
# dropped pending more calibration nights.
ALCOR_AIRMASS_TERM = 0.40   # mag/airmass, single term for R/G/B
ALCOR_BRIGHT_CUT = -12.5    # instr mag; brighter is non-linear, calibration void

# minimum unmasked pixel counts for the Gaussian fits
_GAUSS_MIN_LUM_PIXELS = 8            # 4-parameter luminance shape fit
_GAUSS_MIN_CHANNEL_PIXELS = 3        # 1-parameter per-channel amplitude

# Time-indexed lens calibrations. Each epoch holds the raw-frame geometry as
# absolutes: the optical-axis pixel (xcen, ycen) in the raw FITS frame (the
# zenith pixel when axis_tilt is zero), the azimuth
# rotation, the radial_coeffs (k1, k3, k5), and the horizon_radius (pixels from
# zenith to alt=0). An optional "tangential_coeffs": (P1, P2) holds the
# Brown-Conrady decentering (sensor-tilt) terms, dimensionless like the k's;
# epochs without the key mean (0.0, 0.0) (alcor_calibration fills the default).
# An optional "axis_tilt": (t_n, t_e) holds the optical-axis tilt from the
# zenith as components toward north and east, in DEGREES (the axis points at
# alt 90 - hypot(t_n, t_e), az atan2(t_e, t_n)); epochs without the key mean
# (0.0, 0.0). With nonzero tilt, xcen/ycen is the optical-axis pixel (the
# distortion center), not the zenith pixel.
# The camera geometry drifts over time (mount/focus), so the
# epoch nearest in time to an image is used (see alcor_calibration). Add a new
# epoch by pasting the dict that fit_alcor_wcs prints. `epoch` is the calibration
# night at day precision (UT, not local night -- do not "fix" it to local).
ALCOR_CALIBRATIONS = [
    {"epoch": "2024-09-05", "xcen": 703.586, "ycen": 704.803, "rotation": -0.9642,
     "radial_coeffs": (1.0, 0.047841687068536774, 0.1163038015749883),
     "tangential_coeffs": (-0.0003040188173761858, 0.0006700812069651288),
     "axis_tilt": (-0.8225394950126477, -0.6160139387466032),
     "horizon_radius": 747.2},
    # The camera was not moved or changed between 2024 and 2026; this epoch is
    # consistent with 2024 within the fit uncertainty (center stable ~1px, axis
    # tilt agrees to ~0.03 deg / ~1.5 deg in lean azimuth, ~0.05 deg rotation
    # drift). It is kept as a separate entry so per-era geometry is supported
    # if the camera is ever moved/refocused.
    {"epoch": "2026-05-19", "xcen": 703.537, "ycen": 703.832, "rotation": -1.0177,
     "radial_coeffs": (1.0, 0.05337073600079686, 0.1111394504753296),
     "tangential_coeffs": (-0.0005518035497827486, 0.0006929046299086498),
     "axis_tilt": (-0.860477755807792, -0.6088848881222786),
     "horizon_radius": 747.2},
]


def _calibration_epochs():
    """
    Return [(Time, calibration_dict), ...] for the configured epochs.
    """
    return [(Time(c["epoch"], scale="utc"), c) for c in ALCOR_CALIBRATIONS]


def alcor_calibration(time=None):
    """
    Return the calibration dict whose epoch is nearest in time to ``time``.

    ``time`` is an astropy ``Time``. An exact tie resolves to the more recent
    epoch. ``time=None`` returns the most recent epoch (the default for
    time-agnostic calls). The returned dict is a copy and may be mutated freely;
    ``tangential_coeffs`` and ``axis_tilt`` are filled with ``(0.0, 0.0)`` for
    epochs that omit them.
    """
    epochs = _calibration_epochs()
    if time is None:
        cal = dict(max(epochs, key=lambda e: e[0].jd)[1])
    else:
        jds = np.array([e[0].jd for e in epochs])
        dt = np.abs(jds - Time(time).jd)
        # primary: smallest |dt|; tie-break: largest jd (more recent)
        order = np.lexsort((-jds, dt))
        cal = dict(epochs[order[0]][1])
    cal.setdefault("tangential_coeffs", (0.0, 0.0))
    cal.setdefault("axis_tilt", (0.0, 0.0))
    return cal


# Module-level defaults track the most-recent epoch so existing default-argument
# references (in _predict_pixels, build_alcor_wcs, etc.) keep working unchanged.
_LATEST_CALIBRATION = alcor_calibration()
ALCOR_ROTATION = _LATEST_CALIBRATION["rotation"]
ALCOR_XCEN = _LATEST_CALIBRATION["xcen"]
ALCOR_YCEN = _LATEST_CALIBRATION["ycen"]
ALCOR_RADIAL_COEFFS = _LATEST_CALIBRATION["radial_coeffs"]
ALCOR_TANGENTIAL_COEFFS = _LATEST_CALIBRATION["tangential_coeffs"]
ALCOR_AXIS_TILT = _LATEST_CALIBRATION["axis_tilt"]


# Time-indexed photometric zeropoints calibrated on clear dark nights. They map
# instrument R/G/B aperture magnitudes to catalog Johnson R/V/B via
#   cat_mag = (instr_mag - ALCOR_AIRMASS_TERM*airmass) + zp + color_coeff*(B-V)
# with the channel->catalog assignment G->V, R->R, B->B (see
# ALCOR_ZEROPOINT_BANDS). G->V is essentially color-flat; R and B carry sizeable
# B-V color terms set by the instrument bandpasses. The zeropoints were fit with
# ALCOR_AIRMASS_TERM held fixed, so the term and the zeropoints are a matched
# set. They are stable to ~0.03 mag across the two epochs (~21 months); add a new
# epoch like ALCOR_CALIBRATIONS and the nearest epoch in time is used (see
# alcor_zeropoint). Derived by docs/scripts/zeropoint_calib.py. `epoch` is the
# calibration night (UT, not local night -- do not "fix" it).
ALCOR_ZEROPOINTS = [
    {"epoch": "2024-09-05",
     "r": {"zp": 14.670, "color_coeff": -0.323},
     "g": {"zp": 15.438, "color_coeff": -0.023},
     "b": {"zp": 14.988, "color_coeff": 0.479}},
    {"epoch": "2026-05-19",
     "r": {"zp": 14.639, "color_coeff": -0.343},
     "g": {"zp": 15.423, "color_coeff": -0.038},
     "b": {"zp": 15.015, "color_coeff": 0.470}},
]
# instrument channel -> catalog Johnson band measured against
ALCOR_ZEROPOINT_BANDS = {"r": "R", "g": "V", "b": "B"}


def alcor_zeropoint(time=None):
    """
    Return the photometric-zeropoint dict whose epoch is nearest ``time``.

    Mirrors :func:`alcor_calibration`: ``time`` is an astropy ``Time`` (an exact
    tie resolves to the more recent epoch), and ``time=None`` returns the most
    recent epoch. The returned dict is a deep-enough copy that its per-band
    sub-dicts may be mutated freely without corrupting the table.
    """
    epochs = [(Time(z["epoch"], scale="utc"), z) for z in ALCOR_ZEROPOINTS]
    if time is None:
        chosen = max(epochs, key=lambda e: e[0].jd)[1]
    else:
        jds = np.array([e[0].jd for e in epochs])
        dt = np.abs(jds - Time(time).jd)
        # primary: smallest |dt|; tie-break: largest jd (more recent)
        order = np.lexsort((-jds, dt))
        chosen = epochs[order[0]][1]
    return {key: (dict(value) if isinstance(value, dict) else value)
            for key, value in chosen.items()}


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


def _axis_frame(alt, az, t_n, t_e):
    """
    Axis-centered polar coordinates (z', A') [deg] of sky points (alt, az)
    [deg] for an optical axis tilted off the zenith.

    The axis leans eps = hypot(t_n, t_e) deg from the zenith toward azimuth
    A0 = atan2(t_e, t_n) (components toward north and east). The sky is
    rotated by the exact minimal rotation -- by eps about the horizontal axis
    at azimuth A0 + 90 -- that carries the optical axis to the pole, so
    A' -> az continuously as eps -> 0. z' is the true angular distance from
    the axis.
    """
    eps = np.radians(np.hypot(t_n, t_e))
    a0 = np.arctan2(t_e, t_n)
    alt_r = np.radians(np.asarray(alt, dtype=float))
    az_r = np.radians(np.asarray(az, dtype=float))
    # unit vectors: x toward north (az=0), y toward east (az=90), z up
    vx = np.cos(alt_r) * np.cos(az_r)
    vy = np.cos(alt_r) * np.sin(az_r)
    vz = np.sin(alt_r)
    # Rodrigues rotation by -eps about n = (-sin A0, cos A0, 0):
    # v' = v cos(eps) - (n x v) sin(eps) + n (n.v)(1 - cos(eps))
    nx, ny = -np.sin(a0), np.cos(a0)
    ndv = nx * vx + ny * vy
    c, s = np.cos(eps), np.sin(eps)
    wx = c * vx - s * (ny * vz) + (1.0 - c) * ndv * nx
    wy = c * vy - s * (-nx * vz) + (1.0 - c) * ndv * ny
    wz = c * vz - s * (nx * vy - ny * vx)
    zp = np.degrees(np.arccos(np.clip(wz, -1.0, 1.0)))
    ap = np.degrees(np.arctan2(wy, wx))
    return zp, ap


def _tangential_delta(u, v, p1, p2, horizon_radius):
    """
    Brown-Conrady tangential (decentering) displacement in raw pixels.

    ``u``, ``v`` are pixel offsets from the optical-axis pixel
    ``(xcen, ycen)``; ``p1``/``p2`` are
    dimensionless (normalized by ``horizon_radius``), like the radial k
    coefficients. This is the pix->world displacement the WCS SIP applies
    (see build_alcor_wcs); it is an exact degree-2 polynomial.
    """
    H = float(horizon_radius)
    du = (p1 / H) * (3.0 * u**2 + v**2) + (2.0 * p2 / H) * u * v
    dv = (p2 / H) * (u**2 + 3.0 * v**2) + (2.0 * p1 / H) * u * v
    return du, dv


def _predict_pixels(
    alt,
    az,
    xcen=ALCOR_XCEN,
    ycen=ALCOR_YCEN,
    rotation=0.0,
    radial_coeffs=ALCOR_RADIAL_COEFFS,
    horizon_radius=ALCOR_HORIZON_RADIUS,
    tangential_coeffs=(0.0, 0.0),
    axis_tilt=(0.0, 0.0),
):
    """
    Forward lens model: map altitude/azimuth (deg) to RAW-frame pixel
    coordinates (x=column, y=row, 0-based).

    The optical axis sits at ``(xcen, ycen)`` (the zenith pixel when
    ``axis_tilt`` is zero); ``rotation`` is the camera azimuth
    zero-point offset (deg). The lens plate solution
    ``z = 90*(k1*rho + k3*rho**3 + k5*rho**5)`` (``rho = r/horizon_radius``,
    ``z = 90 - alt``) is inverted for ``rho`` via Newton's method. The sky's
    azimuth runs opposite to the sensor's polar angle (an all-sky camera images
    the sky as seen from below), so the pixel angle is ``rotation - az``; north
    (az=0) lands toward +y. The matching WCS encodes the same mapping in its PC
    rotation matrix (see ``build_alcor_wcs``).

    ``tangential_coeffs`` (P1, P2) adds Brown-Conrady decentering, defined like
    the k's on the pix->world side (`_tangential_delta`). It is inverted with a
    fixed-point loop that re-solves the radial part exactly each pass, so the
    contraction is governed by the (tiny, ~4*P) tangential derivative rather
    than the O(k3, k5) radial one; three passes reach well below 1e-3 px for
    |P| up to ~1e-2.

    ``axis_tilt`` (t_n, t_e) tilts the optical axis off the zenith (degrees
    toward north/east; see `_axis_frame`). The model is azimuthally symmetric
    about the AXIS: the radial inversion runs in the axis distance z' and the
    pixel angle is ``rotation - A'``. With nonzero tilt, (xcen, ycen) is the
    optical-axis pixel, not the zenith pixel.
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    coeffs = tuple(float(c) for c in radial_coeffs)

    tn, te = (float(c) for c in axis_tilt)
    if tn != 0.0 or te != 0.0:
        zp, ap = _axis_frame(alt, az, tn, te)
    else:
        zp = 90.0 - alt
        ap = az

    rho = _invert_radial(zp, coeffs)
    r = horizon_radius * rho
    ang = np.radians(rotation - ap)
    u = r * np.sin(ang)
    v = r * np.cos(ang)

    p1, p2 = (float(c) for c in tangential_coeffs)
    if p1 != 0.0 or p2 != 0.0:
        k1 = coeffs[0]
        H = float(horizon_radius)
        # Linear-pixel target of the SIP equation t = (u,v) + D_rad + D_tan:
        # the radial displacement preserves direction, so |t| = H*z'/(90*k1).
        s = H * zp / (90.0 * k1)
        tu = s * np.sin(ang)
        tv = s * np.cos(ang)
        for _ in range(3):
            du, dv = _tangential_delta(u, v, p1, p2, H)
            wu = tu - du
            wv = tv - dv
            wr = np.hypot(wu, wv)
            safe = np.where(wr > 0.0, wr, 1.0)
            rho_w = _invert_radial(90.0 * k1 * wr / H, coeffs)
            scale = np.where(wr > 0.0, H * rho_w / safe, 0.0)
            u = wu * scale
            v = wv * scale

    x = xcen + u
    y = ycen + v
    return x, y


def _fit_params(alt, az, obs_x, obs_y, init_params,
                horizon_radius=ALCOR_HORIZON_RADIUS, fit_k5=False):
    """
    Robust least-squares fit of the lens geometry to matched stars.

    By default fits (xcen, ycen, rotation, k3, P1, P2) with k1 held at 1.0 (the
    zenith plate scale is set by horizon_radius) and k5 at 0.0. k3 and k5 are
    nearly collinear in rho over [0, 1], so fitting both is ill-conditioned on
    *dirty* data and runs away to large cancelling values (e.g. k3=-0.58,
    k5=3.6) that are unphysical despite a tolerable RMS -- which is why k3
    alone is the default (the model the shipped 2024 constants used). With
    ``fit_k5=True`` the odd quintic term k5 is fit as well: appropriate only on
    a clean, well-distributed match set (asterism-verified, spanning the full
    zenith range), where the radial residual that k3 alone leaves near the
    horizon can be captured.

    The Brown-Conrady tangential terms (P1, P2) are always fit: unlike k3/k5
    they are well-conditioned against the other parameters (their displacement
    grows as r**2 and varies once per azimuth revolution, while a center shift
    is constant and rotation grows as r), and they capture the sensor-tilt /
    decentering signature the azimuthally-symmetric radial basis cannot.

    The axis-tilt components (t_n, t_e) are likewise always fit: their
    tangential signature falls off as 1/tan(z), which no other parameter can
    produce (translation is constant, rotation grows as r, Brown-Conrady as
    r**2), so the term is well-conditioned. The returned dict also carries
    axis_tilt=(t_n, t_e).

    A ``soft_l1`` loss downweights mismatched/noise detections, common in this
    sparse bright-star field. Returns an updated params dict with
    radial_coeffs=(1.0, k3, k5) (k5=0.0 unless ``fit_k5``) and
    tangential_coeffs=(P1, P2).
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    obs_x = np.asarray(obs_x, dtype=float)
    obs_y = np.asarray(obs_y, dtype=float)
    init_k3 = init_params["radial_coeffs"][1]
    init_k5 = init_params["radial_coeffs"][2]
    init_p1, init_p2 = init_params.get("tangential_coeffs", (0.0, 0.0))
    init_tn, init_te = init_params.get("axis_tilt", (0.0, 0.0))

    p0 = [init_params["xcen"], init_params["ycen"],
          init_params["rotation"], init_k3]
    if fit_k5:
        p0.append(init_k5)
    p0 += [init_p1, init_p2, init_tn, init_te]
    p0 = np.asarray(p0, dtype=float)

    def unpack(p):
        if fit_k5:
            xcen, ycen, rot, k3, k5, p1, p2, tn, te = p
        else:
            xcen, ycen, rot, k3, p1, p2, tn, te = p
            k5 = 0.0
        return xcen, ycen, rot, k3, k5, p1, p2, tn, te

    def residuals(p):
        xcen, ycen, rot, k3, k5, p1, p2, tn, te = unpack(p)
        x, y = _predict_pixels(alt, az, xcen=xcen, ycen=ycen, rotation=rot,
                               radial_coeffs=(1.0, k3, k5),
                               horizon_radius=horizon_radius,
                               tangential_coeffs=(p1, p2),
                               axis_tilt=(tn, te))
        return np.concatenate([x - obs_x, y - obs_y])

    result = least_squares(residuals, p0, loss="soft_l1", f_scale=3.0)
    xcen, ycen, rot, k3, k5, p1, p2, tn, te = unpack(result.x)
    return dict(xcen=float(xcen), ycen=float(ycen), rotation=float(rot),
                radial_coeffs=(1.0, float(k3), float(k5)),
                tangential_coeffs=(float(p1), float(p2)),
                axis_tilt=(float(tn), float(te)),
                horizon_radius=float(horizon_radius))


def assign_alcor_matches(cat, det, params, tolerance,
                         horizon_radius=ALCOR_HORIZON_RADIUS,
                         n_neighbors=5, min_corroborating=2, pattern_tol=3.0,
                         brightness=True):
    """
    Assign catalog stars to detected sources against a *fixed* geometry.

    This never refits the geometry internally; ``params`` (``xcen``, ``ycen``,
    ``rotation``, ``radial_coeffs``, and optional ``tangential_coeffs``,
    ``axis_tilt``, ``horizon_radius``) is the fixed geometry used for the
    whole frame. The steps are:

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
        cat["Alt"], cat["Az"], xcen=params["xcen"], ycen=params["ycen"],
        rotation=params["rotation"], radial_coeffs=tuple(params["radial_coeffs"]),
        horizon_radius=params.get("horizon_radius", horizon_radius),
        tangential_coeffs=tuple(params.get("tangential_coeffs", (0.0, 0.0))),
        axis_tilt=tuple(params.get("axis_tilt", (0.0, 0.0))),
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
    """
    Return the observation Time (UT) from a FITS file's DATE (creation) header.

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
    cube, _, _ = load_alcor_fits(filename, badpix="repair")  # repair hot pixels so they aren't detected as stars
    cat = alcor_reference_altaz(time, vmag_limit=vmag_limit, min_alt=min_alt)
    det = detect_alcor_stars(cube, fwhm=fwhm, threshold_sigma=threshold_sigma,
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

    The fit runs directly on the raw frame, so the recovered (xcen, ycen,
    rotation, radial_coeffs, tangential_coeffs, axis_tilt) are the ABSOLUTE
    raw-frame geometry constants for the
    night, suitable for baking into ``ALCOR_CALIBRATIONS`` (the night's
    ``horizon_radius`` is carried through from the seed epoch). It is warm-started
    from the nearest existing epoch (via :func:`alcor_calibration` at the night's
    median time).

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

    init = dict(xcen=base["xcen"], ycen=base["ycen"],
                rotation=base["rotation"], radial_coeffs=base["radial_coeffs"],
                tangential_coeffs=base.get("tangential_coeffs", (0.0, 0.0)),
                axis_tilt=base.get("axis_tilt", (0.0, 0.0)),
                horizon_radius=base["horizon_radius"])
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
            params = _fit_params(alt, az, x, y, init_params=params,
                                 horizon_radius=base["horizon_radius"], fit_k5=fit_k5)

    # Final pool at the tightest tolerance, then 3*MAD outlier rejection + refit.
    pooled = pool(params, float(tolerance))
    if pooled is None:
        raise RuntimeError("No matched stars across the selected frames.")
    alt, az, x, y = pooled

    px, py = _predict_pixels(alt, az, xcen=params["xcen"], ycen=params["ycen"],
                             rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]),
                             horizon_radius=base["horizon_radius"],
                             tangential_coeffs=tuple(params["tangential_coeffs"]),
                             axis_tilt=tuple(params["axis_tilt"]))
    resid = np.hypot(px - x, py - y)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    good = resid < np.median(resid) + 3.0 * 1.4826 * mad
    if good.sum() >= 3:
        params = _fit_params(alt[good], az[good], x[good], y[good], init_params=params,
                             horizon_radius=base["horizon_radius"], fit_k5=fit_k5)
    px, py = _predict_pixels(alt[good], az[good], xcen=params["xcen"],
                             ycen=params["ycen"], rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]),
                             horizon_radius=base["horizon_radius"],
                             tangential_coeffs=tuple(params["tangential_coeffs"]),
                             axis_tilt=tuple(params["axis_tilt"]))
    rms = float(np.sqrt(np.mean((px - x[good]) ** 2 + (py - y[good]) ** 2)))

    return {
        **params,
        "horizon_radius": base["horizon_radius"],
        "epoch": epoch,
        "n_matched": int(good.sum()),
        "residual_rms": rms,
        "matched_fraction": float(int(good.sum()) / available) if available else 0.0,
        "alt": alt[good], "az": az[good], "x": x[good], "y": y[good],
    }


def save_alcor_residual_plot(alt, az, obs_x, obs_y, params, output_file,
                             horizon_radius=ALCOR_HORIZON_RADIUS,
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
    direction from the optical-axis pixel ``(xcen, ycen)`` to the star) and a
    **tangential** component (perpendicular), which
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

    hr = params.get("horizon_radius", horizon_radius)
    cenx = float(params["xcen"])
    ceny = float(params["ycen"])
    # "before" baseline: same center and rotation, but the idealized equidistant
    # radial mapping -- so the panel isolates what the fitted radial term removes.
    ix, iy = _predict_pixels(alt, az, xcen=cenx, ycen=ceny,
                             rotation=params["rotation"],
                             radial_coeffs=(1.0, 0.0, 0.0), horizon_radius=hr)
    fx, fy = _predict_pixels(alt, az, xcen=cenx, ycen=ceny,
                             rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]),
                             horizon_radius=hr,
                             tangential_coeffs=tuple(
                                 params.get("tangential_coeffs", (0.0, 0.0))),
                             axis_tilt=tuple(
                                 params.get("axis_tilt", (0.0, 0.0))))
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

    # Bin the residual vectors onto a grid and average per cell, so coherent
    # structure survives while incoherent (mismatch) scatter cancels out. The raw
    # frame is not centered on the optical axis, so the grid spans the detector
    # extent implied by the (xcen, ycen) center plus the horizon radius.
    extent = max(cenx, ceny) + hr
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
    ax_v.plot(cenx, ceny, "r+", ms=14, label="optical axis")
    ax_v.set_aspect("equal")
    ax_v.set_xlim(0, extent)
    ax_v.set_ylim(0, extent)
    ax_v.set_title(f"binned mean residual vectors ({nbins}x{nbins}, x{amp:.0f})")
    ax_v.set_xlabel("x (pix)")
    ax_v.set_ylabel("y (pix)")
    ax_v.legend(loc="upper right")
    fig.colorbar(q, ax=ax_v, label="mean |residual| (pix)")

    # Radial/tangential decomposition about the optical-axis pixel.
    vx = obs_x - cenx
    vy = obs_y - ceny
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


def _base_arc_wcs(xcen, ycen, rotation, k1, horizon_radius,
                  axis_tilt=(0.0, 0.0)):
    """
    Linear ARC WCS (no SIP) reproducing the raw forward model's linear part.

    crpix is the 1-based optical-axis pixel; the PC matrix is the pure rotation
    (det=+1) that matches ``_predict_pixels`` (the sky/sensor handedness lives in
    the ``rotation - az`` azimuth convention, encoded by the ARC longitude axis).
    A nonzero ``axis_tilt`` moves the projection pole to the tilted optical
    axis: CRVAL = (A0, 90 - eps) and LONPOLE = A0, which makes the WCS native
    frame coincide with the minimal-rotation frame of ``_axis_frame`` (native
    longitude phi = A' + 180; the celestial pole sits at A' = A0 + 180).
    """
    cdelt = 90.0 * k1 / horizon_radius
    rot = np.radians(rotation)
    c, s = np.cos(rot), np.sin(rot)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
    wcs.wcs.crpix = [xcen + 1.0, ycen + 1.0]
    wcs.wcs.cdelt = [cdelt, cdelt]
    wcs.wcs.pc = [[c, -s], [s, c]]
    tn, te = axis_tilt
    if tn != 0.0 or te != 0.0:
        eps = float(np.hypot(tn, te))
        a0 = float(np.degrees(np.arctan2(te, tn))) % 360.0
        wcs.wcs.crval = [a0, 90.0 - eps]
        wcs.wcs.lonpole = a0
    else:
        wcs.wcs.crval = [0.0, 90.0]
        wcs.wcs.lonpole = 0.0
    return wcs


def _sip_poly_eval(coef, u, v):
    """
    Evaluate a SIP coefficient matrix (coef[p, q] * u**p * v**q) at (u, v).
    """
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


def build_alcor_wcs(xcen=ALCOR_XCEN, ycen=ALCOR_YCEN, rotation=ALCOR_ROTATION,
                    radial_coeffs=ALCOR_RADIAL_COEFFS,
                    horizon_radius=ALCOR_HORIZON_RADIUS, sip_degree=5,
                    tangential_coeffs=ALCOR_TANGENTIAL_COEFFS,
                    axis_tilt=ALCOR_AXIS_TILT):
    """
    Build the raw-frame alt/az ARC WCS for the alcor sensor.

    The optical axis is at pixel ``(xcen, ycen)`` (the zenith pixel when
    ``axis_tilt`` is zero); ``rotation`` is the PC rotation matrix (the
    sky/sensor handedness is in the ``rotation - az`` convention, see
    ``_predict_pixels``); the radial ``k3``/``k5`` distortion is an exact
    analytic SIP centered on ``(xcen, ycen)``. Cached on its hashable args;
    returns a fresh copy.

    The lens is parametrized as a plate solution that maps the detector directly
    to the sky: ``z = 90*(k1*rho + k3*rho**3 + k5*rho**5)`` with ``rho =
    r / horizon_radius`` the normalized detector radius and ``z = 90 - alt`` the
    zenith angle (an odd-power, symmetric-fisheye polynomial in detector radius;
    the ``k5`` term needs degree 5, hence ``sip_degree=5``). The Cartesian
    displacement of this radial map is an exact degree-5 polynomial in the
    detector pixel offsets, so the SIP coefficients are constructed analytically
    (not fitted) and reproduce the plate solution to numerical precision over the
    whole FOV. The radial polynomial is rotation/reflection invariant, so the
    same A/B coefficients hold in the raw frame -- only the SIP reference pixel
    moves to the optical-axis pixel.

    ``tangential_coeffs`` (P1, P2) adds Brown-Conrady decentering; its Cartesian
    displacement is an exact degree-2 polynomial (see ``_tangential_delta``), so
    it joins the analytic SIP without approximation.

    ``axis_tilt`` (t_n, t_e) tilts the optical axis off the zenith. This is
    pure FITS-WCS geometry -- CRVAL moves to (A0, 90 - eps) with LONPOLE = A0
    -- so the SIP is untouched and the mapping stays exact; with nonzero tilt
    (xcen, ycen) is the optical-axis pixel, and the zenith pixel must be
    obtained via ``world_to_pixel`` of alt=90 rather than CRPIX.
    """
    return _build_alcor_wcs_cached(
        float(xcen), float(ycen), float(rotation),
        tuple(float(c) for c in radial_coeffs),
        float(horizon_radius), int(sip_degree),
        tuple(float(c) for c in tangential_coeffs),
        tuple(float(c) for c in axis_tilt),
    ).deepcopy()


@lru_cache(maxsize=32)
def _build_alcor_wcs_cached(xcen, ycen, rotation, radial_coeffs, horizon_radius,
                            sip_degree, tangential_coeffs=(0.0, 0.0),
                            axis_tilt=(0.0, 0.0)):
    k1, k3, k5 = radial_coeffs
    p1, p2 = tangential_coeffs
    base = _base_arc_wcs(xcen, ycen, rotation, k1, horizon_radius,
                         axis_tilt=axis_tilt)
    if (abs(k3) < 1e-12 and abs(k5) < 1e-12
            and abs(p1) < 1e-12 and abs(p2) < 1e-12):
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
    # Brown-Conrady tangential (decentering) terms: an exact degree-2 polynomial
    # in the pixel offsets (see _tangential_delta). The radial terms occupy only
    # odd-total-degree slots, the tangential only even ones -- no collisions.
    a[2, 0] = 3.0 * p1 / H; a[0, 2] = p1 / H; a[1, 1] = 2.0 * p2 / H
    b[0, 2] = 3.0 * p2 / H; b[2, 0] = p2 / H; b[1, 1] = 2.0 * p1 / H
    ap, bp = _fit_sip_inverse(a, b, int(round(H)), sip_degree)

    wcs = base.deepcopy()
    wcs.wcs.ctype = ["RA---ARC-SIP", "DEC--ARC-SIP"]
    wcs.sip = Sip(a, b, ap, bp, [xcen + 1.0, ycen + 1.0])
    return wcs


def detect_alcor_stars(im, fwhm=3.0, threshold_sigma=5.0, max_detections=200):
    """
    Detect point sources in an alcor frame.

    A ``(3, ny, nx)`` RGB cube is averaged over its channels into a luminance
    frame; a 2D frame is used as-is. The background level is estimated with a
    sigma-clipped median, and `~photutils.detection.DAOStarFinder` extracts
    sources above ``threshold_sigma`` times the background noise.

    Parameters
    ----------
    im : ndarray
        A 2D frame ``(ny, nx)`` or a raw ``(3, ny, nx)`` RGB cube, as returned by
        ``load_alcor_fits``.
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
    arr = np.asarray(im, dtype=float)
    if arr.ndim == 3:
        lum = arr.mean(axis=0)            # (3, ny, nx) R,G,B -> luminance
    elif arr.ndim == 2:
        lum = arr
    else:
        raise ValueError(f"expected a 2D frame or (3, ny, nx) cube, got {arr.shape}")
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
    return Path(str(files(__package__) / "data" / "badpix"))


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


def alcor_named_reference_altaz(time, vmag_limit=5.5, min_alt=20.0,
                                refraction=True, location=MMT_LOCATION):
    """
    Load ``bright_star_sloan_named.fits`` and compute Alt/Az at ``time``.

    This is the star-photometry catalog path: unlike
    :func:`alcor_reference_altaz`, it keeps the ``NAME`` column used as the CSV
    row index. Stars are filtered to ``Vmag <= vmag_limit`` and ``Alt >=
    min_alt``.
    """
    catpath = files(__package__) / "data" / "bright_star_sloan_named.fits"
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


def _catalog_value_to_python(value):
    """
    Convert an Astropy table scalar to a JSON-friendly Python scalar.
    """
    if np.ma.is_masked(value):
        return None
    if isinstance(value, bytes):
        return value.decode().strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.generic):
        return value.item()
    return value


def lookup_sloan_photometry(star_name, case_sensitive=False):
    """
    Return the ``bright_star_sloan_named.fits`` row for ``star_name`` as a dict.

    Matching is against the catalog ``NAME`` column, with surrounding whitespace
    ignored. By default matching is case-insensitive. A missing name raises
    ``KeyError``; an ambiguous name raises ``ValueError`` with the matching HD
    numbers.
    """
    query = str(star_name).strip()
    if not query:
        raise ValueError("star_name must not be empty")

    catpath = files(__package__) / "data" / "bright_star_sloan_named.fits"
    cat = Table.read(str(catpath))
    names = np.array([str(name).strip() for name in cat["NAME"]])
    if case_sensitive:
        match = names == query
    else:
        match = np.char.lower(names) == query.lower()

    matches = cat[match]
    if len(matches) == 0:
        raise KeyError(f"star {star_name!r} not found in bright_star_sloan_named.fits")
    if len(matches) > 1:
        hds = ", ".join(str(_catalog_value_to_python(row["HD"])) for row in matches)
        raise ValueError(f"star name {star_name!r} is ambiguous; matching HD numbers: {hds}")

    row = matches[0]
    return {col: _catalog_value_to_python(row[col]) for col in matches.colnames}


def _alcor_star_labels(cat):
    """
    Return stable, unique row labels for the named bright-star catalog.
    """
    raw_names = [str(name).strip() for name in cat["NAME"]]
    hd = []
    for value in cat["HD"]:
        try:
            hd.append(None if np.ma.is_masked(value) else int(value))
        except (TypeError, ValueError):
            hd.append(None)
    base = []
    for index, (name, hd_value) in enumerate(zip(raw_names, hd), start=1):
        if name and name != "--":
            base.append(name)
        elif hd_value is not None:
            base.append(f"HD {hd_value}")
        else:
            base.append(f"unnamed {index}")
    totals = {}
    for label in base:
        totals[label] = totals.get(label, 0) + 1
    counts = {}
    labels = []
    for label, hd_value in zip(base, hd):
        counts[label] = counts.get(label, 0) + 1
        if counts[label] == 1 and totals[label] == 1:
            labels.append(label)
        elif hd_value is not None:
            labels.append(f"{label} (HD {hd_value})")
        else:
            labels.append(f"{label} {counts[label]}")
    return labels


def _corner_bias(cube, size=10):
    """
    Per-channel median bias from square corner regions.
    """
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3 or cube.shape[0] != 3:
        raise ValueError(f"expected a (3, ny, nx) cube, got {cube.shape}")
    _, ny, nx = cube.shape
    if ny < size or nx < size:
        raise ValueError(f"image is smaller than the {size}x{size} bias regions")
    corners = [
        cube[:, :size, :size],
        cube[:, :size, -size:],
        cube[:, -size:, :size],
        cube[:, -size:, -size:],
    ]
    pixels = np.concatenate([corner.reshape(3, -1) for corner in corners], axis=1)
    return np.median(pixels, axis=1)


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


def _aperture_saturated(image, xcen, ycen, aperture_radius, saturation):
    """
    Return True if any pixel within the circular aperture reaches ``saturation``.

    Operates on the raw image (the saturation ceiling is a raw-ADU value), so
    callers pass the unmodified cube channel rather than the bias-subtracted
    data.
    """
    ny, nx = image.shape
    x0 = max(0, int(np.floor(xcen - aperture_radius)))
    x1 = min(nx, int(np.ceil(xcen + aperture_radius)) + 1)
    y0 = max(0, int(np.floor(ycen - aperture_radius)))
    y1 = min(ny, int(np.ceil(ycen + aperture_radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    aperture = np.hypot(xx - xcen, yy - ycen) <= aperture_radius
    if not aperture.any():
        return False
    return bool(np.any(image[y0:y1, x0:x1][aperture] >= saturation))


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


def _default_alcor_photometry_output(filename):
    stem = str(filename)
    for ext in (".fits.bz2", ".fits.gz", ".fits"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return Path(stem + "_phot.csv")


def _default_alcor_photometry_check_plot_output(filename):
    stem = str(filename)
    for ext in (".fits.bz2", ".fits.gz", ".fits"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return Path(stem + "_phot.pdf")


def _alcor_display_rgb(cube, powerstretch=0.75, contrast=0.35,
                       gscale=0.7, bscale=1.7):
    """
    Build a stretched display RGB image from a raw Alcor cube.

    This is display-only preprocessing: the raw cube returned by
    :func:`load_alcor_fits` remains untouched, but visualization subtracts the
    per-channel corner bias before color scaling so the bias pedestal does not
    dominate the color balance.
    """
    data = np.asarray(cube, dtype=float) - _corner_bias(cube)[:, None, None]
    rgb = np.transpose(data, (1, 2, 0))                  # (ny, nx, 3)
    rgb[:, :, 1] *= gscale
    rgb[:, :, 2] *= bscale
    stretch = viz.PowerStretch(powerstretch) + viz.ZScaleInterval(contrast=contrast)
    return stretch(rgb)


def _flux_mag(flux):
    """Map a measured flux to ``(flux, mag)`` with the non-detection convention.

    A finite, positive flux yields ``(float(flux), -2.5*log10(flux))``. Any
    non-finite or non-positive flux is treated as a non-detection and recorded
    as ``(0.0, nan)`` so the star still appears in the output: a catalog star
    that is above the horizon but invisible (e.g. behind cloud) is itself a
    strong extinction signal, so it must not be silently dropped.
    """
    if np.isfinite(flux) and flux > 0.0:
        return float(flux), float(-2.5 * np.log10(flux))
    return 0.0, np.nan


def _aperture_measure(data, cube, x, y, aperture_radius, annulus_width,
                      saturation, channels):
    """Per-channel aperture flux/background/saturation at ``(x, y)``.

    Returns a dict with ``flux_<ch>``, ``background_<ch>`` and ``sat_<ch>`` for
    each channel (raw measurements; magnitudes and validity are the caller's
    decision).
    """
    cols = {}
    for index, channel in enumerate(channels):
        flux, background = _aperture_annulus_photometry(
            data[index], x, y, aperture_radius, annulus_width)
        cols[f"flux_{channel}"] = flux
        cols[f"background_{channel}"] = background
        cols[f"sat_{channel}"] = _aperture_saturated(
            cube[index], x, y, aperture_radius, saturation)
    return cols


def _gaussian_measure(data, cube, lum_frame, x, y, aperture_radius,
                      annulus_width, mask_threshold, saturation, channels):
    """Constrained-Gaussian per-channel measurement at ``(x, y)``.

    Returns a dict with the fitted ``xcen``, ``ycen``, ``fwhm`` and per-channel
    ``flux_<ch>``, ``background_<ch>`` and ``sat_<ch>`` (the saturation flag is
    evaluated at the fitted center), or None when the luminance fit fails.
    """
    fit = _gaussian_psf_photometry(
        data, cube, lum_frame, x, y, aperture_radius, annulus_width,
        mask_threshold)
    if fit is None:
        return None
    cols = {"xcen": fit["xcen"], "ycen": fit["ycen"], "fwhm": fit["fwhm"]}
    for index, channel in enumerate(channels):
        cols[f"flux_{channel}"] = fit[f"flux_{channel}"]
        cols[f"background_{channel}"] = fit[f"background_{channel}"]
        cols[f"sat_{channel}"] = _aperture_saturated(
            cube[index], fit["xcen"], fit["ycen"], aperture_radius, saturation)
    return cols


def _airmass(altitude_deg):
    """
    Kasten-Young airmass for an apparent altitude in degrees (scalar or array).
    """
    alt = np.asarray(altitude_deg, dtype=float)
    return 1.0 / (np.sin(np.radians(alt))
                  + 0.50572 * (alt + 6.07995) ** -1.6364)


def _catalog_calibration_map():
    """
    Map each named-catalog star label to its catalog colors and magnitudes.

    Keyed by the same labels :func:`_alcor_star_labels` assigns to photometry
    rows, so the map joins directly against a photometry DataFrame. Each value is
    a dict with ``BV`` (the B-V color) and the catalog Johnson magnitudes ``R``
    (= V-(V-R)), ``V`` (= Vmag) and ``B`` (= V+(B-V)); any entry whose catalog
    color is missing is ``nan``.
    """
    catpath = files(__package__) / "data" / "bright_star_sloan_named.fits"
    cat = Table.read(str(catpath))
    labels = _alcor_star_labels(cat)
    out = {}
    for label, row in zip(labels, cat):
        def _value(name):
            value = _catalog_value_to_python(row[name])
            return np.nan if value is None else float(value)
        v = _value("Vmag")
        bv = _value("B-V")
        vr = _value("V-R")
        out[label] = {"BV": bv, "V": v, "R": v - vr, "B": v + bv}
    return out


def _zeropoint_row_params(df, time):
    """
    Per-row ``(zp, color_coeff)`` arrays for each band.

    The zeropoint epoch is resolved from ``time`` when given, else per row from
    an ``OBSTIME`` column when present, else the most recent epoch. Returns
    ``(zp, color)`` where each is a ``{band: ndarray}`` aligned to ``df``'s rows.
    """
    n = len(df)
    epoch_jds = np.array([Time(z["epoch"], scale="utc").jd
                          for z in ALCOR_ZEROPOINTS])
    if time is not None:
        row_jd = np.full(n, Time(time).jd)
    elif n and "OBSTIME" in df.columns:
        row_jd = Time(np.asarray(df["OBSTIME"], dtype="datetime64[ns]")).jd
        row_jd = np.atleast_1d(np.asarray(row_jd, dtype=float))
    else:
        row_jd = np.full(n, epoch_jds.max())
    # nearest epoch per row; iterate ascending jd so a tie resolves to the more
    # recent epoch (matching alcor_zeropoint / alcor_calibration).
    pick = np.zeros(n, dtype=int)
    best = np.full(n, np.inf)
    for k in np.argsort(epoch_jds):
        dist = np.abs(row_jd - epoch_jds[k])
        take = dist <= best
        pick[take] = k
        best[take] = dist[take]
    zp, color = {}, {}
    for band in ALCOR_ZEROPOINT_BANDS:
        zp_vals = np.array([z[band]["zp"] for z in ALCOR_ZEROPOINTS])
        cc_vals = np.array([z[band]["color_coeff"] for z in ALCOR_ZEROPOINTS])
        zp[band] = zp_vals[pick]
        color[band] = cc_vals[pick]
    return zp, color


def alcor_calibrate_photometry(df, time=None):
    """
    Add calibrated catalog-system magnitudes and cloud-extinction offsets.

    For every instrument magnitude column in ``df`` (``mag_{r,g,b}`` and any
    ``_ap``/``_gauss`` suffixed variants) this adds two columns:

    ``cal_{band}[suffix]``
        the magnitude on the catalog system,
        ``(instr_mag - ALCOR_AIRMASS_TERM*airmass) + zp + color_coeff*(B-V)``,
        ``nan`` where the instrument mag is non-finite or brighter than
        ``ALCOR_BRIGHT_CUT`` (the CMOS non-linear regime, where the calibration
        is invalid).
    ``ext_{band}[suffix]``
        ``cal - catalog_mag``: the offset from the star's catalog magnitude --
        the line-of-sight extinction (e.g. cloud attenuation) in magnitudes,
        positive when the star is dimmer than catalog.

    Zeropoints come from :func:`alcor_zeropoint`; the epoch is resolved from
    ``time`` when given, else per row from an ``OBSTIME`` column when present,
    else the most recent epoch. Star names come from a ``name`` column when
    present, else the index. ``B-V`` and the catalog Johnson R/V/B are looked up
    by name (channel->catalog G->V, R->R, B->B); stars absent from the catalog or
    lacking the needed color get ``nan``. Requires an ``altitude`` column.
    Returns a new DataFrame.
    """
    if "altitude" not in df.columns:
        raise ValueError(
            "alcor_calibrate_photometry requires an 'altitude' column")
    df = df.copy()
    names = (df["name"] if "name" in df.columns
             else df.index.to_series()).astype(str)
    cmap = _catalog_calibration_map()
    bv = names.map(lambda name: cmap.get(name, {}).get("BV", np.nan)).to_numpy(
        dtype=float)
    catmag = {
        band: names.map(
            lambda name, cat=cat: cmap.get(name, {}).get(cat, np.nan)
        ).to_numpy(dtype=float)
        for band, cat in ALCOR_ZEROPOINT_BANDS.items()
    }
    airmass = _airmass(df["altitude"].to_numpy(dtype=float))
    zp, color = _zeropoint_row_params(df, time)
    for band in ALCOR_ZEROPOINT_BANDS:
        for col in (f"mag_{band}", f"mag_{band}_ap", f"mag_{band}_gauss"):
            if col not in df.columns:
                continue
            suffix = col[len(f"mag_{band}"):]
            instr = df[col].to_numpy(dtype=float)
            cal = (instr - ALCOR_AIRMASS_TERM * airmass
                   + zp[band] + color[band] * bv)
            cal = np.where(np.isfinite(instr) & (instr > ALCOR_BRIGHT_CUT),
                           cal, np.nan)
            df[f"cal_{band}{suffix}"] = cal
            df[f"ext_{band}{suffix}"] = cal - catmag[band]
    return df


def alcor_star_photometry(filename, output_file=None, aperture_radius=4.0,
                          annulus_width=1.0, min_altitude=20.0,
                          vmag_limit=5.5, refraction=True, masks_dir=None,
                          check_plot=False, check_radius=680,
                          sun_alt_max=-12.0, saturation=ALCOR_SATURATION,
                          gaussian=False,
                          mask_threshold=ALCOR_NONLINEAR_THRESHOLD,
                          both=False):
    """
    Measure fixed-position aperture photometry for bright named stars.

    The image is loaded with :func:`load_alcor_fits` using bad-pixel repair, then
    a per-channel bias level is subtracted from the median of the four 10x10
    image corners. Catalog stars from ``bright_star_sloan_named.fits`` with
    ``Vmag <= vmag_limit`` and altitude above ``min_altitude`` are projected
    into the raw image via the frame WCS. Each channel is measured with a
    circular aperture and a surrounding annulus. Each channel also carries a
    ``sat_*`` flag, True when any raw pixel inside the aperture reaches
    ``saturation`` (the 15-bit ceiling by default), so saturated measurements
    can be filtered downstream without discarding the unsaturated channels.

    When ``gaussian`` is True, photometry instead fits a circular Gaussian whose
    center and width are pinned from a luminance (channel-summed) fit with the
    non-linear core masked (raw pixels at/above ``mask_threshold`` excluded), and
    recovers each channel's amplitude from the linear wings. The reported flux is
    the analytic Gaussian integral, which is robust to the CMOS non-linearity
    that suppresses aperture-sum flux for bright stars before saturation. The
    luminance FWHM is reported in the ``fwhm`` column (NaN in aperture mode).

    Every catalog star above ``min_altitude`` produces a row in every frame,
    even when it is not detected: a star that is above the horizon but invisible
    (e.g. behind cloud) carries ``flux = 0`` and ``mag = NaN`` per channel rather
    than being dropped, because that non-detection is itself a strong extinction
    signal (and the measured ``background_*`` is still recorded). The only stars
    absent from the output are those below ``min_altitude`` or fainter than
    ``vmag_limit``. The per-channel ``background_*`` is finite for a measurable
    position even at a non-detection, and NaN only when the position is off-frame.

    When ``both`` is True, every star is measured with *both* methods in a single
    pass and written to one combined CSV: the aperture columns are suffixed
    ``_ap`` and the Gaussian columns ``_gauss``, alongside the shared
    WCS-predicted ``xcen``/``ycen`` and the Gaussian-fitted
    ``xcen_gauss``/``ycen_gauss``/``fwhm``. The aperture flux is 0 at a
    non-detection; the Gaussian columns are NaN when the fit cannot run at all
    (a structural failure, distinct from a measured zero). Rows sort by
    ``flux_g_ap``. ``both`` takes precedence over ``gaussian``.

    Every measured magnitude is calibrated to the catalog system via
    :func:`alcor_calibrate_photometry` (using the frame time to resolve the
    zeropoint epoch), adding per-channel ``cal_*`` (calibrated catalog-system
    magnitude: G->V, R->R, B->B) and ``ext_*`` (the calibrated-minus-catalog
    offset, i.e. the line-of-sight cloud extinction in magnitudes). Both are NaN
    for measurements brighter than ``ALCOR_BRIGHT_CUT`` (CMOS non-linear regime)
    or for stars lacking a catalog color.

    Returns
    -------
    phot : `pandas.DataFrame`
        Rows indexed by star name. Columns are ``altitude``, ``azimuth``,
        ``xcen``, ``ycen`` and per-channel ``flux_*``, ``mag_*``, ``cal_*``,
        ``ext_*``, ``background_*``, ``sat_*`` for ``r``, ``g``, ``b`` (suffixed
        ``_ap``/``_gauss`` in ``both`` mode). One row per catalog star above
        ``min_altitude``; non-detections carry ``flux = 0`` / ``mag = NaN``. Rows
        are sorted by descending ``flux_g``. Empty only when the frame is
        rejected (Sun above ``sun_alt_max``).
    output_file : `~pathlib.Path` or None
        CSV path written, or None when the frame is rejected.
    """
    if aperture_radius <= 0:
        raise ValueError("aperture_radius must be positive")
    if annulus_width <= 0:
        raise ValueError("annulus_width must be positive")
    channels = ("r", "g", "b")
    if both:
        columns = ["altitude", "azimuth", "xcen", "ycen"]
        for channel in channels:
            columns += [f"flux_{channel}_ap", f"mag_{channel}_ap",
                        f"cal_{channel}_ap", f"ext_{channel}_ap",
                        f"background_{channel}_ap", f"sat_{channel}_ap"]
        columns += ["xcen_gauss", "ycen_gauss", "fwhm"]
        for channel in channels:
            columns += [f"flux_{channel}_gauss", f"mag_{channel}_gauss",
                        f"cal_{channel}_gauss", f"ext_{channel}_gauss",
                        f"background_{channel}_gauss", f"sat_{channel}_gauss"]
        sort_key = "flux_g_ap"
    else:
        columns = ["altitude", "azimuth", "xcen", "ycen", "fwhm"]
        for channel in channels:
            columns += [f"flux_{channel}", f"mag_{channel}",
                        f"cal_{channel}", f"ext_{channel}",
                        f"background_{channel}", f"sat_{channel}"]
        sort_key = "flux_g"
    empty = pd.DataFrame(columns=columns)
    empty.index.name = "name"

    filename = Path(filename)
    time = _alcor_frame_time(filename)
    if time is None:
        raise ValueError(f"could not determine frame time from {filename}")
    sun_alt = _sun_altitude(time)
    if sun_alt > sun_alt_max:
        print(
            f"Warning: rejecting {filename}: Sun altitude {sun_alt:.1f} deg "
            f"is greater than {sun_alt_max:.1f} deg.",
            file=sys.stderr,
        )
        return empty, None

    cube, wcs, _ = load_alcor_fits(filename, badpix="repair",
                                   masks_dir=masks_dir)
    bias = _corner_bias(cube, size=10)
    data = cube.astype(float, copy=False) - bias[:, None, None]

    cat = alcor_named_reference_altaz(
        time, vmag_limit=vmag_limit, min_alt=min_altitude,
        refraction=refraction,
    )
    # all_world2pix with quiet=True returns the best (possibly unconverged)
    # estimate instead of warning: the iterative SIP/radial inverse fails its
    # tight tolerance for stars at alt <~ 1 deg (largest radii, steepest radial
    # distortion), but the returned pixel is still good to well under a pixel and
    # those horizon stars are the least critical. Equivalent to world_to_pixel_values
    # otherwise (verified byte-identical above the horizon).
    xcen, ycen = wcs.all_world2pix(cat["Az"], cat["Alt"], 0, quiet=True)
    xcen = np.asarray(xcen, dtype=float)
    ycen = np.asarray(ycen, dtype=float)

    rows = []
    labels = []
    cat_labels = _alcor_star_labels(cat)
    lum_frame = data.sum(axis=0) if (gaussian or both) else None
    for i, (x, y) in enumerate(zip(xcen, ycen)):
        base = {"altitude": float(cat["Alt"][i]),
                "azimuth": float(cat["Az"][i])}
        if both:
            ap = _aperture_measure(data, cube, x, y, aperture_radius,
                                   annulus_width, saturation, channels)
            gfit = _gaussian_measure(data, cube, lum_frame, x, y,
                                     aperture_radius, annulus_width,
                                     mask_threshold, saturation, channels)
            row = dict(base, xcen=float(x), ycen=float(y))
            for channel in channels:
                flux, mag = _flux_mag(ap[f"flux_{channel}"])
                row[f"flux_{channel}_ap"] = flux
                row[f"mag_{channel}_ap"] = mag
                row[f"background_{channel}_ap"] = ap[f"background_{channel}"]
                row[f"sat_{channel}_ap"] = ap[f"sat_{channel}"]
            if gfit is not None:
                row["xcen_gauss"] = gfit["xcen"]
                row["ycen_gauss"] = gfit["ycen"]
                row["fwhm"] = gfit["fwhm"]
                for channel in channels:
                    flux, mag = _flux_mag(gfit[f"flux_{channel}"])
                    row[f"flux_{channel}_gauss"] = flux
                    row[f"mag_{channel}_gauss"] = mag
                    row[f"background_{channel}_gauss"] = \
                        gfit[f"background_{channel}"]
                    row[f"sat_{channel}_gauss"] = gfit[f"sat_{channel}"]
            else:
                # the Gaussian fit could not run: no estimate for this method,
                # so its columns are NaN (the aperture flux above still carries
                # the detection / non-detection). A genuinely empty frame yields
                # a degenerate fit with flux 0, not a None fit, so this branch is
                # a structural failure, not the cloud non-detection signal.
                row["xcen_gauss"] = np.nan
                row["ycen_gauss"] = np.nan
                row["fwhm"] = np.nan
                for channel in channels:
                    row[f"flux_{channel}_gauss"] = np.nan
                    row[f"mag_{channel}_gauss"] = np.nan
                    row[f"background_{channel}_gauss"] = np.nan
                    row[f"sat_{channel}_gauss"] = np.nan
        elif gaussian:
            gfit = _gaussian_measure(data, cube, lum_frame, x, y,
                                     aperture_radius, annulus_width,
                                     mask_threshold, saturation, channels)
            if gfit is None:
                # the fit could not run: keep the star at its WCS-predicted
                # position with NaN measurements (no estimate available).
                row = dict(base, xcen=float(x), ycen=float(y), fwhm=np.nan)
                for channel in channels:
                    row[f"flux_{channel}"] = np.nan
                    row[f"mag_{channel}"] = np.nan
                    row[f"background_{channel}"] = np.nan
                    row[f"sat_{channel}"] = np.nan
            else:
                row = dict(base, xcen=gfit["xcen"], ycen=gfit["ycen"],
                           fwhm=gfit["fwhm"])
                for channel in channels:
                    flux, mag = _flux_mag(gfit[f"flux_{channel}"])
                    row[f"flux_{channel}"] = flux
                    row[f"mag_{channel}"] = mag
                    row[f"background_{channel}"] = gfit[f"background_{channel}"]
                    row[f"sat_{channel}"] = gfit[f"sat_{channel}"]
        else:
            ap = _aperture_measure(data, cube, x, y, aperture_radius,
                                   annulus_width, saturation, channels)
            row = dict(base, xcen=float(x), ycen=float(y), fwhm=np.nan)
            for channel in channels:
                flux, mag = _flux_mag(ap[f"flux_{channel}"])
                row[f"flux_{channel}"] = flux
                row[f"mag_{channel}"] = mag
                row[f"background_{channel}"] = ap[f"background_{channel}"]
                row[f"sat_{channel}"] = ap[f"sat_{channel}"]
        rows.append(row)
        labels.append(cat_labels[i])

    phot = pd.DataFrame(rows, index=labels, columns=columns)
    phot.index.name = "name"
    phot = alcor_calibrate_photometry(phot, time=time)
    phot = phot.sort_values(sort_key, ascending=False, na_position="last")
    output_file = (_default_alcor_photometry_output(filename)
                   if output_file is None else Path(output_file))
    phot.to_csv(output_file)
    if check_plot:
        check_plot_file = (_default_alcor_photometry_check_plot_output(filename)
                           if check_plot is True else Path(check_plot))
        save_alcor_photometry_check_plot(
            filename,
            phot,
            check_plot_file,
            aperture_radius=aperture_radius,
            annulus_width=annulus_width,
            radius=check_radius,
        )
    return phot, output_file


def collect_alcor_photometry(inputs):
    """
    Collect per-star photometry from a set of ``*_phot.csv`` files.

    Each input file is one frame's :func:`alcor_star_photometry` output. The
    observation time is parsed from the file's ``YYYY_MM_DD__HH_MM_SS``
    filename stamp (local MST, so UT = stamp + 7h); the CSVs carry no time
    information internally, so files whose names do not parse are skipped with
    a warning, as are unreadable/malformed CSVs.

    Parameters
    ----------
    inputs : str, Path, or iterable of str/Path
        A directory to glob for ``*_phot.csv``, or the CSV paths themselves.

    Returns
    -------
    ~pandas.DataFrame
        The combined photometry with ``name`` as a regular column and a UT
        ``OBSTIME`` datetime column, sorted by ``name`` then ``OBSTIME`` so
        ``df.groupby("name")`` yields each star's time-ordered measurements.

    Raises
    ------
    ValueError
        If no usable input files remain after skipping.
    """
    if isinstance(inputs, (str, Path)) and Path(inputs).is_dir():
        files = sorted(Path(inputs).glob("*_phot.csv"))
    else:
        files = [Path(f) for f in inputs]

    frames = []
    for f in files:
        obstime = _filename_ut_datetime(f)
        if obstime is None:
            print(f"Warning: skipping {f.name}: no parseable timestamp "
                  "in filename", file=sys.stderr)
            continue
        try:
            phot = pd.read_csv(f)
        except (OSError, ValueError, UnicodeDecodeError,
                pd.errors.ParserError) as exc:
            print(f"Warning: skipping {f.name}: {exc}", file=sys.stderr)
            continue
        if "name" not in phot.columns:
            print(f"Warning: skipping {f.name}: no 'name' column",
                  file=sys.stderr)
            continue
        phot.insert(1, "OBSTIME", pd.Timestamp(obstime))
        frames.append(phot)

    if not frames:
        raise ValueError(f"No usable *_phot.csv files in {inputs}.")

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["name", "OBSTIME"], ignore_index=True)


def load_alcor_fits(filename, wcs=None, badpix="repair", masks_dir=None):
    """
    Load an alcor OMEA 8C FITS file and return ``(cube, wcs, mask)``.

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
                              horizon_radius=cal["horizon_radius"],
                              tangential_coeffs=cal["tangential_coeffs"],
                              axis_tilt=cal["axis_tilt"])

    return cube, wcs, mask


def alcor_proc_fits(filename, output_file=None, overwrite=False, **kwargs):
    """
    Process an alcor OMEA 8C FITS file via `load_alcor_fits` and write a new
    FITS file containing the raw ``(3, ny, nx)`` RGB cube (native orientation)
    with the raw-frame alt/az WCS encoded in the header.

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
        Forwarded to `load_alcor_fits` (``wcs``, ``badpix``, ``masks_dir``).

    Returns
    -------
    output_file : `~pathlib.Path`
        Path to the written FITS file.
    """
    cube, wcs, _ = load_alcor_fits(filename, **kwargs)
    if output_file is None:
        stem = str(filename)
        for ext in (".fits.bz2", ".fits.gz", ".fits"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        output_file = stem + "_proc.fits"
    output_file = Path(output_file)

    hdu = fits.PrimaryHDU(data=cube.astype(np.float32),
                          header=wcs.to_header(relax=True))
    hdu.writeto(output_file, overwrite=overwrite)
    return output_file


def alcor_keogram(input_dir, pattern="*.fits.bz2", workers=1, progress=False, progress_file=None, **kwargs):
    """
    Build a keogram from a directory of alcor OMEA 8C FITS images.

    Each input image is loaded with `load_alcor_fits`, and the zenith column
    (the column through the WCS alt=90 pixel) of the raw RGB cube is copied
    into the next column of the keogram.
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
        Forwarded to `load_alcor_fits` (``wcs``, ``badpix``, ``masks_dir``).

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

    cube, wcs, _ = load_alcor_fits(filename, **kwargs)
    zx, _ = wcs.world_to_pixel_values(0.0, 90.0)
    zcol = int(round(float(zx)))                          # 0-based zenith column
    return index, timestamp, cube[:, :, zcol].T, filename.name  # (ny, 3) RGB strip


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


def plot_alcor_fits(filename, outimage=None, outfig=None, radius=680,
                    powerstretch=0.75, contrast=0.35, gscale=0.7, bscale=1.7,
                    figsize=12):
    """
    Take a FITS file as produced by the alcor OMEA 8C and create an annotated
    all-sky figure for display.

    The raw cube and its WCS are loaded; the display RGB is bias-subtracted,
    stretched, cropped to a ``radius``-pixel square around the WCS zenith, and
    rendered with ``origin="lower"`` (north-up). Geometry comes entirely from
    the WCS.

    Parameters
    ----------
    filename : str
        FITS filename of image. Uses astropy.io.fits so gz and bz2 extensions are allowed.
    outimage : str (default=None)
        If not None, write out raw, unannotated cropped image.
    outfig : str (default=None)
        If not None, write out annotated image as produced by matplotlib.
    radius : float (default=680)
        Half-width (pixels) of the display crop around the zenith.
    powerstretch : float (default=0.75)
        Power of the stretch function to use.
    contrast : float (default=0.35)
        ZScale contrast factor.
    gscale : float (default=0.7)
        Scale factor to apply to green channel.
    bscale : float (default=1.7)
        Scale factor to apply to blue channel.
    figsize : float (default=12)
        Size of matplotlib figure in inches.
    """
    cube, wcs, _ = load_alcor_fits(filename)
    # The factors to scale the green and blue channels were determined
    # empirically and provide a reasonably good white/color balance for both day
    # and night images. Subtract the per-channel bias first; otherwise the raw
    # pedestal is color-scaled too and the image turns purple.
    rgb = _alcor_display_rgb(cube, powerstretch=powerstretch, contrast=contrast,
                             gscale=gscale, bscale=bscale)

    zx, zy = wcs.world_to_pixel_values(0.0, 90.0)
    xz = int(round(float(zx)))                            # 0-based zenith
    yz = int(round(float(zy)))
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
    # Use the WCS to map altitude ticks to the correct radial fraction. The polar
    # overlay spans the figure region, so r=1 corresponds to `radius` pixels from zenith.
    tick_alts = np.array([75, 60, 45, 30, 15])
    px, py = wcs.world_to_pixel_values(np.zeros_like(tick_alts, dtype=float),
                                       tick_alts.astype(float))
    yticks = np.hypot(px - xz, py - yz) / radius
    ylabels = [f" {a}°" for a in tick_alts]
    pax.set_yticks(yticks, labels=ylabels, color="white", alpha=0.5, fontsize=16)
    pax.set_rlabel_position(90)
    pax.tick_params(grid_alpha=0.5)
    pax.tick_params(axis='x', labelsize=16, labelcolor='silver', pad=10)

    if outfig is not None:
        plt.savefig(outfig, transparent=True, bbox_inches='tight', pad_inches=0)

    return fig


def save_alcor_photometry_check_plot(filename, phot, output_file,
                                     aperture_radius=4.0, annulus_width=1.0,
                                     radius=680, powerstretch=0.75,
                                     contrast=0.35, gscale=0.7, bscale=1.7,
                                     figsize=12):
    """
    Save a ``plot_alcor_fits`` rendering with measured apertures overlaid.

    ``phot`` is the DataFrame returned by :func:`alcor_star_photometry`, with
    raw-frame ``xcen`` and ``ycen`` columns. Apertures outside the displayed crop
    are skipped.
    """
    output_file = Path(output_file)
    fig = plot_alcor_fits(
        filename,
        outfig=None,
        radius=radius,
        powerstretch=powerstretch,
        contrast=contrast,
        gscale=gscale,
        bscale=bscale,
        figsize=figsize,
    )
    ax = fig.axes[0]

    cube, wcs, _ = load_alcor_fits(filename)
    zx, zy = wcs.world_to_pixel_values(0.0, 90.0)
    xz = int(round(float(zx)))
    yz = int(round(float(zy)))
    ny, nx = cube.shape[1:]
    yl, yu = max(0, yz - radius), min(ny, yz + radius)
    xl, xu = max(0, xz - radius), min(nx, xz + radius)
    annulus_inner = aperture_radius + 1.0
    outer = annulus_inner + annulus_width

    for _, row in phot.iterrows():
        x = float(row["xcen"])
        y = float(row["ycen"])
        if x + outer < xl or x - outer > xu or y + outer < yl or y - outer > yu:
            continue
        cx = x - xl
        cy = y - yl
        ax.add_patch(Circle((cx, cy), outer, facecolor="none",
                            edgecolor="cyan", linewidth=0.7, alpha=0.35))
        ax.add_patch(Circle((cx, cy), annulus_inner, facecolor="none",
                            edgecolor="cyan", linewidth=0.6, alpha=0.25,
                            linestyle="--"))
        ax.add_patch(Circle((cx, cy), aperture_radius, facecolor="none",
                            edgecolor="yellow", linewidth=0.9, alpha=0.8))

    fig.savefig(output_file, transparent=True, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_file


def alcor_proc_fits_cli():
    """
    CLI entry point for `alcor_proc_fits`. Writes a processed FITS file with
    the alt/az WCS encoded in the header.
    """
    parser = argparse.ArgumentParser(
        description="Process an alcor OMEA 8C FITS image into a raw (3, ny, nx) FITS cube with raw-frame alt/az WCS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument("-o", "--output", default=None, help="Output FITS path (default: <input>_proc.fits).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")
    args = parser.parse_args()

    out = alcor_proc_fits(
        args.filename,
        output_file=args.output,
        overwrite=args.overwrite,
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
    parser.add_argument("--radius", type=int, default=680,
                        help="Half-width (pixels) of the display crop around the zenith.")
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
        radius=args.radius,
        powerstretch=args.powerstretch,
        contrast=args.contrast,
        gscale=args.gscale,
        bscale=args.bscale,
        figsize=args.figsize,
    )
    print(outfig)


def alcor_star_photometry_cli():
    """
    CLI entry point for ``alcor_star_photometry``. Writes fixed-position
    aperture photometry for Vmag-limited named bright stars.
    """
    parser = argparse.ArgumentParser(
        description="Measure Alcor RGB aperture photometry for named bright stars at WCS-predicted positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path (default: <input>_phot.csv).",
    )
    parser.add_argument("--aperture-radius", type=float, default=4.0,
                        help="Circular aperture radius in pixels (also the Gaussian fit window).")
    parser.add_argument("--annulus-width", type=float, default=1.0,
                        help="Background annulus width in pixels.")
    parser.add_argument("--min-altitude", type=float, default=20.0,
                        help="Minimum catalog-star altitude in degrees.")
    parser.add_argument("--vmag-limit", type=float, default=5.5,
                        help="Faintest V magnitude to measure.")
    parser.add_argument("--no-refraction", action="store_true",
                        help="Disable atmospheric refraction in the catalog Alt/Az transform.")
    parser.add_argument("--masks-dir", default=None,
                        help="Bad-pixel mask directory (default: $ALCOR_BADPIX_DIR, then packaged masks).")
    parser.add_argument("--sun-alt-max", type=float, default=-12.0,
                        help="Reject images with Sun altitude greater than this (deg).")
    parser.add_argument("--saturation", type=float, default=ALCOR_SATURATION,
                        help="Raw-ADU level at/above which an aperture pixel flags the channel saturated.")
    parser.add_argument("--gaussian", action="store_true",
                        help="Use constrained-Gaussian PSF photometry instead of aperture sums.")
    parser.add_argument("--both", action="store_true",
                        help="Measure both aperture and Gaussian in one pass into a single "
                             "combined CSV (columns suffixed _ap / _gauss). Overrides --gaussian.")
    parser.add_argument("--mask-threshold", type=float,
                        default=ALCOR_NONLINEAR_THRESHOLD,
                        help="Raw-ADU level at/above which a pixel is excluded from the Gaussian fit.")
    parser.add_argument("--check-plot", action="store_true",
                        help="Write an aperture-overlay check plot as <input>_phot.pdf.")
    parser.add_argument("--check-radius", type=int, default=680,
                        help="Half-width in pixels of the check-plot crop around the zenith.")
    args = parser.parse_args()

    _, output_file = alcor_star_photometry(
        args.filename,
        output_file=args.output,
        aperture_radius=args.aperture_radius,
        annulus_width=args.annulus_width,
        min_altitude=args.min_altitude,
        vmag_limit=args.vmag_limit,
        refraction=not args.no_refraction,
        masks_dir=args.masks_dir,
        check_plot=args.check_plot,
        check_radius=args.check_radius,
        sun_alt_max=args.sun_alt_max,
        saturation=args.saturation,
        gaussian=args.gaussian,
        mask_threshold=args.mask_threshold,
        both=args.both,
    )
    if output_file is not None:
        print(output_file)
    if args.check_plot and output_file is not None:
        print(_default_alcor_photometry_check_plot_output(args.filename))


def _format_calibration_entry(result):
    """
    Format a calibration result as a paste-ready ALCOR_CALIBRATIONS entry.
    """
    rc = tuple(float(c) for c in result["radial_coeffs"])
    tc = tuple(float(c) for c in result.get("tangential_coeffs", (0.0, 0.0)))
    at = tuple(float(c) for c in result.get("axis_tilt", (0.0, 0.0)))
    return (f'    {{"epoch": "{result["epoch"]}", '
            f'"xcen": {result["xcen"]:.3f}, '
            f'"ycen": {result["ycen"]:.3f}, '
            f'"rotation": {result["rotation"]:.4f}, '
            f'"radial_coeffs": {rc!r}, '
            f'"tangential_coeffs": {tc!r}, '
            f'"axis_tilt": {at!r}, '
            f'"horizon_radius": {result["horizon_radius"]:.1f}}},')


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
    tn, te = result.get("axis_tilt", (0.0, 0.0))
    eps = float(np.hypot(tn, te))
    a0 = float(np.degrees(np.arctan2(te, tn))) % 360.0
    print(f"# axis tilt: eps={eps:.4f} deg toward az={a0:.1f} deg")
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
    """
    CLI entry point for :func:`create_badpix_mask` (run daily from cron).
    """
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
