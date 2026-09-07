"""Star detection, catalog matching, and the WCS geometry fit."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, hstack
from astropy.time import Time
from photutils.detection import DAOStarFinder

from .config import ALCOR_HORIZON_RADIUS, alcor_calibration
from .timeutils import (
    _filename_ut_datetime, _frame_time, _read_frame_date, select_dark_frames
)
from .wcs import _predict_pixels
from .catalogs import alcor_reference_altaz
from .io import load_alcor_fits


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
