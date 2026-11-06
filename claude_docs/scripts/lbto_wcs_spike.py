"""
Phase-0 spike: can the MMT lens model fit the LBTO/VATT OMEA frames?

The question this answers, and nothing more: does the odd-power radial +
Brown-Conrady + axis-tilt distortion model that ``ALCOR_CALIBRATIONS`` encodes
describe a DIFFERENT fisheye in front of a DIFFERENT sensor? If it does, the
instrument-profile refactor is worth doing; if the residuals are structured and
large, the geometry layer needs rethinking before anything else.

Three things are unknown for LBTO and are searched rather than assumed:

* **parity** -- an all-sky camera images the sky from below, so handedness is a
  property of the optical train, not a convention we may pick. The MMT's WCS
  puts north at +y with a det=+1 PC matrix; if LBTO's train has one more
  reflection, no rotation will ever match. Both hypotheses are tried by
  reflecting the DETECTIONS (x -> nx-1-x), which costs nothing, rather than
  re-detecting on a flipped cube.
* **rotation** -- the camera's azimuth clocking, scanned over the full circle.
* **horizon_radius** -- the plate scale, i.e. pixels from the optical axis to
  altitude 0. The visible image circle is a good first guess but its edge is
  not exactly alt=0, so it is scanned too. ``_fit_params`` holds this fixed
  (the zenith plate scale is set by it), which is why it belongs in the seed
  search rather than the fit.

The seed objective is the number of asterism-verified matches, which is a
sharply peaked function of the seed: at a wrong rotation the pattern check
rejects essentially everything.

Two LBTO-specific details that are easy to get wrong:

* ``DATE-OBS`` is **UT** here, because their acquisition host runs UTC. At the
  MMT the same keyword is local MST. The rule is "exposure start in the host's
  timezone", not a site constant, so the time is taken from ``DATE-OBS`` plus
  half the exposure rather than from the filename parser.
* The exposure is 60 s. At this plate scale the sky moves ~4 px in that time
  near the celestial equator, so stars are short TRAILS, not points. The fit's
  residual floor is therefore expected to be well above the MMT's ~0.35 px, and
  a large residual here is not by itself evidence against the lens model.

Usage:
  lbto_wcs_spike.py <allsky-*.fits.fz>... -o OUT.png [--cache DIR]
"""

import argparse
import os
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from skycam_utils.alcor import (
    alcor_reference_altaz,
    assign_alcor_matches,
    detect_alcor_stars,
    save_alcor_residual_plot,
)
from skycam_utils.alcor.wcs import _predict_pixels
from skycam_utils.alcor.wcsfit import _fit_params

# Mt Graham. SITELAT/SITELONG in the LBTO headers are 'dd:mm:ss:00' -- a
# four-field variant of the usual sexagesimal, so they are transcribed here
# rather than parsed.
LBTO_LOCATION = EarthLocation.from_geodetic("-109:53:31", "+32:42:01", 3190 * u.m)

BAYER = "RGGB"          # BAYERIND = 4, as at the MMT


def _raw_to_cube(raw, bayer=BAYER):
    """
    Bilinear demosaic + 2x2 mean of a full-sensor RAW16 mosaic.

    A local copy of ``omea.imaging.raw_to_cube`` restricted to the full-frame,
    no-mirror case, so the spike does not depend on the omea checkout being
    importable. (Phase 2 of the design moves that function into skycam_utils,
    at which point this goes away.)
    """
    if bayer != "RGGB":
        raise ValueError("the spike only implements RGGB")
    ny, nx = raw.shape[0] // 2 * 2, raw.shape[1] // 2 * 2
    p = np.pad(raw[:ny, :nx].astype(np.float32), 2, mode="reflect")

    def s(dy, dx):
        return p[2 + dy:2 + dy + ny:2, 2 + dx:2 + dx + nx:2]

    r = (9 * s(0, 0) + 3 * s(0, 2) + 3 * s(2, 0) + s(2, 2)) / 16
    b = (9 * s(1, 1) + 3 * s(-1, 1) + 3 * s(1, -1) + s(-1, -1)) / 16
    g = (6 * (s(0, 1) + s(1, 0))
         + s(-1, 0) + s(0, -1) + s(2, 1) + s(1, 2)) / 16
    return np.stack([r, g, b])


def _frame(path, cache, fwhm, threshold_sigma, max_detections, vmag_limit,
           min_alt):
    """Return ``(time, cat, det, shape)`` for one frame, caching detections."""
    tag = f"f{fwhm:g}_t{threshold_sigma:g}_n{max_detections}"
    key = (None if cache is None
           else Path(cache) / f"{Path(path).stem}_{tag}_det.npz")
    hdr = fits.getheader(path, 1)
    exptime = float(hdr["EXPOSURE"])
    # DATE-OBS is UT at LBTO (their host runs UTC); use mid-exposure.
    time = Time(hdr["DATE-OBS"], format="isot", scale="utc") + exptime / 2 * u.s

    if key is not None and key.exists():
        z = np.load(key)
        det = Table({"xcentroid": z["x"], "ycentroid": z["y"], "flux": z["f"]})
        shape = tuple(z["shape"])
    else:
        with fits.open(path) as hdul:
            raw = hdul[1].data
        cube = _raw_to_cube(raw)
        shape = cube.shape[1:]
        det = detect_alcor_stars(cube, fwhm=fwhm,
                                 threshold_sigma=threshold_sigma,
                                 max_detections=max_detections)
        if key is not None:
            key.parent.mkdir(parents=True, exist_ok=True)
            np.savez(key, x=np.asarray(det["xcentroid"], dtype=float),
                     y=np.asarray(det["ycentroid"], dtype=float),
                     f=np.asarray(det["flux"], dtype=float),
                     shape=np.asarray(shape))
    cat = alcor_reference_altaz(time, vmag_limit=vmag_limit, min_alt=min_alt,
                                location=LBTO_LOCATION)
    return time, cat, det, shape


def _reflect(det, nx):
    """Detections under the opposite image parity."""
    out = Table(det)
    out["xcentroid"] = nx - 1.0 - np.asarray(det["xcentroid"], dtype=float)
    return out


def _params(xcen, ycen, rotation, horizon_radius, k3=0.0, k5=0.0):
    return dict(xcen=xcen, ycen=ycen, rotation=rotation,
                radial_coeffs=(1.0, k3, k5), tangential_coeffs=(0.0, 0.0),
                axis_tilt=(0.0, 0.0), horizon_radius=horizon_radius)


def _predict(alt, az, p):
    return _predict_pixels(alt, az, xcen=p["xcen"], ycen=p["ycen"],
                           rotation=p["rotation"],
                           radial_coeffs=tuple(p["radial_coeffs"]),
                           horizon_radius=p["horizon_radius"],
                           tangential_coeffs=tuple(p["tangential_coeffs"]),
                           axis_tilt=tuple(p["axis_tilt"]))


# --- cold-start solving ----------------------------------------------------
#
# fit_alcor_wcs warm-starts from the nearest known epoch, so its asterism-
# verified matcher always has a good seed. A new instrument has no epoch, and
# the pattern check cannot bootstrap itself: at a wrong seed it rejects
# everything, and the global optimum is then indistinguishable from any local
# one that happens to line up a corner of the frame. So the seed is found with
# plain MUTUAL nearest-neighbour hits on bright stars (unambiguous, and cheap),
# and only once the geometry is close does the asterism matcher take over.

def _pool_nn(frames, p, tol):
    """Mutual-nearest-neighbour matches: both directions must agree."""
    a, z, xs, ys = [], [], [], []
    for alt, az, dx, dy, tree in frames:
        px, py = _predict(alt, az, p)
        ok = np.isfinite(px) & np.isfinite(py)
        if not ok.any():
            continue
        pts = np.column_stack([px[ok], py[ok]])
        dist, idx = tree.query(pts)
        # reverse: nearest predicted star to each matched detection
        back = np.full(len(dx), -1)
        order = np.argsort(-dist)          # keep the closest claim per detection
        back[idx[order]] = np.arange(len(pts))[order]
        keep = (dist < tol) & (back[idx] == np.arange(len(pts)))
        if not keep.any():
            continue
        a.append(alt[ok][keep])
        z.append(az[ok][keep])
        xs.append(dx[idx[keep]])
        ys.append(dy[idx[keep]])
    if not a:
        return None
    return (np.concatenate(a), np.concatenate(z),
            np.concatenate(xs), np.concatenate(ys))


def _seed_score(frames, p, tol):
    pooled = _pool_nn(frames, p, tol)
    if pooled is None:
        return 0, 0
    alt, az, _, _ = pooled
    # Reward coverage as well as count: a corner lock scores many matches in
    # one place, which is exactly the failure mode to avoid.
    octants = len(np.unique((np.asarray(az) // 45).astype(int)))
    return len(alt) * octants, len(alt)


_FREE = ("xcen", "ycen", "rotation", "hr", "k3", "k5", "p1", "p2", "tn", "te")


def _flat(p):
    return dict(xcen=p["xcen"], ycen=p["ycen"], rotation=p["rotation"],
                hr=p["horizon_radius"],
                k3=p["radial_coeffs"][1], k5=p["radial_coeffs"][2],
                p1=p["tangential_coeffs"][0], p2=p["tangential_coeffs"][1],
                tn=p["axis_tilt"][0], te=p["axis_tilt"][1])


def _unflat(v):
    return dict(xcen=v["xcen"], ycen=v["ycen"], rotation=v["rotation"],
                radial_coeffs=(1.0, v["k3"], v["k5"]),
                tangential_coeffs=(v["p1"], v["p2"]),
                axis_tilt=(v["tn"], v["te"]), horizon_radius=v["hr"])


def _fit_stage(alt, az, x, y, p, free):
    """
    Least-squares fit of only the named parameters, the rest held at ``p``.

    ``_fit_params`` always frees the tangential and axis-tilt terms, which is
    right for a warm start but ruinous here: on a partial match set those two
    absorb the seed error and lock the solution into the corner it started in.
    Opening the model in stages keeps each stage well-conditioned.
    """
    v = _flat(p)
    names = [n for n in _FREE if n in free]
    scale = {"xcen": 10.0, "ycen": 10.0, "rotation": 1.0, "hr": 10.0,
             "k3": 0.05, "k5": 0.05, "p1": 1e-3, "p2": 1e-3,
             "tn": 0.1, "te": 0.1}

    def resid(q):
        w = dict(v)
        w.update(zip(names, q))
        px, py = _predict(alt, az, _unflat(w))
        r = np.concatenate([px - x, py - y])
        return np.where(np.isfinite(r), r, 1e3)

    out = least_squares(resid, np.array([v[n] for n in names], dtype=float),
                        x_scale=[scale[n] for n in names],
                        loss="soft_l1", f_scale=3.0)
    w = dict(v)
    w.update(zip(names, out.x))
    return _unflat(w)


def _pool(frames, params, tolerance, pattern_tol):
    a, z, xs, ys = [], [], [], []
    for _, cat, det in frames:
        m = assign_alcor_matches(cat, det, params=params, tolerance=tolerance,
                                 horizon_radius=params["horizon_radius"],
                                 pattern_tol=pattern_tol)
        if len(m) == 0:
            continue
        a.append(np.asarray(m["Alt"], dtype=float))
        z.append(np.asarray(m["Az"], dtype=float))
        xs.append(np.asarray(m["xcentroid"], dtype=float))
        ys.append(np.asarray(m["ycentroid"], dtype=float))
    if not a:
        return None
    return (np.concatenate(a), np.concatenate(z),
            np.concatenate(xs), np.concatenate(ys))


def _circle(path):
    """Centre and radius of the illuminated circle, in CUBE pixels."""
    with fits.open(path) as hdul:
        raw = hdul[1].data.astype(np.float32)
    ny, nx = raw.shape
    bias = float(np.median(np.concatenate(
        [raw[:30, :30].ravel(), raw[:30, -30:].ravel(),
         raw[-30:, :30].ravel(), raw[-30:, -30:].ravel()])))
    k = 32
    c = (raw[:ny // k * k, :nx // k * k] - bias).reshape(
        ny // k, k, nx // k, k).mean(axis=(1, 3))
    m = c > 0.25 * np.median(c[c > 0])
    ys, xs = np.where(m)
    # Spans in RAW pixels; raw px -> cube px is a factor of 2 (the 2x2 bin).
    span_x = (xs.max() - xs.min() + 1) * k
    span_y = (ys.max() - ys.min() + 1) * k
    cx = 0.5 * (xs.min() + xs.max() + 1) * k / 2
    cy = 0.5 * (ys.min() + ys.max() + 1) * k / 2
    # mean diameter / 2 -> raw radius, / 2 again -> cube radius
    rad = (span_x + span_y) / 8.0
    return cx, cy, rad


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames", nargs="+")
    p.add_argument("-o", "--output", required=True, help="residual plot")
    p.add_argument("--cache", default=None, help="directory for cached detections")
    p.add_argument("--vmag-limit", type=float, default=4.0)
    p.add_argument("--min-alt", type=float, default=10.0)
    p.add_argument("--fwhm", type=float, default=3.5)
    p.add_argument("--threshold-sigma", type=float, default=5.0)
    p.add_argument("--max-detections", type=int, default=400)
    p.add_argument("--scan-frames", type=int, default=3,
                   help="frames used for the seed search")
    p.add_argument("--rot-step", type=float, default=1.0)
    p.add_argument("--seed-tol", type=float, default=25.0)
    p.add_argument("--seed-vmag", type=float, default=3.0,
                   help="bright-star limit for the seed search")
    p.add_argument("--seed-detections", type=int, default=80,
                   help="brightest detections used for the seed search")
    p.add_argument("--fit-detections", type=int, default=400,
                   help="brightest detections used in the fit stages")
    p.add_argument("--seed-from", default=None,
                   help="skip the grid search: "
                        "parity,xcen,ycen,rotation,hr,k3,k5 (from "
                        "lbto_wcs_groundtruth.py). A blind search over this "
                        "model is unreliable on a cold start; a few objects "
                        "identified by eye pin it in one step.")
    p.add_argument("--k3-min", type=float, default=-0.35)
    p.add_argument("--k3-max", type=float, default=0.35)
    p.add_argument("--k3-steps", type=int, default=15)
    p.add_argument("--tolerance", type=float, default=6.0)
    p.add_argument("--tolerance-start", type=float, default=40.0)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--pattern-tol", type=float, default=6.0)
    p.add_argument("--fit-k5", action="store_true", default=True)
    args = p.parse_args()

    cx, cy, rad = _circle(args.frames[0])
    print(f"image circle (cube px): centre ({cx:.0f}, {cy:.0f})  radius {rad:.0f}")

    loaded = []
    for f in sorted(args.frames):
        time, cat, det, shape = _frame(
            f, args.cache, args.fwhm, args.threshold_sigma,
            args.max_detections, args.vmag_limit, args.min_alt)
        print(f"  {Path(f).name}: {len(det)} detections, {len(cat)} catalog stars "
              f"above alt {args.min_alt:g}  @ {time.isot}")
        loaded.append((time, cat, det))
    ny, nx = shape
    print(f"cube {shape}")

    # --- seed search: parity x rotation x plate scale ------------------------
    scan = loaded[:args.scan_frames]
    rots = np.arange(0.0, 360.0, args.rot_step)
    # The illuminated circle's edge is near alt 0 but not exactly on it (the
    # MMT's lens sees slightly past the horizon), so scan generously around it.
    radii = rad * np.linspace(0.86, 1.14, 15)

    def nnframes(subset, parity, vmax, ndet=None):
        """
        (alt, az, detx, dety, kdtree) per frame, for nearest-neighbour work.

        ``ndet`` keeps only the brightest detections, and for the SEED it must
        be small. Chance coincidence scales with the detection density: at the
        ~50 px seed tolerance, 1200 detections over this 15 Mpx frame give
        ~0.6 random hits per catalog star, so the seed objective becomes pure
        noise and the search returns a different "best" geometry every run.
        The brightest ~80 cut that to ~0.03 per star, where a true solution
        stands an order of magnitude clear of chance.
        """
        out = []
        for _, cat, det in subset:
            d = det if parity > 0 else _reflect(det, nx)
            keep = np.asarray(cat["Vmag"], dtype=float) <= vmax
            dx = np.asarray(d["xcentroid"], dtype=float)
            dy = np.asarray(d["ycentroid"], dtype=float)
            if ndet is not None and len(dx) > ndet:
                order = np.argsort(-np.asarray(d["flux"], dtype=float))[:ndet]
                dx, dy = dx[order], dy[order]
            out.append((np.asarray(cat["Alt"], dtype=float)[keep],
                        np.asarray(cat["Az"], dtype=float)[keep],
                        dx, dy, cKDTree(np.column_stack([dx, dy]))))
        return out

    def chance(ndet, ncat, tol):
        return ndet * np.pi * tol ** 2 / (nx * ny) * ncat

    if args.seed_from:
        f = [float(t) for t in args.seed_from.split(",")]
        parity = int(f[0])
        seed = _params(f[1], f[2], f[3], f[4], k3=f[5], k5=f[6])
        print(f"\nseed given: parity {parity:+d}  rotation {f[3]:.3f} deg  "
              f"horizon_radius {f[4]:.1f} px  k3 {f[5]:+.4f}  k5 {f[6]:+.4f}")
        nbest = -1
    else:
        nseedcat = sum(int((np.asarray(c["Vmag"], dtype=float) <= args.seed_vmag).sum())
                       for _, c, _ in scan)
        print(f"seed search: {nseedcat} catalog stars V<={args.seed_vmag:g} over "
              f"{len(scan)} frames vs the brightest {args.seed_detections} detections; "
              f"expected chance hits {chance(args.seed_detections, nseedcat, args.seed_tol):.1f}")

        # The RADIAL TERM must be in the seed grid, not left to the fit. The
        # seed's k3=k5=0 is an equidistant projection (r proportional to zenith
        # angle); a lens of any other family departs from that by far more than
        # any usable tolerance. An equisolid-angle fisheye differs by 0.041 in
        # normalised radius at z=45 deg, which at this plate scale is 66 px -- so
        # with k3 fixed at 0 only the annulus where the two happen to agree can
        # ever match, which is exactly the ~10% hit rate seen without it.
        k3s = np.linspace(args.k3_min, args.k3_max, args.k3_steps)
        best = (0, 0, None, None)
        for parity in (+1, -1):
            sframes = nnframes(scan, parity, args.seed_vmag, args.seed_detections)
            pbest = 0
            for hr in radii:
                for k3 in k3s:
                    for rot in rots:
                        pars = _params(cx, cy, float(rot), float(hr), k3=float(k3))
                        score, n = _seed_score(sframes, pars, args.seed_tol)
                        pbest = max(pbest, n)
                        if score > best[0]:
                            best = (score, n, parity, pars)
            print(f"  parity {parity:+d}: best {pbest} bright-star hits "
                  f"over {len(scan)} frames")
        _, nbest, parity, seed = best
        if seed is None:
            raise SystemExit("no seed geometry produced any matches")
        print(f"\nseed: parity {parity:+d}  rotation {seed['rotation']:.1f} deg  "
              f"horizon_radius {seed['horizon_radius']:.0f} px  ({nbest} hits)")

    frames = ([(t, c, d) for t, c, d in loaded] if parity > 0
              else [(t, c, _reflect(d, nx)) for t, c, d in loaded])
    ncat = sum(int((np.asarray(c["Vmag"], dtype=float) <= args.vmag_limit).sum())
               for _, c, _ in loaded)
    nn = nnframes(loaded, parity, args.vmag_limit, args.fit_detections)

    # --- open the model in stages, tightening as it goes ---------------------
    # Every stage reports the chance-coincidence expectation beside the match
    # count. A stage whose matches are the same order as chance is fitting
    # noise, however healthy the count looks.
    stages = [
        (args.seed_tol, ("xcen", "ycen", "rotation")),
        (args.seed_tol, ("xcen", "ycen", "rotation", "hr")),
        (20.0, ("xcen", "ycen", "rotation", "hr", "k3")),
        (14.0, ("xcen", "ycen", "rotation", "hr", "k3", "k5")),
        (10.0, ("xcen", "ycen", "rotation", "hr", "k3", "k5", "p1", "p2")),
        (8.0, _FREE),
        (args.tolerance, _FREE),
    ]
    params = dict(seed)
    for tol, free in stages:
        pooled = _pool_nn(nn, params, tol)
        exp = chance(args.fit_detections, ncat, tol)
        if pooled is None:
            print(f"  tol {tol:5.1f}: no matches")
            continue
        alt, az, x, y = pooled
        octs = len(np.unique((az // 45).astype(int)))
        print(f"  nn tol {tol:5.1f}: {len(alt):5d} matched "
              f"(chance {exp:5.1f}), {octs}/8 octants, {len(free)} free")
        if len(alt) >= 2 * len(free):
            params = _fit_stage(alt, az, x, y, params, free)

    # Hand over to the asterism-verified matcher now the seed is good.
    for tol in np.geomspace(args.tolerance * 2.5, args.tolerance, 3):
        pooled = _pool(frames, params, float(tol), args.pattern_tol)
        if pooled is None:
            continue
        alt, az, x, y = pooled
        print(f"  asterism tol {tol:4.1f}: {len(alt):5d} matched")
        if len(alt) >= 20:
            params = _fit_stage(alt, az, x, y, params, _FREE)

    pooled = _pool(frames, params, args.tolerance, args.pattern_tol)
    if pooled is None:
        raise SystemExit("no matches survived the tightened tolerance")
    alt, az, x, y = pooled
    px, py = _predict_pixels(alt, az, xcen=params["xcen"], ycen=params["ycen"],
                             rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]),
                             horizon_radius=params["horizon_radius"],
                             tangential_coeffs=tuple(params["tangential_coeffs"]),
                             axis_tilt=tuple(params["axis_tilt"]))
    resid = np.hypot(px - x, py - y)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
    good = resid < np.median(resid) + 3.0 * 1.4826 * mad
    params = _fit_stage(alt[good], az[good], x[good], y[good], params, _FREE)
    px, py = _predict_pixels(alt[good], az[good], xcen=params["xcen"],
                             ycen=params["ycen"], rotation=params["rotation"],
                             radial_coeffs=tuple(params["radial_coeffs"]),
                             horizon_radius=params["horizon_radius"],
                             tangential_coeffs=tuple(params["tangential_coeffs"]),
                             axis_tilt=tuple(params["axis_tilt"]))
    rms = float(np.sqrt(np.mean((px - x[good]) ** 2 + (py - y[good]) ** 2)))
    available = sum(len(c) for _, c, _ in frames)

    print(f"\nparity          {parity:+d}  "
          f"({'same as MMT' if parity > 0 else 'MIRRORED vs MMT'})")
    print(f"xcen            {params['xcen']:.3f}")
    print(f"ycen            {params['ycen']:.3f}")
    print(f"rotation        {params['rotation']:.4f}")
    print(f"radial_coeffs   {tuple(round(c, 6) for c in params['radial_coeffs'])}")
    print(f"tangential      {tuple(round(c, 7) for c in params['tangential_coeffs'])}")
    print(f"axis_tilt       {tuple(round(c, 5) for c in params['axis_tilt'])}")
    print(f"horizon_radius  {params['horizon_radius']:.1f}")
    print(f"n_matched       {int(good.sum())}")
    print(f"residual_rms    {rms:.3f} px")
    print(f"matched_frac    {good.sum() / available:.3f}")
    # plate scale for context
    print(f"plate scale     {params['horizon_radius'] / 90:.2f} px/deg "
          f"(MMT: {747.2 / 90:.2f})")

    # Is the floor trailing? Over a 60 s exposure the sky turns 0.25 deg, so a
    # star's trail length is proportional to its distance from the celestial
    # pole -- zero at the pole, maximal on the equator. If the residual tracks
    # that, the limit is the exposure, not the lens model.
    lat = float(LBTO_LOCATION.lat.deg)
    sa, ca = np.radians(alt[good]), np.radians(az[good])
    sl, cl = np.radians(lat), 0.0
    # angular distance from the north celestial pole (az 0, alt = latitude)
    cospole = (np.sin(sa) * np.sin(sl) + np.cos(sa) * np.cos(sl) * np.cos(ca))
    pole = np.degrees(np.arccos(np.clip(cospole, -1, 1)))
    trail = params["horizon_radius"] / 90.0 * 0.25 * np.sin(np.radians(pole))
    r = np.hypot(px - x[good], py - y[good])
    if len(r) > 3:
        corr = float(np.corrcoef(trail, r)[0, 1])
        print(f"\ntrail model     {trail.min():.1f}-{trail.max():.1f} px "
              f"(60 s at this scale); residual vs trail length r = {corr:+.2f}")
        near = pole < 45
        if near.sum() > 2 and (~near).sum() > 2:
            print(f"  median residual  within 45 deg of pole {np.median(r[near]):.2f} px"
                  f"   beyond {np.median(r[~near]):.2f} px")

    save_alcor_residual_plot(alt[good], az[good], x[good], y[good], params,
                             args.output,
                             horizon_radius=params["horizon_radius"])
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
