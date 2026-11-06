"""
Solve the LBTO/VATT OMEA geometry from hand-identified objects.

``fit_alcor_wcs`` always warm-starts from a known ``ALCOR_CALIBRATIONS`` epoch,
so its asterism-verified matcher never has to bootstrap itself. A new
instrument has no epoch, and a blind search over (parity, rotation, plate
scale, radial term) proved unreliable on these frames: the match objective is
dominated by chance coincidence long before the grid is fine enough, and it
converges reproducibly onto aliases that put Sirius 107 px from its own image.

A handful of objects identified by eye removes the problem entirely -- three
over-determine the centre, rotation and plate scale, and the rest check the
radial term and settle parity by measurement rather than by search. This is the
bootstrap path any new camera needs; it is not LBTO-specific.

Pixel coordinates are RAW-mosaic (the 9576 x 6388 image in extension 1), as
DS9 reports them. Conversion to the 2x2-binned cube frame the pipeline works
in: a cube pixel j covers raw pixels 2j and 2j+1, so its centre sits at raw
2j + 0.5, giving cube = (raw0 - 0.5) / 2 with raw0 the 0-based raw coordinate.

Usage: lbto_wcs_groundtruth.py <frame.fits.fz> [-o OUT.png]
"""

import argparse

import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import AltAz, SkyCoord, get_body
from astropy.io import fits
from astropy.time import Time
from scipy.optimize import least_squares

from skycam_utils.alcor import (
    ALCOR_HUMIDITY, ALCOR_OBSWL, ALCOR_PRESSURE, ALCOR_TEMPERATURE,
)
from skycam_utils.alcor.wcs import _predict_pixels

from lbto_wcs_spike import LBTO_LOCATION

# Objects identified by eye on allsky-2026-11-04-08-19-20-utc.fits.fz, as
# 1-based DS9 coordinates on the RAW mosaic. Planets are resolved from the
# ephemeris; stars from SIMBAD-style names via their ICRS positions, since
# Polaris (a Cepheid) and the planets are absent from the calibration catalog
# by construction.
GROUND_TRUTH = {
    "sirius":    (3273, 5283),
    "mars":      (1925, 3107),
    "aldebaran": (4449, 4051),
    "capella":   (4108, 2848),
    "polaris":   (4516, 1165),
    "saturn":    (6891, 3899),
    "markab":    (7284, 2771),
}

PLANETS = ("mars", "saturn")

# ICRS positions (J2000) of the named stars, so the solver does not depend on
# which catalog happens to carry which name.
STARS = {
    "sirius":    (101.287155, -16.716116),
    "aldebaran": (68.980163, 16.509302),
    "capella":   (79.172328, 45.997991),
    "polaris":   (37.954561, 89.264109),
    "markab":    (346.190224, 15.205267),
}

PARAMS = ("xcen", "ycen", "rotation", "hr", "k3", "k5", "p1", "p2", "tn", "te")


def _altaz(name, time):
    frame = AltAz(obstime=time, location=LBTO_LOCATION, pressure=ALCOR_PRESSURE,
                  temperature=ALCOR_TEMPERATURE, relative_humidity=ALCOR_HUMIDITY,
                  obswl=ALCOR_OBSWL)
    if name in PLANETS:
        body = get_body(name, time, LBTO_LOCATION)
    else:
        ra, dec = STARS[name]
        body = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    aa = body.transform_to(frame)
    return float(aa.alt.deg), float(aa.az.deg)


def _unflat(v):
    return dict(xcen=v["xcen"], ycen=v["ycen"], rotation=v["rotation"],
                radial_coeffs=(1.0, v["k3"], v["k5"]),
                tangential_coeffs=(v["p1"], v["p2"]),
                axis_tilt=(v["tn"], v["te"]), horizon_radius=v["hr"])


def _predict(alt, az, v, parity, nx):
    p = _unflat(v)
    x, y = _predict_pixels(alt, az, xcen=p["xcen"], ycen=p["ycen"],
                           rotation=p["rotation"],
                           radial_coeffs=tuple(p["radial_coeffs"]),
                           horizon_radius=p["horizon_radius"],
                           tangential_coeffs=tuple(p["tangential_coeffs"]),
                           axis_tilt=tuple(p["axis_tilt"]))
    return (x if parity > 0 else nx - 1.0 - x), y


# A NEGATIVE horizon_radius mirrors the radial direction, which is the same
# map as a parity flip plus a 180 deg rotation -- already covered by the
# parity and rotation parameters. Left unbounded the solver escapes through
# it and reports a nonsense plate scale, so every parameter that has a
# physical range is bounded.
BOUNDS = {"xcen": (0.0, 5000.0), "ycen": (0.0, 3500.0),
          "rotation": (-720.0, 720.0), "hr": (1000.0, 2500.0),
          "k3": (-0.6, 0.6), "k5": (-0.6, 0.6),
          "p1": (-0.02, 0.02), "p2": (-0.02, 0.02),
          "tn": (-5.0, 5.0), "te": (-5.0, 5.0)}


def _solve(alt, az, ox, oy, free, start, parity, nx):
    names = [n for n in PARAMS if n in free]
    scale = {"xcen": 20.0, "ycen": 20.0, "rotation": 2.0, "hr": 20.0,
             "k3": 0.05, "k5": 0.05, "p1": 1e-3, "p2": 1e-3,
             "tn": 0.2, "te": 0.2}

    def resid(q):
        v = dict(start)
        v.update(zip(names, q))
        x, y = _predict(alt, az, v, parity, nx)
        r = np.concatenate([x - ox, y - oy])
        return np.where(np.isfinite(r), r, 1e4)

    p0 = np.array([np.clip(start[n], *BOUNDS[n]) for n in names], dtype=float)
    out = least_squares(resid, p0, x_scale=[scale[n] for n in names],
                        bounds=(np.array([BOUNDS[n][0] for n in names]),
                                np.array([BOUNDS[n][1] for n in names])),
                        max_nfev=20000)
    v = dict(start)
    v.update(zip(names, out.x))
    # per-object rms, so it is comparable with the residual table below
    x, y = _predict(alt, az, v, parity, nx)
    return v, float(np.sqrt(np.mean((x - ox) ** 2 + (y - oy) ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frame")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--ds9", action="store_true", default=True,
                   help="ground-truth coordinates are 1-based DS9 values")
    args = p.parse_args()

    hdr = fits.getheader(args.frame, 1)
    exptime = float(hdr["EXPOSURE"])
    time = Time(hdr["DATE-OBS"], format="isot", scale="utc") + exptime / 2 * u.s
    nx_raw, ny_raw = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    nx, ny = nx_raw // 2, ny_raw // 2
    print(f"{args.frame}")
    print(f"  mid-exposure {time.isot} UT   raw {nx_raw} x {ny_raw} -> cube {nx} x {ny}")

    names = list(GROUND_TRUTH)
    alt = np.array([_altaz(n, time)[0] for n in names])
    az = np.array([_altaz(n, time)[1] for n in names])
    raw = np.array([GROUND_TRUTH[n] for n in names], dtype=float)
    if args.ds9:
        raw -= 1.0
    ox, oy = (raw[:, 0] - 0.5) / 2.0, (raw[:, 1] - 0.5) / 2.0

    print("\n  object       az      alt    raw x   raw y   cube x  cube y")
    for n, a, z, r, cxp, cyp in zip(names, alt, az, raw, ox, oy):
        print(f"  {n:<11s} {z:6.2f} {a:7.2f}  {r[0]:7.1f} {r[1]:7.1f}  "
              f"{cxp:7.1f} {cyp:7.1f}")

    # Centre of the illuminated circle as the starting guess.
    cx0, cy0 = nx_raw / 2 / 2, ny_raw / 2 / 2
    hr0 = 1595.0

    best = None
    for parity in (+1, -1):
        for rot0 in np.arange(0.0, 360.0, 10.0):
            start = dict(xcen=cx0, ycen=cy0, rotation=float(rot0), hr=hr0,
                         k3=0.0, k5=0.0, p1=0.0, p2=0.0, tn=0.0, te=0.0)
            v, rms = _solve(alt, az, ox, oy,
                            ("xcen", "ycen", "rotation", "hr"), start, parity, nx)
            if best is None or rms < best[0]:
                best = (rms, parity, v)
    rms, parity, v = best
    print(f"\n4-parameter solve: parity {parity:+d}  rms {rms:.2f} px  "
          f"rotation {v['rotation'] % 360:.3f}  hr {v['hr']:.1f}")

    for free in (("xcen", "ycen", "rotation", "hr", "k3"),
                 ("xcen", "ycen", "rotation", "hr", "k3", "k5")):
        v, rms = _solve(alt, az, ox, oy, free, v, parity, nx)
        print(f"{len(free)}-parameter solve: rms {rms:.2f} px  "
              f"rotation {v['rotation'] % 360:.3f}  hr {v['hr']:.1f}  "
              f"k3 {v['k3']:+.4f}  k5 {v['k5']:+.4f}")

    v["rotation"] = v["rotation"] % 360.0
    px, py = _predict(alt, az, v, parity, nx)
    res = np.hypot(px - ox, py - oy)
    print("\n  object        pred x   pred y    dx     dy   resid")
    for n, a, b, c, d, r in zip(names, px, py, px - ox, py - oy, res):
        print(f"  {n:<11s} {a:8.1f} {b:8.1f} {c:6.1f} {d:6.1f} {r:7.2f}")
    print(f"\n  rms {np.sqrt(np.mean(res ** 2)):.2f} px   max {res.max():.2f} px")
    print(f"  plate scale {v['hr'] / 90:.2f} cube px/deg "
          f"({v['hr'] / 90 / 2:.2f} per raw px; MMT {747.2 / 90:.2f})")

    print("\n  seed for the full fit:")
    print(f"    parity={parity:+d}  xcen={v['xcen']:.3f}  ycen={v['ycen']:.3f}")
    print(f"    rotation={v['rotation']:.4f}  horizon_radius={v['hr']:.2f}")
    print(f"    radial_coeffs=(1.0, {v['k3']:.6f}, {v['k5']:.6f})")

    if args.output:
        fig, ax = plt.subplots(figsize=(8.2, 6.0))
        ax.scatter(ox, oy, s=70, facecolor="none", edgecolor="#1f77b4",
                   label="identified")
        ax.scatter(px, py, s=25, color="#d62728", label="predicted")
        for n, a, b in zip(names, ox, oy):
            ax.annotate(n, (a, b), fontsize=8, xytext=(6, 5),
                        textcoords="offset points")
        ax.add_patch(plt.Circle((v["xcen"], v["ycen"]), v["hr"], fill=False,
                                ec="0.6", ls="--", lw=1.0))
        ax.plot([v["xcen"]], [v["ycen"]], "k+", ms=12)
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect("equal")
        ax.set_xlabel("cube x [px]")
        ax.set_ylabel("cube y [px]")
        ax.set_title(f"LBTO geometry from {len(names)} identified objects -- "
                     f"rms {np.sqrt(np.mean(res ** 2)):.2f} px", fontsize=10.5)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(args.output, dpi=140)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
