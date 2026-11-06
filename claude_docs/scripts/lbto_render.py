"""
Colour all-sky renders of the LBTO/VATT OMEA frames with an alt/az grid.

Demonstration figures for the measured geometry: the fitted WCS is used to
draw the altitude/azimuth grid and to place the labels, so every line and
every marker in the output is a prediction. If the calibration were wrong the
labels would miss their stars, which makes this the figure that shows the
calibration works rather than merely asserting it.

Three things here are not reusable from the MMT code and are done locally:

* ``_add_alcor_alt_az_grid`` is FIGURE-scoped, so it cannot serve a multi-panel
  figure; the grid is drawn per-axes in data coordinates instead.
* the MMT display stretch is tuned for its own sensor, so the per-channel
  scaling is re-tuned here.
* the frame must be mirrored in x before the fitted geometry applies, because
  the fit was done against mirrored detections (LBTO's image parity is the
  opposite of the MMT's).

Usage: lbto_render.py <frame.fits.fz> -o OUT.png [--labels] [--grid]
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
from astropy.visualization import ZScaleInterval

from skycam_utils.alcor import (
    ALCOR_HUMIDITY, ALCOR_OBSWL, ALCOR_PRESSURE, ALCOR_TEMPERATURE,
    alcor_named_reference_altaz,
)
from skycam_utils.alcor.wcs import _predict_pixels

from lbto_wcs_spike import LBTO_LOCATION, _raw_to_cube

# The fitted 2026-11-04 geometry (claude_docs/scripts/lbto_wcs_spike.py).
# Applies to the 2x2-binned cube AFTER mirroring in x.
LBTO_GEOMETRY = dict(
    xcen=2334.130, ycen=1611.948, rotation=171.6477,
    radial_coeffs=(1.0, 0.119277, 0.016480),
    tangential_coeffs=(0.000581, 0.0006389),
    axis_tilt=(3.02652, -3.67126),
    horizon_radius=1825.6,
)

PLANETS = ("mars", "saturn", "jupiter", "venus")

# Per-channel gains for the display stretch. The MMT's values are tuned to its
# own sensor and colour balance; these are set for the ASI6200MC.
GSCALE, BSCALE = 1.00, 1.22


def predict(alt, az, geom=LBTO_GEOMETRY):
    return _predict_pixels(np.atleast_1d(alt), np.atleast_1d(az),
                           xcen=geom["xcen"], ycen=geom["ycen"],
                           rotation=geom["rotation"],
                           radial_coeffs=tuple(geom["radial_coeffs"]),
                           horizon_radius=geom["horizon_radius"],
                           tangential_coeffs=tuple(geom["tangential_coeffs"]),
                           axis_tilt=tuple(geom["axis_tilt"]))


def display_rgb(cube, power=0.55, contrast=0.12, balance=True):
    """
    Corner-bias-subtracted, white-balanced, power + ZScale stretch.

    The white balance is grey-world over the illuminated sky rather than the
    MMT's hand-tuned per-channel gains: those gains are a property of that
    sensor's colour response, and carrying them over turns this sensor's sky
    green. Grey-world needs nothing tuned and is honest about the fact that
    the absolute colour of the night sky here is not calibrated.
    """
    cube = np.asarray(cube, dtype=np.float32)
    bias = np.array([np.median(np.concatenate([
        c[:20, :20].ravel(), c[:20, -20:].ravel(),
        c[-20:, :20].ravel(), c[-20:, -20:].ravel()])) for c in cube])
    rgb = cube - bias[:, None, None]
    if balance:
        sky = rgb[1] > np.percentile(rgb[1], 60)       # the illuminated disc
        med = np.array([np.median(c[sky]) for c in rgb])
        rgb = rgb * (med.mean() / np.maximum(med, 1e-6))[:, None, None]
    else:
        rgb[1] *= GSCALE
        rgb[2] *= BSCALE
    lo, hi = ZScaleInterval(contrast=contrast).get_limits(rgb[1])
    out = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** power
    return np.moveaxis(out, 0, -1)


def displayer(nx, ny, north_up):
    """
    Map fitted-frame pixels to display pixels.

    The fitted geometry lives in the camera's own frame, where north falls at
    DECREASING y. Rotating the view by 180 degrees puts north up and east
    left, matching how the MMT's all-sky frames are presented. It is a
    rotation, not a mirror, so the image keeps the handedness an all-sky
    camera actually has; only the presentation changes, never the WCS.
    """
    if not north_up:
        return lambda x, y: (x, y)
    return lambda x, y: (nx - 1.0 - np.asarray(x), ny - 1.0 - np.asarray(y))


def alt_az_grid(ax, nx, ny, disp, alts=(0, 15, 30, 45, 60, 75),
                azs=range(0, 360, 30), color="0.75", lw=0.7, fontsize=7.5):
    """Draw the grid in data coordinates from the fitted geometry."""
    fine = np.arange(0, 360.25, 0.5)
    for alt in alts:
        x, y = disp(*predict(np.full_like(fine, float(alt)), fine))
        ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        seg = np.where(ok, 1, 0)
        breaks = np.flatnonzero(np.diff(seg) != 0) + 1
        for part in np.split(np.arange(len(fine)), breaks):
            if len(part) > 1 and ok[part[0]]:
                ax.plot(x[part], y[part], "-", color=color, lw=lw, alpha=0.8)
    ralt = np.arange(0, 88.5, 0.5)
    for az in azs:
        x, y = disp(*predict(ralt, np.full_like(ralt, float(az))))
        ok = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
        if ok.sum() > 1:
            ax.plot(x[ok], y[ok], "-", color=color, lw=lw, alpha=0.8)
    # altitude labels along azimuth 90, azimuth labels just inside alt 0
    for alt in alts:
        if alt == 0:
            continue
        x, y = disp(*predict(np.array([float(alt)]), np.array([90.0])))
        if np.isfinite(x[0]):
            ax.text(x[0], y[0], f"{alt}°", color=color, fontsize=fontsize,
                    ha="center", va="center",
                    bbox=dict(fc="black", ec="none", alpha=0.45, pad=0.9))
    for az, lab in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        x, y = disp(*predict(np.array([6.0]), np.array([float(az)])))
        if np.isfinite(x[0]):
            ax.text(x[0], y[0], lab, color="white", fontsize=fontsize + 3.5,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(fc="black", ec="none", alpha=0.5, pad=1.4))


def objects(time, vmag_limit=2.2, min_alt=12.0):
    """(name, alt, az) for bright catalog stars plus the naked-eye planets."""
    out = []
    cat = alcor_named_reference_altaz(time, vmag_limit=vmag_limit,
                                      min_alt=min_alt, location=LBTO_LOCATION)
    for row in cat:
        out.append((str(row["NAME"]).strip(), float(row["Alt"]), float(row["Az"])))
    frame = AltAz(obstime=time, location=LBTO_LOCATION, pressure=ALCOR_PRESSURE,
                  temperature=ALCOR_TEMPERATURE, relative_humidity=ALCOR_HUMIDITY,
                  obswl=ALCOR_OBSWL)
    for name in PLANETS:
        aa = get_body(name, time, LBTO_LOCATION).transform_to(frame)
        if float(aa.alt.deg) > min_alt:
            out.append((name.capitalize(), float(aa.alt.deg), float(aa.az.deg)))
    # Polaris is a Cepheid and so is absent from the calibration catalog.
    aa = SkyCoord(ra=37.954561 * u.deg, dec=89.264109 * u.deg,
                  frame="icrs").transform_to(frame)
    out.append(("Polaris", float(aa.alt.deg), float(aa.az.deg)))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frame")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--vmag-limit", type=float, default=2.2)
    p.add_argument("--min-alt", type=float, default=12.0)
    p.add_argument("--no-grid", action="store_true")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--no-north-up", action="store_true",
                   help="keep the camera's native orientation (north at -y) "
                        "instead of rotating 180 deg to north-up, east-left")
    p.add_argument("--crop", nargs=4, type=int, default=None,
                   metavar=("X0", "X1", "Y0", "Y1"),
                   help="zoom on a cube-pixel box, to show the fit up close")
    p.add_argument("--figsize", nargs=2, type=float, default=(13.0, 8.9))
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    hdr = fits.getheader(args.frame, 1)
    exptime = float(hdr["EXPOSURE"])
    time = Time(hdr["DATE-OBS"], format="isot", scale="utc") + exptime / 2 * u.s
    with fits.open(args.frame) as hdul:
        cube = _raw_to_cube(hdul[1].data)
    cube = cube[:, :, ::-1]            # the fitted geometry is for the mirrored frame
    ny, nx = cube.shape[1:]
    rgb = display_rgb(cube)
    north_up = not args.no_north_up
    disp = displayer(nx, ny, north_up)
    if north_up:
        rgb = rgb[::-1, ::-1]          # 180 deg: north up, east left

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    if not args.no_grid:
        alt_az_grid(ax, nx, ny, disp)
    zoom = args.crop
    if not args.no_labels:
        n = 0
        for name, alt, az in objects(time, args.vmag_limit, args.min_alt):
            x, y = disp(*predict(np.array([alt]), np.array([az])))
            if not np.isfinite(x[0]) or not (0 <= x[0] < nx and 0 <= y[0] < ny):
                continue
            ax.plot(x[0], y[0], "o", mfc="none", mec="#ffdd33", ms=13, mew=1.3)
            ax.annotate(name, (x[0], y[0]), xytext=(11, 8),
                        textcoords="offset points", color="#ffdd33",
                        fontsize=8.5,
                        path_effects=None)
            n += 1
        print(f"labelled {n} objects")

    if zoom:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    else:
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
    ax.set_aspect("equal")
    orient = ("displayed north-up / east-left (frame rotated 180°)"
              if north_up else "camera's native orientation")
    ax.set_xlabel(f"x [2x2-binned px, {'rotated' if north_up else 'native'} frame]")
    ax.set_ylabel(f"y [2x2-binned px, {'rotated' if north_up else 'native'} frame]")
    ax.set_title(
        f"LBTO/VATT all-sky  {time.isot} UT   {exptime:.0f} s   "
        f"{hdr.get('INSTRUME', '')}\n"
        f"grid and labels placed by the fitted WCS (0.36 px RMS); {orient}",
        fontsize=10.5)
    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
