"""
Median all-sky surface-brightness map over the clear, moonless archive.

One night's sky-brightness map is a picture of that night. Stacking the frames
of 141 graded-clear nights (``clear_dark_sample.py`` -> ``clear_frame_list.py``)
gives a picture of the SITE: the light domes, the airglow gradient, and the
residual optical scatter, with the weather and the Moon averaged out.

**The Milky Way is masked, not averaged.** A fixed (az, alt) samples a different
galactic latitude every hour of every night, so the plane sweeps across the
whole dome over a year and would otherwise leave a smeared ridge that looks like
site structure. Every superpixel of every frame therefore carries its own
galactic latitude and is dropped when ``|b| < --bmin`` (10 deg). Over a year's
clear nights each direction still keeps most of its samples, so the map fills
in -- the coverage map is written alongside so that can be checked rather than
assumed.

Method, and why each step is the way it is:

* **Two-stage median.** Frames within a night are correlated (same airglow, same
  aerosol, same dome soiling), so a flat median over all ~22k frames would let a
  few long clear nights dominate. Each night is reduced to its own median map
  first, then the nights are combined -- one vote per night.
* **4x4 superpixels** (``--bin``), about 0.5 deg at the zenith. Sky brightness
  structure is smooth on far larger scales, and binning by a median inside each
  block is what removes the stars: a star pushes pixels BRIGHTER only, so a
  per-block median rejects it while a mean would not (the same reason
  ``allsky_mv_best`` takes a cone median, not a darkest pixel).
* **The per-pixel geometry is computed ONCE per night, not per frame.** The
  solid-angle and alt/az grids cost a few seconds over 2M pixels; the per-frame
  work is then only bias, exposure scaling and a log. This mirrors
  ``_alcor_cone_indices`` in ``alcor/night.py`` and is what makes 22k frames
  tractable. It is also why the per-night WCS is resolved from the night's date:
  frames either side of the 2025-07-11 epoch boundary get the right geometry.
* Output is the **raw frame with the raw-frame ARC WCS**, like every other alcor
  product, so it overlays pixel-for-pixel on the extinction and sky-brightness
  maps. The superpixel map is expanded back by nearest neighbour rather than the
  WCS being re-scaled -- binning a SIP WCS correctly means rescaling every
  coefficient, and a blocky-but-exact astrometric solution beats a smooth and
  subtly wrong one.

Usage:
  sky_median_map.py <frames.csv.gz> <archive-dir> -o OUT.fits [--stride 5]
                    [--bin 4] [--bmin 10] [--workers N] [--nights N]

``<archive-dir>`` holds the raw ``<night>/<frame>.fits.bz2`` tree. Nothing from
the products tree is needed: the frame list carries the times and exposures.
"""

import argparse
import os
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, SkyCoord
from astropy.io import fits
from astropy.time import Time
import astropy.units as u

from skycam_utils.alcor import (
    ALCOR_CALIB_EXPTIME,
    ALCOR_FIELD_RADIUS,
    alcor_calibration,
    alcor_zeropoint,
    build_alcor_wcs,
    load_alcor_fits,
    load_alcor_badpix_mask,
    load_alcor_horizon_mask,
)
from skycam_utils.alcor.io import _corner_bias
from skycam_utils.alcor.skybright import _alcor_pixel_solid_angle
from skycam_utils.astrometry import MMT_LOCATION

# North galactic pole, ICRS.
RA_NGP = 192.85948
DEC_NGP = 27.12825

_GEOM = {}   # per-worker cache, keyed by night: the precomputed night geometry

CAL_KEYS = ("xcen", "ycen", "rotation", "radial_coeffs",
            "tangential_coeffs", "axis_tilt", "horizon_radius")


def galactic_b(ra_deg, dec_deg):
    """Galactic latitude in degrees, closed form (no SkyCoord per frame)."""
    ra = np.radians(ra_deg - RA_NGP)
    dec = np.radians(dec_deg)
    dg = np.radians(DEC_NGP)
    s = np.sin(dec) * np.sin(dg) + np.cos(dec) * np.cos(dg) * np.cos(ra)
    return np.degrees(np.arcsin(np.clip(s, -1.0, 1.0)))


def _check_galactic_b():
    """Pin the closed form against astropy once, so the mask cannot be silently wrong."""
    rng = np.random.default_rng(0)
    ra = rng.uniform(0, 360, 500)
    dec = rng.uniform(-30, 89, 500)
    ref = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs").galactic.b.deg
    err = np.max(np.abs(ref - galactic_b(ra, dec)))
    if err > 0.02:
        raise SystemExit(f"galactic_b disagrees with astropy by {err:.4f} deg")
    return err


def _night_geometry(night, nbin, horizon_mask, ref_time, shape, badpix=True):
    """
    Everything that depends on the night but not on the frame.

    Returns the solid-angle-derived constants plus, for each VALID superpixel,
    its flat index and the hour angle / declination of its centre. Hour angle
    and declination are fixed for a fixed (az, alt) at a fixed site, so the
    galactic latitude of a superpixel at any later time follows from the local
    sidereal time alone -- one vectorised rotation per frame instead of a
    coordinate transform over 125k directions.
    """
    cal = alcor_calibration(Time(night))
    wcs = build_alcor_wcs(**{k: cal[k] for k in CAL_KEYS if k in cal})

    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    az, alt = wcs.pixel_to_world_values(xx.astype(float), yy.astype(float))
    omega = _alcor_pixel_solid_angle(az, alt)

    blank = ~np.isfinite(alt) | ~np.isfinite(omega) | (omega <= 0)
    # Outside the illuminated image circle the signal is near zero and the map
    # reports a confident-looking ~25 mag/arcsec^2 that is pure artifact.
    crpix = wcs.wcs.crpix
    r = np.hypot(xx - (crpix[0] - 1.0), yy - (crpix[1] - 1.0))
    blank |= r > ALCOR_FIELD_RADIUS
    if horizon_mask:
        hmask, _ = load_alcor_horizon_mask(Time(night))
        blank |= hmask
    # Hot pixels are EXCLUDED here rather than repaired per frame. Repair is
    # ~1.9 s of the ~2.4 s each frame costs, and masking is both cheaper and
    # more honest: a repaired pixel is an interpolated guess that still enters
    # the block median, while a masked one simply does not vote. It matters
    # because the hot-pixel pattern is FIXED -- left in, it biases the same
    # superpixels on every night and so survives the across-night median (over
    # one night it shifts ~1700 of 1.2M superpixels by up to 0.07 mag).
    if badpix:
        bmask, _ = load_alcor_badpix_mask(Time(night))
        blank |= bmask[1]                       # G channel is the one measured

    nby = (ny // nbin) * nbin
    nbx = (nx // nbin) * nbin
    zp_g = alcor_zeropoint(Time(night))["g"]["zp"]

    # Superpixel centres, from the unblanked pixels only.
    def _blocks(a):
        return a[:nby, :nbx].reshape(nby // nbin, nbin, nbx // nbin, nbin)

    def _bin_mean(a, valid):
        return np.nanmean(_blocks(np.where(valid, a, np.nan)), axis=(1, 3))

    good = ~blank
    frac = _blocks(good).mean(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        # Averaged as a unit vector: a plain mean of azimuth breaks at the
        # 0/360 wrap, which is exactly where the Tucson dome sits.
        sazc = _bin_mean(np.cos(np.radians(az)), good)
        sazs = _bin_mean(np.sin(np.radians(az)), good)
        salt = _bin_mean(alt, good)
    saz = np.degrees(np.arctan2(sazs, sazc)) % 360.0

    ok = (frac > 0.5) & np.isfinite(salt) & np.isfinite(saz)
    idx = np.flatnonzero(ok.ravel())

    t0 = Time(ref_time, scale="utc", location=MMT_LOCATION)
    altaz = AltAz(az=saz.ravel()[idx] * u.deg, alt=salt.ravel()[idx] * u.deg,
                  obstime=t0, location=MMT_LOCATION)
    icrs = SkyCoord(altaz).icrs
    lst0 = t0.sidereal_time("apparent").deg
    ha = (lst0 - icrs.ra.deg) % 360.0
    dec = icrs.dec.deg

    return dict(wcs=wcs, omega=omega, blank=blank, zp_g=zp_g, shape=(ny, nx),
                nby=nby, nbx=nbx, nbin=nbin, sgrid=ok.shape, idx=idx,
                ha=ha, dec=dec)


def _init(nbin, horizon_mask, archive, bmin, shape, badpix):
    global _CFG
    _CFG = dict(nbin=nbin, horizon_mask=horizon_mask, archive=archive,
                bmin=bmin, shape=shape, badpix=badpix)


def reduce_night(args):
    """Median superpixel map for one night, Milky Way masked out."""
    night, rows = args
    cfg = _CFG
    nbin = cfg["nbin"]

    if night not in _GEOM:
        _GEOM.clear()          # one night in flight per worker
        _GEOM[night] = _night_geometry(night, nbin, cfg["horizon_mask"],
                                       rows["OBSTIME"].iloc[0], cfg["shape"],
                                       cfg["badpix"])
    g = _GEOM[night]
    nby, nbx, idx = g["nby"], g["nbx"], g["idx"]
    nsuper = g["sgrid"][0] * g["sgrid"][1]

    times = Time(pd.to_datetime(rows["OBSTIME"]).to_numpy(), scale="utc",
                 location=MMT_LOCATION)
    lst = np.atleast_1d(times.sidereal_time("apparent").deg)

    stack = np.full((len(rows), len(idx)), np.nan, dtype=np.float32)
    nread = 0
    for i, (fname, expt) in enumerate(zip(rows["filename"], rows["exposure"])):
        path = os.path.join(cfg["archive"], night, fname)
        if not os.path.exists(path):
            continue
        try:
            cube, _, _ = load_alcor_fits(path, wcs=g["wcs"], badpix=None)
        except Exception:
            continue
        if cube.shape[1:] != cfg["shape"]:
            raise SystemExit(f"{path}: shape {cube.shape[1:]} != {cfg['shape']}")
        nread += 1

        g_raw = np.asarray(cube[1], dtype=float)
        g20 = (g_raw - _corner_bias(cube)[1]) * (ALCOR_CALIB_EXPTIME / float(expt))
        with np.errstate(divide="ignore", invalid="ignore"):
            surf = g20 / g["omega"]
            mu = np.where(surf > 0, -2.5 * np.log10(surf) + g["zp_g"], np.nan)
        mu[g["blank"]] = np.nan

        # Median inside each 4x4 block: a star only pushes pixels brighter, so
        # the block median rejects it where a mean would smear it in.
        blocks = mu[:nby, :nbx].reshape(nby // nbin, nbin, nbx // nbin, nbin)
        with np.errstate(invalid="ignore"):
            sp = np.nanmedian(blocks, axis=(1, 3)).ravel()[idx]

        b = galactic_b((lst[i] - g["ha"]) % 360.0, g["dec"])
        sp[np.abs(b) < cfg["bmin"]] = np.nan
        stack[i] = sp

    if nread == 0:
        return night, None, None, 0

    with np.errstate(invalid="ignore"):
        med = np.nanmedian(stack, axis=0)
    cov = np.sum(np.isfinite(stack), axis=0)

    full = np.full(nsuper, np.nan, dtype=np.float32)
    full[idx] = med
    fullcov = np.zeros(nsuper, dtype=np.int32)
    fullcov[idx] = cov
    return night, full.reshape(g["sgrid"]), fullcov.reshape(g["sgrid"]), nread


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames", help="clear_dark_frames.csv.gz from clear_frame_list.py")
    p.add_argument("archive", help="raw archive root (<night>/<frame>.fits.bz2)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--stride", type=int, default=5,
                   help="use every Nth frame of each night (default 5)")
    p.add_argument("--bin", type=int, default=4, dest="nbin",
                   help="superpixel size in raw pixels (default 4, ~0.5 deg)")
    p.add_argument("--bmin", type=float, default=10.0,
                   help="mask |galactic latitude| below this, in deg (default 10)")
    p.add_argument("--no-horizon-mask", action="store_true")
    p.add_argument("--no-badpix-mask", action="store_true",
                   help="keep hot pixels instead of masking them out")
    p.add_argument("--nights", type=int, default=None, help="first N nights only (testing)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    err = _check_galactic_b()
    print(f"galactic_b vs astropy: max {err:.4f} deg", file=sys.stderr)

    frames = pd.read_csv(args.frames)
    groups = [(n, r.iloc[::args.stride].reset_index(drop=True))
              for n, r in frames.groupby("night", sort=True)]
    if args.nights:
        groups = groups[:args.nights]
    total = sum(len(r) for _, r in groups)
    print(f"{len(groups)} nights, {total:,} frames at stride {args.stride}",
          file=sys.stderr)

    # Every geometry grid is precomputed per night, so the frame shape has to be
    # known before any worker starts. Take it from the data rather than a
    # constant: the sensor is not square (1411 x 1422) and guessing it wrong
    # fails only later, inside a worker.
    probe = os.path.join(args.archive, groups[0][0], groups[0][1]["filename"].iloc[0])
    shape = load_alcor_fits(probe, badpix=None)[0].shape[1:]
    print(f"frame shape {shape[0]} x {shape[1]} (from {os.path.basename(probe)})",
          file=sys.stderr)

    maps, covs, nights_used = [], [], []
    t0 = _time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(args.nbin, not args.no_horizon_mask,
                                       args.archive, args.bmin, shape,
                                       not args.no_badpix_mask)) as ex:
        futs = [ex.submit(reduce_night, g) for g in groups]
        for f in as_completed(futs):
            night, med, cov, nread = f.result()
            done += 1
            if med is None:
                print(f"[{done}/{len(groups)}] {night}: no frames read", file=sys.stderr)
                continue
            maps.append(med)
            covs.append(cov)
            nights_used.append(night)
            el = _time.time() - t0
            print(f"[{done}/{len(groups)}] {night}: {nread} frames  "
                  f"({el/done:.1f}s/night, {el/60:.1f} min elapsed)", file=sys.stderr)

    if not maps:
        raise SystemExit("no nights reduced")

    cube = np.stack(maps)
    with np.errstate(invalid="ignore"):
        sky = np.nanmedian(cube, axis=0)
    nnights = np.sum(np.isfinite(cube), axis=0).astype(np.int32)
    nframes = np.sum(np.stack(covs), axis=0).astype(np.int32)

    # Expand superpixels back to the raw frame; pad the remainder the binning
    # could not cover so the original WCS still applies unchanged.
    ny, nx = shape
    def _expand(a, dtype):
        big = np.repeat(np.repeat(a, args.nbin, axis=0), args.nbin, axis=1)
        out = np.full((ny, nx), np.nan if dtype == np.float32 else 0, dtype=dtype)
        out[:big.shape[0], :big.shape[1]] = big
        return out

    cal = alcor_calibration(Time(sorted(nights_used)[len(nights_used) // 2]))
    wcs = build_alcor_wcs(**{k: cal[k] for k in CAL_KEYS if k in cal})
    hdr = wcs.to_header(relax=True)
    hdr["BUNIT"] = ("mag/arcsec2", "observed V surface brightness")
    hdr["NNIGHTS"] = (len(nights_used), "clear dark nights combined")
    hdr["NFRAMES"] = (int(total), "frames requested at STRIDE")
    hdr["STRIDE"] = (args.stride, "every Nth frame of each night")
    hdr["SPBIN"] = (args.nbin, "superpixel size, raw pixels")
    hdr["BMIN"] = (args.bmin, "|galactic latitude| masked below this, deg")
    hdr["HORIZMSK"] = (not args.no_horizon_mask, "horizon mask applied")
    hdr["BADPIX"] = (not args.no_badpix_mask, "hot pixels masked out")
    hdr["COMMENT"] = "Median over nights of each night's median frame map."
    hdr["COMMENT"] = "Milky Way masked per frame, not averaged: see BMIN."

    hdus = [fits.PrimaryHDU(_expand(sky, np.float32), header=hdr),
            fits.ImageHDU(_expand(nnights, np.int32), name="NNIGHTS"),
            fits.ImageHDU(_expand(nframes, np.int32), name="NFRAMES"),
            fits.BinTableHDU.from_columns(
                [fits.Column(name="NIGHT", format="10A", array=np.array(nights_used))],
                name="NIGHTS")]
    fits.HDUList(hdus).writeto(args.output, overwrite=args.overwrite)

    finite = np.isfinite(sky)
    print(f"\nwrote {args.output}", file=sys.stderr)
    print(f"  {len(nights_used)} nights, {finite.sum():,} of {sky.size:,} "
          f"superpixels filled ({100*finite.mean():.1f}%)", file=sys.stderr)
    if finite.any():
        print(f"  per-superpixel night coverage: min {nnights[finite].min()}, "
              f"median {int(np.median(nnights[finite]))}", file=sys.stderr)
        print(f"  brightness range: {np.nanmin(sky):.2f} to {np.nanmax(sky):.2f} "
              f"mag/arcsec2", file=sys.stderr)


if __name__ == "__main__":
    main()
