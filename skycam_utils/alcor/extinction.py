"""
Cloud-extinction maps built from fixed-position star photometry.

Every frame measures ``ext_g = calibrated mag - catalog mag`` for a few hundred
bright stars at known ``(az, alt)``: a sparse, irregular sampling of the
line-of-sight extinction across the dome. This module averages a short block of
frames, kernel-smooths the surviving measurements onto the sky, and writes the
result in the camera's raw pixel frame with the alt/az ARC WCS attached -- the
same geometry as :func:`alcor_sky_brightness_fits` and :func:`alcor_proc_fits`,
so a map overlays pixel-for-pixel on every other product and a pointing is
looked up with a single ``world_to_pixel`` call.

Smoothing is done on a regular 1-degree alt/az grid and then resampled to the
raw frame, never directly onto the raw pixels. That is a performance
requirement, not a tidiness one: the raw frame is ~2 million pixels and a block
carries ~350 stars, so a direct kernel average is ~700 million great-circle
distances per map -- far too slow for a five-minute cadence -- while the same
kernel on a 1-degree grid is ~11 million and runs in milliseconds. An 8-degree
kernel is not resolved any better by a finer grid, so nothing is lost.

Non-detections carry the strongest signal and have no extinction value at all
(``flux = 0`` -> ``mag`` NaN -> ``ext`` NaN), so they would silently vanish
exactly where extinction is highest. They are tracked separately, recorded in
the file's ``STARS`` table with ``detected = False``, and drawn as open markers:
each is a **lower limit**, not a missing measurement.
"""

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from astropy.io import fits
from astropy.time import Time

from scipy.ndimage import map_coordinates

from .config import (
    ALCOR_EXT_CADENCE, ALCOR_EXT_FRAME_SHAPE, ALCOR_EXT_GRID_STEP,
    ALCOR_EXT_KERNEL_CUT, ALCOR_EXT_KERNEL_SIGMA, ALCOR_EXT_LOOKUP_RADIUS,
    ALCOR_EXT_LOST_MIN_FRAMES, ALCOR_EXT_MAG_WINDOW, ALCOR_EXT_MIN_ALTITUDE,
    ALCOR_EXT_MIN_BLOCK_FRAMES, ALCOR_EXT_MIN_WEIGHT, ALCOR_EXT_NFRAMES,
    ALCOR_EXT_VMAX, ALCOR_HORIZON_RADIUS, alcor_calibration
)
from .wcs import build_alcor_wcs
from .photometry import collect_alcor_photometry


# Per-pixel (az, alt) for a given WCS + shape is the same array every call and
# costs ~2 million SIP transforms to build, so it is memoized. The key is the
# WCS header text, which changes exactly when the geometry does.
_ALTAZ_CACHE = {}


def _pixel_altaz(wcs, shape):
    """Per-pixel ``(az, alt)`` in degrees for a raw frame of ``shape``."""
    key = (tuple(shape), wcs.to_header_string())
    cached = _ALTAZ_CACHE.get(key)
    if cached is not None:
        return cached
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    az, alt = wcs.pixel_to_world_values(xx.astype(float), yy.astype(float))
    out = (np.asarray(az, dtype=np.float32), np.asarray(alt, dtype=np.float32))
    _ALTAZ_CACHE[key] = out
    return out


def _unit_vectors(az_deg, alt_deg):
    """Cartesian unit vectors for horizontal coordinates, for great circles."""
    az = np.radians(np.asarray(az_deg, dtype=float))
    alt = np.radians(np.asarray(alt_deg, dtype=float))
    return np.stack([np.cos(alt) * np.cos(az),
                     np.cos(alt) * np.sin(az),
                     np.sin(alt)], axis=-1)


def _resolve_column(df, stem, band, method):
    """
    Find ``<stem>_<band>`` allowing for the ``_ap``/``_gauss`` suffixes that
    ``--both`` mode adds. A CSV written in single-method mode has no suffix at
    all, so both spellings have to be accepted or the combined and plain
    schemas cannot share this code.
    """
    for name in (f"{stem}_{band}_{method}", f"{stem}_{band}"):
        if name in df.columns:
            return name
    raise KeyError(f"no {stem}_{band} column (tried the "
                   f"'_{method}' suffix); have: {sorted(df.columns)[:12]}...")


def alcor_extinction_stars(df, band="g", method="ap",
                           mag_window=ALCOR_EXT_MAG_WINDOW,
                           min_altitude=ALCOR_EXT_MIN_ALTITUDE,
                           min_frames=ALCOR_EXT_MIN_BLOCK_FRAMES,
                           lost_min_frames=ALCOR_EXT_LOST_MIN_FRAMES):
    """
    Reduce a block of per-frame photometry to one extinction value per star.

    ``df`` is the collected photometry of the frames in the block (the output of
    :func:`collect_alcor_photometry`, or any frame with the same columns).
    Returns ``(measured, lost)``:

    ``measured``
        indexed by star name, with ``az``, ``alt``, ``ext`` (the mean over the
        block) and ``n`` (frames contributing). Restricted to non-variable,
        unsaturated stars inside ``mag_window`` -- the instrumental-magnitude
        range free of both the bright-star CMOS non-linearity and the faint-end
        bias -- above ``min_altitude``, seen in at least ``min_frames`` frames.
    ``lost``
        stars with no detection at all in at least ``lost_min_frames`` frames
        of the block. These have no extinction value, only a position; the
        extinction there is a lower limit.

    Variables are excluded because a variable has no single catalog magnitude,
    so its ``ext`` is NaN by construction -- differencing against a catalog
    magnitude would conflate cloud with the star's own variation.
    """
    ext_col = _resolve_column(df, "ext", band, method)
    mag_col = _resolve_column(df, "mag", band, method)
    flux_col = _resolve_column(df, "flux", band, method)
    sat_col = _resolve_column(df, "sat", band, method)

    if "name" in df.columns:
        names = df["name"]
    else:
        names = df.index.to_series()
    work = df.assign(_name=names.to_numpy())

    if "variable" in work.columns:
        work = work[~work["variable"].astype(bool)]
    work = work[~work[sat_col].astype(bool)]
    work = work[work["altitude"] > min_altitude]

    usable = (np.isfinite(work[ext_col])
              & work[mag_col].between(*mag_window))
    good = work[usable]
    grouped = good.groupby("_name", sort=False)
    measured = pd.DataFrame({
        "az": grouped["azimuth"].mean(),
        "alt": grouped["altitude"].mean(),
        "ext": grouped[ext_col].mean(),
        "n": grouped.size(),
    })
    measured = measured[measured["n"] >= min_frames]

    missing = work[(work[flux_col] <= 0) | ~np.isfinite(work[flux_col])]
    grouped = missing.groupby("_name", sort=False)
    lost = pd.DataFrame({
        "az": grouped["azimuth"].mean(),
        "alt": grouped["altitude"].mean(),
        "n": grouped.size(),
    })
    lost = lost[lost["n"] >= lost_min_frames]
    # A star can be measured in some frames of the block and lost in others;
    # keep it as a measurement, since a real value beats a lower limit.
    lost = lost[~lost.index.isin(measured.index)]

    return measured, lost


def alcor_extinction_grid(measured, step=ALCOR_EXT_GRID_STEP,
                          min_altitude=ALCOR_EXT_MIN_ALTITUDE,
                          kernel_sigma=ALCOR_EXT_KERNEL_SIGMA,
                          kernel_cut=ALCOR_EXT_KERNEL_CUT,
                          min_weight=ALCOR_EXT_MIN_WEIGHT):
    """
    Gaussian kernel average of the scattered measurements onto a regular
    alt/az grid.

    Returns ``(grid, az_values, alt_values)`` where ``grid`` has shape
    ``(n_alt, n_az)`` and is NaN wherever the summed kernel weight falls below
    ``min_weight`` -- i.e. where too few stars are close enough for the value to
    mean anything. Blanking rather than extrapolating matters: the regions with
    no usable stars are the *most* extinguished ones, so filling them in would
    quietly replace the worst cloud with a comfortable interpolation.
    """
    az_values = np.arange(0.0, 360.0, step)
    alt_values = np.arange(min_altitude, 90.0 + step, step)
    grid_az, grid_alt = np.meshgrid(az_values, alt_values)

    if not len(measured):
        return np.full(grid_az.shape, np.nan), az_values, alt_values

    star_vec = _unit_vectors(measured["az"].to_numpy(),
                             measured["alt"].to_numpy())
    grid_vec = _unit_vectors(grid_az.ravel(), grid_alt.ravel())
    sep = np.degrees(np.arccos(np.clip(grid_vec @ star_vec.T, -1.0, 1.0)))

    weight = np.exp(-0.5 * (sep / kernel_sigma) ** 2)
    weight[sep > kernel_cut * kernel_sigma] = 0.0
    wsum = weight.sum(axis=1)

    out = np.full(wsum.shape, np.nan)
    ok = wsum > min_weight
    out[ok] = (weight[ok] @ measured["ext"].to_numpy()) / wsum[ok]
    return out.reshape(grid_az.shape), az_values, alt_values


def alcor_extinction_map(measured, wcs=None, shape=None, time=None,
                         step=ALCOR_EXT_GRID_STEP,
                         min_altitude=ALCOR_EXT_MIN_ALTITUDE, **kwargs):
    """
    Resample the alt/az extinction grid into the camera's raw pixel frame.

    ``wcs`` defaults to the calibration epoch nearest ``time``; ``shape`` to the
    sensor's ``(ny, nx)``. Returns ``(map, wcs, grid)`` -- the raw-frame float32
    array (NaN outside the measurable region), the WCS it is on, and the
    intermediate alt/az grid, which callers may archive instead of the much
    larger raw map.
    """
    if wcs is None:
        cal = alcor_calibration(time)
        wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                              rotation=cal["rotation"],
                              radial_coeffs=cal["radial_coeffs"],
                              horizon_radius=cal["horizon_radius"],
                              tangential_coeffs=cal["tangential_coeffs"],
                              axis_tilt=cal["axis_tilt"])
    if shape is None:
        shape = ALCOR_EXT_FRAME_SHAPE

    grid, az_values, alt_values = alcor_extinction_grid(
        measured, step=step, min_altitude=min_altitude, **kwargs)

    az, alt = _pixel_altaz(wcs, shape)

    # Azimuth wraps, so pad the grid circularly by one column and shift the
    # index; without this every pixel between 359 deg and 0 deg interpolates
    # across the whole grid instead of across the seam.
    padded = np.concatenate([grid[:, -1:], grid, grid[:, :1]], axis=1)
    coords = np.stack([
        (alt - alt_values[0]) / step,
        (np.mod(az, 360.0)) / step + 1.0,
    ])
    # order=1 keeps NaN propagation local: a blank grid cell blanks only the
    # pixels that actually interpolate from it.
    sampled = map_coordinates(padded, coords, order=1, mode="nearest",
                              cval=np.nan)
    sampled = np.asarray(sampled, dtype=np.float32)
    sampled[alt < min_altitude] = np.nan
    sampled[~np.isfinite(alt)] = np.nan
    return sampled, wcs, grid


def _pixels_per_degree(wcs=None, header=None):
    """
    Radial plate scale in px/deg. The ARC projection is equidistant in radius,
    so one number is exact along the radial direction anywhere in the field.
    """
    if header is not None and "PIXSCALE" in header:
        return float(header["PIXSCALE"])
    if wcs is not None:
        try:
            return float(abs(1.0 / wcs.wcs.cdelt[1]))
        except (AttributeError, IndexError, ZeroDivisionError):
            pass
    return ALCOR_HORIZON_RADIUS / 90.0


def _extinction_header(wcs, measured, lost, times, band, method,
                       kernel_sigma, min_weight, mag_window, nframes):
    """Raw-frame WCS plus the full provenance of the map's calibration chain."""
    header = wcs.to_header(relax=True)
    header["BUNIT"] = ("mag", "G-band cloud extinction")
    header["NFRAMES"] = (int(nframes), "frames averaged into this map")
    header["NSTARS"] = (int(len(measured)), "stars with a measured extinction")
    header["NLOST"] = (int(len(lost)), "stars lost entirely (lower limits)")
    header["EXTBAND"] = (str(band), "photometric channel")
    header["EXTMETH"] = (str(method), "aperture or gaussian photometry")
    header["KERNSIG"] = (float(kernel_sigma), "smoothing kernel sigma (deg)")
    header["MINWT"] = (float(min_weight), "min summed kernel weight per cell")
    header["MAGLO"] = (float(mag_window[0]), "bright end of the usable window")
    header["MAGHI"] = (float(mag_window[1]), "faint end of the usable window")
    header["PIXSCALE"] = (_pixels_per_degree(wcs), "radial plate scale (px/deg)")
    # Reserved for the clear-night flat field. The measured clear-night trend is
    # ~ -0.12 mag at alt 20-30 and ~ -0.09 at 75-90 against ~0 at 45-60; it is
    # recorded, not corrected, until several clear nights are in hand.
    header["EXTFLAT"] = ("none", "clear-night flat field applied")
    if len(times):
        header["TSTART"] = (Time(min(times)).isot, "UT of the first frame")
        header["TEND"] = (Time(max(times)).isot, "UT of the last frame")
        header["TMID"] = (Time(np.median([Time(t).mjd for t in times]),
                               format="mjd").isot, "UT midpoint")
    return header


def _stars_hdu(measured, lost):
    """
    One ``STARS`` table holding both measurements and lower limits, so a map is
    self-diagnosing and the lost stars survive into the file rather than
    existing only on the plot.
    """
    names = list(measured.index.astype(str)) + list(lost.index.astype(str))
    az = np.concatenate([measured["az"].to_numpy(), lost["az"].to_numpy()])
    alt = np.concatenate([measured["alt"].to_numpy(), lost["alt"].to_numpy()])
    ext = np.concatenate([measured["ext"].to_numpy(),
                          np.full(len(lost), np.nan)])
    nfr = np.concatenate([measured["n"].to_numpy(), lost["n"].to_numpy()])
    detected = np.concatenate([np.ones(len(measured), bool),
                               np.zeros(len(lost), bool)])
    width = max((len(n) for n in names), default=1)
    cols = [
        fits.Column(name="name", format=f"{width}A", array=np.array(names)),
        fits.Column(name="az", format="E", unit="deg", array=az),
        fits.Column(name="alt", format="E", unit="deg", array=alt),
        fits.Column(name="ext", format="E", unit="mag", array=ext),
        fits.Column(name="nframes", format="J", array=nfr.astype(np.int32)),
        fits.Column(name="detected", format="L", array=detected),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="STARS")
    hdu.header["COMMENT"] = "detected=F: star lost entirely; ext is a lower limit"
    return hdu


def alcor_extinction_fits(inputs, output_file=None, band="g", method="ap",
                          latest=None, nframes=ALCOR_EXT_NFRAMES,
                          wcs=None, shape=None, overwrite=False,
                          kernel_sigma=ALCOR_EXT_KERNEL_SIGMA,
                          min_weight=ALCOR_EXT_MIN_WEIGHT,
                          mag_window=ALCOR_EXT_MAG_WINDOW,
                          min_altitude=ALCOR_EXT_MIN_ALTITUDE,
                          return_products=False):
    """
    Build an extinction map from per-frame photometry CSVs and write it as FITS.

    ``inputs`` is a directory of ``*_phot.csv``, a list of such paths, or an
    already-collected DataFrame. With ``latest`` set, only the ``latest`` newest
    frames are used -- the real-time path, where the map must reflect the sky
    now rather than the whole night.

    The primary HDU is the raw-frame float32 map with the alt/az ARC WCS; a
    ``STARS`` extension carries every contributing star. Returns the output
    path, or ``(path, map, measured, lost)`` with ``return_products``.
    """
    if isinstance(inputs, pd.DataFrame):
        df = inputs
    else:
        df = collect_alcor_photometry(inputs)

    if "OBSTIME" in df.columns:
        df = df.sort_values("OBSTIME")
        stamps = np.sort(df["OBSTIME"].unique())
        if latest:
            stamps = stamps[-int(latest):]
            df = df[df["OBSTIME"].isin(stamps)]
        times = [pd.Timestamp(s).to_pydatetime() for s in stamps]
    else:
        times = []

    measured, lost = alcor_extinction_stars(
        df, band=band, method=method, mag_window=mag_window,
        min_altitude=min_altitude)
    if not len(measured):
        raise ValueError("no usable star measurements in the given frames")

    mid = Time(np.median([Time(t).mjd for t in times]), format="mjd") \
        if times else None
    ext_map, wcs, _ = alcor_extinction_map(
        measured, wcs=wcs, shape=shape, time=mid, kernel_sigma=kernel_sigma,
        min_weight=min_weight, min_altitude=min_altitude)

    header = _extinction_header(wcs, measured, lost, times, band, method,
                                kernel_sigma, min_weight, mag_window,
                                len(times) or nframes)
    hdul = fits.HDUList([fits.PrimaryHDU(data=ext_map, header=header),
                         _stars_hdu(measured, lost)])

    if output_file is None:
        stamp = mid.isot.replace(":", "").replace("-", "")[:15] if mid else "map"
        output_file = Path(f"alcor_extinction_{stamp}.fits")
    output_file = Path(output_file)
    hdul.writeto(output_file, overwrite=overwrite)

    if return_products:
        return output_file, ext_map, measured, lost
    return output_file


def alcor_extinction_at(source, az, alt, radius=ALCOR_EXT_LOOKUP_RADIUS):
    """
    Extinction in magnitudes at one horizontal position, for the redis publisher.

    ``source`` is a path to a map written by :func:`alcor_extinction_fits`, an
    open ``HDUList``, or a ``(data, header)`` pair. ``radius`` is a great-circle
    radius in **degrees**: the value returned is the median over that cone
    rather than a single pixel. The map is already smoothed on an 8-degree
    kernel so the cone adds little smoothing -- its job is robustness, surviving
    a blank pixel, a resampling edge, or a position that lands just outside the
    valid region. Returns NaN when nothing in the cone is measurable.
    """
    if isinstance(source, (str, Path)):
        with fits.open(source) as hdul:
            data = np.asarray(hdul[0].data, dtype=float)
            header = hdul[0].header
            return _lookup(data, header, az, alt, radius)
    if isinstance(source, fits.HDUList):
        return _lookup(np.asarray(source[0].data, dtype=float),
                       source[0].header, az, alt, radius)
    data, header = source
    return _lookup(np.asarray(data, dtype=float), header, az, alt, radius)


def _lookup(data, header, az, alt, radius):
    from astropy.wcs import WCS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(header)
    x, y = wcs.world_to_pixel_values(float(az), float(alt))
    if not (np.isfinite(x) and np.isfinite(y)):
        return float("nan")

    ny, nx = data.shape
    if radius <= 0:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < nx and 0 <= yi < ny):
            return float("nan")
        return float(data[yi, xi])

    rpx = radius * _pixels_per_degree(wcs=wcs, header=header)
    x0, x1 = int(np.floor(x - rpx)), int(np.ceil(x + rpx)) + 1
    y0, y1 = int(np.floor(y - rpx)), int(np.ceil(y + rpx)) + 1
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, nx), min(y1, ny)
    if x1 <= x0 or y1 <= y0:
        return float("nan")

    patch = data[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= rpx ** 2
    values = patch[inside]
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan")
    return float(np.median(values))


def _draw_extinction(ax, measured, lost, grid, az_values, alt_values, norm,
                     cmap, min_altitude=ALCOR_EXT_MIN_ALTITUDE):
    """
    Polar rendering shared by the single plot and the animation frames.

    Orientation matches the all-sky renderings: zenith at centre, north up,
    azimuth increasing counter-clockwise so east is at the left -- the view
    looking up, the same convention as ``_add_alcor_alt_az_grid``.
    """
    # Close the azimuth seam before drawing. The grid spans 0..359 deg, and
    # pcolormesh draws quads only BETWEEN adjacent columns, so without the
    # wrap column there is an undrawn one-degree wedge at north.
    plot_az = np.append(az_values, az_values[0] + 360.0)
    plot_grid = np.concatenate([grid, grid[:, :1]], axis=1)
    grid_az, grid_alt = np.meshgrid(plot_az, alt_values)
    # gouraud, not flat: at the rim a 1-degree grid cell is physically large
    # and flat shading renders the map visibly blocky there.
    mesh = ax.pcolormesh(np.radians(grid_az), 90.0 - grid_alt, plot_grid,
                         norm=norm, cmap=cmap, shading="gouraud")
    if len(measured):
        ax.scatter(np.radians(measured["az"]), 90.0 - measured["alt"],
                   s=1.5, c="0.25", alpha=0.35, linewidths=0)
    if len(lost):
        ax.scatter(np.radians(lost["az"]), 90.0 - lost["alt"], s=26,
                   facecolors="none", edgecolors="#00d0ff", linewidths=1.1,
                   zorder=5)
    ax.set_theta_zero_location("N")
    ax.set_ylim(0, 90.0 - min_altitude)
    ax.set_yticklabels([])
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=11)
    ax.grid(alpha=0.25, lw=0.5)
    return mesh


def plot_alcor_extinction(measured, lost, output_file=None, title="",
                          subtitle=None, vmax=ALCOR_EXT_VMAX,
                          cmap="inferno_r", figsize=(6.4, 7.0), dpi=140,
                          min_altitude=ALCOR_EXT_MIN_ALTITUDE, **kwargs):
    """
    Render one extinction map as the observer-facing all-sky figure.

    The colour scale is pinned to ``0 .. vmax`` by default rather than
    autoscaled, so that maps from different times and different nights are
    directly comparable and a clear sky always looks the same.
    """
    grid, az_values, alt_values = alcor_extinction_grid(
        measured, min_altitude=min_altitude, **kwargs)
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    mesh = _draw_extinction(ax, measured, lost, grid, az_values, alt_values,
                            norm, cmap, min_altitude=min_altitude)
    if subtitle is None:
        med = np.nanmedian(measured["ext"]) if len(measured) else np.nan
        subtitle = (f"{len(measured)} stars measured, median {med:+.2f} mag"
                    + (f",  {len(lost)} lost" if len(lost) else ""))
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, pad=14)

    cax = fig.add_axes([0.15, 0.075, 0.7, 0.018])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label("G-band extinction [mag]      open circles: stars lost "
                   "entirely (lower limit)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if output_file is not None:
        fig.savefig(output_file, dpi=dpi, facecolor="white")
        plt.close(fig)
        return Path(output_file)
    return fig


def plot_alcor_extinction_fits(filename, output_file=None, **kwargs):
    """
    Re-render a saved extinction map without recomputing it, using the ``STARS``
    table the file carries. ``output_file`` defaults to the input with a
    ``.png`` suffix.
    """
    filename = Path(filename)
    if output_file is None:
        stem = str(filename)
        for ext in (".fits.gz", ".fits"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        output_file = stem + ".png"

    with fits.open(filename) as hdul:
        header = hdul[0].header
        rows = hdul["STARS"].data
        # FITS is big-endian on disk; pandas refuses a big-endian buffer on a
        # little-endian host, so each column is converted to native order.
        table = pd.DataFrame({name: _native(rows[name])
                              for name in rows.names})
    table = table.set_index("name")
    measured = table[table["detected"]].rename(columns={"nframes": "n"})
    lost = table[~table["detected"]].rename(columns={"nframes": "n"})

    title = kwargs.pop("title", None)
    if title is None:
        title = f"alcor cloud extinction  ·  {header.get('TMID', '')}"
    return plot_alcor_extinction(measured[["az", "alt", "ext", "n"]],
                                 lost[["az", "alt", "n"]],
                                 output_file=output_file, title=title, **kwargs)


def _native(array):
    """A FITS column in the host's native byte order (pandas requires it)."""
    array = np.asarray(array)
    if array.dtype.byteorder not in ("=", "|"):
        return array.astype(array.dtype.newbyteorder("="))
    return array


def _blocks(df, nframes, stride=None):
    """
    ``nframes``-frame blocks in time order, advancing by ``stride`` (default: no
    overlap). Block membership is resolved once by frame index rather than with
    a boolean ``isin`` per block: the latter is O(n_blocks * n_rows) and
    dominates the run time of a whole-night animation.
    """
    stride = stride or nframes
    stamps = np.sort(df["OBSTIME"].unique())
    pos = np.searchsorted(stamps, df["OBSTIME"].to_numpy())
    for i in range(0, len(stamps) - nframes + 1, stride):
        sel = (pos >= i) & (pos < i + nframes)
        yield stamps[i + nframes // 2], df[sel]


def alcor_extinction_movie(inputs, output_file=None, frames_dir=None,
                           nframes=ALCOR_EXT_NFRAMES, stride=None, fps=12,
                           band="g", method="ap", vmax=ALCOR_EXT_VMAX,
                           cmap="inferno_r", title=None, keep_frames=False,
                           min_altitude=ALCOR_EXT_MIN_ALTITUDE, quiet=False,
                           **kwargs):
    """
    Animate a night's extinction maps.

    Renders one frame per block and encodes them with ffmpeg. ``stride``
    defaults to ``nframes`` (independent, non-overlapping averages); a smaller
    stride overlaps successive windows, which looks smoother but correlates
    adjacent frames -- worth knowing before reading structure off the movie.

    The figure and its axes are built once and the axes cleared per frame:
    rebuilding a polar figure for every block costs more than all of the
    numerical work in a night combined.
    """
    if isinstance(inputs, pd.DataFrame):
        df = inputs
    else:
        df = collect_alcor_photometry(inputs)
    if "OBSTIME" not in df.columns:
        raise ValueError("photometry has no OBSTIME column; cannot animate")
    df = df.sort_values("OBSTIME")

    source = Path(inputs) if isinstance(inputs, (str, Path)) else None
    if title is None:
        title = source.name if source is not None else "alcor extinction"
    if output_file is None:
        output_file = Path(f"{title}_extinction.mp4")
    output_file = Path(output_file)

    import tempfile
    tmp = None
    if frames_dir is None:
        tmp = tempfile.TemporaryDirectory()
        frames_dir = tmp.name
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("frame_*.png"):
        stale.unlink()

    norm = Normalize(vmin=0.0, vmax=vmax)
    fig = plt.figure(figsize=(6.4, 7.0))
    ax = fig.add_subplot(111, projection="polar")
    cax = fig.add_axes([0.15, 0.095, 0.7, 0.018])

    all_blocks = list(_blocks(df, nframes, stride))
    if not all_blocks:
        raise ValueError(f"fewer than {nframes} frames; nothing to animate")
    t0 = pd.Timestamp(all_blocks[0][0])
    t1 = pd.Timestamp(all_blocks[-1][0])

    written = 0
    first = True
    for i, (stamp, sub) in enumerate(all_blocks):
        measured, lost = alcor_extinction_stars(sub, band=band, method=method,
                                                min_altitude=min_altitude)
        if not len(measured):
            continue
        grid, az_values, alt_values = alcor_extinction_grid(
            measured, min_altitude=min_altitude, **kwargs)
        ax.clear()
        mesh = _draw_extinction(ax, measured, lost, grid, az_values,
                                alt_values, norm, cmap,
                                min_altitude=min_altitude)
        ts = pd.Timestamp(stamp)
        med = np.nanmedian(measured["ext"])
        # The night label already names the date, so repeating the UT date here
        # reads as two different dates on any frame after midnight.
        ax.set_title(f"{title}  ·  {ts.strftime('%H:%M')} UT\n"
                     f"{len(measured)} stars measured, median {med:+.2f} mag"
                     + (f",  {len(lost)} lost" if len(lost) else ""),
                     fontsize=11, pad=14)

        # A progress bar under the map: without it a viewer cannot tell how far
        # through the night a given frame sits.
        frac = (ts - t0) / (t1 - t0) if t1 > t0 else 0.0
        fig.patches.clear()
        fig.patches.append(plt.Rectangle((0.15, 0.030), 0.70, 0.006,
                                         transform=fig.transFigure,
                                         facecolor="0.85", zorder=3))
        fig.patches.append(plt.Rectangle((0.15, 0.030), 0.70 * frac, 0.006,
                                         transform=fig.transFigure,
                                         facecolor="0.25", zorder=4))
        if first:
            cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal",
                                extend="max")
            cbar.set_label(f"G-band extinction [mag]   ({nframes}-frame "
                           f"average, ~{nframes * ALCOR_EXT_CADENCE / 60:.0f}"
                           f" min)", fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            first = False
        fig.savefig(frames_dir / f"frame_{i:04d}.png", dpi=110,
                    facecolor="white")
        written += 1
    plt.close(fig)

    if not written:
        raise ValueError("no block produced a usable map; nothing to animate")
    if not quiet:
        print(f"rendered {written} frames")

    _encode(frames_dir, output_file, fps)
    if tmp is not None and not keep_frames:
        tmp.cleanup()
    return output_file


def _encode(frames_dir, output_file, fps):
    """
    ffmpeg encode of the rendered frames. The pad filter is required, not
    cosmetic: H.264 needs even pixel dimensions and matplotlib will happily
    emit odd ones.
    """
    cmd = [
        "ffmpeg", "-y", "-framerate", str(int(fps)),
        "-pattern_type", "glob",
        "-i", str(Path(frames_dir) / "frame_*.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        str(output_file),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as err:
        raise RuntimeError("ffmpeg is required to encode the animation") from err
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"ffmpeg failed: {err.stderr.decode('utf-8', 'replace')[-400:]}"
        ) from err
