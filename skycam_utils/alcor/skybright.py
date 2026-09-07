"""Calibrated V mag/arcsec^2 surface-brightness maps and sampling cones."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from .config import (
    ALCOR_CALIB_EXPTIME, ALCOR_FIELD_RADIUS, ALCOR_RADIUS,
    ALCOR_SB_APERTURE_RADIUS, ALCOR_SB_BEST_MIN_ALTITUDE, ALCOR_SB_TARGETS,
    alcor_zeropoint
)
from .timeutils import _alcor_frame_time, _read_frame_exposure
from .wcs import _alcor_pixel_solid_angle
from .masks import load_alcor_horizon_mask
from .io import _corner_bias, load_alcor_fits
from .display import _add_alcor_alt_az_grid, _alcor_zenith_crop_bounds


def _alcor_sky_brightness_map(cube, wcs, time, exposure,
                              saturation=None,
                              field_radius=ALCOR_FIELD_RADIUS):
    """
    Full-frame V mag/arcsec^2 surface-brightness map from an alcor cube.

    Returns ``(mu, alt)``: the green channel calibrated to V magnitudes per
    square arcsecond -- corner-bias-subtracted, scaled to the
    :data:`ALCOR_CALIB_EXPTIME` reference exposure via ``exposure`` (counts are
    linear in exposure), divided by the WCS per-pixel solid angle
    (:func:`_alcor_pixel_solid_angle`), and offset by the epoch G->V zeropoint
    with no airmass term -- plus the per-pixel altitude grid in degrees. Pixels
    the WCS cannot project (off the sky), pixels with raw G at or above
    ``saturation`` when one is given (clipped/non-linear -- ``None``, the
    default, keeps them: see :data:`ALCOR_SB_SATURATION` for why blanking them
    misleads), and pixels farther than ``field_radius``
    from the optical axis (outside the illuminated image circle -- see
    :data:`ALCOR_FIELD_RADIUS`; pass None to keep them) are blanked to NaN in
    ``mu``. Geometric masking (horizon mask / altitude floor) is left to the
    caller, since callers differ in their default sky cutoff.
    """
    g_raw = np.asarray(cube[1], dtype=float)
    bias = _corner_bias(cube)[1]
    # Counts scaled to the calibration's reference exposure (counts/20 s).
    g20 = (g_raw - bias) * (ALCOR_CALIB_EXPTIME / exposure)

    ny, nx = g_raw.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    az, alt = wcs.pixel_to_world_values(xx.astype(float), yy.astype(float))
    omega = _alcor_pixel_solid_angle(az, alt)
    zp_g = alcor_zeropoint(time)["g"]["zp"]

    with np.errstate(divide="ignore", invalid="ignore"):
        surf = g20 / omega
        mu = np.where(surf > 0, -2.5 * np.log10(surf) + zp_g, np.nan)

    blank = ~np.isfinite(mu) | ~np.isfinite(alt)
    if saturation is not None:
        blank |= g_raw >= saturation
    if field_radius is not None:
        # The optical axis is CRPIX (1-based), which is where build_alcor_wcs
        # puts (xcen, ycen) -- so the WCS stays the single source of geometry.
        ax, ay = (float(c) - 1.0 for c in wcs.wcs.crpix[:2])
        blank |= (xx - ax) ** 2 + (yy - ay) ** 2 > float(field_radius) ** 2
    mu = np.where(blank, np.nan, mu)
    return mu, alt



def _alcor_best_cone_targets(radius_deg=ALCOR_SB_APERTURE_RADIUS,
                             min_altitude=ALCOR_SB_BEST_MIN_ALTITUDE):
    """
    Candidate cone centres tiling the sky above ``min_altitude``.

    Rings are spaced one cone diameter apart in altitude, starting one radius
    above the floor so the lowest ring just touches it, and within a ring the
    azimuth step is widened by ``1/cos(alt)`` so the cones tile at roughly
    constant spacing on the sphere instead of bunching near the zenith. The
    zenith itself is included, which makes ``allsky_mv_best`` always at least as
    dark as ``allsky_mv_zenith``.

    Returns a dict of ``name -> (azimuth, altitude)`` in degrees, the same shape
    of input :func:`_alcor_cone_indices` takes for the fixed targets.
    """
    step = 2.0 * radius_deg
    targets = {}
    alt = min_altitude + radius_deg
    while alt < 90.0:
        n = max(1, int(round(360.0 * np.cos(np.radians(alt)) / step)))
        for i in range(n):
            az = 360.0 * i / n
            targets[f"best_{alt:.0f}_{az:03.0f}"] = (az, alt)
        alt += step
    targets["best_zenith"] = (0.0, 90.0)
    return targets



def _alcor_cone_indices(wcs, shape, targets=None,
                        radius_deg=ALCOR_SB_APERTURE_RADIUS, exclude=None):
    """
    Flat pixel indices of the sampling cones around fixed ``(az, alt)`` targets.

    ``targets`` maps a name to an ``(azimuth, altitude)`` pair in degrees
    (default :data:`ALCOR_SB_TARGETS`); a pixel belongs to a cone when its
    great-circle separation from that direction is at most ``radius_deg``.
    ``exclude`` is an optional ``(ny, nx)`` boolean array of pixels to drop (the
    horizon/obstruction mask), so terrain cannot drag a low-altitude cone faint.

    The cones depend only on the WCS, so a caller processing a whole night
    builds them once and reuses them for every frame. Returns a dict mapping
    each name to a flat ``int64`` index array (possibly empty).
    """
    if targets is None:
        targets = ALCOR_SB_TARGETS
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    az, alt = wcs.pixel_to_world_values(xx.astype(float), yy.astype(float))

    az_r = np.radians(az)
    alt_r = np.radians(alt)
    sin_alt = np.sin(alt_r)
    cos_alt = np.cos(alt_r)
    valid = np.isfinite(alt)
    if exclude is not None:
        valid &= ~np.asarray(exclude, dtype=bool)

    cos_radius = np.cos(np.radians(radius_deg))
    cones = {}
    for name, (az0, alt0) in targets.items():
        az0_r = np.radians(az0)
        alt0_r = np.radians(alt0)
        with np.errstate(invalid="ignore"):
            cos_sep = (sin_alt * np.sin(alt0_r)
                       + cos_alt * np.cos(alt0_r) * np.cos(az_r - az0_r))
            inside = valid & (cos_sep >= cos_radius)
        cones[name] = np.flatnonzero(inside)
    return cones



def plot_alcor_sky_brightness(filename, outimage=None, outfig=None,
                              radius=ALCOR_RADIUS, fov_altitude=-2.0,
                              horizon_mask=False, saturation=None,
                              vmin=None, vmax=None, cmap="cividis_r", figsize=12):
    """
    Render an alcor frame as a V-band sky-surface-brightness map.

    The green channel is converted to V magnitudes per square arcsecond using the
    G->V photometric calibration (see :data:`ALCOR_ZEROPOINTS`). Per pixel, the
    corner-bias-subtracted G counts are scaled to the calibration's reference
    exposure (:data:`ALCOR_CALIB_EXPTIME`, 20 s, the dark-condition default; the
    frame's ``EXPOSURE`` header sets the actual time and counts are linear in it),
    divided by the per-pixel solid angle, and converted with the G zeropoint::

        mu = -2.5 * log10(g20 / omega_arcsec2) + zp_g

    where ``g20`` is counts/(20 s) and ``omega_arcsec2`` is the pixel solid angle
    derived from the WCS, so the ARC projection's zenith->horizon plate-scale
    change and the SIP distortion are both accounted for exactly. No airmass term
    is applied: the map is the *observed* sky brightness, so horizon light domes,
    airglow and Milky-Way gradients, and scattered moonlight all show as real
    structure. The G band is essentially color-flat (G~=V), so no color term is
    used; the absolute scale inherits the zeropoint's ~0.03 mag epoch stability.

    Masking: with a ``saturation`` given, pixels with raw G at or above it are
    clipped/non-linear
    and are blanked, as are non-sky pixels -- by default everything below
    ``fov_altitude`` degrees, or, when ``horizon_mask`` is True, the
    obstruction/terrain mask from :func:`load_alcor_horizon_mask` (which already
    includes below-horizon). The map is cropped to a ``radius``-pixel square
    around the WCS zenith and rendered north-up (``origin="lower"``) with an
    alt/az grid and a colorbar in V mag / arcsec^2. The upper-right corner is
    annotated with the sigma-clipped median zenith brightness (the cap above
    altitude 85 deg).

    Parameters
    ----------
    filename : str
        FITS filename of an alcor OMEA 8C frame (gz/bz2 allowed).
    outimage : str, optional
        If set, save the colour-mapped cropped surface-brightness array here.
    outfig : str, optional
        If set, save the annotated figure (extension picks the backend).
    radius : float (default ALCOR_RADIUS)
        Half-width (pixels) of the display crop around the zenith.
    fov_altitude : float (default -2.0)
        Sky cutoff in degrees: pixels below this altitude are masked. Ignored
        when ``horizon_mask`` is True.
    horizon_mask : bool (default False)
        Use the full horizon/obstruction mask instead of the altitude cutoff.
    saturation : int or None (default None)
        Raw-ADU level at/above which G pixels are masked as non-linear.
    vmin, vmax : float, optional
        Colorbar limits in mag/arcsec^2 (default: robust autoscale of the sky).
    cmap : str (default "cividis_r")
        Matplotlib colormap (perceptually uniform, colour-blind friendly); the
        reversed default renders bright sky light and dark sky dark.
    figsize : float (default 12)
        matplotlib figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cube, wcs, _ = load_alcor_fits(filename)
    time = _alcor_frame_time(filename)
    exposure = _read_frame_exposure(filename)

    mu, alt = _alcor_sky_brightness_map(cube, wcs, time, exposure,
                                        saturation=saturation)
    ny, nx = mu.shape

    # Geometric (non-sky) masking: the full horizon/obstruction mask, or an
    # altitude floor when not using it (or when the mask is unavailable).
    if horizon_mask:
        hmask, _ = load_alcor_horizon_mask(time)
        geo_blank = (np.asarray(hmask, dtype=bool) if hmask is not None
                     else (alt < fov_altitude))
    else:
        geo_blank = (alt < fov_altitude)
    mu = np.where(geo_blank, np.nan, mu)

    # Sigma-clipped median zenith brightness from the cap above altitude 85 deg.
    zen = mu[np.isfinite(mu) & (alt > 85.0)]
    zenith_mu = float(sigma_clipped_stats(zen)[1]) if zen.size else float("nan")

    xz, yz, xl, xu, yl, yu = _alcor_zenith_crop_bounds(wcs, ny, nx, radius)
    mu_crop = mu[yl:yu, xl:xu]
    cx, cy = xz - xl, yz - yl

    finite = mu_crop[np.isfinite(mu_crop)]
    if vmin is None:
        vmin = float(np.percentile(finite, 1)) if finite.size else None
    if vmax is None:
        vmax = float(np.percentile(finite, 99)) if finite.size else None

    if outimage is not None:
        plt.imsave(outimage, np.flipud(mu_crop), cmap=cmap, vmin=vmin, vmax=vmax)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    circle = Circle((cx, cy), radius, facecolor="none", edgecolor=(0, 0, 0),
                    linewidth=1, alpha=0.5)
    ax.add_patch(circle)
    ax.axis("off")
    im_plot = ax.imshow(mu_crop, origin="lower", cmap=cmap_obj,
                        vmin=vmin, vmax=vmax)
    im_plot.set_clip_path(circle)

    # Annotation inside the FOV (altitude rings, grid lines) is dark so it reads
    # against the light sky colormap. Annotation outside the FOV (azimuth labels,
    # colorbar, zenith readout) sits on the transparent background, so use a
    # medium-dark grey that reads on a light page (matches plot_alcor_fits).
    inside_color = "0.2"
    outside_color = "0.5"
    # Dedicated colorbar axes on the far-right margin, clear of the west (270deg)
    # azimuth label; the main image axes stays at its 111 position so the polar
    # alt/az overlay aligns with it.
    cax = fig.add_axes([0.97, 0.28, 0.018, 0.44])
    cbar = fig.colorbar(im_plot, cax=cax)
    cbar.ax.invert_yaxis()   # dark (faint sky) at the bottom, bright at the top
    cbar.set_label("$m_{V}$ (mag arcsec$^{-2}$)", fontsize=14,
                   color=outside_color)
    cbar.ax.tick_params(colors=outside_color)
    cbar.outline.set_edgecolor(outside_color)

    _add_alcor_alt_az_grid(fig, wcs, xz, yz, radius, color=inside_color,
                           label_color=outside_color, grid_color="0.1",
                           grid_alpha=0.6)

    fig.text(0.90, 0.95,
             f"$m_{{V}}$(zenith) = {zenith_mu:.2f} mag arcsec$^{{-2}}$",
             ha="right", va="top", color=outside_color, fontsize=15)

    if outfig is not None:
        plt.savefig(outfig, transparent=True, bbox_inches="tight", pad_inches=0)

    return fig



def _alcor_sb_fits_header(wcs, time, exposure, saturation, horizon_mask):
    """
    FITS header for a calibrated surface-brightness map: the raw-frame alt/az
    WCS plus the provenance of the calibration chain that produced the values.
    """
    zp = alcor_zeropoint(time)
    header = wcs.to_header(relax=True)
    header["BUNIT"] = ("mag/arcsec2", "observed V surface brightness")
    header["ZP_G"] = (zp["g"]["zp"], "G->V zeropoint applied (mag)")
    header["ZP_EPOCH"] = (zp["epoch"], "ALCOR_ZEROPOINTS epoch used")
    header["EXPOSURE"] = (exposure, "frame exposure (s)")
    header["CALIBEXP"] = (ALCOR_CALIB_EXPTIME, "reference exposure for counts (s)")
    header["SATLEVEL"] = (
        "none" if saturation is None else saturation,
        "raw G blanked at/above this ADU" if saturation is not None
        else "saturated pixels kept, not blanked")
    header["HORIZMSK"] = (bool(horizon_mask), "horizon/obstruction mask applied")
    return header



def alcor_sky_brightness_fits(filename, output_file=None, horizon_mask=False,
                              saturation=None, overwrite=False,
                              **kwargs):
    """
    Calibrate an alcor OMEA 8C frame to a V mag/arcsec^2 surface-brightness map
    and write it as a 2-D FITS image with the raw-frame alt/az WCS attached.

    The green channel is bias-subtracted, exposure-normalised to
    :data:`ALCOR_CALIB_EXPTIME`, divided by the WCS per-pixel solid angle, and
    converted with the epoch G->V zeropoint (no airmass term) -- the same
    calibration as :func:`plot_alcor_sky_brightness`, but written as a
    full-frame ``float32`` FITS data product in the camera's native orientation
    so the attached WCS resolves directly (matching :func:`alcor_proc_fits`).
    Bad pixels are repaired by default (``badpix="repair"``).

    Off-frame pixels, and with a ``saturation`` given the pixels with raw G at or
    above it, are blanked
    to NaN. With ``horizon_mask=True`` the not-sky region from
    :func:`load_alcor_horizon_mask` is additionally blanked; otherwise no
    altitude floor is applied (every on-sky pixel keeps its calibrated value).

    Parameters
    ----------
    filename : str or `~pathlib.Path`
        Input alcor FITS frame (gz/bz2 allowed).
    output_file : str or `~pathlib.Path` or None (default=None)
        Output path. If None, derived from `filename` by replacing the first
        `.fits` extension with `_sb.fits`.
    horizon_mask : bool (default=False)
        Additionally blank the full horizon/obstruction mask.
    saturation : int or None (default None)
        Raw-ADU level at/above which G pixels are blanked as non-linear.
    overwrite : bool (default=False)
        Passed through to `fits.PrimaryHDU.writeto`.
    **kwargs
        Forwarded to `load_alcor_fits` (``wcs``, ``masks_dir``, ...). ``badpix``
        defaults to ``"repair"`` here but may be overridden.

    Returns
    -------
    output_file : `~pathlib.Path`
        Path to the written FITS file.
    """
    kwargs.setdefault("badpix", "repair")
    cube, wcs, _ = load_alcor_fits(filename, **kwargs)
    time = _alcor_frame_time(filename)
    exposure = _read_frame_exposure(filename)

    mu, _ = _alcor_sky_brightness_map(cube, wcs, time, exposure,
                                      saturation=saturation)
    if horizon_mask:
        hmask, _ = load_alcor_horizon_mask(time)
        if hmask is not None:
            mu = np.where(np.asarray(hmask, dtype=bool), np.nan, mu)

    if output_file is None:
        stem = str(filename)
        for ext in (".fits.bz2", ".fits.gz", ".fits"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        output_file = stem + "_sb.fits"
    output_file = Path(output_file)

    header = _alcor_sb_fits_header(wcs, time, exposure, saturation, horizon_mask)
    hdu = fits.PrimaryHDU(data=mu.astype(np.float32), header=header)
    hdu.writeto(output_file, overwrite=overwrite)
    return output_file



def _cone_median(mu, indices):
    """Median of a flattened surface-brightness map over one cone's pixels."""
    if indices.size == 0:
        return float("nan")
    values = mu.reshape(-1)[indices]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


# Plot styling for the summary tracks: column -> (label, colour, linewidth).
# The darkest-cone track is drawn heaviest because it is the one that actually
# answers "how dark did it get"; the two light domes are the thinnest.
ALCOR_SB_SUMMARY_STYLE = {
    "allsky_mv_best": ("darkest cone (floating)", "#222222", 1.6),
    "allsky_mv_zenith": ("zenith", "#1f77b4", 1.4),
    "allsky_mv_tucson": ("Tucson dome", "#d62728", 1.2),
    "allsky_mv_nogales": ("Nogales dome", "#ff7f0e", 1.2),
}


def _sb_summary_label(column, label):
    """
    Append a fixed cone's (az, alt) to its label so the legend says where the
    track was measured. The floating darkest cone has no fixed position, so it
    is left alone.
    """
    target = ALCOR_SB_TARGETS.get(column)
    if target is None or column == "allsky_mv_zenith":
        return label
    az, alt = target
    return f"{label} (az {az:g}, alt {alt:g})"


def plot_alcor_sb_summary(filename, output_file=None, title=None,
                          figsize=(13, 8), dpi=140):
    """
    Plot one night's sky-brightness summary from the ``sky_brightness.csv``
    written by :func:`alcor_process_night`.

    The upper panel draws every ``allsky_mv_*`` track against UT with the
    magnitude axis **inverted**, so a brighter sky runs downward the way it does
    on the sky-brightness maps and keograms. Astronomical twilight (Sun above
    -18 deg) is shaded, which is where the near-vertical ramps at both ends come
    from -- the night selection is Sun < -12, so a run always includes some
    twilight. The lower panel carries the Moon's altitude and, on a right-hand
    axis, the altitude at which the floating darkest cone was found.

    The title reports the night's dark-sky medians, computed over frames with
    the Sun below -18 deg **and** the Moon below the horizon, so a moonlit
    stretch cannot drag them; a night with no such frames simply omits them.

    ``output_file`` defaults to the input path with a ``.png`` suffix (so the
    standard ``sky_brightness.csv`` becomes ``sky_brightness.png`` beside it),
    and its extension picks the matplotlib backend. Returns that path.
    """
    filename = Path(filename)
    if output_file is None:
        output_file = filename.with_suffix(".png")
    output_file = Path(output_file)

    df = pd.read_csv(filename, parse_dates=["OBSTIME"]).sort_values("OBSTIME")
    if df.empty:
        raise ValueError(f"{filename} contains no rows to plot")
    times = df["OBSTIME"]

    fig, (ax, axm) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.06},
    )

    twilight = df["sun_alt"] > -18
    for axis in (ax, axm):
        axis.fill_between(times, 0, 1, where=twilight,
                          transform=axis.get_xaxis_transform(),
                          color="0.85", zorder=0, lw=0)

    for column, (label, color, lw) in ALCOR_SB_SUMMARY_STYLE.items():
        if column in df:
            ax.plot(times, df[column], color=color, lw=lw,
                    label=_sb_summary_label(column, label))

    # ALCOR_SB_TARGET_UNIT is the CSV's plain-text unit; the axis wants mathtext.
    ax.set_ylabel("sky surface brightness  [V mag arcsec$^{-2}$]")
    ax.invert_yaxis()
    ax.grid(alpha=0.25)

    # Anchor the legend in DATA coordinates via the xaxis transform, just left
    # of the morning twilight band. An axes-fraction anchor is wrong by the 5%
    # x-margins and lands on top of the shading.
    lo, hi = ax.get_xlim()
    pad = 0.015 * (hi - lo)
    right = hi - pad
    morning = times[twilight & (times > times.iloc[len(times) // 2])]
    if len(morning):
        right = min(right, mdates.date2num(morning.iloc[0]) - pad)
    ax.legend(loc="upper right", bbox_to_anchor=(right, 0.97),
              bbox_transform=ax.get_xaxis_transform(),
              framealpha=0.9, fontsize=9)

    if title is None:
        title = filename.parent.name
    heading = f"{title}   n={len(df)} frames"
    dark = df[(df["sun_alt"] < -18) & (df["moon_alt"] < 0)]
    if len(dark):
        # Only the two sky tracks: the light domes are not a darkness measure,
        # and listing all four overflows the title.
        medians = ", ".join(
            f"{name} {dark[column].median():.2f}"
            for column, name in (("allsky_mv_zenith", "zenith"),
                                 ("allsky_mv_best", "darkest cone"))
            if column in dark and np.isfinite(dark[column].median())
        )
        if medians:
            heading += f"   dark-sky median: {medians} mag arcsec$^{{-2}}$"
    ax.set_title(heading)

    axm.plot(times, df["moon_alt"], color="#6a51a3", lw=1.3)
    axm.axhline(0, color="k", lw=0.6, ls=":")
    axm.set_ylabel("moon alt [deg]", fontsize=9)
    axm.grid(alpha=0.25)

    if "best_alt" in df:
        axb = axm.twinx()
        axb.scatter(times, df["best_alt"], s=3, color="#2ca02c", alpha=0.5)
        axb.set_ylabel("darkest-cone alt [deg]", fontsize=9, color="#2ca02c")
        axb.tick_params(axis="y", labelcolor="#2ca02c", labelsize=8)
        axb.set_ylim(0, 95)

    axm.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    axm.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axm.set_xlabel("UT")

    fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_file
