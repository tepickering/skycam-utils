"""Rendering annotated all-sky RGB images."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import astropy.visualization as viz

from .io import _corner_bias, load_alcor_fits


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



def _alcor_zenith_crop_bounds(wcs, ny, nx, radius):
    """
    Integer crop bounds for a ``radius``-pixel square around the WCS zenith.

    Returns ``(xz, yz, xl, xu, yl, yu)``: the 0-based zenith pixel and the crop
    slice limits clipped to the ``(ny, nx)`` image.
    """
    zx, zy = wcs.world_to_pixel_values(0.0, 90.0)
    xz = int(round(float(zx)))
    yz = int(round(float(zy)))
    yl, yu = max(0, yz - radius), min(ny, yz + radius)
    xl, xu = max(0, xz - radius), min(nx, xz + radius)
    return xz, yz, xl, xu, yl, yu



def _add_alcor_alt_az_grid(fig, wcs, xz, yz, radius, color="white",
                           label_color="0.5", grid_color=None,
                           grid_alpha=0.5):
    """
    Overlay an altitude/azimuth polar grid on an alcor crop figure.

    ``(xz, yz)`` is the zenith pixel and ``radius`` the crop half-width, so r=1
    on the polar axes is ``radius`` pixels from the zenith. Altitude-ring radii
    are mapped through the WCS so they track the real (distorted) geometry. The
    polar axes is added at the default 111 position, matching a full-figure image
    axes created by ``plt.subplots``. ``color`` is the altitude-label colour (the
    rings sit inside the FOV), ``label_color`` the azimuth-label colour (those sit
    outside the FOV), and ``grid_color`` (when given) the grid-line colour.
    """
    pax = fig.add_subplot(111, polar=True, label="polar")
    pax.set_facecolor("None")
    pax.set_theta_zero_location("N")
    tick_alts = np.array([75, 60, 45, 30, 15])
    px, py = wcs.world_to_pixel_values(np.zeros_like(tick_alts, dtype=float),
                                       tick_alts.astype(float))
    yticks = np.hypot(px - xz, py - yz) / radius
    ylabels = [f" {a}°" for a in tick_alts]
    pax.set_yticks(yticks, labels=ylabels, color=color, alpha=0.5, fontsize=16)
    pax.set_rlabel_position(90)
    grid_kw = {"grid_alpha": grid_alpha}
    if grid_color is not None:
        grid_kw["grid_color"] = grid_color
    pax.tick_params(**grid_kw)
    pax.tick_params(axis="x", labelsize=16, labelcolor=label_color, pad=10)
    return pax



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

    ny, nx = rgb.shape[:2]
    xz, yz, xl, xu, yl, yu = _alcor_zenith_crop_bounds(wcs, ny, nx, radius)
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

    # Altitude/azimuth polar overlay (r=1 -> `radius` px from zenith).
    _add_alcor_alt_az_grid(fig, wcs, xz, yz, radius)

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
