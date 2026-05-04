import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.wcs import WCS

import astropy.visualization as viz


def load_alcor_fits(filename, rotation=0.4, xcen=696, ycen=698, radius=680, horizon_radius=662):
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
    rotation : float (default=0.4)
        Camera rotation w.r.t. true north, in degrees.
    xcen : int (default=696)
        X center of illuminated region in original image coordinates.
    ycen : int (default=698)
        Y center of illuminated region in original image coordinates.
    radius : int (default=680)
        Half-width of the trimmed square around (xcen, ycen).
    horizon_radius : float (default=662)
        Pixel radius from zenith at which altitude=0.

    Returns
    -------
    im : ndarray
        Zenith-centered, north-up image of shape (2*radius, 2*radius, 3).
    wcs : `astropy.wcs.WCS`
        ARC-projection WCS mapping pixel (x, y) ↔ (azimuth, altitude).
    """
    with fits.open(filename) as hdul:
        data = hdul[0].data
    im = np.transpose(data, axes=(1, 2, 0)) - 2000  # 2000 is a bit above the normal bias level of the camera.
    im[im < 0] = 0
    im = im * 1.0
    xl = xcen - radius
    xu = xcen + radius
    yl = ycen - radius
    yu = ycen + radius
    im = im[yl:yu, xl:xu, :]
    im = np.flipud(rotate(im, rotation, reshape=False))

    cdelt = 90.0 / horizon_radius
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---ARC', 'DEC--ARC']
    wcs.wcs.crpix = [radius + 0.5, radius + 0.5]
    wcs.wcs.crval = [0.0, 90.0]
    wcs.wcs.cdelt = [cdelt, cdelt]
    # Native pole == celestial pole here (CRVAL2=+90), so set LONPOLE explicitly:
    # leaving it to wcslib's default produces a 180° azimuth offset because its
    # default for this degenerate case is 180°, not the spec's 0°.
    wcs.wcs.lonpole = 0.0

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
        output_file = str(filename).replace(".fits", "_proc.fits", 1)
    output_file = Path(output_file)

    cube = np.transpose(np.flipud(im), axes=(2, 0, 1)).astype(np.float32)
    hdu = fits.PrimaryHDU(data=cube, header=wcs.to_header())
    hdu.writeto(output_file, overwrite=overwrite)
    return output_file


def plot_alcor_fits(filename, outimage=None, outfig=None, rotation=0.4, xcen=696, ycen=698, radius=680,
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
    rotation : float (default=0.4)
        Camera rotation w.r.t. true north. Default is empirically from alcor software.
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
        description="Process an alcor OMEA 8C FITS image into a zenith-centered, north-up FITS file with alt/az WCS."
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument("-o", "--output", default=None, help="Output FITS path (default: <input>_proc.fits).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")
    parser.add_argument("--rotation", type=float, default=0.4, help="Camera rotation w.r.t. true north (deg).")
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


def plot_alcor_fits_cli():
    """
    CLI entry point for `plot_alcor_fits`. Writes an annotated PDF figure by
    default, named after the input file with `.fits` replaced by `.pdf`.
    """
    parser = argparse.ArgumentParser(
        description="Render an annotated all-sky figure from an alcor OMEA 8C FITS image."
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument(
        "-o", "--outfig", default=None,
        help="Output figure path (default: <input>.pdf). Format inferred from extension."
    )
    parser.add_argument("--outimage", default=None, help="If set, also write the raw stretched image to this path.")
    parser.add_argument("--rotation", type=float, default=0.4, help="Camera rotation w.r.t. true north (deg).")
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
        outfig = str(args.filename).replace(".fits", ".pdf", 1)

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
