import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.dates as mdates
from astropy.io import fits
from astropy.time import Time
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
        stem = str(filename)
        for ext in (".fits.bz2", ".fits.gz", ".fits"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        output_file = stem + "_proc.fits"
    output_file = Path(output_file)

    cube = np.transpose(np.flipud(im), axes=(2, 0, 1)).astype(np.float32)
    hdu = fits.PrimaryHDU(data=cube, header=wcs.to_header())
    hdu.writeto(output_file, overwrite=overwrite)
    return output_file


def alcor_keogram(input_dir, pattern="*.fits.bz2", workers=1, progress=False, progress_file=None, **kwargs):
    """
    Build a keogram from a directory of alcor OMEA 8C FITS images.

    Each input image is loaded with `load_alcor_fits`, and the center column
    of the processed RGB image is copied into the next column of the keogram.
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
        Forwarded to `load_alcor_fits` (rotation, xcen, ycen, radius,
        horizon_radius).

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

    im, _ = load_alcor_fits(filename, **kwargs)
    center_column = im.shape[1] // 2
    return index, timestamp, im[:, center_column, :], filename.name


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


def alcor_keogram_cli():
    """
    CLI entry point for `alcor_keogram`. Writes a timestamp-labeled keogram
    figure and, optionally, the DATE header values used for the x-axis.
    """
    parser = argparse.ArgumentParser(
        description="Build a keogram from the center columns of alcor OMEA 8C FITS images."
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
    parser.add_argument("--rotation", type=float, default=0.4, help="Camera rotation w.r.t. true north (deg).")
    parser.add_argument("--xcen", type=int, default=696, help="X center of illuminated region.")
    parser.add_argument("--ycen", type=int, default=698, help="Y center of illuminated region.")
    parser.add_argument("--radius", type=int, default=680, help="Half-width of trimmed square around (xcen, ycen).")
    parser.add_argument("--horizon-radius", type=float, default=662, help="Pixel radius from zenith at altitude=0.")
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
        rotation=args.rotation,
        xcen=args.xcen,
        ycen=args.ycen,
        radius=args.radius,
        horizon_radius=args.horizon_radius,
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
        description="Render a timestamp-labeled keogram plot from an alcor keogram FITS file."
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
