"""Raw RGB and calibrated surface-brightness keograms."""

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.resources import files
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from astropy.io import fits
from astropy.time import Time

import astropy.visualization as viz

from .io import load_alcor_fits


def alcor_keogram(input_dir, pattern="*.fits.bz2", workers=1, progress=False, progress_file=None, **kwargs):
    """
    Build a keogram from a directory of alcor OMEA 8C FITS images.

    Each input image is loaded with `load_alcor_fits`, and the zenith column
    (the column through the WCS alt=90 pixel) of the raw RGB cube is copied
    into the next column of the keogram.
    The DATE header keyword from each FITS file is collected in the same order
    as the keogram columns.

    Parameters
    ----------
    input_dir : str or `~pathlib.Path`
        Directory containing alcor FITS images.
    pattern : str (default ``"*.fits.bz2"``)
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
        Forwarded to `load_alcor_fits` (``wcs``, ``badpix``, ``masks_dir``).

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

    cube, wcs, _ = load_alcor_fits(filename, **kwargs)
    zx, _ = wcs.world_to_pixel_values(0.0, 90.0)
    zcol = int(round(float(zx)))                          # 0-based zenith column
    return index, timestamp, cube[:, :, zcol].T, filename.name  # (ny, 3) RGB strip



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
    altitude=None,
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
    altitude : ndarray or None (default=None)
        Per-row altitude in degrees (:func:`_keogram_row_altitude`). When given,
        the horizon crossings are marked.

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
        ax.imshow(im, aspect="auto", origin="lower")
        _set_keogram_yaxis(ax, keogram.shape[0], altitude=altitude)
    else:
        yedges = np.arange(keogram.shape[0] + 1)
        ax.pcolormesh(timestamp_edges, yedges, im, shading="flat", rasterized=True)
        _set_keogram_yaxis(ax, keogram.shape[0], edges=True, altitude=altitude)
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



def _keogram_row_altitude(wcs, nrows, zcol):
    """
    Altitude in degrees of each row of a keogram column.

    A keogram column is the raw zenith pixel column, so its rows run from the
    south end of the sensor, up through the zenith, to the north end -- crossing
    altitude 0 twice. This is what lets a plotter mark the horizon and scale on
    sky pixels alone.
    """
    rows = np.arange(int(nrows), dtype=float)
    _, alt = wcs.pixel_to_world_values(np.full(rows.shape, float(zcol)), rows)
    return np.asarray(alt, dtype=float)



def _keogram_horizon_rows(altitude):
    """
    Row indices where a keogram column crosses altitude 0, south end first.

    Returns an empty list when ``altitude`` is None or never crosses (a sensor
    whose column stays above the horizon).
    """
    if altitude is None:
        return []
    alt = np.asarray(altitude, dtype=float)
    sky = alt > 0.0
    if not sky.any():
        return []
    edges = np.flatnonzero(np.diff(sky.astype(int)) != 0)
    # +0.5: the crossing lies between the two rows that straddle it.
    return [float(e) + 0.5 for e in edges]



def _set_keogram_yaxis(ax, nrows, edges=False, altitude=None):
    """
    Label a keogram's y axis north-up, with the row index increasing upward.

    A keogram column is the raw zenith pixel column, and :func:`build_alcor_wcs`
    puts north at *increasing* y, so row 0 is the south end and row ``nrows - 1``
    is the north end. The axis must therefore run bottom-to-top as S / Z / N:
    drawing with ``origin="upper"`` or calling ``invert_yaxis`` silently renders
    the keogram upside down (north at the bottom under an "N" label at the top).
    Setting the limits explicitly here keeps that decision in one place.

    ``edges`` selects the tick positions for a ``pcolormesh`` drawn on
    ``nrows + 1`` cell edges rather than an ``imshow`` on ``nrows`` pixel centres.

    When ``altitude`` (per-row degrees, from :func:`_keogram_row_altitude`) is
    given, the two altitude-0 crossings are drawn as dashed lines. The column
    runs past the horizon at both ends -- that band is terrain and the light
    domes above it, and without the marks it is indistinguishable from low sky.
    """
    upper = nrows if edges else nrows - 1
    ax.set_ylim(0, upper)
    ax.set_yticks([0, upper / 2.0, upper])
    ax.set_yticklabels(["S", "Z", "N"])
    for row in _keogram_horizon_rows(altitude):
        ax.axhline(row, color="0.65", ls="--", lw=0.8, alpha=0.8)



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



def _row_altitude_hdu(altitude):
    """
    ``ROWALT`` extension carrying the per-row altitude of a keogram column.

    Written alongside ``TIMESTAMPS`` so a saved keogram is self-describing: a
    plotter can mark the horizon and scale on sky alone without re-resolving the
    calibration. Returns None when there is nothing to write.
    """
    if altitude is None:
        return None
    altitude = np.asarray(altitude, dtype=np.float32)
    column = fits.Column(name="ALTITUDE", format="E", unit="deg", array=altitude)
    return fits.BinTableHDU.from_columns([column], name="ROWALT")



def _load_row_altitude(filename):
    """Per-row altitude from a keogram FITS, or None for a file written without it."""
    with fits.open(filename) as hdul:
        if "ROWALT" not in hdul:
            return None
        return np.asarray(hdul["ROWALT"].data["ALTITUDE"], dtype=float)



def save_alcor_keogram_fits(keogram, timestamps, output_file="keogram.fits",
                            overwrite=False, altitude=None):
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
    altitude : ndarray or None (default=None)
        Per-row altitude in degrees; written as a ``ROWALT`` extension.

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
    rowalt = _row_altitude_hdu(altitude)
    if rowalt is not None:
        hdul.append(rowalt)
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
    kwargs.setdefault("altitude", _load_row_altitude(filename))
    return save_alcor_keogram_plot(keogram, timestamps, output_file, **kwargs)



def save_alcor_sb_keogram_fits(keogram, timestamps, output_file="sb_keogram.fits",
                               overwrite=False, altitude=None):
    """
    Save a calibrated sky-brightness keogram and its timestamps to a FITS file.

    The surface-brightness sibling of :func:`save_alcor_keogram_fits`: the data
    are a single 2-D ``float32`` plane in V mag/arcsec^2 rather than an RGB cube,
    with NaN where the sky brightness could not be measured (off-frame, saturated,
    or a frame that failed to process). The ``TIMESTAMPS`` table extension has the
    same form, so the two products pair column-for-column.

    Parameters
    ----------
    keogram : ndarray
        Sky-brightness keogram of shape (image_height, number_of_images), as
        built by :func:`alcor_process_night`.
    timestamps : sequence of str
        UT timestamps corresponding to the keogram columns.
    output_file : str or `~pathlib.Path` (default="sb_keogram.fits")
        Output FITS filename.
    overwrite : bool (default=False)
        Passed through to `fits.HDUList.writeto`.
    altitude : ndarray or None (default=None)
        Per-row altitude in degrees; written as a ``ROWALT`` extension.

    Returns
    -------
    output_file : `~pathlib.Path`
        Path to the written FITS file.
    """
    output_file = Path(output_file)
    primary = fits.PrimaryHDU(data=np.asarray(keogram, dtype=np.float32))
    primary.header["CTYPE1"] = "TIME"
    primary.header["CTYPE2"] = "OFFSET"
    primary.header["BUNIT"] = ("mag/arcsec2", "observed V surface brightness")

    timestamps = np.asarray(timestamps, dtype=str)
    width = max(1, max(len(timestamp) for timestamp in timestamps))
    columns = [fits.Column(name="DATE", format=f"{width}A", array=timestamps)]
    table = fits.BinTableHDU.from_columns(columns, name="TIMESTAMPS")

    hdul = fits.HDUList([primary, table])
    rowalt = _row_altitude_hdu(altitude)
    if rowalt is not None:
        hdul.append(rowalt)
    hdul.writeto(output_file, overwrite=overwrite)
    return output_file



def load_alcor_sb_keogram_fits(filename):
    """
    Load a sky-brightness keogram FITS file written by
    :func:`save_alcor_sb_keogram_fits`.

    Returns ``(keogram, timestamps)``: the 2-D V mag/arcsec^2 plane and the
    ``DATE`` values from the ``TIMESTAMPS`` extension.
    """
    with fits.open(filename) as hdul:
        keogram = np.asarray(hdul[0].data, dtype=float)
        timestamps = list(hdul["TIMESTAMPS"].data["DATE"])

    return keogram, timestamps



def _sb_keogram_limits(im, altitude, vmin, vmax):
    """
    Fill in missing colour limits for a sky-brightness keogram.

    Percentiles are taken over pixels ABOVE THE HORIZON when ``altitude`` is
    known. The column runs a few degrees below the horizon at both ends, and the
    terrain and light domes there are a couple of magnitudes brighter than sky;
    including them would compress away the sky contrast that makes the keogram
    readable, so they saturate the bright end of the colormap instead.
    """
    if vmin is not None and vmax is not None:
        return vmin, vmax
    scale_from = im
    if altitude is not None:
        sky = np.asarray(altitude, dtype=float) > 0.0
        if sky.any():
            scale_from = im[sky]
    finite = scale_from[np.isfinite(scale_from)]
    if finite.size:
        low, high = np.percentile(finite, [1.0, 99.0])
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax
    return vmin, vmax



def save_alcor_sb_keogram_plot(keogram, timestamps, output_file, vmin=None,
                               vmax=None, cmap="cividis_r", figsize=(12, 6),
                               dpi=150, altitude=None):
    """
    Save a timestamp-labeled plot of a calibrated sky-brightness keogram.

    The surface-brightness sibling of :func:`save_alcor_keogram_plot`. The y axis
    is the same raw zenith column (north at the top, zenith in the middle, south
    at the bottom), so this plot stacks row-for-row against the raw RGB keogram;
    the colour axis is observed V mag/arcsec^2 with a colorbar, and unmeasurable
    pixels are left blank.

    Parameters
    ----------
    keogram : ndarray
        Sky-brightness keogram of shape (image_height, number_of_images).
    timestamps : sequence of str
        UT timestamps corresponding to the keogram columns.
    output_file : str or `~pathlib.Path`
        Output figure filename. The format is inferred from the extension.
    vmin, vmax : float or None (default=None)
        Colour limits in mag/arcsec^2. When None, the 1st and 99th percentiles
        are used (the range varies a lot with moonlight) -- taken over pixels
        ABOVE THE HORIZON when ``altitude`` is given. The column runs several
        degrees below the horizon at both ends, and terrain and the light domes
        there are a couple of magnitudes brighter than sky; letting them into
        the percentile clip would compress away the sky contrast that makes the
        keogram readable. They saturate the bright end instead, which reads
        correctly as "brighter than the scale".
    cmap : str (default="cividis_r")
        Matplotlib colormap; reversed so bright sky reads bright.
    figsize : tuple (default=(12, 6))
        Matplotlib figure size in inches.
    dpi : int (default=150)
        Output figure resolution.
    altitude : ndarray or None (default=None)
        Per-row altitude in degrees (:func:`_keogram_row_altitude`). When given,
        the horizon crossings are marked and the colour scale is set from sky
        pixels alone.

    Returns
    -------
    output_file : `~pathlib.Path`
        Path to the written plot.
    """
    output_file = Path(output_file)
    im = np.asarray(keogram, dtype=float)

    vmin, vmax = _sb_keogram_limits(im, altitude, vmin, vmax)

    times = _parse_timestamps(timestamps)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("0.15")
    if times is None:
        timestamp_edges = None
    else:
        timestamp_edges = _timestamp_edges(mdates.date2num(times))

    if timestamp_edges is None:
        mesh = ax.imshow(im, aspect="auto", origin="lower", cmap=cmap,
                         vmin=vmin, vmax=vmax, interpolation="nearest")
        _set_keogram_yaxis(ax, im.shape[0], altitude=altitude)
        ax.set_xlim(-0.5, im.shape[1] - 0.5)
    else:
        yedges = np.arange(im.shape[0] + 1)
        mesh = ax.pcolormesh(timestamp_edges, yedges, im, shading="flat",
                             cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
        _set_keogram_yaxis(ax, im.shape[0], edges=True, altitude=altitude)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate()
    ax.set_xlabel("UT")

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("V mag/arcsec$^2$")

    fig.tight_layout()
    fig.savefig(output_file, dpi=dpi)
    plt.close(fig)
    return output_file



def plot_alcor_sb_keogram_fits(filename, output_file=None, **kwargs):
    """
    Create a sky-brightness keogram plot from a FITS file written by
    :func:`save_alcor_sb_keogram_fits`. ``output_file`` defaults to the input
    with its FITS suffix replaced by ``.png``; ``**kwargs`` are forwarded to
    :func:`save_alcor_sb_keogram_plot`.
    """
    filename = Path(filename)
    if output_file is None:
        stem = str(filename)
        for ext in (".fits.gz", ".fits"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        output_file = stem + ".png"

    keogram, timestamps = load_alcor_sb_keogram_fits(filename)
    kwargs.setdefault("altitude", _load_row_altitude(filename))
    return save_alcor_sb_keogram_plot(keogram, timestamps, output_file, **kwargs)



def _parse_timestamps(timestamps):
    clean_timestamps = [timestamp for timestamp in timestamps if timestamp]
    if len(clean_timestamps) != len(timestamps):
        return None

    try:
        return Time(timestamps).datetime
    except ValueError:
        return None
