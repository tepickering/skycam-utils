"""Console-script entry points for the Alcor tools."""

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import (
    ALCOR_BADPIX_POLE_RADIUS, ALCOR_BADPIX_RIM_DILATION,
    ALCOR_NONLINEAR_THRESHOLD, ALCOR_RADIUS, ALCOR_SATURATION,
    ALCOR_SB_APERTURE_RADIUS, ALCOR_SB_BEST_MIN_ALTITUDE
)
from .timeutils import _alcor_frame_calibration
from .wcs import build_alcor_wcs
from .wcsfit import (
    _format_calibration_entry, fit_alcor_wcs, save_alcor_residual_plot
)
from .badpix import create_badpix_mask
from .horizon import alcor_median_stack, create_horizon_mask
from .io import alcor_proc_fits
from .photometry import (
    _default_alcor_photometry_check_plot_output, alcor_star_photometry
)
from .display import plot_alcor_fits
from .skybright import alcor_sky_brightness_fits, plot_alcor_sky_brightness
from .keogram import (
    _keogram_row_altitude, alcor_keogram, plot_alcor_keogram_fits,
    plot_alcor_sb_keogram_fits, save_alcor_keogram_fits,
    save_alcor_keogram_plot
)
from .night import alcor_process_night
from .archive import (
    ALCOR_ARCHIVE_MAX_CONSECUTIVE_FAILURES, ALCOR_ARCHIVE_MIN_AGE,
    alcor_archive_status, alcor_process_archive
)


def alcor_proc_fits_cli():
    """
    CLI entry point for `alcor_proc_fits`. Writes a processed FITS file with
    the alt/az WCS encoded in the header.
    """
    parser = argparse.ArgumentParser(
        description="Process an alcor OMEA 8C FITS image into a raw (3, ny, nx) FITS cube with raw-frame alt/az WCS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument("-o", "--output", default=None, help="Output FITS path (default: <input>_proc.fits).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")
    args = parser.parse_args()

    out = alcor_proc_fits(
        args.filename,
        output_file=args.output,
        overwrite=args.overwrite,
    )
    print(out)



def alcor_keogram_cli():
    """
    CLI entry point for `alcor_keogram`. Writes a timestamp-labeled keogram
    figure and, optionally, the DATE header values used for the x-axis.
    """
    parser = argparse.ArgumentParser(
        description="Build a keogram from the center columns of alcor OMEA 8C FITS images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    keogram, timestamps, used = alcor_keogram(
        input_dir,
        pattern=args.pattern,
        workers=args.workers,
        progress=not args.no_progress,
    )
    row_altitude = None
    if used:
        cal = _alcor_frame_calibration(used[0])
        wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                              rotation=cal["rotation"],
                              radial_coeffs=cal["radial_coeffs"],
                              horizon_radius=cal["horizon_radius"],
                              tangential_coeffs=cal["tangential_coeffs"],
                              axis_tilt=cal["axis_tilt"])
        zx, _ = wcs.world_to_pixel_values(0.0, 90.0)
        row_altitude = _keogram_row_altitude(
            wcs, keogram.shape[0], int(round(float(zx))))
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
        altitude=row_altitude,
    )
    fits_output = save_alcor_keogram_fits(
        keogram,
        timestamps,
        fits_output,
        overwrite=True,
        altitude=row_altitude,
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
        description="Render a timestamp-labeled keogram plot from an alcor keogram FITS file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        description="Render an annotated all-sky figure from an alcor OMEA 8C FITS image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument(
        "-o", "--outfig", default=None,
        help="Output figure path (default: <input>.pdf). Format inferred from extension."
    )
    parser.add_argument("--outimage", default=None, help="If set, also write the raw stretched image to this path.")
    parser.add_argument("--radius", type=int, default=680,
                        help="Half-width (pixels) of the display crop around the zenith.")
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
        radius=args.radius,
        powerstretch=args.powerstretch,
        contrast=args.contrast,
        gscale=args.gscale,
        bscale=args.bscale,
        figsize=args.figsize,
    )
    print(outfig)



def plot_alcor_sky_brightness_cli():
    """
    CLI entry point for `plot_alcor_sky_brightness`. Writes a V mag/arcsec^2
    sky-brightness map, named after the input file with the extension replaced
    by `.pdf` unless `-o` is given.
    """
    parser = argparse.ArgumentParser(
        description="Render an alcor OMEA 8C frame as a V mag/arcsec^2 sky-surface-brightness map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument(
        "-o", "--outfig", default=None,
        help="Output figure path (default: <input>.pdf). Format inferred from extension."
    )
    parser.add_argument("--outimage", default=None,
                        help="If set, also write the colour-mapped surface-brightness image to this path.")
    parser.add_argument("--radius", type=int, default=ALCOR_RADIUS,
                        help="Half-width (pixels) of the display crop around the zenith.")
    parser.add_argument("--fov-altitude", type=float, default=-2.0,
                        help="Mask pixels below this altitude (deg). Ignored with --horizon-mask.")
    parser.add_argument("--horizon-mask", action="store_true",
                        help="Mask non-sky with the full horizon/obstruction mask instead of the altitude cutoff.")
    parser.add_argument("--saturation", type=int, default=None,
                        help="Blank raw G pixels at or above this ADU level (clipped/non-linear). OFF by default: a blanked pixel renders as background, so the brightest sources would appear as dark holes.")
    parser.add_argument("--vmin", type=float, default=None, help="Colorbar lower limit (mag/arcsec^2).")
    parser.add_argument("--vmax", type=float, default=None, help="Colorbar upper limit (mag/arcsec^2).")
    parser.add_argument("--cmap", default="cividis_r", help="Matplotlib colormap.")
    parser.add_argument("--figsize", type=float, default=12, help="Matplotlib figure size in inches.")
    args = parser.parse_args()

    outfig = args.outfig
    if outfig is None:
        stem = str(args.filename)
        for ext in (".fits.bz2", ".fits.gz", ".fits"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        outfig = stem + "_skybright.pdf"

    plot_alcor_sky_brightness(
        args.filename,
        outimage=args.outimage,
        outfig=outfig,
        radius=args.radius,
        fov_altitude=args.fov_altitude,
        horizon_mask=args.horizon_mask,
        saturation=args.saturation,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        figsize=args.figsize,
    )
    print(outfig)



def alcor_sky_brightness_cli():
    """
    CLI entry point for `alcor_sky_brightness_fits`. Writes a calibrated
    V mag/arcsec^2 FITS data product with the raw-frame alt/az WCS attached,
    named after the input file with `_sb.fits` unless `-o` is given.
    """
    parser = argparse.ArgumentParser(
        description="Calibrate an alcor OMEA 8C frame to a V mag/arcsec^2 FITS image with the raw-frame alt/az WCS attached.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output FITS path (default: <input>_sb.fits).")
    parser.add_argument("--horizon-mask", action="store_true",
                        help="Also blank the full horizon/obstruction mask (default blanks only off-frame + saturated pixels).")
    parser.add_argument("--saturation", type=int, default=None,
                        help="Blank raw G pixels at or above this ADU level (clipped/non-linear). OFF by default: a blanked pixel renders as background, so the brightest sources would appear as dark holes.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it exists.")
    args = parser.parse_args()

    out = alcor_sky_brightness_fits(
        args.filename,
        output_file=args.output,
        horizon_mask=args.horizon_mask,
        saturation=args.saturation,
        overwrite=args.overwrite,
    )
    print(out)



def plot_alcor_sb_keogram_fits_cli():
    """
    CLI entry point for :func:`plot_alcor_sb_keogram_fits`, so a saved
    sky-brightness keogram can be re-rendered at different colour limits without
    reprocessing the night.
    """
    parser = argparse.ArgumentParser(
        description="Plot a calibrated alcor sky-brightness keogram FITS file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input sky-brightness keogram FITS file.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output plot path (default: the input with a .png suffix).")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Colorbar lower limit (mag/arcsec^2); default is the 1st percentile.")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Colorbar upper limit (mag/arcsec^2); default is the 99th percentile.")
    parser.add_argument("--cmap", default="cividis_r", help="Matplotlib colormap.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(12, 6),
                        metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--dpi", type=int, default=150, help="Output figure resolution.")
    args = parser.parse_args()

    out = plot_alcor_sb_keogram_fits(
        args.filename,
        output_file=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )
    print(out)



def alcor_process_night_cli():
    """
    CLI entry point for :func:`alcor_process_night`: process one archived night
    into per-frame photometry, the ``sky_brightness.csv`` summary, and a
    calibrated nighttime keogram.
    """
    parser = argparse.ArgumentParser(
        description="Process one archived night of alcor frames into star photometry, a sky-brightness summary, and a calibrated keogram.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("night_dir", help="Directory of one night's alcor frames.")
    parser.add_argument("-o", "--out-dir", default=None,
                        help="Directory for all products (default: alongside the frames).")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob for input frames.")
    parser.add_argument("--sun-alt-max", type=float, default=-12.0,
                        help="Night is the Sun below this altitude (deg). No Moon cut is applied.")
    parser.add_argument("--sb-aperture-radius", type=float,
                        default=ALCOR_SB_APERTURE_RADIUS,
                        help="Angular radius of each sky-brightness sampling cone (deg).")
    parser.add_argument("--best-min-altitude", type=float,
                        default=ALCOR_SB_BEST_MIN_ALTITUDE,
                        help="Altitude floor for the allsky_mv_best darkest-cone search (deg).")
    parser.add_argument("--no-horizon-mask", action="store_true",
                        help="Do not blank not-sky pixels in the maps, cones, and keogram.")
    parser.add_argument("--sb-saturation", type=int, default=None,
                        help="Blank raw G pixels at or above this ADU level (clipped/non-linear). OFF by default: a blanked pixel renders as background, so the brightest sources would appear as dark holes.")
    parser.add_argument("--write-sb-fits", action="store_true",
                        help="Also keep each frame's full surface-brightness map as <frame>_sb.fits.")
    parser.add_argument("--median-stack", action="store_true",
                        help="Also build the per-channel raw median stack (<night>_median.fits) for bad-pixel tracking.")
    parser.add_argument("--day-keogram", action="store_true",
                        help="Also build the full-day raw RGB keogram (<night>_keogram.fits/.png), daylight included.")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-measure star photometry even where <frame>_phot.csv already exists.")
    parser.add_argument("--aperture-radius", type=float, default=4.0,
                        help="Star aperture radius in pixels (also the Gaussian fit window).")
    parser.add_argument("--annulus-width", type=float, default=1.0,
                        help="Star background annulus width in pixels.")
    parser.add_argument("--min-altitude", type=float, default=20.0,
                        help="Minimum catalog-star altitude to measure (deg).")
    parser.add_argument("--vmag-limit", type=float, default=5.5,
                        help="Faintest catalog star Vmag to measure.")
    parser.add_argument("--no-variables", dest="variables", action="store_false",
                        help="Do not measure the bright-variable catalog "
                             "(bright_variable_vsx.fits) alongside the "
                             "calibration stars.")
    parser.add_argument("--gaussian", action="store_true",
                        help="Use constrained-Gaussian PSF photometry instead of apertures.")
    parser.add_argument("--both", action="store_true",
                        help="Measure aperture AND Gaussian photometry (overrides --gaussian).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Strided-subsample the night to at most this many frames.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Directory for the median-stack scratch memmap.")
    parser.add_argument("--masks-dir", default=None,
                        help="Override the bad-pixel masks directory.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes (default: one per core; 1 runs serially).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing per-frame output files.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not report progress to stderr.")
    args = parser.parse_args()

    def log(message):
        print(message, file=sys.stderr)

    result = alcor_process_night(
        args.night_dir,
        out_dir=args.out_dir,
        pattern=args.pattern,
        sun_alt_max=args.sun_alt_max,
        sb_aperture_radius=args.sb_aperture_radius,
        best_min_altitude=args.best_min_altitude,
        horizon_mask=not args.no_horizon_mask,
        sb_saturation=args.sb_saturation,
        write_sb_fits=args.write_sb_fits,
        median_stack=args.median_stack,
        day_keogram=args.day_keogram,
        reprocess=args.reprocess,
        max_frames=args.max_frames,
        scratch_dir=args.scratch_dir,
        masks_dir=args.masks_dir,
        workers=args.workers,
        overwrite=args.overwrite,
        log=None if args.quiet else log,
        aperture_radius=args.aperture_radius,
        annulus_width=args.annulus_width,
        min_altitude=args.min_altitude,
        vmag_limit=args.vmag_limit,
        variables=args.variables,
        gaussian=args.gaussian,
        both=args.both,
    )
    for path in ("summary_file", "keogram_file", "keogram_plot",
                 "day_keogram_file", "day_keogram_plot",
                 "photometry_file", "median_file"):
        if result[path] is not None:
            print(result[path])
    if result["errors"]:
        print(f"# {len(result['errors'])} frames failed", file=sys.stderr)



def alcor_process_archive_cli():
    """
    CLI entry point for :func:`alcor_process_archive`: run every night of an
    archive through the night driver, resumably.
    """
    parser = argparse.ArgumentParser(
        description="Process every night of a skycam archive into star "
                    "photometry, sky-brightness summaries, and keograms. "
                    "Resumable: rerun the same command to continue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("archive_dir", nargs="?", default=None,
                        help="Archive root holding one directory per night. "
                             "Optional only with --status.")
    parser.add_argument("-o", "--out-dir", required=True,
                        help="Products tree. All state lives in "
                             "<out-dir>/.archive_state/ (ledger.json, PAUSE, "
                             "archive.log); each night writes to <out-dir>/<night>/.")
    parser.add_argument("--status", action="store_true",
                        help="Report the run's progress from the ledger and exit. "
                             "Does not need the archive mounted.")
    parser.add_argument("--start", default=None,
                        help="Earliest night to process (YYYY-MM-DD, inclusive).")
    parser.add_argument("--end", default=None,
                        help="Latest night to process (YYYY-MM-DD, inclusive).")
    parser.add_argument("--nights", nargs="+", default=None,
                        help="Explicit night directory names instead of scanning.")
    parser.add_argument("--reverse", action="store_true",
                        help="Process newest nights first.")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob for input frames.")
    parser.add_argument("--min-age", type=float, default=ALCOR_ARCHIVE_MIN_AGE,
                        help="Hours a night must be unchanged before it is "
                             "processed. The archive syncs from the camera host, "
                             "so a night modified more recently may still be "
                             "arriving. 0 disables the rule.")
    parser.add_argument("--prune-jpegs", action="store_true",
                        help="DELETE each night's vendor JPEGs once its products "
                             "are written and verified. Irreversible; frees "
                             "~7.3 GB per night.")
    parser.add_argument("--prune-dry-run", action="store_true",
                        help="Report what --prune-jpegs would free, deleting "
                             "nothing. Overrides --prune-jpegs.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt nights previously recorded as failed.")
    parser.add_argument("--force-options", action="store_true",
                        help="Continue even though the photometry options differ "
                             "from the ones this ledger was started with. The "
                             "combined photometry will mix schemas.")
    parser.add_argument("--max-consecutive-failures", type=int,
                        default=ALCOR_ARCHIVE_MAX_CONSECUTIVE_FAILURES,
                        help="Abort after this many nights fail in a row (an "
                             "unmounted archive fails every one instantly).")
    parser.add_argument("--verbose", action="store_true",
                        help="Log every per-frame line instead of a throttled "
                             "progress line each minute.")
    # Forwarded to alcor_process_night.
    parser.add_argument("--sun-alt-max", type=float, default=-12.0,
                        help="Night is the Sun below this altitude (deg).")
    parser.add_argument("--day-keogram", action="store_true",
                        help="Also build each night's full-day raw RGB keogram.")
    parser.add_argument("--median-stack", action="store_true",
                        help="Also build each night's raw median stack.")
    parser.add_argument("--write-sb-fits", action="store_true",
                        help="Keep each frame's full surface-brightness map.")
    parser.add_argument("--no-horizon-mask", action="store_true",
                        help="Do not blank not-sky pixels.")
    parser.add_argument("--reprocess", action="store_true",
                        help="Re-measure star photometry even where "
                             "<frame>_phot.csv already exists.")
    parser.add_argument("--aperture-radius", type=float, default=4.0,
                        help="Star aperture radius in pixels.")
    parser.add_argument("--annulus-width", type=float, default=1.0,
                        help="Star background annulus width in pixels.")
    parser.add_argument("--min-altitude", type=float, default=20.0,
                        help="Minimum catalog-star altitude to measure (deg).")
    parser.add_argument("--vmag-limit", type=float, default=5.5,
                        help="Faintest catalog star Vmag to measure.")
    parser.add_argument("--no-variables", dest="variables", action="store_false",
                        help="Do not measure the bright-variable catalog.")
    parser.add_argument("--gaussian", action="store_true",
                        help="Use constrained-Gaussian PSF photometry.")
    parser.add_argument("--both", action="store_true",
                        help="Measure aperture AND Gaussian photometry.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Directory for the median-stack scratch memmap. "
                             "Point this at local storage, not the archive.")
    parser.add_argument("--masks-dir", default=None,
                        help="Override the bad-pixel masks directory.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes per night (default: one per core).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Strided-subsample each night to at most this many "
                             "frames. For smoke tests, not production runs.")
    args = parser.parse_args()

    if args.status:
        alcor_archive_status(args.out_dir)
        return
    if args.archive_dir is None:
        parser.error("archive_dir is required unless --status is given")

    alcor_process_archive(
        args.archive_dir, out_dir=args.out_dir, start=args.start, end=args.end,
        nights=args.nights, reverse=args.reverse, pattern=args.pattern,
        min_age=args.min_age, prune_jpegs=args.prune_jpegs,
        prune_dry_run=args.prune_dry_run, retry_failed=args.retry_failed,
        force_options=args.force_options,
        max_consecutive_failures=args.max_consecutive_failures,
        verbose=args.verbose, log=lambda message: print(message, file=sys.stderr),
        install_signals=True,
        sun_alt_max=args.sun_alt_max, day_keogram=args.day_keogram,
        median_stack=args.median_stack, write_sb_fits=args.write_sb_fits,
        horizon_mask=not args.no_horizon_mask, reprocess=args.reprocess,
        aperture_radius=args.aperture_radius, annulus_width=args.annulus_width,
        min_altitude=args.min_altitude, vmag_limit=args.vmag_limit,
        variables=args.variables, gaussian=args.gaussian, both=args.both,
        scratch_dir=args.scratch_dir, masks_dir=args.masks_dir,
        workers=args.workers, max_frames=args.max_frames,
    )


def alcor_star_photometry_cli():
    """
    CLI entry point for ``alcor_star_photometry``. Writes fixed-position
    aperture photometry for Vmag-limited named bright stars.
    """
    parser = argparse.ArgumentParser(
        description="Measure Alcor RGB aperture photometry for named bright stars at WCS-predicted positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("filename", help="Input alcor FITS file.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path (default: <input>_phot.csv).",
    )
    parser.add_argument("--aperture-radius", type=float, default=4.0,
                        help="Circular aperture radius in pixels (also the Gaussian fit window).")
    parser.add_argument("--annulus-width", type=float, default=1.0,
                        help="Background annulus width in pixels.")
    parser.add_argument("--min-altitude", type=float, default=20.0,
                        help="Minimum catalog-star altitude in degrees.")
    parser.add_argument("--vmag-limit", type=float, default=5.5,
                        help="Faintest V magnitude to measure.")
    parser.add_argument("--no-refraction", action="store_true",
                        help="Disable atmospheric refraction in the catalog Alt/Az transform.")
    parser.add_argument("--masks-dir", default=None,
                        help="Bad-pixel mask directory (default: $ALCOR_BADPIX_DIR, then packaged masks).")
    parser.add_argument("--sun-alt-max", type=float, default=-12.0,
                        help="Reject images with Sun altitude greater than this (deg).")
    parser.add_argument("--saturation", type=float, default=ALCOR_SATURATION,
                        help="Raw-ADU level at/above which an aperture pixel flags the channel saturated.")
    parser.add_argument("--no-variables", dest="variables", action="store_false",
                        help="Do not measure the bright-variable catalog "
                             "(bright_variable_vsx.fits) alongside the "
                             "calibration stars.")
    parser.add_argument("--gaussian", action="store_true",
                        help="Use constrained-Gaussian PSF photometry instead of aperture sums.")
    parser.add_argument("--both", action="store_true",
                        help="Measure both aperture and Gaussian in one pass into a single "
                             "combined CSV (columns suffixed _ap / _gauss). Overrides --gaussian.")
    parser.add_argument("--mask-threshold", type=float,
                        default=ALCOR_NONLINEAR_THRESHOLD,
                        help="Raw-ADU level at/above which a pixel is excluded from the Gaussian fit.")
    parser.add_argument("--check-plot", action="store_true",
                        help="Write an aperture-overlay check plot as <input>_phot.pdf.")
    parser.add_argument("--check-radius", type=int, default=680,
                        help="Half-width in pixels of the check-plot crop around the zenith.")
    args = parser.parse_args()

    _, output_file = alcor_star_photometry(
        args.filename,
        output_file=args.output,
        aperture_radius=args.aperture_radius,
        annulus_width=args.annulus_width,
        min_altitude=args.min_altitude,
        vmag_limit=args.vmag_limit,
        variables=args.variables,
        refraction=not args.no_refraction,
        masks_dir=args.masks_dir,
        check_plot=args.check_plot,
        check_radius=args.check_radius,
        sun_alt_max=args.sun_alt_max,
        saturation=args.saturation,
        gaussian=args.gaussian,
        mask_threshold=args.mask_threshold,
        both=args.both,
    )
    if output_file is not None:
        print(output_file)
    if args.check_plot and output_file is not None:
        print(_default_alcor_photometry_check_plot_output(args.filename))



def fit_alcor_wcs_cli():
    """
    CLI entry point for ``fit_alcor_wcs``. Aggregates bright-star matches across
    the dark-sky frames of a night and prints the refined geometry constants
    ready to paste into the module defaults.
    """
    parser = argparse.ArgumentParser(
        description="Calibrate the alcor lens WCS from bright stars across a night.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory containing alcor FITS images.")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob pattern for input files.")
    parser.add_argument("--vmag-limit", type=float, default=4.0, help="Faintest Vmag to use.")
    parser.add_argument("--sun-alt-max", type=float, default=-18.0,
                        help="Use frames with Sun altitude below this (deg).")
    parser.add_argument("--moon-alt-max", type=float, default=-6.0,
                        help="Use frames with Moon altitude below this (deg); "
                             "moonlight scatter corrupts source detection. "
                             "Pass 90 to disable the Moon cut.")
    parser.add_argument("--min-alt", type=float, default=10.0, help="Minimum star altitude (deg).")
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="Final (tightest) match tolerance in pixels; the matcher "
                             "tightens to this from --tolerance-start over --match-rounds rounds.")
    parser.add_argument("--tolerance-start", type=float, default=12.0,
                        help="Initial (loosest) match tolerance in pixels.")
    parser.add_argument("--match-rounds", type=int, default=4,
                        help="Number of tightening rounds from --tolerance-start to --tolerance.")
    parser.add_argument("--pattern-tol", type=float, default=3.0,
                        help="Asterism corroboration tolerance in pixels: how closely a "
                             "neighbor's offset must match the local constellation. Loosen "
                             "to keep matches where real distortion bends the local pattern.")
    parser.add_argument("--min-corroborating", type=int, default=2,
                        help="Minimum neighbors that must corroborate a match's local pattern.")
    parser.add_argument("--n-neighbors", type=int, default=5,
                        help="Nearest catalog neighbors checked in asterism verification.")
    parser.add_argument("--fit-k5", action="store_true",
                        help="Also fit the odd quintic radial term k5 (richer radial "
                             "distortion model; use only on a clean, full-zenith match set).")
    parser.add_argument("--max-detections", type=int, default=200,
                        help="Keep only the brightest N detections per frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap number of frames used.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes for per-frame detection "
                             "(default: one per available core).")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print per-file processing/rejection messages.")
    parser.add_argument("--residual-plot", default=None, help="Optional residual-vs-zenith PNG path.")
    args = parser.parse_args()

    log = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    result = fit_alcor_wcs(
        args.input_dir, pattern=args.pattern, vmag_limit=args.vmag_limit,
        sun_alt_max=args.sun_alt_max, moon_alt_max=args.moon_alt_max,
        min_alt=args.min_alt, tolerance=args.tolerance,
        tolerance_start=args.tolerance_start, match_rounds=args.match_rounds,
        n_neighbors=args.n_neighbors, min_corroborating=args.min_corroborating,
        pattern_tol=args.pattern_tol, fit_k5=args.fit_k5,
        max_detections=args.max_detections,
        max_frames=args.max_frames, workers=args.workers, log=log,
    )
    print(f"# matched stars: {result['n_matched']}")
    print(f"# residual RMS (pix): {result['residual_rms']:.3f}")
    print(f"# matched fraction: {result['matched_fraction']:.3f}")
    tn, te = result.get("axis_tilt", (0.0, 0.0))
    eps = float(np.hypot(tn, te))
    a0 = float(np.degrees(np.arctan2(te, tn))) % 360.0
    print(f"# axis tilt: eps={eps:.4f} deg toward az={a0:.1f} deg")
    print("# add this entry to ALCOR_CALIBRATIONS in alcor.py:")
    print(_format_calibration_entry(result))
    if args.residual_plot is not None:
        out = save_alcor_residual_plot(result["alt"], result["az"], result["x"],
                                       result["y"], result, args.residual_plot)
        print(out)



def create_badpix_mask_cli():
    """
    CLI entry point for :func:`create_badpix_mask` (run daily from cron).
    """
    parser = argparse.ArgumentParser(
        description="Build a date-stamped alcor bad-pixel mask from a night of frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("day_dir", help="Directory of one night's alcor frames.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: $ALCOR_BADPIX_DIR or packaged data/badpix).")
    parser.add_argument("--min-frames", type=int, default=500,
                        help="Minimum dark frames required to generate a mask.")
    parser.add_argument("--z-thresh", type=float, default=25.0,
                        help="Robust-sigma threshold for a hot pixel.")
    parser.add_argument("--ksize", type=int, default=5, help="High-pass median kernel (px).")
    parser.add_argument("--rim-dilation", type=int, default=ALCOR_BADPIX_RIM_DILATION,
                        help="Dilate the horizon mask by this many px before "
                             "excluding not-sky from the search; 0 to search "
                             "the whole frame.")
    parser.add_argument("--pole-radius", type=int, default=ALCOR_BADPIX_POLE_RADIUS,
                        help="Radius (px) of the celestial-pole exclusion disc, "
                             "where star trails survive the night median; 0 to "
                             "disable.")
    parser.add_argument("--horizon-dir", default=None,
                        help="Horizon-mask directory (default: $ALCOR_HORIZON_DIR "
                             "or packaged data/horizon).")
    parser.add_argument("--sun-alt-max", type=float, default=-18.0,
                        help="Use frames with Sun altitude below this (deg).")
    parser.add_argument("--moon-alt-max", type=float, default=-6.0,
                        help="Use frames with Moon altitude below this (deg); pass 90 to disable.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap frames used (strided) to bound runtime/scratch.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Directory for the temporary memmap (default: system temp).")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob for input frames.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print per-step progress messages.")
    args = parser.parse_args()

    log = None if args.quiet else (lambda message: print(message, file=sys.stderr))
    out = create_badpix_mask(
        args.day_dir, out_dir=args.out_dir, min_frames=args.min_frames,
        z_thresh=args.z_thresh, ksize=args.ksize, sun_alt_max=args.sun_alt_max,
        moon_alt_max=args.moon_alt_max, rim_dilation=args.rim_dilation,
        pole_radius=args.pole_radius, horizon_dir=args.horizon_dir,
        max_frames=args.max_frames, scratch_dir=args.scratch_dir,
        pattern=args.pattern, log=log)
    if out is None:
        print("# no mask written (insufficient dark frames)")
    else:
        print(out)



def alcor_median_stack_cli():
    """CLI entry point for :func:`alcor_median_stack`."""
    parser = argparse.ArgumentParser(
        description="Median-stack a cloudy night's alcor frames into a luminance image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("night_dir", help="Directory of one (cloudy) night's frames.")
    parser.add_argument("-o", "--out", default=None,
                        help="Output FITS (default: <night-name>_median.fits).")
    parser.add_argument("--sun-alt-max", type=float, default=-18.0,
                        help="Use frames with Sun altitude below this (deg).")
    parser.add_argument("--moon-alt-max", type=float, default=90.0,
                        help="Use frames with Moon altitude below this (deg); 90 disables.")
    parser.add_argument("--no-badpix", action="store_true",
                        help="Do not zero the nearest bad-pixel mask before combining.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap frames used (strided) to bound runtime/scratch.")
    parser.add_argument("--scratch-dir", default=None,
                        help="Directory for the temporary memmap (default: system temp).")
    parser.add_argument("--pattern", default="*.fits.bz2", help="Glob for input frames.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print per-step progress messages.")
    args = parser.parse_args()

    log = None if args.quiet else (lambda m: print(m, file=sys.stderr))
    out = alcor_median_stack(
        args.night_dir, out_path=args.out, sun_alt_max=args.sun_alt_max,
        moon_alt_max=args.moon_alt_max, badpix=not args.no_badpix,
        max_frames=args.max_frames, scratch_dir=args.scratch_dir,
        pattern=args.pattern, log=log)
    print(out)



def create_horizon_mask_cli():
    """CLI entry point for :func:`create_horizon_mask`."""
    parser = argparse.ArgumentParser(
        description="Build a date-stamped alcor horizon (sky/not-sky) mask from a cloudy-night median.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("median", help="Cloudy-night luminance median FITS (from alcor_median_stack).")
    parser.add_argument("--epoch", default=None,
                        help="Mask date YYYY-MM-DD (default: parsed from the median filename).")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: $ALCOR_HORIZON_DIR or packaged data/horizon).")
    parser.add_argument("--phot-nights", nargs="+", default=None,
                        help="Dirs of *_phot.csv for the SW->W undetected-star patch "
                             "(omitted: that sector uses Sobel edges only).")
    parser.add_argument("--edge-pct", type=float, default=96.0, help="Sobel-edge wall percentile.")
    parser.add_argument("--edge-dilate", type=int, default=1, help="Wall dilation iterations.")
    parser.add_argument("--open-radius", type=int, default=3, help="Morphological opening radius (px).")
    parser.add_argument("--sector", type=float, nargs=2, default=[225.0, 270.0],
                        metavar=("AZ_LO", "AZ_HI"), help="Undetected-patch azimuth sector (deg).")
    parser.add_argument("--und-thr", type=float, default=0.5, help="Undetected fraction = obstructed.")
    parser.add_argument("--und-mincount", type=int, default=15, help="Min star transits per cell.")
    parser.add_argument("--rim-alt", type=float, default=1.5,
                        help="A not-sky blob reaching below this alt is rim-connected (kept).")
    parser.add_argument("--rod-area-min", type=int, default=400,
                        help="Keep isolated not-sky blobs at least this size (px): the lightning rod.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print per-step progress messages.")
    args = parser.parse_args()

    log = None if args.quiet else (lambda m: print(m, file=sys.stderr))
    out = create_horizon_mask(
        args.median, epoch=args.epoch, out_dir=args.out_dir,
        phot_nights=args.phot_nights, edge_pct=args.edge_pct,
        edge_dilate=args.edge_dilate, open_radius=args.open_radius,
        sector=tuple(args.sector), und_thr=args.und_thr,
        und_mincount=args.und_mincount, rim_alt=args.rim_alt,
        rod_area_min=args.rod_area_min, log=log)
    print(out)
