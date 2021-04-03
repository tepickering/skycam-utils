# Licensed under a 3-clause BSD style license - see LICENSE.rst

import datetime
import argparse
import multiprocessing
import warnings

from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd

import astropy
import astropy.units as u
import astropy.wcs as wcs
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.time import Time
from astropy.utils import iers
from astropy.coordinates import AltAz, get_moon, get_sun
from astropy.visualization import ZScaleInterval, SqrtStretch, ImageNormalize

from .photometry import make_background, make_segmentation_image, make_catalog, load_mask, match_stars, load_skycam_catalog
from .astrometry import solve_field, load_wcs, update_altaz, MMT_LOCATION


warnings.filterwarnings('ignore')
iers.conf.auto_download = False
iers.conf.auto_max_age = None


def get_ut(hdr, year=2021):
    """
    When the UT is actually valid in the stellacam image headers, it's one format for
    2011-2012 (and maybe earlier) and another for 2015 onwards.
    """
    if 'FOO' in hdr['UT']:
        return None

    if year < 2013:
        tobs = Time(f"{hdr['DATE']}T{hdr['UT']}", scale='utc')
    else:
        dt = datetime.datetime.strptime(hdr['UT'], "%a %b %d %H:%M:%S %Y")
        tobs = Time(dt, scale='utc')
    return tobs


def stellacam_strip_image(rootdir, writefile=True, outfile=None, compressed=True, year=2021):
    """
    Process a directory of stellacam all-sky images and generate a strip image composed of the center column of
    each all-sky image. This provides a strip chart of the sky as it passes overhead. A FITS HDU list is returned
    and optionally written to disk that consists of the image data, a 2D mask to reject data where the sun is > -18 deg
    or the moon > -10 deg, and an array containing timestamps for each image in matplotlib date2num format.

    Parameters:
        rootdir : str or `~pathlib.PosixPath`
            Directory of stellacam all-sky camera images to process

        writefile : bool
            Write generated HDU list to FITS file or not

        outfile : None or str
            If writefile=True, specify output filename. If none, will default to strip_<rootdir>.fits.

        compressed : bool
            Toggle processing compressed *.fits.gz files if True or *.fits if False.

        year : int (default=2021)
            Year the data was taken. Used to determine how to parse image header.

    Returns:
        hdul : `~astropy.io.fits.HDUList`
    """
    rootdir = Path(rootdir)
    if compressed:
        files = rootdir.glob("*.fits.gz")
    else:
        files = rootdir.glob("*.fits")
    strips = []
    masks = []
    times = []
    for f in files:
        with fits.open(f) as hdul:
            im = hdul[0].data.astype(float)
            hdr = hdul[0].header
            utc = get_ut(hdr, year=year)
            times.append(utc)
            aa_frame = AltAz(obstime=utc, location=MMT_LOCATION)
            moon = get_moon(utc, MMT_LOCATION)
            sun = get_sun(utc)
            moon_aa = moon.transform_to(aa_frame)
            sun_aa = sun.transform_to(aa_frame)
            if sun_aa.alt < 0 * u.deg:  # only process images while the sun is below the horizon
                moon_down = moon_aa.alt < -10 * u.deg
                sun_down = sun_aa.alt < -18 * u.deg
                if hdr['FRAME'] == '256 Frames' and hdr['GAIN'] == 106 and moon_down and sun_down:
                    masks.append(np.zeros(480))
                else:
                    masks.append(np.ones(480))
                strip = np.copy(im[:, 319])
                strips.append(strip)
    st_im = np.flipud(np.swapaxes(np.array(strips), 0, 1))
    st_mask = np.flipud(np.swapaxes(np.array(masks), 0, 1))
    masked = CCDData(st_im, unit="adu", mask=st_mask)
    hdul = masked.to_hdu()
    mt = mdates.date2num(Time(times).to_datetime())
    col = fits.Column(name="mtime", array=mt, format='D')
    hdul.append(fits.BinTableHDU.from_columns([col]))
    if writefile:
        if outfile is None:
            outfile = f"strip_{rootdir}.fits"
        hdul.writeto(outfile, overwrite=True)
    return hdul


def load_strip_image(fitsfile):
    """
    Load strip image data from fitsfile and return CCDData object and array of image observation times.

    Parameters:
        fitsfile : str or `~pathlib.Path`
            Filename of FITS file to load

    Returns:
        (ccd_data, ut_array) : (`~astropy.nddata.CCDData`, `~np.ndarray`)
            CCDData object containing image and mask data, float array containing UT observation times in matplotlib format
    """
    ut_array = np.array(fits.open(fitsfile)[2].data, dtype=float)
    ccd_data = CCDData.read(fitsfile, hdu=0)
    return ccd_data, ut_array


def plot_strip_image(fitsfile, savefile=None, masked=False, cmap='viridis', contrast=0.2, stretch=SqrtStretch()):
    """
    Generate 2D plot of strip image

    Parameters:
        fitsfile : str or `~pathlib.Path`
            FITS file to load strip image data from

        savefile : str or `~pathlib.Path` (default: None)
            Optional filename to write figure to.

        masked : bool (default: False)
            If true, apply mask defining dark sky (moon < -10 deg, sun < -18 deg) before plotting.

        cmap : str (default: 'viridis')
            Colormap to use for 2D data

        contrast : float (default: 0.2)
            Contrast parameter to pass to ZScaleInterval

        stretch : Subclass of `~astropy.visualization.BaseStretch` (default: `~astropy.visualization.SqrtStretch`)
            Stretch class to use for mapping data values to figure output
    """
    ccd_data, ut_array = load_strip_image(fitsfile)
    if masked:
        plot_data = ccd_data
    else:
        plot_data = ccd_data.data
    fig, ax = plt.subplots(figsize=(18, 6))
    ysize = plot_data.data.shape[0]
    norm = ImageNormalize(plot_data, interval=ZScaleInterval(contrast=contrast), stretch=stretch)
    extent = (ut_array[0], ut_array[-1], 0, ysize-1)
    ax.imshow(plot_data, extent=extent, aspect='auto', norm=norm, cmap=cmap)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.set_yticks([0, ysize/2, ysize-1])
    ax.set_yticklabels(['S', 'Z', 'N'])
    ax.set_xlabel("UT")
    dt = mdates.num2date(ut_array[-1])
    ax.set_title(dt.strftime("%Y-%m-%d"))
    if savefile is not None:
        fig.savefig(savefile)
    return fig


def process_asi_image(fitsfile):
    """
    Process a FITS image from the ASI all-sky camera to astrometrically calibrate the inner portion of the image,
    perform photometry of the sources detected in that portion of the image, measure a photometric zeropoint using
    bright calibration stars, and produce a calibrated sky background image in mag/arcsec^2.

    fitsfile : str or `~pathlib.PosixPath`
        FITS file to process.
    """
    if isinstance(fitsfile, str):
        fitsfile = Path(fitsfile)
    im = CCDData.read(fitsfile, unit=u.adu)

    bkg = make_background(im)

    bkg_image = CCDData(bkg.background, unit=u.adu)
    bkg_image.write(fitsfile.with_suffix(".bkg.fits"), overwrite=True)

    diff = CCDData(im.data - bkg_image.data, unit=u.adu)
    diff_fp = fitsfile.with_suffix(".subt.fits")
    diff.write(diff_fp, overwrite=True)

    solved_fp = solve_field(diff_fp)

    solved = CCDData.read(solved_fp)

    segm = make_segmentation_image(solved)

    s_cat, p_cat = make_catalog(solved, segm, solved.wcs)
    # catalog.write(fitsfile.with_suffix(".cat.fits"), overwrite=True)

    filt = im.header['FILTER']
    filt_col = f'{filt}_mag'

    phot_off = p_cat[filt_col] - s_cat['obs_mag']
    cut = p_cat[filt_col] < 4.0
    zp = phot_off[cut].mean()

    pix_scales = wcs.utils.proj_plane_pixel_scales(solved.wcs)
    pix_area = pix_scales[0] * pix_scales[1] * 3600.**2

    sky_mag = CCDData(zp + (-2.5 * np.log10(bkg_image.data/pix_area)), unit=u.mag / u.arcsec**2)

    sky_mag.write(fitsfile.with_suffix(".sky.fits"), overwrite=True)

    return s_cat, p_cat


def process_stellacam_image(fitsfile, year, write=False, zp=0., return_products=False):
    """
    Process a FITS image from the stellacam all-sky camera to extract the background, perform photometry
    of the sources  detected in the image, and, if provided a wcs, measure a photometric zeropoint using
    bright calibration stars to produce a calibrated sky background image in mag/arcsec^2.

    fitsfile : str or `~pathlib.PosixPath`
        FITS file to process.

    year : int
        Required to set which WCS and field mask to use.

    write : bool
        If True, write intermediate FITS outputs. Matched catalog is always written.

    zp : float
        Photometric zeropoint. Default to 0 for raw instrumental magnitudes.
    """
    if isinstance(fitsfile, str):
        fitsfile = Path(fitsfile)

    with fits.open(fitsfile) as hdul:
        im = hdul[0].data
        hdr = hdul[0].header

    tobs = get_ut(hdr, year=year)

    if tobs is None:
        print(f"No valid UT in {fitsfile}...")
        return None

    # we only do the full photometric analysis when the camera is in the dark sky steady state
    # configuration of 256 frames integration with a gain of 106.
    if hdr['FRAME'] != '256 Frames' or hdr['GAIN'] != 106:
        print(f"Not processing {fitsfile} with frame={hdr['FRAME']} and gain={hdr['GAIN']}...")
        return None
    else:
        print(f"Processing {fitsfile}...")

    mask = load_mask(year=year)
    wcs = load_wcs(year=year)
    tobs.format = 'iso'

    skycat = update_altaz(load_skycam_catalog(), time=tobs)

    bkg = make_background(im, boxsize=(5, 5), filter_size=(3, 3), inmask=mask)

    bkg_image = CCDData(bkg.background * mask, unit=u.adu)
    if write:
        bkg_image.write(fitsfile.with_suffix(".bkg.fits"), overwrite=True)

    diff = CCDData(im.data * mask - bkg_image.data, unit=u.adu)
    if write:
        diff_fp = fitsfile.with_suffix(".subt.fits")
        diff.write(diff_fp, overwrite=True)

    segm = make_segmentation_image(diff.data)

    catalog = make_catalog(im, segm, background=bkg)
    matched = match_stars(skycat, catalog, wcs, max_sep=2.5*u.deg)
    matched['UT'] = tobs.value
    try:
        matched.write(fitsfile.with_suffix(".cat.csv"), overwrite=True)
    except Exception as e:
        print(f"Oops! Can't write CSV output for {fitsfile}: {e}")
        pass

    if wcs is not None:
        pix_scales = astropy.wcs.utils.proj_plane_pixel_scales(wcs)
        pix_area = pix_scales[0] * pix_scales[1] * 3600.**2
    else:
        # stellacam pixel scale is about 0.27 deg/pixel
        pix_area = (0.27 * 3600)**2

    sky_mag = CCDData(zp + (-2.5 * np.log10(bkg_image.data/pix_area)), unit=u.mag / u.arcsec**2)

    if write:
        sky_mag.write(fitsfile.with_suffix(".sky.fits"), overwrite=True)

    if return_products:
        return bkg_image, diff, segm, sky_mag, matched
    else:
        return None


def process_stellacam_dir():
    """
    Work through and process directory of stellacam allsky camera images. This function is meant to be the core of
    a command-line script.
    """
    parser = argparse.ArgumentParser(description="Script for doing photometric processing of a directory of stellacam images.")

    parser.add_argument(
        'rootdir',
        metavar="<directory of allsky images>",
        help="Directory containing stellecam allsky camera images. Directory name expected to be of format YYYYMMDD."
    )

    parser.add_argument(
        '--writefits',
        help="Write intermediate images out as FITS files.",
        action="store_true"
    )

    parser.add_argument(
        '--zeropoint',
        help="Photometric zeropoint to apply to source magnitudes. (default: 0.0)",
        type=float,
        default=0.0
    )

    parser.add_argument(
        '--nproc',
        help="Number of processors to utilize. (default: 6)",
        type=int,
        default=6
    )

    parser.add_argument(
        "-z",
        help="Process .fits.gz files",
        action="store_true"
    )

    parser.add_argument(
        "-s", "--strip",
        help="Process directory to create strip image and plot",
        action="store_true"
    )

    args = parser.parse_args()

    rootdir = Path(args.rootdir)
    year = int(rootdir.name[0:4])

    if args.strip:
        out_fits = f"strip_{rootdir}.fits"
        out_pdf = f"strip_{rootdir}.pdf"
        stellacam_strip_image(rootdir, writefile=True, outfile=out_fits, compressed=args.z, year=year)
        plot_strip_image(out_fits, savefile=out_pdf, masked=False, cmap='viridis', contrast=0.2, stretch=SqrtStretch())
    else:
        if args.z:
            files = sorted(list(rootdir.glob("*.fits.gz")))
        else:
            files = sorted(list(rootdir.glob("*.fits")))

        process = partial(process_stellacam_image, year=year, write=args.writefits, zp=args.zeropoint)
        with multiprocessing.Pool(processes=args.nproc) as pool:
            pool.map(process, files)

        print("Now go through the output CSVs, group the data by star, and output the photometry for each star...")
        frames = []
        for csv in rootdir.glob("*.csv"):
            frames.append(pd.read_csv(csv))

        if len(frames) > 0:
            df = pd.concat(frames)
            g = df.groupby('Star Name')
            for k in g.groups.keys():
                g.get_group(k).to_csv(rootdir / f"star_{k.replace(' ', '_').lower()}.csv")
        else:
            print(f"No photometry extracted for {rootdir.name}...")


if __name__ == "__main__":
    process_stellacam_dir()
