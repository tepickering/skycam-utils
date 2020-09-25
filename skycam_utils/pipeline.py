# Licensed under a 3-clause BSD style license - see LICENSE.rst

import datetime
import argparse
import multiprocessing
import warnings

from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd

import astropy
import astropy.units as u
import astropy.wcs as wcs
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.time import Time
from astropy.utils import iers

from .photometry import make_background, make_segmentation_image, make_catalog, load_mask, match_stars, load_skycam_catalog
from .astrometry import solve_field, load_wcs, update_altaz


warnings.filterwarnings('ignore')
iers.conf.auto_download = False
iers.conf.auto_max_age = None


def get_ut(hdr, year=2020):
    """
    When the UT is actually valid in the stellacam image headers, it's one format for
    2011-2012 (and maybe earlier) and another for 2015 onwards.
    """
    if year < 2013:
        tobs = Time(f"{hdr['DATE']}T{hdr['UT']}", scale='utc')
    else:
        dt = datetime.datetime.strptime(hdr['UT'], "%a %b %d %H:%M:%S %Y")
        tobs = Time(dt, scale='utc')
    return tobs


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

    # we only do the full photometric analysis when the camera is in the dark sky steady state
    # configuration of 256 frames integration with a gain of 106.
    if hdr['FRAME'] != '256 Frames' or hdr['GAIN'] != 106:
        print(f"Not processing {fitsfile} with frame={hdr['FRAME']} and gain={hdr['GAIN']}...")
        return None
    else:
        print(f"Processing {fitsfile}...")

    mask = load_mask(year=year)
    wcs = load_wcs(year=year)
    tobs = get_ut(hdr, year=year)
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
    except:
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

    args = parser.parse_args()

    rootdir = Path(args.rootdir)
    year = int(rootdir.name[0:4])
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
