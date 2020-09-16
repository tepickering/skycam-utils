# Licensed under a 3-clause BSD style license - see LICENSE.rst

from pathlib import Path

import numpy as np

import astropy
import astropy.units as u
import astropy.wcs as wcs
from astropy.io import fits
from astropy.nddata import CCDData

from .photometry import make_background, make_segmentation_image, make_catalog
from .astrometry import solve_field


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


def process_stellacam_image(fitsfile, wcs=None, mask=None, write=False, zp=0.):
    """
    Process a FITS image from the stellacam all-sky camera to extract the background, perform photometry
    of the sources  detected in the image, and, if provided a wcs, measure a photometric zeropoint using
    bright calibration stars to produce a calibrated sky background image in mag/arcsec^2.

    fitsfile : str or `~pathlib.PosixPath`
        FITS file to process.

    wcs : `~astropy.wcs.WCS` object
        WCS to map (x,y) to (az, alt)

    mask : ndarray-like
        Mask image to define regions outside of the FOV or obscured by the MMT or flagpole.

    write : bool
        If True, write outputs to FITS files.

    zp : float
        Photometric zeropoint. Default to 0 for raw instrumental magnitudes.
    """
    if isinstance(fitsfile, str):
        fitsfile = Path(fitsfile)

    with fits.open(fitsfile) as hdul:
        im = hdul[0].data

    if mask is None:
        mask = np.ones_like(im)

    bkg = make_background(im, boxsize=(5, 5), filter_size=(3, 3), inmask=mask)

    bkg_image = CCDData(bkg.background * mask, unit=u.adu)
    if write:
        bkg_image.write(fitsfile.with_suffix(".bkg.fits"), overwrite=True)

    diff = CCDData(im.data * mask - bkg_image.data, unit=u.adu)
    if write:
        diff_fp = fitsfile.with_suffix(".subt.fits")
        diff.write(diff_fp, overwrite=True)

    segm = make_segmentation_image(diff.data)

    catalog = make_catalog(im, segm)
    if write:
        catalog.write(fitsfile.with_suffix(".cat.csv"), overwrite=True)

    if wcs is not None:
        pix_scales = astropy.wcs.utils.proj_plane_pixel_scales(wcs)
        pix_area = pix_scales[0] * pix_scales[1] * 3600.**2
    else:
        # stellacam pixel scale is about 0.27 deg/pixel
        pix_area = (0.27 * 3600)**2

    sky_mag = CCDData(zp + (-2.5 * np.log10(bkg_image.data/pix_area)), unit=u.mag / u.arcsec**2)

    if write:
        sky_mag.write(fitsfile.with_suffix(".sky.fits"), overwrite=True)

    return bkg_image, diff, segm, sky_mag, catalog
