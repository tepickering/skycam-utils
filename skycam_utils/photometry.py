# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import pkg_resources

import numpy as np

import astropy.units as u
from astropy import stats
from astropy.convolution import Gaussian2DKernel
from astropy.table import Table
from astropy.coordinates import SkyCoord

import photutils


def load_bright_star_catalog():
    """
    Load the catalog containing Sloan photometry for the brightest stars
    """
    catpath = pkg_resources.resource_filename(__name__, os.path.join("data", "bright_star_sloan.fits"))
    phot_cat = Table.read(catpath, memmap=True)
    phot_cat['coords'] = SkyCoord(phot_cat['_RAJ2000'], phot_cat['_DEJ2000'], frame='icrs', unit='deg')
    cut = phot_cat['g_mag'] < 5.0
    return phot_cat[cut]


def load_skycam_catalog():
    """
    Load the curated skycam catalog that combines the bright star catalog with Sloan photometry
    with a larger catalog that includes the star names as well.
    """
    catpath = pkg_resources.resource_filename(__name__, os.path.join("data", "skycam_stars.csv"))
    skycam_cat = Table.read(catpath)
    return skycam_cat


def make_background(data, sigma=3., snr=3., npixels=4, boxsize=(10, 10), filter_size=(5, 5), mask_sources=True):
    """
    Use photutils to create a background model from the input data.

    data : 2D `~numpy.ndarray`
        Data from which to extract background model

    sigma : float (default: 2.0)
        Number of sigma to use for sigma clipping

    snr : float (default: 2.0)
        SNR to use when masking sources

    npixels : int (default: 7)
        Number of connected pixels to use when masking sources

    boxsize : tuple or int (default: (7, 7))
        Size of box used to create the gridded background map

    filter_size : tuple or int (default: (3, 3))
        Window size of the 2D median filter to apply to the low-res background map

    mask_source : bool (default: True)
        If true, then use `~photutils.make_source_mask` to mask sources before creating background
    """
    sigma_clip = stats.SigmaClip(sigma=sigma)
    bkg_estimator = photutils.SExtractorBackground()
    if mask_sources:
        mask = photutils.make_source_mask(data, snr=snr, npixels=npixels)
        bkg = photutils.Background2D(
            data,
            boxsize,
            filter_size=filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            mask=mask
        )
    else:
        bkg = photutils.Background2D(
            data,
            boxsize,
            filter_size=filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator
        )
    return bkg


def make_segmentation_image(data, fwhm=2.0, snr=5.0, x_size=5, y_size=5, npixels=7, nlevels=32, contrast=0.001, deblend=True):
    """
    Use photutils to create a segmentation image containing detected sources.

    data : 2D `~numpy.ndarray`
        Image to segment into sources.

    fwhm : float (default: 2.0)
        FWHM of the kernel used to filter the image.

    snr : float (default: 5.0)
        Source S/N used to set detection threshold.

    x_size : int (default: 5)
        X size of the 2D `~astropy.convolution.Gaussian2DKernel` filter.

    y_size : int (default: 5)
        Y size of the 2D `~astropy.convolution.Gaussian2DKernel` filter.

    npixels : int (default: 7)
        Number of connected pixels required to be considered a source.

    nlevels : int (default: 32)
        Number of multi-thresholding levels to use when deblending sources.

    contrast : float (default: 0.001)
        Fraction of the total blended flux that a local peak must have to be considered a separate object.

    deblend : bool (default: True)
        If true, deblend sources after creating segmentation image.
    """
    sigma = fwhm * stats.gaussian_fwhm_to_sigma
    kernel = Gaussian2DKernel(sigma, x_size=x_size, y_size=y_size)
    kernel.normalize()
    threshold = photutils.detect_threshold(data, snr=snr)
    segm = photutils.detect_sources(data, threshold, npixels=npixels, filter_kernel=kernel)
    if deblend:
        segm = photutils.deblend_sources(data, segm, npixels=npixels, filter_kernel=kernel, nlevels=nlevels, contrast=contrast)
    return segm


def make_catalog(data, segm, border_width=10):
    """
    Measure source properties from data, segmentation image, and wcs and match against photometric catalog. Returned
    trimmed and matched combination catalog.

    data : 2D `~numpy.ndarray`
        Image containing sources to extract.

    segm : `~photutils.segmentation.SegmentationImage`
        Segmentation image created from data.

    border_width : int (default: 10)
        Remove source labels within border_wiidth from the edges of the data.
    """
    segm.remove_border_labels(border_width=border_width)
    prop_cat = photutils.source_properties(data, segm)
    cat = prop_cat.to_table()
    cat['obs_mag'] = -2.5 * np.log10(cat['source_sum'])
    cat.keep_columns(['id', 'xcentroid', 'ycentroid', 'source_sum', 'background_mean', 'obs_mag'])
    return cat
