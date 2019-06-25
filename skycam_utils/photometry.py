# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import pkg_resources
from pathlib import Path

import numpy as np

import astropy.units as u
from astropy import stats
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData

import ccdproc
import photutils


def load_bright_star_catalog():
    """
    Load the catalog containing Sloan photometry for the brightest stars
    """
    catpath = pkg_resources.resource_filename(__name__, os.path.join("data", "bright_star_sloan.fits"))
    phot_cat = Table.read(catpath)
    return phot_cat


def make_background(data, sigma=2., snr=2., npixels=7, boxsize=(7, 7), filter_size=(3, 3), mask_sources=True):
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
    sigma = fwhm * stats.gaussian_fwhm_sigma
    kernel = Gaussian2DKernel(sigma, x_size=x_size, y_size=y_size)
    kernel.normalize()
    threshold = photutils.detect_threshold(data, snr=snr)
    segm = photutils.detect_sources(data, threshold, npixels=npixels, filter_kernel=kernel)
    if deblend:
        segm = photutils.deblend_sources(data, segm, npixels=npixels, filter_kernel=kernel, nlevels=nlevels, contrast=contrast)
    return segm
