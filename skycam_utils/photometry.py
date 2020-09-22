# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import pkg_resources
import warnings

import numpy as np

import astropy.units as u
from astropy import stats
from astropy.convolution import Gaussian2DKernel
from astropy.table import Table, hstack, unique
from astropy.coordinates import SkyCoord
from astropy.io import fits

import photutils


warnings.filterwarnings('ignore')


def load_mask(year=2020):
    """
    Load appropriate mask image for the given year
    """
    if year in [2017, 2018, 2019, 2020]:
        mask_file = "mask_2017_2020.fits"
    elif year in [2011, 2012]:
        mask_file = "mask_2011.fits"
    elif year in [2015, 2016]:
        mask_file = "mask_2016.fits"
    else:
        print(f"Mask not yet implemented for {year}.")
        return None

    mask_path = pkg_resources.resource_filename(__name__, os.path.join("data", mask_file))

    with fits.open(mask_path) as hdul:
        im = hdul[0].data

    return im


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


def make_background(data, sigma=3., snr=3., npixels=4, boxsize=(10, 10),
                    filter_size=(5, 5), mask_sources=True, inmask=None):
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

    mask_sources : bool (default: True)
        If true, then use `~photutils.make_source_mask` to mask sources before creating background
    """
    sigma_clip = stats.SigmaClip(sigma=sigma)
    bkg_estimator = photutils.SExtractorBackground()
    if inmask is not None:
        cov_mask = np.zeros_like(inmask, dtype=bool)
        cov_mask[inmask != np.nan] = False
        cov_mask[inmask == np.nan] = True
    else:
        cov_mask = np.zeros_like(data, dtype=bool)

    if mask_sources:
        src_mask = photutils.make_source_mask(data, nsigma=snr, npixels=npixels, mask=cov_mask)
        mask = (cov_mask | src_mask)
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
            bkg_estimator=bkg_estimator,
            mask=cov_mask
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
    threshold = photutils.detect_threshold(data, nsigma=snr)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        segm = photutils.detect_sources(data, threshold, npixels=npixels, filter_kernel=kernel)
        if deblend:
            segm = photutils.deblend_sources(data, segm, npixels=npixels, filter_kernel=kernel, nlevels=nlevels, contrast=contrast)

    return segm


def make_catalog(data, segm, border_width=10, background=None):
    """
    Measure source properties from data, segmentation image, and wcs and match against photometric catalog. Returned
    trimmed and matched combination catalog.

    data : 2D `~numpy.ndarray`
        Image containing sources to extract.

    segm : `~photutils.segmentation.SegmentationImage`
        Segmentation image created from data.

    border_width : int (default: 10)
        Remove source labels within border_wiidth from the edges of the data.

    background : None or `~photutils.Background2D`
        Background to pass into photutils.source_properties()
    """
    segm.remove_border_labels(border_width=border_width)
    if background is not None:
        prop_cat = photutils.source_properties(data, segm, background=background.background)
    else:
        prop_cat = photutils.source_properties(data, segm)
    cat = prop_cat.to_table()
    cat['obs_mag'] = -2.5 * np.log10(cat['source_sum'])
    cat.keep_columns(['id', 'xcentroid', 'ycentroid', 'source_sum', 'background_mean', 'obs_mag'])
    return cat


def match_stars(skycat, srccat, in_wcs, max_sep=1.5*u.deg):
    """
    skycat : `~astropy.table.Table`
        Skycam catalog as produced by load_skycam_catalog() with Alt and Az columns
        added for the appropriate time.

    srccat : `~astropy.table.Table`
        Source catalog as produced by make_catalog().

    in_wcs : `~astropy.wcs.WCS`
        WCS used to create Alt/Az for skycat.

    max_sep : `~astropy.units.Quantity` (default: 1 degree)
        Separation criterium for valid matching.
    """
    pred_az, pred_alt = in_wcs.all_pix2world(srccat['xcentroid'], srccat['ycentroid'], 0)
    pred_coord = SkyCoord(ra=pred_az*u.deg, dec=pred_alt*u.deg)
    act_coord = SkyCoord(ra=skycat['Az'], dec=skycat['Alt'])
    idx, d2d, d3d = pred_coord.match_to_catalog_sky(act_coord, nthneighbor=1)
    sep_constraint = d2d < max_sep
    matches = srccat[sep_constraint]
    cat_matches = skycat[idx[sep_constraint]]
    matched_cat = hstack([cat_matches, matches])
    matched_cat.sort('obs_mag')

    # If we get multiple matches, keep the brightest one. May be wrong, but at least consistent
    matched_cat = unique(matched_cat, keys='Star Name', keep='first')
    return matched_cat
