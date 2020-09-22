# Licensed under a 3-clause BSD style license - see LICENSE.rst

import os
import subprocess
import pkg_resources
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import astropy.units as u
from astropy.nddata import CCDData, Cutout2D
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
from astropy.wcs.wcs import WCSHDO_SIP
from .fit_wcs import wcs_zea

MMT_LOCATION = EarthLocation.from_geodetic("-110:53:04.4", "31:41:19.6", 2600 * u.m)


def load_wcs(year=2020):
    """
    Load appropriate WCS for the given year
    """
    if year in [2017, 2018, 2019, 2020]:
        wcs_file = "wcs_2019.fits"
    elif year in [2011, 2012]:
        wcs_file = "wcs_2011.fits"
    elif year in [2015, 2016]:
        wcs_file = "wcs_2016.fits"
    else:
        print(f"WCS not yet implemented for {year}.")
        return None

    wcs_path = pkg_resources.resource_filename(__name__, os.path.join("data", wcs_file))

    wcs = WCS(wcs_path)

    return wcs


def initial_wcs_fit(catalog, x_key='xcentroid', y_key='ycentroid', alt_key='Alt', az_key='Az', crpix1=324, crpix2=235, cdelt=0.3):
    """
    Use wcs_zea() routine borrowed from LSST to make an initial linear fit to an all-sky camera image.
    The input catalog needs to have both X/Y image positions and object Alt/Az.
    """
    fun = wcs_zea(
        catalog[x_key],
        catalog[y_key],
        catalog[alt_key],
        catalog[az_key],
        crpix1=crpix1,
        crpix2=crpix2,
        a_order=2,
        b_order=2
    )
    init = np.array([crpix1, crpix2, 1, 1, cdelt, 0.003, 0.003, cdelt])
    x0 = init
    fit_result = minimize(fun, x0)
    wcs_initial = fun.return_wcs(fit_result.x)
    return wcs_initial


def wcs_sip_fit(cat_df, init_wcs, sip_degree=3, x_key='xcentroid', y_key='ycentroid', alt_key='Alt', az_key='Az'):
    """
    Take an initial WCS and refine it with SIP distortion terms by fitting to a larger catalog in a pandas
    datafrom, cat_df. The input dataframe needs to have both X/Y image positions and object Alt/Az.
    """
    wcs_refined = fit_wcs_from_points(
        (cat_df[x_key].array, cat_df[y_key].array),
        SkyCoord(cat_df[az_key]*u.deg, cat_df[alt_key]*u.deg),
        projection=init_wcs,
        proj_point=SkyCoord(0*u.deg, 90*u.deg),
        sip_degree=sip_degree
    )
    return wcs_refined


def write_sip(in_wcs, outfile, overwrite=False):
    """
    Write input WCS to a FITS file and include the SIP information in the header. This is not done by default
    and you need to set an obscure flag to make it happen.
    """
    in_wcs.to_fits(relax=WCSHDO_SIP).writeto(outfile, overwrite=overwrite)


def update_altaz(cat, time=Time.now(), ra='RA', dec='Dec', ra_unit=u.deg, dec_unit=u.deg, location=MMT_LOCATION):
    """
    Update Alt and Az columns for input catalog for given time and location
    """
    coords = SkyCoord(ra=cat[ra]*ra_unit, dec=cat[dec]*dec_unit)
    ack = coords.transform_to(AltAz(obstime=time, location=location))
    cat['Alt'], cat['Az'] = ack.alt, ack.az
    return cat


def solve_field(fitsfile, sigma=3.0, x_size=1800, y_size=1800):
    """
    Run astronomy.net's solver to find the WCS for at least part of an all-sky image. The solver doesn't work
    well with a full image so by default we trim it down to the central part of the image. This is sufficient for
    photometric calibration purposes.

    fitsfile : string or `~pathlib.PosixPath`
        Original FITS image.

    sigma: float (default: 3.0)
        Threshold in number of sigma to be considered a source.

    x_size: int (default: 1800)
        X size of the trimmed image that is solved.

    y_size: int (default: 1800)
        Y size of the trimmged image.
    """

    if isinstance(fitsfile, str):
        fitsfile = Path(fitsfile)
    im = CCDData.read(fitsfile)

    xmid = int(im.shape[1]/2)
    ymid = int(im.shape[0]/2)

    trimmed = CCDData(Cutout2D(im, (xmid, ymid), (y_size, x_size), copy=True).data, unit=u.adu)

    trimmed_path = fitsfile.with_suffix(".trimmed.fits")
    trimmed.write(trimmed_path, overwrite=True)

    subprocess.run(
        [
            "solve-field",
            str(trimmed_path),
            "--continue",
            "--no-background-subtraction",
            "--sigma",
            "3.0",
            "--keep-xylist",
            "%s.xy",
            "-L",
            "100",
            "-H",
            "150",
            "-u",
            "app"
        ]
    )

    solved_path = fitsfile.with_suffix(".trimmed.new")
    return solved_path
