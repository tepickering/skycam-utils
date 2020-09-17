# Licensed under a 3-clause BSD style license - see LICENSE.rst

import subprocess
from pathlib import Path

import astropy.units as u
from astropy.nddata import CCDData, Cutout2D
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astropy.wcs import WCS

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
