# Licensed under a 3-clause BSD style license - see LICENSE.rst


import subprocess
from pathlib import Path

import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.nddata import CCDData, Cutout2D


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
            "120",
            "-H",
            "150",
            "-u",
            "app"
        ]
    )

    solved_path = fitsfile.with_suffix(".trimmed.new")
    return solved_path
