# Licensed under a 3-clause BSD style license - see LICENSE.rst


import subprocess
from pathlib import Path

import numpy as np

from astropy.io import fits


def solve_field(fitsfile, sigma=3.0. x_size=1800, y_size=1800):
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
