"""Reading Alcor RGB FITS frames, the corner bias, and the processed cube."""

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

from .config import alcor_calibration
from .timeutils import (
    _alcor_frame_time, _filename_ut_datetime, _read_frame_date
)
from .wcs import build_alcor_wcs
from .masks import _apply_badpix_repair, load_alcor_badpix_mask


def _corner_bias(cube, size=10):
    """
    Per-channel median bias from square corner regions.
    """
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3 or cube.shape[0] != 3:
        raise ValueError(f"expected a (3, ny, nx) cube, got {cube.shape}")
    _, ny, nx = cube.shape
    if ny < size or nx < size:
        raise ValueError(f"image is smaller than the {size}x{size} bias regions")
    corners = [
        cube[:, :size, :size],
        cube[:, :size, -size:],
        cube[:, -size:, :size],
        cube[:, -size:, -size:],
    ]
    pixels = np.concatenate([corner.reshape(3, -1) for corner in corners], axis=1)
    return np.median(pixels, axis=1)



def load_alcor_fits(filename, wcs=None, badpix="repair", masks_dir=None):
    """
    Load an alcor OMEA 8C FITS file and return ``(cube, wcs, mask)``.

    The raw ``(3, ny, nx)`` RGB cube is returned in native FITS orientation,
    unmodified except for optional bad-pixel repair: no transpose, trim, rotate,
    shift, or flipud, and no bias subtraction. Geometry lives entirely in the
    returned WCS.

    Parameters
    ----------
    filename : str or Path
        FITS file. Compressed (.gz, .bz2) inputs are supported.
    wcs : `astropy.wcs.WCS` or None (default=None)
        Geometry WCS. When None, the calibration epoch nearest the frame's time
        is resolved and ``build_alcor_wcs`` constructs the raw-frame ARC WCS.
    badpix : str or None or path or ndarray (default="repair")
        "repair" repairs flagged pixels per channel with their local 5x5 median;
        None leaves the cube untouched (the mask is still resolved and returned);
        a path or (3, ny, nx) bool array uses that mask explicitly (and repairs).
    masks_dir : str or None (default=None)
        Override the bad-pixel masks directory (else $ALCOR_BADPIX_DIR, else the
        packaged data/badpix/).

    Returns
    -------
    cube : ndarray
        Raw ``(3, ny, nx)`` float32 cube (channels 0,1,2 = R,G,B).
    wcs : `astropy.wcs.WCS`
        Raw-frame ARC WCS mapping pixel (x, y) <-> (azimuth, altitude).
    mask : ndarray or None
        ``(3, ny, nx)`` bool bad-pixel mask in native orientation, or None when
        no mask is available.
    """
    with fits.open(filename) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float32)   # (3, ny, nx)

    # --- resolve the bad-pixel mask (explicit, or nearest-date) ---
    mask = None
    if isinstance(badpix, np.ndarray):
        cand = badpix.astype(bool)
    elif isinstance(badpix, Path) or (isinstance(badpix, str) and badpix != "repair"):
        cand = np.asarray(fits.getdata(badpix)).astype(bool)
    else:                                                   # "repair" or None
        cand = None
        try:
            dt = _filename_ut_datetime(filename)
            t = (Time(dt) if dt is not None
                 else Time(_read_frame_date(filename), format="isot", scale="utc"))
            cand, _ = load_alcor_badpix_mask(t, masks_dir=masks_dir)
        except (KeyError, OSError, ValueError):
            cand = None
    if cand is not None and cand.shape == cube.shape:
        mask = cand

    if badpix is not None and mask is not None:
        cube = _apply_badpix_repair(cube, mask)

    if wcs is None:
        cal = alcor_calibration(_alcor_frame_time(filename))
        wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                              rotation=cal["rotation"],
                              radial_coeffs=cal["radial_coeffs"],
                              horizon_radius=cal["horizon_radius"],
                              tangential_coeffs=cal["tangential_coeffs"],
                              axis_tilt=cal["axis_tilt"])

    return cube, wcs, mask



def alcor_proc_fits(filename, output_file=None, overwrite=False, **kwargs):
    """
    Process an alcor OMEA 8C FITS file via `load_alcor_fits` and write a new
    FITS file containing the raw ``(3, ny, nx)`` RGB cube (native orientation)
    with the raw-frame alt/az WCS encoded in the header.

    Parameters
    ----------
    filename : str or `~pathlib.Path`
        Input FITS file.
    output_file : str or `~pathlib.Path` or None (default=None)
        Output path. If None, derived from `filename` by replacing the
        first `.fits` substring with `_proc.fits`.
    overwrite : bool (default=False)
        Passed through to `fits.PrimaryHDU.writeto`.
    **kwargs
        Forwarded to `load_alcor_fits` (``wcs``, ``badpix``, ``masks_dir``).

    Returns
    -------
    output_file : `~pathlib.Path`
        Path to the written FITS file.
    """
    cube, wcs, _ = load_alcor_fits(filename, **kwargs)
    if output_file is None:
        stem = str(filename)
        for ext in (".fits.bz2", ".fits.gz", ".fits"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        output_file = stem + "_proc.fits"
    output_file = Path(output_file)

    hdu = fits.PrimaryHDU(data=cube.astype(np.float32),
                          header=wcs.to_header(relax=True))
    hdu.writeto(output_file, overwrite=overwrite)
    return output_file
