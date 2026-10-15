"""
Aperture blending: the flux that neighbouring stars add to a measurement.

The photometry aperture is 4 px and the plate scale is only 6.6-8.3 px/deg, so
the aperture is ~0.5 deg across and any other star inside it adds its light.
That is not a small correction: on a clear night ~80% of the stars in
``ALCOR_EXT_MAG_WINDOW`` carry more than 1% neighbour flux, and left alone it
reads as a NEGATIVE extinction (the star looks brighter than its catalog
magnitude) -- -0.11 mag in the median on 2026-10-14, and -0.8 for real pairs
half an aperture apart (HD 166229 + HIP 89008, 24').

Which neighbours fall inside depends on where the star is, not just on the
catalog: the ARC projection is equidistant radially but stretched tangentially
(z / sin z), and the fitted distortion shrinks the radial scale further toward
the horizon, so the 4 px aperture is a ~0.61 x 0.39 deg ellipse at alt 20 and a
0.48 deg circle at the zenith. So the neighbour list is stored as sky offsets
(``bright_star_neighbors.fits``, east/north of each catalog star) and projected
per measurement: the offsets are rotated into the local (radial, tangential)
frame by the parallactic angle and scaled by the WCS plate scales at the star's
altitude. Each neighbour then contributes the fraction of its Gaussian PSF that
lands in the circular aperture (a Rice CDF), relative to the target's own.
"""

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.stats import rice

from .config import (
    ALCOR_BLEND_APERTURE_RADIUS, ALCOR_BLEND_EFFICIENCY, ALCOR_BLEND_PSF_SIGMA,
    ALCOR_CALIBRATIONS,
    _DATA_ROOT, alcor_calibration
)
from ..astrometry import MMT_LOCATION
from .wcs import build_alcor_wcs


@lru_cache(maxsize=1)
def _neighbor_table():
    """
    The packaged neighbour list as a DataFrame with per-band neighbour mags.

    Hipparcos carries V and B-V but no R, so the neighbour's R is ``V - (V-R)``
    with ``V-R`` from a quadratic in ``B-V`` fit to the calibration catalog
    itself (which has both colours).
    """
    nb = Table.read(str(_DATA_ROOT / "bright_star_neighbors.fits")).to_pandas()
    nb["LABEL"] = [x.decode() if isinstance(x, bytes) else str(x)
                   for x in nb["LABEL"]]
    nb["LABEL"] = nb["LABEL"].str.strip()
    cat = Table.read(str(_DATA_ROOT / "bright_star_sloan_named.fits"))
    bv = np.asarray(cat["B-V"], dtype=float)
    vr = np.asarray(cat["V-R"], dtype=float)
    ok = np.isfinite(bv) & np.isfinite(vr)
    coeff = np.polyfit(bv[ok], vr[ok], 2)
    v = nb["V"].to_numpy(dtype=float)
    nbv = nb["BV"].to_numpy(dtype=float)
    nbv = np.where(np.isfinite(nbv), nbv, np.nanmedian(nbv))
    nb["mag_g"] = v
    nb["mag_b"] = v + nbv
    nb["mag_r"] = v - np.polyval(coeff, nbv)
    return nb[["LABEL", "DE", "DN", "mag_r", "mag_g", "mag_b"]]


@lru_cache(maxsize=len(ALCOR_CALIBRATIONS) + 1)
def _plate_scale_table(epoch):
    """
    ``(alt_grid, radial, tangential)`` plate scales in px/deg for a calibration
    epoch, averaged over azimuth (the variation with azimuth is under 2%).
    """
    cal = next((c for c in ALCOR_CALIBRATIONS if c["epoch"] == epoch), None)
    if cal is None:
        cal = alcor_calibration(None)
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])
    alt = np.arange(0.0, 89.5, 0.5)
    az = np.arange(0.0, 360.0, 30.0)
    aa, hh = np.meshgrid(az, alt)
    step = 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x0, y0 = wcs.world_to_pixel_values(aa, hh)
        xr, yr = wcs.world_to_pixel_values(aa, hh + step)
        xt, yt = wcs.world_to_pixel_values(
            aa + step / np.cos(np.radians(hh)), hh)
    radial = np.hypot(xr - x0, yr - y0).mean(axis=1) / step
    tangential = np.hypot(xt - x0, yt - y0).mean(axis=1) / step
    return alt, radial, tangential


def _parallactic_angle(az_deg, alt_deg, lat_deg):
    """
    Parallactic angle (rad) of a horizontal position: the position angle, east
    of north, of the direction toward the zenith.
    """
    az, alt, lat = np.radians(az_deg), np.radians(alt_deg), np.radians(lat_deg)
    sin_dec = np.sin(lat) * np.sin(alt) + np.cos(lat) * np.cos(alt) * np.cos(az)
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))
    sin_h = -np.sin(az) * np.cos(alt) / np.cos(dec)
    cos_h = (np.sin(alt) - np.sin(lat) * sin_dec) / (np.cos(lat) * np.cos(dec))
    return np.arctan2(sin_h, np.tan(lat) * np.cos(dec) - sin_dec * cos_h)


def _encircled(d_px, radius, sigma):
    """Fraction of a circular Gaussian offset ``d_px`` inside ``radius``."""
    return rice.cdf(radius / sigma, np.asarray(d_px, dtype=float) / sigma)


def alcor_blend_terms(df, target_mags, time=None,
                      aperture_radius=ALCOR_BLEND_APERTURE_RADIUS,
                      psf_sigma=ALCOR_BLEND_PSF_SIGMA,
                      efficiency=ALCOR_BLEND_EFFICIENCY, location=MMT_LOCATION):
    """
    Per-row blend term in magnitudes for each band: ``{band: ndarray}``.

    ``df`` needs ``azimuth``, ``altitude`` and a star label (``name`` column,
    else the index); ``target_mags`` maps each band (``r``/``g``/``b``) to the
    rows' catalog magnitudes in that band. The term is
    ``-2.5 log10(1 + e * sum_n 10**(-0.4 (m_n - m_target)) * f(d_n) / f(0))``
    with ``f`` the encircled fraction of a Gaussian PSF in the aperture, ``d_n``
    the neighbour's pixel distance and ``e`` the measured ``efficiency`` -- so it is <= 0, exactly 0 for an isolated star,
    and NaN where the target magnitude is unknown (a variable). Add it to the
    catalog magnitude to get what the aperture should actually see.
    """
    n = len(df)
    names = (df["name"] if "name" in df.columns
             else df.index.to_series()).astype(str).str.strip().to_numpy()
    out = {band: np.zeros(n) for band in target_mags}
    for band, mags in target_mags.items():
        out[band] = np.where(np.isfinite(mags), 0.0, np.nan)
    if not n:
        return out

    nb = _neighbor_table()
    rows = pd.DataFrame({"row": np.arange(n), "LABEL": names})
    pairs = rows.merge(nb, on="LABEL", how="inner")
    if not len(pairs):
        return out

    epoch = alcor_calibration(time)["epoch"]
    alt_grid, radial, tangential = _plate_scale_table(epoch)

    i = pairs["row"].to_numpy()
    az = df["azimuth"].to_numpy(dtype=float)[i]
    alt = df["altitude"].to_numpy(dtype=float)[i]
    q = _parallactic_angle(az, alt, location.lat.deg)
    de = pairs["DE"].to_numpy(dtype=float)
    dn = pairs["DN"].to_numpy(dtype=float)
    d_up = de * np.sin(q) + dn * np.cos(q)           # toward the zenith
    d_across = de * np.cos(q) - dn * np.sin(q)
    s_r = np.interp(alt, alt_grid, radial)
    s_t = np.interp(alt, alt_grid, tangential)
    d_px = np.hypot(s_r * d_up, s_t * d_across)
    weight = efficiency * _encircled(d_px, aperture_radius, psf_sigma) / \
        _encircled(0.0, aperture_radius, psf_sigma)

    for band, mags in target_mags.items():
        contrib = 10 ** (-0.4 * (pairs[f"mag_{band}"].to_numpy(dtype=float)
                                 - np.asarray(mags, dtype=float)[i])) * weight
        ratio = np.bincount(i, weights=np.nan_to_num(contrib), minlength=n)
        term = -2.5 * np.log10(1.0 + ratio)
        out[band] = np.where(np.isfinite(mags), term, np.nan)
    return out
