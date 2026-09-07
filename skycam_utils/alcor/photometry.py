"""Fixed-position RGB aperture and Gaussian PSF stellar photometry."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from astropy.table import Table
from astropy.time import Time

from .config import (
    _DATA_ROOT,
    ALCOR_AIRMASS_TERM, ALCOR_BRIGHT_CUT, ALCOR_COLOR_FLAT_TOL,
    ALCOR_NONLINEAR_THRESHOLD, ALCOR_SATURATION, ALCOR_ZEROPOINTS,
    ALCOR_ZEROPOINT_BANDS, _GAUSS_MIN_CHANNEL_PIXELS, _GAUSS_MIN_LUM_PIXELS
)
from .timeutils import _alcor_frame_time, _filename_ut_datetime, _sun_altitude
from .catalogs import (
    _alcor_star_labels, _catalog_value_to_python,
    alcor_photometry_reference_altaz
)
from .io import _corner_bias, load_alcor_fits
from .display import save_alcor_photometry_check_plot


def _annulus_background(image, xcen, ycen, aperture_radius, annulus_width):
    """
    Median background in the circular annulus around ``(xcen, ycen)``.

    The annulus runs from ``aperture_radius + 1`` to
    ``aperture_radius + 1 + annulus_width``. Returns NaN when the annulus falls
    entirely outside the image.
    """
    ny, nx = image.shape
    annulus_inner = aperture_radius + 1.0
    outer = annulus_inner + annulus_width
    x0 = max(0, int(np.floor(xcen - outer)))
    x1 = min(nx, int(np.ceil(xcen + outer)) + 1)
    y0 = max(0, int(np.floor(ycen - outer)))
    y1 = min(ny, int(np.ceil(ycen + outer)) + 1)
    if x0 >= x1 or y0 >= y1:
        return np.nan
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot(xx - xcen, yy - ycen)
    annulus = (rr > annulus_inner) & (rr <= outer)
    if not annulus.any():
        return np.nan
    return float(np.median(image[y0:y1, x0:x1][annulus]))



def _aperture_annulus_photometry(image, xcen, ycen, aperture_radius,
                                 annulus_width):
    """
    Circular aperture flux with a local median annulus background.
    """
    background = _annulus_background(image, xcen, ycen, aperture_radius,
                                     annulus_width)
    if not np.isfinite(background):
        return np.nan, np.nan
    ny, nx = image.shape
    x0 = max(0, int(np.floor(xcen - aperture_radius)))
    x1 = min(nx, int(np.ceil(xcen + aperture_radius)) + 1)
    y0 = max(0, int(np.floor(ycen - aperture_radius)))
    y1 = min(ny, int(np.ceil(ycen + aperture_radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return np.nan, np.nan
    yy, xx = np.mgrid[y0:y1, x0:x1]
    aperture = np.hypot(xx - xcen, yy - ycen) <= aperture_radius
    if not aperture.any():
        return np.nan, np.nan
    flux = float(np.sum(image[y0:y1, x0:x1][aperture] - background))
    return flux, background



def _aperture_saturated(image, xcen, ycen, aperture_radius, saturation):
    """
    Return True if any pixel within the circular aperture reaches ``saturation``.

    Operates on the raw image (the saturation ceiling is a raw-ADU value), so
    callers pass the unmodified cube channel rather than the bias-subtracted
    data.
    """
    ny, nx = image.shape
    x0 = max(0, int(np.floor(xcen - aperture_radius)))
    x1 = min(nx, int(np.ceil(xcen + aperture_radius)) + 1)
    y0 = max(0, int(np.floor(ycen - aperture_radius)))
    y1 = min(ny, int(np.ceil(ycen + aperture_radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    aperture = np.hypot(xx - xcen, yy - ycen) <= aperture_radius
    if not aperture.any():
        return False
    return bool(np.any(image[y0:y1, x0:x1][aperture] >= saturation))



def _gaussian_channel_amplitude(data, background, profile, fit_mask):
    """
    Linear least-squares amplitude of ``data - background`` projected onto a
    fixed unit-Gaussian ``profile`` over ``fit_mask``.

    With the PSF shape held fixed, the best-fit amplitude is the closed-form
    projection ``sum(g*d) / sum(g*g)`` over the unmasked pixels, so the linear
    wings (the only unmasked pixels for a bright star) set the amplitude and the
    suppressed core does not bias it. Returns NaN when the projection is
    degenerate (no profile weight in the mask).
    """
    g = profile[fit_mask]
    denom = float(np.sum(g * g))
    if denom <= 0.0:
        return np.nan
    d = data[fit_mask] - background
    return float(np.sum(g * d) / denom)



def _gaussian_psf_photometry(data, cube, lum_frame, xcen, ycen,
                             aperture_radius, annulus_width, mask_threshold):
    """
    Constrained-Gaussian PSF photometry for one star.

    Pins the PSF center and width from a luminance (channel-summed) fit with the
    non-linear core masked, then recovers each channel's amplitude as the linear
    projection of the background-subtracted, masked aperture data onto the fixed
    unit-Gaussian profile. Flux is the analytic Gaussian integral
    ``2*pi*A*sigma**2``.

    Parameters
    ----------
    data : (3, ny, nx) float `~numpy.ndarray`
        Bias-subtracted RGB cube.
    cube : (3, ny, nx) `~numpy.ndarray`
        Raw RGB cube; the linearity mask is computed on it because the threshold
        is a raw-ADU level.
    lum_frame : (ny, nx) `~numpy.ndarray`
        Precomputed luminance frame ``data.sum(axis=0)``.
    xcen, ycen : float
        WCS-predicted pixel position; the fit seed.
    aperture_radius, annulus_width : float
    mask_threshold : float
        Raw-ADU level at/above which a pixel is excluded from the fit.

    Returns
    -------
    dict or None
        Keys ``xcen``, ``ycen``, ``fwhm`` and per-channel ``flux_<ch>`` and
        ``background_<ch>``. None if the fit cannot be trusted.
    """
    ny, nx = data.shape[1:]
    # Box generous enough to hold the aperture around any allowed fitted center
    # (the center may drift up to aperture_radius from the seed).
    box_r = int(np.ceil(2.0 * aperture_radius)) + 1
    xi = int(np.floor(xcen))
    yi = int(np.floor(ycen))
    x0 = max(0, xi - box_r)
    x1 = min(nx, xi + box_r + 1)
    y0 = max(0, yi - box_r)
    y1 = min(ny, yi + box_r + 1)
    if x0 >= x1 or y0 >= y1:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1]
    raw_box = cube[:, y0:y1, x0:x1]

    # --- luminance shape fit (amp, center, sigma) over the linear core+wings ---
    rr_seed = np.hypot(xx - xcen, yy - ycen)
    in_aperture = rr_seed <= aperture_radius
    lum_linear = np.all(raw_box < mask_threshold, axis=0)
    lum_mask = in_aperture & lum_linear
    if int(lum_mask.sum()) < _GAUSS_MIN_LUM_PIXELS:
        return None

    lum_bkg = _annulus_background(lum_frame, xcen, ycen, aperture_radius,
                                  annulus_width)
    if not np.isfinite(lum_bkg):
        return None
    lum_sub = lum_frame[y0:y1, x0:x1] - lum_bkg

    xf = xx[lum_mask].astype(float)
    yf = yy[lum_mask].astype(float)
    zf = lum_sub[lum_mask]
    amp0 = float(np.max(zf))
    if not np.isfinite(amp0) or amp0 <= 0.0:
        amp0 = 1.0
    sigma0 = aperture_radius / 2.0

    def residual(p):
        amp, cx, cy, sigma = p
        model = amp * np.exp(-((xf - cx) ** 2 + (yf - cy) ** 2)
                             / (2.0 * sigma ** 2))
        return model - zf

    lower = [0.0, xcen - aperture_radius, ycen - aperture_radius, 1e-3]
    upper = [np.inf, xcen + aperture_radius, ycen + aperture_radius,
             aperture_radius]
    try:
        result = least_squares(residual, [amp0, xcen, ycen, sigma0],
                               bounds=(lower, upper), max_nfev=200)
    except (ValueError, RuntimeError):
        return None
    if not result.success:
        return None
    _, cx, cy, sigma = result.x
    if not np.all(np.isfinite([cx, cy, sigma])):
        return None
    if sigma <= 0.0 or sigma >= aperture_radius:
        return None
    if np.hypot(cx - xcen, cy - ycen) > aperture_radius:
        return None

    # --- per-channel amplitude from the linear wings, shape fixed ------------
    rr_fit = np.hypot(xx - cx, yy - cy)
    in_aperture_fit = rr_fit <= aperture_radius
    profile = np.exp(-rr_fit ** 2 / (2.0 * sigma ** 2))
    norm = 2.0 * np.pi * sigma ** 2

    out = {
        "xcen": float(cx),
        "ycen": float(cy),
        "fwhm": float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma),
    }
    for idx, channel in enumerate(("r", "g", "b")):
        ch_bkg = _annulus_background(data[idx], cx, cy, aperture_radius,
                                     annulus_width)
        if not np.isfinite(ch_bkg):
            return None
        ch_mask = in_aperture_fit & (raw_box[idx] < mask_threshold)
        if int(ch_mask.sum()) < _GAUSS_MIN_CHANNEL_PIXELS:
            return None
        amp_ch = _gaussian_channel_amplitude(
            data[idx, y0:y1, x0:x1], ch_bkg, profile, ch_mask)
        if not np.isfinite(amp_ch):
            return None
        out[f"flux_{channel}"] = amp_ch * norm
        out[f"background_{channel}"] = float(ch_bkg)
    return out



def _default_alcor_photometry_output(filename):
    stem = str(filename)
    for ext in (".fits.bz2", ".fits.gz", ".fits"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return Path(stem + "_phot.csv")



def _default_alcor_photometry_check_plot_output(filename):
    stem = str(filename)
    for ext in (".fits.bz2", ".fits.gz", ".fits"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    return Path(stem + "_phot.pdf")



def _flux_mag(flux):
    """Map a measured flux to ``(flux, mag)`` with the non-detection convention.

    A finite, positive flux yields ``(float(flux), -2.5*log10(flux))``. Any
    non-finite or non-positive flux is treated as a non-detection and recorded
    as ``(0.0, nan)`` so the star still appears in the output: a catalog star
    that is above the horizon but invisible (e.g. behind cloud) is itself a
    strong extinction signal, so it must not be silently dropped.
    """
    if np.isfinite(flux) and flux > 0.0:
        return float(flux), float(-2.5 * np.log10(flux))
    return 0.0, np.nan



def _aperture_measure(data, cube, x, y, aperture_radius, annulus_width,
                      saturation, channels):
    """Per-channel aperture flux/background/saturation at ``(x, y)``.

    Returns a dict with ``flux_<ch>``, ``background_<ch>`` and ``sat_<ch>`` for
    each channel (raw measurements; magnitudes and validity are the caller's
    decision).
    """
    cols = {}
    for index, channel in enumerate(channels):
        flux, background = _aperture_annulus_photometry(
            data[index], x, y, aperture_radius, annulus_width)
        cols[f"flux_{channel}"] = flux
        cols[f"background_{channel}"] = background
        cols[f"sat_{channel}"] = _aperture_saturated(
            cube[index], x, y, aperture_radius, saturation)
    return cols



def _gaussian_measure(data, cube, lum_frame, x, y, aperture_radius,
                      annulus_width, mask_threshold, saturation, channels):
    """Constrained-Gaussian per-channel measurement at ``(x, y)``.

    Returns a dict with the fitted ``xcen``, ``ycen``, ``fwhm`` and per-channel
    ``flux_<ch>``, ``background_<ch>`` and ``sat_<ch>`` (the saturation flag is
    evaluated at the fitted center), or None when the luminance fit fails.
    """
    fit = _gaussian_psf_photometry(
        data, cube, lum_frame, x, y, aperture_radius, annulus_width,
        mask_threshold)
    if fit is None:
        return None
    cols = {"xcen": fit["xcen"], "ycen": fit["ycen"], "fwhm": fit["fwhm"]}
    for index, channel in enumerate(channels):
        cols[f"flux_{channel}"] = fit[f"flux_{channel}"]
        cols[f"background_{channel}"] = fit[f"background_{channel}"]
        cols[f"sat_{channel}"] = _aperture_saturated(
            cube[index], fit["xcen"], fit["ycen"], aperture_radius, saturation)
    return cols



def _airmass(altitude_deg):
    """
    Kasten-Young airmass for an apparent altitude in degrees (scalar or array).
    """
    alt = np.asarray(altitude_deg, dtype=float)
    return 1.0 / (np.sin(np.radians(alt))
                  + 0.50572 * (alt + 6.07995) ** -1.6364)



def _catalog_calibration_map():
    """
    Map each named-catalog star label to its catalog colors and magnitudes.

    Keyed by the same labels :func:`_alcor_star_labels` assigns to photometry
    rows, so the map joins directly against a photometry DataFrame. Each value is
    a dict with ``BV`` (the B-V color) and the catalog Johnson magnitudes ``R``
    (= V-(V-R)), ``V`` (= Vmag) and ``B`` (= V+(B-V)); any entry whose catalog
    color is missing is ``nan``.
    """
    catpath = _DATA_ROOT / "bright_star_sloan_named.fits"
    cat = Table.read(str(catpath))
    labels = _alcor_star_labels(cat)
    out = {}
    for label, row in zip(labels, cat):
        def _value(name):
            value = _catalog_value_to_python(row[name])
            return np.nan if value is None else float(value)
        v = _value("Vmag")
        bv = _value("B-V")
        vr = _value("V-R")
        out[label] = {"BV": bv, "V": v, "R": v - vr, "B": v + bv}

    # Variables contribute a colour but NO catalog magnitude. That is the whole
    # point: cal_* is the light curve, while ext_* (= cal - catalog) would
    # conflate cloud extinction with the star's own variation, so it must stay
    # nan -- which falls out of subtracting nan. Calibration-catalog entries win
    # on a label collision, since those stars do have a defensible catalog mag.
    varpath = _DATA_ROOT / "bright_variable_vsx.fits"
    var = Table.read(str(varpath))
    for label, row in zip(_alcor_star_labels(var), var):
        if label in out:
            continue
        bv = _catalog_value_to_python(row["B-V"])
        out[label] = {"BV": np.nan if bv is None else float(bv),
                      "V": np.nan, "R": np.nan, "B": np.nan}
    return out



def _zeropoint_row_params(df, time):
    """
    Per-row ``(zp, color_coeff)`` arrays for each band.

    The zeropoint epoch is resolved from ``time`` when given, else per row from
    an ``OBSTIME`` column when present, else the most recent epoch. Returns
    ``(zp, color)`` where each is a ``{band: ndarray}`` aligned to ``df``'s rows.
    """
    n = len(df)
    epoch_jds = np.array([Time(z["epoch"], scale="utc").jd
                          for z in ALCOR_ZEROPOINTS])
    if time is not None:
        row_jd = np.full(n, Time(time).jd)
    elif n and "OBSTIME" in df.columns:
        row_jd = Time(np.asarray(df["OBSTIME"], dtype="datetime64[ns]")).jd
        row_jd = np.atleast_1d(np.asarray(row_jd, dtype=float))
    else:
        row_jd = np.full(n, epoch_jds.max())
    # nearest epoch per row; iterate ascending jd so a tie resolves to the more
    # recent epoch (matching alcor_zeropoint / alcor_calibration).
    pick = np.zeros(n, dtype=int)
    best = np.full(n, np.inf)
    for k in np.argsort(epoch_jds):
        dist = np.abs(row_jd - epoch_jds[k])
        take = dist <= best
        pick[take] = k
        best[take] = dist[take]
    zp, color = {}, {}
    for band in ALCOR_ZEROPOINT_BANDS:
        zp_vals = np.array([z[band]["zp"] for z in ALCOR_ZEROPOINTS])
        cc_vals = np.array([z[band]["color_coeff"] for z in ALCOR_ZEROPOINTS])
        zp[band] = zp_vals[pick]
        color[band] = cc_vals[pick]
    return zp, color



def alcor_calibrate_photometry(df, time=None):
    """
    Add calibrated catalog-system magnitudes and cloud-extinction offsets.

    For every instrument magnitude column in ``df`` (``mag_{r,g,b}`` and any
    ``_ap``/``_gauss`` suffixed variants) this adds two columns:

    ``cal_{band}[suffix]``
        the magnitude on the catalog system,
        ``(instr_mag - ALCOR_AIRMASS_TERM*airmass) + zp + color_coeff*(B-V)``,
        ``nan`` where the instrument mag is non-finite or brighter than
        ``ALCOR_BRIGHT_CUT`` (the CMOS non-linear regime, where the calibration
        is invalid).
    ``ext_{band}[suffix]``
        ``cal - catalog_mag``: the offset from the star's catalog magnitude --
        the line-of-sight extinction (e.g. cloud attenuation) in magnitudes,
        positive when the star is dimmer than catalog.

    Zeropoints come from :func:`alcor_zeropoint`; the epoch is resolved from
    ``time`` when given, else per row from an ``OBSTIME`` column when present,
    else the most recent epoch. Star names come from a ``name`` column when
    present, else the index. ``B-V`` and the catalog Johnson R/V/B are looked up
    by name (channel->catalog G->V, R->R, B->B); stars absent from the catalog or
    lacking the needed color get ``nan`` -- except that a colour-flat band (see
    :data:`ALCOR_COLOR_FLAT_TOL`) falls back to ``B-V = 0`` rather than discard
    the measurement. Variable stars carry a colour but no catalog magnitude, so
    they get a real ``cal_*`` -- the light curve -- and ``ext_*`` of ``nan``,
    since differencing against a catalog magnitude would conflate cloud
    extinction with the star's own variation. Requires an ``altitude`` column.
    Returns a new DataFrame.
    """
    if "altitude" not in df.columns:
        raise ValueError(
            "alcor_calibrate_photometry requires an 'altitude' column")
    df = df.copy()
    names = (df["name"] if "name" in df.columns
             else df.index.to_series()).astype(str)
    cmap = _catalog_calibration_map()
    bv = names.map(lambda name: cmap.get(name, {}).get("BV", np.nan)).to_numpy(
        dtype=float)
    catmag = {
        band: names.map(
            lambda name, cat=cat: cmap.get(name, {}).get(cat, np.nan)
        ).to_numpy(dtype=float)
        for band, cat in ALCOR_ZEROPOINT_BANDS.items()
    }
    airmass = _airmass(df["altitude"].to_numpy(dtype=float))
    zp, color = _zeropoint_row_params(df, time)
    for band in ALCOR_ZEROPOINT_BANDS:
        # For a colour-flat band a missing B-V costs less than the photometric
        # scatter, so assume zero rather than throw the measurement away.
        flat = bool(np.all(np.abs(color[band]) <= ALCOR_COLOR_FLAT_TOL))
        bv_band = np.where(np.isfinite(bv), bv, 0.0) if flat else bv
        for col in (f"mag_{band}", f"mag_{band}_ap", f"mag_{band}_gauss"):
            if col not in df.columns:
                continue
            suffix = col[len(f"mag_{band}"):]
            instr = df[col].to_numpy(dtype=float)
            cal = (instr - ALCOR_AIRMASS_TERM * airmass
                   + zp[band] + color[band] * bv_band)
            cal = np.where(np.isfinite(instr) & (instr > ALCOR_BRIGHT_CUT),
                           cal, np.nan)
            df[f"cal_{band}{suffix}"] = cal
            df[f"ext_{band}{suffix}"] = cal - catmag[band]
    return df



def alcor_star_photometry(filename, output_file=None, aperture_radius=4.0,
                          annulus_width=1.0, min_altitude=20.0,
                          vmag_limit=5.5, refraction=True, masks_dir=None,
                          variables=True,
                          check_plot=False, check_radius=680,
                          sun_alt_max=-12.0, saturation=ALCOR_SATURATION,
                          gaussian=False,
                          mask_threshold=ALCOR_NONLINEAR_THRESHOLD,
                          both=False, frame=None):
    """
    Measure fixed-position aperture photometry for bright named stars.

    The image is loaded with :func:`load_alcor_fits` using bad-pixel repair, then
    a per-channel bias level is subtracted from the median of the four 10x10
    image corners. Catalog stars from ``bright_star_sloan_named.fits`` with
    ``Vmag <= vmag_limit`` and altitude above ``min_altitude`` are projected
    into the raw image via the frame WCS. Each channel is measured with a
    circular aperture and a surrounding annulus. Each channel also carries a
    ``sat_*`` flag, True when any raw pixel inside the aperture reaches
    ``saturation`` (the 15-bit ceiling by default), so saturated measurements
    can be filtered downstream without discarding the unsaturated channels.

    When ``gaussian`` is True, photometry instead fits a circular Gaussian whose
    center and width are pinned from a luminance (channel-summed) fit with the
    non-linear core masked (raw pixels at/above ``mask_threshold`` excluded), and
    recovers each channel's amplitude from the linear wings. The reported flux is
    the analytic Gaussian integral, which is robust to the CMOS non-linearity
    that suppresses aperture-sum flux for bright stars before saturation. The
    luminance FWHM is reported in the ``fwhm`` column (NaN in aperture mode).

    Every catalog star above ``min_altitude`` produces a row in every frame,
    even when it is not detected: a star that is above the horizon but invisible
    (e.g. behind cloud) carries ``flux = 0`` and ``mag = NaN`` per channel rather
    than being dropped, because that non-detection is itself a strong extinction
    signal (and the measured ``background_*`` is still recorded). The only stars
    absent from the output are those below ``min_altitude`` or fainter than
    ``vmag_limit``. The per-channel ``background_*`` is finite for a measurable
    position even at a non-detection, and NaN only when the position is off-frame.

    When ``both`` is True, every star is measured with *both* methods in a single
    pass and written to one combined CSV: the aperture columns are suffixed
    ``_ap`` and the Gaussian columns ``_gauss``, alongside the shared
    WCS-predicted ``xcen``/``ycen`` and the Gaussian-fitted
    ``xcen_gauss``/``ycen_gauss``/``fwhm``. The aperture flux is 0 at a
    non-detection; the Gaussian columns are NaN when the fit cannot run at all
    (a structural failure, distinct from a measured zero). Rows sort by
    ``flux_g_ap``. ``both`` takes precedence over ``gaussian``.

    With ``variables`` (the default) the bright-variable catalog is measured
    alongside the calibration stars and every row carries a ``variable`` flag.
    Those stars have no single catalog magnitude, so their ``ext_*`` is NaN while
    ``cal_*`` -- the light curve -- is real. See
    :func:`alcor_photometry_reference_altaz`.

    Every measured magnitude is calibrated to the catalog system via
    :func:`alcor_calibrate_photometry` (using the frame time to resolve the
    zeropoint epoch), adding per-channel ``cal_*`` (calibrated catalog-system
    magnitude: G->V, R->R, B->B) and ``ext_*`` (the calibrated-minus-catalog
    offset, i.e. the line-of-sight cloud extinction in magnitudes). Both are NaN
    for measurements brighter than ``ALCOR_BRIGHT_CUT`` (CMOS non-linear regime)
    or for stars lacking a catalog color.

    Pass ``frame`` as an already-loaded ``(cube, wcs, mask)`` tuple (the
    :func:`load_alcor_fits` return) to skip the internal load. The cube must
    already be bad-pixel-repaired, since that is what the internal load does.
    This exists so a batch driver that needs the same frame for something else
    (see :func:`alcor_process_night`) decompresses each file only once; the Sun
    check still runs from the filename, so a rejected frame is still rejected.

    Returns
    -------
    phot : `pandas.DataFrame`
        Rows indexed by star name. Columns are ``altitude``, ``azimuth``,
        ``variable``, ``xcen``, ``ycen`` and per-channel ``flux_*``, ``mag_*``, ``cal_*``,
        ``ext_*``, ``background_*``, ``sat_*`` for ``r``, ``g``, ``b`` (suffixed
        ``_ap``/``_gauss`` in ``both`` mode). One row per catalog star above
        ``min_altitude``; non-detections carry ``flux = 0`` / ``mag = NaN``. Rows
        are sorted by descending ``flux_g``. Empty only when the frame is
        rejected (Sun above ``sun_alt_max``).
    output_file : `~pathlib.Path` or None
        CSV path written, or None when the frame is rejected.
    """
    if aperture_radius <= 0:
        raise ValueError("aperture_radius must be positive")
    if annulus_width <= 0:
        raise ValueError("annulus_width must be positive")
    channels = ("r", "g", "b")
    if both:
        columns = ["altitude", "azimuth", "variable", "xcen", "ycen"]
        for channel in channels:
            columns += [f"flux_{channel}_ap", f"mag_{channel}_ap",
                        f"cal_{channel}_ap", f"ext_{channel}_ap",
                        f"background_{channel}_ap", f"sat_{channel}_ap"]
        columns += ["xcen_gauss", "ycen_gauss", "fwhm"]
        for channel in channels:
            columns += [f"flux_{channel}_gauss", f"mag_{channel}_gauss",
                        f"cal_{channel}_gauss", f"ext_{channel}_gauss",
                        f"background_{channel}_gauss", f"sat_{channel}_gauss"]
        sort_key = "flux_g_ap"
    else:
        columns = ["altitude", "azimuth", "variable", "xcen", "ycen", "fwhm"]
        for channel in channels:
            columns += [f"flux_{channel}", f"mag_{channel}",
                        f"cal_{channel}", f"ext_{channel}",
                        f"background_{channel}", f"sat_{channel}"]
        sort_key = "flux_g"
    empty = pd.DataFrame(columns=columns)
    empty.index.name = "name"

    filename = Path(filename)
    time = _alcor_frame_time(filename)
    if time is None:
        raise ValueError(f"could not determine frame time from {filename}")
    sun_alt = _sun_altitude(time)
    if sun_alt > sun_alt_max:
        print(
            f"Warning: rejecting {filename}: Sun altitude {sun_alt:.1f} deg "
            f"is greater than {sun_alt_max:.1f} deg.",
            file=sys.stderr,
        )
        return empty, None

    if frame is None:
        cube, wcs, _ = load_alcor_fits(filename, badpix="repair",
                                       masks_dir=masks_dir)
    else:
        cube, wcs = frame[0], frame[1]
    bias = _corner_bias(cube, size=10)
    data = cube.astype(float, copy=False) - bias[:, None, None]

    cat = alcor_photometry_reference_altaz(
        time, vmag_limit=vmag_limit, min_alt=min_altitude,
        refraction=refraction, variables=variables,
    )
    # all_world2pix with quiet=True returns the best (possibly unconverged)
    # estimate instead of warning: the iterative SIP/radial inverse fails its
    # tight tolerance for stars at alt <~ 1 deg (largest radii, steepest radial
    # distortion), but the returned pixel is still good to well under a pixel and
    # those horizon stars are the least critical. Equivalent to world_to_pixel_values
    # otherwise (verified byte-identical above the horizon).
    xcen, ycen = wcs.all_world2pix(cat["Az"], cat["Alt"], 0, quiet=True)
    xcen = np.asarray(xcen, dtype=float)
    ycen = np.asarray(ycen, dtype=float)

    rows = []
    labels = []
    cat_labels = _alcor_star_labels(cat)
    lum_frame = data.sum(axis=0) if (gaussian or both) else None
    for i, (x, y) in enumerate(zip(xcen, ycen)):
        base = {"altitude": float(cat["Alt"][i]),
                "azimuth": float(cat["Az"][i]),
                "variable": bool(cat["Variable"][i])
                            if "Variable" in cat.colnames else False}
        if both:
            ap = _aperture_measure(data, cube, x, y, aperture_radius,
                                   annulus_width, saturation, channels)
            gfit = _gaussian_measure(data, cube, lum_frame, x, y,
                                     aperture_radius, annulus_width,
                                     mask_threshold, saturation, channels)
            row = dict(base, xcen=float(x), ycen=float(y))
            for channel in channels:
                flux, mag = _flux_mag(ap[f"flux_{channel}"])
                row[f"flux_{channel}_ap"] = flux
                row[f"mag_{channel}_ap"] = mag
                row[f"background_{channel}_ap"] = ap[f"background_{channel}"]
                row[f"sat_{channel}_ap"] = ap[f"sat_{channel}"]
            if gfit is not None:
                row["xcen_gauss"] = gfit["xcen"]
                row["ycen_gauss"] = gfit["ycen"]
                row["fwhm"] = gfit["fwhm"]
                for channel in channels:
                    flux, mag = _flux_mag(gfit[f"flux_{channel}"])
                    row[f"flux_{channel}_gauss"] = flux
                    row[f"mag_{channel}_gauss"] = mag
                    row[f"background_{channel}_gauss"] = \
                        gfit[f"background_{channel}"]
                    row[f"sat_{channel}_gauss"] = gfit[f"sat_{channel}"]
            else:
                # the Gaussian fit could not run: no estimate for this method,
                # so its columns are NaN (the aperture flux above still carries
                # the detection / non-detection). A genuinely empty frame yields
                # a degenerate fit with flux 0, not a None fit, so this branch is
                # a structural failure, not the cloud non-detection signal.
                row["xcen_gauss"] = np.nan
                row["ycen_gauss"] = np.nan
                row["fwhm"] = np.nan
                for channel in channels:
                    row[f"flux_{channel}_gauss"] = np.nan
                    row[f"mag_{channel}_gauss"] = np.nan
                    row[f"background_{channel}_gauss"] = np.nan
                    row[f"sat_{channel}_gauss"] = np.nan
        elif gaussian:
            gfit = _gaussian_measure(data, cube, lum_frame, x, y,
                                     aperture_radius, annulus_width,
                                     mask_threshold, saturation, channels)
            if gfit is None:
                # the fit could not run: keep the star at its WCS-predicted
                # position with NaN measurements (no estimate available).
                row = dict(base, xcen=float(x), ycen=float(y), fwhm=np.nan)
                for channel in channels:
                    row[f"flux_{channel}"] = np.nan
                    row[f"mag_{channel}"] = np.nan
                    row[f"background_{channel}"] = np.nan
                    row[f"sat_{channel}"] = np.nan
            else:
                row = dict(base, xcen=gfit["xcen"], ycen=gfit["ycen"],
                           fwhm=gfit["fwhm"])
                for channel in channels:
                    flux, mag = _flux_mag(gfit[f"flux_{channel}"])
                    row[f"flux_{channel}"] = flux
                    row[f"mag_{channel}"] = mag
                    row[f"background_{channel}"] = gfit[f"background_{channel}"]
                    row[f"sat_{channel}"] = gfit[f"sat_{channel}"]
        else:
            ap = _aperture_measure(data, cube, x, y, aperture_radius,
                                   annulus_width, saturation, channels)
            row = dict(base, xcen=float(x), ycen=float(y), fwhm=np.nan)
            for channel in channels:
                flux, mag = _flux_mag(ap[f"flux_{channel}"])
                row[f"flux_{channel}"] = flux
                row[f"mag_{channel}"] = mag
                row[f"background_{channel}"] = ap[f"background_{channel}"]
                row[f"sat_{channel}"] = ap[f"sat_{channel}"]
        rows.append(row)
        labels.append(cat_labels[i])

    phot = pd.DataFrame(rows, index=labels, columns=columns)
    phot.index.name = "name"
    phot = alcor_calibrate_photometry(phot, time=time)
    phot = phot.sort_values(sort_key, ascending=False, na_position="last")
    output_file = (_default_alcor_photometry_output(filename)
                   if output_file is None else Path(output_file))
    phot.to_csv(output_file)
    if check_plot:
        check_plot_file = (_default_alcor_photometry_check_plot_output(filename)
                           if check_plot is True else Path(check_plot))
        save_alcor_photometry_check_plot(
            filename,
            phot,
            check_plot_file,
            aperture_radius=aperture_radius,
            annulus_width=annulus_width,
            radius=check_radius,
        )
    return phot, output_file



def collect_alcor_photometry(inputs):
    """
    Collect per-star photometry from a set of ``*_phot.csv`` files.

    Each input file is one frame's :func:`alcor_star_photometry` output. The
    observation time is parsed from the file's ``YYYY_MM_DD__HH_MM_SS``
    filename stamp (local MST, so UT = stamp + 7h); the CSVs carry no time
    information internally, so files whose names do not parse are skipped with
    a warning, as are unreadable/malformed CSVs.

    Parameters
    ----------
    inputs : str, Path, or iterable of str/Path
        A directory to glob for ``*_phot.csv``, or the CSV paths themselves.

    Returns
    -------
    ~pandas.DataFrame
        The combined photometry with ``name`` as a regular column and a UT
        ``OBSTIME`` datetime column, sorted by ``name`` then ``OBSTIME`` so
        ``df.groupby("name")`` yields each star's time-ordered measurements.

    Raises
    ------
    ValueError
        If no usable input files remain after skipping.
    """
    if isinstance(inputs, (str, Path)) and Path(inputs).is_dir():
        files = sorted(Path(inputs).glob("*_phot.csv"))
    else:
        files = [Path(f) for f in inputs]

    frames = []
    for f in files:
        obstime = _filename_ut_datetime(f)
        if obstime is None:
            print(f"Warning: skipping {f.name}: no parseable timestamp "
                  "in filename", file=sys.stderr)
            continue
        try:
            phot = pd.read_csv(f)
        except (OSError, ValueError, UnicodeDecodeError,
                pd.errors.ParserError) as exc:
            print(f"Warning: skipping {f.name}: {exc}", file=sys.stderr)
            continue
        if "name" not in phot.columns:
            print(f"Warning: skipping {f.name}: no 'name' column",
                  file=sys.stderr)
            continue
        phot.insert(1, "OBSTIME", pd.Timestamp(obstime))
        frames.append(phot)

    if not frames:
        raise ValueError(f"No usable *_phot.csv files in {inputs}.")

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["name", "OBSTIME"], ignore_index=True)
