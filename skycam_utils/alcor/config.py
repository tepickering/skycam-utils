"""Camera constants, per-epoch WCS calibrations, and photometric zeropoints."""

from importlib.resources import files

import numpy as np
from astropy.time import Time

import astropy.units as u



# The packaged data assets (catalogs, bad-pixel and horizon masks) live at the
# top of the distribution, in ``skycam_utils/data``. That is one level above
# this subpackage: back when alcor was a single top-level module
# ``files(__package__)`` landed there, but from inside ``skycam_utils.alcor`` it
# would resolve to ``skycam_utils/alcor/data``, so the root is named explicitly.
_DATA_ROOT = files("skycam_utils") / "data"

# Nominal raw-frame geometry defaults. The real geometry is per-epoch (see
# ALCOR_CALIBRATIONS) and is threaded explicitly through every call path;
# ALCOR_HORIZON_RADIUS is only a fallback default for the rare arg-less call.
# It is the plate scale: the pixel radius from the optical-axis pixel at which
# the angular distance from the axis reaches 90 deg (altitude=0 when axis_tilt
# is zero; cdelt = 90/horizon_radius deg/px). ALCOR_RADIUS is the default
# display-crop half-width for plot_alcor_fits.
ALCOR_RADIUS = 680

ALCOR_HORIZON_RADIUS = 747

# Raw-ADU ceiling: the OMEA 8C delivers 15-bit data, so pixels peg at 2**15 - 1.
ALCOR_SATURATION = 32767

ALCOR_NONLINEAR_THRESHOLD = 15000   # raw ADU; per-pixel non-linearity onset


# Photometric-calibration reference exposure. The zeropoints (ALCOR_ZEROPOINTS)
# were derived from frames taken at the dark-condition default of 20 s, so the
# instrumental fluxes are effectively counts/(20 s). Counts are linear in
# exposure, so raw counts from a frame at a different EXPOSURE are scaled to this
# reference before the zeropoint is applied (see plot_alcor_sky_brightness).
ALCOR_CALIB_EXPTIME = 20.0   # seconds

# Sky-brightness saturation / strong-non-linearity ceiling: raw pixels at or
# above this are clipped or badly non-linear. Lower than the 15-bit hard ceiling
# ALCOR_SATURATION used for star sat_* flags, because the per-pixel response
# departs from linear well before clipping.
#
# NOT applied by default any more. Blanking these pixels was actively
# misleading on a surface-brightness map: a NaN renders as the axes background,
# so the very brightest sources -- a saturated star or planet crossing the
# frame -- appeared as DARK holes when the truth is that they are too bright to
# measure. Keeping the value is the lesser error. Clipping loses flux, so a
# saturated pixel reads slightly too FAINT, which makes the rendered peak a
# lower bound on the real surface brightness rather than an inversion of it.
# Pass an explicit `saturation` to restore the mask.
ALCOR_SB_SATURATION = 25000   # raw ADU


# Bad-pixel detection is restricted to sky pixels. The 5 px median high-pass that
# isolates hot pixels also fires on any sharp edge, and the horizon is full of
# them -- terrain, buildings, ground lights, the rim itself. Measured on the
# 2026-01-11 night median, 78% of the z>25 candidates fell inside the horizon
# mask or within a few pixels of its rim, so the mask was mostly a map of the
# skyline rather than of the sensor (which also made the NBAD* aging counts
# meaningless). The horizon mask is dilated by this many pixels so the rim
# gradient itself is excluded, not just the not-sky side of it.
ALCOR_BADPIX_RIM_DILATION = 4   # pixels


# Radius in pixels, about the OPTICAL AXIS, of the camera's illuminated image
# circle. Beyond it the fisheye simply delivers no light: measured on four night
# medians spanning 2026-01 to 2026-06, the median signal above bias falls from
# ~180 counts at radius 676-679 to 42 at 679-682 and then flat at a few counts,
# an unambiguous optical edge at the same radius on every night. It sits at
# altitude ~-2.5 deg, INSIDE the nominal ALCOR_HORIZON_RADIUS (747) where alt=0
# would fall without distortion -- the sensor reaches a couple of degrees below
# the true horizon and no further. Surface-brightness maps blank everything
# outside it, because dividing a near-zero signal by a solid angle yields a
# confident-looking ~25 mag/arcsec^2 that is an artifact, not dark sky.
# Numerically equal to ALCOR_RADIUS, but that one is a display-crop half-width;
# this is a property of the optics, so they are kept separate.
ALCOR_FIELD_RADIUS = 680   # pixels


# A band whose colour coefficient is within this of zero is treated as colour
# flat, so a star with no catalogued B-V still gets a calibrated magnitude
# instead of a nan. G's coefficient is -0.038, so even a very red star (B-V=1.5)
# costs under 0.06 mag -- below the per-frame photometric scatter. R (-0.343)
# and B (+0.47) are nowhere near flat and correctly stay nan without a colour.
# This matters for the variable catalog, where 42 of 642 stars have no B-V.
ALCOR_COLOR_FLAT_TOL = 0.05   # mag per mag of B-V



# ...and a disc this big around the north celestial pole is excluded too. The
# detector assumes a night median is trail-free, which holds everywhere except
# at the pole: Polaris (V=1.98) moves only ~10 px in a night, so its trail
# survives the median and gets flagged as a cluster of hot pixels -- a different
# cluster every night, since the trail lands at a different hour angle. The disc
# costs ~700 px of sensor area out of 2M.
ALCOR_BADPIX_POLE_RADIUS = 15   # pixels


# Fixed sky-brightness sampling apertures for the nightly archive summary
# (alcor_process_night). Each is the median surface brightness of the pixels
# within ALCOR_SB_APERTURE_RADIUS degrees of a fixed (azimuth, altitude), so the
# same patch of sky is reported for every frame of every night: the zenith as the
# dark-sky reference, and two low-altitude cones aimed at the Tucson and Nogales
# light domes.
ALCOR_SB_APERTURE_RADIUS = 5.0    # deg, angular radius of each sampling cone

ALCOR_SB_TARGETS = {              # column name -> (azimuth, altitude) in deg
    "allsky_mv_zenith": (0.0, 90.0),
    "allsky_mv_tucson": (0.0, 15.0),
    "allsky_mv_nogales": (190.0, 15.0),
}

ALCOR_SB_TARGET_DESCRIPTIONS = {
    "allsky_mv_zenith":
        "Allsky visible magnitude at zenith, median within 5 deg radius aperture",
    "allsky_mv_tucson":
        "Allsky visible magnitude toward Tucson, median within 5 deg radius "
        "aperture centered at az=0, alt=15",
    "allsky_mv_nogales":
        "Allsky visible magnitude toward Nogales, median within 5 deg radius "
        "aperture centered at az=190, alt=15",
}

ALCOR_SB_TARGET_UNIT = "V mag/arcsec^2"


# --- cloud-extinction maps (alcor/extinction.py) -------------------------
# The sensor's raw (ny, nx). Extinction maps are built from photometry CSVs
# with no frame in hand, so the shape cannot be read from a header.
ALCOR_EXT_FRAME_SHAPE = (1411, 1422)

# 10 frames (~5 min) is where frame-to-frame scatter still beats down as
# uncorrelated noise: measured sigma(N) tracks sigma(1)/sqrt(N) to within 5% out
# to N=12 and has clearly departed by N=30, so averaging longer buys little and
# costs update cadence.
ALCOR_EXT_NFRAMES = 10
ALCOR_EXT_CADENCE = 27.5          # s between frames, for labelling only

# Instrumental-magnitude window free of BOTH end biases: brighter than -11
# enters the CMOS non-linear regime, fainter than -9.5 picks up the faint-end
# bias. See ALCOR_BRIGHT_CUT for the harder limit applied to cal_*/ext_*.
ALCOR_EXT_MAG_WINDOW = (-11.0, -9.5)

ALCOR_EXT_MIN_ALTITUDE = 20.0     # matches alcor_star_photometry's default
ALCOR_EXT_GRID_STEP = 1.0         # deg; well under the smoothing kernel
ALCOR_EXT_KERNEL_SIGMA = 8.0      # deg, great-circle smoothing scale
ALCOR_EXT_KERNEL_CUT = 3.0        # truncate the kernel at 3 sigma
ALCOR_EXT_MIN_WEIGHT = 1.5        # blank a cell below this summed weight
ALCOR_EXT_MIN_BLOCK_FRAMES = 3    # frames a star needs for a block mean
ALCOR_EXT_LOST_MIN_FRAMES = 5     # frames undetected before it counts as lost

# Great-circle radius in DEGREES for alcor_extinction_at. The map is already
# smoothed on ALCOR_EXT_KERNEL_SIGMA, so this adds little smoothing -- its job
# is robustness against a blank pixel, a resampling edge, or a pointing that
# lands just outside the valid region.
ALCOR_EXT_LOOKUP_RADIUS = 1.0

# Fixed colour-scale top, so maps from different times and nights compare
# directly and a clear sky always renders the same.
ALCOR_EXT_VMAX = 1.5

# The fixed cones above sample named directions; allsky_mv_best instead reports the
# DARKEST such cone anywhere above this altitude. The zenith is not a reliable
# darkness measure -- the Milky Way transits through it -- so the darkest patch is
# what says how dark the site actually got. It is the same statistic in the same
# units (median inside a 5 deg cone), just at a floating position rather than a
# fixed one, so it is directly comparable to the three named columns.
ALCOR_SB_BEST_MIN_ALTITUDE = 30.0   # deg


# Adopted photometric calibration (see ALCOR_ZEROPOINTS). A single achromatic
# extinction term applies to all three bands, and instrument magnitudes brighter
# than the bright cut are in the CMOS non-linear regime where the calibration is
# invalid. The airmass term was established in docs/scripts/zeropoint_calib.py.
# The bright cut is set by docs/scripts/nonlin_binned.py: with intra-pixel jitter
# averaged out (15-min per-star medians), the bright-star magnitude deficit onsets
# near -11.5 but stays small and ~linear out to -12.5, then accelerates steeply.
# -12.5 is the linear-regime cutoff; brighter than that is sparse/inconclusive and
# dropped pending more calibration nights.
ALCOR_AIRMASS_TERM = 0.40   # mag/airmass, single term for R/G/B

ALCOR_BRIGHT_CUT = -12.5    # instr mag; brighter is non-linear, calibration void


# minimum unmasked pixel counts for the Gaussian fits
_GAUSS_MIN_LUM_PIXELS = 8            # 4-parameter luminance shape fit

_GAUSS_MIN_CHANNEL_PIXELS = 3        # 1-parameter per-channel amplitude


# Time-indexed lens calibrations. Each epoch holds the raw-frame geometry as
# absolutes: the optical-axis pixel (xcen, ycen) in the raw FITS frame (the
# zenith pixel when axis_tilt is zero), the azimuth
# rotation, the radial_coeffs (k1, k3, k5), and the horizon_radius (pixels from
# zenith to alt=0). An optional "tangential_coeffs": (P1, P2) holds the
# Brown-Conrady decentering (sensor-tilt) terms, dimensionless like the k's;
# epochs without the key mean (0.0, 0.0) (alcor_calibration fills the default).
# An optional "axis_tilt": (t_n, t_e) holds the optical-axis tilt from the
# zenith as components toward north and east, in DEGREES (the axis points at
# alt 90 - hypot(t_n, t_e), az atan2(t_e, t_n)); epochs without the key mean
# (0.0, 0.0). With nonzero tilt, xcen/ycen is the optical-axis pixel (the
# distortion center), not the zenith pixel.
# The camera geometry drifts over time (mount/focus), so the
# epoch nearest in time to an image is used (see alcor_calibration). Add a new
# epoch by pasting the dict that fit_alcor_wcs prints. `epoch` is the calibration
# night at day precision (UT, not local night -- do not "fix" it to local).
ALCOR_CALIBRATIONS = [
    {"epoch": "2024-09-05", "xcen": 703.586, "ycen": 704.803, "rotation": -0.9642,
     "radial_coeffs": (1.0, 0.047841687068536774, 0.1163038015749883),
     "tangential_coeffs": (-0.0003040188173761858, 0.0006700812069651288),
     "axis_tilt": (-0.8225394950126477, -0.6160139387466032),
     "horizon_radius": 747.2},
    # The camera was not moved or changed between 2024 and 2026; this epoch is
    # consistent with 2024 within the fit uncertainty (center stable ~1px, axis
    # tilt agrees to ~0.03 deg / ~1.5 deg in lean azimuth, ~0.05 deg rotation
    # drift). It is kept as a separate entry so per-era geometry is supported
    # if the camera is ever moved/refocused.
    {"epoch": "2026-05-19", "xcen": 703.537, "ycen": 703.832, "rotation": -1.0177,
     "radial_coeffs": (1.0, 0.05337073600079686, 0.1111394504753296),
     "tangential_coeffs": (-0.0005518035497827486, 0.0006929046299086498),
     "axis_tilt": (-0.860477755807792, -0.6088848881222786),
     "horizon_radius": 747.2},
]



def _calibration_epochs():
    """
    Return [(Time, calibration_dict), ...] for the configured epochs.
    """
    return [(Time(c["epoch"], scale="utc"), c) for c in ALCOR_CALIBRATIONS]



def alcor_calibration(time=None):
    """
    Return the calibration dict whose epoch is nearest in time to ``time``.

    ``time`` is an astropy ``Time``. An exact tie resolves to the more recent
    epoch. ``time=None`` returns the most recent epoch (the default for
    time-agnostic calls). The returned dict is a copy and may be mutated freely;
    ``tangential_coeffs`` and ``axis_tilt`` are filled with ``(0.0, 0.0)`` for
    epochs that omit them.
    """
    epochs = _calibration_epochs()
    if time is None:
        cal = dict(max(epochs, key=lambda e: e[0].jd)[1])
    else:
        jds = np.array([e[0].jd for e in epochs])
        dt = np.abs(jds - Time(time).jd)
        # primary: smallest |dt|; tie-break: largest jd (more recent)
        order = np.lexsort((-jds, dt))
        cal = dict(epochs[order[0]][1])
    cal.setdefault("tangential_coeffs", (0.0, 0.0))
    cal.setdefault("axis_tilt", (0.0, 0.0))
    return cal



# Module-level defaults track the most-recent epoch so existing default-argument
# references (in _predict_pixels, build_alcor_wcs, etc.) keep working unchanged.
_LATEST_CALIBRATION = alcor_calibration()

ALCOR_ROTATION = _LATEST_CALIBRATION["rotation"]

ALCOR_XCEN = _LATEST_CALIBRATION["xcen"]

ALCOR_YCEN = _LATEST_CALIBRATION["ycen"]

ALCOR_RADIAL_COEFFS = _LATEST_CALIBRATION["radial_coeffs"]

ALCOR_TANGENTIAL_COEFFS = _LATEST_CALIBRATION["tangential_coeffs"]

ALCOR_AXIS_TILT = _LATEST_CALIBRATION["axis_tilt"]



# Time-indexed photometric zeropoints calibrated on clear dark nights. They map
# instrument R/G/B aperture magnitudes to catalog Johnson R/V/B via
#   cat_mag = (instr_mag - ALCOR_AIRMASS_TERM*airmass) + zp + color_coeff*(B-V)
# with the channel->catalog assignment G->V, R->R, B->B (see
# ALCOR_ZEROPOINT_BANDS). G->V is essentially color-flat; R and B carry sizeable
# B-V color terms set by the instrument bandpasses. The zeropoints were fit with
# ALCOR_AIRMASS_TERM held fixed, so the term and the zeropoints are a matched
# set. They are stable to ~0.03 mag across the two epochs (~21 months); add a new
# epoch like ALCOR_CALIBRATIONS and the nearest epoch in time is used (see
# alcor_zeropoint). Derived by docs/scripts/zeropoint_calib.py. `epoch` is the
# calibration night (UT, not local night -- do not "fix" it).
ALCOR_ZEROPOINTS = [
    {"epoch": "2024-09-05",
     "r": {"zp": 14.670, "color_coeff": -0.323},
     "g": {"zp": 15.438, "color_coeff": -0.023},
     "b": {"zp": 14.988, "color_coeff": 0.479}},
    {"epoch": "2026-05-19",
     "r": {"zp": 14.639, "color_coeff": -0.343},
     "g": {"zp": 15.423, "color_coeff": -0.038},
     "b": {"zp": 15.015, "color_coeff": 0.470}},
]

# instrument channel -> catalog Johnson band measured against
ALCOR_ZEROPOINT_BANDS = {"r": "R", "g": "V", "b": "B"}



def alcor_zeropoint(time=None):
    """
    Return the photometric-zeropoint dict whose epoch is nearest ``time``.

    Mirrors :func:`alcor_calibration`: ``time`` is an astropy ``Time`` (an exact
    tie resolves to the more recent epoch), and ``time=None`` returns the most
    recent epoch. The returned dict is a deep-enough copy that its per-band
    sub-dicts may be mutated freely without corrupting the table.
    """
    epochs = [(Time(z["epoch"], scale="utc"), z) for z in ALCOR_ZEROPOINTS]
    if time is None:
        chosen = max(epochs, key=lambda e: e[0].jd)[1]
    else:
        jds = np.array([e[0].jd for e in epochs])
        dt = np.abs(jds - Time(time).jd)
        # primary: smallest |dt|; tie-break: largest jd (more recent)
        order = np.lexsort((-jds, dt))
        chosen = epochs[order[0]][1]
    return {key: (dict(value) if isinstance(value, dict) else value)
            for key, value in chosen.items()}



ALCOR_PRESSURE = 760 * u.hPa        # ~0.75 atm at the MMT 2600 m elevation

ALCOR_TEMPERATURE = 10 * u.deg_C

ALCOR_HUMIDITY = 0.2

ALCOR_OBSWL = 0.55 * u.micron



# Summary columns derived from the floating darkest-cone search, appended after
# the fixed ALCOR_SB_TARGETS columns.
ALCOR_SB_BEST_COLUMNS = ("allsky_mv_best", "best_az", "best_alt")
