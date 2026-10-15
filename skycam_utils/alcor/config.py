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


# Consensus bad-pixel masks (create_badpix_consensus_masks). A single night's
# detections are a stable sensor core plus one-night sky structure: over 644
# archived nights (2025-01..2026-10) 99k pixel/channels were flagged at least
# once but only ~1000 on more than half the nights, and the transients sit at a
# median altitude of 5 deg toward the light domes -- sky gradient and ground
# lights just above the dilated horizon mask, different every night. A pixel is
# therefore masked for a calendar month only when it is flagged on at least
# ALCOR_BADPIX_CONSENSUS of that month's nights...
ALCOR_BADPIX_CONSENSUS = 0.5
# ...counting only nights sensitive enough to see it. The detector thresholds in
# noise units, and moonlight raises the G-channel noise of a night median from
# ~6 to ~20 ADU, so a moonlit night reaches only the brighter hot pixels: January
# 2026 held both of its full-Moon stretches and an all-night vote dropped 170
# real defects for that one month. Nights with G sigma above this cap do not
# vote; that leaves 13-19 voting nights a month, and the masks then grow
# smoothly (~27 defects a month net, sensor aging) with no lunar signature.
ALCOR_BADPIX_SIGMA_MAX = 9.0   # ADU, robust sigma of the G high-pass residual
# A defect is single-channel: one pixel in G (edge neighbours ~1/8 of the
# peak), a one-sided 2x2 block at ~1, 1/3, 1/3, 1/5 in R/B from the demosaic.
# The fainter footprint pixels read z ~10-30 on a dark night, so they clear
# z>25 only some nights and lose the monthly vote. Detections are grown into
# same-channel 3x3 neighbours above this z (one step of hysteresis); without it
# the monthly masks are 26-30% smaller and ~500 pixels at a median z ~15, every
# one adjacent to a masked defect, go unrepaired.
ALCOR_BADPIX_GROW_THRESH = 8.0
# A month with fewer voting nights than this gets no mask (the nearest-date
# loader then falls back to a neighbouring month).
ALCOR_BADPIX_MIN_NIGHTS = 8


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

# Stars within this many raw pixels of not-sky (load_alcor_horizon_mask) are
# dropped from the extinction maps, as measurements AND as lost stars.
# Photometry is fixed-position, so a star behind the flagpole or a building is
# still "measured", and light from the obstruction lands in its aperture: on a
# clear night (2026-10-14) stars on not-sky pixels read 0.15-0.2 mag too BRIGHT
# against stars at the same altitude clear of it, and within 8 px (the 5 px
# aperture+annulus plus centring slop) nearly as much. 8 px is the reach of that
# mechanism. A weaker ~0.05 mag excess persists further out, but there is no
# plateau to set a margin by -- a wider one mostly removes low-altitude stars and
# so trades against the residual altitude trend (EXTFLAT), which is where it
# belongs. Found because the flagpole sits ~4 deg below where the MMT often
# points; note that the bulk of that target's -0.2 was aperture BLENDING, not
# the flagpole.
ALCOR_EXT_HORIZON_MARGIN = 8.0

# --- per-frame registration (alcor_star_photometry) -------------------------
# The fitted geometry is stationary, but the image itself TRANSLATES on the
# sensor by 1-2 px over months: bright-star centroids minus WCS predictions are
# a pure shift, the same in every azimuth quadrant and stable to +-0.15 px over a
# night, running dy -1.0 (2025-01) -> 0.0 (2026-05-18, the fitted epoch) -> +1.3
# to +1.8 (2026-07..10), with dx ~ -0.1. A 4 px aperture still captures the star
# with that offset, but it changes which NEIGHBOURS fall inside, so the blend
# correction cannot be right without it. Each frame is therefore registered: the
# median centroid offset of its bright stars shifts every aperture.
ALCOR_REGISTER_VMAG = 4.0         # stars bright enough to centroid in any frame
ALCOR_REGISTER_MIN_ALTITUDE = 30.0
ALCOR_REGISTER_BOX = 5            # half-width (px) of the centroid box
ALCOR_REGISTER_MIN_SNR = 20.0     # peak / robust box noise to use a star
ALCOR_REGISTER_MIN_STARS = 10     # fewer (cloud): leave the frame unregistered
ALCOR_REGISTER_MAX_SHIFT = 4.0    # px; a larger median offset is not trusted

# --- aperture blending (alcor/blend.py) -----------------------------------
# The default photometry aperture radius (px) the blend terms assume, matching
# alcor_star_photometry's default. A different --aperture-radius changes which
# neighbours land inside, so pass it through to alcor_calibrate_photometry.
ALCOR_BLEND_APERTURE_RADIUS = 4.0
# Gaussian PSF sigma (px): FWHM 1.6 px, the median of the --both Gaussian fits
# on 2026-10-03 (1.4 near the zenith, 1.8 at alt 20-30). The PSF is so much
# smaller than the aperture that this only softens the aperture edge.
ALCOR_BLEND_PSF_SIGMA = 0.68
# Fraction of a neighbour's PREDICTED flux that the aperture actually records.
# Fit jointly with an offset and an airmass slope over every window star of two
# registered nights: 0.87 (2026-10-14, ~20k rows) and 0.91 (2026-05-18, ~81k).
# Flat in neighbour brightness and in distance from the aperture centre, so it is
# a single factor -- most plausibly Hipparcos V against the camera's G passband
# plus the pixelated aperture edge -- not the faint-end non-linearity.
ALCOR_BLEND_EFFICIENCY = 0.9

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

# Clear-night altitude flat, per band (mag), SUBTRACTED from the calibrated mag:
#   cal = instr - ALCOR_AIRMASS_TERM*airmass - flat_b(alt) + zp + color*(B-V)
# It is what ext_* read on clear nights with only the airmass term applied: the
# same shape on every clear night from 2025-01-01 to 2026-10-14 (seven nights,
# per-night bins agree to ~0.01), azimuthally symmetric to ~0.01, and the same
# for bright and faint stars at a given altitude -- so it is a fixed property of
# the instrument, not the atmosphere or a magnitude effect. In g it rises from
# -0.12 at alt 20 to +0.06 at alt 45 and falls back to -0.11 at the zenith: the
# single airmass term over-corrects low altitudes (as if k ~0.28 rather than
# 0.40) and on-axis stars read bright (vignetting). The two are degenerate for
# a fixed all-sky camera, so this is an empirical correction, not a split into
# atmosphere and optics. Nodes are a quartic in zenith distance fit to 2.5 deg
# bins of per-(night, star)-offset residuals of isolated window stars (six
# registered nights, ~100k rows a band; claude_docs/scripts/altitude_flat.py).
# The LEVEL of a flat is degenerate with the zeropoint; it is left as measured,
# since applying it moves the zeropoint fits by <= 0.01 mag with mixed sign --
# within how well a refit reproduces at all -- so ALCOR_ZEROPOINTS stand. It cut
# the zeropoint fit's per-star rms from 0.097 to 0.076 in G (2026-10-14).
# np.interp holds the end values below alt 20, where photometry stops.
ALCOR_ALTITUDE_FLAT_EPOCH = "2026-10-14"
ALCOR_ALTITUDE_FLAT = {
    "alt": (20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0,
            75.0, 80.0, 85.0, 90.0),
    "r": (-0.134, -0.064, -0.014, 0.020, 0.040, 0.050, 0.052, 0.048, 0.041,
          0.032, 0.022, 0.013, 0.006, 0.000, -0.002),
    "g": (-0.124, -0.052, -0.002, 0.031, 0.049, 0.056, 0.054, 0.044, 0.030,
          0.012, -0.010, -0.034, -0.061, -0.090, -0.124),
    "b": (-0.074, -0.037, -0.006, 0.018, 0.032, 0.038, 0.035, 0.024, 0.008,
          -0.012, -0.032, -0.048, -0.056, -0.051, -0.026),
}

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
# set. Add a new epoch like ALCOR_CALIBRATIONS and the nearest epoch in time is
# used (see alcor_zeropoint). Derived by claude_docs/scripts/zeropoint_calib.py.
# `epoch` is the calibration night (UT, not local night -- do not "fix" it).
#
# They are DE-BLENDED zeropoints (zeropoint_calib.py --blend, 2026-10-14): the
# instrumental mag is corrected by the per-row aperture-blending term before the
# fit, and alcor_calibrate_photometry adds the same term back to the catalog
# side, so the two must change together. The original fits absorbed the average
# neighbour light, which made them ~0.1 mag too high in G; de-blending also cut
# the per-star rms of the fit from 0.134 to 0.078 mag in G on 2026-05-18 (0.139
# to 0.097 on 2026-10-14), with R and B improving similarly.
#   2026-05-19  refit on registered-equivalent photometry (that night's image
#               offset was ~0 px, so its archived CSVs were already centred).
#   2026-10-15  new, from registered photometry of the clear 2026-10-14 night.
#               ~0.1 mag above May in every band: the dome-throughput drift, not
#               a calibration change. Nearest-epoch resolution makes this a step
#               at ~2026-07-31 where the drift is really a ramp; a time-dependent
#               zeropoint is the proper fix.
#   2024-09-05  its night (2024-09-04) is not on any mounted volume, so it was
#               NOT refit: it is shifted by the 2026-05-19 blend-vs-no-blend
#               difference (zp r -0.111 g -0.098 b -0.061; color r +0.049
#               g +0.022 b -0.045), the absorbed neighbour light being a property
#               of the catalog and the star sample rather than of the night.
#               Refit it if the frames turn up.
ALCOR_ZEROPOINTS = [
    {"epoch": "2024-09-05",
     "r": {"zp": 14.559, "color_coeff": -0.274},
     "g": {"zp": 15.340, "color_coeff": -0.001},
     "b": {"zp": 14.927, "color_coeff": 0.434}},
    {"epoch": "2026-05-19",
     "r": {"zp": 14.521, "color_coeff": -0.287},
     "g": {"zp": 15.331, "color_coeff": -0.022},
     "b": {"zp": 14.961, "color_coeff": 0.415}},
    {"epoch": "2026-10-15",
     "r": {"zp": 14.618, "color_coeff": -0.285},
     "g": {"zp": 15.429, "color_coeff": -0.013},
     "b": {"zp": 15.058, "color_coeff": 0.422}},
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
