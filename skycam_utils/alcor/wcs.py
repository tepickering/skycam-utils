"""Raw-frame ARC alt/az WCS construction and the lens-distortion model."""

from functools import lru_cache

import numpy as np
from astropy.wcs import WCS, Sip

import astropy.units as u

from .config import (
    ALCOR_AXIS_TILT, ALCOR_HORIZON_RADIUS, ALCOR_RADIAL_COEFFS,
    ALCOR_ROTATION, ALCOR_TANGENTIAL_COEFFS, ALCOR_XCEN, ALCOR_YCEN
)


def _invert_radial(z_deg, radial_coeffs, n_iter=8):
    """
    Invert ``z = 90*(k1*rho + k3*rho**3 + k5*rho**5)`` for the normalized
    detector radius ``rho`` using Newton's method. ``z_deg`` is the zenith angle
    in degrees. Assumes the polynomial is monotonic over the field of view
    (true for physical near-equidistant coefficients).
    """
    k1, k3, k5 = radial_coeffs
    if k1 <= 0:
        raise ValueError("radial coefficient k1 must be positive")
    t = np.asarray(z_deg, dtype=float) / 90.0
    rho = t / k1  # equidistant first guess
    for _ in range(n_iter):
        g = k1 * rho + k3 * rho**3 + k5 * rho**5
        gp = k1 + 3.0 * k3 * rho**2 + 5.0 * k5 * rho**4
        rho = rho - (g - t) / gp
    return rho



def _axis_frame(alt, az, t_n, t_e):
    """
    Axis-centered polar coordinates (z', A') [deg] of sky points (alt, az)
    [deg] for an optical axis tilted off the zenith.

    The axis leans eps = hypot(t_n, t_e) deg from the zenith toward azimuth
    A0 = atan2(t_e, t_n) (components toward north and east). The sky is
    rotated by the exact minimal rotation -- by eps about the horizontal axis
    at azimuth A0 + 90 -- that carries the optical axis to the pole, so
    A' -> az continuously as eps -> 0. z' is the true angular distance from
    the axis.
    """
    eps = np.radians(np.hypot(t_n, t_e))
    a0 = np.arctan2(t_e, t_n)
    alt_r = np.radians(np.asarray(alt, dtype=float))
    az_r = np.radians(np.asarray(az, dtype=float))
    # unit vectors: x toward north (az=0), y toward east (az=90), z up
    vx = np.cos(alt_r) * np.cos(az_r)
    vy = np.cos(alt_r) * np.sin(az_r)
    vz = np.sin(alt_r)
    # Rodrigues rotation by -eps about n = (-sin A0, cos A0, 0):
    # v' = v cos(eps) - (n x v) sin(eps) + n (n.v)(1 - cos(eps))
    nx, ny = -np.sin(a0), np.cos(a0)
    ndv = nx * vx + ny * vy
    c, s = np.cos(eps), np.sin(eps)
    wx = c * vx - s * (ny * vz) + (1.0 - c) * ndv * nx
    wy = c * vy - s * (-nx * vz) + (1.0 - c) * ndv * ny
    wz = c * vz - s * (nx * vy - ny * vx)
    zp = np.degrees(np.arccos(np.clip(wz, -1.0, 1.0)))
    ap = np.degrees(np.arctan2(wy, wx))
    return zp, ap



def _tangential_delta(u, v, p1, p2, horizon_radius):
    """
    Brown-Conrady tangential (decentering) displacement in raw pixels.

    ``u``, ``v`` are pixel offsets from the optical-axis pixel
    ``(xcen, ycen)``; ``p1``/``p2`` are
    dimensionless (normalized by ``horizon_radius``), like the radial k
    coefficients. This is the pix->world displacement the WCS SIP applies
    (see build_alcor_wcs); it is an exact degree-2 polynomial.
    """
    H = float(horizon_radius)
    du = (p1 / H) * (3.0 * u**2 + v**2) + (2.0 * p2 / H) * u * v
    dv = (p2 / H) * (u**2 + 3.0 * v**2) + (2.0 * p1 / H) * u * v
    return du, dv



def _predict_pixels(
    alt,
    az,
    xcen=ALCOR_XCEN,
    ycen=ALCOR_YCEN,
    rotation=0.0,
    radial_coeffs=ALCOR_RADIAL_COEFFS,
    horizon_radius=ALCOR_HORIZON_RADIUS,
    tangential_coeffs=(0.0, 0.0),
    axis_tilt=(0.0, 0.0),
):
    """
    Forward lens model: map altitude/azimuth (deg) to RAW-frame pixel
    coordinates (x=column, y=row, 0-based).

    The optical axis sits at ``(xcen, ycen)`` (the zenith pixel when
    ``axis_tilt`` is zero); ``rotation`` is the camera azimuth
    zero-point offset (deg). The lens plate solution
    ``z = 90*(k1*rho + k3*rho**3 + k5*rho**5)`` (``rho = r/horizon_radius``,
    ``z = 90 - alt``) is inverted for ``rho`` via Newton's method. The sky's
    azimuth runs opposite to the sensor's polar angle (an all-sky camera images
    the sky as seen from below), so the pixel angle is ``rotation - az``; north
    (az=0) lands toward +y. The matching WCS encodes the same mapping in its PC
    rotation matrix (see ``build_alcor_wcs``).

    ``tangential_coeffs`` (P1, P2) adds Brown-Conrady decentering, defined like
    the k's on the pix->world side (`_tangential_delta`). It is inverted with a
    fixed-point loop that re-solves the radial part exactly each pass, so the
    contraction is governed by the (tiny, ~4*P) tangential derivative rather
    than the O(k3, k5) radial one; three passes reach well below 1e-3 px for
    |P| up to ~1e-2.

    ``axis_tilt`` (t_n, t_e) tilts the optical axis off the zenith (degrees
    toward north/east; see `_axis_frame`). The model is azimuthally symmetric
    about the AXIS: the radial inversion runs in the axis distance z' and the
    pixel angle is ``rotation - A'``. With nonzero tilt, (xcen, ycen) is the
    optical-axis pixel, not the zenith pixel.
    """
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)
    coeffs = tuple(float(c) for c in radial_coeffs)

    tn, te = (float(c) for c in axis_tilt)
    if tn != 0.0 or te != 0.0:
        zp, ap = _axis_frame(alt, az, tn, te)
    else:
        zp = 90.0 - alt
        ap = az

    rho = _invert_radial(zp, coeffs)
    r = horizon_radius * rho
    ang = np.radians(rotation - ap)
    u = r * np.sin(ang)
    v = r * np.cos(ang)

    p1, p2 = (float(c) for c in tangential_coeffs)
    if p1 != 0.0 or p2 != 0.0:
        k1 = coeffs[0]
        H = float(horizon_radius)
        # Linear-pixel target of the SIP equation t = (u,v) + D_rad + D_tan:
        # the radial displacement preserves direction, so |t| = H*z'/(90*k1).
        s = H * zp / (90.0 * k1)
        tu = s * np.sin(ang)
        tv = s * np.cos(ang)
        for _ in range(3):
            du, dv = _tangential_delta(u, v, p1, p2, H)
            wu = tu - du
            wv = tv - dv
            wr = np.hypot(wu, wv)
            safe = np.where(wr > 0.0, wr, 1.0)
            rho_w = _invert_radial(90.0 * k1 * wr / H, coeffs)
            scale = np.where(wr > 0.0, H * rho_w / safe, 0.0)
            u = wu * scale
            v = wv * scale

    x = xcen + u
    y = ycen + v
    return x, y



def _base_arc_wcs(xcen, ycen, rotation, k1, horizon_radius,
                  axis_tilt=(0.0, 0.0)):
    """
    Linear ARC WCS (no SIP) reproducing the raw forward model's linear part.

    crpix is the 1-based optical-axis pixel; the PC matrix is the pure rotation
    (det=+1) that matches ``_predict_pixels`` (the sky/sensor handedness lives in
    the ``rotation - az`` azimuth convention, encoded by the ARC longitude axis).
    A nonzero ``axis_tilt`` moves the projection pole to the tilted optical
    axis: CRVAL = (A0, 90 - eps) and LONPOLE = A0, which makes the WCS native
    frame coincide with the minimal-rotation frame of ``_axis_frame`` (native
    longitude phi = A' + 180; the celestial pole sits at A' = A0 + 180).
    """
    cdelt = 90.0 * k1 / horizon_radius
    rot = np.radians(rotation)
    c, s = np.cos(rot), np.sin(rot)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
    wcs.wcs.crpix = [xcen + 1.0, ycen + 1.0]
    wcs.wcs.cdelt = [cdelt, cdelt]
    wcs.wcs.pc = [[c, -s], [s, c]]
    tn, te = axis_tilt
    if tn != 0.0 or te != 0.0:
        eps = float(np.hypot(tn, te))
        a0 = float(np.degrees(np.arctan2(te, tn))) % 360.0
        wcs.wcs.crval = [a0, 90.0 - eps]
        wcs.wcs.lonpole = a0
    else:
        wcs.wcs.crval = [0.0, 90.0]
        wcs.wcs.lonpole = 0.0
    return wcs



def _sip_poly_eval(coef, u, v):
    """
    Evaluate a SIP coefficient matrix (coef[p, q] * u**p * v**q) at (u, v).
    """
    out = np.zeros_like(u, dtype=float)
    n = coef.shape[0]
    for p in range(n):
        for q in range(n):
            c = coef[p, q]
            if c != 0.0:
                out = out + c * u**p * v**q
    return out



def _fit_sip_inverse(a, b, radius, sip_degree):
    """
    Fit the approximate inverse SIP coefficients (AP, BP) for forward
    coefficients (A, B) over a pixel grid. The inverse of a radial polynomial is
    not itself polynomial, so AP/BP are a least-squares approximation (used by
    external tools and as the initial guess for astropy's iterative
    world->pixel solve, which refines to machine precision using A/B).
    """
    g = np.linspace(-radius, radius, 50)
    uu, vv = np.meshgrid(g, g)
    u = uu.ravel()
    v = vv.ravel()
    fu = u + _sip_poly_eval(a, u, v)
    fv = v + _sip_poly_eval(b, u, v)
    terms = [(p, q) for p in range(sip_degree + 1) for q in range(sip_degree + 1)
             if 1 <= p + q <= sip_degree]
    design = np.column_stack([fu**p * fv**q for (p, q) in terms])
    coef_u, _, _, _ = np.linalg.lstsq(design, u - fu, rcond=None)
    coef_v, _, _, _ = np.linalg.lstsq(design, v - fv, rcond=None)
    ap = np.zeros((sip_degree + 1, sip_degree + 1))
    bp = np.zeros((sip_degree + 1, sip_degree + 1))
    for (p, q), cu, cv in zip(terms, coef_u, coef_v):
        ap[p, q] = cu
        bp[p, q] = cv
    return ap, bp



def build_alcor_wcs(xcen=ALCOR_XCEN, ycen=ALCOR_YCEN, rotation=ALCOR_ROTATION,
                    radial_coeffs=ALCOR_RADIAL_COEFFS,
                    horizon_radius=ALCOR_HORIZON_RADIUS, sip_degree=5,
                    tangential_coeffs=ALCOR_TANGENTIAL_COEFFS,
                    axis_tilt=ALCOR_AXIS_TILT):
    """
    Build the raw-frame alt/az ARC WCS for the alcor sensor.

    The optical axis is at pixel ``(xcen, ycen)`` (the zenith pixel when
    ``axis_tilt`` is zero); ``rotation`` is the PC rotation matrix (the
    sky/sensor handedness is in the ``rotation - az`` convention, see
    ``_predict_pixels``); the radial ``k3``/``k5`` distortion is an exact
    analytic SIP centered on ``(xcen, ycen)``. Cached on its hashable args;
    returns a fresh copy.

    The lens is parametrized as a plate solution that maps the detector directly
    to the sky: ``z = 90*(k1*rho + k3*rho**3 + k5*rho**5)`` with ``rho =
    r / horizon_radius`` the normalized detector radius and ``z = 90 - alt`` the
    zenith angle (an odd-power, symmetric-fisheye polynomial in detector radius;
    the ``k5`` term needs degree 5, hence ``sip_degree=5``). The Cartesian
    displacement of this radial map is an exact degree-5 polynomial in the
    detector pixel offsets, so the SIP coefficients are constructed analytically
    (not fitted) and reproduce the plate solution to numerical precision over the
    whole FOV. The radial polynomial is rotation/reflection invariant, so the
    same A/B coefficients hold in the raw frame -- only the SIP reference pixel
    moves to the optical-axis pixel.

    ``tangential_coeffs`` (P1, P2) adds Brown-Conrady decentering; its Cartesian
    displacement is an exact degree-2 polynomial (see ``_tangential_delta``), so
    it joins the analytic SIP without approximation.

    ``axis_tilt`` (t_n, t_e) tilts the optical axis off the zenith. This is
    pure FITS-WCS geometry -- CRVAL moves to (A0, 90 - eps) with LONPOLE = A0
    -- so the SIP is untouched and the mapping stays exact; with nonzero tilt
    (xcen, ycen) is the optical-axis pixel, and the zenith pixel must be
    obtained via ``world_to_pixel`` of alt=90 rather than CRPIX.
    """
    return _build_alcor_wcs_cached(
        float(xcen), float(ycen), float(rotation),
        tuple(float(c) for c in radial_coeffs),
        float(horizon_radius), int(sip_degree),
        tuple(float(c) for c in tangential_coeffs),
        tuple(float(c) for c in axis_tilt),
    ).deepcopy()



@lru_cache(maxsize=32)
def _build_alcor_wcs_cached(xcen, ycen, rotation, radial_coeffs, horizon_radius,
                            sip_degree, tangential_coeffs=(0.0, 0.0),
                            axis_tilt=(0.0, 0.0)):
    k1, k3, k5 = radial_coeffs
    p1, p2 = tangential_coeffs
    base = _base_arc_wcs(xcen, ycen, rotation, k1, horizon_radius,
                         axis_tilt=axis_tilt)
    if (abs(k3) < 1e-12 and abs(k5) < 1e-12
            and abs(p1) < 1e-12 and abs(p2) < 1e-12):
        return base

    # Analytic SIP for the radial plate solution. The Cartesian displacement of
    # z = 90*(k1*rho + k3*rho**3 + k5*rho**5) is A_u = u*(k3*rho**2 + k5*rho**4)/k1
    # with rho = sqrt(u**2 + v**2)/horizon_radius -- an exact degree-5 polynomial.
    H = float(horizon_radius)
    c3 = k3 / (k1 * H**2)
    c5 = k5 / (k1 * H**4)
    a = np.zeros((sip_degree + 1, sip_degree + 1))
    b = np.zeros((sip_degree + 1, sip_degree + 1))
    a[3, 0] = c3; a[1, 2] = c3
    a[5, 0] = c5; a[3, 2] = 2 * c5; a[1, 4] = c5
    b[0, 3] = c3; b[2, 1] = c3
    b[0, 5] = c5; b[2, 3] = 2 * c5; b[4, 1] = c5
    # Brown-Conrady tangential (decentering) terms: an exact degree-2 polynomial
    # in the pixel offsets (see _tangential_delta). The radial terms occupy only
    # odd-total-degree slots, the tangential only even ones -- no collisions.
    a[2, 0] = 3.0 * p1 / H; a[0, 2] = p1 / H; a[1, 1] = 2.0 * p2 / H
    b[0, 2] = 3.0 * p2 / H; b[2, 0] = p2 / H; b[1, 1] = 2.0 * p1 / H
    ap, bp = _fit_sip_inverse(a, b, int(round(H)), sip_degree)

    wcs = base.deepcopy()
    wcs.wcs.ctype = ["RA---ARC-SIP", "DEC--ARC-SIP"]
    wcs.sip = Sip(a, b, ap, bp, [xcen + 1.0, ycen + 1.0])
    return wcs



def _alcor_pixel_solid_angle(az_deg, alt_deg):
    """
    Per-pixel solid angle [arcsec^2] from (azimuth, altitude) grids in degrees.

    The pixel->sky map is differentiated numerically: each pixel's sky direction
    becomes a unit vector and the solid angle is the magnitude of the cross
    product of the per-pixel tangent vectors d(vec)/dx and d(vec)/dy (1-pixel
    spacing). Working in unit-vector space rather than on the angles makes this
    exact for the ARC projection's zenith->horizon plate-scale change and the SIP
    distortion, and immune to the azimuth wrap at north. NaN propagates from any
    pixel the WCS does not project (off the sky).
    """
    lam = np.radians(np.asarray(az_deg, dtype=float))
    phi = np.radians(np.asarray(alt_deg, dtype=float))
    cphi = np.cos(phi)
    vx = cphi * np.cos(lam)
    vy = cphi * np.sin(lam)
    vz = np.sin(phi)
    dvx_dy, dvx_dx = np.gradient(vx)
    dvy_dy, dvy_dx = np.gradient(vy)
    dvz_dy, dvz_dx = np.gradient(vz)
    # |(dvec/dx) x (dvec/dy)| = area subtended per unit pixel area = solid angle
    cx = dvy_dx * dvz_dy - dvz_dx * dvy_dy
    cy = dvz_dx * dvx_dy - dvx_dx * dvz_dy
    cz = dvx_dx * dvy_dy - dvy_dx * dvx_dy
    omega_sr = np.sqrt(cx * cx + cy * cy + cz * cz)
    arcsec_per_rad = 180.0 * 3600.0 / np.pi
    return omega_sr * arcsec_per_rad * arcsec_per_rad
