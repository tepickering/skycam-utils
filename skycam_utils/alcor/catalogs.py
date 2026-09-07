"""Bright-star and bright-variable catalogs in alt/az at a given time."""


import numpy as np
from astropy.table import Table, vstack

from astropy.coordinates import SkyCoord, AltAz

from ..astrometry import MMT_LOCATION

from .config import (
    _DATA_ROOT,
    ALCOR_HUMIDITY, ALCOR_OBSWL, ALCOR_PRESSURE, ALCOR_TEMPERATURE
)


def alcor_reference_altaz(time, vmag_limit=3.0, min_alt=5.0, refraction=True,
                           location=MMT_LOCATION):
    """
    Load ``bright_star_sloan.fits``, filter to ``Vmag <= vmag_limit``, and compute
    Alt/Az at ``time`` and ``location``. Stars below ``min_alt`` are dropped.

    When ``refraction`` is True the AltAz frame includes atmospheric refraction
    using nominal MMT pressure/temperature; this matters most at large zenith
    angle, where the radial distortion is also largest.

    Parameters
    ----------
    time : `~astropy.time.Time`
        Observation time (scalar).
    vmag_limit : float (default=3.0)
        Faintest V magnitude to keep.
    min_alt : float (default=5.0)
        Minimum altitude (deg) to keep.
    refraction : bool (default=True)
        If True, include atmospheric refraction in the AltAz transform.
    location : `~astropy.coordinates.EarthLocation` (default=MMT_LOCATION)
        Observatory location.

    Returns
    -------
    cat : `~astropy.table.Table`
        Catalog rows with added ``Alt`` and ``Az`` columns (degrees), filtered to
        ``Vmag <= vmag_limit`` and ``Alt >= min_alt``.
    """
    catpath = _DATA_ROOT / "bright_star_sloan.fits"
    cat = Table.read(str(catpath))
    cat = cat[cat["Vmag"] <= vmag_limit]

    coords = SkyCoord(cat["_RAJ2000"], cat["_DEJ2000"], unit="deg", frame="icrs")
    if refraction:
        frame = AltAz(obstime=time, location=location, pressure=ALCOR_PRESSURE,
                      temperature=ALCOR_TEMPERATURE, relative_humidity=ALCOR_HUMIDITY,
                      obswl=ALCOR_OBSWL)
    else:
        frame = AltAz(obstime=time, location=location)
    altaz = coords.transform_to(frame)
    cat["Alt"] = altaz.alt.deg
    cat["Az"] = altaz.az.deg
    cat = cat[cat["Alt"] >= min_alt]
    return cat



def alcor_named_reference_altaz(time, vmag_limit=5.5, min_alt=20.0,
                                refraction=True, location=MMT_LOCATION):
    """
    Load ``bright_star_sloan_named.fits`` and compute Alt/Az at ``time``.

    This is the star-photometry catalog path: unlike
    :func:`alcor_reference_altaz`, it keeps the ``NAME`` column used as the CSV
    row index. Stars are filtered to ``Vmag <= vmag_limit`` and ``Alt >=
    min_alt``.
    """
    catpath = _DATA_ROOT / "bright_star_sloan_named.fits"
    cat = Table.read(str(catpath))
    cat = cat[cat["Vmag"] <= vmag_limit]

    coords = SkyCoord(cat["_RAJ2000"], cat["_DEJ2000"], unit="deg", frame="icrs")
    if refraction:
        frame = AltAz(obstime=time, location=location, pressure=ALCOR_PRESSURE,
                      temperature=ALCOR_TEMPERATURE, relative_humidity=ALCOR_HUMIDITY,
                      obswl=ALCOR_OBSWL)
    else:
        frame = AltAz(obstime=time, location=location)
    altaz = coords.transform_to(frame)
    cat["Alt"] = altaz.alt.deg
    cat["Az"] = altaz.az.deg
    cat = cat[cat["Alt"] >= min_alt]
    return cat



def _catalog_value_to_python(value):
    """
    Convert an Astropy table scalar to a JSON-friendly Python scalar.
    """
    if np.ma.is_masked(value):
        return None
    if isinstance(value, bytes):
        return value.decode().strip()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.generic):
        return value.item()
    return value



def lookup_sloan_photometry(star_name, case_sensitive=False):
    """
    Return the ``bright_star_sloan_named.fits`` row for ``star_name`` as a dict.

    Matching is against the catalog ``NAME`` column, with surrounding whitespace
    ignored. By default matching is case-insensitive. A missing name raises
    ``KeyError``; an ambiguous name raises ``ValueError`` with the matching HD
    numbers.
    """
    query = str(star_name).strip()
    if not query:
        raise ValueError("star_name must not be empty")

    catpath = _DATA_ROOT / "bright_star_sloan_named.fits"
    cat = Table.read(str(catpath))
    names = np.array([str(name).strip() for name in cat["NAME"]])
    if case_sensitive:
        match = names == query
    else:
        match = np.char.lower(names) == query.lower()

    matches = cat[match]
    if len(matches) == 0:
        raise KeyError(f"star {star_name!r} not found in bright_star_sloan_named.fits")
    if len(matches) > 1:
        hds = ", ".join(str(_catalog_value_to_python(row["HD"])) for row in matches)
        raise ValueError(f"star name {star_name!r} is ambiguous; matching HD numbers: {hds}")

    row = matches[0]
    return {col: _catalog_value_to_python(row[col]) for col in matches.colnames}



def alcor_variable_reference_altaz(time, vmag_limit=5.5, min_alt=20.0,
                                   refraction=True, location=MMT_LOCATION):
    """
    Load ``bright_variable_vsx.fits`` and compute Alt/Az at ``time``.

    The variable-star sibling of :func:`alcor_named_reference_altaz`. The
    calibration catalog excludes variables by construction -- correctly, since
    they would corrupt the zeropoints -- which left the camera never measuring
    any of them; Polaris was not merely unmeasured but was being flagged as a
    cluster of hot pixels. This catalog is the other half: AAVSO VSX entries
    brighter than V=6.0 at maximum with amplitude >= 0.05 mag, excluding
    eruptive/cataclysmic types (see ``claude_docs/scripts/build_variable_catalog.py``).

    ``vmag_limit`` filters on ``Vmax``, the brightness at MAXIMUM light -- what
    decides whether the star is ever measurable. A large-amplitude Mira spends
    most of its cycle far below the limit, and those non-detections are data.
    """
    catpath = _DATA_ROOT / "bright_variable_vsx.fits"
    cat = Table.read(str(catpath))
    cat = cat[cat["Vmag"] <= vmag_limit]

    coords = SkyCoord(cat["_RAJ2000"], cat["_DEJ2000"], unit="deg", frame="icrs")
    if refraction:
        frame = AltAz(obstime=time, location=location, pressure=ALCOR_PRESSURE,
                      temperature=ALCOR_TEMPERATURE, relative_humidity=ALCOR_HUMIDITY,
                      obswl=ALCOR_OBSWL)
    else:
        frame = AltAz(obstime=time, location=location)
    altaz = coords.transform_to(frame)
    cat["Alt"] = altaz.alt.deg
    cat["Az"] = altaz.az.deg
    cat = cat[cat["Alt"] >= min_alt]
    return cat



def alcor_photometry_reference_altaz(time, vmag_limit=5.5, min_alt=20.0,
                                     refraction=True, variables=True,
                                     location=MMT_LOCATION,
                                     match_radius=20.0):
    """
    The combined star list a photometry pass measures, with a ``Variable`` flag.

    Concatenates :func:`alcor_named_reference_altaz` with
    :func:`alcor_variable_reference_altaz`, dropping variables that are already
    in the calibration catalog (matched within ``match_radius`` arcsec) so no
    star is measured twice. A star present in both keeps its calibration-catalog
    row -- it has real catalog magnitudes, so its ``ext_*`` stays meaningful --
    and is merely flagged. Returns a table with at least ``NAME``, ``HD``,
    ``Alt``, ``Az`` and ``Variable``.
    """
    named = alcor_named_reference_altaz(
        time, vmag_limit=vmag_limit, min_alt=min_alt, refraction=refraction,
        location=location)
    out = named[[c for c in ("NAME", "HD", "Alt", "Az")
                 if c in named.colnames]].copy()
    out["Variable"] = np.zeros(len(out), dtype=bool)
    if not variables:
        return out

    var = alcor_variable_reference_altaz(
        time, vmag_limit=vmag_limit, min_alt=min_alt, refraction=refraction,
        location=location)
    can_match = all(c in named.colnames for c in ("_RAJ2000", "_DEJ2000"))
    if len(var) and len(named) and can_match:
        vc = SkyCoord(var["_RAJ2000"], var["_DEJ2000"], unit="deg")
        nc = SkyCoord(named["_RAJ2000"], named["_DEJ2000"], unit="deg")
        _, sep, _ = vc.match_to_catalog_sky(nc)
        already = sep.arcsec < match_radius
        # flag the calibration rows that are known variables
        _, sep_n, _ = nc.match_to_catalog_sky(vc)
        out["Variable"] = sep_n.arcsec < match_radius
        var = var[~already]
    if len(var) == 0:
        return out

    add = Table()
    add["NAME"] = [str(n).strip() for n in var["NAME"]]
    add["HD"] = np.array(var["HD"], dtype=int)
    add["Alt"] = np.array(var["Alt"], dtype=float)
    add["Az"] = np.array(var["Az"], dtype=float)
    add["Variable"] = np.ones(len(add), dtype=bool)
    return vstack([out, add], metadata_conflicts="silent")



def _alcor_star_labels(cat):
    """
    Return stable, unique row labels for the named bright-star catalog.
    """
    raw_names = [str(name).strip() for name in cat["NAME"]]
    hd = []
    for value in cat["HD"]:
        try:
            hd.append(None if np.ma.is_masked(value) else int(value))
        except (TypeError, ValueError):
            hd.append(None)
    base = []
    for index, (name, hd_value) in enumerate(zip(raw_names, hd), start=1):
        if name and name != "--":
            base.append(name)
        elif hd_value is not None:
            base.append(f"HD {hd_value}")
        else:
            base.append(f"unnamed {index}")
    totals = {}
    for label in base:
        totals[label] = totals.get(label, 0) + 1
    counts = {}
    labels = []
    for label, hd_value in zip(base, hd):
        counts[label] = counts.get(label, 0) + 1
        if counts[label] == 1 and totals[label] == 1:
            labels.append(label)
        elif hd_value is not None:
            labels.append(f"{label} (HD {hd_value})")
        else:
            labels.append(f"{label} {counts[label]}")
    return labels
