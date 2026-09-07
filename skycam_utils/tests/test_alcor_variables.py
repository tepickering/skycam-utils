# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from astropy.time import Time

os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "skycam-utils-matplotlib"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from skycam_utils.alcor import (  # noqa: E402
    ALCOR_COLOR_FLAT_TOL,
    alcor_calibrate_photometry,
    alcor_photometry_reference_altaz,
    alcor_variable_reference_altaz,
    _catalog_calibration_map,
)

TIME = Time("2026-05-18T08:00:00")


def _variable_catalog():
    from importlib.resources import files
    return Table.read(str(files("skycam_utils") / "data" / "bright_variable_vsx.fits"))


def test_variable_catalog_shape_and_content():
    cat = _variable_catalog()
    for col in ("NAME", "_RAJ2000", "_DEJ2000", "Type", "Vmax", "Vmin",
                "Amplitude", "Period", "Vmag", "B-V", "HD"):
        assert col in cat.colnames

    amp = np.array(cat["Amplitude"], dtype=float)
    assert (amp >= 0.05 - 1e-9).all()          # the selection threshold
    assert (np.array(cat["Vmax"], dtype=float) <= 6.0 + 1e-9).all()
    # Vmag is maximum light, which is what decides measurability
    np.testing.assert_allclose(np.array(cat["Vmag"], dtype=float),
                               np.array(cat["Vmax"], dtype=float))

    names = [str(n).strip() for n in cat["NAME"]]
    # Polaris is the star that motivated the catalog: a 0.07 mag Cepheid that a
    # 0.2 mag amplitude cut would have excluded.
    assert "alf UMi" in names
    for expected in ("alf Ori", "bet Per", "omi Cet", "del Cep"):
        assert expected in names


def test_variable_catalog_excludes_eruptive():
    """A nova's tabulated max is a historical outburst, not a recurring state."""
    cat = _variable_catalog()
    types = [str(t).strip() for t in cat["Type"]]
    for t in types:
        assert not t.startswith(("NA", "NB", "NC", "NR", "SN", "UG", "V838MON"))
    names = [str(n).strip() for n in cat["NAME"]]
    # the four historical supernovae and Nova Vul 1670, all now V 16-23
    for gone in ("B Cas", "V0843 Oph", "SN 1987A", "S And", "CK Vul"):
        assert gone not in names
    # ...but eta Car stays: genuinely variable at V~4.5 today
    assert "eta Car" in names


def test_high_amplitude_variables_are_out_of_the_calibration_catalog():
    from importlib.resources import files
    cal = Table.read(str(files("skycam_utils") / "data"
                         / "bright_star_sloan_named.fits"))
    names = {str(n).strip() for n in cal["NAME"]}
    # the two eclipsing binaries that had leaked in
    assert "psi Cen" not in names
    assert "N Sco" not in names
    # ...but low-amplitude variables that are still sound standards stay
    assert "Arcturus" in names
    assert "Capella" in names


def test_reference_altaz_flags_and_deduplicates():
    combined = alcor_photometry_reference_altaz(TIME, vmag_limit=5.5,
                                                min_alt=20.0)
    assert "Variable" in combined.colnames
    assert combined["Variable"].sum() > 0

    named_only = alcor_photometry_reference_altaz(TIME, vmag_limit=5.5,
                                                  min_alt=20.0,
                                                  variables=False)
    assert not named_only["Variable"].any()
    assert len(combined) > len(named_only)

    # no star appears twice: the dedup is by position, so check labels
    labels = [str(n).strip() for n in combined["NAME"] if str(n).strip() != "--"]
    assert len(labels) == len(set(labels))


def test_variable_reference_filters_on_maximum_light():
    bright = alcor_variable_reference_altaz(TIME, vmag_limit=3.0, min_alt=0.0)
    faint = alcor_variable_reference_altaz(TIME, vmag_limit=6.0, min_alt=0.0)
    assert 0 < len(bright) < len(faint)
    assert (np.array(bright["Vmax"], dtype=float) <= 3.0).all()


def test_calibration_map_gives_variables_colour_but_no_catalog_mag():
    cmap = _catalog_calibration_map()
    polaris = cmap["alf UMi"]
    assert np.isfinite(polaris["BV"])           # colour from the BSC match
    for band in ("V", "R", "B"):
        assert np.isnan(polaris[band])          # no single catalog magnitude

    # a calibration star keeps its real magnitudes
    assert np.isfinite(cmap["Vega"]["V"])


def test_variables_get_a_light_curve_but_no_extinction():
    df = pd.DataFrame({"name": ["Vega", "alf UMi", "alf Ori"],
                       "altitude": [70.0] * 3,
                       "mag_g": [-10.0] * 3,
                       "mag_r": [-10.0] * 3})
    out = alcor_calibrate_photometry(df, time=TIME)

    # cal_* is the light curve and must be real for a variable
    assert np.isfinite(out.loc[1, "cal_g"])
    assert np.isfinite(out.loc[2, "cal_g"])
    # ext_* would conflate cloud with the star's own variation
    assert np.isnan(out.loc[1, "ext_g"])
    assert np.isnan(out.loc[2, "ext_g"])
    # a genuine standard still gets both
    assert np.isfinite(out.loc[0, "cal_g"]) and np.isfinite(out.loc[0, "ext_g"])


def test_colour_flat_band_survives_a_missing_colour():
    """G's colour term is -0.038, so an unknown B-V costs less than the noise."""
    from skycam_utils.alcor import alcor_zeropoint
    zp = alcor_zeropoint(TIME)
    assert abs(zp["g"]["color_coeff"]) <= ALCOR_COLOR_FLAT_TOL
    assert abs(zp["r"]["color_coeff"]) > ALCOR_COLOR_FLAT_TOL

    df = pd.DataFrame({"name": ["not a real star"], "altitude": [70.0],
                       "mag_g": [-10.0], "mag_r": [-10.0], "mag_b": [-10.0]})
    out = alcor_calibrate_photometry(df, time=TIME)
    assert np.isfinite(out.loc[0, "cal_g"])     # colour-flat: assume B-V = 0
    assert np.isnan(out.loc[0, "cal_r"])        # colour-sensitive: stay nan
    assert np.isnan(out.loc[0, "cal_b"])
