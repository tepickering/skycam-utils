"""Build the packaged bright-variable catalog from AAVSO VSX.

Why this exists: bright_star_sloan_named.fits excludes variable stars by
construction, which is correct -- they would corrupt the zeropoints -- but it
means the all-sky camera has never measured any of them. Polaris (HD 8890,
V=1.98, a classical Cepheid) is the worked example: it was not merely unmeasured
but was being masked as a cluster of hot pixels, because nothing knew it was a
real source. An all-sky camera images the same bright stars every clear night
for years at no scheduling cost, which is a good match for bright variables.

Selection, and the evidence for each cut:
  * VSX variability flag V == 0 (confirmed variable, not suspected).
  * `max` given in the V band, and `min` a real magnitude rather than an
    amplitude (VSX flags the latter with f_min='Y').
  * max <= 6.0. Measured on 2026-05-18 above alt 40, the per-star scatter over a
    night is 0.08 mag at V 2.5-4.0, 0.14 at V 4.5-5.0 and 0.21 at V 5.0-5.5,
    with 100% detection throughout -- so V=6 is about where a single frame stops
    being useful, though nightly means go deeper.
  * amplitude (min - max) >= 0.05 mag. Per-frame scatter is the wrong yardstick
    here: at 1000-1500 frames a night the nightly MEAN is good to ~0.005 mag, so
    the binding limit is the ~0.03 mag epoch-to-epoch zeropoint systematic, and
    0.05 mag sits comfortably above it. The smallest amplitudes need folding
    across nights rather than a single night -- which is precisely what a camera
    that runs every clear night is for. Polaris (alf UMi, DCEPS, 1.96-2.03, so
    0.07 mag, P=3.97 d) is the case that set this threshold; a 0.2 mag cut
    excluded it along with most low-amplitude Cepheids and Be stars.
  * NOT eruptive/cataclysmic. For a nova the tabulated `max` is a one-time
    historical outburst, so the star sits at minimum permanently and is not a
    monitoring target; 57 such entries would otherwise pass the cuts.

Also removes from the calibration catalogs the variables that leaked into them
AND vary by more than CAL_REMOVE_AMPLITUDE -- two eclipsing binaries, psi Cen and
N Sco. Low-amplitude variables that are also good standards (Arcturus, Capella)
stay. B-V comes from a Bright Star Catalogue cross-match, since VSX carries no
colors and the calibration catalog by construction does not contain these stars.

Usage:  python claude_docs/scripts/build_variable_catalog.py [--dry-run]
"""
import argparse
import re

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astroquery.vizier import Vizier

VMAG_MAX = 6.0
AMPLITUDE_MIN = 0.05
# Amplitude above which a variable is unfit to be a photometric STANDARD and is
# removed from the calibration catalogs. Deliberately far looser than
# AMPLITUDE_MIN: the two thresholds answer different questions. A star varying by
# 0.05-0.1 mag is worth a light curve AND is still a perfectly good standard when
# hundreds are averaged -- Arcturus (0.04) and Capella are in this class, and
# removing them would degrade the zeropoints rather than protect them. A 0.3 mag
# eclipser is not. Only 2 stars (psi Cen, N Sco) cross this line.
CAL_REMOVE_AMPLITUDE = 0.2
MATCH_RADIUS = 20 * u.arcsec

# Types whose catalogued maximum is a one-off eruption rather than a state the
# star returns to: classical/recurrent novae, supernovae, dwarf novae, symbiotic
# and R CrB stars.
# The terminator must include whitespace: VSX writes supernova types as
# "SN I", "SN Ia", "SN II-P", and without \s four historical supernovae
# survive the cut -- Tycho's (B Cas, 1572), Kepler's (V0843 Oph, 1604),
# S And (1885) and SN 1987A, all now V 16-22. V838MON is here for the same
# reason: its one member passing the cuts is CK Vul, Nova Vulpeculae 1670, whose
# 2.6 mag maximum is 350 years stale and which now sits at V=23. Everything else
# above 5 mag of amplitude is a Mira or an SR with a real period -- stars that do
# return to maximum -- plus eta Car, which is genuinely variable at V~4.5 today.
ERUPTIVE = re.compile(r"^(N[ABCLR]?|SN|UG[SZ]?|ZAND|RCB|V838MON)($|[\s/:+])")

CAL_CATALOG = "skycam_utils/data/bright_star_sloan_named.fits"
CAL_CATALOG_PLAIN = "skycam_utils/data/bright_star_sloan.fits"
OUT = "skycam_utils/data/bright_variable_vsx.fits"


def main(dry_run=False):
    vizier = Vizier(columns=["*"], row_limit=-1, timeout=600)
    vizier.column_filters = {"max": f"<{VMAG_MAX + 0.5}"}
    vsx = vizier.get_catalogs("B/vsx/vsx")[0]
    print(f"VSX rows with max < {VMAG_MAX + 0.5}: {len(vsx)}")

    flag = np.array(vsx["V"])
    mx = np.array(vsx["max"], dtype=float)
    mn = np.array(vsx["min"], dtype=float)
    f_min = np.array([str(x) for x in vsx["f_min"]])
    n_max = np.array([str(x) for x in vsx["n_max"]])
    types = np.array([str(x) for x in vsx["Type"]])
    eruptive = np.array([bool(ERUPTIVE.match(t)) for t in types])

    keep = (
        (flag == 0)
        & np.isfinite(mx) & np.isfinite(mn)
        & (f_min != "Y") & (n_max == "V")
        & (mx <= VMAG_MAX) & ((mn - mx) >= AMPLITUDE_MIN)
        & ~eruptive
    )
    without_eruptive_cut = keep | (eruptive & (
        (flag == 0) & np.isfinite(mx) & np.isfinite(mn) & (f_min != "Y")
        & (n_max == "V") & (mx <= VMAG_MAX) & ((mn - mx) >= AMPLITUDE_MIN)))
    print(f"  confirmed, V-band, max<={VMAG_MAX}, amp>={AMPLITUDE_MIN}: "
          f"{int(without_eruptive_cut.sum())}, of which "
          f"{int(without_eruptive_cut.sum() - keep.sum())} eruptive/cataclysmic "
          f"dropped -> {int(keep.sum())}")
    var = vsx[keep]

    vcoord = SkyCoord(np.array(var["RAJ2000"]) * u.deg,
                      np.array(var["DEJ2000"]) * u.deg)

    # B-V has to come from the Bright Star Catalogue directly. Cross-matching
    # against bright_star_sloan_named.fits recovers almost nothing, for the
    # reason this whole catalog exists: that file excludes variables, so the
    # only hits are the handful that leaked in.
    bsc_q = Vizier(columns=["HD", "Vmag", "B-V", "U-B", "RAJ2000", "DEJ2000"],
                   row_limit=-1, timeout=600)
    bsc_q.column_filters = {"Vmag": f"<{VMAG_MAX + 0.5}"}
    bsc = bsc_q.get_catalogs("V/50/catalog")[0]
    print(f"BSC rows with Vmag < {VMAG_MAX + 0.5}: {len(bsc)}")
    # the BSC serves RA/Dec as sexagesimal strings, RA in hours
    bcoord = SkyCoord(np.array([str(x) for x in bsc["RAJ2000"]]),
                      np.array([str(x) for x in bsc["DEJ2000"]]),
                      unit=(u.hourangle, u.deg))
    bidx, bsep, _ = vcoord.match_to_catalog_sky(bcoord)
    bmatched = bsep < MATCH_RADIUS

    cal = Table.read(CAL_CATALOG)
    ccoord = SkyCoord(np.array(cal["_RAJ2000"], dtype=float) * u.deg,
                      np.array(cal["_DEJ2000"], dtype=float) * u.deg)
    idx, sep, _ = vcoord.match_to_catalog_sky(ccoord)
    matched = sep < MATCH_RADIUS

    out = Table()
    out["NAME"] = [str(x).strip() for x in var["Name"]]
    out["_RAJ2000"] = np.array(var["RAJ2000"], dtype=float)
    out["_DEJ2000"] = np.array(var["DEJ2000"], dtype=float)
    out["Type"] = [str(x).strip() for x in var["Type"]]
    out["Vmax"] = np.array(var["max"], dtype=float)
    out["Vmin"] = np.array(var["min"], dtype=float)
    out["Amplitude"] = out["Vmin"] - out["Vmax"]
    out["Period"] = np.array(var["Period"], dtype=float)
    out["SpType"] = [str(x).strip() for x in var["Sp"]]
    # Vmag is the catalog's "how bright is this star" column, used for the same
    # vmag_limit filtering the calibration catalog gets. Maximum light is the
    # right choice: it decides whether the star is ever measurable at all.
    out["Vmag"] = out["Vmax"]

    bv = np.full(len(var), np.nan)
    bv[bmatched] = np.array(bsc["B-V"], dtype=float)[bidx[bmatched]]
    out["B-V"] = bv
    hd = np.full(len(var), -1, dtype=int)
    hd[bmatched] = np.array([int(x) if str(x).strip() not in ("", "--") else -1
                             for x in bsc["HD"]])[bidx[bmatched]]
    out["HD"] = hd
    print(f"  matched to the BSC: {int(bmatched.sum())}/{len(out)}; "
          f"B-V known for {int(np.isfinite(bv).sum())}")

    amplitude = np.array(out["Amplitude"], dtype=float)
    dup = matched & (amplitude >= CAL_REMOVE_AMPLITUDE)
    print(f"  in {CAL_CATALOG}: {int(matched.sum())} of {len(out)}; "
          f"{int(dup.sum())} vary by >= {CAL_REMOVE_AMPLITUDE} mag and will be "
          f"removed: {[str(cal['NAME'][j]).strip() for j in idx[dup]]}")
    print(f"  the other {int((matched & ~dup).sum())} stay -- low-amplitude "
          f"variables are still sound standards in an ensemble")

    if dry_run:
        print("dry run; nothing written")
        return

    out.write(OUT, overwrite=True)
    print(f"wrote {OUT}  ({len(out)} stars)")

    # Drop only the high-amplitude leaks from the calibration catalogs.
    badcoord = vcoord[dup]
    if len(badcoord):
        for path in (CAL_CATALOG, CAL_CATALOG_PLAIN):
            t = Table.read(path)
            coords = SkyCoord(np.array(t["_RAJ2000"], dtype=float) * u.deg,
                              np.array(t["_DEJ2000"], dtype=float) * u.deg)
            _, sep_cal, _ = coords.match_to_catalog_sky(badcoord)
            drop = sep_cal < MATCH_RADIUS
            if drop.any():
                t[~drop].write(path, overwrite=True)
                print(f"  {path}: dropped {int(drop.sum())}, "
                      f"{len(t)} -> {int((~drop).sum())} rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    main(**vars(p.parse_args()))
