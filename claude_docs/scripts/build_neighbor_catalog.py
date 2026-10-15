"""Build the packaged aperture-neighbour catalog from Hipparcos.

Why this exists: the photometry aperture is 4 px, and the plate scale is only
6.6-8.3 px/deg (radially) -- so the aperture is ~0.5 deg across, and any other
star inside it adds its flux to the measurement. Measured on the clear
2026-10-14 night, ~80% of the stars in ALCOR_EXT_MAG_WINDOW carry more than 1%
neighbour flux, the per-star ext_g tracks the predicted blend term at r~0.7
(on every night tested), and the blending alone took the night's median
extinction from ~0 to -0.11. The worst cases are real pairs half an aperture
apart: HD 166229 + HIP 89008 (24') read -0.79 against a predicted -0.79, and
nu Dra (24 + 25 Dra, 1') -0.69 each.

For every calibration-catalog star this records each Hipparcos neighbour within
SEARCH_RADIUS as a tangent-plane offset (east, north) plus its V and B-V. The
offset is all that is needed: which of those neighbours actually fall in the
aperture depends on where the star is on the sky -- the aperture is a 0.61 x
0.39 deg ellipse at alt 20 and a 0.48 deg circle at the zenith -- so that is
resolved per measurement by skycam_utils.alcor.blend, not here.

The target's own Hipparcos entry is excluded, and finding it is less trivial
than "the nearest": catalog and Hipparcos positions disagree by up to ~40"
(30 Cas's own entry sits 33" away), so the self entry is the nearest Hipparcos star
within SELF_RADIUS whose V matches the catalog V (to SELF_DV), else the best V
match out to SELF_SEARCH, else simply the nearest within SELF_RADIUS. The order
matters: searching wide first takes the twin of an equal pair (24/25 Dra, 62"
apart and 0.03 mag different) as the star itself. Counting a star's own light as a neighbour
predicts a -0.75 mag blend that does not exist. Where Hipparcos resolves a
close pair (< SELF_RADIUS) that the catalog lists as one star, the catalog V is
the combined light, so every component within SELF_RADIUS counts as "self"
whenever their combined V is closer to the catalog V than the best single
match is.

Hipparcos is complete to V~8; it is queried to NEIGHBOUR_VMAX = 9.5. Fainter
field stars add a near-uniform few-percent that the zeropoints absorb.

Usage:  python claude_docs/scripts/build_neighbor_catalog.py [--dry-run]
"""
import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from skycam_utils.alcor.catalogs import _alcor_star_labels

DATA = Path(__file__).resolve().parents[2] / "skycam_utils" / "data"
CATALOG = DATA / "bright_star_sloan_named.fits"
OUTPUT = DATA / "bright_star_neighbors.fits"

NEIGHBOUR_VMAX = 9.5
# The aperture's largest extent is 4 px / 6.6 px/deg = 0.61 deg (radially, at
# alt 20), plus ~3 PSF sigma (0.7 px); 0.8 deg covers it with margin.
SEARCH_RADIUS = 0.8 * u.deg
SELF_RADIUS = 30 * u.arcsec
SELF_SEARCH = 90 * u.arcsec
SELF_DV = 0.3
# Anything this close is part of the target's own catalog light: a component
# the catalog does not resolve, or a duplicate entry (1 Equ has two, 10" apart).
SELF_ALWAYS = 5 * u.arcsec

VIZIER = ("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=I/239/hip_main"
          "&-out=HIP,RAICRS,DEICRS,Vmag,B-V&Vmag=%3C{vmax}&-out.max=unlimited")


def fetch_hipparcos(vmax=NEIGHBOUR_VMAX):
    with urllib.request.urlopen(VIZIER.format(vmax=vmax), timeout=600) as resp:
        text = resp.read().decode()
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    # header, units, dashes, then data
    df = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[3:])), sep="\t")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["RAICRS", "DEICRS", "Vmag"]).reset_index(drop=True)


def build(hip, cat):
    labels = _alcor_star_labels(cat)
    cc = SkyCoord(np.asarray(cat["_RAJ2000"]) * u.deg,
                  np.asarray(cat["_DEJ2000"]) * u.deg)
    hc = SkyCoord(hip["RAICRS"].to_numpy() * u.deg,
                  hip["DEICRS"].to_numpy() * u.deg)
    ic, ih, sep, _ = hc.search_around_sky(cc, SEARCH_RADIUS)
    pairs = pd.DataFrame({"c": ic, "h": ih, "sep": sep.arcsec})
    pairs["V"] = hip["Vmag"].to_numpy()[pairs["h"]]
    vcat = np.asarray(cat["Vmag"], dtype=float)

    rows = []
    nself_multi = 0
    for c, grp in pairs.groupby("c"):
        close = grp[grp["sep"] < SELF_RADIUS.to_value(u.arcsec)]
        wide = grp[grp["sep"] < SELF_SEARCH.to_value(u.arcsec)]
        dv = (wide["V"] - vcat[c]).abs()
        selfh = set()
        best = None
        dvc = (close["V"] - vcat[c]).abs()
        if len(close) and dvc.min() <= SELF_DV:
            # a V-consistent entry within SELF_RADIUS: the nearest such one.
            # Searching wider first would take the TWIN of an equal pair (24 Dra
            # is 62" from 25 Dra and 0.03 mag from it) as the star itself.
            best = close[dvc <= SELF_DV].sort_values("sep").iloc[0]
        elif len(wide) and dv.min() <= SELF_DV:
            best = wide.loc[dv.idxmin()]
        elif len(close):
            best = close.loc[close["sep"].idxmin()]
        if best is not None:
            selfh = {int(best["h"])}
            if len(close) > 1:
                combined = -2.5 * np.log10(np.sum(10 ** (-0.4 * close["V"])))
                if abs(combined - vcat[c]) < abs(best["V"] - vcat[c]):
                    selfh = set(close["h"]) | selfh
                    nself_multi += 1
        selfh |= set(grp.loc[grp["sep"] < SELF_ALWAYS.to_value(u.arcsec), "h"])
        for _, p in grp.iterrows():
            if int(p["h"]) in selfh:
                continue
            rows.append((c, int(p["h"])))

    ci = np.array([r[0] for r in rows], dtype=int)
    hi = np.array([r[1] for r in rows], dtype=int)
    de, dn = cc[ci].spherical_offsets_to(hc[hi])
    out = Table()
    out["LABEL"] = np.array([labels[i] for i in ci])
    out["TARGET_HD"] = np.array([int(cat["HD"][i]) if not np.ma.is_masked(cat["HD"][i])
                                 else -1 for i in ci], dtype=np.int32)
    out["HIP"] = hip["HIP"].to_numpy()[hi].astype(np.int32)
    out["DE"] = de.deg.astype(np.float32)
    out["DN"] = dn.deg.astype(np.float32)
    out["V"] = hip["Vmag"].to_numpy()[hi].astype(np.float32)
    out["BV"] = hip["B-V"].to_numpy()[hi].astype(np.float32)
    out["DE"].unit = out["DN"].unit = "deg"
    out.meta["COMMENT"] = ["Hipparcos (I/239) neighbours of bright_star_sloan_named.fits",
                           "DE/DN: tangent-plane offset east/north of the target",
                           "built by claude_docs/scripts/build_neighbor_catalog.py"]
    out.meta["NBRVMAX"] = NEIGHBOUR_VMAX
    out.meta["SEARCHR"] = SEARCH_RADIUS.to_value(u.deg)
    return out, nself_multi


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cat = Table.read(str(CATALOG))
    hip = fetch_hipparcos()
    print(f"Hipparcos V<{NEIGHBOUR_VMAX}: {len(hip)} stars")
    out, nmulti = build(hip, cat)
    ntargets = len(set(out["LABEL"]))
    print(f"{len(out)} neighbour rows for {ntargets} of {len(cat)} catalog stars; "
          f"{nmulti} targets matched to a resolved close pair")
    if args.dry_run:
        return
    out.write(str(OUTPUT), overwrite=True)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
