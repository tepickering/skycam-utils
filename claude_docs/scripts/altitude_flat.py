"""Derive the clear-night altitude flat (ALCOR_ALTITUDE_FLAT) from registered
photometry of several clear nights.

Why: with the blend correction and frame registration in place (2026-10-14),
clear-night ext_* still ran -0.12 at alt 20, +0.05 at alt 45 and -0.10 at the
zenith in G. That shape is the same on every clear night tested (2025-01-01 to
2026-10-14, per-night bins within ~0.01 of each other), azimuthally symmetric to
~0.01, and the same for bright and faint stars at a given altitude -- so it is
the instrument, not the atmosphere and not a magnitude effect. The single
airmass term over-corrects the low altitudes and on-axis stars read bright
(vignetting); the two are degenerate for a fixed all-sky camera, so it is
corrected empirically rather than split into atmosphere and optics.

Method, per band:
  * calibrate WITHOUT the flat (alcor_calibrate_photometry(flat=False));
  * keep non-variable stars above alt 20, clear of not-sky by
    ALCOR_EXT_HORIZON_MARGIN, isolated (blend > -0.02) and inside
    ALCOR_EXT_MAG_WINDOW by their EXPECTED instrumental mag (mag - ext), not the
    measured one -- selecting on a noisy measured mag biases the faint edge;
  * alternate medians: F(alt bin) and an offset per (night, star), which
    absorbs catalog errors and each night's throughput;
  * fit a quartic in zenith distance to the 2.5 deg bins and evaluate it at
    the 5 deg nodes printed at the end.
The flat's level is degenerate with the zeropoints; check with
zeropoint_calib.py --blend [--flat] that it moves them by <~0.01 or refit.

Input: per-frame *_phot.csv written by alcor_star_photometry with registration
on (the default), dark moonless frames of clear nights -- ~200 frames a night
is plenty. Nights used for the shipped table: 2025-01-01, 2026-05-15,
2026-05-18, 2026-06-20, 2026-08-16, 2026-09-18, 2026-10-14 (2026-10-03 left
out: cloud near the zenith).

Usage:  altitude_flat.py NIGHT_DIR [NIGHT_DIR ...]
"""
import glob
import sys
import warnings

import numpy as np
import pandas as pd
from astropy.time import Time

from skycam_utils.alcor import (ALCOR_EXT_HORIZON_MARGIN, ALCOR_EXT_MAG_WINDOW,
                                alcor_calibrate_photometry,
                                collect_alcor_photometry)
from skycam_utils.alcor.extinction import _star_horizon_distance

warnings.filterwarnings("ignore")
EDGES = np.arange(20.0, 90.1, 2.5)
NODES = np.arange(20.0, 90.1, 5.0)
DEG = 4


def load(night_dir):
    df = collect_alcor_photometry(sorted(glob.glob(f"{night_dir}/*_phot.csv")))
    frames = []
    for t, g in df.groupby("OBSTIME"):
        g = alcor_calibrate_photometry(g, time=Time(t), flat=False)
        g["hdist"] = _star_horizon_distance(g, Time(t))
        frames.append(g)
    df = pd.concat(frames, ignore_index=True)
    df = df[~df["variable"].astype(bool) & (df["altitude"] > 20)
            & (df["hdist"] > ALCOR_EXT_HORIZON_MARGIN)]
    df["key"] = night_dir.rstrip("/") + "|" + df["name"]
    return df


def band_flat(df, band):
    lo, hi = ALCOR_EXT_MAG_WINDOW
    expected = df["mag_g"] - df["ext_g"]
    x = df[np.isfinite(df[f"ext_{band}"]) & expected.between(lo, hi)
           & (df[f"blend_{band}"] > -0.02)].copy()
    x["bin"] = pd.cut(x["altitude"], EDGES)
    ext = x[f"ext_{band}"]
    s = pd.Series(0.0, index=x["key"].unique())
    for _ in range(10):
        F = (ext - x["key"].map(s)).groupby(x["bin"], observed=False).median()
        s = (ext - x["bin"].map(F).astype(float)).groupby(x["key"]).median()
    res = ext - x["key"].map(s)
    g = res.groupby(x["bin"], observed=False)
    med, n = g.median().to_numpy(), g.size().to_numpy()
    mad = g.apply(lambda v: 1.4826 * np.median(np.abs(v - v.median()))).to_numpy()
    # rows of one star in one night are correlated; ~5 frames per independent sample
    err = 1.25 * mad / np.sqrt(np.maximum(n / 5, 1))
    z = 90.0 - 0.5 * (EDGES[1:] + EDGES[:-1])
    c = np.polyfit(z, med, DEG, w=1 / err)
    chi2 = np.sum(((med - np.polyval(c, z)) / err) ** 2) / (len(z) - DEG - 1)
    return np.polyval(c, 90.0 - NODES), med, chi2, x["key"].nunique()


def main():
    df = pd.concat([load(d) for d in sys.argv[1:]], ignore_index=True)
    print(f"alt nodes: {tuple(float(a) for a in NODES)}")
    for band in "rgb":
        nodes, med, chi2, nstar = band_flat(df, band)
        print(f"{band}: {nstar} night-stars, reduced chi2 {chi2:.2f}")
        print(f"   bins : {' '.join(f'{v:+.3f}' for v in med)}")
        print(f'   "{band}": ({", ".join(f"{v:.3f}" for v in nodes)}),')


if __name__ == "__main__":
    main()
