"""
How many frames must be averaged to beat the frame-to-frame photometric scatter?

Uses a clear, moon-free night (2025-01-01), where anything left after removing
each star's slow airmass/extinction trend is measurement noise rather than sky.
For every well-measured star we compute the robust scatter of block means,
sigma(N), against block size N, and compare it to the sigma(1)/sqrt(N) line that
uncorrelated noise would follow. Where the measured curve leaves that line is
where averaging stops paying.

Also reports the star's on-sensor drift rate, since intra-pixel sensitivity
noise is a function of sub-pixel phase: a star that crosses a whole pixel
between frames re-randomizes its phase every frame (so the noise averages down
like white noise), while a star near the celestial pole barely moves and should
stay correlated for many frames.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Instrumental-magnitude window that is free of the bright-star CMOS
# non-linearity and the faint-end bias (see the usable-window analysis).
MAG_LO, MAG_HI = -11.0, -9.5
MIN_FRAMES = 500
DETREND_DEG = 3          # gentle: the Allan statistic kills the rest
BLOCKS = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200]


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def sigma_vs_block(resid, blocks=BLOCKS):
    """
    Allan-style scatter of block means vs block size: the robust scatter of the
    DIFFERENCE between adjacent block means, over sqrt(2).

    Differencing adjacent blocks is what makes this usable here. A plain scatter
    of block means is contaminated by whatever slow trend the detrending failed
    to remove -- and, worse, by the trend it over-removed: a whole-night
    polynomial suppresses power at large N and drives the curve BELOW the
    white-noise line, which reads as "averaging works better than random",
    an impossibility. Adjacent blocks are minutes apart, so any smooth trend
    cancels in the difference and only the noise survives. For uncorrelated
    noise this still equals sigma(1)/sqrt(N).
    """
    out = {}
    for n in blocks:
        nblk = len(resid) // n
        if nblk < 12:                     # too few blocks to measure a scatter
            continue
        means = resid[: nblk * n].reshape(nblk, n).mean(axis=1)
        out[n] = robust_std(np.diff(means)) / np.sqrt(2.0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv")
    p.add_argument("--band", default="g")
    p.add_argument("--method", default="ap", choices=["ap", "gauss"])
    p.add_argument("--min-altitude", type=float, default=30.0)
    p.add_argument("--detrend-deg", type=int, default=DETREND_DEG)
    p.add_argument("--plot", default=None, help="Write the sigma(N) figure here.")
    p.add_argument("-o", "--output", default=None)
    args = p.parse_args()

    mag = f"mag_{args.band}_{args.method}"
    cal = f"cal_{args.band}_{args.method}"
    sat = f"sat_{args.band}_{args.method}"
    flux = f"flux_{args.band}_{args.method}"
    cols = ["name", "OBSTIME", "altitude", "variable", "xcen", "ycen",
            mag, cal, sat, flux]

    df = pd.read_csv(args.csv, usecols=cols, parse_dates=["OBSTIME"])
    print(f"loaded {len(df):,} rows, {df['name'].nunique()} stars")

    keep = (
        (~df["variable"].astype(bool))
        & (~df[sat].astype(bool))
        & (df[flux] > 0)
        & np.isfinite(df[cal])
        & df[mag].between(MAG_LO, MAG_HI)
        & (df["altitude"] > args.min_altitude)
    )
    df = df[keep].sort_values(["name", "OBSTIME"])
    print(f"{len(df):,} rows survive the usable-window cuts "
          f"({df['name'].nunique()} stars)")

    rows = []
    curves = {}
    for name, g in df.groupby("name", sort=False):
        if len(g) < MIN_FRAMES:
            continue
        t = (g["OBSTIME"] - g["OBSTIME"].iloc[0]).dt.total_seconds().to_numpy()
        y = g[cal].to_numpy()
        # Remove only the slow trend; a low-order polynomial in time cannot
        # absorb the frame-to-frame jitter we are trying to measure.
        trend = np.polyval(np.polyfit(t / t[-1], y, args.detrend_deg), t / t[-1])
        resid = y - trend

        sig = sigma_vs_block(resid)
        if 1 not in sig:
            continue
        curves[name] = sig

        step = np.hypot(np.diff(g["xcen"].to_numpy()),
                        np.diff(g["ycen"].to_numpy()))
        rows.append({
            "name": name,
            "nframes": len(g),
            "mag": g[mag].median(),
            "drift_px_per_frame": np.median(step),
            "sigma1": sig[1],
            **{f"sigma{n}": sig.get(n, np.nan) for n in BLOCKS[1:]},
        })

    res = pd.DataFrame(rows).sort_values("drift_px_per_frame")
    print(f"\n{len(res)} stars with >= {MIN_FRAMES} usable frames\n")

    # Pooled curve: median across stars, vs the white-noise expectation.
    print(f"{'N':>5} {'sigma(N)':>10} {'white':>10} {'ratio':>7}   "
          f"{'equiv. minutes':>14}")
    s1 = res["sigma1"].median()
    cadence = 27.5    # s/frame on this night (1571 frames over ~12 h)
    for n in BLOCKS:
        col = "sigma1" if n == 1 else f"sigma{n}"
        if col not in res:
            continue
        s = res[col].median()
        if not np.isfinite(s):
            continue
        white = s1 / np.sqrt(n)
        print(f"{n:5d} {s:10.4f} {white:10.4f} {s/white:7.2f}   "
              f"{n * cadence / 60:14.1f}")

    # Drift dependence: intra-pixel phase re-randomizes only when the star
    # actually moves across the sensor.
    print("\nby drift rate (slowest = nearest the celestial pole):")
    res["bin"] = pd.cut(res["drift_px_per_frame"],
                        [0, 0.25, 0.5, 0.75, 1.0, 10.0])
    for b, g in res.groupby("bin", observed=True):
        print(f"  {str(b):>14}  n={len(g):3d}  sigma(1)={g['sigma1'].median():.4f}  "
              f"sigma(20)/white={g['sigma20'].median() / (g['sigma1'].median() / np.sqrt(20)):.2f}")

    if args.plot:
        # Drop block sizes that no longer have enough blocks to measure a
        # scatter -- their median is NaN and would poison the whole curve.
        def _col(n):
            return "sigma1" if n == 1 else f"sigma{n}"
        ns = [n for n in BLOCKS
              if _col(n) in res and np.isfinite(res[_col(n)].median())]
        med = np.array([res[_col(n)].median() for n in ns])
        q1 = np.array([res[_col(n)].quantile(.25) for n in ns])
        q3 = np.array([res[_col(n)].quantile(.75) for n in ns])
        mins = np.array(ns) * cadence / 60.0

        fig, ax = plt.subplots(figsize=(9, 6.5))
        ax.fill_between(ns, q1, q3, color="#1f77b4", alpha=0.18,
                        label="per-star interquartile range")
        ax.plot(ns, med, "o-", color="#1f77b4", lw=1.8, ms=5,
                label=f"measured $\\sigma(N)$  ({len(res)} stars)")
        ax.plot(ns, med[0] / np.sqrt(ns), "--", color="#d62728", lw=1.5,
                label=r"uncorrelated noise, $\sigma(1)/\sqrt{N}$")

        # Mark where the measured curve leaves the white-noise line by >10%.
        ratio = med / (med[0] / np.sqrt(ns))
        depart = next((n for n, r in zip(ns, ratio) if r > 1.10), None)
        if depart:
            ax.axvline(depart, color="0.4", ls=":", lw=1.2)
            ax.annotate(f"averaging stops paying\nN $\\approx$ {depart} frames"
                        f"  ({depart * cadence / 60:.0f} min)",
                        xy=(depart, med[0] / np.sqrt(depart)),
                        xytext=(depart * 1.25, med[0] / np.sqrt(depart) * 2.1),
                        fontsize=10, color="0.25",
                        arrowprops=dict(arrowstyle="->", color="0.45", lw=1.1))

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N  (frames averaged)")
        ax.set_ylabel(r"$\sigma(N)$  [mag]")
        ax.set_title("2025-01-01 (clear, moon-free): how far frame averaging goes\n"
                     f"per-frame scatter {med[0]:.3f} mag; floor $\\approx$ "
                     f"{med.min():.3f} mag", fontsize=11)
        ax.grid(which="both", alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")

        top = ax.secondary_xaxis("top", functions=(lambda n: n * cadence / 60,
                                                   lambda m: m * 60 / cadence))
        top.set_xlabel("equivalent time [minutes]")

        fig.savefig(args.plot, dpi=140, bbox_inches="tight")
        print(f"\nwrote {args.plot}")

    if args.output:
        res.drop(columns="bin").to_csv(args.output, index=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
