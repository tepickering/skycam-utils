"""
Grade nights by stellar extinction and select the clear-dark sample.

Companion to ``night_clarity.py`` (which does the heavy per-night reduction)
and ``sb_year_summary.py`` (sky brightness). Sky brightness alone cannot grade
a night -- the darkest nights of 2025 are overcast -- so clarity is taken from
``ext_g_ap``, the per-star calibrated-minus-catalog extinction, and sky
brightness is used only to confirm it.

The one non-obvious step is the BASELINE. Extinction cannot be negative, yet
the stable-night floor drifts from ~0.00 mag in the first half of 2025 to
~-0.10 by October and stays there into 2026-01. That is instrumental
throughput, not sky: it is far too large for the 0.015 mag that separates the
two ALCOR_ZEROPOINTS epochs, it is not a step at the epoch boundary
(2025-07-11), and it is not seasonal (2026-01 sits with 2025-10..12, not with
2025-01). An ABSOLUTE 0.05 mag cut therefore means different things in
different months -- 0.05 above the floor in spring, 0.15 above it in autumn --
so the cut is applied to extinction measured RELATIVE to a rolling median of
the photometric-candidate nights.

Usage: clear_dark_sample.py <clarity.csv> [...] -o OUT.png --sample OUT.csv
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLEAR_EXT = 0.05          # the clear-dark criterion, in mag, after baseline
STABLE_SCATTER = 0.025    # night-internal scatter marking a photometric night
BASELINE_DAYS = 45
MIN_FRAMES = 100


def baseline_track(tab):
    """Rolling median of the photometric-candidate nights, on a daily grid."""
    cand = tab[(tab["ext_scatter"] < STABLE_SCATTER)
               & (tab["frac_frames_clear"] > 0.95)].dropna(subset=["ext_med"])
    s = cand.set_index("date")["ext_med"].sort_index()
    grid = pd.date_range(tab["date"].min(), tab["date"].max(), freq="D")
    daily = (s.reindex(s.index.union(grid)).interpolate("time")
             .reindex(grid))
    track = daily.rolling(BASELINE_DAYS, center=True, min_periods=5).median()
    return track.bfill().ffill(), len(cand)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clarity", nargs="+", help="night_clarity.py output CSVs")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--sample", default=None, help="write the sample list here")
    p.add_argument("--grade", default=None, help="write all graded nights here")
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args()

    tab = pd.concat([pd.read_csv(f) for f in args.clarity], ignore_index=True)
    tab["date"] = pd.to_datetime(tab["night"])
    tab = tab.sort_values("date").reset_index(drop=True)

    track, n_cand = baseline_track(tab)
    tab["baseline"] = tab["date"].map(track)
    tab["ext_corr"] = tab["ext_med"] - tab["baseline"]

    ok = tab.dropna(subset=["ext_corr"]).copy()
    sample = ok[(ok["ext_corr"] <= CLEAR_EXT)
                & (ok["ext_scatter"] < 2 * STABLE_SCATTER)
                & (ok["n_frames"] >= MIN_FRAMES)]
    print(f"{len(tab)} nights, {len(ok)} reduced, {n_cand} photometric "
          f"candidates define the baseline")
    print(f"baseline swing {track.min():+.3f} to {track.max():+.3f} mag")
    print(f"CLEAR-DARK SAMPLE: {len(sample)} nights")

    if args.sample:
        sample.drop(columns=["date"]).to_csv(args.sample, index=False)
        print(f"wrote {args.sample}")
    if args.grade:
        tab.drop(columns=["date"]).to_csv(args.grade, index=False)
        print(f"wrote {args.grade}")

    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.36,
                          wspace=0.28, left=0.065, right=0.975, top=0.90,
                          bottom=0.085)

    # (a) the drift, and why an absolute cut is wrong -----------------------
    ax = fig.add_subplot(gs[0, :])
    top = 1.0
    y = ok["ext_med"].clip(upper=top)
    over = ok["ext_med"] > top
    ax.scatter(ok.loc[~over, "date"], y[~over], s=18, c="#c44e52",
               edgecolors="none", label="night median extinction")
    ax.scatter(ok.loc[over, "date"], y[over], s=22, marker="^", c="#c44e52",
               edgecolors="none", label=f"> {top:g} mag (off scale)")
    ax.scatter(sample["date"], sample["ext_med"], s=18, c="#1f77b4",
               edgecolors="none", label="clear-dark sample")
    ax.plot(track.index, track.values, "-", color="#111111", lw=1.8,
            label="throughput baseline (45 d)")
    ax.plot(track.index, track.values + CLEAR_EXT, "--", color="#111111",
            lw=1.1, label=f"baseline + {CLEAR_EXT:g} mag cut")
    ax.axhline(CLEAR_EXT, color="#888888", ls=":", lw=1.1,
               label=f"absolute {CLEAR_EXT:g} mag cut")
    ax.set_ylim(-0.20, top + 0.05)
    ax.set_ylabel("G-band extinction  [mag]")
    ax.set_title("night median extinction: the stable-night floor drifts "
                 "0.11 mag, so the cut has to track it", fontsize=10.5, pad=6)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92, ncol=3)

    # (b) the corrected distribution ----------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    bins = np.arange(-0.15, 1.01, 0.02)
    ax.hist(ok["ext_corr"].clip(upper=1.0), bins=bins, color="#c44e52",
            alpha=0.85)
    ax.axvline(CLEAR_EXT, color="k", ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("extinction relative to baseline  [mag]")
    ax.set_ylabel("nights")
    ax.set_title(f"{len(sample)} of {len(ok)} nights pass", fontsize=10)
    ax.grid(alpha=0.25)

    # (c) two independent measures agree ------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    m = ok.dropna(subset=["flicker"])
    ax.scatter(m["flicker"], m["ext_corr"].clip(upper=1.5), s=14,
               c="#c44e52", edgecolors="none", label="all nights")
    q = sample.dropna(subset=["flicker"])
    ax.scatter(q["flicker"], q["ext_corr"], s=14, c="#1f77b4",
               edgecolors="none", label="clear-dark sample")
    ax.axhline(CLEAR_EXT, color="k", ls="--", lw=1.0)
    ax.set_xscale("log")
    rho = m["flicker"].corr(m["ext_corr"], method="spearman")
    ax.set_xlabel("sky-brightness flicker  [mag]")
    ax.set_ylabel("extinction rel. baseline  [mag]")
    ax.set_title(f"photometry vs sky brightness  ($\\rho$ = {rho:.2f})",
                 fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    # (d) how the sample falls through the year -----------------------------
    ax = fig.add_subplot(gs[1, 2])
    mon = ok["night"].str[:7]
    tot = mon.value_counts().sort_index()
    got = sample["night"].str[:7].value_counts().reindex(tot.index).fillna(0)
    x = np.arange(len(tot))
    ax.bar(x, tot.values, color="#dddddd", label="nights reduced")
    ax.bar(x, got.values, color="#1f77b4", label="clear-dark")
    ax.set_xticks(x)
    ax.set_xticklabels([m[2:] for m in tot.index], rotation=90, fontsize=7)
    ax.set_ylabel("nights")
    ax.set_title("monthly yield", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Alcor night grading from stellar extinction -- "
                 f"clear-dark sample: {len(sample)} nights "
                 f"(extinction $\\leq$ {CLEAR_EXT:g} mag above baseline)",
                 fontsize=13, y=0.965)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
