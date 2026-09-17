"""
Year-scale sky-brightness summary from a season of ``sky_brightness.csv`` files.

One night's CSV is already plotted by the packaged ``plot_alcor_sb_summary``;
this is the population view over a whole archive run -- what the site's darkness
distribution looks like, how it moves through the year, what moonlight costs,
and how far the two light domes sit above the sky.

It stays a script rather than a CLI for the same reason the montage does: a
year summary is a way of looking at an archive, not a data product.

Usage: sb_year_summary.py <products-dir> -o OUT.png [--year 2025]
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dark-sky selection: the same cut plot_alcor_sb_summary uses for its title
# medians -- astronomical twilight AND no Moon, so neither can drag a median.
SUN_DARK = -18.0
MOON_DOWN = 0.0
MIN_DARK_FRAMES = 100

# Frame-to-frame flicker (robust scatter of successive zenith differences, so
# the slow nightly trend cancels) separates clear from cloudy far better than
# the median does -- see the 2025-01-01/02/03 graded test set, where it ran
# 0.006 / 0.008 / 0.024. The cut here is that separation applied to a year.
#
# It is load-bearing for this summary, NOT a refinement: the four DARKEST
# nights of 2025 are all overcast. Thick cloud at a dark site blocks starlight
# and airglow along with the city light, so an overcast monsoon night reads
# ~1.5 mag darker than a clear one and would otherwise take the headline.
# Darkest is not best.
CLEAR_FLICKER = 0.010


def robust_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def load(products, year, cache):
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache)
    cols = ["OBSTIME", "sun_alt", "moon_alt", "allsky_mv_zenith",
            "allsky_mv_tucson", "allsky_mv_nogales", "allsky_mv_best",
            "best_alt"]
    frames = []
    for path in sorted(glob.glob(os.path.join(products, f"{year}-*",
                                              "sky_brightness.csv"))):
        night = os.path.basename(os.path.dirname(path))
        try:
            df = pd.read_csv(path, usecols=cols)
        except Exception as exc:                        # empty / truncated
            print(f"  skip {night}: {exc}")
            continue
        df["night"] = night
        frames.append(df)
    if not frames:
        raise SystemExit(f"no sky_brightness.csv under {products} for {year}")
    out = pd.concat(frames, ignore_index=True)
    out["OBSTIME"] = pd.to_datetime(out["OBSTIME"])
    print(f"loaded {len(frames)} nights, {len(out):,} frames")
    if cache:
        out.to_parquet(cache)
    return out


def nightly(dark):
    """Per-night dark-sky statistics, indexed by night date."""
    g = dark.groupby("night")
    tab = pd.DataFrame({
        "n": g.size(),
        "zenith": g["allsky_mv_zenith"].median(),
        "z_lo": g["allsky_mv_zenith"].quantile(0.25),
        "z_hi": g["allsky_mv_zenith"].quantile(0.75),
        "best": g["allsky_mv_best"].median(),
        "tucson": g["allsky_mv_tucson"].median(),
        "nogales": g["allsky_mv_nogales"].median(),
    })
    tab["flicker"] = g["allsky_mv_zenith"].apply(lambda s: robust_std(np.diff(s)))
    tab = tab[tab["n"] >= MIN_DARK_FRAMES]
    tab.index = pd.to_datetime(tab.index)
    return tab.sort_index()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("products")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--year", default="2025")
    p.add_argument("--cache", default=None)
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args()

    df = load(args.products, args.year, args.cache)
    astro = df[df["sun_alt"] < SUN_DARK]
    dark = astro[astro["moon_alt"] < MOON_DOWN]
    tab = nightly(dark)
    print(f"{len(tab)} nights with >= {MIN_DARK_FRAMES} dark moonless frames; "
          f"{len(dark):,} such frames of {len(astro):,} astronomical-dark")

    fig = plt.figure(figsize=(14.5, 9.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0],
                          hspace=0.34, wspace=0.28,
                          left=0.065, right=0.975, top=0.90, bottom=0.085)

    # (a) the year, night by night ------------------------------------------
    clear = tab[tab["flicker"] < CLEAR_FLICKER]
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(tab.index, tab["z_lo"], tab["z_hi"], color="#777777",
                    alpha=0.18, lw=0, label="zenith, night IQR")
    sc = ax.scatter(tab.index, tab["zenith"], c=tab["flicker"], s=22,
                    cmap="viridis_r", vmin=0.004, vmax=0.030, zorder=3,
                    edgecolors="none")
    ax.plot(clear.index, clear["best"], "_", color="#111111", ms=6, lw=1,
            label="darkest cone (clear nights)")
    ax.axhline(clear["zenith"].median(), color="#1f77b4", ls="--", lw=1.1,
               label=f"clear-night zenith median {clear['zenith'].median():.2f}")
    ax.invert_yaxis()
    ax.set_ylabel("sky surface brightness\n[V mag arcsec$^{-2}$]")
    ax.set_title(f"{args.year}: dark-sky nightly medians "
                 f"(Sun < {SUN_DARK:g}$^\\circ$, Moon down) -- "
                 f"{len(tab)} nights, {len(clear)} clear", fontsize=10.5, pad=6)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9, ncol=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.012, aspect=28, extend="both")
    cb.set_label("frame-to-frame flicker [mag]\n(cloud proxy)", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    # (b) the distribution ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    bins = np.arange(18.5, 23.51, 0.05)
    clear_nights = set(clear.index.strftime("%Y-%m-%d"))
    is_clear = dark["night"].isin(clear_nights)
    for sel, lab, color in ((is_clear, "clear nights", "#1f77b4"),
                            (~is_clear, "cloudy / marginal", "#c44e52")):
        v = dark.loc[sel, "allsky_mv_zenith"].dropna()
        if not len(v):
            continue
        ax.hist(v, bins=bins, histtype="step", color=color, lw=1.4,
                label=f"{lab}  med {np.median(v):.2f}")
    ax.set_xlabel("zenith  [V mag arcsec$^{-2}$]")
    ax.set_ylabel("dark moonless frames")
    ax.set_title("per-frame zenith distribution", fontsize=10)
    # The cloudy population is BIMODAL and straddles the clear peak: cloud that
    # reflects city light sits brighter, thick overcast that blocks starlight
    # sits darker. A one-sided "too bright = cloud" test would miss half of it.
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.grid(alpha=0.25)

    # (c) what the Moon costs ------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    edges = np.arange(-20, 65.1, 5.0)
    mid = 0.5 * (edges[:-1] + edges[1:])
    z = astro["allsky_mv_zenith"].to_numpy()
    m = astro["moon_alt"].to_numpy()
    idx = np.digitize(m, edges) - 1
    med = np.array([np.nanmedian(z[idx == i]) if np.any(idx == i) else np.nan
                    for i in range(len(mid))])
    lo = np.array([np.nanpercentile(z[idx == i], 10) if np.any(idx == i)
                   else np.nan for i in range(len(mid))])
    hi = np.array([np.nanpercentile(z[idx == i], 90) if np.any(idx == i)
                   else np.nan for i in range(len(mid))])
    ax.fill_between(mid, lo, hi, color="#8c6d31", alpha=0.25, lw=0,
                    label="10-90%")
    ax.plot(mid, med, "o-", color="#8c6d31", ms=4, lw=1.4, label="median")
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.invert_yaxis()
    ax.set_xlabel("Moon altitude  [deg]")
    ax.set_ylabel("zenith  [V mag arcsec$^{-2}$]")
    ax.set_title("moonlight budget (Sun $<-18^\\circ$)", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    # (d) the light domes ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    for col, lab, color in (("tucson", "Tucson dome (az 0, alt 15)", "#d62728"),
                            ("nogales", "Nogales dome (az 190, alt 15)",
                             "#ff7f0e")):
        d = tab["zenith"] - tab[col]
        ax.plot(tab.index, d, ".", ms=3.5, color=color,
                label=f"{lab}\nmed {d.median():.2f} mag")
    ax.set_ylabel("zenith $-$ dome  [mag]")
    ax.set_title("light domes above the zenith sky", fontsize=10)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, framealpha=0.9, loc="best")

    fig.suptitle(f"Alcor all-sky sky brightness, {args.year} "
                 f"({len(df):,} frames over {df['night'].nunique()} nights)",
                 fontsize=13, y=0.965)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")

    print("\nheadline numbers (clear nights only)")
    print(f"  clear nights {len(clear)} of {len(tab)} "
          f"({100 * len(clear) / len(tab):.0f}%)")
    print(f"  zenith      median {clear['zenith'].median():.2f}  "
          f"best {clear['zenith'].max():.2f} ({clear['zenith'].idxmax().date()})")
    print(f"  darkest cone median {clear['best'].median():.2f}  "
          f"best {clear['best'].max():.2f} ({clear['best'].idxmax().date()})")
    darkest = tab["zenith"].idxmax()
    print(f"  NB darkest night overall is {tab['zenith'].max():.2f} "
          f"({darkest.date()}), flicker {tab.loc[darkest, 'flicker']:.3f} "
          f"-- overcast, not clear")
    print(f"  Tucson dome {(tab['zenith'] - tab['tucson']).median():.2f} mag, "
          f"Nogales {(tab['zenith'] - tab['nogales']).median():.2f} mag "
          f"brighter than zenith")


if __name__ == "__main__":
    main()
