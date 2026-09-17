"""
Sky-brightness statistics over the clear-dark night sample.

``sb_year_summary.py`` shows the whole archive, clear and cloudy together;
this restricts to the nights graded clear by stellar extinction
(``clear_dark_sample.py``) so the numbers describe the SITE rather than the
weather, and compares the four tracks ``alcor_process_night`` records: the
zenith, the floating darkest cone, and the two fixed low light domes.

Two panels need a word of explanation. The zenith is NOT a darkness measure --
the Milky Way transits through it -- so zenith brightness is plotted against
the galactic latitude of the zenith, which is where most of its spread comes
from. And the tracks are plotted against hours from local midnight because the
light domes and the airglow decay differently through the night, which is the
point of having both.

Usage: clear_sky_stats.py <sample.csv> <sb.parquet> [...] -o OUT.png
"""

import argparse

import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.time import Time

from skycam_utils.astrometry import MMT_LOCATION

SUN_DARK = -18.0
MOON_DOWN = 0.0

TRACKS = (
    ("allsky_mv_best", "darkest cone (floating)", "#222222"),
    ("allsky_mv_zenith", "zenith", "#1f77b4"),
    ("allsky_mv_nogales", "Nogales dome (az 190, alt 15)", "#ff7f0e"),
    ("allsky_mv_tucson", "Tucson dome (az 0, alt 15)", "#d62728"),
)


def zenith_galactic_latitude(times):
    """Galactic latitude of the zenith: RA = LST, Dec = site latitude."""
    t = Time(times.to_numpy(), scale="utc", location=MMT_LOCATION)
    lst = t.sidereal_time("apparent")
    dec = np.full(len(t), MMT_LOCATION.lat.deg) * u.deg
    return SkyCoord(ra=lst, dec=dec, frame="icrs").galactic.b.deg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sample", help="clear_dark_sample.csv")
    p.add_argument("sb", nargs="+", help="cached sky_brightness parquet(s)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--dpi", type=int, default=140)
    args = p.parse_args()

    nights = set(pd.read_csv(args.sample)["night"])
    sb = pd.concat([pd.read_parquet(f) for f in args.sb], ignore_index=True)
    df = sb[sb["night"].isin(nights)
            & (sb["sun_alt"] < SUN_DARK)
            & (sb["moon_alt"] < MOON_DOWN)].copy()
    print(f"{len(nights)} clear nights, {len(df):,} dark moonless frames")

    df["OBSTIME"] = pd.to_datetime(df["OBSTIME"])
    # Local solar midnight at this longitude is ~07:23 UT; use UT hour offset.
    ut = df["OBSTIME"].dt.hour + df["OBSTIME"].dt.minute / 60.0
    df["h_mid"] = ((ut - 7.39 + 12) % 24) - 12
    df["gb"] = zenith_galactic_latitude(df["OBSTIME"])

    fig = plt.figure(figsize=(14.5, 8.8))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22, left=0.07,
                          right=0.98, top=0.90, bottom=0.075)

    # (a) distributions ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    bins = np.arange(19.0, 22.51, 0.025)
    stats = []
    for col, lab, color in TRACKS:
        v = df[col].dropna()
        ax.hist(v, bins=bins, histtype="step", color=color, lw=1.5, label=lab)
        stats.append((lab, np.percentile(v, [10, 50, 90])))
    ax.set_xlabel("sky surface brightness  [V mag arcsec$^{-2}$]")
    ax.set_ylabel("frames")
    ax.set_title("clear-night distributions", fontsize=10.5)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.grid(alpha=0.25)
    txt = "\n".join(f"{lab.split(' (')[0]:<14s} {q[1]:.2f}  "
                    f"[{q[0]:.2f}, {q[2]:.2f}]" for lab, q in stats)
    ax.text(0.985, 0.97, "median [10-90%]\n" + txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=7.4, family="monospace",
            bbox=dict(fc="white", ec="0.7", alpha=0.92))

    # (b) through the night --------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    edges = np.arange(-7, 7.01, 0.5)
    mid = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(df["h_mid"], edges) - 1
    for col, lab, color in TRACKS:
        v = df[col].to_numpy()
        med = np.array([np.nanmedian(v[idx == i]) if np.any(idx == i) else np.nan
                        for i in range(len(mid))])
        ax.plot(mid, med, "-", color=color, lw=1.6, label=lab)
    ax.axvline(0, color="k", ls=":", lw=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("hours from local solar midnight")
    ax.set_ylabel("V mag arcsec$^{-2}$")
    ax.set_title("through the night", fontsize=10.5)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    # (c) season -------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    mon = df["OBSTIME"].dt.month
    months = np.arange(1, 13)
    for col, lab, color in TRACKS:
        med = [df.loc[mon == m, col].median() for m in months]
        ax.plot(months, med, "o-", color=color, lw=1.4, ms=4, label=lab)
    ax.invert_yaxis()
    ax.set_xticks(months)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N",
                        "D"])
    ax.set_xlabel("month")
    ax.set_ylabel("V mag arcsec$^{-2}$")
    ax.set_title("seasonal medians", fontsize=10.5)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    # (d) the zenith is not a darkness measure -------------------------------
    ax = fig.add_subplot(gs[1, 1])
    gedges = np.arange(-90, 90.1, 5.0)
    gmid = 0.5 * (gedges[:-1] + gedges[1:])
    gi = np.digitize(df["gb"], gedges) - 1
    for col, lab, color in (TRACKS[1], TRACKS[0]):
        v = df[col].to_numpy()
        med = np.array([np.nanmedian(v[gi == i]) if np.any(gi == i) else np.nan
                        for i in range(len(gmid))])
        ax.plot(gmid, med, "o-", color=color, lw=1.5, ms=3.5, label=lab)
    ax.axvline(0, color="k", ls=":", lw=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("galactic latitude of the zenith  [deg]")
    ax.set_ylabel("V mag arcsec$^{-2}$")
    ax.set_title("why the zenith is not a darkness measure", fontsize=10.5)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.suptitle("Alcor clear-night sky brightness -- "
                 f"{len(nights)} extinction-graded clear nights, "
                 f"{len(df):,} dark moonless frames", fontsize=13, y=0.963)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"wrote {args.output}")

    print("\n            track   median   10%     90%")
    for lab, q in stats:
        print(f"{lab.split(' (')[0]:>17s}   {q[1]:.2f}   {q[0]:.2f}  {q[2]:.2f}")
    z = df["allsky_mv_zenith"]
    print(f"\nzenith at |b| > 40 deg: {z[np.abs(df['gb']) > 40].median():.2f}   "
          f"|b| < 15 deg: {z[np.abs(df['gb']) < 15].median():.2f}")


if __name__ == "__main__":
    main()
