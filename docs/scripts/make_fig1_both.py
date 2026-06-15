"""Aperture-scale (and Gaussian) k vs peak instrumental magnitude, binned
medians per band, from the combined --both photometry. Same frames/stars for
both methods, so the only difference between panels is the flux estimator and
hence its instrumental-mag zeropoint. Goal: locate the aperture-scale knee
that corresponds to the Gaussian-scale -11.5 onset."""
import numpy as np, os, pandas as pd
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

OUT = "/tmp/gplots"; os.makedirs(OUT, exist_ok=True)
GAUSS_ONSET = -11.5
NIGHTS = [
    ("2024-09-04", "/Users/tim/MMT/skycam_data/2024-09-04"),
    ("2026-05-18", "/Volumes/Samsung_4TB/skycam/2026-05-18"),
]
cat = Table.read(os.path.join(os.path.dirname(a.__file__), "data", "bright_star_sloan_named.fits"))
VM = {str(r["NAME"]).strip(): float(r["Vmag"]) for r in cat if r["NAME"] is not None}
CHCOL = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4"}

def airmass(alt):
    ar = np.radians(alt); return 1.0/(np.sin(ar)+0.50572*(alt+6.07995)**-1.6364)

def fit_clip(x, y, nsig=3, niter=6):
    keep = np.ones(len(x), bool)
    for _ in range(niter):
        c = np.polyfit(x[keep], y[keep], 1); r = y-np.polyval(c, x); s = r[keep].std()
        nk = np.abs(r) < nsig*s
        if nk.sum() == keep.sum() or nk.sum() < 5: break
        keep = nk
    return np.polyfit(x[keep], y[keep], 1)

def star_table(df, suffix):
    """per star, per band: k (slope mag-vs-airmass) and peak (2nd pctile) instr
    mag for one method (suffix '_ap' or '_gauss')."""
    sat_cols = [f"sat_{c}{suffix}" for c in "rgb"]
    recs = []
    for name, g in df.groupby("name"):
        v = VM.get(name, np.nan)
        if not (np.isfinite(v) and 2.0 <= v <= 5.0): continue
        # need finite mags in all 3 bands for this method
        good = np.ones(len(g), bool)
        for c in "rgb":
            good &= np.isfinite(g[f"mag_{c}{suffix}"].values)
        sat = np.zeros(len(g), bool)
        for sc in sat_cols:
            sat |= g[sc].fillna(False).astype(bool).values
        clean = good & ~sat
        if clean.sum() < 150: continue
        alt = g["altitude"].values[clean]
        if alt.max()-alt.min() < 35: continue
        X = airmass(alt); rec = dict(name=name, V=v)
        for c in "rgb":
            y = g[f"mag_{c}{suffix}"].values[clean]
            rec[f"k_{c}"] = fit_clip(X, y)[0]
            rec[f"peak_{c}"] = np.percentile(y, 2)
        recs.append(rec)
    return pd.DataFrame(recs)

def binned(x, y, edges, nmin=4):
    cx, cy, lo, hi = [], [], [], []
    idx = np.digitize(x, edges)
    for k in range(1, len(edges)):
        m = idx == k
        if m.sum() >= nmin:
            cx.append(0.5*(edges[k-1]+edges[k]))
            cy.append(np.median(y[m]))
            lo.append(np.percentile(y[m], 25)); hi.append(np.percentile(y[m], 75))
    return np.array(cx), np.array(cy), np.array(lo), np.array(hi)

def plateau_and_knee(G, ch):
    """linear-regime plateau = median k for faint half; knee = brightest peak
    where binned median k is still within 0.05 of plateau."""
    x = G[f"peak_{ch}"].values; y = G[f"k_{ch}"].values
    faint = x > np.median(x)
    plat = np.median(y[faint]) if faint.sum() else np.nan
    return plat

DATA = {tag: {"_ap": star_table(collect_alcor_photometry(nd), "_ap"),
              "_gauss": star_table(collect_alcor_photometry(nd), "_gauss")}
        for tag, nd in NIGHTS}

# ---- main figure: 2 nights x 2 methods (aperture, gaussian) ----
fig, axes = plt.subplots(2, 2, figsize=(14.5, 11))
for row, (tag, nd) in enumerate(NIGHTS):
    for col, (suffix, label) in enumerate([("_ap", "aperture"), ("_gauss", "Gaussian")]):
        ax = axes[row, col]; G = DATA[tag][suffix]
        xmin = np.floor(min(G[f"peak_{c}"].min() for c in "rgb"))
        xmax = np.ceil(max(G[f"peak_{c}"].max() for c in "rgb"))
        edges = np.arange(xmin, xmax+0.5, 0.5)
        plats = []
        for c in "rgb":
            x = G[f"peak_{c}"].values; y = G[f"k_{c}"].values
            ax.scatter(x, y, s=10, c=CHCOL[c], alpha=.10, zorder=1)
            cx, cy, lo, hi = binned(x, y, edges)
            ax.fill_between(cx, lo, hi, color=CHCOL[c], alpha=.12, zorder=2)
            ax.plot(cx, cy, "-o", color=CHCOL[c], lw=2.2, ms=5, label=c.upper(), zorder=4)
            plats.append(plateau_and_knee(G, c))
        plat = np.nanmedian(plats)
        ax.axhline(plat, color="0.4", ls=":", lw=1.3, zorder=3)
        ax.text(xmax-0.1, plat+0.01, f"plateau k≈{plat:.2f}", ha="right", va="bottom",
                fontsize=8.5, color="0.3")
        if suffix == "_gauss":
            ax.axvline(GAUSS_ONSET, color="k", ls="--", lw=1.4)
            ax.text(GAUSS_ONSET, -0.27, " −11.5", fontsize=9, color="k", va="bottom")
        ax.set_xlim(edges[0]-0.3, xmax+0.3); ax.set_ylim(-0.3, 0.7)
        ax.invert_xaxis()
        ax.set_title(f"[{tag}]  {label}", fontsize=12); ax.grid(alpha=.25)
        ax.set_xlabel("peak instrumental mag   (brighter →)")
        ax.set_ylabel("extinction k   (mag / airmass)")
        ax.legend(fontsize=10, loc="lower left")
fig.suptitle("Extinction k vs peak instrumental magnitude — aperture (left) vs Gaussian (right), "
             "binned medians per band (IQR shaded).\nSame frames & stars; only the flux estimator "
             "(and its zeropoint) differs. Dotted = linear-regime plateau.", fontsize=12)
fig.tight_layout(rect=(0,0,1,0.94))
fig.savefig(f"{OUT}/fig1_both_instmag.png", dpi=125)
print("wrote fig1_both_instmag.png")

# ---- report: median zeropoint offset gauss->aperture, to place the knee ----
for tag in DATA:
    Aap, Ag = DATA[tag]["_ap"], DATA[tag]["_gauss"]
    m = Aap.merge(Ag, on="name", suffixes=("_ap", "_gauss"))
    print(f"\n[{tag}]  {len(Aap)} ap stars, {len(Ag)} gauss stars, {len(m)} matched")
    offs = []
    for c in "rgb":
        d = m[f"peak_{c}_ap"] - m[f"peak_{c}_gauss"]
        offs.append(d.median())
        print(f"   {c.upper()}: peak_ap range {Aap[f'peak_{c}'].min():.2f}..{Aap[f'peak_{c}'].max():.2f}"
              f"   median(ap-gauss zeropoint offset) = {d.median():+.2f}")
    print(f"   => aperture-scale knee ≈ {GAUSS_ONSET + np.median(offs):+.2f} "
          f"(Gaussian −11.5 shifted by median offset {np.median(offs):+.2f})")
