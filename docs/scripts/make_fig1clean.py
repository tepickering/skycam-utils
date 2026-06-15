"""Fig 1 (clean): extinction k vs peak instrumental magnitude, binned medians
per band, to sharpen the non-linearity knee and show the R/G/B collapse."""
import numpy as np, os, pandas as pd
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

OUT = "/tmp/gplots"; ONSET = -11.5
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
        if nk.sum() == keep.sum(): break
        keep = nk
    return np.polyfit(x[keep], y[keep], 1)

def gauss_table(df):
    recs = []
    for name, g in df.groupby("name"):
        v = VM.get(name, np.nan)
        if not (np.isfinite(v) and 2.0 <= v <= 5.0): continue
        sat = g[["sat_r","sat_g","sat_b"]].any(axis=1).values
        clean = ~sat
        if clean.sum() < 150: continue
        alt = g["altitude"].values[clean]
        if alt.max()-alt.min() < 35: continue
        X = airmass(alt); rec = dict(name=name, V=v)
        for ch in "rgb":
            y = g[f"mag_{ch}"].values[clean]
            rec[f"k_{ch}"] = fit_clip(X, y)[0]
            rec[f"peak_{ch}"] = np.percentile(y, 2)
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

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
for ax, (tag, nd) in zip(axes, NIGHTS):
    G = gauss_table(collect_alcor_photometry(nd))
    xmin = np.floor(min(G[f"peak_{c}"].min() for c in "rgb"))
    edges = np.arange(xmin, -9.6, 0.5)
    for ch in "rgb":
        x = G[f"peak_{ch}"].values; y = G[f"k_{ch}"].values
        ax.scatter(x, y, s=10, c=CHCOL[ch], alpha=.10, zorder=1)
        cx, cy, lo, hi = binned(x, y, edges)
        ax.fill_between(cx, lo, hi, color=CHCOL[ch], alpha=.12, zorder=2)
        ax.plot(cx, cy, "-o", color=CHCOL[ch], lw=2.2, ms=5, label=f"{ch.upper()}", zorder=4)
    ax.axvline(ONSET, color="k", ls="--", lw=1.4)
    ax.axvspan(edges[0]-0.3, ONSET, color="0.9", zorder=0)
    ax.text(ONSET-0.06, 0.66, "non-linear\n(brighter)", ha="right", va="top", fontsize=8.5, color="0.3")
    ax.text(ONSET+0.06, 0.66, "linear", ha="left", va="top", fontsize=8.5, color="0.3")
    ax.text(ONSET, -0.27, " onset −11.5", fontsize=9, color="k", va="bottom")
    ax.set_xlim(edges[0]-0.3, -9.7); ax.set_ylim(-0.3, 0.7)
    ax.set_title(f"[{tag}]", fontsize=12); ax.grid(alpha=.25)
    ax.set_xlabel("peak instrumental mag   (brighter ←)")
    ax.set_ylabel("extinction k   (mag / airmass)"); ax.legend(fontsize=10, loc="lower right")
fig.suptitle("Extinction k vs peak instrumental magnitude — binned medians per band (IQR shaded).\n"
             "All three bands collapse onto a common knee: k holds a plateau until ~−11.5, then rolls "
             "off as signal-dependent non-linearity sets in.", fontsize=12)
fig.tight_layout(rect=(0,0,1,0.93))
fig.savefig(f"{OUT}/fig1_clean_instmag.png", dpi=125)
print("wrote fig1 clean")
