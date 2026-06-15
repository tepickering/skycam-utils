"""Fig 1 redo: extinction k vs INSTRUMENTAL magnitude per band (not catalog V).
Signal-dependent non-linearity => plotting each band's k against that band's
peak instrumental mag should collapse R/G/B onto a common knee near -11.5."""
import numpy as np, os, pandas as pd
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

OUT = "/tmp/gplots"; ONSET = -11.5
NIGHTS = [
    ("2024-09-04", "/Users/tim/MMT/skycam_data/2024-09-04",
     "/Users/tim/MMT/skycam_data/2024-09-04/ensemble_extinction_2024-09-04.csv"),
    ("2026-05-18", "/Volumes/Samsung_4TB/skycam/2026-05-18",
     "/Volumes/Samsung_4TB/skycam/2026-05-18/ensemble_extinction_2026-05-18.csv"),
]
cat = Table.read(os.path.join(os.path.dirname(a.__file__), "data", "bright_star_sloan_named.fits"))
VM = {str(r["NAME"]).strip(): float(r["Vmag"]) for r in cat if r["NAME"] is not None}
CHCOL = {"r": "tab:red", "g": "tab:green", "b": "tab:blue"}

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
    """per star, per band: gaussian k (slope) and peak (brightest 2%) instr mag."""
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

DATA = {tag: gauss_table(collect_alcor_photometry(nd)) for tag, nd, _ in NIGHTS}

fig, axes = plt.subplots(2, 4, figsize=(19, 9.5))
for row, (tag, nd, apcsv) in enumerate(NIGHTS):
    G = DATA[tag]
    ap = pd.read_csv(apcsv)[["name","k_r","k_g","k_b"]]
    m = G.merge(ap, on="name", suffixes=("_gs","_ap"))
    for i, ch in enumerate("rgb"):
        ax = axes[row, i]
        x = G[f"peak_{ch}"].values
        # gaussian k (filled) vs peak instr mag
        ax.scatter(x, G[f"k_{ch}"].values, c="tab:blue", s=26, label="gaussian k", zorder=3)
        # aperture k (open red), same star's peak instr mag (x-scale = gaussian)
        ax.scatter(m[f"peak_{ch}"].values, m[f"k_{ch}_ap"].values, facecolors="none",
                   edgecolors="tab:red", s=40, lw=1.0, label="aperture k", zorder=4)
        # faint (linear) plateau = median gaussian k for peak fainter than -10.8
        lin = G[G[f"peak_{ch}"] > -10.8]
        if len(lin): ax.axhline(np.median(lin[f"k_{ch}"]), color="0.4", ls=":", lw=1.2)
        ax.axvline(ONSET, color="k", ls="--", lw=1.3)
        ylo, yhi = ax.get_ylim(); ax.axvspan(ax.get_xlim()[0], ONSET, color="0.9", zorder=-1)
        ax.set_title(f"[{tag}]  {ch.upper()} band"); ax.grid(alpha=.25)
        ax.set_xlabel("peak instrumental mag  (brighter ←)")
        if i == 0: ax.set_ylabel("k  (mag / airmass)")
        if row == 0 and i == 0: ax.legend(fontsize=8, loc="lower right")
    # col 3: all bands overlaid (gaussian) -> the collapse test
    ax = axes[row, 3]
    for ch in "rgb":
        ax.scatter(G[f"peak_{ch}"].values, G[f"k_{ch}"].values, c=CHCOL[ch], s=22,
                   alpha=.7, label=f"{ch.upper()}")
    ax.axvline(ONSET, color="k", ls="--", lw=1.3)
    ax.axvspan(ax.get_xlim()[0], ONSET, color="0.9", zorder=-1)
    ax.set_title(f"[{tag}]  all bands overlaid"); ax.grid(alpha=.25)
    ax.set_xlabel("peak instrumental mag  (brighter ←)"); ax.legend(fontsize=9)
fig.suptitle("Fig 1 (redo) — extinction k vs PEAK INSTRUMENTAL magnitude per band  "
             "(filled = Gaussian, open red = aperture).\n"
             "Dashed line / grey band = non-linearity onset at instr mag -11.5; dotted = linear-regime "
             "plateau.  Right column: R/G/B collapse onto a common signal-dependent knee.", fontsize=12)
fig.tight_layout(rect=(0,0,1,0.94))
fig.savefig(f"{OUT}/fig1_redo_instmag.png", dpi=120)
print("wrote fig1 redo")
for tag in DATA:
    G = DATA[tag]
    print(f"\n[{tag}] {len(G)} stars, peak g-mag range {G.peak_g.min():.2f}..{G.peak_g.max():.2f}")
