"""Visualizations of the Gaussian-vs-aperture photometry comparison.
Fig 1: k vs V (aperture vs gaussian)   Fig 2: FWHM vs airmass (the confound)
Fig 3: example mag-vs-airmass light curves from the Gaussian full-night data."""
import numpy as np, os, pandas as pd
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

OUT = "/tmp/gplots"; os.makedirs(OUT, exist_ok=True)
NIGHTS = [
    ("2024-09-04", "/Users/tim/MMT/skycam_data/2024-09-04",
     "/Users/tim/MMT/skycam_data/2024-09-04/ensemble_extinction_2024-09-04.csv"),
    ("2026-05-18", "/Volumes/Samsung_4TB/skycam/2026-05-18",
     "/Volumes/Samsung_4TB/skycam/2026-05-18/ensemble_extinction_2026-05-18.csv"),
]
cat = Table.read(os.path.join(os.path.dirname(a.__file__), "data", "bright_star_sloan_named.fits"))
VM = {str(r["NAME"]).strip(): float(r["Vmag"]) for r in cat if r["NAME"] is not None}

def airmass(alt):
    ar = np.radians(alt); return 1.0/(np.sin(ar)+0.50572*(alt+6.07995)**-1.6364)

def fit_clip(x, y, nsig=3, niter=6):
    keep = np.ones(len(x), bool)
    for _ in range(niter):
        c = np.polyfit(x[keep], y[keep], 1); r = y-np.polyval(c, x); s = r[keep].std()
        nk = np.abs(r) < nsig*s
        if nk.sum() == keep.sum(): break
        keep = nk
    return np.polyfit(x[keep], y[keep], 1), keep

# load full-night gaussian once per night
DATA = {tag: collect_alcor_photometry(nd) for tag, nd, _ in NIGHTS}

def gauss_k_table(df):
    recs = []
    for name, g in df.groupby("name"):
        v = VM.get(name, np.nan)
        if not (np.isfinite(v) and 2.0 <= v <= 4.0): continue
        alt = g["altitude"].values; span = alt.max()-alt.min()
        sat = g[["sat_r","sat_g","sat_b"]].any(axis=1).values
        if span < 40 or (~sat).sum() < 150: continue
        X = airmass(alt[~sat]); rec = dict(name=name, V=v)
        ok = True
        for ch in "rgb":
            y = g[f"mag_{ch}"].values[~sat]
            if len(y) < 150: ok=False; break
            rec[f"k_{ch}"] = fit_clip(X, y)[0][0]
        if ok: recs.append(rec)
    return pd.DataFrame(recs)

# ---------- Fig 1: k vs V, aperture vs gaussian ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
for row, (tag, nd, apcsv) in enumerate(NIGHTS):
    ap = pd.read_csv(apcsv)[["name","V","k_r","k_g","k_b"]]
    gs = gauss_k_table(DATA[tag])
    m = ap.merge(gs, on="name", suffixes=("_ap","_gs"))
    for i, ch in enumerate("rgb"):
        ax = axes[row, i]
        for _, r in m.iterrows():
            ax.plot([r.V_ap]*2, [r[f"k_{ch}_ap"], r[f"k_{ch}_gs"]], color="0.8", lw=.6, zorder=0)
        ax.scatter(m.V_ap, m[f"k_{ch}_ap"], facecolors="none", edgecolors="tab:red", s=42, lw=1.1,
                   label="aperture", zorder=3)
        ax.scatter(m.V_ap, m[f"k_{ch}_gs"], c="tab:blue", s=28, label="gaussian", zorder=4)
        fa = m[m.V_ap >= 3.0]
        ax.axhline(np.median(fa[f"k_{ch}_ap"]), color="tab:red", ls=":", lw=1.2)
        ax.axhline(np.median(fa[f"k_{ch}_gs"]), color="tab:blue", ls=":", lw=1.2)
        ax.axvspan(2.0, 2.6, color="0.93", zorder=-1)
        ax.invert_xaxis(); ax.grid(alpha=.25)
        ax.set_title(f"[{tag}] {ch.upper()}")
        if i == 0: ax.set_ylabel("k  (mag / airmass)")
        if row == 1: ax.set_xlabel("catalog V   (brighter →)")
        if row == 0 and i == 0: ax.legend(fontsize=8, loc="lower left")
fig.suptitle("Fig 1 — Extinction k vs brightness: aperture (open red) vs Gaussian (filled blue)\n"
             "dotted = faint (V≥3) plateau per method · grey band = bright stars (V<2.6) · "
             "Gaussian drops bright k further AND lowers the faint plateau", fontsize=12)
fig.tight_layout(rect=(0,0,1,0.95))
fig.savefig(f"{OUT}/fig1_k_vs_V.png", dpi=120); print("wrote fig1")

# ---------- Fig 2: FWHM vs airmass ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for j, (tag, nd, _) in enumerate(NIGHTS):
    df = DATA[tag]
    sat = df[["sat_r","sat_g","sat_b"]].any(axis=1).values
    v = df["name"].map(lambda n: VM.get(n, np.nan)).values
    sel = (~sat) & np.isfinite(df["fwhm"].values) & (v >= 3.0) & (v <= 4.5)
    X = airmass(df["altitude"].values[sel]); F = df["fwhm"].values[sel]
    ax = axes[j]
    hb = ax.hexbin(X, F, gridsize=45, cmap="viridis", mincnt=1, bins="log")
    bins = np.linspace(1, X.max(), 14); idx = np.digitize(X, bins)
    bx = [X[idx==k].mean() for k in range(1, len(bins)) if (idx==k).sum() > 20]
    by = [np.median(F[idx==k]) for k in range(1, len(bins)) if (idx==k).sum() > 20]
    ax.plot(bx, by, "r.-", lw=2, ms=9, label="binned median")
    c = np.polyfit(X, F, 1)
    ax.set_title(f"[{tag}]  FWHM grows {c[0]:+.2f} px/airmass"); ax.grid(alpha=.25)
    ax.set_xlabel("airmass (horizon →)"); ax.legend(fontsize=9, loc="upper left")
    if j == 0: ax.set_ylabel("Gaussian FWHM (px)")
axes[0].figure.colorbar(hb, ax=axes, fraction=.03, pad=.01, label="log N")
fig.suptitle("Fig 2 — The confound: fitted PSF width grows toward the horizon (faint V3–4.5 stars).\n"
             "With fit window = aperture_radius (4 px ≈ 1.5–2σ near horizon), the analytic integral 2πAσ² "
             "is extrapolated from a truncated core → airmass-dependent flux bias.", fontsize=11)
fig.savefig(f"{OUT}/fig2_fwhm_vs_airmass.png", dpi=120, bbox_inches="tight"); print("wrote fig2")

# ---------- Fig 3: example light curves (gaussian full night, 2026) ----------
tag = "2026-05-18"; df = DATA[tag]
# pick stars near target V with widest span
targets = [2.06, 2.23, 2.46, 3.0, 3.5, 3.9]
picks = []
for tv in targets:
    best = None
    for name, g in df.groupby("name"):
        v = VM.get(name, np.nan)
        if not np.isfinite(v) or abs(v-tv) > 0.25: continue
        sat = g[["sat_r","sat_g","sat_b"]].any(axis=1).values
        span = g["altitude"].values[~sat].max() - g["altitude"].values[~sat].min() if (~sat).sum() else 0
        if (~sat).sum() < 150: continue
        if best is None or span > best[2]:
            best = (name, v, span)
    if best: picks.append(best)
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for ax, (name, v, span) in zip(axes.flat, picks):
    g = df[df["name"] == name]
    sat = g[["sat_r","sat_g","sat_b"]].any(axis=1).values
    X = airmass(g["altitude"].values)
    for ch, col in zip("rgb", ["tab:red","tab:green","tab:blue"]):
        y = g[f"mag_{ch}"].values
        ax.scatter(X[~sat], y[~sat], s=8, c=col, alpha=.4)
        if sat.any(): ax.scatter(X[sat], y[sat], s=8, c=col, marker="x", alpha=.3)
        (slope, b), keep = fit_clip(X[~sat], y[~sat])
        xx = np.linspace(X[~sat].min(), X[~sat].max(), 20)
        ax.plot(xx, slope*xx+b, col, lw=2)
    ax.invert_yaxis(); ax.grid(alpha=.25)
    kg = fit_clip(X[~sat], g["mag_g"].values[~sat])[0][0]
    ax.set_title(f"{name}  V={v:.2f}   k_g={kg:+.3f}", fontsize=10)
    ax.set_xlabel("airmass"); ax.set_ylabel("instr mag (R/G/B)")
fig.suptitle(f"Fig 3 — [{tag}] Gaussian mag vs airmass per star (R/G/B; × = saturated, excluded).\n"
             "Bright stars (top row): shallow/negative slope = non-linearity suppression. "
             "Faint stars (bottom): steeper, well-behaved.", fontsize=11)
fig.tight_layout(rect=(0,0,1,0.94))
fig.savefig(f"{OUT}/fig3_lightcurves.png", dpi=120); print("wrote fig3")
print("done")
