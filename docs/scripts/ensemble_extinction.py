import numpy as np, os, sys
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

NIGHTDIR = sys.argv[1]
TAG = sys.argv[2]
OUTDIR = sys.argv[3]

cat = Table.read(os.path.join(os.path.dirname(a.__file__), "data", "bright_star_sloan_named.fits"))
BV = {str(r["NAME"]).strip(): float(r["B-V"]) for r in cat if r["NAME"] is not None}
VM = {str(r["NAME"]).strip(): float(r["Vmag"]) for r in cat if r["NAME"] is not None}

def airmass(alt):
    ar = np.radians(alt)
    return 1.0 / (np.sin(ar) + 0.50572 * (alt + 6.07995) ** -1.6364)

def fit_clip(x, y, nsig=3, niter=6):
    keep = np.ones(len(x), bool)
    for _ in range(niter):
        c = np.polyfit(x[keep], y[keep], 1)
        r = y - np.polyval(c, x)
        s = r[keep].std()
        nk = np.abs(r) < nsig * s
        if nk.sum() == keep.sum():
            keep = nk; break
        keep = nk
    c = np.polyfit(x[keep], y[keep], 1)
    rms = (y[keep] - np.polyval(c, x[keep])).std()
    return c[0], c[1], rms, keep.sum()

df = collect_alcor_photometry(NIGHTDIR)
print(f"[{TAG}] collected {len(df)} rows, {df['name'].nunique()} stars")

# Selection: V in [2,4], wide airmass coverage, enough clean points
recs = []
for name, g in df.groupby("name"):
    if name not in VM:
        continue
    v, bv = VM[name], BV[name]
    if not (np.isfinite(v) and np.isfinite(bv) and 2.0 <= v <= 4.0):
        continue
    alt = g["altitude"].values
    span = alt.max() - alt.min()
    sat = g[["sat_r", "sat_g", "sat_b"]].any(axis=1).values
    nclean = (~sat).sum()
    if span < 40 or nclean < 150:
        continue
    X = airmass(alt)
    rec = dict(name=name, V=v, BV=bv, span=span, nclean=nclean, nsat=sat.sum())
    ok = True
    for ch in ("r", "g", "b"):
        mag = g[f"mag_{ch}"].values
        xc, yc = X[~sat], mag[~sat]
        if len(xc) < 150:
            ok = False; break
        k, m0, rms, n = fit_clip(xc, yc)
        rec[f"k_{ch}"] = k; rec[f"m0_{ch}"] = m0; rec[f"rms_{ch}"] = rms
    if ok:
        recs.append(rec)

import pandas as pd
E = pd.DataFrame(recs).sort_values("V").reset_index(drop=True)
print(f"[{TAG}] ensemble: {len(E)} stars, V {E.V.min():.2f}-{E.V.max():.2f}, B-V {E.BV.min():+.2f}..{E.BV.max():+.2f}")

# Robust "true" extinction from faint half (least affected by non-linearity)
faint = E[E.V >= 3.0]
ktrue = {ch: np.median(faint[f"k_{ch}"]) for ch in "rgb"}
print(f"[{TAG}] faint-star (V>=3) median k: " + "  ".join(f"k_{c}={ktrue[c]:+.3f}" for c in "rgb"))

# ---- Plot 1: k vs V (non-linearity test) ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
for i, ch in enumerate("rgb"):
    sc = ax[i].scatter(E.V, E[f"k_{ch}"], c=E.BV, cmap="coolwarm_r", vmin=-0.2, vmax=1.6, s=45, edgecolor="k", lw=.4)
    ax[i].axhline(ktrue[ch], color="0.5", ls="--", lw=1, label=f"faint median {ktrue[ch]:+.3f}")
    for _, r in E.iterrows():
        ax[i].annotate(r["name"], (r.V, r[f"k_{ch}"]), fontsize=5.5, alpha=.6, xytext=(2, 2), textcoords="offset points")
    ax[i].set_xlabel("catalog V (brighter →)"); ax[i].set_ylabel(f"k_{ch} (mag/airmass)")
    ax[i].set_title(f"{ch.upper()} channel"); ax[i].invert_xaxis(); ax[i].legend(fontsize=8)
cb = fig.colorbar(sc, ax=ax, fraction=.025, pad=.01); cb.set_label("B-V")
fig.suptitle(f"[{TAG}] Extinction coefficient vs brightness  —  CMOS non-linearity drives bright-star k down", fontsize=12)
out1 = os.path.join(OUTDIR, f"ensemble_k_vs_V_{TAG}.png")
fig.savefig(out1, dpi=110, bbox_inches="tight"); print("wrote", out1)

# ---- Plot 2: k vs B-V (color systematics, faint stars only) ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
for i, ch in enumerate("rgb"):
    ax[i].scatter(faint.BV, faint[f"k_{ch}"], s=45, c="tab:blue", edgecolor="k", lw=.4)
    if len(faint) >= 3:
        c = np.polyfit(faint.BV, faint[f"k_{ch}"], 1)
        xx = np.linspace(faint.BV.min(), faint.BV.max(), 20)
        ax[i].plot(xx, np.polyval(c, xx), "r-", lw=1.5, label=f"slope {c[0]:+.3f}/mag")
    for _, r in faint.iterrows():
        ax[i].annotate(r["name"], (r.BV, r[f"k_{ch}"]), fontsize=5.5, alpha=.6, xytext=(2, 2), textcoords="offset points")
    ax[i].set_xlabel("B-V (catalog)"); ax[i].set_ylabel(f"k_{ch} (mag/airmass)")
    ax[i].set_title(f"{ch.upper()} channel"); ax[i].legend(fontsize=8)
fig.suptitle(f"[{TAG}] Color dependence of extinction (faint V>=3 stars only)", fontsize=12)
out2 = os.path.join(OUTDIR, f"ensemble_k_vs_BV_{TAG}.png")
fig.savefig(out2, dpi=110, bbox_inches="tight"); print("wrote", out2)

# ---- Table ----
cols = ["name", "V", "BV", "nclean", "nsat", "k_r", "k_g", "k_b", "rms_g"]
print(E[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
E.to_csv(os.path.join(OUTDIR, f"ensemble_extinction_{TAG}.csv"), index=False)
