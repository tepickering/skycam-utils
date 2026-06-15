"""Compare Gaussian-fit vs aperture extinction: does the wing-driven Gaussian
flux flatten the bright-star k-vs-V suppression caused by CMOS non-linearity?

Aperture baseline = saved ensemble_extinction_<tag>.csv (computed 2026-06-12).
Gaussian = recomputed here from the now-Gaussian *_phot.csv via the SAME
selection and fit as /tmp/ensemble_extinction.py (apples-to-apples)."""
import numpy as np, os, sys, pandas as pd
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

NIGHTS = [
    ("2024-09-04", "/Users/tim/MMT/skycam_data/2024-09-04",
     "/Users/tim/MMT/skycam_data/2024-09-04/ensemble_extinction_2024-09-04.csv"),
    ("2026-05-18", "/Volumes/Samsung_4TB/skycam/2026-05-18",
     "/Volumes/Samsung_4TB/skycam/2026-05-18/ensemble_extinction_2026-05-18.csv"),
]
OUTDIR = "/tmp"

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
    return c[0], c[1]

def gaussian_ensemble(nightdir):
    """Same selection/fit as the aperture ensemble, on Gaussian *_phot.csv."""
    df = collect_alcor_photometry(nightdir)
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
        rec = dict(name=name, V=v, BV=bv, nclean=int(nclean), nsat=int(sat.sum()))
        ok = True
        for ch in "rgb":
            mag = g[f"mag_{ch}"].values
            xc, yc = X[~sat], mag[~sat]
            if len(xc) < 150:
                ok = False; break
            k, _ = fit_clip(xc, yc)
            rec[f"k_{ch}"] = k
        if ok:
            recs.append(rec)
    return pd.DataFrame(recs).sort_values("V").reset_index(drop=True)

summary = []
fig, axes = plt.subplots(len(NIGHTS), 3, figsize=(15, 8.5), sharex=True)
for row, (tag, nightdir, apcsv) in enumerate(NIGHTS):
    ap = pd.read_csv(apcsv)[["name", "V", "BV", "nclean", "nsat", "k_r", "k_g", "k_b"]]
    gs = gaussian_ensemble(nightdir)
    print(f"\n[{tag}] aperture stars={len(ap)}  gaussian stars={len(gs)}")

    m = ap.merge(gs, on="name", suffixes=("_ap", "_gs"))
    print(f"[{tag}] overlapping stars={len(m)}")

    # Faint-plateau (V>=3) median per method = the 'truth' anchor
    for ch in "rgb":
        fa = m[m.V_ap >= 3.0]
        kt_ap = np.median(fa[f"k_{ch}_ap"])
        kt_gs = np.median(fa[f"k_{ch}_gs"])
        # Bright stars: deficit from plateau (positive = suppressed below plateau)
        br = m[m.V_ap < 2.6]
        def_ap = (kt_ap - br[f"k_{ch}_ap"]).mean()
        def_gs = (kt_gs - br[f"k_{ch}_gs"]).mean()
        summary.append(dict(night=tag, ch=ch, nbright=len(br),
                            plateau_ap=kt_ap, plateau_gs=kt_gs,
                            bright_deficit_ap=def_ap, bright_deficit_gs=def_gs,
                            deficit_reduction=def_ap - def_gs))

    # ---- plot k vs V: aperture (open) vs gaussian (filled) ----
    for i, ch in enumerate("rgb"):
        ax = axes[row, i]
        ax.scatter(m.V_ap, m[f"k_{ch}_ap"], facecolors="none", edgecolors="tab:red",
                   s=42, lw=1.1, label="aperture")
        ax.scatter(m.V_ap, m[f"k_{ch}_gs"], c="tab:blue", s=30, label="gaussian")
        for _, r in m.iterrows():
            ax.plot([r.V_ap, r.V_ap], [r[f"k_{ch}_ap"], r[f"k_{ch}_gs"]],
                    color="0.7", lw=0.6, zorder=0)
        fa = m[m.V_ap >= 3.0]
        ax.axhline(np.median(fa[f"k_{ch}_ap"]), color="tab:red", ls=":", lw=1)
        ax.axhline(np.median(fa[f"k_{ch}_gs"]), color="tab:blue", ls=":", lw=1)
        ax.invert_xaxis()
        ax.set_title(f"[{tag}] {ch.upper()} channel")
        if i == 0:
            ax.set_ylabel("k (mag/airmass)")
        if row == len(NIGHTS) - 1:
            ax.set_xlabel("catalog V (brighter →)")
        if row == 0 and i == 0:
            ax.legend(fontsize=8, loc="lower left")

    # per-star bright-end table
    print(f"[{tag}] bright stars (V<2.6): aperture k -> gaussian k")
    br = m[m.V_ap < 2.6].sort_values("V_ap")
    for _, r in br.iterrows():
        print(f"   {r['name']:<12} V={r.V_ap:.2f}  "
              f"r {r.k_r_ap:+.3f}->{r.k_r_gs:+.3f}   "
              f"g {r.k_g_ap:+.3f}->{r.k_g_gs:+.3f}   "
              f"b {r.k_b_ap:+.3f}->{r.k_b_gs:+.3f}")

fig.suptitle("Bright-star extinction k vs V: aperture (open red) vs Gaussian (filled blue)\n"
             "dotted = faint-star (V>=3) plateau per method; lines connect same star",
             fontsize=12)
out = os.path.join(OUTDIR, "gaussian_vs_aperture_k_vs_V.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
print("\nwrote", out)

S = pd.DataFrame(summary)
print("\n=== SUMMARY: bright-star (V<2.6) deficit below faint plateau ===")
print("(deficit = plateau_k - bright_k; smaller is better; reduction>0 = gaussian flattens)")
print(S.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
