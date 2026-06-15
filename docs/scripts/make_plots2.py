"""Fix Fig 3 (distinct stars) and add Fig 4 (radius test + undersampling)."""
import numpy as np, os, pandas as pd
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

OUT = "/tmp/gplots"
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
    return np.polyfit(x[keep], y[keep], 1)

df = collect_alcor_photometry("/Volumes/Samsung_4TB/skycam/2026-05-18")
tag = "2026-05-18"

# ---------- Fig 3 (fixed): distinct stars spanning V=2 -> 4 ----------
targets = [2.06, 2.45, 2.9, 3.3, 3.6, 3.9]
picks, used = [], set()
for tv in targets:
    best = None
    for name, g in df.groupby("name"):
        if name in used: continue
        v = VM.get(name, np.nan)
        if not np.isfinite(v) or abs(v-tv) > 0.3: continue
        sat = g[["sat_r","sat_g","sat_b"]].any(axis=1).values
        nclean = (~sat).sum()
        if nclean < 150: continue
        span = g["altitude"].values[~sat].max() - g["altitude"].values[~sat].min()
        if best is None or span > best[2]:
            best = (name, v, span)
    if best:
        picks.append(best); used.add(best[0])

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for ax, (name, v, span) in zip(axes.flat, picks):
    g = df[df["name"] == name]
    sat = g[["sat_r","sat_g","sat_b"]].any(axis=1).values
    X = airmass(g["altitude"].values)
    for ch, col in zip("rgb", ["tab:red","tab:green","tab:blue"]):
        y = g[f"mag_{ch}"].values
        ax.scatter(X[~sat], y[~sat], s=8, c=col, alpha=.35)
        if sat.any(): ax.scatter(X[sat], y[sat], s=10, c=col, marker="x", alpha=.4)
        slope, b = fit_clip(X[~sat], y[~sat])
        xx = np.linspace(X[~sat].min(), X[~sat].max(), 20)
        ax.plot(xx, slope*xx+b, col, lw=2)
    ax.invert_yaxis(); ax.grid(alpha=.25)
    kg = fit_clip(X[~sat], g["mag_g"].values[~sat])[0]
    ax.set_title(f"{name}   V={v:.2f}    k_g={kg:+.3f}", fontsize=10)
    ax.set_xlabel("airmass"); ax.set_ylabel("instr mag (R/G/B)")
fig.suptitle(f"Fig 3 — [{tag}] Gaussian mag vs airmass per star (R/G/B dots; × = saturated, excluded).\n"
             "Bright (top): shallow/flat slope = pre-saturation non-linearity suppression. "
             "Faint (bottom): steeper, well-behaved extinction.", fontsize=11)
fig.tight_layout(rect=(0,0,1,0.94))
fig.savefig(f"{OUT}/fig3_lightcurves.png", dpi=120); print("wrote fig3 (fixed)")

# ---------- Fig 4: radius test + undersampling ----------
# radius-test plateaus (from /tmp/radius_test.py on 2024-09-04 dark subset)
plat = {"aperture": dict(r=0.385, g=0.377, b=0.426),
        "gauss r=4": dict(r=0.421, g=0.279, b=0.419),
        "gauss r=10": dict(r=0.435, g=0.269, b=0.383)}
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))
methods = list(plat); chans = "rgb"
xpos = np.arange(len(chans)); w = 0.26
colors = {"aperture":"0.5", "gauss r=4":"tab:blue", "gauss r=10":"tab:cyan"}
for i, mname in enumerate(methods):
    vals = [plat[mname][c] for c in chans]
    axA.bar(xpos + (i-1)*w, vals, w, label=mname, color=colors[mname], edgecolor="k", lw=.5)
axA.set_xticks(xpos); axA.set_xticklabels([c.upper() for c in chans])
axA.set_ylabel("faint-star (V≥3) plateau  k  (mag/airmass)")
axA.set_title("Faint plateau: aperture vs Gaussian @ r=4 vs r=10\n"
              "(2024-09-04 dark subset) — enlarging the window does NOT\n"
              "restore the plateau ⇒ truncation refuted; G stays low", fontsize=10)
axA.legend(fontsize=9); axA.grid(alpha=.25, axis="y")

# undersampling: distribution of sigma = FWHM/2.355 for faint stars near zenith vs horizon
v = df["name"].map(lambda n: VM.get(n, np.nan)).values
faint = np.isfinite(v) & (v >= 3.0) & (v <= 4.5)
sat = df[["sat_r","sat_g","sat_b"]].any(axis=1).values
X = airmass(df["altitude"].values); sig = df["fwhm"].values / 2.3548
zen = faint & ~sat & np.isfinite(sig) & (X < 1.3)
hor = faint & ~sat & np.isfinite(sig) & (X > 2.3)
axB.hist(sig[zen], bins=np.linspace(0, 2.5, 50), alpha=.6, color="tab:orange",
         label=f"near zenith (X<1.3)  median σ={np.median(sig[zen]):.2f}px", density=True)
axB.hist(sig[hor], bins=np.linspace(0, 2.5, 50), alpha=.6, color="tab:purple",
         label=f"near horizon (X>2.3)  median σ={np.median(sig[hor]):.2f}px", density=True)
axB.axvline(0.85, color="k", ls="--", lw=1.2)
axB.text(0.86, axB.get_ylim()[1]*0.9, "Nyquist σ≈0.85px\n(FWHM≈2px)", fontsize=8, va="top")
axB.set_xlabel("fitted Gaussian σ (px)"); axB.set_ylabel("density")
axB.set_title("Why the Gaussian integral is unstable: the PSF is UNDERSAMPLED.\n"
              "Near zenith σ<1px (below Nyquist) → 2πAσ² is sensitive to\n"
              "centroiding/pixelization; bias shifts as σ grows with airmass", fontsize=10)
axB.legend(fontsize=8.5); axB.grid(alpha=.25)
fig.suptitle(f"Fig 4 — [{tag} + 2024-09-04] The mechanism: not fit-window truncation, but an UNDERSAMPLED PSF",
             fontsize=12)
fig.tight_layout(rect=(0,0,1,0.93))
fig.savefig(f"{OUT}/fig4_radius_undersampling.png", dpi=120); print("wrote fig4")
print("picks:", [p[0] for p in picks])
