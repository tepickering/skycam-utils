"""Zeropoint calibration of Alcor instrument R,G,B aperture mags to catalog
Johnson R,V,B, using the adopted recipe:
  - aperture photometry (mag_*_ap)
  - single achromatic airmass term k=0.4 mag/airmass for all three bands
  - per-frame bright cutoff: drop frames where that band's mag < -11.5 (non-linear)
  - drop saturated frames (sat_*_ap)
Model (per star s, band b):  catalog_b(s) = m0_b(s) + ZP_b + c_b*(B-V)
where m0_b(s) = median over clean frames of (mag_b - 0.4*X)  [top-of-atmosphere
instrumental mag]. Regress (catalog_b - m0_b) vs B-V => intercept ZP_b, slope c_b.
Channel->catalog: G->V, R->R(Johnson)=V-(V-R), B->B(Johnson)=V+(B-V).
Tests the prediction: G~V (small color term), R & B color-dependent."""
import numpy as np, os, pandas as pd
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

OUT = "/tmp/gplots"; os.makedirs(OUT, exist_ok=True)
K_AIRMASS = 0.40          # single achromatic extinction term
BRIGHT_CUT = -11.5        # per-frame: keep mag > BRIGHT_CUT
MIN_FRAMES = 20           # clean frames per star for a stable median
NIGHTS = [
    ("2024-09-04", "/Users/tim/MMT/skycam_data/2024-09-04"),
    ("2026-05-18", "/Volumes/Samsung_4TB/skycam/2026-05-18"),
]
# channel -> (catalog band label, function building catalog mag from a catalog row)
cat = Table.read(os.path.join(os.path.dirname(a.__file__), "data", "bright_star_sloan_named.fits"))
def _f(x):
    try: return float(x)
    except Exception: return np.nan
CATALOG = {}   # NAME -> dict(V, B, R, BV)
for row in cat:
    if row["NAME"] is None: continue
    name = str(row["NAME"]).strip()
    V, BV, VR = _f(row["Vmag"]), _f(row["B-V"]), _f(row["V-R"])
    CATALOG[name] = dict(V=V, B=V+BV, R=V-VR, BV=BV)
CHAN = {"r": ("R", "R"), "g": ("V", "G"), "b": ("B", "B")}
CHCOL = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4"}

def airmass(alt):
    ar = np.radians(alt); return 1.0/(np.sin(ar)+0.50572*(alt+6.07995)**-1.6364)

def fit_clip(x, y, nsig=3, niter=6):
    """robust linear fit y = c1*x + c0; returns (slope, intercept, rms, n)."""
    keep = np.isfinite(x) & np.isfinite(y)
    for _ in range(niter):
        c = np.polyfit(x[keep], y[keep], 1); r = y - np.polyval(c, x); s = r[keep].std()
        nk = keep & (np.abs(r) < nsig*s)
        if nk.sum() == keep.sum() or nk.sum() < 8: break
        keep = nk
    c = np.polyfit(x[keep], y[keep], 1)
    rms = (y[keep] - np.polyval(c, x[keep])).std()
    return c[0], c[1], rms, int(keep.sum())

def star_m0(df):
    """per star, per channel: m0 = median(mag_ch_ap - 0.4*X) over clean frames."""
    recs = []
    for name, g in df.groupby("name"):
        c = CATALOG.get(name)
        if c is None or not np.isfinite(c["BV"]): continue
        X = airmass(g["altitude"].values)
        rec = dict(name=name, BV=c["BV"], V=c["V"], B=c["B"], R=c["R"])
        ok = False
        for ch in "rgb":
            mag = g[f"mag_{ch}_ap"].values
            sat = g[f"sat_{ch}_ap"].fillna(False).astype(bool).values
            clean = np.isfinite(mag) & ~sat & (mag > BRIGHT_CUT)
            if clean.sum() >= MIN_FRAMES:
                rec[f"m0_{ch}"] = np.median(mag[clean] - K_AIRMASS*X[clean])
                rec[f"n_{ch}"] = int(clean.sum())
                ok = True
            else:
                rec[f"m0_{ch}"] = np.nan; rec[f"n_{ch}"] = 0
        if ok: recs.append(rec)
    return pd.DataFrame(recs)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
print(f"{'night':12s} {'ch->cat':8s} {'ZP':>7s} {'color c':>9s} {'rms':>6s} {'Nstar':>6s}")
print("-"*52)
summary = {}
for row, (tag, nd) in enumerate(NIGHTS):
    S = star_m0(collect_alcor_photometry(nd))
    for col, ch in enumerate("rgb"):
        catband, chlabel = CHAN[ch]
        x = S["BV"].values
        y = (S[catband] - S[f"m0_{ch}"]).values    # implied ZP + color*(B-V)
        m = np.isfinite(x) & np.isfinite(y)
        slope, zp, rms, n = fit_clip(x[m], y[m])
        summary[(tag, ch)] = (zp, slope, rms, n)
        print(f"{tag:12s} {chlabel}->{catband:5s} {zp:7.3f} {slope:9.3f} {rms:6.3f} {n:6d}")
        ax = axes[row, col]
        ax.scatter(x[m], y[m], s=18, c=CHCOL[ch], alpha=.55, zorder=3)
        xs = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 50)
        ax.plot(xs, slope*xs+zp, "k-", lw=1.8, zorder=4,
                label=f"ZP={zp:.2f}, c={slope:+.2f}\nrms={rms:.3f} ({n} stars)")
        ax.axhline(zp, color="0.5", ls=":", lw=1.0)
        ax.set_title(f"[{tag}]  instr {chlabel} → catalog {catband}", fontsize=12)
        ax.set_xlabel("B−V  (catalog color)")
        ax.set_ylabel(f"catalog {catband} − (m_{chlabel} − 0.4·X)")
        ax.grid(alpha=.25); ax.legend(fontsize=9, loc="best")
fig.suptitle("Zeropoint + color-term calibration (aperture, k=0.40/airmass, bright cut −11.5).\n"
             "Slope = color coefficient vs B−V.  Prediction: G→V flat (small c), "
             "R & B color-dependent (steep c).", fontsize=13)
fig.tight_layout(rect=(0,0,1,0.93))
fig.savefig(f"{OUT}/zeropoint_calib.png", dpi=125)
print("\nwrote zeropoint_calib.png")

# zeropoint-only (no color term) rms, to quantify how much the color term buys
print(f"\n{'night':12s} {'ch':4s} {'rms(ZP only)':>12s} {'rms(ZP+color)':>14s}")
for (tag, ch), (zp, slope, rms, n) in summary.items():
    S = star_m0(collect_alcor_photometry(dict(NIGHTS)[tag]))
    catband, chlabel = CHAN[ch]
    resid = (S[catband] - S[f"m0_{ch}"]).values
    resid = resid[np.isfinite(resid)]
    # robust zeropoint-only rms (3-sig clip on the constant)
    r = resid - np.median(resid)
    keep = np.abs(r) < 3*r.std()
    rms_zp_only = resid[keep].std()
    print(f"{tag:12s} {chlabel:4s} {rms_zp_only:12.3f} {rms:14.3f}")
