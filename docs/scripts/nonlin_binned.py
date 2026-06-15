"""Single signal-level non-linearity slope from 15-minute per-star averages.

The frame-to-frame fan is intra-pixel sensitivity beating against the undersampled
PSF (sub-pixel centroid phase changes the peak response). Over a 15-min window a
star drifts across several pixels, so that jitter averages out, while airmass --
hence signal level -- barely changes, so every frame in the bin sits at one spot on
the deficit curve. We take a robust per-(star, 15-min bin, channel) MEDIAN (also
rejecting the odd blended/bad frame), then pool all bands + both nights and fit one
hinge:  deficit = max(0, s*(knee - mag_inst)).

deficit = (mag_inst - k*X) - true0,   true0 = catalog_band - zp_ch - color_ch*(B-V)
x-axis  = binned raw mag_inst (signal level).

Writes docs/gplots/nonlin_binned.png. Compare against nonlin_combined.py (unbinned).
"""
import numpy as np, os
import pandas as pd
from astropy.table import Table
from astropy.time import Time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import (collect_alcor_photometry, alcor_zeropoint,
                                ALCOR_AIRMASS_TERM, ALCOR_BRIGHT_CUT)

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "gplots"))
os.makedirs(OUT, exist_ok=True)
K = ALCOR_AIRMASS_TERM
BINW = "15min"
MINBIN = 6        # frames required in a bin (must sample several pixels/phases)
FAINT0 = -10.0    # mag_inst fainter than this defines the zeroed linear baseline
NIGHTS = [
    ("2024-09-04", "/Users/tim/MMT/skycam_data/2024-09-04", "2024-09-05"),
    ("2026-05-18", "/Volumes/Samsung_4TB/skycam/2026-05-18", "2026-05-19"),
]
CHCOL = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4"}
NIGHTCOL = {"2024-09-04": "#9467bd", "2026-05-18": "#ff7f0e"}

cat = Table.read(os.path.join(os.path.dirname(a.__file__), "data", "bright_star_sloan_named.fits"))
def _f(x):
    try: return float(x)
    except Exception: return np.nan
CATALOG = {}
for row in cat:
    if row["NAME"] is None: continue
    name = str(row["NAME"]).strip()
    V, BV, VR = _f(row["Vmag"]), _f(row["B-V"]), _f(row["V-R"])
    CATALOG[name] = dict(V=V, BV=BV, R=V-VR, B=V+BV)

def airmass(alt):
    ar = np.radians(alt); return 1.0/(np.sin(ar)+0.50572*(alt+6.07995)**-1.6364)
def catband(c, ch):
    return {"r": c["R"], "g": c["V"], "b": c["B"]}[ch]

def binned_median(x, y, lo, hi, w=0.2, nmin=8):
    bins = np.arange(lo, hi, w); bc = 0.5*(bins[:-1]+bins[1:])
    bm = np.full(len(bc), np.nan); bn = np.zeros(len(bc))
    for j in range(len(bc)):
        sel = (x >= bins[j]) & (x < bins[j+1])
        if sel.sum() >= nmin: bm[j] = np.median(y[sel]); bn[j] = sel.sum()
    ok = np.isfinite(bm)
    return bc[ok], bm[ok], bn[ok]

def fit_hinge_binned(bc, bm, bn, knees):
    best = None
    for kn in knees:
        br = bc < kn
        if br.sum() < 3: continue
        xb, yb, wb = kn-bc[br], bm[br], bn[br]
        s = np.sum(wb*xb*yb)/np.sum(wb*xb*xb)
        if s <= 0: continue
        res = np.sum(wb*np.abs(yb - s*xb))/np.sum(wb)
        if best is None or res < best[0]: best = (res, kn, s)
    return (np.nan, np.nan) if best is None else (best[1], best[2])

# ---- build 15-min per-star binned points, pooled ----
pool_m, pool_d, pool_ch, pool_ni = [], [], [], []
npts_per_bin = []
for tag, nd, epoch in NIGHTS:
    df = collect_alcor_photometry(nd)
    zp = alcor_zeropoint(Time(epoch))
    df = df.copy()
    df["X"] = airmass(df["altitude"].values)
    df["tbin"] = pd.to_datetime(df["OBSTIME"]).dt.floor(BINW)
    for ch in "rgb":
        sub = df[["name", "tbin", "X", f"mag_{ch}_ap", f"sat_{ch}_ap"]].copy()
        sub = sub.rename(columns={f"mag_{ch}_ap": "mag", f"sat_{ch}_ap": "sat"})
        sub = sub[np.isfinite(sub["mag"]) & ~sub["sat"].fillna(False).astype(bool)]
        g = sub.groupby(["name", "tbin"])
        agg = g.agg(mag=("mag", "median"), X=("X", "median"), n=("mag", "size"))
        agg = agg[agg["n"] >= MINBIN].reset_index()
        npts_per_bin.append(agg["n"].values)
        true0 = np.array([
            (catband(CATALOG[n], ch) - zp[ch]["zp"] - zp[ch]["color_coeff"]*CATALOG[n]["BV"])
            if (n in CATALOG and np.isfinite(CATALOG[n]["BV"])) else np.nan
            for n in agg["name"].values])
        mm = agg["mag"].values
        deficit = (mm - K*agg["X"].values) - true0
        ok = np.isfinite(deficit)
        mm, deficit = mm[ok], deficit[ok]
        base = np.median(deficit[mm > FAINT0]) if (mm > FAINT0).sum() else 0.0
        deficit = deficit - base
        pool_m.append(mm); pool_d.append(deficit)
        pool_ch.append(np.full(mm.shape, ch)); pool_ni.append(np.full(mm.shape, tag))
pool_m = np.concatenate(pool_m); pool_d = np.concatenate(pool_d)
pool_ch = np.concatenate(pool_ch); pool_ni = np.concatenate(pool_ni)
npts_per_bin = np.concatenate(npts_per_bin)
print(f"binned points: {len(pool_m)}   median frames/bin: {np.median(npts_per_bin):.0f}"
      f"  (>= {MINBIN} required)")

# Non-linearity can only suppress (deficit >= 0). Significantly negative binned
# points are persistent blends / catalog errors -> clip before fitting.
DEF_FLOOR = -0.35
keep = pool_d > DEF_FLOOR
print(f"clipped {np.sum(~keep)} binned points with deficit < {DEF_FLOOR} (blends/contam)")
pool_m, pool_d = pool_m[keep], pool_d[keep]
pool_ch, pool_ni = pool_ch[keep], pool_ni[keep]

knee = ALCOR_BRIGHT_CUT   # pin at the established per-pixel onset (-11.5)
lo = np.floor(pool_m.min())

def slope_pinned(bc, bm, kn):
    """Unweighted LS slope of the hinge through (kn, 0): each 0.2-mag binned
    median counts equally, so the bright end is not swamped by faint bins."""
    br = bc < kn
    if br.sum() < 2: return np.nan
    x = kn - bc[br]
    return float(np.sum(x*bm[br])/np.sum(x*x))

bc, bm, bn = binned_median(pool_m, pool_d, lo, FAINT0+1.0)
slope = slope_pinned(bc, bm, knee)
br = pool_m < knee
resid = pool_d[br] - slope*(knee-pool_m[br])
pt_rms = resid.std(); mad = np.median(np.abs(resid))
print(f"\nGLOBAL single-slope fit (15-min binned, all bands, both nights), knee pinned {knee}:")
print(f"  slope = {slope:.3f} mag/mag   bright-point rms = {pt_rms:.3f}  MAD = {mad:.3f}"
      f"   Nbright = {int(br.sum())}")
print("  pooled binned-median deficit curve (mag_inst : deficit):")
for c, mdef in zip(bc, bm):
    print(f"    {c:6.2f} : {mdef:+.3f}")

print(f"\nper-BAND slope (knee {knee}):")
for ch in "rgb":
    sel = pool_ch == ch
    b2, m2, _ = binned_median(pool_m[sel], pool_d[sel], lo, FAINT0+1.0)
    print(f"  {ch}: slope = {slope_pinned(b2, m2, knee):.3f}")
print(f"per-NIGHT slope (knee {knee}):")
for tag, _, _ in NIGHTS:
    sel = pool_ni == tag
    b2, m2, _ = binned_median(pool_m[sel], pool_d[sel], lo, FAINT0+1.0)
    print(f"  {tag}: slope = {slope_pinned(b2, m2, knee):.3f}")

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
xs = np.linspace(lo, FAINT0+1.0, 100); hinge = np.where(xs < knee, slope*(knee-xs), 0.0)
ax[0].scatter(pool_m, pool_d, s=8, c="0.6", alpha=.30, zorder=1)
ax[0].plot(bc, bm, "k.-", lw=1.6, ms=7, zorder=4, label="pooled binned median")
ax[0].plot(xs, hinge, "-", color="orange", lw=2.6, zorder=5,
           label=f"single hinge: knee={knee:.2f}, s={slope:.3f}\nrms={pt_rms:.2f}, MAD={mad:.2f}")
ax[0].set_title("15-min binned — all bands + both nights")
ax[0].legend(fontsize=9, loc="upper left")
for ch in "rgb":
    sel = pool_ch == ch
    b2, m2, _ = binned_median(pool_m[sel], pool_d[sel], lo, FAINT0+1.0)
    ax[1].plot(b2, m2, ".-", color=CHCOL[ch], lw=1.4, ms=6, label=ch.upper())
ax[1].plot(xs, hinge, "-", color="orange", lw=2.2, label="global hinge")
ax[1].set_title("Per-band (test: bands agree?)"); ax[1].legend(fontsize=9, loc="upper left")
for tag, _, _ in NIGHTS:
    sel = pool_ni == tag
    b2, m2, _ = binned_median(pool_m[sel], pool_d[sel], lo, FAINT0+1.0)
    ax[2].plot(b2, m2, ".-", color=NIGHTCOL[tag], lw=1.4, ms=6, label=tag)
ax[2].plot(xs, hinge, "-", color="orange", lw=2.2, label="global hinge")
ax[2].set_title("Per-night (test: nights agree?)"); ax[2].legend(fontsize=9, loc="upper left")
for a_ in ax:
    a_.axvline(knee, color="0.5", ls=":", lw=1); a_.axhline(0, color="0.5", lw=0.7)
    a_.set_xlim(lo-0.2, FAINT0+1.0); a_.set_ylim(-0.3, 1.6); a_.invert_xaxis()
    a_.set_xlabel("15-min binned instrumental mag = signal level (brighter →)")
    a_.set_ylabel("deficit (mag, measured − true)"); a_.grid(alpha=.25)
fig.suptitle("Single signal-level non-linearity slope from 15-min per-star averages "
             "(intra-pixel jitter averaged out, airmass ~ fixed within a bin).", fontsize=13)
fig.tight_layout(rect=(0,0,1,0.95))
p = f"{OUT}/nonlin_binned.png"; fig.savefig(p, dpi=120); print("\nwrote", p)
