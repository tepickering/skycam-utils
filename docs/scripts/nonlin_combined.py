"""Single signal-level non-linearity slope, pooling ALL bands and BOTH nights.

The CMOS non-linearity depends only on signal level (ADU): at a given instrumental
magnitude the aperture holds the same counts regardless of Bayer channel, so R, G,
B trace ONE deficit-vs-signal relation. The frame-to-frame fan is intra-pixel
sensitivity variation beating against the undersampled PSF (sub-pixel centroid
phase changes the peak response) -- scatter AROUND the single relation, which
averages out in the binned median.

deficit = (mag_inst - k*X) - true0,   true0 = catalog_band - zp_ch - color_ch*(B-V)
x-axis  = raw mag_inst (the actual signal level in the frame).

Pools (mag_inst, deficit) over {R,G,B} x {2024-09-04, 2026-05-18} with each
(night,channel) faint-baseline-zeroed, fits a single hinge
   deficit = max(0, s*(knee - mag_inst)),
and cross-checks that the per-band and per-night slopes agree within the scatter.
Writes docs/gplots/nonlin_combined.png.
"""
import numpy as np, os
from astropy.table import Table
from astropy.time import Time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import (collect_alcor_photometry, alcor_zeropoint,
                                ALCOR_AIRMASS_TERM)

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "gplots"))
os.makedirs(OUT, exist_ok=True)
K = ALCOR_AIRMASS_TERM
NIGHTS = [
    ("2024-09-04", "/Users/tim/MMT/skycam_data/2024-09-04", "2024-09-05"),
    ("2026-05-18", "/Volumes/Samsung_4TB/skycam/2026-05-18", "2026-05-19"),
]
CHCOL = {"r": "#d62728", "g": "#2ca02c", "b": "#1f77b4"}
NIGHTCOL = {"2024-09-04": "#9467bd", "2026-05-18": "#ff7f0e"}
FAINT0 = -10.0   # mag_inst fainter than this defines the (zeroed) linear baseline

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

def binned_median(x, y, lo, hi, w=0.2, nmin=20):
    bins = np.arange(lo, hi, w); bc = 0.5*(bins[:-1]+bins[1:])
    bm = np.full(len(bc), np.nan); bn = np.zeros(len(bc))
    for j in range(len(bc)):
        sel = (x >= bins[j]) & (x < bins[j+1])
        if sel.sum() >= nmin: bm[j] = np.median(y[sel]); bn[j] = sel.sum()
    ok = np.isfinite(bm)
    return bc[ok], bm[ok], bn[ok]

def fit_hinge_binned(bc, bm, bn, knees):
    """Weighted hinge fit to binned medians: deficit=s*(knee-mag) for mag<knee.
    Weight bright bins by count. Pick knee minimizing weighted |resid|."""
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

# ---- collect pooled points, per (night, channel) faint-zeroed ----
pool_m, pool_d, pool_ch, pool_ni = [], [], [], []
for tag, nd, epoch in NIGHTS:
    df = collect_alcor_photometry(nd)
    zp = alcor_zeropoint(Time(epoch))
    alt = df["altitude"].values; X = airmass(alt); names = df["name"].values
    for ch in "rgb":
        m = df[f"mag_{ch}_ap"].values
        sat = df[f"sat_{ch}_ap"].fillna(False).astype(bool).values
        true0 = np.array([
            (catband(CATALOG[n], ch) - zp[ch]["zp"] - zp[ch]["color_coeff"]*CATALOG[n]["BV"])
            if (n in CATALOG and np.isfinite(CATALOG[n]["BV"])) else np.nan
            for n in names])
        deficit = (m - K*X) - true0
        good = np.isfinite(m) & ~sat & np.isfinite(true0) & np.isfinite(alt)
        mm, dd = m[good], deficit[good]
        base = np.median(dd[mm > FAINT0]) if (mm > FAINT0).sum() else 0.0
        dd = dd - base
        pool_m.append(mm); pool_d.append(dd)
        pool_ch.append(np.full(mm.shape, ch)); pool_ni.append(np.full(mm.shape, tag))
pool_m = np.concatenate(pool_m); pool_d = np.concatenate(pool_d)
pool_ch = np.concatenate(pool_ch); pool_ni = np.concatenate(pool_ni)

lo = np.floor(pool_m.min()); knees = np.arange(-12.5, -11.0, 0.05)

# ---- single global fit on the pooled binned median ----
bc, bm, bn = binned_median(pool_m, pool_d, lo, FAINT0+1.0)
knee, slope = fit_hinge_binned(bc, bm, bn, knees)
br = pool_m < knee
pt_rms = (pool_d[br] - slope*(knee-pool_m[br])).std()
print(f"GLOBAL single-slope fit (all bands, both nights):")
print(f"  knee = {knee:.2f}   slope = {slope:.3f} mag/mag   point rms = {pt_rms:.3f}   Nbright = {br.sum()}")

# ---- per-band and per-night cross-checks ----
print(f"\nper-BAND slope (knee fixed at global {knee:.2f}):")
for ch in "rgb":
    sel = pool_ch == ch
    b2, m2, n2 = binned_median(pool_m[sel], pool_d[sel], lo, FAINT0+1.0)
    _, s = fit_hinge_binned(b2, m2, n2, np.array([knee]))
    print(f"  {ch}: slope = {s:.3f}")
print(f"per-NIGHT slope (knee fixed at global {knee:.2f}):")
for tag, _, _ in NIGHTS:
    sel = pool_ni == tag
    b2, m2, n2 = binned_median(pool_m[sel], pool_d[sel], lo, FAINT0+1.0)
    _, s = fit_hinge_binned(b2, m2, n2, np.array([knee]))
    print(f"  {tag}: slope = {s:.3f}")

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
xs = np.linspace(lo, FAINT0+1.0, 100); hinge = np.where(xs < knee, slope*(knee-xs), 0.0)

ax[0].scatter(pool_m, pool_d, s=3, c="0.7", alpha=.10, zorder=1)
ax[0].plot(bc, bm, "k.-", lw=1.6, ms=7, zorder=4, label="pooled binned median")
ax[0].plot(xs, hinge, "-", color="orange", lw=2.6, zorder=5,
           label=f"single hinge: knee={knee:.2f}, s={slope:.3f}\npoint rms={pt_rms:.2f}")
ax[0].set_title("All bands + both nights pooled")
ax[0].legend(fontsize=9, loc="upper left")

for ch in "rgb":
    sel = pool_ch == ch
    b2, m2, _ = binned_median(pool_m[sel], pool_d[sel], lo, FAINT0+1.0)
    ax[1].plot(b2, m2, ".-", color=CHCOL[ch], lw=1.4, ms=6, label=f"{ch.upper()}")
ax[1].plot(xs, hinge, "-", color="orange", lw=2.2, label="global hinge")
ax[1].set_title("Per-band binned medians (test: bands agree?)")
ax[1].legend(fontsize=9, loc="upper left")

for tag, _, _ in NIGHTS:
    sel = pool_ni == tag
    b2, m2, _ = binned_median(pool_m[sel], pool_d[sel], lo, FAINT0+1.0)
    ax[2].plot(b2, m2, ".-", color=NIGHTCOL[tag], lw=1.4, ms=6, label=tag)
ax[2].plot(xs, hinge, "-", color="orange", lw=2.2, label="global hinge")
ax[2].set_title("Per-night binned medians (test: nights agree?)")
ax[2].legend(fontsize=9, loc="upper left")

for a_ in ax:
    a_.axvline(knee, color="0.5", ls=":", lw=1); a_.axhline(0, color="0.5", lw=0.7)
    a_.set_xlim(lo-0.2, FAINT0+1.0); a_.set_ylim(-0.3, 1.6); a_.invert_xaxis()
    a_.set_xlabel("raw instrumental mag = signal level (brighter →)")
    a_.set_ylabel("deficit (mag, measured − true)"); a_.grid(alpha=.25)
fig.suptitle("Single signal-level non-linearity slope (pooled). "
             "Fan = intra-pixel sensitivity × undersampled PSF (scatter, not a per-band/night law).",
             fontsize=13)
fig.tight_layout(rect=(0,0,1,0.95))
p = f"{OUT}/nonlin_combined.png"; fig.savefig(p, dpi=120); print("\nwrote", p)
