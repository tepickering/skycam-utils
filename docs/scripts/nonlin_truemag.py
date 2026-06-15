"""Build the bright-star non-linearity law in CATALOG-TRUE magnitude space.

nonlin_linear.py regressed deficit against the MEASURED instrumental mag, but
deficit = measured - true has `measured` on both axes -> errors-in-variables bias
that depends on each night's noise distribution (a likely cause of the unstable
slope). Here we fit deficit against the noise-free TRUE top-of-atmosphere mag:

  true0   = catalog_band - zp_ch - color_ch*(B-V)        [x, noise-free]
  meas0   = mag_inst - k*X                               [measured top-of-atm mag]
  deficit = meas0 - true0                                [y, >=0 when bright]

Fit a hinge  deficit = max(0, s*(true_knee - true0)).  The transfer curve
  meas0 = true0 + max(0, s*(true_knee - true0))
is monotonic in true0, so it inverts to a measured->true correction for
application. Test: is `s` now stable across the two nights?

Also reports per-night luminance-FWHM stats -- if the PSF width differs between
nights the deficit law is genuinely night-dependent (peak = flux/2pi.sigma^2),
not just a fitting artifact. Writes docs/gplots/nonlin_truemag.png.
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

def fit_hinge(x, y, knees):
    """Best hinge deficit = s*(knee - x) for x<knee (else 0), s>0. Scan knees,
    least-squares s on the BRIGHT points (x<knee) at each, and judge each knee by
    the residual of those bright points only -- otherwise the overwhelming faint
    bulk (deficit~0) drags the objective to the trivial s~0 solution. Returns
    (knee, s, rms, n)."""
    best = None
    for kn in knees:
        br = x < kn
        if br.sum() < 50: continue
        s = np.sum((kn-x[br])*y[br])/np.sum((kn-x[br])**2)
        if s <= 0: continue
        res = np.median(np.abs(y[br]-s*(kn-x[br])))   # bright points only
        if best is None or res < best[0]: best = (res, kn, s)
    if best is None: return np.nan, np.nan, np.nan, 0
    _, kn, s = best
    br = x < kn
    rms = (y[br]-s*(kn-x[br])).std()
    return kn, s, rms, int(br.sum())

print("Per-night luminance FWHM (px): is the PSF width stable across nights?")
print(f"{'night':12s} {'p25':>6s} {'p50':>6s} {'p75':>6s}")
data = {}
for tag, nd, epoch in NIGHTS:
    df = collect_alcor_photometry(nd)
    data[tag] = (df, epoch)
    fw = df["fwhm"].values
    fw = fw[np.isfinite(fw) & (fw > 0)]
    print(f"{tag:12s} {np.percentile(fw,25):6.2f} {np.percentile(fw,50):6.2f} {np.percentile(fw,75):6.2f}")

print(f"\n{'night':12s} {'ch':3s} {'knee_true':>9s} {'slope':>7s} {'rms':>6s} {'Npt':>6s}  faint_med")
print("-"*60)
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
knees = np.arange(-12.5, -11.0, 0.1)   # bright-regime knee window
slopes = {}
for rowi, (tag, nd, epoch) in enumerate(NIGHTS):
    df, _ = data[tag]
    zp = alcor_zeropoint(Time(epoch))
    alt = df["altitude"].values
    names = df["name"].values
    X = airmass(alt)
    for col, ch in enumerate("rgb"):
        m = df[f"mag_{ch}_ap"].values
        sat = df[f"sat_{ch}_ap"].fillna(False).astype(bool).values
        true0 = np.array([
            (catband(CATALOG[n], ch) - zp[ch]["zp"] - zp[ch]["color_coeff"]*CATALOG[n]["BV"])
            if (n in CATALOG and np.isfinite(CATALOG[n]["BV"])) else np.nan
            for n in names])
        meas0 = m - K*X
        deficit = meas0 - true0
        good = np.isfinite(m) & ~sat & np.isfinite(true0) & np.isfinite(alt)
        tt, dd = true0[good], deficit[good]
        faint_med = np.median(dd[tt > -10.0]) if (tt > -10.0).sum() else 0.0
        dd = dd - faint_med
        kn, s, rms, n = fit_hinge(tt, dd, knees)
        slopes[(tag, ch)] = s
        print(f"{tag:12s} {ch:3s} {kn:9.2f} {s:7.3f} {rms:6.3f} {n:6d}  {faint_med:+.3f}")

        ax = axes[rowi, col]
        ax.scatter(tt, dd, s=5, c=CHCOL[ch], alpha=.15, zorder=2)
        bins = np.arange(np.floor(tt.min()), -9.0, 0.25)
        bc = 0.5*(bins[:-1]+bins[1:]); bm = np.full(len(bc), np.nan)
        for j in range(len(bc)):
            sel = (tt >= bins[j]) & (tt < bins[j+1])
            if sel.sum() >= 10: bm[j] = np.median(dd[sel])
        ax.plot(bc, bm, "k.-", lw=1.2, ms=6, zorder=4, label="binned median")
        if np.isfinite(s):
            xs = np.linspace(tt.min(), -9.0, 60)
            ax.plot(xs, np.where(xs < kn, s*(kn-xs), 0.0), "-", color="orange",
                    lw=2.2, zorder=5, label=f"hinge knee={kn:.2f} s={s:.3f}")
            ax.axvline(kn, color="0.5", ls=":", lw=1)
        ax.axhline(0, color="0.5", lw=0.7); ax.invert_xaxis()
        ax.set_xlim(tt.min()-0.3, -9.0); ax.set_ylim(-0.4, max(1.5, np.nanmax(bm)+0.3))
        ax.set_title(f"[{tag}] {ch.upper()} deficit vs TRUE mag")
        ax.set_xlabel("catalog-true top-of-atm mag (brighter →)")
        ax.set_ylabel("deficit (mag, measured − true)")
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper left")

print("\nslope stability (2024 -> 2026):")
for ch in "rgb":
    s24, s26 = slopes[("2024-09-04", ch)], slopes[("2026-05-18", ch)]
    print(f"  {ch}: {s24:.3f} -> {s26:.3f}  (ratio {s26/s24:.2f})" if s24 else f"  {ch}: n/a")
fig.suptitle("Non-linearity deficit vs CATALOG-TRUE magnitude (errors-in-variables fix). "
             "Test: is the hinge slope now stable across nights?", fontsize=13)
fig.tight_layout(rect=(0,0,1,0.96))
p = f"{OUT}/nonlin_truemag.png"; fig.savefig(p, dpi=120); print("\nwrote", p)
