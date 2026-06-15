"""Does the bright-star deficit collapse onto a single, night-stable curve when
plotted against estimated PEAK magnitude instead of total (aperture) magnitude?

CMOS non-linearity is driven by per-pixel counts, i.e. the PSF peak, not the
integrated flux. For a Gaussian, peak = flux/(2*pi*sigma^2), sigma=fwhm/2.355, so
   peak_mag = mag_inst + 2.5*log10(2*pi*sigma^2)   (additive const just shifts knee)
We have the shared luminance `fwhm` (these are --both CSVs), so we can form
peak_mag per frame and re-test deficit-vs-x for collapse + night stability.

Compares against nonlin_linear.py (deficit vs total mag). Writes
/tmp/gplots/nonlin_peak.png.
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
def fit_clip(x, y, nsig=3, niter=6):
    keep = np.isfinite(x) & np.isfinite(y)
    for _ in range(niter):
        cc = np.polyfit(x[keep], y[keep], 1); r = y-np.polyval(cc, x); s = r[keep].std()
        nk = keep & (np.abs(r) < nsig*s)
        if nk.sum() == keep.sum() or nk.sum() < 8: break
        keep = nk
    cc = np.polyfit(x[keep], y[keep], 1)
    return cc[0], cc[1], (y[keep]-np.polyval(cc, x[keep])).std(), int(keep.sum())

# common peak-mag knee (shift of -11.5 by the median 2.5log10(2pi sigma^2) term);
# determined empirically below from the faint baseline, so just probe near it.
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
print(f"{'night':12s} {'ch':3s} {'pk_knee':>7s} {'slope':>7s} {'rms':>6s} {'Npt':>6s}  faint_med")
print("-"*66)
for rowi, (tag, nd, epoch) in enumerate(NIGHTS):
    zp = alcor_zeropoint(Time(epoch))
    df = collect_alcor_photometry(nd)
    fwhm = df["fwhm"].values
    sig = fwhm/2.3548
    peak_term = 2.5*np.log10(2*np.pi*sig**2)   # mag_inst -> peak_mag offset
    for col, ch in enumerate("rgb"):
        m = df[f"mag_{ch}_ap"].values
        sat = df[f"sat_{ch}_ap"].fillna(False).astype(bool).values
        alt = df["altitude"].values
        names = df["name"].values
        true0 = np.array([
            (catband(CATALOG[n], ch) - zp[ch]["zp"] - zp[ch]["color_coeff"]*CATALOG[n]["BV"])
            if (n in CATALOG and np.isfinite(CATALOG[n]["BV"])) else np.nan
            for n in names])
        X = airmass(alt)
        deficit = (m - K*X) - true0
        peak_mag = m + peak_term
        good = (np.isfinite(m) & ~sat & np.isfinite(true0) & np.isfinite(alt)
                & np.isfinite(peak_mag))
        pm, dd = peak_mag[good], deficit[good]

        # faint baseline from the faintest 40% of peak mags
        thr = np.percentile(pm, 60)
        faint_med = np.median(dd[pm > thr])
        dd = dd - faint_med
        # knee in peak-mag space: where binned median departs 0. Use a robust guess:
        # fit a hinge by scanning candidate knees, pick the one minimizing residual.
        cand = np.arange(np.percentile(pm, 5), np.percentile(pm, 70), 0.1)
        best = None
        for kn in cand:
            br = pm < kn
            if br.sum() < 50: continue
            s = np.sum((kn-pm[br])*dd[br])/np.sum((kn-pm[br])**2)
            if s <= 0: continue
            pred = np.where(pm < kn, s*(kn-pm), 0.0)
            res = np.median(np.abs(dd - pred))
            if best is None or res < best[0]: best = (res, kn, s)
        if best is None:
            pk_knee, slope = np.nan, np.nan; rms = np.nan; n = 0
        else:
            _, pk_knee, slope = best
            br = pm < pk_knee
            pred = slope*(pk_knee-pm[br])
            rms = (dd[br]-pred).std(); n = int(br.sum())
        print(f"{tag:12s} {ch:3s} {pk_knee:7.2f} {slope:7.3f} {rms:6.3f} {n:6d}  {faint_med:+.3f}")

        ax = axes[rowi, col]
        ax.scatter(pm, dd, s=5, c=CHCOL[ch], alpha=.15, zorder=2)
        bins = np.arange(np.floor(pm.min()), np.percentile(pm, 80), 0.25)
        bc = 0.5*(bins[:-1]+bins[1:]); bm = np.full(len(bc), np.nan)
        for j in range(len(bc)):
            sel = (pm >= bins[j]) & (pm < bins[j+1])
            if sel.sum() >= 10: bm[j] = np.median(dd[sel])
        ax.plot(bc, bm, "k.-", lw=1.2, ms=6, zorder=4, label="binned median")
        if best is not None:
            xs = np.linspace(pm.min(), pk_knee+1.5, 50)
            ax.plot(xs, np.where(xs < pk_knee, slope*(pk_knee-xs), 0.0), "-",
                    color="orange", lw=2.2, zorder=5,
                    label=f"hinge knee={pk_knee:.2f} s={slope:.3f}")
            ax.axvline(pk_knee, color="0.5", ls=":", lw=1)
        ax.axhline(0, color="0.5", lw=0.7)
        ax.invert_xaxis()
        ax.set_title(f"[{tag}] {ch.upper()} deficit vs PEAK mag")
        ax.set_xlabel("estimated peak mag (brighter →)")
        ax.set_ylabel("deficit (mag)")
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Bright-star deficit vs estimated PEAK magnitude (flux/2pi.sigma^2). "
             "Test: does this collapse the fan + stabilize slope across nights?", fontsize=13)
fig.tight_layout(rect=(0,0,1,0.96))
p = f"{OUT}/nonlin_peak.png"; fig.savefig(p, dpi=120); print("\nwrote", p)
