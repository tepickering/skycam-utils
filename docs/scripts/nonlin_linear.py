"""Test the hypothesis: the bright-star CMOS magnitude DEFICIT is linear in the
raw instrumental magnitude beyond the -11.5 knee.

deficit(per frame, per channel) = (mag_inst - k*X) - true_m0
  k       = ALCOR_AIRMASS_TERM (0.40)
  X       = Kasten-Young airmass(altitude)
  true_m0 = catalog_band - zp_ch - color_ch*(B-V)   [expected top-of-atm instr mag]

On a clear dark night deficit ~ 0 in the linear regime and grows positive (star
measured too faint) as the star gets bright. Plot/fit deficit vs RAW mag_inst
(the variable ALCOR_BRIGHT_CUT lives in). If the user is right it is a hinge:
deficit = max(0, s*(KNEE - mag_inst)).

Hardcoded local data paths (the two calibration nights, combined *_phot.csv from
alcor_star_photometry --both); writes /tmp/gplots/nonlin_linear.png.
"""
import numpy as np, os
from astropy.table import Table
from astropy.time import Time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import (collect_alcor_photometry, alcor_zeropoint,
                                ALCOR_AIRMASS_TERM, ALCOR_BRIGHT_CUT)

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "gplots"))
os.makedirs(OUT, exist_ok=True)
KNEE = ALCOR_BRIGHT_CUT  # -11.5
K = ALCOR_AIRMASS_TERM   # 0.40
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
    CATALOG[name] = dict(V=V, BV=BV, R=V-VR, B=V+BV)  # catalog Johnson bands

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

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
print(f"{'night':12s} {'ch':3s} {'slope':>7s} {'intcpt':>7s} "
      f"{'knee_fit':>8s} {'rms':>6s} {'Npt':>6s}  faint_med(>knee+1)")
print("-"*78)
for row, (tag, nd, epoch) in enumerate(NIGHTS):
    zp = alcor_zeropoint(Time(epoch))
    df = collect_alcor_photometry(nd)
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
        good = np.isfinite(m) & ~sat & np.isfinite(true0) & np.isfinite(alt)
        mm, dd = m[good], deficit[good]

        # faint-regime baseline (should be ~0): validates zeropoint
        faint = mm > (KNEE + 1.0)
        faint_med = np.median(dd[faint]) if faint.sum() else np.nan
        dd = dd - faint_med  # zero the faint baseline

        # fit the bright side: deficit vs mag_inst for mm < KNEE
        br = mm < KNEE
        slope, intc, rms, n = fit_clip(mm[br], dd[br])
        knee_fit = -intc/slope if slope else np.nan  # where the bright line hits 0

        print(f"{tag:12s} {ch:3s} {slope:7.3f} {intc:7.3f} {knee_fit:8.2f} "
              f"{rms:6.3f} {n:6d}  {faint_med:+.3f}")

        ax = axes[row, col]
        ax.scatter(mm, dd, s=5, c=CHCOL[ch], alpha=.18, zorder=2)
        # binned medians
        bins = np.arange(np.floor(mm.min()), KNEE+2.0, 0.25)
        bc = 0.5*(bins[:-1]+bins[1:]); bm = np.full(len(bc), np.nan)
        for j in range(len(bc)):
            sel = (mm >= bins[j]) & (mm < bins[j+1])
            if sel.sum() >= 10: bm[j] = np.median(dd[sel])
        ax.plot(bc, bm, "k.-", lw=1.2, ms=6, zorder=4, label="binned median")
        # hinge model pinned at KNEE: deficit = slope_pin*(KNEE - mag), slope_pin>0
        # estimate pinned slope from bright points only
        bpin = (mm < KNEE)
        sp = np.sum((KNEE-mm[bpin])*dd[bpin])/np.sum((KNEE-mm[bpin])**2) if bpin.sum() else np.nan
        xs = np.linspace(mm.min(), KNEE+1.0, 50)
        hinge = np.where(xs < KNEE, sp*(KNEE-xs), 0.0)
        ax.plot(xs, hinge, "-", color="orange", lw=2.2, zorder=5,
                label=f"hinge@{KNEE}: s={sp:.3f}")
        ax.axvline(KNEE, color="0.5", ls=":", lw=1)
        ax.axhline(0, color="0.5", ls="-", lw=0.7)
        ax.set_xlim(mm.min()-0.3, KNEE+2.0); ax.set_ylim(-0.4, max(1.5, np.nanmax(bm)+0.3))
        ax.invert_xaxis()
        ax.set_title(f"[{tag}] {ch.upper()} deficit vs instr mag")
        ax.set_xlabel("raw instrumental mag (brighter →)")
        ax.set_ylabel("deficit (mag, measured − true)")
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Bright-star CMOS deficit vs raw instrumental magnitude. "
             "Hypothesis: linear hinge at −11.5 (one slope per channel).", fontsize=13)
fig.tight_layout(rect=(0,0,1,0.96))
p = f"{OUT}/nonlin_linear.png"; fig.savefig(p, dpi=120); print("\nwrote", p)
