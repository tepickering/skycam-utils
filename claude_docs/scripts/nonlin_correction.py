import numpy as np, os, sys
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

NIGHT, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
cat = Table.read(os.path.join(os.path.dirname(a.__file__), "data", "bright_star_sloan_named.fits"))
VM = {str(r["NAME"]).strip(): float(r["Vmag"]) for r in cat if r["NAME"] is not None}

def air(al):
    ar = np.radians(al); return 1/(np.sin(ar)+0.50572*(al+6.07995)**-1.6364)

def fit_slope(x, y, nsig=3, niter=6):
    keep = np.ones(len(x), bool)
    for _ in range(niter):
        c = np.polyfit(x[keep], y[keep], 1); r = y-np.polyval(c, x); s = r[keep].std()
        nk = np.abs(r) < nsig*s
        if nk.sum() == keep.sum(): break
        keep = nk
    return np.polyfit(x[keep], y[keep], 1)

df = collect_alcor_photometry(NIGHT)

# ---- per-star clean photometry, indexed ----
stars = {}
for name, g in df.groupby("name"):
    if name not in VM: continue
    alt = g["altitude"].values
    if alt.max()-alt.min() < 40: continue
    sat = g[["sat_r","sat_g","sat_b"]].any(axis=1).values
    clean = ~sat
    if clean.sum() < 200: continue
    stars[name] = dict(V=VM[name], X=air(alt)[clean],
                       mag={ch: g[f"mag_{ch}"].values[clean] for ch in "rgb"})

# ---- true extinction = faint asymptote (V in 4.5-5.5) ----
ktrue = {}
for ch in "rgb":
    ks = [fit_slope(s["X"], s["mag"][ch])[0] for s in stars.values() if 4.5 <= s["V"] <= 5.5]
    ktrue[ch] = float(np.median(ks))
print(f"[{TAG}] k_true (faint asymptote): " + "  ".join(f"{c}={ktrue[c]:+.3f}" for c in "rgb"))

# ---- iterative deficit curve per channel ----
# Delta(mag_meas) >= 0, measured fainter than linear prediction at high signal.
BINS = np.arange(-15.0, -8.0, 0.25)
BC = 0.5*(BINS[:-1]+BINS[1:])
defcurve = {}   # ch -> (BC, Delta)
for ch in "rgb":
    Delta = np.zeros(len(BC))                       # start: no correction
    use = [s for s in stars.values() if 2.0 <= s["V"] <= 6.5]
    for it in range(5):
        allm, allr = [], []
        for s in use:
            m = s["mag"][ch]; X = s["X"]
            corr = m - np.interp(m, BC, Delta)      # current-best linearized mag
            b = np.median(corr - ktrue[ch]*X)       # robust intercept, slope fixed
            pred = b + ktrue[ch]*X                  # linear (true) prediction
            allm.append(m); allr.append(m - pred)   # raw deficit
        allm = np.concatenate(allm); allr = np.concatenate(allr)
        new = np.full(len(BC), np.nan)
        for j in range(len(BC)):
            sel = (allm >= BINS[j]) & (allm < BINS[j+1])
            if sel.sum() >= 20:
                new[j] = np.median(allr[sel])
        # fill empties, anchor faint end to 0, enforce monotone & non-negative
        ok = np.isfinite(new)
        new = np.interp(BC, BC[ok], new[ok])
        faint0 = np.median(new[BC > -10.0])         # faint baseline ~ 0
        new = new - faint0
        new = np.clip(new, 0, None)
        new = np.maximum.accumulate(new[::-1])[::-1]  # nondecreasing toward bright (left)
        Delta = new
    defcurve[ch] = (BC.copy(), Delta.copy())
    print(f"[{TAG}] {ch}: deficit at mag=-13 -> {np.interp(-13,BC,Delta):.3f},  -12 -> {np.interp(-12,BC,Delta):.3f} mag")

def correct(ch, mag):
    BCx, D = defcurve[ch]
    return mag - np.interp(mag, BCx, D)

# ---- plot deficit curves ----
fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))
for ch, col in [("r","tab:red"),("g","tab:green"),("b","tab:blue")]:
    BCx, D = defcurve[ch]; ax.plot(BCx, D, "o-", color=col, label=f"{ch.upper()}")
ax.set_xlabel("measured instrumental mag (brighter →)"); ax.set_ylabel("Δ deficit (mag, measured − true)")
ax.invert_xaxis(); ax.legend(); ax.set_title(f"[{TAG}] CMOS non-linearity deficit curve (per channel)")
ax.grid(alpha=.3)
fig.tight_layout(); p=os.path.join(OUT,f"nonlin_deficit_{TAG}.png"); fig.savefig(p,dpi=110); print("wrote",p)

# ---- validation: k before/after for bright stars ----
print(f"\n[{TAG}] VALIDATION  k_before -> k_after  (target asymptote in parens)")
bright = sorted([n for n,s in stars.items() if s["V"] <= 3.0], key=lambda n: stars[n]["V"])
hdr = "star            V    " + "  ".join(f"{c.upper()}: bef->aft({ktrue[c]:+.2f})" for c in "rgb")
print(hdr)
for n in bright[:14]:
    s = stars[n]; line = f"{n:14s} {s['V']:.2f}  "
    for ch in "rgb":
        kb = fit_slope(s["X"], s["mag"][ch])[0]
        ka = fit_slope(s["X"], correct(ch, s["mag"][ch]))[0]
        line += f" {kb:+.2f}->{ka:+.2f}  "
    print(line)

# ---- Vega before/after figure ----
if "Vega" in stars:
    s = stars["Vega"]
    fig, axx = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, ch in enumerate("rgb"):
        m = s["mag"][ch]; mc = correct(ch, m); X = s["X"]
        kb = fit_slope(X, m)[0]; ka = fit_slope(X, mc)[0]
        off = np.median(m)
        axx[i].scatter(X, m-off, s=8, c="0.7", label=f"measured k={kb:+.2f}")
        axx[i].scatter(X, mc-off, s=8, c=["tab:red","tab:green","tab:blue"][i], alpha=.5, label=f"corrected k={ka:+.2f}")
        xx=np.linspace(X.min(),X.max(),20)
        axx[i].plot(xx, np.median(mc-off)+ktrue[ch]*(xx-np.median(X)), "k--", lw=1, label=f"true slope {ktrue[ch]:+.2f}")
        axx[i].invert_yaxis(); axx[i].set_xlabel("airmass"); axx[i].set_ylabel(f"mag_{ch} (offset)")
        axx[i].set_title(f"Vega {ch.upper()}"); axx[i].legend(fontsize=8)
    fig.suptitle(f"[{TAG}] Vega before/after non-linearity correction (unsaturated points)", fontsize=12)
    fig.tight_layout(); p=os.path.join(OUT,f"vega_corrected_{TAG}.png"); fig.savefig(p,dpi=110); print("wrote",p)
