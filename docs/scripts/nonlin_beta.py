import numpy as np, os, sys
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

NIGHT, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FSCALE = 1e5  # flux scaling so beta is O(1):  m_corr = m + 2.5 log10(1 - b*f/FSCALE)
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
    return np.polyfit(x[keep], y[keep], 1)[0]

df = collect_alcor_photometry(NIGHT)
stars = {}
for name, g in df.groupby("name"):
    if name not in VM: continue
    alt = g["altitude"].values
    if alt.max()-alt.min() < 40: continue
    sat = g[["sat_r","sat_g","sat_b"]].any(axis=1).values
    clean = ~sat
    if clean.sum() < 200: continue
    stars[name] = dict(V=VM[name], X=air(alt)[clean],
                       flux={ch: g[f"flux_{ch}"].values[clean] for ch in "rgb"},
                       mag={ch: g[f"mag_{ch}"].values[clean] for ch in "rgb"})

ktrue = {}
for ch in "rgb":
    ks = [fit_slope(s["X"], s["mag"][ch]) for s in stars.values() if 4.5 <= s["V"] <= 5.5]
    ktrue[ch] = float(np.median(ks))
print(f"[{TAG}] k_true: " + "  ".join(f"{c}={ktrue[c]:+.3f}" for c in "rgb"))

def corr_mag(ch, mag, flux, b):
    arg = 1.0 - b*flux/FSCALE
    arg = np.clip(arg, 1e-3, None)          # guard near-saturation blow-up
    return mag + 2.5*np.log10(arg)

# fit beta per channel: minimize median_star (k_corr - k_true)^2 over affected stars
betas = {}
fitstars = [s for s in stars.values() if 2.0 <= s["V"] <= 5.0]
for ch in "rgb":
    def obj(b):
        d = []
        for s in fitstars:
            f = s["flux"][ch]
            if b*f.max()/FSCALE >= 0.999: return 1e9   # infeasible
            k = fit_slope(s["X"], corr_mag(ch, s["mag"][ch], f, b))
            d.append((k-ktrue[ch])**2)
        return np.median(d)
    bs = np.linspace(0.0, 0.27, 136)
    vals = [obj(b) for b in bs]
    b0 = bs[int(np.argmin(vals))]
    # refine
    bb = np.linspace(max(0,b0-0.004), b0+0.004, 81)
    b0 = bb[int(np.argmin([obj(b) for b in bb]))]
    betas[ch] = b0
    print(f"[{TAG}] {ch}: beta={b0:.4f} (/1e5 ADU)  -> deficit at f=2e5: {2.5*np.log10(1/(1-b0*2.0)):.3f} mag")

# ---- validation: k vs V before/after ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
for i, ch in enumerate("rgb"):
    V=[]; kb=[]; ka=[]
    for s in stars.values():
        if not (2.0 <= s["V"] <= 5.5): continue
        V.append(s["V"]); kb.append(fit_slope(s["X"], s["mag"][ch]))
        ka.append(fit_slope(s["X"], corr_mag(ch, s["mag"][ch], s["flux"][ch], betas[ch])))
    V=np.array(V)
    ax[i].scatter(V, kb, s=22, c="0.7", label="measured")
    ax[i].scatter(V, ka, s=22, c=["tab:red","tab:green","tab:blue"][i], alpha=.7, label="corrected")
    ax[i].axhline(ktrue[ch], ls="--", c="k", lw=1, label=f"k_true {ktrue[ch]:+.2f}")
    ax[i].invert_xaxis(); ax[i].set_xlabel("catalog V (brighter →)"); ax[i].set_ylabel(f"k_{ch}")
    ax[i].set_title(f"{ch.upper()}  (β={betas[ch]:.3f})"); ax[i].legend(fontsize=8)
fig.suptitle(f"[{TAG}] k vs V before/after single-β non-linearity correction", fontsize=12)
fig.tight_layout(); p=os.path.join(OUT,f"nonlin_kvsV_corrected_{TAG}.png"); fig.savefig(p,dpi=110); print("wrote",p)

# bright-star table
print(f"\n[{TAG}] k_before -> k_after  (target in parens)")
for n in sorted([n for n,s in stars.items() if s["V"]<=2.7], key=lambda n: stars[n]["V"])[:12]:
    s=stars[n]; line=f"{n:13s} V={s['V']:.2f} "
    for ch in "rgb":
        kb=fit_slope(s["X"],s["mag"][ch]); ka=fit_slope(s["X"],corr_mag(ch,s["mag"][ch],s["flux"][ch],betas[ch]))
        line+=f" {ch}:{kb:+.2f}->{ka:+.2f}({ktrue[ch]:+.2f})"
    print(line)
print(f"\n[{TAG}] BETAS = "+repr({c:round(betas[c],4) for c in 'rgb'})+f"  FSCALE={FSCALE:.0e}")
