import numpy as np, os, sys
from astropy.table import Table
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skycam_utils.alcor as a
from skycam_utils.alcor import collect_alcor_photometry

NIGHT, TAG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FS = 1e5
cat = Table.read(os.path.join(os.path.dirname(a.__file__), "data", "bright_star_sloan_named.fits"))
VM = {str(r["NAME"]).strip(): float(r["Vmag"]) for r in cat if r["NAME"] is not None}

def air(al):
    ar = np.radians(al); return 1/(np.sin(ar)+0.50572*(al+6.07995)**-1.6364)
def fit_slope(x, y):
    keep = np.ones(len(x), bool)
    for _ in range(6):
        c = np.polyfit(x[keep], y[keep], 1); r = y-np.polyval(c, x); s = r[keep].std()
        nk = np.abs(r) < 3*s
        if nk.sum() == keep.sum(): break
        keep = nk
    return np.polyfit(x[keep], y[keep], 1)[0]

df = collect_alcor_photometry(NIGHT)
stars = {}
for name, g in df.groupby("name"):
    if name not in VM: continue
    alt = g["altitude"].values
    if alt.max()-alt.min() < 40: continue
    clean = ~g[["sat_r","sat_g","sat_b"]].any(axis=1).values
    if clean.sum() < 200: continue
    stars[name] = dict(V=VM[name], X=air(alt)[clean],
                       flux={ch: g[f"flux_{ch}"].values[clean] for ch in "rgb"},
                       mag={ch: g[f"mag_{ch}"].values[clean] for ch in "rgb"})

ktrue = {ch: float(np.median([fit_slope(s["X"], s["mag"][ch])
                              for s in stars.values() if 4.5 <= s["V"] <= 5.5])) for ch in "rgb"}
print(f"[{TAG}] k_true: " + "  ".join(f"{c}={ktrue[c]:+.3f}" for c in "rgb"))

def model(f, b):                       # absolute magnitude deficit vs measured flux
    return 2.5*np.log10(1.0/np.clip(1.0-b*f/FS, 1e-3, None))

betas = {}
fig, axd = plt.subplots(1, 3, figsize=(15, 4.6))
for i, ch in enumerate("rgb"):
    # iterative faint-anchored zero-points -> absolute deficit
    b = 0.0
    fall = ral = None
    for it in range(4):
        F, R = [], []
        for s in stars.values():
            if not (2.0 <= s["V"] <= 6.5): continue
            f = s["flux"][ch]; m = s["mag"][ch]; X = s["X"]
            order = np.argsort(f)                      # faintest points = most linear
            anc = order[:max(40, len(f)//5)]
            m0 = np.median(m[anc] - ktrue[ch]*X[anc] - model(f[anc], b))
            F.append(f); R.append(m - m0 - ktrue[ch]*X)
        fall = np.concatenate(F); ral = np.concatenate(R)
        # robust binned fit of deficit vs flux
        edges = np.percentile(fall, np.linspace(0, 100, 26))
        bc, br = [], []
        for j in range(len(edges)-1):
            sel = (fall >= edges[j]) & (fall < edges[j+1])
            if sel.sum() >= 30:
                bc.append(np.median(fall[sel])); br.append(np.median(ral[sel]))
        bc, br = np.array(bc), np.array(br)
        fmax = fall.max()
        b, _ = curve_fit(model, bc, br, p0=[0.05], bounds=(0, 0.999*FS/fmax))
        b = float(b[0])
    betas[ch] = b
    # plot binned deficit + fit
    axd[i].scatter(bc/1e3, br, s=25, c=["tab:red","tab:green","tab:blue"][i])
    ff = np.linspace(bc.min(), bc.max(), 100)
    axd[i].plot(ff/1e3, model(ff, b), "k-", label=f"β={b:.4f}")
    axd[i].set_xlabel("measured flux (10³ ADU)"); axd[i].set_ylabel("deficit Δ (mag)")
    axd[i].set_title(f"{ch.upper()}  β={b:.4f}"); axd[i].legend(); axd[i].grid(alpha=.3)
    print(f"[{TAG}] {ch}: beta={b:.4f}  deficit@1e5={model(np.array([1e5]),b)[0]:.3f}  @2e5={model(np.array([2e5]),b)[0]:.3f}")
fig.suptitle(f"[{TAG}] Absolute non-linearity deficit vs flux (faint-anchored)", fontsize=12)
fig.tight_layout(); p=os.path.join(OUT,f"nonlin_deficit2_{TAG}.png"); fig.savefig(p,dpi=110); print("wrote",p)

def corr(ch, mag, flux): return mag - model(flux, betas[ch])

# validation k vs V
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
for i, ch in enumerate("rgb"):
    V=[];kb=[];ka=[]
    for s in stars.values():
        if not (2.0<=s["V"]<=5.5): continue
        V.append(s["V"]); kb.append(fit_slope(s["X"],s["mag"][ch]))
        ka.append(fit_slope(s["X"],corr(ch,s["mag"][ch],s["flux"][ch])))
    V=np.array(V)
    ax[i].scatter(V,kb,s=22,c="0.7",label="measured")
    ax[i].scatter(V,ka,s=22,c=["tab:red","tab:green","tab:blue"][i],alpha=.7,label="corrected")
    ax[i].axhline(ktrue[ch],ls="--",c="k",lw=1,label=f"k_true {ktrue[ch]:+.2f}")
    ax[i].invert_xaxis(); ax[i].set_xlabel("V (brighter →)"); ax[i].set_ylabel(f"k_{ch}")
    ax[i].set_title(f"{ch.upper()} β={betas[ch]:.4f}"); ax[i].legend(fontsize=8)
fig.suptitle(f"[{TAG}] k vs V before/after correction", fontsize=12)
fig.tight_layout(); p=os.path.join(OUT,f"nonlin_kvsV_corrected_{TAG}.png"); fig.savefig(p,dpi=110); print("wrote",p)

print(f"\n[{TAG}] k_before -> k_after (target)")
for n in sorted([n for n,s in stars.items() if s["V"]<=3.0], key=lambda n:stars[n]["V"])[:14]:
    s=stars[n]; line=f"{n:13s} V={s['V']:.2f} "
    for ch in "rgb":
        kb=fit_slope(s["X"],s["mag"][ch]); ka=fit_slope(s["X"],corr(ch,s["mag"][ch],s["flux"][ch]))
        line+=f" {ch}:{kb:+.2f}->{ka:+.2f}"
    print(line)
print(f"\n[{TAG}] BETAS={ {c:round(betas[c],4) for c in 'rgb'} } FS={FS:.0e}")
