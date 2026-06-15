"""Decisive test: is the Gaussian airmass bias caused by the small default
fit window (aperture_radius=4 px)? Pre-select DARK frames spanning the night
(sun altitude from the filename, no FITS load), sample ~55 across that window,
reprocess at aperture_radius=4 and =10, and compare the faint-star (V>=3)
extinction plateau to the aperture baseline (~0.38 r/g for 2024-09-04).
If r=10 restores the plateau, the fit window is the culprit."""
import numpy as np, os, glob, warnings, pandas as pd
warnings.filterwarnings("ignore")
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import AltAz, get_sun
import skycam_utils.alcor as a
from skycam_utils.alcor import alcor_star_photometry, _filename_ut_datetime
from skycam_utils.astrometry import MMT_LOCATION

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
    return np.polyfit(x[keep], y[keep], 1)[0]

NIGHT = "/Users/tim/MMT/skycam_data/2024-09-04"
allframes = sorted(glob.glob(os.path.join(NIGHT, "2024_09_04__*.fits.bz2")))

# vectorized sun altitude from filename time -> keep dark frames
dts = [(f, _filename_ut_datetime(f)) for f in allframes]
dts = [(f, d) for f, d in dts if d is not None]
times = Time([d for _, d in dts])
sun = get_sun(times).transform_to(AltAz(obstime=times, location=MMT_LOCATION)).alt.deg
dark = [f for (f, _), s in zip(dts, np.atleast_1d(sun)) if s < -15.0]
print(f"{len(allframes)} frames, {len(dark)} dark (sun<-15)")
# sample ~55 evenly across the dark window for a wide airmass span
idx = np.linspace(0, len(dark)-1, min(55, len(dark))).round().astype(int)
sample = [dark[i] for i in sorted(set(idx))]
print(f"processing {len(sample)} dark frames at each radius")

plateaus = {}
for radius in (4.0, 10.0):
    outdir = f"/tmp/rtest_r{int(radius)}"
    os.makedirs(outdir, exist_ok=True)
    nproc = 0
    for f in sample:
        out = os.path.join(outdir, os.path.basename(f).replace(".fits.bz2", "_phot.csv"))
        try:
            df = alcor_star_photometry(f, output_file=out, gaussian=True, aperture_radius=radius)
            if df is not None and len(df): nproc += 1
        except Exception as e:
            print("  err", os.path.basename(f), e)
    big = pd.concat([pd.read_csv(c) for c in glob.glob(os.path.join(outdir, "*_phot.csv"))],
                    ignore_index=True)
    ks = {ch: [] for ch in "rgb"}; fwhms = []
    for name, g in big.groupby("name"):
        v = VM.get(name, np.nan)
        if not (3.0 <= v <= 4.0): continue
        sat = g[["sat_r","sat_g","sat_b"]].any(axis=1).values
        gg = g[~sat]
        if len(gg) < 8: continue
        alt = gg["altitude"].values
        if alt.max()-alt.min() < 25: continue
        X = airmass(alt)
        fwhms.append(np.median(gg["fwhm"].values))
        for ch in "rgb":
            ks[ch].append(fit_clip(X, gg[f"mag_{ch}"].values))
    plateaus[radius] = {ch: np.median(ks[ch]) for ch in "rgb"}
    nstars = len(ks["r"])
    print(f"\n=== aperture_radius={radius:g}  ({nproc} frames, {nstars} faint stars, "
          f"median fwhm {np.median(fwhms):.1f}px) ===")
    for ch in "rgb":
        print(f"  faint plateau k_{ch} = {plateaus[radius][ch]:+.3f}")

print("\n=== aperture baseline (saved): k_r=+0.385 k_g=+0.377 k_b=+0.426 ===")
print("Δ(plateau) from r=4 -> r=10:", {ch: round(plateaus[10.0][ch]-plateaus[4.0][ch],3) for ch in "rgb"})
