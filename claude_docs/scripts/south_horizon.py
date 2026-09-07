"""Look at the south horizon in the night medians.

The 2026-06-09 keogram has a hard edge near alt 3.6 deg due south that no other
night shows. This crops the same patch of sensor from each night's median and
plots the altitude profile down the keogram column through it, with the packaged
horizon mask's sky boundary marked for reference.
"""
import datetime
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from skycam_utils.alcor import (alcor_calibration, build_alcor_wcs,
                                load_alcor_horizon_mask)

nights = sorted(glob.glob("/private/tmp/alcor_rerun/*_rgbmed.fits"))
nights = [(f.split("/")[-1][:10], f) for f in nights]

cal = alcor_calibration(Time("2026-06-09"))
wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"], rotation=cal["rotation"],
                      radial_coeffs=cal["radial_coeffs"],
                      horizon_radius=cal["horizon_radius"],
                      tangential_coeffs=cal["tangential_coeffs"],
                      axis_tilt=cal["axis_tilt"])
horizon, hdate = load_alcor_horizon_mask(datetime.date(2026, 6, 9))

R0, R1, C0, C1 = 20, 130, 600, 800          # south horizon, around the zenith column
ZCOL = 698
alt = np.array([float(wcs.pixel_to_world_values(ZCOL, r)[1]) for r in range(R0, R1)])

fig = plt.figure(figsize=(17, 4.2 * ((len(nights) + 1) // 2 + 1) / 2 + 4))
gs = fig.add_gridspec(2, len(nights) + 1, height_ratios=[1.25, 1])

for i, (night, path) in enumerate(nights):
    med = fits.getdata(path)
    lum = med[:, R0:R1, C0:C1].sum(axis=0)
    ax = fig.add_subplot(gs[0, i])
    ax.imshow(np.log10(np.clip(lum, 1, None)), origin="lower", cmap="inferno",
              extent=[C0, C1, R0, R1], aspect="auto", vmin=3.8, vmax=4.9)
    ax.contour(np.arange(C0, C1), np.arange(R0, R1),
               horizon[R0:R1, C0:C1].astype(float), levels=[0.5],
               colors="cyan", linewidths=1.2)
    ax.axvline(ZCOL, color="white", ls=":", lw=1)
    ax.set_title(night, fontsize=11)
    if i:
        ax.set_yticklabels([])

ax = fig.add_subplot(gs[1, :])
for night, path in nights:
    med = fits.getdata(path)
    ax.plot(alt, med[1, R0:R1, ZCOL], lw=1.6, label=night)
edge = np.flatnonzero(~horizon[R0:R1, ZCOL])
if edge.size:
    ax.axvline(alt[edge[0]], color="cyan", ls="--", lw=1.2,
               label=f"horizon-mask sky edge ({hdate})")
ax.set_xlabel("altitude along the keogram column, due south (deg)")
ax.set_ylabel("G counts, night median")
ax.set_xlim(alt.min(), 12)
ax.legend(fontsize=9, ncol=3)
ax.grid(alpha=0.25)

fig.suptitle("South horizon in the night medians  (cyan = packaged horizon-mask sky boundary)",
             fontsize=12)
fig.tight_layout()
out = "claude_docs/gplots/south_horizon.png"
fig.savefig(out, dpi=110)
print("wrote", out)
