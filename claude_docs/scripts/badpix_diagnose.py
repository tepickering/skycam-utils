"""Diagnose what the bad-pixel detector is actually flagging.

Compares two nights' sampled per-channel medians. A genuine hot pixel sits at
the same pixel on both nights; a star trail does not, because the sky has
rotated to a different hour angle. Panel 3 zooms on the north celestial pole,
where Polaris moves so little that the night median keeps its trail.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from scipy.ndimage import median_filter

A, B = "2026-01-11", "2026-06-09"
a = fits.getdata(f"/private/tmp/alcor_rerun/{A}_rgbmed.fits")
b = fits.getdata(f"/private/tmp/alcor_rerun/{B}_rgbmed.fits")


def zmap(med):
    out = np.empty_like(med)
    for c in range(3):
        hp = med[c] - median_filter(med[c], size=5, mode="nearest")
        out[c] = hp / (1.4826 * np.median(np.abs(hp - np.median(hp))))
    return out


za, zb = zmap(a), zmap(b)
ca, cb = (za > 25).any(axis=0), (zb > 25).any(axis=0)
repeat, once = ca & cb, ca & ~cb

fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))

lum = a.sum(axis=0)
lo, hi = ZScaleInterval().get_limits(lum)
axes[0].imshow(lum, origin="lower", cmap="gray", vmin=lo, vmax=hi)
axes[0].set_title(f"{A}  night median (R+G+B, 150 frames)")

axes[1].imshow(lum, origin="lower", cmap="gray", vmin=lo, vmax=hi * 3)
yr, xr = np.where(repeat)
yo, xo = np.where(once)
axes[1].scatter(xo, yo, s=3, c="#ff3b30", lw=0, label=f"{A} only  ({once.sum()})")
axes[1].scatter(xr, yr, s=3, c="#34c759", lw=0, label=f"also on {B}  ({repeat.sum()})")
axes[1].legend(loc="upper right", framealpha=0.85, markerscale=4)
axes[1].set_title("z > 25 candidates: green repeats across nights, red does not")

r0, r1, c0, c1 = 1155, 1200, 675, 720
axes[2].imshow(a[1, r0:r1, c0:c1], origin="lower", cmap="magma",
               extent=[c0, c1, r0, r1])
mask = fits.getdata(f"skycam_utils/data/badpix/alcor_badpix_{A}.fits.gz").astype(bool)
my, mx = np.where(mask[1, r0:r1, c0:c1])
axes[2].scatter(mx + c0, my + r0, s=70, facecolors="none", edgecolors="cyan", lw=1.3,
                label="flagged bad in G")
axes[2].legend(loc="upper right", framealpha=0.85)
axes[2].set_title(f"north celestial pole, G median\nPolaris' nightly trail")

for ax in axes[:2]:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
out = "claude_docs/gplots/badpix_diagnosis.png"
fig.savefig(out, dpi=110)
print("wrote", out)
