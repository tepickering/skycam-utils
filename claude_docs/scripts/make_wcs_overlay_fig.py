"""Regenerate the WCS catalog-overlay docs figure (docs/images/alcor_wcs_overlay.png)
with a zoomed inset that shows individual stars sitting inside their apertures.

Replicates the aperture overlay from skycam_utils.alcor.save_alcor_photometry_check_plot
(plot_alcor_fits render + cyan annulus / yellow aperture circles) and adds a
zoomed_inset around a bright near-zenith star so the star-vs-aperture alignment
is visible at a glance. Not packaged; run from the repo root.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from skycam_utils.alcor import (
    alcor_star_photometry, plot_alcor_fits, load_alcor_fits,
)

FRAME = "/Volumes/Samsung_4TB/skycam/2026-05-18/2026_05_18__23_05_15.fits.bz2"
OUT = "docs/images/alcor_wcs_overlay.png"
RADIUS = 680
APERTURE_RADIUS = 4.0
ANNULUS_WIDTH = 1.0
INSET_HALF = 32          # half-width (px) of the zoom window
ANNULUS_INNER = APERTURE_RADIUS + 1.0
OUTER = ANNULUS_INNER + ANNULUS_WIDTH


def draw_apertures(target_ax, phot, xl, yl, xu, yu, lw=1.0):
    for _, row in phot.iterrows():
        x, y = float(row["xcen"]), float(row["ycen"])
        if x + OUTER < xl or x - OUTER > xu or y + OUTER < yl or y - OUTER > yu:
            continue
        cx, cy = x - xl, y - yl
        target_ax.add_patch(Circle((cx, cy), OUTER, facecolor="none",
                                   edgecolor="cyan", linewidth=0.7 * lw, alpha=0.35))
        target_ax.add_patch(Circle((cx, cy), ANNULUS_INNER, facecolor="none",
                                   edgecolor="cyan", linewidth=0.6 * lw, alpha=0.25,
                                   linestyle="--"))
        target_ax.add_patch(Circle((cx, cy), APERTURE_RADIUS, facecolor="none",
                                   edgecolor="yellow", linewidth=0.9 * lw, alpha=0.9))


def main():
    phot, _ = alcor_star_photometry(
        FRAME, output_file="/tmp/alcor_overlay_phot.csv",
        vmag_limit=4.0, min_altitude=15,
    )

    fig = plot_alcor_fits(FRAME, outfig=None, radius=RADIUS, powerstretch=0.75,
                          contrast=0.35, gscale=0.7, bscale=1.7, figsize=12)
    ax = fig.axes[0]

    cube, wcs, _ = load_alcor_fits(FRAME)
    zx, zy = wcs.world_to_pixel_values(0.0, 90.0)
    xz, yz = int(round(float(zx))), int(round(float(zy)))
    ny, nx = cube.shape[1:]
    yl, yu = max(0, yz - RADIUS), min(ny, yz + RADIUS)
    xl, xu = max(0, xz - RADIUS), min(nx, xz + RADIUS)

    draw_apertures(ax, phot, xl, yl, xu, yu)

    # pick a bright, high-altitude (near-zenith, round PSF) star comfortably
    # inside the crop to anchor the zoom
    fcol = "flux_g" if "flux_g" in phot.columns else "flux_g_ap"
    cand = phot.copy()
    cand["cx"] = cand["xcen"] - xl
    cand["cy"] = cand["ycen"] - yl
    W = xu - xl
    H = yu - yl
    inside = ((cand["cx"] > 100) & (cand["cx"] < W - 100) &
              (cand["cy"] > 100) & (cand["cy"] < H - 100))
    if "altitude" in cand.columns:
        inside &= cand["altitude"] > 55
    cand = cand[inside].sort_values(fcol, ascending=False)
    star = cand.iloc[0]
    cx0, cy0 = float(star["cx"]), float(star["cy"])

    axins = ax.inset_axes([0.63, 0.63, 0.36, 0.36])
    axins.imshow(ax.images[0].get_array(), origin="lower")
    draw_apertures(axins, phot, xl, yl, xu, yu, lw=2.4)
    axins.set_xlim(cx0 - INSET_HALF, cx0 + INSET_HALF)
    axins.set_ylim(cy0 - INSET_HALF, cy0 + INSET_HALF)
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(1.3)
    mark_inset(ax, axins, loc1=2, loc2=3, fc="none", ec="white", lw=0.8, alpha=0.7)

    fig.savefig(OUT, transparent=True, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT}  (inset on star at cropped px ({cx0:.0f}, {cy0:.0f}), "
          f"altitude {float(star.get('altitude', float('nan'))):.1f} deg)")


if __name__ == "__main__":
    main()
