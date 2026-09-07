"""Drop the saturation blanking from already-written SB keograms.

_alcor_sky_brightness_map stopped masking raw G >= ALCOR_SB_SATURATION by
default: on a surface-brightness map a NaN renders as the axes background, so
the very brightest sources appeared as DARK holes. The change only ever
*un*-blanks a pixel, so recomputing the affected frames' columns reproduces
exactly what a full re-run would produce, at a fraction of the cost -- typically
~28 frames of ~1100 per night rather than all of them.

sky_brightness.csv is deliberately not touched: the allsky_mv_* cone medians
were verified unchanged to 1e-6 even on a twilight frame carrying 73k saturated
pixels, because a median over ~5000 sky pixels is robust to a handful of bright
ones and the cones sit away from them.

Usage:  python claude_docs/scripts/unmask_keogram_saturation.py <night-dir> ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from skycam_utils.alcor import (_alcor_frame_time, _alcor_sky_brightness_map,
                                _read_frame_exposure, load_alcor_fits,
                                plot_alcor_sb_keogram_fits)

for night_dir in sys.argv[1:]:
    night_dir = Path(night_dir)
    night = night_dir.name
    path = night_dir / f"{night}_sb_keogram.fits"
    summary = pd.read_csv(night_dir / "sky_brightness.csv")

    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)

    # Rows the field cut left live; a NaN inside them is saturation blanking.
    live_rows = np.isfinite(data).any(axis=1)
    holes = (~np.isfinite(data)) & live_rows[:, None]
    frames = sorted(set(np.where(holes)[1].tolist()))
    if not frames:
        print(f"{night}: nothing blanked, keogram already current")
        continue

    fixed = 0
    for index in frames:
        filename = night_dir / summary.filename.iloc[index]
        cube, wcs, _ = load_alcor_fits(filename, badpix="repair")
        zx, _ = wcs.world_to_pixel_values(0.0, 90.0)
        zcol = int(round(float(zx)))
        mu, _ = _alcor_sky_brightness_map(
            cube, wcs, _alcor_frame_time(filename),
            _read_frame_exposure(filename))
        column = mu[:, zcol].astype(np.float32)
        recovered = int((~np.isfinite(data[:, index]) & np.isfinite(column)).sum())
        data[:, index] = column
        fixed += recovered

    with fits.open(path, mode="update") as hdul:
        hdul[0].data = data
        hdul[0].header["SATLEVEL"] = ("none", "saturated pixels kept, not blanked")
        hdul.flush()

    png = plot_alcor_sb_keogram_fits(path)
    print(f"{night}: recomputed {len(frames)} of {data.shape[1]} frames, "
          f"recovered {fixed} pixels; re-rendered {png}")
