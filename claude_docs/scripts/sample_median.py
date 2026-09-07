"""Fast sampled per-channel night median, for bad-pixel diagnostics.

The packaged `alcor_process_night --median-stack` medians every frame of the
night; for finding pixels that are hot in *every* frame a strided sample of
~150 is just as good and takes a minute instead of twenty.
"""
import sys
import numpy as np
import pandas as pd
from astropy.io import fits
from pathlib import Path
from skycam_utils.alcor import load_alcor_fits

night_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
nsamp = int(sys.argv[3]) if len(sys.argv) > 3 else 150

df = pd.read_csv(night_dir / "sky_brightness.csv")
files = [night_dir / n for n in df.filename]
sel = files[:: max(1, len(files) // nsamp)][:nsamp]

stack = np.empty((len(sel), 3, 1411, 1422), dtype=np.uint16)
for i, f in enumerate(sel):
    cube, _, _ = load_alcor_fits(f, badpix=None)
    stack[i] = cube
med = np.median(stack, axis=0).astype(np.float32)
hdu = fits.PrimaryHDU(med)
hdu.header["NSTACK"] = (len(sel), "frames medianed")
hdu.header["NIGHT"] = night_dir.name
hdu.writeto(out, overwrite=True)
print(f"{night_dir.name}: medianed {len(sel)} of {len(files)} night frames -> {out}")
