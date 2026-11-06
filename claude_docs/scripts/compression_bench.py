"""
Is tile compression (.fits.fz) worth adopting for the MMT frames?

The LBTO camera ships `.fits.fz` and decompresses a 61 Mpx mosaic in 0.56 s,
while the MMT's `.fits.bz2` takes ~0.72 s for a cube ten times smaller. That
is a big enough gap to be worth measuring on real MMT frames rather than
reasoning about, because the bz2 decompress is a large share of the ~2.4 s
``alcor_process_night`` spends per frame, and the archive is years deep.

Measures, per frame and per codec: stored size, decompress wall time,
compress wall time, and whether the round trip is BITWISE identical. The last
is not a formality -- RICE on integer data is lossless, but the same tile
machinery applied to floats quantises by default, so a pipeline that ever
writes float products could silently lose precision.

Usage: compression_bench.py <frame.fits.bz2>... [--tile-whole]
"""

import argparse
import io
import os
import time

import numpy as np
from astropy.io import fits

CODECS = ("RICE_1", "GZIP_1", "HCOMPRESS_1")


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames", nargs="+")
    p.add_argument("--tile-whole", action="store_true",
                   help="one tile per channel plane instead of astropy's default")
    p.add_argument("--codecs", nargs="+", default=["RICE_1", "GZIP_1"])
    args = p.parse_args()

    rows = {c: [] for c in args.codecs}
    rows["bz2"] = []
    for path in args.frames:
        (data, hdr), t_bz2 = timed(
            lambda: (fits.getdata(path), fits.getheader(path)))
        data = np.asarray(data)
        size_bz2 = os.path.getsize(path)
        raw = data.nbytes
        rows["bz2"].append((size_bz2, t_bz2, np.nan, True))

        for codec in args.codecs:
            kw = {}
            if args.tile_whole and data.ndim == 3:
                kw["tile_shape"] = (1,) + data.shape[1:]
            if codec == "HCOMPRESS_1":
                kw["hcomp_scale"] = 0            # 0 = lossless
                kw.setdefault("tile_shape", (1,) + data.shape[1:])
            buf = io.BytesIO()

            def _write():
                hdul = fits.HDUList([fits.PrimaryHDU(),
                                     fits.CompImageHDU(data=data, header=hdr,
                                                       compression_type=codec,
                                                       **kw)])
                hdul.writeto(buf, overwrite=True)

            try:
                _, t_w = timed(_write)
            except Exception as err:                      # codec/shape refused
                print(f"  {codec}: not usable here ({type(err).__name__}: {err})")
                continue
            size = buf.tell()
            buf.seek(0)
            back, t_r = timed(lambda: fits.getdata(buf, 1))
            same = bool(np.array_equal(np.asarray(back), data))
            rows[codec].append((size, t_r, t_w, same))

        print(f"{os.path.basename(path)}  {data.shape} {data.dtype}  "
              f"raw {raw / 1e6:.1f} MB")

    print(f"\n{len(args.frames)} frames, raw {data.nbytes / 1e6:.1f} MB each"
          f"{' (one tile per plane)' if args.tile_whole else ''}")
    print(f"{'codec':<13} {'size MB':>8} {'vs bz2':>7} {'read s':>8} "
          f"{'write s':>8}  lossless")
    base = np.mean([r[0] for r in rows["bz2"]])
    for codec in ["bz2"] + args.codecs:
        rs = rows.get(codec)
        if not rs:
            continue
        size = np.mean([r[0] for r in rs])
        print(f"{codec:<13} {size / 1e6:8.2f} {size / base:7.2f} "
              f"{np.mean([r[1] for r in rs]):8.3f} "
              f"{np.mean([r[2] for r in rs]):8.3f}  "
              f"{all(r[3] for r in rs)}")


if __name__ == "__main__":
    main()
