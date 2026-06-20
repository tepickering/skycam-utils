###############
Bad-pixel Masks
###############

CMOS sensors, even actively cooled ones, have hot and warm
pixels — fixed-location pixels that read high regardless of illumination.
Left in place they masquerade as point sources (which affects star detection
during the :doc:`WCS fit <wcs_calibration>` and
biases :doc:`photometry`) and pollute :doc:`sky_brightness` maps.
``skycam_utils`` ships a date-indexed **bad-pixel mask** for the Alcor OMEA
camera and, by default, repairs the flagged pixels as each frame is loaded.

:func:`~skycam_utils.alcor.load_alcor_badpix_mask` returns ``(mask, date)``,
where ``mask`` is a per-channel ``(3, ny, nx)`` boolean array with ``True`` =
**bad pixel**.

Properties
==========

- **Per-channel** — one plane each for R/G/B (contrast the achromatic
  :doc:`horizon mask <horizon_mask>`). A defect that affects only one channel is
  flagged only in that channel, so the other two are measured normally.
- **A repair mask, by default, not an exclusion mask** — flagged pixels are *replaced* with
  their local median (see below), not merely skipped. The horizon mask, by
  contrast, only selects valid sky and is never used to fill pixels.
- **Date-resolved** — loaded by nearest date (overridable with
  ``$ALCOR_BADPIX_DIR``) from
  ``skycam_utils/data/badpix/alcor_badpix_YYYY-MM-DD.fits.gz``, exactly like the
  WCS-calibration and horizon assets.

How frames consume it
=====================

Repairing is **optional**. :func:`~skycam_utils.alcor.load_alcor_fits` always
resolves the mask and returns it as the third element of ``(cube, wcs, mask)``,
so a caller can either let the loader repair the flagged pixels in place *or*
take the cube untouched and handle the bad pixels downstream — for example, by
folding the mask into the pixel mask passed to a detection or background step.
The ``badpix`` argument selects between these:

- ``"repair"`` (the default) — resolve the nearest-date mask and replace each
  flagged pixel, per channel, with its local 5×5 median (robust to the spike
  itself, so it recovers the underlying sky).
- ``None`` — leave the cube untouched, but still resolve and return the mask for
  the caller to use.
- a path or a ``(3, ny, nx)`` bool array — use that mask explicitly (and repair).

.. code-block:: python

   from skycam_utils.alcor import load_alcor_fits, load_alcor_badpix_mask

   # default badpix="repair": hot pixels are already cleaned in `cube`
   cube, wcs, mask = load_alcor_fits("2026_05_18__04_30_00.fits.bz2")

   # get the unmodified cube but keep the mask to handle downstream yourself
   raw, wcs, mask = load_alcor_fits("2026_05_18__04_30_00.fits.bz2", badpix=None)

   # or resolve a mask directly for a given time
   mask, mask_date = load_alcor_badpix_mask("2026-05-18")

The two consumers that need clean point sources — the WCS fit and fixed-position
:doc:`photometry` — currently load with ``badpix="repair"`` so that hot pixels
are not mistaken for stars; this is a load-bearing default, not a cosmetic one.
Code that would rather carry the mask forward than alter pixels can instead load
with ``badpix=None`` and apply the returned mask itself.

Why the mask is epoch-specific
==============================

This is the sharpest contrast with the rest of the pipeline. The
:doc:`WCS geometry <wcs_calibration>` and the :doc:`photometric zeropoints
<photometry>` are *effectively stationary* — they agree across the 2024-09 and
2026-05 epochs to within the fit uncertainty, so a single calibration suffices.
The bad-pixel population is **not** stable: about half the hot pixels — even the
brightest — turn over between those two epochs. Measured directly, the 2024↔2026
mask overlap is low and **flat across the whole threshold range** (Jaccard ≈ 0.30,
recall ≈ 0.45 from ``z > 25`` all the way to ``z > 250``). That flatness is the
signature of genuine **CMOS aging**: threshold flicker would make the overlap
climb with ``z``, and a global registration shift would collapse the
isolated-pixel overlap toward zero — neither happens.

Pixels near the image centre persist better (~68% over 21 months, within 500 px
of centre) than those in the vignetted outer annulus (~41%), where hot pixels are
~3× denser and mostly fall at or below the horizon anyway.

The consequence is the date-indexed design: masks are **regenerated regularly
and resolved nearest-in-date** to each frame — directly analogous to the
time-indexed ``ALCOR_CALIBRATIONS`` geometry, but stored as files rather than an
in-code table because they are large, but sparse, arrays. The package ships validated baseline
masks for **2024-09-04** and **2026-05-18** so it resolves correctly out of the
box, while an operational daily cron accumulates fresh masks into
``$ALCOR_BADPIX_DIR``.

Building a mask
===============

Unlike the :doc:`horizon mask <horizon_mask>`, the bad-pixel mask has a
**packaged build CLI**, :func:`~skycam_utils.alcor.create_badpix_mask` (entry
point ``create_badpix_mask``), designed to run daily from cron so each night
gets a fresh mask. The build proceeds in three stages:

1. **Dark-frame selection** — keep frames with Sun ``< --sun-alt-max`` (−18°)
   and Moon ``< --moon-alt-max`` (−6°). If fewer than ``--min-frames`` (default
   500) dark frames are available, no mask is written — the median needs enough
   frames to be trail-free.
2. **Night-median stack** (:func:`~skycam_utils.alcor.build_alcor_median_stack`)
   — a per-pixel **median** (not a mean) over the night's dark frames. Because
   stars and trails move between frames while sensor defects do not, the median
   erases the sky and leaves only the static hot pixels; a multi-hour *mean*
   would instead leave smeared star-trail cores that contaminate the detection
   tail. The stack is RAM-bounded: each frame is written to a disk memmap and the
   median is taken in row tiles, so peak memory stays small even for ~1000
   frames.
3. **Hot-pixel detection** (:func:`~skycam_utils.alcor.build_alcor_badpix_mask`)
   — for each channel, a small-kernel median high-pass isolates sharp spikes
   (``resid = img - median_filter(img, ksize)``), and a pixel is flagged where
   its robust z-score ``(resid - median) / (1.4826·MAD)`` exceeds ``--z-thresh``
   (default 25). The separation is stark — the residual's robust sigma is only
   ~5 counts while hot pixels reach residuals of ~25,000 (a ~4000σ dynamic
   range) — so the threshold is not delicate. A second rule keeps real sources
   out of the mask: a spike is treated as a sensor defect only if it fires in
   **at most two of the three channels** — a spike present in all three is a
   genuine broadband source and is excluded from every plane (this removes
   roughly 600–960 real features per epoch).

The mask is dated from the ``YYYY-MM-DD`` in the day-directory name (falling back
to the median dark-frame date) and written gzipped, with header keywords
recording the build (``NSTACK``, ``ZTHRESH``, ``KSIZE``, ``CHRULE``, and the
per-channel ``NBADR``/``NBADG``/``NBADB`` counts).

.. code-block:: bash

   # Build one night's mask into the resolved badpix directory
   create_badpix_mask <skycam_datadir>/2026-05-18

   # Tune the detection and write to an explicit directory
   create_badpix_mask <skycam_datadir>/2026-05-18 \
       --out-dir ./badpix --min-frames 500 --z-thresh 25 --ksize 5

   # Cap the frames used (strided) to bound runtime/scratch; silence progress
   create_badpix_mask <skycam_datadir>/2026-05-18 --max-frames 600 --quiet

The mask builder and loader are exercised by ``test_alcor_badpix.py``.

See :doc:`reference/index` for the full :mod:`skycam_utils.alcor` API.
