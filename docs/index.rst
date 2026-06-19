############
skycam_utils
############

``skycam_utils`` is an MMT-Observatory utility package for analyzing all-sky
camera images. It supports three distinct camera systems whose data formats and
calibration assets differ:

- **Stellacam** — the original narrow-dynamic-range all-sky camera and the main
  target of the Stellacam pipeline (:func:`skycam_utils.pipeline.process_stellacam_dir`).
  Header conventions changed across eras, so the pipeline parses ``UT``/``DATE``
  differently for early (``year < 2013``) and later frames.
- **ASI** (ASI1600) — handled by :func:`skycam_utils.pipeline.process_asi_image`,
  which runs ``astrometry.net`` ``solve-field`` on a central cutout because the
  full all-sky image is not directly solvable.
- **Alcor OMEA 8C** — an RGB CMOS all-sky camera (:mod:`skycam_utils.alcor`). Its
  geometry and photometry pipeline is the most fully developed part of the
  package and the subject of most of this documentation.

The Alcor pipeline is built around a single principle: **the WCS is the one
source of truth for geometry.** A raw frame carries an ARC-projection world
coordinate system that maps every pixel to an (azimuth, altitude) on the sky,
encoding the zenith offset, camera rotation, lens distortion, sensor tilt, and
optical-axis tilt. Everything downstream — star photometry, the horizon mask,
and sky-brightness maps — reads geometry from that WCS rather than re-deriving
it. A central validation result is that the geometry and photometric zeropoints
are **stable across the 2024 and 2026 epochs** (~21 months apart) to within the
fit uncertainty, so a single calibration is effectively stationary.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   wcs_calibration
   photometry
   horizon_mask
   sky_brightness

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index
