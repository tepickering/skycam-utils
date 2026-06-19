#############
Reference/API
#############

Auto-generated API documentation for the ``skycam_utils`` modules. Names
imported into a module from third-party packages (and helpers documented on
another module's page) are skipped so that each page lists only its own public
API.

Alcor OMEA 8C
=============

The RGB all-sky camera module: WCS calibration, star photometry, the horizon
mask, and sky-brightness mapping.

.. automodapi:: skycam_utils.alcor
   :no-inheritance-diagram:
   :skip: as_completed
   :skip: files
   :skip: get_body
   :skip: get_sun
   :skip: hstack
   :skip: least_squares
   :skip: lru_cache
   :skip: median_filter
   :skip: sigma_clipped_stats

Stellacam pipeline
==================

.. automodapi:: skycam_utils.pipeline
   :no-inheritance-diagram:
   :skip: get_body
   :skip: get_sun
   :skip: load_mask
   :skip: load_skycam_catalog
   :skip: load_wcs
   :skip: make_background
   :skip: make_catalog
   :skip: make_segmentation_image
   :skip: match_stars
   :skip: solve_field
   :skip: update_altaz

Photometry
==========

.. automodapi:: skycam_utils.photometry
   :no-inheritance-diagram:
   :skip: files
   :skip: hstack
   :skip: unique

Astrometry
==========

.. automodapi:: skycam_utils.astrometry
   :no-inheritance-diagram:
   :skip: files
   :skip: fit_wcs_from_points
   :skip: minimize

WCS fitting
===========

.. automodapi:: skycam_utils.fit_wcs
   :no-inheritance-diagram:
