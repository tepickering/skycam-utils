# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

`skycam_utils` is an MMT-Observatory utility package for analyzing all-sky camera images. It supports three distinct camera systems whose data formats and calibration assets differ:

- **Stellacam** (the main pipeline target — `pipeline.py`). Header conventions changed across eras, so `get_ut()` parses `UT`/`DATE` differently for `year < 2013` vs. later years.
- **ASI** (ASI1600). Handled by `process_asi_image()`; runs astrometry.net `solve-field` on a central cutout because the full all-sky image is unsolvable directly.
- **Alcor OMEA 8C** (`alcor/`). RGB FITS. The WCS is the single source of geometry: `load_alcor_fits()` returns the raw `(3, ny, nx)` RGB cube **untouched** (no transpose/trim/rotate/shift/flipud, and no bias subtraction — only optional bad-pixel repair) plus a **raw-frame** ARC-projection WCS, as the 3-tuple `(cube, wcs, mask)` (`mask` is the bad-pixel mask aligned to the cube, or `None`). The WCS maps raw pixel ↔ (azimuth, altitude) with altitude=0 at `horizon_radius` pixels from the zenith, encoding all geometry: the zenith offset in CRPIX, the camera rotation in the PC matrix (a pure rotation, det=+1; the sky/sensor handedness lives in the `rotation - az` azimuth convention, since an all-sky camera images the sky from below — north lands toward +y), and the fitted odd-power radial (k3/k5) lens distortion as an exact analytic SIP, plus Brown–Conrady tangential decentering terms (P1/P2, fit always, exact degree-2 SIP) that absorb the sensor-tilt signature (once-per-azimuth residual growing as r²), plus a world-side optical-axis tilt (`axis_tilt`, fit always, encoded as the WCS pole: CRVAL=(A0, 90−ε), LONPOLE=A0) — with nonzero tilt `xcen`/`ycen` is the optical-axis pixel and the zenith must be located via the WCS (alt=90), not CRPIX. It is calibrated by `fit_alcor_wcs()` against `bright_star_sloan.fits` (Vmag <= 4) over dark-sky frames (Sun < -18 deg, Moon < -6 deg); the camera's `DATE` header is the true UT (`DATE-OBS` is local time despite its label). The fitted geometry — absolute raw-frame constants `xcen`/`ycen`/`rotation`/`radial_coeffs`/`tangential_coeffs`/`axis_tilt`/`horizon_radius` (`tangential_coeffs` and `axis_tilt` are optional, `(0, 0)` when absent; `axis_tilt` is in degrees toward north/east) — lives in a time-indexed `ALCOR_CALIBRATIONS` table; when `wcs=None`, `load_alcor_fits` resolves the epoch nearest a frame's time (filename timestamp, DATE-header fallback) and builds the WCS, so frames from different eras (e.g. 2024 vs 2026) each get the right geometry; pass an explicit `wcs=` to override. **A key cross-epoch result: the camera was not moved or refocused between the 2024-09 and 2026-05 epochs (~21 months apart), and the two independently-fit geometries agree to within the fit uncertainty — center stable to ~1px, rotation ~0.05°, axis tilt ~0.03–0.04°, identical `horizon_radius` (747.2) — matched by the ~0.03 mag photometric-zeropoint stability (see `ALCOR_ZEROPOINTS`, below). So a single calibration is effectively stationary; the separate epochs exist only as a safety net should the camera ever be moved, and this stability is itself a primary validation of the geometry+photometry pipeline (it underpins, e.g., stacking both nights' results in a common frame).** (`fit_alcor_wcs` aggregates matches across a whole night; on clean dark frames it matches ~80 stars/frame; with the full model (k5 + P1/P2 + axis tilt) a healthy fit has a matched fraction near 0.7 and a pooled RMS of ~0.35px — the residual floor is a smooth, azimuthally-symmetric ~0.6px p-p radial wiggle from k3/k5 polynomial truncation, not worth chasing. Two gotchas that silently wreck the fit, both fixed: detection MUST run on the bad-pixel-repaired cube — `_detect_alcor_frame` loads `badpix="repair"` — or hot pixels masquerade as stars; and `_fit_params` must carry `horizon_radius` through its returned dict, or the matcher reverts to the module-default `ALCOR_HORIZON_RADIUS` mid-fit and the pool collapses while k3 runs away.) Visualization routines do their own crop and use `origin="lower"`: `plot_alcor_fits()` builds the display RGB (empirically-tuned per-channel `gscale`/`bscale` + power+ZScale stretch), crops a `radius`-pixel square around the WCS zenith, and renders north-up; `plot_alcor_sky_brightness()` is the surface-brightness sibling — it converts the corner-bias-subtracted **G** channel to **observed** V mag/arcsec² (G→V zeropoint from `ALCOR_ZEROPOINTS`, **no airmass term**, so horizon light domes / airglow / Milky-Way gradients are the signal), first scaling raw counts to the calibration's 20 s reference (`ALCOR_CALIB_EXPTIME`, from the `EXPOSURE` header — counts are linear in exposure) and dividing by a **per-pixel solid angle computed exactly from the WCS** (`_alcor_pixel_solid_angle`: the unit-vector Jacobian via finite differences, which captures the ARC zenith→horizon plate-scale change *and* the SIP distortion — a flat plate scale would be wrong by ~0.5 mag); saturated pixels are **not** masked by default — `ALCOR_SB_SATURATION = 25000` (clipped/strongly non-linear, well below the 15-bit `ALCOR_SATURATION`) is still the right threshold but blanking on it was **actively misleading on a surface-brightness map**: a NaN renders as the axes background, so a saturated star or planet crossing the frame appeared as a *dark hole* when the truth is that it is too bright to measure. Keeping the value is the lesser error, because clipping loses flux and so a saturated pixel reads slightly too *faint* — making the rendered peak a lower bound rather than an inversion. It is rare either way (~37 pixels of 1.4M in a night's keogram, in ~28 frames of 1059, typically 1–2 pixels each: the core clears the threshold while the wings do not). Pass an explicit `saturation=` / `--saturation` / `--sb-saturation` to restore the mask; the SB FITS records `SATLEVEL='none'` when it is off. Note this is a different constant from `ALCOR_BRIGHT_CUT`, which never touches the SB map — that one only NaNs `cal_*`/`ext_*` in star photometry. Non-sky is masked either by an altitude floor (`fov_altitude`, default −2°) or, with `horizon_mask=True`, by `load_alcor_horizon_mask`; it reuses the same zenith crop + alt/az polar grid as `plot_alcor_fits` (both now call shared helpers `_alcor_zenith_crop_bounds`/`_add_alcor_alt_az_grid`), draws a `cividis_r` colorbar (bright sky light), and annotates a sigma-clipped median **zenith** brightness (the cap above alt 85°) in the corner — a clear dark frame lands at ≈21.5–21.6 mag/arcsec², which validates the whole exposure→solid-angle→zeropoint chain end-to-end with no free parameters; `alcor_proc_fits()` writes the raw cube + WCS header directly (native orientation, so DS9 shows the camera's native frame while the WCS resolves correctly). The matcher (`assign_alcor_matches`) is seeded by the resolved epoch geometry and uses a kd-tree with local-asterism pattern verification plus a local relative-brightness tie-break (cloud extinction is patchy, so only relative flux among nearby contested stars is trusted) rather than per-frame nearest-neighbor refitting. Star photometry: `alcor_star_photometry()` measures fixed-position per-channel (R/G/B) aperture+annulus photometry at WCS-predicted positions — no source detection — for named bright stars from `bright_star_sloan_named.fits` (the named variant of the calibration catalog; `alcor_named_reference_altaz()` loads it, `lookup_sloan_photometry(star_name)` returns one star's Sloan row by NAME). **The catalog excludes variable stars by construction** — which is right, since they would corrupt the zeropoints — so a few very bright stars are simply absent; Polaris (HD 8890, V=1.98, a classical Cepheid) is the worked example, and its absence is why nothing flagged the fact that the bad-pixel detector was masking it (see `alcor_badpix_search_region`). That gap is now filled by `bright_variable_vsx.fits` (see the bright-variable paragraph below), which `alcor_star_photometry` measures by default. Frames with the Sun above `sun_alt_max` (-12 deg default) are rejected (empty DataFrame, no CSV). The per-channel bias (median of the four 10x10 image corners, `_corner_bias`) is subtracted before measuring — and the same corner-bias subtraction now feeds the display stretch via `_alcor_display_rgb` (otherwise the raw pedestal gets color-scaled and the image turns purple); the raw cube from `load_alcor_fits` stays untouched. Results are a pandas DataFrame (pandas is a core dependency) indexed by star name (HD-number fallback, duplicate labels suffixed), with instrumental mags `-2.5*log10(flux)` and a per-channel `sat_*` boolean (True when any raw aperture pixel reaches `saturation`, default `ALCOR_SATURATION = 32767`, the camera's 15-bit ceiling — checked on the raw cube, not the bias-subtracted data, so bright stars saturated in one channel can be filtered per-channel downstream without losing the others); every catalog star above `min_altitude` produces a row in every frame — a non-detection (flux non-finite or non-positive in a channel, e.g. a star behind cloud) is recorded per-channel as `flux=0`/`mag=NaN` rather than dropped, since the non-detection is itself a strong extinction signal (the measured `background_*` is still kept, finite for an on-frame position and NaN only when off-frame); the only stars absent are those below `min_altitude` or fainter than `vmag_limit`. Rows sort by descending `flux_g`, and a CSV is always written (`<input>_phot.csv` default). A Gaussian-fit path (`gaussian=True`, CLI `--gaussian`) handles the bright-star CMOS non-linearity that suppresses aperture-sum flux before saturation: it fits a circular Gaussian whose center and width are pinned from a luminance (R+G+B) fit with the non-linear core masked (raw pixels >= `mask_threshold`, default `ALCOR_NONLINEAR_THRESHOLD = 15000`, excluded), recovers each channel's amplitude as the linear projection of the masked, background-subtracted aperture onto the fixed unit-Gaussian profile (so the linear wings set the amplitude), and reports the analytic Gaussian integral `2*pi*A*sigma^2` as `flux_*`; the shared luminance FWHM is added as an `fwhm` column (NaN in aperture mode), `xcen`/`ycen` then hold the fitted center, and `sat_*` still reflects raw-core saturation (now informational). A combined path (`both=True`, CLI `--both`, overrides `gaussian`) measures both methods in one pass into a single CSV with the aperture columns suffixed `_ap` and the Gaussian columns `_gauss` (plus the shared WCS-predicted `xcen`/`ycen` and the Gaussian-fitted `xcen_gauss`/`ycen_gauss`/`fwhm`); the aperture flux is 0 at a non-detection while the `_gauss` columns are NaN when the fit cannot run at all (a structural failure, distinct from a measured zero), and rows sort by `flux_g_ap`. `collect_alcor_photometry` is column-agnostic (it only needs `name`), so this wider schema flows through unchanged — useful for measuring aperture and Gaussian on identical frames in one parallelized run for direct comparison. `collect_alcor_photometry(inputs)` gathers a set of those per-frame CSVs (a directory globbed for `*_phot.csv`, or an explicit path list) into one combined DataFrame with a UT `OBSTIME` column (parsed from each `YYYY_MM_DD__HH_MM_SS` filename, MST + 7h) and `name` as a regular column, sorted by `name` then `OBSTIME` so `df.groupby("name")` yields each star's time-ordered light curve for calibration against catalog photometry; unparseable filenames and malformed CSVs are skipped with stderr warnings, and an empty result raises `ValueError`. Photometric calibration: every measured magnitude is converted to the catalog system and `alcor_star_photometry` always writes two extra per-channel columns — `cal_*` (calibrated catalog-system mag) and `ext_*` (calibrated − catalog = the line-of-sight cloud extinction in mag, positive when dimmer). This is done by `alcor_calibrate_photometry(df, time=None)` (the reusable DataFrame helper; resolves the zeropoint epoch from `time`, else a per-row `OBSTIME` column, else the latest epoch; star names from a `name` column or the index; needs an `altitude` column), applying `cal = (mag − ALCOR_AIRMASS_TERM·airmass) + zp + color_coeff·(B−V)` with Kasten-Young airmass and per-band `zp`/`color_coeff` from the time-indexed `ALCOR_ZEROPOINTS` table (resolved by `alcor_zeropoint(time)`, mirroring `ALCOR_CALIBRATIONS`/`alcor_calibration`). The single achromatic extinction term is `ALCOR_AIRMASS_TERM = 0.40` mag/airmass (no chromatic effect; the zeropoints were fit with it held fixed, so they are a matched set); channel→catalog mapping is G→V, R→R (=V−(V−R)), B→B (=V+(B−V)), with G≈V color-flat and R/B carrying real B−V color terms from the instrument bandpasses; zeropoints are stable to ~0.03 mag across the 2024/2026 epochs. `cal_*`/`ext_*` are NaN where the instrument mag is brighter than `ALCOR_BRIGHT_CUT = -12.5` (the CMOS non-linear regime, calibration invalid) or where the star lacks a catalog color. In `--both` mode the calibration columns are suffixed `_ap`/`_gauss` like the rest. The zeropoints were derived by `claude_docs/scripts/zeropoint_calib.py`; the bright cut was widened from -11.5 to -12.5 by `claude_docs/scripts/nonlin_binned.py`, which averages out the intra-pixel-sensitivity × undersampled-PSF jitter with 15-min per-star medians and shows the bright-star magnitude deficit — a single, band- and night-independent function of signal level — onsets near -11.5 but stays small and ~linear (≈0.15–0.20 mag/mag) out to -12.5, then accelerates steeply into a sparse, inconclusive regime that is dropped pending more calibration nights (reference analysis scripts live in `claude_docs/scripts/`, not packaged).

Horizon mask (sky vs not-sky): `load_alcor_horizon_mask(time)` returns `(mask, date)`, a 2-D bool raw-frame mask where `True` = **not-sky** — obstructions above the horizon (terrain, buildings, the lightning rod) plus everything at/below altitude 0 — so valid sky is `~mask`. It is achromatic (one plane shared by R/G/B, unlike the per-channel bad-pixel mask) and an **exclusion** mask: it is not repaired, only used to select valid sky for sky-background / cloud-extinction maps. It is date-resolved (nearest date, `$ALCOR_HORIZON_DIR` override) from `skycam_utils/data/horizon/alcor_horizon_YYYY-MM-DD.fits.gz`, exactly like the calibration / bad-pixel assets, and stable across epochs for the same reason (one epoch covers 2024–2026; add a new epoch only if the camera moves). It is rebuilt by the packaged `alcor_median_stack` + `create_horizon_mask` CLIs (the reference script `claude_docs/scripts/horizon_floodfill.py` now only re-renders the diagnostic figures). The method: a Sobel-edge flood-fill of a cloudy-night median (the smooth, slowly-varying overcast sky leaves only the sharp obstruction edges), treating strong `sobel(log10)` edges as walls, flood-filling the sky from the WCS zenith, and marking everything the fill can't reach as not-sky — so the thin, enclosed lightning-rod spike is captured (the earlier radial altitude-profile extraction in `alcor_horizon_extract.py` structurally missed it, and an az/alt boolean grid was rejected as too coarse). The SW→W building sector (az 225–270), where the Sobel edges break up, is instead filled from the **undetected-star patch** (fixed-position per-frame photometry accumulated over 5 nights by `claude_docs/scripts/sobel_vs_undetected.py`; high undetected fraction = obstructed). A morphological opening (radius 3) severs thin necks so spurious open-sky pockets detach, then a connected-component cleanup drops any not-sky blob that neither reaches the rim nor is rod-sized. Tested in `test_alcor_horizon.py`.

Bright-variable catalog (`bright_variable_vsx.fits`, 637 stars): `bright_star_sloan_named.fits` excludes variable stars **by construction**, which is right — they would corrupt the zeropoints — but it meant the camera had never measured any of them, and Polaris was not merely unmeasured but was being flagged as a cluster of hot pixels because nothing knew it was a real source. This catalog is the other half. It is built by `claude_docs/scripts/build_variable_catalog.py` from AAVSO VSX (Vizier `B/vsx/vsx`) with B−V cross-matched from the Bright Star Catalogue (`V/50/catalog`) — VSX carries no colours, and cross-matching the *calibration* catalog recovers almost nothing for exactly the reason this catalog exists. Selection: VSX flag `V==0` (confirmed, not suspected), V-band `max`, a real `min` rather than an amplitude, `max <= 6.0`, `amplitude >= 0.05` mag, and **not eruptive/cataclysmic**. Two of those cuts have non-obvious rationale. **The amplitude floor is set by systematics, not photon noise**: per-frame scatter is 0.08 mag at V 2.5–4 and 0.21 at V 5–5.5, but at 1000–1500 frames a night the nightly *mean* is good to ~0.005 mag, so the binding limit is the ~0.03 mag epoch-to-epoch zeropoint stability — a 0.2 mag cut would have thrown away Polaris (`alf UMi`, DCEPS, 1.96–2.03, so 0.07 mag, P=3.97 d) along with most low-amplitude Cepheids and Be stars. **The eruptive cut removes stars whose tabulated `max` is a one-off historical outburst** and which now sit permanently at minimum: 58 entries, including Tycho's supernova (`B Cas`, 1572, now V=22), Kepler's (`V0843 Oph`, 1604), `S And` (1885), SN 1987A, and `CK Vul` (Nova Vul 1670, V=23) — note VSX writes these as `SN I`, `SN Ia`, `SN II-P`, `V838MON`, so the type regex must accept whitespace as a terminator or four supernovae survive. Everything else above 5 mag of amplitude is a Mira or SR with a real period (stars that *do* return to maximum), plus `eta Car`, kept because it is genuinely variable at V~4.5 today even though its `Vmax` of −1.0 is the 1843 Great Eruption. `Vmag` is set to `Vmax` (maximum light), since that is what decides whether a star is ever measurable; a large-amplitude Mira spends most of its cycle far below the limit and those non-detections are data.

Two stars, `psi Cen` and `N Sco` (both ~0.3 mag eclipsing binaries), were removed from both calibration catalogs. **The removal threshold (`CAL_REMOVE_AMPLITUDE = 0.2`) is deliberately far looser than the catalog's own 0.05 cut**, because the two answer different questions: 59 of the 637 variables are also in the calibration catalog, and a star varying by 0.05–0.1 mag is worth a light curve *and* is still a sound standard when hundreds are averaged — Arcturus (0.05) and Capella (0.05) are in that class, and removing them would degrade the zeropoints rather than protect them.

Integration: `alcor_variable_reference_altaz()` mirrors `alcor_named_reference_altaz()`, and `alcor_photometry_reference_altaz()` concatenates the two, dropping variables already in the calibration catalog (20″ positional match) so nothing is measured twice — a star in both keeps its calibration row, since it has real catalog magnitudes, and is merely flagged. `alcor_star_photometry(variables=True)` is the **default** (CLI `--no-variables`), adding a `variable` bool column, so `alcor_process_night` and the real-time ingest pick the variables up with no further change. In `alcor_calibrate_photometry` a variable contributes a colour but **no** catalog magnitude, so `cal_*` (the light curve) is real while `ext_*` is NaN — differencing against a catalog magnitude would conflate cloud extinction with the star's own variation. `ALCOR_COLOR_FLAT_TOL = 0.05` lets a colour-flat band survive a missing B−V (37 of 637 stars lack one): G's coefficient is −0.038 so assuming B−V=0 costs under 0.06 mag even for a very red star, while R (−0.343) and B (+0.47) correctly stay NaN. Tested in `test_alcor_variables.py`.

Bad-pixel search region (`alcor_badpix_search_region`): hot-pixel detection runs only where a spike on a smooth background actually means a defect, which is not the whole frame. Two regions are excluded, and both matter. **Not-sky**, from `load_alcor_horizon_mask` dilated by `ALCOR_BADPIX_RIM_DILATION = 4` px so the horizon *step* goes too and not just the terrain behind it — measured on the 2026-01-11 night median, **78% of the z>25 candidates were skyline rather than sensor** (terrain, buildings, ground lights, and the rim gradient the 5 px high-pass fires on), so the shipped masks were largely a map of the horizon and the `NBAD*` counts that `alcor_process_night` stamps for CMOS-aging trending were tracking terrain edges. It also explains why only ~44% of one night's mask pixels repeat on another: most of the non-overlap was rim noise, not aging (the surviving *sky* flags repeat 941/971 across nights five months apart). And **a `ALCOR_BADPIX_POLE_RADIUS = 15` px disc at the north celestial pole** (due true north at an altitude equal to the site latitude, located via the WCS), because the detector's premise is a trail-free night median and that holds everywhere *except* at the pole: Polaris moves only ~10 px in a night, so its trail survives the median and was being flagged as a cluster of hot pixels — a *different* cluster each night, since the trail lands at a different hour angle, which is itself the proof they were not pixels. **A corollary worth remembering: a fixed-row bright line in a keogram is not automatically a bad pixel.** The keogram's zenith column passes ~1 px from the celestial pole, so Polaris is permanently in it; the line at raw row ~1176 (az 359.3, alt 31.7 = the site latitude) is a real star and correct output, and an earlier session's diagnosis of it as an unmasked hot pixel from a badpix epoch gap was wrong. Convert the row to az/alt with the WCS before blaming the sensor; confirm from the night median, where a star trail is elongated, white in all three channels, and *moves* between nights, while a hot pixel is a single-channel point that does not. Tested in `test_alcor_badpix.py`.

Night-level processing (`alcor_process_night`, packaged CLI): the archive driver that turns one night's directory into the standard data products in a single pass. Night is **Sun < -12 deg with no Moon cut** — unlike every other night-selecting entry point here, moonlight is the signal rather than contamination. Each frame is decompressed **once** and feeds three consumers: `alcor_star_photometry` (via its `frame=` passthrough, which skips the internal load), `_alcor_sky_brightness_map`, and optionally a slot in the night's raw median stack. Two invariants are load-bearing. (1) The **sampling cones are built once per night, not per frame** — `_alcor_cone_indices` turns each `ALCOR_SB_TARGETS` (az, alt) into flat pixel indices from the night's WCS (great-circle separation <= `ALCOR_SB_APERTURE_RADIUS` = 5 deg, horizon-masked), and they reach the worker pool through a `ProcessPoolExecutor` *initializer* rather than per-task pickling; per frame it is then just a `median` over ~5000 pixels. The resulting `sky_brightness.csv` columns `allsky_mv_zenith` / `allsky_mv_tucson` / `allsky_mv_nogales` (V mag/arcsec^2, descriptions in `ALCOR_SB_TARGET_DESCRIPTIONS`) sample the *same patch of sky every frame of every night*, which is the whole point — a clear dark zenith reads ~21.5 and the two low-altitude light domes sit ~1 mag brighter. (2) The **median stack is built from RAW cubes** — the worker loads `badpix=None`, writes the raw frame to the memmap slot, and applies `_apply_badpix_repair` itself for the photometry/SB path; stacking repaired data would interpolate away exactly the pixels the stack exists to track. It writes `<night>_median.fits` stamped with `NBADR`/`NBADG`/`NBADB` (a header read gives the night's hot-pixel count, so CMOS aging is trendable) but deliberately writes **no mask** — `create_badpix_mask` still owns that, and its stricter Sun<-18/Moon<-6 selection makes the counts not strictly comparable. The calibrated nighttime keogram takes the **same raw zenith column `alcor_keogram` takes** (not a resampled altitude grid), so the SB and RGB keograms stack row-for-row; `save_alcor_sb_keogram_fits`/`_plot`/`plot_alcor_sb_keogram_fits` are the 2-D `mag/arcsec2` siblings of the RGB keogram trio. With `--day-keogram` the driver also builds the full-day RGB keogram itself (the night columns are free from the main pass, only the daylight remainder is re-read), which is why `scripts/make_movies.sh` now calls this instead of `alcor_keogram`. (3) The driver is **resumable through the per-frame photometry CSVs**: an existing non-empty `<frame>_phot.csv` is reused rather than re-measured — that is the hook for doing photometry as images arrive — but the frame is still read, because the SB map and keogram columns need its pixels and the CSV does not carry them; `--reprocess` forces re-measurement, which is required after changing photometry options. Tested in `test_alcor_night.py`.

Packaging is PEP 621 / `pyproject.toml`-only — there is no `setup.py`, `setup.cfg`, `MANIFEST.in`, or `tox.ini`. The version is generated by `setuptools_scm` into `skycam_utils/_version.py` at build/install time (gitignored). `AGENTS.md` is a symlink to this file — edit `CLAUDE.md` only.

`docs/` is reserved for the Sphinx/Read-the-Docs documentation root (the `# Build docs` command below points there). Non-packaged reference/analysis material — the calibration and analysis scripts (`claude_docs/scripts/`) and the committed reference figures (`claude_docs/gplots/*.png`, kept in the repo on purpose via a `.gitignore` negation) — lives under `claude_docs/`, *not* `docs/`. (It was all renamed out of `docs/` to free that name for Sphinx.)

## Common commands

```bash
# Install for development (editable, with test extras)
pip install -e ".[test]"

# Run the test suite
pytest
# Run a single test file:
pytest skycam_utils/tests/<file>.py::<test>

# Build the package
python -m build

# Build docs
pip install ".[docs]" && sphinx-build -W -b html docs docs/_build/html

# CLI entry points (installed via [project.scripts])
process_stellacam_dir <YYYYMMDD-dir> [--writefits] [--zeropoint Z] [--nproc N] [-z] [-s]
#   -z       : process *.fits.gz instead of *.fits
#   -s       : produce strip image + plot instead of per-frame photometry
#   The directory NAME must start with YYYY — that's how `year` is derived,
#   which selects the WCS and mask files via load_wcs()/load_mask().

alcor_proc_fits <input.fits> [-o OUT] [--overwrite]
#   Writes the raw (3, ny, nx) RGB cube (native orientation) with the raw-frame
#   alt/az WCS in the header. Geometry comes from the WCS, so there are no
#   geometry flags. Default output: <input>_proc.fits.

plot_alcor_fits <input.fits> [-o OUT.pdf] [--outimage RAW] [--radius 680] [--gscale ...] ...
#   Renders an annotated all-sky figure, cropping a --radius-pixel square around
#   the WCS zenith and rendering north-up (origin="lower"). --radius is the
#   display crop only. Default output: <input>.pdf (extension drives the backend).

plot_alcor_sky_brightness <input.fits> [-o OUT.pdf] [--outimage IMG] [--radius 680] [--fov-altitude -2] [--horizon-mask] [--saturation N] [--vmin V] [--vmax V] [--cmap cividis_r]
#   Renders the G channel as an observed V mag/arcsec^2 sky-brightness map:
#   exposure-normalized to 20 s, divided by the WCS per-pixel solid angle, and
#   converted with the G->V zeropoint (no airmass term). Masks non-sky (alt <
#   --fov-altitude, or the full horizon mask with --horizon-mask). --saturation is
#   OFF by default: a blanked pixel renders as background, so the brightest
#   sources would show as DARK holes. Pass a raw-ADU level to restore the mask. Colorbar in mag/arcsec^2; the sigma-clipped median zenith
#   (alt > 85 deg) brightness is annotated. Auto-scales by default (range varies with
#   moonlight). Default output: <input>_skybright.pdf.

alcor_sky_brightness <input.fits> [-o OUT.fits] [--horizon-mask] [--saturation N] [--overwrite]
#   FITS-output sibling of plot_alcor_sky_brightness: writes the calibrated G->V
#   observed V mag/arcsec^2 map as a 2-D float32 FITS in the camera's NATIVE
#   orientation with the raw-frame alt/az WCS attached (so DS9 resolves it; matches
#   alcor_proc_fits). Same calibration chain (badpix repair, exposure-normalized to
#   20 s, WCS per-pixel solid angle, G->V zeropoint, NO airmass) via the shared
#   _alcor_sky_brightness_map helper. NaN-blanks off-frame, and raw G >=
#   --saturation only when one is given (off by default, see above);
#   --horizon-mask additionally blanks not-sky (no altitude floor otherwise). Header
#   carries BUNIT='mag/arcsec2' + provenance (ZP_G, ZP_EPOCH, EXPOSURE, CALIBEXP,
#   SATLEVEL, HORIZMSK). Reusable as alcor_sky_brightness_fits(filename, ...).
#   Default output: <input>_sb.fits.

plot_alcor_sb_summary <sky_brightness.csv> [-o OUT.png] [--title T] [--figsize W H] [--dpi 140]
#   Plots one night's sky_brightness.csv (from alcor_process_night) as a time
#   series: every allsky_mv_* track vs UT with the magnitude axis INVERTED so a
#   brighter sky runs downward (matching the SB maps and keograms), astronomical
#   twilight (Sun > -18) shaded, and a lower panel carrying the Moon's altitude
#   plus, on a right-hand axis, the altitude where the floating darkest cone was
#   found. The title's dark-sky medians are computed over Sun < -18 AND Moon < 0
#   only, so a moonlit stretch cannot drag them, and report the two SKY tracks
#   (zenith, darkest cone) -- the light domes are not a darkness measure. The
#   legend is anchored in DATA coordinates just left of the morning-twilight
#   band; an axes-fraction anchor is wrong by matplotlib's 5% x-margins and
#   lands on the shading. Deliberately NOT wired into alcor_process_night: the
#   archive is already being processed, and the CLI runs over the finished CSVs
#   afterwards. Reusable as plot_alcor_sb_summary(csv, output_file=None, ...).
#   Default output: the input with a .png suffix (extension drives the backend).

alcor_star_photometry <input.fits> [-o OUT.csv] [--aperture-radius 4] [--annulus-width 1] [--min-altitude 20] [--vmag-limit 5.5] [--no-refraction] [--no-variables] [--sun-alt-max -12] [--saturation 32767] [--gaussian] [--both] [--mask-threshold 15000] [--check-plot] [--check-radius 680]
#   Fixed-position RGB aperture photometry of named bright stars at their
#   WCS-predicted pixels (no detection step). Writes <input>_phot.csv indexed
#   by star name, with a per-channel sat_* flag (raw aperture pixel >=
#   --saturation); --check-plot overlays the apertures on the plot_alcor_fits
#   rendering as <input>_phot.pdf. Prints the warning and writes nothing when
#   the Sun is above --sun-alt-max.
#   Always adds per-channel cal_* (catalog-system mag via ALCOR_ZEROPOINTS: G->V,
#   R->R, B->B) and ext_* (cal - catalog = cloud extinction in mag); both NaN for
#   measurements brighter than ALCOR_BRIGHT_CUT=-12.5 (CMOS non-linear) or stars
#   lacking a catalog color. Reusable as alcor_calibrate_photometry(df, time=None).
#   By default it ALSO measures the 637-star bright-variable catalog alongside the
#   calibration stars and adds a `variable` bool column; --no-variables opts out.
#   A variable has no single catalog magnitude, so its ext_* is NaN while cal_* --
#   the light curve -- is real. See the variable-catalog paragraph below.
#   --gaussian switches to constrained-Gaussian PSF photometry (luminance-pinned
#   center/width, non-linear core masked at --mask-threshold, analytic-integral
#   flux) to recover bright-star flux lost to CMOS non-linearity; it adds an fwhm
#   column. --aperture-radius also sets the Gaussian fit window.
#   --both measures aperture AND Gaussian in one pass into a single combined CSV
#   (columns suffixed _ap / _gauss, plus shared WCS-predicted xcen/ycen and the
#   Gaussian-fitted xcen_gauss/ycen_gauss/fwhm); a star is kept when either method
#   gives finite positive flux (failed method NaN), rows sort by flux_g_ap, and it
#   overrides --gaussian. collect_alcor_photometry is column-agnostic so the
#   combined schema flows through unchanged.

alcor_process_night <night-dir> [-o OUT-DIR] [--pattern *.fits.bz2] [--sun-alt-max -12] [--sb-aperture-radius 5] [--no-horizon-mask] [--sb-saturation N] [--best-min-altitude 30] [--write-sb-fits] [--median-stack] [--day-keogram] [--reprocess] [--aperture-radius 4] [--annulus-width 1] [--min-altitude 20] [--vmag-limit 5.5] [--gaussian] [--both] [--max-frames N] [--scratch-dir DIR] [--masks-dir DIR] [--workers N] [--overwrite] [--quiet]
#   Night-level driver: processes one archived night end to end. Selects frames
#   with the Sun below --sun-alt-max (NO Moon cut -- moonlit sky brightness is the
#   signal, not contamination), then decompresses each frame ONCE and derives
#   three things from it: star photometry (<frame>_phot.csv), a calibrated
#   V mag/arcsec^2 surface-brightness map, and optionally a slot in the night's
#   raw median stack. Writes (default alongside the frames, -o redirects):
#     sky_brightness.csv      filename, OBSTIME (UT), exposure, sun_alt, moon_alt,
#                             moon_az, allsky_mv_zenith, allsky_mv_tucson,
#                             allsky_mv_nogales, allsky_mv_best, best_az, best_alt.
#                             The three named columns are the MEDIAN surface
#                             brightness within a --sb-aperture-radius (5 deg) cone
#                             about a FIXED (az, alt) from ALCOR_SB_TARGETS: zenith,
#                             az=0/alt=15 (Tucson dome), az=190/alt=15 (Nogales
#                             dome) -- the same patch of sky every frame of every
#                             night. allsky_mv_best is the same statistic at a
#                             FLOATING position: the darkest of ~100 candidate cones
#                             tiling the sky above --best-min-altitude (30 deg),
#                             with best_az/best_alt recording where it was found.
#                             It exists because the ZENITH IS NOT A DARKNESS
#                             MEASURE -- the Milky Way transits through it, so the
#                             zenith value tracks galactic latitude as much as sky
#                             quality; over a night high galactic latitude does
#                             transit, so the darkest cone is what says how dark the
#                             site actually got. The zenith is one of the candidates,
#                             so allsky_mv_best >= allsky_mv_zenith always. A single
#                             darkest PIXEL would be wrong (stars only push pixels
#                             brighter, so the extreme dark tail is read noise);
#                             hence a cone median, ~7 ms/frame. Horizon-masked by
#                             default so terrain cannot drag the low cones faint.
#     <night>_sb_keogram.fits/.png  the calibrated nighttime keogram: the zenith
#                             column of every SB map, i.e. the SAME raw column
#                             alcor_keogram takes, so it stacks row-for-row against
#                             the raw RGB keogram. 2-D float32, BUNIT=mag/arcsec2,
#                             TIMESTAMPS bintable; NaN where not measurable.
                             The column is taken from the UN-horizon-masked map,
#                             so like the RGB keogram it runs BELOW the horizon at
#                             both ends -- that band is where the light domes are,
#                             and showing it is the point. The stop is the optics,
#                             not the sensor: _alcor_sky_brightness_map blanks
#                             everything beyond ALCOR_FIELD_RADIUS = 680 px from
#                             the optical axis (CRPIX), the edge of the camera's
#                             illuminated image circle, which lands at alt ~-2.5.
#                             Live rows are 24..1383 of 1411. Without that cut the
#                             outermost ~50 rows report a confident-looking ~25
#                             mag/arcsec^2 that is pure artifact -- near-zero
#                             signal divided by a solid angle. The allsky_mv_*
#                             cones are unaffected: _alcor_cone_indices builds
#                             their index sets with exclude=horizon, so terrain
#                             cannot reach them, and they sit far inside the field.
#     <night>_phot.csv        the collect_alcor_photometry rollup (free -- the
#                             per-frame CSVs already exist).
#     <night>_median.fits     --median-stack only: the per-channel median of the
#                             night's RAW frames (bad-pixel repair would erase the
#                             very pixels it tracks), stamped with NSTACK and
#                             NBADR/NBADG/NBADB from build_alcor_badpix_mask, so
#                             per-night hot-pixel counts are a header read and CMOS
#                             aging can be trended. It does NOT write a mask --
#                             create_badpix_mask still owns alcor_badpix_*.fits.gz,
#                             and its stricter Sun<-18/Moon<-6 selection means the
#                             counts are not strictly comparable to a shipped mask.
#                             Needs scratch for the whole night (~12 MB/frame).
#     <night>_keogram.fits/.png  --day-keogram only: the FULL-DAY raw RGB keogram
#                             that alcor_keogram used to build on its own. A day
#                             directory spans local noon -> next morning, so this
#                             covers daylight too: the night frames' columns come
#                             back FREE from the main pass (same cube, same zcol)
#                             and only the daylight remainder needs a second,
#                             column-only read. Timestamps are _alcor_frame_time
#                             (filename first) rather than the raw DATE header
#                             string alcor_keogram stores, so both keograms of a
#                             run share one time source.
#   --write-sb-fits keeps each frame's full SB map as <frame>_sb.fits (off by
#   default: several MB each, and the summary + keogram already carry the signal).
#   RESUME: a frame whose <frame>_phot.csv already exists and is non-empty is NOT
#   re-measured (the real-time ingest will eventually write those as images come
#   in), but IS still read -- the SB map and both keogram columns need its pixels
#   and the CSV does not carry them, so every product stays complete. Photometry is
#   only ~17% of per-frame cost (load+repair is ~63%), so this saves the measurement,
#   not the frame. --reprocess forces re-measurement; use it after changing
#   photometry options, since a CSV written in another mode is otherwise reused
#   as-is and collect_alcor_photometry would pool a mixed schema.
#   Reusable as alcor_process_night(night_dir, ...), which returns a dict of the
#   DataFrame, keogram arrays, and written paths. A frame that fails is recorded in
#   errors[] with NaN values rather than aborting the night.
#   scripts/make_movies.sh calls this (with --day-keogram) in place of its old
#   alcor_keogram call, then publishes both keograms to keograms/<year>/{png,fits}/.

alcor_process_archive <archive-dir> -o <out-dir> [--status] [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--nights N ...] [--reverse] [--min-age 24] [--prune-jpegs] [--prune-dry-run] [--retry-failed] [--force-options] [--max-consecutive-failures 10] [--verbose] [--day-keogram] [--median-stack] [--both] [--workers N] [--scratch-dir DIR]
#   Archive-wide driver: runs alcor_process_night over every YYYY-MM-DD night
#   directory under <archive-dir>, resumably. Hundreds of nights and days of wall
#   clock, so it adds only what that scale needs and no science of its own.
#   ALL STATE LIVES IN <out-dir>/.archive_state/, whose absolute path is the first
#   line of every run:
#     ledger.json   per-night state (pending/running/done/failed), timings, frame
#                   and error counts, product paths, jpegs pruned. Rewritten
#                   atomically after each night; `done` nights are skipped on a
#                   rerun, and a night left `running` by a crash is reset to
#                   `pending`. Frame-level resume then comes free, since
#                   alcor_process_night reuses a non-empty <frame>_phot.csv.
#                   It also stores an OPTIONS FINGERPRINT (both/gaussian/
#                   aperture_radius/annulus_width/min_altitude/vmag_limit/
#                   variables); resuming with any of them changed is REFUSED
#                   unless --force-options, because a rollup built from CSVs
#                   written in two modes is a silently mixed schema.
#     PAUSE         create this file to pause: `touch <out-dir>/.archive_state/PAUSE`
#                   and `rm` it to resume. Checked BETWEEN nights, so a pause lands
#                   at the next boundary (up to ~30 min) and the night in flight
#                   always completes. It is a file, not a signal, so a detached
#                   run can be held without finding its PID. One left in place
#                   holds the NEXT run at its first night.
#     archive.log   the run log, also echoed to stderr.
#   Each night's products go to <out-dir>/<night>/. The archive is read-only
#   except for --prune-jpegs.
#   SKIPS NIGHTS STILL ARRIVING: the archive syncs from the camera host, so a night
#   whose mtime is inside --min-age (24 h) is skipped, logged, and left `pending`
#   -- not `failed`, since nothing failed. The test is MTIME, never the filename
#   timestamps: the current night is still being observed (recent names), while an
#   old night being back-filled has old names and fresh mtimes, and only mtime
#   catches both. Without this a half-arrived night would be recorded `done` and
#   never revisited. --min-age 0 disables it.
#   --prune-jpegs DELETES a night's vendor JPEGs (~7.3 GB/night; "*.jpg" covers the
#   Unwrap_ series too) after its products are written, gated on sky_brightness.csv
#   and <night>_phot.csv existing non-empty and under 10% frame errors. Never
#   prunes a night skipped as too recent. It also prunes already-`done` nights that
#   were never pruned, so it can be enabled partway through or as a later pass.
#   --prune-dry-run reports and deletes nothing.
#   Per-frame log lines are throttled to one progress line a minute (--verbose
#   disables); other messages pass through. One bad night is recorded `failed` and
#   the run continues, but --max-consecutive-failures (10) aborts the run, which is
#   what catches the archive unmounting mid-run. --retry-failed re-attempts them.
#   --status reports counts, per-night timing, an ETA, and the failure list from the
#   ledger alone -- no archive needed, and safe while a run is in flight.
#   Reusable as alcor_process_archive(archive_dir, out_dir, ...) and
#   alcor_archive_status(out_dir).

alcor_keogram <input-dir> [-o OUT.png] [--fits-output OUT.fits] [--pattern ...] [--workers N] [--no-progress] [--powerstretch ...] [--gscale ...] [--bscale ...]
#   Standalone raw RGB keogram over a whole day: the zenith column of each frame's
#   raw cube stacked into an (ny, nframes, 3) image + DATE timestamps. Still useful
#   on its own, but alcor_process_night --day-keogram produces the same product as
#   part of a night run and reuses the night columns it has already loaded.
#   plot_alcor_keogram_fits re-renders a saved one; plot_alcor_sb_keogram_fits is
#   the calibrated sibling and takes --vmin/--vmax/--cmap in mag/arcsec^2.
#   BOTH keogram plotters render NORTH-UP via the shared _set_keogram_yaxis: a
#   keogram column is the raw zenith pixel column and build_alcor_wcs puts north at
#   INCREASING y, so row 0 is south and the axis runs S/Z/N bottom to top. Drawing
#   with origin="upper" or invert_yaxis() silently produces an upside-down keogram
#   (this was a real bug in save_alcor_keogram_plot); test_alcor_night.py pins both
#   the WCS premise and the axis direction.
#   Both savers also write a ROWALT bintable extension holding the per-row altitude
#   in deg (_keogram_row_altitude), so a saved keogram is self-describing and the
#   re-plotters pick it up automatically. With it, _set_keogram_yaxis dashes the two
#   alt=0 crossings, and the SB plotter takes its default vmin/vmax percentiles from
#   SKY ROWS ONLY (_sb_keogram_limits) -- terrain and the domes are ~2 mag brighter
#   than sky and would otherwise compress the sky contrast away; they saturate the
#   bright end instead. A keogram written before ROWALT existed still loads: the
#   extension is optional and its absence just restores the old whole-frame scaling.

fit_alcor_wcs <night-dir> [--pattern ...] [--vmag-limit 4] [--tolerance 3] [--fit-k5] [--max-detections 200] [--sun-alt-max -18] [--moon-alt-max -6] [--residual-plot OUT.png] [--max-frames N] [--workers N] [--quiet]
#   Aggregates bright-star matches across dark frames across a night and prints
#   a ready-to-paste ALCOR_CALIBRATIONS epoch dict (absolute raw-frame constants —
#   xcen, ycen, rotation, radial_coeffs, tangential_coeffs, axis_tilt, horizon_radius — stamped with the night's
#   UT date) to add to alcor/config.py and commit.
#   --fit-k5 is REQUIRED to reproduce the committed calibrations: by default only the
#   cubic k3 radial term is fit, but every committed epoch uses the full quintic (k5)
#   model. Without --fit-k5, k3 runs away absorbing the quintic curvature and the
#   high-zenith residual balloons (pooled RMS ~1.44px k3-only vs ~0.33px k3+k5 on
#   2026-05-18; --residual-plot makes this obvious in the residual-vs-zenith panel).
#   Detections are capped to the brightest --max-detections per frame; matching is
#   seeded by the nearest epoch and done with assign_alcor_matches (cKDTree candidate
#   search, local-asterism pattern verification, and a local relative-brightness
#   tie-break for contested detections), with no per-frame geometry refit. The match
#   tolerance tightens over several rounds from ~12px to --tolerance (~3px). Also
#   prints the matched fraction so contamination/coverage is visible.
#   Dark-frame selection keeps frames with Sun < --sun-alt-max AND Moon < --moon-alt-max
#   (-6deg default; moonlight scatter swamps the faint star field and corrupts detection;
#   pass --moon-alt-max 90 to disable). It parses the UT from each YYYY_MM_DD__HH_MM_SS
#   filename (local MST = UT-7), so it never opens files; oddly-named files fall back to
#   the DATE header.
#   Per-frame load/detect is parallelized across processes (--workers; default: one per core).
#   Prints each file's disposition to stderr (Sun-rejected / no stars / used + count); --quiet silences it.

create_badpix_mask <day-dir> [--out-dir DIR] [--min-frames 500] [--z-thresh 25] [--ksize 5] [--rim-dilation 4] [--pole-radius 15] [--horizon-dir DIR] [--sun-alt-max -18] [--moon-alt-max -6] [--max-frames N] [--scratch-dir DIR] [--pattern *.fits.bz2] [--quiet]
#   Builds a date-stamped per-channel bad-pixel mask from one night of frames and writes
#   alcor_badpix_YYYY-MM-DD.fits.gz to --out-dir (default: $ALCOR_BADPIX_DIR, then packaged
#   data/badpix/). Prints the output path, or "# no mask written" when fewer than
#   --min-frames dark frames (Sun < --sun-alt-max AND Moon < --moon-alt-max, default -18/-6)
#   are available — the existing nearest mask keeps applying.
#   Selects dark frames, builds a trail-free per-pixel night-MEDIAN stack (RAM-bounded via a
#   disk memmap + row tiles; --max-frames strided-caps it), then flags hot pixels per channel
#   with a --ksize (5px) median high-pass and a robust z > --z-thresh (25) cut, keeping only
#   spikes that fire in <=2 of the 3 channels (a 3-channel spike is a real broadband source).
#   Detection runs ONLY inside alcor_badpix_search_region (the robust sigma is still measured
#   over the whole frame — it is a read-noise scale, not a property of the region). Two
#   exclusions, both because the detector assumes a sharp spike on a smooth, trail-free
#   background and neither region delivers one:
#     * NOT-SKY, from the nearest horizon mask (--horizon-dir), dilated --rim-dilation (4) px
#       so the horizon STEP goes too, not just the terrain behind it. This is the big one:
#       measured on the 2026-01-11 median, 78% of z>25 candidates were skyline, not sensor —
#       so the masks were mostly a map of the horizon and the NBAD* aging counts tracked
#       terrain edges rather than the CMOS. 0 searches the whole frame.
#     * A --pole-radius (15) px disc at the NORTH CELESTIAL POLE (due true north at an
#       altitude equal to the site latitude, from the WCS). Stars there barely move, so the
#       night median keeps their trails: Polaris (V=1.98, ~10 px of motion per night) was
#       being flagged as a cluster of hot pixels — a DIFFERENT cluster each night, since the
#       trail lands at a different hour angle, which is itself the proof they are not pixels.
#       0 disables. Costs ~700 px of a 2M-px sensor.
#   Header carries RIMDILAT/POLERAD/NSEARCH alongside NSTACK/ZTHRESH/KSIZE/CHRULE/NBAD*.
#   Mask date is the YYYY-MM-DD in the directory name, else the median dark-frame date.
#   Consumed by load_alcor_fits(badpix="repair") and resolved nearest-in-date (load_alcor_badpix_mask).
#   Unlike fit_alcor_wcs's effectively-stationary epoch, the hot-pixel set AGES (~half turns over
#   2024->2026, CMOS aging), so masks are rebuilt regularly (meant to run daily from cron), not once.

alcor_median_stack <cloudy-night-dir> [-o OUT.fits] [--sun-alt-max -18] [--moon-alt-max 90] [--no-badpix] [--max-frames N] [--scratch-dir DIR] [--pattern *.fits.bz2] [--quiet]
#   Median-stacks one CLOUDY night's frames into a 2-D luminance (R+G+B) FITS with
#   the raw-frame alt/az WCS in the header — the smooth-overcast input whose Sobel
#   edges define the horizon obstructions. Dark-frame selection (Sun < --sun-alt-max);
#   the Moon cut defaults OFF (--moon-alt-max 90) since the cloudy night is chosen by
#   hand. Zeros the nearest bad-pixel mask unless --no-badpix. Default out:
#   <night-name>_median.fits.

create_horizon_mask <median.fits> [--epoch YYYY-MM-DD] [--out-dir DIR] [--phot-nights DIR ...] [--edge-pct 96] [--open-radius 3] [--sector 225 270] [--und-thr 0.5] [--und-mincount 15] [--rim-alt 1.5] [--rod-area-min 400] [--quiet]
#   Sobel-edge flood-fill of the cloudy-night median into alcor_horizon_YYYY-MM-DD.fits.gz
#   (written to --out-dir, default $ALCOR_HORIZON_DIR or packaged data/horizon). Epoch is
#   parsed from the median filename unless --epoch is given; the WCS is built from the
#   nearest calibration. --phot-nights DIRS accumulates the SW->W (az 225-270) undetected-
#   star patch from each dir's *_phot.csv; omit it and that sector falls back to Sobel-only.
#   Consumed by load_alcor_horizon_mask, resolved nearest-in-date. Like create_badpix_mask
#   it needs local raw data and is not reproducible from a bare pip install.
```

`alcor` is a **package**, split by concern out of what was a single 5700-line `alcor.py`: `config` (constants, `ALCOR_CALIBRATIONS`, `ALCOR_ZEROPOINTS`), `timeutils` (frame time, Sun/Moon, dark-frame selection), `wcs` (ARC WCS + distortion model), `wcsfit` (detection, matching, `fit_alcor_wcs`), `masks` (loading the badpix/horizon assets), `badpix`, `horizon`, `catalogs`, `io` (`load_alcor_fits`, `_corner_bias`), `photometry`, `display`, `skybright`, `keogram`, `night` (`alcor_process_night`), `ledger` (the archive run ledger), `archive` (`alcor_process_archive`), and `cli` (all 15 entry points). **`alcor/__init__.py` re-exports the entire namespace**, private names included, so `from skycam_utils.alcor import <anything>` and the `skycam_utils.alcor:*_cli` entry points work exactly as before — nothing outside the package had to change. Dependencies run one way (`config → timeutils → wcs → masks → {badpix, horizon, catalogs} → io → {photometry, display, skybright} → keogram → night → archive → cli`, with `ledger` a dependency-free leaf that only `archive` imports); the two cycles that a naive split creates are worth knowing about, since both are easy to reintroduce: `badpix` needs the horizon mask while `horizon` needs the badpix mask (broken by keeping both *loaders* in `masks`, separate from the *builders*), and `photometry` needs the check plot while `display` needs `_corner_bias` (broken by `_corner_bias` living in `io`, where a frame-level bias estimate belongs). `claude_docs/scripts/split_alcor.py` is the record of how the split was done.

**Tests must patch with the `patch_alcor` fixture** (`skycam_utils/tests/conftest.py`), not `monkeypatch.setattr(alcor, ...)`. When `alcor` was one module there was one namespace, so patching it reached every caller; now each submodule binds its own name (`from .io import load_alcor_fits`), and patching the package re-export leaves the real consumers running the real function — **the test still passes, for the wrong reason**. `patch_alcor(name, value)` replaces the name in every submodule that binds it, which is what the old single-namespace patch actually meant, and raises rather than silently doing nothing if the name is bound nowhere.

Test coverage is concentrated on the Alcor module (`test_alcor.py`, `test_alcor_wcs.py`, `test_alcor_badpix.py`, `test_alcor_horizon.py`, `test_alcor_skybright.py`, `test_alcor_night.py`, run against the bundled `test.fits.bz2` frame and synthetic geometry). The Stellacam pipeline and photometry/astrometry modules have no tests — don't assume coverage exists for code you change there.

## Pipeline architecture

`pipeline.py` is the orchestrator; the photometry/astrometry modules are libraries it composes:

```
process_stellacam_image(fitsfile, year)
  ├─ get_ut(hdr, year)                       # year-dependent header parsing
  ├─ Filters frames: only FRAME='256 Frames' AND GAIN=106 are processed
  │                  (the dark-sky steady-state config — other configs are skipped, not failed)
  ├─ load_mask(year)        / load_wcs(year) # year → packaged FITS in skycam_utils/data/
  ├─ load_skycam_catalog()  → update_altaz() # apply current-time AltAz to the curated star catalog
  ├─ photometry.make_background()            # photutils Background2D, optional source masking
  ├─ photometry.make_segmentation_image()    # detect_sources + deblend
  ├─ photometry.make_catalog()               # source_properties → table with obs_mag
  ├─ photometry.match_stars()                # WCS pix→AltAz, match against skycat by 2.5° sep
  └─ writes .cat.csv (always); .bkg/.subt/.sky FITS only when --writefits
```

`process_stellacam_dir()` then groups all per-frame `.cat.csv` outputs by `Star Name` and writes one `star_<name>.csv` per matched star. Per-frame work is parallelized via `multiprocessing.Pool`.

### Year-keyed calibration assets

`load_wcs()` and `load_mask()` map year → file. Currently supported buckets: 2011–2012, 2015–2016, 2017–2021. **Adding a new year requires extending both functions** and shipping the corresponding FITS in `skycam_utils/data/` (covered by the `skycam_utils = ["data/*"]` glob in `[tool.setuptools.package-data]`).

### WCS fitting (`fit_wcs.py`)

Borrowed from LSST. Provides `wcs_zea` / `wcs_azp` callable classes used as objective functions for `scipy.optimize.minimize` in `astrometry.initial_wcs_fit()`. `wcs_sip_fit()` then refines with SIP distortion via `astropy.wcs.utils.fit_wcs_from_points`. SIP terms only survive a write if you use `write_sip()` — `to_fits()` drops them by default.

## `scripts/` is operational, not packaged

These run on the live skycam host (Windows WSL — paths like `/mnt/d/skycam/...`, `/mnt/c/Users/skycam/...`) and on `ops.mmto.arizona.edu`. They are **not** part of the installed Python package and are not on PYTHONPATH:

- `make_movies.sh` / `daily_movies.sh` — ffmpeg concat-demuxer pipelines that produce `allsky.mp4` and `unwrap.mp4`. The `pad=ceil(iw/2)*2:ceil(ih/2)*2` filter is required because H.264 needs even dimensions; do not remove it. Bitrate/framerate constants in these scripts have been hand-tuned over many commits — change them only when explicitly asked.
- `latest_image.sh` / `latest_movie.sh` — scp/ffmpeg jobs that publish the most-recent image and a rolling 150-frame movie to the public web host.
- `mmt_position.py` / `tcs_logger.py` — Redis publishers reading TCS state and weather telemetry. Default Redis host is `redis.mmto.arizona.edu`; override via `REDISHOST` / `REDISPORT` / `REDISPW` env vars. `mmt_position.py` writes the alcor-format Dublin-JD timestamp (`jd - 2415020 + 1.5`) plus RA/Dec/LST in radians to `/mnt/d/skycam/mmt_position.txt`.

When editing scripts under `scripts/`, assume the absolute paths and the cron/systemd context they run under are load-bearing.

## External dependencies

- **astrometry.net** — `astrometry.solve_field()` shells out to `solve-field` with hardcoded `-L 100 -H 150 -u app` (arcsec-per-pixel scale bounds) and `--no-background-subtraction`. Requires the binary on `PATH` plus appropriate index files installed; it's not a Python dependency.
- **redis** — only needed by the `scripts/` operational tools, not the photometry pipeline.
