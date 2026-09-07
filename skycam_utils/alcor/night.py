"""The night-level archive driver that produces a night's data products."""

import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from astropy.coordinates import AltAz, get_sun, get_body

from ..astrometry import MMT_LOCATION

from .config import (
    ALCOR_SB_APERTURE_RADIUS, ALCOR_SB_BEST_COLUMNS,
    ALCOR_SB_BEST_MIN_ALTITUDE, ALCOR_SB_TARGETS
)
from .timeutils import (
    _alcor_frame_calibration, _alcor_frame_time, _read_frame_exposure,
    select_dark_frames
)
from .wcs import build_alcor_wcs
from .masks import _apply_badpix_repair, load_alcor_horizon_mask
from .badpix import (
    _median_stack_tiles, alcor_badpix_search_region, build_alcor_badpix_mask
)
from .io import load_alcor_fits
from .photometry import alcor_star_photometry, collect_alcor_photometry
from .skybright import (
    _alcor_best_cone_targets, _alcor_cone_indices, _alcor_sb_fits_header,
    _alcor_sky_brightness_map, _cone_median
)
from .keogram import (
    _keogram_row_altitude, save_alcor_keogram_fits, save_alcor_keogram_plot,
    save_alcor_sb_keogram_fits, save_alcor_sb_keogram_plot
)


def _alcor_frame_stem(filename):
    """Path string of a frame with its (possibly compressed) FITS suffix removed."""
    stem = str(filename)
    for ext in (".fits.bz2", ".fits.gz", ".fits"):
        if stem.endswith(ext):
            return stem[: -len(ext)]
    return stem



# Per-worker geometry, installed once by _init_night_worker: the sampling-cone
# pixel indices and the horizon mask both depend only on the night's WCS, so they
# are built once in the parent instead of per frame.
_NIGHT_CONES = None

_NIGHT_HORIZON = None

_NIGHT_BEST_CONES = None

_NIGHT_BEST_TARGETS = None



def _init_night_worker(cones, horizon, best_cones=None, best_targets=None):
    global _NIGHT_CONES, _NIGHT_HORIZON, _NIGHT_BEST_CONES, _NIGHT_BEST_TARGETS
    _NIGHT_CONES = cones
    _NIGHT_HORIZON = horizon
    _NIGHT_BEST_CONES = best_cones or {}
    _NIGHT_BEST_TARGETS = best_targets or {}



def _darkest_cone(mu):
    """
    Darkest 5 deg cone above ALCOR_SB_BEST_MIN_ALTITUDE, as
    ``(brightness, az, alt)``.

    The zenith is contaminated whenever the Milky Way transits it, so the
    darkest patch of sky is what actually measures how dark the site got. This
    reports the same statistic as the fixed cones -- a median inside a 5 deg
    aperture -- at whichever candidate position is faintest, rather than the
    single darkest pixel, which would just track the noise floor (stars only
    push pixels brighter, so the extreme dark tail is read noise).
    """
    best = best_az = best_alt = float("nan")
    for name, idx in _NIGHT_BEST_CONES.items():
        value = _cone_median(mu, idx)
        if np.isfinite(value) and (not np.isfinite(best) or value > best):
            best = value
            best_az, best_alt = _NIGHT_BEST_TARGETS[name]
    return best, best_az, best_alt



def _process_night_frame(task):
    """
    Process one night frame: photometry, sky brightness, and stack slot.

    The frame is decompressed exactly once. The raw cube goes to the median-stack
    memmap slot (bad-pixel repair would erase the very pixels the stack exists to
    track), and the repaired cube feeds star photometry, the surface-brightness
    map, and the raw RGB keogram column. Returns
    ``(index, values, column, rgb_column, exposure, stacked, error)``; on failure
    the values and columns are NaN and ``error`` is the message, so one bad frame
    does not abort the night.

    Star photometry is skipped when ``phot_out`` already exists and is non-empty
    (something measured this frame already -- eventually the real-time ingest),
    unless ``reprocess`` is set. The frame is still read, because the
    surface-brightness map and the keogram columns need its pixels and the
    per-frame CSV does not carry them.
    """
    index, filename, opts = task
    filename = Path(filename)
    zcol = opts["zcol"]
    ny = opts["shape"][1]
    nan_values = {name: float("nan") for name in _NIGHT_CONES}
    nan_values.update({name: float("nan") for name in ALCOR_SB_BEST_COLUMNS})
    nan_column = np.full(ny, np.nan, dtype=np.float32)
    nan_rgb = np.full((ny, 3), np.nan, dtype=np.float32) if opts["rgb_column"] else None

    try:
        cube_raw, wcs, mask = load_alcor_fits(filename, badpix=None,
                                              masks_dir=opts["masks_dir"])

        stacked = False
        stack = opts["stack"]
        if stack is not None and cube_raw.shape == tuple(opts["shape"]):
            cube_mm = np.memmap(stack["path"], dtype=np.uint16, mode="r+",
                                shape=tuple(stack["shape"]))
            cube_mm[index] = np.clip(cube_raw, 0, 65535).astype(np.uint16)
            cube_mm.flush()
            del cube_mm
            stacked = True

        cube = _apply_badpix_repair(cube_raw, mask) if mask is not None else cube_raw

        if opts["phot_out"] is not None and not _photometry_is_done(
                opts["phot_out"], opts["reprocess"]):
            alcor_star_photometry(filename, output_file=opts["phot_out"],
                                  frame=(cube, wcs, mask), **opts["phot_kwargs"])

        time = _alcor_frame_time(filename)
        exposure = _read_frame_exposure(filename)
        mu_full, _ = _alcor_sky_brightness_map(cube, wcs, time, exposure,
                                               saturation=opts["sb_saturation"])
        mu = (np.where(_NIGHT_HORIZON, np.nan, mu_full)
              if _NIGHT_HORIZON is not None else mu_full)

        if opts["sb_out"] is not None:
            header = _alcor_sb_fits_header(wcs, time, exposure,
                                           opts["sb_saturation"],
                                           _NIGHT_HORIZON is not None)
            fits.PrimaryHDU(data=mu.astype(np.float32), header=header).writeto(
                opts["sb_out"], overwrite=opts["overwrite"])

        values = {name: _cone_median(mu, idx) for name, idx in _NIGHT_CONES.items()}
        best, best_az, best_alt = _darkest_cone(mu)
        values["allsky_mv_best"] = best
        values["best_az"] = best_az
        values["best_alt"] = best_alt
        # The keogram column comes from the UNMASKED map: the whole column is
        # inside the illuminated field (its far end is ~711 px from the zenith
        # against a horizon_radius of 747), so it runs a few degrees BELOW the
        # horizon at both ends -- which is exactly where the light domes are, and
        # the point of a calibrated keogram is to show where the light is coming
        # from. The cone medians above are unaffected: _alcor_cone_indices built
        # their index sets with exclude=horizon, so terrain cannot reach them.
        column = mu_full[:, zcol].astype(np.float32)
        # The RGB keogram column is free here: same cube, same zenith column.
        rgb_column = (cube[:, :, zcol].T.astype(np.float32)
                      if opts["rgb_column"] else None)
        return index, values, column, rgb_column, exposure, stacked, None
    except Exception as exc:                                   # noqa: BLE001
        return index, nan_values, nan_column, nan_rgb, float("nan"), False, f"{exc}"



def _photometry_is_done(phot_out, reprocess):
    """True when ``phot_out`` already holds photometry we should not redo."""
    if reprocess:
        return False
    try:
        return Path(phot_out).stat().st_size > 0
    except OSError:
        return False



def _day_keogram_column(task):
    """
    Raw RGB zenith column of one frame, for the daylight half of a day keogram.

    Loads once and takes the night's fixed ``zcol``, so the day keogram and the
    calibrated sky-brightness keogram sample the identical pixel column.
    """
    index, filename, zcol, kwargs = task
    try:
        cube, _, _ = load_alcor_fits(filename, **kwargs)
        return index, cube[:, :, zcol].T.astype(np.float32), None
    except Exception as exc:                                   # noqa: BLE001
        return index, None, f"{exc}"



def _build_day_keogram(frames, night_files, night_times, night_rgb_columns,
                       zcol, max_frames, masks_dir, workers, errors, log):
    """
    Full-day RGB keogram from a day directory, reusing the night's columns.

    A day directory spans local noon to the following morning, so it holds the
    night frames the main pass already loaded plus the daylight remainder. The
    night columns come back free from :func:`_process_night_frame`; only the
    remaining frames need a second, column-only pass. Both halves are merged in
    time order, so the result is directly comparable to the sky-brightness
    keogram, which samples the same ``zcol``.

    Timestamps come from :func:`_alcor_frame_time` (filename first, DATE header
    fallback) rather than the raw DATE header string that :func:`alcor_keogram`
    stores, so both keograms of a run share one time source.

    Returns ``(keogram, timestamps)`` with the keogram shaped
    ``(ny, nframes, 3)``.
    """
    night_set = set(night_files)
    extra = [f for f in frames if f not in night_set]
    if max_frames is not None and len(extra) > max_frames:
        stride = len(extra) // max_frames
        extra = extra[::stride][:max_frames]
    log(f"day keogram: {len(night_files)} night columns reused, "
        f"{len(extra)} daylight frames to load")

    extra_columns = [None] * len(extra)
    tasks = [(index, filename, zcol, {"masks_dir": masks_dir})
             for index, filename in enumerate(extra)]

    def _store(result):
        index, column, error = result
        extra_columns[index] = column
        if error is not None:
            errors.append((extra[index].name, error))
            log(f"{extra[index].name}: {error}")

    if tasks:
        if workers == 1:
            for task in tasks:
                _store(_day_keogram_column(task))
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_day_keogram_column, task)
                           for task in tasks]
                for future in as_completed(futures):
                    _store(future.result())

    entries = [(t, column) for t, column
               in zip(night_times, night_rgb_columns) if column is not None]
    ny = entries[0][1].shape[0] if entries else None
    for filename, column in zip(extra, extra_columns):
        if column is None:                       # failed load, keep the column
            if ny is None:
                continue
            column = np.full((ny, 3), np.nan, dtype=np.float32)
        entries.append((_alcor_frame_time(filename), column))

    entries.sort(key=lambda entry: entry[0].jd)
    keogram = np.stack([column for _, column in entries], axis=1)
    timestamps = [t.isot for t, _ in entries]
    return keogram, timestamps


def alcor_process_night(night_dir, out_dir=None, pattern="*.fits.bz2",
                        sun_alt_max=-12.0, targets=None,
                        sb_aperture_radius=ALCOR_SB_APERTURE_RADIUS,
                        best_min_altitude=ALCOR_SB_BEST_MIN_ALTITUDE,
                        horizon_mask=True, sb_saturation=None,
                        write_sb_fits=False, median_stack=False,
                        day_keogram=False, reprocess=False,
                        max_frames=None, scratch_dir=None, masks_dir=None,
                        workers=None, overwrite=False, log=None, **phot_kwargs):
    """
    Process one archived night of alcor OMEA 8C frames end to end.

    Every frame with the Sun below ``sun_alt_max`` (there is no Moon cut -- the
    Moon's effect on sky brightness is the signal here) is decompressed once and
    turned into three things: fixed-position star photometry
    (:func:`alcor_star_photometry`), a calibrated V mag/arcsec^2 surface-brightness
    map (:func:`_alcor_sky_brightness_map`), and, optionally, a slot in the
    night's raw median stack.

    From the surface-brightness maps the night summary is built: for each entry
    in ``targets`` the median brightness inside a ``sb_aperture_radius``-degree
    cone about a fixed ``(azimuth, altitude)``, so the same patch of sky is
    reported for every frame of every night. The default
    :data:`ALCOR_SB_TARGETS` are the zenith and the Tucson and Nogales light
    domes. The zenith column of every map is also stacked into a calibrated
    nighttime keogram -- the same raw column :func:`alcor_keogram` uses, so the
    calibrated and RGB keograms stack row for row.

    Parameters
    ----------
    night_dir : str or `~pathlib.Path`
        Directory holding one night's frames.
    out_dir : str or `~pathlib.Path` or None (default=None)
        Where every product is written. Defaults to ``night_dir``; pass an
        explicit directory when the archive is read-only or on slow media.
    pattern : str (default="*.fits.bz2")
        Glob pattern selecting frames within `night_dir`.
    sun_alt_max : float (default=-12.0)
        Night is the Sun below this altitude in degrees.
    targets : dict or None (default=None)
        Maps a summary column name to an ``(azimuth, altitude)`` pair in
        degrees. None uses :data:`ALCOR_SB_TARGETS`.
    sb_aperture_radius : float (default ALCOR_SB_APERTURE_RADIUS)
        Angular radius of each sampling cone in degrees.
    best_min_altitude : float (default ALCOR_SB_BEST_MIN_ALTITUDE)
        Altitude floor for the ``allsky_mv_best`` darkest-cone search.
    horizon_mask : bool (default=True)
        Blank not-sky pixels (:func:`load_alcor_horizon_mask`) in the maps, the
        cones, and the keogram, so terrain cannot drag a low-altitude cone faint.
    sb_saturation : int (default ALCOR_SB_SATURATION)
        Raw-ADU level at/above which G pixels are blanked as non-linear.
    write_sb_fits : bool (default=False)
        Also keep each frame's full surface-brightness map as ``<frame>_sb.fits``.
        Off by default: the maps are several MB each and the summary and keogram
        already carry what they are usually wanted for.
    median_stack : bool (default=False)
        Also build the per-channel median of the night's *raw* frames and write
        ``<night>_median.fits``, stamped with the per-channel hot-pixel counts
        from :func:`build_alcor_badpix_mask` so bad-pixel growth can be trended.
        Off by default because it needs scratch space for the whole night
        (~12 MB per frame).
    day_keogram : bool (default=False)
        Also build the full-day raw RGB keogram (``<night>_keogram.fits`` /
        ``.png``). A day directory spans local noon to the following morning, so
        this covers daylight too; the night frames' columns are free from the
        main pass and only the daylight remainder needs a second, column-only
        read. Off by default so a photometry run never touches daylight frames.
    reprocess : bool (default=False)
        Re-measure star photometry even where ``<frame>_phot.csv`` already
        exists. By default an existing non-empty CSV is left alone and reused --
        the frame is still read, because the surface-brightness map and the
        keogram columns need its pixels. Pass this after changing photometry
        options, since a CSV written in another mode is otherwise reused as-is.
    max_frames : int or None (default=None)
        Strided-subsample the night to at most this many frames (and, for the day
        keogram, the daylight remainder to at most this many as well).
    scratch_dir : str or None (default=None)
        Directory for the median-stack memmap (default: the system temp dir).
    masks_dir : str or None (default=None)
        Override the bad-pixel masks directory.
    workers : int or None (default=None)
        Worker processes for the per-frame pass. 1 runs serially; None uses one
        per core.
    overwrite : bool (default=False)
        Overwrite existing output files.
    log : callable or None (default=None)
        Called with progress messages. None is silent.
    **phot_kwargs
        Forwarded to :func:`alcor_star_photometry` (``aperture_radius``,
        ``annulus_width``, ``min_altitude``, ``vmag_limit``, ``gaussian``,
        ``both``, ...). Its ``sun_alt_max`` is pinned to this function's, so it
        never rejects a frame the night selection accepted.

    Returns
    -------
    dict
        ``summary`` (the `~pandas.DataFrame` written to ``sky_brightness.csv``),
        ``keogram`` (the ``(ny, nframes)`` float32 sky-brightness array),
        ``timestamps``, ``day_keogram`` (the ``(ny, nframes, 3)`` RGB array, or
        None), ``files`` (the night frames used, in column order), ``errors`` (a
        list of ``(filename, message)`` for frames that failed), and the paths
        written: ``summary_file``, ``keogram_file``, ``keogram_plot``,
        ``day_keogram_file``, ``day_keogram_plot``, ``photometry_file``,
        ``median_file``.

    Raises
    ------
    FileNotFoundError
        If `pattern` matches nothing in `night_dir`.
    ValueError
        If no frame in `night_dir` has the Sun below `sun_alt_max`.
    """
    def _log(message):
        if log is not None:
            log(message)

    night_dir = Path(night_dir)
    night_name = night_dir.resolve().name
    out_dir = night_dir if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if targets is None:
        targets = ALCOR_SB_TARGETS

    frames = sorted(night_dir.glob(pattern))
    if not frames:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {night_dir}")
    # No Moon cut: moonlit sky brightness is exactly what this measures.
    files_ = select_dark_frames(frames, sun_alt_max=sun_alt_max,
                                moon_alt_max=90.0, log=None)
    if not files_:
        raise ValueError(f"No frames in {night_dir} have the Sun below "
                         f"{sun_alt_max:g} deg")
    if max_frames is not None and len(files_) > max_frames:
        stride = len(files_) // max_frames
        files_ = files_[::stride][:max_frames]
    nframes = len(files_)
    _log(f"{nframes} of {len(frames)} frames are night "
         f"(Sun below {sun_alt_max:g} deg)")

    # Ephemeris for the whole night in one vectorised pass.
    frame_times = [_alcor_frame_time(f) for f in files_]
    if any(t is None for t in frame_times):
        missing = [f.name for f, t in zip(files_, frame_times) if t is None]
        raise ValueError(f"could not determine a frame time for {missing}")
    times = Time(frame_times)
    # Keogram columns follow file order and summary rows follow time order, so
    # put the frames in time order once and keep the two products aligned.
    order = np.argsort(times.jd)
    files_ = [files_[i] for i in order]
    times = times[order]

    altaz = AltAz(obstime=times, location=MMT_LOCATION)
    sun_alt = get_sun(times).transform_to(altaz).alt.deg
    moon = get_body("moon", times, MMT_LOCATION).transform_to(altaz)

    # The geometry is fixed for the night, so resolve it once from the first frame.
    header0 = fits.getheader(files_[0])
    shape = (int(header0["NAXIS3"]), int(header0["NAXIS2"]), int(header0["NAXIS1"]))
    cal = _alcor_frame_calibration(files_[0])
    wcs = build_alcor_wcs(xcen=cal["xcen"], ycen=cal["ycen"],
                          rotation=cal["rotation"],
                          radial_coeffs=cal["radial_coeffs"],
                          horizon_radius=cal["horizon_radius"],
                          tangential_coeffs=cal["tangential_coeffs"],
                          axis_tilt=cal["axis_tilt"])

    horizon = None
    if horizon_mask:
        horizon, horizon_date = load_alcor_horizon_mask(times[0])
        if horizon is not None and horizon.shape != shape[1:]:
            _log(f"horizon mask shape {horizon.shape} does not match "
                 f"{shape[1:]}; not applied")
            horizon = None
        elif horizon is None:
            _log("no horizon mask found; not applied")
        else:
            _log(f"using horizon mask for {horizon_date}")

    cones = _alcor_cone_indices(wcs, shape[1:], targets=targets,
                                radius_deg=sb_aperture_radius, exclude=horizon)
    for name, idx in cones.items():
        _log(f"{name}: {idx.size} pixels within {sb_aperture_radius:g} deg "
             f"of az={targets[name][0]:g}, alt={targets[name][1]:g}")

    # Floating darkest-cone search: candidates tiling the sky above the floor,
    # built once for the night like the fixed cones. Drop any the horizon mask or
    # the frame edge leaves empty so the search never sees an all-NaN cone.
    best_targets = _alcor_best_cone_targets(radius_deg=sb_aperture_radius,
                                            min_altitude=best_min_altitude)
    best_cones = _alcor_cone_indices(wcs, shape[1:], targets=best_targets,
                                     radius_deg=sb_aperture_radius,
                                     exclude=horizon)
    best_cones = {name: idx for name, idx in best_cones.items() if idx.size}
    best_targets = {name: best_targets[name] for name in best_cones}
    _log(f"allsky_mv_best: darkest of {len(best_cones)} cones above "
         f"alt {best_min_altitude:g} deg")
    zx, _ = wcs.world_to_pixel_values(0.0, 90.0)
    zcol = int(np.clip(round(float(zx)), 0, shape[2] - 1))

    memmap_path = None
    stack = None
    try:
        if median_stack:
            tmp = tempfile.NamedTemporaryFile(
                prefix="alcor_night_", suffix=".dat",
                dir=scratch_dir or tempfile.gettempdir(), delete=False)
            tmp.close()
            memmap_path = Path(tmp.name)
            stack_shape = (nframes,) + shape
            cube_mm = np.memmap(memmap_path, dtype=np.uint16, mode="w+",
                                shape=stack_shape)
            del cube_mm
            stack = {"path": str(memmap_path), "shape": stack_shape}
            _log(f"median stack scratch: {memmap_path} "
                 f"({np.prod(stack_shape, dtype=float) * 2 / 1e9:.1f} GB)")

        phot_kwargs = dict(phot_kwargs)
        phot_kwargs["sun_alt_max"] = sun_alt_max
        phot_kwargs["masks_dir"] = masks_dir
        tasks = []
        for index, filename in enumerate(files_):
            stem = Path(_alcor_frame_stem(filename)).name
            tasks.append((index, filename, {
                "zcol": zcol,
                "shape": shape,
                "masks_dir": masks_dir,
                "phot_out": out_dir / f"{stem}_phot.csv",
                "sb_out": (out_dir / f"{stem}_sb.fits") if write_sb_fits else None,
                "sb_saturation": sb_saturation,
                "rgb_column": day_keogram,
                "reprocess": reprocess,
                "overwrite": overwrite,
                "phot_kwargs": phot_kwargs,
                "stack": stack,
            }))

        values = [None] * nframes
        columns = [None] * nframes
        rgb_columns = [None] * nframes
        exposures = np.full(nframes, np.nan)
        stacked = np.zeros(nframes, dtype=bool)
        errors = []

        def _collect(result):
            index, vals, column, rgb_column, exposure, ok, error = result
            values[index] = vals
            columns[index] = column
            rgb_columns[index] = rgb_column
            exposures[index] = exposure
            stacked[index] = ok
            if error is not None:
                errors.append((files_[index].name, error))
                _log(f"{files_[index].name}: {error}")

        if workers == 1:
            _init_night_worker(cones, horizon, best_cones, best_targets)
            for done, task in enumerate(tasks, start=1):
                _collect(_process_night_frame(task))
                _log(f"[{done}/{nframes}] {files_[task[0]].name}")
        else:
            with ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_night_worker,
                                     initargs=(cones, horizon, best_cones,
                                               best_targets)) as executor:
                futures = [executor.submit(_process_night_frame, task)
                           for task in tasks]
                for done, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    _collect(result)
                    _log(f"[{done}/{nframes}] {files_[result[0]].name}")

        summary = pd.DataFrame({
            "filename": [f.name for f in files_],
            "OBSTIME": pd.to_datetime(times.isot),
            "exposure": exposures,
            "sun_alt": sun_alt,
            "moon_alt": moon.alt.deg,
            "moon_az": moon.az.deg,
        })
        for name in list(targets) + list(ALCOR_SB_BEST_COLUMNS):
            summary[name] = [vals[name] for vals in values]
        summary = summary.sort_values("OBSTIME", ignore_index=True)
        summary_file = out_dir / "sky_brightness.csv"
        summary.to_csv(summary_file, index=False)
        _log(f"wrote {summary_file}")

        keogram = np.stack(columns, axis=1)
        timestamps = list(times.isot)
        # Both keograms are the same raw column, so one altitude array serves.
        row_altitude = _keogram_row_altitude(wcs, shape[1], zcol)
        keogram_file = out_dir / f"{night_name}_sb_keogram.fits"
        save_alcor_sb_keogram_fits(keogram, timestamps, keogram_file,
                                   overwrite=True, altitude=row_altitude)
        keogram_plot = out_dir / f"{night_name}_sb_keogram.png"
        save_alcor_sb_keogram_plot(keogram, timestamps, keogram_plot,
                                   altitude=row_altitude)
        _log(f"wrote {keogram_file} and {keogram_plot}")

        day_keogram_array = None
        day_keogram_file = None
        day_keogram_plot = None
        if day_keogram:
            day_keogram_array, day_times = _build_day_keogram(
                frames, files_, times, rgb_columns, zcol, max_frames,
                masks_dir, workers, errors, _log)
            day_keogram_file = out_dir / f"{night_name}_keogram.fits"
            save_alcor_keogram_fits(day_keogram_array, day_times,
                                    day_keogram_file, overwrite=True,
                                    altitude=row_altitude)
            day_keogram_plot = out_dir / f"{night_name}_keogram.png"
            save_alcor_keogram_plot(day_keogram_array, day_times,
                                    day_keogram_plot, altitude=row_altitude)
            _log(f"wrote {day_keogram_file} and {day_keogram_plot}")

        photometry_file = None
        try:
            phot = collect_alcor_photometry([t[2]["phot_out"] for t in tasks
                                             if t[2]["phot_out"].exists()])
        except ValueError as exc:
            _log(f"no combined photometry written: {exc}")
        else:
            photometry_file = out_dir / f"{night_name}_phot.csv"
            phot.to_csv(photometry_file, index=False)
            _log(f"wrote {photometry_file}")

        median_file = None
        if median_stack:
            ok = np.flatnonzero(stacked)
            if ok.size == 0:
                _log("no frames stacked; no median written")
            else:
                cube_mm = np.memmap(memmap_path, dtype=np.uint16, mode="r",
                                    shape=stack["shape"])
                selection = (slice(0, nframes) if ok.size == nframes
                             else ok)
                median = _median_stack_tiles(cube_mm, selection, shape)
                del cube_mm
                median_file = out_dir / f"{night_name}_median.fits"
                mask = build_alcor_badpix_mask(
                    median,
                    valid=alcor_badpix_search_region(
                        shape[1:], time=times[0], wcs=wcs))
                mheader = wcs.to_header(relax=True)
                mheader["BUNIT"] = ("adu", "raw counts, per-channel median")
                mheader["NSTACK"] = (int(ok.size), "frames in the median")
                mheader["SUNALT"] = (sun_alt_max, "night definition (deg)")
                for c, channel in enumerate("RGB"):
                    mheader[f"NBAD{channel}"] = (
                        int(mask[c].sum()), f"{channel} hot pixels in this median")
                fits.PrimaryHDU(data=median, header=mheader).writeto(
                    median_file, overwrite=True)
                _log(f"wrote {median_file} "
                     f"(hot pixels R/G/B: {mask[0].sum()}/{mask[1].sum()}/"
                     f"{mask[2].sum()})")
    finally:
        if memmap_path is not None:
            memmap_path.unlink(missing_ok=True)

    return {
        "summary": summary,
        "summary_file": summary_file,
        "keogram": keogram,
        "keogram_file": keogram_file,
        "keogram_plot": keogram_plot,
        "timestamps": timestamps,
        "day_keogram": day_keogram_array,
        "day_keogram_file": day_keogram_file,
        "day_keogram_plot": day_keogram_plot,
        "photometry_file": photometry_file,
        "median_file": median_file,
        "files": files_,
        "errors": errors,
    }
