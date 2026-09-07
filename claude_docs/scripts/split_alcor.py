"""One-shot: split the monolithic ``skycam_utils/alcor.py`` into a package.

Pure code motion. Every top-level statement of the original module is assigned
to exactly one submodule and moved VERBATIM -- the script slices source lines
rather than unparsing the AST, so comments, blank lines, formatting and
docstrings survive byte-for-byte. Each node absorbs every line between the
previous node's end and its own, which guarantees total coverage: no comment
can fall between two symbols and be lost.

Per-module imports are recomputed from what each module actually references
(the original module imported one flat block for all 5700 lines), and
cross-module references become explicit relative imports.

Kept only as the record of how the split was produced; it is not rerunnable
against the package it created.
"""

import ast
from pathlib import Path

SRC = Path("skycam_utils/alcor.py")
PKG = Path("skycam_utils/alcor")

# symbol -> module. Order here is the order modules are laid out; order within
# a module always follows the original file, so module-level evaluation that
# depends on an earlier definition (_LATEST_CALIBRATION) still works.
LAYOUT = {
    "config": """
        ALCOR_RADIUS ALCOR_HORIZON_RADIUS ALCOR_SATURATION
        ALCOR_NONLINEAR_THRESHOLD ALCOR_CALIB_EXPTIME ALCOR_SB_SATURATION
        ALCOR_BADPIX_RIM_DILATION ALCOR_FIELD_RADIUS ALCOR_COLOR_FLAT_TOL
        ALCOR_BADPIX_POLE_RADIUS ALCOR_SB_APERTURE_RADIUS ALCOR_SB_TARGETS
        ALCOR_SB_TARGET_DESCRIPTIONS ALCOR_SB_TARGET_UNIT
        ALCOR_SB_BEST_MIN_ALTITUDE ALCOR_AIRMASS_TERM ALCOR_BRIGHT_CUT
        _GAUSS_MIN_LUM_PIXELS _GAUSS_MIN_CHANNEL_PIXELS ALCOR_CALIBRATIONS
        _calibration_epochs alcor_calibration _LATEST_CALIBRATION
        ALCOR_ROTATION ALCOR_XCEN ALCOR_YCEN ALCOR_RADIAL_COEFFS
        ALCOR_TANGENTIAL_COEFFS ALCOR_AXIS_TILT ALCOR_ZEROPOINTS
        ALCOR_ZEROPOINT_BANDS alcor_zeropoint ALCOR_PRESSURE ALCOR_TEMPERATURE
        ALCOR_HUMIDITY ALCOR_OBSWL ALCOR_SB_BEST_COLUMNS
    """,
    "timeutils": """
        _sun_altitude _moon_altitude _FILENAME_TIME_RE _MST_TO_UT
        _filename_ut_datetime _read_frame_date _read_frame_exposure
        _alcor_frame_calibration _alcor_frame_time select_dark_frames
        _frame_time
    """,
    "wcs": """
        _invert_radial _axis_frame _tangential_delta _predict_pixels
        _base_arc_wcs _sip_poly_eval _fit_sip_inverse build_alcor_wcs
        _build_alcor_wcs_cached _alcor_pixel_solid_angle
    """,
    "wcsfit": """
        detect_alcor_stars _fit_params assign_alcor_matches _detect_alcor_frame
        fit_alcor_wcs save_alcor_residual_plot _format_calibration_entry
    """,
    "masks": """
        _BADPIX_DATE_RE _resolve_badpix_dir _badpix_date_from_dir
        load_alcor_badpix_mask _apply_badpix_repair _HORIZON_DATE_RE
        _resolve_horizon_dir load_alcor_horizon_mask
    """,
    "badpix": """
        alcor_badpix_search_region build_alcor_badpix_mask _median_stack_tiles
        build_alcor_median_stack create_badpix_mask
    """,
    "horizon": """
        build_alcor_luminance_median alcor_median_stack build_alcor_horizon_mask
        _alcor_undetected_fraction _horizon_epoch create_horizon_mask
    """,
    "catalogs": """
        alcor_reference_altaz alcor_named_reference_altaz
        _catalog_value_to_python lookup_sloan_photometry
        alcor_variable_reference_altaz alcor_photometry_reference_altaz
        _alcor_star_labels
    """,
    "io": "load_alcor_fits alcor_proc_fits _corner_bias",
    "photometry": """
        _annulus_background _aperture_annulus_photometry
        _aperture_saturated _gaussian_channel_amplitude _gaussian_psf_photometry
        _default_alcor_photometry_output
        _default_alcor_photometry_check_plot_output _flux_mag _aperture_measure
        _gaussian_measure _airmass _catalog_calibration_map
        _zeropoint_row_params alcor_calibrate_photometry alcor_star_photometry
        collect_alcor_photometry
    """,
    "display": """
        _alcor_display_rgb _alcor_zenith_crop_bounds _add_alcor_alt_az_grid
        plot_alcor_fits save_alcor_photometry_check_plot
    """,
    "skybright": """
        _alcor_sky_brightness_map _alcor_best_cone_targets _alcor_cone_indices
        plot_alcor_sky_brightness _alcor_sb_fits_header alcor_sky_brightness_fits
        _cone_median
    """,
    "keogram": """
        alcor_keogram _load_alcor_center_column _print_progress
        save_alcor_keogram_plot _keogram_row_altitude _keogram_horizon_rows
        _set_keogram_yaxis _timestamp_edges _row_altitude_hdu _load_row_altitude
        save_alcor_keogram_fits load_alcor_keogram_fits plot_alcor_keogram_fits
        save_alcor_sb_keogram_fits load_alcor_sb_keogram_fits _sb_keogram_limits
        save_alcor_sb_keogram_plot plot_alcor_sb_keogram_fits _parse_timestamps
    """,
    "night": """
        _alcor_frame_stem _NIGHT_CONES _NIGHT_HORIZON _NIGHT_BEST_CONES
        _NIGHT_BEST_TARGETS _init_night_worker _darkest_cone _process_night_frame
        _photometry_is_done _day_keogram_column _build_day_keogram
        alcor_process_night
    """,
    "cli": """
        alcor_proc_fits_cli alcor_keogram_cli plot_alcor_keogram_fits_cli
        plot_alcor_fits_cli plot_alcor_sky_brightness_cli
        alcor_sky_brightness_cli plot_alcor_sb_keogram_fits_cli
        alcor_process_night_cli alcor_star_photometry_cli fit_alcor_wcs_cli
        create_badpix_mask_cli alcor_median_stack_cli create_horizon_mask_cli
    """,
}

DOCSTRINGS = {
    "config": "Camera constants, per-epoch WCS calibrations, and photometric zeropoints.",
    "timeutils": "Frame timestamps, Sun/Moon altitude, and dark-frame selection.",
    "wcs": "Raw-frame ARC alt/az WCS construction and the lens-distortion model.",
    "wcsfit": "Star detection, catalog matching, and the WCS geometry fit.",
    "masks": "Loading the date-resolved bad-pixel and horizon mask assets.",
    "badpix": "Building bad-pixel masks from a night-median stack.",
    "horizon": "Building the sky/not-sky horizon mask from a cloudy-night median.",
    "catalogs": "Bright-star and bright-variable catalogs in alt/az at a given time.",
    "io": "Reading Alcor RGB FITS frames, the corner bias, and the processed cube.",
    "photometry": "Fixed-position RGB aperture and Gaussian PSF stellar photometry.",
    "display": "Rendering annotated all-sky RGB images.",
    "skybright": "Calibrated V mag/arcsec^2 surface-brightness maps and sampling cones.",
    "keogram": "Raw RGB and calibrated surface-brightness keograms.",
    "night": "The night-level archive driver that produces a night's data products.",
    "cli": "Console-script entry points for the Alcor tools.",
}

symbol_module = {}
for mod, names in LAYOUT.items():
    for name in names.split():
        assert name not in symbol_module, f"{name} assigned twice"
        symbol_module[name] = mod

lines = SRC.read_text().splitlines(keepends=True)
tree = ast.parse("".join(lines))


def node_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


# --- partition the source, absorbing every line between nodes ---------------
imports, chunks, cursor = [], {m: [] for m in LAYOUT}, 0
for node in tree.body:
    end = node.end_lineno
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        imports.append(node)
        cursor = end
        continue
    names = node_names(node)
    assert names, f"unhandled top-level node at line {node.lineno}"
    mods = {symbol_module[n] for n in names if n in symbol_module}
    assert len(mods) == 1, f"{names} -> {mods or 'unassigned'}"
    chunks[mods.pop()].append("".join(lines[cursor:end]))
    cursor = end
assert cursor == len(lines), f"trailing lines {cursor}..{len(lines)} unassigned"


def referenced(text):
    """Every bare name loaded anywhere in this source."""
    used = set()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Name):
            used.add(n.id)
        elif isinstance(n, ast.Attribute):
            base = n
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                used.add(base.id)
    return used


def import_for(node, used):
    """Rebuild an import statement with only the names this module uses."""
    if isinstance(node, ast.Import):
        keep = [a for a in node.names if (a.asname or a.name.split(".")[0]) in used]
        if not keep:
            return None
        return "\n".join(
            f"import {a.name}" + (f" as {a.asname}" if a.asname else "") for a in keep
        )
    keep = [a for a in node.names if (a.asname or a.name) in used]
    if not keep:
        return None
    shown = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)
    dots = "." * node.level
    return f"from {dots}{node.module or ''} import {shown}"


# --- dependency graph, checked for cycles before anything is written -------
edges = {}
for mod in LAYOUT:
    body = "\n".join(chunks[mod])
    used = referenced(body)
    edges[mod] = {symbol_module[n] for n in used
                  if n in symbol_module and symbol_module[n] != mod}

state = {}
def visit(m, stack):
    if state.get(m) == "done":
        return
    if state.get(m) == "open":
        raise SystemExit(f"CYCLE: {' -> '.join(stack[stack.index(m):] + [m])}")
    state[m] = "open"
    for dep in sorted(edges[m]):
        visit(dep, stack + [m])
    state[m] = "done"

for mod in LAYOUT:
    visit(mod, [])
print("dependency graph is acyclic:")
for mod in LAYOUT:
    print(f"  {mod:12s} -> {', '.join(sorted(edges[mod])) or '(none)'}")
print()

def wrap_import(prefix, names, width=79):
    """``from .x import a, b`` on one line, or parenthesized if it is too long."""
    flat = prefix + ", ".join(names)
    if len(flat) <= width:
        return flat
    out, line = [], "    "
    for i, name in enumerate(names):
        piece = name + ("," if i < len(names) - 1 else "")
        if len(line) + len(piece) > width - 1:
            out.append(line.rstrip())
            line = "    "
        line += piece + " "
    out.append(line.rstrip())
    return prefix + "(\n" + "\n".join(out) + "\n)"


# the original import block is grouped stdlib / third-party / package by blank
# lines; recover those groups from the line numbers so each module keeps them
groups, current, prev = [], [], None
for node in imports:
    if prev is not None and node.lineno > prev + 1:
        groups.append(current)
        current = []
    current.append(node)
    prev = node.end_lineno
groups.append(current)

PKG.mkdir(exist_ok=True)
for mod in LAYOUT:
    body = "\n".join(chunks[mod]).strip("\n")
    used = referenced(body)
    defined = {n for n, m in symbol_module.items() if m == mod}

    blocks = []
    for group in groups:
        kept = [s for s in (import_for(n, used) for n in group) if s]
        kept = [s.replace("from .astrometry", "from ..astrometry") for s in kept]
        if kept:
            blocks.append(kept)

    local = {}
    for name in sorted(used - defined):
        owner = symbol_module.get(name)
        if owner and owner != mod:
            local.setdefault(owner, []).append(name)
    order = list(LAYOUT)
    if local:
        blocks.append([wrap_import(f"from .{m} import ", sorted(local[m]))
                       for m in sorted(local, key=order.index)])

    out = [f'"""{DOCSTRINGS[mod]}"""']
    for block in blocks:
        out.append("")
        out += block
    out += ["", "", body, ""]
    (PKG / f"{mod}.py").write_text("\n".join(out))
    print(f"{mod:12s} {len(body.splitlines()):5d} lines")

# --- __init__ re-exports the original namespace verbatim -------------------
init = ['"""Utilities for the Alcor OMEA 8C all-sky camera at the MMT.',
        "",
        "This package was split out of a single ``alcor.py``; the module namespace is",
        "re-exported here unchanged, so ``from skycam_utils.alcor import <name>`` and",
        "the ``skycam_utils.alcor:*_cli`` console-script entry points keep working.",
        '"""', ""]
public = []
for mod in LAYOUT:
    names = LAYOUT[mod].split()
    init.append(f"from .{mod} import (  # noqa: F401")
    for name in names:
        init.append(f"    {name},")
    init.append(")")
    public += [n for n in names if not n.startswith("_")]
init += ["", "__all__ = ["] + [f'    "{n}",' for n in sorted(public)] + ["]", ""]
(PKG / "__init__.py").write_text("\n".join(init))
print(f"__init__     {len(public)} public names re-exported")
