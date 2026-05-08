#!/usr/bin/env python3
"""
Unified OBD explorer entrypoint: Qt GUI, HTML generation, headless PNG/PDF export, interactive menu.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import webbrowser

from obd_explorer.constants import DEFAULT_GRAPH_P_STEPS
from obd_explorer.model import TIE_COLOR_AXIS_CHOICES

# HTML colorscale choices: match GUI colormap names (see qt_graphics.CMAP_NAMES).
HTML_COLOR_SCALE_CHOICES: tuple[str, ...] = (
    "gist_rainbow",
    "turbo",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "hsv",
    "jet",
    "nipy_spectral",
    "twilight",
    "coolwarm",
    "winter",
    "spring",
    "summer",
    "autumn",
    "wistia",
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "bwr",
    "seismic",
    "berlin",
    "managua",
    "vanimo",
    "flag",
    "prism",
    "ocean",
    "gist_earth",
    "terrain",
    "gist_stern",
    "gnuplot",
    "gnuplot2",
    "CMRmap",
    "cubehelix",
    "brg",
    "rainbow",
    "gist_ncar",
)

HEATMAP_VALUE_CHOICES: tuple[str, ...] = ("i", "j", "l", "r", "d", "e", "ev_n", "eslope_n")
TIE_HEATMAP_VALUE_CHOICES: tuple[str, ...] = ("i", "j", "l", "r", "d", "e", "ev_n")
HEATMAP_PIXEL_MODE_CHOICES: tuple[str, ...] = ("exact", "annotated")


def _effective_default_graph_shards_dir() -> str:
    return os.path.join("data", "graph_data_shards")


def _effective_default_tie_manifest() -> str:
    return os.path.join("data", "tie_points_shards", "0000_manifest.pkl")


def _effective_graph_manifest_for_cfg(cfg: dict[str, object]) -> str:
    shards_dir = cfg.get("graph_shards_dir")
    if not isinstance(shards_dir, str) or not shards_dir.strip():
        shards_dir = _effective_default_graph_shards_dir()
    p_steps = int(cfg.get("p_steps", DEFAULT_GRAPH_P_STEPS))
    return os.path.join(shards_dir, f"0000_manifest_p{p_steps:05d}.pkl")


def _ensure_output_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _resolve_html_output_path(path: str) -> str:
    """Put relative paths with no directory under ``html/``; keep explicit dirs and absolute paths."""
    raw = path.strip()
    if not raw:
        return raw
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    norm = os.path.normpath(raw)
    dn = os.path.dirname(norm)
    if dn in ("", "."):
        return os.path.join("html", os.path.basename(norm))
    return norm


def _default_html_output_for_variant(variant: int) -> str:
    return os.path.join("html", f"OBDExplorer{variant}.html")


def _resolved_export_output_path(output: str, fmt: str) -> str:
    """Return path with ``.<fmt>``. If basename ends with .png/.pdf/.svg (any case), that suffix is
    replaced so the extension always matches the chosen export format."""
    f = fmt.strip().lower()
    raw = output.strip()
    root, ext = os.path.splitext(raw)
    if ext.lower() in (".png", ".pdf", ".svg"):
        base = root
    else:
        base = raw
    return base + "." + f


def _render_setting_value(key: str, value: object, cfg: dict[str, object]) -> object:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if key == "graph_shards_dir" and value is None:
        return f"{_effective_default_graph_shards_dir()} (default)"
    if key == "tie_manifest" and value is None:
        return f"{_effective_default_tie_manifest()} (default)"
    if key == "graph_manifest" and value is None:
        return f"{_effective_graph_manifest_for_cfg(cfg)} (auto)"
    return value


def _parse_bool(text: str) -> bool:
    """Accept yes/no, true/false, y/n/t/f, any non-empty prefix of those words (case-insensitive), plus 1/0/on/off."""
    norm = text.strip().lower()
    if not norm:
        raise ValueError("Empty input.")
    if norm in ("1", "on"):
        return True
    if norm in ("0", "off"):
        return False
    if "yes".startswith(norm):
        return True
    if "true".startswith(norm):
        return True
    if "no".startswith(norm):
        return False
    if "false".startswith(norm):
        return False
    raise ValueError(
        'Expected "yes" or "no" (or true/false); prefixes such as y, n, t, f are accepted.'
    )


def _parse_opt_str(text: str) -> str | None:
    v = text.strip()
    if v == "" or v.lower() in ("none", "null"):
        return None
    return v


def _parse_choice(text: str, choices: tuple[str, ...], *, field: str) -> str:
    """Resolve input to one of ``choices``: case-insensitive exact match, otherwise require that some
    choice begins with the input (the input is a prefix of that choice). Multiple matches at the same
    minimum length raise ValueError (type more characters to disambiguate)."""
    raw = text.strip()
    if not raw:
        raise ValueError("Empty input.")
    norm = raw.lower()
    lowered = [c.lower() for c in choices]
    for i, lc in enumerate(lowered):
        if norm == lc:
            return choices[i]
    by_prefix = [choices[i] for i, lc in enumerate(lowered) if lc.startswith(norm)]
    if len(by_prefix) == 1:
        return by_prefix[0]
    if len(by_prefix) > 1:
        min_len = min(len(c) for c in by_prefix)
        shortest = [c for c in by_prefix if len(c) == min_len]
        if len(shortest) == 1:
            return shortest[0]
        raise ValueError(
            f'{field}: ambiguous prefix {raw!r}; type more characters to pick among '
            f'{", ".join(sorted(by_prefix))}.'
        )
    raise ValueError(
        f'{field}: no match for {raw!r}; expected one of {", ".join(choices)} '
        "(case-insensitive; enter a choice or any non-empty prefix of one)."
    )


def _parse_tie_direction(text: str) -> str:
    from obd_explorer.geometry import parse_tie_lines_direction

    s = _parse_choice(text, ("up", "down"), field="tie_direction")
    return parse_tie_lines_direction(s)


def _parse_tie_color_axis(text: str) -> str:
    return _parse_choice(text, TIE_COLOR_AXIS_CHOICES, field="tie_color_axis").lower()

def _parse_html_colorscale(text: str) -> str:
    return _parse_choice(text, HTML_COLOR_SCALE_CHOICES, field="colorscale")


def _parse_heatmap_value(text: str) -> str:
    return _parse_choice(text, HEATMAP_VALUE_CHOICES, field="heatmap_value").lower()


def _parse_tie_heatmap_value(text: str) -> str:
    return _parse_choice(text, TIE_HEATMAP_VALUE_CHOICES, field="tie_heatmap_value").lower()


def _parse_tie_load_from(text: str) -> str:
    return _parse_choice(text, ("l", "r"), field="load_ties_from").lower()


def _parse_trim_color_range_percent(text: str) -> int:
    try:
        val = int(str(text).strip())
    except ValueError as e:
        raise ValueError("trim color range percent must be an integer.") from e
    if val < 0 or val > 40:
        raise ValueError("trim color range percent must be in [0, 40].")
    return val


def _parse_heatmap_pixel_mode(text: str) -> str:
    return _parse_choice(text, HEATMAP_PIXEL_MODE_CHOICES, field="exact_pixel_heatmap").lower()


def _parse_mpl_colormap_name(text: str) -> str:
    raw = text.strip()
    if not raw:
        raise ValueError("colormap cannot be empty.")
    from matplotlib import colormaps

    if raw in colormaps:
        return raw
    lowered = raw.lower()
    for name in colormaps:
        if name.lower() == lowered:
            return name
    raise ValueError(f"Unknown matplotlib colormap: {raw!r}.")


def _parse_fill_from(text: str) -> str:
    return _parse_choice(text, ("left", "right"), field="fill_from")


def _parse_vp_p_range(text: str) -> str:
    return _parse_choice(text, ("full", "left", "right"), field="vp_range")


def _parse_export_format(text: str) -> str:
    return _parse_choice(text, ("pdf", "png", "svg"), field="format")


def _parse_export_backend(text: str) -> str:
    return _parse_choice(text, ("pyqtgraph", "matplotlib"), field="backend")


def _parse_variant_explorer(text: str) -> int:
    choice = _parse_choice(text, ("1", "2", "3", "4", "5", "6"), field="variant")
    return int(choice)


# (variant number, one-line description) for interactive HTML explorer only.
_HTML_EXPLORER_VARIANTS: tuple[tuple[int, str], ...] = (
    (
        1,
        "Graph explorer — E[X] or E[rank] vs p or n; binomial curves and tie hairlines (graph + tie shards).",
    ),
    (
        2,
        "Two-panel — binomial strip + PCA score grid, shared n-range controls (graph + tie shards).",
    ),
    (
        3,
        "Quad layout — four tiles including E[X]/E[rank] vs p (graph + tie shards).",
    ),
    (
        4,
        "PCA-only — PCA/score plots without the extra graph tiles (graph + tie shards).",
    ),
    (
        5,
        "Tie vs N — tie-field curves vs n; uses tie slope shards only (tie index from center p≈0.5).",
    ),
    (
        6,
        "Tie vs tie # — same fields as variant 5 but x = tie index; Multiple controls n (tie shards only).",
    ),
)


def _interactive_html_pick_variant() -> int | None:
    print("\nHTML explorer — choose variant:")
    for num, desc in _HTML_EXPLORER_VARIANTS:
        print(f"  {num}) {desc}")
    print("  q) Cancel")
    while True:
        raw = input("Variant [1–6 or q]: ").strip().lower()
        if raw == "":
            continue
        if raw in ("q", "quit"):
            print("Cancelled.")
            return None
        if raw in ("1", "2", "3", "4", "5", "6"):
            return int(raw)
        print("Enter 1–6 or q.", file=sys.stderr)


def _interactive_html_configure_variant(variant: int) -> argparse.Namespace | None:
    default_n_max = 200 if variant in (1, 5, 6) else 100
    cfg: dict[str, object] = {
        "variant": variant,
        "output": _default_html_output_for_variant(variant),
        "n_min": 2,
        "n_max": default_n_max,
        "tie_manifest": None,
        "include_tie_points": True,
        "colorscale": "viridis",
        "verbose": False,
        "open_browser": False,
    }
    if variant not in (5, 6):
        cfg["p_steps"] = DEFAULT_GRAPH_P_STEPS
        cfg["graph_manifest"] = None
        cfg["graph_shards_dir"] = None

    parse_p_steps = _make_parse_p_steps(cfg)

    fields: list[tuple[str, str, callable, str]] = [
        (
            "output",
            "output file path",
            str,
            "Path for generated HTML. Default html/OBDExplorer<variant>.html; bare filenames go under html/.",
        ),
        (
            "n_min",
            "n_min",
            int,
            "Use only n values covered by your precomputed shards (e.g. from OBDsave). "
            "Smallest allowed n_min is 2. Must satisfy n_min <= n_max.",
        ),
        (
            "n_max",
            "n_max",
            int,
            "Upper n bound within precomputed shards. Larger ranges can make HTML generation slower.",
        ),
    ]
    if variant not in (5, 6):
        fields.extend(
            [
                (
                    "p_steps",
                    'p_steps ("1001"|"10001")',
                    parse_p_steps,
                    "Must match a generated graph manifest (e.g. from OBDsave). Larger values can be very slow.",
                ),
                ("graph_manifest", "graph_manifest (or none)", _parse_opt_str, "Path to graph shard manifest."),
                ("graph_shards_dir", "graph_shards_dir (or none)", _parse_opt_str, "Directory containing graph shards/manifests."),
            ]
        )
    fields.extend(
        [
            ("tie_manifest", "tie_manifest (or none)", _parse_opt_str, "Path to tie shard manifest."),
        ]
    )
    if variant in (1, 5, 6):
        fields.append(
            (
                "colorscale",
                "colorscale",
                _parse_html_colorscale,
                "Line colorscale for Multiple mode. Options: "
                + ", ".join(HTML_COLOR_SCALE_CHOICES),
            )
        )
    if variant == 1:
        fields.append(
            (
                "include_tie_points",
                'include tie points ("yes"|"no")',
                _parse_bool,
                "If no, do not load tie shards at build time and keep Swap points disabled in the HTML.",
            )
        )
    fields.extend(
        [
            (
                "verbose",
                'verbose ("yes"|"no")',
                _parse_bool,
                "Print timing and progress on stderr every 10 n values while building embedded HTML data.",
            ),
            ("open_browser", 'open_browser ("yes"|"no")', _parse_bool, "Open generated HTML in your default browser after writing."),
        ]
    )

    key_by_field = {"n_min": "n", "n_max": "m", "verbose": "v"}
    if variant == 1:
        key_by_field["p_steps"] = "p"
        key_by_field["colorscale"] = "c"
    keymap = _build_menu_keymap(
        fields,
        key_by_field=key_by_field,
    )
    if not keymap:
        return None

    while True:
        print(f"\nHTML explorer — variant {variant} settings:")
        print(
            "Hint: enter a key to edit, 'h' for help, 'h<letter>' for help on that row "
            "(e.g. hn for n_min), 'g' to generate.",
        )
        for letter, (key, label, _, _) in keymap.items():
            shown = _render_setting_value(key, cfg[key], cfg)
            print(f"  {letter}) {label}: {shown!r}")
        print("  g) Generate HTML")
        print("  h) Help")
        print("  q) Cancel")
        cmd = input("Choose key to edit, or g/h/hx/q: ").strip().lower()
        if cmd == "":
            continue
        if cmd == "q":
            print("Cancelled.")
            return None
        if cmd == "h":
            print(
                "Help:\n"
                "  - Type a setting key to edit its value.\n"
                "  - Type h<key> for detailed help about that setting (example: ha).\n"
                "  - Type g to generate HTML.\n"
                "  - Type q to cancel.\n"
            )
            continue
        if cmd.startswith("h") and len(cmd) == 2:
            hk = cmd[1]
            field = keymap.get(hk)
            if field is None:
                print(f"Unknown help key: {hk}", file=sys.stderr)
                continue
            _, label, _, detail = field
            print(f"{hk}) {label}\n    {detail}\n")
            continue
        if cmd == "g":
            if int(cfg["variant"]) not in (1, 2, 3, 4, 5, 6):
                print("variant must be 1, 2, 3, 4, 5, or 6.", file=sys.stderr)
                continue
            if int(cfg["n_min"]) > int(cfg["n_max"]):
                print("n_min must be <= n_max.", file=sys.stderr)
                continue
            if int(cfg["variant"]) in (5, 6):
                return argparse.Namespace(**cfg)
            valid_p_steps = _discover_available_graph_p_steps(cfg)
            if int(cfg["p_steps"]) not in valid_p_steps:
                shown = "|".join(str(x) for x in valid_p_steps)
                print(
                    f"p_steps must be one of ({shown}) based on available graph manifests.",
                    file=sys.stderr,
                )
                continue
            return argparse.Namespace(**cfg)
        field_entry = keymap.get(cmd)
        if field_entry is None:
            print("Unknown option.", file=sys.stderr)
            continue
        key, label, parser, _ = field_entry
        cur = _render_setting_value(key, cfg[key], cfg)
        raw = input(f"Set {label} [{cur!r}]: ").strip()
        if raw == "":
            continue
        try:
            cfg[key] = parser(raw)
        except ValueError as e:
            print(f"Invalid value: {e}", file=sys.stderr)


def _interactive_html_settings() -> argparse.Namespace | None:
    v = _interactive_html_pick_variant()
    if v is None:
        return None
    return _interactive_html_configure_variant(v)


def _discover_available_graph_p_steps(cfg: dict[str, object]) -> tuple[int, ...]:
    """Scan graph shard manifests and return discovered p_steps values.

    If no manifest files are present, fall back to known supported presets for now.
    """
    shards_dir = cfg.get("graph_shards_dir")
    if not isinstance(shards_dir, str) or not shards_dir.strip():
        shards_dir = _effective_default_graph_shards_dir()
    found: set[int] = set()
    if os.path.isdir(shards_dir):
        for name in os.listdir(shards_dir):
            m = re.fullmatch(r"0000_manifest_p(\d+)\.pkl", name)
            if m:
                found.add(int(m.group(1)))
    if found:
        return tuple(sorted(found))
    return (1001, 10001)


def _make_parse_p_steps(cfg: dict[str, object]):
    def _parse_p_steps(text: str) -> int:
        v = int(text.strip())
        valid = _discover_available_graph_p_steps(cfg)
        if v not in valid:
            choices = "|".join(str(x) for x in valid)
            raise ValueError(
                f'p_steps must be one of ({choices}) based on available graph manifests.'
            )
        return v

    return _parse_p_steps


def _build_menu_keymap(
    fields: list[tuple[str, str, callable, str]],
    *,
    key_by_field: dict[str, str] | None = None,
) -> dict[str, tuple[str, str, callable, str]]:
    """Assign non-colliding one-char keys, reserving g/h/q for commands."""
    pool = "abcdefijklmnoprstuvwxyz0123456789"
    reserved_cmd = frozenset("ghq")
    if len(fields) > len(pool):
        print("Internal error: too many interactive settings fields.", file=sys.stderr)
        return {}

    if key_by_field is None:
        return {pool[i]: fields[i] for i in range(len(fields))}

    field_by_name = {f[0]: f for f in fields}
    used_letters: set[str] = set()
    keymap: dict[str, tuple[str, str, callable, str]] = {}

    for fname, letter in key_by_field.items():
        ft = field_by_name.get(fname)
        if ft is None:
            print(f"Internal error: unknown field {fname!r} in key_by_field.", file=sys.stderr)
            return {}
        ch = letter.strip().lower()
        if len(ch) != 1:
            print(f"Internal error: menu key for {fname} must be a single character.", file=sys.stderr)
            return {}
        if ch in reserved_cmd:
            print(f"Internal error: menu key {ch!r} is reserved (g/h/q).", file=sys.stderr)
            return {}
        if ch in used_letters:
            print("Internal error: duplicate menu key assignment.", file=sys.stderr)
            return {}
        keymap[ch] = ft
        used_letters.add(ch)

    for ft in fields:
        fname = ft[0]
        if fname in key_by_field:
            continue
        for ch in pool:
            if ch not in used_letters:
                keymap[ch] = ft
                used_letters.add(ch)
                break
        else:
            print("Internal error: could not assign menu key (pool exhausted).", file=sys.stderr)
            return {}

    return keymap


def _interactive_export_settings(fmt: str) -> argparse.Namespace | None:
    cfg: dict[str, object] = {
        "output": os.path.join("plots", "OBDGraphExport"),
        "format": fmt,
        "backend": "pyqtgraph",
        "n_min": 2,
        "n_max": 100,
        "p_steps": DEFAULT_GRAPH_P_STEPS,
        "vp_range": "right",
        "endpoint_chord": False,
        "clip_tie": False,
        "tie_direction": "up",
        "tie_to_border": True,
        "tie_color_left": "i",
        "tie_color_right": "j",
        "tie_colormap": None,
        "fill_colormap": "gist_rainbow",
        "fill_from": "left",
        "flip_fill": True,
        "tie_opacity": 0.9,
        "tie_opacity_k": 0.0,
        "graph_opacity": 0.3,
        "graph_opacity_k": 1.0,
        "tie_line_px": 1.0,
        "graph_line_px": 0.1,
        "width_in": 12.0,
        "height_in": 10.0,
        "dpi": 400,
        "graph_manifest": None,
        "graph_shards_dir": None,
        "tie_manifest": None,
    }

    parse_p_steps = _make_parse_p_steps(cfg)
    fields: list[tuple[str, str, callable, str]] = [
        (
            "output",
            "output file path",
            str,
            "Base path without extension (.png / .pdf / .svg is appended from export format); "
            "if you enter a trailing .png, .pdf, or .svg it is normalized to match the chosen format.",
        ),  # a
        (
            "backend",
            'backend ("pyqtgraph"|"matplotlib")',
            _parse_export_backend,
            "Rendering backend. pyqtgraph is the current native path; matplotlib is an alternate path "
            "for PDF/SVG/PNG exports.",
        ),  # b
        ("clip_tie", 'clip_tie ("yes"|"no")', _parse_bool, "Clip p-range to observed tie range in selected window."),  # c
        ("dpi", "dpi", int, "Export dots-per-inch (used for png/pdf/svg)."),  # d
        ("endpoint_chord", 'endpoint_chord ("yes"|"no")', _parse_bool, "Detrend each curve by subtracting endpoint chord."),  # e
        (
            "format",
            'format ("pdf"|"png"|"svg")',
            _parse_export_format,
            "Export format: pdf (vector), png (raster), or svg (vector). Case-insensitive; prefixes work "
            "(bare p is ambiguous between pdf and png).",
        ),  # f
        (
            "fill_colormap",
            "fill_colormap",
            _parse_opt_str,
            "Interactive default is gist_rainbow. Enter any name registered with Matplotlib as a colormap, or none to disable strip fills.",
        ),  # i
        (
            "fill_from",
            'fill_from ("left"|"right")',
            _parse_fill_from,
            "Strip fill anchored from the left or right in p. Case-insensitive; prefixes such as l, r, le, ri accepted.",
        ),  # j
        (
            "tie_color_left",
            'tie_color_left ("black"|"i"|"j"|"l"|"r"|"d"|"e")',
            _parse_tie_color_axis,
            "Colormap axis for p < 0.5 in full window, or for left p window [0, 0.5]. "
            "d = slope_right − slope_left, e = slope_left − slope_right (from tie shard slopes).",
        ),  # k
        (
            "tie_color_right",
            'tie_color_right ("black"|"i"|"j"|"l"|"r"|"d"|"e")',
            _parse_tie_color_axis,
            "Colormap axis for p ≥ 0.5 in full window, or for right p window [0.5, 1]. Same letters as tie_color_left.",
        ),  # l
        ("flip_fill", 'flip_fill ("yes"|"no")', _parse_bool, "Flip fill direction on left half when enabled."),  # m
        (
            "n_max",
            "n_max",
            int,
            "Must stay within the range covered by precomputed shards for n. Defaults to 100 in this menu; "
            "larger limits can make export much slower.",
        ),  # n
        (
            "n_min",
            "n_min",
            int,
            "Use only n values that fall within graph (and tie) shards you already precomputed (e.g. with OBDsave); "
            "missing shards will fail at runtime. Smallest allowed n_min is 2. Must satisfy n_min <= n_max.",
        ),  # o
        ("graph_manifest", "graph_manifest (or none)", _parse_opt_str, "Path to graph shard manifest; none uses default manifest for p_steps."),  # p
        (
            "p_steps",
            'p_steps ("1001"|"10001")',
            parse_p_steps,
            "Valid p_steps are only ones with pre-generated graph data manifests (generated via OBDsave). "
            "For now: 1001 and 10001. Values above the default can be extremely slow.",
        ),  # q
        (
            "vp_range",
            'vp_range ("full"|"left"|"right")',
            _parse_vp_p_range,
            "Viewport p window: full=[0,1], left=[0,0.5], right=[0.5,1]. "
            "Case-insensitive; non-empty prefixes work (f, fu, l, ri, …).",
        ),  # r
        ("graph_shards_dir", "graph_shards_dir (or none)", _parse_opt_str, "Directory containing graph shard files/manifests."),  # s
        (
            "tie_direction",
            'tie_direction ("up"|"down")',
            _parse_tie_direction,
            "Tie band connectivity along n: up or down (see renderer). Case-insensitive; prefixes such as u, d, do, up accepted.",
        ),  # t
        ("tie_colormap", "tie_colormap", _parse_opt_str, "Interactive default is None. Enter any name registered with Matplotlib as a colormap, or none to skip cmap-based tie coloring."),  # u
        ("graph_opacity", "graph_opacity", float, "Base opacity for graph curves."),  # v
        ("width_in", "width (in)", float, "Export width in inches (used for png/pdf/svg)."),  # w
        ("graph_line_px", "graph_line_px", float, "Graph line width in device pixels."),  # x
        ("height_in", "height (in)", float, "Export height in inches (used for png/pdf/svg)."),  # y
        ("tie_manifest", "tie_manifest (or none)", _parse_opt_str, "Path to tie shard manifest."),  # z
        ("tie_to_border", 'tie_to_border ("yes"|"no")', _parse_bool, "Extend tie lines to viewport border when true."),  # 0
        ("tie_opacity", "tie_opacity", float, "Base opacity for tie lines."),  # 1
        ("tie_opacity_k", "tie_opacity_k", float, "Tie opacity slope factor across n."),  # 2
        ("graph_opacity_k", "graph_opacity_k", float, "Graph opacity slope factor across n."),  # 3
        ("tie_line_px", "tie_line_px", float, "Tie line width in device pixels."),  # 4
    ]
    keymap = _build_menu_keymap(fields)
    if not keymap:
        return None

    while True:
        print("\nExport settings:")
        print("Hint: enter a key to edit, 'h' for help, 'ha' for help on key a, 'g' to export.")
        for letter, (key, label, _, _) in keymap.items():
            shown = _render_setting_value(key, cfg[key], cfg)
            print(f"  {letter}) {label}: {shown!r}")
        print("  g) Proceed with export")
        print("  h) Help")
        print("  q) Cancel")
        cmd = input("Choose key to edit, or g/h/hx/q: ").strip().lower()
        if cmd == "":
            continue
        if cmd == "q":
            print("Cancelled.")
            return None
        if cmd == "h":
            print(
                "Help:\n"
                "  - Type a setting key to edit its value.\n"
                "  - Type h<key> for detailed help about that setting (example: ha).\n"
                "  - Type g to proceed with export.\n"
                "  - Type q to cancel.\n"
            )
            continue
        if cmd.startswith("h") and len(cmd) == 2:
            key = cmd[1]
            field = keymap.get(key)
            if field is None:
                print(f"Unknown help key: {key}", file=sys.stderr)
                continue
            _, label, _, detail = field
            print(f"{key}) {label}\n    {detail}\n")
            continue
        if cmd == "g":
            if str(cfg["format"]).lower() not in ("pdf", "png", "svg"):
                print('format must be "pdf", "png", or "svg".', file=sys.stderr)
                continue
            if str(cfg["backend"]).lower() not in ("pyqtgraph", "matplotlib"):
                print('backend must be "pyqtgraph" or "matplotlib".', file=sys.stderr)
                continue
            if int(cfg["n_min"]) > int(cfg["n_max"]):
                print("n_min must be <= n_max.", file=sys.stderr)
                continue
            if float(cfg["width_in"]) <= 0.0 or float(cfg["height_in"]) <= 0.0:
                print("width (in) and height (in) must be > 0.", file=sys.stderr)
                continue
            if int(cfg["dpi"]) <= 0:
                print("dpi must be > 0.", file=sys.stderr)
                continue
            valid_p_steps = _discover_available_graph_p_steps(cfg)
            if int(cfg["p_steps"]) not in valid_p_steps:
                shown = "|".join(str(x) for x in valid_p_steps)
                print(
                    f"p_steps must be one of ({shown}) based on available graph manifests.",
                    file=sys.stderr,
                )
                continue
            cfg["format"] = str(cfg["format"]).lower()
            cfg["backend"] = str(cfg["backend"]).lower()
            return argparse.Namespace(**cfg)
        field = keymap.get(cmd)
        if field is None:
            print("Unknown option.", file=sys.stderr)
            continue
        key, label, parser, _ = field
        cur = _render_setting_value(key, cfg[key], cfg)
        raw = input(f"Set {label} [{cur!r}]: ").strip()
        if raw == "":
            continue
        try:
            cfg[key] = parser(raw)
        except ValueError as e:
            print(f"Invalid value: {e}", file=sys.stderr)


def _interactive_heatmap_export_settings() -> argparse.Namespace | None:
    cfg: dict[str, object] = {
        "output": os.path.join("plots", "OBDHeatmap"),
        "pixel_mode": "annotated",
        "dpi": 400,
        "n_min": 2,
        "n_max": 1000,
        "p_steps": DEFAULT_GRAPH_P_STEPS,
        "graph_manifest": None,
        "graph_shards_dir": None,
        "vp_range": "full",
        "legend": False,
        "verbose": False,
        "width_in": 12.0,
        "height_in": 10.0,
        "colormap": "viridis",
        "value": "ev_n",
        "trim_color_range_percent": 1,
        "per_n_color_range": False,
        "format": "png",
    }

    parse_p_steps = _make_parse_p_steps(cfg)
    fields: list[tuple[str, str, callable, str]] = [
        (
            "output",
            "output file path",
            str,
            "Base path without extension (.png appended automatically).",
        ),
        (
            "pixel_mode",
            f'exact pixel heatmap ({"/".join(HEATMAP_PIXEL_MODE_CHOICES)})',
            _parse_heatmap_pixel_mode,
            "exact = raw raster export (1:1 data cell to output pixel), annotated = plotted figure with labels/colorbar and not guaranteed 1:1 cell-to-pixel.",
        ),
        (
            "colormap",
            "colormap",
            _parse_mpl_colormap_name,
            "Matplotlib colormap name used for heatmap colors.",
        ),
        ("dpi", "dpi", int, "Export dots-per-inch (png)."),
        ("graph_manifest", "graph_manifest (or none)", _parse_opt_str, "Path to graph shard manifest; none uses default manifest for p_steps."),
        ("graph_shards_dir", "graph_shards_dir (or none)", _parse_opt_str, "Directory containing graph shard files/manifests."),
        ("legend", 'legend ("yes"|"no")', _parse_bool, "Show heatmap color legend (colorbar)."),
        (
            "trim_color_range_percent",
            "trim color range percent (0..40)",
            _parse_trim_color_range_percent,
            "Trim this percent from both low/high tails for color scaling (0 = no trimming).",
        ),
        (
            "per_n_color_range",
            'per-N color range ("yes"|"no")',
            _parse_bool,
            "When yes, compute color scaling per N row; when no, use one global range.",
        ),
        ("verbose", 'verbose ("yes"|"no")', _parse_bool, "Print graph-shard load progress every 10 shards during heatmap export."),
        (
            "n_max",
            "n_max",
            int,
            "Upper n bound. Must satisfy n_min <= n_max and be covered by graph shards.",
        ),
        (
            "n_min",
            "n_min",
            int,
            "Lower n bound. Smallest allowed is 2.",
        ),
        (
            "p_steps",
            'p_steps ("1001"|"10001")',
            parse_p_steps,
            "Must match available graph manifests; controls p-grid density.",
        ),
        (
            "vp_range",
            'vp_range ("full"|"left"|"right")',
            _parse_vp_p_range,
            "Heatmap p window: full=[0,1], left=[0,0.5], right=[0.5,1].",
        ),
        ("width_in", "width (in)", float, "PNG width in inches."),
        ("height_in", "height (in)", float, "PNG height in inches."),
        (
            "value",
            f'value ({"/".join(HEATMAP_VALUE_CHOICES)})',
            _parse_heatmap_value,
            'Heatmap value: i/j/l/r/d/e use nearest-tie proxy per p; "ev_n"/"eslope_n" use graph shards.',
        ),
    ]

    keymap = _build_menu_keymap(
        fields,
        key_by_field={
            "output": "a",
            "pixel_mode": "b",
            "colormap": "c",
            "dpi": "d",
            "graph_manifest": "i",
            "graph_shards_dir": "j",
            "legend": "l",
            "n_max": "m",
            "n_min": "n",
            "p_steps": "p",
            "per_n_color_range": "r",
            "vp_range": "s",
            "trim_color_range_percent": "u",
            "verbose": "v",
            "width_in": "x",
            "height_in": "y",
            "value": "z",
        },
    )
    if not keymap:
        return None

    while True:
        print("\nHeatmap export settings:")
        print("Hint: enter a key to edit, 'h' for help, 'ha' for help on key a, 'g' to export.")
        for letter, (key, label, _, _) in keymap.items():
            shown = _render_setting_value(key, cfg[key], cfg)
            print(f"  {letter}) {label}: {shown!r}")
        print("  g) Proceed with export")
        print("  h) Help")
        print("  q) Cancel")
        cmd = input("Choose key to edit, or g/h/hx/q: ").strip().lower()
        if cmd == "":
            continue
        if cmd == "q":
            print("Cancelled.")
            return None
        if cmd == "h":
            print(
                "Help:\n"
                "  - Type a setting key to edit its value.\n"
                "  - Type h<key> for detailed help about that setting (example: ha).\n"
                "  - Type g to proceed with export.\n"
                "  - Type q to cancel.\n"
            )
            continue
        if cmd.startswith("h") and len(cmd) == 2:
            key = cmd[1]
            field = keymap.get(key)
            if field is None:
                print(f"Unknown help key: {key}", file=sys.stderr)
                continue
            _, label, _, detail = field
            print(f"{key}) {label}\n    {detail}\n")
            continue
        if cmd == "g":
            if int(cfg["n_min"]) > int(cfg["n_max"]):
                print("n_min must be <= n_max.", file=sys.stderr)
                continue
            if float(cfg["width_in"]) <= 0.0 or float(cfg["height_in"]) <= 0.0:
                print("width (in) and height (in) must be > 0.", file=sys.stderr)
                continue
            if int(cfg["dpi"]) <= 0:
                print("dpi must be > 0.", file=sys.stderr)
                continue
            valid_p_steps = _discover_available_graph_p_steps(cfg)
            if int(cfg["p_steps"]) not in valid_p_steps:
                shown = "|".join(str(x) for x in valid_p_steps)
                print(
                    f"p_steps must be one of ({shown}) based on available graph manifests.",
                    file=sys.stderr,
                )
                continue
            cfg["pixel_mode"] = str(cfg["pixel_mode"]).lower()
            cfg["value"] = str(cfg["value"]).lower()
            cfg["format"] = "png"
            return argparse.Namespace(**cfg)
        field = keymap.get(cmd)
        if field is None:
            print("Unknown option.", file=sys.stderr)
            continue
        key, label, parser, _ = field
        cur = _render_setting_value(key, cfg[key], cfg)
        raw = input(f"Set {label} [{cur!r}]: ").strip()
        if raw == "":
            continue
        try:
            cfg[key] = parser(raw)
        except ValueError as e:
            print(f"Invalid value: {e}", file=sys.stderr)


def _interactive_tie_heatmap_export_settings() -> argparse.Namespace | None:
    cfg: dict[str, object] = {
        "output": os.path.join("plots", "OBDTieHeatmap"),
        "pixel_mode": "annotated",
        "dpi": 400,
        "n_min": 2,
        "n_max": 1000,
        "legend": False,
        "verbose": False,
        "width_in": 12.0,
        "height_in": 10.0,
        "colormap": "viridis",
        "value": "d",
        "load_from": "l",
        "trim_color_range_percent": 1,
        "per_n_color_range": False,
        "tie_manifest": None,
        "format": "png",
    }
    fields: list[tuple[str, str, callable, str]] = [
        (
            "output",
            "output file path",
            str,
            "Base path without extension (.png appended automatically).",
        ),
        (
            "pixel_mode",
            f'exact pixel heatmap ({"/".join(HEATMAP_PIXEL_MODE_CHOICES)})',
            _parse_heatmap_pixel_mode,
            "exact = raw raster export (1:1 data cell to output pixel), annotated = plotted figure with labels/colorbar and not guaranteed 1:1 cell-to-pixel.",
        ),
        (
            "colormap",
            "colormap",
            _parse_mpl_colormap_name,
            "Matplotlib colormap name used for heatmap colors.",
        ),
        ("dpi", "dpi", int, "Export dots-per-inch (png)."),
        ("tie_manifest", "tie_manifest (or none)", _parse_opt_str, "Path to tie shard manifest."),
        ("legend", 'legend ("yes"|"no")', _parse_bool, "Show heatmap color legend (colorbar)."),
        (
            "trim_color_range_percent",
            "trim color range percent (0..40)",
            _parse_trim_color_range_percent,
            "Trim this percent from both low/high tails for color scaling (0 = no trimming).",
        ),
        (
            "per_n_color_range",
            'per-N color range ("yes"|"no")',
            _parse_bool,
            "When yes, compute color scaling per N row; when no, use one global range.",
        ),
        ("verbose", 'verbose ("yes"|"no")', _parse_bool, "Print tie-shard load progress every 10 shards during heatmap export."),
        (
            "n_max",
            "n_max",
            int,
            "Upper n bound. Must satisfy n_min <= n_max and be covered by tie shards.",
        ),
        (
            "n_min",
            "n_min",
            int,
            "Lower n bound. Smallest allowed is 2.",
        ),
        ("load_from", 'load ties from ("l"|"r")', _parse_tie_load_from, 'l = center-out, r = end-in (last tie first).'),
        ("width_in", "width (in)", float, "PNG width in inches."),
        ("height_in", "height (in)", float, "PNG height in inches."),
        (
            "value",
            f'value ({"/".join(TIE_HEATMAP_VALUE_CHOICES)})',
            _parse_tie_heatmap_value,
            "Tie value used for color (same set as HTML variants 5/6 except p).",
        ),
    ]
    keymap = _build_menu_keymap(
        fields,
        key_by_field={
            "output": "a",
            "pixel_mode": "b",
            "colormap": "c",
            "dpi": "d",
            "tie_manifest": "i",
            "legend": "l",
            "n_max": "m",
            "n_min": "n",
            "per_n_color_range": "r",
            "load_from": "t",
            "trim_color_range_percent": "u",
            "verbose": "v",
            "width_in": "x",
            "height_in": "y",
            "value": "z",
        },
    )
    if not keymap:
        return None

    while True:
        print("\nN-tie heatmap settings:")
        print("Hint: enter a key to edit, 'h' for help, 'ha' for help on key a, 'g' to export.")
        for letter, (key, label, _, _) in keymap.items():
            shown = _render_setting_value(key, cfg[key], cfg)
            print(f"  {letter}) {label}: {shown!r}")
        print("  g) Proceed with export")
        print("  h) Help")
        print("  q) Cancel")
        cmd = input("Choose key to edit, or g/h/hx/q: ").strip().lower()
        if cmd == "":
            continue
        if cmd == "q":
            print("Cancelled.")
            return None
        if cmd == "h":
            print(
                "Help:\n"
                "  - Type a setting key to edit its value.\n"
                "  - Type h<key> for detailed help about that setting (example: ha).\n"
                "  - Type g to proceed with export.\n"
                "  - Type q to cancel.\n"
            )
            continue
        if cmd.startswith("h") and len(cmd) == 2:
            key = cmd[1]
            field = keymap.get(key)
            if field is None:
                print(f"Unknown help key: {key}", file=sys.stderr)
                continue
            _, label, _, detail = field
            print(f"{key}) {label}\n    {detail}\n")
            continue
        if cmd == "g":
            if int(cfg["n_min"]) > int(cfg["n_max"]):
                print("n_min must be <= n_max.", file=sys.stderr)
                continue
            if float(cfg["width_in"]) <= 0.0 or float(cfg["height_in"]) <= 0.0:
                print("width (in) and height (in) must be > 0.", file=sys.stderr)
                continue
            if int(cfg["dpi"]) <= 0:
                print("dpi must be > 0.", file=sys.stderr)
                continue
            cfg["pixel_mode"] = str(cfg["pixel_mode"]).lower()
            cfg["value"] = str(cfg["value"]).lower()
            cfg["load_from"] = str(cfg["load_from"]).lower()
            cfg["format"] = "png"
            return argparse.Namespace(**cfg)
        field = keymap.get(cmd)
        if field is None:
            print("Unknown option.", file=sys.stderr)
            continue
        key, label, parser, _ = field
        cur = _render_setting_value(key, cfg[key], cfg)
        raw = input(f"Set {label} [{cur!r}]: ").strip()
        if raw == "":
            continue
        try:
            cfg[key] = parser(raw)
        except ValueError as e:
            print(f"Invalid value: {e}", file=sys.stderr)


def _run_gui(argv_extra: list[str] | None = None) -> None:
    import obd_explorer_qt_ui as ui

    if argv_extra:
        sys.argv = [sys.argv[0], *argv_extra]
    ui.main()


def _run_export(args: argparse.Namespace) -> None:
    from obd_explorer.render_headless import HeadlessExportConfig, export_graph_headless

    fmt = args.format
    if fmt is None:
        low = args.output.strip().lower()
        if low.endswith(".pdf"):
            fmt = "pdf"
        elif low.endswith(".png"):
            fmt = "png"
        elif low.endswith(".svg"):
            fmt = "svg"
        else:
            print("Specify --format png|pdf|svg or use .png/.pdf/.svg output path.", file=sys.stderr)
            sys.exit(2)
    fmt = fmt.strip().lower()
    out_path = _resolved_export_output_path(args.output, fmt)
    cfg = HeadlessExportConfig(
        n_min=args.n_min,
        n_max=args.n_max,
        p_steps=args.p_steps,
        vp_p_range=args.vp_range,
        endpoint_chord_detrend=args.endpoint_chord,
        clip_to_tie_range=args.clip_tie,
        tie_lines_direction=args.tie_direction,
        tie_lines_to_border=args.tie_to_border,
        tie_color_left=args.tie_color_left,
        tie_color_right=args.tie_color_right,
        tie_colormap=args.tie_colormap,
        fill_colormap=args.fill_colormap,
        fill_from=args.fill_from,
        flip_fill_on_left=args.flip_fill,
        tie_line_opacity=args.tie_opacity,
        tie_line_opacity_k=args.tie_opacity_k,
        graph_line_opacity=args.graph_opacity,
        graph_line_opacity_k=args.graph_opacity_k,
        tie_line_device_px=args.tie_line_px,
        graph_line_device_px=args.graph_line_px,
        width_in=args.width_in,
        height_in=args.height_in,
        dpi=args.dpi,
        graph_manifest=args.graph_manifest,
        graph_shards_dir=args.graph_shards_dir,
        tie_manifest=args.tie_manifest,
        output_path=out_path,
        export_format=fmt,
        export_backend=args.backend,
    )
    _ensure_output_parent_dir(cfg.output_path)
    export_graph_headless(cfg, verbose=True)


def _run_heatmap_export(args: argparse.Namespace) -> None:
    from obd_explorer.render_headless import HeatmapExportConfig, export_heatmap_headless

    out_path = _resolved_export_output_path(args.output, "png")
    cfg = HeatmapExportConfig(
        n_min=args.n_min,
        n_max=args.n_max,
        p_steps=args.p_steps,
        vp_p_range=args.vp_range,
        value_key=args.value,
        colormap=args.colormap,
        show_legend=bool(getattr(args, "legend", False)),
        width_in=args.width_in,
        height_in=args.height_in,
        dpi=args.dpi,
        graph_manifest=args.graph_manifest,
        graph_shards_dir=args.graph_shards_dir,
        output_path=out_path,
        export_format="png",
        pixel_mode=args.pixel_mode,
        progress_every=(10 if bool(getattr(args, "verbose", False)) else None),
        trim_color_range_percent=int(getattr(args, "trim_color_range_percent", 1)),
        per_n_color_range=bool(getattr(args, "per_n_color_range", False)),
    )
    _ensure_output_parent_dir(cfg.output_path)
    export_heatmap_headless(cfg, verbose=True)


def _run_tie_heatmap_export(args: argparse.Namespace) -> None:
    from obd_explorer.render_headless import TieHeatmapExportConfig, export_tie_heatmap_headless

    out_path = _resolved_export_output_path(args.output, "png")
    cfg = TieHeatmapExportConfig(
        n_min=args.n_min,
        n_max=args.n_max,
        value_key=args.value,
        colormap=args.colormap,
        show_legend=bool(getattr(args, "legend", False)),
        load_from=args.load_from,
        width_in=args.width_in,
        height_in=args.height_in,
        dpi=args.dpi,
        tie_manifest=args.tie_manifest,
        output_path=out_path,
        export_format="png",
        pixel_mode=args.pixel_mode,
        progress_every=(10 if bool(getattr(args, "verbose", False)) else None),
        trim_color_range_percent=int(getattr(args, "trim_color_range_percent", 1)),
        per_n_color_range=bool(getattr(args, "per_n_color_range", False)),
    )
    _ensure_output_parent_dir(cfg.output_path)
    export_tie_heatmap_headless(cfg, verbose=True)


def _run_html(args: argparse.Namespace) -> None:
    raw_out = getattr(args, "output", None)
    if raw_out is None or not str(raw_out).strip():
        raw_out = _default_html_output_for_variant(int(args.variant))
    else:
        raw_out = str(raw_out).strip()
    out = _resolve_html_output_path(raw_out)
    _ensure_output_parent_dir(out)
    v = int(args.variant)
    progress = bool(getattr(args, "verbose", False))
    if v == 1:
        from obd_explorer.explorer1_export import write_explorer1_html

        write_explorer1_html(
            out,
            n_min=args.n_min,
            n_max=args.n_max,
            p_steps=args.p_steps,
            graph_manifest=args.graph_manifest,
            graph_shards_dir=args.graph_shards_dir,
            tie_manifest=args.tie_manifest,
            include_tie_points=bool(getattr(args, "include_tie_points", True)),
            colorscale=getattr(args, "colorscale", "viridis"),
            verbose=True,
            progress=progress,
        )
    elif v == 2:
        from obd_explorer.explorer2_export import write_explorer2_html

        write_explorer2_html(
            out,
            n_min=args.n_min,
            n_max=args.n_max,
            p_steps=args.p_steps,
            graph_manifest=args.graph_manifest,
            graph_shards_dir=args.graph_shards_dir,
            tie_manifest=args.tie_manifest,
            verbose=True,
            progress=progress,
        )
    elif v == 3:
        from obd_explorer.explorer2_export import write_explorer3_quad_html

        write_explorer3_quad_html(
            out,
            n_min=args.n_min,
            n_max=args.n_max,
            p_steps=args.p_steps,
            graph_manifest=args.graph_manifest,
            graph_shards_dir=args.graph_shards_dir,
            tie_manifest=args.tie_manifest,
            verbose=True,
            progress=progress,
        )
    elif v == 4:
        from obd_explorer.explorer2_export import write_explorer4_pca_only_html

        write_explorer4_pca_only_html(
            out,
            n_min=args.n_min,
            n_max=args.n_max,
            p_steps=args.p_steps,
            graph_manifest=args.graph_manifest,
            graph_shards_dir=args.graph_shards_dir,
            tie_manifest=args.tie_manifest,
            verbose=True,
            progress=progress,
        )
    elif v == 5:
        from obd_explorer.explorer5_export import write_explorer5_html

        write_explorer5_html(
            out,
            n_min=args.n_min,
            n_max=args.n_max,
            tie_manifest=args.tie_manifest,
            colorscale=getattr(args, "colorscale", "viridis"),
            verbose=True,
            progress=progress,
        )
    elif v == 6:
        from obd_explorer.explorer6_export import write_explorer6_html

        write_explorer6_html(
            out,
            n_min=args.n_min,
            n_max=args.n_max,
            tie_manifest=args.tie_manifest,
            colorscale=getattr(args, "colorscale", "viridis"),
            verbose=True,
            progress=progress,
        )
    else:
        print("--variant must be 1, 2, 3, 4, 5, or 6.", file=sys.stderr)
        sys.exit(2)
    if args.open_browser:
        webbrowser.open(os.path.abspath(out))


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-min", type=int, default=2)
    p.add_argument("--n-max", type=int, default=100)
    p.add_argument("--p-steps", type=int, default=DEFAULT_GRAPH_P_STEPS)
    p.add_argument("--graph-manifest", default=None, help="Graph shard manifest path.")
    p.add_argument("--graph-shards-dir", default=None)
    p.add_argument("--tie-manifest", default=None, help="Tie shard manifest path.")


def _interactive() -> None:
    while True:
        print(
            "OBDExplorerPlus — choose an option:\n"
            "  1  Launch Qt GUI\n"
            "  2  HTML explorer\n"
            "  3  Export graph (interactive settings)\n"
            "  4  N-p heatmaps (interactive settings)\n"
            "  5  N-tie heatmaps (interactive settings)\n"
            "  q  Quit\n"
        )
        choice = input("Enter choice [1–5 or q]: ").strip().lower()
        if choice == "":
            continue
        if choice in ("q", "quit"):
            return
        if choice == "1":
            _run_gui()
            return
        if choice == "2":
            ns = _interactive_html_settings()
            if ns is None:
                return
            _run_html(ns)
            return
        if choice == "3":
            ns = _interactive_export_settings("pdf")
            if ns is None:
                return
            _run_export(ns)
            return
        if choice == "4":
            ns = _interactive_heatmap_export_settings()
            if ns is None:
                return
            _run_heatmap_export(ns)
            return
        if choice == "5":
            ns = _interactive_tie_heatmap_export_settings()
            if ns is None:
                return
            _run_tie_heatmap_export(ns)
            return
        print("Unknown choice.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="OBD explorer plus (GUI / HTML / export).")
    sub = parser.add_subparsers(dest="cmd")

    p_gui = sub.add_parser("gui", help="Interactive Qt explorer.")
    _add_data_args(p_gui)

    p_html = sub.add_parser("html", help="Write self-contained HTML.")
    p_html.add_argument(
        "--variant",
        type=int,
        choices=(1, 2, 3, 4, 5, 6),
        required=True,
        help="1=graph (shards), 2=two-panel binomial+PCA (shards), 3=quad+E[X] (shards), 4=PCA-only (shards), "
        "5=tie scalar vs n (tie shards), 6=tie scalar vs tie # (tie shards).",
    )
    p_html.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path. If omitted, html/OBDExplorer<variant>.html. Bare filenames are placed under html/.",
    )
    p_html.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the HTML file in the default browser after writing.",
    )
    _add_data_args(p_html)
    p_html.set_defaults(n_max=200)
    p_html.add_argument(
        "--verbose",
        action="store_true",
        help="Print timing and progress to stderr every 10 n values while building embedded HTML data.",
    )
    p_html.add_argument(
        "--colorscale",
        default="viridis",
        choices=HTML_COLOR_SCALE_CHOICES,
        help="Line colorscale for Multiple mode (variants 1, 5, 6). "
        "Available options: " + ", ".join(HTML_COLOR_SCALE_CHOICES),
    )
    p_html.add_argument(
        "--include-tie-points",
        action="store_true",
        dest="include_tie_points",
        default=True,
        help="(Variant 1) include tie points in embedded HTML and enable Swap points when allowed.",
    )
    p_html.add_argument(
        "--no-tie-points",
        action="store_false",
        dest="include_tie_points",
        help="(Variant 1) skip loading tie points at build time and keep Swap points disabled.",
    )

    p_exp = sub.add_parser("export", help="Headless graph export (PyQtGraph or Matplotlib backend).")
    p_exp.add_argument("-o", "--output", required=True)
    p_exp.add_argument("--format", choices=("png", "pdf", "svg"), default=None)
    p_exp.add_argument("--backend", choices=("pyqtgraph", "matplotlib"), default="pyqtgraph")
    _add_data_args(p_exp)
    p_exp.add_argument("--vp-range", default="right")
    p_exp.add_argument("--endpoint-chord", action="store_true")
    p_exp.add_argument("--clip-tie", action="store_true")
    p_exp.add_argument("--tie-direction", default="up")
    p_exp.add_argument("--tie-to-border", action="store_true", default=True)
    p_exp.add_argument("--no-tie-to-border", action="store_false", dest="tie_to_border")
    p_exp.add_argument(
        "--tie-color-left",
        default="i",
        choices=TIE_COLOR_AXIS_CHOICES,
        help="Tie/fill axis for left half (full p) or left p window: black, i, j, l, r, d, e.",
    )
    p_exp.add_argument(
        "--tie-color-right",
        default="j",
        choices=TIE_COLOR_AXIS_CHOICES,
        help="Tie/fill axis for right half (full p) or right p window (same choices as --tie-color-left).",
    )
    p_exp.add_argument(
        "--tie-colormap",
        default=None,
        help="Matplotlib colormap name for tie coloring (default: none). Must be a registered colormap name.",
    )
    p_exp.add_argument(
        "--fill-colormap",
        default="gist_rainbow",
        help="Matplotlib colormap for strip fills (default: gist_rainbow). Use none to disable. Must be a registered colormap name.",
    )
    p_exp.add_argument("--fill-from", default="left")
    p_exp.add_argument("--flip-fill", action="store_true", default=True)
    p_exp.add_argument("--no-flip-fill", action="store_false", dest="flip_fill")
    p_exp.add_argument("--tie-opacity", type=float, default=0.9)
    p_exp.add_argument("--tie-opacity-k", type=float, default=0.0)
    p_exp.add_argument("--graph-opacity", type=float, default=0.3)
    p_exp.add_argument("--graph-opacity-k", type=float, default=1.0)
    p_exp.add_argument("--tie-line-px", type=float, default=1.0)
    p_exp.add_argument("--graph-line-px", type=float, default=0.1)
    p_exp.add_argument("--width-in", type=float, default=12.0)
    p_exp.add_argument("--height-in", type=float, default=8.0)
    p_exp.add_argument("--dpi", type=int, default=400)

    p_hm = sub.add_parser("heatmap", help="N-p heatmap export (graph shards, PNG).")
    p_hm.add_argument("-o", "--output", required=True)
    p_hm.add_argument("--format", choices=("png",), default="png")
    p_hm.add_argument(
        "--pixel-mode",
        choices=HEATMAP_PIXEL_MODE_CHOICES,
        default="annotated",
        help='Exact pixel heatmap mode: "exact" (raw raster) or "annotated" (axes/title/colorbar).',
    )
    p_hm.add_argument("--n-min", type=int, default=2)
    p_hm.add_argument("--n-max", type=int, default=1000)
    p_hm.add_argument("--p-steps", type=int, default=DEFAULT_GRAPH_P_STEPS)
    p_hm.add_argument("--graph-manifest", default=None, help="Graph shard manifest path.")
    p_hm.add_argument("--graph-shards-dir", default=None)
    p_hm.add_argument(
        "--vp-range",
        default="full",
        choices=("full", "left", "right"),
        help="p-axis viewport range for heatmap.",
    )
    p_hm.add_argument(
        "--value",
        default="ev_n",
        choices=HEATMAP_VALUE_CHOICES,
        help='Heatmap value: i/j/l/r/d/e from nearest tie proxy, or "ev_n"/"eslope_n" from graph shards.',
    )
    p_hm.add_argument(
        "--colormap",
        default="viridis",
        help="Matplotlib colormap name for heatmap colors.",
    )
    p_hm.add_argument("--legend", action="store_true", default=False, help="Show heatmap colorbar.")
    p_hm.add_argument(
        "--trim-color-range-percent",
        type=int,
        default=1,
        choices=range(0, 41),
        metavar="0..40",
        help="Trim this percent from both tails for color range (0 disables trimming; default: 1).",
    )
    p_hm.add_argument(
        "--per-n-color-range",
        action="store_true",
        dest="per_n_color_range",
        default=False,
        help="Use row-wise (per-N) color scaling (default: off).",
    )
    p_hm.add_argument(
        "--no-per-n-color-range",
        action="store_false",
        dest="per_n_color_range",
        help="Use one global color range across the whole heatmap (default).",
    )
    p_hm.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress every 10 loaded graph shards.",
    )
    p_hm.add_argument("--width-in", type=float, default=12.0)
    p_hm.add_argument("--height-in", type=float, default=10.0)
    p_hm.add_argument("--dpi", type=int, default=400)

    p_thm = sub.add_parser("tie-heatmap", help="N-tie heatmap export (iterative tie shards, PNG).")
    p_thm.add_argument("-o", "--output", required=True)
    p_thm.add_argument("--format", choices=("png",), default="png")
    p_thm.add_argument(
        "--pixel-mode",
        choices=HEATMAP_PIXEL_MODE_CHOICES,
        default="annotated",
        help='Exact pixel heatmap mode: "exact" (raw raster) or "annotated" (axes/title/colorbar).',
    )
    p_thm.add_argument("--n-min", type=int, default=2)
    p_thm.add_argument("--n-max", type=int, default=1000)
    p_thm.add_argument("--tie-manifest", default=None, help="Tie shard manifest path.")
    p_thm.add_argument(
        "--value",
        default="d",
        choices=TIE_HEATMAP_VALUE_CHOICES,
        help='Tie heatmap value key: i, j, l, r, d, e, or ev_n.',
    )
    p_thm.add_argument(
        "--load-from",
        default="l",
        choices=("l", "r"),
        help='Tie loading direction: "l" center-out, "r" end-in (last tie first).',
    )
    p_thm.add_argument(
        "--colormap",
        default="viridis",
        help="Matplotlib colormap name for heatmap colors.",
    )
    p_thm.add_argument("--legend", action="store_true", default=False, help="Show heatmap colorbar.")
    p_thm.add_argument(
        "--trim-color-range-percent",
        type=int,
        default=1,
        choices=range(0, 41),
        metavar="0..40",
        help="Trim this percent from both tails for color range (0 disables trimming; default: 1).",
    )
    p_thm.add_argument(
        "--per-n-color-range",
        action="store_true",
        dest="per_n_color_range",
        default=False,
        help="Use row-wise (per-N) color scaling (default: off).",
    )
    p_thm.add_argument(
        "--no-per-n-color-range",
        action="store_false",
        dest="per_n_color_range",
        help="Use one global color range across the whole heatmap (default).",
    )
    p_thm.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress every 10 loaded tie shards.",
    )
    p_thm.add_argument("--width-in", type=float, default=12.0)
    p_thm.add_argument("--height-in", type=float, default=10.0)
    p_thm.add_argument("--dpi", type=int, default=400)

    args = parser.parse_args()
    if args.cmd is None:
        _interactive()
        return
    if args.cmd == "gui":
        _run_gui()
    elif args.cmd == "html":
        _run_html(args)
    elif args.cmd == "export":
        _run_export(args)
    elif args.cmd == "heatmap":
        _run_heatmap_export(args)
    elif args.cmd == "tie-heatmap":
        _run_tie_heatmap_export(args)


if __name__ == "__main__":
    main()
