"""
Render E vs P with curves for ``GRAPH_N_MIN`` … ``GRAPH_N_MAX`` and banded swap-point art, then save a PNG.

Uses ``data/graph_data.pkl`` and ``data/tie_points.pkl``.  The figure's binomial grid is set only by
``GRAPH_N_MIN``, ``GRAPH_N_MAX``, and ``GRAPH_P_STEPS``; the pickle may be a **superset** (wider ``n`` and/or
finer ``p``).  Current pickles from ``OBDsaveSourceData.save_graph_data`` include a ``p_values`` list; if it is
absent, the loader assumes a uniform ``[0,1]`` grid from ``p_steps``.
Vertical tie lines (``TIE_COLORMAP``) and/or filled strips (``FILL_COLORMAP``) between consecutive ``n`` curves.
``TIE_LINES_DIRECTION`` (``"up"`` or ``"down"``) controls which curve's kinks emit tie lines and in which
direction; ``TIE_LINES_TO_BORDER`` extends the outermost band to the axis border.

Tie coloring: ``TIE_LINE_COLOR_MODE`` can be ``black``, ``i``, ``j``, or two letters ``ii``/``ij``/``ji``/``jj``.
For two letters with ``VP_P_RANGE`` ``full``, ``p < 0.5`` uses the first letter and ``p >= 0.5`` the second;
otherwise only the first letter applies.  Filled strips use ties at strip right edges (split at 0.5 when needed).

Optional half-window **endpoint chord** detrend (see ``ENDPOINT_CHORD_DETREND``) matches ``OBDgraphExplorer1``.

Requires: matplotlib, numpy.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
import pickle
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

# --- data files ---
DATA_DIR = "data"
GRAPH_DATA_PATH = os.path.join(DATA_DIR, "graph_data.pkl")
TIE_POINTS_PATH = os.path.join(DATA_DIR, "tie_points.pkl")

# --- output ---
PNG_WIDTH_PX = 6000
PNG_HEIGHT_PX = 4000
DPI = 100
OUTPUT_PNG_PATH = "OBDGraphWithTie.png"
# Axes position in figure coordinates (narrow margins around the plot).
SUBPLOT_LEFT = 0.045
SUBPLOT_RIGHT = 0.995
SUBPLOT_BOTTOM = 0.055
SUBPLOT_TOP = 0.965
SAVE_PAD_INCHES = 0.02

# --- plot semantics (match archived explorer defaults: sorted + scale by n) ---
SORTED = True
SCALED = True
# E vs p window: ``"full"`` → p in [0, 1]; ``"left"`` → [0, 0.5]; ``"right"`` → [0.5, 1] (case insensitive).
# Must match ``ax.set_xlim`` and which ``p`` indices are plotted; ties are kept only strictly inside this interval.
VP_P_RANGE = "full"
# If True, subtract the secant through the first/last finite points of each curve in ``p``
# (``OBDgraphExplorer1`` ``subtractEndpointChord``).  **Y before the chord** is raw ``E[rank]`` or
# ``n·p`` (same as explorer ``scaleMode === "endpoint"``), not ``E/n`` — ``SCALED`` is ignored for
# that step.  When ``VP_P_RANGE`` is ``"full"``, each half ``[0, 0.5]`` and ``[0.5, 1]`` is
# detrended independently (the curves meet at zero at ``p = 0.5``).
ENDPOINT_CHORD_DETREND = True
# If True, override the x-axis limits to the range [min_tie_p, max_tie_p] across all loaded tie points.
CLIP_TO_TIE_RANGE = False
# --- graph definition (binomial grid for this figure; must be extractable from ``graph_data.pkl``) ---
# Inclusive ``n``. Pickle must cover every ``n`` in this range.
GRAPH_N_MIN = 2
GRAPH_N_MAX = 100
# Uniform ``p`` on ``[0, 1]``: ``p_i = i / (GRAPH_P_STEPS - 1)`` for ``i = 0 .. GRAPH_P_STEPS - 1``.
# Pickle may use a finer (or equal) grid; each ``p_i`` must appear in the pickle's ``p`` list (see loader).
GRAPH_P_STEPS = 1001
# --- appearance ---
# E/n curves: black.  Base alpha in [0, 1]; use 0 to omit curves.  If ``GRAPH_LINE_OPACITY_K`` is 0,
# every curve uses ``GRAPH_LINE_OPACITY`` unchanged.  Otherwise per-curve alpha is
# ``GRAPH_LINE_OPACITY / (GRAPH_LINE_OPACITY_K * (n - 2) + 1)`` (clipped to [0, 1]).
GRAPH_LINE_OPACITY = 0.3
GRAPH_LINE_OPACITY_K = 1
# Tie art: "black" | "i" | "j" | "ii" | "ij" | "ji" | "jj" (case insensitive).  Single letter: colormap by
# that index per n.  Two letters + ``VP_P_RANGE`` ``full``: left half of [0,1] uses first letter, right half
# second; for ``left``/``right`` window, only the first letter is used.
TIE_LINE_COLOR_MODE = "ji"
TIE_LINE_OPACITY = 0.9
# Direction tie lines extend from each graph line's kinks:
# ``"up"``   → each kink sends a tie line **up** to the next higher curve (or the top border).
#              Source tie ``(p, i, j)`` comes from the **lower** curve in each band.
# ``"down"`` → each kink sends a tie line **down** to the next lower curve (or the bottom border).
#              Source tie ``(p, i, j)`` comes from the **upper** curve in each band.
TIE_LINES_DIRECTION = "down"
# If True, the outermost graph line (top for "up", bottom for "down") extends tie art to the axis border.
# If False, that outermost band is omitted — only interior inter-curve bands are drawn.
TIE_LINES_TO_BORDER = False
# Colormap or color for tie **lines**.  Set to a Matplotlib colormap name (e.g. "gist_rainbow", "hsv",
# "jet", "turbo", "viridis", "plasma", "inferno", "magma", "cividis", "nipy_spectral", "twilight",
# "coolwarm", "winter", "spring", "summer", "autumn", "wistia") to color by ``TIE_LINE_COLOR_MODE``.
# Set to a simple Matplotlib color name (e.g. "black", "red", "blue") to use a fixed color for all
# tie lines.  Set to ``""`` (or any falsy value) to disable tie lines entirely.
TIE_COLORMAP = None #"gist_rainbow" 
# Colormap for filled regions between tie ``p`` values.  Same colormap names as above.
# Set to ``""`` (or any falsy value) to disable fills.  Both ``TIE_COLORMAP`` and ``FILL_COLORMAP``
# can be active at the same time (fills render under tie lines).
FILL_COLORMAP = "gist_rainbow"
# ``"left"`` → each filled region takes the color of the tie line on its **left** edge.
# ``"right"`` → each filled region takes the color of the tie line on its **right** edge.
# Margin regions with no tie on the chosen side are left unfilled.
FILL_FROM = "left"
# If True and ``VP_P_RANGE`` is ``"full"``, the sense of ``FILL_FROM`` is reversed for ``p < 0.5``.
# Combined with a two-letter ``TIE_LINE_COLOR_MODE`` (e.g. ``"ij"``), this means the region just right of
# ``p = 0.5`` uses the second letter (``j``) of the tie at 0.5, and the region just left uses the first (``i``).
# In non-fill mode, the vertical tie line at exactly ``p = 0.5`` is colored as if it were on the right half
# (``p >= 0.5``), since there is only one line at the midpoint.
FLIP_FILL_ON_LEFT = True
# Target stroke thickness in **device pixels** (see ``_lw_device_px``).  Vertical tie segments are
# axis-aligned and usually rasterize to ~1 column; E/n polylines at the same Matplotlib linewidth
# typically read ~2× thicker on Agg, so curves default to half the tie width to match visually.
TIE_LINE_DEVICE_PX = 1.0
GRAPH_LINE_DEVICE_PX = 0.1
# Figure, axes patch, and PNG output background (matplotlib facecolor).
FIGURE_BACKGROUND = "white"


# Tolerance for matching requested ``p`` to coordinates stored in the pickle (uniform grids; float noise).
_P_MATCH_REL_TOL = 1e-15
_P_MATCH_ABS_TOL = 1e-12


@dataclass(frozen=True)
class BinomialGrid:
    """Requested ``n`` range, ``p`` grid, and flattened binomial rows (``n`` outer, ``p`` inner)."""

    n_min: int
    n_max: int
    p_steps: int
    p_values: tuple[float, ...]
    binomial_flat: list

    @property
    def p_half_start(self) -> int:
        return (self.p_steps - 1) // 2


def _uniform_p_values(p_steps: int) -> tuple[float, ...]:
    if p_steps < 2:
        raise ValueError("p_steps must be at least 2")
    return tuple(i / (p_steps - 1) for i in range(p_steps))


def _pickle_p_values(data: dict, path: str) -> tuple[float, ...]:
    """``p`` coordinates for pickle rows, from ``p_values`` key or uniform formula (older pickles)."""
    p_st = int(data["p_steps"])
    raw = data.get("p_values")
    if raw is not None:
        pv = tuple(float(x) for x in raw)
        if len(pv) != p_st:
            print(
                f"ERROR: {path!r} has p_values length {len(pv)} != p_steps={p_st}.\n",
                file=sys.stderr,
            )
            sys.exit(1)
        return pv
    return _uniform_p_values(p_st)


def _map_desired_p_to_pickle_columns(
    desired_p: tuple[float, ...], pkl_p: tuple[float, ...]
) -> list[int | None]:
    """For each desired ``p``, index into pickle's ``p`` axis, or ``None`` if no column matches."""
    out: list[int | None] = []
    for p_d in desired_p:
        j_found: int | None = None
        for j, p_p in enumerate(pkl_p):
            if math.isclose(p_d, p_p, rel_tol=_P_MATCH_REL_TOL, abs_tol=_P_MATCH_ABS_TOL):
                j_found = j
                break
        out.append(j_found)
    return out


def _build_binomial_grid_from_pickle(
    path: str,
    n_req_min: int,
    n_req_max: int,
    p_steps_req: int,
) -> BinomialGrid:
    """
    Load ``graph_data.pkl``, check it is internally consistent, then verify it contains every
    ``(n, p)`` sample needed for the requested uniform grid.  Exit with diagnostics if anything is missing.
    """
    if not os.path.isfile(path):
        print(
            f"ERROR: Graph data file not found:\n  {os.path.abspath(path)}\n",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"ERROR: Could not read graph pickle {path!r}: {e}\n", file=sys.stderr)
        sys.exit(1)

    n_lo = data.get("n_min")
    n_hi = data.get("n_max")
    p_st = data.get("p_steps")
    bd = data.get("binomial_data")
    missing_keys = [k for k, v in (("n_min", n_lo), ("n_max", n_hi), ("p_steps", p_st), ("binomial_data", bd)) if v is None]
    if missing_keys:
        print(
            f"ERROR: {path!r} is missing required key(s): {', '.join(missing_keys)}.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    n_lo, n_hi, p_st = int(n_lo), int(n_hi), int(p_st)
    if n_lo > n_hi:
        print(
            f"ERROR: {path!r} has invalid n_min={n_lo} > n_max={n_hi}.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    if p_st < 2:
        print(
            f"ERROR: {path!r} has p_steps={p_st}; need at least 2.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    expected_len = (n_hi - n_lo + 1) * p_st
    if len(bd) != expected_len:
        print(
            f"ERROR: {path!r} binomial_data length {len(bd)} != expected "
            f"({n_hi} - {n_lo} + 1) * {p_st} = {expected_len}.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    pkl_p = _pickle_p_values(data, path)

    missing_n = [n for n in range(n_req_min, n_req_max + 1) if n < n_lo or n > n_hi]
    if missing_n:
        print(
            f"ERROR: graph_data.pkl covers n = {n_lo}..{n_hi}; requested GRAPH_N_MIN..GRAPH_N_MAX = "
            f"{n_req_min}..{n_req_max}.\n"
            f"  Missing n values: {len(missing_n)} (not in pickle).\n"
            f"  First missing n (up to 10): {missing_n[:10]}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    desired_p = _uniform_p_values(p_steps_req)
    p_col_map = _map_desired_p_to_pickle_columns(desired_p, pkl_p)
    bad_p_ix = [i for i, j in enumerate(p_col_map) if j is None]

    missing_samples: list[tuple[int, float]] = []
    for n in range(n_req_min, n_req_max + 1):
        for p_ix in bad_p_ix:
            missing_samples.append((n, float(desired_p[p_ix])))

    if bad_p_ix:
        p_src = "p_values key in file" if "p_values" in data else "uniform grid from p_steps only"
        print(
            f"ERROR: graph_data.pkl p grid does not contain all requested p coordinates.\n"
            f"  Pickle: p_steps={p_st}, n={n_lo}..{n_hi} ({p_src}).\n"
            f"  Requested: GRAPH_P_STEPS={p_steps_req} (uniform p on [0, 1]).\n"
            f"  Missing (n, p) samples: {len(missing_samples)}.\n"
            f"  Requested p indices with no matching pickle column: {len(bad_p_ix)}.\n"
            f"  First missing (n, p) pairs (up to 10):",
            file=sys.stderr,
        )
        for pair in missing_samples[:10]:
            print(f"    n={pair[0]}, p={pair[1]!r}", file=sys.stderr)
        sys.exit(1)

    out: list = []
    for n in range(n_req_min, n_req_max + 1):
        row0 = (n - n_lo) * p_st
        for p_ix, j_pkl in enumerate(p_col_map):
            k = row0 + j_pkl
            if k < 0 or k >= len(bd):
                print(
                    f"ERROR: Internal index out of range for n={n}, pickle column {j_pkl}.\n",
                    file=sys.stderr,
                )
                sys.exit(1)
            out.append(bd[k])

    return BinomialGrid(
        n_min=n_req_min,
        n_max=n_req_max,
        p_steps=p_steps_req,
        p_values=desired_p,
        binomial_flat=out,
    )


@dataclass(frozen=True)
class TieColorSpec:
    """Parsed ``TIE_LINE_COLOR_MODE``: black, single ``i``/``j``, or dual ``ij``-style pair."""

    black: bool = False
    single_axis: str | None = None
    dual_first: str | None = None
    dual_second: str | None = None

    @staticmethod
    def parse(s: str) -> TieColorSpec:
        v = s.strip().lower()
        if v == "black":
            return TieColorSpec(black=True)
        if len(v) == 1 and v in "ij":
            return TieColorSpec(single_axis=v)
        if len(v) == 2 and v[0] in "ij" and v[1] in "ij":
            return TieColorSpec(dual_first=v[0], dual_second=v[1])
        raise ValueError(
            "TIE_LINE_COLOR_MODE must be 'black', 'i', 'j', or two of [ij] "
            f"(e.g. 'ij'), case insensitive; got {s!r}"
        )

    def axis_at_p(self, p: float, vp_range_norm: str) -> str:
        if self.black:
            return "black"
        if self.single_axis is not None:
            return self.single_axis
        assert self.dual_first is not None and self.dual_second is not None
        if vp_range_norm != "full":
            return self.dual_first
        return self.dual_first if p < 0.5 else self.dual_second


def _tie_ij_rightmost_in(
    ent_sorted: list[tuple[float, int | None, int | None]],
    p_lo: float,
    p_hi: float,
) -> tuple[int | None, int | None]:
    """Among ties with ``p_lo < p <= p_hi``, use the rightmost ``p`` (same spirit as exact right edge)."""
    cand = [e for e in ent_sorted if p_lo + 1e-15 < e[0] <= p_hi + 1e-15]
    if not cand:
        return None, None
    e = max(cand, key=lambda x: x[0])
    return e[1], e[2]


def _expand_strips_for_dual_full(
    strips: list[tuple[float, float, int | None, int | None]],
    ent_sorted: list[tuple[float, int | None, int | None]],
    tie_spec: TieColorSpec,
    vp_range_norm: str,
) -> list[tuple[float, float, int | None, int | None]]:
    """If dual mode on full range, split strips that cross ``p = 0.5`` and set ``(i,j)`` per sub-strip."""
    if tie_spec.dual_first is None or vp_range_norm != "full":
        return strips
    out: list[tuple[float, float, int | None, int | None]] = []
    for p_l, p_r, ti, tj in strips:
        if p_l + 1e-15 < 0.5 < p_r - 1e-15:
            t_left = _tie_ij_rightmost_in(ent_sorted, p_l, 0.5)
            t_right = _tie_ij_rightmost_in(ent_sorted, 0.5, p_r)
            out.append((p_l, 0.5, t_left[0], t_left[1]))
            out.append((0.5, p_r, t_right[0], t_right[1]))
        else:
            out.append((p_l, p_r, ti, tj))
    return out


def _vp_p_window(
    mode: str, grid: BinomialGrid
) -> tuple[float, float, int, int]:
    """``(p_axes_lo, p_axes_hi, p_ix_lo, p_ix_hi)`` inclusive indices into ``grid.p_values``."""
    v = mode.strip().lower()
    ps = grid.p_steps
    ph = grid.p_half_start
    if v == "full":
        return 0.0, 1.0, 0, ps - 1
    if v == "left":
        return 0.0, 0.5, 0, ph
    if v == "right":
        return 0.5, 1.0, ph, ps - 1
    raise ValueError(
        'VP_P_RANGE must be "full", "left", or "right" (case insensitive), '
        f"got {mode!r}"
    )


def _parse_tie_lines_direction(s: str) -> str:
    """Return ``"up"`` or ``"down"`` (normalized)."""
    v = str(s).strip().lower()
    if v == "up":
        return "up"
    if v == "down":
        return "down"
    raise ValueError(
        'TIE_LINES_DIRECTION must be "up" or "down" '
        f"(case insensitive), got {s!r}"
    )


def _tie_source_n_for_band(
    ki: int, nk: int, n_list: list[int], direction: str
) -> int:
    """``n`` whose ties drive this band.

    ``"up"``: source = lower curve (``n_list[ki - 1]``) for interior bands; for the top-border band
    (``ki == len(n_list) - 1`` when used as border), source = ``nk`` (the top curve itself).
    ``"down"``: source = ``nk`` (upper curve).
    """
    if direction == "up":
        return n_list[ki - 1] if ki > 0 else nk
    return nk


_cmap_cache: dict[str, object] = {}


def _cmap_rgb(cmap_name: str, t: float) -> tuple[float, float, float]:
    """RGB sample from a named Matplotlib colormap at ``t`` in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    name = str(cmap_name).strip()
    cmap = _cmap_cache.get(name)
    if cmap is None:
        try:
            cmap = matplotlib.colormaps[name]
        except (AttributeError, KeyError):
            cmap = cm.get_cmap(name)
        _cmap_cache[name] = cmap
    rgba = cmap(t)
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]))


def _is_mpl_color(s: str) -> bool:
    """True if ``s`` is a recognized Matplotlib color name (not a colormap)."""
    try:
        matplotlib.colors.to_rgba(s)
        return True
    except ValueError:
        return False


def _load_tie_draw_entries(
    path: str,
    n_vals: list[int],
    tie_p_min: float,
    tie_p_max: float,
) -> dict[int, list[tuple[float, int | None, int | None]]]:
    """n -> list of (p, i, j) with ``tie_p_min <= p <= tie_p_max``.

    ``i``/``j`` may be None when the pickle has ``float_by_n`` only (no pair lists).
    """
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return {}
    fwp = data.get("float_with_pairs_by_n") or data.get("by_n") or {}
    float_by_n = data.get("float_by_n", {})
    out: dict[int, list[tuple[float, int | None, int | None]]] = {}
    for n in n_vals:
        segs: list[tuple[float, int | None, int | None]] = []
        recs = fwp.get(n)
        if recs is not None:
            for item in recs:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                p_raw, pairs = item[0], item[1]
                pf = float(p_raw)
                if not (tie_p_min <= pf <= tie_p_max):
                    continue
                pr = round(pf, 6)
                plist = pairs or []
                if not plist:
                    segs.append((pr, None, None))
                    continue
                for ij in plist:
                    if not isinstance(ij, (list, tuple)) or len(ij) != 2:
                        continue
                    segs.append((pr, int(ij[0]), int(ij[1])))
        else:
            raw = float_by_n.get(n)
            if raw is None:
                continue
            arr = np.atleast_1d(np.asarray(raw, dtype=float))
            for pf in arr.flat:
                if not np.isfinite(pf) or not (tie_p_min <= float(pf) <= tie_p_max):
                    continue
                segs.append((round(float(pf), 6), None, None))
        if segs:
            out[n] = segs
    return out


def _coord_range_for_n(
    entries: list[tuple[float, int | None, int | None]], key: str
) -> tuple[int, int] | None:
    """Min/max of i or of j over this n's draw entries (not mixed)."""
    idx = 1 if key == "i" else 2
    vals = [e[idx] for e in entries if e[idx] is not None]
    if not vals:
        return None
    return min(vals), max(vals)


def _tie_rgba_for_segment(
    axis: str,
    ii: int | None,
    jj: int | None,
    alpha: float,
    cmap_name: str = "",
    fixed_color: tuple[float, float, float] | None = None,
    i_range: tuple[int, int] | None = None,
    j_range: tuple[int, int] | None = None,
) -> tuple[float, float, float, float]:
    """``axis`` is ``black``, ``i``, or ``j`` (colormap channel for this sample).

    If ``fixed_color`` is set, use it directly (ignoring ``axis`` / ``cmap_name``).
    Otherwise look up ``(i,j)`` in ``cmap_name``.
    ``i_range`` / ``j_range`` are pre-computed ``(lo, hi)`` for the band's entries.
    Missing ``i``/``j`` (or no numeric range for the chosen axis) yields **black** ``(0,0,0,alpha)``.
    """
    if fixed_color is not None:
        return (fixed_color[0], fixed_color[1], fixed_color[2], float(alpha))
    if axis == "black" or ii is None or jj is None:
        return (0.0, 0.0, 0.0, float(alpha))
    if axis == "i":
        if i_range is None:
            return (0.0, 0.0, 0.0, float(alpha))
        lo, hi = i_range
        t = (ii - lo) / (hi - lo) if hi > lo else 0.0
    else:
        if j_range is None:
            return (0.0, 0.0, 0.0, float(alpha))
        lo, hi = j_range
        t = (jj - lo) / (hi - lo) if hi > lo else 0.0
    r, g, b = _cmap_rgb(cmap_name, t)
    return (r, g, b, float(alpha))


def _y_top_bottom_for_band(
    vp_xy: dict[int, tuple[np.ndarray, np.ndarray]],
    n_list: list[int],
    ki: int,
    nk: int,
    p_use: float,
    y_axis_lo: float,
) -> tuple[float, float]:
    """Lower and upper ``y`` at ``p_use`` for swap band ``nk`` (upper curve = ``nk``)."""
    x_c, y_c = vp_xy[nk]
    y_top = interpolate_y_at_p(x_c, y_c, float(p_use))
    if ki == 0:
        y_bot = y_axis_lo
    else:
        x_p, y_p = vp_xy[n_list[ki - 1]]
        y_bot = interpolate_y_at_p(x_p, y_p, float(p_use))
    return y_bot, y_top


def _fill_strip_polygon_xy(
    p_left: float,
    p_right: float,
    x_top: np.ndarray,
    y_top: np.ndarray,
    x_bot: np.ndarray | None,
    y_bot: np.ndarray | None,
    y_flat_bottom: float | None,
) -> tuple[list[float], list[float]]:
    """Vertices for a closed strip: bottom p_left→p_right, right vertical, top p_right→p_left, close.

    Bottom follows the lower-curve polyline when ``x_bot``/``y_bot`` are set; otherwise a horizontal
    segment at ``y_flat_bottom``.  Top follows ``x_top``/``y_top`` between the two ``p`` values.
    """
    yt_l = interpolate_y_at_p(x_top, y_top, p_left)
    yt_r = interpolate_y_at_p(x_top, y_top, p_right)
    if x_bot is None or y_bot is None:
        if y_flat_bottom is None:
            return [], []
        yf = float(y_flat_bottom)
        xb_pts = [float(p_left), float(p_right)]
        yb_pts = [yf, yf]
    else:
        yb_l = interpolate_y_at_p(x_bot, y_bot, p_left)
        yb_r = interpolate_y_at_p(x_bot, y_bot, p_right)
        mb = (x_bot > p_left) & (x_bot < p_right)
        xb_pts = [float(p_left)] + x_bot[mb].astype(float).tolist() + [float(p_right)]
        yb_pts = [float(yb_l)] + y_bot[mb].astype(float).tolist() + [float(yb_r)]

    mt = (x_top > p_left) & (x_top < p_right)
    xf_top = [float(p_left)] + x_top[mt].astype(float).tolist() + [float(p_right)]
    yf_top = [float(yt_l)] + y_top[mt].astype(float).tolist() + [float(yt_r)]
    xr_top = xf_top[::-1]
    yr_top = yf_top[::-1]

    px: list[float] = [xb_pts[0]]
    py: list[float] = [yb_pts[0]]
    for i in range(1, len(xb_pts)):
        px.append(xb_pts[i])
        py.append(yb_pts[i])
    if abs(float(yb_pts[-1]) - float(yr_top[0])) > 1e-14:
        px.append(float(p_right))
        py.append(float(yr_top[0]))
    for i in range(1, len(xr_top)):
        px.append(float(xr_top[i]))
        py.append(float(yr_top[i]))
    return px, py


def _tie_entry_sort_key(
    e: tuple[float, int | None, int | None],
) -> tuple[float, int, int]:
    p, i, j = e[0], e[1], e[2]
    return (p, -1 if i is None else i, -1 if j is None else j)


def _binomial_index(n: int, p_idx: int, grid: BinomialGrid) -> int:
    return (n - grid.n_min) * grid.p_steps + p_idx


def expected_rank(point: dict) -> float:
    n = len(point["x"]) - 1
    y = np.asarray(point["y"], dtype=float)
    perm = np.asarray(point["perm"], dtype=int)
    return float(np.dot(np.arange(n + 1, dtype=float), y[perm]))


def subtract_endpoint_chord(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Subtract the secant through the first and last finite ``(x, y)`` (``OBDgraphExplorer1`` semantics)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = int(min(xs.size, ys.size))
    if n == 0:
        return ys.copy()
    finite = np.isfinite(xs[:n]) & np.isfinite(ys[:n])
    if not np.any(finite):
        return ys[:n].astype(float, copy=True)
    idx = np.flatnonzero(finite)
    i0 = int(idx[0])
    i1 = int(idx[-1])
    if i1 <= i0:
        return ys[:n].astype(float, copy=True)
    x0, x1 = float(xs[i0]), float(xs[i1])
    y0, y1 = float(ys[i0]), float(ys[i1])
    dx = x1 - x0
    if dx == 0.0:
        return ys[:n].astype(float, copy=True)
    out = ys[:n].astype(float, copy=True)
    L = y0 + (y1 - y0) * (xs[:n] - x0) / dx
    m = np.isfinite(out)
    out[m] = out[m] - L[m]
    return out


def interpolate_y_at_p(xa: np.ndarray, ya: np.ndarray, p_target: float) -> float:
    if xa.size == 0:
        return float("nan")
    if p_target <= float(xa[0]):
        return float(ya[0])
    if p_target >= float(xa[-1]):
        return float(ya[-1])
    i = int(np.searchsorted(xa, p_target, side="right")) - 1
    i = max(0, min(i, xa.size - 2))
    x0, x1 = float(xa[i]), float(xa[i + 1])
    y0, y1 = float(ya[i]), float(ya[i + 1])
    if x1 == x0:
        return y0
    t = (p_target - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def build_vp_xy(
    grid: BinomialGrid,
    n: int,
    p_ix_lo: int,
    p_ix_hi: int,
    *,
    explorer_endpoint_y: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """If ``explorer_endpoint_y``, match ``OBDgraphExplorer1`` ``buildVpXY`` when ``scaleMode`` is
    ``endpoint``: ``y = E[rank]`` (sorted) or ``n·p`` (unsorted), ignoring ``SCALED``.
    """
    bd = grid.binomial_flat
    pv = grid.p_values
    num_pts = p_ix_hi - p_ix_lo + 1
    x_arr = np.empty(num_pts, dtype=float)
    y_arr = np.empty(num_pts, dtype=float)
    for k, p_ix in enumerate(range(p_ix_lo, p_ix_hi + 1)):
        idx = _binomial_index(n, p_ix, grid)
        pt = bd[idx]
        p_val = float(pv[p_ix])
        e = expected_rank(pt) if SORTED else n * p_val
        if explorer_endpoint_y:
            y_arr[k] = e
        elif SCALED and not SORTED:
            y_arr[k] = p_val
        elif SCALED:
            y_arr[k] = e / n
        else:
            y_arr[k] = e
        x_arr[k] = p_val
    return x_arr, y_arr


def _lw_device_px(device_pixels: float) -> float:
    """Matplotlib ``linewidth`` in points for ``device_pixels`` at ``DPI`` (72 pt per inch)."""
    return 72.0 * float(device_pixels) / float(DPI)


def main() -> None:
    t_program_start = time.monotonic()
    if GRAPH_N_MIN > GRAPH_N_MAX:
        print("ERROR: GRAPH_N_MIN must be <= GRAPH_N_MAX.", file=sys.stderr)
        sys.exit(1)
    if GRAPH_P_STEPS < 2:
        print("ERROR: GRAPH_P_STEPS must be at least 2.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading graph data from {GRAPH_DATA_PATH}...")
    t_load_graph = time.monotonic()
    grid = _build_binomial_grid_from_pickle(
        GRAPH_DATA_PATH, GRAPH_N_MIN, GRAPH_N_MAX, GRAPH_P_STEPS
    )
    print(
        f"  Grid: n = {grid.n_min}..{grid.n_max}, p_steps = {grid.p_steps} "
        f"(sliced from graph_data.pkl)"
    )
    print(f"Graph data load: {time.monotonic() - t_load_graph:.2f}s")

    try:
        tie_spec = TieColorSpec.parse(TIE_LINE_COLOR_MODE)
        vp_p_min, vp_p_max, p_ix_lo, p_ix_hi = _vp_p_window(VP_P_RANGE, grid)
        tie_dir = _parse_tie_lines_direction(TIE_LINES_DIRECTION)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    vp_range_norm = VP_P_RANGE.strip().lower()

    print(f"Loading tie points from {TIE_POINTS_PATH}...")
    t_load_ties = time.monotonic()
    n_list = list(range(GRAPH_N_MIN, GRAPH_N_MAX + 1))
    tie_draw = _load_tie_draw_entries(
        TIE_POINTS_PATH, n_list, vp_p_min, vp_p_max
    )

    if CLIP_TO_TIE_RANGE:
        all_tie_p = [e[0] for ents in tie_draw.values() for e in ents]
        if all_tie_p:
            tp_lo = float(min(all_tie_p))
            tp_hi = float(max(all_tie_p))
            pv = grid.p_values
            new_ix_lo = next(
                (i for i in range(p_ix_lo, p_ix_hi + 1) if pv[i] >= tp_lo - 1e-12),
                p_ix_lo,
            )
            new_ix_hi = next(
                (i for i in range(p_ix_hi, p_ix_lo - 1, -1) if pv[i] <= tp_hi + 1e-12),
                p_ix_hi,
            )
            p_ix_lo = new_ix_lo
            p_ix_hi = new_ix_hi
            vp_p_min = float(pv[p_ix_lo])
            vp_p_max = float(pv[p_ix_hi])
            print(f"CLIP_TO_TIE_RANGE: clipped to p=[{vp_p_min}, {vp_p_max}] "
                  f"(grid indices {p_ix_lo}..{p_ix_hi})")

    print(f"Tie data load: {time.monotonic() - t_load_ties:.2f}s")

    t_prep_start = time.monotonic()

    endpoint_chord_active = bool(ENDPOINT_CHORD_DETREND)
    if endpoint_chord_active:
        print("Applying endpoint chord detrend"
              + (" (split at p=0.5)." if vp_range_norm == "full" else " (half-range p window)."))

    t_curves = time.monotonic()
    vp_xy: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_y_arrays: list[np.ndarray] = []
    for n in n_list:
        x_arr, y_arr = build_vp_xy(
            grid,
            n,
            p_ix_lo,
            p_ix_hi,
            explorer_endpoint_y=endpoint_chord_active,
        )
        if endpoint_chord_active:
            if vp_range_norm == "full":
                mid_ix = np.searchsorted(x_arr, 0.5, side="right")
                y_left = subtract_endpoint_chord(x_arr[:mid_ix], y_arr[:mid_ix])
                y_right = subtract_endpoint_chord(x_arr[mid_ix - 1:], y_arr[mid_ix - 1:])
                y_arr = np.concatenate([y_left, y_right[1:]])
            else:
                y_arr = subtract_endpoint_chord(x_arr, y_arr)
        vp_xy[n] = (x_arr, y_arr)
        finite_y = y_arr[np.isfinite(y_arr)]
        if finite_y.size > 0:
            all_y_arrays.append(finite_y)

    if not all_y_arrays:
        print("ERROR: No y data for curves.", file=sys.stderr)
        sys.exit(1)

    all_y_concat = np.concatenate(all_y_arrays)
    y_lo = float(all_y_concat.min())
    y_hi = float(all_y_concat.max())
    y_span = y_hi - y_lo
    y_min_span = 1e-3
    if y_span < y_min_span:
        y_mid = (y_lo + y_hi) / 2.0
        y_lo = y_mid - y_min_span / 2.0
        y_hi = y_mid + y_min_span / 2.0
    else:
        y_pad = max(y_span * 0.05, y_min_span * 0.01)
        y_lo -= y_pad
        y_hi += y_pad

    print(f"Curves and y-range: {time.monotonic() - t_curves:.2f}s")

    t_band_prep = time.monotonic()

    tie_alpha = float(TIE_LINE_OPACITY)
    graph_opacity_base = float(np.clip(GRAPH_LINE_OPACITY, 0.0, 1.0))
    graph_opacity_k = float(GRAPH_LINE_OPACITY_K)

    do_fill = bool(FILL_COLORMAP)
    do_tie_lines = bool(TIE_COLORMAP)
    fill_cmap = str(FILL_COLORMAP).strip() if do_fill else ""
    tie_cmap = str(TIE_COLORMAP).strip() if do_tie_lines else ""
    tie_fixed_color: tuple[float, float, float] | None = None
    if do_tie_lines and _is_mpl_color(tie_cmap):
        r, g, b, _ = matplotlib.colors.to_rgba(tie_cmap)
        tie_fixed_color = (float(r), float(g), float(b))

    lw_tie = _lw_device_px(TIE_LINE_DEVICE_PX)
    lw_graph = _lw_device_px(GRAPH_LINE_DEVICE_PX)
    fig_w_in = PNG_WIDTH_PX / DPI
    fig_h_in = PNG_HEIGHT_PX / DPI

    fill_specs: list[
        tuple[list[float], list[float], tuple[float, float, float, float]]
    ] = []
    tie_specs: list[
        tuple[float, float, float, tuple[float, float, float, float]]
    ] = []

    # Build the list of bands to draw.  Each tuple:
    #   (n_geo_upper, n_geo_lower_or_None, n_tie, flat_border_y_or_None)
    # ``n_geo_upper`` / ``n_geo_lower`` are the two bounding curves ordered by geometric position
    # (None → flat border at ``flat_border_y``).
    # ``n_tie`` is which curve's tie list drives the band's art.
    bands: list[tuple[int | None, int | None, int, float | None]] = []

    # Curves sorted by geometric position (bottom to top).  In detrend mode higher n is
    # geometrically lower, so the order reverses.
    geo_order = list(reversed(n_list)) if endpoint_chord_active else list(n_list)

    for i in range(1, len(geo_order)):
        n_above = geo_order[i]
        n_below = geo_order[i - 1]
        n_tie = n_below if tie_dir == "up" else n_above
        bands.append((n_above, n_below, n_tie, None))

    if TIE_LINES_TO_BORDER:
        if tie_dir == "up":
            bands.append((None, geo_order[-1], geo_order[-1], y_hi))
        else:
            bands.append((geo_order[0], None, geo_order[0], y_lo))

    # Draw each band: fills (zorder=1) then tie lines (zorder=1.5) so lines render on top.
    for n_upper, n_lower, n_tie, flat_y in bands:
        entries = tie_draw.get(n_tie, [])
        if not entries:
            continue

        band_i_range = _coord_range_for_n(entries, "i")
        band_j_range = _coord_range_for_n(entries, "j")

        # --- filled strips ---
        if do_fill:
            ps = sorted({e[0] for e in entries})
            if ps:
                ent_sorted = sorted(entries, key=_tie_entry_sort_key)

                _ent_lookup: dict[float, tuple[int | None, int | None]] = {
                    e[0]: (e[1], e[2]) for e in ent_sorted
                }

                def _tie_ij_at(p_anchor: float) -> tuple[int | None, int | None]:
                    return _ent_lookup.get(p_anchor, (None, None))

                fill_from_base = FILL_FROM.strip().lower()
                flip_left = FLIP_FILL_ON_LEFT and vp_range_norm == "full"

                def _fill_side_at(p_mid: float) -> str:
                    if flip_left and p_mid < 0.5:
                        return "right" if fill_from_base == "left" else "left"
                    return fill_from_base

                strips: list[tuple[float, float, int | None, int | None]] = []
                for k in range(len(ps) - 1):
                    p_l, p_r = ps[k], ps[k + 1]
                    side = _fill_side_at(0.5 * (p_l + p_r))
                    if side == "left":
                        ti, tj = _tie_ij_at(p_l)
                    else:
                        ti, tj = _tie_ij_at(p_r)
                    strips.append((p_l, p_r, ti, tj))
                if ps[0] > vp_p_min + 1e-15:
                    side = _fill_side_at(0.5 * (vp_p_min + ps[0]))
                    if side == "right":
                        ti0, tj0 = _tie_ij_at(ps[0])
                        strips.insert(0, (vp_p_min, ps[0], ti0, tj0))
                if ps[-1] < vp_p_max - 1e-15:
                    side = _fill_side_at(0.5 * (ps[-1] + vp_p_max))
                    if side == "left":
                        tia, tja = _tie_ij_at(ps[-1])
                        strips.append((ps[-1], vp_p_max, tia, tja))

                strips = _expand_strips_for_dual_full(
                    strips, ent_sorted, tie_spec, vp_range_norm
                )

                for p_left, p_right, ti, tj in strips:
                    p_mid = 0.5 * (float(p_left) + float(p_right))
                    axis = tie_spec.axis_at_p(p_mid, vp_range_norm)
                    rgba = _tie_rgba_for_segment(
                        axis, ti, tj, tie_alpha,
                        cmap_name=fill_cmap,
                        i_range=band_i_range,
                        j_range=band_j_range,
                    )
                    if n_upper is not None:
                        x_up, y_up = vp_xy[n_upper]
                    else:
                        x_up, y_up = None, None
                    if n_lower is not None:
                        x_dn, y_dn = vp_xy[n_lower]
                    else:
                        x_dn, y_dn = None, None

                    if x_up is not None and x_dn is not None:
                        fx, fy = _fill_strip_polygon_xy(
                            float(p_left), float(p_right),
                            x_up, y_up, x_dn, y_dn, None,
                        )
                    elif x_up is not None:
                        fx, fy = _fill_strip_polygon_xy(
                            float(p_left), float(p_right),
                            x_up, y_up, None, None, flat_y,
                        )
                    else:
                        fx, fy = _fill_strip_polygon_xy(
                            float(p_left), float(p_right),
                            x_dn, y_dn, None, None, flat_y,
                        )
                    if len(fx) < 3 or not all(
                        np.isfinite(v) for v in (*fx, *fy)
                    ):
                        continue
                    fill_specs.append((fx, fy, rgba))

        # --- vertical tie lines ---
        if do_tie_lines:
            for p_use, ti, tj in entries:
                axis = tie_spec.axis_at_p(float(p_use), vp_range_norm)
                rgba = _tie_rgba_for_segment(
                    axis, ti, tj, tie_alpha,
                    cmap_name=tie_cmap,
                    fixed_color=tie_fixed_color,
                    i_range=band_i_range,
                    j_range=band_j_range,
                )
                if n_upper is not None:
                    y_top = interpolate_y_at_p(*vp_xy[n_upper], float(p_use))
                else:
                    y_top = flat_y
                if n_lower is not None:
                    y_bot = interpolate_y_at_p(*vp_xy[n_lower], float(p_use))
                else:
                    y_bot = flat_y
                if np.isfinite(y_bot) and np.isfinite(y_top):
                    tie_specs.append((float(p_use), float(y_bot), float(y_top), rgba))

    t_curve_specs = time.monotonic()
    curve_specs: list[tuple[np.ndarray, np.ndarray, float]] = []
    if graph_opacity_base > 0.0:
        for n in n_list:
            if graph_opacity_k == 0.0:
                alpha_n = graph_opacity_base
            else:
                denom = graph_opacity_k * (n - 2) + 1.0
                if denom <= 0.0:
                    continue
                alpha_n = float(np.clip(graph_opacity_base / denom, 0.0, 1.0))
            if alpha_n <= 0.0:
                continue
            x_arr, y_arr = vp_xy[n]
            curve_specs.append((x_arr, y_arr, alpha_n))

    print(
        f"Band geometry prep (fills, ties): "
        f"{t_curve_specs - t_band_prep:.2f}s"
    )
    print(
        f"Curve overlay list: {time.monotonic() - t_curve_specs:.2f}s"
    )

    if endpoint_chord_active:
        y_label = ("E[rank]" if SORTED else "E[X]") + " − chord"
    else:
        y_label = "E/n" if SCALED else ("E[rank]" if SORTED else "E[X]")
    title = (
        ("E[rank]" if SORTED else "E[X]")
        + ("" if endpoint_chord_active else ("/n" if SCALED else ""))
        + (" (endpoint detrended)" if endpoint_chord_active else "")
        + " vs p (n = "
        + str(GRAPH_N_MIN)
        + "–"
        + str(GRAPH_N_MAX)
        + ", banded swap lines)"
    )

    t_prep_total = time.monotonic() - t_prep_start
    print(f"Prepared plot data (total): {t_prep_total:.2f}s")
    print("Creating graph...")
    t_render_start = time.monotonic()

    fig, ax = plt.subplots(
        figsize=(fig_w_in, fig_h_in),
        dpi=DPI,
        facecolor=FIGURE_BACKGROUND,
    )
    ax.set_facecolor(FIGURE_BACKGROUND)

    for fx, fy, rgba in fill_specs:
        ax.fill(
            fx, fy,
            facecolor=rgba, linewidth=0,
            antialiased=False, zorder=1,
        )
    for p_use, y_bot, y_top, rgba in tie_specs:
        ax.plot(
            [p_use, p_use], [y_bot, y_top],
            color=rgba, linewidth=lw_tie,
            solid_capstyle="butt", antialiased=False, zorder=1.5,
        )
    for x_arr, y_arr, alpha_n in curve_specs:
        ax.plot(
            x_arr,
            y_arr,
            color=(0.0, 0.0, 0.0, alpha_n),
            linewidth=lw_graph,
            solid_capstyle="butt",
            solid_joinstyle="miter",
            antialiased=False,
            zorder=2,
        )

    ax.set_xlim(vp_p_min, vp_p_max)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("p")
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=10, pad=4)
    ax.grid(False)
    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        bottom=SUBPLOT_BOTTOM,
        top=SUBPLOT_TOP,
    )

    t_render = time.monotonic() - t_render_start
    print(f"Rendered graph draw: {t_render:.2f}s")

    t_save = time.monotonic()
    fig.savefig(
        OUTPUT_PNG_PATH,
        dpi=DPI,
        facecolor=FIGURE_BACKGROUND,
        edgecolor="none",
        pad_inches=SAVE_PAD_INCHES,
    )
    print(f"PNG file write: {time.monotonic() - t_save:.2f}s")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PNG_PATH} ({PNG_WIDTH_PX}×{PNG_HEIGHT_PX} px at {DPI} DPI).")
    print(f"Total run time: {time.monotonic() - t_program_start:.2f}s")


if __name__ == "__main__":
    main()
