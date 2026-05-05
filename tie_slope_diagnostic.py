from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from OBDsaveSourceData import (
    DEFAULT_HP_DPS,
    DEFAULT_TIE_SLOPE_A,
    DEFAULT_TIE_SLOPE_EPS_MIN,
    DEFAULT_TIE_SLOPE_FLAT_TOL,
    DEFAULT_TOL_SCALE,
    _expected_sorted_rank_many,
    _expected_sorted_rank_many_high_precision,
    _tie_slope_records_for_n,
    all_tie_points_float_with_pairs,
)

# Default n value for diagnostics.
DEFAULT_DIAGNOSTIC_N = 1000 #554 # 985

# Minima-category labels used in diagnostics.
MINIMA_CATEGORY_DEADZONE = "minima_deadzone"
MINIMA_CATEGORY_RESOLVED = "minima_resolved"
MINIMA_CATEGORY_UNRESOLVED = "minima_unresolved"
# Which flagged points to plot by default:
# - "deadzone": all dead-zone points (resolved + unresolved)
# - "unresolved": unresolved-only
DEFAULT_POINT_FILTER = "deadzone"
# If True, always include one representative tie point when filter selects none.
# This keeps diagnostics useful for a chosen N even with no dead-zone matches.
DEFAULT_INCLUDE_REFERENCE_IF_EMPTY = True

RIGHT_ZOOM_PAD_FRAC = 0.20
# Explicit tie point numbers to include in diagnostics (in addition to default filter).
INCLUDE_TIE_POINT_NUMBERS = [] # [54781] # [-120148, 57391, 57392, 120147, 120148]
# Default lower p cutoff for diagnostics.
MIN_DIAGNOSTIC_P = 0.0


def _extremum_type_from_slopes(slope_left: float, slope_right: float, flat_tol: float) -> str:
    s_l = 0 if (not np.isfinite(slope_left) or abs(float(slope_left)) <= float(flat_tol)) else (1 if slope_left > 0.0 else -1)
    s_r = 0 if (not np.isfinite(slope_right) or abs(float(slope_right)) <= float(flat_tol)) else (1 if slope_right > 0.0 else -1)
    if s_l < 0 and s_r > 0:
        return "minimum"
    if s_l > 0 and s_r < 0:
        return "maximum"
    return "neither"


def _next_available_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    k = 1
    while True:
        candidate = parent / f"{stem}_{k:02d}{suffix}"
        if not candidate.exists():
            return candidate
        k += 1


def _compute_local_eps_arr(
    ties: np.ndarray,
    slope_a: float,
    slope_eps_cap: float,
    slope_eps_min: float,
) -> np.ndarray:
    """Match ``_tie_slope_records_for_n`` epsilon: gap scaling, then hard ``eps_min``."""
    if ties.size == 0:
        return np.array([], dtype=float)
    left_gap = np.empty_like(ties, dtype=float)
    right_gap = np.empty_like(ties, dtype=float)
    if ties.size == 1:
        left_gap[0] = float(ties[0])
        right_gap[0] = float(1.0 - ties[0])
    else:
        left_gap[0] = float(ties[0])
        left_gap[1:] = ties[1:] - ties[:-1]
        right_gap[-1] = float(1.0 - ties[-1])
        right_gap[:-1] = ties[1:] - ties[:-1]
    local_min_gap = np.minimum(left_gap, right_gap)
    eps_arr = np.minimum(float(slope_a) * local_min_gap, float(slope_eps_cap))
    eps_arr = np.where(np.isfinite(eps_arr), eps_arr, float(slope_eps_cap))
    eps_arr = np.maximum(eps_arr, float(slope_eps_min))
    return eps_arr


def _window_for_tie(tp: float, ties: np.ndarray, idx: int, eps: float) -> tuple[float, float]:
    idx = max(0, min(idx, len(ties) - 1))

    left_neighbor = ties[idx - 1] if idx > 0 else max(0.0, tp - 50.0 * eps)
    right_neighbor = ties[idx + 1] if idx < len(ties) - 1 else min(1.0, tp + 50.0 * eps)

    g_left = max(tp - float(left_neighbor), 0.0)
    g_right = max(float(right_neighbor) - tp, 0.0)
    positive_gaps = [g for g in (g_left, g_right) if g > 0.0]
    base = max(min(positive_gaps) if positive_gaps else 1e-6, 1e-6)
    span = min(max(10.0 * base, 20.0 * eps), 0.02)

    x0 = max(0.0, tp - span)
    x1 = min(1.0, tp + span)
    return float(x0), float(x1)


def _initial_eps_slope_data(
    n: int,
    tp: float,
    eps: float,
    evaluator=_expected_sorted_rank_many,
) -> dict[str, float | np.ndarray]:
    p_left = float(np.clip(tp - eps, 0.0, 1.0))
    p_right = float(np.clip(tp + eps, 0.0, 1.0))
    if p_left == tp:
        p_left = float(np.nextafter(tp, 0.0))
    if p_right == tp:
        p_right = float(np.nextafter(tp, 1.0))

    p_eval = np.array([p_left, tp, p_right], dtype=float)
    y_eval = evaluator(n, p_eval)
    left_span = tp - p_left
    right_span = p_right - tp
    slope_left = float((y_eval[1] - y_eval[0]) / left_span) if left_span > 0.0 else 0.0
    slope_right = float((y_eval[2] - y_eval[1]) / right_span) if right_span > 0.0 else 0.0
    return {
        "x_points": p_eval,
        "y_points": y_eval,
        "slope_left": slope_left,
        "slope_right": slope_right,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tie-slope diagnostic PDF with clear tie-point markers.")
    parser.add_argument("--n", type=int, default=DEFAULT_DIAGNOSTIC_N, help="n value to diagnose")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: data/tie_slope_ambiguity_n<N>.pdf)",
    )
    parser.add_argument("--samples", type=int, default=1401, help="Points per local curve")
    parser.add_argument("--slope-a", type=float, default=DEFAULT_TIE_SLOPE_A)
    parser.add_argument("--slope-eps-cap", type=float, default=1e-3)
    parser.add_argument("--slope-flat-tol", type=float, default=DEFAULT_TIE_SLOPE_FLAT_TOL)
    parser.add_argument(
        "--slope-eps-min",
        type=float,
        default=DEFAULT_TIE_SLOPE_EPS_MIN,
        help=(
            "Hard minimum epsilon after gap scaling "
            "(matches OBDsaveSourceData DEFAULT_TIE_SLOPE_EPS_MIN)."
        ),
    )
    parser.add_argument(
        "--point-filter",
        choices=("deadzone", "unresolved", "manual"),
        default=DEFAULT_POINT_FILTER,
        help=(
            "deadzone/unresolved: plot union of that dead-zone subset, manual tie numbers "
            "(see --include-tie-numbers), and eps_floor warnings. "
            "manual: plot only tie numbers from --include-tie-numbers and INCLUDE_TIE_POINT_NUMBERS "
            "(no dead-zone or eps-floor union)."
        ),
    )
    parser.add_argument(
        "--include-tie-numbers",
        type=str,
        default="",
        metavar="LIST",
        help=(
            "Comma-separated zero-centered tie numbers (same convention as printed tie_point_number). "
            "Merged with INCLUDE_TIE_POINT_NUMBERS in the module. Required for meaningful output when "
            "--point-filter manual unless the module list is non-empty."
        ),
    )
    parser.add_argument(
        "--skip-summary-page",
        action="store_true",
        help="Do not write the first PDF page (text summary); only plot pages (or the empty message).",
    )
    parser.add_argument(
        "--no-reference-if-empty",
        action="store_true",
        help=(
            "If the plot set would be empty, do not add the fallback tie nearest p=0.5. "
            "Ignored when --point-filter manual (reference fallback is never used)."
        ),
    )
    args = parser.parse_args()

    n = int(args.n)
    out_path = Path(args.output) if args.output else Path("data") / f"tie_slope_ambiguity_n{n}.pdf"

    recs = all_tie_points_float_with_pairs(n)
    ties = np.array([float(p) for p, _ in recs], dtype=float)
    slope_recs, hit_floor, n_deadzone_used, n_deadzone_ambiguous, stats, warning_rows = (
        _tie_slope_records_for_n(
            n,
            recs,
            slope_a=float(args.slope_a),
            slope_eps_cap=float(args.slope_eps_cap),
            slope_flat_tol=float(args.slope_flat_tol),
            slope_eps_min=float(args.slope_eps_min),
        )
    )

    center_idx = int(np.argmin(np.abs(ties - 0.5))) if ties.size > 0 else 0
    include_set: set[int] = {int(v) for v in INCLUDE_TIE_POINT_NUMBERS}
    raw_include = str(args.include_tie_numbers).strip()
    if raw_include:
        for part in raw_include.split(","):
            part = part.strip()
            if not part:
                continue
            include_set.add(int(part))

    point_filter = str(args.point_filter)
    selected_by_idx: dict[int, tuple[dict, str]] = {}

    if point_filter == "manual":
        for idx_r, r in enumerate(slope_recs):
            tie_number = int(idx_r - center_idx)
            if tie_number in include_set:
                selected_by_idx[int(idx_r)] = (r, "manual_only")
    else:
        all_deadzone_idx: list[int] = []
        unresolved_deadzone_idx: list[int] = []
        for idx_r, r in enumerate(slope_recs):
            p = float(r.get("p", 0.0))
            if bool(r.get("deadzone_used", False)) and p >= float(MIN_DIAGNOSTIC_P):
                all_deadzone_idx.append(int(idx_r))
                if bool(r.get("extremum_ambiguous", False)):
                    unresolved_deadzone_idx.append(int(idx_r))

        if point_filter == "unresolved":
            deadzone_idx = list(unresolved_deadzone_idx)
        else:
            deadzone_idx = list(all_deadzone_idx)

        selected_by_idx = {
            int(i): (slope_recs[int(i)], "deadzone_filter") for i in deadzone_idx
        }
        for idx_r, r in enumerate(slope_recs):
            tie_number = int(idx_r - center_idx)
            if tie_number in include_set:
                selected_by_idx[int(idx_r)] = (r, "manual_include")
        for wr in warning_rows:
            if int(wr.get("eps_floor_hit", 0)) != 1:
                continue
            try:
                idx_wr = int(wr["tie_idx"])
            except (KeyError, TypeError, ValueError):
                continue
            if idx_wr < 0 or idx_wr >= len(slope_recs):
                continue
            r_floor = slope_recs[idx_wr]
            p_floor = float(r_floor.get("p", 0.0))
            if p_floor < float(MIN_DIAGNOSTIC_P):
                continue
            if idx_wr not in selected_by_idx:
                selected_by_idx[idx_wr] = (r_floor, "eps_floor")

    include_reference_if_empty = (
        bool(DEFAULT_INCLUDE_REFERENCE_IF_EMPTY)
        and not bool(args.no_reference_if_empty)
        and point_filter != "manual"
    )
    if (not selected_by_idx) and include_reference_if_empty and ties.size > 0:
        # Deterministic fallback point: tie closest to p=0.5.
        idx_ref = int(np.argmin(np.abs(ties - 0.5)))
        selected_by_idx[idx_ref] = (slope_recs[idx_ref], "reference_fallback")

    deadzone_points: list[dict] = []
    deadzone_points = []
    for idx_sel in sorted(selected_by_idx.keys()):
        rec_sel, src = selected_by_idx[idx_sel]
        rec_copy = dict(rec_sel)
        rec_copy["_tie_idx"] = int(idx_sel)
        rec_copy["diagnostic_source"] = str(src)
        deadzone_points.append(rec_copy)
    minima_deadzone = [
        r for r in slope_recs if r.get("extremum_type") == "minimum" and bool(r.get("deadzone_used", False))
    ]
    minima_resolved = [r for r in minima_deadzone if not bool(r.get("extremum_ambiguous", False))]
    minima_unresolved = [r for r in minima_deadzone if bool(r.get("extremum_ambiguous", False))]
    minima_counts = {
        MINIMA_CATEGORY_DEADZONE: len(minima_deadzone),
        MINIMA_CATEGORY_RESOLVED: len(minima_resolved),
        MINIMA_CATEGORY_UNRESOLVED: len(minima_unresolved),
    }

    local_eps_arr = _compute_local_eps_arr(
        ties,
        float(args.slope_a),
        float(args.slope_eps_cap),
        float(args.slope_eps_min),
    )
    initial_info_by_idx: dict[int, dict[str, object]] = {}
    pre_minima_deadzone_count = 0
    for r in deadzone_points:
        idx_tp = int(r["_tie_idx"])
        idx_tp = max(0, min(idx_tp, len(ties) - 1))
        tp = float(ties[idx_tp])
        eps_local = float(local_eps_arr[idx_tp]) if local_eps_arr.size > 0 else float(args.slope_eps_cap)
        info_float = _initial_eps_slope_data(n, tp, eps_local, evaluator=_expected_sorted_rank_many)
        eval_many_hp = lambda n0, x0: _expected_sorted_rank_many_high_precision(
            n0, x0, dps=DEFAULT_HP_DPS
        )
        info_hp = _initial_eps_slope_data(n, tp, eps_local, evaluator=eval_many_hp)
        pre_type = _extremum_type_from_slopes(
            float(info_float["slope_left"]),
            float(info_float["slope_right"]),
            float(args.slope_flat_tol),
        )
        initial_info_by_idx[idx_tp] = {
            "float": info_float,
            "hp": info_hp,
            "pre_extremum_type": pre_type,
        }
        if pre_type == "minimum":
            pre_minima_deadzone_count += 1

    if local_eps_arr.size > 0:
        eps_summary = (
            f"{float(np.min(local_eps_arr)):.6e} / "
            f"{float(np.median(local_eps_arr)):.6e} / "
            f"{float(np.max(local_eps_arr)):.6e}"
        )
    else:
        eps_summary = "n/a"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = _next_available_output_path(out_path)
    with PdfPages(out_path) as pdf:
        if not bool(args.skip_summary_page):
            fig = plt.figure(figsize=(11, 8.5), facecolor="white")
            n_plot_deadzone = sum(
                1 for r in deadzone_points if str(r.get("diagnostic_source", "")) == "deadzone_filter"
            )
            n_plot_eps_floor = sum(
                1 for r in deadzone_points if str(r.get("diagnostic_source", "")) == "eps_floor"
            )
            n_plot_manual = sum(
                1
                for r in deadzone_points
                if str(r.get("diagnostic_source", "")) in ("manual_include", "manual_only")
            )
            txt = (
                "Tie Slope Ambiguity Diagnostics\n\n"
                f"n = {n}\n"
                f"point filter = {point_filter}\n"
                f"tie point count = {len(ties)}\n"
                f"dead-zone used count = {n_deadzone_used}\n"
                f"plots: deadzone_filter = {n_plot_deadzone}, eps_floor adds = {n_plot_eps_floor}, "
                f"manual = {n_plot_manual}, total pages = {len(deadzone_points)}\n"
                f"unresolved ambiguity count = {n_deadzone_ambiguous}\n"
                f"hit eps floor = {hit_floor}\n"
                f"local eps min/median/max = {eps_summary}\n\n"
                "ambiguous minima summary:\n"
                f"  pre_resolution_deadzone_minima = {pre_minima_deadzone_count}\n"
                f"  {MINIMA_CATEGORY_DEADZONE} = {minima_counts[MINIMA_CATEGORY_DEADZONE]}\n"
                f"  {MINIMA_CATEGORY_RESOLVED} = {minima_counts[MINIMA_CATEGORY_RESOLVED]}\n"
                f"  {MINIMA_CATEGORY_UNRESOLVED} = {minima_counts[MINIMA_CATEGORY_UNRESOLVED]}\n\n"
                "hp 3-point escalation stats:\n"
                f"  attempted={stats.get('hp_3point_attempted', 0)}\n"
                f"  resolved={stats.get('hp_3point_resolved', 0)}\n"
                f"  failed={stats.get('hp_3point_failed', 0)}\n"
                f"  hp_3point_sec={stats.get('hp_3point_sec', 0.0):.6f}\n"
                f"  hp_flat_tol_scale={stats.get('hp_flat_tol_scale', DEFAULT_TOL_SCALE):.1e}\n"
                f"  ulp_step_adjusted={stats.get('ulp_step_adjusted', 0)}\n"
                f"  high_precision_points={stats.get('high_precision_points', 0)}\n"
            )
            fig.text(0.07, 0.92, txt, va="top", family="monospace", fontsize=12, color="black")
            pdf.savefig(fig)
            plt.close(fig)

        if not deadzone_points:
            fig = plt.figure(figsize=(11, 8.5), facecolor="white")
            fig.text(
                0.1,
                0.5,
                f"No points matched filter={point_filter} for this n.",
                fontsize=16,
                color="black",
            )
            pdf.savefig(fig)
            plt.close(fig)
        else:
            for k, rec in enumerate(deadzone_points, start=1):
                idx_tp = int(rec["_tie_idx"])
                idx_tp = max(0, min(idx_tp, len(ties) - 1))
                tp = float(ties[idx_tp])
                eps_local = float(local_eps_arr[idx_tp]) if local_eps_arr.size > 0 else float(args.slope_eps_cap)
                left_neighbor = float(ties[idx_tp - 1]) if idx_tp > 0 else 0.0
                right_neighbor = float(ties[idx_tp + 1]) if idx_tp < (len(ties) - 1) else 1.0
                left_gap = max(tp - left_neighbor, 0.0)
                right_gap = max(right_neighbor - tp, 0.0)
                eval_many = (
                    (lambda n0, x0: _expected_sorted_rank_many_high_precision(n0, x0, dps=DEFAULT_HP_DPS))
                    if bool(rec.get("high_precision_used", False))
                    else _expected_sorted_rank_many
                )
                x0, x1 = _window_for_tie(tp, ties, idx_tp, eps_local)
                x = np.linspace(x0, x1, int(args.samples), dtype=float)
                y = eval_many(n, x)
                tie_idx = idx_tp
                tie_number = int(tie_idx - center_idx)

                init_bundle = initial_info_by_idx.get(idx_tp)
                if init_bundle is None:
                    init_float = _initial_eps_slope_data(n, tp, eps_local, evaluator=_expected_sorted_rank_many)
                    init_hp = _initial_eps_slope_data(
                        n,
                        tp,
                        eps_local,
                        evaluator=lambda n0, x0: _expected_sorted_rank_many_high_precision(
                            n0, x0, dps=DEFAULT_HP_DPS
                        ),
                    )
                    pre_type = _extremum_type_from_slopes(
                        float(init_float["slope_left"]),
                        float(init_float["slope_right"]),
                        float(args.slope_flat_tol),
                    )
                    init_bundle = {"float": init_float, "hp": init_hp, "pre_extremum_type": pre_type}
                    initial_info_by_idx[idx_tp] = init_bundle
                init_float = init_bundle["float"]  # type: ignore[index]
                init_hp = init_bundle["hp"]  # type: ignore[index]

                fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15.5, 8.5), facecolor="white")
                ax_left.set_facecolor("white")
                ax_right.set_facecolor("white")
                ax_left.plot(x, y, color="black", lw=1.7, label="E_sorted(p)", zorder=2)
                ax_left.axvline(tp, color="red", lw=1.2, ls="--", label="ambiguous tie p", zorder=4)

                nearby = ties[(ties >= x0) & (ties <= x1)]
                if nearby.size > 0:
                    y_near = eval_many(n, nearby)
                    y_lo = float(np.min(y))
                    y_hi = float(np.max(y))
                    ax_left.vlines(
                        nearby,
                        ymin=y_lo,
                        ymax=y_hi,
                        colors="royalblue",
                        linewidth=0.9,
                        alpha=0.30,
                        zorder=1,
                    )
                    ax_left.scatter(
                        nearby,
                        y_near,
                        s=34,
                        c="royalblue",
                        edgecolors="white",
                        linewidths=0.6,
                        alpha=0.98,
                        zorder=5,
                        label="tie points in window",
                    )

                y_tp = float(eval_many(n, np.array([tp], dtype=float))[0])
                ax_left.scatter([tp], [y_tp], s=62, c="red", edgecolors="black", linewidths=0.8, zorder=6)
                ax_left.set_xlim(x0, x1)
                ax_left.set_xlabel("p", color="black")
                ax_left.set_ylabel("E_sorted(p)", color="black")
                ax_left.set_title("Left panel: existing view", color="black")
                ax_left.tick_params(colors="black")
                ax_left.grid(True, alpha=0.22, color="0.2")

                # Right panel: zoom into 3-point slope samples only.
                zoom_points = [float(tp), *[float(v) for v in init_float["x_points"]]]
                zoom_points.extend([float(v) for v in init_hp["x_points"]])
                xz0 = max(0.0, min(zoom_points))
                xz1 = min(1.0, max(zoom_points))
                z_span = max(xz1 - xz0, 1e-12)
                xz0 = max(0.0, xz0 - RIGHT_ZOOM_PAD_FRAC * z_span)
                xz1 = min(1.0, xz1 + RIGHT_ZOOM_PAD_FRAC * z_span)
                x_zoom = np.linspace(xz0, xz1, 1401, dtype=float)
                y_zoom_float = _expected_sorted_rank_many(n, x_zoom)
                y_zoom_hp = _expected_sorted_rank_many_high_precision(
                    n, x_zoom, dps=DEFAULT_HP_DPS
                )
                ax_right.plot(
                    x_zoom,
                    y_zoom_float,
                    color="0.75",
                    lw=1.5,
                    label="E_sorted float64",
                    zorder=2,
                )
                ax_right.plot(
                    x_zoom,
                    y_zoom_hp,
                    color="black",
                    lw=1.7,
                    label="E_sorted HP",
                    zorder=3,
                )
                ax_right.axvline(tp, color="red", lw=1.2, ls="--", label="ambiguous tie p", zorder=4)
                y_tp_hp_marker = float(
                    _expected_sorted_rank_many_high_precision(
                        n, np.array([tp], dtype=float), dps=DEFAULT_HP_DPS
                    )[0]
                )
                ax_right.scatter(
                    [tp],
                    [y_tp_hp_marker],
                    s=62,
                    c="red",
                    edgecolors="black",
                    linewidths=0.8,
                    zorder=6,
                )

                nearby_zoom = ties[(ties >= xz0) & (ties <= xz1)]
                if nearby_zoom.size > 0:
                    y_lo_z = float(min(np.min(y_zoom_float), np.min(y_zoom_hp)))
                    y_hi_z = float(max(np.max(y_zoom_float), np.max(y_zoom_hp)))
                    ax_right.vlines(
                        nearby_zoom,
                        ymin=y_lo_z,
                        ymax=y_hi_z,
                        colors="royalblue",
                        linewidth=0.9,
                        alpha=0.30,
                        zorder=1,
                    )
                    y_near_zoom = _expected_sorted_rank_many(n, nearby_zoom)
                    ax_right.scatter(
                        nearby_zoom,
                        y_near_zoom,
                        s=34,
                        c="royalblue",
                        edgecolors="white",
                        linewidths=0.6,
                        alpha=0.98,
                        zorder=5,
                        label="tie points in window",
                    )

                ax_right.scatter(
                    init_float["x_points"],
                    init_float["y_points"],
                    s=28,
                    c="magenta",
                    marker="x",
                    linewidths=1.2,
                    zorder=7,
                    label="float64 3-point samples",
                )
                ax_right.scatter(
                    init_hp["x_points"],
                    init_hp["y_points"],
                    s=30,
                    c="deepskyblue",
                    marker="+",
                    linewidths=1.4,
                    zorder=8,
                    label="HP 3-point samples",
                )
                y_candidates = [float(v) for v in y_zoom_float]
                y_candidates.extend(float(v) for v in y_zoom_hp)
                y_candidates.extend(float(v) for v in init_float["y_points"])
                y_candidates.extend(float(v) for v in init_hp["y_points"])
                # Force right-panel vertical zoom to local diagnostics,
                # so small local slopes are visible rather than visually flat.
                y_min = min(y_candidates)
                y_max = max(y_candidates)
                y_span = y_max - y_min
                if y_span <= 0.0:
                    slope_scale = max(
                        abs(float(init_float["slope_left"])),
                        abs(float(init_float["slope_right"])),
                        abs(float(init_hp["slope_left"])),
                        abs(float(init_hp["slope_right"])),
                    )
                    y_span = max(slope_scale * (xz1 - xz0), 1e-12)
                y_pad = max(0.08 * y_span, 1e-12)
                ax_right.set_xlim(xz0, xz1)
                ax_right.set_ylim(y_min - y_pad, y_max + y_pad)
                ax_right.set_xlabel("p", color="black")
                ax_right.set_ylabel("E_sorted(p)", color="black")
                ax_right.set_title("Right panel: zoom (float64 gray vs HP black)", color="black")
                ax_right.tick_params(colors="black")
                ax_right.grid(True, alpha=0.22, color="0.2")

                hp_slope_left = float(init_hp["slope_left"])
                hp_slope_right = float(init_hp["slope_right"])
                pre_type = str(init_bundle.get("pre_extremum_type", "unknown"))
                post_type = str(rec.get("extremum_type", "unknown"))
                stage_deadzone = int(bool(rec.get("deadzone_used", False)))
                stage_hp = int(bool(rec.get("high_precision_used", False)))
                stage_resolved = int(not bool(rec.get("extremum_ambiguous", False)))
                stage_path = str(rec.get("resolution_path", "unknown"))
                stage_fail_reason = str(rec.get("resolution_failure_reason", ""))
                stage_fail_label = stage_fail_reason if stage_fail_reason else "none"
                diag_source = str(rec.get("diagnostic_source", "unknown"))
                fig.suptitle(
                    f"n={n} source={diag_source} point {k}/{len(deadzone_points)} at p={tp:.12f} | "
                    f"tie_point_number={tie_number:+d} (0 at p≈0.5)\n"
                    f"stage: deadzone={stage_deadzone}, hp3_used={stage_hp}, resolved={stage_resolved} | "
                    f"path={stage_path}, fail_reason={stage_fail_label}\n"
                    f"neighbor_gap_left={left_gap:.6e}, neighbor_gap_right={right_gap:.6e}, eps_used={eps_local:.6e}\n"
                    f"pre_type={pre_type}, post_type={post_type}, unresolved={bool(rec.get('extremum_ambiguous', False))}, "
                    f"quantization_suspect={bool(rec.get('quantization_suspect', False))}\n"
                    f"float64 3pt sL={float(init_float['slope_left']):.6e}, "
                    f"sR={float(init_float['slope_right']):.6e} | "
                    f"hp 3pt sL={hp_slope_left:.6e}, sR={hp_slope_right:.6e} | "
                    f"recorded sL={rec['slope_left']:.6e}, sR={rec['slope_right']:.6e}",
                    color="black",
                )
                ax_left.legend(loc="best")
                ax_right.legend(loc="best")
                fig.tight_layout(rect=[0, 0, 1, 0.94])
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Wrote {out_path}")
    print(
        f"n={n} point_filter={point_filter} deadzone_plotted={len(deadzone_points)} "
        f"unresolved_total={n_deadzone_ambiguous} {MINIMA_CATEGORY_RESOLVED}={minima_counts[MINIMA_CATEGORY_RESOLVED]} "
        f"{MINIMA_CATEGORY_UNRESOLVED}={minima_counts[MINIMA_CATEGORY_UNRESOLVED]} "
        f"{MINIMA_CATEGORY_DEADZONE}={minima_counts[MINIMA_CATEGORY_DEADZONE]} minima_total="
        f"{sum(1 for r in slope_recs if r.get('extremum_type') == 'minimum')}"
    )


if __name__ == "__main__":
    main()

