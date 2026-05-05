#!/usr/bin/env python3
"""Overlay HP features vs N for one cusp row per ``N`` from the cusp sidecar pickle.

**Which row:** use only records with ``extremum_type == "minimum"``. Among those, take the
row with **largest** ``p_float`` / ``p``; if several tie on ``p``, pick the larger
``tie_index``.

If **any** cusp row tied at the global maximum ``p`` for that ``N`` is **not** a ``minimum``
(e.g. ``extremum_ambiguous``), that ``N`` yields **no** selected row (all series NaN): the
highest-``p`` feature on the shard is not a trusted minimum.

**Y-axis assignment (precedence):** slopes (``PLOT_LEFT`` / ``PLOT_RIGHT``) then ``PLOT_P`` then
``PLOT_EV``. The first enabled group uses the main (left) y-axis; the second uses the first
``twinx`` on the right; the third uses a second ``twinx`` (spine offset when needed). Each axis
is autoscaled to the finite values actually plotted for that group in the selected ``N`` range.

When ``PLOT_EV`` is enabled, the EV curve is ``ev_mid_hp / N`` (per-point).

Every ``N`` that passes ``--n-min`` / ``--n-max`` / parity gets a point; missing or non-finite
values are stored as NaN so lines show gaps instead of dropping ``N``.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from OBDsaveSourceData import DEFAULT_CUSP_OUTPUT, load_cusp_data

# --- plot toggles (edit here) ---
PLOT_LEFT = False
PLOT_RIGHT = False
PLOT_P = False
PLOT_EV = True

# Figure size in inches (width, height).
FIGSIZE: tuple[float, float] = (16.0, 8.0)

DEFAULT_N_PARITY: str = "all"

# Inclusive soft N bounds (defaults for ``--n-min`` / ``--n-max``). Set to ``None`` for no bound.
# If the cusp file has no ``N`` in range, the script still runs; the figure may be empty.
N_MIN: int | None = 100
N_MAX: int | None = 1000


def _tie_p(rec: dict) -> float:
    """Same ``p`` coordinate as ``plot_max_local_min_p_vs_n._tie_p``."""
    if "p_float" in rec:
        return float(rec["p_float"])
    return float(rec["p"])


def _minima_records(records: list[dict]) -> list[dict]:
    return [r for r in records if str(r.get("extremum_type", "")) == "minimum"]


def _select_highest_p_minimum_record(records: list[dict]) -> dict | None:
    """Among ``minimum`` rows only, argmax ``p``; tie-break larger ``tie_index``.

    Returns ``None`` if there are no minima, or if any row at the shard-wide maximum ``p``
    is not a ``minimum`` (e.g. ambiguous has the top ``p``).
    """
    if not records:
        return None
    max_p = max(_tie_p(r) for r in records)
    if not math.isfinite(max_p):
        return None
    for r in records:
        if not math.isclose(_tie_p(r), max_p, rel_tol=0.0, abs_tol=1e-12):
            continue
        if str(r.get("extremum_type", "")) != "minimum":
            return None

    pool = _minima_records(records)
    if not pool:
        return None

    def _sort_key(r: dict) -> tuple[float, int]:
        try:
            ti = int(r["tie_index"])
        except (KeyError, TypeError, ValueError):
            ti = 0
        return (_tie_p(r), ti)

    return max(pool, key=_sort_key)


def _parse_hp_number(s: object) -> float | None:
    if s is None:
        return None
    try:
        v = float(str(s).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _last_cusp_p(rec: dict) -> float | None:
    if "p_hp_main" in rec:
        v = _parse_hp_number(rec.get("p_hp_main"))
        if v is not None:
            return v
    if "p_float" in rec:
        return _parse_hp_number(rec["p_float"])
    return _parse_hp_number(rec.get("p"))


def _finite_or_nan(x: float | None) -> float:
    """Matplotlib leaves gaps for NaN y-values."""
    if x is None:
        return float("nan")
    if not np.isfinite(x):
        return float("nan")
    return float(x)


def _autoscale_ylim_from_series(ax, *series: list[float]) -> None:
    """Set y-limits from finite values across one or more aligned series (NaNs ignored)."""
    chunks: list[np.ndarray] = []
    for s in series:
        a = np.asarray(s, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            chunks.append(a)
    if not chunks:
        return
    allv = np.concatenate(chunks)
    lo, hi = float(allv.min()), float(allv.max())
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
        pad = max(abs(lo) * 0.05, 0.02)
    else:
        pad = (hi - lo) * 0.05
    ax.set_ylim(lo - pad, hi + pad)


def main() -> None:
    if not any((PLOT_LEFT, PLOT_RIGHT, PLOT_P, PLOT_EV)):
        raise SystemExit("Enable at least one of PLOT_LEFT, PLOT_RIGHT, PLOT_P, PLOT_EV.")

    parser = argparse.ArgumentParser(
        description="Overlay last-cusp HP slopes / p / EV vs N (see module toggles)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_CUSP_OUTPUT,
        metavar="PATH",
        help=f"Cusp sidecar pickle (default: {DEFAULT_CUSP_OUTPUT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "plot_last_cusp_features.pdf"),
        metavar="PATH",
        help="Output PDF path (default: data/plot_last_cusp_features.pdf).",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=N_MIN,
        metavar="N",
        help=f"Include only n >= N (default: {N_MIN}).",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=N_MAX,
        metavar="N",
        help=f"Include only n <= N (default: {N_MAX}).",
    )
    parser.add_argument(
        "--parity",
        choices=("even", "odd", "all"),
        default=DEFAULT_N_PARITY,
        help=(
            "Restrict plotted N to even-only, odd-only, or all "
            f"(default: {DEFAULT_N_PARITY})."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Highest-p minimum (cusp): features vs N",
        help="Figure title.",
    )
    args = parser.parse_args()

    payload = load_cusp_data(path=args.input, n_list=None, require_all=False)
    n_entries = payload.get("n_entries") or {}
    if not n_entries:
        raise ValueError(f"No n_entries in {args.input!r}")

    n_min = args.n_min
    n_max = args.n_max
    parity = str(args.parity)

    ns: list[int] = []
    left_y: list[float] = []
    right_y: list[float] = []
    p_y: list[float] = []
    ev_y: list[float] = []

    for n_key in sorted(n_entries.keys(), key=lambda k: int(k)):
        ni = int(n_key)
        if n_min is not None and ni < n_min:
            continue
        if n_max is not None and ni > n_max:
            continue
        if parity == "even" and ni % 2 != 0:
            continue
        if parity == "odd" and ni % 2 != 1:
            continue
        block = n_entries[n_key]
        records = block.get("records")
        if not isinstance(records, list):
            records = []
        rec = _select_highest_p_minimum_record(records)

        sl = _parse_hp_number(rec.get("slope_left_hp")) if rec else None
        sr = _parse_hp_number(rec.get("slope_right_hp")) if rec else None
        pv = _last_cusp_p(rec) if rec else None
        ev = _parse_hp_number(rec.get("ev_mid_hp")) if rec else None

        ns.append(ni)
        left_y.append(_finite_or_nan(sl) if PLOT_LEFT else float("nan"))
        right_y.append(_finite_or_nan(sr) if PLOT_RIGHT else float("nan"))
        p_y.append(_finite_or_nan(pv) if PLOT_P else float("nan"))
        if PLOT_EV:
            ev_fin = _finite_or_nan(ev)
            if np.isfinite(ev_fin) and ni != 0:
                ev_y.append(ev_fin / float(ni))
            else:
                ev_y.append(float("nan"))
        else:
            ev_y.append(float("nan"))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not ns:
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor="white")
        ax.text(
            0.5,
            0.5,
            "No N in the requested range with plottable cusp rows.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_axis_off()
        fig.savefig(out_path, format="pdf", dpi=160)
        plt.close(fig)
        print(f"Wrote {out_path} (0 points in range).")
        return

    fig, ax_main = plt.subplots(figsize=FIGSIZE, facecolor="white")
    legend_handles: list = []
    legend_labels: list[str] = []

    plot_slopes = PLOT_LEFT or PLOT_RIGHT
    layers: list[str] = []
    if plot_slopes:
        layers.append("slopes")
    if PLOT_P:
        layers.append("p")
    if PLOT_EV:
        layers.append("ev")

    ax_by_layer: dict[str, object] = {}
    ax_by_layer[layers[0]] = ax_main
    for li in range(1, len(layers)):
        tax = ax_main.twinx()
        ax_by_layer[layers[li]] = tax
    if len(layers) == 3:
        ax_out = ax_by_layer[layers[2]]
        ax_out.spines["right"].set_position(("axes", 1.08))

    ax_sl = ax_by_layer["slopes"] if "slopes" in ax_by_layer else None
    ax_p = ax_by_layer["p"] if "p" in ax_by_layer else None
    ax_ev = ax_by_layer["ev"] if "ev" in ax_by_layer else None

    if PLOT_LEFT and ax_sl is not None:
        (h,) = ax_sl.plot(ns, left_y, color="tab:blue", linewidth=1.5, marker=".", markersize=4)
        legend_handles.append(h)
        legend_labels.append("slope_left_hp")
    if PLOT_RIGHT and ax_sl is not None:
        (h,) = ax_sl.plot(ns, right_y, color="tab:orange", linewidth=1.5, marker=".", markersize=4)
        legend_handles.append(h)
        legend_labels.append("slope_right_hp")
    if plot_slopes and ax_sl is not None:
        series_for_slope: list[list[float]] = []
        if PLOT_LEFT:
            series_for_slope.append(left_y)
        if PLOT_RIGHT:
            series_for_slope.append(right_y)
        _autoscale_ylim_from_series(ax_sl, *series_for_slope)
        if layers[0] == "slopes":
            ax_sl.set_ylabel("HP 3-point slope")

    if PLOT_P and ax_p is not None:
        (h,) = ax_p.plot(ns, p_y, color="tab:red", linewidth=1.5, marker=".", markersize=4)
        legend_handles.append(h)
        legend_labels.append("tie p (HP), highest-p minimum")
        _autoscale_ylim_from_series(ax_p, p_y)
        ax_p.set_ylabel("tie p (HP), highest-p minimum", color="tab:red")
        ax_p.tick_params(axis="y", labelcolor="tab:red")

    if PLOT_EV and ax_ev is not None:
        (h,) = ax_ev.plot(ns, ev_y, color="tab:green", linewidth=1.5, marker=".", markersize=4)
        legend_handles.append(h)
        legend_labels.append("ev_mid_hp / N")
        _autoscale_ylim_from_series(ax_ev, ev_y)
        ax_ev.set_ylabel("E_sorted at tie / N (HP)", color="tab:green")
        ax_ev.tick_params(axis="y", labelcolor="tab:green")

    ax_main.set_xlim(float(min(ns)), float(max(ns)))
    ax_main.set_xlabel("N")
    ax_main.set_title(args.title)
    ax_main.grid(True, alpha=0.3)

    if legend_handles:
        ax_main.legend(legend_handles, legend_labels, loc="best")

    fig.tight_layout()
    if len(layers) >= 2:
        fig.subplots_adjust(right=0.84 if len(layers) == 3 else 0.88)

    fig.savefig(out_path, format="pdf", dpi=160)
    plt.close(fig)

    print(f"Wrote {out_path} ({len(ns)} points)")


if __name__ == "__main__":
    main()
