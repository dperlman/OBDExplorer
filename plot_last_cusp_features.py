#!/usr/bin/env python3
"""Overlay features vs N for one cusp row per ``N`` from the cusp table.

**Which row:** every row of the cusp table is a certified cusp; take the one with the **largest**
``p_float`` (the last cusp); if several tie on ``p``, pick the larger ``tie_index``.

**Y-axis assignment (precedence):** slopes (``PLOT_LEFT`` / ``PLOT_RIGHT``) then ``PLOT_P`` then
``PLOT_EV``. The first enabled group uses the main (left) y-axis; the second uses the first
``twinx`` on the right; the third uses a second ``twinx`` (spine offset when needed). Each axis
is autoscaled to the finite values actually plotted for that group in the selected ``N`` range.

When ``PLOT_EV`` is enabled, the EV curve is ``E / N`` (per-point).

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

from OBDsaveSourceData import DEFAULT_CUSP_OUTPUT, load_cusp_table

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


def _last_cusp_rows(cusps: dict[str, np.ndarray]) -> dict[int, int]:
    """Row of the last cusp (largest ``p``; ties broken by larger ``tie_index``) for each n."""
    order = np.lexsort((cusps["tie_index"], cusps["p"], cusps["n"]))
    n_sorted = cusps["n"][order]
    last = np.flatnonzero(np.r_[n_sorted[1:] != n_sorted[:-1], True])
    return {int(n_sorted[k]): int(order[k]) for k in last}


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
        description="Overlay last-cusp slopes / p / EV vs N (see module toggles)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_CUSP_OUTPUT,
        metavar="PATH",
        help=f"Cusp table (Parquet; default: {DEFAULT_CUSP_OUTPUT}).",
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
        default="Last cusp (highest p): features vs N",
        help="Figure title.",
    )
    args = parser.parse_args()

    cusps = load_cusp_table(path=args.input)
    if cusps["n"].size == 0:
        raise ValueError(f"No cusps in {args.input!r}")
    last_row = _last_cusp_rows(cusps)

    n_min = args.n_min
    n_max = args.n_max
    parity = str(args.parity)

    ns: list[int] = []
    left_y: list[float] = []
    right_y: list[float] = []
    p_y: list[float] = []
    ev_y: list[float] = []

    for ni in sorted(last_row):
        if n_min is not None and ni < n_min:
            continue
        if n_max is not None and ni > n_max:
            continue
        if parity == "even" and ni % 2 != 0:
            continue
        if parity == "odd" and ni % 2 != 1:
            continue
        k = last_row[ni]
        sl = float(cusps["slope_left"][k])
        sr = float(cusps["slope_right"][k])
        pv = float(cusps["p"][k])
        ev = float(cusps["E"][k])

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
        legend_labels.append("slope_left")
    if PLOT_RIGHT and ax_sl is not None:
        (h,) = ax_sl.plot(ns, right_y, color="tab:orange", linewidth=1.5, marker=".", markersize=4)
        legend_handles.append(h)
        legend_labels.append("slope_right")
    if plot_slopes and ax_sl is not None:
        series_for_slope: list[list[float]] = []
        if PLOT_LEFT:
            series_for_slope.append(left_y)
        if PLOT_RIGHT:
            series_for_slope.append(right_y)
        _autoscale_ylim_from_series(ax_sl, *series_for_slope)
        if layers[0] == "slopes":
            ax_sl.set_ylabel("slope at the last cusp")

    if PLOT_P and ax_p is not None:
        (h,) = ax_p.plot(ns, p_y, color="tab:red", linewidth=1.5, marker=".", markersize=4)
        legend_handles.append(h)
        legend_labels.append("tie p, last cusp")
        _autoscale_ylim_from_series(ax_p, p_y)
        ax_p.set_ylabel("tie p, last cusp", color="tab:red")
        ax_p.tick_params(axis="y", labelcolor="tab:red")

    if PLOT_EV and ax_ev is not None:
        (h,) = ax_ev.plot(ns, ev_y, color="tab:green", linewidth=1.5, marker=".", markersize=4)
        legend_handles.append(h)
        legend_labels.append("E / N")
        _autoscale_ylim_from_series(ax_ev, ev_y)
        ax_ev.set_ylabel("E at the last cusp / N", color="tab:green")
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
