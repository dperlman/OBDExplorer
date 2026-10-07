"""Plot tie-point statistics from the explorer pickle (no tie computation here).

Generate data with ``OBDsaveSourceData.py`` (e.g. ``--save-tie-points`` or ``--all``).
Default input: ``data/tie_points.pkl`` with keys:

  - ``float_with_pairs_by_n``: ``n -> list[(p, [(i,j), ...]), ...]`` (pair plots)
  - ``float_by_n``: ``n -> ndarray of p`` (float-only plots; used if pairs missing for that n)

Optional legacy pickles may use key ``by_n`` instead of ``float_with_pairs_by_n`` for the same
record shape.
"""
import os
import pickle

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

# Same default as OBDsaveSourceData.DEFAULT_TIE_OUTPUT / OBDgraphExplorer1
DATA_DIR = "data"
DEFAULT_TIE_POINTS_PATH = os.path.join(DATA_DIR, "tie_points.pkl")


def load_float_with_pairs_by_n(path: str) -> dict:
    """``n -> [(p, [(i,j), ...]), ...]`` from ``float_with_pairs_by_n`` or legacy ``by_n``."""
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        return {}
    return dict(data.get("float_with_pairs_by_n") or data.get("by_n") or {})


def _line_rgba_for_t(t: float, *, alpha: float = 0.35) -> str:
    """Map ``t`` in [0, 1] to Turbo, as ``rgba`` with given alpha."""
    t = min(max(float(t), 0.0), 1.0)
    c = sample_colorscale("Turbo", [t], colortype="rgb")[0]
    if c.startswith("rgb("):
        inner = c[4:-1].strip()
        return f"rgba({inner},{alpha})"
    return c


def load_tie_points_p_by_n(path: str) -> dict[int, np.ndarray]:
    """Merged tie p values per n (explorer semantics).

    Prefers ``float_with_pairs_by_n`` (``[p for p, _ in recs]``); falls back to ``float_by_n``.
    """
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        return {}
    fwp = data.get("float_with_pairs_by_n") or data.get("by_n") or {}
    fb = data.get("float_by_n", {})
    keys = sorted(set(fwp.keys()) | set(fb.keys()))
    out: dict[int, np.ndarray] = {}
    for n in keys:
        if n in fwp:
            recs = fwp[n]
            out[n] = np.array([float(p) for p, _ in recs], dtype=float)
        else:
            out[n] = np.atleast_1d(np.asarray(fb[n], dtype=float))
    return out


def plot_tie_vertical_segments_p_n(path: str = DEFAULT_TIE_POINTS_PATH) -> None:
    """Vertical segments (p, n−1)–(p, n) for p in [0.5, 1], n = 2…100.

    Color is by swap index ``i`` in ``(i, j)``. For each ``n``, ``i`` is linearly rescaled to
    [0, 1] from ``i_min`` to ``i_max`` among segments in the p window so small ``n`` still
    uses the full Turbo range.
    """
    by_n = load_float_with_pairs_by_n(path)
    if not by_n:
        print(f"No pair data in {path} (need float_with_pairs_by_n or by_n)")
        return
    # rgba string -> batched polyline (NaN breaks) to limit trace count
    rgba_buckets: dict[str, tuple[list[float], list[float]]] = {}
    for n in range(2, 101):
        if n not in by_n:
            continue
        recs = by_n[n]
        segs: list[tuple[float, int, int]] = []
        for p, pairs in recs:
            pf = float(p)
            if not (0.5 <= pf <= 1) or not np.isfinite(pf):
                continue
            for ij in pairs:
                if len(ij) != 2:
                    continue
                i, j = int(ij[0]), int(ij[1])
                segs.append((pf, i, j))
        if not segs:
            continue
        i_lo = min(s[1] for s in segs)
        i_hi = max(s[1] for s in segs)
        for pf, i, _j in segs:
            t = (i - i_lo) / (i_hi - i_lo) if i_hi > i_lo else 0.0
            rgba = _line_rgba_for_t(t)
            if rgba not in rgba_buckets:
                rgba_buckets[rgba] = ([], [])
            xs, ys = rgba_buckets[rgba]
            xs.extend([pf, pf, float("nan")])
            ys.extend([float(n - 1), float(n), float("nan")])
    if not rgba_buckets:
        print("No segments for n = 2 … 100 and p in [0.5, 1]")
        return
    traces: list[go.Scatter] = []
    for rgba in sorted(rgba_buckets):
        xs, ys = rgba_buckets[rgba]
        if not xs:
            continue
        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(color=rgba, width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig = go.Figure(
        data=traces,
        layout=dict(
            title="Tie points: color = i (swap index), i scaled per n to full Turbo; p in [0.5, 1]",
            xaxis_title="p",
            yaxis_title="n",
            xaxis=dict(range=[0.5, 1], constrain="domain"),
            yaxis=dict(range=[1, 100], constrain="domain"),
        ),
    )
    fig.show()


def plot_count_vs_n(path: str = DEFAULT_TIE_POINTS_PATH) -> None:
    """Plot number of tie points vs n (merged p count per n)."""
    float_by_n = load_tie_points_p_by_n(path)
    if not float_by_n:
        print(f"No tie points found in {path}")
        return
    n_vals = sorted(float_by_n)
    counts = [np.sum((0 < np.atleast_1d(float_by_n[n])) & (np.atleast_1d(float_by_n[n]) < 1)) for n in n_vals]
    fig = go.Figure(
        data=[go.Scatter(x=n_vals, y=counts, mode="lines+markers")],
        layout=dict(
            xaxis_title="n",
            yaxis_title="Number of tie points",
            title="Number of tie points vs n",
        ),
    )
    fig.show()


def plot_scatter_n2_to_100(path: str = DEFAULT_TIE_POINTS_PATH) -> None:
    """For n=2 to 100, scatter gaps (differences between subsequent p) vs n with small points."""
    float_by_n = load_tie_points_p_by_n(path)
    if not float_by_n:
        print(f"No tie points found in {path}")
        return
    x_list = []
    y_list = []
    for n in range(2, 101):
        if n not in float_by_n:
            continue
        arr = np.atleast_1d(float_by_n[n])
        p_vals = np.sort(arr[(0 < arr) & (arr < 1)])
        if p_vals.size < 2:
            continue
        diffs = np.diff(p_vals)
        for gap in diffs:
            x_list.append(n)
            y_list.append(float(gap))
    if not x_list:
        print("No gaps in n=2..100")
        return
    fig = go.Figure(
        data=[go.Scatter(x=x_list, y=y_list, mode="markers", marker=dict(size=2, opacity=0.6))],
        layout=dict(
            xaxis_title="n",
            yaxis_title="Gap (difference between subsequent p)",
            title="Tie-point gaps vs n (n=2 to 100)",
        ),
    )
    fig.show()


def main() -> None:
    """Requires ``data/tie_points.pkl`` from ``OBDsaveSourceData.py --save-tie-points`` (or ``--all``)."""
    plot_tie_vertical_segments_p_n(DEFAULT_TIE_POINTS_PATH)


if __name__ == "__main__":
    main()
