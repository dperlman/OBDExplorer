"""Load tie_points.pkl and plot (count vs n, or KDE of p for largest n). Float data only."""
import pickle

import numpy as np
import plotly.graph_objects as go


def load_tie_points_float(path: str = "tie_points.pkl") -> dict:
    """Load pickle; return float_by_n (n -> array of p). Uses float_by_n only."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("float_by_n", {})


def plot_count_vs_n(path: str = "tie_points.pkl") -> None:
    """Plot number of tie points vs n."""
    float_by_n = load_tie_points_float(path)
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


def plot_scatter_n2_to_100(path: str = "tie_points.pkl") -> None:
    """Load pkl, for n=2 to 100 scatter gaps (differences between subsequent p) vs n with small points."""
    float_by_n = load_tie_points_float(path)
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


def main(path: str = "tie_points.pkl") -> None:
    plot_scatter_n2_to_100(path)


if __name__ == "__main__":
    main()
