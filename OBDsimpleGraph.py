import plotly.graph_objects as go
import numpy as np
from scipy.stats import binom

# --- Constants (edit these) ---
P_MIN, P_MAX = 0.5, 1.0
P_NUM = 1001
N_MIN, N_MAX = 2,100
SCALE_BEG = False  # if True, scale each curve so it starts at 0 and ends at 1
# -----------------------------

# Expected value of the "ordered" binomial: sort the PMF (q_0 <= ... <= q_n), then E = sum(k * q_k).
# Vectorized over p_vals: one call per n instead of per (n, p).
def expected_ordered_binomial_vectorized(n, p_vals):
    k_vals = np.arange(n + 1, dtype=float)
    # pmf shape (n+1, len(p_vals))
    pmf = binom.pmf(k_vals[:, np.newaxis], n, p_vals[np.newaxis, :])
    pmf_sorted = np.sort(pmf, axis=0)
    return np.sum(k_vals[:, np.newaxis] * pmf_sorted, axis=0)

p_vals = np.linspace(P_MIN, P_MAX, P_NUM)

fig = go.Figure()
for n in range(N_MIN, N_MAX + 1):
    ev = expected_ordered_binomial_vectorized(n, p_vals)
    y = ev / n
    if SCALE_BEG:
        y_min, y_max = y.min(), y.max()
        if y_max > y_min:
            y = (y - y_min) / (y_max - y_min)
        else:
            y = np.zeros_like(y)
    fig.add_trace(go.Scatter(x=p_vals, y=y, mode="lines", name=f"n={n}", line=dict(color="black", width=1)))

y_axis_title = "E[ordered binomial] / n (scaled 0–1)" if SCALE_BEG else "E[ordered binomial] / n"
fig.update_layout(
    title=f"Expected value of ordered binomial, p ∈ [{P_MIN}, {P_MAX}], n = {N_MIN}..{N_MAX}",
    xaxis_title="p",
    yaxis_title=y_axis_title,
    showlegend=True,
)
fig.show()
