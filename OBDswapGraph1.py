import math
import numpy as np
import plotly.graph_objects as go
from scipy.stats import binom

# --- Constants (edit these) ---
N_MIN, N_MAX = 10000, 10100
LOG_SCALE = False
# -----------------------------


def expected_ordered_binomial(n, p):
    """Sort the binomial(n, p) PMF (q_0 <= ... <= q_n), then E = sum(k * q_k)."""
    pmf = np.array([binom.pmf(k, n, p) for k in range(n + 1)])
    q = np.sort(pmf)
    return np.sum(np.arange(n + 1) * q)


def first_swap_prob(n):
    """Different formula for even vs odd n."""
    if n % 2 == 0:
        return 0.5 * (n + 2) / (n + 1)
    else:
        sqrt_n3 = math.sqrt(n + 3)
        sqrt_n1 = math.sqrt(n - 1)
        return sqrt_n3 / (sqrt_n3 + sqrt_n1)


n_vals = np.arange(N_MIN, N_MAX + 1, dtype=int)
# E[ordered binomial] at first-swap p minus E[ordered binomial] at p=0.5
y_vals = np.array([
    expected_ordered_binomial(n, first_swap_prob(n)) - expected_ordered_binomial(n, 0.5)
    for n in n_vals
])

fig = go.Figure()
fig.add_trace(go.Scatter(x=n_vals, y=y_vals, mode="lines", line=dict(color="black", width=1)))
fig.update_layout(
    title=f"E[ordered bin] at first_swap_prob(n) − E[ordered bin] at p=0.5, n = {N_MIN}..{N_MAX}",
    xaxis_title="n",
    yaxis_title="E at first swap − E at p=0.5",
    yaxis=dict(type="log" if LOG_SCALE else "linear"),
)
fig.show()
