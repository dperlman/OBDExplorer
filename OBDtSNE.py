import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binom
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# --- Constants (edit these) ---
P_MIN, P_MAX = 0.5, 1.0
P_NUM = 1001
N_MIN, N_MAX = 2, 10
METHOD = "PCA" # "TSNE" or "UMAP" or "PCA"
PERPLEXITY = 50 # only used for t-SNE
N_NEIGHBORS = 50 # only used for UMAP
MIN_DIST = 0.01 # only used for UMAP

# -----------------------------

p_vals = np.linspace(P_MIN, P_MAX, P_NUM)

CELL_SIZE = 600  # px per subplot (square); increase for a larger figure in the browser
R = N_MAX - N_MIN + 1  # number of rows
method_upper = METHOD.upper()
if method_upper == "TSNE":
    dim1_label, dim2_label = "t-SNE 1", "t-SNE 2"
elif method_upper == "PCA":
    dim1_label, dim2_label = "PC1", "PC2"
else:
    dim1_label, dim2_label = "UMAP 1", "UMAP 2"

# Build subplot titles: row r = n = N_MIN + r - 1
subplot_titles = []
for n in range(N_MIN, N_MAX + 1):
    subplot_titles.append(f"n={n} (binomial)")
    subplot_titles.append(f"n={n} (ordered)")

fig = make_subplots(
    rows=R, cols=2,
    subplot_titles=subplot_titles,
    vertical_spacing=0.06,
    horizontal_spacing=0.08,
)

for i, n in enumerate(range(N_MIN, N_MAX + 1)):
    row = i + 1
    k_vals = np.arange(n + 1, dtype=float)
    unsorted_pmf = binom.pmf(k_vals[:, np.newaxis], n, p_vals[np.newaxis, :])  # (n+1, P_NUM)
    ordered_pmf = np.sort(unsorted_pmf, axis=0)

    X_unsorted = unsorted_pmf.T   # (P_NUM, n+1)
    X_ordered = ordered_pmf.T     # (P_NUM, n+1)
    X_combined = np.vstack([X_unsorted, X_ordered])

    if method_upper == "TSNE":
        reducer = TSNE(n_components=2, random_state=0, perplexity=min(PERPLEXITY, (2 * P_NUM) - 1))
        embed_combined = reducer.fit_transform(X_combined)
    elif method_upper == "PCA":
        reducer = PCA(n_components=2, random_state=0)
        embed_combined = reducer.fit_transform(X_combined)
    else:  # UMAP
        reducer = umap.UMAP(n_components=2, n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST, random_state=0, n_jobs=1)
        embed_combined = reducer.fit_transform(X_combined)
    embed_unsorted = embed_combined[:P_NUM]
    embed_ordered = embed_combined[P_NUM:]

    x_min, x_max = embed_combined[:, 0].min(), embed_combined[:, 0].max()
    y_min, y_max = embed_combined[:, 1].min(), embed_combined[:, 1].max()

    fig.add_trace(
        go.Scatter(
            x=embed_unsorted[:, 0], y=embed_unsorted[:, 1],
            mode="markers",
            marker=dict(size=4, color=p_vals, colorscale="Viridis", colorbar=dict(title="p"), showscale=True),
            text=[f"p={p:.4f}" for p in p_vals],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=row, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=embed_ordered[:, 0], y=embed_ordered[:, 1],
            mode="markers",
            marker=dict(size=4, color=p_vals, colorscale="Viridis", colorbar=dict(title="p"), showscale=True),
            text=[f"p={p:.4f}" for p in p_vals],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=row, col=2,
    )

    fig.update_xaxes(range=[x_min, x_max], row=row, col=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[y_min, y_max], row=row, col=1)
    fig.update_xaxes(range=[x_min, x_max], row=row, col=2)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[y_min, y_max], row=row, col=2)

fig.update_layout(
    title_text=f"{method_upper}: binomial vs ordered, n = {N_MIN}..{N_MAX}, p ∈ [{P_MIN}, {P_MAX}]",
    width=2 * CELL_SIZE,
    height=R * CELL_SIZE,
)
# Axis labels on bottom row only
fig.update_xaxes(title_text=dim1_label, row=R, col=1)
fig.update_yaxes(title_text=dim2_label, row=R, col=1)
fig.update_xaxes(title_text=dim1_label, row=R, col=2)
fig.update_yaxes(title_text=dim2_label, row=R, col=2)
fig.show()
