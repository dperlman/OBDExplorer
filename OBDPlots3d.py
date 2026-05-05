import plotly.graph_objects as go
import numpy as np
from scipy.stats import binom

# Curve: binomial(n=2) probabilities as p runs from 0 to 1
# x = P(X=0), y = P(X=1), z = P(X=2); curve lies on the plane and inside the triangle
p_vals = np.linspace(0.5, 1, 101)
curve_x = binom.pmf(0, 2, p_vals)
curve_y = binom.pmf(1, 2, p_vals)
curve_z = binom.pmf(2, 2, p_vals)

# Second curve: same PMF values but sorted (min, mid, max) at each p
pmf_matrix = np.stack([curve_x, curve_y, curve_z], axis=0)  # 3 x n
pmf_sorted = np.sort(pmf_matrix, axis=0)
sorted_x, sorted_y, sorted_z = pmf_sorted[0], pmf_sorted[1], pmf_sorted[2]

# Six regions by ordering of x,y,z. Partition the triangle into 6 triangles from center O
# to the 6 boundary points (3 corners + 3 edge midpoints). Order around: A, M_xy, B, M_yz, C, M_xz.
O = (1 / 3, 1 / 3, 1 / 3)
A = (1, 0, 0)
M_xy = (0.5, 0.5, 0)
B = (0, 1, 0)
M_yz = (0, 0.5, 0.5)
C = (0, 0, 1)
M_xz = (0.5, 0, 0.5)
# Vertices: 0=O, 1=A, 2=M_xy, 3=B, 4=M_yz, 5=C, 6=M_xz
vertices = [O, A, M_xy, B, M_yz, C, M_xz]
# Six triangles (O + two boundary points each), color by ordering
# 0: O-M_yz-C (x<y<z), 1: O-C-M_xz (x<z<y), 2: O-M_xy-B (y<x<z), 3: O-B-M_yz (y<z<x),
# 4: O-A-M_xy (z<y<x), 5: O-M_xz-A (z<x<y)
triangles = [(0, 4, 5), (0, 5, 6), (0, 2, 3), (0, 3, 4), (0, 1, 2), (0, 6, 1)]
colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]

x = np.array([v[0] for v in vertices])
y = np.array([v[1] for v in vertices])
z = np.array([v[2] for v in vertices])
i_face = [t[0] for t in triangles]
j_face = [t[1] for t in triangles]
k_face = [t[2] for t in triangles]
face_colors = colors

curve = go.Scatter3d(
    x=curve_x, y=curve_y, z=curve_z,
    mode="lines",
    line=dict(color="darkred", width=6),
    name="Binomial(n=2)",
)
curve_sorted = go.Scatter3d(
    x=sorted_x, y=sorted_y, z=sorted_z,
    mode="lines",
    line=dict(color="darkblue", width=6),
    name="Binomial(n=2), sorted",
)
fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i_face, j=j_face, k=k_face, facecolor=face_colors), curve, curve_sorted])
fig.update_layout(
    title="Plane x + y + z = 1 (x, y, z ≥ 0)",
    scene=dict(
        xaxis=dict(title="P(sum=0)", range=[0, 1]),
        yaxis=dict(title="P(sum=1)", range=[0, 1]),
        zaxis=dict(title="P(sum=2)", range=[0, 1]),
        aspectmode="cube",
    ),
)
fig.show()
