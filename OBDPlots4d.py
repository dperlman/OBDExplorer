import plotly.graph_objects as go
import numpy as np
from scipy.stats import binom

# 3-simplex (tetrahedron): points (x,y,z,w) with x+y+z+w=1, x,y,z,w >= 0.
# Embed as a regular tetrahedron in 3D so it looks "straight on" (like an equilateral triangle view).

def simplex_to_regular_tetrahedron(x, y, z):
    """Map probability (x,y,z) with w=1-x-y-z to a regular tetrahedron in 3D.
    Vertices (1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1) -> (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1).
    """
    w = 1 - x - y - z
    # Barycentric: x*v0 + y*v1 + z*v2 + w*v3
    v0, v1, v2, v3 = (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
    if np.isscalar(x):
        return (x * v0[0] + y * v1[0] + z * v2[0] + w * v3[0],
                x * v0[1] + y * v1[1] + z * v2[1] + w * v3[1],
                x * v0[2] + y * v1[2] + z * v2[2] + w * v3[2])
    return (
        x * v0[0] + y * v1[0] + z * v2[0] + w * v3[0],
        x * v0[1] + y * v1[1] + z * v2[1] + w * v3[1],
        x * v0[2] + y * v1[2] + z * v2[2] + w * v3[2],
    )

# Curve: binomial(n=3) as p runs from 0.5 to 1
p_vals = np.linspace(0.5, 1, 101)
cx = binom.pmf(0, 3, p_vals)
cy = binom.pmf(1, 3, p_vals)
cz = binom.pmf(2, 3, p_vals)
curve_x, curve_y, curve_z = simplex_to_regular_tetrahedron(cx, cy, cz)

# Second curve: same PMF values but sorted (min, ..., max) at each p, then map to tetrahedron
cw = binom.pmf(3, 3, p_vals)
pmf_matrix = np.stack([cx, cy, cz, cw], axis=0)  # 4 x n
pmf_sorted = np.sort(pmf_matrix, axis=0)
sx = pmf_sorted[0]
sy = pmf_sorted[1]
sz = pmf_sorted[2]
# (sx, sy, sz) is the sorted point in simplex coords; fourth is pmf_sorted[3] = 1-sx-sy-sz
sorted_x, sorted_y, sorted_z = simplex_to_regular_tetrahedron(sx, sy, sz)

# Tetrahedron vertices in probability coords, then map to regular tetrahedron
vertices_simplex = [
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0),
]
vertices = [simplex_to_regular_tetrahedron(v[0], v[1], v[2]) for v in vertices_simplex]
faces = [(0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3)]
face_colors = ["rgba(230,25,75,0.4)", "rgba(60,180,75,0.4)", "rgba(67,99,216,0.4)", "rgba(245,130,49,0.4)"]

x = np.array([v[0] for v in vertices])
y = np.array([v[1] for v in vertices])
z = np.array([v[2] for v in vertices])
i_face = [f[0] for f in faces]
j_face = [f[1] for f in faces]
k_face = [f[2] for f in faces]

curve = go.Scatter3d(
    x=curve_x, y=curve_y, z=curve_z,
    mode="lines",
    line=dict(color="darkred", width=6),
    name="Binomial(n=3)",
)
curve_sorted = go.Scatter3d(
    x=sorted_x, y=sorted_y, z=sorted_z,
    mode="lines",
    line=dict(color="darkblue", width=6),
    name="Binomial(n=3), sorted",
)
# Point (0,1,2,3)/6 in 4D simplex -> (0, 1/6, 1/3) in (x,y,z) coords, w=1/2
pt_simplex = (0, 1 / 6, 1 / 3)
pt_x, pt_y, pt_z = simplex_to_regular_tetrahedron(*pt_simplex)
point_marker = go.Scatter3d(
    x=[pt_x], y=[pt_y], z=[pt_z],
    mode="markers",
    marker=dict(size=4, color="black", symbol="diamond"),
    name="(0,1,2,3)/6",
)
fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, i=i_face, j=j_face, k=k_face, facecolor=face_colors), curve, curve_sorted, point_marker])
# Camera: look from (1,1,1) toward centroid so the opposite face is seen straight on (equilateral triangle)
fig.update_layout(
    title="3-simplex (regular tetrahedron): P(sum=0)+P(sum=1)+P(sum=2)+P(sum=3)=1",
    scene=dict(
        xaxis=dict(title="P(sum=0)", range=[-1.5, 1.5]),
        yaxis=dict(title="P(sum=1)", range=[-1.5, 1.5]),
        zaxis=dict(title="P(sum=2)", range=[-1.5, 1.5]),
        aspectmode="cube",
        camera=dict(
            eye=dict(x=1.8, y=1.8, z=1.8),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0),
        ),
    ),
)
fig.show()
