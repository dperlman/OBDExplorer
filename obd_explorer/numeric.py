"""NumPy-only helpers (no PyQt / pyqtgraph)."""

from __future__ import annotations

import numpy as np


def expected_rank(point: dict) -> float:
    n = len(point["x"]) - 1
    y = np.asarray(point["y"], dtype=float)
    perm = np.asarray(point["perm"], dtype=int)
    return float(np.dot(np.arange(n + 1, dtype=float), y[perm]))


def subtract_endpoint_chord(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Subtract secant through first and last finite (x, y) pairs."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = int(min(xs.size, ys.size))
    if n == 0:
        return ys.copy()
    finite = np.isfinite(xs[:n]) & np.isfinite(ys[:n])
    if not np.any(finite):
        return ys[:n].astype(float, copy=True)
    idx = np.flatnonzero(finite)
    i0 = int(idx[0])
    i1 = int(idx[-1])
    if i1 <= i0:
        return ys[:n].astype(float, copy=True)
    x0, x1 = float(xs[i0]), float(xs[i1])
    y0, y1 = float(ys[i0]), float(ys[i1])
    dx = x1 - x0
    if dx == 0.0:
        return ys[:n].astype(float, copy=True)
    out = ys[:n].astype(float, copy=True)
    L = y0 + (y1 - y0) * (xs[:n] - x0) / dx
    m = np.isfinite(out)
    out[m] = out[m] - L[m]
    return out


def interpolate_y_at_p(xa: np.ndarray, ya: np.ndarray, p_target: float) -> float:
    if xa.size == 0:
        return float("nan")
    if p_target <= float(xa[0]):
        return float(ya[0])
    if p_target >= float(xa[-1]):
        return float(ya[-1])
    i = int(np.searchsorted(xa, p_target, side="right")) - 1
    i = max(0, min(i, xa.size - 2))
    x0, x1 = float(xa[i]), float(xa[i + 1])
    y0, y1 = float(ya[i]), float(ya[i + 1])
    if x1 == x0:
        return y0
    t = (p_target - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
