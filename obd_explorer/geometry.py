"""Tie band geometry (numpy); colormap RGB lives in qt_graphics."""

from __future__ import annotations

import numpy as np

from obd_explorer.grid import BinomialGrid
from obd_explorer.model import TieColorSpec
from obd_explorer.numeric import interpolate_y_at_p


def tie_ij_rightmost_in(
    ent_sorted: list[tuple[float, int | None, int | None]],
    p_lo: float,
    p_hi: float,
) -> tuple[int | None, int | None]:
    cand = [e for e in ent_sorted if p_lo + 1e-15 < e[0] <= p_hi + 1e-15]
    if not cand:
        return None, None
    e = max(cand, key=lambda x: x[0])
    return e[1], e[2]


def expand_strips_for_dual_full(
    strips: list[tuple[float, float, int | None, int | None]],
    ent_sorted: list[tuple[float, int | None, int | None]],
    tie_spec: TieColorSpec,
    vp_range_norm: str,
) -> list[tuple[float, float, int | None, int | None]]:
    if vp_range_norm != "full" or tie_spec.left == tie_spec.right:
        return strips
    out: list[tuple[float, float, int | None, int | None]] = []
    for p_l, p_r, ti, tj in strips:
        if p_l + 1e-15 < 0.5 < p_r - 1e-15:
            t_left = tie_ij_rightmost_in(ent_sorted, p_l, 0.5)
            t_right = tie_ij_rightmost_in(ent_sorted, 0.5, p_r)
            out.append((p_l, 0.5, t_left[0], t_left[1]))
            out.append((0.5, p_r, t_right[0], t_right[1]))
        else:
            out.append((p_l, p_r, ti, tj))
    return out


def vp_p_window(
    mode: str, grid: BinomialGrid
) -> tuple[float, float, int, int]:
    v = mode.strip().lower()
    ps = grid.p_steps
    ph = grid.p_half_start
    if v == "full":
        return 0.0, 1.0, 0, ps - 1
    if v == "left":
        return 0.0, 0.5, 0, ph
    if v == "right":
        return 0.5, 1.0, ph, ps - 1
    raise ValueError(
        'VP_P_RANGE must be "full", "left", or "right" (case insensitive), '
        f"got {mode!r}"
    )


def parse_tie_lines_direction(s: str) -> str:
    v = str(s).strip().lower()
    if v == "up":
        return "up"
    if v == "down":
        return "down"
    raise ValueError(
        'TIE_LINES_DIRECTION must be "up" or "down" '
        f"(case insensitive), got {s!r}"
    )


def tie_source_n_for_band(ki: int, nk: int, n_list: list[int], direction: str) -> int:
    if direction == "up":
        return n_list[ki - 1] if ki > 0 else nk
    return nk


def tie_entry_for_p(entries: list[tuple], p: float) -> tuple | None:
    """Return the tie entry whose ``p`` is closest to ``p`` (for slope / metadata lookup)."""
    if not entries:
        return None
    best: tuple | None = None
    best_d = float("inf")
    for e in entries:
        d = abs(float(e[0]) - float(p))
        if d < best_d:
            best_d = d
            best = e
    return best


def coord_range_for_n(entries: list[tuple], key: str) -> tuple[int, int] | tuple[float, float] | None:
    if key == "i":
        idx = 1
        vals = [e[idx] for e in entries if len(e) > idx and e[idx] is not None]
        if not vals:
            return None
        return min(vals), max(vals)
    if key == "j":
        idx = 2
        vals = [e[idx] for e in entries if len(e) > idx and e[idx] is not None]
        if not vals:
            return None
        return min(vals), max(vals)
    if key == "l":
        vals = [
            float(e[3])
            for e in entries
            if len(e) > 3 and e[3] is not None and np.isfinite(float(e[3]))
        ]
        if not vals:
            return None
        return float(min(vals)), float(max(vals))
    if key == "r":
        vals = [
            float(e[4])
            for e in entries
            if len(e) > 4 and e[4] is not None and np.isfinite(float(e[4]))
        ]
        if not vals:
            return None
        return float(min(vals)), float(max(vals))
    if key == "d":
        vals = []
        for e in entries:
            if len(e) > 4 and e[3] is not None and e[4] is not None:
                sl = float(e[3])
                sr = float(e[4])
                if np.isfinite(sl) and np.isfinite(sr):
                    vals.append(sr - sl)
        if not vals:
            return None
        return float(min(vals)), float(max(vals))
    if key == "e":
        vals = []
        for e in entries:
            if len(e) > 4 and e[3] is not None and e[4] is not None:
                sl = float(e[3])
                sr = float(e[4])
                if np.isfinite(sl) and np.isfinite(sr):
                    vals.append(sl - sr)
        if not vals:
            return None
        return float(min(vals)), float(max(vals))
    raise ValueError(
        f'coord_range_for_n key must be "i", "j", "l", "r", "d", or "e"; got {key!r}'
    )


def fill_strip_polygon_xy(
    p_left: float,
    p_right: float,
    x_top: np.ndarray,
    y_top: np.ndarray,
    x_bot: np.ndarray | None,
    y_bot: np.ndarray | None,
    y_flat_bottom: float | None,
) -> tuple[list[float], list[float]]:
    yt_l = interpolate_y_at_p(x_top, y_top, p_left)
    yt_r = interpolate_y_at_p(x_top, y_top, p_right)
    if x_bot is None or y_bot is None:
        if y_flat_bottom is None:
            return [], []
        yf = float(y_flat_bottom)
        xb_pts = [float(p_left), float(p_right)]
        yb_pts = [yf, yf]
    else:
        yb_l = interpolate_y_at_p(x_bot, y_bot, p_left)
        yb_r = interpolate_y_at_p(x_bot, y_bot, p_right)
        mb = (x_bot > p_left) & (x_bot < p_right)
        xb_pts = [float(p_left)] + x_bot[mb].astype(float).tolist() + [float(p_right)]
        yb_pts = [float(yb_l)] + y_bot[mb].astype(float).tolist() + [float(yb_r)]

    mt = (x_top > p_left) & (x_top < p_right)
    xf_top = [float(p_left)] + x_top[mt].astype(float).tolist() + [float(p_right)]
    yf_top = [float(yt_l)] + y_top[mt].astype(float).tolist() + [float(yt_r)]
    xr_top = xf_top[::-1]
    yr_top = yf_top[::-1]

    px: list[float] = [xb_pts[0]]
    py: list[float] = [yb_pts[0]]
    for i in range(1, len(xb_pts)):
        px.append(xb_pts[i])
        py.append(yb_pts[i])
    if abs(float(yb_pts[-1]) - float(yr_top[0])) > 1e-14:
        px.append(float(p_right))
        py.append(float(yr_top[0]))
    for i in range(1, len(xr_top)):
        px.append(float(xr_top[i]))
        py.append(float(yr_top[i]))
    return px, py


def tie_entry_sort_key(
    e: tuple[float, int | None, int | None],
) -> tuple[float, int, int]:
    p, i, j = e[0], e[1], e[2]
    return (p, -1 if i is None else i, -1 if j is None else j)
