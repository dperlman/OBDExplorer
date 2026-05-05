"""PyQtGraph graphics items and colormap helpers."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

_cmap_cache: dict[str, object] = {}


def cmap_rgb(cmap_name: str, t: float) -> tuple[float, float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    name = str(cmap_name).strip()
    cmap_pg = _cmap_cache.get(name)
    if cmap_pg is None:
        cmap_pg = pg.colormap.getFromMatplotlib(name)
        _cmap_cache[name] = cmap_pg
    rgba = cmap_pg.map(np.array([t], dtype=float))[0]
    return (float(rgba[0]) / 255.0, float(rgba[1]) / 255.0, float(rgba[2]) / 255.0)


def is_color_name(s: str) -> bool:
    return QtGui.QColor(str(s).strip()).isValid()


def tie_rgba_for_color_key(
    key: str,
    *,
    ii: int | None,
    jj: int | None,
    slope_left: float | None,
    slope_right: float | None,
    alpha: float,
    cmap_name: str = "",
    fixed_color: tuple[float, float, float] | None = None,
    i_range: tuple[int, int] | None = None,
    j_range: tuple[int, int] | None = None,
    l_range: tuple[float, float] | None = None,
    r_range: tuple[float, float] | None = None,
    d_range: tuple[float, float] | None = None,
    e_range: tuple[float, float] | None = None,
) -> tuple[float, float, float, float]:
    if fixed_color is not None:
        return (fixed_color[0], fixed_color[1], fixed_color[2], float(alpha))
    k = str(key).strip().lower()
    if k == "black":
        return (0.0, 0.0, 0.0, float(alpha))
    if k == "i":
        if ii is None or i_range is None:
            return (0.0, 0.0, 0.0, float(alpha))
        lo, hi = i_range
        t = (ii - lo) / (hi - lo) if hi > lo else 0.0
    elif k == "j":
        if jj is None or j_range is None:
            return (0.0, 0.0, 0.0, float(alpha))
        lo, hi = j_range
        t = (jj - lo) / (hi - lo) if hi > lo else 0.0
    elif k == "l":
        if slope_left is None or not np.isfinite(float(slope_left)) or l_range is None:
            return (0.0, 0.0, 0.0, float(alpha))
        lo, hi = l_range
        v = float(slope_left)
        t = (v - lo) / (hi - lo) if hi > lo else 0.0
    elif k == "r":
        if slope_right is None or not np.isfinite(float(slope_right)) or r_range is None:
            return (0.0, 0.0, 0.0, float(alpha))
        lo, hi = r_range
        v = float(slope_right)
        t = (v - lo) / (hi - lo) if hi > lo else 0.0
    elif k == "d":
        if (
            slope_left is None
            or slope_right is None
            or not np.isfinite(float(slope_left))
            or not np.isfinite(float(slope_right))
            or d_range is None
        ):
            return (0.0, 0.0, 0.0, float(alpha))
        lo, hi = d_range
        v = float(slope_right) - float(slope_left)
        t = (v - lo) / (hi - lo) if hi > lo else 0.0
    elif k == "e":
        if (
            slope_left is None
            or slope_right is None
            or not np.isfinite(float(slope_left))
            or not np.isfinite(float(slope_right))
            or e_range is None
        ):
            return (0.0, 0.0, 0.0, float(alpha))
        lo, hi = e_range
        v = float(slope_left) - float(slope_right)
        t = (v - lo) / (hi - lo) if hi > lo else 0.0
    else:
        raise ValueError(
            f"Unknown tie color key {key!r} (expected black, i, j, l, r, d, e)"
        )
    r, g, b = cmap_rgb(cmap_name, float(np.clip(t, 0.0, 1.0)))
    return (r, g, b, float(alpha))


# Matplotlib names as registered (see colormap reference); diverging + miscellaneous
# from https://matplotlib.org/stable/gallery/color/colormap_reference.html — omit names
# already listed above to avoid duplicates.
CMAP_NAMES = (
    "gist_rainbow",
    "turbo",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "hsv",
    "jet",
    "nipy_spectral",
    "twilight",
    "coolwarm",
    "winter",
    "spring",
    "summer",
    "autumn",
    "wistia",
    # Diverging
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "bwr",
    "seismic",
    "berlin",
    "managua",
    "vanimo",
    # Miscellaneous
    "flag",
    "prism",
    "ocean",
    "gist_earth",
    "terrain",
    "gist_stern",
    "gnuplot",
    "gnuplot2",
    "CMRmap",
    "cubehelix",
    "brg",
    "rainbow",
    "gist_ncar",
)
FIXED_COLOR_NAMES = ("black", "red", "blue", "green", "orange", "purple")


def populate_cmap_combo(combo: QtWidgets.QComboBox, *, include_none: bool) -> None:
    combo.clear()
    if include_none:
        combo.addItem("(none)", None)
    for name in CMAP_NAMES:
        combo.addItem(name, name)
    for name in FIXED_COLOR_NAMES:
        combo.addItem(f"color: {name}", name)


class PolyFillBatch(pg.GraphicsObject):
    """Many filled polygons in data coordinates (single paint pass)."""

    def __init__(
        self,
        fills: list[tuple[list[float], list[float], QtGui.QColor]],
        x_lo: float,
        x_hi: float,
        y_lo: float,
        y_hi: float,
    ) -> None:
        super().__init__()
        self._fills = fills
        self._x_lo = x_lo
        self._x_hi = x_hi
        self._y_lo = y_lo
        self._y_hi = y_hi

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            self._x_lo,
            self._y_lo,
            self._x_hi - self._x_lo,
            self._y_hi - self._y_lo,
        ).normalized()

    def paint(self, p: QtGui.QPainter, *_args) -> None:
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        for fx, fy, brush in self._fills:
            poly = QtGui.QPolygonF()
            for i in range(len(fx)):
                poly.append(QtCore.QPointF(float(fx[i]), float(fy[i])))
            p.setBrush(brush)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.drawPolygon(poly)


class TieLineBatch(pg.GraphicsObject):
    """Vertical tie segments in data coordinates.

    Each segment is either ``(p, y_bot, y_top, QColor)`` solid, or
    ``(p, y_bot, y_top, QColor, QColor)`` with a vertical linear gradient (bottom → top).
    """

    def __init__(
        self,
        segments: list[
            tuple[float, float, float, QtGui.QColor]
            | tuple[float, float, float, QtGui.QColor, QtGui.QColor]
        ],
        x_lo: float,
        x_hi: float,
        y_lo: float,
        y_hi: float,
        width_px: float,
    ) -> None:
        super().__init__()
        self._segs = segments
        self._x_lo = x_lo
        self._x_hi = x_hi
        self._y_lo = y_lo
        self._y_hi = y_hi
        self._w = float(width_px)

    def boundingRect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            self._x_lo,
            self._y_lo,
            self._x_hi - self._x_lo,
            self._y_hi - self._y_lo,
        ).normalized()

    def paint(self, p: QtGui.QPainter, *_args) -> None:
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        for seg in self._segs:
            p_val = float(seg[0])
            y_bot = float(seg[1])
            y_top = float(seg[2])
            if len(seg) == 5:
                c0, c1 = seg[3], seg[4]
                grad = QtGui.QLinearGradient(
                    QtCore.QPointF(p_val, y_bot), QtCore.QPointF(p_val, y_top)
                )
                grad.setColorAt(0.0, c0)
                grad.setColorAt(1.0, c1)
                pen = QtGui.QPen(QtGui.QBrush(grad), self._w)
            else:
                pen = QtGui.QPen(seg[3])
                pen.setWidthF(self._w)
            pen.setCosmetic(True)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
            p.setPen(pen)
            p.drawLine(
                QtCore.QPointF(p_val, y_bot),
                QtCore.QPointF(p_val, y_top),
            )
