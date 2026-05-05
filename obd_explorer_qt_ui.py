"""
Interactive desktop explorer for OBD graph curves, tie lines, and fills (PyQt6 + PyQtGraph).

Loads graph data from graph shard manifests and tie data from tie shard manifests
(see ``OBDsaveSourceData`` default paths under ``data/``).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from obd_explorer.constants import DEFAULT_GRAPH_P_STEPS, FIGURE_BACKGROUND, SCALED, SORTED
from obd_explorer.geometry import (
    coord_range_for_n,
    expand_strips_for_dual_full,
    fill_strip_polygon_xy,
    parse_tie_lines_direction,
    tie_entry_for_p,
    tie_entry_sort_key,
    vp_p_window,
)
from obd_explorer.grid import build_vp_xy, resolve_binomial_grid
from obd_explorer.model import TIE_COLOR_AXIS_CHOICES, TieColorSpec
from obd_explorer.numeric import interpolate_y_at_p, subtract_endpoint_chord
from obd_explorer.qt_graphics import (
    PolyFillBatch,
    TieLineBatch,
    is_color_name,
    populate_cmap_combo,
    tie_rgba_for_color_key,
)
from obd_explorer.tie_data import resolve_tie_draw_entries
class OBDGraphExplorerWindow(QtWidgets.QMainWindow):
    """Main window: plot + control panel + geometry cache for fast redraws."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OBD Graph Explorer (Qt)")
        self.resize(1280, 820)

        self.grid = resolve_binomial_grid(
            n_min=2, n_max=100, p_steps=DEFAULT_GRAPH_P_STEPS
        )
        self._n_lo = self.grid.n_min
        self._n_hi = self.grid.n_max

        t0 = time.monotonic()
        self._curve_raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._curve_det: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        p_last = self.grid.p_steps - 1
        for n in range(self._n_lo, self._n_hi + 1):
            x_arr, y_arr = build_vp_xy(
                self.grid,
                n,
                0,
                p_last,
                explorer_endpoint_y=False,
                sorted_mode=SORTED,
                scaled_mode=SCALED,
            )
            self._curve_raw[n] = (x_arr, y_arr)
            xe, ye = build_vp_xy(
                self.grid,
                n,
                0,
                p_last,
                explorer_endpoint_y=True,
                sorted_mode=SORTED,
                scaled_mode=SCALED,
            )
            mid_ix = int(np.searchsorted(xe, 0.5, side="right"))
            y_left = subtract_endpoint_chord(xe[:mid_ix], ye[:mid_ix])
            y_right = subtract_endpoint_chord(xe[mid_ix - 1 :], ye[mid_ix - 1 :])
            y_det = np.concatenate([y_left, y_right[1:]])
            self._curve_det[n] = (xe, y_det)

        n_list_full = list(range(self._n_lo, self._n_hi + 1))
        self._tie_full = resolve_tie_draw_entries(
            n_vals=n_list_full, tie_p_min=0.0, tie_p_max=1.0
        )
        print(f"Precompute curves + ties: {time.monotonic() - t0:.2f}s", flush=True)

        self._cached_geom_key: tuple | None = None
        self._geom_pkg: dict | None = None
        self._fill_item: PolyFillBatch | None = None
        self._tie_item: TieLineBatch | None = None
        self._curve_items: dict[int, pg.PlotDataItem] = {}

        self._rebuild_timer = QtCore.QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.timeout.connect(self._rebuild_plot)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(268)
        scroll.setMaximumWidth(320)
        panel = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(panel)

        # --- Range ---
        g_range = QtWidgets.QGroupBox("Range")
        gl = QtWidgets.QFormLayout(g_range)
        self.spin_n_min = QtWidgets.QSpinBox()
        self.spin_n_max = QtWidgets.QSpinBox()
        for sp in (self.spin_n_min, self.spin_n_max):
            sp.setRange(self._n_lo, self._n_hi)
        self.spin_n_min.setValue(self._n_lo)
        self.spin_n_max.setValue(self._n_hi)
        gl.addRow("n min", self.spin_n_min)
        gl.addRow("n max", self.spin_n_max)
        self.combo_p_range = QtWidgets.QComboBox()
        self.combo_p_range.addItem("Full [0, 1]", "full")
        self.combo_p_range.addItem("Left [0, 0.5]", "left")
        self.combo_p_range.addItem("Right [0.5, 1]", "right")
        self.combo_p_range.setCurrentIndex(0)
        gl.addRow("p window", self.combo_p_range)
        pv.addWidget(g_range)

        # --- Mode ---
        g_mode = QtWidgets.QGroupBox("Mode")
        ml = QtWidgets.QVBoxLayout(g_mode)
        self.chk_detrend = QtWidgets.QCheckBox("Endpoint chord detrend")
        self.chk_clip = QtWidgets.QCheckBox("Clip to tie p range")
        ml.addWidget(self.chk_detrend)
        ml.addWidget(self.chk_clip)
        pv.addWidget(g_mode)

        # --- Tie lines ---
        g_tie = QtWidgets.QGroupBox("Tie lines")
        tl = QtWidgets.QFormLayout(g_tie)
        self.combo_tie_cmap = QtWidgets.QComboBox()
        populate_cmap_combo(self.combo_tie_cmap, include_none=True)
        self.combo_tie_cmap.setCurrentIndex(0)
        tl.addRow("Colormap", self.combo_tie_cmap)
        self.combo_tie_color_left = QtWidgets.QComboBox()
        self.combo_tie_color_right = QtWidgets.QComboBox()
        for axis in TIE_COLOR_AXIS_CHOICES:
            self.combo_tie_color_left.addItem(axis, axis)
            self.combo_tie_color_right.addItem(axis, axis)
        ix_i = self.combo_tie_color_left.findData("i")
        if ix_i >= 0:
            self.combo_tie_color_left.setCurrentIndex(ix_i)
        ix_j = self.combo_tie_color_right.findData("j")
        if ix_j >= 0:
            self.combo_tie_color_right.setCurrentIndex(ix_j)
        tl.addRow("Tie color (left / 0–0.5)", self.combo_tie_color_left)
        tl.addRow("Tie color (right / 0.5–1)", self.combo_tie_color_right)
        self.combo_dir = QtWidgets.QComboBox()
        self.combo_dir.addItem("up", "up")
        self.combo_dir.addItem("down", "down")
        tl.addRow("Direction", self.combo_dir)
        self.chk_to_border = QtWidgets.QCheckBox("Tie lines to border")
        self.chk_to_border.setChecked(True)
        tl.addRow(self.chk_to_border)
        self.slider_tie_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_tie_opacity.setRange(0, 100)
        self.slider_tie_opacity.setValue(90)
        tl.addRow("Opacity", self.slider_tie_opacity)
        self.spin_tie_opacity_k = QtWidgets.QDoubleSpinBox()
        self.spin_tie_opacity_k.setRange(0.0, 10.0)
        self.spin_tie_opacity_k.setSingleStep(0.1)
        self.spin_tie_opacity_k.setValue(0.0)
        self.spin_tie_opacity_k.setToolTip(
            "0 = uniform tie line opacity; else alpha scales per band as "
            "opacity / (K × (n_tie − 2) + 1), matching curve opacity K."
        )
        tl.addRow("Opacity K", self.spin_tie_opacity_k)
        self.spin_tie_width = QtWidgets.QDoubleSpinBox()
        self.spin_tie_width.setRange(0.1, 5.0)
        self.spin_tie_width.setSingleStep(0.1)
        self.spin_tie_width.setValue(1.0)
        tl.addRow("Width (px)", self.spin_tie_width)
        pv.addWidget(g_tie)

        # --- Fill ---
        g_fill = QtWidgets.QGroupBox("Fill")
        fl = QtWidgets.QFormLayout(g_fill)
        self.combo_fill_cmap = QtWidgets.QComboBox()
        populate_cmap_combo(self.combo_fill_cmap, include_none=True)
        ix = self.combo_fill_cmap.findData("gist_rainbow")
        if ix >= 0:
            self.combo_fill_cmap.setCurrentIndex(ix)
        fl.addRow("Colormap", self.combo_fill_cmap)
        self.combo_fill_from = QtWidgets.QComboBox()
        self.combo_fill_from.addItem("left", "left")
        self.combo_fill_from.addItem("right", "right")
        fl.addRow("Fill from", self.combo_fill_from)
        self.chk_flip_fill = QtWidgets.QCheckBox("Flip fill on left (p < 0.5)")
        self.chk_flip_fill.setChecked(True)
        fl.addRow(self.chk_flip_fill)
        pv.addWidget(g_fill)

        # --- Curves ---
        g_curves = QtWidgets.QGroupBox("Curves")
        cl = QtWidgets.QFormLayout(g_curves)
        self.slider_graph_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_graph_opacity.setRange(0, 100)
        self.slider_graph_opacity.setValue(30)
        cl.addRow("Opacity", self.slider_graph_opacity)
        self.spin_graph_opacity_k = QtWidgets.QDoubleSpinBox()
        self.spin_graph_opacity_k.setRange(0.0, 10.0)
        self.spin_graph_opacity_k.setSingleStep(0.1)
        self.spin_graph_opacity_k.setValue(1.0)
        cl.addRow("Opacity K", self.spin_graph_opacity_k)
        self.spin_graph_width = QtWidgets.QDoubleSpinBox()
        self.spin_graph_width.setRange(0.05, 5.0)
        self.spin_graph_width.setSingleStep(0.05)
        self.spin_graph_width.setValue(0.1)
        cl.addRow("Width (px)", self.spin_graph_width)
        pv.addWidget(g_curves)

        pv.addStretch(1)
        scroll.setWidget(panel)
        h.addWidget(scroll)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(FIGURE_BACKGROUND)
        self.plot_item = self.plot_widget.plotItem
        self.plot_item.showGrid(x=False, y=False)
        h.addWidget(self.plot_widget, stretch=1)

        for n in range(self._n_lo, self._n_hi + 1):
            pen = pg.mkPen(
                color=(0, 0, 0, 80), width=0.1, cosmetic=True
            )
            c = self.plot_item.plot([], [], pen=pen)
            c.setZValue(2)
            self._curve_items[n] = c

        for w in (
            self.spin_n_min,
            self.spin_n_max,
            self.combo_p_range,
            self.chk_detrend,
            self.chk_clip,
            self.combo_tie_cmap,
            self.combo_tie_color_left,
            self.combo_tie_color_right,
            self.combo_dir,
            self.chk_to_border,
            self.slider_tie_opacity,
            self.spin_tie_opacity_k,
            self.spin_tie_width,
            self.combo_fill_cmap,
            self.combo_fill_from,
            self.chk_flip_fill,
            self.slider_graph_opacity,
            self.spin_graph_opacity_k,
            self.spin_graph_width,
        ):
            if isinstance(w, QtWidgets.QAbstractSlider):
                w.valueChanged.connect(self._schedule_rebuild)
            elif isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(self._schedule_rebuild)
            elif isinstance(w, QtWidgets.QAbstractButton):
                w.toggled.connect(self._schedule_rebuild)
            elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                w.valueChanged.connect(self._schedule_rebuild)
            else:
                raise TypeError(f"Unexpected control widget type: {type(w)!r}")

        self.statusBar().showMessage("Ready")

        self._schedule_rebuild()

    def _schedule_rebuild(self) -> None:
        self._rebuild_timer.start(0)

    def _geom_key(self) -> tuple:
        return (
            int(self.spin_n_min.value()),
            int(self.spin_n_max.value()),
            str(self.combo_p_range.currentData()),
            bool(self.chk_detrend.isChecked()),
            str(self.combo_dir.currentData()),
            bool(self.chk_to_border.isChecked()),
            bool(self.chk_clip.isChecked()),
            str(self.combo_fill_from.currentData()),
            bool(self.chk_flip_fill.isChecked()),
            str(self.combo_tie_color_left.currentData()),
            str(self.combo_tie_color_right.currentData()),
        )

    def _compute_geometry_package(self) -> dict | None:
        n_min = int(self.spin_n_min.value())
        n_max = int(self.spin_n_max.value())
        if n_min > n_max:
            n_min, n_max = n_max, n_min
        n_list = list(range(n_min, n_max + 1))
        if not n_list:
            return None

        vp_range_norm = str(self.combo_p_range.currentData())
        try:
            tie_dir = parse_tie_lines_direction(str(self.combo_dir.currentData()))
        except ValueError:
            return None
        try:
            tie_spec = TieColorSpec.parse_lr(
                str(self.combo_tie_color_left.currentData()),
                str(self.combo_tie_color_right.currentData()),
            )
        except ValueError:
            return None

        endpoint_detrend = bool(self.chk_detrend.isChecked())
        clip_to_tie = bool(self.chk_clip.isChecked())
        to_border = bool(self.chk_to_border.isChecked())
        fill_from_base = str(self.combo_fill_from.currentData()).strip().lower()
        flip_left = bool(self.chk_flip_fill.isChecked())

        grid = self.grid
        vp_p_min, vp_p_max, p_ix_lo, p_ix_hi = vp_p_window(vp_range_norm, grid)

        tie_draw: dict[int, list[tuple]] = {}
        for n in n_list:
            ents = [
                e
                for e in self._tie_full.get(n, [])
                if vp_p_min <= e[0] <= vp_p_max
            ]
            if ents:
                tie_draw[n] = ents

        if clip_to_tie:
            all_tie_p = [e[0] for ents in tie_draw.values() for e in ents]
            if all_tie_p:
                tp_lo = float(min(all_tie_p))
                tp_hi = float(max(all_tie_p))
                pv = grid.p_values
                new_ix_lo = next(
                    (i for i in range(p_ix_lo, p_ix_hi + 1) if pv[i] >= tp_lo - 1e-12),
                    p_ix_lo,
                )
                new_ix_hi = next(
                    (i for i in range(p_ix_hi, p_ix_lo - 1, -1) if pv[i] <= tp_hi + 1e-12),
                    p_ix_hi,
                )
                p_ix_lo = new_ix_lo
                p_ix_hi = new_ix_hi
                vp_p_min = float(pv[p_ix_lo])
                vp_p_max = float(pv[p_ix_hi])

        curves_src = self._curve_det if endpoint_detrend else self._curve_raw
        vp_xy: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        all_y_arrays: list[np.ndarray] = []
        for n in n_list:
            x_f, y_f = curves_src[n]
            x_arr = x_f[p_ix_lo : p_ix_hi + 1]
            y_arr = y_f[p_ix_lo : p_ix_hi + 1]
            vp_xy[n] = (x_arr, y_arr)
            finite_y = y_arr[np.isfinite(y_arr)]
            if finite_y.size > 0:
                all_y_arrays.append(finite_y)

        if not all_y_arrays:
            return None

        all_y_concat = np.concatenate(all_y_arrays)
        y_lo = float(all_y_concat.min())
        y_hi = float(all_y_concat.max())
        y_span = y_hi - y_lo
        y_min_span = 1e-3
        if y_span < y_min_span:
            y_mid = (y_lo + y_hi) / 2.0
            y_lo = y_mid - y_min_span / 2.0
            y_hi = y_mid + y_min_span / 2.0
        else:
            y_pad = max(y_span * 0.05, y_min_span * 0.01)
            y_lo -= y_pad
            y_hi += y_pad

        fill_records: list[tuple] = []
        tie_records: list[tuple] = []

        geo_order = list(reversed(n_list)) if endpoint_detrend else list(n_list)
        bands: list[tuple[int | None, int | None, int, float | None]] = []
        for i in range(1, len(geo_order)):
            n_above = geo_order[i]
            n_below = geo_order[i - 1]
            n_tie = n_below if tie_dir == "up" else n_above
            bands.append((n_above, n_below, n_tie, None))
        if to_border and geo_order:
            if tie_dir == "up":
                bands.append((None, geo_order[-1], geo_order[-1], y_hi))
            else:
                bands.append((geo_order[0], None, geo_order[0], y_lo))

        for n_upper, n_lower, n_tie, flat_y in bands:
            entries = tie_draw.get(n_tie, [])
            if not entries:
                continue
            band_i_range = coord_range_for_n(entries, "i")
            band_j_range = coord_range_for_n(entries, "j")
            band_l_range = coord_range_for_n(entries, "l")
            band_r_range = coord_range_for_n(entries, "r")
            band_d_range = coord_range_for_n(entries, "d")
            band_e_range = coord_range_for_n(entries, "e")

            ps = sorted({e[0] for e in entries})
            if ps:
                ent_sorted = sorted(entries, key=tie_entry_sort_key)
                _ent_lookup: dict[float, tuple[int | None, int | None]] = {
                    e[0]: (e[1], e[2]) for e in ent_sorted
                }

                def _tie_ij_at(p_anchor: float) -> tuple[int | None, int | None]:
                    return _ent_lookup.get(p_anchor, (None, None))

                flip_fill = flip_left and vp_range_norm == "full"

                def _fill_side_at(p_mid: float) -> str:
                    if flip_fill and p_mid < 0.5:
                        return "right" if fill_from_base == "left" else "left"
                    return fill_from_base

                strips: list[tuple[float, float, int | None, int | None]] = []
                for k in range(len(ps) - 1):
                    p_l, p_r = ps[k], ps[k + 1]
                    side = _fill_side_at(0.5 * (p_l + p_r))
                    if side == "left":
                        ti, tj = _tie_ij_at(p_l)
                    else:
                        ti, tj = _tie_ij_at(p_r)
                    strips.append((p_l, p_r, ti, tj))
                if ps[0] > vp_p_min + 1e-15:
                    side = _fill_side_at(0.5 * (vp_p_min + ps[0]))
                    if side == "right":
                        ti0, tj0 = _tie_ij_at(ps[0])
                        strips.insert(0, (vp_p_min, ps[0], ti0, tj0))
                if ps[-1] < vp_p_max - 1e-15:
                    side = _fill_side_at(0.5 * (ps[-1] + vp_p_max))
                    if side == "left":
                        tia, tja = _tie_ij_at(ps[-1])
                        strips.append((ps[-1], vp_p_max, tia, tja))
                strips = expand_strips_for_dual_full(
                    strips, ent_sorted, tie_spec, vp_range_norm
                )

                for p_left, p_right, ti, tj in strips:
                    p_mid = 0.5 * (float(p_left) + float(p_right))
                    if n_upper is not None:
                        x_up, y_up = vp_xy[n_upper]
                    else:
                        x_up, y_up = None, None
                    if n_lower is not None:
                        x_dn, y_dn = vp_xy[n_lower]
                    else:
                        x_dn, y_dn = None, None
                    if x_up is not None and x_dn is not None:
                        fx, fy = fill_strip_polygon_xy(
                            float(p_left),
                            float(p_right),
                            x_up,
                            y_up,
                            x_dn,
                            y_dn,
                            None,
                        )
                    elif x_up is not None:
                        fx, fy = fill_strip_polygon_xy(
                            float(p_left),
                            float(p_right),
                            x_up,
                            y_up,
                            None,
                            None,
                            flat_y,
                        )
                    else:
                        fx, fy = fill_strip_polygon_xy(
                            float(p_left),
                            float(p_right),
                            x_dn,
                            y_dn,
                            None,
                            None,
                            flat_y,
                        )
                    if len(fx) < 3 or not all(np.isfinite(v) for v in (*fx, *fy)):
                        continue
                    ent_sl = tie_entry_for_p(ent_sorted, p_mid)
                    sl = ent_sl[3] if ent_sl and len(ent_sl) > 3 else None
                    sr = ent_sl[4] if ent_sl and len(ent_sl) > 4 else None
                    fill_records.append(
                        (
                            list(fx),
                            list(fy),
                            float(p_mid),
                            ti,
                            tj,
                            band_i_range,
                            band_j_range,
                            band_l_range,
                            band_r_range,
                            band_d_range,
                            band_e_range,
                            sl,
                            sr,
                        )
                    )

            for e in entries:
                p_use = float(e[0])
                ti, tj = e[1], e[2]
                sl = e[3] if len(e) > 3 else None
                sr = e[4] if len(e) > 4 else None
                if n_upper is not None:
                    y_top = interpolate_y_at_p(*vp_xy[n_upper], float(p_use))
                else:
                    y_top = flat_y
                if n_lower is not None:
                    y_bot = interpolate_y_at_p(*vp_xy[n_lower], float(p_use))
                else:
                    y_bot = flat_y
                if np.isfinite(y_bot) and np.isfinite(y_top):
                    tie_records.append(
                        (
                            float(p_use),
                            float(y_bot),
                            float(y_top),
                            ti,
                            tj,
                            band_i_range,
                            band_j_range,
                            band_l_range,
                            band_r_range,
                            band_d_range,
                            band_e_range,
                            int(n_tie),
                            sl,
                            sr,
                        )
                    )

        return {
            "tie_spec": tie_spec,
            "fill_records": fill_records,
            "tie_records": tie_records,
            "y_lo": y_lo,
            "y_hi": y_hi,
            "vp_p_min": vp_p_min,
            "vp_p_max": vp_p_max,
            "n_list": n_list,
            "vp_range_norm": vp_range_norm,
            "endpoint_detrend": endpoint_detrend,
            "vp_xy": vp_xy,
        }

    def _rebuild_plot(self) -> None:
        t0 = time.monotonic()
        self.statusBar().showMessage("Computing…")

        gkey = self._geom_key()
        if self._cached_geom_key != gkey or self._geom_pkg is None:
            pkg_new = self._compute_geometry_package()
            self._geom_pkg = pkg_new
            self._cached_geom_key = gkey if pkg_new is not None else None

        pkg = self._geom_pkg
        if pkg is None:
            self.statusBar().showMessage("Nothing to plot (check n range)")
            return

        tie_spec = pkg["tie_spec"]
        vp_range_norm = pkg["vp_range_norm"]
        endpoint_detrend = pkg["endpoint_detrend"]
        vp_p_min = pkg["vp_p_min"]
        vp_p_max = pkg["vp_p_max"]
        y_lo = pkg["y_lo"]
        y_hi = pkg["y_hi"]
        n_list = pkg["n_list"]
        vp_xy = pkg["vp_xy"]

        do_fill = bool(self.combo_fill_cmap.currentData())
        tie_cmap_raw = self.combo_tie_cmap.currentData()
        do_tie_lines = tie_cmap_raw is not None and str(tie_cmap_raw).strip() != ""
        fill_cmap = str(self.combo_fill_cmap.currentData()).strip() if do_fill else ""
        tie_cmap = str(tie_cmap_raw).strip() if do_tie_lines else ""

        tie_opacity_base = self.slider_tie_opacity.value() / 100.0
        tie_opacity_k = float(self.spin_tie_opacity_k.value())
        tie_fixed_color: tuple[float, float, float] | None = None
        if do_tie_lines and is_color_name(tie_cmap):
            qc_tie = QtGui.QColor(tie_cmap)
            tie_fixed_color = (qc_tie.redF(), qc_tie.greenF(), qc_tie.blueF())

        fill_batches: list[tuple[list[float], list[float], QtGui.QColor]] = []
        if do_fill:
            for rec in pkg["fill_records"]:
                (
                    fx,
                    fy,
                    p_mid,
                    ti,
                    tj,
                    ir,
                    jr,
                    lr,
                    rr,
                    dr,
                    er,
                    sl,
                    sr,
                ) = rec
                key = tie_spec.key_at_p(float(p_mid), vp_range_norm)
                rgba = tie_rgba_for_color_key(
                    key,
                    ii=ti,
                    jj=tj,
                    slope_left=sl,
                    slope_right=sr,
                    alpha=tie_opacity_base,
                    cmap_name=fill_cmap,
                    i_range=ir,
                    j_range=jr,
                    l_range=lr,
                    r_range=rr,
                    d_range=dr,
                    e_range=er,
                )
                r, g, b, a = rgba
                fill_batches.append(
                    (fx, fy, QtGui.QColor.fromRgbF(r, g, b, a))
                )

        tie_segments: list[tuple[float, float, float, QtGui.QColor]] = []
        if do_tie_lines:
            for rec in pkg["tie_records"]:
                (
                    p_use,
                    y_bot,
                    y_top,
                    ti,
                    tj,
                    ir,
                    jr,
                    lr,
                    rr,
                    dr,
                    er,
                    n_tie,
                    sl,
                    sr,
                ) = rec
                if tie_opacity_k == 0.0:
                    seg_alpha = tie_opacity_base
                else:
                    denom_tl = tie_opacity_k * (int(n_tie) - 2) + 1.0
                    if denom_tl <= 0.0:
                        continue
                    seg_alpha = float(
                        np.clip(tie_opacity_base / denom_tl, 0.0, 1.0)
                    )
                if seg_alpha <= 0.0:
                    continue
                key = tie_spec.key_at_p(float(p_use), vp_range_norm)
                rgba = tie_rgba_for_color_key(
                    key,
                    ii=ti,
                    jj=tj,
                    slope_left=sl,
                    slope_right=sr,
                    alpha=seg_alpha,
                    cmap_name=tie_cmap,
                    fixed_color=tie_fixed_color,
                    i_range=ir,
                    j_range=jr,
                    l_range=lr,
                    r_range=rr,
                    d_range=dr,
                    e_range=er,
                )
                c0 = QtGui.QColor.fromRgbF(rgba[0], rgba[1], rgba[2], rgba[3])
                tie_segments.append((float(p_use), float(y_bot), float(y_top), c0))

        b_lo_x, b_hi_x, b_lo_y, b_hi_y = vp_p_min, vp_p_max, y_lo, y_hi

        if self._fill_item is not None:
            self.plot_item.removeItem(self._fill_item)
            self._fill_item = None
        if self._tie_item is not None:
            self.plot_item.removeItem(self._tie_item)
            self._tie_item = None

        if fill_batches:
            self._fill_item = PolyFillBatch(
                fill_batches, b_lo_x, b_hi_x, b_lo_y, b_hi_y
            )
            self._fill_item.setZValue(-10)
            self.plot_item.addItem(self._fill_item)
        if tie_segments:
            self._tie_item = TieLineBatch(
                tie_segments,
                b_lo_x,
                b_hi_x,
                b_lo_y,
                b_hi_y,
                float(self.spin_tie_width.value()),
            )
            self._tie_item.setZValue(-5)
            self.plot_item.addItem(self._tie_item)

        graph_opacity_base = self.slider_graph_opacity.value() / 100.0
        graph_opacity_k = float(self.spin_graph_opacity_k.value())
        lw_graph = float(self.spin_graph_width.value())

        for n in range(self._n_lo, self._n_hi + 1):
            item = self._curve_items[n]
            if n < n_list[0] or n > n_list[-1]:
                item.setVisible(False)
                continue
            if graph_opacity_base <= 0.0:
                item.setVisible(False)
                continue
            if graph_opacity_k == 0.0:
                alpha_n = graph_opacity_base
            else:
                denom = graph_opacity_k * (n - 2) + 1.0
                alpha_n = float(np.clip(graph_opacity_base / denom, 0.0, 1.0))
            if alpha_n <= 0.0:
                item.setVisible(False)
                continue
            x_arr, y_arr = vp_xy[n]
            pen = pg.mkPen(
                color=(0, 0, 0, int(round(255.0 * alpha_n))),
                width=lw_graph,
                cosmetic=True,
            )
            item.setPen(pen)
            item.setData(x_arr, y_arr)
            item.setVisible(True)

        if endpoint_detrend:
            y_label = ("E[rank]" if SORTED else "E[X]") + " − chord"
        else:
            y_label = "E/n" if SCALED else ("E[rank]" if SORTED else "E[X]")
        self.plot_item.setLabel("bottom", "p")
        self.plot_item.setLabel("left", y_label)
        title = (
            ("E[rank]" if SORTED else "E[X]")
            + ("" if endpoint_detrend else ("/n" if SCALED else ""))
            + (" (endpoint detrended)" if endpoint_detrend else "")
            + " vs p (n = "
            + str(n_list[0])
            + "–"
            + str(n_list[-1])
            + ", banded swap lines)"
        )
        self.plot_item.setTitle(title, size="10pt")

        self.plot_item.setRange(
            xRange=(vp_p_min, vp_p_max),
            yRange=(y_lo, y_hi),
            padding=0.0,
        )

        dt = time.monotonic() - t0
        msg = f"Rebuilt in {dt:.2f}s"
        if dt > 0.5:
            msg += " (large n range / fills)"
        self.statusBar().showMessage(msg)



def main() -> None:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Helvetica"))
    win = OBDGraphExplorerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
