"""Offscreen PyQtGraph figure + PNG/PDF export (headless)."""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, SVGExporter
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtGui import QPageLayout, QPageSize, QPainter
from PyQt6.QtCore import QMarginsF, QSizeF
from PyQt6.QtPrintSupport import QPrinter

from obd_explorer.constants import SCALED, SORTED
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
from obd_explorer.model import TieColorSpec
from obd_explorer.numeric import interpolate_y_at_p, subtract_endpoint_chord
from obd_explorer.qt_graphics import PolyFillBatch, TieLineBatch, is_color_name, tie_rgba_for_color_key
from obd_explorer.tie_data import resolve_tie_draw_entries

@dataclass
class HeadlessExportConfig:
    n_min: int = 2
    n_max: int = 100
    p_steps: int = 1001
    vp_p_range: str = "right"
    endpoint_chord_detrend: bool = False
    clip_to_tie_range: bool = False
    tie_lines_direction: str = "up"
    tie_lines_to_border: bool = True
    tie_color_left: str = "i"
    tie_color_right: str = "j"
    tie_colormap: str | None = None
    fill_colormap: str | None = "gist_rainbow"
    fill_from: str = "left"
    flip_fill_on_left: bool = True
    tie_line_opacity: float = 0.9
    tie_line_opacity_k: float = 0.0
    graph_line_opacity: float = 0.3
    graph_line_opacity_k: float = 1.0
    tie_line_device_px: float = 1.0
    graph_line_device_px: float = 0.1
    width_in: float = 12.0
    height_in: float = 8.0
    dpi: int = 400
    figure_background: str = "white"
    graph_manifest: str | None = None
    graph_shards_dir: str | None = None
    tie_manifest: str | None = None
    output_path: str = "OBDGraphWithTiePyQTGraph.png"
    export_format: str = "png"  # png | pdf | svg
    export_backend: str = "pyqtgraph"  # pyqtgraph | matplotlib


def export_graph_headless(cfg: HeadlessExportConfig, *, verbose: bool = True) -> None:
    t0 = time.monotonic()
    if cfg.n_min > cfg.n_max:
        print("ERROR: n_min must be <= n_max.", file=sys.stderr)
        sys.exit(1)
    if cfg.p_steps < 2:
        print("ERROR: p_steps must be at least 2.", file=sys.stderr)
        sys.exit(1)
    if float(cfg.width_in) <= 0.0 or float(cfg.height_in) <= 0.0:
        print("ERROR: width_in and height_in must be > 0.", file=sys.stderr)
        sys.exit(1)
    if int(cfg.dpi) <= 0:
        print("ERROR: dpi must be > 0.", file=sys.stderr)
        sys.exit(1)

    fmt = cfg.export_format.strip().lower()
    if fmt not in ("png", "pdf", "svg"):
        print("ERROR: export_format must be png, pdf, or svg.", file=sys.stderr)
        sys.exit(1)
    backend = cfg.export_backend.strip().lower()
    if backend not in ("pyqtgraph", "matplotlib"):
        print("ERROR: export_backend must be pyqtgraph or matplotlib.", file=sys.stderr)
        sys.exit(1)
    render_w = max(1, int(round(float(cfg.width_in) * int(cfg.dpi))))
    render_h = max(1, int(round(float(cfg.height_in) * int(cfg.dpi))))

    grid = resolve_binomial_grid(
        n_min=cfg.n_min,
        n_max=cfg.n_max,
        p_steps=cfg.p_steps,
        graph_manifest_path=cfg.graph_manifest,
        graph_shards_dir=cfg.graph_shards_dir,
    )

    try:
        tie_spec = TieColorSpec.parse_lr(cfg.tie_color_left, cfg.tie_color_right)
        vp_p_min, vp_p_max, p_ix_lo, p_ix_hi = vp_p_window(cfg.vp_p_range, grid)
        tie_dir = parse_tie_lines_direction(cfg.tie_lines_direction)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    vp_range_norm = cfg.vp_p_range.strip().lower()
    n_list = list(range(cfg.n_min, cfg.n_max + 1))
    tie_draw = resolve_tie_draw_entries(
        n_vals=n_list,
        tie_p_min=vp_p_min,
        tie_p_max=vp_p_max,
        tie_manifest_path=cfg.tie_manifest,
    )

    if cfg.clip_to_tie_range:
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

    endpoint_active = bool(cfg.endpoint_chord_detrend)
    vp_xy: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    all_y_arrays: list[np.ndarray] = []
    for n in n_list:
        x_arr, y_arr = build_vp_xy(
            grid,
            n,
            p_ix_lo,
            p_ix_hi,
            explorer_endpoint_y=endpoint_active,
            sorted_mode=SORTED,
            scaled_mode=SCALED,
        )
        if endpoint_active:
            if vp_range_norm == "full":
                mid_ix = int(np.searchsorted(x_arr, 0.5, side="right"))
                y_left = subtract_endpoint_chord(x_arr[:mid_ix], y_arr[:mid_ix])
                y_right = subtract_endpoint_chord(x_arr[mid_ix - 1 :], y_arr[mid_ix - 1 :])
                y_arr = np.concatenate([y_left, y_right[1:]])
            else:
                y_arr = subtract_endpoint_chord(x_arr, y_arr)
        vp_xy[n] = (x_arr, y_arr)
        finite_y = y_arr[np.isfinite(y_arr)]
        if finite_y.size > 0:
            all_y_arrays.append(finite_y)

    if not all_y_arrays:
        print("ERROR: No y data for curves.", file=sys.stderr)
        sys.exit(1)

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

    tie_alpha = float(np.clip(cfg.tie_line_opacity, 0.0, 1.0))
    tie_line_opacity_k = float(cfg.tie_line_opacity_k)
    graph_opacity_base = float(np.clip(cfg.graph_line_opacity, 0.0, 1.0))
    graph_opacity_k = float(cfg.graph_line_opacity_k)

    do_fill = bool(cfg.fill_colormap)
    do_tie_lines = bool(cfg.tie_colormap)
    fill_cmap = str(cfg.fill_colormap).strip() if do_fill else ""
    tie_cmap = str(cfg.tie_colormap).strip() if do_tie_lines else ""
    tie_fixed_color: tuple[float, float, float] | None = None
    if do_tie_lines and is_color_name(tie_cmap):
        qc_tie = QtGui.QColor(tie_cmap)
        tie_fixed_color = (qc_tie.redF(), qc_tie.greenF(), qc_tie.blueF())

    lw_tie = float(cfg.tie_line_device_px)
    lw_graph = float(cfg.graph_line_device_px)

    fill_batches: list[tuple[list[float], list[float], QtGui.QColor]] = []
    tie_segments: list[tuple[float, float, float, QtGui.QColor]] = []

    geo_order = list(reversed(n_list)) if endpoint_active else list(n_list)
    bands: list[tuple[int | None, int | None, int, float | None]] = []
    for i in range(1, len(geo_order)):
        n_above = geo_order[i]
        n_below = geo_order[i - 1]
        n_tie = n_below if tie_dir == "up" else n_above
        bands.append((n_above, n_below, n_tie, None))

    if cfg.tie_lines_to_border:
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

        if do_fill:
            ps = sorted({e[0] for e in entries})
            if ps:
                ent_sorted = sorted(entries, key=tie_entry_sort_key)
                _ent_lookup: dict[float, tuple[int | None, int | None]] = {
                    e[0]: (e[1], e[2]) for e in ent_sorted
                }

                def _tie_ij_at(p_anchor: float) -> tuple[int | None, int | None]:
                    return _ent_lookup.get(p_anchor, (None, None))

                fill_from_base = cfg.fill_from.strip().lower()
                flip_left = cfg.flip_fill_on_left and vp_range_norm == "full"

                def _fill_side_at(p_mid: float) -> str:
                    if flip_left and p_mid < 0.5:
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
                    key = tie_spec.key_at_p(p_mid, vp_range_norm)
                    ent_sl = tie_entry_for_p(ent_sorted, p_mid)
                    sl = ent_sl[3] if ent_sl and len(ent_sl) > 3 else None
                    sr = ent_sl[4] if ent_sl and len(ent_sl) > 4 else None
                    rgba = tie_rgba_for_color_key(
                        key,
                        ii=ti,
                        jj=tj,
                        slope_left=sl,
                        slope_right=sr,
                        alpha=tie_alpha,
                        cmap_name=fill_cmap,
                        i_range=band_i_range,
                        j_range=band_j_range,
                        l_range=band_l_range,
                        r_range=band_r_range,
                        d_range=band_d_range,
                        e_range=band_e_range,
                    )
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
                    r, g, b, a = rgba
                    fill_batches.append((fx, fy, QtGui.QColor.fromRgbF(r, g, b, a)))

        if do_tie_lines:
            for e in entries:
                p_use = float(e[0])
                ti, tj = e[1], e[2]
                sl = e[3] if len(e) > 3 else None
                sr = e[4] if len(e) > 4 else None
                if tie_line_opacity_k == 0.0:
                    seg_alpha = tie_alpha
                else:
                    denom_tl = tie_line_opacity_k * (n_tie - 2) + 1.0
                    if denom_tl <= 0.0:
                        continue
                    seg_alpha = float(np.clip(tie_alpha / denom_tl, 0.0, 1.0))
                if seg_alpha <= 0.0:
                    continue
                key = tie_spec.key_at_p(p_use, vp_range_norm)
                rgba = tie_rgba_for_color_key(
                    key,
                    ii=ti,
                    jj=tj,
                    slope_left=sl,
                    slope_right=sr,
                    alpha=seg_alpha,
                    cmap_name=tie_cmap,
                    fixed_color=tie_fixed_color,
                    i_range=band_i_range,
                    j_range=band_j_range,
                    l_range=band_l_range,
                    r_range=band_r_range,
                    d_range=band_d_range,
                    e_range=band_e_range,
                )
                if n_upper is not None:
                    y_top = interpolate_y_at_p(*vp_xy[n_upper], p_use)
                else:
                    y_top = flat_y
                if n_lower is not None:
                    y_bot = interpolate_y_at_p(*vp_xy[n_lower], p_use)
                else:
                    y_bot = flat_y
                if np.isfinite(y_bot) and np.isfinite(y_top):
                    c0 = QtGui.QColor.fromRgbF(rgba[0], rgba[1], rgba[2], rgba[3])
                    tie_segments.append((p_use, float(y_bot), float(y_top), c0))

    curve_specs: list[tuple[np.ndarray, np.ndarray, float]] = []
    if graph_opacity_base > 0.0:
        for n in n_list:
            if graph_opacity_k == 0.0:
                alpha_n = graph_opacity_base
            else:
                denom = graph_opacity_k * (n - 2) + 1.0
                if denom <= 0.0:
                    continue
                alpha_n = float(np.clip(graph_opacity_base / denom, 0.0, 1.0))
            if alpha_n <= 0.0:
                continue
            x_arr, y_arr = vp_xy[n]
            curve_specs.append((x_arr, y_arr, alpha_n))

    if endpoint_active:
        y_label = ("E[rank]" if SORTED else "E[X]") + " − chord"
    else:
        y_label = "E/n" if SCALED else ("E[rank]" if SORTED else "E[X]")
    title = (
        ("E[rank]" if SORTED else "E[X]")
        + ("" if endpoint_active else ("/n" if SCALED else ""))
        + (" (endpoint detrended)" if endpoint_active else "")
        + " vs p (n = "
        + str(cfg.n_min)
        + "–"
        + str(cfg.n_max)
        + ", banded swap lines)"
    )

    out = cfg.output_path
    if backend == "matplotlib":
        _export_matplotlib(
            out_path=out,
            fmt=fmt,
            width_in=float(cfg.width_in),
            height_in=float(cfg.height_in),
            dpi=int(cfg.dpi),
            x_lo=vp_p_min,
            x_hi=vp_p_max,
            y_lo=y_lo,
            y_hi=y_hi,
            y_label=y_label,
            title=title,
            fill_batches=fill_batches,
            tie_segments=tie_segments,
            curve_specs=curve_specs,
            lw_tie=lw_tie,
            lw_graph=lw_graph,
            figure_background=cfg.figure_background,
        )
    else:
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        app.setFont(QtGui.QFont("Helvetica"))

        pw = pg.PlotWidget()
        pw.resize(render_w, render_h)
        pw.setBackground(cfg.figure_background)
        plot_item = pw.plotItem
        plot_item.showGrid(x=False, y=False)
        plot_item.setRange(
            xRange=(vp_p_min, vp_p_max),
            yRange=(y_lo, y_hi),
            padding=0.0,
        )
        plot_item.setLabel("bottom", "p")
        plot_item.setLabel("left", y_label)
        plot_item.setTitle(title, size="10pt")

        pw.show()
        app.processEvents()

        b_lo_x, b_hi_x, b_lo_y, b_hi_y = vp_p_min, vp_p_max, y_lo, y_hi
        if fill_batches:
            fb_item = PolyFillBatch(fill_batches, b_lo_x, b_hi_x, b_lo_y, b_hi_y)
            fb_item.setZValue(-10)
            plot_item.addItem(fb_item)
        if tie_segments:
            tb_item = TieLineBatch(tie_segments, b_lo_x, b_hi_x, b_lo_y, b_hi_y, lw_tie)
            tb_item.setZValue(-5)
            plot_item.addItem(tb_item)

        for x_arr, y_arr, alpha_n in curve_specs:
            pen = pg.mkPen(
                color=(0, 0, 0, int(round(255.0 * alpha_n))),
                width=lw_graph,
                cosmetic=True,
            )
            curve = plot_item.plot(x_arr, y_arr, pen=pen)
            curve.setZValue(2)

        app.processEvents()

        if fmt == "png":
            exporter = ImageExporter(plot_item)
            exp_params = exporter.parameters()
            exp_params["width"] = render_w
            exp_params["height"] = render_h
            exp_params["antialias"] = False
            exp_params["background"] = QtGui.QColor(cfg.figure_background)
            exporter.export(out)
        elif fmt == "svg":
            _export_svg(plot_item, out, render_w, render_h, cfg.figure_background)
        else:
            _export_pdf_vector_simple(
                plot_item,
                out,
                width_in=float(cfg.width_in),
                height_in=float(cfg.height_in),
                dpi=int(cfg.dpi),
                figure_background=cfg.figure_background,
            )
        pw.close()

    if verbose:
        print(f"Wrote {os.path.abspath(out)} ({fmt.upper()}).")
        print(f"Export time: {time.monotonic() - t0:.2f}s")


def _export_pdf_vector_simple(
    plot_item: pg.PlotItem,
    path: str,
    *,
    width_in: float,
    height_in: float,
    dpi: int,
    figure_background: str,
) -> None:
    """Baseline PyQtGraph vector PDF export through Qt print backend."""
    printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
    printer.setOutputFileName(path)
    printer.setResolution(int(dpi))
    printer.setPageSize(
        QPageSize(QSizeF(float(width_in), float(height_in)), QPageSize.Unit.Inch)
    )
    printer.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Inch)

    exporter = ImageExporter(plot_item)
    source_rect = QtCore.QRectF(exporter.getSourceRect())
    page_rect_dev = printer.pageRect(QPrinter.Unit.DevicePixel)
    target_rect = QtCore.QRectF(
        float(page_rect_dev.left()),
        float(page_rect_dev.top()),
        float(max(1.0, page_rect_dev.width())),
        float(max(1.0, page_rect_dev.height())),
    )
    qc_bg = QtGui.QColor(figure_background)

    painter = QPainter(printer)
    try:
        painter.fillRect(target_rect, qc_bg)
        exporter.setExportMode(True, {"painter": painter, "background": qc_bg})
        exporter.getScene().render(painter, target_rect, source_rect)
    finally:
        exporter.setExportMode(False)
        painter.end()


def _export_matplotlib(
    *,
    out_path: str,
    fmt: str,
    width_in: float,
    height_in: float,
    dpi: int,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    y_label: str,
    title: str,
    fill_batches: list[tuple[list[float], list[float], QtGui.QColor]],
    tie_segments: list[tuple[float, float, float, QtGui.QColor]],
    curve_specs: list[tuple[np.ndarray, np.ndarray, float]],
    lw_tie: float,
    lw_graph: float,
    figure_background: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(float(width_in), float(height_in)), dpi=int(dpi))
    fig.patch.set_facecolor(figure_background)
    ax.set_facecolor(figure_background)
    # Use a tighter layout than Matplotlib defaults so exported pages are not mostly margins.
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.10, top=0.95)
    ax.set_xlim(float(x_lo), float(x_hi))
    ax.set_ylim(float(y_lo), float(y_hi))
    ax.set_xlabel("p")
    ax.set_ylabel(y_label)
    ax.set_title(title)

    for fx, fy, brush in fill_batches:
        rgba = (brush.redF(), brush.greenF(), brush.blueF(), brush.alphaF())
        ax.fill(fx, fy, color=rgba, linewidth=0.0)
    for p_val, y_bot, y_top, color in tie_segments:
        rgba = (color.redF(), color.greenF(), color.blueF(), color.alphaF())
        ax.plot([p_val, p_val], [y_bot, y_top], color=rgba, linewidth=float(lw_tie))
    for x_arr, y_arr, alpha_n in curve_specs:
        ax.plot(x_arr, y_arr, color=(0.0, 0.0, 0.0, float(alpha_n)), linewidth=float(lw_graph))

    fig.savefig(out_path, format=fmt)
    plt.close(fig)


def _export_svg(
    plot_item: pg.PlotItem,
    path: str,
    width_px: int,
    height_px: int,
    figure_background: str,
) -> None:
    """Vector SVG via PyQtGraph ``SVGExporter`` (custom SVG assembly, not ``QSvgGenerator``)."""
    exporter = SVGExporter(plot_item)
    params = exporter.parameters()
    params["width"] = float(width_px)
    params["height"] = float(height_px)
    params["background"] = QtGui.QColor(figure_background)
    exporter.export(path)
