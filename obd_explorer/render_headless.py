"""Offscreen PyQtGraph figure + PNG/PDF export (headless)."""

from __future__ import annotations

import os
import pickle
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


HEATMAP_VALUE_CHOICES: tuple[str, ...] = ("i", "j", "l", "r", "d", "e", "ev_n", "eslope_n")
TIE_HEATMAP_VALUE_CHOICES: tuple[str, ...] = ("i", "j", "l", "r", "d", "e", "ev_n")
HEATMAP_PIXEL_MODE_CHOICES: tuple[str, ...] = ("exact", "annotated")
GRAPH_HEATMAP_VALUE_CHOICES: tuple[str, ...] = ("ev_n", "eslope_n")
TIE_PROXY_HEATMAP_VALUE_CHOICES: tuple[str, ...] = ("i", "j", "l", "r", "d", "e")


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


@dataclass
class HeatmapExportConfig:
    n_min: int = 2
    n_max: int = 1000
    p_steps: int = 1001
    p_min: float = 0.5
    p_max: float = 0.6
    value_key: str = "ev_n"  # ev_n|eslope_n
    colormap: str = "viridis"
    show_legend: bool = False
    width_in: float = 12.0
    height_in: float = 10.0
    dpi: int = 400
    graph_manifest: str | None = None
    graph_shards_dir: str | None = None
    output_path: str = "OBDHeatmap.png"
    export_format: str = "png"  # png only
    pixel_mode: str = "annotated"  # exact | annotated
    progress_every: int | None = None
    trim_color_range_percent: int = 1
    per_n_color_range: bool = False


@dataclass
class TieHeatmapExportConfig:
    n_min: int = 2
    n_max: int = 1000
    value_key: str = "d"  # i|j|l|r|d|e|ev_n
    colormap: str = "viridis"
    show_legend: bool = False
    load_from: str = "l"  # l: center-out, r: end-in
    width_in: float = 12.0
    height_in: float = 10.0
    dpi: int = 400
    tie_manifest: str | None = None
    output_path: str = "OBDTieHeatmap.png"
    export_format: str = "png"  # png only
    pixel_mode: str = "annotated"  # exact | annotated
    progress_every: int | None = None
    trim_color_range_percent: int = 1
    per_n_color_range: bool = False


def _compute_color_range(values: np.ndarray, *, trim_color_range_percent: int) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    pct = int(trim_color_range_percent)
    if pct < 0 or pct > 40:
        raise ValueError(f"trim_color_range_percent must be in [0, 40], got {trim_color_range_percent!r}.")
    if pct > 0:
        lo = float(np.percentile(finite, float(pct)))
        hi = float(np.percentile(finite, float(100 - pct)))
    else:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    return lo, hi


def _normalize_heat_for_render(
    heat: np.ndarray,
    *,
    trim_color_range_percent: int,
    per_n_color_range: bool,
) -> tuple[np.ndarray, float | None, float | None]:
    if per_n_color_range:
        out = np.full_like(heat, np.nan, dtype=float)
        for row_i in range(heat.shape[0]):
            row = np.asarray(heat[row_i, :], dtype=float)
            rg = _compute_color_range(row, trim_color_range_percent=trim_color_range_percent)
            if rg is None:
                continue
            lo, hi = rg
            mask = np.isfinite(row)
            if not np.any(mask):
                continue
            if hi <= lo:
                out[row_i, mask] = 0.5
            else:
                out[row_i, mask] = (row[mask] - lo) / (hi - lo)
        return out, 0.0, 1.0

    rg = _compute_color_range(heat, trim_color_range_percent=trim_color_range_percent)
    if rg is None:
        return np.asarray(heat, dtype=float), None, None
    lo, hi = rg
    if hi <= lo:
        hi = lo + 1.0e-12
    return np.asarray(heat, dtype=float), lo, hi


def _tie_heatmap_value_at_record(
    *,
    n: int,
    rec: tuple,
    slope_rec: dict | None,
    value_key: str,
) -> float:
    pairs = rec[1] if isinstance(rec, (list, tuple)) and len(rec) >= 2 else None
    i_val: float = float("nan")
    j_val: float = float("nan")
    if pairs:
        first = pairs[0]
        if isinstance(first, (list, tuple)) and len(first) == 2:
            i_val = float(first[0])
            j_val = float(first[1])

    sl: float = float("nan")
    sr: float = float("nan")
    evn: float = float("nan")
    if isinstance(slope_rec, dict):
        raw_sl = slope_rec.get("slope_left")
        raw_sr = slope_rec.get("slope_right")
        raw_es = slope_rec.get("expected_sorted")
        if raw_sl is not None and np.isfinite(float(raw_sl)):
            sl = float(raw_sl)
        if raw_sr is not None and np.isfinite(float(raw_sr)):
            sr = float(raw_sr)
        if raw_es is not None and np.isfinite(float(raw_es)) and n > 0:
            evn = float(raw_es) / float(n)

    if value_key == "i":
        return i_val
    if value_key == "j":
        return j_val
    if value_key == "l":
        return sl
    if value_key == "r":
        return sr
    if value_key == "d":
        return (sr - sl) if np.isfinite(sr) and np.isfinite(sl) else float("nan")
    if value_key == "e":
        return (sl - sr) if np.isfinite(sr) and np.isfinite(sl) else float("nan")
    if value_key == "ev_n":
        return evn
    raise ValueError(f"Unsupported tie heatmap value key: {value_key!r}")


def _canonical_center_index_for_recs(n: int, recs: list) -> int | None:
    from OBDsaveSourceData import _is_canonical_center_tie

    for rec_idx, item in enumerate(recs):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        pairs = item[1]
        plist = list(pairs) if pairs else []
        if _is_canonical_center_tie(int(n), plist):
            return rec_idx
    return None


def _nearest_values_by_p_grid(
    p_source: np.ndarray,
    v_source: np.ndarray,
    p_target: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor sample ``v_source`` at each ``p_target`` using sorted ``p_source``."""
    if p_source.size == 0 or v_source.size == 0 or p_target.size == 0:
        return np.full(p_target.shape, np.nan, dtype=float)
    order = np.argsort(p_source)
    p_sorted = p_source[order]
    v_sorted = v_source[order]
    idx = np.searchsorted(p_sorted, p_target, side="left")
    left = np.clip(idx - 1, 0, p_sorted.size - 1)
    right = np.clip(idx, 0, p_sorted.size - 1)
    dl = np.abs(p_target - p_sorted[left])
    dr = np.abs(p_sorted[right] - p_target)
    choose_right = dr < dl
    picked = np.where(choose_right, right, left)
    return v_sorted[picked].astype(float, copy=False)


def export_heatmap_headless(cfg: HeatmapExportConfig, *, verbose: bool = True) -> None:
    """Export heatmap image: x=p (viewport range), y=n, color from graph or tie-proxy data."""
    t0 = time.monotonic()
    if cfg.n_min > cfg.n_max:
        print("ERROR: n_min must be <= n_max.", file=sys.stderr)
        sys.exit(1)
    if cfg.p_steps < 2:
        print("ERROR: p_steps must be at least 2.", file=sys.stderr)
        sys.exit(1)
    p_min = float(cfg.p_min)
    p_max = float(cfg.p_max)
    if not np.isfinite(p_min) or not np.isfinite(p_max):
        print("ERROR: p_min and p_max must be finite.", file=sys.stderr)
        sys.exit(1)
    if p_min > p_max:
        print("ERROR: p_min must be <= p_max.", file=sys.stderr)
        sys.exit(1)
    if p_min < 0.0 or p_max > 1.0:
        print("ERROR: p_min/p_max must lie within [0, 1].", file=sys.stderr)
        sys.exit(1)
    if float(cfg.width_in) <= 0.0 or float(cfg.height_in) <= 0.0:
        print("ERROR: width_in and height_in must be > 0.", file=sys.stderr)
        sys.exit(1)
    if int(cfg.dpi) <= 0:
        print("ERROR: dpi must be > 0.", file=sys.stderr)
        sys.exit(1)

    fmt = cfg.export_format.strip().lower()
    if fmt != "png":
        print("ERROR: heatmap export_format must be png.", file=sys.stderr)
        sys.exit(1)
    pixel_mode = str(cfg.pixel_mode).strip().lower()
    if pixel_mode not in HEATMAP_PIXEL_MODE_CHOICES:
        print(
            "ERROR: heatmap pixel_mode must be one of "
            + ", ".join(HEATMAP_PIXEL_MODE_CHOICES)
            + f"; got {cfg.pixel_mode!r}.",
            file=sys.stderr,
        )
        sys.exit(1)
    val_key = str(cfg.value_key).strip().lower()
    if val_key not in HEATMAP_VALUE_CHOICES:
        print(
            "ERROR: heatmap value_key must be one of "
            + ", ".join(HEATMAP_VALUE_CHOICES)
            + f"; got {cfg.value_key!r}.",
            file=sys.stderr,
        )
        sys.exit(1)

    p_steps = int(cfg.p_steps)
    p_vals = np.linspace(p_min, p_max, p_steps, dtype=float)
    n_vals = list(range(cfg.n_min, cfg.n_max + 1))
    x_vals = p_vals
    heat = np.full((len(n_vals), len(x_vals)), np.nan, dtype=float)
    total = len(n_vals)

    if val_key in GRAPH_HEATMAP_VALUE_CHOICES:
        if verbose:
            print(
                f"[heatmap] source=graph_shards value={val_key} n={cfg.n_min}..{cfg.n_max} p_steps={cfg.p_steps} p={p_min:.6f}..{p_max:.6f}",
                file=sys.stderr,
            )
        from OBDsaveSourceData import (
            DEFAULT_GRAPH_SHARDS_DIR,
            _resolve_graph_manifest_path,
            _resolve_manifest_shard_path,
        )

        manifest_path = _resolve_graph_manifest_path(
            cfg.graph_manifest,
            None,
            cfg.graph_shards_dir or DEFAULT_GRAPH_SHARDS_DIR,
        )
        if not os.path.isfile(manifest_path):
            print(
                f"ERROR: Graph shard manifest not found: {os.path.abspath(manifest_path)}",
                file=sys.stderr,
            )
            sys.exit(1)

        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)
        if not isinstance(manifest, dict):
            print(f"ERROR: invalid graph shard manifest payload in {manifest_path!r}.", file=sys.stderr)
            sys.exit(1)
        if manifest.get("format") != "obd.graph_data.shards.v2":
            print(
                f"ERROR: unsupported graph shard manifest format {manifest.get('format')!r} in {manifest_path!r}.",
                file=sys.stderr,
            )
            sys.exit(1)
        manifest_p_vals = np.asarray(manifest.get("p_values", []), dtype=float)
        if manifest_p_vals.size < 2:
            print("ERROR: invalid p_values in graph manifest.", file=sys.stderr)
            sys.exit(1)
        n_entries = manifest.get("n_entries", {})
        if not isinstance(n_entries, dict):
            print("ERROR: invalid n_entries in graph manifest.", file=sys.stderr)
            sys.exit(1)

        for step_index, n in enumerate(n_vals, start=1):
            entry = n_entries.get(str(int(n)))
            if not isinstance(entry, dict):
                continue
            shard_ref = str(entry.get("shard_path", ""))
            if not shard_ref:
                continue
            shard_path = _resolve_manifest_shard_path(manifest_path, shard_ref)
            if not os.path.isfile(shard_path):
                continue

            with open(shard_path, "rb") as f:
                shard = pickle.load(f)
            if not isinstance(shard, dict):
                continue
            if shard.get("format") != "obd.graph_data.n_shard.v2":
                continue
            if "expected_sorted_by_p" not in shard or "expected_sorted_slope_by_p" not in shard:
                continue

            expected_sorted = np.asarray(shard["expected_sorted_by_p"], dtype=float)
            expected_sorted_slope = np.asarray(shard["expected_sorted_slope_by_p"], dtype=float)
            if expected_sorted.size != manifest_p_vals.size or expected_sorted_slope.size != manifest_p_vals.size:
                continue

            if val_key == "ev_n":
                src_vals = expected_sorted / float(n)
            else:  # eslope_n
                src_vals = expected_sorted_slope / float(n)
            row_vals = np.interp(x_vals, manifest_p_vals, src_vals)
            heat[step_index - 1, :] = row_vals

            if cfg.progress_every and total:
                pe = int(cfg.progress_every)
                if step_index % pe == 0 or step_index == total:
                    elapsed = time.monotonic() - t0
                    print(
                        f"[heatmap] graph shards: n={n} step {step_index}/{total} elapsed {elapsed:.2f}s",
                        file=sys.stderr,
                    )
    else:
        if verbose:
            print(
                f"[heatmap] source=tie_shards_nearest_proxy value={val_key} n={cfg.n_min}..{cfg.n_max} p_steps={cfg.p_steps} p={p_min:.6f}..{p_max:.6f}",
                file=sys.stderr,
            )
        from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, iter_tie_points_from_shards

        need_slopes = val_key in ("l", "r", "d", "e")
        n_rows = iter_tie_points_from_shards(
            path=DEFAULT_TIE_OUTPUT,
            n_list=n_vals,
            require_all=False,
            progress=cfg.progress_every,
            include_float_by_n=False,
            include_float_with_pairs_by_n=True,
            include_tie_slope_by_n=need_slopes,
        )
        row_by_n = {n: i for i, n in enumerate(n_vals)}
        for n, payload_for_n in n_rows:
            rr = row_by_n.get(int(n))
            if rr is None:
                continue
            recs = payload_for_n.get("float_with_pairs_by_n") if isinstance(payload_for_n, dict) else None
            if not isinstance(recs, list) or not recs:
                continue
            slope_list = payload_for_n.get("tie_slope_by_n") if isinstance(payload_for_n, dict) else None
            slope_recs = list(slope_list) if isinstance(slope_list, list) else []

            p_list: list[float] = []
            v_list: list[float] = []
            for rec_idx, rec in enumerate(recs):
                if not isinstance(rec, (list, tuple)) or len(rec) != 2:
                    continue
                p_val = float(rec[0])
                if not np.isfinite(p_val):
                    continue
                slope_rec = slope_recs[rec_idx] if rec_idx < len(slope_recs) and isinstance(slope_recs[rec_idx], dict) else None
                v = _tie_heatmap_value_at_record(
                    n=int(n),
                    rec=rec,
                    slope_rec=slope_rec,
                    value_key=val_key,
                )
                if not np.isfinite(v):
                    continue
                p_list.append(p_val)
                v_list.append(float(v))
            if not p_list:
                continue
            row_vals = _nearest_values_by_p_grid(
                np.asarray(p_list, dtype=float),
                np.asarray(v_list, dtype=float),
                x_vals,
            )
            heat[rr, :] = row_vals

    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib import image as mimage

    plot_data, vmin, vmax = _normalize_heat_for_render(
        heat,
        trim_color_range_percent=int(cfg.trim_color_range_percent),
        per_n_color_range=bool(cfg.per_n_color_range),
    )
    masked = np.ma.masked_invalid(plot_data)
    cmap = plt.get_cmap(cfg.colormap).copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    if pixel_mode == "exact":
        norm = None if (vmin is None or vmax is None) else colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba = cmap(norm(masked) if norm is not None else masked)
        mimage.imsave(cfg.output_path, rgba, format="png")
        if verbose:
            print(f"Wrote {os.path.abspath(cfg.output_path)} (PNG).")
            print(f"Export time: {time.monotonic() - t0:.2f}s")
        return

    fig, ax = plt.subplots(figsize=(float(cfg.width_in), float(cfg.height_in)), dpi=int(cfg.dpi))
    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[float(x_vals[0]), float(x_vals[-1]), float(cfg.n_min), float(cfg.n_max)],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("p")
    ax.set_ylabel("n")
    if val_key == "ev_n":
        title_label = "E_sorted/n"
    elif val_key == "eslope_n":
        title_label = "(d/dp E_sorted)/n"
    elif val_key == "d":
        title_label = "d (r-l) via nearest tie"
    elif val_key == "e":
        title_label = "e (l-r) via nearest tie"
    else:
        title_label = f"{val_key} via nearest tie"
    range_label = "per-N" if bool(cfg.per_n_color_range) else "global"
    trim_pct = int(cfg.trim_color_range_percent)
    trim_label = f"trim {trim_pct}-{100 - trim_pct}%" if trim_pct > 0 else "full range"
    ax.set_title(f"N-p heatmap: {title_label} (p={p_min:.4f}..{p_max:.4f}; {range_label}; {trim_label})")
    if bool(cfg.show_legend):
        cbar = fig.colorbar(im, ax=ax)
        if bool(cfg.per_n_color_range):
            cbar.set_label(f"{title_label} (row-normalized)")
        else:
            cbar.set_label(title_label)
    fig.tight_layout()
    fig.savefig(cfg.output_path, format="png")
    plt.close(fig)

    if verbose:
        print(f"Wrote {os.path.abspath(cfg.output_path)} (PNG).")
        print(f"Export time: {time.monotonic() - t0:.2f}s")


def export_tie_heatmap_headless(cfg: TieHeatmapExportConfig, *, verbose: bool = True) -> None:
    """Export N-tie heatmap: x=tie index position, y=n, color=tie-derived scalar."""
    t0 = time.monotonic()
    if cfg.n_min > cfg.n_max:
        print("ERROR: n_min must be <= n_max.", file=sys.stderr)
        sys.exit(1)
    if float(cfg.width_in) <= 0.0 or float(cfg.height_in) <= 0.0:
        print("ERROR: width_in and height_in must be > 0.", file=sys.stderr)
        sys.exit(1)
    if int(cfg.dpi) <= 0:
        print("ERROR: dpi must be > 0.", file=sys.stderr)
        sys.exit(1)

    fmt = cfg.export_format.strip().lower()
    if fmt != "png":
        print("ERROR: tie heatmap export_format must be png.", file=sys.stderr)
        sys.exit(1)
    pixel_mode = str(cfg.pixel_mode).strip().lower()
    if pixel_mode not in HEATMAP_PIXEL_MODE_CHOICES:
        print(
            "ERROR: tie heatmap pixel_mode must be one of "
            + ", ".join(HEATMAP_PIXEL_MODE_CHOICES)
            + f"; got {cfg.pixel_mode!r}.",
            file=sys.stderr,
        )
        sys.exit(1)
    value_key = str(cfg.value_key).strip().lower()
    if value_key not in TIE_HEATMAP_VALUE_CHOICES:
        print(
            "ERROR: tie heatmap value_key must be one of "
            + ", ".join(TIE_HEATMAP_VALUE_CHOICES)
            + f"; got {cfg.value_key!r}.",
            file=sys.stderr,
        )
        sys.exit(1)
    load_from = str(cfg.load_from).strip().lower()
    if load_from not in ("l", "r"):
        print('ERROR: load_from must be "l" or "r".', file=sys.stderr)
        sys.exit(1)

    from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, iter_tie_points_from_shards

    n_vals = list(range(cfg.n_min, cfg.n_max + 1))
    row_by_n = {n: i for i, n in enumerate(n_vals)}
    max_ties = 1000
    heat = np.full((len(n_vals), max_ties), np.nan, dtype=float)

    tie_manifest = cfg.tie_manifest or DEFAULT_TIE_OUTPUT
    if not os.path.isfile(tie_manifest):
        print(f"ERROR: tie shard manifest not found: {os.path.abspath(tie_manifest)}", file=sys.stderr)
        sys.exit(1)
    n_rows = iter_tie_points_from_shards(
        path=tie_manifest,
        n_list=n_vals,
        require_all=False,
        progress=cfg.progress_every,
        include_float_by_n=False,
        include_float_with_pairs_by_n=True,
        include_tie_slope_by_n=True,
    )
    for n, payload_for_n in n_rows:
        if n not in row_by_n:
            continue
        recs = payload_for_n.get("float_with_pairs_by_n")
        if not isinstance(recs, list) or not recs:
            continue
        slope_list = payload_for_n.get("tie_slope_by_n")
        slope_recs = list(slope_list) if isinstance(slope_list, list) else []

        rr = row_by_n[n]
        if load_from == "l":
            # Left mode must match HTML variant 5 semantics:
            # native tie index t maps to rec_idx = center_idx + t, with t in 1..1000.
            center_idx = _canonical_center_index_for_recs(int(n), recs)
            if center_idx is None:
                raise ValueError(f"n={n}: missing canonical center tie point in float_with_pairs_by_n")
            m_nonneg = len(recs) - int(center_idx)
            n_take = min(max_ties, max(0, m_nonneg - 1))
        else:
            n_take = min(max_ties, len(recs))
        for pos in range(n_take):
            if load_from == "l":
                rec_idx = int(center_idx) + (pos + 1)
            else:
                rec_idx = len(recs) - 1 - pos
            rec = recs[rec_idx]
            slope_rec = slope_recs[rec_idx] if rec_idx < len(slope_recs) else None
            heat[rr, pos] = _tie_heatmap_value_at_record(
                n=n,
                rec=rec,
                slope_rec=slope_rec if isinstance(slope_rec, dict) else None,
                value_key=value_key,
            )

    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib import image as mimage

    plot_data, vmin, vmax = _normalize_heat_for_render(
        heat,
        trim_color_range_percent=int(cfg.trim_color_range_percent),
        per_n_color_range=bool(cfg.per_n_color_range),
    )
    masked = np.ma.masked_invalid(plot_data)
    cmap = plt.get_cmap(cfg.colormap).copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    if pixel_mode == "exact":
        norm = None if (vmin is None or vmax is None) else colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        rgba = cmap(norm(masked) if norm is not None else masked)
        mimage.imsave(cfg.output_path, rgba, format="png")
        if verbose:
            print(f"Wrote {os.path.abspath(cfg.output_path)} (PNG).")
            print(f"Export time: {time.monotonic() - t0:.2f}s")
        return

    if load_from == "l":
        x_lo, x_hi = 1.0, float(max_ties)
        x_label = "tie # from center (1..1000; center tie 0 excluded)"
    else:
        x_lo, x_hi = 1.0, float(max_ties)
        x_label = "tie # from end (1 = last)"

    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=[x_lo, x_hi, float(cfg.n_min), float(cfg.n_max)],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("n")
    range_label = "per-N" if bool(cfg.per_n_color_range) else "global"
    trim_pct = int(cfg.trim_color_range_percent)
    trim_label = f"trim {trim_pct}-{100 - trim_pct}%" if trim_pct > 0 else "full range"
    ax.set_title(
        f"N-tie heatmap: {value_key} ({'left' if load_from == 'l' else 'right'} load; {range_label}; {trim_label})"
    )
    if bool(cfg.show_legend):
        cbar = fig.colorbar(im, ax=ax)
        if bool(cfg.per_n_color_range):
            cbar.set_label(f"{value_key} (row-normalized)")
        else:
            cbar.set_label(value_key)
    fig.tight_layout()
    fig.savefig(cfg.output_path, format="png")
    plt.close(fig)

    if verbose:
        print(f"Wrote {os.path.abspath(cfg.output_path)} (PNG).")
        print(f"Export time: {time.monotonic() - t0:.2f}s")


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
