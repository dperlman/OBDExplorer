"""Generate HTML explorer #7: nearest tie values graph explorer."""

from __future__ import annotations

import base64
import os

import numpy as np

from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, iter_tie_points_from_shards
from obd_explorer.explorer7_html import build_explorer7_html


_TIE_FIELDS = ("i", "j", "l", "r", "d", "e")


def _pack_float32_base64(values: np.ndarray) -> str:
    if values.size == 0:
        return ""
    arr = np.asarray(values, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _nearest_values_by_p_grid(p_source: np.ndarray, v_source: np.ndarray, p_target: np.ndarray) -> np.ndarray:
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


def _tie_proxy_rows_by_n(
    *,
    n_vals: list[int],
    p_values: np.ndarray,
    tie_manifest: str | None,
    progress: bool = False,
) -> dict[str, dict[str, str]]:
    by_field: dict[str, dict[str, str]] = {k: {} for k in _TIE_FIELDS}
    man = tie_manifest or DEFAULT_TIE_OUTPUT
    if not os.path.isfile(man):
        return by_field

    n_rows = iter_tie_points_from_shards(
        path=man,
        n_list=n_vals,
        require_all=False,
        progress=(10 if progress else None),
        include_float_by_n=False,
        include_float_with_pairs_by_n=True,
        include_tie_slope_by_n=True,
    )
    for n, payload in n_rows:
        recs = payload.get("float_with_pairs_by_n") if isinstance(payload, dict) else None
        if not isinstance(recs, list) or not recs:
            continue
        slope_list = payload.get("tie_slope_by_n") if isinstance(payload, dict) else None
        slopes = list(slope_list) if isinstance(slope_list, list) else []

        p_src: list[float] = []
        val_src: dict[str, list[float]] = {k: [] for k in _TIE_FIELDS}
        for rec_idx, rec in enumerate(recs):
            if not isinstance(rec, (list, tuple)) or len(rec) != 2:
                continue
            p = float(rec[0])
            if not np.isfinite(p):
                continue
            pairs = rec[1] if isinstance(rec[1], list) else []
            i_val = float("nan")
            j_val = float("nan")
            if pairs:
                first = pairs[0]
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    i_val = float(first[0])
                    j_val = float(first[1])

            sl = float("nan")
            sr = float("nan")
            if rec_idx < len(slopes) and isinstance(slopes[rec_idx], dict):
                s = slopes[rec_idx]
                raw_sl = s.get("slope_left")
                raw_sr = s.get("slope_right")
                if raw_sl is not None and np.isfinite(float(raw_sl)):
                    sl = float(raw_sl)
                if raw_sr is not None and np.isfinite(float(raw_sr)):
                    sr = float(raw_sr)
            d_val = (sr - sl) if np.isfinite(sr) and np.isfinite(sl) else float("nan")
            e_val = (sl - sr) if np.isfinite(sr) and np.isfinite(sl) else float("nan")

            p_src.append(p)
            val_src["i"].append(i_val)
            val_src["j"].append(j_val)
            val_src["l"].append(sl)
            val_src["r"].append(sr)
            val_src["d"].append(d_val)
            val_src["e"].append(e_val)

        if not p_src:
            continue
        p_arr = np.asarray(p_src, dtype=float)
        for field in _TIE_FIELDS:
            v_arr = np.asarray(val_src[field], dtype=float)
            finite = np.isfinite(v_arr)
            if not np.any(finite):
                continue
            row = _nearest_values_by_p_grid(p_arr[finite], v_arr[finite], p_values)
            by_field[field][str(int(n))] = _pack_float32_base64(row)
    return by_field


def write_explorer7_html(
    output_path: str,
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
    p_min: float = 0.5,
    p_max: float = 0.6,
    tie_manifest: str | None = None,
    colorscale: str = "viridis",
    verbose: bool = True,
    progress: bool = False,
) -> None:
    if p_steps < 2:
        raise ValueError("p_steps must be at least 2")
    p_lo = float(p_min)
    p_hi = float(p_max)
    if not np.isfinite(p_lo) or not np.isfinite(p_hi):
        raise ValueError("p_min and p_max must be finite floats.")
    if p_lo > p_hi:
        raise ValueError("p_min must be <= p_max.")
    p_values = np.linspace(p_lo, p_hi, int(p_steps), dtype=float)
    n_vals = list(range(n_min, n_max + 1))
    tie_proxy = _tie_proxy_rows_by_n(
        n_vals=n_vals,
        p_values=p_values,
        tie_manifest=tie_manifest,
        progress=progress,
    )
    html = build_explorer7_html(
        tie_proxy_by_field_packed=tie_proxy,
        n_min=n_min,
        n_max=n_max,
        p_steps=p_steps,
        p_values=[float(x) for x in p_values],
        colorscale=colorscale,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Wrote {output_path}.")

