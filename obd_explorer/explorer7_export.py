"""Generate HTML explorer #7: nearest tie values graph explorer."""

from __future__ import annotations

import base64
import os

import numpy as np

from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, iter_tie_tables, tie_center_index
from obd_explorer.explorer7_html import build_explorer7_html


_TIE_FIELDS = ("i", "j", "l", "r", "d")


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

    columns = {"i": "i", "j": "j", "l": "slope_left", "r": "slope_right", "d": "log10_D"}
    n_rows = iter_tie_tables(
        man,
        n_list=n_vals,
        columns=("p",) + tuple(columns.values()),
        require_all=False,
        progress=(10 if progress else None),
    )
    for n, table in n_rows:
        if table["p"].size == 0:
            continue
        # Variant 7 rule: skip the center tie p = 1/2 (native tie index 0).
        keep = np.ones(table["p"].size, dtype=bool)
        keep[tie_center_index(table)] = False
        p_src = table["p"][keep].astype(float)
        val_src = {field: table[col][keep].astype(float) for field, col in columns.items()}

        if not p_src.size:
            continue
        p_arr = p_src
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

