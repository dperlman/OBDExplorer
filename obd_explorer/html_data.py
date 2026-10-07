"""Materialize graph/tie payloads for embedded HTML (explorer1-style)."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np

from OBDsaveSourceData import tie_center_index

from obd_explorer.grid import BinomialGrid


def _html_verbose_n_tick(
    *,
    tag: str,
    n: int,
    n_min: int,
    n_max: int,
    step_index: int,
    t0: float,
    verbose: bool,
    extra: str = "",
) -> None:
    """Print timing/status every 10 completed ``n`` steps (and on the last ``n``)."""
    if not verbose:
        return
    total = n_max - n_min + 1
    if step_index % 10 != 0 and n != n_max:
        return
    elapsed = time.perf_counter() - t0
    suf = f" {extra}" if extra else ""
    print(
        f"[html] {tag}: n={n} step {step_index}/{total} elapsed {elapsed:.2f}s{suf}",
        file=sys.stderr,
    )


def materialize_binomial_series_for_js(
    grid: BinomialGrid,
    *,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """Flat list of ``{x, y, perm}`` per (n, p) in row-major order (matches legacy pickle)."""
    t0 = time.perf_counter()
    out: list[dict[str, Any]] = []
    if grid.rows_by_n is not None:
        n_lo, n_hi = grid.n_min, grid.n_max
        if progress:
            print(
                f"[html] binomial series: n in [{n_lo}, {n_hi}] ({n_hi - n_lo + 1} values)",
                file=sys.stderr,
            )
        for n in range(n_lo, n_hi + 1):
            rows = grid.rows_by_n[n]
            y = np.asarray(rows["y"])
            perm = np.asarray(rows["perm"])
            for p_ix in range(grid.p_steps):
                yy = y[p_ix].astype(float).tolist()
                pp = perm[p_ix].astype(int).tolist()
                xv = list(range(n + 1))
                out.append({"x": xv, "y": yy, "perm": pp})
            k = n - n_lo + 1
            rows_so_far = len(out)
            _html_verbose_n_tick(
                tag="binomial series",
                n=n,
                n_min=n_lo,
                n_max=n_hi,
                step_index=k,
                t0=t0,
                verbose=progress,
                extra=f"curve_blocks={rows_so_far}",
            )
        if progress:
            elapsed = time.perf_counter() - t0
            print(
                f"[html] binomial series: done curve_blocks={len(out)} in {elapsed:.2f}s",
                file=sys.stderr,
            )
        return out

    assert grid.binomial_flat is not None
    if progress:
        elapsed = time.perf_counter() - t0
        print(
            f"[html] binomial series: using pre-materialized flat len={len(grid.binomial_flat)} "
            f"({elapsed*1000:.1f} ms)",
            file=sys.stderr,
        )
    return grid.binomial_flat


def tie_ps_above_half(table: dict[str, np.ndarray]) -> list[float]:
    """Tie ``p`` in ``(0.5, 1)`` of one n's tie table, rounded to 6 decimals, ascending."""
    return sorted(round(p, 6) for p in table["p"].tolist() if 0.5 < p < 1)


def tie_points_by_n_for_explorer1(
    tie_tables: dict[int, dict[str, np.ndarray]],
    n_min: int,
    n_max: int,
    *,
    progress: bool = False,
) -> dict[str, list[float]]:
    """``TIE_POINTS_BY_N`` for HTML: string keys -> tie p list in (0.5, 1)."""
    out: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    if progress:
        print(
            f"[html] tie points for hairlines: n in [{n_min}, {n_max}]",
            file=sys.stderr,
        )
    for n in range(n_min, n_max + 1):
        try:
            table = tie_tables.get(n)
            if table is None:
                continue
            above_half = tie_ps_above_half(table)
            if above_half:
                out[str(n)] = above_half
        finally:
            k = n - n_min + 1
            _html_verbose_n_tick(
                tag="tie hairlines",
                n=n,
                n_min=n_min,
                n_max=n_max,
                step_index=k,
                t0=t0,
                verbose=progress,
                extra=f"n_with_ties={len(out)}",
            )
    if progress:
        elapsed = time.perf_counter() - t0
        print(
            f"[html] tie hairlines: done n_with_ties={len(out)} in {elapsed:.2f}s",
            file=sys.stderr,
        )
    return out


def json_dumps_p_labels(p_values: tuple[float, ...]) -> str:
    return json.dumps([round(float(p), 4) for p in p_values])


# Variant 5/6 embedded tie rows: union of (a) up to 1000 ties from center outward along valid_rows,
# and (b) up to 1000 ties from the last tie backward; duplicates removed; sorted by valid_rows index.
# At most 2000 rows ⇒ embedded indices 0..1999.
EXPLORER5_CENTER_ARM_LENGTH = 1000
EXPLORER5_TAIL_ARM_LENGTH = 1000
EXPLORER5_EMBEDDED_ROW_COUNT = EXPLORER5_CENTER_ARM_LENGTH + EXPLORER5_TAIL_ARM_LENGTH
EXPLORER5_MAX_TIE_INDEX = EXPLORER5_EMBEDDED_ROW_COUNT - 1


TIE_EXPLORER5_COLUMNS = ("p", "i", "j", "slope_left", "slope_right", "log10_D", "E")


def _tie_explorer5_series_row_for_n(n: int, table: dict[str, np.ndarray]) -> dict[str, Any] | None:
    """Build one variant 5/6 embedded row payload for a single ``n`` from its tie table."""
    m = int(table["p"].size)
    if m == 0:
        return None
    center_idx = tie_center_index(table)
    m_nonneg = m - center_idx

    forward_native = set(range(min(EXPLORER5_CENTER_ARM_LENGTH, m_nonneg)))
    backward_native = set(range(max(0, m_nonneg - EXPLORER5_TAIL_ARM_LENGTH), m_nonneg))
    rows = [center_idx + t for t in sorted(forward_native | backward_native)]

    def _vals(name: str) -> list[float | None]:
        col = table[name]
        return [float(v) if np.isfinite(v) else None for v in col[rows].tolist()]

    ev = table["E"][rows]
    return {
        "p": [round(p, 6) for p in table["p"][rows].tolist()],
        "i": [int(v) for v in table["i"][rows].tolist()],
        "j": [int(v) for v in table["j"][rows].tolist()],
        "l": _vals("slope_left"),
        "r": _vals("slope_right"),
        "d": _vals("log10_D"),
        "ev_n": [float(v) / float(n) if np.isfinite(v) else None for v in ev.tolist()],
    }


def tie_explorer5_series_by_n(
    tie_tables: dict[int, dict[str, np.ndarray]],
    n_min: int,
    n_max: int,
    *,
    progress: bool = False,
) -> dict[str, dict[str, Any]]:
    """Per-n tie arrays for HTML explorer variants 5 and 6.

    Native tie index for variants 5/6 is defined on the **non-negative side only**:
    index ``0`` is the center tie point ``p = 1/2``, and index ``t`` maps to row
    ``center_idx + t`` of the n's tie table.

    Let ``m_nonneg = rows - center_idx`` (ties from center through last tie).
    Select native indices by union of:

    - **Forward arm:** ``0 .. EXPLORER5_CENTER_ARM_LENGTH-1`` (clipped by ``m_nonneg``)
    - **Backward arm:** the last ``EXPLORER5_TAIL_ARM_LENGTH`` indices in ``0..m_nonneg-1``

    Union is sorted ascending by native index. Embedded row ``0`` is native index ``0``
    (center tie), and embedded last row is native index ``m_nonneg-1`` (last tie).
    """
    return tie_explorer5_series_by_n_stream(
        ((n, tie_tables[n]) for n in range(n_min, n_max + 1) if n in tie_tables),
        n_min,
        n_max,
        progress=progress,
    )


def tie_explorer5_series_by_n_stream(
    n_rows: Any,
    n_min: int,
    n_max: int,
    *,
    progress: bool = False,
) -> dict[str, dict[str, Any]]:
    """Per-n tie arrays for variants 5/6 from streamed ``(n, tie_table)`` pairs."""
    out: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()
    total = max(0, int(n_max) - int(n_min) + 1)
    if progress:
        print(
            f"[html] tie explorer embed(stream): n in [{n_min}, {n_max}]",
            file=sys.stderr,
        )

    processed = 0
    for n, table in n_rows:
        n_int = int(n)
        if n_int < n_min or n_int > n_max:
            continue
        try:
            row = _tie_explorer5_series_row_for_n(n_int, table)
            if row is not None:
                out[str(n_int)] = row
        finally:
            processed += 1
            _html_verbose_n_tick(
                tag="tie explorer embed(stream)",
                n=n_int,
                n_min=n_min,
                n_max=n_max,
                step_index=processed,
                t0=t0,
                verbose=progress,
                extra=f"stored_keys={len(out)} last_rows={len(out[str(n_int)]['p']) if str(n_int) in out else 0}",
            )

    if progress:
        elapsed = time.perf_counter() - t0
        print(
            f"[html] tie explorer embed(stream): done stored_n={len(out)} processed={processed}/{total} in {elapsed:.2f}s",
            file=sys.stderr,
        )
    return out
