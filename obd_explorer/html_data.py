"""Materialize graph/tie payloads for embedded HTML (explorer1-style)."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np

from OBDsaveSourceData import _is_canonical_center_tie

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


def tie_ps_above_half_from_pair_records(recs: list) -> list[float]:
    if not recs:
        return []
    pts = [float(p) for p, _ in recs]
    return sorted([round(p, 6) for p in pts if 0.5 < p < 1])


def tie_points_by_n_for_explorer1(
    tie_payload: dict[str, Any],
    n_min: int,
    n_max: int,
    *,
    progress: bool = False,
) -> dict[str, list[float]]:
    """``TIE_POINTS_BY_N`` for HTML: string keys -> tie p list in (0.5, 1)."""
    float_with_pairs_by_n = tie_payload.get("float_with_pairs_by_n") or {}
    float_by_n = tie_payload.get("float_by_n") or {}
    out: dict[str, list[float]] = {}
    t0 = time.perf_counter()
    if progress:
        print(
            f"[html] tie points for hairlines: n in [{n_min}, {n_max}]",
            file=sys.stderr,
        )
    for n in range(n_min, n_max + 1):
        try:
            recs = float_with_pairs_by_n.get(n)
            if recs is None and isinstance(float_with_pairs_by_n, dict):
                recs = float_with_pairs_by_n.get(str(n))
            if recs is not None:
                above_half = tie_ps_above_half_from_pair_records(recs)
            else:
                raw = float_by_n.get(n)
                if raw is None and isinstance(float_by_n, dict):
                    raw = float_by_n.get(str(n))
                if raw is None:
                    continue
                try:
                    pts = [float(x) for x in raw]
                except (TypeError, ValueError):
                    pts = [float(raw)]
                above_half = sorted([round(p, 6) for p in pts if 0.5 < p < 1])
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


def tie_explorer5_series_by_n(
    tie_payload: dict[str, Any],
    n_min: int,
    n_max: int,
    *,
    progress: bool = False,
) -> dict[str, dict[str, Any]]:
    """Per-n tie arrays for HTML explorer variants 5 and 6.

    Native tie index for variants 5/6 is defined on the **non-negative side only**:
    index ``0`` is the canonical center tie point, and index ``t`` maps to record
    ``rec_idx = center_idx + t``.

    Let ``m_nonneg = len(recs) - center_idx`` (ties from center through last tie).
    Select native indices by union of:

    - **Forward arm:** ``0 .. EXPLORER5_CENTER_ARM_LENGTH-1`` (clipped by ``m_nonneg``)
    - **Backward arm:** the last ``EXPLORER5_TAIL_ARM_LENGTH`` indices in ``0..m_nonneg-1``

    Union is sorted ascending by native index. Embedded row ``0`` is native index ``0``
    (center tie), and embedded last row is native index ``m_nonneg-1`` (last tie).

    Slopes align with ``tie_slope_by_n[rec_idx]`` (record order in the shard).

    If no canonical center tie row is found, this is treated as invalid data and raises ``ValueError``.
    """
    float_with_pairs_by_n = tie_payload.get("float_with_pairs_by_n") or {}
    slope_by_n = tie_payload.get("tie_slope_by_n") or {}
    out: dict[str, dict[str, Any]] = {}
    t0 = time.perf_counter()
    if progress:
        print(
            f"[html] tie explorer embed: n in [{n_min}, {n_max}]",
            file=sys.stderr,
        )
    for n in range(n_min, n_max + 1):
        try:
            recs = float_with_pairs_by_n.get(n)
            if recs is None and isinstance(float_with_pairs_by_n, dict):
                recs = float_with_pairs_by_n.get(str(n))
            if not recs:
                continue
            slope_recs = slope_by_n.get(n)
            if slope_recs is None and isinstance(slope_by_n, dict):
                slope_recs = slope_by_n.get(str(n))
            slope_list: list[dict[str, Any]] = list(slope_recs) if isinstance(slope_recs, list) else []

            m = len(recs)
            if m == 0:
                continue

            center_idx: int | None = None
            tie_ps: list[float] = [float("nan")] * m
            for rec_idx, item in enumerate(recs):
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                p_raw = float(item[0])
                tie_ps[rec_idx] = p_raw
                pairs = item[1]
                plist = list(pairs) if pairs else []
                if _is_canonical_center_tie(int(n), plist):
                    center_idx = rec_idx
                    break

            if center_idx is None:
                raise ValueError(f"n={n}: missing canonical center tie point in float_with_pairs_by_n")

            m_nonneg = m - center_idx
            if m_nonneg <= 0:
                raise ValueError(f"n={n}: invalid center index {center_idx} for {m} tie rows")
            forward_native: set[int] = set()
            for t in range(EXPLORER5_CENTER_ARM_LENGTH):
                ri = center_idx + t
                if 0 <= t < m_nonneg and 0 <= ri < m:
                    forward_native.add(t)

            backward_native: set[int] = set()
            tail_start = max(0, m_nonneg - EXPLORER5_TAIL_ARM_LENGTH)
            for t in range(tail_start, m_nonneg):
                ri = center_idx + t
                if 0 <= ri < m:
                    backward_native.add(t)

            selected_native = sorted(forward_native | backward_native)
            if not selected_native:
                continue

            ps: list[float] = []
            iv: list[int | None] = []
            jv: list[int | None] = []
            lv: list[float | None] = []
            rv: list[float | None] = []
            ev_ns: list[float | None] = []
            for t in selected_native:
                rec_idx = center_idx + t
                if not (0 <= rec_idx < m):
                    continue
                item = recs[rec_idx]
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                _, pairs = item[0], item[1]
                pf = float(item[0])
                pi, pj = None, None
                plist = pairs or []
                if plist:
                    first = plist[0]
                    if isinstance(first, (list, tuple)) and len(first) == 2:
                        pi, pj = int(first[0]), int(first[1])
                sl: float | None = None
                sr: float | None = None
                evn: float | None = None
                if rec_idx < len(slope_list) and isinstance(slope_list[rec_idx], dict):
                    sd = slope_list[rec_idx]
                    raw_sl = sd.get("slope_left")
                    raw_sr = sd.get("slope_right")
                    if raw_sl is not None and np.isfinite(float(raw_sl)):
                        sl = float(raw_sl)
                    if raw_sr is not None and np.isfinite(float(raw_sr)):
                        sr = float(raw_sr)
                    raw_es = sd.get("expected_sorted")
                    if raw_es is not None and np.isfinite(float(raw_es)):
                        evn = float(raw_es) / float(n)
                ps.append(round(pf, 6))
                iv.append(pi)
                jv.append(pj)
                lv.append(sl)
                rv.append(sr)
                ev_ns.append(evn)

            dv: list[float | None] = []
            ev: list[float | None] = []
            for a, b in zip(lv, rv):
                if a is not None and b is not None:
                    dv.append(float(b - a))
                    ev.append(float(a - b))
                else:
                    dv.append(None)
                    ev.append(None)

            out[str(n)] = {
                "p": ps,
                "i": iv,
                "j": jv,
                "l": lv,
                "r": rv,
                "d": dv,
                "e": ev,
                "ev_n": ev_ns,
            }
        finally:
            k = n - n_min + 1
            last_rows = len(out[str(n)]["p"]) if str(n) in out else 0
            _html_verbose_n_tick(
                tag="tie explorer embed",
                n=n,
                n_min=n_min,
                n_max=n_max,
                step_index=k,
                t0=t0,
                verbose=progress,
                extra=f"stored_keys={len(out)} last_rows={last_rows}",
            )
    if progress:
        elapsed = time.perf_counter() - t0
        print(
            f"[html] tie explorer embed: done stored_n={len(out)} in {elapsed:.2f}s",
            file=sys.stderr,
        )
    return out
