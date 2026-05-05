"""Tie upper bounds for explorer2 PCA (shard / pickle dict shapes, analytic fallback)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import comb


def _all_tie_points(n: int, tol: float = 1e-10) -> np.ndarray:
    out: list[float] = []
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            ratio = comb(n, j, exact=False) / comb(n, i, exact=False)
            if ratio <= 0 or not np.isfinite(ratio):
                continue
            exp = 1.0 / (j - i)
            p = 1.0 / (1.0 + ratio**exp)
            if 0 < p < 1 and np.isfinite(p):
                out.append(float(p))
    arr = np.sort(np.array(out, dtype=float))
    if len(arr) > 1:
        keep = np.concatenate([[True], np.diff(arr) > tol])
        arr = arr[keep]
    return arr


def last_tie_above_half(n: int) -> float:
    arr = _all_tie_points(n, tol=1e-10)
    above_half = arr[arr > 0.5]
    if len(above_half) == 0:
        return 1.0 - 1e-6
    return float(above_half[-1])


def last_tie_from_pair_records(recs: list) -> float:
    """Largest tie ``p`` in ``(0.5, 1)`` from ``float_with_pairs_by_n`` records."""
    best: float | None = None
    for item in recs:
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            continue
        try:
            p = float(item[0])
        except (TypeError, ValueError):
            continue
        if 0.5 < p < 1:
            if best is None or p > best:
                best = p
    return float(best) if best is not None else 1.0 - 1e-6


def last_tie_by_n_from_payload(tie_payload: dict[str, Any], n_vals: list[int]) -> dict[int, float]:
    """Per-n last tie in ``(0.5, 1)``, matching legacy explorer2 tie loading."""
    float_with_pairs_by_n = tie_payload.get("float_with_pairs_by_n") or {}
    float_by_n = tie_payload.get("float_by_n") or {}
    out: dict[int, float] = {}
    for n in n_vals:
        recs = float_with_pairs_by_n.get(n)
        if recs is None:
            recs = float_with_pairs_by_n.get(str(n))
        if recs is not None:
            out[n] = last_tie_from_pair_records(recs)
            continue
        raw = float_by_n.get(n)
        if raw is None:
            raw = float_by_n.get(str(n))
        if raw is not None:
            try:
                pts = [float(x) for x in raw]
            except (TypeError, ValueError):
                pts = [float(raw)]
            above = sorted(round(p, 6) for p in pts if 0.5 < p < 1)
            out[n] = float(above[-1]) if above else 1.0 - 1e-6
            continue
        out[n] = last_tie_above_half(n)
    return out
