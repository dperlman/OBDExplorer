"""Tie upper bounds for explorer2 PCA (from the tie tables, analytic fallback)."""

from __future__ import annotations

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


def last_tie_from_table(table: dict[str, np.ndarray]) -> float:
    """Largest tie ``p`` in ``(0.5, 1)`` of one n's tie table."""
    p = table["p"]
    above = p[(p > 0.5) & (p < 1)]
    return float(above.max()) if above.size else 1.0 - 1e-6


def last_tie_by_n_from_tables(tie_tables: dict[int, dict[str, np.ndarray]], n_vals: list[int]) -> dict[int, float]:
    """Per-n last tie in ``(0.5, 1)``; analytic fallback for any n without a tie table."""
    return {
        n: (last_tie_from_table(tie_tables[n]) if n in tie_tables else last_tie_above_half(n))
        for n in n_vals
    }
