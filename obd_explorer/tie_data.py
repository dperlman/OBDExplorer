"""Tie-point draw entries for bands (from the per-n tie tables)."""

from __future__ import annotations

import os

import numpy as np

# (p, i, j, slope_left, slope_right, log10_D); values may be None if missing.
# log10_D is log10 of the slope jump D = slope_right - slope_left, stored exactly in the tie tables:
# never recompute it as the difference of the two slopes, which loses it to cancellation.
TieDrawEntry = tuple[float, int | None, int | None, float | None, float | None, float | None]

_DRAW_COLUMNS = ("p", "i", "j", "slope_left", "slope_right", "log10_D")


def _finite_or_none(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def tie_draw_entries_from_table(
    table: dict[str, np.ndarray], tie_p_min: float, tie_p_max: float
) -> list[TieDrawEntry]:
    """Draw entries for one n's tie table, restricted to ``tie_p_min <= p <= tie_p_max``."""
    p = table["p"]
    keep = np.flatnonzero((p >= tie_p_min) & (p <= tie_p_max))
    return [
        (
            round(float(p[k]), 6),
            int(table["i"][k]),
            int(table["j"][k]),
            _finite_or_none(table["slope_left"][k]),
            _finite_or_none(table["slope_right"][k]),
            _finite_or_none(table["log10_D"][k]),
        )
        for k in keep.tolist()
    ]


def resolve_tie_draw_entries(
    *,
    n_vals: list[int],
    tie_p_min: float,
    tie_p_max: float,
    tie_manifest_path: str | None = None,
) -> dict[int, list[TieDrawEntry]]:
    """Load tie segments from the tie tables (``OBDsaveSourceData`` layout)."""
    from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, iter_tie_tables

    man = tie_manifest_path or DEFAULT_TIE_OUTPUT
    if not os.path.isfile(man):
        return {}
    out: dict[int, list[TieDrawEntry]] = {}
    for n, table in iter_tie_tables(man, n_list=n_vals, columns=_DRAW_COLUMNS, require_all=False):
        segs = tie_draw_entries_from_table(table, tie_p_min, tie_p_max)
        if segs:
            out[n] = segs
    return out
