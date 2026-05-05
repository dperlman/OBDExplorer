"""Tie-point draw entries for bands (from tie-shard manifest bundle)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# (p, i, j, slope_left, slope_right); slopes may be None if missing from payload.
TieDrawEntry = tuple[float, int | None, int | None, float | None, float | None]


def load_tie_draw_entries_from_payload(
    data: dict[str, Any],
    n_vals: list[int],
    tie_p_min: float,
    tie_p_max: float,
) -> dict[int, list[TieDrawEntry]]:
    """Parse ``float_with_pairs_by_n`` / ``float_by_n`` plus optional ``tie_slope_by_n``."""
    fwp = data.get("float_with_pairs_by_n") or data.get("by_n") or {}
    float_by_n = data.get("float_by_n", {})
    slope_by_n = data.get("tie_slope_by_n") or {}
    out: dict[int, list[TieDrawEntry]] = {}
    for n in n_vals:
        segs: list[TieDrawEntry] = []
        recs = fwp.get(n)
        if recs is None and isinstance(fwp, dict):
            recs = fwp.get(str(n))
        slope_recs = slope_by_n.get(n)
        if slope_recs is None and isinstance(slope_by_n, dict):
            slope_recs = slope_by_n.get(str(n))
        slope_list: list[dict[str, Any]] = list(slope_recs) if isinstance(slope_recs, list) else []

        if recs is not None:
            for rec_idx, item in enumerate(recs):
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                p_raw, pairs = item[0], item[1]
                pf = float(p_raw)
                if not (tie_p_min <= pf <= tie_p_max):
                    continue
                pr = round(pf, 6)
                slope_sl: float | None = None
                slope_sr: float | None = None
                if rec_idx < len(slope_list) and isinstance(slope_list[rec_idx], dict):
                    sd = slope_list[rec_idx]
                    sl = sd.get("slope_left")
                    sr = sd.get("slope_right")
                    if sl is not None and np.isfinite(float(sl)):
                        slope_sl = float(sl)
                    if sr is not None and np.isfinite(float(sr)):
                        slope_sr = float(sr)
                plist = pairs or []
                if not plist:
                    segs.append((pr, None, None, slope_sl, slope_sr))
                    continue
                for ij in plist:
                    if not isinstance(ij, (list, tuple)) or len(ij) != 2:
                        continue
                    segs.append((pr, int(ij[0]), int(ij[1]), slope_sl, slope_sr))
        else:
            raw = float_by_n.get(n)
            if raw is None and isinstance(float_by_n, dict):
                raw = float_by_n.get(str(n))
            if raw is None:
                continue
            arr = np.atleast_1d(np.asarray(raw, dtype=float))
            for pf in arr.flat:
                if not np.isfinite(pf) or not (tie_p_min <= float(pf) <= tie_p_max):
                    continue
                segs.append((round(float(pf), 6), None, None, None, None))
        if segs:
            out[n] = segs
    return out


def resolve_tie_draw_entries(
    *,
    n_vals: list[int],
    tie_p_min: float,
    tie_p_max: float,
    tie_manifest_path: str | None = None,
) -> dict[int, list[TieDrawEntry]]:
    """Load tie segments from the tie-point shard manifest (``OBDsaveSourceData`` layout)."""
    from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, load_tie_points_from_shards

    man = tie_manifest_path or DEFAULT_TIE_OUTPUT
    if not os.path.isfile(man):
        return {}
    payload = load_tie_points_from_shards(man, n_list=n_vals, require_all=False)
    return load_tie_draw_entries_from_payload(payload, n_vals, tie_p_min, tie_p_max)
