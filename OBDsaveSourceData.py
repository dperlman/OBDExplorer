import argparse
import concurrent.futures as cf
import csv
from datetime import datetime
import math
import os
import pickle
import sys
import tempfile
import time
from mpmath import mp
import numpy as np
from scipy.special import comb, gammaln
from sympy import S, binomial

# Default action when running this script with no CLI flags.
# Allowed values: "all", "tie", "graph", "cusp".
# Note: "graph" uses the sharded graph-data path (manifest + per-n shards).
# If DEFAULT_ACTION == "all", run order is: graph -> tie points -> cusp sidecar.
DEFAULT_ACTION = "all"

# Defaults aligned with OBDgraphExplorer1.py (graph) and CLI tie-point range
DEFAULT_TIE_N_MIN = 2
DEFAULT_TIE_N_MAX = 1000
DEFAULT_GRAPH_N_MIN = 2
DEFAULT_GRAPH_N_MAX = 1000
DEFAULT_GRAPH_P_STEPS = 1001
DATA_DIR = "data"
DEFAULT_TIE_SHARDS_DIR = os.path.join(DATA_DIR, "tie_points_shards")
DEFAULT_TIE_MANIFEST_FILENAME = "0000_manifest.pkl"
DEFAULT_TIE_OUTPUT = os.path.join(DEFAULT_TIE_SHARDS_DIR, DEFAULT_TIE_MANIFEST_FILENAME)
DEFAULT_CUSP_OUTPUT = os.path.join(DATA_DIR, "tieCuspSlopes.pkl")
DEFAULT_GRAPH_SHARDS_DIR = os.path.join(DATA_DIR, "graph_data_shards")
DEFAULT_GRAPH_SHARDS_MANIFEST = os.path.join(
    DEFAULT_GRAPH_SHARDS_DIR, f"0000_manifest_p{DEFAULT_GRAPH_P_STEPS:05d}.pkl"
)
DEFAULT_GRAPH_OUTPUT = DEFAULT_GRAPH_SHARDS_MANIFEST
DEFAULT_SLOPE_OUTPUT = os.path.join(DATA_DIR, "slope_data.pkl")
LOG_DIR = "log"
DEFAULT_TIE_SLOPE_A = 0.1
DEFAULT_TIE_SLOPE_FLAT_TOL = 1e-8 ## on 2026-05-02: was 1e-15. did a bunch of empirical tests and none of the N tested were any different at 1e-8 vs 1e-15 so this is more justifiably cautious.
DEFAULT_TIE_SLOPE_EPS_MIN = 1e-13
DEFAULT_HP_DPS = 40
DEFAULT_TOL_SCALE = 1e12
DEFAULT_WORKERS = 8

_EXPECTED_RANK_N_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
_EXPECTED_RANK_N_MP_CACHE: dict[tuple[int, int], tuple[list[mp.mpf], list[mp.mpf]]] = {}


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if missing (e.g. ``data/`` for ``data/graph_data.pkl``)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _atomic_pickle_dump(path: str, payload: dict) -> None:
    """Write pickle atomically: dump to temp file in same directory, then os.replace()."""
    _ensure_parent_dir(path)
    parent = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_pickle_", suffix=".pkl", dir=parent)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _tie_shard_filename_for_n(n: int) -> str:
    return f"tie_points_n{int(n):04d}.pkl"


def _graph_shard_filename_for_n(n: int, p_steps: int) -> str:
    return f"graph_n{int(n):04d}_p{int(p_steps):05d}.pkl"


def _graph_manifest_filename_for_p_steps(p_steps: int) -> str:
    return f"0000_manifest_p{int(p_steps):05d}.pkl"


def _resolve_graph_manifest_path(
    manifest_path: str | None, p_steps: int | None, shards_dir: str = DEFAULT_GRAPH_SHARDS_DIR
) -> str:
    """Return the graph shard manifest path. No directory scanning or alternate-``p_steps`` fallback.

    If ``manifest_path`` is set, it is used. Otherwise the path is
    ``<shards_dir>/0000_manifest_p{ps:05d}.pkl`` with ``ps = p_steps`` or, when
    ``p_steps`` is omitted, ``DEFAULT_GRAPH_P_STEPS`` (1001, i.e. ``p01001``).
    If that file is missing, callers must fail; we do not look for another manifest.
    """
    if manifest_path:
        return manifest_path
    ps = int(p_steps) if p_steps is not None else DEFAULT_GRAPH_P_STEPS
    return os.path.join(shards_dir, _graph_manifest_filename_for_p_steps(ps))


def _resolve_manifest_shard_path(manifest_path: str, shard_ref: str) -> str:
    if os.path.isabs(shard_ref):
        return shard_ref
    manifest_parent = os.path.dirname(os.path.abspath(manifest_path)) or "."
    return os.path.join(manifest_parent, shard_ref)

# Which n to run when n_list is None (e.g. print_*_table, save_tie_points); matches CLI tie defaults
N_LIST = list(range(DEFAULT_TIE_N_MIN, DEFAULT_TIE_N_MAX + 1))

# Symbolic tie math (all_tie_points_exact) is used only in print_comparison_table—not when saving pickles.


def _canonical_center_pair_ij(n: int) -> tuple[int, int]:
    """Representative for the symmetry family ``i + j == n`` at ``p == 1/2``.

    Uses the two indices closest to the center with ``i < j`` (same as skipping the
    symmetric diagonal in the float loop): ``i = (n - 1) // 2``, ``j = n - i``.
    E.g. ``n = 4`` → ``(1, 3)``; ``n = 5`` → ``(2, 3)``.
    """
    ni = int(n)
    i = (ni - 1) // 2
    j = ni - i
    return (int(i), int(j))


def _is_canonical_center_tie(n: int, pairs: list[tuple[int, int]]) -> bool:
    """True iff this tie record is the lone center tie at ``p = 1/2`` (canonical symmetric pair)."""
    return pairs == [_canonical_center_pair_ij(n)]


def all_tie_points_float_with_pairs(n: int) -> list[tuple[float, list[tuple[int, int]]]]:
    """
    Canonical numeric tie points from (i, j) crossings.

    - Skips every symmetric pair with ``i + j == n`` (same ``p`` as every other symmetric pair).
    - Inserts one center tie ``p = 1/2`` with the canonical symmetric pair (closest to center, ``i < j``).
    - Keeps separate entries for all other pairs; no numeric merge beyond that.

    Returns ``(p, [(i,j)])`` rows sorted by ``p``.
    """
    ni = int(n)
    if ni < 1:
        return []
    out: list[tuple[float, tuple[int, int]]] = []
    for i in range(ni + 1):
        for j in range(i + 1, ni + 1):
            if i + j == ni:
                continue
            ratio = comb(ni - i, j - i, exact=False) / comb(j, i, exact=False)
            if ratio <= 0 or not np.isfinite(ratio):
                continue
            exp = 1.0 / (j - i)
            p = 1.0 / (1.0 + ratio**exp)
            if 0 < p < 1 and np.isfinite(p):
                out.append((p, (i, j)))
    out.sort(key=lambda x: x[0])
    rows: list[tuple[float, list[tuple[int, int]]]] = [(float(p), [pair]) for p, pair in out]
    rows.append((0.5, [_canonical_center_pair_ij(ni)]))
    rows.sort(key=lambda x: x[0])
    return rows


def all_tie_points(n: int) -> np.ndarray:
    """Sorted tie ``p`` values (same order as ``all_tie_points_float_with_pairs``).

    Symmetric pairs ``i+j==n`` are skipped; one canonical center tie ``p=1/2`` (see ``_canonical_center_pair_ij``).
    """
    recs = all_tie_points_float_with_pairs(n)
    if not recs:
        return np.array([], dtype=float)
    return np.array([p for p, _ in recs], dtype=float)


def _expected_sorted_rank_many(n: int, p_values: np.ndarray) -> np.ndarray:
    """Vectorized E_sorted for many p values at fixed n."""
    p_arr = np.asarray(p_values, dtype=float).reshape(-1)
    cached = _EXPECTED_RANK_N_CACHE.get(int(n))
    if cached is None:
        n_float = float(n)
        ks = np.arange(n + 1, dtype=float)
        n_minus_ks = n_float - ks
        log_coeff = gammaln(n_float + 1.0) - gammaln(ks + 1.0) - gammaln(n_minus_ks + 1.0)
        cached = (ks, n_minus_ks, log_coeff, n_float)
        _EXPECTED_RANK_N_CACHE[int(n)] = cached
    ks, n_minus_ks, log_coeff, n_float = cached

    out = np.empty(p_arr.size, dtype=float)
    for i, p in enumerate(p_arr):
        if p <= 0.0:
            pmf = np.zeros(n + 1, dtype=float)
            pmf[0] = 1.0
        elif p >= 1.0:
            pmf = np.zeros(n + 1, dtype=float)
            pmf[-1] = 1.0
        else:
            # Log-domain PMF avoids underflow at large n and extreme tails.
            p_f = float(p)
            q_f = 1.0 - p_f
            log_pmf = log_coeff + (ks * math.log(p_f)) + (n_minus_ks * math.log(q_f))
            m = float(np.max(log_pmf))
            w = np.exp(log_pmf - m)
            s = float(np.sum(w))
            if s <= 0.0 or not np.isfinite(s):
                pmf = np.zeros(n + 1, dtype=float)
                pmf[int(round(n_float * p_f))] = 1.0
            else:
                pmf = w / s
        # Stable sort preserves index order for equal probabilities: key=(pmf[i], i).
        perm = np.argsort(pmf, kind="stable")
        out[i] = float(np.dot(ks, pmf[perm]))
    return out


def _expected_sorted_rank_at_p(n: int, p: float) -> float:
    """Expected sorted rank E_sorted for one p (thin wrapper)."""
    return float(_expected_sorted_rank_many(n, np.array([p], dtype=float))[0])


def _expected_sorted_rank_many_high_precision(
    n: int, p_values: np.ndarray, dps: int = DEFAULT_HP_DPS
) -> np.ndarray:
    """Higher-precision E_sorted for many p values at fixed n (slow path)."""
    p_arr = np.asarray(p_values, dtype=float).reshape(-1)
    out = np.empty(p_arr.size, dtype=float)
    if p_arr.size == 0:
        return out

    n_int = int(n)
    dps_int = max(40, int(dps))
    with mp.workdps(dps_int):
        cache_key = (n_int, dps_int)
        cached = _EXPECTED_RANK_N_MP_CACHE.get(cache_key)
        if cached is None:
            ks_mpf = [mp.mpf(k) for k in range(n_int + 1)]
            log_coeff = [mp.log(mp.binomial(n_int, k)) for k in range(n_int + 1)]
            cached = (ks_mpf, log_coeff)
            _EXPECTED_RANK_N_MP_CACHE[cache_key] = cached
        ks_mpf, log_coeff = cached

        for i, p in enumerate(p_arr):
            p_f = float(p)
            if p_f <= 0.0:
                out[i] = 0.0
                continue
            if p_f >= 1.0:
                out[i] = float(n_int)
                continue

            p_mp = mp.mpf(p_f)
            q_mp = mp.mpf(1.0) - p_mp
            log_p = mp.log(p_mp)
            log_q = mp.log(q_mp)

            log_pmf = [log_coeff[k] + (ks_mpf[k] * log_p) + (mp.mpf(n_int - k) * log_q) for k in range(n_int + 1)]
            m = max(log_pmf)
            w = [mp.e ** (v - m) for v in log_pmf]
            s = mp.fsum(w)
            if not mp.isfinite(s) or s <= 0:
                out[i] = float(int(round(float(n_int) * p_f)))
                continue
            pmf = [wk / s for wk in w]
            perm = sorted(range(n_int + 1), key=lambda idx: (pmf[idx], idx))
            out[i] = float(mp.fsum((mp.mpf(rank) * pmf[idx]) for rank, idx in enumerate(perm)))
    return out


def _slope_sign(v: float, flat_tol: float) -> int:
    if not np.isfinite(v) or abs(float(v)) <= float(flat_tol):
        return 0
    return 1 if float(v) > 0.0 else -1


def _expected_sorted_rank_at_p_high_precision(
    n: int, p: float | str | mp.mpf, dps: int = DEFAULT_HP_DPS
) -> mp.mpf:
    """High-precision E_sorted for one p."""
    n_int = int(n)
    dps_int = max(40, int(dps))
    with mp.workdps(dps_int):
        p_mp = mp.mpf(p)
        if p_mp <= 0:
            return mp.mpf(0)
        if p_mp >= 1:
            return mp.mpf(n_int)
        cache_key = (n_int, dps_int)
        cached = _EXPECTED_RANK_N_MP_CACHE.get(cache_key)
        if cached is None:
            ks_mpf = [mp.mpf(k) for k in range(n_int + 1)]
            log_coeff = [mp.log(mp.binomial(n_int, k)) for k in range(n_int + 1)]
            cached = (ks_mpf, log_coeff)
            _EXPECTED_RANK_N_MP_CACHE[cache_key] = cached
        ks_mpf, log_coeff = cached
        q_mp = mp.mpf(1) - p_mp
        log_p = mp.log(p_mp)
        log_q = mp.log(q_mp)
        log_pmf = [log_coeff[k] + (ks_mpf[k] * log_p) + (mp.mpf(n_int - k) * log_q) for k in range(n_int + 1)]
        m = max(log_pmf)
        w = [mp.e ** (v - m) for v in log_pmf]
        s = mp.fsum(w)
        if not mp.isfinite(s) or s <= 0:
            return mp.mpf(int(round(float(n_int) * float(p_mp))))
        pmf = [wk / s for wk in w]
        perm = sorted(range(n_int + 1), key=lambda idx: (pmf[idx], idx))
        return mp.fsum((mp.mpf(rank) * pmf[idx]) for rank, idx in enumerate(perm))


def _three_point_slopes_high_precision(
    n: int,
    p_left: float,
    tp: float,
    p_right: float,
    dps: int = DEFAULT_HP_DPS,
) -> tuple[float, float, bool]:
    """HP 3-point slopes around tp; returns (slope_left, slope_right, failed)."""
    try:
        with mp.workdps(max(40, int(dps))):
            p_l = mp.mpf(repr(float(p_left)))
            p_t = mp.mpf(repr(float(tp)))
            p_r = mp.mpf(repr(float(p_right)))
            e_l = _expected_sorted_rank_at_p_high_precision(int(n), p_l, dps=dps)
            e_t = _expected_sorted_rank_at_p_high_precision(int(n), p_t, dps=dps)
            e_r = _expected_sorted_rank_at_p_high_precision(int(n), p_r, dps=dps)
            dl = p_t - p_l
            dr = p_r - p_t
            s_l = mp.mpf(0) if dl <= 0 else (e_t - e_l) / dl
            s_r = mp.mpf(0) if dr <= 0 else (e_r - e_t) / dr
            return float(s_l), float(s_r), False
    except Exception:
        return float("nan"), float("nan"), True


def _deadzone_resolution_label(rec: dict) -> str:
    """Short outcome after dead-zone HP escalation (deadzone ties only)."""
    reason = str(rec.get("resolution_failure_reason", ""))
    if reason == "hp_eval_failed":
        return "hp_eval_failed"
    if reason == "hp_flat_sign":
        return "ambiguous_hp_flat_sign"
    return "resolved"


def _tie_slope_records_for_n(
    n: int,
    recs: list[tuple[float, list[tuple[int, int]]]],
    *,
    slope_a: float = DEFAULT_TIE_SLOPE_A,
    slope_eps_cap: float = 1e-3,
    slope_flat_tol: float = DEFAULT_TIE_SLOPE_FLAT_TOL,
    slope_eps_min: float = DEFAULT_TIE_SLOPE_EPS_MIN,
) -> tuple[list[dict], bool, int, int, dict[str, int | float], list[dict[str, str | int | float]]]:
    """Build per-tie slope metadata for one n.

    Returns (records, hit_eps_floor, n_deadzone_used, n_deadzone_ambiguous, stats, warning_rows), where:
      - n_deadzone_used counts tie points where left or right slope hit the flat dead-zone at base eps.
      - n_deadzone_ambiguous counts dead-zone points unresolved by HP 3-point refinement.
      - stats includes HP 3-point counts/timing and epsilon adjustment counters.
      - warning_rows: one dict per tie with any anomaly (eps floor, ulp step, or dead-zone), for CSV logs.
    """
    if not recs:
        return [], False, 0, 0, {
            "ulp_step_adjusted": 0,
            "hp_3point_sec": 0.0,
            "high_precision_points": 0,
            "hp_3point_attempted": 0,
            "hp_3point_resolved": 0,
            "hp_3point_failed": 0,
            "hp_flat_tol_scale": float(DEFAULT_TOL_SCALE),
        }, []

    ties = [float(p) for p, _pairs in recs]
    tp_arr = np.asarray(ties, dtype=float)

    # Local epsilon per tie point: scale by nearest adjacent tie gap.
    left_gap_arr = np.empty_like(tp_arr, dtype=float)
    right_gap_arr = np.empty_like(tp_arr, dtype=float)
    if tp_arr.size == 1:
        left_gap_arr[0] = float(tp_arr[0])
        right_gap_arr[0] = float(1.0 - tp_arr[0])
    else:
        left_gap_arr[0] = float(tp_arr[0])
        left_gap_arr[1:] = tp_arr[1:] - tp_arr[:-1]
        right_gap_arr[-1] = float(1.0 - tp_arr[-1])
        right_gap_arr[:-1] = tp_arr[1:] - tp_arr[:-1]
    local_min_gap_arr = np.minimum(left_gap_arr, right_gap_arr)

    eps_arr = np.minimum(float(slope_a) * local_min_gap_arr, float(slope_eps_cap))
    eps_arr = np.where(np.isfinite(eps_arr), eps_arr, float(slope_eps_cap))
    hit_eps_floor_per_tie = eps_arr < float(slope_eps_min)
    hit_floor = bool(np.any(hit_eps_floor_per_tie))
    # Absolute minimum epsilon (hard floor).
    eps_arr = np.maximum(eps_arr, float(slope_eps_min))

    p_left = np.clip(tp_arr - eps_arr, 0.0, 1.0)
    p_right = np.clip(tp_arr + eps_arr, 0.0, 1.0)

    # Ensure strict one-sided sampling around tp.
    left_same = p_left == tp_arr
    right_same = p_right == tp_arr
    if np.any(left_same):
        p_left[left_same] = np.nextafter(tp_arr[left_same], 0.0)
    if np.any(right_same):
        p_right[right_same] = np.nextafter(tp_arr[right_same], 1.0)
    n_ulp_adjusted = int(np.count_nonzero(left_same) + np.count_nonzero(right_same))

    # Initial 3-point slopes are now only used as a dead-zone trigger.
    p_all = np.concatenate([p_left, tp_arr, p_right])
    e_all = _expected_sorted_rank_many(n, p_all)
    m = tp_arr.size
    e_left_arr = e_all[:m]
    e_mid_arr = e_all[m : 2 * m]
    e_right_arr = e_all[2 * m :]

    left_span = tp_arr - p_left
    right_span = p_right - tp_arr
    slope_left_arr = np.divide(
        (e_mid_arr - e_left_arr),
        left_span,
        out=np.zeros_like(e_mid_arr, dtype=float),
        where=left_span > 0.0,
    )
    slope_right_arr = np.divide(
        (e_right_arr - e_mid_arr),
        right_span,
        out=np.zeros_like(e_mid_arr, dtype=float),
        where=right_span > 0.0,
    )
    dead_left_mask = np.isfinite(slope_left_arr) & (np.abs(slope_left_arr) <= float(slope_flat_tol))
    dead_right_mask = np.isfinite(slope_right_arr) & (np.abs(slope_right_arr) <= float(slope_flat_tol))
    dead_mask = dead_left_mask | dead_right_mask
    n_deadzone_used = int(np.count_nonzero(dead_mask))

    s_left_arr = np.where(
        (~np.isfinite(slope_left_arr)) | (np.abs(slope_left_arr) <= float(slope_flat_tol)),
        0,
        np.where(slope_left_arr > 0.0, 1, -1),
    )
    s_right_arr = np.where(
        (~np.isfinite(slope_right_arr)) | (np.abs(slope_right_arr) <= float(slope_flat_tol)),
        0,
        np.where(slope_right_arr > 0.0, 1, -1),
    )
    # Fast path (non-dead-zone): keep original 3-point sign classification.
    s_left_final = s_left_arr.copy()
    s_right_final = s_right_arr.copy()
    dir_change_arr = (s_left_final * s_right_final) < 0
    dir_change_ambiguous_arr = np.zeros(tp_arr.size, dtype=bool)
    quantization_suspect_arr = np.zeros(tp_arr.size, dtype=bool)
    high_precision_used_arr = np.zeros(tp_arr.size, dtype=bool)
    resolution_path_arr = np.full(tp_arr.size, "non_deadzone_fast", dtype=object)
    resolution_failure_reason_arr = np.full(tp_arr.size, "", dtype=object)
    hp_3point_attempted = 0
    hp_3point_resolved = 0
    hp_3point_failed = 0
    t_hp3point = 0.0

    dead_idx = np.flatnonzero(dead_mask)
    for idx_j in dead_idx:
        quantization_suspect_arr[idx_j] = True
        tp = float(tp_arr[idx_j])
        resolution_path_arr[idx_j] = "deadzone_hp3point"
        high_precision_used_arr[idx_j] = True
        hp_3point_attempted += 1
        hp_flat_tol = float(slope_flat_tol) / float(DEFAULT_TOL_SCALE)

        t0_hp = time.perf_counter()
        hp_s_left, hp_s_right, hp_failed = _three_point_slopes_high_precision(
            n=int(n),
            p_left=float(p_left[idx_j]),
            tp=tp,
            p_right=float(p_right[idx_j]),
            dps=DEFAULT_HP_DPS,
        )
        t_hp3point += time.perf_counter() - t0_hp

        if hp_failed:
            hp_3point_failed += 1
            dir_change_ambiguous_arr[idx_j] = True
            resolution_failure_reason_arr[idx_j] = "hp_eval_failed"
            continue

        slope_left_arr[idx_j] = float(hp_s_left)
        slope_right_arr[idx_j] = float(hp_s_right)
        s_left_final[idx_j] = _slope_sign(float(hp_s_left), float(hp_flat_tol))
        s_right_final[idx_j] = _slope_sign(float(hp_s_right), float(hp_flat_tol))
        dir_change_arr[idx_j] = bool(s_left_final[idx_j] * s_right_final[idx_j] < 0)
        if s_left_final[idx_j] == 0 or s_right_final[idx_j] == 0:
            dir_change_ambiguous_arr[idx_j] = True
            resolution_failure_reason_arr[idx_j] = "hp_flat_sign"
            continue
        hp_3point_resolved += 1
        dir_change_ambiguous_arr[idx_j] = False

    extremum_type_arr = np.full(tp_arr.size, "neither", dtype=object)
    minima_mask = (s_left_final < 0) & (s_right_final > 0)
    maxima_mask = (s_left_final > 0) & (s_right_final < 0)
    extremum_type_arr[minima_mask] = "minimum"
    extremum_type_arr[maxima_mask] = "maximum"

    n_deadzone_ambiguous = int(np.count_nonzero(dir_change_ambiguous_arr))
    out: list[dict] = []
    for i, tp in enumerate(ties):
        slope_left = float(slope_left_arr[i])
        slope_right = float(slope_right_arr[i])
        out.append(
            {
                "p": float(tp),
                "expected_sorted": float(e_mid_arr[i]),
                "slope_left": slope_left,
                "slope_right": slope_right,
                "dir_change": bool(dir_change_arr[i]),
                "extremum_type": str(extremum_type_arr[i]),
                "extremum_ambiguous": bool(dir_change_ambiguous_arr[i]),
                "deadzone_used": bool(dead_mask[i]),
                "dir_change_ambiguous": bool(dir_change_ambiguous_arr[i]),
                "quantization_suspect": bool(quantization_suspect_arr[i]),
                "high_precision_used": bool(high_precision_used_arr[i]),
                "resolution_path": str(resolution_path_arr[i]),
                "resolution_failure_reason": str(resolution_failure_reason_arr[i]),
            }
        )

    center_idx = int(np.argmin(np.abs(tp_arr - 0.5))) if tp_arr.size > 0 else 0
    warning_rows: list[dict[str, str | int | float]] = []
    for i, tp in enumerate(ties):
        pairs_list = list(recs[i][1]) if i < len(recs) else []
        if _is_canonical_center_tie(int(n), pairs_list):
            continue
        eps_floor_hit = bool(hit_eps_floor_per_tie[i])
        ulp_l = bool(left_same[i])
        ulp_r = bool(right_same[i])
        dead = bool(dead_mask[i])
        if not (eps_floor_hit or ulp_l or ulp_r or dead):
            continue
        rec_i = out[i]
        flags: list[str] = []
        if eps_floor_hit:
            flags.append("eps_floor")
        if ulp_l or ulp_r:
            flags.append("ulp_step_adjusted")
        if dead:
            flags.append("deadzone")
        if dead_left_mask[i] and dead_right_mask[i]:
            flat_side = "both"
        elif dead_left_mask[i]:
            flat_side = "left"
        elif dead_right_mask[i]:
            flat_side = "right"
        else:
            flat_side = ""
        methods_tried = ""
        deadzone_resolution = "na"
        if dead:
            methods_tried = "float64_3point;hp_3point_mpmath"
            deadzone_resolution = _deadzone_resolution_label(rec_i)
        pair_rows = pairs_list if pairs_list else [(-1, -1)]
        for pi, pj in pair_rows:
            warning_rows.append(
                {
                    "n": int(n),
                    "tie_idx": int(i),
                    "tie_index_signed": int(i - center_idx),
                    "tie_p": f"{float(tp):.17g}",
                    "i": int(pi),
                    "j": int(pj),
                    "anomaly_flags": ",".join(flags),
                    "eps_floor_hit": int(eps_floor_hit),
                    "ulp_adjust_left": int(ulp_l),
                    "ulp_adjust_right": int(ulp_r),
                    "deadzone_used": int(dead),
                    "deadzone_flat_side": flat_side,
                    "methods_tried": methods_tried,
                    "deadzone_resolution": deadzone_resolution,
                    "resolution_path": str(rec_i.get("resolution_path", "")),
                    "resolution_failure_reason": str(rec_i.get("resolution_failure_reason", "")),
                    "extremum_type": str(rec_i.get("extremum_type", "")),
                    "extremum_ambiguous": int(bool(rec_i.get("extremum_ambiguous", False))),
                    "dir_change": int(bool(rec_i.get("dir_change", False))),
                }
            )

    return out, hit_floor, n_deadzone_used, n_deadzone_ambiguous, {
        "ulp_step_adjusted": int(n_ulp_adjusted),
        "hp_3point_attempted": int(hp_3point_attempted),
        "hp_3point_resolved": int(hp_3point_resolved),
        "hp_3point_failed": int(hp_3point_failed),
        "hp_flat_tol_scale": float(DEFAULT_TOL_SCALE),
        "hp_3point_sec": float(t_hp3point),
        "high_precision_points": int(np.count_nonzero(high_precision_used_arr)),
    }, warning_rows


def all_tie_points_exact(n: int) -> tuple[np.ndarray, list, list[list[tuple[int, int]]]]:
    """
    Return (arr, symbolic_list, pair_groups): one SymPy tie ``p`` per non-symmetric ``(i,j)``
    pair plus exactly one ``p=S(1)/2`` for the canonical symmetric pair (see ``_canonical_center_pair_ij``).

    Skips symmetric pairs ``i+j==n`` (same mathematics as ``all_tie_points_float_with_pairs``).

    Duplicate ``p`` from unrelated pairs remains separate rows.

    ``pair_groups[k]`` is ``[(i, j)]``.
    """
    n_sym = S(n)
    n_int = int(n)
    if n_int < 1:
        return np.array([], dtype=float), [], []
    rows: list[tuple[float, int, int, object]] = []
    for i in range(n_int + 1):
        for j in range(i + 1, n_int + 1):
            if i + j == n_int:
                continue
            ratio = binomial(n_sym, j) / binomial(n_sym, i)
            if ratio <= 0:
                continue
            exp = S(1) / (j - i)
            base = ratio ** exp
            p = S(1) / (1 + base)
            keep = False
            try:
                if p > 0 and p < 1:
                    keep = True
            except TypeError:
                keep = True
            if not keep:
                continue
            try:
                val = float(p.evalf())
            except (TypeError, ValueError):
                continue
            if 0 < val < 1 and np.isfinite(val):
                rows.append((val, int(i), int(j), p))
    ic, jc = _canonical_center_pair_ij(n_int)
    rows.append((0.5, ic, jc, S.Half))
    rows.sort(key=lambda t: (t[0], t[1], t[2]))
    arr = np.array([t[0] for t in rows], dtype=float) if rows else np.array([], dtype=float)
    syms = [t[3] for t in rows]
    pair_groups = [[(t[1], t[2])] for t in rows]
    return arr, syms, pair_groups


def _arrays_match(a: np.ndarray, b: np.ndarray, atol: float = 1e-9) -> bool:
    """True if both arrays have the same length and pairwise values are within atol."""
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    return np.allclose(a, b, atol=atol, rtol=0)


def print_comparison_table(n_list: list[int] | None = None, atol: float = 1e-9) -> None:
    """Print a table comparing float vs exact tie-point counts, match, and max pairwise diff."""
    ns = n_list if n_list is not None else N_LIST
    float_by_n = {}
    symbolic_by_n = {}
    print("n   len(float) len(exact)  match    max|diff|")
    print("-" * 45)
    for n in ns:
        a = all_tie_points(n)
        b, syms, _pg = all_tie_points_exact(n)
        float_by_n[n] = a
        symbolic_by_n[n] = syms
        match = _arrays_match(a, b, atol=atol)
        if len(a) == len(b) and len(a) > 0:
            max_diff = float(np.max(np.abs(a - b)))
        else:
            max_diff = float("nan")
        print(f"{n:2}   {len(a):9} {len(b):9}  {str(match):5}   {max_diff:.2e}")
    print()
    for n in ns:
        print(f"n={n}: {symbolic_by_n[n]}")


def load_tie_points_from_shards(
    path: str = DEFAULT_TIE_OUTPUT,
    n_list: list[int] | None = None,
    require_all: bool = True,
    *,
    progress: int | None = None,
) -> dict:
    """Load tie-point data from shard manifest and rebuild monolithic dict structure.

    Returns a dict with keys:
      - float_by_n
      - float_with_pairs_by_n
      - tie_slope_by_n

    If ``n_list`` is provided, only those n values are loaded. If ``require_all`` is True,
    raises when requested n is missing from the manifest or shard file.

    If ``progress`` is a positive integer ``N``, prints timing on stderr every ``N`` processed
    ``n`` values (and on the last). ``None`` or ``0`` disables progress reporting.
    """
    with open(path, "rb") as f:
        manifest = pickle.load(f)

    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid tie manifest payload in {path!r}: expected dict.")
    if manifest.get("format") != "obd.tie_points_slope.shards.v1":
        raise ValueError(
            f"Unsupported tie manifest format in {path!r}: {manifest.get('format')!r}."
        )

    n_entries = manifest.get("n_entries", {})
    if not isinstance(n_entries, dict):
        raise ValueError(f"Invalid n_entries in manifest {path!r}: expected dict.")

    if n_list is None:
        target_ns = sorted(int(k) for k in n_entries.keys())
    else:
        target_ns = [int(n) for n in n_list]

    progress_every: int | None = None
    if progress is not None and progress > 0:
        progress_every = int(progress)

    float_by_n: dict[int, np.ndarray] = {}
    float_with_pairs_by_n: dict[int, list[tuple[float, list[tuple[int, int]]]]] = {}
    tie_slope_by_n: dict[int, list[dict]] = {}

    total = len(target_ns)
    t0 = time.perf_counter()
    if progress_every:
        if total == 0:
            print("[html] tie shards: no n values to load", file=sys.stderr)
        else:
            print(
                f"[html] tie shards: loading {total} n values from {path!r}",
                file=sys.stderr,
            )

    for step_index, n in enumerate(target_ns, start=1):
        try:
            entry = n_entries.get(str(n))
            if not isinstance(entry, dict):
                if require_all:
                    raise ValueError(f"Missing shard manifest entry for n={n} in {path!r}.")
                continue

            shard_ref = str(entry.get("shard_path", ""))
            if not shard_ref:
                if require_all:
                    raise ValueError(f"Missing shard_path for n={n} in manifest {path!r}.")
                continue

            shard_path = _resolve_manifest_shard_path(path, shard_ref)
            if not os.path.exists(shard_path):
                if require_all:
                    raise FileNotFoundError(
                        f"Shard file not found for n={n}: {shard_path!r} (from {path!r})."
                    )
                continue

            with open(shard_path, "rb") as f:
                shard_payload = pickle.load(f)

            if not isinstance(shard_payload, dict):
                if require_all:
                    raise ValueError(f"Invalid shard payload for n={n}: {shard_path!r}.")
                continue

            if "float_by_n" not in shard_payload or "float_with_pairs_by_n" not in shard_payload or "tie_slope_by_n" not in shard_payload:
                if require_all:
                    raise ValueError(
                        f"Shard payload missing required keys for n={n}: {shard_path!r}."
                    )
                continue

            float_by_n[n] = shard_payload["float_by_n"]
            float_with_pairs_by_n[n] = shard_payload["float_with_pairs_by_n"]
            tie_slope_by_n[n] = shard_payload["tie_slope_by_n"]
        finally:
            if progress_every and total:
                pe = progress_every
                if step_index % pe == 0 or step_index == total:
                    elapsed = time.perf_counter() - t0
                    nk = len(float_with_pairs_by_n)
                    print(
                        f"[html] tie shards: n={n} step {step_index}/{total} "
                        f"elapsed {elapsed:.2f}s loaded_n={nk}",
                        file=sys.stderr,
                    )

    if progress_every and total:
        elapsed = time.perf_counter() - t0
        print(
            f"[html] tie shards: done loaded_n={len(float_with_pairs_by_n)} in {elapsed:.2f}s",
            file=sys.stderr,
        )

    return {
        "float_by_n": float_by_n,
        "float_with_pairs_by_n": float_with_pairs_by_n,
        "tie_slope_by_n": tie_slope_by_n,
    }


def load_graph_data_from_shards(
    manifest_path: str | None = None,
    shards_dir: str = DEFAULT_GRAPH_SHARDS_DIR,
    p_steps: int | None = None,
    n_list: list[int] | None = None,
    require_all: bool = True,
) -> dict:
    """Load graph-data shards and return grouped rows by n."""
    resolved_manifest_path = _resolve_graph_manifest_path(manifest_path, p_steps, shards_dir)
    if not os.path.isfile(resolved_manifest_path):
        want_ps = int(p_steps) if p_steps is not None else DEFAULT_GRAPH_P_STEPS
        raise FileNotFoundError(
            f"Graph shard manifest not found for p_steps={want_ps} (no other manifest is tried): "
            f"{os.path.abspath(resolved_manifest_path)}"
        )
    with open(resolved_manifest_path, "rb") as f:
        manifest = pickle.load(f)

    if not isinstance(manifest, dict):
        raise ValueError(
            f"Invalid graph-shard manifest payload in {resolved_manifest_path!r}: expected dict."
        )
    if manifest.get("format") != "obd.graph_data.shards.v2":
        raise ValueError(
            f"Unsupported graph-shard manifest format in {resolved_manifest_path!r}: {manifest.get('format')!r}."
        )

    n_entries = manifest.get("n_entries", {})
    if not isinstance(n_entries, dict):
        raise ValueError(
            f"Invalid n_entries in graph-shard manifest {resolved_manifest_path!r}: expected dict."
        )

    if n_list is None:
        target_ns = sorted(int(k) for k in n_entries.keys())
    else:
        target_ns = [int(n) for n in n_list]

    rows_by_n: dict[int, dict[str, np.ndarray]] = {}
    for n in target_ns:
        entry = n_entries.get(str(n))
        if not isinstance(entry, dict):
            if require_all:
                raise ValueError(
                    f"Missing graph-shard manifest entry for n={n} in {resolved_manifest_path!r}."
                )
            continue

        shard_ref = str(entry.get("shard_path", ""))
        if not shard_ref:
            if require_all:
                raise ValueError(
                    f"Missing shard_path for n={n} in graph-shard manifest {resolved_manifest_path!r}."
                )
            continue

        shard_path = _resolve_manifest_shard_path(resolved_manifest_path, shard_ref)
        if not os.path.exists(shard_path):
            if require_all:
                raise FileNotFoundError(
                    f"Graph shard file not found for n={n}: {shard_path!r} (from {resolved_manifest_path!r})."
                )
            continue

        with open(shard_path, "rb") as f:
            shard_payload = pickle.load(f)

        if not isinstance(shard_payload, dict):
            if require_all:
                raise ValueError(f"Invalid graph shard payload for n={n}: {shard_path!r}.")
            continue
        if shard_payload.get("format") != "obd.graph_data.n_shard.v2":
            if require_all:
                raise ValueError(
                    f"Unsupported graph shard format for n={n}: {shard_payload.get('format')!r}."
                )
            continue
        if (
            "y" not in shard_payload
            or "perm" not in shard_payload
            or "expected_sorted_by_p" not in shard_payload
            or "expected_sorted_slope_by_p" not in shard_payload
        ):
            if require_all:
                raise ValueError(
                    f"Graph shard payload missing required keys for n={n}: {shard_path!r}."
                )
            continue

        rows_by_n[n] = {
            "y": np.asarray(shard_payload["y"]),
            "perm": np.asarray(shard_payload["perm"]),
            "expected_sorted_by_p": np.asarray(shard_payload["expected_sorted_by_p"]),
            "expected_sorted_slope_by_p": np.asarray(shard_payload["expected_sorted_slope_by_p"]),
        }

    p_values = np.asarray(manifest.get("p_values", []), dtype=np.float32)
    return {
        "format": "obd.graph_data.shards.v2",
        "n_min": int(manifest.get("n_min", min(rows_by_n) if rows_by_n else 0)),
        "n_max": int(manifest.get("n_max", max(rows_by_n) if rows_by_n else -1)),
        "p_steps": int(manifest.get("p_steps", len(p_values))),
        "p_half_start": int((int(manifest.get("p_steps", len(p_values))) - 1) // 2),
        "p_values": p_values,
        "rows_by_n": rows_by_n,
        "manifest_path": resolved_manifest_path,
    }


def load_cusp_data(
    path: str = DEFAULT_CUSP_OUTPUT,
    n_list: list[int] | None = None,
    require_all: bool = True,
) -> dict:
    """Load cusp-sidecar pickle written by ``save_cusp_data_from_tie_shards``.

    Returns a dict with file metadata plus ``n_entries``: ``dict[int, dict]`` mapping each
    ``n`` to its stored block (``center_index``, ``center_p_float``, ``count_candidates``,
    ``updated_at``, ``records``, and ``n`` when present).

    On disk, ``n_entries`` uses string keys; this loader normalizes them to ``int`` keys.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid cusp payload in {path!r}: expected dict.")
    fmt = payload.get("format")
    if fmt != "obd.tie_cusp_slopes.v3":
        raise ValueError(
            f"Unsupported cusp format in {path!r}: {fmt!r} (expected 'obd.tie_cusp_slopes.v3')."
        )

    raw_entries = payload.get("n_entries")
    if not isinstance(raw_entries, dict):
        raise ValueError(f"Invalid n_entries in cusp file {path!r}: expected dict.")

    if n_list is None:
        target_ns = sorted(int(k) for k in raw_entries.keys())
    else:
        target_ns = [int(n) for n in n_list]

    n_entries: dict[int, dict] = {}
    for n in target_ns:
        entry = raw_entries.get(str(n))
        if entry is None and n in raw_entries:
            entry = raw_entries.get(n)
        if not isinstance(entry, dict):
            if require_all:
                raise ValueError(f"Missing cusp entry for n={n} in {path!r}.")
            continue
        n_entries[int(n)] = entry

    return {
        "format": str(fmt),
        "created_at": str(payload.get("created_at", "")),
        "updated_at": str(payload.get("updated_at", "")),
        "source_tie_manifest": str(payload.get("source_tie_manifest", "")),
        "hp_dps": int(payload["hp_dps"]),
        "include_hp_three_point": bool(payload.get("include_hp_three_point", False)),
        "n_entries": n_entries,
    }


def _mp_to_str(v: mp.mpf, dps: int) -> str:
    return mp.nstr(v, n=max(20, int(dps)), strip_zeros=False)


def _tie_p_from_pair_high_precision(n: int, i: int, j: int, dps: int) -> mp.mpf:
    """Crossing probability ``p`` for pair ``(i, j)`` using the simplified stable ratio."""
    with mp.workdps(max(40, int(dps))):
        if j <= i:
            raise ValueError(f"Invalid pair ({i}, {j}): require j > i.")
        ratio = mp.binomial(int(n) - int(i), int(j) - int(i)) / mp.binomial(int(j), int(i))
        exp = mp.mpf(1) / mp.mpf(int(j - i))
        base = ratio**exp
        return mp.mpf(1) / (mp.mpf(1) + base)


def _local_eps_for_tie_index(
    tie_arr: np.ndarray,
    idx: int,
    slope_a: float,
    slope_eps_cap: float,
    slope_eps_min: float,
) -> float:
    tp = float(tie_arr[idx])
    left_neighbor = float(tie_arr[idx - 1]) if idx > 0 else 0.0
    right_neighbor = float(tie_arr[idx + 1]) if idx < (tie_arr.size - 1) else 1.0
    left_gap = max(tp - left_neighbor, 0.0)
    right_gap = max(right_neighbor - tp, 0.0)
    local_min_gap = min(left_gap, right_gap)
    eps = min(float(slope_a) * local_min_gap, float(slope_eps_cap))
    eps = max(eps, float(slope_eps_min))
    return float(eps)


def _hp_three_point_block(
    n: int,
    p_mid: mp.mpf,
    eps: float,
    dps: int,
) -> dict[str, str]:
    with mp.workdps(max(40, int(dps))):
        p_l = max(mp.mpf(0), p_mid - mp.mpf(repr(float(eps))))
        p_r = min(mp.mpf(1), p_mid + mp.mpf(repr(float(eps))))
        if p_l == p_mid:
            p_l = mp.nexttoward(p_mid, mp.mpf(0))
        if p_r == p_mid:
            p_r = mp.nexttoward(p_mid, mp.mpf(1))
        e_l = _expected_sorted_rank_at_p_high_precision(int(n), p_l, dps=dps)
        e_m = _expected_sorted_rank_at_p_high_precision(int(n), p_mid, dps=dps)
        e_r = _expected_sorted_rank_at_p_high_precision(int(n), p_r, dps=dps)
        dl = p_mid - p_l
        dr = p_r - p_mid
        s_l = mp.mpf(0) if dl <= 0 else (e_m - e_l) / dl
        s_r = mp.mpf(0) if dr <= 0 else (e_r - e_m) / dr
        return {
            "p_left_hp": _mp_to_str(p_l, dps),
            "p_mid_hp": _mp_to_str(p_mid, dps),
            "p_right_hp": _mp_to_str(p_r, dps),
            "ev_left_hp": _mp_to_str(e_l, dps),
            "ev_mid_hp": _mp_to_str(e_m, dps),
            "ev_right_hp": _mp_to_str(e_r, dps),
            "slope_left_hp": _mp_to_str(s_l, dps),
            "slope_right_hp": _mp_to_str(s_r, dps),
        }


def save_cusp_data_from_tie_shards(
    tie_manifest_path: str = DEFAULT_TIE_OUTPUT,
    path: str = DEFAULT_CUSP_OUTPUT,
    n_list: list[int] | None = None,
    hp_dps: int = DEFAULT_HP_DPS,
    include_hp_three_point: bool = True,
    slope_a: float = DEFAULT_TIE_SLOPE_A,
    slope_eps_cap: float = 1e-3,
    slope_eps_min: float = DEFAULT_TIE_SLOPE_EPS_MIN,
    save_every: int = 20,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> None:
    """Build cusp-only sidecar file from existing tie slope shards.

    Cusp candidates are points where extremum_type == "minimum" OR extremum_ambiguous is True.
    """
    tie_manifest_abs = os.path.abspath(tie_manifest_path)
    if not os.path.isfile(tie_manifest_abs):
        print(
            "ERROR: cusp generation requires an existing tie-point manifest and shard pickles.\n"
            f"  Missing file (resolved path): {tie_manifest_abs}\n"
            "\n"
            "  Build tie data first, for example:\n"
            f"    python OBDsaveSourceData.py --save-tie-points\n"
            "  Or run the full explorer bundle (graph + tie shards + cusp):\n"
            f"    python OBDsaveSourceData.py --all\n"
            "\n"
            "  If your manifest lives elsewhere, pass:\n"
            f"    --tie-output PATH   (then --save-cusp-data uses the same path)\n",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    with open(tie_manifest_path, "rb") as f:
        manifest = pickle.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid tie manifest payload in {tie_manifest_path!r}: expected dict.")
    if manifest.get("format") != "obd.tie_points_slope.shards.v1":
        raise ValueError(
            f"Unsupported tie manifest format in {tie_manifest_path!r}: {manifest.get('format')!r}."
        )
    n_entries = manifest.get("n_entries", {})
    if not isinstance(n_entries, dict):
        raise ValueError(f"Invalid n_entries in manifest {tie_manifest_path!r}: expected dict.")
    if n_list is None:
        target_ns = sorted(int(k) for k in n_entries.keys())
    else:
        target_ns = [int(n) for n in n_list]

    if save_every < 1:
        raise ValueError("save_every must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload: dict = {
        "format": "obd.tie_cusp_slopes.v3",
        "created_at": now,
        "updated_at": now,
        "source_tie_manifest": os.path.abspath(tie_manifest_path),
        "hp_dps": int(hp_dps),
        "include_hp_three_point": bool(include_hp_three_point),
        "n_entries": {},
    }

    def _print_verbose_header() -> None:
        print("n   calc_sec  io_sec  iter_total  candidates  ambiguous")
        print("-" * 64)

    def _checkpoint_write_output() -> float:
        payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t_write0 = time.perf_counter()
        _atomic_pickle_dump(path, payload)
        return time.perf_counter() - t_write0

    manifest_parent = os.path.dirname(os.path.abspath(tie_manifest_path)) or "."
    pending_items: list[tuple[int, str]] = []
    for n in target_ns:
        entry = n_entries.get(str(int(n)))
        if not isinstance(entry, dict):
            raise ValueError(f"Missing shard manifest entry for n={n} in {tie_manifest_path!r}.")
        shard_ref = str(entry.get("shard_path", ""))
        if not shard_ref:
            raise ValueError(f"Missing shard_path for n={n} in manifest {tie_manifest_path!r}.")
        shard_path = os.path.join(manifest_parent, shard_ref) if not os.path.isabs(shard_ref) else shard_ref
        if not os.path.exists(shard_path):
            raise FileNotFoundError(
                f"Shard file not found for n={n}: {shard_path!r} (from {tie_manifest_path!r})."
            )
        pending_items.append((int(n), os.path.abspath(shard_path)))

    total_candidates = 0
    total_ambiguous = 0
    n_computed = 0
    n_since_save = 0
    n_output_writes = 0
    t_total = time.perf_counter()
    header_printed = False
    payload_dirty = False

    if verbose and pending_items:
        _print_verbose_header()
        header_printed = True

    def _finalize_one(result: dict) -> None:
        nonlocal total_candidates, total_ambiguous
        nonlocal n_computed, n_since_save, n_output_writes, payload_dirty
        n_val = int(result["n"])
        calc_sec = float(result["calc_sec"])
        io_sec = float(result["io_sec"])
        iter_total_sec = float(result["iter_total_sec"])
        n_ambiguous = int(result["ambiguous_count"])
        recs_out = list(result["records"])

        payload["n_entries"][str(n_val)] = {
            "n": n_val,
            "center_index": int(result["center_index"]),
            "center_p_float": float(result["center_p_float"]),
            "count_candidates": int(len(recs_out)),
            "updated_at": str(result["updated_at"]),
            "records": recs_out,
        }
        payload_dirty = True

        total_candidates += int(len(recs_out))
        total_ambiguous += int(n_ambiguous)
        n_computed += 1
        n_since_save += 1

        if verbose:
            print(
                f"cusp n={n_val:4d} candidates={len(recs_out):4d} "
                f"ambiguous={n_ambiguous:4d} "
                f"calc_sec={calc_sec:.3f} io_sec={io_sec:.3f} iter_total_sec={iter_total_sec:.3f}",
                flush=True,
            )

        if n_since_save >= save_every:
            t_write = _checkpoint_write_output()
            n_since_save = 0
            n_output_writes += 1
            payload_dirty = False
            if verbose:
                _print_verbose_header()
            else:
                print(
                    "Checkpoint cusp output write: "
                    f"n={n_val}, computed={n_computed}, writes={n_output_writes}, write_sec={t_write:.4f}",
                    flush=True,
                )

    if pending_items:
        if workers == 1:
            for n, shard_path in pending_items:
                result = _compute_cusp_from_shard(
                    n=n,
                    shard_path=shard_path,
                    hp_dps=int(hp_dps),
                    include_hp_three_point=bool(include_hp_three_point),
                    slope_a=float(slope_a),
                    slope_eps_cap=float(slope_eps_cap),
                    slope_eps_min=float(slope_eps_min),
                )
                _finalize_one(result)
        else:
            with cf.ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = [
                    executor.submit(
                        _compute_cusp_from_shard,
                        n=int(n),
                        shard_path=str(shard_path),
                        hp_dps=int(hp_dps),
                        include_hp_three_point=bool(include_hp_three_point),
                        slope_a=float(slope_a),
                        slope_eps_cap=float(slope_eps_cap),
                        slope_eps_min=float(slope_eps_min),
                    )
                    for n, shard_path in pending_items
                ]
                for fut in cf.as_completed(futures):
                    result = fut.result()
                    _finalize_one(result)

    if payload_dirty:
        _checkpoint_write_output()
        n_output_writes += 1

    elapsed = time.perf_counter() - t_total
    if verbose and header_printed:
        print("-" * 64)
    print(
        f"Wrote cusp sidecar ({total_candidates} candidate points, ambiguous={total_ambiguous}, "
        f"writes={n_output_writes}, save_every={save_every}) "
        f"to {path} from {tie_manifest_path} in {elapsed:.2f}s"
    )
    if verbose:
        print(
            f"cusp config: hp_dps={hp_dps}, include_hp_three_point={int(include_hp_three_point)}, "
            f"workers={workers}, slope_eps_min={slope_eps_min:.17g}"
        )


def _compute_cusp_from_shard(
    n: int,
    shard_path: str,
    hp_dps: int,
    include_hp_three_point: bool,
    slope_a: float,
    slope_eps_cap: float,
    slope_eps_min: float,
) -> dict:
    """Worker task: compute cusp records for one n from one tie-slope shard."""
    t_iter0 = time.perf_counter()
    io_sec = 0.0
    calc_sec = 0.0
    t_io0 = time.perf_counter()
    with open(shard_path, "rb") as f:
        shard_payload = pickle.load(f)
    io_sec += time.perf_counter() - t_io0

    if not isinstance(shard_payload, dict):
        raise ValueError(f"Invalid shard payload for n={n}: {shard_path!r}.")
    if (
        "float_by_n" not in shard_payload
        or "float_with_pairs_by_n" not in shard_payload
        or "tie_slope_by_n" not in shard_payload
    ):
        raise ValueError(f"Shard payload missing required keys for n={n}: {shard_path!r}.")

    t_calc0 = time.perf_counter()
    tie_arr = np.asarray(shard_payload["float_by_n"], dtype=float).reshape(-1)
    pair_recs = list(shard_payload["float_with_pairs_by_n"])
    slope_recs = list(shard_payload["tie_slope_by_n"])

    n_slope = len(slope_recs)
    if tie_arr.size != n_slope or len(pair_recs) != n_slope:
        raise ValueError(
            f"cusp shard n={n}: length mismatch slope_recs={n_slope}, "
            f"float_by_n.size={tie_arr.size}, pairs={len(pair_recs)} ({shard_path!r})."
        )

    center_idx = 0
    center_p_float = 0.5
    recs_out: list[dict] = []
    n_ambiguous = 0

    if tie_arr.size > 0:
        center_idx = int(np.argmin(np.abs(tie_arr - 0.5)))
        center_p_float = float(tie_arr[center_idx])

        for idx, rec in enumerate(slope_recs):
            is_min = str(rec.get("extremum_type", "")) == "minimum"
            is_ambig = bool(rec.get("extremum_ambiguous", False))
            if not (is_min or is_ambig):
                continue
            p_float = float(rec.get("p", 0.0))
            tie_index = int(idx - center_idx)
            is_center = bool(tie_index == 0)
            pairs = list(pair_recs[idx][1]) if 0 <= idx < len(pair_recs) else []
            eps_local = _local_eps_for_tie_index(
                tie_arr=tie_arr,
                idx=idx,
                slope_a=float(slope_a),
                slope_eps_cap=float(slope_eps_cap),
                slope_eps_min=float(slope_eps_min),
            )

            hp_candidates: list[mp.mpf] = []
            if is_center:
                hp_candidates = [mp.mpf("0.5")]
            else:
                for i, j in pairs:
                    try:
                        hp_candidates.append(
                            _tie_p_from_pair_high_precision(int(n), int(i), int(j), int(hp_dps))
                        )
                    except Exception:
                        continue
                if not hp_candidates:
                    hp_candidates = [mp.mpf(repr(float(p_float)))]

            p_hp_main = (
                min(hp_candidates, key=lambda v: abs(float(v) - p_float))
                if hp_candidates
                else mp.mpf(repr(p_float))
            )
            hp_core = _hp_three_point_block(
                n=int(n),
                p_mid=p_hp_main,
                eps=float(eps_local),
                dps=int(hp_dps),
            )
            hp_block: dict[str, str] | None = hp_core if include_hp_three_point else None

            rec_out = {
                "n": int(n),
                "p_float": float(p_float),
                "tie_index": int(tie_index),
                "is_center_tie": bool(is_center),
                "pair_count": int(len(pairs)),
                "pairs": [(int(i), int(j)) for i, j in pairs],
                "extremum_type": str(rec.get("extremum_type", "neither")),
                "extremum_ambiguous": bool(is_ambig),
                "resolution_path": str(rec.get("resolution_path", "")),
                "resolution_failure_reason": str(rec.get("resolution_failure_reason", "")),
                "p_hp_main": _mp_to_str(p_hp_main, int(hp_dps)),
                "p_hp_candidate": _mp_to_str(p_hp_main, int(hp_dps)),
                "hp_pair_recompute_skipped": bool(is_center),
                "eps_local_float": float(eps_local),
                "ev_mid_hp": hp_core["ev_mid_hp"],
                "slope_left_hp": hp_core["slope_left_hp"],
                "slope_right_hp": hp_core["slope_right_hp"],
                "hp_three_point": hp_block,
                "diagnostic_source_hint": "minimum_or_ambiguous",
            }
            recs_out.append(rec_out)
            n_ambiguous += int(is_ambig)

    calc_sec += time.perf_counter() - t_calc0

    return {
        "n": int(n),
        "center_index": int(center_idx),
        "center_p_float": float(center_p_float),
        "records": recs_out,
        "ambiguous_count": int(n_ambiguous),
        "calc_sec": float(calc_sec),
        "io_sec": float(io_sec),
        "iter_total_sec": float(time.perf_counter() - t_iter0),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _compute_tie_shard(
    n: int,
    slope_a: float,
    slope_flat_tol: float,
    slope_eps_cap: float,
    slope_eps_min: float,
    shards_dir_abs: str,
) -> dict:
    """Worker task: compute tie points + slope metadata for one n and write one shard."""
    t0 = time.perf_counter()
    recs = all_tie_points_float_with_pairs(int(n))
    slope_recs, hit_floor, n_deadzone_used, n_deadzone_ambiguous, slope_stats, warning_rows = (
        _tie_slope_records_for_n(
            int(n),
            recs,
            slope_a=float(slope_a),
            slope_eps_cap=float(slope_eps_cap),
            slope_flat_tol=float(slope_flat_tol),
            slope_eps_min=float(slope_eps_min),
        )
    )
    local_min_ps = [float(r["p"]) for r in slope_recs if str(r.get("extremum_type", "")) == "minimum"]
    local_min_count = len(local_min_ps)
    max_local_min_p = max(local_min_ps) if local_min_ps else None
    n_ambig_after_max = 0
    if max_local_min_p is not None:
        n_ambig_after_max = sum(
            1
            for r in slope_recs
            if bool(r.get("dir_change_ambiguous", False))
            and float(r.get("p", -1.0)) > float(max_local_min_p)
        )
    tie_point_count = len(recs)
    t_compute = time.perf_counter() - t0

    t_write0 = time.perf_counter()
    shard_path_abs = os.path.join(shards_dir_abs, _tie_shard_filename_for_n(int(n)))
    _atomic_pickle_dump(
        shard_path_abs,
        {
            "n": int(n),
            "float_by_n": np.array([p for p, _ in recs], dtype=float),
            "float_with_pairs_by_n": recs,
            "tie_slope_by_n": slope_recs,
        },
    )
    t_write = time.perf_counter() - t_write0

    return {
        "n": int(n),
        "tie_point_count": int(tie_point_count),
        "local_min_count": int(local_min_count),
        "max_local_min_p": (float(max_local_min_p) if max_local_min_p is not None else None),
        "hit_eps_floor": int(bool(hit_floor)),
        "deadzone_used": int(n_deadzone_used),
        "deadzone_ambiguous": int(n_deadzone_ambiguous),
        "ulp_step_adjusted": int(slope_stats.get("ulp_step_adjusted", 0)),
        "hp_3point_sec": float(slope_stats.get("hp_3point_sec", 0.0)),
        "high_precision_points": int(slope_stats.get("high_precision_points", 0)),
        "hp_3point_attempted": int(slope_stats.get("hp_3point_attempted", 0)),
        "hp_3point_resolved": int(slope_stats.get("hp_3point_resolved", 0)),
        "hp_3point_failed": int(slope_stats.get("hp_3point_failed", 0)),
        "hp_flat_tol_scale": float(slope_stats.get("hp_flat_tol_scale", DEFAULT_TOL_SCALE)),
        "n_ambig_after_max": int(n_ambig_after_max),
        "compute_sec": float(t_compute),
        "write_sec": float(t_write),
        "shard_path_abs": shard_path_abs,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warning_rows": warning_rows,
    }


def save_tie_points(
    n_list: list[int] | None = None,
    path: str = DEFAULT_TIE_OUTPUT,
    shards_dir: str = DEFAULT_TIE_SHARDS_DIR,
    log_dir: str = LOG_DIR,
    atol: float = 1e-9,
    slope_a: float = DEFAULT_TIE_SLOPE_A,
    slope_flat_tol: float = DEFAULT_TIE_SLOPE_FLAT_TOL,
    slope_eps_cap: float = 1e-3,
    slope_eps_min: float = DEFAULT_TIE_SLOPE_EPS_MIN,
    save_every: int = 10,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> None:
    """Compute numeric tie points for each n and save in sharded per-n pickle files.

    Writes one shard per n under ``shards_dir`` and updates a manifest pickle at ``path``.
    Saves manifest periodically every ``save_every`` computed n (plus a final save).
    Also writes a timestamped CSV run log in ``log_dir`` with per-n tabular timing/stat rows,
    and ``tie_points_warnings_<timestamp>.csv`` with one row per tie point that had epsilon-floor,
    ULP step adjustment, or dead-zone escalation anomalies.

    Manifest keys:
      format: schema/version marker for the sharded layout.
      shards_dir: absolute path to shard directory.
      n_entries[str(n)]: metadata for each available shard.
    """
    ns = n_list if n_list is not None else N_LIST
    _ensure_parent_dir(path)
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    shards_dir_abs = os.path.abspath(shards_dir)

    t_start_dt = datetime.now()
    run_started_at = t_start_dt.strftime("%Y-%m-%d %H:%M:%S")
    run_started_slug = t_start_dt.strftime("%Y%m%d_%H%M%S")
    csv_log_path = os.path.join(log_dir, f"tie_points_verbose_{run_started_slug}.csv")
    warnings_log_path = os.path.join(log_dir, f"tie_points_warnings_{run_started_slug}.csv")

    manifest: dict = {}
    try:
        with open(path, "rb") as f:
            manifest = pickle.load(f)
    except FileNotFoundError:
        manifest = {}
    except (EOFError, pickle.UnpicklingError) as e:
        print(
            f"WARNING: Could not read existing tie manifest {path!r} ({e}). "
            "Starting from empty data and rebuilding."
        )
        manifest = {}
    except Exception:
        raise

    if not isinstance(manifest, dict):
        manifest = {}
    if manifest.get("format") != "obd.tie_points_slope.shards.v1":
        manifest = {
            "format": "obd.tie_points_slope.shards.v1",
            "created_at": run_started_at,
            "shards_dir": os.path.abspath(shards_dir),
            "n_entries": {},
        }
    manifest.setdefault("n_entries", {})
    manifest["shards_dir"] = os.path.abspath(shards_dir)
    n_entries = manifest["n_entries"]

    if not ns:
        print("save_tie_points: empty n_list, nothing to do.")
        return

    if save_every < 1:
        raise ValueError("save_every must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    def _print_verbose_header() -> None:
        print(
            "n   compute_sec  write_sec  iter_total  tie_points  "
            "local_min        max_local_min_p"
        )
        print("-" * 95)

    def _checkpoint_write_manifest() -> float:
        t_write0 = time.perf_counter()
        _atomic_pickle_dump(path, manifest)
        return time.perf_counter() - t_write0

    csv_fields = [
        "run_started_at",
        "n",
        "compute_sec",
        "write_sec",
        "iter_total_sec",
        "tie_points",
        "local_min",
        "max_local_min_p",
        "hit_eps_floor",
        "deadzone_used",
        "deadzone_ambiguous",
        "ulp_step_adjusted",
        "hp_3point_sec",
        "high_precision_points",
        "hp_3point_attempted",
        "hp_3point_resolved",
        "hp_3point_failed",
        "hp_flat_tol_scale",
    ]
    warning_fields = [
        "run_started_at",
        "n",
        "tie_idx",
        "tie_index_signed",
        "tie_p",
        "i",
        "j",
        "anomaly_flags",
        "eps_floor_hit",
        "ulp_adjust_left",
        "ulp_adjust_right",
        "deadzone_used",
        "deadzone_flat_side",
        "methods_tried",
        "deadzone_resolution",
        "resolution_path",
        "resolution_failure_reason",
        "extremum_type",
        "extremum_ambiguous",
        "dir_change",
    ]
    with open(csv_log_path, "w", newline="") as csv_f, open(warnings_log_path, "w", newline="") as warn_f:
        csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
        csv_writer.writeheader()
        warn_writer = csv.DictWriter(warn_f, fieldnames=warning_fields)
        warn_writer.writeheader()
        warn_f.flush()

        t_total = time.perf_counter()
        n_skipped = 0
        n_computed = 0
        n_since_save = 0
        n_manifest_writes = 0
        header_printed = False
        manifest_dirty = False
        pending_ns: list[int] = []

        for n in ns:
            entry = n_entries.get(str(int(n)))
            shard_ok = False
            if isinstance(entry, dict):
                shard_ref = str(entry.get("shard_path", ""))
                if shard_ref:
                    shard_path_existing = _resolve_manifest_shard_path(path, shard_ref)
                    shard_ok = os.path.exists(shard_path_existing)
            if shard_ok:
                n_skipped += 1
            else:
                pending_ns.append(int(n))

        if verbose and pending_ns:
            _print_verbose_header()
            header_printed = True

        def _finalize_one(result: dict) -> None:
            nonlocal n_computed, n_since_save, n_manifest_writes, manifest_dirty
            n_val = int(result["n"])
            tie_point_count = int(result["tie_point_count"])
            local_min_count = int(result["local_min_count"])
            max_local_min_p = result["max_local_min_p"]
            hit_floor = bool(result["hit_eps_floor"])
            n_deadzone_used = int(result["deadzone_used"])
            n_deadzone_ambiguous = int(result["deadzone_ambiguous"])
            ulp_step_adjusted = int(result["ulp_step_adjusted"])
            hp_3point_sec = float(result.get("hp_3point_sec", 0.0))
            high_precision_points = int(result.get("high_precision_points", 0))
            hp_3point_attempted = int(result.get("hp_3point_attempted", 0))
            hp_3point_resolved = int(result.get("hp_3point_resolved", 0))
            hp_3point_failed = int(result.get("hp_3point_failed", 0))
            hp_flat_tol_scale = float(result.get("hp_flat_tol_scale", DEFAULT_TOL_SCALE))
            n_ambig_after_max = int(result["n_ambig_after_max"])
            t_compute = float(result["compute_sec"])
            t_write = float(result["write_sec"])
            t_iter = t_compute + t_write
            shard_path_abs = str(result["shard_path_abs"])
            updated_at = str(result["updated_at"])

            if hit_floor:
                print(
                    "WARNING: tie slope epsilon hit floor "
                    f"for n={n_val} (eps_floor=tie_slope_eps_min={slope_eps_min:.17g})."
                )
            if n_deadzone_used > 0:
                print(
                    "INFO: tie slope dead-zone used "
                    f"for n={n_val} at {n_deadzone_used} tie point(s) "
                    f"(abs(slope) <= tie_slope_flat_tol={slope_flat_tol}); "
                    "HP 3-point escalation applied."
                )
                print(
                    "INFO: dead-zone resolution details "
                    f"for n={n_val}: hp_3point_attempted={hp_3point_attempted}, "
                    f"hp_3point_resolved={hp_3point_resolved}, "
                    f"hp_3point_failed={hp_3point_failed}, "
                    f"hp_flat_tol_scale={hp_flat_tol_scale:.1e}, "
                    f"ulp_step_adjusted={ulp_step_adjusted}, "
                    f"hp_3point_sec={hp_3point_sec:.6f}, "
                    f"high_precision_points={high_precision_points}, "
                    f"still_ambiguous={n_deadzone_ambiguous}."
                )
                if n_deadzone_ambiguous > 0:
                    print(
                        "WARNING: unresolved dir_change ambiguity remains "
                        f"for n={n_val} at {n_deadzone_ambiguous} tie point(s) after "
                        "float64 + HP 3-point checks."
                    )
                    print(
                        "DIAGNOSTIC COMMAND: "
                        "python tie_slope_diagnostic.py "
                        f"--n {n_val} "
                        "--point-filter unresolved "
                        f"--slope-a {slope_a:.17g} "
                        f"--slope-eps-cap {slope_eps_cap:.17g} "
                        f"--slope-flat-tol {slope_flat_tol:.17g} "
                        f"--slope-eps-min {slope_eps_min:.17g}"
                    )
                if n_ambig_after_max > 0:
                    print(
                        "WARNING: ambiguous dead-zone tie points occur above max_local_min_p "
                        f"for n={n_val} ({n_ambig_after_max} point(s)); "
                        "max_local_min_p may be unreliable."
                    )

            manifest_parent = os.path.dirname(os.path.abspath(path)) or "."
            shard_ref = os.path.relpath(shard_path_abs, start=manifest_parent)
            n_entries[str(n_val)] = {
                "n": n_val,
                "shard_path": shard_ref,
                "tie_points": tie_point_count,
                "local_min": local_min_count,
                "max_local_min_p": (
                    float(max_local_min_p) if max_local_min_p is not None else None
                ),
                "updated_at": updated_at,
            }
            manifest_dirty = True
            n_since_save += 1
            n_computed += 1

            csv_writer.writerow(
                {
                    "run_started_at": run_started_at,
                    "n": n_val,
                    "compute_sec": f"{t_compute:.6f}",
                    "write_sec": f"{t_write:.6f}",
                    "iter_total_sec": f"{t_iter:.6f}",
                    "tie_points": tie_point_count,
                    "local_min": local_min_count,
                    "max_local_min_p": (
                        f"{float(max_local_min_p):.12f}" if max_local_min_p is not None else ""
                    ),
                    "hit_eps_floor": int(hit_floor),
                    "deadzone_used": n_deadzone_used,
                    "deadzone_ambiguous": n_deadzone_ambiguous,
                    "ulp_step_adjusted": ulp_step_adjusted,
                    "hp_3point_sec": f"{hp_3point_sec:.6f}",
                    "high_precision_points": high_precision_points,
                    "hp_3point_attempted": hp_3point_attempted,
                    "hp_3point_resolved": hp_3point_resolved,
                    "hp_3point_failed": hp_3point_failed,
                    "hp_flat_tol_scale": f"{hp_flat_tol_scale:.1e}",
                }
            )
            csv_f.flush()

            for wr in result.get("warning_rows", []):
                row_out = dict(wr)
                row_out["run_started_at"] = run_started_at
                warn_writer.writerow(row_out)
            warn_f.flush()

            if verbose:
                max_p_str = f"{float(max_local_min_p):.6f}" if max_local_min_p is not None else "none"
                print(
                    f"{n_val:2}   {t_compute:10.4f}s  {t_write:9.4f}s  {t_iter:10.4f}s  "
                    f"{tie_point_count:10d}  {local_min_count:15d}  {max_p_str}",
                    flush=True,
                )

            if n_since_save >= save_every:
                t_manifest_write = _checkpoint_write_manifest()
                n_since_save = 0
                n_manifest_writes += 1
                manifest_dirty = False
                if verbose:
                    _print_verbose_header()
                else:
                    print(
                        "Checkpoint tie manifest write: "
                        f"n={n_val}, computed={n_computed}, skipped={n_skipped}, "
                        f"writes={n_manifest_writes}, write_sec={t_manifest_write:.4f}",
                        flush=True,
                    )

        if pending_ns:
            if workers == 1:
                for n in pending_ns:
                    result = _compute_tie_shard(
                        n=n,
                        slope_a=float(slope_a),
                        slope_flat_tol=float(slope_flat_tol),
                        slope_eps_cap=float(slope_eps_cap),
                        slope_eps_min=float(slope_eps_min),
                        shards_dir_abs=shards_dir_abs,
                    )
                    _finalize_one(result)
            else:
                with cf.ProcessPoolExecutor(max_workers=int(workers)) as executor:
                    futures = [
                        executor.submit(
                            _compute_tie_shard,
                            n=int(n),
                            slope_a=float(slope_a),
                            slope_flat_tol=float(slope_flat_tol),
                            slope_eps_cap=float(slope_eps_cap),
                            slope_eps_min=float(slope_eps_min),
                            shards_dir_abs=shards_dir_abs,
                        )
                        for n in pending_ns
                    ]
                    for fut in cf.as_completed(futures):
                        result = fut.result()
                        _finalize_one(result)

        if manifest_dirty:
            _checkpoint_write_manifest()
            n_manifest_writes += 1

        elapsed = time.perf_counter() - t_total
        if verbose and header_printed:
            print("-" * 95)
        if n_computed == 0 and n_skipped == len(ns):
            print(
                f"Tie points up to date ({len(ns)} n values, all skipped) - "
                f"manifest={path}, shards_dir={shards_dir}, "
                f"log={csv_log_path}, warnings_log={warnings_log_path}"
            )
        else:
            print(
                f"Wrote tie points shards ({n_computed} n computed, {n_skipped} skipped, "
                f"{n_manifest_writes} manifest writes, save_every={save_every}) "
                f"to shards_dir={shards_dir} with manifest={path} in {elapsed:.2f}s "
                f"(log={csv_log_path}, warnings_log={warnings_log_path})"
            )


def print_timing_table(n_list: list[int] | None = None) -> None:
    """Print run times: numeric tie points (via all_tie_points) vs exact symbolic path."""
    ns = n_list if n_list is not None else N_LIST
    print("n   time(numeric)  time(exact)")
    print("-" * 34)
    for n in ns:
        t0 = time.perf_counter()
        all_tie_points(n)
        t_num = time.perf_counter() - t0
        t0 = time.perf_counter()
        all_tie_points_exact(n)
        t_ex = time.perf_counter() - t0
        print(f"{n:2}   {t_num:11.4f}s  {t_ex:10.4f}s")


def _graph_binom_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) for Binomial(n, p). Used for graph explorer binomial_data pickles."""
    if p <= 0 or p >= 1:
        return 1.0 if (k == 0 and p <= 0) or (k == n and p >= 1) else 0.0
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def save_graph_data(
    path: str | None = None,
    shards_dir: str = DEFAULT_GRAPH_SHARDS_DIR,
    n_min: int = DEFAULT_GRAPH_N_MIN,
    n_max: int = DEFAULT_GRAPH_N_MAX,
    p_steps: int = DEFAULT_GRAPH_P_STEPS,
    save_every: int = 20,
    verbose: bool = False,
) -> None:
    """Precompute graph data into per-n shards with a manifest."""
    if n_min > n_max:
        raise ValueError("n_min must be <= n_max")
    if p_steps < 2:
        raise ValueError("p_steps must be at least 2")
    if save_every < 1:
        raise ValueError("save_every must be at least 1")

    path = _resolve_graph_manifest_path(path, int(p_steps), shards_dir)
    _ensure_parent_dir(path)
    os.makedirs(shards_dir, exist_ok=True)
    p_values = np.linspace(0.0, 1.0, int(p_steps), dtype=np.float32)

    manifest: dict = {}
    try:
        with open(path, "rb") as f:
            manifest = pickle.load(f)
    except FileNotFoundError:
        manifest = {}
    except (EOFError, pickle.UnpicklingError) as e:
        print(
            f"WARNING: Could not read existing graph shard manifest {path!r} ({e}). "
            "Starting from empty graph-shard data."
        )
        manifest = {}
    except Exception:
        raise

    if not isinstance(manifest, dict):
        manifest = {}
    if manifest.get("format") != "obd.graph_data.shards.v2":
        manifest = {
            "format": "obd.graph_data.shards.v2",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "p_steps": int(p_steps),
            "p_values": p_values,
            "n_min": int(n_min),
            "n_max": int(n_max),
            "shards_dir": os.path.abspath(shards_dir),
            "n_entries": {},
        }
    else:
        manifest_p_steps = int(manifest.get("p_steps", -1))
        if manifest_p_steps != int(p_steps):
            raise ValueError(
                f"Graph shard manifest p_steps={manifest_p_steps} does not match requested p_steps={p_steps}."
            )
        manifest["p_values"] = p_values

    manifest.setdefault("n_entries", {})
    manifest["shards_dir"] = os.path.abspath(shards_dir)
    n_entries = manifest["n_entries"]

    def _checkpoint_write_manifest() -> float:
        t_write0 = time.perf_counter()
        manifest["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _atomic_pickle_dump(path, manifest)
        return time.perf_counter() - t_write0

    t_total = time.perf_counter()
    n_computed = 0
    n_skipped = 0
    n_since_save = 0
    n_manifest_writes = 0
    manifest_dirty = False

    def _print_verbose_header() -> None:
        print("n   compute_sec  write_sec  iter_total  p_points  k_count")
        print("-" * 66)

    if verbose:
        _print_verbose_header()

    for n in range(n_min, n_max + 1):
        entry = n_entries.get(str(int(n)))
        shard_ok = False
        if isinstance(entry, dict):
            shard_ref = str(entry.get("shard_path", ""))
            shard_p_steps = int(entry.get("p_steps", -1))
            shard_k_count = int(entry.get("k_count", -1))
            if shard_ref and shard_p_steps == int(p_steps) and shard_k_count == int(n + 1):
                shard_path_existing = _resolve_manifest_shard_path(path, shard_ref)
                shard_ok = os.path.exists(shard_path_existing)
        if shard_ok:
            n_skipped += 1
            continue

        t0 = time.perf_counter()
        y_arr = np.empty((int(p_steps), int(n + 1)), dtype=np.float32)
        perm_arr = np.empty((int(p_steps), int(n + 1)), dtype=np.uint16)
        expected_sorted_arr = np.empty(int(p_steps), dtype=np.float32)
        ks_arr = np.arange(n + 1, dtype=np.float64)
        n_float = float(n)
        n_minus_ks_arr = n_float - ks_arr
        log_coeff = gammaln(n_float + 1.0) - gammaln(ks_arr + 1.0) - gammaln(n_minus_ks_arr + 1.0)
        for p_idx, p in enumerate(p_values):
            p_f = float(p)
            if p_f <= 0.0:
                pmf = np.zeros(n + 1, dtype=np.float64)
                pmf[0] = 1.0
            elif p_f >= 1.0:
                pmf = np.zeros(n + 1, dtype=np.float64)
                pmf[-1] = 1.0
            else:
                log_pmf = log_coeff + (ks_arr * math.log(p_f)) + (n_minus_ks_arr * math.log(1.0 - p_f))
                m = float(np.max(log_pmf))
                w = np.exp(log_pmf - m)
                s = float(np.sum(w))
                if s <= 0.0 or not np.isfinite(s):
                    pmf = np.zeros(n + 1, dtype=np.float64)
                    pmf[int(round(n_float * p_f))] = 1.0
                else:
                    pmf = w / s
            y_arr[p_idx, :] = pmf.astype(np.float32)
            perm_idx = np.argsort(pmf, kind="stable")
            perm_arr[p_idx, :] = perm_idx.astype(np.uint16)
            expected_sorted_arr[p_idx] = float(np.dot(ks_arr, pmf[perm_idx]))
        p_values_f64 = np.asarray(p_values, dtype=np.float64)
        expected_sorted_slope_arr = np.gradient(expected_sorted_arr.astype(np.float64), p_values_f64).astype(np.float32)
        t_compute = time.perf_counter() - t0

        t_write0 = time.perf_counter()
        shard_path_abs = os.path.join(
            os.path.abspath(shards_dir),
            _graph_shard_filename_for_n(n, int(p_steps)),
        )
        _atomic_pickle_dump(
            shard_path_abs,
            {
                "format": "obd.graph_data.n_shard.v2",
                "n": int(n),
                "p_steps": int(p_steps),
                "y": y_arr,
                "perm": perm_arr,
                "expected_sorted_by_p": expected_sorted_arr,
                "expected_sorted_slope_by_p": expected_sorted_slope_arr,
            },
        )
        manifest_parent = os.path.dirname(os.path.abspath(path)) or "."
        shard_ref = os.path.relpath(shard_path_abs, start=manifest_parent)
        n_entries[str(int(n))] = {
            "n": int(n),
            "shard_path": shard_ref,
            "rows": int(p_steps),
            "p_steps": int(p_steps),
            "k_count": int(n + 1),
            "dtype_y": "float32",
            "dtype_perm": "uint16",
            "dtype_expected_sorted": "float32",
            "dtype_expected_sorted_slope": "float32",
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        manifest_dirty = True
        n_since_save += 1
        if n_since_save >= save_every:
            t_manifest_write = _checkpoint_write_manifest()
            n_since_save = 0
            n_manifest_writes += 1
            manifest_dirty = False
            if verbose:
                _print_verbose_header()
            if not verbose:
                print(
                    "Checkpoint graph manifest write: "
                    f"n={n}, computed={n_computed + 1}, skipped={n_skipped}, "
                    f"writes={n_manifest_writes}, write_sec={t_manifest_write:.4f}"
                )
        t_write = time.perf_counter() - t_write0

        n_computed += 1
        if verbose:
            t_iter = t_compute + t_write
            print(
                f"{n:2}   {t_compute:10.4f}s  {t_write:9.4f}s  {t_iter:10.4f}s  "
                f"{p_steps:8d}  {n + 1:7d}"
            )

    if n_entries:
        ns_avail = sorted(int(k) for k in n_entries.keys())
        manifest["n_min"] = int(min(ns_avail))
        manifest["n_max"] = int(max(ns_avail))
    else:
        manifest["n_min"] = int(n_min)
        manifest["n_max"] = int(n_max)

    if manifest_dirty:
        _checkpoint_write_manifest()
        n_manifest_writes += 1

    elapsed = time.perf_counter() - t_total
    if verbose:
        print("-" * 66)
    if n_computed == 0 and n_skipped == (n_max - n_min + 1):
        print(
            f"Graph data shards up to date ({n_max - n_min + 1} n values, all skipped) — "
            f"manifest={path}, shards_dir={shards_dir}"
        )
    else:
        print(
            f"Wrote graph data shards ({n_computed} n computed, {n_skipped} skipped, "
            f"{n_manifest_writes} manifest writes, save_every={save_every}) "
            f"to shards_dir={shards_dir} with manifest={path} in {elapsed:.2f}s"
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Save tie-point pickle, graph precompute data, slope-by-p data, and/or print tie "
            "comparison/timing tables. "
            "With no flags, saves the OBDgraphExplorer1 bundle (same as -a/--all)."
        )
    )
    p.add_argument(
        "-a",
        "--all",
        action="store_true",
        help=(
            "Save graph data and tie points for OBDgraphExplorer1: same defaults as "
            f"--save-graph-data (n={DEFAULT_GRAPH_N_MIN}..{DEFAULT_GRAPH_N_MAX}, "
            f"p_steps={DEFAULT_GRAPH_P_STEPS}), tie points for that same n range "
            f"(writes graph manifest {DEFAULT_GRAPH_OUTPUT} and tie manifest {DEFAULT_TIE_OUTPUT}), "
            f"then builds the cusp sidecar ({DEFAULT_CUSP_OUTPUT}). "
            "--tie-n-min / --tie-n-max are not used for the tie save. "
            "Equivalent to --save-graph-data, --save-tie-points over the graph n-range, then "
            "--save-cusp-data for that same n-range. Combine with --graph-n-min etc. to change scope."
        ),
    )
    p.add_argument(
        "--save-tie-points",
        action="store_true",
        help="Compute numeric tie points (with i,j pairs) and save/update a pickle file.",
    )
    p.add_argument(
        "--save-cusp-data",
        action="store_true",
        help=(
            "Build cusp-only HP sidecar from tie slope shards "
            f"(default output: {DEFAULT_CUSP_OUTPUT})."
        ),
    )
    p.add_argument(
        "--tie-slope-a",
        type=float,
        default=DEFAULT_TIE_SLOPE_A,
        metavar="A",
        help=(
            "Adaptive epsilon factor for tie slope metadata: "
            "epsilon=min(A*min_inter_tie_dist, 0.001)."
        ),
    )
    p.add_argument(
        "--tie-slope-eps-cap",
        type=float,
        default=1e-3,
        metavar="EPS",
        help="Upper cap for tie slope epsilon (default: 0.001).",
    )
    p.add_argument(
        "--tie-slope-flat-tol",
        type=float,
        default=DEFAULT_TIE_SLOPE_FLAT_TOL,
        metavar="TOL",
        help=(
            "Absolute slope tolerance used to classify near-zero as flat "
            "for direction-change detection."
        ),
    )
    p.add_argument(
        "--tie-slope-eps-min",
        type=float,
        default=DEFAULT_TIE_SLOPE_EPS_MIN,
        metavar="EPS",
        help=(
            "Hard minimum epsilon for tie slope 3-point sampling after gap scaling "
            f"(default: {DEFAULT_TIE_SLOPE_EPS_MIN}). Also used for cusp HP aux epsilon."
        ),
    )
    p.add_argument(
        "--tie-save-every",
        type=int,
        default=20,
        metavar="K",
        help=(
            "Checkpoint cadence for tie manifest writes: save after every K computed n values "
            "(plus a final save). Each n is still written to its own shard file."
        ),
    )
    p.add_argument(
        "--tie-workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="K",
        help=(
            "Process workers for tie-points+slope computation; each worker computes one n "
            f"at a time (default: {DEFAULT_WORKERS})."
        ),
    )
    p.add_argument(
        "--tie-output",
        default=DEFAULT_TIE_OUTPUT,
        metavar="PATH",
        help=f"Output path for tie-point manifest (default: {DEFAULT_TIE_OUTPUT}).",
    )
    p.add_argument(
        "--cusp-output",
        default=DEFAULT_CUSP_OUTPUT,
        metavar="PATH",
        help=f"Output path for cusp sidecar data (default: {DEFAULT_CUSP_OUTPUT}).",
    )
    p.add_argument(
        "--cusp-save-every",
        type=int,
        default=20,
        metavar="K",
        help=(
            "Checkpoint cadence for cusp output writes: save after every K computed n values "
            "(plus a final save)."
        ),
    )
    p.add_argument(
        "--cusp-workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="K",
        help=(
            "Process workers for cusp computation from tie-slope shards; each worker computes one n "
            f"at a time (default: {DEFAULT_WORKERS})."
        ),
    )
    p.add_argument(
        "--tie-shards-dir",
        default=DEFAULT_TIE_SHARDS_DIR,
        metavar="DIR",
        help=f"Directory for per-n tie-point+slope shards (default: {DEFAULT_TIE_SHARDS_DIR}).",
    )
    p.add_argument(
        "--tie-log-dir",
        default=LOG_DIR,
        metavar="DIR",
        help=f"Directory for timestamped tie-point CSV logs (default: {LOG_DIR}).",
    )
    p.add_argument(
        "--tie-n-min",
        type=int,
        default=DEFAULT_TIE_N_MIN,
        metavar="N",
        help=f"Minimum n for tie points, comparison table, and timing table (default: {DEFAULT_TIE_N_MIN}).",
    )
    p.add_argument(
        "--tie-n-max",
        type=int,
        default=DEFAULT_TIE_N_MAX,
        metavar="N",
        help=f"Maximum n for tie points, inclusive (default: {DEFAULT_TIE_N_MAX}). "
        "Also used as n range for --print-comparison-table and --print-timing-table.",
    )
    p.add_argument(
        "--print-comparison-table",
        action="store_true",
        help="Print float vs exact (symbolic) tie-point comparison table (uses --tie-n-min / --tie-n-max).",
    )
    p.add_argument(
        "--print-timing-table",
        action="store_true",
        help="Print numeric vs exact (symbolic) tie-point timing table (uses --tie-n-min / --tie-n-max).",
    )
    p.add_argument(
        "--save-graph-data",
        action="store_true",
        help="Precompute binomial data for the graph explorer and save a pickle file.",
    )
    p.add_argument(
        "--graph-output",
        default=None,
        metavar="PATH",
        help=(
            "Output path for graph shard manifest. "
            f"Default is auto-derived from p_steps, e.g. {os.path.join(DEFAULT_GRAPH_SHARDS_DIR, _graph_manifest_filename_for_p_steps(DEFAULT_GRAPH_P_STEPS))}."
        ),
    )
    p.add_argument(
        "--graph-shards-dir",
        default=DEFAULT_GRAPH_SHARDS_DIR,
        metavar="DIR",
        help=f"Directory for per-n graph data shards (default: {DEFAULT_GRAPH_SHARDS_DIR}).",
    )
    p.add_argument(
        "--graph-save-every",
        type=int,
        default=20,
        metavar="K",
        help=(
            "Checkpoint cadence for graph shard manifest writes: save after every K computed n values "
            "(plus a final save)."
        ),
    )
    p.add_argument(
        "--graph-n-min",
        type=int,
        default=DEFAULT_GRAPH_N_MIN,
        metavar="N",
        help=f"Minimum n for graph data (default: {DEFAULT_GRAPH_N_MIN}).",
    )
    p.add_argument(
        "--graph-n-max",
        type=int,
        default=DEFAULT_GRAPH_N_MAX,
        metavar="N",
        help=f"Maximum n for graph data, inclusive (default: {DEFAULT_GRAPH_N_MAX}).",
    )
    p.add_argument(
        "--p-steps",
        type=int,
        default=DEFAULT_GRAPH_P_STEPS,
        metavar="K",
        help=f"Number of p grid values from 0 to 1 inclusive (default: {DEFAULT_GRAPH_P_STEPS}).",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Per-n progress for graph and tie saves; default is summary only.",
    )
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    has_explicit_action = (
        args.all
        or args.save_tie_points
        or args.save_cusp_data
        or args.save_graph_data
        or args.print_comparison_table
        or args.print_timing_table
    )
    if not has_explicit_action:
        default_action = str(DEFAULT_ACTION).strip().lower()
        if default_action == "all":
            args.all = True
        elif default_action == "tie":
            args.save_tie_points = True
        elif default_action == "cusp":
            args.save_cusp_data = True
        elif default_action == "graph":
            args.save_graph_data = True
        else:
            parser.error(
                f"Invalid DEFAULT_ACTION value: {DEFAULT_ACTION!r}. "
                "Use one of: all, tie, cusp, graph."
            )

    if args.all or args.save_graph_data:
        if args.graph_n_min > args.graph_n_max:
            parser.error("--graph-n-min must be <= --graph-n-max")
        if args.p_steps < 2:
            parser.error("--p-steps must be at least 2")

    need_tie_arg_range = (
        args.print_comparison_table
        or args.print_timing_table
        or (args.save_tie_points and not args.all)
        or (args.save_cusp_data and not args.all)
    )
    if need_tie_arg_range and args.tie_n_min > args.tie_n_max:
        parser.error("--tie-n-min must be <= --tie-n-max")
    if args.tie_slope_a <= 0:
        parser.error("--tie-slope-a must be > 0")
    if args.tie_slope_eps_cap <= 0:
        parser.error("--tie-slope-eps-cap must be > 0")
    if args.tie_slope_flat_tol < 0:
        parser.error("--tie-slope-flat-tol must be >= 0")
    if args.tie_slope_eps_min < 0:
        parser.error("--tie-slope-eps-min must be >= 0")
    if args.tie_save_every < 1:
        parser.error("--tie-save-every must be >= 1")
    if args.tie_workers < 1:
        parser.error("--tie-workers must be >= 1")
    if args.graph_save_every < 1:
        parser.error("--graph-save-every must be >= 1")
    if args.cusp_save_every < 1:
        parser.error("--cusp-save-every must be >= 1")
    if args.cusp_workers < 1:
        parser.error("--cusp-workers must be >= 1")

    tie_ns = list(range(args.tie_n_min, args.tie_n_max + 1))
    if args.print_comparison_table:
        print_comparison_table(n_list=tie_ns)
    if args.print_timing_table:
        print_timing_table(n_list=tie_ns)

    if args.all:
        if args.verbose:
            print(
                "Saving graph explorer bundle: graph data + tie points + cusp sidecar "
                f"(n={args.graph_n_min}..{args.graph_n_max}, p_steps={args.p_steps})..."
            )
        save_graph_data(
            path=args.graph_output,
            shards_dir=args.graph_shards_dir,
            n_min=args.graph_n_min,
            n_max=args.graph_n_max,
            p_steps=args.p_steps,
            save_every=args.graph_save_every,
            verbose=args.verbose,
        )
        tie_ns_explorer = list(range(args.graph_n_min, args.graph_n_max + 1))
        save_tie_points(
            n_list=tie_ns_explorer,
            path=args.tie_output,
            shards_dir=args.tie_shards_dir,
            log_dir=args.tie_log_dir,
            slope_a=args.tie_slope_a,
            slope_flat_tol=args.tie_slope_flat_tol,
            slope_eps_cap=args.tie_slope_eps_cap,
            slope_eps_min=args.tie_slope_eps_min,
            save_every=args.tie_save_every,
            workers=args.tie_workers,
            verbose=args.verbose,
        )
        save_cusp_data_from_tie_shards(
            tie_manifest_path=args.tie_output,
            path=args.cusp_output,
            n_list=tie_ns_explorer,
            save_every=args.cusp_save_every,
            workers=args.cusp_workers,
            verbose=args.verbose,
            slope_a=args.tie_slope_a,
            slope_eps_cap=args.tie_slope_eps_cap,
            slope_eps_min=args.tie_slope_eps_min,
        )
    else:
        if args.save_tie_points:
            save_tie_points(
                n_list=tie_ns,
                path=args.tie_output,
                shards_dir=args.tie_shards_dir,
                log_dir=args.tie_log_dir,
                slope_a=args.tie_slope_a,
                slope_flat_tol=args.tie_slope_flat_tol,
                slope_eps_cap=args.tie_slope_eps_cap,
                slope_eps_min=args.tie_slope_eps_min,
                save_every=args.tie_save_every,
                workers=args.tie_workers,
                verbose=args.verbose,
            )
        if args.save_graph_data:
            save_graph_data(
                path=args.graph_output,
                shards_dir=args.graph_shards_dir,
                n_min=args.graph_n_min,
                n_max=args.graph_n_max,
                p_steps=args.p_steps,
                save_every=args.graph_save_every,
                verbose=args.verbose,
            )
        if args.save_cusp_data:
            save_cusp_data_from_tie_shards(
                tie_manifest_path=args.tie_output,
                path=args.cusp_output,
                n_list=tie_ns,
                save_every=args.cusp_save_every,
                workers=args.cusp_workers,
                verbose=args.verbose,
                slope_a=args.tie_slope_a,
                slope_eps_cap=args.tie_slope_eps_cap,
                slope_eps_min=args.tie_slope_eps_min,
            )
