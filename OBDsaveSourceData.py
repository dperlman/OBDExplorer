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
import numpy as np
import obd_core
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
DEFAULT_WORKERS = 8

# Shard formats.  v2 (2026-10-05) replaced the finite-difference tie slopes of v1, which were wrong
# (negative slope jumps at most tie points, missed and mislabelled cusps), with exact ones from
# obd_core; v1 files are refused rather than read.
TIE_MANIFEST_FORMAT = "obd.tie_points_slope.shards.v2"
TIE_SHARD_FORMAT = "obd.tie_points_slope.n_shard.v2"
CUSP_FORMAT = "obd.tie_cusp_slopes.v4"
try:
    from importlib.metadata import version as _pkg_version

    OBD_CORE_VERSION = _pkg_version("obd-core")
except Exception:
    OBD_CORE_VERSION = "unknown"


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


def _tie_records_for_n(
    n: int,
) -> tuple[list[tuple[float, list[tuple[int, int]]]], list[dict], dict[str, int]]:
    """Every tie point of ``n`` with exact slopes and certified cusp verdicts, from ``obd_core``.

    Returns ``(recs, slope_recs, stats)``. ``recs`` is ``[(p, [(i, j)])]`` sorted by ``p``, the same
    layout as ``all_tie_points_float_with_pairs``: the mirror pairs ``i + j == n`` share the one
    center tie ``p = 1/2``, labelled with the canonical pair. ``slope_recs[k]`` belongs to
    ``recs[k]``:

    - ``p``, ``expected_sorted``: the tie point and E(n, p) there.
    - ``slope_left``, ``slope_right``: the exact one-sided slopes E'_- and E'_+ (from the ranking
      just left of the tie, not finite differences).
    - ``log10_D``: log10 of the slope jump D = E'_+ - E'_- > 0, computed from the pair mass and so
      exact far below double range. Never recompute it as ``slope_right - slope_left``: for most
      tie points D is many orders smaller than the slopes and the difference is 0 or noise.
    - ``is_cusp``: certified local minimum of E (double-precision screen, interval arithmetic for
      anything borderline). A tie point is never a local maximum, so there is no other extremum.
    - ``decided_by``: ``double``, ``iv50``/``iv100``/``iv200``, ``exact`` (adjacent pairs, rational
      p*), ``symmetry`` (the center tie) or ``UNRESOLVED``.

    ``stats`` has ``n_checked`` (tie points that needed interval arithmetic) and ``n_unresolved``.
    """
    ni = int(n)
    t = obd_core.tie_table(ni, both_halves=True)
    center_pair = _canonical_center_pair_ij(ni)
    recs: list[tuple[float, list[tuple[int, int]]]] = []
    slope_recs: list[dict] = []
    for i, j, p, e, sl, sr, ld, cusp, how in zip(
        t["i"].tolist(),
        t["j"].tolist(),
        t["pstar"].tolist(),
        t["E"].tolist(),
        t["slope_left"].tolist(),
        t["slope_right"].tolist(),
        t["log10_D"].tolist(),
        t["is_cusp"].tolist(),
        t["decided_by"].tolist(),
    ):
        pair = center_pair if (i, j) == (0, ni) else (int(i), int(j))
        recs.append((float(p), [pair]))
        slope_recs.append(
            {
                "p": float(p),
                "expected_sorted": float(e),
                "slope_left": float(sl),
                "slope_right": float(sr),
                "log10_D": float(ld),
                "is_cusp": bool(cusp),
                "decided_by": str(how),
            }
        )
    decided = t["decided_by"]
    stats = {
        "n_checked": int(sum(1 for d in decided if d.startswith("iv") or d in ("exact", "UNRESOLVED"))),
        "n_unresolved": int(sum(1 for d in decided if d == "UNRESOLVED")),
    }
    return recs, slope_recs, stats


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


def _check_tie_manifest_format(manifest: dict, path: str) -> None:
    fmt = manifest.get("format")
    if fmt == TIE_MANIFEST_FORMAT:
        return
    hint = ""
    if fmt == "obd.tie_points_slope.shards.v1":
        hint = (
            " v1 shards hold finite-difference tie slopes that are wrong (negative slope jumps,"
            " missed cusps); rebuild them with: python OBDsaveSourceData.py --save-tie-points"
        )
    raise ValueError(f"Unsupported tie manifest format in {path!r}: {fmt!r}.{hint}")


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
    float_by_n: dict[int, np.ndarray] = {}
    float_with_pairs_by_n: dict[int, list[tuple[float, list[tuple[int, int]]]]] = {}
    tie_slope_by_n: dict[int, list[dict]] = {}
    for n, n_payload in iter_tie_points_from_shards(
        path=path,
        n_list=n_list,
        require_all=require_all,
        progress=progress,
        include_float_by_n=True,
        include_float_with_pairs_by_n=True,
        include_tie_slope_by_n=True,
    ):
        float_by_n[n] = n_payload["float_by_n"]
        float_with_pairs_by_n[n] = n_payload["float_with_pairs_by_n"]
        tie_slope_by_n[n] = n_payload["tie_slope_by_n"]

    return {
        "float_by_n": float_by_n,
        "float_with_pairs_by_n": float_with_pairs_by_n,
        "tie_slope_by_n": tie_slope_by_n,
    }


def iter_tie_points_from_shards(
    path: str = DEFAULT_TIE_OUTPUT,
    n_list: list[int] | None = None,
    require_all: bool = True,
    *,
    progress: int | None = None,
    include_float_by_n: bool = False,
    include_float_with_pairs_by_n: bool = True,
    include_tie_slope_by_n: bool = True,
):
    """Yield tie-point shard payload one ``n`` at a time.

    Each yielded item is ``(n, payload_for_n)`` where ``payload_for_n`` only includes
    requested keys (from ``float_by_n``, ``float_with_pairs_by_n``, ``tie_slope_by_n``).
    This allows callers to stream over shards without keeping the full manifest payload
    resident in RAM.
    """
    if (
        not include_float_by_n
        and not include_float_with_pairs_by_n
        and not include_tie_slope_by_n
    ):
        raise ValueError("iter_tie_points_from_shards: at least one include_* flag must be True.")

    with open(path, "rb") as f:
        manifest = pickle.load(f)

    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid tie manifest payload in {path!r}: expected dict.")
    _check_tie_manifest_format(manifest, path)

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

    total = len(target_ns)
    t0 = time.perf_counter()
    yielded = 0
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

            need_keys: list[str] = []
            if include_float_by_n:
                need_keys.append("float_by_n")
            if include_float_with_pairs_by_n:
                need_keys.append("float_with_pairs_by_n")
            if include_tie_slope_by_n:
                need_keys.append("tie_slope_by_n")
            missing = [k for k in need_keys if k not in shard_payload]
            if missing:
                if require_all:
                    raise ValueError(
                        f"Shard payload missing required keys for n={n}: {missing} in {shard_path!r}."
                    )
                continue

            out_payload: dict[str, object] = {}
            if include_float_by_n:
                out_payload["float_by_n"] = shard_payload["float_by_n"]
            if include_float_with_pairs_by_n:
                out_payload["float_with_pairs_by_n"] = shard_payload["float_with_pairs_by_n"]
            if include_tie_slope_by_n:
                out_payload["tie_slope_by_n"] = shard_payload["tie_slope_by_n"]
            yielded += 1
            yield n, out_payload
        finally:
            if progress_every and total:
                pe = progress_every
                if step_index % pe == 0 or step_index == total:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[html] tie shards: n={n} step {step_index}/{total} "
                        f"elapsed {elapsed:.2f}s loaded_n={yielded}",
                        file=sys.stderr,
                    )

    if progress_every and total:
        elapsed = time.perf_counter() - t0
        print(
            f"[html] tie shards: done loaded_n={yielded} in {elapsed:.2f}s",
            file=sys.stderr,
        )


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
    ``n`` to its stored block (``n``, ``center_index``, ``center_p_float``, ``count_cusps``,
    ``updated_at``, ``records``).

    On disk, ``n_entries`` uses string keys; this loader normalizes them to ``int`` keys.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid cusp payload in {path!r}: expected dict.")
    fmt = payload.get("format")
    if fmt != CUSP_FORMAT:
        raise ValueError(
            f"Unsupported cusp format in {path!r}: {fmt!r} (expected {CUSP_FORMAT!r}). "
            "Rebuild it with: python OBDsaveSourceData.py --save-cusp-data"
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
        "n_entries": n_entries,
    }


def save_cusp_data_from_tie_shards(
    tie_manifest_path: str = DEFAULT_TIE_OUTPUT,
    path: str = DEFAULT_CUSP_OUTPUT,
    n_list: list[int] | None = None,
    save_every: int = 20,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> None:
    """Build the cusp-only sidecar file from existing tie shards.

    One record per certified cusp (``is_cusp``), copied from the tie shard: the values are already
    exact there, so nothing is recomputed.
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
    _check_tie_manifest_format(manifest, tie_manifest_path)
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
        "format": CUSP_FORMAT,
        "created_at": now,
        "updated_at": now,
        "source_tie_manifest": os.path.abspath(tie_manifest_path),
        "n_entries": {},
    }

    def _checkpoint_write_output() -> float:
        payload["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t_write0 = time.perf_counter()
        _atomic_pickle_dump(path, payload)
        return time.perf_counter() - t_write0

    pending_items: list[tuple[int, str]] = []
    for n in target_ns:
        entry = n_entries.get(str(int(n)))
        if not isinstance(entry, dict):
            raise ValueError(f"Missing shard manifest entry for n={n} in {tie_manifest_path!r}.")
        shard_ref = str(entry.get("shard_path", ""))
        if not shard_ref:
            raise ValueError(f"Missing shard_path for n={n} in manifest {tie_manifest_path!r}.")
        shard_path = _resolve_manifest_shard_path(tie_manifest_path, shard_ref)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(
                f"Shard file not found for n={n}: {shard_path!r} (from {tie_manifest_path!r})."
            )
        pending_items.append((int(n), os.path.abspath(shard_path)))

    total_cusps = 0
    n_computed = 0
    n_since_save = 0
    n_output_writes = 0
    t_total = time.perf_counter()
    payload_dirty = False

    def _finalize_one(result: dict) -> None:
        nonlocal total_cusps, n_computed, n_since_save, n_output_writes, payload_dirty
        n_val = int(result["n"])
        recs_out = list(result["records"])
        payload["n_entries"][str(n_val)] = {
            "n": n_val,
            "center_index": int(result["center_index"]),
            "center_p_float": float(result["center_p_float"]),
            "count_cusps": int(len(recs_out)),
            "updated_at": str(result["updated_at"]),
            "records": recs_out,
        }
        payload_dirty = True
        total_cusps += int(len(recs_out))
        n_computed += 1
        n_since_save += 1
        if verbose:
            print(
                f"cusp n={n_val:4d} cusps={len(recs_out):4d} "
                f"io_sec={float(result['io_sec']):.3f} iter_total_sec={float(result['iter_total_sec']):.3f}",
                flush=True,
            )
        if n_since_save >= save_every:
            t_write = _checkpoint_write_output()
            n_since_save = 0
            n_output_writes += 1
            payload_dirty = False
            if not verbose:
                print(
                    "Checkpoint cusp output write: "
                    f"n={n_val}, computed={n_computed}, writes={n_output_writes}, write_sec={t_write:.4f}",
                    flush=True,
                )

    if pending_items:
        if workers == 1:
            for n, shard_path in pending_items:
                _finalize_one(_compute_cusp_from_shard(n=n, shard_path=shard_path))
        else:
            with cf.ProcessPoolExecutor(max_workers=int(workers)) as executor:
                futures = [
                    executor.submit(_compute_cusp_from_shard, n=int(n), shard_path=str(shard_path))
                    for n, shard_path in pending_items
                ]
                for fut in cf.as_completed(futures):
                    _finalize_one(fut.result())

    if payload_dirty:
        _checkpoint_write_output()
        n_output_writes += 1

    elapsed = time.perf_counter() - t_total
    print(
        f"Wrote cusp sidecar ({total_cusps} cusps over {n_computed} n, writes={n_output_writes}, "
        f"save_every={save_every}) to {path} from {tie_manifest_path} in {elapsed:.2f}s"
    )


def _compute_cusp_from_shard(n: int, shard_path: str) -> dict:
    """Worker task: the cusp records for one n, read from its tie shard."""
    t_iter0 = time.perf_counter()
    with open(shard_path, "rb") as f:
        shard_payload = pickle.load(f)
    io_sec = time.perf_counter() - t_iter0

    if not isinstance(shard_payload, dict):
        raise ValueError(f"Invalid shard payload for n={n}: {shard_path!r}.")
    for key in ("float_by_n", "float_with_pairs_by_n", "tie_slope_by_n"):
        if key not in shard_payload:
            raise ValueError(f"Shard payload missing {key!r} for n={n}: {shard_path!r}.")

    tie_arr = np.asarray(shard_payload["float_by_n"], dtype=float).reshape(-1)
    pair_recs = list(shard_payload["float_with_pairs_by_n"])
    slope_recs = list(shard_payload["tie_slope_by_n"])
    if tie_arr.size != len(slope_recs) or len(pair_recs) != len(slope_recs):
        raise ValueError(
            f"cusp shard n={n}: length mismatch slope_recs={len(slope_recs)}, "
            f"float_by_n.size={tie_arr.size}, pairs={len(pair_recs)} ({shard_path!r})."
        )

    center_idx = int(np.argmin(np.abs(tie_arr - 0.5))) if tie_arr.size else 0
    center_p_float = float(tie_arr[center_idx]) if tie_arr.size else 0.5
    recs_out: list[dict] = []
    for idx, rec in enumerate(slope_recs):
        if not bool(rec.get("is_cusp", False)):
            continue
        tie_index = int(idx - center_idx)
        recs_out.append(
            {
                "n": int(n),
                "p_float": float(rec["p"]),
                "tie_index": tie_index,
                "is_center_tie": bool(tie_index == 0),
                "pairs": [(int(i), int(j)) for i, j in pair_recs[idx][1]],
                "expected_sorted": float(rec["expected_sorted"]),
                "slope_left": float(rec["slope_left"]),
                "slope_right": float(rec["slope_right"]),
                "log10_D": float(rec["log10_D"]),
                "decided_by": str(rec["decided_by"]),
            }
        )

    return {
        "n": int(n),
        "center_index": int(center_idx),
        "center_p_float": float(center_p_float),
        "records": recs_out,
        "io_sec": float(io_sec),
        "iter_total_sec": float(time.perf_counter() - t_iter0),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _compute_tie_shard(n: int, shards_dir_abs: str) -> dict:
    """Worker task: compute every tie point of one n (exact slopes, certified cusps) and write one shard."""
    t0 = time.perf_counter()
    recs, slope_recs, stats = _tie_records_for_n(int(n))
    cusp_ps = [float(r["p"]) for r in slope_recs if r["is_cusp"]]
    t_compute = time.perf_counter() - t0

    t_write0 = time.perf_counter()
    shard_path_abs = os.path.join(shards_dir_abs, _tie_shard_filename_for_n(int(n)))
    _atomic_pickle_dump(
        shard_path_abs,
        {
            "format": TIE_SHARD_FORMAT,
            "n": int(n),
            "obd_core_version": OBD_CORE_VERSION,
            "float_by_n": np.array([p for p, _ in recs], dtype=float),
            "float_with_pairs_by_n": recs,
            "tie_slope_by_n": slope_recs,
        },
    )
    t_write = time.perf_counter() - t_write0

    return {
        "n": int(n),
        "tie_point_count": len(recs),
        "local_min_count": len(cusp_ps),
        "max_local_min_p": (max(cusp_ps) if cusp_ps else None),
        "n_checked": int(stats["n_checked"]),
        "n_unresolved": int(stats["n_unresolved"]),
        "compute_sec": float(t_compute),
        "write_sec": float(t_write),
        "shard_path_abs": shard_path_abs,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_tie_points(
    n_list: list[int] | None = None,
    path: str = DEFAULT_TIE_OUTPUT,
    shards_dir: str = DEFAULT_TIE_SHARDS_DIR,
    log_dir: str = LOG_DIR,
    save_every: int = 10,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> None:
    """Compute every tie point for each n and save it in sharded per-n pickle files.

    Writes one shard per n under ``shards_dir`` and updates a manifest pickle at ``path``.
    Saves the manifest every ``save_every`` computed n (plus a final save), and writes a
    timestamped CSV run log in ``log_dir`` with one row per n.

    A manifest in an older format is discarded and every n is rebuilt: older shards hold
    finite-difference slopes that are wrong (negative slope jumps, missed cusps).

    Manifest keys:
      format: schema/version marker for the sharded layout.
      shards_dir: absolute path to shard directory.
      n_entries[str(n)]: metadata for each available shard (``local_min`` = number of cusps).
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

    if not isinstance(manifest, dict):
        manifest = {}
    if manifest.get("format") != TIE_MANIFEST_FORMAT:
        if manifest.get("format"):
            print(
                f"Tie manifest {path!r} is format {manifest.get('format')!r}; "
                f"rebuilding every n as {TIE_MANIFEST_FORMAT!r}."
            )
        manifest = {
            "format": TIE_MANIFEST_FORMAT,
            "created_at": run_started_at,
            "shards_dir": os.path.abspath(shards_dir),
            "n_entries": {},
        }
    manifest.setdefault("n_entries", {})
    manifest["shards_dir"] = os.path.abspath(shards_dir)
    manifest["obd_core_version"] = OBD_CORE_VERSION
    n_entries = manifest["n_entries"]

    if not ns:
        print("save_tie_points: empty n_list, nothing to do.")
        return
    if save_every < 1:
        raise ValueError("save_every must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

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
        "n_checked",
        "n_unresolved",
    ]
    with open(csv_log_path, "w", newline="") as csv_f:
        csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields, lineterminator="\n")
        csv_writer.writeheader()

        t_total = time.perf_counter()
        n_skipped = 0
        n_computed = 0
        n_since_save = 0
        n_manifest_writes = 0
        n_unresolved_total = 0
        manifest_dirty = False
        pending_ns: list[int] = []

        for n in ns:
            entry = n_entries.get(str(int(n)))
            shard_ok = False
            if isinstance(entry, dict):
                shard_ref = str(entry.get("shard_path", ""))
                if shard_ref:
                    shard_ok = os.path.exists(_resolve_manifest_shard_path(path, shard_ref))
            if shard_ok:
                n_skipped += 1
            else:
                pending_ns.append(int(n))

        if verbose and pending_ns:
            print("n     compute_sec  write_sec  tie_points  cusps  checked  max_cusp_p")
            print("-" * 72)

        def _finalize_one(result: dict) -> None:
            nonlocal n_computed, n_since_save, n_manifest_writes, manifest_dirty, n_unresolved_total
            n_val = int(result["n"])
            max_local_min_p = result["max_local_min_p"]
            t_compute = float(result["compute_sec"])
            t_write = float(result["write_sec"])
            n_unresolved = int(result["n_unresolved"])
            n_unresolved_total += n_unresolved
            if n_unresolved:
                print(
                    f"WARNING: n={n_val}: {n_unresolved} tie point(s) UNRESOLVED even at 200 digits; "
                    "they are stored as is_cusp=False."
                )

            manifest_parent = os.path.dirname(os.path.abspath(path)) or "."
            n_entries[str(n_val)] = {
                "n": n_val,
                "shard_path": os.path.relpath(str(result["shard_path_abs"]), start=manifest_parent),
                "tie_points": int(result["tie_point_count"]),
                "local_min": int(result["local_min_count"]),
                "max_local_min_p": (float(max_local_min_p) if max_local_min_p is not None else None),
                "n_checked": int(result["n_checked"]),
                "n_unresolved": n_unresolved,
                "updated_at": str(result["updated_at"]),
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
                    "iter_total_sec": f"{t_compute + t_write:.6f}",
                    "tie_points": int(result["tie_point_count"]),
                    "local_min": int(result["local_min_count"]),
                    "max_local_min_p": (
                        f"{float(max_local_min_p):.12f}" if max_local_min_p is not None else ""
                    ),
                    "n_checked": int(result["n_checked"]),
                    "n_unresolved": n_unresolved,
                }
            )
            csv_f.flush()

            if verbose:
                max_p_str = f"{float(max_local_min_p):.6f}" if max_local_min_p is not None else "none"
                print(
                    f"{n_val:5d} {t_compute:11.4f}s {t_write:9.4f}s {int(result['tie_point_count']):11d} "
                    f"{int(result['local_min_count']):6d} {int(result['n_checked']):8d}  {max_p_str}",
                    flush=True,
                )

            if n_since_save >= save_every:
                t_manifest_write = _checkpoint_write_manifest()
                n_since_save = 0
                n_manifest_writes += 1
                manifest_dirty = False
                if not verbose:
                    print(
                        "Checkpoint tie manifest write: "
                        f"n={n_val}, computed={n_computed}, skipped={n_skipped}, "
                        f"writes={n_manifest_writes}, write_sec={t_manifest_write:.4f}",
                        flush=True,
                    )

        if pending_ns:
            if workers == 1:
                for n in pending_ns:
                    _finalize_one(_compute_tie_shard(n=n, shards_dir_abs=shards_dir_abs))
            else:
                # largest n first, so the slowest shards do not start last
                order = sorted(pending_ns, reverse=True)
                with cf.ProcessPoolExecutor(max_workers=int(workers)) as executor:
                    futures = [
                        executor.submit(_compute_tie_shard, n=int(n), shards_dir_abs=shards_dir_abs)
                        for n in order
                    ]
                    for fut in cf.as_completed(futures):
                        _finalize_one(fut.result())

        if manifest_dirty:
            _checkpoint_write_manifest()
            n_manifest_writes += 1

        elapsed = time.perf_counter() - t_total
        if n_computed == 0 and n_skipped == len(ns):
            print(
                f"Tie points up to date ({len(ns)} n values, all skipped) - "
                f"manifest={path}, shards_dir={shards_dir}, log={csv_log_path}"
            )
        else:
            print(
                f"Wrote tie points shards ({n_computed} n computed, {n_skipped} skipped, "
                f"{n_manifest_writes} manifest writes, save_every={save_every}, "
                f"unresolved={n_unresolved_total}) to shards_dir={shards_dir} with manifest={path} "
                f"in {elapsed:.2f}s (log={csv_log_path})"
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
        help=(
            "Compute every tie point (with i,j pairs, exact slopes, log10 D and certified cusps, "
            "via obd_core) and save/update the tie shards."
        ),
    )
    p.add_argument(
        "--save-cusp-data",
        action="store_true",
        help=(
            "Build the cusp-only sidecar from the tie shards "
            f"(default output: {DEFAULT_CUSP_OUTPUT})."
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
            "Process workers for the tie-point computation; each worker computes one n "
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
            "Process workers for the cusp sidecar build; each worker reads one n "
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
        )
    else:
        if args.save_tie_points:
            save_tie_points(
                n_list=tie_ns,
                path=args.tie_output,
                shards_dir=args.tie_shards_dir,
                log_dir=args.tie_log_dir,
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
            )
