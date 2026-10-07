import argparse
import concurrent.futures as cf
import csv
from datetime import datetime
import json
import math
import os
import sys
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
DEFAULT_TIE_SHARDS_DIR = os.path.join(DATA_DIR, "tie_points")
DEFAULT_TIE_MANIFEST_FILENAME = "manifest.json"
DEFAULT_TIE_OUTPUT = os.path.join(DEFAULT_TIE_SHARDS_DIR, DEFAULT_TIE_MANIFEST_FILENAME)
DEFAULT_CUSP_OUTPUT = os.path.join(DATA_DIR, "tie_cusps.parquet")
# Graph data: one HDF5 file per p grid, data/graph_data_p<steps>.h5 (the "graph manifest" path
# throughout the code is that file; the "graph shards dir" is the directory that holds it).
DEFAULT_GRAPH_SHARDS_DIR = DATA_DIR
DEFAULT_GRAPH_SHARDS_MANIFEST = os.path.join(
    DEFAULT_GRAPH_SHARDS_DIR, f"graph_data_p{DEFAULT_GRAPH_P_STEPS:05d}.h5"
)
DEFAULT_GRAPH_OUTPUT = DEFAULT_GRAPH_SHARDS_MANIFEST
LOG_DIR = "log"
DEFAULT_WORKERS = 8

# Tie-point storage.  Parquet since 2026-10-06 (one table per n plus a JSON manifest; before that,
# pickled lists of dicts).  The pickle formats are refused rather than read: v1 (before 2026-10-05)
# held finite-difference tie slopes that were wrong -- negative slope jumps at most tie points,
# missed and mislabelled cusps.
TIE_FORMAT = "obd.tie_points.parquet.v3"
CUSP_FORMAT = "obd.tie_cusps.parquet.v5"
# Graph data v4 (2026-10-07): one uncompressed HDF5 file per p grid, replacing a pickle per n with
# the same content (v3).  v3 (2026-10-06) took E from obd_core in float64 and dropped v2's
# np.gradient slope.  Uncompressed because on this machine's SSD decompression costs more than it
# saves: measured load of n=2..1000 0.78 s uncompressed vs 3.0 s (lzf) and 7.5 s (gzip).
GRAPH_FORMAT = "obd.graph_data.hdf5.v4"
TIE_COLUMNS: tuple[str, ...] = (
    "p", "i", "j", "E", "slope_left", "slope_right", "log10_D", "is_cusp", "decided_by",
)
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


def _tie_shard_filename_for_n(n: int) -> str:
    return os.path.join(f"n={int(n):05d}", "part.parquet")


def _graph_manifest_filename_for_p_steps(p_steps: int) -> str:
    return f"graph_data_p{int(p_steps):05d}.h5"


def _resolve_graph_manifest_path(
    manifest_path: str | None, p_steps: int | None, shards_dir: str = DEFAULT_GRAPH_SHARDS_DIR
) -> str:
    """Return the graph data file (HDF5). No directory scanning or alternate-``p_steps`` fallback.

    If ``manifest_path`` is set, it is used. Otherwise the path is
    ``<shards_dir>/graph_data_p{ps:05d}.h5`` with ``ps = p_steps`` or, when
    ``p_steps`` is omitted, ``DEFAULT_GRAPH_P_STEPS`` (1001, i.e. ``p01001``).
    If that file is missing, callers must fail; we do not look for another p grid.
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

# Symbolic tie math (all_tie_points_exact) is used only in print_comparison_table, not when saving data.


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


def _tie_table_for_n(n: int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Every tie point of ``n`` (both halves, sorted by ``p``) as columns, from ``obd_core``.

    Columns (see ``TIE_COLUMNS``):

    - ``p``: the tie point; ``i``, ``j``: its pair. The mirror pairs ``i + j == n`` share the one
      center tie ``p = 1/2``, which carries the canonical pair (``_canonical_center_pair_ij``).
    - ``E``: E(n, p) at the tie point.
    - ``slope_left``, ``slope_right``: the exact one-sided slopes E'_- and E'_+ (from the ranking
      just left of the tie, not finite differences).
    - ``log10_D``: log10 of the slope jump D = E'_+ - E'_- > 0, computed from the pair mass and so
      exact far below double range. Never recompute it as ``slope_right - slope_left``: for most
      tie points D is many orders smaller than the slopes and the difference is 0 or noise.
    - ``is_cusp``: certified local minimum of E (double-precision screen, interval arithmetic for
      anything borderline). A tie point is never a local maximum, so there is no other extremum.
    - ``decided_by``: ``double``, ``iv50``/``iv100``/``iv200``, ``exact`` (adjacent pairs, rational
      p*), ``symmetry`` (the center tie) or ``UNRESOLVED``.

    Also returns ``stats``: ``n_checked`` (tie points that needed interval or exact arithmetic) and
    ``n_unresolved``.
    """
    ni = int(n)
    t = obd_core.tie_table(ni, both_halves=True)
    i = t["i"].astype(np.int16)
    j = t["j"].astype(np.int16)
    center = (t["i"] == 0) & (t["j"] == ni)
    ci, cj = _canonical_center_pair_ij(ni)
    i[center] = ci
    j[center] = cj
    decided = np.asarray(t["decided_by"], dtype=object)
    table = {
        "p": t["pstar"].astype(np.float64),
        "i": i,
        "j": j,
        "E": t["E"].astype(np.float64),
        "slope_left": t["slope_left"].astype(np.float64),
        "slope_right": t["slope_right"].astype(np.float64),
        "log10_D": t["log10_D"].astype(np.float64),
        "is_cusp": t["is_cusp"].astype(bool),
        "decided_by": decided,
    }
    stats = {
        "n_checked": int(sum(1 for d in decided if d.startswith("iv") or d in ("exact", "UNRESOLVED"))),
        "n_unresolved": int(sum(1 for d in decided if d == "UNRESOLVED")),
    }
    return table, stats


def tie_center_index(table: dict[str, np.ndarray]) -> int:
    """Row index of the center tie ``p = 1/2`` in a per-n tie table (it is stored as exactly 0.5)."""
    hit = np.flatnonzero(table["p"] == 0.5)
    if hit.size != 1:
        raise ValueError(f"tie table has {hit.size} rows at p = 1/2; expected exactly one")
    return int(hit[0])


def _write_tie_parquet(path: str, n: int, table: dict[str, np.ndarray]) -> None:
    """One n's tie table as Parquet: zstd, BYTE_STREAM_SPLIT on the floats, no dictionary pages.

    Same settings as the ordered-binomial-cusps plotting datasets (dictionary encoding defeats
    BYTE_STREAM_SPLIT); ``decided_by`` is the one column stored as an Arrow dictionary.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {
        "p": pa.array(table["p"]),
        "i": pa.array(table["i"]),
        "j": pa.array(table["j"]),
        "E": pa.array(table["E"]),
        "slope_left": pa.array(table["slope_left"]),
        "slope_right": pa.array(table["slope_right"]),
        "log10_D": pa.array(table["log10_D"]),
        "is_cusp": pa.array(table["is_cusp"]),
        "decided_by": pa.array(list(table["decided_by"]), type=pa.string()).dictionary_encode(),
    }
    tbl = pa.table(cols).replace_schema_metadata(
        {b"format": TIE_FORMAT.encode(), b"n": str(int(n)).encode(), b"obd_core_version": OBD_CORE_VERSION.encode()}
    )
    _ensure_parent_dir(path)
    tmp = path + ".tmp"
    pq.write_table(
        tbl,
        tmp,
        compression="zstd",
        use_dictionary=False,
        use_byte_stream_split=[k for k in tbl.column_names if pa.types.is_floating(tbl.schema.field(k).type)],
    )
    os.replace(tmp, path)


def _read_tie_parquet(path: str, columns: list[str] | tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    tbl = pq.read_table(path, columns=list(columns) if columns is not None else None)
    return {name: _column_to_numpy(tbl.column(name)) for name in tbl.column_names}


def _column_to_numpy(col) -> np.ndarray:
    """Arrow column -> numpy; a dictionary (string) column becomes an object array via its codes."""
    import pyarrow as pa

    arr = col.combine_chunks()
    if pa.types.is_dictionary(arr.type):
        labels = np.asarray(arr.dictionary.to_pylist(), dtype=object)
        return labels[arr.indices.to_numpy(zero_copy_only=False)]
    return arr.to_numpy(zero_copy_only=False)


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


def _atomic_json_dump(path: str, payload: dict) -> None:
    _ensure_parent_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def load_tie_manifest(path: str = DEFAULT_TIE_OUTPUT) -> dict:
    """The tie-point manifest (JSON). Raises on anything but the current format."""
    if path.endswith(".pkl"):
        raise ValueError(
            f"{path!r} is a pickle tie manifest from before 2026-10-06; the tie points now live in "
            f"Parquet under {DEFAULT_TIE_SHARDS_DIR!r}. Rebuild with: python OBDsaveSourceData.py --save-tie-points"
        )
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict) or manifest.get("format") != TIE_FORMAT:
        fmt = manifest.get("format") if isinstance(manifest, dict) else None
        raise ValueError(f"Unsupported tie manifest format in {path!r}: {fmt!r} (expected {TIE_FORMAT!r}).")
    return manifest


def iter_tie_tables(
    path: str = DEFAULT_TIE_OUTPUT,
    n_list: list[int] | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
    require_all: bool = True,
    *,
    progress: int | None = None,
):
    """Yield ``(n, table)`` for each requested ``n``, one n at a time.

    ``table`` maps column name -> numpy array, rows sorted by ``p`` (see ``_tie_table_for_n`` for the
    columns). ``columns`` restricts what is read (Parquet reads only those columns). Missing ``n``
    raise when ``require_all``, else are skipped. ``progress=N`` prints timing every N values of n.
    """
    manifest = load_tie_manifest(path)
    n_entries = manifest.get("n_entries", {})
    target_ns = sorted(int(k) for k in n_entries) if n_list is None else [int(n) for n in n_list]
    if columns is not None:
        unknown = [c for c in columns if c not in TIE_COLUMNS]
        if unknown:
            raise ValueError(f"Unknown tie columns {unknown}; available: {TIE_COLUMNS}")
    every = int(progress) if progress else 0
    total = len(target_ns)
    t0 = time.perf_counter()
    yielded = 0
    for step, n in enumerate(target_ns, start=1):
        entry = n_entries.get(str(n))
        shard_path = _resolve_manifest_shard_path(path, str(entry.get("path", ""))) if isinstance(entry, dict) else ""
        if not shard_path or not os.path.isfile(shard_path):
            if require_all:
                raise FileNotFoundError(f"No tie table for n={n} in {path!r}.")
        else:
            yielded += 1
            yield n, _read_tie_parquet(shard_path, columns)
        if every and (step % every == 0 or step == total):
            print(
                f"[ties] n={n} step {step}/{total} loaded={yielded} elapsed {time.perf_counter() - t0:.2f}s",
                file=sys.stderr,
            )


def load_tie_tables(
    path: str = DEFAULT_TIE_OUTPUT,
    n_list: list[int] | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
    require_all: bool = True,
    *,
    progress: int | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """``{n: table}`` for the requested ``n`` (see ``iter_tie_tables``)."""
    return dict(iter_tie_tables(path, n_list, columns, require_all, progress=progress))


def load_graph_data(
    manifest_path: str | None = None,
    shards_dir: str = DEFAULT_GRAPH_SHARDS_DIR,
    p_steps: int | None = None,
    n_list: list[int] | None = None,
    require_all: bool = True,
) -> dict:
    """Load the graph data (HDF5) for the requested n and return grouped rows by n.

    ``manifest_path`` is the HDF5 file; if None it is ``<shards_dir>/graph_data_p<steps>.h5``.
    Returns ``format``, ``n_min``, ``n_max`` (of the file), ``p_steps``, ``p_half_start``,
    ``p_values`` (float32), ``rows_by_n[n]`` = {``y``, ``perm``, ``expected_sorted_by_p``} and
    ``manifest_path``.
    """
    import h5py

    path = _resolve_graph_manifest_path(manifest_path, p_steps, shards_dir)
    if not os.path.isfile(path):
        want_ps = int(p_steps) if p_steps is not None else DEFAULT_GRAPH_P_STEPS
        raise FileNotFoundError(
            f"Graph data file not found for p_steps={want_ps} (no other p grid is tried): "
            f"{os.path.abspath(path)}.  Build it with: python OBDsaveSourceData.py --save-graph-data"
        )
    with h5py.File(path, "r") as h:
        fmt = h.attrs.get("format")
        if fmt != GRAPH_FORMAT:
            raise ValueError(
                f"Unsupported graph data format in {path!r}: {fmt!r} (expected {GRAPH_FORMAT!r}). "
                "Rebuild it with: python OBDsaveSourceData.py --save-graph-data"
            )
        p_values = np.asarray(h["p_values"][()], dtype=np.float32)
        available = sorted(int(name[1:]) for name in h.keys() if name.startswith("n"))
        target_ns = available if n_list is None else [int(n) for n in n_list]
        rows_by_n: dict[int, dict[str, np.ndarray]] = {}
        for n in target_ns:
            g = h.get(_graph_group_name(n))
            if g is None:
                if require_all:
                    raise ValueError(f"No graph data for n={n} in {path!r}.")
                continue
            rows_by_n[n] = {k: g[k][()] for k in ("y", "perm", "expected_sorted_by_p")}
    ps = int(p_values.size)
    return {
        "format": GRAPH_FORMAT,
        "n_min": int(available[0]) if available else 0,
        "n_max": int(available[-1]) if available else -1,
        "p_steps": ps,
        "p_half_start": int((ps - 1) // 2),
        "p_values": p_values,
        "rows_by_n": rows_by_n,
        "manifest_path": path,
    }


def _graph_group_name(n: int) -> str:
    return f"n{int(n):05d}"


def load_cusp_table(
    path: str = DEFAULT_CUSP_OUTPUT,
    n_list: list[int] | None = None,
    columns: list[str] | tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """The cusp table written by ``save_cusp_table``: one row per certified cusp, all n.

    Columns: ``n``, ``tie_index`` (signed, relative to the center tie ``p = 1/2``, which is always a
    cusp) and the tie columns ``p, i, j, E, slope_left, slope_right, log10_D, decided_by``. Rows are
    sorted by ``n`` then ``p``. ``n_list`` keeps only those n.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    meta = pq.read_schema(path).metadata or {}
    fmt = meta.get(b"format", b"").decode()
    if fmt != CUSP_FORMAT:
        raise ValueError(
            f"Unsupported cusp table format in {path!r}: {fmt!r} (expected {CUSP_FORMAT!r}). "
            "Rebuild it with: python OBDsaveSourceData.py --save-cusp-data"
        )
    cols = list(columns) if columns is not None else None
    if cols is not None and n_list is not None and "n" not in cols:
        cols = ["n"] + cols
    tbl = pq.read_table(path, columns=cols)
    if n_list is not None:
        tbl = tbl.filter(pc.is_in(tbl.column("n"), value_set=pa.array([int(n) for n in n_list], type=pa.int16())))
    return {name: _column_to_numpy(tbl.column(name)) for name in tbl.column_names}


def save_cusp_table(
    tie_manifest_path: str = DEFAULT_TIE_OUTPUT,
    path: str = DEFAULT_CUSP_OUTPUT,
    n_list: list[int] | None = None,
    verbose: bool = False,
) -> None:
    """Write the cusp table: every certified cusp (``is_cusp``) of every n, from the tie tables.

    Nothing is recomputed; the values are copied from the tie tables.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not os.path.isfile(tie_manifest_path):
        print(
            "ERROR: the cusp table is built from the tie tables, and there is no tie manifest at "
            f"{os.path.abspath(tie_manifest_path)}.\n"
            "  Build them first:  python OBDsaveSourceData.py --save-tie-points\n"
            "  or everything:     python OBDsaveSourceData.py --all",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)
    t0 = time.perf_counter()
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ("n", "tie_index") + TIE_COLUMNS if k != "is_cusp"}
    n_done = 0
    for n, table in iter_tie_tables(tie_manifest_path, n_list, require_all=True, progress=(100 if verbose else None)):
        rows = np.flatnonzero(table["is_cusp"])
        center = tie_center_index(table)
        parts["n"].append(np.full(rows.size, n, dtype=np.int16))
        parts["tie_index"].append((rows - center).astype(np.int32))
        for k in TIE_COLUMNS:
            if k != "is_cusp":
                parts[k].append(table[k][rows])
        n_done += 1
    cols = {}
    for k, chunks in parts.items():
        arr = np.concatenate(chunks) if chunks else np.array([])
        cols[k] = pa.array(list(arr), type=pa.string()).dictionary_encode() if k == "decided_by" else pa.array(arr)
    tbl = pa.table(cols).replace_schema_metadata(
        {
            b"format": CUSP_FORMAT.encode(),
            b"source_tie_manifest": os.path.abspath(tie_manifest_path).encode(),
            b"created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode(),
            b"obd_core_version": OBD_CORE_VERSION.encode(),
        }
    )
    _ensure_parent_dir(path)
    tmp = path + ".tmp"
    pq.write_table(
        tbl,
        tmp,
        compression="zstd",
        use_dictionary=False,
        use_byte_stream_split=[k for k in tbl.column_names if pa.types.is_floating(tbl.schema.field(k).type)],
    )
    os.replace(tmp, path)
    print(
        f"Wrote cusp table ({tbl.num_rows} cusps over {n_done} n) to {path} "
        f"from {tie_manifest_path} in {time.perf_counter() - t0:.2f}s"
    )


def _compute_tie_shard(n: int, shards_dir_abs: str) -> dict:
    """Worker task: every tie point of one n (exact slopes, certified cusps), written as Parquet."""
    t0 = time.perf_counter()
    table, stats = _tie_table_for_n(int(n))
    cusp_ps = table["p"][table["is_cusp"]]
    t_compute = time.perf_counter() - t0

    t_write0 = time.perf_counter()
    shard_path_abs = os.path.join(shards_dir_abs, _tie_shard_filename_for_n(int(n)))
    _write_tie_parquet(shard_path_abs, int(n), table)
    t_write = time.perf_counter() - t_write0

    return {
        "n": int(n),
        "tie_point_count": int(table["p"].size),
        "cusp_count": int(cusp_ps.size),
        "max_cusp_p": (float(cusp_ps.max()) if cusp_ps.size else None),
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
    """Compute every tie point for each n and save one Parquet table per n.

    Tables go to ``<shards_dir>/n=NNNNN/part.parquet``; the JSON manifest at ``path`` lists them with
    per-n counts. The manifest is saved every ``save_every`` computed n (plus a final save), and a
    timestamped CSV run log goes to ``log_dir``. Any n already in the manifest with its file present
    is skipped; a manifest in another format is discarded and every n rebuilt.

    Manifest keys: ``format``, ``created_at``, ``updated_at``, ``obd_core_version``, and
    ``n_entries[str(n)]`` with ``n``, ``path`` (relative to the manifest), ``tie_points``,
    ``cusps``, ``max_cusp_p``, ``n_checked``, ``n_unresolved``, ``updated_at``.
    """
    ns = n_list if n_list is not None else N_LIST
    _ensure_parent_dir(path)
    os.makedirs(shards_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    shards_dir_abs = os.path.abspath(shards_dir)

    t_start_dt = datetime.now()
    run_started_at = t_start_dt.strftime("%Y-%m-%d %H:%M:%S")
    csv_log_path = os.path.join(log_dir, f"tie_points_verbose_{t_start_dt.strftime('%Y%m%d_%H%M%S')}.csv")

    try:
        manifest = load_tie_manifest(path)
    except FileNotFoundError:
        manifest = {}
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Tie manifest {path!r} is not usable ({e}); rebuilding every n.")
        manifest = {}
    if not manifest:
        manifest = {"format": TIE_FORMAT, "created_at": run_started_at, "n_entries": {}}
    manifest["obd_core_version"] = OBD_CORE_VERSION
    n_entries = manifest["n_entries"]

    if not ns:
        print("save_tie_points: empty n_list, nothing to do.")
        return
    if save_every < 1:
        raise ValueError("save_every must be at least 1")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    def _checkpoint_write_manifest() -> None:
        manifest["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _atomic_json_dump(path, manifest)

    pending_ns: list[int] = []
    n_skipped = 0
    for n in ns:
        entry = n_entries.get(str(int(n)))
        ok = isinstance(entry, dict) and os.path.isfile(_resolve_manifest_shard_path(path, str(entry.get("path", ""))))
        if ok:
            n_skipped += 1
        else:
            pending_ns.append(int(n))

    csv_fields = [
        "run_started_at", "n", "compute_sec", "write_sec", "iter_total_sec",
        "tie_points", "cusps", "max_cusp_p", "n_checked", "n_unresolved",
    ]
    with open(csv_log_path, "w", newline="") as csv_f:
        csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields, lineterminator="\n")
        csv_writer.writeheader()
        t_total = time.perf_counter()
        counters = {"computed": 0, "since_save": 0, "writes": 0, "unresolved": 0}
        manifest_parent = os.path.dirname(os.path.abspath(path)) or "."

        if verbose and pending_ns:
            print("n     compute_sec  write_sec  tie_points  cusps  checked  max_cusp_p")
            print("-" * 72)

        def _finalize_one(result: dict) -> None:
            n_val = int(result["n"])
            max_p = result["max_cusp_p"]
            n_unres = int(result["n_unresolved"])
            counters["unresolved"] += n_unres
            if n_unres:
                print(
                    f"WARNING: n={n_val}: {n_unres} tie point(s) UNRESOLVED even at 200 digits; "
                    "they are stored as is_cusp=False."
                )
            n_entries[str(n_val)] = {
                "n": n_val,
                "path": os.path.relpath(str(result["shard_path_abs"]), start=manifest_parent),
                "tie_points": int(result["tie_point_count"]),
                "cusps": int(result["cusp_count"]),
                "max_cusp_p": (float(max_p) if max_p is not None else None),
                "n_checked": int(result["n_checked"]),
                "n_unresolved": n_unres,
                "updated_at": str(result["updated_at"]),
            }
            counters["computed"] += 1
            counters["since_save"] += 1
            t_c, t_w = float(result["compute_sec"]), float(result["write_sec"])
            csv_writer.writerow(
                {
                    "run_started_at": run_started_at,
                    "n": n_val,
                    "compute_sec": f"{t_c:.6f}",
                    "write_sec": f"{t_w:.6f}",
                    "iter_total_sec": f"{t_c + t_w:.6f}",
                    "tie_points": int(result["tie_point_count"]),
                    "cusps": int(result["cusp_count"]),
                    "max_cusp_p": (f"{float(max_p):.12f}" if max_p is not None else ""),
                    "n_checked": int(result["n_checked"]),
                    "n_unresolved": n_unres,
                }
            )
            csv_f.flush()
            if verbose:
                max_p_str = f"{float(max_p):.6f}" if max_p is not None else "none"
                print(
                    f"{n_val:5d} {t_c:11.4f}s {t_w:9.4f}s {int(result['tie_point_count']):11d} "
                    f"{int(result['cusp_count']):6d} {int(result['n_checked']):8d}  {max_p_str}",
                    flush=True,
                )
            if counters["since_save"] >= save_every:
                _checkpoint_write_manifest()
                counters["since_save"] = 0
                counters["writes"] += 1
                if not verbose:
                    print(
                        f"Checkpoint tie manifest write: n={n_val}, computed={counters['computed']}, "
                        f"skipped={n_skipped}, writes={counters['writes']}",
                        flush=True,
                    )

        if pending_ns:
            if workers == 1:
                for n in pending_ns:
                    _finalize_one(_compute_tie_shard(n=n, shards_dir_abs=shards_dir_abs))
            else:
                # largest n first, so the slowest tables do not start last
                with cf.ProcessPoolExecutor(max_workers=int(workers)) as executor:
                    futures = [
                        executor.submit(_compute_tie_shard, n=int(n), shards_dir_abs=shards_dir_abs)
                        for n in sorted(pending_ns, reverse=True)
                    ]
                    for fut in cf.as_completed(futures):
                        _finalize_one(fut.result())

        if counters["since_save"] or not os.path.isfile(path):
            _checkpoint_write_manifest()
            counters["writes"] += 1

        elapsed = time.perf_counter() - t_total
        if counters["computed"] == 0 and n_skipped == len(ns):
            print(f"Tie points up to date ({len(ns)} n values, all skipped) - manifest={path}")
        else:
            print(
                f"Wrote tie tables ({counters['computed']} n computed, {n_skipped} skipped, "
                f"{counters['writes']} manifest writes, unresolved={counters['unresolved']}) "
                f"to {shards_dir} with manifest={path} in {elapsed:.2f}s (log={csv_log_path})"
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
    """Precompute the graph data for n_min..n_max into one uncompressed HDF5 file.

    ``path`` defaults to ``<shards_dir>/graph_data_p<steps>.h5``.  Layout: file attributes
    ``format``, ``p_steps``, ``created_at``, ``updated_at``, ``obd_core_version``; dataset
    ``p_values`` (float32); one group ``nNNNNN`` per n with ``y`` (p_steps x n+1 float32, the
    masses), ``perm`` (p_steps x n+1 uint16, their stable argsort) and ``expected_sorted_by_p``
    (float64, E from obd_core).  An n already in the file is skipped; a file in another format is
    rebuilt.  The file is flushed every ``save_every`` computed n.
    """
    import h5py

    if n_min > n_max:
        raise ValueError("n_min must be <= n_max")
    if p_steps < 2:
        raise ValueError("p_steps must be at least 2")
    if save_every < 1:
        raise ValueError("save_every must be at least 1")

    path = _resolve_graph_manifest_path(path, int(p_steps), shards_dir)
    _ensure_parent_dir(path)
    p_values = np.linspace(0.0, 1.0, int(p_steps), dtype=np.float32)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mode = "a"
    if os.path.exists(path):
        try:
            with h5py.File(path, "r") as h:
                ok = h.attrs.get("format") == GRAPH_FORMAT and int(h.attrs.get("p_steps", -1)) == int(p_steps)
        except OSError as e:
            print(f"WARNING: could not read {path!r} ({e}); rebuilding it.")
            ok = False
        if not ok:
            mode = "w"
    with h5py.File(path, mode) as h:
        if "format" not in h.attrs:
            h.attrs["format"] = GRAPH_FORMAT
            h.attrs["p_steps"] = int(p_steps)
            h.attrs["created_at"] = now
            h.create_dataset("p_values", data=p_values)
        h.attrs["obd_core_version"] = OBD_CORE_VERSION

        t_total = time.perf_counter()
        n_computed = 0
        n_skipped = 0
        if verbose:
            print("n      compute_sec  write_sec")
            print("-" * 32)
        for n in range(n_min, n_max + 1):
            name = _graph_group_name(n)
            if name in h:
                n_skipped += 1
                continue

            t0 = time.perf_counter()
            y_arr = np.empty((int(p_steps), int(n + 1)), dtype=np.float32)
            perm_arr = np.empty((int(p_steps), int(n + 1)), dtype=np.uint16)
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
            # E itself from obd_core (same masses and ranking as the tie tables), not re-derived here
            expected_sorted_arr, _, _ = obd_core.E_slopes_at(int(n), np.asarray(p_values, dtype=np.float64))
            t_compute = time.perf_counter() - t0

            t_write0 = time.perf_counter()
            g = h.create_group(name)
            g.attrs["n"] = int(n)
            g.attrs["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            g.create_dataset("y", data=y_arr)
            g.create_dataset("perm", data=perm_arr)
            g.create_dataset("expected_sorted_by_p", data=expected_sorted_arr)
            n_computed += 1
            if n_computed % save_every == 0:
                h.attrs["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                h.flush()
                if not verbose:
                    print(f"Checkpoint graph data: n={n}, computed={n_computed}, skipped={n_skipped}", flush=True)
            t_write = time.perf_counter() - t_write0
            if verbose:
                print(f"{n:5d}  {t_compute:11.4f}s {t_write:9.4f}s", flush=True)
        h.attrs["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elapsed = time.perf_counter() - t_total
    if n_computed == 0:
        print(f"Graph data up to date ({n_max - n_min + 1} n values, all present) in {path}")
    else:
        print(f"Wrote graph data ({n_computed} n computed, {n_skipped} skipped) to {path} in {elapsed:.2f}s")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Save the tie-point tables (Parquet), graph data shards and cusp table, and/or print tie "
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
            f"(writes graph data {DEFAULT_GRAPH_OUTPUT} and tie manifest {DEFAULT_TIE_OUTPUT}), "
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
            "Build the cusp table (every certified cusp of every n) from the tie tables "
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
        help=f"Tie-point manifest (JSON) path (default: {DEFAULT_TIE_OUTPUT}).",
    )
    p.add_argument(
        "--cusp-output",
        default=DEFAULT_CUSP_OUTPUT,
        metavar="PATH",
        help=f"Cusp table (Parquet) path (default: {DEFAULT_CUSP_OUTPUT}).",
    )
    p.add_argument(
        "--tie-shards-dir",
        default=DEFAULT_TIE_SHARDS_DIR,
        metavar="DIR",
        help=f"Directory for the per-n tie tables, n=NNNNN/part.parquet (default: {DEFAULT_TIE_SHARDS_DIR}).",
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
        help="Precompute the graph data (masses, their ranking, E on a p grid) into one HDF5 file.",
    )
    p.add_argument(
        "--graph-output",
        default=None,
        metavar="PATH",
        help=(
            "Graph data file (HDF5). "
            f"Default is auto-derived from p_steps, e.g. {os.path.join(DEFAULT_GRAPH_SHARDS_DIR, _graph_manifest_filename_for_p_steps(DEFAULT_GRAPH_P_STEPS))}."
        ),
    )
    p.add_argument(
        "--graph-shards-dir",
        default=DEFAULT_GRAPH_SHARDS_DIR,
        metavar="DIR",
        help=f"Directory holding the graph data file (default: {DEFAULT_GRAPH_SHARDS_DIR}).",
    )
    p.add_argument(
        "--graph-save-every",
        type=int,
        default=20,
        metavar="K",
        help=(
            "Flush cadence for the graph data file: flush after every K computed n values "
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
        save_cusp_table(
            tie_manifest_path=args.tie_output,
            path=args.cusp_output,
            n_list=tie_ns_explorer,
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
            save_cusp_table(
                tie_manifest_path=args.tie_output,
                path=args.cusp_output,
                n_list=tie_ns,
                verbose=args.verbose,
            )
