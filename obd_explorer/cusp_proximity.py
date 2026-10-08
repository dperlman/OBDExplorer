"""How long p waits for a cusp: for each p, the smallest n with a cusp within distance r of p.

    N_r(p) = min { n : some cusp p* of n has |p* - p| <= r }

Read straight from the cusp table (``data/tie_cusps.parquet``, every certified cusp of every n), so
nothing is recomputed.  Each grid p is exact for the data: the smallest n of the cusps within r of it.  A p with no such n up to the
largest n available is "not reached".  Every cusp found so far has p* < 0.657 (ordered-binomial-cusps
FACTS S5, n <= 5000), so p_max defaults to 0.657.

``extend_to`` = N carries the search past the tables, up to n = N, with OBD-core's windowed tie
table (``tie_table(n, p_range=...)``, obd-core >= 0.6.0): for each further n it computes only the
tie points within r of the p not yet reached, so the cost follows what is left, not the whole
band.  Blocks of consecutive n run in parallel; each p keeps the smallest n that reaches it, so
the result is the same as a sequential search.  It searches p <= extend_p_max only (default
0.6525: past n ~ 1250 every cusp found so far has p* < 0.6525, FACTS S5, so the edge above would
cost the most and find nothing).  ``min_pair_mass`` is passed through: faster, not proved complete.

The search is cached in ``window_cache`` (default ``data/cusp_windows/``, see WindowCache): for
each n, the p intervals already searched and the cusps found in them.  A later run computes only
what its windows add, so rerunning a plot takes seconds and an interrupted run resumes.

``points = "ties"`` asks the same about tie points instead of cusps: the first n with ANY tie point
within r of p.  For n <= 1000 it streams OBD's tie tables (data/tie_points, read one n at a time,
p column only); past them the windowed search keeps every tie point found, not just the cusps
(cache data/tie_windows).  Tie points fill (1/2, 1) about n^2/4 per n, so most p are reached by
n ~ (3 ln 2 / r)^(1/3); the slow places are the two edges, above 1/2 (the first tie point is at
1/2 + ~1/(2(n+1))) and below 1 (the last is at n/(n+1)).

``r_power`` = k plots N_r(p) * r^k instead.  With k = 1/2: across the band the median is about 0.53 at every r
from 1e-3 to 1e-6, what cusps scattered at random with density ~2.3 per unit p per n would give
(median sqrt(ln 2 / (2.3 r))), so several r collapse onto one picture and what departs from that
(the spikes at simple fractions, the edge near 1/2) stands out.  Those two grow like 1/r instead
(near 1/2 the first cusp is at about 1/2 + 1/(2(n+1))), which k = 1 lines up.

``cusps_csv`` extends the data past the cusp table with ordered-binomial-cusps' certified catalogue
``cusps/cusps_all.csv`` (n = 3..5000; columns ``n`` and ``pstar``, the cusps with p* > 1/2): its rows
for n above the cusp table's last n are added, with their mirror images 1 - p* and the center cusp
p = 1/2.  For n <= 1000 the two hold exactly the same cusps with bit-identical p* (checked
2026-10-07).
"""

from __future__ import annotations

import dataclasses
import math
import os

import numpy as np

from obd_explorer.constants import FIGURE_BACKGROUND


@dataclasses.dataclass
class CuspProximityExportConfig:
    r_values: tuple[float, ...] = (0.001,)
    p_min: float = 0.5
    p_max: float = 0.657                # 1.0 for points="ties"
    points: str = "cusps"               # "cusps" or "ties": which points p waits for
    p_steps: int | None = None          # None: spacing r / points_per_r, separately for each r
    points_per_r: float = 10.0
    r_power: float = 0.0                # plot N_r(p) * r**r_power
    n_max: int | None = None            # None: every n in the cusp table
    log_n: bool = True
    marker_size: float = 2.0
    width_in: float = 12.0
    height_in: float = 7.0
    dpi: int = 300
    cusp_table: str | None = None
    cusps_csv: str | None = None        # ordered-binomial-cusps' cusps_all.csv, for n past the table
    extend_to: int | None = None        # search on with windowed tie tables up to this n
    extend_p_max: float = 0.6525        # ... but only for p up to this (1.0 for points="ties")
    min_pair_mass: float | None = None  # passed to tie_table: faster, not proved complete
    workers: int = 8
    window_cache: str | None = os.path.join("data", "cusp_windows")   # None: no cache
    output_path: str = "plots/N-pFirstCuspWithinR.png"


def first_n_within_r(cusp_n: np.ndarray, cusp_p: np.ndarray, grid: np.ndarray, r: float) -> np.ndarray:
    """For each grid p (sorted), the smallest n with a cusp within r of p (0 where no cusp has one).

    Each cusp marks the grid points in [p* - r, p* + r] and every point keeps the smallest n that
    marks it; a cusp covers at most 2 r / (grid spacing) + 1 points, so this is one pass per offset.
    """
    lo = np.searchsorted(grid, cusp_p - r, side="left")
    hi = np.searchsorted(grid, cusp_p + r, side="right")
    keep = hi > lo
    lo, hi, nn = lo[keep], hi[keep], cusp_n[keep].astype(np.int64)
    none = np.iinfo(np.int64).max
    best = np.full(grid.size, none, dtype=np.int64)
    for k in range(int((hi - lo).max()) if lo.size else 0):
        m = lo + k < hi
        np.minimum.at(best, lo[m] + k, nn[m])
    best[best == none] = 0
    return best


CACHE_FORMAT = "obd.cusp_windows.parquet.v1"


def _window_cusps(task):
    """The cusps (points="cusps") or all tie points ("ties") with p* > 1/2 of one n inside the
    windows: one windowed tie table for all of them.  Returns (n, windows, i, j, pstar, decided_by)."""
    import obd_core

    n, windows, min_pair_mass, points = task
    t = obd_core.tie_table(n, p_range=windows, min_pair_mass=min_pair_mass)
    c = t["is_cusp"] if points == "cusps" else np.ones(len(t["i"]), bool)
    return n, windows, t["i"][c], t["j"][c], t["pstar"][c], t["decided_by"][c].astype(str)


def _merge(intervals) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _subtract(wins, cov) -> list[tuple[float, float]]:
    """The parts of the sorted, disjoint intervals ``wins`` not inside the sorted, disjoint ``cov``."""
    out, k = [], 0
    for a, b in wins:
        cur = a
        while k < len(cov) and cov[k][1] < cur:
            k += 1
        t = k
        while t < len(cov) and cov[t][0] <= b and cur < b:
            if cov[t][0] > cur:
                out.append((cur, cov[t][0]))
            cur = max(cur, cov[t][1])
            t += 1
        if cur < b:
            out.append((cur, b))
    return out


class WindowCache:
    """Points found by windowed searches (``extend_first_n``), with the p intervals searched for each n.

    A directory of Parquet parts, one pair per block of n written:
        coverage/part-*.parquet   n (int32), lo, hi (float64), min_pair_mass (float64; 0 = complete
                                  search): every tie point of n with lo <= p* <= hi was examined
        cusps/part-*.parquet      n (int32), i, j (int32), pstar (float64), decided_by (string): the
        (ties/ for points="ties") certified cusps found there (every tie point, for "ties")
    Schema metadata: format (CACHE_FORMAT), obd_core_version, created_at.  Parts are written whole
    to a temporary name and renamed, so an interrupted run leaves only complete parts.

    Coverage from a search with min_pair_mass m counts only for runs with min_pair_mass >= m: a
    complete search (m = 0) serves every run, a filtered one never serves a complete run.  The
    cusps themselves are certified either way and are always used.
    """

    def __init__(self, path: str, min_pair_mass: float | None = None, points: str = "cusps"):
        import pyarrow.parquet as pq

        self.path = path
        self.sub = points                      # "cusps" or "ties": the subdirectory of found points
        self.tau = float(min_pair_mass or 0.0)
        self.cov: dict[int, list[list[float]]] = {}
        self.cusp_p: dict[int, np.ndarray] = {}
        self._pending: list[tuple] = []
        for sub in ("coverage", self.sub):
            os.makedirs(os.path.join(path, sub), exist_ok=True)
        raw: dict[int, list] = {}
        for name in sorted(os.listdir(os.path.join(path, "coverage"))):
            if not name.endswith(".parquet"):
                continue
            t = pq.read_table(os.path.join(path, "coverage", name))
            if (t.schema.metadata or {}).get(b"format", b"").decode() != CACHE_FORMAT:
                raise ValueError(f"{path}/coverage/{name}: not {CACHE_FORMAT}")
            n, lo, hi, m = (t[c].to_numpy() for c in ("n", "lo", "hi", "min_pair_mass"))
            ok = m <= self.tau
            for a, b, c in zip(n[ok].tolist(), lo[ok].tolist(), hi[ok].tolist()):
                raw.setdefault(a, []).append((b, c))
        self.cov = {k: _merge(v) for k, v in raw.items()}
        ps: dict[int, list] = {}
        for name in sorted(os.listdir(os.path.join(path, self.sub))):
            if not name.endswith(".parquet"):
                continue
            t = pq.read_table(os.path.join(path, self.sub, name), columns=["n", "pstar"])
            for a, b in zip(t["n"].to_numpy().tolist(), t["pstar"].to_numpy().tolist()):
                ps.setdefault(a, []).append(b)
        self.cusp_p = {k: np.unique(v) for k, v in ps.items()}

    def gaps(self, n: int, wins) -> list[tuple[float, float]]:
        return _subtract(wins, self.cov.get(n, []))

    def cusps_in(self, n: int, wins) -> np.ndarray:
        c = self.cusp_p.get(n)
        if c is None or c.size == 0:
            return np.empty(0)
        out = [c[np.searchsorted(c, a, "left"):np.searchsorted(c, b, "right")] for a, b in wins]
        return np.concatenate(out) if out else np.empty(0)

    def add(self, n, windows, i, j, pstar, decided_by) -> None:
        self.cov[n] = _merge(self.cov.get(n, []) + [list(w) for w in windows])
        if len(pstar):
            self.cusp_p[n] = np.unique(np.concatenate([self.cusp_p.get(n, np.empty(0)), pstar]))
        self._pending.append((n, windows, i, j, pstar, decided_by))

    def flush(self) -> None:
        if not self._pending:
            return
        import time
        from datetime import datetime

        import pyarrow as pa
        import pyarrow.parquet as pq
        from importlib.metadata import version

        meta = {"format": CACHE_FORMAT, "obd_core_version": version("obd-core"),
                "created_at": datetime.now().isoformat(timespec="seconds")}
        cov_rows = [(n, a, b) for n, w, *_ in self._pending for a, b in w]
        cov = pa.table({"n": pa.array([r[0] for r in cov_rows], pa.int32()),
                        "lo": pa.array([r[1] for r in cov_rows], pa.float64()),
                        "hi": pa.array([r[2] for r in cov_rows], pa.float64()),
                        "min_pair_mass": pa.array([self.tau] * len(cov_rows), pa.float64())})
        cus = pa.table({"n": pa.array(np.concatenate([np.full(len(r[4]), r[0]) for r in self._pending]), pa.int32()),
                        "i": pa.array(np.concatenate([r[2] for r in self._pending]), pa.int32()),
                        "j": pa.array(np.concatenate([r[3] for r in self._pending]), pa.int32()),
                        "pstar": pa.array(np.concatenate([r[4] for r in self._pending]), pa.float64()),
                        "decided_by": pa.array(np.concatenate([r[5] for r in self._pending]).astype(str), pa.string())})
        ns = [r[0] for r in self._pending]
        stem = f"part-{min(ns):06d}-{max(ns):06d}-{time.time_ns()}"
        # cusps first: coverage without its cusps would hide them, cusps without coverage are harmless
        for sub, tbl in ((self.sub, cus), ("coverage", cov)):
            final = os.path.join(self.path, sub, stem + ".parquet")
            pq.write_table(tbl.replace_schema_metadata(meta), final + ".tmp", compression="zstd")
            os.replace(final + ".tmp", final)
        self._pending = []


def _windows(points: np.ndarray, r: float, p_floor: float) -> list[tuple[float, float]]:
    """[p - r, p + r] around each (sorted) point, merged where they overlap or nearly touch."""
    out: list[list[float]] = []
    for p in points.tolist():
        lo, hi = max(p - r, p_floor), p + r
        if out and lo <= out[-1][1] + 2 * r:
            out[-1][1] = hi
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


def extend_first_n(first: np.ndarray, grid: np.ndarray, r: float, n_from: int, n_to: int, *,
                   p_max: float, min_pair_mass: float | None = None, workers: int = 8,
                   cache: str | None = None, points: str = "cusps", verbose: bool = False) -> np.ndarray:
    """Fill in ``first`` (0 = not reached) for n = n_from+1 .. n_to, searching only near the p not
    yet reached (p <= p_max), with windowed tie tables.  ``cache``: a WindowCache directory, read
    for what earlier runs searched and appended to after every block.  Returns the updated copy."""
    import time
    from concurrent.futures import ProcessPoolExecutor

    first = first.copy()
    block = 4 * workers
    t0 = time.time()
    wc = WindowCache(cache, min_pair_mass, points) if cache else None
    computed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        n = n_from + 1
        while n <= n_to:
            todo = np.flatnonzero((first == 0) & (grid <= p_max) & (grid > 0.5 + r))
            if todo.size == 0:
                break
            wins = _windows(grid[todo], r, 0.5)
            ns = list(range(n, min(n + block, n_to + 1)))
            tasks = [(k, wc.gaps(k, wins) if wc else wins, min_pair_mass, points) for k in ns]
            tasks = [t for t in tasks if t[1]]
            found = {k: np.empty(0) for k in ns}
            for res in ex.map(_window_cusps, tasks):
                computed += 1
                if wc:
                    wc.add(*res)
                else:
                    found[res[0]] = res[4]
            if wc:
                wc.flush()
                found = {k: wc.cusps_in(k, wins) for k in ns}
            for m in ns:
                cps = found[m]
                if cps.size == 0:
                    continue
                left = np.flatnonzero(first == 0)
                g = grid[left]
                k = np.searchsorted(np.sort(cps), g)
                c = np.sort(cps)
                d = np.minimum(np.abs(g - c[np.clip(k - 1, 0, c.size - 1)]), np.abs(c[np.clip(k, 0, c.size - 1)] - g))
                first[left[d <= r]] = m
            if verbose and (ns[0] - n_from - 1) % (block * 25) == 0:
                print(f"  r = {r:g}: n = {ns[-1]}, {todo.size} p still open in {len(wins)} windows, "
                      f"{computed} n computed (rest from the cache), {time.time() - t0:.0f}s")
            n = ns[-1] + 1
    return first


def first_n_from_tie_tables(grid: np.ndarray, r: float, n_max: int | None = None,
                            manifest: str | None = None) -> tuple[np.ndarray, int]:
    """For each grid p, the smallest n <= n_max with a tie point within r of p (0: none), streaming
    OBD's tie tables one n at a time (p column only).  Returns (first, the last n read)."""
    from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, iter_tie_tables, load_tie_manifest

    man = manifest or DEFAULT_TIE_OUTPUT
    ns = sorted(int(k) for k in load_tie_manifest(man)["n_entries"])
    ns = [n for n in ns if n_max is None or n <= n_max]
    first = np.zeros(grid.size, np.int64)
    for n, tab in iter_tie_tables(man, n_list=ns, columns=["p"]):
        todo = np.flatnonzero(first == 0)
        if todo.size == 0:
            break
        c, x = tab["p"], grid[todo]
        k = np.searchsorted(c, x)
        d = np.minimum(np.abs(x - c[np.clip(k - 1, 0, c.size - 1)]), np.abs(c[np.clip(k, 0, c.size - 1)] - x))
        first[todo[d <= r]] = n
    return first, ns[-1]


def load_cusps(cfg: CuspProximityExportConfig) -> tuple[np.ndarray, np.ndarray, int]:
    """Every cusp (n, p) available, sorted by n then p, and the cusp table's last n."""
    from OBDsaveSourceData import DEFAULT_CUSP_OUTPUT, load_cusp_table

    path = cfg.cusp_table or DEFAULT_CUSP_OUTPUT
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found; build it with: python OBDsaveSourceData.py --save-cusp-data")
    table = load_cusp_table(path, columns=["n", "p"])
    cusp_n, cusp_p = table["n"].astype(np.int64), table["p"]
    n_table = int(cusp_n.max())
    if cfg.cusps_csv:
        import pyarrow.csv as pacsv

        csv = pacsv.read_csv(cfg.cusps_csv, convert_options=pacsv.ConvertOptions(include_columns=["n", "pstar"]))
        xn, xp = csv["n"].to_numpy().astype(np.int64), csv["pstar"].to_numpy()
        new = xn > n_table
        xn, xp = xn[new], xp[new]
        centers = np.unique(xn)
        cusp_n = np.concatenate([cusp_n, xn, xn, centers])
        cusp_p = np.concatenate([cusp_p, xp, 1.0 - xp, np.full(centers.size, 0.5)])
        order = np.lexsort((cusp_p, cusp_n))
        cusp_n, cusp_p = cusp_n[order], cusp_p[order]
    if cfg.n_max is not None:
        keep = cusp_n <= cfg.n_max
        cusp_n, cusp_p = cusp_n[keep], cusp_p[keep]
    return cusp_n, cusp_p, n_table


def _grid(cfg: CuspProximityExportConfig, r: float) -> np.ndarray:
    steps = cfg.p_steps
    if steps is None:
        steps = int(math.ceil((cfg.p_max - cfg.p_min) / (r / cfg.points_per_r) - 1e-9)) + 1
    return np.linspace(cfg.p_min, cfg.p_max, steps)


def _format_r(r: float) -> str:
    e = math.log10(r)
    if abs(e - round(e)) < 1e-9:
        return f"10^{{{int(round(e))}}}"
    return f"{r:g}"


def export_cusp_proximity(cfg: CuspProximityExportConfig, verbose: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    ties = cfg.points == "ties"
    word = "tie point" if ties else "cusp"
    if ties:
        from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, load_tie_manifest

        avail = [int(k) for k in load_tie_manifest(DEFAULT_TIE_OUTPUT)["n_entries"]]
        n_table = n_data = max(k for k in avail if cfg.n_max is None or k <= cfg.n_max)
    else:
        cusp_n, cusp_p, n_table = load_cusps(cfg)
        n_data = int(cusp_n.max())
    n_top = max(n_data, cfg.extend_to or 0)
    r_min = min(cfg.r_values)

    # Smallest r first: it has the most points; the sparser grids of larger r draw on top of it.
    r_sorted = sorted(cfg.r_values)
    if len(r_sorted) == 1:
        colors = ["#1f5fa8"]
    else:
        cmap = plt.get_cmap("viridis")
        colors = [cmap(0.9 * (1 - k / (len(r_sorted) - 1))) for k in range(len(r_sorted))]
    # With r / points_per_r spacing, a larger r has fewer points; larger marks keep it visible.
    sizes = {r: cfg.marker_size * math.sqrt(r / r_min) for r in r_sorted} if cfg.p_steps is None \
        else {r: cfg.marker_size for r in r_sorted}
    scaled = cfg.r_power != 0
    scale = {r: r ** cfg.r_power for r in r_sorted}
    ceiling = {r: n_top * scale[r] for r in r_sorted}           # n_top in plotted units

    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in), facecolor=FIGURE_BACKGROUND)
    top = max(ceiling.values())
    unreached_y = top * (1.6 if cfg.log_n else 1.06)
    any_unreached = False
    n_points = {}
    medians = []
    for r, color in zip(r_sorted, colors):
        grid = _grid(cfg, r)
        n_points[r] = grid.size
        if ties:
            first, _ = first_n_from_tie_tables(grid, r, n_max=n_data)
        else:
            first = first_n_within_r(cusp_n, cusp_p, grid, r)
        if cfg.extend_to is not None and cfg.extend_to > n_data:
            first = extend_first_n(first, grid, r, n_data, cfg.extend_to, p_max=cfg.extend_p_max,
                                   min_pair_mass=cfg.min_pair_mass, workers=cfg.workers,
                                   cache=cfg.window_cache, points=cfg.points, verbose=verbose)
        hit = first > 0
        y = first[hit] * scale[r]
        ax.scatter(grid[hit], y, s=sizes[r], lw=0, color=color, rasterized=True)
        if hit.any():
            medians.append(float(np.median(y)))
        if (~hit).any():
            any_unreached = True
            pale = tuple(0.65 + 0.35 * v for v in to_rgb(color))   # opaque, so stacked points stay pale
            ax.scatter(grid[~hit], np.full((~hit).sum(), unreached_y), s=sizes[r], lw=0, color=pale,
                       rasterized=True)
        if scaled or len(r_sorted) == 1:
            ax.axhline(ceiling[r], color=color if len(r_sorted) > 1 else "0.6", lw=0.6, ls=":")
        if verbose:
            reached = grid[hit]
            print(f"r = {r:g}: {hit.sum()} of {grid.size} p reached by n <= {n_top}"
                  + (f" (largest n needed {first[hit].max()}, last p reached {reached.max():.6f}, "
                     f"median n*sqrt(r) {np.median(first[hit]) * math.sqrt(r):.3f})" if hit.any() else ""))

    if cfg.log_n:
        ax.set_yscale("log")
    if not scaled and len(r_sorted) > 1:
        ax.axhline(n_top, color="0.6", lw=0.6, ls=":")
    if cfg.r_power == 0.5 and medians:
        med = float(np.median(medians))
        ax.axhline(med, color="0.35", lw=0.7, ls="--")
        ax.text(cfg.p_max - 0.01 * (cfg.p_max - cfg.p_min), med, f"median ≈ {med:.2f}", ha="right", va="center",
                fontsize=9, color="0.2", bbox=dict(facecolor="white", edgecolor="0.6", lw=0.5, pad=2))
    if any_unreached:
        ax.text(cfg.p_min, unreached_y, f"  pale points: none up to n = {n_top}", ha="left", va="center",
                fontsize=9, color="0.35")
    ax.set_xlim(cfg.p_min - 0.005 * (cfg.p_max - cfg.p_min), cfg.p_max + 0.005 * (cfg.p_max - cfg.p_min))
    y_floor = 1.5 * r_min ** cfg.r_power
    if cfg.log_n:
        ax.set_ylim(y_floor, unreached_y * 1.8 if any_unreached else top * 1.3)
    else:
        ax.set_ylim(0, unreached_y * 1.05 if any_unreached else top * 1.02)
    ax.set_xlabel("p")
    factor = {0.5: "√r", 1.0: "r"}.get(cfg.r_power, f"r^{cfg.r_power:g}")
    ax.set_ylabel(f"(first n with a {word} within r of p) × {factor}" if scaled
                  else f"first n with a {word} within r of p")
    r_text = ", ".join(f"${_format_r(r)}$" for r in sorted(cfg.r_values, reverse=True))
    pts = (f"{n_points[r_sorted[0]]} values of p" if len(r_sorted) == 1
           else f"p spacing r/{cfg.points_per_r:g}" if cfg.p_steps is None else f"{cfg.p_steps} values of p")
    head = (f"First n with a {word} within r of p, times {factor}" if scaled
            else f"How far up n must go before a {word} comes within r of p")
    ax.set_title(f"{head}  (r = {r_text}; {pts}; {word}s of n = 2–{n_top})"
                 + (f"\ncusps for n ≤ {n_table}: OBD tie tables; n = {n_table + 1}–{n_data}: ordered-binomial-cusps "
                    "catalogue (identical for n ≤ 1000)" if n_data > n_table else "")
                 + (f"\ntie points for n ≤ {n_data}: OBD tie tables" if ties else "")
                 + (f"\nn = {n_data + 1}–{n_top}: windowed search near the p not yet reached, p ≤ {cfg.extend_p_max:g}"
                    + (f", pairs with f(i) ≥ {cfg.min_pair_mass:g} only" if cfg.min_pair_mass else "")
                    if n_top > n_data else "")
                 + ("\ndotted lines: n = " + str(n_top) + " for each r" if scaled and len(r_sorted) > 1 else ""),
                 fontsize=11)
    if len(r_sorted) > 1:
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], ls="", marker="o", ms=5, color=c, label=f"$r = {_format_r(r)}$")
                   for r, c in sorted(zip(r_sorted, colors), reverse=True)]
        ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.9)
    ax.grid(True, which="major", color="0.9", lw=0.6)
    fig.tight_layout()
    fig.savefig(cfg.output_path, dpi=cfg.dpi, facecolor=FIGURE_BACKGROUND)
    plt.close(fig)
    if verbose:
        print(f"wrote {cfg.output_path}")
