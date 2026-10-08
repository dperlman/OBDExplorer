"""How long p waits for a cusp: for each p, the smallest n with a cusp within distance r of p.

    N_r(p) = min { n : some cusp p* of n has |p* - p| <= r }

Read straight from the cusp table (``data/tie_cusps.parquet``, every certified cusp of every n), so
nothing is recomputed.  Each grid p is exact for the data: the smallest n of the cusps within r of it.  A p with no such n up to the
largest n available is "not reached".  Every cusp found so far has p* < 0.657 (ordered-binomial-cusps
FACTS S5, n <= 5000), so p_max defaults to 0.657.

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
    p_max: float = 0.657
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

    cusp_n, cusp_p, n_table = load_cusps(cfg)
    n_top = int(cusp_n.max())
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
        first = first_n_within_r(cusp_n, cusp_p, grid, r)
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
    ax.set_ylabel(f"(first n with a cusp within r of p) × {factor}" if scaled
                  else "first n with a cusp within r of p")
    r_text = ", ".join(f"${_format_r(r)}$" for r in sorted(cfg.r_values, reverse=True))
    pts = (f"{n_points[r_sorted[0]]} values of p" if len(r_sorted) == 1
           else f"p spacing r/{cfg.points_per_r:g}" if cfg.p_steps is None else f"{cfg.p_steps} values of p")
    head = (f"First n with a cusp within r of p, times {factor}" if scaled
            else "How far up n must go before a cusp comes within r of p")
    ax.set_title(f"{head}  (r = {r_text}; {pts}; cusps of n = 2–{n_top})"
                 + (f"\ncusps for n ≤ {n_table}: OBD tie tables; n = {n_table + 1}–{n_top}: ordered-binomial-cusps "
                    "catalogue (identical for n ≤ 1000)" if n_top > n_table else "")
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
