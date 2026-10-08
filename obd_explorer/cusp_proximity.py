"""How long p waits for a cusp: for each p, the smallest n with a cusp within distance r of p.

    N_r(p) = min { n : some cusp p* of n has |p* - p| <= r }

Read straight from the cusp table (``data/tie_cusps.parquet``, every certified cusp of every n), so
nothing is recomputed.  Each grid p is exact for the data: it is the first n, in increasing order,
whose sorted cusps (one ``searchsorted``) put one within r of p.  A p with no such n up to the
table's largest n is "not reached"; every cusp found so far has p* < 0.657 (ordered-binomial-cusps
FACTS S5), so beyond that plus r no n reaches p.
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
    p_max: float = 1.0
    p_steps: int | None = None          # None: spacing r/10 for the smallest r
    n_max: int | None = None            # None: every n in the cusp table
    log_n: bool = True
    marker_size: float = 2.0
    width_in: float = 12.0
    height_in: float = 7.0
    dpi: int = 300
    cusp_table: str | None = None
    output_path: str = "plots/N-pFirstCuspWithinR.png"


def first_n_within_r(cusp_n: np.ndarray, cusp_p: np.ndarray, grid: np.ndarray, r: float) -> np.ndarray:
    """For each grid p, the smallest n with a cusp within r of p (0 where no n in the table has one).

    ``cusp_n``/``cusp_p`` are the cusp table's columns, sorted by n then p.
    """
    first = np.zeros(grid.size, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(cusp_n)) + 1]
    ends = np.r_[starts[1:], cusp_n.size]
    for lo, hi in zip(starts, ends):
        todo = np.flatnonzero(first == 0)
        if todo.size == 0:
            break
        c = cusp_p[lo:hi]
        g = grid[todo]
        k = np.searchsorted(c, g)
        below = c[np.clip(k - 1, 0, c.size - 1)]
        above = c[np.clip(k, 0, c.size - 1)]
        dist = np.minimum(np.abs(g - below), np.abs(above - g))
        first[todo[dist <= r]] = int(cusp_n[lo])
    return first


def _grid(cfg: CuspProximityExportConfig) -> np.ndarray:
    steps = cfg.p_steps
    if steps is None:
        steps = int(math.ceil((cfg.p_max - cfg.p_min) / (min(cfg.r_values) / 10.0))) + 1
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

    from OBDsaveSourceData import DEFAULT_CUSP_OUTPUT, load_cusp_table

    path = cfg.cusp_table or DEFAULT_CUSP_OUTPUT
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found; build it with: python OBDsaveSourceData.py --save-cusp-data")
    table = load_cusp_table(path, columns=["n", "p"])
    cusp_n, cusp_p = table["n"].astype(np.int64), table["p"]
    if cfg.n_max is not None:
        keep = cusp_n <= cfg.n_max
        cusp_n, cusp_p = cusp_n[keep], cusp_p[keep]
    n_top = int(cusp_n.max())
    grid = _grid(cfg)

    r_sorted = sorted(cfg.r_values, reverse=True)   # largest r first: its n are lowest
    if len(r_sorted) == 1:
        colors = ["#1f5fa8"]
    else:
        cmap = plt.get_cmap("viridis")
        colors = [cmap(k / (len(r_sorted) - 1) * 0.9) for k in range(len(r_sorted))]

    fig, ax = plt.subplots(figsize=(cfg.width_in, cfg.height_in), facecolor=FIGURE_BACKGROUND)
    unreached_y = n_top * (1.6 if cfg.log_n else 1.06)
    for r, color in zip(r_sorted, colors):
        first = first_n_within_r(cusp_n, cusp_p, grid, r)
        hit = first > 0
        label = f"$r = {_format_r(r)}$"
        ax.scatter(grid[hit], first[hit], s=cfg.marker_size, lw=0, color=color, label=label, rasterized=True)
        if (~hit).any():
            pale = tuple(0.65 + 0.35 * v for v in to_rgb(color))   # opaque, so stacked points stay pale
            ax.scatter(grid[~hit], np.full((~hit).sum(), unreached_y), s=cfg.marker_size, lw=0, color=pale,
                       rasterized=True)
        if verbose:
            reached = grid[hit]
            print(f"r = {r:g}: {hit.sum()} of {grid.size} p reached by n <= {n_top}"
                  + (f" (largest n needed {first[hit].max()}, last p reached {reached.max():.6f})" if hit.any() else ""))

    if cfg.log_n:
        ax.set_yscale("log")
    ax.axhline(n_top, color="0.6", lw=0.6, ls=":")
    ax.text(cfg.p_max, unreached_y, f"  none up to n = {n_top}  ", ha="right", va="bottom", fontsize=9, color="0.35")
    ax.set_xlim(cfg.p_min - 0.005 * (cfg.p_max - cfg.p_min), cfg.p_max + 0.005 * (cfg.p_max - cfg.p_min))
    if cfg.log_n:
        ax.set_ylim(1.5, unreached_y * 1.8)
    else:
        ax.set_ylim(0, unreached_y * 1.05)
    ax.set_xlabel("p")
    ax.set_ylabel("first n with a cusp within r of p")
    r_text = ", ".join(f"${_format_r(r)}$" for r in sorted(cfg.r_values, reverse=True))
    ax.set_title(f"How far up n must go before a cusp comes within r of p  (r = {r_text}; "
                 f"{grid.size} values of p; cusps of n = 2–{n_top})", fontsize=11)
    if len(r_sorted) > 1:
        ax.legend(markerscale=4, loc="center right", frameon=False)
    ax.grid(True, which="major", color="0.9", lw=0.6)
    fig.tight_layout()
    fig.savefig(cfg.output_path, dpi=cfg.dpi, facecolor=FIGURE_BACKGROUND)
    plt.close(fig)
    if verbose:
        print(f"wrote {cfg.output_path}")
