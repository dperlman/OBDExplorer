"""The first-n-within-r function over the whole (p, r) plane, as one image.

    FCW(p, r) = min { n : some point (cusp, or tie point) of n lies within r of p }

The cusp-proximity plots are horizontal slices of this at one r each.  Here every r at once: for
each sampled p, the closest approach so far, m_N(p) = min over n <= N of the distance from p to the
nearest point of n, only ever steps down as N grows, and FCW(p, r) is the first N with m_N(p) <= r.
So one pass over n fills every r: whenever m drops from m_old to m_new at n, the rows with
m_new <= r < m_old get FCW = n.  Each (p, r) cell is written once.

Colour: log10(FCW * r^alpha), the trend divided out -- alpha = 1/2 for cusps (median FCW ~ 0.53 /
sqrt(r)) and 1/3 for tie points (~ (3 ln 2 / r)^(1/3)) -- so the background is level and what departs
from it (the Farey spikes of the cusps, the edges) stands out.  Each pixel averages over ``samples``
values of p inside its column.  Grey: more than half of the column needs n beyond the data.

``points = "lag"`` combines the two: log10(FCW / FTW), where FTW is the same function for tie points.
Every cusp is a tie point, so FCW >= FTW: the lag is how many more n pass, after the first tie point
comes within r of p, before one that close is a cusp.  Its trend r^(-1/2) / r^(-1/3) = r^(-1/6) is
divided out (alpha = 1/6).

``points = "grid"`` is the same function for the bare grid of fractions k/(2(n+1)), which cusps sit
just above (RESEARCH_LOG fact 12): F(p, r) = the first n with a fraction k/(2(n+1)) within r of p,
pure arithmetic.  ``points = "residual"`` draws log10(FCW / F): where cusps follow the grid (near
0), where they reach p sooner than the grid alone (negative: off-grid offsets, several cusps per
grid point) and where later (positive: grid points that carry no cusp).  No trend to remove
(alpha = 0): both grow like r^(-1/2).

Data: cusps from the cusp table plus ordered-binomial-cusps' catalogue (n <= 5000, every cusp of
every n); tie points from OBD's tie tables (n <= 1000).  No windowed search: the record distance
needs every point of every n, which only the complete tables have.
"""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import numpy as np

from obd_explorer.constants import FIGURE_BACKGROUND


@dataclasses.dataclass
class ProximityMapConfig:
    points: str = "cusps"               # "cusps", "ties", "lag" (cusps over ties), "grid", "residual" (cusps over grid)
    p_min: float = 0.5
    p_max: float | None = None          # None: 0.657 for cusps, 1.0 for ties
    r_min: float = 1e-8
    r_max: float = 1e-2
    alpha: float | None = None          # None: 1/2 cusps and grid, 1/3 ties, 1/6 lag, 0 residual
    samples: int = 4                    # values of p per pixel column
    n_max: int | None = None            # None: all the complete data
    label_denominator: int | None = None  # label fractions a/b with b <= this (None: 12 cusps, 8 ties)
    colormap: str = "magma"
    width_in: float = 12.0
    height_in: float = 7.0
    dpi: int = 600
    cusp_table: str | None = None
    cusps_csv: str | None = None
    output_path: str = "plots/N-pFirstWithinRMap.png"


_MARGIN = {"left": 0.07, "right": 0.10, "bottom": 0.085, "top": 0.15}   # right: room for the colorbar


def _layout(cfg):
    fw, fh = int(round(cfg.width_in * cfg.dpi)), int(round(cfg.height_in * cfg.dpi))
    left, bottom = int(round(_MARGIN["left"] * fw)), int(round(_MARGIN["bottom"] * fh))
    w = fw - left - int(round(_MARGIN["right"] * fw))
    h = fh - bottom - int(round(_MARGIN["top"] * fh))
    return fw, fh, left, bottom, w, h


def _point_sets(cfg):
    """Yield (n, sorted p of every point of n) for n = 2, 3, ..., and finally the last n."""
    if cfg.points == "grid":
        # every fraction k/(2(n+1)) that can come within r_max of [p_min, p_max], n = 2..n_max
        n_max = cfg.n_max or 5000
        for n in range(2, n_max + 1):
            d = 2 * (n + 1)
            k = np.arange(math.floor((cfg.p_min - cfg.r_max) * d), math.ceil((cfg.p_max + cfg.r_max) * d) + 1)
            yield n, k / d
        return
    if cfg.points == "ties":
        from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, iter_tie_tables, load_tie_manifest

        ns = sorted(int(k) for k in load_tie_manifest(DEFAULT_TIE_OUTPUT)["n_entries"])
        ns = [n for n in ns if cfg.n_max is None or n <= cfg.n_max]
        for n, tab in iter_tie_tables(DEFAULT_TIE_OUTPUT, n_list=ns, columns=["p"]):
            yield n, tab["p"]
        return
    from obd_explorer.cusp_proximity import CuspProximityExportConfig, load_cusps

    cn, cp, _ = load_cusps(CuspProximityExportConfig(cusp_table=cfg.cusp_table, cusps_csv=cfg.cusps_csv,
                                                     n_max=cfg.n_max))
    starts = np.r_[0, np.flatnonzero(np.diff(cn)) + 1]
    ends = np.r_[starts[1:], cn.size]
    for a, b in zip(starts, ends):
        yield int(cn[a]), cp[a:b]


def compute_map(cfg, w: int, h: int, verbose: bool = False):
    """(mean log10 FCW per pixel, share of the column's samples reached per pixel, r of each row,
    the last n, the p samples).  Rows go up in r: row 0 is r_min."""
    import time

    s = cfg.samples
    pw = (cfg.p_max - cfg.p_min) / w
    x = np.arange(w * s)
    p = cfg.p_min + (x // s + ((x % s) + 0.5) / s) * pw
    col = x // s
    lr_min, lr_max = math.log10(cfg.r_min), math.log10(cfg.r_max)
    r_rows = 10 ** (lr_min + (np.arange(h) + 0.5) / h * (lr_max - lr_min))     # row centres
    sumlog = np.zeros(h * w)
    cnt = np.zeros(h * w)
    best = np.full(p.size, np.inf)                       # closest approach so far, per sample
    hi_row = np.full(p.size, h)                          # rows >= this are already filled
    t0 = time.time()
    n_last = 0
    for n, c in _point_sets(cfg):
        n_last = n
        k = np.searchsorted(c, p)
        d = np.minimum(np.abs(p - c[np.clip(k - 1, 0, c.size - 1)]), np.abs(c[np.clip(k, 0, c.size - 1)] - p))
        dec = np.flatnonzero(d < best)
        if dec.size == 0:
            continue
        best[dec] = d[dec]
        lo = np.searchsorted(r_rows, best[dec], "left")  # first row with r >= the new record
        hi = hi_row[dec]
        hi_row[dec] = np.minimum(hi, lo)
        L = hi - lo
        keep = L > 0
        if not keep.any():
            continue
        dec, lo, L = dec[keep], lo[keep], L[keep]
        tot = int(L.sum())
        offs = np.arange(tot) - np.repeat(np.cumsum(L) - L, L)
        rows = np.repeat(lo, L) + offs
        flat = rows * w + np.repeat(col[dec], L)
        np.add.at(sumlog, flat, math.log10(n))            # only the cells this n fills
        np.add.at(cnt, flat, 1.0)
        if verbose and n % 500 == 0:
            print(f"  n = {n}: {time.time() - t0:.0f}s")
    mean = np.where(cnt > 0, sumlog / np.maximum(cnt, 1), np.nan).reshape(h, w)
    return mean, (cnt / s).reshape(h, w), r_rows, n_last


def _fractions(lo: float, hi: float, bmax: int) -> list[Fraction]:
    out = {Fraction(a, b) for b in range(2, bmax + 1) for a in range(1, b) if lo <= a / b <= hi}
    return sorted(out)


def export_proximity_map(cfg: ProximityMapConfig, verbose: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FixedLocator, FuncFormatter

    ties = cfg.points == "ties"
    lag = cfg.points == "lag"
    resid = cfg.points == "residual"
    if cfg.p_max is None:
        cfg.p_max = 1.0 if ties else 0.657
    if cfg.alpha is None:
        cfg.alpha = 1 / 3 if ties else 1 / 6 if lag else 0.0 if resid else 1 / 2
    bmax = cfg.label_denominator or (8 if ties else 12)
    word = "tie point" if ties else "fraction k/(2(n+1))" if cfg.points == "grid" else "cusp"
    fw, fh, left, bottom, w, h = _layout(cfg)
    if lag:
        mc, rc, r_rows, n_last = compute_map(dataclasses.replace(cfg, points="cusps"), w, h, verbose)
        mt, rt, _, n_ties = compute_map(dataclasses.replace(cfg, points="ties"), w, h, verbose)
        mean, reached = mc - mt, np.minimum(rc, rt)          # mean log10(FCW) - mean log10(FTW)
    elif resid:
        mc, rc, r_rows, n_last = compute_map(dataclasses.replace(cfg, points="cusps"), w, h, verbose)
        mg, rg, _, _ = compute_map(dataclasses.replace(cfg, points="grid", n_max=cfg.n_max or n_last), w, h, verbose)
        mean, reached = mc - mg, np.minimum(rc, rg)          # mean log10(FCW) - mean log10(F)
    else:
        mean, reached, r_rows, n_last = compute_map(cfg, w, h, verbose)
    val = mean + cfg.alpha * np.log10(r_rows)[:, None]     # log10(FCW r^alpha), mean over the samples
    unknown = reached < 0.5
    ok = np.isfinite(val) & ~unknown
    vmin, vmax = np.quantile(val[ok], [0.005, 0.995])
    if resid:                                                # diverging, centred on "follows the grid"
        vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
    norm = Normalize(vmin, vmax)
    cmap = plt.get_cmap("RdBu_r" if resid and cfg.colormap == "magma" else cfg.colormap)
    rgb = cmap(norm(np.where(ok, val, vmin)))[..., :3]
    rgb[~ok] = (0.82, 0.82, 0.82)

    fig = plt.figure(figsize=(fw / cfg.dpi, fh / cfg.dpi), dpi=cfg.dpi, facecolor=FIGURE_BACKGROUND)
    ax = fig.add_axes([left / fw, bottom / fh, w / fw, h / fh])
    lr0, lr1 = math.log10(cfg.r_min), math.log10(cfg.r_max)
    ax.imshow(rgb[::-1], extent=[cfg.p_min, cfg.p_max, lr0, lr1], origin="upper", aspect="auto",
              interpolation="none")
    ax.set_xlim(cfg.p_min, cfg.p_max)
    ax.set_ylim(lr0, lr1)
    ax.yaxis.set_major_locator(FixedLocator(list(range(math.ceil(lr0), math.floor(lr1) + 1))))
    ax.yaxis.set_minor_locator(FixedLocator([e + math.log10(m) for e in range(math.floor(lr0), math.ceil(lr1))
                                             for m in range(2, 10) if lr0 <= e + math.log10(m) <= lr1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{int(round(v))}}}$"))
    ax.set_xlabel("p")
    ax.set_ylabel("r")
    top = ax.secondary_xaxis("top")
    fr = _fractions(cfg.p_min, cfg.p_max, bmax)
    top.set_xticks([float(f) for f in fr])
    top.set_xticklabels([f"{f.numerator}/{f.denominator}" for f in fr], fontsize=6, rotation=90)
    top.tick_params(length=3, pad=1)
    cax = fig.add_axes([(left + w + 0.015 * fw) / fw, bottom / fh, 0.012, h / fh])
    cb = fig.colorbar(ScalarMappable(norm, cmap), cax=cax)
    a_txt = {0.5: "½", 1 / 3: "⅓", 1 / 6: "⅙"}.get(min((0.5, 1 / 3, 1 / 6), key=lambda a: abs(a - cfg.alpha)))
    if min(abs(a - cfg.alpha) for a in (0.5, 1 / 3, 1 / 6)) > 1e-9:
        a_txt = f"{cfg.alpha:g}"
    if resid:
        cb.set_label("log₁₀(FCW / F)   (blue: cusps sooner than the grid; red: later)", fontsize=8)
        fig.suptitle(f"Cusps against the grid of fractions k/(2(n+1)): log₁₀(FCW / F), F = first n with a fraction "
                     f"k/(2(n+1)) within r of p\n"
                     f"{w * cfg.samples:,} values of p, {cfg.samples} per pixel column; cusps of n ≤ {n_last} "
                     f"(OBD cusp table and the ordered-binomial-cusps catalogue), fractions of the same n\n"
                     f"grey: more than half the column unreached by either; fractions with denominator ≤ {bmax} marked above",
                     fontsize=10, y=1 - 0.008)
    elif lag:
        cb.set_label(f"log₁₀(FCW / FTW · r^{a_txt})", fontsize=9)
        fig.suptitle(f"Cusp lag: first n with a cusp within r of p (FCW) over first n with any tie point within r (FTW)  "
                     f"(colour: log₁₀(FCW/FTW · r^{a_txt}))\n"
                     f"{w * cfg.samples:,} values of p, {cfg.samples} per pixel column; cusps of n ≤ {n_last} "
                     f"(OBD cusp table and the ordered-binomial-cusps catalogue), tie points of n ≤ {n_ties} (OBD tie tables)\n"
                     f"grey: more than half the column unreached by either; fractions with denominator ≤ {bmax} marked above",
                     fontsize=10, y=1 - 0.008)
    else:
        cb.set_label(f"log₁₀(first n · r^{a_txt})", fontsize=9)
        src = (f"{word}s of n ≤ {n_last}: OBD tie tables" if ties else
               f"cusps of n ≤ {n_last}: OBD cusp table (n ≤ 1000) and the ordered-binomial-cusps catalogue")
        fig.suptitle(f"First n with a {word} within r of p, every r at once  (colour: log₁₀(first n · r^{a_txt}))\n"
                     f"{w * cfg.samples:,} values of p, {cfg.samples} per pixel column; {src}\n"
                     f"grey: more than half the column needs n > {n_last}; fractions with denominator ≤ {bmax} marked above",
                     fontsize=10, y=1 - 0.008)
    fig.savefig(cfg.output_path, dpi=cfg.dpi, facecolor=FIGURE_BACKGROUND)
    plt.close(fig)
    if verbose:
        print(f"wrote {cfg.output_path} ({w} x {h} pixels; colour range {vmin:.2f}..{vmax:.2f}; "
              f"unknown {unknown.mean():.1%} of pixels)")
