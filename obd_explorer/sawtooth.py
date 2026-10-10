"""The sawtooth: E and its slope over a few grid steps, showing where cusps come from.

Tie points cluster just above the grid fractions k/(2(n+1)) (RESEARCH_LOG fact 12), ordered by width.
Walking right through a cluster, each tie point kicks E's slope up by its jump D, and the slope
climbs from negative to positive: the cusp is the kick that carries it across zero.  Between
clusters E is a smooth concave arc and the slope falls back: a smooth maximum.  So E has one dip
per grid fraction (below p ~ 0.652) and a hump between.  The wide cusps (width >= 2 sqrt(n), which
are exactly the F3 < 0 ones) are the exception: a tie point with a tiny jump landing just after the
slope has crossed zero going down, past a maximum, and flipping it back (RESEARCH_LOG 2026-10-09).

Three panels: E itself (untilted, so its minima and maxima are where they really are); a close-up of
one cluster, where E is visibly a polygon with a corner at every tie point; and E's slope, broken at
every jump of at least jump_min (the many smaller ones, from wide pairs with tiny masses, are far
below what the eye can see).  E and the slope come from obd_core.E_slopes_at on a
dense grid; the tie points and their certified
cusp verdicts from obd_core.tie_table(n, p_range=...), so any n works.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from obd_explorer.constants import FIGURE_BACKGROUND


@dataclasses.dataclass
class SawtoothConfig:
    n: int = 1000
    p: float = 0.6                      # the window starts half a grid step below this
    steps: float = 3.0                  # window width, in grid steps 1/(2(n+1))
    samples: int = 60001                # dense p grid for E and its slope
    jump_min: float = 1e-3              # the slope line is broken at jumps at least this big
    zoom_ties: int = 4                  # close-up: this many tie points either side of the first cusp
    width_in: float = 12.0
    height_in: float = 11.0
    dpi: int = 200
    output_path: str = "plots/Sawtooth.png"


def export_sawtooth(cfg: SawtoothConfig, verbose: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import obd_core

    n = cfg.n
    step = 1 / (2 * (n + 1))
    lo, hi = cfg.p - 0.5 * step, cfg.p + (cfg.steps - 0.5) * step
    p = np.linspace(lo, hi, cfg.samples)
    E, sl, _ = obd_core.E_slopes_at(n, p)
    t = obd_core.tie_table(n, p_range=(lo, hi))
    tp, ti, tj, tE, tsl, tsr, tc, tlD = (t[c] for c in ("pstar", "i", "j", "E", "slope_left", "slope_right",
                                                         "is_cusp", "log10_D"))
    narrow = (tj - ti) < 2 * np.sqrt(n)
    wide = tc & ~narrow
    fall, rise = narrow & ~tc & (tsr < 0), narrow & ~tc & (tsl > 0)
    big = tlD >= math.log10(cfg.jump_min)                  # jumps big enough to draw

    # the zoom: the narrow tie points of the first cusp's column, a few either side of the cusp
    c0 = np.flatnonzero(tc & narrow)[0]
    col = np.flatnonzero(narrow & (ti + tj == ti[c0] + tj[c0]))
    col = col[np.argsort(tp[col])]
    at = int(np.flatnonzero(col == c0)[0])
    zk = col[max(at - cfg.zoom_ties, 0):at + cfg.zoom_ties + 1]
    gap = np.median(np.diff(tp[zk])) if zk.size > 1 else step / 100
    zlo, zhi = tp[zk].min() - 0.45 * gap, tp[zk].max() + 0.45 * gap
    zp = np.linspace(zlo, zhi, 4001)
    zE = obd_core.E_slopes_at(n, zp)[0]
    inz = (tp > zlo) & (tp < zhi)
    other = inz & (ti + tj != ti[c0] + tj[c0])            # tie points of other columns in the close-up
    others = int(other.sum())
    other_max = float(10 ** tlD[other].max()) if others else 0.0

    fig, (a1, az, a2) = plt.subplots(3, 1, figsize=(cfg.width_in, cfg.height_in),
                                     gridspec_kw={"height_ratios": [1, 1, 1.25]}, facecolor=FIGURE_BACKGROUND)
    a1.sharex(a2)
    grid = [k for k in range(int(np.ceil(lo * 2 * (n + 1))), int(np.floor(hi * 2 * (n + 1))) + 1)]
    for k in grid:
        for a in (a1, a2):
            a.axvline(k * step, color="0.8", lw=0.8, zorder=0)
    for a in (a1, a2):
        a.axvspan(zlo, zhi, color="#ffd54f", alpha=0.35, lw=0, zorder=0)

    # 1. E itself, untilted: its minima and maxima are where they really are
    a1.plot(p, E, color="0.2", lw=1)
    a1.scatter(tp[tc & narrow], tE[tc & narrow], s=40, color="#2ca02c", zorder=3,
               label="cusp: a local minimum of E (the kick that carries the slope across zero)")
    a1.scatter(tp[wide], tE[wide], s=60, color="#ff7f0e", marker="D", zorder=3,
               label="wide cusp (width ≥ 2√n, F3 < 0): a tiny kick just past a maximum, a microscopic dip")
    for q in np.flatnonzero(wide):
        a1.annotate(f"pair ({ti[q]}, {tj[q]}), width {(tj[q] - ti[q]) / np.sqrt(n):.2f}√n:\n"
                    f"slope {tsl[q]:+.4f} → {tsr[q]:+.4f}, a minimum too shallow to see here",
                    (tp[q], tE[q]), xytext=(30, -42), textcoords="offset points", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
                    arrowprops=dict(arrowstyle="->", color="0.4"))
    a1.set_ylabel(f"E(n = {n}, p)")
    a1.ticklabel_format(axis="y", useOffset=False)
    a1.tick_params(labelbottom=False)

    # 2. one cluster close up: E is a polygon, a corner at every tie point
    az.plot(zp, zE, color="0.2", lw=1)
    for mask, colr, size in ((fall, "#1f77b4", 18), (rise, "#d62728", 18), (tc & narrow, "#2ca02c", 50)):
        k = np.intersect1d(zk, np.flatnonzero(mask))
        az.scatter(tp[k], tE[k], s=size, color=colr, zorder=3)
    for q in zk:
        az.annotate(f"{tj[q] - ti[q]}", (tp[q], tE[q]), xytext=(0, 7), textcoords="offset points",
                    fontsize=6, ha="center", color="0.35")
    az.set_xlim(zlo, zhi)
    az.ticklabel_format(axis="both", useOffset=False)
    az.set_ylabel("E, close up")
    az.set_title(f"Close-up of the shaded stretch: the tie points of one column (i + j = {ti[c0] + tj[c0]}), labelled by "
                 f"width.  E bends at each one by its slope jump (0.1 to 2 here)"
                 + (f";\nthe {others} tie points of other columns in this range bend it by at most {other_max:.0e}, "
                    f"far too little to see" if others else ""), fontsize=9)

    # 3. the slope, broken at every jump of at least jump_min: a gap in the line is a jump
    T = np.sort(tp[big])
    seg = np.searchsorted(T, p)
    cut = np.flatnonzero(np.diff(seg) != 0) + 1
    a2.plot(np.insert(p, cut, np.nan), np.insert(sl, cut, np.nan), color="0.2", lw=1,
            label=f"slope of E: continuous between tie points, jumping up at each one (line broken at jumps ≥ "
                  f"{cfg.jump_min:g}; {int((~big).sum())} smaller ones too small to see)")
    a2.scatter(tp[fall], tsr[fall], s=9, color="#1f77b4", zorder=3, label="narrow tie point, E falling through it")
    a2.scatter(tp[rise], tsl[rise], s=9, color="#d62728", zorder=3, label="narrow tie point, E rising through it")
    a2.scatter(tp[tc & narrow], np.zeros((tc & narrow).sum()), s=40, color="#2ca02c", zorder=4)
    a2.scatter(tp[wide], np.zeros(wide.sum()), s=60, color="#ff7f0e", marker="D", zorder=4)
    a2.axhline(0, color="0.5", lw=0.7)
    a2.set_ylabel("slope of E")
    a2.set_xlabel("p")
    top = a2.get_ylim()[1]
    for k in grid:
        a2.text(k * step, top, f"{k}/{2 * (n + 1)} ", fontsize=7, color="0.45", va="top", ha="right", rotation=90)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h2 + h1, l2 + l1, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"The sawtooth, n = {n}: just above each grid fraction k/(2(n+1)) (grey lines) a cluster of tie points "
                 f"kicks E's slope up past zero (a cusp);\nbetween clusters the slope falls back smoothly (a smooth "
                 f"maximum of E).  Shaded: the stretch shown close up in the middle panel", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(cfg.output_path, dpi=cfg.dpi, facecolor=FIGURE_BACKGROUND)
    plt.close(fig)
    if verbose:
        print(f"wrote {cfg.output_path} (n = {n}, p {lo:.6f}..{hi:.6f}: {len(tp)} tie points, "
              f"{int(narrow.sum())} narrow, {int(big.sum())} jumps drawn, {int(tc.sum())} cusps of which {int(wide.sum())} wide)")
