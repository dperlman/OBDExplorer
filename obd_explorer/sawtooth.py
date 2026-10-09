"""The sawtooth: E and its slope over a few grid steps, showing where cusps come from.

Tie points cluster just above the grid fractions k/(2(n+1)) (RESEARCH_LOG fact 12), ordered by width.
Walking right through a cluster, each tie point kicks E's slope up by its jump D, and the slope
climbs from negative to positive: the cusp is the kick that carries it across zero.  Between
clusters E is a smooth concave arc and the slope falls back: a smooth maximum.  So E has one dip
per grid fraction (below p ~ 0.652) and a hump between.  The wide cusps (width >= 2 sqrt(n), which
are exactly the F3 < 0 ones) are the exception: a tie point with a tiny jump landing just after the
slope has crossed zero going down, past a maximum, and flipping it back (RESEARCH_LOG 2026-10-09).

E and the slope come from obd_core.E_slopes_at on a dense grid; the tie points and their certified
cusp verdicts from obd_core.tie_table(n, p_range=...), so any n works.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from obd_explorer.constants import FIGURE_BACKGROUND


@dataclasses.dataclass
class SawtoothConfig:
    n: int = 1000
    p: float = 0.6                      # the window starts half a grid step below this
    steps: float = 3.0                  # window width, in grid steps 1/(2(n+1))
    samples: int = 60001                # dense p grid for E and its slope
    width_in: float = 12.0
    height_in: float = 8.0
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
    tp, ti, tj, tE, tsl, tsr, tc = (t[c] for c in ("pstar", "i", "j", "E", "slope_left", "slope_right", "is_cusp"))
    narrow = (tj - ti) < 2 * np.sqrt(n)
    wide = tc & ~narrow
    fit = np.polyfit(p, E, 1)
    det = lambda q, e: e - np.polyval(fit, q)              # E with the straight-line tilt removed

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(cfg.width_in, cfg.height_in), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1.2]}, facecolor=FIGURE_BACKGROUND)
    grid = [k for k in range(int(np.ceil(lo * 2 * (n + 1))), int(np.floor(hi * 2 * (n + 1))) + 1)]
    for k in grid:
        for a in (a1, a2):
            a.axvline(k * step, color="0.8", lw=0.8, zorder=0)
    a1.plot(p, det(p, E), color="0.2", lw=1)
    a1.scatter(tp[tc & narrow], det(tp[tc & narrow], tE[tc & narrow]), s=40, color="#2ca02c", zorder=3,
               label="cusp (the kick that carries the slope across zero)")
    a1.scatter(tp[wide], det(tp[wide], tE[wide]), s=60, color="#ff7f0e", marker="D", zorder=3,
               label="wide cusp (width ≥ 2√n, F3 < 0): a tiny kick just past a maximum")
    for q in np.flatnonzero(wide):
        a1.annotate(f"pair ({ti[q]}, {tj[q]}), width {(tj[q] - ti[q]) / np.sqrt(n):.2f}√n:\n"
                    f"slope {tsl[q]:+.4f} → {tsr[q]:+.4f}, a microscopic dip",
                    (tp[q], det(tp[q], tE[q])), xytext=(42, -38), textcoords="offset points", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
                    arrowprops=dict(arrowstyle="->", color="0.4"))
    a1.set_ylabel("E − (straight line fitted\nover this stretch)")
    a2.plot(p, sl, color="0.2", lw=1, label="slope of E (left slope)")
    fall, rise = narrow & ~tc & (tsr < 0), narrow & ~tc & (tsl > 0)
    a2.scatter(tp[fall], tsr[fall], s=9, color="#1f77b4", zorder=3, label="narrow tie point, E falling through it")
    a2.scatter(tp[rise], tsl[rise], s=9, color="#d62728", zorder=3, label="narrow tie point, E rising through it")
    a2.scatter(tp[tc & narrow], np.zeros((tc & narrow).sum()), s=40, color="#2ca02c", zorder=4)
    a2.scatter(tp[wide], np.zeros(wide.sum()), s=60, color="#ff7f0e", marker="D", zorder=4)
    a2.axhline(0, color="0.5", lw=0.7)
    a2.set_ylabel("slope of E")
    a2.set_xlabel("p")
    top = a1.get_ylim()[1]
    for k in grid:
        a1.text(k * step, top, f"{k}/{2 * (n + 1)} ", fontsize=7, color="0.45", va="top", ha="right", rotation=90)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h2 + h1, l2 + l1, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"The sawtooth, n = {n}: inside each cluster of tie points (just above each grid fraction k/(2(n+1)), "
                 f"grey lines)\nthe slope is kicked up past zero (a cusp); between clusters it falls back smoothly "
                 f"(a smooth maximum of E)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(cfg.output_path, dpi=cfg.dpi, facecolor=FIGURE_BACKGROUND)
    plt.close(fig)
    if verbose:
        print(f"wrote {cfg.output_path} (n = {n}, p {lo:.6f}..{hi:.6f}: {len(tp)} tie points, "
              f"{int(narrow.sum())} narrow, {int(tc.sum())} cusps of which {int(wide.sum())} wide)")
