"""Every tie point of one n as a picture: pair space, coloured by where E's slope changes sign.

A tie point (i, j) sits on the grid point (i+j+1)/(2(n+1)) plus a small offset (RESEARCH_LOG fact
12), so (grid position, width (j-i)/sqrt(n)) puts pair space on a (p, width) plane.  Each pixel shows
its nearest lattice point (i, j).  The colour is the kink position u = S_-/kappa, the left slope over
the slope jump (kappa = (j-i) f(i)): the tie point is a cusp exactly when -1 < u < 0.

    blue    u <= -1   E falls through the tie point (both one-sided slopes negative)
    red     u >= 0    E rises through it
    green   -1 < u < 0, a cusp, shaded by u

Blue and red deepen with log10 |u| (resp. log10(1+u)) over six decades.  u is formed in logs from
ln f(i), so tie points whose kink underflows double range do not overflow.  The verdicts are
OBD-core's certified ones (tie_table); u itself is double precision.

With supersample = k each pixel is the average of k x k samples, so that when a pixel spans several
lattice points (large n, small image) it shows their mix instead of one of them picked at random.
"""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np

from obd_explorer.constants import FIGURE_BACKGROUND


@dataclasses.dataclass
class PairMapConfig:
    n: int = 1000
    p_min: float = 0.5
    p_max: float = 0.7
    width_max: float = 8.0              # top of the plot, in units of sqrt(n)
    decades: float = 6.0                # blue/red shading range in log10 |u|
    workers: int = 8
    supersample: int = 1                # average k x k samples per pixel
    n_label: bool = False               # a large "n = ..." in the corner (for animation frames)
    width_in: float = 12.0
    height_in: float = 7.0
    dpi: int = 600
    output_path: str = "plots/PairMap.png"


_MARGIN = {"left": 0.07, "right": 0.17, "bottom": 0.085, "top": 0.1}


def pair_u(n: int, workers: int = 8, verbose: bool = False, pool=None) -> tuple[np.ndarray, np.ndarray]:
    """(U, C): U[i, m] = u = S_-/kappa of the tie point (i, i+m), C[i, m] its certified cusp flag."""
    from multiprocessing import Pool

    import obd_core

    t0 = time.time()
    if pool is not None:
        tab = obd_core.tie_table(n, workers=workers, pool=pool)
    elif workers > 1:
        with Pool(workers) as pool:
            tab = obd_core.tie_table(n, workers=workers, pool=pool)
    else:
        tab = obd_core.tie_table(n)
    s = slice(1, None)                                       # row 0 is the axis p = 1/2
    i = tab["i"][s].astype(np.int64)
    m = tab["j"][s].astype(np.int64) - i
    sm = tab["S_minus"][s]
    lu = np.log(np.abs(sm) + 1e-300) - np.log(m) - tab["ln_fi"][s]
    U = np.full((n + 1, n + 1), np.nan)
    U[i, m] = np.sign(sm) * np.exp(np.clip(lu, -700, 700))
    C = np.zeros((n + 1, n + 1), bool)
    C[i, m] = tab["is_cusp"][s]
    if verbose:
        print(f"n = {n}: {i.size:,} tie points, {int(C.sum())} cusps, {time.time() - t0:.0f}s")
    return U, C


def export_pair_map(cfg: PairMapConfig, verbose: bool = False, pool=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    n = cfg.n
    U, C = pair_u(n, cfg.workers, verbose, pool)
    ss = max(int(cfg.supersample), 1)
    fw, fh = int(round(cfg.width_in * cfg.dpi)), int(round(cfg.height_in * cfg.dpi))
    left, bottom = int(round(_MARGIN["left"] * fw)), int(round(_MARGIN["bottom"] * fh))
    w = fw - left - int(round(_MARGIN["right"] * fw))
    h = fh - bottom - int(round(_MARGIN["top"] * fh))
    xs = cfg.p_min + (np.arange(w * ss) + 0.5) / (w * ss) * (cfg.p_max - cfg.p_min)
    ys = (np.arange(h * ss) + 0.5) / (h * ss) * cfg.width_max
    X, Y = np.meshgrid(xs, ys)
    m_raw = np.round(Y * math.sqrt(n)).astype(np.int64)
    m = np.clip(m_raw, 1, n - 1)                             # rows past width n-1 (small n) stay blank
    # nearest i + j + 1 = 2(n+1)x with the parity of the width: i + j = n + band, j - i = m
    s0 = np.round((2 * (n + 1) * X - 1 - m) / 2) * 2 + 1 + m
    i = ((s0 - 1 - m) // 2).astype(np.int64)
    ok = (i >= 1) & (i + m <= n) & (2 * i + m > n) & (m_raw <= n - 1)
    u = np.full(X.shape, np.nan)
    cusp = np.zeros(X.shape, bool)
    u[ok] = U[i[ok], m[ok]]
    cusp[ok] = C[i[ok], m[ok]]
    fall = ok & ~cusp & (u <= -1)
    rise = ok & ~cusp & (u > -1)                             # u >= 0, or a borderline row certified NOT
    blues, reds = plt.get_cmap("Blues"), plt.get_cmap("Reds")
    greens = LinearSegmentedColormap.from_list("cusp", ["#f7f73a", "#3bb54a", "#0d5c2e"])
    lv = lambda v: np.clip(np.log10(np.maximum(v, 1.0)) / cfg.decades, 0, 1)
    rgb = np.ones(X.shape + (3,))
    rgb[fall] = blues(0.2 + 0.8 * lv(-u[fall]))[:, :3]
    rgb[rise] = reds(0.2 + 0.8 * lv(1 + np.maximum(u[rise], 0)))[:, :3]
    rgb[cusp] = greens(np.clip(-u[cusp], 0, 1))[:, :3]
    if ss > 1:
        rgb = rgb.reshape(h, ss, w, ss, 3).mean(axis=(1, 3))

    fig = plt.figure(figsize=(fw / cfg.dpi, fh / cfg.dpi), dpi=cfg.dpi, facecolor=FIGURE_BACKGROUND)
    ax = fig.add_axes([left / fw, bottom / fh, w / fw, h / fh])
    ax.imshow(rgb, extent=[cfg.p_min, cfg.p_max, 0, cfg.width_max], origin="lower", aspect="auto",
              interpolation="none")
    if cfg.n_label:
        ax.text(0.985, 0.975, f"n = {n}", transform=ax.transAxes, ha="right", va="top", fontsize=22,
                bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.9, pad=6))
    ax.set_xlabel("grid position (i+j+1) / (2(n+1))  ≈  p*")
    ax.set_ylabel("width (j − i) / √n")
    bar_w, gap = 0.010, 0.045
    x0 = (left + w) / fw + 0.012
    for k, (cm, norm, label) in enumerate((
            (blues, Normalize(0, cfg.decades), "falling: log₁₀ |u|"),
            (reds, Normalize(0, cfg.decades), "rising: log₁₀ (1+u)"),
            (greens, Normalize(-1, 0), "cusp: u"))):
        cax = fig.add_axes([x0 + k * (bar_w + gap), bottom / fh, bar_w, h / fh])
        if k < 2:
            sm_ = ScalarMappable(norm, LinearSegmentedColormap.from_list(f"c{k}", cm(np.linspace(0.2, 1, 64))))
        else:
            sm_ = ScalarMappable(Normalize(-1, 0), LinearSegmentedColormap.from_list("g", greens(np.linspace(1, 0, 64))))
        cb = fig.colorbar(sm_, cax=cax)
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"Every tie point of n = {n} in pair space: grid position against width/√n  "
                 f"(colour: u = S₋/κ, the left slope over the slope jump)\n"
                 f"blue: E falls through it (u ≤ −1); red: E rises (u ≥ 0); green: a cusp (−1 < u < 0, certified by OBD-core)",
                 fontsize=10, y=1 - 0.01)
    fig.savefig(cfg.output_path, dpi=cfg.dpi, facecolor=FIGURE_BACKGROUND)
    plt.close(fig)
    if verbose:
        print(f"wrote {cfg.output_path} ({w} x {h} pixels)")
