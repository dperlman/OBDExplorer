"""Every tie point of one n as a picture: pair space, coloured by where E's slope changes sign.

A tie point (i, j) sits on the grid point (i+j+1)/(2(n+1)) plus a small offset (RESEARCH_LOG fact
12), so (grid position, width (j-i)/sqrt(n)) puts pair space on a (p, width) plane.  Each pixel shows
its nearest lattice point (i, j).  The colour is the kink position u = S_-/kappa, the left slope over
the slope jump (kappa = (j-i) f(i)): the tie point is a cusp exactly when -1 < u < 0.

    blue    u <= -1   E falls through the tie point (both one-sided slopes negative)
    red     u >= 0    E rises through it
    bright  -1 < u < 0, a cusp: white where its V is symmetric (u = -1/2), aquamarine as u -> -1
            (the right arm flat: it nearly kept falling), yellow as u -> 0 (the left arm flat)
    grey    no tie point (widths past n - 1, the axis column p = 1/2)

Blue and red deepen with log10 |u| (resp. log10(1+u)) over six decades, from mid-lightness (CIE L*
72 and 69) down; every cusp colour has L* >= 92, so the cusps stand out by lightness, not only by
hue (red-green colour vision).  u is formed in logs from ln f(i), so tie points whose kink underflows
double range do not overflow.  The verdicts are OBD-core's certified ones (tie_table); u itself is
double precision.

Below p = 1/2 the picture is the mirror image, by E(p) = E(1-p): the tie point (i, j) at grid
position x mirrors to (n-j, n-i) at 1 - x with the same width, its one-sided slopes swapped and
negated, so u -> -1 - u (falling <-> rising, aquamarine <-> yellow, white stays white).

With supersample = k each pixel averages k x k samples, so that when a pixel spans several lattice
points (large n, wide p range) it shows their mix instead of one of them picked at random; except
that a pixel with any cusp sample shows the cusps' colour, so the cusp curve never fades into its
neighbours.
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
    supersample: int = 2                # average k x k samples per pixel (cusps take priority)
    n_label: bool = False               # a large "n = ..." in the corner (for animation frames)
    width_in: float = 12.0
    height_in: float = 7.0
    dpi: int = 600
    output_path: str = "plots/PairMap.png"


STYLE = 2                               # bump when the drawing changes: cached animation frames are redrawn
_MARGIN = {"left": 0.07, "right": 0.17, "bottom": 0.085, "top": 0.115}
_BLUE_FLOOR, _RED_FLOOR = 0.45, 0.40    # where the Blues / Reds ramps start: L* 72 and 69
_CUSP_COLOURS = ("#7FFFD4", "#FFFFFF", "#FFEB3B")   # u = -1, -1/2, 0: L* 92, 100, 92
_EMPTY = 0.5                            # grey, L* 53


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


def _colour_maps():
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    blues = LinearSegmentedColormap.from_list("fall", plt.get_cmap("Blues")(np.linspace(_BLUE_FLOOR, 1, 256)))
    reds = LinearSegmentedColormap.from_list("rise", plt.get_cmap("Reds")(np.linspace(_RED_FLOOR, 1, 256)))
    cusp = LinearSegmentedColormap.from_list("cusp", _CUSP_COLOURS)
    return blues, reds, cusp


def sample_colours(U: np.ndarray, C: np.ndarray, i: np.ndarray, m: np.ndarray, ok: np.ndarray,
                   flip: np.ndarray, decades: float, maps=None) -> tuple[np.ndarray, np.ndarray]:
    """(rgb, cusp) for the sample points showing tie point (i, i+m) of the p* > 1/2 half (where ok),
    drawn as its own mirror image where flip (u -> -1 - u: falling <-> rising); elsewhere _EMPTY."""
    blues, reds, cuspmap = maps or _colour_maps()
    lv = lambda v: np.clip(np.log10(np.maximum(v, 1.0)) / decades, 0, 1)
    u = np.full(i.shape, np.nan)
    cusp = np.zeros(i.shape, bool)
    u[ok] = U[i[ok], m[ok]]
    cusp[ok] = C[i[ok], m[ok]]
    fall0 = u <= -1                                          # else u >= 0, or a borderline row certified NOT
    fall = ok & ~cusp & np.where(flip, ~fall0, fall0)        # the mirror swaps falling and rising
    rise = ok & ~cusp & ~fall
    u[flip] = -1 - u[flip]
    rgb = np.full(u.shape + (3,), _EMPTY, np.float32)
    rgb[fall] = blues(lv(-u[fall]))[:, :3]
    rgb[rise] = reds(lv(1 + np.maximum(u[rise], 0)))[:, :3]
    rgb[cusp] = cuspmap(np.clip(u[cusp] + 1, 0, 1))[:, :3]
    return rgb, cusp


def downsample(rgb: np.ndarray, cusp: np.ndarray, ss: int) -> np.ndarray:
    """Average ss x ss blocks of samples into pixels; a pixel with any cusp sample shows the cusps' colour."""
    if ss == 1:
        return rgb
    hb, wb = rgb.shape[0] // ss, rgb.shape[1] // ss
    blk = rgb.reshape(hb, ss, wb, ss, 3)
    cb = cusp.reshape(hb, ss, wb, ss)
    pix = blk.mean(axis=(1, 3))
    nc = cb.sum(axis=(1, 3))
    has = nc > 0
    pix[has] = (blk * cb[..., None]).sum(axis=(1, 3))[has] / nc[has][:, None]
    return pix


def pair_rgb(U: np.ndarray, C: np.ndarray, n: int, cfg: PairMapConfig, w: int, h: int,
             chunk_rows: int = 64) -> np.ndarray:
    """The (h, w, 3) image, row 0 at width 0, computed a band of rows at a time."""
    maps = _colour_maps()
    ss = max(int(cfg.supersample), 1)
    X = cfg.p_min + (np.arange(w * ss) + 0.5) / (w * ss) * (cfg.p_max - cfg.p_min)
    flip = X < 0.5                                           # mirror half: E(p) = E(1-p)
    Xe = np.where(flip, 1 - X, X)
    out = np.empty((h, w, 3), np.float32)
    for r0 in range(0, h, chunk_rows):
        r1 = min(r0 + chunk_rows, h)
        ys = (np.arange(r0 * ss, r1 * ss) + 0.5) / (h * ss) * cfg.width_max
        m_raw = np.round(ys * math.sqrt(n)).astype(np.int64)[:, None]
        m = np.clip(m_raw, 1, n - 1)                         # rows past width n-1 (small n) stay empty
        # nearest i + j + 1 = 2(n+1)x with the parity of the width: i + j = n + band, j - i = m
        s0 = np.round((2 * (n + 1) * Xe[None, :] - 1 - m) / 2) * 2 + 1 + m
        i = ((s0 - 1 - m) // 2).astype(np.int64)
        m = np.broadcast_to(m, i.shape)
        ok = (i >= 1) & (i + m <= n) & (2 * i + m > n) & (m_raw <= n - 1)
        rgb, cusp = sample_colours(U, C, i, m, ok, np.broadcast_to(flip[None, :], i.shape), cfg.decades, maps)
        out[r0:r1] = downsample(rgb, cusp, ss)
    return out


def export_pair_map(cfg: PairMapConfig, verbose: bool = False, pool=None, tables=None) -> None:
    """Draw cfg.output_path; tables = pair_u(cfg.n) if already computed (one n, several ranges)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    n = cfg.n
    U, C = tables if tables is not None else pair_u(n, cfg.workers, verbose, pool)
    fw, fh = int(round(cfg.width_in * cfg.dpi)), int(round(cfg.height_in * cfg.dpi))
    left, bottom = int(round(_MARGIN["left"] * fw)), int(round(_MARGIN["bottom"] * fh))
    w = fw - left - int(round(_MARGIN["right"] * fw))
    h = fh - bottom - int(round(_MARGIN["top"] * fh))
    rgb = pair_rgb(U, C, n, cfg, w, h)
    blues, reds, cuspmap = _colour_maps()

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
            (cuspmap, Normalize(-1, 0), "cusp: u  (white: a symmetric V)"))):
        cax = fig.add_axes([x0 + k * (bar_w + gap), bottom / fh, bar_w, h / fh])
        cb = fig.colorbar(ScalarMappable(norm, cm), cax=cax)
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    mirror = ";  below ½: the mirror image, E(p) = E(1−p)" if cfg.p_min < 0.5 else ""
    fig.suptitle(f"Every tie point of n = {n} in pair space: grid position against width/√n, "
                 f"coloured by u = S₋/κ (the left slope over the slope jump)\n"
                 f"blue: E falls through it (u ≤ −1);  red: E rises (u ≥ 0);  bright: a cusp (−1 < u < 0, "
                 f"certified by OBD-core)\n"
                 f"a cusp is white where its V is symmetric (u = −½), aquamarine or yellow where it barely is one{mirror}",
                 fontsize=9.5, y=1 - 0.008)
    fig.savefig(cfg.output_path, dpi=cfg.dpi, facecolor=FIGURE_BACKGROUND)
    plt.close(fig)
    if verbose:
        print(f"wrote {cfg.output_path} ({w} x {h} pixels, {cfg.supersample}x{cfg.supersample} samples each)")
