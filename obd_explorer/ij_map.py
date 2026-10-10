"""Every tie point of one n in (i, j) coordinates: all pairs i < j, the companion of pair_map.

pair_map is this picture rotated 45 degrees and stretched across the diagonal by about 2 sqrt(n), cut
off at width 8 sqrt(n): the right view for the cusps, which all sit within ~1.2 sqrt(n) of the
diagonal.  Here every pair is shown, out to width n.  A wide pair's slope jump is tiny, so its colour
is just the sign of E' at its tie point p*; pairs tying at the same p* lie on one curve, so each blue
band is the set of pairs whose tie point lands in one of E's downhill stretches (just below a grid
fraction, 0.348 < p < 0.652).  The picture is symmetric about the anti-diagonal i + j = n (p = 1/2),
with blue and red swapped: the mirror of (i, j) is (n-j, n-i), by E(p) = E(1-p).

Colours as in pair_map (blue falls, red rises, bright = cusp, grey = no tie point: the axis i + j = n
and the pairs (0, j)); below the diagonal is not part of the picture.  With zoom = (lo, hi) a second
panel shows lo <= i, j <= hi, marked on the first.  Pixels average supersample^2 samples, cusps first.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from obd_explorer.constants import FIGURE_BACKGROUND
from obd_explorer.pair_map import _colour_maps, downsample, pair_u, sample_colours


@dataclasses.dataclass
class IJMapConfig:
    n: int = 4000
    zoom: tuple[int, int] | None = None  # (lo, hi): a second panel with lo <= i, j <= hi
    decades: float = 6.0
    supersample: int = 3
    workers: int = 8
    panel_px: int = 2400                # each panel is panel_px x panel_px pixels
    dpi: int = 300
    output_path: str = "plots/IJMap.png"


def ij_rgb(U: np.ndarray, C: np.ndarray, n: int, lo: int, hi: int, px: int, ss: int, decades: float,
           chunk_rows: int = 64) -> np.ndarray:
    """(px, px, 3) image of lo <= i, j <= hi, row 0 at j = lo."""
    maps = _colour_maps()
    span = hi - lo + 1
    I = np.rint(lo - 0.5 + (np.arange(px * ss) + 0.5) / (px * ss) * span).astype(np.int64)[None, :]
    out = np.empty((px, px, 3), np.float32)
    for r0 in range(0, px, chunk_rows):
        r1 = min(r0 + chunk_rows, px)
        J = np.rint(lo - 0.5 + (np.arange(r0 * ss, r1 * ss) + 0.5) / (px * ss) * span).astype(np.int64)[:, None]
        up = I + J > n                                       # p* > 1/2; else draw the mirror (n-j, n-i)
        a = np.where(up, I, n - J)
        m = np.where(up, J, n - I) - a
        ok = (J > I) & (I + J != n) & (a >= 1) & (m >= 1) & (a + m <= n)
        a, m = np.where(ok, a, 0), np.where(ok, m, 0)
        rgb, cusp = sample_colours(U, C, a, m, ok, ~up, decades, maps)
        rgb[np.broadcast_to(J <= I, ok.shape)] = 1.0         # below the diagonal: not part of the picture
        out[r0:r1] = downsample(rgb, cusp, ss)
    return out


def export_ij_map(cfg: IJMapConfig, verbose: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    n = cfg.n
    U, C = pair_u(n, cfg.workers, verbose)
    panels = [(0, n)] + ([tuple(cfg.zoom)] if cfg.zoom else [])
    side = cfg.panel_px / cfg.dpi
    fig = plt.figure(figsize=(side * len(panels) + 0.9 * len(panels) + 2.3, side + 1.5), dpi=cfg.dpi,
                     facecolor=FIGURE_BACKGROUND)
    W, H = fig.get_size_inches()
    for k, (lo, hi) in enumerate(panels):
        ax = fig.add_axes([(0.75 + k * (side + 0.9)) / W, 0.65 / H, side / W, side / H])
        img = ij_rgb(U, C, n, lo, hi, cfg.panel_px, cfg.supersample, cfg.decades)
        ax.imshow(img, origin="lower", extent=[lo - 0.5, hi + 0.5, lo - 0.5, hi + 0.5], interpolation="none")
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        if lo <= n / 2 <= hi:
            a0, a1 = max(lo, n - hi), min(hi, n - lo)
            ax.plot([a0, a1], [n - a0, n - a1], color="k", lw=0.6, ls="--")
        if k == 0:
            ax.set_title(f"all pairs i < j  (dashed: i + j = n, p = ½)", fontsize=10)
            if cfg.zoom:
                zl, zh = cfg.zoom
                ax.add_patch(plt.Rectangle((zl, zl), zh - zl, zh - zl, fill=False, ec="k", lw=0.8))
        else:
            ax.set_title(f"close-up: {lo} ≤ i, j ≤ {hi}  (the cusps lie within about 1.2√n ≈ "
                         f"{1.2 * np.sqrt(n):.0f} of the diagonal)", fontsize=10)
    blues, reds, cuspmap = _colour_maps()
    x0 = (0.75 + len(panels) * (side + 0.9) - 0.45) / W
    for k, (cm, norm, label) in enumerate((
            (blues, Normalize(0, cfg.decades), "falling: log₁₀ |u|"),
            (reds, Normalize(0, cfg.decades), "rising: log₁₀ (1+u)"),
            (cuspmap, Normalize(-1, 0), "cusp: u  (white: a symmetric V)"))):
        cax = fig.add_axes([x0 + k * 0.72 / W, 0.65 / H, 0.12 / W, side / H])
        cb = fig.colorbar(ScalarMappable(norm, cm), cax=cax)
        cb.set_label(label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"Every tie point of n = {n} in (i, j) coordinates, coloured by u = S₋/κ (the left slope over the slope "
                 f"jump), as in the pair-space plots\nblue: E falls through it;  red: E rises;  bright: a cusp (white "
                 f"where its V is symmetric);  grey: no tie point.  Symmetric about i + j = n with blue and red swapped "
                 f"(E(p) = E(1−p))", fontsize=10, y=1 - 0.12 / H)
    fig.savefig(cfg.output_path, dpi=cfg.dpi, facecolor=FIGURE_BACKGROUND)
    plt.close(fig)
    if verbose:
        print(f"wrote {cfg.output_path} ({len(panels)} panel(s) of {cfg.panel_px} px, {cfg.supersample}x{cfg.supersample} samples each)")
