"""The pair-space picture (pair_map) at log-spaced n, stitched into a looping animation.

Frame k shows n = round(n_min * (n_max/n_min)^(k/(frames-1))), with the same axes throughout
(grid position across, width/sqrt(n) up), so the structure stays in place while it sharpens.  Frames
are PNGs cached in frames_dir under a hash of the drawing settings: a rerun re-encodes without
recomputing, and changing n_max or the frame count only computes the new n.  Several p ranges can be
made in one run: each n's tie table is computed once and drawn in every range, and an output path
containing {range} is written once per range ("0-1", "0.5-1", ...).

Outputs by extension (ffmpeg): .gif loops forever on its own; .mp4 (H.264) is far smaller but loops
only where the player is told to (docs/index.html uses <video loop autoplay muted>).  The last frame
is held for hold_end seconds before the loop restarts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import time

import numpy as np

from obd_explorer.pair_map import STYLE, PairMapConfig, export_pair_map, pair_u


@dataclasses.dataclass
class PairMapMovieConfig:
    n_min: int = 50
    n_max: int = 8000
    frames: int = 100
    fps: float = 10.0
    hold_end: float = 2.0               # seconds the last frame stays up before the loop restarts
    ranges: tuple[tuple[float, float], ...] = ((0.5, 0.7),)   # (p_min, p_max) per animation
    width_max: float = 8.0
    decades: float = 6.0
    supersample: int = 3
    workers: int = 8
    width_in: float = 12.8              # 1920 x 1080 at 150 dpi
    height_in: float = 7.2
    dpi: int = 150
    gif_width: int = 1280               # GIFs are scaled to this width (pixels) to keep them small
    frames_dir: str = "plots/pair-map-frames"
    outputs: tuple[str, ...] = ("plots/PairMap-movie.gif",)


def movie_n_values(cfg: PairMapMovieConfig) -> list[int]:
    n = np.round(np.geomspace(cfg.n_min, cfg.n_max, cfg.frames)).astype(int)
    return sorted(set(int(v) for v in n))


def range_label(r: tuple[float, float]) -> str:
    return f"{r[0]:g}-{r[1]:g}"


def _frame_dir(cfg: PairMapMovieConfig, r: tuple[float, float]) -> str:
    keys = ("width_max", "decades", "supersample", "width_in", "height_in", "dpi")
    tag = json.dumps({**{k: getattr(cfg, k) for k in keys}, "p_min": r[0], "p_max": r[1], "style": STYLE},
                     sort_keys=True)
    return os.path.join(cfg.frames_dir, f"{range_label(r)}-{hashlib.sha1(tag.encode()).hexdigest()[:10]}")


def render_frames(cfg: PairMapMovieConfig, verbose: bool = False) -> dict[tuple[float, float], list[str]]:
    """{range: frame paths in n order}, rendering only the frames not already cached."""
    from multiprocessing import Pool

    ns = movie_n_values(cfg)
    paths = {}
    for r in cfg.ranges:
        os.makedirs(_frame_dir(cfg, r), exist_ok=True)
        paths[r] = [os.path.join(_frame_dir(cfg, r), f"n{n:06d}.png") for n in ns]
    todo = [k for k in range(len(ns)) if any(not os.path.isfile(paths[r][k]) for r in cfg.ranges)]
    if verbose:
        print(f"{len(ns)} frames, n = {ns[0]}..{ns[-1]}, ranges {', '.join(map(range_label, cfg.ranges))}; "
              f"{len(todo)} n to render")
    t0 = time.time()
    with Pool(cfg.workers) as pool:
        for c, k in enumerate(todo):
            tables = pair_u(ns[k], cfg.workers, pool=pool)
            for r in cfg.ranges:
                p = paths[r][k]
                if os.path.isfile(p):
                    continue
                fc = PairMapConfig(n=ns[k], p_min=r[0], p_max=r[1], width_max=cfg.width_max, decades=cfg.decades,
                                   workers=cfg.workers, supersample=cfg.supersample, n_label=True,
                                   width_in=cfg.width_in, height_in=cfg.height_in, dpi=cfg.dpi,
                                   output_path=p + ".tmp.png")
                export_pair_map(fc, tables=tables)
                os.replace(fc.output_path, p)
            del tables
            if verbose:
                print(f"  [{c + 1}/{len(todo)}] n = {ns[k]}  ({time.time() - t0:.0f}s)", flush=True)
    return paths


def encode(paths: list[str], cfg: PairMapMovieConfig, output: str, comment: str = "") -> None:
    """Stitch the frames into output (.gif or .mp4) with ffmpeg, holding the last one hold_end seconds."""
    lst = output + ".frames.txt"
    with open(lst, "w") as f:
        for p in paths:
            f.write(f"file '{os.path.abspath(p)}'\nduration {1 / cfg.fps:.6f}\n")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst]
    hold = f"tpad=stop_mode=clone:stop_duration={cfg.hold_end}"   # concat ignores the last file's duration
    ext = os.path.splitext(output)[1].lower()
    if ext == ".gif":
        cmd += ["-vf", f"fps={cfg.fps},{hold},scale={cfg.gif_width}:-1:flags=lanczos,split[a][b];"
                       "[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a",
                "-loop", "0"]
    elif ext == ".mp4":
        cmd += ["-vf", f"fps={cfg.fps},{hold},format=yuv420p", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                "-movflags", "+faststart"]
    else:
        raise ValueError(f"{output!r}: write .gif or .mp4")
    if comment:
        cmd += ["-metadata", f"comment={comment}"]
    try:
        subprocess.run(cmd + [output], check=True)
    finally:
        os.remove(lst)


def export_pair_map_movie(cfg: PairMapMovieConfig, args=None, verbose: bool = False) -> None:
    from obd_explorer.png_metadata import stamp_items

    paths = render_frames(cfg, verbose)
    for r in cfg.ranges:
        for out in cfg.outputs:
            if "{range}" not in out and len(cfg.ranges) > 1:
                raise ValueError(f"{out!r}: put {{range}} in the output name when making several ranges")
            out = out.replace("{range}", range_label(r))
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            encode(paths[r], cfg, out, json.dumps(stamp_items(out, cfg, args), sort_keys=True))
            if verbose:
                print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
