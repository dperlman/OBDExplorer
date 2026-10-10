"""The pair-space picture (pair_map) at log-spaced n, stitched into a looping animation.

Frame k shows n = round(n_min * (n_max/n_min)^(k/(frames-1))), with the same axes throughout
(grid position across, width/sqrt(n) up), so the structure stays in place while it sharpens.  Frames
are PNGs cached in frames_dir under a hash of the drawing settings: a rerun re-encodes without
recomputing, and changing n_max or the frame count only computes the new n.

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

from obd_explorer.pair_map import PairMapConfig, export_pair_map


@dataclasses.dataclass
class PairMapMovieConfig:
    n_min: int = 50
    n_max: int = 8000
    frames: int = 100
    fps: float = 10.0
    hold_end: float = 2.0               # seconds the last frame stays up before the loop restarts
    p_min: float = 0.5
    p_max: float = 0.7
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


def _frame_dir(cfg: PairMapMovieConfig) -> str:
    keys = ("p_min", "p_max", "width_max", "decades", "supersample", "width_in", "height_in", "dpi")
    tag = json.dumps({k: getattr(cfg, k) for k in keys}, sort_keys=True)
    return os.path.join(cfg.frames_dir, hashlib.sha1(tag.encode()).hexdigest()[:10])


def render_frames(cfg: PairMapMovieConfig, verbose: bool = False) -> list[str]:
    from multiprocessing import Pool

    out = _frame_dir(cfg)
    os.makedirs(out, exist_ok=True)
    ns = movie_n_values(cfg)
    paths = [os.path.join(out, f"n{n:06d}.png") for n in ns]
    todo = [(n, p) for n, p in zip(ns, paths) if not os.path.isfile(p)]
    if verbose:
        print(f"{len(ns)} frames, n = {ns[0]}..{ns[-1]}; {len(ns) - len(todo)} cached in {out}, {len(todo)} to render")
    t0 = time.time()
    with Pool(cfg.workers) as pool:
        for k, (n, p) in enumerate(todo):
            fc = PairMapConfig(n=n, p_min=cfg.p_min, p_max=cfg.p_max, width_max=cfg.width_max, decades=cfg.decades,
                               workers=cfg.workers, supersample=cfg.supersample, n_label=True,
                               width_in=cfg.width_in, height_in=cfg.height_in, dpi=cfg.dpi, output_path=p + ".tmp.png")
            export_pair_map(fc, pool=pool)
            os.replace(fc.output_path, p)
            if verbose:
                print(f"  [{k + 1}/{len(todo)}] n = {n}  ({time.time() - t0:.0f}s)", flush=True)
    return paths


def encode(paths: list[str], cfg: PairMapMovieConfig, output: str, comment: str = "") -> None:
    """Stitch the frames into output (.gif or .mp4) with ffmpeg, holding the last one hold_end seconds."""
    lst = output + ".frames.txt"
    with open(lst, "w") as f:
        for k, p in enumerate(paths):
            d = 1 / cfg.fps + (cfg.hold_end if k == len(paths) - 1 else 0)
            f.write(f"file '{os.path.abspath(p)}'\nduration {d:.6f}\n")
        f.write(f"file '{os.path.abspath(paths[-1])}'\n")     # concat needs the last file twice for its duration
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst]
    ext = os.path.splitext(output)[1].lower()
    if ext == ".gif":
        cmd += ["-vf", f"fps={cfg.fps},scale={cfg.gif_width}:-1:flags=lanczos,split[a][b];"
                       "[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=sierra2_4a",
                "-loop", "0"]
    elif ext == ".mp4":
        cmd += ["-vf", f"fps={cfg.fps},format=yuv420p", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
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
    for out in cfg.outputs:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        encode(paths, cfg, out, json.dumps(stamp_items(out, cfg, args), sort_keys=True))
        if verbose:
            print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
