"""A soundtrack for a pair-map movie: n is the time axis, one slice of sound per frame.

    python sonification/soundtrack.py plots/PairMap-n50-10000-sqrt32.mp4

reads the movie's own settings (the obd:cli_args its encoder stamped in the file), rebuilds its
frame list (pair_map_movie.movie_n_values), renders a stereo WAV of exactly the movie's length and
writes a copy of the movie with that sound into sonification/out/.  The tie points are the certified
cusps of OBD-core's shared store; the axis p = 1/2 is left out.

Two layers, also written as separate WAVs (stems) for mixing elsewhere:

    sparkle   every wide cusp (z = (j-i)/(2 sqrt(npq)) >= z_wide; there are none between ~1.3 and 3)
              of the frame is a short bell ping: pitch = height in the picture (z, log scale, ~1.6 to
              ~4.5 kHz), stereo position = where it sits across the picture (p from p_min to p_max,
              its mirror 1 - p too when the range reaches below 1/2), brightness = how lopsided its
              V is (|2u + 1|: a white, symmetric cusp is a pure tone, an aquamarine or yellow one
              carries a strong inharmonic overtone).  Onsets are spread over the frame.
    bed       the ordinary cusps near p = 1/2 as a held chord: a cusp at x = n(p - 1/2) sounds
              harmonic 1 + 2x of base_hz (band m ~ harmonic 2m + 1), as clip 4 of sonify.py, cross-
              faded frame to frame.  It settles as the bands lock in (n-invariance) and then holds.

During the movie's end hold (hold_end) the last frame's bed sustains and fades.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repo root

from obd_explorer.pair_map_movie import PairMapMovieConfig, movie_n_values  # noqa: E402

SR = 44100
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def movie_settings(path: str) -> dict:
    """The pair-map-movie command-line options stamped in a movie's comment tag."""
    tag = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "json", path],
                         check=True, capture_output=True, text=True).stdout
    comment = json.loads(tag)["format"].get("tags", {}).get("comment")
    if not comment:
        raise ValueError(f"{path}: no settings stamped in the movie (made before pair-map-movie stamped them?)")
    return json.loads(json.loads(comment)["obd:cli_args"])


def movie_schedule(args: dict, path: str) -> tuple[list[int], float, float, tuple[float, float]]:
    """(n per frame, fps, hold_end, (p_min, p_max)) of the movie."""
    cfg = PairMapMovieConfig(n_min=args["n_min"], n_max=args["n_max"], frames=args["frames"],
                             spacing=args["spacing"], every_n_until=args.get("every_n_until", 0))
    ranges = [tuple(r) for r in (args.get("range") or [(0.5, 0.7)])]
    r = next((r for r in ranges if f"{r[0]:g}-{r[1]:g}" in os.path.basename(path)), ranges[0])
    return movie_n_values(cfg), float(args["fps"]), float(args.get("hold_end", 2.0)), r


def frame_cusps(st, n: int, p_range: tuple[float, float]) -> dict:
    """The cusps of n inside p_range (mirrors 1 - p included), with z and the kink position u."""
    c = st.cusps(n, columns=["i", "j", "pstar", "decided_by", "S_minus", "ln_fi"])
    k = c["decided_by"] != "symmetry"
    p, w, sm, lfi = c["pstar"][k], (c["j"] - c["i"])[k].astype(float), c["S_minus"][k], c["ln_fi"][k]
    u = np.sign(sm) * np.exp(np.clip(np.log(np.abs(sm) + 1e-300) - np.log(w) - lfi, -700, 700))
    z = w / (2 * np.sqrt(n * p * (1 - p)))
    p = np.concatenate([p, 1 - p])                          # E(p) = E(1-p): (i, j) at p mirrors to 1 - p
    z, u = np.concatenate([z, z]), np.concatenate([u, -1 - u])
    keep = (p >= p_range[0]) & (p <= p_range[1]) & (p != 0.5)
    return dict(p=p[keep], z=z[keep], u=u[keep])


def pitch(z: np.ndarray) -> np.ndarray:
    return 110 * 2 ** (z / 3.6 * 4.5)                       # sonify.py clip 1: z = 3 -> 1.6 kHz, 4.2 -> 4.5 kHz


def render(ns: list[int], fps: float, hold_end: float, p_range: tuple[float, float], z_wide: float = 2.0,
           base_hz: float = 55.0, bed_cap_hz: float = 3000.0, ring: float = 0.6, verbose: bool = False
           ) -> tuple[np.ndarray, np.ndarray]:
    """(sparkle, bed) as (samples, 2) stereo arrays of the movie's exact length."""
    from obd_core import store

    st = store.Store()
    hop = SR / fps
    total = int(round((len(ns) / fps + max(hold_end, 0)) * SR))
    sparkle = np.zeros((total + int(ring * SR), 2))
    win = np.hanning(int(2 * hop))                          # bed: overlap-add, hop one frame, 50% overlap
    pad = len(win)
    bed = np.zeros(pad + total + 2 * len(win))
    tail = int(ring * SR)
    t_ring = np.arange(tail) / SR
    env = (1 - np.exp(-t_ring / 0.002)) * np.exp(-t_ring / 0.12)
    for f, n in enumerate(ns):
        c = frame_cusps(st, n, p_range)
        t0 = f * hop
        # bed: ordinary cusps near 1/2 as harmonics 1 + 2x of base_hz (phase from absolute time)
        x = n * np.abs(c["p"][c["z"] < z_wide] - 0.5)
        x = np.unique(np.round(x, 9))                       # a mirror pair is one partial, not two
        x = x[base_hz * (1 + 2 * x) < bed_cap_hz]
        i0 = int(round(t0 - hop / 2))                       # window centred on the frame; bed starts pad early
        tt = (i0 + np.arange(len(win))) / SR
        seg = (np.sin(2 * np.pi * base_hz * (1 + 2 * x)[:, None] * tt[None, :]) / np.sqrt(1 + 2 * x)[:, None]).sum(0)
        bed[pad + i0:pad + i0 + len(win)] += seg * win
        # sparkle: wide cusps as pings, onsets spread over the frame, deterministic per n
        wide = c["z"] >= z_wide
        rng = np.random.default_rng(n)
        for p, z, u, d in zip(c["p"][wide], c["z"][wide], c["u"][wide], rng.random(wide.sum())):
            fz = pitch(z)
            a = min(1.0, abs(2 * u + 1))                    # 0 symmetric (white) .. 1 lopsided
            tone = np.sin(2 * np.pi * fz * t_ring) + 0.6 * a * np.sin(2 * np.pi * 2.76 * fz * t_ring)
            pan = (p - p_range[0]) / (p_range[1] - p_range[0])
            gains = np.array([np.cos(pan * np.pi / 2), np.sin(pan * np.pi / 2)])
            s0 = int(t0 + d * hop)
            sparkle[s0:s0 + tail] += (tone * env)[:, None] * gains[None, :]
        if verbose and (f % 50 == 0 or f == len(ns) - 1):
            print(f"  frame {f + 1}/{len(ns)}  n = {n}: {len(x)} bed partials, {int(wide.sum())} wide cusps",
                  flush=True)
    # end hold: the last frame's chord held (same partials, continuous phase) and faded out
    if hold_end > 0:
        i0 = int(round(len(ns) * hop - hop / 2))
        tt = (i0 + np.arange(total - i0)) / SR
        hold = (np.sin(2 * np.pi * base_hz * (1 + 2 * x)[:, None] * tt[None, :]) / np.sqrt(1 + 2 * x)[:, None]).sum(0)
        fade = np.r_[np.hanning(len(win))[:len(win) // 2], np.ones(max(0, len(tt) - len(win) // 2))][:len(tt)]
        fade *= np.minimum(1, (len(tt) - np.arange(len(tt))) / (0.5 * hold_end * SR))
        bed[pad + i0:pad + total] += hold * fade
    bed = bed[pad:pad + total]
    sparkle = sparkle[:total]
    return sparkle, np.repeat(bed[:, None], 2, axis=1)


def norm(x: np.ndarray) -> np.ndarray:
    return x / (np.abs(x).max() + 1e-12)


def write_wav(path: str, x: np.ndarray) -> None:
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("movie", help="A pair-map-movie .mp4 (its settings are read from the file).")
    ap.add_argument("--z-wide", type=float, default=2.0, help="Cusps at least this wide (in sd) sparkle.")
    ap.add_argument("--sparkle-gain", type=float, default=0.8)
    ap.add_argument("--bed-gain", type=float, default=0.45)
    ap.add_argument("--base-hz", type=float, default=55.0, help="The bed's fundamental.")
    ap.add_argument("-o", "--output", help="Movie with sound (default sonification/out/<movie>-sound.mp4).")
    a = ap.parse_args()

    ns, fps, hold_end, p_range = movie_schedule(movie_settings(a.movie), a.movie)
    print(f"{a.movie}: {len(ns)} frames, n = {ns[0]}..{ns[-1]}, {fps:g} fps, hold {hold_end:g} s, "
          f"p in [{p_range[0]:g}, {p_range[1]:g}]")
    sparkle, bed = render(ns, fps, hold_end, p_range, z_wide=a.z_wide, base_hz=a.base_hz, verbose=True)
    mix = a.sparkle_gain * norm(sparkle) + a.bed_gain * norm(bed)
    mix *= 0.89 / max(np.abs(mix).max(), 1e-12)

    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, os.path.splitext(os.path.basename(a.movie))[0])
    write_wav(stem + "-sound.wav", mix)
    write_wav(stem + "-sparkle.wav", 0.89 * norm(sparkle))
    write_wav(stem + "-bed.wav", 0.89 * norm(bed))
    out = a.output or stem + "-sound.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.movie, "-i", stem + "-sound.wav",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-map_metadata", "0", "-movflags", "+faststart", out], check=True)
    print(f"wrote {stem}-sound.wav, -sparkle.wav, -bed.wav ({len(mix) / SR:.2f} s) and {out}")


if __name__ == "__main__":
    main()
