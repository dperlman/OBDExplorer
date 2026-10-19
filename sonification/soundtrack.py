"""A soundtrack for a pair-map movie: n is the time axis, one slice of sound per frame.

    python sonification/soundtrack.py plots/PairMap-0.3-0.7-n3000-3299-log0.2-32.mp4

reads the movie's own settings (the obd:cli_args its encoder stamped in the file), rebuilds its
frame list (pair_map_movie.movie_n_values), renders a stereo WAV exactly as long as the movie's
frames and writes a copy of the movie with that sound into sonification/out/.  An end hold, if the
movie has one, is cut off.  The cusps are the certified ones of OBD-core's shared store; the axis
p = 1/2 is left out.  Terms as in the pair-space plots: a pair (i, j) has width j - i, its column
is i + j, and column k means i + j = n + k (k columns right of the axis; the mirror side is -k).

Two layers, also written as separate WAVs (stems) for mixing elsewhere:

    sparkle   every wide cusp of the frame (width >= wide * sqrt(n); the cusp curve's cusps stay
              below ~1.25 sqrt(n) and the wide ones, on the lower edges of the blue arcs, sit at
              ~3-4.2 sqrt(n), nothing in between) is a short bell ping: pitch from width/sqrt(n)
              (log scale: 3 -> ~1.5 kHz, 4.2 -> ~4.4 kHz), stereo position from its p* (below),
              brightness from its cusp colour (u): a white, symmetric cusp is a pure tone, an
              aquamarine or yellow one carries a strong inharmonic overtone.  Onsets are spread
              over the frame.
    chord     the cusps on the cusp curve as one held chord, one note per column: a cusp at p*
              sounds 1 + 2 n (p* - 1/2) times base_hz, which for column k is about harmonic k + 1
              (column 1 the fundamental), nudged by the tie point's offset from its grid position.
              A column with two cusps gives two close notes that beat.  Cross-faded frame to frame;
              it builds as columns fill in at small n and then holds (the n-invariance).

The stereo stage.  The cusps are mirror-symmetric about p = 1/2 (E(p) = E(1-p)), so the half
p >= 1/2 holds all the information.  stage "half" (the default) sounds each cusp once, panned from
p = 1/2 (left) to stage_max (right; default 0.6525, the S5 cusp limit, so the cusps use the whole
stage), whichever side of the picture it is drawn on.  stage "mirror" pans across the picture
instead (p_min left, p_max right), so a cusp drawn on both sides of 1/2 pings twice, mirrored.
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


def movie_schedule(args: dict, path: str) -> tuple[list[int], float, tuple[float, float]]:
    """(n per frame, fps, (p_min, p_max)) of the movie."""
    cfg = PairMapMovieConfig(n_min=args["n_min"], n_max=args["n_max"], frames=args["frames"],
                             spacing=args["spacing"], every_n_until=args.get("every_n_until", 0))
    ranges = [tuple(r) for r in (args.get("range") or [(0.5, 0.7)])]
    r = next((r for r in ranges if f"{r[0]:g}-{r[1]:g}" in os.path.basename(path)), ranges[0])
    return movie_n_values(cfg), float(args["fps"]), r


def frame_cusps(st, n: int, p_range: tuple[float, float], mirror: bool) -> dict:
    """The cusps of n drawn in the picture, with width/sqrt(n) and u: with mirror, every cusp inside
    p_range (mirrors 1 - p included); without, each cusp once at its p* >= 1/2, if either it or its
    mirror is inside p_range."""
    c = st.cusps(n, columns=["i", "j", "pstar", "decided_by", "S_minus", "ln_fi"])
    k = c["decided_by"] != "symmetry"
    p, m, sm, lfi = c["pstar"][k], (c["j"] - c["i"])[k].astype(float), c["S_minus"][k], c["ln_fi"][k]
    u = np.sign(sm) * np.exp(np.clip(np.log(np.abs(sm) + 1e-300) - np.log(m) - lfi, -700, 700))
    w = m / np.sqrt(n)
    if not mirror:
        keep = ((p >= p_range[0]) & (p <= p_range[1])) | ((1 - p >= p_range[0]) & (1 - p <= p_range[1]))
        return dict(p=p[keep], w=w[keep], u=u[keep])
    p = np.concatenate([p, 1 - p])                          # E(p) = E(1-p): (i, j) at p mirrors to 1 - p
    w, u = np.concatenate([w, w]), np.concatenate([u, -1 - u])
    keep = (p >= p_range[0]) & (p <= p_range[1])
    return dict(p=p[keep], w=w[keep], u=u[keep])


def pitch(w: np.ndarray) -> np.ndarray:
    return 110 * 2 ** (1.25 * w)                            # width/sqrt(n) 3 -> 1.5 kHz, 4.2 -> 4.4 kHz


def render(ns: list[int], fps: float, p_range: tuple[float, float], wide: float = 2.0, base_hz: float = 55.0,
           stage: str = "half", stage_max: float = 0.6525, chord_cap_hz: float = 3000.0, ring: float = 0.6,
           verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(sparkle, chord) as (samples, 2) stereo arrays, len(ns) / fps seconds long."""
    from obd_core import store

    st = store.Store()
    hop = SR / fps
    total = int(round(len(ns) / fps * SR))
    win = np.hanning(int(2 * hop))                          # chord: overlap-add, hop one frame, 50% overlap
    pad = len(win)
    chord = np.zeros(pad + total + 2 * len(win))
    tail = int(ring * SR)
    sparkle = np.zeros((total + tail, 2))
    t_ring = np.arange(tail) / SR
    env = (1 - np.exp(-t_ring / 0.002)) * np.exp(-t_ring / 0.12)
    left, right = (0.5, stage_max) if stage == "half" else p_range
    for f, n in enumerate(ns):
        c = frame_cusps(st, n, p_range, mirror=stage == "mirror")
        t0 = f * hop
        # chord: the cusp curve's cusps, one note per column (phase from absolute time)
        x = n * np.abs(c["p"][c["w"] < wide] - 0.5)
        x = np.unique(np.round(x, 9))                       # a cusp and its mirror are one note
        x = x[base_hz * (1 + 2 * x) < chord_cap_hz]
        i0 = int(round(t0 - hop / 2))                       # window centred on the frame
        tt = (i0 + np.arange(len(win))) / SR
        seg = (np.sin(2 * np.pi * base_hz * (1 + 2 * x)[:, None] * tt[None, :]) / np.sqrt(1 + 2 * x)[:, None]).sum(0)
        chord[pad + i0:pad + i0 + len(win)] += seg * win
        # sparkle: wide cusps as pings, onsets spread over the frame, deterministic per n
        wd = c["w"] >= wide
        rng = np.random.default_rng(n)
        for p, w, u, d in zip(c["p"][wd], c["w"][wd], c["u"][wd], rng.random(wd.sum())):
            fw = pitch(w)
            a = min(1.0, abs(2 * u + 1))                    # 0 white (symmetric V) .. 1 aquamarine / yellow
            tone = np.sin(2 * np.pi * fw * t_ring) + 0.6 * a * np.sin(2 * np.pi * 2.76 * fw * t_ring)
            pan = min(1.0, max(0.0, (p - left) / (right - left)))
            gains = np.array([np.cos(pan * np.pi / 2), np.sin(pan * np.pi / 2)])
            s0 = int(t0 + d * hop)
            sparkle[s0:s0 + tail] += (tone * env)[:, None] * gains[None, :]
        if verbose and (f % 50 == 0 or f == len(ns) - 1):
            print(f"  frame {f + 1}/{len(ns)}  n = {n}: {len(x)} chord notes, {int(wd.sum())} wide cusps", flush=True)
    chord = chord[pad:pad + total]
    return sparkle[:total], np.repeat(chord[:, None], 2, axis=1)


def norm(x: np.ndarray) -> np.ndarray:
    return x / (np.abs(x).max() + 1e-12)


def write_wav(path: str, x: np.ndarray) -> None:
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("movie", help="A pair-map-movie .mp4 (its settings are read from the file).")
    ap.add_argument("--wide", type=float, default=2.0, help="Cusps at least this wide, in units of sqrt(n), sparkle.")
    ap.add_argument("--sparkle-gain", type=float, default=0.8)
    ap.add_argument("--chord-gain", type=float, default=0.45)
    ap.add_argument("--base-hz", type=float, default=55.0, help="The chord's fundamental (column 1).")
    ap.add_argument("--stage", choices=("half", "mirror"), default="half",
                    help="half: each cusp once, panned from p = 1/2 to --stage-max; mirror: across the picture, "
                         "a cusp and its mirror both.")
    ap.add_argument("--stage-max", type=float, default=0.6525,
                    help="half: the p at the right of the stage (default the S5 cusp limit).")
    ap.add_argument("-o", "--output", help="Movie with sound (default sonification/out/<movie>-sound.mp4).")
    a = ap.parse_args()

    ns, fps, p_range = movie_schedule(movie_settings(a.movie), a.movie)
    print(f"{a.movie}: {len(ns)} frames, n = {ns[0]}..{ns[-1]}, {fps:g} fps, p in [{p_range[0]:g}, {p_range[1]:g}]")
    sparkle, chord = render(ns, fps, p_range, wide=a.wide, base_hz=a.base_hz, stage=a.stage,
                            stage_max=a.stage_max, verbose=True)
    mix = a.sparkle_gain * norm(sparkle) + a.chord_gain * norm(chord)
    mix *= 0.89 / max(np.abs(mix).max(), 1e-12)

    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, os.path.splitext(os.path.basename(a.movie))[0] + ("-mirror" if a.stage == "mirror" else ""))
    write_wav(stem + "-sound.wav", mix)
    write_wav(stem + "-sparkle.wav", 0.89 * norm(sparkle))
    write_wav(stem + "-chord.wav", 0.89 * norm(chord))
    out = a.output or stem + "-sound.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.movie, "-i", stem + "-sound.wav",
                    "-map", "0:v", "-map", "1:a", "-frames:v", str(len(ns)), "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-t", f"{len(ns) / fps:.6f}", "-map_metadata", "0", "-movflags", "+faststart",
                    out], check=True)
    print(f"wrote {stem}-sound.wav, -sparkle.wav, -chord.wav ({len(mix) / SR:.2f} s) and {out}")


if __name__ == "__main__":
    main()
