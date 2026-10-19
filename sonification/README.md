# Sonification

Five audio prototypes of the cusp structure: mono WAVs, 44.1 kHz, 16-bit, 25–40 s each.
Made in a claude.ai chat on 2026-10-17 and brought here, like everything in this repo, as a
way of *presenting* the data, not as evidence.

```bash
cd sonification && python sonify.py
```

writes the five clips to `sonification/out/` (not in git) in about 3 s. Tie points and cusps
come from OBD-core's shared store (`~/obd-data`, certified verdicts); `python sonify.py --quick`
uses `sonify_ties.py` instead, the chat's quick double-precision finder (no certification, memory
~n³/4 floats, so not past n ≈ 800). Both give the same audio: over n = 20, 100, 200, 300 and
600 the quick finder's cusps are exactly the certified ones (7, 34, 71, 105, 212), and its 900
tie points at n = 60 agree with the store's in p* and D to ~1e-14. Clip 3's E′(p) comes from
`obd_core.E_slopes_at`; the chat's own formula agreed with it to 1.6e-10 relative.

| clip | what you hear |
|---|---|
| `1_pair_space_spectrogram` (41 s) | the pair-space plot played left to right. Time = p from ½ to 0.66; pitch = width in standard deviations, z = (j−i)/(2√(npq)), log-scale from 110 Hz to ~2.5 kHz. Pools every 4th n from 200 to 400: 5,430 cusps, 71 of them wide (z ≥ 2), played louder and longer. |
| `2_tie_clicks_n60` (37 s) | a sweep of p from ½ to 0.74 at n = 60. Every tie point is a noise click, loudness ∝ log D (the slope jump); every cusp a bell (880 + 1320 Hz). The bands come out as a pulse of about one beat per second. Quiet by nature. |
| `3_Eprime_waveform_n10_to_400` (26 s) | E′(p) used directly as the waveform: one sweep of p ∈ [½, 1) per 55 Hz cycle, each n held 0.2 s, n = 10, 13, …, 400. |
| `4_cusp_chords_n4_to_300` (33 s) | one chord per n, held 0.11 s. A cusp at x = n(p−½) sounds harmonic 1+2x of 110 Hz, so band m ≈ harmonic m+1. The low notes lock in as n grows: the n-invariance as sound. |
| `5_timbre_binomial_then_OBD_n20` (25 s) | additive synthesis at n = 20: partial k+1 of 110 Hz has amplitude f_p(k). Two 12 s sweeps of p from 0.03 to 0.97, first the binomial masses as they are, then sorted (the OBD). For the OBD the spectral centroid is E(p), so the second sweep is dullest at p = ½. |

## Soundtracks for the pair-map movies

```bash
python sonification/soundtrack.py plots/PairMap-emergence-0-1-n10000-sqrt.mp4
```

makes a soundtrack with n as the time axis, one slice of sound per frame. It reads the movie's own
settings (stamped in the file by `pair-map-movie`), so it lines up with any movie without being told
the frame list, fps or end hold. It writes the mix and two stems (`-sparkle.wav`, `-bed.wav`) to
`sonification/out/`, plus a copy of the movie with the sound (`-sound.mp4`). About 6 s for a
514-frame movie.

- **Sparkle**: each wide cusp in the frame (z ≥ 2; ordinary cusps stay below ~1.3 and the wide ones
  sit at 3–4.2, nothing in between) is a short bell ping. Pitch = height in the picture (z: ~1.6 to
  4.5 kHz), left–right = where it sits across the picture, brightness = how lopsided its V is (a
  white, symmetric cusp is a pure tone; aquamarine or yellow adds an inharmonic overtone). Wide cusps
  flicker from n = 59, one or two in about half the frames, and grow to ~40 per side at n = 10,000,
  so the sparkle thickens through the movie.
- **Bed**: the ordinary cusps near p = ½ as a held chord, mapped as in clip 4 (a cusp at
  x = n(p−½) → harmonic 1+2x of 55 Hz). It builds over the first few seconds as the bands lock in,
  then holds: the n-invariance. During the end hold it sustains and fades.

`--z-wide`, `--sparkle-gain`, `--bed-gain` and `--base-hz` adjust it.

Ideas not yet done: lay clip 1 or 4 under the pair-space movie; use z rather than raw width as
the vertical axis in both; extend clip 4 past n = 300 (the store has certified cusps to 10,000).
