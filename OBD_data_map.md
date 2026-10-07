# OBD Data Map

This document is a reference for the data files, data structures and dtypes written by `OBDsaveSourceData.py`, and for where the numbers in them come from. All of the mathematics (tie points, E, slopes, cusp verdicts) comes from [OBD-core](https://github.com/dperlman/OBD-core) (`obd_core`); nothing here re-derives it.

## Quick Index

- [High-level data outputs](#high-level-data-outputs)
- [Pathway 1: tie tables](#pathway-1-tie-tables)
  - [Tie manifest](#tie-manifest)
  - [Tie table columns](#tie-table-columns)
  - [How the tie values are computed](#how-the-tie-values-are-computed)
  - [Tie CSV log columns](#tie-csv-log-columns)
- [Pathway 2: graph data shards](#pathway-2-graph-data-shards)
- [Pathway 3: cusp table](#pathway-3-cusp-table)
- [Loaders](#loaders)
- [Practical interpretation notes](#practical-interpretation-notes)

### Note on Quick Index links (Cursor / VS Code)

The built-in Markdown preview (same stack Cursor inherits) assigns each heading an `id` using the **GitHub-style slug** in [`slugify.ts`](https://github.com/microsoft/vscode/blob/main/extensions/markdown-language-features/src/slugify.ts): lowercase, strip punctuation with a large regex (same idea as the [`github-slugger`](https://github.com/Flet/github-slugger) package), then replace whitespace with `-`. Characters such as `:` and `+` are **removed** (not turned into `-`), so e.g. `Pathway 1: tie tables` becomes `pathway-1-tie-tables`. The fragments in the index above are written to match that algorithm.

## High-level data outputs

- `data/tie_points/`
  - one Parquet table per n: `n=NNNNN/part.parquet` (every tie point of n, both halves)
  - manifest: `manifest.json`
  - built by `OBDsaveSourceData.py --save-tie-points`
- `data/tie_cusps.parquet`
  - every certified cusp of every n, one table
  - built by `OBDsaveSourceData.py --save-cusp-data` (from the tie tables)
- `data/graph_data_shards/`
  - per-`n` graph data shards (pickle) for the graph explorers
  - manifest: `0000_manifest_p<steps>.pkl`
- `log/`
  - timestamped CSV run logs for the tie pathway

All of `data/` is git-ignored and rebuilt by the commands above.

## Pathway 1: tie tables

Code path: `save_tie_points(...)` → `_compute_tie_shard(...)` → `_tie_table_for_n(...)` → `obd_core.tie_table(n, both_halves=True)`.

Parquet settings match the ordered-binomial-cusps plotting datasets: zstd, `BYTE_STREAM_SPLIT` on the float columns, dictionary pages off (they defeat byte-stream-split); `decided_by` is stored as an Arrow dictionary column. Each file's schema metadata carries `format`, `n` and `obd_core_version`. At n = 1000 a table has 500,001 rows and every column can be read on its own, so a loader that needs only `p` and `log10_D` reads only those.

### Tie manifest

`data/tie_points/manifest.json`:

- `format`: `"obd.tie_points.parquet.v3"`
- `created_at`, `updated_at`: timestamp strings
- `obd_core_version`: version of OBD-core that computed the tables
- `n_entries[str(n)]`:
  - `n`: `int`
  - `path`: table path, relative to the manifest
  - `tie_points`: `int`
  - `cusps`: `int` (certified local minima of E)
  - `max_cusp_p`: `float | null` (p of the last cusp)
  - `n_checked`: `int` (tie points whose verdict needed interval or exact arithmetic)
  - `n_unresolved`: `int` (0 so far)
  - `updated_at`

The loaders refuse anything else. The two pickle formats that came before are gone: v1 (before 2026-10-05) held finite-difference slopes that were wrong, with negative slope jumps at most tie points, missed cusps and impossible "maximum" labels; v2 (2026-10-05) held correct values as pickled lists of dicts.

### Tie table columns

Rows are the tie points of n sorted by `p`, both halves of (0, 1).

| column | dtype | meaning |
|---|---|---|
| `p` | float64 | the tie point |
| `i`, `j` | int16 | its pair; the center tie `p = 1/2` (stored as exactly 0.5) carries the canonical pair `((n-1)//2, n-(n-1)//2)` and stands for all mirror pairs `i + j = n` |
| `E` | float64 | E(n, p) at the tie point |
| `slope_left`, `slope_right` | float64 | E'₋ and E'₊, the exact one-sided slopes |
| `log10_D` | float64 | log₁₀ of the slope jump D = E'₊ − E'₋ = (j−i)·f(i)/(p q) > 0 |
| `is_cusp` | bool | certified local minimum of E at this tie point |
| `decided_by` | string | `double`, `iv50`/`iv100`/`iv200` (interval arithmetic at that many digits), `exact` (integer arithmetic for adjacent pairs, whose p* is rational; needed at n = 2, where S₋ is exactly 0), `symmetry` (the center tie) or `UNRESOLVED` |

**Use `log10_D` for the slope jump, never `slope_right - slope_left`.** D spans hundreds of orders of magnitude (the median at n = 1000 is below double range) while the slopes are O(1)–O(n), so the difference of the two slopes is 0 or rounding noise for most tie points. `log10_D` comes from the pair mass and is exact.

There is no "maximum": a tie point is never a local maximum of E (the kink is always convex, D > 0), so every tie point is either a cusp or neither.

### How the tie values are computed

All of it comes from `obd_core.tie_table(n, both_halves=True)`, the same code [ordered-binomial-cusps](https://github.com/dperlman/ordered-binomial-cusps) uses:

1. Every pair `0 <= i < j <= n` with `i + j > n` (so p* > 1/2) is screened by a numba kernel: masses by recurrence from the mode, normalised by their own sum, ranks by a two-pointer merge with the ranking just left of p*.
2. The left slope is S₋/(p q) with S₋ = Σ wₖ f(k)(k − n p); the right slope adds the kink (j−i)·f(i)/(p q). No finite differences.
3. A tie point is a cusp iff S₋ < 0 < S₊. That is decided in double precision with a margin; anything near the margin, or with masses whose order double precision cannot resolve, goes to mpmath interval arithmetic at 50, 100 and then 200 digits, and an adjacent pair that interval arithmetic cannot settle goes to exact integer arithmetic.
4. p = 1/2 is added as the axis row; the p* < 1/2 half is the mirror image under E(p) = E(1 − p): (i, j) at p* becomes (n−j, n−i) at 1 − p*, with the same E, D and verdict, and the slopes swapped and negated.

### Tie CSV log columns

`save_tie_points` writes `log/tie_points_verbose_<timestamp>.csv`: `run_started_at`, `n`, `compute_sec`, `write_sec`, `iter_total_sec`, `tie_points`, `cusps`, `max_cusp_p`, `n_checked`, `n_unresolved`.

## Pathway 2: graph data shards

Code path: `save_graph_data(...)` and `load_graph_data_from_shards(...)`. These feed the graph explorers (variants 1–4 and the Qt/headless graph export).

Manifest `0000_manifest_p<steps>.pkl`: `format` `"obd.graph_data.shards.v3"`, `created_at`, `updated_at`, `p_steps`, `p_values` (`float32`, 0..1), `n_min`, `n_max`, `shards_dir`, and `n_entries[str(n)]` with `n`, `shard_path`, `rows` (= `p_steps`), `p_steps`, `k_count` (= `n+1`), dtype descriptors (`dtype_y` `float32`, `dtype_perm` `uint16`, `dtype_expected_sorted` `float64`), `updated_at`.

Each shard (`format` `"obd.graph_data.n_shard.v3"`):

- `n`, `p_steps`
- `y`: `(p_steps, n+1)` `float32`, the binomial masses (for drawing)
- `perm`: `(p_steps, n+1)` `uint16`, the stable argsort of the masses (for drawing)
- `expected_sorted_by_p`: `(p_steps,)` `float64`, E(n, p) at each grid p, from `obd_core.E_slopes_at`

v3 (2026-10-06) took E from obd_core and dropped `expected_sorted_slope_by_p`, which v2 computed with `np.gradient` over the grid: a secant across every kink inside each grid step, not E'(p). The heatmap's `eslope_n` now computes the exact slope at each pixel instead (below).

## Pathway 3: cusp table

Code path: `save_cusp_table(...)` (`--save-cusp-data`); load with `load_cusp_table(...)`.

`data/tie_cusps.parquet` holds every certified cusp of every n, copied from the tie tables (nothing is recomputed), so cusp plots do not need to read every tie point. Schema metadata: `format` `"obd.tie_cusps.parquet.v5"`, `source_tie_manifest`, `created_at`, `obd_core_version`.

Columns: `n` (int16), `tie_index` (int32; signed, relative to the center tie `p = 1/2`, which is always a cusp, so `tie_index = 0` there), and the tie columns `p`, `i`, `j`, `E`, `slope_left`, `slope_right`, `log10_D`, `decided_by`. Rows are sorted by `n`, then `p`.

## Loaders

All in `OBDsaveSourceData.py`; tables are `dict[str, np.ndarray]` (column name → array; `decided_by` is an object array of strings).

- `load_tie_manifest(path)` → the manifest dict.
- `iter_tie_tables(path, n_list=None, columns=None, require_all=True, progress=None)` → yields `(n, table)`, one n at a time. Pass `columns` to read only what you need.
- `load_tie_tables(...)` → `{n: table}`, same arguments.
- `tie_center_index(table)` → the row of `p = 1/2`.
- `load_cusp_table(path, n_list=None, columns=None)` → one table for all cusps.
- `load_graph_data_from_shards(...)` → `format`, `n_min`, `n_max`, `p_steps`, `p_half_start`, `p_values`, `rows_by_n[n]` with `y`, `perm`, `expected_sorted_by_p`, and `manifest_path`.

## Practical interpretation notes

- Every tie point is either a cusp (`is_cusp`) or nothing; there are no maxima at tie points.
- For the kink size use `log10_D`; explorers and heatmaps show it as the `d` field.
- For E or its slope at arbitrary p, use `obd_core.E_slopes_at(n, p_array)`: it returns E and the exact one-sided slopes, which differ only where p falls on a tie point.
