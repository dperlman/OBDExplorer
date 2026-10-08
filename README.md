# OBDExplorer

**Live demos: [dperlman.github.io/OBDExplorer](https://dperlman.github.io/OBDExplorer/)**:
seven interactive explorers and example plots in the browser, nothing to install. They are built
from this repo and published from [`docs/`](docs/).

Interactive visualization of the **ordered binomial distribution**: E(n,p) = Σ w_k f(k), where
f(k) are the Binomial(n,p) masses and w_k is the rank of f(k) among them. The plots show E, its
tie points (where two masses are equal and E has a kink), the slope jumps there, and the cusps
(the tie points that are local minima of E).

This repo is for **looking**, not proving. The research results live in
[ordered-binomial-cusps](https://github.com/dperlman/ordered-binomial-cusps), and all the
mathematics comes from [OBD-core](https://github.com/dperlman/OBD-core), which both repos share.

## Quick start

```bash
conda env create -f environment.yml     # once: creates the "obd" environment
conda activate obd
python OBDsaveSourceData.py --all       # once: builds data/ (~3 min, ~8 GB)
python OBDExplorerPlus.py               # interactive menu: HTML explorers, exports, heatmaps, GUI
```

`pip install -r requirements.txt` works instead of conda. The desktop GUI needs the optional
extras: `pip install ".[gui]"`.

## Using it

Everything runs through `OBDExplorerPlus.py`: with no arguments it opens a menu, or call a
subcommand directly (`--help` on each lists its options):

| command | what it makes |
|---|---|
| `python OBDExplorerPlus.py html --variant N -o FILE.html` | a self-contained HTML explorer (variants 1–7, described on the [demo page](https://dperlman.github.io/OBDExplorer/)); bare filenames go to `html/` |
| `python OBDExplorerPlus.py export -o FILE.png` | a static E/n graph with tie lines (PNG, PDF or SVG) |
| `python OBDExplorerPlus.py heatmap -o FILE.png --value V` | an N–p heatmap: `d` (log₁₀ of the slope jump at the nearest tie point), `l`/`r` (slopes), `i`/`j` (the pair), `ev_n` (E/n), `eslope_n` (the exact slope E′/n) |
| `python OBDExplorerPlus.py tie-heatmap -o FILE.png --value V` | an N–tie-index heatmap of the same tie values |
| `python OBDExplorerPlus.py cusp-proximity -o FILE.png --r R` | for each p, the first n with a cusp within r of p (points, log n); several `--r` values overlay; p that no n reaches sit on a pale row at the top. p runs from 0.5 to 0.657, above every cusp found (ordered-binomial-cusps FACTS S5). `--cusps-csv` adds that repo's certified catalogue `cusps/cusps_all.csv` for n = 1001–5000 |
| `python OBDExplorerPlus.py gui` | the interactive desktop explorer (needs the GUI extras) |

Analysis plots: `python plot_last_cusp_features.py` (features of the last cusp against n) and
`python plot_n_local_min_tie_points.py` (cusp counts against n) write PDFs to `data/`.

## Data

All the data is generated locally into [`data/`](data/) and is not in git. Rebuild everything
with `python OBDsaveSourceData.py --all`, or one part with `--save-tie-points`,
`--save-cusp-data` or `--save-graph-data`. [data/README.md](data/README.md) lists what each file
is, its size and what uses it; [OBD_data_map.md](OBD_data_map.md) documents the formats and
columns.

The tie values come from OBD-core and are exact: one-sided slopes from the ranking, the slope
jump as `log10_D`, and cusp flags proved by interval arithmetic where double precision is not
enough. Every rebuild checks each new table on all rows against bounds proved from the
definitions, then samples a few against OBD-core's rigorous reference. To check what's on disk
at any time (about 10 s):

```bash
python OBDsaveSourceData.py --check-tie-points
```
 Never recompute a slope jump as `slope_right - slope_left`: for most tie points it is far
below double precision. Use `log10_D`.

## Rebuilding the published site

GitHub Pages serves `docs/` from `main`. After a change that affects them, rebuild the pages with
the settings they were made with:

```bash
python OBDExplorerPlus.py html --variant 1 -o docs/OBDExplorer1.html --n-min 2 --n-max 100     # likewise variants 2, 3, 4
python OBDExplorerPlus.py html --variant 5 -o docs/OBDExplorer5.html --n-min 2 --n-max 200 --colorscale hsv   # likewise variant 6
python OBDExplorerPlus.py html --variant 7 -o docs/OBDExplorer7.html --n-min 2 --n-max 200 --p-steps 3001 --p-min 0.5 --p-max 0.6 --colorscale hsv
python OBDExplorerPlus.py heatmap -o docs/plots/N-pHeatmapLog10D.png --value d --colormap hsv --trim-color-range-percent 3 --p-min 0.5 --p-max 0.6 --p-steps 3001
python OBDExplorerPlus.py heatmap -o docs/plots/N-pHeatmapAnnotated.png --value eslope_n --colormap prism --p-min 0 --p-max 1 --height-in 8
python OBDExplorerPlus.py tie-heatmap -o docs/plots/N-tieHeatmapExact.png --pixel-mode exact --value d --colormap hsv
python OBDExplorerPlus.py cusp-proximity -o docs/plots/N-pFirstCuspWithinR1e-3.png --r 0.001 --cusps-csv ../ordered-binomial-cusps/cusps/cusps_all.csv
python OBDExplorerPlus.py cusp-proximity -o docs/plots/N-pFirstCuspWithinR1e-4.png --r 0.0001 --cusps-csv ../ordered-binomial-cusps/cusps/cusps_all.csv
```

The `export` graphs are listed with their commands in [docs/plots/README.md](docs/plots/README.md).
Every PNG made by `OBDExplorerPlus.py` also records its command and all its settings inside the
file (`python -m obd_explorer.png_metadata FILE.png` prints them), so any image can be rebuilt
from itself. Link any new page or plot from [`docs/index.html`](docs/index.html), and add it to
`docs/plots/README.md`.

## What's where

| path | what |
|---|---|
| `OBDExplorerPlus.py` | the entry point (menu and subcommands above) |
| `OBDsaveSourceData.py` | builds and loads everything in `data/` |
| `obd_explorer/` | the explorer implementations: HTML builders and exporters (`explorer*_*.py`), headless rendering (`render_headless.py`), the cusp-proximity plot (`cusp_proximity.py`), data loading (`grid.py`, `tie_data.py`, `html_data.py`) |
| `obd_explorer_qt_ui.py` | the desktop GUI |
| `plot_*.py` | analysis plots (PDFs into `data/`) |
| `docs/` | the GitHub Pages site: explorer pages, `plots/`, `index.html` |
| `data/`, `html/`, `plots/` | generated data and output, not in git (apart from placeholders); `html/` is where `html` writes by default |
| `log/` | per-run logs from the data builds; new ones are not tracked |
| `OBD_data_map.md` | data formats and columns |
| `archive/` | old scripts that no longer run, kept for reference |
| `OBDexplorer1.py`, `OBDexplorer3.py`, `OBD2Dprojection.py`, `OBDinteractiveBinomial.py`, `OBDPlots3d.py`, `OBDPlots4d.py`, `OBDsimpleGraph.py`, `OBDswapGraph1.py`, `OBDtSNE.py` | earlier standalone experiments (PCA projections, 3D/4D plots, simple graphs); they compute their own data |
| `OBDgraphExplorer1.py`, `OBDexplorer2.py`, `OBDGraphExplorerQT.py`, `OBDGraphWithTiePyQTGraph.py` | thin shims that forward to `OBDExplorerPlus.py` |
| `OBDtiePointFormulas.py`, `diagnostic_tie_p_method_agreement.py`, `test_exact_tie_point_agreement.py` | checks of the tie-point formulas against high precision |

## Environment and dependencies

- **Environment.** The `obd` conda environment (`environment.yml`) or `pip install -r requirements.txt`.
  Packaging metadata is in `pyproject.toml`.
- **OBD-core.** `requirements.txt` installs OBD-core at a pinned release tag. To pick up a core
  change, tag a release there and bump the pin here. How to use it (`tie_table`, `E_slopes_at`,
  what is certified, pitfalls) and how to check any value rigorously with
  `obd_core.reference` (e.g. `reference.tie(n, i, j)`): see the
  [OBD-core README](https://github.com/dperlman/OBD-core#readme). Never re-derive the
  mathematics here.
- **One numerical stack.** numpy, numba, llvmlite and mpmath are pinned to exact versions by
  OBD-core's `constraints.txt`, which `requirements.txt` applies (`environment.yml` repeats the
  pins for conda). With identical versions this repo and ordered-binomial-cusps compute
  bit-identical results, and numba's compiled-code cache, which is per numba version, stays
  valid. `import obd_core` warns if the environment differs.
- **Storage.** The tie tables are Parquet (`pyarrow`); the graph data is HDF5 (`h5py`).
