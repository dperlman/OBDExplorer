# Example plots

The images linked from the [demo page](https://dperlman.github.io/OBDExplorer/), with the exact
commands that made them (run from the repository root, after `python OBDsaveSourceData.py --all`
has built the data). Every PNG made by `OBDExplorerPlus.py` also carries its command and all its
settings inside the file, as PNG text chunks; read them with

```bash
python -m obd_explorer.png_metadata docs/plots/N-pColorTieGraphFull.png
```

| image | command |
|---|---|
| `N-pColorTieGraphFull.png` | `python OBDExplorerPlus.py export -o docs/plots/N-pColorTieGraphFull.png --dpi 500 --vp-range full --tie-color-left j --tie-color-right i` |
| `N-pColorTieGraphHalf.png` | `python OBDExplorerPlus.py export -o docs/plots/N-pColorTieGraphHalf.png --dpi 500 --tie-color-left j --tie-color-right i` |
| `N-pColorTieGraphDetrendFull.png` | `python OBDExplorerPlus.py export -o docs/plots/N-pColorTieGraphDetrendFull.png --backend matplotlib --dpi 500 --vp-range full --endpoint-chord --tie-direction down --tie-color-left j --tie-color-right i` |
| `N-pHeatmapLog10D.png` | `python OBDExplorerPlus.py heatmap -o docs/plots/N-pHeatmapLog10D.png --value d --colormap hsv --trim-color-range-percent 3 --p-min 0.5 --p-max 0.6 --p-steps 3001` |
| `N-pHeatmapAnnotated.png` | `python OBDExplorerPlus.py heatmap -o docs/plots/N-pHeatmapAnnotated.png --value eslope_n --colormap prism --p-min 0 --p-max 1 --height-in 8` |
| `N-tieHeatmapExact.png` | `python OBDExplorerPlus.py tie-heatmap -o docs/plots/N-tieHeatmapExact.png --pixel-mode exact --value d --colormap hsv` |
| `N-pFirstCuspWithinR1e-3.png` | `python OBDExplorerPlus.py cusp-proximity -o docs/plots/N-pFirstCuspWithinR1e-3.png --r 0.001 --cusps-csv ../ordered-binomial-cusps/cusps/cusps_all.csv` |
| `N-pFirstCuspWithinR1e-4.png` | `python OBDExplorerPlus.py cusp-proximity -o docs/plots/N-pFirstCuspWithinR1e-4.png --r 0.0001 --cusps-csv ../ordered-binomial-cusps/cusps/cusps_all.csv` |
| `N-pFirstCuspWithinR1e-5.png` | `python OBDExplorerPlus.py cusp-proximity -o docs/plots/N-pFirstCuspWithinR1e-5.png --r 0.00001 --cusps-csv ../ordered-binomial-cusps/cusps/cusps_all.csv` |
| `N-pFirstCuspWithinR1e-6.png` | `python OBDExplorerPlus.py cusp-proximity -o docs/plots/N-pFirstCuspWithinR1e-6.png --r 0.000001 --cusps-csv ../ordered-binomial-cusps/cusps/cusps_all.csv --points-per-r 20 --marker-size 0.1 --dpi 600` |
| `N-pFirstCuspWithinR1e-6-n20000.png` | `python OBDExplorerPlus.py cusp-proximity -o docs/plots/N-pFirstCuspWithinR1e-6-n20000.png --r 0.000001 --cusps-csv ../ordered-binomial-cusps/cusps/cusps_all.csv --points-per-r 20 --marker-size 0.1 --dpi 600 --extend-to 20000` (hours; the search past n = 5000 is cached in `data/cusp_windows/`, so a rerun takes seconds) |
| `N-pFirstCuspWithinR-scaled.png` | `python OBDExplorerPlus.py cusp-proximity -o docs/plots/N-pFirstCuspWithinR-scaled.png --r 0.001 0.0001 0.00001 0.000001 --scale-sqrt-r --points-per-r 20 --marker-size 0.1 --dpi 600 --cusps-csv ../ordered-binomial-cusps/cusps/cusps_all.csv` |
| `N-pFirstCuspWithinR-timesR.png` | `python OBDExplorerPlus.py cusp-proximity -o docs/plots/N-pFirstCuspWithinR-timesR.png --r 0.001 0.0001 0.00001 0.000001 --r-power 1 --points-per-r 20 --marker-size 0.1 --dpi 600 --cusps-csv ../ordered-binomial-cusps/cusps/cusps_all.csv` |

The `N-pFirstCuspWithinR` plots also read the certified cusp catalogue of
[ordered-binomial-cusps](https://github.com/dperlman/ordered-binomial-cusps) (`cusps/cusps_all.csv`,
n ≤ 5000, built there by `cusps_fast.py`) for the n past this repo's tie tables; for n ≤ 1000 it
holds exactly the same cusps as `data/tie_cusps.parquet`.

`N-pColorTieGraphHalf_old.png` and `N-pColorTieGraphDetrendFull_old.png` are the originals from
May 2026, kept as they were. Their settings were not recorded, and the current code doesn't
reproduce them exactly: the regenerated versions above differ in some fill colours and, for the
detrended one, in figure layout. (`N-pColorTieGraphFull.png` was reproduced, so it replaced its
original.)
