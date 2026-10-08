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
| `N-pFirstCuspWithinR.png` | `python OBDExplorerPlus.py cusp-proximity -o docs/plots/N-pFirstCuspWithinR.png` |

`N-pColorTieGraphHalf_old.png` and `N-pColorTieGraphDetrendFull_old.png` are the originals from
May 2026, kept as they were. Their settings were not recorded, and the current code doesn't
reproduce them exactly: the regenerated versions above differ in some fill colours and, for the
detrended one, in figure layout. (`N-pColorTieGraphFull.png` was reproduced, so it replaced its
original.)
