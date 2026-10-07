# data/

Everything in this folder is generated, and none of it is in git (only this README is). Rebuild it
all with one command from the repository root, in the `obd` environment:

```bash
python OBDsaveSourceData.py --all
```

That takes about 3 minutes on an 8-core machine and needs about 8 GB of disk. Rerunning it skips
anything already built; to rebuild one part from scratch, delete it first.

| what | path | size | built by | used by |
|---|---|---|---|---|
| tie tables: every tie point of each n = 2–1000, with E, exact slopes, log₁₀ D and certified cusp flags | `tie_points/n=NNNNN/part.parquet` + `tie_points/manifest.json` | 5.1 GB | `--save-tie-points` (~2 min) | explorers 5–7, heatmaps, Qt GUI tie lines, cusp table |
| cusp table: every certified cusp of every n | `tie_cusps.parquet` | 12 MB | `--save-cusp-data` (~6 s) | `plot_last_cusp_features.py` |
| graph data: masses, their ranking and E on a 1001-point p grid, n = 2–1000 | `graph_data_p01001.h5` (HDF5) | 3.0 GB | `--save-graph-data` (~40 s) | explorers 1–4, graph export, Qt GUI |

`--all` runs the three in the order graph, tie points, cusp table. Formats and columns are
documented in [OBD_data_map.md](../OBD_data_map.md).

Analysis PDFs written by the plotting scripts (`last_cusp_*.pdf`, `n_local_min_tie_points.pdf`) also
land here; rerun `plot_last_cusp_features.py` or `plot_n_local_min_tie_points.py` to refresh them.
