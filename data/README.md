# data/

Everything in this folder is generated, and none of it is in git (only this README is). Rebuild it
all with one command from the repository root, in the `obd` environment:

```bash
python OBDsaveSourceData.py --all
```

That takes a few minutes on an 8-core machine. Rerunning it skips anything already built; to rebuild
one part from scratch, delete it first.

**Tie tables, cusps and tie positions are no longer kept here.** They live in the shared store,
`~/obd-data` (OBD-core's `obd_core.store`), which every plot reads through; `--save-tie-points` and
`--check-tie-points` fill and check the store. The old `tie_points/`, `tie_windows/` and
`cusp_windows/` directories were retired on 2026-10-12 after the store was checked against them
(identical, bit for bit).

| what | path | size | built by | used by |
|---|---|---|---|---|
| tie tables: every tie point of each n = 2–1000 | the store, `~/obd-data/ties/` | 5 GB | `--save-tie-points` fills the store | explorers 5–7, heatmaps, Qt GUI tie lines, cusp table |
| cusp table: every certified cusp of n = 2–1000, with its index among the tie points (a derived cache, built from the store) | `tie_cusps.parquet` | 12 MB | `--save-cusp-data` (~6 s) | `plot_last_cusp_features.py`, `cusp-proximity` plot |
| the cusp-proximity plot's search past the tables | the store (`cusps/`, `positions/`, with coverage) | grows with use | `cusp-proximity --extend-to N` (reruns read it) | `cusp-proximity --extend-to N` |
| graph data: masses, their ranking and E on a 1001-point p grid, n = 2–1000 | `graph_data_p01001.h5` (HDF5) | 3.0 GB | `--save-graph-data` (~40 s) | explorers 1–4, graph export, Qt GUI |

`--all` runs the three in the order graph, tie points, cusp table. Building the tie tables also
checks them: every row against proved invariants, then a sample against OBD-core's rigorous
reference. Run the same check on the data as it stands with
`python OBDsaveSourceData.py --check-tie-points`. Formats and columns are documented in
[OBD_data_map.md](../OBD_data_map.md).

Analysis PDFs written by the plotting scripts (`last_cusp_*.pdf`, `n_local_min_tie_points.pdf`) also
land here; rerun `plot_last_cusp_features.py` or `plot_n_local_min_tie_points.py` to refresh them.
