# Archived scripts

These scripts predate the current data pipeline and no longer run: they read monolithic pickles
(`tie_points.pkl`, `graph_data.pkl`) that have not been produced since May 2026. They are kept for
reference only and are not packaged or maintained.

Their jobs are now done by `OBDExplorerPlus.py` (graph and heatmap exports, HTML explorers) on
top of the tie tables and graph shards that `OBDsaveSourceData.py` builds; see
[OBD_data_map.md](../OBD_data_map.md).

| script | what it did |
|---|---|
| `OBDtiePointsExplorer1.py` | interactive tie-point explorer |
| `OBDGraphWithTieFastplotlib.py` | E/n graph with tie lines, fastplotlib backend |
| `OBDGraphWithTieMatplotlib.py` | E/n graph with tie lines, matplotlib backend |
| `OBDplotTiePoints.py` | tie-point count vs n, KDE of p |
| `OBDplotTiePoints2.py` | tie-point pair plots |
| `OBDgraphExplorer1 swap points lines archive.py` | an older variant of the graph explorer |
