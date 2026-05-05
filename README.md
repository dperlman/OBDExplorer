# OBDExplorer

OBDExplorer is a research-oriented codebase for exploring the deep structure of
the Ordered Binomial Distribution (OBD) through data generation, geometric/tie
analysis, and interactive visualization tools.

This repository is organized around a few workflows:

- generate/load OBD-derived source data
- compute tie points and related geometric diagnostics
- inspect results in desktop and HTML explorers
- export plots and intermediate artifacts for analysis

## Key files and folders

- `OBDExplorerPlus.py`
  - Main entry point for running explorer variants and related workflows.
- `OBDsaveSourceData.py`
  - Core data-generation/loading pipeline used by explorers and diagnostics.
- `obd_explorer/`
  - Shared package with explorer implementations and core modules.
  - `explorer*_html.py`, `explorer*_export.py`: HTML explorer UI builders and export wiring.
  - `geometry.py`, `tie_data.py`, `model.py`, `numeric.py`: core computation and tie geometry logic.
  - `qt_graphics.py`, `render_headless.py`: rendering support for Qt/headless paths.
- `obd_explorer_qt_ui.py`
  - Qt UI integration for desktop interactive exploration.
- `OBDgraphExplorer1.py`, `OBDexplorer1.py`, `OBDexplorer2.py`, `OBDexplorer3.py`
  - Explorer-oriented scripts for different analysis/visualization paths.
- `diagnostic_*.py`, `plot_*.py`, `tie_*diagnostic*.py`, `test_exact_tie_point_agreement.py`
  - Diagnostics, plotting tools, and validation scripts.
- `html/`
  - Generated HTML explorer outputs (ignored by git in this repo setup).
- `html_old/`
  - Older generated HTML artifacts (also ignored by git).
- `data/`
  - Large local datasets and shard outputs (ignored by git).
- `log/`
  - Generated logs and verbose tie-point analysis outputs.
- `plots/`, `last_cusp_plots/`
  - Generated plot exports and analysis figures.

## Notes

- This repo is configured to keep large generated artifacts out of GitHub:
  - `data/`, `html/`, and `html_old/` are git-ignored.
- The project Python dependencies are expected in the `obd` conda environment.

## Environment and dependencies

- Core (non-GUI) dependencies:
  - `pip install -r requirements.txt`
  - or use `environment.yml` with conda
- GUI is optional and only needed for GUI-based workflows:
  - `pip install ".[gui]"`
- Standard packaging/dependency metadata is in `pyproject.toml`.
