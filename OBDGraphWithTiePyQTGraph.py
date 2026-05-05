"""
Deprecated shim — headless PNG/PDF/SVG export via ``OBDExplorerPlus.py export``.

Default output matches the former ``OUTPUT_PNG_PATH`` (``OBDGraphWithTiePyQTGraph.png``).

Use: ``python OBDExplorerPlus.py export -o OBDGraphWithTiePyQTGraph.png --format png``
(or ``--format pdf`` / ``--format svg``).
"""

from __future__ import annotations

import warnings

from obd_explorer.render_headless import HeadlessExportConfig, export_graph_headless


def main() -> None:
    warnings.warn(
        "OBDGraphWithTiePyQTGraph.py is deprecated. Use:\n"
        "  python OBDExplorerPlus.py export -o OBDGraphWithTiePyQTGraph.png --format png\n",
        DeprecationWarning,
        stacklevel=2,
    )
    cfg = HeadlessExportConfig(
        output_path="OBDGraphWithTiePyQTGraph.png",
        export_format="png",
    )
    export_graph_headless(cfg, verbose=True)


if __name__ == "__main__":
    main()
