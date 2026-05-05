"""
Deprecated shim — HTML explorer #1 is generated via ``OBDExplorerPlus.py html --variant 1``.

Use: ``python OBDExplorerPlus.py html --variant 1 -o OBDgraphExplorer1.html``
"""

from __future__ import annotations

import warnings

from obd_explorer.explorer1_export import write_explorer1_html


def main(
    output_path: str = "OBDgraphExplorer1.html",
    graph_manifest: str | None = None,
    tie_manifest: str | None = None,
) -> None:
    warnings.warn(
        "OBDgraphExplorer1.py is deprecated. Use:\n"
        "  python OBDExplorerPlus.py html --variant 1 -o OBDgraphExplorer1.html\n",
        DeprecationWarning,
        stacklevel=2,
    )
    write_explorer1_html(
        output_path,
        n_min=2,
        n_max=100,
        p_steps=1001,
        graph_manifest=graph_manifest,
        tie_manifest=tie_manifest,
        verbose=True,
    )


if __name__ == "__main__":
    main()
