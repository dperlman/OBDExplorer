"""
Deprecated shim — HTML explorer #2 (PCA) is generated via ``OBDExplorerPlus.py html --variant 2``.

Use: ``python OBDExplorerPlus.py html --variant 2 -o OBDexplorer2.html --n-min 2 --n-max 50``
"""

from __future__ import annotations

import warnings

from obd_explorer.explorer2_export import write_explorer2_html


def main(
    output_path: str = "OBDexplorer2.html",
    graph_manifest: str | None = None,
    tie_manifest: str | None = None,
) -> None:
    warnings.warn(
        "OBDexplorer2.py is deprecated. Use:\n"
        "  python OBDExplorerPlus.py html --variant 2 -o OBDexplorer2.html --n-min 2 --n-max 50\n",
        DeprecationWarning,
        stacklevel=2,
    )
    write_explorer2_html(
        output_path,
        n_min=2,
        n_max=50,
        p_steps=1001,
        graph_manifest=graph_manifest,
        tie_manifest=tie_manifest,
        verbose=True,
    )


if __name__ == "__main__":
    main()
