"""
Deprecated shim — interactive Qt explorer lives in ``obd_explorer_qt_ui`` / ``OBDExplorerPlus.py``.

Use: ``python OBDExplorerPlus.py gui``
"""

from __future__ import annotations

import warnings


def main() -> None:
    warnings.warn(
        "OBDGraphExplorerQT.py is deprecated. Use:\n"
        "  python OBDExplorerPlus.py gui\n",
        DeprecationWarning,
        stacklevel=2,
    )
    import obd_explorer_qt_ui as ui

    ui.main()


if __name__ == "__main__":
    main()
