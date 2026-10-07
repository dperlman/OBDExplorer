"""Generate HTML explorer variant 5 (tie scalar vs n) from the tie tables."""

from __future__ import annotations

import os
import sys

from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, iter_tie_tables

from obd_explorer.explorer5_html import build_explorer5_html
from obd_explorer.html_data import TIE_EXPLORER5_COLUMNS, tie_explorer5_series_by_n_stream


def write_explorer5_html(
    output_path: str,
    *,
    n_min: int,
    n_max: int,
    tie_manifest: str | None = None,
    colorscale: str = "viridis",
    verbose: bool = True,
    progress: bool = False,
) -> None:
    n_vals = list(range(n_min, n_max + 1))
    man = tie_manifest or DEFAULT_TIE_OUTPUT
    if os.path.isfile(man):
        n_rows = iter_tie_tables(
            man,
            n_list=n_vals,
            columns=TIE_EXPLORER5_COLUMNS,
            require_all=False,
            progress=(10 if progress else None),
        )
        tie_data = tie_explorer5_series_by_n_stream(n_rows, n_min, n_max, progress=progress)
    else:
        tie_data = {}
    if not tie_data:
        if verbose:
            print(
                f"WARNING: No tie manifest at {man!r}; plot will be empty.\n",
                file=sys.stderr,
            )
    html = build_explorer5_html(tie_data, n_min=n_min, n_max=n_max, colorscale=colorscale)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Wrote {output_path}.")
