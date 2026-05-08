"""Generate HTML explorer variant 6 (tie scalar vs tie index; tie slope shards only)."""

from __future__ import annotations

import os
import sys

from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, iter_tie_points_from_shards

from obd_explorer.explorer6_html import build_explorer6_html
from obd_explorer.html_data import tie_explorer5_series_by_n_stream


def write_explorer6_html(
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
        n_rows = iter_tie_points_from_shards(
            path=man,
            n_list=n_vals,
            require_all=False,
            progress=(10 if progress else None),
            include_float_by_n=False,
            include_float_with_pairs_by_n=True,
            include_tie_slope_by_n=True,
        )
        tie_data = tie_explorer5_series_by_n_stream(n_rows, n_min, n_max, progress=progress)
    else:
        tie_data = {}
    if not tie_data:
        if verbose:
            print(
                f"WARNING: No tie shard manifest at {man!r}; plot will be empty.\n",
                file=sys.stderr,
            )
    html = build_explorer6_html(tie_data, n_min=n_min, n_max=n_max, colorscale=colorscale)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Wrote {output_path}.")
