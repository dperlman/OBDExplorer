"""Generate OBDgraphExplorer1-style HTML from shard data."""

from __future__ import annotations

import os
import sys

from OBDsaveSourceData import DEFAULT_TIE_OUTPUT, load_tie_points_from_shards

from obd_explorer.explorer1_html import build_explorer1_html
from obd_explorer.grid import resolve_binomial_grid
from obd_explorer.html_data import (
    materialize_binomial_series_for_js,
    tie_points_by_n_for_explorer1,
)


def _load_tie_payload(
    tie_manifest: str | None,
    n_vals: list[int],
    *,
    progress: int | None = None,
) -> dict:
    man = tie_manifest or DEFAULT_TIE_OUTPUT
    if os.path.isfile(man):
        return load_tie_points_from_shards(
            man,
            n_list=n_vals,
            require_all=False,
            progress=progress,
        )
    return {"float_with_pairs_by_n": {}, "float_by_n": {}}


def write_explorer1_html(
    output_path: str,
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
    graph_manifest: str | None = None,
    graph_shards_dir: str | None = None,
    tie_manifest: str | None = None,
    include_tie_points: bool = True,
    colorscale: str = "viridis",
    verbose: bool = True,
    progress: bool = False,
) -> None:
    grid = resolve_binomial_grid(
        n_min=n_min,
        n_max=n_max,
        p_steps=p_steps,
        graph_manifest_path=graph_manifest,
        graph_shards_dir=graph_shards_dir,
    )
    binomial_data = materialize_binomial_series_for_js(grid, progress=progress)
    n_vals = list(range(n_min, n_max + 1))
    if include_tie_points:
        tie_payload = _load_tie_payload(
            tie_manifest,
            n_vals,
            progress=(10 if progress else None),
        )
        if not tie_payload.get("float_with_pairs_by_n") and not tie_payload.get("float_by_n"):
            if verbose:
                man = tie_manifest or DEFAULT_TIE_OUTPUT
                print(
                    f"WARNING: No tie shard manifest at {man!r}; "
                    "swap-point hairlines may be empty.\n",
                    file=sys.stderr,
                )
        tie_points_by_n = tie_points_by_n_for_explorer1(tie_payload, n_min, n_max, progress=progress)
    else:
        tie_points_by_n = {}
    html = build_explorer1_html(
        binomial_data,
        tie_points_by_n,
        n_min=n_min,
        n_max=n_max,
        p_steps=p_steps,
        p_values=[float(x) for x in grid.p_values],
        include_tie_points=include_tie_points,
        colorscale=colorscale,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Wrote {output_path}.")
