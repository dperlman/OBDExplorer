"""BinomialGrid and construction from graph shards."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

from obd_explorer.numeric import expected_rank


def _point_dict_from_shard_row(
    y_row: np.ndarray, perm_row: np.ndarray, n: int
) -> dict:
    y = np.asarray(y_row, dtype=float)
    perm = np.asarray(perm_row, dtype=int)
    x = np.arange(n + 1, dtype=float)
    return {"x": x, "y": y, "perm": perm}


@dataclass(frozen=True)
class BinomialGrid:
    """n range, p grid, and shard ``rows_by_n`` (or materialized ``binomial_flat``)."""

    n_min: int
    n_max: int
    p_steps: int
    p_values: tuple[float, ...]
    binomial_flat: list | None
    rows_by_n: dict[int, dict[str, np.ndarray]] | None

    @property
    def p_half_start(self) -> int:
        return (self.p_steps - 1) // 2


def _binomial_index(n: int, p_idx: int, grid: BinomialGrid) -> int:
    return (n - grid.n_min) * grid.p_steps + p_idx


def build_vp_xy(
    grid: BinomialGrid,
    n: int,
    p_ix_lo: int,
    p_ix_hi: int,
    *,
    explorer_endpoint_y: bool = False,
    sorted_mode: bool = True,
    scaled_mode: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Viewport x=p and y=E curves over p index range."""
    pv = grid.p_values
    num_pts = p_ix_hi - p_ix_lo + 1
    x_arr = np.empty(num_pts, dtype=float)
    y_arr = np.empty(num_pts, dtype=float)

    if grid.rows_by_n is not None:
        rows = grid.rows_by_n[n]
        exp_sorted = np.asarray(rows["expected_sorted_by_p"], dtype=float)
        y_all = np.asarray(rows["y"])
        perm_all = np.asarray(rows["perm"])
        for k, p_ix in enumerate(range(p_ix_lo, p_ix_hi + 1)):
            p_val = float(pv[p_ix])
            if sorted_mode:
                e = float(exp_sorted[p_ix])
            else:
                pt = _point_dict_from_shard_row(y_all[p_ix], perm_all[p_ix], n)
                e = expected_rank(pt)
            if explorer_endpoint_y:
                y_arr[k] = e
            elif scaled_mode and not sorted_mode:
                y_arr[k] = p_val
            elif scaled_mode:
                y_arr[k] = e / n
            else:
                y_arr[k] = e
            x_arr[k] = p_val
        return x_arr, y_arr

    bd = grid.binomial_flat
    assert bd is not None
    for k, p_ix in enumerate(range(p_ix_lo, p_ix_hi + 1)):
        idx = _binomial_index(n, p_ix, grid)
        pt = bd[idx]
        p_val = float(pv[p_ix])
        e = expected_rank(pt) if sorted_mode else n * p_val
        if explorer_endpoint_y:
            y_arr[k] = e
        elif scaled_mode and not sorted_mode:
            y_arr[k] = p_val
        elif scaled_mode:
            y_arr[k] = e / n
        else:
            y_arr[k] = e
        x_arr[k] = p_val
    return x_arr, y_arr


def build_binomial_grid_from_shards(
    *,
    n_req_min: int,
    n_req_max: int,
    p_steps_req: int | None = None,
    manifest_path: str | None = None,
    shards_dir: str | None = None,
) -> BinomialGrid:
    """Load graph shards via OBDsaveSourceData.load_graph_data_from_shards."""
    from OBDsaveSourceData import DEFAULT_GRAPH_SHARDS_DIR, load_graph_data_from_shards

    n_list = list(range(n_req_min, n_req_max + 1))
    payload = load_graph_data_from_shards(
        manifest_path=manifest_path,
        shards_dir=shards_dir or DEFAULT_GRAPH_SHARDS_DIR,
        p_steps=p_steps_req,
        n_list=n_list,
        require_all=True,
    )
    p_steps = int(payload["p_steps"])
    if p_steps_req is not None and p_steps != int(p_steps_req):
        print(
            f"ERROR: shard manifest p_steps={p_steps} != requested {p_steps_req}.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    pv_arr = np.asarray(payload["p_values"], dtype=np.float32)
    p_values = tuple(float(x) for x in pv_arr.tolist())
    rows_by_n: dict[int, dict[str, np.ndarray]] = {}
    rb = payload["rows_by_n"]
    for n in n_list:
        if n not in rb:
            print(f"ERROR: missing graph shard data for n={n}.\n", file=sys.stderr)
            sys.exit(1)
        rows_by_n[n] = rb[n]
    return BinomialGrid(
        n_min=n_req_min,
        n_max=n_req_max,
        p_steps=p_steps,
        p_values=p_values,
        binomial_flat=None,
        rows_by_n=rows_by_n,
    )


def resolve_binomial_grid(
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
    graph_manifest_path: str | None = None,
    graph_shards_dir: str | None = None,
) -> BinomialGrid:
    """Load the binomial grid from graph shard manifests (see ``OBDsaveSourceData``)."""
    from OBDsaveSourceData import DEFAULT_GRAPH_SHARDS_DIR, _resolve_graph_manifest_path

    resolved = _resolve_graph_manifest_path(
        graph_manifest_path, p_steps, graph_shards_dir or DEFAULT_GRAPH_SHARDS_DIR
    )
    if not os.path.isfile(resolved):
        print(
            f"ERROR: Graph shard manifest not found for p_steps={p_steps}:\n"
            f"  {os.path.abspath(resolved)}\n"
            "  There is no fallback to a different p_grid. Build shards for this p_steps\n"
            "  (e.g. OBDsaveSourceData --save-graph-data) or pass --graph-manifest / --graph-shards-dir.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    return build_binomial_grid_from_shards(
        n_req_min=n_min,
        n_req_max=n_max,
        p_steps_req=p_steps,
        manifest_path=graph_manifest_path,
        shards_dir=graph_shards_dir,
    )
