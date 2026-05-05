"""HTML export for explorer variants 2–4 using shard graph data + tie payload (PCA-only for variant 4)."""

from __future__ import annotations

import sys
import time

from OBDsaveSourceData import DEFAULT_TIE_OUTPUT

from obd_explorer.explorer1_export import _load_tie_payload
from obd_explorer.explorer2_dual_html import build_explorer2_dual_panel_html_document
from obd_explorer.explorer2_html import build_explorer2_html_document
from obd_explorer.explorer2_pca import precompute_pca, precompute_tie_pca
from obd_explorer.explorer4_pca_only_html import build_explorer4_pca_only_html_document
from obd_explorer.explorer2_tie import last_tie_by_n_from_payload
from obd_explorer.grid import resolve_binomial_grid
from obd_explorer.html_data import materialize_binomial_series_for_js


def _html_payload_for_explorer2_3(
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
    graph_manifest: str | None,
    graph_shards_dir: str | None,
    tie_manifest: str | None,
    verbose: bool,
    progress: bool = False,
) -> tuple:
    """Shared shard load + PCA precompute for dual (2), quad (3), and PCA-only (4) HTML."""
    n_vals = list(range(n_min, n_max + 1))
    p_default_idx = p_steps // 2

    grid = resolve_binomial_grid(
        n_min=n_min,
        n_max=n_max,
        p_steps=p_steps,
        graph_manifest_path=graph_manifest,
        graph_shards_dir=graph_shards_dir,
    )
    binomial_data = materialize_binomial_series_for_js(grid, progress=progress)
    p_grid = [float(x) for x in grid.p_values]

    tie_payload = _load_tie_payload(
        tie_manifest,
        n_vals,
        progress=(10 if progress else None),
    )
    if verbose and not tie_payload.get("float_with_pairs_by_n") and not tie_payload.get(
        "float_by_n"
    ):
        man = tie_manifest or DEFAULT_TIE_OUTPUT
        print(
            f"WARNING: No tie shard manifest at {man!r}; "
            "using analytic last-tie upper bounds where needed.\n",
            file=sys.stderr,
        )

    last_tie_by_n = last_tie_by_n_from_payload(tie_payload, n_vals)

    if verbose:
        print(f"Precomputing PCA (n={n_min}..{n_max}, p_steps={p_steps})...")
    t_pca = time.perf_counter()
    if progress:
        print(
            "[html] PCA: starting 6 passes over n (no per-n progress in PCA passes)",
            file=sys.stderr,
        )
    data_full_unsorted, _, _ = precompute_pca(n_vals, p_steps, 0.0, 1.0, False)
    data_full_sorted, _, _ = precompute_pca(n_vals, p_steps, 0.0, 1.0, True)
    data_half_unsorted, _, _ = precompute_pca(n_vals, p_steps, 0.5, 1.0, False)
    data_half_sorted, _, _ = precompute_pca(n_vals, p_steps, 0.5, 1.0, True)
    data_tie_unsorted, _, _ = precompute_tie_pca(n_vals, p_steps, False, last_tie_by_n)
    data_tie_sorted, _, _ = precompute_tie_pca(n_vals, p_steps, True, last_tie_by_n)
    if progress:
        print(
            f"[html] PCA: 6 passes finished in {time.perf_counter() - t_pca:.2f}s",
            file=sys.stderr,
        )

    return (
        binomial_data,
        p_grid,
        data_full_unsorted,
        data_full_sorted,
        data_half_unsorted,
        data_half_sorted,
        data_tie_unsorted,
        data_tie_sorted,
        last_tie_by_n,
        p_default_idx,
    )


def write_explorer2_html(
    output_path: str,
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
    graph_manifest: str | None = None,
    graph_shards_dir: str | None = None,
    tie_manifest: str | None = None,
    verbose: bool = True,
    progress: bool = False,
) -> None:
    """Two-panel HTML: binomial + PCA, shared n slider (``html_old/OBDexplorer2`` layout)."""
    (
        binomial_data,
        p_grid,
        data_full_unsorted,
        data_full_sorted,
        data_half_unsorted,
        data_half_sorted,
        data_tie_unsorted,
        data_tie_sorted,
        last_tie_by_n,
        p_default_idx,
    ) = _html_payload_for_explorer2_3(
        n_min=n_min,
        n_max=n_max,
        p_steps=p_steps,
        graph_manifest=graph_manifest,
        graph_shards_dir=graph_shards_dir,
        tie_manifest=tie_manifest,
        verbose=verbose,
        progress=progress,
    )

    html = build_explorer2_dual_panel_html_document(
        binomial_data,
        p_grid,
        data_full_unsorted,
        data_full_sorted,
        data_half_unsorted,
        data_half_sorted,
        data_tie_unsorted,
        data_tie_sorted,
        last_tie_by_n,
        n_min=n_min,
        n_max=n_max,
        p_steps=p_steps,
        p_default_idx=p_default_idx,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Wrote {output_path}.")


def write_explorer3_quad_html(
    output_path: str,
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
    graph_manifest: str | None = None,
    graph_shards_dir: str | None = None,
    tie_manifest: str | None = None,
    verbose: bool = True,
    progress: bool = False,
) -> None:
    """Quad-layout HTML: binomial, PCA, E[X]/E[rank] vs p, controls (explorer3-style)."""
    (
        binomial_data,
        p_grid,
        data_full_unsorted,
        data_full_sorted,
        data_half_unsorted,
        data_half_sorted,
        data_tie_unsorted,
        data_tie_sorted,
        last_tie_by_n,
        p_default_idx,
    ) = _html_payload_for_explorer2_3(
        n_min=n_min,
        n_max=n_max,
        p_steps=p_steps,
        graph_manifest=graph_manifest,
        graph_shards_dir=graph_shards_dir,
        tie_manifest=tie_manifest,
        verbose=verbose,
        progress=progress,
    )

    html = build_explorer2_html_document(
        binomial_data,
        p_grid,
        data_full_unsorted,
        data_full_sorted,
        data_half_unsorted,
        data_half_sorted,
        data_tie_unsorted,
        data_tie_sorted,
        last_tie_by_n,
        n_min=n_min,
        n_max=n_max,
        p_steps=p_steps,
        p_default_idx=p_default_idx,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Wrote {output_path}.")


def write_explorer4_pca_only_html(
    output_path: str,
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
    graph_manifest: str | None = None,
    graph_shards_dir: str | None = None,
    tie_manifest: str | None = None,
    verbose: bool = True,
    progress: bool = False,
) -> None:
    """PCA-only HTML (no binomial panel, no p slider / Show p)."""
    (
        _binomial_data,
        _p_grid,
        data_full_unsorted,
        data_full_sorted,
        data_half_unsorted,
        data_half_sorted,
        data_tie_unsorted,
        data_tie_sorted,
        last_tie_by_n,
        _p_default_idx,
    ) = _html_payload_for_explorer2_3(
        n_min=n_min,
        n_max=n_max,
        p_steps=p_steps,
        graph_manifest=graph_manifest,
        graph_shards_dir=graph_shards_dir,
        tie_manifest=tie_manifest,
        verbose=verbose,
        progress=progress,
    )

    html = build_explorer4_pca_only_html_document(
        data_full_unsorted,
        data_full_sorted,
        data_half_unsorted,
        data_half_sorted,
        data_tie_unsorted,
        data_tie_sorted,
        last_tie_by_n,
        n_min=n_min,
        n_max=n_max,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Wrote {output_path}.")
