"""Generate HTML explorer variant 5 (tie scalar vs n) from tie shards."""

from __future__ import annotations

import sys

from OBDsaveSourceData import DEFAULT_TIE_OUTPUT

from obd_explorer.explorer5_html import build_explorer5_html
from obd_explorer.explorer1_export import _load_tie_payload
from obd_explorer.html_data import tie_explorer5_series_by_n


def write_explorer5_html(
    output_path: str,
    *,
    n_min: int,
    n_max: int,
    tie_manifest: str | None = None,
    verbose: bool = True,
    progress: bool = False,
) -> None:
    n_vals = list(range(n_min, n_max + 1))
    tie_payload = _load_tie_payload(
        tie_manifest,
        n_vals,
        progress=(10 if progress else None),
    )
    if not tie_payload.get("float_with_pairs_by_n") and not tie_payload.get("float_by_n"):
        if verbose:
            man = tie_manifest or DEFAULT_TIE_OUTPUT
            print(
                f"WARNING: No tie shard manifest at {man!r}; plot will be empty.\n",
                file=sys.stderr,
            )
    tie_data = tie_explorer5_series_by_n(tie_payload, n_min, n_max, progress=progress)
    html = build_explorer5_html(tie_data, n_min=n_min, n_max=n_max)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    if verbose:
        print(f"Wrote {output_path}.")
