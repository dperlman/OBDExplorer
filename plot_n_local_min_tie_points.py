from __future__ import annotations

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from OBDsaveSourceData import DEFAULT_TIE_OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 2-page PDF: page 1 = local minima count by N, "
            "page 2 = residuals of that count after a linear least-squares fit in N."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_TIE_OUTPUT,
        help=(
            "Tie-point manifest pickle (sharded layout; default: same as "
            f"OBDsaveSourceData.DEFAULT_TIE_OUTPUT = {DEFAULT_TIE_OUTPUT!r})."
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "n_local_min_tie_points.pdf"),
        help="Output PDF path (default: data/n_local_min_tie_points.pdf).",
    )
    parser.add_argument(
        "--title-prefix",
        default="Local Minima Diagnostics",
        help="Title prefix used on both pages.",
    )
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        manifest = pickle.load(f)

    if not isinstance(manifest, dict):
        raise ValueError(f"Expected dict in {args.input!r}, got {type(manifest)!r}.")
    if manifest.get("format") != "obd.tie_points_slope.shards.v1":
        raise ValueError(
            f"Unsupported tie manifest format in {args.input!r}: {manifest.get('format')!r}. "
            "Expected 'obd.tie_points_slope.shards.v1'."
        )

    n_entries = manifest.get("n_entries") or {}
    if not isinstance(n_entries, dict) or not n_entries:
        raise ValueError(f"No n_entries in {args.input!r}. Run OBDsaveSourceData.py --save-tie-points first.")

    n_vals: list[int] = []
    local_min_counts: list[int] = []

    for n_key in sorted(n_entries.keys(), key=lambda k: int(k)):
        entry = n_entries[n_key]
        if not isinstance(entry, dict):
            continue
        ni = int(entry.get("n", n_key))
        total_tie_points = entry.get("tie_points")
        local_min_count = entry.get("local_min")
        if total_tie_points is None or local_min_count is None:
            continue
        total_tie_points = int(total_tie_points)
        local_min_count = int(local_min_count)
        if total_tie_points <= 0:
            continue

        n_vals.append(ni)
        local_min_counts.append(local_min_count)

    if not n_vals:
        raise ValueError(
            f"No usable manifest rows with tie_points and local_min in {args.input!r}. "
            "Regenerate tie points with a current OBDsaveSourceData.py save."
        )

    if len(n_vals) < 2:
        raise ValueError("Need at least two N values in the manifest to form a linear fit and residuals.")

    x = np.asarray(n_vals, dtype=float)
    y = np.asarray(local_min_counts, dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = np.polyval((slope, intercept), x)
    residuals = y - fitted

    out_parent = os.path.dirname(os.path.abspath(args.output))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    with PdfPages(args.output) as pdf:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(
            n_vals,
            local_min_counts,
            color="tab:blue",
            linewidth=1.6,
            marker="o",
            markersize=2.8,
        )
        ax1.set_xlabel("N")
        ax1.set_ylabel("n_local_min")
        ax1.set_title(f"{args.title_prefix}: # local minima by N")
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6)
        ax2.plot(
            n_vals,
            residuals,
            color="tab:green",
            linewidth=1.6,
            marker="o",
            markersize=2.8,
        )
        ax2.set_xlabel("N")
        ax2.set_ylabel("residual (n_local_min − linear fit)")
        ax2.set_title(
            f"{args.title_prefix}: residuals vs N  "
            f"(fit: n_local_min ≈ {float(slope):.6g}·N + {float(intercept):.6g})"
        )
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        pdf.savefig(fig2)
        plt.close(fig2)

    print(f"Wrote {args.output}")
    print(f"n_count={len(n_vals)} n_min={min(n_vals)} n_max={max(n_vals)}")
    print(f"linear_fit slope={float(slope):.12g} intercept={float(intercept):.12g}")
    print(
        "residual_min="
        f"{float(np.min(residuals)):.6f} "
        "residual_max="
        f"{float(np.max(residuals)):.6f}"
    )


if __name__ == "__main__":
    main()
