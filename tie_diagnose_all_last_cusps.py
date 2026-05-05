#!/usr/bin/env python3
"""Run tie_slope_diagnostic once per N for the last cusp tie (per cusp sidecar file).

For each N, ``last`` cusp means maximum ``tie_index`` among cusp records; tie-break by
maximum ``p_float``. Invokes ``tie_slope_diagnostic.py`` with ``--point-filter manual`` and
``--skip-summary-page``, writing PDFs under ``last_cusp_plots/`` by default.

After all per-N PDFs are written, merges them into one multi-page PDF (see ``--combined-name``).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from OBDsaveSourceData import DEFAULT_CUSP_OUTPUT, load_cusp_data

DEFAULT_COMBINED_NAME = "tie_slope_last_cusp_all.pdf"


def _merge_pdfs(paths: list[Path], output: Path) -> None:
    """Concatenate PDFs in order. Uses ``pypdf`` if installed; otherwise Ghostscript ``gs``."""
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        from pypdf import PdfReader, PdfWriter

        writer = PdfWriter()
        for p in existing:
            reader = PdfReader(str(p))
            for page in reader.pages:
                writer.add_page(page)
        with open(output, "wb") as f:
            writer.write(f)
        return
    except ImportError:
        pass

    gs = shutil.which("gs")
    if gs:
        cmd = [
            gs,
            "-dBATCH",
            "-dNOPAUSE",
            "-q",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={output}",
            *[str(p) for p in existing],
        ]
        subprocess.run(cmd, check=True)
        return

    raise RuntimeError(
        "Cannot merge PDFs: install pypdf (pip install pypdf) or Ghostscript (gs) on PATH."
    )


def _last_cusp_tie_number(records: list[dict]) -> tuple[int, float] | None:
    """Return (tie_index, p_float) for the last cusp record, or None if empty."""
    if not records:
        return None
    best_ti: int | None = None
    best_p = float("-inf")
    for r in records:
        try:
            ti = int(r["tie_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if "p_float" in r:
            p = float(r["p_float"])
        else:
            p = float(r.get("p", 0.0))
        if best_ti is None or ti > best_ti or (ti == best_ti and p > best_p):
            best_ti = ti
            best_p = p
    if best_ti is None:
        return None
    return int(best_ti), float(best_p)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    diag_script = repo_root / "tie_slope_diagnostic.py"

    parser = argparse.ArgumentParser(
        description=(
            "For each N in the cusp pickle, find the last cusp tie_index and run "
            "tie_slope_diagnostic.py (manual filter only, no summary PDF page)."
        )
    )
    parser.add_argument(
        "--cusp",
        type=str,
        default=DEFAULT_CUSP_OUTPUT,
        metavar="PATH",
        help=f"Cusp sidecar pickle (default: {DEFAULT_CUSP_OUTPUT}).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="last_cusp_plots",
        metavar="DIR",
        help="Output directory for PDFs (default: last_cusp_plots).",
    )
    parser.add_argument("--n-min", type=int, default=None, metavar="N", help="Only process n >= N.")
    parser.add_argument("--n-max", type=int, default=None, metavar="N", help="Only process n <= N.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running tie_slope_diagnostic.",
    )
    parser.add_argument(
        "--combined-name",
        type=str,
        default=DEFAULT_COMBINED_NAME,
        metavar="FILENAME",
        help=(
            "Multi-page PDF written under --out-dir merging all per-N PDFs "
            f"(default: {DEFAULT_COMBINED_NAME})."
        ),
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not build the merged multi-page PDF at the end.",
    )
    args = parser.parse_args()

    if not diag_script.is_file():
        raise FileNotFoundError(f"Missing {diag_script}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = load_cusp_data(path=args.cusp, n_list=None, require_all=False)
    n_entries = payload.get("n_entries") or {}
    if not n_entries:
        raise ValueError(f"No n_entries in {args.cusp!r}")

    n_min = args.n_min
    n_max = args.n_max

    planned: list[tuple[int, int, float, Path]] = []
    for n_key in sorted(n_entries.keys(), key=lambda k: int(k)):
        ni = int(n_key)
        if n_min is not None and ni < int(n_min):
            continue
        if n_max is not None and ni > int(n_max):
            continue
        block = n_entries[n_key]
        records = block.get("records")
        if not isinstance(records, list) or not records:
            continue
        picked = _last_cusp_tie_number(records)
        if picked is None:
            continue
        tie_num, p_last = picked
        out_pdf = out_dir / f"tie_slope_last_cusp_n{ni}.pdf"
        planned.append((ni, tie_num, p_last, out_pdf))

    if not planned:
        print("No cusp records to process (check n range and cusp file).", file=sys.stderr)
        sys.exit(1)

    for ni, tie_num, p_last, out_pdf in planned:
        cmd = [
            sys.executable,
            str(diag_script),
            "--n",
            str(ni),
            "--output",
            str(out_pdf),
            "--point-filter",
            "manual",
            "--include-tie-numbers",
            str(tie_num),
            "--skip-summary-page",
            "--no-reference-if-empty",
        ]
        print(f"n={ni} last_cusp tie_index={tie_num} p_float={p_last:.17g} -> {out_pdf}")
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(repo_root), check=True)

    combined_path = out_dir / str(args.combined_name)
    if not args.no_combined:
        pdf_paths = [p for _, _, _, p in planned]
        if args.dry_run:
            print(f"Would merge {len(pdf_paths)} PDF(s) -> {combined_path}")
        else:
            _merge_pdfs(pdf_paths, combined_path)
            print(f"Wrote combined PDF: {combined_path}")


if __name__ == "__main__":
    main()
