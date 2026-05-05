from __future__ import annotations

import csv
import math
import os
import time

from OBDsaveSourceData import (
    DEFAULT_TIE_SLOPE_A,
    DEFAULT_TIE_SLOPE_EPS_MIN,
    DEFAULT_TIE_SLOPE_FLAT_TOL,
    all_tie_points_float_with_pairs,
    _tie_slope_records_for_n,
)

# Sentinel n values to test. Edit as needed.
DEFAULT_N_VALUES = [798, 809, 903, 1000]

# Sweep of candidate flat tolerances. First value is baseline for diffs.
DEFAULT_FLAT_TOLS = [DEFAULT_TIE_SLOPE_FLAT_TOL, 1e-12, 1e-10, 1e-9, 1e-8]

# Slope settings aligned with main pipeline defaults unless edited here.
DEFAULT_SLOPE_A = DEFAULT_TIE_SLOPE_A
DEFAULT_SLOPE_EPS_CAP = 1e-3
DEFAULT_SLOPE_EPS_MIN = DEFAULT_TIE_SLOPE_EPS_MIN

# Optional CSV summary output (None disables file write).
DEFAULT_OUTPUT_CSV = os.path.join("tmp", "flat_tol_sensitivity_summary.csv")


def _max_local_min_p(records: list[dict]) -> float | None:
    mins = [float(r["p"]) for r in records if str(r.get("extremum_type", "")) == "minimum"]
    return max(mins) if mins else None


def _summary(records: list[dict], stats: dict[str, int | float]) -> dict[str, int | float | None]:
    return {
        "points": len(records),
        "deadzone_used": int(sum(int(bool(r.get("deadzone_used", False))) for r in records)),
        "high_precision_used": int(sum(int(bool(r.get("high_precision_used", False))) for r in records)),
        "ambiguous": int(sum(int(bool(r.get("extremum_ambiguous", False))) for r in records)),
        "minimum": int(sum(int(str(r.get("extremum_type", "")) == "minimum") for r in records)),
        "maximum": int(sum(int(str(r.get("extremum_type", "")) == "maximum") for r in records)),
        "neither": int(sum(int(str(r.get("extremum_type", "")) == "neither") for r in records)),
        "max_local_min_p": _max_local_min_p(records),
        "hp_3point_attempted": int(stats.get("hp_3point_attempted", 0)),
        "hp_3point_resolved": int(stats.get("hp_3point_resolved", 0)),
        "hp_3point_failed": int(stats.get("hp_3point_failed", 0)),
    }


def _compare_records(
    baseline_records: list[dict],
    candidate_records: list[dict],
) -> dict[str, int | float]:
    n_baseline = len(baseline_records)
    n_candidate = len(candidate_records)
    n_common = min(n_baseline, n_candidate)

    out: dict[str, int | float] = {
        "matched_points": int(n_common),
        "baseline_points": int(n_baseline),
        "candidate_points": int(n_candidate),
        "length_mismatch": int(n_baseline != n_candidate),
        "changed_any": 0,
        "changed_extremum_type": 0,
        "changed_extremum_ambiguous": 0,
        "changed_dir_change": 0,
    }
    for idx in range(n_common):
        rb = baseline_records[idx]
        rc = candidate_records[idx]
        de = rb.get("extremum_type") != rc.get("extremum_type")
        da = bool(rb.get("extremum_ambiguous")) != bool(rc.get("extremum_ambiguous"))
        dd = bool(rb.get("dir_change")) != bool(rc.get("dir_change"))
        if de:
            out["changed_extremum_type"] += 1
        if da:
            out["changed_extremum_ambiguous"] += 1
        if dd:
            out["changed_dir_change"] += 1
        if de or da or dd:
            out["changed_any"] += 1
    return out


def main() -> None:
    n_values = list(DEFAULT_N_VALUES)
    tols = list(DEFAULT_FLAT_TOLS)
    if not n_values:
        raise ValueError("DEFAULT_N_VALUES is empty.")
    if len(tols) < 2:
        raise ValueError("DEFAULT_FLAT_TOLS needs at least two values.")

    baseline_tol = float(tols[0])
    print(
        f"flat_tol sensitivity sweep: baseline={baseline_tol:.1e}, "
        f"candidates={[f'{float(t):.1e}' for t in tols[1:]]}"
    )
    print(f"N values: {n_values}")

    rows: list[dict[str, int | float | str | None]] = []
    t_start = time.perf_counter()

    for n in n_values:
        print(f"\n=== n={n} ===")
        t_n0 = time.perf_counter()
        recs = all_tie_points_float_with_pairs(int(n))
        by_tol: dict[float, tuple[list[dict], dict[str, int | float], dict[str, int | float | None]]] = {}

        for tol in tols:
            slope_records, _hit_floor, _n_deadzone_used, _n_deadzone_ambiguous, slope_stats, _wr = (
                _tie_slope_records_for_n(
                    int(n),
                    recs,
                    slope_a=float(DEFAULT_SLOPE_A),
                    slope_eps_cap=float(DEFAULT_SLOPE_EPS_CAP),
                    slope_flat_tol=float(tol),
                    slope_eps_min=float(DEFAULT_SLOPE_EPS_MIN),
                )
            )
            s = _summary(slope_records, slope_stats)
            by_tol[float(tol)] = (slope_records, slope_stats, s)

            print(
                f"tol={float(tol):.1e} points={s['points']} deadzone={s['deadzone_used']} "
                f"hp={s['high_precision_used']} ambig={s['ambiguous']} "
                f"min={s['minimum']} max={s['maximum']} max_local_min_p={s['max_local_min_p']}"
            )

        b_records, _b_stats, b_summary = by_tol[baseline_tol]
        for tol in tols[1:]:
            c_records, _c_stats, c_summary = by_tol[float(tol)]
            cmp = _compare_records(b_records, c_records)
            max_diff = 0
            if b_summary["max_local_min_p"] != c_summary["max_local_min_p"]:
                max_diff = 1

            print(
                f"  vs baseline tol={float(tol):.1e}: "
                f"delta_deadzone={int(c_summary['deadzone_used']) - int(b_summary['deadzone_used'])} "
                f"delta_hp={int(c_summary['high_precision_used']) - int(b_summary['high_precision_used'])} "
                f"changed_any={cmp['changed_any']} changed_type={cmp['changed_extremum_type']} "
                f"changed_ambig={cmp['changed_extremum_ambiguous']} changed_dir={cmp['changed_dir_change']} "
                f"max_local_min_changed={max_diff}"
            )

            row: dict[str, int | float | str | None] = {
                "n": int(n),
                "baseline_tol": baseline_tol,
                "candidate_tol": float(tol),
                "baseline_deadzone_used": int(b_summary["deadzone_used"]),
                "candidate_deadzone_used": int(c_summary["deadzone_used"]),
                "delta_deadzone_used": int(c_summary["deadzone_used"]) - int(b_summary["deadzone_used"]),
                "baseline_high_precision_used": int(b_summary["high_precision_used"]),
                "candidate_high_precision_used": int(c_summary["high_precision_used"]),
                "delta_high_precision_used": int(c_summary["high_precision_used"])
                - int(b_summary["high_precision_used"]),
                "changed_any": int(cmp["changed_any"]),
                "changed_extremum_type": int(cmp["changed_extremum_type"]),
                "changed_extremum_ambiguous": int(cmp["changed_extremum_ambiguous"]),
                "changed_dir_change": int(cmp["changed_dir_change"]),
                "matched_points": int(cmp["matched_points"]),
                "baseline_max_local_min_p": b_summary["max_local_min_p"],
                "candidate_max_local_min_p": c_summary["max_local_min_p"],
                "max_local_min_changed": int(max_diff),
            }
            rows.append(row)

        print(f"n={n} completed in {time.perf_counter() - t_n0:.2f}s")

    t_total = time.perf_counter() - t_start
    print(f"\nSweep completed in {t_total:.2f}s")

    if DEFAULT_OUTPUT_CSV:
        out_path = str(DEFAULT_OUTPUT_CSV)
        out_parent = os.path.dirname(os.path.abspath(out_path))
        if out_parent:
            os.makedirs(out_parent, exist_ok=True)
        fieldnames = [
            "n",
            "baseline_tol",
            "candidate_tol",
            "baseline_deadzone_used",
            "candidate_deadzone_used",
            "delta_deadzone_used",
            "baseline_high_precision_used",
            "candidate_high_precision_used",
            "delta_high_precision_used",
            "changed_any",
            "changed_extremum_type",
            "changed_extremum_ambiguous",
            "changed_dir_change",
            "matched_points",
            "baseline_max_local_min_p",
            "candidate_max_local_min_p",
            "max_local_min_changed",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV summary: {out_path}")


if __name__ == "__main__":
    main()
