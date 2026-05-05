"""
Compare per-(i,j) tie probabilities: float64 comb formula vs high-precision
reference (mpmath, same formula; evaluated in high precision).

Every ``SPOT_CHECK_INTERVAL`` valid pairs (configurable), also evaluate SymPy's
closed form and report |sympy - mpmath| and |sympy - float| as a sanity check
that mpmath matches symbolic arithmetic.

Imports ``all_tie_points`` / ``all_tie_points_exact`` for the final array-level summary line.
"""

from __future__ import annotations

import argparse

import numpy as np
from mpmath import mp
from scipy.special import comb
from sympy import S, binomial

from OBDsaveSourceData import all_tie_points, all_tie_points_exact


# Default n for tie enumeration (override with ``python ... -N 200``).
DEFAULT_N = 100

# Decimal places for mpmath reference (same rational formula as float path).
MP_DPS = 80

# SymPy evalf precision (informally aligned with MP_DPS for spot comparisons).
SYM_EVALF_DIGITS = 80

# Run a SymPy spot check every this many valid (i,j) pairs (0 = disable spots).
SPOT_CHECK_INTERVAL = 10

# Alert if |sympy - mpmath| exceeds this (both should agree well beyond float64).
SPOT_SYM_MPMATH_MAX_WARN = 1e-70


def _p_float(n: int, i: int, j: int) -> float | None:
    ratio = comb(n, j, exact=False) / comb(n, i, exact=False)
    if ratio <= 0 or not np.isfinite(ratio):
        return None
    exp = 1.0 / (j - i)
    p = 1.0 / (1.0 + ratio**exp)
    if 0 < p < 1 and np.isfinite(p):
        return float(p)
    return None


def _p_high_precision(n: int, i: int, j: int) -> float | None:
    """Same tie formula as float path, evaluated in mpmath (integer binomials)."""
    old = mp.dps
    try:
        mp.dps = MP_DPS
        ratio = mp.binomial(n, j) / mp.binomial(n, i)
        if ratio <= 0:
            return None
        exp = mp.mpf(1) / (j - i)
        p = mp.mpf(1) / (1 + ratio**exp)
        pf = float(p)
    finally:
        mp.dps = old
    if 0 < pf < 1 and np.isfinite(pf):
        return pf
    return None


def _p_sympy_float(n: int, i: int, j: int) -> float | None:
    """Symbolic tie formula from ``all_tie_points_exact``, reduced with evalf."""
    n_sym = S(n)
    ratio = binomial(n_sym, j) / binomial(n_sym, i)
    if ratio <= 0:
        return None
    exp = S(1) / (j - i)
    base = ratio**exp
    p = S(1) / (1 + base)
    try:
        if not (p > 0 and p < 1):
            return None
    except TypeError:
        pass
    try:
        val = float(p.evalf(SYM_EVALF_DIGITS))
    except (TypeError, ValueError):
        return None
    if 0 < val < 1 and np.isfinite(val):
        return val
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For each valid (i,j), print float vs mpmath tie p and |diff|. "
            "Optionally every K pairs, SymPy spot line: |sym-mpmath|, |sym-float|. "
            "Append ``all_tie_points_exact`` summary at end."
        )
    )
    parser.add_argument(
        "-N",
        "--binomial-n",
        type=int,
        default=DEFAULT_N,
        dest="binomial_n",
        metavar="N",
        help=f"Binomial n (default: {DEFAULT_N}). Use -N so ``conda run`` does not swallow ``--n``.",
    )
    parser.add_argument(
        "--spot-interval",
        type=int,
        default=SPOT_CHECK_INTERVAL,
        metavar="K",
        help=(
            "Run SymPy vs mpmath spot check every K valid pairs "
            f"(default: {SPOT_CHECK_INTERVAL}; use 0 to disable)."
        ),
    )
    args = parser.parse_args()
    n = int(args.binomial_n)
    if n < 1:
        raise SystemExit("n must be >= 1")

    spot_interval = int(args.spot_interval)
    total_pairs = 0
    global_max = 0.0
    spot_count = 0
    spot_max_sym_mpmath = 0.0
    spot_max_sym_float = 0.0

    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            pf = _p_float(n, i, j)
            pref = _p_high_precision(n, i, j)
            if pf is None or pref is None:
                continue
            d = abs(pf - pref)
            total_pairs += 1
            if d > global_max:
                global_max = d
            print(
                f"i={i} j={j}  p_float={pf:.17g}  p_ref={pref:.17g}  "
                f"abs_diff={d:.6e}"
            )

            do_spot = spot_interval > 0 and total_pairs % spot_interval == 0
            if do_spot:
                ps = _p_sympy_float(n, i, j)
                if ps is None:
                    print(
                        f"  SPOT i={i} j={j}  sympy_eval_failed (unexpected)"
                    )
                else:
                    d_sm = abs(ps - pref)
                    d_sf = abs(ps - pf)
                    spot_count += 1
                    if d_sm > spot_max_sym_mpmath:
                        spot_max_sym_mpmath = d_sm
                    if d_sf > spot_max_sym_float:
                        spot_max_sym_float = d_sf
                    flag = ""
                    if d_sm > SPOT_SYM_MPMATH_MAX_WARN:
                        flag = "  ** warn: |sympy-mpmath| large"
                    print(
                        f"  SPOT i={i} j={j}  |sympy-mpmath|={d_sm:.6e}  "
                        f"|sympy-float|={d_sf:.6e}{flag}"
                    )

    print(
        f"per-(i,j) summary: n={n} valid_pairs={total_pairs} "
        f"overall_max_abs_diff={global_max:.6e}"
    )
    if spot_interval > 0:
        print(
            f"sympy spots: interval={spot_interval} count={spot_count} "
            f"max|sympy-mpmath|={spot_max_sym_mpmath:.6e} "
            f"max|sympy-float|={spot_max_sym_float:.6e}"
        )

    a_float = all_tie_points(n)
    a_exact, _syms, _pg = all_tie_points_exact(n)
    print(f"arrays: len(float)={len(a_float)} len(exact)={len(a_exact)}")
    if len(a_float) == len(a_exact) and len(a_float) > 0:
        arr_diff = float(np.max(np.abs(a_float - a_exact)))
        print(f"arrays max_abs_diff(float vs exact enumeration)={arr_diff:.6e}")
    else:
        print(
            "arrays: length mismatch between float and exact tie enumeration (unexpected with shared skip logic)."
        )


if __name__ == "__main__":
    main()
