from __future__ import annotations

"""
Compare five ways to evaluate the same tie-point probability p(n, i, j) across a range of n.

Evaluates, for each N in [DEFAULT_N_MIN, DEFAULT_N_MAX], a fixed-size list of (i, j) pairs
(see ``_pairs_for_n``) and records:

  * **m1** ``m1_float_direct`` - float64 scipy ``comb``: C(n,j)/C(n,i) in the standard ratio form.
  * **m2** ``m2_float_simplified`` - float64 scipy with the algebraically equivalent ratio
    C(n-i,j-i)/C(j,i) (better conditioned for some (n,i,j)).
  * **m3** ``m3_mp_direct`` - same ratio as m1 in mpmath at ``DEFAULT_MP_DPS``.
  * **m4** ``m4_mp_simplified`` - same ratio as m2 in mpmath at ``DEFAULT_MP_DPS``.
  * **m5** ``m5_sympy20`` - SymPy baseline: symbolic ``binomial`` form with ``evalf``; tries an
    exact symbolic path first, then falls back to a numeric SymPy path if exact evaluation fails
    for a pair.

For every pair it prints (and gathers into a report) the five p values plus pairwise absolute
differences (float vs mp direct/simplified vs SymPy spreads, timing per N). The divergence figure
plots each of m1-m4 versus the SymPy reference (signed or absolute error via
``DEFAULT_DIVERGENCE_ERROR_MODE``), with optional connector lines keyed by pair order inside each N.

**How to run**

Requires the project ``obd`` conda env (numpy, scipy, sympy, mpmath, plotly). From the repo root:

    python diagnostic_tie_p_method_agreement.py

There is no argparse: behavior is controlled by module-level constants at the top (N range,
``DEFAULT_PAIR_COUNT``, ``DEFAULT_MP_DPS``, ``DEFAULT_SYMPY_DIGITS``, plot basename/filetype,
``PLOT_M1``-``PLOT_M4``, connector toggles, jitter, PDF size). Change those and re-run.

**Outputs**

* Progress and the full numerical report go to stdout.
* A Plotly divergence chart is written to ``{DEFAULT_DIVERGENCE_PLOT_BASENAME}.html`` or ``.pdf``
  depending on ``DEFAULT_DIVERGENCE_PLOT_FILETYPE``. Static PDF export needs Kaleido
  (``pip install kaleido``) if not already installed.

For a conda-based agent/shell invocation, prefix with ``conda run -n obd`` as in the repo cursor rule.
"""

from collections import deque
import math
import os
import time
from typing import Callable

from mpmath import mp
import plotly.graph_objects as go
from plotly.offline import plot as plotly_plot
from scipy.special import comb
from sympy import S, binomial

DEFAULT_N_MIN = 900
DEFAULT_N_MAX = 950
DEFAULT_MP_DPS = 80
DEFAULT_SYMPY_DIGITS = 40
DEFAULT_PAIR_COUNT = 100
# Anchor fraction used for pair seeds: (k*N), (0.5*N), ((1-k)*N).
DEFAULT_K_FRACTION = 0.1
DEFAULT_DIVERGENCE_PLOT_FILETYPE = "pdf"  # Allowed: "html" or "pdf"
DEFAULT_DIVERGENCE_PLOT_BASENAME = os.path.join("data", "diagnostic_tie_p_divergence_plot")
# Divergence mode for plot y-values: "absolute" => |method - sympy|, "signed" => (method - sympy).
DEFAULT_DIVERGENCE_ERROR_MODE = "signed"  # Allowed: "absolute" or "signed"
# Optional thin connector lines for matching (i,j) across N, per method.
DEFAULT_CONNECT_M1_LINES = True
DEFAULT_CONNECT_M2_LINES = True
DEFAULT_CONNECT_M3_LINES = False
DEFAULT_CONNECT_M4_LINES = False
# Plot visibility toggles per method (m1..m4).
PLOT_M1 = False
PLOT_M2 = True
PLOT_M3 = True
PLOT_M4 = True
# Horizontal jitter in N-units for divergence scatter points.
DEFAULT_X_JITTER_WIDTH = 0.16
# Plotly static-image defaults are 700x500; doubled for PDF output.
DEFAULT_PDF_WIDTH = 1400
DEFAULT_PDF_HEIGHT = 1000

METHOD_NAMES = ("m1_float_direct", "m2_float_simplified", "m3_mp_direct", "m4_mp_simplified", "m5_sympy20")


def _safe_abs_diff(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return float("nan")
    return abs(a - b)


def _fmt_val(v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return "nan"
    return f"{v:.17g}"


def _fmt_diff(v: float) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.3e}"


def _p_float_direct(n: int, i: int, j: int) -> float | None:
    ratio = comb(n, j, exact=False) / comb(n, i, exact=False)
    if ratio <= 0 or not math.isfinite(ratio):
        return None
    p = 1.0 / (1.0 + ratio ** (1.0 / (j - i)))
    if 0 < p < 1 and math.isfinite(p):
        return float(p)
    return None


def _p_float_simplified(n: int, i: int, j: int) -> float | None:
    ratio = comb(n - i, j - i, exact=False) / comb(j, i, exact=False)
    if ratio <= 0 or not math.isfinite(ratio):
        return None
    p = 1.0 / (1.0 + ratio ** (1.0 / (j - i)))
    if 0 < p < 1 and math.isfinite(p):
        return float(p)
    return None


def _p_mp_direct(n: int, i: int, j: int, mp_dps: int) -> float | None:
    old = mp.dps
    try:
        mp.dps = int(mp_dps)
        ratio = mp.binomial(n, j) / mp.binomial(n, i)
        if ratio <= 0:
            return None
        p = mp.mpf(1) / (1 + ratio ** (mp.mpf(1) / (j - i)))
        val = float(p)
    finally:
        mp.dps = old
    if 0 < val < 1 and math.isfinite(val):
        return val
    return None


def _p_mp_simplified(n: int, i: int, j: int, mp_dps: int) -> float | None:
    old = mp.dps
    try:
        mp.dps = int(mp_dps)
        ratio = mp.binomial(n - i, j - i) / mp.binomial(j, i)
        if ratio <= 0:
            return None
        p = mp.mpf(1) / (1 + ratio ** (mp.mpf(1) / (j - i)))
        val = float(p)
    finally:
        mp.dps = old
    if 0 < val < 1 and math.isfinite(val):
        return val
    return None


def _p_sympy_direct(n: int, i: int, j: int, digits: int) -> float | None:
    n_sym = S(n)
    ratio = binomial(n_sym, j) / binomial(n_sym, i)
    if ratio <= 0:
        return None
    p_expr = S(1) / (1 + ratio ** (S(1) / (j - i)))
    try:
        if not (p_expr > 0 and p_expr < 1):
            return None
    except TypeError:
        pass
    try:
        val = float(p_expr.evalf(int(digits)))
    except (TypeError, ValueError):
        return None
    if 0 < val < 1 and math.isfinite(val):
        return val
    return None


def _p_sympy_numeric(n: int, i: int, j: int, digits: int) -> float | None:
    """Numeric SymPy fallback (high precision float eval, not exact symbolic)."""
    guard_digits = int(max(digits + 20, 40))
    n_sym = S(n)
    num = binomial(n_sym, j).evalf(guard_digits)
    den = binomial(n_sym, i).evalf(guard_digits)
    if den == 0:
        return None
    ratio = num / den
    try:
        ratio_f = float(ratio)
    except (TypeError, ValueError):
        ratio_f = float("nan")
    if ratio_f <= 0 or not math.isfinite(ratio_f):
        return None
    exp = S(1) / (j - i)
    p_num = S(1) / (1 + ratio ** exp)
    try:
        val = float(p_num.evalf(int(max(digits, 1))))
    except (TypeError, ValueError):
        return None
    if 0 < val < 1 and math.isfinite(val):
        return val
    return None


def _p_sympy_exact_with_fallback(n: int, i: int, j: int, digits: int) -> tuple[float | None, bool]:
    """
    Try exact symbolic SymPy first.
    If exact path errors, fall back to numeric SymPy and return (value, exact_failed=True).
    """
    try:
        val = _p_sympy_direct(n, i, j, digits)
        if val is not None:
            return val, False
    except Exception:
        # For diagnostics we continue by falling back to numeric mode.
        pass
    return _p_sympy_numeric(n, i, j, digits), True


def _pairs_for_n(n: int, pair_count: int = DEFAULT_PAIR_COUNT) -> list[tuple[int, int]]:
    """
    Coarse-to-fine heuristic:
    - Seed with close/far pairs around kN, 0.5N, (1-k)N.
    - If more pairs are requested, refine gaps in breadth-first order.
    """
    if pair_count <= 0:
        return []

    k = float(DEFAULT_K_FRACTION)
    a = int(k * n)
    m = int(0.5 * n)
    b = int((1.0 - k) * n)

    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add_pair(i: int, j: int) -> bool:
        if len(out) >= pair_count:
            return False
        ii = max(0, min(int(i), n - 1))
        jj = max(ii + 1, min(int(j), n))
        pair = (ii, jj)
        if pair in seen:
            return False
        seen.add(pair)
        out.append(pair)
        return True

    # Requested baseline ordering: first, outside, inside, last.
    for i, j in ((a, a + 1), (a, b), (m, m + 1), (b, b + 1)):
        add_pair(i, j)
        if len(out) >= pair_count:
            return out

    # Breadth-first refinement across the two main gaps, then their children.
    gaps: deque[tuple[int, int]] = deque([(a, m), (m, b)])
    while gaps and len(out) < pair_count:
        left, right = gaps.popleft()
        if right - left < 2:
            continue
        center = int((left + right) / 2)

        # Mix far and close pairs within each gap.
        for i, j in (
            (left, center),
            (center, center + 1),
            (center, right),
            (left, left + 1),
            (right, right + 1),
        ):
            add_pair(i, j)
            if len(out) >= pair_count:
                return out

        if center - left >= 2:
            gaps.append((left, center))
        if right - center >= 2:
            gaps.append((center, right))

    # Small-range fallback to guarantee enough pairs if needed.
    if len(out) < pair_count:
        for i in range(n):
            for j in range(i + 1, n + 1):
                add_pair(i, j)
                if len(out) >= pair_count:
                    return out
    return out


def _evaluate_with_timing(
    n: int,
    pairs: list[tuple[int, int]],
    method: Callable[[int, int, int], float | None],
) -> tuple[list[float | None], float]:
    t0 = time.perf_counter()
    vals = [method(n, i, j) for i, j in pairs]
    t1 = time.perf_counter()
    return vals, (t1 - t0)


def main() -> None:
    n_min = int(DEFAULT_N_MIN)
    n_max = int(DEFAULT_N_MAX)
    mp_dps = int(DEFAULT_MP_DPS)
    sympy_digits = int(DEFAULT_SYMPY_DIGITS)
    pair_count = int(DEFAULT_PAIR_COUNT)
    if n_min > n_max:
        raise ValueError("n_min must be <= n_max")
    error_mode = str(DEFAULT_DIVERGENCE_ERROR_MODE).strip().lower()
    if error_mode not in ("absolute", "signed"):
        raise ValueError(
            f"Invalid DEFAULT_DIVERGENCE_ERROR_MODE={DEFAULT_DIVERGENCE_ERROR_MODE!r}; "
            "expected 'absolute' or 'signed'."
        )

    total_timings = {name: 0.0 for name in METHOD_NAMES}
    global_max_spread = 0.0
    global_sympy_exact_fail_count = 0
    method_x_offsets = {
        "m1_float_direct": -0.18,
        "m2_float_simplified": -0.06,
        "m3_mp_direct": 0.06,
        "m4_mp_simplified": 0.18,
    }
    # Colorblind-safe palette (Okabe-Ito inspired).
    method_colors = {
        "m1_float_direct": "#0072B2",      # blue
        "m2_float_simplified": "#E69F00",  # orange
        "m3_mp_direct": "#CC79A7",         # reddish purple
        "m4_mp_simplified": "#56B4E9",     # sky blue
    }
    method_connect_lines = {
        "m1_float_direct": bool(DEFAULT_CONNECT_M1_LINES),
        "m2_float_simplified": bool(DEFAULT_CONNECT_M2_LINES),
        "m3_mp_direct": bool(DEFAULT_CONNECT_M3_LINES),
        "m4_mp_simplified": bool(DEFAULT_CONNECT_M4_LINES),
    }
    method_plot_enabled = {
        "m1_float_direct": bool(PLOT_M1),
        "m2_float_simplified": bool(PLOT_M2),
        "m3_mp_direct": bool(PLOT_M3),
        "m4_mp_simplified": bool(PLOT_M4),
    }
    divergence_points: dict[str, dict[str, list]] = {
        "m1_float_direct": {"x": [], "y": [], "text": [], "order_idx": [], "n": []},
        "m2_float_simplified": {"x": [], "y": [], "text": [], "order_idx": [], "n": []},
        "m3_mp_direct": {"x": [], "y": [], "text": [], "order_idx": [], "n": []},
        "m4_mp_simplified": {"x": [], "y": [], "text": [], "order_idx": [], "n": []},
    }

    report_lines: list[str] = [
        "Columns: N i j | p_m1 p_m2 p_m3 p_m4 p_m5 | "
        "|m1-m3| |m2-m4| |m1-m2| |m3-m4| |m3-m5| |m4-m5| spread"
    ]

    for n in range(n_min, n_max + 1):
        pairs = _pairs_for_n(n, pair_count=pair_count)

        vals_m1, t_m1 = _evaluate_with_timing(n, pairs, lambda nn, i, j: _p_float_direct(nn, i, j))
        vals_m2, t_m2 = _evaluate_with_timing(n, pairs, lambda nn, i, j: _p_float_simplified(nn, i, j))
        vals_m3, t_m3 = _evaluate_with_timing(n, pairs, lambda nn, i, j: _p_mp_direct(nn, i, j, mp_dps))
        vals_m4, t_m4 = _evaluate_with_timing(n, pairs, lambda nn, i, j: _p_mp_simplified(nn, i, j, mp_dps))
        t_m5_start = time.perf_counter()
        vals_m5: list[float | None] = []
        sympy_exact_fail_count = 0
        for i, j in pairs:
            v5, exact_failed = _p_sympy_exact_with_fallback(n, i, j, sympy_digits)
            vals_m5.append(v5)
            if exact_failed:
                sympy_exact_fail_count += 1
        global_sympy_exact_fail_count += sympy_exact_fail_count
        t_m5 = time.perf_counter() - t_m5_start

        total_timings["m1_float_direct"] += t_m1
        total_timings["m2_float_simplified"] += t_m2
        total_timings["m3_mp_direct"] += t_m3
        total_timings["m4_mp_simplified"] += t_m4
        total_timings["m5_sympy20"] += t_m5

        n_spreads: list[float] = []
        n_max_diffs = {
            "|m1-m3|": 0.0,
            "|m2-m4|": 0.0,
            "|m1-m2|": 0.0,
            "|m3-m4|": 0.0,
            "|m3-m5|": 0.0,
            "|m4-m5|": 0.0,
        }

        for idx, (i, j) in enumerate(pairs):
            m1 = vals_m1[idx]
            m2 = vals_m2[idx]
            m3 = vals_m3[idx]
            m4 = vals_m4[idx]
            m5 = vals_m5[idx]

            d13 = _safe_abs_diff(m1, m3)
            d24 = _safe_abs_diff(m2, m4)
            d12 = _safe_abs_diff(m1, m2)
            d34 = _safe_abs_diff(m3, m4)
            d35 = _safe_abs_diff(m3, m5)
            d45 = _safe_abs_diff(m4, m5)

            finite_vals = [v for v in (m1, m2, m3, m4, m5) if v is not None and math.isfinite(v)]
            spread = (max(finite_vals) - min(finite_vals)) if finite_vals else float("nan")
            if math.isfinite(spread):
                n_spreads.append(spread)
                global_max_spread = max(global_max_spread, spread)

            for key, val in (
                ("|m1-m3|", d13),
                ("|m2-m4|", d24),
                ("|m1-m2|", d12),
                ("|m3-m4|", d34),
                ("|m3-m5|", d35),
                ("|m4-m5|", d45),
            ):
                if math.isfinite(val):
                    n_max_diffs[key] = max(n_max_diffs[key], val)

            # Divergence plot uses SymPy as reference baseline.
            if m5 is not None and math.isfinite(m5):
                for method_name, method_value in (
                    ("m1_float_direct", m1),
                    ("m2_float_simplified", m2),
                    ("m3_mp_direct", m3),
                    ("m4_mp_simplified", m4),
                ):
                    if method_value is None or not math.isfinite(method_value):
                        continue
                    signed_diff = method_value - m5
                    if error_mode == "absolute":
                        raw_diff = abs(signed_diff)
                        # Keep log-scale display stable when divergence is exactly zero.
                        y_plot = max(raw_diff, 1e-30)
                    else:
                        raw_diff = signed_diff
                        y_plot = raw_diff
                    # Deterministic micro-jitter so overlapping points at same N are visible.
                    seed = (int(i) * 1315423911) ^ (int(j) * 2654435761)
                    frac = (seed & 0xFFFF) / 65535.0  # [0,1]
                    centered = frac - 0.5  # [-0.5, 0.5]
                    x_plot = (
                        float(n)
                        + float(method_x_offsets.get(method_name, 0.0))
                        + centered * float(DEFAULT_X_JITTER_WIDTH)
                    )
                    divergence_points[method_name]["x"].append(x_plot)
                    divergence_points[method_name]["y"].append(y_plot)
                    divergence_points[method_name]["text"].append(
                        f"N={n}, order={idx}, i={i}, j={j}, signed_diff={signed_diff:.3e}, abs_diff={abs(signed_diff):.3e}"
                    )
                    divergence_points[method_name]["order_idx"].append(int(idx))
                    divergence_points[method_name]["n"].append(int(n))

            report_lines.append(
                f"N={n:3d} i={i:3d} j={j:3d} | "
                f"{_fmt_val(m1)} {_fmt_val(m2)} {_fmt_val(m3)} {_fmt_val(m4)} {_fmt_val(m5)} | "
                f"{_fmt_diff(d13)} {_fmt_diff(d24)} {_fmt_diff(d12)} {_fmt_diff(d34)} {_fmt_diff(d35)} {_fmt_diff(d45)} {_fmt_diff(spread)}"
            )

        mean_spread = (sum(n_spreads) / len(n_spreads)) if n_spreads else float("nan")
        max_spread = max(n_spreads) if n_spreads else float("nan")

        timing_line = (
            f"N={n} timings_sec: m1={t_m1:.6f} m2={t_m2:.6f} m3={t_m3:.6f} m4={t_m4:.6f} m5={t_m5:.6f}"
        )
        summary_line = (
            f"N={n} summary: mean_spread={_fmt_diff(mean_spread)} max_spread={_fmt_diff(max_spread)} "
            f"max|m1-m3|={_fmt_diff(n_max_diffs['|m1-m3|'])} "
            f"max|m2-m4|={_fmt_diff(n_max_diffs['|m2-m4|'])} "
            f"max|m1-m2|={_fmt_diff(n_max_diffs['|m1-m2|'])} "
            f"max|m3-m4|={_fmt_diff(n_max_diffs['|m3-m4|'])} "
            f"max|m3-m5|={_fmt_diff(n_max_diffs['|m3-m5|'])} "
            f"max|m4-m5|={_fmt_diff(n_max_diffs['|m4-m5|'])} "
            f"sympy_exact_fails={sympy_exact_fail_count}"
        )
        report_lines.append(timing_line)
        report_lines.append(summary_line)
        report_lines.append("-" * 120)

        # Live progress so long runs show activity immediately.
        print(f"[progress] {timing_line}")
        print(f"[progress] {summary_line}")

    n_count = (n_max - n_min + 1)
    report_lines.append("GLOBAL TIMING (sum across N):")
    for name in METHOD_NAMES:
        report_lines.append(
            f"  {name}: total_sec={total_timings[name]:.6f} avg_per_N_sec={total_timings[name] / n_count:.6f}"
        )
    report_lines.append(f"GLOBAL sympy_exact_fails={global_sympy_exact_fail_count}")
    report_lines.append(f"GLOBAL max_spread={_fmt_diff(global_max_spread)}")

    fig = go.Figure()
    for method_name, marker_symbol in (
        ("m1_float_direct", "circle"),
        ("m2_float_simplified", "square"),
        ("m3_mp_direct", "diamond"),
        ("m4_mp_simplified", "cross"),
    ):
        if not method_plot_enabled.get(method_name, True):
            continue
        if method_connect_lines.get(method_name, False):
            by_order: dict[int, list[tuple[int, float, float]]] = {}
            for n_val, order_val, x_val, y_val in zip(
                divergence_points[method_name]["n"],
                divergence_points[method_name]["order_idx"],
                divergence_points[method_name]["x"],
                divergence_points[method_name]["y"],
            ):
                by_order.setdefault(order_val, []).append((n_val, x_val, y_val))
            for order_key, pts in by_order.items():
                if len(pts) < 2:
                    continue
                pts.sort(key=lambda t: t[0])
                fig.add_trace(
                    go.Scatter(
                        x=[p[1] for p in pts],
                        y=[p[2] for p in pts],
                        mode="lines",
                        line={"color": method_colors[method_name], "width": 1},
                        opacity=0.5,
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{method_name} line order={order_key}",
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=divergence_points[method_name]["x"],
                y=divergence_points[method_name]["y"],
                mode="markers",
                name=method_name,
                marker={
                    "symbol": marker_symbol,
                    "size": 7,
                    "color": method_colors[method_name],
                },
                text=divergence_points[method_name]["text"],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Tie p divergence from SymPy reference ({error_mode})",
        xaxis_title="N",
        yaxis_title=("abs(method - sympy)" if error_mode == "absolute" else "method - sympy"),
        template="plotly_white",
    )
    if error_mode == "absolute":
        fig.update_yaxes(type="log")
    else:
        fig.update_yaxes(type="linear", zeroline=True, zerolinewidth=1.2, zerolinecolor="#444")
    plot_filetype = str(DEFAULT_DIVERGENCE_PLOT_FILETYPE).strip().lower()
    if plot_filetype not in ("html", "pdf"):
        raise ValueError(
            f"Invalid DEFAULT_DIVERGENCE_PLOT_FILETYPE={DEFAULT_DIVERGENCE_PLOT_FILETYPE!r}; "
            "expected 'html' or 'pdf'."
        )
    plot_path = f"{DEFAULT_DIVERGENCE_PLOT_BASENAME}.{plot_filetype}"
    out_parent = os.path.dirname(os.path.abspath(plot_path))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    if plot_filetype == "html":
        plotly_plot(fig, filename=plot_path, auto_open=True)
        report_lines.append(f"Wrote interactive divergence plot: {plot_path}")
    else:
        fig.write_image(plot_path, width=DEFAULT_PDF_WIDTH, height=DEFAULT_PDF_HEIGHT)
        report_lines.append(f"Wrote divergence plot PDF: {plot_path}")

    print()
    print("=== COMPLETE REPORT ===")
    for line in report_lines:
        print(line)


if __name__ == "__main__":
    main()
