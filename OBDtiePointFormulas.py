"""
Closed-form formulas for the first and second tie points (p where P(X=i)=P(X=j)) above p=0.5.

General tie point for pair (i,j), 0 <= i < j <= n:
  p = 1 / (1 + [C(n,j)/C(n,i)]^(1/(j-i)))

--- FIRST tie point above 0.5 ---

  Even n:  (i,j) = (n/2, n/2+1),  j-i = 1
    p1 = (n+2) / (2(n+1))  =  0.5 * (n+2)/(n+1)

  Odd n:   (i,j) = ((n-1)/2, (n+3)/2),  j-i = 2
    p1 = sqrt(n+3) / (sqrt(n+3) + sqrt(n-1))

--- SECOND tie point above 0.5 ---

  We are sure there is no single (i,j) pattern for all n (e.g. "second = (n/2+1, n/2+2)
  for even n" fails at n=4). The second-smallest p > 0.5 can come from a different pair:
  e.g. n=4, order is (2,3)->0.6, (1,4)->1/(1+(1/4)^(1/3))~0.6135, (3,4)->0.8, so second
  is (1,4). A closed form could still exist if the (i,j) that gives the second tie can
  be described as a function of n (e.g. piecewise); then second_tie = the usual formula
  for that (i,j). Until then, use second_tie_above_half(n) for a numeric value from
  all_tie_points.
"""

import math


def first_tie_above_half(n: int) -> float:
    """First tie point p > 0.5 (same as first_swap_prob in OBDswapGraph1)."""
    if n % 2 == 0:
        return 0.5 * (n + 2) / (n + 1)
    else:
        return math.sqrt(n + 3) / (math.sqrt(n + 3) + math.sqrt(n - 1))


def second_tie_above_half(n: int) -> float:
    """Second tie point p > 0.5. No simple closed form for all n; computed from all_tie_points."""
    from OBDsaveSourceData import all_tie_points
    arr = all_tie_points(n)
    above_half = arr[arr > 0.5]
    if len(above_half) < 2:
        raise ValueError(f"n={n} has fewer than 2 tie points above 0.5")
    return float(above_half[1])


def _check_against_all_tie_points(n_max: int = 20) -> None:
    """Verify first-tie formula against all_tie_points (optional)."""
    from OBDsaveSourceData import all_tie_points
    for n in range(2, n_max + 1):
        arr = all_tie_points(n)
        above_half = arr[arr > 0.5]
        if len(above_half) >= 1:
            p1 = first_tie_above_half(n)
            assert abs(above_half[0] - p1) < 1e-6, f"n={n} first: {above_half[0]} vs {p1}"


if __name__ == "__main__":
    try:
        _check_against_all_tie_points()
        print("First tie formula matches all_tie_points for n=2..20.")
    except ImportError:
        print("Run from project with numpy/OBDsaveSourceData to verify.")
