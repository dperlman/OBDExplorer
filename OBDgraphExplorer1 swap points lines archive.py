"""
Simple line graph: expected value (or E[rank] when sorted) vs p or n. n=2..100.
E vs P: dual-thumb n control; E vs N: dual-thumb p control (same layout). Multiple overlays curves in both modes.
Checkboxes: sorted, scaled, log x / log y, swap points (E vs P; hairlines), multiple.

Binomial series and tie points are loaded from ``data/`` (see ``GRAPH_DATA_PATH``, ``TIE_POINTS_PATH``).
Tie pickles use ``float_with_pairs_by_n``; the explorer builds plain p lists with ``[p for p, _ in recs]``.
Generate data with ``OBDsaveSourceData.py`` (e.g. default run or ``--save-tie-points``).

Generated HTML loads Plotly; dual n range (Codeconvey-style) for E vs P;
a two-thumb slider pattern from [Codeconvey](https://codeconvey.com/pure-css-range-slider-with-2-handles/) /
[CodePen PoGpyKa](https://codepen.io/scottbram/pen/PoGpyKa) (no extra widget script).
Python: stdlib json, pickle only (binomial PMF lives in OBDsaveSourceData for --save-graph-data).
"""
import json
import os
import pickle
import sys

N_MIN, N_MAX = 2, 100
N_VALS = list(range(N_MIN, N_MAX + 1))
P_STEPS = 1001
P_HALF_START = (P_STEPS - 1) // 2  # index where p = 0.5
p_values = [i / (P_STEPS - 1) for i in range(P_STEPS)]

DATA_DIR = "data"
GRAPH_DATA_PATH = os.path.join(DATA_DIR, "graph_data.pkl")
TIE_POINTS_PATH = os.path.join(DATA_DIR, "tie_points.pkl")


def _cmd_save_graph_data() -> str:
    return (
        "python OBDsaveSourceData.py --save-graph-data "
        f"--graph-n-min {N_MIN} --graph-n-max {N_MAX} --p-steps {P_STEPS}"
    )


def _cmd_save_tie_points() -> str:
    return (
        "python OBDsaveSourceData.py --save-tie-points "
        f"--tie-n-min {N_MIN} --tie-n-max {N_MAX}"
    )


def _load_binomial_series_from_pickle(path: str) -> list:
    """
    Load graph_data.pkl; return binomial_data list for n = N_MIN..N_MAX and p grid P_STEPS.
    Exit process with instructions if file missing or coverage insufficient.
    """
    if not os.path.isfile(path):
        print(
            f"ERROR: Graph data file not found:\n  {os.path.abspath(path)}\n\n"
            "Create it from the project root with:\n"
            f"  {_cmd_save_graph_data()}\n",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(
            f"ERROR: Could not read graph pickle {path!r}: {e}\n\n"
            f"Regenerate with:\n  {_cmd_save_graph_data()}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    n_lo = data.get("n_min")
    n_hi = data.get("n_max")
    p_st = data.get("p_steps")
    bd = data.get("binomial_data")
    if n_lo is None or n_hi is None or p_st is None or bd is None:
        print(
            f"ERROR: {path!r} is missing required keys "
            "(need n_min, n_max, p_steps, binomial_data).\n\n"
            f"Regenerate with:\n  {_cmd_save_graph_data()}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    n_lo, n_hi, p_st = int(n_lo), int(n_hi), int(p_st)
    issues: list[str] = []
    if n_lo > N_MIN:
        issues.append(
            f"  n_min={n_lo} is greater than explorer N_MIN={N_MIN} "
            f"(need graph data with n_min <= {N_MIN})."
        )
    if n_hi < N_MAX:
        issues.append(
            f"  n_max={n_hi} is less than explorer N_MAX={N_MAX} "
            f"(need graph data with n_max >= {N_MAX})."
        )
    if p_st != P_STEPS:
        issues.append(
            f"  p_steps={p_st} must equal explorer P_STEPS={P_STEPS} "
            "(same p grid; regenerate with --p-steps matching the explorer)."
        )

    expected_len = (n_hi - n_lo + 1) * p_st
    if len(bd) != expected_len:
        issues.append(
            f"  binomial_data length {len(bd)} != expected ({n_hi}-{n_lo}+1)*{p_st} = {expected_len}."
        )

    if issues:
        print(
            f"ERROR: Graph data in {path!r} does not meet explorer requirements "
            f"(n={N_MIN}..{N_MAX}, p_steps={P_STEPS}):\n"
            + "\n".join(issues)
            + "\n\nRegenerate or extend with (adjust flags as needed):\n"
            f"  {_cmd_save_graph_data()}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    out: list = []
    for n in range(N_MIN, N_MAX + 1):
        start = (n - n_lo) * p_st
        chunk = bd[start : start + p_st]
        if len(chunk) != P_STEPS:
            print(
                f"ERROR: Internal slice for n={n} has length {len(chunk)}, expected {P_STEPS}.\n\n"
                f"Regenerate with:\n  {_cmd_save_graph_data()}\n",
                file=sys.stderr,
            )
            sys.exit(1)
        out.extend(chunk)

    return out


def _tie_ps_above_half_from_pair_records(
    recs: list,
) -> list[float]:
    """
    Current pickle format: list of (p, [(i, j), ...]). Plain tie p values are [p for p, _ in recs].
    Return sorted p in (0.5, 1) for swap-point hairlines.
    """
    if not recs:
        return []
    pts = [float(p) for p, _ in recs]
    return sorted([round(p, 6) for p in pts if 0.5 < p < 1])


def _load_tie_points(path: str) -> dict:
    """Load tie_points.pkl; return dict n -> list of p in (0.5, 1) for n in N_VALS. Empty dict if missing.

    Prefers ``float_with_pairs_by_n`` (OBDsaveSourceData current format). Falls back to ``float_by_n``
    if a key is missing there (older pickles).
    """
    if not os.path.isfile(path):
        print(
            f"WARNING: Tie points file not found:\n  {os.path.abspath(path)}\n\n"
            "Swap-points will have no effect until you create it, e.g.:\n"
            f"  {_cmd_save_tie_points()}\n"
        )
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(
            f"WARNING: Could not read tie points {path!r}: {e}\n"
            f"Swap-points disabled. Regenerate with:\n  {_cmd_save_tie_points()}\n"
        )
        return {}
    float_with_pairs_by_n = data.get("float_with_pairs_by_n", {})
    float_by_n = data.get("float_by_n", {})
    out = {}
    for n in N_VALS:
        recs = float_with_pairs_by_n.get(n)
        if recs is not None:
            above_half = _tie_ps_above_half_from_pair_records(recs)
        else:
            raw = float_by_n.get(n)
            if raw is None:
                continue
            try:
                pts = [float(x) for x in raw]
            except (TypeError, ValueError):
                pts = [float(raw)]
            above_half = sorted([round(p, 6) for p in pts if 0.5 < p < 1])
        if above_half:
            out[n] = above_half

    missing_n = [n for n in N_VALS if n not in out]
    if missing_n:
        print(
            "WARNING: Tie data does not include float_with_pairs_by_n (or float_by_n) for all explorer n "
            f"({N_MIN}..{N_MAX}). Missing or empty for n in: "
            f"{missing_n[:10]}{'...' if len(missing_n) > 10 else ''}\n"
            "Swap-points may be incomplete. Extend coverage with e.g.:\n"
            f"  {_cmd_save_tie_points()}\n"
            "(use --tie-n-min / --tie-n-max to cover the full n range.)\n"
        )
    return out


def _build_html(binomial_data: list, tie_points_by_n: dict) -> str:
    p_labels_json = json.dumps([round(p, 4) for p in p_values])
    tie_points_json = json.dumps(tie_points_by_n)
    p_idx_max = P_STEPS - 1
    p_default_hi = min(P_HALF_START + 50, p_idx_max)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; overflow: hidden; box-sizing: border-box; font-family: sans-serif; }}
    * {{ box-sizing: border-box; }}
    .container {{ display: flex; flex-direction: column; width: 100vw; height: 100vh; min-height: 0; }}
    .graph-wrap {{ flex: 1; min-height: 0; width: 100%; }}
    .controls {{
      flex-shrink: 0;
      display: flex;
      flex-wrap: nowrap;
      justify-content: flex-start;
      align-items: center;
      gap: 12px;
      padding: 10px 12px 12px;
      overflow-x: auto;
      overflow-y: hidden;
      white-space: nowrap;
    }}
    .controls > * {{ flex: 0 0 auto; }}
    .controls label {{ display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }}
    /* Dual n / p range: pattern from https://codeconvey.com/pure-css-range-slider-with-2-handles/ */
    #ctrl-n-range .obd-dual-slider, #ctrl-p-range .obd-dual-slider {{
      position: relative;
      height: 14px;
      border-radius: 10px;
      text-align: left;
      margin: 6px 0;
      width: 240px;
      max-width: 85vw;
      flex: 0 0 auto;
    }}
    #ctrl-n-range .obd-dual-slider-trackwrap, #ctrl-p-range .obd-dual-slider-trackwrap {{
      position: absolute;
      left: 13px;
      right: 15px;
      height: 14px;
    }}
    #ctrl-n-range .obd-dual-inverse-left, #ctrl-p-range .obd-dual-inverse-left {{
      position: absolute;
      left: 0;
      height: 10px;
      border-radius: 10px;
      background-color: #ccc;
      margin: 0 7px;
    }}
    #ctrl-n-range .obd-dual-inverse-right, #ctrl-p-range .obd-dual-inverse-right {{
      position: absolute;
      right: 0;
      height: 10px;
      border-radius: 10px;
      background-color: #ccc;
      margin: 0 7px;
    }}
    #ctrl-n-range .obd-dual-range, #ctrl-p-range .obd-dual-range {{
      position: absolute;
      left: 0;
      top: -1px;
      height: 16px;
      border-radius: 14px;
      background-color: #31689b;
    }}
    #ctrl-n-range .obd-dual-thumb, #ctrl-p-range .obd-dual-thumb {{
      z-index: 2;
      position: absolute;
      top: -5px;
      margin-left: -11px;
      width: 24px;
      height: 24px;
      border-radius: 25%;
      background-color: #fff;
      box-shadow: 0 3px 8px rgba(0, 0, 0, 0.4);
      outline: none;
      cursor: pointer;
      pointer-events: none;
    }}
    #ctrl-n-range .obd-dual-slider > input[type=range], #ctrl-p-range .obd-dual-slider > input[type=range] {{
      position: absolute;
      pointer-events: none;
      -webkit-appearance: none;
      z-index: 3;
      height: 14px;
      top: -2px;
      width: 100%;
      opacity: 0;
      cursor: pointer;
    }}
    #ctrl-n-range .obd-dual-slider > input[type=range]::-webkit-slider-thumb, #ctrl-p-range .obd-dual-slider > input[type=range]::-webkit-slider-thumb {{
      pointer-events: all;
      width: 24px;
      height: 24px;
      border: 0 none;
      border-radius: 0;
      background: transparent;
      -webkit-appearance: none;
    }}
    #ctrl-n-range .obd-dual-slider > input[type=range]::-moz-range-thumb, #ctrl-p-range .obd-dual-slider > input[type=range]::-moz-range-thumb {{
      pointer-events: all;
      width: 24px;
      height: 24px;
      border: 0 none;
      border-radius: 0;
      background: transparent;
    }}
    #ctrl-n-range .obd-dual-slider > input[type=range]::-ms-thumb, #ctrl-p-range .obd-dual-slider > input[type=range]::-ms-thumb {{
      pointer-events: all;
      width: 24px;
      height: 24px;
      border: 0 none;
      border-radius: 0;
      background: transparent;
    }}
    #ctrl-n-range .obd-dual-slider > input[type=range]::-ms-fill-lower,
    #ctrl-n-range .obd-dual-slider > input[type=range]::-ms-fill-upper,
    #ctrl-p-range .obd-dual-slider > input[type=range]::-ms-fill-lower,
    #ctrl-p-range .obd-dual-slider > input[type=range]::-ms-fill-upper {{
      background: transparent;
      border: 0 none;
    }}
    #ctrl-n-range .obd-dual-slider > input[type=range]::-ms-track, #ctrl-p-range .obd-dual-slider > input[type=range]::-ms-track {{
      background: transparent;
      color: transparent;
    }}
    #ctrl-n-range .obd-dual-slider > input[type=range]::-moz-range-track, #ctrl-p-range .obd-dual-slider > input[type=range]::-moz-range-track {{
      background: transparent;
      color: transparent;
      -moz-appearance: none;
    }}
    #ctrl-n-range .obd-dual-slider > input[type=range]:focus, #ctrl-p-range .obd-dual-slider > input[type=range]:focus {{
      outline: none;
    }}
    #ctrl-n-range .obd-dual-slider > input[type=range]:focus::-webkit-slider-runnable-track, #ctrl-p-range .obd-dual-slider > input[type=range]:focus::-webkit-slider-runnable-track {{
      background: transparent;
      border: transparent;
    }}
    #ctrl-n-range .obd-dual-slider > input[type=range]::-ms-tooltip, #ctrl-p-range .obd-dual-slider > input[type=range]::-ms-tooltip {{
      display: none;
    }}
    .ctrl-panel {{ display: flex; align-items: center; gap: 8px; }}
    #ctrl-n {{ align-items: center; flex-wrap: nowrap; min-width: 0; flex: 0 0 auto; max-width: none; }}
    #ctrl-n-range, #ctrl-p-range {{
      display: none;
      flex-direction: row;
      align-items: center;
      gap: 10px;
      overflow: visible;
      flex-shrink: 0;
      max-width: none;
    }}
    /* Slider before label so thumb track stays fixed; label width won't push the control */
    #ctrl-n-range .obd-n-range-label, #ctrl-p-range .obd-p-range-label {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    #n-range-side-label, #p-range-side-label {{
      display: inline-block;
      min-width: 12ch;
      font-weight: bold;
      font-variant-numeric: tabular-nums;
      text-align: left;
    }}
    #ctrl-n-range.obd-dual-single-mode .obd-dual-thumb-right, #ctrl-p-range.obd-dual-single-mode .obd-dual-thumb-right {{
      visibility: hidden;
      pointer-events: none;
    }}
    #ctrl-n-range.obd-dual-single-mode #n-range-high, #ctrl-p-range.obd-dual-single-mode #p-range-high {{
      pointer-events: none;
      z-index: 2;
    }}
    #ctrl-n-range.obd-dual-single-mode #n-range-low, #ctrl-p-range.obd-dual-single-mode #p-range-low {{
      z-index: 4;
    }}
    label.multiple-disabled {{ opacity: 0.45; cursor: not-allowed; }}
    label.multiple-disabled input {{ cursor: not-allowed; }}
    label.swap-points-disabled {{ opacity: 0.45; cursor: not-allowed; }}
    label.swap-points-disabled input {{ cursor: not-allowed; }}
    label.every-step-disabled {{ opacity: 0.45; cursor: not-allowed; }}
    label.every-step-disabled input {{ cursor: not-allowed; }}
    #every-step {{ width: 4.5em; vertical-align: middle; }}
  </style>
</head>
<body>
  <div class="container">
    <div id="graph" class="graph-wrap"></div>
    <div class="controls">
      <label for="mode">Graph:</label>
      <select id="mode">
        <option value="vp">E vs P</option>
        <option value="vn">E vs N</option>
      </select>
      <label><input type="checkbox" id="sorted"> Sorted</label>
      <label><input type="checkbox" id="scaled" checked> Scaled</label>
      <label><input type="checkbox" id="log-x"> Log x</label>
      <label><input type="checkbox" id="log-y"> Log y</label>
      <label id="swap-points-label" for="swap-points"><input type="checkbox" id="swap-points"> Swap points</label>
      <label id="multiple-label" for="multiple"><input type="checkbox" id="multiple"> Multiple</label>
      <label id="every-label" class="controls-every" for="every-step">Every... <input type="number" id="every-step" min="1" step="1" value="1" aria-label="Plot every nth curve when Multiple is on"></label>
      <div id="ctrl-n" class="ctrl-panel">
        <div id="ctrl-n-range">
          <div id="n-dual-range" class="obd-dual-slider" aria-label="n range">
            <div class="obd-dual-slider-trackwrap">
              <div class="obd-dual-inverse-left"></div>
              <div class="obd-dual-inverse-right"></div>
              <div class="obd-dual-range"></div>
              <span class="obd-dual-thumb obd-dual-thumb-left" aria-hidden="true"></span>
              <span class="obd-dual-thumb obd-dual-thumb-right" aria-hidden="true"></span>
            </div>
            <input type="range" id="n-range-low" min="{N_MIN}" max="{N_MAX}" step="1" value="{N_MIN}" aria-label="Minimum n">
            <input type="range" id="n-range-high" min="{N_MIN}" max="{N_MAX}" step="1" value="10" aria-label="Maximum n">
          </div>
          <label for="n-range-low" class="obd-n-range-label"><b>n</b> <span id="n-range-side-label">= {N_MIN}</span></label>
        </div>
      </div>
      <div id="ctrl-p" class="ctrl-panel" style="display: none;">
        <div id="ctrl-p-range">
          <div id="p-dual-range" class="obd-dual-slider" aria-label="p range">
            <div class="obd-dual-slider-trackwrap">
              <div class="obd-dual-inverse-left"></div>
              <div class="obd-dual-inverse-right"></div>
              <div class="obd-dual-range"></div>
              <span class="obd-dual-thumb obd-dual-thumb-left" aria-hidden="true"></span>
              <span class="obd-dual-thumb obd-dual-thumb-right" aria-hidden="true"></span>
            </div>
            <input type="range" id="p-range-low" min="{P_HALF_START}" max="{p_idx_max}" step="1" value="{P_HALF_START}" aria-label="Minimum p index">
            <input type="range" id="p-range-high" min="{P_HALF_START}" max="{p_idx_max}" step="1" value="{p_default_hi}" aria-label="Maximum p index">
          </div>
          <label for="p-range-low" class="obd-p-range-label"><b>p</b> <span id="p-range-side-label">= {p_values[P_HALF_START]:.3f}</span></label>
        </div>
      </div>
    </div>
  </div>

  <script>
    const BINOMIAL_DATA = {json.dumps(binomial_data)};
    const P_LABELS = {p_labels_json};
    const TIE_POINTS_BY_N = {tie_points_json};

    function getBinomialIndex(n, pIdx) {{ return (n - {N_MIN}) * {P_STEPS} + pIdx; }}
    const P_HALF_START = {P_HALF_START};
    const P_IDX_MAX = {p_idx_max};
    var nRangeLo = {N_MIN}, nRangeHi = 10;
    var pRangeLo = {P_HALF_START}, pRangeHi = {p_default_hi};

    function updateNRangeSideLabel() {{
      var el = document.getElementById("n-range-side-label");
      if (!el || document.getElementById("mode").value !== "vp") return;
      var mult = document.getElementById("multiple").checked;
      if (mult) {{
        var a = Math.min(nRangeLo, nRangeHi), b = Math.max(nRangeLo, nRangeHi);
        el.textContent = "= " + a + "–" + b;
      }} else {{
        var n = Math.min(nRangeLo, nRangeHi);
        el.textContent = "= " + n;
      }}
    }}

    function updatePRangeSideLabel() {{
      var el = document.getElementById("p-range-side-label");
      if (!el || document.getElementById("mode").value !== "vn") return;
      var mult = document.getElementById("multiple").checked;
      function pFmt(i) {{ return parseFloat(P_LABELS[i]).toFixed(3); }}
      if (mult) {{
        var a = Math.min(pRangeLo, pRangeHi), b = Math.max(pRangeLo, pRangeHi);
        el.textContent = "= " + pFmt(a) + "–" + pFmt(b);
      }} else {{
        var pi = Math.min(pRangeLo, pRangeHi);
        el.textContent = "= " + pFmt(pi);
      }}
    }}

    function obdDualRangePct(v) {{
      return ((v - {N_MIN}) / ({N_MAX} - {N_MIN})) * 100;
    }}

    function applyObdDualRangeVisual() {{
      var wrap = document.querySelector("#n-dual-range .obd-dual-slider-trackwrap");
      if (!wrap) return;
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10);
      var hi = parseInt(hiIn.value, 10);
      var pLo = obdDualRangePct(lo);
      var pHi = obdDualRangePct(hi);
      var invL = wrap.querySelector(".obd-dual-inverse-left");
      var invR = wrap.querySelector(".obd-dual-inverse-right");
      var rangeEl = wrap.querySelector(".obd-dual-range");
      var thL = wrap.querySelector(".obd-dual-thumb-left");
      var thR = wrap.querySelector(".obd-dual-thumb-right");
      if (invL) invL.style.width = pLo + "%";
      if (invR) invR.style.width = (100 - pHi) + "%";
      if (rangeEl) {{
        rangeEl.style.left = pLo + "%";
        rangeEl.style.right = (100 - pHi) + "%";
      }}
      if (thL) thL.style.left = pLo + "%";
      if (thR) thR.style.left = pHi + "%";
    }}

    function syncObdDualRangeFromGlobals() {{
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!loIn || !hiIn) return;
      var mult = document.getElementById("multiple").checked;
      if (mult) {{
        var a = Math.min(nRangeLo, nRangeHi), b = Math.max(nRangeLo, nRangeHi);
        loIn.value = String(a);
        hiIn.value = String(b);
      }} else {{
        var n = Math.min(nRangeLo, nRangeHi);
        nRangeLo = n;
        nRangeHi = n;
        loIn.value = String(n);
        hiIn.value = String(n);
      }}
      applyObdDualRangeVisual();
    }}

    function obdPDualRangePct(idx) {{
      var span = P_IDX_MAX - P_HALF_START;
      if (span <= 0) return 0;
      return ((idx - P_HALF_START) / span) * 100;
    }}

    function applyObdPDualRangeVisual() {{
      var wrap = document.querySelector("#p-dual-range .obd-dual-slider-trackwrap");
      if (!wrap) return;
      var loIn = document.getElementById("p-range-low");
      var hiIn = document.getElementById("p-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10);
      var hi = parseInt(hiIn.value, 10);
      var pLo = obdPDualRangePct(lo);
      var pHi = obdPDualRangePct(hi);
      var invL = wrap.querySelector(".obd-dual-inverse-left");
      var invR = wrap.querySelector(".obd-dual-inverse-right");
      var rangeEl = wrap.querySelector(".obd-dual-range");
      var thL = wrap.querySelector(".obd-dual-thumb-left");
      var thR = wrap.querySelector(".obd-dual-thumb-right");
      if (invL) invL.style.width = pLo + "%";
      if (invR) invR.style.width = (100 - pHi) + "%";
      if (rangeEl) {{
        rangeEl.style.left = pLo + "%";
        rangeEl.style.right = (100 - pHi) + "%";
      }}
      if (thL) thL.style.left = pLo + "%";
      if (thR) thR.style.left = pHi + "%";
    }}

    function syncObdPDualRangeFromGlobals() {{
      var loIn = document.getElementById("p-range-low");
      var hiIn = document.getElementById("p-range-high");
      if (!loIn || !hiIn) return;
      var mult = document.getElementById("multiple").checked;
      if (mult) {{
        var a = Math.min(pRangeLo, pRangeHi), b = Math.max(pRangeLo, pRangeHi);
        loIn.value = String(a);
        hiIn.value = String(b);
      }} else {{
        var p = Math.min(pRangeLo, pRangeHi);
        pRangeLo = p;
        pRangeHi = p;
        loIn.value = String(p);
        hiIn.value = String(p);
      }}
      applyObdPDualRangeVisual();
    }}

    function onObdDualRangeLowInput() {{
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10);
      var hi = parseInt(hiIn.value, 10);
      var mult = document.getElementById("multiple").checked;
      if (!mult) {{
        hi = lo;
        hiIn.value = String(hi);
      }} else if (lo > hi) {{
        lo = hi;
        loIn.value = String(lo);
      }}
      nRangeLo = lo;
      nRangeHi = hi;
      updateNRangeSideLabel();
      applyObdDualRangeVisual();
      if (document.getElementById("mode").value === "vp")
        updateGraph();
    }}

    function onObdDualRangeHighInput() {{
      if (!document.getElementById("multiple").checked) return;
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10);
      var hi = parseInt(hiIn.value, 10);
      if (hi < lo) {{ hi = lo; hiIn.value = String(hi); }}
      nRangeLo = lo;
      nRangeHi = hi;
      updateNRangeSideLabel();
      applyObdDualRangeVisual();
      if (document.getElementById("mode").value === "vp")
        updateGraph();
    }}

    function onObdPDualRangeLowInput() {{
      var loIn = document.getElementById("p-range-low");
      var hiIn = document.getElementById("p-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10);
      var hi = parseInt(hiIn.value, 10);
      var mult = document.getElementById("multiple").checked;
      if (!mult) {{
        hi = lo;
        hiIn.value = String(hi);
      }} else if (lo > hi) {{
        lo = hi;
        loIn.value = String(lo);
      }}
      pRangeLo = lo;
      pRangeHi = hi;
      updatePRangeSideLabel();
      applyObdPDualRangeVisual();
      if (document.getElementById("mode").value === "vn")
        updateGraph();
    }}

    function onObdPDualRangeHighInput() {{
      if (!document.getElementById("multiple").checked) return;
      var loIn = document.getElementById("p-range-low");
      var hiIn = document.getElementById("p-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10);
      var hi = parseInt(hiIn.value, 10);
      if (hi < lo) {{ hi = lo; hiIn.value = String(hi); }}
      pRangeLo = lo;
      pRangeHi = hi;
      updatePRangeSideLabel();
      applyObdPDualRangeVisual();
      if (document.getElementById("mode").value === "vn")
        updateGraph();
    }}

    function expectedRank(point) {{
      var n = point.x.length - 1;
      var e = 0;
      for (var i = 0; i <= n; i++) e += i * point.y[point.perm[i]];
      return e;
    }}

    /** Linear interpolate y at pTarget given VP curve samples (same x/y as traces). */
    function interpolateYAtP(xa, ya, pTarget) {{
      if (!xa || !ya || !xa.length) return NaN;
      if (pTarget <= xa[0]) return ya[0];
      if (pTarget >= xa[xa.length - 1]) return ya[ya.length - 1];
      for (var ii = 0; ii < xa.length - 1; ii++) {{
        if (pTarget >= xa[ii] && pTarget <= xa[ii + 1]) {{
          var tt = (pTarget - xa[ii]) / (xa[ii + 1] - xa[ii]);
          return ya[ii] + tt * (ya[ii + 1] - ya[ii]);
        }}
      }}
      return ya[ya.length - 1];
    }}

    function buildVpXY(n, sorted, scaled) {{
      var xArr = [], yArr = [];
      for (var i = P_HALF_START; i < {P_STEPS}; i++) {{
        var pVal = parseFloat(P_LABELS[i]);
        xArr.push(pVal);
        var pt = BINOMIAL_DATA[getBinomialIndex(n, i)];
        var e = sorted ? expectedRank(pt) : n * pVal;
        if (scaled && !sorted) yArr.push(pVal);
        else yArr.push(scaled ? e / n : e);
      }}
      return {{ x: xArr, y: yArr }};
    }}

    function colorForN(n, nLo, nHi) {{
      var lo = nLo !== undefined ? nLo : {N_MIN};
      var hi = nHi !== undefined ? nHi : {N_MAX};
      var span = hi - lo;
      var t = span > 0 ? (n - lo) / span : 0;
      return "hsl(" + Math.round(240 * (1 - t)) + ", 70%, 45%)";
    }}

    function colorForPIdx(pIdx, pLo, pHi) {{
      var lo = pLo !== undefined ? pLo : P_HALF_START;
      var hi = pHi !== undefined ? pHi : P_IDX_MAX;
      var span = hi - lo;
      var t = span > 0 ? (pIdx - lo) / span : 0;
      return "hsl(" + Math.round(240 * (1 - t)) + ", 70%, 45%)";
    }}

    function getMultipleStride() {{
      var el = document.getElementById("every-step");
      if (!el) return 1;
      var v = parseInt(el.value, 10);
      if (!isFinite(v) || v < 1) return 1;
      return v;
    }}

    function syncEveryControl() {{
      var mult = document.getElementById("multiple").checked;
      var inp = document.getElementById("every-step");
      var lab = document.getElementById("every-label");
      if (!inp || !lab) return;
      if (!mult) {{
        inp.disabled = true;
        lab.classList.add("every-step-disabled");
      }} else {{
        inp.disabled = false;
        lab.classList.remove("every-step-disabled");
      }}
    }}

    function updateGraph() {{
      const mode = document.getElementById("mode").value;
      const sorted = document.getElementById("sorted").checked;
      const scaled = document.getElementById("scaled").checked;
      const logXAxis = document.getElementById("log-x").checked;
      const logYAxis = document.getElementById("log-y").checked;
      const swapPoints = document.getElementById("swap-points").checked;
      const multiple = mode === "vp" && document.getElementById("multiple").checked;
      const multipleVn = mode === "vn" && document.getElementById("multiple").checked;
      var xArr, yArr, xTitle, xRange, titleSuffix, traces, showLegend;
      var vpXYByN = {{}};
      if (mode === "vp") {{
        xTitle = "p";
        if (!logXAxis) xRange = [0.5, 1];
        if (multiple) {{
          var nLo = Math.min(nRangeLo, nRangeHi), nHi = Math.max(nRangeLo, nRangeHi);
          var strideVp = getMultipleStride();
          titleSuffix = "n=" + nLo + "–" + nHi + (strideVp > 1 ? ", every " + strideVp : "");
          traces = [];
          yArr = [];
          for (var nv = nLo; nv <= nHi; nv += strideVp) {{
            var xy = buildVpXY(nv, sorted, scaled);
            vpXYByN[nv] = xy;
            for (var j = 0; j < xy.y.length; j++) yArr.push(xy.y[j]);
            traces.push({{ x: xy.x, y: xy.y, mode: "lines", name: "n=" + nv, showlegend: false, line: {{ color: colorForN(nv, nLo, nHi), width: 1.5 }} }});
          }}
          showLegend = false;
        }} else {{
          const n = parseInt(document.getElementById("n-range-low").value, 10);
          var one = buildVpXY(n, sorted, scaled);
          xArr = one.x;
          yArr = one.y;
          titleSuffix = "n=" + n;
          traces = [{{ x: xArr, y: yArr, mode: "lines", line: {{ color: "blue", width: 1.5 }}, showlegend: false }}];
          showLegend = false;
        }}
      }} else {{
        xTitle = "n";
        if (!logXAxis) xRange = [{N_MIN}, {N_MAX}];
        if (multipleVn) {{
          var pLoI = Math.min(pRangeLo, pRangeHi), pHiI = Math.max(pRangeLo, pRangeHi);
          var strideVn = getMultipleStride();
          titleSuffix = "p=" + parseFloat(P_LABELS[pLoI]).toFixed(3) + "–" + parseFloat(P_LABELS[pHiI]).toFixed(3) + (strideVn > 1 ? ", every " + strideVn : "");
          traces = [];
          yArr = [];
          for (var pIx = pLoI; pIx <= pHiI; pIx += strideVn) {{
            var pValM = parseFloat(P_LABELS[pIx]);
            var xPart = [], yPart = [];
            for (var nVal = {N_MIN}; nVal <= {N_MAX}; nVal++) {{
              var idxM = getBinomialIndex(nVal, pIx);
              if (idxM >= BINOMIAL_DATA.length) {{ yPart.push(NaN); xPart.push(nVal); continue; }}
              var ptM = BINOMIAL_DATA[idxM];
              var eM = sorted ? expectedRank(ptM) : nVal * pValM;
              xPart.push(nVal);
              if (scaled && !sorted) yPart.push(pValM);
              else yPart.push(scaled ? eM / nVal : eM);
            }}
            for (var jm = 0; jm < yPart.length; jm++) yArr.push(yPart[jm]);
            traces.push({{ x: xPart, y: yPart, mode: "lines", name: "p=" + pValM.toFixed(3), showlegend: false, line: {{ color: colorForPIdx(pIx, pLoI, pHiI), width: 1.5 }} }});
          }}
          showLegend = false;
        }} else {{
          const pIdx = parseInt(document.getElementById("p-range-low").value, 10);
          const pVal = parseFloat(P_LABELS[pIdx]);
          xArr = [];
          yArr = [];
          for (var nVal = {N_MIN}; nVal <= {N_MAX}; nVal++) {{
            var idx = getBinomialIndex(nVal, pIdx);
            if (idx >= BINOMIAL_DATA.length) {{ yArr.push(NaN); xArr.push(nVal); continue; }}
            var pt = BINOMIAL_DATA[idx];
            var e = sorted ? expectedRank(pt) : nVal * pVal;
            xArr.push(nVal);
            if (scaled && !sorted) yArr.push(pVal);
            else yArr.push(scaled ? e / nVal : e);
          }}
          titleSuffix = "p=" + pVal.toFixed(3);
          traces = [{{ x: xArr, y: yArr, mode: "lines", line: {{ color: "blue", width: 1.5 }}, showlegend: false }}];
          showLegend = false;
        }}
      }}
      var yLabel = scaled ? "E/n" : (sorted ? "E[rank]" : "E[X]");
      var Y_MIN_SPAN = 1e-3;
      var yDataMin = Infinity, yDataMax = -Infinity;
      for (var idy = 0; idy < yArr.length; idy++) {{
        var yd = yArr[idy];
        if (typeof yd === "number" && isFinite(yd)) {{
          yDataMin = Math.min(yDataMin, yd);
          yDataMax = Math.max(yDataMax, yd);
        }}
      }}
      if (!isFinite(yDataMin)) {{ yDataMin = 0; yDataMax = 1; }}
      var yLo = Infinity, yHi = -Infinity;
      for (var i = 0; i < yArr.length; i++) {{
        if (typeof yArr[i] === "number" && isFinite(yArr[i])) {{
          yLo = Math.min(yLo, yArr[i]);
          yHi = Math.max(yHi, yArr[i]);
        }}
      }}
      if (!isFinite(yLo)) {{ yLo = 0; yHi = 1; }}
      var ySpan = yHi - yLo;
      if (ySpan < Y_MIN_SPAN) {{
        var yMid = (yLo + yHi) / 2;
        yLo = yMid - Y_MIN_SPAN / 2;
        yHi = yMid + Y_MIN_SPAN / 2;
      }} else {{
        var yPad = Math.max(ySpan * 0.05, Y_MIN_SPAN * 0.01);
        yLo -= yPad;
        yHi += yPad;
      }}
      var shapes = [];
      if (mode === "vp" && swapPoints) {{
        if (!multiple) {{
          var tiePts = TIE_POINTS_BY_N[parseInt(document.getElementById("n-range-low").value, 10)];
          if (tiePts && tiePts.length) {{
            for (var i = 0; i < tiePts.length; i++) {{
              var xh = tiePts[i];
              shapes.push({{ type: "line", x0: xh, x1: xh, y0: 0, y1: 1, yref: "paper", line: {{ color: "gray", width: 0.5 }} }});
            }}
          }}
        }} else {{
          var nLoM = Math.min(nRangeLo, nRangeHi), nHiM = Math.max(nRangeLo, nRangeHi);
          var strideM = getMultipleStride();
          var nListM = [];
          for (var nvM = nLoM; nvM <= nHiM; nvM += strideM) nListM.push(nvM);
          for (var ki = 0; ki < nListM.length; ki++) {{
            var nk = nListM[ki];
            var tiesN = TIE_POINTS_BY_N[nk];
            if (!tiesN || !tiesN.length) continue;
            for (var ti = 0; ti < tiesN.length; ti++) {{
              var pUse = tiesN[ti];
              var xhM = pUse;
              var xyCur = vpXYByN[nk];
              if (!xyCur) continue;
              var xyPrev = ki === 0 ? null : vpXYByN[nListM[ki - 1]];
              var yRawBot = ki === 0 ? null : (xyPrev ? interpolateYAtP(xyPrev.x, xyPrev.y, pUse) : NaN);
              var yRawTop = interpolateYAtP(xyCur.x, xyCur.y, pUse);
              var yBot = ki === 0 ? (logYAxis ? Math.max(yDataMin, 1e-15) * 0.95 : yLo) : yRawBot;
              var yTop = yRawTop;
              if (typeof yBot === "number" && typeof yTop === "number" && isFinite(yBot) && isFinite(yTop)) {{
                shapes.push({{
                  type: "line", x0: xhM, x1: xhM, y0: yBot, y1: yTop,
                  xref: "x", yref: "y", line: {{ color: "gray", width: 0.5 }}
                }});
              }}
            }}
          }}
        }}
      }}
      var axisNote = "";
      if (logXAxis) axisNote += " (log\u2081\u2080 x)";
      if (logYAxis) axisNote += " (log\u2081\u2080 y)";
      var xaxisLayout = {{ title: xTitle, showgrid: !(mode === "vp" && swapPoints) }};
      if (logXAxis) {{
        xaxisLayout.type = "log";
        xaxisLayout.exponentformat = "power";
        if (mode === "vp") {{
          xaxisLayout.range = [Math.log10(0.5 * 0.98), Math.log10(1)];
        }} else {{
          xaxisLayout.range = [Math.log10({N_MIN} * 0.98), Math.log10({N_MAX} * 1.02)];
        }}
      }} else {{
        xaxisLayout.range = xRange;
      }}
      var yaxisLayout = {{ title: yLabel, autorange: false }};
      if (logYAxis) {{
        var yPosLo = Infinity;
        for (var ipy = 0; ipy < yArr.length; ipy++) {{
          var ypy = yArr[ipy];
          if (typeof ypy === "number" && isFinite(ypy) && ypy > 0) yPosLo = Math.min(yPosLo, ypy);
        }}
        if (!isFinite(yPosLo)) yPosLo = 1e-15;
        var yPosHi = Math.max(yDataMax > 0 ? yDataMax : yPosLo, yPosLo * 1.001);
        yaxisLayout.type = "log";
        yaxisLayout.exponentformat = "power";
        yaxisLayout.range = [Math.log10(yPosLo * 0.98), Math.log10(yPosHi * 1.02)];
      }} else {{
        yaxisLayout.range = [yLo, yHi];
      }}
      var layout = {{
        title: {{ text: (sorted ? "E[rank]" : "E[X]") + (scaled ? "/n" : "") + " vs " + xTitle + axisNote + " (" + titleSuffix + ")" }},
        xaxis: xaxisLayout,
        yaxis: yaxisLayout,
        margin: {{ t: 50, r: 40, b: 50, l: 60 }},
        showlegend: showLegend,
        shapes: shapes
      }};
      Plotly.react("graph", traces, layout);
    }}

    function syncNControls() {{
      if (document.getElementById("mode").value !== "vp") return;
      var wrap = document.getElementById("ctrl-n-range");
      wrap.style.display = "flex";
      var m = document.getElementById("multiple").checked;
      if (m) wrap.classList.remove("obd-dual-single-mode");
      else wrap.classList.add("obd-dual-single-mode");
    }}

    function syncPControls() {{
      if (document.getElementById("mode").value !== "vn") return;
      var wrap = document.getElementById("ctrl-p-range");
      if (!wrap) return;
      wrap.style.display = "flex";
      var m = document.getElementById("multiple").checked;
      if (m) wrap.classList.remove("obd-dual-single-mode");
      else wrap.classList.add("obd-dual-single-mode");
    }}

    /** Swap points apply only in E vs P (single or multiple n). */
    function syncSwapPointsControl() {{
      var mode = document.getElementById("mode").value;
      var relevant = mode === "vp";
      var swapEl = document.getElementById("swap-points");
      var swapLab = document.getElementById("swap-points-label");
      if (!swapEl || !swapLab) return;
      if (!relevant) {{
        swapEl.disabled = true;
        swapLab.classList.add("swap-points-disabled");
      }} else {{
        swapEl.disabled = false;
        swapLab.classList.remove("swap-points-disabled");
      }}
    }}

    function setMode(mode) {{
      document.getElementById("ctrl-n").style.display = mode === "vp" ? "flex" : "none";
      document.getElementById("ctrl-p").style.display = mode === "vn" ? "flex" : "none";
      var mult = document.getElementById("multiple");
      var multLab = document.getElementById("multiple-label");
      mult.disabled = false;
      multLab.classList.remove("multiple-disabled");
      syncSwapPointsControl();
      syncNControls();
      syncPControls();
      syncEveryControl();
      if (mode === "vp") {{
        syncObdDualRangeFromGlobals();
        updateNRangeSideLabel();
      }}
      if (mode === "vn") {{
        syncObdPDualRangeFromGlobals();
        updatePRangeSideLabel();
      }}
      updateGraph();
    }}

    document.getElementById("mode").addEventListener("change", function() {{
      setMode(this.value);
    }});
    document.getElementById("n-range-low").addEventListener("input", onObdDualRangeLowInput);
    document.getElementById("n-range-high").addEventListener("input", onObdDualRangeHighInput);
    document.getElementById("multiple").addEventListener("change", function() {{
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!this.checked && loIn && hiIn) {{
        var lo = parseInt(loIn.value, 10);
        var nSel = lo;
        nRangeLo = nSel;
        nRangeHi = nSel;
        loIn.value = String(nSel);
        hiIn.value = String(nSel);
      }}
      var pLoIn = document.getElementById("p-range-low");
      var pHiIn = document.getElementById("p-range-high");
      if (!this.checked && pLoIn && pHiIn) {{
        var pSel = parseInt(pLoIn.value, 10);
        pRangeLo = pSel;
        pRangeHi = pSel;
        pLoIn.value = String(pSel);
        pHiIn.value = String(pSel);
      }}
      syncNControls();
      syncPControls();
      updateNRangeSideLabel();
      updatePRangeSideLabel();
      syncObdDualRangeFromGlobals();
      syncObdPDualRangeFromGlobals();
      syncEveryControl();
      syncSwapPointsControl();
      updateGraph();
    }});
    document.getElementById("every-step").addEventListener("change", function() {{
      if (document.getElementById("multiple").checked)
        updateGraph();
    }});
    document.getElementById("every-step").addEventListener("input", function() {{
      if (document.getElementById("multiple").checked)
        updateGraph();
    }});
    document.getElementById("p-range-low").addEventListener("input", onObdPDualRangeLowInput);
    document.getElementById("p-range-high").addEventListener("input", onObdPDualRangeHighInput);
    document.getElementById("sorted").addEventListener("change", updateGraph);
    document.getElementById("scaled").addEventListener("change", updateGraph);
    document.getElementById("log-x").addEventListener("change", updateGraph);
    document.getElementById("log-y").addEventListener("change", updateGraph);
    document.getElementById("swap-points").addEventListener("change", updateGraph);

    syncObdDualRangeFromGlobals();
    syncObdPDualRangeFromGlobals();
    updateNRangeSideLabel();
    updatePRangeSideLabel();
    setMode(document.getElementById("mode").value);
  </script>
</body>
</html>
"""


def main(
    output_path: str = "OBDgraphExplorer1.html",
    graph_data_path: str = GRAPH_DATA_PATH,
    tie_points_path: str = TIE_POINTS_PATH,
) -> None:
    print(f"Loading graph data from {graph_data_path}...")
    binomial_data = _load_binomial_series_from_pickle(graph_data_path)
    print(f"  Loaded binomial series ({len(binomial_data)} points).")
    print(f"Loading tie points from {tie_points_path}...")
    tie_points_by_n = _load_tie_points(tie_points_path)
    if tie_points_by_n:
        print(
            f"  Loaded tie points for n in {{{', '.join(str(n) for n in sorted(tie_points_by_n)[:5])}}}..."
        )
    elif os.path.isfile(tie_points_path):
        print("  File found but no tie points in (0.5,1) for explorer n; swap points may have no effect.")
    html = _build_html(binomial_data, tie_points_by_n)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {output_path}. Open in a browser (no server needed).")


if __name__ == "__main__":
    main()
