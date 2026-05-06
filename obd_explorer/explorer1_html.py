"""Plotly explorer #1 HTML template (embedded data)."""

from __future__ import annotations

import base64
import json
from matplotlib import colormaps
from matplotlib.colors import to_hex
import numpy as np


def _pack_float32_base64(values: list[float] | tuple[float, ...]) -> str:
    if not values:
        return ""
    arr = np.asarray(values, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _pack_uint16_base64(values: list[int] | tuple[int, ...]) -> str:
    if not values:
        return ""
    arr = np.asarray(values, dtype=np.uint16)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _pack_binomial_payload_base64_by_n(
    binomial_data: list,
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
) -> dict[str, object]:
    packed_by_n: dict[str, dict[str, object]] = {}
    for n in range(n_min, n_max + 1):
        row_len = n + 1
        y_flat: list[float] = []
        perm_flat: list[int] = []
        base_idx = (n - n_min) * p_steps
        for p_idx in range(p_steps):
            pt = binomial_data[base_idx + p_idx]
            y_vals = pt.get("y") if isinstance(pt, dict) else None
            perm_vals = pt.get("perm") if isinstance(pt, dict) else None
            if not isinstance(y_vals, list) or not isinstance(perm_vals, list):
                raise ValueError(f"Invalid binomial row for n={n}, p_idx={p_idx}")
            if len(y_vals) != row_len or len(perm_vals) != row_len:
                raise ValueError(
                    f"Unexpected row length for n={n}, p_idx={p_idx}: "
                    f"got y={len(y_vals)} perm={len(perm_vals)} expected={row_len}"
                )
            y_flat.extend(float(v) for v in y_vals)
            perm_flat.extend(int(v) for v in perm_vals)
        packed_by_n[str(n)] = {
            "row_len": row_len,
            "y_f32_b64": _pack_float32_base64(y_flat),
            "perm_u16_b64": _pack_uint16_base64(perm_flat),
        }
    return {
        "n_min": n_min,
        "n_max": n_max,
        "p_steps": p_steps,
        "by_n": packed_by_n,
    }


def _pack_tie_points_base64_by_n(tie_points_by_n: dict) -> dict[str, str]:
    packed: dict[str, str] = {}
    for n_key, vals in tie_points_by_n.items():
        if not isinstance(vals, list):
            continue
        packed[str(n_key)] = _pack_float32_base64([float(v) for v in vals])
    return packed


def build_explorer1_html(
    binomial_data: list,
    tie_points_by_n: dict,
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
    p_values: list[float],
    include_tie_points: bool = True,
    colorscale: str = "viridis",
) -> str:
    binomial_packed_json = json.dumps(
        _pack_binomial_payload_base64_by_n(
            binomial_data,
            n_min=n_min,
            n_max=n_max,
            p_steps=p_steps,
        ),
        separators=(",", ":"),
    )
    p_half_start = (p_steps - 1) // 2
    p_labels_json = json.dumps([round(p, 4) for p in p_values])
    tie_points_packed_json = json.dumps(_pack_tie_points_base64_by_n(tie_points_by_n), separators=(",", ":"))
    color_lut_json = json.dumps([to_hex(colormaps[colorscale](i / 255.0), keep_alpha=False) for i in range(256)])
    p_idx_max = p_steps - 1
    p_default_hi = min(p_half_start + 50, p_idx_max)
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
      <label for="scale-mode">Y scale</label>
      <select id="scale-mode" aria-label="Y axis scaling">
        <option value="unscaled">Unscaled</option>
        <option value="by_n" selected>Scale by n</option>
        <option value="endpoint">Endpoint scale</option>
      </select>
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
            <input type="range" id="n-range-low" min="{n_min}" max="{n_max}" step="1" value="{n_min}" aria-label="Minimum n">
            <input type="range" id="n-range-high" min="{n_min}" max="{n_max}" step="1" value="10" aria-label="Maximum n">
          </div>
          <label for="n-range-low" class="obd-n-range-label"><b>n</b> <span id="n-range-side-label">= {n_min}</span></label>
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
            <input type="range" id="p-range-low" min="{p_half_start}" max="{p_idx_max}" step="1" value="{p_half_start}" aria-label="Minimum p index">
            <input type="range" id="p-range-high" min="{p_half_start}" max="{p_idx_max}" step="1" value="{p_default_hi}" aria-label="Maximum p index">
          </div>
          <label for="p-range-low" class="obd-p-range-label"><b>p</b> <span id="p-range-side-label">= {p_values[p_half_start]:.3f}</span></label>
        </div>
      </div>
    </div>
  </div>

  <script>
    const BINOMIAL_PACKED = {binomial_packed_json};
    const P_LABELS = {p_labels_json};
    const TIE_POINTS_PACKED = {tie_points_packed_json};
    const INCLUDE_TIE_POINTS = {"true" if include_tie_points else "false"};
    const COLOR_LUT = {color_lut_json};

    function decodeBase64Bytes(b64) {{
      if (!b64) return new Uint8Array(0);
      var raw = atob(b64);
      var bytes = new Uint8Array(raw.length);
      for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      return bytes;
    }}

    function decodeBase64Float32(b64) {{
      var bytes = decodeBase64Bytes(b64);
      return new Float32Array(bytes.buffer);
    }}

    function decodeBase64Uint16(b64) {{
      var bytes = decodeBase64Bytes(b64);
      return new Uint16Array(bytes.buffer);
    }}

    function decodeTiePointsPacked(src) {{
      var out = {{}};
      for (var nKey in src) {{
        if (!Object.prototype.hasOwnProperty.call(src, nKey)) continue;
        out[nKey] = decodeBase64Float32(src[nKey] || "");
      }}
      return out;
    }}

    const TIE_POINTS_BY_N = decodeTiePointsPacked(TIE_POINTS_PACKED);
    const BINOMIAL_CACHE_BY_N = {{}};

    function getBinomialRowData(n) {{
      if (Object.prototype.hasOwnProperty.call(BINOMIAL_CACHE_BY_N, n)) return BINOMIAL_CACHE_BY_N[n];
      var packed = BINOMIAL_PACKED.by_n[String(n)];
      if (!packed) {{
        BINOMIAL_CACHE_BY_N[n] = null;
        return null;
      }}
      var row = {{
        rowLen: packed.row_len,
        y: decodeBase64Float32(packed.y_f32_b64 || ""),
        perm: decodeBase64Uint16(packed.perm_u16_b64 || "")
      }};
      BINOMIAL_CACHE_BY_N[n] = row;
      return row;
    }}

    function expectedRankByNP(n, pIdx) {{
      var row = getBinomialRowData(n);
      if (!row) return NaN;
      var rowLen = row.rowLen;
      if (pIdx < 0 || pIdx >= BINOMIAL_PACKED.p_steps) return NaN;
      var base = pIdx * rowLen;
      var e = 0;
      for (var i = 0; i < rowLen; i++) {{
        var permIdx = row.perm[base + i];
        e += i * row.y[base + permIdx];
      }}
      return e;
    }}

    const P_HALF_START = {p_half_start};
    const P_IDX_MAX = {p_idx_max};
    var nRangeLo = {n_min}, nRangeHi = 10;
    var pRangeLo = {p_half_start}, pRangeHi = {p_default_hi};

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
      return ((v - {n_min}) / ({n_max} - {n_min})) * 100;
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

    /** Subtract secant through first/last finite (x,y); all such curves share y=0 at both ends. */
    function subtractEndpointChord(xs, ys) {{
      var n = Math.min(xs.length, ys.length);
      var i0 = -1, i1 = -1;
      for (var i = 0; i < n; i++) {{
        if (typeof ys[i] === "number" && isFinite(ys[i]) && typeof xs[i] === "number" && isFinite(xs[i])) {{
          if (i0 < 0) i0 = i;
          i1 = i;
        }}
      }}
      if (i0 < 0 || i1 <= i0) return ys.slice();
      var x0 = xs[i0], x1 = xs[i1], y0 = ys[i0], y1 = ys[i1];
      var dx = x1 - x0;
      if (dx === 0) return ys.slice();
      var out = [];
      for (var k = 0; k < n; k++) {{
        if (typeof ys[k] !== "number" || !isFinite(ys[k])) {{ out.push(ys[k]); continue; }}
        var L = y0 + (y1 - y0) * (xs[k] - x0) / dx;
        out.push(ys[k] - L);
      }}
      return out;
    }}

    function buildVpXY(n, sorted, scaleMode) {{
      var xArr = [], yArr = [];
      for (var i = P_HALF_START; i < {p_steps}; i++) {{
        var pVal = parseFloat(P_LABELS[i]);
        xArr.push(pVal);
        var e = sorted ? expectedRankByNP(n, i) : n * pVal;
        if (scaleMode === "by_n") {{
          if (!sorted) yArr.push(pVal);
          else yArr.push(e / n);
        }} else yArr.push(e);
      }}
      if (scaleMode === "endpoint") yArr = subtractEndpointChord(xArr, yArr);
      return {{ x: xArr, y: yArr }};
    }}

    function colorFromLut(t) {{
      var u = Math.max(0, Math.min(1, t));
      var idx = Math.round(u * (COLOR_LUT.length - 1));
      return COLOR_LUT[idx];
    }}

    function colorForN(n, nLo, nHi) {{
      var lo = nLo !== undefined ? nLo : {n_min};
      var hi = nHi !== undefined ? nHi : {n_max};
      var span = hi - lo;
      var t = span > 0 ? (n - lo) / span : 0;
      return colorFromLut(t);
    }}

    function colorForPIdx(pIdx, pLo, pHi) {{
      var lo = pLo !== undefined ? pLo : P_HALF_START;
      var hi = pHi !== undefined ? pHi : P_IDX_MAX;
      var span = hi - lo;
      var t = span > 0 ? (pIdx - lo) / span : 0;
      return colorFromLut(t);
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
      const sorted = true;
      const scaleMode = document.getElementById("scale-mode").value;
      const logXAxis = document.getElementById("log-x").checked;
      const logYAxis = document.getElementById("log-y").checked;
      const swapPoints = document.getElementById("swap-points").checked;
      const multiple = mode === "vp" && document.getElementById("multiple").checked;
      const multipleVn = mode === "vn" && document.getElementById("multiple").checked;
      var xArr, yArr, xTitle, xRange, titleSuffix, traces, showLegend;
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
            var xy = buildVpXY(nv, sorted, scaleMode);
            for (var j = 0; j < xy.y.length; j++) yArr.push(xy.y[j]);
            traces.push({{ x: xy.x, y: xy.y, mode: "lines", name: "n=" + nv, showlegend: false, line: {{ color: colorForN(nv, nLo, nHi), width: 1.5 }} }});
          }}
          showLegend = false;
        }} else {{
          const n = parseInt(document.getElementById("n-range-low").value, 10);
          var one = buildVpXY(n, sorted, scaleMode);
          xArr = one.x;
          yArr = one.y;
          titleSuffix = "n=" + n;
          traces = [{{ x: xArr, y: yArr, mode: "lines", line: {{ color: "blue", width: 1.5 }}, showlegend: false }}];
          showLegend = false;
        }}
      }} else {{
        xTitle = "n";
        if (!logXAxis) xRange = [{n_min}, {n_max}];
        if (multipleVn) {{
          var pLoI = Math.min(pRangeLo, pRangeHi), pHiI = Math.max(pRangeLo, pRangeHi);
          var strideVn = getMultipleStride();
          titleSuffix = "p=" + parseFloat(P_LABELS[pLoI]).toFixed(3) + "–" + parseFloat(P_LABELS[pHiI]).toFixed(3) + (strideVn > 1 ? ", every " + strideVn : "");
          traces = [];
          yArr = [];
          for (var pIx = pLoI; pIx <= pHiI; pIx += strideVn) {{
            var pValM = parseFloat(P_LABELS[pIx]);
            var xPart = [], yPart = [];
            for (var nVal = {n_min}; nVal <= {n_max}; nVal++) {{
              var eM = sorted ? expectedRankByNP(nVal, pIx) : nVal * pValM;
              if (!Number.isFinite(eM) && sorted) {{ yPart.push(NaN); xPart.push(nVal); continue; }}
              xPart.push(nVal);
              if (scaleMode === "by_n") {{
                if (!sorted) yPart.push(pValM);
                else yPart.push(eM / nVal);
              }} else yPart.push(eM);
            }}
            if (scaleMode === "endpoint") yPart = subtractEndpointChord(xPart, yPart);
            for (var jm = 0; jm < yPart.length; jm++) yArr.push(yPart[jm]);
            traces.push({{ x: xPart, y: yPart, mode: "lines", name: "p=" + pValM.toFixed(3), showlegend: false, line: {{ color: colorForPIdx(pIx, pLoI, pHiI), width: 1.5 }} }});
          }}
          showLegend = false;
        }} else {{
          const pIdx = parseInt(document.getElementById("p-range-low").value, 10);
          const pVal = parseFloat(P_LABELS[pIdx]);
          xArr = [];
          yArr = [];
          for (var nVal = {n_min}; nVal <= {n_max}; nVal++) {{
            var e = sorted ? expectedRankByNP(nVal, pIdx) : nVal * pVal;
            if (!Number.isFinite(e) && sorted) {{ yArr.push(NaN); xArr.push(nVal); continue; }}
            xArr.push(nVal);
            if (scaleMode === "by_n") {{
              if (!sorted) yArr.push(pVal);
              else yArr.push(e / nVal);
            }} else yArr.push(e);
          }}
          if (scaleMode === "endpoint") yArr = subtractEndpointChord(xArr, yArr);
          titleSuffix = "p=" + pVal.toFixed(3);
          traces = [{{ x: xArr, y: yArr, mode: "lines", line: {{ color: "blue", width: 1.5 }}, showlegend: false }}];
          showLegend = false;
        }}
      }}
      var yLabel;
      if (scaleMode === "by_n") yLabel = "E/n";
      else if (scaleMode === "endpoint") yLabel = (sorted ? "E[rank]" : "E[X]") + " \u2212 chord";
      else yLabel = sorted ? "E[rank]" : "E[X]";
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
      if (mode === "vp" && swapPoints && !multiple) {{
        var tiePts = TIE_POINTS_BY_N[parseInt(document.getElementById("n-range-low").value, 10)];
        if (tiePts && tiePts.length) {{
          for (var i = 0; i < tiePts.length; i++) {{
            var xh = tiePts[i];
            shapes.push({{ type: "line", x0: xh, x1: xh, y0: 0, y1: 1, yref: "paper", line: {{ color: "gray", width: 0.5 }} }});
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
          xaxisLayout.range = [Math.log10({n_min} * 0.98), Math.log10({n_max} * 1.02)];
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
        title: {{ text: (sorted ? "E[rank]" : "E[X]") + (scaleMode === "by_n" ? "/n" : scaleMode === "endpoint" ? " (endpoint detrended)" : "") + " vs " + xTitle + axisNote + " (" + titleSuffix + ")" }},
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

    /** Swap points: E vs P and single n only (not vn, not multiple). */
    function syncSwapPointsControl() {{
      var mode = document.getElementById("mode").value;
      var mult = document.getElementById("multiple").checked;
      var allowed = INCLUDE_TIE_POINTS && mode === "vp" && !mult;
      var swapEl = document.getElementById("swap-points");
      var swapLab = document.getElementById("swap-points-label");
      if (!swapEl || !swapLab) return;
      if (!allowed) {{
        swapEl.disabled = true;
        swapLab.classList.add("swap-points-disabled");
        swapEl.checked = false;
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
    document.getElementById("scale-mode").addEventListener("change", updateGraph);
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
