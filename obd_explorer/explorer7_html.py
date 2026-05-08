"""Plotly explorer #7 HTML template (nearest tie values on configurable p-grid)."""

from __future__ import annotations

import json
from matplotlib import colormaps
from matplotlib.colors import to_hex


def build_explorer7_html(
    tie_proxy_by_field_packed: dict[str, dict[str, str]],
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
    p_values: list[float],
    colorscale: str = "viridis",
) -> str:
    p_labels_json = json.dumps([round(float(p), 6) for p in p_values])
    tie_json = json.dumps(tie_proxy_by_field_packed, separators=(",", ":"))
    color_lut_json = json.dumps([to_hex(colormaps[colorscale](i / 255.0), keep_alpha=False) for i in range(256)])
    p_idx_max = p_steps - 1
    p_default_hi = min(50, p_idx_max)
    n_default_hi = max(n_min, min(10, n_max))
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

    /* Dual n/p range slider style (same interaction model as variant 1). */
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
    #ctrl-n, #ctrl-p {{ align-items: center; flex-wrap: nowrap; min-width: 0; flex: 0 0 auto; max-width: none; }}
    #ctrl-n-range, #ctrl-p-range {{
      display: none;
      flex-direction: row;
      align-items: center;
      gap: 10px;
      overflow: visible;
      flex-shrink: 0;
      max-width: none;
    }}
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
        <option value="vp">Value vs P</option>
        <option value="vn">Value vs N</option>
      </select>
      <label for="field">Field:</label>
      <select id="field">
        <option value="i">i</option>
        <option value="j">j</option>
        <option value="l">l</option>
        <option value="r">r</option>
        <option value="d" selected>d (r-l)</option>
        <option value="e">e (l-r)</option>
      </select>
      <label><input type="checkbox" id="log-x"> Log x</label>
      <label><input type="checkbox" id="log-y"> Log y</label>
      <label><input type="checkbox" id="multiple"> Multiple</label>
      <label id="every-label" for="every-step">Every... <input type="number" id="every-step" min="1" step="1" value="1" aria-label="Plot every nth curve when Multiple is on"></label>

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
            <input type="range" id="n-range-low" min="{n_min}" max="{n_max}" step="1" value="{n_min}">
            <input type="range" id="n-range-high" min="{n_min}" max="{n_max}" step="1" value="{n_default_hi}">
          </div>
          <label for="n-range-low" class="obd-n-range-label"><b>n</b> <span id="n-range-side-label">= {n_min}</span></label>
        </div>
      </div>

      <div id="ctrl-p" class="ctrl-panel" style="display:none;">
        <div id="ctrl-p-range">
          <div id="p-dual-range" class="obd-dual-slider" aria-label="p range">
            <div class="obd-dual-slider-trackwrap">
              <div class="obd-dual-inverse-left"></div>
              <div class="obd-dual-inverse-right"></div>
              <div class="obd-dual-range"></div>
              <span class="obd-dual-thumb obd-dual-thumb-left" aria-hidden="true"></span>
              <span class="obd-dual-thumb obd-dual-thumb-right" aria-hidden="true"></span>
            </div>
            <input type="range" id="p-range-low" min="0" max="{p_idx_max}" step="1" value="0">
            <input type="range" id="p-range-high" min="0" max="{p_idx_max}" step="1" value="{p_default_hi}">
          </div>
          <label for="p-range-low" class="obd-p-range-label"><b>p</b> <span id="p-range-side-label">= {p_values[0]:.3f}</span></label>
        </div>
      </div>
    </div>
  </div>

  <script>
    const TIE_PROXY_BY_FIELD_PACKED = {tie_json};
    const P_LABELS = {p_labels_json};
    const COLOR_LUT = {color_lut_json};
    const P_IDX_MAX = {p_idx_max};
    const P_MIN = parseFloat(P_LABELS[0]);
    const P_MAX = parseFloat(P_LABELS[P_LABELS.length - 1]);
    var nRangeLo = {n_min}, nRangeHi = {n_default_hi};
    var pRangeLo = 0, pRangeHi = {p_default_hi};

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
    const TIE_PROXY_CACHE_BY_FIELD = {{}};
    function getTieProxyRow(field, n) {{
      if (!Object.prototype.hasOwnProperty.call(TIE_PROXY_CACHE_BY_FIELD, field)) {{
        var src = TIE_PROXY_BY_FIELD_PACKED[field] || {{}};
        var dec = {{}};
        for (var nKey in src) {{
          if (!Object.prototype.hasOwnProperty.call(src, nKey)) continue;
          dec[nKey] = decodeBase64Float32(src[nKey] || "");
        }}
        TIE_PROXY_CACHE_BY_FIELD[field] = dec;
      }}
      return TIE_PROXY_CACHE_BY_FIELD[field][String(n)] || null;
    }}

    function fieldDisplay(field) {{
      if (field === "d") return "d (r-l)";
      if (field === "e") return "e (l-r)";
      return field;
    }}
    function valueByNP(n, pIdx, field) {{
      var row = getTieProxyRow(field, n);
      if (!row || pIdx < 0 || pIdx >= row.length) return NaN;
      var v = row[pIdx];
      return Number.isFinite(v) ? v : NaN;
    }}
    function colorFromLut(t) {{
      var u = Math.max(0, Math.min(1, t));
      var idx = Math.round(u * (COLOR_LUT.length - 1));
      return COLOR_LUT[idx];
    }}
    function getStride() {{
      var el = document.getElementById("every-step");
      var v = el ? parseInt(el.value, 10) : 1;
      return (isFinite(v) && v >= 1) ? v : 1;
    }}
    function syncEveryControl() {{
      var mult = document.getElementById("multiple").checked;
      var inp = document.getElementById("every-step");
      var lab = document.getElementById("every-label");
      if (!inp || !lab) return;
      inp.disabled = !mult;
      if (mult) lab.classList.remove("every-step-disabled");
      else lab.classList.add("every-step-disabled");
    }}

    function updateNRangeSideLabel() {{
      var el = document.getElementById("n-range-side-label");
      if (!el || document.getElementById("mode").value !== "vp") return;
      var mult = document.getElementById("multiple").checked;
      if (mult) {{
        var a = Math.min(nRangeLo, nRangeHi), b = Math.max(nRangeLo, nRangeHi);
        el.textContent = "= " + a + "-" + b;
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
        el.textContent = "= " + pFmt(a) + "-" + pFmt(b);
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
      var pLo = obdDualRangePct(lo), pHi = obdDualRangePct(hi);
      var invL = wrap.querySelector(".obd-dual-inverse-left");
      var invR = wrap.querySelector(".obd-dual-inverse-right");
      var rangeEl = wrap.querySelector(".obd-dual-range");
      var thL = wrap.querySelector(".obd-dual-thumb-left");
      var thR = wrap.querySelector(".obd-dual-thumb-right");
      if (invL) invL.style.width = pLo + "%";
      if (invR) invR.style.width = (100 - pHi) + "%";
      if (rangeEl) {{ rangeEl.style.left = pLo + "%"; rangeEl.style.right = (100 - pHi) + "%"; }}
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
        loIn.value = String(a); hiIn.value = String(b);
      }} else {{
        var n = Math.min(nRangeLo, nRangeHi);
        nRangeLo = n; nRangeHi = n;
        loIn.value = String(n); hiIn.value = String(n);
      }}
      applyObdDualRangeVisual();
    }}
    function onObdDualRangeLowInput() {{
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10), hi = parseInt(hiIn.value, 10);
      var mult = document.getElementById("multiple").checked;
      if (!mult) {{ hi = lo; hiIn.value = String(hi); }}
      else if (lo > hi) {{ lo = hi; loIn.value = String(lo); }}
      nRangeLo = lo; nRangeHi = hi;
      updateNRangeSideLabel();
      applyObdDualRangeVisual();
      if (document.getElementById("mode").value === "vp") updateGraph();
    }}
    function onObdDualRangeHighInput() {{
      if (!document.getElementById("multiple").checked) return;
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10), hi = parseInt(hiIn.value, 10);
      if (hi < lo) {{ hi = lo; hiIn.value = String(hi); }}
      nRangeLo = lo; nRangeHi = hi;
      updateNRangeSideLabel();
      applyObdDualRangeVisual();
      if (document.getElementById("mode").value === "vp") updateGraph();
    }}

    function obdPDualRangePct(idx) {{
      var span = P_IDX_MAX;
      if (span <= 0) return 0;
      return (idx / span) * 100;
    }}
    function applyObdPDualRangeVisual() {{
      var wrap = document.querySelector("#p-dual-range .obd-dual-slider-trackwrap");
      if (!wrap) return;
      var loIn = document.getElementById("p-range-low");
      var hiIn = document.getElementById("p-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10), hi = parseInt(hiIn.value, 10);
      var pLo = obdPDualRangePct(lo), pHi = obdPDualRangePct(hi);
      var invL = wrap.querySelector(".obd-dual-inverse-left");
      var invR = wrap.querySelector(".obd-dual-inverse-right");
      var rangeEl = wrap.querySelector(".obd-dual-range");
      var thL = wrap.querySelector(".obd-dual-thumb-left");
      var thR = wrap.querySelector(".obd-dual-thumb-right");
      if (invL) invL.style.width = pLo + "%";
      if (invR) invR.style.width = (100 - pHi) + "%";
      if (rangeEl) {{ rangeEl.style.left = pLo + "%"; rangeEl.style.right = (100 - pHi) + "%"; }}
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
        loIn.value = String(a); hiIn.value = String(b);
      }} else {{
        var p = Math.min(pRangeLo, pRangeHi);
        pRangeLo = p; pRangeHi = p;
        loIn.value = String(p); hiIn.value = String(p);
      }}
      applyObdPDualRangeVisual();
    }}
    function onObdPDualRangeLowInput() {{
      var loIn = document.getElementById("p-range-low");
      var hiIn = document.getElementById("p-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10), hi = parseInt(hiIn.value, 10);
      var mult = document.getElementById("multiple").checked;
      if (!mult) {{ hi = lo; hiIn.value = String(hi); }}
      else if (lo > hi) {{ lo = hi; loIn.value = String(lo); }}
      pRangeLo = lo; pRangeHi = hi;
      updatePRangeSideLabel();
      applyObdPDualRangeVisual();
      if (document.getElementById("mode").value === "vn") updateGraph();
    }}
    function onObdPDualRangeHighInput() {{
      if (!document.getElementById("multiple").checked) return;
      var loIn = document.getElementById("p-range-low");
      var hiIn = document.getElementById("p-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10), hi = parseInt(hiIn.value, 10);
      if (hi < lo) {{ hi = lo; hiIn.value = String(hi); }}
      pRangeLo = lo; pRangeHi = hi;
      updatePRangeSideLabel();
      applyObdPDualRangeVisual();
      if (document.getElementById("mode").value === "vn") updateGraph();
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
      wrap.style.display = "flex";
      var m = document.getElementById("multiple").checked;
      if (m) wrap.classList.remove("obd-dual-single-mode");
      else wrap.classList.add("obd-dual-single-mode");
    }}

    function updateGraph() {{
      const mode = document.getElementById("mode").value;
      const field = document.getElementById("field").value;
      const logX = document.getElementById("log-x").checked;
      const logY = document.getElementById("log-y").checked;
      const multiple = document.getElementById("multiple").checked;
      const stride = getStride();
      var traces = [], yAll = [], xTitle = mode === "vp" ? "p" : "n", titleSuffix = "";

      if (mode === "vp") {{
        var nLo = Math.min(nRangeLo, nRangeHi);
        var nHi = Math.max(nRangeLo, nRangeHi);
        if (!multiple) {{ nLo = nHi = Math.min(nRangeLo, nRangeHi); }}
        for (var n = nLo; n <= nHi; n += stride) {{
          var xs = [], ys = [];
          for (var pIdx = 0; pIdx <= P_IDX_MAX; pIdx++) {{
            var x = parseFloat(P_LABELS[pIdx]);
            var v = valueByNP(n, pIdx, field);
            xs.push(x);
            ys.push(v);
          }}
          for (var yi = 0; yi < ys.length; yi++) yAll.push(ys[yi]);
          var lineColorVp = multiple ? colorFromLut((n - nLo) / Math.max(1, nHi - nLo)) : "blue";
          traces.push({{
            x: xs,
            y: ys,
            mode: "lines",
            type: "scatter",
            showlegend: false,
            name: "n=" + n,
            line: {{ color: lineColorVp, width: 1.5 }}
          }});
        }}
        titleSuffix = "n=" + nLo + (multiple ? "–" + nHi : "");
      }} else {{
        var pLo = parseInt(document.getElementById("p-range-low").value, 10);
        var pHi = parseInt(document.getElementById("p-range-high").value, 10);
        if (pLo > pHi) {{ var t2 = pLo; pLo = pHi; pHi = t2; }}
        if (!multiple) pHi = pLo;
        pRangeLo = pLo; pRangeHi = pHi;
        for (var pIx = pLo; pIx <= pHi; pIx += stride) {{
          var xs2 = [], ys2 = [];
          for (var nv = {n_min}; nv <= {n_max}; nv++) {{
            xs2.push(nv);
            var vv = valueByNP(nv, pIx, field);
            ys2.push(vv);
          }}
          for (var yj = 0; yj < ys2.length; yj++) yAll.push(ys2[yj]);
          var lineColorVn = multiple ? colorFromLut((pIx - pLo) / Math.max(1, pHi - pLo)) : "blue";
          traces.push({{
            x: xs2,
            y: ys2,
            mode: "lines",
            type: "scatter",
            showlegend: false,
            name: "p=" + parseFloat(P_LABELS[pIx]).toFixed(4),
            line: {{ color: lineColorVn, width: 1.5 }}
          }});
        }}
        titleSuffix = "p=" + parseFloat(P_LABELS[pLo]).toFixed(4) + (multiple ? "–" + parseFloat(P_LABELS[pHi]).toFixed(4) : "");
      }}

      var yMin = Infinity, yMax = -Infinity;
      for (var k = 0; k < yAll.length; k++) {{
        var yv = yAll[k];
        if (typeof yv === "number" && isFinite(yv)) {{
          if (yv < yMin) yMin = yv;
          if (yv > yMax) yMax = yv;
        }}
      }}
      if (!isFinite(yMin)) {{ yMin = 0; yMax = 1; }}
      var ySpan = yMax - yMin;
      if (ySpan <= 0) ySpan = 1;
      var yPad = Math.max(ySpan * 0.05, 1e-12);
      var yLabel = fieldDisplay(field);

      var xaxis = {{ title: xTitle }};
      if (logX) {{
        xaxis.type = "log";
        if (mode === "vp") xaxis.range = [Math.log10(P_MIN), Math.log10(P_MAX)];
        else xaxis.range = [Math.log10({n_min}), Math.log10({n_max})];
      }} else {{
        if (mode === "vp") xaxis.range = [P_MIN, P_MAX];
        else xaxis.range = [{n_min}, {n_max}];
      }}
      var yaxis = {{ title: yLabel }};
      if (logY) {{
        var yPosLo = Infinity, yPosHi = 0;
        for (var m = 0; m < yAll.length; m++) {{
          var yp = yAll[m];
          if (typeof yp === "number" && isFinite(yp) && yp > 0) {{
            yPosLo = Math.min(yPosLo, yp);
            yPosHi = Math.max(yPosHi, yp);
          }}
        }}
        if (!isFinite(yPosLo)) yPosLo = 1e-12;
        if (!(yPosHi > yPosLo)) yPosHi = yPosLo * 1.001;
        yaxis.type = "log";
        yaxis.range = [Math.log10(yPosLo * 0.98), Math.log10(yPosHi * 1.02)];
      }} else {{
        yaxis.range = [yMin - yPad, yMax + yPad];
      }}
      var title = "OBD Nearest Tie Graph " + yLabel + " vs " + xTitle + " (" + titleSuffix + ")";
      Plotly.react("graph", traces, {{
        title: {{ text: title }},
        xaxis: xaxis,
        yaxis: yaxis,
        margin: {{ t: 50, r: 40, b: 50, l: 60 }},
        showlegend: false
      }});
    }}

    function setMode(mode) {{
      document.getElementById("ctrl-n").style.display = mode === "vp" ? "flex" : "none";
      document.getElementById("ctrl-p").style.display = mode === "vn" ? "flex" : "none";
      syncNControls();
      syncPControls();
      updateNRangeSideLabel();
      updatePRangeSideLabel();
      syncObdDualRangeFromGlobals();
      syncObdPDualRangeFromGlobals();
    }}

    document.getElementById("mode").addEventListener("change", function() {{ setMode(this.value); updateGraph(); }});
    document.getElementById("field").addEventListener("change", updateGraph);
    document.getElementById("log-x").addEventListener("change", updateGraph);
    document.getElementById("log-y").addEventListener("change", updateGraph);
    document.getElementById("multiple").addEventListener("change", function() {{
      syncEveryControl();
      syncNControls();
      syncPControls();
      syncObdDualRangeFromGlobals();
      syncObdPDualRangeFromGlobals();
      updateNRangeSideLabel();
      updatePRangeSideLabel();
      updateGraph();
    }});
    document.getElementById("every-step").addEventListener("input", updateGraph);

    document.getElementById("n-range-low").addEventListener("input", onObdDualRangeLowInput);
    document.getElementById("n-range-high").addEventListener("input", onObdDualRangeHighInput);
    document.getElementById("p-range-low").addEventListener("input", onObdPDualRangeLowInput);
    document.getElementById("p-range-high").addEventListener("input", onObdPDualRangeHighInput);

    syncEveryControl();
    setMode(document.getElementById("mode").value);
    updateGraph();
  </script>
</body>
</html>
"""

