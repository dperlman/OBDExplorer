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
      flex-shrink: 0; display: flex; flex-wrap: nowrap; justify-content: flex-start; align-items: center;
      gap: 12px; padding: 10px 12px 12px; overflow-x: auto; overflow-y: hidden; white-space: nowrap;
    }}
    .controls > * {{ flex: 0 0 auto; }}
    .controls label {{ display: inline-flex; align-items: center; gap: 6px; cursor: pointer; }}
    .ctrl-panel {{ display: flex; align-items: center; gap: 8px; }}
    .range-num {{ width: 80px; }}
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
      <label for="scale-mode">Y scale</label>
      <select id="scale-mode" aria-label="Y axis scaling">
        <option value="unscaled">Unscaled</option>
        <option value="by_n" selected>Scale by n</option>
        <option value="endpoint">Endpoint detrended</option>
      </select>
      <label><input type="checkbox" id="log-x"> Log x</label>
      <label><input type="checkbox" id="log-y"> Log y</label>
      <label><input type="checkbox" id="multiple"> Multiple</label>
      <label id="every-label" for="every-step">Every... <input type="number" id="every-step" min="1" step="1" value="1" aria-label="Plot every nth curve when Multiple is on"></label>

      <div id="ctrl-n" class="ctrl-panel">
        <label><b>n</b></label>
        <input type="range" id="n-range-low" min="{n_min}" max="{n_max}" step="1" value="{n_min}">
        <input type="range" id="n-range-high" min="{n_min}" max="{n_max}" step="1" value="{n_max}">
        <input type="number" id="n-num-low" class="range-num" min="{n_min}" max="{n_max}" step="1" value="{n_min}">
        <input type="number" id="n-num-high" class="range-num" min="{n_min}" max="{n_max}" step="1" value="{n_max}">
      </div>

      <div id="ctrl-p" class="ctrl-panel" style="display:none;">
        <label><b>p</b></label>
        <input type="range" id="p-range-low" min="0" max="{p_idx_max}" step="1" value="0">
        <input type="range" id="p-range-high" min="0" max="{p_idx_max}" step="1" value="{p_default_hi}">
        <input type="number" id="p-num-low" class="range-num" min="0" max="{p_idx_max}" step="1" value="0">
        <input type="number" id="p-num-high" class="range-num" min="0" max="{p_idx_max}" step="1" value="{p_default_hi}">
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
      var x0 = xs[i0], x1 = xs[i1], y0 = ys[i0], y1 = ys[i1], dx = x1 - x0;
      if (dx === 0) return ys.slice();
      var out = [];
      for (var k = 0; k < n; k++) {{
        if (typeof ys[k] !== "number" || !isFinite(ys[k])) {{ out.push(ys[k]); continue; }}
        var L = y0 + (y1 - y0) * (xs[k] - x0) / dx;
        out.push(ys[k] - L);
      }}
      return out;
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
    function syncRangeNums(prefix) {{
      var loR = document.getElementById(prefix + "-range-low");
      var hiR = document.getElementById(prefix + "-range-high");
      var loN = document.getElementById(prefix + "-num-low");
      var hiN = document.getElementById(prefix + "-num-high");
      if (!loR || !hiR || !loN || !hiN) return;
      loN.value = loR.value; hiN.value = hiR.value;
    }}
    function syncRangesFromNums(prefix) {{
      var loR = document.getElementById(prefix + "-range-low");
      var hiR = document.getElementById(prefix + "-range-high");
      var loN = document.getElementById(prefix + "-num-low");
      var hiN = document.getElementById(prefix + "-num-high");
      if (!loR || !hiR || !loN || !hiN) return;
      var lo = parseInt(loN.value, 10), hi = parseInt(hiN.value, 10);
      if (!isFinite(lo)) lo = parseInt(loR.min, 10);
      if (!isFinite(hi)) hi = parseInt(hiR.max, 10);
      if (lo > hi) hi = lo;
      loR.value = String(lo); hiR.value = String(hi);
    }}

    function updateGraph() {{
      const mode = document.getElementById("mode").value;
      const field = document.getElementById("field").value;
      const scaleMode = document.getElementById("scale-mode").value;
      const logX = document.getElementById("log-x").checked;
      const logY = document.getElementById("log-y").checked;
      const multiple = document.getElementById("multiple").checked;
      const stride = getStride();
      var traces = [], yAll = [], xTitle = mode === "vp" ? "p" : "n", titleSuffix = "";

      if (mode === "vp") {{
        var nLo = parseInt(document.getElementById("n-range-low").value, 10);
        var nHi = parseInt(document.getElementById("n-range-high").value, 10);
        if (nLo > nHi) {{ var tmp = nLo; nLo = nHi; nHi = tmp; }}
        if (!multiple) nHi = nLo;
        for (var n = nLo; n <= nHi; n += stride) {{
          var xs = [], ys = [];
          for (var pIdx = 0; pIdx <= P_IDX_MAX; pIdx++) {{
            var x = parseFloat(P_LABELS[pIdx]);
            var v = valueByNP(n, pIdx, field);
            xs.push(x);
            if (scaleMode === "by_n") ys.push(v / n);
            else ys.push(v);
          }}
          if (scaleMode === "endpoint") ys = subtractEndpointChord(xs, ys);
          for (var yi = 0; yi < ys.length; yi++) yAll.push(ys[yi]);
          traces.push({{
            x: xs,
            y: ys,
            mode: "lines",
            type: "scatter",
            showlegend: false,
            name: "n=" + n,
            line: {{ color: colorFromLut((n - nLo) / Math.max(1, nHi - nLo)), width: 1.5 }}
          }});
        }}
        titleSuffix = "n=" + nLo + (multiple ? "–" + nHi : "");
      }} else {{
        var pLo = parseInt(document.getElementById("p-range-low").value, 10);
        var pHi = parseInt(document.getElementById("p-range-high").value, 10);
        if (pLo > pHi) {{ var t2 = pLo; pLo = pHi; pHi = t2; }}
        if (!multiple) pHi = pLo;
        for (var pIx = pLo; pIx <= pHi; pIx += stride) {{
          var xs2 = [], ys2 = [];
          for (var nv = {n_min}; nv <= {n_max}; nv++) {{
            xs2.push(nv);
            var vv = valueByNP(nv, pIx, field);
            if (scaleMode === "by_n") ys2.push(vv / nv);
            else ys2.push(vv);
          }}
          if (scaleMode === "endpoint") ys2 = subtractEndpointChord(xs2, ys2);
          for (var yj = 0; yj < ys2.length; yj++) yAll.push(ys2[yj]);
          traces.push({{
            x: xs2,
            y: ys2,
            mode: "lines",
            type: "scatter",
            showlegend: false,
            name: "p=" + parseFloat(P_LABELS[pIx]).toFixed(4),
            line: {{ color: colorFromLut((pIx - pLo) / Math.max(1, pHi - pLo)), width: 1.5 }}
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
      if (scaleMode === "by_n") yLabel += "/n";
      if (scaleMode === "endpoint") yLabel += " (endpoint detrended)";

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
    }}

    document.getElementById("mode").addEventListener("change", function() {{ setMode(this.value); updateGraph(); }});
    document.getElementById("field").addEventListener("change", updateGraph);
    document.getElementById("scale-mode").addEventListener("change", updateGraph);
    document.getElementById("log-x").addEventListener("change", updateGraph);
    document.getElementById("log-y").addEventListener("change", updateGraph);
    document.getElementById("multiple").addEventListener("change", function() {{ syncEveryControl(); updateGraph(); }});
    document.getElementById("every-step").addEventListener("input", updateGraph);

    ["n-range-low","n-range-high"].forEach(function(id) {{
      document.getElementById(id).addEventListener("input", function() {{ syncRangeNums("n"); updateGraph(); }});
    }});
    ["n-num-low","n-num-high"].forEach(function(id) {{
      document.getElementById(id).addEventListener("input", function() {{ syncRangesFromNums("n"); syncRangeNums("n"); updateGraph(); }});
    }});
    ["p-range-low","p-range-high"].forEach(function(id) {{
      document.getElementById(id).addEventListener("input", function() {{ syncRangeNums("p"); updateGraph(); }});
    }});
    ["p-num-low","p-num-high"].forEach(function(id) {{
      document.getElementById(id).addEventListener("input", function() {{ syncRangesFromNums("p"); syncRangeNums("p"); updateGraph(); }});
    }});

    syncEveryControl();
    syncRangeNums("n");
    syncRangeNums("p");
    setMode(document.getElementById("mode").value);
    updateGraph();
  </script>
</body>
</html>
"""

