"""Explorer2 two-panel HTML: binomial (left), PCA (right), shared n slider below (``html_old/OBDexplorer2`` layout)."""

from __future__ import annotations

import base64
import json
import numpy as np


def _pack_float32_base64(values: object) -> str:
    if not isinstance(values, (list, tuple)) or not values:
        return ""
    arr = np.empty(len(values), dtype=np.float32)
    for idx, raw in enumerate(values):
        arr[idx] = np.float32(np.nan if raw is None else raw)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _pack_uint16_base64(values: object) -> str:
    if not isinstance(values, (list, tuple)) or not values:
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
            y_flat.extend(float(v) for v in y_vals)
            perm_flat.extend(int(v) for v in perm_vals)
        packed_by_n[str(n)] = {
            "row_len": row_len,
            "y_f32_b64": _pack_float32_base64(y_flat),
            "perm_u16_b64": _pack_uint16_base64(perm_flat),
        }
    return {"n_min": n_min, "n_max": n_max, "p_steps": p_steps, "by_n": packed_by_n}


def _pack_projection_rows_base64(rows: list) -> list[dict[str, object]]:
    packed_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            packed_rows.append({})
            continue
        out_row: dict[str, object] = {}
        for key, val in row.items():
            if isinstance(val, list):
                out_row[key] = {"_f32_b64": _pack_float32_base64(val)}
            else:
                out_row[key] = val
        packed_rows.append(out_row)
    return packed_rows


def build_explorer2_dual_panel_html_document(
    binomial_data: list,
    p_grid: list[float],
    data_full_unsorted: list,
    data_full_sorted: list,
    data_half_unsorted: list,
    data_half_sorted: list,
    data_tie_unsorted: list,
    data_tie_sorted: list,
    last_tie_by_n: dict[int, float],
    *,
    n_min: int,
    n_max: int,
    p_steps: int,
    p_default_idx: int,
) -> str:
    binomial_packed_json = json.dumps(
        _pack_binomial_payload_base64_by_n(binomial_data, n_min=n_min, n_max=n_max, p_steps=p_steps),
        separators=(",", ":"),
    )
    p_labels_json = json.dumps([round(float(p), 4) for p in p_grid])
    last_tie_json = json.dumps(last_tie_by_n)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    html, body {{ margin: 0; box-sizing: border-box; }}
    * {{ box-sizing: border-box; }}
    .row {{ display: flex; flex-wrap: wrap; margin: 10px 0; }}
    .col {{ flex: 1; min-width: 380px; padding: 0 8px; box-sizing: border-box; }}
    .col .graph {{ width: 100%; height: 480px; }}
    .controls {{ margin-top: 8px; }}
    .controls-right-wrap {{ display: flex; flex-direction: column; gap: 8px; margin-top: 8px; }}
    .controls-right-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }}
    .n-row {{ justify-content: center; margin-top: 16px; }}
    .n-row .n-wrap {{ max-width: 500px; width: 100%; margin: 0 auto; padding: 0 8px; }}
  </style>
</head>
<body>
  <div class="row">
    <div class="col">
      <div id="graph-left" class="graph"></div>
      <div class="controls">
        <label for="p-slider"><b>p</b> = <span id="p-value">0.50</span></label>
        <input type="range" id="p-slider" min="0" max="{p_steps - 1}" value="{p_default_idx}" style="width: 100%;">
      </div>
      <div class="controls" style="margin-top: 8px;">
        <label for="display-left"><b>display</b></label>
        <select id="display-left" style="margin-left: 6px;">
          <option value="unsorted">unsorted</option>
          <option value="sorted">sorted</option>
        </select>
      </div>
    </div>
    <div class="col">
      <div id="graph-right" class="graph"></div>
      <div class="controls-right-wrap">
        <div class="controls-right-row">
          <span><label for="pca-range"><b>range for PCA</b></label>
          <select id="pca-range" style="margin-left: 4px;">
            <option value="full">0-1</option>
            <option value="half" selected>0.5-1</option>
            <option value="tie">0.5-last tie</option>
          </select></span>
          <span><label for="pca-data"><b>data for PCA</b></label>
          <select id="pca-data" style="margin-left: 4px;">
            <option value="unsorted">unsorted</option>
            <option value="sorted">sorted</option>
          </select></span>
          <span><label for="pca-components"><b>display components</b></label>
          <select id="pca-components" style="margin-left: 4px;">
            <option value="12" selected>1,2</option>
            <option value="23">2,3</option>
            <option value="34">3,4</option>
            <option value="13">1,3</option>
            <option value="24">2,4</option>
            <option value="14">1,4</option>
          </select></span>
        </div>
        <div class="controls-right-row">
          <span><label for="display-right"><b>display</b></label>
          <select id="display-right" style="margin-left: 4px;">
            <option value="unsorted">unsorted</option>
            <option value="sorted">sorted</option>
          </select></span>
          <span>
          <label for="display-range" style="margin-right:4px;"><b>range</b></label>
          <select id="display-range" style="margin-left: 4px;">
            <option value="full">0–1</option>
            <option value="half" selected>0.5–1</option>
            <option value="tie">0.5–last tie</option>
          </select></span>
          <label><input type="checkbox" id="simplex-points"> simplex</label>
          <label><input type="checkbox" id="autozoom" checked> autozoom</label>
          <label><input type="checkbox" id="show-p"> Show p</label>
          <label><input type="checkbox" id="pca-connect-lines"> lines</label>
        </div>
      </div>
    </div>
  </div>
  <div class="row n-row">
    <div class="n-wrap">
      <label for="n-slider"><b>n</b> = <span id="n-value">{n_min}</span></label>
      <input type="range" id="n-slider" min="{n_min}" max="{n_max}" value="{n_min}" style="width: 100%;">
    </div>
  </div>

  <script>
    const BINOMIAL_PACKED = {binomial_packed_json};
    const P_LABELS = {p_labels_json};
    const DATA_FULL_UNSORTED_PACKED = {json.dumps(_pack_projection_rows_base64(data_full_unsorted), separators=(",", ":"))};
    const DATA_FULL_SORTED_PACKED = {json.dumps(_pack_projection_rows_base64(data_full_sorted), separators=(",", ":"))};
    const DATA_HALF_UNSORTED_PACKED = {json.dumps(_pack_projection_rows_base64(data_half_unsorted), separators=(",", ":"))};
    const DATA_HALF_SORTED_PACKED = {json.dumps(_pack_projection_rows_base64(data_half_sorted), separators=(",", ":"))};
    const DATA_TIE_UNSORTED_PACKED = {json.dumps(_pack_projection_rows_base64(data_tie_unsorted), separators=(",", ":"))};
    const DATA_TIE_SORTED_PACKED = {json.dumps(_pack_projection_rows_base64(data_tie_sorted), separators=(",", ":"))};
    const LAST_TIE_BY_N = {last_tie_json};
    const FIXED_RANGE = {{ x: [-1, 1], y: [-1, 1] }};
    const BINOMIAL_CACHE_BY_N = {{}};

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

    function decodeProjectionRowsPacked(rows) {{
      var out = [];
      for (var r = 0; r < rows.length; r++) {{
        var row = rows[r] || {{}};
        var dst = {{}};
        for (var key in row) {{
          if (!Object.prototype.hasOwnProperty.call(row, key)) continue;
          var val = row[key];
          if (val && typeof val === "object" && Object.prototype.hasOwnProperty.call(val, "_f32_b64")) dst[key] = decodeBase64Float32(val._f32_b64 || "");
          else dst[key] = val;
        }}
        out.push(dst);
      }}
      return out;
    }}

    const DATA_FULL_UNSORTED = decodeProjectionRowsPacked(DATA_FULL_UNSORTED_PACKED);
    const DATA_FULL_SORTED = decodeProjectionRowsPacked(DATA_FULL_SORTED_PACKED);
    const DATA_HALF_UNSORTED = decodeProjectionRowsPacked(DATA_HALF_UNSORTED_PACKED);
    const DATA_HALF_SORTED = decodeProjectionRowsPacked(DATA_HALF_SORTED_PACKED);
    const DATA_TIE_UNSORTED = decodeProjectionRowsPacked(DATA_TIE_UNSORTED_PACKED);
    const DATA_TIE_SORTED = decodeProjectionRowsPacked(DATA_TIE_SORTED_PACKED);

    function getDataIndex(n) {{ return n - {n_min}; }}

    function getBinomialPoint(n, pIdx) {{
      if (!Object.prototype.hasOwnProperty.call(BINOMIAL_CACHE_BY_N, n)) {{
        var row = BINOMIAL_PACKED.by_n[String(n)];
        if (!row) return null;
        BINOMIAL_CACHE_BY_N[n] = {{
          rowLen: row.row_len,
          y: decodeBase64Float32(row.y_f32_b64 || ""),
          perm: decodeBase64Uint16(row.perm_u16_b64 || "")
        }};
      }}
      var d = BINOMIAL_CACHE_BY_N[n];
      if (!d || pIdx < 0 || pIdx >= BINOMIAL_PACKED.p_steps) return null;
      var base = pIdx * d.rowLen;
      var x = Array.from({{ length: d.rowLen }}, function(_, i) {{ return i; }});
      var y = Array.from(d.y.slice(base, base + d.rowLen));
      var perm = Array.from(d.perm.slice(base, base + d.rowLen));
      return {{ x: x, y: y, perm: perm }};
    }}

    function getProjectionData() {{
      const range = document.getElementById("pca-range").value;
      const data = document.getElementById("pca-data").value;
      if (range === "full" && data === "unsorted") return DATA_FULL_UNSORTED;
      if (range === "full" && data === "sorted") return DATA_FULL_SORTED;
      if (range === "half" && data === "unsorted") return DATA_HALF_UNSORTED;
      if (range === "half" && data === "sorted") return DATA_HALF_SORTED;
      if (range === "tie" && data === "unsorted") return DATA_TIE_UNSORTED;
      return DATA_TIE_SORTED;
    }}

    function getComponentsView() {{ return document.getElementById("pca-components").value; }}

    function getAxisRangeFromXY(xy) {{
      if (!xy.x.length) return FIXED_RANGE;
      let xMin = Math.min(...xy.x), xMax = Math.max(...xy.x);
      let yMin = Math.min(...xy.y), yMax = Math.max(...xy.y);
      const padX = Math.max((xMax - xMin) * 0.05, 1e-6) || 0.01;
      const padY = Math.max((yMax - yMin) * 0.05, 1e-6) || 0.01;
      return {{ x: [xMin - padX, xMax + padX], y: [yMin - padY, yMax + padY] }};
    }}

    function getAxisRange(d, xy) {{
      if (!document.getElementById("autozoom").checked) return FIXED_RANGE;
      return getAxisRangeFromXY(xy);
    }}

    function filterByPRange(d, n, pRange) {{
      const p = d.p;
      let indices = [];
      const lastTie = LAST_TIE_BY_N[n];
      for (let i = 0; i < p.length; i++) {{
        if (pRange === "full") indices.push(i);
        else if (pRange === "half" && p[i] >= 0.5) indices.push(i);
        else if (pRange === "tie" && p[i] >= 0.5 && p[i] <= lastTie) indices.push(i);
      }}
      return {{
        unsorted_pc1: indices.map(i => d.unsorted_pc1[i]),
        unsorted_pc2: indices.map(i => d.unsorted_pc2[i]),
        unsorted_pc3: indices.map(i => d.unsorted_pc3[i]),
        unsorted_pc4: indices.map(i => d.unsorted_pc4[i]),
        ordered_pc1: indices.map(i => d.ordered_pc1[i]),
        ordered_pc2: indices.map(i => d.ordered_pc2[i]),
        ordered_pc3: indices.map(i => d.ordered_pc3[i]),
        ordered_pc4: indices.map(i => d.ordered_pc4[i]),
        p: indices.map(i => d.p[i])
      }};
    }}

    function getXYFromFiltered(filtered, displayData, componentsView) {{
      let px, py;
      const S = displayData === "sorted";
      if (componentsView === "34") {{
        px = S ? filtered.ordered_pc3 : filtered.unsorted_pc3;
        py = S ? filtered.ordered_pc4 : filtered.unsorted_pc4;
      }} else if (componentsView === "23") {{
        px = S ? filtered.ordered_pc2 : filtered.unsorted_pc2;
        py = S ? filtered.ordered_pc3 : filtered.unsorted_pc3;
      }} else if (componentsView === "13") {{
        px = S ? filtered.ordered_pc1 : filtered.unsorted_pc1;
        py = S ? filtered.ordered_pc3 : filtered.unsorted_pc3;
      }} else if (componentsView === "24") {{
        px = S ? filtered.ordered_pc2 : filtered.unsorted_pc2;
        py = S ? filtered.ordered_pc4 : filtered.unsorted_pc4;
      }} else if (componentsView === "14") {{
        px = S ? filtered.ordered_pc1 : filtered.unsorted_pc1;
        py = S ? filtered.ordered_pc4 : filtered.unsorted_pc4;
      }} else {{
        px = S ? filtered.ordered_pc1 : filtered.unsorted_pc1;
        py = S ? filtered.ordered_pc2 : filtered.unsorted_pc2;
      }}
      return {{ x: px, y: py }};
    }}

    function getVertexXY(d, componentsView) {{
      if (componentsView === "34") return {{ x: d.vertex_pc3, y: d.vertex_pc4 }};
      if (componentsView === "23") return {{ x: d.vertex_pc2, y: d.vertex_pc3 }};
      if (componentsView === "13") return {{ x: d.vertex_pc1, y: d.vertex_pc3 }};
      if (componentsView === "24") return {{ x: d.vertex_pc2, y: d.vertex_pc4 }};
      if (componentsView === "14") return {{ x: d.vertex_pc1, y: d.vertex_pc4 }};
      return {{ x: d.vertex_pc1, y: d.vertex_pc2 }};
    }}

    function colorForK(k, n) {{
      var t = n > 0 ? k / n : 0;
      return "hsl(" + Math.round(240 * (1 - t)) + ", 70%, 50%)";
    }}

    const layoutLeft = {{
      title: {{ text: "Binomial(n=2, p=0.50)" }},
      xaxis: {{ title: "k", dtick: 1, showticklabels: false }},
      yaxis: {{ title: "P(X = k)", range: [-0.15, 1] }},
      margin: {{ t: 50, r: 20, b: 45, l: 50 }},
      bargap: 0, bargroupgap: 0, showlegend: false
    }};

    const layoutRight = {{
      title: {{ text: "PCA: n=2" }},
      xaxis: {{ title: "", scaleanchor: "y", scaleratio: 1, range: [-1, 1], showgrid: true, showticklabels: false }},
      yaxis: {{ title: "", range: [-1, 1], showgrid: true, showticklabels: false }},
      margin: {{ t: 50, r: 20, b: 45, l: 50 }},
      showlegend: false
    }};

    function updateLeft() {{
      const n = parseInt(document.getElementById("n-slider").value, 10);
      const pIdx = parseInt(document.getElementById("p-slider").value, 10);
      document.getElementById("p-value").textContent = P_LABELS[pIdx].toFixed(4);
      document.getElementById("n-value").textContent = n;
      const sortLeft = document.getElementById("display-left").value === "sorted";
      const point = getBinomialPoint(n, pIdx);
      if (!point) return;
      let xData, yData, colors, xaxisOverride;
      if (sortLeft && point.perm) {{
        xData = point.x;
        yData = point.perm.map(function(i) {{ return point.y[i]; }});
        colors = point.perm.map(function(k) {{ return colorForK(k, n); }});
        xaxisOverride = {{ tickvals: point.x, ticktext: point.perm.map(String) }};
      }} else {{
        xData = point.x;
        yData = point.y;
        colors = point.x.map(function(k) {{ return colorForK(k, n); }});
        xaxisOverride = {{ }};
      }}
      var layoutUpdate = {{ ...layoutLeft, title: {{ text: "Binomial(n=" + n + ", p=" + P_LABELS[pIdx].toFixed(4) + ")" }} }};
      if (Object.keys(xaxisOverride).length) layoutUpdate.xaxis = {{ ...layoutLeft.xaxis, ...xaxisOverride, showticklabels: false }};
      var stripY = xData.map(function() {{ return -0.1; }});
      var xMin = Math.min(...xData), xMax = Math.max(...xData);
      Plotly.react("graph-left", [
        {{ x: xData, y: stripY, type: "bar", marker: {{ color: colors }}, width: 1, offsetgroup: "bars", alignmentgroup: "bars" }},
        {{ x: xData, y: yData, type: "bar", marker: {{ color: colors }}, width: 0.5, offsetgroup: "bars", alignmentgroup: "bars" }},
        {{ x: [xMin - 0.5, xMax + 0.5], y: [0, 0], mode: "lines", line: {{ color: "black", width: 1 }}, showlegend: false }}
      ], layoutUpdate);
    }}

    function updateRight() {{
      const n = parseInt(document.getElementById("n-slider").value, 10);
      const pIdx = parseInt(document.getElementById("p-slider").value, 10);
      document.getElementById("n-value").textContent = n;
      const proj = getProjectionData();
      const d = proj[getDataIndex(n)];
      const displayRange = document.getElementById("display-range").value;
      const displayRight = document.getElementById("display-right").value;
      const componentsView = getComponentsView();
      const filtered = filterByPRange(d, n, displayRange);
      const xy = getXYFromFiltered(filtered, displayRight, componentsView);
      const axisRange = getAxisRange(d, xy);
      const autozoomOn = document.getElementById("autozoom").checked;
      const layoutUpdate = {{
        ...layoutRight,
        title: {{ text: "PCA: n=" + n + " (" + displayRight + ")" }},
        xaxis: {{ ...layoutRight.xaxis, range: axisRange.x, ...(autozoomOn ? {{ scaleanchor: false }} : {{}}) }},
        yaxis: {{ ...layoutRight.yaxis, range: axisRange.y }}
      }};
      var traces = [];
      if (document.getElementById("pca-connect-lines").checked && xy.x.length > 1) {{
        traces.push({{
          x: xy.x, y: xy.y, mode: "lines",
          line: {{ color: "#888888", width: 0.5, shape: "linear" }},
          showlegend: false, hoverinfo: "skip"
        }});
      }}
      traces.push({{
        x: xy.x,
        y: xy.y,
        mode: "markers",
        marker: {{ size: 4, color: filtered.p, colorscale: "Viridis", colorbar: {{ title: "p", tickformat: ".2f" }}, showscale: true }},
        text: filtered.p.map(function(p) {{ return "p=" + p.toFixed(3); }}),
        hovertemplate: "%{{text}}<extra></extra>",
        showlegend: false
      }});
      if (document.getElementById("simplex-points").checked && d.vertex_pc1) {{
        var vxy = getVertexXY(d, componentsView);
        traces.push({{
          x: vxy.x, y: vxy.y, mode: "markers",
          marker: {{ size: 8, color: "red", symbol: "diamond" }},
          text: d.vertex_pc1.map(function(_, i) {{ return "k=" + i; }}),
          hovertemplate: "%{{text}}<extra></extra>",
          showlegend: false
        }});
      }}
      if (document.getElementById("show-p").checked) {{
        const pVal = parseFloat(P_LABELS[pIdx]);
        let idx = 0;
        let best = Math.abs(filtered.p[0] - pVal);
        for (let i = 1; i < filtered.p.length; i++) {{
          const dlt = Math.abs(filtered.p[i] - pVal);
          if (dlt < best) {{ best = dlt; idx = i; }}
        }}
        traces.push({{ x: [xy.x[idx]], y: [xy.y[idx]], mode: "markers", marker: {{ size: 12, color: "black", symbol: "circle" }},
          text: ["p=" + pVal.toFixed(4)], hovertemplate: "%{{text}}<extra></extra>", showlegend: false }});
      }}
      Plotly.react("graph-right", traces, layoutUpdate);
    }}

    function updateAll() {{ updateLeft(); updateRight(); }}

    var binInit = getBinomialPoint({n_min}, {p_default_idx});
    var colorsInit = binInit.x.map(function(k) {{ return colorForK(k, binInit.x.length - 1); }});
    var xMinInit = Math.min(...binInit.x), xMaxInit = Math.max(...binInit.x);
    Plotly.newPlot("graph-left", [
      {{ x: binInit.x, y: binInit.x.map(function() {{ return -0.1; }}), type: "bar", marker: {{ color: colorsInit }}, width: 1, offsetgroup: "bars", alignmentgroup: "bars" }},
      {{ x: binInit.x, y: binInit.y, type: "bar", marker: {{ color: colorsInit }}, width: 0.5, offsetgroup: "bars", alignmentgroup: "bars" }},
      {{ x: [xMinInit - 0.5, xMaxInit + 0.5], y: [0, 0], mode: "lines", line: {{ color: "black", width: 1 }}, showlegend: false }}
    ], layoutLeft, {{ responsive: true }});

    var projInit = getProjectionData()[getDataIndex({n_min})];
    var displayRangeInit = document.getElementById("display-range").value;
    var displayRightInit = document.getElementById("display-right").value;
    var compInit = getComponentsView();
    var filteredInit = filterByPRange(projInit, {n_min}, displayRangeInit);
    var xyInit = getXYFromFiltered(filteredInit, displayRightInit, compInit);
    var axisRangeInit = getAxisRange(projInit, xyInit);
    var autozoomInit = document.getElementById("autozoom").checked;
    var layoutRightInit = {{
      ...layoutRight,
      xaxis: {{ ...layoutRight.xaxis, range: axisRangeInit.x, ...(autozoomInit ? {{ scaleanchor: false }} : {{}}) }},
      yaxis: {{ ...layoutRight.yaxis, range: axisRangeInit.y }}
    }};
    var tracesRightInit = [];
    if (document.getElementById("pca-connect-lines").checked && xyInit.x.length > 1) {{
      tracesRightInit.push({{
        x: xyInit.x, y: xyInit.y, mode: "lines",
        line: {{ color: "#888888", width: 0.5, shape: "linear" }},
        showlegend: false, hoverinfo: "skip"
      }});
    }}
    tracesRightInit.push({{
      x: xyInit.x, y: xyInit.y, mode: "markers",
      marker: {{ size: 4, color: filteredInit.p, colorscale: "Viridis", colorbar: {{ title: "p", tickformat: ".2f" }}, showscale: true }},
      text: filteredInit.p.map(function(p) {{ return "p=" + p.toFixed(3); }}),
      hovertemplate: "%{{text}}<extra></extra>",
      showlegend: false
    }});
    Plotly.newPlot("graph-right", tracesRightInit, layoutRightInit, {{ responsive: true }});

    document.getElementById("n-slider").addEventListener("input", updateAll);
    document.getElementById("p-slider").addEventListener("input", function() {{ updateLeft(); if (document.getElementById("show-p").checked) updateRight(); }});
    document.getElementById("display-left").addEventListener("change", updateLeft);
    document.getElementById("display-right").addEventListener("change", updateRight);
    document.getElementById("pca-range").addEventListener("change", updateRight);
    document.getElementById("pca-data").addEventListener("change", updateRight);
    document.getElementById("pca-components").addEventListener("change", updateRight);
    document.getElementById("display-range").addEventListener("change", updateRight);
    document.getElementById("simplex-points").addEventListener("change", updateRight);
    document.getElementById("autozoom").addEventListener("change", updateRight);
    document.getElementById("show-p").addEventListener("change", updateRight);
    document.getElementById("pca-connect-lines").addEventListener("change", updateRight);

    updateAll();
  </script>
</body>
</html>
"""
