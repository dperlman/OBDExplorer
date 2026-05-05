"""Variant 4 HTML: PCA projection only (no binomial panel, no p slider, no Show p)."""

from __future__ import annotations

import json


def build_explorer4_pca_only_html_document(
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
) -> str:
    last_tie_json = json.dumps(last_tie_by_n)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    html, body {{ margin: 0; box-sizing: border-box; }}
    * {{ box-sizing: border-box; }}
    .wrap {{ max-width: 960px; margin: 0 auto; padding: 10px 12px 20px; }}
    #graph-pca {{ width: 100%; height: min(72vh, 640px); min-height: 420px; }}
    .controls-wrap {{ display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }}
    .controls-row {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }}
    .n-row {{ margin-top: 14px; max-width: 520px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div id="graph-pca"></div>
    <div class="controls-wrap">
      <div class="controls-row">
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
      <div class="controls-row">
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
        <label><input type="checkbox" id="pca-connect-lines"> lines</label>
      </div>
    </div>
    <div class="n-row">
      <label for="n-slider"><b>n</b> = <span id="n-value">{n_min}</span></label>
      <input type="range" id="n-slider" min="{n_min}" max="{n_max}" value="{n_min}" style="width: 100%;">
    </div>
  </div>

  <script>
    const DATA_FULL_UNSORTED = {json.dumps(data_full_unsorted)};
    const DATA_FULL_SORTED = {json.dumps(data_full_sorted)};
    const DATA_HALF_UNSORTED = {json.dumps(data_half_unsorted)};
    const DATA_HALF_SORTED = {json.dumps(data_half_sorted)};
    const DATA_TIE_UNSORTED = {json.dumps(data_tie_unsorted)};
    const DATA_TIE_SORTED = {json.dumps(data_tie_sorted)};
    const LAST_TIE_BY_N = {last_tie_json};
    const FIXED_RANGE = {{ x: [-1, 1], y: [-1, 1] }};

    function getDataIndex(n) {{ return n - {n_min}; }}

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

    const layoutPca = {{
      title: {{ text: "PCA: n={n_min}" }},
      xaxis: {{ title: "", scaleanchor: "y", scaleratio: 1, range: [-1, 1], showgrid: true, showticklabels: false }},
      yaxis: {{ title: "", range: [-1, 1], showgrid: true, showticklabels: false }},
      margin: {{ t: 50, r: 20, b: 45, l: 50 }},
      showlegend: false
    }};

    function updatePca() {{
      const n = parseInt(document.getElementById("n-slider").value, 10);
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
        ...layoutPca,
        title: {{ text: "PCA: n=" + n + " (" + displayRight + ")" }},
        xaxis: {{ ...layoutPca.xaxis, range: axisRange.x, ...(autozoomOn ? {{ scaleanchor: false }} : {{}}) }},
        yaxis: {{ ...layoutPca.yaxis, range: axisRange.y }}
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
      Plotly.react("graph-pca", traces, layoutUpdate);
    }}

    var projInit = getProjectionData()[getDataIndex({n_min})];
    var displayRangeInit = document.getElementById("display-range").value;
    var displayRightInit = document.getElementById("display-right").value;
    var compInit = getComponentsView();
    var filteredInit = filterByPRange(projInit, {n_min}, displayRangeInit);
    var xyInit = getXYFromFiltered(filteredInit, displayRightInit, compInit);
    var axisRangeInit = getAxisRange(projInit, xyInit);
    var autozoomInit = document.getElementById("autozoom").checked;
    var layoutPcaInit = {{
      ...layoutPca,
      title: {{ text: "PCA: n={n_min} (" + displayRightInit + ")" }},
      xaxis: {{ ...layoutPca.xaxis, range: axisRangeInit.x, ...(autozoomInit ? {{ scaleanchor: false }} : {{}}) }},
      yaxis: {{ ...layoutPca.yaxis, range: axisRangeInit.y }}
    }};
    var tracesInit = [];
    if (document.getElementById("pca-connect-lines").checked && xyInit.x.length > 1) {{
      tracesInit.push({{
        x: xyInit.x, y: xyInit.y, mode: "lines",
        line: {{ color: "#888888", width: 0.5, shape: "linear" }},
        showlegend: false, hoverinfo: "skip"
      }});
    }}
    tracesInit.push({{
      x: xyInit.x, y: xyInit.y, mode: "markers",
      marker: {{ size: 4, color: filteredInit.p, colorscale: "Viridis", colorbar: {{ title: "p", tickformat: ".2f" }}, showscale: true }},
      text: filteredInit.p.map(function(p) {{ return "p=" + p.toFixed(3); }}),
      hovertemplate: "%{{text}}<extra></extra>",
      showlegend: false
    }});
    Plotly.newPlot("graph-pca", tracesInit, layoutPcaInit, {{ responsive: true }});

    document.getElementById("n-slider").addEventListener("input", updatePca);
    document.getElementById("display-right").addEventListener("change", updatePca);
    document.getElementById("pca-range").addEventListener("change", updatePca);
    document.getElementById("pca-data").addEventListener("change", updatePca);
    document.getElementById("pca-components").addEventListener("change", updatePca);
    document.getElementById("display-range").addEventListener("change", updatePca);
    document.getElementById("simplex-points").addEventListener("change", updatePca);
    document.getElementById("autozoom").addEventListener("change", updatePca);
    document.getElementById("pca-connect-lines").addEventListener("change", updatePca);

    updatePca();
  </script>
</body>
</html>
"""
