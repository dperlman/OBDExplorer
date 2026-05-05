"""
Combined explorer: left = interactive binomial bar chart, right = PCA 2D projection.
Precomputed data; n=2..100, 1001 p values. No server — open generated HTML in browser.
"""
import json
import numpy as np
from scipy.stats import binom
from sklearn.decomposition import PCA

N_MIN, N_MAX = 2, 100
N_VALS = list(range(N_MIN, N_MAX + 1))
P_STEPS = 1001
P_NUM = 1001
p_values = np.linspace(0.0, 1.0, P_STEPS)
p_default_idx = P_STEPS // 2


def _precompute_binomial():
    """Return list of {x, y, perm} for each (n, p). Index = (n - N_MIN) * P_STEPS + p_idx."""
    out = []
    for n in N_VALS:
        k = np.arange(n + 1, dtype=int)
        for p in p_values:
            pmf = binom.pmf(k, n, p)
            y = [round(float(v), 6) for v in pmf]
            perm = np.argsort(pmf, kind="stable").tolist()
            out.append({"x": k.tolist(), "y": y, "perm": perm})
    return out


def _precompute_pca(p_min: float, p_max: float):
    """For each n, run PCA on [unsorted; ordered] PMF rows for p in [p_min, p_max]; return (list of dicts, range)."""
    p_vals = np.linspace(p_min, p_max, P_NUM)
    out = []
    all_x_min, all_x_max = float("inf"), float("-inf")
    all_y_min, all_y_max = float("inf"), float("-inf")
    for n in N_VALS:
        k_vals = np.arange(n + 1, dtype=float)
        unsorted_pmf = binom.pmf(k_vals[:, np.newaxis], n, p_vals[np.newaxis, :])
        ordered_pmf = np.sort(unsorted_pmf, axis=0)
        X_unsorted = unsorted_pmf.T
        X_ordered = ordered_pmf.T
        X_combined = np.vstack([X_unsorted, X_ordered])
        reducer = PCA(n_components=2, random_state=0)
        embed_combined = reducer.fit_transform(X_combined)
        embed_unsorted = embed_combined[:P_NUM]
        embed_ordered = embed_combined[P_NUM:]
        vertices = np.eye(n + 1)
        vertex_embed = reducer.transform(vertices)
        all_embed = np.vstack([embed_combined, vertex_embed])
        x_min = float(all_embed[:, 0].min())
        x_max = float(all_embed[:, 0].max())
        y_min = float(all_embed[:, 1].min())
        y_max = float(all_embed[:, 1].max())
        all_x_min = min(all_x_min, x_min)
        all_x_max = max(all_x_max, x_max)
        all_y_min = min(all_y_min, y_min)
        all_y_max = max(all_y_max, y_max)
        p_list = [round(float(p), 6) for p in p_vals]
        out.append({
            "n": n,
            "p": p_list,
            "unsorted_x": embed_unsorted[:, 0].tolist(),
            "unsorted_y": embed_unsorted[:, 1].tolist(),
            "ordered_x": embed_ordered[:, 0].tolist(),
            "ordered_y": embed_ordered[:, 1].tolist(),
            "vertex_x": vertex_embed[:, 0].tolist(),
            "vertex_y": vertex_embed[:, 1].tolist(),
            "xMin": x_min, "xMax": x_max, "yMin": y_min, "yMax": y_max,
        })
    return out, {"xMin": all_x_min, "xMax": all_x_max, "yMin": all_y_min, "yMax": all_y_max}


def _merge_range(a: dict, b: dict) -> dict:
    return {
        "xMin": min(a["xMin"], b["xMin"]),
        "xMax": max(a["xMax"], b["xMax"]),
        "yMin": min(a["yMin"], b["yMin"]),
        "yMax": max(a["yMax"], b["yMax"]),
    }


def _build_html(
    binomial_data: list,
    data_pca_full: list,
    data_pca_half: list,
    global_range: dict,
) -> str:
    p_labels_json = json.dumps([round(p, 4) for p in p_values])
    range_x = json.dumps([global_range["xMin"], global_range["xMax"]])
    range_y = json.dumps([global_range["yMin"], global_range["yMax"]])
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    .row {{ display: flex; flex-wrap: wrap; margin: 10px 0; }}
    .col {{ flex: 1; min-width: 400px; padding: 0 8px; box-sizing: border-box; }}
    .col .graph {{ width: 100%; height: 450px; }}
    .controls {{ margin-top: 8px; }}
    .n-row {{ justify-content: center; margin-top: 16px; }}
    .n-row .n-wrap {{ max-width: 500px; width: 100%; }}
  </style>
</head>
<body>
  <div class="row">
    <div class="col">
      <div id="graph-left" class="graph"></div>
      <div class="controls">
        <label for="p-slider"><b>p</b> = <span id="p-value">0.50</span></label>
        <input type="range" id="p-slider" min="0" max="{P_STEPS - 1}" value="{p_default_idx}" style="width: 100%;">
      </div>
      <div class="controls" style="margin-top: 8px;">
        <button type="button" id="sort-btn">Sort bars (low to high)</button>
      </div>
    </div>
    <div class="col">
      <div id="graph-right" class="graph"></div>
      <div class="controls">
        <label for="p-range"><b>p range</b></label>
        <select id="p-range" style="margin-left: 8px;">
          <option value="full">full (0 to 1)</option>
          <option value="half" selected>half (0.5 to 1)</option>
        </select>
        <label style="margin-left: 16px;"><input type="checkbox" id="simplex-points"> simplex points</label>
        <label style="margin-left: 16px;"><input type="checkbox" id="show-p"> Show p</label>
      </div>
    </div>
  </div>
  <div class="row n-row">
    <div class="n-wrap">
      <label for="n-slider"><b>n</b> = <span id="n-value">{N_MIN}</span></label>
      <input type="range" id="n-slider" min="{N_MIN}" max="{N_MAX}" value="{N_MIN}" style="width: 100%;">
    </div>
  </div>

  <script>
    const BINOMIAL_DATA = {json.dumps(binomial_data)};
    const P_LABELS = {p_labels_json};
    const PROJECTION_DATA_PCA_FULL = {json.dumps(data_pca_full)};
    const PROJECTION_DATA_PCA_HALF = {json.dumps(data_pca_half)};
    const AXIS_RANGE_X = {range_x};
    const AXIS_RANGE_Y = {range_y};
    let sortBars = false;

    function getBinomialIndex(n, pIdx) {{
      return (n - {N_MIN}) * {P_STEPS} + pIdx;
    }}
    function getProjIndex(n) {{ return n - {N_MIN}; }}
    function getProjectionData() {{
      return document.getElementById("p-range").value === "full" ? PROJECTION_DATA_PCA_FULL : PROJECTION_DATA_PCA_HALF;
    }}

    const P_SLIDER_FULL_MIN = 0;
    const P_SLIDER_FULL_MAX = {P_STEPS - 1};
    const P_SLIDER_HALF_MIN = {P_STEPS // 2};
    const P_SLIDER_HALF_MAX = {P_STEPS - 1};

    function syncPSliderToPRange() {{
      const isFull = document.getElementById("p-range").value === "full";
      const pSlider = document.getElementById("p-slider");
      const min = isFull ? P_SLIDER_FULL_MIN : P_SLIDER_HALF_MIN;
      const max = isFull ? P_SLIDER_FULL_MAX : P_SLIDER_HALF_MAX;
      pSlider.min = min;
      pSlider.max = max;
      let val = parseInt(pSlider.value, 10);
      if (val < min) pSlider.value = min;
      else if (val > max) pSlider.value = max;
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
      title: {{ text: "PCA: n=2 (binomial)" }},
      xaxis: {{ title: "", scaleanchor: "y", scaleratio: 1, range: AXIS_RANGE_X, showgrid: true, showticklabels: false }},
      yaxis: {{ title: "", range: AXIS_RANGE_Y, showgrid: true, showticklabels: false }},
      margin: {{ t: 50, r: 20, b: 45, l: 50 }},
      showlegend: false
    }};

    function updateLeft() {{
      const n = parseInt(document.getElementById("n-slider").value, 10);
      const pIdx = parseInt(document.getElementById("p-slider").value, 10);
      document.getElementById("p-value").textContent = P_LABELS[pIdx].toFixed(4);
      const point = BINOMIAL_DATA[getBinomialIndex(n, pIdx)];
      let xData, yData, colors, xaxisOverride;
      if (sortBars && point.perm) {{
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
      Plotly.react("graph-left", [
        {{ x: xData, y: stripY, type: "bar", marker: {{ color: colors }}, width: 1, offsetgroup: "bars", alignmentgroup: "bars" }},
        {{ x: xData, y: yData, type: "bar", marker: {{ color: colors }}, width: 0.5, offsetgroup: "bars", alignmentgroup: "bars" }}
      ], layoutUpdate);
    }}

    function updateRight() {{
      const n = parseInt(document.getElementById("n-slider").value, 10);
      const pIdx = parseInt(document.getElementById("p-slider").value, 10);
      document.getElementById("n-value").textContent = n;
      const proj = getProjectionData();
      const d = proj[getProjIndex(n)];
      const xData = sortBars ? d.ordered_x : d.unsorted_x;
      const yData = sortBars ? d.ordered_y : d.unsorted_y;
      const label = sortBars ? "ordered" : "binomial";
      var layoutUpdate = {{ ...layoutRight, title: {{ text: "PCA: n=" + n + " (" + label + ")" }}, xaxis: {{ ...layoutRight.xaxis, range: AXIS_RANGE_X }}, yaxis: {{ ...layoutRight.yaxis, range: AXIS_RANGE_Y }} }};
      var traces = [{{
        x: xData, y: yData, mode: "markers",
        marker: {{ size: 4, color: d.p, colorscale: "Viridis", showscale: false }},
        text: d.p.map(function(p) {{ return "p=" + p.toFixed(4); }}), hovertemplate: "%{{text}}<extra></extra>",
        showlegend: false
      }}];
      if (document.getElementById("simplex-points").checked && d.vertex_x) {{
        traces.push({{ x: d.vertex_x, y: d.vertex_y, mode: "markers", marker: {{ size: 8, color: "red", symbol: "diamond" }},
          text: d.vertex_x.map(function(_, i) {{ return "k=" + i; }}), hovertemplate: "%{{text}}<extra></extra>", showlegend: false }});
      }}
      if (document.getElementById("show-p").checked) {{
        const isFull = document.getElementById("p-range").value === "full";
        const idx = isFull ? pIdx : 2 * pIdx - 1000;
        if (isFull || (idx >= 0 && idx <= 1000)) {{
          const i = isFull ? pIdx : idx;
          const px = sortBars ? d.ordered_x[i] : d.unsorted_x[i];
          const py = sortBars ? d.ordered_y[i] : d.unsorted_y[i];
          traces.push({{ x: [px], y: [py], mode: "markers", marker: {{ size: 12, color: "black", symbol: "circle" }},
            text: ["p=" + P_LABELS[pIdx].toFixed(4)], hovertemplate: "%{{text}}<extra></extra>", showlegend: false }});
        }}
      }}
      Plotly.react("graph-right", traces, layoutUpdate);
    }}

    function updateAll() {{ updateLeft(); updateRight(); }}

    var binInit = BINOMIAL_DATA[getBinomialIndex({N_MIN}, {p_default_idx})];
    var colorsInit = binInit.x.map(function(k) {{ return colorForK(k, binInit.x.length - 1); }});
    Plotly.newPlot("graph-left", [
      {{ x: binInit.x, y: binInit.x.map(function() {{ return -0.1; }}), type: "bar", marker: {{ color: colorsInit }}, width: 1, offsetgroup: "bars", alignmentgroup: "bars" }},
      {{ x: binInit.x, y: binInit.y, type: "bar", marker: {{ color: colorsInit }}, width: 0.5, offsetgroup: "bars", alignmentgroup: "bars" }}
    ], layoutLeft, {{ responsive: true }});

    var projInit = getProjectionData()[getProjIndex({N_MIN})];
    var tracesRight = [{{
      x: projInit.unsorted_x, y: projInit.unsorted_y, mode: "markers",
      marker: {{ size: 4, color: projInit.p, colorscale: "Viridis", showscale: false }},
      text: projInit.p.map(function(p) {{ return "p=" + p.toFixed(4); }}), hovertemplate: "%{{text}}<extra></extra>", showlegend: false
    }}];
    Plotly.newPlot("graph-right", tracesRight, layoutRight, {{ responsive: true }});

    document.getElementById("n-slider").addEventListener("input", updateAll);
    document.getElementById("p-slider").addEventListener("input", function() {{ updateLeft(); if (document.getElementById("show-p").checked) updateRight(); }});
    document.getElementById("p-range").addEventListener("change", function() {{
      syncPSliderToPRange();
      updateAll();
    }});
    document.getElementById("simplex-points").addEventListener("change", updateRight);
    document.getElementById("show-p").addEventListener("change", updateRight);
    document.getElementById("sort-btn").addEventListener("click", function() {{
      sortBars = !sortBars;
      document.getElementById("sort-btn").textContent = sortBars ? "Show by k (unsort)" : "Sort bars (low to high)";
      updateAll();
    }});
    syncPSliderToPRange();
    updateAll();
  </script>
</body>
</html>
"""


def main(output_path: str = "OBDexplorer1.html") -> None:
    print("Precomputing binomial data (n=2..100, 1001 p)...")
    binomial_data = _precompute_binomial()
    print("Precomputing PCA full...")
    data_pca_full, range_full = _precompute_pca(0.0, 1.0)
    print("Precomputing PCA half...")
    data_pca_half, range_half = _precompute_pca(0.5, 1.0)
    global_range = _merge_range(range_full, range_half)
    html = _build_html(binomial_data, data_pca_full, data_pca_half, global_range)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {output_path}. Open in a browser (no server needed).")


if __name__ == "__main__":
    main()
