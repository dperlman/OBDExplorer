"""
Combined explorer: viewport-based 4 quadrants. Top-left = binomial bar chart, top-right = PCA projection,
bottom-left = E[X] vs p (p from 0.5 to 1) for current n, bottom-right = controls. n=2..20. No server.
"""
import json
import numpy as np
from scipy.stats import binom
from sklearn.decomposition import PCA


def _all_tie_points(n: int, tol: float = 1e-10) -> np.ndarray:
    """All tie points p in (0, 1) for Binomial(n, p); merged within tol. Returns sorted array."""
    from scipy.special import comb
    out = []
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            ratio = comb(n, j, exact=False) / comb(n, i, exact=False)
            if ratio <= 0 or not np.isfinite(ratio):
                continue
            exp = 1.0 / (j - i)
            p = 1.0 / (1.0 + ratio ** exp)
            if 0 < p < 1 and np.isfinite(p):
                out.append(p)
    arr = np.sort(np.array(out, dtype=float))
    if len(arr) > 1:
        keep = np.concatenate([[True], np.diff(arr) > tol])
        arr = arr[keep]
    return arr


def _last_tie_above_half(n: int) -> float:
    """Tie point closest to p=1 (largest p < 1 where a tie occurs)."""
    arr = _all_tie_points(n, tol=1e-10)
    above_half = arr[arr > 0.5]
    if len(above_half) == 0:
        return 1.0 - 1e-6
    return float(above_half[-1])


N_MIN, N_MAX = 2, 20
N_VALS = list(range(N_MIN, N_MAX + 1))
P_NUM = 1001
P_STEPS = 1001
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


def _precompute_pca(p_min: float, p_max: float, fit_on_sorted: bool):
    """Fit PCA (4 components) on p in [p_min, p_max]; then project full p in [0, 1]. Return (list of dicts, r12, r23)."""
    p_vals_fit = np.linspace(p_min, p_max, P_NUM)
    p_vals_full = np.linspace(0.0, 1.0, P_NUM)
    out = []
    r12 = {"xMin": float("inf"), "xMax": float("-inf"), "yMin": float("inf"), "yMax": float("-inf")}
    r23 = {"xMin": float("inf"), "xMax": float("-inf"), "yMin": float("inf"), "yMax": float("-inf")}
    for n in N_VALS:
        k_vals = np.arange(n + 1, dtype=float)
        unsorted_pmf_fit = binom.pmf(k_vals[:, np.newaxis], n, p_vals_fit[np.newaxis, :])
        ordered_pmf_fit = np.sort(unsorted_pmf_fit, axis=0)
        X_unsorted_fit = unsorted_pmf_fit.T
        X_ordered_fit = ordered_pmf_fit.T
        X_fit = X_ordered_fit if fit_on_sorted else X_unsorted_fit
        n_comp = min(4, n + 1)
        reducer = PCA(n_components=n_comp, random_state=0)
        reducer.fit(X_fit)
        unsorted_pmf_full = binom.pmf(k_vals[:, np.newaxis], n, p_vals_full[np.newaxis, :])
        ordered_pmf_full = np.sort(unsorted_pmf_full, axis=0)
        X_unsorted_full = unsorted_pmf_full.T
        X_ordered_full = ordered_pmf_full.T
        embed_unsorted = reducer.transform(X_unsorted_full)
        embed_ordered = reducer.transform(X_ordered_full)
        vertices = np.eye(n + 1)
        vertex_embed = reducer.transform(vertices)
        if n_comp < 4:
            embed_unsorted = np.hstack([embed_unsorted, np.zeros((embed_unsorted.shape[0], 4 - n_comp))])
            embed_ordered = np.hstack([embed_ordered, np.zeros((embed_ordered.shape[0], 4 - n_comp))])
            vertex_embed = np.hstack([vertex_embed, np.zeros((vertex_embed.shape[0], 4 - n_comp))])
        curve_embed = np.vstack([embed_unsorted, embed_ordered])
        r12["xMin"] = min(r12["xMin"], float(curve_embed[:, 0].min()))
        r12["xMax"] = max(r12["xMax"], float(curve_embed[:, 0].max()))
        r12["yMin"] = min(r12["yMin"], float(curve_embed[:, 1].min()))
        r12["yMax"] = max(r12["yMax"], float(curve_embed[:, 1].max()))
        r23["xMin"] = min(r23["xMin"], float(curve_embed[:, 1].min()))
        r23["xMax"] = max(r23["xMax"], float(curve_embed[:, 1].max()))
        range_12_x = [float(curve_embed[:, 0].min()), float(curve_embed[:, 0].max())]
        range_12_y = [float(curve_embed[:, 1].min()), float(curve_embed[:, 1].max())]
        range_23_x = [float(curve_embed[:, 1].min()), float(curve_embed[:, 1].max())]
        range_23_y = [float(curve_embed[:, 2].min()), float(curve_embed[:, 2].max())]
        p_list = [round(float(p), 6) for p in p_vals_full]
        out.append({
            "n": n,
            "p": p_list,
            "unsorted_pc1": embed_unsorted[:, 0].tolist(),
            "unsorted_pc2": embed_unsorted[:, 1].tolist(),
            "unsorted_pc3": embed_unsorted[:, 2].tolist(),
            "unsorted_pc4": embed_unsorted[:, 3].tolist(),
            "ordered_pc1": embed_ordered[:, 0].tolist(),
            "ordered_pc2": embed_ordered[:, 1].tolist(),
            "ordered_pc3": embed_ordered[:, 2].tolist(),
            "ordered_pc4": embed_ordered[:, 3].tolist(),
            "vertex_pc1": vertex_embed[:, 0].tolist(),
            "vertex_pc2": vertex_embed[:, 1].tolist(),
            "vertex_pc3": vertex_embed[:, 2].tolist(),
            "vertex_pc4": vertex_embed[:, 3].tolist(),
            "range_12_x": range_12_x,
            "range_12_y": range_12_y,
            "range_23_x": range_23_x,
            "range_23_y": range_23_y,
        })
    return out, r12, r23


def _precompute_tie_pca(fit_on_sorted: bool):
    """Fit PCA (4 components) on p in [0.5, last_tie]; then project full p in [0, 1]. Return (list of dicts, r12, r23)."""
    p_vals_full = np.linspace(0.0, 1.0, P_NUM)
    out = []
    r12 = {"xMin": float("inf"), "xMax": float("-inf"), "yMin": float("inf"), "yMax": float("-inf")}
    r23 = {"xMin": float("inf"), "xMax": float("-inf"), "yMin": float("inf"), "yMax": float("-inf")}
    for n in N_VALS:
        p_tie = _last_tie_above_half(n)
        p_vals_fit = np.linspace(0.5, p_tie, P_NUM)
        k_vals = np.arange(n + 1, dtype=float)
        unsorted_pmf_fit = binom.pmf(k_vals[:, np.newaxis], n, p_vals_fit[np.newaxis, :])
        ordered_pmf_fit = np.sort(unsorted_pmf_fit, axis=0)
        X_unsorted_fit = unsorted_pmf_fit.T
        X_ordered_fit = ordered_pmf_fit.T
        X_fit = X_ordered_fit if fit_on_sorted else X_unsorted_fit
        n_comp = min(4, n + 1)
        reducer = PCA(n_components=n_comp, random_state=0)
        reducer.fit(X_fit)
        unsorted_pmf_full = binom.pmf(k_vals[:, np.newaxis], n, p_vals_full[np.newaxis, :])
        ordered_pmf_full = np.sort(unsorted_pmf_full, axis=0)
        X_unsorted_full = unsorted_pmf_full.T
        X_ordered_full = ordered_pmf_full.T
        embed_unsorted = reducer.transform(X_unsorted_full)
        embed_ordered = reducer.transform(X_ordered_full)
        vertices = np.eye(n + 1)
        vertex_embed = reducer.transform(vertices)
        if n_comp < 4:
            embed_unsorted = np.hstack([embed_unsorted, np.zeros((embed_unsorted.shape[0], 4 - n_comp))])
            embed_ordered = np.hstack([embed_ordered, np.zeros((embed_ordered.shape[0], 4 - n_comp))])
            vertex_embed = np.hstack([vertex_embed, np.zeros((vertex_embed.shape[0], 4 - n_comp))])
        curve_embed = np.vstack([embed_unsorted, embed_ordered])
        r12["xMin"] = min(r12["xMin"], float(curve_embed[:, 0].min()))
        r12["xMax"] = max(r12["xMax"], float(curve_embed[:, 0].max()))
        r12["yMin"] = min(r12["yMin"], float(curve_embed[:, 1].min()))
        r12["yMax"] = max(r12["yMax"], float(curve_embed[:, 1].max()))
        r23["xMin"] = min(r23["xMin"], float(curve_embed[:, 1].min()))
        r23["xMax"] = max(r23["xMax"], float(curve_embed[:, 1].max()))
        range_12_x = [float(curve_embed[:, 0].min()), float(curve_embed[:, 0].max())]
        range_12_y = [float(curve_embed[:, 1].min()), float(curve_embed[:, 1].max())]
        range_23_x = [float(curve_embed[:, 1].min()), float(curve_embed[:, 1].max())]
        range_23_y = [float(curve_embed[:, 2].min()), float(curve_embed[:, 2].max())]
        p_list = [round(float(p), 6) for p in p_vals_full]
        out.append({
            "n": n,
            "p": p_list,
            "unsorted_pc1": embed_unsorted[:, 0].tolist(),
            "unsorted_pc2": embed_unsorted[:, 1].tolist(),
            "unsorted_pc3": embed_unsorted[:, 2].tolist(),
            "unsorted_pc4": embed_unsorted[:, 3].tolist(),
            "ordered_pc1": embed_ordered[:, 0].tolist(),
            "ordered_pc2": embed_ordered[:, 1].tolist(),
            "ordered_pc3": embed_ordered[:, 2].tolist(),
            "ordered_pc4": embed_ordered[:, 3].tolist(),
            "vertex_pc1": vertex_embed[:, 0].tolist(),
            "vertex_pc2": vertex_embed[:, 1].tolist(),
            "vertex_pc3": vertex_embed[:, 2].tolist(),
            "vertex_pc4": vertex_embed[:, 3].tolist(),
            "range_12_x": range_12_x,
            "range_12_y": range_12_y,
            "range_23_x": range_23_x,
            "range_23_y": range_23_y,
        })
    return out, r12, r23


def _build_html(
    binomial_data: list,
    data_full_unsorted: list,
    data_full_sorted: list,
    data_half_unsorted: list,
    data_half_sorted: list,
    data_tie_unsorted: list,
    data_tie_sorted: list,
    last_tie_by_n: dict,
) -> str:
    p_labels_json = json.dumps([round(p, 4) for p in p_values])
    last_tie_json = json.dumps(last_tie_by_n)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; overflow: hidden; box-sizing: border-box; }}
    * {{ box-sizing: border-box; }}
    .quad-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      grid-template-rows: 1fr 1fr;
      width: 100vw;
      height: 100vh;
    }}
    .quad {{ min-height: 0; min-width: 0; display: flex; flex-direction: column; padding: 6px; }}
    .quad-graph {{ flex: 1; min-height: 0; }}
    .quad-controls {{ overflow-y: auto; }}
  </style>
</head>
<body>
  <div class="quad-grid">
    <div class="quad">
      <div id="graph-left" class="quad-graph"></div>
    </div>
    <div class="quad">
      <div id="graph-right" class="quad-graph"></div>
    </div>
    <div class="quad">
      <div id="graph-evp" class="quad-graph"></div>
    </div>
    <div class="quad quad-controls">
      <div style="margin-bottom: 8px;">
        <label for="p-slider"><b>p</b> = <span id="p-value">0.50</span></label>
        <input type="range" id="p-slider" min="0" max="{P_STEPS - 1}" value="{p_default_idx}" style="width: 100%;">
      </div>
      <div style="margin-bottom: 8px;">
        <label for="display-left"><b>display (left)</b></label>
        <select id="display-left"><option value="unsorted">unsorted</option><option value="sorted">sorted</option></select>
      </div>
      <div style="margin-bottom: 8px;">
        <label for="n-slider"><b>n</b> = <span id="n-value">{N_MIN}</span></label>
        <input type="range" id="n-slider" min="{N_MIN}" max="{N_MAX}" value="{N_MIN}" style="width: 100%;">
      </div>
      <hr style="margin: 10px 0;">
      <div style="margin-bottom: 6px;"><label for="pca-range"><b>range for PCA</b></label> <select id="pca-range"><option value="full">0-1</option><option value="half" selected>0.5-1</option><option value="tie">0.5-last tie</option></select></div>
      <div style="margin-bottom: 6px;"><label for="pca-data"><b>data for PCA</b></label> <select id="pca-data"><option value="unsorted">unsorted</option><option value="sorted">sorted</option></select></div>
      <div style="margin-bottom: 6px;"><label for="pca-components"><b>components</b></label> <select id="pca-components"><option value="12" selected>1,2</option><option value="23">2,3</option><option value="34">3,4</option></select></div>
      <div style="margin-bottom: 6px;"><label for="display-right"><b>display (right)</b></label> <select id="display-right"><option value="unsorted">unsorted</option><option value="sorted">sorted</option></select></div>
      <div style="margin-bottom: 6px;"><label for="display-range"><b>range for display</b></label> <select id="display-range"><option value="full">0–1</option><option value="half" selected>0.5–1</option><option value="tie">0.5–last tie</option></select></div>
      <div style="margin-bottom: 6px;">
        <label><input type="checkbox" id="simplex-points"> simplex</label>
        <label><input type="checkbox" id="autozoom" checked> autozoom</label>
        <label><input type="checkbox" id="show-p"> Show p</label>
      </div>
    </div>
  </div>

  <script>
    const BINOMIAL_DATA = {json.dumps(binomial_data)};
    const P_LABELS = {p_labels_json};
    const DATA_FULL_UNSORTED = {json.dumps(data_full_unsorted)};
    const DATA_FULL_SORTED = {json.dumps(data_full_sorted)};
    const DATA_HALF_UNSORTED = {json.dumps(data_half_unsorted)};
    const DATA_HALF_SORTED = {json.dumps(data_half_sorted)};
    const DATA_TIE_UNSORTED = {json.dumps(data_tie_unsorted)};
    const DATA_TIE_SORTED = {json.dumps(data_tie_sorted)};
    const LAST_TIE_BY_N = {last_tie_json};
    const FIXED_RANGE = {{ x: [-1, 1], y: [-1, 1] }};

    function getBinomialIndex(n, pIdx) {{ return (n - {N_MIN}) * {P_STEPS} + pIdx; }}
    function getDataIndex(n) {{ return n - {N_MIN}; }}

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
      if (componentsView === "34") {{
        px = displayData === "sorted" ? filtered.ordered_pc3 : filtered.unsorted_pc3;
        py = displayData === "sorted" ? filtered.ordered_pc4 : filtered.unsorted_pc4;
      }} else if (componentsView === "23") {{
        px = displayData === "sorted" ? filtered.ordered_pc2 : filtered.unsorted_pc2;
        py = displayData === "sorted" ? filtered.ordered_pc3 : filtered.unsorted_pc3;
      }} else {{
        px = displayData === "sorted" ? filtered.ordered_pc1 : filtered.unsorted_pc1;
        py = displayData === "sorted" ? filtered.ordered_pc2 : filtered.unsorted_pc2;
      }}
      return {{ x: px, y: py }};
    }}

    function getVertexXY(d, componentsView) {{
      if (componentsView === "34") return {{ x: d.vertex_pc3, y: d.vertex_pc4 }};
      if (componentsView === "23") return {{ x: d.vertex_pc2, y: d.vertex_pc3 }};
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

    const P_HALF_START = {P_STEPS // 2};
    const layoutEvp = {{
      title: {{ text: "E[X] vs p (0.5 to 1), n=2" }},
      xaxis: {{ title: "p", range: [0.5, 1], showgrid: true }},
      yaxis: {{ title: "E[X]", showgrid: true }},
      margin: {{ t: 40, r: 20, b: 40, l: 50 }},
      showlegend: false
    }};

    function expectedRank(point) {{
      var n = point.x.length - 1;
      var e = 0;
      for (var i = 0; i <= n; i++) e += i * point.y[point.perm[i]];
      return e;
    }}

    function updateEvp() {{
      const n = parseInt(document.getElementById("n-slider").value, 10);
      const pIdx = parseInt(document.getElementById("p-slider").value, 10);
      const sortLeft = document.getElementById("display-left").value === "sorted";
      var pArr = [];
      for (var i = P_HALF_START; i < {P_STEPS}; i++) pArr.push(parseFloat(P_LABELS[i]));
      var yArr;
      if (sortLeft) {{
        yArr = [];
        for (var i = P_HALF_START; i < {P_STEPS}; i++) {{
          var pt = BINOMIAL_DATA[getBinomialIndex(n, i)];
          yArr.push(expectedRank(pt));
        }}
      }} else {{
        yArr = pArr.map(function(p) {{ return n * p; }});
      }}
      var traces = [{{
        x: pArr,
        y: yArr,
        mode: "lines",
        line: {{ color: "blue", width: 1.5 }},
        showlegend: false
      }}];
      var pVal = parseFloat(P_LABELS[pIdx]);
      if (pVal >= 0.5 && pVal <= 1) {{
        var yVal = sortLeft ? expectedRank(BINOMIAL_DATA[getBinomialIndex(n, pIdx)]) : n * pVal;
        traces.push({{ x: [pVal], y: [yVal], mode: "markers", marker: {{ size: 10, color: "black", symbol: "circle" }},
          showlegend: false }});
      }}
      var layoutUpdate = {{
        ...layoutEvp,
        title: {{ text: (sortLeft ? "E[rank] (sorted)" : "E[X]") + " vs p (0.5 ≤ p ≤ 1), n=" + n }},
        xaxis: {{ ...layoutEvp.xaxis, title: "p", range: [0.5, 1] }},
        yaxis: {{ ...layoutEvp.yaxis, title: sortLeft ? "E[rank]" : "E[X]" }}
      }};
      Plotly.react("graph-evp", traces, layoutUpdate);
    }}

    function updateLeft() {{
      const n = parseInt(document.getElementById("n-slider").value, 10);
      const pIdx = parseInt(document.getElementById("p-slider").value, 10);
      document.getElementById("p-value").textContent = P_LABELS[pIdx].toFixed(4);
      const sortLeft = document.getElementById("display-left").value === "sorted";
      const point = BINOMIAL_DATA[getBinomialIndex(n, pIdx)];
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
      var traces = [{{
        x: xy.x,
        y: xy.y,
        mode: "markers",
        marker: {{ size: 4, color: filtered.p, colorscale: "Viridis", colorbar: {{ title: "p", tickformat: ".2f" }}, showscale: true }},
        text: filtered.p.map(function(p) {{ return "p=" + p.toFixed(3); }}),
        hovertemplate: "%{{text}}<extra></extra>",
        showlegend: false
      }}];
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
          const d = Math.abs(filtered.p[i] - pVal);
          if (d < best) {{ best = d; idx = i; }}
        }}
        traces.push({{ x: [xy.x[idx]], y: [xy.y[idx]], mode: "markers", marker: {{ size: 12, color: "black", symbol: "circle" }},
          text: ["p=" + pVal.toFixed(4)], hovertemplate: "%{{text}}<extra></extra>", showlegend: false }});
      }}
      Plotly.react("graph-right", traces, layoutUpdate);
    }}

    function updateAll() {{ updateLeft(); updateRight(); updateEvp(); }}

    var binInit = BINOMIAL_DATA[getBinomialIndex({N_MIN}, {p_default_idx})];
    var colorsInit = binInit.x.map(function(k) {{ return colorForK(k, binInit.x.length - 1); }});
    var xMinInit = Math.min(...binInit.x), xMaxInit = Math.max(...binInit.x);
    Plotly.newPlot("graph-left", [
      {{ x: binInit.x, y: binInit.x.map(function() {{ return -0.1; }}), type: "bar", marker: {{ color: colorsInit }}, width: 1, offsetgroup: "bars", alignmentgroup: "bars" }},
      {{ x: binInit.x, y: binInit.y, type: "bar", marker: {{ color: colorsInit }}, width: 0.5, offsetgroup: "bars", alignmentgroup: "bars" }},
      {{ x: [xMinInit - 0.5, xMaxInit + 0.5], y: [0, 0], mode: "lines", line: {{ color: "black", width: 1 }}, showlegend: false }}
    ], layoutLeft, {{ responsive: true }});

    var projInit = getProjectionData()[getDataIndex({N_MIN})];
    var displayRangeInit = document.getElementById("display-range").value;
    var displayRightInit = document.getElementById("display-right").value;
    var compInit = getComponentsView();
    var filteredInit = filterByPRange(projInit, {N_MIN}, displayRangeInit);
    var xyInit = getXYFromFiltered(filteredInit, displayRightInit, compInit);
    var axisRangeInit = getAxisRange(projInit, xyInit);
    var autozoomInit = document.getElementById("autozoom").checked;
    var layoutRightInit = {{
      ...layoutRight,
      xaxis: {{ ...layoutRight.xaxis, range: axisRangeInit.x, ...(autozoomInit ? {{ scaleanchor: false }} : {{}}) }},
      yaxis: {{ ...layoutRight.yaxis, range: axisRangeInit.y }}
    }};
    var tracesRightInit = [{{
      x: xyInit.x, y: xyInit.y, mode: "markers",
      marker: {{ size: 4, color: filteredInit.p, colorscale: "Viridis", colorbar: {{ title: "p", tickformat: ".2f" }}, showscale: true }},
      text: filteredInit.p.map(function(p) {{ return "p=" + p.toFixed(3); }}),
      hovertemplate: "%{{text}}<extra></extra>",
      showlegend: false
    }}];
    Plotly.newPlot("graph-right", tracesRightInit, layoutRightInit, {{ responsive: true }});

    var pArrInit = [];
    for (var i = P_HALF_START; i < {P_STEPS}; i++) pArrInit.push(parseFloat(P_LABELS[i]));
    var eArrInit = pArrInit.map(function(p) {{ return {N_MIN} * p; }});
    Plotly.newPlot("graph-evp", [
      {{ x: pArrInit, y: eArrInit, mode: "lines", line: {{ color: "blue", width: 1.5 }}, showlegend: false }}
    ], {{ ...layoutEvp, title: {{ text: "E[X] = n p (0.5 ≤ p ≤ 1), n=" + {N_MIN} }} }}, {{ responsive: true }});

    document.getElementById("n-slider").addEventListener("input", updateAll);
    document.getElementById("p-slider").addEventListener("input", function() {{ updateLeft(); updateEvp(); if (document.getElementById("show-p").checked) updateRight(); }});
    document.getElementById("display-left").addEventListener("change", function() {{ updateLeft(); updateEvp(); }});
    document.getElementById("display-right").addEventListener("change", updateRight);
    document.getElementById("pca-range").addEventListener("change", updateRight);
    document.getElementById("pca-data").addEventListener("change", updateRight);
    document.getElementById("pca-components").addEventListener("change", updateRight);
    document.getElementById("display-range").addEventListener("change", updateRight);
    document.getElementById("simplex-points").addEventListener("change", updateRight);
    document.getElementById("autozoom").addEventListener("change", updateRight);
    document.getElementById("show-p").addEventListener("change", updateRight);

    updateAll();
  </script>
</body>
</html>
"""


def main(output_path: str = "OBDexplorer3.html") -> None:
    print("Precomputing binomial data (n=2..20, 1001 p)...")
    binomial_data = _precompute_binomial()
    print("Precomputing PCA full (unsorted, sorted)...")
    data_full_unsorted, _, _ = _precompute_pca(0.0, 1.0, fit_on_sorted=False)
    data_full_sorted, _, _ = _precompute_pca(0.0, 1.0, fit_on_sorted=True)
    print("Precomputing PCA half (unsorted, sorted)...")
    data_half_unsorted, _, _ = _precompute_pca(0.5, 1.0, fit_on_sorted=False)
    data_half_sorted, _, _ = _precompute_pca(0.5, 1.0, fit_on_sorted=True)
    print("Precomputing PCA tie (unsorted, sorted)...")
    data_tie_unsorted, _, _ = _precompute_tie_pca(fit_on_sorted=False)
    data_tie_sorted, _, _ = _precompute_tie_pca(fit_on_sorted=True)
    last_tie_by_n = {n: _last_tie_above_half(n) for n in N_VALS}
    html = _build_html(
        binomial_data,
        data_full_unsorted, data_full_sorted,
        data_half_unsorted, data_half_sorted,
        data_tie_unsorted, data_tie_sorted,
        last_tie_by_n,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {output_path}. Open in a browser (no server needed).")


if __name__ == "__main__":
    main()
