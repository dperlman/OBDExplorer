"""
Interactive 2D projection of binomial PMF vectors (PCA).
Six transforms: 3 ranges (0-1, 0.5-1, 0.5-last tie) x 2 data (sorted/unsorted). Range for PCA, data for PCA,
data for display, range for display; N slider. No server.
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

# Match OBDtSNE.py PCA setup
P_NUM = 1001
N_MIN, N_MAX = 2, 20
N_VALS = list(range(N_MIN, N_MAX + 1))


def _precompute_pca(p_min: float, p_max: float, fit_on_sorted: bool):
    """Fit PCA (4 components) on p in [p_min, p_max]; then project full p in [0, 1]. Return (list of dicts, range_12, range_23)."""
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
        n_comp = min(4, n + 1)  # n+1 features (PMF dimension); PCA allows at most that many components
        reducer = PCA(n_components=n_comp, random_state=0)
        reducer.fit(X_fit)
        unsorted_pmf_full = binom.pmf(k_vals[:, np.newaxis], n, p_vals_full[np.newaxis, :])
        ordered_pmf_full = np.sort(unsorted_pmf_full, axis=0)
        X_unsorted_full = unsorted_pmf_full.T
        X_ordered_full = ordered_pmf_full.T
        embed_unsorted = reducer.transform(X_unsorted_full)   # (P_NUM, n_comp)
        embed_ordered = reducer.transform(X_ordered_full)
        vertices = np.eye(n + 1)
        vertex_embed = reducer.transform(vertices)
        # Pad to 4 columns when n_comp < 4 (e.g. n=2 gives only 3 components)
        if n_comp < 4:
            pad = np.zeros((embed_unsorted.shape[0], 4 - n_comp))
            embed_unsorted = np.hstack([embed_unsorted, pad])
            embed_ordered = np.hstack([embed_ordered, np.zeros((embed_ordered.shape[0], 4 - n_comp))])
            vertex_embed = np.hstack([vertex_embed, np.zeros((vertex_embed.shape[0], 4 - n_comp))])
        # Axis range from p-curve data only (exclude simplex vertices)
        curve_embed = np.vstack([embed_unsorted, embed_ordered])
        # View 1,2: cols 0,1
        r12["xMin"] = min(r12["xMin"], float(curve_embed[:, 0].min()))
        r12["xMax"] = max(r12["xMax"], float(curve_embed[:, 0].max()))
        r12["yMin"] = min(r12["yMin"], float(curve_embed[:, 1].min()))
        r12["yMax"] = max(r12["yMax"], float(curve_embed[:, 1].max()))
        # View 2,3: cols 1,2
        r23["xMin"] = min(r23["xMin"], float(curve_embed[:, 1].min()))
        r23["xMax"] = max(r23["xMax"], float(curve_embed[:, 1].max()))
        r23["yMin"] = min(r23["yMin"], float(curve_embed[:, 2].min()))
        r23["yMax"] = max(r23["yMax"], float(curve_embed[:, 2].max()))
        # Per-record range for autozoom (current n's curve only)
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


def _merge_range(a: dict, b: dict) -> dict:
    return {
        "xMin": min(a["xMin"], b["xMin"]),
        "xMax": max(a["xMax"], b["xMax"]),
        "yMin": min(a["yMin"], b["yMin"]),
        "yMax": max(a["yMax"], b["yMax"]),
    }


def _precompute_tie_pca(fit_on_sorted: bool):
    """Fit PCA (4 components) on p in [0.5, last_tie]; then project full p in [0, 1]. Return (list of dicts, range_12, range_23)."""
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
        # Axis range from p-curve data only (exclude simplex vertices)
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
    data_full_unsorted: list,
    data_full_sorted: list,
    data_half_unsorted: list,
    data_half_sorted: list,
    data_tie_unsorted: list,
    data_tie_sorted: list,
    last_tie_by_n: dict,
) -> str:
    last_tie_json = json.dumps(last_tie_by_n)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
  <div id="graph" style="width: 90%; height: 575px; margin: 20px auto;"></div>
  <div style="margin: 20px auto; width: 500px;">
    <div style="margin-top: 10px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
      <span><label for="pca-range"><b>range for PCA</b></label>
      <select id="pca-range" style="margin-left: 6px;">
        <option value="full">0-1</option>
        <option value="half" selected>0.5-1</option>
        <option value="tie">0.5-last tie point</option>
      </select></span>
      <span><label for="pca-data"><b>data for PCA</b></label>
      <select id="pca-data" style="margin-left: 6px;">
        <option value="unsorted">unsorted</option>
        <option value="sorted">sorted</option>
      </select></span>
      <span><label for="pca-components"><b>projection components</b></label>
      <select id="pca-components" style="margin-left: 6px;">
        <option value="12" selected>1,2</option>
        <option value="23">2,3</option>
        <option value="34">3,4</option>
      </select></span>
    </div>
    <div style="margin-top: 10px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
      <span><label for="display-data"><b>display</b></label>
      <select id="display-data" style="margin-left: 6px;">
        <option value="unsorted">unsorted</option>
        <option value="sorted">sorted</option>
      </select></span>
      <span>
      <select id="display-range" style="margin-left: 6px;">
        <option value="full">0–1</option>
        <option value="half" selected>0.5–1</option>
        <option value="tie">0.5–last tie point</option>
      </select></span>
    </div>
    <div style="margin-top: 10px;">
      <label for="n-slider"><b>n</b> = <span id="n-value">{N_MIN}</span></label>
      <input type="range" id="n-slider" min="{N_MIN}" max="{N_MAX}" value="{N_MIN}" style="width: 100%;">
    </div>
    <div style="margin-top: 10px;">
      <label><input type="checkbox" id="simplex-points"> simplex points</label>
      <label style="margin-left: 14px;"><input type="checkbox" id="autozoom" checked> autozoom</label>
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

    function getDataIndex(n) {{
      return n - {N_MIN};
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

    function getMethodLabel() {{
      const range = document.getElementById("pca-range").value;
      const data = document.getElementById("pca-data").value;
      let r = range === "full" ? "0-1" : (range === "tie" ? "0.5-last tie" : "0.5-1");
      return "PCA (" + r + ", " + data + ")";
    }}

    function getComponentsView() {{
      return document.getElementById("pca-components").value;
    }}

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

    function getAxisTitles() {{
      const view = getComponentsView();
      if (view === "34") return {{ x: "PC3", y: "PC4" }};
      if (view === "23") return {{ x: "PC2", y: "PC3" }};
      return {{ x: "PC1", y: "PC2" }};
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

    const layout = {{
      title: {{ text: "PCA: n=2 (binomial)" }},
      xaxis: {{ title: "", scaleanchor: "y", scaleratio: 1, range: [-1, 1], showgrid: true, showticklabels: false }},
      yaxis: {{ title: "", range: [-1, 1], showgrid: true, showticklabels: false }},
      margin: {{ t: 60, r: 20, b: 50, l: 50 }},
      showlegend: false
    }};

    var proj = getProjectionData();
    var d = proj[getDataIndex({N_MIN})];
    var displayRange = document.getElementById("display-range").value;
    var displayData = document.getElementById("display-data").value;
    var componentsView = getComponentsView();
    var filtered = filterByPRange(d, {N_MIN}, displayRange);
    var xy = getXYFromFiltered(filtered, displayData, componentsView);
    var axisRange = getAxisRange(d, xy);
    var initialTraces = [{{
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
      initialTraces.push({{
        x: vxy.x,
        y: vxy.y,
        mode: "markers",
        marker: {{ size: 8, color: "red", symbol: "diamond" }},
        text: d.vertex_pc1.map(function(_, i) {{ return "k=" + i; }}),
        hovertemplate: "%{{text}}<extra></extra>",
        showlegend: false
      }});
    }}
    var autozoomOn = document.getElementById("autozoom").checked;
    var initialLayout = {{
      ...layout,
      xaxis: {{ ...layout.xaxis, range: axisRange.x, ...(autozoomOn ? {{ scaleanchor: false }} : {{}}) }},
      yaxis: {{ ...layout.yaxis, range: axisRange.y }}
    }};
    Plotly.newPlot("graph", initialTraces, initialLayout, {{ responsive: true }});

    function updatePlot() {{
      const n = parseInt(document.getElementById("n-slider").value, 10);
      document.getElementById("n-value").textContent = n;
      const proj = getProjectionData();
      const d = proj[getDataIndex(n)];
      const displayRange = document.getElementById("display-range").value;
      const displayData = document.getElementById("display-data").value;
      const componentsView = getComponentsView();
      const filtered = filterByPRange(d, n, displayRange);
      const xy = getXYFromFiltered(filtered, displayData, componentsView);
      const methodLabel = getMethodLabel();
      const axes = getAxisTitles();
      const axisRange = getAxisRange(d, xy);
      const autozoomOn = document.getElementById("autozoom").checked;
      const layoutUpdate = {{
        ...layout,
        title: {{ text: methodLabel + ": n=" + n + " (" + displayData + ")" }},
        xaxis: {{ ...layout.xaxis, title: axes.x, range: axisRange.x, ...(autozoomOn ? {{ scaleanchor: false }} : {{}}) }},
        yaxis: {{ ...layout.yaxis, title: axes.y, range: axisRange.y }}
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
          x: vxy.x,
          y: vxy.y,
          mode: "markers",
          marker: {{ size: 8, color: "red", symbol: "diamond" }},
          text: d.vertex_pc1.map(function(_, i) {{ return "k=" + i; }}),
          hovertemplate: "%{{text}}<extra></extra>",
          showlegend: false
        }});
      }}
      Plotly.react("graph", traces, layoutUpdate);
    }}

    document.getElementById("pca-range").addEventListener("change", updatePlot);
    document.getElementById("pca-data").addEventListener("change", updatePlot);
    document.getElementById("pca-components").addEventListener("change", updatePlot);
    document.getElementById("display-data").addEventListener("change", updatePlot);
    document.getElementById("display-range").addEventListener("change", updatePlot);
    document.getElementById("n-slider").addEventListener("input", updatePlot);
    document.getElementById("simplex-points").addEventListener("change", updatePlot);
    document.getElementById("autozoom").addEventListener("change", updatePlot);
  </script>
</body>
</html>
"""


def main(output_path: str = "OBD2Dprojection.html") -> None:
    data_full_unsorted, r_full_u_12, r_full_u_23 = _precompute_pca(0.0, 1.0, fit_on_sorted=False)
    data_full_sorted, r_full_s_12, r_full_s_23 = _precompute_pca(0.0, 1.0, fit_on_sorted=True)
    data_half_unsorted, r_half_u_12, r_half_u_23 = _precompute_pca(0.5, 1.0, fit_on_sorted=False)
    data_half_sorted, r_half_s_12, r_half_s_23 = _precompute_pca(0.5, 1.0, fit_on_sorted=True)
    print("Precomputing Tie point restricted PCA (unsorted, sorted)...")
    data_tie_unsorted, r_tie_u_12, r_tie_u_23 = _precompute_tie_pca(fit_on_sorted=False)
    data_tie_sorted, r_tie_s_12, r_tie_s_23 = _precompute_tie_pca(fit_on_sorted=True)
    last_tie_by_n = {n: _last_tie_above_half(n) for n in N_VALS}
    html = _build_html(
        data_full_unsorted, data_full_sorted,
        data_half_unsorted, data_half_sorted,
        data_tie_unsorted, data_tie_sorted,
        last_tie_by_n,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {output_path}. Open this file in a browser (no server needed).")


if __name__ == "__main__":
    main()
