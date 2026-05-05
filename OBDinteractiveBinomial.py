"""
Interactive bar chart of Binomial(n, p) in the browser.
Precomputed PMFs embedded in HTML; two sliders (n, p) update the chart via JavaScript.
No server — open the generated HTML file from disk.
"""
import json

import numpy as np
from scipy.stats import binom

N_MIN, N_MAX = 2, 101
N_VALS = list(range(N_MIN, N_MAX + 1))
# 501 steps so middle index 250 gives p = 0.5 exactly
P_STEPS = 501
p_values = np.linspace(0.0, 1.0, P_STEPS)


def _precompute_data():
    """Return list of {x: [0..n], y: [PMF], perm: sort permutation (indices for low-to-high)} for each (n, p). Index = (n - N_MIN) * P_STEPS + p_idx."""
    out = []
    for n in N_VALS:
        k = np.arange(n + 1, dtype=int)
        for p in p_values:
            pmf = binom.pmf(k, n, p)
            # Round for smaller JSON; 6 decimals is plenty for display
            y = [round(float(v), 6) for v in pmf]
            # perm: indices that sort y ascending (left-to-right = lowest to highest); stable so equal probs (e.g. zeros at p=0/1) keep natural order
            perm = np.argsort(pmf, kind="stable").tolist()
            out.append({"x": k.tolist(), "y": y, "perm": perm})
    return out


def _build_html(data: list) -> str:
    data_json = json.dumps(data)
    # p_idx 0..(P_STEPS-1) -> p from 0 to 1
    p_labels_json = json.dumps([round(p, 4) for p in p_values])
    p_default_idx = P_STEPS // 2  # index for p ≈ 0.5
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
  <div id="graph" style="width: 90%; height: 575px; margin: 20px auto;"></div>
  <div style="margin: 20px auto; width: 500px;">
    <div style="margin-top: 10px;">
      <label for="n-slider"><b>n</b> = <span id="n-value">2</span></label>
      <input type="range" id="n-slider" min="{N_MIN}" max="{N_MAX}" value="2" style="width: 100%;">
    </div>
    <div style="margin-top: 10px;">
      <label for="p-slider"><b>p</b> = <span id="p-value">0.50</span></label>
      <input type="range" id="p-slider" min="0" max="{P_STEPS - 1}" value="{p_default_idx}" style="width: 100%;">
    </div>
    <div style="margin-top: 14px;">
      <button type="button" id="sort-btn">Sort bars (low to high)</button>
    </div>
  </div>

  <script>
    const BINOMIAL_DATA = {data_json};
    const P_LABELS = {p_labels_json};
    let sortBars = false;

    function getIndex(n, pIdx) {{
      return (n - {N_MIN}) * {P_STEPS} + pIdx;
    }}

    function colorForK(k, n) {{
      var t = n > 0 ? k / n : 0;
      var hue = Math.round(240 * (1 - t));
      return "hsl(" + hue + ", 70%, 50%)";
    }}

    const layout = {{
      title: {{ text: "Binomial(n=2, p=0.50)" }},
      xaxis: {{ title: "k", dtick: 1, showticklabels: false }},
      yaxis: {{ title: "P(X = k)", range: [-0.15, 1] }},
      margin: {{ t: 60, r: 20, b: 50, l: 50 }},
      bargap: 0,
      bargroupgap: 0,
      showlegend: false
    }};

    const initial = BINOMIAL_DATA[getIndex(2, {p_default_idx})];
    var initialColors = initial.x.map(function(k) {{ return colorForK(k, initial.x.length - 1); }});
    var stripY = initial.x.map(function() {{ return -0.1; }});
    Plotly.newPlot("graph", [
      {{
        x: initial.x,
        y: stripY,
        type: "bar",
        marker: {{ color: initialColors }},
        width: 1,
        offsetgroup: "bars",
        alignmentgroup: "bars"
      }},
      {{
        x: initial.x,
        y: initial.y,
        type: "bar",
        marker: {{ color: initialColors }},
        width: 0.5,
        offsetgroup: "bars",
        alignmentgroup: "bars"
      }}
    ], layout, {{ responsive: true }});
    document.getElementById("p-value").textContent = P_LABELS[{p_default_idx}].toFixed(4);

    function updatePlot() {{
      const n = parseInt(document.getElementById("n-slider").value, 10);
      const pIdx = parseInt(document.getElementById("p-slider").value, 10);
      document.getElementById("n-value").textContent = n;
      document.getElementById("p-value").textContent = P_LABELS[pIdx].toFixed(4);

      const point = BINOMIAL_DATA[getIndex(n, pIdx)];
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
      var layoutUpdate = {{ ...layout, title: {{ text: "Binomial(n=" + n + ", p=" + P_LABELS[pIdx].toFixed(4) + ")" }} }};
      if (Object.keys(xaxisOverride).length) layoutUpdate.xaxis = {{ ...layout.xaxis, ...xaxisOverride, showticklabels: false }};
      var stripY = xData.map(function() {{ return -0.1; }});
      Plotly.react("graph", [
        {{
          x: xData,
          y: stripY,
          type: "bar",
          marker: {{ color: colors }},
          width: 1,
          offsetgroup: "bars",
          alignmentgroup: "bars"
        }},
        {{
          x: xData,
          y: yData,
          type: "bar",
          marker: {{ color: colors }},
          width: 0.5,
          offsetgroup: "bars",
          alignmentgroup: "bars"
        }}
      ], layoutUpdate);
    }}

    function toggleSort() {{
      sortBars = !sortBars;
      document.getElementById("sort-btn").textContent = sortBars ? "Show by k (unsort)" : "Sort bars (low to high)";
      updatePlot();
    }}

    document.getElementById("n-slider").addEventListener("input", updatePlot);
    document.getElementById("p-slider").addEventListener("input", updatePlot);
    document.getElementById("sort-btn").addEventListener("click", toggleSort);
  </script>
</body>
</html>
"""


def main(output_path: str = "OBDinteractiveBinomial.html") -> None:
    data = _precompute_data()
    html = _build_html(data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {output_path}. Open this file in a browser (no server needed).")


# Run this file to generate the HTML, then open the file in a browser:
#   python OBDinteractiveBinomial.py
if __name__ == "__main__":
    main()
