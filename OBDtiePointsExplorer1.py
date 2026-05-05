"""
Generate ``OBDtiePointsExplorer1.html``: interactive tie-point segment plot (p vs n).

Loads ``data/tie_points.pkl`` (``float_with_pairs_by_n`` or legacy ``by_n``). Vertical segments
from ``(p, n-1)`` to ``(p, n)`` for ``p`` in a chosen window (``[0, 1]`` or ``[0.5, 1]``), ``n`` from 2 to 100.

Controls: **p range** (0–1 vs 0.5–1). **Color**: one menu for 0.5–1 mode; for **0–1** mode, two menus —
**Color (0–0.5)** and **Color (0.5–1)** — each with black / color by ``i`` / ``j`` / ``ij`` (per-``n`` scaling
within that half). ``ij`` splits each segment at ``n-0.5`` (bottom ``j``, top ``i``).

Create the pickle with ``OBDsaveSourceData.py --save-tie-points`` (or ``--all``).
"""
from __future__ import annotations

import json
import os
import pickle
import sys

from plotly.colors import sample_colorscale

# Match OBDgraphExplorer1
DATA_DIR = "data"
TIE_POINTS_PATH = os.path.join(DATA_DIR, "tie_points.pkl")
N_MIN_SEG = 2
N_MAX_SEG = 100
LINE_ALPHA = 0.35
OUTPUT_DEFAULT = "OBDtiePointsExplorer1.html"


def _cmd_save_tie_points() -> str:
    return (
        "python OBDsaveSourceData.py --save-tie-points "
        "--tie-n-min 2 --tie-n-max 100"
    )


def _load_float_with_pairs_by_n(path: str) -> dict[int, list]:
    if not os.path.isfile(path):
        print(
            f"WARNING: Tie points file not found:\n  {os.path.abspath(path)}\n\n"
            f"Create it with:\n  {_cmd_save_tie_points()}\n",
            file=sys.stderr,
        )
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(
            f"WARNING: Could not read tie points {path!r}: {e}\n"
            f"Regenerate with:\n  {_cmd_save_tie_points()}\n",
            file=sys.stderr,
        )
        return {}
    fwp = data.get("float_with_pairs_by_n") or data.get("by_n") or {}
    return {int(k): v for k, v in fwp.items()}


def _tie_data_for_json(by_n: dict[int, list]) -> dict[str, list]:
    """JSON-safe: str(n) -> [[p, [[i,j], ...]], ...]."""
    out: dict[str, list] = {}
    for n, recs in by_n.items():
        row = []
        for item in recs:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            p, pairs = item[0], item[1]
            pp = float(p)
            plist = []
            for ij in pairs:
                if not isinstance(ij, (list, tuple)) or len(ij) != 2:
                    continue
                plist.append([int(ij[0]), int(ij[1])])
            row.append([pp, plist])
        out[str(int(n))] = row
    return out


def _turbo_lut_256() -> list[list[int]]:
    lut: list[list[int]] = []
    for k in range(256):
        t = k / 255.0
        s = sample_colorscale("Turbo", [t], colortype="rgb")[0]
        inner = s[s.index("(") + 1 : s.index(")")].replace(" ", "")
        parts = inner.split(",")
        lut.append([int(parts[0]), int(parts[1]), int(parts[2])])
    return lut


def _build_html(tie_json_str: str, turbo_lut_json: str) -> str:
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
      align-items: center;
      gap: 12px;
      padding: 10px 12px 12px;
    }}
    .controls label {{ display: inline-flex; align-items: center; gap: 8px; }}
    .color-dual {{ display: inline-flex; align-items: center; gap: 8px; }}
  </style>
</head>
<body>
  <div class="container">
    <div id="graph" class="graph-wrap"></div>
    <div class="controls">
      <label for="p-range">p range</label>
      <select id="p-range" aria-label="Horizontal p window">
        <option value="full">0 to 1</option>
        <option value="half" selected>0.5 to 1</option>
      </select>
      <span id="color-wrap-low" class="color-dual" style="display: none;">
        <label for="color-mode-low">Color (0–0.5)</label>
        <select id="color-mode-low" aria-label="Color for p in [0, 0.5)">
          <option value="black">black</option>
          <option value="by_i" selected>color by i</option>
          <option value="by_j">color by j</option>
          <option value="by_ij">color by ij</option>
        </select>
      </span>
      <span id="color-wrap-high">
        <label for="color-mode-high" id="color-label-high">Color</label>
        <select id="color-mode-high" aria-label="Segment color mode">
          <option value="black">black</option>
          <option value="by_i" selected>color by i</option>
          <option value="by_j">color by j</option>
          <option value="by_ij">color by ij</option>
        </select>
      </span>
    </div>
  </div>
  <script>
    const TIE_BY_N = {tie_json_str};
    const TURBO_LUT = {turbo_lut_json};
    const N_LO = {N_MIN_SEG};
    const N_HI = {N_MAX_SEG};
    const ALPHA = {LINE_ALPHA};

    function getPRange() {{
      var v = document.getElementById("p-range").value;
      if (v === "full") return {{ lo: 0, hi: 1 }};
      return {{ lo: 0.5, hi: 1 }};
    }}

    function syncColorUi() {{
      var full = document.getElementById("p-range").value === "full";
      var lowWrap = document.getElementById("color-wrap-low");
      var lab = document.getElementById("color-label-high");
      lowWrap.style.display = full ? "inline-flex" : "none";
      lab.textContent = full ? "Color (0.5–1)" : "Color";
    }}

    function turboRgba(t) {{
      var tt = Math.max(0, Math.min(1, t));
      var idx = Math.min(255, Math.floor(tt * 255));
      var c = TURBO_LUT[idx];
      return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + ALPHA + ")";
    }}

    function getRecs(n) {{
      var k = String(n);
      return TIE_BY_N[k] || [];
    }}

    /** band: "L" = [0, 0.5), "R" or "H" = [0.5, 1] */
    function collectSegs(n2, band) {{
      var recs2 = getRecs(n2);
      var segs = [];
      for (var rj = 0; rj < recs2.length; rj++) {{
        var p2 = recs2[rj][0];
        var pairs2 = recs2[rj][1];
        if (!isFinite(p2)) continue;
        var ok = false;
        if (band === "L") ok = p2 >= 0 && p2 < 0.5;
        else ok = p2 >= 0.5 && p2 <= 1;
        if (!ok) continue;
        for (var pj = 0; pj < pairs2.length; pj++) {{
          var pr = pairs2[pj];
          if (!pr || pr.length !== 2) continue;
          segs.push({{ p: p2, i: pr[0], j: pr[1] }});
        }}
      }}
      return segs;
    }}

    function addColoredBucketsForSegs(buckets, n2, segs, colorMode) {{
      if (!segs.length) return;
      if (colorMode === "by_ij") {{
        var i_lo = i_hi = segs[0].i;
        var j_lo = j_hi = segs[0].j;
        for (var si = 1; si < segs.length; si++) {{
          if (segs[si].i < i_lo) i_lo = segs[si].i;
          if (segs[si].i > i_hi) i_hi = segs[si].i;
          if (segs[si].j < j_lo) j_lo = segs[si].j;
          if (segs[si].j > j_hi) j_hi = segs[si].j;
        }}
        var yMid = n2 - 0.5;
        for (var sij = 0; sij < segs.length; sij++) {{
          var sg = segs[sij];
          var t_i = i_hi > i_lo ? (sg.i - i_lo) / (i_hi - i_lo) : 0;
          var t_j = j_hi > j_lo ? (sg.j - j_lo) / (j_hi - j_lo) : 0;
          var rgbaTop = turboRgba(t_i);
          var rgbaBot = turboRgba(t_j);
          if (!buckets[rgbaBot]) buckets[rgbaBot] = {{ xs: [], ys: [] }};
          buckets[rgbaBot].xs.push(sg.p, sg.p, NaN);
          buckets[rgbaBot].ys.push(n2 - 1, yMid, NaN);
          if (!buckets[rgbaTop]) buckets[rgbaTop] = {{ xs: [], ys: [] }};
          buckets[rgbaTop].xs.push(sg.p, sg.p, NaN);
          buckets[rgbaTop].ys.push(yMid, n2, NaN);
        }}
        return;
      }}
      var lo, hi, key;
      if (colorMode === "by_j") {{
        lo = hi = segs[0].j;
        for (var s = 1; s < segs.length; s++) {{
          if (segs[s].j < lo) lo = segs[s].j;
          if (segs[s].j > hi) hi = segs[s].j;
        }}
        key = "j";
      }} else {{
        lo = hi = segs[0].i;
        for (var s2 = 1; s2 < segs.length; s2++) {{
          if (segs[s2].i < lo) lo = segs[s2].i;
          if (segs[s2].i > hi) hi = segs[s2].i;
        }}
        key = "i";
      }}
      for (var s3 = 0; s3 < segs.length; s3++) {{
        var seg = segs[s3];
        var v = key === "j" ? seg.j : seg.i;
        var t = hi > lo ? (v - lo) / (hi - lo) : 0;
        var rgba = turboRgba(t);
        if (!buckets[rgba]) buckets[rgba] = {{ xs: [], ys: [] }};
        buckets[rgba].xs.push(seg.p, seg.p, NaN);
        buckets[rgba].ys.push(n2 - 1, n2, NaN);
      }}
    }}

    function bucketsToTraces(buckets) {{
      var traces = [];
      var rgbs = Object.keys(buckets).sort();
      for (var b = 0; b < rgbs.length; b++) {{
        var bk = rgbs[b];
        var g = buckets[bk];
        traces.push({{
          x: g.xs, y: g.ys, mode: "lines",
          line: {{ color: bk, width: 1 }},
          showlegend: false, hoverinfo: "skip"
        }});
      }}
      return traces;
    }}

    function buildOneBand(colorMode, band) {{
      if (colorMode === "black") {{
        var xs = [], ys = [];
        for (var n = N_LO; n <= N_HI; n++) {{
          var segsB = collectSegs(n, band);
          for (var q = 0; q < segsB.length; q++) {{
            var p = segsB[q].p;
            xs.push(p, p, NaN);
            ys.push(n - 1, n, NaN);
          }}
        }}
        if (!xs.length) return [];
        return [{{
          x: xs, y: ys, mode: "lines",
          line: {{ color: "rgba(0,0,0," + ALPHA + ")", width: 1 }},
          showlegend: false, hoverinfo: "skip"
        }}];
      }}
      var buckets = {{}};
      for (var n2 = N_LO; n2 <= N_HI; n2++) {{
        addColoredBucketsForSegs(buckets, n2, collectSegs(n2, band), colorMode);
      }}
      return bucketsToTraces(buckets);
    }}

    function buildAllSegments() {{
      var isFull = document.getElementById("p-range").value === "full";
      var modeHi = document.getElementById("color-mode-high").value;
      if (!isFull) return buildOneBand(modeHi, "R");
      var modeLo = document.getElementById("color-mode-low").value;
      return buildOneBand(modeLo, "L").concat(buildOneBand(modeHi, "R"));
    }}

    function modeLabel(m) {{
      if (m === "black") return "black";
      if (m === "by_j") return "Turbo j";
      if (m === "by_ij") return "Turbo ij";
      return "Turbo i";
    }}

    function titleForGraph() {{
      var isFull = document.getElementById("p-range").value === "full";
      var pr = getPRange();
      var pLo = pr.lo, pHi = pr.hi;
      var modeHi = document.getElementById("color-mode-high").value;
      var prs = "p ∈ [" + pLo + ", " + pHi + "]";
      if (!isFull) {{
        if (modeHi === "black") return "Tie points (" + prs + ", n = 2 … 100) — black";
        if (modeHi === "by_j") return "Tie points (" + prs + ") — Turbo, j scaled per n";
        if (modeHi === "by_ij") return "Tie points (" + prs + ") — bottom j / top i, scaled per n";
        return "Tie points (" + prs + ") — Turbo, i scaled per n";
      }}
      var modeLo = document.getElementById("color-mode-low").value;
      return "Tie points p ∈ [0, 1] — [0, 0.5): " + modeLabel(modeLo) + "; [0.5, 1]: " + modeLabel(modeHi) + " (each scaled per n)";
    }}

    function updateGraph() {{
      syncColorUi();
      var pr = getPRange();
      var pLo = pr.lo, pHi = pr.hi;
      var traces = buildAllSegments();
      var layout = {{
        title: {{ text: titleForGraph() }},
        xaxis: {{ title: "p", range: [pLo, pHi], constrain: "domain" }},
        yaxis: {{ title: "n", range: [1, N_HI], constrain: "domain" }},
        margin: {{ t: 50, r: 30, b: 50, l: 55 }}
      }};
      Plotly.react("graph", traces.length ? traces : [{{ x: [], y: [], type: "scatter", mode: "markers" }}], layout);
    }}

    document.getElementById("color-mode-low").addEventListener("change", updateGraph);
    document.getElementById("color-mode-high").addEventListener("change", updateGraph);
    document.getElementById("p-range").addEventListener("change", updateGraph);
    syncColorUi();
    updateGraph();
  </script>
</body>
</html>
"""


def main(
    output_path: str = OUTPUT_DEFAULT,
    tie_points_path: str = TIE_POINTS_PATH,
) -> None:
    print(f"Loading tie points from {tie_points_path}...")
    by_n = _load_float_with_pairs_by_n(tie_points_path)
    if by_n:
        print(f"  Loaded pair records for n in {{{', '.join(str(n) for n in sorted(by_n)[:8])}}}… ({len(by_n)} keys)")
    else:
        print("  No pair data; HTML will show an empty plot until tie_points.pkl is available.")
    tie_json = json.dumps(_tie_data_for_json(by_n))
    turbo_json = json.dumps(_turbo_lut_256())
    html = _build_html(tie_json, turbo_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {output_path}. Open in a browser (no server needed).")


if __name__ == "__main__":
    main()
