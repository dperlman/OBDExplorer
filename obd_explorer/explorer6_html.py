"""Plotly explorer #6 HTML: tie scalar vs tie index (N chosen via dual slider + Multiple)."""

from __future__ import annotations

import base64
import json
from matplotlib import colormaps
from matplotlib.colors import to_hex
import numpy as np

from obd_explorer.html_data import EXPLORER5_MAX_TIE_INDEX


_TIE_PACK_FIELDS = ("p", "i", "j", "l", "r", "d", "e", "ev_n")


def _pack_float32_base64(values: object) -> str:
    if not isinstance(values, (list, tuple)) or not values:
        return ""
    arr = np.empty(len(values), dtype=np.float32)
    for idx, raw in enumerate(values):
        arr[idx] = np.float32(np.nan if raw is None else raw)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _pack_tie_payload_float32_base64(tie_data: dict[str, object]) -> dict[str, dict[str, str]]:
    packed: dict[str, dict[str, str]] = {}
    for n_key, payload in tie_data.items():
        if not isinstance(payload, dict):
            continue
        packed[str(n_key)] = {field: _pack_float32_base64(payload.get(field)) for field in _TIE_PACK_FIELDS}
    return packed


def build_explorer6_html(
    tie_data: dict[str, object],
    *,
    n_min: int,
    n_max: int,
    colorscale: str = "viridis",
) -> str:
    tie_packed_json = json.dumps(_pack_tie_payload_float32_base64(tie_data), separators=(",", ":"))
    color_lut_json = json.dumps([to_hex(colormaps[colorscale](i / 255.0), keep_alpha=False) for i in range(256)])
    tm = EXPLORER5_MAX_TIE_INDEX
    ui_max = min(999, tm)
    tie_default_hi = ui_max
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
      gap: 4px 6px;
      padding: 4px 6px 5px;
      overflow-x: auto;
      overflow-y: hidden;
      white-space: nowrap;
    }}
    .controls > * {{ flex: 0 0 auto; }}
    .controls label {{ display: inline-flex; align-items: center; gap: 3px; cursor: pointer; font-size: 13px; }}
    .controls select#field {{ font-size: 13px; }}
    #ctrl-tie-range .obd-dual-slider, #ctrl-n-range .obd-dual-slider {{
      position: relative;
      height: 14px;
      border-radius: 10px;
      text-align: left;
      margin: 2px 0;
      width: 240px;
      max-width: 85vw;
      flex: 0 0 auto;
    }}
    #ctrl-n-range .obd-dual-slider {{
      width: 320px;
    }}
    #ctrl-tie-range .obd-dual-slider-trackwrap, #ctrl-n-range .obd-dual-slider-trackwrap {{
      position: absolute;
      left: 13px;
      right: 15px;
      height: 14px;
    }}
    #ctrl-tie-range .obd-dual-inverse-left, #ctrl-n-range .obd-dual-inverse-left {{
      position: absolute;
      left: 0;
      height: 10px;
      border-radius: 10px;
      background-color: #ccc;
      margin: 0 7px;
    }}
    #ctrl-tie-range .obd-dual-inverse-right, #ctrl-n-range .obd-dual-inverse-right {{
      position: absolute;
      right: 0;
      height: 10px;
      border-radius: 10px;
      background-color: #ccc;
      margin: 0 7px;
    }}
    #ctrl-tie-range .obd-dual-range, #ctrl-n-range .obd-dual-range {{
      position: absolute;
      left: 0;
      top: -1px;
      height: 16px;
      border-radius: 14px;
      background-color: #31689b;
    }}
    #ctrl-tie-range .obd-dual-thumb, #ctrl-n-range .obd-dual-thumb {{
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
    #ctrl-tie-range .obd-dual-slider > input[type=range], #ctrl-n-range .obd-dual-slider > input[type=range] {{
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
    #ctrl-tie-range .obd-dual-slider > input[type=range]::-webkit-slider-thumb, #ctrl-n-range .obd-dual-slider > input[type=range]::-webkit-slider-thumb {{
      pointer-events: all;
      width: 24px;
      height: 24px;
      border: 0 none;
      border-radius: 0;
      background: transparent;
      -webkit-appearance: none;
    }}
    #ctrl-tie-range .obd-dual-slider > input[type=range]::-moz-range-thumb, #ctrl-n-range .obd-dual-slider > input[type=range]::-moz-range-thumb {{
      pointer-events: all;
      width: 24px;
      height: 24px;
      border: 0 none;
      border-radius: 0;
      background: transparent;
    }}
    #ctrl-tie-range .obd-dual-slider > input[type=range]::-ms-thumb, #ctrl-n-range .obd-dual-slider > input[type=range]::-ms-thumb {{
      pointer-events: all;
      width: 24px;
      height: 24px;
      border: 0 none;
      border-radius: 0;
      background: transparent;
    }}
    #ctrl-tie-range .obd-dual-slider > input[type=range]::-ms-fill-lower,
    #ctrl-tie-range .obd-dual-slider > input[type=range]::-ms-fill-upper,
    #ctrl-n-range .obd-dual-slider > input[type=range]::-ms-fill-lower,
    #ctrl-n-range .obd-dual-slider > input[type=range]::-ms-fill-upper {{
      background: transparent;
      border: 0 none;
    }}
    #ctrl-tie-range .obd-dual-slider > input[type=range]::-ms-track, #ctrl-n-range .obd-dual-slider > input[type=range]::-ms-track {{
      background: transparent;
      color: transparent;
    }}
    #ctrl-tie-range .obd-dual-slider > input[type=range]::-moz-range-track, #ctrl-n-range .obd-dual-slider > input[type=range]::-moz-range-track {{
      background: transparent;
      color: transparent;
      -moz-appearance: none;
    }}
    #ctrl-tie-range .obd-dual-slider > input[type=range]:focus, #ctrl-n-range .obd-dual-slider > input[type=range]:focus {{
      outline: none;
    }}
    #ctrl-tie-range .obd-dual-slider > input[type=range]:focus::-webkit-slider-runnable-track, #ctrl-n-range .obd-dual-slider > input[type=range]:focus::-webkit-slider-runnable-track {{
      background: transparent;
      border: transparent;
    }}
    #ctrl-tie-range .obd-dual-slider > input[type=range]::-ms-tooltip, #ctrl-n-range .obd-dual-slider > input[type=range]::-ms-tooltip {{
      display: none;
    }}
    .ctrl-panel {{ display: flex; align-items: center; gap: 4px; }}
    #ctrl-tie, #ctrl-n {{ align-items: center; flex-wrap: nowrap; min-width: 0; flex: 0 0 auto; max-width: none; }}
    #ctrl-tie-range, #ctrl-n-range {{
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: 4px;
      overflow: visible;
      flex-shrink: 0;
      max-width: none;
    }}
    #ctrl-tie-range .obd-tie-range-label, #ctrl-n-range .obd-n-range-label {{
      display: inline-flex;
      align-items: center;
      gap: 2px;
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .range-num {{
      width: 4.8em;
      min-width: 4.8em;
      font-variant-numeric: tabular-nums;
    }}
    .range-to {{
      font-size: 13px;
    }}
    #ctrl-n-range.obd-dual-single-mode .obd-dual-thumb-right {{
      visibility: hidden;
      pointer-events: none;
    }}
    #ctrl-n-range.obd-dual-single-mode #n-range-high {{
      pointer-events: none;
      z-index: 2;
    }}
    #ctrl-n-range.obd-dual-single-mode #n-range-low {{
      z-index: 4;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div id="graph" class="graph-wrap"></div>
    <div class="controls">
      <label for="field">Plot:</label>
      <select id="field" aria-label="Tie value to plot">
        <option value="i">i</option>
        <option value="j">j</option>
        <option value="l">l (slope left)</option>
        <option value="r">r (slope right)</option>
        <option value="d">d (r − l)</option>
        <option value="e">e (l − r)</option>
        <option value="p">p (tie)</option>
        <option value="ev_n">ev/n</option>
      </select>
      <label><input type="checkbox" id="freeze-y"> Freeze y</label>
      <label><input type="checkbox" id="end-ties"> End ties</label>
      <label id="multiple-label" for="multiple"><input type="checkbox" id="multiple"> Multiple</label>
      <div id="ctrl-n" class="ctrl-panel">
        <div id="ctrl-n-range" class="obd-dual-single-mode">
          <div id="n-dual-range" class="obd-dual-slider" aria-label="n value or range">
            <div class="obd-dual-slider-trackwrap">
              <div class="obd-dual-inverse-left"></div>
              <div class="obd-dual-inverse-right"></div>
              <div class="obd-dual-range"></div>
              <span class="obd-dual-thumb obd-dual-thumb-left" aria-hidden="true"></span>
              <span class="obd-dual-thumb obd-dual-thumb-right" aria-hidden="true"></span>
            </div>
            <input type="range" id="n-range-low" min="{n_min}" max="{n_max}" step="1" value="{n_min}" aria-label="Minimum n">
            <input type="range" id="n-range-high" min="{n_min}" max="{n_max}" step="1" value="{n_min}" aria-label="Maximum n">
          </div>
          <label for="n-range-low" class="obd-n-range-label"><b>N=</b></label>
          <input type="number" id="n-num-low" class="range-num" min="{n_min}" max="{n_max}" step="1" value="{n_min}" aria-label="N minimum">
          <span class="range-to">to</span>
          <input type="number" id="n-num-high" class="range-num" min="{n_min}" max="{n_max}" step="1" value="{n_min}" aria-label="N maximum" disabled>
        </div>
      </div>
      <div id="ctrl-tie" class="ctrl-panel">
        <div id="ctrl-tie-range">
          <div id="tie-dual-range" class="obd-dual-slider" aria-label="Tie index range">
            <div class="obd-dual-slider-trackwrap">
              <div class="obd-dual-inverse-left"></div>
              <div class="obd-dual-inverse-right"></div>
              <div class="obd-dual-range"></div>
              <span class="obd-dual-thumb obd-dual-thumb-left" aria-hidden="true"></span>
              <span class="obd-dual-thumb obd-dual-thumb-right" aria-hidden="true"></span>
            </div>
            <input type="range" id="tie-range-low" min="0" max="{ui_max}" step="1" value="0" aria-label="Minimum tie index">
            <input type="range" id="tie-range-high" min="0" max="{ui_max}" step="1" value="{tie_default_hi}" aria-label="Maximum tie index">
          </div>
          <label for="tie-range-low" class="obd-tie-range-label"><b>Tie #</b></label>
          <input type="number" id="tie-num-low" class="range-num" min="0" max="{ui_max}" step="1" value="0" aria-label="Tie minimum index">
          <span class="range-to">to</span>
          <input type="number" id="tie-num-high" class="range-num" min="0" max="{ui_max}" step="1" value="{tie_default_hi}" aria-label="Tie maximum index">
        </div>
      </div>
      <label><input type="checkbox" id="log-x"> Log x</label>
      <label><input type="checkbox" id="log-y"> Log y</label>
      <label><input type="checkbox" id="show-points" checked> Points</label>
    </div>
  </div>

  <script>
    const TIE_DATA_PACKED = {tie_packed_json};
    const TIE_IDX_MAX = {tm};
    const USER_TIE_MAX = {ui_max};
    const COLOR_LUT = {color_lut_json};
    const TIE_FIELDS = ["p", "i", "j", "l", "r", "d", "e", "ev_n"];

    function decodeBase64Float32(b64) {{
      if (!b64) return new Float32Array(0);
      var raw = atob(b64);
      var bytes = new Uint8Array(raw.length);
      for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      return new Float32Array(bytes.buffer);
    }}

    function decodePackedTieData(src) {{
      var out = {{}};
      for (var nKey in src) {{
        if (!Object.prototype.hasOwnProperty.call(src, nKey)) continue;
        var row = src[nKey] || {{}};
        var decoded = {{}};
        for (var i = 0; i < TIE_FIELDS.length; i++) {{
          var field = TIE_FIELDS[i];
          decoded[field] = decodeBase64Float32(row[field] || "");
        }}
        out[nKey] = decoded;
      }}
      return out;
    }}

    const TIE_DATA = decodePackedTieData(TIE_DATA_PACKED);
    function colorFromLut(t) {{
      var u = Math.max(0, Math.min(1, t));
      var idx = Math.round(u * (COLOR_LUT.length - 1));
      return COLOR_LUT[idx];
    }}

    var nRangeLo = {n_min}, nRangeHi = {n_min};
    var tieRangeLo = 0, tieRangeHi = {tie_default_hi};
    var frozenYRange = null;

    function fieldLabel() {{
      var sel = document.getElementById("field");
      return sel ? sel.options[sel.selectedIndex].text : "";
    }}

    function nMultiple() {{
      var el = document.getElementById("multiple");
      return el && el.checked;
    }}

    function endTiesMode() {{
      var el = document.getElementById("end-ties");
      return el && el.checked;
    }}

    function tieRowLen(s) {{
      if (!s) return 0;
      var p = s.p;
      if (p && p.length) return p.length;
      var f = document.getElementById("field").value;
      var arr = s[f];
      return arr && arr.length ? arr.length : 0;
    }}

    function tiePhysicalIndex(s, uiIdx, endTies) {{
      var len = tieRowLen(s);
      if (len <= 0) return -1;
      if (!endTies) {{
        if (uiIdx < 0 || uiIdx >= len) return -1;
        return uiIdx;
      }}
      if (uiIdx < 0 || uiIdx > USER_TIE_MAX) return -1;
      var offsetFromEnd = uiIdx;
      var phys = len - 1 - offsetFromEnd;
      return phys >= 0 ? phys : -1;
    }}

    function tieShownFromUi(uiIdx, endTies) {{
      return endTies ? (uiIdx + 1) : uiIdx;
    }}

    function tieUiFromShown(shownIdx, endTies) {{
      return endTies ? (shownIdx - 1) : shownIdx;
    }}

    function clampInt(v, lo, hi, fallback) {{
      var x = parseInt(v, 10);
      if (!isFinite(x)) return fallback;
      if (x < lo) return lo;
      if (x > hi) return hi;
      return x;
    }}

    function syncNNumberInputs() {{
      var lo = document.getElementById("n-num-low");
      var hi = document.getElementById("n-num-high");
      if (!lo || !hi) return;
      var a = Math.min(nRangeLo, nRangeHi);
      var b = Math.max(nRangeLo, nRangeHi);
      lo.value = String(a);
      hi.value = String(b);
      hi.disabled = !nMultiple();
    }}

    function syncTieNumberInputs() {{
      var lo = document.getElementById("tie-num-low");
      var hi = document.getElementById("tie-num-high");
      if (!lo || !hi) return;
      var endLast = endTiesMode();
      var shownMin = endLast ? 1 : 0;
      var shownMax = endLast ? (USER_TIE_MAX + 1) : USER_TIE_MAX;
      var a = Math.min(tieRangeLo, tieRangeHi);
      var b = Math.max(tieRangeLo, tieRangeHi);
      var sa = tieShownFromUi(a, endLast);
      var sb = tieShownFromUi(b, endLast);
      lo.value = String(Math.min(sa, sb));
      hi.value = String(Math.max(sa, sb));
      lo.min = String(shownMin);
      lo.max = String(shownMax);
      hi.min = String(shownMin);
      hi.max = String(shownMax);
    }}

    function obdNDualRangePct(v) {{
      var span = {n_max} - {n_min};
      if (span <= 0) return 0;
      return ((v - {n_min}) / span) * 100;
    }}

    function applyObdNDualRangeVisual() {{
      var wrap = document.querySelector("#n-dual-range .obd-dual-slider-trackwrap");
      if (!wrap) return;
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10);
      var hi = parseInt(hiIn.value, 10);
      var pLo = obdNDualRangePct(lo);
      var pHi = obdNDualRangePct(hi);
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

    function syncObdNDualRangeFromGlobals() {{
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!loIn || !hiIn) return;
      if (nMultiple()) {{
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
      applyObdNDualRangeVisual();
      syncNNumberInputs();
    }}

    function onNRangeLowInput() {{
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10);
      var hi = parseInt(hiIn.value, 10);
      if (!nMultiple()) {{
        hi = lo;
        hiIn.value = String(hi);
      }} else if (lo > hi) {{
        lo = hi;
        loIn.value = String(lo);
      }}
      nRangeLo = lo;
      nRangeHi = hi;
      applyObdNDualRangeVisual();
      syncNNumberInputs();
      updateGraph();
    }}

    function onNRangeHighInput() {{
      if (!nMultiple()) return;
      var loIn = document.getElementById("n-range-low");
      var hiIn = document.getElementById("n-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10);
      var hi = parseInt(hiIn.value, 10);
      if (hi < lo) {{ hi = lo; hiIn.value = String(hi); }}
      nRangeLo = lo;
      nRangeHi = hi;
      applyObdNDualRangeVisual();
      syncNNumberInputs();
      updateGraph();
    }}

    function onNNumLowInput() {{
      var lo = clampInt(document.getElementById("n-num-low").value, {n_min}, {n_max}, nRangeLo);
      var hi = nRangeHi;
      if (!nMultiple()) {{
        hi = lo;
      }} else if (lo > hi) {{
        hi = lo;
      }}
      nRangeLo = lo;
      nRangeHi = hi;
      syncObdNDualRangeFromGlobals();
      updateGraph();
    }}

    function onNNumHighInput() {{
      if (!nMultiple()) return;
      var lo = nRangeLo;
      var hi = clampInt(document.getElementById("n-num-high").value, {n_min}, {n_max}, nRangeHi);
      if (hi < lo) hi = lo;
      nRangeHi = hi;
      syncObdNDualRangeFromGlobals();
      updateGraph();
    }}

    function syncNMultipleMode() {{
      var wrap = document.getElementById("ctrl-n-range");
      if (!wrap) return;
      if (nMultiple()) wrap.classList.remove("obd-dual-single-mode");
      else wrap.classList.add("obd-dual-single-mode");
      syncObdNDualRangeFromGlobals();
    }}

    function obdTieDualRangePct(v) {{
      if (USER_TIE_MAX <= 0) return 0;
      return (v / USER_TIE_MAX) * 100;
    }}

    function applyObdTieDualRangeVisual() {{
      var wrap = document.querySelector("#tie-dual-range .obd-dual-slider-trackwrap");
      if (!wrap) return;
      var loIn = document.getElementById("tie-range-low");
      var hiIn = document.getElementById("tie-range-high");
      if (!loIn || !hiIn) return;
      var lo = parseInt(loIn.value, 10);
      var hi = parseInt(hiIn.value, 10);
      var pLo = obdTieDualRangePct(lo);
      var pHi = obdTieDualRangePct(hi);
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

    function syncObdTieDualRangeFromGlobals() {{
      var loIn = document.getElementById("tie-range-low");
      var hiIn = document.getElementById("tie-range-high");
      if (!loIn || !hiIn) return;
      tieRangeLo = clampInt(tieRangeLo, 0, USER_TIE_MAX, 0);
      tieRangeHi = clampInt(tieRangeHi, 0, USER_TIE_MAX, tieRangeLo);
      var a = Math.min(tieRangeLo, tieRangeHi), b = Math.max(tieRangeLo, tieRangeHi);
      loIn.value = String(a);
      hiIn.value = String(b);
      applyObdTieDualRangeVisual();
      syncTieNumberInputs();
    }}

    function onTieRangeLowInput() {{
      var loIn = document.getElementById("tie-range-low");
      var hiIn = document.getElementById("tie-range-high");
      if (!loIn || !hiIn) return;
      var lo = clampInt(loIn.value, 0, USER_TIE_MAX, tieRangeLo);
      var hi = clampInt(hiIn.value, 0, USER_TIE_MAX, tieRangeHi);
      if (lo > hi) {{ lo = hi; loIn.value = String(lo); }}
      tieRangeLo = lo;
      tieRangeHi = hi;
      applyObdTieDualRangeVisual();
      syncTieNumberInputs();
      updateGraph();
    }}

    function onTieRangeHighInput() {{
      var loIn = document.getElementById("tie-range-low");
      var hiIn = document.getElementById("tie-range-high");
      if (!loIn || !hiIn) return;
      var lo = clampInt(loIn.value, 0, USER_TIE_MAX, tieRangeLo);
      var hi = clampInt(hiIn.value, 0, USER_TIE_MAX, tieRangeHi);
      if (hi < lo) {{ hi = lo; hiIn.value = String(hi); }}
      tieRangeLo = lo;
      tieRangeHi = hi;
      applyObdTieDualRangeVisual();
      syncTieNumberInputs();
      updateGraph();
    }}

    function onTieNumLowInput() {{
      var endLast = endTiesMode();
      var shownMin = endLast ? 1 : 0;
      var shownMax = endLast ? (USER_TIE_MAX + 1) : USER_TIE_MAX;
      var shownFallback = tieShownFromUi(tieRangeLo, endLast);
      var shownLo = clampInt(document.getElementById("tie-num-low").value, shownMin, shownMax, shownFallback);
      var shownHi = tieShownFromUi(tieRangeHi, endLast);
      if (shownLo > shownHi) shownHi = shownLo;
      var uiLo = tieUiFromShown(shownLo, endLast);
      var uiHi = tieUiFromShown(shownHi, endLast);
      tieRangeLo = Math.min(uiLo, uiHi);
      tieRangeHi = Math.max(uiLo, uiHi);
      syncObdTieDualRangeFromGlobals();
      updateGraph();
    }}

    function onTieNumHighInput() {{
      var endLast = endTiesMode();
      var shownMin = endLast ? 1 : 0;
      var shownMax = endLast ? (USER_TIE_MAX + 1) : USER_TIE_MAX;
      var shownLo = tieShownFromUi(tieRangeLo, endLast);
      var shownHiFallback = tieShownFromUi(tieRangeHi, endLast);
      var shownHi = clampInt(document.getElementById("tie-num-high").value, shownMin, shownMax, shownHiFallback);
      if (shownHi < shownLo) shownHi = shownLo;
      var uiLo = tieUiFromShown(shownLo, endLast);
      var uiHi = tieUiFromShown(shownHi, endLast);
      tieRangeLo = Math.min(uiLo, uiHi);
      tieRangeHi = Math.max(uiLo, uiHi);
      syncObdTieDualRangeFromGlobals();
      updateGraph();
    }}

    function colorForN(nVal, nLo, nHi) {{
      var lo = nLo !== undefined ? nLo : {n_min};
      var hi = nHi !== undefined ? nHi : {n_max};
      var span = hi - lo;
      var t = span > 0 ? (nVal - lo) / span : 0;
      return colorFromLut(t);
    }}

    function yAtTieIndex(s, field, uiIdx, endTies) {{
      var tIdx = tiePhysicalIndex(s, uiIdx, endTies);
      if (tIdx < 0) return {{ y: NaN, pStr: "" }};
      var arr = s[field];
      var len = arr ? arr.length : 0;
      if (tIdx >= len) return {{ y: NaN, pStr: "" }};
      var v = arr[tIdx];
      var y = Number.isFinite(v) ? v : NaN;
      var pp = (s.p && s.p.length > tIdx) ? s.p[tIdx] : NaN;
      var pStr = "";
      if (field !== "p" && Number.isFinite(pp))
        pStr = "p=" + pp.toFixed(4);
      return {{ y: y, pStr: pStr }};
    }}

    function buildTracesAndTitle() {{
      var field = document.getElementById("field").value;
      var endLast = endTiesMode();
      var traceMode = document.getElementById("show-points").checked ? "lines+markers" : "lines";
      var nLo = Math.min(nRangeLo, nRangeHi), nHi = Math.max(nRangeLo, nRangeHi);
      var tLo = Math.min(tieRangeLo, tieRangeHi), tHi = Math.max(tieRangeLo, tieRangeHi);
      var tsLo = Math.min(tieShownFromUi(tLo, endLast), tieShownFromUi(tHi, endLast));
      var tsHi = Math.max(tieShownFromUi(tLo, endLast), tieShownFromUi(tHi, endLast));
      var traces = [];
      var titleSuffix;
      if (!nMultiple()) {{
        var n = nLo;
        titleSuffix = "n=" + n + ", tie #" + tsLo + "–" + tsHi;
        var s = TIE_DATA[String(n)];
        var xs = [], ys = [], text = [];
        for (var t = tLo; t <= tHi; t++) {{
          if (!s) {{
            xs.push(tieShownFromUi(t, endLast)); ys.push(NaN); text.push("");
            continue;
          }}
          var got = yAtTieIndex(s, field, t, endLast);
          xs.push(tieShownFromUi(t, endLast));
          ys.push(got.y);
          text.push("n=" + n + " " + got.pStr);
        }}
        traces.push({{
          x: xs,
          y: ys,
          mode: traceMode,
          type: "scatter",
          name: field + " @n=" + n,
          text: text,
          hovertemplate: "tie #=%{{x}}<br>" + field + "=%{{y}}<br>%{{text}}<extra></extra>",
          showlegend: false,
          line: {{ color: "blue", width: 1.5 }}
        }});
      }} else {{
        titleSuffix = "n=" + nLo + "–" + nHi + ", tie #" + tsLo + "–" + tsHi;
        for (var nv = nLo; nv <= nHi; nv++) {{
          var s2 = TIE_DATA[String(nv)];
          var xsM = [], ysM = [], textM = [];
          for (var ti = tLo; ti <= tHi; ti++) {{
            var tiShown = tieShownFromUi(ti, endLast);
            if (!s2) {{
              xsM.push(tiShown); ysM.push(NaN); textM.push("");
              continue;
            }}
            var got2 = yAtTieIndex(s2, field, ti, endLast);
            xsM.push(tiShown);
            ysM.push(got2.y);
            textM.push("n=" + nv + " " + got2.pStr);
          }}
          traces.push({{
            x: xsM,
            y: ysM,
            mode: traceMode,
            type: "scatter",
            name: "n=" + nv,
            text: textM,
            hovertemplate: "tie #=%{{x}}<br>" + field + "=%{{y}}<br>%{{text}}<extra></extra>",
            showlegend: false,
            line: {{ color: colorForN(nv, nLo, nHi), width: 1.5 }}
          }});
        }}
      }}
      return {{ traces: traces, titleSuffix: titleSuffix, field: field }};
    }}

    function getCurrentYRange() {{
      var gd = document.getElementById("graph");
      if (!gd || !gd._fullLayout) return null;
      var ax = gd._fullLayout.yaxis;
      if (!ax) return null;
      if (ax.range && ax.range.length >= 2) {{
        return [Number(ax.range[0]), Number(ax.range[1])];
      }}
      if (ax._rl && ax._rl.length >= 2) {{
        return [Number(ax._rl[0]), Number(ax._rl[1])];
      }}
      return null;
    }}

    function updateGraph() {{
      var pack = buildTracesAndTitle();
      var logX = document.getElementById("log-x").checked;
      var logY = document.getElementById("log-y").checked;
      var showLeg = nMultiple();
      var endLast = endTiesMode();
      var layout = {{
        title: "Tie " + pack.field + " vs tie # (" + pack.titleSuffix + ")",
        margin: {{ t: 48, b: 48, l: 56, r: 24 }},
        xaxis: {{ title: endLast ? "Tie # from end (1 = last)" : "Tie #", type: logX ? "log" : "linear" }},
        yaxis: {{ title: fieldLabel(), type: logY ? "log" : "linear" }},
        showlegend: showLeg,
        legend: {{ orientation: "v", x: 1.02, y: 1 }}
      }};
      var fyChk = document.getElementById("freeze-y");
      if (fyChk && fyChk.checked && frozenYRange) {{
        layout.yaxis.range = [frozenYRange[0], frozenYRange[1]];
        layout.yaxis.autorange = false;
      }} else {{
        layout.yaxis.autorange = true;
      }}
      Plotly.react("graph", pack.traces, layout, {{ responsive: true }});
    }}

    document.getElementById("field").addEventListener("change", function() {{
      var fy = document.getElementById("freeze-y");
      if (fy) fy.checked = false;
      frozenYRange = null;
      updateGraph();
    }});
    document.getElementById("freeze-y").addEventListener("change", function() {{
      var el = document.getElementById("freeze-y");
      if (el && el.checked) {{
        frozenYRange = getCurrentYRange();
        if (!frozenYRange) el.checked = false;
      }} else {{
        frozenYRange = null;
      }}
      updateGraph();
    }});
    document.getElementById("multiple").addEventListener("change", function() {{
      syncNMultipleMode();
      updateGraph();
    }});
    document.getElementById("log-x").addEventListener("change", updateGraph);
    document.getElementById("log-y").addEventListener("change", updateGraph);
    document.getElementById("show-points").addEventListener("change", updateGraph);
    document.getElementById("end-ties").addEventListener("change", function() {{
      syncTieNumberInputs();
      updateGraph();
    }});
    document.getElementById("n-range-low").addEventListener("input", onNRangeLowInput);
    document.getElementById("n-range-high").addEventListener("input", onNRangeHighInput);
    document.getElementById("n-num-low").addEventListener("input", onNNumLowInput);
    document.getElementById("n-num-high").addEventListener("input", onNNumHighInput);
    document.getElementById("tie-range-low").addEventListener("input", onTieRangeLowInput);
    document.getElementById("tie-range-high").addEventListener("input", onTieRangeHighInput);
    document.getElementById("tie-num-low").addEventListener("input", onTieNumLowInput);
    document.getElementById("tie-num-high").addEventListener("input", onTieNumHighInput);

    syncObdNDualRangeFromGlobals();
    syncObdTieDualRangeFromGlobals();
    updateGraph();
    window.addEventListener("resize", function() {{
      Plotly.Plots.resize(document.getElementById("graph"));
    }});
  </script>
</body>
</html>
"""
