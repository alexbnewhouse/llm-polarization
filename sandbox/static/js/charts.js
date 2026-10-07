// Hand-rolled inline SVG charts: a grouped bar chart with SE whiskers and a multi-series line chart with
// SE bands and a crosshair. Colours are CSS variables (--series-1..8) so light and dark mode both work;
// a series keeps its colour by entity (colorIndex), never by rank. Labels go in via textContent.

import { h } from "./ui.js";

const NS = "http://www.w3.org/2000/svg";

function s(tag, attrs, ...children) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "text") el.textContent = String(v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, String(v));
  }
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const seriesColor = (i) => `var(--series-${(((i || 0) % 8) + 8) % 8 + 1})`;

export function niceTicks(min, max, count = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const span = max - min;
  const raw = span / Math.max(1, count);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const lo = Math.floor(min / step) * step;
  const hi = Math.ceil(max / step) * step;
  const out = [];
  for (let v = lo; v <= hi + step / 2; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

function fmtTick(v) {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (a >= 10) return v.toFixed(0);
  if (a >= 1) return v.toFixed(1).replace(/\.0$/, "");
  return v.toFixed(2).replace(/0$/, "");
}

// ---- tooltip -----------------------------------------------------------------------------------------------------

let tipEl = null;

function tip() {
  if (!tipEl) {
    tipEl = h("div", { class: "chart-tip", role: "tooltip" });
    document.body.appendChild(tipEl);
  }
  return tipEl;
}

function showTip(evt, rows, heading) {
  const t = tip();
  t.replaceChildren();
  if (heading) t.appendChild(h("div", { class: "chart-tip-head" }, heading));
  for (const r of rows) {
    t.appendChild(
      h(
        "div",
        { class: "chart-tip-row" },
        r.color ? h("span", { class: r.shape === "rect" ? "key-rect" : "key-line", style: { background: r.color } }) : null,
        h("strong", null, r.value),
        h("span", { class: "chart-tip-label" }, r.label),
      ),
    );
  }
  t.style.display = "block";
  const pad = 14;
  const x = evt.clientX + pad;
  const y = evt.clientY + pad;
  const w = t.offsetWidth;
  const hgt = t.offsetHeight;
  t.style.left = `${Math.min(x, window.innerWidth - w - 8)}px`;
  t.style.top = `${Math.min(y, window.innerHeight - hgt - 8)}px`;
}

function hideTip() {
  if (tipEl) tipEl.style.display = "none";
}

function legend(items, shape) {
  return h(
    "div",
    { class: "chart-legend" },
    items.map((it) =>
      h(
        "span",
        { class: "legend-item" },
        h("span", { class: shape === "rect" ? "key-rect" : "key-line", style: { background: seriesColor(it.colorIndex) } }),
        h("span", null, it.label),
        it.note ? h("span", { class: "muted" }, ` ${it.note}`) : null,
      ),
    ),
  );
}

function figure(title, svg, extra) {
  return h("figure", { class: "chart" }, title ? h("figcaption", null, title) : null, svg, extra);
}

function barPath(x, y0, w, y1, r) {
  // A bar from baseline y0 to value y1, rounded (radius r) at the data end only.
  const up = y1 < y0;
  const hgt = Math.abs(y1 - y0);
  const rr = Math.min(r, hgt, w / 2);
  if (up) {
    return `M${x},${y0} V${y1 + rr} Q${x},${y1} ${x + rr},${y1} H${x + w - rr} Q${x + w},${y1} ${x + w},${y1 + rr} V${y0} Z`;
  }
  return `M${x},${y0} V${y1 - rr} Q${x},${y1} ${x + rr},${y1} H${x + w - rr} Q${x + w},${y1} ${x + w},${y1 - rr} V${y0} Z`;
}

// groups: [{key, label, bars: [{key, label, value, se, n, colorIndex}]}]; legendItems: [{label, colorIndex, note}]
export function barChart({ groups, title, yLabel, legendItems, width = 760, height = 300, valueDigits = 3 }) {
  const m = { l: 64, r: 16, t: 14, b: 44 };
  const pw = width - m.l - m.r;
  const ph = height - m.t - m.b;
  const vals = [];
  for (const g of groups) for (const b of g.bars) {
    if (b.value === null || b.value === undefined || !Number.isFinite(b.value)) continue;
    const se = Number.isFinite(b.se) ? b.se : 0;
    vals.push(b.value - se, b.value + se);
  }
  let lo = Math.min(0, ...vals);
  let hi = Math.max(0, ...vals);
  if (lo === hi) {
    lo = -0.1;
    hi = 0.1;
  }
  const ticks = niceTicks(lo, hi, 5);
  lo = ticks[0];
  hi = ticks[ticks.length - 1];
  const y = (v) => m.t + ph - ((v - lo) / (hi - lo)) * ph;
  const svg = s("svg", {
    class: "chart-svg",
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": title || "bar chart",
  });
  if (title) svg.appendChild(s("title", { text: title }));
  for (const t of ticks) {
    svg.appendChild(s("line", { class: "grid", x1: m.l, x2: width - m.r, y1: y(t), y2: y(t) }));
    svg.appendChild(s("text", { class: "tick", x: m.l - 8, y: y(t) + 4, "text-anchor": "end", text: fmtTick(t) }));
  }
  svg.appendChild(s("line", { class: "baseline", x1: m.l, x2: width - m.r, y1: y(0), y2: y(0) }));
  if (yLabel) {
    svg.appendChild(
      s("text", { class: "axis-label", x: 14, y: m.t + ph / 2, transform: `rotate(-90 14 ${m.t + ph / 2})`, "text-anchor": "middle", text: yLabel }),
    );
  }
  const band = pw / Math.max(1, groups.length);
  groups.forEach((g, gi) => {
    const nb = Math.max(1, g.bars.length);
    const bw = Math.max(4, Math.min(24, (band * 0.82) / nb - 2));
    const block = nb * bw + (nb - 1) * 2;
    const x0 = m.l + gi * band + (band - block) / 2;
    g.bars.forEach((b, bi) => {
      const x = x0 + bi * (bw + 2);
      if (b.value === null || b.value === undefined || !Number.isFinite(b.value)) {
        svg.appendChild(s("text", { class: "tick", x: x + bw / 2, y: y(0) - 4, "text-anchor": "middle", text: "–" }));
        return;
      }
      const color = seriesColor(b.colorIndex);
      const rows = [
        { value: (b.value >= 0 ? "+" : "") + b.value.toFixed(valueDigits), label: `${b.label}` },
        { value: Number.isFinite(b.se) ? `±${b.se.toFixed(valueDigits)}` : "—", label: "SE" },
        { value: b.n ?? "—", label: "n" },
      ];
      const bar = s("path", { class: "bar", d: barPath(x, y(0), bw, y(b.value), 4), style: { fill: color }, tabindex: 0 });
      bar.appendChild(s("title", { text: `${g.label} · ${b.label}: ${b.value.toFixed(valueDigits)} (SE ${Number.isFinite(b.se) ? b.se.toFixed(valueDigits) : "—"}, n ${b.n ?? "—"})` }));
      const enter = (e) => {
        bar.classList.add("hover");
        showTip(e, rows.map((r, i) => ({ ...r, color: i === 0 ? color : null, shape: "rect" })), g.label);
      };
      bar.addEventListener("pointermove", enter);
      bar.addEventListener("pointerleave", () => {
        bar.classList.remove("hover");
        hideTip();
      });
      bar.addEventListener("focus", (e) => {
        const r = bar.getBoundingClientRect();
        enter({ clientX: r.right, clientY: r.top });
      });
      bar.addEventListener("blur", hideTip);
      svg.appendChild(bar);
      if (Number.isFinite(b.se) && b.se > 0) {
        const cx = x + bw / 2;
        const a = y(b.value - b.se);
        const c = y(b.value + b.se);
        svg.appendChild(s("line", { class: "whisker", x1: cx, x2: cx, y1: a, y2: c }));
        svg.appendChild(s("line", { class: "whisker", x1: cx - 4, x2: cx + 4, y1: a, y2: a }));
        svg.appendChild(s("line", { class: "whisker", x1: cx - 4, x2: cx + 4, y1: c, y2: c }));
      }
    });
    svg.appendChild(s("text", { class: "group-label", x: m.l + gi * band + band / 2, y: height - m.b + 20, "text-anchor": "middle", text: g.label }));
  });
  return figure(title, svg, legendItems && legendItems.length > 1 ? legend(legendItems, "rect") : null);
}

// series: [{key, label, colorIndex, points: [{x, y, se, n}]}]
export function lineChart({ series, title, xLabel, yLabel, yDomain, width = 760, height = 280, valueDigits = 2 }) {
  const m = { l: 56, r: 18, t: 14, b: 44 };
  const pw = width - m.l - m.r;
  const ph = height - m.t - m.b;
  const xs = new Set();
  const ys = [];
  for (const sr of series) for (const p of sr.points) {
    if (!Number.isFinite(p.x)) continue;
    xs.add(p.x);
    if (Number.isFinite(p.y)) {
      ys.push(p.y);
      if (Number.isFinite(p.se)) ys.push(p.y - p.se, p.y + p.se);
    }
  }
  const xList = [...xs].sort((a, b) => a - b);
  let xlo = xList.length ? xList[0] : 0;
  let xhi = xList.length ? xList[xList.length - 1] : 1;
  if (xlo === xhi) {
    xlo -= 1;
    xhi += 1;
  }
  let [ylo, yhi] = yDomain || [Math.min(...ys), Math.max(...ys)];
  if (!Number.isFinite(ylo) || !Number.isFinite(yhi)) [ylo, yhi] = [0, 1];
  const yt = niceTicks(ylo, yhi, 5);
  if (!yDomain) {
    ylo = yt[0];
    yhi = yt[yt.length - 1];
  }
  const xt = niceTicks(xlo, xhi, Math.min(10, Math.max(2, xList.length - 1))).filter((t) => t >= xlo && t <= xhi && Number.isInteger(t));
  const X = (v) => m.l + ((v - xlo) / (xhi - xlo)) * pw;
  const Y = (v) => m.t + ph - ((v - ylo) / (yhi - ylo)) * ph;
  const svg = s("svg", { class: "chart-svg", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": title || "line chart" });
  if (title) svg.appendChild(s("title", { text: title }));
  for (const t of yt) {
    if (t < ylo - 1e-9 || t > yhi + 1e-9) continue;
    svg.appendChild(s("line", { class: "grid", x1: m.l, x2: width - m.r, y1: Y(t), y2: Y(t) }));
    svg.appendChild(s("text", { class: "tick", x: m.l - 8, y: Y(t) + 4, "text-anchor": "end", text: fmtTick(t) }));
  }
  svg.appendChild(s("line", { class: "baseline", x1: m.l, x2: width - m.r, y1: m.t + ph, y2: m.t + ph }));
  for (const t of xt) {
    svg.appendChild(s("text", { class: "tick", x: X(t), y: m.t + ph + 18, "text-anchor": "middle", text: fmtTick(t) }));
  }
  if (xLabel) svg.appendChild(s("text", { class: "axis-label", x: m.l + pw / 2, y: height - 6, "text-anchor": "middle", text: xLabel }));
  if (yLabel) {
    svg.appendChild(
      s("text", { class: "axis-label", x: 14, y: m.t + ph / 2, transform: `rotate(-90 14 ${m.t + ph / 2})`, "text-anchor": "middle", text: yLabel }),
    );
  }
  const dense = xList.length > 50;
  for (const sr of series) {
    const color = seriesColor(sr.colorIndex);
    const pts = sr.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)).sort((a, b) => a.x - b.x);
    if (!pts.length) continue;
    const banded = pts.filter((p) => Number.isFinite(p.se));
    if (banded.length > 1) {
      const top = banded.map((p) => `${X(p.x)},${Y(Math.min(yhi, p.y + p.se))}`);
      const bot = banded.slice().reverse().map((p) => `${X(p.x)},${Y(Math.max(ylo, p.y - p.se))}`);
      svg.appendChild(s("polygon", { class: "band", points: [...top, ...bot].join(" "), style: { fill: color } }));
    }
    svg.appendChild(s("polyline", { class: "line", points: pts.map((p) => `${X(p.x)},${Y(p.y)}`).join(" "), style: { stroke: color } }));
    if (!dense || pts.length === 1) {
      for (const p of pts) svg.appendChild(s("circle", { class: "dot", cx: X(p.x), cy: Y(p.y), r: 4, style: { fill: color } }));
    }
  }
  // Crosshair: snap to the nearest x and list every series there.
  const cross = s("line", { class: "crosshair", x1: 0, x2: 0, y1: m.t, y2: m.t + ph, style: { display: "none" } });
  svg.appendChild(cross);
  const hit = s("rect", { class: "hit", x: m.l, y: m.t, width: pw, height: ph });
  const toSvgX = (evt) => {
    const r = svg.getBoundingClientRect();
    return ((evt.clientX - r.left) / r.width) * width;
  };
  hit.addEventListener("pointermove", (e) => {
    if (!xList.length) return;
    const sx = toSvgX(e);
    let best = xList[0];
    for (const x of xList) if (Math.abs(X(x) - sx) < Math.abs(X(best) - sx)) best = x;
    cross.setAttribute("x1", X(best));
    cross.setAttribute("x2", X(best));
    cross.style.display = "";
    const rows = [];
    for (const sr of series) {
      const p = sr.points.find((q) => q.x === best);
      if (!p || !Number.isFinite(p.y)) continue;
      const extra = [Number.isFinite(p.se) ? `±${p.se.toFixed(valueDigits)}` : null, p.n !== undefined && p.n !== null ? `n ${p.n}` : null]
        .filter(Boolean)
        .join(", ");
      rows.push({ value: p.y.toFixed(valueDigits), label: `${sr.label}${extra ? ` (${extra})` : ""}`, color: seriesColor(sr.colorIndex) });
    }
    showTip(e, rows, `${xLabel || "x"} ${best}`);
  });
  hit.addEventListener("pointerleave", () => {
    cross.style.display = "none";
    hideTip();
  });
  svg.appendChild(hit);
  return figure(title, svg, series.length > 1 ? legend(series.map((sr) => ({ label: sr.label, colorIndex: sr.colorIndex, note: sr.note })), "line") : null);
}
