// Analysis tab of a run: the mentor's survey shift (post − pre) by the levels of a chosen axis, per battery
// (scale-normalised, with SE whiskers) and per item, and the judge's score across turns per level. Descriptive
// views for a sandbox; the paper's estimands live in analysis/.

import { api } from "../api.js";
import { h, mount, badge, fmtInt, fmtNum, fmtSigned, deltaClass, toastError } from "../ui.js";
import { barChart, lineChart, seriesColor } from "../charts.js";
import { rs, currentDataDir, levelLabel } from "./runstate.js";
import { card } from "./common.js";

const an = { runId: null, factor: undefined, metric: undefined, judge: undefined };

export function renderAnalysis(body, ctx, runId) {
  if (an.runId !== runId) {
    an.runId = runId;
    an.factor = undefined;
    an.metric = undefined;
    an.judge = undefined;
  }
  if (an.factor === undefined) {
    // Default to the axis the study is about: ideology when the run has it, else the first with >1 level.
    const f = (rs.summary && rs.summaryRunId === runId && rs.summary.factors) || {};
    const keys = Object.keys(f);
    an.factor = keys.includes("ideology") ? "ideology" : keys.find((k) => (f[k] || []).length > 1) || "";
  }
  const controls = h("div", { class: "toolbar wrap" });
  const content = h("div", { class: "analysis" }, h("p", { class: "muted" }, "Loading…"));
  mount(body, controls, content);

  const load = async () => {
    content.classList.add("stale");
    try {
      const res = await api.analysis(runId, currentDataDir(), { factor: an.factor, metric: an.metric, judge: an.judge });
      if (!ctx.alive()) return;
      an.factor = res.factor ?? an.factor ?? "";
      an.metric = res.metric ?? an.metric;
      an.judge = res.judge ?? an.judge;
      drawControls(res);
      drawContent(res);
    } catch (err) {
      if (!ctx.alive()) return;
      toastError(err, "Analysis");
      mount(content, h("div", { class: "error-box" }, err.message));
    } finally {
      content.classList.remove("stale");
    }
  };

  const select = (label, options, value, onChange) => {
    const sel = h("select", { class: "input", "aria-label": label }, options.map(([v, l]) => h("option", { value: v }, l)));
    sel.value = value ?? "";
    sel.addEventListener("change", () => onChange(sel.value));
    return h("label", { class: "inline-field" }, h("span", null, label), sel);
  };

  const drawControls = (res) => {
    const factors = Object.keys(res.factors || {});
    const metrics = res.metrics_available || [];
    const judges = res.judges || [];
    mount(
      controls,
      select("by", [["", "(all dyads)"], ...factors.map((k) => [k, k])], res.factor || "", (v) => {
        an.factor = v;
        load();
      }),
      metrics.length
        ? select("metric", metrics.map((m) => [m, m]), res.metric || metrics[0], (v) => {
            an.metric = v;
            load();
          })
        : null,
      judges.length > 1
        ? select(
            "judge",
            judges.map((j) => {
              const sha = typeof j === "string" ? j : j.sha256 || j.judge_sha256 || j.file || "";
              const name = typeof j === "string" ? j.slice(0, 12) : `${j.alias || ""} ${String(sha).slice(0, 8)}`.trim();
              return [sha, name];
            }),
            res.judge || "",
            (v) => {
              an.judge = v;
              load();
            },
          )
        : null,
      h("span", { class: "muted small" }, `Over the latest complete attempt of each dyad (${fmtInt((res.status_counts || {}).complete || 0)} complete), survey rows with origin = run.`),
    );
  };

  const drawContent = (res) => {
    const sv = res.survey || {};
    const levels = (sv.levels || []).map(levelLabel);
    // Colour follows the level (its place in the axis's level order), not its rank among the levels with data.
    const colorOf = new Map();
    for (const l of [...((res.factors || {})[res.factor] || []).map(levelLabel), ...levels]) if (!colorOf.has(l)) colorOf.set(l, colorOf.size);
    const sc = res.scores || {};
    for (const l of (sc.levels || []).map(levelLabel)) if (!colorOf.has(l)) colorOf.set(l, colorOf.size);
    const nD = sv.n_dyads || {};
    const batteries = sv.batteries || [];
    const items = sv.items || [];
    const byB = sv.by_battery || {};
    const byI = sv.by_item || {};
    const factorName = res.factor || "all dyads";

    let surveyBlock;
    if (!levels.length || !batteries.length) {
      surveyBlock = h("p", { class: "muted" }, "No complete dyad has both survey phases yet.");
    } else {
      const groups = batteries.map((b) => ({
        key: b,
        label: b,
        bars: levels.map((l) => {
          const cell = (byB[l] || {})[b] || {};
          return { key: l, label: l, value: cell.mean_delta_norm ?? null, se: cell.se_delta_norm ?? null, n: cell.n ?? null, colorIndex: colorOf.get(l) };
        }),
      }));
      const legendItems = levels.map((l) => ({ label: l, colorIndex: colorOf.get(l), note: `n=${nD[l] ?? "?"}` }));
      const itemTable = h(
        "div",
        { class: "table-wrap" },
        h(
          "table",
          { class: "table compact item-table" },
          h(
            "thead",
            null,
            h(
              "tr",
              null,
              h("th", null, "item"),
              h("th", null, "battery"),
              levels.map((l) => h("th", { class: "num" }, h("span", { class: "key-rect", style: { background: seriesColor(colorOf.get(l)) } }), ` ${l}`)),
            ),
          ),
          h(
            "tbody",
            null,
            items.map((it) =>
              h(
                "tr",
                null,
                h("td", { class: "mono small" }, it.item_id),
                h("td", { class: "small" }, it.battery, it.scale ? h("span", { class: "muted" }, ` ${it.scale.min}–${it.scale.max}`) : null),
                levels.map((l) => {
                  const c = (byI[l] || {})[it.item_id];
                  if (!c || !c.n) return h("td", { class: "num muted" }, "—");
                  return h(
                    "td",
                    { class: "num item-cell", title: `n ${c.n}; SE ${c.se_delta === null || c.se_delta === undefined ? "— (n < 2)" : fmtNum(c.se_delta, 3)}` },
                    h("div", { class: "small muted" }, `${fmtNum(c.mean_pre, 2)} → ${fmtNum(c.mean_post, 2)}`),
                    h(
                      "div",
                      null,
                      h("span", { class: `delta ${deltaClass(c.mean_delta)}` }, fmtSigned(c.mean_delta, 2)),
                      h("span", { class: "muted small" }, `${c.se_delta === null || c.se_delta === undefined ? "" : ` ±${fmtNum(c.se_delta, 2)}`} n=${c.n}`),
                    ),
                  );
                }),
              ),
            ),
          ),
        ),
      );
      surveyBlock = h(
        "div",
        null,
        barChart({ groups, legendItems, title: `Mean normalised shift (post − pre) / (max − min) by ${factorName}`, yLabel: "normalised Δ", valueDigits: 3 }),
        h("p", { class: "muted small" }, "Whiskers are ±1 SE (sample SD / √n; none below n = 2). Positive = the mentor moved up the item's scale after the dialogue."),
        h("h3", null, "By item"),
        itemTable,
      );
    }

    let scoreBlock;
    if (sc.error) {
      scoreBlock = h("div", { class: "banner banner-warn inline" }, sc.error);
    } else {
      const series = Object.entries(sc.series || {}).map(([lv, pts]) => {
        const l = levelLabel(lv === "null" ? null : lv);
        return {
          key: l,
          label: l,
          colorIndex: colorOf.has(l) ? colorOf.get(l) : 0,
          note: undefined,
          points: (pts || []).map((p) => ({ x: Number(p.turn), y: p.mean === null || p.mean === undefined ? NaN : Number(p.mean), se: p.se === null || p.se === undefined ? NaN : Number(p.se), n: p.n })),
        };
      });
      series.sort((a, b) => a.colorIndex - b.colorIndex);
      scoreBlock = series.some((s) => s.points.length)
        ? lineChart({ series, title: `Judge ${sc.metric || res.metric || "score"} by turn and ${factorName}`, xLabel: "turn", yLabel: "mean score", yDomain: [0, 1] })
        : h("p", { class: "muted" }, `No ${sc.metric || res.metric || ""} scores yet. Score the run (Actions) first.`);
    }

    mount(
      content,
      card("Mentor survey shift", { help: `By ${factorName}. Each battery's bar is the mean over dyads of the scale-normalised delta, so a 0–10 thermometer and a 1–5 item compare.` }, surveyBlock),
      card(
        "Judge score across turns",
        { help: "Mean judge score per level and turn (band: ±1 SE). For the mentor's alignment use scope stance; for seeker adherence, prompt_to_line or line_to_line." },
        sc.judge ? h("p", { class: "muted small" }, "judge ", badge(String(sc.judge).slice(0, 12), "neutral")) : null,
        scoreBlock,
      ),
    );
  };

  load();
}
