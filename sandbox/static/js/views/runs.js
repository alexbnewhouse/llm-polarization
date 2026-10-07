// Runs view: the runs in a data dir; one run's models, hashes, settings, status and per-dyad progress
// (refreshed by tailing its row files); its analysis; and actions (score, flags, agreement, re-administer).
// Routes: #/runs, #/runs/<run_id>[/<tab>], #/runs/<run_id>/dyads/<dyad_id>.

import { api } from "../api.js";
import {
  h, mount, button, badge, statusChip, progressBar, fmtInt, fmtTime, kv, mono, shortHash, toast, toastError, pagedRows, emptyState,
} from "../ui.js";
import { state } from "../state.js";
import { viewHeader, card } from "./common.js";
import { rs, currentDataDir, setDataDir, setAuto, loadSummary, startLive, isLive, levelLabel, runHref, dyadHref } from "./runstate.js";
import { renderDyad } from "./dyad.js";
import { renderAnalysis } from "./analysis.js";
import { jobConsole } from "../jobconsole.js";

const RUN_ID_RE = /^[A-Za-z0-9_.-]+$/;
const TABS = [
  ["dyads", "Dyads"],
  ["analysis", "Analysis"],
  ["actions", "Actions"],
];
const ui = { tab: "dyads", filters: { status: "", search: "", cond: {} }, runIdForFilters: null };
const actions = { runId: null, config: null, judgeConfig: null, judge: { url: "", gguf_path: "" }, jobs: [], form: {} };

export function render(root, ctx) {
  const dd = ctx.query.get("data_dir");
  if (dd) setDataDir(dd);
  const [runId, sub, dyadId] = ctx.params;
  if (!runId) return renderList(root, ctx);
  if (!RUN_ID_RE.test(runId)) {
    mount(root, viewHeader("Runs"), emptyState(`“${runId}” is not a run id.`));
    return;
  }
  if (sub === "dyads" && dyadId) return renderDyad(root, ctx, runId, dyadId);
  ui.tab = sub && TABS.some(([t]) => t === sub) ? sub : "dyads";
  return renderRun(root, ctx, runId);
}

// ---- data dir bar ------------------------------------------------------------------------------------------------

function dataDirBar(ctx, onChange) {
  const dirs = [...new Set([...((state.meta && state.meta.data_dirs) || []), currentDataDir()])];
  const sel = h("select", { class: "input", "aria-label": "Data directory" }, dirs.map((d) => h("option", { value: d }, d)), h("option", { value: "__other" }, "other…"));
  sel.value = currentDataDir();
  const other = h("input", { type: "text", class: "input mono", placeholder: "path to a data dir", hidden: true, "aria-label": "Other data directory" });
  sel.addEventListener("change", () => {
    if (sel.value === "__other") {
      other.hidden = false;
      other.focus();
      return;
    }
    other.hidden = true;
    setDataDir(sel.value);
    onChange();
  });
  other.addEventListener("change", () => {
    if (!other.value.trim()) return;
    setDataDir(other.value.trim());
    onChange();
  });
  return h("div", { class: "toolbar" }, h("label", { class: "inline-field" }, h("span", null, "data dir"), sel), other, button("Refresh", onChange, { small: true }));
}

// ---- run list ----------------------------------------------------------------------------------------------------

function renderList(root, ctx) {
  const host = h("div", { class: "card" }, h("p", { class: "muted" }, "Loading…"));
  const load = async () => {
    host.classList.add("stale");
    try {
      const res = await api.runs(currentDataDir());
      if (!ctx.alive()) return;
      const runs = [...((res && res.runs) || [])].sort((a, b) => String(b.mtime || b.started_at || "").localeCompare(String(a.mtime || a.started_at || "")));
      if (!runs.length) {
        mount(host, emptyState(`No runs in ${res && res.data_dir ? res.data_dir : currentDataDir()} yet.`, h("p", { class: "muted small" }, "Runs started here or from a terminal (python -m harness.run run …) appear once they write data/<run_id>/.")));
        return;
      }
      const tbody = h("tbody");
      pagedRows(
        tbody,
        runs,
        (r) => {
          const sc = r.status_counts || {};
          const planned = r.planned ?? r.n_dyads;
          return h(
            "tr",
            { class: "clickable", onclick: (e) => { if (e.target.tagName !== "A") location.hash = runHref(r.run_id); } },
            h("td", null, h("a", { href: runHref(r.run_id), class: "mono link" }, r.run_id)),
            h("td", { class: "small" }, fmtTime(r.started_at || r.mtime)),
            h("td", { class: "progress-cell" }, progressBar(sc.complete || 0, planned, { title: `${fmtInt(sc.complete || 0)} complete of ${planned === null || planned === undefined ? "?" : fmtInt(planned)} planned` })),
            h("td", null, h("span", { class: "counts" }, countChips(sc))),
            h("td", { class: "num" }, fmtInt(r.n_dyads)),
            h("td", { class: "small" }, modelName(r.seeker)),
            h("td", { class: "small" }, modelName(r.mentor)),
            h("td", null, r.has_scores ? badge("scored", "info") : null, r.has_flags ? badge("flags", "neutral") : null, r.has_manifest === false ? badge("no manifest", "warn") : null),
          );
        },
        500,
        8,
      );
      mount(
        host,
        h(
          "div",
          { class: "table-wrap tall" },
          h(
            "table",
            { class: "table sticky" },
            h("thead", null, h("tr", null, h("th", null, "run"), h("th", null, "started"), h("th", null, "complete / planned"), h("th", null, "status"), h("th", { class: "num" }, "dyads"), h("th", null, "seeker"), h("th", null, "mentor"), h("th"))),
            tbody,
          ),
        ),
      );
    } catch (err) {
      if (!ctx.alive()) return;
      toastError(err, "Could not list runs");
      mount(host, h("div", { class: "error-box" }, err.message));
    } finally {
      host.classList.remove("stale");
    }
  };
  mount(root, viewHeader("Runs", "Every run in the data dir, whether started here or from a terminal."), dataDirBar(ctx, load), host);
  load();
}

function modelName(m) {
  if (!m) return h("span", { class: "muted" }, "—");
  const name = m.alias || (m.model_path ? String(m.model_path).split(/[\\/]/).pop() : null) || "?";
  return h("span", { title: `${m.model_path || ""}\n${m.model_sha256 || ""}` }, name, " ", m.model_sha256 ? shortHash(m.model_sha256, 8) : null);
}

function countChips(sc) {
  return [
    badge(`${fmtInt(sc.complete || 0)} complete`, "ok"),
    sc.running ? badge(`${fmtInt(sc.running)} running`, "info") : null,
    sc.failed ? badge(`${fmtInt(sc.failed)} failed`, "error") : null,
  ];
}

// ---- one run -------------------------------------------------------------------------------------------------------

function renderRun(root, ctx, runId) {
  if (ui.runIdForFilters !== runId) {
    ui.filters = { status: "", search: "", cond: {} };
    ui.runIdForFilters = runId;
  }
  const head = h("div", { class: "run-head" });
  const infoHost = h("div");
  const tabBar = h("div", { class: "tabs", role: "tablist" });
  const body = h("div", { class: "tab-body" });
  mount(
    root,
    h("nav", { class: "crumbs" }, h("a", { href: "#/runs", class: "link" }, "Runs"), h("span", null, " / "), h("span", { class: "mono" }, runId)),
    head,
    infoHost,
    tabBar,
    body,
  );
  mount(head, h("h1", { class: "mono" }, runId), h("p", { class: "muted" }, "Loading…"));

  let dyads = [];
  const byId = new Map();
  let liveStop = null;
  const statusHost = h("div", { class: "run-status" });

  const recount = () => {
    const sc = { complete: 0, failed: 0, running: 0 };
    for (const d of dyads) sc[d.status === "complete" ? "complete" : d.status === "failed" ? "failed" : "running"]++;
    return sc;
  };

  const drawStatus = (s) => {
    const sc = dyads.length ? recount() : s.status_counts || {};
    const planned = s.planned ?? dyads.length;
    const autoBox = h("input", { type: "checkbox", checked: rs.auto });
    autoBox.addEventListener("change", () => {
      setAuto(autoBox.checked);
      drawStatus(s);
    });
    mount(
      statusHost,
      progressBar(sc.complete || 0, planned, { title: "complete / planned" }),
      h("span", { class: "counts" }, countChips(sc)),
      h("span", { class: "muted small" }, `${fmtInt(dyads.length)} started of ${planned === null || planned === undefined ? "?" : fmtInt(planned)} planned`),
      h("label", { class: "field-check live-toggle", title: "Tail the run's files every 2 s" }, autoBox, h("span", null, "auto-refresh"), rs.auto && isLive({ status_counts: sc }) ? badge("live", "info") : null),
    );
  };

  const drawInfo = (s) => {
    const man = s.manifest || null;
    const cfg = (man && man.config) || {};
    const gen = cfg.generation || {};
    const roles = s.roles || {};
    const base = (p) => (p ? String(p).split(/[\\/]/).pop() : "—");
    const fileRef = (p, sha) => (p ? h("span", { title: `${p}${sha ? `\n${sha}` : ""}` }, h("span", { class: "mono small" }, base(p)), sha ? " " : null, sha ? shortHash(sha, 8) : null) : "—");
    const modelCell = (r) => h("td", { class: "nowrap", title: r ? `${r.model_path || ""}${r.url ? `\n${r.url}` : ""}` : "" }, r ? r.alias || base(r.model_path) : "—");
    const roleRow = (name, r) =>
      h(
        "tr",
        null,
        h("td", null, name),
        modelCell(r),
        h("td", null, r ? shortHash(r.model_sha256) : "—"),
        h("td", null, r ? shortHash(r.template_sha256, 8) : "—"),
        h(
          "td",
          { class: "small nowrap", title: r && r.build_info ? `build ${r.build_info}` : "" },
          r && r.family ? r.family : h("span", { class: "muted" }, "—"),
          r && r.build_info ? h("span", { class: "muted mono" }, ` · ${String(r.build_info).slice(0, 14)}`) : null,
        ),
        h("td", { class: "num" }, r && r.total_slots !== undefined && r.total_slots !== null ? String(r.total_slots) : "—"),
      );
    const models = man
      ? h(
          "div",
          { class: "table-wrap" },
          h(
            "table",
            { class: "table compact" },
            h("thead", null, h("tr", null, h("th", null, "role"), h("th", null, "model"), h("th", null, "sha256"), h("th", null, "template"), h("th", null, "family · build"), h("th", { class: "num" }, "slots"))),
            h(
              "tbody",
              null,
              roleRow("seeker", roles.seeker),
              roleRow("mentor", roles.mentor),
              (s.judges || []).map((j) =>
                h(
                  "tr",
                  null,
                  h("td", null, "judge"),
                  modelCell(j),
                  h("td", null, shortHash(j.model_sha256)),
                  h("td", { class: "small muted", colspan: 3 }, `scope ${j.scope || "?"}${j.subsample ? `, subsample ${j.subsample}` : ""} · ${fmtTime(j.ts)}`),
                ),
              ),
            ),
          ),
        )
      : h("p", { class: "muted" }, "No manifest.json in this run directory: the models and settings are unknown.");
    mount(
      infoHost,
      h(
        "div",
        { class: "run-info" },
        card("Models", models),
        card(
          "Settings",
          man
            ? kv([
                ["started", fmtTime(man.started_at)],
                ["harness", man.harness_commit ? h("span", null, shortHash(man.harness_commit, 10), man.harness_dirty ? badge("dirty tree", "warn", "The harness had uncommitted changes when this run started") : null) : "—"],
                ["generation", h("span", { class: "mono small" }, ["temperature", "top_p", "n_predict", "timeout", "enable_thinking"].filter((k) => k in gen).map((k) => `${k}=${gen[k]}`).join("  ") || "—")],
                ["seed · now", h("span", { class: "mono small" }, `run_seed=${cfg.run_seed ?? "?"}  now=${cfg.now ?? "?"}  concurrency=${cfg.concurrency === null || cfg.concurrency === undefined ? "auto" : cfg.concurrency}`)],
                ["manifest", man.input_manifest ? fileRef(man.input_manifest.path, man.input_manifest.sha256) : "—"],
                ["batteries", man.batteries ? h("span", null, fileRef(man.batteries.path, man.batteries.sha256), man.batteries.n_items ? h("span", { class: "muted small" }, ` · ${man.batteries.n_items} items`) : null) : "—"],
                ["path", h("span", { class: "mono small", title: s.path || "" }, s.path || "—")],
              ])
            : kv([["path", h("span", { class: "mono small" }, s.path || "—")]]),
        ),
      ),
    );
  };

  const drawTabs = () => {
    mount(
      tabBar,
      TABS.map(([id, label]) =>
        h(
          "a",
          { class: ["tab", ui.tab === id ? "active" : null], role: "tab", "aria-selected": ui.tab === id ? "true" : "false", href: runHref(runId, `/${id}`) },
          label,
        ),
      ),
    );
  };

  // ---- dyad table ----
  let tbody = null;
  let shownCount = null;
  let showReason = false;
  const dyadRow = (d, factorKeys) => {
    const tr = h(
      "tr",
      { dataset: { dyad: d.dyad_id } },
      h("td", null, h("a", { class: "mono link", href: dyadHref(runId, d.dyad_id) }, d.dyad_id)),
      factorKeys.map((k) => h("td", { class: "mono small" }, levelLabel(d.condition ? d.condition[k] : undefined))),
      h("td", { class: "small" }, d.persona_mode || "—"),
      h("td", { class: "num" }, d.attempts && d.attempts > 1 ? h("span", { title: `${d.attempts} attempts` }, `${d.attempt} of ${d.attempts}`) : String(d.attempt ?? "—")),
      h("td", null, statusChip(d.status)),
      h("td", { class: "progress-cell" }, progressBar(d.turns_done || 0, d.n_turns, { kind: d.status === "failed" ? "error" : d.status === "complete" ? "ok" : null })),
      showReason ? h("td", { class: "small reason", title: d.reason || "" }, d.reason || "") : null,
      h("td", { class: "small muted nowrap", title: d.last_ts ? String(d.last_ts) : "" }, d.last_ts ? fmtTime(d.last_ts).slice(11) : "—"),
    );
    byId.set(d.dyad_id, { d, tr });
    return tr;
  };

  const filtered = (factorKeys) => {
    const f = ui.filters;
    const q = f.search.trim().toLowerCase();
    return dyads.filter((d) => {
      if (f.status && d.status !== f.status) return false;
      if (q && !String(d.dyad_id).toLowerCase().includes(q)) return false;
      for (const k of factorKeys) {
        const want = f.cond[k];
        if (want === undefined || want === "") continue;
        if (levelLabel(d.condition ? d.condition[k] : undefined) !== want) return false;
      }
      return true;
    });
  };

  const drawDyads = () => {
    const s = rs.summary;
    const factors = (s && s.factors) || {};
    const factorKeys = Object.keys(factors);
    const filterBar = h(
      "div",
      { class: "toolbar wrap" },
      factorKeys.map((k) => {
        const sel = h("select", { class: "input", "aria-label": `Filter by ${k}` }, h("option", { value: "" }, `${k}: all`), (factors[k] || []).map((lv) => h("option", { value: levelLabel(lv) }, levelLabel(lv))));
        sel.value = ui.filters.cond[k] || "";
        sel.addEventListener("change", () => {
          ui.filters.cond[k] = sel.value;
          drawTable();
        });
        return sel;
      }),
      (() => {
        const sel = h("select", { class: "input", "aria-label": "Filter by status" }, h("option", { value: "" }, "status: all"), ["complete", "running", "failed"].map((st) => h("option", { value: st }, st)));
        sel.value = ui.filters.status;
        sel.addEventListener("change", () => {
          ui.filters.status = sel.value;
          drawTable();
        });
        return sel;
      })(),
      (() => {
        const inp = h("input", { type: "search", class: "input", placeholder: "dyad id contains…", value: ui.filters.search, "aria-label": "Search dyad ids" });
        inp.addEventListener("input", () => {
          ui.filters.search = inp.value;
          drawTable();
        });
        return inp;
      })(),
      (shownCount = h("span", { class: "muted small" })),
    );
    const tableHost = h("div");
    const drawTable = () => {
      byId.clear();
      const list = filtered(factorKeys);
      tbody = h("tbody");
      shownCount.textContent = `${fmtInt(list.length)} of ${fmtInt(dyads.length)} dyads`;
      if (!list.length) {
        mount(tableHost, h("p", { class: "muted" }, dyads.length ? "No dyad matches the filters." : "No dyad has started yet."));
        return;
      }
      showReason = dyads.some((d) => d.reason);
      pagedRows(tbody, list, (d) => dyadRow(d, factorKeys), 500, factorKeys.length + 7);
      mount(
        tableHost,
        h(
          "div",
          { class: "table-wrap tall" },
          h(
            "table",
            { class: "table sticky dyad-table" },
            h("thead", null, h("tr", null, h("th", null, "dyad"), factorKeys.map((k) => h("th", null, k)), h("th", null, "mode"), h("th", { class: "num" }, "attempt"), h("th", null, "status"), h("th", null, "turns"), showReason ? h("th", null, "reason") : null, h("th", null, "last"))),
            tbody,
          ),
        ),
      );
    };
    drawTable();
    mount(body, filterBar, tableHost);
    body._redrawTable = drawTable;
  };

  // Fold newly tailed rows into the dyad list: status rows move a dyad's attempt and status, mentor turn rows
  // its progress. Each touched table row is redrawn once; an unknown dyad means the summary is stale.
  const applyRows = (rows) => {
    const index = new Map(dyads.map((d) => [d.dyad_id, d]));
    const touched = new Set();
    let changedStatus = false;
    let unknown = false;
    for (const r of rows.status || []) {
      const d = index.get(r.dyad_id);
      if (!d) {
        unknown = true;
        continue;
      }
      if ((r.attempt || 0) < (d.attempt || 0)) continue;
      if ((r.attempt || 0) > (d.attempt || 0)) {
        d.attempt = r.attempt;
        d.attempts = Math.max(d.attempts || 1, r.attempt);
        d.turns_done = 0;
      }
      d.status = r.status === "started" ? "running" : r.status;
      d.reason = r.status === "failed" ? r.reason || null : null;
      d.last_ts = r.ts || d.last_ts;
      changedStatus = true;
      touched.add(d.dyad_id);
    }
    for (const r of rows.turns || []) {
      const d = index.get(r.dyad_id);
      if (!d) {
        unknown = true;
        continue;
      }
      if (r.attempt !== undefined && d.attempt !== undefined && r.attempt !== d.attempt) continue;
      const ok = !r.error && r.finish_reason !== "error";
      if (ok && r.agent === "mentor" && Number(r.turn) > (d.turns_done || 0)) d.turns_done = Number(r.turn);
      d.last_ts = r.ts || d.last_ts;
      touched.add(d.dyad_id);
    }
    if (touched.size) {
      const factorKeys = Object.keys((rs.summary && rs.summary.factors) || {});
      for (const id of touched) {
        const e = byId.get(id);
        if (e && e.tr.isConnected) e.tr.replaceWith(dyadRow(index.get(id), factorKeys));
      }
    }
    if (changedStatus) drawStatus(rs.summary);
    if (unknown) refreshSummary();
  };

  let refreshing = false;
  const refreshSummary = async () => {
    if (refreshing) return;
    refreshing = true;
    try {
      const s = await loadSummary(runId);
      if (ctx.alive()) onSummary(s);
    } catch (err) {
      toastError(err, `Run ${runId}`);
    } finally {
      refreshing = false;
    }
  };

  const onSummary = (s) => {
    dyads = (s.dyads || []).map((d) => ({ ...d }));
    drawStatus(s);
    drawInfo(s);
    if (ui.tab === "dyads" && body._redrawTable) body._redrawTable();
  };

  const drawBody = () => {
    body._redrawTable = null;
    if (ui.tab === "analysis") renderAnalysis(body, ctx, runId);
    else if (ui.tab === "actions") drawActions(body, ctx, runId);
    else drawDyads();
  };

  (async () => {
    try {
      const cached = rs.summaryRunId === runId && !!rs.summary;
      const s = cached ? rs.summary : await loadSummary(runId);
      if (!ctx.alive()) return;
      dyads = (s.dyads || []).map((d) => ({ ...d }));
      mount(head, h("div", { class: "run-title" }, h("h1", { class: "mono title-id" }, runId), s.has_scores ? badge("scored", "info") : null, s.has_flags ? badge("flags", "neutral") : null), statusHost);
      drawStatus(s);
      drawInfo(s);
      drawTabs();
      drawBody();
      if (cached) refreshSummary();
      liveStop = startLive(ctx, runId, { onRows: applyRows, onSummary, isOn: () => rs.auto && (ui.tab === "dyads" || ui.tab === "actions") });
      ctx.cleanup(() => liveStop && liveStop());
    } catch (err) {
      if (!ctx.alive()) return;
      toastError(err, `Could not open run ${runId}`);
      mount(head, h("h1", { class: "mono" }, runId), h("div", { class: "error-box" }, err.message, err.status === 404 ? ` (data dir: ${currentDataDir()})` : ""));
      mount(body, dataDirBar(ctx, () => ctx.rerender()));
    }
  })();
}

// ---- actions ---------------------------------------------------------------------------------------------------

function drawActions(body, ctx, runId) {
  if (actions.runId !== runId) {
    actions.runId = runId;
    actions.config = null;
    actions.judgeConfig = null;
    actions.jobs = [];
  }
  const s = rs.summary || {};
  const f = actions.form;
  const jobsHost = h("div", { class: "job-list" });
  const drawJobs = () => mount(jobsHost, actions.jobs.map((j) => jobConsole(j, { alive: ctx.alive, showRunLink: false })));

  const configBox = h("div");
  const drawConfig = () => {
    mount(
      configBox,
      kv([
        ["config", actions.config ? h("span", null, mono(actions.config)) : h("span", { class: "muted" }, "none yet")],
        ["second judge", actions.judgeConfig ? mono(actions.judgeConfig) : h("span", { class: "muted" }, "none")],
      ]),
    );
  };
  const makeConfig = async (judge) => {
    try {
      const res = await api.runConfig(runId, currentDataDir(), judge);
      if (judge) actions.judgeConfig = res.path;
      else actions.config = res.path;
      drawActions(body, ctx, runId);
    } catch (err) {
      toastError(err, "Could not derive a config");
    }
  };
  const jUrl = h("input", { type: "text", class: "input mono", placeholder: "http://127.0.0.1:8098", value: actions.judge.url, "aria-label": "Second judge url" });
  const jGguf = h("input", { type: "text", class: "input mono", placeholder: "(from /props)", value: actions.judge.gguf_path, "aria-label": "Second judge gguf_path" });
  jUrl.addEventListener("input", () => (actions.judge.url = jUrl.value.trim()));
  jGguf.addEventListener("input", () => (actions.judge.gguf_path = jGguf.value.trim()));
  drawConfig();

  const configChoice = () => {
    const sel = h("select", { class: "input", "aria-label": "Config" });
    if (actions.config) sel.appendChild(h("option", { value: actions.config }, "run config (its judge)"));
    if (actions.judgeConfig) sel.appendChild(h("option", { value: actions.judgeConfig }, "second-judge config"));
    if (!sel.options.length) sel.appendChild(h("option", { value: "" }, "make a config first"));
    return sel;
  };

  const start = async (kind, fields) => {
    if (!fields.config) return toast("Make a config from this run first (Config, on the left).", { kind: "error", timeout: 6000 });
    try {
      const job = await api.startJob({ kind, run_id: runId, ...fields });
      actions.jobs = [job, ...actions.jobs.filter((j) => j.id !== job.id)].slice(0, 3);
      drawJobs();
    } catch (err) {
      toastError(err, `Could not start ${kind}`);
    }
  };

  const num = (name, placeholder, step = "any") => {
    const el = h("input", { type: "number", class: "input input-num", step, placeholder, value: f[name] ?? "", "aria-label": name });
    el.addEventListener("input", () => (f[name] = el.value));
    return el;
  };
  const sel = (name, options, def) => {
    const el = h("select", { class: "input", "aria-label": name }, options.map((o) => (Array.isArray(o) ? h("option", { value: o[0] }, o[1]) : h("option", { value: o }, o))));
    el.value = f[name] ?? def ?? (Array.isArray(options[0]) ? options[0][0] : options[0]);
    el.addEventListener("change", () => (f[name] = el.value));
    return el;
  };

  // score
  const scoreCfg = configChoice();
  const scope = sel("scope", [["pilot", "pilot: every turn, both agents"], ["main", "main: seeker on turns 4, 8, … + final"], ["stance", "stance: mentor alignment, same cadence"]], "main");
  const subsample = num("subsample", "e.g. 0.1");
  // flags
  const flagCfg = configChoice();
  const threshold = num("threshold", "required");
  const flagMetric = sel("flag_metric", ["prompt_to_line", "line_to_line"], "prompt_to_line");
  const runLength = num("run_length", "3", "1");
  const judgeSel = sel("flag_judge", [["", "(only judge)"], ...((s.judge_shas || []).map((j) => [j, j.slice(0, 12)]))], "");
  // agreement
  const agrCfg = configChoice();
  const agrMetric = sel("agr_metric", [["", "(default)"], ...((s.metrics && s.metrics.length ? s.metrics : ["alignment", "prompt_to_line", "line_to_line"]).map((m) => [m, m]))], "");
  // survey
  const svCfg = configChoice();
  const phase = sel("phase", ["post", "pre"], "post");

  const numOrNull = (v) => (v === undefined || v === null || String(v).trim() === "" ? null : Number(v));

  mount(
    body,
    h(
      "div",
      { class: "grid-2" },
      card(
        "Config",
        { help: "score, flags, agreement and survey take a config. It is derived from this run's manifest.json → config, so the run-affecting settings match the run (workspace/configs/<run_id>.json)." },
        configBox,
        h("div", { class: "card-foot" }, button("Make config from this run", () => makeConfig(null), { kind: "primary", small: true })),
        h("h3", null, "Second judge (optional)"),
        h("p", { class: "muted small" }, "For the two-judge stance subsample and agreement: the same config with the judge replaced."),
        h("div", { class: "form-row" }, h("label", { class: "field" }, h("span", { class: "field-label" }, "judge url"), jUrl), h("label", { class: "field" }, h("span", { class: "field-label" }, "gguf_path"), jGguf)),
        h("div", { class: "card-foot" }, button("Make second-judge config", () => {
          if (!actions.judge.url) return toast("A judge url is required.", { kind: "error", timeout: 6000 });
          makeConfig({ url: actions.judge.url, gguf_path: actions.judge.gguf_path || null });
        }, { small: true })),
      ),
      h(
        "div",
        { class: "stack" },
        card(
          "Score",
          { help: "Runs the judge over the transcripts. Idempotent per judge: rows already scored are skipped." },
          h("div", { class: "form-row" }, h("label", { class: "field" }, h("span", { class: "field-label" }, "config"), scoreCfg), h("label", { class: "field grow" }, h("span", { class: "field-label" }, "scope"), scope), h("label", { class: "field" }, h("span", { class: "field-label" }, "subsample"), subsample)),
          button("Score", () => start("score", { config: scoreCfg.value, scope: scope.value, ...(numOrNull(subsample.value) !== null ? { subsample: numOrNull(subsample.value) } : {}) }), { kind: "primary", small: true }),
        ),
        card(
          "Flags",
          { help: "Flags dialogues where seeker adherence stays under the threshold for run_length consecutive scored seeker turns. The threshold is calibrated on pilot hand labels: there is no default. Flagged dialogues are kept, not excluded." },
          h(
            "div",
            { class: "form-row" },
            h("label", { class: "field" }, h("span", { class: "field-label" }, "config"), flagCfg),
            h("label", { class: "field" }, h("span", { class: "field-label" }, "threshold"), threshold),
            h("label", { class: "field" }, h("span", { class: "field-label" }, "metric"), flagMetric),
            h("label", { class: "field" }, h("span", { class: "field-label" }, "run length"), runLength),
            (s.judge_shas || []).length > 1 ? h("label", { class: "field" }, h("span", { class: "field-label" }, "judge"), judgeSel) : null,
          ),
          button("Flag", () => {
            const t = numOrNull(threshold.value);
            if (t === null || !Number.isFinite(t)) return toast("A threshold is required: it is calibrated on the pilot's hand labels, so there is no default.", { kind: "error", timeout: 6000 });
            const fields = { config: flagCfg.value, threshold: t, metric: flagMetric.value };
            const rl = numOrNull(runLength.value);
            if (rl !== null) fields.run_length = rl;
            if (judgeSel.value) fields.judge = judgeSel.value;
            start("flags", fields);
          }, { kind: "primary", small: true }),
        ),
        card(
          "Agreement",
          { help: "Cross-judge agreement on the dyads both judges scored." },
          h("div", { class: "form-row" }, h("label", { class: "field" }, h("span", { class: "field-label" }, "config"), agrCfg), h("label", { class: "field" }, h("span", { class: "field-label" }, "metric"), agrMetric)),
          button("Agreement", () => start("agreement", { config: agrCfg.value, ...(agrMetric.value ? { metric: agrMetric.value } : {}) }), { kind: "primary", small: true }),
        ),
        card(
          "Re-administer survey",
          { help: "Asks the mentor the survey again for every complete dyad; rows get origin = readministered (analysis uses origin = run)." },
          h("div", { class: "form-row" }, h("label", { class: "field" }, h("span", { class: "field-label" }, "config"), svCfg), h("label", { class: "field" }, h("span", { class: "field-label" }, "phase"), phase)),
          button("Re-administer", () => start("survey", { config: svCfg.value, phase: phase.value }), { kind: "primary", small: true }),
        ),
      ),
    ),
    jobsHost,
  );
  drawJobs();
}
