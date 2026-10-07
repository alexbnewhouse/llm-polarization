// Study view: name and description, how big the design is (axes, cells, dialogues, messages, compute),
// the cell table, and every validation issue with its path.

import { h, mount, badge, fmtInt, fmtDuration, plural, pagedRows, storageGet, storageSet } from "../ui.js";
import { state, issues, hasErrors } from "../state.js";
import { field } from "../forms.js";
import { viewHeader, card, issueRow, sortIssues } from "./common.js";

function statCard(label, value, detail, extra) {
  return h("div", { class: "stat" }, h("div", { class: "stat-label" }, label), h("div", { class: "stat-value" }, value), detail ? h("div", { class: "stat-detail" }, detail) : null, extra);
}

function selectedRows(cells) {
  if (!cells) return null;
  let n = 0;
  let any = false;
  for (const c of cells) {
    if (c.selected === false) continue;
    any = true;
    n += Number(c.n_rows) || 0;
  }
  return any ? n : 0;
}

function fileLabel() {
  if (state.path) return state.path;
  if (state.kind === "preset" && state.name === "repo") {
    const src = (state.meta && state.meta.repo_sources) || {};
    const files = ["grid", "catalogue", "batteries", "config"].map((k) => src[k]).filter(Boolean);
    return files.length ? files.join(", ") : "built from the repo's files";
  }
  if (state.kind === "preset") return "built in (a preset)";
  return state.name ? `studies/${state.name}.study.json` : "not saved yet";
}

export function render(root, ctx) {
  const spec = state.spec;
  const statsHost = h("div", { class: "stats" });
  const cellsHost = h("div");
  const issuesHost = h("div");
  const shapeHost = h("div", { class: "shape-line" });

  const tpm = h("input", { type: "number", class: "input input-num", min: "1", step: "1", value: String(storageGet("tokPerMsg", 200)), "aria-label": "Tokens per message" });
  const tps = h("input", { type: "number", class: "input input-num", min: "0.1", step: "any", value: String(storageGet("tokPerSec", 100)), "aria-label": "Tokens per second" });
  for (const [el, key] of [[tpm, "tokPerMsg"], [tps, "tokPerSec"]]) {
    el.addEventListener("input", () => {
      const v = Number(el.value);
      if (Number.isFinite(v) && v > 0) storageSet(key, v);
      drawStats();
    });
  }

  mount(
    root,
    viewHeader("Study", "A study is a set of crossed axes, persona templates, a survey and models. Edits stay in this page until you Save."),
    card(
      "About",
      { help: null },
      h(
        "div",
        { class: "grid-2" },
        field("Name", "name", { help: "The study's title, recorded in the export's assignment log." }),
        h(
          "div",
          { class: "about-meta" },
          h("div", { class: "muted small" }, "File"),
          h("div", { class: "mono small" }, fileLabel()),
          h("div", { class: "muted small" }, "Kind"),
          h("div", null, state.kind === "preset" ? (state.name === "repo" ? "preset, rebuilt from the repo's files on every load" : "preset (Save as to keep a copy)") : state.kind === "saved" ? "saved study" : "new, unsaved"),
          shapeHost,
        ),
      ),
      field("Description", "description", { type: "textarea", rows: 2, optional: true, emptyDelete: false }),
    ),
    card("Size", { help: "From the server's summary of the spec as it is now. Messages are dialogues × turns × 2 (seeker and mentor)." }, statsHost, h(
      "div",
      { class: "compute-inputs" },
      h("label", { class: "inline-field" }, h("span", null, "Tokens per message"), tpm),
      h("label", { class: "inline-field" }, h("span", null, "Aggregate tok/s"), tps),
      h("span", { class: "muted small" }, "Generation only, summed over the server's parallel slots; prefill is assumed cached (KV reuse). See models/RUN_APPROACH.md for measured tok/s per arm."),
    )),
    card("Cells", { help: "Treated cells are the cartesian product of the factors' levels; each control cell is one combination of the control's `by` levels." }, cellsHost),
    h("section", { class: "card", dataset: { path: "#issues" } }, h("div", { class: "card-head" }, h("h2", null, "Issues")), issuesHost),
  );

  function drawStats() {
    const v = state.validation;
    const sm = v && v.summary;
    if (!sm) {
      mount(statsHost, h("p", { class: "muted" }, v ? "No summary: the spec could not be read. See the issues below." : "Checking…"));
      return;
    }
    const factors = sm.factors || [];
    const axesDetail = factors.map((f) => `${f.key} ${f.n_levels}`).join(" × ") || "no factors";
    let nestedDetail = null;
    if (sm.nested) {
      const counts = Object.values(sm.nested.n_variants || {});
      const lo = counts.length ? Math.min(...counts) : 0;
      const hi = counts.length ? Math.max(...counts) : 0;
      nestedDetail = `${sm.nested.key} nested in ${sm.nested.within}: ${lo === hi ? lo : `${lo}–${hi}`} per level`;
    }
    const subset = !!(spec.randomization && Array.isArray(spec.randomization.cells));
    const rowsDesign = sm.rows_design !== undefined && sm.rows_design !== null ? Number(sm.rows_design) : null;
    const sel = selectedRows(state.cells);
    // rows: what an export writes (the selected cells); rows_design: the full design.
    const runRows = Number(sm.rows) || (subset && sel !== null ? sel : 0);
    const rows = rowsDesign !== null ? rowsDesign : subset ? null : runRows;
    const nTurns = Number(sm.n_turns) || 0;
    const runMessages = sm.messages !== undefined && sm.messages !== null ? Number(sm.messages) : runRows * nTurns * 2;
    const perMsg = Number(tpm.value) > 0 ? Number(tpm.value) : 200;
    const perSec = Number(tps.value) > 0 ? Number(tps.value) : 100;
    const seconds = (runMessages * perMsg) / perSec;
    const modes = Object.entries(sm.rows_per_mode || {}).map(([m, n]) => `${fmtInt(n)} ${m}`).join(", ");
    mount(
      statsHost,
      statCard("Axes", String(factors.length), axesDetail, nestedDetail ? h("div", { class: "stat-detail" }, nestedDetail) : null),
      statCard(
        "Cells",
        fmtInt((Number(sm.cells_treated) || 0) + (Number(sm.cells_control) || 0)),
        `${fmtInt(sm.cells_treated)} treated + ${fmtInt(sm.cells_control || 0)} control`,
      ),
      statCard("Dialogues", fmtInt(rows ?? runRows), modes || null, subset ? h("div", { class: "stat-detail accent" }, `subset: ${fmtInt(runRows)} rows in ${fmtInt(sm.cells_selected ?? 0)} cells`) : null),
      statCard("Messages", fmtInt(runMessages), `${plural(nTurns, "turn")} × 2 per dialogue${subset ? " (subset)" : ""}`),
      statCard(
        "Compute",
        fmtDuration(seconds),
        `${fmtInt(runMessages * perMsg)} tokens at ${perSec} tok/s${subset ? " (subset)" : ""}`,
      ),
    );
  }

  function drawShape() {
    const v = state.validation;
    if (!v) return mount(shapeHost);
    mount(
      shapeHost,
      h("div", { class: "muted small" }, "Export engine"),
      v.repo_shaped
        ? h("div", null, badge("harness.randomize", "repo"), " the repo's own randomizer: this is the exact repo study")
        : h("div", null, badge("sandbox.study", "neutral"), v.repo_shape_reason ? h("span", { class: "muted small" }, ` ${v.repo_shape_reason}`) : null),
    );
  }

  function drawCells() {
    if (state.cells === null) {
      mount(cellsHost, h("p", { class: "muted" }, state.cellsError ? `Cells unavailable: ${state.cellsError}` : "Checking…"));
      return;
    }
    const cells = state.cells;
    if (!cells.length) {
      mount(cellsHost, h("p", { class: "muted" }, "No cells: add a factor with at least one level."));
      return;
    }
    const sm = state.validation && state.validation.summary;
    // Columns: the condition keys the cells actually carry, in condition-key order.
    const present = new Set();
    for (const c of cells) for (const k of Object.keys(c.condition || {})) present.add(k);
    const keys = [];
    for (const k of [...((sm && sm.condition_keys) || []), ...present]) if (present.has(k) && !keys.includes(k)) keys.push(k);
    const subset = spec.randomization && Array.isArray(spec.randomization.cells);
    const tbody = h("tbody");
    const total = cells.reduce((a, c) => a + (Number(c.n_rows) || 0), 0);
    const table = h(
      "table",
      { class: "table sticky" },
      h("thead", null, h("tr", null, h("th", null, "Kind"), h("th", null, "Cell"), keys.map((k) => h("th", null, k)), h("th", null, "Variants"), h("th", { class: "num" }, "Rows"), subset ? h("th", null, "In subset") : null)),
      tbody,
      h("tfoot", null, h("tr", null, h("td", { colspan: 2 + keys.length + 1 }, `${fmtInt(cells.length)} cells`), h("td", { class: "num" }, fmtInt(total)), subset ? h("td") : null)),
    );
    pagedRows(
      tbody,
      cells,
      (c) =>
        h(
          "tr",
          { class: [c.kind === "control" ? "row-control" : null, subset && c.selected === false ? "row-dim" : null] },
          h("td", null, badge(c.kind || "treated", c.kind === "control" ? "neutral" : "info")),
          h("td", { class: "mono" }, c.key),
          keys.map((k) => h("td", { class: "mono small" }, c.condition && k in c.condition ? (c.condition[k] === null ? "(none)" : String(c.condition[k])) : "—")),
          h("td", { class: "mono small variants-cell", title: (c.variants || []).join(", ") }, (c.variants || []).length ? (c.variants || []).join(", ") : "—"),
          h("td", { class: "num" }, fmtInt(c.n_rows)),
          subset ? h("td", null, c.selected === false ? "" : "✓") : null,
        ),
      500,
      keys.length + 5,
    );
    mount(cellsHost, h("div", { class: "table-wrap tall" }, table));
  }

  function drawIssues() {
    const list = sortIssues(issues());
    if (!state.validation) return mount(issuesHost, h("p", { class: "muted" }, "Checking…"));
    if (!list.length) return mount(issuesHost, h("p", { class: "ok-line" }, "✓ No issues. The study validates and can be exported."));
    mount(
      issuesHost,
      hasErrors() ? h("p", { class: "muted" }, "Errors block export; warnings are for you to judge. Click an issue to go to the field.") : h("p", { class: "muted" }, "Only warnings: export is allowed."),
      h("ul", { class: "issue-list" }, list.map((i) => issueRow(i))),
    );
  }

  const drawAll = () => {
    drawStats();
    drawShape();
    drawCells();
    drawIssues();
  };
  ctx.on("validated", drawAll);
  drawAll();
}
