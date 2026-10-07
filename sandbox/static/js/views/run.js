// Run view: randomization (with the repo's presets), a cell subset, the manifest preview, Export, the
// pre-flight `check`, `run` with a run id, the CLI line for every step, and the job's live console.

import { api } from "../api.js";
import {
  h, mount, button, badge, toast, toastError, debounce, fmtInt, kv, mono, copyButton, commandLine, details, confirmDialog,
} from "../ui.js";
import { state, touch, hasErrors, issueCounts, PERSONA_MODES, clone, mockRoles, isRepoDataDir } from "../state.js";
import { control, field } from "../forms.js";
import { viewHeader, card, issuePanel, mockDataDirWarning } from "./common.js";
import { jobConsole } from "../jobconsole.js";

const RUN_ID_RE = /^[A-Za-z0-9_.-]+$/;
const PRESET_LABELS = { pilot: "Pilot", wave1: "Wave 1" };
const local = { runId: null, cellFilter: "", jobs: [], manifest: null, manifestVersion: -1, openRow: null };

function stamp() {
  const d = new Date();
  const p = (x) => String(x).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

function defaultRunId() {
  const prefix = (state.spec.randomization && state.spec.randomization.prefix) || "s";
  return `sandbox-${prefix}-${stamp()}`;
}

function presetMatches(r, p) {
  return Object.entries(p).every(([k, v]) => JSON.stringify(r[k]) === JSON.stringify(v));
}

function randomizationCard(ctx) {
  const spec = state.spec;
  const r = spec.randomization || (spec.randomization = { seed: 1, n_per_cell: 1, modes: ["reinforced"], n_turns: 10, prefix: "s", cells: null });
  const presets = (state.meta && state.meta.randomization_presets) || {};
  const presetRow =
    Object.keys(presets).length && state.validation && state.validation.repo_shaped
      ? h(
          "div",
          { class: "preset-row" },
          h("span", { class: "muted small" }, "Repo presets (prompts/README.md):"),
          Object.entries(presets).map(([key, p]) => {
            const active = presetMatches(r, p);
            const desc = [
              p.seed !== undefined ? `seed ${p.seed}` : null,
              p.n_per_cell !== undefined ? `${p.n_per_cell}/cell` : null,
              p.modes ? p.modes.join("+") : null,
              p.n_turns !== undefined ? `${p.n_turns} turns` : null,
              p.prefix ? `prefix ${p.prefix}` : null,
            ].filter(Boolean).join(", ");
            return button(
              h("span", null, h("strong", null, PRESET_LABELS[key] || key), h("span", { class: "muted small" }, ` ${desc}`)),
              () => {
                spec.randomization = { ...r, ...clone(p) };
                touch();
                ctx.rerender();
              },
              { small: true, class: active ? "active" : null, title: `Set the randomization to the ${PRESET_LABELS[key] || key} preset` },
            );
          }),
        )
      : null;
  const modes = Array.isArray(r.modes) ? r.modes : [];
  const modeBoxes = h(
    "div",
    { class: "check-row", dataset: { path: "randomization.modes" } },
    PERSONA_MODES.map((m) => {
      const cb = h("input", { type: "checkbox", checked: modes.includes(m) });
      cb.addEventListener("change", () => {
        const set = new Set(Array.isArray(r.modes) ? r.modes : []);
        if (cb.checked) set.add(m);
        else set.delete(m);
        r.modes = PERSONA_MODES.filter((x) => set.has(x)).concat([...set].filter((x) => !PERSONA_MODES.includes(x)));
        touch();
      });
      return h("label", { class: "field-check" }, cb, h("span", null, m));
    }),
  );
  const c = spec.control;
  let controlN = null;
  if (c) {
    const same = c.n_per_cell === null || c.n_per_cell === undefined;
    const sameBox = h("input", { type: "checkbox", checked: same });
    sameBox.addEventListener("change", () => {
      c.n_per_cell = sameBox.checked ? null : Number(r.n_per_cell) || 1;
      touch();
      ctx.rerender();
    });
    controlN = h(
      "div",
      { class: "field" },
      h("span", { class: "field-label" }, "control n per cell"),
      h("div", { class: "inline-group" }, control("control.n_per_cell", { type: "int", min: 1, disabled: same, ariaLabel: "Control n per cell" }), h("label", { class: "field-check" }, sameBox, h("span", null, "same"))),
      h("span", { class: "field-help" }, "Control rows are always one mode"),
    );
  }
  return card(
    "Randomization",
    { path: "randomization", help: "One RNG seeded with seed draws every dyad's seed in design order, then shuffles the rows so cells interleave across the servers' slots. A row's text and seed depend only on the design and the seed." },
    presetRow,
    h(
      "div",
      { class: "form-row" },
      field("seed", "randomization.seed", { type: "int" }),
      field("n per cell", "randomization.n_per_cell", { type: "int", min: 1, help: "Split evenly across a level's variants" }),
      controlN,
      field("n_turns", "randomization.n_turns", { type: "int", min: 1, help: "Exchanges, not messages" }),
      field("prefix", "randomization.prefix", { mono: true, help: "Starts every dyad_id" }),
    ),
    h("div", { class: "field" }, h("span", { class: "field-label" }, "persona modes"), modeBoxes, h("span", { class: "field-help" }, "once: system prompt only. reinforced: the reminder is appended after the history on every seeker turn. Treated rows are multiplied by modes.")),
    longRunHint(ctx),
  );
}

function longRunHint(ctx) {
  const host = h("div", { class: "hint-box" });
  const draw = () => {
    const spec = state.spec;
    const nTurns = Number(spec.randomization && spec.randomization.n_turns) || 0;
    const nPredict = Number(spec.run && spec.run.generation && spec.run.generation.n_predict) || 0;
    const perSlot = nTurns * 2 * nPredict + 2048;
    const limit = spec.run ? spec.run.cache_reuse_limit : null;
    mount(
      host,
      h("strong", null, "Long run: "),
      `${fmtInt(nTurns * 2)} messages per dyad. A dialogue needs ≈ ${fmtInt(perSlot)} tokens of context per slot (n_turns × 2 × n_predict + 2,048), so a server with 8 slots needs -c ≈ ${fmtInt(perSlot * 8)}. `,
      "`check --manifest` verifies the fit against each server before anything runs. ",
      limit !== null && limit !== undefined && Number(limit) <= 2 * nPredict
        ? h("span", { class: "warn-text" }, `cache_reuse_limit (${limit}) should exceed 2 × n_predict (${2 * nPredict}) plus the reminder.`)
        : null,
    );
  };
  ctx.on("changed", draw);
  draw();
  return host;
}

function subsetCard(ctx) {
  const spec = state.spec;
  const host = h("div");
  const filter = h("input", { type: "search", class: "input", placeholder: "Filter cells, e.g. immigration", value: local.cellFilter, "aria-label": "Filter cells" });
  const draw = () => {
    const r = spec.randomization || {};
    const cells = state.cells;
    if (!cells) {
      mount(host, h("p", { class: "muted" }, state.cellsError ? `Cells unavailable: ${state.cellsError}` : "Waiting for the cells…"));
      return;
    }
    const all = cells.map((c) => c.key);
    const sel = new Set(Array.isArray(r.cells) ? r.cells : all);
    const q = local.cellFilter.trim().toLowerCase();
    const visible = cells.filter((c) => !q || c.key.toLowerCase().includes(q));
    const rowsSel = cells.reduce((a, c) => a + (sel.has(c.key) ? Number(c.n_rows) || 0 : 0), 0);
    const rowsAll = cells.reduce((a, c) => a + (Number(c.n_rows) || 0), 0);
    const commit = () => {
      r.cells = sel.size === all.length && all.every((k) => sel.has(k)) ? null : all.filter((k) => sel.has(k));
      spec.randomization = r;
      touch();
      draw();
    };
    mount(
      host,
      h(
        "div",
        { class: "subset-bar" },
        Array.isArray(r.cells) ? badge(`subset: ${sel.size} of ${all.length} cells`, "info") : badge("all cells", "neutral"),
        h("span", { class: "muted small" }, `${fmtInt(rowsSel)} of ${fmtInt(rowsAll)} rows`),
        filter,
        button(q ? "Select shown" : "Select all", () => {
          for (const c of visible) sel.add(c.key);
          commit();
        }, { small: true }),
        button(q ? "Clear shown" : "Select none", () => {
          for (const c of visible) sel.delete(c.key);
          commit();
        }, { small: true }),
        Array.isArray(r.cells)
          ? button("Use all (null)", () => {
              r.cells = null;
              touch();
              draw();
            }, { small: true })
          : null,
      ),
      h(
        "div",
        { class: "cell-checks", dataset: { path: "randomization.cells" } },
        visible.map((c) => {
          const cb = h("input", { type: "checkbox", checked: sel.has(c.key) });
          cb.addEventListener("change", () => {
            if (cb.checked) sel.add(c.key);
            else sel.delete(c.key);
            commit();
          });
          return h("label", { class: ["cell-check", c.kind === "control" ? "is-control" : null] }, cb, h("span", { class: "mono" }, c.key), h("span", { class: "muted small" }, fmtInt(c.n_rows)));
        }),
      ),
    );
  };
  filter.addEventListener("input", () => {
    local.cellFilter = filter.value;
    const pos = filter.selectionStart;
    draw();
    filter.focus();
    try {
      filter.setSelectionRange(pos, pos);
    } catch {
      /* not all input types support it */
    }
  });
  ctx.on("validated", draw);
  draw();
  return card(
    "Cells to run",
    { help: "A subset is applied after the shuffle, so its rows are byte-identical to the same cells' rows in the full design: this is how a two-cell pilot at 100 turns is made." },
    host,
  );
}

function conditionChips(cond) {
  return h(
    "span",
    { class: "cond" },
    Object.entries(cond || {}).map(([k, v]) => h("span", { class: "cond-item" }, h("span", { class: "muted" }, `${k}=`), h("span", { class: "mono" }, v === null ? "null" : String(v)))),
  );
}

function manifestCard(ctx) {
  const host = h("div");
  let seq = 0;
  const show = () => {
    const m = local.manifest;
    if (!m) return;
    const rows = m.rows || [];
    const tbody = h("tbody");
    for (const r of rows) {
      const tr = h(
        "tr",
        { class: "clickable", title: "Show the persona", onclick: () => {
          local.openRow = local.openRow === r.dyad_id ? null : r.dyad_id;
          show();
        } },
        h("td", { class: "mono" }, r.dyad_id),
        h("td", null, conditionChips(r.condition)),
        h("td", null, r.persona_mode),
        h("td", { class: "num mono" }, String(r.seed)),
        h("td", { class: "num" }, String(r.n_turns)),
      );
      tbody.appendChild(tr);
      if (local.openRow === r.dyad_id) {
        tbody.appendChild(
          h(
            "tr",
            { class: "detail-row" },
            h(
              "td",
              { colspan: 5 },
              h("div", { class: "preview-label" }, "Persona"),
              h("div", { class: "persona-text" }, r.persona_text || ""),
              h("div", { class: "preview-label" }, "Reminder"),
              h("div", { class: "persona-text reminder" }, r.persona_reminder || h("span", { class: "muted" }, "(none)")),
            ),
          ),
        );
      }
    }
    const a = m.assignment || {};
    mount(
      host,
      h("p", { class: "muted small" }, `Showing the first ${rows.length} of ${fmtInt(m.total)} rows, in file order (after the shuffle${a.cells_filter ? " and the subset" : ""}). Click a row for its persona.`),
      h(
        "div",
        { class: "table-wrap" },
        h("table", { class: "table" }, h("thead", null, h("tr", null, h("th", null, "dyad_id"), h("th", null, "condition"), h("th", null, "mode"), h("th", { class: "num" }, "seed"), h("th", { class: "num" }, "turns"))), tbody),
      ),
      details(
        "Assignment summary",
        kv(
          ["engine", "rng_seed", "n_per_cell", "n_control", "modes", "n_turns", "prefix", "rows", "rows_per_mode", "variants_per_level", "cells_filter"]
            .filter((k) => k in a)
            .map((k) => [k, h("span", { class: "mono small" }, typeof a[k] === "object" && a[k] !== null ? JSON.stringify(a[k]) : String(a[k]))]),
        ),
      ),
    );
  };
  const load = async () => {
    if (!state.validation) return mount(host, h("p", { class: "muted" }, "Checking…"));
    if (hasErrors()) {
      local.manifest = null;
      return mount(host, h("p", { class: "muted" }, `Fix the ${issueCounts().errors} error(s) to preview the manifest.`));
    }
    if (local.manifest && local.manifestVersion === state.validatedVersion) return show();
    const my = ++seq;
    host.classList.add("stale");
    try {
      const res = await api.manifest(state.spec, 20);
      if (my !== seq || !ctx.alive()) return;
      local.manifest = res;
      local.manifestVersion = state.validatedVersion;
      show();
    } catch (err) {
      if (my !== seq || !ctx.alive()) return;
      toastError(err, "Manifest preview");
      mount(host, h("div", { class: "error-box" }, err.message));
    } finally {
      host.classList.remove("stale");
    }
  };
  const later = debounce(load, 300);
  ctx.on("validated", later);
  ctx.cleanup(() => later.cancel());
  load();
  return card("Manifest preview", { help: "POST /api/study/manifest: the rows the export will write (the first 20)." }, host);
}

function exportResult(ctx) {
  const ex = state.lastExport;
  if (!ex) return null;
  const res = ex.result || {};
  const stale = JSON.stringify(state.spec) !== ex.specJson;
  const unchanged = res.unchanged_from_repo || {};
  // The exact repo study only when the exported grid, catalogue and batteries are the repo's own files (or the
  // server says so outright); the repo's randomizer alone is a structural match.
  const repoExact =
    res.engine === "harness.randomize" &&
    (typeof res.repo_exact === "boolean" ? res.repo_exact : ["grid", "catalogue", "batteries"].every((k) => unchanged[k] === true));
  const files = ["manifest", "assignment", "config", "batteries", "grid", "catalogue", "study"].filter((k) => res[k]);
  const cmds = res.commands || {};
  const root = state.meta && state.meta.root;
  return h(
    "div",
    { class: "export-result" },
    stale ? h("div", { class: "banner banner-warn inline" }, "The study changed since this export. Export again before running, or the run uses the exported design.") : null,
    h(
      "div",
      { class: "export-head" },
      h("span", { class: "muted small" }, "Exported to"),
      h("code", { class: "mono" }, res.dir || "?"),
      copyButton(res.dir || ""),
    ),
    kv([
      [
        "engine",
        res.engine === "harness.randomize"
          ? repoExact
            ? h("span", null, badge("harness.randomize", "repo"), " the repo's randomizer and files: the exact repo study")
            : h("span", null, badge("harness.randomize", "neutral"), " the repo's randomizer; the design differs from the repo's files (below)")
          : badge(res.engine || "?", "neutral"),
      ],
      ["rows", fmtInt(res.rows)],
      ["cells", res.cells_filter ? h("span", { class: "mono small" }, Array.isArray(res.cells_filter) ? res.cells_filter.join(", ") : JSON.stringify(res.cells_filter)) : "all"],
      [
        "from the repo",
        h(
          "span",
          { class: "chips" },
          ["grid", "catalogue", "batteries"].map((k) =>
            unchanged[k] === true
              ? badge(`${k}: repo file reused`, "ok", "Identical to the file in the repo; the export points at it, so the hashes match a CLI-made manifest")
              : unchanged[k] === false
                ? badge(`${k}: written to export`, "warn", "Differs from the repo's file; a copy was written into the export directory")
                : badge(`${k}: n/a`, "neutral"),
          ),
        ),
      ],
    ]),
    Array.isArray(res.notes) && res.notes.length ? h("ul", { class: "notes" }, res.notes.map((n) => h("li", null, typeof n === "string" ? n : JSON.stringify(n)))) : null,
    details("Files", kv(files.map((k) => [k, mono(res[k])])), false),
    h(
      "div",
      { class: "commands" },
      h("div", { class: "muted small" }, `Each step as a shell line, run from the repo root${root ? ` (${root})` : ""}:`),
      ["randomize", "check", "run", "score"].filter((k) => cmds[k]).map((k) => commandLine(cmds[k], k)),
    ),
  );
}

function launchCard(ctx) {
  const spec = state.spec;
  const exportHost = h("div");
  const jobsHost = h("div", { class: "job-list" });
  const drawExport = () => mount(exportHost, exportResult(ctx) || h("p", { class: "muted" }, "Not exported yet. Export writes the manifest, the assignment log and the config into workspace/exports/."));
  const drawJobs = () => {
    mount(jobsHost, local.jobs.map((j) => jobConsole(j, { alive: ctx.alive, dataDir: spec.run && spec.run.data_dir, onUpdate: (nj) => Object.assign(j, nj) })));
  };
  const exportBtn = button("Export", async () => {
    if (hasErrors()) {
      toast(`Fix the ${issueCounts().errors} error(s) first: errors block export.`, { kind: "error" });
      return;
    }
    exportBtn.disabled = true;
    // Snapshot the spec before the request and send exactly that: edits made while it is in flight are not in
    // the export, so they must count as changes since it (the stale-export warning).
    const exported = state.spec;
    const specJson = JSON.stringify(exported);
    try {
      const res = await api.exportStudy(JSON.parse(specJson));
      if (state.spec !== exported) {
        toast(`Exported ${fmtInt(res.rows)} rows of the study that was open (${res.dir || "?"}); another study is loaded now.`, { kind: "warn", timeout: 7000 });
        return;
      }
      state.lastExport = { result: res, specJson, at: Date.now() };
      toast(`Exported ${fmtInt(res.rows)} rows`, { kind: "ok" });
      ctx.rerender();
    } catch (err) {
      toastError(err, "Export failed");
    } finally {
      exportBtn.disabled = false;
    }
  }, { kind: "primary", title: "POST /api/study/export" });

  const runIdInput = h("input", { type: "text", class: "input mono", value: local.runId || defaultRunId(), spellcheck: "false", "aria-label": "run id" });
  runIdInput.addEventListener("input", () => (local.runId = runIdInput.value.trim()));
  const runIdErr = h("span", { class: "field-error" });

  const start = async (kind) => {
    const ex = state.lastExport;
    if (!ex) return toast("Export first.", { kind: "error" });
    const res = ex.result;
    const fields = { kind, config: res.config, manifest: res.manifest };
    if (kind === "run") {
      const id = runIdInput.value.trim();
      if (!RUN_ID_RE.test(id)) {
        runIdErr.textContent = "Letters, digits, ., _ and - only.";
        return;
      }
      runIdErr.textContent = "";
      fields.run_id = id;
      local.runId = id;
      const stale = JSON.stringify(state.spec) !== ex.specJson;
      const mock = mockRoles();
      if (stale || mock.length) {
        const exported = (() => {
          try {
            return JSON.parse(ex.specJson);
          } catch {
            return null;
          }
        })();
        const dd = exported && exported.run ? exported.run.data_dir : spec.run && spec.run.data_dir;
        const msg = h(
          "div",
          null,
          stale ? h("p", null, "The study changed since the export; this run uses the exported design, not what is on screen.") : null,
          mock.length ? h("p", { class: "warn-text" }, `The ${mock.join(", ")} point at the MOCK backend: synthetic text, never for data.`) : null,
          mock.length && isRepoDataDir(dd) ? h("p", { class: "warn-text" }, `It writes into ${dd || "data"}/, beside real runs (manifest.json there is git-trackable). Set data_dir in Models and export again to keep it out.`) : null,
          h("p", null, `Start run ${id}?`),
        );
        if (!(await confirmDialog("Start the run?", msg, "Start run"))) return;
      }
    }
    try {
      const job = await api.startJob(fields);
      local.jobs = [job, ...local.jobs.filter((j) => j.id !== job.id)].slice(0, 4);
      drawJobs();
    } catch (err) {
      toastError(err, `Could not start ${kind}`);
    }
  };

  drawExport();
  drawJobs();
  ctx.on("changed", () => {
    const banner = exportHost.querySelector(".banner-warn");
    if (state.lastExport && !banner && JSON.stringify(state.spec) !== state.lastExport.specJson) drawExport();
  });
  const runIdRow = h(
    "div",
    { class: "launch-row" },
    button("Pre-flight check", () => start("check"), { disabled: !state.lastExport, title: "python -m harness.run check --config … --manifest …: parity, cache reuse, context fit" }),
    h("label", { class: "inline-field" }, h("span", null, "run id"), runIdInput),
    button("↻", () => {
      local.runId = defaultRunId();
      runIdInput.value = local.runId;
    }, { small: true, title: "A fresh default run id" }),
    button("Run", () => start("run"), { kind: "primary", disabled: !state.lastExport, title: "python -m harness.run run … --run-id …" }),
    runIdErr,
  );
  const watch = local.runId || runIdInput.value;
  return card(
    "Export and run",
    { help: "Export writes everything a run needs; the run is the real harness CLI as a subprocess, with all of its guards and provenance. Re-running an id resumes it." },
    h("div", { class: "launch-row" }, exportBtn, h("span", { class: "muted small" }, hasErrors() ? `Blocked by ${issueCounts().errors} error(s).` : "")),
    exportHost,
    h("hr"),
    runIdRow,
    h("p", { class: "muted small" }, "Run exits 0 when every dyad completed, 2 when some failed (they are retried as new attempts on resume), 130 when stopped, 1 when it refused to start. ",
      h("a", { class: "link", href: `#/runs/${encodeURIComponent(watch)}${spec.run && spec.run.data_dir ? `?data_dir=${encodeURIComponent(spec.run.data_dir)}` : ""}` }, "Watch run →")),
    jobsHost,
  );
}

export function render(root, ctx) {
  mount(
    root,
    viewHeader("Run", "Randomize, export, check, run. Every step is the harness's own code."),
    issuePanel(ctx, ["randomization"]),
    mockDataDirWarning(ctx),
    randomizationCard(ctx),
    subsetCard(ctx),
    manifestCard(ctx),
    launchCard(ctx),
  );
}
