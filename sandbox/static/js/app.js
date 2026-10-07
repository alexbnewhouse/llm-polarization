// Entry point: the top bar (study picker, New, Save, Save as, validity pill, repo badge), the left nav, the
// hash router and the view lifecycle. Views are modules exporting render(root, ctx).

import { api } from "./api.js";
import {
  h, mount, button, badge, toast, toastError, modal, confirmDialog, promptDialog, storageGet, emptyState, fmtTime,
} from "./ui.js";
import {
  state, on, loadStudy, setStudy, refreshStudies, saveStudy, refreshMock, mockRoles, issues, issueCounts, NAME_RE,
  minimalStudy, clone, readDraft, clearDraft, isRepoShaped, isRepoExact, repoExactNote, isRepoDataDir,
} from "./state.js";
import { VIEW_LABELS, viewForPath, consumePendingFocus, highlightPath, goto } from "./nav.js";
import * as studyView from "./views/study.js";
import * as axesView from "./views/axes.js";
import * as personasView from "./views/personas.js";
import * as instrumentView from "./views/instrument.js";
import * as modelsView from "./views/models.js";
import * as runView from "./views/run.js";
import * as runsView from "./views/runs.js";
import * as jobsView from "./views/jobs.js";

const VIEWS = [
  { id: "study", group: "Design", mod: studyView, needsStudy: true, hint: "Name, size, cells, issues" },
  { id: "axes", group: "Design", mod: axesView, needsStudy: true, hint: "Factors, nested variants, lookup tables" },
  { id: "personas", group: "Design", mod: personasView, needsStudy: true, hint: "Templates, derived slots, control, preview" },
  { id: "instrument", group: "Design", mod: instrumentView, needsStudy: true, hint: "The mentor's survey items" },
  { id: "models", group: "Design", mod: modelsView, needsStudy: true, hint: "Seeker, mentor, judge, sampling" },
  { id: "run", group: "Design", mod: runView, needsStudy: true, hint: "Randomization, export, check, run" },
  { id: "runs", group: "Observe", mod: runsView, needsStudy: false, hint: "Runs on disk, transcripts, analysis" },
  { id: "jobs", group: "Observe", mod: jobsView, needsStudy: false, hint: "Harness subprocesses and their logs" },
];

const $ = (sel) => document.querySelector(sel);

// ---- router ---------------------------------------------------------------------------------------------------

let current = { token: 0, cleanups: [], view: null, root: null };

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, query] = raw.split("?");
  const parts = path.split("/").filter(Boolean).map((p) => {
    try {
      return decodeURIComponent(p);
    } catch {
      return p;
    }
  });
  return { view: parts[0] || "study", parts: parts.slice(1), query: new URLSearchParams(query || "") };
}

function renderRoute({ keepScroll = false } = {}) {
  const route = parseHash();
  const v = VIEWS.find((x) => x.id === route.view) || VIEWS[0];
  for (const fn of current.cleanups) {
    try {
      fn();
    } catch (err) {
      console.error(err);
    }
  }
  const token = current.token + 1;
  const cleanups = [];
  const main = $("#main");
  const scroll = keepScroll ? main.scrollTop : 0;
  const root = h("div", { class: `view view-${v.id}` });
  current = { token, cleanups, view: v.id, root };
  mount(main, root);
  const ctx = {
    route,
    params: route.parts,
    query: route.query,
    alive: () => current.token === token,
    on: (evt, fn) => cleanups.push(on(evt, fn)),
    cleanup: (fn) => cleanups.push(fn),
    rerender: () => {
      if (current.token === token) renderRoute({ keepScroll: true });
    },
    markIssues: () => markIssues(root),
  };
  if (v.needsStudy && !state.spec) {
    mount(root, emptyState("No study is loaded. Pick one in the top bar, or start a new one."));
  } else {
    try {
      v.mod.render(root, ctx);
    } catch (err) {
      console.error(err);
      mount(root, h("div", { class: "error-box" }, h("strong", null, "This view failed to render. "), String(err && err.message ? err.message : err)));
    }
  }
  main.scrollTop = scroll;
  markIssues(root);
  drawNav();
  const focus = consumePendingFocus();
  if (focus) requestAnimationFrame(() => highlightPath(root, focus));
}

// ---- issue marks on bound inputs ---------------------------------------------------------------------------------

function markIssues(root) {
  if (!root) return;
  const byPath = new Map();
  for (const i of issues()) {
    if (!byPath.has(i.path)) byPath.set(i.path, []);
    byPath.get(i.path).push(i);
  }
  for (const el of root.querySelectorAll("[data-path]")) {
    const list = byPath.get(el.dataset.path);
    const err = !!(list && list.some((i) => i.level === "error"));
    el.classList.toggle("has-error", err);
    el.classList.toggle("has-warning", !!list && !err);
    if (list) {
      if (!("origTitle" in el.dataset)) el.dataset.origTitle = el.getAttribute("title") || "";
      el.title = list.map((i) => `${i.level}: ${i.message}`).join("\n");
    } else if ("origTitle" in el.dataset) {
      if (el.dataset.origTitle) el.title = el.dataset.origTitle;
      else el.removeAttribute("title");
      delete el.dataset.origTitle;
    }
  }
}

// ---- left nav ----------------------------------------------------------------------------------------------------

function drawNav() {
  const counts = {};
  for (const i of issues()) {
    const v = viewForPath(i.path);
    counts[v] = counts[v] || { errors: 0, warnings: 0 };
    if (i.level === "error") counts[v].errors++;
    else counts[v].warnings++;
  }
  const groups = [...new Set(VIEWS.map((v) => v.group))];
  mount(
    $("#sidenav"),
    groups.map((g) =>
      h(
        "div",
        { class: "nav-group" },
        h("div", { class: "nav-group-label" }, g),
        VIEWS.filter((v) => v.group === g).map((v) => {
          const c = counts[v.id];
          const disabled = v.needsStudy && !state.spec;
          return h(
            "a",
            {
              class: ["nav-item", current.view === v.id ? "active" : null, disabled ? "disabled" : null],
              href: `#/${v.id}`,
              title: v.hint,
              "aria-current": current.view === v.id ? "page" : undefined,
            },
            h("span", { class: "nav-label" }, VIEW_LABELS[v.id]),
            c && c.errors ? h("span", { class: "nav-count nav-count-error", title: `${c.errors} errors` }, String(c.errors)) : null,
            c && !c.errors && c.warnings ? h("span", { class: "nav-count nav-count-warn", title: `${c.warnings} warnings` }, String(c.warnings)) : null,
          );
        }),
      ),
    ),
    h(
      "div",
      { class: "nav-foot muted small" },
      state.meta ? h("div", null, `harness ${state.meta.harness_version || "?"}`) : null,
      state.meta && state.meta.sandbox_version ? h("div", null, `sandbox ${state.meta.sandbox_version}`) : null,
    ),
  );
}

// ---- top bar -----------------------------------------------------------------------------------------------------

function drawStudyBar() {
  const bar = $("#study-bar");
  const select = h("select", { class: "input study-picker", "aria-label": "Study", title: "Open a study" });
  const presets = state.studies.filter((s) => s.kind === "preset");
  const saved = state.studies.filter((s) => s.kind !== "preset");
  const opt = (s) => h("option", { value: s.name }, `${s.title || s.name}${s.title && s.title !== s.name ? `  (${s.name})` : ""}`);
  if (!state.name) select.appendChild(h("option", { value: "" }, `${state.spec ? state.spec.name || "New study" : "No study"} (new, unsaved)`));
  if (presets.length) select.appendChild(h("optgroup", { label: "Presets" }, presets.map(opt)));
  if (saved.length) select.appendChild(h("optgroup", { label: "Saved in studies/" }, saved.map(opt)));
  select.value = state.name || "";
  select.addEventListener("change", async () => {
    const name = select.value;
    if (!name || name === state.name) return;
    if (state.dirty && !(await confirmDiscard())) {
      select.value = state.name || "";
      return;
    }
    try {
      await loadStudy(name);
    } catch (err) {
      toastError(err, `Could not open study ${name}`);
      select.value = state.name || "";
    }
  });
  const isPreset = state.kind === "preset";
  mount(
    bar,
    select,
    state.dirty ? h("span", { class: "dirty-dot", title: "Unsaved changes" }, "● unsaved") : null,
    button("New", onNew, { small: true, title: "Start a new study" }),
    button("Save", onSave, {
      small: true,
      kind: state.dirty ? "primary" : null,
      disabled: !state.spec,
      title: isPreset ? "The repo preset is rebuilt from the repo's files and cannot be saved over: use Save as" : "Save to studies/<name>.study.json (Ctrl+S)",
    }),
    button("Save as…", onSaveAs, { small: true, disabled: !state.spec }),
  );
}

function drawStatus() {
  const bar = $("#status-bar");
  if (!state.spec) {
    mount(bar);
    return;
  }
  let pill;
  const v = state.validation;
  if (!v) {
    pill = h("span", { class: "pill pill-pending" }, "checking…");
  } else {
    const { errors, warnings } = issueCounts();
    const cls = errors ? "pill-error" : warnings ? "pill-warn" : "pill-ok";
    const text = errors
      ? `${errors} error${errors === 1 ? "" : "s"}${warnings ? ` · ${warnings} warning${warnings === 1 ? "" : "s"}` : ""}`
      : warnings
        ? `valid · ${warnings} warning${warnings === 1 ? "" : "s"}`
        : "valid";
    pill = h(
      "button",
      {
        type: "button",
        class: ["pill", cls, state.validating ? "stale" : null],
        title: errors ? "Errors block export. Click to see them." : "Click to see the issues",
        onclick: () => goto("study", "#issues"),
      },
      text,
    );
  }
  // "exact repo study" only when validate says the grid, catalogue and instrument equal the repo's files
  // (repo_exact); repo-shaped alone is a structural check, so a rewritten study still exports through the
  // repo's randomizer without being the repo study.
  let repoBadge = null;
  if (isRepoExact()) {
    repoBadge = badge(
      "exact repo study",
      "repo",
      "The repo's own files: the grid, catalogue and instrument equal prompts/grid.json, the persona catalogue and instruments/batteries.json, and export goes through harness.randomize (build_manifest + write_manifest), so the manifest is the one the randomize CLI writes.",
    );
    repoBadge.dataset.repo = "exact";
  } else if (isRepoShaped()) {
    repoBadge = badge("harness.randomize engine", "neutral", `Exports through the repo's randomizer (harness.randomize); ${repoExactNote()}`);
    repoBadge.dataset.repo = "shaped";
  } else if (v && state.spec && state.spec.repo) {
    repoBadge = badge("sandbox engine", "neutral", `Not expressible as the repo's files: ${v.repo_shape_reason || "see validation"}. Export goes through sandbox.study.`);
    repoBadge.dataset.repo = "sandbox";
  }
  mount(bar, repoBadge, pill);
}

function drawBanners() {
  const host = $("#banners");
  const roles = state.spec ? mockRoles() : [];
  const parts = [];
  if (roles.length) {
    parts.push(
      h(
        "div",
        { class: "banner banner-mock", role: "status" },
        h("strong", null, "MOCK — synthetic text, never for data. "),
        `This study's ${roles.join(", ")} point${roles.length === 1 ? "s" : ""} at the demo mock servers: GGUFs with no weights and deterministic filler replies. Use it to try the pipeline end to end, never for results.`,
        isRepoDataDir(state.spec.run && state.spec.run.data_dir) ? h("strong", null, ` Its data_dir is “${(state.spec.run && state.spec.run.data_dir) || "data"}”: runs would land beside real data.`) : null,
        h("a", { href: "#/models", class: "link" }, " Models →"),
      ),
    );
  }
  if (draftOffer) parts.push(draftOffer);
  mount(host, parts);
}

// ---- study actions -------------------------------------------------------------------------------------------------

function confirmDiscard() {
  const what = state.spec && state.spec.name ? `“${state.spec.name}”` : "This study";
  return confirmDialog("Discard unsaved changes?", `${what} has unsaved changes. Discard them?`, "Discard", "danger");
}

function slugify(s) {
  const out = String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  return out && NAME_RE.test(out) && out !== "repo" && out !== "blank" ? out : "my-study";
}

function reportSaved(res, name) {
  const errs = (res.issues || []).filter((i) => i.level === "error").length;
  const where = res.path || `studies/${name}.study.json`;
  if (errs) toast(`Saved ${where} as a draft with ${errs} error${errs === 1 ? "" : "s"}: export stays blocked until they are fixed.`, { kind: "warn", timeout: 7000 });
  else toast(`Saved ${where}`, { kind: "ok" });
}

async function onSave() {
  if (!state.spec) return;
  if (!state.name || state.kind === "preset") return onSaveAs();
  try {
    reportSaved(await saveStudy(state.name), state.name);
  } catch (err) {
    toastError(err, "Save failed");
  }
}

async function onSaveAs() {
  if (!state.spec) return;
  const suggestion = state.kind === "saved" && state.name ? state.name : slugify(state.spec.name);
  const name = await promptDialog({
    title: "Save study as",
    label: "File name",
    value: suggestion,
    help: "Saved as studies/<name>.study.json. Lowercase letters, digits, - and _; starts with a letter or digit.",
    okLabel: "Save",
    validate: (v) => {
      if (!v) return "A name is required.";
      if (!NAME_RE.test(v)) return "Use lowercase letters, digits, - and _, starting with a letter or digit.";
      const ex = state.studies.find((s) => s.name === v);
      if (v === "repo" || v === "blank" || (ex && ex.kind === "preset")) return `“${v}” is the name of a preset and cannot be saved over; pick another name.`;
      return null;
    },
  });
  if (!name) return;
  const existing = state.studies.find((s) => s.name === name);
  if (existing && name !== state.name) {
    const ok = await confirmDialog("Overwrite?", `studies/${name}.study.json already exists. Overwrite it with this study?`, "Overwrite", "danger");
    if (!ok) return;
  }
  try {
    reportSaved(await saveStudy(name), name);
  } catch (err) {
    toastError(err, "Save failed");
  }
}

async function onNew() {
  if (state.dirty && !(await confirmDiscard())) return;
  let pick = null;
  const choice = (value, title, text) =>
    h("button", { type: "button", class: "choice", onclick: () => pick && pick(value) }, h("strong", null, title), h("span", { class: "muted" }, text));
  const body = h(
    "div",
    { class: "choices" },
    h("p", { class: "muted" }, "Start from:"),
    state.spec ? choice("copy", "A copy of the current study", "Everything as it is now; save it under a new name.") : null,
    choice("repo", "The repo study", "prompts/grid.json, the persona catalogue, instruments/batteries.json and the config, exactly as on disk."),
    choice("minimal", "A small generic study", "A topic and one other axis; no nesting, tables or control. Keeps this study's models (seeker, mentor, judge, sampling)."),
  );
  const value = await modal({
    title: "New study",
    body,
    actions: [{ label: "Cancel", value: undefined }],
    onOpen: (_dlg, finish) => (pick = finish),
  });
  if (!value) return;
  try {
    if (value === "copy") {
      const spec = clone(state.spec);
      spec.name = `Copy of ${spec.name || "study"}`;
      setStudy(spec, { dirty: true });
    } else if (value === "repo") {
      const res = await api.study("repo");
      setStudy(res.spec, { dirty: true });
    } else {
      setStudy(await blankStudy(), { dirty: true });
    }
    goto("study");
  } catch (err) {
    toastError(err, "Could not start a new study");
  }
}

// The server's blank study when it offers one (GET /api/studies/blank), else a minimal one built here; either
// way it keeps the current study's run block, so the models the scientist set up carry over.
async function blankStudy() {
  let spec = null;
  if (state.studies.some((s) => s.name === "blank")) {
    try {
      spec = (await api.study("blank")).spec;
    } catch {
      spec = null;
    }
  }
  if (!spec || typeof spec !== "object") return minimalStudy(state.spec);
  if (state.spec && state.spec.run) spec.run = clone(state.spec.run);
  return spec;
}

// ---- unsaved draft from a previous session --------------------------------------------------------------------------

let draftOffer = null;

function offerDraft() {
  const d = readDraft();
  if (!d || !d.spec) return;
  const label = d.kind === "preset" ? `the ${d.name} preset` : d.name ? `studies/${d.name}` : `“${d.spec.name || "new study"}”`;
  draftOffer = h(
    "div",
    { class: "banner banner-draft", role: "status" },
    h("span", null, `Unsaved changes to ${label} from ${fmtTime(new Date(d.at || Date.now()).toISOString())} were left in this browser. `),
    button("Restore", () => {
      draftOffer = null;
      setStudy(d.spec, { name: d.name || null, kind: d.kind || null, path: d.path || null, dirty: true });
      drawBanners();
    }, { small: true, kind: "primary" }),
    button("Discard", () => {
      draftOffer = null;
      clearDraft();
      drawBanners();
    }, { small: true }),
  );
}

// ---- boot ----------------------------------------------------------------------------------------------------------

async function boot() {
  window.addEventListener("hashchange", () => renderRoute());
  window.addEventListener("beforeunload", (e) => {
    if (state.dirty) {
      e.preventDefault();
      e.returnValue = "";
    }
  });
  window.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
      e.preventDefault();
      onSave();
    }
  });
  on("loaded", () => {
    drawStudyBar();
    drawStatus();
    drawBanners();
    const v = VIEWS.find((x) => x.id === current.view);
    if (!current.root || !v || v.needsStudy) renderRoute();
    else drawNav();
  });
  on("dirty", drawStudyBar);
  on("studies", drawStudyBar);
  on("validating", drawStatus);
  on("validated", () => {
    drawStatus();
    drawNav();
    drawBanners();
    markIssues(current.root);
  });
  on("mock", drawBanners);
  on("changed", drawBanners);

  offerDraft();
  try {
    state.meta = await api.meta();
  } catch (err) {
    toastError(err, "Could not read /api/meta");
  }
  try {
    await refreshStudies();
  } catch (err) {
    toastError(err, "Could not list studies");
  }
  refreshMock();
  const last = storageGet("lastStudy", "repo");
  const name = state.studies.some((s) => s.name === last) ? last : state.studies.length ? state.studies[0].name : null;
  if (name) {
    try {
      await loadStudy(name); // emits "loaded", which draws everything
      return;
    } catch (err) {
      toastError(err, `Could not open study ${name}`);
    }
  }
  drawStudyBar();
  drawStatus();
  drawBanners();
  renderRoute();
}

boot();
