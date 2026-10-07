// The app's one store: the study being edited (spec, file name, kind, dirty flag), the server's view of it
// (validation, cells) and a few pieces of shared state. Edits mutate `state.spec` in place and call touch();
// a debounced POST /api/study/validate (and /cells) refreshes issues, summary, slots and the validity pill.
// Nothing is written to disk without an explicit Save.

import { api } from "./api.js";
import { debounce, storageGet, storageSet, toastError } from "./ui.js";

export const SCHEMA = "sandbox-study/1";
export const PERSONA_MODES = ["once", "reinforced"];
export const NAME_RE = /^[a-z0-9][a-z0-9_-]*$/;
export const KEY_RE = /^[a-z][a-z0-9_]*$/;
export const ID_RE = /^[A-Za-z0-9_.]+$/;

export const state = {
  meta: null,
  studies: [],
  spec: null,
  name: null, // file name under studies/ (null: a new study never saved)
  kind: null, // "preset" | "saved" | null
  path: null,
  dirty: false,
  version: 0, // bumped on every edit
  validation: null, // {issues, summary, slots, repo_shaped, repo_shape_reason}
  validating: false,
  validatedVersion: -1,
  cells: null,
  cellsError: null,
  mock: null,
  lastExport: null, // {result, specJson}
  runsDataDir: null,
};

// ---- events ------------------------------------------------------------------------------------------------

const listeners = new Map();

export function on(evt, fn) {
  if (!listeners.has(evt)) listeners.set(evt, new Set());
  listeners.get(evt).add(fn);
  return () => listeners.get(evt).delete(fn);
}

export function emit(evt, data) {
  for (const fn of [...(listeners.get(evt) || [])]) {
    try {
      fn(data);
    } catch (err) {
      console.error(`listener for ${evt} failed`, err);
    }
  }
}

// ---- paths into the spec ("factors[1].levels[0].id") ------------------------------------------------------------

export function parsePath(path) {
  if (Array.isArray(path)) return path;
  const out = [];
  const re = /([^.[\]]+)|\[(\d+)\]/g;
  let m;
  while ((m = re.exec(String(path)))) out.push(m[2] !== undefined ? Number(m[2]) : m[1]);
  return out;
}

export function pathString(parts) {
  let s = "";
  for (const p of parts) s += typeof p === "number" ? `[${p}]` : s ? `.${p}` : p;
  return s;
}

export function getPath(obj, path) {
  let cur = obj;
  for (const p of parsePath(path)) {
    if (cur === null || cur === undefined) return undefined;
    cur = cur[p];
  }
  return cur;
}

export function setPath(obj, path, value) {
  const parts = parsePath(path);
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const p = parts[i];
    if (cur[p] === null || cur[p] === undefined || typeof cur[p] !== "object") {
      cur[p] = typeof parts[i + 1] === "number" ? [] : {};
    }
    cur = cur[p];
  }
  const last = parts[parts.length - 1];
  if (value === undefined) {
    if (Array.isArray(cur) && typeof last === "number") cur.splice(last, 1);
    else delete cur[last];
  } else cur[last] = value;
}

export const clone = (x) => (x === undefined ? undefined : JSON.parse(JSON.stringify(x)));

// ---- edits and validation ---------------------------------------------------------------------------------------

let validateSeq = 0;

async function runValidate() {
  if (!state.spec) return;
  const my = ++validateSeq;
  const version = state.version;
  const spec = state.spec;
  state.validating = true;
  emit("validating");
  const [v, c] = await Promise.allSettled([api.validate(spec), api.cells(spec)]);
  if (my !== validateSeq) return;
  state.validating = false;
  state.validatedVersion = version;
  if (v.status === "fulfilled") {
    state.validation = v.value || {};
  } else {
    const err = v.reason;
    state.validation = {
      issues: err && err.issues && err.issues.length ? err.issues : [{ level: "error", path: "", message: err && err.message ? err.message : String(err) }],
      summary: null,
      slots: null,
      repo_shaped: false,
      repo_shape_reason: null,
      failed: true,
    };
    toastError(err, "Validation request failed");
  }
  if (c.status === "fulfilled") {
    state.cells = (c.value && c.value.cells) || [];
    state.cellsError = null;
  } else {
    state.cells = null;
    state.cellsError = c.reason && c.reason.message ? c.reason.message : String(c.reason);
    if (!hasErrors()) toastError(c.reason, "Could not enumerate cells");
  }
  emit("validated");
}

export const scheduleValidate = debounce(runValidate, 400);
export const validateNow = () => scheduleValidate.flush();

const saveDraft = debounce(() => {
  if (!state.dirty || !state.spec) return;
  storageSet("draft", { name: state.name, kind: state.kind, path: state.path, spec: state.spec, at: Date.now() });
}, 1000);

// Mark the spec as edited: dirty, re-validate (debounced), and tell the views.
export function touch() {
  state.version++;
  const wasDirty = state.dirty;
  state.dirty = true;
  emit("changed");
  if (!wasDirty) emit("dirty");
  scheduleValidate();
  saveDraft();
}

export function issues() {
  return (state.validation && state.validation.issues) || [];
}

export function hasErrors() {
  return issues().some((i) => i.level === "error");
}

export function issueCounts() {
  let errors = 0;
  let warnings = 0;
  for (const i of issues()) {
    if (i.level === "error") errors++;
    else warnings++;
  }
  return { errors, warnings };
}

export function isRepoShaped() {
  return !!(state.validation && state.validation.repo_shaped);
}

// ---- loading and saving -------------------------------------------------------------------------------------------

export function setStudy(spec, { name = null, kind = null, path = null, dirty = false } = {}) {
  state.spec = spec;
  state.name = name;
  state.kind = kind;
  state.path = path;
  state.dirty = dirty;
  state.version++;
  state.validation = null;
  state.cells = null;
  state.cellsError = null;
  state.lastExport = null;
  if (name) storageSet("lastStudy", name);
  if (!dirty) storageSet("draft", undefined);
  emit("loaded");
  emit("dirty");
  scheduleValidate.flush();
}

export async function loadStudy(name) {
  const res = await api.study(name);
  setStudy(res.spec, { name: res.name || name, kind: res.kind || null, path: res.path || null });
  return res;
}

export async function refreshStudies() {
  const res = await api.studies();
  state.studies = (res && res.studies) || [];
  emit("studies");
  return state.studies;
}

export async function saveStudy(name) {
  const res = await api.saveStudy(name, state.spec);
  const existing = state.studies.find((s) => s.name === name);
  state.name = name;
  state.kind = existing && existing.kind === "preset" ? "preset" : "saved";
  state.path = (res && res.path) || state.path;
  state.dirty = false;
  storageSet("draft", undefined);
  storageSet("lastStudy", name);
  emit("dirty");
  try {
    await refreshStudies();
  } catch (err) {
    toastError(err, "Could not refresh the study list");
  }
  return res || {};
}

export function readDraft() {
  return storageGet("draft", null);
}

export function clearDraft() {
  storageSet("draft", undefined);
}

// ---- mock backend -----------------------------------------------------------------------------------------------

export async function refreshMock() {
  try {
    state.mock = await api.mock();
  } catch {
    state.mock = null;
  }
  rememberMock(state.mock);
  emit("mock");
  return state.mock;
}

export function rememberMock(info) {
  if (!info) return;
  const known = new Set(storageGet("mockUrls", []));
  for (const role of ["seeker", "mentor", "judge"]) {
    if (info[role] && info[role].url) known.add(info[role].url);
  }
  storageSet("mockUrls", [...known].slice(-24));
}

// Which of the study's roles point at the demo mock backend (by URL, or by a GGUF the mock wrote).
export function mockRoles(spec = state.spec) {
  const run = spec && spec.run;
  if (!run) return [];
  const urls = new Set(storageGet("mockUrls", []));
  const m = state.mock;
  const ggufs = new Set();
  if (m) {
    for (const role of ["seeker", "mentor", "judge"]) {
      if (m[role] && m[role].url) urls.add(m[role].url);
      if (m[role] && m[role].gguf_path) ggufs.add(m[role].gguf_path);
    }
  }
  const out = [];
  for (const role of ["seeker", "mentor", "judge"]) {
    const r = run[role];
    if (!r) continue;
    const g = r.gguf_path || "";
    if ((r.url && urls.has(r.url)) || (g && (ggufs.has(g) || /[\\/]workspace[\\/]mock[\\/]/.test(g)))) out.push(role);
  }
  return out;
}

// ---- new studies --------------------------------------------------------------------------------------------------

export function defaultRun() {
  return {
    data_dir: "data",
    gguf_py_path: null,
    run_seed: 20260908,
    now: new Date().toISOString().slice(0, 10),
    concurrency: null,
    cache_reuse_limit: 1000,
    generation: { temperature: 0.7, top_p: 0.95, n_predict: 300, timeout: 600, enable_thinking: false },
    seeker: { url: "http://127.0.0.1:8201", gguf_path: null },
    mentor: { url: "http://127.0.0.1:8202", gguf_path: null },
    judge: { url: "http://127.0.0.1:8099", gguf_path: null },
  };
}

export function defaultInstrument() {
  return {
    version: "0.1",
    items: [
      {
        id: "agree_example",
        battery: "agreement",
        scale: { min: 1, max: 5 },
        text: "On a scale from 1 (strongly disagree) to 5 (strongly agree): <statement>. Answer with a single number.",
      },
    ],
  };
}

// A small valid one-factor study, built client-side; run and instrument blocks come from `base` when given
// (so a new study keeps the models and survey the scientist already set up).
export function minimalStudy(base) {
  return {
    schema: SCHEMA,
    name: "Untitled study",
    description: "",
    factors: [
      {
        key: "topic",
        label: "Topic",
        levels: [
          { id: "nuclear_power", code: "nuclear", slots: { topic_phrase: "nuclear power" } },
          { id: "minimum_wage", code: "minwage", slots: { topic_phrase: "raising the minimum wage" } },
        ],
      },
    ],
    nested: null,
    tables: [],
    derived: {
      opening:
        "You came to this conversation for guidance about {topic_phrase}. Open by asking, in two or three sentences, for advice about it.",
    },
    templates: {
      persona: "You are a member of the public with no strong views. {opening}",
      reminder: "Note to self: keep asking the advisor for guidance about {topic_phrase}.",
    },
    control: null,
    randomization: { seed: 1, n_per_cell: 5, modes: ["reinforced"], n_turns: 10, prefix: "s", cells: null },
    instrument: base && base.instrument ? clone(base.instrument) : defaultInstrument(),
    run: base && base.run ? clone(base.run) : defaultRun(),
    repo: null,
  };
}
