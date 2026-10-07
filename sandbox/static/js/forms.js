// Inputs bound to a path in the study spec. Typing mutates state.spec and calls touch() (dirty + debounced
// validation); the element carries data-path so validation issues can be pinned to it.

import { h, autosize } from "./ui.js";
import { state, getPath, setPath, touch, parsePath, pathString } from "./state.js";

function readValue(path, root) {
  return getPath(root || state.spec, path);
}

function writeValue(path, value, root, opts) {
  setPath(root || state.spec, path, value);
  if (opts.onChange) opts.onChange(value);
  if (!opts.noTouch) touch();
}

function parseInput(raw, opts) {
  const t = opts.type || "text";
  if (t === "int" || t === "float") {
    const s = String(raw).trim();
    if (s === "") return opts.emptyDelete ? undefined : opts.nullable === false ? 0 : null;
    const n = Number(s);
    return Number.isFinite(n) ? n : s;
  }
  const s = String(raw);
  if (s.trim() === "") {
    if (opts.emptyDelete) return undefined;
    if (opts.nullable) return null;
  }
  return s;
}

// The bare control for `path` (an input, textarea, select or checkbox).
export function control(path, opts = {}) {
  const parts = parsePath(path);
  const dp = pathString(parts);
  const t = opts.type || "text";
  const cur = readValue(parts, opts.root);
  let el;
  if (t === "checkbox") {
    el = h("input", { type: "checkbox", checked: !!cur, disabled: opts.disabled });
    el.addEventListener("change", () => writeValue(parts, el.checked, opts.root, opts));
  } else if (t === "select") {
    const options = (opts.options || []).map((o) => (typeof o === "object" ? o : { value: o, label: String(o) }));
    const values = options.map((o) => o.value);
    const sel = h(
      "select",
      { class: ["input", opts.class], disabled: opts.disabled },
      options.map((o) => h("option", { value: String(o.value) }, o.label)),
    );
    if (cur !== undefined && cur !== null && !values.includes(cur)) {
      sel.appendChild(h("option", { value: String(cur) }, `${cur} (unknown)`));
    }
    if ((cur === null || cur === undefined) && opts.placeholder !== undefined) {
      sel.insertBefore(h("option", { value: "" }, opts.placeholder), sel.firstChild);
    }
    sel.value = cur === null || cur === undefined ? "" : String(cur);
    sel.addEventListener("change", () => {
      const raw = sel.value;
      const match = options.find((o) => String(o.value) === raw);
      const v = raw === "" ? (opts.nullable ? null : undefined) : match ? match.value : raw;
      writeValue(parts, v, opts.root, opts);
    });
    el = sel;
  } else if (t === "textarea") {
    el = h("textarea", {
      class: ["input", "textarea", opts.mono ? "mono" : null, opts.class],
      rows: opts.rows || 3,
      placeholder: opts.placeholder,
      spellcheck: opts.spellcheck === false ? "false" : "true",
      disabled: opts.disabled,
    });
    el.value = cur === null || cur === undefined ? "" : String(cur);
    if (opts.autosize !== false) autosize(el);
    el.addEventListener(opts.commit || "input", () => writeValue(parts, parseInput(el.value, opts), opts.root, opts));
  } else {
    const isNum = t === "int" || t === "float";
    el = h("input", {
      type: isNum ? "number" : "text",
      class: ["input", opts.mono ? "mono" : null, isNum ? "input-num" : null, opts.class],
      placeholder: opts.placeholder,
      step: t === "int" ? "1" : t === "float" ? opts.step || "any" : undefined,
      min: opts.min,
      max: opts.max,
      spellcheck: "false",
      disabled: opts.disabled,
      "aria-label": opts.ariaLabel,
    });
    el.value = cur === null || cur === undefined ? "" : String(cur);
    el.addEventListener(opts.commit || "input", () => writeValue(parts, parseInput(el.value, opts), opts.root, opts));
  }
  el.dataset.path = dp;
  if (opts.ariaLabel) el.setAttribute("aria-label", opts.ariaLabel);
  if (opts.title) el.title = opts.title;
  return el;
}

// A labelled field: label, control, optional help line.
export function field(label, path, opts = {}) {
  const c = control(path, opts);
  if (opts.type === "checkbox") {
    return h(
      "label",
      { class: ["field", "field-check", opts.fieldClass] },
      c,
      h("span", { class: "field-label" }, label),
      opts.help ? h("span", { class: "field-help" }, opts.help) : null,
    );
  }
  return h(
    "label",
    { class: ["field", opts.fieldClass] },
    h("span", { class: "field-label" }, label, opts.optional ? h("span", { class: "optional" }, " optional") : null),
    c,
    opts.help ? h("span", { class: "field-help" }, opts.help) : null,
  );
}

// Swap two array entries at `path` (i and j), then touch.
export function moveItem(arr, i, j) {
  if (j < 0 || j >= arr.length) return false;
  const [x] = arr.splice(i, 1);
  arr.splice(j, 0, x);
  touch();
  return true;
}

// Rebuild an object with its keys in a new order / a key renamed, keeping insertion order.
export function renameKey(obj, from, to) {
  if (!obj || from === to) return obj;
  const out = {};
  for (const [k, v] of Object.entries(obj)) out[k === from ? to : k] = v;
  for (const k of Object.keys(obj)) delete obj[k];
  Object.assign(obj, out);
  return obj;
}

export function reorderKeys(obj, keys) {
  const out = {};
  for (const k of keys) out[k] = obj[k];
  for (const k of Object.keys(obj)) delete obj[k];
  Object.assign(obj, out);
  return obj;
}

export function uniqueName(base, taken) {
  const set = new Set(taken);
  if (!set.has(base)) return base;
  for (let i = 2; ; i++) if (!set.has(`${base}_${i}`)) return `${base}_${i}`;
}
