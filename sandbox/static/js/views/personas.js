// Personas view: the persona and reminder templates (with slot chips that insert at the cursor), derived
// slots, the no-persona control, and a live preview of any cell and variant rendered by the server.

import { api } from "../api.js";
import { h, mount, button, iconButton, badge, debounce, details, toast, toastError, promptDialog, fmtInt, kv } from "../ui.js";
import { state, touch, KEY_RE, PERSONA_MODES, clone, pathString } from "../state.js";
import { control, field, renameKey, reorderKeys, uniqueName } from "../forms.js";
import { viewHeader, card, issuePanel, slotChips } from "./common.js";

let stashedControl = null;
const preview = { cellKey: null, variant: null };

function defaultControl(spec) {
  const fs = (spec.factors || []).map((f) => f.key);
  const factor = (spec.nested && spec.nested.within) || fs[1] || fs[0] || "";
  return {
    factor,
    level: "none",
    by: fs.filter((k) => k !== factor && k === "topic"),
    persona: spec.derived && "opening" in spec.derived ? "{opening}" : "",
    reminder: "",
    persona_mode: "reinforced",
    n_per_cell: null,
  };
}

function templatesCard(ctx) {
  let target = null;
  const persona = control("templates.persona", { type: "textarea", rows: 6, ariaLabel: "Persona template" });
  const reminder = control("templates.reminder", { type: "textarea", rows: 3, ariaLabel: "Reminder template" });
  for (const ta of [persona, reminder]) ta.addEventListener("focus", () => (target = ta));
  return card(
    "Seeker templates",
    { path: "templates", help: "Filled with Python str.format_map per row; an unfilled {slot} is an error, never left in a prompt; the result is stripped." },
    slotChips(ctx, "treated", () => target || persona),
    h("label", { class: "field" }, h("span", { class: "field-label" }, "Persona (the seeker's system prompt)"), persona, h("span", { class: "field-help" }, "It carries the instruction to open the conversation by asking for guidance.")),
    h("label", { class: "field" }, h("span", { class: "field-label" }, "Reminder (a note to self, appended after the history on every seeker turn in reinforced mode)"), reminder),
  );
}

function derivedCard(ctx) {
  const spec = state.spec;
  const derived = spec.derived && typeof spec.derived === "object" && !Array.isArray(spec.derived) ? spec.derived : null;
  const entries = derived ? Object.entries(derived) : [];
  let target = null;
  const rows = entries.map(([name], i) => {
    const nameInput = h("input", { type: "text", class: "input mono", value: name, spellcheck: "false", "aria-label": "Derived slot name" });
    nameInput.dataset.path = pathString(["derived", name]);
    nameInput.addEventListener("change", () => {
      const nu = nameInput.value.trim();
      if (!nu || nu === name) {
        nameInput.value = name;
        return;
      }
      if (nu in derived) {
        toast(`A derived slot {${nu}} already exists.`, { kind: "error", timeout: 6000 });
        nameInput.value = name;
        return;
      }
      renameKey(derived, name, nu);
      touch();
      ctx.rerender();
    });
    const ta = control(["derived", name], { type: "textarea", rows: 2, ariaLabel: `Template for {${name}}` });
    ta.addEventListener("focus", () => (target = ta));
    const keys = Object.keys(derived);
    return h(
      "div",
      { class: "derived-row", dataset: { path: pathString(["derived", name]) } },
      h("div", { class: "derived-name" }, h("span", { class: "muted mono" }, "{"), nameInput, h("span", { class: "muted mono" }, "} =")),
      ta,
      h(
        "div",
        { class: "row-actions" },
        iconButton("↑", "Earlier (later slots may use earlier ones)", () => {
          if (i > 0) {
            [keys[i - 1], keys[i]] = [keys[i], keys[i - 1]];
            reorderKeys(derived, keys);
            touch();
            ctx.rerender();
          }
        }, { disabled: i === 0 }),
        iconButton("↓", "Later", () => {
          if (i < keys.length - 1) {
            [keys[i + 1], keys[i]] = [keys[i], keys[i + 1]];
            reorderKeys(derived, keys);
            touch();
            ctx.rerender();
          }
        }, { disabled: i === keys.length - 1 }),
        iconButton("✕", "Remove derived slot", () => {
          delete derived[name];
          touch();
          ctx.rerender();
        }, { class: "danger" }),
      ),
    );
  });
  return card(
    "Derived slots",
    {
      path: "derived",
      help: "Slots built from other slots, in order: each is filled from the level, variant and table slots plus the derived slots above it. A control cell keeps only the derived slots it can fill.",
    },
    entries.length ? slotChips(ctx, "treated", () => target) : null,
    entries.length ? h("div", { class: "derived-list" }, rows) : h("p", { class: "muted" }, "None."),
    h(
      "div",
      { class: "card-foot" },
      button("+ Derived slot", async () => {
        const taken = Object.keys(derived || {});
        const name = await promptDialog({
          title: "New derived slot",
          label: "Slot name",
          value: uniqueName("derived", taken),
          validate: (v) => (!v ? "Required" : !KEY_RE.test(v) ? "Lowercase: a letter, then letters, digits or _." : taken.includes(v) ? "Already exists" : null),
          okLabel: "Add",
        });
        if (!name) return;
        if (!derived) spec.derived = {};
        spec.derived[name] = "";
        touch();
        ctx.rerender();
      }, { small: true }),
    ),
  );
}

function controlCard(ctx) {
  const spec = state.spec;
  const c = spec.control;
  const toggle = h("input", { type: "checkbox", checked: !!c });
  toggle.addEventListener("change", () => {
    if (toggle.checked) {
      spec.control = stashedControl || defaultControl(spec);
      stashedControl = null;
    } else {
      stashedControl = clone(spec.control);
      spec.control = null;
    }
    touch();
    ctx.rerender();
  });
  const head = h(
    "label",
    { class: "field-check toggle-line" },
    toggle,
    h("span", null, "Include a no-persona control cell for each combination of the ", h("code", null, "by"), " levels"),
  );
  if (!c) {
    return card("Control", { path: "control", help: "Off. The control is the seeker without a persona (normally just the opening request), so the mentor's movement can be compared against no treatment." }, head);
  }
  let target = null;
  const persona = control("control.persona", { type: "textarea", rows: 3, ariaLabel: "Control persona" });
  const reminder = control("control.reminder", { type: "textarea", rows: 2, ariaLabel: "Control reminder" });
  for (const ta of [persona, reminder]) ta.addEventListener("focus", () => (target = ta));
  const fkeys = (spec.factors || []).map((f) => f.key);
  const by = Array.isArray(c.by) ? c.by : [];
  const byBoxes = h(
    "div",
    { class: "check-row", dataset: { path: "control.by" } },
    fkeys
      .filter((k) => k !== c.factor)
      .map((k) => {
        const cb = h("input", { type: "checkbox", checked: by.includes(k) });
        cb.addEventListener("change", () => {
          const set = new Set(Array.isArray(c.by) ? c.by : []);
          if (cb.checked) set.add(k);
          else set.delete(k);
          c.by = fkeys.filter((x) => set.has(x));
          touch();
        });
        return h("label", { class: "field-check" }, cb, h("span", { class: "mono" }, k));
      }),
  );
  const same = c.n_per_cell === null || c.n_per_cell === undefined;
  const nInput = control("control.n_per_cell", { type: "int", min: 1, ariaLabel: "Control n per cell", disabled: same });
  const sameBox = h("input", { type: "checkbox", checked: same });
  sameBox.addEventListener("change", () => {
    c.n_per_cell = sameBox.checked ? null : Number((spec.randomization && spec.randomization.n_per_cell) || 1);
    touch();
    ctx.rerender();
  });
  return card(
    "Control",
    { path: "control", help: "One cell per combination of the by levels. Its condition has the by levels, the control level for the control factor, and null for everything else." },
    head,
    h(
      "div",
      { class: "form-row" },
      field("control factor", "control.factor", {
        type: "select",
        options: fkeys,
        onChange: (v) => {
          if (Array.isArray(c.by)) c.by = c.by.filter((k) => k !== v);
          ctx.rerender();
        },
      }),
      field("level label", "control.level", { mono: true, help: "Must not be a level of the control factor" }),
      field("persona mode", "control.persona_mode", { type: "select", options: PERSONA_MODES }),
      h(
        "div",
        { class: "field" },
        h("span", { class: "field-label" }, "n per cell"),
        h("div", { class: "inline-group" }, nInput, h("label", { class: "field-check" }, sameBox, h("span", null, "same as treated"))),
      ),
    ),
    h("div", { class: "field" }, h("span", { class: "field-label" }, "by"), byBoxes),
    slotChips(ctx, "control", () => target || persona),
    h("label", { class: "field" }, h("span", { class: "field-label" }, "Control persona"), persona),
    h("label", { class: "field" }, h("span", { class: "field-label" }, "Control reminder"), reminder),
  );
}

function previewCard(ctx) {
  const cellSel = h("select", { class: "input", "aria-label": "Cell to preview" });
  const varSel = h("select", { class: "input", "aria-label": "Variant to preview" });
  const varField = h("label", { class: "field" }, h("span", { class: "field-label" }, "Variant"), varSel);
  const out = h("div", { class: "preview-out", "aria-live": "polite" });
  let seq = 0;
  let rendered = false;

  const currentCell = () => (state.cells || []).find((c) => c.key === preview.cellKey) || null;

  const fillSelectors = () => {
    const cells = state.cells || [];
    if (!cells.length) {
      mount(cellSel, h("option", { value: "" }, state.cellsError ? "cells unavailable" : "no cells"));
      cellSel.disabled = true;
      varField.hidden = true;
      return;
    }
    cellSel.disabled = false;
    if (!cells.some((c) => c.key === preview.cellKey)) preview.cellKey = cells[0].key;
    const treated = cells.filter((c) => c.kind !== "control");
    const ctrl = cells.filter((c) => c.kind === "control");
    mount(
      cellSel,
      treated.length ? h("optgroup", { label: "Treated" }, treated.map((c) => h("option", { value: c.key }, c.key))) : null,
      ctrl.length ? h("optgroup", { label: "Control" }, ctrl.map((c) => h("option", { value: c.key }, c.key))) : null,
    );
    cellSel.value = preview.cellKey;
    const cell = currentCell();
    const variants = (cell && cell.variants) || [];
    if (!variants.length) {
      varField.hidden = true;
      preview.variant = null;
    } else {
      varField.hidden = false;
      if (!variants.includes(preview.variant)) preview.variant = variants[0];
      mount(varSel, variants.map((v) => h("option", { value: v }, v)));
      varSel.value = preview.variant;
    }
  };

  const draw = async () => {
    const cell = currentCell();
    if (!cell) {
      mount(out, h("p", { class: "muted" }, state.cells === null ? "Waiting for the cells…" : "Nothing to preview."));
      return;
    }
    const my = ++seq;
    out.classList.add("stale");
    try {
      const kind = cell.kind === "control" ? "control" : "treated";
      const res = await api.render(state.spec, kind, cell.condition || {}, kind === "control" ? null : preview.variant);
      if (my !== seq || !ctx.alive()) return;
      const text = res.persona_text ?? "";
      const rem = res.persona_reminder ?? "";
      const slots = res.slots && typeof res.slots === "object" ? res.slots : null;
      rendered = true;
      mount(
        out,
        h("div", { class: "preview-label" }, h("span", null, "Persona"), h("span", { class: "muted small" }, `${fmtInt(text.length)} chars ≈ ${fmtInt(Math.ceil(text.length / 4))} tokens`)),
        h("div", { class: "persona-text", dataset: { testid: "persona-text" } }, text || h("span", { class: "muted" }, "(empty)")),
        h("div", { class: "preview-label" }, h("span", null, "Reminder"), h("span", { class: "muted small" }, rem ? `${fmtInt(rem.length)} chars` : "")),
        h("div", { class: "persona-text reminder" }, rem || h("span", { class: "muted" }, "(empty: only allowed where the mode is once)")),
        slots ? details(`Slot values (${Object.keys(slots).length})`, kv(Object.entries(slots).map(([k, v]) => [`{${k}}`, h("span", { class: "slot-val" }, typeof v === "string" ? v : JSON.stringify(v))]), "kv-slots")) : null,
      );
    } catch (err) {
      if (my !== seq || !ctx.alive()) return;
      rendered = true;
      if (err && err.status === 0) toastError(err, "Preview");
      mount(
        out,
        h(
          "div",
          { class: "error-box", role: "alert" },
          h("strong", null, "This cell does not render. "),
          h("span", null, err && err.message ? err.message : String(err)),
          err && err.issues && err.issues.length ? h("ul", null, err.issues.slice(0, 6).map((i) => h("li", null, `${i.path ? i.path + ": " : ""}${i.message || i}`))) : null,
        ),
      );
    } finally {
      if (my === seq) out.classList.remove("stale");
    }
  };
  const redraw = debounce(draw, 400);

  cellSel.addEventListener("change", () => {
    preview.cellKey = cellSel.value;
    preview.variant = null;
    fillSelectors();
    draw();
  });
  varSel.addEventListener("change", () => {
    preview.variant = varSel.value;
    draw();
  });
  ctx.on("changed", redraw);
  ctx.on("validated", () => {
    const before = `${preview.cellKey}|${preview.variant}`;
    fillSelectors();
    if (`${preview.cellKey}|${preview.variant}` !== before || !rendered) redraw();
  });
  ctx.cleanup(() => redraw.cancel());
  fillSelectors();
  draw();

  return h(
    "section",
    { class: "card preview-card" },
    h("div", { class: "card-head" }, h("h2", null, "Preview"), badge("live", "info", "Re-rendered by the server as you type")),
    h("div", { class: "form-row" }, h("label", { class: "field grow" }, h("span", { class: "field-label" }, "Cell"), cellSel), varField),
    out,
  );
}

export function render(root, ctx) {
  mount(
    root,
    viewHeader("Personas", "What the seeker is told. The mentor gets nothing beyond its chat template."),
    issuePanel(ctx, ["templates", "derived", "control"]),
    h(
      "div",
      { class: "split" },
      h("div", { class: "split-main" }, templatesCard(ctx), derivedCard(ctx), controlCard(ctx)),
      h("div", { class: "split-side" }, previewCard(ctx)),
    ),
  );
}
