// Axes view: the crossed factors (key, label, levels with id, code and slot columns), persona variants
// nested in one factor, and lookup tables of text keyed by a combination of levels.

import { h, mount, button, iconButton, badge, confirmDialog, promptDialog, autosize, toast } from "../ui.js";
import { state, touch, getPath, setPath, clone, KEY_RE, pathString } from "../state.js";
import { control, field, renameKey, uniqueName } from "../forms.js";
import { viewHeader, card, issuePanel, factorByKey, levelIds, combos } from "./common.js";

let stashedNested = null;

function factors() {
  const spec = state.spec;
  if (!Array.isArray(spec.factors)) spec.factors = [];
  return spec.factors;
}

function slotColumns(objs) {
  const cols = [];
  for (const o of objs) for (const k of Object.keys((o && o.slots) || {})) if (!cols.includes(k)) cols.push(k);
  return cols;
}

function slotNameValidator(taken) {
  return (v) => {
    if (!v) return "A name is required.";
    if (!KEY_RE.test(v)) return "Slot names are lowercase: a letter, then letters, digits or _.";
    if (taken.includes(v)) return `“${v}” is already a column here.`;
    return null;
  };
}

// ---- reference upkeep when keys and ids change ---------------------------------------------------------------------

function renameFactorRefs(oldKey, newKey) {
  const spec = state.spec;
  if (spec.nested && spec.nested.within === oldKey) spec.nested.within = newKey;
  for (const t of spec.tables || []) if (Array.isArray(t.by)) t.by = t.by.map((k) => (k === oldKey ? newKey : k));
  if (spec.control) {
    if (spec.control.factor === oldKey) spec.control.factor = newKey;
    if (Array.isArray(spec.control.by)) spec.control.by = spec.control.by.map((k) => (k === oldKey ? newKey : k));
  }
}

function renameAtDepth(obj, depth, from, to) {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return;
  if (depth === 0) {
    if (from in obj && !(to in obj)) renameKey(obj, from, to);
    return;
  }
  for (const v of Object.values(obj)) renameAtDepth(v, depth - 1, from, to);
}

function deleteAtDepth(obj, depth, key) {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return 0;
  if (depth === 0) {
    if (key in obj) {
      delete obj[key];
      return 1;
    }
    return 0;
  }
  let n = 0;
  for (const v of Object.values(obj)) n += deleteAtDepth(v, depth - 1, key);
  return n;
}

function renameLevelRefs(fi, factorKey, from, to) {
  const spec = state.spec;
  if (spec.nested && spec.nested.within === factorKey && spec.nested.variants && from in spec.nested.variants && !(to in spec.nested.variants)) {
    renameKey(spec.nested.variants, from, to);
  }
  for (const t of spec.tables || []) {
    const d = Array.isArray(t.by) ? t.by.indexOf(factorKey) : -1;
    if (d >= 0) renameAtDepth(t.values, d, from, to);
  }
  const r = spec.randomization;
  if (r && Array.isArray(r.cells)) {
    const n = factors().length;
    r.cells = r.cells.map((key) => {
      const parts = String(key).split("/");
      if (parts.length === n && parts[fi] === from) parts[fi] = to;
      return parts.join("/");
    });
  }
}

// ---- factors ---------------------------------------------------------------------------------------------------------

function factorCard(ctx, f, fi) {
  const fs = factors();
  const base = ["factors", fi];
  const levels = Array.isArray(f.levels) ? f.levels : (f.levels = []);
  const cols = slotColumns(levels);

  const keyInput = h("input", { type: "text", class: "input mono key-input", value: f.key || "", spellcheck: "false", "aria-label": "Factor key", placeholder: "key" });
  keyInput.dataset.path = pathString([...base, "key"]);
  keyInput.addEventListener("change", () => {
    const nu = keyInput.value.trim();
    const old = f.key;
    f.key = nu;
    if (old && nu && old !== nu) renameFactorRefs(old, nu);
    touch();
    ctx.rerender();
  });

  const addSlotCol = async () => {
    const name = await promptDialog({
      title: `Add a slot column to ${f.key || "factor"}`,
      label: "Slot name",
      help: "Each level gets its own value; templates use it as {name}. Lowercase letters, digits and _.",
      validate: slotNameValidator(cols),
      okLabel: "Add",
    });
    if (!name) return;
    for (const l of levels) {
      l.slots = l.slots || {};
      if (!(name in l.slots)) l.slots[name] = "";
    }
    touch();
    ctx.rerender();
  };
  const renameSlotCol = async (col) => {
    const name = await promptDialog({ title: `Rename slot {${col}}`, label: "New name", value: col, validate: slotNameValidator(cols.filter((c) => c !== col)), okLabel: "Rename" });
    if (!name || name === col) return;
    for (const l of levels) if (l.slots && col in l.slots) renameKey(l.slots, col, name);
    touch();
    toast(`Renamed {${col}} to {${name}} in ${f.key}'s levels. Templates that use {${col}} still need editing.`, { kind: "info", timeout: 6000 });
    ctx.rerender();
  };
  const removeSlotCol = async (col) => {
    if (!(await confirmDialog(`Remove slot {${col}}?`, `This deletes the {${col}} value from every level of ${f.key}.`, "Remove", "danger"))) return;
    for (const l of levels) if (l.slots) delete l.slots[col];
    touch();
    ctx.rerender();
  };

  const levelRow = (l, li) => {
    const lp = [...base, "levels", li];
    const idInput = h("input", { type: "text", class: "input mono id-input", value: l && l.id !== undefined ? String(l.id) : "", spellcheck: "false", "aria-label": "Level id", placeholder: "level_id" });
    idInput.dataset.path = pathString([...lp, "id"]);
    idInput.addEventListener("change", () => {
      const nu = idInput.value.trim();
      const old = l.id;
      l.id = nu;
      if (old && nu && old !== nu) renameLevelRefs(fi, f.key, old, nu);
      touch();
      ctx.rerender();
    });
    const tr = h(
      "tr",
      { dataset: { path: pathString(lp) } },
      h("td", { class: "num muted" }, String(li + 1)),
      h("td", null, idInput),
      h("td", null, control([...lp, "code"], { mono: true, emptyDelete: true, placeholder: "(id)", ariaLabel: "Code", class: "code-input" })),
      cols.map((c) => {
        const ta = control([...lp, "slots", c], { type: "textarea", rows: 1, ariaLabel: `${c} for ${l.id}` });
        return h("td", { class: "slot-cell" }, ta);
      }),
      h(
        "td",
        { class: "row-actions" },
        iconButton("↑", "Move up", () => {
          if (li > 0) {
            levels.splice(li - 1, 0, levels.splice(li, 1)[0]);
            touch();
            ctx.rerender();
          }
        }, { disabled: li === 0 }),
        iconButton("↓", "Move down", () => {
          if (li < levels.length - 1) {
            levels.splice(li + 1, 0, levels.splice(li, 1)[0]);
            touch();
            ctx.rerender();
          }
        }, { disabled: li === levels.length - 1 }),
        iconButton("✕", "Remove level", async () => {
          const spec = state.spec;
          const nv = spec.nested && spec.nested.within === f.key && spec.nested.variants && Array.isArray(spec.nested.variants[l.id]) ? spec.nested.variants[l.id].length : 0;
          const tabs = (spec.tables || []).filter((t) => Array.isArray(t.by) && t.by.includes(f.key)).length;
          const extra = [nv ? `its ${nv} nested variant${nv === 1 ? "" : "s"}` : null, tabs ? `its entries in ${tabs} lookup table${tabs === 1 ? "" : "s"}` : null].filter(Boolean);
          const msg = `Remove level “${l.id}” of ${f.key}${extra.length ? `, with ${extra.join(" and ")}` : ""}?`;
          if (!(await confirmDialog("Remove level?", msg, "Remove", "danger"))) return;
          levels.splice(li, 1);
          if (nv) delete spec.nested.variants[l.id];
          for (const t of spec.tables || []) {
            const d = Array.isArray(t.by) ? t.by.indexOf(f.key) : -1;
            if (d >= 0) deleteAtDepth(t.values, d, l.id);
          }
          touch();
          ctx.rerender();
        }, { class: "danger" }),
      ),
    );
    return tr;
  };

  const table = h(
    "table",
    { class: "table edit-table" },
    h(
      "thead",
      null,
      h(
        "tr",
        null,
        h("th", { class: "num" }, "#"),
        h("th", null, "id"),
        h("th", { title: "Optional short code used in dyad_id instead of the id" }, "code"),
        cols.map((c) =>
          h(
            "th",
            { class: "slot-col" },
            h("span", { class: "mono" }, `{${c}}`),
            iconButton("✎", `Rename slot ${c}`, () => renameSlotCol(c), { class: "tiny" }),
            iconButton("✕", `Remove slot ${c}`, () => removeSlotCol(c), { class: "tiny danger" }),
          ),
        ),
        h("th", { class: "row-actions" }, button("+ slot column", addSlotCol, { small: true, title: "Add a slot every level of this factor fills" })),
      ),
    ),
    h("tbody", null, levels.map(levelRow)),
  );

  return h(
    "section",
    { class: "card factor-card", dataset: { path: pathString(base) } },
    h(
      "div",
      { class: "card-head factor-head" },
      h("span", { class: "factor-index", title: "Crossing order: factor 1 is outermost" }, `Factor ${fi + 1}`),
      h("label", { class: "inline-field" }, h("span", null, "key"), keyInput),
      h("label", { class: "inline-field grow" }, h("span", null, "label"), control([...base, "label"], { emptyDelete: true, placeholder: "(optional)", ariaLabel: "Factor label" })),
      h(
        "div",
        { class: "card-actions" },
        iconButton("↑", "Move factor up (outer)", () => {
          if (fi > 0) {
            fs.splice(fi - 1, 0, fs.splice(fi, 1)[0]);
            touch();
            ctx.rerender();
          }
        }, { disabled: fi === 0 }),
        iconButton("↓", "Move factor down (inner)", () => {
          if (fi < fs.length - 1) {
            fs.splice(fi + 1, 0, fs.splice(fi, 1)[0]);
            touch();
            ctx.rerender();
          }
        }, { disabled: fi === fs.length - 1 }),
        iconButton("✕", "Remove factor", async () => {
          const spec = state.spec;
          const refs = [];
          if (spec.nested && spec.nested.within === f.key) refs.push("the nested variants are within it");
          if ((spec.tables || []).some((t) => Array.isArray(t.by) && t.by.includes(f.key))) refs.push("lookup tables are keyed by it");
          if (spec.control && (spec.control.factor === f.key || (spec.control.by || []).includes(f.key))) refs.push("the control refers to it");
          const msg = h("div", null, h("p", null, `Remove factor “${f.key}” and its ${levels.length} levels?`), refs.length ? h("p", { class: "warn-text" }, `Note: ${refs.join("; ")}. Validation will point at what to fix.`) : null);
          if (!(await confirmDialog("Remove factor?", msg, "Remove", "danger"))) return;
          fs.splice(fi, 1);
          touch();
          ctx.rerender();
        }, { class: "danger" }),
      ),
    ),
    h("div", { class: "table-wrap" }, table),
    h(
      "div",
      { class: "card-foot" },
      button("+ Add level", () => {
        const ids = levelIds(f);
        const id = uniqueName(`level_${levels.length + 1}`, ids);
        const slots = {};
        for (const c of cols) slots[c] = "";
        levels.push({ id, slots });
        touch();
        ctx.rerender();
      }, { small: true }),
      h("span", { class: "muted small" }, `${levels.length} level${levels.length === 1 ? "" : "s"} · {${f.key || "key"}} is the level id; each slot column is a {slot} every level fills.`),
    ),
  );
}

// ---- nested variants -------------------------------------------------------------------------------------------------

function nestedSection(ctx) {
  const spec = state.spec;
  const n = spec.nested;
  if (!n) {
    return card(
      "Nested persona variants",
      { path: "nested", help: "Off: each treated cell has one implicit persona. Turn on to register several personas (e.g. social roles) per level of one factor; a cell's rows are split evenly across them." },
      button("Turn on nested variants", () => {
        if (stashedNested) {
          spec.nested = stashedNested;
          stashedNested = null;
        } else {
          const within = (factors()[1] || factors()[0] || {}).key || "";
          const f = factorByKey(within);
          const variants = {};
          for (const id of levelIds(f)) variants[id] = [{ id: `${id}_a`, slots: {} }];
          spec.nested = { key: "variant", label: "Variant", within, max_per_level: 1, variants };
        }
        touch();
        ctx.rerender();
      }, { kind: "primary", small: true }),
    );
  }
  n.variants = n.variants && typeof n.variants === "object" ? n.variants : {};
  const parent = factorByKey(n.within);
  const parentIds = levelIds(parent);
  const all = Object.values(n.variants).flatMap((v) => (Array.isArray(v) ? v : []));
  const cols = slotColumns(all);
  const maxPer = Number(n.max_per_level) || 0;

  const addCol = async () => {
    const name = await promptDialog({ title: "Add a variant slot column", label: "Slot name", help: "Every variant fills it; templates use it as {name}.", validate: slotNameValidator(cols), okLabel: "Add" });
    if (!name) return;
    for (const v of all) {
      v.slots = v.slots || {};
      if (!(name in v.slots)) v.slots[name] = "";
    }
    touch();
    ctx.rerender();
  };
  const renameCol = async (col) => {
    const name = await promptDialog({ title: `Rename slot {${col}}`, label: "New name", value: col, validate: slotNameValidator(cols.filter((c) => c !== col)), okLabel: "Rename" });
    if (!name || name === col) return;
    for (const v of all) if (v.slots && col in v.slots) renameKey(v.slots, col, name);
    touch();
    ctx.rerender();
  };
  const removeCol = async (col) => {
    if (!(await confirmDialog(`Remove slot {${col}}?`, `This deletes {${col}} from every variant.`, "Remove", "danger"))) return;
    for (const v of all) if (v.slots) delete v.slots[col];
    touch();
    ctx.rerender();
  };

  const withinSel = h(
    "select",
    { class: "input", "aria-label": "Nested within" },
    factors().map((f) => h("option", { value: f.key }, f.key)),
  );
  if (!factors().some((f) => f.key === n.within)) withinSel.insertBefore(h("option", { value: n.within || "" }, `${n.within || "—"} (not a factor)`), withinSel.firstChild);
  withinSel.value = n.within || "";
  withinSel.dataset.path = "nested.within";
  withinSel.addEventListener("change", async () => {
    const nu = withinSel.value;
    const has = all.length > 0;
    if (has && !(await confirmDialog("Change the parent factor?", `Variants are registered per level of the parent factor. Switching to “${nu}” starts every level of ${nu} with one empty variant; the current ${all.length} variants are discarded.`, "Switch", "danger"))) {
      withinSel.value = n.within;
      return;
    }
    n.within = nu;
    const variants = {};
    for (const id of levelIds(factorByKey(nu))) variants[id] = [{ id: `${id}_a`, slots: Object.fromEntries(cols.map((c) => [c, ""])) }];
    n.variants = variants;
    touch();
    ctx.rerender();
  });

  const head = h(
    "div",
    { class: "form-row" },
    field("key", "nested.key", { mono: true, help: "Condition key and slot holding the variant id", commit: "change", onChange: () => ctx.rerender() }),
    field("label", "nested.label", { emptyDelete: true, optional: true }),
    h("label", { class: "field" }, h("span", { class: "field-label" }, "within factor"), withinSel, h("span", { class: "field-help" }, "Variants are nested in this factor's levels")),
    field("max per level", "nested.max_per_level", { type: "int", min: 1, help: "Each level registers 1..max variants" }),
  );

  const variantTable = (levelId) => {
    const list = Array.isArray(n.variants[levelId]) ? n.variants[levelId] : null;
    const vp = ["nested", "variants", levelId];
    const rows = (list || []).map((v, vi) =>
      h(
        "tr",
        { dataset: { path: pathString([...vp, vi]) } },
        h("td", { class: "num muted" }, String(vi + 1)),
        h("td", null, control([...vp, vi, "id"], { mono: true, ariaLabel: "Variant id", commit: "change", placeholder: "variant_id" })),
        cols.map((c) => h("td", { class: "slot-cell" }, control([...vp, vi, "slots", c], { type: "textarea", rows: 1, ariaLabel: `${c} for ${v.id}` }))),
        h(
          "td",
          { class: "row-actions" },
          iconButton("↑", "Move up", () => {
            if (vi > 0) {
              list.splice(vi - 1, 0, list.splice(vi, 1)[0]);
              touch();
              ctx.rerender();
            }
          }, { disabled: vi === 0 }),
          iconButton("↓", "Move down", () => {
            if (vi < list.length - 1) {
              list.splice(vi + 1, 0, list.splice(vi, 1)[0]);
              touch();
              ctx.rerender();
            }
          }, { disabled: vi === list.length - 1 }),
          iconButton("✕", "Remove variant", () => {
            list.splice(vi, 1);
            touch();
            ctx.rerender();
          }, { class: "danger" }),
        ),
      ),
    );
    const count = list ? list.length : 0;
    const known = parentIds.includes(levelId);
    return h(
      "div",
      { class: ["variant-group", known ? null : "unknown"], dataset: { path: pathString(vp) } },
      h(
        "div",
        { class: "variant-group-head" },
        h("strong", { class: "mono" }, levelId),
        badge(`${count} / ${maxPer || "?"}`, count === 0 || (maxPer && count > maxPer) ? "error" : "neutral", "variants registered / max per level"),
        known ? null : badge("not a level of " + (n.within || "?"), "warn"),
        known
          ? button("+ variant", () => {
              if (!Array.isArray(n.variants[levelId])) n.variants[levelId] = [];
              const taken = all.map((v) => v.id);
              n.variants[levelId].push({ id: uniqueName(`${levelId}_${String.fromCharCode(97 + count)}`, taken), slots: Object.fromEntries(cols.map((c) => [c, ""])) });
              touch();
              ctx.rerender();
            }, { small: true, disabled: maxPer > 0 && count >= maxPer, title: maxPer > 0 && count >= maxPer ? "At max per level: raise it first" : "Add a variant to this level" })
          : button("Remove these", () => {
              delete n.variants[levelId];
              touch();
              ctx.rerender();
            }, { small: true, kind: "danger" }),
      ),
      count
        ? h(
            "div",
            { class: "table-wrap" },
            h(
              "table",
              { class: "table edit-table" },
              h("thead", null, h("tr", null, h("th", { class: "num" }, "#"), h("th", null, "id"), cols.map((c) => h("th", { class: "slot-col" }, h("span", { class: "mono" }, `{${c}}`))), h("th"))),
              h("tbody", null, rows),
            ),
          )
        : h("p", { class: "muted small" }, "No variants: every level needs at least one."),
    );
  };

  const groups = [...parentIds.filter((x) => x), ...Object.keys(n.variants).filter((k) => !parentIds.includes(k))];

  return card(
    "Nested persona variants",
    {
      path: "nested",
      help: `Personas nested in ${n.within || "a factor"}: {${n.key || "key"}} is the variant id and each slot column is a {slot}. Rows of a cell are split evenly across its level's variants (n per cell must divide).`,
      actions: [
        h("span", { class: "muted small" }, "Slot columns:"),
        cols.map((c) =>
          h("span", { class: "col-chip" }, h("span", { class: "mono" }, `{${c}}`), iconButton("✎", `Rename ${c}`, () => renameCol(c), { class: "tiny" }), iconButton("✕", `Remove ${c}`, () => removeCol(c), { class: "tiny danger" })),
        ),
        button("+ slot column", addCol, { small: true }),
        button("Turn off", async () => {
          if (!(await confirmDialog("Turn off nested variants?", "The study goes back to one implicit persona per cell. The variants are kept in this page until you reload, so turning it back on restores them.", "Turn off"))) return;
          stashedNested = clone(n);
          spec.nested = null;
          touch();
          ctx.rerender();
        }, { small: true }),
      ],
    },
    head,
    h("div", { class: "variant-groups" }, groups.map(variantTable)),
  );
}

// ---- lookup tables ---------------------------------------------------------------------------------------------------

function restructure(values, oldBy, newBy) {
  const out = {};
  for (const combo of combos(newBy)) {
    const dict = Object.fromEntries(newBy.map((k, i) => [k, combo[i]]));
    const oldPath = oldBy.map((k) => (k in dict ? dict[k] : levelIds(factorByKey(k))[0]));
    const v = oldBy.length ? getPath(values, oldPath) : values;
    if (Array.isArray(v)) {
      if (combo.length) setPath(out, combo, clone(v));
      else return clone(v);
    }
  }
  return out;
}

function tableCard(ctx, t, ti) {
  const spec = state.spec;
  const base = ["tables", ti];
  const by = Array.isArray(t.by) ? t.by : (t.by = []);
  const setBy = (nu) => {
    t.values = restructure(t.values || {}, by, nu);
    t.by = nu;
    touch();
    ctx.rerender();
  };
  const avail = factors().map((f) => f.key).filter((k) => !by.includes(k));
  const addSel = h("select", { class: "input", "aria-label": "Add a factor to by" }, h("option", { value: "" }, "+ factor"), avail.map((k) => h("option", { value: k }, k)));
  addSel.addEventListener("change", () => {
    if (addSel.value) setBy([...by, addSel.value]);
  });
  const byChips = h(
    "div",
    { class: "by-chips", dataset: { path: pathString([...base, "by"]) } },
    by.map((k, i) =>
      h(
        "span",
        { class: "col-chip" },
        i > 0 ? iconButton("←", "Move left", () => setBy([...by.slice(0, i - 1), k, by[i - 1], ...by.slice(i + 1)]), { class: "tiny" }) : null,
        h("span", { class: "mono" }, k),
        iconButton("✕", `Stop keying by ${k}`, () => setBy(by.filter((x) => x !== k)), { class: "tiny danger" }),
      ),
    ),
    avail.length ? addSel : null,
  );
  const joinCtl = control([...base, "join"], { mono: true, ariaLabel: "Join string", class: "join-input" });
  const joinShow = h("code", { class: "mono muted small" }, JSON.stringify(t.join ?? ""));
  joinCtl.addEventListener("input", () => (joinShow.textContent = JSON.stringify(joinCtl.value)));

  const cs = combos(by);
  const cells = cs.map((combo) => {
    const p = [...base, "values", ...combo];
    const cur = by.length ? getPath(spec, p) : t.values;
    const items = Array.isArray(cur) ? cur : [];
    const count = h("span", { class: "muted small" }, `${items.length} item${items.length === 1 ? "" : "s"}`);
    const ta = h("textarea", { class: "input textarea", rows: Math.max(2, items.length), spellcheck: "true", "aria-label": `${t.name} for ${combo.join(" · ")}` });
    ta.value = items.join("\n");
    ta.dataset.path = pathString(p);
    autosize(ta);
    ta.addEventListener("input", () => {
      const lines = ta.value.split("\n").map((x) => x.trim()).filter((x) => x !== "");
      if (by.length) setPath(spec, p, lines);
      else t.values = lines;
      count.textContent = `${lines.length} item${lines.length === 1 ? "" : "s"}`;
      touch();
    });
    return h("div", { class: "combo" }, h("div", { class: "combo-head" }, h("span", { class: "mono" }, combo.join(" · ") || "(all)"), count), ta);
  });

  return h(
    "section",
    { class: "card table-card", dataset: { path: pathString(base) } },
    h(
      "div",
      { class: "card-head factor-head" },
      h("label", { class: "inline-field" }, h("span", null, "name"), control([...base, "name"], { mono: true, commit: "change", ariaLabel: "Table name", onChange: () => ctx.rerender() })),
      h("label", { class: "inline-field" }, h("span", null, "join"), joinCtl, joinShow),
      h("label", { class: "inline-field" }, h("span", null, "item slot"), control([...base, "item_slot"], { mono: true, nullable: true, placeholder: "e.g. anchor_{i}", ariaLabel: "Item slot" })),
      h(
        "div",
        { class: "card-actions" },
        iconButton("✕", "Remove table", async () => {
          if (!(await confirmDialog("Remove table?", `Remove lookup table “${t.name}” and all its text?`, "Remove", "danger"))) return;
          spec.tables.splice(ti, 1);
          touch();
          ctx.rerender();
        }, { class: "danger" }),
      ),
    ),
    h("div", { class: "form-row" }, h("span", { class: "field-label" }, "keyed by"), byChips),
    h(
      "p",
      { class: "card-help" },
      `{${t.name || "name"}} is the list joined with the join string`,
      t.item_slot ? `; {${String(t.item_slot).replace("{i}", "1")}}, {${String(t.item_slot).replace("{i}", "2")}} … are the items one by one` : "",
      ". One item per line.",
    ),
    cs.length ? h("div", { class: "combos" }, cells) : h("p", { class: "muted" }, "Pick at least one factor with levels to key this table by."),
  );
}

// ---- view ------------------------------------------------------------------------------------------------------------

export function render(root, ctx) {
  const spec = state.spec;
  const fs = factors();
  const tables = Array.isArray(spec.tables) ? spec.tables : [];
  mount(
    root,
    viewHeader("Axes", "The experimental conditions: crossed factors (outermost first), personas nested in one factor, and stance text looked up by levels. Every level's slots fill the persona templates."),
    issuePanel(ctx, ["factors", "nested", "tables"]),
    h("h2", { class: "section-title" }, "Factors", h("span", { class: "muted small" }, ` ${fs.length} crossed`)),
    fs.map((f, fi) => factorCard(ctx, f, fi)),
    h(
      "div",
      { class: "add-row" },
      button("+ Add factor", () => {
        const key = uniqueName("factor", fs.map((f) => f.key));
        fs.push({ key, label: "", levels: [{ id: "level_1", slots: {} }, { id: "level_2", slots: {} }] });
        touch();
        ctx.rerender();
      }),
    ),
    nestedSection(ctx),
    h("h2", { class: "section-title" }, "Lookup tables", h("span", { class: "muted small" }, ` ${tables.length}`)),
    tables.length ? null : h("p", { class: "muted" }, "None. A lookup table holds text for each combination of some factors' levels (e.g. stance anchors by ideology × topic)."),
    tables.map((t, ti) => tableCard(ctx, t, ti)),
    h(
      "div",
      { class: "add-row" },
      button("+ Add lookup table", () => {
        if (!Array.isArray(spec.tables)) spec.tables = [];
        const name = uniqueName("lookup", spec.tables.map((t) => t.name));
        const first = fs[0] ? [fs[0].key] : [];
        spec.tables.push({ name, by: first, join: " ", item_slot: null, values: {} });
        touch();
        ctx.rerender();
      }),
    ),
  );
}
