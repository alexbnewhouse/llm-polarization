// Pieces shared by the views: headers, cards, the per-view issue panel, slot chips.

import { h, mount, badge, insertAtCursor } from "../ui.js";
import { state, issues } from "../state.js";
import { gotoIssue, viewForPath, VIEW_LABELS } from "../nav.js";

export function viewHeader(title, subtitle, ...actions) {
  return h(
    "header",
    { class: "view-header" },
    h("div", null, h("h1", null, title), subtitle ? h("p", { class: "view-sub" }, subtitle) : null),
    actions.length ? h("div", { class: "view-actions" }, actions) : null,
  );
}

export function card(title, ...children) {
  let opts = {};
  if (children.length && children[0] && children[0].constructor === Object && !(children[0] instanceof Node)) opts = children.shift();
  return h(
    "section",
    { class: ["card", opts.class], dataset: opts.path ? { path: opts.path } : undefined, id: opts.id },
    title || opts.actions
      ? h("div", { class: "card-head" }, title ? h("h2", null, title) : h("span"), opts.actions ? h("div", { class: "card-actions" }, opts.actions) : null)
      : null,
    opts.help ? h("p", { class: "card-help" }, opts.help) : null,
    children,
  );
}

export function issueRow(i, showView = true) {
  const view = viewForPath(i.path);
  return h(
    "li",
    null,
    h(
      "button",
      { type: "button", class: `issue issue-${i.level === "error" ? "error" : "warning"}`, onclick: () => gotoIssue(i.path), title: `Go to ${VIEW_LABELS[view]}` },
      badge(i.level === "error" ? "error" : "warning", i.level === "error" ? "error" : "warn"),
      i.path ? h("code", { class: "issue-path" }, i.path) : null,
      h("span", { class: "issue-msg" }, i.message || ""),
      showView ? h("span", { class: "issue-view muted" }, `${VIEW_LABELS[view]} →`) : null,
    ),
  );
}

export function sortIssues(list) {
  return [...list].sort((a, b) => (a.level === b.level ? 0 : a.level === "error" ? -1 : 1));
}

// A compact list of the issues whose path starts with one of `prefixes`, kept current on validation.
export function issuePanel(ctx, prefixes) {
  const host = h("div", { class: "issue-panel" });
  const draw = () => {
    const mine = sortIssues(
      issues().filter((i) => prefixes.some((p) => i.path === p || String(i.path || "").startsWith(p + ".") || String(i.path || "").startsWith(p + "["))),
    );
    if (!mine.length) {
      mount(host);
      host.hidden = true;
      return;
    }
    host.hidden = false;
    const shown = mine.slice(0, 12);
    mount(
      host,
      h("ul", { class: "issue-list compact" }, shown.map((i) => issueRow(i, false))),
      mine.length > shown.length ? h("p", { class: "muted small" }, `… ${mine.length - shown.length} more on the Study view.`) : null,
    );
  };
  ctx.on("validated", draw);
  draw();
  return host;
}

const SOURCE_HELP = {
  factor: "the level id of factor",
  level: "a slot of the chosen level of factor",
  nested: "the variant id of",
  variant: "a slot of the chosen variant of",
  table: "a lookup table:",
  derived: "a derived slot",
};

export function slotChip(slot, onInsert) {
  const src = String(slot.source || "");
  const [kind, ref] = src.split(":");
  return h(
    "button",
    {
      type: "button",
      class: `chip chip-${kind || "other"}`,
      title: `${SOURCE_HELP[kind] || src} ${ref || ""}`.trim() + ` — click to insert {${slot.name}}`,
      onmousedown: (e) => e.preventDefault(), // keep the textarea's cursor
      onclick: () => onInsert(`{${slot.name}}`),
    },
    `{${slot.name}}`,
  );
}

// Chips for the slots of `which` ("treated" | "control"), inserting into the target the getter returns.
export function slotChips(ctx, which, getTarget) {
  const host = h("div", { class: "chips", "aria-label": `Slots available to ${which} templates` });
  const draw = () => {
    const slots = state.validation && state.validation.slots ? state.validation.slots[which] || [] : null;
    if (!slots) {
      mount(host, h("span", { class: "muted small" }, "Slots appear once the study validates."));
      return;
    }
    if (!slots.length) {
      mount(host, h("span", { class: "muted small" }, "No slots."));
      return;
    }
    mount(host, h("span", { class: "chips-label muted small" }, "Insert:"), slots.map((s) => slotChip(s, (text) => insertAtCursor(getTarget(), text))));
  };
  ctx.on("validated", draw);
  draw();
  return host;
}

export function levelIds(factor) {
  return ((factor && factor.levels) || []).map((l) => (l ? l.id : undefined));
}

export function factorByKey(key) {
  return ((state.spec && state.spec.factors) || []).find((f) => f && f.key === key) || null;
}

export function combos(keys) {
  let out = [[]];
  for (const k of keys) {
    const f = factorByKey(k);
    const ids = levelIds(f).filter((x) => x !== undefined && x !== null && x !== "");
    const next = [];
    for (const c of out) for (const id of ids) next.push([...c, id]);
    out = next;
  }
  return out;
}
