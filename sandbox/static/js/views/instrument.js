// Instrument view: the mentor's survey items (id, battery, scale, text) in administration order.

import { h, mount, button, iconButton, badge, kv } from "../ui.js";
import { state, touch } from "../state.js";
import { control, field, uniqueName } from "../forms.js";
import { viewHeader, card, issuePanel } from "./common.js";

export function render(root, ctx) {
  const spec = state.spec;
  if (!spec.instrument || typeof spec.instrument !== "object") {
    mount(
      root,
      viewHeader("Instrument", "The mentor's pre/post survey."),
      issuePanel(ctx, ["instrument"]),
      card("No instrument", button("Create an empty instrument", () => {
        spec.instrument = { version: "0.1", items: [] };
        touch();
        ctx.rerender();
      }, { kind: "primary" })),
    );
    return;
  }
  const inst = spec.instrument;
  const items = Array.isArray(inst.items) ? inst.items : [];
  const batteries = [...new Set(items.map((i) => i && i.battery).filter(Boolean))];
  const listId = "battery-names";
  const extras = Object.entries(inst).filter(([k]) => k !== "version" && k !== "items");

  const row = (it, i) => {
    const p = ["instrument", "items", i];
    const bat = control([...p, "battery"], { mono: true, ariaLabel: "Battery" });
    bat.setAttribute("list", listId);
    return h(
      "tr",
      { dataset: { path: `instrument.items[${i}]` } },
      h("td", { class: "num muted" }, String(i + 1)),
      h("td", null, control([...p, "id"], { mono: true, ariaLabel: "Item id" })),
      h("td", null, bat),
      h("td", null, control([...p, "scale", "min"], { type: "int", ariaLabel: "Scale min", class: "input-xs" })),
      h("td", null, control([...p, "scale", "max"], { type: "int", ariaLabel: "Scale max", class: "input-xs" })),
      h("td", { class: "text-cell" }, control([...p, "text"], { type: "textarea", rows: 2, ariaLabel: "Item text" })),
      h(
        "td",
        { class: "row-actions" },
        iconButton("↑", "Earlier", () => {
          if (i > 0) {
            items.splice(i - 1, 0, items.splice(i, 1)[0]);
            touch();
            ctx.rerender();
          }
        }, { disabled: i === 0 }),
        iconButton("↓", "Later", () => {
          if (i < items.length - 1) {
            items.splice(i + 1, 0, items.splice(i, 1)[0]);
            touch();
            ctx.rerender();
          }
        }, { disabled: i === items.length - 1 }),
        iconButton("✕", "Remove item", () => {
          items.splice(i, 1);
          touch();
          ctx.rerender();
        }, { class: "danger" }),
      ),
    );
  };

  const counts = {};
  for (const it of items) if (it && it.battery) counts[it.battery] = (counts[it.battery] || 0) + 1;

  mount(
    root,
    viewHeader("Instrument", "The mentor's survey: asked before turn 1 in a fresh context, and after the last turn branching off the mentor's own view of the dialogue. Replies are schema-constrained to {\"answer\": <int>}."),
    issuePanel(ctx, ["instrument"]),
    card(
      null,
      h(
        "div",
        { class: "form-row" },
        field("version", "instrument.version", { mono: true, help: "Bump when wording changes; the batteries file's sha256 goes onto every survey row." }),
        h("div", { class: "field" }, h("span", { class: "field-label" }, "Batteries"), h("div", { class: "chips" }, Object.entries(counts).map(([b, n]) => badge(`${b} · ${n}`, "neutral")))),
      ),
      extras.length
        ? h("div", { class: "muted small" }, h("span", null, "Other keys kept as they are: "), kv(extras.map(([k, v]) => [k, h("span", { class: "mono small" }, typeof v === "string" ? v : JSON.stringify(v))]), "kv-inline"))
        : null,
    ),
    card(
      `Items (${items.length})`,
      { path: "instrument.items", help: "Order is administration order, identical pre and post. Each item is one question with an integer answer on [min, max]; analysis normalises shifts by (max − min) so scales compare." },
      h("datalist", { id: listId }, batteries.map((b) => h("option", { value: b }))),
      h(
        "div",
        { class: "table-wrap" },
        h(
          "table",
          { class: "table edit-table items-table" },
          h("thead", null, h("tr", null, h("th", { class: "num" }, "#"), h("th", null, "id"), h("th", null, "battery"), h("th", null, "min"), h("th", null, "max"), h("th", null, "text"), h("th"))),
          h("tbody", null, items.map(row)),
        ),
      ),
      h(
        "div",
        { class: "card-foot" },
        button("+ Add item", () => {
          if (!Array.isArray(inst.items)) inst.items = [];
          const last = inst.items[inst.items.length - 1];
          inst.items.push({
            id: uniqueName("item", inst.items.map((x) => x && x.id)),
            battery: last ? last.battery : "agreement",
            scale: last && last.scale ? { ...last.scale } : { min: 1, max: 5 },
            text: "",
          });
          touch();
          ctx.rerender();
        }, { small: true }),
      ),
    ),
  );
}
