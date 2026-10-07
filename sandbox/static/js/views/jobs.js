// Jobs view: every harness subprocess the sandbox started (check, run, score, flags, agreement, survey), its
// status and exit meaning, and a live log viewer with stop.

import { api } from "../api.js";
import { h, mount, button, badge, statusChip, fmtTime, toastError, poll, emptyState, pagedRows } from "../ui.js";
import { viewHeader, card } from "./common.js";
import { jobConsole } from "../jobconsole.js";

let selected = null;

export function render(root, ctx) {
  const listHost = h("div");
  const consoleHost = h("div");
  if (ctx.params[0]) selected = ctx.params[0];
  mount(
    root,
    viewHeader("Jobs", "Subprocesses started from the sandbox: python -m harness.run <kind> …, cwd the repo root. Logs live in workspace/jobs/."),
    card(null, listHost),
    consoleHost,
  );
  let jobs = [];
  let consoleFor = null;
  let first = true;

  const drawConsole = () => {
    const job = jobs.find((j) => j.id === selected);
    if (!job) {
      consoleFor = null;
      mount(consoleHost, jobs.length ? h("p", { class: "muted" }, "Pick a job to see its log.") : null);
      return;
    }
    if (consoleFor === job.id) return;
    consoleFor = job.id;
    mount(consoleHost, card(`Log of ${job.id}`, jobConsole(job, { alive: ctx.alive })));
  };

  const drawList = () => {
    if (!jobs.length) {
      mount(listHost, emptyState("No jobs yet. Pre-flight checks and runs started from the Run view, and actions from a run, appear here."));
      return;
    }
    const tbody = h("tbody");
    const labels = jobs.some((j) => j.label);
    pagedRows(
      tbody,
      jobs,
      (j) =>
        h(
          "tr",
          { class: ["clickable", j.id === selected ? "selected" : null], onclick: (e) => {
            if (e.target.closest("button, a")) return;
            selected = j.id;
            drawList();
            drawConsole();
          } },
          h("td", { class: "mono small" }, j.id),
          h("td", null, badge(j.kind || "?", "neutral")),
          labels ? h("td", { class: "small" }, j.label || "") : null,
          h("td", null, j.run_id ? h("a", { class: "mono link", href: `#/runs/${encodeURIComponent(j.run_id)}` }, j.run_id) : h("span", { class: "muted" }, "—")),
          h("td", null, statusChip(j.status)),
          h("td", { class: "small" }, j.meaning || "", j.returncode !== null && j.returncode !== undefined ? h("span", { class: "muted mono" }, ` (${j.returncode})`) : null),
          h("td", { class: "small muted" }, fmtTime(j.started_at)),
          h("td", { class: "small muted" }, fmtTime(j.ended_at)),
          h(
            "td",
            { class: "row-actions" },
            j.status === "running"
              ? button(j.stop_requested ? "Force stop" : "Stop", async () => {
                  try {
                    await api.stopJob(j.id);
                    refresh();
                  } catch (err) {
                    toastError(err, "Stop failed");
                  }
                }, { small: true, kind: "danger", title: j.stop_requested ? "Force stop sends SIGKILL" : "SIGINT: in-flight dyads finish, queued ones never start" })
              : null,
          ),
        ),
      200,
      9,
    );
    mount(
      listHost,
      h(
        "div",
        { class: "table-wrap tall" },
        h(
          "table",
          { class: "table sticky" },
          h("thead", null, h("tr", null, h("th", null, "id"), h("th", null, "kind"), labels ? h("th", null, "label") : null, h("th", null, "run"), h("th", null, "status"), h("th", null, "meaning"), h("th", null, "started"), h("th", null, "ended"), h("th"))),
          tbody,
        ),
      ),
    );
  };

  const refresh = async () => {
    const res = await api.jobs();
    if (!ctx.alive()) return false;
    const next = [...((res && res.jobs) || [])].sort((a, b) => String(b.started_at || "").localeCompare(String(a.started_at || "")));
    const key = (list) => JSON.stringify(list.map((j) => [j.id, j.status, j.returncode, !!j.stop_requested]));
    const changed = key(next) !== key(jobs);
    jobs = next;
    if (!selected && jobs.length) selected = jobs[0].id;
    if (changed || first) drawList();
    first = false;
    drawConsole();
    return true;
  };

  mount(listHost, h("p", { class: "muted" }, "Loading…"));
  poll(refresh, 2000, ctx.alive, (err) => toastError(err, "Could not list jobs"));
}
