// A live console for one job (a `python -m harness.run ...` subprocess): status, the argv as a copyable
// shell line, and the log, polled from GET /api/jobs/<id>/log?offset= every second while it runs.

import { api } from "./api.js";
import { h, mount, badge, button, statusChip, commandLine, fmtTime, poll, toastError } from "./ui.js";

const MAX_LOG_CHARS = 1_500_000;

// The run a job writes lives in its config's data_dir (job.data_dir, recorded by the server); a caller that
// knows better (the Run view knows the study's) passes dataDir.
function runLink(runId, dataDir) {
  const q = dataDir ? `?data_dir=${encodeURIComponent(dataDir)}` : "";
  return h("a", { href: `#/runs/${encodeURIComponent(runId)}${q}`, class: "link" }, `Watch run ${runId} →`);
}

export function jobConsole(initialJob, { alive = () => true, onUpdate, showRunLink = true, dataDir } = {}) {
  let job = initialJob;
  let offset = 0;
  const created = Date.now();
  const head = h("div", { class: "job-head" });
  const logText = document.createTextNode("");
  const pre = h("pre", { class: "log", tabindex: 0, "aria-label": "Job output" }, logText);
  const root = h("div", { class: "job-console" }, head, h("div", { class: "job-argv" }), pre);

  const drawHead = () => {
    const running = job.status === "running";
    mount(
      head,
      h(
        "div",
        { class: "job-title" },
        badge(job.kind || "job", "neutral"),
        h("strong", null, job.label || job.id),
        statusChip(job.status),
        job.meaning ? h("span", { class: "muted" }, job.meaning) : null,
        job.returncode !== null && job.returncode !== undefined ? h("span", { class: "muted mono" }, `exit ${job.returncode}`) : null,
      ),
      h(
        "div",
        { class: "job-meta" },
        h("span", { class: "muted small" }, `started ${fmtTime(job.started_at)}`),
        job.ended_at ? h("span", { class: "muted small" }, `ended ${fmtTime(job.ended_at)}`) : null,
        showRunLink && job.run_id
          ? runLink(job.run_id, dataDir || job.data_dir)
          : null,
        running
          ? button(job.stop_requested ? "Force stop" : "Stop", async () => {
              try {
                job = (await api.stopJob(job.id)) || job;
                drawHead();
              } catch (err) {
                toastError(err, "Stop failed");
              }
            }, {
              kind: "danger",
              small: true,
              title: job.stop_requested
                ? "Already asked to stop (in-flight dyads are finishing). Force stop sends SIGKILL: the dyads in flight are cut off."
                : "SIGINT: in-flight dyads finish, queued ones never start",
            })
          : null,
        job.status === "lost" ? h("span", { class: "warn-text small" }, "lost: the server restarted and can no longer follow this process") : null,
      ),
    );
    const argvBox = root.querySelector(".job-argv");
    mount(argvBox, job.argv ? commandLine(job.argv) : null);
    if (onUpdate) onUpdate(job);
  };
  drawHead();

  const appendLog = (text) => {
    if (!text) return;
    const nearBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 40;
    logText.appendData(text);
    if (logText.length > MAX_LOG_CHARS) logText.deleteData(0, logText.length - MAX_LOG_CHARS);
    if (nearBottom) pre.scrollTop = pre.scrollHeight;
  };

  poll(
    async () => {
      const res = await api.jobLog(job.id, offset);
      if (res) {
        appendLog(res.text || "");
        if (typeof res.offset === "number") offset = res.offset;
        if (res.job) {
          const changed = res.job.status !== job.status || res.job.returncode !== job.returncode || res.job.stop_requested !== job.stop_requested;
          job = res.job;
          if (changed) drawHead();
        }
        // Keep draining while the job runs or while the log still has text to give.
        if (job.status !== "running" && !res.text) return false;
      }
      return true;
    },
    1000,
    () => alive() && (root.isConnected || Date.now() - created < 3000),
    (err) => toastError(err, `Log of job ${job.id}`),
  );
  return root;
}
