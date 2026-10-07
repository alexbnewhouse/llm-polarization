// State shared by the run views (list, run detail, dyad detail): which data dir, the loaded run summary, and
// the live-refresh loop that tails the run's row files by byte offset.

import { api } from "../api.js";
import { poll, storageGet, storageSet, toastError } from "../ui.js";
import { state } from "../state.js";

export const rs = {
  dataDir: null,
  summary: null,
  summaryRunId: null,
  auto: storageGet("runsAuto", true),
};

export function currentDataDir() {
  if (!rs.dataDir) {
    rs.dataDir =
      storageGet("runsDataDir", null) ||
      (state.meta && Array.isArray(state.meta.data_dirs) && state.meta.data_dirs[0]) ||
      (state.spec && state.spec.run && state.spec.run.data_dir) ||
      "data";
  }
  return rs.dataDir;
}

export function setDataDir(d) {
  const v = String(d || "").trim() || "data";
  if (v !== rs.dataDir) {
    rs.summary = null;
    rs.summaryRunId = null;
  }
  rs.dataDir = v;
  storageSet("runsDataDir", v);
}

export function setAuto(on) {
  rs.auto = !!on;
  storageSet("runsAuto", rs.auto);
}

export async function loadSummary(runId) {
  const s = await api.run(runId, currentDataDir());
  rs.summary = s;
  rs.summaryRunId = runId;
  return s;
}

export function isLive(summary) {
  const sc = (summary && summary.status_counts) || {};
  return (Number(sc.running) || 0) > 0;
}

// While `isOn()`, tail the run every 2 s from the summary's byte offsets (when the server reports them) and
// hand new rows to onRows; refetch the summary every 10 s (every 5 s when there are no offsets to tail from).
export function startLive(ctx, runId, { onRows, onSummary, isOn }) {
  let offsets = rs.summary && rs.summary.offsets ? { ...rs.summary.offsets } : null;
  let lastSummary = Date.now();
  return poll(
    async () => {
      if (!isOn()) return true;
      if (offsets) {
        const res = await api.tail(runId, currentDataDir(), offsets);
        if (!ctx.alive()) return false;
        if (res && res.offsets) offsets = { ...offsets, ...res.offsets };
        if (res && res.rows && onRows) onRows(res.rows);
      }
      if (Date.now() - lastSummary >= (offsets ? 10000 : 5000)) {
        lastSummary = Date.now();
        const s = await loadSummary(runId);
        if (!ctx.alive()) return false;
        if (!offsets && s.offsets) offsets = { ...s.offsets };
        if (onSummary) onSummary(s);
      }
      return true;
    },
    2000,
    ctx.alive,
    (err) => toastError(err, "Live refresh"),
  );
}

export function levelLabel(v) {
  return v === null || v === undefined ? "(none)" : String(v);
}

export function runHref(runId, extra = "") {
  const dd = currentDataDir();
  return `#/runs/${encodeURIComponent(runId)}${extra}${dd ? `?data_dir=${encodeURIComponent(dd)}` : ""}`;
}

export function dyadHref(runId, dyadId, attempt) {
  const dd = currentDataDir();
  const q = [dd ? `data_dir=${encodeURIComponent(dd)}` : null, attempt ? `attempt=${attempt}` : null].filter(Boolean).join("&");
  return `#/runs/${encodeURIComponent(runId)}/dyads/${encodeURIComponent(dyadId)}${q ? `?${q}` : ""}`;
}
