// State shared by the run views (list, run detail, dyad detail): which data dir, the loaded run summary, and
// the live-refresh loop that tails the run's row files by byte offset while the run is live, and the MOCK marks.

import { api } from "../api.js";
import { badge, poll, storageGet, storageSet, toastError } from "../ui.js";
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

// ---- liveness ----------------------------------------------------------------------------------------------------

// No new row for this long, and no run job of this sandbox writing it: a run whose process is gone (a killed
// run leaves its in-flight dyads at "started" forever, so status_counts alone would keep it "running").
export const STALL_MS = 20 * 60 * 1000;

// A harness timestamp ("2026-10-07T12:34:56+0200") as epoch ms, or 0.
export function parseTs(ts) {
  if (!ts) return 0;
  const t = Date.parse(String(ts).replace(/([+-]\d\d)(\d\d)$/, "$1:$2"));
  return Number.isFinite(t) ? t : 0;
}

// The newest row a summary has seen (a dyad's last status or turn row, else the run's start), in epoch ms.
export function summaryActivity(s) {
  let last = parseTs(s && s.manifest && s.manifest.started_at);
  for (const d of (s && s.dyads) || []) last = Math.max(last, parseTs(d.last_ts));
  return last;
}

const normDir = (d) => String(d || "").replace(/\\/g, "/").replace(/^\.\//, "").replace(/\/+$/, "");

function sameDataDir(a, b) {
  const x = normDir(a);
  const y = normDir(b);
  return !x || !y || x === y || x.endsWith(`/${y}`) || y.endsWith(`/${x}`);
}

// Whether a `run` job of this sandbox (or a detached one an earlier sandbox started) is writing this run.
export async function runJobRunning(runId, dataDir = currentDataDir()) {
  try {
    const res = await api.jobs();
    return ((res && res.jobs) || []).some((j) => j.kind === "run" && j.run_id === runId && j.status === "running" && sameDataDir(j.data_dir, dataDir));
  } catch {
    return false;
  }
}

// Is the run still being written? Live while a dyad runs, while fewer dyads have finished than were planned,
// or while a run job is writing it; but a run with neither a job nor a new row for STALL_MS is treated as
// stopped. `lastSeen` is the newest row time the caller has tailed since the summary.
// -> {live, reason: "job" | "running" | "pending" | "finished" | "stalled", since?}
export async function runLiveness(runId, s, lastSeen = 0) {
  if (await runJobRunning(runId)) return { live: true, reason: "job" };
  const sc = (s && s.status_counts) || {};
  const running = Number(sc.running) || 0;
  const finished = (Number(sc.complete) || 0) + (Number(sc.failed) || 0);
  const planned = s && s.planned !== null && s.planned !== undefined ? Number(s.planned) : null;
  if (!(running > 0 || (planned !== null && Number.isFinite(planned) && finished < planned))) return { live: false, reason: "finished" };
  const last = Math.max(Number(lastSeen) || 0, summaryActivity(s));
  if (!last || Date.now() - last > STALL_MS) return { live: false, reason: "stalled", since: last || null };
  return { live: true, reason: running > 0 ? "running" : "pending" };
}

// ---- the live loop -------------------------------------------------------------------------------------------------

// While the run is live: every `interval` ms (when `isOn()`), tail the run from the summary's byte offsets and
// hand new rows to onRows, call onTick, and every 10 s (5 s with no offsets to tail from) refetch the summary
// (unless `summary` is false) and re-check `liveness`. Once the run is not live the loop stops (nothing is
// fetched any more) and onState reports it paused; resume() and refresh() start it again. With `isOn()` false
// (auto-refresh off, another tab) the loop idles without fetching.
// Returns {stop, pause(reason), resume(), refresh(), paused, reason}.
export function startLive(ctx, runId, { onRows, onSummary, onTick, isOn, liveness, onState, summary = true, interval = 2000 }) {
  let offsets = rs.summary && rs.summaryRunId === runId && rs.summary.offsets ? { ...rs.summary.offsets } : null;
  let lastCheck = Date.now();
  let lastSeen = 0;
  let stopPoll = null;
  const check = liveness || ((s) => runLiveness(runId, s || (rs.summaryRunId === runId ? rs.summary : null), lastSeen));
  const ctl = { paused: false, reason: null, stopped: false };
  Object.defineProperty(ctl, "running", { get: () => !!stopPoll && !ctl.paused });
  const report = () => {
    if (onState && ctx.alive()) onState(ctl);
  };
  const halt = () => {
    if (stopPoll) stopPoll();
    stopPoll = null;
  };
  ctl.stop = () => {
    ctl.stopped = true;
    halt();
  };
  ctl.pause = (reason) => {
    halt();
    ctl.paused = true;
    ctl.reason = reason || "finished";
    report();
  };
  const tick = async () => {
    if (!isOn()) return true;
    if (offsets) {
      const res = await api.tail(runId, currentDataDir(), offsets);
      if (!ctx.alive()) return false;
      if (res && res.offsets) offsets = { ...offsets, ...res.offsets };
      if (res && res.rows) {
        for (const rows of Object.values(res.rows)) for (const r of rows || []) lastSeen = Math.max(lastSeen, parseTs(r.ts));
        if (onRows) onRows(res.rows);
      }
    }
    if (onTick) {
      await onTick();
      if (!ctx.alive() || ctl.paused || ctl.stopped) return false;
    }
    if (Date.now() - lastCheck >= (offsets ? 10000 : 5000)) {
      lastCheck = Date.now();
      let s = null;
      if (summary) {
        s = await loadSummary(runId);
        if (!ctx.alive()) return false;
        if (!offsets && s.offsets) offsets = { ...s.offsets };
        if (onSummary) onSummary(s);
      }
      const lv = await check(s);
      if (!ctx.alive() || ctl.stopped) return false;
      if (!lv.live) {
        stopPoll = null; // returning false ends this poll
        ctl.paused = true;
        ctl.reason = lv.reason;
        ctl.since = lv.since || null;
        report();
        return false;
      }
    }
    return true;
  };
  ctl.resume = () => {
    if (ctl.stopped || !ctx.alive()) return;
    const changed = ctl.paused || !stopPoll;
    ctl.paused = false;
    ctl.reason = null;
    lastCheck = Date.now();
    if (!stopPoll) stopPoll = poll(tick, interval, ctx.alive, (err) => toastError(err, "Live refresh"));
    if (changed) report();
  };
  // One fetch of the run summary (when `summary`), then resume the loop if the run is live, else stay paused.
  ctl.refresh = async () => {
    let s = null;
    if (summary) {
      s = await loadSummary(runId);
      if (!ctx.alive()) return null;
      if (s.offsets) offsets = { ...s.offsets };
      lastSeen = 0;
      if (onSummary) onSummary(s);
    }
    const lv = await check(s);
    if (!ctx.alive() || ctl.stopped) return lv;
    if (lv.live) ctl.resume();
    else {
      ctl.since = lv.since || null;
      ctl.pause(lv.reason);
    }
    return lv;
  };
  // Start only if the run is live now; a finished run is never polled.
  (async () => {
    let lv;
    try {
      lv = await check(null);
    } catch {
      lv = { live: true };
    }
    if (!ctx.alive() || ctl.stopped || ctl.paused) return;
    if (lv.live) ctl.resume();
    else {
      ctl.since = lv.since || null;
      ctl.pause(lv.reason);
    }
  })();
  return ctl;
}

export function pausedText(ctl, what = "run") {
  if (!ctl || !ctl.paused) return null;
  if (ctl.reason === "stalled") {
    const mins = ctl.since ? Math.round((Date.now() - ctl.since) / 60000) : null;
    return `no new rows${mins !== null ? ` for ${mins} min` : ""} and no run job — auto-refresh paused`;
  }
  return `${what} finished — auto-refresh paused`;
}

// ---- mock runs -------------------------------------------------------------------------------------------------------

// A role (run summary roles.*, or a run list's seeker/mentor) served by the sandbox's demo mock backend.
export function isMockRole(r) {
  if (!r) return false;
  if (r.build_info === "sandbox-mock") return true;
  if (typeof r.alias === "string" && r.alias.startsWith("sandbox-mock")) return true;
  const base = r.model_path ? String(r.model_path).split(/[\\/]/).pop() : "";
  return base.startsWith("sandbox-mock-");
}

export function mockBadge(roles) {
  return badge("MOCK", "mock", `${roles && roles.length ? `${roles.join(" and ")}: ` : ""}the sandbox's demo mock servers (build_info sandbox-mock). Synthetic text, never for data.`);
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
